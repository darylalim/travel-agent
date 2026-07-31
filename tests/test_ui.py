"""Tests for the Streamlit runtime seam.

No network, no model calls, no Streamlit server: everything here operates on
the plain data structures `stream_turn` routes into, which is where the
interesting decisions live.

The case worth understanding is `test_caveat_survives_an_empty_result_set`.
Subagent tool results never reach the parent's message history — deepagents
folds a subagent's state back without its `messages` — so these payloads are
captured mid-stream instead. If that capture ever regresses, the synthetic
data warning disappears silently and the traveler sees invented fares
presented as real.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool

from travel_agent import ui
from travel_agent.subagents import build_subagents
from travel_agent.tools.availability import date_offset
from travel_agent.tools.budget import summarize_budget
from travel_agent.ui import (
    CANCEL_BANDS,
    CAPTURED_TOOLS,
    STOP_BANDS,
    TripRecord,
    _tool_label,
    _unpack,
    cancel_band,
    costing_series,
    currency_code,
    money_format,
    offers,
    search_label,
    source_column,
    stop_band,
    stream_turn,
)

# What `build_agent` binds to the main agent besides web search, named directly
# rather than enumerated. Reading them off a real agent would mean calling
# `build_agent()`, which constructs a model client — the one thing this suite
# never does.
_MAIN_AGENT_TOOLS = frozenset({summarize_budget.name, date_offset.name})

# The scout's toolset, pinned. See the test below for why this is a whole-set
# assertion rather than a naming convention.
_SCOUT_TOOLS = frozenset({"search_flights", "search_stays", "date_offset"})


def _tool_name(tool: Any) -> str:
    """Name of a bound tool, whether it arrived as a `BaseTool` or a callable.

    `SubAgent` types its `tools` as a sequence of any of the three forms
    deepagents accepts, so a bare `.name` is not sound. Every tool in this
    project is `@tool`-decorated and takes the first branch; the fallback keeps
    a plain function readable rather than dropping it from the set, which would
    make the pinned assertion below pass for the wrong reason.
    """
    if isinstance(tool, BaseTool):
        return tool.name
    return getattr(tool, "__name__", repr(tool))


def _subagent_tools() -> dict[str, frozenset[str]]:
    """Tool names bound to each subagent, keyed by subagent name.

    `build_subagents([])` is handed an explicit empty search list rather than
    `None`, which would call `build_search_tools()` and read `TAVILY_API_KEY` —
    ambient configuration this suite is built to ignore. A `SubAgent` is a
    TypedDict, so its tools are a key rather than an attribute.
    """
    return {
        subagent["name"]: frozenset(_tool_name(tool) for tool in subagent.get("tools") or [])
        for subagent in build_subagents([])
    }


def _tool_message(name: str, payload: dict) -> ToolMessage:
    """A tool result shaped the way LangChain delivers one.

    A `@tool` returning a dict arrives with its payload JSON-encoded on
    `content`; this mirrors that exactly.
    """
    return ToolMessage(content=json.dumps(payload), name=name, tool_call_id=f"call-{name}")


# Checkpoint namespaces as LangGraph emits them under `subgraphs=True`: the
# root agent's is empty, and anything running inside a `task` call carries the
# path that got it there. That difference is the only thing separating the
# traveler's answer from subagent chatter.
_ROOT: tuple[str, ...] = ()
_SCOUT = ("task:1f0c9a", "agent:availability-scout")


class _FakeAgent:
    """Stands in for the compiled graph, recording how it was streamed.

    `stream_turn` reaches its agent through `ui.get_agent`, which is wrapped in
    `@st.cache_resource`. Patching the module attribute replaces the lookup
    before the cache is ever consulted, so no agent is built, no model client
    is constructed and no Streamlit runtime is required.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self.stream_kwargs: dict[str, Any] = {}

    def stream(self, _payload: Any, **kwargs: Any) -> Iterator[Any]:
        self.stream_kwargs = kwargs
        yield from self._script


def _run(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
    on_activity: Any = None,
) -> tuple[_FakeAgent, TripRecord, list[str]]:
    """Drive one turn over a scripted stream; return the agent, record and prose."""
    agent = _FakeAgent(script)
    monkeypatch.setattr(ui, "get_agent", lambda *args, **kwargs: agent)
    record = TripRecord()
    prose = list(stream_turn("thread", "5 days in Kyoto", record, on_activity))
    return agent, record, prose


def test_every_captured_tool_name_matches_a_tool_that_can_actually_run():
    """`CAPTURED_TOOLS` names tools by string, so a rename breaks it in silence.

    The literals live in `ui.py`; the `@tool` functions they name live two
    packages away, and nothing but this test connects them. Rename one and
    `record_tool` stops matching it: the Trip page renders with no table, no
    figures, and — the part that matters — no synthetic-data notice, because
    `caveats()` reads payloads that were never stored. Nothing raises, and the
    page looks like a trip nobody has searched for yet.
    """
    reachable = _MAIN_AGENT_TOOLS.union(*_subagent_tools().values())
    assert set(CAPTURED_TOOLS) <= reachable


def test_the_scouts_toolset_is_pinned_so_a_new_search_has_to_decide_about_capture():
    """A fourth search tool must not reach the traveler uncaptured.

    `search_flights` and `search_stays` are bound only to `availability-scout`,
    and every offer they return carries `synthetic` plus a response-level
    `warning`. A search for trains or car hire would reach the traveler through
    the same page, but `record_tool` filters on `CAPTURED_TOOLS` — so its
    results would be dropped before `caveats()` ever saw them, putting
    real-looking figures in the chat with no provenance anywhere on the Trip
    page. That is the exact failure the data-honesty invariant exists to
    prevent, arriving through the one path the invariant does not name.

    Pinned as a whole set rather than checked against a `search_` prefix. The
    prefix is a convention nothing enforces, and a tool named
    `find_rail_passes` would clear it while capturing nothing. This fails on
    *any* change to what the scout can do, which is the intent: the fix is to
    add the tool to `CAPTURED_TOOLS` and give the Trip page somewhere to render
    it, not to edit this set until the test goes green.
    """
    assert _subagent_tools()["availability-scout"] == _SCOUT_TOOLS
    # Both searches are captured; `date_offset` returns a date, not inventory.
    assert _SCOUT_TOOLS - set(CAPTURED_TOOLS) == {"date_offset"}


def test_stream_turn_asks_for_the_nested_messages_it_depends_on(monkeypatch):
    """`subgraphs=True` is load-bearing and deleting it breaks nothing loudly.

    `langgraph.pregel._messages` drops every message whose checkpoint namespace
    is nested unless the flag is set. Since `search_flights` and `search_stays`
    are bound only to `availability-scout`, dropping it means the Trip page
    never sees a single offer — no tables, and no synthetic-data notice, since
    `caveats()` reads payloads that were never captured. The turn still streams
    a perfectly good answer, so nothing anywhere reports a problem.

    `values` earns its place too: it is what `absorb_files` reads, and without
    it the workspace and itinerary silently stop updating.
    """
    agent, _, _ = _run(
        monkeypatch, [(_ROOT, "messages", (AIMessageChunk(content="Hi", id="r1"), {}))]
    )

    assert agent.stream_kwargs["subgraphs"] is True
    assert set(agent.stream_kwargs["stream_mode"]) == {"messages", "values"}
    # The thread id is what scopes `/trip/*`, so a turn addressed to the wrong
    # one would answer against another trip's brief and itinerary.
    assert agent.stream_kwargs["config"]["configurable"]["thread_id"] == "thread"


def test_a_subagents_search_is_captured_while_its_prose_stays_out_of_the_bubble(monkeypatch):
    """The two halves of the fix have to hold at once, so drive them together.

    `subgraphs=True` is what makes subagent messages visible at all — and the
    same flag is why the scout's own commentary now arrives in the stream and
    has to be filtered back out by namespace. Capturing everything would put
    "I checked three airports" in the traveler's answer; filtering everything
    would drop the offers. The existing tests cover `_unpack` and `record_tool`
    separately; only running a turn shows they are wired to each other.
    """
    activity: list[tuple[str, str]] = []

    # `on_activity` is called with two positional arguments, so a bare
    # `activity.append` would not do — it takes one.
    def note(call_id: str, label: str) -> None:
        activity.append((call_id, label))

    _, record, prose = _run(
        monkeypatch,
        [
            # The scout announces its search — activity trail, never the bubble.
            (
                _SCOUT,
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        id="s1",
                        tool_call_chunks=[
                            {
                                "name": "search_flights",
                                "args": "{}",
                                "id": "call-1",
                                "index": 0,
                                "type": "tool_call_chunk",
                            }
                        ],
                    ),
                    {},
                ),
            ),
            # The payload the Trip page cannot get any other way: it never
            # reaches the parent's message history, so this is the only chance.
            (
                _SCOUT,
                "messages",
                (_tool_message("search_flights", {"offers": [], "warning": "Sample offers."}), {}),
            ),
            # Subagent prose. Belongs to the activity panel, not the transcript.
            (
                _SCOUT,
                "messages",
                (AIMessageChunk(content="I checked three airports.", id="s2"), {}),
            ),
            (
                _ROOT,
                "values",
                {"files": {"/trip/brief.md": {"content": "2 people", "encoding": "utf-8"}}},
            ),
            # Thinking blocks stream as content too; `.text` keeps them out.
            (
                _ROOT,
                "messages",
                (AIMessageChunk(content=[{"type": "thinking", "thinking": "hm"}], id="r1"), {}),
            ),
            (_ROOT, "messages", (AIMessageChunk(content="Kyoto for five days.", id="r2"), {})),
            # A subagent emits state of its own, and only the root's is
            # complete. Placed last on purpose: absorbed, it would overwrite
            # the workspace captured above with nothing, so the assertion
            # below is what proves the namespace check is doing work.
            (_SCOUT, "values", {"files": {}}),
        ],
        note,
    )

    # Only the root agent's answer — and only its text, never its reasoning.
    assert prose == ["Kyoto for five days."]
    # The nested result was captured, so the caveat survives an empty search.
    assert record.caveats() == ["Sample offers."]
    assert record.files == {"/trip/brief.md": "2 people"}
    # The scout's search is announced from the tier that issued it, so the
    # traveler watches the search rather than an opaque `task` call.
    assert activity == [("call-1", "Searching flights")]


def test_unpack_handles_namespaced_and_bare_stream_items():
    assert _unpack((("tools:1", "model:2"), "messages", "payload")) == (
        ("tools:1", "model:2"),
        "messages",
        "payload",
    )
    assert _unpack(("values", {"files": {}})) == ((), "values", {"files": {}})


def test_record_keeps_search_results_and_drops_failures():
    record = TripRecord()
    record.record_tool(_tool_message("search_flights", {"offers": [{"total_fare": 500}]}))
    record.record_tool(_tool_message("search_flights", {"error": "bad date"}))
    record.record_tool(_tool_message("write_file", {"path": "/trip/brief.md"}))

    # The failed call is dropped, so the view keeps showing the last good
    # search rather than blanking because the newest attempt was rejected.
    assert len(record.payloads["search_flights"]) == 1
    assert "write_file" not in record.payloads
    assert record.latest("search_flights") == {"offers": [{"total_fare": 500}]}


def test_caveat_survives_an_empty_result_set():
    """A synthetic search that found nothing must still be labelled.

    Providers declare synthetic-ness up front precisely so that zero results
    do not read as "we checked real inventory and found none".
    """
    record = TripRecord()
    record.record_tool(
        _tool_message("search_flights", {"offers": [], "warning": "These are sample offers."})
    )

    assert record.caveats() == ["These are sample offers."]
    assert offers(record.latest("search_flights")) == []


def test_caveats_are_deduped_but_keep_flight_and_lodging_separate():
    record = TripRecord()
    record.record_tool(_tool_message("search_flights", {"offers": [], "warning": "Duffel test."}))
    record.record_tool(_tool_message("search_flights", {"offers": [], "warning": "Duffel test."}))
    record.record_tool(_tool_message("search_stays", {"offers": [], "warning": "Sample lodging."}))

    # Both sides of one plan can be synthetic for different reasons, so a
    # single caveat must not stand in for both.
    assert record.caveats() == ["Duffel test.", "Sample lodging."]


def test_absorb_files_renders_file_data():
    record = TripRecord()
    record.absorb_files(
        {"files": {"/trip/brief.md": {"content": "# Brief\n2 people", "encoding": "utf-8"}}}
    )
    assert record.files == {"/trip/brief.md": "# Brief\n2 people"}


def test_absorb_files_ignores_state_without_files():
    record = TripRecord()
    record.absorb_files({"messages": []})
    assert record.files == {}


def test_offers_tolerates_missing_and_malformed_payloads():
    assert offers(None) == []
    assert offers({}) == []
    assert offers({"offers": "not a list"}) == []
    assert offers({"offers": [{"total_fare": 1}, "junk"]}) == [{"total_fare": 1}]


def test_task_label_sharpens_once_the_subagent_name_parses():
    # `task` arguments stream in as partial JSON, so the label starts generic.
    assert _tool_label("task", {}) == "Delegating to a subagent"
    assert _tool_label("task", {"subagent_type": "availability-scout"}) == (
        "Delegating to availability-scout"
    )
    assert _tool_label("search_flights", None) == "Searching flights"


def test_every_search_is_kept_in_order():
    """The scout varies its searches, so later ones do not replace earlier ones.

    The trip view offers a choice between them rather than merging: two
    searches are usually different questions, and the union sorted by price
    would rank a cheaper flight on other dates above a dearer one on the
    dates actually requested.
    """
    record = TripRecord()
    for fare in (500, 600, 700):
        record.record_tool(_tool_message("search_flights", {"offers": [{"total_fare": fare}]}))

    assert len(record.payloads["search_flights"]) == 3
    assert record.latest("search_flights") == {"offers": [{"total_fare": 700}]}


def test_search_label_describes_a_flight_query():
    payload = {
        "offers": [
            {
                "origin": "SFO",
                "destination": "NRT",
                "depart_date": "2026-09-10",
                "return_date": "2026-09-15",
                "travelers": 2,
            }
        ]
    }
    assert search_label(payload) == "SFO→NRT · Sep 10–Sep 15 · 2 travelers"


def test_search_label_handles_one_way_and_a_single_traveler():
    payload = {
        "offers": [
            {
                "origin": "BER",
                "destination": "LIS",
                "depart_date": "2026-10-02",
                "return_date": None,
                "travelers": 1,
            }
        ]
    }
    assert search_label(payload) == "BER→LIS · Oct 2 · 1 traveler"


def test_search_label_describes_a_lodging_query():
    payload = {
        "offers": [
            {
                "location": "kyoto",
                "check_in": "2026-09-10",
                "check_out": "2026-09-15",
                "guests": 2,
            }
        ]
    }
    assert search_label(payload) == "Kyoto · Sep 10–Sep 15 · 2 guests"


def _cabin_payload(requested: str, offer_cabin: str = "business") -> dict:
    return {
        "requested_cabin": requested,
        "offers": [
            {
                "origin": "SFO",
                "destination": "NRT",
                "depart_date": "2026-09-10",
                "return_date": "2026-09-15",
                "travelers": 2,
                "cabin": offer_cabin,
            }
        ],
    }


def test_search_label_names_a_non_default_cabin():
    """Two searches differing only by cabin must not present as the same option."""
    assert search_label(_cabin_payload("business")) == (
        "SFO→NRT · Sep 10–Sep 15 · 2 travelers · Business"
    )
    assert search_label(_cabin_payload("premium_economy")).endswith("· Premium economy")


def test_search_label_reads_the_request_not_the_first_offer():
    """A downgraded business search must still read "Business".

    Taking cabin off `offers[0]` — the way every other part of the label is
    built — would label this search "economy" and erase the mismatch from the
    one control a human is looking at.
    """
    downgraded = _cabin_payload("business", offer_cabin="economy")

    assert search_label(downgraded).endswith("· Business")


def test_search_label_suppresses_economy():
    """Economy is the default; labelling it adds a word that separates nothing."""
    payload = _cabin_payload("economy", offer_cabin="economy")

    assert search_label(payload) == "SFO→NRT · Sep 10–Sep 15 · 2 travelers"


def test_search_label_leaves_a_lodging_query_alone():
    """Stays carry no cabin, and a stray key must not grow one."""
    payload = {
        "requested_cabin": "business",
        "offers": [
            {
                "location": "kyoto",
                "check_in": "2026-09-10",
                "check_out": "2026-09-15",
                "guests": 2,
            }
        ],
    }
    assert search_label(payload) == "Kyoto · Sep 10–Sep 15 · 2 guests"


def test_search_label_degrades_rather_than_raising():
    # An empty search still needs a label — it is one of the options offered.
    assert search_label({"offers": []}) == "No results"
    assert search_label(None) == "No results"
    # A malformed date is dropped from the label instead of blowing it up.
    assert search_label(
        {"offers": [{"origin": "SFO", "destination": "NRT", "depart_date": "soon"}]}
    ) == ("SFO→NRT")


def test_stop_band_folds_the_tail_but_never_invents_one():
    """Three or more stops share the last band; an unknown count gets none.

    The bands are a colour domain for a scatter, which is an all-pairs form and
    so caps at three hues. All three are spent on real counts, leaving no slot
    to mean "we could not tell" — and "Two or more stops" is a claim about the
    itinerary, not a safe default. So an unreadable count returns None and the
    caller leaves the offer off the chart, exactly as it does for an offer whose
    fare would not parse.
    """
    assert stop_band(0) == "Nonstop"
    assert stop_band(1) == "One stop"
    assert stop_band(2) == "Two or more stops"
    # The fold: a four-stop itinerary does not earn a fourth colour.
    assert stop_band(4) == STOP_BANDS[2]

    # `map_offer` reads `stops` off the first slice, so an offer carrying no
    # slices at all arrives as None. That is the reachable path, not a guard.
    assert stop_band(None) is None
    assert stop_band("two") is None
    # Negative counts are nonsense rather than unknown, and nonstop is the
    # honest reading of "fewer than one stop".
    assert stop_band(-1) == "Nonstop"


def test_cancel_band_does_not_report_an_unknown_policy_as_a_no():
    """ "Not stated" exists so absence never becomes a negative finding.

    The same reasoning as warning about synthetic data on an empty result set: a
    traveler choosing on flexibility must not be told a policy was checked when
    it was not. Unlike the stop bands there is room for it — only two of these
    three carry a real value.
    """
    assert cancel_band(True) == "Free cancellation"
    assert cancel_band(False) == "No free cancellation"

    for absent in (None, "yes", 1, 0, ""):
        assert cancel_band(absent) == "Not stated", absent


def test_the_band_lists_stay_within_the_all_pairs_colour_budget():
    """Three is the ceiling, and adding a fourth is not a one-line change.

    A scatter can place any two dots side by side, so every pair of hues has to
    clear colour-blind separation — a strictly harder gate than the adjacent-only
    one a legend implies, and one the theme palette clears with three slots. A
    fourth band would need a fourth hue that no ordering of the palette
    separates, so this guards a fact about the palette, not a style preference.
    """
    for bands in (STOP_BANDS, CANCEL_BANDS):
        assert len(bands) == 3, bands
        # Duplicates would silently collapse two meanings onto one colour.
        assert len(set(bands)) == 3, bands


def test_money_format_falls_back_without_a_preset():
    """Only three currencies have a Streamlit preset; the rest need a format."""
    assert currency_code([{"currency": "JPY"}]) == "JPY"
    assert money_format([{"currency": "JPY"}]) == "yen"
    # An unlisted code still has to render as money, with the code visible.
    assert money_format([{"currency": "SEK"}]) == "%.2f SEK"
    # A blank or missing code falls through to the next offer, then the default.
    assert money_format([{"currency": ""}, {"currency": "EUR"}]) == "euro"
    assert money_format([{}]) == "dollar"
    assert money_format([]) == "dollar"


def test_source_column_keys_off_synthetic_never_the_provider():
    """Provenance is a property of the offer, not of where it came from.

    A provider called `duffel` returns fictional fares and sample lodging under
    a test token, and real inventory of both under a live one — so a `duffel`
    offer can be either. Reading the provider name gets this wrong in both
    directions; reading `synthetic` cannot.
    """
    labelled = source_column(
        [
            {"source": "duffel", "synthetic": True},
            {"source": "duffel", "synthetic": False},
            {"source": "sample-data", "synthetic": True},
        ]
    )
    assert [item["provenance"] for item in labelled] == ["Sample data", "Live", "Sample data"]
    # The original fields survive; provenance is added, not substituted.
    assert labelled[0]["source"] == "duffel"


def test_costing_series_needs_two_points_to_be_a_series():
    """A one-point trend is a dot, and an empty one is not a chart.

    These feed `st.metric(chart_data=...)`, which draws whatever it is handed —
    so the guard has to live here rather than being inferred from the render.
    """

    def costing(total):
        return {"total_estimated": total, "budget_total": 4000.0}

    assert costing_series([], "total_estimated") is None
    assert costing_series([costing(4300)], "total_estimated") is None
    assert costing_series([costing(4300), costing(3700)], "total_estimated") == [4300.0, 3700.0]

    # Payloads missing the key are skipped rather than read as zero, which would
    # draw a costing that never happened.
    assert costing_series([costing(4300), {}, costing(3700)], "total_estimated") == [4300.0, 3700.0]
    assert costing_series([costing(4300), {"total_estimated": None}], "total_estimated") is None
    # Ints become floats so the series is uniform for the sparkline.
    assert costing_series([costing(1), costing(2)], "total_estimated") == [1.0, 2.0]

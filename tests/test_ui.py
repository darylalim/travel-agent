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

from langchain_core.messages import ToolMessage

from travel_agent.ui import TripRecord, _tool_label, _unpack, offers, search_label


def _tool_message(name: str, payload: dict) -> ToolMessage:
    """A tool result shaped the way LangChain delivers one.

    A `@tool` returning a dict arrives with its payload JSON-encoded on
    `content`; this mirrors that exactly.
    """
    return ToolMessage(content=json.dumps(payload), name=name, tool_call_id=f"call-{name}")


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


def test_search_label_degrades_rather_than_raising():
    # An empty search still needs a label — it is one of the options offered.
    assert search_label({"offers": []}) == "No results"
    assert search_label(None) == "No results"
    # A malformed date is dropped from the label instead of blowing it up.
    assert search_label(
        {"offers": [{"origin": "SFO", "destination": "NRT", "depart_date": "soon"}]}
    ) == ("SFO→NRT")

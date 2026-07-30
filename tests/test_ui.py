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

from travel_agent.ui import (
    CANCEL_BANDS,
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
)


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

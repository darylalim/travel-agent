"""Tests for the two Streamlit page scripts.

Separate from `test_ui.py`, which drives the plain data structures the stream
routes into and touches no Streamlit machinery. These run the page scripts
themselves through `streamlit.testing.v1.AppTest` — a headless script runner,
so still no server and no browser — because the behaviour they pin lives in
Streamlit's own semantics rather than in this project's data:

- `default=` seeds a keyed widget's *first* render only, so a moving default
  has to be written to session state instead.
- Widget state outlives a trip, so "Start a new trip" alone does not reset a
  selector that opted into `persist_state="session"`.
- Streamlit's stop and rerun control flow raises through `BaseException`, so
  `except Exception` does not see it.
- Vega derives a chart's colour domain from the data unless the spec pins one,
  so a band absent from *this* search hands its hue to the next band along.
- An offer with no usable price must not reach a chart, where a missing fare
  would plot at the origin — the cheapest and quickest thing on screen.
- A chart mixing real and sample offers draws them as identical marks, so the
  blend has to be named in a caption.

Each of those is silent when it breaks: the app renders, nothing errors, and
the traveler is shown the wrong search, a mispriced dot, or loses a turn from
the transcript.

`stream_turn` is patched out, so no agent is built and no model is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from streamlit.runtime.scriptrunner_utils.exceptions import StopException
from streamlit.testing.v1 import AppTest

from travel_agent import ui
from travel_agent.ui import TripRecord

PAGES = Path(__file__).resolve().parents[1] / "app_pages"


def _flight_search(depart_date: str) -> ToolMessage:
    """One `search_flights` result, shaped the way LangChain delivers one."""
    payload = {
        "offers": [
            {
                "carrier": "Meridian Air",
                "origin": "SFO",
                "destination": "NRT",
                "depart_date": depart_date,
                "return_date": None,
                "travelers": 2,
                "total_fare": 900.0,
                "stops": 0,
                "currency": "USD",
                "synthetic": True,
            }
        ]
    }
    return ToolMessage(content=json.dumps(payload), name="search_flights", tool_call_id="call")


def _flight_offer(
    fare: float | None,
    minutes: int | None,
    stops: int,
    *,
    synthetic: bool = True,
) -> dict[str, Any]:
    """One flight offer, carrying only the fields the trip page reads."""
    return {
        "carrier": "Meridian Air",
        "origin": "SFO",
        "destination": "NRT",
        "depart_date": "2026-09-10",
        "return_date": None,
        "travelers": 2,
        "total_fare": fare,
        "duration_minutes": minutes,
        "stops": stops,
        "currency": "USD",
        "synthetic": synthetic,
    }


def _stay_offer(
    cost: float,
    rating: float,
    *,
    free: bool | None = True,
    synthetic: bool = True,
) -> dict[str, Any]:
    """One lodging offer, carrying only the fields the trip page reads."""
    return {
        "name": "Central boutique hotel",
        "kind": "hotel",
        "location": "Kyoto",
        "check_in": "2026-09-10",
        "check_out": "2026-09-17",
        "nights": 7,
        "guests": 2,
        "nightly_rate": round(cost / 7, 2),
        "total_cost": cost,
        "guest_rating": rating,
        "free_cancellation": free,
        "currency": "USD",
        "synthetic": synthetic,
    }


def _search(
    name: str,
    offers: list[dict[str, Any]],
    warning: str | None = None,
) -> ToolMessage:
    """A search result carrying several offers, as LangChain delivers one."""
    payload: dict[str, Any] = {"offers": offers}
    if warning is not None:
        payload["warning"] = warning
    return ToolMessage(content=json.dumps(payload), name=name, tool_call_id="call")


def _costing(by_category: dict[str, float], **extra: Any) -> ToolMessage:
    """A `summarize_budget` result, enough for the spend bar to render."""
    payload: dict[str, Any] = {
        "currency": "USD",
        "budget_total": 4000.0,
        "total_estimated": sum(by_category.values()),
        "by_category": by_category,
        **extra,
    }
    return ToolMessage(content=json.dumps(payload), name="summarize_budget", tool_call_id="budget")


def _statuses(app: AppTest) -> list[tuple[str, str]]:
    return [(element.label, element.state) for element in app.status]


def _charts(app: AppTest) -> list[dict[str, Any]]:
    """Vega-Lite specs rendered on the page, in render order."""
    return [json.loads(element.proto.spec) for element in app.get("vega_lite_chart")]


def _captions(app: AppTest) -> list[str]:
    return [element.value for element in app.caption]


def _table(app: AppTest, index: int = 0) -> tuple[list[str], dict[str, Any]]:
    """One rendered table's column order and its column config, as sent.

    Both come off the proto rather than the call site, so a key present in
    `column_config` but missing from `column_order` reads as what the traveler
    actually sees: absent.
    """
    element = app.dataframe[index]
    return list(element.proto.column_order), json.loads(element.proto.columns)


@pytest.fixture
def trip_page() -> AppTest:
    """The Trip page with an empty first trip."""
    app = AppTest.from_file(str(PAGES / "trip.py"), default_timeout=60)
    app.session_state["thread_id"] = "trip-one"
    app.session_state["record"] = TripRecord()
    return app


@pytest.fixture
def chat_page(monkeypatch: pytest.MonkeyPatch):
    """Factory for the Plan page, with `stream_turn` replaced by `stream`."""

    def build(stream: Any) -> AppTest:
        monkeypatch.setattr(ui, "stream_turn", stream)
        app = AppTest.from_file(str(PAGES / "chat.py"), default_timeout=60)
        app.session_state["thread_id"] = "thread"
        app.session_state["record"] = TripRecord()
        app.session_state["messages"] = []
        app.session_state["seen_caveats"] = []
        app.run()
        return app

    return build


def _fails(thread_id, prompt, record, on_activity=None):
    """A turn that delegates twice, streams a little, then dies."""
    if on_activity is not None:
        on_activity("call-1", "Searching flights")
        on_activity("call-2", "Costing the trip")
    yield "Day 1: Fushimi Inari"
    raise RuntimeError("overloaded_error: upstream capacity")


def _interrupted(thread_id, prompt, record, on_activity=None):
    """A turn stopped the way the stop button and a page switch stop one."""
    yield "Day 1: Fushimi Inari"
    raise StopException


def _quiet(thread_id, prompt, record, on_activity=None):
    """A turn answered straight from the model, with no tool calls."""
    yield "I plan trips: flights, lodging, budget, and a day-by-day itinerary."


def test_a_failed_turn_stays_in_history(chat_page):
    """The traveler's question must not be left hanging with no reply.

    It is appended before the run starts, so a turn that raises and appends
    nothing replays as a question with a blank space under it — and the error
    is drawn for that run only, so nothing explains the gap.
    """
    app = chat_page(_fails)
    app.chat_input[0].set_value("5 days in Kyoto").run()

    user, assistant = app.session_state["messages"]
    assert user["content"] == "5 days in Kyoto"
    # Whatever streamed before the failure is kept: st.write_stream returns the
    # joined text only on success, and the traveler already watched this arrive.
    assert assistant["content"] == "Day 1: Fushimi Inari"
    assert "overloaded_error" in assistant["error"]


def test_a_failed_turn_replays_as_a_failure(chat_page):
    """The replayed bubble must match the live one element for element.

    Deriving the status from `len(activity)` renders a failed turn as a green
    "2 steps", so the message changes colour on the next interaction.
    """
    app = chat_page(_fails)
    app.chat_input[0].set_value("5 days in Kyoto").run()
    live = _statuses(app)

    app.run()
    assert _statuses(app) == live == [("Could not finish", "error")]


def test_a_turn_without_tool_calls_still_replays_its_status(chat_page):
    """The live run always creates a status, so the replay has to as well.

    Skipping it when no tools ran drops an element that was on screen a moment
    earlier, which is the same visible rearrangement.
    """
    app = chat_page(_quiet)
    app.chat_input[0].set_value("what can you do?").run()
    live = _statuses(app)

    app.run()
    assert _statuses(app) == live == [("0 steps", "complete")]


def test_an_interrupted_turn_is_recorded(chat_page):
    """The stop button and a mid-stream page switch must not lose the turn.

    Both raise through `BaseException`, so `except Exception` never sees them
    and only a `finally` can record what happened.
    """
    app = chat_page(_interrupted)
    app.chat_input[0].set_value("5 days in Kyoto").run()

    assert len(app.session_state["messages"]) == 2
    assistant = app.session_state["messages"][-1]
    assert assistant["content"] == "Day 1: Fushimi Inari"
    assert assistant["status_label"] == "Interrupted"

    app.run()
    assert [element.value for element in app.error] == [assistant["error"]]


def test_the_failure_notice_belongs_to_the_assistant_bubble(chat_page):
    """Only assistant entries carry a failure, like activity and caveats."""
    app = chat_page(_quiet)
    app.session_state["messages"] = [
        {"role": "user", "content": "5 days in Kyoto", "error": "not mine to render"}
    ]
    app.run()

    assert [element.value for element in app.error] == []


def test_a_new_search_re_points_the_selector(trip_page):
    """A search the traveler has not seen is the one worth showing.

    `default=` seeds only the first render, so the control would stay on the
    second search while the caption still counted three.
    """
    record = trip_page.session_state["record"]
    record.record_tool(_flight_search("2026-09-10"))
    record.record_tool(_flight_search("2026-09-11"))
    trip_page.run()
    assert trip_page.segmented_control[0].value == 1

    record.record_tool(_flight_search("2026-09-12"))
    trip_page.run()
    assert trip_page.segmented_control[0].value == 2


def test_an_explicit_search_choice_survives_a_rerun(trip_page):
    """Re-pointing is for new searches, not for every rerun.

    A page switch reruns the script; that must not drag the traveler off the
    query they were reading.
    """
    record = trip_page.session_state["record"]
    record.record_tool(_flight_search("2026-09-10"))
    record.record_tool(_flight_search("2026-09-11"))
    trip_page.run()

    trip_page.segmented_control[0].set_value(0).run()
    trip_page.run()
    assert trip_page.segmented_control[0].value == 0


def test_a_new_trip_re_points_the_selector(trip_page):
    """Widget state outlives a trip, so the count alone is not enough.

    "Start a new trip" swaps the record and the thread but leaves the
    selection behind. Keyed on the count alone, the new trip's second search
    matches the old trip's and the control silently keeps the stale index —
    showing an older query under a caption that says otherwise.
    """
    first = trip_page.session_state["record"]
    first.record_tool(_flight_search("2026-09-10"))
    first.record_tool(_flight_search("2026-09-11"))
    trip_page.run()
    trip_page.segmented_control[0].set_value(0).run()

    second = TripRecord()
    trip_page.session_state["record"] = second
    trip_page.session_state["thread_id"] = "trip-two"

    # The first search of a new trip renders no control at all, so the count
    # is never synced down — which is what made the stale value collide.
    second.record_tool(_flight_search("2026-11-01"))
    trip_page.run()
    assert not trip_page.segmented_control

    second.record_tool(_flight_search("2026-11-02"))
    trip_page.run()
    assert trip_page.segmented_control[0].value == 1


def test_the_search_selector_cannot_be_left_blank(trip_page):
    """Deselecting would strand the table with no query named against it.

    Clicking the selected segment clears a `required=False` control, and the
    stamp will not re-point it because the search count has not changed — so
    the blank persists while the figures below stay on screen.
    """
    record = trip_page.session_state["record"]
    record.record_tool(_flight_search("2026-09-10"))
    record.record_tool(_flight_search("2026-09-11"))
    trip_page.run()

    assert trip_page.segmented_control[0].proto.required is True


def test_the_tradeoff_chart_needs_three_plottable_offers(trip_page):
    """Two dots describe a line, not a tradeoff.

    The chart exists so the price-versus-convenience frontier can be read as
    position. Below three offers there is no frontier to see and the table says
    it better, so the chart stays away rather than dressing up two numbers.
    """
    two = TripRecord()
    two.record_tool(
        _search("search_flights", [_flight_offer(1840, 640, 0), _flight_offer(1520, 775, 1)])
    )
    trip_page.session_state["record"] = two
    trip_page.run()
    assert _charts(trip_page) == []

    three = TripRecord()
    three.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0),
                _flight_offer(1520, 775, 1),
                _flight_offer(1310, 910, 2),
            ],
        )
    )
    trip_page.session_state["record"] = three
    trip_page.run()
    assert len(_charts(trip_page)) == 1


def test_an_offer_with_no_usable_figure_is_left_off_the_chart(trip_page):
    """`total_fare` is `None`, never `0`, when a price could not be parsed.

    That distinction exists so an unpriced offer is never recommended as the
    cheapest, and a chart has to honour it too: plotting `None` as zero would
    put the offer at the origin, the cheapest *and* quickest thing on screen.
    So the count that gates the chart is the count of offers with both figures,
    not the count of offers — three rows here, only one of them plottable.

    The table still lists all three, because a missing fare renders there as the
    dash it is.
    """
    record = trip_page.session_state["record"]
    record.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0),
                _flight_offer(None, 775, 1),
                _flight_offer(1310, None, 2),
            ],
        )
    )
    trip_page.run()

    assert _charts(trip_page) == []
    assert len(trip_page.dataframe) == 1


def test_an_offer_with_an_unreadable_stop_count_is_left_off_the_chart(trip_page):
    """Every stop band claims a real count, so an unknown one cannot be drawn.

    `map_offer` reads `stops` off the first slice, so an offer carrying none
    arrives with `stops: None`. Plotted, it would fall outside the pinned colour
    domain — a dot the legend does not key, in whatever colour Vega picks next.

    Two sound offers here rather than three, so excluding the third drops the
    count under the gate and the missing chart is what makes this observable. At
    three sound offers the chart renders either way and the bug hides.
    """
    record = trip_page.session_state["record"]
    record.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0),
                _flight_offer(1520, 775, 1),
                {**_flight_offer(1310, 910, 2), "stops": None},
            ],
        )
    )
    trip_page.run()

    assert _charts(trip_page) == []
    assert len(trip_page.dataframe) == 1


def test_a_chart_mixing_real_and_sample_offers_says_so(trip_page):
    """A blend of sources is the one case neither other safeguard covers.

    The standing notice at the top of the page speaks for a wholly synthetic
    set, and the table labels every row. Between them sits a mixed set, which a
    chart draws as identical dots — so the blend is named in a caption. A
    uniform set gets no such caption; repeating the standing notice under every
    chart would train the traveler to skip it.
    """
    mixed = TripRecord()
    mixed.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0, synthetic=True),
                _flight_offer(1520, 775, 1, synthetic=False),
                _flight_offer(1310, 910, 2, synthetic=False),
            ],
            warning="Some of these offers are synthetic.",
        )
    )
    trip_page.session_state["record"] = mixed
    trip_page.run()
    assert any("Mixed sources" in caption for caption in _captions(trip_page))

    uniform = TripRecord()
    uniform.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0),
                _flight_offer(1520, 775, 1),
                _flight_offer(1310, 910, 2),
            ],
            warning="These are synthetic sample offers.",
        )
    )
    trip_page.session_state["record"] = uniform
    trip_page.run()
    assert not any("Mixed sources" in caption for caption in _captions(trip_page))


def test_a_band_missing_from_one_search_does_not_repaint_the_others(trip_page):
    """Colour follows the band, never its position in this particular search.

    `availability-scout` varies its searches and the selector above flips
    between them, so the hues have to mean the same thing in each. Vega derives
    a colour domain from the data unless the spec pins one: with no one-stop
    offers in this search, an unpinned domain would hand that slot's hue to
    "Two or more stops", and a traveler who learned "the teal ones are nonstop"
    on the previous search is now reading a different chart. The same goes for
    a trip with no `transport` line shifting every later category up a slot.

    Nothing raises when this is wrong — the dots simply change meaning between
    two clicks of the selector — so the full domain is asserted here instead.
    """
    record = trip_page.session_state["record"]
    record.record_tool(_costing({"flights": 1840.0, "lodging": 1600.0}))
    record.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0),
                _flight_offer(1990, 615, 0),
                # Three stops, so it folds into the last band and leaves the
                # middle one with nothing in it.
                _flight_offer(1310, 910, 3),
            ],
        )
    )
    trip_page.run()

    # The spend bar renders above the flight scatter.
    spend, flights = _charts(trip_page)
    assert flights["encoding"]["color"]["scale"]["domain"] == [
        "Nonstop",
        "One stop",
        "Two or more stops",
    ]
    # The spend bar carries no colour encoding at all, which is what keeps it out
    # of this problem: one hue for every bar, so no two fills ever touch and no
    # pairlist has to be validated. See the comment at its call site — a stacked
    # part-to-whole version was reverted precisely because pinning a domain there
    # put non-adjacent palette slots in contact.
    assert "color" not in spend["encoding"]


def test_the_stays_chart_plots_cost_against_rating(trip_page):
    """The lodging scatter is a near-copy of the flight one, so pin what differs.

    Both charts run through one `_tradeoff_chart` and differ only in a `_Tradeoff`
    config: fields, titles and band vocabulary. That is exactly the shape where a
    swapped axis or a stale field name renders silently and looks plausible — a
    cost plotted on the rating axis is still a chart. Asserting the encoding
    catches it; looking at the page does not.
    """
    record = trip_page.session_state["record"]
    record.record_tool(
        _search(
            "search_stays",
            [
                _stay_offer(1876.0, 8.9, free=True),
                _stay_offer(1498.0, 8.6, free=False),
                _stay_offer(1043.0, 9.1, free=None),
            ],
        )
    )
    trip_page.run()

    (stays,) = _charts(trip_page)
    encoding = stays["encoding"]
    assert encoding["x"]["field"] == "guest_rating"
    assert encoding["x"]["title"] == "Guest rating"
    assert encoding["y"]["field"] == "total_cost"
    assert encoding["y"]["title"] == "Total cost (USD)"
    # Cancellation bands, not stop bands, and pinned in full so the absent one
    # cannot hand its hue to another.
    assert encoding["color"]["scale"]["domain"] == [
        "Free cancellation",
        "No free cancellation",
        "Not stated",
    ]
    # Neither axis includes zero: ratings cluster in a narrow band near the top
    # of a 0-10 scale and would otherwise collapse onto one edge.
    assert encoding["x"]["scale"]["zero"] is False
    assert encoding["y"]["scale"]["zero"] is False


def test_the_amount_due_at_the_property_reaches_the_table(trip_page):
    """A charge the total may exclude must not be dropped by the whitelist.

    `map_stay_result` keeps `total_amount` and `due_at_accommodation_amount`
    apart on purpose: Duffel's docs disagree about whether the second sits
    inside the first, so summing double-counts under one reading and
    subtracting understates under the other. That care is undone if the second
    figure never renders — `column_order` is a whitelist that hides every key
    it does not name, so leaving this one out shows a column headed "Total"
    that may exclude a charge due at the desk, on the one surface the agent
    cannot attach a caveat to.

    Asserted off the proto because the failure is invisible: the page renders,
    nothing raises, the table is simply one column narrower.
    """
    record = trip_page.session_state["record"]
    record.record_tool(
        _search(
            "search_stays",
            [
                {**_stay_offer(1876.0, 8.9), "due_at_accommodation": 140.0},
                {**_stay_offer(1498.0, 8.6, free=False), "due_at_accommodation": 90.0},
                # A live offer always carries the key and nulls it when the
                # rate has no such charge (`map_stay_result`); a sample offer
                # omits it entirely. Both reach the column as its placeholder,
                # so it has to tolerate being partly — or wholly — empty.
                _stay_offer(1043.0, 9.1, free=None),
            ],
        )
    )
    trip_page.run()

    order, columns = _table(trip_page)
    assert "due_at_accommodation" in order
    # Immediately after the total it qualifies; a traveler reading left to
    # right meets the caveat while the number it qualifies is still on screen.
    assert order.index("due_at_accommodation") == order.index("total_cost") + 1
    assert columns["due_at_accommodation"]["label"] == "Due at property"
    # The help text is the only place the reason survives to the traveler, so
    # it is part of the contract rather than decoration.
    assert "never added together" in columns["due_at_accommodation"]["help"]


def test_the_cabin_an_offer_came_back_in_reaches_the_table(trip_page):
    """Cabin is only worth requesting if what returned is visible.

    Same whitelist hazard as the charge above: `column_order` hides every key
    it does not name, so a Cabin column dropped from it would leave a business
    search looking exactly like an economy one. The help text carries the only
    explanation of "mixed" the traveler ever gets.
    """
    record = trip_page.session_state["record"]
    record.record_tool(
        _search(
            "search_flights",
            [
                {**_flight_offer(9000.0, 640, 0), "cabin": "business"},
                {**_flight_offer(9400.0, 700, 1), "cabin": "mixed"},
            ],
        )
    )
    trip_page.run()

    order, columns = _table(trip_page)
    assert "cabin" in order
    assert columns["cabin"]["label"] == "Cabin"
    assert "mixed" in columns["cabin"]["help"].lower()


def test_a_cabin_mismatch_is_reported_under_the_flights_table(trip_page):
    """The note is scoped to the selected search, not the page-wide caveats.

    Those accumulate across the trip and never clear, which is right for "these
    prices are not real" and wrong for a fact about one query's results.
    """
    record = trip_page.session_state["record"]
    payload = {
        "offers": [{**_flight_offer(900.0, 640, 0), "cabin": "economy"}],
        "requested_cabin": "business",
        "cabin_note": "Searched business, but 1 of 1 offers came back as economy.",
    }
    record.record_tool(
        ToolMessage(content=json.dumps(payload), name="search_flights", tool_call_id="call")
    )
    trip_page.run()

    assert any("came back as economy" in caption for caption in _captions(trip_page))
    # Not promoted into the standing warnings, which are the honesty channel.
    assert not any("came back as economy" in element.value for element in trip_page.warning)


def test_a_wholly_sample_data_chart_says_so_on_the_chart(trip_page):
    """The standing notice is too far away to serve a chart further down.

    It renders once at the top of the page, above the KPI row, the spend bar, the
    flight chart and its table — so by the time the stays scatter is on screen it
    has scrolled off, and `provenance` otherwise lives only in a tooltip that a
    touch device cannot reach. The table under each chart labels every row; the
    chart has to label itself, and not only when sources are mixed.
    """
    record = trip_page.session_state["record"]
    record.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0),
                _flight_offer(1520, 775, 1),
                _flight_offer(1310, 910, 2),
            ],
            warning="These are synthetic sample offers.",
        )
    )
    trip_page.run()
    assert any("Every dot here is sample data" in caption for caption in _captions(trip_page))

    # Real inventory says nothing — silence is the absence of a caveat, not a
    # claim, and the table's provenance column still reads "Live" per row.
    live = TripRecord()
    live.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0, synthetic=False),
                _flight_offer(1520, 775, 1, synthetic=False),
                _flight_offer(1310, 910, 2, synthetic=False),
            ],
        )
    )
    trip_page.session_state["record"] = live
    trip_page.run()
    assert not any("sample data" in caption for caption in _captions(trip_page))


def test_the_outbound_leg_note_appears_only_for_a_round_trip(trip_page):
    """Stops and journey time cover `slices[0]`, which is half a return trip.

    `map_offer` reads both off the first slice and `search_flights` appends a
    second when a return date is given, so a round trip whose return leg connects
    twice is still coloured "Nonstop" on an axis titled "Journey time". A one-way
    search has nothing omitted, so the note would be noise there.
    """
    one_way = TripRecord()
    one_way.record_tool(
        _search(
            "search_flights",
            [
                _flight_offer(1840, 640, 0),
                _flight_offer(1520, 775, 1),
                _flight_offer(1310, 910, 2),
            ],
        )
    )
    trip_page.session_state["record"] = one_way
    trip_page.run()
    assert not any("outbound leg" in caption for caption in _captions(trip_page))

    returning = TripRecord()
    returning.record_tool(
        _search(
            "search_flights",
            [
                {**_flight_offer(1840, 640, 0), "return_date": "2026-09-17"},
                {**_flight_offer(1520, 775, 1), "return_date": "2026-09-17"},
                {**_flight_offer(1310, 910, 2), "return_date": "2026-09-17"},
            ],
        )
    )
    trip_page.session_state["record"] = returning
    trip_page.run()
    assert any("outbound leg" in caption for caption in _captions(trip_page))


def test_one_fallback_format_reaches_every_money_surface(trip_page):
    """A money figure on this page must never render without its unit.

    `summarize_budget` takes `currency` as a free-form `str` and only USD, EUR
    and JPY have a Streamlit preset, so any other trip falls to the printf
    branch, which the four surfaces below once spelled separately.

    What is pinned is that one derivation reaches all four surfaces, **not**
    that the cards and the table agree about which currency. They read
    independent sources — the cards `budget["currency"]`, which the model
    supplies, and the table `currency_code(offers)`, which the provider does —
    so a plan budgeted in USD against JPY fares is correct and must keep
    rendering `$3,440.00` above `¥185,000`. Making either read the other would
    relabel real fares with a currency they are not quoted in, which is the
    misrepresentation the data-honesty invariant exists to prevent. The
    fixture sets both to SEK only so the *format* can be compared.

    Asserted on `proto.format`, which is the only place it is observable: the
    format is applied in the browser, so the proto body is `3440.0` either way
    and a value-based assertion passes on both spellings. Same reason
    `column_order` is read off the proto rather than the call site.
    """
    record = trip_page.session_state["record"]
    record.record_tool(
        _costing(
            {"flights": 21000.0, "lodging": 13400.0},
            currency="SEK",
            remaining=-1600.0,
            budget_total=32800.0,
            percent_of_budget_used=104.9,
            over_budget=True,
            overage=1600.0,
        )
    )
    record.record_tool(
        _search("search_flights", [{**_flight_offer(21000.0, 640, 0), "currency": "SEK"}])
    )
    trip_page.run()

    cards = {card.label: card.proto.format for card in trip_page.metric}
    assert cards["Estimated total"] == "%,.2f SEK"
    assert cards["Remaining"] == "%,.2f SEK"
    # The table under them is the comparison that matters: one derivation, so
    # the two cannot disagree about a currency neither of them presets.
    _, columns = _table(trip_page)
    assert columns["total_fare"]["type_config"]["format"] == "%,.2f SEK"
    # Percent is a unit of its own and stays a percent.
    assert cards["Budget used"] == "%.0f%%"

    # And the spend bar's money axis, the third surface on this page carrying a
    # figure in the trip's currency. Found by its encoding rather than by index
    # because `horizontal=True` swaps them: the categorical column lands on `y`
    # and the money column on `x`, which is also why the page passes `y_label`
    # to title an axis Vega draws along the bottom. A later "fix" to `x_label`
    # would title the categories instead and fail here.
    spend_bar = next(
        chart for chart in _charts(trip_page) if chart["encoding"]["y"].get("field") == "category"
    )
    assert spend_bar["encoding"]["x"]["title"] == "Amount (SEK)"

    # The fourth surface, and the only one whose figure is formatted in Python:
    # `st.error` takes finished text, so a raw interpolation printed the float
    # repr `1600.0` beneath cards the same payload renders as `1,600.00 SEK`.
    assert trip_page.error[0].value == "Over budget by 1,600.00 SEK."


def test_a_plan_with_no_budget_shows_the_total_and_no_verdict(trip_page: AppTest):
    """Without a budget the page shows the estimate and says there is nothing to compare.

    `summarize_budget` leaves the comparison keys out when no budget was set.
    Rendering "Remaining" and "Budget used" anyway would show two blank cards
    beside the total, which reads as a budget that exists and was not worked
    out, the opposite of what happened.
    """
    record = trip_page.session_state["record"]
    record.record_tool(_costing({"flights": 420.0, "lodging": 525.0}, budget_total=None))
    trip_page.run()

    assert [card.label for card in trip_page.metric] == ["Estimated total"]
    assert not trip_page.error
    assert any("No budget set" in caption for caption in _captions(trip_page))

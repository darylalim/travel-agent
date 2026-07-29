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

Each of those is silent when it breaks: the app renders, nothing errors, and
the traveler is shown the wrong search or loses a turn from the transcript.

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


def _statuses(app: AppTest) -> list[tuple[str, str]]:
    return [(element.label, element.state) for element in app.status]


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

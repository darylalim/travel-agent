"""The current-date middleware — the agent's only source of today's date.

Without it a model fills in the year from training, and a real run searched a
December that had already passed. Every failure here is silent: the agent
still answers, just about the wrong year.
"""

from __future__ import annotations

from datetime import date

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage

from travel_agent import clock
from travel_agent.clock import CurrentDateMiddleware, current_date_note
from travel_agent.subagents import build_subagents


def _request(system: str) -> ModelRequest:
    return ModelRequest(
        model=FakeListChatModel(responses=[]),
        messages=[],
        system_message=SystemMessage(content=system),
    )


def _run(middleware: CurrentDateMiddleware, system: str) -> str:
    """Pass one request through and return the system text the model would get."""
    seen: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen.append(request)
        return ModelResponse(result=[])

    middleware.wrap_model_call(_request(system), handler)
    assert seen[0].system_message is not None
    return seen[0].system_message.text


def test_the_note_names_the_weekday_date_and_year():
    note = current_date_note(date(2026, 10, 2))
    assert note.startswith("Today is Friday, 2 October 2026.")


def test_the_date_is_appended_after_the_static_prompt(monkeypatch):
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 2))
    text = _run(CurrentDateMiddleware(), "Static prompt.")
    # Appended, not prepended: the static prompt stays the cache prefix.
    assert text.startswith("Static prompt.")
    assert text.endswith(current_date_note(date(2026, 10, 2)))


def test_the_date_is_read_per_call_not_at_construction(monkeypatch):
    """`graph` is built at import and `langgraph dev` can run for days."""
    middleware = CurrentDateMiddleware()
    texts = []
    for day in (date(2026, 10, 2), date(2026, 10, 3)):
        monkeypatch.setattr(clock, "today", lambda day=day: day)
        texts.append(_run(middleware, "Static prompt."))

    assert "2 October" in texts[0]
    assert "3 October" in texts[1]


def test_every_subagent_is_told_the_date():
    # Subagents do not inherit the main agent's middleware.
    for subagent in build_subagents([]):
        kinds = [type(m) for m in subagent.get("middleware", [])]
        assert CurrentDateMiddleware in kinds, subagent["name"]

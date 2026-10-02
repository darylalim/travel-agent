"""The current date, and the middleware that tells every agent what it is.

A model has no clock. Asked for "early December" with no year, Opus 5.5 picked
a December that had already passed, the scout ran its whole search pass
against it, Duffel rejected every flight, and the sample provider happily
priced lodging for the past. The date therefore goes into the system prompt
on **every model call** rather than into `prompts.py`: `agent.py` builds the
module-level `graph` once at import time, and `langgraph dev` can keep that
process up for days, so a date baked into prompt text goes stale silently.

`today()` is the one read of the clock. The tool layer's past-date check and
the middleware both go through it, so `tests/conftest.py` can pin it — the
suite's fixture dates would otherwise start failing the day they pass.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date, datetime

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse


def today() -> date:
    """Today's date in the local timezone of the machine running the agent.

    The traveler's timezone is unknown here, so the server's is the honest
    choice — and the past-date check only rejects dates *before* it, so a
    traveler a day ahead of the server can still search their own today.
    """
    return datetime.now().astimezone().date()


def current_date_note(day: date) -> str:
    """The system-prompt line stating the date, e.g. "Today is Friday, 2 October 2026."."""
    return (
        f"Today is {day:%A}, {day.day} {day:%B %Y}. A date given without a year "
        "means its next occurrence on or after today."
    )


class CurrentDateMiddleware(AgentMiddleware):
    """Append today's date to the system message before each model call.

    Appended rather than prepended so the static prompt ahead of it stays a
    stable cache prefix; the line itself changes once a day.
    """

    def _with_date(self, request: ModelRequest) -> ModelRequest:
        note = current_date_note(today())
        return request.override(
            system_message=append_to_system_message(request.system_message, note)
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._with_date(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._with_date(request))

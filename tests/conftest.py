"""Test isolation.

Provider selection is read from the environment at call time, so without this
the suite picks up whatever a developer has exported — and the combination the
README tells you to set (`TRAVEL_AGENT_PROVIDER=duffel` plus a token) makes
the sample-provider tests fail and sends a real request to api.duffel.com.
Tests must not depend on ambient configuration or touch the network.

The wall clock is ambient too. The search tools reject a date before
`clock.today()`, and the suite's fixture dates are literals, so each one would
start failing the day it passes — on a schedule, with no change to the code.

LangSmith tracing is the third. `.env.example` sets `LANGSMITH_TRACING=true`,
and anything that exports it makes every graph and tool invocation here POST
runs to the LangSmith API — and the tracer's own callbacks shift the scripted
fake model in `test_memory.py` enough to fail it.
"""

from __future__ import annotations

from datetime import date

import pytest
from langsmith import tracing_context

from travel_agent import clock
from travel_agent.tools.availability import _build_provider

# Earlier than every literal date in the suite. A test about the past-date
# check sets its own `today` on top of this.
PINNED_TODAY = date(2026, 1, 1)


@pytest.fixture(autouse=True)
def pin_today(monkeypatch: pytest.MonkeyPatch):
    """Freeze `clock.today()` so fixture dates never fall into the past."""
    monkeypatch.setattr(clock, "today", lambda: PINNED_TODAY)


@pytest.fixture(autouse=True)
def disable_tracing():
    """Turn LangSmith tracing off, whatever the environment says.

    A context override rather than `delenv`: langsmith reads its env vars
    through an `lru_cache`, so a value seen once outlives the variable, while
    `tracing_context` is consulted before the environment on every check.
    """
    with tracing_context(enabled=False):
        yield


_PROVIDER_ENV = ("TRAVEL_AGENT_PROVIDER", "DUFFEL_API_TOKEN", "DUFFEL_SUPPLIER_TIMEOUT_MS")


@pytest.fixture(autouse=True)
def isolate_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Unset provider config and drop the cached provider around every test."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    _build_provider.cache_clear()
    yield
    _build_provider.cache_clear()

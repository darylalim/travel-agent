"""Test isolation.

Provider selection is read from the environment at call time, so without this
the suite picks up whatever a developer has exported — and the combination the
README tells you to set (`TRAVEL_AGENT_PROVIDER=duffel` plus a token) makes
the sample-provider tests fail and sends a real request to api.duffel.com.
Tests must not depend on ambient configuration or touch the network.
"""

from __future__ import annotations

import pytest

from travel_agent.tools.availability import _build_provider

_PROVIDER_ENV = ("TRAVEL_AGENT_PROVIDER", "DUFFEL_API_TOKEN", "DUFFEL_SUPPLIER_TIMEOUT_MS")


@pytest.fixture(autouse=True)
def isolate_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Unset provider config and drop the cached provider around every test."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    _build_provider.cache_clear()
    yield
    _build_provider.cache_clear()

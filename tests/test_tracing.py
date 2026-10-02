"""LangSmith tracing: off for the suite, labelled for the entry points.

Tracing itself needs no code — LangChain attaches a tracer whenever
`LANGSMITH_TRACING=true` — so what is pinned here is the edges: that the
suite's override in `conftest.py` really beats an exported variable, and that
the CLI and Streamlit runs carry the name and tag that tell them apart.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.run_helpers import get_tracing_context
from langsmith.utils import get_env_var, tracing_is_enabled

import travel_agent.agent as agent_module
from travel_agent import main, ui
from travel_agent.config import TRACE_NAME

# `@overload`s stacked on an `lru_cache` hide `cache_clear` from the type checker.
_clear_env_cache = cast(Any, get_env_var).cache_clear


@pytest.fixture
def tracing_exported(monkeypatch: pytest.MonkeyPatch):
    """Export the variables `.env.example` sets, past langsmith's env cache."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_never_sent")
    # A closed local port, so a regression fails here instead of reaching
    # api.smith.langchain.com.
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:9")
    _clear_env_cache()
    yield
    _clear_env_cache()


def test_the_exported_variable_would_trace_without_the_override(tracing_exported):
    # Guards the next test: if this environment did not turn tracing on, "no
    # tracer attached" would pass whether or not conftest disabled anything.
    assert tracing_is_enabled({**get_tracing_context(), "enabled": None}) is True


def test_a_run_gets_no_tracer_even_when_tracing_is_exported(tracing_exported):
    managers: list[Any] = []

    def capture(_x: int, config: RunnableConfig) -> None:
        managers.append(config.get("callbacks"))

    RunnableLambda(capture).invoke(0)
    (manager,) = managers
    assert isinstance(manager, BaseCallbackManager)
    assert not any(isinstance(h, LangChainTracer) for h in manager.handlers)


def test_streamlit_runs_are_named_and_tagged():
    config = ui.thread_config("trip-1")
    assert config["configurable"] == {"thread_id": "trip-1"}
    assert config["run_name"] == TRACE_NAME
    assert config["tags"] == ["streamlit"]


def test_cli_runs_are_named_and_tagged(monkeypatch: pytest.MonkeyPatch):
    seen: list[dict[str, Any]] = []

    class _Agent:
        def stream(self, _input, config, **_kwargs):
            seen.append(config)
            return iter(())

    monkeypatch.setattr(agent_module, "build_agent", lambda *a, **k: _Agent())
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")

    assert main.main(["plan a trip", "--thread", "t"]) == 0
    assert seen == [{"configurable": {"thread_id": "t"}, "run_name": TRACE_NAME, "tags": ["cli"]}]

"""Scripted tools serve each example its own inventory and search results.

Every failure here is silent in a real run: a script that leaks into another
example scores that example against inventory it never asked for, and a
scripted offer with the wrong shape or warning tests a contract no real
provider has. No network, no model calls.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import travel_agent.agent as agent_module
from evals import run as harness
from evals import scripted
from travel_agent.tools.availability import (
    SAMPLE_DISCLAIMER,
    SampleProvider,
    search_flights,
    search_stays,
)

LIVE_BUSINESS = {
    "flights": {
        "source": "duffel",
        "synthetic": False,
        "rules": [
            {
                "match": {"origin": "LHR", "destination": "SIN"},
                "offers": [
                    {"cabin": "premium_economy", "expires_in_seconds": 1800},
                    {"cabin": "premium_economy", "expires_in_seconds": 1800},
                ],
            }
        ],
    }
}
FLIGHT: dict[str, Any] = {
    "origin": "LHR",
    "destination": "SIN",
    "depart_date": "2027-03-01",
    "return_date": "2027-03-08",
    "travelers": 1,
    "cabin": "business",
}
STAY: dict[str, Any] = {
    "location": "Singapore",
    "check_in": "2027-03-01",
    "check_out": "2027-03-08",
    "guests": 1,
}


@pytest.fixture(autouse=True)
def scripted_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", scripted.PROVIDER)
    monkeypatch.setattr(scripted, "_SCRIPTS", {})


def _inputs(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def _register(text: str, script: dict) -> dict:
    inputs = _inputs(text)
    scripted.register([SimpleNamespace(inputs=inputs, metadata={"scripted_tools": script})])
    return inputs


# --- inventory ---------------------------------------------------------------


def test_a_live_scripted_flight_has_no_warning_and_keeps_the_cabin_it_came_back_in():
    with scripted.active(_register("brief", LIVE_BUSINESS)):
        payload = search_flights.invoke(FLIGHT)
    assert "warning" not in payload
    assert {o["cabin"] for o in payload["offers"]} == {"premium_economy"}
    assert all(o["synthetic"] is False and o["live_mode"] for o in payload["offers"])
    assert all(o["expires_at"] and o["expires_in_seconds"] == 1800 for o in payload["offers"])
    # The tool's own mismatch handling runs on scripted offers, unchanged.
    assert payload["requested_cabin"] == "business" and payload["cabin_note"]


def test_a_kind_the_script_leaves_out_is_still_labelled_sample_data():
    with scripted.active(_register("brief", LIVE_BUSINESS)):
        payload = search_stays.invoke(STAY)
    assert payload["warning"] == SAMPLE_DISCLAIMER
    assert all(o["synthetic"] for o in payload["offers"])


def test_an_unmatched_search_is_plain_sample_data_and_still_warned():
    # A scout trying a nearby airport gets sample offers. The side is live, so
    # the provider's note is silent, and the per-offer backstop has to warn.
    gatwick: dict[str, Any] = {**FLIGHT, "origin": "LGW"}
    with scripted.active(_register("brief", LIVE_BUSINESS)):
        provider = scripted.ScriptedProvider()
        offers = provider.search_flights(**gatwick)
        payload = search_flights.invoke(gatwick)
    assert offers == SampleProvider().search_flights(**gatwick)
    assert payload["warning"]


def test_without_a_script_the_provider_is_sample_data():
    with scripted.active(_inputs("no script registered")):
        provider = scripted.ScriptedProvider()
        assert provider.search_flights(**FLIGHT) == SampleProvider().search_flights(**FLIGHT)
        assert provider.synthetic_note("flights") == SAMPLE_DISCLAIMER


def test_repeated_searches_walk_the_response_sequence_and_stay_on_the_last():
    rule = {
        "match": {"origin": "SEA"},
        "responses": [
            {"offers": [{"expires_in_seconds": 12}]},
            {"offers": [{"expires_in_seconds": 1700}]},
        ],
    }
    script = {"flights": {"source": "duffel", "synthetic": False, "rules": [rule]}}
    sea = {**FLIGHT, "origin": "SEA", "destination": "ICN", "return_date": None}
    with scripted.active(_register("brief", script)):
        seen = [search_flights.invoke(sea)["offers"][0]["expires_in_seconds"] for _ in range(3)]
    assert seen == [12, 1700, 1700]


def test_scripted_offers_carry_only_keys_a_real_provider_emits():
    with scripted.active(_register("brief", LIVE_BUSINESS)):
        offer = scripted.ScriptedProvider().search_flights(**FLIGHT)[0]
    sample = set(SampleProvider().search_flights(**FLIGHT)[0])
    assert sample <= set(offer) <= sample | scripted.LIVE_ONLY_KEYS["flights"]


# --- web search --------------------------------------------------------------

SEARCH = {
    "search": [
        {"match_any": ["restaurant"], "result": {"answer": "a", "results": [{"title": "t"}]}},
        {"match_any": ["weather"], "error": "432"},
    ],
    "search_default": "empty",
}


def test_scripted_search_answers_fails_or_comes_back_empty_by_rule():
    with scripted.active(_register("brief", SEARCH)):
        tool = scripted.ScriptedSearch()
        found = tool.invoke({"query": "vegetarian restaurant Amsterdam"})
        failed = tool.invoke({"query": "Amsterdam weather in August"})
        empty = tool.invoke({"query": "canal cruise"})
    assert found["answer"] == "a" and found["results"][0]["title"] == "t"
    assert empty["results"] == []
    # What the agent reads, as it read the real outage.
    assert str(failed).startswith("{'error': ValueError(\"Error 432: This request exceeds")


def test_only_an_example_that_scripts_search_replaces_the_real_tool():
    assert scripted.search_tools_for(_register("with search", SEARCH))
    assert scripted.search_tools_for(_register("without", LIVE_BUSINESS)) is None


def test_build_agent_hands_the_given_search_to_the_main_agent(monkeypatch):
    model = _Searcher()
    monkeypatch.setattr(agent_module, "_chat_model", lambda *a, **k: model)
    tool = scripted.ScriptedSearch()
    agent = agent_module.build_agent(search_tools=[tool])
    assert agent.nodes["tools"].bound.tools_by_name["tavily_search"] is tool


# --- isolation under concurrency ---------------------------------------------


class _Searcher(BaseChatModel):
    """Stateless, so one instance can serve concurrent runs: search, then stop."""

    @property
    def _llm_type(self) -> str:
        return "scripted-searcher"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _Searcher:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            reply = AIMessage("done")
        else:
            reply = AIMessage(
                "", tool_calls=[{"name": "search_flights", "args": FLIGHT, "id": "search"}]
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_concurrent_runs_each_see_only_their_own_script(monkeypatch):
    # The tool runs on a LangGraph worker thread inside the subagent, not on
    # the thread that set the script. A global swap would leak one example's
    # inventory into the other.
    monkeypatch.setattr(agent_module, "_chat_model", lambda *a, **k: _Searcher())
    first_class = {
        "flights": {
            **LIVE_BUSINESS["flights"],
            "rules": [{"match": {"origin": "LHR"}, "offers": [{"cabin": "first"}]}],
        }
    }
    briefs = {
        "premium_economy": _register("Search LHR-SIN, run A.", LIVE_BUSINESS),
        "first": _register("Search LHR-SIN, run B.", first_class),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = {
            cabin: pool.submit(harness.run_subagent, "availability-scout", inputs)
            for cabin, inputs in briefs.items()
        }
    for cabin, future in runs.items():
        result = next(r for r in future.result()["tool_results"] if r["name"] == "search_flights")
        assert f'"cabin": "{cabin}"' in result["content"], cabin


def test_a_patched_fare_carries_a_total_that_agrees_with_it():
    script = {
        "flights": {
            "source": "duffel",
            "synthetic": False,
            "rules": [{"match": {"origin": "LHR"}, "offers": [{"fare_per_traveler": 1184.6}]}],
        }
    }
    with scripted.active(_register("brief", script)):
        offer = scripted.ScriptedProvider().search_flights(**{**FLIGHT, "travelers": 2})[0]
    assert offer["fare_per_traveler"] == 1184.6 and offer["total_fare"] == 2369.2

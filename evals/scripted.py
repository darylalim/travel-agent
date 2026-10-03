"""Scripted tools: an example decides what the search tools return.

Sample data is deterministic but tame: every offer comes back in the cabin asked
for, never expires, and every search is synthetic. The cases the prompts work
hardest at — a business search answered in premium economy, a `mixed` round
trip, an offer seconds from expiry, live flights beside sample lodging, research
that partly fails — never arise from it, and Duffel and Tavily cost credits and
cannot be told to misbehave. So an example can carry a script in its metadata,
under `scripted_tools`:

    "scripted_tools": {
      "flights": {"source": "duffel", "synthetic": false, "rules": [
        {"match": {"origin": "LHR"}, "offers": [{"cabin": "premium_economy"}]}]},
      "stays": {...},
      "search": [{"match_any": ["restaurant"], "result": {"answer": "...", "results": []}},
                 {"match_any": ["weather"], "error": "432"}],
      "search_default": "error"
    }

A flight or stay rule answers the first search whose arguments match, with one
response or a `responses` sequence for repeated searches. Its offers start as
sample offers and the script patches only what differs, so their keys stay those
a real provider emits. A kind the script leaves out is sample data, exactly as
today. A search rule answers the first query containing any of its words.

Each run reads its own script from a `ContextVar` set around the stream, so
examples with different scripts can run concurrently: LangGraph carries the
context into the threads it runs tools on, the subagent's included.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_tavily.tavily_search import TavilySearchInput
from pydantic import BaseModel

from travel_agent.tools import availability
from travel_agent.tools.availability import SAMPLE_DISCLAIMER, SampleProvider

PROVIDER = "scripted"

# Keys a scripted offer may set beyond a sample offer's own: those a live Duffel
# offer adds. `test_duffel.py` pins the flight set against `map_offer`.
LIVE_ONLY_KEYS = {
    "flights": frozenset({"live_mode", "offer_id", "expires_at", "expires_in_seconds", "slices"}),
    "stays": frozenset({"due_at_accommodation", "expires_at", "expires_in_seconds"}),
}

# What a Tavily key out of credits answers, as the agent saw it in a real run.
# Only the first 97 characters were captured; the tail is a reconstruction.
TAVILY_432 = (
    "Error 432: This request exceeds your plan's set usage limit. "
    "Please upgrade your plan or contact support@tavily.com"
)


@dataclass
class _Run:
    """One run's script, and how many times each rule has answered so far."""

    script: dict
    calls: dict[tuple[str, int], int] = field(default_factory=dict)


_CURRENT: contextvars.ContextVar[_Run | None] = contextvars.ContextVar("scripted", default=None)
_SCRIPTS: dict[str, dict] = {}


def _input_key(inputs: dict) -> str:
    return json.dumps(inputs, sort_keys=True)


def register(examples: list[Any]) -> int:
    """Index the scripts of LangSmith examples by their inputs; return how many.

    Keyed by inputs because that is all `evaluate()` hands a run function.
    `test_eval_datasets.py` keeps scripted inputs unique across every dataset.
    """
    count = 0
    for example in examples:
        script = (example.metadata or {}).get("scripted_tools")
        if script:
            _SCRIPTS[_input_key(example.inputs)] = script
            count += 1
    return count


@contextmanager
def active(inputs: dict) -> Iterator[None]:
    """Make the script registered for `inputs`, if any, the one this run reads."""
    script = _SCRIPTS.get(_input_key(inputs))
    token = _CURRENT.set(_Run(script) if script else None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def search_tools_for(inputs: dict) -> list[BaseTool] | None:
    """The scripted search for an example that scripts one, else None (the real one)."""
    script = _SCRIPTS.get(_input_key(inputs))
    return [ScriptedSearch()] if script and "search" in script else None


# --- availability ------------------------------------------------------------


def _matches(field_name: str, expected: Any, actual: Any) -> bool:
    if field_name in {"origin", "destination"}:
        return str(actual).strip().upper() == str(expected).upper()
    if field_name == "location":
        return str(expected).casefold() in str(actual).casefold()
    return actual == expected


def _seconds_from_now(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


class ScriptedProvider(SampleProvider):
    """Serves the current run's script, and sample data wherever it is silent."""

    name = PROVIDER

    def _side(self, kind: str) -> dict | None:
        run = _CURRENT.get()
        return run.script.get(kind) if run else None

    def synthetic_note(self, kind: availability.SearchKind) -> str | None:
        side = self._side(kind)
        if side is None or side.get("synthetic", True):
            return (side or {}).get("warning", SAMPLE_DISCLAIMER)
        return None

    def _respond(self, kind: str, args: dict, base: list[dict]) -> list[dict]:
        side, run = self._side(kind), _CURRENT.get()
        if side is None or run is None:
            return base
        for index, rule in enumerate(side.get("rules", [])):
            if all(_matches(f, v, args.get(f)) for f, v in rule.get("match", {}).items()):
                responses = rule.get("responses") or [{"offers": rule.get("offers", [])}]
                seen = run.calls.get((kind, index), 0)
                run.calls[(kind, index)] = seen + 1
                response = responses[min(seen, len(responses) - 1)]
                return [
                    self._offer(kind, side, base, i, patch)
                    for i, patch in enumerate(response["offers"])
                ]
        return base

    @staticmethod
    def _offer(kind: str, side: dict, base: list[dict], index: int, patch: dict) -> dict:
        offer = dict(base[index % len(base)]) if base else {}
        offer |= {"source": side.get("source", PROVIDER), "synthetic": side.get("synthetic", True)}
        if not offer["synthetic"] and kind == "flights":
            offer |= {"live_mode": True, "offer_id": f"off_scripted_{index}"}
        offer |= patch
        # Derived like a real provider derives them, so a patch cannot leave a
        # total that disagrees with its own per-traveler fare.
        if "fare_per_traveler" in patch and "total_fare" not in patch:
            offer["total_fare"] = round(patch["fare_per_traveler"] * offer["travelers"], 2)
        if "expires_in_seconds" in patch and "expires_at" not in patch:
            offer["expires_at"] = _seconds_from_now(patch["expires_in_seconds"])
        return offer

    def search_flights(
        self,
        origin,
        destination,
        depart_date,
        return_date,
        travelers,
        cabin=availability.DEFAULT_CABIN,
    ) -> list[dict]:
        base = super().search_flights(
            origin, destination, depart_date, return_date, travelers, cabin
        )
        args = {
            "origin": origin,
            "destination": destination,
            "depart_date": depart_date,
            "return_date": return_date,
            "travelers": travelers,
            "cabin": cabin,
        }
        return self._respond("flights", args, base)

    def search_stays(self, location, check_in, check_out, guests, max_nightly_rate) -> list[dict]:
        base = super().search_stays(location, check_in, check_out, guests, max_nightly_rate)
        args = {
            "location": location,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
            "max_nightly_rate": max_nightly_rate,
        }
        return self._respond("stays", args, base)


availability._PROVIDERS[PROVIDER] = ScriptedProvider


# --- web search --------------------------------------------------------------


def _search_result(query: str, result: dict) -> dict:
    """A canned answer in the shape Tavily returns."""
    return {
        "query": query,
        "follow_up_questions": None,
        "answer": result.get("answer"),
        "images": [],
        "results": [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "score": 0.9,
                "raw_content": None,
            }
            for r in result.get("results", [])
        ],
        "response_time": 0.5,
    }


class ScriptedSearch(BaseTool):
    """Stands in for `tavily_search`: same name, same arguments, scripted answers."""

    name: str = "tavily_search"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    args_schema: type[BaseModel] = TavilySearchInput

    def _run(self, query: str, **kwargs: Any) -> dict:
        run = _CURRENT.get()
        script = run.script if run else {}
        for rule in script.get("search", []):
            if any(word.casefold() in query.casefold() for word in rule["match_any"]):
                if "error" in rule:
                    return {"error": ValueError(TAVILY_432)}
                return _search_result(query, rule["result"])
        if script.get("search_default", "error") == "empty":
            return _search_result(query, {})
        # The real tool's own shape for a failed call: a dict holding the
        # exception, which the agent reads as `{'error': ValueError("…")}`.
        return {"error": ValueError(TAVILY_432)}

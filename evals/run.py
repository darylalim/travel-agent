"""Run the evaluation datasets against the real models and score them in LangSmith.

    uv run python -m evals.run budget_analyst --limit 1 --no-judge   # cheapest smoke test
    uv run python -m evals.run budget_analyst availability_scout
    uv run python -m evals.run trajectory --limit 1                  # one full trip
    uv run python -m evals.run trajectory final_response
    uv run python -m evals.run budget_analyst_hard --repetitions 3   # a pass rate per example
    uv run python -m evals.run final_response --keys lisbon-over-budget --repetitions 3
    uv run python -m evals.run trajectory --no-web-search            # Tavily out of credits
    uv run python -m evals.run availability_scout_scripted --repetitions 3   # scripted tools

`--no-web-search` matters more than it looks. A Tavily key with no credits left
still builds the search tool, which then answers every call with an error the
agent quietly works around, so a run would measure a broken search while its
metadata said search was on.

The `_hard` datasets hold examples a plausible agent gets wrong, so their scores
can move; the others are a regression set expected to stay at 1. Run the hard
ones with `--repetitions`, since a single pass or fail is a sample, not a rate.

The subagent datasets run one subagent each on its real model, through the
real graph. A scripted `Dispatcher` takes the main agent's seat: it makes a
single `task` call carrying the example's brief and then stops. That way the
subagent gets exactly what production gives it, with deepagents' own middleware
stack, the filesystem and `CurrentDateMiddleware`. A subagent rebuilt by hand
would drift from that, and a drifted harness scores a stack nobody runs.

`trajectory` and the `final_response` sets run the whole agent instead: `build_agent()`
with its default model and effort, exactly as the CLI builds it, on a fresh
checkpointer and store so no example meets a traveler profile an earlier one
wrote. Several requests appear in both datasets, so one process runs each
request once and scores the same transcript in both. That halves the cost of
the overlap and makes the two scores describe one run.

Tool calls are captured **during** the stream, with `subgraphs=True`, for the
reason `ui.py` gives: deepagents folds a subagent back into the parent without
its messages, so none of them are left in state once the run ends.

The datasets assume sample data, so `main()` sets
`TRAVEL_AGENT_PROVIDER=sample-data` over whatever `.env` says. Against a live
token, every "labels it as sample data" criterion would mark a correct run wrong.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from collections.abc import Callable, Iterable
from datetime import date
from functools import cache
from typing import Any

from deepagents.backends.utils import file_data_to_string
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from evals import scripted
from evals.upload import load_datasets

# Dataset file stem -> the subagent it scores. Names are checked against the
# roster in tests, since a stale one would dispatch to a subagent that is gone.
SUBAGENT_FOR = {
    "availability_scout": "availability-scout",
    "availability_scout_hard": "availability-scout",
    "availability_scout_scripted": "availability-scout",
    "budget_analyst": "budget-analyst",
    "budget_analyst_hard": "budget-analyst",
}
# Dataset file stems scored by running the whole agent, main model included.
AGENT_DATASETS = ("final_response", "final_response_hard", "final_response_scripted", "trajectory")
# Datasets whose examples carry `scripted_tools`; see `evals/scripted.py`.
SCRIPTED_DATASETS = frozenset({"availability_scout_scripted", "final_response_scripted"})

# Different from the main agent's model, so no model grades its own transcript.
# Sonnet 5.5 rejects forced `tool_choice`, which LangChain's default structured
# output relies on, so the judge uses `method="json_schema"`. That asks for
# native structured output (`output_config.format`) instead.
JUDGE_MODEL = "anthropic:claude-sonnet-5-5"
JUDGE_EFFORT = "medium"
# Server-side refusal fallback: if Sonnet 5.5 refuses to grade a transcript, the
# API reruns the request on a fallback model instead of returning nothing.
_JUDGE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Dispatcher(BaseChatModel):
    """Stand-in main agent that hands the brief to one subagent, then stops.

    Stateless on purpose: it decides from the messages it is given, so one
    instance can serve concurrent examples.
    """

    subagent: str

    @property
    def _llm_type(self) -> str:
        return "eval-dispatcher"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Dispatcher:
        return self  # the one call it makes names its tool directly

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            reply = AIMessage("Done.")
        else:
            brief = next(m for m in reversed(messages) if isinstance(m, HumanMessage)).text
            reply = AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"subagent_type": self.subagent, "description": brief},
                        "id": f"dispatch-{uuid.uuid4().hex[:8]}",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


# langchain-anthropic's `input_tokens` already includes both cache counts, so
# uncached input is `input_tokens - cache_read - cache_creation`.
USAGE_FIELDS = ("model_calls", "input_tokens", "cache_read", "cache_creation", "output_tokens")


def _add_usage(totals: dict, message: AIMessage) -> None:
    usage = message.usage_metadata
    if not usage:  # scripted models report none
        return
    details = usage.get("input_token_details") or {}
    totals["model_calls"] += 1
    totals["input_tokens"] += usage["input_tokens"]
    totals["cache_read"] += details.get("cache_read") or 0
    # When the API breaks cache writes down by TTL, langchain-anthropic reports
    # them under these keys and sets `cache_creation` to 0. Reading
    # `cache_creation` alone reported no writes on every real run.
    totals["cache_creation"] += sum(
        details.get(key) or 0
        for key in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    totals["output_tokens"] += usage["output_tokens"]


def run_subagent(subagent: str, inputs: dict) -> dict:
    """Run one subagent on a brief and return what the evaluators read.

    Returns `response` (the subagent's closing text, which is what the main
    agent would receive), `tool_calls` (every call the subagent made, in
    order), `files` (the workspace it left behind) and `usage` (the
    subagent's summed token counts, for pricing a run from measurement rather
    than from assumed reply lengths).
    """
    # Deferred like main.py and ui.py: building the agent reads the environment.
    from travel_agent.agent import build_agent

    agent = build_agent(
        Dispatcher(subagent=subagent), checkpointer=MemorySaver(), store=InMemoryStore()
    )
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}}
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    response = ""
    usage = dict.fromkeys(USAGE_FIELDS, 0)
    # A deep copy, because the graph coerces message dicts into message objects
    # in place, and `evaluate()` passes this same `inputs` on to every
    # evaluator. Without it, `rubric` reads a `HumanMessage` where it expects
    # the example's dict.
    # Consumed whole inside the script's context: tools run while the stream
    # is read, and an example with no script runs exactly as before.
    with scripted.active(inputs):
        stream = list(
            agent.stream(copy.deepcopy(inputs), config, stream_mode="updates", subgraphs=True)
        )
    for namespace, update in stream:
        for node, value in (update or {}).items():
            messages = value.get("messages", []) if isinstance(value, dict) else []
            for message in messages if isinstance(messages, list) else [messages]:
                if namespace and node == "model" and isinstance(message, AIMessage):
                    tool_calls += [
                        {"name": c["name"], "args": c["args"]} for c in message.tool_calls
                    ]
                    _add_usage(usage, message)
                elif namespace and isinstance(message, ToolMessage):
                    tool_results.append({"name": message.name, "content": message.text})
                elif not namespace and isinstance(message, ToolMessage):
                    response = message.text
    files = agent.get_state(config).values.get("files", {})
    return {
        "response": response,
        "tool_calls": tool_calls,
        # What the tools returned, so the judge can check a claim against the
        # data behind it. Without these it saw a stay called non-refundable,
        # could not see `free_cancellation: false`, and failed a correct run.
        "tool_results": tool_results,
        "files": {path: file_data_to_string(data) for path, data in files.items()},
        "usage": usage,
    }


def run_agent(inputs: dict) -> dict:
    """Run the whole agent on a request and return what the evaluators read.

    `tool_calls` holds every call at every level, in order. Each is tagged
    with `agent` ("main" or "subagent") and `step`, the main agent's model turn
    it belongs to; a subagent's calls carry the turn that delegated to it.
    Trajectory evaluators read only the main agent's calls, and order them by
    `step`: calls made in one turn run concurrently, so position within a turn
    says nothing about what one call could see of another's result.

    `response` is the main agent's last reply. `usage` is keyed by model,
    since a full trip spans three of them at different prices.
    """
    from travel_agent.agent import build_agent

    agent = build_agent(
        checkpointer=MemorySaver(),
        store=InMemoryStore(),
        search_tools=scripted.search_tools_for(inputs),
    )
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}}
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    response = ""
    usage: dict[str, dict] = {}
    step = 0
    # Consumed whole inside the script's context: tools run while the stream
    # is read, and an example with no script runs exactly as before.
    with scripted.active(inputs):
        stream = list(
            agent.stream(copy.deepcopy(inputs), config, stream_mode="updates", subgraphs=True)
        )
    for namespace, update in stream:
        level = "subagent" if namespace else "main"
        for node, value in (update or {}).items():
            messages = value.get("messages", []) if isinstance(value, dict) else []
            for message in messages if isinstance(messages, list) else [messages]:
                if node == "model" and isinstance(message, AIMessage):
                    if not namespace:
                        step += 1
                        response = message.text
                    tool_calls += [
                        {"agent": level, "step": step, "name": c["name"], "args": c["args"]}
                        for c in message.tool_calls
                    ]
                    if message.usage_metadata:
                        model = message.response_metadata.get("model_name", "unknown")
                        _add_usage(usage.setdefault(model, dict.fromkeys(USAGE_FIELDS, 0)), message)
                elif isinstance(message, ToolMessage):
                    tool_results.append(
                        {"agent": level, "name": message.name, "content": message.text}
                    )
    files = agent.get_state(config).values.get("files", {})
    return {
        "subject": "agent",  # how the judge refers to what it is grading
        "response": response,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "files": {path: file_data_to_string(data) for path, data in files.items()},
        "usage": usage,
    }


_agent_runs: dict[str, dict] = {}


def run_agent_shared(inputs: dict) -> dict:
    """`run_agent`, run once per distinct request for the life of the process.

    Datasets are evaluated one after another, so a request the first one ran
    is never in flight when the second asks for it. A reused run is marked, so
    the summary does not read its tokens as a second spend.
    """
    key = json.dumps(inputs, sort_keys=True)
    if key in _agent_runs:
        return copy.deepcopy(_agent_runs[key]) | {"reused": True}
    outputs = run_agent(inputs)
    _agent_runs[key] = outputs
    return copy.deepcopy(outputs)


# --- trajectory ----------------------------------------------------------------


def _main_calls(outputs: dict) -> list[dict]:
    return [c for c in outputs["tool_calls"] if c["agent"] == "main"]


def _delegations(outputs: dict) -> list[tuple[int, str]]:
    """(step, subagent) for each `task` call the main agent made."""
    return [
        (c["step"], c["args"].get("subagent_type", ""))
        for c in _main_calls(outputs)
        if c["name"] == "task"
    ]


def _verdict(problems: list[str], passed: str) -> dict:
    return {"score": int(not problems), "comment": "; ".join(problems) or passed}


def delegation(outputs: dict, reference_outputs: dict) -> dict:
    """Every required subagent was delegated to, and no forbidden one."""
    used = {name for _, name in _delegations(outputs)}
    problems = []
    if missing := sorted(set(reference_outputs["required_subagents"]) - used):
        problems.append(f"never delegated to {missing}")
    if forbidden := sorted(set(reference_outputs["forbidden_subagents"]) & used):
        problems.append(f"delegated to forbidden {forbidden}")
    return _verdict(problems, f"delegated to {sorted(used)}")


def delegation_order(outputs: dict, reference_outputs: dict) -> dict:
    """`ordered_subagents` happened in order, and each `required_any_of` was met.

    "After" means a later model turn. A scout and an analyst briefed in the
    same turn run side by side, so the analyst never saw the scout's prices,
    however the two calls are listed.
    """
    ordered, any_of = reference_outputs["ordered_subagents"], reference_outputs["required_any_of"]
    if not ordered and not any_of:
        return {"score": None, "comment": "no ordering to check"}
    delegations, problems = _delegations(outputs), []
    last = 0
    for name in ordered:
        steps = [s for s, n in delegations if n == name and s > last]
        if not steps:
            problems.append(f"{name} not delegated after turn {last}")
            break
        last = min(steps)
    for group in any_of:
        subagents, tools = group.get("subagents", []), group.get("tools", [])
        options = " or ".join([*subagents, *tools])
        after = group.get("after_subagent")
        start = min((s for s, n in delegations if n == after), default=None) if after else 0
        if start is None:
            problems.append(f"{after} never delegated, so {options} cannot follow it")
            continue
        met = any(
            c["step"] > start
            and (
                c["name"] in tools
                or (c["name"] == "task" and c["args"].get("subagent_type") in subagents)
            )
            for c in _main_calls(outputs)
        )
        if not met:
            problems.append(f"no {options}" + (f" in a turn after {after}" if after else ""))
    return _verdict(problems, "in order")


def tool_use(outputs: dict, reference_outputs: dict) -> dict:
    """The main agent called every required tool itself, and no forbidden one."""
    used = {c["name"] for c in _main_calls(outputs)}
    problems = []
    if missing := sorted(set(reference_outputs["required_tools"]) - used):
        problems.append(f"never called {missing}")
    if forbidden := sorted(set(reference_outputs["forbidden_tools"]) & used):
        problems.append(f"called forbidden {forbidden}")
    return _verdict(problems, f"called {sorted(used)}")


def file_access(outputs: dict, reference_outputs: dict) -> dict:
    """Required files were written and forbidden ones not read, by the main agent.

    Read off the calls rather than the final state: `/memories/` routes to the
    store, so a profile write never appears in the state's `files`.
    """
    calls = _main_calls(outputs)

    def paths(*names: str) -> set:
        return {c["args"].get("file_path") for c in calls if c["name"] in names}

    problems = []
    written = paths("write_file", "edit_file")
    if missing := sorted(set(reference_outputs["required_file_writes"]) - written):
        problems.append(f"never wrote {missing}")
    if read := sorted(set(reference_outputs["forbidden_file_reads"]) & paths("read_file")):
        problems.append(f"read {read}")
    return _verdict(problems, f"wrote {sorted(p for p in written if p)}")


_SEARCH_DATE_FIELDS = ("depart_date", "return_date", "check_in", "check_out")


def searches_not_in_past(outputs: dict, reference_outputs: dict) -> dict:
    """No search, at any level, asked for a date that had already passed.

    The one trajectory check that reads subagent calls. The scout searches the
    dates its brief gives it, so a past-dated search is the main agent's brief
    showing through, and the tool answers it only with an error. Substituting
    future dates is fine; a real run did that, and v2 failed it for delegating.
    """
    from travel_agent.clock import today

    searches = [c for c in outputs["tool_calls"] if c["name"] in {"search_flights", "search_stays"}]
    if not searches:
        return {"score": None, "comment": "no searches"}
    past = set()
    for call in searches:
        for field in _SEARCH_DATE_FIELDS:
            try:
                if date.fromisoformat(str(call["args"].get(field))) < today():
                    past.add(str(call["args"][field]))
            except ValueError:  # absent or malformed; the tool rejects the latter itself
                continue
    return _verdict([f"searched past dates {sorted(past)}"] if past else [], "all dates ahead")


# --- availability-scout ------------------------------------------------------


def _scout_tools() -> dict:
    from travel_agent.tools.availability import search_flights, search_stays

    return {tool.name: tool for tool in (search_flights, search_stays)}


def _same(field: str, expected: Any, actual: Any) -> bool:
    """Compare one search argument the way the tool itself would read it."""
    from travel_agent.tools.availability import _normalize_cabin

    if expected is None:
        return actual is None
    if actual is None:
        return False
    if field == "cabin":
        try:
            return _normalize_cabin(str(actual)) == expected
        except ValueError:
            return False
    if field in {"origin", "destination"}:
        return str(actual).strip().upper() == expected
    if field == "location":  # "Kyoto, Japan" is still a search for Kyoto
        return expected.casefold() in str(actual).casefold()
    if field in {"travelers", "guests", "max_nightly_rate"}:
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _with_defaults(name: str, args: dict) -> dict:
    """A call's arguments with the tool's own defaults filled in.

    Leaving `cabin` out is a search for economy; the reference spells it.
    """
    schema = _scout_tools()[name].args
    return {field: args.get(field, spec.get("default")) for field, spec in schema.items()}


def _expected_calls(reference_outputs: dict) -> list[tuple[str, dict]]:
    """Each expected call as (tool, arguments).

    A tool maps to one argument set, or to a list when the brief needs several
    searches of one kind, such as each leg of an open-jaw trip. An argument set
    may be partial: only the fields it names are checked.
    """
    pairs = []
    for name, expected in reference_outputs.get("expected_calls", {}).items():
        pairs += [(name, e) for e in (expected if isinstance(expected, list) else [expected])]
    return pairs


def search_arguments(outputs: dict, reference_outputs: dict) -> dict:
    """Share of expected searches that some call made with every briefed argument.

    Any call, not the first: the scout is told to vary its searches, and the
    order is its own. A real run searched lodging from the arrival day first,
    since a flight that crosses the date line lands the next day, and only
    then searched the briefed dates. Reading only the first call failed it for
    being more careful than the brief.
    """
    expected_calls = _expected_calls(reference_outputs)
    if not expected_calls:
        return {"score": None, "comment": "no searches expected"}
    misses = []
    for name, expected in expected_calls:
        calls = [
            _with_defaults(name, c["args"]) for c in outputs["tool_calls"] if c["name"] == name
        ]
        if not calls:
            misses.append(f"{name} never called")
            continue
        wrong_per_call = [
            [f for f, value in expected.items() if not _same(f, value, actual.get(f))]
            for actual in calls
        ]
        if all(wrong_per_call):  # no call matched; report the nearest miss
            nearest, actual = min(zip(wrong_per_call, calls, strict=True), key=lambda p: len(p[0]))
            fields = ", ".join(f"{f}={actual.get(f)!r}" for f in nearest)
            wanted = ", ".join(f"{f}={expected[f]!r}" for f in nearest)
            misses.append(f"{name}: no call had {wanted}; nearest had {fields}")
    score = (len(expected_calls) - len(misses)) / len(expected_calls)
    return {"score": score, "comment": "; ".join(misses) or "all briefed arguments sent"}


def cabin_as_briefed(outputs: dict, reference_outputs: dict) -> dict:
    """Every flight search used the briefed cabin, including the varied retries.

    Separate from `search_arguments`, which passes once any call matches the
    brief: the prompt lets the scout vary airports and dates, never the cabin,
    so here a single off-brief cabin fails.
    """
    briefed = {
        e["cabin"]
        for name, e in _expected_calls(reference_outputs)
        if name == "search_flights" and "cabin" in e
    }
    if not briefed:
        return {"score": None, "comment": "no flight cabin briefed"}
    cabins = [
        _with_defaults("search_flights", c["args"])["cabin"]
        for c in outputs["tool_calls"]
        if c["name"] == "search_flights"
    ]
    if not cabins:
        return {"score": 0, "comment": "search_flights never called"}
    wrong = [c for c in cabins if not any(_same("cabin", b, c) for b in briefed)]
    return {"score": int(not wrong), "comment": f"searched {cabins}"}


def forbidden_arguments(outputs: dict, reference_outputs: dict) -> dict:
    """No call passed a value the brief rules out, at any level.

    For a figure that must be converted before it reaches a tool, such as a
    nightly ceiling briefed in yen: `max_nightly_rate` is USD, so ¥20,000
    passed as is searches for twenty-thousand-dollar rooms.
    """
    forbidden = reference_outputs.get("forbidden_arguments", {})
    if not forbidden:
        return {"score": None, "comment": "no values ruled out"}
    hits = [
        f"{c['name']}({field}={c['args'][field]!r})"
        for c in outputs["tool_calls"]
        for field, values in forbidden.get(c["name"], {}).items()
        if c["args"].get(field) is not None
        and any(_same(field, value, c["args"][field]) for value in values)
    ]
    return _verdict([f"passed {hits}"] if hits else [], "none passed")


def forbidden_tools_unused(outputs: dict, reference_outputs: dict) -> dict:
    """The subagent did not call a tool the brief ruled out."""
    forbidden = set(reference_outputs.get("forbidden_tools", []))
    used = sorted({c["name"] for c in outputs["tool_calls"]} & forbidden)
    return {"score": int(not used), "comment": f"called {used}" if used else "none called"}


def refreshed_expiring_offer(outputs: dict, reference_outputs: dict) -> dict:
    """A search that came back seconds from expiry was run again.

    The prompt says to re-search rather than hand back an offer that has
    expired or is about to, and the only evidence of that is a second call.
    """
    expiring = reference_outputs.get("expiring_search")
    if not expiring:
        return {"score": None, "comment": "no expiring offer in this example"}
    name, match = expiring["name"], expiring["match"]
    runs = [
        c
        for c in outputs["tool_calls"]
        if c["name"] == name and all(_same(f, v, c["args"].get(f)) for f, v in match.items())
    ]
    return _verdict([] if len(runs) >= 2 else [f"searched {len(runs)} time(s)"], "searched again")


# --- budget-analyst ----------------------------------------------------------


def _last_budget_call(outputs: dict) -> dict | None:
    calls = [c for c in outputs["tool_calls"] if c["name"] == "summarize_budget"]
    return calls[-1]["args"] if calls else None


def budget_total(outputs: dict, reference_outputs: dict) -> dict:
    """The analyst's final `summarize_budget` call totals to the reference.

    Re-runs the real tool on the analyst's own arguments rather than reading
    its prose, so a correct sentence over a wrong call still scores 0. A
    reference `budget_total` of null expects a call with no budget at all.
    """
    from travel_agent.tools.budget import summarize_budget

    if reference_outputs["expected_total"] is None:
        return {"score": None, "comment": "the total depends on an exchange rate the analyst picks"}
    args = _last_budget_call(outputs)
    if args is None:
        return {"score": 0, "comment": "summarize_budget never called"}
    try:
        result = summarize_budget.invoke(args)
    except (TypeError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
        return {"score": 0, "comment": f"call rejected: {exc}"}
    if "error" in result:
        return {"score": 0, "comment": result["error"]}
    ok = (
        result["total_estimated"] == reference_outputs["expected_total"]
        and result["budget_total"] == reference_outputs["budget_total"]
    )
    return {
        "score": int(ok),
        "comment": f"total {result['total_estimated']} against {result['budget_total']}; "
        f"expected {reference_outputs['expected_total']} against "
        f"{reference_outputs['budget_total']}",
    }


def budget_currency(outputs: dict, reference_outputs: dict) -> dict:
    """The analyst passed the trip's currency rather than the USD default."""
    args = _last_budget_call(outputs)
    if args is None:
        return {"score": 0, "comment": "summarize_budget never called"}
    currency = str(args.get("currency", "USD")).strip().upper()
    return {"score": int(currency == reference_outputs["currency"]), "comment": currency}


def due_at_accommodation_excluded(outputs: dict, reference_outputs: dict) -> dict:
    """An amount the brief said may sit outside the total is not a cost line.

    Catches the separate line, per unit or as a subtotal. Folding the amount
    into the lodging line instead moves the total, which `budget_total` scores.
    """
    excluded = reference_outputs.get("excluded_amounts", [])
    if not excluded:
        return {"score": None, "comment": "nothing to exclude in this example"}
    args = _last_budget_call(outputs)
    if args is None:
        return {"score": 0, "comment": "summarize_budget never called"}
    costed = _lines_costing(args, excluded)
    return {"score": int(not costed), "comment": f"costed {costed}" if costed else "kept out"}


def _lines_costing(args: dict, amounts: list) -> list[dict]:
    """Items whose amount, per unit or as a subtotal, is one of `amounts`."""
    targets = {float(x) for x in amounts}
    lines = []
    for item in args.get("items", []):
        amount, quantity = float(item.get("amount", 0)), float(item.get("quantity", 1))
        if {amount, amount * quantity} & targets:
            lines.append(item)
    return lines


def foreign_amounts_converted(outputs: dict, reference_outputs: dict) -> dict:
    """A figure briefed in another currency is not costed as if it were the trip's.

    The rate is the analyst's to pick, so the total cannot be scored exactly;
    what can be is the raw foreign figure turning up unchanged.
    """
    unconverted = reference_outputs.get("unconverted_amounts", [])
    if not unconverted:
        return {"score": None, "comment": "no foreign amounts in this example"}
    args = _last_budget_call(outputs)
    if args is None:
        return {"score": 0, "comment": "summarize_budget never called"}
    raw = _lines_costing(args, unconverted)
    return _verdict([f"costed unconverted {raw}"] if raw else [], "converted")


def booked_lines_confirmed(outputs: dict, reference_outputs: dict) -> dict:
    """Costs the brief says are already paid go in as `estimated: false`.

    That flag is what tells the traveler which part of the total is settled,
    and which lines a cut cannot touch.
    """
    categories = set(reference_outputs.get("confirmed_categories", []))
    if not categories:
        return {"score": None, "comment": "nothing already paid in this example"}
    args = _last_budget_call(outputs)
    if args is None:
        return {"score": 0, "comment": "summarize_budget never called"}
    lines = [i for i in args.get("items", []) if i.get("category") in categories]
    if not lines:
        return {"score": 0, "comment": f"no {sorted(categories)} line costed"}
    estimated = [i.get("label") for i in lines if i.get("estimated", True)]
    return _verdict([f"left estimated: {estimated}"] if estimated else [], "marked confirmed")


def budget_not_invented(outputs: dict, reference_outputs: dict) -> dict:
    """With no budget given, no `summarize_budget` call, at any level, supplies one.

    Read from the calls, since a reply can avoid naming a figure the plan was
    still measured against.
    """
    if reference_outputs.get("budget_given", True):
        return {"score": None, "comment": "a budget was given"}
    calls = [c for c in outputs["tool_calls"] if c["name"] == "summarize_budget"]
    if not calls:
        return {"score": None, "comment": "summarize_budget never called"}
    invented = sorted({c["args"]["budget_total"] for c in calls if c["args"].get("budget_total")})
    return _verdict([f"costed against {invented}"] if invented else [], "no budget supplied")


# --- rubric judge ------------------------------------------------------------


class CriterionVerdict(BaseModel):
    criterion: str = Field(description="The criterion, copied verbatim.")
    reasoning: str = Field(description="One or two sentences citing the transcript.")
    met: bool


class RubricGrade(BaseModel):
    verdicts: list[CriterionVerdict]


JUDGE_PROMPT = """\
You are grading one run of a travel-planning {subject} against a rubric.

Judge each criterion independently and only on the evidence below. A criterion
about something the {subject} said is met only if its response says it; a
criterion about a tool call is met only if the tool calls show it. Check what
the {subject} claims about the data against the tool results it was given: a
claim the results support is correct, even if a criterion warns against making
it without support. Do not give credit for intent. When the evidence is silent,
the criterion is not met.

<brief>
{brief}
</brief>

<tool_calls>
{tool_calls}
</tool_calls>

<tool_results>
{tool_results}
</tool_results>

<files_written>
{files}
</files_written>

<response>
{response}
</response>

<criteria>
{criteria}
</criteria>

Return one verdict per criterion, in the order given.
"""


@cache
def _judge():
    from langchain.chat_models import init_chat_model

    from travel_agent.config import DEFAULT_MAX_TOKENS

    model = init_chat_model(
        JUDGE_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        reasoning_effort=JUDGE_EFFORT,
        betas=[_JUDGE_FALLBACK_BETA],
        model_kwargs={"fallbacks": "default"},
    )
    return model.with_structured_output(RubricGrade, method="json_schema")


def rubric(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Share of the example's rubric criteria an LLM judge finds met."""
    criteria = reference_outputs.get("criteria", [])
    if not criteria:
        return {"score": None, "comment": "no criteria"}
    prompt = JUDGE_PROMPT.format(
        # "subagent" unless the run says otherwise, so the subagent datasets
        # keep grading on the exact prompt their earlier experiments used.
        subject=outputs.get("subject", "subagent"),
        brief=inputs["messages"][-1]["content"],
        tool_calls=json.dumps(outputs["tool_calls"], indent=1, default=str),
        tool_results="\n\n".join(
            f"## {r['name']}\n{r['content']}" for r in outputs.get("tool_results", [])
        )
        or "(none)",
        files="\n\n".join(f"## {p}\n{t}" for p, t in outputs["files"].items()) or "(none)",
        response=outputs["response"] or "(empty)",
        criteria="\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1)),
    )
    grade = _judge().invoke(prompt)
    if not isinstance(grade, RubricGrade) or len(grade.verdicts) != len(criteria):
        return {"score": None, "comment": f"judge returned an unusable grade: {grade!r}"}
    met = sum(v.met for v in grade.verdicts)
    failed = [f"✗ {v.criterion}: {v.reasoning}" for v in grade.verdicts if not v.met]
    return {"score": met / len(criteria), "comment": "\n".join(failed) or "all criteria met"}


# Evaluators take some of `inputs`, `outputs` and `reference_outputs`, by name.
EVALUATORS: dict[str, list[Callable[..., dict]]] = {
    "availability_scout": [search_arguments, cabin_as_briefed, forbidden_tools_unused, rubric],
    "budget_analyst": [budget_total, budget_currency, due_at_accommodation_excluded, rubric],
    "trajectory": [delegation, delegation_order, tool_use, file_access, searches_not_in_past],
    "final_response": [rubric],
    # The hard sets add checks the regression sets do not carry, so the
    # regression experiments keep the columns they were first scored with.
    "availability_scout_hard": [
        search_arguments,
        cabin_as_briefed,
        forbidden_tools_unused,
        forbidden_arguments,
        rubric,
    ],
    "budget_analyst_hard": [
        budget_total,
        budget_currency,
        foreign_amounts_converted,
        booked_lines_confirmed,
        budget_not_invented,
        rubric,
    ],
    "availability_scout_scripted": [
        search_arguments,
        cabin_as_briefed,
        forbidden_tools_unused,
        refreshed_expiring_offer,
        rubric,
    ],
    "final_response_scripted": [search_arguments, searches_not_in_past, rubric],
    "final_response_hard": [
        search_arguments,
        forbidden_arguments,
        searches_not_in_past,
        budget_not_invented,
        rubric,
    ],
}


def _usage_lines(outputs: dict) -> list[str]:
    """Token counts: one line for a subagent run, one per model for a whole trip."""
    if outputs.get("reused"):
        return ["same run as an earlier dataset in this process; tokens counted there"]
    usage = outputs.get("usage", {})

    def counts(totals: dict) -> str:
        return " ".join(f"{field}={totals.get(field, 0)}" for field in USAGE_FIELDS)

    if usage and all(isinstance(v, dict) for v in usage.values()):
        return [f"{model}: {counts(totals)}" for model, totals in sorted(usage.items())]
    return [counts(usage)]


def summarize(rows: Iterable[Any]) -> list[str]:
    """Per-example scores and token usage, for the terminal.

    The experiment page has the same scores, but token counts are what turn a
    cost estimate into a measurement, so they are printed alongside. Any score
    below 1 prints its comment too: a bare 0.5 cannot tell a wrong agent from a
    reference no run could meet, and telling those apart is the point.
    """
    lines = []
    for row in rows:
        key = (row["example"].metadata or {}).get("key", row["example"].id)
        run = row["run"]
        if run.error or not run.outputs:
            lines.append(f"  {key}: run failed: {run.error}")
            continue
        scores = ", ".join(
            f"{r.key}={'n/a' if r.score is None else round(r.score, 2)}"
            for r in row["evaluation_results"]["results"]
        )
        lines.append(f"  {key}: {scores}\n    " + "\n    ".join(_usage_lines(run.outputs)))
        for r in row["evaluation_results"]["results"]:
            if r.score is not None and r.score < 1 and r.comment:
                lines.append(f"    {r.key}: " + r.comment.replace("\n", "\n      "))
    return lines


def pass_rates(rows: Iterable[Any]) -> list[str]:
    """Mean score per example and evaluator across repetitions.

    One run is a sample, not a rate: the hard sets exist to move, and a single
    pass or fail cannot show which way. Empty when nothing ran twice.
    """
    scores: dict[str, dict[str, list[float]]] = {}
    runs: dict[str, int] = {}
    for row in rows:
        key = (row["example"].metadata or {}).get("key", row["example"].id)
        runs[key] = runs.get(key, 0) + 1
        for r in row["evaluation_results"]["results"]:
            if r.score is not None:
                scores.setdefault(key, {}).setdefault(r.key, []).append(r.score)
    if max(runs.values(), default=0) < 2:
        return []
    lines = ["  pass rates:"]
    for key, count in runs.items():
        means = ", ".join(
            f"{name}={sum(values) / len(values):.2f} ({len(values)})"
            for name, values in scores.get(key, {}).items()
        )
        lines.append(f"    {key} over {count} runs: {means or 'nothing scored'}")
    return lines


def agent_target(repetitions: int) -> Callable[[dict], dict]:
    """The whole-agent run function for an evaluation.

    Sharing a run across datasets is only sound for one repetition: repeated,
    every run of an example would be served the first one's transcript, and
    three identical scores would read as a stable pass rate.
    """
    return run_agent_shared if repetitions == 1 else run_agent


def _key(example: Any) -> str:
    return (example.metadata or {}).get("key", str(example.id))


def unknown_keys(stems: list[str], keys: list[str]) -> list[str]:
    """Requested example keys that no selected dataset holds.

    Checked against the local files, before anything runs: a typo would
    otherwise evaluate zero examples, or every dataset but the one meant.
    """
    datasets = load_datasets()
    known = {example["key"] for stem in stems for example in datasets[stem]["examples"]}
    return sorted(set(keys) - known)


def apply_run_environment(*, no_web_search: bool) -> bool:
    """Pin the provider and, if asked, drop web search; return whether search is on.

    Called once the dotenv file has loaded. Dropping the key rather than
    leaving it set is the point: with no Tavily credits the tool returns an
    error the agent quietly works around, so a run would measure a broken
    search while its metadata said search was on.
    """
    os.environ["TRAVEL_AGENT_PROVIDER"] = "sample-data"
    if no_web_search:
        os.environ.pop("TAVILY_API_KEY", None)
    return bool(os.getenv("TAVILY_API_KEY"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", choices=sorted([*SUBAGENT_FOR, *AGENT_DATASETS]))
    parser.add_argument("--limit", type=int, help="Run only the first N examples of each.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM rubric judge.")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Run each example N times, for pass rates."
    )
    parser.add_argument(
        "--keys", nargs="+", metavar="KEY", help="Run only the examples with these keys."
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Run without Tavily even if its key is set; research examples are skipped.",
    )
    args = parser.parse_args(argv)
    if args.no_judge and "final_response" in args.datasets:
        # The judge is its only evaluator: every trip would be paid for and none scored.
        parser.error("final_response is scored only by the judge; drop --no-judge")
    if args.keys and (unknown := unknown_keys(args.datasets, args.keys)):
        parser.error(f"no example {unknown} in {args.datasets}")

    from dotenv import load_dotenv

    load_dotenv()
    has_web_search = apply_run_environment(no_web_search=args.no_web_search)

    from langsmith import Client, evaluate

    from travel_agent.config import DEFAULT_EFFORT, DEFAULT_MODEL, SUBAGENT_MODELS

    client = Client()
    datasets = load_datasets()
    for stem in args.datasets:
        name = datasets[stem]["name"]
        evaluators = [e for e in EVALUATORS[stem] if not (args.no_judge and e is rubric)]
        examples = list(client.list_examples(dataset_name=name, limit=args.limit))
        if args.keys:
            examples = [e for e in examples if _key(e) in args.keys]
            if not examples:
                print(f"{name}: none of {args.keys} here; skipped")
                continue
        metadata: dict[str, Any] = {
            "judge": None if args.no_judge else JUDGE_MODEL,
            "provider": scripted.PROVIDER if stem in SCRIPTED_DATASETS else "sample-data",
            "repetitions": args.repetitions,
        }
        # Per dataset, since datasets run one after another. The scripted
        # provider serves sample data wherever a script is silent.
        if stem in SCRIPTED_DATASETS:
            os.environ["TRAVEL_AGENT_PROVIDER"] = scripted.PROVIDER
            print(f"{name}: {scripted.register(examples)} scripted examples")
        else:
            os.environ["TRAVEL_AGENT_PROVIDER"] = "sample-data"
        if stem in SUBAGENT_FOR:
            subagent = SUBAGENT_FOR[stem]
            print(f"{name}: {len(examples)} examples on {SUBAGENT_MODELS[subagent].model}")
            metadata |= {
                "subagent": subagent,
                "model": SUBAGENT_MODELS[subagent].model,
                "effort": SUBAGENT_MODELS[subagent].effort,
            }

            def target(inputs: dict, subagent: str = subagent) -> dict:
                return run_subagent(subagent, inputs)

        else:
            # Without Tavily the researcher has no tools, so an example that
            # needs it measures the missing key, not the agent.
            if not has_web_search:
                needs = [e for e in examples if (e.metadata or {}).get("requires_web_search")]
                if needs:
                    print(f"{name}: skipping {[_key(e) for e in needs]}: web search is off")
                    examples = [e for e in examples if e not in needs]
            print(f"{name}: {len(examples)} examples on the whole agent ({DEFAULT_MODEL})")
            metadata |= {
                "model": DEFAULT_MODEL,
                "effort": DEFAULT_EFFORT,
                "subagent_models": {n: s.model for n, s in SUBAGENT_MODELS.items()},
                "web_search": has_web_search,
            }
            target = agent_target(args.repetitions)

        results = evaluate(
            target,
            data=examples,
            evaluators=evaluators,
            experiment_prefix=stem,
            max_concurrency=args.concurrency,
            num_repetitions=args.repetitions,
            metadata=metadata,
        )
        print(f"  experiment: {results.experiment_name}")
        rows = list(results)  # read twice below
        print(*summarize(rows), *pass_rates(rows), sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

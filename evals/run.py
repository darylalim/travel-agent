"""Run the subagent datasets against the real models and score them in LangSmith.

    uv run python -m evals.run budget_analyst --limit 1 --no-judge   # cheapest smoke test
    uv run python -m evals.run budget_analyst availability_scout

Each example runs one subagent on its real model, through the real graph. A
scripted `Dispatcher` takes the main agent's seat: it makes a single `task` call
carrying the example's brief and then stops. That way the subagent gets exactly
what production gives it, with deepagents' own middleware stack, the filesystem
and `CurrentDateMiddleware`. A subagent rebuilt by hand would drift from that,
and a drifted harness scores a stack nobody runs.

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
from collections.abc import Iterable
from functools import cache
from typing import Any

from deepagents.backends.utils import file_data_to_string
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from evals.upload import load_datasets

# Dataset file stem -> the subagent it scores. Names are checked against the
# roster in tests, since a stale one would dispatch to a subagent that is gone.
SUBAGENT_FOR = {
    "availability_scout": "availability-scout",
    "budget_analyst": "budget-analyst",
}

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
    response = ""
    usage = dict.fromkeys(USAGE_FIELDS, 0)
    # A deep copy, because the graph coerces message dicts into message objects
    # in place, and `evaluate()` passes this same `inputs` on to every
    # evaluator. Without it, `rubric` reads a `HumanMessage` where it expects
    # the example's dict.
    stream = agent.stream(copy.deepcopy(inputs), config, stream_mode="updates", subgraphs=True)
    for namespace, update in stream:
        for node, value in (update or {}).items():
            messages = value.get("messages", []) if isinstance(value, dict) else []
            for message in messages if isinstance(messages, list) else [messages]:
                if namespace and node == "model" and isinstance(message, AIMessage):
                    tool_calls += [
                        {"name": c["name"], "args": c["args"]} for c in message.tool_calls
                    ]
                    _add_usage(usage, message)
                elif not namespace and isinstance(message, ToolMessage):
                    response = message.text
    files = agent.get_state(config).values.get("files", {})
    return {
        "response": response,
        "tool_calls": tool_calls,
        "files": {path: file_data_to_string(data) for path, data in files.items()},
        "usage": usage,
    }


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


def search_arguments(outputs: dict, reference_outputs: dict) -> dict:
    """Share of expected searches whose first call carried the briefed arguments."""
    expected_calls = reference_outputs["expected_calls"]
    misses = []
    for name, expected in expected_calls.items():
        first = next((c for c in outputs["tool_calls"] if c["name"] == name), None)
        if first is None:
            misses.append(f"{name} never called")
            continue
        actual = _with_defaults(name, first["args"])
        wrong = [f for f, value in expected.items() if not _same(f, value, actual.get(f))]
        if wrong:
            misses.append(f"{name}: {', '.join(f'{f}={actual.get(f)!r}' for f in wrong)}")
    score = (len(expected_calls) - len(misses)) / len(expected_calls)
    return {"score": score, "comment": "; ".join(misses) or "all briefed arguments sent"}


def cabin_as_briefed(outputs: dict, reference_outputs: dict) -> dict:
    """Every flight search used the briefed cabin, including the varied retries.

    Separate from `search_arguments`, which only reads the first call: the
    prompt lets the scout vary airports and dates, never the cabin.
    """
    expected = reference_outputs["expected_calls"].get("search_flights")
    if expected is None:
        return {"score": None, "comment": "no flight search expected"}
    cabins = [
        _with_defaults("search_flights", c["args"])["cabin"]
        for c in outputs["tool_calls"]
        if c["name"] == "search_flights"
    ]
    if not cabins:
        return {"score": 0, "comment": "search_flights never called"}
    wrong = [c for c in cabins if not _same("cabin", expected["cabin"], c)]
    return {"score": int(not wrong), "comment": f"searched {cabins}"}


def forbidden_tools_unused(outputs: dict, reference_outputs: dict) -> dict:
    """The subagent did not call a tool the brief ruled out."""
    forbidden = set(reference_outputs.get("forbidden_tools", []))
    used = sorted({c["name"] for c in outputs["tool_calls"]} & forbidden)
    return {"score": int(not used), "comment": f"called {used}" if used else "none called"}


# --- budget-analyst ----------------------------------------------------------


def _last_budget_call(outputs: dict) -> dict | None:
    calls = [c for c in outputs["tool_calls"] if c["name"] == "summarize_budget"]
    return calls[-1]["args"] if calls else None


def budget_total(outputs: dict, reference_outputs: dict) -> dict:
    """The analyst's final `summarize_budget` call totals to the reference.

    Re-runs the real tool on the analyst's own arguments rather than reading
    its prose, so a correct sentence over a wrong call still scores 0.
    """
    from travel_agent.tools.budget import summarize_budget

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
    excluded = {float(x) for x in reference_outputs.get("excluded_amounts", [])}
    if not excluded:
        return {"score": None, "comment": "nothing to exclude in this example"}
    args = _last_budget_call(outputs)
    if args is None:
        return {"score": 0, "comment": "summarize_budget never called"}
    costed = []
    for item in args.get("items", []):
        amount, quantity = float(item.get("amount", 0)), float(item.get("quantity", 1))
        if {amount, amount * quantity} & excluded:
            costed.append(item)
    return {"score": int(not costed), "comment": f"costed {costed}" if costed else "kept out"}


# --- rubric judge ------------------------------------------------------------


class CriterionVerdict(BaseModel):
    criterion: str = Field(description="The criterion, copied verbatim.")
    reasoning: str = Field(description="One or two sentences citing the transcript.")
    met: bool


class RubricGrade(BaseModel):
    verdicts: list[CriterionVerdict]


JUDGE_PROMPT = """\
You are grading one run of a travel-planning subagent against a rubric.

Judge each criterion independently and only on the evidence below. A criterion
about something the subagent said is met only if its response says it; a
criterion about a tool call is met only if the tool calls show it. Do not give
credit for intent. When the evidence is silent, the criterion is not met.

<brief>
{brief}
</brief>

<tool_calls>
{tool_calls}
</tool_calls>

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
        brief=inputs["messages"][-1]["content"],
        tool_calls=json.dumps(outputs["tool_calls"], indent=1, default=str),
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


EVALUATORS = {
    "availability_scout": [search_arguments, cabin_as_briefed, forbidden_tools_unused, rubric],
    "budget_analyst": [budget_total, budget_currency, due_at_accommodation_excluded, rubric],
}


def summarize(rows: Iterable[Any]) -> list[str]:
    """Per-example scores and token usage, for the terminal.

    The experiment page has the same scores, but token counts are what turn a
    cost estimate into a measurement, so they are printed alongside.
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
        usage = run.outputs.get("usage", {})
        tokens = " ".join(f"{field}={usage.get(field, 0)}" for field in USAGE_FIELDS)
        lines.append(f"  {key}: {scores}\n    {tokens}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", choices=sorted(SUBAGENT_FOR))
    parser.add_argument("--limit", type=int, help="Run only the first N examples of each.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM rubric judge.")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    os.environ["TRAVEL_AGENT_PROVIDER"] = "sample-data"

    from langsmith import Client, evaluate

    from travel_agent.config import SUBAGENT_MODELS

    client = Client()
    datasets = load_datasets()
    for stem in args.datasets:
        subagent, name = SUBAGENT_FOR[stem], datasets[stem]["name"]
        evaluators = [e for e in EVALUATORS[stem] if not (args.no_judge and e is rubric)]
        examples = list(client.list_examples(dataset_name=name, limit=args.limit))
        print(f"{name}: {len(examples)} examples on {SUBAGENT_MODELS[subagent].model}")

        def target(inputs: dict, subagent: str = subagent) -> dict:
            return run_subagent(subagent, inputs)

        results = evaluate(
            target,
            data=examples,
            evaluators=evaluators,
            experiment_prefix=stem,
            max_concurrency=args.concurrency,
            metadata={
                "subagent": subagent,
                "model": SUBAGENT_MODELS[subagent].model,
                "effort": SUBAGENT_MODELS[subagent].effort,
                "judge": None if args.no_judge else JUDGE_MODEL,
                "provider": "sample-data",
            },
        )
        print(f"  experiment: {results.experiment_name}")
        print(*summarize(results), sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

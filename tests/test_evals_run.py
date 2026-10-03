"""The eval harness measures what it claims to.

Every failure here is a wrong score rather than an error: a run function that
loses the subagent's tool calls scores every run 0, and an evaluator that reads
the wrong field scores it 1. Nothing in LangSmith distinguishes either from a
real result, so the harness is driven here with scripted models. No network,
no model calls.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import travel_agent.agent as agent_module
from evals import run as harness
from evals.upload import load_datasets
from travel_agent.subagents import build_subagents


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # the replies already name their tools
        return self


def _replies(*messages: AIMessage) -> Iterator[AIMessage]:
    yield from messages


def _call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c-{name}"}])


@pytest.fixture
def scripted_subagent(monkeypatch: pytest.MonkeyPatch):
    """Make every subagent replay `replies`; the dispatcher stays real."""

    def install(*replies: AIMessage) -> None:
        model = _Scripted(messages=_replies(*replies))
        monkeypatch.setattr(agent_module, "_chat_model", lambda *a, **k: model)

    return install


BRIEF = {"messages": [{"role": "user", "content": "Cost this against $100."}]}


# --- run function ------------------------------------------------------------


def test_the_run_captures_the_subagents_calls_reply_and_files(scripted_subagent):
    items = [{"label": "Food", "category": "food", "amount": 10, "quantity": 2}]
    scripted_subagent(
        _call("summarize_budget", items=items, budget_total=100),
        _call("write_file", file_path="/trip/budget.md", content="# Budget\n20 of 100"),
        AIMessage("Total $20, fits."),
    )
    outputs = harness.run_subagent("budget-analyst", BRIEF)

    # These live only in the subagent's namespace. Reading state after the run
    # would find none of them, so this is the capture working, not a tautology.
    assert [c["name"] for c in outputs["tool_calls"]] == ["summarize_budget", "write_file"]
    assert outputs["tool_calls"][0]["args"]["items"] == items
    assert outputs["response"] == "Total $20, fits."
    assert outputs["files"] == {"/trip/budget.md": "# Budget\n20 of 100"}
    # What the tools returned, from the real tools, for the judge to check against.
    assert [r["name"] for r in outputs["tool_results"]] == ["summarize_budget", "write_file"]
    assert '"total_estimated": 20.0' in outputs["tool_results"][0]["content"]


def _billed(message: AIMessage, total: int, read: int, written: int, out: int) -> AIMessage:
    message.usage_metadata = {
        "input_tokens": total,
        "output_tokens": out,
        "total_tokens": total + out,
        "input_token_details": {"cache_read": read, "cache_creation": written},
    }
    return message


def test_the_run_sums_the_subagents_token_usage(scripted_subagent):
    items = [{"label": "Food", "category": "food", "amount": 10, "quantity": 2}]
    scripted_subagent(
        _billed(_call("summarize_budget", items=items, budget_total=100), 3000, 0, 2800, 120),
        _billed(AIMessage("Total $20, fits."), 3400, 2800, 400, 90),
    )
    usage = harness.run_subagent("budget-analyst", BRIEF)["usage"]
    # The dispatcher is scripted and reports nothing, so only the subagent counts.
    assert usage == {
        "model_calls": 2,
        "input_tokens": 6400,
        "cache_read": 2800,
        "cache_creation": 3200,
        "output_tokens": 210,
    }


def test_cache_writes_reported_by_ttl_are_counted(scripted_subagent):
    # The shape langchain-anthropic produces when the API itemises writes by
    # TTL: `cache_creation` is 0 and the real count sits under the TTL key.
    # Built untyped because langchain-core's `InputTokenDetails` does not
    # declare the TTL keys, which is how reading only `cache_creation` looked
    # complete.
    reply = _billed(AIMessage("Nothing to cost."), 5000, 0, 0, 50)
    assert reply.usage_metadata is not None
    details: Any = {
        "cache_read": 0,
        "cache_creation": 0,
        "ephemeral_5m_input_tokens": 4300,
        "ephemeral_1h_input_tokens": 0,
    }
    reply.usage_metadata["input_token_details"] = details
    scripted_subagent(reply)
    assert harness.run_subagent("budget-analyst", BRIEF)["usage"]["cache_creation"] == 4300


def test_the_summary_prints_scores_and_tokens_and_survives_a_failed_run():
    from types import SimpleNamespace as NS

    from langsmith.evaluation import EvaluationResult

    usage = dict.fromkeys(harness.USAGE_FIELDS, 0) | {"model_calls": 3, "output_tokens": 500}
    ok = {
        "example": NS(metadata={"key": "kyoto-fits"}, id="e1"),
        "run": NS(error=None, outputs={"usage": usage}),
        "evaluation_results": {
            "results": [
                EvaluationResult(key="budget_total", score=1, comment="total 1610"),
                EvaluationResult(key="due_at_accommodation_excluded", score=None),
                EvaluationResult(key="rubric", score=0.5, comment="✗ a: why\n✗ b: why"),
            ]
        },
    }
    failed = {
        "example": NS(metadata={"key": "lisbon-over"}, id="e2"),
        "run": NS(error="RateLimitError", outputs=None),
        "evaluation_results": {"results": []},
    }
    lines = harness.summarize([ok, failed])
    assert "kyoto-fits: budget_total=1, due_at_accommodation_excluded=n/a, rubric=0.5" in lines[0]
    assert "model_calls=3" in lines[0] and "output_tokens=500" in lines[0]
    # Only the imperfect score explains itself; a passing comment is noise.
    assert lines[1] == "    rubric: ✗ a: why\n      ✗ b: why"
    assert lines[2] == "  lisbon-over: run failed: RateLimitError"
    assert not any("total 1610" in line for line in lines)


def test_the_run_leaves_the_examples_inputs_untouched(scripted_subagent):
    # `evaluate()` hands this same object to every evaluator after the run.
    # The graph coerces message dicts to message objects in place, which made
    # `rubric` crash on every example until the run function copied it.
    scripted_subagent(AIMessage("Nothing to cost."))
    inputs = {"messages": [{"role": "user", "content": "Cost this against $100."}]}
    harness.run_subagent("budget-analyst", inputs)
    assert inputs == {"messages": [{"role": "user", "content": "Cost this against $100."}]}


def test_the_dispatcher_hands_over_the_brief_verbatim_and_once():
    dispatcher = harness.Dispatcher(subagent="budget-analyst")
    first = dispatcher.invoke([HumanMessage("Cost this against $100.")])
    assert first.tool_calls[0]["name"] == "task"
    assert first.tool_calls[0]["args"] == {
        "subagent_type": "budget-analyst",
        "description": "Cost this against $100.",
    }
    after = dispatcher.invoke(
        [HumanMessage("x"), first, ToolMessage("done", tool_call_id=first.tool_calls[0]["id"])]
    )
    assert not after.tool_calls


def test_every_dataset_is_scored_and_each_subagent_one_is_a_real_subagent():
    roster = {subagent["name"] for subagent in build_subagents([])}
    assert set(harness.SUBAGENT_FOR.values()) <= roster
    # Every dataset file, so a fifth one has to decide how it is run.
    assert {*harness.SUBAGENT_FOR, *harness.AGENT_DATASETS} == set(load_datasets())
    assert set(harness.EVALUATORS) == set(load_datasets())
    assert not set(harness.SUBAGENT_FOR) & set(harness.AGENT_DATASETS)


# --- whole-agent run ---------------------------------------------------------


def _from(model: str, message: AIMessage, total: int, out: int) -> AIMessage:
    message.response_metadata = {"model_name": model}
    return _billed(message, total, 0, 0, out)


def test_the_agent_run_tags_each_call_with_its_agent_and_turn(scripted_subagent):
    # One script serves the main agent and the subagent alike; the run is
    # sequential, so they consume it in this order.
    items = [{"label": "Food", "category": "food", "amount": 10, "quantity": 2}]
    scripted_subagent(
        _from(
            "claude-opus-5-5",
            _call("task", subagent_type="budget-analyst", description="Cost it."),
            9000,
            200,
        ),
        _from(
            "claude-haiku-4-5",
            _call("summarize_budget", items=items, budget_total=100),
            3000,
            100,
        ),
        _from("claude-haiku-4-5", AIMessage("Total $20, fits."), 3200, 50),
        _call("write_file", file_path="/trip/itinerary.md", content="# Plan"),
        AIMessage("Here is your plan."),
    )
    outputs = harness.run_agent(BRIEF)

    assert [(c["agent"], c["step"], c["name"]) for c in outputs["tool_calls"]] == [
        ("main", 1, "task"),
        ("subagent", 1, "summarize_budget"),  # made during the turn that delegated
        ("main", 2, "write_file"),
    ]
    assert outputs["response"] == "Here is your plan."
    assert outputs["files"] == {"/trip/itinerary.md": "# Plan"}
    # The judge sees the subagent's raw tool result as well as the prose the
    # main agent got back, which is all a main-agent transcript would hold.
    results = {(r["agent"], r["name"]): r["content"] for r in outputs["tool_results"]}
    assert '"total_estimated": 20.0' in results[("subagent", "summarize_budget")]
    assert results[("main", "task")] == "Total $20, fits."
    assert outputs["usage"] == {
        "claude-opus-5-5": {
            "model_calls": 1,
            "input_tokens": 9000,
            "cache_read": 0,
            "cache_creation": 0,
            "output_tokens": 200,
        },
        "claude-haiku-4-5": {
            "model_calls": 2,
            "input_tokens": 6200,
            "cache_read": 0,
            "cache_creation": 0,
            "output_tokens": 150,
        },
    }
    assert outputs["subject"] == "agent"


def test_a_request_in_two_datasets_runs_once(monkeypatch: pytest.MonkeyPatch):
    runs = []

    def fake_run(inputs: dict) -> dict:
        runs.append(inputs)
        return {"tool_calls": [{"agent": "main", "step": 1, "name": "task", "args": {}}]}

    monkeypatch.setattr(harness, "run_agent", fake_run)
    monkeypatch.setattr(harness, "_agent_runs", {})
    first = harness.run_agent_shared(BRIEF)
    first["tool_calls"].clear()  # an evaluator that mutates must not reach the cache
    second = harness.run_agent_shared({"messages": [dict(BRIEF["messages"][0])]})
    assert len(runs) == 1
    assert "reused" not in first and second["reused"] is True
    assert second["tool_calls"] == [{"agent": "main", "step": 1, "name": "task", "args": {}}]


def test_a_reused_run_does_not_print_its_tokens_twice():
    assert harness._usage_lines({"reused": True, "usage": {"m": {}}}) == [
        "same run as an earlier dataset in this process; tokens counted there"
    ]
    per_model = {"b": dict.fromkeys(harness.USAGE_FIELDS, 1), "a": {}}
    lines = harness._usage_lines({"usage": per_model})
    assert lines[0].startswith("a: model_calls=0") and lines[1].startswith("b: model_calls=1")


def test_final_response_without_the_judge_is_refused_before_anything_runs(capsys):
    with pytest.raises(SystemExit):
        harness.main(["final_response", "--no-judge"])
    assert "only by the judge" in capsys.readouterr().err


# --- trajectory evaluators ---------------------------------------------------

TRAJECTORY_REF = {
    "required_subagents": ["availability-scout"],
    "forbidden_subagents": ["destination-researcher"],
    "ordered_subagents": [],
    "required_any_of": [
        {
            "subagents": ["budget-analyst"],
            "tools": ["summarize_budget"],
            "after_subagent": "availability-scout",
        }
    ],
    "required_tools": ["task"],
    "forbidden_tools": ["delete"],
    "required_file_writes": ["/trip/itinerary.md"],
    "forbidden_file_reads": ["/memories/traveler_profile.md"],
}


def _trip(*turns: list[tuple[str, dict]]) -> dict:
    """Main-agent calls, one list per model turn."""
    calls = [
        {"agent": "main", "step": step, "name": name, "args": args}
        for step, turn in enumerate(turns, 1)
        for name, args in turn
    ]
    return {"tool_calls": calls, "response": "", "files": {}}


def _task(subagent: str) -> tuple[str, dict]:
    return ("task", {"subagent_type": subagent, "description": "brief"})


SCORED_TRIP = _trip(
    [_task("availability-scout")],
    [("summarize_budget", {"items": [], "budget_total": 4000})],
    [("write_file", {"file_path": "/trip/itinerary.md", "content": "# Plan"})],
)
# The scout's search, made during the turn that delegated to it.
SCORED_TRIP["tool_calls"].insert(
    1,
    {
        "agent": "subagent",
        "step": 1,
        "name": "search_flights",
        "args": {"depart_date": "2027-04-05", "return_date": "2027-04-10"},
    },
)


def test_a_trip_costed_after_scouting_passes_every_trajectory_check():
    for evaluator in harness.EVALUATORS["trajectory"]:
        assert evaluator(SCORED_TRIP, TRAJECTORY_REF)["score"] == 1, evaluator


def test_costing_in_the_same_turn_as_the_scout_is_not_after_it():
    # Listed second, but run alongside the scout: the analyst never saw its prices.
    parallel = _trip(
        [_task("availability-scout"), _task("budget-analyst")],
        [("write_file", {"file_path": "/trip/itinerary.md"})],
    )
    result = harness.delegation_order(parallel, TRAJECTORY_REF)
    assert result["score"] == 0
    assert (
        "budget-analyst or summarize_budget in a turn after availability-scout"
        in (result["comment"])
    )


def test_a_subagents_own_calls_do_not_count_for_the_main_agent():
    # The scout searching is not the main agent searching, and the analyst
    # calling summarize_budget is the analyst's call, not a direct one.
    run = _trip([_task("availability-scout")])
    run["tool_calls"] += [
        {"agent": "subagent", "step": 2, "name": "summarize_budget", "args": {}},
        {
            "agent": "subagent",
            "step": 2,
            "name": "read_file",
            "args": {"file_path": "/memories/traveler_profile.md"},
        },
    ]
    assert harness.delegation_order(run, TRAJECTORY_REF)["score"] == 0
    assert harness.file_access(run, TRAJECTORY_REF)["comment"] == (
        "never wrote ['/trip/itinerary.md']"
    )


def test_ordered_subagents_are_a_subsequence_across_turns():
    ref = {
        **TRAJECTORY_REF,
        "ordered_subagents": ["destination-researcher", "availability-scout"],
        "required_any_of": [],
    }
    in_order = _trip([_task("destination-researcher")], [_task("availability-scout")])
    reversed_ = _trip([_task("availability-scout")], [_task("destination-researcher")])
    together = _trip([_task("destination-researcher"), _task("availability-scout")])
    assert harness.delegation_order(in_order, ref)["score"] == 1
    assert harness.delegation_order(reversed_, ref)["score"] == 0
    assert harness.delegation_order(together, ref)["score"] == 0


def test_delegation_and_tools_report_both_kinds_of_failure():
    run = _trip([_task("destination-researcher"), ("delete", {"file_path": "/trip/x"})])
    delegated = harness.delegation(run, TRAJECTORY_REF)
    assert delegated["score"] == 0
    assert "never delegated to ['availability-scout']" in delegated["comment"]
    assert "forbidden ['destination-researcher']" in delegated["comment"]
    assert harness.tool_use(run, TRAJECTORY_REF) == {
        "score": 0,
        "comment": "called forbidden ['delete']",
    }


def test_an_edit_counts_as_a_write_and_a_profile_read_fails():
    run = _trip(
        [("read_file", {"file_path": "/memories/traveler_profile.md"})],
        [("edit_file", {"file_path": "/trip/itinerary.md", "old_string": "a", "new_string": "b"})],
    )
    assert harness.file_access(run, TRAJECTORY_REF) == {
        "score": 0,
        "comment": "read ['/memories/traveler_profile.md']",
    }


def test_a_search_dated_in_the_past_fails_at_any_level():
    # The pinned today is 2026-01-01. The scout searching what its brief said
    # is the main agent's brief showing through, so subagent calls count here.
    substituted = _trip([_task("availability-scout")])
    substituted["tool_calls"].append(
        {
            "agent": "subagent",
            "step": 1,
            "name": "search_flights",
            "args": {"depart_date": "2026-12-01", "return_date": "2026-12-04"},
        }
    )
    assert harness.searches_not_in_past(substituted, TRAJECTORY_REF)["score"] == 1
    as_given = copy.deepcopy(substituted)
    as_given["tool_calls"][-1]["args"]["depart_date"] = "2025-12-01"
    assert harness.searches_not_in_past(as_given, TRAJECTORY_REF) == {
        "score": 0,
        "comment": "searched past dates ['2025-12-01']",
    }


def test_no_search_is_not_a_pass_on_dates():
    # Vacuous, so n/a: a 1 here would inflate every example that never searches.
    assert harness.searches_not_in_past(_trip(), TRAJECTORY_REF)["score"] is None


def test_no_ordering_in_the_reference_is_not_applicable():
    ref = {**TRAJECTORY_REF, "required_any_of": []}
    assert harness.delegation_order(_trip(), ref)["score"] is None


def _ideal_trip(reference: dict) -> dict:
    """The shortest main-agent trajectory a reference describes."""
    turns: list[list[tuple[str, dict]]] = []
    ordered = reference["ordered_subagents"]
    for name in [*ordered, *(s for s in reference["required_subagents"] if s not in ordered)]:
        turns.append([_task(name)])
    for group in reference["required_any_of"]:
        tools = group.get("tools", [])
        turns.append([(tools[0], {})] if tools else [_task(group["subagents"][0])])
    called = {name for turn in turns for name, _ in turn}
    turns += [[(tool, {})] for tool in reference["required_tools"] if tool not in called]
    turns += [[("write_file", {"file_path": p})] for p in reference["required_file_writes"]]
    return _trip(*turns)


@pytest.mark.parametrize(
    "example",
    load_datasets()["trajectory"]["examples"],
    ids=lambda e: e["key"],
)
def test_every_trajectory_reference_is_reachable(example):
    # A reference no run can meet still scores, just always wrong, and a low
    # score reads as a worse agent. The dataset tests check its names; this
    # checks the evaluators can actually award it full marks.
    reference = example["outputs"]
    ideal = _ideal_trip(reference)
    for evaluator in harness.EVALUATORS["trajectory"]:
        assert evaluator(ideal, reference)["score"] in {1, None}, evaluator


# --- availability-scout evaluators -------------------------------------------

FLIGHTS = {
    "origin": "SFO",
    "destination": "KIX",
    "depart_date": "2027-04-05",
    "return_date": "2027-04-10",
    "travelers": 2,
    "cabin": "economy",
}
SCOUT_REF = {"expected_calls": {"search_flights": FLIGHTS}, "forbidden_tools": ["search_stays"]}


def _outputs(*calls: tuple[str, dict]) -> dict:
    return {"tool_calls": [{"name": n, "args": a} for n, a in calls], "response": "", "files": {}}


def test_search_arguments_reads_omitted_fields_as_the_tools_defaults():
    # A scout that leaves `cabin` out searched economy; that is a pass.
    args = {k: v for k, v in FLIGHTS.items() if k != "cabin"}
    assert harness.search_arguments(_outputs(("search_flights", args)), SCOUT_REF)["score"] == 1


def test_search_arguments_forgives_what_the_tool_forgives():
    args = {**FLIGHTS, "origin": "sfo", "cabin": "Economy"}
    assert harness.search_arguments(_outputs(("search_flights", args)), SCOUT_REF)["score"] == 1


def test_a_varied_search_before_the_briefed_one_still_passes():
    # The calls a real `business-for-two` run made: lodging from the arrival
    # day first (LAX-SYD crosses the date line), then the briefed dates. The
    # first-call rule scored that 0.5 for being more careful than the brief.
    ref = {
        "expected_calls": {
            "search_stays": {
                "location": "Sydney",
                "check_in": "2027-01-20",
                "check_out": "2027-02-03",
                "guests": 2,
            }
        }
    }
    briefed = {
        "location": "Sydney",
        "check_in": "2027-01-20",
        "check_out": "2027-02-03",
        "guests": 2,
    }
    arrival_day = {**briefed, "check_in": "2027-01-21"}
    run = _outputs(("search_stays", arrival_day), ("search_stays", briefed))
    assert harness.search_arguments(run, ref)["score"] == 1


def test_no_call_matching_the_brief_fails_and_names_the_nearest_miss():
    shifted = {**FLIGHTS, "depart_date": "2027-04-06"}
    wrong_twice = {**shifted, "origin": "OAK"}
    result = harness.search_arguments(
        _outputs(("search_flights", wrong_twice), ("search_flights", shifted)), SCOUT_REF
    )
    assert result["score"] == 0
    # The nearest call is wrong on one field only, so that is the one named.
    assert "depart_date='2027-04-06'" in result["comment"] and "OAK" not in result["comment"]


def test_a_missing_search_scores_zero_with_a_reason():
    result = harness.search_arguments(_outputs(), SCOUT_REF)
    assert result == {"score": 0, "comment": "search_flights never called"}


def test_a_location_with_its_country_still_matches():
    ref = {"expected_calls": {"search_stays": {"location": "Kyoto", "guests": 2}}}
    outputs = _outputs(("search_stays", {"location": "Kyoto, Japan", "guests": 2}))
    assert harness.search_arguments(outputs, ref)["score"] == 1


def test_cabin_is_checked_on_every_flight_search_not_just_the_first():
    upgraded_retry = _outputs(
        ("search_flights", FLIGHTS), ("search_flights", {**FLIGHTS, "cabin": "business"})
    )
    assert harness.cabin_as_briefed(upgraded_retry, SCOUT_REF)["score"] == 0
    assert harness.cabin_as_briefed(_outputs(("search_flights", FLIGHTS)), SCOUT_REF)["score"] == 1


def test_cabin_is_not_applicable_without_a_flight_search():
    ref = {"expected_calls": {"search_stays": {"location": "Kyoto"}}}
    assert harness.cabin_as_briefed(_outputs(), ref)["score"] is None


def test_a_forbidden_tool_scores_zero():
    used = _outputs(("search_flights", FLIGHTS), ("search_stays", {"location": "Kyoto"}))
    assert harness.forbidden_tools_unused(used, SCOUT_REF)["score"] == 0
    assert harness.forbidden_tools_unused(_outputs(), SCOUT_REF)["score"] == 1


# --- budget-analyst evaluators -----------------------------------------------

LINES = [
    {"label": "Flights", "category": "flights", "amount": 450, "quantity": 1},
    {"label": "Lodging", "category": "lodging", "amount": 960, "quantity": 1},
    {"label": "Food", "category": "food", "amount": 50, "quantity": 4},
]
BUDGET_REF = {
    "budget_total": 2000,
    "currency": "USD",
    "expected_total": 1610,
    "excluded_amounts": [85],
}


def _budget(items: list[dict], **extra) -> dict:
    return _outputs(("summarize_budget", {"items": items, "budget_total": 2000, **extra}))


def test_budget_total_reruns_the_real_tool_on_the_analysts_arguments():
    assert harness.budget_total(_budget(LINES), BUDGET_REF)["score"] == 1
    padded = [*LINES, {"label": "Resort fee", "category": "fees", "amount": 85}]
    assert harness.budget_total(_budget(padded), BUDGET_REF)["score"] == 0


def test_budget_total_scores_the_last_call():
    # The analyst may correct itself; the corrected call is its answer.
    wrong = ("summarize_budget", {"items": LINES[:1], "budget_total": 2000})
    right = ("summarize_budget", {"items": LINES, "budget_total": 2000})
    assert harness.budget_total(_outputs(wrong, right), BUDGET_REF)["score"] == 1


def test_a_call_the_tool_rejects_scores_zero_instead_of_crashing():
    bad = [{"label": "x", "category": "not-a-category", "amount": 1}]
    result = harness.budget_total(_budget(bad), BUDGET_REF)
    assert result["score"] == 0 and "rejected" in result["comment"]


def test_currency_defaults_to_usd_when_the_analyst_omits_it():
    assert harness.budget_currency(_budget(LINES), BUDGET_REF)["score"] == 1
    sek = {**BUDGET_REF, "currency": "SEK"}
    assert harness.budget_currency(_budget(LINES), sek)["score"] == 0
    assert harness.budget_currency(_budget(LINES, currency="sek"), sek)["score"] == 1


@pytest.mark.parametrize(
    "line",
    [
        {"label": "Resort fee", "category": "fees", "amount": 85},
        {"label": "Resort fee", "category": "fees", "amount": 21.25, "quantity": 4},
    ],
    ids=["as-one-line", "spread-per-night"],
)
def test_the_due_at_accommodation_amount_must_not_be_a_line(line):
    assert harness.due_at_accommodation_excluded(_budget([*LINES, line]), BUDGET_REF)["score"] == 0
    assert harness.due_at_accommodation_excluded(_budget(LINES), BUDGET_REF)["score"] == 1


def test_the_exclusion_check_is_not_applicable_without_an_excluded_amount():
    ref = {k: v for k, v in BUDGET_REF.items() if k != "excluded_amounts"}
    assert harness.due_at_accommodation_excluded(_budget(LINES), ref)["score"] is None


# --- rubric judge ------------------------------------------------------------


class _FixedJudge:
    def __init__(self, *met: bool):
        self.met, self.prompts = met, []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return harness.RubricGrade(
            verdicts=[
                harness.CriterionVerdict(criterion=f"c{i}", reasoning="r", met=m)
                for i, m in enumerate(self.met)
            ]
        )


def test_rubric_scores_the_share_of_criteria_met(monkeypatch: pytest.MonkeyPatch):
    judge = _FixedJudge(True, False, True, True)
    monkeypatch.setattr(harness, "_judge", lambda: judge)
    ref = {"criteria": ["a", "b", "c", "d"]}
    outputs = {
        **_budget(LINES),
        "tool_results": [{"name": "search_stays", "content": '{"free_cancellation": false}'}],
        "response": "Total $1610.",
        "files": {"/trip/budget.md": "x"},
    }
    result = harness.rubric(BRIEF, outputs, ref)
    assert result["score"] == 0.75 and "c1" in result["comment"]
    # The judge sees every kind of evidence a criterion can be about. Without
    # the tool results it failed a scout for correctly calling a stay with
    # `free_cancellation: false` non-refundable: it could not see the field.
    prompt = judge.prompts[0]
    assert "Total $1610." in prompt and "summarize_budget" in prompt and "/trip/budget.md" in prompt
    assert '"free_cancellation": false' in prompt


def test_the_judge_names_what_it_grades_without_rewording_the_subagent_prompt(monkeypatch):
    judge = _FixedJudge(True)
    monkeypatch.setattr(harness, "_judge", lambda: judge)
    ref = {"criteria": ["a"]}
    harness.rubric(BRIEF, _budget(LINES), ref)
    harness.rubric(BRIEF, {**_budget(LINES), "subject": "agent"}, ref)
    subagent_prompt, agent_prompt = judge.prompts
    # The subagent experiments already in LangSmith were graded on this text.
    assert subagent_prompt.startswith(
        "You are grading one run of a travel-planning subagent against a rubric."
    )
    assert "the subagent claims about the data" in subagent_prompt
    assert agent_prompt.startswith(
        "You are grading one run of a travel-planning agent against a rubric."
    )
    assert "subagent" not in agent_prompt.split("<brief>")[0]


def test_a_grade_with_the_wrong_number_of_verdicts_is_not_a_score(monkeypatch):
    monkeypatch.setattr(harness, "_judge", lambda: _FixedJudge(True))
    result = harness.rubric(BRIEF, _budget(LINES), {"criteria": ["a", "b"]})
    assert result["score"] is None


def test_the_judge_request_never_forces_a_tool(monkeypatch: pytest.MonkeyPatch):
    # Sonnet 5.5 rejects forced `tool_choice`, which LangChain's default
    # structured output sends. Built offline: constructing a client is not a call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    harness._judge.cache_clear()
    try:
        chain = harness._judge()
    finally:
        harness._judge.cache_clear()
    binding = chain.first
    payload = binding.bound._get_request_payload("grade this", **binding.kwargs)
    assert "tool_choice" not in payload and "tools" not in payload
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["effort"] == harness.JUDGE_EFFORT
    assert payload["model"] == harness.JUDGE_MODEL.split(":", 1)[1]
    assert payload["fallbacks"] == "default"


# --- hard-set evaluators -----------------------------------------------------


OPEN_JAW = {
    "expected_calls": {
        "search_flights": [
            {"origin": "ORD", "destination": "CDG", "return_date": None, "cabin": "economy"},
            {"origin": "CDG", "destination": "FCO", "return_date": None, "cabin": "economy"},
        ]
    }
}


def test_each_expected_call_in_a_list_needs_its_own_match():
    first_leg = ("search_flights", {"origin": "ORD", "destination": "CDG"})
    round_trip = ("search_flights", {"origin": "ORD", "destination": "CDG", "return_date": "x"})
    second_leg = ("search_flights", {"origin": "CDG", "destination": "FCO"})
    assert harness.search_arguments(_outputs(first_leg), OPEN_JAW)["score"] == 0.5
    assert harness.search_arguments(_outputs(first_leg, second_leg), OPEN_JAW)["score"] == 1
    # A round trip is the familiar shape and the wrong one; the miss names why.
    result = harness.search_arguments(_outputs(round_trip, second_leg), OPEN_JAW)
    assert result["score"] == 0.5 and "return_date=None" in result["comment"]


def test_no_expected_searches_is_not_applicable():
    assert harness.search_arguments(_outputs(), {"criteria": ["a"]})["score"] is None
    assert harness.cabin_as_briefed(_outputs(), {"criteria": ["a"]})["score"] is None


def test_cabin_is_checked_against_every_briefed_leg():
    business = ("search_flights", {**FLIGHTS, "cabin": "business"})
    assert harness.cabin_as_briefed(_outputs(("search_flights", FLIGHTS)), OPEN_JAW)["score"] == 1
    assert harness.cabin_as_briefed(_outputs(business), OPEN_JAW)["score"] == 0


YEN = {"forbidden_arguments": {"search_stays": {"max_nightly_rate": [20000]}}}


def test_an_unconverted_ceiling_fails_at_any_level():
    stays = {"location": "Kyoto", "guests": 2}
    raw = {
        "agent": "subagent",
        "name": "search_stays",
        "args": {**stays, "max_nightly_rate": "20000"},
    }
    converted = _outputs(("search_stays", {**stays, "max_nightly_rate": 133}))
    assert harness.forbidden_arguments(converted, YEN)["score"] == 1
    assert harness.forbidden_arguments({"tool_calls": [raw]}, YEN)["score"] == 0
    # Leaving the ceiling out and filtering by the converted figure is allowed.
    assert harness.forbidden_arguments(_outputs(("search_stays", stays)), YEN)["score"] == 1
    assert harness.forbidden_arguments(converted, {})["score"] is None


def test_a_total_that_rests_on_a_chosen_rate_is_not_scored_exactly():
    ref = {"budget_total": 3000, "currency": "EUR", "expected_total": None}
    assert harness.budget_total(_budget(LINES), ref)["score"] is None


def test_a_raw_foreign_figure_is_caught_per_unit_or_as_a_subtotal():
    ref = {"unconverted_amounts": [650, 1300], "currency": "EUR"}
    per_unit = [{"label": "Flights", "category": "flights", "amount": 650, "quantity": 2}]
    subtotal = [{"label": "Flights", "category": "flights", "amount": 1300}]
    converted = [{"label": "Flights", "category": "flights", "amount": 598, "quantity": 2}]
    assert harness.foreign_amounts_converted(_budget(per_unit), ref)["score"] == 0
    assert harness.foreign_amounts_converted(_budget(subtotal), ref)["score"] == 0
    assert harness.foreign_amounts_converted(_budget(converted), ref)["score"] == 1


def test_paid_flights_must_be_marked_confirmed():
    ref = {"confirmed_categories": ["flights"]}
    paid = [{"label": "Flights", "category": "flights", "amount": 890, "estimated": False}]
    unmarked = [{"label": "Flights", "category": "flights", "amount": 890}]
    assert harness.booked_lines_confirmed(_budget(paid), ref)["score"] == 1
    assert harness.booked_lines_confirmed(_budget(unmarked), ref)["score"] == 0
    assert harness.booked_lines_confirmed(_budget(LINES[1:]), ref)["comment"] == (
        "no ['flights'] line costed"
    )


def test_a_budget_nobody_gave_fails_at_any_level():
    ref = {"budget_given": False}
    # The Rome run's shape: costed against a "reference line" of $1500.
    invented = {
        "tool_calls": [
            {
                "agent": "main",
                "name": "summarize_budget",
                "args": {"items": [], "budget_total": 1500},
            }
        ]
    }
    assert harness.budget_not_invented(invented, ref) == {
        "score": 0,
        "comment": "costed against [1500]",
    }
    none_given = _outputs(("summarize_budget", {"items": LINES}))
    assert harness.budget_not_invented(none_given, ref)["score"] == 1
    assert harness.budget_not_invented(_outputs(), ref)["score"] is None
    assert harness.budget_not_invented(invented, {"budget_total": 1500})["score"] is None


def test_an_analyst_that_invents_a_budget_fails_the_no_budget_total():
    # Today's tool cannot total a plan without a budget, so this is the only
    # call it accepts, and it is the wrong one.
    ref = {"budget_total": None, "expected_total": 1610, "currency": "USD"}
    result = harness.budget_total(_budget(LINES), ref)
    assert (
        result["score"] == 0 and "against 2000.0; expected 1610 against None" in result["comment"]
    )


# --- repetitions -------------------------------------------------------------


def _row(key: str, **scores: float | None) -> dict:
    from types import SimpleNamespace as NS

    from langsmith.evaluation import EvaluationResult

    return {
        "example": NS(metadata={"key": key}, id=key),
        "run": NS(error=None, outputs={"usage": {}}),
        "evaluation_results": {
            "results": [EvaluationResult(key=k, score=v) for k, v in scores.items()]
        },
    }


def test_pass_rates_average_each_check_across_repetitions():
    rows = [
        _row("yen-ceiling", forbidden_arguments=1, rubric=1.0),
        _row("yen-ceiling", forbidden_arguments=0, rubric=0.5),
        _row("yen-ceiling", forbidden_arguments=1, rubric=None),
    ]
    assert harness.pass_rates(rows) == [
        "  pass rates:",
        "    yen-ceiling over 3 runs: forbidden_arguments=0.67 (3), rubric=0.75 (2)",
    ]
    assert harness.pass_rates(rows[:1]) == []


def test_an_unknown_key_is_refused_before_anything_runs(capsys):
    # Checked against the local files: a typo must not reach LangSmith and
    # evaluate zero examples, or every dataset but the one meant.
    assert harness.unknown_keys(["final_response"], ["lisbon-over-budget"]) == []
    assert harness.unknown_keys(["trajectory"], ["lisbon-over-budget"]) == ["lisbon-over-budget"]
    with pytest.raises(SystemExit):
        harness.main(["final_response", "--keys", "lisbon-over-budgte"])
    assert "lisbon-over-budgte" in capsys.readouterr().err


def test_no_web_search_drops_the_key_even_when_one_is_set(monkeypatch: pytest.MonkeyPatch):
    # With no credits the key is still set, and the tool answers with an error
    # the agent works around: search would be broken while reported as on.
    monkeypatch.setenv("TAVILY_API_KEY", "placeholder-not-a-key")
    # Registered with monkeypatch so the function's own writes are undone.
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    assert harness.apply_run_environment(no_web_search=False) is True
    assert harness.apply_run_environment(no_web_search=True) is False
    assert "TAVILY_API_KEY" not in os.environ
    assert os.environ["TRAVEL_AGENT_PROVIDER"] == "sample-data"


def test_repeated_runs_never_share_a_transcript():
    # Shared, every repetition would be served the first run, and three
    # identical scores would read as a stable pass rate.
    assert harness.agent_target(1) is harness.run_agent_shared
    assert harness.agent_target(3) is harness.run_agent


def test_an_offer_seconds_from_expiry_must_be_searched_again():
    ref = {"expiring_search": {"name": "search_flights", "match": {"origin": "SEA"}}}
    sea = ("search_flights", {"origin": "sea", "destination": "ICN"})
    assert harness.refreshed_expiring_offer(_outputs(sea, sea), ref)["score"] == 1
    once = harness.refreshed_expiring_offer(_outputs(sea), ref)
    assert once == {"score": 0, "comment": "searched 1 time(s)"}
    # A different route is not a refresh of the expiring one.
    elsewhere = ("search_flights", {"origin": "PDX", "destination": "ICN"})
    assert harness.refreshed_expiring_offer(_outputs(sea, elsewhere), ref)["score"] == 0
    assert harness.refreshed_expiring_offer(_outputs(sea), {})["score"] is None

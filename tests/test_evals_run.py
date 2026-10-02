"""The eval harness measures what it claims to.

Every failure here is a wrong score rather than an error: a run function that
loses the subagent's tool calls scores every run 0, and an evaluator that reads
the wrong field scores it 1. Nothing in LangSmith distinguishes either from a
real result, so the harness is driven here with scripted models. No network,
no model calls.
"""

from __future__ import annotations

from collections.abc import Iterator

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


def test_the_summary_prints_scores_and_tokens_and_survives_a_failed_run():
    from types import SimpleNamespace as NS

    from langsmith.evaluation import EvaluationResult

    usage = dict.fromkeys(harness.USAGE_FIELDS, 0) | {"model_calls": 3, "output_tokens": 500}
    ok = {
        "example": NS(metadata={"key": "kyoto-fits"}, id="e1"),
        "run": NS(error=None, outputs={"usage": usage}),
        "evaluation_results": {
            "results": [
                EvaluationResult(key="budget_total", score=1),
                EvaluationResult(key="due_at_accommodation_excluded", score=None),
            ]
        },
    }
    failed = {
        "example": NS(metadata={"key": "lisbon-over"}, id="e2"),
        "run": NS(error="RateLimitError", outputs=None),
        "evaluation_results": {"results": []},
    }
    lines = harness.summarize([ok, failed])
    assert "kyoto-fits: budget_total=1, due_at_accommodation_excluded=n/a" in lines[0]
    assert "model_calls=3" in lines[0] and "output_tokens=500" in lines[0]
    assert lines[1] == "  lisbon-over: run failed: RateLimitError"


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


def test_every_scored_dataset_maps_to_a_real_subagent_and_has_evaluators():
    roster = {subagent["name"] for subagent in build_subagents([])}
    assert set(harness.SUBAGENT_FOR.values()) <= roster
    assert set(harness.SUBAGENT_FOR) <= set(load_datasets())
    assert set(harness.EVALUATORS) == set(harness.SUBAGENT_FOR)


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


def test_search_arguments_only_reads_the_first_call():
    # The scout is told to vary later searches, so a shifted retry is fine,
    # but a first search on the wrong date is not.
    shifted = {**FLIGHTS, "depart_date": "2027-04-06"}
    good_then_varied = _outputs(("search_flights", FLIGHTS), ("search_flights", shifted))
    varied_first = _outputs(("search_flights", shifted), ("search_flights", FLIGHTS))
    assert harness.search_arguments(good_then_varied, SCOUT_REF)["score"] == 1
    result = harness.search_arguments(varied_first, SCOUT_REF)
    assert result["score"] == 0 and "depart_date" in result["comment"]


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
    outputs = {**_budget(LINES), "response": "Total $1610.", "files": {"/trip/budget.md": "x"}}
    result = harness.rubric(BRIEF, outputs, ref)
    assert result["score"] == 0.75 and "c1" in result["comment"]
    # The judge sees all three kinds of evidence a criterion can be about.
    prompt = judge.prompts[0]
    assert "Total $1610." in prompt and "summarize_budget" in prompt and "/trip/budget.md" in prompt


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

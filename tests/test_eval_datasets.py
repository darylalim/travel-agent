"""The evaluation datasets agree with the code they grade.

A reference that names a renamed subagent, a tool argument that no longer
exists, or a budget total that the real arithmetic disagrees with fails
nothing at upload time. It just scores every run wrong, and a low score reads
as a worse agent rather than a broken dataset. So each reference is checked
here against the live roster, tool schemas and `summarize_budget` itself.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.tools import BaseTool

import travel_agent.agent as agent_module
from evals import upload
from travel_agent.prompts import MEMORY_PATH, WORKSPACE
from travel_agent.subagents import build_subagents
from travel_agent.tools.availability import _normalize_cabin
from travel_agent.tools.budget import summarize_budget
from travel_agent.tools.search import build_search_tools

DATASETS = upload.load_datasets()
ROSTER = {subagent["name"]: subagent for subagent in build_subagents([])}


def _examples(stem: str) -> list[dict]:
    return DATASETS[stem]["examples"]


def _ids(stem: str) -> list[str]:
    return [example["key"] for example in _examples(stem)]


def test_all_four_datasets_are_present():
    # A renamed file would otherwise drop out of every parametrised test below.
    assert set(DATASETS) == {
        "final_response",
        "trajectory",
        "availability_scout",
        "budget_analyst",
    }


def test_dataset_names_are_unique():
    names = [dataset["name"] for dataset in DATASETS.values()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("stem", sorted(DATASETS))
def test_example_keys_are_unique_within_a_dataset(stem):
    keys = _ids(stem)
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("stem", sorted(DATASETS))
def test_inputs_are_the_graphs_own_input_shape(stem):
    for example in _examples(stem):
        messages = example["inputs"]["messages"]
        assert messages, example["key"]
        for message in messages:
            assert message["role"] in {"user", "assistant"}, example["key"]
            assert isinstance(message["content"], str) and message["content"].strip()


@pytest.mark.parametrize("stem", sorted(DATASETS))
def test_rubric_criteria_are_non_empty_sentences(stem):
    for example in _examples(stem):
        for criterion in example["outputs"].get("criteria", []):
            assert isinstance(criterion, str) and criterion.strip(), example["key"]


# --- trajectory -------------------------------------------------------------


@pytest.fixture(scope="module")
def main_agent_tools() -> set[str]:
    """Tool names the real main agent is built with, minus web search.

    Built with a fake model so no client is constructed. Web search depends on
    `TAVILY_API_KEY`, so it is deliberately left out: no reference may rely on
    it existing.
    """
    model = GenericFakeChatModel(messages=iter([]))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(agent_module, "_chat_model", lambda *a, **k: model)
        patch.delenv("TAVILY_API_KEY", raising=False)
        agent = agent_module.build_agent()
    return set(agent.nodes["tools"].bound.tools_by_name)


@pytest.fixture(scope="module")
def web_search_tools() -> set[str]:
    """The web search tool names, built offline with a placeholder key.

    A reference may offer one only as an alternative, in an example marked
    `requires_web_search`, since the agent lacks it without a key.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TAVILY_API_KEY", "placeholder-not-a-key")
        return {tool.name for tool in build_search_tools()}


@pytest.mark.parametrize("example", _examples("trajectory"), ids=_ids("trajectory"))
def test_trajectory_names_only_real_subagents(example):
    outputs = example["outputs"]
    named = {
        *outputs["required_subagents"],
        *outputs["forbidden_subagents"],
        *outputs["ordered_subagents"],
    }
    assert named <= set(ROSTER)
    assert not set(outputs["required_subagents"]) & set(outputs["forbidden_subagents"])
    assert set(outputs["ordered_subagents"]) <= set(outputs["required_subagents"])


@pytest.mark.parametrize("example", _examples("trajectory"), ids=_ids("trajectory"))
def test_trajectory_names_only_real_main_agent_tools(example, main_agent_tools):
    outputs = example["outputs"]
    assert set(outputs["required_tools"]) <= main_agent_tools
    assert set(outputs["forbidden_tools"]) <= main_agent_tools
    assert not set(outputs["required_tools"]) & set(outputs["forbidden_tools"])
    # Delegating at all requires `task`, so a required subagent with `task`
    # forbidden is a reference no run can satisfy.
    if outputs["required_subagents"]:
        assert "task" not in outputs["forbidden_tools"]


@pytest.mark.parametrize("example", _examples("trajectory"), ids=_ids("trajectory"))
def test_every_alternative_in_required_any_of_is_reachable(
    example, main_agent_tools, web_search_tools
):
    # An alternative naming a forbidden or nonexistent path still reads as
    # "either is fine" while only one of them can ever pass.
    outputs = example["outputs"]
    for group in outputs["required_any_of"]:
        assert set(group) <= {"subagents", "tools", "after_subagent"}, group
        subagents, tools = set(group.get("subagents", [])), set(group.get("tools", []))
        assert subagents or tools, group
        assert subagents <= set(ROSTER) and tools <= main_agent_tools | web_search_tools
        if tools & web_search_tools:
            assert example["metadata"]["requires_web_search"], example["key"]
        assert not subagents & set(outputs["forbidden_subagents"])
        assert not tools & set(outputs["forbidden_tools"])
        if subagents:
            assert "task" not in outputs["forbidden_tools"]
        # "After" a delegation that need not happen is no ordering at all.
        if after := group.get("after_subagent"):
            assert after in outputs["required_subagents"]


def test_a_full_trip_may_be_costed_without_the_analyst():
    # A real run costed the Kyoto trip with `summarize_budget` directly, as the
    # main prompt allows. v1 required `budget-analyst` and failed that run.
    full_trip = next(
        e for e in _examples("trajectory") if e["key"] == "full-trip-costs-after-scouting"
    )
    outputs = full_trip["outputs"]
    assert "budget-analyst" not in outputs["required_subagents"]
    assert {
        "subagents": ["budget-analyst"],
        "tools": ["summarize_budget"],
        "after_subagent": "availability-scout",
    } in outputs["required_any_of"]


@pytest.mark.parametrize("example", _examples("trajectory"), ids=_ids("trajectory"))
def test_trajectory_file_paths_follow_the_prompt_constants(example):
    outputs = example["outputs"]
    for path in outputs["required_file_writes"] + outputs["forbidden_file_reads"]:
        assert path == MEMORY_PATH or path.startswith(f"{WORKSPACE}/"), path


def test_research_examples_say_they_need_web_search(web_search_tools):
    # Without TAVILY_API_KEY the researcher has no tools and the main agent no
    # search, so an example that requires research should be skippable rather
    # than a guaranteed failure.
    for example in _examples("trajectory"):
        outputs = example["outputs"]
        research = {"destination-researcher", *web_search_tools}
        needs = bool(research & set(outputs["required_subagents"])) or any(
            research & {*group.get("subagents", []), *group.get("tools", [])}
            for group in outputs["required_any_of"]
        )
        assert example["metadata"]["requires_web_search"] is needs, example["key"]


def test_research_may_be_done_with_web_search_directly():
    # A real run answered the visa question with two `tavily_search` calls of
    # its own, as the prompt allows for a couple of tool calls. v2 required
    # `destination-researcher` and failed it.
    outputs = next(
        e for e in _examples("trajectory") if e["key"] == "entry-requirements-go-to-research"
    )["outputs"]
    assert "destination-researcher" not in outputs["required_subagents"]
    assert "task" not in outputs["required_tools"]
    assert {"subagents": ["destination-researcher"], "tools": ["tavily_search"]} in (
        outputs["required_any_of"]
    )


def test_past_dates_may_be_planned_on_substitutes():
    # A real run flagged the passed dates, moved them a year on and sent the
    # scout to search those. v2 forbade the scout and failed it; what the
    # example is about, a search dated in the past, is `searches_not_in_past`.
    outputs = next(e for e in _examples("trajectory") if e["key"] == "past-dates-not-searched")[
        "outputs"
    ]
    assert "availability-scout" not in outputs["forbidden_subagents"]


# --- availability-scout -----------------------------------------------------


def _scout_tools() -> dict[str, BaseTool]:
    # Read off the roster rather than imported, so a tool unbound from the
    # scout fails here instead of being "expected" of a subagent that lacks it.
    tools = ROSTER["availability-scout"]["tools"]
    assert all(isinstance(tool, BaseTool) for tool in tools)
    return {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}


@pytest.mark.parametrize("example", _examples("availability_scout"), ids=_ids("availability_scout"))
def test_scout_references_are_calls_the_real_tools_accept(example):
    tools = _scout_tools()
    outputs = example["outputs"]
    assert set(outputs["forbidden_tools"]) <= set(tools)
    assert not set(outputs["expected_calls"]) & set(outputs["forbidden_tools"])
    for name, args in outputs["expected_calls"].items():
        assert set(args) <= set(tools[name].args), name
        result = tools[name].invoke(args)
        assert "error" not in result, (name, result)


@pytest.mark.parametrize("example", _examples("availability_scout"), ids=_ids("availability_scout"))
def test_scout_reference_dates_come_from_the_brief(example):
    # The scout can only search dates it was given. A reference date missing
    # from the brief is one edit that moved the input and not the answer.
    brief = example["inputs"]["messages"][-1]["content"]
    for name, args in example["outputs"]["expected_calls"].items():
        for field in ("depart_date", "return_date", "check_in", "check_out"):
            if args.get(field):
                assert args[field] in brief, (name, field)


@pytest.mark.parametrize("example", _examples("availability_scout"), ids=_ids("availability_scout"))
def test_scout_reference_cabins_are_already_canonical(example):
    # An evaluator compares the scout's `cabin` argument by equality, and the
    # tool forgives "Premium Economy". A reference spelled that way would fail
    # a scout that sent the canonical value.
    flights = example["outputs"]["expected_calls"].get("search_flights")
    if flights:
        assert _normalize_cabin(flights["cabin"]) == flights["cabin"]


# --- budget-analyst ---------------------------------------------------------


@pytest.mark.parametrize("example", _examples("budget_analyst"), ids=_ids("budget_analyst"))
def test_budget_references_match_the_real_arithmetic(example):
    outputs = example["outputs"]
    result = summarize_budget.invoke(
        {
            "items": outputs["expected_items"],
            "budget_total": outputs["budget_total"],
            "currency": outputs["currency"],
        }
    )
    assert result["total_estimated"] == outputs["expected_total"]
    assert result["over_budget"] is outputs["over_budget"]


@pytest.mark.parametrize("example", _examples("budget_analyst"), ids=_ids("budget_analyst"))
def test_excluded_amounts_are_not_in_the_reference_lines(example):
    excluded = set(example["outputs"].get("excluded_amounts", []))
    amounts = {item["amount"] for item in example["outputs"]["expected_items"]}
    assert not excluded & amounts


# --- staleness --------------------------------------------------------------


def test_a_passed_date_is_stale_unless_the_example_is_about_it():
    example = {"inputs": {"messages": [{"role": "user", "content": "2026-03-01 to 2026-03-04"}]}}
    assert upload.stale_dates(example, date(2026, 3, 2)) == ["2026-03-01"]
    reference = {**example, "outputs": {"depart_date": "2026-02-01"}}
    assert upload.stale_dates(reference, date(2026, 3, 1)) == ["2026-02-01"]
    assert upload.stale_dates(example, date(2026, 3, 1)) == []
    example["metadata"] = {"dates_intentionally_past": True}
    assert upload.stale_dates(example, date(2027, 1, 1)) == []


def test_only_examples_marked_as_past_carry_past_dates():
    # Run against the dates the datasets were written for, not the wall
    # clock: the moment they pass is a reason to move them, not a red suite.
    authored = date(2026, 10, 2)
    assert upload._check(DATASETS, authored) == []
    marked = [
        example["key"]
        for dataset in DATASETS.values()
        for example in dataset["examples"]
        if example.get("metadata", {}).get("dates_intentionally_past")
    ]
    assert marked and all(
        upload.stale_dates({"inputs": example["inputs"], "outputs": example["outputs"]}, authored)
        for dataset in DATASETS.values()
        for example in dataset["examples"]
        if example["key"] in marked
    )

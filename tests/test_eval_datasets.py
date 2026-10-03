"""The evaluation datasets agree with the code they grade.

A reference that names a renamed subagent, a tool argument that no longer
exists, or a budget total that the real arithmetic disagrees with fails
nothing at upload time. It just scores every run wrong, and a low score reads
as a worse agent rather than a broken dataset. So each reference is checked
here against the live roster, tool schemas and `summarize_budget` itself.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.tools import BaseTool

import travel_agent.agent as agent_module
from evals import run as harness
from evals import scripted, upload
from travel_agent.prompts import MEMORY_PATH, WORKSPACE
from travel_agent.subagents import build_subagents
from travel_agent.tools.availability import SampleProvider, _normalize_cabin
from travel_agent.tools.budget import summarize_budget
from travel_agent.tools.search import build_search_tools

DATASETS = upload.load_datasets()
ROSTER = {subagent["name"]: subagent for subagent in build_subagents([])}


def _examples(stem: str) -> list[dict]:
    return DATASETS[stem]["examples"]


def _ids(stem: str) -> list[str]:
    return [example["key"] for example in _examples(stem)]


def test_every_dataset_is_present():
    # A renamed file would otherwise drop out of every parametrised test below.
    assert set(DATASETS) == {
        "final_response",
        "final_response_hard",
        "final_response_scripted",
        "trajectory",
        "availability_scout",
        "availability_scout_hard",
        "availability_scout_scripted",
        "budget_analyst",
        "budget_analyst_hard",
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


def _family(*stems: str) -> list:
    """Every example across `stems`, as pytest params with stem-qualified ids."""
    return [
        pytest.param(example, id=f"{stem}/{example['key']}")
        for stem in stems
        for example in _examples(stem)
    ]


SCOUT = _family("availability_scout", "availability_scout_hard", "availability_scout_scripted")
# Examples whose references name searches, wherever the searching happens.
SEARCHES = _family(
    "availability_scout",
    "availability_scout_hard",
    "availability_scout_scripted",
    "final_response_hard",
    "final_response_scripted",
)
BUDGET = _family("budget_analyst", "budget_analyst_hard")


def _scout_tools() -> dict[str, BaseTool]:
    # Read off the roster rather than imported, so a tool unbound from the
    # scout fails here instead of being "expected" of a subagent that lacks it.
    tools = ROSTER["availability-scout"]["tools"]
    assert all(isinstance(tool, BaseTool) for tool in tools)
    return {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}


def _expected(example: dict) -> list[tuple[str, dict]]:
    return harness._expected_calls(example["outputs"])


@pytest.mark.parametrize("example", SEARCHES)
def test_search_references_name_real_arguments(example):
    tools = _scout_tools()
    outputs = example["outputs"]
    for name, args in _expected(example):
        assert set(args) <= set(tools[name].args), name
    for name, fields in outputs.get("forbidden_arguments", {}).items():
        assert set(fields) <= set(tools[name].args), name
        # A reference that expects a value it also forbids fails every run.
        for field, values in fields.items():
            for expected_name, args in _expected(example):
                if expected_name == name and field in args:
                    assert args[field] not in values, (name, field)


@pytest.mark.parametrize("example", SCOUT)
def test_scout_references_are_calls_the_real_tools_accept(example):
    # Partial references are fine for the scout too: the tool fills the rest
    # with its own defaults, as `search_arguments` assumes.
    tools = _scout_tools()
    outputs = example["outputs"]
    assert set(outputs["forbidden_tools"]) <= set(tools)
    assert not set(outputs["expected_calls"]) & set(outputs["forbidden_tools"])
    for name, args in _expected(example):
        result = tools[name].invoke(args)
        assert "error" not in result, (name, result)


@pytest.mark.parametrize("example", SEARCHES)
def test_scout_reference_dates_come_from_the_brief(example):
    # The scout can only search dates it was given. A reference date missing
    # from the brief is one edit that moved the input and not the answer.
    brief = example["inputs"]["messages"][-1]["content"]
    for name, args in _expected(example):
        for field in ("depart_date", "return_date", "check_in", "check_out"):
            if args.get(field):
                assert args[field] in brief, (name, field)


@pytest.mark.parametrize("example", SEARCHES)
def test_scout_reference_cabins_are_already_canonical(example):
    # An evaluator compares the scout's `cabin` argument by equality, and the
    # tool forgives "Premium Economy". A reference spelled that way would fail
    # a scout that sent the canonical value.
    for name, args in _expected(example):
        if name == "search_flights" and "cabin" in args:
            assert _normalize_cabin(args["cabin"]) == args["cabin"]


def test_the_empty_ceiling_is_below_every_sample_stay_however_the_city_is_spelled():
    # The sample seed includes the location string, so a ceiling picked just
    # under one spelling's cheapest stay could return offers for another.
    # The warning must survive the empty result: it is the only thing left
    # saying sample data, since `sources` is empty too.
    example = next(
        e for e in _examples("availability_scout_hard") if e["key"] == "ceiling-below-everything"
    )
    args = example["outputs"]["expected_calls"]["search_stays"]
    for location in ("Edinburgh", "Edinburgh, UK", "Edinburgh, Scotland", "EDINBURGH"):
        result = _scout_tools()["search_stays"].invoke({**args, "location": location})
        assert result["offers"] == [] and result["sources"] == [], location
        assert result["warning"], location


# --- budget-analyst ---------------------------------------------------------


def _total(items: list[dict]) -> float:
    return round(sum(item["amount"] * item.get("quantity", 1) for item in items), 2)


@pytest.mark.parametrize("example", BUDGET)
def test_budget_references_match_the_real_arithmetic(example):
    outputs = example["outputs"]
    if outputs["expected_total"] is None:
        # Only an exchange rate the analyst picks leaves the total open, and
        # then the foreign figures are what the example scores instead.
        assert outputs["unconverted_amounts"], example["key"]
        return
    if outputs["budget_total"] is None:
        assert _total(outputs["expected_items"]) == outputs["expected_total"]
        assert outputs["over_budget"] is None
        return
    result = summarize_budget.invoke(
        {
            "items": outputs["expected_items"],
            "budget_total": outputs["budget_total"],
            "currency": outputs["currency"],
        }
    )
    assert result["total_estimated"] == outputs["expected_total"]
    assert result["over_budget"] is outputs["over_budget"]


@pytest.mark.parametrize("example", BUDGET)
def test_excluded_amounts_are_not_in_the_reference_lines(example):
    outputs = example["outputs"]
    ruled_out = {*outputs.get("excluded_amounts", []), *outputs.get("unconverted_amounts", [])}
    amounts = {item["amount"] for item in outputs.get("expected_items", [])}
    assert not ruled_out & amounts


@pytest.mark.parametrize("example", BUDGET)
def test_confirmed_categories_are_confirmed_in_the_reference(example):
    outputs = example["outputs"]
    for category in outputs.get("confirmed_categories", []):
        lines = [i for i in outputs["expected_items"] if i["category"] == category]
        assert lines and all(i.get("estimated") is False for i in lines), category


def test_the_no_budget_references_are_calls_the_tool_accepts():
    # Both were written while `summarize_budget` required a budget, so they
    # could only fail. Pinned against the real tool now that it can total a
    # plan with none, so a required budget coming back fails here first.
    no_budget = [
        example
        for stem in ("budget_analyst_hard", "final_response_hard")
        for example in _examples(stem)
        if example["outputs"].get("budget_given") is False
    ]
    assert len(no_budget) == 2
    for example in no_budget:
        assert "known_gap" not in example["metadata"], example["key"]
    analyst = next(e for e in no_budget if "expected_items" in e["outputs"])["outputs"]
    result = summarize_budget.invoke({"items": analyst["expected_items"]})
    assert result["total_estimated"] == analyst["expected_total"]
    assert result["budget_total"] is None and "over_budget" not in result


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


# --- scripted tools ---------------------------------------------------------

SCRIPTED = _family("availability_scout_scripted", "final_response_scripted")
_TOOL_FOR = {"flights": "search_flights", "stays": "search_stays"}


@pytest.mark.parametrize("example", SCRIPTED)
def test_every_scripted_example_carries_a_script_the_hook_can_read(example):
    script = example["metadata"]["scripted_tools"]
    assert set(script) <= {"flights", "stays", "search", "search_default"}, example["key"]
    tools = _scout_tools()
    sample = {
        "flights": SampleProvider().search_flights("SFO", "NRT", "2027-01-10", None, 1)[0],
        "stays": SampleProvider().search_stays("Kyoto", "2027-01-10", "2027-01-12", 1, None)[0],
    }
    for kind in ("flights", "stays"):
        for rule in script.get(kind, {}).get("rules", []):
            assert set(rule["match"]) <= set(tools[_TOOL_FOR[kind]].args), (kind, rule)
            responses = rule.get("responses") or [{"offers": rule["offers"]}]
            allowed = set(sample[kind]) | scripted.LIVE_ONLY_KEYS[kind]
            for response in responses:
                for patch in response["offers"]:
                    # A key no real provider emits would test a shape nobody serves.
                    assert set(patch) <= allowed, (kind, sorted(set(patch) - allowed))
    for rule in script.get("search", []):
        assert rule["match_any"] and ("result" in rule) != ("error" in rule), rule
    if "search" in script:
        assert script.get("search_default") in {"error", "empty"}


@pytest.mark.parametrize("example", SCRIPTED)
def test_every_scripted_rule_is_reached_by_the_search_the_brief_asks_for(example):
    # A rule no briefed search matches falls back to sample data, and the
    # example quietly measures nothing.
    script = example["metadata"]["scripted_tools"]
    expected = _expected(example)
    for kind in ("flights", "stays"):
        for rule in script.get(kind, {}).get("rules", []):
            reached = any(
                name == _TOOL_FOR[kind]
                and all(
                    f in args and scripted._matches(f, v, args[f]) for f, v in rule["match"].items()
                )
                for name, args in expected
            )
            assert reached, (example["key"], kind, rule["match"])


def test_scripted_inputs_are_unique_across_every_dataset():
    # Scripts are looked up by inputs, and whole-agent runs are shared by
    # inputs, so a scripted request repeated anywhere would run on the wrong
    # tools in one of the two places.
    # The regression sets share requests on purpose; only scripted ones must not.
    where: dict[str, list[str]] = {}
    for stem, dataset in DATASETS.items():
        for example in dataset["examples"]:
            key = json.dumps(example["inputs"], sort_keys=True)
            where.setdefault(key, []).append(f"{stem}/{example['key']}")
    for param in SCRIPTED:
        example = param.values[0]
        places = where[json.dumps(example["inputs"], sort_keys=True)]
        assert len(places) == 1, places


@pytest.mark.parametrize(
    ("query", "answered"),
    [
        # Queries from the real Amsterdam run the outage cut short: the ones
        # that failed there must fail here, and the restaurant one succeed.
        ("best vegetarian vegan restaurants Amsterdam wheelchair accessible step-free", True),
        ("Amsterdam August weather average temperature rain crowds", False),
        ("ARTIS zoo wheelchair accessible; A'DAM Lookout lift; Vondelpark wheelchair paths", False),
        ("USD to EUR exchange rate today", False),
        ("GVB wheelchair accessible tram metro Amsterdam low-floor lifts", False),
        ("Rijksmuseum Van Gogh Museum ticket prices wheelchair accessible", True),
    ],
)
def test_the_amsterdam_script_answers_only_what_it_means_to(query, answered):
    example = next(
        e for e in _examples("final_response_scripted") if e["key"] == "amsterdam-partial-research"
    )
    scripted.register([SimpleNamespace(inputs=example["inputs"], metadata=example["metadata"])])
    try:
        with scripted.active(example["inputs"]):
            result = scripted.ScriptedSearch().invoke({"query": query})
    finally:
        scripted._SCRIPTS.clear()
    assert ("error" not in result) is answered, query

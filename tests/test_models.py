"""Per-subagent model wiring.

Every failure here is silent at runtime: a subagent whose name drifts from its
`SUBAGENT_MODELS` key simply inherits the main agent's model, and nothing
errors. These tests are the only place that drift is visible.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from travel_agent.agent import _chat_model
from travel_agent.config import DEFAULT_MAX_TOKENS, SUBAGENT_MODELS
from travel_agent.subagents import build_subagents


def test_every_configured_model_names_a_real_subagent():
    roster = {subagent["name"] for subagent in build_subagents([])}
    assert set(SUBAGENT_MODELS) == roster


def test_haiku_4_5_is_never_sent_an_effort():
    # Haiku 4.5 rejects `output_config.effort`; the request would 400.
    for name, spec in SUBAGENT_MODELS.items():
        if "claude-haiku-4-5" in spec.model:
            assert spec.effort is None, name


def test_the_analyst_request_carries_its_effort_and_no_sampling(
    monkeypatch: pytest.MonkeyPatch,
):
    # The built payload, not the call site: API version 2026-10-21 answers any
    # sampling parameter with a 400, and a dropped effort would raise nothing.
    # Built offline: constructing a client is not a call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    spec = SUBAGENT_MODELS["budget-analyst"]
    client = _chat_model(spec.model, spec.effort, DEFAULT_MAX_TOKENS)
    payload = client._get_request_payload("cost this trip")  # ty: ignore[unresolved-attribute]
    assert payload["model"] == spec.model.split(":", 1)[1]
    assert payload["output_config"]["effort"] == spec.effort
    assert not {"temperature", "top_p", "top_k"} & set(payload)


def test_a_supplied_model_is_attached_to_its_subagent_only():
    model = FakeListChatModel(responses=[])
    roster = build_subagents([], {"budget-analyst": model})
    models = {subagent["name"]: subagent.get("model") for subagent in roster}
    assert models == {
        "destination-researcher": None,
        "availability-scout": None,
        "budget-analyst": model,
    }


def test_no_mapping_leaves_every_subagent_inheriting():
    assert all("model" not in subagent for subagent in build_subagents([]))

"""The traveler profile reaches the system prompt without a tool call.

`prompts.py` tells the main agent the profile is already loaded and not to
`read_file` it. That is only true while three things agree: the `/memories/`
route, the store namespace, and `memory=[MEMORY_PATH]` on the agent. Break any
one and nothing errors — the agent is told not to look, and never meets the
traveler. These drive the real agent with a scripted model, so all three are
exercised together. No network, no model calls.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

import travel_agent.agent as agent_module
from travel_agent.prompts import MEMORY_PATH


class _ScriptedModel(GenericFakeChatModel):
    """Replays scripted replies and records the messages each call received."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):  # the replies already name their tools
        return self

    def _generate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return super()._generate(messages, *args, **kwargs)


@pytest.fixture
def run_turn(monkeypatch: pytest.MonkeyPatch):
    """Build one agent over a shared store; each call runs a turn on its own thread."""
    script: list[AIMessage] = []
    model = _ScriptedModel(messages=_replay(script))
    monkeypatch.setattr(agent_module, "_chat_model", lambda *a, **k: model)
    agent = agent_module.build_agent(checkpointer=MemorySaver(), store=InMemoryStore())

    def run(thread: str, *replies: AIMessage) -> str:
        script.extend(replies)
        start = len(model.calls)
        agent.invoke(
            {"messages": [{"role": "user", "content": "plan a trip"}]},
            {"configurable": {"thread_id": thread}},
        )
        return model.calls[start][0].text

    return run


def _replay(script: list[AIMessage]) -> Iterator[AIMessage]:
    # Endless on purpose: each turn appends its replies, and running out is an
    # IndexError naming the bug rather than a generator that quietly ended.
    while True:
        yield script.pop(0)


def test_no_profile_says_so_in_the_prompt(run_turn):
    system = run_turn("first", AIMessage(content="ok"))
    assert "<agent_memory>\n(No memory loaded)" in system


def test_a_profile_written_in_one_conversation_is_in_the_next_ones_prompt(run_turn):
    write = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"file_path": MEMORY_PATH, "content": "Prefers aisle seats."},
                "id": "w1",
            }
        ],
    )
    run_turn("first", write, AIMessage(content="noted"))

    system = run_turn("second", AIMessage(content="ok"))
    assert "Prefers aisle seats." in system

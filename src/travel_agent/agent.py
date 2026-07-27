"""Travel agent graph.

Two entry points:

- `graph` — module-level, loaded by `langgraph dev` / LangGraph Platform via
  `langgraph.json`. It deliberately has **no** checkpointer or store: the
  server supplies both, and `StateBackend()` / `StoreBackend()` pick them up
  from the LangGraph execution context at call time.
- `build_agent(...)` — for standalone use (the CLI, tests), where you pass
  persistence in yourself.
"""

from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from travel_agent.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from travel_agent.prompts import MAIN_AGENT_PROMPT, MEMORY_PATH
from travel_agent.subagents import build_subagents
from travel_agent.tools.availability import date_offset
from travel_agent.tools.budget import summarize_budget
from travel_agent.tools.search import build_search_tools

__all__ = ["DEFAULT_MODEL", "build_agent", "build_backend", "graph"]

# Files under /memories/ route to the store and survive across conversations;
# everything else lands in graph state and is scoped to one thread. Longest
# prefix wins, so /memories/ takes precedence over the default backend.
MEMORY_PREFIX = "/memories/"


def build_backend() -> CompositeBackend:
    """Ephemeral scratch space by default, persistent memory under /memories/."""
    return CompositeBackend(
        default=StateBackend(),
        routes={MEMORY_PREFIX: StoreBackend()},
    )


def build_agent(
    model: str = DEFAULT_MODEL,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """Build the travel agent.

    Args:
        model: Provider-prefixed model id passed to `init_chat_model`.
        checkpointer: Conversation persistence. Leave `None` under
            `langgraph dev` / LangGraph Platform — the server provides it.
        store: Cross-thread store backing `/memories/`. Leave `None` under
            `langgraph dev` / LangGraph Platform — the server provides it.
        max_tokens: Output ceiling. On Claude Opus 5 thinking is on by
            default and counts against this, so keep it generous.
    """
    search_tools = build_search_tools()
    return create_deep_agent(
        model=init_chat_model(model, max_tokens=max_tokens),
        tools=[*search_tools, summarize_budget, date_offset],
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=build_subagents(search_tools),
        backend=build_backend(),
        memory=[MEMORY_PATH],
        checkpointer=checkpointer,
        store=store,
        name="travel-agent",
    )


# Entry point for langgraph.json. Persistence is injected by the server.
graph = build_agent()

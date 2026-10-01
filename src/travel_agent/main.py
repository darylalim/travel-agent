"""Standalone CLI, for driving the agent without the LangGraph server.

Persistence here is in-process: traveler memory lives as long as the process
does. `langgraph dev` is the path that actually persists memory across runs —
use this for quick one-off checks and debugging.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from travel_agent.config import DEFAULT_EFFORT, DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a trip with the travel agent.")
    parser.add_argument(
        "prompt", nargs="*", help="Opening request; omit for an interactive session."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--thread", default="cli", help="Conversation thread id.")
    args = parser.parse_args(argv)

    load_dotenv()

    # Imported after load_dotenv(): travel_agent.agent builds the graph at
    # import time for langgraph.json, and that reads TAVILY_API_KEY to decide
    # whether search is available. Importing it at module scope would build
    # the agent against an unloaded environment, then build it again below.
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    from travel_agent.agent import build_agent

    # DEFAULT_EFFORT is tuned for DEFAULT_MODEL; any other model runs at its
    # own default, since not every model accepts effort (Haiku 4.5 rejects it).
    effort = DEFAULT_EFFORT if args.model == DEFAULT_MODEL else None
    agent = build_agent(
        args.model, effort=effort, checkpointer=MemorySaver(), store=InMemoryStore()
    )
    config = {"configurable": {"thread_id": args.thread}}

    print("Traveler memory is in-process only here; use `langgraph dev` to persist it.\n")

    opening = " ".join(args.prompt).strip()
    while True:
        if opening:
            user_input, opening = opening, ""
        else:
            try:
                user_input = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if user_input.lower() in {"exit", "quit"}:
                return 0
            if not user_input:
                continue

        print()
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            # `.text` is a `TextAccessor`, which subclasses `str` *and* stays
            # callable for backwards compatibility — so `callable(text)` is
            # always True and calling it takes the accessor deprecated in
            # langchain-core 1.0 and removed in 2.0. The value is already the
            # string, and it holds text blocks only, so thinking never prints.
            if text := getattr(chunk, "text", None):
                print(text, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    sys.exit(main())

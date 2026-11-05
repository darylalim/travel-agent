"""Standalone CLI, for driving the agent without the LangGraph server.

Trips and traveler memory are saved to the SQLite file in `persistence.py`, so
the profile outlives the process. Each run starts a new trip unless `--thread`
names an earlier one: a fixed default would resume the last conversation, and
its `/trip/*` files, without saying so.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv

from travel_agent.config import DEFAULT_EFFORT, DEFAULT_MODEL, TRACE_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a trip with the travel agent.")
    parser.add_argument(
        "prompt", nargs="*", help="Opening request; omit for an interactive session."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--thread", help="Thread id of a trip to resume (default: start a new trip)."
    )
    args = parser.parse_args(argv)

    load_dotenv()

    # Imported after load_dotenv(): travel_agent.agent builds the graph at
    # import time for langgraph.json, and that reads TAVILY_API_KEY to decide
    # whether search is available. Importing it at module scope would build
    # the agent against an unloaded environment, then build it again below.
    from travel_agent.agent import build_agent
    from travel_agent.persistence import database_path, open_storage

    # DEFAULT_EFFORT is tuned for DEFAULT_MODEL; any other model runs at its
    # own default, since not every model accepts effort (Haiku 4.5 rejects it).
    effort = DEFAULT_EFFORT if args.model == DEFAULT_MODEL else None
    checkpointer, store = open_storage()
    agent = build_agent(args.model, effort=effort, checkpointer=checkpointer, store=store)
    thread = args.thread or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread},
        "run_name": TRACE_NAME,
        "tags": ["cli"],
    }

    print(f"Saving to {database_path()}. Resume this trip with --thread {thread}\n")

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

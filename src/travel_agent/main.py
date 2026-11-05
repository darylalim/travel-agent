"""Standalone CLI, for driving the agent without the LangGraph server.

Trips and traveler memory are saved to the SQLite file in `persistence.py`, so
the profile outlives the process. Each run starts a new trip unless `--thread`
names an earlier one: a fixed default would resume the last conversation, and
its `/trip/*` files, without saying so.

A resumed trip tells the agent how old its prices are, as a reopened one does in
the UI. The UI reads that from its saved snapshot, which a CLI trip does not have,
so `prices_dated_on` reads it from the checkpoint history.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import StateSnapshot

from travel_agent import clock
from travel_agent.config import DEFAULT_EFFORT, DEFAULT_MODEL, TRACE_NAME
from travel_agent.prompts import resumed_message

# The one subagent holding the search tools. `test_persistence.py` pins the name
# to the roster, because a stale one finds no search and dates every trip by its
# last checkpoint.
SCOUT = "availability-scout"


def _scout_results(messages: Sequence[BaseMessage]) -> set[str]:
    """The ids of the scout calls that have returned."""
    calls = {
        call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["args"].get("subagent_type") == SCOUT
    }
    return {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id in calls
    }


def prices_dated_on(history: Iterable[StateSnapshot]) -> str | None:
    """The newest date any price in a trip can be from, or None for a new trip.

    `history` is newest first. The last checkpoint is the wrong date, because a
    later turn with no search would make old fares look newer than they are. So
    this finds where the last scout result first appeared. A trip with no search
    falls back to the last checkpoint, which is still a true upper bound.

    A checkpoint stamps UTC, and the note is read in local dates.
    """
    snapshots = list(history)
    if not snapshots:
        return None
    dated_at = snapshots[0].created_at
    seen: set[str] = set()
    for snapshot in reversed(snapshots):
        results = _scout_results(snapshot.values.get("messages", []))
        if results - seen:
            dated_at = snapshot.created_at
        seen |= results
    if dated_at is None:
        return None
    return clock.long_date(datetime.fromisoformat(dated_at).astimezone().date())


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

    # An id with no history is a new trip, with nothing old to warn about.
    resume_note_on = prices_dated_on(agent.get_state_history(config)) if args.thread else None

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
        content = resumed_message(user_input, resume_note_on) if resume_note_on else user_input
        for chunk, _ in agent.stream(
            {"messages": [{"role": "user", "content": content}]},
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
        # Only after the stream ends: a turn that raises ends the process, and
        # the next resume sends the note again.
        resume_note_on = None
        print("\n")


if __name__ == "__main__":
    sys.exit(main())

"""Durable storage for the CLI and the Streamlit UI.

`langgraph dev` supplies its own checkpointer and store, so this is only for the
two entry points that wire theirs in-process. Evals and tests stay on the
in-memory pair on purpose: each example needs a store no other run wrote.

The default path hangs off the home directory rather than the working directory.
A relative path would give a different launch directory a new, empty database,
and a missing profile is skipped silently, so the only symptom would be an agent
that has never met the traveler.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

DB_ENV = "TRAVEL_AGENT_DB"

# A writer holds the file for milliseconds, but the CLI and Streamlit can share
# it, and sqlite3's 5s default turns a brief overlap into a failed turn.
_BUSY_TIMEOUT_SECONDS = 30


def database_path() -> Path:
    """Where trips and the traveler profile live.

    Read at call time, because the environment file loads after this module may
    be imported. A blank value counts as unset, since `Path("")` is the working
    directory.
    """
    configured = os.getenv(DB_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".travel-agent" / "travel-agent.sqlite"


def _connect(**kwargs: Any) -> sqlite3.Connection:
    """Open a connection that lasts the life of the process.

    Not `from_conn_string`: the saver's and the store's are both context
    managers that close on exit, and the Streamlit agent outlives any block it
    could sit in. `check_same_thread=False` is what lets LangGraph's worker
    threads share the connection.

    The file holds traveler profiles and trips, so it starts private to the
    owner. SQLite gives its `-wal` and `-shm` files the same mode.
    """
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.touch(mode=0o600, exist_ok=True)
    return sqlite3.connect(path, check_same_thread=False, timeout=_BUSY_TIMEOUT_SECONDS, **kwargs)


def open_store() -> SqliteStore:
    """Open the store behind `/memories/`, which every entry point here shares."""
    store = SqliteStore(_connect(isolation_level=None))  # the store requires autocommit
    # The saver sets itself up on first use; the store does not, and fails on
    # its first query without this.
    store.setup()
    return store


def open_storage() -> tuple[SqliteSaver, SqliteStore]:
    """Open both, for an entry point that can resume a trip.

    Each gets its own connection because each holds its own lock.
    """
    return SqliteSaver(_connect()), open_store()

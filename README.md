# travel-agent

Travel agent built on Deep Agents.

Researches destinations, gathers flight and lodging options, builds a
day-by-day itinerary, costs it against a budget, and remembers traveler
preferences across conversations.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

Only `ANTHROPIC_API_KEY` is required. Without `TAVILY_API_KEY` the agent still
plans, costs, and writes itineraries — it just cannot research anything on the
web.

## Run

```bash
uv run langgraph dev          # LangGraph Studio at :2024 — the main way to run it
uv run python -m travel_agent.main "5 days in Kyoto in September, 2 people, $4000"
```

`langgraph dev` is the recommended path: the server supplies the checkpointer
and store, so traveler memory persists across runs and you get a visual trace
of every subagent call. The CLI keeps state in-process only.

## How it works

`create_deep_agent` supplies planning (`write_todos`), a filesystem, and
delegation (`task`) out of the box. This project adds:

| Piece | Where | What it does |
|---|---|---|
| System prompts | `prompts.py` | Workspace layout, memory contract, delegation rules |
| Subagents | `subagents.py` | `destination-researcher`, `availability-scout`, `budget-analyst` |
| Web search | `tools/search.py` | Tavily, optional |
| Flights & lodging | `tools/availability.py` | Provider seam — see below |
| Budget math | `tools/budget.py` | Pure arithmetic, so the model never adds up a column itself |
| Graph | `agent.py` | Wires the above together; exports `graph` for `langgraph.json` |

### Storage

A `CompositeBackend` splits the agent's filesystem in two:

- `/trip/*` → `StateBackend`, scoped to one conversation. Scratch space for
  the brief, research notes, options, and the itinerary.
- `/memories/*` → `StoreBackend`, shared across conversations. Holds
  `traveler_profile.md`, which is loaded into the system prompt at startup via
  `memory=[...]` and is safely skipped on the first run when it does not exist
  yet.

Both backends resolve state and store from the LangGraph execution context, so
`agent.py` passes **no** checkpointer or store into the graph the server loads.
`build_agent()` accepts them for standalone use.

### ⚠️ Flights and lodging return sample data

There is no booking API wired up. `tools/availability.py` defines an
`AvailabilityProvider` protocol and ships `SampleProvider`, which returns
deterministic synthetic offers labelled `source: "sample-data"`, alongside an
explicit warning on every response. The system prompts require the agent to
pass that caveat through to the traveler, so it will not present these numbers
as real prices or bookable availability.

To go live, implement the protocol against Amadeus, Duffel, or similar,
register it in `_PROVIDERS`, and point `TRAVEL_AGENT_PROVIDER` at it. Setting
`source` to the real provider name makes the sample-data caveat drop out of the
agent's replies automatically.

## Develop

```bash
uv run pytest        # tool logic; no model calls, no network
uv run ruff check .
uv run ruff format .
```

## Stack

`deepagents` 0.6.12 · `langchain` 1.3+ · `langgraph` 1.2+ ·
`claude-opus-5` via `langchain-anthropic`. Requires Python 3.11+.

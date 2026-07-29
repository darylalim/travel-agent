# travel-agent

[![CI](https://github.com/darylalim/travel-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/travel-agent/actions/workflows/ci.yml)

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
uv run streamlit run streamlit_app.py          # browser UI at :8501
uv run python -m travel_agent.main "5 days in Kyoto in September, 2 people, $4000"
```

`langgraph dev` is the recommended path: the server supplies the checkpointer
and store, so traveler memory persists across runs and you get a visual trace
of every subagent call. The CLI keeps state in-process only.

The Streamlit UI sits between the two. It wires persistence in-process like
the CLI, but holds the agent in `st.cache_resource`, which is scoped to the
server process rather than the session — so the traveler profile under
`/memories/` survives a page reload and a second browser tab, and only dies
when you restart the server.

### The browser UI

Two pages, wired with `st.navigation`:

- **Plan** — chat with the agent. Its prose streams token by token; tool calls
  appear as a collapsed activity trail beside it, so you can watch it delegate
  to `availability-scout` without the subagent's own chatter filling the
  transcript.
- **Trip** — the same trip as figures: budget KPIs against the ceiling, cost
  by category, flight and lodging comparison tables, and the itinerary. Each
  table names the query behind it, and when the scout tried several — nearby
  airports, dates shifted a day — you get a control to switch between them.
  They are deliberately not merged into one table: those are different
  questions, and a combined list sorted by price would rank a cheaper flight
  on other dates above a dearer one on the dates you actually asked for.

The Trip page reads the **structured tool payloads**, not the markdown the
agent wrote, so `synthetic` and `warning` reach the screen as data. Every
synthetic result is labelled in the table and repeated as a standing notice at
the top of the page — a caveat that only existed in chat scrollback would be
too easy to scroll past.

Getting those payloads takes some care. deepagents folds a subagent's state
back into the parent *without* its `messages`, substituting a single
`ToolMessage` holding only the subagent's closing prose. Since `search_flights`
and `search_stays` are bound only to `availability-scout`, reading message
history after a run finds no search results at all. The UI therefore captures
them mid-stream, with `subgraphs=True` so nested messages are emitted in the
first place. `tests/test_ui.py` pins the behaviour.

## How it works

`create_deep_agent` supplies planning (`write_todos`), a filesystem, and
delegation (`task`) out of the box. This project adds:

| Piece | Where | What it does |
|---|---|---|
| System prompts | `prompts.py` | Workspace layout, memory contract, delegation rules |
| Subagents | `subagents.py` | `destination-researcher`, `availability-scout`, `budget-analyst` |
| Web search | `tools/search.py` | Tavily, optional |
| Flights & lodging | `tools/availability.py`, `tools/duffel.py` | Provider seam — see below |
| Budget math | `tools/budget.py` | Pure arithmetic, so the model never adds up a column itself |
| Graph | `agent.py` | Wires the above together; exports `graph` for `langgraph.json` |
| Browser UI | `streamlit_app.py`, `app_pages/` | Two-page Streamlit app; theme in `.streamlit/config.toml` |
| UI runtime | `ui.py` | Cached agent, turn streaming, and the state readback the pages use |

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

### Flight and lodging sources

`tools/availability.py` defines an `AvailabilityProvider` protocol. Pick one
with `TRAVEL_AGENT_PROVIDER`:

| Value | Flights | Lodging |
|---|---|---|
| `sample-data` (default) | Synthetic | Synthetic |
| `duffel` + test token | **Synthetic** (Duffel test inventory) | Synthetic |
| `duffel` + live token | **Real** | Synthetic |

The safety property: **the traveler is never shown synthetic inventory
described as real.** Every offer carries `synthetic: bool`, and a response
carrying any synthetic offers gets a `warning`. The prompts tell the agent to
key off those two fields.

Note the middle row. A Duffel *test* token returns fictional airlines at
invented fares — synthetic despite coming from a live API over the network, so
it is labelled exactly like sample data. Judging by the provider name would
get this wrong in two different ways at once: test-mode flights would look
real, and lodging under `duffel` would too.

An empty result set is warned about as well, since providers declare
synthetic-ness up front. Otherwise "no offers" from a synthetic source would
read as "we checked real inventory and found nothing available".

Sample offers are deterministic — seeded off the query, so prices don't drift
mid-conversation.

**Duffel is read-only.** It creates offer requests and reads offers back; it
never calls `POST /air/orders`, so nothing is booked and no payment is taken,
on a test token or a live one. Adding booking would mean putting order
creation behind Deep Agents' `interrupt_on` human-approval gate — that is a
deliberate decision, not a config change.

To use it: create a test token in the Duffel dashboard under *Developer test
mode* (`duffel_test_…`), set `DUFFEL_API_TOKEN` and
`TRAVEL_AGENT_PROVIDER=duffel` in `.env`. Test-mode results are fictional and
are labelled as such — see the table above.

Two things worth knowing:

- **Offers expire, usually in minutes.** Each carries `expires_at` and
  `expires_in_seconds`, and the prompts tell the agent to re-search rather than
  quote a stale offer.
- **Results are capped at 20 offers.** One real test-mode SFO→NRT search
  returned **630** — handing them all to the model would cost a context window.
  The trim keeps the cheapest, the fastest, and the fewest-stops options, so a
  nonstop still survives when the cheapest fares are all multi-stop. Responses
  carry `total_found`, `count`, and a `truncated` note.
- **Duffel Stays is not wired up**, so lodging still returns sample data. Cabin
  selection is likewise out of scope — every search is economy.

We call Duffel's REST API over `httpx` rather than the `duffel-api` PyPI
package, which was last released in 2023, is classified Alpha, and would add a
`requests` dependency. `tests/test_duffel.py` verifies the request shape
against `httpx.MockTransport`, and `tests/conftest.py` clears the provider env
vars for every test — so the suite needs no token and never reaches the
network, whatever you have exported.

To add another provider (Amadeus, Skyscanner, Duffel Stays): implement the
protocol, register a factory in `_PROVIDERS`, and set `source` to the provider
name so the sample-data caveat drops out of the agent's replies automatically.

## Develop

```bash
uv run pytest        # tool logic; no model calls, no network
uv run ruff check .
uv run ruff format .
```

## Stack

`deepagents` 0.6.12 · `langchain` 1.3+ · `langgraph` 1.2+ ·
`claude-opus-5` via `langchain-anthropic` · Duffel API `v2` over `httpx` ·
`streamlit` 1.60+ for the browser UI. Requires Python 3.11+.

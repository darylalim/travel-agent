# Travel Agent

[![CI](https://github.com/darylalim/travel-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/travel-agent/actions/workflows/ci.yml)

Travel planning agent on Deep Agents. Research destinations and search flight and lodging inventory via Duffel.

It gathers options, builds a day-by-day itinerary, costs it against a budget,
and remembers traveler preferences across conversations.

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
uv run langgraph dev                     # LangGraph Studio at :2024 — the main way to run it
uv run streamlit run streamlit_app.py    # browser UI at :8501
uv run python -m travel_agent.main "5 days in Kyoto in September, 2 people, $4000"
```

The three differ in more than ergonomics — each wires persistence differently:

| Entry point | Checkpointer and store | `/memories/` survives |
|---|---|---|
| `langgraph dev` | Supplied by the server | Across runs |
| `streamlit run` | In-process, cached process-wide | A page reload and a second tab, but not a server restart |
| `python -m travel_agent.main` | In-process | Nothing — it dies with the process |

`langgraph dev` is the recommended path: memory persists and you get a visual
trace of every subagent call. The middle row follows from `st.cache_resource`
being scoped to the **server process** rather than the session.

### The browser UI

Two pages, wired with `st.navigation`:

- **Plan** — chat with the agent. Its prose streams token by token; tool calls
  appear as a collapsed activity trail beside it, so you can watch it delegate
  to `availability-scout` without the subagent's own chatter filling the
  transcript.
- **Trip** — the same trip as figures: budget KPIs against the ceiling, cost by
  category, flight and lodging comparison tables, and the itinerary. Each table
  names the query behind it, and when the scout tried several — nearby airports,
  dates shifted a day — you get a control to switch between them. They are
  deliberately not merged: those are different questions, and a combined list
  sorted by price would rank a cheaper flight on other dates above a dearer one
  on the dates you actually asked for.

The Trip page reads the **structured tool payloads**, not the markdown the agent
wrote, so `synthetic` and `warning` reach the screen as data. Every synthetic
result is labelled in the table and repeated as a standing notice at the top of
the page — a caveat that only existed in chat scrollback would be too easy to
scroll past.

Getting those payloads takes care. deepagents folds a subagent's state back into
the parent *without* its `messages`, substituting a single `ToolMessage` holding
only the subagent's closing prose. Since `search_flights` and `search_stays` are
bound only to `availability-scout`, reading message history after a run finds no
search results at all. The UI therefore captures them mid-stream, with
`subgraphs=True` so nested messages are emitted in the first place.
`tests/test_ui.py` pins the behaviour.

## How it works

`create_deep_agent` supplies a filesystem and delegation (`task`) out of the
box. This project adds:

| Piece | Where | What it does |
|---|---|---|
| System prompts | `prompts.py` | Workspace layout, memory contract, delegation rules |
| Subagents | `subagents.py` | `destination-researcher`, `availability-scout`, `budget-analyst` |
| Web search | `tools/search.py` | Tavily, optional |
| Flights & lodging | `tools/availability.py`, `tools/duffel.py`, `tools/duffel_stays.py` | Provider seam — see below |
| Budget math | `tools/budget.py` | Pure arithmetic, so the model never adds up a column itself |
| Graph | `agent.py` | Wires the above together; exports `graph` for `langgraph.json` |
| Browser UI | `streamlit_app.py`, `app_pages/` | Two-page Streamlit app; theme in `.streamlit/config.toml` |
| UI runtime | `ui.py` | Cached agent, turn streaming, and the state readback the pages use |

### Storage

A `CompositeBackend` splits the agent's filesystem in two:

- `/trip/*` → `StateBackend`, scoped to one conversation. Scratch space for the
  brief, research notes, options, and the itinerary.
- `/memories/*` → `StoreBackend`, shared across conversations. Holds
  `traveler_profile.md`, loaded into the system prompt at startup via
  `memory=[...]` and safely skipped on the first run when it does not exist yet.

Both backends resolve state and store from the LangGraph execution context, so
`agent.py` passes **no** checkpointer or store into the graph the server loads.
`build_agent()` accepts them for standalone use.

### Flight and lodging sources

`tools/availability.py` defines an `AvailabilityProvider` protocol. Pick one
with `TRAVEL_AGENT_PROVIDER`:

| Value | Flights | Lodging |
|---|---|---|
| `sample-data` (default) | Synthetic | Synthetic |
| `duffel` + test token | **Synthetic** (Duffel test inventory) | Synthetic (sample data) |
| `duffel` + live token | **Real** | **Real** (Duffel Stays) |

The safety property: **the traveler is never shown synthetic inventory
described as real.** Every offer carries `synthetic: bool`, and a response
carrying any synthetic offers gets a `warning`. The prompts tell the agent to
key off those two fields.

Note the middle row. A Duffel *test* token returns fictional airlines at
invented fares — synthetic despite coming from a live API over the network, so
it is labelled exactly like sample data. Judging by the provider name would get
this wrong in both directions: test-mode flights under `duffel` would look real,
and real lodging under `duffel` would look synthetic if the rule were "lodging is
always sample data".

Lodging is the row that moves. Duffel's Stays test inventory exists at a single
coordinate pair, so a test-mode search of an actual city returns nothing at
all — not an error, just an empty list. Sample data answers the question the
traveler asked, so a test token routes lodging there and says so; a live token
searches Duffel Stays for real.

An empty result set is warned about as well, since providers declare
synthetic-ness up front — otherwise "no offers" from a synthetic source would
read as "we checked real inventory and found nothing available". Sample offers
are deterministic, seeded off the query, so prices don't drift mid-conversation.

To enable Duffel: create a test token in the Duffel dashboard under *Developer
test mode* (`duffel_test_…`), then set `DUFFEL_API_TOKEN` and
`TRAVEL_AGENT_PROVIDER=duffel` in `.env`. Test-mode results are fictional and
are labelled as such — see the table above.

**Duffel is read-only.** It creates offer requests and searches and reads the
results back; it never calls `POST /air/orders` or `POST /stays/bookings`, and
never creates a quote, so nothing is booked and no payment is taken, on a test
token or a live one. Adding booking would mean putting order creation behind
Deep Agents' `interrupt_on` human-approval gate — a deliberate decision, not a
config change.

The verb is not the discriminator: `/stays/search` is itself a `POST`, and it is
the one call the lodging path is built to make. So `.claude/hooks/no-booking.sh`
matches on the **path**, and only where one actually constructs a request URL —
a bare `"/stays/bookings"` in a docstring or a test assertion stays allowed,
because the hook must never block testing the invariant it protects.

A few things worth knowing:

- **Offers expire, usually in minutes.** Each carries `expires_at` and
  `expires_in_seconds`, and the prompts tell the agent to re-search rather than
  quote a stale offer.
- **Results are capped at 20 offers.** One real test-mode SFO→NRT search
  returned **630** — handing them all to the model would cost a context window.
  The trim keeps the cheapest, the fastest, and the fewest-stops options, so a
  nonstop still survives when the cheapest fares are all multi-stop. Responses
  carry `total_found`, `count`, and a `truncated` note.
- **Live lodging search covers a fixed list of cities.** Duffel Stays searches
  by latitude and longitude — there is no city-name form, and Duffel's own guide
  says you will "probably need to use a geocoding service". Rather than take that
  dependency and another API key, the provider ships a static table and refuses
  what it does not know, naming the cities that do work. A nearest-match guess
  would label results with a city nobody asked for, which is far harder to
  notice than an error.
- **A stay's two prices are reported, never summed.** Duffel's docs disagree
  about whether `due_at_accommodation_amount` sits inside `total_amount` or on
  top of it, so both are surfaced as they arrive. Adding them double-counts under
  one reading and subtracting understates under the other, and either way the
  traveler would be shown a confident number nobody can source.
- **Cabin class is validated at the tool layer, never guessed.** It defaults to
  economy; "Premium Economy" and `premium-economy` both land on
  `premium_economy`, while anything unrecognised is an error naming the four that
  work — coercing "biz" to business commits the traveler to a fare several times
  the one they meant.
- **Cabin is a preference rather than a filter.** A search can succeed without
  returning what it asked for, so the response echoes `requested_cabin`, each
  offer carries the cabin it actually came back in (`mixed` when one itinerary's
  legs disagree), and a `cabin_note` appears when those differ. That note is
  deliberately separate from `warning`: a real fare in the wrong cabin is not the
  same problem as a fictional one, and a live search with no honesty caveat to
  raise must not borrow that banner. The 20-offer trim selects the requested
  cabin first for the same reason — a lower cabin is always cheaper, so trimming
  by price alone would answer a business search with twenty economy fares and
  raise nothing.
- **Multi-room lodging search is out of scope** — one room, all guests.

Duffel's REST API is called over `httpx` directly rather than through the
`duffel-api` PyPI package, which was last released in 2023, is classified Alpha,
and would add a `requests` dependency. `tests/test_duffel.py` verifies the
request shape against `httpx.MockTransport`, and `tests/conftest.py` clears the
provider env vars for every test — so the suite needs no token and never reaches
the network, whatever you have exported.

To add another provider (Amadeus, Skyscanner, Booking.com): implement the
protocol, register a factory in `_PROVIDERS`, and set `source` to the provider
name so the sample-data caveat drops out of the agent's replies automatically.

## Develop

```bash
uv run pytest                     # tool logic; no model calls, no network
uv run ruff check .
uv run ruff format .
uv run ty check
bash .claude/hooks/test-hooks.sh  # the Claude Code hooks under .claude/
```

CI runs the same checks on push and pull request, with tests across Python
3.11–3.13 and the hook suite on Linux and macOS.

## Stack

`deepagents` 0.7.x · `langchain` 1.3+ · `langgraph` 1.2+ ·
`claude-opus-5` via `langchain-anthropic` · Duffel API `v2` over `httpx` ·
`streamlit` 1.60+ for the browser UI. Requires Python 3.11+.

## License

MIT — see [LICENSE](LICENSE). The stack it builds on is permissive throughout:
`deepagents`, `langchain` and `langgraph` are MIT, `httpx` and `python-dotenv`
BSD-3-Clause, `streamlit` Apache-2.0.

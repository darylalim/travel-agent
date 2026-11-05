# Travel Agent

[![CI](https://github.com/darylalim/travel-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/travel-agent/actions/workflows/ci.yml)

Multi-agent travel planner on Deep Agents. Researches destinations, searches flights and stays via Duffel, and builds a costed day-by-day itinerary. Planning only: it never books.

It also remembers traveler preferences across conversations.

Two things it will not do:

- **It never books.** It searches and reads, and takes no payment, on a test
  token or a live one.
- **It never shows sample data as real.** Every offer says whether it is
  synthetic, and the chat and the Trip page both carry that label.

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
uv run python -m travel_agent.main '5 days in Kyoto in September, 2 people, $4000'
```

Each wires persistence differently:

| Entry point | Checkpointer and store | `/memories/` survives |
|---|---|---|
| `langgraph dev` | Supplied by the server | Across runs |
| `streamlit run` | SQLite file | Across runs |
| `python -m travel_agent.main` | SQLite file | Across runs |

`langgraph dev` is the recommended path, because you get a visual trace of every
subagent call. The CLI and the browser UI share one file, by default
`~/.travel-agent/travel-agent.sqlite`; set `TRAVEL_AGENT_DB` to move it. It holds
your traveler profile and your trips unencrypted, and its profile is separate
from the one `langgraph dev` keeps.

Each CLI run starts a new trip and prints a thread id. Pass it back with
`--thread` to pick that trip up again. The browser UI saves each trip after every
reply and lists them on its Past trips page. CLI trips are not listed there,
because the CLI does not capture the search results that page restores.

### The browser UI

Three pages:

- **Plan:** chat with the agent. Its prose streams in, and tool calls appear as a
  collapsed activity trail beside it.
- **Trip:** the same trip as figures: budget against the ceiling, cost by
  category, flight and lodging tables, and the itinerary. When the scout tried
  several queries, such as nearby airports or shifted dates, a control switches
  between them. They are not merged, because a combined list sorted by price
  would rank a cheaper flight on other dates above the dates you asked for.
- **Past trips:** reopen an earlier trip and carry on planning. Its chat and
  figures come back as they were saved, and the Trip page says that those prices
  date from then. You can also delete a trip here, apart from the one that is open.

Every synthetic result is labelled in its table and repeated as a standing notice
at the top of the Trip page.

### Tracing

Set `LANGSMITH_API_KEY` in `.env` and every entry point traces to LangSmith, with
each subagent call nested under the main agent's. `.env.example` already sets
`LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=travel-agent`.

A trace records full inputs and outputs: the traveler profile, the prompts and
every offer payload, including live Duffel fares. Set `LANGSMITH_TRACING=false` if
that should not leave the machine. The test suite never traces.

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

## Flight and lodging sources

`tools/availability.py` defines an `AvailabilityProvider` protocol. Pick one
with `TRAVEL_AGENT_PROVIDER`:

| Value | Flights | Lodging |
|---|---|---|
| `sample-data` (default) | Synthetic | Synthetic |
| `duffel` + test token | **Synthetic** (Duffel test inventory) | Synthetic (sample data) |
| `duffel` + live token | **Real** | **Real** (Duffel Stays) |

Every offer carries `synthetic: bool`, and a response carrying any synthetic
offers gets a `warning`. The prompts tell the agent to key off those two fields,
never the provider name. A Duffel *test* token returns fictional airlines at
invented fares from a live API, so it is labelled exactly like sample data, and
a live token returns real inventory under the same provider name.

Duffel's Stays test inventory sits at one coordinate pair, so a test token routes
lodging to sample data and says so. An empty result is warned about too, or "no
offers" from a synthetic source would read as "we checked real inventory".

To enable Duffel: create a test token in the Duffel dashboard under *Developer
test mode* (`duffel_test_…`), then set `DUFFEL_API_TOKEN` and
`TRAVEL_AGENT_PROVIDER=duffel` in `.env`. Test-mode results are fictional and
are labelled as such — see the table above.

**Duffel is read-only.** It never calls `POST /air/orders` or
`POST /stays/bookings` and never creates a quote. Adding booking would mean
putting order creation behind Deep Agents' `interrupt_on` human-approval gate, a
deliberate decision rather than a config change. `tests/test_no_booking.py`
enforces this.

A few things worth knowing:

- **Offers expire, usually in minutes.** Each carries `expires_at` and
  `expires_in_seconds`, and the agent re-searches rather than quote a stale one.
- **Results are capped at 20 offers.** The trim keeps the cheapest, the fastest
  and the fewest-stops options, so a nonstop survives when the cheapest fares are
  all multi-stop. Responses carry `total_found`, `count` and a `truncated` note.
- **Live lodging search covers a fixed list of cities.** Duffel Stays searches by
  coordinates, so the provider ships a static table and refuses a city it does
  not know, naming the ones that work, rather than guess the nearest.
- **A stay's two prices are reported, never summed.** Duffel's docs disagree about
  whether `due_at_accommodation_amount` sits inside `total_amount`.
- **Cabin is a preference rather than a filter.** It defaults to economy, each
  offer carries the cabin it came back in, and a `cabin_note` flags a mismatch.
- **Multi-room lodging search is out of scope:** one room, all guests.

Duffel's REST API is called over `httpx` directly, because the `duffel-api` PyPI
package was last released in 2023, is classified Alpha and would add a `requests`
dependency. The test suite needs no token and never reaches the network.

To add another provider (Amadeus, Skyscanner, Booking.com): implement the
protocol, register a factory in `_PROVIDERS`, and set `source` to the provider
name so the sample-data caveat drops out of the agent's replies automatically.

## Evaluations

`evals/` holds nine LangSmith datasets, and `evals/run.py` scores them against
the real models. They need `LANGSMITH_API_KEY`, and they spend real tokens.

| Sets | What they are for |
|---|---|
| Regression: `final_response`, `trajectory`, `availability_scout`, `budget_analyst` | Catching breakage in replies, delegation and each subagent |
| `_hard` | Briefs a plausible agent gets wrong, run with `--repetitions` for a pass rate |
| `_scripted` | Inventory and research sample data never produces, such as a cabin downgrade or partly failed research |

```bash
uv run python -m evals.upload --dry-run                          # validate the datasets, no keys needed
uv run python -m evals.upload                                    # upload any not already in LangSmith
uv run python -m evals.run budget_analyst --limit 1 --no-judge   # cheapest real run, a few cents
uv run python -m evals.run trajectory --limit 1                  # one full trip, about $1
uv run python -m evals.run budget_analyst_hard --repetitions 3   # a pass rate, about $0.50
```

Pass `--no-web-search` if Tavily is out of credits. Each run also writes its
transcripts and scores to `evals/results/`, which is gitignored.

## Develop

```bash
uv run pytest                     # tool logic; no model calls, no network
uv run ruff check .
uv run ruff format .
uv run ty check
bash .claude/hooks/test-hooks.sh  # the Claude Code hooks under .claude/
```

CI runs the same checks on push and pull request, with tests on Python 3.11 and
3.13 — the two ends of the supported range — and the hook suite on Linux and
macOS.

### Releasing

Bump `[project] version` in `pyproject.toml` and push to `main`. Once the checks
pass, CI tags that commit `vX.Y.Z` and publishes a GitHub release whose notes are
the commit subjects since the previous tag. Nothing else is required, and nothing
is uploaded to a package index — the PyPI name `travel-agent` belongs to an
unrelated project, so a release is a tag and notes.

## Stack

`deepagents` 0.7.x · `langchain` 1.3+ · `langgraph` 1.2+ ·
`claude-opus-5-5` (main agent), `claude-sonnet-5-5` and `claude-haiku-5-5` (subagents) via `langchain-anthropic` · Duffel API `v2` over `httpx` ·
`streamlit` 1.60+ for the browser UI. Requires Python 3.11+.

## License

MIT — see [LICENSE](LICENSE). The stack it builds on is permissive throughout:
`deepagents`, `langchain` and `langgraph` are MIT, `httpx` and `python-dotenv`
BSD-3-Clause, `streamlit` Apache-2.0.

The dev environment is the exception. `langgraph-cli[inmem]`, which runs
`langgraph dev`, brings in `langgraph-api` and `langgraph-runtime-inmem`
(Elastic-2.0, which restricts offering the software to others as a hosted or
managed service) and `forbiddenfruit` (metadata lists both GPLv3+ and MIT). None
of them is a runtime dependency.

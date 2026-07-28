# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A travel planner built on Deep Agents. `README.md` explains what the system does
and why; this file covers what breaks when you edit it.

## Commands

```bash
uv sync                                  # installs the dev group too
uv run langgraph dev                     # LangGraph Studio at :2024 — main way to run it
uv run python -m travel_agent.main "5 days in Kyoto, 2 people, $4000"

uv run pytest                            # 47 tests, ~0.04s, no network, no model calls
uv run pytest tests/test_duffel.py::test_supplier_timeout_is_clamped_to_duffels_range
uv run ruff check . && uv run ruff format . && uv run ty check

bash .claude/hooks/test-hooks.sh         # 77 cases pinning the Claude Code hooks
```

The hooks in `.claude/` enforce parts of this file mechanically: the data-honesty
co-change rule, Duffel's read-only constraint, `_PROVIDER_ENV` coverage, and the
`.env` credentials. They are shell regexes whose character classes and verb lists
are load-bearing — an adversarial review of the first draft found thirteen real
defects, one of which deleted imports Claude had just written. `test-hooks.sh`
pins every fix, so run it after touching a hook.

When working with Python, invoke the relevant `/astral:<skill>` (from the
`astral-sh/astral` plugin) for `uv`, `ty`, and `ruff` to ensure best practices
are followed.

`langgraph dev` and the CLI differ in more than ergonomics: the server supplies
the checkpointer and store, so `/memories/` persists across runs. The CLI wires
`MemorySaver`/`InMemoryStore` in-process, so traveler memory dies with the
process.

Three places pin 3.11 and have to stay in agreement: `[tool.ruff]
target-version`, `[tool.ty.environment] python-version`, and `langgraph.json`'s
`python_version`. So avoid 3.12+ syntax even though the local `.venv` may be
newer. The ty pin is explicit rather than inferred from `requires-python`,
which is deliberately looser (`<4.0`) for packaging.

## The invariant everything else serves

**The traveler is never shown synthetic inventory described as real.** Both
post-scaffold commits were data-honesty fixes, and each had to touch
`availability.py`, `duffel.py`, and the tests together — the property cannot be
maintained in one place. Four enforcement points must stay in agreement:

1. Every offer carries `synthetic: bool` alongside `source`.
2. Providers implement `synthetic_note(kind)`, consulted **even when a search
   returns nothing** — otherwise an empty result reads as "we checked real
   inventory and found none."
3. `_wrap()` in `availability.py` attaches the resulting `warning`.
4. The `@tool` docstrings and `prompts.py` tell the model to key off
   `warning`/`synthetic` — **never the provider name**.

Point 4 is the subtle one. `synthetic` is not a property of the provider:
Duffel in *test mode* returns fictional fares, and Duffel *lodging* falls
through to `SampleProvider`. Both are synthetic under a provider named
`duffel`. Judging by provider name gets this wrong in two directions at once.

When you change an offer's shape, a provider, or a warning, update the tool
docstring in the same edit — the docstring is the model's only interface to
that contract, and prose in `prompts.py` may restate it.

## Import-time graph construction

`langgraph.json` needs a module-level `graph`, so `agent.py:77` runs
`build_agent()` **as an import side effect** — constructing a model client and
the full tool stack. Two things exist solely because of this:

- `config.py` holds `DEFAULT_MODEL` / `DEFAULT_MAX_TOKENS` so callers wanting a
  constant don't build an agent. Put new shared constants here, not in `agent.py`.
- `main.py` defers importing `travel_agent.agent` until **after** `load_dotenv()`
  (see its inline comment), because graph construction reads `TAVILY_API_KEY` to
  decide whether search exists. Hoisting that import to module scope builds the
  agent against an empty environment.

The module-level `graph` deliberately passes **no** checkpointer or store —
`StateBackend`/`StoreBackend` resolve them from the LangGraph execution context
at call time. `build_agent()` accepts them only for standalone use.

## Storage routing

`CompositeBackend` splits the agent's filesystem by path prefix, longest match
winning:

- `/trip/*` → `StateBackend`, scoped to one conversation (brief, research,
  options, budget, itinerary).
- `/memories/*` → `StoreBackend`, shared across conversations. Holds
  `traveler_profile.md`, loaded into the system prompt via `memory=[MEMORY_PATH]`
  and safely skipped when absent on a first run.

Both paths are string constants in `prompts.py` (`WORKSPACE`, `MEMORY_PATH`) and
are interpolated into prompt text — change them there, not inline.

## Provider seam

`AvailabilityProvider` is a `Protocol` in `availability.py`; `_PROVIDERS` maps
`TRAVEL_AGENT_PROVIDER` values to factories. To add one (Amadeus, Duffel Stays):
implement the protocol, register a factory, and set each offer's `source` to the
provider name.

- `_build_provider` is `@cache`d, so construction and its side effects (the
  live-token warning) happen once per process. Failed construction isn't cached,
  so a corrected token takes effect immediately.
- Date and traveler-count validation lives at the **tool** layer, not per
  provider, so behaviour is identical whichever provider is configured.
- Tools return `{"error": ...}` dicts rather than raising — the model reads them
  as text. `DuffelError` subclasses `ValueError` to ride the same path.
- `total_fare` is `None`, never `0`, when a price is unparseable; unpriced offers
  are dropped. A zero fare sorts to the front and gets recommended as cheapest.
- Results are capped at 20. One real test-mode SFO→NRT search returned 630.
  `_select_flights` keeps the cheapest, fastest, and fewest-stops offers so a
  nonstop survives when all the cheap fares are multi-stop.

**Duffel is read-only** — offer requests and reads, never `POST /air/orders`.
Adding booking means putting order creation behind Deep Agents' `interrupt_on`
human-approval gate; that is a deliberate design decision, not a config change.

Duffel v2 response shape, easy to get wrong: `live_mode` is top level;
`duration` sits on each **slice** (covering layovers — summing segments
understates multi-stop trips); `cabin_class` is nested at
`slices[].segments[].passengers[].cabin_class`, not on the offer.

## Tests

No network, no model calls, no token required. `httpx.MockTransport` verifies
request construction; `map_offer` is pure and tested against a payload shaped to
Duffel's documented schema.

`tests/conftest.py` has an **autouse** fixture that unsets provider env vars and
clears the `_build_provider` cache around every test. Without it the suite picks
up whatever a developer exported — and the combination the README tells you to
set sends real requests to `api.duffel.com`. **Add any new provider env var to
`_PROVIDER_ENV`.**

## Prompts

`prompts.py` is tuned for Claude Opus 5, which by default writes long responses,
self-verifies unasked, and expands scope. The prompts counter those tendencies —
no "double-check your work" scaffolding, explicit conciseness and scope
discipline. Don't add verification scaffolding back in.

Subagent `description` fields in `subagents.py` are what the main agent reads
when deciding to delegate, so they carry the "brief me completely in one call"
contract — subagents are stateless and cannot ask follow-up questions.

`DEFAULT_MAX_TOKENS` is generous (16k) because thinking is on by default on Opus 5
and counts against the ceiling.

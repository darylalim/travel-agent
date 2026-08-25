# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A travel planner built on Deep Agents. `README.md` explains what the system does
and why; this file covers what breaks when you edit it.

## Commands

```bash
uv sync                                  # installs the dev group too
uv run langgraph dev                     # LangGraph Studio at :2024 — main way to run it
uv run streamlit run streamlit_app.py    # browser UI at :8501
uv run python -m travel_agent.main "5 days in Kyoto, 2 people, $4000"

uv run pytest                            # 172 tests, ~4s, no network, no model calls
uv run pytest tests/test_duffel.py::test_supplier_timeout_is_clamped_to_duffels_range
uv run ruff check . && uv run ruff format . && uv run ty check

bash .claude/hooks/test-hooks.sh         # 61 cases pinning the Claude Code hooks
```

`ruff check --fix` is never run here, and `/astral:ruff` will suggest it. There
is no `[tool.ruff.lint] select` in `pyproject.toml`, so the default rule set
includes F401, whose "safe" fix deletes unused imports — including ones the edit
in progress just wrote. `ci.yml` runs `--check` only; `python-gate.sh` applies
`ruff format` and never `--fix`.

Three hooks run from `.claude/`. Two survive the cull below, for two different
reasons. `protect-env.sh` guards the credentials file: reading it copies live
tokens into the transcript, writing it can repoint the agent at real inventory
with nothing in the diff, and because the file is gitignored **no later gate can
see either**. `python-gate.sh` applies `ruff format` to the edited file, which is
the one thing CI structurally cannot do — `ci.yml` runs `--check` and can only
report. `turn-gate.sh` is the whole-project half split out of `python-gate.sh` by
the same commit; the split is described below.

That is the test for whether a hook belongs here: it prevents something
irreversible or invisible that no later gate catches. *Earlier* is not
*essential*. Three hooks that failed that test were removed, and the properties
they guarded moved into pytest, where they also run in CI on both ends of the
supported Python range and for contributors not using Claude Code:

| Was | Is now |
|---|---|
| `no-booking.sh` — regex over the edit text | `tests/test_no_booking.py` — AST walk over `src/` |
| `provider-env-drift.sh` — regex over `tools/` | `tests/test_env_isolation.py` — AST walk plus the imported tuple |
| `honesty-cochange.sh` — Stop hook | cut, not moved |

The co-change hook demanded a test change alongside every provider change. Its
predicate was satisfied by `864e44e` — the commit that actually shipped fictional
Duffel fares labelled `source: "duffel"` with no warning — so it could not have
caught the one incident it existed for. A proxy that the incident satisfies is
not a guard.

`protect-env.sh` is shell regexes whose character classes and verb lists are
load-bearing; `python-gate.sh` matches with `case` globs, and its hazard is the
one above — it must never run `ruff check --fix`. An adversarial review of the
first draft found thirteen real defects across the five hooks, one of which
deleted imports Claude had just written. `test-hooks.sh` pins every fix, so run
it after touching a hook.

The per-edit and per-turn gates split along one line: **per-file checks fire on
the edit, whole-project checks fire at the turn boundary.** `python-gate.sh`
runs `ruff format`, `ruff check` and `ty check` on the one file that changed
(~0.1s); `turn-gate.sh` runs project-wide `ty check` and the suite on `Stop` and
`SubagentStop` (~5s, and skipped entirely when no Python changed). Do not move
the suite back onto the edit. A change spanning `availability.py`, `duffel.py`
and `tests/` — the shape this file mandates for the data-honesty invariant — is
red at every intermediate edit, and a per-edit `exit 2` there says "fix this
before continuing" about a state that is merely unfinished. The cheapest way to
comply mid-refactor is to weaken the assertion.

`SubagentStop` is not optional: `Stop` does not fire for Task subagents, so
Python edited inside `availability-scout` would end its turn ungated.

`.github/workflows/ci.yml` runs the same commands on push and PR, plus two
checks with no local equivalent: it asserts the three-way 3.11 pin below, and
it imports the module-level `graph` — a path `pytest` deliberately never takes,
so an import-time break in `agent.py` surfaces there instead of in `langgraph
dev`. CI syncs with `uv sync --locked`, so editing a dependency without
re-locking fails it.

Three jobs, five runners, and each one answers a question no other job does.
Lint, types, the pin assertion and the graph import share a single 3.11 runner,
because their environments were byte-identical and the graph import's whole
point is the interpreter it runs on. Tests run on **3.11 and 3.13 only** — the
two ends of `requires-python`; nothing is in between, since `numpy` is the sole
package `uv.lock` forks (at 3.12) and both legs straddle it. The hook suite runs
on Linux *and* macOS, and the macOS leg is the load-bearing one: `protect-env.sh`
fails **open** on a grep error, so a GNU-only regex construct passes 61/61 on
ubuntu while disarming every deny case on the shell the hooks actually run in.

`cancel-in-progress` is conditional on the ref. Cancelling a superseded PR run
is the point; cancelling a run on `main` is not, because each commit there is a
permanent point in history and nothing recomputes a verdict it never got. The
`release` job below is the reason that stopped being merely untidy.

`astral-sh/setup-uv` is pinned to a **full version**, in all three jobs that use
it. It is not a style choice and `@v10` is not a shorter spelling of it: from v8
the action stopped publishing floating major tags altogether — a response to the
tj-actions supply-chain attack — so a major-only ref fails to resolve. Releases
from v8 on are immutable, which is what makes a version tag as firm as a commit
sha here. `actions/checkout@v5` still publishes a major tag and still runs on
node24, so it is left as it is.

### Releasing

A fourth job tags and publishes when `[project] version` changes on `main`. It
`needs` the other three, so a release is the one place the macOS hook leg's
verdict reaches someone who is not the author.

The gate is **level-triggered**: it asks `origin` whether `refs/tags/vX` exists
and never looks at what the push changed. A `HEAD^` diff is wrong here three
ways over — checkout fetches one commit, a push can carry several and the bump
need not be in the last one, and a re-run re-reads the same diff and tags twice.
Asking about the tag converges from any history shape, so a cancelled or failed
run is *completed* by the next push rather than duplicated or skipped.

Four things are load-bearing and none of them is obvious:

- **Tag and release are probed separately.** A run that tagged and then died
  leaves a tag with no release, and a single "already released?" gate calls that
  state finished forever. Two probes let the next push create only the missing
  half.
- **The version must match `\d+\.\d+\.\d+`.** Otherwise `0.2.0rc1` cuts a real,
  published, *latest* release, and there is no prerelease practice here to fall
  back on — the first rc would invent one by accident.
- **It refuses to go backwards.** A revert of the bump, a cherry-pick or a
  rewritten `main` can leave `pyproject.toml` below a version that already
  shipped. A release is not a state you can take back; a tag that moves detaches
  every clone.
- **The tag is pinned to `github.sha`,** not to a branch name. Letting it resolve
  to `main`'s head at API-call time points it at whatever landed since — a commit
  nothing verified.

`contents: write` is scoped to that job alone. The other three run `uv sync`,
which executes build hooks from several hundred third-party packages, and none
of them needs a writable token.

**There is no PyPI step, and adding one is not a config change.** The name
`travel-agent` is taken on PyPI by an unrelated project and `travel_agent`
normalises to the same one, so upload 403s and a Trusted Publisher cannot even
be created. Publishing would mean renaming the distribution — and the built
wheel is `src/travel_agent` only, so it carries `ui.py` but not
`streamlit_app.py`, `app_pages/`, `.streamlit/config.toml` or `langgraph.json`.
An installed release cannot run the UI. Nothing is attached to the release for
that reason.

Keep tagging and releasing in **one job**. A tag pushed with `GITHUB_TOKEN` does
not trigger `on: push: tags:`, so splitting this across two workflows gives you
a green run, a tag on origin, and no release anywhere — with nothing red to
notice.

When working with Python, invoke the relevant `/astral:<skill>` (from the
`astral-sh/astral` plugin) for `uv`, `ty`, and `ruff` to ensure best practices
are followed.

The three entry points differ in more than ergonomics. `langgraph dev` has the
server supply the checkpointer and store, so `/memories/` persists across runs.
The CLI wires `MemorySaver`/`InMemoryStore` in-process, so traveler memory dies
with the process. The Streamlit UI wires them in-process too, but holds the
agent in `st.cache_resource`, which is scoped to the **server process** rather
than the session — so memory survives a page reload and a second tab, and dies
on server restart.

Three places pin 3.11 and have to stay in agreement: `[tool.ruff]
target-version`, `[tool.ty.environment] python-version`, and `langgraph.json`'s
`python_version`. So avoid 3.12+ syntax even though the local `.venv` may be
newer. The ty pin is explicit rather than inferred from `requires-python`,
which is deliberately looser (`<4.0`) for packaging.

## The invariant everything else serves

**The traveler is never shown synthetic inventory described as real.** Both
post-scaffold commits were data-honesty fixes, and each had to touch
`availability.py`, `duffel.py`, and the tests together — the property cannot be
maintained in one place. Five enforcement points must stay in agreement:

1. Every offer carries `synthetic: bool` alongside `source`.
2. Providers implement `synthetic_note(kind)`, consulted **even when a search
   returns nothing** — otherwise an empty result reads as "we checked real
   inventory and found none."
3. `_wrap()` in `availability.py` attaches the resulting `warning`.
4. The `@tool` docstrings and `prompts.py` tell the model to key off
   `warning`/`synthetic` — **never the provider name**.
5. The Streamlit UI reaches the traveler without the model in between, so it
   keys off the same two fields directly: a per-row provenance column, a
   standing notice on the Trip page, and a per-chart caption
   (`_provenance_note`) — a chart has no reachable tooltip on touch, so it
   labels itself too, and a fourth chart is a fourth call site. See the hazard
   under "Streamlit UI" — the naive way to fetch those fields returns nothing,
   silently.

Point 4 is the subtle one. `synthetic` is not a property of the provider:
Duffel in *test mode* returns fictional fares for flights and routes lodging to
`SampleProvider`, so both are synthetic under a provider named `duffel` — while
the *same* provider on a live token returns real inventory for both. Judging by
provider name gets this wrong in both directions, and the direction it gets
wrong changed when Duffel Stays landed. Key off `warning`/`synthetic`.

Lodging is the part that moved, and the reason is worth keeping: Duffel's Stays
test inventory sits at one coordinate pair, so a test-token search of a real
city returns an empty list rather than an error. Sample data is both the more
useful and the more honest answer there, which is why the test/live split lives
in `DuffelProvider.search_stays` rather than being pushed down into
`duffel_stays.py`.

When you change an offer's shape, a provider, or a warning, update the tool
docstring in the same edit — the docstring is the model's only interface to
that contract, and prose in `prompts.py` may restate it.

## Import-time graph construction

`langgraph.json` needs a module-level `graph`, so the last line of `agent.py`
runs `build_agent()` **as an import side effect** — constructing a model client
and the full tool stack. Two things exist solely because of this:

- `config.py` holds `DEFAULT_MODEL` / `DEFAULT_MAX_TOKENS` so callers wanting a
  constant don't build an agent. Put new shared constants here, not in `agent.py`.
- `main.py` defers importing `travel_agent.agent` until **after** `load_dotenv()`
  (see its inline comment), because graph construction reads `TAVILY_API_KEY` to
  decide whether search exists. Hoisting that import to module scope builds the
  agent against an empty environment.
- `ui.py` defers the same import into `get_agent()` for the same reason, so
  `streamlit_app.py` can import `ui` at module scope after `load_dotenv()`.
  Third site of one rule: anything reaching `travel_agent.agent` must do so
  after the environment is loaded.

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

The store itself still resolves from the execution context, but its
**namespace** does not: deepagents 0.7 made `StoreBackend(namespace=...)`
required and deleted the fallback that used to infer it. `_memories_namespace`
in `agent.py` reproduces that fallback exactly — `(assistant_id, "filesystem")`
under `langgraph dev` and Platform, `("filesystem",)` in-process. Don't
"simplify" it to the flat tuple: profiles already written under `langgraph dev`
live under the two-part key, a memory file that isn't there is skipped
silently, and the only symptom is an agent that has never met the traveler.

Both paths are string constants in `prompts.py` (`WORKSPACE`, `MEMORY_PATH`) and
are interpolated into prompt text — but two uncoupled copies have to move with
them, neither loudly. `MEMORY_PREFIX` in `agent.py` is the `CompositeBackend`
route key: if it stops being a prefix of `MEMORY_PATH`, the profile falls through
to `StateBackend` and traveler memory silently stops persisting. `ui.py`'s
`WORKSPACE_FILES` and `ITINERARY_PATH` hardcode `/trip/...`, and the Trip page
looks them up by exact path in `record.files`, so a renamed workspace renders an
empty page. Nothing tests either agreement.

## Provider seam

`AvailabilityProvider` is a `Protocol` in `availability.py`; `_PROVIDERS` maps
`TRAVEL_AGENT_PROVIDER` values to factories. To add one (Amadeus, Booking.com):
implement the protocol, register a factory, and set each offer's `source` to the
provider name.

`duffel` is one entry serving both kinds. Flights live in `duffel.py`, lodging
in `duffel_stays.py`, and `DuffelProvider` composes them — `duffel_stays.py`
imports `API_BASE`, `DUFFEL_SOURCE`, `DuffelError`, `parse_amount`,
`raise_for_status` and `seconds_until` from `duffel.py` at module scope, so
`duffel.py` imports `fetch_stays` **inside** `search_stays` or the two cycle.
Same lazy-import trick `_load_duffel` uses one level up, for the same reason.

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

**Duffel is read-only** — searches and reads, never `POST /air/orders` or
`POST /stays/bookings`, and never a quote. Adding booking means putting order
creation behind Deep Agents' `interrupt_on` human-approval gate; that is a
deliberate design decision, not a config change. `tests/test_no_booking.py`
enforces this by **path, not verb** — `/stays/search` is itself a POST, and is
the one call the lodging path is built to make.

It walks the AST of every file under `src/` and fails on any booking, quote,
payment or cancellation path appearing in a string literal that is not a
docstring. That shape matters. The hook this replaced matched the edit text, so
it only recognised a path sitting immediately after a `}` — `path =
"/air/orders"` followed by `f"{API_BASE}{path}"` walked straight through, in a
file that already hoists `API_BASE`. It also denied the accurate sentence this
document asks you to write into a tool docstring. Both go away here: a request
cannot reach an endpoint whose path is not a literal somewhere, comments never
enter the tree, and docstrings are exempt by construction. Covering a new
provider is a row in `BOOKING_PATHS`.

Duffel v2 Air response shape, easy to get wrong: `live_mode` is top level;
`duration` sits on each **slice** (covering layovers — summing segments
understates multi-stop trips); `cabin_class` is nested at
`slices[].segments[].passengers[].cabin_class`, not on the offer.

Cabin selection turns that nesting into three traps, because Duffel honours
`cabin_class` as a **preference, not a filter** — a search can succeed and
return something else:

- **Do not type the `@tool` parameter as the `Literal`.** `CabinClass` exists
  and is used on the provider signatures, but `search_flights` takes a plain
  `str`. LangChain builds a pydantic schema from the signature, so a `Literal`
  validates *before* the body runs and **raises** instead of returning
  `{"error": ...}` — the one thing every other bad input here is careful not to
  do. Normalising in the body is what keeps that contract, and it is also what
  makes "Premium Economy" work at all.
- **`_select_flights` has to know the requested cabin.** A lower cabin is
  always cheaper and the trim keeps the cheapest, so trimming by price alone
  answers a business search with twenty economy fares. Nothing errors; the
  table is simply the wrong question. The requested cabin is selected from
  first, and the rest tops up.
- **The mismatch rides `cabin_note`, never `warning`.** `warning` means one
  thing — not real inventory — and a *live* search never sets it, so reusing it
  would put the "illustrative sample prices" banner on a real bookable fare.
  `search_label` reads `requested_cabin` off the payload for the same family of
  reason: read off `offers[0]` and a downgraded business search labels itself
  economy, erasing the mismatch from the one control a human looks at.

`_cabin_class` reports `MIXED_CABIN` rather than the first cabin found, since a
round trip can come back business out and economy home — same reasoning as
preferring a slice's own duration over a segment sum.

**Both providers must emit the same offer keys**, and nothing about the code
makes that obvious — `map_offer` and `SampleProvider.search_flights` build
their dicts independently, hundreds of lines apart. A key one sets and the
other omits fails silently in one direction only: `.get` returns `None` and
every reader downstream treats `None` as "nothing to say", so the symptom is a
caption that quietly stops rendering on the provider returning *real*
inventory. `depart_date`/`return_date` drifted exactly this way and cost two
things at once — `search_label` dropped the dates from the search picker
(making two date-varied searches indistinguishable, the one job `_pick_search`
has) and the Trip page's outbound-leg caveat never fired on a live round trip.
Duffel echoes neither date, so `map_offer` derives both from the slices.
`test_both_providers_agree_on_the_shape_of_a_flight_offer` pins the agreement
and lists the five keys Duffel is allowed to add, so a sixth has to be a
decision rather than a drift.

Duffel v2 **Stays** is shaped differently enough that Air intuitions mislead:

- Results arrive at `data.results[]`, not `data.offers`.
- **There is no `live_mode` field anywhere on Stays.** The token is the only
  signal, which is why `map_stay_result` takes `synthetic` as a given rather
  than reading it. Do not go looking for the flag that works on Air.
- Search is by `geographic_coordinates` — there is **no city-name form**. Hence
  `CITY_COORDINATES`, and hence an unknown city being an error rather than a
  nearest match.
- `review_score` is 0–10 and `rating` is 1–5 stars. Only the first may feed
  `guest_rating`; the Trip page renders it on a hardcoded 0–10 progress bar and
  axis, so backfilling stars shows a four-star hotel as 4/10.
- An **empty** `cancellation_timeline` means non-refundable; an **absent** one
  means unknown. `free_cancellation` is omitted for the second, because
  `cancel_band` renders a missing key as "Not stated" and that is a different
  claim from "no".
- `total_amount` and `due_at_accommodation_amount` are **not summed**. Duffel's
  own docs disagree about whether the second sits inside the first, so both are
  passed through as they arrive. Adding them double-counts under one reading;
  subtracting understates under the other.
- No `supplier_timeout` — that knob is Air-only, so the Stays timeout is a flat
  constant and its error message must not point at `DUFFEL_SUPPLIER_TIMEOUT_MS`.

`max_nightly_rate` is a bare USD number and a live search is not always in USD.
The ceiling is applied only to offers quoted in USD; others pass through
unfiltered rather than being silently dropped, since an empty result set says
nothing about why it is empty.

## Streamlit UI

`streamlit_app.py` is the entry point, pages are `app_pages/`, and `ui.py` is
the seam that knows about the agent. Pages stay presentational; anything that
touches LangGraph belongs in `ui.py`. Appearance lives entirely in
`.streamlit/config.toml` — never CSS injected from Python, whose internal class
names are unstable across releases and fight the theme rather than extend it.

Charts split that rule along one line: **colour comes from the theme, geometry
does not.** A Vega spec omits `color.scale.range` so `theme="streamlit"` fills it
from `chartCategoricalColors`, and no hex is ever written in Python. Mark size,
opacity and legend padding are set in the spec; the chart height is a
`st.vega_lite_chart` kwarg and the card height a `_CARD_HEIGHT` constant — all in
Python, because `config.toml` has no vocabulary for them. Don't claim otherwise
in a comment — an earlier draft said a spec "keeps every appearance value in
config.toml" while
hardcoding six of them, which sends the next editor looking in the TOML for a dot
size that was never there.

**Subagent tool results never reach the parent's message history.** deepagents
folds a subagent's state back without its `messages` (`_EXCLUDED_STATE_KEYS` in
`middleware/subagents.py`), substituting a single `ToolMessage` that carries
only the subagent's closing prose. `search_flights` and `search_stays` are bound
**only** to `availability-scout`, so reading `get_state().values["messages"]`
after a run finds no search results at all — `warning` and `synthetic` become
unreachable and the honesty notice silently stops rendering. Nothing errors.

So `ui.py` captures payloads **during** the stream, and passes `subgraphs=True`
so nested messages are emitted in the first place — without it
`langgraph/pregel/_messages.py` drops every message whose checkpoint namespace
is nested. That same flag is why prose has to be filtered back out by
namespace: root-namespace chunks are the main agent's answer, anything deeper
is subagent chatter that belongs in the activity trail.

`tests/test_ui.py` pins this against a scripted fake stream rather than only
describing it: that `subgraphs=True` and both stream modes are requested at
all, that a namespaced `ToolMessage` is captured while namespaced prose *and*
subagent state emissions are not, and the zero-offer case. It also pins
`CAPTURED_TOOLS`, whose entries are string literals two packages from the tools
they name — `availability-scout`'s toolset is pinned whole, so a fourth search
has to decide about capture instead of silently bypassing the honesty notice.

Other things that bite here:

- `st.write_stream` cannot consume LangChain chunks. Its LangChain branch reads
  `chunk.content`, which langchain-anthropic makes a **list of blocks** — so it
  falls through to a raw `st.write` and, with thinking on by default, renders
  reasoning as JSON in the chat. Yield `str(message.text)`; `.text` keeps text
  blocks only and returns `""` for thinking.
- **The chunk accumulator resets on a message-id change** (`stream_turn`).
  `subgraphs=True` interleaves the main agent's and the scout's
  `AIMessageChunk`s in one stream, and langchain's `merge_lists` merges two
  tool-call chunks whenever the `index` matches and the ids are merely *not
  inconsistent* — a continuation chunk carries `id=None`, so the scout's index-0
  args concatenate onto the main agent's index-0 `task` args. Folding the branch
  back into a plain `accumulated + message` corrupts `subagent_type`, mislabels
  the activity trail, and raises nothing. Nothing in the suite covers it.
- `st.write_stream` returns a `str` only if **every** yielded item is one; one
  non-`str` yield silently makes it a list. Progress goes through the
  `on_activity` callback, never the yield channel.
- Don't render the reply after `st.write_stream` — it already replaced its own
  placeholder, so a second write duplicates the message.
- `@st.cache_resource` is process-wide: that is what makes `/memories/` outlive
  a reload, and it also means every browser session shares one store. Editing
  the body of `get_agent()` invalidates the cache, dropping traveler memory on
  that hot-reload.
- Repeat searches are kept separate on purpose. `availability-scout` is
  prompted to vary its query, so `record.payloads["search_flights"]` holds
  *different questions*, not retries. Merging them into one table would sort a
  cheaper flight on other dates above a dearer one on the requested dates —
  the same class of misrepresentation the data-honesty invariant exists to
  prevent. `_pick_search` names each query instead; don't "simplify" it into a
  concatenation.
- `default=` seeds a keyed widget's **first render only**. `_pick_search` wants
  a *moving* default — each new search — so it writes `st.session_state[key]`
  before the widget renders instead. Passing both logs "created with a default
  value but also had its value set via the Session State API". Nothing errors
  when this is wrong; the control just quietly stops tracking.
- Widget state outlives a trip. "Start a new trip" swaps the record and the
  thread id, but a widget using `persist_state="session"` keeps its value, so
  anything gating on trip identity must say so — `_pick_search` stamps
  `(thread_id, len(searches))`, because the count alone collides across trips.
  Scratch keys go in the one reserved `_search_stamps` dict: session state is a
  single flat namespace shared with widget keys.
- `st.segmented_control` defaults to `required=False`, so a single-select
  control can be cleared by clicking the selected option. Where there is no
  meaningful empty state — one of N searches is always on screen — pass
  `required=True`.
- Streamlit's stop button and a mid-stream page switch raise `StopException` /
  `RerunException`, which subclass **`BaseException`, not `Exception`**. So
  `except Exception` cannot see them (and must not catch them — that breaks the
  stop button); anything that has to survive an interrupted turn goes in a
  `finally`.
- The pages directory must be `app_pages/`. A directory named `pages/` triggers
  Streamlit's legacy auto-discovery alongside `st.navigation`.
- `st.set_page_config` is called once, first, in the entry point. A second call
  in a page silently overrides it.
- **`st.dataframe`'s `column_order` is a whitelist and hides what it does not
  name.** A key configured in `column_config` but missing from the order is
  simply absent from the table — which is how `due_at_accommodation` came to be
  dropped from a column headed "Total" that may exclude it. Assert on
  `element.proto.column_order`, never the call site: `column_config` alone
  passes either way. Any field the data-honesty invariant put there needs that
  assertion, because the failure renders perfectly and raises nothing.
- **Don't pass `icon=` to `st.expander`.** `st.status` *is* an expandable
  carrying an icon, so that is how AppTest tells them apart — `element_tree.py`
  routes any `expandable` with an icon to `Status` and everything else to
  `Expander`. An expander given `icon=` therefore vanishes from `at.expander`
  and shows up in `at.status` with a meaningless `state`, which `test_pages.py`
  compares element for element. The browser renders it correctly either way;
  only the page's testability breaks. Prefix the icon into the label instead.
- **One palette serves both modes by choice, not by constraint.** As of
  Streamlit 1.62 `chartCategoricalColors` *is* accepted in `[theme.light]` /
  `[theme.dark]` and their sidebar sections, inheriting from `[theme]` when
  unset; only seven keys are genuinely top-level-only (`base`, `baseFontSize`,
  `baseFontWeight`, `fontFaces`, `metricValueFontSize`, `metricValueFontWeight`,
  `showSidebarBorder`). Because a single top-level palette is what is
  configured, a usable hue has to sit in the *intersection* of the two OKLCH
  lightness bands (light 0.43–0.77, dark 0.48–0.67), i.e. inside the dark one.
  Splitting the palette per mode would lift that constraint — a real option, not
  an impossibility. Eyeballing this is how the first palette
  ended up with five of seven slots too light to hold against the dark surface.
  Validate with the dataviz skill's `validate_palette.js` before changing a
  value, and run `--pairs all` for the first three slots — a scatter can put any
  two marks side by side, so it needs every pair separated, not just neighbours.
- **The pairlist to validate follows the chart form, and a pinned domain changes
  it.** Pinning `color.scale.domain` is what keeps hues stable when a category or
  band is missing, but it also means the marks that end up *touching* are
  whichever values the data has — not the palette's adjacent pairs. A stacked
  spend bar was reverted over this: `{flights, lodging, food, other}` put slot 3
  against slot 7 at CVD ΔE 0.8, and a stacked bar has no gap, stroke or label
  left once colour fails, because the 2px surface gap the spec wants needs a
  surface colour a Vega spec cannot read. Magnitude comparisons stay a
  single-hue `st.bar_chart`, which has no adjacency to validate.
- **`axis.values` cannot extend a scale domain.** Vega's `validTicks` silently
  drops any tick that falls outside the range, so a tick pinned to the budget
  vanished whenever the estimate came in under it — taking every other value
  label with it and leaving a caption pointing at a line that was not drawn. A
  reference line placed past the data needs the scale widened too, or it is not a
  reference line.

An assistant bubble is rebuilt from `st.session_state.messages` on every rerun,
so whatever the live run rendered has to be reconstructable from the stored
entry — including the status element's label and state. Deriving them from
`len(activity)` replays a failed turn as a green "N steps", and the message
visibly rearranges itself on the next interaction.

## Tests

No network, no model calls, no token required. `httpx.MockTransport` verifies
request construction; `map_offer` is pure and tested against a payload shaped to
Duffel's documented schema. `test_ui.py` runs no Streamlit server — it drives
the plain data structures the stream routes into, which is where the UI's
decisions actually live.

`test_pages.py` runs the page scripts themselves through
`streamlit.testing.v1.AppTest`, a headless script runner — still no server, no
browser, and `stream_turn` is patched out so no agent is built. It exists for
the hazards listed under "Streamlit UI" above, which live in Streamlit's own
widget and control-flow semantics rather than in this project's data, and which
all fail silently: the page renders, nothing raises, and the traveler is shown
the wrong search or loses a turn from the transcript. Reach for `AppTest` when
a bug is only observable across two reruns.

A page script cannot be imported, so nothing defined inside `app_pages/` is
unit-testable — `AppTest` can only see what the page renders. Pure data mapping
therefore lives in `ui.py` even when it is purely presentational (`search_label`,
`stop_band`, `cancel_band`) and is tested directly in `test_ui.py`. Keep
`AppTest` for what only a rendered page can show: which elements appear, in what
order, and what a second rerun does.

`tests/conftest.py` has an **autouse** fixture that unsets provider env vars and
clears the `_build_provider` cache around every test. Without it the suite picks
up whatever a developer exported — and the combination the README tells you to
set sends real requests to `api.duffel.com`. **Add any new provider env var to
`_PROVIDER_ENV`.**

`test_env_isolation.py` enforces that mechanically, by AST rather than by name:
it treats any function that reads the environment through one of its own
parameters as a wrapper, so `duffel.py` reaching `os.getenv` via `_env_int` is
seen, and a future `_env_str` is covered the day it is written. It imports
`_PROVIDER_ENV` directly, so deleting the tuple is an `ImportError` rather than
a check that quietly passes. Two cases guard the guard —
`test_the_scan_still_sees_the_reads_that_exist_today` and
`test_a_new_wrapper_shape_is_detected_without_naming_it` — because a static scan
that stops matching reads green, which is the same failure class as the drift it
looks for.

Note the direction that is actually dangerous. Adding a var and forgetting the
tuple is loud on the machine where it matters: `test_tools.py` sets
`TRAVEL_AGENT_PROVIDER` itself and would go red. **Emptying `_PROVIDER_ENV` is
the silent one** — a CI runner has nothing exported, so it stays green forever.
That is the direction only this test covers, and the reason it is a test rather
than the hook it replaced, which never ran in CI at all.

## Prompts

`prompts.py` is tuned for Claude Opus 5, which by default writes long responses,
self-verifies unasked, and expands scope. The prompts counter those tendencies —
no "double-check your work" scaffolding, explicit conciseness and scope
discipline. Don't add verification scaffolding back in.

From deepagents 0.7 the library contributes **no** base prompt: it passes `""`,
and no `HarnessProfile` matches `anthropic:claude-opus-5`. `MAIN_AGENT_PROMPT`
is therefore the entire system prompt, and `TASK_SYSTEM_PROMPT` is gone too, so
`subagents.py`'s `description` fields are the only surviving statement of the
delegation contract. Write prompts as the whole thing, not as a complement to
something upstream. The no-preamble and parallel-tool-call rules under "Working
style" were part of that deleted base and are restated here on purpose — but
don't restore the rest of it wholesale: it contained "your first attempt is
rarely correct — iterate", which is exactly what this file exists to counter.

Subagent `description` fields in `subagents.py` are what the main agent reads
when deciding to delegate, so they carry the "brief me completely in one call"
contract — subagents are stateless and cannot ask follow-up questions.

`DEFAULT_MAX_TOKENS` is generous (16k) because thinking is on by default on Opus 5
and counts against the ceiling.

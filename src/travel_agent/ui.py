"""Runtime seam between Streamlit and the travel agent.

The page scripts under `app_pages/` stay thin — layout and widgets only.
Everything that knows how the agent is built, how a thread is addressed, and
how to get data back out of LangGraph lives here.

Three facts about the agent shape this module. Each was checked against the
installed packages rather than assumed, because each is counter-intuitive:

- `get_state(config).values` carries exactly `files` and `messages`. `files`
  is a `dict[str, FileData]`; the `DeltaChannel` behind it materialises to a
  plain dict on read and successive writes merge, so the workspace
  accumulates across turns.

- **Tool results from subagents never reach the parent's message history.**
  `deepagents.middleware.subagents` excludes `messages` when folding a
  subagent's state back (`_EXCLUDED_STATE_KEYS`) and substitutes a single
  `ToolMessage` holding only the subagent's closing prose. `search_flights`
  and `search_stays` are bound *only* to `availability-scout`, so reading
  `values["messages"]` after a run finds no search results at all — the
  `warning` and `synthetic` fields would be permanently unreachable. They are
  therefore captured **during** the stream instead, into `TripRecord`.

- Reaching those subagent messages at all requires `subgraphs=True`.
  `langgraph.pregel._messages` drops any message whose checkpoint namespace is
  nested unless that flag is set. The same flag is why subagent *prose* has to
  be filtered back out by namespace: only the root agent's tokens belong in
  the chat bubble.

A `@tool` returning a dict arrives with its payload JSON-encoded on
`ToolMessage.content`, which is what lets the trip view show real figures
rather than re-parsing the markdown the agent wrote.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

import streamlit as st
from langchain_core.messages import AIMessageChunk, ToolMessage

from travel_agent.config import DEFAULT_MODEL

# Safe at module scope, unlike `travel_agent.agent`: this module only defines
# tools, reading no environment and building no model client on import.
from travel_agent.tools.availability import DEFAULT_CABIN

# The deliverable files the agent is prompted to write, in the order a reader
# wants them: what was asked, what was found, what it costs, what to do.
WORKSPACE_FILES: tuple[tuple[str, str, str], ...] = (
    ("/trip/brief.md", "Brief", ":material/assignment:"),
    ("/trip/research.md", "Research", ":material/travel_explore:"),
    ("/trip/options.md", "Options", ":material/flight:"),
    ("/trip/budget.md", "Budget", ":material/payments:"),
    ("/trip/itinerary.md", "Itinerary", ":material/map:"),
)

ITINERARY_PATH = "/trip/itinerary.md"

# Tools whose structured output the trip view renders directly.
CAPTURED_TOOLS = ("search_flights", "search_stays", "summarize_budget")

# Friendlier names for the activity panel than the raw tool names.
_TOOL_LABELS = {
    "search_flights": "Searching flights",
    "search_stays": "Searching places to stay",
    "summarize_budget": "Costing the trip",
    "write_file": "Writing to the workspace",
    "edit_file": "Revising the workspace",
    "read_file": "Reading the workspace",
    "ls": "Listing the workspace",
    "write_todos": "Planning",
    "task": "Delegating to a subagent",
}


@st.cache_resource(show_spinner="Starting the travel agent…")
def get_agent(model: str = DEFAULT_MODEL) -> Any:
    """Build the agent once per server process.

    `cache_resource` rather than `cache_data` because the agent is a live
    object graph — an Anthropic client, an HTTP pool, a checkpointer — not
    something to serialise. The default global scope is deliberate: it is what
    lets the in-memory store behind `/memories/` outlive a page reload. It
    also means every browser session shares that store, which is right for
    local single-traveler use and would need namespacing before deploying.

    `travel_agent.agent` is imported *inside* this function on purpose. That
    module builds its graph as an import side effect and reads
    `TAVILY_API_KEY` while doing so, so importing it at module scope would
    construct the agent before `load_dotenv()` ran and silently drop web
    search. `main.py` defers the same import for the same reason.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    from travel_agent.agent import build_agent

    return build_agent(model, checkpointer=MemorySaver(), store=InMemoryStore())


def thread_config(thread_id: str) -> dict[str, Any]:
    """Address one conversation.

    The thread id scopes the checkpointer, and with it `/trip/*` — a new
    thread is a new trip. `/memories/` is keyed by the store instead, so the
    traveler profile deliberately crosses threads.
    """
    return {"configurable": {"thread_id": thread_id}}


def _decode(message: ToolMessage) -> dict[str, Any] | None:
    """Decode a tool result, or None when it is not a JSON object."""
    content = message.content
    if not isinstance(content, str):
        return None
    try:
        decoded = json.loads(content)
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


@dataclass
class TripRecord:
    """Structured results gathered while the agent works.

    Lives in session state and accumulates across turns, because a later
    question ("what about flying Tuesday instead?") should not blank the trip
    view until its search comes back.
    """

    payloads: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    def record_tool(self, message: ToolMessage) -> None:
        """Keep a tool result if it is one the trip view can render.

        Failed calls are dropped rather than stored. These tools return
        `{"error": ...}` dicts instead of raising, and `ToolMessage.status`
        stays `"success"` for them, so the payload is the only signal.
        """
        if message.name not in CAPTURED_TOOLS:
            return
        payload = _decode(message)
        if payload is None or "error" in payload:
            return
        self.payloads.setdefault(message.name, []).append(payload)

    def absorb_files(self, values: dict[str, Any]) -> None:
        """Refresh the workspace snapshot from a full state emission."""
        from deepagents.backends.protocol import FileData
        from deepagents.backends.utils import file_data_to_string

        files = values.get("files")
        if not isinstance(files, dict):
            return
        entries = cast("dict[str, FileData]", files)
        self.files = {path: file_data_to_string(data) for path, data in sorted(entries.items())}

    def latest(self, tool_name: str) -> dict[str, Any] | None:
        """Most recent result from one tool, or None if it never ran."""
        found = self.payloads.get(tool_name)
        return found[-1] if found else None

    def caveats(self) -> list[str]:
        """Distinct synthetic-data warnings raised by any search so far.

        Keyed off the `warning` field, never the provider name. A provider
        called `duffel` returns fictional fares and sample lodging under a test
        token and real inventory of both under a live one, so the name says
        nothing about whether any particular result is real.

        Order is preserved rather than sorted: flights and lodging can carry
        different caveats in one plan, and a warning survives even when the
        search that raised it returned nothing — an unannotated empty result
        would read as "we checked real inventory and found none".
        """
        seen: list[str] = []
        for tool_name in ("search_flights", "search_stays"):
            for payload in self.payloads.get(tool_name, []):
                warning = payload.get("warning")
                if isinstance(warning, str) and warning not in seen:
                    seen.append(warning)
        return seen


def offers(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Offer list from a search payload, tolerating a missing one."""
    if not payload:
        return []
    found = payload.get("offers")
    return [offer for offer in found if isinstance(offer, dict)] if isinstance(found, list) else []


def _short_date(value: Any) -> str:
    """`2026-09-10` becomes `Sep 10`, or `""` if it is not an ISO date."""
    if not isinstance(value, str):
        return ""
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return ""
    return f"{parsed:%b} {parsed.day}"


def _date_span(start: Any, end: Any) -> str:
    first, last = _short_date(start), _short_date(end)
    if first and last:
        return f"{first}–{last}"
    return first or last


def _count(value: Any, noun: str) -> str:
    if not isinstance(value, int) or value < 1:
        return ""
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


def _cabin_suffix(requested: Any) -> str:
    """Name a non-default cabin, or `""` when there is nothing worth adding.

    Economy is suppressed because it is the default: labelling every ordinary
    search "Economy" would add a word to each one to distinguish none of them.
    """
    if not isinstance(requested, str) or requested == DEFAULT_CABIN:
        return ""
    return requested.replace("_", " ").capitalize()


def search_label(payload: dict[str, Any] | None) -> str:
    """Describe the query a search answered, e.g. `SFO→NRT · Sep 10–Sep 15 · 2 travelers`.

    `availability-scout` is prompted to vary its searches — nearby airports,
    dates shifted by a day, a different neighbourhood — so two payloads are
    usually *different questions* rather than retries. Showing them merged
    would rank a cheaper flight on other dates above a dearer one on the
    requested dates, so the UI keeps them separate and names each query.

    Most of the label is read off the offers, because every offer records the
    query it matched. Cabin is the exception, and has to be: it comes from the
    payload's `requested_cabin`, never from an offer's own `cabin`. Cabin is a
    preference rather than a filter, so a business search whose cheapest result
    was downgraded to economy would — read off `offers[0]` — label itself
    "economy" and silently erase the very mismatch worth seeing.
    """
    items = offers(payload)
    if not items:
        return "No results"
    first = items[0]
    if first.get("origin"):
        parts = [
            f"{first.get('origin')}→{first.get('destination')}",
            _date_span(first.get("depart_date"), first.get("return_date")),
            _count(first.get("travelers"), "traveler"),
            _cabin_suffix((payload or {}).get("requested_cabin")),
        ]
    else:
        parts = [
            str(first.get("location") or "").title(),
            _date_span(first.get("check_in"), first.get("check_out")),
            _count(first.get("guests"), "guest"),
        ]
    return " · ".join(part for part in parts if part) or "Search"


# Colour bands for the Trip page's two tradeoff scatters. They live here rather
# than in the page because they are pure data mapping — the same reason
# `search_label` does — and because a page script cannot be imported, so nothing
# in `app_pages/` is unit-testable.
#
# Three bands each, and that is a ceiling rather than a tidy number. A scatter is
# an "all-pairs" form: any two dots can end up side by side, so *every* pair of
# hues has to clear colour-blind separation, not just the pairs a legend places
# next to each other. That gate is strictly harder than adjacent-only, and three
# slots of the theme palette are what clear it. A fourth band is not a fourth
# colour away — no ordering of the palette separates four.
STOP_BANDS = ("Nonstop", "One stop", "Two or more stops")
CANCEL_BANDS = ("Free cancellation", "No free cancellation", "Not stated")


def stop_band(stops: Any) -> str | None:
    """Bucket a flight offer's stop count into one of `STOP_BANDS`.

    Three or more stops fold into the last band rather than earning a colour of
    their own. **None when the count is unreadable**, and callers drop those
    offers instead of plotting them — the same treatment `total_fare` gets when
    a price will not parse.

    That is reachable rather than defensive: `map_offer` reads `stops` off the
    first slice, and an offer carrying no slices at all leaves it `None`. All
    three bands here make a positive claim about a real count, so there is no
    slot left to say "we could not tell" — and "Two or more stops" is a claim,
    not a safe default. An offer whose stops nobody can read is better left off
    the chart than drawn in the least attractive band it might not belong to.
    """
    if not isinstance(stops, (int, float)):
        return None
    if stops <= 0:
        return STOP_BANDS[0]
    if stops == 1:
        return STOP_BANDS[1]
    return STOP_BANDS[2]


def cancel_band(free: Any) -> str:
    """Bucket a stay's cancellation policy into one of `CANCEL_BANDS`.

    Anything that is not an explicit bool lands in "Not stated" rather than
    defaulting to "No free cancellation" — the same reasoning as warning about
    synthetic data on an *empty* result set. Absent information is not a
    negative finding, and a traveler choosing on flexibility must not be told a
    policy was checked when it was not.

    Unlike `stop_band` this never returns None, and the difference is not an
    inconsistency: only two of these three bands carry a real value, so the
    third slot is free to mean "unknown" honestly. All three stop bands are
    spoken for.
    """
    if not isinstance(free, bool):
        return CANCEL_BANDS[2]
    return CANCEL_BANDS[0] if free else CANCEL_BANDS[1]


# Only these three have a Streamlit currency preset; anything else needs a
# printf format built from the code itself.
CURRENCY_PRESETS = {"USD": "dollar", "EUR": "euro", "JPY": "yen"}


def currency_code(items: list[dict[str, Any]], fallback: str = "USD") -> str:
    """Currency these offers are quoted in, for axis titles and number formats."""
    for item in items:
        code = item.get("currency")
        if isinstance(code, str) and code:
            return code
    return fallback


def money_format(items: list[dict[str, Any]], fallback: str = "USD") -> str:
    """Number format for whatever currency these offers are quoted in."""
    code = currency_code(items, fallback)
    return CURRENCY_PRESETS.get(code, f"%.2f {code}")


def source_column(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add a visible provenance field to each offer.

    Derived from each offer's own `synthetic` flag, never from the provider
    name: the same provider called `duffel` returns fictional fares and sample
    lodging on a test token, and real inventory of both on a live one.

    This is the single derivation of provenance for the whole page — the table
    column, the chart tooltips and the chart's own provenance caption all read
    the field it writes, rather than each re-reading `synthetic` on its own.
    """
    return [
        {**item, "provenance": "Sample data" if item.get("synthetic") else "Live"} for item in items
    ]


def costing_series(history: list[dict[str, Any]], key: str) -> list[float] | None:
    """One figure's run across successive costings, for a metric sparkline.

    None below two points, because a one-point trend is a dot.

    Named "costings" rather than "history" on purpose. `summarize_budget` is
    re-run whenever the plan changes, but nothing guarantees consecutive
    payloads are consecutive versions of *one* plan: `BUDGET_PROMPT` asks the
    analyst to propose cuts, and the main agent can cost several variants
    against a stateless subagent inside a single turn. So the values are the
    costings in the order they were run and no more than that — which is why the
    card renders them as bars rather than a line.
    """
    values = [
        float(payload[key]) for payload in history if isinstance(payload.get(key), (int, float))
    ]
    return values if len(values) >= 2 else None


def _unpack(item: Any) -> tuple[tuple[str, ...], str, Any]:
    """Normalise a stream item to `(namespace, mode, data)`.

    With both `subgraphs=True` and a list `stream_mode`, LangGraph yields
    3-tuples. The 2-tuple branch keeps this honest if either is ever dropped.
    """
    if isinstance(item, tuple) and len(item) == 3:
        namespace, mode, data = item
        return tuple(namespace), mode, data
    mode, data = item
    return (), mode, data


def _tool_label(name: str | None, args: dict[str, Any] | None) -> str:
    """Human-readable description of a tool call in flight."""
    label = _TOOL_LABELS.get(name or "", f"Running {name}" if name else "Working")
    if name == "task" and args:
        subagent = args.get("subagent_type")
        if isinstance(subagent, str) and subagent:
            return f"Delegating to {subagent}"
    return label


def stream_turn(
    thread_id: str,
    prompt: str,
    record: TripRecord,
    on_activity: Callable[[str, str], None] | None = None,
) -> Iterator[str]:
    """Run one turn, yielding the main agent's prose token by token.

    Only `str` is ever yielded, which matters more than it looks:
    `st.write_stream` returns a plain string only when every yielded item is
    one, and it cannot render a LangChain chunk itself — with thinking enabled
    the chunk's `content` is a list of blocks, which would land in the chat as
    raw JSON. `message.text` is the accessor that keeps text blocks only, so
    reasoning never reaches the traveler.

    Everything that is not main-agent prose is routed sideways: tool results
    into `record`, progress into `on_activity(call_id, label)`. Nothing else
    is yielded, so the caller can hand this straight to `st.write_stream`.
    """
    agent = get_agent()
    accumulated: AIMessageChunk | None = None
    announced: dict[str, str] = {}

    for item in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=thread_config(thread_id),
        stream_mode=["messages", "values"],
        subgraphs=True,
    ):
        namespace, mode, data = _unpack(item)

        if mode == "values":
            # Subagents emit their own state too; only the root's is complete.
            if not namespace and isinstance(data, dict):
                record.absorb_files(data)
            continue

        message = data[0] if isinstance(data, tuple) else data

        if isinstance(message, ToolMessage):
            record.record_tool(message)
            continue

        if not isinstance(message, AIMessageChunk):
            continue

        # Tool calls are announced from whichever tier issues them, so the
        # traveler sees the availability-scout's searches, not just `task`.
        if message.id and accumulated is not None and accumulated.id != message.id:
            accumulated = message
        else:
            accumulated = message if accumulated is None else accumulated + message
        if on_activity is not None:
            for call in accumulated.tool_calls:
                call_id = call.get("id") or ""
                label = _tool_label(call.get("name"), call.get("args"))
                if call_id and announced.get(call_id) != label:
                    announced[call_id] = label
                    on_activity(call_id, label)

        # Subagent prose belongs in the activity panel's summary, not the
        # chat bubble — the main agent restates what matters.
        if namespace:
            continue
        text = message.text
        if text:
            yield str(text)

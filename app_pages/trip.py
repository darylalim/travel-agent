"""Trip page: the plan as figures rather than conversation.

Reads the structured payloads captured while the agent worked, not the
markdown it wrote — so fares, durations and the `synthetic` flag arrive as
data instead of prose. Offers are passed to `st.dataframe` as plain
`list[dict]`; there is no pandas import here, so pandas stays a transitive
dependency of Streamlit rather than an undeclared direct one.

The title lives in `streamlit_app.py`; pages do not set their own.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from travel_agent.ui import ITINERARY_PATH, WORKSPACE_FILES, offers, search_label

# Only these three have currency presets; anything else needs a printf format.
_CURRENCY_PRESETS = {"USD": "dollar", "EUR": "euro", "JPY": "yen"}

record = st.session_state.record
budget = record.latest("summarize_budget")
flight_searches = record.payloads.get("search_flights", [])
stay_searches = record.payloads.get("search_stays", [])

if not any((flight_searches, stay_searches, budget, record.files)):
    st.info(
        "Nothing planned yet. Describe a trip on the Plan page and the "
        "figures will appear here as the agent gathers them.",
        icon=":material/tips_and_updates:",
    )
    st.stop()


def _money(items: list[dict[str, Any]], fallback: str = "USD") -> str:
    """Number format for whatever currency these offers are quoted in."""
    for item in items:
        code = item.get("currency")
        if isinstance(code, str) and code:
            return _CURRENCY_PRESETS.get(code, f"%.2f {code}")
    return _CURRENCY_PRESETS.get(fallback, "%.2f")


def _source_column(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add a visible provenance column.

    Derived from each offer's own `synthetic` flag, never from the provider
    name: Duffel test mode returns fictional fares, and Duffel lodging falls
    through to sample data, so the provider is wrong in both directions.
    """
    return [
        {**item, "provenance": "Sample data" if item.get("synthetic") else "Live"} for item in items
    ]


def _truncation_note(payload: dict[str, Any] | None) -> None:
    note = (payload or {}).get("truncated")
    if isinstance(note, str):
        st.caption(f":material/filter_list: {note}")


def _pick_search(searches: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Choose which of several searches to show, newest by default.

    Deliberately not merged. `availability-scout` varies its searches, so the
    union of two result sets is not one comparable list — sorting it by price
    would rank a cheaper flight on other dates above a dearer one on the dates
    actually asked for. Naming each query keeps that distinction visible.

    Options are indices rather than labels so two identical-looking queries
    stay distinguishable.
    """
    if not searches:
        return None
    if len(searches) == 1:
        st.caption(f":material/search: {search_label(searches[0])}")
        return searches[0]

    newest = len(searches) - 1
    chosen = st.segmented_control(
        "Which search",
        options=list(range(len(searches))),
        format_func=lambda index: search_label(searches[index]),
        default=newest,
        key=key,
        label_visibility="collapsed",
        # Survives a trip to the Plan page and back; widget values otherwise
        # reset on every page switch.
        persist_state="session",
    )
    index = chosen if isinstance(chosen, int) else newest
    st.caption(f":material/search: {len(searches)} searches run — showing one.")
    return searches[index]


# Standing caveats: repeated here rather than left in chat scrollback, so the
# figures below are never read without them.
for caveat in record.caveats():
    st.warning(caveat, icon=":material/warning:")

if budget:
    total = budget.get("total_estimated")
    ceiling = budget.get("budget_total")
    remaining = budget.get("remaining")
    used = budget.get("percent_of_budget_used")
    money = _CURRENCY_PRESETS.get(str(budget.get("currency", "USD")), "%.2f")

    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Estimated total",
            total,
            delta=(total - ceiling) if isinstance(total, (int, float)) and ceiling else None,
            delta_color="inverse",
            delta_description="vs. budget" if ceiling else None,
            format=money,
            border=True,
            height="stretch",
            help="Every priced line the budget analyst was given.",
        )
        st.metric(
            "Remaining",
            remaining,
            format=money,
            border=True,
            height="stretch",
        )
        with st.container(border=True, height="stretch"):
            st.metric("Budget used", used, format="%.0f%%")
            if isinstance(used, (int, float)):
                # st.progress rejects floats outside 0-1, and overspending
                # legitimately exceeds 100%.
                st.progress(min(max(used / 100, 0.0), 1.0))

    if budget.get("over_budget"):
        st.error(
            f"Over budget by {budget.get('overage')} {budget.get('currency', '')}.".strip(),
            icon=":material/trending_up:",
        )

    by_category = budget.get("by_category")
    if isinstance(by_category, dict) and by_category:
        st.subheader("Where the money goes", anchor=False)
        st.bar_chart(
            [{"category": name, "amount": value} for name, value in by_category.items()],
            x="category",
            y="amount",
            horizontal=True,
            height=260,
        )

flights = None
if flight_searches:
    st.subheader("Flights", anchor=False)
    flights = _pick_search(flight_searches, "flights_search")

flight_offers = _source_column(offers(flights))
if flight_offers:
    fare_format = _money(flight_offers)
    st.dataframe(
        flight_offers,
        hide_index=True,
        row_height=40,
        placeholder="—",
        column_order=(
            "carrier",
            "total_fare",
            "fare_per_traveler",
            "duration_minutes",
            "stops",
            "depart_time_local",
            "cabin",
            "provenance",
        ),
        column_config={
            "carrier": st.column_config.TextColumn("Carrier", width="medium", pinned=True),
            "total_fare": st.column_config.NumberColumn(
                "Total",
                format=fare_format,
                help="All travelers. A dash means no usable price was returned.",
            ),
            "fare_per_traveler": st.column_config.NumberColumn("Per traveler", format=fare_format),
            "duration_minutes": st.column_config.NumberColumn(
                "Duration",
                format="%d min",
                help="Whole journey, layovers included.",
            ),
            "stops": st.column_config.NumberColumn(
                "Stops", format="%d", width="small", alignment="center"
            ),
            "depart_time_local": st.column_config.TextColumn("Departs", width="small"),
            "cabin": st.column_config.TextColumn("Cabin", width="small"),
            "provenance": st.column_config.TextColumn(
                "Data",
                width="small",
                help="Sample data is illustrative and cannot be booked.",
            ),
        },
    )
    _truncation_note(flights)
elif flight_searches:
    # A search that legitimately found nothing. The caveat above still stands:
    # an unannotated empty result would read as "we checked real inventory".
    st.info("That search returned no flights.", icon=":material/search_off:")

stays = None
if stay_searches:
    st.subheader("Places to stay", anchor=False)
    stays = _pick_search(stay_searches, "stays_search")

stay_offers = _source_column(offers(stays))
if stay_offers:
    rate_format = _money(stay_offers)
    st.dataframe(
        stay_offers,
        hide_index=True,
        row_height=40,
        placeholder="—",
        column_order=(
            "name",
            "kind",
            "nightly_rate",
            "nights",
            "total_cost",
            "guest_rating",
            "free_cancellation",
            "provenance",
        ),
        column_config={
            "name": st.column_config.TextColumn("Property", width="large", pinned=True),
            "kind": st.column_config.TextColumn("Type", width="small"),
            "nightly_rate": st.column_config.NumberColumn("Per night", format=rate_format),
            "nights": st.column_config.NumberColumn("Nights", format="%d", width="small"),
            "total_cost": st.column_config.NumberColumn("Total", format=rate_format),
            # max_value is required: a float column defaults to a 0-1 scale,
            # which would render every 0-10 rating as a full bar.
            "guest_rating": st.column_config.ProgressColumn(
                "Rating", format="%.1f", min_value=0, max_value=10
            ),
            "free_cancellation": st.column_config.CheckboxColumn("Free cancel", width="small"),
            "provenance": st.column_config.TextColumn("Data", width="small"),
        },
    )
    _truncation_note(stays)
elif stay_searches:
    st.info("That search returned no places to stay.", icon=":material/search_off:")

itinerary = record.files.get(ITINERARY_PATH)
if itinerary:
    st.subheader("Itinerary", anchor=False)
    with st.container(border=True):
        st.markdown(itinerary)

written = [entry for entry in WORKSPACE_FILES if record.files.get(entry[0])]
if written:
    st.subheader("Workspace", anchor=False)
    st.caption("What the agent wrote while planning. Cleared when you start a new trip.")
    for path, label, icon in written:
        if path == ITINERARY_PATH:
            continue
        with st.expander(f"{icon} {label}"):
            st.markdown(record.files[path])

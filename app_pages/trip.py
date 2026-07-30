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

from travel_agent.ui import (
    CANCEL_BANDS,
    ITINERARY_PATH,
    STOP_BANDS,
    WORKSPACE_FILES,
    cancel_band,
    offers,
    search_label,
    stop_band,
)

# Only these three have currency presets; anything else needs a printf format.
_CURRENCY_PRESETS = {"USD": "dollar", "EUR": "euro", "JPY": "yen"}

# One reserved slot for `_pick_search`'s bookkeeping. Session state is a single
# flat namespace shared with widget keys, so per-selector scratch keys derived
# from the widget's own name (`flights_search_count`) sit one plausible widget
# away from a silent collision.
_SEARCH_STAMPS = "_search_stamps"

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


def _currency_code(items: list[dict[str, Any]], fallback: str = "USD") -> str:
    """Currency these offers are quoted in, for axis titles and number formats."""
    for item in items:
        code = item.get("currency")
        if isinstance(code, str) and code:
            return code
    return fallback


def _money(items: list[dict[str, Any]], fallback: str = "USD") -> str:
    """Number format for whatever currency these offers are quoted in."""
    code = _currency_code(items, fallback)
    return _CURRENCY_PRESETS.get(code, f"%.2f {code}")


# Mirrors the `Category` literal in `tools/budget.py`. Pinned in full, and never
# narrowed to the categories actually present: dropping the absent ones would
# shift every later category up a slot, so a trip with no `transport` line would
# paint `fees` in `transport`'s colour.
_CATEGORY_ORDER = ("flights", "lodging", "food", "activities", "transport", "fees", "other")

# Shared by every chart on this page, so the three legends read as one system.
# The padding is not cosmetic: Vega's default leaves 11-14px between horizontal
# entries, and an 8px swatch fills nearly all of it, so a long label runs into
# the swatch of the entry after it ("No free cancellation● Not stated"). There is
# room to spare at this width — the default is simply tight.
_LEGEND = {"orient": "bottom", "title": None, "columnPadding": 18, "labelOffset": 6}


def _mixed_source_note(plotted: list[dict[str, Any]]) -> None:
    """Name a blend of real and synthetic offers inside one chart.

    The standing notice at the top of the page covers a wholly synthetic set,
    and the table labels every row. A *mixed* set is the gap between them: the
    chart plots real and invented figures as identical marks, so the blend has
    to be said out loud.
    """
    synthetic = sum(1 for offer in plotted if offer.get("synthetic"))
    if 0 < synthetic < len(plotted):
        st.caption(
            f":material/warning: Mixed sources — {synthetic} of {len(plotted)} plotted "
            f"offers {'is' if synthetic == 1 else 'are'} sample data. "
            "The table below labels each row."
        )


def _tradeoff_scatter(
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    x_title: str,
    y_field: str,
    y_title: str,
    bands: tuple[str, ...],
    tooltip: list[dict[str, Any]],
) -> None:
    """Draw one "cheap versus good" scatter, shared by flights and stays.

    The scout is asked for "the best on price, the best on convenience, and the
    tradeoff between them". In a table that comparison means reading two numeric
    columns against each other row by row; as position it is one glance, and the
    offers nobody should pick — worse on both counts than something else on the
    list — fall in one corner.

    A hand-written Vega-Lite spec rather than `st.scatter_chart`, for two things
    the sugar cannot express:

    - **`zero: False`.** `st.scatter_chart` emits a bare `"scale": {}`, so Vega
      applies its default and includes zero on both axes. Fares spanning
      $1,275-$1,990 then occupy the top third of the plot with two thirds empty,
      and ratings clustered in 7.8-9.1 collapse onto one edge. Bars must start at
      zero because length carries the value; dots do not, so dropping it is
      honest here and it is the difference between a readable chart and a stripe.
    - **A pinned colour domain.** The sugar derives the colour domain from
      whatever is in *this* search. A flight search with no one-stop offers would
      hand that slot's hue to the next band along, so a traveler flipping between
      searches with the selector above would watch "the teal ones" change
      meaning — the recolour-on-filter anti-pattern. Passing hex values to
      `st.scatter_chart` pins the colours but sets `legend: null`, trading the
      mislabel for no key at all.

    `range` is deliberately absent: `theme="streamlit"` (the default) supplies it
    from `chartCategoricalColors`, so appearance stays in `.streamlit/config.toml`
    and no hex is duplicated into Python. Those first three slots are validated
    for all-pairs CVD separation in both modes — see the note in that file.
    """
    st.vega_lite_chart(
        rows,
        {
            # ~10px across, satisfying the >=8px marker floor. Semi-opaque
            # instead of the 2px surface ring the spec asks for on overlapping
            # dots: that ring has to be painted in the surface colour, which
            # differs between light and dark, and a Vega spec cannot read the
            # active theme — hardcoding either would break one mode and pull
            # appearance out of config.toml.
            "mark": {"type": "circle", "size": 90, "opacity": 0.85},
            "encoding": {
                "x": {
                    "field": x_field,
                    "type": "quantitative",
                    "title": x_title,
                    "scale": {"zero": False, "nice": True, "padding": 14},
                    "axis": {"grid": False},
                },
                "y": {
                    "field": y_field,
                    "type": "quantitative",
                    "title": y_title,
                    "scale": {"zero": False, "nice": True, "padding": 14},
                    "axis": {"grid": True, "format": ",.0f"},
                },
                "color": {
                    "field": "band",
                    "type": "nominal",
                    "scale": {"domain": list(bands)},
                    "legend": dict(_LEGEND),
                },
                "tooltip": tooltip,
            },
        },
        height=300,
    )


def _flight_tradeoff_chart(items: list[dict[str, Any]]) -> None:
    """Fare against journey time, coloured by how many stops it costs.

    An offer needs all three figures to be drawable, `stops` included: every
    stop band makes a positive claim, so `stop_band` returns None rather than
    guessing, and a dot with no band would fall outside the pinned colour
    domain — drawn, but keyed to nothing in the legend.
    """
    plottable = [
        offer
        for offer in items
        if isinstance(offer.get("duration_minutes"), (int, float))
        and isinstance(offer.get("total_fare"), (int, float))
        and stop_band(offer.get("stops")) is not None
    ]
    # Two points describe a line, not a tradeoff; the table says it better.
    if len(plottable) < 3:
        return

    axis_money = f"Total fare ({_currency_code(plottable)})"
    _tradeoff_scatter(
        [
            {
                "duration_minutes": offer["duration_minutes"],
                "total_fare": offer["total_fare"],
                "band": stop_band(offer.get("stops")),
                "carrier": offer.get("carrier") or "—",
                "provenance": offer.get("provenance") or "—",
            }
            for offer in plottable
        ],
        x_field="duration_minutes",
        x_title="Journey time (minutes)",
        y_field="total_fare",
        y_title=axis_money,
        bands=STOP_BANDS,
        tooltip=[
            {"field": "carrier", "type": "nominal", "title": "Carrier"},
            {
                "field": "total_fare",
                "type": "quantitative",
                "title": axis_money,
                "format": ",.2f",
            },
            {"field": "duration_minutes", "type": "quantitative", "title": "Minutes"},
            {"field": "band", "type": "nominal", "title": "Stops"},
            {"field": "provenance", "type": "nominal", "title": "Data"},
        ],
    )
    st.caption(":material/scatter_plot: One dot per offer — down is cheaper, left is quicker.")
    _mixed_source_note(plottable)


def _stay_tradeoff_chart(items: list[dict[str, Any]]) -> None:
    """Total cost against guest rating, coloured by cancellation policy.

    The lodging counterpart to the flight chart: price against quality rather
    than price against time, so the good corner is bottom-right instead of
    bottom-left. `kind` would be the obvious third variable and is not usable —
    hotel/apartment/guesthouse/hostel is five values against an all-pairs budget
    of three, and any bucketing of it would be arbitrary. Cancellation policy is
    genuinely three-valued, orthogonal to both axes, and the thing a traveler
    weighs against a good price.
    """
    plottable = [
        offer
        for offer in items
        if isinstance(offer.get("guest_rating"), (int, float))
        and isinstance(offer.get("total_cost"), (int, float))
    ]
    if len(plottable) < 3:
        return

    axis_money = f"Total cost ({_currency_code(plottable)})"
    _tradeoff_scatter(
        [
            {
                "guest_rating": offer["guest_rating"],
                "total_cost": offer["total_cost"],
                "band": cancel_band(offer.get("free_cancellation")),
                "name": offer.get("name") or "—",
                "provenance": offer.get("provenance") or "—",
            }
            for offer in plottable
        ],
        x_field="guest_rating",
        x_title="Guest rating",
        y_field="total_cost",
        y_title=axis_money,
        bands=CANCEL_BANDS,
        tooltip=[
            {"field": "name", "type": "nominal", "title": "Property"},
            {
                "field": "total_cost",
                "type": "quantitative",
                "title": axis_money,
                "format": ",.2f",
            },
            {"field": "guest_rating", "type": "quantitative", "title": "Rating"},
            {"field": "band", "type": "nominal", "title": "Cancellation"},
            {"field": "provenance", "type": "nominal", "title": "Data"},
        ],
    )
    st.caption(
        ":material/scatter_plot: One dot per place — down is cheaper, right is better rated."
    )
    _mixed_source_note(plottable)


def _budget_bar(by_category: dict[str, Any], ceiling: Any, currency: str) -> None:
    """Spend as one stacked bar, with the budget marked on the axis.

    Part-to-whole rather than the plain magnitude bar this replaces: the
    question the KPI row cannot answer is *which* categories fill the ceiling.
    The trade is that comparing two categories against each other is now harder
    than comparing two bar lengths — their exact figures stay in the tooltips,
    the KPI row, and `/trip/budget.md`.

    The ceiling is a gridline, not a `rule` layer, because Vega strokes an
    unstyled rule in literal `black` — invisible on the dark surface — and the
    only alternative is hardcoding a colour that is wrong in one mode. Axis
    gridlines are drawn in the active theme's own colour, so pinning a single
    gridline to the budget gets a reference line for free and keeps every
    appearance value in config.toml.

    Segments touch: the 2px surface gap the spec asks for between stacked fills
    would need that same unavailable surface colour, and a stroke around each
    segment is explicitly the wrong mechanism. Adjacent-pair CVD separation is
    validated for all seven slots in both modes, which is the gate that makes
    touching segments legible.
    """
    rows = [
        {"category": name, "amount": value, "order": index}
        for index, name in enumerate(_CATEGORY_ORDER)
        if isinstance(value := by_category.get(name), (int, float))
    ]
    # Any category the agent reports that budget.py's literal does not list.
    # Appended rather than dropped, so an unrecognised line is still visible.
    rows += [
        {"category": name, "amount": value, "order": len(_CATEGORY_ORDER) + index}
        for index, (name, value) in enumerate(sorted(by_category.items()))
        if name not in _CATEGORY_ORDER and isinstance(value, (int, float))
    ]
    if not rows:
        return

    domain = [row["category"] for row in sorted(rows, key=lambda row: row["order"])]
    known = list(_CATEGORY_ORDER) + [name for name in domain if name not in _CATEGORY_ORDER]
    ticks = [0.0]
    if isinstance(ceiling, (int, float)) and ceiling > 0:
        ticks.append(float(ceiling))

    st.vega_lite_chart(
        rows,
        {
            "mark": {"type": "bar", "height": 34},
            "encoding": {
                "x": {
                    "field": "amount",
                    "type": "quantitative",
                    "stack": "zero",
                    "title": f"Spend ({currency})",
                    # Zero stays: this is a bar, and length carries the value.
                    #
                    # `zindex: 1` lifts the axis above the marks. Vega draws
                    # gridlines under them by default, so the budget line was
                    # hidden by the very bar it exists to be crossed by —
                    # visible either side of the stack and not where it counts.
                    "axis": {"grid": True, "values": ticks, "format": ",.0f", "zindex": 1},
                },
                "color": {
                    "field": "category",
                    "type": "nominal",
                    "scale": {"domain": known},
                    "legend": dict(_LEGEND),
                },
                "order": {"field": "order", "type": "quantitative"},
                "tooltip": [
                    {"field": "category", "type": "nominal", "title": "Category"},
                    {
                        "field": "amount",
                        "type": "quantitative",
                        "title": f"Spend ({currency})",
                        "format": ",.2f",
                    },
                ],
            },
        },
        height=170,
    )


def _budget_series(history: list[dict[str, Any]], key: str) -> list[float] | None:
    """One figure's run across successive costings, for a metric sparkline.

    `summarize_budget` is called again whenever the plan changes, so the payload
    list is a history rather than a set of retries — which is what makes a
    sparkline meaningful here and not just decoration. None below two points:
    a one-point trend line is a dot.
    """
    values = [
        float(payload[key]) for payload in history if isinstance(payload.get(key), (int, float))
    ]
    return values if len(values) >= 2 else None


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
    """Choose which of several searches to show, jumping to each new one.

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
    # `default=` seeds only the first render, so a third search would leave the
    # control parked on the second while the caption claimed otherwise. Writing
    # the widget's session state before it renders re-points it, and leaves a
    # deliberate choice alone in between. Passing `default=` as well is what
    # logs "created with a default value but also had its value set via the
    # Session State API".
    #
    # The stamp carries the thread id because widget state outlives a trip:
    # "Start a new trip" swaps the record and the thread but not the selection,
    # and the new trip's second search would match the old trip's count — so a
    # count alone silently re-parks the control on the stale index.
    stamps = st.session_state.setdefault(_SEARCH_STAMPS, {})
    stamp = (st.session_state.get("thread_id"), len(searches))
    if stamps.get(key) != stamp:
        stamps[key] = stamp
        st.session_state[key] = newest

    chosen = st.segmented_control(
        "Which search",
        options=list(range(len(searches))),
        format_func=lambda index: search_label(searches[index]),
        key=key,
        # Without this the traveler can clear the selection by clicking the
        # selected segment, leaving the control blank while the table below
        # still shows a search — and the stamp above, unchanged, never
        # re-points it. There is no meaningful "no search selected" state.
        required=True,
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

    # Every costing so far, oldest first, so the cards can show where the
    # estimate has been rather than only where it landed.
    costings = record.payloads.get("summarize_budget", [])

    # Fixed rather than "stretch": a sparkline sizes to its own content and wins
    # over stretch, so the card carrying one ends up taller than its neighbours
    # and the row goes ragged. One height for all three keeps them square.
    _CARD_HEIGHT = 200

    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Estimated total",
            total,
            delta=(total - ceiling) if isinstance(total, (int, float)) and ceiling else None,
            delta_color="inverse",
            delta_description="vs. budget" if ceiling else None,
            format=money,
            border=True,
            height=_CARD_HEIGHT,
            help="Every priced line the budget analyst was given.",
            # Only this card gets a trend. The other two figures are this one
            # rearranged — `remaining` is the ceiling minus it, `used` is it over
            # the ceiling — so their sparklines would be the same series mirrored
            # and rescaled. Three copies of one shape read as three findings.
            chart_data=_budget_series(costings, "total_estimated"),
            chart_type="line",
        )
        st.metric(
            "Remaining",
            remaining,
            format=money,
            border=True,
            height=_CARD_HEIGHT,
            help="Budget less the estimate. Negative when the plan overshoots.",
        )
        with st.container(border=True, height=_CARD_HEIGHT):
            st.metric(
                "Budget used",
                used,
                format="%.0f%%",
                help="Share of the ceiling the estimate accounts for.",
            )
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
        _budget_bar(by_category, ceiling, str(budget.get("currency", "USD")))
        if isinstance(ceiling, (int, float)) and ceiling > 0:
            st.caption(
                ":material/straighten: The right-hand gridline is the budget — "
                "a bar that crosses it is the overage."
            )

flights = None
if flight_searches:
    st.subheader("Flights", anchor=False)
    flights = _pick_search(flight_searches, "flights_search")

flight_offers = _source_column(offers(flights))
if flight_offers:
    # Shape first, detail second: the chart answers "which of these is worth
    # considering", the table answers "what exactly is it".
    _flight_tradeoff_chart(flight_offers)
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
    # Same order as the flights section: shape first, then the detail.
    _stay_tradeoff_chart(stay_offers)
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
        # The icon is prefixed into the label rather than passed as `icon=`,
        # which would put it in the expander's own slot and align the labels.
        # `st.status` is an expandable carrying an icon, so that is exactly how
        # AppTest tells the two apart (`element_tree.py`, "expandable"): an
        # expander with `icon=` is parsed as a `Status` with a meaningless
        # `state`, vanishing from `at.expander` and polluting `at.status` —
        # which `test_pages.py` compares element for element. Nothing renders
        # differently in the browser; the page just stops being testable.
        with st.expander(f"{icon} {label}"):
            st.markdown(record.files[path])

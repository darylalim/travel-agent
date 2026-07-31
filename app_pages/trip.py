"""Trip page: the plan as figures rather than conversation.

Reads the structured payloads captured while the agent worked, not the
markdown it wrote — so fares, durations and the `synthetic` flag arrive as
data instead of prose. Offers are passed to `st.dataframe` as plain
`list[dict]`; there is no pandas import here, so pandas stays a transitive
dependency of Streamlit rather than an undeclared direct one.

The title lives in `streamlit_app.py`; pages do not set their own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st

from travel_agent.ui import (
    CANCEL_BANDS,
    CURRENCY_PRESETS,
    ITINERARY_PATH,
    STOP_BANDS,
    WORKSPACE_FILES,
    cancel_band,
    costing_series,
    currency_code,
    money_format,
    offers,
    search_label,
    source_column,
    stop_band,
)

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


# Shared by both scatters, so their legends read as one system. The padding is
# not cosmetic: Vega's default leaves 11-14px between horizontal entries, and an
# 8px swatch fills nearly all of it, so a long label runs into the swatch of the
# entry after it ("No free cancellation● Not stated"). There is room to spare at
# this width — the default is simply tight.
_LEGEND = {"orient": "bottom", "title": None, "columnPadding": 18, "labelOffset": 6}


def _provenance_note(plotted: list[dict[str, Any]]) -> None:
    """Say on the chart itself where its figures came from.

    The standing notice lives at the top of the page, above the KPI row, the
    spend bar, the flight chart and its table — so by the time the stays scatter
    is on screen it has scrolled away, and the tooltip carrying `provenance` is
    unreachable on a touch device. The table below each chart labels every row;
    the chart is a surface reaching the traveler with no model in between and has
    to label itself too, whether the set is wholly synthetic or a blend.

    Reads `provenance`, which `source_column` already derived, rather than
    re-reading `synthetic`: two independent readings of one flag is how the
    table and the chart drift apart.
    """
    if not plotted:
        return
    sample = sum(1 for offer in plotted if offer.get("provenance") == "Sample data")
    if not sample:
        return
    if sample == len(plotted):
        st.caption(
            ":material/warning: Every dot here is sample data — illustrative planning "
            "figures, not real availability, and nothing here is bookable."
        )
    else:
        st.caption(
            f":material/warning: Mixed sources — {sample} of {len(plotted)} plotted "
            f"offers {'is' if sample == 1 else 'are'} sample data. "
            "The table below labels each row."
        )


@dataclass(frozen=True)
class _Tradeoff:
    """Everything that differs between the flight and stay tradeoff scatters.

    Both charts answer the same question — is this offer worth considering, given
    something cheaper or better exists — so they share one spec, one legend, one
    caption convention and one provenance note. Only the fields, titles and band
    vocabulary change, and they live here so a third chart is a fourth instance
    rather than a third copy of forty lines.
    """

    x_field: str
    x_title: str
    x_tooltip: str
    y_field: str
    y_prefix: str
    name_field: str
    name_title: str
    bands: tuple[str, ...]
    band_title: str
    band_source: str
    band_of: Callable[[Any], str | None]
    caption: str


_FLIGHT_TRADEOFF = _Tradeoff(
    x_field="duration_minutes",
    x_title="Journey time (minutes)",
    x_tooltip="Minutes",
    y_field="total_fare",
    y_prefix="Total fare",
    name_field="carrier",
    name_title="Carrier",
    bands=STOP_BANDS,
    band_title="Stops",
    band_source="stops",
    band_of=stop_band,
    caption=":material/scatter_plot: One dot per offer — down is cheaper, left is quicker.",
)

# Price against quality rather than price against time, so the good corner is
# bottom-right instead of bottom-left. `kind` would be the obvious third variable
# and is not usable: hotel/apartment/guesthouse/hostel is five values against an
# all-pairs budget of three, and any bucketing of it would be arbitrary.
# Cancellation policy is genuinely three-valued, orthogonal to both axes, and the
# thing a traveler weighs against a good price.
_STAY_TRADEOFF = _Tradeoff(
    x_field="guest_rating",
    x_title="Guest rating",
    x_tooltip="Rating",
    y_field="total_cost",
    y_prefix="Total cost",
    name_field="name",
    name_title="Property",
    bands=CANCEL_BANDS,
    band_title="Cancellation",
    band_source="free_cancellation",
    band_of=cancel_band,
    caption=":material/scatter_plot: One dot per place — down is cheaper, right is better rated.",
)


def _tradeoff_chart(items: list[dict[str, Any]], cfg: _Tradeoff) -> list[dict[str, Any]]:
    """Draw one "cheap versus good" scatter; return the offers actually plotted.

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

    Colour is the only thing this takes from the theme: `range` is deliberately
    absent, so `theme="streamlit"` fills it from `chartCategoricalColors` and no
    hex is duplicated into Python. Geometry — mark size, opacity, the chart
    height, the legend padding — is set here rather than in `config.toml`, which
    has no vocabulary for it.

    An offer needs every plotted figure to be drawable, its band included: each
    band makes a positive claim, so `band_of` returns None rather than guessing,
    and a dot with no band would fall outside the pinned domain — drawn, but
    keyed to nothing in the legend. Bands are computed once per offer and carried
    forward, so the filter and the projection cannot drift apart.
    """
    banded = [
        (offer, cfg.band_of(offer.get(cfg.band_source)))
        for offer in items
        if isinstance(offer.get(cfg.x_field), (int, float))
        and isinstance(offer.get(cfg.y_field), (int, float))
    ]
    plotted = [(offer, band) for offer, band in banded if band is not None]
    # Two points describe a line, not a tradeoff; the table says it better.
    if len(plotted) < 3:
        return []

    axis_money = f"{cfg.y_prefix} ({currency_code([offer for offer, _ in plotted])})"
    st.vega_lite_chart(
        [
            {
                cfg.x_field: offer[cfg.x_field],
                cfg.y_field: offer[cfg.y_field],
                "band": band,
                "name": offer.get(cfg.name_field) or "—",
                "provenance": offer.get("provenance") or "—",
            }
            for offer, band in plotted
        ],
        {
            # ~10px across, satisfying the >=8px marker floor. Semi-opaque
            # instead of the 2px surface ring the spec asks for on overlapping
            # dots: that ring has to be painted in the surface colour, which
            # differs between light and dark, and a Vega spec cannot read the
            # active theme — hardcoding either would break one mode.
            "mark": {"type": "circle", "size": 90, "opacity": 0.85},
            "encoding": {
                "x": {
                    "field": cfg.x_field,
                    "type": "quantitative",
                    "title": cfg.x_title,
                    "scale": {"zero": False, "nice": True, "padding": 14},
                    "axis": {"grid": False},
                },
                "y": {
                    "field": cfg.y_field,
                    "type": "quantitative",
                    "title": axis_money,
                    "scale": {"zero": False, "nice": True, "padding": 14},
                    "axis": {"grid": True, "format": ",.0f"},
                },
                "color": {
                    "field": "band",
                    "type": "nominal",
                    "scale": {"domain": list(cfg.bands)},
                    "legend": dict(_LEGEND),
                },
                "tooltip": [
                    {"field": "name", "type": "nominal", "title": cfg.name_title},
                    {
                        "field": cfg.y_field,
                        "type": "quantitative",
                        "title": axis_money,
                        "format": ",.2f",
                    },
                    {"field": cfg.x_field, "type": "quantitative", "title": cfg.x_tooltip},
                    {"field": "band", "type": "nominal", "title": cfg.band_title},
                    {"field": "provenance", "type": "nominal", "title": "Data"},
                ],
            },
        },
        height=300,
    )
    st.caption(cfg.caption)
    return [offer for offer, _ in plotted]


def _outbound_only_note(plotted: list[dict[str, Any]]) -> None:
    """Flag that a round trip's stops and journey time cover the outbound leg.

    `map_offer` reads both off `slices[0]`, and `search_flights` builds a second
    slice whenever a return date is given — so on a return itinerary the colour
    band and the x position describe half the journey, and an offer whose return
    leg connects twice still lands in the chart's "Nonstop" hue. The table's own
    Duration help says the same thing.

    Sample offers carry one notional duration rather than per-leg figures, so on
    those the note is conservative rather than exact. That is the safe direction
    for a figure already labelled synthetic, and it avoids keying the wording off
    the provider name — which says nothing reliable, as the honesty invariant
    elsewhere on this page keeps pointing out.
    """
    if any(offer.get("return_date") for offer in plotted):
        st.caption(
            ":material/flight_land: Stops and journey time describe the outbound leg; "
            "the return is not in these figures."
        )


def _truncation_note(payload: dict[str, Any] | None) -> None:
    note = (payload or {}).get("truncated")
    if isinstance(note, str):
        st.caption(f":material/filter_list: {note}")


def _cabin_note(payload: dict[str, Any] | None) -> None:
    """Say when a search did not come back in the cabin it asked for.

    Scoped to the selected search, like the truncation note beside it, rather
    than joining the standing caveats at the top of the page: those accumulate
    across the whole trip and are never cleared, which is right for "these
    prices are not real" and wrong for a fact about one query's results.
    """
    note = (payload or {}).get("cabin_note")
    if isinstance(note, str):
        st.caption(f":material/flight_class: {note}")


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
    money = CURRENCY_PRESETS.get(str(budget.get("currency", "USD")), "%.2f")

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
            help=(
                "Every priced line the budget analyst was given. The bars are each "
                "costing run so far, not one plan tracked over time."
            ),
            # Only this card carries the costings. The other two figures are this
            # one rearranged — `remaining` is the ceiling minus it, `used` is it
            # over the ceiling — so their sparklines would be the same series
            # mirrored and rescaled. Three copies of one shape read as three
            # findings.
            #
            # Bars rather than a line. The marks are successive costings in the
            # order they ran, which is not the same as one plan revised over
            # time: BUDGET_PROMPT asks the analyst to propose cuts, and the main
            # agent can cost several variants against a stateless subagent inside
            # one turn. A line asserts a trend through them; bars show them as
            # the separate figures they are. Same reasoning that stops
            # `_pick_search` merging two searches into one table.
            chart_data=costing_series(costings, "total_estimated"),
            chart_type="bar",
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
        # One hue, one bar per category. The job is comparing magnitudes and a
        # bar length does that better than anything else; a single series needs
        # no legend, because the subheading already names what is plotted.
        #
        # Deliberately *not* a stacked part-to-whole bar with the budget marked
        # on the axis. That was built and reverted, for two measured reasons.
        #
        # `axis.values` cannot extend a scale domain, so Vega's `validTicks`
        # silently drops a tick sitting past the data: with a 3,440 estimate
        # against a 4,000 budget the rendered labels were just ["0"], no budget
        # gridline and no value scale at all. The reference line only ever
        # appeared when over budget — the one case the KPI delta and the notice
        # above already state — while the caption claimed it unconditionally.
        #
        # And pinning the colour domain to all seven categories, which is what
        # keeps hues stable across re-costings, means the segments that actually
        # touch are whichever categories the trip has, not the palette's adjacent
        # pairs. An ordinary {flights, lodging, food, other} puts slot 3 against
        # slot 7 at CVD ΔE 0.8 and 6.3 normal-vision — both hard failures — and a
        # stacked bar has no gap, stroke or label left to separate them once
        # colour fails. Only adjacent pairs were validated, which is the wrong
        # pairlist for a form whose adjacencies depend on the data.
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

flight_offers = source_column(offers(flights))
if flight_offers:
    # Shape first, detail second: the chart answers "which of these is worth
    # considering", the table answers "what exactly is it".
    _plotted = _tradeoff_chart(flight_offers, _FLIGHT_TRADEOFF)
    _outbound_only_note(_plotted)
    _provenance_note(_plotted)
    fare_format = money_format(flight_offers)
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
                # Not "whole journey", which this said and is not true of a
                # return trip: `map_offer` reads duration off `slices[0]`, so on
                # a round trip both this and Stops describe the outbound leg.
                help="Outbound leg, layovers included. A return leg is not counted here.",
            ),
            "stops": st.column_config.NumberColumn(
                "Stops", format="%d", width="small", alignment="center"
            ),
            "depart_time_local": st.column_config.TextColumn("Departs", width="small"),
            "cabin": st.column_config.TextColumn(
                "Cabin",
                width="small",
                help="What the offer came back as, which need not be what was "
                "searched. 'Mixed' means its legs are not all the same class.",
            ),
            "provenance": st.column_config.TextColumn(
                "Data",
                width="small",
                help="Sample data is illustrative and cannot be booked.",
            ),
        },
    )
    _truncation_note(flights)
    _cabin_note(flights)
elif flight_searches:
    # A search that legitimately found nothing. The caveat above still stands:
    # an unannotated empty result would read as "we checked real inventory".
    st.info("That search returned no flights.", icon=":material/search_off:")

stays = None
if stay_searches:
    st.subheader("Places to stay", anchor=False)
    stays = _pick_search(stay_searches, "stays_search")

stay_offers = source_column(offers(stays))
if stay_offers:
    # Same order as the flights section: shape first, then the detail.
    _provenance_note(_tradeoff_chart(stay_offers, _STAY_TRADEOFF))
    rate_format = money_format(stay_offers)
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
            "due_at_accommodation",
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
            # Shown beside the total, never folded into it — Duffel's docs
            # disagree about whether this amount is already inside
            # `total_amount`, so `map_stay_result` passes both through
            # unsummed. `column_order` is a whitelist, so omitting this key
            # here is what silently dropped the figure: the traveler read a
            # column headed "Total" that may exclude a charge due at the desk,
            # on the one surface the agent cannot caveat. Sample offers never
            # carry it and render as `placeholder` instead.
            "due_at_accommodation": st.column_config.NumberColumn(
                "Due at property",
                format=rate_format,
                help=(
                    "Charged at the property rather than upfront. Reported "
                    "separately because sources disagree about whether it is "
                    "already included in the total, so the two are never added "
                    "together."
                ),
            ),
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

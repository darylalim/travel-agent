"""Duffel Stays: live lodging search.

Read-only, exactly like the flight side. This module runs one search and reads
the results back; it never creates a quote or a booking, so nothing is reserved
and no payment is taken. `.claude/hooks/no-booking.sh` denies the URL shapes for
`/stays/quotes` and `/stays/bookings` the same way it denies `/air/orders`.

Contract, taken from Duffel's v2 Stays reference:

- `POST https://api.duffel.com/stays/search`, with the same headers as Air.
- Results arrive at `data.results[]` — **not** `data.offers`, which is the Air
  shape. Price sits at the top of each result (`cheapest_rate_total_amount`,
  `cheapest_rate_currency`); the property sits under a nested `accommodation`.
- **Stays carries no `live_mode` field**, unlike an Air offer. So this module
  cannot read synthetic-ness off the payload the way `map_offer` does — it is
  decided by the token and never reaches here at all under a test one. See
  `DuffelProvider.search_stays`.
- No pagination, and no `supplier_timeout`: that knob is Air-only, so the HTTP
  timeout here is a flat constant rather than derived from one.
- Same `{"errors": [...]}` envelope as Air, so `describe_error` is reused.

**Coordinates, not place names.** `location.geographic_coordinates` is
required, and Duffel's own guide says you "will probably need to use a
geocoding service" to get there from a city name. Rather than take a geocoder
dependency and an API key, this module ships a small static table and refuses
what it does not know: an unmatched city is an error naming what is supported,
never a nearest-match guess at somewhere the traveler did not ask for.

**Two prices, and we do not add them up.** A rate carries `total_amount` and
`due_at_accommodation_amount`, and Duffel's documentation is inconsistent about
whether the second is part of the first: the search-result schema calls it "the
amount *of the rate* that is due at the accommodation" (a subset), while the
Stays key-concepts page defines the total as covering "taxes and fees due at
time of booking" (which reads as additive). Both are surfaced verbatim and
neither is combined — summing them double-counts under one reading, and
subtracting understates under the other. Either way the traveler would be shown
a confident number nobody can source.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx

from travel_agent.tools.duffel import (
    API_BASE,
    DUFFEL_SOURCE,
    DuffelError,
    parse_amount,
    raise_for_status,
    seconds_until,
)

logger = logging.getLogger(__name__)

# Duffel gives Stays no supplier-timeout knob, so there is nothing to derive
# this from — it is a flat number, generous because accommodation suppliers are
# documented as occasionally slow. Deliberately not an env var: adding a knob
# nobody has asked for also means adding it to conftest's _PROVIDER_ENV.
_HTTP_TIMEOUT_S = 45.0

# City-centre coordinates, so the radius has to cover a central hotel district
# rather than a neighbourhood.
_SEARCH_RADIUS_KM = 5

# `max_nightly_rate` is documented on the tool as a USD figure, and every
# SampleProvider offer is priced in USD. A live search is not: Tokyo comes back
# in JPY. Comparing the two silently empties the result set, so the ceiling is
# only applied to offers quoted in this currency — see `_within_ceiling`.
_CEILING_CURRENCY = "USD"

# A curated table, not an attempt at completeness. Duffel Stays needs
# coordinates and this project takes no geocoder dependency, so the supported
# set is finite and says so: `resolve_location` names every entry when it
# refuses. Extending it is a one-line data edit.
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "amsterdam": (52.3676, 4.9041),
    "bangkok": (13.7563, 100.5018),
    "barcelona": (41.3874, 2.1686),
    "berlin": (52.5200, 13.4050),
    "chicago": (41.8781, -87.6298),
    "dubai": (25.2048, 55.2708),
    "hong kong": (22.3193, 114.1694),
    "istanbul": (41.0082, 28.9784),
    "kyoto": (35.0116, 135.7681),
    "lisbon": (38.7223, -9.1393),
    "london": (51.5072, -0.1276),
    "los angeles": (34.0522, -118.2437),
    "madrid": (40.4168, -3.7038),
    "mexico city": (19.4326, -99.1332),
    "new york": (40.7128, -74.0060),
    "osaka": (34.6937, 135.5023),
    "paris": (48.8566, 2.3522),
    "rome": (41.9028, 12.4964),
    "san francisco": (37.7749, -122.4194),
    "seoul": (37.5665, 126.9780),
    "singapore": (1.3521, 103.8198),
    "sydney": (-33.8688, 151.2093),
    "tokyo": (35.6762, 139.6503),
    "vienna": (48.2082, 16.3738),
}


def resolve_location(location: str) -> tuple[float, float]:
    """Coordinates for a city name, or a ValueError naming what is supported.

    Exact match only, after case and whitespace normalisation. No fuzzy or
    nearest-match fallback: searching Osaka for someone who asked about Kobe
    and labelling the results "Kobe" is the same misrepresentation as quoting
    sample data as real, and it would be far harder to notice.
    """
    coordinates = CITY_COORDINATES.get(location.strip().lower())
    if coordinates is None:
        supported = ", ".join(sorted(name.title() for name in CITY_COORDINATES))
        raise DuffelError(
            f"Live lodging search has no coordinates for {location!r}. Duffel Stays "
            f"searches by latitude and longitude, and this project maps a fixed set "
            f"of cities: {supported}. Search one of those, or set "
            f"TRAVEL_AGENT_PROVIDER=sample-data for illustrative offers anywhere."
        )
    return coordinates


def _nights(check_in: str, check_out: str) -> int:
    """Nights between two ISO dates, floored at zero.

    The tool layer has already rejected a non-ISO or reversed range by the time
    a provider runs, so this only guards the pure mapper against being called
    directly with something the tool would not have passed on.
    """
    try:
        span = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
    except (TypeError, ValueError):
        return 0
    return max(span, 0)


def _guest_rating(accommodation: dict[str, Any]) -> float | None:
    """Reviewer score on Duffel's 0-10 scale, or None when unrated.

    Reads `review_score` and **never** falls back to `rating`, which is a 1-5
    star classification. The Trip page renders this in a
    `ProgressColumn(min_value=0, max_value=10)` and plots it on a 0-10 axis, so
    a 4-star property backfilled into that field would render as 4/10 — a good
    hotel shown as a poor one, from a number nobody wrote down.
    """
    return parse_amount(accommodation.get("review_score"))


def _cheapest_rate(accommodation: dict[str, Any]) -> dict[str, Any] | None:
    """The cheapest priced rate across every room, or None if none is priced.

    Duffel does not label which nested rate produced the result-level
    `cheapest_rate_total_amount`, so this recomputes it rather than matching on
    the formatted string. The point is to report the cancellation policy that
    belongs to the price actually being shown, not to an arbitrary other room.
    """
    priced = [
        (rate, amount)
        for room in accommodation.get("rooms") or []
        if isinstance(room, dict)
        for rate in room.get("rates") or []
        if isinstance(rate, dict) and (amount := parse_amount(rate.get("total_amount"))) is not None
    ]
    return min(priced, key=lambda pair: pair[1])[0] if priced else None


def _free_cancellation(rate: dict[str, Any] | None) -> bool | None:
    """Whether a rate has a window in which cancelling costs nothing.

    Three outcomes, and the distance between them is the whole point:

    - `None` — nothing to read. `cancel_band` renders that as "Not stated",
      which is not the same claim as "no free cancellation".
    - `False` — an *empty* timeline. Duffel documents that as non-refundable,
      so it is a real finding rather than an absence of one.
    - `True` — some window refunds the full amount.

    A timeline that only ever refunds part of the price is `False`. Cancellable
    for a fee is not free cancellation, and a traveler choosing on flexibility
    would read a `True` here as exactly that.
    """
    if rate is None:
        return None
    timeline = rate.get("cancellation_timeline")
    if not isinstance(timeline, list):
        return None
    if not timeline:
        return False
    total = parse_amount(rate.get("total_amount"))
    if total is None:
        return None
    return any(
        parse_amount(entry.get("refund_amount")) == total
        for entry in timeline
        if isinstance(entry, dict)
    )


def map_stay_result(
    result: dict[str, Any],
    location: str,
    check_in: str,
    check_out: str,
    guests: int,
) -> dict[str, Any]:
    """Normalise one `data.results[]` entry into the shape the tools return.

    Pure and transport-free so it can be tested without the network, the same
    way `map_offer` is. The query parameters are threaded in because a Stays
    result echoes only its dates back, and the UI builds a search label from
    the offer rather than from the call that produced it.

    `synthetic` is hardcoded False rather than read from the payload: Stays has
    no `live_mode` field, and a test token never reaches this module at all.
    """
    accommodation = result.get("accommodation") or {}
    address = (accommodation.get("location") or {}).get("address") or {}

    nights = _nights(check_in, check_out)
    total_cost = parse_amount(result.get("cheapest_rate_total_amount"))
    due_at_accommodation = parse_amount(result.get("cheapest_rate_due_at_accommodation_amount"))

    offer: dict[str, Any] = {
        "source": DUFFEL_SOURCE,
        "synthetic": False,
        "name": accommodation.get("name"),
        "location": address.get("city_name") or location,
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "guests": guests,
        "nightly_rate": round(total_cost / nights, 2)
        if total_cost is not None and nights
        else None,
        "total_cost": round(total_cost, 2) if total_cost is not None else None,
        # Surfaced beside the total, never folded into it — see the module
        # docstring on why the two are not summed.
        "due_at_accommodation": round(due_at_accommodation, 2)
        if due_at_accommodation is not None
        else None,
        "currency": result.get("cheapest_rate_currency"),
        # Stays results expire like Air offers, so the agent must re-search
        # rather than quote a stale price.
        "expires_at": result.get("expires_at"),
        "expires_in_seconds": seconds_until(result.get("expires_at")),
        # No "origin" key, ever: ui.py's search_label uses its presence to tell
        # a flight search from a lodging one, so adding it here would relabel
        # this search as a flight.
        #
        # No "kind" key either — Duffel's search result has no property-type
        # field, and guessing "hotel" would put an invented classification in a
        # column the traveler reads as fact.
    }

    # Omitted rather than nulled when unknown. The UI distinguishes "absent"
    # from "false" for both of these, and a missing key is what reaches the
    # honest branch.
    if (rating := _guest_rating(accommodation)) is not None:
        offer["guest_rating"] = rating
    if (free := _free_cancellation(_cheapest_rate(accommodation))) is not None:
        offer["free_cancellation"] = free
    return offer


def _within_ceiling(offer: dict[str, Any], max_nightly_rate: float) -> bool:
    """Whether an offer clears a nightly-rate ceiling that can be compared to it.

    An offer priced in another currency is **kept**, not dropped. The ceiling
    arrives as a bare USD number, so testing it against a JPY nightly rate
    would quietly return nothing for every Tokyo search under a realistic
    budget. Dropping inventory nobody can compare is a worse failure than
    showing it: the offer carries its own `currency`, so an over-budget room is
    visible as such, while a silently empty result set is not.
    """
    if offer.get("currency") != _CEILING_CURRENCY:
        return True
    nightly = offer.get("nightly_rate")
    return nightly is not None and nightly <= max_nightly_rate


def fetch_stays(
    http: httpx.Client,
    location: str,
    check_in: str,
    check_out: str,
    guests: int,
    max_nightly_rate: float | None,
) -> list[dict[str, Any]]:
    """Search Duffel Stays and return mapped offers, cheapest first.

    Takes the caller's pooled client rather than building one, so flights and
    lodging share a connection pool and one set of auth headers. Only the
    timeout differs, and it is overridden per request.
    """
    latitude, longitude = resolve_location(location)  # raises before any request

    payload = {
        "data": {
            # One room for the party. Multi-room search would need a tool-level
            # parameter, and splitting guests across rooms is a booking
            # decision rather than a search one.
            "rooms": 1,
            "check_in_date": check_in,
            "check_out_date": check_out,
            "guests": [{"type": "adult"} for _ in range(guests)],
            "location": {
                "radius": _SEARCH_RADIUS_KM,
                "geographic_coordinates": {"latitude": latitude, "longitude": longitude},
            },
        }
    }

    try:
        response = http.post(
            f"{API_BASE}/stays/search",
            json=payload,
            timeout=_HTTP_TIMEOUT_S,
        )
    except httpx.TimeoutException as exc:
        # Deliberately does not name DUFFEL_SUPPLIER_TIMEOUT_MS the way the
        # flight path does: that knob is Air-only and has no effect here.
        raise DuffelError(
            f"Duffel Stays search timed out after {_HTTP_TIMEOUT_S:.0f}s. "
            "Accommodation suppliers can be slow; try a narrower search."
        ) from exc
    except httpx.HTTPError as exc:
        raise DuffelError(f"Could not reach Duffel: {exc}") from exc

    raise_for_status(response)

    # `results`, not `offers` — the Air shape does not apply here.
    raw = (response.json().get("data") or {}).get("results") or []
    mapped = [map_stay_result(r, location, check_in, check_out, guests) for r in raw]

    # A room with no usable price would sort to the front as the cheapest, so
    # drop it rather than quote a phantom rate.
    priced = [offer for offer in mapped if offer["total_cost"] is not None]
    if dropped := len(mapped) - len(priced):
        logger.warning("Discarded %d Duffel Stays result(s) with no usable price.", dropped)

    if max_nightly_rate is not None:
        priced = [offer for offer in priced if _within_ceiling(offer, max_nightly_rate)]

    priced.sort(key=lambda offer: offer["total_cost"])
    return priced

"""Flight and lodging search, across pluggable providers.

Two providers are registered: `sample-data` (synthetic offers, the default)
and `duffel` (live flight search; lodging still falls through to sample data).
Select one with `TRAVEL_AGENT_PROVIDER`.

The safety property this module exists to hold: **the traveler must never be
shown synthetic inventory described as real**. It is enforced in three places
that have to stay in agreement:

1. Every offer carries `synthetic: bool` alongside its `source`.
2. Providers declare `synthetic_note(kind)` so a search returning *no* offers
   still says whether it was querying real inventory. Keying off the offers
   alone would silently drop the caveat on an empty result set.
3. `_wrap` attaches the resulting `warning`, and the tool docstrings tell the
   model to key off `warning`/`synthetic` — never the provider name, which
   says nothing about whether a given offer is real. Duffel in test mode and
   Duffel lodging are both synthetic under a provider named `duffel`.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import date, timedelta
from functools import cache
from typing import Literal, Protocol

from langchain_core.tools import tool

SAMPLE_SOURCE = "sample-data"
SearchKind = Literal["flights", "stays"]

SAMPLE_DISCLAIMER = (
    "These are synthetic sample offers, not live availability. Prices, times, "
    "and seat/room counts are illustrative only and nothing here is bookable. "
    "Tell the traveler this explicitly when you use these figures."
)
_GENERIC_SYNTHETIC_WARNING = (
    "Some of these offers are synthetic and do not reflect real availability "
    "or real prices. Say so explicitly when you use them."
)


class AvailabilityProvider(Protocol):
    """Source of flight and lodging offers."""

    name: str

    def synthetic_note(self, kind: SearchKind) -> str | None:
        """Disclaimer for this kind of search, or None if it returns real inventory.

        Consulted even when a search returns nothing, so an empty result set
        is never mistaken for "we checked real inventory and found none".
        """
        ...

    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: str,
        return_date: str | None,
        travelers: int,
    ) -> list[dict]: ...

    def search_stays(
        self,
        location: str,
        check_in: str,
        check_out: str,
        guests: int,
        max_nightly_rate: float | None,
    ) -> list[dict]: ...


def _seed(*parts: str) -> int:
    """Stable pseudo-random seed derived from query parameters.

    Deterministic on purpose: the same query returns the same offers, so the
    agent does not see prices shift underneath it mid-conversation and the
    tests are reproducible.
    """
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16)


def _cost(offer: dict, field: str) -> float:
    """Sort key for offer dicts, whose values are a mixed-type union."""
    value = offer[field]
    return float(value) if isinstance(value, (int, float)) else float("inf")


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


class SampleProvider:
    """Deterministic synthetic offers for development and demos."""

    name = SAMPLE_SOURCE

    _CARRIERS = ("Meridian Air", "Northwind", "Blue Corridor", "Anser Airways")
    _STAY_STYLES = (
        ("Central boutique hotel", "hotel", 8.9),
        ("Riverside apartment", "apartment", 8.6),
        ("Business district hotel", "hotel", 8.2),
        ("Quiet guesthouse", "guesthouse", 9.1),
        ("Budget hostel, private room", "hostel", 7.8),
    )

    def synthetic_note(self, kind: SearchKind) -> str | None:
        return SAMPLE_DISCLAIMER

    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: str,
        return_date: str | None,
        travelers: int,
    ) -> list[dict]:
        depart = _parse_date(depart_date, "depart_date")
        if return_date is not None:
            returning = _parse_date(return_date, "return_date")
            if returning < depart:
                raise ValueError("return_date cannot fall before depart_date.")

        seed = _seed(origin, destination, depart_date, return_date or "one-way")
        base_fare = 180 + (seed % 640)
        offers = []

        for index in range(4):
            carrier = self._CARRIERS[(seed + index) % len(self._CARRIERS)]
            stops = index % 3 if index else 0
            # Nonstop carries a premium; each stop discounts the fare.
            fare = round((base_fare * (1.28 if stops == 0 else 1.0 - 0.09 * stops)) + index * 23, 2)
            # Includes notional layover time, so it is comparable with the
            # slice-level duration Duffel reports.
            duration_minutes = 240 + (seed % 300) + stops * 95 + index * 15
            depart_hour = 6 + ((seed // (index + 1)) % 15)

            offers.append(
                {
                    "source": SAMPLE_SOURCE,
                    "synthetic": True,
                    "carrier": carrier,
                    "origin": origin.upper(),
                    "destination": destination.upper(),
                    "depart_date": depart_date,
                    "return_date": return_date,
                    "depart_time_local": f"{depart_hour:02d}:{(seed + index * 7) % 60:02d}",
                    "stops": stops,
                    "duration_minutes": duration_minutes,
                    "cabin": "economy",
                    "fare_per_traveler": fare,
                    "total_fare": round(fare * travelers, 2),
                    "travelers": travelers,
                    "currency": "USD",
                }
            )

        offers.sort(key=lambda offer: _cost(offer, "total_fare"))
        return offers

    def search_stays(
        self,
        location: str,
        check_in: str,
        check_out: str,
        guests: int,
        max_nightly_rate: float | None,
    ) -> list[dict]:
        start = _parse_date(check_in, "check_in")
        end = _parse_date(check_out, "check_out")
        nights = (end - start).days
        if nights < 1:
            raise ValueError("check_out must be at least one day after check_in.")

        seed = _seed(location, check_in, check_out, str(guests))
        base_rate = 70 + (seed % 210)
        offers = []

        for index, (label, kind, rating) in enumerate(self._STAY_STYLES):
            nightly = round(base_rate * (1.45 - index * 0.18) + (seed % (index + 4)), 2)
            if max_nightly_rate is not None and nightly > max_nightly_rate:
                continue
            offers.append(
                {
                    "source": SAMPLE_SOURCE,
                    "synthetic": True,
                    "name": f"{label} ({location.title()})",
                    "kind": kind,
                    "location": location,
                    "check_in": check_in,
                    "check_out": check_out,
                    "nights": nights,
                    "guests": guests,
                    "nightly_rate": nightly,
                    "total_cost": round(nightly * nights, 2),
                    "guest_rating": rating,
                    "free_cancellation": (seed + index) % 2 == 0,
                    "currency": "USD",
                }
            )

        offers.sort(key=lambda offer: _cost(offer, "total_cost"))
        return offers


def _load_duffel() -> AvailabilityProvider:
    """Imported lazily so httpx and the token check only load when selected."""
    from travel_agent.tools.duffel import DuffelProvider

    return DuffelProvider()


_PROVIDERS: dict[str, Callable[[], AvailabilityProvider]] = {
    SAMPLE_SOURCE: SampleProvider,
    "duffel": _load_duffel,
}


@cache
def _build_provider(key: str) -> AvailabilityProvider:
    """Construct a provider once per process.

    Cached so provider setup — and its side effects, like the live-token
    warning — happens once rather than on every tool call. Failed
    construction is not cached, so a fixed token takes effect immediately.
    Tests clear this via the autouse fixture in conftest.py.
    """
    try:
        factory = _PROVIDERS[key]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown TRAVEL_AGENT_PROVIDER={key!r}. Registered providers: {known}"
        ) from None
    return factory()


def get_provider() -> AvailabilityProvider:
    """Return the configured provider, defaulting to sample data."""
    return _build_provider(os.getenv("TRAVEL_AGENT_PROVIDER", SAMPLE_SOURCE))


def _wrap(offers: list[dict], provider: AvailabilityProvider, kind: SearchKind) -> dict:
    """Package offers for the model, warning whenever any are synthetic.

    The provider's own note is authoritative and applies even to an empty
    result list; the per-offer `synthetic` flags are a backstop for providers
    that mix real and synthetic results in one response.
    """
    note = provider.synthetic_note(kind)
    payload: dict = {
        "provider": provider.name,
        "sources": sorted({str(offer.get("source", "unknown")) for offer in offers}),
        "count": len(offers),
        "offers": offers,
    }
    if note:
        payload["warning"] = note
    elif any(offer.get("synthetic") for offer in offers):
        payload["warning"] = _GENERIC_SYNTHETIC_WARNING
    return payload


@tool
def search_flights(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str | None = None,
    travelers: int = 1,
) -> dict:
    """Search flight options between two airports.

    Returns offers sorted cheapest first.

    Check the response for a `warning` key, and each offer for `synthetic:
    true`. Either means those offers are not real availability: describe them
    to the traveler as illustrative planning figures, never as real prices or
    something bookable. Do not judge this by the `provider` name — a live
    provider can return synthetic offers, and a search returning no offers at
    all still carries the warning.

    Offers from real inventory carry `expires_at` and `expires_in_seconds`.
    Search again rather than quoting one that has expired.

    Args:
        origin: Origin airport IATA code, e.g. "SFO".
        destination: Destination airport IATA code, e.g. "NRT".
        depart_date: Outbound date as YYYY-MM-DD.
        return_date: Return date as YYYY-MM-DD. Omit for one-way.
        travelers: Number of travelers on the booking.
    """
    if travelers < 1:
        return {"error": "travelers must be at least 1."}
    try:
        # Validated here rather than per provider, so the tool behaves the
        # same way whichever provider is configured.
        depart = _parse_date(depart_date, "depart_date")
        if return_date is not None and _parse_date(return_date, "return_date") < depart:
            return {"error": "return_date cannot fall before depart_date."}
        provider = get_provider()
        offers = provider.search_flights(origin, destination, depart_date, return_date, travelers)
    except ValueError as exc:
        return {"error": str(exc)}
    return _wrap(offers, provider, "flights")


@tool
def search_stays(
    location: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    max_nightly_rate: float | None = None,
) -> dict:
    """Search places to stay in a location for a date range.

    Returns offers sorted by total cost.

    Check the response for a `warning` key, and each offer for `synthetic:
    true`. Either means those offers are not real availability: describe them
    to the traveler as illustrative planning figures, never as real prices or
    something bookable. Do not judge this by the `provider` name — lodging is
    synthetic even when the provider is a live flight source.

    Args:
        location: City or neighbourhood to search, e.g. "Kyoto" or "Shibuya".
        check_in: Arrival date as YYYY-MM-DD.
        check_out: Departure date as YYYY-MM-DD.
        guests: Number of guests.
        max_nightly_rate: Optional ceiling on nightly rate, in USD.
    """
    if guests < 1:
        return {"error": "guests must be at least 1."}
    try:
        if _parse_date(check_out, "check_out") <= _parse_date(check_in, "check_in"):
            return {"error": "check_out must be at least one day after check_in."}
        provider = get_provider()
        offers = provider.search_stays(location, check_in, check_out, guests, max_nightly_rate)
    except ValueError as exc:
        return {"error": str(exc)}
    return _wrap(offers, provider, "stays")


@tool
def date_offset(start_date: str, days: int) -> dict:
    """Shift an ISO date by a number of days, for building day-by-day plans.

    Returns `{"date": "YYYY-MM-DD"}`, or `{"error": ...}` if the input was not
    an ISO date. Never write the error text into an itinerary as if it were a
    date.

    Args:
        start_date: The reference date as YYYY-MM-DD.
        days: Days to add; negative values move backwards.
    """
    try:
        shifted = _parse_date(start_date, "start_date") + timedelta(days=days)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"date": shifted.isoformat()}

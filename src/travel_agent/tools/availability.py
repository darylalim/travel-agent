"""Flight and lodging search.

There is no live booking API wired up. Rather than pretend otherwise, this
module defines a provider seam and ships one implementation — `SampleProvider`
— that returns deterministic, clearly-labelled synthetic offers.

Every offer carries `source`. `SampleProvider` sets it to `"sample-data"`, and
the system prompts instruct the agent to surface that label to the traveler.
This keeps the agent's planning behaviour exercisable end-to-end without ever
presenting a fabricated price as a real one.

To go live, implement `AvailabilityProvider` against Amadeus, Duffel,
Skyscanner, or similar, register it in `_PROVIDERS`, and set
`TRAVEL_AGENT_PROVIDER` to its key. A real provider should set `source` to its
own name so the "sample data" caveat drops out of the agent's replies
automatically.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import date, timedelta
from typing import Protocol

from langchain_core.tools import tool

SAMPLE_SOURCE = "sample-data"

_SAMPLE_DISCLAIMER = (
    "These are synthetic sample offers, not live availability. Prices, times, "
    "and seat/room counts are illustrative only and nothing here is bookable. "
    "Tell the traveler this explicitly when you use these figures."
)


class AvailabilityProvider(Protocol):
    """Source of flight and lodging offers."""

    name: str

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
            duration_minutes = 240 + (seed % 300) + stops * 95 + index * 15
            depart_hour = 6 + ((seed // (index + 1)) % 15)

            offers.append(
                {
                    "source": SAMPLE_SOURCE,
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


def get_provider() -> AvailabilityProvider:
    """Return the configured provider, defaulting to sample data."""
    key = os.getenv("TRAVEL_AGENT_PROVIDER", SAMPLE_SOURCE)
    try:
        factory = _PROVIDERS[key]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown TRAVEL_AGENT_PROVIDER={key!r}. Registered providers: {known}"
        ) from None
    return factory()


def _wrap(offers: list[dict], provider: AvailabilityProvider) -> dict:
    """Package offers for the model, warning whenever any are synthetic.

    The warning keys off each offer's own `source`, not the provider name: a
    live provider can still fall back to sample data for part of its surface
    (Duffel covers flights but not stays), and those offers must stay labelled.
    """
    sources = sorted({str(offer.get("source", "unknown")) for offer in offers})
    payload: dict = {
        "provider": provider.name,
        "sources": sources,
        "count": len(offers),
        "offers": offers,
    }
    if SAMPLE_SOURCE in sources:
        payload["warning"] = _SAMPLE_DISCLAIMER
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

    Returns offers sorted cheapest first. Check the `provider` field on the
    response: when it is `sample-data`, the offers are illustrative only and
    must be described that way to the traveler.

    Args:
        origin: Origin airport IATA code, e.g. "SFO".
        destination: Destination airport IATA code, e.g. "NRT".
        depart_date: Outbound date as YYYY-MM-DD.
        return_date: Return date as YYYY-MM-DD. Omit for one-way.
        travelers: Number of travelers on the booking.
    """
    if travelers < 1:
        return {"error": "travelers must be at least 1."}
    provider = get_provider()
    try:
        offers = provider.search_flights(origin, destination, depart_date, return_date, travelers)
    except ValueError as exc:
        return {"error": str(exc)}
    return _wrap(offers, provider)


@tool
def search_stays(
    location: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    max_nightly_rate: float | None = None,
) -> dict:
    """Search places to stay in a location for a date range.

    Returns offers sorted by total cost. Check the `provider` field on the
    response: when it is `sample-data`, the offers are illustrative only and
    must be described that way to the traveler.

    Args:
        location: City or neighbourhood to search, e.g. "Kyoto" or "Shibuya".
        check_in: Arrival date as YYYY-MM-DD.
        check_out: Departure date as YYYY-MM-DD.
        guests: Number of guests.
        max_nightly_rate: Optional ceiling on nightly rate, in USD.
    """
    if guests < 1:
        return {"error": "guests must be at least 1."}
    provider = get_provider()
    try:
        offers = provider.search_stays(location, check_in, check_out, guests, max_nightly_rate)
    except ValueError as exc:
        return {"error": str(exc)}
    return _wrap(offers, provider)


@tool
def date_offset(start_date: str, days: int) -> str:
    """Shift an ISO date by a number of days, for building day-by-day plans.

    Args:
        start_date: The reference date as YYYY-MM-DD.
        days: Days to add; negative values move backwards.
    """
    try:
        return (_parse_date(start_date, "start_date") + timedelta(days=days)).isoformat()
    except ValueError as exc:
        return str(exc)

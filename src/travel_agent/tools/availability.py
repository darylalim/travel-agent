"""Flight and lodging search, across pluggable providers.

Two providers are registered: `sample-data` (synthetic offers, the default)
and `duffel` (live flight and lodging search under a live token; both fall back
to synthetic offers under a test token). Select one with
`TRAVEL_AGENT_PROVIDER`.

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
import re
from collections.abc import Callable
from datetime import date, timedelta
from functools import cache
from typing import ClassVar, Literal, Protocol

from langchain_core.tools import tool

from travel_agent import clock

SAMPLE_SOURCE = "sample-data"
SearchKind = Literal["flights", "stays"]

# Duffel's own spelling, reused verbatim so no provider needs a translation
# table. Internal only: the `@tool` parameter below stays a plain `str`,
# because LangChain builds a pydantic schema from the signature and would
# reject an unrecognised value **before** the function body runs — raising
# where every other bad input here returns `{"error": ...}` for the model to
# read. Normalising in the body is what keeps that contract.
CabinClass = Literal["economy", "premium_economy", "business", "first"]
CABIN_CLASSES: tuple[CabinClass, ...] = ("economy", "premium_economy", "business", "first")
DEFAULT_CABIN: CabinClass = "economy"

# One itinerary whose legs are not all the same class. A sentinel value on the
# existing `cabin` field rather than a second field, because the Trip page's
# `column_order` is a whitelist that silently hides any key it does not name.
MIXED_CABIN = "mixed"

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
        cabin: CabinClass = DEFAULT_CABIN,
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
    value = offer.get(field)
    return float(value) if isinstance(value, (int, float)) else float("inf")


# A live search can return hundreds of offers — one real Duffel test-mode
# query returned 630. Handing them all to the model costs a context window
# and a lot of money, so responses are bounded.
MAX_FLIGHT_OFFERS = 20
MAX_STAY_OFFERS = 20


def _representative(offers: list[dict], limit: int) -> list[dict]:
    """Trim a large offer set to a bounded but still representative selection.

    Cutting purely by price would hide every nonstop whenever the cheapest
    fares are all long multi-stop itineraries — and the scout is asked for the
    tradeoff between price and convenience. So keep the cheapest, the fastest,
    and the fewest-stops options, then return them in price order.
    """
    if len(offers) <= limit:
        return offers

    indices = range(len(offers))
    groups = (
        sorted(indices, key=lambda i: _cost(offers[i], "total_fare"))[: limit // 2],
        sorted(indices, key=lambda i: _cost(offers[i], "duration_minutes"))[: limit // 4],
        sorted(
            indices,
            key=lambda i: (_cost(offers[i], "stops"), _cost(offers[i], "total_fare")),
        )[: limit // 4],
    )

    chosen: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for index in group:
            if index not in seen:
                seen.add(index)
                chosen.append(index)

    # The groups overlap heavily when offers are similar — the cheapest can
    # also be the fastest — so top the selection back up to the budget with
    # the next cheapest rather than returning a needlessly thin list.
    if len(chosen) < limit:
        for index in sorted(indices, key=lambda i: _cost(offers[i], "total_fare")):
            if index not in seen:
                seen.add(index)
                chosen.append(index)
                if len(chosen) == limit:
                    break

    chosen = chosen[:limit]
    return [offers[i] for i in sorted(chosen, key=lambda i: _cost(offers[i], "total_fare"))]


def _select_flights(
    offers: list[dict],
    limit: int = MAX_FLIGHT_OFFERS,
    requested_cabin: str | None = None,
) -> list[dict]:
    """Trim to `limit` offers without losing the cabin that was asked for.

    The trim keeps the cheapest, and a lower cabin is always cheaper. Duffel
    honours `cabin_class` as a preference, so a business search can return a
    mix — and cutting that by price alone would fill the whole allowance with
    economy fares and drop every business offer the traveler actually asked
    for. Nothing would error; the table would simply be the wrong cabin.

    So the requested cabin is selected from first and the remainder tops up
    whatever budget is left. With no cabin requested, or when every offer
    matches, this is exactly `_representative` over the whole set.
    """
    if len(offers) <= limit:
        return offers
    if requested_cabin is None:
        return _representative(offers, limit)

    asked = [offer for offer in offers if offer.get("cabin") == requested_cabin]
    kept = _representative(asked, limit)
    if len(kept) < limit:
        others = [offer for offer in offers if offer.get("cabin") != requested_cabin]
        kept = kept + _representative(others, limit - len(kept))
    return sorted(kept, key=lambda offer: _cost(offer, "total_fare"))


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


def _parse_future_date(value: str, field: str) -> date:
    """Parse a search date and reject one before today.

    Checked here rather than left to each provider: Duffel refuses a past date
    itself, but the sample provider prices one happily, so without this the
    same mistake fails on one provider and returns plausible offers on the
    other.
    """
    day = _parse_date(value, field)
    today = clock.today()
    if day < today:
        raise ValueError(
            f"{field} {day.isoformat()} is in the past (today is {today.isoformat()}). "
            "No provider can search it. If the traveler gave no year, they mean the "
            "next occurrence."
        )
    return day


def _normalize_cabin(value: str) -> CabinClass:
    """Canonicalise a cabin name, or raise naming the ones that work.

    Case and separators are forgiven — "Premium Economy" and "premium-economy"
    both land on `premium_economy` — because the model writes these calls from
    the traveler's own words. Nothing else is: no prefix or fuzzy matching,
    since coercing "biz" to business quietly commits the traveler to a fare
    several times the one they meant, while an error naming the four valid
    values is something the model can read and retry.
    """
    key = re.sub(r"[\s-]+", "_", value.strip().lower())
    if key not in CABIN_CLASSES:
        known = ", ".join(CABIN_CLASSES)
        raise ValueError(f"cabin must be one of {known}, got {value!r}")
    # No cast needed: ty narrows `key` to `CabinClass` off the membership check.
    return key


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

    # Cabin scales the fare but is deliberately **not** part of `_seed`: these
    # are the same four flights whichever cabin is asked for, priced for a
    # different seat, which is both the more realistic model and the reason a
    # cabin search does not reshuffle carriers and departure times underneath
    # the traveler. Indexed directly rather than with `.get`, so a fifth
    # `CabinClass` added without a multiplier fails loudly here.
    _CABIN_FARE_MULTIPLIERS: ClassVar[dict[CabinClass, float]] = {
        "economy": 1.0,
        "premium_economy": 1.6,
        "business": 2.8,
        "first": 4.5,
    }

    def synthetic_note(self, kind: SearchKind) -> str | None:
        return SAMPLE_DISCLAIMER

    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: str,
        return_date: str | None,
        travelers: int,
        cabin: CabinClass = DEFAULT_CABIN,
    ) -> list[dict]:
        depart = _parse_date(depart_date, "depart_date")
        if return_date is not None:
            returning = _parse_date(return_date, "return_date")
            if returning < depart:
                raise ValueError("return_date cannot fall before depart_date.")

        seed = _seed(origin, destination, depart_date, return_date or "one-way")
        base_fare = 180 + (seed % 640)
        cabin_rate = self._CABIN_FARE_MULTIPLIERS[cabin]
        offers = []

        for index in range(4):
            carrier = self._CARRIERS[(seed + index) % len(self._CARRIERS)]
            stops = index % 3 if index else 0
            # Nonstop carries a premium; each stop discounts the fare.
            seat = (base_fare * (1.28 if stops == 0 else 1.0 - 0.09 * stops)) + index * 23
            fare = round(seat * cabin_rate, 2)
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
                    "cabin": cabin,
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


# What survived the trim, and what to vary to see the rest. Both differ by
# kind: flights keep a three-way representative selection (`_select_flights`)
# while stays are a straight cheapest-first slice, and "nearby airports" is not
# advice you can act on when looking for a hotel.
_TRUNCATION_NOTES: dict[SearchKind, tuple[str, str]] = {
    "flights": (
        "the cheapest, the fastest, and those with the fewest stops",
        "different dates, nearby airports, a price ceiling",
    ),
    "stays": (
        "the cheapest",
        "different dates, a different neighbourhood, a price ceiling",
    ),
}


def _cabin_mismatch_note(requested: CabinClass, offers: list[dict]) -> str | None:
    """Caveat for offers that did not come back in the cabin that was asked for.

    Deliberately *not* folded into `warning`. That field means one thing —
    these offers are not real inventory — and a live search never sets it, so
    reusing it for a cabin caveat would put the traveler's "illustrative sample
    prices" banner on a real, bookable fare. The two caveats are independent
    and stay independent.

    An offer whose cabin is unreadable is left out rather than counted as a
    mismatch: absence is not a negative finding, the same reason
    `free_cancellation` is omitted rather than set false when nobody knows.
    """
    found = sorted({cabin for offer in offers if isinstance(cabin := offer.get("cabin"), str)})
    differing = [cabin for cabin in found if cabin != requested]
    if not differing:
        return None
    count = sum(1 for offer in offers if offer.get("cabin") in differing)
    listed = " and ".join(filter(None, [", ".join(differing[:-1]), differing[-1]]))
    mixed = (
        f" {MIXED_CABIN!r} means one itinerary's legs are not all the same class."
        if MIXED_CABIN in differing
        else ""
    )
    return (
        f"Searched {requested}, but {count} of {len(offers)} offers came back as "
        f"{listed} — cabin is a preference, not a filter.{mixed} "
        "Read each offer's own cabin rather than describing them all as the "
        "cabin that was requested."
    )


def _wrap(
    offers: list[dict],
    provider: AvailabilityProvider,
    kind: SearchKind,
    total_found: int | None = None,
) -> dict:
    """Package offers for the model, warning whenever any are synthetic.

    The provider's own note is authoritative and applies even to an empty
    result list; the per-offer `synthetic` flags are a backstop for providers
    that mix real and synthetic results in one response.
    """
    note = provider.synthetic_note(kind)
    found = len(offers) if total_found is None else total_found
    payload: dict = {
        "provider": provider.name,
        "sources": sorted({str(offer.get("source", "unknown")) for offer in offers}),
        "total_found": found,
        "count": len(offers),
        "offers": offers,
    }
    if found > len(offers):
        kept, advice = _TRUNCATION_NOTES[kind]
        payload["truncated"] = (
            f"Showing {len(offers)} of {found} offers — {kept}. "
            f"Narrow the search ({advice}) to surface other options."
        )
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
    cabin: str = DEFAULT_CABIN,
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

    A live search can match hundreds of flights. When `total_found` exceeds
    `count`, a `truncated` note explains what was kept — narrow the search
    rather than assuming the returned set is everything available.

    `requested_cabin` echoes the cabin that was searched for. Each offer's own
    `cabin` is what actually came back, which need not match: cabin is a
    preference rather than a filter, and an offer reads `"mixed"` when the legs
    of one itinerary are not all the same class. Whenever any offer differs, a
    `cabin_note` says so. It is a separate key from `warning` on purpose —
    `warning` only ever means the offers are not real inventory, and a real
    fare in the wrong cabin is a different problem. Quote an offer's own cabin,
    never the requested one.

    Args:
        origin: Origin airport IATA code, e.g. "SFO".
        destination: Destination airport IATA code, e.g. "NRT".
        depart_date: Outbound date as YYYY-MM-DD. A date before today is an
            error; the system prompt states today's date.
        return_date: Return date as YYYY-MM-DD. Omit for one-way.
        travelers: Number of travelers on the booking.
        cabin: One of economy, premium_economy, business, first. Case and
            spaces or hyphens are forgiven; anything else is an error rather
            than a guess. Defaults to economy.
    """
    if travelers < 1:
        return {"error": "travelers must be at least 1."}
    try:
        # Validated here rather than per provider, so the tool behaves the
        # same way whichever provider is configured.
        depart = _parse_future_date(depart_date, "depart_date")
        if return_date is not None and _parse_date(return_date, "return_date") < depart:
            return {"error": "return_date cannot fall before depart_date."}
        wanted = _normalize_cabin(cabin)
        provider = get_provider()
        found = provider.search_flights(
            origin, destination, depart_date, return_date, travelers, wanted
        )
    except ValueError as exc:
        return {"error": str(exc)}

    selected = _select_flights(found, requested_cabin=wanted)
    payload = _wrap(selected, provider, "flights", total_found=len(found))
    payload["requested_cabin"] = wanted
    if note := _cabin_mismatch_note(wanted, selected):
        payload["cabin_note"] = note
    return payload


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
    something bookable. Judge every search on its own result — never by the
    `provider` name, and never by what a flight search in the same
    conversation returned.

    `total_cost` is the price the provider quotes. Some offers also carry
    `due_at_accommodation`, payable on arrival; the two figures may or may not
    overlap depending on the source, so report them side by side and do not
    add them together into a single total.

    `free_cancellation` is **absent** when the policy is unknown. Absent is not
    the same as false — do not tell the traveler a stay is non-refundable
    unless the field is present and false.

    Offers from real inventory carry `expires_at` and `expires_in_seconds`.
    Search again rather than quoting one that has expired.

    Args:
        location: City to search, e.g. "Kyoto". Sample data accepts any
            free-text place; live lodging search covers a fixed list of cities
            and the error names them when it does not recognise one.
        check_in: Arrival date as YYYY-MM-DD. A date before today is an
            error; the system prompt states today's date.
        check_out: Departure date as YYYY-MM-DD.
        guests: Number of guests.
        max_nightly_rate: Optional ceiling on nightly rate, in USD. Applied
            only to offers quoted in USD — an offer priced in another currency
            is returned regardless, so check its `currency` before treating it
            as within budget.
    """
    if guests < 1:
        return {"error": "guests must be at least 1."}
    try:
        if _parse_date(check_out, "check_out") <= _parse_future_date(check_in, "check_in"):
            return {"error": "check_out must be at least one day after check_in."}
        provider = get_provider()
        found = provider.search_stays(location, check_in, check_out, guests, max_nightly_rate)
    except ValueError as exc:
        return {"error": str(exc)}
    # Already sorted by total cost, so the cheapest N is a fair trim.
    return _wrap(found[:MAX_STAY_OFFERS], provider, "stays", total_found=len(found))


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

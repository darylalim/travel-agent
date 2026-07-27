"""Duffel flight search.

Read-only. This module creates offer requests and reads the offers back; it
never creates an order, so no booking is made and no payment is taken. Adding
booking would mean `POST /air/orders`, which should sit behind a human
approval gate rather than in the agent's autonomous loop.

Contract verified against Duffel's docs (API version `v2`):

- `POST https://api.duffel.com/air/offer_requests?return_offers=true`
- Headers: `Authorization: Bearer <token>`, `Duffel-Version: v2`,
  `Accept: application/json`, `Content-Type: application/json`
- Request and response are both wrapped in a top-level `data` key; offers
  come back at `data.offers`
- Errors arrive as `{"errors": [{code, type, title, message, ...}], "meta": {...}}`,
  with 422 for validation failures and 429 for rate limiting

We talk to the REST API over `httpx` rather than the `duffel-api` PyPI
package: that package was last released in 2023, is still classified Alpha,
and would drag in `requests`.

Out of scope for now, deliberately: Duffel Stays (lodging still comes from
`SampleProvider`) and cabin selection (every search is economy, matching the
sample provider).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DUFFEL_SOURCE = "duffel"
API_BASE = "https://api.duffel.com"
API_VERSION = "v2"

# Duffel caps supplier_timeout at 2-60s and defaults to 20s. Our HTTP timeout
# must exceed it, or we hang up before Duffel returns the offers it has.
_SUPPLIER_TIMEOUT_MS = 20_000
_SUPPLIER_TIMEOUT_BOUNDS = (2_000, 60_000)
_HTTP_TIMEOUT_MARGIN_S = 15.0

_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class DuffelError(ValueError):
    """A Duffel request failed. Subclasses ValueError so tools surface it as text."""


def parse_iso_duration(value: str | None) -> int | None:
    """Convert an ISO 8601 duration such as `PT10H30M` to whole minutes."""
    if not value:
        return None
    match = _ISO_DURATION.match(value)
    if not match:
        return None
    parts = {key: int(raw) for key, raw in match.groupdict(default="0").items()}
    total = parts["days"] * 1440 + parts["hours"] * 60 + parts["minutes"] + parts["seconds"] // 60
    return total or None


def _seconds_until(timestamp: str | None) -> int | None:
    """Seconds remaining until an ISO 8601 timestamp, floored at zero."""
    if not timestamp:
        return None
    try:
        expires = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return max(0, int((expires - datetime.now(UTC)).total_seconds()))


def _local_time(timestamp: str | None) -> str | None:
    """Extract HH:MM from a Duffel local departure/arrival timestamp."""
    if not timestamp or "T" not in timestamp:
        return None
    return timestamp.split("T", 1)[1][:5]


def _carrier_names(segments: list[dict[str, Any]]) -> list[str]:
    """Distinct marketing carriers across a slice's segments."""
    names = {
        name
        for seg in segments
        if isinstance(name := (seg.get("marketing_carrier") or {}).get("name"), str)
    }
    return sorted(names)


def _map_slice(slice_: dict[str, Any]) -> dict[str, Any]:
    segments = slice_.get("segments") or []
    durations = [parse_iso_duration(seg.get("duration")) for seg in segments]
    known = [d for d in durations if d is not None]

    return {
        "origin": (slice_.get("origin") or {}).get("iata_code"),
        "destination": (slice_.get("destination") or {}).get("iata_code"),
        "departing_at": segments[0].get("departing_at") if segments else None,
        "arriving_at": segments[-1].get("arriving_at") if segments else None,
        # Sum of segment durations; excludes layover time, which Duffel does
        # not break out on the slice. Treat as flight time, not door to door.
        "flight_minutes": sum(known) if len(known) == len(segments) and known else None,
        "stops": max(len(segments) - 1, 0),
        "carriers": _carrier_names(segments),
    }


def map_offer(offer: dict[str, Any], travelers: int) -> dict[str, Any]:
    """Normalise a Duffel offer into the shape the agent's tools return.

    Kept pure and separate from transport so it can be tested against a
    recorded payload without touching the network.
    """
    slices = [_map_slice(s) for s in (offer.get("slices") or [])]
    outbound = slices[0] if slices else {}

    total = float(offer.get("total_amount") or 0.0)
    per_traveler = round(total / travelers, 2) if travelers else total

    return {
        "source": DUFFEL_SOURCE,
        "offer_id": offer.get("id"),
        "carrier": (offer.get("owner") or {}).get("name"),
        "origin": outbound.get("origin"),
        "destination": outbound.get("destination"),
        "depart_time_local": _local_time(outbound.get("departing_at")),
        "stops": outbound.get("stops"),
        "duration_minutes": outbound.get("flight_minutes"),
        "cabin": offer.get("cabin_class"),
        "fare_per_traveler": per_traveler,
        "total_fare": round(total, 2),
        "travelers": travelers,
        "currency": offer.get("total_currency"),
        # Duffel offers go stale in minutes. The agent must re-search rather
        # than quote an expired offer, so surface the deadline explicitly.
        "expires_at": offer.get("expires_at"),
        "expires_in_seconds": _seconds_until(offer.get("expires_at")),
        "slices": slices,
    }


def _describe_error(response: httpx.Response) -> str:
    """Turn a Duffel error body into one readable line."""
    request_id = response.headers.get("x-request-id", "unknown")
    try:
        errors = response.json().get("errors") or []
    except ValueError:
        errors = []

    if errors:
        first = errors[0]
        detail = first.get("message") or first.get("title") or "no detail given"
        pointer = (first.get("source") or {}).get("pointer")
        located = f" at {pointer}" if pointer else ""
        code = first.get("code", "unknown")
        return f"Duffel {response.status_code} [{code}]{located}: {detail} (request {request_id})"

    return f"Duffel {response.status_code}: {response.text[:200]} (request {request_id})"


class DuffelProvider:
    """Live flight offers from Duffel. Lodging falls through to sample data."""

    name = DUFFEL_SOURCE

    def __init__(
        self,
        token: str | None = None,
        supplier_timeout_ms: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._transport = transport  # test seam; None uses the real network
        self._token = token or os.getenv("DUFFEL_API_TOKEN", "")
        if not self._token:
            raise DuffelError(
                "DUFFEL_API_TOKEN is not set. Create a test token in the Duffel "
                "dashboard (it starts with 'duffel_test_') and add it to .env, or "
                "set TRAVEL_AGENT_PROVIDER=sample-data to use sample offers."
            )

        low, high = _SUPPLIER_TIMEOUT_BOUNDS
        requested = supplier_timeout_ms or int(
            os.getenv("DUFFEL_SUPPLIER_TIMEOUT_MS", _SUPPLIER_TIMEOUT_MS)
        )
        self._supplier_timeout_ms = min(max(requested, low), high)
        self._http_timeout_s = self._supplier_timeout_ms / 1000 + _HTTP_TIMEOUT_MARGIN_S

        if self._token.startswith("duffel_live_"):
            # Search is read-only, so this costs nothing — but the operator
            # should know they are hitting real inventory.
            logger.warning(
                "Using a live Duffel token. Searches hit real inventory; this "
                "integration is read-only and never creates an order."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Duffel-Version": API_VERSION,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        }

    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: str,
        return_date: str | None,
        travelers: int,
    ) -> list[dict]:
        slices: list[dict[str, str]] = [
            {
                "origin": origin.upper(),
                "destination": destination.upper(),
                "departure_date": depart_date,
            }
        ]
        if return_date:
            slices.append(
                {
                    "origin": destination.upper(),
                    "destination": origin.upper(),
                    "departure_date": return_date,
                }
            )

        payload = {
            "data": {
                "slices": slices,
                "passengers": [{"type": "adult"} for _ in range(travelers)],
                "cabin_class": "economy",
            }
        }

        try:
            with httpx.Client(timeout=self._http_timeout_s, transport=self._transport) as client:
                response = client.post(
                    f"{API_BASE}/air/offer_requests",
                    params={
                        "return_offers": "true",
                        "supplier_timeout": self._supplier_timeout_ms,
                    },
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise DuffelError(
                f"Duffel search timed out after {self._http_timeout_s:.0f}s. "
                "Try a narrower search or raise DUFFEL_SUPPLIER_TIMEOUT_MS."
            ) from exc
        except httpx.HTTPError as exc:
            raise DuffelError(f"Could not reach Duffel: {exc}") from exc

        if response.status_code == 429:
            retry = response.headers.get("ratelimit-reset", "shortly")
            raise DuffelError(f"Duffel rate limit hit; resets {retry}.")
        if response.status_code >= 400:
            raise DuffelError(_describe_error(response))

        offers = (response.json().get("data") or {}).get("offers") or []
        mapped = [map_offer(offer, travelers) for offer in offers]
        mapped.sort(key=lambda offer: offer["total_fare"])
        return mapped

    def search_stays(
        self,
        location: str,
        check_in: str,
        check_out: str,
        guests: int,
        max_nightly_rate: float | None,
    ) -> list[dict]:
        """Lodging is not wired to Duffel Stays yet — returns sample data.

        Offers keep `source: "sample-data"`, so the caller still attaches the
        not-real-data warning and the agent still passes that caveat on.
        """
        from travel_agent.tools.availability import SampleProvider

        return SampleProvider().search_stays(
            location, check_in, check_out, guests, max_nightly_rate
        )

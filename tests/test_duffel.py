"""Tests for the Duffel provider.

No network: request construction is verified through `httpx.MockTransport`,
and the offer mapping runs against a payload shaped like a real Duffel
response.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from travel_agent.tools.availability import SAMPLE_SOURCE, search_stays
from travel_agent.tools.duffel import (
    DUFFEL_SOURCE,
    DuffelError,
    DuffelProvider,
    map_offer,
    parse_iso_duration,
)

TEST_TOKEN = "duffel_test_abc123"


def _offer(expires_at: str, total: str = "912.40") -> dict:
    """One return offer, shaped like Duffel's `data.offers[]` entries."""
    return {
        "id": "off_0000AaBbCc",
        "expires_at": expires_at,
        "total_amount": total,
        "total_currency": "USD",
        "base_amount": "780.00",
        "tax_amount": "132.40",
        "cabin_class": "economy",
        "owner": {"name": "Duffel Airways", "iata_code": "ZZ"},
        "slices": [
            {
                "origin": {"iata_code": "SFO"},
                "destination": {"iata_code": "NRT"},
                "segments": [
                    {
                        "departing_at": "2026-09-12T08:25:00",
                        "arriving_at": "2026-09-12T11:40:00",
                        "duration": "PT3H15M",
                        "marketing_carrier": {"name": "Duffel Airways"},
                        "operating_carrier": {"name": "Duffel Airways"},
                    },
                    {
                        "departing_at": "2026-09-12T13:10:00",
                        "arriving_at": "2026-09-13T16:55:00",
                        "duration": "PT10H45M",
                        "marketing_carrier": {"name": "Partner Air"},
                        "operating_carrier": {"name": "Partner Air"},
                    },
                ],
            },
            {
                "origin": {"iata_code": "NRT"},
                "destination": {"iata_code": "SFO"},
                "segments": [
                    {
                        "departing_at": "2026-09-20T17:00:00",
                        "arriving_at": "2026-09-20T09:30:00",
                        "duration": "PT9H30M",
                        "marketing_carrier": {"name": "Duffel Airways"},
                        "operating_carrier": {"name": "Duffel Airways"},
                    }
                ],
            },
        ],
    }


def _future(minutes: int = 25) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


# --- duration parsing -------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("PT3H15M", 195),
        ("PT45M", 45),
        ("P1DT2H", 1560),
        ("PT30S", None),  # under a minute rounds away to nothing
        ("", None),
        (None, None),
        ("not-a-duration", None),
    ],
)
def test_parse_iso_duration(value, expected):
    assert parse_iso_duration(value) == expected


# --- offer mapping ----------------------------------------------------------


def test_map_offer_normalises_to_the_shared_shape():
    mapped = map_offer(_offer(_future()), travelers=2)

    assert mapped["source"] == DUFFEL_SOURCE
    assert mapped["offer_id"] == "off_0000AaBbCc"
    assert mapped["carrier"] == "Duffel Airways"
    assert mapped["origin"] == "SFO"
    assert mapped["destination"] == "NRT"
    assert mapped["currency"] == "USD"
    assert mapped["cabin"] == "economy"
    assert mapped["total_fare"] == pytest.approx(912.40)
    assert mapped["fare_per_traveler"] == pytest.approx(456.20)
    assert mapped["travelers"] == 2
    # Outbound has two segments, so one stop; duration is the sum of both legs.
    assert mapped["stops"] == 1
    assert mapped["duration_minutes"] == 195 + 645
    assert mapped["depart_time_local"] == "08:25"


def test_map_offer_keeps_both_slices_of_a_return_trip():
    mapped = map_offer(_offer(_future()), travelers=1)

    assert len(mapped["slices"]) == 2
    outbound, inbound = mapped["slices"]
    assert (outbound["origin"], outbound["destination"]) == ("SFO", "NRT")
    assert (inbound["origin"], inbound["destination"]) == ("NRT", "SFO")
    assert outbound["carriers"] == ["Duffel Airways", "Partner Air"]
    assert inbound["stops"] == 0


def test_map_offer_surfaces_expiry_so_stale_offers_are_not_quoted():
    live = map_offer(_offer(_future(minutes=30)), travelers=1)
    assert 0 < live["expires_in_seconds"] <= 30 * 60

    expired = map_offer(_offer((datetime.now(UTC) - timedelta(minutes=5)).isoformat()), travelers=1)
    assert expired["expires_in_seconds"] == 0


def test_map_offer_tolerates_a_sparse_payload():
    mapped = map_offer({"id": "off_x", "total_amount": "100.00"}, travelers=1)

    assert mapped["total_fare"] == pytest.approx(100.0)
    assert mapped["slices"] == []
    assert mapped["carrier"] is None
    assert mapped["duration_minutes"] is None


# --- request construction ---------------------------------------------------


def test_search_flights_sends_the_documented_request():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"offers": [_offer(_future())]}})

    provider = DuffelProvider(token=TEST_TOKEN, transport=httpx.MockTransport(handler))
    offers = provider.search_flights("sfo", "nrt", "2026-09-12", "2026-09-20", travelers=2)

    assert seen["url"].startswith("https://api.duffel.com/air/offer_requests")
    assert "return_offers=true" in seen["url"]
    assert "supplier_timeout=20000" in seen["url"]

    assert seen["headers"]["authorization"] == f"Bearer {TEST_TOKEN}"
    assert seen["headers"]["duffel-version"] == "v2"
    assert seen["headers"]["accept"] == "application/json"

    data = seen["body"]["data"]
    assert data["cabin_class"] == "economy"
    assert data["passengers"] == [{"type": "adult"}, {"type": "adult"}]
    # A return trip is two slices, the second reversed.
    assert data["slices"] == [
        {"origin": "SFO", "destination": "NRT", "departure_date": "2026-09-12"},
        {"origin": "NRT", "destination": "SFO", "departure_date": "2026-09-20"},
    ]
    assert offers[0]["source"] == DUFFEL_SOURCE


def test_one_way_search_sends_a_single_slice():
    def handler(request: httpx.Request) -> httpx.Response:
        assert len(json.loads(request.content)["data"]["slices"]) == 1
        return httpx.Response(200, json={"data": {"offers": []}})

    provider = DuffelProvider(token=TEST_TOKEN, transport=httpx.MockTransport(handler))
    assert provider.search_flights("SFO", "NRT", "2026-09-12", None, travelers=1) == []


def test_offers_come_back_cheapest_first():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "offers": [
                        _offer(_future(), total="900.00"),
                        _offer(_future(), total="410.50"),
                        _offer(_future(), total="655.25"),
                    ]
                }
            },
        )

    provider = DuffelProvider(token=TEST_TOKEN, transport=httpx.MockTransport(handler))
    fares = [
        offer["total_fare"]
        for offer in provider.search_flights("SFO", "NRT", "2026-09-12", None, travelers=1)
    ]
    assert fares == sorted(fares) == [410.50, 655.25, 900.00]


def test_supplier_timeout_is_clamped_to_duffels_range():
    assert DuffelProvider(token=TEST_TOKEN, supplier_timeout_ms=1)._supplier_timeout_ms == 2_000
    assert (
        DuffelProvider(token=TEST_TOKEN, supplier_timeout_ms=999_999)._supplier_timeout_ms == 60_000
    )


# --- failure modes ----------------------------------------------------------


def test_missing_token_explains_how_to_fix_it(monkeypatch):
    monkeypatch.delenv("DUFFEL_API_TOKEN", raising=False)
    with pytest.raises(DuffelError, match="DUFFEL_API_TOKEN"):
        DuffelProvider()


def test_validation_error_is_reported_with_field_and_request_id():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            headers={"x-request-id": "req_123"},
            json={
                "errors": [
                    {
                        "code": "validation_required",
                        "title": "Missing field",
                        "message": "Origin is required",
                        "source": {"field": "origin", "pointer": "/slices/0/origin"},
                    }
                ]
            },
        )

    provider = DuffelProvider(token=TEST_TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(DuffelError) as exc:
        provider.search_flights("SFO", "NRT", "2026-09-12", None, travelers=1)

    message = str(exc.value)
    assert "422" in message
    assert "validation_required" in message
    assert "/slices/0/origin" in message
    assert "req_123" in message


def test_rate_limit_is_reported_distinctly():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"ratelimit-reset": "Mon, 27 Jul 2026 03:00:00 GMT"})

    provider = DuffelProvider(token=TEST_TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(DuffelError, match="rate limit"):
        provider.search_flights("SFO", "NRT", "2026-09-12", None, travelers=1)


def test_timeout_suggests_the_knob_to_turn():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    provider = DuffelProvider(token=TEST_TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(DuffelError, match="DUFFEL_SUPPLIER_TIMEOUT_MS"):
        provider.search_flights("SFO", "NRT", "2026-09-12", None, travelers=1)


# --- the mixed-source case --------------------------------------------------


def test_lodging_still_returns_labelled_sample_data_under_duffel(monkeypatch):
    """Duffel covers flights only, so stays must keep the sample-data caveat."""
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    monkeypatch.setenv("DUFFEL_API_TOKEN", TEST_TOKEN)

    result = search_stays.invoke(
        {"location": "Kyoto", "check_in": "2026-09-14", "check_out": "2026-09-18", "guests": 2}
    )

    assert result["provider"] == DUFFEL_SOURCE
    assert result["sources"] == [SAMPLE_SOURCE]
    assert all(offer["source"] == SAMPLE_SOURCE for offer in result["offers"])
    # The warning follows the offers, not the provider name.
    assert "warning" in result

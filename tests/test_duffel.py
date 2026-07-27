"""Tests for the Duffel provider.

No network: request construction is verified through `httpx.MockTransport`,
and the offer mapping runs against a payload shaped to Duffel's documented v2
offer schema — `live_mode` at the top level, `duration` on each slice, and
`cabin_class` nested at `slices[].segments[].passengers[].cabin_class`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from travel_agent.tools.availability import SAMPLE_SOURCE, search_flights, search_stays
from travel_agent.tools.duffel import (
    DUFFEL_SOURCE,
    DuffelError,
    DuffelProvider,
    map_offer,
    parse_iso_duration,
)

TEST_TOKEN = "duffel_test_abc123"
LIVE_TOKEN = "duffel_live_abc123"


def _segment(depart: str, arrive: str, duration: str, carrier: str) -> dict:
    return {
        "departing_at": depart,
        "arriving_at": arrive,
        "duration": duration,
        "marketing_carrier": {"name": carrier},
        "operating_carrier": {"name": carrier},
        # Cabin lives here, not on the offer.
        "passengers": [{"cabin_class": "economy", "cabin_class_marketing_name": "Economy Basic"}],
    }


def _offer(expires_at: str, total: str | None = "912.40", *, live_mode: bool = False) -> dict:
    """One return offer, shaped like Duffel's `data.offers[]` entries."""
    offer = {
        "id": "off_0000AaBbCc",
        "live_mode": live_mode,
        "expires_at": expires_at,
        "total_currency": "USD",
        "base_amount": "780.00",
        "tax_amount": "132.40",
        "owner": {"name": "Duffel Airways", "iata_code": "ZZ"},
        "slices": [
            {
                # Whole-slice duration: 14h30m, longer than the 13h45m of
                # flying time below because it includes the layover.
                "duration": "PT14H30M",
                "origin": {"iata_code": "SFO"},
                "destination": {"iata_code": "NRT"},
                "segments": [
                    _segment(
                        "2026-09-12T08:25:00", "2026-09-12T11:40:00", "PT3H15M", "Duffel Airways"
                    ),
                    _segment(
                        "2026-09-12T13:10:00", "2026-09-13T16:55:00", "PT10H45M", "Partner Air"
                    ),
                ],
            },
            {
                "duration": "PT9H30M",
                "origin": {"iata_code": "NRT"},
                "destination": {"iata_code": "SFO"},
                "segments": [
                    _segment(
                        "2026-09-20T17:00:00", "2026-09-20T09:30:00", "PT9H30M", "Duffel Airways"
                    )
                ],
            },
        ],
    }
    if total is not None:
        offer["total_amount"] = total
    return offer


def _future(minutes: int = 25) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


def _responds(payload: dict, status: int = 200, headers: dict | None = None) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda _: httpx.Response(status, json=payload, headers=headers or {})
    )


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
    mapped = map_offer(_offer(_future(), live_mode=True), travelers=2)

    assert mapped["source"] == DUFFEL_SOURCE
    assert mapped["offer_id"] == "off_0000AaBbCc"
    assert mapped["carrier"] == "Duffel Airways"
    assert mapped["origin"] == "SFO"
    assert mapped["destination"] == "NRT"
    assert mapped["currency"] == "USD"
    assert mapped["total_fare"] == pytest.approx(912.40)
    assert mapped["fare_per_traveler"] == pytest.approx(456.20)
    assert mapped["travelers"] == 2
    assert mapped["stops"] == 1
    assert mapped["depart_time_local"] == "08:25"


def test_cabin_is_read_from_the_nested_passenger_field():
    """Duffel has no top-level cabin_class; it sits under segments[].passengers[]."""
    mapped = map_offer(_offer(_future()), travelers=1)
    assert mapped["cabin"] == "economy"

    # An offer with no passenger cabin info reports None rather than guessing.
    bare = map_offer({"id": "off_x", "total_amount": "10.00", "slices": []}, travelers=1)
    assert bare["cabin"] is None


def test_duration_uses_the_slice_total_so_layovers_are_counted():
    """Summing segments would understate a multi-stop trip by the layover."""
    mapped = map_offer(_offer(_future()), travelers=1)

    assert mapped["duration_minutes"] == 870  # PT14H30M
    assert mapped["duration_minutes"] > 195 + 645, "must exceed pure flying time"


def test_duration_falls_back_to_segment_sum_when_the_slice_omits_it():
    offer = _offer(_future())
    del offer["slices"][0]["duration"]

    assert map_offer(offer, travelers=1)["duration_minutes"] == 195 + 645


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


def test_missing_price_maps_to_none_rather_than_zero():
    """A $0 fare would sort to the front and be recommended as the cheapest."""
    mapped = map_offer(_offer(_future(), total=None), travelers=2)

    assert mapped["total_fare"] is None
    assert mapped["fare_per_traveler"] is None


def test_map_offer_tolerates_a_sparse_payload():
    mapped = map_offer({"id": "off_x", "total_amount": "100.00"}, travelers=1)

    assert mapped["total_fare"] == pytest.approx(100.0)
    assert mapped["slices"] == []
    assert mapped["carrier"] is None
    assert mapped["duration_minutes"] is None


# --- test mode vs live mode -------------------------------------------------


def test_test_mode_offers_are_marked_synthetic():
    """Test-token inventory is fictional and must never read as real pricing."""
    mapped = map_offer(_offer(_future(), live_mode=False), travelers=1)

    assert mapped["synthetic"] is True
    assert mapped["live_mode"] is False


def test_live_mode_offers_are_not_marked_synthetic():
    mapped = map_offer(_offer(_future(), live_mode=True), travelers=1)

    assert mapped["synthetic"] is False
    assert mapped["live_mode"] is True


def test_test_token_provider_declares_flights_synthetic():
    provider = DuffelProvider(token=TEST_TOKEN)
    assert provider.test_mode is True
    assert "test mode" in (provider.synthetic_note("flights") or "")


def test_live_token_provider_declares_flights_real():
    provider = DuffelProvider(token=LIVE_TOKEN)
    assert provider.test_mode is False
    assert provider.synthetic_note("flights") is None
    # Lodging is still sample data even on a live token.
    assert provider.synthetic_note("stays") is not None


def test_search_with_a_test_token_warns_through_the_tool(monkeypatch):
    """End to end: a test token must produce a warning on the tool response."""
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    monkeypatch.setenv("DUFFEL_API_TOKEN", TEST_TOKEN)

    from travel_agent.tools import availability

    monkeypatch.setitem(
        availability._PROVIDERS,
        "duffel",
        lambda: DuffelProvider(
            token=TEST_TOKEN,
            transport=_responds({"data": {"offers": [_offer(_future(), live_mode=False)]}}),
        ),
    )

    result = search_flights.invoke(
        {"origin": "SFO", "destination": "NRT", "depart_date": "2026-09-12"}
    )

    assert result["count"] == 1
    assert result["offers"][0]["synthetic"] is True
    assert "warning" in result, "test-mode inventory must carry a not-real-data warning"
    assert "not real" in result["warning"]


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
    provider = DuffelProvider(
        token=TEST_TOKEN,
        transport=_responds(
            {
                "data": {
                    "offers": [
                        _offer(_future(), total="900.00"),
                        _offer(_future(), total="410.50"),
                        _offer(_future(), total="655.25"),
                    ]
                }
            }
        ),
    )
    fares = [
        offer["total_fare"]
        for offer in provider.search_flights("SFO", "NRT", "2026-09-12", None, travelers=1)
    ]
    assert fares == sorted(fares) == [410.50, 655.25, 900.00]


def test_unpriced_offers_are_dropped_not_sorted_to_the_front():
    """An offer with no total must never surface as the cheapest flight."""
    provider = DuffelProvider(
        token=TEST_TOKEN,
        transport=_responds(
            {
                "data": {
                    "offers": [
                        _offer(_future(), total=None),
                        _offer(_future(), total="410.50"),
                    ]
                }
            }
        ),
    )
    offers = provider.search_flights("SFO", "NRT", "2026-09-12", None, travelers=1)

    assert len(offers) == 1
    assert offers[0]["total_fare"] == pytest.approx(410.50)


def test_supplier_timeout_is_clamped_to_duffels_range():
    assert DuffelProvider(token=TEST_TOKEN, supplier_timeout_ms=1)._supplier_timeout_ms == 2_000
    assert (
        DuffelProvider(token=TEST_TOKEN, supplier_timeout_ms=999_999)._supplier_timeout_ms == 60_000
    )


def test_explicit_zero_timeout_is_not_mistaken_for_unset(monkeypatch):
    monkeypatch.setenv("DUFFEL_SUPPLIER_TIMEOUT_MS", "45000")
    # 0 is falsy but explicit: it must clamp to the floor, not read the env.
    assert DuffelProvider(token=TEST_TOKEN, supplier_timeout_ms=0)._supplier_timeout_ms == 2_000


@pytest.mark.parametrize("raw", ["", "   ", "not-a-number"])
def test_malformed_timeout_env_falls_back_instead_of_crashing(monkeypatch, raw):
    monkeypatch.setenv("DUFFEL_SUPPLIER_TIMEOUT_MS", raw)
    assert DuffelProvider(token=TEST_TOKEN)._supplier_timeout_ms == 20_000


# --- failure modes ----------------------------------------------------------


def test_missing_token_explains_how_to_fix_it(monkeypatch):
    monkeypatch.delenv("DUFFEL_API_TOKEN", raising=False)
    with pytest.raises(DuffelError, match="DUFFEL_API_TOKEN"):
        DuffelProvider()


def test_validation_error_is_reported_with_field_and_request_id():
    provider = DuffelProvider(
        token=TEST_TOKEN,
        transport=_responds(
            {
                "errors": [
                    {
                        "code": "validation_required",
                        "title": "Missing field",
                        "message": "Origin is required",
                        "source": {"field": "origin", "pointer": "/slices/0/origin"},
                    }
                ]
            },
            status=422,
            headers={"x-request-id": "req_123"},
        ),
    )
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


def test_bad_dates_are_rejected_before_any_network_call(monkeypatch):
    """Date validation belongs to the tool, so it applies to every provider."""
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    monkeypatch.setenv("DUFFEL_API_TOKEN", TEST_TOKEN)

    def explode(_: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the network with an invalid date")

    from travel_agent.tools import availability

    monkeypatch.setitem(
        availability._PROVIDERS,
        "duffel",
        lambda: DuffelProvider(token=TEST_TOKEN, transport=httpx.MockTransport(explode)),
    )

    result = search_flights.invoke(
        {"origin": "SFO", "destination": "NRT", "depart_date": "12/09/2026"}
    )
    assert "error" in result
    assert "ISO date" in result["error"]


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
    assert all(offer["synthetic"] is True for offer in result["offers"])
    # The warning follows the offers, not the provider name.
    assert "warning" in result

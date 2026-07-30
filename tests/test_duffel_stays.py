"""Tests for the Duffel Stays provider.

No network: request construction goes through `httpx.MockTransport`, and the
mapping runs against payloads shaped to Duffel's documented v2 Stays schema —
results at `data.results[]` rather than `data.offers`, price at the top of each
result, the property nested under `accommodation`, and no `live_mode` field
anywhere.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from travel_agent.tools.availability import SAMPLE_SOURCE, search_stays
from travel_agent.tools.duffel import DUFFEL_SOURCE, DuffelError, DuffelProvider
from travel_agent.tools.duffel_stays import (
    CITY_COORDINATES,
    fetch_stays,
    map_stay_result,
    resolve_location,
)

TEST_TOKEN = "duffel_test_abc123"
LIVE_TOKEN = "duffel_live_abc123"

CHECK_IN = "2026-09-14"
CHECK_OUT = "2026-09-18"  # four nights


def _rate(total: str | None = "799.00", timeline: list | None = None) -> dict:
    """One entry in `accommodation.rooms[].rates[]`."""
    rate: dict = {"id": "rat_0000AaBbCc", "name": "Best Available Rate"}
    if total is not None:
        rate["total_amount"] = total
        rate["total_currency"] = "USD"
    if timeline is not None:
        rate["cancellation_timeline"] = timeline
    return rate


def _result(
    total: str | None = "799.00",
    *,
    name: str = "Duffel Test Hotel",
    review_score: float | str | None = 8.8,
    rating: int | None = 3,
    due_at_accommodation: str | None = "39.95",
    currency: str = "USD",
    city_name: str | None = "London",
    rates: list[dict] | None = None,
    expires_at: str | None = None,
) -> dict:
    """One entry in `data.results[]`, shaped to Duffel's documented schema."""
    accommodation: dict = {"id": "acc_0000AWr2Vs", "name": name}
    if city_name is not None:
        accommodation["location"] = {"address": {"city_name": city_name}}
    if rating is not None:
        accommodation["rating"] = rating
    if review_score is not None:
        accommodation["review_score"] = review_score
    if rates is not None:
        accommodation["rooms"] = [{"name": "Double Suite", "rates": rates}]

    result: dict = {
        "id": "srr_0000ASVBuJ",
        "accommodation": accommodation,
        "cheapest_rate_currency": currency,
    }
    if total is not None:
        result["cheapest_rate_total_amount"] = total
    if due_at_accommodation is not None:
        result["cheapest_rate_due_at_accommodation_amount"] = due_at_accommodation
    if expires_at is not None:
        result["expires_at"] = expires_at
    return result


def _future(minutes: int = 25) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


def _responds(payload: dict, status: int = 200, headers: dict | None = None) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda _: httpx.Response(status, json=payload, headers=headers or {})
    )


def _explodes() -> httpx.MockTransport:
    """A transport that fails the test if anything reaches the network."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    return httpx.MockTransport(handler)


def _map(result: dict, location: str = "London", guests: int = 2) -> dict:
    return map_stay_result(result, location, CHECK_IN, CHECK_OUT, guests)


# --- location resolution ----------------------------------------------------


def test_known_cities_resolve_to_coordinates():
    latitude, longitude = resolve_location("Kyoto")

    assert (latitude, longitude) == CITY_COORDINATES["kyoto"]
    assert 34 < latitude < 36


@pytest.mark.parametrize("written", ["kyoto", "KYOTO", "  Kyoto  ", "KyOtO"])
def test_city_lookup_ignores_case_and_surrounding_space(written):
    assert resolve_location(written) == CITY_COORDINATES["kyoto"]


def test_unknown_city_names_what_is_supported_instead_of_guessing():
    """Refusing beats a nearest match: results labelled with a city nobody asked
    for would be far harder to notice than an error."""
    with pytest.raises(DuffelError) as excinfo:
        resolve_location("Kobe")

    message = str(excinfo.value)
    assert "Kobe" in message
    assert "Kyoto" in message, "the error should name cities that do work"
    assert "sample-data" in message


def test_unknown_city_reaches_the_tool_as_an_error_not_an_exception(monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    monkeypatch.setenv("DUFFEL_API_TOKEN", LIVE_TOKEN)

    from travel_agent.tools import availability

    monkeypatch.setitem(
        availability._PROVIDERS,
        "duffel",
        lambda: DuffelProvider(token=LIVE_TOKEN, transport=_explodes()),
    )

    result = search_stays.invoke(
        {"location": "Atlantis", "check_in": CHECK_IN, "check_out": CHECK_OUT, "guests": 2}
    )

    assert "error" in result
    assert "Atlantis" in result["error"]


# --- offer mapping ----------------------------------------------------------


def test_map_stay_result_normalises_to_the_shared_shape():
    mapped = _map(_result(expires_at=_future()))

    assert mapped["source"] == DUFFEL_SOURCE
    assert mapped["synthetic"] is False
    assert mapped["name"] == "Duffel Test Hotel"
    assert mapped["location"] == "London"
    assert mapped["check_in"] == CHECK_IN
    assert mapped["check_out"] == CHECK_OUT
    assert mapped["nights"] == 4
    assert mapped["guests"] == 2
    assert mapped["total_cost"] == 799.00
    assert mapped["nightly_rate"] == 199.75
    assert mapped["currency"] == "USD"
    assert mapped["expires_in_seconds"] > 0


def test_a_stay_never_carries_an_origin_key():
    """`ui.py`'s search_label reads `origin` to tell a flight from a stay, so an
    origin here would relabel a lodging search as a flight."""
    assert "origin" not in _map(_result())


def test_a_stay_never_carries_a_guessed_property_kind():
    """Duffel's result has no property-type field; inventing "hotel" would put a
    made-up classification in a column the traveler reads as fact."""
    assert "kind" not in _map(_result())


def test_the_two_prices_are_reported_separately_and_never_summed():
    """Duffel's docs disagree about whether due-at-accommodation sits inside the
    total or on top of it, so combining them would state a number nobody can
    source — in either direction."""
    mapped = _map(_result(total="799.00", due_at_accommodation="39.95"))

    assert mapped["total_cost"] == 799.00
    assert mapped["due_at_accommodation"] == 39.95
    assert "amount_due_now" not in mapped
    assert mapped["total_cost"] != 799.00 + 39.95


def test_a_stay_with_no_arrival_fee_reported_carries_none():
    assert _map(_result(due_at_accommodation=None))["due_at_accommodation"] is None


def test_missing_price_maps_to_none_rather_than_zero():
    """A zero total sorts to the front and gets recommended as the cheapest."""
    mapped = _map(_result(total=None))

    assert mapped["total_cost"] is None
    assert mapped["nightly_rate"] is None


def test_unparseable_price_maps_to_none():
    assert _map(_result(total="on request"))["total_cost"] is None


def test_the_city_falls_back_to_the_query_when_the_address_is_missing():
    assert _map(_result(city_name=None), location="Kyoto")["location"] == "Kyoto"


# --- guest rating -----------------------------------------------------------


def test_guest_rating_reads_review_score_not_the_star_rating():
    """`rating` is 1-5 stars, `review_score` is 0-10, and the Trip page renders
    this field on a 0-10 progress bar and axis. Backfilling stars would show a
    four-star property as 4/10."""
    mapped = _map(_result(rating=4, review_score=8.7))

    assert mapped["guest_rating"] == 8.7


def test_an_unrated_property_omits_the_rating_rather_than_scoring_it_zero():
    mapped = _map(_result(review_score=None, rating=4))

    assert "guest_rating" not in mapped, "a 4-star property is not a 0/10 property"


def test_a_stringified_review_score_is_still_read():
    assert _map(_result(review_score="8.8"))["guest_rating"] == 8.8


# --- cancellation policy ----------------------------------------------------


def test_an_empty_cancellation_timeline_is_a_real_non_refundable_finding():
    """Duffel documents the empty list as non-refundable — that is a finding,
    not missing information."""
    mapped = _map(_result(rates=[_rate(timeline=[])]))

    assert mapped["free_cancellation"] is False


def test_a_full_refund_window_is_free_cancellation():
    mapped = _map(
        _result(
            rates=[_rate("799.00", timeline=[{"before": _future(), "refund_amount": "799.00"}])]
        )
    )

    assert mapped["free_cancellation"] is True


def test_a_partial_refund_is_not_free_cancellation():
    """Cancellable for a fee is not free, and a traveler picking on flexibility
    would read True as exactly that."""
    mapped = _map(
        _result(
            rates=[_rate("799.00", timeline=[{"before": _future(), "refund_amount": "400.00"}])]
        )
    )

    assert mapped["free_cancellation"] is False


def test_an_absent_timeline_omits_the_field_rather_than_claiming_no():
    """Absent is not a negative finding — `cancel_band` renders a missing key as
    "Not stated", which is a different claim from "no free cancellation"."""
    mapped = _map(_result(rates=[_rate(timeline=None)]))

    assert "free_cancellation" not in mapped


def test_a_result_with_no_rates_at_all_omits_the_field():
    assert "free_cancellation" not in _map(_result(rates=None))


def test_the_policy_reported_belongs_to_the_cheapest_rate():
    """Duffel does not say which nested rate produced the headline price, so the
    policy has to come from the one actually being shown."""
    mapped = _map(
        _result(
            rates=[
                _rate("950.00", timeline=[{"before": _future(), "refund_amount": "950.00"}]),
                _rate("799.00", timeline=[]),
            ]
        )
    )

    assert mapped["free_cancellation"] is False, "the 799 rate is the one priced"


# --- request construction ---------------------------------------------------


def test_stays_search_sends_the_documented_request():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"results": [_result()]}})

    provider = DuffelProvider(token=LIVE_TOKEN, transport=httpx.MockTransport(handler))
    offers = provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=2, max_nightly_rate=None)

    assert seen["url"] == "https://api.duffel.com/stays/search"
    assert seen["headers"]["authorization"] == f"Bearer {LIVE_TOKEN}"
    assert seen["headers"]["duffel-version"] == "v2"

    data = seen["body"]["data"]
    assert data["rooms"] == 1
    assert data["check_in_date"] == CHECK_IN
    assert data["check_out_date"] == CHECK_OUT
    assert data["guests"] == [{"type": "adult"}, {"type": "adult"}]
    assert data["location"]["geographic_coordinates"] == {
        "latitude": CITY_COORDINATES["kyoto"][0],
        "longitude": CITY_COORDINATES["kyoto"][1],
    }
    assert data["location"]["radius"] >= 1
    assert offers[0]["source"] == DUFFEL_SOURCE


def test_a_search_never_reaches_a_booking_endpoint():
    """The integration is read-only: search only, no quote and no booking."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": {"results": []}})

    provider = DuffelProvider(token=LIVE_TOKEN, transport=httpx.MockTransport(handler))
    provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=None)

    assert seen and all("/stays/bookings" not in url for url in seen)
    assert all("/stays/quotes" not in url for url in seen)


def test_offers_come_back_cheapest_first():
    transport = _responds(
        {"data": {"results": [_result("900.00"), _result("400.00"), _result("650.00")]}}
    )
    provider = DuffelProvider(token=LIVE_TOKEN, transport=transport)

    offers = provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=None)

    assert [offer["total_cost"] for offer in offers] == [400.00, 650.00, 900.00]


def test_unpriced_results_are_dropped_not_sorted_to_the_front():
    transport = _responds({"data": {"results": [_result(None), _result("400.00")]}})
    provider = DuffelProvider(token=LIVE_TOKEN, transport=transport)

    offers = provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=None)

    assert [offer["total_cost"] for offer in offers] == [400.00]


def test_the_nightly_ceiling_filters_offers_priced_in_the_same_currency():
    # 400.00 over four nights is 100/night; 900.00 is 225/night.
    transport = _responds({"data": {"results": [_result("900.00"), _result("400.00")]}})
    provider = DuffelProvider(token=LIVE_TOKEN, transport=transport)

    offers = provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=150)

    assert [offer["total_cost"] for offer in offers] == [400.00]


def test_the_nightly_ceiling_does_not_silently_drop_another_currency():
    """The ceiling is a bare USD number. Testing it against a yen rate would
    empty every Tokyo search under a realistic budget, and an empty result set
    says nothing about why."""
    transport = _responds({"data": {"results": [_result("120000", currency="JPY")]}})
    provider = DuffelProvider(token=LIVE_TOKEN, transport=transport)

    offers = provider.search_stays("Tokyo", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=150)

    assert len(offers) == 1
    assert offers[0]["currency"] == "JPY"


def test_an_unknown_city_is_rejected_before_any_network_call():
    provider = DuffelProvider(token=LIVE_TOKEN, transport=_explodes())

    with pytest.raises(DuffelError):
        provider.search_stays("Atlantis", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=None)


# --- failure modes ----------------------------------------------------------


def test_rate_limit_is_reported_distinctly():
    transport = _responds({}, status=429, headers={"ratelimit-reset": "60"})
    provider = DuffelProvider(token=LIVE_TOKEN, transport=transport)

    with pytest.raises(DuffelError, match="rate limit"):
        provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=None)


def test_validation_errors_reuse_the_shared_duffel_error_format():
    transport = _responds(
        {"errors": [{"code": "validation_error", "message": "check_out_date is invalid"}]},
        status=422,
        headers={"x-request-id": "req_123"},
    )
    provider = DuffelProvider(token=LIVE_TOKEN, transport=transport)

    with pytest.raises(DuffelError, match="check_out_date is invalid") as excinfo:
        provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=None)

    assert "req_123" in str(excinfo.value)


def test_timeout_does_not_point_at_a_knob_that_does_nothing_here():
    """`DUFFEL_SUPPLIER_TIMEOUT_MS` is an Air-only parameter — telling someone to
    tune it for a stays timeout would send them somewhere with no effect."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = DuffelProvider(token=LIVE_TOKEN, transport=httpx.MockTransport(handler))

    with pytest.raises(DuffelError) as excinfo:
        provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=1, max_nightly_rate=None)

    assert "timed out" in str(excinfo.value)
    assert "DUFFEL_SUPPLIER_TIMEOUT_MS" not in str(excinfo.value)


def test_fetch_stays_needs_no_provider_to_be_exercised():
    """The transport seam is a plain client, so the search is testable without
    constructing a provider or reading a token."""
    client = httpx.Client(transport=_responds({"data": {"results": [_result("400.00")]}}))

    offers = fetch_stays(client, "Kyoto", CHECK_IN, CHECK_OUT, 2, None)

    assert [offer["total_cost"] for offer in offers] == [400.00]


# --- test mode vs live mode -------------------------------------------------


def test_a_test_token_answers_from_sample_data_without_touching_duffel():
    """Duffel's Stays test inventory sits at one coordinate pair, so a test-mode
    search of a real city returns nothing. Sample data answers the question
    asked, and is labelled as sample data."""
    provider = DuffelProvider(token=TEST_TOKEN, transport=_explodes())

    offers = provider.search_stays("Kyoto", CHECK_IN, CHECK_OUT, guests=2, max_nightly_rate=None)

    assert offers, "sample data should still answer the query"
    assert all(offer["source"] == SAMPLE_SOURCE for offer in offers)
    assert all(offer["synthetic"] is True for offer in offers)


def test_a_test_token_accepts_a_city_the_live_table_does_not_know():
    """Sample data needs no coordinates, so the city list must not constrain it."""
    provider = DuffelProvider(token=TEST_TOKEN, transport=_explodes())

    assert provider.search_stays("Atlantis", CHECK_IN, CHECK_OUT, 2, None)


def test_live_lodging_reaches_the_tool_with_no_warning(monkeypatch):
    """End to end: real inventory carries no caveat, which has never been true
    of lodging before."""
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    monkeypatch.setenv("DUFFEL_API_TOKEN", LIVE_TOKEN)

    from travel_agent.tools import availability

    monkeypatch.setitem(
        availability._PROVIDERS,
        "duffel",
        lambda: DuffelProvider(
            token=LIVE_TOKEN,
            transport=_responds({"data": {"results": [_result(expires_at=_future())]}}),
        ),
    )

    result = search_stays.invoke(
        {"location": "Kyoto", "check_in": CHECK_IN, "check_out": CHECK_OUT, "guests": 2}
    )

    assert result["count"] == 1
    assert result["sources"] == [DUFFEL_SOURCE]
    assert result["offers"][0]["synthetic"] is False
    assert "warning" not in result


def test_an_empty_live_search_says_nothing_was_found_without_a_caveat(monkeypatch):
    """The mirror of the sample-data rule: real inventory that returned nothing
    is a real finding, and must not be dressed up as a caveat."""
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    monkeypatch.setenv("DUFFEL_API_TOKEN", LIVE_TOKEN)

    from travel_agent.tools import availability

    monkeypatch.setitem(
        availability._PROVIDERS,
        "duffel",
        lambda: DuffelProvider(token=LIVE_TOKEN, transport=_responds({"data": {"results": []}})),
    )

    result = search_stays.invoke(
        {"location": "Kyoto", "check_in": CHECK_IN, "check_out": CHECK_OUT, "guests": 2}
    )

    assert result["count"] == 0
    assert "warning" not in result


# --- truncation -------------------------------------------------------------


def test_a_trimmed_stays_search_describes_what_it_kept_in_stays_terms(monkeypatch):
    """The note is shown to the traveler, and flights' wording — fastest, fewest
    stops — describes nothing a hotel has."""
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")
    monkeypatch.setenv("DUFFEL_API_TOKEN", LIVE_TOKEN)

    from travel_agent.tools import availability

    many = {"data": {"results": [_result(f"{400 + index}.00") for index in range(25)]}}
    monkeypatch.setitem(
        availability._PROVIDERS,
        "duffel",
        lambda: DuffelProvider(token=LIVE_TOKEN, transport=_responds(many)),
    )

    result = search_stays.invoke(
        {"location": "Kyoto", "check_in": CHECK_IN, "check_out": CHECK_OUT, "guests": 2}
    )

    assert result["total_found"] == 25
    assert result["count"] == 20
    assert "cheapest" in result["truncated"]
    assert "fastest" not in result["truncated"]
    assert "airports" not in result["truncated"]

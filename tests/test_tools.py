"""Tests for the pure-logic tools — no model calls, no network."""

from __future__ import annotations

import pytest

from travel_agent.tools.availability import (
    CABIN_CLASSES,
    SAMPLE_SOURCE,
    date_offset,
    search_flights,
    search_stays,
)
from travel_agent.tools.budget import summarize_budget


def _items() -> list[dict]:
    return [
        {"label": "SFO-NRT return", "category": "flights", "amount": 900.0, "quantity": 2},
        {"label": "Shinjuku hotel", "category": "lodging", "amount": 180.0, "quantity": 5},
        {"label": "JR Pass", "category": "transport", "amount": 220.0, "quantity": 2},
    ]


def test_budget_totals_and_categories():
    result = summarize_budget.invoke({"items": _items(), "budget_total": 5000.0, "currency": "USD"})

    assert result["total_estimated"] == pytest.approx(1800 + 900 + 440)
    assert result["over_budget"] is False
    assert result["remaining"] == pytest.approx(5000 - 3140)
    assert result["by_category"]["flights"] == pytest.approx(1800.0)
    # Categories come back largest-first.
    assert list(result["by_category"]) == ["flights", "lodging", "transport"]
    assert result["largest_line_items"][0]["label"] == "SFO-NRT return"


def test_budget_flags_overage():
    result = summarize_budget.invoke({"items": _items(), "budget_total": 2000.0})

    assert result["over_budget"] is True
    assert result["overage"] == pytest.approx(1140.0)
    assert result["percent_of_budget_used"] == pytest.approx(157.0)


def test_budget_rejects_empty_and_nonpositive():
    assert "error" in summarize_budget.invoke({"items": [], "budget_total": 100.0})
    assert "error" in summarize_budget.invoke({"items": _items(), "budget_total": 0.0})


def test_flight_offers_are_labelled_sorted_and_deterministic():
    query = {
        "origin": "SFO",
        "destination": "NRT",
        "depart_date": "2026-09-12",
        "return_date": "2026-09-20",
        "travelers": 2,
    }
    result = search_flights.invoke(query)

    assert result["provider"] == SAMPLE_SOURCE
    assert "warning" in result, "sample offers must carry the not-real-data warning"

    fares = [offer["total_fare"] for offer in result["offers"]]
    assert fares == sorted(fares)
    assert all(offer["source"] == SAMPLE_SOURCE for offer in result["offers"])
    assert all(offer["synthetic"] is True for offer in result["offers"])
    assert all(offer["total_fare"] == offer["fare_per_traveler"] * 2 for offer in result["offers"])

    # Same query, same offers — prices must not drift mid-conversation.
    assert search_flights.invoke(query) == result


def _flight_query(**overrides) -> dict:
    query = {
        "origin": "SFO",
        "destination": "NRT",
        "depart_date": "2026-09-12",
        "return_date": "2026-09-20",
        "travelers": 2,
    }
    return query | overrides


def test_cabin_defaults_to_economy_and_is_echoed():
    result = search_flights.invoke(_flight_query())

    assert result["requested_cabin"] == "economy"
    assert all(offer["cabin"] == "economy" for offer in result["offers"])
    # Sample offers always come back in the cabin asked for, so there is
    # nothing to caveat.
    assert "cabin_note" not in result


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("business", "business"),
        ("Business", "business"),
        ("PREMIUM ECONOMY", "premium_economy"),
        ("premium-economy", "premium_economy"),
        ("  First  ", "first"),
    ],
)
def test_cabin_case_and_separators_are_forgiven(given, expected):
    result = search_flights.invoke(_flight_query(cabin=given))

    assert result["requested_cabin"] == expected
    assert all(offer["cabin"] == expected for offer in result["offers"])


@pytest.mark.parametrize("given", ["coach", "biz", "first class", "", "premium"])
def test_unrecognised_cabins_are_errors_rather_than_guesses(given):
    """Coercing "biz" to business commits the traveler to a fare they never asked for."""
    result = search_flights.invoke(_flight_query(cabin=given))

    assert "error" in result
    assert "premium_economy" in result["error"], "the error must name what does work"
    assert "offers" not in result


def test_every_documented_cabin_is_priced():
    """Pins `_CABIN_FARE_MULTIPLIERS` against `CABIN_CLASSES` — two lists that must agree.

    The multiplier is indexed directly rather than with `.get`, so a fifth
    cabin added without a fare would raise here rather than silently pricing
    business at economy rates.
    """
    fares = {}
    for cabin in CABIN_CLASSES:
        result = search_flights.invoke(_flight_query(cabin=cabin))
        assert "error" not in result
        fares[cabin] = result["offers"][0]["total_fare"]

    ordered = [fares[cabin] for cabin in CABIN_CLASSES]
    assert ordered == sorted(ordered), "a dearer cabin must not come back cheaper"
    assert len(set(ordered)) == len(ordered), "each cabin must be priced distinctly"


def test_cabin_changes_the_price_but_not_the_flights():
    """Cabin is deliberately not part of `_seed`: same flights, different seat."""
    economy = search_flights.invoke(_flight_query(cabin="economy"))
    business = search_flights.invoke(_flight_query(cabin="business"))

    def shape(result):
        return [(offer["carrier"], offer["stops"], offer["depart_time_local"]) for offer in result]

    assert shape(economy["offers"]) == shape(business["offers"])
    assert economy["offers"][0]["total_fare"] < business["offers"][0]["total_fare"]


def test_stay_offers_respect_nightly_ceiling():
    result = search_stays.invoke(
        {
            "location": "Kyoto",
            "check_in": "2026-09-14",
            "check_out": "2026-09-18",
            "guests": 2,
            "max_nightly_rate": 150.0,
        }
    )

    assert all(offer["nightly_rate"] <= 150.0 for offer in result["offers"])
    assert all(offer["nights"] == 4 for offer in result["offers"])
    assert all(
        offer["total_cost"] == pytest.approx(offer["nightly_rate"] * 4)
        for offer in result["offers"]
    )


def test_empty_result_still_carries_the_synthetic_warning():
    """A zero-result sample search must not read as 'we checked, there is nothing'."""
    result = search_stays.invoke(
        {
            "location": "Kyoto",
            "check_in": "2026-09-14",
            "check_out": "2026-09-18",
            "max_nightly_rate": 1.0,  # filters everything out
        }
    )

    assert result["count"] == 0
    assert result["offers"] == []
    assert "warning" in result, "an empty synthetic search still is not real availability"


def test_invalid_dates_return_errors_not_exceptions():
    assert "error" in search_flights.invoke(
        {"origin": "SFO", "destination": "NRT", "depart_date": "12/09/2026"}
    )
    assert "error" in search_stays.invoke(
        {"location": "Kyoto", "check_in": "2026-09-18", "check_out": "2026-09-14"}
    )
    assert "error" in search_flights.invoke(
        {
            "origin": "SFO",
            "destination": "NRT",
            "depart_date": "2026-09-20",
            "return_date": "2026-09-12",
        }
    )


def test_unknown_provider_is_reported_as_a_tool_error(monkeypatch):
    """Provider construction failures must not escape the tool as exceptions."""
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "typo-provider")

    result = search_flights.invoke(
        {"origin": "SFO", "destination": "NRT", "depart_date": "2026-09-12"}
    )
    assert "error" in result
    assert "Registered providers" in result["error"]


def test_missing_duffel_token_is_reported_as_a_tool_error(monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")

    result = search_flights.invoke(
        {"origin": "SFO", "destination": "NRT", "depart_date": "2026-09-12"}
    )
    assert "error" in result
    assert "DUFFEL_API_TOKEN" in result["error"]


def test_date_offset_separates_dates_from_errors():
    assert date_offset.invoke({"start_date": "2026-09-12", "days": 3}) == {"date": "2026-09-15"}
    assert date_offset.invoke({"start_date": "2026-09-12", "days": -1}) == {"date": "2026-09-11"}

    # An error must not come back through the same channel as a date, or it
    # ends up pasted into the itinerary as a day header.
    bad = date_offset.invoke({"start_date": "12/09/2026", "days": 3})
    assert "date" not in bad
    assert "error" in bad

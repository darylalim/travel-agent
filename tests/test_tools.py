"""Tests for the pure-logic tools — no model calls, no network."""

from __future__ import annotations

import pytest

from travel_agent.tools.availability import SAMPLE_SOURCE, date_offset, search_flights, search_stays
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
    assert all(offer["total_fare"] == offer["fare_per_traveler"] * 2 for offer in result["offers"])

    # Same query, same offers — prices must not drift mid-conversation.
    assert search_flights.invoke(query) == result


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


def test_invalid_dates_return_errors_not_exceptions():
    assert "error" in search_flights.invoke(
        {"origin": "SFO", "destination": "NRT", "depart_date": "12/09/2026"}
    )
    assert "error" in search_stays.invoke(
        {"location": "Kyoto", "check_in": "2026-09-18", "check_out": "2026-09-14"}
    )


def test_date_offset():
    assert date_offset.invoke({"start_date": "2026-09-12", "days": 3}) == "2026-09-15"
    assert date_offset.invoke({"start_date": "2026-09-12", "days": -1}) == "2026-09-11"

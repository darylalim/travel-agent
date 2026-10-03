"""Budget arithmetic.

The agent is good at deciding what belongs in a budget and bad at adding up
a column of numbers. This tool does the arithmetic so the model does not
have to; everything here is a pure function with no I/O.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

Category = Literal["flights", "lodging", "food", "activities", "transport", "fees", "other"]


class BudgetItem(BaseModel):
    """A single estimated cost line in a trip budget."""

    label: str = Field(description="What this cost is, e.g. 'SFO-NRT return, 2 pax'")
    category: Category = Field(description="Cost category this line rolls up into")
    amount: float = Field(description="Cost per unit, in the trip currency")
    quantity: int = Field(default=1, ge=1, description="Number of units, e.g. nights or people")
    estimated: bool = Field(
        default=True,
        description="False only when this is a confirmed price the traveler has been quoted",
    )


@tool
def summarize_budget(
    items: list[BudgetItem],
    budget_total: float | None = None,
    currency: str = "USD",
) -> dict:
    """Total a list of trip cost lines, and compare them against a budget if there is one.

    Use this for every budget calculation rather than adding figures up in
    your head. Returns per-category subtotals, the grand total and the largest
    line items. With a budget it also returns how much remains, whether the
    plan is over, and by how much. Without one, `budget_total` comes back null
    and there is no comparison: report the total, not a verdict.

    Args:
        items: Every cost line in the plan. Multiply-out lines (5 nights of
            lodging) should use `quantity`, not a pre-multiplied `amount`.
        budget_total: The traveler's total budget for the trip. Leave it out
            when they have not set one. Never supply a placeholder: a made-up
            budget comes back as an over- or under-budget verdict on a number
            the traveler never gave.
        currency: Currency code all amounts are expressed in. The tool does no
            conversion, so convert any figure quoted in another currency
            before passing it.
    """
    if budget_total is not None and budget_total <= 0:
        return {"error": "budget_total must be greater than zero; leave it out if there is none."}
    if not items:
        return {"error": "No cost items supplied; nothing to total."}

    by_category: dict[str, float] = {}
    lines = []
    total = 0.0

    for item in items:
        subtotal = round(item.amount * item.quantity, 2)
        total += subtotal
        by_category[item.category] = round(by_category.get(item.category, 0.0) + subtotal, 2)
        lines.append(
            {
                "label": item.label,
                "category": item.category,
                "unit_amount": round(item.amount, 2),
                "quantity": item.quantity,
                "subtotal": subtotal,
                "estimated": item.estimated,
            }
        )

    total = round(total, 2)
    largest = sorted(lines, key=lambda line: line["subtotal"], reverse=True)[:3]

    # The comparison keys are absent, not null, without a budget: a reader that
    # treats null as "nothing to say" would otherwise read `over_budget: None`
    # as a plan that fits.
    comparison = (
        {
            "remaining": round(budget_total - total, 2),
            "over_budget": total > budget_total,
            "overage": round(total - budget_total, 2) if total > budget_total else 0.0,
            "percent_of_budget_used": round(total / budget_total * 100, 1),
        }
        if budget_total is not None
        else {}
    )
    return {
        "currency": currency,
        "budget_total": None if budget_total is None else round(budget_total, 2),
        "total_estimated": total,
        **comparison,
        "by_category": dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)),
        "largest_line_items": largest,
        "all_lines_estimated": all(item.estimated for item in items),
        "line_count": len(lines),
    }

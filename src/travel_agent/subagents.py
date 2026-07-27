"""Subagent definitions.

Three subagents, each mapping to a chunk of work the main agent can hand off
whole: research, option gathering, and costing. Subagents are stateless — the
`description` is what the main agent reads when deciding to delegate, so it
carries the "brief me completely" contract.
"""

from __future__ import annotations

from deepagents import SubAgent
from langchain_core.tools import BaseTool

from travel_agent.prompts import AVAILABILITY_PROMPT, BUDGET_PROMPT, RESEARCHER_PROMPT, WORKSPACE
from travel_agent.tools.availability import date_offset, search_flights, search_stays
from travel_agent.tools.budget import summarize_budget
from travel_agent.tools.search import build_search_tools


def build_subagents(search_tools: list[BaseTool] | None = None) -> list[SubAgent]:
    """Assemble the subagent roster for the main travel agent.

    Args:
        search_tools: Web search tools to give the researcher. Pass the same
            list the main agent got so the tools are constructed once.
    """
    if search_tools is None:
        search_tools = build_search_tools()

    return [
        SubAgent(
            name="destination-researcher",
            description=(
                "Researches a destination on the web: attractions, neighbourhoods, "
                "local transport, seasonality, entry requirements, and closures. "
                f"Give it the destination, the travel dates, and what the traveler "
                f"cares about; it writes findings to {WORKSPACE}/research.md and "
                "returns the highlights."
            ),
            system_prompt=RESEARCHER_PROMPT,
            tools=search_tools,
        ),
        SubAgent(
            name="availability-scout",
            description=(
                "Gathers flight and lodging candidates. Give it origin and "
                "destination airports, exact dates, traveler count, and any budget "
                f"ceiling; it writes candidates to {WORKSPACE}/options.md and "
                "returns a cheapest-vs-most-convenient comparison."
            ),
            system_prompt=AVAILABILITY_PROMPT,
            tools=[search_flights, search_stays, date_offset],
        ),
        SubAgent(
            name="budget-analyst",
            description=(
                "Costs a plan against the traveler's budget. Give it every cost "
                "line with amounts and the total budget; it writes the breakdown "
                f"to {WORKSPACE}/budget.md and returns the total, whether it fits, "
                "and specific cuts if it does not."
            ),
            system_prompt=BUDGET_PROMPT,
            tools=[summarize_budget],
        ),
    ]

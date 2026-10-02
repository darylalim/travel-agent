"""Constants that callers need without paying for agent construction.

Importing `travel_agent.agent` builds the graph — a model client and the whole
tool stack — as a side effect, because `langgraph.json` needs a module-level
`graph`. Anything that only wants a default belongs here instead.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ModelSpec(NamedTuple):
    """A provider-prefixed model id and the effort to run it at.

    `effort=None` sends no effort at all, leaving the model on its own default.
    That is required, not just allowed, for Haiku 4.5: it rejects the parameter.
    """

    model: str
    effort: Effort | None


DEFAULT_MODEL = "anthropic:claude-opus-5-5"

# Opus 5.5's own default, written out so the choice is visible in a diff. The
# main agent mostly delegates and synthesises, and the prompts already push
# against over-thinking. Unmeasured: raise to "high" if itineraries get worse.
DEFAULT_EFFORT: Effort = "medium"

# Per-subagent models, keyed by `SubAgent` name. A subagent missing from this
# map inherits the main agent's model — silently, so `test_models.py` pins the
# keys to the roster.
#
# - The researcher reads the most (raw search results) and needs the least
#   judgment, so it takes the cheaper model at reduced effort.
# - The scout stays at "high", Sonnet 5.5's own default. Its cost is mostly
#   input (offer payloads), which effort does not touch, so "medium" saves
#   little; it also makes fewer, more consolidated calls, and the alternate
#   airports and shifted dates it would drop are what surface a nonstop or a
#   cheaper day. Try "medium" if the scout is slow — fewer Duffel round trips —
#   and check it still varies its queries. Effort is not the guard on carrying
#   `warning` through; the tool docstrings and AVAILABILITY_PROMPT are.
# - The analyst hands every figure to `summarize_budget` and mostly arranges
#   cost lines, which Haiku handles.
SUBAGENT_MODELS: dict[str, ModelSpec] = {
    "destination-researcher": ModelSpec("anthropic:claude-sonnet-5-5", "medium"),
    "availability-scout": ModelSpec("anthropic:claude-sonnet-5-5", "high"),
    "budget-analyst": ModelSpec("anthropic:claude-haiku-4-5", None),
}

# Thinking is always on for Opus 5.5 and counts against this ceiling, so keep
# it generous. It applies to every subagent too.
DEFAULT_MAX_TOKENS = 16_000

# Root run name for LangSmith traces from the in-process entry points. It
# matches the graph id in `langgraph.json`, so CLI and Streamlit traces sit
# under the same name as the ones Studio produces. Each entry point adds its
# own tag (`cli`, `streamlit`) so they can still be told apart.
TRACE_NAME = "travel_agent"

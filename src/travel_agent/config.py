"""Constants that callers need without paying for agent construction.

Importing `travel_agent.agent` builds the graph — a model client and the whole
tool stack — as a side effect, because `langgraph.json` needs a module-level
`graph`. Anything that only wants a default belongs here instead.
"""

from __future__ import annotations

DEFAULT_MODEL = "anthropic:claude-opus-5"

# Claude Opus 5 thinks by default and thinking counts against this ceiling,
# so keep it generous.
DEFAULT_MAX_TOKENS = 16_000

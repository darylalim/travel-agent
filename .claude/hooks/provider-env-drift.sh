#!/usr/bin/env bash
# PostToolUse (Edit|Write) — every provider env var must appear in _PROVIDER_ENV.
#
# CLAUDE.md, on tests/conftest.py: "**Add any new provider env var to
# _PROVIDER_ENV.**" The autouse fixture only unsets the names listed in that
# tuple. Anything missing leaks the developer's exported shell config into the
# suite — and the combination the README tells you to set
# (TRAVEL_AGENT_PROVIDER=duffel plus a token) makes the "no network" tests send
# real requests to api.duffel.com. Nothing fails loudly when that happens,
# which is exactly why it needs a mechanical check.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$root" 2>/dev/null || exit 0

file=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)
case "$file" in
  *src/travel_agent/tools/*.py|*tests/conftest.py) ;;
  *) exit 0 ;;
esac
[ -d src/travel_agent/tools ] && [ -f tests/conftest.py ] || exit 0

# Newlines are squeezed to spaces first so a call ruff has wrapped across lines
# is still seen. All four idioms this codebase could plausibly use are covered,
# in either quote style: os.getenv("X"), os.environ.get("X"), os.environ["X"],
# and _env_int("X", ...) — duffel.py reaches os.getenv through that last one,
# so a bare `os.getenv(` scan would miss DUFFEL_SUPPLIER_TIMEOUT_MS.
joined=$(find src/travel_agent/tools -name '*.py' -exec cat {} + 2>/dev/null | tr '\n' ' ')
[ -n "$joined" ] || exit 0

reads='(os\.getenv|os\.environ\.get|_env_int)\([[:space:]]*["'\''][A-Z][A-Z0-9_]{2,}["'\'']'
reads="$reads"'|os\.environ\[[[:space:]]*["'\''][A-Z][A-Z0-9_]{2,}["'\'']'
declared=$(grep -oE "$reads" <<<"$joined" | grep -oE '[A-Z][A-Z0-9_]{2,}' | sort -u)
[ -n "$declared" ] || exit 0

# Read the _PROVIDER_ENV tuple, not every uppercase string in the file — a name
# mentioned only in a docstring is not actually unset by the fixture. [^=]*
# tolerates an annotated form: `_PROVIDER_ENV: tuple[str, ...] = (...)`.
tuple=$(sed -n '/_PROVIDER_ENV[^=]*=/,/)/p' tests/conftest.py 2>/dev/null)
if [ -z "$tuple" ]; then
  if grep -q '_PROVIDER_ENV' tests/conftest.py 2>/dev/null; then
    # Present but shaped in a way this sed range cannot bracket. Fail OPEN:
    # scan the whole file rather than block on a parse we are unsure about.
    tuple=$(cat tests/conftest.py)
  else
    printf '_PROVIDER_ENV is gone from tests/conftest.py. The autouse fixture in that file is the only thing keeping ambient provider config (and therefore real api.duffel.com traffic) out of the test suite. Restore it or replace it deliberately.\n' >&2
    exit 2
  fi
fi
covered=$(grep -oE '["'\''][A-Z][A-Z0-9_]{2,}["'\'']' <<<"$tuple" | tr -d "\"'" | sort -u)

# TAVILY_API_KEY gates web search, not provider selection — deliberately excused.
excused="TAVILY_API_KEY"

missing=""
for name in $declared; do
  if ! grep -qxF "$name" <<<"$covered
$excused"; then
    missing="$missing $name"
  fi
done

if [ -n "$missing" ]; then
  {
    printf 'Env var(s) read in src/travel_agent/tools/ but absent from _PROVIDER_ENV in tests/conftest.py:%s\n\n' "$missing"
    printf 'The autouse fixture only unsets the names in that tuple, so anything missing lets a developer'"'"'s exported shell config reach the tests. With TRAVEL_AGENT_PROVIDER=duffel and a token exported, the "no network" suite starts hitting api.duffel.com and nothing fails loudly.\n\n'
    printf 'Add the name(s) to _PROVIDER_ENV, or add to the excused list in .claude/hooks/provider-env-drift.sh with a reason.\n'
  } >&2
  exit 2
fi
exit 0

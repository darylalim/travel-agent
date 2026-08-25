#!/usr/bin/env bash
# Stop / SubagentStop — the project must be green when a turn ends.
#
# This is the half of the old python-gate.sh that could not stay per-edit. A
# change spanning availability.py, duffel.py and tests/ is red at every
# intermediate edit, so blocking there pushes toward weakening the test rather
# than finishing the change. A turn boundary is the granularity the invariant
# actually has: by then the change is whole.
#
# Whole-project checks live here for the same reason — `ty check` across the
# project sees a signature change before its callers are updated, which is not
# an error, it is a refactor in progress. Per-file ty runs in python-gate.sh.
#
# No fingerprint state is needed (unlike the deleted honesty-cochange.sh, which
# blocked on a proxy the model was not expected to clear and so re-blocked every
# turn until it was suppressed). "The project is red" is a condition the model
# fixes, and stop_hook_active suppresses the re-stop within one continuation.
#
# SubagentStop is not optional: Stop does not fire for Task subagents, so
# Python edited inside availability-scout would otherwise end its turn ungated.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$root" 2>/dev/null || exit 0

case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac
command -v uv >/dev/null 2>&1 || exit 0

input=$(cat)
[ "$(jq -r '.stop_hook_active // false' <<<"$input" 2>/dev/null)" = "true" ] && exit 0

# Nothing to check if no Python changed in this working tree.
if git rev-parse --git-dir >/dev/null 2>&1; then
  git status --porcelain -- '*.py' 2>/dev/null | grep -q . || exit 0
fi

if ! out=$( { uv run ty check && uv run pytest -q; } 2>&1 ); then
  {
    printf 'The project is red at the end of this turn.\n\n'
    printf '%s\n' "$out" | tail -c 6000
    printf '\nGate: ty check -> pytest, over the whole project.\n'
  } >&2
  exit 2
fi
exit 0

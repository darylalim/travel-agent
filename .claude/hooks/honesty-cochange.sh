#!/usr/bin/env bash
# Stop — refuse to end a turn that changed a provider but no test.
#
# CLAUDE.md: the data-honesty invariant "cannot be maintained in one place" and
# "each had to touch availability.py, duffel.py, and the tests together". Git
# history agrees: every post-scaffold commit touching either provider file also
# touched tests/.
#
# Blocking is keyed to a FINGERPRINT of the provider diff, not to
# stop_hook_active. stop_hook_active only suppresses the immediate re-stop
# within one continuation; it resets on the next user turn, so on its own this
# hook would re-block the end of every turn for the rest of the session —
# including read-only turns that changed nothing. With the fingerprint, each
# distinct provider change is flagged exactly once; edit the provider further
# and the new state is flagged again, which is the intended behaviour.
set -uo pipefail

input=$(cat)

if [ "$(jq -r '.stop_hook_active // false' <<<"$input" 2>/dev/null)" = "true" ]; then
  exit 0
fi

root="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$root" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# Every file that can decide whether an offer is real. duffel_stays.py belongs
# here for the same reason the other two do: it sets `synthetic` and derives
# `free_cancellation` and `guest_rating`. NOTE this list is spelled twice — here
# and in the regex below — and they must agree, or a file is watched by the
# fingerprint but never triggers the check.
PROVIDERS="src/travel_agent/tools/availability.py src/travel_agent/tools/duffel.py src/travel_agent/tools/duffel_stays.py"

# Porcelain paths are always relative to the repo root. $NF picks the
# destination side of a rename ("R  old -> new").
changed=$(git status --porcelain -- src tests 2>/dev/null | awk '{print $NF}')
[ -n "$changed" ] || exit 0

grep -qE '^src/travel_agent/tools/(availability|duffel|duffel_stays)\.py$' <<<"$changed" || exit 0
grep -q '^tests/' <<<"$changed" && exit 0

# State lives outside the repo so it never shows up in git status or a diff.
state_dir="${TMPDIR:-/tmp}/claude-hooks-cochange"
mkdir -p "$state_dir" 2>/dev/null || exit 0
# shellcheck disable=SC2086
fingerprint=$(git diff HEAD -- $PROVIDERS 2>/dev/null | cksum)
state_file="$state_dir/$(cksum <<<"$root" | tr -d ' ')"

if [ -f "$state_file" ] && [ "$(cat "$state_file" 2>/dev/null)" = "$fingerprint" ]; then
  exit 0
fi
printf '%s' "$fingerprint" >"$state_file" 2>/dev/null

jq -n '{decision: "block", reason: (
  "A provider file (availability.py, duffel.py or duffel_stays.py) has uncommitted changes, but nothing under tests/ does.\n\n" +
  "The data-honesty invariant — the traveler is never shown synthetic inventory described as real — spans four enforcement points that must agree:\n" +
  "  1. every offer carries `synthetic: bool` alongside `source`\n" +
  "  2. providers implement `synthetic_note(kind)`, consulted even when a search returns nothing\n" +
  "  3. `_wrap()` attaches the resulting `warning`\n" +
  "  4. the @tool docstrings key off `warning`/`synthetic`, never the provider name\n\n" +
  "Both prior data-honesty commits touched availability.py, duffel.py and the tests in the same change. Either extend the tests to cover what you changed, or state explicitly why this change cannot affect the invariant.\n\n" +
  "This exact provider diff will not be flagged again — only a further change to those files will."
)}'
exit 0

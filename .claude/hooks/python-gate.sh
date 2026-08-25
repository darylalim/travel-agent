#!/usr/bin/env bash
# PostToolUse (Edit|Write) — format, lint and type-check the file that was edited.
#
# Deliberately does NOT run pytest. Wall clock was the smaller problem (the old
# whole-suite gate cost ~5.1s on every edited .py, against a header that still
# claimed 0.7s for 47 tests). The real one is that a cross-cutting change is red
# at edits 1..n-1 by construction, and CLAUDE.md mandates exactly those changes:
# the data-honesty invariant "cannot be maintained in one place" and "each had
# to touch availability.py, duffel.py, and the tests together". A per-edit
# `exit 2` therefore fires on a state that is legitimately incomplete and says
# "fix this before continuing" — and the cheapest way to comply mid-refactor is
# to weaken the assertion. The suite runs at the turn boundary instead, where
# the change is whole; see turn-gate.sh.
#
# Everything left is scoped to the ONE file that changed, so its verdict is
# correct without any other file having landed yet. That is the line the split
# follows: per-file checks here, whole-project checks per turn.
#
# `ruff format` is applied rather than checked because that is the one thing CI
# structurally cannot do — ci.yml runs `ruff format --check` and can only
# report. Applying it here means the file on disk already matches what CI will
# demand.
#
# `ruff check --fix` is deliberately NOT run: this project has no
# [tool.ruff.lint] select, so ruff's default set includes F401, whose "safe" fix
# DELETES unused imports. In a two-step edit that adds an import and then adds
# the code using it, --fix reverts step one on disk — observed reverting a real
# edit to duffel.py to an empty diff while exiting 0. Lint problems are reported
# for Claude to fix, never fixed behind its back.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$root" 2>/dev/null || exit 0

# uv lives in ~/.local/bin, which a non-interactive hook shell may not have.
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac
command -v uv >/dev/null 2>&1 || exit 0

file=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)
[ -n "$file" ] || exit 0
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
# Only gate files inside this project — an edit elsewhere must not run our tools.
case "$file" in
  "$root"/*) ;;
  /*) exit 0 ;;
esac
[ -f "$file" ] || exit 0

before=$(cksum <"$file")
uv run ruff format -q "$file" >/dev/null 2>&1
after=$(cksum <"$file")

if ! out=$( { uv run ruff check "$file" && uv run ty check "$file"; } 2>&1 ); then
  {
    printf 'Quality gate failed after editing %s\n\n' "${file#"$root"/}"
    printf '%s\n' "$out" | tail -c 6000
    printf '\nFix this before continuing. Gate: ruff check -> ty check, on this file only.\n'
    printf 'Nothing was auto-fixed except whitespace/layout; your code is on disk as you wrote it.\n'
  } >&2
  exit 2
fi

if [ "$before" != "$after" ]; then
  printf 'ruff format adjusted whitespace/layout in %s (no code removed) — re-read it before your next edit.\n' "${file#"$root"/}"
fi
exit 0

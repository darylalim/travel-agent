#!/usr/bin/env bash
# PostToolUse (Edit|Write) — run the documented quality gate on any edited Python file.
#
# The whole gate costs ~0.7s wall clock here (ruff 0.05s, ty 0.11s, pytest 0.55s
# for 47 tests), so there is no reason to guess which tests are relevant: run
# everything.
#
# `ruff format` is applied automatically because a formatter only moves bytes
# around. `ruff check --fix` is deliberately NOT run: this project has no
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
# Only gate files inside this project — an edit elsewhere must not run our suite.
case "$file" in
  "$root"/*) ;;
  /*) exit 0 ;;
esac
[ -f "$file" ] || exit 0

before=$(cksum <"$file")
uv run ruff format -q "$file" >/dev/null 2>&1
after=$(cksum <"$file")

if ! out=$( { uv run ruff check "$file" && uv run ty check && uv run pytest -q; } 2>&1 ); then
  {
    printf 'Quality gate failed after editing %s\n\n' "${file#"$root"/}"
    printf '%s\n' "$out" | tail -c 6000
    printf '\nFix this before continuing. Gate: ruff check -> ty check -> pytest.\n'
    printf 'Nothing was auto-fixed except whitespace/layout; your code is on disk as you wrote it.\n'
  } >&2
  exit 2
fi

if [ "$before" != "$after" ]; then
  printf 'ruff format adjusted whitespace/layout in %s (no code removed) — re-read it before your next edit.\n' "${file#"$root"/}"
fi
exit 0

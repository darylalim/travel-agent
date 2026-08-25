#!/usr/bin/env bash
# Tests for the hooks in this directory.
#
#   bash .claude/hooks/test-hooks.sh
#
# Each case feeds a synthetic hook payload on stdin and asserts the contract:
# a PreToolUse hook denies by printing permissionDecision:"deny" on stdout and
# exiting 0; a PostToolUse or Stop hook reports by exiting 2 with stderr.
# "Allow" means silence and exit 0.
#
# These hooks are built out of quote-boundary regexes, and an adversarial review
# of the first draft found thirteen real defects in them — including one where
# `ruff check --fix` deleted imports Claude had just written, reverting a real
# edit to an empty diff. Most of the cases below pin a specific one of those
# defects. Character classes and verb lists here are load-bearing; change one
# and run this.
#
# Three hooks this file used to cover are gone, and their property did not go
# with them — it moved to pytest, where it also runs in CI on three Python
# versions and on contributors who are not using Claude Code:
#   no-booking.sh          -> tests/test_no_booking.py
#   provider-env-drift.sh  -> tests/test_env_isolation.py
#   honesty-cochange.sh    -> cut; its predicate ("a test file also changed")
#                             was satisfied by the one commit that shipped the
#                             defect it existed to prevent.
#
# No network, no model calls. Most fixtures go in mktemp dirs. A few cases need
# a real file inside the project, because the gates deliberately refuse to run
# for a file outside $CLAUDE_PROJECT_DIR; those are dropped by
# drop_project_files the moment their case ends — turn-gate.sh checks the WHOLE
# project, so a deliberately-broken fixture left in place fails every later
# case — and the EXIT trap catches whatever an interrupted run would otherwise
# strand at the repo root.
set -uo pipefail

ROOT=${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
H="$ROOT/.claude/hooks"
export CLAUDE_PROJECT_DIR="$ROOT"

TMPFILES=()
TMPDIRS=()
cleanup() {
  local f d
  for f in ${TMPFILES+"${TMPFILES[@]}"}; do rm -f "$f"; done
  for d in ${TMPDIRS+"${TMPDIRS[@]}"}; do rm -rf "$d"; done
}
trap cleanup EXIT INT TERM

# A file inside the project, registered with the trap before it is written.
# Call drop_project_files as soon as a case is done: turn-gate.sh runs
# `ty check` and `pytest` over the WHOLE project, so a deliberately-broken
# fixture left lying around fails every later case. The trap is the safety net
# for an interrupted run; this is the correctness mechanism.
project_file() { TMPFILES+=("$ROOT/$1"); printf '%s' "$2" > "$ROOT/$1"; }
drop_project_files() { local f; for f in ${TMPFILES+"${TMPFILES[@]}"}; do rm -f "$f"; done; TMPFILES=(); }
scratch() { local d; d=$(mktemp -d); TMPDIRS+=("$d"); printf '%s' "$d"; }

pass=0; fail=0
ok()  { pass=$((pass+1)); printf '  ok   %s\n' "$1"; }
bad() { fail=$((fail+1)); printf '  FAIL %s -- %s\n' "$1" "$2"; }
run() { OUT=$(printf '%s' "$2" | bash "$H/$1" 2>/tmp/hook-test-err); RC=$?; ERR=$(cat /tmp/hook-test-err); }
bash_payload() { jq -n --arg c "$1" '{tool_name:"Bash",tool_input:{command:$c}}'; }

deny()  { printf '%s' "$OUT" | jq -e '.hookSpecificOutput.permissionDecision=="deny"' >/dev/null 2>&1 \
            && ok "$1" || bad "$1" "want deny; rc=$RC out=${OUT:0:120}"; }
allow() { [ -z "$OUT" ] && [ "$RC" = 0 ] && ok "$1" || bad "$1" "want allow; rc=$RC out=${OUT:0:120}"; }
rc_is() { [ "$RC" = "$1" ] && ok "$2" || bad "$2" "want rc=$1 got rc=$RC; err=${ERR:0:160}"; }

for f in "$H"/*.sh; do bash -n "$f" || bad "syntax" "$f does not parse"; done

# --------------------------------------------------------------------------
echo "== python-gate.sh (per-edit, per-file) =="
# --------------------------------------------------------------------------
run python-gate.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/README.md"}}'
allow "non-python file skipped"
run python-gate.sh '{"tool_name":"Edit","tool_input":{"file_path":"/etc/hosts.py"}}'
allow "python file outside the project skipped"
run python-gate.sh '{"tool_name":"Bash","tool_input":{}}'
allow "missing file_path skipped"

# THE defect: ruff's default rule set includes F401, whose "safe" fix DELETES
# unused imports. Step 1 of a two-step edit stages imports before the code
# using them; --fix reverted a real edit to duffel.py to an empty diff.
project_file _test_imports.py '"""Step 1 of a two-step edit."""

import json

from travel_agent.tools.duffel import DuffelError


def stub() -> None:
    raise NotImplementedError
'
run python-gate.sh '{"tool_name":"Write","tool_input":{"file_path":"'"$ROOT"'/_test_imports.py"}}'
{ grep -q '^import json' "$ROOT/_test_imports.py" && grep -q 'import DuffelError' "$ROOT/_test_imports.py"; } \
  && ok "staged imports SURVIVE the gate (no --fix)" \
  || bad "staged imports survive" "imports were deleted from disk"
rc_is 2 "unused imports are reported via exit 2, not silently removed"
drop_project_files

project_file _test_fmt.py 'x=1
y  =  2
'
run python-gate.sh '{"tool_name":"Write","tool_input":{"file_path":"'"$ROOT"'/_test_fmt.py"}}'
{ [ "$RC" = 0 ] && grep -q '^x = 1' "$ROOT/_test_fmt.py"; } \
  && ok "ruff format still auto-applied (layout only, never deletes code)" \
  || bad "format retained" "rc=$RC"
drop_project_files

project_file _test_bad.py 'def f() -> int:
    return "not an int"
'
run python-gate.sh '{"tool_name":"Write","tool_input":{"file_path":"'"$ROOT"'/_test_bad.py"}}'
{ [ "$RC" = 2 ] && [ -n "$ERR" ]; } && ok "type error surfaces as exit 2 with stderr" \
  || bad "type error detection" "rc=$RC err=${ERR:0:160}"
drop_project_files

# The whole point of the split: a per-edit gate must NOT run the suite. A file
# that is fine on its own but breaks another file's tests has to pass here, or
# the gate blocks every intermediate state of a cross-cutting change — which is
# exactly the shape CLAUDE.md mandates for the data-honesty invariant.
project_file _test_slow.py 'VALUE = 1
'
run python-gate.sh '{"tool_name":"Write","tool_input":{"file_path":"'"$ROOT"'/_test_slow.py"}}'
rc_is 0 "clean file passes without the suite being consulted"
drop_project_files

run python-gate.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/config.py"}}'
rc_is 0 "clean real source file passes the per-file gate"

# --------------------------------------------------------------------------
echo "== turn-gate.sh (per-turn, whole project) =="
# --------------------------------------------------------------------------
run turn-gate.sh '{"stop_hook_active":true}'
allow "stop_hook_active short-circuits (no re-block inside a continuation)"

# A turn that changed no Python has nothing new to break, and paying ~5s at the
# end of a conversational turn is the friction this hook was moved to avoid.
G=$(scratch)
(cd "$G" && git init -q && echo hi > a.md \
 && git add -A && git -c user.email=t@t -c user.name=t commit -qm init \
 && echo more >> a.md)
OUT=$(printf '{"stop_hook_active":false}' | CLAUDE_PROJECT_DIR="$G" bash "$H/turn-gate.sh" 2>/dev/null); RC=$?
allow "no python changed in the working tree -> skipped"

# ty runs over the whole project here, which is what per-file ty cannot see.
project_file _test_turn_bad.py 'def f() -> int:
    return "not an int"
'
run turn-gate.sh '{"stop_hook_active":false}'
{ [ "$RC" = 2 ] && [ -n "$ERR" ]; } && ok "whole-project type error blocks the turn" \
  || bad "turn-gate ty" "rc=$RC err=${ERR:0:160}"
drop_project_files

# pytest really is in this gate — the half that moved out of python-gate.sh.
project_file tests/test_zz_hook_fixture.py 'def test_deliberately_red():
    assert False
'
run turn-gate.sh '{"stop_hook_active":false}'
{ [ "$RC" = 2 ] && [ -n "$ERR" ]; } && ok "failing test blocks the turn" \
  || bad "turn-gate pytest" "rc=$RC err=${ERR:0:160}"
drop_project_files

run turn-gate.sh '{"stop_hook_active":false}'
rc_is 0 "green project ends the turn cleanly"

# --------------------------------------------------------------------------
echo "== protect-env.sh =="
# --------------------------------------------------------------------------
run protect-env.sh '{"tool_name":"Read","tool_input":{"file_path":"'"$ROOT"'/.env"}}'
deny "Read of the real credentials file"
run protect-env.sh '{"tool_name":"Write","tool_input":{"file_path":"'"$ROOT"'/.env","content":"X=1"}}'
deny "Write (can repoint the agent at live inventory with no diff)"
run protect-env.sh '{"tool_name":"Grep","tool_input":{"pattern":"=","path":"'"$ROOT"'/.env","output_mode":"content"}}'
deny "Grep of an explicitly-named ignored file"
run protect-env.sh '{"tool_name":"Read","tool_input":{"file_path":"'"$ROOT"'/tests/fixtures/.env"}}'
deny "denied at any depth, not only at the repo root"
run protect-env.sh '{"tool_name":"Read","tool_input":{"file_path":"'"$ROOT"'/.env.example"}}'
allow "the documented example file stays readable"
run protect-env.sh '{"tool_name":"Grep","tool_input":{"pattern":"=","path":"'"$ROOT"'/.env.example"}}'
allow "Grep of the example file"
run protect-env.sh '{"tool_name":"Grep","tool_input":{"pattern":"x","path":"'"$ROOT"'/src"}}'
allow "ordinary source Grep"
run protect-env.sh '{"tool_name":"Read","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/agent.py"}}'
allow "ordinary source Read"

# Fails closed: any verb that is not provably metadata-only is denied.
while IFS= read -r c; do
  run protect-env.sh "$(bash_payload "$c")"
  deny "bash denied: $c"
done <<'CASES'
cat .env
less .env
source .env
cp .env /tmp/x
grep DUFFEL .env
diff .env .env.example
sort .env
base64 .env
dd if=.env
nvim .env
perl -ne 'print' .env
echo K=v >> .env
python -c "print(1)" < .env
echo $(cat .env)
export K=`head -1 .env`
{ cat .env; }
cd /tmp && cat .env
test -f .env; cat .env
cp .env.example .env
CASES

# git and find are safe only in SHAPE. The danger is in an argument, not the
# verb, so a verb list alone let all four of these through.
while IFS= read -r c; do
  run protect-env.sh "$(bash_payload "$c")"
  deny "bash denied (argument-shaped): $c"
done <<'CASES'
git diff --no-index .env .env.example
git add -f .env
git show :.env
find . -name .env -exec cat {} +
find . -name .env -delete
CASES

# Metadata-only verbs, and tokens that are not really paths, stay allowed.
while IFS= read -r c; do
  run protect-env.sh "$(bash_payload "$c")"
  allow "bash allowed: $c"
done <<'CASES'
git commit -m "Document .env handling"
find . -name ".env*"
test -f .env && echo present
ls -la .env
stat .env
wc -l .env
grep -rn "\.env" README.md
rg "\.env" .
cat .env.example
grep X .environment
NODE_ENV=test npm run x
echo $(cat README.md)
echo skipping .env
uv run pytest -q
CASES

# `printf ... | grep -q` returns 141 (SIGPIPE) under `set -o pipefail` once the
# input exceeds the 64 KiB pipe buffer, which inverts the test and silently
# allows a large command through. Both hooks use a herestring instead.
BIG=$(head -c 70000 /dev/zero | tr '\0' 'a')
run protect-env.sh "$(bash_payload "cat .env $BIG")"
deny "70 KB command mentioning the credentials file still denied"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]

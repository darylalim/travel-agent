#!/usr/bin/env bash
# Tests for the hooks in this directory.
#
#   bash .claude/hooks/test-hooks.sh
#
# Each case feeds a synthetic hook payload on stdin and asserts the contract:
# a PreToolUse hook denies by printing permissionDecision:"deny" on stdout and
# exiting 0; a PostToolUse hook reports by exiting 2 with stderr; a Stop hook
# blocks by printing decision:"block". "Allow" means silence and exit 0.
#
# These hooks are 300 lines of shell built out of quote-boundary regexes, and
# an adversarial review of the first draft found thirteen real defects in them
# — including one where `ruff check --fix` deleted imports Claude had just
# written, reverting a real edit to an empty diff. Most of the cases below pin
# a specific one of those defects. Character classes and verb lists here are
# load-bearing; change one and run this.
#
# No network, no model calls. Most fixtures go in mktemp dirs. A few cases need
# a real file inside the project, because python-gate deliberately refuses to
# run the suite for a file outside $CLAUDE_PROJECT_DIR; those are dropped by
# drop_project_files the moment their case ends — python-gate type-checks and
# tests the WHOLE project, so a deliberately-broken fixture left in place fails
# every later case — and the EXIT trap catches whatever an interrupted run
# would otherwise strand at the repo root.
set -uo pipefail

ROOT=${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
H="$ROOT/.claude/hooks"
export CLAUDE_PROJECT_DIR="$ROOT"
STATE_DIR="${TMPDIR:-/tmp}/claude-hooks-cochange"

TMPFILES=()
TMPDIRS=()
cleanup() {
  local f d
  for f in ${TMPFILES+"${TMPFILES[@]}"}; do rm -f "$f"; done
  for d in ${TMPDIRS+"${TMPDIRS[@]}"}; do rm -rf "$d"; done
  rm -rf "$STATE_DIR"
}
trap cleanup EXIT INT TERM

# A .py inside the project, registered with the trap before it is written.
# Call drop_project_files as soon as a case is done: python-gate runs
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
block() { printf '%s' "$OUT" | jq -e '.decision=="block"' >/dev/null 2>&1 \
            && ok "$1" || bad "$1" "want block; got ${OUT:0:120}"; }
rc_is() { [ "$RC" = "$1" ] && ok "$2" || bad "$2" "want rc=$1 got rc=$RC; err=${ERR:0:160}"; }

for f in "$H"/*.sh; do bash -n "$f" || bad "syntax" "$f does not parse"; done

# --------------------------------------------------------------------------
echo "== python-gate.sh =="
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
drop_project_files   # must go before the clean-file case: ty checks the whole project

run python-gate.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/config.py"}}'
rc_is 0 "clean real source file passes the full gate"

# --------------------------------------------------------------------------
echo "== no-booking.sh =="
# --------------------------------------------------------------------------
# duffel.py's own module docstring contains the bare text "POST /air/orders".
run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/tools/duffel.py","new_string":"Adding booking would mean `POST /air/orders`, which should sit behind a human approval gate."}}'
allow "docstring mentioning POST /air/orders is not blocked"
run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/tools/duffel.py","new_string":"assert \"/air/orders\" not in seen[\"url\"]"}}'
allow "assertion quoting the path is not blocked"
run no-booking.sh '{"tool_name":"Write","tool_input":{"file_path":"'"$ROOT"'/tests/test_duffel.py","content":"assert seen[\"url\"] != f\"{API_BASE}/air/orders\""}}'
allow "tests/ exempt, so the invariant can be given a regression test"
run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/prompts.py","new_string":"# Never call \"/air/orders\" from the agent loop."}}'
allow "comment quoting the path is not blocked"
run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"a.py","new_string":"r = post(f\"{API_BASE}/air/offer_requests\", json=p)"}}'
allow "offer_requests, the real code path, is allowed"
run no-booking.sh '{"tool_name":"Write","tool_input":{"file_path":"README.md","content":"POST \"/air/orders\" is never called."}}'
allow "non-python file skipped"

run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"a.py","new_string":"post(f\"{API_BASE}/air/orders\", json=p)"}}'
deny "f-string URL construction denied"
run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"a.py","new_string":"url = API_BASE + \"/air/orders\""}}'
deny "explicit concatenation denied"
run no-booking.sh '{"tool_name":"Write","tool_input":{"file_path":"a.py","content":"U = \"https://api.duffel.com/air/orders\""}}'
deny "full-host URL denied"
run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"a.py","new_string":"client.orders.create(offer_id=x)"}}'
deny "duffel-api SDK .orders.create( denied"
run no-booking.sh '{"tool_name":"Edit","tool_input":{"file_path":"a.py","new_string":"r = self._http().post(f\"{API_BASE}/air/payments\", json=p)"}}'
deny "/air/payments denied"

# `printf ... | grep -q` returns 141 under pipefail past the 64 KiB pipe
# buffer, which inverts the guard. Herestrings are why this passes.
BIG=$(head -c 70000 /dev/zero | tr '\0' 'x')
run no-booking.sh "$(jq -n --arg s "post(f\"{API_BASE}/air/orders\") # $BIG" \
  '{tool_name:"Edit",tool_input:{file_path:"a.py",new_string:$s}}')"
deny "70 KB payload with booking code still denied (no SIGPIPE inversion)"

# --------------------------------------------------------------------------
echo "== protect-env.sh =="
# --------------------------------------------------------------------------
run protect-env.sh '{"tool_name":"Read","tool_input":{"file_path":"'"$ROOT"'/.env"}}'
deny "Read .env"
run protect-env.sh '{"tool_name":"Write","tool_input":{"file_path":"'"$ROOT"'/.env","content":"X=1"}}'
deny "Write .env"
run protect-env.sh '{"tool_name":"Grep","tool_input":{"pattern":"=","path":"'"$ROOT"'/.env","output_mode":"content"}}'
deny "Grep(path=.env) — rg reads an explicitly-named ignored file"
run protect-env.sh '{"tool_name":"Read","tool_input":{"file_path":"'"$ROOT"'/.env.example"}}'
allow "Read .env.example"
run protect-env.sh '{"tool_name":"Grep","tool_input":{"pattern":"=","path":"'"$ROOT"'/.env.example"}}'
allow "Grep(path=.env.example)"
run protect-env.sh '{"tool_name":"Grep","tool_input":{"pattern":"x","path":"'"$ROOT"'/src"}}'
allow "Grep(path=src)"
run protect-env.sh '{"tool_name":"Read","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/agent.py"}}'
allow "Read a normal source file"

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
uv run pytest -q
CASES

run protect-env.sh "$(bash_payload "cat .env $BIG")"
deny "70 KB command mentioning .env still denied (no SIGPIPE inversion)"

# --------------------------------------------------------------------------
echo "== provider-env-drift.sh =="
# --------------------------------------------------------------------------
run provider-env-drift.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/src/travel_agent/tools/duffel.py"}}'
rc_is 0 "real repo passes clean (TAVILY_API_KEY correctly excused)"
run provider-env-drift.sh '{"tool_name":"Edit","tool_input":{"file_path":"'"$ROOT"'/README.md"}}'
rc_is 0 "non-matching file skipped"

# drift <tools-body> <conftest-body> -> exit code
drift() {
  local f; f=$(scratch)
  mkdir -p "$f/src/travel_agent/tools" "$f/tests"
  printf '%s\n' "$1" > "$f/src/travel_agent/tools/amadeus.py"
  printf '%s\n' "$2" > "$f/tests/conftest.py"
  printf '{"tool_name":"Edit","tool_input":{"file_path":"src/travel_agent/tools/amadeus.py"}}' \
    | CLAUDE_PROJECT_DIR="$f" bash "$H/provider-env-drift.sh" >/dev/null 2>&1
  echo $?
}
TUPLE='_PROVIDER_ENV = ("TRAVEL_AGENT_PROVIDER", "DUFFEL_API_TOKEN")'
ANNOT='_PROVIDER_ENV: tuple[str, ...] = ("TRAVEL_AGENT_PROVIDER", "DUFFEL_API_TOKEN")'

[ "$(drift 'import os
K = os.getenv("AMADEUS_API_KEY", "")' "$TUPLE")" = 2 ] \
  && ok 'os.getenv("X") detected' || bad 'os.getenv' 'not flagged'
[ "$(drift 'import os
K = os.environ["AMADEUS_API_KEY"]' "$TUPLE")" = 2 ] \
  && ok 'os.environ["X"] subscript detected' || bad 'subscript' 'not flagged'
[ "$(drift "import os
K = os.getenv('AMADEUS_API_KEY', '')" "$TUPLE")" = 2 ] \
  && ok "single-quoted name detected" || bad 'single quotes' 'not flagged'
[ "$(drift 'import os
K = os.getenv(
    "AMADEUS_API_KEY",
    "",
)' "$TUPLE")" = 2 ] \
  && ok 'call wrapped across lines detected' || bad 'wrapped call' 'not flagged'
[ "$(drift 'import os
K = os.getenv("DUFFEL_API_TOKEN", "")' "$ANNOT")" = 0 ] \
  && ok 'annotated tuple parses, not reported as gone' || bad 'annotated tuple' 'false alarm'
[ "$(drift 'import os
K = os.getenv("AMADEUS_API_KEY", "")' "$ANNOT")" = 2 ] \
  && ok 'annotated tuple still flags a real miss' || bad 'annotated tuple miss' 'not flagged'
[ "$(drift 'import os
K = os.getenv("TAVILY_API_KEY", "")' "$TUPLE")" = 0 ] \
  && ok 'TAVILY_API_KEY excused (search, not provider selection)' || bad 'excused list' 'wrongly flagged'
[ "$(drift 'import os
K = os.getenv("AMADEUS_API_KEY", "")' 'import pytest')" = 2 ] \
  && ok 'deleted _PROVIDER_ENV blocks' || bad 'deleted tuple' 'not flagged'
[ "$(drift 'import os
K = os.getenv("AMADEUS_API_KEY", "")' '# _PROVIDER_ENV removed')" = 2 ] \
  && ok 'unparseable tuple fails open to a whole-file scan, still blocks' \
  || bad 'unparseable tuple' 'wrong rc'

# --------------------------------------------------------------------------
echo "== honesty-cochange.sh =="
# --------------------------------------------------------------------------
run honesty-cochange.sh '{"stop_hook_active":false}'
allow "clean working tree allows stop"
run honesty-cochange.sh '{"stop_hook_active":true}'
allow "stop_hook_active short-circuits"

G=$(scratch)
(cd "$G" && git init -q && mkdir -p src/travel_agent/tools tests \
 && echo 'x = 1' > src/travel_agent/tools/availability.py \
 && echo 'y = 1' > tests/test_t.py \
 && git add -A && git -c user.email=t@t -c user.name=t commit -qm init \
 && echo 'x = 2' > src/travel_agent/tools/availability.py)
stop() { printf '{"stop_hook_active":false}' | CLAUDE_PROJECT_DIR="$G" bash "$H/honesty-cochange.sh" 2>/dev/null; }

OUT=$(stop); block "provider changed, tests untouched -> block"
# Keyed to a fingerprint of the provider diff, not stop_hook_active, which
# resets each turn and would re-block every turn for the rest of the session.
OUT=$(stop); RC=0; allow "same diff on a later turn -> no re-block"
(cd "$G" && echo 'x = 3' >> src/travel_agent/tools/availability.py)
OUT=$(stop); block "further provider change -> blocks again"
(cd "$G" && echo 'y = 2' > tests/test_t.py)
OUT=$(stop); RC=0; allow "tests touched -> allow"
(cd "$G" && git checkout -q -- . && echo 'z = 1' > src/travel_agent/prompts.py)
OUT=$(stop); RC=0; allow "unrelated src file -> allow"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" = 0 ]

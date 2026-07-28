#!/usr/bin/env bash
# PreToolUse (Read|Edit|Write|Grep|Bash) — keep the real .env out of reach.
#
# .env holds live ANTHROPIC_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY and
# DUFFEL_API_TOKEN. Two distinct risks:
#   reading  — copies real credentials into the transcript, which is then sent
#              to the API on every subsequent turn;
#   writing  — can flip TRAVEL_AGENT_PROVIDER to duffel or swap a duffel_test_
#              token for a live one, repointing the agent at real inventory
#              with nothing visible in the diff, since .env is gitignored.
#
# Grep is covered because ripgrep reads an explicitly-named ignored file even
# though a broad search skips it, so Grep(path=".env") would dump the whole
# file into the transcript with no hook in the way.
#
# The Bash check FAILS CLOSED. An allowlist of "reader" commands was tried
# first and is unwinnable: dd, nvim, perl -ne, busybox cat and every future
# tool walk straight through it. So any command touching .env as a path token
# is denied unless its verb is one of a short metadata-only set that cannot
# print file contents. The false positives that motivated the narrowing in the
# first place survive without depending on the verb set at all —
# `find . -name ".env*"` and `grep -rn "\.env" README.md` both fail the token
# boundary test, and `test -f .env` / `git commit -m "... .env ..."` are safe
# verbs. This stops accidents, not a determined bypass: `python -c` with a
# computed filename defeats any regex. It is a guardrail, not a sandbox.
#
# .env.example documents the full variable contract, so nothing legitimate
# needs the real file. To stop covering shell commands entirely, drop "Bash"
# from this hook's matcher in .claude/settings.json.
set -uo pipefail

input=$(cat)

reason="\
.env holds live ANTHROPIC_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY and DUFFEL_API_TOKEN.

Reading it copies real credentials into the transcript. Writing it can repoint the agent at live inventory invisibly, because .env is gitignored and the change never shows up in a diff.

Read .env.example instead — it documents every variable and what it does. If .env genuinely needs to change, ask the user to edit it themselves; they can run the command directly with a leading '!' in the prompt."

deny() {
  jq -n --arg r "$reason" '{hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: $r
  }}'
  exit 0
}

# Read / Edit / Write take file_path; Grep takes path. Compare the basename, so
# .env.example and .env.sample pass through untouched.
for field in file_path path; do
  target=$(jq -r --arg f "$field" '.tool_input[$f] // empty' <<<"$input" 2>/dev/null)
  [ -n "$target" ] || continue
  case "${target%/}" in
    */.env|*/.env.local|*/.env.*.local|.env|.env.local|.env.*.local) deny ;;
  esac
done

cmd=$(jq -r '.tool_input.command // empty' <<<"$input" 2>/dev/null)
[ -n "$cmd" ] || exit 0

# .env as a real path token. The trailing class excludes "." so .env.example
# and .environment never match; the leading class excludes "\" so a regex
# literal such as grep "\.env" does not either.
token='(^|[[:space:]"'\''=/`({])\.env([[:space:]"'\'';|&)`}<>]|$)'

# Commands that cannot print or alter file contents. Everything else is denied.
safe='test|\[|find|ls|stat|file|wc|touch|mkdir|rmdir|basename|dirname|realpath|git'

# Split on every construct that can start a fresh command, so `test -f .env &&
# cat .env` is judged on the segment that actually reads the file, not on the
# harmless verb that happens to come first. Redirection characters are split
# points too, which is what makes `echo K=v >> .env` land in a verb-less
# segment and get denied.
segments=$(printf '%s' "$cmd" | tr ';|&`(){}<>' '\n')

# Herestring, not a pipe: `printf ... | grep -q` returns 141 (SIGPIPE) under
# `set -o pipefail` once input exceeds the 64 KiB pipe buffer, which inverts
# the test and silently allows the command through. A herestring also keeps
# the loop in this shell, so `verdict` survives it.
verdict=allow
while IFS= read -r seg; do
  grep -qE "$token" <<<"$seg" || continue
  # Strip leading whitespace, VAR=val prefixes, then sudo, then take word one.
  verb=$(sed -e 's/^[[:space:]]*//' \
             -e 's/^\([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]][[:space:]]*\)*//' \
             -e 's/^sudo[[:space:]][[:space:]]*//' \
             -e 's/[[:space:]].*$//' <<<"$seg")
  if ! grep -qxE "$safe" <<<"$verb"; then
    verdict=deny
    break
  fi
done <<<"$segments"

[ "$verdict" = deny ] && deny
exit 0

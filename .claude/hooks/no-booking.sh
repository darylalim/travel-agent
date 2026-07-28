#!/usr/bin/env bash
# PreToolUse (Edit|Write) — refuse edits that add Duffel order or payment creation.
#
# CLAUDE.md: "Duffel is read-only — offer requests and reads, never POST
# /air/orders. Adding booking means putting order creation behind Deep Agents'
# interrupt_on human-approval gate; that is a deliberate design decision, not a
# config change." Creating an order spends real money and cannot be undone from
# here, so this is a hard deny rather than a reminder.
#
# The match targets URL CONSTRUCTION only, not any mention of the path. Two
# things must keep working:
#   - duffel.py's module docstring already contains the bare text
#     "POST /air/orders", and rewording it must not trip the hook;
#   - a regression test must be able to assert the endpoint is never called,
#     e.g. `assert "/air/orders" not in seen["url"]`.
# So a plain quoted "/air/orders" is allowed, and only the forms that actually
# build a request URL are denied:
#     f"{API_BASE}/air/orders"          -> "}" immediately before the path
#     "https://api.duffel.com/air/..."  -> full host
#     API_BASE + "/air/orders"          -> explicit concatenation
#     client.orders.create(...)         -> duffel-api SDK call
# Files under tests/ are exempt entirely: the suite is offline by construction
# (conftest's autouse fixture plus httpx.MockTransport), so no test can create a
# real order, and the hook must never block testing the invariant it protects.
set -uo pipefail

input=$(cat)

file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
case "$file" in
  */tests/*|tests/*) exit 0 ;;
esac

body=$(printf '%s' "$input" | jq -r '
  [.tool_input.content,
   .tool_input.new_string,
   (.tool_input.edits[]?.new_string)]
  | map(select(. != null)) | join("\n")' 2>/dev/null)
[ -n "$body" ] || exit 0

# Herestring, not a pipe: `printf ... | grep -q` returns 141 (SIGPIPE) under
# `set -o pipefail` once the input exceeds the 64 KiB pipe buffer, which
# inverts the test and silently allows a large payload through.
PATTERN='\}/air/(orders|payments)'
PATTERN="$PATTERN"'|duffel\.com/air/(orders|payments)'
PATTERN="$PATTERN"'|API_BASE[[:space:]]*\+[[:space:]]*["'\''][[:space:]]*/air/(orders|payments)'
PATTERN="$PATTERN"'|\.orders\.create\('

if grep -qE "$PATTERN" <<<"$body"; then
  jq -n '{hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: (
      "This edit builds a request URL for Duffel order or payment creation. The Duffel integration is deliberately read-only — it creates offer requests and reads offers back, and never POSTs to /air/orders (see CLAUDE.md, \"Provider seam\", and the duffel.py module docstring).\n\n" +
      "Order creation spends real money and is not reversible from the agent loop. It is only acceptable behind Deep Agents interrupt_on human-approval gate, which is an architectural decision for the user to make explicitly — not something to add as part of a provider edit.\n\n" +
      "Stop and ask the user. Note this hook allows a plain quoted \"/air/orders\" (so assertions and prose are fine) and exempts tests/ entirely; it only fires on URL construction such as f\"{API_BASE}/air/orders\"."
    )
  }}'
  exit 0
fi
exit 0

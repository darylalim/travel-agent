"""Duffel stays read-only: no booking, quote, payment or cancellation URL in src/.

CLAUDE.md, "Provider seam": booking spends real money and is not reversible from
the agent loop, so it is only acceptable behind Deep Agents' `interrupt_on`
human-approval gate — a deliberate architectural decision, not something that
arrives inside a provider edit.

This replaces `.claude/hooks/no-booking.sh`, which matched a regex against the
text of an edit and so could only recognise a path literal sitting immediately
after a `}`. Four spellings walked straight through it, including the most
idiomatic one in a file that already hoists `API_BASE`:

    path = "/air/orders"
    url = f"{API_BASE}{path}"          # allowed by the hook, denied here

An AST walk has no such blind spot. A request cannot reach an endpoint whose
path is not a string literal *somewhere*, and every literal is visited whatever
expression it is later assembled into. Two exemptions the hook needed nine test
cases to approximate are structural here instead: comments never enter the tree
at all, and docstrings are excluded by construction — so the accurate sentence
CLAUDE.md asks you to write into a tool docstring stops being a denial.

The verb is still not the discriminator: `/stays/search` is itself a POST and is
the one call the lodging path is built to make. Paths are. Extending cover to a
new provider — Amadeus, Booking.com — is a row in BOOKING_PATHS.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "travel_agent"

# Every Duffel path that creates, pays for, changes or cancels an order. Quotes
# are here because a quote creates server-side state on the way to a booking.
BOOKING_PATHS = (
    "/air/orders",
    "/air/payments",
    "/air/order_cancellations",
    "/air/order_change_requests",
    "/stays/bookings",
    "/stays/quotes",
    "/actions/cancel",
)


def _literal_strings(source: str) -> list[str]:
    """Every string literal in `source` except docstrings and bare string statements.

    A string standing alone as a statement is documentation whatever its
    position — module, class and function docstrings, and the attribute
    docstrings that follow an assignment. None of them can become a URL.
    """
    tree = ast.parse(source)
    documentation = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in documentation
    ]


def _booking_literals(source: str) -> list[tuple[str, str]]:
    """(literal, path) for every booking path appearing in executable string data."""
    return [
        (literal, path)
        for literal in _literal_strings(source)
        for path in BOOKING_PATHS
        if path in literal
    ]


def test_no_source_file_builds_a_booking_url():
    offences = [
        f"{path.relative_to(SRC.parent.parent)}: {literal!r} contains {booking!r}"
        for path in sorted(SRC.rglob("*.py"))
        for literal, booking in _booking_literals(path.read_text(encoding="utf-8"))
    ]
    assert not offences, (
        "Duffel is read-only. These string literals build a booking, quote, payment "
        "or cancellation URL:\n  " + "\n  ".join(offences) + "\n\n"
        "Booking is not reversible from the agent loop and belongs behind Deep Agents' "
        "interrupt_on human-approval gate — an architectural decision for the user to "
        "make explicitly. Prose and docstrings naming these paths are fine; only "
        "executable string data is checked."
    )


def test_the_scan_still_sees_what_it_should():
    """Guard the guard: a static check that quietly stops matching reads as green."""
    caught = "post(f'{API_BASE}/air/orders', json=payload)"
    hoisted = "path = '/air/orders'\nurl = f'{API_BASE}{path}'"
    concatenated = "url = API_BASE + '/stays/bookings'"
    full_host = "URL = 'https://api.duffel.com/air/orders'"
    for source in (caught, hoisted, concatenated, full_host):
        assert _booking_literals(source), f"scan went blind on: {source!r}"

    searching = "post(f'{API_BASE}/stays/search', json=payload)"
    assert not _booking_literals(searching), "searching must never be flagged"

    documented = '"""Never POST /air/orders — see CLAUDE.md."""\nx = 1'
    assert not _booking_literals(documented), "a docstring is not a booking call"

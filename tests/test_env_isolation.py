"""Every provider variable the tools read must be unset by conftest's fixture.

`conftest._PROVIDER_ENV` is the only thing keeping ambient shell configuration
out of the suite, and the combination the README tells you to export
(`TRAVEL_AGENT_PROVIDER=duffel` plus a token) turns the "no network" tests into
real requests to api.duffel.com. A name the tools read but the tuple omits is
therefore a live hole, and nothing about it is loud on the machine where it
matters.

This replaces `.claude/hooks/provider-env-drift.sh`, which checked the same
property but could only ever run on a developer's own edit. Two consequences:

- The property was invisible to CI. Emptying `_PROVIDER_ENV` stayed green on
  every runner by construction, because a fresh runner has nothing exported.
  As a test it fails here, on three Python versions, on any contributor's push.
- The hook read the tuple with `sed -n '/_PROVIDER_ENV[^=]*=/,/)/p'`, a range
  that stops at the first line containing `)`. A trailing comment on the
  multi-line tuple that ruff itself produces once a fourth name passes
  `line-length = 100` would have reported present names as missing. Importing
  the tuple cannot misread it, and deleting it raises ImportError rather than
  failing open.
"""

from __future__ import annotations

import ast
from pathlib import Path

from conftest import _PROVIDER_ENV

TOOLS = Path(__file__).resolve().parent.parent / "src" / "travel_agent" / "tools"

# TAVILY_API_KEY gates web search, not provider selection: leaving it set cannot
# send a request to a travel supplier, and unsetting it would change which tools
# the agent is built with. Everything else belongs in _PROVIDER_ENV.
EXCUSED = frozenset({"TAVILY_API_KEY"})


def _read_target(node: ast.AST) -> ast.expr | None:
    """The name expression of a direct environment read, or None."""
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "getenv" and isinstance(func.value, ast.Name):
                return node.args[0]
            if (
                func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
            ):
                return node.args[0]
    if isinstance(node, ast.Subscript):
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "environ":
            return node.slice
    return None


def _wrapper_names(tree: ast.Module) -> set[str]:
    """Functions that read the environment through one of their own parameters.

    duffel.py reaches os.getenv via `_env_int(name, default)`, so scanning for
    the direct idioms alone never sees DUFFEL_SUPPLIER_TIMEOUT_MS. Matching the
    shape rather than the name means a later `_env_str` or `_env_bool` is
    covered the day it is written, which the hook's hardcoded idiom list was not.
    """
    wrappers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {arg.arg for arg in node.args.args} | {arg.arg for arg in node.args.kwonlyargs}
        for inner in ast.walk(node):
            target = _read_target(inner)
            if isinstance(target, ast.Name) and target.id in params:
                wrappers.add(node.name)
                break
    return wrappers


def _first_string_argument(node: ast.AST, wrappers: set[str]) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in wrappers
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _trees() -> dict[Path, ast.Module]:
    return {
        path: ast.parse(path.read_text(encoding="utf-8")) for path in sorted(TOOLS.rglob("*.py"))
    }


def _names_read() -> set[str]:
    """Every environment variable name the tools read, directly or via a wrapper."""
    trees = _trees()
    wrappers: set[str] = set()
    for tree in trees.values():
        wrappers |= _wrapper_names(tree)

    names: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            target = _read_target(node)
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                names.add(target.value)
            wrapped = _first_string_argument(node, wrappers)
            if wrapped is not None:
                names.add(wrapped)
    return names


def test_every_env_var_the_tools_read_is_unset_by_the_fixture():
    missing = sorted(_names_read() - set(_PROVIDER_ENV) - EXCUSED)
    assert not missing, (
        f"Read under src/travel_agent/tools/ but absent from _PROVIDER_ENV: {missing}\n\n"
        "The autouse fixture in tests/conftest.py only unsets the names in that tuple, so "
        "anything missing lets an exported shell config reach the suite — and with "
        "TRAVEL_AGENT_PROVIDER=duffel plus a token exported, the offline tests start "
        "hitting api.duffel.com without failing loudly.\n\n"
        "Add the name to _PROVIDER_ENV, or to EXCUSED here with a reason."
    )


def test_the_scan_still_sees_the_reads_that_exist_today():
    """Guard the guard: a scan that quietly stops matching reads as green."""
    assert {"TRAVEL_AGENT_PROVIDER", "DUFFEL_API_TOKEN", "TAVILY_API_KEY"} <= _names_read()

    wrappers: set[str] = set()
    for tree in _trees().values():
        wrappers |= _wrapper_names(tree)
    assert "_env_int" in wrappers, "wrapper detection went blind on duffel.py's _env_int"
    assert "DUFFEL_SUPPLIER_TIMEOUT_MS" in _names_read(), "indirect read no longer seen"


def test_a_new_wrapper_shape_is_detected_without_naming_it():
    source = (
        "import os\n"
        "def _env_str(name: str, default: str) -> str:\n"
        "    return os.getenv(name, default)\n"
        "TOKEN = _env_str('AMADEUS_API_KEY', '')\n"
    )
    tree = ast.parse(source)
    wrappers = _wrapper_names(tree)
    assert wrappers == {"_env_str"}
    found = {
        name
        for node in ast.walk(tree)
        if (name := _first_string_argument(node, wrappers)) is not None
    }
    assert found == {"AMADEUS_API_KEY"}

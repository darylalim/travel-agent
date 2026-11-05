"""Trips and the traveler profile survive a restart of the CLI or the UI.

A "restart" here is a second `open_storage()` on a new agent, so nothing can
survive through a Python object. These drive the real agent with a scripted
model, because what has to round-trip is the state the graph actually writes.
No network, no model calls.
"""

from __future__ import annotations

import stat
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from pydantic import Field

import travel_agent.agent as agent_module
from travel_agent import clock, main, ui
from travel_agent.config import SUBAGENT_MODELS
from travel_agent.persistence import DB_ENV, database_path, open_storage
from travel_agent.prompts import MEMORY_PATH, WORKSPACE, resumed_message


class _ScriptedModel(GenericFakeChatModel):
    """Replays scripted replies and records the messages each call received."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):  # the replies already name their tools
        return self

    def _generate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return super()._generate(messages, *args, **kwargs)


def _replay(script: list[AIMessage]) -> Iterator[AIMessage]:
    while True:
        yield script.pop(0)


def _write(path: str, content: str) -> AIMessage:
    call = {"name": "write_file", "args": {"file_path": path, "content": content}, "id": "w1"}
    return AIMessage(content="", tool_calls=[call])


@pytest.fixture
def restarted(monkeypatch: pytest.MonkeyPatch):
    """Each call is a fresh process: new connections, new agent, same file."""

    def start(*replies: AIMessage) -> tuple[Any, _ScriptedModel]:
        # The fake's streaming path splits `content` into tokens, so a tool call
        # with no text yields nothing and raises under `stream_mode="messages"`.
        model = _ScriptedModel(messages=_replay(list(replies)), disable_streaming=True)
        monkeypatch.setattr(agent_module, "_chat_model", lambda *a, **k: model)
        checkpointer, store = open_storage()
        return agent_module.build_agent(checkpointer=checkpointer, store=store), model

    return start


def _turn(agent: Any, thread: str) -> dict[str, Any]:
    return agent.invoke(
        {"messages": [{"role": "user", "content": "plan a trip"}]},
        {"configurable": {"thread_id": thread}},
    )


def test_the_profile_survives_a_restart(restarted):
    agent, _ = restarted(_write(MEMORY_PATH, "Prefers aisle seats."), AIMessage(content="noted"))
    _turn(agent, "first")

    agent, model = restarted(AIMessage(content="ok"))
    _turn(agent, "second")
    assert "Prefers aisle seats." in model.calls[0][0].text


def test_a_trip_survives_a_restart_on_its_own_thread_only(restarted):
    brief = f"{WORKSPACE}brief.md"
    agent, _ = restarted(_write(brief, "Kyoto in April."), AIMessage(content="saved"))
    _turn(agent, "kyoto")

    agent, model = restarted(AIMessage(content="resumed"), AIMessage(content="new"))
    resumed = _turn(agent, "kyoto")
    assert brief in resumed["files"]
    assert any("saved" in message.text for message in model.calls[0])

    assert brief not in _turn(agent, "lisbon").get("files", {})


def test_a_thinking_block_comes_back_intact_after_a_restart(restarted, caplog):
    """A damaged block is a 400 on the next turn, and only on a resumed trip."""
    thinking = {"type": "thinking", "thinking": "Kyoto first.", "signature": "sig"}
    reply = AIMessage(content=[thinking, {"type": "text", "text": "saved"}])
    agent, _ = restarted(reply)
    _turn(agent, "kyoto")

    agent, model = restarted(AIMessage(content="resumed"))
    _turn(agent, "kyoto")
    replayed = [m for m in model.calls[0] if isinstance(m, AIMessage)]
    assert isinstance(replayed[0].content, list)
    assert thinking in replayed[0].content
    assert "unregistered" not in caplog.text.lower()


def test_a_delegated_turn_streams_the_way_the_ui_streams_it(restarted):
    """`subgraphs=True` writes nested checkpoints, which `invoke` never reaches."""
    delegate = {
        "name": "task",
        "args": {"description": "cost it", "subagent_type": "budget-analyst"},
        "id": "t1",
    }
    agent, model = restarted(
        AIMessage(content="", tool_calls=[delegate]),
        AIMessage(content="the analyst's answer"),
        AIMessage(content="done"),
    )
    items = list(
        agent.stream(
            {"messages": [{"role": "user", "content": "plan a trip"}]},
            {"configurable": {"thread_id": "kyoto"}},
            stream_mode=["messages", "updates"],
            subgraphs=True,
        )
    )
    assert any(namespace for namespace, *_ in items)
    assert len(model.calls) == 3

    agent, model = restarted(AIMessage(content="resumed"))
    _turn(agent, "kyoto")
    assert any("the analyst's answer" in message.text for message in model.calls[0])


def test_concurrent_turns_share_the_connections(restarted):
    """Streamlit serves each browser session on its own thread, over one agent."""
    agent, _ = restarted(*[AIMessage(content="ok") for _ in range(8)])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda n: _turn(agent, f"trip-{n}"), range(8)))
    assert all(result["messages"][-1].text == "ok" for result in results)


def test_the_store_is_usable_as_soon_as_it_is_opened():
    _, store = open_storage()
    assert store.get(("filesystem",), "/traveler_profile.md") is None


def test_opening_twice_keeps_what_the_first_wrote():
    _, store = open_storage()
    store.put(("filesystem",), "key", {"value": 1})
    _, again = open_storage()
    item = again.get(("filesystem",), "key")
    assert item is not None and item.value == {"value": 1}


def test_the_database_and_its_directory_are_created_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = tmp_path / "new" / "trips.sqlite"
    monkeypatch.setenv(DB_ENV, str(path))
    open_storage()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_setting_means_the_home_directory_default(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    monkeypatch.setenv(DB_ENV, value)
    assert database_path() == Path.home() / ".travel-agent" / "travel-agent.sqlite"


def test_a_configured_path_expands_the_home_directory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(DB_ENV, "~/trips.sqlite")
    assert database_path() == Path.home() / "trips.sqlite"


def test_the_suite_never_points_at_the_real_database(tmp_path: Path):
    assert database_path().parent == tmp_path


class _Spy:
    """Stands in for the agent and records how it was built and addressed."""

    def __init__(self) -> None:
        self.built: list[dict[str, Any]] = []
        self.threads: list[str] = []
        self.sent: list[str] = []
        self.history: list[Any] = []

    def build(self, *_args: Any, **kwargs: Any) -> _Spy:
        self.built.append(kwargs)
        return self

    def stream(self, input: Any, config: dict[str, Any], **_kwargs: Any) -> Iterator[Any]:
        self.threads.append(config["configurable"]["thread_id"])
        self.sent.append(input["messages"][0]["content"])
        return iter(())

    def get_state_history(self, _config: dict[str, Any]) -> Iterator[Any]:
        return iter(self.history)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    fake = _Spy()
    monkeypatch.setattr(agent_module, "build_agent", fake.build)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")
    return fake


def test_the_cli_saves_trips_and_the_profile(spy: _Spy):
    assert main.main(["plan a trip"]) == 0
    assert isinstance(spy.built[0]["checkpointer"], SqliteSaver)
    assert isinstance(spy.built[0]["store"], SqliteStore)


def test_the_ui_saves_trips_and_the_profile(spy: _Spy):
    ui.get_agent.clear()
    try:
        ui.get_agent()
    finally:
        ui.get_agent.clear()
    assert isinstance(spy.built[0]["checkpointer"], SqliteSaver)
    assert isinstance(spy.built[0]["store"], SqliteStore)


def _restart_the_ui(monkeypatch: pytest.MonkeyPatch, *replies: AIMessage) -> _ScriptedModel:
    """A new server process: nothing cached, same file."""
    ui.get_agent.clear()
    ui._storage.clear()
    model = _ScriptedModel(messages=_replay(list(replies)), disable_streaming=True)
    monkeypatch.setattr(agent_module, "_chat_model", lambda *a, **k: model)
    return model


def test_a_reopened_trip_continues_the_conversation_after_a_restart(
    monkeypatch: pytest.MonkeyPatch,
):
    _restart_the_ui(monkeypatch, _write(f"{WORKSPACE}/brief.md", "Kyoto"), AIMessage(content="ok"))
    record = ui.TripRecord()
    list(ui.stream_turn("kyoto", "plan Kyoto", record))
    assert f"{WORKSPACE}/brief.md" in record.files

    model = _restart_the_ui(monkeypatch, AIMessage(content="sure"))
    try:
        list(ui.stream_turn("kyoto", "make it six days", ui.TripRecord()))
    finally:
        ui.get_agent.clear()
    assert "plan Kyoto" in [message.text for message in model.calls[0]]


def test_deleting_a_trip_removes_its_checkpoints_and_keeps_the_profile(
    monkeypatch: pytest.MonkeyPatch,
):
    _restart_the_ui(
        monkeypatch,
        _write(MEMORY_PATH, "Prefers aisle seats."),
        AIMessage(content="ok"),
        AIMessage(content="ok"),
    )
    try:
        list(ui.stream_turn("kyoto", "plan Kyoto", ui.TripRecord()))
        list(ui.stream_turn("lisbon", "plan Lisbon", ui.TripRecord()))
    finally:
        ui.get_agent.clear()
    ui.save_trip("kyoto", [{"role": "user", "content": "plan Kyoto"}], ui.TripRecord(), [])

    ui.delete_trip("kyoto")

    checkpointer, store = ui._storage()
    assert ui.saved_trips() == []
    assert checkpointer.get_tuple({"configurable": {"thread_id": "kyoto"}}) is None
    assert checkpointer.get_tuple({"configurable": {"thread_id": "lisbon"}}) is not None
    assert store.get(("filesystem",), "/traveler_profile.md") is not None


def test_each_cli_run_is_a_new_trip_unless_it_names_one(
    spy: _Spy, capsys: pytest.CaptureFixture[str]
):
    """A fixed default would quietly resume the previous conversation."""
    main.main(["plan a trip"])
    main.main(["plan a trip"])
    assert spy.threads[0] != spy.threads[1]
    assert f"--thread {spy.threads[1]}" in capsys.readouterr().out

    main.main(["plan a trip", "--thread", spy.threads[0]])
    assert spy.threads[2] == spy.threads[0]


def _snapshot(created_at: str, *messages: BaseMessage) -> Any:
    return SimpleNamespace(created_at=created_at, values={"messages": list(messages)})


def _delegated(subagent: str, call_id: str = "t1") -> list[BaseMessage]:
    call = {"name": "task", "args": {"subagent_type": subagent}, "id": call_id}
    return [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="found", name="task", tool_call_id=call_id),
    ]


def test_a_new_thread_has_no_prices_to_date():
    assert main.prices_dated_on([]) is None


def test_a_turn_with_no_search_does_not_make_old_fares_look_newer():
    """Newest first, as `get_state_history` returns them."""
    searched = _delegated("availability-scout")
    history = [
        _snapshot("2026-11-20T12:00:00+00:00", *searched, HumanMessage(content="and the hotel?")),
        _snapshot("2026-09-14T12:00:00+00:00", *searched),
        _snapshot("2026-09-13T12:00:00+00:00", searched[0]),
    ]
    assert main.prices_dated_on(history) == "Sep 14, 2026"


def test_a_later_search_moves_the_date():
    first = _delegated("availability-scout", "t1")
    second = _delegated("availability-scout", "t2")
    history = [
        _snapshot("2026-10-02T12:00:00+00:00", *first, *second),
        _snapshot("2026-09-14T12:00:00+00:00", *first),
    ]
    assert main.prices_dated_on(history) == "Oct 2, 2026"


def test_a_scout_still_running_has_not_priced_anything():
    """The call alone is not a result, so it cannot date one."""
    call = _delegated("availability-scout")[0]
    history = [
        _snapshot("2026-11-20T12:00:00+00:00", call),
        _snapshot("2026-09-14T12:00:00+00:00", call),
    ]
    assert main.prices_dated_on(history) == "Nov 20, 2026"


def test_a_trip_with_no_search_falls_back_to_its_last_checkpoint():
    history = [
        _snapshot("2026-11-20T12:00:00+00:00", *_delegated("destination-researcher")),
        _snapshot("2026-09-14T12:00:00+00:00"),
    ]
    assert main.prices_dated_on(history) == "Nov 20, 2026"


def test_a_checkpoint_is_dated_in_local_time(monkeypatch: pytest.MonkeyPatch):
    """Checkpoints stamp UTC, which is already tomorrow on an evening in Honolulu."""
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    time.tzset()
    try:
        dated = main.prices_dated_on([_snapshot("2026-09-15T02:00:00+00:00")])
    finally:
        monkeypatch.undo()
        time.tzset()
    assert dated == "Sep 14, 2026"


def test_the_scout_name_is_on_the_roster():
    """A stale name finds no search and quietly dates every trip by its last turn."""
    assert main.SCOUT in SUBAGENT_MODELS


def test_only_the_first_turn_of_a_resumed_cli_trip_carries_the_note(
    spy: _Spy, monkeypatch: pytest.MonkeyPatch
):
    spy.history = [_snapshot("2026-09-14T12:00:00+00:00")]
    replies = iter(["And the hotel?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))

    main.main(["Is it still fine?", "--thread", "kyoto"])

    assert spy.sent == [resumed_message("Is it still fine?", "Sep 14, 2026"), "And the hotel?"]


def test_the_note_waits_for_the_first_message_of_an_interactive_resume(
    spy: _Spy, monkeypatch: pytest.MonkeyPatch
):
    """No opening prompt, and a blank line is not a turn."""
    spy.history = [_snapshot("2026-09-14T12:00:00+00:00")]
    replies = iter(["", "Is it still fine?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))

    main.main(["--thread", "kyoto"])

    assert spy.sent == [resumed_message("Is it still fine?", "Sep 14, 2026")]


def test_a_new_cli_trip_carries_no_note(spy: _Spy):
    spy.history = [_snapshot("2026-09-14T12:00:00+00:00")]  # never read without --thread
    main.main(["plan a trip"])
    assert spy.sent == ["plan a trip"]


def test_a_thread_id_with_no_history_carries_no_note(spy: _Spy):
    main.main(["plan a trip", "--thread", "new"])
    assert spy.sent == ["plan a trip"]


def test_a_real_resume_reaches_the_agent_with_the_date_of_its_checkpoints(
    restarted, monkeypatch: pytest.MonkeyPatch
):
    """The fakes above stand in for the history, so this pins its real shape."""
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")
    delegate = {
        "name": "task",
        "args": {"description": "find fares", "subagent_type": main.SCOUT},
        "id": "t1",
    }
    agent, _ = restarted(
        AIMessage(content="", tool_calls=[delegate]),
        AIMessage(content="a fare"),
        AIMessage(content="here is the plan"),
    )
    _turn(agent, "kyoto")

    _, model = restarted(AIMessage(content="searching again"))
    assert main.main(["Is it still fine?", "--thread", "kyoto"]) == 0

    today = datetime.now().astimezone().date()
    assert model.calls[0][-1].text == resumed_message("Is it still fine?", clock.long_date(today))

"""Plan page: converse with the travel agent.

Element order inside an assistant bubble is fixed — activity, then any data
caveat, then the prose, then any failure notice — and the replay loop below
reproduces it exactly. If the live run and the replay disagree, the message
visibly rearranges itself on the next interaction.

The title lives in `streamlit_app.py`; pages do not set their own.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import streamlit as st

from travel_agent.ui import stream_turn

SUGGESTIONS = {
    "5 days in Kyoto": "5 days in Kyoto in September, 2 people, $4000 total, flying from SFO.",
    "A week in Lisbon": "A week in Lisbon in October for 2, around $3000, relaxed pace.",
    "Long weekend, no flights": "A long weekend somewhere within 3 hours of Berlin by train.",
}

record = st.session_state.record
thread_id = st.session_state.thread_id


def _render_activity(entry: dict[str, Any]) -> None:
    """Reproduce the status element the live run showed for this turn.

    The live run always creates one, so the replay must too — including a turn
    that called no tools and a turn that failed. Deriving the label and state
    from `len(activity)` instead replays a failure as a green "2 steps" and
    drops the element entirely when nothing was delegated.
    """
    labels = entry.get("activity", [])
    with st.status(
        entry.get("status_label", f"{len(labels)} steps"),
        state=entry.get("status_state", "complete"),
        type="compact",
    ):
        for label in labels:
            st.write(label)


# Replay first, so the new exchange below is drawn exactly once this run.
for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        if entry["role"] != "assistant":
            st.markdown(entry["content"])
            continue
        _render_activity(entry)
        for caveat in entry.get("caveats", []):
            st.warning(caveat, icon=":material/warning:")
        st.markdown(entry["content"])
        if entry.get("error"):
            st.error(entry["error"], icon=":material/error:")

# Suggestions stand in for a first prompt and disappear once the chat starts.
suggested = None
if not st.session_state.messages:
    st.caption("Describe a trip, or start from one of these.")
    choice = st.pills("Suggestions", list(SUGGESTIONS), label_visibility="collapsed")
    if choice:
        suggested = SUGGESTIONS[choice]

typed = st.chat_input(
    "Where are you going, when, how many people, and what is the budget?",
    submit_mode="disable",
)
prompt = typed or suggested

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    slots: dict[str, Any] = {}
    labels: dict[str, str] = {}
    streamed: list[str] = []
    fresh: list[str] = []
    # These describe an interrupted turn, and stay that way if the block below
    # unwinds without reaching either branch — which is what Streamlit's stop
    # button and a mid-stream page switch do. Both raise through `BaseException`
    # rather than `Exception`, so the `finally` is the only place that sees them.
    failure: str | None = "The agent was interrupted before it finished."
    status_label, status_state = "Interrupted", "error"

    try:
        with st.chat_message("assistant"):
            # Both created before the stream so they keep their slot above the
            # prose even though they are filled during and after it.
            status = st.status("Working…", state="running", type="compact")
            caveat_box = st.container()

            def on_activity(call_id: str, label: str) -> None:
                """Rewrite one line per tool call.

                A `task` call's arguments stream in as partial JSON, so its
                label sharpens from "Delegating to a subagent" to the actual
                subagent name. Rewriting a held slot keeps one line; appending
                would stack every intermediate guess.
                """
                if call_id not in slots:
                    slots[call_id] = status.empty()
                slots[call_id].write(label)
                labels[call_id] = label

            def tee(chunks: Iterator[str]) -> Iterator[str]:
                """Keep a copy of the prose as it streams.

                `st.write_stream` returns the joined text only when it
                completes. If the run raises or is interrupted it has already
                painted whatever arrived first, and that text would be lost —
                so the replayed bubble would show less than the live one did.
                Collecting here keeps the two equal, and the joined chunks
                match what `st.write_stream` would have returned.
                """
                for chunk in chunks:
                    streamed.append(chunk)
                    yield chunk

            try:
                # Only `str` is ever yielded, so this streams as markdown
                # rather than falling through to a raw `st.write`.
                st.write_stream(tee(stream_turn(thread_id, prompt, record, on_activity)))
            except Exception as exc:  # noqa: BLE001 - surfaced to the traveler, not swallowed
                failure = f"The agent stopped: {exc}"
                status_label, status_state = "Could not finish", "error"
            else:
                failure = None
                status_label, status_state = f"{len(labels)} steps", "complete"
            status.update(label=status_label, state=status_state)

            if failure is not None:
                st.error(failure, icon=":material/error:")

            # Caveats are shown once, when first raised; the Trip page keeps
            # the standing list so a warning is never only visible in
            # scrollback.
            fresh = [c for c in record.caveats() if c not in st.session_state.seen_caveats]
            st.session_state.seen_caveats.extend(fresh)
            for caveat in fresh:
                caveat_box.warning(caveat, icon=":material/warning:")
    finally:
        # In a `finally` so an interrupted turn is recorded too. The traveler's
        # message went into history before the run started, so bailing out here
        # would replay a question with no answer beneath it — and the status and
        # error above are drawn for this run only.
        #
        # This cannot double-append: `st.chat_input` returns None on the rerun
        # that follows, and the suggestion pills are gone once history is
        # non-empty, so `prompt` is falsy and this block is not reached again.
        prose = "".join(streamed)
        if prose:
            content = prose
        elif failure is not None:
            content = "_The agent stopped before writing anything._"
        else:
            content = "_The agent returned no text for that turn._"

        # No further write of the prose — st.write_stream already replaced its
        # own placeholder with the finished markdown, so rendering it again
        # duplicates the message.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "activity": list(labels.values()),
                "caveats": fresh,
                "error": failure,
                # Stored rather than re-derived: `len(activity)` cannot tell a
                # failed turn from a complete one, and the replay has to match
                # the live bubble element for element.
                "status_label": status_label,
                "status_state": status_state,
            }
        )

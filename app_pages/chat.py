"""Plan page: converse with the travel agent.

Element order inside an assistant bubble is fixed — activity, then any data
caveat, then the prose — and the replay loop below reproduces it exactly. If
the live run and the replay disagree, the message visibly rearranges itself
on the next interaction.

The title lives in `streamlit_app.py`; pages do not set their own.
"""

from __future__ import annotations

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


def _render_activity(labels: list[str]) -> None:
    """Collapsed summary of what the agent did for one turn."""
    if not labels:
        return
    with st.status(f"{len(labels)} steps", state="complete", type="compact"):
        for label in labels:
            st.write(label)


# Replay first, so the new exchange below is drawn exactly once this run.
for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        if entry["role"] == "assistant":
            _render_activity(entry.get("activity", []))
            for caveat in entry.get("caveats", []):
                st.warning(caveat, icon=":material/warning:")
        st.markdown(entry["content"])

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

    with st.chat_message("assistant"):
        # Both created before the stream so they keep their slot above the
        # prose even though they are filled during and after it.
        status = st.status("Working…", state="running", type="compact")
        caveat_box = st.container()

        slots: dict[str, Any] = {}
        labels: dict[str, str] = {}

        def on_activity(call_id: str, label: str) -> None:
            """Rewrite one line per tool call.

            A `task` call's arguments stream in as partial JSON, so its label
            sharpens from "Delegating to a subagent" to the actual subagent
            name. Rewriting a held slot keeps one line; appending would stack
            every intermediate guess.
            """
            if call_id not in slots:
                slots[call_id] = status.empty()
            slots[call_id].write(label)
            labels[call_id] = label

        failure: str | None = None
        try:
            # Only `str` is ever yielded, so this returns a plain string.
            reply = st.write_stream(stream_turn(thread_id, prompt, record, on_activity))
        except Exception as exc:  # noqa: BLE001 - surfaced to the traveler, not swallowed
            # Deliberately not BaseException: Streamlit's stop/rerun control
            # flow subclasses it, and catching that would break the stop button.
            reply = ""
            failure = str(exc)
            status.update(label="Could not finish", state="error")
        else:
            status.update(label=f"{len(labels)} steps", state="complete")

        if failure is not None:
            st.error(f"The agent stopped: {failure}", icon=":material/error:")

        # Caveats are shown once, when first raised; the Trip page keeps the
        # standing list so a warning is never only visible in scrollback.
        fresh = [c for c in record.caveats() if c not in st.session_state.seen_caveats]
        st.session_state.seen_caveats.extend(fresh)
        for caveat in fresh:
            caveat_box.warning(caveat, icon=":material/warning:")

    # No further write of `reply` — st.write_stream already replaced its own
    # placeholder with the finished markdown, so rendering again duplicates it.
    if reply or failure is None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply or "_The agent returned no text for that turn._",
                "activity": list(labels.values()),
                "caveats": fresh,
            }
        )

"""Past trips page: reopen a trip saved by an earlier session.

Opening one restores the transcript and the Trip page from its snapshot, and
the agent picks the conversation up from its checkpoint. `ui.save_trip` says why
it takes both.

The title lives in `streamlit_app.py`; pages do not set their own.
"""

from __future__ import annotations

import streamlit as st

from travel_agent.ui import MAX_LISTED_TRIPS, delete_trip, saved_trips

# Inline rather than `st.dialog`, which `AppTest` cannot see, and deleting is
# the one step on this page that cannot be taken back.
_CONFIRMING = "_confirming_delete"

trips = saved_trips()

if not trips:
    st.info(
        "No saved trips yet. Each trip is saved here after every reply on the Plan page.",
        icon=":material/tips_and_updates:",
    )
    st.stop()

for trip in trips:
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        with st.container(gap="xxsmall"):
            # `st.text` rather than markdown: the title is the traveler's own
            # words, and this is the one render that needs no escaping at all.
            st.text(trip.title)
            st.caption(trip.summary)
        # The open trip offers no delete: its next turn would save it again.
        if trip.thread_id == st.session_state.thread_id:
            st.badge("Open now", color="gray")
        elif st.session_state.get(_CONFIRMING) == trip.thread_id:
            st.caption("Delete for good?")
            if st.button("Delete", key=f"delete-{trip.thread_id}", width="content"):
                delete_trip(trip.thread_id)
                del st.session_state[_CONFIRMING]
                st.rerun()
            if st.button("Keep", key=f"keep-{trip.thread_id}", width="content"):
                del st.session_state[_CONFIRMING]
                st.rerun()
        else:
            if st.button("Open", key=f"open-{trip.thread_id}", width="content"):
                st.session_state.thread_id = trip.thread_id
                st.session_state.messages = trip.messages
                st.session_state.record = trip.record
                st.session_state.seen_caveats = trip.seen_caveats
                st.session_state.reopened_on = trip.saved_on
                st.session_state.resume_note_on = trip.saved_on
                st.session_state.pop(_CONFIRMING, None)
                st.switch_page("app_pages/chat.py")
            if st.button(
                "Delete",
                key=f"ask-delete-{trip.thread_id}",
                icon=":material/delete:",
                width="content",
            ):
                st.session_state[_CONFIRMING] = trip.thread_id
                st.rerun()

if len(trips) == MAX_LISTED_TRIPS:
    st.caption(f"Showing the {MAX_LISTED_TRIPS} most recent trips.")

"""Past trips page: reopen a trip saved by an earlier session.

Opening one restores the transcript and the Trip page from its snapshot, and
the agent picks the conversation up from its checkpoint. `ui.save_trip` says why
it takes both.

The title lives in `streamlit_app.py`; pages do not set their own.
"""

from __future__ import annotations

import streamlit as st

from travel_agent.ui import MAX_LISTED_TRIPS, saved_trips

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
        if trip.thread_id == st.session_state.thread_id:
            st.badge("Open now", color="gray")
        elif st.button("Open", key=f"open-{trip.thread_id}", width="content"):
            st.session_state.thread_id = trip.thread_id
            st.session_state.messages = trip.messages
            st.session_state.record = trip.record
            st.session_state.seen_caveats = trip.seen_caveats
            st.session_state.reopened_on = trip.saved_on
            st.switch_page("app_pages/chat.py")

if len(trips) == MAX_LISTED_TRIPS:
    st.caption(f"Showing the {MAX_LISTED_TRIPS} most recent trips.")

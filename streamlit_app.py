"""Streamlit entry point: `uv run streamlit run streamlit_app.py`.

A third way to drive the agent, alongside `langgraph dev` and the CLI. It shares
the CLI's SQLite file, so the traveler profile survives a server restart. Trips
do not: the transcript lives in session state and the checkpoints in memory, so
a past trip cannot be reopened here.

`load_dotenv()` runs before anything that reaches `travel_agent.agent`,
because that module builds its graph at import time and reads `TAVILY_API_KEY`
while doing so. `travel_agent.ui` is safe to import here — it defers the agent
import into a cached function for exactly this reason.
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Safe to import here: `travel_agent.ui` defers the agent import into a cached
# factory, so nothing builds a graph before `load_dotenv()` has run.
from travel_agent.ui import PLAN_WIDTH, TRIP_WIDTH, TripRecord

st.set_page_config(
    page_title="Travel agent",
    page_icon=":material/travel_explore:",
    layout="wide",
)

# Shared state is initialised here rather than in a page, so either page can
# be the first one the traveler lands on.
#
# One thread id per trip: it scopes the checkpointer, so starting a new trip
# clears /trip/* while leaving the traveler profile under /memories/ intact.
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("record", TripRecord())
st.session_state.setdefault("seen_caveats", [])

if not os.getenv("ANTHROPIC_API_KEY"):
    st.title(":material/travel_explore: Travel agent")
    st.error(
        "`ANTHROPIC_API_KEY` is not set, so the agent cannot run. "
        "Copy `.env.example` to `.env` and fill it in, then reload this page.",
        icon=":material/key_off:",
    )
    st.stop()

pages = [
    st.Page("app_pages/chat.py", title="Plan", icon=":material/forum:", default=True),
    st.Page("app_pages/trip.py", title="Trip", icon=":material/map:"),
]
page = st.navigation(pages, position="top")

with (
    st.container(horizontal_alignment="center"),
    st.container(width=PLAN_WIDTH if page.title == "Plan" else TRIP_WIDTH),
):
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        # Grey on purpose, and never green. A colour reading as "live" would be
        # a claim about the data, which the provider name cannot support: the
        # same provider called `duffel` returns fictional fares and sample
        # lodging on a test token, and real inventory on a live one. Provenance
        # is stated per offer on the Trip page, off each offer's own
        # `synthetic` flag.
        st.badge(
            os.getenv("TRAVEL_AGENT_PROVIDER", "sample-data"),
            icon=":material/inventory_2:",
            color="gray",
            help="Availability provider. Every offer is labelled individually on the Trip page.",
        )
        if not os.getenv("TAVILY_API_KEY"):
            st.badge(
                "No web research",
                icon=":material/travel_explore:",
                color="gray",
                help="`TAVILY_API_KEY` is not set, so the agent cannot research destinations.",
            )
        st.space("stretch")
        if st.button(
            "Start a new trip",
            icon=":material/restart_alt:",
            width="content",
            help="Clears this trip's workspace. The traveler profile is kept.",
        ):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.record = TripRecord()
            st.session_state.seen_caveats = []
            st.rerun()

    # Pages do not set their own title; the entry script owns it.
    st.title(f"{page.icon} {page.title}")
    page.run()

"""
Streamlit UI for VEHS Hazard Q&A (LangChain + LangGraph)

Run:
  streamlit run streamlit_app.py
"""
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, Any, List

# Ensure OpenMP duplicate runtime doesn't crash FAISS on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import pandas as pd

# Import the compiled LangGraph app from bot.py
from bot import app as graph_app, is_hazard_query

st.set_page_config(page_title="EPCL Data Analyst", layout="wide")

# --- Light, polished styling for Engro Chemicals ---
st.markdown(
    """
    <style>
      .app-header { display:flex; align-items:center; gap: 14px; padding: 8px 0 14px; border-bottom: 1px solid #e7efe9; }
      .app-title { font-size: 1.6rem; font-weight: 700; color: #0B1A12; margin: 0; }
      .app-subtitle { color: #2f6f45; margin: 0; font-size: 0.95rem; }
      .muted { color: #587a66; }
      .card { background: #FFFFFF; border: 1px solid #e7efe9; border-radius: 12px; padding: 14px 16px; margin-top: 10px; }
      .section-title { margin: 6px 0 6px; font-weight: 700; color: #0B1A12; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with optional Engro logo
LOGO_PATH = "svglogo.svg"
header_cols = st.columns([1, 9])
logo_file = Path(LOGO_PATH)
if logo_file.exists():
    header_cols[0].image(LOGO_PATH, caption="Engro Chemicals", width=64)
else:
    header_cols[0].markdown(":seedling:")
header_cols[1].markdown("""
<div class="app-header">
  <div>
    <p class="app-title">EPCL Data Analyst</p>
    <p class="app-subtitle">Insights across incidents, hazards, audits, inspections, and reports</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Session-scoped thread for checkpointer/memory continuity
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ui-{uuid.uuid4()}"
if "qna_log" not in st.session_state:
    st.session_state.qna_log = []  # list of {query, answer, chunks: [{source,id,score,preview}]}

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    location = st.text_input("Location", value="")
    department = st.text_input("Department", value="")
    col1, col2 = st.columns(2)
    with col1:
        start_dt = st.date_input("Start date", value=None)
    with col2:
        end_dt = st.date_input("End date", value=None)

    st.markdown("---")
    # Generic service key presence (no vendor naming)
    if os.environ.get("OPENAI_API_KEY"):
        st.success("Service key detected")
    else:
        st.warning("Service key not set")
    if st.button("Clear History"):
        st.session_state.qna_log = []
        # Reset conversational context for the backend as well
        st.session_state.thread_id = f"ui-{uuid.uuid4()}"
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"]
        st.rerun()

# Main input
default_q = (
    "What are the most concerned hazards and what steps should we take to avoid it turning into an incident?"
)
question = st.text_area("Question", value=default_q, height=100)

ask = st.button("Ask", type="primary")

if ask:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Please set it in your environment and restart.")
        st.stop()

    # Build filters dict compactly
    filters: Dict[str, Any] = {}
    if location:
        filters["location"] = location
    if department:
        filters["department"] = department
    if isinstance(start_dt, date):
        filters["start_date"] = start_dt.isoformat()
    if isinstance(end_dt, date):
        filters["end_date"] = end_dt.isoformat()

    state = {
        "query": question,
        "filters": filters,
        "retrieved": [],
        "analytics": {},
        "answer": "",
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    ai_ok = False
    error_msg = None
    with st.spinner("Thinking..."):
        try:
            final = graph_app.invoke(state, config=config)
            ai_ok = True
        except Exception as e:
            final = {}
            ai_ok = False
            error_msg = str(e)

    answer = final.get("answer", "")
    analytics = final.get("analytics", {}) or {}
    top: List[Dict[str, Any]] = analytics.get("top", []) if isinstance(analytics, dict) else []

    # Run status block (generic labels, no vendor names)
    st.subheader("Run Status")
    cols = st.columns(3)
    with cols[0]:
        if os.environ.get("OPENAI_API_KEY"):
            st.success("Service Key: Present")
        else:
            st.error("Service Key: Missing")
    with cols[1]:
        if ai_ok:
            st.success("AI Response: Generated")
        else:
            st.error("AI Response: Failed")
            if error_msg:
                st.caption(error_msg[:300])
    with cols[2]:
        st.info(f"Retrieved Chunks: {len(final.get('retrieved', []))}")

    # Build rows for retrieved chunks for logging
    retrieved = final.get("retrieved", [])
    rows = []
    for d in retrieved[:12]:
        meta = getattr(d, "metadata", {}) or {}
        cid = meta.get("record_id") or ""
        src = meta.get("source_sheet") or ""
        score = meta.get("score")
        content = (getattr(d, "page_content", "") or "")[:220].replace("\n", " ")
        rows.append({
            "source": src,
            "id": cid,
            "score": score,
            "preview": content,
        })

    # Append to history log (we will render the latest section below)
    st.session_state.qna_log.append({
        "query": question,
        "answer": answer,
        "chunks": rows,
    })

# --- Render Latest result and Follow-up input (outside Ask block) ---
if st.session_state.qna_log:
    latest = st.session_state.qna_log[-1]
    st.subheader("Answer")
    if latest.get("context_included"):
        st.caption("Context included")
    st.markdown(latest.get("answer", ""))

    latest_rows = latest.get("chunks") or []
    if latest_rows:
        with st.expander("Retrieved chunks"):
            st.dataframe(pd.DataFrame(latest_rows), use_container_width=True)
            st.markdown("---")
            st.markdown("Snippets:")
            for r in latest_rows:
                st.markdown(f"- **[{r['source']}:{r['id']}]** (score={r['score']}) {r['preview']}")

    st.markdown("---")
    st.subheader("Ask a follow-up")
    followup = st.text_area("", value="", height=80, key="followup_input")
    ask_followup = st.button("Ask follow-up", type="primary", key="ask_followup_btn")

    if ask_followup and followup.strip():
        # Build filters dict compactly (reuse sidebar values)
        filters: Dict[str, Any] = {}
        if location:
            filters["location"] = location
        if department:
            filters["department"] = department
        if isinstance(start_dt, date):
            filters["start_date"] = start_dt.isoformat()
        if isinstance(end_dt, date):
            filters["end_date"] = end_dt.isoformat()

        # Compose contextual query with last few turns
        context_turns = st.session_state.qna_log[-3:]
        context_block = "\n\n".join([
            f"Q: {t.get('query','')}\nA: {t.get('answer','')}" for t in context_turns
        ])
        composite_query = f"Context:\n{context_block}\n\nFollow-up question: {followup}"

        state = {
            "query": composite_query,
            "filters": filters,
            "retrieved": [],
            "analytics": {},
            "answer": "",
        }
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        ai_ok = False
        error_msg = None
        with st.spinner("Thinking..."):
            try:
                final = graph_app.invoke(state, config=config)
                ai_ok = True
            except Exception as e:
                final = {}
                ai_ok = False
                error_msg = str(e)

        # Extract follow-up results before rendering
        answer = final.get("answer", "")
        retrieved = final.get("retrieved", [])
        rows = []
        for d in retrieved[:12]:
            meta = getattr(d, "metadata", {}) or {}
            cid = meta.get("record_id") or ""
            src = meta.get("source_sheet") or ""
            score = meta.get("score")
            content = (getattr(d, "page_content", "") or "")[:220].replace("\n", " ")
            rows.append({
                "source": src,
                "id": cid,
                "score": score,
                "preview": content,
            })

        # Show run status for follow-up as well
        st.subheader("Run Status")
        cols = st.columns(3)
        with cols[0]:
            if os.environ.get("OPENAI_API_KEY"):
                st.success("Service Key: Present")
            else:
                st.error("Service Key: Missing")
        with cols[1]:
            if ai_ok:
                st.success("AI Response: Generated")
            else:
                st.error("AI Response: Failed")
                if error_msg:
                    st.caption(error_msg[:300])
        with cols[2]:
            st.info(f"Retrieved Chunks: {len(final.get('retrieved', []))}")

        # Inline display of follow-up answer so users see it immediately here
        st.markdown("---")
        st.subheader("Answer")
        st.caption("Context included")
        st.markdown(answer or "")
        if rows:
            with st.expander("Retrieved chunks"):
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                st.markdown("---")
                st.markdown("Snippets:")
                for r in rows:
                    st.markdown(f"- **[{r['source']}:{r['id']}]** (score={r['score']}) {r['preview']}")

        st.session_state.qna_log.append({
            "query": followup,
            "answer": answer,
            "chunks": rows,
            "context_included": True,
        })
        # No explicit rerun here so the inline answer stays in view

    # History (older entries only)
    if len(st.session_state.qna_log) > 1:
        st.markdown("---")
        st.subheader("History")
        for i, entry in enumerate(reversed(st.session_state.qna_log[:-1]), start=1):
            st.markdown(f"**{i}. Q:** {entry.get('query','')}")
            with st.expander("Answer"):
                st.markdown(entry.get("answer", ""))
            chunks = entry.get("chunks") or []
            if chunks:
                with st.expander("Chunks"):
                    st.dataframe(pd.DataFrame(chunks), use_container_width=True)
                    st.markdown("Snippets:")
                    for r in chunks[:10]:
                        st.markdown(f"- **[{r['source']}:{r['id']}]** (score={r['score']}) {r['preview']}")

# Footer (generic)
st.markdown("\n\n— EPCL Data Analyst")

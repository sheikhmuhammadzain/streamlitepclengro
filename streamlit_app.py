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
try:
    from PIL import Image
except Exception:
    Image = None

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
      /* Sidebar inputs - lighter borders */
      [data-testid="stSidebar"] .stTextInput input,
      [data-testid="stSidebar"] .stDateInput input,
      [data-testid="stSidebar"] .stTextArea textarea {
        border: 1px solid #e5ece8 !important;
        background: #fbfdfb !important;
        border-radius: 10px !important;
      }
      /* Chat avatars: consistent size (3rem), transparent background, don't crop */
      [data-testid="stChatMessageAvatar"] {
        width: 3rem !important;
        height: 3rem !important;
        display: flex; align-items: center; justify-content: center;
      }
      [data-testid="stChatMessageAvatar"] img {
        width: 3rem !important;
        height: 3rem !important;
        object-fit: contain !important;
        background: transparent !important;
      }
      /* Stronger override: target avatar and its direct children inside chat message */
      [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"],
      [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] > img,
      [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] > span,
      [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] > div {
        width: 3rem !important;
        height: 3rem !important;
        min-width: 3rem !important;
        min-height: 3rem !important;
        max-width: 3rem !important;
        max-height: 3rem !important;
        object-fit: contain !important;
        flex-shrink: 0 !important;
      }
      /* Generic fallback: any image inside a chat message */
      [data-testid="stChatMessage"] img {
        width: 3rem !important;
        height: 3rem !important;
        object-fit: contain !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with optional Engro logo
LOGO_PATH = "svglogo.svg"
header_cols = st.columns([1, 9])
logo_file = Path(LOGO_PATH)
if logo_file.exists():
    header_cols[0].image(LOGO_PATH, caption="Engro Chemicals", width=65)
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

# Choose the assistant avatar (prefer SVG logo) and keep it transparent
def get_assistant_avatar():
    if "assistant_avatar" in st.session_state:
        return st.session_state.assistant_avatar
    avatar = LOGO_PATH if Path(LOGO_PATH).exists() else ("logo.png" if Path("logo.png").exists() else "💼")
    st.session_state.assistant_avatar = avatar
    return avatar

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

_ = "Chat-like interface using Streamlit chat elements"
USE_CHAT = hasattr(st, "chat_message") and hasattr(st, "chat_input")

if USE_CHAT:
    # Render existing conversation
    # Assistant avatar uses padded Engro logo to avoid cropping
    assistant_avatar = get_assistant_avatar()
    # Inline style injection to ensure avatar sizing overrides runtime styles
    try:
        from streamlit.components.v1 import html as components_html
        components_html(
            """
            <style>
              [data-testid='stChatMessage'] [data-testid='stChatMessageAvatar'],
              [data-testid='stChatMessage'] [data-testid='stChatMessageAvatar'] > *,
              [data-testid='stChatMessage'] img {
                width: 3rem !important;
                height: 3rem !important;
                min-width: 3rem !important;
                min-height: 3rem !important;
                object-fit: contain !important;
                flex-shrink: 0 !important;
              }
            </style>
            """,
            height=0,
        )
    except Exception:
        pass
    for turn in st.session_state.qna_log:
        with st.chat_message("user"):
            st.markdown(turn.get("query", ""))
        with st.chat_message("assistant", avatar=assistant_avatar):
            if turn.get("context_included"):
                st.caption("Context included")
            st.markdown(turn.get("answer", ""))
            rows = turn.get("chunks") or []
            if rows:
                with st.expander("Thoughts "):
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.markdown("Snippets:")
                    for r in rows[:10]:
                        st.markdown(f"- **[{r['source']}:{r['id']}]** (score={r['score']}) {r['preview']}")

    # Bottom chat input
    prompt = st.chat_input("Ask a question")
    if prompt:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Please set it in your environment and restart.")
            st.stop()

        # Build filters dict from sidebar
        filters: Dict[str, Any] = {}
        if location:
            filters["location"] = location
        if department:
            filters["department"] = department
        if isinstance(start_dt, date):
            filters["start_date"] = start_dt.isoformat()
        if isinstance(end_dt, date):
            filters["end_date"] = end_dt.isoformat()

        # Create contextual query using last few turns if available
        context_included = bool(st.session_state.qna_log)
        query_to_send = prompt
        if context_included:
            context_turns = st.session_state.qna_log[-3:]
            context_block = "\n\n".join([
                f"Q: {t.get('query','')}\nA: {t.get('answer','')}" for t in context_turns
            ])
            query_to_send = f"Context:\n{context_block}\n\nQuestion: {prompt}"

        state = {
            "query": query_to_send,
            "filters": filters,
            "retrieved": [],
            "analytics": {},
            "answer": "",
        }
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner("Thinking..."):
                try:
                    final = graph_app.invoke(state, config=config)
                except Exception as e:
                    final = {"answer": f"There was an error generating a response: {e}", "retrieved": []}

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

            if context_included:
                st.caption("Context included")
            st.markdown(answer or "")
            if rows:
                with st.expander("Sources"):
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.markdown("Snippets:")
                    for r in rows[:10]:
                        st.markdown(f"- **[{r['source']}:{r['id']}]** (score={r['score']}) {r['preview']}")

        st.session_state.qna_log.append({
            "query": prompt,
            "answer": answer,
            "chunks": rows,
            "context_included": context_included,
        })
else:
    # Fallback simple input for older Streamlit versions
    default_q = (
        "What are the most concerned hazards and what steps should we take to avoid it turning into an incident?"
    )
    question = st.text_area("Question", value=default_q, height=100)
    ask = st.button("Ask", type="primary")
    if ask:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Please set it in your environment and restart.")
            st.stop()
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
        with st.spinner("Thinking..."):
            try:
                final = graph_app.invoke(state, config=config)
            except Exception as e:
                final = {"answer": f"There was an error generating a response: {e}", "retrieved": []}
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
        st.session_state.qna_log.append({
            "query": question,
            "answer": answer,
            "chunks": rows,
        })

# Footer (generic)
st.markdown("\n\n— EPCL Data Analyst")

"""
LangGraph pipeline for VEHS hazard Q&A with:
- RAG over Excel sheets (Hazard ID, Audit Findings, Inspection Findings, Incident)
- Analytics-based ranking (frequency × severity × recency)
- Concrete prevention steps (playbook) + citations to rows

Run:
  python bot.py
  # or provide a custom question
  python bot.py "In HTDC last quarter, what hazards are most concerning and how to prevent incidents?"
"""
import os
# Workaround for OpenMP duplicate runtime on Windows when using FAISS + MKL-based stacks
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
from typing import Optional, List, Dict, Any, TypedDict

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

XLSX_PATH = "EPCL_VEHS_Data_Processed.xlsx"
PERSIST_DIR = "vehsvdb"

# --------- Load sheets once for analytics ---------
SHEETS: Dict[str, pd.DataFrame] = {}


def is_hazard_query(q: str) -> bool:
    """Heuristic: detect if the user is asking for hazard ranking/steps vs general QA."""
    if not q:
        return False
    ql = q.lower()
    hazard_terms = [
        "hazard", "top hazards", "most concerned hazards", "prevent", "incident",
        "ppe", "housekeeping", "barric", "permit", "lopc", "leak", "isolation plan",
    ]
    return any(t in ql for t in hazard_terms)
if os.path.exists(XLSX_PATH):
    SHEETS = pd.read_excel(XLSX_PATH, sheet_name=None)
    for name, df in SHEETS.items():
        for col in df.columns:
            if any(k in col.lower() for k in ["date", "entered", "start"]):
                df[col] = pd.to_datetime(df[col], errors="coerce")
else:
    print(f"Warning: {XLSX_PATH} not found. Analytics may not work until the file is present.")

# --------- Vector store / retriever ---------
EMB = OpenAIEmbeddings(model="text-embedding-3-small")
try:
    VSTORE = FAISS.load_local(PERSIST_DIR, EMB, allow_dangerous_deserialization=True)
    RETRIEVER = VSTORE.as_retriever(search_kwargs={"k": 6})
except Exception as e:
    VSTORE = None
    RETRIEVER = None
    print(f"Warning: Could not load FAISS index from '{PERSIST_DIR}'. Build it first via `python build_index.py`.\n{e}")

# --------- LLM ---------
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --------- Hazard tagging rules and playbook ---------
TAG_RULES: List[tuple[str, str]] = [
    ("Permit Management", r"\bpermit|permits\b"),
    ("Isolation Plan Accuracy", r"\bisolation plan|re-energiz"),
    ("Firewater System Misuse", r"\bfirewater\b"),
    ("Housekeeping/Trip", r"\bcable|housekeeping|trip(ping)?\b"),
    ("PPE Compliance", r"\bppe|mask|glove|helmet|goggles\b"),
    ("Barrication/Tools", r"\bbarric(ad|ation)|warning lights|tools\b"),
    ("Mechanical Integrity/Aging", r"\bend of service life|aging|not yet been inspected|mechanical integrity|\bMI\b"),
    ("LOPC/Leakage", r"\bleak|\bLOPC\b|leakage\b"),
]

PLAYBOOK: Dict[str, List[str]] = {
    "Permit Management": [
        "Keep permits at work-front and CCR; cross-reference every permit to an isolation diagram.",
        "Increase permit-audit sample size; include 'critical isolation' checklist.",
    ],
    "Isolation Plan Accuracy": [
        "Field-verify isolation plans and update when work changes; enforce MOC for deviations.",
        "Run pre-energization checks with two-person verification.",
    ],
    "Firewater System Misuse": [
        "Physically decouple process-water tie-ins; add interlocks and signage.",
        "Weekly verification of firewater hydraulics; MOC for temporary tie-ins.",
    ],
    "Housekeeping/Trip": [
        "Implement cable management (trays/mats), daily 5S checks, defined walkways.",
    ],
    "PPE Compliance": [
        "Ensure point-of-use PPE availability; supervisor spot-checks; toolbox talks on specific gaps.",
    ],
    "Barrication/Tools": [
        "Standardize barricading kits and visual standards; pre-job barricade checks.",
    ],
    "Mechanical Integrity/Aging": [
        "Refresh RBI and circuit coverage; clear inspection backlogs; monitor IOWs.",
        "Escalate repeat failures to RCFA and redesign; adjust PM frequencies.",
    ],
    "LOPC/Leakage": [
        "Hose/fitting integrity checks; quick-connect standards; leak near-miss reporting.",
    ],
    "Other": [
        "Review the finding text and assign a specific control owner.",
    ],
}


def to_sev(val) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val)
    if "C3" in s:
        return 3
    if "C2" in s:
        return 2
    if "C1" in s:
        return 1
    if "C0" in s:
        return 0
    # try numeric
    try:
        return int(float(s))
    except Exception:
        return None


def tag_text(txt: str) -> List[str]:
    t = (txt or "").lower()
    tags = [name for name, pat in TAG_RULES if re.search(pat, t)]
    return tags or ["Other"]


def apply_filters(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    loc = f.get("location")
    dept = f.get("department")
    start = pd.to_datetime(f.get("start_date"), errors="coerce") if f.get("start_date") else None
    end = pd.to_datetime(f.get("end_date"), errors="coerce") if f.get("end_date") else None

    if loc and "location" in out.columns:
        out = out[out["location"].astype(str).str.contains(str(loc), case=False, na=False)]
    if dept and "department" in out.columns:
        out = out[out["department"].astype(str).str.contains(str(dept), case=False, na=False)]

    # Try likely date columns
    date_cols = [c for c in out.columns if any(k in c.lower() for k in ["occurrence", "reported", "start", "entered"]) ]
    if (start is not None or end is not None) and date_cols:
        dcol = date_cols[0]
        if start is not None:
            out = out[pd.to_datetime(out[dcol], errors="coerce") >= start]
        if end is not None:
            out = out[pd.to_datetime(out[dcol], errors="coerce") <= end]
    return out


# ---------- Analytics ----------

def hazard_analytics(filters: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    from collections import defaultdict

    def safe_col(df: pd.DataFrame, name: str, default=None):
        return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)

    frames = []

    # Hazard ID
    if "Hazard ID" in SHEETS:
        df = apply_filters(SHEETS["Hazard ID"], filters)
        if not df.empty:
            df = df.copy()
            sev_series = safe_col(df, "worst_case_consequence_potential_hazard_id")
            df["severity"] = sev_series.map(to_sev)
            text_cols = [c for c in ["title", "description", "violation_type_hazard_id"] if c in df.columns]
            if text_cols:
                df["text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
            else:
                df["text"] = ""
            df["tags"] = df["text"].map(tag_text)
            df["date"] = pd.to_datetime(df.get("occurrence_date"), errors="coerce")
            keep_cols = [c for c in ["tags", "severity", "date", "incident_id", "location", "department"] if c in df.columns]
            frames.append(("haz", df[keep_cols]))

    # Audit Findings
    if "Audit Findings" in SHEETS:
        df = apply_filters(SHEETS["Audit Findings"], filters)
        if not df.empty:
            df = df.copy()
            sev_series = safe_col(df, "worst_case_consequence")
            df["severity"] = sev_series.map(to_sev)
            text_cols = [c for c in ["audit_title", "finding"] if c in df.columns]
            if text_cols:
                df["text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
            else:
                df["text"] = ""
            df["tags"] = df["text"].map(tag_text)
            df["date"] = pd.to_datetime(df.get("start_date"), errors="coerce")
            keep_cols = [c for c in ["tags", "severity", "date", "audit_id", "location"] if c in df.columns or c in ["tags", "severity", "date"]]
            frames.append(("aud", df[keep_cols]))

    # Inspection Findings (severity mild if unknown)
    if "Inspection Findings" in SHEETS:
        df = apply_filters(SHEETS["Inspection Findings"], filters)
        if not df.empty:
            df = df.copy()
            df["severity"] = 1
            text_cols = [c for c in ["audit_title", "finding", "question"] if c in df.columns]
            if text_cols:
                df["text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
            else:
                df["text"] = ""
            df["tags"] = df["text"].map(tag_text)
            df["date"] = pd.to_datetime(df.get("start_date"), errors="coerce")
            keep_cols = [c for c in ["tags", "severity", "date", "audit_id", "location"] if c in df.columns or c in ["tags", "severity", "date"]]
            frames.append(("ins", df[keep_cols]))

    # Combine scoring
    agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "sev_sum": 0, "sev_n": 0, "recent": 0, "samples": []})

    horizon = pd.Timestamp.today() - pd.Timedelta(days=180)

    for source_name, df in frames:
        for _, r in df.iterrows():
            tags = r["tags"] if isinstance(r.get("tags"), list) else [r.get("tags")]
            for t in tags:
                if t is None:
                    continue
                a = agg[t]
                a["count"] += 1
                sev = r.get("severity")
                if pd.notna(sev):
                    a["sev_sum"] += sev
                    a["sev_n"] += 1
                d = r.get("date")
                try:
                    if pd.notna(d) and d >= horizon:
                        a["recent"] += 1
                except Exception:
                    pass
                # Keep a few sample IDs for citations
                rid = r.get("incident_id") if "incident_id" in r else r.get("audit_id")
                if rid is not None and len(a["samples"]) < 5:
                    a["samples"].append({"source": source_name, "id": rid})

    scored = []
    for tag, val in agg.items():
        avg_sev = (val["sev_sum"] / val["sev_n"]) if val["sev_n"] else 1.0
        concern = val["count"] + 0.75 * avg_sev + 0.5 * val["recent"]
        scored.append(
            {
                "hazard": tag,
                "count": val["count"],
                "avg_sev": round(avg_sev, 2),
                "recent": val["recent"],
                "concern_score": round(concern, 2),
                "samples": val["samples"],
                "steps": PLAYBOOK.get(tag, PLAYBOOK["Other"]),
            }
        )
    scored.sort(key=lambda x: x["concern_score"], reverse=True)
    return {"top": scored[:top_n]}


# ------------- LangGraph state + nodes -------------
class GraphState(TypedDict):
    query: str
    filters: Dict[str, Any]
    retrieved: List[Document]
    analytics: Dict[str, Any]
    answer: str


# 1) parse_filters node
filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract optional filters (location, department, start_date, end_date in ISO YYYY-MM-DD) from the user question. "
            "Respond with a JSON object containing zero or more of these keys. If a key is unknown, omit it.",
        ),
        ("human", "{query}"),
    ]
)


def parse_filters(state: GraphState) -> GraphState:
    q = state["query"]
    messages = filter_prompt.format_messages(query=q)
    resp = LLM.invoke(messages)
    # Best-effort JSON extraction
    filt: Dict[str, Any] = {}
    try:
        txt = (resp.content or "").strip()
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1:
            filt = json.loads(txt[start : end + 1])
        # Normalize keys
        filt = {k: v for k, v in filt.items() if k in {"location", "department", "start_date", "end_date"} and v}
    except Exception:
        filt = {}
    # Return only the key this node updates to avoid concurrent writes
    return {"filters": filt}


# 2) retrieve_docs node

def retrieve_docs(state: GraphState) -> GraphState:
    q = state["query"]
    f = state.get("filters", {})
    filt_terms = " ".join(str(v) for v in [f.get("location"), f.get("department")] if v)
    full_query = f"{q} {filt_terms}".strip()
    docs: List[Document] = []
    if VSTORE is not None:
        # Get similarity scores and attach to metadata for UI display
        try:
            results = VSTORE.similarity_search_with_score(full_query, k=6)
            docs = []
            for d, score in results:
                md = d.metadata or {}
                md["score"] = float(score) if score is not None else None
                d.metadata = md
                docs.append(d)
        except Exception:
            # Fallback to retriever if similarity with score not available
            if RETRIEVER is not None:
                docs = RETRIEVER.invoke(full_query)
    elif RETRIEVER is not None:
        # Use the modern retriever API to avoid deprecation warnings
        docs = RETRIEVER.invoke(full_query)
    else:
        print("Retriever not available (missing FAISS index). Run `python build_index.py` first.")
    # Return only updated key
    return {"retrieved": docs}


# 3) hazard_analytics node

def run_analytics(state: GraphState) -> GraphState:
    f = state.get("filters", {})
    ana = hazard_analytics(f, top_n=6) if SHEETS else {"top": []}
    # Return only updated key
    return {"analytics": ana}


# 4) synthesize_answer node
hazard_synth_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a safety assistant. Always answer in clear, layperson-friendly language. "
            "Follow this exact structure and order using Markdown headings:\n\n"
            "### Summary\n"
            "2-4 sentences in plain language that directly answer the question and give practical, prescriptive guidance.\n\n"
            "### Data insights\n"
            "3-6 concise bullets highlighting key trends/metrics from analytics and retrieved context. Keep wording simple.\n\n"
            "### Details\n"
            "A short ranked list of top hazard themes with 1-sentence 'why it matters' for each.\n\n"
            "### Actions\n"
            "Concise, practical prevention steps.\n\n"
            "### Citations\n"
            "List [Sheet:ID] pairs used. If none, omit the section."
        ),
        (
            "human",
            "User question: {query}\n\n"
            "Filters: {filters}\n\n"
            "Top hazards (JSON): {analytics}\n\n"
            "Retrieved snippets (for context):\n{snippets}",
        ),
    ]
)

# General QA synthesis prompt (source-grounded with citations)
general_qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful data assistant. Use retrieved context to answer. "
            "Write for non-experts and follow this exact structure and order using Markdown headings:\n\n"
            "### Summary\n"
            "2-4 sentences in simple language that directly answer the question and, when appropriate, give prescriptive guidance.\n\n"
            "### Data insights\n"
            "3-6 short bullets with the most relevant facts from the context (numbers, trends, locations, dates).\n\n"
            "### Details\n"
            "Any additional clarifications or steps as bullets.\n\n"
            "### Citations\n"
            "List [Sheet:ID] pairs used. If unsure, say so."
        ),
        (
            "human",
            "Question: {query}\n\nFilters: {filters}\n\nRetrieved snippets:\n{snippets}",
        ),
    ]
)


def synthesize_answer(state: GraphState) -> GraphState:
    # Build small snippet list + citations
    snippets: List[str] = []
    for d in state.get("retrieved", [])[:5]:
        meta = d.metadata or {}
        cid = meta.get("record_id") or ""
        src = meta.get("source_sheet") or ""
        content = (d.page_content or "")[:350].replace("\n", " ")
        snippets.append(f"[{src}:{cid}] {content}")

    q = state.get("query", "")
    if is_hazard_query(q):
        messages = hazard_synth_prompt.format_messages(
            query=q,
            filters=state.get("filters", {}),
            analytics=state.get("analytics", {}),
            snippets="\n".join(snippets),
        )
    else:
        messages = general_qa_prompt.format_messages(
            query=q,
            filters=state.get("filters", {}),
            snippets="\n".join(snippets),
        )
    resp = LLM.invoke(messages)
    # Return only updated key
    return {"answer": resp.content}


# Build graph
graph = StateGraph(GraphState)
graph.add_node("parse_filters", parse_filters)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("run_analytics", run_analytics)
graph.add_node("synthesize_answer", synthesize_answer)

# Edges
graph.set_entry_point("parse_filters")
graph.add_edge("parse_filters", "retrieve_docs")
graph.add_edge("retrieve_docs", "run_analytics")
graph.add_edge("run_analytics", "synthesize_answer")
graph.add_edge("synthesize_answer", END)

memory = MemorySaver()  # optional checkpointing
app = graph.compile(checkpointer=memory)


# Simple CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = (
            "What are the most concerned hazards and what steps should we take to avoid it turning into an incident?"
        )

    state: GraphState = {"query": question, "filters": {}, "retrieved": [], "analytics": {}, "answer": ""}
    # Provide a thread_id to satisfy MemorySaver (checkpointer) requirements
    config = {"configurable": {"thread_id": "cli-session"}}
    final = app.invoke(state, config=config)
    print("\n=== ANSWER ===\n")
    print(final.get("answer", "No answer produced."))

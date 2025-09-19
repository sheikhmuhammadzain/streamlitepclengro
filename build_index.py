"""
Build a FAISS vector store from the Excel workbook sheets with useful metadata for citations.

Usage:
  python build_index.py

Inputs:
  - EPCL_VEHS_Data_Processed.xlsx
Outputs:
  - vehsvdb/ (FAISS index directory)
"""
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

XLSX_PATH = "EPCL_VEHS_Data_Processed.xlsx"
TXT_PATH = "excel_analysis_report.txt"
PERSIST_DIR = "vehsvdb"


def coerce_dt(s):
    return pd.to_datetime(s, errors="coerce")


def load_sheets(xlsx: str) -> Dict[str, pd.DataFrame]:
    sheets = pd.read_excel(xlsx, sheet_name=None)
    # Coerce common date columns found in reports
    for name, df in sheets.items():
        for col in df.columns:
            if any(k in col.lower() for k in ["date", "entered", "start"]):
                df[col] = coerce_dt(df[col])
    return sheets


def to_docs(sheets: Dict[str, pd.DataFrame]) -> List[Document]:
    docs: List[Document] = []

    def add_doc(content: str, meta: Dict[str, Any]):
        content = (content or "").strip()
        if not content:
            return
        docs.append(Document(page_content=content, metadata=meta))

    def serialize_row(r: pd.Series, max_chars: int = 800) -> str:
        # Fallback: include a compact key:value dump of the row for general QA
        parts = []
        for k, v in r.items():
            if pd.notna(v):
                s = str(v)
                if s and s.lower() != "nan":
                    parts.append(f"{k}: {s}")
            if sum(len(p) for p in parts) > max_chars:
                break
        return " | ".join(parts)

    # Generic pass: index all sheets
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        df_local = df.copy()
        # Identify common metadata columns
        id_cols = [c for c in df_local.columns if any(k in c.lower() for k in ["incident_id", "audit_id", "hazard_id", "record_id", "id", "finding_id"]) ]
        loc_col = "location" if "location" in df_local.columns else None
        dept_col = "department" if "department" in df_local.columns else None
        date_cols = [c for c in df_local.columns if any(k in c.lower() for k in ["occurrence", "reported", "start", "entered", "date"]) ]

        for idx, r in df_local.iterrows():
            # Best-effort ID selection
            rid = None
            for c in id_cols:
                val = r.get(c)
                if pd.notna(val):
                    rid = str(val)
                    break
            if not rid:
                rid = f"{sheet_name}-{idx}"

            text = serialize_row(r)
            meta: Dict[str, Any] = {
                "source_sheet": sheet_name,
                "record_id": rid,
            }
            if loc_col:
                meta["location"] = r.get(loc_col)
            if dept_col:
                meta["department"] = r.get(dept_col)
            if date_cols:
                meta["date"] = r.get(date_cols[0])
            add_doc(text, meta)

    # Also index supplemental analysis report text if present
    txt_path = Path(TXT_PATH)
    if txt_path.exists():
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        # Split on blank lines into paragraphs
        paragraphs = [p.strip() for p in raw.splitlines()]
        for i, p in enumerate(paragraphs):
            if not p or len(p) < 40:
                continue
            add_doc(
                p,
                {
                    "source_sheet": "TextReport",
                    "record_id": f"TXT-{i+1}",
                },
            )

    return docs


def main():
    if not Path(XLSX_PATH).exists():
        raise FileNotFoundError(
            f"Missing {XLSX_PATH}. Place it in the current directory before running."
        )
    sheets = load_sheets(XLSX_PATH)
    docs = to_docs(sheets)
    total = len(docs)
    print(f"Prepared {total} docs")

    if total == 0:
        raise RuntimeError("No documents prepared for indexing.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build FAISS index in batches to avoid token-per-request limits
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    BATCH_SIZE = 64  # keep batches modest to stay under token limits

    vs = None
    for i in range(0, total, BATCH_SIZE):
        bt = texts[i:i+BATCH_SIZE]
        bm = metas[i:i+BATCH_SIZE]
        if vs is None:
            vs = FAISS.from_texts(bt, embeddings, metadatas=bm)
        else:
            vs.add_texts(bt, metadatas=bm)
        print(f"Indexed {min(i+BATCH_SIZE, total)}/{total}")

    # Persist
    Path(PERSIST_DIR).mkdir(exist_ok=True)
    vs.save_local(PERSIST_DIR)
    print(f"Saved FAISS index to {PERSIST_DIR}")


if __name__ == "__main__":
    main()

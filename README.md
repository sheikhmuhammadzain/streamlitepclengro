# VEHS Hazard Q&A (LangChain + LangGraph)

This minimal stack answers:

> What are the most concerned hazards and what steps should we take to avoid it turning into an incident?

You get:
- RAG over your sheets (`Incident`, `Hazard ID`, `Audit Findings`, `Inspection Findings`)
- Analytics-based ranking (frequency × severity × recency) of hazard themes
- Concrete prevention steps (playbook) + citations to your rows
- Simple LangGraph: `parse_filters → (retrieve_docs + run_analytics in parallel) → synthesize_answer`

## Files
- `EPCL_VEHS_Data_Processed.xlsx` — your processed source workbook
- `build_index.py` — builds FAISS vector store from the workbook
- `bot.py` — LangGraph app
- `requirements.txt` — Python dependencies

## Install (Windows PowerShell)
1) Optional: create and activate a venv
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Set your OpenAI API key
- Temporary for current PowerShell session:
```
$env:OPENAI_API_KEY = "your_openai_api_key"
```
- Or persist for future sessions (open a NEW terminal after this):
```
setx OPENAI_API_KEY "your_openai_api_key"
```

## Build the index
```
python build_index.py
```
This creates `vehsvdb/` with a FAISS index.

## Run the bot
Default question:
```
python bot.py
```
Custom question:
```
python bot.py "In HTDC last quarter, what hazards are most concerning and how do we prevent incidents?"
```

## Example queries
- In HTDC last quarter, what hazards are most concerning and how do we prevent incidents?
- For PVC, top hazards and preventive steps backed by findings?

## Notes & tweaks
- Expand `TAG_RULES` in `bot.py` or replace with an LLM classifier node if desired.
- For streaming output, wrap `synthesize_answer` with LangChain streaming callbacks.
- To serve an API, wrap `app.invoke()` in a FastAPI endpoint.
- If your sheet names/columns differ, adjust ingestion logic in `build_index.py` and analytics in `bot.py` accordingly.

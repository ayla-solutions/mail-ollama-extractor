"""
FastAPI service: mail-ollama-extractor
- Receives plain-text "email + attachment" body plus optional subject/received_at/graph_id.
- Classifies → {category, priority}.
- If Invoice → extracts invoice fields (incl. short 'description').
- If Customer Requests → summary (model) + deterministic ticket number (code).
- Returns JSON only. No persistence here (caller updates Dataverse).
"""

import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from schemas import ExtractIn, ExtractOut
from llm import classify_text, extract_invoice, extract_customer_request_summary
from utils import trim_text, yyyymmdd_from_iso, compute_ticket

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("mail-ollama-extractor")

app = FastAPI(title="mail-ollama-extractor", version="1.2.0")

@app.get("/")
def root():
    return {"ok": True, "service": "mail-ollama-extractor"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/extract", response_model=ExtractOut)
def extract(payload: ExtractIn):
    """
    Contract:
      - We receive body_text (plain text of email + attachments).
      - Optional: subject, received_at (ISO), graph_id (Graph message id).
      - Step 1: classify → category, priority (deterministic, seeded).
      - Step 2:
            * Invoice           → extract invoice fields (incl. description)
            * Customer Requests → summary (model) + ticket_number (code; deterministic)
            * General / Misc    → nothing else
    """
    try:
        if not payload.body_text or not payload.body_text.strip():
            raise HTTPException(status_code=422, detail="body_text is required")

        # Keep context bounded; let models see the subject if provided
        text = trim_text(payload.body_text.strip())
        if payload.subject:
            text = f"Subject: {payload.subject}\n\n{text}"

        # 1) Classify (strict JSON; deterministic by seed)
        cls = classify_text(text, graph_id=payload.graph_id)
        cat = (cls.category or "").strip()
        pr  = (cls.priority or "").strip()
        cat_norm = cat.lower()

        out: Dict[str, Any] = {"category": cat, "priority": pr}

        # 2) Extract per category
        if cat_norm in ("invoice", "invoices"):
            out["invoice"] = extract_invoice(text, graph_id=payload.graph_id).model_dump()

        elif cat_norm in ("customer requests", "customer request"):
            # a) concise summary (deterministic)
            summary = extract_customer_request_summary(text, graph_id=payload.graph_id).summary
            # b) deterministic ticket number
            date = yyyymmdd_from_iso(payload.received_at)
            ticket = compute_ticket(date_yyyymmdd=date, seed=(payload.graph_id or ""))
            out["request"] = {"summary": summary, "ticket_number": ticket}

        # general / misc → no extras
        return {"ok": True, "data": out}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Extractor failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Extractor failed: {e}"}
        )

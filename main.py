"""
FastAPI service: mail-ollama-extractor

Upgrades:
- New /classify endpoint (fast path).
- /extract now composes text from subject + body + attachments, classifies first,
  then conditionally performs heavy extraction.
- Strict, sentence-case categories/priorities with robust fallbacks.
- Structured JSON logging (minimal, no raw content).
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from schemas import ExtractIn, ExtractOut, InvoiceFields, RequestFields
from llm import classify_text, extract_invoice, extract_customer_request_summary
from utils import (
    trim_text,
    yyyymmdd_from_iso,
    compute_ticket,
    sha256_8,
    compose_email_text,
    title_case,
    log_event,
    MAX_CHARS_CLASSIFY,
    MAX_CHARS_EXTRACT,
)

# ----------------------------
# Bootstrap
# ----------------------------
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("mail-ollama-extractor")

# Slow-call thresholds (ms)
SLOW_CLASSIFY_WARN_MS = int(os.getenv("SLOW_CLASSIFY_WARN_MS", "20000"))
SLOW_INVOICE_WARN_MS = int(os.getenv("SLOW_INVOICE_WARN_MS", "25000"))
SLOW_REQUEST_WARN_MS = int(os.getenv("SLOW_REQUEST_WARN_MS", "15000"))

# ----------------------------
# Canonical categories / priorities
# ----------------------------
_ALLOWED_CATS = {"General", "Invoice", "Customer Requests", "Misc"}
_ALLOWED_PRIOS = {"High", "Medium", "Low"}

def canon_category(raw: Optional[str]) -> str:
    """
    Map anything to one of the allowed categories, emit Title Case.
    Handles plurals/casing variants.
    """
    key = (raw or "").strip().lower()
    key_sing = key[:-1] if key.endswith("s") else key
    if key.startswith("invoice"):
        return "Invoice"
    if key_sing in {"customer request", "customer request"} or "customer request" in key:
        return "Customer Requests"
    if key in {"misc", "miscellaneous"}:
        return "Misc"
    if key == "general":
        return "General"
    # default
    return "General"

def canon_priority(raw: Optional[str]) -> str:
    key = title_case(raw)
    return key if key in _ALLOWED_PRIOS else "Low"

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="mail-ollama-extractor", version="2.0.0")

@app.get("/")
def root():
    return {"ok": True, "service": "mail-ollama-extractor"}

@app.get("/health")
def health():
    return {"ok": True}

# ----------------------------
# NEW: Classification only (fast)
# ----------------------------
@app.post("/classify")
def classify_endpoint(payload: ExtractIn):
    body_text_raw = (payload.body_text or "").strip()
    if not body_text_raw:
        raise HTTPException(status_code=422, detail="body_text is required")

    subject = (payload.subject or "").strip()
    graph_id = (payload.graph_id or "").strip()
    att_texts: List[str] = payload.attachments_text or []

    # Compose minimal text for classification
    text_small = compose_email_text(subject, trim_text(body_text_raw, MAX_CHARS_CLASSIFY), [])
    t0 = time.monotonic()
    cls = classify_text(text_small, graph_id=graph_id)
    took = int((time.monotonic() - t0) * 1000)

    cat = canon_category(getattr(cls, "category", None))
    pri = canon_priority(getattr(cls, "priority", None))

    log_event(
        log,
        "classify",
        graph_id=graph_id or "-",
        body_sha=sha256_8(body_text_raw),
        took_ms=took,
        category=cat,
        priority=pri,
        attachments=len(att_texts),
        warn=took >= SLOW_CLASSIFY_WARN_MS,
    )
    return {"ok": True, "data": {"category": cat, "priority": pri}}

# ----------------------------
# Full extraction (classify -> extract-if-needed)
# ----------------------------
@app.post("/extract", response_model=ExtractOut)
def extract(payload: ExtractIn):
    try:
        # Validate
        body_text_raw = (payload.body_text or "").strip()
        if not body_text_raw:
            raise HTTPException(status_code=422, detail="body_text is required")

        subject = (payload.subject or "").strip()
        graph_id = (payload.graph_id or "").strip()
        received_at_iso = (payload.received_at or "").strip()
        att_texts: List[str] = payload.attachments_text or []

        # Quick classification on a smaller window
        text_small = compose_email_text(subject, trim_text(body_text_raw, MAX_CHARS_CLASSIFY), [])
        t0 = time.monotonic()
        cls = classify_text(text_small, graph_id=graph_id)
        classify_ms = int((time.monotonic() - t0) * 1000)

        raw_cat = getattr(cls, "category", None)
        raw_pri = getattr(cls, "priority", None)

        cat = canon_category(raw_cat)
        pri = canon_priority(raw_pri)

        log_event(
            log,
            "classify_in_extract",
            graph_id=graph_id or "-",
            body_sha=sha256_8(body_text_raw),
            took_ms=classify_ms,
            category=cat,
            priority=pri,
            attachments=len(att_texts),
            warn=classify_ms >= SLOW_CLASSIFY_WARN_MS,
        )

        out: Dict[str, Any] = {"category": cat, "priority": pri}

        # Only build the bigger prompt if needed
        if cat == "Invoice" or cat == "invoice" or cat == "Invoices" or cat == "invoices":
            text_big = compose_email_text(
                subject,
                trim_text(body_text_raw, MAX_CHARS_EXTRACT),
                [trim_text(a, MAX_CHARS_EXTRACT) for a in att_texts if a],
            )
            t2 = time.monotonic()
            inv = extract_invoice(text_big, graph_id=graph_id)
            inv_ms = int((time.monotonic() - t2) * 1000)

            present = sorted([k for k, v in inv.model_dump().items() if v is not None])
            log_event(
                log,
                "invoice_extract",
                graph_id=graph_id or "-",
                took_ms=inv_ms,
                fields_present=len(present),
                warn=inv_ms >= SLOW_INVOICE_WARN_MS,
            )
            out["invoice"] = inv.model_dump()

        elif cat == "Customer Requests":
            text_big = compose_email_text(
                subject,
                trim_text(body_text_raw, MAX_CHARS_EXTRACT),
                [trim_text(a, MAX_CHARS_EXTRACT) for a in att_texts if a],
            )
            t4 = time.monotonic()
            summary = extract_customer_request_summary(text_big, graph_id=graph_id).summary
            sum_ms = int((time.monotonic() - t4) * 1000)

            # Deterministic ticket number
            date_yyyymmdd = yyyymmdd_from_iso(received_at_iso)
            ticket = compute_ticket(date_yyyymmdd, seed=graph_id or "")

            log_event(
                log,
                "request_extract",
                graph_id=graph_id or "-",
                took_ms=sum_ms,
                ticket=ticket,
                warn=sum_ms >= SLOW_REQUEST_WARN_MS,
            )
            out["request"] = {"summary": summary, "ticket_number": ticket}
        else:
            # General / Misc → no extra fields
            log_event(log, "no_extraction_needed", graph_id=graph_id or "-", category=cat)

        log_event(
            log,
            "extract_done",
            graph_id=graph_id or "-",
            category=out.get("category"),
            priority=out.get("priority"),
        )
        return {"ok": True, "data": out}

    except HTTPException:
        raise
    except Exception as e:
        log_event(log, "extract_error", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Extractor failed: {e}"},
        )

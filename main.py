"""
mail-ollama-extractor (fully instrumented)
------------------------------------------
Receives a combined plain-text "email + attachments" body and optional subject/received_at/graph_id.
Steps:
  1) Validate & trim input; include Subject in prompt if provided.
  2) Classify deterministically via Ollama → {category, priority}.
  3) If Invoice      → extract invoice fields (incl. description).
     If Cust.Request → summarize + compute deterministic ticket number.
  4) Return JSON only. No persistence here.

This version adds:
- Per-request correlation id (request_id) + optional graph_id correlation
- Detailed timings for classify/extract blocks
- Payload size + first-N char previews (safe) for debugging
- Clear error paths with structured logs
"""

import os
import time
import uuid
import hashlib
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from schemas import ExtractIn, ExtractOut
from llm import classify_text, extract_invoice, extract_customer_request_summary
from utils import trim_text, yyyymmdd_from_iso, compute_ticket

# ------------------------------------------------------------------------------
# Config & Logging
# ------------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("mail-ollama-extractor")

PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "280"))
SLOW_CLASSIFY_MS = int(os.getenv("SLOW_CLASSIFY_MS", "3000"))
SLOW_INVOICE_MS  = int(os.getenv("SLOW_INVOICE_MS", "5000"))
SLOW_SUMMARY_MS  = int(os.getenv("SLOW_SUMMARY_MS", "4000"))

def _preview(s: str | None, lim: int = PREVIEW_CHARS) -> Dict[str, Any]:
    if not s:
        return {"len": 0, "preview": ""}
    s = s.strip()
    return {"len": len(s), "preview": (s[:lim] + ("…" if len(s) > lim else ""))}

def _digest(s: str | None, n: int = 8) -> str:
    """Small fingerprint to correlate inputs without logging full text."""
    if not s:
        return ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]

app = FastAPI(title="mail-ollama-extractor (instrumented)", version="1.2.0")

# ------------------------------------------------------------------------------
# Middleware: assign request_id and basic access log
# ------------------------------------------------------------------------------
@app.middleware("http")
async def correlate_request(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()

    log.info(
        "http_request_start",
        extra={"request_id": rid, "method": request.method, "path": request.url.path, "query": str(request.url.query)},
    )

    try:
        resp = await call_next(request)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log.info("http_request_end", extra={"request_id": rid, "status_code": resp.status_code, "elapsed_ms": elapsed_ms})
        resp.headers["X-Request-ID"] = rid
        return resp
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log.exception("http_request_error", extra={"request_id": rid, "elapsed_ms": elapsed_ms})
        return JSONResponse(status_code=500, content={"ok": False, "error": "Unhandled error in extractor"})

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "mail-ollama-extractor (instrumented)"}

@app.get("/health")
def health():
    return {"ok": True, "preview_chars": PREVIEW_CHARS}

@app.post("/extract", response_model=ExtractOut)
def extract(payload: ExtractIn):
    """
    Contract:
      - body_text: REQUIRED (plain text of email + attachments)
      - subject, received_at, graph_id: OPTIONAL (assist determinism & prompts)
    """
    # ---------- Validate ----------
    if not payload.body_text or not payload.body_text.strip():
        log.warning("bad_request_missing_body_text", extra={"graph_id": payload.graph_id})
        raise HTTPException(status_code=422, detail="body_text is required")

    # ---------- Prepare text ----------
    # limit size proactively (latency/cost control), add Subject if provided
    body_text = trim_text(payload.body_text.strip())
    text_for_prompt = body_text
    if payload.subject:
        text_for_prompt = f"Subject: {payload.subject}\n\n{body_text}"

    # Compute small fingerprints for debug without leaking full content
    text_fp   = _digest(text_for_prompt)
    subject_p = _preview(payload.subject)
    body_p    = _preview(body_text)

    log.info(
        "extract_request_received",
        extra={
            "graph_id": payload.graph_id,
            "text_fp": text_fp,
            "subject": subject_p,
            "body_text": body_p,
            "received_at": payload.received_at,
        },
    )

    out: Dict[str, Any] = {}
    t0 = time.perf_counter()

    try:
        # ---------- Step 1: Classify ----------
        c0 = time.perf_counter()
        cls = classify_text(text_for_prompt, graph_id=payload.graph_id)
        c1 = time.perf_counter()
        cls_ms = int((c1 - c0) * 1000)
        if cls_ms > SLOW_CLASSIFY_MS:
            log.warning("slow_classify", extra={"graph_id": payload.graph_id, "elapsed_ms": cls_ms})

        category = (cls.category or "").strip()
        priority = (cls.priority or "").strip()
        out.update({"category": category, "priority": priority})

        log.info(
            "classify_ok",
            extra={
                "graph_id": payload.graph_id,
                "elapsed_ms": cls_ms,
                "category": category,
                "priority": priority,
            },
        )

        # ---------- Step 2: Per-category extraction ----------
        cat_norm = category.lower()
        if cat_norm in ("invoice", "invoices"):
            i0 = time.perf_counter()
            inv = extract_invoice(text_for_prompt, graph_id=payload.graph_id).model_dump()
            i1 = time.perf_counter()
            inv_ms = int((i1 - i0) * 1000)
            if inv_ms > SLOW_INVOICE_MS:
                log.warning("slow_invoice_extract", extra={"graph_id": payload.graph_id, "elapsed_ms": inv_ms})

            # Log which keys are present (not values)
            present_keys = [k for k, v in inv.items() if v not in (None, "", [])]
            out["invoice"] = inv

            log.info(
                "invoice_extract_ok",
                extra={"graph_id": payload.graph_id, "elapsed_ms": inv_ms, "present_fields": present_keys},
            )

        elif cat_norm in ("customer requests", "customer request"):
            r0 = time.perf_counter()
            summary = extract_customer_request_summary(text_for_prompt, graph_id=payload.graph_id).summary
            r1 = time.perf_counter()
            sum_ms = int((r1 - r0) * 1000)
            if sum_ms > SLOW_SUMMARY_MS:
                log.warning("slow_request_summary", extra={"graph_id": payload.graph_id, "elapsed_ms": sum_ms})

            date_yyyymmdd = yyyymmdd_from_iso(payload.received_at)
            ticket = compute_ticket(date_yyyymmdd=date_yyyymmdd, seed=(payload.graph_id or ""))

            out["request"] = {"summary": summary, "ticket_number": ticket}
            log.info(
                "customer_request_ok",
                extra={
                    "graph_id": payload.graph_id,
                    "elapsed_ms": sum_ms,
                    "summary_len": len(summary or ""),
                    "ticket_number": ticket,
                },
            )

        # general / misc → nothing else
        t1 = time.perf_counter()
        total_ms = int((t1 - t0) * 1000)
        log.info(
            "extract_ok",
            extra={"graph_id": payload.graph_id, "total_elapsed_ms": total_ms, "category": out.get("category")},
        )
        return {"ok": True, "data": out}

    except HTTPException:
        raise
    except Exception as e:
        total_ms = int((time.perf_counter() - t0) * 1000)
        log.exception("extract_fail", extra={"graph_id": payload.graph_id, "total_elapsed_ms": total_ms})
        return JSONResponse(status_code=500, content={"ok": False, "error": f"Extractor failed: {e}"})

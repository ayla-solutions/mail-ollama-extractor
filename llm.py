"""
Ollama integration with strict JSON outputs.
- Deterministic: temperature=0.0 + per-request seed (graph_id + text).
- Robust JSON parsing: strips code fences and extracts the first complete JSON value.
- Invoice: includes short 'description' derived from items/particulars.
- Customer Request: we only ask for a summary; ticket numbers are computed in code.
"""

import json
import os
import re
from typing import Dict, Any, Optional, Literal, Type

from pydantic import BaseModel
from dotenv import load_dotenv
import ollama

from utils import trim_text

load_dotenv()

# ----------------------------
# Model & host configuration
# ----------------------------
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "mail-classifier-small")
INVOICE_MODEL    = os.getenv("INVOICE_MODEL",   "invoice-extractor-small")
REQUEST_MODEL    = os.getenv("REQUEST_MODEL",   "request-summarizer-small")

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# Deterministic defaults
OLLAMA_TEMPERATURE = _get_float("OLLAMA_TEMPERATURE", 0.0)   # no sampling noise
OLLAMA_NUM_PREDICT = _get_int("OLLAMA_NUM_PREDICT", 200)     # general calls
OLLAMA_NUM_CTX     = _get_int("OLLAMA_NUM_CTX", 3072)        # long attachments
OLLAMA_KEEP_ALIVE  = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Specific cap for invoices (use 0 to allow EOS/unlimited)
INVOICE_NUM_PREDICT = _get_int("INVOICE_NUM_PREDICT", 400)

client = ollama.Client(host=OLLAMA_HOST)

# ----------------------------
# Deterministic seed helper
# ----------------------------
def _det_seed(graph_id: Optional[str], text: str) -> int:
    """Stable 32-bit seed from graph_id + text digest."""
    import hashlib
    h = hashlib.sha256((graph_id or "").encode("utf-8") + text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

# ----------------------------
# Schemas (Pydantic v2)
# ----------------------------
Category  = Literal["General", "Invoice", "Customer Requests", "Misc"]
Priority  = Literal["High", "Medium", "Low"]

class Classification(BaseModel):
    category: Category
    priority: Priority

class InvoiceFields(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date:   Optional[str] = None
    due_date:       Optional[str] = None
    invoice_amount: Optional[str] = None
    payment_link:   Optional[str] = None
    bsb:            Optional[str] = None
    account_number: Optional[str] = None
    account_name:   Optional[str] = None
    biller_code:    Optional[str] = None
    payment_reference: Optional[str] = None
    description:    Optional[str] = None  # NEW

class RequestSummary(BaseModel):
    summary: str

# ----------------------------
# Prompt helper
# ----------------------------
def _wrap_email_prompt(text: str) -> str:
    return f"Email+attachments:\n{trim_text(text).strip()}\n"

# ----------------------------
# Robust JSON helpers
# ----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]  # drop first fence line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s

def _first_complete_json(s: str) -> Optional[str]:
    """
    Return the first complete top-level JSON object/array from s,
    using brace/bracket counting with quote/escape handling.
    """
    start_idx = None
    stack = []
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if start_idx is None:
            if ch in "{[":
                start_idx = i
                stack = [ch]
                in_str = False
                esc = False
            continue

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            open_ch = stack.pop()
            if (open_ch, ch) not in (("{", "}"), ("[", "]")):
                return None
            if not stack:
                return s[start_idx:i+1]
    return None

# ----------------------------
# Core generate with strict JSON
# ----------------------------
def _gen(
    model: str,
    prompt: str,
    schema_model: Type[BaseModel],
    seed: Optional[int] = None,
    num_predict_override: Optional[int] = None
) -> Dict[str, Any]:
    """
    Prefer JSON Schema; fallback to 'format="json"'. Always try to
    extract the first complete JSON value if raw parse fails.
    """
    schema = schema_model.model_json_schema()
    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "num_predict": OLLAMA_NUM_PREDICT,
        "num_ctx":     OLLAMA_NUM_CTX,
        "top_p":       1,
        "mirostat":    0,
    }
    if num_predict_override is not None:
        options["num_predict"] = num_predict_override
    if seed is not None:
        options["seed"] = seed

    # 1) JSON Schema
    try:
        resp = client.generate(
            model=model,
            prompt=prompt,
            format=schema,
            options=options,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        raw = _strip_code_fences((resp.get("response") or "").strip())
        try:
            return json.loads(raw)
        except Exception:
            extracted = _first_complete_json(raw)
            if extracted:
                return json.loads(extracted)
            raise
    except Exception:
        # 2) Plain JSON mode
        resp = client.generate(
            model=model,
            prompt=prompt,
            format="json",
            options=options,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        raw = _strip_code_fences((resp.get("response") or "").strip())
        try:
            return json.loads(raw)
        except Exception:
            extracted = _first_complete_json(raw)
            if extracted:
                return json.loads(extracted)
            raise RuntimeError(f"Model returned non-JSON: {raw[:200]}...")

# ----------------------------
# Tasks
# ----------------------------
def classify_text(text: str, graph_id: Optional[str] = None) -> Classification:
    prompt = _wrap_email_prompt(text) + """
You are a strict JSON classifier.

Allowed categories (choose exactly one):
- General
- Invoice
- Customer Requests
- Misc

Priority (choose one): High, Medium, Low.
- Use "High" ONLY if explicit urgency words appear: "urgent", "asap", "immediate", "critical", "priority: high".
- If a due date/time is present WITHOUT those words → "Medium".
- Otherwise → "Low".

Return only the JSON object matching the schema.
"""
    data = _gen(CLASSIFIER_MODEL, prompt, Classification, seed=_det_seed(graph_id, text))
    return Classification(**data)

def extract_invoice(text: str, graph_id: Optional[str] = None) -> InvoiceFields:
    prompt = _wrap_email_prompt(text) + """
Extract invoice details strictly from the text. Do not guess.
If a field isn't present, set it to null.
Keep numbers/strings exactly as they appear (no reformatting).

Also produce a concise "description" of what the invoice is for:
- 1–2 sentences max.
- Base ONLY on items/particulars/services visible in the text.
- Name the key item(s) or service(s) if listed.
- Do NOT include totals or payment details.
- If unclear, set description to null.

Return JSON with these keys ONLY:
- invoice_number
- invoice_date
- due_date
- invoice_amount
- payment_link
- bsb
- account_number
- account_name
- biller_code
- payment_reference
- description
"""
    data = _gen(
        INVOICE_MODEL,
        prompt,
        InvoiceFields,
        seed=_det_seed(graph_id, text),
        num_predict_override=INVOICE_NUM_PREDICT,  # bigger cap for invoices
    )
    return InvoiceFields(**data)

def extract_customer_request_summary(text: str, graph_id: Optional[str] = None) -> RequestSummary:
    prompt = _wrap_email_prompt(text) + """
Summarize the customer's request in 2–3 sentences (plain English, no assumptions).
Return JSON with only:
{ "summary": string }
"""
    data = _gen(REQUEST_MODEL, prompt, RequestSummary, seed=_det_seed(graph_id, text))
    return RequestSummary(**data)

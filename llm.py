"""
    Ollama integration with strict JSON outputs and deterministic seeding.
    Upgrades:
    - Stronger classification prompt (exact allowed categories/priorities + casing).
    - Separate caps for classify vs extract (faster).
    - Invoice fallback regex parser to fill missing fields.
"""

import json
import os
from typing import Dict, Any, Optional, Literal, Type

from pydantic import BaseModel
from dotenv import load_dotenv
import ollama
import re

from utils import trim_text, title_case, MAX_CHARS_CLASSIFY, MAX_CHARS_EXTRACT

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
# Schemas (local for LLM I/O)
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
    description:    Optional[str] = None

class RequestSummary(BaseModel):
    summary: str

# ----------------------------
# JSON helpers
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
            raise

# ----------------------------
# Classification instructions
# ----------------------------
_ALLOWED_CATS = ["General", "Invoice", "Customer Requests", "Misc"]
_ALLOWED_PRIOS = ["High", "Medium", "Low"]

_CLASSIFY_INSTRUCTIONS = f"""
You are a STRICT JSON classifier for business emails.

Allowed categories (exactly one, case-sensitive):
- General
- Invoice
- Customer Requests
- Misc

Allowed priorities:
- High
- Medium
- Low

Rules:
- Read the ENTIRE email body AND any attachment text included in the prompt.
- Use the exact casing shown above; do not use plurals.
- Decide by context and content; do not rely on isolated keywords like “invoice” or “bill”.
- Never invent information not present in the text.

Category guidelines:
- “Invoice”: choose this category only if the email clearly contains **at least three** invoice‑specific cues.  Examples of cues include: (a) an attached invoice or bill document; (b) an explicit invoice number or reference; (c) an invoice date or due date; (d) a total amount or amount due; (e) payment instructions or bank/account details.  If fewer than three of these cues are present—even if the words “invoice” or “payment” appear—do **not** choose “Invoice”; instead classify as “General” or “Misc” as appropriate.
- “Customer Requests”: the sender is asking for help, information or action that is not a bill; e.g., support tickets, queries, complaints or account changes.
- “Misc”: automated notifications, marketing emails or system alerts that are unrelated to customer service or billing.
- Otherwise → “General”.

Priority guidelines:
- Use “High” **ONLY** if the email explicitly expresses urgency (e.g., “urgent”, “asap”, “immediately”, “critical”) or there is a due date that is imminent.
- Use “Medium” when there is a due date/time but no explicit urgency words.
- Otherwise → “Low”.

Return only this JSON object:
{{"category": <one of {_ALLOWED_CATS}>, "priority": <one of {_ALLOWED_PRIOS}>}}
"""

_INVOICE_INSTRUCTIONS = """
Extract invoice details strictly from the text provided (email + attachments). Do not guess.
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

_REQUEST_INSTRUCTIONS = """
Summarize the customer's request in 2–3 sentences (plain English, no assumptions).
Return JSON with only:
{ "summary": string }
"""

# ----------------------------
# Public tasks
# ----------------------------
def classify_text(text: str, graph_id: Optional[str] = None) -> Classification:
    # Keep classification fast by capping input smaller
    text_small = trim_text(text, MAX_CHARS_CLASSIFY)
    prompt = f"{text_small}\n\n{_CLASSIFY_INSTRUCTIONS}"
    data = _gen(CLASSIFIER_MODEL, prompt, Classification, seed=_det_seed(graph_id, text_small))
    # Hard guard: ensure Title Case and valid values
    category = title_case(data.get("category"))
    priority = title_case(data.get("priority"))
    if category not in _ALLOWED_CATS:
        # try to coerce common variants
        low = (category or "").lower()
        if low.startswith("invoice"):
            category = "Invoice"
        elif low.startswith("customer request"):
            category = "Customer Requests"
        elif "misc" in low:
            category = "Misc"
        else:
            category = "General"
    if priority not in _ALLOWED_PRIOS:
        priority = "Low"
    return Classification(category=category, priority=priority)

def extract_invoice(text: str, graph_id: Optional[str] = None) -> InvoiceFields:
    # Use larger cap for extraction
    text_big = trim_text(text, MAX_CHARS_EXTRACT)
    prompt = f"{text_big}\n\n{_INVOICE_INSTRUCTIONS}"
    data = _gen(
        INVOICE_MODEL,
        prompt,
        InvoiceFields,
        seed=_det_seed(graph_id, text_big),
        num_predict_override=INVOICE_NUM_PREDICT,
    )
    inv = InvoiceFields(**data)

    # Fallback: if too few fields present, run regex extractor and merge
    present = [k for k, v in inv.model_dump().items() if v]
    if len(present) <= 2:
        fallback = _fallback_invoice_parse(text_big)
        merged = inv.model_dump()
        for k, v in fallback.items():
            if not merged.get(k) and v:
                merged[k] = v
        inv = InvoiceFields(**merged)

    return inv

def extract_customer_request_summary(text: str, graph_id: Optional[str] = None) -> RequestSummary:
    text_big = trim_text(text, MAX_CHARS_EXTRACT)
    prompt = f"{text_big}\n\n{_REQUEST_INSTRUCTIONS}"
    data = _gen(REQUEST_MODEL, prompt, RequestSummary, seed=_det_seed(graph_id, text_big))
    # Safety trim
    data["summary"] = (data.get("summary") or "").strip()
    return RequestSummary(**data)

# ----------------------------
# Regex fallback for invoices
# ----------------------------
_NUM = r"[0-9][0-9,]*"
_AMT = rf"(?:{_NUM}(?:\.\d{{2}})?)"

def _search(pattern: str, text: str, flags=re.I):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

def _fallback_invoice_parse(text: str) -> Dict[str, Optional[str]]:
    fields: Dict[str, Optional[str]] = {
        "invoice_number": _search(r"\bInvoice(?:\s*(?:No\.?|#|Number))?[:\-\s]*([A-Za-z0-9\-\/]+)", text),
        "invoice_date":   _search(r"\bInvoice\s*Date[:\-\s]*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})", text)
                          or _search(r"\bDate[:\-\s]*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})", text),
        "due_date":       _search(r"\b(?:Due\s*Date|Payment\s*Due)[:\-\s]*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})", text),
        "invoice_amount": _search(r"\b(?:Total\s*Due|Total\s*Amount|Amount\s*Due)[:\-\s]*(\$?\s*" + _AMT + ")", text),
        "payment_link":   _search(r"(https?://\S+)", text),
        "bsb":            _search(r"\bBSB[:\-\s]*([0-9]{3}[- ]?[0-9]{3})", text),
        "account_number": _search(r"\bAccount\s*Number[:\-\s]*([0-9]{5,})", text),
        "account_name":   _search(r"\bAccount\s*Name[:\-\s]*([^\n]+)", text),
        "biller_code":    _search(r"\bBiller\s*Code[:\-\s]*([0-9]+)", text),
        "payment_reference": _search(r"\bPayment\s*Reference[:\-\s]*([A-Za-z0-9\-]+)", text),
        "description":    None,  # description has to come from LLM; fallback can't infer
    }
    return fields

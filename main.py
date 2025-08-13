from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os, re

from extractors import extract_attachment_text
from llm import (
    classify_mail, extract_invoice, extract_meeting, extract_timesheet, extract_request,
    Classification
)
from dataverse_client import create_row, col
from datetime import datetime
import re

def ensure_ddmmyyyy(s: str | None) -> str | None:
    if not s: 
        return None
    s = s.strip()
    # Try common formats and coerce to dd-mm-yyyy
    for fmt in ("%d-%m-%Y","%Y-%m-%d","%d/%m/%Y","%d %b %Y","%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%d-%m-%Y")
        except Exception:
            pass
    # Last-ditch: pick day/month/year in text
    m = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', s)
    if m:
        d, mth, y = m.groups()
        return f"{int(d):02d}-{int(mth):02d}-{y}"
    return s  # leave as-is if we can't parse

load_dotenv()

app = FastAPI(title="Mail LLM Extractor")

@app.get("/")
def health():
    return {"ok": True, "message": "API running"}

class Attachment(BaseModel):
    filename: str
    content_base64: str

class MailIn(BaseModel):
    subject: str
    sender: str
    body_html: Optional[str] = ""
    body_text: Optional[str] = ""
    received_at: Optional[str] = None  # ISO
    attachments: List[Attachment] = []
    push_to_dataverse: bool = True

def _normalize_body(m: MailIn) -> str:
    body = (m.body_text or "") + "\n" + (m.body_html or "")
    return body.strip()

def _regex_invoice_fallback(text: str) -> dict:
    # Helpful sanity check if LLM misses critical fields
    bsb = None
    acc = None
    amt = None
    # BSB: 6 digits with optional hyphen
    m = re.search(r"\b(?:BSB[:\s]*)?(\d{3}[-\s]?\d{3})\b", text, re.I)
    if m: bsb = m.group(1)
    # Account number: 6-12 digits heuristic
    m = re.search(r"\b(?:account(?:\s*no\.?| number)[:\s]*)?(\d{6,12})\b", text, re.I)
    if m: acc = m.group(1)
    # Amount: AUD with $ or numbers
    m = re.search(r"\b(?:total|amount due|amount)\s*[:$]?\s*\$?\s*([0-9][0-9,]*\.?[0-9]{0,2})\b", text, re.I)
    if m: amt = float(m.group(1).replace(",",""))
    return {"bsb": bsb, "account_number": acc, "invoice_amount": amt}

def _regex_invoice_number(text: str, subject: str) -> Optional[str]:
    pats = [
        r"(?i)\b(?:invoice|inv|tax\s*invoice|reference|ref)\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/]{2,})",
        r"(?i)\b(INV[-_/]?[A-Z0-9]{2,})\b",
    ]
    for p in pats:
        m = re.search(p, subject) or re.search(p, text)
        if m:
            val = m.group(1).strip().rstrip(".,)")
            return val[:40]
    return None

@app.post("/extract")
def extract(m: MailIn):
    attach_text, names_csv = extract_attachment_text([(a.filename, a.content_base64) for a in m.attachments])
    text_blob = "\n\n".join([_normalize_body(m), attach_text]).strip()

    # 1) classify
    cls: Classification = classify_mail(text_blob)

    # 2) per-category extraction
    category = cls.category
    result: Dict[str, Any] = {"category": category, "priority": cls.priority}

    inv = meet = ts = req = None
    if category == "Invoices":
        inv = extract_invoice(text_blob).model_dump()
        fb = _regex_invoice_fallback(text_blob)
        for k, v in fb.items():
            if (inv.get(k) is None) and v is not None:
                inv[k] = v
        invnum = _regex_invoice_number(text_blob, m.subject)
        if not inv.get("invoice_number") and invnum:
            inv["invoice_number"] = invnum
        result["invoice"] = inv
    elif category == "Meeting Requests":
        meet = extract_meeting(text_blob).model_dump()
        # normalize online
        if meet.get("meeting_link") and not meet.get("location"):
            meet["location"] = "Online"
        result["meeting"] = meet
    elif category == "Timesheets":
        ts = extract_timesheet(text_blob).model_dump()
        result["timesheet"] = ts
    elif category in ("Customer Requests","Team Member Requests"):
        req = extract_request(text_blob).model_dump()
        result["request"] = req

    # 3) optionally push to Dataverse as new row
    if m.push_to_dataverse:
        payload = {
          col("COL_SUBJECT"): m.subject,
          col("COL_SENDER"): m.sender,
          col("COL_RECEIVED_AT"): m.received_at,
          col("COL_CATEGORY"): category,
          col("COL_PRIORITY"): cls.priority,
          col("COL_ATTACHMENT_TEXT"): text_blob[:900000],  # safeguard
          col("COL_ATTACHMENT_NAMES"): names_csv,
          col("COL_SUMMARY"): (req or meet or inv or ts or {}),  # dump in Summary JSON column if you have one
        }
        # invoice specifics (+ default paid=false)
        if inv:
            payload.update({
            col("COL_INV_DUE_DATE"): ensure_ddmmyyyy(inv.get("due_date")),
            col("COL_INV_NUMBER"): inv.get("invoice_number"),
            col("COL_INV_DATE"): ensure_ddmmyyyy(inv.get("invoice_date")),
            col("COL_INV_AMOUNT"): inv.get("invoice_amount"),
            col("COL_INV_BSB"): inv.get("bsb"),
            col("COL_INV_ACC_NO"): inv.get("account_number"),
            col("COL_INV_ACC_NAME"): inv.get("account_name"),
            col("COL_INV_PAYMENT_LINK"): inv.get("payment_link"),
            col("COL_PAID"): False,
            })

        if meet:
            payload.update({
            col("COL_MEET_DATE"): ensure_ddmmyyyy(meet.get("date")),
            col("COL_MEET_TIME"): meet.get("time"),
            col("COL_MEET_LOCATION"): meet.get("location"),
            col("COL_MEET_LINK"): meet.get("meeting_link"),
            })

        if ts:
            payload.update({
            col("COL_TS_START"): ensure_ddmmyyyy(ts.get("period_start")),
            col("COL_TS_END"): ensure_ddmmyyyy(ts.get("period_end")),
            col("COL_TS_AL"): ts.get("annual_leave_hours"),
            col("COL_TS_PL"): ts.get("personal_leave_hours"),
            col("COL_TS_TOTAL"): ts.get("total_hours"),
            col("COL_TS_APPROVAL_LINKS"): ", ".join(ts.get("approval_links") or []),
            })
        if req:
            payload.update({
              col("COL_REQ_TITLE"): req.get("title"),
              col("COL_REQ_OVERVIEW"): req.get("overview"),
              col("COL_REQ_NUMBER"): req.get("request_number"),
            })
        # remove None keys
        payload = {k:v for k,v in payload.items() if k and v is not None}
        create_row(payload)

    return {"ok": True, "data": result}

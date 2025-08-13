import json
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
import ollama
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "mail-classifier")
INVOICE_MODEL = os.getenv("INVOICE_MODEL", "invoice-extractor")
MEETING_MODEL = os.getenv("MEETING_MODEL", "meeting-extractor")
TIMESHEET_MODEL = os.getenv("TIMESHEET_MODEL", "timesheet-extractor")
REQUEST_MODEL = os.getenv("REQUEST_MODEL", "request-extractor")

client = ollama.Client(host=OLLAMA_HOST)

Category = Literal["General","Invoices","Meeting Requests","Customer Requests","Timesheets","Team Member Requests","Miscellaneous"]
Priority = Literal["High","Medium","Low"]

class Classification(BaseModel):
    category: Category
    priority: Priority

class InvoiceFields(BaseModel):
    invoice_number: Optional[str] = None          # <— new
    due_date: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_amount: Optional[float] = None
    bsb: Optional[str] = None
    account_number: Optional[str] = None
    account_name: Optional[str] = None
    payment_link: Optional[str] = None

class MeetingFields(BaseModel):
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None  # "Online" if virtual
    meeting_link: Optional[str] = None

class TimesheetFields(BaseModel):
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    annual_leave_hours: Optional[float] = None
    personal_leave_hours: Optional[float] = None
    total_hours: Optional[float] = None
    approval_links: List[str] = Field(default_factory=list)

class RequestFields(BaseModel):
    title: Optional[str] = None
    overview: Optional[str] = None
    request_number: Optional[str] = None

def _gen(model: str, prompt: str, schema: BaseModel) -> dict:
    # Ollama's python client: use format to enforce schema (same pattern you used in your invoice reader)
    # Response: {'model':..., 'response': '<json string>', ...}
    r = client.generate(model=model, prompt=prompt, format=schema.model_json_schema())
    return json.loads(r["response"])

def classify_mail(text_blob: str) -> Classification:
    prompt = f"""Classify the email and attachments.
Text:
{text_blob}
Return JSON with fields: category, priority."""
    data = _gen(CLASSIFIER_MODEL, prompt, Classification)
    return Classification(**data)

def extract_invoice(text_blob: str) -> InvoiceFields:
    prompt = f"""Extract Australian invoice fields from the following text.
Text:
{text_blob}
Return JSON with: invoice_number, due_date (dd-mm-yyyy), invoice_date (dd-mm-yyyy),
invoice_amount (number), bsb (6 digits, hyphen allowed), account_number, account_name, payment_link (URL or null).
If a field is missing, return null."""
    data = _gen(INVOICE_MODEL, prompt, InvoiceFields)
    return InvoiceFields(**data)

def extract_meeting(text_blob: str) -> MeetingFields:
    prompt = f"""Extract meeting details from the text.
Text:
{text_blob}
Return JSON: date (dd-mm-yyyy), time (HH:MM 24h), location, meeting_link."""
    ...

def extract_timesheet(text_blob: str) -> TimesheetFields:
    prompt = f"""Extract timesheet details.
Text:
{text_blob}
Return JSON: period_start (dd-mm-yyyy), period_end (dd-mm-yyyy), annual_leave_hours, personal_leave_hours, total_hours, approval_links (array)."""
    ...

def extract_request(text_blob: str) -> RequestFields:
    prompt = f"""Extract request info.
Text:
{text_blob}
Return JSON: title, overview, request_number (if any)."""
    data = _gen(REQUEST_MODEL, prompt, RequestFields)
    return RequestFields(**data)
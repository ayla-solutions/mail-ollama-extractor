"""
Pydantic schemas for request/response & LLM IO.
"""

from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field

# ----------------------------
# Request from Mail Classification API
# ----------------------------
class ExtractIn(BaseModel):
    body_text: str = Field(..., description="Plain text of the email body")
    # If you have OCR/parsed text from attachments, pass them here individually.
    attachments_text: Optional[List[str]] = Field(
        default=None, description="List of plain-text contents from attachments"
    )
    # Optional pass-throughs (not used for reasoning; used for deterministic tickets/logging)
    graph_id: Optional[str] = Field(default=None, description="Microsoft Graph message id")
    subject: Optional[str] = Field(default=None, description="Email subject")
    received_at: Optional[str] = Field(default=None, description="ISO-8601 (used for ticket date)")

# ----------------------------
# Classifier result
# ----------------------------
Category = Literal["General", "Invoice", "Customer Requests", "Misc"]
Priority = Literal["High", "Medium", "Low"]

class Classification(BaseModel):
    category: Category
    priority: Priority

# ----------------------------
# Invoice extraction result
# ----------------------------
class InvoiceFields(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    invoice_amount: Optional[str] = None
    payment_link: Optional[str] = None
    bsb: Optional[str] = None
    account_number: Optional[str] = None
    account_name: Optional[str] = None
    biller_code: Optional[str] = None
    payment_reference: Optional[str] = None
    description: Optional[str] = None  # SHORT purpose-of-invoice

# ----------------------------
# Customer request (response shape)
# ----------------------------
class RequestFields(BaseModel):
    summary: str
    ticket_number: Optional[str] = None

# ----------------------------
# Final API response
# ----------------------------
class ExtractOut(BaseModel):
    ok: bool = True
    data: Dict[str, Any]

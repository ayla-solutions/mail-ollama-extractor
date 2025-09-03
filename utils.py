"""
Utilities:
- trim_text: bounds input size for latency/cost control.
- title_case: sentence-case (per-word Title Case) normalizer.
- compose_email_text: consistent prompt assembly from subject/body/attachments.
- yyyymmdd_from_iso: ISO -> YYYYMMDD (fallback to today).
- compute_ticket: deterministic ticket numbers (prefix + date + last6 of seed).
- preview/hash helpers for logs (kept minimal in structured logging).
- log_event: minimal JSON structured logging (no raw text).
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from datetime import datetime
import logging
from typing import Iterable, Optional

# ----------------------------
# Size bounds (characters)
# ----------------------------
MAX_CHARS_EXTRACT = int(os.getenv("EXTRACTOR_MAX_CHARS", "12000"))
MAX_CHARS_CLASSIFY = int(os.getenv("CLASSIFY_MAX_CHARS", "4000"))

def trim_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    return s.strip()[:max_chars]

# ----------------------------
# Dates & tickets
# ----------------------------
def yyyymmdd_from_iso(iso: str | None) -> str:
    """2025-08-26T23:12:00Z -> '20250826' (fallback to UTC today if missing/invalid)."""
    try:
        dt = datetime.fromisoformat((iso or "").replace("Z", "+00:00"))
    except Exception:
        dt = datetime.utcnow()
    return dt.strftime("%Y%m%d")

def compute_ticket(date_yyyymmdd: str, seed: str, prefix: str = None, counter: str | None = None) -> str:
    """
    Deterministic ticket: PREFIX + date + last6(seed alnums).
    If TICKET_PREFIX env var is set, that wins; else use provided prefix or 'REQ-'.
    """
    env_prefix = os.getenv("TICKET_PREFIX")
    prefix = env_prefix if env_prefix is not None else (prefix or "REQ-")

    alnums = "".join(ch for ch in (seed or "") if ch.isalnum()).upper()
    last6 = alnums[-6:] if len(alnums) >= 6 else alnums
    ticket = f"{prefix}{date_yyyymmdd}-{last6}"
    if counter:
        ticket += f"-{counter}"
    return ticket

# ----------------------------
# Safe previews / hashing (used sparingly)
# ----------------------------
REDACT_PATTERNS = [re.compile(r"\b\d{6,}\b")]  # long digit runs (account numbers, refs)

def sha256_8(s: str) -> str:
    """Short digest for correlation in logs."""
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:8]

def _redact_line(s: str) -> str:
    if not s:
        return ""
    out = s.replace("\n", " ").replace("\r", " ").strip()
    for pat in REDACT_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out

# ----------------------------
# Text helpers
# ----------------------------
def title_case(s: Optional[str]) -> str:
    """Title-case each word (what you called 'sentence case'—first letter caps)."""
    s = (s or "").strip()
    if not s:
        return s
    return " ".join(w.capitalize() for w in s.split())

def compose_email_text(
    subject: Optional[str],
    body_text: str,
    attachments_text: Optional[Iterable[str]],
) -> str:
    """
    Compose a consistent prompt block with clear sections.
    Keep composition here so LLM wrappers can stay simple.
    """
    parts = []
    if subject:
        parts.append(f"Subject: {subject.strip()}")
    parts.append("Email Body:")
    parts.append(body_text.strip())
    att = attachments_text or []
    if att:
        parts.append("\nAttachments:")
        for i, a in enumerate(att, start=1):
            snippet = (a or "").strip()
            if snippet:
                parts.append(f"--- Attachment {i} ---\n{snippet}")
    return "\n\n".join(parts)

# ----------------------------
# Structured logging
# ----------------------------
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower().strip()

def log_event(logger: logging.Logger, event: str, **fields):
    """
    Minimal JSON structured logging; only small, safe fields.
    DO NOT log raw email/attachment text.
    """
    payload = {"event": event, **fields}
    if LOG_FORMAT == "json":
        logger.info(json.dumps(payload, ensure_ascii=False))
    else:
        # Plain fallback
        flat = " ".join(f"{k}={v}" for k, v in payload.items())
        logger.info(flat)

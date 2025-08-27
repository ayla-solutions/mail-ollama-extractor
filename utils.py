"""
Utilities:
- trim_text: bounds input size for latency/cost control.
- yyyymmdd_from_iso: ISO -> YYYYMMDD (fallback to today).
- compute_ticket: deterministic ticket numbers (prefix + date + last6 of seed).
"""

import os
from datetime import datetime

# Hard cap on input size sent to LLM (characters)
MAX_CHARS = int(os.getenv("EXTRACTOR_MAX_CHARS", "12000"))

def trim_text(s: str, max_chars: int = MAX_CHARS) -> str:
    if not s:
        return ""
    return s.strip()[:max_chars]

def yyyymmdd_from_iso(iso: str | None) -> str:
    """2025-08-26T23:12:00Z -> '20250826' (fallback to UTC today if missing/invalid)."""
    try:
        dt = datetime.fromisoformat((iso or "").replace("Z", "+00:00"))
    except Exception:
        dt = datetime.utcnow()
    return dt.strftime("%Y%m%d")

def compute_ticket(date_yyyymmdd: str, seed: str, prefix: str = "REQ-", counter: str | None = None) -> str:
    """
    Use last 6 alphanumerics of seed (uppercased) for deterministic ticket numbers.
    """
    alnums = "".join(ch for ch in (seed or "") if ch.isalnum()).upper()
    last6 = alnums[-6:] if len(alnums) >= 6 else alnums
    ticket = f"{prefix}{date_yyyymmdd}-{last6}"
    if counter:
        ticket += f"-{counter}"
    return ticket

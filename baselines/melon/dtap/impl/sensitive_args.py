from __future__ import annotations


REFERENCE_FOCUS: dict[str, list[str]] = {
    "send_email": ["recipients"],
    "send_money": ["recipient", "amount"],
}

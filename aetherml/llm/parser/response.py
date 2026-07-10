"""LLM response parser — defensive parsing, display text only.

This module validates and cleans LLM responses.  It treats all response
content as plain display text — no eval(), exec(), or code execution.
The response is never passed to any function that could interpret it as
code.

Design:
- Defensive parsing: if the response is unexpected format, return it
  as-is rather than crashing.
- No code execution: the response is never eval'd, exec'd, or passed
  to any interpreter.
- Sanitization: strip any potential control characters that could cause
  rendering issues in Markdown.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def parse_response(raw_response: str) -> str:
    """Parse and clean an LLM response for display.

    Args:
        raw_response: The raw text response from the LLM.

    Returns:
        A cleaned string safe for Markdown display.
    """
    if not raw_response or not isinstance(raw_response, str):
        logger.warning("LLM returned empty or non-string response.")
        return ""

    # Strip leading/trailing whitespace
    cleaned = raw_response.strip()

    # Remove potential control characters (except newlines and tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Limit response length to prevent runaway output
    max_length = 5000
    if len(cleaned) > max_length:
        logger.warning(
            "LLM response truncated from %d to %d characters.",
            len(cleaned),
            max_length,
        )
        cleaned = cleaned[:max_length]

    return cleaned


def validate_response_is_text(response: str) -> bool:
    """Validate that a response contains only safe display text.

    This is a safety check — the response should never contain
    code, HTML script tags, or other potentially dangerous content
    that could be interpreted as instructions.

    Returns True if the response is safe, False otherwise.
    """
    if not isinstance(response, str):
        return False

    # Check for script tags or code execution patterns
    dangerous_patterns = [
        r"<script",
        r"javascript:",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            logger.warning(
                "LLM response contains potentially dangerous pattern: %s",
                pattern,
            )
            return False

    return True

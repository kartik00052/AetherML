"""LLM integration for AetherML — narrative generation only.

This is the framework's only non-deterministic, external-call-dependent
component.  It wraps a language model (Gemma) to generate natural-language
narrative summaries of pipeline results.

The LLM integration is strictly opt-in and isolated to this package
and ``agents/reporting/``.  No other agent or computational module
imports from or depends on this package.

Public API:
    - ``GemmaClient``: thin API wrapper with timeout and retry.
    - ``build_narrative_prompt()``: secure prompt construction.
    - ``parse_response()``: defensive response parsing.
"""

from aetherml.llm.gemma.client import GemmaClient
from aetherml.llm.parser.response import parse_response
from aetherml.llm.prompts.narrative import build_narrative_prompt

__all__ = [
    "GemmaClient",
    "build_narrative_prompt",
    "parse_response",
]

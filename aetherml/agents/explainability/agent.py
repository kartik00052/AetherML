"""Explainability agent (stub).

Generates model explanations using SHAP, LIME, or feature importance.
Writes to ``state.explanation_report``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="explainability",
    description="Generate model explanations and feature importance reports.",
)


class ExplainabilityAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

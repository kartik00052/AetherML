"""Evaluation agent (stub).

Evaluates trained model performance using standard metrics.
Writes to ``state.evaluation_report``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="evaluation",
    description="Evaluate trained model performance against metrics.",
)


class EvaluationAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

"""Target detection agent (stub).

Automatically identifies the target variable and task type
(classification, regression, etc.).  Writes to ``state.target_column``
and ``state.task_type``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="target_detection",
    description="Identify the target variable and determine the ML task type.",
)


class TargetDetectionAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

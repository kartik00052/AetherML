"""Validation agent (stub).

Validates data schema, types, and quality constraints.
Orchestrates ``data.validators`` and writes results to ``state.validation_report``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="validation",
    description="Validate data schema, types, and quality constraints.",
)


class ValidationAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

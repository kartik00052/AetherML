"""Reporting agent (stub).

Generates final pipeline reports (HTML, PDF, Markdown).
Orchestrates ``reports`` templates and formatters.
Writes to ``state.final_report``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="reporting",
    description="Generate final pipeline reports in various formats.",
)


class ReportingAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

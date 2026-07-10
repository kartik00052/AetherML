"""Engine selection agent (stub).

Workflow-level agent that selects and configures the appropriate
computation engine based on data characteristics.  This is distinct
from ``engines.engine_selector`` (which is the low-level selector);
this agent makes the decision within the LangGraph workflow.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="engine_selection",
    description="Select the best computation engine based on data characteristics.",
)


class EngineSelectionAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

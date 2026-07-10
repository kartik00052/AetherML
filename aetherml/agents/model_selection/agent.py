"""Model selection agent (stub, merged automl).

Selects candidate models and runs automated ML pipeline search.
Writes to ``state.candidate_models``, ``state.best_pipeline``,
and ``state.trained_model``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="model_selection",
    description="Select models and run automated ML pipeline search.",
)


class ModelSelectionAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

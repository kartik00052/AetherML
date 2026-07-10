"""Feature engineering agent (stub).

Orchestrates feature creation using ``ml.feature_engineering``.
Writes to ``state.features`` and ``state.feature_names``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="feature_engineering",
    description="Create new features from processed data.",
)


class FeatureEngineeringAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

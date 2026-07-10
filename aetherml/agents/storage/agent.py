"""Storage agent (stub).

Persists models, data, and artifacts to durable storage.
Orchestrates ``database.mlflow`` for model registry and experiment tracking.
Writes to ``state.artifact_uri``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="storage",
    description="Persist models, data, and artifacts to durable storage.",
)


class StorageAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

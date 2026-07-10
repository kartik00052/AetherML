"""EDA agent (stub, merged profiling).

Performs exploratory data analysis and data profiling.
Orchestrates ``data.profilers`` for statistical profiling and produces
visualisation summaries.  Writes to ``state.data_profile`` and ``state.eda_report``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="eda",
    description="Perform exploratory data analysis and statistical profiling.",
)


class EDAAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

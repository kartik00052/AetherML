"""RAG agent (stub).

Retrieves knowledge from the vector store to augment ML decisions.
Orchestrates ``rag.embeddings``, ``rag.retrieval``, and ``rag.knowledge_base``.
Writes to ``state.rag_context``.

TODO: Implement in the next pass.
"""

from __future__ import annotations

from typing import Any

from aetherml.agents.base import AgentResult, Tool, _StubAgent

_stub = _StubAgent(
    name="rag",
    description="Retrieve knowledge for augmented ML decisions.",
)


class RAGAgent:
    name = _stub.name
    description = _stub.description

    async def run(self, state: Any) -> AgentResult:
        return await _stub.run(state)

    def get_tools(self) -> list[Tool]:
        return _stub.get_tools()

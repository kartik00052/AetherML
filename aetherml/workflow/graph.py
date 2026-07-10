"""Workflow graph — LangGraph StateGraph definition.

Builds the directed graph that orchestrates agent execution.  The graph
is constructed dynamically via ``build_graph()``, which accepts
pre-initialised agents and wires them into the LangGraph topology.

Current topology (linear):
    upload → etl → [end]

Future topology (full pipeline):
    upload → validation → engine_selection → etl → [profiling ∥ eda]
    → feature_engineering → target_detection → model_selection
    → evaluation → explainability → reporting → storage

Design:
- The graph is rebuilt on every ``build_graph()`` call.  This is cheap
  (LangGraph compiles quickly) and keeps the function side-effect-free.
- ``WorkflowState`` is passed as the graph's state schema.
- Conditional edges are defined in ``router.py``; the graph imports
  them by reference.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from aetherml.agents.base import BaseAgent
from aetherml.workflow.nodes import make_node
from aetherml.workflow.router import route_after_etl, route_after_upload
from aetherml.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def build_graph(agents: dict[str, BaseAgent]) -> Any:
    """Build and compile the LangGraph workflow graph.

    Args:
        agents: Mapping of agent name → agent instance.  Must include
            at least ``"upload"`` and ``"etl"``.  Additional agents are
            ignored in the current linear pipeline but will be wired in
            future passes.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for execution.
    """
    graph = StateGraph(WorkflowState)

    # ── Add nodes ───────────────────────────────────────────────────
    upload_agent = agents["upload"]
    etl_agent = agents["etl"]

    graph.add_node("upload", make_node(upload_agent))
    graph.add_node("etl", make_node(etl_agent))

    # ── Wire edges ──────────────────────────────────────────────────
    graph.set_entry_point("upload")
    graph.add_conditional_edges("upload", route_after_upload, {"etl": "etl", "__end__": END})
    graph.add_conditional_edges("etl", route_after_etl, {"__end__": END})

    logger.info("Workflow graph built: upload → etl → end")
    return graph.compile()

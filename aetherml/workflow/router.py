"""Workflow router — conditional routing logic for the LangGraph graph.

This module defines routing functions that determine which node executes
next based on the current workflow state.

Routing functions return generic labels (``"proceed"`` or ``"__end__"``).
``build_graph()`` maps ``"proceed"`` to the actual next stage name based
on which stages are included in the pipeline.  This keeps routing
functions decoupled from the specific graph topology.

Current implementation: linear routing (Upload → ETL → Validation →
EDA → End).

Known future improvements (TODO):
- Add feedback loops: Evaluation → Feature Engineering when metrics are poor.
- Add parallel branching: Profiling + EDA running concurrently.
- Add skip logic: allow users to bypass optional stages (EDA, explainability).
- Add conditional branches: different paths for classification vs regression.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)


def route_after_upload(state: Any) -> Literal["proceed", "__end__"]:
    """Route after the upload node.

    If data was loaded successfully, proceed to the next stage.
    Otherwise, end the workflow.
    """
    if getattr(state, "raw_data", None) is not None:
        logger.info("Upload succeeded — proceeding.")
        return "proceed"
    logger.warning("Upload produced no data — ending workflow.")
    return "__end__"


def route_after_etl(state: Any) -> Literal["proceed", "__end__"]:
    """Route after the ETL node.

    If ETL produced processed data, proceed to the next stage.
    Otherwise, end the workflow.
    """
    if getattr(state, "processed_data", None) is not None:
        logger.info("ETL succeeded — proceeding.")
        return "proceed"
    logger.warning("ETL produced no processed data — ending workflow.")
    return "__end__"


def route_after_validation(state: Any) -> Literal["proceed", "__end__"]:
    """Route after the Validation node.

    If validation produced validated data, proceed to the next stage.
    Otherwise, end the workflow.
    """
    if getattr(state, "validated_data", None) is not None:
        logger.info("Validation succeeded — proceeding.")
        return "proceed"
    logger.warning("Validation produced no validated data — ending workflow.")
    return "__end__"


def route_after_eda(state: Any) -> Literal["proceed", "__end__"]:
    """Route after the EDA node.

    In the current pipeline, EDA is the last implemented stage.
    Future agents will be added here as they are implemented.
    """
    if getattr(state, "data_profile", None) is not None:
        logger.info("EDA succeeded — pipeline complete.")
    else:
        logger.warning("EDA produced no data profile.")
    return "__end__"

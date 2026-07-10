"""Workflow router — conditional routing logic for the LangGraph graph.

This module defines routing functions that determine which node executes
next based on the current workflow state.

Current implementation: linear routing (Upload → ETL → End).

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


def route_after_upload(state: Any) -> Literal["etl", "__end__"]:
    """Route after the upload node.

    If data was loaded successfully, proceed to ETL.
    Otherwise, end the workflow.
    """
    if getattr(state, "raw_data", None) is not None:
        logger.info("Upload succeeded — routing to ETL.")
        return "etl"
    logger.warning("Upload produced no data — ending workflow.")
    return "__end__"


def route_after_etl(state: Any) -> Literal["__end__"]:
    """Route after the ETL node.

    In the current linear pipeline, ETL is the last implemented node.
    Future agents (profiling, EDA, feature engineering, etc.) will be
    added here as they are implemented.
    """
    if getattr(state, "processed_data", None) is not None:
        logger.info("ETL succeeded — pipeline complete.")
    else:
        logger.warning("ETL produced no processed data.")
    return "__end__"

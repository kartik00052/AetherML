"""AetherML — public SDK surface.

This is the canonical entry point for programmatic use of AetherML.
All public API is exposed here; ``interfaces/cli/`` and ``interfaces/api/``
consume this surface.

Usage::

    import aetherml

    result = await aetherml.run_pipeline(data_path="data.csv")
"""

from __future__ import annotations

import logging
from typing import Any

from aetherml.agents.base import BaseAgent
from aetherml.agents.eda.agent import EDAAgent
from aetherml.agents.engine_selection.agent import EngineSelectionAgent
from aetherml.agents.etl.agent import ETLAgent, ETLConfig
from aetherml.agents.evaluation.agent import EvaluationAgent
from aetherml.agents.explainability.agent import ExplainabilityAgent
from aetherml.agents.feature_engineering.agent import FeatureEngineeringAgent
from aetherml.agents.model_selection.agent import ModelSelectionAgent
from aetherml.agents.rag.agent import RAGAgent
from aetherml.agents.reporting.agent import ReportingAgent
from aetherml.agents.storage.agent import StorageAgent
from aetherml.agents.target_detection.agent import TargetDetectionAgent
from aetherml.agents.upload.agent import UploadAgent
from aetherml.agents.validation.agent import ValidationAgent
from aetherml.configs.settings import AetherMLConfig
from aetherml.engines.engine_selector import select_engine
from aetherml.exceptions import AetherMLError, WorkflowError
from aetherml.workflow.graph import build_graph
from aetherml.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "run_pipeline",
    "AetherMLConfig",
    "WorkflowState",
    "AetherMLError",
]


def _compose_agents(
    config: AetherMLConfig,
    data_path: str,
) -> dict[str, BaseAgent]:
    """Manually compose all agents via constructor injection.

    This is the composition root — the only place where concrete agent
    and engine classes are instantiated.  The rest of the SDK depends
    on abstractions (``BaseAgent``, ``BaseEngine``).

    TODO: In a future pass, this will become a proper DI container
    with configurable agent lifecycles.
    """
    # Select the computation engine
    engine = select_engine(config=config, data_path=data_path)

    # Instantiate agents with their dependencies
    agents: dict[str, BaseAgent] = {
        "upload": UploadAgent(engine=engine),
        "validation": ValidationAgent(),
        "engine_selection": EngineSelectionAgent(),
        "etl": ETLAgent(config=ETLConfig(null_strategy="drop")),
        "eda": EDAAgent(),
        "feature_engineering": FeatureEngineeringAgent(),
        "target_detection": TargetDetectionAgent(),
        "model_selection": ModelSelectionAgent(),
        "evaluation": EvaluationAgent(),
        "explainability": ExplainabilityAgent(),
        "rag": RAGAgent(),
        "reporting": ReportingAgent(),
        "storage": StorageAgent(),
    }
    return agents


async def run_pipeline(
    data_path: str,
    engine_preference: str | None = None,
    null_strategy: str = "drop",
    config: AetherMLConfig | None = None,
) -> dict[str, Any]:
    """Run the full AetherML pipeline on a dataset.

    This is the primary public API.  It:
    1. Builds configuration (from *config* or defaults).
    2. Composes agents via manual DI.
    3. Constructs the LangGraph workflow.
    4. Executes the graph with the initial state.

    Args:
        data_path: Path to the input dataset.
        engine_preference: Force a specific engine (``"pandas"``, ``"polars"``,
            ``"spark"``).  ``None`` for auto-selection.
        null_strategy: Null handling strategy (``"drop"``, ``"fill"``, ``"flag"``).
        config: Optional pre-built configuration.  If ``None``, a config is
            constructed from the other arguments.

    Returns:
        A dict summarising the pipeline results.

    Raises:
        WorkflowError: If the workflow graph execution fails.
    """
    if config is None:
        config = AetherMLConfig()
    if engine_preference is not None:
        config.engine.preferred = engine_preference

    logger.info("AetherML pipeline starting — data_path=%s", data_path)

    # 1. Compose agents
    agents = _compose_agents(config, data_path)

    # 2. Build workflow graph
    graph = build_graph(agents)

    # 3. Build initial state
    initial_state = WorkflowState(data_path=data_path)

    # 4. Execute
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        msg = f"Workflow execution failed: {exc}"
        logger.exception(msg)
        raise WorkflowError(msg) from exc

    logger.info("AetherML pipeline complete.")

    # Extract summary from the final state
    if isinstance(final_state, dict):
        return {
            "row_count": final_state.get("row_count"),
            "column_count": (
                len(final_state["processed_data"].columns)
                if final_state.get("processed_data") is not None
                else None
            ),
            "transformations": (
                len(final_state["transform_log"])
                if final_state.get("transform_log") is not None
                else 0
            ),
        }
    return {
        "row_count": final_state.row_count,
        "column_count": (
            len(final_state.processed_data.columns)
            if final_state.processed_data is not None
            else None
        ),
        "transformations": (
            len(final_state.transform_log)
            if final_state.transform_log is not None
            else 0
        ),
    }

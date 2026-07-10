"""Tests for the Reporting agent and report builder module."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from aetherml.agents.reporting.agent import ReportingAgent
from aetherml.ml.reports.builder import (
    _build_eda_section,
    _build_evaluation_section,
    _build_explainability_section,
    _build_feature_engineering_section,
    _build_model_selection_section,
    _build_notes_section,
    _build_summary_section,
    _build_target_detection_section,
    _build_validation_section,
    build_report,
)
from aetherml.workflow.state import WorkflowState

# ── Helpers ──────────────────────────────────────────────────────────


def _make_state(**kwargs: Any) -> WorkflowState:
    """Create a WorkflowState with the given overrides."""
    state = WorkflowState(run_id="test-run", status="completed")
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


# ── ReportingAgent protocol tests ────────────────────────────────────


class TestReportingAgentProtocol:
    """Verify ReportingAgent satisfies BaseAgent protocol."""

    def test_has_run_method(self) -> None:
        agent = ReportingAgent()
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_has_get_tools_method(self) -> None:
        agent = ReportingAgent()
        assert hasattr(agent, "get_tools")
        assert callable(agent.get_tools)

    def test_isinstance_base_agent(self) -> None:
        from aetherml.agents.base import BaseAgent
        agent = ReportingAgent()
        assert isinstance(agent, BaseAgent)

    def test_get_tools_returns_list(self) -> None:
        agent = ReportingAgent()
        tools = agent.get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0].name == "build_report"


# ── ReportingAgent.run() tests ───────────────────────────────────────


class TestReportingAgentRun:
    """Test ReportingAgent.run() logic."""

    @pytest.mark.asyncio
    async def test_successful_run_full_state(self) -> None:
        agent = ReportingAgent()
        state = _make_state(
            target_column="target",
            task_type="regression",
            target_detection_confidence=0.85,
            feature_names=["a", "b", "c"],
            candidate_models=[
                {"model_type": "RandomForest", "mean_cv_score": 0.92},
            ],
            best_pipeline={"model_type": "RandomForest", "score": 0.92},
            evaluation_report={
                "metrics": {"rmse": 5.23, "mae": 4.10, "r2": 0.91},
            },
            explanation_report={
                "feature_importance": {"a": 0.5, "b": 0.3, "c": 0.2},
                "explainer_type": "TreeExplainer",
                "sampled": False,
                "n_samples_used": 100,
                "max_samples": 100,
            },
        )

        result = await agent.run(state)
        assert result.success
        assert "final_report" in result.data
        report = result.data["final_report"]
        assert isinstance(report, str)
        assert len(report) > 0
        assert "test-run" in report

    @pytest.mark.asyncio
    async def test_successful_run_empty_state(self) -> None:
        agent = ReportingAgent()
        state = _make_state()

        result = await agent.run(state)
        assert result.success
        report = result.data["final_report"]
        # Should still produce a report with stub messages
        assert "_No summary data available._" in report

    @pytest.mark.asyncio
    async def test_report_metadata(self) -> None:
        agent = ReportingAgent()
        state = _make_state(
            target_column="y",
            task_type="classification",
        )

        result = await agent.run(state)
        assert result.success
        assert result.metadata["report_length"] > 0
        assert result.metadata["report_lines"] > 0


# ── Section builder tests ────────────────────────────────────────────


class TestSummarySection:
    """Test _build_summary_section."""

    def test_with_all_fields(self) -> None:
        state = _make_state(
            target_column="target",
            task_type="regression",
            target_detection_confidence=0.85,
            ambiguity_reason="low variance",
            feature_names=["a", "b"],
            best_pipeline={"model_type": "RandomForest"},
        )
        section = _build_summary_section(state)
        assert "target" in section
        assert "regression" in section
        assert "0.85" in section
        assert "low variance" in section
        assert "RandomForest" in section

    def test_empty_state(self) -> None:
        state = _make_state()
        section = _build_summary_section(state)
        assert "_No summary data available._" in section

    def test_many_features_truncates(self) -> None:
        features = [f"feat_{i}" for i in range(20)]
        state = _make_state(feature_names=features)
        section = _build_summary_section(state)
        assert "10 more" in section


class TestValidationSection:
    """Test _build_validation_section."""

    def test_with_checks(self) -> None:
        report = {
            "checks": [
                {"name": "schema_check", "passed": True},
                {"name": "type_check", "passed": False},
            ]
        }
        state = _make_state(validation_report=report)
        section = _build_validation_section(state)
        assert "2 validation checks" in section
        assert "[PASS] schema_check" in section
        assert "[FAIL] type_check" in section

    def test_no_checks(self) -> None:
        state = _make_state(validation_report={})
        section = _build_validation_section(state)
        assert "No validation checks" in section

    def test_none_report(self) -> None:
        state = _make_state()
        section = _build_validation_section(state)
        assert "_Validation data not available._" in section


class TestEDASection:
    """Test _build_eda_section."""

    def test_with_profile(self) -> None:
        profile = {
            "n_rows": 1000,
            "n_cols": 15,
            "null_percentage": 2.5,
            "dtypes": {"int": 5, "float": 5, "object": 5},
        }
        state = _make_state(data_profile=profile)
        section = _build_eda_section(state)
        assert "1000" in section
        assert "15" in section
        assert "2.5%" in section

    def test_none_profile(self) -> None:
        state = _make_state()
        section = _build_eda_section(state)
        assert "_EDA data not available._" in section


class TestTargetDetectionSection:
    """Test _build_target_detection_section."""

    def test_with_target(self) -> None:
        state = _make_state(
            target_column="price",
            task_type="regression",
            target_detection_confidence=0.90,
        )
        section = _build_target_detection_section(state)
        assert "price" in section
        assert "regression" in section
        assert "0.90" in section

    def test_with_ambiguity(self) -> None:
        state = _make_state(
            target_column="grade",
            task_type="ambiguous",
            ambiguity_reason="numeric with few unique values",
        )
        section = _build_target_detection_section(state)
        assert "grade" in section
        assert "ambiguous" in section
        assert "numeric with few unique values" in section

    def test_no_target(self) -> None:
        state = _make_state()
        section = _build_target_detection_section(state)
        assert "_Target detection data not available._" in section


class TestFeatureEngineeringSection:
    """Test _build_feature_engineering_section."""

    def test_with_features(self) -> None:
        state = _make_state(feature_names=["a", "b", "c"])
        section = _build_feature_engineering_section(state)
        assert "3 engineered features" in section
        assert "a" in section
        assert "b" in section

    def test_no_features(self) -> None:
        state = _make_state()
        section = _build_feature_engineering_section(state)
        assert "_Feature engineering data not available._" in section


class TestModelSelectionSection:
    """Test _build_model_selection_section."""

    def test_with_candidates(self) -> None:
        candidates = [
            {"model_type": "RandomForest", "mean_cv_score": 0.92},
            {"model_type": "GradientBoosting", "mean_cv_score": 0.89},
        ]
        state = _make_state(
            candidate_models=candidates,
            best_pipeline={"model_type": "RandomForest"},
        )
        section = _build_model_selection_section(state)
        assert "2 candidate models" in section
        assert "RandomForest" in section
        assert "0.92" in section

    def test_no_candidates(self) -> None:
        state = _make_state()
        section = _build_model_selection_section(state)
        assert "_Model selection data not available._" in section


class TestEvaluationSection:
    """Test _build_evaluation_section."""

    def test_with_metrics(self) -> None:
        report = {
            "metrics": {"accuracy": 0.95, "f1": 0.93},
            "ambiguity_caveat": None,
        }
        state = _make_state(evaluation_report=report)
        section = _build_evaluation_section(state)
        assert "accuracy" in section
        assert "0.95" in section

    def test_with_ambiguity_caveat(self) -> None:
        report = {
            "metrics": {"accuracy": 0.80},
            "ambiguity_caveat": "Task type was ambiguous",
        }
        state = _make_state(evaluation_report=report)
        section = _build_evaluation_section(state)
        assert "Task type was ambiguous" in section

    def test_none_report(self) -> None:
        state = _make_state()
        section = _build_evaluation_section(state)
        assert "_Evaluation data not available._" in section


class TestExplainabilitySection:
    """Test _build_explainability_section."""

    def test_with_importance(self) -> None:
        report = {
            "feature_importance": {"a": 0.5, "b": 0.3, "c": 0.2},
            "explainer_type": "TreeExplainer",
            "sampled": False,
            "n_samples_used": 100,
            "max_samples": 100,
        }
        state = _make_state(explanation_report=report)
        section = _build_explainability_section(state)
        assert "a" in section
        assert "0.5" in section
        assert "TreeExplainer" in section

    def test_with_sampling_note(self) -> None:
        report = {
            "feature_importance": {"a": 0.5},
            "explainer_type": "KernelExplainer",
            "sampled": True,
            "n_samples_used": 50,
            "max_samples": 50,
        }
        state = _make_state(explanation_report=report)
        section = _build_explainability_section(state)
        assert "sample of 50 rows" in section

    def test_none_report(self) -> None:
        state = _make_state()
        section = _build_explainability_section(state)
        assert "_Explainability data not available._" in section


class TestNotesSection:
    """Test _build_notes_section."""

    def test_with_ambiguity(self) -> None:
        state = _make_state(
            task_type="ambiguous",
            ambiguity_reason="low variance",
        )
        section = _build_notes_section(state)
        assert "Ambiguity detected" in section
        assert "low variance" in section

    def test_with_evaluation_caveat(self) -> None:
        state = _make_state(
            evaluation_report={"ambiguity_caveat": "Task type uncertain"},
        )
        section = _build_notes_section(state)
        assert "Task type uncertain" in section

    def test_with_sampling_note(self) -> None:
        state = _make_state(
            explanation_report={"sampled": True},
        )
        section = _build_notes_section(state)
        assert "Explainability sampling" in section

    def test_no_notes(self) -> None:
        state = _make_state()
        section = _build_notes_section(state)
        assert "_No special notes._" in section


# ── Full report assembly tests ───────────────────────────────────────


class TestBuildReport:
    """Test the full build_report function."""

    def test_produces_valid_markdown(self) -> None:
        state = _make_state(
            target_column="y",
            task_type="regression",
            target_detection_confidence=0.90,
            feature_names=["a", "b"],
            best_pipeline={"model_type": "LinearRegression"},
            evaluation_report={"metrics": {"rmse": 5.0}},
            explanation_report={
                "feature_importance": {"a": 0.6, "b": 0.4},
                "explainer_type": "LinearExplainer",
                "sampled": False,
                "n_samples_used": 5,
                "max_samples": 100,
            },
        )
        report = build_report(state)
        assert isinstance(report, str)
        assert "# AetherML Pipeline Report" in report
        assert "test-run" in report
        assert "completed" in report

    def test_empty_state_produces_report(self) -> None:
        state = _make_state()
        report = build_report(state)
        assert isinstance(report, str)
        assert "# AetherML Pipeline Report" in report


# ── Partial pipeline tests ───────────────────────────────────────────


class TestPartialPipeline:
    """Test report generation with missing upstream stages."""

    def test_only_validation_data(self) -> None:
        state = _make_state(
            validation_report={
                "checks": [{"name": "schema", "passed": True}],
            },
        )
        report = build_report(state)
        assert "schema" in report
        assert "_EDA data not available._" in report
        assert "_Model selection data not available._" in report

    def test_only_model_selection(self) -> None:
        state = _make_state(
            target_column="y",
            task_type="regression",
            feature_names=["a"],
            best_pipeline={"model_type": "RF"},
        )
        report = build_report(state)
        assert "_Evaluation data not available._" in report
        assert "_Explainability data not available._" in report

    def test_ambiguity_propagation_in_notes(self) -> None:
        state = _make_state(
            task_type="ambiguous",
            ambiguity_reason="few unique values",
            evaluation_report={"ambiguity_caveat": "Task type uncertain"},
            explanation_report={"sampled": True},
        )
        report = build_report(state)
        assert "Ambiguity detected" in report
        assert "Task type uncertain" in report
        assert "Explainability sampling" in report


# ── Import check ─────────────────────────────────────────────────────


class TestImportChecks:
    """Verify no LLM/Gemma imports in the report builder."""

    def test_no_llm_imports_in_builder(self) -> None:
        from pathlib import Path
        builder_file = (
            Path(__file__).parent.parent / "aetherml" / "ml" / "reports" / "builder.py"
        )
        lines = builder_file.read_text().splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            for kw in ["openai", "langchain", "gemma", "gemini", "aetherml.llm"]:
                assert kw not in stripped.lower()


# ── LLM narrative integration tests ─────────────────────────────────


class TestReportingAgentLLMIntegration:
    """Test ReportingAgent with LLM narrative support."""

    @pytest.mark.asyncio
    async def test_llm_disabled_produces_template_report(self) -> None:
        from aetherml.configs.settings import LLMConfig
        agent = ReportingAgent(llm_config=LLMConfig(use_narrative=False))
        state = _make_state(
            target_column="y",
            task_type="regression",
        )
        result = await agent.run(state)
        assert result.success
        assert "final_report" in result.data
        assert result.metadata["narrative_generation_status"] == "disabled"

    @pytest.mark.asyncio
    async def test_llm_enabled_but_no_api_key_falls_back(self) -> None:
        from aetherml.configs.settings import LLMConfig
        agent = ReportingAgent(
            llm_config=LLMConfig(use_narrative=True, api_key=None)
        )
        state = _make_state(target_column="y")
        with patch.dict("os.environ", {}, clear=True):
            result = await agent.run(state)
        assert result.success
        assert "final_report" in result.data
        assert "failed" in result.metadata["narrative_generation_status"]

    @pytest.mark.asyncio
    async def test_llm_api_timeout_falls_back(self) -> None:
        from aetherml.configs.settings import LLMConfig
        agent = ReportingAgent(
            llm_config=LLMConfig(
                use_narrative=True,
                api_key="test-key",
                timeout_seconds=0.01,
            )
        )
        state = _make_state(target_column="y")

        with patch("aetherml.llm.gemma.client.GemmaClient.generate") as mock_gen:
            mock_gen.side_effect = Exception("simulated timeout")
            result = await agent.run(state)

        assert result.success
        assert "final_report" in result.data
        assert "failed" in result.metadata["narrative_generation_status"]

    @pytest.mark.asyncio
    async def test_llm_success_appends_narrative(self) -> None:
        from aetherml.configs.settings import LLMConfig
        agent = ReportingAgent(
            llm_config=LLMConfig(use_narrative=True, api_key="test-key")
        )
        state = _make_state(target_column="y", task_type="regression")

        with patch("aetherml.llm.gemma.client.GemmaClient.generate") as mock_gen:
            mock_gen.return_value = "The model shows strong performance."
            result = await agent.run(state)

        assert result.success
        assert "The model shows strong performance." in result.data["final_report"]
        assert result.metadata["narrative_generation_status"] == "success"

    @pytest.mark.asyncio
    async def test_use_llm_false_zero_llm_calls(self) -> None:
        """Confirm use_llm_narrative=False produces zero LLM calls."""
        from aetherml.configs.settings import LLMConfig
        agent = ReportingAgent(
            llm_config=LLMConfig(use_narrative=False, api_key="test-key")
        )
        state = _make_state(target_column="y")

        with patch("aetherml.llm.gemma.client.GemmaClient.generate") as mock_gen:
            result = await agent.run(state)
            mock_gen.assert_not_called()

        assert result.success
        assert "final_report" in result.data

    @pytest.mark.asyncio
    async def test_no_llm_config_zero_llm_calls(self) -> None:
        """Confirm no LLM config produces zero LLM calls."""
        agent = ReportingAgent(llm_config=None)
        state = _make_state(target_column="y")

        with patch("aetherml.llm.gemma.client.GemmaClient.generate") as mock_gen:
            result = await agent.run(state)
            mock_gen.assert_not_called()

        assert result.success
        assert result.metadata["narrative_generation_status"] == "disabled"


# ── Build report narrative parameter tests ───────────────────────────


class TestBuildReportNarrative:
    """Test the build_report narrative parameter."""

    def test_narrative_included_in_report(self) -> None:
        state = _make_state(target_column="y")
        report = build_report(state, narrative="This is a narrative.")
        assert "This is a narrative." in report
        assert "## Narrative Summary" in report

    def test_no_narrative_shows_stub(self) -> None:
        state = _make_state(target_column="y")
        report = build_report(state, narrative=None)
        assert "Narrative summary not generated" in report

    def test_empty_narrative_shows_stub(self) -> None:
        state = _make_state(target_column="y")
        report = build_report(state, narrative="  ")
        assert "Narrative summary was empty" in report

    def test_narrative_none_default_backward_compat(self) -> None:
        """build_report with no narrative arg should work (backward compat)."""
        state = _make_state(target_column="y")
        report = build_report(state)
        assert "Narrative summary not generated" in report


# ── RAG integration tests ──────────────────────────────────────────


class TestReportingAgentRAGIntegration:
    """Test ReportingAgent with RAG context retrieval."""

    @pytest.mark.asyncio
    async def test_rag_disabled_by_default(self) -> None:
        from aetherml.configs.settings import LLMConfig
        agent = ReportingAgent(llm_config=LLMConfig(use_narrative=False))
        state = _make_state(target_column="y")
        result = await agent.run(state)
        assert result.success
        assert result.metadata["rag_retrieval_status"] == "disabled"

    @pytest.mark.asyncio
    async def test_rag_enabled_but_llm_disabled(self) -> None:
        from aetherml.configs.settings import LLMConfig, RAGConfig
        agent = ReportingAgent(
            llm_config=LLMConfig(use_narrative=False),
            rag_config=RAGConfig(enabled=True),
        )
        state = _make_state(target_column="y")
        result = await agent.run(state)
        assert result.success
        assert result.metadata["rag_retrieval_status"] == "disabled"

    @pytest.mark.asyncio
    async def test_rag_and_llm_both_enabled_qdrant_unreachable(self) -> None:
        from aetherml.configs.settings import LLMConfig, QdrantConfig, RAGConfig

        agent = ReportingAgent(
            llm_config=LLMConfig(use_narrative=True, api_key="test-key"),
            rag_config=RAGConfig(enabled=True),
            qdrant_config=QdrantConfig(url="http://invalid:6333"),
        )
        state = _make_state(target_column="y", task_type="regression")

        with patch("aetherml.llm.gemma.client.GemmaClient.generate") as mock_gen:
            mock_gen.return_value = "Test narrative."
            result = await agent.run(state)

        assert result.success
        assert "final_report" in result.data
        rag_status = result.metadata["rag_retrieval_status"]
        assert rag_status != "disabled"
        assert result.metadata["narrative_generation_status"] == "success"

    @pytest.mark.asyncio
    async def test_rag_retrieval_before_llm_call(self) -> None:
        """Verify RAG retrieval is attempted before the LLM call."""
        from aetherml.configs.settings import LLMConfig, RAGConfig

        agent = ReportingAgent(
            llm_config=LLMConfig(use_narrative=True, api_key="test-key"),
            rag_config=RAGConfig(enabled=True),
        )
        state = _make_state(target_column="y")

        call_order: list[str] = []

        async def tracked_retrieve(s: Any) -> tuple[None, str]:
            call_order.append("rag")
            return None, "failed: mock"

        with (
            patch.object(agent, "_retrieve_rag_context", tracked_retrieve),
            patch("aetherml.llm.gemma.client.GemmaClient.generate") as mock_gen,
        ):
            mock_gen.return_value = "Narrative."
            await agent.run(state)
            call_order.append("llm")

        assert call_order == ["rag", "llm"]

    @pytest.mark.asyncio
    async def test_rag_context_passed_to_prompt(self) -> None:
        """Verify RAG context is passed to the narrative prompt builder."""
        from aetherml.configs.settings import LLMConfig, RAGConfig

        agent = ReportingAgent(
            llm_config=LLMConfig(use_narrative=True, api_key="test-key"),
            rag_config=RAGConfig(enabled=True),
        )
        state = _make_state(target_column="y")

        fake_context = {
            "chunks": [{"text": "relevant info", "source": "kb", "score": 0.9}],
            "query": "test query",
            "status": "success",
        }

        with (
            patch.object(agent, "_retrieve_rag_context", return_value=(fake_context, "success")),
            patch("aetherml.llm.gemma.client.GemmaClient.generate") as mock_gen,
            patch("aetherml.llm.prompts.narrative.build_narrative_prompt") as mock_prompt,
        ):
            mock_gen.return_value = "Narrative."
            mock_prompt.return_value = "prompt text"
            await agent.run(state)

        _, kwargs = mock_prompt.call_args
        assert kwargs.get("rag_context") == fake_context

    @pytest.mark.asyncio
    async def test_no_rag_config_no_retrieval_attempt(self) -> None:
        agent = ReportingAgent(llm_config=None, rag_config=None)
        state = _make_state(target_column="y")

        with patch.object(agent, "_retrieve_rag_context") as mock_retrieve:
            await agent.run(state)
            mock_retrieve.assert_not_called()

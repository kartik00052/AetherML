"""Tests for the AetherML REST API (FastAPI)."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from aetherml import __version__
from aetherml.interfaces.api.app import app
from aetherml.interfaces.api.models import (
    ErrorResponse,
    HealthResponse,
    PipelineRequest,
    UploadPipelineResponse,
)


@pytest.fixture()
def client() -> Any:
    """Create a test client from the FastAPI app."""
    from starlette.testclient import TestClient

    return TestClient(app)


# ── Health endpoint ──────────────────────────────────────────────


class TestHealthEndpoint:
    """GET /health"""

    def test_returns_200(self, client: Any) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_version_matches_sdk(self, client: Any) -> None:
        body = client.get("/health").json()
        assert body["version"] == __version__
        assert body["status"] == "ok"

    def test_python_field_present(self, client: Any) -> None:
        body = client.get("/health").json()
        assert "python" in body
        assert isinstance(body["python"], str)

    def test_platform_field_present(self, client: Any) -> None:
        body = client.get("/health").json()
        assert "platform" in body
        assert body["platform"] in ("windows", "linux", "darwin")

    def test_engines_field_present(self, client: Any) -> None:
        body = client.get("/health").json()
        assert "engines" in body
        assert isinstance(body["engines"], list)
        assert "pandas" in body["engines"]


# ── Pipeline endpoint ────────────────────────────────────────────


class TestPipelineEndpoint:
    """POST /pipeline"""

    def test_missing_data_path_returns_422(self, client: Any) -> None:
        resp = client.post("/pipeline", json={})
        assert resp.status_code == 422

    def test_empty_data_path_returns_422(self, client: Any) -> None:
        resp = client.post("/pipeline", json={"data_path": ""})
        assert resp.status_code == 422

    def test_rejects_extra_fields(self, client: Any) -> None:
        resp = client.post(
            "/pipeline",
            json={"data_path": "dummy.csv", "unknown_field": True},
        )
        assert resp.status_code == 422

    def test_validates_enum_null_strategy(self, client: Any) -> None:
        resp = client.post(
            "/pipeline",
            json={"data_path": "dummy.csv", "null_strategy": "bogus"},
        )
        assert resp.status_code == 422

    def test_validates_enum_engine(self, client: Any) -> None:
        resp = client.post(
            "/pipeline",
            json={"data_path": "dummy.csv", "engine": "bogus"},
        )
        assert resp.status_code == 422

    def test_validates_invalid_stage(self, client: Any) -> None:
        resp = client.post(
            "/pipeline",
            json={"data_path": "dummy.csv", "stages": ["nonexistent_stage"]},
        )
        assert resp.status_code == 422

    def test_valid_stages_accepted(self, client: Any) -> None:
        mock_result = {"row_count": 100}
        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(
                "/pipeline",
                json={"data_path": "dummy.csv", "stages": ["upload", "etl"]},
            )
            assert resp.status_code == 200

    def test_success_returns_200(self, client: Any) -> None:
        mock_result = {"row_count": 100, "column_count": 5}
        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post("/pipeline", json={"data_path": "dummy.csv"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["result"]["row_count"] == 100

    def test_workflow_error_returns_500(self, client: Any) -> None:
        from aetherml.exceptions import WorkflowError

        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            side_effect=WorkflowError("bad graph"),
        ):
            resp = client.post("/pipeline", json={"data_path": "dummy.csv"})
            assert resp.status_code == 500
            body = resp.json()
            assert body["error_type"] == "WorkflowError"

    def test_config_error_returns_422(self, client: Any) -> None:
        from aetherml.exceptions import ConfigurationError

        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            side_effect=ConfigurationError("bad config"),
        ):
            resp = client.post("/pipeline", json={"data_path": "dummy.csv"})
            assert resp.status_code == 422
            body = resp.json()
            assert body["error_type"] == "ConfigurationError"

    def test_generic_aetherml_error_returns_422(self, client: Any) -> None:
        from aetherml.exceptions import AetherMLError

        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            side_effect=AetherMLError("something broke"),
        ):
            resp = client.post("/pipeline", json={"data_path": "dummy.csv"})
            assert resp.status_code == 422
            body = resp.json()
            assert body["error_type"] == "AetherMLError"

    def test_unhandled_error_returns_500(self, client: Any) -> None:
        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected"),
        ):
            resp = client.post("/pipeline", json={"data_path": "dummy.csv"})
            assert resp.status_code == 500
            body = resp.json()
            assert body["error_type"] == "InternalServerError"


# ── Upload endpoint ──────────────────────────────────────────────


class TestUploadEndpoint:
    """POST /pipeline/upload"""

    def test_missing_file_returns_422(self, client: Any) -> None:
        resp = client.post("/pipeline/upload", data={})
        assert resp.status_code == 422

    def test_success_returns_200(self, client: Any) -> None:
        mock_result = {"row_count": 50}
        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            csv_content = b"col1,col2\n1,2\n3,4"
            resp = client.post(
                "/pipeline/upload",
                files={"file": ("data.csv", io.BytesIO(csv_content), "text/csv")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["filename"] == "data.csv"
            assert body["result"]["row_count"] == 50

    def test_upload_with_form_params(self, client: Any) -> None:
        mock_result = {"row_count": 10}
        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            csv_content = b"a,b\n1,2"
            resp = client.post(
                "/pipeline/upload",
                files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
                data={"engine": "pandas", "null_strategy": "fill", "stages": "upload,etl"},
            )
            assert resp.status_code == 200
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs["engine_preference"] == "pandas"
            assert call_kwargs.kwargs["null_strategy"] == "fill"
            assert call_kwargs.kwargs["stages"] == ["upload", "etl"]

    def test_temp_file_cleaned_up(self, client: Any) -> None:
        import os

        captured_paths: list[str] = []

        async def fake_run_pipeline(**kwargs: Any) -> dict[str, Any]:
            captured_paths.append(kwargs["data_path"])
            assert os.path.exists(kwargs["data_path"])
            return {"row_count": 1}

        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            side_effect=fake_run_pipeline,
        ):
            csv_content = b"x,y\n1,2"
            resp = client.post(
                "/pipeline/upload",
                files={"file": ("data.csv", io.BytesIO(csv_content), "text/csv")},
            )
            assert resp.status_code == 200
            for p in captured_paths:
                assert not os.path.exists(p)

    def test_file_not_found_error_returns_404(self, client: Any) -> None:
        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("dataset.csv not found"),
        ):
            resp = client.post("/pipeline", json={"data_path": "dataset.csv"})
            assert resp.status_code == 404
            body = resp.json()
            assert body["error_type"] == "FileNotFoundError"


# ── Middleware ────────────────────────────────────────────────────


class TestMiddleware:
    """Request ID, timing, and CORS headers."""

    def test_request_id_header_present(self, client: Any) -> None:
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_request_id_from_client_preserved(self, client: Any) -> None:
        resp = client.get("/health", headers={"X-Request-ID": "my-custom-id"})
        assert resp.headers["X-Request-ID"] == "my-custom-id"

    def test_timing_header_present(self, client: Any) -> None:
        resp = client.get("/health")
        assert "X-Process-Time" in resp.headers
        assert float(resp.headers["X-Process-Time"]) >= 0

    def test_cors_headers_present(self, client: Any) -> None:
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers


# ── Error response format ────────────────────────────────────────


class TestErrorResponseFormat:
    """All error responses use consistent ErrorResponse format."""

    def test_error_response_has_required_fields(self, client: Any) -> None:
        resp = client.post("/pipeline", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert "error_type" in body or "detail" in body

    def test_workflow_error_format(self, client: Any) -> None:
        from aetherml.exceptions import WorkflowError

        with patch(
            "aetherml.interfaces.api.app.run_pipeline",
            new_callable=AsyncMock,
            side_effect=WorkflowError("graph failed"),
        ):
            resp = client.post("/pipeline", json={"data_path": "x.csv"})
            body = resp.json()
            assert body["status"] == "error"
            assert body["error_type"] == "WorkflowError"
            assert body["message"] == "graph failed"


# ── Pydantic models ─────────────────────────────────────────────


class TestModels:
    """Pydantic request/response models."""

    def test_pipeline_request_defaults(self) -> None:
        req = PipelineRequest(data_path="data.csv")
        assert req.engine is None
        assert req.null_strategy == "drop"
        assert req.stages is None

    def test_pipeline_request_rejects_empty_data_path(self) -> None:
        with pytest.raises(ValidationError):
            PipelineRequest(data_path="")

    def test_pipeline_request_rejects_invalid_stage(self) -> None:
        with pytest.raises(ValidationError):
            PipelineRequest(data_path="data.csv", stages=["bogus"])

    def test_pipeline_request_accepts_valid_stages(self) -> None:
        req = PipelineRequest(
            data_path="data.csv",
            stages=["upload", "etl", "validation"],
        )
        assert req.stages == ["upload", "etl", "validation"]

    def test_health_response_fields(self) -> None:
        resp = HealthResponse(
            status="ok",
            version="0.1.0",
            python="3.11.0",
            platform="linux",
            engines=["pandas", "polars"],
        )
        d = resp.model_dump()
        assert d["status"] == "ok"
        assert d["version"] == "0.1.0"
        assert d["platform"] == "linux"
        assert d["engines"] == ["pandas", "polars"]

    def test_error_response_fields(self) -> None:
        resp = ErrorResponse(error_type="WorkflowError", message="failed")
        d = resp.model_dump()
        assert d["error_type"] == "WorkflowError"
        assert d["message"] == "failed"
        assert d["status"] == "error"

    def test_upload_pipeline_response_fields(self) -> None:
        resp = UploadPipelineResponse(
            status="ok",
            filename="data.csv",
            result={"row_count": 100},
        )
        d = resp.model_dump()
        assert d["filename"] == "data.csv"
        assert d["result"]["row_count"] == 100

"""FastAPI application for AetherML.

Endpoints:
    POST /pipeline          — run the full ML pipeline on a dataset
    POST /pipeline/upload   — upload a file and run the pipeline
    GET  /health            — liveness / version / readiness check
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import tempfile
import time
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aetherml import __version__, run_pipeline
from aetherml.exceptions import AetherMLError, ConfigurationError, WorkflowError
from aetherml.interfaces.api.models import (
    ErrorResponse,
    HealthResponse,
    PipelineRequest,
    PipelineResponse,
    UploadPipelineResponse,
)

logger = logging.getLogger(__name__)

_OPENAPI_TAGS = [
    {
        "name": "system",
        "description": "Health checks and version information.",
    },
    {
        "name": "pipeline",
        "description": "Run the AetherML automated ML pipeline on a dataset.",
    },
]

app = FastAPI(
    title="AetherML",
    description="Automated Machine Learning lifecycle SDK — REST API.",
    version=__version__,
    openapi_tags=_OPENAPI_TAGS,
    license_info={"name": "MIT"},
)


# ── Middleware ────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next: Any) -> JSONResponse:
    """Inject a unique X-Request-ID into every response."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    try:
        response: JSONResponse = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception in request pipeline")
        response = JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_type="InternalServerError",
                message="An unexpected error occurred.",
            ).model_dump(),
        )
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def _timing_middleware(request: Request, call_next: Any) -> JSONResponse:
    """Add X-Process-Time header to every response."""
    start = time.perf_counter()
    try:
        response: JSONResponse = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception in request pipeline")
        response = JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_type="InternalServerError",
                message="An unexpected error occurred.",
            ).model_dump(),
        )
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response


# ── Exception handlers ───────────────────────────────────────────


@app.exception_handler(AetherMLError)
async def _aetherml_error_handler(_req: Request, exc: AetherMLError) -> JSONResponse:
    status_code = 500 if isinstance(exc, WorkflowError) else 422
    body = ErrorResponse(
        error_type=type(exc).__name__,
        message=str(exc),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(ConfigurationError)
async def _config_error_handler(
    _req: Request, exc: ConfigurationError
) -> JSONResponse:
    body = ErrorResponse(
        error_type="ConfigurationError",
        message=str(exc),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(FileNotFoundError)
async def _file_not_found_handler(
    _req: Request, exc: FileNotFoundError
) -> JSONResponse:
    body = ErrorResponse(
        error_type="FileNotFoundError",
        message=str(exc),
    )
    return JSONResponse(status_code=404, content=body.model_dump())


@app.exception_handler(Exception)
async def _unhandled_error_handler(_req: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    body = ErrorResponse(
        error_type="InternalServerError",
        message="An unexpected error occurred.",
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# ── Endpoints ────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness, version, and readiness check."""
    engines: list[str] = ["pandas"]
    try:
        import polars  # noqa: F401

        engines.append("polars")
    except ImportError:
        pass
    try:
        import pyspark  # noqa: F401

        engines.append("spark")
    except ImportError:
        pass

    return HealthResponse(
        status="ok",
        version=__version__,
        python=sys.version,
        platform=platform.system().lower(),
        engines=engines,
    )


@app.post(
    "/pipeline",
    response_model=PipelineResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["pipeline"],
)
async def run_pipeline_endpoint(body: PipelineRequest) -> PipelineResponse:
    """Run the AetherML pipeline on a dataset at a known path."""
    stages: list[str] | None = list(body.stages) if body.stages is not None else None
    result = await run_pipeline(
        data_path=body.data_path,
        engine_preference=body.engine,
        null_strategy=body.null_strategy,
        stages=stages,
    )
    return PipelineResponse(status="ok", result=result)


@app.post(
    "/pipeline/upload",
    response_model=UploadPipelineResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["pipeline"],
)
async def upload_pipeline_endpoint(
    file: UploadFile = File(..., description="Dataset file to upload."),  # noqa: B008
    engine: str | None = Form(None, description="Force engine: pandas, polars, spark."),  # noqa: B008
    null_strategy: str = Form("drop", description="Null handling: drop, fill, flag."),  # noqa: B008
    stages: str | None = Form(None, description="Comma-separated stage names."),  # noqa: B008
) -> UploadPipelineResponse:
    """Upload a file and run the AetherML pipeline on it.

    The uploaded file is saved to a temporary directory, processed,
    and automatically cleaned up after the pipeline completes.
    """
    tmp_dir = tempfile.mkdtemp(prefix="aetherml_")
    filename = file.filename or "uploaded_data"
    tmp_path = os.path.join(tmp_dir, filename)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        parsed_stages: list[str] | None = None
        if stages:
            parsed_stages = [s.strip() for s in stages.split(",") if s.strip()]

        result = await run_pipeline(
            data_path=tmp_path,
            engine_preference=engine,
            null_strategy=null_strategy,
            stages=parsed_stages,
        )
        return UploadPipelineResponse(status="ok", filename=filename, result=result)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            logger.warning("Failed to clean up temp file %s", tmp_path)

# REST API

PhronesisML exposes a FastAPI-based REST API for running the ML pipeline over HTTP. Same SDK underneath, different interface.

!!! info
    Install the API extras first: `pip install phronesisml[api]`

---

## Starting the Server

### With Uvicorn

```bash
uvicorn phronesisml.interfaces.api.app:app --host 0.0.0.0 --port 8000
```

### With Docker

```bash
docker run -p 8000:8000 ghcr.io/kartik00052/phronesisml:v0.2.0
```

### Programmatic

```python
from phronesisml.interfaces.api.app import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Endpoints

### System Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check, version, available engines |
| GET | `/version` | SDK version info |
| GET | `/capabilities` | Supported formats, engines, stages, limits |
| GET | `/docs` | Swagger UI (interactive API docs) |
| GET | `/redoc` | ReDoc (alternative API docs) |

### Pipeline Endpoints

All pipeline endpoints accept a file upload (multipart form) and optional parameters.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Full pipeline → profile |
| POST | `/clean` | Upload + ETL |
| POST | `/validate` | Upload + ETL + Validation |
| POST | `/detect-target` | Upload → Target Detection |
| POST | `/engineer` | Upload → Feature Engineering |
| POST | `/recommend-model` | Upload → Model Selection |
| POST | `/train` | Full pipeline → Trained Model |
| POST | `/evaluate` | Upload → Model Evaluation |
| POST | `/explain` | Upload → SHAP Explainability |
| POST | `/report` | Full pipeline → Markdown Report |

### Job Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs` | List all jobs |
| GET | `/jobs/{job_id}` | Get job status and result |

---

## Parameters

All pipeline endpoints accept these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | *(required)* | Dataset file (multipart upload) |
| `engine` | string | `null` | Force engine: `"pandas"`, `"polars"`, `"spark"` |
| `null_strategy` | string | `"drop"` | Null handling: `"drop"`, `"fill"`, `"flag"` |

---

## cURL Examples

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "0.2.0",
  "engines": ["pandas", "polars"]
}
```

### Analyze a Dataset

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@data.csv" \
  -F "engine=pandas" \
  -F "null_strategy=drop"
```

### Full Training

```bash
curl -X POST http://localhost:8000/train \
  -F "file=@customers.csv" \
  -F "engine=polars" \
  -F "null_strategy=fill"
```

### Check Job Status

```bash
curl http://localhost:8000/jobs/{job_id}
```

---

## Python Requests Examples

### Analyze

```python
import requests

with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        data={"engine": "polars"},
    )
print(response.json())
```

### Train

```python
import requests

with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/train",
        files={"file": f},
        data={"null_strategy": "fill"},
    )
result = response.json()
print(f"Job ID: {result['data']['job_id']}")
```

### Poll for Results

```python
import requests
import time

# Start training
with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/train",
        files={"file": f},
    )
job_id = response.json()["data"]["job_id"]

# Poll until complete
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}")
    data = status.json()
    if data["status"] == "completed":
        print("Training complete!")
        print(data["result"])
        break
    elif data["status"] == "failed":
        print(f"Training failed: {data['error']}")
        break
    time.sleep(1)
```

---

## Response Format

### Success

```json
{
  "success": true,
  "data": {
    "job_id": "abc123",
    "status": "completed",
    "result": { ... }
  }
}
```

### Error

```json
{
  "success": false,
  "error": {
    "code": "UNSUPPORTED_FORMAT",
    "message": "File format '.xyz' is not supported.",
    "documentation": "https://kartik00052.github.io/Phronesisml/troubleshooting/"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNSUPPORTED_FORMAT` | 400 | File format not recognized |
| `FILE_TOO_LARGE` | 413 | File exceeds 2 GB limit |
| `VALIDATION_FAILED` | 422 | Data validation failed |
| `PIPELINE_FAILED` | 500 | Internal pipeline error |
| `JOB_NOT_FOUND` | 404 | Job ID doesn't exist |

---

## Interactive Docs

Once the server is running, visit:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

The Swagger UI lets you:

1. Browse all endpoints
2. See request/response schemas
3. Test endpoints with real data
4. Download OpenAPI spec

---

## Docker Deployment

### Basic

```bash
docker run -p 8000:8000 ghcr.io/kartik00052/phronesisml:v0.2.0
```

### With Volume Mount

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  ghcr.io/kartik00052/phronesisml:v0.2.0
```

### Docker Compose

```yaml
services:
  phronesisml:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PHRONESISML_ENGINE=polars
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## CORS Configuration

The API allows all origins by default. For production, restrict CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Rate Limiting

!!! warning
    The current API does not include rate limiting. For production use, add rate limiting via a reverse proxy (nginx, Traefik) or middleware.

---

## Related

- [Simple API](simple-api.md) — One-liner Python functions
- [Advanced API](advanced-api.md) — Full pipeline control
- [CLI](cli.md) — Command-line interface

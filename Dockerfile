# ── Build stage ──────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# System deps needed at build time (polars, shap native libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY aetherml/__init__.py aetherml/__init__.py

# Install package with API extras only (no dev/test deps in image)
RUN pip install --no-cache-dir --prefix=/install ".[api]"


# ── Runtime stage ────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL maintainer="Kartik Sharma <kartiksharma18852@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/Kartik00052/AetherML"

# Minimal runtime deps (libgomp for sklearn, libstdc++ for polars)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy full source code
WORKDIR /app
COPY . .

# Non-root user
RUN groupadd -r aetherml && useradd -r -g aetherml -d /app aetherml && \
    chown -R aetherml:aetherml /app
USER aetherml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "aetherml.interfaces.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

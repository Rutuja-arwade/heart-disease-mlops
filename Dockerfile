# syntax=docker/dockerfile:1
# Heart disease risk API - multi-stage build producing a slim, non-root image.

ARG PYTHON_VERSION=3.11

# ---- builder: compile the virtualenv once -----------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# ---- runtime -----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG MODEL_VERSION=dev
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="FastAPI service predicting heart disease risk (UCI Cleveland)" \
      org.opencontainers.image.source="https://github.com/Rutuja-arwade/heart-disease-mlops" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${MODEL_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_METADATA_PATH=/app/models/model_metadata.json \
    LOG_LEVEL=INFO \
    PORT=8000 \
    WORKERS=1

# Numeric UID/GID so Kubernetes can enforce runAsNonRoot.
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY models/model.joblib models/model_metadata.json ./models/
COPY sample_request.json ./
RUN mkdir -p /app/logs && chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/ready', timeout=2)" || exit 1

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --no-access-log"]

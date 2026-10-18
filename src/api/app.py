"""FastAPI service exposing the heart disease risk model.

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Endpoints: GET / , /health, /ready, /model/info, /metrics; POST /predict, /predict/batch
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src import __version__, config
from src.api import monitoring as mon
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.predict import load_metadata, predict_records

logger = mon.configure_logging()


class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.metadata: dict = {}
        self.started = time.time()

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    def load(self) -> None:
        import joblib

        model_path = Path(os.getenv("MODEL_PATH", config.MODEL_FILE))
        metadata_path = Path(os.getenv("MODEL_METADATA_PATH", config.MODEL_METADATA_FILE))
        try:
            self.model = joblib.load(model_path)
            self.metadata = load_metadata(metadata_path)
            mon.MODEL_LOADED.set(1)
            mon.MODEL_INFO.info(
                {
                    "version": self.version,
                    "family": str(self.metadata.get("model_family", "unknown")),
                    "mlflow_run_id": str(self.metadata.get("mlflow_run_id", "")),
                }
            )
            logger.info("model loaded", extra={"fields": {"path": str(model_path), "model_version": self.version}})
        except Exception:
            self.model = None
            mon.MODEL_LOADED.set(0)
            logger.exception("model failed to load", extra={"fields": {"path": str(model_path)}})


state = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version=__version__,
    description=(
        "Predicts the presence of heart disease from 13 clinical attributes (UCI Cleveland). "
        "Returns the predicted class, its confidence and a risk band. Prometheus metrics at /metrics."
    ),
    lifespan=lifespan,
)


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    mon.IN_PROGRESS.inc()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        elapsed = time.perf_counter() - start
        mon.IN_PROGRESS.dec()
        endpoint = _endpoint_label(request)
        if endpoint != "/metrics":
            mon.REQUESTS.labels(request.method, endpoint, str(status_code)).inc()
            mon.LATENCY.labels(request.method, endpoint).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "duration_ms": round(elapsed * 1000, 2),
                        "client": request.client.host if request.client else None,
                        "user_agent": request.headers.get("user-agent"),
                    }
                },
            )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Model-Version"] = state.version
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    mon.VALIDATION_ERRORS.labels(_endpoint_label(request)).inc()
    logger.warning(
        "validation error",
        extra={"fields": {"request_id": getattr(request.state, "request_id", None), "errors": exc.errors()}},
    )
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


def _require_model():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.model


@app.get("/", tags=["service"])
def root():
    return {
        "service": "heart-disease-risk-api",
        "version": __version__,
        "model_version": state.version,
        "docs": "/docs",
        "endpoints": ["/health", "/ready", "/model/info", "/predict", "/predict/batch", "/metrics"],
        "hostname": socket.gethostname(),
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    """Liveness probe: the process is up."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        model_version=state.version,
        uptime_seconds=round(time.time() - state.started, 1),
    )


@app.get("/ready", tags=["service"])
def ready():
    """Readiness probe: only route traffic once the model is in memory."""
    _require_model()
    return {"status": "ready", "model_version": state.version}


@app.get("/model/info", tags=["model"])
def model_info():
    _require_model()
    keys = [
        "model_name",
        "model_family",
        "model_version",
        "mlflow_run_id",
        "trained_at",
        "decision_threshold",
        "features",
        "best_params",
        "cv_metrics",
        "test_metrics",
    ]
    return {k: state.metadata.get(k) for k in keys}


def _score(request: Request, records: list[dict]) -> list[dict]:
    model = _require_model()
    start = time.perf_counter()
    results = predict_records(model, records, state.threshold)
    mon.INFERENCE_LATENCY.observe(time.perf_counter() - start)
    for record, result in zip(records, results, strict=False):
        mon.PREDICTIONS.labels(result["label"], result["risk_level"]).inc()
        mon.PROBABILITY.observe(result["probability_disease"])
        logger.info(
            "prediction",
            extra={
                "fields": {
                    "request_id": request.state.request_id,
                    "model_version": state.version,
                    "prediction": result["prediction"],
                    "probability_disease": result["probability_disease"],
                    "risk_level": result["risk_level"],
                    "features": record,
                }
            },
        )
    return results


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(patient: PatientFeatures, request: Request):
    """Score one patient. Returns prediction (0/1), confidence and risk level."""
    result = _score(request, [patient.model_dump()])[0]
    return PredictionResponse(
        request_id=request.state.request_id, threshold=state.threshold, model_version=state.version, **result
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["model"])
def predict_batch(batch: BatchRequest, request: Request):
    """Score up to 500 patients in one call."""
    results = _score(request, [r.model_dump() for r in batch.records])
    return BatchResponse(
        request_id=request.state.request_id, model_version=state.version, count=len(results), predictions=results
    )


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

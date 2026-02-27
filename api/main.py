"""FastAPI application for the LGD prediction service.

The API surface here is minimal by design: four endpoints that cover the
prediction use case, the operational use case, and the monitoring use case.
Each endpoint exists because it satisfies a distinct requirement that has been
specified — the prediction endpoints for CECL reserve estimation and stress testing,
the health endpoint for container orchestration, the model info endpoint for audit
and governance review, and the monitoring endpoint for ongoing model risk management.

The lifespan context manager handles model loading at startup rather than on first
request. This is the correct choice because the model loading time (disk I/O plus
checkpoint deserialization) would add 200-500ms to the first request in a cold-start
scenario, which would violate the latency target and produce misleading p99 latency
measurements. The startup cost is paid once when the container starts, which is the
right trade-off for a service with a defined warm-up period in the deployment lifecycle.

The model_loaded flag is set in the lifespan handler rather than inferred from whether
_model is None, because the flag needs to be accessible at the endpoint level without
importing the predict module's internal state. This creates a small risk of inconsistency
(model_loaded=True but _model is actually None) that could occur if the predict module
is reloaded. In practice, with a single-process FastAPI server under uvicorn, this
inconsistency does not occur.

Environment variable overrides for the three paths (LGD_MODEL_PATH, LGD_SCALER_PATH,
LGD_DB_PATH) allow the same container image to be deployed in different environments
without rebuilding. Development uses the defaults from configs/default.yaml; staging
and production can override via environment variables set by the orchestration system.
This is a pragmatic concession to deployment reality that does not compromise the
principle that paths should come from configuration rather than being hardcoded.

The training_means dict in monitoring_summary() is the weakest point in the monitoring
architecture. These values should be loaded from a JSON artifact produced at training
time rather than hardcoded here — a hardcoded dict will drift out of sync when the
model is retrained on different data. The correct fix is to have train.py save the
training feature means alongside the model checkpoint and have the API load them
from that artifact. This is documented as a known limitation in docs/model_card.md.

Endpoints:
    POST /predict              — Single loan LGD prediction with MC Dropout CI
    POST /predict/batch        — Batch prediction for up to 1,000 loans
    GET  /health               — Service health and model loaded status
    GET  /model/info           — Model architecture and training metrics
    GET  /monitoring/summary   — Input distribution drift statistics

Usage:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import monitoring as mon
from api import predict as pred
from api.schemas import (
    BatchLGDPrediction,
    BatchLoanFeatures,
    HealthResponse,
    LGDPrediction,
    LoanFeatures,
    ModelInfoResponse,
    MonitoringSummaryResponse,
)

# ---------------------------------------------------------------------------
# Configuration
#
# Paths are read from environment variables with fallbacks to the defaults
# in configs/default.yaml. The environment variable names use a LGD_ prefix
# to prevent collisions with other services that might set MODEL_PATH or DB_PATH
# in the same environment.
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("LGD_MODEL_PATH", "models/lgd_model.pt")
SCALER_PATH = os.environ.get("LGD_SCALER_PATH", "models/scaler.pt")
DB_PATH = os.environ.get("LGD_DB_PATH", "api/predictions.db")
MODEL_VERSION = pred.MODEL_VERSION

_prediction_logger: mon.PredictionLogger | None = None
_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading at startup and cleanup at shutdown.

    FastAPI's lifespan context manager replaces the deprecated @app.on_event("startup")
    pattern and provides a cleaner way to handle startup and shutdown logic. Everything
    before the yield runs at startup; everything after runs at shutdown. Currently
    the shutdown phase has no cleanup logic — SQLite connections are closed by the
    context manager in each database operation, and the model is held in memory until
    the process terminates.

    The FileNotFoundError on model loading is caught and logged rather than re-raised,
    allowing the API to start in a degraded state where /health reports model_loaded=False.
    This is the correct behavior for a container that may start before the model artifact
    is available — the container orchestrator will route traffic based on the health
    check, and the service will accept traffic once the model is loaded on a subsequent
    startup cycle.
    """
    global _prediction_logger, _model_loaded

    _prediction_logger = mon.PredictionLogger(DB_PATH)

    try:
        pred.load_model(MODEL_PATH, SCALER_PATH)
        _model_loaded = True
    except FileNotFoundError:
        _model_loaded = False

    yield

    # Shutdown: currently no cleanup required.


app = FastAPI(
    title="LGD Prediction API",
    description=(
        "Predict Loss Given Default (LGD) for residential mortgage loans "
        "using a PyTorch MLP trained on Freddie Mac SFLP data (2010–2015 vintages). "
        "For use in CECL reserve estimation and macro stress testing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Convert RuntimeError to 503 Service Unavailable.

    The most common RuntimeError in this API is "Model not loaded. Call load_model()
    first." from predict.py, which occurs when a prediction is attempted before the
    model has been loaded. Returning 503 rather than 500 signals to the caller that
    the service is temporarily unavailable (model not yet loaded) rather than that
    an unexpected internal error has occurred.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Return service health status and model availability.

    This endpoint is designed to be polled by container orchestrators (Kubernetes
    readiness probe, ECS health check) to determine whether the service is ready
    to receive traffic. The model_loaded field is the key signal: traffic should
    only be routed to instances where model_loaded=True. An instance with
    model_loaded=False is starting up or experienced a model load failure and
    is not ready to serve predictions.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        model_version=MODEL_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info() -> ModelInfoResponse:
    """Return current model metadata, architecture configuration, and training metrics.

    This endpoint provides the information a model risk reviewer or governance team
    needs to understand what model is currently serving predictions: its version,
    architecture (layer dimensions, dropout rate), input feature list, and the
    train/val/test metrics computed at training time. The metrics here are fixed at
    training time and do not update with new predictions — they reflect the model's
    performance at the point of deployment, not its ongoing performance in production.
    Ongoing performance monitoring is the responsibility of the /monitoring/summary
    endpoint and the downstream model risk monitoring process.
    """
    info = pred.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=LGDPrediction, tags=["Prediction"])
async def predict_single(loan: LoanFeatures) -> LGDPrediction:
    """Predict LGD for a single residential mortgage loan.

    Returns the predicted LGD as a value in [0, 1], where 0 represents full
    recovery and 1 represents total loss (loss equal to the outstanding balance).
    The 90% confidence interval is derived from MC Dropout: 100 stochastic
    forward passes through the network with Dropout enabled. The interval captures
    model uncertainty, not the inherent variability of LGD outcomes.

    Every successful prediction is logged to the monitoring database with the
    prediction_id as the primary key. The prediction_id is returned in the response
    so that the calling system can reference it in any downstream audit trail.

    Raises:
        503: If the model artifact has not been loaded (model_loaded=False in /health).
        422: If any input field fails Pydantic validation (automatic).
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check model file path and run training first.",
        )

    features = loan.model_dump()

    lgd, ci_lower, ci_upper, prediction_id = pred.predict_single(features)

    if _prediction_logger is not None:
        _prediction_logger.log_prediction(
            prediction_id=prediction_id,
            features=features,
            predicted_lgd=lgd,
            model_version=MODEL_VERSION,
        )

    return LGDPrediction(
        lgd=round(lgd, 6),
        confidence_interval_90=(round(ci_lower, 6), round(ci_upper, 6)),
        model_version=MODEL_VERSION,
        prediction_id=prediction_id,
    )


@app.post("/predict/batch", response_model=BatchLGDPrediction, tags=["Prediction"])
async def predict_batch(batch: BatchLoanFeatures) -> BatchLGDPrediction:
    """Predict LGD for a batch of residential mortgage loans (up to 1,000).

    Batch inference runs a single set of MC Dropout forward passes across all
    loans simultaneously, which is more efficient than sequential single-loan
    calls. The batch size limit of 1,000 is enforced by the Pydantic schema
    (BatchLoanFeatures.loans has max_length=1000). For portfolios larger than
    1,000 loans, callers should split the batch at the application layer.

    All predictions in the batch are logged individually to the monitoring database
    with unique prediction_ids. The response includes all predictions and an n_loans
    count for quick verification.

    Raises:
        503: If the model artifact has not been loaded.
        422: If any loan in the batch fails validation (the entire batch is rejected).
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check model file path and run training first.",
        )

    features_list = [loan.model_dump() for loan in batch.loans]
    results = pred.predict_batch(features_list)

    predictions = []
    for features, (lgd, ci_lower, ci_upper, prediction_id) in zip(features_list, results):
        if _prediction_logger is not None:
            _prediction_logger.log_prediction(
                prediction_id=prediction_id,
                features=features,
                predicted_lgd=lgd,
                model_version=MODEL_VERSION,
            )
        predictions.append(
            LGDPrediction(
                lgd=round(lgd, 6),
                confidence_interval_90=(round(ci_lower, 6), round(ci_upper, 6)),
                model_version=MODEL_VERSION,
                prediction_id=prediction_id,
            )
        )

    return BatchLGDPrediction(predictions=predictions, n_loans=len(predictions))


@app.get("/monitoring/summary", response_model=MonitoringSummaryResponse, tags=["Monitoring"])
async def monitoring_summary() -> MonitoringSummaryResponse:
    """Return prediction distribution statistics and input feature drift metrics.

    This endpoint is intended for the model risk monitoring workflow. It should be
    called regularly (daily or weekly) to detect distribution shift in the inputs
    being scored. The ks_statistics field indicates how much each feature's mean
    in the serving population has shifted from its mean in the training population.
    Statistics above approximately 0.10 (10% mean shift) warrant investigation.

    The training_means values here are hardcoded as a placeholder. In a production
    deployment, these would be loaded from a JSON artifact saved alongside the model
    checkpoint during training, so that they automatically update when the model is
    retrained on new data. The current hardcoded values approximate the 2010-2015
    Freddie Mac SFLP training distribution for the numeric features.
    """
    if _prediction_logger is None:
        return MonitoringSummaryResponse(n_predictions=0)

    # TODO: Load these from training artifacts (models/training_feature_means.json)
    # rather than hardcoding. The hardcoded values here are approximate for the
    # 2010-2015 Freddie Mac SFLP training dataset.
    training_means = {
        "orig_ltv": 78.5,
        "orig_upb": 220000.0,
        "orig_interest_rate": 4.5,
        "orig_term": 339.0,
        "ltv_at_default": 95.0,
        "months_delinquent_at_default": 4.5,
        "hpi_change": -0.05,
        "unemployment_rate_at_default": 7.5,
    }

    summary = _prediction_logger.get_summary(training_feature_means=training_means)
    return MonitoringSummaryResponse(**summary)

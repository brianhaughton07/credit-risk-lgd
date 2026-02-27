"""Pydantic request and response schemas for the LGD prediction API.

The schema definitions here serve two purposes that are sometimes treated as
separate concerns but are more useful when addressed together. The first is input
validation: Pydantic will reject any request that violates the field constraints
before the request reaches the prediction logic, returning a 422 Unprocessable
Entity response with a precise error message identifying the invalid field. This
protects the model from receiving inputs it cannot handle gracefully and produces
error messages that an API consumer can act on. The second is documentation: FastAPI
generates an OpenAPI specification from these schema classes, which means the API
documentation reflects the actual constraints enforced at runtime rather than being
a separate artifact that can drift out of sync.

Each constraint on each field reflects a specific business rule about what constitutes
a valid input to this model. The LTV constraint of le=200 is not arbitrary — it is
the upper bound beyond which an LTV value almost certainly represents a data error
rather than an extreme loan characteristic, given that the model was trained on the
2010-2015 Freddie Mac SFLP dataset where LTVs above 200 indicate severely distressed
properties after significant price declines, not normal origination characteristics.
Similarly, months_delinquent_at_default ge=1 reflects the model's conditional framing:
a loan at default has by definition missed at least one payment. Zero delinquency
months would indicate the loan was never delinquent, which is inconsistent with the
default condition.

The Literal types for property_type, occupancy_status, and channel are directly derived
from the Freddie Mac SFLP data dictionary. Any value not in these lists was not present
in training data and cannot be encoded by the LabelEncoder — the constraint prevents
the encoding step from returning an "Unknown" sentinel that the model has not seen.

The json_schema_extra examples on LoanFeatures and LGDPrediction are chosen to
represent a plausible California single-family primary-residence loan from the post-
crisis period — one that would produce a moderate LGD prediction rather than an extreme
value. This makes the API documentation immediately useful for testing and
integration without requiring the consumer to construct a valid payload from scratch.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field


class LoanFeatures(BaseModel):
    """Input features for a single loan LGD prediction request.

    All twelve features listed here correspond to the feature engineering pipeline
    in src/data/features.py. The split between origination characteristics (orig_ltv,
    orig_upb, orig_interest_rate, orig_term, property_type, occupancy_status, channel,
    state) and default-time characteristics (ltv_at_default, months_delinquent_at_default,
    hpi_change, unemployment_rate_at_default) reflects the data availability context
    of LGD prediction: the model is applied at the point of default, so both origination
    and current-state features are known.

    In a production CECL workflow, the origination features come from the loan
    origination system, the delinquency count comes from the servicing system, and
    the macroeconomic features (hpi_change, unemployment_rate_at_default) come from
    the macroeconomic data vendor or internal economics team. The API does not
    integrate with any of those systems — it receives a pre-assembled feature vector
    and returns a prediction. The assembly responsibility belongs to the calling system.
    """

    orig_ltv: float = Field(
        ..., ge=0, le=200, description="Original loan-to-value ratio (%)"
    )
    orig_upb: float = Field(
        ..., gt=0, description="Original unpaid principal balance ($)"
    )
    orig_interest_rate: float = Field(
        ..., ge=0, le=30, description="Interest rate at origination (%)"
    )
    orig_term: int = Field(
        ..., ge=60, le=480, description="Original loan term in months"
    )
    ltv_at_default: float = Field(
        ..., ge=0, le=300, description="Estimated LTV at time of default (%)"
    )
    months_delinquent_at_default: int = Field(
        ..., ge=1, le=120, description="Months delinquent at time of default"
    )
    hpi_change: float = Field(
        ..., ge=-1.0, le=5.0,
        description="HPI change from origination to default (e.g., -0.20 = 20% decline)"
    )
    unemployment_rate_at_default: float = Field(
        ..., ge=0, le=50, description="State unemployment rate at time of default (%)"
    )
    property_type: Literal["SF", "CO", "CP", "MH", "PU"] = Field(
        ..., description="Property type: SF=Single Family, CO=Condo, CP=Co-op, MH=Manufactured, PU=PUD"
    )
    occupancy_status: Literal["P", "S", "I"] = Field(
        ..., description="Occupancy: P=Primary, S=Second Home, I=Investment"
    )
    channel: Literal["R", "B", "C", "T"] = Field(
        ..., description="Origination channel: R=Retail, B=Broker, C=Correspondent, T=TPO"
    )
    state: str = Field(
        ..., min_length=2, max_length=2, description="Two-letter state code (e.g., CA, TX)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "orig_ltv": 85.0,
                "orig_upb": 250000.0,
                "orig_interest_rate": 4.5,
                "orig_term": 360,
                "ltv_at_default": 105.0,
                "months_delinquent_at_default": 6,
                "hpi_change": -0.15,
                "unemployment_rate_at_default": 8.5,
                "property_type": "SF",
                "occupancy_status": "P",
                "channel": "R",
                "state": "CA",
            }
        }
    }


class LGDPrediction(BaseModel):
    """Response schema for a single LGD prediction request.

    The prediction_id field is the audit trail anchor. Every prediction made by
    the API is logged to the SQLite monitoring database with this ID as the primary
    key, which means any prediction returned to a consumer can later be retrieved,
    contextualized, and used for model performance monitoring. In a regulated
    environment, the ability to trace a prediction to the inputs that produced it
    is not a nice-to-have — it is a prerequisite for model validation and ongoing
    monitoring under SR 11-7.

    The confidence_interval_90 is an approximate 90% interval derived from
    MC Dropout sampling (100 forward passes with dropout enabled). Consumers of
    this API should understand that this interval captures epistemic uncertainty
    about the model's prediction, not a frequentist 90% coverage guarantee over
    future LGD outcomes. A wider interval signals that the model is less confident
    about the prediction — typically because the loan characteristics are unusual
    relative to the training distribution.
    """

    lgd: float = Field(
        ..., ge=0, le=1, description="Predicted loss given default (0=full recovery, 1=total loss)"
    )
    confidence_interval_90: Tuple[float, float] = Field(
        ..., description="Approximate 90% confidence interval [lower, upper] via MC Dropout"
    )
    model_version: str = Field(..., description="Model version identifier for audit logging")
    prediction_id: str = Field(..., description="UUID for prediction logging and audit trail")

    model_config = {
        "json_schema_extra": {
            "example": {
                "lgd": 0.32,
                "confidence_interval_90": [0.18, 0.47],
                "model_version": "lgdnet-v1",
                "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }
    }


class BatchLoanFeatures(BaseModel):
    """Request schema for batch LGD prediction.

    The max_length=1000 constraint is not an arbitrary limit — it reflects the
    memory footprint of running 100 MC Dropout passes on a batch of 1000 loans.
    Larger batches would require proportionally more memory and would push the
    API toward or beyond the 200ms latency target specified in the success criteria.
    Consumers with larger portfolios should split requests at the application layer.
    """

    loans: list[LoanFeatures] = Field(..., min_length=1, max_length=1000)


class BatchLGDPrediction(BaseModel):
    """Response schema for batch LGD prediction."""

    predictions: list[LGDPrediction]
    n_loans: int


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint.

    The model_loaded field is the key field for container orchestration: it allows
    a health check to distinguish between "the process is running but no model is
    loaded" (which is a valid startup state before the model artifact is copied in)
    and "the process is running and ready to serve predictions." Container orchestrators
    that route traffic based on health checks should route to this service only when
    model_loaded is True.
    """
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    """Response schema for the /model/info endpoint.

    This endpoint exists for the same reason that the model checkpoint includes
    metrics: a consumer or reviewer should be able to inspect the model's
    performance characteristics through the API without requiring access to
    MLflow or the training artifacts. The train_metrics, val_metrics, and
    test_metrics fields expose the metrics computed at training time, which
    are the metrics the model was validated against.
    """
    model_version: str
    model_type: str
    input_features: list[str]
    architecture: dict
    train_metrics: Optional[dict] = None
    val_metrics: Optional[dict] = None
    test_metrics: Optional[dict] = None


class MonitoringSummaryResponse(BaseModel):
    """Response schema for the /monitoring/summary endpoint.

    The ks_statistics field is somewhat misnamed — it is not a true KS (Kolmogorov-
    Smirnov) statistic, which would require the full distribution of each feature.
    It is a mean-shift metric: |mean_serving - mean_training| / |mean_training|,
    expressed as a fraction. Values above approximately 0.10 (10% shift in feature
    means) warrant investigation. The name is preserved for consistency with common
    monitoring terminology, but consumers should be aware of the actual computation.
    """
    n_predictions: int
    mean_predicted_lgd: Optional[float] = None
    feature_means: Optional[dict] = None
    ks_statistics: Optional[dict] = None
    period: Optional[str] = None

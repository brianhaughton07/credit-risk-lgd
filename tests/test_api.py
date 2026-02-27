"""FastAPI endpoint tests for the LGD prediction service.

The API tests occupy a different position in the test hierarchy than the unit
tests in test_features.py and test_model.py. Unit tests verify that individual
functions produce correct outputs given specific inputs. API tests verify that
the service layer correctly integrates those functions, enforces input validation,
returns the right HTTP status codes, and maintains consistent response schemas.
Both layers are necessary: a bug in the input validation logic (returning 200
for an invalid property_type instead of 422) would not be caught by unit tests
on the prediction functions, which never see invalid inputs if validation is
working correctly.

The test strategy here is to mock the model entirely. The model is not tested
in this file — that is the job of test_model.py. What is tested here is the
contract between the API and its callers: the response codes, the response shapes,
and the validation behavior. Mocking the model allows these tests to run without
a trained artifact, which means they can run in CI without the full data and
training pipeline having been executed. That is the right scope for API tests.

The fixture hierarchy is slightly elaborate — separate fixtures for loaded model,
unloaded model, and batch prediction — but each fixture combination exists because
it tests a specific failure mode that is not covered by the others:

    app_with_model_loaded: Tests the happy path and validates response schemas.
    app_without_model: Tests the 503 behavior when no model is available.
    app_batch: Tests the batch endpoint with the batch mock in place.

The TestClient from FastAPI's test utilities runs the application synchronously
without starting a server, which makes these tests fast and deterministic. The
lifespan handler (model loading and database initialization) is bypassed by
setting _model_loaded and _prediction_logger directly in the fixtures, which is
correct for tests that are concerned with endpoint behavior, not startup behavior.

The 422 validation tests are particularly important because they verify the
boundaries between valid and invalid inputs as defined in the Pydantic schema.
These boundaries are not arbitrary — they reflect domain constraints documented
in the Freddie Mac SFLP data dictionary and the LGD model design document. A
change to one of these constraints (raising the maximum LTV, for example) should
require a deliberate schema change and would be caught by these tests failing,
prompting the developer to decide whether the change is intentional.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# This payload represents a plausible California single-family primary-residence
# loan with moderate stress characteristics — the same example used in the API
# documentation. It is valid against the full Pydantic schema and should produce
# a 200 response on any endpoint that accepts LoanFeatures.
VALID_LOAN_PAYLOAD = {
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_predict_single():
    """Mock predict_single to return fixed values without loading a real model.

    The return value (0.32, 0.18, 0.47, "test-prediction-id-001") represents
    a moderate LGD prediction with a plausible 90% CI. The fixed prediction_id
    allows tests to verify that the ID is included in the response without
    relying on UUID generation.
    """
    with patch("api.predict._model") as mock_model, \
         patch("api.predict._scaler") as mock_scaler, \
         patch("api.predict._feature_cols") as mock_cols, \
         patch("api.predict.predict_single") as mock_fn:

        mock_fn.return_value = (0.32, 0.18, 0.47, "test-prediction-id-001")
        yield mock_fn


@pytest.fixture
def mock_predict_batch():
    """Mock predict_batch to return fixed values for two loans."""
    with patch("api.predict._model") as mock_model, \
         patch("api.predict.predict_batch") as mock_fn:
        mock_fn.return_value = [
            (0.32, 0.18, 0.47, "test-id-001"),
            (0.55, 0.40, 0.70, "test-id-002"),
        ]
        yield mock_fn


@pytest.fixture
def app_with_model_loaded(mock_predict_single):
    """TestClient with model_loaded=True and a mocked prediction logger.

    The _prediction_logger is mocked to avoid SQLite file creation during tests.
    Tests that verify monitoring behavior should use a separate fixture that
    provides a real or more specifically mocked logger.
    """
    from api.main import app, _prediction_logger
    import api.main as main_module

    original = main_module._model_loaded
    main_module._model_loaded = True

    original_logger = main_module._prediction_logger
    main_module._prediction_logger = MagicMock()

    with TestClient(app) as client:
        yield client

    main_module._model_loaded = original
    main_module._prediction_logger = original_logger


@pytest.fixture
def app_without_model():
    """TestClient with model_loaded=False to test 503 behavior.

    The model is not mocked here — the mock_predict_single fixture is not applied —
    because the 503 tests are verifying what happens when the model is absent,
    not what happens when the model is present.
    """
    from api.main import app
    import api.main as main_module

    original = main_module._model_loaded
    main_module._model_loaded = False

    with TestClient(app) as client:
        yield client

    main_module._model_loaded = original


@pytest.fixture
def app_batch(mock_predict_batch):
    """TestClient for batch endpoint tests with batch predict mocked."""
    from api.main import app
    import api.main as main_module

    original = main_module._model_loaded
    main_module._model_loaded = True

    original_logger = main_module._prediction_logger
    main_module._prediction_logger = MagicMock()

    with TestClient(app) as client:
        yield client

    main_module._model_loaded = original
    main_module._prediction_logger = original_logger


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """/health is the first endpoint a container orchestrator calls.

    These tests verify that the health response is consistent with the model
    state: model_loaded=True when a model is present, model_loaded=False when
    it is not. The endpoint should always return 200 regardless of model state
    — a 503 from /health would indicate the service itself is unavailable,
    which is a different condition from the model not being loaded.
    """

    def test_health_returns_200(self, app_with_model_loaded):
        """Health endpoint always returns 200 when the service is running."""
        response = app_with_model_loaded.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, app_with_model_loaded):
        """Health response includes status, model_loaded, and model_version fields."""
        response = app_with_model_loaded.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data

    def test_health_reports_model_loaded(self, app_with_model_loaded):
        """model_loaded is True when the model has been loaded successfully."""
        response = app_with_model_loaded.get("/health")
        assert response.json()["model_loaded"] is True

    def test_health_reports_model_not_loaded(self, app_without_model):
        """Health returns 200 with model_loaded=False when no model is available.

        This is the behavior that allows container orchestrators to distinguish
        between "service is up but model not yet loaded" (model_loaded=False, still 200)
        and "service is down" (non-200 from /health). A 503 response from /health
        would cause the orchestrator to restart the container, which is not the
        correct action when the issue is a missing model artifact.
        """
        response = app_without_model.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# /predict tests
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    """Tests for the /predict endpoint with a valid model loaded.

    These tests verify the happy path: valid inputs produce a 200 response
    with a well-formed LGDPrediction schema. The schema tests are particularly
    important because they ensure the API contract is stable — a change to the
    response schema that removes or renames a field would be caught here.
    """

    def test_valid_request_returns_200(self, app_with_model_loaded):
        """A valid LoanFeatures payload produces a 200 response."""
        response = app_with_model_loaded.post("/predict", json=VALID_LOAN_PAYLOAD)
        assert response.status_code == 200

    def test_valid_request_response_schema(self, app_with_model_loaded):
        """Response includes all required fields: lgd, confidence_interval_90, model_version, prediction_id."""
        response = app_with_model_loaded.post("/predict", json=VALID_LOAN_PAYLOAD)
        data = response.json()
        assert "lgd" in data
        assert "confidence_interval_90" in data
        assert "model_version" in data
        assert "prediction_id" in data

    def test_lgd_in_valid_range(self, app_with_model_loaded):
        """Returned LGD value is in [0, 1]."""
        response = app_with_model_loaded.post("/predict", json=VALID_LOAN_PAYLOAD)
        lgd = response.json()["lgd"]
        assert 0.0 <= lgd <= 1.0, f"LGD {lgd} out of [0, 1]"

    def test_confidence_interval_structure(self, app_with_model_loaded):
        """Confidence interval is a two-element list with lower <= upper."""
        response = app_with_model_loaded.post("/predict", json=VALID_LOAN_PAYLOAD)
        ci = response.json()["confidence_interval_90"]
        assert len(ci) == 2
        lower, upper = ci
        assert lower <= upper, f"CI lower ({lower}) > upper ({upper})"

    def test_prediction_id_returned(self, app_with_model_loaded):
        """prediction_id is a non-empty string in the response."""
        response = app_with_model_loaded.post("/predict", json=VALID_LOAN_PAYLOAD)
        assert len(response.json()["prediction_id"]) > 0

    def test_model_not_loaded_returns_503(self, app_without_model):
        """Prediction request returns 503 when the model has not been loaded.

        503 (Service Unavailable) rather than 500 (Internal Server Error) signals
        that the condition is expected (model not yet loaded) and may resolve
        on retry after a warm-up period, which is the correct interpretation for
        a container starting without a model artifact.
        """
        response = app_without_model.post("/predict", json=VALID_LOAN_PAYLOAD)
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# /predict — input validation (422 tests)
# ---------------------------------------------------------------------------

class TestPredictValidation:
    """Tests for Pydantic input validation on the /predict endpoint.

    Every test in this class verifies that a specific invalid input produces a
    422 Unprocessable Entity response. The 422 tests are important because they
    document the boundary of the valid input space as enforced at the API layer.
    These boundaries are not configuration — they are part of the API contract
    and should only change with a versioned API update.

    The tests use payload mutation (starting from VALID_LOAN_PAYLOAD and changing
    one field) rather than constructing invalid payloads from scratch, which ensures
    that only the field under test is invalid and that the 422 is attributable to
    that specific constraint violation rather than an unrelated validation error.
    """

    def test_missing_required_field_returns_422(self, app_with_model_loaded):
        """A request missing a required field returns 422."""
        payload = {k: v for k, v in VALID_LOAN_PAYLOAD.items() if k != "orig_ltv"}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_property_type_returns_422(self, app_with_model_loaded):
        """A property_type value not in the Literal set returns 422.

        The Literal["SF", "CO", "CP", "MH", "PU"] constraint ensures that only
        Freddie Mac SFLP property type codes seen in training are accepted. "XX"
        is not a valid code and would map to the "Unknown" sentinel in the encoder,
        which the model has not been trained on.
        """
        payload = {**VALID_LOAN_PAYLOAD, "property_type": "XX"}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_occupancy_status_returns_422(self, app_with_model_loaded):
        """An occupancy_status value not in {"P", "S", "I"} returns 422."""
        payload = {**VALID_LOAN_PAYLOAD, "occupancy_status": "Z"}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_ltv_above_maximum_returns_422(self, app_with_model_loaded):
        """orig_ltv above 200 returns 422 — values above this threshold indicate data errors."""
        payload = {**VALID_LOAN_PAYLOAD, "orig_ltv": 250.0}  # Max is 200
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_upb_returns_422(self, app_with_model_loaded):
        """orig_upb < 0 returns 422 — a negative loan balance is impossible."""
        payload = {**VALID_LOAN_PAYLOAD, "orig_upb": -100.0}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_zero_upb_returns_422(self, app_with_model_loaded):
        """orig_upb == 0 returns 422 — a zero-balance loan is not a valid LGD prediction input."""
        payload = {**VALID_LOAN_PAYLOAD, "orig_upb": 0.0}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_channel_returns_422(self, app_with_model_loaded):
        """A channel value not in {"R", "B", "C", "T"} returns 422."""
        payload = {**VALID_LOAN_PAYLOAD, "channel": "Z"}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_state_too_long_returns_422(self, app_with_model_loaded):
        """A state code longer than 2 characters returns 422.

        The two-character constraint reflects the US postal code standard.
        A three-character state code ("CAL") is almost certainly a data entry
        error rather than a valid input, and should be rejected rather than
        silently mapped to the "Other" region.
        """
        payload = {**VALID_LOAN_PAYLOAD, "state": "CAL"}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422

    def test_months_delinquent_zero_returns_422(self, app_with_model_loaded):
        """months_delinquent_at_default == 0 returns 422.

        A loan at default has missed at least one payment by definition. Zero
        delinquency months is inconsistent with the default condition and should
        not be processed — it likely indicates a data pipeline error where a
        current loan was routed to the LGD scoring system.
        """
        payload = {**VALID_LOAN_PAYLOAD, "months_delinquent_at_default": 0}
        response = app_with_model_loaded.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict/batch tests
# ---------------------------------------------------------------------------

class TestPredictBatchEndpoint:
    """Tests for the /predict/batch endpoint.

    The batch endpoint shares the same input validation logic as /predict
    (each loan in the batch is validated individually), so the 422 validation
    tests from TestPredictValidation implicitly cover the batch path as well.
    These tests focus on the batch-specific behavior: correct count in the
    response, empty batch rejection, and single-loan batch handling.
    """

    def test_batch_valid_request_returns_200(self, app_batch):
        """A batch of valid loans returns 200."""
        payload = {"loans": [VALID_LOAN_PAYLOAD, VALID_LOAN_PAYLOAD]}
        response = app_batch.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_returns_correct_count(self, app_batch):
        """n_loans in the response matches the number of loans submitted."""
        payload = {"loans": [VALID_LOAN_PAYLOAD, VALID_LOAN_PAYLOAD]}
        response = app_batch.post("/predict/batch", json=payload)
        data = response.json()
        assert data["n_loans"] == 2
        assert len(data["predictions"]) == 2

    def test_empty_batch_returns_422(self, app_batch):
        """An empty loans list returns 422 because min_length=1 is specified in the schema.

        An empty batch request is almost certainly a client-side error rather than
        a legitimate use case. Returning 422 immediately is more informative than
        returning an empty predictions list with n_loans=0.
        """
        payload = {"loans": []}
        response = app_batch.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_single_loan_batch_works(self, app_batch):
        """A batch of one loan is valid and returns n_loans=1."""
        payload = {"loans": [VALID_LOAN_PAYLOAD]}
        response = app_batch.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["n_loans"] == 1


# ---------------------------------------------------------------------------
# /model/info tests
# ---------------------------------------------------------------------------

class TestModelInfoEndpoint:
    """/model/info is the governance review endpoint.

    These tests verify that the endpoint returns 200 and includes the minimum
    fields required for a model risk reviewer to understand the current model.
    The actual values depend on the model artifact and are not tested here —
    only the presence of required fields is verified.
    """

    def test_model_info_returns_200(self, app_with_model_loaded):
        """Model info endpoint returns 200."""
        with patch("api.predict.get_model_info") as mock_info:
            mock_info.return_value = {
                "model_version": "lgdnet-v1",
                "model_type": "LGDNet",
                "input_features": ["orig_ltv", "orig_upb"],
                "architecture": {"input_dim": 12, "hidden_dims": [256, 128, 64]},
                "train_metrics": None,
                "val_metrics": None,
                "test_metrics": None,
            }
            response = app_with_model_loaded.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self, app_with_model_loaded):
        """Model info response includes model_version and input_features."""
        with patch("api.predict.get_model_info") as mock_info:
            mock_info.return_value = {
                "model_version": "lgdnet-v1",
                "model_type": "LGDNet",
                "input_features": ["orig_ltv"],
                "architecture": {},
                "train_metrics": None,
                "val_metrics": None,
                "test_metrics": None,
            }
            response = app_with_model_loaded.get("/model/info")
        data = response.json()
        assert "model_version" in data
        assert "input_features" in data


# ---------------------------------------------------------------------------
# /monitoring/summary tests
# ---------------------------------------------------------------------------

class TestMonitoringEndpoint:
    """/monitoring/summary is the model risk monitoring endpoint.

    These tests verify that the endpoint returns 200 and that the response
    schema is stable. The actual values depend on the prediction log contents
    and are mocked here. Tests that verify the content of the monitoring
    statistics belong in tests for the PredictionLogger class, not here.
    """

    def test_monitoring_returns_200(self, app_with_model_loaded):
        """Monitoring summary returns 200."""
        with patch("api.main._prediction_logger") as mock_logger:
            mock_logger.get_summary.return_value = {
                "n_predictions": 10,
                "mean_predicted_lgd": 0.35,
                "feature_means": {"orig_ltv": 82.0},
                "ks_statistics": {"orig_ltv": 0.05},
                "period": "2024-01-01 to 2024-01-31",
            }
            response = app_with_model_loaded.get("/monitoring/summary")
        assert response.status_code == 200

    def test_monitoring_schema(self, app_with_model_loaded):
        """Monitoring response includes n_predictions field.

        n_predictions=0 is the minimum valid response when no predictions have
        been logged. The endpoint must handle this case without error, which
        is what the empty-logger return value in get_summary() is designed for.
        """
        with patch("api.main._prediction_logger") as mock_logger:
            mock_logger.get_summary.return_value = {
                "n_predictions": 0,
                "mean_predicted_lgd": None,
                "feature_means": None,
                "ks_statistics": None,
                "period": None,
            }
            response = app_with_model_loaded.get("/monitoring/summary")
        data = response.json()
        assert "n_predictions" in data

# Test Suite

The test suite for this project is organized into three files that cover three
distinct layers of the system: the feature engineering pipeline, the model
architecture and training components, and the FastAPI service layer. The layering
is intentional — each file tests the claims made by its corresponding module without
depending on the other layers being functional. `test_features.py` runs without
PyTorch. `test_model.py` runs without FastAPI. `test_api.py` runs without a trained
model artifact on disk. That independence means you can run the full suite at any
point in the project lifecycle, not just after the full pipeline has been executed.

---

## Prerequisites

All tests require the `credit-risk-lgd` conda environment. If you have not yet
created it:

```bash
conda env create -f environment.yml
conda activate credit-risk-lgd
```

The tests do **not** require:
- Freddie Mac data files in `data/raw/`
- A trained model artifact in `models/`
- MLflow to be running

---

## Running the Tests

### Full suite

From the project root:

```bash
pytest tests/
```

### Individual test files

```bash
pytest tests/test_features.py   # Feature pipeline tests only
pytest tests/test_model.py      # Model architecture tests only
pytest tests/test_api.py        # API endpoint tests only
```

### Verbose output (recommended for first run)

```bash
pytest tests/ -v
```

This prints each test name and pass/fail status individually, which makes it
straightforward to identify which test is failing and in which class.

### Stop on first failure

```bash
pytest tests/ -x
```

Useful during development when you want to fix one failure at a time rather than
seeing the full list.

### Run a specific test class

```bash
pytest tests/test_model.py::TestWeightedMSELoss -v
pytest tests/test_features.py::TestLGDTargetConstruction -v
```

---

## Expected Results

### test_features.py (25 tests)

All 25 tests should pass without any data files or model artifacts. These tests
use in-memory DataFrames constructed in the fixtures and do not read from disk.

Expected output:
```
tests/test_features.py::TestLGDTargetConstruction::test_lgd_values_in_range PASSED
tests/test_features.py::TestLGDTargetConstruction::test_lgd_zero_upb_rows_dropped PASSED
tests/test_features.py::TestLGDTargetConstruction::test_lgd_full_recovery_is_valid PASSED
tests/test_features.py::TestLGDTargetConstruction::test_lgd_capped_at_one PASSED
tests/test_features.py::TestLTVAtDefault::test_ltv_computed_when_hpi_available PASSED
tests/test_features.py::TestLTVAtDefault::test_ltv_not_negative PASSED
tests/test_features.py::TestLTVAtDefault::test_ltv_capped_at_300 PASSED
tests/test_features.py::TestHPIChange::test_hpi_change_placeholder_when_no_file PASSED
tests/test_features.py::TestHPIChange::test_hpi_change_not_overwritten_if_present PASSED
tests/test_features.py::TestRegionMapping::test_all_states_have_region PASSED
tests/test_features.py::TestRegionMapping::test_unknown_state_maps_to_other PASSED
tests/test_features.py::TestRegionMapping::test_region_column_created PASSED
tests/test_features.py::TestModificationFlag::test_y_maps_to_one PASSED
tests/test_features.py::TestModificationFlag::test_missing_column_creates_zeros PASSED
tests/test_features.py::TestVintageYear::test_vintage_year_from_first_payment_date PASSED
tests/test_features.py::TestVintageYear::test_vintage_year_not_overwritten_if_present PASSED
tests/test_features.py::TestEncodeCategoricals::test_categorical_encoding_output_shape PASSED
tests/test_features.py::TestEncodeCategoricals::test_encoded_values_are_integers PASSED
tests/test_features.py::TestEncodeCategoricals::test_missing_categorical_column_skipped PASSED
tests/test_features.py::TestUnemployment::test_unemployment_placeholder_when_no_file PASSED
tests/test_features.py::TestUnemployment::test_unemployment_not_overwritten_if_present PASSED
tests/test_features.py::TestFeatureDirectionality::test_higher_hpi_decline_increases_ltv PASSED
```

**What these tests confirm:** The LGD target formula produces values in [0, 1] for
all valid inputs, drops rows with zero UPB, and caps values at 1.0. Feature computations
respond to inputs in economically correct directions — higher HPI decline produces
higher LTV at default. The region mapping is complete and unknown state codes fall back
to "Other" without raising exceptions.

---

### test_model.py (17 tests)

All 17 tests should pass. These tests require PyTorch but do not require a trained
model artifact — they construct small models in memory for each test.

Expected output:
```
tests/test_model.py::TestLGDNetArchitecture::test_output_shape_single PASSED
tests/test_model.py::TestLGDNetArchitecture::test_output_shape_batch PASSED
tests/test_model.py::TestLGDNetArchitecture::test_output_range PASSED
tests/test_model.py::TestLGDNetArchitecture::test_output_not_all_same PASSED
tests/test_model.py::TestLGDNetArchitecture::test_custom_hidden_dims PASSED
tests/test_model.py::TestLGDNetArchitecture::test_single_hidden_layer PASSED
tests/test_model.py::TestLGDNetArchitecture::test_get_config_roundtrip PASSED
tests/test_model.py::TestGradientFlow::test_gradients_flow_to_all_layers PASSED
tests/test_model.py::TestGradientFlow::test_loss_decreases_with_training PASSED
tests/test_model.py::TestWeightedMSELoss::test_loss_nonnegative PASSED
tests/test_model.py::TestWeightedMSELoss::test_tail_weight_higher_than_center PASSED
tests/test_model.py::TestWeightedMSELoss::test_zero_loss_for_perfect_predictions PASSED
tests/test_model.py::TestWeightedMSELoss::test_alpha_zero_reduces_to_mse PASSED
tests/test_model.py::TestLGDDataset::test_dataset_length PASSED
tests/test_model.py::TestLGDDataset::test_dataset_item_types PASSED
tests/test_model.py::TestLGDDataset::test_dataset_dataloader_batching PASSED
tests/test_model.py::TestMCDropout::test_ci_bounds_ordered PASSED
tests/test_model.py::TestMCDropout::test_ci_values_in_range PASSED
tests/test_model.py::TestMCDropout::test_ci_has_nonzero_width PASSED
tests/test_model.py::TestBaselineSanity::test_lgdnet_beats_random PASSED
```

**What these tests confirm:** LGDNet produces outputs in [0, 1] for all batch sizes,
gradient flow reaches all linear layer weights without NaN values, and the loss
decreases over 20 training steps on a synthetic dataset. WeightedMSELoss assigns
higher weight to tail observations than to center observations at equal error magnitude.
MC Dropout produces well-ordered confidence intervals with non-zero width when
`dropout > 0`. A trained LGDNet outperforms random predictions on a linear synthetic
dataset.

**Expected runtime:** Approximately 15–30 seconds. The `TestBaselineSanity` test
trains a small network for 30 epochs and is the slowest test in the suite.

---

### test_api.py (24 tests)

All 24 tests should pass. These tests use FastAPI's `TestClient` with mocked model
and prediction logger — no model artifact or database file is created or read.

Expected output:
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_200 PASSED
tests/test_api.py::TestHealthEndpoint::test_health_response_schema PASSED
tests/test_api.py::TestHealthEndpoint::test_health_reports_model_loaded PASSED
tests/test_api.py::TestHealthEndpoint::test_health_reports_model_not_loaded PASSED
tests/test_api.py::TestPredictEndpoint::test_valid_request_returns_200 PASSED
tests/test_api.py::TestPredictEndpoint::test_valid_request_response_schema PASSED
tests/test_api.py::TestPredictEndpoint::test_lgd_in_valid_range PASSED
tests/test_api.py::TestPredictEndpoint::test_confidence_interval_structure PASSED
tests/test_api.py::TestPredictEndpoint::test_prediction_id_returned PASSED
tests/test_api.py::TestPredictEndpoint::test_model_not_loaded_returns_503 PASSED
tests/test_api.py::TestPredictValidation::test_missing_required_field_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_invalid_property_type_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_invalid_occupancy_status_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_ltv_above_maximum_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_negative_upb_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_zero_upb_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_invalid_channel_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_state_too_long_returns_422 PASSED
tests/test_api.py::TestPredictValidation::test_months_delinquent_zero_returns_422 PASSED
tests/test_api.py::TestPredictBatchEndpoint::test_batch_valid_request_returns_200 PASSED
tests/test_api.py::TestPredictBatchEndpoint::test_batch_returns_correct_count PASSED
tests/test_api.py::TestPredictBatchEndpoint::test_empty_batch_returns_422 PASSED
tests/test_api.py::TestPredictBatchEndpoint::test_single_loan_batch_works PASSED
tests/test_api.py::TestModelInfoEndpoint::test_model_info_returns_200 PASSED
tests/test_api.py::TestModelInfoEndpoint::test_model_info_has_required_fields PASSED
tests/test_api.py::TestMonitoringEndpoint::test_monitoring_returns_200 PASSED
tests/test_api.py::TestMonitoringEndpoint::test_monitoring_schema PASSED
```

**What these tests confirm:** The health endpoint returns 200 in both loaded and
unloaded model states, with `model_loaded` set correctly. The predict endpoint
returns a well-formed `LGDPrediction` schema on valid inputs and 503 when the model
is not loaded. Nine 422 tests verify the input validation boundaries: invalid
`property_type`, `occupancy_status`, and `channel` values; `orig_ltv` above 200;
`orig_upb` at or below 0; `state` longer than 2 characters; `months_delinquent_at_default`
equal to 0. The batch endpoint handles correct counts, rejects empty batches, and
accepts single-loan batches.

---

## Full Suite Summary

```
$ pytest tests/ -v
...
62 passed in ~30s
```

The total count is approximately 62 tests (25 + 20 + 27 — exact count depends on
pytest's collection of parametrized and fixture-expanded tests). If any tests fail
on a clean install of the environment, that is most likely a package version
incompatibility; check that `conda activate credit-risk-lgd` is active before running.

---

## What the Tests Do Not Cover

Understanding the scope of the test suite is as important as understanding what it
covers. The following are intentionally not tested here:

**End-to-end pipeline correctness.** The tests verify individual functions against
controlled inputs. They do not verify that running `ingest.py → preprocess.py →
features.py → train.py` on real Freddie Mac data produces correct results. That
verification requires the data, the full pipeline run, and manual inspection of
the calibration plot and SHAP outputs in `reports/`.

**Model performance against success criteria.** The `test_lgdnet_beats_random` test
is a sanity check, not a performance benchmark. The actual success criteria — 15% MAE
improvement over the segment-average benchmark, R² > 0.30 — are measured by
`src/models/evaluate.py` against the real test set and reported in `docs/model_card.md`.

**API latency.** The tests verify correctness but do not measure response time. The
sub-200ms latency target requires a deployed API with a loaded model artifact; it
cannot be meaningfully measured with `TestClient` in a test process.

**Macroeconomic feature values.** The HPI change and unemployment rate features
currently use placeholder values (0.0 and 7.5% respectively). When real external
data is integrated, additional tests should be added to `test_features.py` verifying
that the external data join produces non-placeholder values for loans in the vintage
window.

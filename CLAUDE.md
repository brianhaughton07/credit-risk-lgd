# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n credit-risk-lgd python=3.11
conda activate credit-risk-lgd
# macOS prereqs for LightGBM/XGBoost:
brew install cmake libomp
pip install -r requirements.txt
```

## Common Commands

```bash
# Run full pipeline (requires data in data/raw/)
python src/data/ingest.py --config configs/default.yaml
python src/data/preprocess.py --config configs/default.yaml
python src/data/features.py --config configs/default.yaml
python src/models/baseline.py --config configs/default.yaml
python src/models/train.py --config configs/default.yaml
python src/models/evaluate.py --config configs/default.yaml

# Tests
pytest tests/ -v                  # all tests
pytest tests/test_api.py -v       # single file
pytest tests/test_model.py::TestLGDNet::test_output_range -v  # single test

# API
uvicorn api.main:app --reload     # dev server at http://localhost:8000
# API docs at http://localhost:8000/docs

# MLflow UI
mlflow ui --backend-store-uri ./mlruns   # http://localhost:5000
```

## Architecture Overview

This is a conditional LGD (Loss Given Default) prediction pipeline for residential mortgages — "conditional" because the model only scores loans that have already defaulted. The pipeline stages are strictly ordered and independent:

1. **`src/data/ingest.py`** — Reads Freddie Mac pipe-delimited flat files (no headers, positional columns). Filters to defaulted loans (`zero_balance_code` in `{02,03,06,09}`) at ingestion to avoid loading hundreds of millions of performance rows. Writes parquet to `data/interim/`.

2. **`src/data/preprocess.py`** — Type casting, imputation, and LGD target construction: `LGD = Net Loss / UPB at Default`, capped to `[0, 1]`. Zero-UPB rows are dropped.

3. **`src/data/features.py`** — Engineered features: LTV at default, HPI change, regional encoding (state → census region). External macro placeholders (HPI, unemployment) default to `0.0`/`7.5` until joined from real data.

4. **`src/models/`** — Three models: segment-average benchmark (baseline), LightGBM/Ridge baselines (`baseline.py`), and `LGDNet` (`lgd_net.py`). Training uses MLflow logging with early stopping. `evaluate.py` produces calibration plots, SHAP values, and macro scenario analysis.

5. **`api/`** — FastAPI service. Loads model once at startup via `lifespan`. MC Dropout (100 passes) provides 90% CI on predictions. Predictions are logged to SQLite for drift monitoring.

## Key Design Constraints

**Config is the single source of truth.** All hyperparameters, paths, and feature lists live in `configs/default.yaml` and are accessed via `src/utils/config.py`'s `load_config()`. Do not hardcode values from the config elsewhere. `load_config()` can be called without arguments from anywhere — it walks up to the project root automatically.

**Feature column order is load-bearing.** The lists in `configs/default.yaml` under `features.categorical` and `features.numeric` determine the column order in the feature matrix. This order must match the trained `StandardScaler` and model weights. Adding/reordering features requires retraining.

**MC Dropout drives uncertainty.** `dropout=0.3` in `ModelConfig` serves dual purpose: training regularizer and inference uncertainty mechanism. Setting `dropout=0` eliminates the CI capability in the API.

**The column schemas in `ingest.py` are the single source of truth for Freddie Mac file layout.** `ORIGINATION_COLUMNS` and `PERFORMANCE_COLUMNS` are positional — Freddie Mac files have no headers. Misaligned columns fail silently with wrong values, not exceptions.

**API starts in degraded state without a model artifact.** If `models/lgd_model.pt` is missing, `/health` returns `model_loaded=False` and prediction endpoints return 503. This is intentional for container deployment ordering.

**Known limitation:** `training_means` in `api/main.py`'s `monitoring_summary()` endpoint are hardcoded approximations. They should be loaded from `models/training_feature_means.json` saved at training time.

## Data

Raw Freddie Mac SFLP files go in `data/raw/origination/` (origination files: `historical_data_*.txt`) and `data/raw/performance/` (performance files: `historical_data_time_*.txt`). See `data/README.md` for download instructions. All `data/` subdirectories are gitignored.

Vintages 2010–2015 are the target window (configured in `configs/default.yaml`). Train/val/test split is 70/15/15, stratified by `vintage_year` to preserve vintage proportions across splits.

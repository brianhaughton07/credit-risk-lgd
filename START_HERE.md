# START HERE

## Environment

**Activate the conda env first — everything else assumes this:**

```bash
conda activate credit-risk-lgd
```

All scripts must run from the project root (`credit-risk-lgd/`) so the relative paths in `configs/default.yaml` resolve correctly:

```bash
cd /Users/brian/Documents/GitHub/projects/credit-risk-lgd
```

No other environment variables are required for the data pipeline or training. The API reads three optional overrides at startup (`LGD_MODEL_PATH`, `LGD_SCALER_PATH`, `LGD_DB_PATH`), but the defaults in `configs/default.yaml` work fine for a local run.

---

## A note on verbose logging

All pipeline scripts emit **structured JSON logs** by default — one JSON object per line, suitable for log aggregators. In a terminal, that is not readable. Two options:

**Option A — pipe through `jq` (recommended, no code change):**
```bash
python src/data/ingest.py 2>&1 | jq .
```

**Option B — switch to human-readable format for your session.** In `src/utils/logging.py`, `get_logger()` accepts `json_format=False`. You can do a temporary global switch by adding one line to each script you run, or patch it at the top of any script:

```python
# At the top of any script, after imports:
import src.utils.logging as _log_mod
_orig = _log_mod.get_logger
_log_mod.get_logger = lambda name, **kw: _orig(name, json_format=False, **kw)
```

The simpler practical path is `jq`. The examples below assume `jq` is available; drop `| jq .` if you prefer raw JSON or want to redirect to a file.

---

## Pre-condition: Data

The pipeline requires Freddie Mac SFLP files on disk before step 1. If you have not yet downloaded them, see `data/README.md` for the registration link and directory layout. Files go to:

```
data/raw/origination/historical_data_YYYYQQ.txt   (24 files, 2010Q1–2015Q4)
data/raw/performance/historical_data_time_YYYYQQ.txt  (24 files)
```

The full 2010–2015 dataset requires ~16 GB RAM for ingestion. For pipeline testing without all the data, any subset of quarterly files placed in the correct directories will run — the ingestion loop processes whatever files it finds.

---

## Pipeline Sequence

### Step 1 — Ingest

Reads the raw pipe-delimited flat files, validates schema, filters performance data to defaulted loans only, writes parquet to `data/interim/`.

```bash
python src/data/ingest.py --config configs/default.yaml 2>&1 | jq .
```

**What to look for:**
- `"message": "Reading historical_data_YYYYQQ.txt"` — one entry per file processed
- `"message": "Staged origination to data/interim/origination.parquet: NNN loans"` — expect 150,000–200,000 loans for the full 2010–2015 vintage window
- `"message": "Staged defaulted performance to data/interim/performance_defaults.parquet: NNN loans"` — this count is the default loan universe the model trains on; expect 50,000–80,000
- Any `"level": "WARNING"` with `on_bad_lines` indicates malformed rows in a quarterly file — a small number is normal

---

### Step 2 — Preprocess

Casts raw string columns to numeric types, applies domain-informed imputation, constructs the LGD target variable, merges origination and performance, writes `data/processed/cleaned.parquet`.

```bash
python src/data/preprocess.py --config configs/default.yaml 2>&1 | jq .
```

**What to look for:**
- `"message": "LGD target constructed: NNN loans"` — should match the defaulted loan count from step 1 minus any zero-UPB rows dropped
- `"message": "LGD distribution"` — the summary stats. For the 2010–2015 cohort, expect mean LGD around 0.35–0.45, with spikes near 0.0 and 1.0 in the percentiles
- Any warning about high null rates on specific columns is expected for `hpi_change` and `unemployment_rate_at_default` (placeholders until external data is joined)

---

### Step 3 — Feature Engineering

Computes LTV at default, HPI change (placeholder 0.0), unemployment (placeholder 7.5%), encodes categoricals via LabelEncoder, writes `data/processed/features.parquet`.

```bash
python src/data/features.py --config configs/default.yaml 2>&1 | jq .
```

**What to look for:**
- `"message": "Features written to data/processed/features.parquet: NNN rows, NN columns"`
- Log entries for each feature function: `add_region`, `compute_ltv_at_default`, `encode_categoricals`
- `"level": "WARNING"` about HPI and unemployment placeholders is expected and intentional — the TODOs in `features.py` document what external data integration would replace them

---

### Step 4 — Baseline Models (optional but recommended)

Trains segment-average benchmark, Ridge, and LightGBM. Logs MAE improvement percentages. This establishes the bar that LGDNet must beat.

```bash
python src/models/baseline.py --config configs/default.yaml 2>&1 | jq .
```

**What to look for:**
- `"message": "Benchmark — test MAE=X.XXXX"` — this is the segment-average number; record it
- `"message": "Ridge — test MAE=X.XXXX, MAE improvement vs benchmark=XX.X%"` — Ridge typically achieves 5–12% improvement over segment average
- `"message": "LightGBM — test MAE=X.XXXX, MAE improvement vs benchmark=XX.X%"` — LightGBM typically achieves 10–18%; if LGDNet does not match or exceed this, the training procedure needs investigation

---

### Step 5 — Train LGDNet

Runs the full training loop: 70/15/15 split stratified by vintage year, WeightedMSELoss with tail emphasis, early stopping on validation MAE, ReduceLROnPlateau scheduling. Saves `models/lgd_model.pt` and `models/scaler.pt`. Logs to `mlruns/`.

```bash
python src/models/train.py --config configs/default.yaml 2>&1 | jq .
```

**What to look for:**
- `"message": "Training on device: cpu"` (or `cuda` if a GPU is available)
- `"message": "Data splits: train=NNN, val=NNN, test=NNN; features=12"` — verify the 70/15/15 split looks right
- Per-epoch log lines every 10 epochs: watch `val_mae` trend downward for the first 20–40 epochs, then plateau
- `"message": "Early stopping at epoch NN"` — typical early stopping is between epoch 30 and 70 depending on the data
- `"message": "Final metrics — train MAE=X.XXXX, R²=X.XX | val MAE=X.XXXX | test MAE=X.XXXX"` — for the success criteria: test MAE must be ≥15% below the benchmark MAE from step 4, and test R² must be above 0.30
- `"message": "Model saved to models/lgd_model.pt"` — confirm this appears before moving to step 6

---

### Step 6 — Evaluate

Generates the calibration decile plot, SHAP importance chart, and macro stress scenario analysis. Saves to `reports/`. Logs to MLflow.

```bash
python src/models/evaluate.py --config configs/default.yaml 2>&1 | jq .
```

**What to look for:**
- `"message": "Test set — MAE=X.XXXX, RMSE=X.XXXX, R²=X.XX"` — final reported numbers
- `"message": "Top 5 SHAP features by mean |SHAP|"` — expect `ltv_at_default` and (with placeholder data) some origination features; `hpi_change` will rank lower until real HPI data is integrated
- `"message": "Macro scenario analysis results"` — with placeholder unemployment and HPI values (7.5% and 0.0), all four scenarios will show identical mean LGD; this is expected and will change once real external data is joined
- `"message": "Calibration plot saved to reports/calibration_plot.png"` — open this file; it should show points tracking close to the 45-degree diagonal across all deciles

---

### Step 7 — Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**What to look for at startup:**
- `INFO: Started server process`
- `INFO: Waiting for application startup`
- No `FileNotFoundError` in the startup output (which would mean `models/lgd_model.pt` is missing)

**Verify it is working:**

```bash
curl -s http://localhost:8000/health | jq .
```
Should return `{"status":"ok","model_loaded":true,"model_version":"lgdnet-v1"}`.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "orig_ltv": 85.0, "orig_upb": 250000.0, "orig_interest_rate": 4.5,
    "orig_term": 360, "ltv_at_default": 105.0, "months_delinquent_at_default": 6,
    "hpi_change": -0.15, "unemployment_rate_at_default": 8.5,
    "property_type": "SF", "occupancy_status": "P", "channel": "R", "state": "CA"
  }' | jq .
```
Should return a `lgd` value in [0,1] with a `confidence_interval_90` array.

---

## MLflow UI (optional)

After steps 4–6, you can browse all logged runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Opens at `http://localhost:5000`. The `lgd-prediction` experiment will show runs for `lightgbm_baseline`, `lgdnet`, and `evaluation` side by side with all metrics and artifacts.

# Credit Risk: Loss Given Default (LGD) Prediction

End-to-end ML pipeline for predicting Loss Given Default on residential mortgages using Freddie Mac Single Family Loan Performance data (2010–2015 vintages).

## Business Context

**Loss Given Default (LGD)** — the fraction of a loan balance unrecovered after default — is a core input to:
- **CECL reserves** (ASC 326): Expected Credit Loss = PD × LGD × EAD
- **Stress testing** (DFAST/CCAR): How does LGD change under rising unemployment and falling home prices?

Most institutions estimate LGD using segment-average historical rates. This project builds a loan-level ML model that improves on that baseline, demonstrates regulatory explainability via SHAP, and serves predictions through a production-ready REST API.

---

## Repository Structure

```
credit-risk-lgd/
├── configs/default.yaml         # All paths, hyperparameters, feature lists
├── data/
│   ├── README.md                # Download instructions for Freddie Mac SFLP
│   ├── raw/                     # .gitignored — Freddie Mac raw files
│   ├── interim/                 # .gitignored — staged parquet files
│   └── processed/               # .gitignored — model-ready feature matrices
├── src/
│   ├── data/
│   │   ├── ingest.py            # Validate and stage raw data
│   │   ├── preprocess.py        # Type casting, missing values, LGD target
│   │   └── features.py          # Feature engineering pipeline
│   ├── models/
│   │   ├── baseline.py          # Ridge + LightGBM baselines; segment benchmark
│   │   ├── lgd_net.py           # LGDNet PyTorch MLP architecture
│   │   ├── train.py             # Training loop with MLflow
│   │   └── evaluate.py          # Metrics, calibration, SHAP, scenarios
│   └── utils/
│       ├── config.py            # Typed Config dataclass from YAML
│       └── logging.py           # Structured JSON logging
├── api/
│   ├── main.py                  # FastAPI app
│   ├── schemas.py               # Pydantic request/response models
│   ├── predict.py               # Model loading and inference
│   └── monitoring.py            # SQLite prediction logging + drift stats
├── tests/
│   ├── test_features.py         # Feature unit tests
│   ├── test_model.py            # LGDNet output/gradient tests
│   └── test_api.py              # FastAPI endpoint tests
├── notebooks/                   # Analysis notebooks (see below)
└── docs/
    ├── data_dictionary.md       # All features, types, derivations
    └── model_card.md            # Intended use, limitations, metrics
```

---

## Modeling Approach

### LGD Target Construction

LGD is derived from Freddie Mac performance data (not a raw column):

```
LGD = Net Loss / UPB at Default

Net Loss = UPB_at_default - Net_Proceeds + Foreclosure_Costs - MI_Recovery
```

- LGD ∈ [0, 1]: 0 = full recovery, 1 = total loss
- Only defaulted loans are included (conditional model)
- Loans with zero UPB at resolution are excluded

### Benchmark

All accuracy comparisons are against **segment-average historical LGD rates** grouped by (property_type, occupancy_status, vintage_year) — the approach most institutions currently use for CECL provisioning.

### Models

| Model | Notes |
|---|---|
| Segment-average benchmark | Baseline; no ML |
| Ridge Regression | Linear baseline |
| LightGBM | Non-parametric baseline; strong contender |
| **LGDNet** | PyTorch MLP with weighted MSE tail loss; primary model |

### LGDNet Architecture

```
Input (12 features)
  → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
  → Linear(1)   → Sigmoid
Output: LGD ∈ [0, 1]
```

**Loss function:** Weighted MSE with higher weight near LGD = 0 and LGD = 1 (tail emphasis). See `docs/model_card.md` for rationale.

**Uncertainty:** MC Dropout (100 samples) for approximate 90% confidence intervals.

---

## Success Criteria

| Metric | Target |
|---|---|
| MAE improvement vs benchmark | ≥ 15% |
| RMSE improvement vs benchmark | ≥ 10% |
| R² on test set | > 0.30 |
| Calibration | Predicted vs. actual decile plot holds in tails |
| Macro sensitivity | Rising unemployment / falling HPI → higher LGD |
| API latency | < 200ms single-loan inference |

---

## How to Run

### 1. Setup

```bash
# Clone
git clone https://github.com/brianhaughton07/credit-risk-lgd.git
cd credit-risk-lgd

# macOs prerequisite - required before pip install
xcode-select --install
brew install cmake libomp

# Create and activate the environment
conda create -n credit-risk-lgd python=3.11
conda activate credit-risk-lgd

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Follow instructions in `data/README.md` to download Freddie Mac SFLP files and place them in `data/raw/origination/` and `data/raw/performance/`.

### 3. Run the Pipeline

```bash
# Step 1: Ingest and validate raw data
python src/data/ingest.py --config configs/default.yaml

# Step 2: Preprocess and construct LGD target
python src/data/preprocess.py --config configs/default.yaml

# Step 3: Feature engineering
python src/data/features.py --config configs/default.yaml

# Step 4: Train baseline models
python src/models/baseline.py --config configs/default.yaml

# Step 5: Train LGDNet
python src/models/train.py --config configs/default.yaml

# Step 6: Evaluate
python src/models/evaluate.py --config configs/default.yaml
```

### 4. View Experiment Tracking

```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

### 5. Start the API

```bash
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

**Sample request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "state": "CA"
  }'
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single loan LGD prediction with 90% CI |
| `/predict/batch` | POST | Batch prediction (up to 1000 loans) |
| `/health` | GET | Service health check |
| `/model/info` | GET | Model version, metrics, feature list |
| `/monitoring/summary` | GET | Prediction distribution drift statistics |

---

## Findings

*To be completed after training on full dataset.*

| Finding | Detail |
|---|---|
| Best model | TBD |
| Test MAE | TBD |
| Benchmark MAE | TBD |
| MAE improvement | TBD |
| R² | TBD |
| Top SHAP features | TBD |
| Macro sensitivity | TBD |

---

## Technologies

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy, pyarrow |
| Baseline models | scikit-learn, LightGBM |
| Deep learning | PyTorch |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Testing | pytest, httpx |
| Visualization | matplotlib, seaborn |

---

## Regulatory Context

This model supports compliance with ASC 326 (CECL) and DFAST/CCAR stress testing requirements. See `docs/model_card.md` for the full model card including intended use, out-of-scope uses, limitations, and monitoring requirements as specified by SR 11-7 model risk management guidance.

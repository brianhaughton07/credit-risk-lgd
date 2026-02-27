# Credit Risk: Loss Given Default (LGD) Prediction
### End-to-End ML Pipeline · Freddie Mac Single Family Loan Performance Data

---

## Project Summary

Predict **Loss Given Default (LGD)** — the fraction of a loan balance unrecovered after default — using Freddie Mac's publicly available Single Family Loan Performance dataset. LGD is a core regulatory capital calculation under Basel III and a live concern for every mortgage lender, servicer, and investor. Framing the problem this way signals genuine domain fluency rather than a generic regression exercise.

**Target variable:** `loss_given_default` — a continuous value in [0, 1] representing the proportion of outstanding balance lost at resolution.

---

## Repository Structure

```
credit-risk-lgd/
│
├── README.md                        # Project overview, findings, how to run
├── LICENSE                          # MIT
├── .gitignore
├── requirements.txt
├── environment.yml                  # Conda env for reproducibility
│
├── data/
│   ├── README.md                    # Data source, download instructions, schema
│   ├── raw/                         # .gitignored — Freddie Mac raw files
│   ├── interim/                     # .gitignored — partially processed
│   └── processed/                   # .gitignored — model-ready features
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA — distributions, missingness, correlations
│   ├── 02_feature_engineering.ipynb # Feature construction and selection
│   ├── 03_modeling_baseline.ipynb   # Baseline models (linear regression, sklearn)
│   ├── 04_modeling_pytorch.ipynb    # PyTorch MLP — architecture search, tuning
│   └── 05_model_evaluation.ipynb   # Final evaluation, SHAP explainability
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingest.py                # Download, validate, and stage raw data
│   │   ├── preprocess.py            # Cleaning, type casting, missing value strategy
│   │   └── features.py              # Feature engineering pipeline
│   ├── models/
│   │   ├── baseline.py              # sklearn baseline (Ridge, GBM)
│   │   ├── lgd_net.py               # PyTorch MLP architecture
│   │   ├── train.py                 # Training loop with MLflow logging
│   │   └── evaluate.py              # Metrics, SHAP, residual analysis
│   └── utils/
│       ├── config.py                # Centralized config (paths, hyperparams)
│       └── logging.py               # Structured logging setup
│
├── api/
│   ├── main.py                      # FastAPI app
│   ├── schemas.py                   # Pydantic request/response models
│   ├── predict.py                   # Model loading and inference logic
│   └── monitoring.py                # Prediction logging for drift detection
│
├── tests/
│   ├── test_features.py             # Unit tests for feature engineering
│   ├── test_model.py                # Model output shape/range assertions
│   └── test_api.py                  # FastAPI endpoint tests (pytest + httpx)
│
├── mlruns/                          # .gitignored — MLflow tracking directory
├── models/                          # .gitignored — serialized model artifacts
│
└── docs/
    ├── data_dictionary.md           # All features, types, derivations
    └── model_card.md                # Model card: intended use, limitations, metrics
```

---

## Data

### Source
**Freddie Mac Single Family Loan Performance Data**
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

Registration required (free). Two file types per quarter:
- **Origination files** — static loan characteristics at origination
- **Performance files** — monthly performance records through resolution

Use a focused subset to start: **2010–2015 origination vintages**, performance data through resolution. This window captures the post-crisis period with meaningful default rates while remaining computationally tractable.

### Target Construction
LGD is not a raw column — you derive it:

```
LGD = Net Loss / Unpaid Principal Balance at Default

Net Loss = (UPB at Default) - (Net Proceeds from Disposition)
         + (Foreclosure Costs) + (MI Recovery, if any, subtract)
```

Loans that do not default are excluded from the training set. This is a **conditional model** — you predict LGD *given* that default has occurred. Note this clearly in the README and model card; it's an important modeling decision that interviewers will ask about.

### Key Features (after engineering)

**Origination characteristics**
- `orig_ltv` — original loan-to-value ratio (most predictive single feature for LGD)
- `orig_upb` — original unpaid principal balance
- `orig_interest_rate`
- `orig_term` — 15 vs 30 year
- `num_borrowers`
- `first_time_homebuyer_flag`
- `property_type` — single family, condo, co-op, etc.
- `occupancy_status` — primary, second home, investment
- `channel` — retail, broker, correspondent (proxy for underwriting quality)
- `state` — encode carefully; too many dummies; consider regional groupings or target encoding

**At-default characteristics** (from performance data)
- `ltv_at_default` — current LTV at time of default; more predictive than original LTV
- `months_delinquent_at_default`
- `modification_flag` — was the loan modified prior to default?
- `hpi_change` — change in House Price Index from origination to default (requires joining external HPI data — worth the effort; it's a strong feature and shows domain knowledge)

**Macroeconomic context** (external join)
- `unemployment_rate_at_default` — county or state level
- `hpi_index_at_default` — FHFA HPI by MSA

---

## Modeling Approach

### Phase 1 — Baseline (sklearn)
Build interpretable baselines first. This is good engineering practice and gives you a benchmark the PyTorch model must beat.

- **Ridge Regression** — linear baseline, useful for understanding linear feature effects
- **Gradient Boosting (XGBoost or LightGBM)** — strong non-parametric baseline; if PyTorch doesn't clearly beat this, that's a finding worth documenting honestly

Metrics to track for every model:
- **MAE** (Mean Absolute Error) — primary metric; intuitive in business terms
- **RMSE** — penalizes large errors more; relevant for tail risk
- **R²** — overall explanatory power
- **Calibration plot** — predicted vs actual LGD bucketed; critical for regulatory use

### Phase 2 — PyTorch MLP (LGDNet)

```python
# Illustrative architecture — tune during development
class LGDNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())   # Constrains output to [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()
```

**Loss function:** Consider a **Beta regression loss** or a weighted MSE that penalizes errors in the tails (LGD = 0 and LGD = 1 are common spikes). Standard MSE underweights these. Document your choice.

**Training discipline:**
- Train/validation/test split: 70/15/15, stratified by vintage year to avoid data leakage
- Early stopping on validation MAE with patience=10
- Learning rate scheduler: ReduceLROnPlateau
- All hyperparameter choices logged to MLflow

### Phase 3 — Explainability (SHAP)
Use `shap.DeepExplainer` for the PyTorch model. Generate:
- Global feature importance bar chart
- Beeswarm plot showing feature value effects
- Waterfall plot for a sample high-LGD and low-LGD prediction

This is non-negotiable for a credit risk model — regulators and business stakeholders require explanation. Including it demonstrates you understand the deployment context, not just the modeling.

---

## MLflow Experiment Tracking

Every training run logs:
- All hyperparameters
- Train/val/test MAE, RMSE, R²
- Calibration plot as artifact
- Model artifact (serialized with `mlflow.pytorch.log_model`)
- SHAP summary plot as artifact
- Git commit hash (for reproducibility)

```python
with mlflow.start_run(run_name=run_name):
    mlflow.log_params(hyperparams)
    mlflow.log_metrics({"val_mae": val_mae, "val_rmse": val_rmse, "r2": r2})
    mlflow.pytorch.log_model(model, "lgd_model")
    mlflow.log_artifact("reports/calibration_plot.png")
    mlflow.set_tag("git_commit", git_hash)
```

Run the MLflow UI locally:
```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## FastAPI Service

The trained model is served as a REST API. This is what separates a portfolio project from a notebook exercise.

### Endpoints

```
POST /predict          — Single loan LGD prediction
POST /predict/batch    — Batch prediction (list of loans)
GET  /health           — Service health check
GET  /model/info       — Current model version, training metrics, feature list
```

### Request/Response Schema (Pydantic)

```python
class LoanFeatures(BaseModel):
    orig_ltv: float = Field(..., ge=0, le=200, description="Original LTV ratio")
    orig_upb: float = Field(..., gt=0)
    ltv_at_default: float = Field(..., ge=0, le=300)
    months_delinquent: int = Field(..., ge=1, le=120)
    property_type: Literal["SF", "CO", "CP", "MH", "PU"]
    occupancy_status: Literal["P", "S", "I"]
    state: str = Field(..., min_length=2, max_length=2)
    hpi_change: float
    modification_flag: bool

class LGDPrediction(BaseModel):
    lgd: float = Field(..., ge=0, le=1, description="Predicted loss given default")
    confidence_interval_90: tuple[float, float]
    model_version: str
    prediction_id: str   # UUID for monitoring
```

### Monitoring Hook
Every prediction is logged to a local SQLite store (or CSV append) with timestamp, input features, and predicted LGD. This enables:
- Input drift detection (compare feature distributions over time vs training data)
- Prediction drift (is the output distribution shifting?)
- A simple `GET /monitoring/summary` endpoint that returns basic drift statistics

In production this would feed a proper monitoring stack; for the portfolio, the pattern matters more than the scale.

---

## Model Card

Document in `docs/model_card.md`:

- **Intended use:** Internal credit risk estimation for residential mortgage portfolios; regulatory capital calculation support
- **Out-of-scope use:** Decisions on individual loan approvals; real-time consumer-facing applications
- **Training data:** Freddie Mac SFLP, 2010–2015 vintages, defaulted loans only
- **Performance:** MAE on held-out test set (state this clearly)
- **Known limitations:** Conditional model — does not predict probability of default; performance may degrade for loan types underrepresented in training data; no guarantee of performance outside the origination vintage range
- **Ethical considerations:** LTV and geography are legitimate credit risk factors; state-level encoding should be monitored for disparate impact by protected class

The model card is a small but high-signal addition. Most portfolio projects don't have one.

---

## How to Run

```bash
# 1. Clone and set up environment
git clone https://github.com/brianhaughton07/credit-risk-lgd.git
cd credit-risk-lgd
conda env create -f environment.yml
conda activate credit-risk-lgd

# 2. Download data (follow instructions in data/README.md)

# 3. Run the pipeline
python src/data/ingest.py
python src/data/preprocess.py
python src/data/features.py
python src/models/train.py --config configs/default.yaml

# 4. View experiment results
mlflow ui

# 5. Start the API
uvicorn api.main:app --reload

# 6. Run tests
pytest tests/
```

---

## Build Sequence (Recommended Order)

This is a large project. Build it incrementally so each phase is shippable.

| Phase | Deliverable | Signal It Sends |
|---|---|---|
| 1 | EDA notebook + data README | Domain understanding, communication |
| 2 | Feature engineering + data pipeline | Engineering discipline |
| 3 | Baseline sklearn models + MLflow | Methodical model development |
| 4 | PyTorch LGDNet + SHAP | ML depth |
| 5 | FastAPI service + Pydantic schemas | Production engineering instinct |
| 6 | Monitoring hooks + model card | Deployment maturity |

Push each phase as a PR-style commit with a meaningful message. Even working alone, this commit history tells a story.

---

## Technologies

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| Baseline models | scikit-learn, XGBoost |
| Deep learning | PyTorch |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Testing | pytest, httpx |
| Visualization | matplotlib, seaborn |
| Environment | Conda, requirements.txt |

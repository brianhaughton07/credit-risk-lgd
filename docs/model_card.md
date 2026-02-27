# Model Card: LGDNet
## Loss Given Default Prediction for Residential Mortgages

---

## Model Summary

| Field | Value |
|---|---|
| **Model name** | LGDNet |
| **Version** | v1 |
| **Type** | Multi-layer perceptron (PyTorch) |
| **Task** | Regression — predict LGD ∈ [0, 1] |
| **Training data** | Freddie Mac SFLP, 2010–2015 origination vintages, defaulted loans only |
| **Model owner** | Credit Risk Analytics |
| **Last updated** | 2024 |

---

## Intended Use

**Primary use cases:**
1. **CECL reserve estimation** — estimate expected credit losses for mortgage portfolios under ASC 326. LGDNet provides loan-level LGD estimates that feed into expected credit loss = PD × LGD × EAD calculations.
2. **Macro stress testing** — estimate portfolio LGD under adverse macroeconomic scenarios (rising unemployment, declining home prices) as required under DFAST/CCAR frameworks.

**Intended users:**
- Credit risk analytics teams at banks, mortgage servicers, and GSEs
- Model risk management functions performing model validation
- Regulators reviewing internal capital models

---

## Out-of-Scope Uses

This model is **not** intended for:
- **Individual loan approval decisions** — LGDNet is a conditional model that assumes default has already occurred. It does not assess creditworthiness.
- **Real-time consumer-facing applications** — The model is designed for batch portfolio analysis, not consumer credit scoring.
- **Commercial real estate or auto loans** — Trained exclusively on single-family residential mortgages. Application to other loan types is not validated.
- **Origination vintages outside 2010–2015** — Performance may degrade for pre-crisis origination cohorts (different underwriting standards) or very recent originations (different rate/HPI environment).
- **Non-U.S. mortgages** — Geographic features are calibrated to U.S. regional dynamics.

---

## Architecture

```
Input (12 features)
     ↓
Linear(12 → 256) → BatchNorm → ReLU → Dropout(0.3)
     ↓
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.3)
     ↓
Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
     ↓
Linear(64 → 1) → Sigmoid
     ↓
Output: LGD ∈ [0, 1]
```

**Parameters:** ~50,000

---

## Loss Function

Weighted MSE with tail emphasis:

```python
weight = 1 + alpha * (|y - 0.5| * 2)²
loss = mean(weight * (pred - y)²)
```

With `alpha = 2.0`:
- At y = 0.0 or 1.0: weight = 3.0 (3× more weight than center)
- At y = 0.5: weight = 1.0 (standard MSE)

**Rationale:** LGD distributions have bi-modal spikes at 0 (full recovery via short sale or rapid REO) and 1 (total loss on vacant/abandoned properties). These tail outcomes drive concentration risk in stress scenarios. Standard MSE underweights them because they represent a smaller fraction of the distribution mass. Regulators specifically focus on the LGD tail when reviewing stress test assumptions.

---

## Training Protocol

| Parameter | Value |
|---|---|
| Train / Val / Test split | 70 / 15 / 15 |
| Stratification | By vintage year (2010–2015) |
| Optimizer | Adam (lr = 0.001) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | Val MAE, patience = 10 |
| Max epochs | 100 |
| Batch size | 512 |
| Gradient clipping | max_norm = 1.0 |

**Data leakage prevention:** Train/val/test splits are stratified by vintage year to ensure that the model is evaluated on loans from the same origination cohorts as training, but never tested on loans it was trained on. No performance data from the post-default period is used as a feature (only resolution outcomes).

---

## Performance Metrics

*To be completed after training on full dataset.*

| Metric | Train | Val | Test | Segment-Avg Benchmark | Improvement |
|---|---|---|---|---|---|
| MAE | TBD | TBD | TBD | TBD | Target: ≥15% |
| RMSE | TBD | TBD | TBD | TBD | Target: ≥10% |
| R² | TBD | TBD | TBD | N/A | Target: >0.30 |

**Benchmark:** Segment-average historical LGD rates grouped by (property_type, occupancy_status, vintage_year). This is the approach most institutions currently use for portfolio provisioning. The improvement targets represent the minimum business value threshold for adopting a more complex model.

**Calibration:** Assessed via predicted-vs-actual decile plot. The model is considered calibrated if predictions in each decile are within ±5 percentage points of realized LGD.

**Macroeconomic sensitivity:** Verified via scenario analysis (see `src/models/evaluate.py`). Rising unemployment (+2%, +4%, +6%) and falling HPI (-10%, -20%, -30%) should each produce monotonically increasing predicted LGD.

---

## Uncertainty Quantification

Prediction confidence intervals are generated via **MC Dropout** (100 forward passes with dropout enabled at inference time). This approximates epistemic (model) uncertainty.

**Limitations of MC Dropout CI:**
- Does not quantify aleatoric uncertainty (irreducible outcome uncertainty)
- CIs are not rigorously calibrated — should not be interpreted as frequentist confidence intervals
- CIs widen appropriately for out-of-distribution inputs, providing a useful signal but not a formal bound

For regulatory stress testing, the mean prediction should be used. CIs provide useful supplementary information for single-loan analysis.

---

## Explainability

SHAP DeepExplainer is used to generate:
- **Global feature importance** — mean |SHAP| bar chart
- **Beeswarm plot** — feature value effects across test set
- **Waterfall plots** — individual prediction explanations for sample high-LGD and low-LGD loans

Expected SHAP feature ranking (based on domain theory; to be verified against actual model output):
1. `ltv_at_default` — primary determinant of collateral coverage
2. `hpi_change` — macroeconomic collateral erosion
3. `occupancy_status` — investment properties have higher LGD
4. `orig_ltv` — initial equity cushion
5. `unemployment_rate_at_default` — macroeconomic distress signal

If the realized SHAP rankings diverge significantly from domain expectations, this should be documented and investigated before deployment.

---

## Known Limitations

| Limitation | Description | Mitigation |
|---|---|---|
| **Conditional model** | Predicts LGD *given* default — does not estimate probability of default. Must be combined with a separate PD model for full ECL calculation. | Clearly documented; PD × LGD × EAD workflow required. |
| **Vintage constraint** | Trained on 2010–2015 originations (post-crisis). May underestimate LGD for pre-crisis high-LTV vintages or overestimate for stronger underwriting environments. | Monitor performance by origination vintage; retrain as new data accumulates. |
| **HPI placeholder** | In production, `hpi_change` requires MSA-level FHFA HPI data joined at origination and default dates. The pipeline currently provides a national average placeholder. | Implement full MSA-level HPI join before production deployment. |
| **Unemployment placeholder** | `unemployment_rate_at_default` uses a state-level average placeholder. County-level data would improve granularity. | Use BLS LAUS county data in production implementation. |
| **Static model** | Model is static once trained. LGD dynamics may shift in new interest rate or housing market environments. | Schedule quarterly monitoring reviews; trigger retraining if MAE degrades >5% from baseline. |
| **Single-model ensemble** | Uses a single neural network. Ensemble methods (e.g., average of 5 models with different seeds) would provide more stable estimates. | Consider ensembling in v2 if model uncertainty is a regulatory concern. |

---

## Ethical Considerations

**Geographic encoding:** State-level features are encoded as census regions (Northeast, Southeast, Midwest, Southwest, West). This is a legitimate credit risk factor (regional real estate markets differ materially). Regional encoding should be monitored for disparate impact on protected classes if the model is used in any underwriting-adjacent context.

**Investment property flag:** `occupancy_status = I` (investment property) increases predicted LGD. This is supported by empirical LGD literature and reflects legitimate servicer engagement differences. It is not a protected characteristic under ECOA/FHA.

**Out-of-distribution behavior:** The model should not be used for loan types or geographies substantially underrepresented in the training data (e.g., agricultural properties, U.S. territories). SHAP waterfall plots for out-of-distribution inputs should be reviewed before using predictions.

---

## Regulatory Context

This model is designed to support compliance with:
- **ASC 326 (CECL)** — Expected credit loss estimation requirement
- **DFAST / CCAR** — Federal stress testing framework
- **SR 11-7** — Federal Reserve guidance on model risk management

Under SR 11-7, this model should undergo:
- Independent model validation prior to use in regulatory capital calculations
- Documentation review by Model Risk Management
- Annual backtesting against realized loss data

This model card is part of that documentation. A complete model validation package should also include: developmental evidence, outcomes analysis, sensitivity analysis, and limitations assessment.

---

## Monitoring Requirements

| Metric | Threshold | Frequency |
|---|---|---|
| Mean predicted LGD vs training mean | ±10% | Monthly |
| Input feature distributions (KS statistic) | > 0.1 for any feature | Monthly |
| MAE on new defaulted loans | > 5% degradation from test MAE | Quarterly |
| SHAP top-3 feature stability | Rank change > 2 positions | Quarterly |

Monitoring data is logged via the `/monitoring/summary` API endpoint and stored in `api/predictions.db`.

---

## Version History

| Version | Date | Changes |
|---|---|---|
| v1 | 2024 | Initial model trained on 2010–2015 Freddie Mac SFLP data |

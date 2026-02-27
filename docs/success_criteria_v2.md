# Success Criteria
## Credit Risk: Loss Given Default (LGD) Prediction

---

### Framing

A model is successful not when it achieves a threshold in isolation, but when it produces estimates that are more accurate, more defensible, and more actionable than what the institution is currently using. All criteria below are evaluated against a benchmark of segment-average historical LGD rates — the approach most institutions currently rely on — because that is the actual alternative this model is competing with.

---

### Primary Criteria

**Predictive Accuracy**

The model must demonstrate measurable improvement over the historical average benchmark on held-out test data representing loans the model did not see during training.

| Metric | Definition | Target |
|---|---|---|
| Mean Absolute Error (MAE) | Average absolute difference between predicted and realized LGD | Minimum 15% reduction vs. benchmark |
| Root Mean Squared Error (RMSE) | Penalizes large errors more than small ones; relevant for tail exposure | Minimum 10% reduction vs. benchmark |
| R² | Proportion of LGD variance explained by the model | > 0.30 on test set |

MAE is the primary metric. In provisioning and stress testing contexts, average error magnitude has a direct dollar interpretation that business stakeholders can act on — a 1% MAE improvement on a $1B defaulted portfolio represents $10M in reserve accuracy.

**Calibration**

A well-calibrated model produces predicted LGD distributions that match realized distributions across the full range of outcomes, not just at the mean. Calibration is assessed through a predicted-versus-actual decile plot and a statistical calibration test. A model that is accurate on average but systematically biased in the tails — where concentration risk actually lives — is not fit for stress testing purposes, regardless of its aggregate error metrics.

**Macroeconomic Sensitivity**

The model must respond in economically coherent directions when stress scenario inputs change. Rising unemployment should increase predicted LGD. Declining home prices should increase predicted LGD. These sensitivities need to be directionally consistent and of plausible magnitude relative to the historical data. This criterion is evaluated through scenario analysis in the model evaluation notebook rather than by a single summary statistic, because it requires examining the model's behavior across a range of conditions rather than its average performance on a fixed test set.

---

### Secondary Criteria

**Explainability**

Individual predictions must be explainable at the feature level using SHAP values. This is a practical regulatory requirement — model risk management functions at regulated institutions expect it — and a credibility requirement for risk management adoption. Feature attributions must be economically sensible: LTV at default and HPI change should rank among the strongest predictors. A model that produces accurate estimates but cannot explain them is operationally limited in the contexts this project is designed to serve.

**Operational Readiness**

The model is operationally ready when predictions are served reliably via a REST API with sub-200ms response time for single-loan inference, input validation rejects malformed requests with informative error messages, prediction logs are captured for ongoing monitoring, and the full pipeline from data ingestion through inference is reproducible from a clean environment using documented instructions.

**Documentation**

The model ships with a completed model card covering intended use, known limitations, training data characteristics, and performance metrics. This is not a supplementary deliverable — it is a required one. Risk management functions at regulated institutions expect model documentation, and a portfolio project that omits it signals unfamiliarity with how models actually get used.

---

### What Success Does Not Mean

LGD has irreducible uncertainty. A realized loss depends on factors that are unknowable at prediction time — how long the foreclosure process takes, what the local market looks like at the moment of property disposition, whether a borrower reengages during the loss mitigation process. The goal is better estimates, not perfect ones, and the success criteria above are calibrated accordingly.

It also bears saying that if the Gradient Boosting baseline performs comparably to the PyTorch model, that is a legitimate finding and will be documented as such. Demonstrating that a simpler model is sufficient for a given problem is as valuable as demonstrating that a more complex one is necessary.

---

### Evaluation Checkpoints

| Milestone | Criteria Evaluated |
|---|---|
| Baseline models complete | Accuracy vs. benchmark (MAE, RMSE, R²) |
| LGDNet training complete | Accuracy, calibration, comparison to baseline |
| Explainability analysis complete | SHAP feature rankings, directional coherence |
| API complete | Latency, input validation, prediction logging |
| Project complete | Full criteria review, model card finalized |

---

*Document version 1.0 — for review and alignment prior to model development*

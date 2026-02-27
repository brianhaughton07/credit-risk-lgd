# Business Problem Statement
## Credit Risk: Loss Given Default (LGD) Prediction

---

### The Problem

Financial institutions holding residential mortgage portfolios face a measurement problem that is both technically difficult and consequentially expensive: when a loan defaults, how much of the outstanding balance will actually be lost? This quantity — Loss Given Default, or LGD — is not observable until a defaulted loan reaches final resolution, a process that routinely takes years. Yet institutions must estimate it today, at the loan level, with enough accuracy to satisfy two obligations that are difficult to meet simultaneously.

**Loan Loss Provisioning under CECL.** The Current Expected Credit Loss standard (ASC 326) requires institutions to recognize lifetime expected credit losses at origination and to update those estimates each reporting period. LGD is a direct multiplier in that calculation. Underestimating it creates reserve shortfalls and regulatory exposure; overestimating it unnecessarily constrains capital and suppresses reported earnings. The cost of imprecision runs in both directions, and the margin for error narrows as portfolio scale increases.

**Portfolio Risk Management and Stress Testing.** Regulators and internal risk functions require forward-looking loss estimates under adverse economic scenarios — declining home prices, rising unemployment, regional market disruption. LGD is sensitive to exactly those conditions, meaning a model that performs adequately under normal circumstances may produce unreliable estimates precisely when accuracy matters most. Institutions need estimates that respond to macroeconomic inputs in a principled, defensible way.

---

### Why Existing Approaches Fall Short

Most institutions estimate LGD using historical average loss rates, segmented by broad loan categories. That approach has three practical limitations that become more significant as portfolios grow and regulatory expectations tighten.

Historical averages are backward-looking by construction. They reflect prior loss experience under prior economic conditions, not the current composition of the portfolio or the current direction of the housing market. They collapse meaningful variation — LTV at default, property type, geographic concentration, borrower characteristics — into segment-level summaries that may obscure material differences in individual loan risk. And they cannot respond to stress scenario inputs in a principled way, which means stress test results that rely on them require management judgment overlays that are difficult to audit and harder to defend.

---

### The Opportunity

A machine learning model trained on granular loan-level performance data — with features capturing origination characteristics, the state of the loan at default, and macroeconomic context — can produce materially more accurate LGD estimates than segment averages allow. The practical value of that improvement is not abstract: a reduction in mean absolute error of one percentage point applied to a $1B defaulted portfolio represents $10M in reserve accuracy, in either direction. At the portfolio level, the business case for better estimation is straightforward.

---

### Scope

This project builds and validates an LGD prediction model for residential single-family mortgages, trained on Freddie Mac Single Family Loan Performance data covering the 2010–2015 origination vintages. The model is scoped to loans that have already defaulted; it does not predict probability of default. Combining this model with a separate PD model to produce expected loss (EL = PD × LGD × EAD) is a natural extension, but outside the current scope.

---

*Document version 1.0 — for review and alignment prior to model development*

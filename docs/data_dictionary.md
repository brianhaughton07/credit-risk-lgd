# Data Dictionary
## Credit Risk: Loss Given Default (LGD) Prediction

---

## Target Variable

| Feature | Type | Range | Derivation | Notes |
|---|---|---|---|---|
| `loss_given_default` | float | [0, 1] | `Net Loss / UPB at Default` | 0 = full recovery, 1 = total loss. Computed only for defaulted loans. Capped at 1.0. |

**Derivation formula:**
```
LGD = Net Loss / UPB at Default

Net Loss = UPB_at_default - Net_Proceeds + Foreclosure_Costs - MI_Recovery

where:
  UPB_at_default             = current_upb (pos 3)  — UPB at the resolution record
  Net_Proceeds               = net_sale_proceeds (pos 15) — net proceeds from property sale
  Foreclosure_Costs          = expenses (pos 17) — total expenses at resolution
  MI_Recovery                = mi_recoveries (pos 14) — mortgage insurance recoveries
```

Column positions refer to the Monthly Performance Data File layout in `FreddieMac_SFH_file_layout.xlsx`. When all three component columns are present, the formula above is applied. When they are not, `actual_loss` (pos 22, "Actual Loss Calculation") is used directly as the pre-computed net loss.

---

## Origination Features

| Feature | Type | Range / Values | Source | Notes |
|---|---|---|---|---|
| `orig_ltv` | float | [0, 200] | Freddie Mac origination file, field 12 | Original loan-to-value ratio (%). Most predictive single feature for LGD. High LTV → less equity cushion → higher loss on default. |
| `orig_upb` | float | > 0 | Freddie Mac origination file, field 11 | Original unpaid principal balance ($). Proxy for loan size. Larger loans may have different loss dynamics. |
| `orig_interest_rate` | float | [0, 30] | Freddie Mac origination file, field 13 | Interest rate at origination (%). Higher rate may proxy for higher credit risk at origination. |
| `orig_term` | int | [60, 480] | Freddie Mac origination file, field 22 | Original loan term in months. 360 = 30-year, 180 = 15-year. Shorter terms build equity faster, reducing LGD risk. |
| `property_type` | categorical | SF, CO, CP, MH, PU | Freddie Mac origination file, field 18 | SF=Single Family, CO=Condo, CP=Co-op, MH=Manufactured Housing, PU=Planned Unit Development. Property type affects marketability and thus recovery rates. |
| `occupancy_status` | categorical | P, S, I | Freddie Mac origination file, field 8 | P=Primary, S=Second Home, I=Investment Property. Investment properties typically have higher LGD due to lower servicer engagement. |
| `channel` | categorical | R, B, C, T | Freddie Mac origination file, field 14 | R=Retail, B=Broker, C=Correspondent, T=TPO. Proxy for underwriting quality and borrower relationship. |
| `state` | categorical | 2-letter state code | Freddie Mac origination file, field 17 | State of property. Encoded as census region (5 groups) to reduce dimensionality. Regional real estate market dynamics affect recovery. |

---

## At-Default Features

| Feature | Type | Range | Source | Derivation |
|---|---|---|---|---|
| `ltv_at_default` | float | [0, 300] | Computed | `current_upb / estimated_property_value_at_default × 100`. Estimated property value = `orig_upb / (orig_ltv/100) × (1 + hpi_change)`. More predictive than orig_ltv because it captures equity erosion. |
| `months_delinquent_at_default` | int | [1, 120] | Freddie Mac performance file | Number of months delinquent when the loan reached final resolution. Proxy for workout duration; longer delinquency typically correlates with higher costs. |

---

## Macroeconomic Features

| Feature | Type | Range | Source | Notes |
|---|---|---|---|---|
| `hpi_change` | float | [-1.0, 5.0] | Computed from FHFA HPI | `(HPI_at_default / HPI_at_origination) - 1`. Negative = price decline. Critical regulator stress variable. Declining HPI → lower collateral value → higher LGD. |
| `unemployment_rate_at_default` | float | [0, 50] | BLS LAUS | State-level unemployment rate at the month of default. Rising unemployment → distressed borrowers, fewer alternative buyers, higher foreclosure costs. |

---

## Engineered Features (Internal)

| Feature | Type | Source | Notes |
|---|---|---|---|
| `region` | categorical | Derived from `state` | Census region mapping: Northeast, Southeast, Midwest, Southwest, West. Used internally; `state` is the API-facing field. |
| `modification_flag_numeric` | int | {0, 1} | Derived from `modification_flag` (Y/N) | 1 = loan was modified prior to default. Modifications may indicate weaker borrower, but can also reduce LGD by preserving equity. |
| `vintage_year` | int | [2010, 2015] | Derived from `first_payment_date` | Used for stratified train/val/test splits only. Not included as a model input feature — including it would encode macroeconomic regime, which is already captured by `hpi_change` and `unemployment_rate_at_default`. |

---

## Features NOT Included (and Why)

| Feature | Reason Excluded |
|---|---|
| `credit_score` | Valid predictor of PD (probability of default); less theoretically motivated for LGD after default has already occurred. Could be added in a future version. |
| `orig_dti` | Available at origination but missing rate is high (>20%); imputation at this rate would introduce substantial noise. |
| `seller_name` / `servicer_name` | Too many unique values; would require complex encoding and risk overfitting to specific institutions. |
| `loan_purpose` | Weak predictor in domain literature for LGD specifically; excluded to reduce noise. |
| `postal_code` / `msa` | Highly granular, many missing values; regional encoding via `state` → `region` captures the key geographic variation. |

---

## Missing Value Strategy

| Feature | Strategy | Rationale |
|---|---|---|
| `credit_score` | Median imputation | Minority missing; median is robust to distribution shape |
| `orig_dti` | Median imputation | Minority missing; median preferred over mean given right skew |
| `mip` (mortgage insurance premium) | Fill with 0 | No MI is a meaningful state, not a missing observation |
| `expenses`, `legal_costs`, `maintenance_costs`, `taxes_insurance`, `misc_expenses`, `mi_recoveries`, `non_mi_recoveries` | Fill with 0 | Missing indicates no cost incurred or no recovery received, not an unknown amount |
| `modification_flag` | Fill with 'N' | Absence of record indicates no modification |

---

## Data Sources

| Source | URL | License |
|---|---|---|
| Freddie Mac SFLP | https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset | Free for research (registration required) |
| FHFA HPI | https://www.fhfa.gov/DataTools/Downloads/Pages/House-Price-Index-Datasets.aspx | Public domain |
| BLS LAUS | https://www.bls.gov/lau/ | Public domain |

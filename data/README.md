# Data

## Source

**Freddie Mac Single Family Loan Performance (SFLP) Data**

URL: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

Registration is required but free. The dataset is publicly available for research purposes.

## Download Instructions

1. Create a free account at https://freddiemac.com
2. Navigate to Research → Datasets → Single Family Loan-Level Dataset
3. Accept the Terms and Conditions
4. Download the following files for **2010–2015 origination vintages**:

### Files Required

**Origination files** (`historical_data_<YEAR>Q<Q>.zip`):
- `historical_data_2010Q1.zip` through `historical_data_2015Q4.zip`

Each zip contains a pipe-delimited origination file: `historical_data_<YEAR>Q<Q>.txt`

**Performance files** (`historical_data_time_<YEAR>Q<Q>.zip`):
- `historical_data_time_2010Q1.zip` through `historical_data_time_2015Q4.zip`

Each zip contains a pipe-delimited performance file: `historical_data_time_<YEAR>Q<Q>.txt`

5. Extract all files into `data/raw/origination/` and `data/raw/performance/` respectively.

## Directory Structure

```
data/
├── README.md               # This file
├── raw/
│   ├── origination/        # Raw origination text files (gitignored)
│   └── performance/        # Raw performance text files (gitignored)
├── interim/                # Partially processed files (gitignored)
└── processed/              # Model-ready feature matrices (gitignored)
```

## Schema

### Origination File Columns

| Position | Name | Description |
|---|---|---|
| 1 | credit_score | FICO score at origination |
| 2 | first_payment_date | YYYYMM |
| 3 | first_time_homebuyer | Y/N/U |
| 4 | maturity_date | YYYYMM |
| 5 | msa | Metropolitan Statistical Area code |
| 6 | mip | Mortgage Insurance Percentage |
| 7 | num_units | Number of units |
| 8 | occupancy_status | P=Primary, S=Second Home, I=Investment |
| 9 | orig_cltv | Original combined LTV |
| 10 | orig_dti | Original debt-to-income ratio |
| 11 | orig_upb | Original unpaid principal balance |
| 12 | orig_ltv | Original loan-to-value ratio |
| 13 | orig_interest_rate | Interest rate at origination |
| 14 | channel | R=Retail, B=Broker, C=Correspondent, T=TPO |
| 15 | prepayment_penalty | Y/N |
| 16 | amortization_type | FRM/ARM |
| 17 | property_state | Two-letter state code |
| 18 | property_type | SF/CO/CP/MH/PU |
| 19 | postal_code | 5-digit zip |
| 20 | loan_seq_num | Unique loan identifier |
| 21 | loan_purpose | P=Purchase, C=Cashout Refi, N=No-Cashout Refi |
| 22 | orig_loan_term | Loan term in months |
| 23 | num_borrowers | Number of borrowers |
| 24 | seller_name | Seller/originator name |
| 25 | servicer_name | Current servicer name |
| 26 | super_conforming_flag | Y/N |

### Performance File Columns (key fields)

| Position | Name | Description |
|---|---|---|
| 1 | loan_seq_num | Unique loan identifier (join key) |
| 2 | monthly_reporting_period | YYYYMM |
| 3 | current_upb | Current unpaid principal balance |
| 4 | loan_age | Months since origination |
| 5 | remaining_months | Months remaining to maturity |
| 6 | adj_months_to_maturity | Adjusted months to maturity |
| 7 | maturity_date | YYYYMM |
| 8 | msa | MSA at reporting period |
| 9 | current_delinquency_status | 0-6+ months delinquent, RA=REO, F=Foreclosure |
| 10 | modification_flag | Y/N |
| 11 | zero_balance_code | Disposition code: 01=Prepaid, 02=3rd-party sale, 03=Short sale, 06=Repurchase, 09=REO |
| 12 | zero_balance_effective_date | YYYYMM of disposition |
| 13 | last_paid_installment_date | YYYYMM |
| 14 | foreclosure_date | YYYYMM |
| 15 | disposition_date | YYYYMM |
| 16 | foreclosure_costs | Costs incurred during foreclosure |
| 17 | property_preservation_costs | Property preservation costs |
| 18 | asset_recovery_costs | Other asset recovery costs |
| 19 | misc_holding_expenses | Miscellaneous holding costs |
| 20 | associated_taxes | Taxes associated with property |
| 21 | net_sale_proceeds | Net proceeds from property sale |
| 22 | credit_enhancement_proceeds | MI or other credit enhancement receipts |
| 23 | repurchase_make_whole | Repurchase or make-whole proceeds |
| 24 | other_foreclosure_proceeds | Other proceeds |
| 25 | non_mi_recovery | Non-MI recovery amounts |
| 26 | net_recovery | Net recovery amount |
| 27 | net_loss | Net loss on the loan |
| 28 | modification_flag_2 | Secondary modification flag |

## LGD Target Construction

LGD is computed only for loans that reach default (zero_balance_code in {02, 03, 06, 09}):

```
LGD = Net Loss / UPB at Default

where:
  Net Loss = UPB_at_default - Net_Proceeds + Foreclosure_Costs - MI_Recovery
  UPB_at_default = current_upb at the time of first serious delinquency (90+ days)
```

- LGD is capped at 1.0 (losses cannot exceed the outstanding balance)
- LGD = 0.0 (full recovery) is a valid outcome and is included in training
- Loans without default are excluded — this is a conditional LGD model

## External Data Sources

**FHFA House Price Index (HPI)**
- URL: https://www.fhfa.gov/DataTools/Downloads/Pages/House-Price-Index-Datasets.aspx
- Download the "All Transactions" HPI at the MSA level
- Used to compute `hpi_change` (HPI index at default / HPI index at origination - 1)

**Bureau of Labor Statistics Unemployment Rate**
- URL: https://www.bls.gov/lau/ (Local Area Unemployment Statistics)
- Download state-level monthly unemployment rates
- Used for `unemployment_rate_at_default`

## Data Volume

Approximate record counts after filtering to defaulted loans (2010–2015 vintages):
- Origination records: ~150,000–200,000 defaulted loans
- Performance records: ~5–10M rows (all monthly snapshots)

Processing the full dataset requires ~16GB RAM. For development, a 10% random sample is sufficient for pipeline testing.

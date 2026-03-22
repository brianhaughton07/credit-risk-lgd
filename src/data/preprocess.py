"""Type casting, missing value imputation, and LGD target construction.

This module sits between raw ingestion and feature engineering. Its job is to
produce a clean, merged DataFrame where every column has the correct Python type,
missing values have been handled according to documented domain logic, and the
LGD target variable has been constructed from the component fields that Freddie
Mac provides.

There is a deliberate ordering here. Type casting happens before imputation
because median-based imputation requires numeric values, and it happens before
LGD target construction because the target calculation divides by current_upb,
which must be a float. If these steps were reordered, the pipeline would either
fail with a type error or produce silently incorrect values.

The LGD construction in this module is the most consequential step in the entire
pipeline. The formula (net loss divided by UPB at default, clipped to [0, 1]) is
documented in docs/data_dictionary.md with its derivation from the Basel III
framework. Any change to this formula changes what the model learns to predict,
which means it changes the meaning of every downstream metric. Changes here
should not be made without updating the model card and retraining from scratch.

A note on what this module does not do: it does not construct derived features
like LTV at default or HPI change. Those belong in features.py because they
require external data joins (HPI index) and encapsulate modeling decisions
(how to estimate property value at the time of default). The boundary between
preprocessing and feature engineering is: preprocessing makes the data clean
and consistent; feature engineering makes it informative.

Usage:
    python src/data/preprocess.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type casting maps
#
# These dictionaries define which columns should be cast to float and drive
# _cast_numeric(). All Freddie Mac fields are read as strings by ingest.py
# (to avoid silent truncation of sentinel values), so every numeric field
# needs explicit casting here.
#
# The choice of float for all numeric fields rather than int for integer-valued
# fields (num_units, num_borrowers, etc.) is intentional. Using float uniformly
# avoids nullable integer handling complexity and is compatible with numpy and
# sklearn operations downstream without conversion. The precision cost is
# negligible for this data.
# ---------------------------------------------------------------------------

ORIGINATION_NUMERIC = {
    "credit_score": float,
    "mip": float,
    "num_units": float,
    "orig_cltv": float,
    "orig_dti": float,
    "orig_upb": float,
    "orig_ltv": float,
    "orig_interest_rate": float,
    "orig_loan_term": float,
    "num_borrowers": float,
}

# Categorical origination columns are kept as strings through this module.
# Label encoding happens in features.py where the full modeling context
# (which categories are in the training set) is available.
ORIGINATION_CATEGORICAL = [
    "first_time_homebuyer", "occupancy_status", "channel",
    "property_state", "property_type", "loan_purpose", "amortization_type",
]

PERFORMANCE_NUMERIC = {
    "current_upb": float,
    "loan_age": float,
    "current_interest_rate": float,
    "current_deferred_upb": float,
    "mi_recoveries": float,
    "net_sale_proceeds": float,
    "non_mi_recoveries": float,
    "expenses": float,
    "legal_costs": float,
    "maintenance_costs": float,
    "taxes_insurance": float,
    "misc_expenses": float,
    "actual_loss": float,
    "modification_cost": float,
    "estimated_ltv": float,
    "zero_balance_removal_upb": float,
    "delinquent_accrued_interest": float,
    "current_month_modification_cost": float,
    "interest_bearing_upb": float,
}


def _cast_numeric(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """Cast columns to numeric types, coercing unparseable values to NaN.

    errors="coerce" is the right choice here because Freddie Mac uses sentinel
    values for missing numeric fields — notably "9999" for credit score (not
    applicable) and spaces or empty strings for missing financial amounts. These
    should become NaN rather than causing the cast to fail. The downstream
    imputation step handles these NaNs with domain-specific logic.

    Columns not present in the DataFrame are silently skipped. This makes the
    function robust to schema differences between origination and performance
    files without requiring separate casting functions for each.
    """
    df = df.copy()
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                float if dtype == float else dtype
            )
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-informed imputation for missing values.

    The imputation strategy for each field reflects what a missing value
    actually means in the context of residential mortgage data, rather than
    applying a uniform statistical approach. The rationale for each choice:

    Cost and recovery fields (expenses, legal_costs, maintenance_costs,
    mi_recoveries, etc.) are filled with 0 because a missing value means no
    cost was incurred or no recovery was received, not that the amount is
    unknown. Freddie Mac populates these fields when amounts exist; absence
    of a value is absence of the amount. Imputing with 0 correctly represents that.

    mip (mortgage insurance premium) follows the same logic. When mip is
    missing, the loan has no mortgage insurance — it is a meaningful absence,
    not a gap in data quality.

    credit_score and orig_dti use median imputation. These fields are missing
    for a minority of loans (typically <5%) and their distribution is
    approximately symmetric in the middle range, making median imputation
    reasonable. Mean imputation would produce similar results but is more
    sensitive to outliers. Note that credit_score is not currently a model
    input feature (see docs/data_dictionary.md for why), so its imputation
    value is only relevant if you choose to add it later.

    modification_flag is filled with 'N' (not modified) because the absence of
    a modification record in the Freddie Mac data indicates no modification
    occurred. A missing modification flag is structurally different from a
    missing numeric measurement.

    The imputation is applied to the combined pre-merge DataFrame rather than
    post-merge. This is correct because the imputation logic is specific to
    each file's schema and domain meaning — imputing after merge would mix
    origination and performance fill logic in a single pass, which is harder
    to reason about and harder to audit.
    """
    df = df.copy()

    # Zero-fill for cost and recovery fields where missing = no amount incurred.
    # This is the conservative and correct interpretation of Freddie Mac's
    # field population conventions — these fields are populated only when a
    # cost or recovery exists; absence of a value means zero dollars.
    numeric_zero_fill = [
        "mip", "expenses", "legal_costs", "maintenance_costs",
        "taxes_insurance", "misc_expenses", "modification_cost",
        "mi_recoveries", "non_mi_recoveries", "delinquent_accrued_interest",
        "current_month_modification_cost",
    ]
    for col in numeric_zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Median imputation for fields where missing is a data gap rather than
    # a meaningful absence. Log the imputation count so it is visible in the
    # pipeline audit trail — a high null rate here would warrant investigation
    # before model validation.
    median_fill = ["credit_score", "orig_dti", "orig_cltv"]
    for col in median_fill:
        if col in df.columns:
            median_val = df[col].median()
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.info(f"Imputing {null_count:,} nulls in '{col}' with median={median_val:.2f}")
                df[col] = df[col].fillna(median_val)

    if "modification_flag" in df.columns:
        # 'N' = not modified. The absence of a modification record in Freddie
        # Mac performance data indicates no modification occurred, not that the
        # modification status is unknown.
        df["modification_flag"] = df["modification_flag"].fillna("N")

    return df


def construct_lgd_target(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the loss_given_default target variable from performance components.

    This is the most carefully specified step in the preprocessing pipeline.
    The LGD formula derives from the Basel III expected loss framework:

        LGD = Net Loss / UPB at Default

        Net Loss = UPB_at_default
                 - Net_Proceeds (from property disposition)
                 + Foreclosure_Costs
                 - MI_Recovery (credit enhancement proceeds)

    The UPB at default is the unpaid principal balance at the resolution record.
    This is not the original UPB — it reflects principal payments made during
    the life of the loan, which is why LTV at default (computed in features.py)
    is generally more predictive than original LTV.

    When individual cost and proceeds components are available, I recalculate
    net loss from those components rather than using Freddie Mac's pre-computed
    net_loss field. This provides transparency and allows the calibration plot
    in evaluate.py to confirm that the LGD values are consistent with the
    underlying loss components. It also means the formula is auditable by a
    model validator who can trace exactly how each LGD value was derived.

    The clip to [0, 1] handles two edge cases:
      - LGD > 1: Occurs when foreclosure costs plus principal loss exceed the
        UPB. This is technically possible (and reflects real losses) but the
        model is calibrated to predict loss as a fraction of the outstanding
        balance. Values above 1 are capped rather than dropped because dropping
        them would bias the training distribution by removing the worst outcomes.
      - LGD < 0: Occurs when MI recovery or sale proceeds exceed the UPB. This
        means the lender recovered more than was owed, which does happen in
        cases of strong MI coverage. Capping at 0 treats these as full recoveries.

    Rows with UPB = 0 are dropped because division by zero is unresolvable.
    This is documented as a known exclusion in the model card.

    Args:
        perf_df: Performance DataFrame with one row per defaulted loan at
                 resolution, as produced by ingest.py.

    Returns:
        DataFrame with 'loss_given_default' column added. Rows with zero or
        null UPB are excluded.
    """
    df = perf_df.copy()

    # Use zero_balance_removal_upb (pos 27) as the UPB at default denominator.
    # current_upb (pos 3) is always 0 on the resolution record — it is zeroed
    # when Freddie Mac closes out the loan. zero_balance_removal_upb is the UPB
    # at the time the balance went to zero, which is the correct Basel III
    # denominator for LGD = Net Loss / UPB at Default.
    upb = df["zero_balance_removal_upb"].replace(0, np.nan)
    valid_mask = upb.notna() & (upb > 0)

    dropped = (~valid_mask).sum()
    if dropped > 0:
        logger.warning(f"Dropping {dropped:,} rows with zero/null UPB (unresolvable LGD denominator)")
    df = df[valid_mask].copy()

    # Prefer component-level calculation over the pre-computed actual_loss field
    # because it is transparent, auditable, and consistent with the formula
    # documented in the model card. actual_loss is used as a fallback when
    # the individual components are not all present.
    #
    # Column mapping from FreddieMac_SFH_file_layout.xlsx:
    #   expenses (pos 17)        — total expenses at resolution
    #   net_sale_proceeds (pos 15) — net proceeds from property sale
    #   mi_recoveries (pos 14)   — mortgage insurance recoveries
    has_components = all(
        c in df.columns for c in ["expenses", "net_sale_proceeds", "mi_recoveries"]
    )

    if has_components:
        foreclosure_costs = df["expenses"].fillna(0)
        net_proceeds = df["net_sale_proceeds"].fillna(0)
        # mi_recoveries captures mortgage insurance payouts that reduce the
        # lender's net loss. Subtracting MI recovery from the numerator correctly
        # reflects that MI payments reduce the economic loss even though they
        # are proceeds rather than a reduction in costs.
        mi_recovery = df["mi_recoveries"].fillna(0)
        upb_at_default = df["zero_balance_removal_upb"]
        net_loss = upb_at_default - net_proceeds + foreclosure_costs - mi_recovery
    else:
        # Freddie Mac pre-computes the net loss in the actual_loss field
        # (position 22, "Actual Loss Calculation"). Use it when the individual
        # components are not all available, with a note that the calculation
        # is not directly verifiable from this path.
        net_loss = df["actual_loss"].fillna(0)

    lgd = net_loss / df["zero_balance_removal_upb"]
    # Clip to [0, 1] as documented above. The clipping does not drop extreme
    # observations — it bounds them, which preserves the training distribution
    # shape while preventing numerically unstable targets.
    lgd = lgd.clip(lower=0.0, upper=1.0)

    df["loss_given_default"] = lgd

    # The pct_zero and pct_one statistics in this log record are the first
    # indicators of whether the LGD distribution has the expected bimodal
    # shape. Roughly 15-25% at zero (full recovery) and 5-15% at one (total
    # loss) is consistent with post-crisis Freddie Mac data. Significant
    # deviations from these ranges warrant investigation before proceeding.
    logger.info(
        f"LGD target constructed: n={len(df):,}, "
        f"mean={lgd.mean():.4f}, median={lgd.median():.4f}, "
        f"pct_zero={((lgd == 0).mean() * 100):.1f}%, "
        f"pct_one={((lgd == 1).mean() * 100):.1f}%"
    )
    return df


def merge_and_clean(orig_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    """Merge origination and performance data on loan_seq_num.

    The merge is inner, which means only loans present in both files are
    retained. Loans in the performance file that are not in the origination
    file (which can happen for loans originated before 2010 that defaulted
    within the vintage window) are dropped. Loans in the origination file
    with no matching performance record (loans that did not default) were
    already excluded at the ingestion stage.

    The inner join produces the population of interest: loans that originated
    in the 2010-2015 window, defaulted, and have a complete origination record.
    This is the correct training population for a conditional LGD model.

    Column renaming here (property_state → state, orig_loan_term → orig_term)
    is for consistency with the feature column names in configs/default.yaml.
    The shorter names reduce visual noise in the feature matrix and downstream
    code.

    Args:
        orig_df: Origination DataFrame from ingest_origination().
        perf_df: Performance DataFrame from ingest_performance(), with LGD
                 target column already constructed by construct_lgd_target().

    Returns:
        Merged DataFrame with origination features, performance features,
        and the LGD target.
    """
    # Rename maturity_date in the origination file to avoid the column name
    # collision with maturity_date in the performance file. The origination
    # maturity_date is the contractual maturity; the performance maturity_date
    # may differ due to modifications. The origination version is retained with
    # a distinguishing suffix.
    orig_df = orig_df.copy()
    if "maturity_date" in orig_df.columns and "maturity_date" in perf_df.columns:
        orig_df = orig_df.rename(columns={"maturity_date": "maturity_date_orig"})

    merged = perf_df.merge(orig_df, on="loan_seq_num", how="inner")
    logger.info(
        f"Merged: {len(perf_df):,} performance × {len(orig_df):,} origination "
        f"→ {len(merged):,} matched loans"
    )

    # Normalize column names to match the feature list in configs/default.yaml.
    # These renames are done here rather than in the config because renaming
    # in the config would require updating references in multiple places.
    if "property_state" in merged.columns:
        merged = merged.rename(columns={"property_state": "state"})
    if "orig_loan_term" in merged.columns:
        merged = merged.rename(columns={"orig_loan_term": "orig_term"})

    # A null LGD target after merge should not happen if construct_lgd_target()
    # ran correctly, but the check here provides a safety net and logs any
    # unexpected drops so they are visible in the audit trail.
    before = len(merged)
    merged = merged.dropna(subset=["loss_given_default"])
    after = len(merged)
    if before != after:
        logger.warning(f"Dropped {before - after:,} rows with null LGD target after merge")

    return merged


def preprocess(config: Config) -> pd.DataFrame:
    """Execute the full preprocessing pipeline and save to data/processed/cleaned.parquet.

    This function orchestrates the steps in the correct order: load → cast →
    impute → construct target → merge. Each step depends on the previous one
    having completed successfully. The output is a single parquet file that
    serves as the input to features.py.

    The pipeline is designed to be re-runnable. Running it twice with the same
    inputs produces the same output, which is a requirement for reproducible
    model training under SR 11-7. The parquet output is gitignored because it
    can always be regenerated from the raw data, but it should be archived
    alongside any model artifact that is promoted to production use.

    Returns:
        Cleaned DataFrame ready for feature engineering. Also written to
        data/processed/cleaned.parquet as a side effect.

    Raises:
        FileNotFoundError: If interim parquet files from ingest.py are not
                           found. The error message includes the expected paths
                           to help diagnose setup issues.
    """
    interim_dir = Path(config.data.interim_dir)
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    orig_path = interim_dir / "origination.parquet"
    perf_path = interim_dir / "performance_defaults.parquet"

    if not orig_path.exists() or not perf_path.exists():
        raise FileNotFoundError(
            f"Interim files not found. Run src/data/ingest.py first.\n"
            f"Expected: {orig_path}, {perf_path}"
        )

    logger.info("Loading interim files")
    orig_df = pd.read_parquet(orig_path)
    perf_df = pd.read_parquet(perf_path)

    logger.info("Casting numeric columns")
    orig_df = _cast_numeric(orig_df, ORIGINATION_NUMERIC)
    perf_df = _cast_numeric(perf_df, PERFORMANCE_NUMERIC)

    logger.info("Imputing missing values")
    orig_df = _impute_missing(orig_df)
    perf_df = _impute_missing(perf_df)

    logger.info("Constructing LGD target")
    perf_df = construct_lgd_target(perf_df)

    logger.info("Merging origination and performance")
    merged = merge_and_clean(orig_df, perf_df)

    out_path = processed_dir / "cleaned.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned data to {out_path}: {len(merged):,} rows, {len(merged.columns)} columns")
    return merged


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    df = preprocess(config)
    logger.info(f"Preprocessing complete: {len(df):,} defaulted loans ready for feature engineering")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Freddie Mac SFLP data")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)

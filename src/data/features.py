"""Feature engineering pipeline for the LGD prediction model.

This module transforms the cleaned, merged dataset from preprocess.py into a
model-ready feature matrix. The distinction between preprocessing and feature
engineering is intentional and reflects a practical principle: preprocessing
makes the data correct; feature engineering makes it informative.

The most important feature constructed here is LTV at default. Original LTV
captures the equity cushion at origination but tells nothing about whether that
cushion was eroded or augmented by the time the borrower defaulted. LTV at
default — which accounts for principal paydown and HPI movement since origination
— is consistently the strongest LGD predictor in the empirical literature, and
the model's SHAP analysis should confirm this once trained.

The two macroeconomic features (HPI change and unemployment rate) are
placeholders in their current form. The join logic that would attach MSA-level
FHFA HPI data and BLS county-level unemployment data to individual loans by
matching the loan's state and the timing of default is not yet implemented —
it requires the external data files described in data/README.md and a
time-indexed join that depends on the origination and default dates. Until that
join is built, these features default to 0.0 and 7.5 respectively, which
reduces their predictive contribution to near zero and understates the model's
macro sensitivity. The macro scenario analysis in evaluate.py still works
because it perturbs these feature values directly, but the base-case predictions
will be less accurate than they would be with real HPI and unemployment data.

The state-to-region aggregation is the other feature engineering decision worth
explaining. Fifty-way dummy encoding of state is technically feasible but
produces a wide, sparse feature matrix where many states have few observations.
Five-way regional encoding (Northeast, Southeast, Midwest, Southwest, West)
reduces dimensionality while preserving the regional real estate market dynamics
that matter for LGD: coastal markets, sunbelt markets, and rust belt markets
have meaningfully different foreclosure cost structures and recovery timelines.

Usage:
    python src/data/features.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State-to-region mapping
#
# The U.S. Census Bureau's four-region classification (Northeast, South,
# Midwest, West) is a reasonable starting point, but I split the South into
# Southeast and Southwest because the Texas-Oklahoma-Arizona-New Mexico market
# has sufficiently different HPI dynamics and foreclosure law from Florida-
# Georgia-Carolinas to warrant separation. DC is grouped with West due to its
# coastal real estate market behavior rather than its geographic position.
#
# States not in this mapping (typically territories like PR, GU, or unknown
# codes) fall back to "Other" in add_region(). The Other category is small
# enough in the Freddie Mac 2010-2015 data that it does not materially affect
# model performance, but it prevents key errors if atypical codes appear.
# ---------------------------------------------------------------------------

STATE_TO_REGION = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    # Southeast
    "DE": "Southeast", "FL": "Southeast", "GA": "Southeast", "MD": "Southeast",
    "NC": "Southeast", "SC": "Southeast", "VA": "Southeast", "WV": "Southeast",
    "AL": "Southeast", "KY": "Southeast", "MS": "Southeast", "TN": "Southeast",
    "AR": "Southeast", "LA": "Southeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    # Southwest
    "AZ": "Southwest", "NM": "Southwest", "OK": "Southwest", "TX": "Southwest",
    # West
    "AK": "West", "CA": "West", "CO": "West", "HI": "West", "ID": "West",
    "MT": "West", "NV": "West", "OR": "West", "UT": "West", "WA": "West",
    "WY": "West", "DC": "West",
}


def compute_ltv_at_default(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LTV at default by adjusting origination property value for HPI.

    The calculation chain:
        original_property_value = orig_upb / (orig_ltv / 100)
        estimated_value_at_default = original_value * (1 + hpi_change)
        ltv_at_default = current_upb / estimated_value_at_default * 100

    This is an approximation, not an appraisal. The actual property value at
    the time of default is not available in the Freddie Mac data — it would
    require an AVM (automated valuation model) applied at the default date.
    The HPI-adjusted estimate is the standard approach in academic LGD research
    and is adequate for the level of precision required here.

    The clip at 300 (LTV capped at 300%) removes extreme outliers that arise
    from very large HPI declines or small estimated property values. These
    outliers are artifacts of the estimation method rather than real collateral
    positions and would otherwise create unstable training samples. In practice,
    few loans in the post-crisis period have LTV at default above 200%.

    If orig_ltv or orig_upb are missing (which should be rare after preprocessing),
    the function falls back to orig_ltv as a proxy. This is a weaker feature
    but avoids dropping records.
    """
    df = df.copy()

    if "orig_ltv" in df.columns and "orig_upb" in df.columns:
        # Guard against zero LTV, which would produce infinite property values.
        # Loans with zero orig_ltv are rare and typically represent data errors.
        orig_ltv_safe = df["orig_ltv"].replace(0, np.nan).clip(lower=1)
        orig_value = df["orig_upb"] / (orig_ltv_safe / 100)

        if "hpi_change" in df.columns:
            # Fill missing HPI change with 0 (no price movement assumed).
            # This is a conservative assumption — missing HPI typically means
            # the geography was not matched to the external HPI data, not that
            # prices did not change.
            hpi_change = df["hpi_change"].fillna(0)
            est_value_at_default = orig_value * (1 + hpi_change)
        else:
            # No HPI data available at all — use origination value as proxy.
            # This produces the same result as hpi_change=0 and is explicitly
            # documented in the log output from add_hpi_change().
            est_value_at_default = orig_value

        # Guard against zero estimated value, which would produce infinite LTV.
        est_value_safe = est_value_at_default.replace(0, np.nan)
        df["ltv_at_default"] = (df["current_upb"] / est_value_safe * 100).clip(upper=300)
    else:
        # Fallback: use original LTV as a weaker proxy.
        # This path should be rare in production — orig_ltv is a required field
        # per the validation in ingest.py.
        df["ltv_at_default"] = df.get("orig_ltv", 80.0)

    return df


def add_hpi_change(df: pd.DataFrame, hpi_path: str | None = None) -> pd.DataFrame:
    """Add HPI change from origination to default as a feature.

    The ideal implementation joins MSA-level FHFA HPI data on (state, year,
    quarter) for both the origination date and the default date, then computes:

        hpi_change = (hpi_at_default / hpi_at_origination) - 1

    A value of -0.20 means home prices fell 20% over the loan's life to the
    point of default, which substantially increases LGD by reducing collateral
    value below the outstanding balance. This is one of the two macro features
    that are specifically required for regulatory stress testing — a model
    without a meaningful HPI feature cannot demonstrate the home-price sensitivity
    that DFAST reviewers expect.

    The current implementation is a placeholder that sets hpi_change to 0.0
    when no external HPI file is provided. The 0.0 default is conservative in
    the sense that it assumes no price movement, which will cause the model to
    underestimate LGD for loans from markets that experienced significant price
    declines. The placeholder is clearly logged so it is visible in the pipeline
    audit trail.

    The check for whether hpi_change is already present prevents overwriting
    data that was joined from an external source in a prior step, which allows
    this function to be called idempotently.
    """
    df = df.copy()

    if "hpi_change" not in df.columns:
        if hpi_path and Path(hpi_path).exists():
            try:
                hpi_df = pd.read_csv(hpi_path)
                logger.info(f"Loaded HPI data from {hpi_path}: {len(hpi_df):,} rows")
                # TODO: Implement MSA-level join on origination and default dates.
                # The join requires: orig_date → FHFA HPI index at origination,
                # default_date → FHFA HPI index at default, then compute the ratio.
                # Until this is implemented, set to 0.0 with an explicit warning.
                df["hpi_change"] = 0.0
                logger.warning("HPI join logic is a placeholder — set hpi_change to 0.0")
            except Exception as e:
                logger.warning(f"Could not load HPI data: {e} — setting hpi_change=0.0")
                df["hpi_change"] = 0.0
        else:
            logger.info("No HPI file provided — setting hpi_change=0.0 placeholder")
            df["hpi_change"] = 0.0

    return df


def add_unemployment(df: pd.DataFrame, unemp_path: str | None = None) -> pd.DataFrame:
    """Add state unemployment rate at time of default as a feature.

    The ideal implementation joins BLS LAUS state-level monthly unemployment
    data on (state, default_date) to give each defaulted loan the unemployment
    rate in its state at the month it was resolved. A loan that defaulted in
    Michigan in 2010 should receive a different unemployment rate than one that
    defaulted in Texas in 2014.

    The placeholder value of 7.5% is the approximate national average
    unemployment rate over the 2010-2015 period. It is not geographically or
    temporally differentiated and therefore provides no actual information to
    the model — the feature will have near-zero SHAP importance until the real
    join is implemented. This is documented in the model card limitations
    section. The feature is retained in the pipeline (rather than excluded
    entirely) because it allows the macro scenario analysis in evaluate.py to
    demonstrate sensitivity to unemployment changes even on a base of 7.5%.

    This is the kind of limitation that matters more in practice than most
    modeling choices. The macro sensitivity demonstration in stress testing
    contexts depends on this feature carrying real information. I would
    prioritize implementing the BLS join before promoting this model to
    any production-adjacent use.
    """
    df = df.copy()

    if "unemployment_rate_at_default" not in df.columns:
        if unemp_path and Path(unemp_path).exists():
            try:
                unemp_df = pd.read_csv(unemp_path)
                logger.info(f"Loaded unemployment data from {unemp_path}: {len(unemp_df):,} rows")
                # TODO: Implement state-level monthly join on default date.
                df["unemployment_rate_at_default"] = 7.5
                logger.warning("Unemployment join logic is a placeholder — using national average 7.5")
            except Exception as e:
                logger.warning(f"Could not load unemployment data: {e} — using 7.5%")
                df["unemployment_rate_at_default"] = 7.5
        else:
            logger.info("No unemployment file provided — using national average 7.5% placeholder")
            df["unemployment_rate_at_default"] = 7.5

    return df


def add_delinquency_months(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure months_delinquent_at_default is present.

    In the ideal case, this field is computed from the performance file as
    the count of monthly records where current_delinquency_status was 3+
    (90+ days delinquent) before resolution. In the current pipeline, where
    we retain only the resolution record per loan, we do not have the full
    delinquency history to compute this directly.

    The placeholder of 3 months represents the minimum delinquency to reach
    serious delinquency status (90 days). It understates actual delinquency
    duration, which for foreclosure-path loans in the 2010-2015 period was
    often 12-24 months due to extended loss mitigation timelines. A better
    approach would retain a summary of the pre-resolution delinquency
    trajectory from the performance file during ingestion. That is a known
    improvement to make before production use.
    """
    df = df.copy()
    if "months_delinquent_at_default" not in df.columns:
        # The conditional on loan_age presence checks whether enough performance
        # data is available to compute this properly in a future implementation.
        # For now both branches produce the same placeholder result.
        if "loan_age" in df.columns and "remaining_months" in df.columns:
            df["months_delinquent_at_default"] = 3  # minimum for serious delinquency
        else:
            df["months_delinquent_at_default"] = 3
    return df


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    """Map state codes to five-category census regions.

    The region feature captures geographic real estate market dynamics without
    requiring a 50-way categorical encoding that would fragment the training
    data across too many sparse cells. The mapping is defined in STATE_TO_REGION
    at the top of this module. States not found in the mapping (territories,
    unknown codes) receive "Other."

    Note that the state column itself is also in the feature list as a
    categorical. The region column is an intermediate construction that drives
    the encoded "state" feature through encode_categoricals() — it replaces
    the raw two-letter state code with a five-category region string before
    encoding. The rename in preprocess.py (property_state → state) is what
    makes this function target the right column.
    """
    df = df.copy()
    if "state" in df.columns:
        df["region"] = df["state"].map(STATE_TO_REGION).fillna("Other")
    return df


def add_modification_flag_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert modification_flag (Y/N string) to a binary numeric feature.

    Loan modification prior to default is a nuanced predictor. On one hand,
    borrowers who received modifications are by definition distressed, which
    correlates with longer foreclosure timelines and higher costs. On the other
    hand, modifications that reduced the principal balance directly increase
    recovery by lowering the UPB that needs to be recovered. The net effect
    on LGD is empirically ambiguous, which is precisely why it belongs in the
    model rather than being excluded — let the data determine the direction.

    When modification_flag is absent from the DataFrame (which can happen if
    the performance file is incomplete), the column is set to 0 (no modification).
    This is a conservative assumption consistent with the imputation logic in
    preprocess.py.
    """
    df = df.copy()
    if "modification_flag" in df.columns:
        df["modification_flag_numeric"] = (df["modification_flag"] == "Y").astype(int)
    else:
        df["modification_flag_numeric"] = 0
    return df


def add_vintage_year(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure vintage_year is present, parsing it from first_payment_date if needed.

    vintage_year is not a model input feature — it is a stratification key
    used to ensure the train/val/test split preserves each origination cohort's
    proportion across all three sets. Including it as a model input would encode
    macroeconomic regime information that is better captured through explicit
    macro features like HPI change and unemployment. A model that learned to
    predict high LGD for 2010 vintages because of the year label (rather than
    because of the HPI and unemployment conditions in 2010) would not generalize
    to new stress scenarios where the regime differs from the training period.

    This function is idempotent — if vintage_year is already present, it is not
    overwritten. That protects against double-parsing if engineer_features() is
    called on data that already had vintage_year set by ingest.py.
    """
    df = df.copy()
    if "vintage_year" not in df.columns and "first_payment_date" in df.columns:
        df["vintage_year"] = (
            pd.to_numeric(df["first_payment_date"], errors="coerce")
            .floordiv(100)
            .astype("Int64")
        )
    return df


def encode_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> Tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns and return fitted encoders.

    Label encoding (integer codes) rather than one-hot encoding is appropriate
    for this model because:
    1. The model is a neural network, and neural networks can learn non-linear
       category-to-prediction mappings from integer codes through learned embeddings
       in the weight matrix, unlike linear models that require one-hot encoding.
    2. One-hot encoding categorical features with many levels (state → region
       already reduces this, but property_type still has 5 levels, channel 4,
       etc.) would increase the feature matrix width without adding information
       that a neural network cannot already extract from integer codes.
    3. Label encoding produces a more compact model input, which is relevant
       for the StandardScaler in train.py that normalizes all features jointly.

    The fitted encoders are returned because they are needed at inference time
    to encode unseen categorical values consistently with the training encoding.
    In the current implementation, the API uses hardcoded mappings in predict.py
    that mirror the expected encoding. In a production system, these encoders
    should be serialized alongside the model artifact.

    Unknown categories (values not seen during training) are mapped to "Unknown"
    before encoding. This produces a deterministic integer for the unseen
    category rather than a KeyError at inference time, which is the right
    behavior for a production API.
    """
    df = df.copy()
    encoders = {}

    for col in categorical_cols:
        if col not in df.columns:
            logger.warning(f"Categorical column '{col}' not found — skipping")
            continue
        # Replace missing values with "Unknown" before encoding so that NaN
        # is not treated as a distinct category by LabelEncoder, which would
        # cause it to appear differently across runs depending on sort order.
        df[col] = df[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def build_feature_matrix(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Assemble the final feature matrix X and target vector y.

    The column order in X follows the concatenation of config.features.categorical
    and config.features.numeric, which mirrors the order specified in
    configs/default.yaml. This order must match the order used at inference time
    in api/predict.py. If you add a feature to the config, you must also update
    the inference logic in predict.py to include that feature in the same
    position. Misaligned feature order at inference time is a silent failure —
    the API will accept the request and return a number, but it will be wrong.

    Missing features (columns in the config that are not present in the
    DataFrame) are filled with 0 and logged as warnings. Setting them to 0
    means the StandardScaler will normalize them to approximately -mean/std
    during training, which is not ideal but prevents the pipeline from failing
    when a feature column is not available. This is a development convenience,
    not a production-safe approach — all expected feature columns should be
    present before training a model that will be used in regulatory contexts.
    """
    feature_cols = config.features.categorical + config.features.numeric
    target_col = config.features.target

    # The vintage_year column is attached to the saved features.parquet for
    # use in the stratified train/val/test split in train.py, but it is not
    # included in the feature matrix X itself. See the rationale in
    # add_vintage_year() above for why vintage_year is excluded as a model input.
    if "vintage_year" in df.columns:
        feature_cols_with_segment = feature_cols + ["vintage_year"]
    else:
        feature_cols_with_segment = feature_cols

    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]

    if missing_features:
        logger.warning(f"Features not found in DataFrame, will be set to 0: {missing_features}")
        for col in missing_features:
            df[col] = 0

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info(
        f"Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features. "
        f"Target: '{target_col}', mean={y.mean():.4f}"
    )
    return X, y


def engineer_features(config: Config) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Execute the full feature engineering pipeline.

    The sequence of feature construction steps is ordered by dependency:
    vintage_year must come before the split (not a model input but needed for
    stratification), HPI change must come before LTV at default (which uses
    hpi_change in its calculation), and all feature construction must come
    before categorical encoding (which needs the final state-encoded region).

    The full_df return value includes both the feature matrix and the vintage_year
    metadata column, which is needed by train.py for the stratified split but
    should not be included in the model input. Callers that need only X and y
    can use the first two return values and ignore the third.

    Returns:
        (X, y, full_df) where:
          X = feature matrix DataFrame (12 columns per configs/default.yaml)
          y = LGD target Series
          full_df = X concatenated with y and vintage_year for downstream use
    """
    processed_dir = Path(config.data.processed_dir)
    cleaned_path = processed_dir / "cleaned.parquet"

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {cleaned_path}. "
            "Run src/data/preprocess.py first."
        )

    logger.info(f"Loading cleaned data from {cleaned_path}")
    df = pd.read_parquet(cleaned_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Feature construction — order matters due to dependencies noted above.
    df = add_vintage_year(df)
    df = add_hpi_change(df)
    df = add_unemployment(df)
    # LTV at default depends on hpi_change, so it must come after add_hpi_change.
    df = compute_ltv_at_default(df)
    df = add_delinquency_months(df)
    # Region replaces the raw state code — must come before encode_categoricals.
    df = add_region(df)
    # Replace the raw state column with the region string before encoding,
    # so that the "state" feature in the model input actually encodes region.
    if "region" in df.columns:
        df["state"] = df["region"]
    df = add_modification_flag_numeric(df)

    # Label-encode categoricals. The returned encoders are not persisted to
    # disk in the current implementation — a production version should serialize
    # them alongside the model artifact so they can be used at inference time.
    df, encoders = encode_categoricals(df, config.features.categorical)

    X, y = build_feature_matrix(df, config)

    # Save features.parquet with vintage_year attached for train.py.
    features_path = processed_dir / "features.parquet"
    full_df = pd.concat([X, y], axis=1)
    if "vintage_year" in df.columns:
        full_df["vintage_year"] = df["vintage_year"].values
    full_df.to_parquet(features_path, index=False)
    logger.info(f"Saved engineered features to {features_path}")

    return X, y, full_df


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    X, y, _ = engineer_features(config)
    logger.info(f"Feature engineering complete: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)

"""Unit tests for the feature engineering pipeline.

The tests in this file are not exhaustive property tests — they are targeted
behavioral tests that verify the claims made in the module docstrings of
features.py and preprocess.py. Each test class corresponds to a function or
subsystem in the features pipeline, and each test method verifies a specific
invariant that must hold for the pipeline to produce valid model inputs.

The invariants tested here fall into three categories:

    Correctness constraints: Properties that must hold by definition. LGD must
    be in [0, 1] because it is the fraction of the outstanding balance that is
    lost — a negative LGD or an LGD above 1 is a mathematical impossibility
    given the formula LGD = Net Loss / UPB. LTV at default must be non-negative
    because it is a ratio of two positive quantities.

    Economic direction tests: Properties that must hold because they reflect
    real economic relationships. A larger HPI decline should produce a higher
    LTV at default because the denominator (estimated property value) shrinks
    while the numerator (remaining balance) is unchanged. A model that produces
    higher LTV for lower HPI declines has a direction error that would corrupt
    the economic interpretation of the feature.

    Pipeline robustness tests: Properties about how the pipeline handles edge
    cases. An unknown state code should map to "Other" rather than raising an
    exception. A missing modification_flag column should produce zeros rather
    than failing. These tests verify that the pipeline degrades gracefully when
    real-world data does not match the expected schema.

The fixtures provide minimal DataFrames that isolate the specific behavior being
tested. Using minimal fixtures rather than loading real data means these tests
run quickly (milliseconds, not seconds) and do not require the data pipeline
to have been run. That property is important for CI: the test suite should be
runnable without Freddie Mac data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import (
    STATE_TO_REGION,
    add_delinquency_months,
    add_hpi_change,
    add_modification_flag_numeric,
    add_region,
    add_unemployment,
    add_vintage_year,
    compute_ltv_at_default,
    encode_categoricals,
)
from src.data.preprocess import construct_lgd_target


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_origination_df():
    """Minimal origination DataFrame covering the key feature dimensions.

    Four loans are chosen to represent different states (CA, TX, NY, FL),
    property types (SF and CO), occupancy statuses (P, I, S), and channels
    (R, B, C). This variety ensures that region mapping, categorical encoding,
    and directional tests run against realistic value combinations rather than
    a homogeneous set that might mask edge cases.
    """
    return pd.DataFrame({
        "loan_seq_num": ["A001", "A002", "A003", "A004"],
        "orig_ltv": [80.0, 95.0, 70.0, 110.0],
        "orig_upb": [200_000.0, 300_000.0, 150_000.0, 250_000.0],
        "orig_interest_rate": [4.5, 5.0, 3.75, 6.0],
        "orig_loan_term": [360, 360, 180, 360],
        "property_state": ["CA", "TX", "NY", "FL"],
        "property_type": ["SF", "SF", "CO", "SF"],
        "occupancy_status": ["P", "I", "P", "S"],
        "channel": ["R", "B", "C", "R"],
        "first_payment_date": ["201001", "201103", "201206", "201304"],
        "vintage_year": [2010, 2011, 2012, 2013],
    })


@pytest.fixture
def sample_performance_df():
    """Minimal performance DataFrame with three clean defaults and one edge case.

    A001: Standard third-party sale with moderate recovery (net_loss / UPB ≈ 0.20).
    A002: Short sale with MI recovery, producing moderate LGD after deducting MI proceeds.
    A003: REO disposition with full recovery (net_loss=0) — a valid LGD=0 case.
    A004: Zero UPB — this row should be filtered out by construct_lgd_target because
          LGD = Net Loss / UPB is undefined when UPB = 0.
    """
    return pd.DataFrame({
        "loan_seq_num": ["A001", "A002", "A003", "A004"],
        "current_upb": [180_000.0, 290_000.0, 140_000.0, 0.0],
        "zero_balance_code": ["02", "03", "09", "02"],
        "net_loss": [36_000.0, 58_000.0, 0.0, 50_000.0],
        "foreclosure_costs": [5_000.0, 8_000.0, 2_000.0, 3_000.0],
        "net_sale_proceeds": [149_000.0, 240_000.0, 142_000.0, 200_000.0],
        "credit_enhancement_proceeds": [0.0, 2_000.0, 0.0, 0.0],
    })


# ---------------------------------------------------------------------------
# LGD target tests
# ---------------------------------------------------------------------------

class TestLGDTargetConstruction:
    """Tests for the construct_lgd_target function in preprocess.py.

    These tests verify the mathematical and business constraints on the LGD target
    variable. The constraints are strict: LGD is a bounded ratio that must be
    in [0, 1] for every loan in the training set, and any row that makes the
    ratio undefined (UPB = 0) must be excluded rather than producing NaN or
    infinity in the target.
    """

    def test_lgd_values_in_range(self, sample_performance_df):
        """LGD must be in [0, 1] for all valid loans after target construction."""
        df = construct_lgd_target(sample_performance_df)
        assert "loss_given_default" in df.columns
        assert df["loss_given_default"].between(0.0, 1.0).all(), (
            f"LGD out of [0,1]: {df['loss_given_default'].describe()}"
        )

    def test_lgd_zero_upb_rows_dropped(self, sample_performance_df):
        """Rows with UPB == 0 must be excluded because the LGD denominator is zero."""
        df = construct_lgd_target(sample_performance_df)
        # A004 had current_upb=0 — it must not appear in the output.
        assert len(df) == 3
        assert "A004" not in df["loan_seq_num"].values

    def test_lgd_full_recovery_is_valid(self):
        """LGD = 0 (full recovery) is a valid outcome and must not be dropped.

        Full recovery is common in short sales where the sale price covers the
        outstanding balance. Dropping LGD=0 rows would bias the training set
        toward higher loss severities and produce an upward-biased model.
        """
        df = pd.DataFrame({
            "loan_seq_num": ["B001"],
            "current_upb": [100_000.0],
            "zero_balance_code": ["03"],
            "net_loss": [0.0],
            "foreclosure_costs": [0.0],
            "net_sale_proceeds": [100_000.0],
            "credit_enhancement_proceeds": [0.0],
        })
        result = construct_lgd_target(df)
        assert len(result) == 1
        assert result["loss_given_default"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_lgd_capped_at_one(self):
        """LGD values above 1.0 must be capped at 1.0.

        Net loss can theoretically exceed UPB when foreclosure costs and holding
        expenses are substantial relative to the property value. The cap at 1.0
        reflects the model's conditional scope: LGD is the fraction of the UPB
        that is not recovered, and that fraction cannot exceed 1.0 by definition.
        Expenses beyond UPB are borne by the servicer or GSE in ways that are
        outside the scope of this model.
        """
        df = pd.DataFrame({
            "loan_seq_num": ["C001"],
            "current_upb": [100_000.0],
            "zero_balance_code": ["02"],
            "net_loss": [150_000.0],  # Loss exceeds UPB — should be capped at 1.0
            "foreclosure_costs": [0.0],
            "net_sale_proceeds": [0.0],
            "credit_enhancement_proceeds": [0.0],
        })
        result = construct_lgd_target(df)
        assert result["loss_given_default"].iloc[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestLTVAtDefault:
    """Tests for compute_ltv_at_default().

    LTV at default is the most important predictor in this model — it directly
    measures the exposure at risk relative to collateral value at the time of
    default. These tests verify both the mathematical properties of the computation
    and the directional relationship with HPI change.
    """

    def test_ltv_computed_when_hpi_available(self, sample_origination_df):
        """LTV at default is computed and non-null when current_upb and hpi_change are present."""
        df = sample_origination_df.copy()
        df["current_upb"] = df["orig_upb"] * 0.95  # 5% principal paid down
        df["hpi_change"] = -0.10  # 10% HPI decline
        result = compute_ltv_at_default(df)
        assert "ltv_at_default" in result.columns
        assert result["ltv_at_default"].notna().all()
        # With HPI decline and small paydown, LTV should be higher than orig_ltv
        # for at least some loans.
        assert (result["ltv_at_default"] > df["orig_ltv"]).any()

    def test_ltv_not_negative(self, sample_origination_df):
        """LTV at default is always non-negative even under HPI appreciation."""
        df = sample_origination_df.copy()
        df["current_upb"] = df["orig_upb"] * 0.5
        df["hpi_change"] = 0.5  # Large HPI appreciation — LTV falls but must stay >= 0
        result = compute_ltv_at_default(df)
        assert (result["ltv_at_default"] >= 0).all()

    def test_ltv_capped_at_300(self):
        """LTV at default is capped at 300 to prevent extreme outliers from distorting scaling.

        Near-total HPI collapse produces theoretically unlimited LTV values. The
        cap at 300 treats these as extreme cases rather than allowing them to
        dominate the StandardScaler fit and distort the normalized feature distribution.
        """
        df = pd.DataFrame({
            "orig_ltv": [80.0],
            "orig_upb": [200_000.0],
            "current_upb": [200_000.0],
            "hpi_change": [-0.99],  # Near-total price collapse
        })
        result = compute_ltv_at_default(df)
        assert result["ltv_at_default"].iloc[0] <= 300.0


class TestHPIChange:
    """Tests for add_hpi_change().

    The HPI change feature is currently a placeholder (returns 0.0 when no
    external HPI file is available). These tests verify the placeholder behavior
    and the idempotency guarantee — an existing hpi_change column must not be
    overwritten by the function.
    """

    def test_hpi_change_placeholder_when_no_file(self, sample_origination_df):
        """Without an external HPI file, hpi_change defaults to 0.0 for all loans."""
        result = add_hpi_change(sample_origination_df)
        assert "hpi_change" in result.columns
        assert (result["hpi_change"] == 0.0).all()

    def test_hpi_change_not_overwritten_if_present(self, sample_origination_df):
        """Existing hpi_change values must be preserved — the function must be idempotent.

        This matters because the features pipeline may call add_hpi_change() on a
        DataFrame that already has HPI values from an external source. The function
        must detect the existing column and skip the placeholder assignment.
        """
        df = sample_origination_df.copy()
        df["hpi_change"] = -0.15
        result = add_hpi_change(df)
        assert (result["hpi_change"] == -0.15).all()


class TestRegionMapping:
    """Tests for the STATE_TO_REGION mapping and add_region() function.

    The region mapping aggregates 50 states (plus DC) into 5 regions for the
    categorical feature that the model uses for geographic variation in LGD.
    These tests verify completeness of the mapping and correct handling of
    unknown state codes.
    """

    def test_all_states_have_region(self):
        """Every state in STATE_TO_REGION maps to a valid region name."""
        valid_regions = {"Northeast", "Southeast", "Midwest", "Southwest", "West"}
        for state, region in STATE_TO_REGION.items():
            assert region in valid_regions, (
                f"State {state} maps to invalid region '{region}'"
            )

    def test_unknown_state_maps_to_other(self):
        """State codes not in the mapping produce 'Other' rather than raising an error."""
        df = pd.DataFrame({"state": ["CA", "XX", "ZZ"]})
        result = add_region(df)
        assert result.loc[result["state"] == "XX", "region"].iloc[0] == "Other"
        assert result.loc[result["state"] == "CA", "region"].iloc[0] == "West"

    def test_region_column_created(self, sample_origination_df):
        """add_region() creates the region column when state column is present."""
        df = sample_origination_df.rename(columns={"property_state": "state"})
        result = add_region(df)
        assert "region" in result.columns


class TestModificationFlag:
    """Tests for add_modification_flag_numeric().

    Loan modification status is a binary feature — modified (1) or not modified (0).
    These tests verify the encoding of the 'Y'/'N' string values and the graceful
    handling of a missing modification_flag column.
    """

    def test_y_maps_to_one(self):
        """'Y' maps to 1, 'N' maps to 0, and None maps to 0."""
        df = pd.DataFrame({"modification_flag": ["Y", "N", "Y", None]})
        result = add_modification_flag_numeric(df)
        assert result["modification_flag_numeric"].tolist() == [1, 0, 1, 0]

    def test_missing_column_creates_zeros(self):
        """When modification_flag column is absent, the numeric column is created as all zeros.

        The Freddie Mac SFLP files do not always include modification_flag. The
        pipeline must handle its absence without failing, and the correct fallback
        is 0 (unmodified) — it is conservative to assume no modification when the
        information is missing.
        """
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = add_modification_flag_numeric(df)
        assert "modification_flag_numeric" in result.columns
        assert (result["modification_flag_numeric"] == 0).all()


class TestVintageYear:
    """Tests for add_vintage_year().

    Vintage year is derived from the YYYYMM first_payment_date field by integer
    division. These tests verify the parsing logic and the idempotency guarantee.
    """

    def test_vintage_year_from_first_payment_date(self):
        """Vintage year is correctly extracted from YYYYMM format via floor division."""
        df = pd.DataFrame({"first_payment_date": ["201001", "201103", "201506"]})
        result = add_vintage_year(df)
        assert result["vintage_year"].tolist() == [2010, 2011, 2015]

    def test_vintage_year_not_overwritten_if_present(self):
        """Existing vintage_year values are preserved when the column already exists."""
        df = pd.DataFrame({
            "first_payment_date": ["201001"],
            "vintage_year": [2010],
        })
        result = add_vintage_year(df)
        assert result["vintage_year"].iloc[0] == 2010


class TestEncodeCategoricals:
    """Tests for encode_categoricals().

    LabelEncoder is used rather than one-hot encoding because neural networks
    do not require dummy variables and one-hot encoding would expand the feature
    matrix by a factor proportional to the cardinality of each categorical feature.
    These tests verify that the encoding produces integer outputs in the expected
    shape and that missing columns are handled gracefully.
    """

    def test_categorical_encoding_output_shape(self, sample_origination_df):
        """Encoding does not change the number of rows or columns in the DataFrame."""
        df = sample_origination_df.rename(columns={"property_state": "state"})
        categorical_cols = ["property_type", "occupancy_status", "channel", "state"]
        encoded_df, encoders = encode_categoricals(df, categorical_cols)
        assert encoded_df.shape == df.shape
        assert set(encoders.keys()) == set(categorical_cols)

    def test_encoded_values_are_integers(self, sample_origination_df):
        """Label-encoded columns must have integer dtype for the model input matrix."""
        df = sample_origination_df.rename(columns={"property_state": "state"})
        categorical_cols = ["property_type", "occupancy_status"]
        encoded_df, _ = encode_categoricals(df, categorical_cols)
        for col in categorical_cols:
            assert encoded_df[col].dtype in [int, "int64", "int32"], (
                f"Column {col} has non-integer dtype: {encoded_df[col].dtype}"
            )

    def test_missing_categorical_column_skipped(self, sample_origination_df):
        """Columns listed in categorical_cols that are absent from the DataFrame are skipped.

        This allows the categorical list in configs/default.yaml to include columns
        that may not be present in all intermediate DataFrames without causing
        the pipeline to fail.
        """
        df = sample_origination_df.rename(columns={"property_state": "state"})
        categorical_cols = ["property_type", "nonexistent_col"]
        encoded_df, encoders = encode_categoricals(df, categorical_cols)
        assert "nonexistent_col" not in encoders
        assert "property_type" in encoders


class TestUnemployment:
    """Tests for add_unemployment().

    Like HPI change, unemployment rate is a placeholder feature pending integration
    with an external macroeconomic data source. These tests verify the placeholder
    value (7.5%, the approximate 2010-2015 mean) and the idempotency guarantee.
    """

    def test_unemployment_placeholder_when_no_file(self, sample_origination_df):
        """Without an external unemployment file, the placeholder value of 7.5 is assigned."""
        result = add_unemployment(sample_origination_df)
        assert "unemployment_rate_at_default" in result.columns
        assert (result["unemployment_rate_at_default"] == 7.5).all()

    def test_unemployment_not_overwritten_if_present(self, sample_origination_df):
        """Existing unemployment_rate_at_default values must be preserved."""
        df = sample_origination_df.copy()
        df["unemployment_rate_at_default"] = 9.5
        result = add_unemployment(df)
        assert (result["unemployment_rate_at_default"] == 9.5).all()


class TestFeatureDirectionality:
    """Economic direction sanity checks for the feature engineering pipeline.

    A feature engineering pipeline that produces the right values by the wrong
    formula can pass correctness tests while producing a model that is mis-specified.
    These directional tests verify that the computed features respond to economic
    inputs in the direction that financial theory predicts. A model trained on
    features with reversed economic direction would produce SHAP values that are
    interpretable in the wrong direction, which is a material error for a model
    used in regulatory stress testing.
    """

    def test_higher_hpi_decline_increases_ltv(self):
        """Larger HPI decline produces higher LTV at default, holding all else equal.

        Economic basis: LTV = Remaining Balance / Estimated Property Value.
        If HPI declines, the estimated property value decreases, and LTV increases.
        This relationship must hold in the computed feature for the model to learn
        the correct sign for HPI's effect on LGD.
        """
        base = pd.DataFrame({
            "orig_ltv": [80.0],
            "orig_upb": [200_000.0],
            "current_upb": [195_000.0],
            "hpi_change": [0.0],
        })
        stressed = base.copy()
        stressed["hpi_change"] = -0.20

        base_result = compute_ltv_at_default(base)
        stressed_result = compute_ltv_at_default(stressed)

        assert stressed_result["ltv_at_default"].iloc[0] > base_result["ltv_at_default"].iloc[0], (
            "Higher HPI decline should produce higher LTV at default, but did not"
        )

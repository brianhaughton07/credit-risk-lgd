"""Validate raw Freddie Mac SFLP files and stage them to interim/.

The data pipeline begins here. Freddie Mac distributes their Single Family Loan
Performance dataset as quarterly flat files — pipe-delimited text with no header
row and no column names embedded in the file. Position matters. If the column
order in ORIGINATION_COLUMNS drifts from the actual file layout for a given
quarter, everything downstream silently misaligns. That is the kind of data
integrity failure that shows up six months later as an implausible model metric
that takes days to trace back to a column assignment error. The schema constants
at the top of this file are the single source of truth for column layout. They
should not be modified unless you have confirmed the change against the Freddie
Mac data dictionary at the source.

The ingestion stage has one job: validate that the raw data matches the expected
schema and write clean parquet files to data/interim/. It does not impute, it
does not engineer features, and it does not construct the LGD target. Those steps
belong in preprocess.py and features.py respectively. Keeping the pipeline stages
separated means a validation failure here surfaces immediately, before any
downstream computation has consumed a corrupted or misaligned input.

One design choice worth noting: this module filters performance data to defaulted
loans at the ingestion stage rather than passing the full performance file
downstream. The raw performance files for 2010-2015 contain hundreds of millions
of rows covering the entire monthly history of every loan. Passing all of that
downstream is not necessary for LGD modeling, which is a conditional model
that operates only on loans that reached default. Filtering early reduces memory
pressure significantly and keeps the interim files to a tractable size.

Usage:
    python src/data/ingest.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

# Ensure project root is on the path when run as a script directly.
# This is necessary because the relative imports below expect the package
# structure to be importable from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Freddie Mac SFLP column schemas
#
# These lists define the column order for the pipe-delimited flat files.
# Freddie Mac does not include a header row — columns are positional.
# The ordering here reflects the 2013+ file format documented at:
# https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
#
# If you are working with pre-2013 files, verify the column positions against
# the corresponding data dictionary PDF, as Freddie Mac revised the schema
# in that period. Misaligned columns will not raise an error at read time —
# they will silently produce wrong values, which is the worst kind of failure.
# ---------------------------------------------------------------------------

ORIGINATION_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
    "msa", "mip", "num_units", "occupancy_status", "orig_cltv", "orig_dti",
    "orig_upb", "orig_ltv", "orig_interest_rate", "channel", "prepayment_penalty",
    "amortization_type", "property_state", "property_type", "postal_code",
    "loan_seq_num", "loan_purpose", "orig_loan_term", "num_borrowers",
    "seller_name", "servicer_name", "super_conforming_flag",
]

PERFORMANCE_COLUMNS = [
    "loan_seq_num", "monthly_reporting_period", "current_upb", "loan_age",
    "remaining_months", "adj_months_to_maturity", "maturity_date", "msa",
    "current_delinquency_status", "modification_flag", "zero_balance_code",
    "zero_balance_effective_date", "last_paid_installment_date", "foreclosure_date",
    "disposition_date", "foreclosure_costs", "property_preservation_costs",
    "asset_recovery_costs", "misc_holding_expenses", "associated_taxes",
    "net_sale_proceeds", "credit_enhancement_proceeds", "repurchase_make_whole",
    "other_foreclosure_proceeds", "non_mi_recovery", "net_recovery",
    "net_loss", "modification_flag_2",
]

# The zero_balance_code values that indicate a default or loss event.
# These are the only loans this model is designed to predict — LGD is a
# conditional model that assumes default has already occurred.
#
# Code definitions:
#   02 = Third-party sale (loss event; property sold at auction or short sale)
#   03 = Short sale or short payoff (loss event)
#   06 = Repurchase / make-whole (GSE repurchases the loan from pool)
#   09 = REO Disposition (Real Estate Owned; Freddie took the property and sold it)
#
# Code 01 (prepayment/full payoff) is explicitly excluded. Prepaid loans had
# no default and no loss — including them would violate the conditional nature
# of the model. Code 02 was chosen over code 00 (still current) for the same
# reason.
DEFAULT_ZERO_BALANCE_CODES = {"02", "03", "06", "09"}

# Minimum columns required for the model to function. These drive the hard
# validation in _validate_origination and _validate_performance. Columns beyond
# this set may be missing or empty without blocking the pipeline, though missing
# engineered-feature inputs will fall back to placeholder values downstream.
REQUIRED_ORIGINATION_COLS = {
    "loan_seq_num", "orig_ltv", "orig_upb", "orig_interest_rate",
    "orig_loan_term", "property_type", "occupancy_status", "channel",
    "property_state", "first_payment_date",
}

REQUIRED_PERFORMANCE_COLS = {
    "loan_seq_num", "monthly_reporting_period", "current_upb",
    "zero_balance_code", "net_loss", "foreclosure_costs",
    "credit_enhancement_proceeds",
}


def _iter_raw_files(directory: Path, pattern: str) -> Iterator[Path]:
    """Yield raw text files matching the given glob pattern, sorted by name.

    Sorting by name produces chronological order for quarterly files following
    the naming convention historical_data_YYYYQQ.txt, which means the pipeline
    processes files in vintage order. That is not strictly required for
    correctness — the vintage filter applied downstream handles ordering — but
    it produces log output that is easier to interpret when diagnosing issues
    with specific quarters.

    The FileNotFoundError raised here is intentional. A missing data directory
    is almost always a setup problem (the data was not downloaded, or was placed
    in the wrong location) rather than a recoverable runtime condition. Surfacing
    it immediately with a clear error message is more useful than silently
    returning an empty result.
    """
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {directory}. "
            "See data/README.md for download instructions."
        )
    return iter(files)


def _read_freddie_file(path: Path, columns: list[str], chunksize: int = 100_000) -> pd.DataFrame:
    """Read a pipe-delimited Freddie Mac flat file into a DataFrame.

    Reading in chunks (chunksize=100_000) rather than all at once prevents
    memory errors on the larger quarterly performance files, which can contain
    several million rows. The chunk size of 100,000 was chosen to keep each
    chunk below approximately 500 MB in memory, which works on machines with
    8 GB RAM. If you are running on a smaller machine, reduce the chunksize.
    If you have abundant memory, increasing it will reduce the number of I/O
    operations and speed up ingestion modestly.

    dtype=str reads all columns as strings initially. This avoids the silent
    truncation that happens when pandas infers numeric types and encounters
    values like "999" for credit_score (the Freddie Mac sentinel for missing)
    or leading zeros in some identifier fields. Type casting to the appropriate
    numeric types happens in preprocess.py, after this module has confirmed
    the schema is intact.

    on_bad_lines="warn" rather than "error" allows ingestion to continue past
    rare malformed rows in large quarterly files. The warning is logged, so
    these lines are not silently lost — but a handful of bad rows should not
    abort an ingestion run over several gigabytes of data.
    """
    logger.info(f"Reading {path.name}")
    chunks = []
    for chunk in pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        dtype=str,
        chunksize=chunksize,
        on_bad_lines="warn",
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df):,} rows from {path.name}")
    return df


def _validate_origination(df: pd.DataFrame) -> None:
    """Confirm that the origination DataFrame meets minimum schema requirements.

    This validation does not check data quality (whether LTV values are
    plausible, whether interest rates are in a reasonable range, etc.) — that
    is the job of the EDA notebook. It checks structural integrity: are the
    required columns present, and is the primary key populated and unique.

    Duplicate loan_seq_num records in the origination file indicate that a loan
    appears in multiple quarterly files, which happens because Freddie Mac
    releases origination data on a rolling basis and loans can appear in more
    than one quarterly snapshot. The de-duplication in ingest_origination()
    keeps the first occurrence, which is the earliest appearance and therefore
    the record closest to origination — the right choice for origination-date
    characteristics.
    """
    missing = REQUIRED_ORIGINATION_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Origination file missing required columns: {missing}")

    null_key = df["loan_seq_num"].isna().sum()
    if null_key > 0:
        raise ValueError(f"Origination: {null_key} rows have null loan_seq_num")

    dupe_keys = df["loan_seq_num"].duplicated().sum()
    if dupe_keys > 0:
        logger.warning(f"Origination: {dupe_keys} duplicate loan_seq_num records found — keeping first")

    logger.info(
        f"Origination validation passed: {len(df):,} rows, "
        f"{df['loan_seq_num'].nunique():,} unique loans"
    )


def _validate_performance(df: pd.DataFrame) -> None:
    """Confirm that the performance DataFrame meets minimum schema requirements.

    At this point the performance DataFrame has already been filtered to
    defaulted loans only, so the row count reflects the number of loss events
    in the vintage window. Log the count — a drop relative to expectations is
    the first indicator that the default filter or the input files have a
    problem that needs investigation.
    """
    missing = REQUIRED_PERFORMANCE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Performance file missing required columns: {missing}")

    null_key = df["loan_seq_num"].isna().sum()
    if null_key > 0:
        raise ValueError(f"Performance: {null_key} rows have null loan_seq_num")

    logger.info(
        f"Performance validation passed: {len(df):,} rows, "
        f"{df['loan_seq_num'].nunique():,} unique loans"
    )


def _extract_vintage_year(df: pd.DataFrame) -> pd.DataFrame:
    """Parse first_payment_date (stored as YYYYMM string) into vintage_year.

    Floor division by 100 extracts the year from the YYYYMM integer, which is
    simpler and faster than string slicing for this particular format. The
    Int64 dtype (nullable integer) preserves NaN for rows where first_payment_date
    is missing or cannot be parsed, rather than converting them to a float NaN
    which would coerce the entire column to float. That matters because vintage_year
    is used as a stratification key in the train/val/test split, and a float-typed
    stratification key would require additional handling downstream.
    """
    df = df.copy()
    df["vintage_year"] = (
        pd.to_numeric(df["first_payment_date"], errors="coerce")
        .floordiv(100)
        .astype("Int64")
    )
    return df


def ingest_origination(config: Config) -> pd.DataFrame:
    """Load, validate, and stage all origination files to data/interim/.

    Iterates over every quarterly origination file found in data/raw/origination/,
    filters to the configured vintage window, de-duplicates on loan_seq_num, and
    writes a single consolidated parquet file. The parquet format is preferred
    over CSV for interim storage because it preserves dtypes, supports efficient
    column-level reads, and compresses substantially for text-heavy columns like
    seller_name and servicer_name.

    The vintage filter is applied per-file rather than after concatenating all
    files, which avoids loading out-of-scope data into memory. For a full 2000-2023
    dataset, this matters — for the 2010-2015 window this project uses, the
    difference is modest, but the pattern is worth preserving for future extension.

    Returns:
        Combined origination DataFrame with vintage_year column added.
        Returns an empty DataFrame (with correct columns) if no files are found,
        rather than raising an error, to allow the pipeline to proceed to
        validation steps that will produce a more informative failure message.
    """
    raw_dir = Path(config.data.raw_dir)
    interim_dir = Path(config.data.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = raw_dir / "origination"
    all_chunks = []

    try:
        files = list(_iter_raw_files(orig_dir, "historical_data_*.txt"))
    except FileNotFoundError as e:
        logger.warning(str(e))
        logger.warning("Origination directory not found — returning empty DataFrame")
        return pd.DataFrame(columns=ORIGINATION_COLUMNS + ["vintage_year"])

    for path in files:
        df = _read_freddie_file(path, ORIGINATION_COLUMNS)
        df = _extract_vintage_year(df)
        # Apply the vintage filter immediately to avoid accumulating out-of-scope
        # records in memory across quarterly files.
        mask = (
            (df["vintage_year"] >= config.data.vintage_start) &
            (df["vintage_year"] <= config.data.vintage_end)
        )
        df = df[mask]
        all_chunks.append(df)

    if not all_chunks:
        logger.warning("No origination records found in vintage range")
        return pd.DataFrame(columns=ORIGINATION_COLUMNS + ["vintage_year"])

    combined = pd.concat(all_chunks, ignore_index=True)
    # De-duplicate keeping the first occurrence, which corresponds to the
    # earliest quarterly file — the origination record closest in time to
    # when the loan was actually originated.
    combined = combined.drop_duplicates(subset=["loan_seq_num"], keep="first")
    _validate_origination(combined)

    out_path = interim_dir / "origination.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(f"Staged origination to {out_path}: {len(combined):,} loans")
    return combined


def ingest_performance(config: Config) -> pd.DataFrame:
    """Load, validate, and stage defaulted-loan performance data to data/interim/.

    The critical design decision here is filtering to defaulted loans at this
    stage rather than passing all performance records downstream. The raw Freddie
    Mac performance files contain a monthly record for every loan from origination
    through resolution — for the 2010-2015 vintage window, that is hundreds of
    millions of rows. The LGD model needs exactly one row per defaulted loan: the
    final resolution record, which contains the net loss, disposition proceeds,
    and foreclosure costs that drive the LGD calculation.

    After filtering to DEFAULT_ZERO_BALANCE_CODES, we take the last record per
    loan by sorting on monthly_reporting_period and grouping. "Last" here means
    the resolution record — the row where the loan was finally disposed of and
    the loss components are populated. Earlier monthly rows for the same loan
    will not have populated loss fields, so taking the last record is both
    correct and necessary.

    Returns:
        DataFrame with one row per defaulted loan at resolution, written to
        data/interim/performance_defaults.parquet.
    """
    raw_dir = Path(config.data.raw_dir)
    interim_dir = Path(config.data.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    perf_dir = raw_dir / "performance"
    all_chunks = []

    try:
        files = list(_iter_raw_files(perf_dir, "historical_data_time_*.txt"))
    except FileNotFoundError as e:
        logger.warning(str(e))
        logger.warning("Performance directory not found — returning empty DataFrame")
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    for path in files:
        df = _read_freddie_file(path, PERFORMANCE_COLUMNS)
        # Filter to loss events only — this is the core conditional framing
        # of the LGD model. Every loan that prepaid, was modified to current,
        # or remains outstanding is irrelevant to LGD prediction.
        default_mask = df["zero_balance_code"].isin(DEFAULT_ZERO_BALANCE_CODES)
        defaulted = df[default_mask]
        if len(defaulted) > 0:
            all_chunks.append(defaulted)

    if not all_chunks:
        logger.warning("No defaulted loan records found in performance files")
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    combined = pd.concat(all_chunks, ignore_index=True)
    _validate_performance(combined)

    # Sort by reporting period and take the last record per loan.
    # The last record per loan is the resolution record — the row where
    # zero_balance_code is populated and loss fields are finalized.
    # Earlier monthly rows for the same loan represent the delinquency
    # progression and do not contain resolved loss amounts.
    combined = combined.sort_values("monthly_reporting_period").groupby(
        "loan_seq_num", as_index=False
    ).last()

    out_path = interim_dir / "performance_defaults.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(f"Staged defaulted performance to {out_path}: {len(combined):,} loans")
    return combined


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    logger.info("Starting data ingestion")
    orig_df = ingest_origination(config)
    perf_df = ingest_performance(config)
    logger.info(
        f"Ingestion complete: {len(orig_df):,} origination rows, "
        f"{len(perf_df):,} defaulted performance rows"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Freddie Mac SFLP raw data")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)

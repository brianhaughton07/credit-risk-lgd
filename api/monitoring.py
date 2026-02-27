"""Prediction logging and distribution drift monitoring for the LGD API.

Every prediction served by this API is logged to a SQLite database. That is not
instrumentation added after the fact — it is a design requirement for a model
operating in a regulatory context. SR 11-7 guidance on model risk management
requires ongoing monitoring of model performance and input distributions, and that
monitoring is only possible if predictions are recorded. A model that cannot answer
the question "what has the predicted LGD distribution looked like over the past 30
days?" is not meeting the documentation standard expected of models used for CECL
reserve estimation.

The SQLite storage choice is appropriate for the deployment context here: a single-
process API with moderate prediction volume. SQLite handles thousands of writes per
hour without contention issues in a single-writer environment, and it requires no
infrastructure setup — the database file is created automatically at the db_path
location when PredictionLogger is instantiated. If this API were deployed behind a
load balancer with multiple worker processes writing simultaneously, or if prediction
volume exceeded tens of thousands per hour, the SQLite approach would need to be
replaced with a PostgreSQL or similar database. For the current scope, SQLite provides
the monitoring capability at the lowest operational cost.

The INSERT OR IGNORE pattern on prediction_id (the UUID primary key) makes the
log write idempotent. A prediction with a given prediction_id will be logged at most
once, even if the request is retried. This is the correct behavior for a prediction
log: retried requests should not produce duplicate monitoring entries that would
distort the distribution statistics.

The drift metric in get_summary() is a mean-shift statistic rather than a true
Kolmogorov-Smirnov test. A proper KS test would require the full empirical distribution
of each feature from the training data, which would require storing that distribution
at training time and loading it here. The mean-shift statistic is a pragmatic approximation
that is available with only the training feature means, which are lightweight enough
to hardcode (in main.py) or load from a simple JSON file. The approximation is sufficient
for detecting gross distribution shifts — a 20% shift in mean orig_ltv is a meaningful
signal even if the full distributional comparison would be more precise.

Schema:
    predictions table:
        prediction_id  TEXT PRIMARY KEY  (UUID from uuid.uuid4())
        timestamp      TEXT NOT NULL     (ISO 8601 UTC, e.g. 2024-01-15T14:32:01.123456+00:00)
        features_json  TEXT NOT NULL     (JSON-serialized input feature dict)
        predicted_lgd  REAL NOT NULL     (model output, in [0, 1])
        model_version  TEXT NOT NULL     (e.g. "lgdnet-v1")
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class PredictionLogger:
    """Append-only SQLite logger for LGD prediction monitoring.

    The append-only design is deliberate. Prediction records are not updated
    or deleted after writing — the log is an audit trail, and audit trails are
    immutable by definition. If a prediction was made and logged, that fact persists
    regardless of subsequent model updates or corrections. This is not an engineering
    preference; it is the expectation of the regulatory framework that governs these
    models.

    The _connection() context manager handles the commit/rollback lifecycle for each
    database operation. Each database interaction (a write in log_prediction, a
    series of reads in get_summary) runs within a single connection that is committed
    on success and rolled back on exception. This prevents partial writes — if the
    INSERT succeeds but a subsequent operation fails, the INSERT will be rolled back
    rather than persisted in an inconsistent state.

    Args:
        db_path: Path to the SQLite database file. The parent directory will be
                 created if it does not exist. If the file does not exist, SQLite
                 will create it at first connection. If it does exist, the schema
                 will be created if the predictions table is not present (CREATE
                 TABLE IF NOT EXISTS is idempotent).
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id  TEXT PRIMARY KEY,
        timestamp      TEXT NOT NULL,
        features_json  TEXT NOT NULL,
        predicted_lgd  REAL NOT NULL,
        model_version  TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager providing a SQLite connection with automatic commit/rollback.

        Each call creates a new connection rather than reusing a persistent one.
        For a low-concurrency API, this is simpler and avoids connection pool
        management. The overhead of creating a SQLite connection is negligible
        compared to the prediction latency budget.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create the predictions table if it does not already exist."""
        with self._connection() as conn:
            conn.execute(self.CREATE_TABLE_SQL)

    def log_prediction(
        self,
        prediction_id: str,
        features: dict,
        predicted_lgd: float,
        model_version: str,
    ) -> None:
        """Append a single prediction record to the log.

        The features dict is serialized as JSON and stored in the features_json
        column. This denormalized approach avoids the need for a separate features
        table with a column for each feature — a schema change every time a feature
        is added — at the cost of requiring JSON parsing in get_summary() when
        computing feature distribution statistics. For the prediction volume this
        API is designed for, the parse cost is acceptable.

        The timestamp is recorded in UTC with timezone information included in the
        ISO 8601 string. Local timestamps are not used because the monitoring
        dashboard may aggregate predictions across time zones, and local timestamps
        would create ambiguity in the period field returned by get_summary().

        INSERT OR IGNORE rather than INSERT handles the idempotency requirement:
        if a request is retried and the first attempt's prediction_id is submitted
        again, the second INSERT will be silently ignored rather than raising a
        UNIQUE constraint violation.

        Args:
            prediction_id: UUID string identifying this prediction.
            features: Input feature dict as received from the LoanFeatures schema.
            predicted_lgd: Model output value, in [0, 1].
            model_version: Model version string for provenance tracking.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        features_json = json.dumps(features)

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO predictions
                    (prediction_id, timestamp, features_json, predicted_lgd, model_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (prediction_id, timestamp, features_json, predicted_lgd, model_version),
            )

    def get_summary(
        self,
        training_feature_means: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute monitoring summary statistics from the prediction log.

        The summary has two components. The first — n_predictions, mean_predicted_lgd,
        and period — characterizes the volume and central tendency of predictions served
        since the last monitoring reset. A shift in mean_predicted_lgd (for example,
        the mean rising from 0.32 to 0.45 over 30 days) is a signal worth investigating:
        it could reflect a genuine change in the portfolio being scored, or it could
        indicate data quality issues in the input features being supplied to the API.

        The second component — feature_means and ks_statistics — characterizes whether
        the input feature distribution of serving requests resembles the training data.
        Features with ks_statistics > 0.10 (10% mean shift) warrant investigation because
        the model was not trained on inputs with those characteristics and may be
        extrapolating in ways that are not reflected in the test set metrics. The
        ks_statistics computation requires training_feature_means, which in the current
        implementation are hardcoded in api/main.py. The correct production approach
        is to save the training feature means as a JSON artifact during training
        and load them here.

        All feature_json records are loaded into memory for the drift computation.
        This is acceptable for a monitoring endpoint that is called infrequently
        (daily or on-demand, not on every prediction request). If the prediction volume
        were high enough that loading all records into memory were problematic, an
        incremental aggregation table updated on each write would be the right approach.

        Args:
            training_feature_means: Dict of feature_name → training set mean for
                                    numeric features. Used to compute mean-shift statistics.

        Returns:
            Summary dict with n_predictions, mean_predicted_lgd, feature_means,
            ks_statistics, and period fields. All values are None if n_predictions == 0.
        """
        with self._connection() as conn:
            count_row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
            n_predictions = count_row[0] if count_row else 0

            if n_predictions == 0:
                return {
                    "n_predictions": 0,
                    "mean_predicted_lgd": None,
                    "feature_means": None,
                    "ks_statistics": None,
                    "period": None,
                }

            lgd_row = conn.execute(
                "SELECT AVG(predicted_lgd), MIN(timestamp), MAX(timestamp) FROM predictions"
            ).fetchone()
            mean_lgd = lgd_row[0]
            min_ts = lgd_row[1]
            max_ts = lgd_row[2]

            rows = conn.execute(
                "SELECT features_json, predicted_lgd FROM predictions"
            ).fetchall()

        # Aggregate feature means from the stored JSON blobs.
        # Only numeric values are aggregated — string values (property_type, state, etc.)
        # are not included in feature_means because their means are not interpretable.
        feature_sums: Dict[str, float] = {}
        feature_counts: Dict[str, int] = {}
        lgd_values = []

        for features_json, lgd in rows:
            lgd_values.append(lgd)
            try:
                features = json.loads(features_json)
                for k, v in features.items():
                    if isinstance(v, (int, float)):
                        feature_sums[k] = feature_sums.get(k, 0) + float(v)
                        feature_counts[k] = feature_counts.get(k, 0) + 1
            except (json.JSONDecodeError, TypeError):
                # A malformed JSON record in the database is a data quality issue
                # but should not prevent the summary from completing. The record is
                # skipped and the summary is computed from the remaining records.
                continue

        feature_means = {
            k: feature_sums[k] / feature_counts[k]
            for k in feature_sums
            if feature_counts.get(k, 0) > 0
        }

        # Mean-shift statistic: |serving_mean - training_mean| / |training_mean|.
        # Division by training_mean normalizes the shift to a fraction of the training
        # scale, making it comparable across features with different magnitudes.
        # A shift of 0.10 (10%) is comparable regardless of whether the feature is
        # orig_ltv (range ~40-120) or orig_interest_rate (range ~3-8).
        ks_statistics = None
        if training_feature_means and feature_means:
            ks_statistics = {}
            for feature, train_mean in training_feature_means.items():
                if feature in feature_means and train_mean != 0:
                    drift = abs(feature_means[feature] - train_mean) / abs(train_mean)
                    ks_statistics[feature] = round(drift, 4)

        return {
            "n_predictions": n_predictions,
            "mean_predicted_lgd": round(mean_lgd, 6) if mean_lgd is not None else None,
            "feature_means": {k: round(v, 4) for k, v in feature_means.items()},
            "ks_statistics": ks_statistics,
            "period": f"{min_ts} to {max_ts}" if min_ts else None,
        }

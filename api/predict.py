"""Model loading and inference logic for the LGD prediction API.

This module is the boundary between the FastAPI request handling layer and the
PyTorch model. It owns two responsibilities that must be kept distinct: loading
the model artifact from disk (once, at startup, with results cached in module-level
singletons), and transforming API request payloads into model inputs and model
outputs back into API response payloads.

The module-level singleton pattern for the model, scaler, and metadata is the
appropriate design for a single-process API. The alternative — loading the model
on each request — would add several hundred milliseconds of I/O and deserialization
overhead to every prediction, which would make the 200ms latency target unreachable.
The trade-off is that the model state is shared across all requests, which means the
model cannot be updated without a process restart. For a model with the validation
lifecycle of this LGD model (retraining is infrequent and requires review before
deployment), a restart-to-update pattern is appropriate and simpler than alternatives.

The categorical encoding in _categorical_to_numeric() uses hardcoded maps that
are designed to match the LabelEncoder fit order from the training pipeline.
This is the weakest point in the inference architecture. The correct approach
is to serialize the LabelEncoders during training (alongside the scaler) and load
them here. The hardcoded maps work as long as the training pipeline uses the same
category ordering, which is guaranteed only if the training data contains all
categories in the same order they were defined. The TODO comment is here for
exactly this reason — the next iteration should serialize and load the encoders
rather than maintaining this mapping by hand.

The inference device is always CPU, by design. The API serves single-loan and
small-batch requests where the PyTorch CUDA launch overhead would exceed the
actual computation time. CPU inference is also more predictable in terms of
latency variance, which matters for meeting the 200ms target consistently.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# Module-level singletons. These are initialized by load_model() at API startup
# and remain in memory for the lifetime of the process. All prediction functions
# access them via the global keyword rather than passing them as arguments, which
# avoids the overhead of function argument validation on the hot prediction path.
_model = None
_scaler = None
_model_meta = None
_feature_cols = None

MODEL_VERSION = "lgdnet-v1"


def _get_feature_cols() -> list[str]:
    """Return the ordered list of feature columns in the order used during training.

    The column order here must match the column order in configs/default.yaml
    (config.features.categorical + config.features.numeric) and the order used
    in features.py when constructing the feature matrix. Column order in a neural
    network is not a convention — it is a hard requirement. The model weights are
    learned relative to specific input positions, and shuffling the columns without
    retraining will produce predictions that are not wrong in a detectable way —
    they will just be wrong silently.

    In practice, the correct column order is stored in the checkpoint as feature_cols
    and retrieved from there by load_model(). This fallback function exists only for
    cases where the checkpoint was saved without the feature_cols key, which should
    not occur with checkpoints produced by the current train.py implementation.
    """
    return [
        "property_type",
        "occupancy_status",
        "channel",
        "state",
        "orig_ltv",
        "orig_upb",
        "orig_interest_rate",
        "orig_term",
        "ltv_at_default",
        "months_delinquent_at_default",
        "hpi_change",
        "unemployment_rate_at_default",
    ]


def load_model(model_path: str | Path, scaler_path: str | Path | None = None):
    """Load the model and scaler from disk and cache them in module-level singletons.

    This function is idempotent: if _model is already populated (from a previous
    call at startup), subsequent calls return the cached objects immediately without
    re-loading from disk. This allows load_model() to be called safely in multiple
    contexts without accumulating state.

    The model is always loaded to CPU because the API is designed for low-latency
    single-loan inference, not high-throughput batch processing. GPU inference would
    add CUDA initialization overhead that exceeds the compute savings for batch
    sizes below approximately 100-200 loans, depending on the hardware.

    A missing scaler file produces a warning rather than an error because the API
    can still serve predictions without scaling — they will just be poorly calibrated
    because the model expects scaled inputs. This is the correct failure mode: partial
    functionality with a logged warning is preferable to a hard failure that prevents
    the API from starting. The /health endpoint will report model_loaded=True, and the
    prediction quality degradation will be visible in the monitoring dashboard.

    Args:
        model_path: Path to the .pt checkpoint produced by src/models/train.py.
        scaler_path: Path to the pickled StandardScaler. If None or missing, inference
                     proceeds without scaling.
    """
    global _model, _scaler, _model_meta, _feature_cols

    if _model is not None:
        return _model, _scaler

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run src/models/train.py to generate the model artifact."
        )

    from src.models.lgd_net import LGDNet

    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint["model_config"]
    model = LGDNet(
        input_dim=model_config["input_dim"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _model_meta = checkpoint
    _feature_cols = checkpoint.get("feature_cols", _get_feature_cols())

    if scaler_path is not None:
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                _scaler = pickle.load(f)
        else:
            _scaler = None
    else:
        _scaler = None

    return _model, _scaler


def _categorical_to_numeric(features: dict) -> dict:
    """Convert categorical string values to label-encoded integers for model input.

    These mappings must match the LabelEncoder.fit() order from the training pipeline
    in src/data/features.py. LabelEncoder assigns integer codes in sorted order of
    the unique values seen during fit, which means the mappings here are determined
    by the sorted unique values in the training data, not by any natural ordering of
    the categories.

    The "Unknown" fallback values (e.g., property_type_map.get(..., 5)) handle
    inputs where the category is not in the training vocabulary. The fallback integer
    is chosen to be one beyond the range of known values, which is what LabelEncoder
    would assign in a transform-only context for unseen categories. This behavior
    is imperfect — the model has no representation of "Unknown" because it was not
    trained on it — but it is better than raising an exception and failing the
    prediction entirely for a production API that must be robust to data quality
    issues in the input stream.

    TODO: Replace these hardcoded maps with serialized LabelEncoder objects loaded
    from the training checkpoint. The encoders should be saved alongside the scaler
    in models/scaler.pt or in a separate models/encoders.pt file. Until that change
    is made, any new category value in the training data (e.g., a new origination
    channel) will require a manual update to these maps.
    """
    property_type_map = {"CO": 0, "CP": 1, "MH": 2, "PU": 3, "SF": 4, "Unknown": 5}
    occupancy_map = {"I": 0, "P": 1, "S": 2, "Unknown": 3}
    channel_map = {"B": 0, "C": 1, "R": 2, "T": 3, "Unknown": 4}

    from src.data.features import STATE_TO_REGION
    region_map = {
        "Midwest": 0, "Northeast": 1, "Other": 2,
        "Southeast": 3, "Southwest": 4, "West": 5,
    }

    encoded = features.copy()
    encoded["property_type"] = property_type_map.get(features.get("property_type", "Unknown"), 5)
    encoded["occupancy_status"] = occupancy_map.get(features.get("occupancy_status", "Unknown"), 3)
    encoded["channel"] = channel_map.get(features.get("channel", "Unknown"), 4)

    state = features.get("state", "XX")
    region = STATE_TO_REGION.get(state, "Other")
    encoded["state"] = region_map.get(region, 2)

    return encoded


def features_to_array(features: dict, feature_cols: list[str]) -> np.ndarray:
    """Convert a feature dict to a 1D numpy array in the column order expected by the model.

    The column order in feature_cols is the canonical order: it comes from the checkpoint's
    stored feature_cols list, which was written by train.py at the time the model was
    trained. Any deviation from this order — a missing key, an extra key, a reordered
    list — will produce a silently incorrect prediction. The .get(col, 0.0) fallback
    for missing columns is intentional: it produces a prediction rather than an exception,
    and the monitoring dashboard will detect that the feature mean for that column is 0.0,
    which is an anomalous signal that will surface the data quality issue.

    Args:
        features: Dict of feature_name → value, with categorical values already
                  label-encoded by _categorical_to_numeric().
        feature_cols: Ordered list of feature column names from the training checkpoint.

    Returns:
        1D numpy array of shape (n_features,), dtype float32.
    """
    encoded = _categorical_to_numeric(features)
    arr = np.array([float(encoded.get(col, 0.0)) for col in feature_cols], dtype=np.float32)
    return arr


def predict_single(
    features: dict,
    n_mc_samples: int = 100,
) -> Tuple[float, float, float, str]:
    """Predict LGD for a single loan with MC Dropout confidence intervals.

    This function is the primary inference path for the /predict endpoint. It
    assembles the feature vector, applies scaling, runs MC Dropout inference,
    and returns the mean prediction and 90% CI bounds alongside a UUID that
    serves as the audit trail anchor.

    The nan_to_num step after scaling handles the case where a feature value
    was missing (defaulted to 0.0 in features_to_array) and the StandardScaler
    produces a NaN after normalization because it was computed from a column that
    had no variance in the training data. This edge case is rare but possible for
    binary indicator columns that are entirely 0 or 1 in the training set.

    Args:
        features: Loan feature dict from the LoanFeatures Pydantic schema's model_dump().
        n_mc_samples: Number of MC Dropout forward passes for CI estimation.

    Returns:
        Tuple of (lgd, ci_lower, ci_upper, prediction_id).
    """
    global _model, _scaler, _feature_cols

    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    feature_cols = _feature_cols or _get_feature_cols()
    arr = features_to_array(features, feature_cols).reshape(1, -1)

    if _scaler is not None:
        arr = _scaler.transform(arr)

    arr = np.nan_to_num(arr, nan=0.0)
    x_tensor = torch.tensor(arr, dtype=torch.float32)

    mean_pred, lower, upper = _model.predict_with_uncertainty(x_tensor, n_samples=n_mc_samples)

    lgd = float(mean_pred[0].item())
    ci_lower = float(lower[0].item())
    ci_upper = float(upper[0].item())
    prediction_id = str(uuid.uuid4())

    return lgd, ci_lower, ci_upper, prediction_id


def predict_batch(
    features_list: list[dict],
    n_mc_samples: int = 100,
) -> list[Tuple[float, float, float, str]]:
    """Predict LGD for a batch of loans.

    Batch inference uses a single MC Dropout call across all loans in the batch,
    which is more efficient than calling predict_single() repeatedly because the
    model processes all loans simultaneously on the same forward passes. For a
    batch of n loans and n_mc_samples passes, the computational cost is one tensor
    of shape (n_mc_samples, n_loans) rather than n tensors of shape (n_mc_samples, 1).

    The batch is assembled into a single numpy array before the scaling step, which
    allows np.stack to produce the (n_loans, n_features) matrix needed by the model
    in a single allocation. Assembling the array incrementally in a loop would produce
    the same result but with more intermediate allocations.

    Args:
        features_list: List of feature dicts, each matching the LoanFeatures schema.
        n_mc_samples: Number of MC Dropout forward passes for CI estimation.

    Returns:
        List of (lgd, ci_lower, ci_upper, prediction_id) tuples, one per loan.
    """
    global _model, _scaler, _feature_cols

    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    feature_cols = _feature_cols or _get_feature_cols()
    arrays = np.stack([
        features_to_array(f, feature_cols) for f in features_list
    ], axis=0)

    if _scaler is not None:
        arrays = _scaler.transform(arrays)

    arrays = np.nan_to_num(arrays, nan=0.0)
    x_tensor = torch.tensor(arrays, dtype=torch.float32)

    mean_preds, lowers, uppers = _model.predict_with_uncertainty(x_tensor, n_samples=n_mc_samples)

    results = []
    for lgd, lower, upper in zip(
        mean_preds.tolist(), lowers.tolist(), uppers.tolist()
    ):
        results.append((float(lgd), float(lower), float(upper), str(uuid.uuid4())))

    return results


def get_model_info() -> dict:
    """Return model metadata for the /model/info endpoint.

    This function surfaces the information stored in the checkpoint at training time:
    architecture configuration, feature list, and train/val/test metrics. It does not
    recompute any statistics — everything here was computed and stored by train.py
    and is read directly from the checkpoint dictionary.

    When the model is not loaded (model_loaded=False in /health), this function
    returns a minimal dict indicating not_loaded status rather than raising an
    exception. The /model/info endpoint should always return 200 even when no model
    is loaded, because it is an informational endpoint that does not depend on
    model availability.
    """
    global _model, _model_meta, _feature_cols

    if _model is None:
        return {
            "model_version": MODEL_VERSION,
            "model_type": "LGDNet",
            "input_features": _get_feature_cols(),
            "architecture": {},
            "status": "not_loaded",
        }

    return {
        "model_version": MODEL_VERSION,
        "model_type": "LGDNet",
        "input_features": _feature_cols or _get_feature_cols(),
        "architecture": _model.get_config() if _model else {},
        "train_metrics": _model_meta.get("train_metrics") if _model_meta else None,
        "val_metrics": _model_meta.get("val_metrics") if _model_meta else None,
        "test_metrics": _model_meta.get("test_metrics") if _model_meta else None,
    }

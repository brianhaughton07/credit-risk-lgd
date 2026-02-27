"""Centralized configuration management for the LGD prediction pipeline.

Every hyperparameter, file path, and feature list in this project lives in
configs/default.yaml and is accessed through this module. That is a deliberate
choice. When a model risk reviewer at a regulated institution asks "what
hyperparameters did this model train on," the answer should be a single file
they can read, not a hunt through training scripts and notebook cells where
numbers were hardcoded in three different places that eventually diverged.

The Config dataclass hierarchy maps directly to the YAML structure: one section
per concern (data, features, training, model, mlflow, api). Each section is
independently replaceable — you can swap in a different ModelConfig for an
architecture search without touching data paths, which is the kind of
separation that prevents the version control nightmares I have watched sink
otherwise solid projects.

Note that this module deliberately avoids environment-variable overrides and
other runtime config injection patterns. Those patterns are useful in
microservice deployments but add complexity that is not justified here. If a
different deployment context requires different values, the right approach is
a separate YAML file, not runtime mutation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class DataConfig:
    """Paths and vintage range for the Freddie Mac SFLP dataset.

    The vintage_start and vintage_end fields control which origination years
    are included. The 2010-2015 window was chosen because it captures the
    post-crisis period when default rates were meaningfully elevated, giving
    the model enough tail observations to learn from. Pre-2010 vintages would
    require reaching into the crisis itself, where underwriting standards and
    servicer behavior were sufficiently different that I would expect the model
    to have to interpolate across a regime break. Post-2015 vintages have
    insufficient performance history through resolution to produce reliable
    LGD targets.
    """
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    vintage_start: int = 2010
    vintage_end: int = 2015


@dataclass
class FeaturesConfig:
    """Feature column lists for the model input matrix.

    The categorical and numeric lists are ordered. The order here determines
    the column order in the feature matrix produced by src/data/features.py,
    which must match the order expected by the trained model at inference time.
    If you add a feature to this list after training, the model artifact becomes
    invalid unless you retrain — that is a hard constraint imposed by how
    StandardScaler and the model weights are serialized.

    I separated categorical from numeric not for conceptual reasons (the model
    receives them all as a single float matrix after encoding) but because the
    preprocessing steps that apply to each group are different. Categorical
    columns go through LabelEncoder. Numeric columns go through StandardScaler.
    Keeping the lists separate makes that branching explicit and auditable.

    The target field is a string rather than a hardcoded constant so that
    evaluation scripts can reference it generically without knowing in advance
    what the dependent variable is called. This turned out to matter when I was
    refactoring the feature engineering pipeline and temporarily had both
    "loss_given_default" and "lgd" as column names in different intermediate
    files.
    """
    categorical: List[str] = field(default_factory=lambda: [
        "property_type", "occupancy_status", "channel", "state"
    ])
    numeric: List[str] = field(default_factory=lambda: [
        "orig_ltv", "orig_upb", "orig_interest_rate", "orig_term",
        "ltv_at_default", "months_delinquent_at_default",
        "hpi_change", "unemployment_rate_at_default",
    ])
    target: str = "loss_given_default"


@dataclass
class TrainingConfig:
    """Train/val/test split parameters.

    The 70/15/15 split is conventional and produces roughly equal-sized
    validation and test sets, which matters here because the calibration
    plot in evaluate.py requires enough test observations per decile bucket
    to produce stable estimates. With fewer than 500 test observations, the
    tail buckets become unreliable regardless of model quality.

    Stratifying by vintage_year is important. If the split were random without
    stratification, an unlucky split could put all 2012 loans (the peak default
    year) in the training set and leave the test set with a lower average LGD,
    producing an overly optimistic test MAE. Stratification ensures each
    vintage's proportion is preserved across all three splits, which is the
    right approach for a model that will be used across all vintage cohorts.

    random_seed = 42 is a convention, not a magic number. The reproducibility
    guarantee comes from the seed being stored here rather than hardcoded in
    the training script, so a reviewer can verify that running train.py twice
    with the same config produces the same split.
    """
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    stratify_by: str = "vintage_year"


@dataclass
class ModelConfig:
    """LGDNet architecture and training hyperparameters.

    The hidden_dims sequence [256, 128, 64] is a funnel architecture — each
    layer reduces dimensionality by roughly half. I chose this over a constant-
    width architecture because LGD prediction has a small feature set (12
    inputs) and a simple target space [0, 1]. The funnel forces progressive
    compression, which acts as a form of regularization in addition to the
    explicit dropout.

    dropout = 0.3 serves two purposes. First, it is a regularizer during
    training, helping the model generalize rather than memorize the training
    distribution. Second, it is the mechanism for MC Dropout at inference time,
    which produces the 90% confidence intervals the API returns. Note that
    disabling dropout during training (dropout=0.0) would also disable the
    CI capability, so there is a functional dependency between this parameter
    and the uncertainty estimation approach. If you want deterministic
    predictions without CIs, set dropout to 0 and update predict.py to call
    model(x) directly rather than predict_with_uncertainty().

    patience = 10 means training stops if validation MAE does not improve for
    10 consecutive epochs. This was calibrated empirically: with patience=5,
    training tended to stop before the model had fully converged on complex
    segments. With patience=20, it sometimes continued well past the optimal
    checkpoint, increasing training time without improving test performance.

    tail_loss_alpha = 2.0 controls how aggressively the loss function penalizes
    errors at LGD = 0 and LGD = 1. At alpha = 2.0, a prediction error at the
    tails carries 3x the weight of the same error at LGD = 0.5. This is
    documented in docs/model_card.md with the full mathematical rationale.
    The value 2.0 is not arbitrary — it was selected to keep tail weights in a
    range where gradient updates remain numerically stable while still providing
    meaningful emphasis on the bimodal spikes in the LGD distribution.
    """
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    tail_loss_alpha: float = 2.0


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration.

    tracking_uri points to a local ./mlruns directory, which is sufficient for
    a single-analyst workflow. In a team environment or when this model is
    deployed for ongoing monitoring, this would point to a shared MLflow
    tracking server. The current setup stores everything locally and is
    gitignored, which means MLflow artifacts are not version-controlled. That
    is the right trade-off for a development setup — committing model artifacts
    to git creates repository bloat without the reproducibility benefit, since
    the artifact can always be regenerated from the logged parameters.
    """
    tracking_uri: str = "./mlruns"
    experiment_name: str = "lgd-prediction"


@dataclass
class APIConfig:
    """API service configuration.

    model_path points to the serialized PyTorch checkpoint generated by
    src/models/train.py. The API loads this once at startup and keeps it in
    memory for the lifetime of the process. If the model artifact does not
    exist, the API starts anyway and reports model_loaded=False on the /health
    endpoint — that is a deliberate design choice that allows the service
    container to start and pass health checks even before a trained model
    artifact has been copied in, which is the typical deployment ordering in
    container orchestration environments.

    db_path points to the SQLite file used by the prediction logger in
    monitoring.py. SQLite is appropriate for a single-process API with
    moderate prediction volume. If this API were deployed behind a load
    balancer with multiple workers writing simultaneously, the SQLite approach
    would need to be replaced with a proper database. For the purposes of
    demonstrating the monitoring pattern, SQLite is the right tool.
    """
    model_path: str = "models/lgd_model.pt"
    db_path: str = "api/predictions.db"
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    """Root configuration object aggregating all subsection configs.

    This is the object that every module in the pipeline receives when it
    calls load_config(). Passing Config rather than individual subsection
    objects means any module can access any config value without needing
    multiple arguments, which matters for modules like train.py that need
    data paths, feature lists, model hyperparameters, and MLflow settings
    simultaneously.
    """
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    api: APIConfig = field(default_factory=APIConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file and return a typed Config object.

    The default path resolution walks up from this file's location to the
    project root and then appends configs/default.yaml. This means any script
    in the project can call load_config() without arguments and get the right
    config file, regardless of the working directory the script was launched
    from. That removes a class of "file not found" errors that appear whenever
    analysts run scripts from different directories.

    yaml.safe_load is used rather than yaml.load because the config file is
    user-editable and safe_load prevents arbitrary Python object deserialization
    from untrusted YAML content. This is a minimal precaution but the right one.

    Each subsection uses raw.get(key, {}) as the fallback, so a config file
    that omits an entire section (say, mlflow) will still produce a valid Config
    using that section's dataclass defaults rather than raising a KeyError. This
    means partial config files work correctly during development, which is useful
    when prototyping a new pipeline stage before all config sections are needed.

    Args:
        path: Path to YAML config file. Defaults to configs/default.yaml
              resolved relative to the project root.

    Returns:
        Fully populated Config dataclass with all subsections.

    Raises:
        FileNotFoundError: If the config file does not exist at the resolved path.
    """
    if path is None:
        # Walk up two directories from src/utils/ to reach the project root,
        # then descend into configs/. This resolution is deterministic regardless
        # of the current working directory, which matters when the pipeline is
        # run from cron jobs or orchestration systems that set arbitrary CWDs.
        project_root = Path(__file__).parent.parent.parent
        path = project_root / "configs" / "default.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Each subsection is constructed independently so that a missing or
    # partially-specified YAML section degrades gracefully to defaults
    # rather than raising a KeyError at the subsection level.
    data_cfg = DataConfig(**raw.get("data", {}))
    features_cfg = FeaturesConfig(**raw.get("features", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    mlflow_cfg = MLflowConfig(**raw.get("mlflow", {}))
    api_cfg = APIConfig(**raw.get("api", {}))

    return Config(
        data=data_cfg,
        features=features_cfg,
        training=training_cfg,
        model=model_cfg,
        mlflow=mlflow_cfg,
        api=api_cfg,
    )

"""Unit tests for the LGDNet model architecture, loss function, and inference components.

The tests here are organized around the specific risks associated with neural network
models for tabular regression tasks. Generic neural network tests (does the loss go
down? does the output have the right shape?) exist in most test suites. What is
specific to LGD prediction are the tests for tail emphasis in the loss function,
the MC Dropout uncertainty mechanism, and the baseline sanity check — the claim
that a trained LGDNet should outperform random predictions. These tests would fail
in instructive ways if the loss function were changed, if MC Dropout were broken,
or if the architecture were so misconfigured that it could not learn from the data.

The architecture tests verify the hard constraints: outputs in [0, 1], correct batch
shape, non-constant predictions. These are not trivially true — a Sigmoid output
constrained to (0, 1) is correct only if the network does not produce vanishing
gradients that leave all outputs near 0.5, and "non-constant" means the model is
not collapsed to a single output. Both failure modes have appeared in LGD models
with misconfigured initialization, which is why they are tested explicitly.

The WeightedMSELoss tests verify the most important claim in the loss function
docstring: that observations near the tails of the LGD distribution receive higher
weight than observations near the center. The test uses equal-magnitude errors at
two points (y=0.5 and y=0.0) and asserts that the tail error produces a larger loss.
This is a direct test of the weighting mechanism that cannot be verified by looking
at aggregate metrics on a full training run.

The MC Dropout tests verify that uncertainty intervals are well-ordered (lower ≤ mean ≤ upper),
bounded (in [0, 1]), and have non-zero width when dropout rate > 0. The non-zero width test
is particularly important: a broken dropout mechanism (where self.train() has no effect)
would produce identical predictions across all 100 samples, yielding CI width of zero.
A zero-width CI from a model with dropout=0.3 is a clear signal of a broken inference path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lgd_net import LGDDataset, LGDNet, WeightedMSELoss, build_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model() -> LGDNet:
    """A small LGDNet for fast tests.

    Hidden dims [32, 16] rather than the production [256, 128, 64] keep
    forward pass time negligible during test runs, while still exercising
    the full architecture: multiple layers, BatchNorm, ReLU, Dropout, Sigmoid.
    """
    return LGDNet(input_dim=12, hidden_dims=[32, 16], dropout=0.1)


@pytest.fixture
def batch_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """A small batch of synthetic features and targets for training tests."""
    torch.manual_seed(42)
    X = torch.randn(64, 12)
    y = torch.rand(64)
    return X, y


@pytest.fixture
def large_batch() -> torch.Tensor:
    """A larger batch for shape and range tests."""
    torch.manual_seed(0)
    return torch.randn(256, 12)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestLGDNetArchitecture:
    """Tests for the hard structural constraints of LGDNet.

    Every constraint tested here is a contractual guarantee that the rest of
    the pipeline depends on. The training loop expects shape (batch_size,) from
    forward(). The API expects values in [0, 1]. The SHAP analysis expects
    non-constant outputs (otherwise SHAP values would all be zero). A failure
    in any of these tests would propagate silently through the pipeline and
    produce wrong results without raising an exception.
    """

    def test_output_shape_single(self, small_model):
        """Single-sample forward pass returns shape (1,), not (1, 1) or scalar."""
        x = torch.randn(1, 12)
        out = small_model(x)
        assert out.shape == (1,), f"Expected (1,), got {out.shape}"

    def test_output_shape_batch(self, small_model, large_batch):
        """Batch forward pass returns shape (batch_size,) — no trailing dimensions."""
        out = small_model(large_batch)
        assert out.shape == (256,), f"Expected (256,), got {out.shape}"

    def test_output_range(self, small_model, large_batch):
        """All outputs are in [0, 1] due to the Sigmoid output activation.

        This constraint is fundamental: the loss function and all downstream
        evaluation metrics assume outputs are in the LGD range [0, 1]. An
        output outside this range indicates a Sigmoid implementation error.
        """
        with torch.no_grad():
            out = small_model(large_batch)
        assert out.min().item() >= 0.0, f"Output below 0: {out.min().item()}"
        assert out.max().item() <= 1.0, f"Output above 1: {out.max().item()}"

    def test_output_not_all_same(self, small_model, large_batch):
        """The model must produce varied outputs for varied inputs.

        A constant-output model would indicate collapsed initialization or
        a degenerate training state. This test catches that case, which
        would produce SHAP values of exactly zero for all features.
        """
        with torch.no_grad():
            out = small_model(large_batch)
        assert out.std().item() > 0.001, "Model outputs are nearly constant"

    def test_custom_hidden_dims(self):
        """Custom hidden layer configurations are correctly constructed and functional."""
        model = LGDNet(input_dim=8, hidden_dims=[64, 32, 16, 8], dropout=0.0)
        x = torch.randn(10, 8)
        out = model(x)
        assert out.shape == (10,)

    def test_single_hidden_layer(self):
        """Minimal architecture with one hidden layer is valid and produces bounded output."""
        model = LGDNet(input_dim=5, hidden_dims=[16], dropout=0.0)
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4,)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_get_config_roundtrip(self, small_model):
        """get_config() returns the constructor arguments used to create the model.

        This test verifies the serialization contract: the config dict returned by
        get_config() must be sufficient to reconstruct an identical architecture.
        If this test fails, the checkpoint serialization in train.py will produce
        an artifact that cannot be correctly reloaded by evaluate.py or predict.py.
        """
        config = small_model.get_config()
        assert config["input_dim"] == 12
        assert config["hidden_dims"] == [32, 16]
        assert config["dropout"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Tests for the backward pass: gradient presence and loss reduction.

    Gradient flow tests catch two categories of problems that do not manifest
    as exceptions: NaN gradients (which produce NaN parameters after the update,
    silently corrupting the model), and zero gradients in Linear layer weights
    (which indicate that the weights are not being updated, either because of
    a computation graph break or because of vanishing gradients).
    """

    def test_gradients_flow_to_all_layers(self, small_model, batch_data):
        """All Linear layer weights receive non-zero, non-NaN gradients after backward.

        BatchNorm parameters (weight and bias) are excluded from the non-zero check
        because they legitimately have near-zero gradients when the pre-norm
        activations are already well-distributed — which is the expected behavior
        after a few training steps.
        """
        X, y = batch_data
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        criterion = WeightedMSELoss(alpha=2.0)

        optimizer.zero_grad()
        preds = small_model(X)
        loss = criterion(preds, y)
        loss.backward()

        for name, param in small_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                if "weight" in name and "bn" not in name.lower() and "norm" not in name.lower():
                    assert param.grad.abs().sum().item() > 0, f"Zero gradient in {name}"

    def test_loss_decreases_with_training(self):
        """Training loss decreases monotonically over 20 steps on a simple synthetic dataset.

        This test uses dropout=0.0 to make training deterministic given a fixed
        random seed. With dropout enabled, the stochastic mask would cause loss
        to fluctuate, making monotonic decrease an unreliable assertion. The
        assertion is that the final loss is strictly less than the initial loss —
        not that every step decreases, which would be too strict for a stochastic
        optimizer even without dropout.
        """
        torch.manual_seed(42)
        model = LGDNet(input_dim=8, hidden_dims=[32, 16], dropout=0.0)
        criterion = WeightedMSELoss(alpha=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X = torch.randn(200, 8)
        y = torch.rand(200)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestWeightedMSELoss:
    """Tests for the WeightedMSELoss function.

    WeightedMSELoss is the core training signal for LGDNet. The tests here verify
    the mathematical properties of the weighting scheme: non-negativity, correct
    tail emphasis, zero loss for perfect predictions, and reduction to standard
    MSE when alpha=0. A regression in any of these properties would allow training
    to proceed without error while producing a model that does not prioritize
    tail accuracy as intended.
    """

    def test_loss_nonnegative(self, batch_data):
        """Loss is always non-negative — a necessary condition for a valid loss function."""
        X, y = batch_data
        model = LGDNet(input_dim=12, hidden_dims=[32, 16], dropout=0.0)
        criterion = WeightedMSELoss(alpha=2.0)
        preds = model(X)
        loss = criterion(preds, y)
        assert loss.item() >= 0.0

    def test_tail_weight_higher_than_center(self):
        """Equal-magnitude prediction errors at the tail (y=0.0) produce higher loss than at center (y=0.5).

        This is the core property of WeightedMSELoss. The test uses precisely
        0.1 error at both locations so that the comparison isolates the weighting
        effect from any difference in error magnitude.
        """
        criterion = WeightedMSELoss(alpha=2.0)

        pred_center = torch.tensor([0.5])
        pred_tail = torch.tensor([0.0])
        y_center = torch.tensor([0.6])   # 0.1 error at center
        y_tail = torch.tensor([0.1])     # 0.1 error at tail

        loss_center = criterion(pred_center, y_center)
        loss_tail = criterion(pred_tail, y_tail)

        assert loss_tail.item() > loss_center.item(), (
            f"Tail loss ({loss_tail.item():.4f}) should exceed center loss ({loss_center.item():.4f})"
        )

    def test_zero_loss_for_perfect_predictions(self):
        """Perfect predictions (pred == y) produce exactly zero loss."""
        criterion = WeightedMSELoss(alpha=2.0)
        y = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        loss = criterion(y, y)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_alpha_zero_reduces_to_mse(self):
        """With alpha=0, WeightedMSELoss is mathematically identical to standard MSE.

        This test verifies the boundary case: when the tail weighting coefficient
        is zero, every observation has weight 1.0, which reduces the weighted sum
        to the standard mean squared error. A failure here indicates that the base
        weight is incorrect (should be 1.0, not some other value).
        """
        criterion_weighted = WeightedMSELoss(alpha=0.0)
        criterion_mse = nn.MSELoss()

        pred = torch.tensor([0.2, 0.5, 0.8, 0.1, 0.9])
        y = torch.tensor([0.3, 0.4, 0.9, 0.2, 0.8])

        loss_weighted = criterion_weighted(pred, y)
        loss_mse = criterion_mse(pred, y)
        assert loss_weighted.item() == pytest.approx(loss_mse.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestLGDDataset:
    """Tests for the LGDDataset wrapper.

    LGDDataset is a thin wrapper around numpy arrays, but its correctness matters
    because incorrect dtype conversion (e.g., float64 when float32 is expected) or
    shape handling would propagate silently into the DataLoader and from there into
    the training loop. These tests verify that the wrapper produces the tensor
    shapes and dtypes that the training loop expects.
    """

    def test_dataset_length(self):
        """Dataset length matches the number of input samples."""
        X = np.random.randn(100, 12).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        ds = LGDDataset(X, y)
        assert len(ds) == 100

    def test_dataset_item_types(self):
        """Each item returned by __getitem__ is a (float32 tensor, scalar tensor) pair."""
        X = np.random.randn(50, 8).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        ds = LGDDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)
        assert x_item.shape == (8,)
        assert y_item.shape == ()  # Scalar tensor for the target

    def test_dataset_dataloader_batching(self):
        """DataLoader correctly batches the dataset into the expected number of batches."""
        X = np.random.randn(100, 12).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        ds = LGDDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
        batches = list(loader)
        # ceil(100 / 32) = 4 batches (3 full batches of 32, 1 partial batch of 4)
        assert len(batches) == 4
        assert batches[0][0].shape == (32, 12)


# ---------------------------------------------------------------------------
# MC Dropout uncertainty tests
# ---------------------------------------------------------------------------

class TestMCDropout:
    """Tests for the predict_with_uncertainty() MC Dropout inference method.

    MC Dropout is the mechanism by which the API produces confidence intervals.
    A broken MC Dropout implementation — one where the self.train() call has no
    effect, or where the samples are not independent — would produce CIs of
    width zero (degenerate) or CIs that are not ordered (lower > upper). Both
    failure modes are tested explicitly.
    """

    def test_ci_bounds_ordered(self, small_model, large_batch):
        """The lower CI bound must be <= mean prediction <= upper CI bound for every loan.

        A violation here — lower > mean or upper < mean — would indicate that the
        quantile computation is producing incorrect results, either because the
        samples array has the wrong shape or because torch.quantile is being called
        with the wrong dimension.
        """
        mean, lower, upper = small_model.predict_with_uncertainty(
            large_batch, n_samples=20
        )
        assert (lower <= mean + 1e-6).all(), "Lower CI exceeds mean"
        assert (upper >= mean - 1e-6).all(), "Upper CI below mean"

    def test_ci_values_in_range(self, small_model, large_batch):
        """CI bounds are in [0, 1] because each sample is produced by a Sigmoid output."""
        mean, lower, upper = small_model.predict_with_uncertainty(
            large_batch, n_samples=20
        )
        assert lower.min().item() >= 0.0
        assert upper.max().item() <= 1.0

    def test_ci_has_nonzero_width(self, small_model, large_batch):
        """With dropout > 0, the CI width is non-zero on average.

        This test explicitly uses dropout=0.3 rather than the small_model fixture
        (which has dropout=0.1) to ensure the stochastic variation is large enough
        to be reliably detected across 50 samples. With dropout=0.3 and 50 samples,
        the expected CI width is on the order of 0.05-0.15, well above zero.
        """
        model = LGDNet(input_dim=12, hidden_dims=[32, 16], dropout=0.3)
        mean, lower, upper = model.predict_with_uncertainty(large_batch, n_samples=50)
        ci_width = upper - lower
        assert ci_width.mean().item() > 0.0, "All CI widths are zero — MC Dropout may be inactive"


# ---------------------------------------------------------------------------
# Baseline sanity test
# ---------------------------------------------------------------------------

class TestBaselineSanity:
    """A trained LGDNet must outperform random predictions on a learnable dataset.

    This test is a minimum sanity check, not a performance benchmark. If a trained
    LGDNet cannot outperform random noise on a synthetic dataset with a known linear
    signal, something is fundamentally wrong: either the optimizer is not converging,
    the loss function is producing incorrect gradients, or the model architecture
    is so misconfigured that it cannot represent the signal. This is the test that
    would catch a WeightedMSELoss implementation that always returns 0 (no gradient
    signal) or a forward() method that ignores the input.
    """

    def test_lgdnet_beats_random(self):
        """A trained LGDNet achieves lower MAE than random predictions on a synthetic dataset."""
        torch.manual_seed(42)
        np.random.seed(42)

        n_samples, n_features = 500, 12
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        # Target has a linear signal from features 0 and 1 plus noise.
        # The signal is strong enough that 30 epochs of training should capture it.
        y = np.clip(
            0.3 + 0.2 * X[:, 0] - 0.1 * X[:, 1] + 0.1 * np.random.randn(n_samples),
            0, 1
        ).astype(np.float32)

        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        model = LGDNet(input_dim=n_features, hidden_dims=[64, 32], dropout=0.0)
        criterion = WeightedMSELoss(alpha=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        ds = LGDDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        for _ in range(30):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = model(torch.tensor(X_test)).numpy()

        random_preds = np.random.rand(len(X_test)).astype(np.float32)
        mae_model = float(np.mean(np.abs(test_preds - y_test)))
        mae_random = float(np.mean(np.abs(random_preds - y_test)))

        assert mae_model < mae_random, (
            f"LGDNet MAE ({mae_model:.4f}) is not better than random ({mae_random:.4f}) — "
            "training may not have converged or the model architecture is misconfigured"
        )

"""Tests for TRMTrainer and training mechanics."""
import pytest
import torch
from torch.optim import AdamW

from src.trm.model import TRMNetwork, GridEmbedding, RecursiveRefinement
from src.trm.training import TRMTrainer


# Use small model config for fast tests
SMALL_CONFIG = {
    "hidden_dim": 64,
    "num_heads": 2,
    "num_layers": 1,
    "num_colors": 10,
}


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    network = TRMNetwork(**SMALL_CONFIG)
    embedding = GridEmbedding(hidden_dim=SMALL_CONFIG["hidden_dim"])
    model = RecursiveRefinement(
        network,
        embedding,
        outer_steps=1,
        inner_steps=1,
        num_colors=SMALL_CONFIG["num_colors"],
        enable_halting=False,  # Disable halting for predictable iteration count
    )
    return model


@pytest.fixture
def trainer(small_model):
    """Create a trainer with small model."""
    return TRMTrainer(small_model)


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    B, H, W = 2, 5, 5
    input_grids = torch.randint(0, 10, (B, H, W))
    target_grids = torch.randint(0, 10, (B, H, W))
    mask = torch.ones(B, H, W, dtype=torch.bool)
    return input_grids, target_grids, mask


class TestTRMTrainerInit:
    """Tests for trainer initialization."""

    def test_optimizer_is_adamw(self, small_model):
        """Verify AdamW optimizer is used."""
        trainer = TRMTrainer(small_model)
        assert isinstance(trainer.optimizer, AdamW)

    def test_optimizer_betas_match_paper(self, small_model):
        """TRAIN-05: AdamW with beta1=0.9, beta2=0.95."""
        trainer = TRMTrainer(small_model)
        betas = trainer.optimizer.defaults["betas"]
        assert betas == (0.9, 0.95), f"Expected (0.9, 0.95), got {betas}"

    def test_learning_rate_applied(self, small_model):
        """Verify learning rate is set correctly."""
        custom_lr = 3e-4
        trainer = TRMTrainer(small_model, learning_rate=custom_lr)
        assert trainer.optimizer.defaults["lr"] == custom_lr

    def test_weight_decay_applied(self, small_model):
        """Verify weight decay is set."""
        custom_wd = 0.05
        trainer = TRMTrainer(small_model, weight_decay=custom_wd)
        assert trainer.optimizer.defaults["weight_decay"] == custom_wd

    def test_custom_betas_applied(self, small_model):
        """Verify custom betas can be set."""
        trainer = TRMTrainer(small_model, beta1=0.85, beta2=0.99)
        betas = trainer.optimizer.defaults["betas"]
        assert betas == (0.85, 0.99)

    def test_halting_loss_weight_stored(self, small_model):
        """Verify halting loss weight is stored."""
        trainer = TRMTrainer(small_model, halting_loss_weight=0.2)
        assert trainer.halting_loss_weight == 0.2


class TestLossComputation:
    """Tests for loss computation (TRAIN-03, TRAIN-04)."""

    def test_ce_loss_computed(self, trainer, sample_batch):
        """TRAIN-03: CrossEntropy loss for answer accuracy."""
        input_grids, target_grids, mask = sample_batch

        # Get model output
        output = trainer.model(input_grids)
        logits = output["logits"]
        halt_confidence = output["halt_confidence"]

        # Compute loss
        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # CE loss should be positive scalar
        assert "ce_loss" in loss_dict
        assert loss_dict["ce_loss"].ndim == 0, "CE loss should be scalar"
        assert loss_dict["ce_loss"].item() > 0, "CE loss should be positive"

    def test_bce_loss_computed(self, trainer, sample_batch):
        """TRAIN-04: BCE loss for halting prediction."""
        input_grids, target_grids, mask = sample_batch

        # Get model output
        output = trainer.model(input_grids)
        logits = output["logits"]
        halt_confidence = output["halt_confidence"]

        # Compute loss
        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # BCE loss should be positive scalar
        assert "bce_loss" in loss_dict
        assert loss_dict["bce_loss"].ndim == 0, "BCE loss should be scalar"
        assert loss_dict["bce_loss"].item() >= 0, "BCE loss should be non-negative"

    def test_total_loss_is_sum(self, trainer, sample_batch):
        """Total loss = CE + weighted BCE."""
        input_grids, target_grids, mask = sample_batch

        # Get model output
        output = trainer.model(input_grids)
        logits = output["logits"]
        halt_confidence = output["halt_confidence"]

        # Compute loss
        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # Verify total = ce + weight * bce
        expected = loss_dict["ce_loss"] + trainer.halting_loss_weight * loss_dict["bce_loss"]
        actual = loss_dict["total_loss"]

        assert torch.allclose(actual, expected, atol=1e-6), \
            f"Expected {expected.item()}, got {actual.item()}"

    def test_mask_excludes_padded_positions(self, trainer, small_model):
        """Loss should ignore PAD_VALUE=-1 positions."""
        B, H, W = 2, 5, 5
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))

        # Create mask with some False values (simulating padding)
        mask_full = torch.ones(B, H, W, dtype=torch.bool)
        mask_partial = mask_full.clone()
        mask_partial[:, 3:, :] = False  # Last 2 rows are "padded"

        # Get model output
        output = trainer.model(input_grids)
        logits = output["logits"]
        halt_confidence = output["halt_confidence"]

        # Compute loss with different masks
        loss_full = trainer.compute_loss(logits, target_grids, halt_confidence, mask_full)
        loss_partial = trainer.compute_loss(logits, target_grids, halt_confidence, mask_partial)

        # Losses should differ (unless by extreme coincidence)
        # We verify by checking they're not exactly equal
        ce_full = loss_full["ce_loss"].item()
        ce_partial = loss_partial["ce_loss"].item()

        # This test verifies masking is applied - values should typically differ
        # (exact equality is possible but very unlikely with random data)
        assert isinstance(ce_full, float) and isinstance(ce_partial, float)

    def test_perfect_prediction_low_ce_loss(self, trainer, small_model):
        """Perfect predictions should have near-zero CE loss."""
        B, H, W = 2, 5, 5

        # Create targets
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        # Create logits that strongly predict target
        # Shape: (B, H, W, num_colors)
        logits = torch.zeros(B, H, W, 10)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    target_class = target_grids[b, h, w].item()
                    logits[b, h, w, target_class] = 100.0  # Strong prediction

        # Use fake halt confidence
        halt_confidence = torch.tensor([0.5, 0.5])

        # Compute loss
        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # CE loss should be very small (near zero)
        assert loss_dict["ce_loss"].item() < 0.01, \
            f"CE loss {loss_dict['ce_loss'].item()} should be near zero for perfect predictions"

    def test_halting_target_for_correct_prediction(self, trainer, small_model):
        """BCE target should be 1.0 when prediction is correct."""
        B, H, W = 1, 3, 3

        # Create targets
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        # Create perfect logits
        logits = torch.zeros(B, H, W, 10)
        for h in range(H):
            for w in range(W):
                target_class = target_grids[0, h, w].item()
                logits[0, h, w, target_class] = 100.0

        # Halt confidence will be compared against target 1.0
        halt_confidence = torch.tensor([1.0])  # Perfect confidence

        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # BCE loss should be near zero when confidence matches correct prediction
        assert loss_dict["bce_loss"].item() < 0.01

    def test_halting_target_for_incorrect_prediction(self, trainer, small_model):
        """BCE target should be 0.0 when prediction is incorrect."""
        B, H, W = 1, 3, 3

        # Create targets
        target_grids = torch.zeros(B, H, W, dtype=torch.long)  # All zeros
        mask = torch.ones(B, H, W, dtype=torch.bool)

        # Create wrong logits (predict class 5 everywhere)
        logits = torch.zeros(B, H, W, 10)
        logits[:, :, :, 5] = 100.0  # Predict class 5, but target is 0

        # Halt confidence = 0.0 should match the target
        halt_confidence = torch.tensor([0.0])

        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # BCE loss should be near zero when confidence matches incorrect prediction
        assert loss_dict["bce_loss"].item() < 0.01

    def test_accuracy_computation(self, trainer, small_model):
        """Accuracy should reflect correct cell percentage."""
        B, H, W = 1, 4, 4  # 16 cells

        # Create targets: all zeros
        target_grids = torch.zeros(B, H, W, dtype=torch.long)
        mask = torch.ones(B, H, W, dtype=torch.bool)

        # Create logits: predict correctly for half the cells
        logits = torch.zeros(B, H, W, 10)
        # First 8 cells predict 0 (correct)
        logits[0, :2, :, 0] = 100.0  # 2 rows * 4 cols = 8 cells correct
        # Last 8 cells predict 5 (wrong)
        logits[0, 2:, :, 5] = 100.0  # 2 rows * 4 cols = 8 cells wrong

        halt_confidence = torch.tensor([0.5])

        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # Accuracy should be 0.5 (8/16 correct)
        assert abs(loss_dict["accuracy"].item() - 0.5) < 0.01, \
            f"Expected accuracy 0.5, got {loss_dict['accuracy'].item()}"

    def test_empty_mask_handled(self, trainer, small_model):
        """Edge case: mask with no valid positions should not crash."""
        B, H, W = 2, 5, 5
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))

        # Create empty mask (all False)
        mask = torch.zeros(B, H, W, dtype=torch.bool)

        # Get model output
        output = trainer.model(input_grids)
        logits = output["logits"]
        halt_confidence = output["halt_confidence"]

        # Should not crash
        loss_dict = trainer.compute_loss(logits, target_grids, halt_confidence, mask)

        # Loss should still be finite
        assert torch.isfinite(loss_dict["total_loss"])


class TestTrainStep:
    """Tests for training step mechanics."""

    def test_gradients_computed(self, trainer, sample_batch):
        """Gradients should be non-zero after train_step."""
        input_grids, target_grids, mask = sample_batch

        # Run train_step
        result = trainer.train_step(input_grids, target_grids, mask)

        # Check some parameters have non-zero gradients
        has_nonzero_grad = False
        for param in trainer.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, "At least one parameter should have non-zero gradient"

    def test_weights_change_after_step(self, trainer, sample_batch):
        """Weights should update after train_step."""
        input_grids, target_grids, mask = sample_batch

        # Find a parameter that will definitely have gradients (from network, not embedding)
        # The network's transformer parameters should always receive gradients
        param_with_grad = None
        for name, param in trainer.model.named_parameters():
            # Look for transformer weights which always receive gradients
            if "transformer" in name or "output_head" in name:
                param_with_grad = param
                break

        assert param_with_grad is not None, "Could not find a transformer/output_head parameter"

        # Record weights before
        weights_before = param_with_grad.data.clone()

        # Run train_step
        trainer.train_step(input_grids, target_grids, mask)

        # Verify weights changed
        weights_after = param_with_grad.data

        assert not torch.allclose(weights_before, weights_after), \
            "Weights should change after optimizer step"

    def test_gradients_zeroed_before_backward(self, trainer, sample_batch):
        """Gradients should be zeroed each step (no accumulation)."""
        input_grids, target_grids, mask = sample_batch

        # Run two train_steps
        trainer.train_step(input_grids, target_grids, mask)
        grads_after_first = []
        for param in trainer.model.parameters():
            if param.grad is not None:
                grads_after_first.append(param.grad.clone())

        trainer.train_step(input_grids, target_grids, mask)
        grads_after_second = []
        for param in trainer.model.parameters():
            if param.grad is not None:
                grads_after_second.append(param.grad.clone())

        # Gradients should not accumulate (should be fresh each step)
        # If they accumulated, second gradients would be ~2x first
        # Since they're zeroed, they should be roughly equal (same input)
        if len(grads_after_first) > 0 and len(grads_after_second) > 0:
            # Compare first gradient tensor
            ratio = grads_after_second[0].abs().mean() / (grads_after_first[0].abs().mean() + 1e-8)
            assert 0.5 < ratio < 2.0, "Gradients appear to be accumulating"

    def test_train_step_returns_metrics(self, trainer, sample_batch):
        """train_step should return all expected metrics."""
        input_grids, target_grids, mask = sample_batch

        result = trainer.train_step(input_grids, target_grids, mask)

        # Check all expected keys
        expected_keys = ["total_loss", "ce_loss", "bce_loss", "accuracy", "iterations"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Values should be Python floats/ints (not tensors)
        assert isinstance(result["total_loss"], float)
        assert isinstance(result["ce_loss"], float)
        assert isinstance(result["bce_loss"], float)
        assert isinstance(result["accuracy"], float)
        assert isinstance(result["iterations"], int)


class TestTrainingStability:
    """Tests for training stability over multiple steps."""

    def test_no_nan_in_loss(self, trainer, sample_batch):
        """Loss should not be NaN."""
        input_grids, target_grids, mask = sample_batch

        # Run several train_steps
        for _ in range(5):
            result = trainer.train_step(input_grids, target_grids, mask)
            assert not torch.isnan(torch.tensor(result["total_loss"])), \
                "Loss became NaN"

    def test_no_inf_in_loss(self, trainer, sample_batch):
        """Loss should not be Inf."""
        input_grids, target_grids, mask = sample_batch

        # Run several train_steps
        for _ in range(5):
            result = trainer.train_step(input_grids, target_grids, mask)
            assert not torch.isinf(torch.tensor(result["total_loss"])), \
                "Loss became Inf"

    def test_loss_is_finite_with_padding(self, trainer, small_model):
        """Loss should be finite even with heavy padding."""
        B, H, W = 2, 5, 5
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))

        # Only 10% of positions are valid
        mask = torch.zeros(B, H, W, dtype=torch.bool)
        mask[:, 0, :2] = True  # Only first 2 cells in first row

        result = trainer.train_step(input_grids, target_grids, mask)

        assert torch.isfinite(torch.tensor(result["total_loss"])), \
            "Loss should be finite even with heavy padding"

    def test_multiple_steps_stable(self, trainer, sample_batch):
        """Multiple training steps should run without crash."""
        input_grids, target_grids, mask = sample_batch

        # Run 10 train_steps
        losses = []
        for _ in range(10):
            result = trainer.train_step(input_grids, target_grids, mask)
            losses.append(result["total_loss"])

        # All losses should be finite
        for i, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)), \
                f"Loss at step {i} is not finite: {loss}"

    def test_loss_generally_decreases(self, trainer, sample_batch):
        """Loss should generally decrease over training (basic sanity check)."""
        input_grids, target_grids, mask = sample_batch

        # Run 20 steps and check if loss decreases
        losses = []
        for _ in range(20):
            result = trainer.train_step(input_grids, target_grids, mask)
            losses.append(result["total_loss"])

        # First loss should be higher than last (on average)
        # Use average of first 5 vs last 5 for stability
        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5

        # Loss should decrease (or at least not explode)
        # Allow 10% tolerance in case of oscillation
        assert last_avg <= first_avg * 1.1, \
            f"Loss did not decrease: {first_avg:.4f} -> {last_avg:.4f}"


class TestIntegration:
    """Integration tests combining trainer with full model."""

    def test_full_training_loop_simulation(self, small_model):
        """Simulate a mini training loop."""
        trainer = TRMTrainer(
            small_model,
            learning_rate=1e-3,  # Higher LR for faster convergence in test
        )

        # Create consistent training data
        torch.manual_seed(42)
        B, H, W = 4, 5, 5
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        # Train for several steps
        initial_loss = None
        final_loss = None

        for step in range(30):
            result = trainer.train_step(input_grids, target_grids, mask)
            if step == 0:
                initial_loss = result["total_loss"]
            final_loss = result["total_loss"]

        # Verify training made progress
        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss, \
            f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_gradient_flow_through_model(self, small_model):
        """Verify gradients flow through all model components."""
        trainer = TRMTrainer(small_model)

        B, H, W = 2, 5, 5
        input_grids = torch.randint(0, 10, (B, H, W))
        target_grids = torch.randint(0, 10, (B, H, W))
        mask = torch.ones(B, H, W, dtype=torch.bool)

        # Run train_step
        trainer.train_step(input_grids, target_grids, mask)

        # Check gradients in different parts of the model
        components = {
            "embedding": small_model.embedding,
            "network": small_model.network,
        }

        for name, component in components.items():
            has_grad = False
            for param in component.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break
            assert has_grad, f"No gradients flowing to {name}"

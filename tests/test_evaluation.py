"""Tests for exact-match accuracy evaluation (TDD RED phase)."""
import torch
import pytest

from src.trm.evaluation.evaluator import compute_exact_match_accuracy, evaluate_batch


class TestComputeExactMatchAccuracy:
    """Test exact-match accuracy computation."""

    def test_perfect_match_returns_one(self):
        """When all pixels match, accuracy should be 1.0."""
        # Arrange: 2x2 grid with all correct predictions
        logits = torch.zeros(1, 2, 2, 10)  # (B=1, H=2, W=2, num_colors=10)
        logits[0, :, :, 5] = 10.0  # Highest logit for color 5

        targets = torch.full((1, 2, 2), 5)  # All pixels are color 5
        mask = torch.ones(1, 2, 2, dtype=torch.bool)  # All positions valid

        # Act
        accuracy = compute_exact_match_accuracy(logits, targets, mask)

        # Assert
        assert accuracy.shape == (1,), "Should return per-sample accuracy"
        assert accuracy[0].item() == 1.0, "Perfect match should return 1.0"

    def test_single_pixel_mismatch_returns_zero(self):
        """When ANY pixel differs, accuracy should be 0.0."""
        # Arrange: 2x2 grid with one wrong pixel
        logits = torch.zeros(1, 2, 2, 10)
        logits[0, 0, 0, 5] = 10.0  # First pixel predicts 5
        logits[0, 0, 1, 5] = 10.0  # Second pixel predicts 5
        logits[0, 1, 0, 5] = 10.0  # Third pixel predicts 5
        logits[0, 1, 1, 3] = 10.0  # Fourth pixel predicts 3 (WRONG)

        targets = torch.full((1, 2, 2), 5)  # All pixels should be 5
        mask = torch.ones(1, 2, 2, dtype=torch.bool)

        # Act
        accuracy = compute_exact_match_accuracy(logits, targets, mask)

        # Assert
        assert accuracy[0].item() == 0.0, "Any mismatch should return 0.0"

    def test_padding_excluded_from_comparison(self):
        """Padded positions (mask=False) should not affect accuracy."""
        # Arrange: 2x2 grid with padding
        logits = torch.zeros(1, 2, 2, 10)
        logits[0, 0, 0, 5] = 10.0  # Valid pixel predicts 5
        logits[0, 0, 1, 9] = 10.0  # Padding pixel predicts 9 (ignored)
        logits[0, 1, 0, 9] = 10.0  # Padding pixel predicts 9 (ignored)
        logits[0, 1, 1, 9] = 10.0  # Padding pixel predicts 9 (ignored)

        targets = torch.full((1, 2, 2), 5)  # All pixels are 5
        mask = torch.tensor([[[True, False], [False, False]]])  # Only (0,0) valid

        # Act
        accuracy = compute_exact_match_accuracy(logits, targets, mask)

        # Assert
        assert accuracy[0].item() == 1.0, "Padding pixels should be excluded"

    def test_empty_mask_returns_one(self):
        """When mask is all False (no valid positions), return 1.0 (vacuously true)."""
        # Arrange: No valid positions
        logits = torch.zeros(1, 2, 2, 10)
        targets = torch.zeros(1, 2, 2)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)  # All padding

        # Act
        accuracy = compute_exact_match_accuracy(logits, targets, mask)

        # Assert
        assert accuracy[0].item() == 1.0, "Empty mask should return 1.0 (vacuously true)"

    def test_batch_processing(self):
        """Should handle multiple samples independently."""
        # Arrange: Batch of 3 samples
        logits = torch.zeros(3, 2, 2, 10)

        # Sample 0: All correct
        logits[0, :, :, 5] = 10.0

        # Sample 1: One wrong pixel
        logits[1, 0, 0, 5] = 10.0
        logits[1, 0, 1, 5] = 10.0
        logits[1, 1, 0, 5] = 10.0
        logits[1, 1, 1, 3] = 10.0  # Wrong

        # Sample 2: All correct
        logits[2, :, :, 7] = 10.0

        targets = torch.tensor([
            [[5, 5], [5, 5]],  # Sample 0 target
            [[5, 5], [5, 5]],  # Sample 1 target
            [[7, 7], [7, 7]],  # Sample 2 target
        ])
        mask = torch.ones(3, 2, 2, dtype=torch.bool)

        # Act
        accuracy = compute_exact_match_accuracy(logits, targets, mask)

        # Assert
        assert accuracy.shape == (3,), "Should return per-sample accuracy"
        assert accuracy[0].item() == 1.0, "Sample 0 should be correct"
        assert accuracy[1].item() == 0.0, "Sample 1 should be incorrect"
        assert accuracy[2].item() == 1.0, "Sample 2 should be correct"

    def test_different_grid_sizes_per_sample(self):
        """Should handle different amounts of padding per sample."""
        # Arrange: Batch with different valid regions
        logits = torch.zeros(2, 3, 3, 10)

        # Sample 0: 2x2 valid region (bottom-right is padding)
        logits[0, 0:2, 0:2, 5] = 10.0

        # Sample 1: 3x1 valid region (rightmost columns are padding)
        logits[1, :, 0, 7] = 10.0

        targets = torch.tensor([
            [[5, 5, 0], [5, 5, 0], [0, 0, 0]],  # Sample 0
            [[7, 0, 0], [7, 0, 0], [7, 0, 0]],  # Sample 1
        ])

        mask = torch.tensor([
            [[True, True, False], [True, True, False], [False, False, False]],
            [[True, False, False], [True, False, False], [True, False, False]],
        ])

        # Act
        accuracy = compute_exact_match_accuracy(logits, targets, mask)

        # Assert
        assert accuracy[0].item() == 1.0, "Sample 0: all valid pixels match"
        assert accuracy[1].item() == 1.0, "Sample 1: all valid pixels match"

    def test_argmax_conversion(self):
        """Should convert logits to predictions via argmax on last dimension."""
        # Arrange: Logits with different highest values
        logits = torch.zeros(1, 2, 2, 10)
        logits[0, 0, 0, 3] = 5.0  # Highest at index 3
        logits[0, 0, 1, 7] = 8.0  # Highest at index 7
        logits[0, 1, 0, 0] = 2.0  # Highest at index 0
        logits[0, 1, 1, 9] = 1.0  # Highest at index 9

        targets = torch.tensor([[[3, 7], [0, 9]]])
        mask = torch.ones(1, 2, 2, dtype=torch.bool)

        # Act
        accuracy = compute_exact_match_accuracy(logits, targets, mask)

        # Assert
        assert accuracy[0].item() == 1.0, "Should use argmax to get predictions"


class TestEvaluateBatch:
    """Test batch evaluation with model inference."""

    def test_evaluate_batch_returns_correct_structure(self):
        """Should return predictions, per-sample accuracy, and mean accuracy."""
        # Arrange: Simple mock model
        class MockModel:
            def __call__(self, x):
                B, H, W = x.shape
                logits = torch.zeros(B, H, W, 10)
                logits[:, :, :, 5] = 10.0  # Predict 5 everywhere
                halt_conf = torch.ones(B) * 0.9
                return {"logits": logits, "halt_confidence": halt_conf}

        model = MockModel()
        input_grids = torch.zeros(2, 3, 3)
        target_grids = torch.full((2, 3, 3), 5)  # All 5s
        mask = torch.ones(2, 3, 3, dtype=torch.bool)

        # Act
        result = evaluate_batch(model, input_grids, target_grids, mask)

        # Assert
        assert "predictions" in result, "Should return predictions"
        assert "accuracy" in result, "Should return per-sample accuracy"
        assert "mean_accuracy" in result, "Should return mean accuracy"

        assert result["predictions"].shape == (2, 3, 3), "Predictions shape"
        assert result["accuracy"].shape == (2,), "Per-sample accuracy shape"
        assert isinstance(result["mean_accuracy"], float), "Mean should be float"
        assert result["mean_accuracy"] == 1.0, "All predictions correct"

    def test_evaluate_batch_uses_no_grad(self):
        """Should run inference without gradient tracking."""
        # Arrange
        class MockModel:
            def __init__(self):
                self.param = torch.nn.Parameter(torch.ones(1))

            def __call__(self, x):
                B, H, W = x.shape
                logits = torch.zeros(B, H, W, 10)
                logits[:, :, :, 5] = self.param * 10.0
                halt_conf = torch.ones(B) * 0.9
                return {"logits": logits, "halt_confidence": halt_conf}

        model = MockModel()
        input_grids = torch.zeros(1, 2, 2)
        target_grids = torch.zeros(1, 2, 2)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)

        # Act
        with torch.no_grad():
            result = evaluate_batch(model, input_grids, target_grids, mask)

        # Assert
        assert not result["predictions"].requires_grad, "Should not track gradients"

    def test_evaluate_batch_computes_correct_mean(self):
        """Mean accuracy should be average of per-sample accuracies."""
        # Arrange: Model that predicts different things
        class MockModel:
            def __call__(self, x):
                B, H, W = x.shape
                logits = torch.zeros(B, H, W, 10)
                # Sample 0: predict 5
                logits[0, :, :, 5] = 10.0
                # Sample 1: predict 3
                logits[1, :, :, 3] = 10.0
                # Sample 2: predict 7
                logits[2, :, :, 7] = 10.0
                halt_conf = torch.ones(B) * 0.9
                return {"logits": logits, "halt_confidence": halt_conf}

        model = MockModel()
        input_grids = torch.zeros(3, 2, 2)
        target_grids = torch.tensor([
            [[5, 5], [5, 5]],  # Matches sample 0
            [[5, 5], [5, 5]],  # Doesn't match sample 1
            [[7, 7], [7, 7]],  # Matches sample 2
        ])
        mask = torch.ones(3, 2, 2, dtype=torch.bool)

        # Act
        result = evaluate_batch(model, input_grids, target_grids, mask)

        # Assert
        expected_mean = (1.0 + 0.0 + 1.0) / 3.0
        assert abs(result["mean_accuracy"] - expected_mean) < 1e-6, "Mean should be 2/3"
        assert result["accuracy"][0].item() == 1.0
        assert result["accuracy"][1].item() == 0.0
        assert result["accuracy"][2].item() == 1.0

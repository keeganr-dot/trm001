"""Exact-match accuracy evaluation for ARC-AGI tasks."""
import torch
import torch.nn as nn


def compute_exact_match_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact-match accuracy for ARC-AGI evaluation.

    Exact-match means ALL pixels must match perfectly. Returns 1.0 only when
    every non-padded pixel in the sample matches the target, 0.0 otherwise.

    This is the evaluation metric specified in EVAL-01: pixel-perfect matching
    (100% or 0% per task). No partial credit is given.

    Args:
        logits: Predicted logits of shape (B, H, W, num_colors)
        targets: Target grid of shape (B, H, W) with values 0-9
        mask: Boolean mask of shape (B, H, W), True for valid positions

    Returns:
        Tensor of shape (B,) where each element is 1.0 (perfect match)
        or 0.0 (any mismatch). Empty masks return 1.0 (vacuously true).

    Example:
        >>> logits = torch.zeros(2, 3, 3, 10)
        >>> logits[0, :, :, 5] = 10.0  # Sample 0 predicts all 5s
        >>> logits[1, 0, 0, 3] = 10.0  # Sample 1 has one wrong pixel
        >>> logits[1, 0, 1:, 5] = 10.0
        >>> logits[1, 1:, :, 5] = 10.0
        >>> targets = torch.full((2, 3, 3), 5)
        >>> mask = torch.ones(2, 3, 3, dtype=torch.bool)
        >>> accuracy = compute_exact_match_accuracy(logits, targets, mask)
        >>> accuracy
        tensor([1.0, 0.0])  # Sample 0 perfect, sample 1 has mismatch
    """
    B, H, W, num_colors = logits.shape

    # Convert logits to predictions via argmax on last dimension
    predictions = torch.argmax(logits, dim=-1)  # (B, H, W)

    # Compute per-sample exact-match accuracy
    accuracies = []
    for b in range(B):
        sample_mask = mask[b]  # (H, W)

        # Handle empty mask (no valid positions) - vacuously true
        if sample_mask.sum() == 0:
            accuracies.append(1.0)
            continue

        # Extract valid positions for this sample
        sample_preds = predictions[b][sample_mask]  # (N_valid,)
        sample_targets = targets[b][sample_mask]  # (N_valid,)

        # Check if ALL predictions match targets
        is_perfect = (sample_preds == sample_targets).all().float().item()
        accuracies.append(is_perfect)

    # Return as tensor
    return torch.tensor(accuracies, device=logits.device, dtype=torch.float32)


def evaluate_batch(
    model: nn.Module,
    input_grids: torch.Tensor,
    target_grids: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Evaluate model on a batch with exact-match accuracy.

    Runs model inference in no_grad mode for efficiency, then computes
    exact-match accuracy for each sample.

    Args:
        model: Model to evaluate (must return dict with "logits" key)
        input_grids: Input grid tensor of shape (B, H, W) with values 0-9 or -1
        target_grids: Target grid tensor of shape (B, H, W) with values 0-9
        mask: Boolean mask of shape (B, H, W), True for valid positions

    Returns:
        Dictionary containing:
            - predictions: Predicted grid of shape (B, H, W) with values 0-9
            - accuracy: Per-sample exact-match accuracy of shape (B,)
            - mean_accuracy: Mean accuracy across batch (float)

    Example:
        >>> model = RecursiveRefinement(...)
        >>> input_grids = torch.zeros(4, 10, 10)
        >>> target_grids = torch.zeros(4, 10, 10)
        >>> mask = torch.ones(4, 10, 10, dtype=torch.bool)
        >>> result = evaluate_batch(model, input_grids, target_grids, mask)
        >>> result["mean_accuracy"]
        0.75  # 3 out of 4 samples correct
    """
    # Run inference without gradient tracking
    with torch.no_grad():
        output = model(input_grids)
        logits = output["logits"]  # (B, H, W, num_colors)

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=-1)  # (B, H, W)

        # Compute exact-match accuracy
        accuracy = compute_exact_match_accuracy(logits, target_grids, mask)

        # Compute mean accuracy
        mean_accuracy = accuracy.mean().item()

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "mean_accuracy": mean_accuracy,
    }

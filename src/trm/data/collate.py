"""Custom collate function for ARC-AGI variable-size grids."""
import torch
from typing import List, Dict, Any


PAD_VALUE = -1  # Distinguishable from valid colors 0-9


def arc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for ARC-AGI tasks with variable grid sizes.

    For training, we typically want to batch individual train pairs, not entire tasks.
    This collate function handles the task-level batching; pair-level batching
    can be done by flattening train pairs from multiple tasks.

    Args:
        batch: List of task dicts from ARCDataset.__getitem__

    Returns:
        Dict with:
            - task_ids: List[str]
            - train_inputs: Tensor (B, max_pairs, max_H, max_W) padded
            - train_outputs: Tensor (B, max_pairs, max_H, max_W) padded
            - test_inputs: Tensor (B, max_pairs, max_H, max_W) padded
            - test_outputs: Tensor (B, max_pairs, max_H, max_W) padded
            - train_input_masks: Tensor (B, max_pairs, max_H, max_W) bool - True for valid input positions
            - train_output_masks: Tensor (B, max_pairs, max_H, max_W) bool - True for valid output positions
            - test_input_masks: Tensor (B, max_pairs, max_H, max_W) bool - True for valid input positions
            - test_output_masks: Tensor (B, max_pairs, max_H, max_W) bool - True for valid output positions
            - num_train_pairs: Tensor (B,) - actual train pair count per task
            - num_test_pairs: Tensor (B,) - actual test pair count per task
    """
    batch_size = len(batch)
    task_ids = [item["task_id"] for item in batch]

    # Find max dimensions across all grids in batch
    max_train_pairs = max(len(item["train_inputs"]) for item in batch)
    max_test_pairs = max(len(item["test_inputs"]) for item in batch)

    # Find max H and W across all train grids (inputs and outputs)
    all_train_grids = []
    for item in batch:
        all_train_grids.extend(item["train_inputs"])
        all_train_grids.extend(item["train_outputs"])
    max_train_h = max(g.shape[0] for g in all_train_grids)
    max_train_w = max(g.shape[1] for g in all_train_grids)

    # Find max H and W across all test grids (inputs and outputs)
    all_test_grids = []
    for item in batch:
        all_test_grids.extend(item["test_inputs"])
        all_test_grids.extend(item["test_outputs"])
    max_test_h = max(g.shape[0] for g in all_test_grids)
    max_test_w = max(g.shape[1] for g in all_test_grids)

    # Initialize output tensors with PAD_VALUE
    train_inputs = torch.full(
        (batch_size, max_train_pairs, max_train_h, max_train_w),
        PAD_VALUE,
        dtype=torch.long,
    )
    train_outputs = torch.full(
        (batch_size, max_train_pairs, max_train_h, max_train_w),
        PAD_VALUE,
        dtype=torch.long,
    )
    test_inputs = torch.full(
        (batch_size, max_test_pairs, max_test_h, max_test_w),
        PAD_VALUE,
        dtype=torch.long,
    )
    test_outputs = torch.full(
        (batch_size, max_test_pairs, max_test_h, max_test_w),
        PAD_VALUE,
        dtype=torch.long,
    )

    # Initialize separate masks for inputs and outputs (False = padded, True = valid)
    train_input_masks = torch.zeros(
        (batch_size, max_train_pairs, max_train_h, max_train_w), dtype=torch.bool
    )
    train_output_masks = torch.zeros(
        (batch_size, max_train_pairs, max_train_h, max_train_w), dtype=torch.bool
    )
    test_input_masks = torch.zeros(
        (batch_size, max_test_pairs, max_test_h, max_test_w), dtype=torch.bool
    )
    test_output_masks = torch.zeros(
        (batch_size, max_test_pairs, max_test_h, max_test_w), dtype=torch.bool
    )

    # Track actual pair counts
    num_train_pairs = torch.zeros(batch_size, dtype=torch.long)
    num_test_pairs = torch.zeros(batch_size, dtype=torch.long)

    # Fill in actual data
    for b, item in enumerate(batch):
        # Train pairs
        num_train = len(item["train_inputs"])
        num_train_pairs[b] = num_train
        for p in range(num_train):
            inp = item["train_inputs"][p]
            out = item["train_outputs"][p]
            h_in, w_in = inp.shape
            h_out, w_out = out.shape

            train_inputs[b, p, :h_in, :w_in] = inp
            train_input_masks[b, p, :h_in, :w_in] = True

            train_outputs[b, p, :h_out, :w_out] = out
            train_output_masks[b, p, :h_out, :w_out] = True

        # Test pairs
        num_test = len(item["test_inputs"])
        num_test_pairs[b] = num_test
        for p in range(num_test):
            inp = item["test_inputs"][p]
            out = item["test_outputs"][p]
            h_in, w_in = inp.shape
            h_out, w_out = out.shape

            test_inputs[b, p, :h_in, :w_in] = inp
            test_input_masks[b, p, :h_in, :w_in] = True

            test_outputs[b, p, :h_out, :w_out] = out
            test_output_masks[b, p, :h_out, :w_out] = True

    return {
        "task_ids": task_ids,
        "train_inputs": train_inputs,
        "train_outputs": train_outputs,
        "test_inputs": test_inputs,
        "test_outputs": test_outputs,
        "train_input_masks": train_input_masks,
        "train_output_masks": train_output_masks,
        "test_input_masks": test_input_masks,
        "test_output_masks": test_output_masks,
        # Keep combined masks for backward compatibility (union of input/output)
        "train_masks": train_input_masks | train_output_masks,
        "test_masks": test_input_masks | test_output_masks,
        "num_train_pairs": num_train_pairs,
        "num_test_pairs": num_test_pairs,
    }

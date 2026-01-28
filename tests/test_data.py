"""Comprehensive tests for ARC-AGI data pipeline.

Tests cover all Phase 2 success criteria:
- SC1: Official ARC-AGI dataset loads (400 training, 400 evaluation)
- SC2: DataLoader produces batches with variable grid sizes correctly padded
- SC3: Each task's train/test pairs accessible with input/output grids as tensors
- SC4: Grid tensors correctly represent 10-color ARC format (values 0-9)
"""
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.trm.data import ARCDataset, arc_collate_fn, PAD_VALUE


class TestARCDataset:
    """Tests for ARC-AGI dataset loading."""

    @pytest.fixture
    def dataset(self):
        return ARCDataset("data", split="training")

    def test_dataset_size(self, dataset):
        """SC1: Official ARC-AGI dataset loads - 400 training tasks."""
        assert len(dataset) == 400

    def test_evaluation_split(self):
        """SC1: Evaluation split also loads correctly."""
        ds = ARCDataset("data", split="evaluation")
        assert len(ds) == 400

    def test_invalid_split_raises(self):
        """Invalid split raises ValueError."""
        with pytest.raises(ValueError):
            ARCDataset("data", split="invalid")

    def test_task_structure(self, dataset):
        """SC3: Each task has train/test pairs accessible."""
        task = dataset[0]
        assert "task_id" in task
        assert "train_inputs" in task
        assert "train_outputs" in task
        assert "test_inputs" in task
        assert "test_outputs" in task
        # Train pairs typically 2-5, test pairs typically 1
        assert len(task["train_inputs"]) >= 1
        assert len(task["test_inputs"]) >= 1

    def test_task_id_is_string(self, dataset):
        """Task ID is a string (filename without extension)."""
        task = dataset[0]
        assert isinstance(task["task_id"], str)
        assert len(task["task_id"]) > 0

    def test_tensor_dtype(self, dataset):
        """SC4: Grid tensors are correct dtype for embedding lookup."""
        task = dataset[0]
        assert task["train_inputs"][0].dtype == torch.long
        assert task["train_outputs"][0].dtype == torch.long
        assert task["test_inputs"][0].dtype == torch.long
        assert task["test_outputs"][0].dtype == torch.long

    def test_value_range(self, dataset):
        """SC4: Grid values are 0-9 (10-color ARC format)."""
        task = dataset[0]
        grid = task["train_inputs"][0]
        assert grid.min() >= 0
        assert grid.max() <= 9

    def test_value_range_across_tasks(self, dataset):
        """All tasks have values in valid 0-9 range."""
        for i in range(min(50, len(dataset))):
            task = dataset[i]
            for grids in [task["train_inputs"], task["train_outputs"],
                          task["test_inputs"], task["test_outputs"]]:
                for grid in grids:
                    assert grid.min() >= 0, f"Task {i} has value < 0"
                    assert grid.max() <= 9, f"Task {i} has value > 9"

    def test_variable_grid_sizes(self, dataset):
        """SC2: Grids have variable sizes (1x1 to 30x30)."""
        sizes = set()
        for i in range(min(50, len(dataset))):
            task = dataset[i]
            for grid in task["train_inputs"]:
                sizes.add(grid.shape)
        # Should have multiple different sizes
        assert len(sizes) > 1, "Expected variable grid sizes"

    def test_grid_dimensions(self, dataset):
        """Each grid is 2D tensor with height and width."""
        task = dataset[0]
        grid = task["train_inputs"][0]
        assert grid.dim() == 2
        assert grid.shape[0] > 0  # height
        assert grid.shape[1] > 0  # width

    def test_train_test_pair_counts_match(self, dataset):
        """Train inputs/outputs have same count, test inputs/outputs have same count."""
        for i in range(min(20, len(dataset))):
            task = dataset[i]
            assert len(task["train_inputs"]) == len(task["train_outputs"])
            assert len(task["test_inputs"]) == len(task["test_outputs"])


class TestDataLoader:
    """Tests for batched data loading with padding."""

    @pytest.fixture
    def dataloader(self):
        ds = ARCDataset("data", split="training")
        return DataLoader(ds, batch_size=4, collate_fn=arc_collate_fn)

    def test_batch_dimensions(self, dataloader):
        """SC2: DataLoader produces batches with consistent dimensions."""
        batch = next(iter(dataloader))
        assert batch["train_inputs"].dim() == 4  # (B, pairs, H, W)
        assert batch["train_outputs"].dim() == 4
        assert batch["test_inputs"].dim() == 4
        assert batch["test_outputs"].dim() == 4
        # Separate masks for inputs and outputs
        assert batch["train_input_masks"].shape == batch["train_inputs"].shape
        assert batch["train_output_masks"].shape == batch["train_outputs"].shape
        assert batch["test_input_masks"].shape == batch["test_inputs"].shape
        assert batch["test_output_masks"].shape == batch["test_outputs"].shape
        # Combined masks also available (union of input/output extents)
        assert batch["train_masks"].shape == batch["train_inputs"].shape
        assert batch["test_masks"].shape == batch["test_inputs"].shape

    def test_batch_size(self, dataloader):
        """Batch size matches expected value."""
        batch = next(iter(dataloader))
        assert len(batch["task_ids"]) == 4
        assert batch["train_inputs"].shape[0] == 4

    def test_padding_value_inputs(self, dataloader):
        """Input padding uses distinguishable value, valid inputs are 0-9."""
        batch = next(iter(dataloader))
        # Valid input positions (from input mask) should have 0-9
        valid_inputs = batch["train_inputs"][batch["train_input_masks"]]
        assert valid_inputs.min() >= 0
        assert valid_inputs.max() <= 9

    def test_padding_value_outputs(self, dataloader):
        """Output padding uses distinguishable value, valid outputs are 0-9."""
        batch = next(iter(dataloader))
        # Valid output positions (from output mask) should have 0-9
        valid_outputs = batch["train_outputs"][batch["train_output_masks"]]
        assert valid_outputs.min() >= 0
        assert valid_outputs.max() <= 9

    def test_input_masks_correct(self, dataloader):
        """Input masks correctly identify valid vs padded positions."""
        batch = next(iter(dataloader))
        # Where input mask is False, input value should be PAD_VALUE
        invalid = batch["train_inputs"][~batch["train_input_masks"]]
        if len(invalid) > 0:
            assert (invalid == PAD_VALUE).all()

    def test_output_masks_correct(self, dataloader):
        """Output masks correctly identify valid vs padded positions."""
        batch = next(iter(dataloader))
        # Where output mask is False, output value should be PAD_VALUE
        invalid = batch["train_outputs"][~batch["train_output_masks"]]
        if len(invalid) > 0:
            assert (invalid == PAD_VALUE).all()

    def test_combined_mask_is_union(self, dataloader):
        """Combined mask is union of input and output masks."""
        batch = next(iter(dataloader))
        expected = batch["train_input_masks"] | batch["train_output_masks"]
        assert (batch["train_masks"] == expected).all()

    def test_num_pairs_tensor(self, dataloader):
        """Pair count tensors have correct shape and values."""
        batch = next(iter(dataloader))
        assert batch["num_train_pairs"].shape == (4,)
        assert batch["num_test_pairs"].shape == (4,)
        assert batch["num_train_pairs"].dtype == torch.long
        assert (batch["num_train_pairs"] >= 1).all()
        assert (batch["num_test_pairs"] >= 1).all()

    def test_task_ids_list(self, dataloader):
        """Task IDs are returned as list of strings."""
        batch = next(iter(dataloader))
        assert isinstance(batch["task_ids"], list)
        assert len(batch["task_ids"]) == 4
        assert all(isinstance(tid, str) for tid in batch["task_ids"])

    def test_full_iteration(self, dataloader):
        """Can iterate through entire dataset without errors."""
        count = 0
        for batch in dataloader:
            count += len(batch["task_ids"])
        assert count == 400


class TestPadValue:
    """Tests for PAD_VALUE constant."""

    def test_pad_value_distinguishable(self):
        """PAD_VALUE is distinguishable from valid colors 0-9."""
        assert PAD_VALUE < 0 or PAD_VALUE > 9

    def test_pad_value_is_minus_one(self):
        """PAD_VALUE is specifically -1."""
        assert PAD_VALUE == -1


class TestIntegration:
    """End-to-end integration tests."""

    def test_training_workflow(self):
        """Simulated training loop processes batches correctly."""
        ds = ARCDataset("data", split="training")
        loader = DataLoader(ds, batch_size=8, collate_fn=arc_collate_fn, shuffle=True)

        # Process a few batches as training would
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            # Verify shapes are consistent
            B = len(batch["task_ids"])
            assert batch["train_inputs"].shape[0] == B
            assert batch["train_input_masks"].shape[0] == B
            assert batch["train_output_masks"].shape[0] == B
            assert batch["num_train_pairs"].shape == (B,)

    def test_evaluation_workflow(self):
        """Evaluation split loads and batches correctly."""
        ds = ARCDataset("data", split="evaluation")
        loader = DataLoader(ds, batch_size=4, collate_fn=arc_collate_fn)

        batch = next(iter(loader))
        assert len(batch["task_ids"]) == 4
        assert batch["train_inputs"].dim() == 4

    def test_different_batch_sizes(self):
        """Collate function handles various batch sizes."""
        ds = ARCDataset("data", split="training")

        for batch_size in [1, 2, 4, 8, 16]:
            loader = DataLoader(ds, batch_size=batch_size, collate_fn=arc_collate_fn)
            batch = next(iter(loader))
            assert batch["train_inputs"].shape[0] == batch_size

    def test_mask_enables_loss_computation(self):
        """Masks can be used to compute loss only on valid output positions."""
        ds = ARCDataset("data", split="training")
        loader = DataLoader(ds, batch_size=4, collate_fn=arc_collate_fn)
        batch = next(iter(loader))

        # Use output mask for loss computation (outputs are predictions)
        outputs = batch["train_outputs"]
        masks = batch["train_output_masks"]

        # Valid outputs for loss
        valid_outputs = outputs[masks]
        assert valid_outputs.min() >= 0
        assert valid_outputs.max() <= 9

        # Can compute cross-entropy loss on valid positions
        num_valid = masks.sum().item()
        assert num_valid > 0

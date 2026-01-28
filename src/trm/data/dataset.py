"""ARC-AGI Dataset implementation."""
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class ARCDataset(Dataset):
    """PyTorch Dataset for ARC-AGI tasks.

    Each ARC-AGI task consists of:
    - Training examples: input/output grid pairs demonstrating the pattern
    - Test examples: input grids requiring output prediction

    All grids use values 0-9 representing colors.
    """

    def __init__(self, data_dir: str, split: str = "training"):
        """Initialize ARC-AGI dataset.

        Args:
            data_dir: Path to data/ directory containing training/ and evaluation/
            split: Either "training" or "evaluation"

        Raises:
            ValueError: If split is not "training" or "evaluation"
            FileNotFoundError: If split directory doesn't exist
        """
        if split not in ("training", "evaluation"):
            raise ValueError(f"split must be 'training' or 'evaluation', got {split}")

        self.data_dir = Path(data_dir)
        self.split = split
        split_dir = self.data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Load all tasks eagerly (small dataset, ~800 tasks total)
        json_files = sorted(split_dir.glob("*.json"))
        self.tasks = []

        for json_file in json_files:
            task_id = json_file.stem  # Filename without .json extension
            with open(json_file, "r") as f:
                task_data = json.load(f)
            self.tasks.append((task_id, task_data))

    def __len__(self) -> int:
        """Return number of tasks in dataset."""
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict:
        """Get a single task by index.

        Args:
            idx: Task index

        Returns:
            Dictionary containing:
                - task_id: str (filename without .json)
                - train_inputs: List[Tensor] - each shape (H, W), dtype=torch.long
                - train_outputs: List[Tensor] - each shape (H, W), dtype=torch.long
                - test_inputs: List[Tensor] - each shape (H, W), dtype=torch.long
                - test_outputs: List[Tensor] - each shape (H, W), dtype=torch.long

            Grid values are 0-9 (10 colors). Grids have variable sizes.
        """
        task_id, task_data = self.tasks[idx]

        # Convert train pairs
        train_inputs = [
            torch.tensor(pair["input"], dtype=torch.long)
            for pair in task_data["train"]
        ]
        train_outputs = [
            torch.tensor(pair["output"], dtype=torch.long)
            for pair in task_data["train"]
        ]

        # Convert test pairs
        test_inputs = [
            torch.tensor(pair["input"], dtype=torch.long)
            for pair in task_data["test"]
        ]
        test_outputs = [
            torch.tensor(pair["output"], dtype=torch.long)
            for pair in task_data["test"]
        ]

        return {
            "task_id": task_id,
            "train_inputs": train_inputs,
            "train_outputs": train_outputs,
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
        }

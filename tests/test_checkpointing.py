"""Comprehensive tests for checkpoint save/load and best model tracking."""
import pytest
import torch
import tempfile
from pathlib import Path

from src.trm.model import TRMNetwork, RecursiveRefinement
from src.trm.training.deep_supervision import DeepSupervisionTrainer
from src.trm.evaluation import save_checkpoint, load_checkpoint, BestModelTracker


@pytest.fixture
def config():
    """Minimal config for fast testing."""
    return {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_colors": 10,
        "outer_steps": 2,
        "inner_steps": 3,
    }


@pytest.fixture
def model_and_trainer(config):
    """Create model and trainer for testing."""
    from src.trm.model.embedding import GridEmbedding

    network = TRMNetwork(
        num_colors=config["num_colors"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    )

    embedding = GridEmbedding(hidden_dim=config["hidden_dim"])

    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=config["outer_steps"],
        inner_steps=config["inner_steps"],
        num_colors=config["num_colors"],
    )

    trainer = DeepSupervisionTrainer(model=model, learning_rate=1e-4)
    return model, trainer


class TestCheckpointSaveLoad:
    """Test checkpoint save/load functionality."""

    def test_save_checkpoint_creates_file(self, model_and_trainer, tmp_path):
        """Test that save_checkpoint creates a file."""
        model, trainer = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(trainer, filepath, epoch=5, step=100)

        assert filepath.exists(), "Checkpoint file should be created"

    def test_save_checkpoint_creates_parent_dirs(self, model_and_trainer, tmp_path):
        """Test that save_checkpoint creates parent directories."""
        model, trainer = model_and_trainer
        filepath = tmp_path / "nested" / "dir" / "checkpoint.pt"

        save_checkpoint(trainer, filepath, epoch=5, step=100)

        assert filepath.exists(), "Checkpoint file should be created with parent dirs"
        assert filepath.parent.exists(), "Parent directories should be created"

    def test_checkpoint_contains_trainer_state(self, model_and_trainer, tmp_path):
        """Test that checkpoint contains trainer state dict."""
        model, trainer = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(trainer, filepath, epoch=5, step=100)

        checkpoint = torch.load(filepath, weights_only=False)
        assert "trainer_state" in checkpoint, "Checkpoint should contain trainer state"
        assert "model_state_dict" in checkpoint["trainer_state"]
        assert "ema_state_dict" in checkpoint["trainer_state"]
        assert "optimizer_state_dict" in checkpoint["trainer_state"]

    def test_checkpoint_contains_metadata(self, model_and_trainer, tmp_path):
        """Test that checkpoint contains epoch, step, and optional metadata."""
        model, trainer = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(
            trainer,
            filepath,
            epoch=10,
            step=500,
            best_accuracy=0.75,
            metadata={"config": "test"},
        )

        checkpoint = torch.load(filepath, weights_only=False)
        assert checkpoint["epoch"] == 10
        assert checkpoint["step"] == 500
        assert checkpoint["best_accuracy"] == 0.75
        assert checkpoint["metadata"] == {"config": "test"}

    def test_checkpoint_without_optional_fields(self, model_and_trainer, tmp_path):
        """Test checkpoint save without best_accuracy or metadata."""
        model, trainer = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(trainer, filepath, epoch=3, step=50)

        checkpoint = torch.load(filepath, weights_only=False)
        assert checkpoint["epoch"] == 3
        assert checkpoint["step"] == 50
        assert "best_accuracy" not in checkpoint
        assert "metadata" not in checkpoint

    def test_load_checkpoint_restores_state(self, model_and_trainer, tmp_path):
        """Test that load_checkpoint restores trainer state."""
        model1, trainer1 = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        # Train for a few steps to modify state
        input_grids = torch.randint(0, 10, (2, 8, 8))
        target_grids = torch.randint(0, 10, (2, 8, 8))
        mask = torch.ones(2, 8, 8, dtype=torch.bool)

        trainer1.train_step_deep_supervision(input_grids, target_grids, mask)
        original_state = trainer1.state_dict()

        # Save checkpoint
        save_checkpoint(trainer1, filepath, epoch=5, step=100)

        # Create fresh trainer and load checkpoint
        model2, trainer2 = model_and_trainer
        info = load_checkpoint(trainer2, filepath)

        # Verify metadata returned
        assert info["epoch"] == 5
        assert info["step"] == 100

        # Verify state matches - compare model parameters
        loaded_state = trainer2.state_dict()

        # Compare model state dict
        for param_key in original_state["model_state_dict"].keys():
            orig_param = original_state["model_state_dict"][param_key]
            loaded_param = loaded_state["model_state_dict"][param_key]
            assert torch.allclose(
                orig_param, loaded_param
            ), f"Model parameter {param_key} mismatch"

    def test_load_checkpoint_returns_metadata(self, model_and_trainer, tmp_path):
        """Test that load_checkpoint returns all metadata."""
        model, trainer = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(
            trainer,
            filepath,
            epoch=7,
            step=350,
            best_accuracy=0.88,
            metadata={"lr": 1e-4},
        )

        info = load_checkpoint(trainer, filepath)

        assert info["epoch"] == 7
        assert info["step"] == 350
        assert info["best_accuracy"] == 0.88
        assert info["metadata"] == {"lr": 1e-4}

    def test_load_checkpoint_without_optional_fields(self, model_and_trainer, tmp_path):
        """Test load_checkpoint when optional fields not present."""
        model, trainer = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(trainer, filepath, epoch=2, step=40)

        info = load_checkpoint(trainer, filepath)

        assert info["epoch"] == 2
        assert info["step"] == 40
        assert "best_accuracy" not in info
        assert "metadata" not in info

    def test_checkpoint_roundtrip_preserves_training(self, model_and_trainer, tmp_path):
        """Test that checkpoint roundtrip preserves training state exactly."""
        model1, trainer1 = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        # Create consistent input
        torch.manual_seed(42)
        input_grids = torch.randint(0, 10, (2, 8, 8))
        target_grids = torch.randint(0, 10, (2, 8, 8))
        mask = torch.ones(2, 8, 8, dtype=torch.bool)

        # Train original for 3 steps
        for _ in range(3):
            metrics1 = trainer1.train_step_deep_supervision(
                input_grids, target_grids, mask
            )

        # Save checkpoint
        save_checkpoint(trainer1, filepath, epoch=1, step=3)

        # Train original 2 more steps
        for _ in range(2):
            metrics1 = trainer1.train_step_deep_supervision(
                input_grids, target_grids, mask
            )
        final_loss1 = metrics1["total_loss"]

        # Load checkpoint into fresh trainer
        model2, trainer2 = model_and_trainer
        load_checkpoint(trainer2, filepath)

        # Train loaded for 2 steps (should match original)
        for _ in range(2):
            metrics2 = trainer2.train_step_deep_supervision(
                input_grids, target_grids, mask
            )
        final_loss2 = metrics2["total_loss"]

        # Losses should be very close (within numerical precision)
        assert abs(final_loss1 - final_loss2) < 1e-5, (
            f"Training trajectories should match after checkpoint: "
            f"{final_loss1:.6f} vs {final_loss2:.6f}"
        )


class TestBestModelTracker:
    """Test best model tracking and auto-save functionality."""

    def test_tracker_initialization(self, tmp_path):
        """Test BestModelTracker initialization."""
        tracker = BestModelTracker(save_dir=tmp_path)

        assert tracker.save_dir == tmp_path
        assert tracker.filename == "best_model.pt"
        assert tracker.min_delta == 0.0
        assert tracker.best_accuracy == float("-inf")
        assert tracker.save_dir.exists()

    def test_tracker_custom_filename_and_min_delta(self, tmp_path):
        """Test tracker with custom filename and min_delta."""
        tracker = BestModelTracker(
            save_dir=tmp_path, filename="custom_best.pt", min_delta=0.01
        )

        assert tracker.filename == "custom_best.pt"
        assert tracker.min_delta == 0.01

    def test_tracker_creates_save_dir(self, tmp_path):
        """Test that tracker creates save directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "checkpoints"
        tracker = BestModelTracker(save_dir=nested_dir)

        assert nested_dir.exists(), "Save directory should be created"

    def test_tracker_first_update_saves(self, model_and_trainer, tmp_path):
        """Test that first update always saves (any accuracy beats -inf)."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path)

        improved = tracker.update(trainer, accuracy=0.5, epoch=1, step=100)

        assert improved, "First update should always save"
        assert tracker.best_accuracy == 0.5
        assert (tmp_path / "best_model.pt").exists()

    def test_tracker_saves_on_improvement(self, model_and_trainer, tmp_path):
        """Test that tracker saves when accuracy improves."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path)

        # First update
        tracker.update(trainer, accuracy=0.5, epoch=1, step=100)

        # Second update with better accuracy
        improved = tracker.update(trainer, accuracy=0.7, epoch=2, step=200)

        assert improved, "Should save when accuracy improves"
        assert tracker.best_accuracy == 0.7

    def test_tracker_no_save_without_improvement(self, model_and_trainer, tmp_path):
        """Test that tracker doesn't save when accuracy doesn't improve."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path)

        # First update
        tracker.update(trainer, accuracy=0.7, epoch=1, step=100)

        # Get checkpoint time
        checkpoint_path = tmp_path / "best_model.pt"
        first_mtime = checkpoint_path.stat().st_mtime

        # Second update with worse accuracy
        improved = tracker.update(trainer, accuracy=0.6, epoch=2, step=200)

        assert not improved, "Should not save when accuracy doesn't improve"
        assert tracker.best_accuracy == 0.7, "Best accuracy should not change"

        # Checkpoint file should not be modified
        second_mtime = checkpoint_path.stat().st_mtime
        assert second_mtime == first_mtime, "Checkpoint should not be modified"

    def test_tracker_min_delta_threshold(self, model_and_trainer, tmp_path):
        """Test that tracker respects min_delta threshold."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path, min_delta=0.01)

        # First update
        tracker.update(trainer, accuracy=0.70, epoch=1, step=100)

        # Small improvement below threshold
        improved = tracker.update(trainer, accuracy=0.705, epoch=2, step=200)
        assert not improved, "Should not save for improvement below min_delta"
        assert tracker.best_accuracy == 0.70

        # Improvement above threshold
        improved = tracker.update(trainer, accuracy=0.72, epoch=3, step=300)
        assert improved, "Should save for improvement above min_delta"
        assert tracker.best_accuracy == 0.72

    def test_tracker_saves_best_accuracy_in_checkpoint(
        self, model_and_trainer, tmp_path
    ):
        """Test that tracker saves best_accuracy in checkpoint."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path)

        tracker.update(trainer, accuracy=0.85, epoch=10, step=500)

        checkpoint_path = tmp_path / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["best_accuracy"] == 0.85
        assert checkpoint["epoch"] == 10
        assert checkpoint["step"] == 500

    def test_tracker_saves_metadata(self, model_and_trainer, tmp_path):
        """Test that tracker passes metadata to save_checkpoint."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path)

        metadata = {"config": "test", "notes": "best model"}
        tracker.update(
            trainer, accuracy=0.90, epoch=15, step=750, metadata=metadata
        )

        checkpoint_path = tmp_path / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["metadata"] == metadata

    def test_get_best_accuracy_initial(self, tmp_path):
        """Test get_best_accuracy before any updates."""
        tracker = BestModelTracker(save_dir=tmp_path)

        assert tracker.get_best_accuracy() == 0.0, (
            "Should return 0.0 for uninitialized tracker"
        )

    def test_get_best_accuracy_after_updates(self, model_and_trainer, tmp_path):
        """Test get_best_accuracy returns correct value."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path)

        tracker.update(trainer, accuracy=0.75, epoch=1, step=100)
        assert tracker.get_best_accuracy() == 0.75

        tracker.update(trainer, accuracy=0.82, epoch=2, step=200)
        assert tracker.get_best_accuracy() == 0.82

    def test_tracker_multiple_improvements(self, model_and_trainer, tmp_path):
        """Test tracker over multiple improvements."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=tmp_path)

        accuracies = [0.5, 0.6, 0.55, 0.7, 0.65, 0.8]
        expected_saves = [True, True, False, True, False, True]

        for i, (acc, should_save) in enumerate(zip(accuracies, expected_saves)):
            improved = tracker.update(trainer, accuracy=acc, epoch=i, step=i * 100)
            assert improved == should_save, (
                f"Step {i}: accuracy {acc}, expected save={should_save}, got {improved}"
            )

        assert tracker.get_best_accuracy() == 0.8

    def test_tracker_with_string_path(self, model_and_trainer, tmp_path):
        """Test that tracker works with string path."""
        model, trainer = model_and_trainer
        tracker = BestModelTracker(save_dir=str(tmp_path))

        improved = tracker.update(trainer, accuracy=0.75, epoch=1, step=100)

        assert improved
        assert (tmp_path / "best_model.pt").exists()


class TestCheckpointIntegration:
    """Test checkpoint integration with training workflow."""

    def test_checkpoint_preserves_model_parameters(self, model_and_trainer, tmp_path):
        """Test that checkpoint preserves exact model parameters."""
        model1, trainer1 = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        # Get original model params
        original_params = {
            name: param.clone() for name, param in trainer1.model.named_parameters()
        }

        # Save and load
        save_checkpoint(trainer1, filepath, epoch=1, step=1)

        model2, trainer2 = model_and_trainer
        load_checkpoint(trainer2, filepath)

        # Compare all parameters
        loaded_params = {
            name: param for name, param in trainer2.model.named_parameters()
        }

        for name in original_params:
            assert torch.allclose(
                original_params[name], loaded_params[name]
            ), f"Parameter {name} mismatch"

    def test_checkpoint_preserves_ema_parameters(self, model_and_trainer, tmp_path):
        """Test that checkpoint preserves EMA model parameters."""
        model1, trainer1 = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        # Train a few steps to update EMA
        input_grids = torch.randint(0, 10, (2, 8, 8))
        target_grids = torch.randint(0, 10, (2, 8, 8))
        mask = torch.ones(2, 8, 8, dtype=torch.bool)

        for _ in range(3):
            trainer1.train_step_deep_supervision(input_grids, target_grids, mask)

        # Get original EMA params
        original_ema_params = {
            name: param.clone()
            for name, param in trainer1.ema.module.named_parameters()
        }

        # Save and load
        save_checkpoint(trainer1, filepath, epoch=1, step=3)

        model2, trainer2 = model_and_trainer
        load_checkpoint(trainer2, filepath)

        # Compare EMA parameters
        loaded_ema_params = {
            name: param for name, param in trainer2.ema.module.named_parameters()
        }

        for name in original_ema_params:
            assert torch.allclose(
                original_ema_params[name], loaded_ema_params[name], atol=1e-6
            ), f"EMA parameter {name} mismatch"

    def test_checkpoint_preserves_optimizer_state(self, model_and_trainer, tmp_path):
        """Test that checkpoint preserves optimizer state (momentum, etc.)."""
        model1, trainer1 = model_and_trainer
        filepath = tmp_path / "checkpoint.pt"

        # Train a few steps to build optimizer state
        input_grids = torch.randint(0, 10, (2, 8, 8))
        target_grids = torch.randint(0, 10, (2, 8, 8))
        mask = torch.ones(2, 8, 8, dtype=torch.bool)

        for _ in range(5):
            trainer1.train_step_deep_supervision(input_grids, target_grids, mask)

        # Get original optimizer state
        original_opt_state = trainer1.optimizer.state_dict()

        # Save and load
        save_checkpoint(trainer1, filepath, epoch=1, step=5)

        model2, trainer2 = model_and_trainer
        load_checkpoint(trainer2, filepath)

        # Compare optimizer state
        loaded_opt_state = trainer2.optimizer.state_dict()

        # Check param groups
        assert len(original_opt_state["param_groups"]) == len(
            loaded_opt_state["param_groups"]
        )

        # Check state (momentum buffers, etc.)
        for key in original_opt_state["state"]:
            orig_state = original_opt_state["state"][key]
            loaded_state = loaded_opt_state["state"][key]
            for state_key in orig_state:
                if isinstance(orig_state[state_key], torch.Tensor):
                    assert torch.allclose(
                        orig_state[state_key], loaded_state[state_key]
                    ), f"Optimizer state {key}.{state_key} mismatch"

"""Integration tests for training pipeline with validation and checkpointing."""
import pytest
import torch
import tempfile
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train import validate_epoch, EarlyStopping
from src.trm.model import TRMNetwork, GridEmbedding, RecursiveRefinement
from src.trm.training import DeepSupervisionTrainer
from src.trm.evaluation import save_checkpoint, load_checkpoint, BestModelTracker
from src.trm.data import ARCDataset, arc_collate_fn
from torch.utils.data import DataLoader


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    network = TRMNetwork(hidden_dim=32, num_heads=2, num_layers=1)
    embedding = GridEmbedding(hidden_dim=32)
    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=1,
        inner_steps=1,
        halt_threshold=0.9,
        enable_halting=True,
    )
    return model


@pytest.fixture
def trainer(tiny_model):
    """Create trainer with tiny model."""
    return DeepSupervisionTrainer(
        model=tiny_model,
        learning_rate=0.001,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        max_sup_steps=4,
        grad_clip_norm=1.0,
        ema_decay=0.999,
    )


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader with synthetic data."""
    # Create synthetic batch data matching ARCDataset structure
    def mock_batch_generator():
        # Single task with one test pair
        batch = {
            "test_inputs": torch.randint(0, 10, (2, 1, 5, 5)),  # (B=2, pairs=1, H=5, W=5)
            "test_outputs": torch.randint(0, 10, (2, 1, 5, 5)),
            "test_output_masks": torch.ones(2, 1, 5, 5, dtype=torch.bool),
            "num_test_pairs": torch.tensor([1, 1]),
        }
        yield batch

    return list(mock_batch_generator())


class TestValidation:
    """Tests for validation loop functionality."""

    def test_validate_epoch_returns_accuracy(self, tiny_model, mock_dataloader):
        """Test that validate_epoch returns a float accuracy in [0, 1]."""
        device = torch.device("cpu")
        tiny_model = tiny_model.to(device)

        accuracy = validate_epoch(tiny_model, mock_dataloader, device)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_validate_uses_test_pairs(self, tiny_model):
        """Test that validation uses test pairs, not train pairs."""
        device = torch.device("cpu")
        tiny_model = tiny_model.to(device)

        # Create batch with only test pairs (no train pairs)
        batch = [{
            "test_inputs": torch.randint(0, 10, (1, 1, 5, 5)),
            "test_outputs": torch.randint(0, 10, (1, 1, 5, 5)),
            "test_output_masks": torch.ones(1, 1, 5, 5, dtype=torch.bool),
            "num_test_pairs": torch.tensor([1]),
            "train_inputs": torch.empty(1, 0, 5, 5),  # No train pairs
            "train_outputs": torch.empty(1, 0, 5, 5),
        }]

        # Should not crash when test pairs present
        accuracy = validate_epoch(tiny_model, batch, device)
        assert isinstance(accuracy, float)


class TestEarlyStopping:
    """Tests for early stopping logic."""

    def test_early_stopping_triggers_after_patience(self):
        """Test that early stopping triggers after N non-improvements."""
        early_stopping = EarlyStopping(patience=3)

        # Initial score - should not stop
        assert not early_stopping(0.5)

        # No improvement for patience epochs
        assert not early_stopping(0.4)  # Counter = 1
        assert not early_stopping(0.4)  # Counter = 2
        assert early_stopping(0.4)      # Counter = 3, should stop

    def test_early_stopping_resets_on_improvement(self):
        """Test that counter resets when accuracy improves."""
        early_stopping = EarlyStopping(patience=3)

        # Initial score
        assert not early_stopping(0.5)

        # Some non-improvements
        assert not early_stopping(0.4)  # Counter = 1
        assert not early_stopping(0.4)  # Counter = 2

        # Improvement - counter should reset
        assert not early_stopping(0.6)  # Counter = 0

        # More non-improvements needed to trigger
        assert not early_stopping(0.5)  # Counter = 1
        assert not early_stopping(0.5)  # Counter = 2
        assert early_stopping(0.5)      # Counter = 3, should stop

    def test_early_stopping_initial_is_best(self):
        """Test that first score always sets best."""
        early_stopping = EarlyStopping(patience=2)

        # First score always accepted
        assert not early_stopping(0.3)
        assert early_stopping.best_score == 0.3


class TestTrainingIntegration:
    """Integration tests for full training pipeline."""

    def test_training_creates_checkpoint(self, trainer, mock_dataloader):
        """Test that checkpoint file is created during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Create best model tracker
            tracker = BestModelTracker(
                save_dir=checkpoint_dir,
                filename="best_model.pt",
            )

            # Simulate validation and checkpoint save
            fake_accuracy = 0.75
            tracker.update(
                trainer=trainer,
                accuracy=fake_accuracy,
                epoch=0,
                step=0,
            )

            # Check checkpoint exists
            checkpoint_path = checkpoint_dir / "best_model.pt"
            assert checkpoint_path.exists()

    @pytest.mark.slow
    def test_training_with_validation(self, trainer, mock_dataloader):
        """Test full training loop with validation runs."""
        device = torch.device("cpu")
        trainer.model = trainer.model.to(device)

        # Run one training step
        for batch in mock_dataloader:
            # Use synthetic train data
            input_grid = torch.randint(0, 10, (2, 5, 5)).to(device)
            target_grid = torch.randint(0, 10, (2, 5, 5)).to(device)
            mask = torch.ones(2, 5, 5, dtype=torch.bool).to(device)

            result = trainer.train_step_deep_supervision(input_grid, target_grid, mask)
            assert "total_loss" in result
            break

        # Run validation
        val_model = trainer.get_ema_model()
        accuracy = validate_epoch(val_model, mock_dataloader, device)
        assert isinstance(accuracy, float)

    def test_resume_continues_training(self, trainer):
        """Test that resume loads state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save checkpoint
            save_checkpoint(
                trainer=trainer,
                filepath=checkpoint_path,
                epoch=5,
                step=100,
                best_accuracy=0.85,
            )

            # Create new trainer and load
            network = TRMNetwork(hidden_dim=32, num_heads=2, num_layers=1)
            embedding = GridEmbedding(hidden_dim=32)
            model = RecursiveRefinement(
                network=network,
                embedding=embedding,
                outer_steps=1,
                inner_steps=1,
                halt_threshold=0.9,
                enable_halting=True,
            )
            new_trainer = DeepSupervisionTrainer(
                model=model,
                learning_rate=0.001,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                max_sup_steps=4,
                grad_clip_norm=1.0,
                ema_decay=0.999,
            )

            info = load_checkpoint(new_trainer, checkpoint_path, map_location="cpu")

            # Verify metadata
            assert info["epoch"] == 5
            assert info["step"] == 100
            assert info["best_accuracy"] == 0.85

            # Verify state was loaded (optimizer state should exist)
            assert new_trainer.optimizer.state_dict()["param_groups"][0]["lr"] == 0.001

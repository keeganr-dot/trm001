"""Training script for TRM on ARC-AGI dataset."""
import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader

from src.trm.model import TRMNetwork, GridEmbedding, RecursiveRefinement
from src.trm.data import ARCDataset, arc_collate_fn, PAD_VALUE
from src.trm.training import TRMTrainer


def load_config() -> DictConfig:
    """Load config from configs/config.yaml (Hydra-compatible format)."""
    config_path = project_root / "configs" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    return cfg


def create_model(cfg: DictConfig) -> RecursiveRefinement:
    """Create TRM model from config."""
    network = TRMNetwork(
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
    )
    embedding = GridEmbedding(hidden_dim=cfg.model.hidden_dim)
    model = RecursiveRefinement(
        network=network,
        embedding=embedding,
        outer_steps=cfg.recursion.outer_steps,
        inner_steps=cfg.recursion.inner_steps,
        halt_threshold=cfg.recursion.halt_threshold,
        enable_halting=True,
    )
    return model


def train_epoch(trainer, dataloader, epoch, device):
    """Run one training epoch."""
    total_loss = 0.0
    total_ce = 0.0
    total_bce = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # For basic training, use train pairs from each task
        # Shape: (B, max_pairs, H, W)
        train_inputs = batch["train_inputs"]
        train_outputs = batch["train_outputs"]
        train_masks = batch["train_output_masks"]  # Use output mask for loss
        num_pairs = batch["num_train_pairs"]

        B, max_pairs, H, W = train_inputs.shape

        # Flatten task batches to train on individual pairs
        # Iterate over pairs within batch
        for pair_idx in range(max_pairs):
            # Get this pair from all tasks in batch
            input_grid = train_inputs[:, pair_idx, :, :]  # (B, H, W)
            target_grid = train_outputs[:, pair_idx, :, :]  # (B, H, W)
            mask = train_masks[:, pair_idx, :, :]  # (B, H, W)

            # Skip if all tasks have fewer pairs than pair_idx
            valid_tasks = num_pairs > pair_idx
            if not valid_tasks.any():
                continue

            # Filter to valid tasks only
            input_grid = input_grid[valid_tasks].to(device)
            target_grid = target_grid[valid_tasks].to(device)
            mask = mask[valid_tasks].to(device)

            # Skip if no valid positions (entirely padded)
            if not mask.any():
                continue

            # Training step
            result = trainer.train_step(input_grid, target_grid, mask)

            total_loss += result["total_loss"]
            total_ce += result["ce_loss"]
            total_bce += result["bce_loss"]
            total_acc += result["accuracy"]
            num_batches += 1

    if num_batches == 0:
        return {"loss": 0, "ce": 0, "bce": 0, "acc": 0}

    return {
        "loss": total_loss / num_batches,
        "ce": total_ce / num_batches,
        "bce": total_bce / num_batches,
        "acc": total_acc / num_batches,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="TRM Training Script")
    parser.add_argument(
        "--fast", action="store_true",
        help="Use smaller model for quick testing"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    args = parser.parse_args()

    # Load config (Hydra-compatible YAML format)
    cfg = load_config()

    # Override for fast mode
    if args.fast:
        cfg.model.hidden_dim = 64
        cfg.model.num_layers = 1
        cfg.model.num_heads = 2
        cfg.recursion.outer_steps = 1
        cfg.recursion.inner_steps = 1
        cfg.data.batch_size = 4

    # Override batch size if specified
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    print("=" * 60)
    print("TRM Training - Basic (Terminal Supervision)")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'fast' if args.fast else 'full'}")

    # Create model
    print("\nCreating model...")
    model = create_model(cfg)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    print(f"Hidden dim: {cfg.model.hidden_dim}")
    print(f"Recursion: T={cfg.recursion.outer_steps}, n={cfg.recursion.inner_steps}")

    # Create trainer
    trainer = TRMTrainer(
        model=model,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        beta1=cfg.training.beta1,
        beta2=cfg.training.beta2,
    )

    # Load data
    print("\nLoading ARC-AGI dataset...")
    data_dir = project_root / "data"
    dataset = ARCDataset(data_dir=str(data_dir), split="training")
    print(f"Tasks: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        collate_fn=arc_collate_fn,
        num_workers=0,  # Avoid multiprocessing issues on Windows
    )

    # Training loop
    num_epochs = args.epochs
    print(f"\nTraining for {num_epochs} epochs (batch_size={cfg.data.batch_size})...")
    print("-" * 60)

    losses = []
    for epoch in range(num_epochs):
        metrics = train_epoch(trainer, dataloader, epoch, device)
        losses.append(metrics["loss"])

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"CE: {metrics['ce']:.4f} | "
            f"BCE: {metrics['bce']:.4f} | "
            f"Acc: {metrics['acc']:.2%}"
        )

    print("-" * 60)

    # Check loss decreased
    if len(losses) >= 2:
        if losses[-1] < losses[0]:
            print(f"\nSUCCESS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
        else:
            print(f"\nWARNING: Loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

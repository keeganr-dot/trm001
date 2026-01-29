"""Deep supervision trainer with gradient detachment."""
import torch
import torch.nn as nn
from torch.optim import AdamW

from .trainer import TRMTrainer
from ..model import RecursiveRefinement


class DeepSupervisionTrainer(TRMTrainer):
    """
    Training with deep supervision at intermediate recursion steps.

    Extends TRMTrainer with:
    - Loss computation at each supervision step (TRAIN-01)
    - Gradient detachment between steps (TRAIN-02)
    - Configurable max supervision steps (TRAIN-08)
    - Gradient clipping for stability (TRAIN-07)

    Args:
        model: RecursiveRefinement instance
        learning_rate: Learning rate for AdamW (default 1e-4)
        weight_decay: Weight decay for AdamW (default 0.01)
        beta1: First beta for AdamW (default 0.9)
        beta2: Second beta for AdamW (default 0.95)
        halting_loss_weight: Weight for BCE halting loss (default 0.1)
        max_sup_steps: Maximum supervision steps (TRAIN-08, default 16)
        grad_clip_norm: Max gradient norm for clipping (TRAIN-07, default 1.0)
    """

    def __init__(
        self,
        model: RecursiveRefinement,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.95,
        halting_loss_weight: float = 0.1,
        max_sup_steps: int = 16,
        grad_clip_norm: float = 1.0,
    ):
        super().__init__(
            model, learning_rate, weight_decay, beta1, beta2, halting_loss_weight
        )
        self.max_sup_steps = max_sup_steps
        self.grad_clip_norm = grad_clip_norm

    def train_step_deep_supervision(
        self,
        input_grids: torch.Tensor,
        target_grids: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Training step with deep supervision.

        Implements manual supervision loop with gradient detachment:
        1. Initialize y, z states
        2. For each supervision step:
           a. Run inner loop (n iterations) to refine z
           b. Update y from (y, z)
           c. Compute loss at this step
           d. DETACH y and z before next step (TRAIN-02)
        3. Normalize total loss by steps taken
        4. Backward, clip gradients, optimizer step

        Args:
            input_grids: Input grid (B, H, W) with values 0-9 or -1
            target_grids: Target grid (B, H, W) with values 0-9
            mask: Boolean mask (B, H, W), True for valid positions

        Returns:
            Dictionary with:
                - total_loss: Normalized loss value
                - ce_loss: Average CE loss across steps
                - bce_loss: Average BCE loss across steps
                - accuracy: Final step accuracy
                - steps: Number of supervision steps taken
                - grad_norm: Gradient norm before clipping
                - halted_early: Whether halting stopped early
        """
        self.model.train()
        self.optimizer.zero_grad()

        B, H, W = input_grids.shape
        device = input_grids.device
        num_colors = self.model.num_colors

        # Initialize states
        y = torch.zeros(B, H, W, num_colors, device=device, dtype=torch.float32)
        z = torch.zeros_like(y)

        total_loss = 0.0
        total_ce_loss = 0.0
        total_bce_loss = 0.0
        steps_taken = 0
        halted_early = False
        last_accuracy = 0.0
        last_halt_confidence = None

        # Deep supervision loop
        for sup_step in range(self.max_sup_steps):
            # Inner loop: refine latent z
            for i in range(self.model.inner_steps):
                combined = self.model._combine_states(
                    states=[input_grids, y, z],
                    is_logits=[False, True, True]
                )
                output = self.model._forward_through_network(combined)
                z = output["logits"]

            # Outer step: update answer y
            combined_yz = self.model._combine_states(
                states=[y, z],
                is_logits=[True, True]
            )
            output = self.model._forward_through_network(combined_yz)
            y = output["logits"]
            halt_confidence = output["halt_confidence"]
            last_halt_confidence = halt_confidence

            # Compute loss at this supervision step (TRAIN-01)
            loss_dict = self.compute_loss(y, target_grids, halt_confidence, mask)
            step_loss = loss_dict["total_loss"]

            total_loss = total_loss + step_loss
            total_ce_loss = total_ce_loss + loss_dict["ce_loss"]
            total_bce_loss = total_bce_loss + loss_dict["bce_loss"]
            last_accuracy = loss_dict["accuracy"]
            steps_taken += 1

            # CRITICAL: Detach states between supervision steps (TRAIN-02)
            # This prevents backpropagation through previous supervision iterations
            # Memory stays constant because each step's graph is discarded
            z = z.detach()
            y = y.detach()

            # Check halting condition
            if self.model.enable_halting:
                if (halt_confidence >= self.model.halt_threshold).all():
                    halted_early = True
                    break

        # Normalize loss by steps taken (prevents loss scale varying with iterations)
        normalized_loss = total_loss / max(steps_taken, 1)

        # Backward pass
        normalized_loss.backward()

        # Gradient clipping (TRAIN-07)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.grad_clip_norm
        )

        # Optimizer step
        self.optimizer.step()

        return {
            "total_loss": normalized_loss.item(),
            "ce_loss": (total_ce_loss / max(steps_taken, 1)).item() if isinstance(total_ce_loss, torch.Tensor) else total_ce_loss / max(steps_taken, 1),
            "bce_loss": (total_bce_loss / max(steps_taken, 1)).item() if isinstance(total_bce_loss, torch.Tensor) else total_bce_loss / max(steps_taken, 1),
            "accuracy": last_accuracy.item() if isinstance(last_accuracy, torch.Tensor) else last_accuracy,
            "steps": steps_taken,
            "grad_norm": grad_norm.item(),
            "halted_early": halted_early,
        }

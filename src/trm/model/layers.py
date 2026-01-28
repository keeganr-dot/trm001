"""Core transformer components for TRM architecture."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center (no mean subtraction).
    Formula: x * rsqrt(mean(x^2) + eps) * weight

    This is more efficient and works as well as LayerNorm for transformer models.

    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Only weight parameter, no bias (per paper)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_normalized = x / rms
        return self.weight * x_normalized


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit for feed-forward networks.

    SwiGLU(x) = Swish(x @ W_gate) * (x @ W_up) @ W_down
    where Swish(x) = x * sigmoid(x)

    Paper uses expansion factor for intermediate dimension.
    Standard is 4x, but gating requires 2x more parameters, so we use 8/3 * 2 ≈ 5.33x.

    Args:
        dim: Input/output dimension (hidden_dim)
        expansion_factor: Multiplier for intermediate dimension (default 8/3)
    """

    def __init__(self, dim: int, expansion_factor: float = 8/3):
        super().__init__()
        # Intermediate dimension for gating
        # Factor of 2 because we have both gate and up projections
        intermediate_dim = int(dim * expansion_factor * 2)

        # Gate and up projections (no bias per paper)
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)

        # Down projection back to dim (no bias per paper)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Output tensor of same shape
        """
        # Swish activation: x * sigmoid(x)
        gate = F.silu(self.gate_proj(x))  # silu is swish

        # Element-wise gating
        up = self.up_proj(x)
        gated = gate * up

        # Project back down
        return self.down_proj(gated)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 2D grids.

    Encodes position information by rotating features in paired dimensions.
    For 2D grids, we apply rotation based on both row and column positions.

    This is applied to query and key tensors (not value) in attention.

    Args:
        dim: Dimension per attention head
        max_seq_len: Maximum sequence length to precompute (default 2048)
        base: Base for frequency computation (default 10000)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute frequency values
        # Pairs of dimensions get rotated at different frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding.

        Args:
            x: Input tensor of shape (B, seq_len, num_heads, head_dim)
               or (B, seq_len, head_dim)
            positions: Position indices of shape (B, seq_len) - can be 1D positions
                      or flattened 2D positions

        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        # positions: (B, seq_len)
        # inv_freq: (dim/2,)

        # Compute rotation angles: (B, seq_len, dim/2)
        freqs = torch.einsum("bi,j->bij", positions.float(), self.inv_freq)

        # Create rotation matrix using cos and sin
        # (B, seq_len, dim/2) -> (B, seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()

        # Apply rotation to x
        x_shape = x.shape
        if len(x_shape) == 4:
            # (B, seq_len, num_heads, head_dim)
            B, seq_len, num_heads, head_dim = x_shape

            # Expand cos/sin for broadcasting with num_heads dimension
            # cos_emb, sin_emb: (B, seq_len, dim) -> (B, seq_len, 1, dim)
            cos_emb = cos_emb.unsqueeze(2)
            sin_emb = sin_emb.unsqueeze(2)

            # Reshape x for rotation: split into real and imaginary parts
            # (B, seq_len, num_heads, head_dim) -> (B, seq_len, num_heads, head_dim/2, 2)
            x_reshaped = x.reshape(B, seq_len, num_heads, head_dim // 2, 2)

            # Extract real and imaginary (pairs of dimensions)
            x_real = x_reshaped[..., 0]  # (B, seq_len, num_heads, head_dim/2)
            x_imag = x_reshaped[..., 1]  # (B, seq_len, num_heads, head_dim/2)

            # Apply rotation: [cos -sin] [x_real]
            #                 [sin  cos] [x_imag]
            cos_part = cos_emb[..., :head_dim//2]  # (B, seq_len, 1, head_dim/2)
            sin_part = sin_emb[..., :head_dim//2]  # (B, seq_len, 1, head_dim/2)

            x_rotated_real = x_real * cos_part - x_imag * sin_part
            x_rotated_imag = x_real * sin_part + x_imag * cos_part

            # Stack and reshape back
            x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
            x_rotated = x_rotated.reshape(B, seq_len, num_heads, head_dim)
        else:
            # (B, seq_len, head_dim)
            B, seq_len, head_dim = x_shape

            # Reshape x for rotation
            x_reshaped = x.reshape(B, seq_len, head_dim // 2, 2)
            x_real = x_reshaped[..., 0]  # (B, seq_len, head_dim/2)
            x_imag = x_reshaped[..., 1]  # (B, seq_len, head_dim/2)

            # Apply rotation
            cos_part = cos_emb[..., :head_dim//2]  # (B, seq_len, head_dim/2)
            sin_part = sin_emb[..., :head_dim//2]  # (B, seq_len, head_dim/2)

            x_rotated_real = x_real * cos_part - x_imag * sin_part
            x_rotated_imag = x_real * sin_part + x_imag * cos_part

            # Stack and reshape back
            x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
            x_rotated = x_rotated.reshape(B, seq_len, head_dim)

        return x_rotated


def apply_rotary_pos_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    row_pos: torch.Tensor,
    col_pos: torch.Tensor,
    rope: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2D rotary position embeddings to query and key.

    For grid data, we encode both row and column positions.
    We split the head dimension in half and apply row positions to first half,
    column positions to second half.

    Args:
        q: Query tensor (B, seq_len, num_heads, head_dim)
        k: Key tensor (B, seq_len, num_heads, head_dim)
        row_pos: Row positions (B, seq_len)
        col_pos: Column positions (B, seq_len)
        rope: RotaryEmbedding instance

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as input
    """
    # Split head dimension for 2D position encoding
    head_dim = q.shape[-1]
    half_dim = head_dim // 2

    # Apply row positions to first half of dimensions
    q_row = rope(q[..., :half_dim], row_pos)
    k_row = rope(k[..., :half_dim], row_pos)

    # Apply column positions to second half of dimensions
    q_col = rope(q[..., half_dim:], col_pos)
    k_col = rope(k[..., half_dim:], col_pos)

    # Concatenate back
    q_rotated = torch.cat([q_row, q_col], dim=-1)
    k_rotated = torch.cat([k_row, k_col], dim=-1)

    return q_rotated, k_rotated

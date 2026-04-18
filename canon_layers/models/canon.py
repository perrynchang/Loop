"""
Canon Layers: Enhancing Horizontal Information Flow
Implements causal 1-d convolution with kernel size 4 across neighboring tokens.

h'_t = h_t + conv1d([h_{t-3}, h_{t-2}, h_{t-1}, h_t])

Four insertion points per Transformer block:
  Canon-A: before attention (after RMSNorm), dim = d
  Canon-B: inside attention, after Q/K/V projections, dim = 3d
  Canon-C: before MLP (after RMSNorm), dim = d
  Canon-D: inside MLP, before activation, dim = 4d (standard) or 16d (gated)
"""
import torch
import torch.nn as nn
import math


class CanonLayer(nn.Module):
    """
    A single Canon layer: causal conv1d with kernel size 4.
    Applied to sequences of shape (batch, seq_len, dim).

    With residual: h' = h + causal_conv1d(h)
    Without residual: h' = causal_conv1d(h)
    """

    def __init__(self, dim: int, residual: bool = True, kernel_size: int = 4):
        super().__init__()
        self.dim = dim
        self.residual = residual
        self.kernel_size = kernel_size

        # Depthwise (grouped) conv1d: each dimension has its own 4-element filter
        # Input shape for conv1d: (batch, channels, seq_len)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal padding: pad left
            groups=dim,               # depthwise convolution (per-channel weights)
            bias=False,
        )
        # Initialize with kaiming uniform (same as paper's default init)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            h': (batch, seq_len, dim)
        """
        # Transpose for conv1d: (batch, dim, seq_len)
        x_t = x.transpose(1, 2)
        # Apply conv and trim to seq_len (remove future-token padding)
        out = self.conv(x_t)[..., :x.size(1)]
        # Transpose back: (batch, seq_len, dim)
        out = out.transpose(1, 2)
        if self.residual:
            return x + out
        return out


class CanonABCD(nn.Module):
    """
    Full-score Canon: applies Canon at positions A, B, C, D within a Transformer block.
    This module holds all four Canon layers; insertion into the model is done in TransformerBlock.
    """

    def __init__(self, d_model: int, d_attn: int = None, d_mlp: int = None,
                 residual: bool = True, positions: str = "ABCD"):
        """
        Args:
            d_model: hidden dimension (for Canon-A and Canon-C)
            d_attn: dimension for Canon-B (typically 3*d_model for Q+K+V)
            d_mlp: dimension for Canon-D (typically 4*d_model for standard MLP)
            residual: use residual connections
            positions: which positions to enable, e.g. "ABCD", "AC", "AbCD"
        """
        super().__init__()
        self.positions = positions.upper() if positions else ""

        d_attn = d_attn or 3 * d_model
        d_mlp = d_mlp or 4 * d_model

        self.canon_a = CanonLayer(d_model, residual) if 'A' in positions else None
        self.canon_b = CanonLayer(d_attn, residual) if 'B' in positions else None
        self.canon_c = CanonLayer(d_model, residual) if 'C' in positions else None
        self.canon_d = CanonLayer(d_mlp, residual) if 'D' in positions else None

    def apply_a(self, x):
        return self.canon_a(x) if self.canon_a is not None else x

    def apply_b(self, qkv):
        return self.canon_b(qkv) if self.canon_b is not None else qkv

    def apply_c(self, x):
        return self.canon_c(x) if self.canon_c is not None else x

    def apply_d(self, x):
        return self.canon_d(x) if self.canon_d is not None else x

    def extra_repr(self):
        return f"positions={self.positions}"


def count_canon_params(model: nn.Module) -> int:
    """Count parameters in all Canon layers."""
    return sum(p.numel() for m in model.modules()
               if isinstance(m, CanonLayer) for p in m.parameters())

"""
Transformer with Canon Layers (Llama-style).
Implements RoPE and NoPE variants with optional Canon-ABCD integration.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .canon import CanonABCD


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000,
                 rope_fraction: float = 1.0):
        """
        rope_fraction: fraction of dimensions to apply RoPE to (1.0=full, 0.25=quarter).
        """
        super().__init__()
        self.dim = dim
        self.rope_dim = int(dim * rope_fraction) // 2 * 2  # must be even
        theta = 1.0 / (base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer('theta', theta)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.theta.device).float()
        freqs = torch.outer(t, self.theta)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, q, k, seq_len):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        def rotate(x):
            # Apply RoPE to first rope_dim dimensions
            x_rope = x[..., :self.rope_dim]
            x_pass = x[..., self.rope_dim:]
            x1, x2 = x_rope[..., :self.rope_dim // 2], x_rope[..., self.rope_dim // 2:]
            rotated = torch.cat([-x2, x1], dim=-1)
            x_rope = x_rope * cos + rotated * sin
            return torch.cat([x_rope, x_pass], dim=-1)

        return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 4096,
                 rope: bool = True, rope_fraction: float = 1.0, canon: CanonABCD = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.canon = canon

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_fraction=rope_fraction) if rope else None
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Q, K, V projections
        qkv = self.qkv_proj(x)  # (B, T, 3*C)

        # Canon-B: applied after Q/K/V projections
        if self.canon is not None:
            qkv = self.canon.apply_b(qkv)

        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Apply RoPE
        if self.rope is not None:
            q, k = self.rope(q, k, T)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.mask[:T, :T].bool().logical_not(), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class GatedMLP(nn.Module):
    """Llama-style gated MLP with SiLU activation."""

    def __init__(self, d_model: int, intermediate_size: int = None, canon: CanonABCD = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 8 * d_model // 3  # ~2.67x, often rounded to nice number
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.canon = canon
        self.intermediate_size = intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Canon-D: applied inside MLP before activation
        if self.canon is not None:
            # Concatenate gate and up for Canon-D (dim = 2 * intermediate_size)
            combined = torch.cat([gate, up], dim=-1)
            combined = self.canon.apply_d(combined)
            gate, up = combined.split(self.intermediate_size, dim=-1)

        hidden = F.silu(gate) * up
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, intermediate_size: int = None,
                 max_seq_len: int = 4096, rope: bool = True, rope_fraction: float = 1.0,
                 canon_positions: str = "", canon_residual: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Build Canon module if any positions are requested
        canon = None
        if canon_positions:
            intermediate = intermediate_size or (8 * d_model // 3)
            canon = CanonABCD(
                d_model=d_model,
                d_attn=3 * d_model,
                d_mlp=2 * intermediate,  # gate + up projections
                residual=canon_residual,
                positions=canon_positions,
            )
        self.canon = canon

        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, rope, rope_fraction, canon)
        self.mlp = GatedMLP(d_model, intermediate_size, canon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual architecture
        residual = x
        normed = self.norm1(x)

        # Canon-A: before attention, after RMSNorm
        if self.canon is not None:
            normed = self.canon.apply_a(normed)

        x = residual + self.attn(normed)

        residual = x
        normed = self.norm2(x)

        # Canon-C: before MLP, after RMSNorm
        if self.canon is not None:
            normed = self.canon.apply_c(normed)

        x = residual + self.mlp(normed)
        return x


class TransformerLM(nn.Module):
    """
    Llama-style Transformer language model with optional Canon layers.

    Configurations:
      - RoPE: rope=True, rope_fraction=1.0
      - NoPE: rope=False
      - RoPE(quarter): rope=True, rope_fraction=0.25
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        intermediate_size: int = None,
        max_seq_len: int = 2048,
        rope: bool = True,
        rope_fraction: float = 1.0,
        canon_positions: str = "",
        canon_residual: bool = True,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                rope=rope,
                rope_fraction=rope_fraction,
                canon_positions=canon_positions,
                canon_residual=canon_residual,
            )
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return self.lm_head(h)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            return sum(p.numel() for name, p in self.named_parameters()
                       if 'embedding' not in name and 'lm_head' not in name)
        return sum(p.numel() for p in self.parameters())


def build_transformer(
    vocab_size: int,
    size: str = "8L512D",
    rope: bool = True,
    rope_fraction: float = 1.0,
    canon_positions: str = "",
    canon_residual: bool = True,
    max_seq_len: int = 2048,
    tie_weights: bool = False,
) -> TransformerLM:
    """
    Build a Transformer model matching the paper's size conventions.

    size: one of "8L512D", "12L512D", "8L768D", "12L768D"
      L = number of layers, D = hidden dimension
    """
    parts = size.upper().split("L")
    n_layers = int(parts[0])
    d_model = int(parts[1].replace("D", ""))
    n_heads = d_model // 64  # 64 dims per head
    # Intermediate size: paper uses 8d^2/3 * 3 = 8d (after both gate and up count)
    # For Llama, intermediate_size = 8d/3 * 2 ≈ 5.33d (rounded), but paper sets it so
    # MLP has 8d^2 params; with gate+up+down that's 2*d*I + I*d = 3*I*d = 8d^2 => I = 8d/3
    intermediate_size = int(8 * d_model / 3)
    # Round to multiple of 64
    intermediate_size = (intermediate_size + 63) // 64 * 64

    return TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        rope=rope,
        rope_fraction=rope_fraction,
        canon_positions=canon_positions,
        canon_residual=canon_residual,
        tie_weights=tie_weights,
    )

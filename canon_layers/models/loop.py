"""
Looped Language Model (LoopLM) from "Scaling Latent Reasoning via Looped Language Models" (Zhu et al., 2025).

Architecture:
  - A stack of N transformer layers is applied T_max times (recurrent steps)
  - All recurrent steps share the same weights
  - An exit gate at each step outputs an instantaneous exit probability λ_t
  - Training uses an entropy-regularized objective (Stage I)

Stage I loss (Eq. 4 from paper):
  L = sum_t [p_ϕ(t|x) * L(t)] - β * H(p_ϕ(·|x))

where p_ϕ is the exit distribution computed from the gate outputs.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock, RMSNorm


class LoopLM(nn.Module):
    """
    Looped Language Model with weight-shared transformer blocks and learned exit gates.

    The same n_layers transformer blocks are applied T_max times.
    Each recurrent step has its own exit gate (a small linear layer).
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
        T_max: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.T_max = T_max

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Shared blocks — applied T_max times per forward pass
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

        # One exit gate per recurrent step; uses last-token hidden state → scalar
        self.exit_gates = nn.ModuleList([
            nn.Linear(d_model, 1, bias=True)
            for _ in range(T_max)
        ])

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

    def _one_loop(self, h: torch.Tensor) -> torch.Tensor:
        """Apply all shared blocks once."""
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor, T: int = None) -> torch.Tensor:
        """
        Standard forward pass (T_max loops). Returns final logits.
        Compatible with the existing training/evaluation API.
        """
        T = T or self.T_max
        h = self.embedding(x)
        for _ in range(T):
            h = self._one_loop(h)
        return self.lm_head(self.norm_f(h))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _compute_exit_distribution(self, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Convert per-step instantaneous exit probabilities to a valid discrete distribution.

        Args:
            lambdas: (B, T) — λ_t ∈ (0,1) for each step and example
        Returns:
            p_exit: (B, T) — probability of exiting at each step (sums to 1)
        """
        B, T = lambdas.shape
        S = torch.ones(B, device=lambdas.device)  # survival probability S_0 = 1
        p_list = []
        for t in range(T):
            if t < T - 1:
                p_t = lambdas[:, t] * S          # p̃_t = λ_t · S_{t-1}
                S = S * (1.0 - lambdas[:, t])    # S_t = S_{t-1} · (1 - λ_t)
                p_list.append(p_t)
            else:
                p_list.append(S)                 # remaining mass → last step
        return torch.stack(p_list, dim=1)        # (B, T)

    def loop_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        beta: float = 0.1,
        T: int = None,
    ):
        """
        Stage I entropy-regularized training loss (Equation 4 of the paper).

          L = Σ_t [p_ϕ(t|x) · L(t)] − β · H(p_ϕ(·|x))

        Args:
            x:       (B, seq_len) input token ids
            targets: (B, seq_len) target token ids
            beta:    KL/entropy coefficient (uniform prior → entropy regularisation)
            T:       number of recurrent steps (default: T_max)

        Returns:
            total_loss: scalar tensor
            metrics:    dict with diagnostic scalars
        """
        T = T or self.T_max
        B, seq = targets.shape
        V = self.vocab_size

        # --- forward through all T loops ---
        h = self.embedding(x)
        all_logits, lambda_list = [], []

        for t in range(T):
            h = self._one_loop(h)

            # Logits at this step
            logits_t = self.lm_head(self.norm_f(h))
            all_logits.append(logits_t)

            # Exit gate: scalar per example, derived from last-token hidden state
            lambda_t = torch.sigmoid(
                self.exit_gates[t](h[:, -1, :])  # (B, 1)
            ).squeeze(-1)                         # (B,)
            lambda_list.append(lambda_t)

        lambdas = torch.stack(lambda_list, dim=1)  # (B, T)
        p_exit = self._compute_exit_distribution(lambdas)  # (B, T)

        # --- per-example, per-step cross-entropy loss ---
        per_step_losses = []
        for logits_t in all_logits:
            loss_t = F.cross_entropy(
                logits_t.reshape(-1, V),
                targets.reshape(-1),
                reduction='none',
            ).reshape(B, seq).mean(dim=1)  # (B,)
            per_step_losses.append(loss_t)

        per_step_losses = torch.stack(per_step_losses, dim=1)  # (B, T)

        # --- expected task loss (weighted by exit distribution) ---
        expected_loss = (p_exit * per_step_losses).sum(dim=1).mean()

        # --- entropy regularisation H(p) = -Σ p·log(p) ---
        eps = 1e-8
        entropy = -(p_exit * (p_exit + eps).log()).sum(dim=1).mean()

        total_loss = expected_loss - beta * entropy

        with torch.no_grad():
            steps = torch.arange(1, T + 1, device=x.device, dtype=torch.float)
            avg_exit = (p_exit * steps).sum(dim=1).mean().item()

        return total_loss, {
            'expected_loss': expected_loss.item(),
            'entropy': entropy.item(),
            'avg_exit_step': avg_exit,
        }

    # ------------------------------------------------------------------
    # Inference with early exit
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_with_exit(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ):
        """
        Inference pass with early exit.
        Terminates when CDF(t|x) = Σ_{j≤t} p_ϕ(j|x) ≥ threshold.

        Args:
            x:         (B, seq_len) input token ids
            threshold: exit when accumulated probability ≥ this value
        Returns:
            logits:     (B, seq_len, vocab) — logits at the exit step
            exit_steps: (B,) — which step (1-indexed) each example exited at
        """
        B = x.shape[0]
        h = self.embedding(x)

        S = torch.ones(B, device=x.device)
        CDF = torch.zeros(B, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)

        exit_logits = None
        exit_steps = torch.full((B,), self.T_max, dtype=torch.long, device=x.device)

        for t in range(self.T_max):
            h = self._one_loop(h)
            logits_t = self.lm_head(self.norm_f(h))

            lambda_t = torch.sigmoid(
                self.exit_gates[t](h[:, -1, :])
            ).squeeze(-1)

            if t < self.T_max - 1:
                p_t = lambda_t * S
                S = S * (1.0 - lambda_t)
                CDF = CDF + p_t
            else:
                CDF = torch.ones(B, device=x.device)  # always exit at last step

            newly_done = (CDF >= threshold) & ~done

            if exit_logits is None:
                exit_logits = logits_t
            else:
                mask = newly_done[:, None, None]
                exit_logits = torch.where(mask.expand_as(logits_t), logits_t, exit_logits)

            exit_steps = torch.where(
                newly_done,
                torch.tensor(t + 1, device=x.device),
                exit_steps,
            )
            done = done | newly_done
            if done.all():
                break

        return exit_logits, exit_steps

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            return sum(
                p.numel() for name, p in self.named_parameters()
                if 'embedding' not in name and 'lm_head' not in name
            )
        return sum(p.numel() for p in self.parameters())


def build_loop_transformer(
    vocab_size: int,
    size: str = "8L512D",
    rope: bool = True,
    rope_fraction: float = 1.0,
    canon_positions: str = "",
    canon_residual: bool = True,
    max_seq_len: int = 2048,
    tie_weights: bool = False,
    T_max: int = 4,
) -> LoopLM:
    """
    Build a LoopLM matching the paper's size conventions.

    size: e.g. "8L512D"  (L = layers per loop block, D = hidden dim)
    T_max: number of recurrent steps
    """
    parts = size.upper().split("L")
    n_layers = int(parts[0])
    d_model = int(parts[1].replace("D", ""))
    n_heads = d_model // 64
    intermediate_size = int(8 * d_model / 3)
    intermediate_size = (intermediate_size + 63) // 64 * 64

    return LoopLM(
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
        T_max=T_max,
    )

"""Vanilla Block-Sparse Featurizer (BSF). With group_size=1 it is a BatchTopK SAE.

Adapted from the Vanilla BSF in https://github.com/goodfire-ai/block-sparse-featurizer
(Fel, Kowal, Jacobs, Hazra, Bhalla et al., "Structuring Sparsity: Block-Sparse Featurizers
Capture Visual Concept Manifolds", 2025). Only the configuration used in this paper is kept:

- a free linear encoder and a free decoder (no bias, no per-step renormalization), initialized
  with unit-Frobenius Gaussian decoder blocks and encoder = decoder transpose, b_enc = 0;
- block selection by score_g = ||a_g||_2 * ||D_g||_F (scale-aware, because the decoder is free);
- BatchTopK: during training keep the N*k largest block scores of the batch; at inference keep
  blocks whose score exceeds a fixed threshold, an EMA (decay 0.99) of the training thresholds;
- AuxK: blocks not selected for `dead_window` steps reconstruct the (detached) residual with
  their top-k' codes, k' = min(k, #dead), weighted by (1/k) * min(#dead/k, 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def unit_blocks(W: torch.Tensor, n_groups: int, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """Scale each (group_size, d) block of W to unit Frobenius norm."""
    Wb = W.reshape(n_groups, group_size, -1)
    norm = Wb.flatten(1).norm(dim=1).clamp_min(eps).reshape(n_groups, 1, 1)
    return (Wb / norm).reshape(W.shape)


class VanillaBSF(nn.Module):
    """n_groups blocks of group_size latent coordinates over a d-dimensional input.

    encode: pre = x @ W_enc + b_enc, viewed as [N, G, b]; block g is active iff
            ||pre_g|| * ||W_dec_g||_F >= batchtopk_threshold; z_g = pre_g if active else 0.
    decode: x_hat = z @ W_dec.
    """

    # Above this many bytes the AuxK reconstruction uses a dense scatter + one matmul instead of
    # gathering an (N, k, b, d) copy of the selected decoder blocks (same sum up to fp rounding).
    AUX_GATHER_BUDGET_BYTES = 16 << 30

    def __init__(self, d: int, n_groups: int, group_size: int, k: int, bt_decay: float = 0.99):
        super().__init__()
        self.d, self.n_groups, self.group_size, self.k = d, n_groups, group_size, k
        self.bt_decay = float(bt_decay)
        self.aux_coef = 1.0 / max(k, 1)  # AuxK weight before the dead-fraction fade
        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))
        self.register_buffer("batchtopk_threshold", torch.zeros(()))

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)

    def scores(self, pre: torch.Tensor) -> torch.Tensor:
        dec_norm = self.W_dec.reshape(self.n_groups, self.group_size, -1).flatten(1).norm(dim=1)
        return pre.norm(dim=-1) * dec_norm

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference encoding with the fixed threshold. Returns z [N, G, b]."""
        pre = self.pre(x)
        mask = (self.scores(pre) >= self.batchtopk_threshold).to(pre.dtype)
        return pre * mask.unsqueeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(z.shape[0], -1) @ self.W_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def _aux_recon(self, pre, score, allowed, k):
        """Reconstruction from each row's top-k blocks among `allowed` [N, G]."""
        atoms = self.W_dec.reshape(self.n_groups, self.group_size, self.d)
        s = torch.where(allowed, score, score.new_tensor(-torch.inf))
        top = s.topk(k, dim=-1).indices                                   # [N, k]
        idx = top.unsqueeze(-1).expand(-1, -1, self.group_size)
        z_sel = pre.gather(1, idx)
        n = pre.shape[0]
        if n * k * self.group_size * self.d * pre.element_size() > self.AUX_GATHER_BUDGET_BYTES:
            z_dense = pre.new_zeros(n, self.n_groups, self.group_size).scatter(1, idx, z_sel)
            return z_dense.reshape(n, -1) @ atoms.reshape(-1, self.d)
        return torch.einsum("nkg,nkgd->nd", z_sel, atoms[top])

    def loss(self, x: torch.Tensor, target: torch.Tensor, dead_mask: torch.Tensor | None = None):
        """Training loss on a batch: BatchTopK selection (updates the EMA threshold), MSE to the
        clean `target`, plus the AuxK term when some blocks are dead. Returns (loss, info)."""
        pre = self.pre(x)
        score = self.scores(pre)
        n_keep = max(1, min(score.shape[0] * self.k, score.numel()))
        thr = torch.topk(score.reshape(-1), n_keep, sorted=True).values[-1]
        mask = (score >= thr).to(score.dtype)
        with torch.no_grad():
            self.batchtopk_threshold.mul_(self.bt_decay).add_(
                (1.0 - self.bt_decay) * thr.detach().to(self.batchtopk_threshold.dtype))
        x_hat = self.decode(pre * mask.unsqueeze(-1))
        recon = (target - x_hat).pow(2).mean()
        info = {"recon": recon.item(), "fired": mask.detach().bool().any(0)}
        total = recon
        if self.aux_coef > 0 and dead_mask is not None and bool(dead_mask.any()):
            allowed = dead_mask.unsqueeze(0) & ~mask.bool()
            if bool(allowed.any()):
                n_dead = int(dead_mask.sum())
                x_aux = self._aux_recon(pre, score, allowed, min(self.k, n_dead))
                aux = ((target - x_hat).detach() - x_aux).pow(2).mean()
                total = total + self.aux_coef * min(n_dead / max(self.k, 1), 1.0) * aux
                info["aux_dead"] = aux.item()
        return total, info


class DeadTracker:
    """Steps since each block was last selected; dead after `window` steps without selection."""

    def __init__(self, n_groups: int, window: int = 1000, device=None):
        self.window = int(window)
        self.steps = torch.zeros(n_groups, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, fired: torch.Tensor):
        self.steps = torch.where(fired, torch.zeros_like(self.steps), self.steps + 1)

    @property
    def dead_mask(self) -> torch.Tensor:
        return self.steps >= self.window

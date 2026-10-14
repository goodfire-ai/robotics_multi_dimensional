"""Load a featurizer checkpoint and encode / decode residual rows.

This file depends only on torch: import it or copy it. A checkpoint (model.pt from train.train) is a
torch.save dict, self-contained for encoding (torch.load(path, weights_only=True)):
  featurizer       variant="vanilla", d, n_groups (G), group_size (b; b = 1 is the TopK SAE), l0 (k),
                   selection_mode="batchtopk", state_dict {W_enc [d, G*b], b_enc [G*b], W_dec [G*b, d],
                   batchtopk_threshold []}
  input transform  postnorm_gamma [d] or None, rms_eps, pos_mean [P, d], mean_norm, fisher_W / fisher_Winv
                   [d, d] or None, scale
  other            norm_lo, norm_hi (do not encode rows outside them), model, layer, site, metrics,
                   format, format_version

    from train.encode import Featurizer
    f = Featurizer.load("model.pt", device="cuda")
    keep = f.in_window(h)                  # h: raw residual rows [N, d] of layer f.layer
    z = f.encode_raw(h[keep], pos[keep])   # [N, G, b] block codes (zero for inactive blocks)
    active = z.norm(dim=-1) > 0            # [N, G] which blocks fire
    h_hat = f.reconstruct_raw(h[keep], pos[keep])

Input transform for a raw row h at sequence position p (in this order):
  1. post-norm view (if postnorm_gamma is set): v = gamma * h / sqrt(mean(h^2) + rms_eps); v = h otherwise
  2. x' = (v - pos_mean[p]) / mean_norm      pos_mean = per-position mean minus the grand mean
                                              (zero offset for positions outside the table)
  3. Fisher lane only: x' = x' @ fisher_W
  4. x = scale * x'
Encoding: pre = x @ W_enc + b_enc viewed as [G, b]; block g is active iff
||pre_g|| * ||W_dec_g||_F >= batchtopk_threshold; decoding: x_hat = z @ W_dec.
"""
from __future__ import annotations

import torch


class InputTransform:
    """Raw residual rows -> featurizer inputs, and featurizer-space vectors -> residual space."""

    def __init__(self, ck: dict, device="cpu"):
        f32 = dict(device=device, dtype=torch.float32)
        g = ck.get("postnorm_gamma")
        self.gamma = None if g is None else torch.as_tensor(g).to(**f32)
        self.rms_eps = float(ck.get("rms_eps", 1e-6))
        pm = torch.as_tensor(ck["pos_mean"]).to(**f32)
        self.n_positions = pm.shape[0]
        self.pos_mean = torch.cat([pm, pm.new_zeros(1, pm.shape[1])])  # last row: unknown position
        self.mean_norm = float(ck["mean_norm"])
        W, Wi = ck.get("fisher_W"), ck.get("fisher_Winv")
        self.W = None if W is None else torch.as_tensor(W).to(**f32)
        self.Winv = None if Wi is None else torch.as_tensor(Wi).to(**f32)
        self.scale = float(ck["scale"])

    def offset(self, pos: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(pos, device=self.pos_mean.device).long()
        ok = (pos >= 0) & (pos < self.n_positions)
        return self.pos_mean[torch.where(ok, pos, torch.full_like(pos, self.n_positions))]

    def rms(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float().pow(2).mean(-1, keepdim=True) + self.rms_eps).sqrt()

    def __call__(self, h: torch.Tensor, pos: torch.Tensor, quantize: bool = False) -> torch.Tensor:
        """quantize=True reproduces the training-time fp16 round trips (after step 2 and 4)."""
        v = h.float()
        if self.gamma is not None:
            v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.rms_eps) * self.gamma
        x = (v - self.offset(pos)) / self.mean_norm
        if quantize:
            x = x.half().float()
        if self.W is not None:
            x = x @ self.W
        x = x * self.scale
        return x.half().float() if quantize else x

    def delta_to_residual(self, dx: torch.Tensor, h_ref: torch.Tensor) -> torch.Tensor:
        """Map a featurizer-space difference to a raw-residual difference at the operating point
        h_ref (the post-norm rms of h_ref is held fixed). Use it to steer: h + delta_to_residual(...)."""
        v = dx.float() / self.scale
        if self.Winv is not None:
            v = v @ self.Winv
        v = v * self.mean_norm
        if self.gamma is not None:
            v = v * self.rms(h_ref) / self.gamma
        return v

    def to_residual(self, x: torch.Tensor, pos: torch.Tensor, h_ref: torch.Tensor) -> torch.Tensor:
        """Map a featurizer-space vector (e.g. a reconstruction) back to the raw residual space."""
        v = x.float() / self.scale
        if self.Winv is not None:
            v = v @ self.Winv
        v = v * self.mean_norm + self.offset(pos)
        if self.gamma is not None:
            v = v * self.rms(h_ref) / self.gamma
        return v


class Featurizer:
    """Inference-only Vanilla BSF (group_size=1 is a TopK SAE) plus its input transform."""

    def __init__(self, ck: dict, device="cpu"):
        assert ck.get("variant", "vanilla") == "vanilla" and ck.get("selection_mode") == "batchtopk"
        sd = ck["state_dict"]
        self.d, self.n_groups, self.group_size, self.k = (int(ck["d"]), int(ck["n_groups"]),
                                                          int(ck["group_size"]), int(ck["l0"]))
        self.W_enc = sd["W_enc"].to(device, torch.float32)
        self.b_enc = sd["b_enc"].to(device, torch.float32)
        self.W_dec = sd["W_dec"].to(device, torch.float32)
        self.threshold = float(sd["batchtopk_threshold"])
        self.dec_norm = self.W_dec.reshape(self.n_groups, self.group_size, -1).flatten(1).norm(dim=1)
        self.transform = InputTransform(ck, device)
        self.norm_lo, self.norm_hi = float(ck["norm_lo"]), float(ck["norm_hi"])
        self.model, self.layer, self.metrics = ck.get("model"), ck.get("layer"), ck.get("metrics", {})

    @classmethod
    def load(cls, path: str, device="cpu") -> "Featurizer":
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    def in_window(self, h: torch.Tensor) -> torch.Tensor:
        """Rows whose raw L2 norm lies in the training window; do not encode the others."""
        n = h.float().norm(dim=-1)
        return (n >= self.norm_lo) & (n <= self.norm_hi)

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Block selection scores [N, G]; a block is active iff its score >= self.threshold."""
        return self.pre(x).norm(dim=-1) * self.dec_norm

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre(x)
        mask = (pre.norm(dim=-1) * self.dec_norm >= self.threshold).to(pre.dtype)
        return pre * mask.unsqueeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(z.shape[0], -1) @ self.W_dec

    def __call__(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def encode_raw(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.encode(self.transform(h, pos))

    def reconstruct_raw(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self(self.transform(h, pos))
        return self.transform.to_residual(x_hat, pos, h)


@torch.no_grad()
def evaluate(model, x_held, device, Winv: torch.Tensor | None = None) -> dict:
    """Held-out metrics of a featurizer (anything with forward(x) -> (x_hat, z)) on model-space
    inputs x_held [N, d] (array or tensor). R^2 = 1 - SS_res / SS_tot around the eval-set mean.
    With Winv (Fisher lane) also report R^2 after mapping inputs and errors back through Winv."""
    x_held = torch.as_tensor(x_held)
    n_groups, gs = model.n_groups, model.group_size
    chunk = min(100_000, max(4096, 750_000_000 // (n_groups * gs)))
    xm = x_held.float().mean(0, keepdim=True).to(device)
    ss_res = ss_tot = ss_res_u = ss_tot_u = l0 = 0.0
    fired = torch.zeros(n_groups, dtype=torch.bool, device=device)
    for i in range(0, len(x_held), chunk):
        xb = x_held[i:i + chunk].to(device).float()
        x_hat, z = model(xb)
        ss_res += float((xb - x_hat).pow(2).sum())
        ss_tot += float((xb - xm).pow(2).sum())
        if Winv is not None:
            ss_res_u += float(((xb - x_hat) @ Winv).pow(2).sum())
            ss_tot_u += float(((xb - xm) @ Winv).pow(2).sum())
        act = z.norm(dim=-1) > 1e-6
        l0 += float(act.float().sum())
        fired |= act.any(0)
    out = dict(heldout_r2=1.0 - ss_res / max(ss_tot, 1e-12), heldout_l0=l0 / max(len(x_held), 1),
               dead=int((~fired).sum()), n_eval=int(len(x_held)))
    if Winv is not None:
        out["heldout_r2_unwhitened"] = 1.0 - ss_res_u / max(ss_tot_u, 1e-12)
    return out

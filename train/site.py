"""Reading an activation site and the CPU-side input transform.

Activation site format (one directory per layer, e.g. sites/molmoact2_L13/; collection code is not
part of this repository, any code that writes this layout works):
  part{P}_x.bin        raw fp16 residual rows [n, d], C-order, no normalization
  part{P}_ep.bin       int32 per row: episode index (TRAIN = ep % 20 != 0, held-out = ep % 20 == 0)
  part{P}_pos.bin      int32 per row: sequence position in the prefix (indexes the pos_mean table)
  part{P}_lab.bin      int32 per row: 1 = image patch, 0 = text or state token
  part{P}_meta.json    required {part_id, n_tokens, d}; optional {model, layer, site, norm_lo, norm_hi}
  window.json          {norm_lo, norm_hi}: the raw-norm window used at capture (used when the metas
                       lack it); rows outside it are never encoded
  postnorm_gamma.npy   float32 [d], the input-RMSNorm weight of block L+1 (post-norm view
                       v = gamma * h / sqrt(mean(h^2) + eps)); only for `finalize --postnorm-gamma site`
Rows of one frame are consecutive in increasing position order. Rows are addressed by a global index:
parts concatenated in part-id order.
"""
from __future__ import annotations

import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FP16_MAX = np.float16(65504)


def finite(x: np.ndarray) -> np.ndarray:
    """Clamp fp16-saturated values (stored as +-inf) to +-65504; finite input is returned as is."""
    if np.isfinite(x).all():
        return x
    return np.nan_to_num(x, nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)


def eval_split(ep: np.ndarray, eval_ep_mod: int = 20, eval_cap: int = 1_000_000):
    """Train rows = episodes with ep % eval_ep_mod != 0. Eval rows = the other episodes' rows,
    thinned to `eval_cap` evenly spaced rows. Returns (train_mask [N] bool, eval_rows int64)."""
    train_mask = (ep % eval_ep_mod) != 0
    eval_rows = np.nonzero(~train_mask)[0]
    if len(eval_rows) > eval_cap:
        eval_rows = eval_rows[np.unique(np.linspace(0, len(eval_rows) - 1, eval_cap).astype(np.int64))]
    return train_mask, eval_rows


def postnorm_np(x32: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    """Post-norm view gamma * x / sqrt(mean(x^2) + eps), fp32 (the next block's input RMSNorm)."""
    rms = np.sqrt((x32 ** 2).mean(axis=1, keepdims=True) + np.float32(eps))
    return x32 * (gamma[None, :] / rms)


class Site:
    def __init__(self, site_dir: str):
        self.dir = site_dir
        metas = sorted(glob.glob(os.path.join(site_dir, "part*_meta.json")),
                       key=lambda p: int(re.search(r"part(\d+)_meta", p).group(1)))
        self.parts = []
        for mp in metas:
            m = json.load(open(mp))
            pid, n, d = int(m["part_id"]), int(m["n_tokens"]), int(m["d"])
            x_path = os.path.join(site_dir, f"part{pid}_x.bin")
            n = min(n, os.path.getsize(x_path) // (2 * d))  # a part still being written
            if n > 0:
                self.parts.append(dict(pid=pid, n=n, d=d, meta=m, x_path=x_path))
        assert self.parts, f"no parts under {site_dir}"
        self.meta = self.parts[0]["meta"]
        self.d = self.parts[0]["d"]
        self.offsets = np.cumsum([0] + [p["n"] for p in self.parts])
        self.N = int(self.offsets[-1])

        def side(name):
            return np.concatenate([np.fromfile(os.path.join(site_dir, f"part{p['pid']}_{name}.bin"),
                                               dtype=np.int32, count=p["n"]) for p in self.parts])
        self.ep, self.pos, self.lab = side("ep"), side("pos"), side("lab")
        win_path = os.path.join(site_dir, "window.json")
        win = json.load(open(win_path)) if os.path.exists(win_path) else {}
        self.norm_lo = float(self.meta.get("norm_lo", win.get("norm_lo", win.get("lo"))))
        self.norm_hi = float(self.meta.get("norm_hi", win.get("norm_hi", win.get("hi"))))
        self._fds = None

    def read(self, part_idx: int, start: int, stop: int) -> np.ndarray:
        """Rows [start, stop) of one part as fp16 [stop - start, d] (thread-safe pread)."""
        if self._fds is None:
            self._fds = [os.open(p["x_path"], os.O_RDONLY) for p in self.parts]
        want, off = (stop - start) * self.d * 2, start * self.d * 2
        chunks, got = [], 0
        while got < want:
            b = os.pread(self._fds[part_idx], want - got, off + got)
            if not b:
                raise IOError(f"short read in part {self.parts[part_idx]['pid']}")
            chunks.append(b)
            got += len(b)
        return finite(np.frombuffer(b"".join(chunks), np.float16).reshape(stop - start, self.d))

    def gather(self, rows: np.ndarray, threads: int = 8) -> np.ndarray:
        """Random-access read of rows by global index -> fp16 [len(rows), d]."""
        rows = np.asarray(rows, np.int64)
        out = np.empty((len(rows), self.d), np.float16)
        order = np.argsort(rows, kind="stable")
        sr = rows[order]

        def one(pi):
            a, b = np.searchsorted(sr, self.offsets[pi]), np.searchsorted(sr, self.offsets[pi + 1])
            if a < b:
                mm = np.memmap(self.parts[pi]["x_path"], np.float16, "r", shape=(self.parts[pi]["n"], self.d))
                out[order[a:b]] = mm[sr[a:b] - self.offsets[pi]]

        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, range(len(self.parts))))
        return finite(out)

    def chunk_specs(self, chunk_rows: int):
        """(part_idx, start, stop) chunks, visiting the parts round-robin."""
        per_part = [[(pi, i, min(i + chunk_rows, p["n"])) for i in range(0, p["n"], chunk_rows)]
                    for pi, p in enumerate(self.parts)]
        specs, depth = [], max(len(pp) for pp in per_part)
        for j in range(depth):
            specs += [pp[j] for pp in per_part if j < len(pp)]
        return specs


def normalize_np(x16: np.ndarray, pos: np.ndarray, pos_mean: np.ndarray, mean_norm: float,
                 gamma: np.ndarray | None, eps: float) -> np.ndarray:
    """Raw fp16 rows -> fp16 normalized rows ((postnorm(h) - pos_mean[p]) / mean_norm).
    The CPU twin of encode.InputTransform (used for the scale sample and the eval split)."""
    x = x16.astype(np.float32)
    if gamma is not None:
        x = postnorm_np(x, gamma, eps)
    return ((x - pos_mean[pos]) / float(mean_norm)).astype(np.float16)  # python float keeps fp32

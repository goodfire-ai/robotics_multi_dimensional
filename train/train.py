"""Train Vanilla BSFs and TopK SAEs (block size b = 1) on a finalized cache; all configs share one GPU
and one pass over the data.

    # the paper's canonical pair (default): BSF G=4d/3, b=3, k=21 and SAE G=4d, b=1, k=63
    python -m train.train --cache caches/molmoact2_L13 --out runs/molmoact2_L13
    # Fisher-weighted BSF: the same trainer on whitened inputs (train.fisher)
    python -m train.train --cache caches/molmoact2_L13 --out runs/molmoact2_L13_fisher \
        --fisher-w fisher/L13/fisher_W_L13.npz --configs 4d/3:3:21

--configs takes G:b:k triplets; G is an integer or a multiple of d ("4d", "4d/3", rounded).
Recipe (the paper's): per epoch the TRAIN rows are read in 16,384-row chunks (parts round-robin,
chunk order reshuffled every epoch) into an 8M-row shuffle buffer; 16,384-row batches drawn from it
are transformed on the GPU (post-norm view, position offset, / mean_norm, optional Fisher whitening,
* scale with E||x||^2 = d) and every config takes one Adam step (lr 1e-4) on each batch, with
denoising noise of per-token std 0.1 ||x|| / sqrt(d). The batch sequence depends only on the stream
settings, so configs trained in different jobs see identical data; init and noise follow --seed.
Output per config: OUT/G{G}_b{b}_k{k}_s{seed}/{model.pt, metrics.json, train_log.json}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from train.bsf import DeadTracker, VanillaBSF
from train.encode import InputTransform, evaluate
from train.site import Site, eval_split, normalize_np


@dataclass
class Config:
    n_groups: int
    group_size: int
    k: int
    seed: int = 0

    @property
    def name(self) -> str:
        return f"G{self.n_groups}_b{self.group_size}_k{self.k}_s{self.seed}"


class ShuffleBuffer:
    """Seeded streaming shuffle: fill to capacity, then for every batch sample `bs` random slots,
    emit them and refill the slots with the next stream rows. flush() permutes what is left and
    emits the full batches (the tail is dropped)."""

    def __init__(self, capacity: int, d: int, bs: int, seed: int):
        self.cap, self.bs = max(int(capacity), int(bs)), int(bs)
        self.x = np.empty((self.cap, d), np.float16)
        self.pos = np.empty(self.cap, np.int32)
        self.ridx = np.empty(self.cap, np.int64)
        self.fill = 0
        self.rng = np.random.default_rng(seed)
        self.staged: list = []

    def _take(self, n):
        out = []
        while n > 0:
            seg = self.staged[0]
            k = min(n, len(seg[0]))
            out.append(tuple(a[:k] for a in seg))
            if k == len(seg[0]):
                self.staged.pop(0)
            else:
                self.staged[0] = tuple(a[k:] for a in seg)
            n -= k
        return out[0] if len(out) == 1 else tuple(np.concatenate(c) for c in zip(*out))

    def push(self, x, pos, ridx):
        if len(x) == 0:
            return
        self.staged.append((x, pos, ridx))
        while True:
            avail = sum(len(s[0]) for s in self.staged)
            if self.fill < self.cap:
                k = min(self.cap - self.fill, avail)
                if k == 0:
                    return
                sl = slice(self.fill, self.fill + k)
                self.x[sl], self.pos[sl], self.ridx[sl] = self._take(k)
                self.fill += k
                continue
            if avail < self.bs:
                return
            idx = self.rng.choice(self.cap, self.bs, replace=False)
            yield self.x[idx], self.pos[idx], self.ridx[idx]
            self.x[idx], self.pos[idx], self.ridx[idx] = self._take(self.bs)

    def flush(self):
        rem = [(self.x[:self.fill], self.pos[:self.fill], self.ridx[:self.fill])] + self.staged
        x, pos, ridx = (np.concatenate(c) for c in zip(*rem))
        self.staged, self.fill = [], 0
        perm = self.rng.permutation(len(x))
        for i in range(0, len(x) - self.bs + 1, self.bs):
            sl = perm[i:i + self.bs]
            yield x[sl], pos[sl], ridx[sl]


class StreamData:
    """A finalized cache: the site, the input transform (incl. the scale), the eval split and
    the per-epoch batch stream."""

    def __init__(self, cache: str, fisher_w: str | None = None, stream_seed: int = 1,
                 buffer_rows: int = 8_000_000, chunk_rows: int = 16_384, batch: int = 16_384,
                 readers: int = 6, prefetch: int = 6, threads: int = 8):
        t0 = time.time()
        self.meta = json.load(open(os.path.join(cache, "cache_meta.json")))
        self.site = site = Site(self.meta["site_dir"])
        self.d = d = site.d
        self.pos_mean = np.load(os.path.join(cache, "pos_mean.npz"))["pos_mean"].astype(np.float32)
        self.gamma = (np.load(os.path.join(cache, "postnorm_gamma.npy")).astype(np.float32)
                      if self.meta["postnorm"] else None)
        self.mean_norm = float(self.meta["mean_norm"])
        self.rms_eps = float(self.meta["rms_eps"])
        self.W = self.Winv = None
        self.fisher_meta = None
        if fisher_w:
            fz = np.load(fisher_w)
            self.W, self.Winv = fz["W"].astype(np.float32), fz["Winv"].astype(np.float32)
            assert self.W.shape == (d, d)
            self.fisher_meta = {k: fz[k].item() for k in ("delta", "gnorm_lo", "gnorm_hi", "n_tokens",
                                                          "mean_norm_used") if k in fz.files}
            used = self.fisher_meta.get("mean_norm_used", self.mean_norm)
            assert abs(used - self.mean_norm) < 1e-6 * self.mean_norm, \
                f"fisher_W was built for mean_norm {used}, cache has {self.mean_norm}"
        self.stream_seed, self.buffer_rows, self.chunk_rows, self.batch = (stream_seed, buffer_rows,
                                                                           chunk_rows, batch)
        self.readers, self.prefetch = readers, prefetch
        self.train_mask, self.eval_rows = eval_split(site.ep, self.meta["eval_ep_mod"], self.meta["eval_cap"])
        self.n_train = int(self.train_mask.sum())
        assert self.n_train == self.meta["n_train"], "site changed since finalize"

        def model_space(rows):  # raw rows -> fp32 normalized (and whitened) rows
            xf = normalize_np(site.gather(rows, threads), site.pos[rows], self.pos_mean,
                              self.mean_norm, self.gamma, self.rms_eps).astype(np.float32)
            return xf if self.W is None else xf @ self.W

        # scale so that E||x||^2 = d, estimated on a fixed 200k-row sample of the TRAIN rows
        sub = np.random.default_rng(0).choice(self.n_train, min(200_000, self.n_train), replace=False)
        xf = model_space(np.nonzero(self.train_mask)[0][sub])
        self.scale = float(np.sqrt(d) / np.sqrt((xf ** 2).sum(axis=1).mean()))
        del xf
        xn = np.concatenate([normalize_np(site.gather(r, threads), site.pos[r], self.pos_mean, self.mean_norm,
                                          self.gamma, self.rms_eps) for r in np.array_split(self.eval_rows, 10)])
        xf = xn.astype(np.float32) if self.W is None else xn.astype(np.float32) @ self.W
        self.x_held = (xf * self.scale).astype(np.float16)
        del xn, xf
        print(f"[stream] {cache}: N={site.N:,} train={self.n_train:,} eval={len(self.x_held):,} d={d} "
              f"mean_norm={self.mean_norm:.4f} scale={self.scale:.4f} postnorm={self.gamma is not None} "
              f"fisher={self.W is not None} ({time.time() - t0:.0f}s)", flush=True)

    def transform_fields(self) -> dict:
        """The input-transform fields of a checkpoint (see train/encode.py)."""
        t = lambda a: None if a is None else torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
        return dict(postnorm_gamma=t(self.gamma), rms_eps=self.rms_eps, pos_mean=t(self.pos_mean),
                    mean_norm=self.mean_norm, fisher_W=t(self.W), fisher_Winv=t(self.Winv),
                    scale=self.scale)

    def _read_ahead(self, specs):
        with ThreadPoolExecutor(self.readers) as ex:
            futs = deque(ex.submit(self.site.read, *s) for s in specs[:self.prefetch])
            for k, spec in enumerate(specs):
                x = futs.popleft().result()
                if k + self.prefetch < len(specs):
                    futs.append(ex.submit(self.site.read, *specs[k + self.prefetch]))
                yield spec, x

    def epoch_batches(self, epoch: int, max_batches: int | None = None):
        """Yields (x fp16 [B, d] raw rows, pos int32 [B], row index int64 [B]) for one epoch."""
        specs = self.site.chunk_specs(self.chunk_rows)
        order = np.random.default_rng(self.stream_seed * 31_337 + epoch).permutation(len(specs))
        specs = [specs[i] for i in order]
        buf = ShuffleBuffer(min(self.buffer_rows, self.n_train), self.d, self.batch,
                            seed=self.stream_seed * 100_003 + epoch)
        n_out = 0

        def stream():
            for (pi, i, j), x in self._read_ahead(specs):
                gs = int(self.site.offsets[pi]) + i
                m = self.train_mask[gs:gs + j - i]
                if m.any():
                    yield from buf.push(x[m], self.site.pos[gs:gs + j - i][m],
                                        np.arange(gs, gs + j - i, dtype=np.int64)[m])
            yield from buf.flush()

        for b in stream():
            yield b
            n_out += 1
            if max_batches and n_out >= max_batches:
                return


class ConfigRun:
    """One featurizer config: model, Adam, dead-block tracker and its own noise generator."""

    def __init__(self, cfg: Config, d: int, device, lr: float = 1e-4, snr: float = 0.1):
        self.cfg = cfg
        torch.default_generator.manual_seed(cfg.seed)  # model init is drawn on the CPU generator
        self.model = VanillaBSF(d, cfg.n_groups, cfg.group_size, cfg.k).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.tracker = DeadTracker(cfg.n_groups, 1000, device)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(cfg.seed)
        self.snr, self.sqrt_d = snr, math.sqrt(d)
        self.log = {"loss": [], "recon": [], "aux_dead": [], "epochs": []}
        self.epoch_start = 0  # index into log["loss"] where the current epoch began

    def step(self, x: torch.Tensor):
        """Denoising step: the input gets Gaussian noise of per-token std snr * ||x|| / sqrt(d),
        the target is the clean x."""
        sigma = self.snr * x.norm(dim=-1, keepdim=True) / self.sqrt_d
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=self.rng)
        loss, info = self.model.loss(x + sigma * noise, x, self.tracker.dead_mask)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.tracker.update(info["fired"])
        self.log["loss"].append(float(loss.item()))
        self.log["recon"].append(info["recon"])
        self.log["aux_dead"].append(info.get("aux_dead", 0.0))


def train(data: StreamData, configs: list[Config], out_dir: str, epochs: int = 10, device="cuda",
          lr: float = 1e-4, snr: float = 0.1, max_batches: int | None = None) -> dict:
    """Train every config on the shared stream and save one checkpoint directory per config."""
    torch.set_num_threads(8)
    transform = InputTransform(data.transform_fields(), device)
    runs = [ConfigRun(c, data.d, device, lr, snr) for c in configs]
    q: queue.Queue = queue.Queue(maxsize=4)

    def producer():
        try:
            for ep in range(1, epochs + 1):
                for b in data.epoch_batches(ep, max_batches):
                    q.put(("batch", b))
                q.put(("epoch_end", ep))
            q.put(("stop", None))
        except Exception as e:  # surface reader errors in the training loop
            q.put(("error", e))

    threading.Thread(target=producer, daemon=True).start()
    t0, n_batches = time.time(), 0
    held_sub = torch.as_tensor(data.x_held[:20_000])
    while True:
        kind, item = q.get()
        if kind == "error":
            raise item
        if kind == "stop":
            break
        if kind == "batch":
            x16, pos, _ridx = item
            x = transform(torch.from_numpy(x16).to(device),
                          torch.from_numpy(pos.astype(np.int64)).to(device), quantize=True)
            for r in runs:
                r.step(x)
            n_batches += 1
            if n_batches % 500 == 0:
                print(f"[train] {n_batches} batches, {n_batches / (time.time() - t0):.2f} b/s", flush=True)
            continue
        for r in runs:  # epoch end: short summary on 20k held-out rows
            r.model.eval()
            m = evaluate(r.model, held_sub, device)
            r.model.train()
            steps = slice(r.epoch_start, len(r.log["loss"]))
            r.epoch_start = len(r.log["loss"])
            n = max(1, steps.stop - steps.start)
            e = dict(epoch=item, loss=sum(r.log["loss"][steps]) / n, recon=sum(r.log["recon"][steps]) / n,
                     r2_20k=m["heldout_r2"], l0_20k=m["heldout_l0"], dead_blocks=int(r.tracker.dead_mask.sum()))
            r.log["epochs"].append(e)
            print(f"[train] {r.cfg.name} epoch {item}/{epochs}: loss={e['loss']:.4f} R2(20k)={e['r2_20k']:.4f} "
                  f"L0={e['l0_20k']:.1f} dead={e['dead_blocks']}/{r.cfg.n_groups}", flush=True)

    train_s = time.time() - t0
    results = {}
    Winv = None if data.Winv is None else torch.from_numpy(data.Winv).to(device)
    for r in runs:
        r.model.eval()
        metrics = evaluate(r.model, data.x_held, device, Winv)
        metrics.update(n_train=data.n_train, epochs=epochs, batches=len(r.log["loss"]), lr=lr, snr=snr,
                       seed=r.cfg.seed, stream_seed=data.stream_seed, buffer_rows=data.buffer_rows,
                       chunk_rows=data.chunk_rows, batch=data.batch, train_s=round(train_s, 1),
                       batchtopk_threshold=float(r.model.batchtopk_threshold), fisher=data.fisher_meta)
        results[r.cfg.name] = save_checkpoint(r, data, os.path.join(out_dir, r.cfg.name), metrics)
        print(f"[train] {r.cfg.name}: heldout R2={metrics['heldout_r2']:.4f} "
              f"L0={metrics['heldout_l0']:.2f} dead={metrics['dead']}", flush=True)
    return results


def save_checkpoint(run: ConfigRun, data: StreamData, outdir: str, metrics: dict) -> dict:
    os.makedirs(outdir, exist_ok=True)
    m = run.model
    ck = dict(format="vla-bsf/featurizer", format_version=1, variant="vanilla", d=m.d,
              n_groups=m.n_groups, group_size=m.group_size, l0=m.k, selection_mode="batchtopk",
              state_dict={k: v.detach().cpu() for k, v in m.state_dict().items()},
              **data.transform_fields(), norm_lo=data.meta["norm_lo"], norm_hi=data.meta["norm_hi"],
              model=data.meta.get("model"), layer=data.meta.get("layer"), site=data.meta.get("site"),
              metrics=metrics)
    torch.save(ck, os.path.join(outdir, "model.pt.tmp"))
    os.replace(os.path.join(outdir, "model.pt.tmp"), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(outdir, "train_log.json"), "w") as f:
        json.dump(run.log, f)
    return metrics


def n_groups_of(spec: str, d: int) -> int:
    m = re.fullmatch(r"([\d.]*)d(?:/([\d.]+))?", spec)
    return int(spec) if m is None else round(float(m.group(1) or 1) * d / float(m.group(2) or 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="cache directory written by train.finalize")
    ap.add_argument("--out", required=True, help="one subdirectory per config is written here")
    ap.add_argument("--configs", nargs="+", default=["4d/3:3:21", "4d:1:63"], help="G:b:k per featurizer")
    ap.add_argument("--fisher-w", help="fisher_W_L{layer}.npz from `train.fisher whiten`")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch (smoke tests)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.cache, "cache_meta.json")))["d"]
    configs = [Config(n_groups_of(g, d), int(b), int(k), a.seed) for g, b, k in (c.split(":") for c in a.configs)]
    print("[train] configs: " + ", ".join(c.name for c in configs), flush=True)
    train(StreamData(a.cache, a.fisher_w), configs, a.out, a.epochs, a.device, max_batches=a.max_batches)


if __name__ == "__main__":
    main()

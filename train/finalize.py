"""Finalize an activation site into a training cache (CPU; bound by filesystem throughput).

    python -m train.finalize --site sites/molmoact2_L13 --out caches/molmoact2_L13

Two streaming passes over the TRAIN rows (episodes with ep % 20 != 0), in the post-norm view
v = gamma * h / rms(h) when a gamma is given (raw residual otherwise):
  1. fp64 per-position sums -> grand mean and the per-position offset table
     pos_mean[p] = mean(v | pos = p) - grand_mean  (zero for positions never seen in training);
  2. mean_norm = mean over TRAIN rows of ||v - pos_mean[p]||_2 (fp64 accumulation).
The grand mean is deliberately NOT subtracted by the input transform: featurizers are trained on
(v - pos_mean[p]) / mean_norm, and the always-on mean direction is carried by the dictionary.

Output (the cache directory):
  cache_meta.json      site path, model/layer, row counts, norm window, mean_norm, eval split rule
  pos_mean.npz         pos_mean [P, d] f32, grand_mean [d] f64, counts [P] i64
  postnorm_gamma.npy   [d] f32, only for post-norm caches
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from train.site import Site, eval_split, postnorm_np

SUM_CHUNK = 262_144    # rows per reduction chunk in pass 1
NORM_CHUNK = 16_384    # rows per partial sum in pass 2 (partials are added in round-robin order)


def _position_sums(site, pi, train_mask, P, gamma, eps):
    part, off = site.parts[pi], site.offsets[pi]
    sums = np.zeros((P, site.d), np.float64)
    counts = np.zeros(P, np.int64)
    for i in range(0, part["n"], SUM_CHUNK):
        j = min(i + SUM_CHUNK, part["n"])
        m = train_mask[off + i:off + j]
        if not m.any():
            continue
        x = site.read(pi, i, j).astype(np.float32)[m]
        pp = site.pos[off + i:off + j][m]
        if gamma is not None:
            x = postnorm_np(x, gamma, eps)
        order = np.argsort(pp, kind="stable")
        xs, ps = x[order], pp[order]
        bounds = np.r_[0, 1 + np.nonzero(np.diff(ps))[0]]
        sums[ps[bounds]] += np.add.reduceat(xs, bounds, axis=0, dtype=np.float64)
        counts += np.bincount(pp, minlength=P)
    return sums, counts


def _norm_partials(site, pi, train_mask, pos_mean, gamma, eps):
    part, off = site.parts[pi], site.offsets[pi]
    out = {}
    for i in range(0, part["n"], SUM_CHUNK):
        x = site.read(pi, i, min(i + SUM_CHUNK, part["n"]))
        for a in range(i, i + len(x), NORM_CHUNK):
            b = min(a + NORM_CHUNK, i + len(x))
            m = train_mask[off + a:off + b]
            if not m.any():
                continue
            xb = x[a - i:b - i][m].astype(np.float32)
            if gamma is not None:
                xb = postnorm_np(xb, gamma, eps)
            r = xb - pos_mean[site.pos[off + a:off + b][m]]
            out[(pi, a)] = (float(np.linalg.norm(r, axis=1).sum(dtype=np.float64)), len(r))
    return out


def finalize(site_dir: str, out: str, gamma_path: str | None = None, rms_eps: float = 1e-6,
             eval_ep_mod: int = 20, eval_cap: int = 1_000_000, threads: int = 8) -> dict:
    t0 = time.time()
    site = Site(site_dir)
    d, P = site.d, int(site.pos.max()) + 1
    train_mask, eval_rows = eval_split(site.ep, eval_ep_mod, eval_cap)
    n_train = int(train_mask.sum())
    gamma = None
    if gamma_path:
        gamma = np.load(gamma_path).astype(np.float32)
        assert gamma.shape == (d,), f"gamma shape {gamma.shape} != ({d},)"
    print(f"[finalize] {site_dir}: {len(site.parts)} parts, N={site.N:,} d={d} P={P} "
          f"train={n_train:,} eval={len(eval_rows):,} postnorm={gamma is not None}", flush=True)

    with ThreadPoolExecutor(threads) as ex:  # pass 1: per-position sums, merged in part order
        res = list(ex.map(lambda pi: _position_sums(site, pi, train_mask, P, gamma, rms_eps),
                          range(len(site.parts))))
    sums = np.zeros((P, d), np.float64)
    counts = np.zeros(P, np.int64)
    for s, c in res:
        sums += s
        counts += c
    assert int(counts.sum()) == n_train
    grand = sums.sum(axis=0) / max(1, counts.sum())
    live = counts > 0
    pos_mean = np.zeros((P, d), np.float32)
    pos_mean[live] = (sums[live] / counts[live, None] - grand).astype(np.float32)
    print(f"[finalize] pass 1 done ({time.time() - t0:.0f}s): {int(live.sum())} live positions", flush=True)

    with ThreadPoolExecutor(threads) as ex:  # pass 2: mean norm of the position-adjusted rows
        partials = {}
        for r in ex.map(lambda pi: _norm_partials(site, pi, train_mask, pos_mean, gamma, rms_eps),
                        range(len(site.parts))):
            partials.update(r)
    s, n = 0.0, 0
    for pi, a, _ in site.chunk_specs(NORM_CHUNK):
        if (pi, a) in partials:
            s += partials[(pi, a)][0]
            n += partials[(pi, a)][1]
    assert n == n_train
    mean_norm = s / n
    print(f"[finalize] pass 2 done ({time.time() - t0:.0f}s): mean_norm={mean_norm:.6f}", flush=True)

    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "pos_mean.npz"), pos_mean=pos_mean, grand_mean=grand, counts=counts)
    if gamma is not None:
        np.save(os.path.join(out, "postnorm_gamma.npy"), gamma)
    lab_t = site.lab[train_mask]
    meta = dict(site_dir=os.path.abspath(site_dir), model=site.meta.get("model"),
                layer=site.meta.get("layer"), site=site.meta.get("site"), d=d, n_positions=P,
                n_rows=site.N, n_train=n_train, n_eval_total=int(site.N - n_train),
                n_eval=int(len(eval_rows)), n_train_img=int((lab_t == 1).sum()),
                n_train_txt=int((lab_t == 0).sum()), n_episodes=int(len(np.unique(site.ep))),
                eval_ep_mod=eval_ep_mod, eval_cap=eval_cap, norm_lo=site.norm_lo,
                norm_hi=site.norm_hi, mean_norm=mean_norm, postnorm=gamma is not None,
                rms_eps=rms_eps, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(out, "cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[finalize] wrote {out} ({time.time() - t0:.0f}s)", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="activation site directory (format in train/site.py)")
    ap.add_argument("--out", required=True, help="cache directory to write")
    ap.add_argument("--postnorm-gamma", default="site",
                    help="post-norm gains: 'site' (SITE/postnorm_gamma.npy, see train/site.py), a [d] .npy "
                         "path, or 'none' to train on the raw residual")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    gamma = None if a.postnorm_gamma == "none" else (
        os.path.join(a.site, "postnorm_gamma.npy") if a.postnorm_gamma == "site" else a.postnorm_gamma)
    pj = os.path.join(a.site, "postnorm.json")
    eps = float(json.load(open(pj))["rms_eps"]) if gamma and os.path.exists(pj) else 1e-6
    finalize(a.site, a.out, gamma, eps, threads=a.threads)


if __name__ == "__main__":
    main()

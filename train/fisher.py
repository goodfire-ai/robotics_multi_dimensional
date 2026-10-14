"""Fisher-weighted BSFs: pooled second moment J of the action log-likelihood gradient at a residual
site, in the featurizer's input coordinates, and the whitening matrix W_f built from it.

    python -m train.fisher bounds --frames F --layers 13 --caches caches/molmoact2_L13 --out fisher/L13
    python -m train.fisher accum  --frames F --layers 13 --caches caches/molmoact2_L13 --out fisher/L13 --shard 0 --nshards 8
    python -m train.fisher whiten --out fisher/L13 --layer 13      # -> fisher/L13/fisher_W_L13.npz
    python -m train.train --cache caches/molmoact2_L13 --fisher-w fisher/L13/fisher_W_L13.npz --configs 4d/3:3:21

F is a frames directory, e.g. every 8th frame of the harvest episodes (episode ids = the site's;
TRAIN episodes are ep % 20 != 0). bounds/accum need the MolmoAct2 environment (transformers 4.57,
trust_remote_code); whiten needs only torch and numpy.

Frames directory format:
  ep{E:04d}.json                  {episode, instruction, grid (14), cams (["ext", "wrist"]),
                                   token_layout ({cam: [start, end)} of its image tokens),
                                   id2text ({token id: text}), queries: [{q, frame, state, ids}]}
                                   (q numbers the recorded frames, frame is the DROID frame index,
                                   state is the 8-D robot state, ids the prompt token ids, -1 for patches)
  frames/ep{E:04d}/q{Q:03d}_{cam}.jpg   the RGB frame per query and camera
Outputs in --out: bounds.json (winsorization bounds of ||g|| per layer), shard{s}.npz (J_L{L} = sum g g^T
in float64, n_rows_L{L}, mean_norm_L{L}, done_eps) and fisher_W_L{L}.npz (W, Winv, J, evals_raw,
evals_reg, delta, n_tokens, gnorm_lo, gnorm_hi, mean_norm_used).

Per query frame: the model decodes its own FAST action tokens (greedy, predict_action discrete mode);
one teacher-forced forward over [prompt || action tokens] with a zero additive tap on every requested
block gives m = sum_k log p(action_k); one backward gives g_L = dm/dh_L for all layers. Prompt rows
(image patches, then text) whose fp16 residual norm is inside the layer's window are mapped to the
featurizer's coordinates with each token's rms held fixed, g_n = g * mean_norm * r / gamma.
  bounds  every 20th TRAIN episode: ||g_n|| percentiles [p0.5, p99.5] per layer -> OUT/bounds.json
  accum   TRAIN episodes of shard s: winsorize ||g_n|| into the bounds (direction kept) and accumulate
          J_sum += g_n^T g_n (fp32 GEMM, fp64 sum) -> OUT/shard{s}.npz (resumable)
  whiten  J = sum(J_sum) / N = U diag(lam) U^T;  W_f = U diag(sqrt(lam / mean(lam) + delta)) U^T and
          W_f^-1 likewise. Training on x' W_f weights the error by J / mean(lam) + delta I; delta = 0.25
          keeps a plain-MSE share of delta / (1 + delta) = 20% of the objective.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import torch

FP16_MAX = 65504.0


# ------------------------------------------------------------------ MolmoAct2 (inference helpers)
def load_model(model_id: str, device: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor
    local = snapshot_download(model_id)
    processor = AutoProcessor.from_pretrained(local, trust_remote_code=True, extra_special_tokens={})
    model = AutoModelForImageTextToText.from_pretrained(local, trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16).to(device).eval()

    def move_and_cast(inputs, dev):  # also cast float inputs (pixels) to the model dtype
        return {k: (v.to(dev).to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                    else v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    model._move_inputs_to_device = move_and_cast
    model.requires_grad_(False)
    return model, processor


def load_action_tokenizer(model, norm_tag: str, repo: str = "allenai/MolmoAct2-FAST-Tokenizer"):
    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast
    p = snapshot_download(repo)
    cfg = json.load(open(os.path.join(p, "processor_config.json")))
    module_name, class_name = cfg["auto_map"]["AutoProcessor"].rsplit(".", 1)
    spec = importlib.util.spec_from_file_location("fast_action_processor", os.path.join(p, f"{module_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = PreTrainedTokenizerFast.from_pretrained(p, local_files_only=True)
    proc = getattr(mod, class_name)(tok, **{k: v for k, v in cfg.items() if k not in ("auto_map", "processor_class")})
    stats = model._get_robot_stats()
    horizon = int(stats.get_action_horizon(norm_tag) or model.model._resolve_action_horizon())
    proc(np.zeros((1, horizon, int(model.config.max_action_dim)), np.float32))  # initialize shapes
    return proc


def prep_query(model, processor, action_tok, norm_tag, images, state, task):
    """Teacher-forcing inputs [prompt || the model's own greedy action tokens] (mirrors predict_action)."""
    MOD = sys.modules[type(model).__module__]
    stats = model._get_robot_stats()
    norm_tag = stats.validate_tag(norm_tag)
    meta = stats.get_metadata(norm_tag)
    state_str = MOD._build_discrete_state_string(
        np.asarray(stats.normalize_state(state, norm_tag), np.float32), int(model.config.num_state_tokens or 0))
    text = MOD._build_robot_text(
        task=MOD._normalize_question_text(str(task or "")), style="robot_action", discrete_state_string=state_str,
        setup_type=str(meta.get("setup_type", "") or ""), control_mode=str(meta.get("control_mode", "") or ""),
        add_setup_tokens=bool(model.config.add_setup_tokens),
        add_control_tokens=bool(model.config.add_control_tokens), num_images=model._count_images(images))
    inputs = model._drop_trivial_attention_mask(
        model._move_inputs_to_device(processor(text=text, images=images, return_tensors="pt"), model.device))
    with torch.inference_mode():
        out = model.predict_action(processor=processor, images=images, task=task, state=state, norm_tag=norm_tag,
                                   inference_action_mode="discrete", action_tokenizer=action_tok,
                                   enable_depth_reasoning=False, normalize_language=True, enable_cuda_graph=False)
    gen = out.generated_token_ids.detach().clone().to(model.device).long()
    gen = gen[None, :] if gen.dim() == 1 else gen
    inp = dict(inputs)
    prompt_len = int(inp["input_ids"].shape[1])
    inp["input_ids"] = torch.cat([inp["input_ids"], gen], dim=1)
    for key, fill in (("attention_mask", torch.ones_like), ("token_type_ids", torch.zeros_like)):
        if inputs.get(key) is not None:
            inp[key] = torch.cat([inputs[key], fill(gen.to(inputs[key].device))], dim=1).to(inputs[key].dtype)
    ids = inp["input_ids"][0]
    start, n_act = int(model.config.action_token_start_id), int(model.config.num_action_tokens)
    is_act = (ids >= start) & (ids < start + n_act)
    is_act[:prompt_len] = False
    act_pos = torch.nonzero(is_act).flatten()
    assert act_pos.numel() > 0, "no generated action tokens"
    return dict(inputs=inp, act_pos=act_pos, chosen=ids[act_pos], prompt_len=prompt_len)


class ResidualTap:
    """Forward hook re-emitting a block's output as (h + delta) with delta a zero grad leaf:
    the forward is unchanged and grad(m, delta) = dm/dh. The fp32 residual is kept in .h."""

    def __init__(self):
        self.delta, self.h = None, None

    def __call__(self, module, args, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        x = t.float()
        self.h = x.detach()
        if self.delta is None or self.delta.shape != x.shape:
            self.delta = torch.zeros_like(x).requires_grad_(True)
        x2 = (x + self.delta).to(t.dtype)
        return (x2,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else x2


def llm_block(model, layer: int):
    mods = [m for n, m in model.named_modules()
            if n.endswith(f"transformer.blocks.{layer}") and "vision" not in n and "action" not in n]
    assert len(mods) == 1, f"LLM block {layer}: {len(mods)} candidates"
    return mods[0]


def gradients(model, prep, blocks: dict, taps: dict):
    """One grad-enabled teacher-forced forward -> (m, {L: g [T, d]}, {L: h [T, d]}) in fp32."""
    handles = [blocks[L].register_forward_hook(taps[L]) for L in blocks]
    try:
        with torch.enable_grad():
            out = model(**prep["inputs"], use_cache=False)
            logits = (out.logits if hasattr(out, "logits") else out[0])[0, prep["act_pos"] - 1, :].float()
            m = torch.log_softmax(logits, -1).gather(1, prep["chosen"][:, None]).sum()
    finally:
        for h in handles:
            h.remove()
    gs = torch.autograd.grad(m, [taps[L].delta for L in blocks])
    return float(m.detach()), {L: g[0] for L, g in zip(blocks, gs)}, {L: taps[L].h[0] for L in blocks}


# ------------------------------------------------------------------ covector map
class LayerMap:
    """Window filter + covector map of one layer, from its training cache."""

    def __init__(self, cache: str, device):
        meta = json.load(open(os.path.join(cache, "cache_meta.json")))
        self.lo, self.hi = float(meta["norm_lo"]), float(meta["norm_hi"])
        self.mean_norm, self.eps = float(meta["mean_norm"]), float(meta["rms_eps"])
        self.gamma = (torch.from_numpy(np.load(os.path.join(cache, "postnorm_gamma.npy"))).float().to(device)
                      if meta["postnorm"] else None)

    def __call__(self, h_rows, g_rows):
        """-> g_n of the in-window rows [n_keep, d] (h is fp16-quantized first, as in the site)."""
        h16 = torch.nan_to_num(h_rows.half().float(), nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
        norms = h16.norm(dim=1)
        keep = (norms >= self.lo) & (norms <= self.hi)
        hk, gk = h16[keep], g_rows[keep]
        if self.gamma is None:
            return gk * self.mean_norm
        r = torch.sqrt((hk * hk).mean(dim=1, keepdim=True) + self.eps)
        return gk * ((self.mean_norm * r) / self.gamma[None, :])


def episode_ids(frames_dir: str) -> list[int]:
    return sorted(int(re.search(r"ep(\d+)\.json$", p).group(1)) for p in glob.glob(os.path.join(frames_dir, "ep*.json")))


def queries(frames_dir: str, ep: int):
    from PIL import Image
    j = json.load(open(os.path.join(frames_dir, f"ep{ep:04d}.json")))
    for q in j["queries"]:
        imgs = [Image.fromarray(np.array(Image.open(os.path.join(
            frames_dir, "frames", f"ep{ep:04d}", f"q{int(q['q']):03d}_{c}.jpg")).convert("RGB"))) for c in j["cams"]]
        yield imgs, np.asarray(q["state"], np.float32), j.get("instruction", "")


def run_episode(ctx, ep: int) -> dict:
    """-> {L: [g_n of each query's in-window prompt rows]}."""
    gn = {L: [] for L in ctx["layers"]}
    for imgs, state, task in queries(ctx["frames"], ep):
        prep = prep_query(ctx["model"], ctx["processor"], ctx["atok"], ctx["norm_tag"], imgs, state, task)
        _m, g, h = gradients(ctx["model"], prep, ctx["blocks"], ctx["taps"])
        ids = prep["inputs"]["input_ids"][0, :prep["prompt_len"]].cpu().numpy()
        rows = np.concatenate([np.where(ids == ctx["ipid"])[0], np.where(ids != ctx["ipid"])[0]])
        rows_t = torch.from_numpy(rows).to(ctx["device"])
        for L in ctx["layers"]:
            gn[L].append(ctx["maps"][L](h[L][rows_t], g[L][rows_t]))
    return gn


def harvest(a):
    assert len(a.layers) == len(a.caches), "one --caches entry per --layers entry"
    os.makedirs(a.out, exist_ok=True)
    train_eps = [e for e in episode_ids(a.frames) if e % 20 != 0]
    eps = ([int(e) for e in a.eps.split(",")] if a.eps else
           train_eps[::20] if a.command == "bounds" else train_eps[a.shard::a.nshards])
    t0 = time.time()
    model, processor = load_model(a.checkpoint, a.device)
    ctx = dict(model=model, processor=processor, atok=load_action_tokenizer(model, "franka_droid"),
               norm_tag="franka_droid", ipid=int(model.config.image_patch_id), device=a.device, frames=a.frames,
               layers=a.layers, blocks={L: llm_block(model, L) for L in a.layers},
               taps={L: ResidualTap() for L in a.layers},
               maps={L: LayerMap(c, a.device) for L, c in zip(a.layers, a.caches)})
    print(f"[fisher] model loaded ({time.time() - t0:.0f}s); {a.command} layers={a.layers} episodes={len(eps)}",
          flush=True)

    if a.command == "bounds":
        norms = {L: [] for L in a.layers}
        for i, ep in enumerate(eps):
            for L, gl in run_episode(ctx, ep).items():
                norms[L].append(torch.cat(gl).norm(dim=1).cpu().numpy())
            print(f"[fisher] bounds ep{ep:04d} ({i + 1}/{len(eps)}, {time.time() - t0:.0f}s)", flush=True)
        bounds = {}
        for L in a.layers:
            nl = np.concatenate(norms[L])
            nl = nl[nl > 0]
            bounds[str(L)] = [float(np.percentile(nl, 0.5)), float(np.percentile(nl, 99.5))]
            print(f"[fisher] L{L}: {len(nl):,} rows, ||g_n|| bounds {bounds[str(L)]}", flush=True)
        json.dump(dict(bounds=bounds, episodes=eps, layers=a.layers), open(os.path.join(a.out, "bounds.json"), "w"),
                  indent=1)
        return

    bounds = json.load(open(os.path.join(a.out, "bounds.json")))["bounds"]
    path = os.path.join(a.out, f"shard{a.shard}.npz")
    state = dict(np.load(path)) if os.path.exists(path) else {}   # resume a preempted shard
    done = set(state.get("done_eps", np.zeros(0, np.int64)).tolist())
    d = int(model.config.text_config.hidden_size)
    J = {L: torch.from_numpy(state[f"J_L{L}"]).to(a.device) if f"J_L{L}" in state
         else torch.zeros(d, d, dtype=torch.float64, device=a.device) for L in a.layers}
    n_rows = {L: int(state.get(f"n_rows_L{L}", 0)) for L in a.layers}

    def save():
        np.savez(path + ".tmp.npz", done_eps=np.array(sorted(done), np.int64), bounds=json.dumps(bounds),
                 **{f"J_L{L}": J[L].cpu().numpy() for L in a.layers},
                 **{f"n_rows_L{L}": n_rows[L] for L in a.layers},
                 **{f"mean_norm_L{L}": ctx["maps"][L].mean_norm for L in a.layers})
        os.replace(path + ".tmp.npz", path)

    todo = [e for e in eps if e not in done]
    for i, ep in enumerate(todo):
        for L, gl in run_episode(ctx, ep).items():
            lo, hi = bounds[str(L)]
            g = torch.cat(gl)
            nrm = g.norm(dim=1)
            g = g * (torch.clamp(nrm, lo, hi) / torch.clamp(nrm, min=1e-30))[:, None]
            J[L] += (g.T @ g).double()
            n_rows[L] += len(g)
        done.add(ep)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            save()
            print(f"[fisher] shard {a.shard}: {len(done)}/{len(eps)} episodes, rows L{a.layers[0]}="
                  f"{n_rows[a.layers[0]]:,} ({time.time() - t0:.0f}s)", flush=True)
    save()


# ------------------------------------------------------------------ whitening
def whitening(J: np.ndarray, delta: float = 0.25):
    """fp64 eigendecomposition of a symmetric PSD J -> (W, Winv, evals, evals_reg)."""
    evals, U = torch.linalg.eigh(torch.from_numpy(np.asarray(J, np.float64)))
    evals = torch.clamp(evals, min=0.0)
    lam_reg = evals / evals.mean() + delta
    W = (U * torch.sqrt(lam_reg)) @ U.T
    Winv = (U * lam_reg.rsqrt()) @ U.T
    return W.numpy(), Winv.numpy(), evals.numpy(), lam_reg.numpy()


def whiten(a):
    J, n, info = None, 0, {}
    for p in sorted(glob.glob(os.path.join(a.out, "shard*.npz"))):
        z = np.load(p)
        Js = z[f"J_L{a.layer}"].astype(np.float64)
        J = Js if J is None else J + Js
        n += int(z[f"n_rows_L{a.layer}"])
        b = json.loads(str(z["bounds"]))[str(a.layer)]
        assert not info or b == [info["gnorm_lo"], info["gnorm_hi"]], f"{p}: different winsorize bounds"
        info = dict(gnorm_lo=b[0], gnorm_hi=b[1], mean_norm_used=float(z[f"mean_norm_L{a.layer}"]))
    assert n > 0, f"no gradient rows in {a.out}/shard*.npz"
    J = J / n
    J = 0.5 * (J + J.T)
    W, Winv, evals, lam_reg = whitening(J, a.delta)
    out = os.path.join(a.out, f"fisher_W_L{a.layer}.npz")
    np.savez(out, W=W.astype(np.float32), Winv=Winv.astype(np.float32), J=J.astype(np.float32),
             evals_raw=evals, evals_reg=lam_reg, delta=a.delta, n_tokens=n,
             created=time.strftime("%Y-%m-%dT%H:%M:%S"), **info)
    print(f"[fisher] N={n:,} tokens, cond(W)={float(np.sqrt(lam_reg.max() / lam_reg.min())):.2f}, "
          f"{int((lam_reg - a.delta > a.delta).sum())}/{len(J)} directions above the floor -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["bounds", "accum", "whiten"])
    ap.add_argument("--out", required=True, help="bounds.json, shard{s}.npz and fisher_W_L{layer}.npz live here")
    ap.add_argument("--frames", help="frames directory (format in this module's docstring)")
    ap.add_argument("--layers", type=int, nargs="+", help="bounds/accum: layers (one backward serves all)")
    ap.add_argument("--caches", nargs="+", help="bounds/accum: training cache per layer (same order)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--eps", default=None, help="comma-separated episode ids (overrides the default set)")
    ap.add_argument("--checkpoint", default="allenai/MolmoAct2-DROID")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, help="whiten: the layer to build")
    ap.add_argument("--delta", type=float, default=0.25, help="whiten: plain-MSE floor")
    a = ap.parse_args()
    whiten(a) if a.command == "whiten" else harvest(a)


if __name__ == "__main__":
    main()

import argparse
import json
import math
import os
import time

import torch

from .grf import sample_fields, add_noise, make_mask, observe
from .floor import exact_floor, prior_variance, effective_modes, patch_covariance, conditional_floor, wiener_matrix
from .models import build, count_params


def get_args():
    p = argparse.ArgumentParser()
    # data / condition
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--h", default="200", help="altitude(s) in metres, comma separated; one is sampled per patch")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--dx", type=float, default=100.0)
    p.add_argument("--sigma0", type=float, default=50.0, help="ground-level rms of the anomaly, nT")
    p.add_argument("--sigma", type=float, default=1.0, help="observation noise, nT")
    p.add_argument("--mask", default="random", choices=["random", "lines"])
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--spacing", type=int, default=4)
    p.add_argument("--D", type=int, default=0, help="dataset size in patches; 0 = fresh data every step")
    p.add_argument("--teacher", type=int, default=0,
                   help="train against the exact conditional mean W_M y_O on this many fixed masks instead of the "
                        "sampled field; the loss then has no floor and measures approximation + optimisation only")
    p.add_argument("--noisy_target", action="store_true",
                   help="train and evaluate against the noisy hidden values instead of the clean field (loss shifts by sigma^2)")
    p.add_argument("--fixed_mask", action="store_true",
                   help="one mask for every patch, train and val, the same mask linear_ref.py uses for this seed")
    # model
    p.add_argument("--model", default="vit", choices=["vit", "unet"])
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--no_altitude", action="store_true")
    p.add_argument("--mup", action="store_true")
    p.add_argument("--base_width", type=int, default=64)
    # optimisation
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="mixed precision on cuda: bf16 where supported, else fp16 + loss scaling")
    p.add_argument("--compile", action="store_true")
    # bookkeeping
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--floor_masks", type=int, default=8, help="masks averaged for the exact floor; 0 skips it")
    p.add_argument("--out", default="runs/run.json")
    p.add_argument("--save", action="store_true", help="save init and final weights next to the json")
    return p.parse_args()


class Pool:
    """A dataset of D patches that is never stored: block j is regenerated from
    seed (seed, j), noise included, so every epoch sees the same data. Masks
    are fresh each time, as in MAE. D=0 means unlimited fresh data."""

    def __init__(self, a, hs, seed, device):
        self.a, self.hs, self.seed, self.device = a, hs, seed, device
        self.nblocks = a.D // a.batch if a.D > 0 else 0
        self.epoch_rng = torch.Generator().manual_seed(seed + 7)
        self.order, self.pos = [], 0

    def block(self, j):
        gen = torch.Generator(device=self.device).manual_seed(self.seed * 1000003 + j)
        hidx = torch.randint(0, len(self.hs), (self.a.batch,), generator=gen, device=self.device)
        h = torch.tensor(self.hs, device=self.device)[hidx]
        clean = sample_fields(self.a.batch, self.a.n, self.a.beta, h, self.a.dx, self.a.sigma0, gen, self.device)
        noisy = add_noise(clean, self.a.sigma, gen)
        return clean, noisy, h

    def next(self, step):
        if self.nblocks == 0:
            return self.block(step)
        if self.pos == 0:
            self.order = torch.randperm(self.nblocks, generator=self.epoch_rng).tolist()
        j = self.order[self.pos]
        self.pos = (self.pos + 1) % self.nblocks
        return self.block(j)


def masked_mse(pred, target, mask):
    err = ((pred - target) ** 2 * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
    return err


class Teacher:
    """A set of fixed masks with their exact conditional-mean matrices. Targets are W_M y_O
    written into the hidden pixels; observed pixels keep the field (they are not in the loss)."""
    def __init__(self, a, hs, device):
        gen = torch.Generator(device=device).manual_seed(1000 + a.seed)
        self.masks = make_mask(a.mask, a.teacher, a.n, a.ratio, a.spacing, gen, device)      # (K, n, n), True = hidden
        self.W = {}
        for h in hs:
            K = patch_covariance(a.n, a.beta, float(h), a.dx, a.sigma0)
            self.W[float(h)] = [wiener_matrix(K, m.flatten().cpu(), a.sigma)[0].float().to(device) for m in self.masks]

    def batch_masks(self, batch, gen):
        idx = torch.randint(0, len(self.masks), (batch,), generator=gen, device=self.masks.device)
        return self.masks[idx], idx

    def targets(self, noisy, idx, h):
        # exact conditional mean for each example's mask (single-altitude runs only)
        h = float(h.flatten()[0]) if torch.is_tensor(h) else float(h)
        out = noisy.clone()
        flat = noisy.flatten(1)
        for k in range(len(self.masks)):
            sel = idx == k
            if sel.any():
                hid = self.masks[k].flatten()
                pred = flat[sel][:, ~hid] @ self.W[h][k].T
                tmp = out[sel].flatten(1); tmp[:, hid] = pred; out[sel] = tmp.view(-1, *noisy.shape[1:])
        return out


def fixed_mask(a, device):
    # same generator sequence as linear_ref.py, so the fixed-mask study and the linear
    # reference see the identical mask for a given seed
    gen = torch.Generator(device=device).manual_seed(1000 + a.seed)
    return make_mask(a.mask, 1, a.n, a.ratio, a.spacing, gen, device)[0]


def make_val(a, hs, device, fmask=None, teacher=None):
    # fixed validation set: fresh fields, fixed masks, one batch per altitude per index
    gen = torch.Generator(device=device).manual_seed(10_000 + a.seed)
    batches = []
    for _ in range(a.eval_batches):
        for h in hs:
            clean = sample_fields(a.batch, a.n, a.beta, float(h), a.dx, a.sigma0, gen, device)
            noisy = add_noise(clean, a.sigma, gen)
            if teacher is not None:
                mask, idx = teacher.batch_masks(a.batch, gen)
                clean = teacher.targets(noisy, idx, h)
            else:
                mask = make_mask(a.mask, a.batch, a.n, a.ratio, a.spacing, gen, device) if fmask is None \
                    else fmask[None].expand(a.batch, -1, -1)
            batches.append((clean, noisy, mask, h))
    return batches


def make_train_eval(a, pool, hs, device, fmask=None):
    # a fixed subset of the training fields with fixed masks, so the train/test gap is
    # measured on the same footing as the validation loss, not from one stochastic batch
    if pool.nblocks == 0:
        return []
    gen = torch.Generator(device=device).manual_seed(20_000 + a.seed)
    batches = []
    for j in range(min(a.eval_batches, pool.nblocks)):
        clean, noisy, h = pool.block(j)
        mask = make_mask(a.mask, a.batch, a.n, a.ratio, a.spacing, gen, device) if fmask is None \
            else fmask[None].expand(a.batch, -1, -1)
        for hv in hs:
            sel = h == hv
            if sel.any():
                batches.append((clean[sel], noisy[sel], mask[sel], hv))
    return batches


def amp_dtype(a, device):
    if not (a.amp and device == "cuda"):
        return None
    try:
        ok = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        ok = torch.cuda.get_device_capability()[0] >= 8
    return torch.bfloat16 if ok else torch.float16      # 2080 Ti class cards get fp16


@torch.no_grad()
def evaluate(model, val, a, scale, hmax, device, dtype):
    model.eval()
    out = {}
    for clean, noisy, mask, h in val:
        if a.noisy_target:
            clean = noisy
        x = observe(noisy, mask, scale)
        hn = torch.full((clean.shape[0],), h / hmax, device=device)
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            pred = model(x, hn).float() * scale
        out.setdefault(h, []).append(masked_mse(pred, clean, mask.float()).mean().item())
    model.train()
    return {str(h): sum(v) / len(v) for h, v in out.items()}


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * step / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hs = [float(x) for x in a.h.split(",")]
    hmax = max(800.0, max(hs))            # altitude fed to the model as h / hmax
    scale = a.sigma0                      # units change only; keeps activations O(1) in bf16

    model = build(a.model, a.n, a.width, a.depth, a.heads, not a.no_altitude, a.mup, a.base_width, a.patch).to(device)
    nparams = count_params(model)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.save:
        torch.save(model.state_dict(), a.out.replace(".json", "_init.pt"))
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.wd), betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]
    dtype = amp_dtype(a, device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    if a.compile:
        model = torch.compile(model)

    fmask = fixed_mask(a, device) if a.fixed_mask else None
    teacher = Teacher(a, hs, device) if a.teacher else None
    info = dict(config=vars(a), params=nparams, device=device, precision=str(dtype).replace("torch.", "") if dtype else "fp32",
                gpu=torch.cuda.get_device_name() if device == "cuda" else "cpu", floor={}, prior_var={}, modes={})
    for h in hs:
        info["prior_var"][str(h)] = prior_variance(a.n, a.beta, h, a.dx, a.sigma0)
        info["modes"][str(h)] = effective_modes(a.n, a.beta, h, a.sigma, a.dx, a.sigma0)
        if a.fixed_mask:
            K = patch_covariance(a.n, a.beta, h, a.dx, a.sigma0)
            info["floor"][str(h)] = conditional_floor(K, fmask.flatten().cpu(), a.sigma).item()
        elif a.floor_masks > 0:
            fl, err = exact_floor(a.n, a.beta, h, a.sigma, a.mask, a.ratio, a.spacing, a.dx, a.sigma0, a.floor_masks)
            info["floor"][str(h)] = fl
            info.setdefault("floor_err", {})[str(h)] = err
    print(f"params {nparams/1e6:.2f}M  floor {info['floor']}  prior {info['prior_var']}")

    pool = Pool(a, hs, a.seed, device)
    val = make_val(a, hs, device, fmask, teacher)
    train_eval = make_train_eval(a, pool, hs, device, fmask)
    mask_gen = torch.Generator(device=device).manual_seed(a.seed + 99)
    log, best = [], {str(h): float("inf") for h in hs}
    train_loss = None                       # set after the first optimiser step
    t0, tick, seen = time.time(), None, 0

    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(model, val, a, scale, hmax, device, dtype)
            for k in v:
                best[k] = min(best[k], v[k])
            tv = evaluate(model, train_eval, a, scale, hmax, device, dtype) if train_eval else None
            log.append(dict(step=step, train=None if step == 0 else float(train_loss), val=v, train_eval=tv,
                            time=time.time() - t0))
            print(f"step {step:6d}  train {log[-1]['train']}  val {v}")
        if step == a.steps:
            break
        if step == min(a.warmup, a.steps // 4):
            tick, seen = time.time(), 0        # throughput timer starts after warmup

        clean, noisy, h = pool.next(step)
        if teacher is not None:
            mask, idx = teacher.batch_masks(a.batch, mask_gen)
            clean = teacher.targets(noisy, idx, h)
        else:
            mask = make_mask(a.mask, a.batch, a.n, a.ratio, a.spacing, mask_gen, device) if fmask is None \
                else fmask[None].expand(a.batch, -1, -1)
        x = observe(noisy, mask, scale)
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * lr_at(step, a) / a.lr
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            pred = model(x, h / hmax)
        target = noisy if a.noisy_target else clean
        loss = masked_mse(pred.float(), target / scale, mask.float()).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        train_loss = loss.item() * scale ** 2
        if tick is not None:
            seen += a.batch

    wall = time.time() - t0
    vals = [e["val"] for e in log]
    tail = vals[max(1, int(0.75 * len(vals))):]
    converged = {}
    for k in best:
        first, last = tail[0][k], tail[-1][k]
        converged[k] = abs(last - first) / max(first, 1e-12) < 0.05
    info.update(dict(log=log, best_val=best, final_val=vals[-1], train_final=train_loss,
                     train_eval_final=log[-1]["train_eval"], converged=converged,
                     wall_time_s=wall,
                     samples_per_s=seen / (time.time() - tick) if tick and seen else a.steps * a.batch / wall,
                     peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
                     D_actual=pool.nblocks * a.batch if pool.nblocks else 0))
    json.dump(info, open(a.out, "w"), indent=1)
    if a.save:
        sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save(sd, a.out.replace(".json", ".pt"))
    print(f"done in {wall/60:.1f} min, best val {best}, floor {info['floor']}")


if __name__ == "__main__":
    main()

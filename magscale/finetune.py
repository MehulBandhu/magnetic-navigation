"""Fine-tune a trained network on real EMAG2 tiles. The cheap baseline for the generator
question: if a few hundred real tiles close most of the gap to the Gaussian estimator, the
tails are learnable from little data; if not, the generator is the way.

Tiles come from the same pipeline as the evaluation (land compilations, square pixels,
detrended, rescaled to the training amplitude), from a training region with a stride of half a
tile, with random flips and rotations; the loss is the hidden-pixel MSE under a fresh 75% mask
each step, with 1 nT of noise added. The other region is never seen, and `emag2 run --regions`
on it is the test.

python -m magscale.finetune --run runs/vitxxl_b3.5_hmix_D131072_s0_noalt.json --train australia \\
       --steps 2000 --out runs/vitxxl_finetune_australia.json
python -m magscale.emag2 run --run runs/vitxxl_finetune_australia.json --regions north_america --tag _ft_aus
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from scipy.ndimage import zoom

from .emag2 import region_slice, detrend, load_model, H_EQ, DX
from .floor import prior_variance
from .grf import random_mask, observe


def real_tiles(grid, code, region, n=64, stride=32):
    rs, cs = region_slice(region)
    sub, subcode = grid[rs, cs], code[rs, cs]
    target_rms = math.sqrt(prior_variance(n, 3.5, H_EQ, DX, 50.0))
    tiles = []
    for i in range(0, sub.shape[0] - n + 1, stride):
        lat = -90 + (rs.start + i + n / 2) / 30
        wcols = int(round(n / math.cos(math.radians(lat))))
        for j in range(0, sub.shape[1] - wcols + 1, max(stride, wcols // 2)):
            raw = sub[i:i + n, j:j + wcols]
            cc = subcode[i:i + n, j:j + wcols]
            if np.isnan(raw).any() or np.mean((cc > 0) & (cc < 100)) < 0.95:
                continue
            t = zoom(raw.astype(np.float64), (1.0, n / wcols), order=1)[:, :n]
            if t.shape != (n, n):
                continue
            t = detrend(t)
            if t.std() < 1.0:
                continue
            tiles.append(t * (target_rms / t.std()))
    return torch.tensor(np.stack(tiles), dtype=torch.float32)


def augment(x, gen):
    k = int(torch.randint(0, 4, (1,), generator=gen))
    x = torch.rot90(x, k, dims=(1, 2))
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(2)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/vitxxl_b3.5_hmix_D131072_s0_noalt.json")
    p.add_argument("--grid", default="data/emag2_upcont.npy")
    p.add_argument("--train", default="australia", help="region(s) to fine-tune on, comma separated")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/vitxxl_finetune_australia.json")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    grid = np.load(a.grid); code = np.load(a.grid.replace("upcont", "code"))
    tiles = torch.cat([real_tiles(grid, code, r) for r in a.train.split(",")])
    print(f"{len(tiles)} training tiles from {a.train}")
    model, c = load_model(a.run, device)
    model.train()
    train_hs = [float(x) for x in str(c["h"]).split(",")]
    h_in = torch.full((a.batch,), H_EQ / max(800.0, max(train_hs)), device=device)
    sigma0 = c["sigma0"]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(a.seed)
    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(tiles), (a.batch,), generator=gen)
        clean = augment(tiles[idx], gen).to(device)
        noisy = clean + a.sigma * torch.randn(clean.shape, generator=gen).to(device)
        mask = random_mask(a.batch, clean.shape[-1], 0.75, gen).to(device)
        pred = model(observe(noisy, mask, sigma0), h_in) * sigma0
        loss = (((pred - clean) ** 2) * mask).sum() / mask.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == a.steps:
            log.append(dict(step=step, train=float(loss), time=time.time() - t0))
            print(f"step {step:5d}  train {float(loss):.3f}")
    model.eval()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), a.out.replace(".json", ".pt"))
    src = json.load(open(a.run))
    json.dump(dict(config=dict(src["config"], finetune=dict(source=a.run, train=a.train, steps=a.steps, lr=a.lr, batch=a.batch, tiles=len(tiles))),
                   params=src["params"], log=log, wall_time_s=time.time() - t0), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

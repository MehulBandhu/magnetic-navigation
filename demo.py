"""One-command demo: generate a map, hide 75% of it, reconstruct it three ways, score them.

    python demo.py                      # about a minute on a laptop CPU
    python demo.py --ckpt runs/vitxxl_b3.5_h200_D131072_s0_long.json   # adds the network column

Without a checkpoint the demo still runs (interpolation and exact Gaussian oracle); the
network column appears when a trained model's json + pt are present.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from scipy.interpolate import griddata

from magscale.grf import sample_fields, add_noise, make_mask, observe
from magscale.floor import patch_covariance, wiener_matrix, prior_variance
from magscale.models import build


def interpolate(noisy, mask):
    # no prior: linear interpolation of the observed pixels
    n = noisy.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    obs = ~mask
    out = griddata((yy[obs], xx[obs]), noisy[obs], (yy, xx), method="linear")
    nearest = griddata((yy[obs], xx[obs]), noisy[obs], (yy, xx), method="nearest")
    out[np.isnan(out)] = nearest[np.isnan(out)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=3.5)
    p.add_argument("--h", type=float, default=200.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--ckpt", default="runs/vitxxl_b3.5_h200_D131072_s0_long.json")
    p.add_argument("--out", default="figures/demo.png")
    a = p.parse_args()
    n, dx, sigma0 = 64, 100.0, 50.0

    gen = torch.Generator().manual_seed(a.seed)
    clean = sample_fields(1, n, a.beta, a.h, dx, sigma0, gen)
    noisy = add_noise(clean, a.sigma, gen)
    mask = make_mask("random", 1, n, a.ratio, 4, gen)
    hid = mask[0].flatten()

    # the exact Gaussian conditional mean for this mask, and its error
    K = patch_covariance(n, a.beta, a.h, dx, sigma0)
    W, var = wiener_matrix(K, hid, a.sigma)
    oracle = noisy[0].double().flatten().clone()
    oracle[hid] = W @ noisy[0].double().flatten()[~hid]
    oracle = oracle.view(n, n)
    floor = var.mean().item()

    recon = {"linear interpolation": torch.from_numpy(interpolate(noisy[0].numpy(), mask[0].numpy())),
             "exact Gaussian oracle": oracle}
    if os.path.exists(a.ckpt) and os.path.exists(a.ckpt.replace(".json", ".pt")):
        r = json.load(open(a.ckpt))
        c = r["config"]
        model = build(c["model"], c["n"], c["width"], c["depth"], c["heads"], not c["no_altitude"], c["mup"], c["base_width"], c["patch"])
        model.load_state_dict(torch.load(a.ckpt.replace(".json", ".pt"), map_location="cpu"))
        model.eval()
        with torch.no_grad():
            x = observe(noisy, mask, sigma0)
            pred = model(x, torch.tensor([a.h / max(800.0, a.h)]))[0] * sigma0
        net = noisy[0].clone(); net[mask[0]] = pred[mask[0]]
        recon[f"network ({r['params']/1e6:.0f}M)"] = net
    else:
        print(f"no checkpoint at {a.ckpt}; showing interpolation and oracle only")
        if a.out == "figures/demo.png":
            a.out = "figures/demo_nonetwork.png"      # keep the committed network version intact

    m = mask[0]
    print(f"beta={a.beta} h={a.h:.0f} m sigma={a.sigma} nT, {int(a.ratio*100)}% hidden. prior variance {prior_variance(n, a.beta, a.h, dx, sigma0):.1f} nT^2")
    print(f"{'method':24s} {'hidden-pixel MSE [nT^2]':>24s} {'rms [nT]':>9s}")
    errs = {}
    for name, img in recon.items():
        e = ((img.double() - clean[0].double()) ** 2)[m].mean().item()
        errs[name] = e
        print(f"{name:24s} {e:24.3f} {math.sqrt(e):9.3f}")
    print(f"{'exact floor (this mask)':24s} {floor:24.3f} {math.sqrt(floor):9.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 2 + len(recon)
    fig, ax = plt.subplots(2, cols, figsize=(3.1 * cols, 6))
    v = clean[0].abs().max().item()
    obs = noisy[0].clone(); obs[m] = float("nan")
    ax[0, 0].imshow(clean[0], cmap="RdBu_r", vmin=-v, vmax=v); ax[0, 0].set_title("true map", fontsize=9)
    ax[0, 1].imshow(obs, cmap="RdBu_r", vmin=-v, vmax=v); ax[0, 1].set_title(f"observed ({int((1-a.ratio)*100)}%)", fontsize=9)
    ax[1, 0].axis("off"); ax[1, 1].axis("off")
    ax[1, 0].text(0, 0.5, f"floor for this mask\n{floor:.3f} nT^2\n({math.sqrt(floor):.2f} nT rms)", fontsize=10, va="center")
    for i, (name, img) in enumerate(recon.items()):
        ax[0, i + 2].imshow(img, cmap="RdBu_r", vmin=-v, vmax=v); ax[0, i + 2].set_title(name, fontsize=9)
        ax[1, i + 2].imshow((img - clean[0]) * m, cmap="RdBu_r", vmin=-3 * a.sigma, vmax=3 * a.sigma)
        ax[1, i + 2].set_title(f"error on hidden pixels: {errs[name]:.3f} nT^2", fontsize=9)
    for x in ax.flatten():
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"beta={a.beta}, h={a.h:.0f} m, sigma={a.sigma} nT, 64 x 64 patch = 6.4 km", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("figure:", a.out)


if __name__ == "__main__":
    main()

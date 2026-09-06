"""Magnetic navigation on reconstructed maps.

1. Altitude law. Fly a fixed trajectory over the true anomaly map at several
   altitudes, estimate position by map matching (the ML estimator: shift the
   trajectory until the measured profile fits the map), and compare the RMS
   error with the Cramer-Rao bound from Problem 1C. Prediction: sigma_pos ~ h^((4-beta)/2).
2. Map quality. Same navigation at one altitude on maps reconstructed from a
   flight-line survey by the trained network, by the Gaussian conditional mean
   applied on overlapping 64-pixel tiles ("tiled Wiener": exact per tile, not the
   global conditional), and by linear interpolation between lines, plus the true
   map, averaged over several independent maps.

python -m magscale.magnav --ckpt runs/vitxxl_b3.5_h200_D131072_s0_lines.json
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

from .grf import sample_fields
from .floor import patch_covariance, wiener_matrix
from .models import build

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- maps
def true_map(n, beta, h, dx, sigma0, seed):
    gen = torch.Generator().manual_seed(seed)
    return sample_fields(1, n, beta, h, dx, sigma0, gen)[0].double().numpy()


def survey_mask(n, spacing, offset=0):
    # observed rows every `spacing` pixels, the survey flight lines
    # observed rows every `spacing` pixels from `offset`. The last spacing-1 rows have no line
    # below them and are extrapolated by every method; the network was trained on exactly this
    # regular pattern, so no closing line is added (an irregular final line is out of distribution
    # for it and produced 100 nT errors in the bottom tile).
    m = np.ones((n, n), dtype=bool)
    m[offset::spacing, :] = False
    return m


def line_interp(noisy, mask):
    # what a survey processor does with no prior: interpolate between lines column by column
    n = noisy.shape[0]
    rows = np.where(~mask[:, 0])[0]
    out = np.empty_like(noisy)
    for j in range(n):
        out[:, j] = np.interp(np.arange(n), rows, noisy[rows, j])
    return out


def tile_apply(noisy, mask, fn, patch=64, stride=32):
    """apply a patch reconstructor over the map with overlapping tiles and a centre-weighted
    blend, so no pixel has to be extrapolated from a tile edge"""
    n = noisy.shape[0]
    w1 = np.full(patch, 0.05); w1[patch // 4: 3 * patch // 4] = 1.0
    w = np.outer(w1, w1)
    acc, wsum = np.zeros_like(noisy), np.zeros_like(noisy)
    starts = sorted(set(list(range(0, n - patch + 1, stride)) + [n - patch]))
    for i in starts:
        for j in starts:
            pred = fn(noisy[i:i + patch, j:j + patch], mask[i:i + patch, j:j + patch])
            acc[i:i + patch, j:j + patch] += w * pred
            wsum[i:i + patch, j:j + patch] += w
    out = acc / wsum
    out[~mask] = noisy[~mask]
    return out


def wiener_recon(noisy, mask, beta, h, sigma, dx, sigma0, patch=64):
    # exact conditional mean per tile, with the covariance of the grid this map was generated
    # on (2n), not the training patches' 128 grid; the Wiener matrix is cached per mask pattern
    K = patch_covariance(patch, beta, h, dx, sigma0, gen_n=2 * noisy.shape[0])
    cache = {}

    def fn(blk, m):
        key = m.tobytes()
        if key not in cache:
            cache[key] = wiener_matrix(K, torch.from_numpy(m.flatten()), sigma)[0].numpy()
        pred = blk.flatten().copy()
        pred[m.flatten()] = cache[key] @ blk.flatten()[~m.flatten()]
        return pred.reshape(patch, patch)
    return tile_apply(noisy, mask, fn, patch)


@torch.no_grad()
def network_recon(noisy, mask, ckpt, h):
    r = json.load(open(ckpt))
    c = r["config"]
    model = build(c["model"], c["n"], c["width"], c["depth"], c["heads"], not c["no_altitude"], c["mup"], c["base_width"], c["patch"])
    model.load_state_dict(torch.load(ckpt.replace(".json", ".pt"), map_location="cpu"))
    model.eval()
    hmax = max(800.0, h)

    def fn(blk, m):
        b = torch.from_numpy(blk).float()[None]
        mm = torch.from_numpy(m)[None]
        x = torch.stack([torch.where(mm, torch.zeros_like(b), b) / c["sigma0"], mm.float()], 1)
        return model(x, torch.tensor([h / hmax])).numpy()[0].astype(np.float64) * c["sigma0"]
    return tile_apply(noisy, mask, fn, c["n"]), r["floor"].get(str(float(h)))


# ---------------------------------------------------------------- navigation
def bilinear(m, y, x):
    n = m.shape[0]
    y = np.clip(y, 0, n - 1.001); x = np.clip(x, 0, n - 1.001)
    y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    fy = y - y0; fx = x - x0
    return ((1 - fy) * (1 - fx) * m[y0, x0] + (1 - fy) * fx * m[y0, x0 + 1]
            + fy * (1 - fx) * m[y0 + 1, x0] + fy * fx * m[y0 + 1, x0 + 1])


def trajectory(nsteps, y0, x0, heading, turn_at, dturn):
    # straight, one turn, straight again: gradients along the two legs span R^2
    ys, xs = [y0], [x0]
    th = heading
    for k in range(1, nsteps):
        if k == turn_at:
            th += dturn
        ys.append(ys[-1] + math.sin(th)); xs.append(xs[-1] + math.cos(th))
    return np.array(ys), np.array(xs)


def grid_search(z, mp, ys, xs, cy, cx, half, step):
    offs = np.arange(-half, half + 1e-9, step)
    best = (np.inf, 0.0, 0.0)
    for dy in offs:
        pred = np.stack([bilinear(mp, ys + cy + dy, xs + cx + dx) for dx in offs])   # (len(offs), nsteps)
        cost = ((pred - z[None]) ** 2).sum(1)
        i = int(np.argmin(cost))
        if cost[i] < best[0]:
            best = (cost[i], cy + dy, cx + offs[i])
    return best[1], best[2]


def map_match(z, mp, ys, xs, search, fine=0.01):
    # ML position offset: shift the known trajectory shape until the profile fits the map.
    # coarse grid over the whole search box, then a fine grid around the best cell, so the
    # answer is resolved to `fine` pixels (1 m at dx=100) instead of the coarse step.
    cy, cx = grid_search(z, mp, ys, xs, 0.0, 0.0, search, 0.25)
    cy, cx = grid_search(z, mp, ys, xs, cy, cx, 0.3, 0.05)
    return grid_search(z, mp, ys, xs, cy, cx, 0.06, fine)


def crb(mp, ys, xs, sigma_eff, dx):
    # summed Fisher information along the trajectory, gradients from the map used for matching.
    # with a reconstructed map this is a heuristic effective-noise bound: map errors are spatially
    # correlated, and here they are folded into sigma_eff as if they were white
    gy = bilinear(np.gradient(mp, axis=0), ys, xs)
    gx = bilinear(np.gradient(mp, axis=1), ys, xs)
    J = np.array([[np.sum(gy * gy), np.sum(gy * gx)], [np.sum(gy * gx), np.sum(gx * gx)]]) / sigma_eff ** 2
    return math.sqrt(np.trace(np.linalg.inv(J))) * dx     # metres, both axes combined


def placements(n, trials, nsteps, rng, search=6.0):
    # fixed set of trajectories and noise seeds, reused across altitudes and map sources
    # so that comparisons between altitudes and map sources are paired
    out = []
    margin = nsteps + search + 2
    for _ in range(trials):
        y0 = rng.uniform(margin, n - margin); x0 = rng.uniform(margin, n - margin)
        ys, xs = trajectory(nsteps, y0, x0, rng.uniform(0, 2 * math.pi), nsteps // 2, math.radians(rng.choice([-90, 90])))
        out.append((ys, xs, rng.standard_normal(nsteps)))
    return out


def navigate(mp_truth, mp_used, sigma, dx, trials, search=6.0):
    # returns rms position error (m), its bootstrap standard error, mean bound (m), map rms error
    errs, bounds = [], []
    map_err = float(np.sqrt(np.mean((mp_used - mp_truth) ** 2)))
    sigma_eff = math.hypot(sigma, map_err)
    for ys, xs, noise in trials:
        z = bilinear(mp_truth, ys, xs) + sigma * noise
        dy, dxx = map_match(z, mp_used, ys, xs, search)
        errs.append(math.hypot(dy, dxx) * dx)
        bounds.append(crb(mp_used, ys, xs, sigma_eff, dx))
    e = np.square(errs)
    rng = np.random.default_rng(0)
    se = float(np.std([math.sqrt(np.mean(rng.choice(e, len(e)))) for _ in range(300)]))
    return float(math.sqrt(np.mean(e))), se, float(np.mean(bounds)), map_err


def spectral_gradient(n, beta, h, dx, sigma0):
    # rms gradient of the map family from its spectrum on the generation grid, i.e. the
    # finite-map version of Problem 1C(b). This is what the CRB should track.
    from .grf import ground_amplitude
    A0, k = ground_amplitude(2 * n, beta, dx, sigma0)
    return math.sqrt(((k ** 2 * A0 ** 2 * torch.exp(-2 * k * h)).sum() / (2 * n) ** 2).item())


# ---------------------------------------------------------------- experiments
def altitude_law(a, trials):
    print("\n== position error vs altitude on the true map ==")
    hs = [100, 200, 400, 800]
    res = {}
    for beta in a.betas:
        rows = []
        for h in hs:
            mp = true_map(a.n, beta, h, a.dx, a.sigma0, a.seed)
            rms, se, bound, _ = navigate(mp, mp, a.sigma, a.dx, trials)
            spec = a.sigma / spectral_gradient(a.n, beta, h, a.dx, a.sigma0)
            rows.append((h, rms, bound, spec, se))
            print(f"beta {beta}  h {h:4d} m  rms error {rms:7.2f} +- {se:4.2f} m   CRB {bound:7.2f} m   sigma/G_h {spec:7.2f} m")
        rows = np.array(rows)
        lh = np.log(rows[:, 0])
        slope = np.polyfit(lh, np.log(rows[:, 1]), 1)[0]
        slope_crb = np.polyfit(lh, np.log(rows[:, 2]), 1)[0]
        slope_spec = np.polyfit(lh, np.log(rows[:, 3]), 1)[0]
        print(f"beta {beta}: measured slope {slope:.2f}, CRB slope {slope_crb:.2f}, finite-map spectrum slope {slope_spec:.2f}, "
              f"asymptotic (4-beta)/2 = {(4-beta)/2:.2f}")
        res[beta] = dict(rows=rows.tolist(), slope=slope, slope_crb=slope_crb, slope_spec=slope_spec, gamma=(4 - beta) / 2)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for beta, r in res.items():
        rows = np.array(r["rows"])
        l, = ax.loglog(rows[:, 0], rows[:, 1], "o-", label=f"beta={beta}: map matching, slope {r['slope']:.2f}")
        ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 4], fmt="none", ecolor=l.get_color(), capsize=3)
        ax.loglog(rows[:, 0], rows[:, 2], "--", c=l.get_color(), label=f"beta={beta}: CRB along trajectory, slope {r['slope_crb']:.2f}")
        ax.loglog(rows[:, 0], rows[:, 3] * rows[0, 2] / rows[0, 3], "-.", c=l.get_color(), lw=1,
                  label=f"beta={beta}: sigma/G_h from the spectrum, slope {r['slope_spec']:.2f}")
        ax.loglog(rows[:, 0], rows[0, 2] * (rows[:, 0] / rows[0, 0]) ** r["gamma"], ":", c=l.get_color(), lw=1,
                  label=f"beta={beta}: asymptotic h^{r['gamma']:.2f}")
    ax.set_xlabel("altitude h [m]"); ax.set_ylabel(f"position error after {a.nsteps} samples [m]")
    ax.set_title(f"{a.n * a.dx / 1000:.0f} km map, sigma = {a.sigma} nT, 1 sample per {a.dx:.0f} m", fontsize=9)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{a.out}/magnav_altitude{getattr(a, 'suffix', '')}.png", dpi=130); plt.close(fig)
    return res


def map_quality(a, trials, rng):
    """navigation on maps reconstructed from a survey, averaged over several generated maps"""
    print("\n== navigation on reconstructed survey maps ==")
    beta, h = 3.5, a.h
    have_net = a.ckpt and os.path.exists(a.ckpt) and os.path.exists(a.ckpt.replace(".json", ".pt"))
    a.suffix = "" if have_net else "_nonetwork"
    if not have_net:
        print(f"(no network checkpoint at {a.ckpt}; showing interpolation and tiled Wiener only; "
              f"outputs get a _nonetwork suffix so the committed results are not overwritten)")
    names = ["linear interpolation", "tiled Wiener"] + (["network"] if have_net else [])
    acc = defaultdict(list)          # (spacing, name) -> list of (nav rms, map rms) per map
    truth_rms = []
    panels = None
    for k in range(a.maps):
        truth = true_map(a.n, beta, h, a.dx, a.sigma0, a.seed + 100 * k)
        noisy = truth + a.sigma * rng.standard_normal(truth.shape)
        truth_rms.append(navigate(truth, truth, a.sigma, a.dx, trials)[0])
        for spacing in a.spacings:
            mask = survey_mask(a.n, spacing)
            maps = {"linear interpolation": line_interp(noisy, mask),
                    "tiled Wiener": wiener_recon(noisy, mask, beta, h, a.sigma, a.dx, a.sigma0, 64)}
            if have_net and spacing == json.load(open(a.ckpt))["config"]["spacing"]:
                maps["network"], _ = network_recon(noisy, mask, a.ckpt, h)
            for name, mp in maps.items():
                rms, _, bound, merr = navigate(truth, mp, a.sigma, a.dx, trials)
                acc[(spacing, name)].append((rms, merr, bound))
            if k == 0:
                panels = panels or []
                panels.append((spacing, mask, maps, truth, noisy))
    ref = (float(np.mean(truth_rms)), float(np.std(truth_rms)))
    print(f"{'true map':24s} nav rms error {ref[0]:6.1f} +- {ref[1]:4.1f} m   ({a.maps} maps x {len(trials)} trajectories)")
    rows = {}
    for (spacing, name), v in sorted(acc.items()):
        v = np.array(v)
        rows[f"{spacing}:{name}"] = dict(nav_rms=float(v[:, 0].mean()), nav_sd=float(v[:, 0].std()),
                                        map_rms=float(v[:, 1].mean()), bound=float(v[:, 2].mean()))
        print(f"lines every {spacing * a.dx:5.0f} m  {name:22s} map rms {v[:, 1].mean():6.3f} nT   "
              f"nav rms {v[:, 0].mean():6.1f} +- {v[:, 0].std():4.1f} m   effective-noise bound {v[:, 2].mean():5.1f} m")

    ncol = 1 + len(names)
    fig, ax = plt.subplots(2 * len(panels), ncol, figsize=(3.1 * ncol, 6.0 * len(panels)), squeeze=False)
    for r, (spacing, mask, maps, truth, noisy) in enumerate(panels):
        v = np.abs(truth).max()
        obs = noisy.copy(); obs[mask] = np.nan
        ax[2 * r, 0].imshow(obs, cmap="RdBu_r", vmin=-v, vmax=v)
        ax[2 * r, 0].set_title(f"survey: lines every {spacing * a.dx:.0f} m", fontsize=9)
        ax[2 * r + 1, 0].imshow(truth, cmap="RdBu_r", vmin=-v, vmax=v)
        ax[2 * r + 1, 0].set_title(f"true map: nav {ref[0]:.0f} m", fontsize=9)
        for i, name in enumerate(names):
            if name not in maps:
                ax[2 * r, i + 1].text(0.5, 0.5, "network not evaluated:\ntrained at 400 m line spacing", ha="center", va="center", fontsize=8)
                ax[2 * r, i + 1].axis("off"); ax[2 * r + 1, i + 1].axis("off"); continue
            q = rows[f"{spacing}:{name}"]
            ax[2 * r, i + 1].imshow(maps[name], cmap="RdBu_r", vmin=-v, vmax=v); ax[2 * r, i + 1].set_title(name, fontsize=9)
            ax[2 * r + 1, i + 1].imshow(maps[name] - truth, cmap="RdBu_r", vmin=-5 * a.sigma, vmax=5 * a.sigma)
            ax[2 * r + 1, i + 1].set_title(f"map err {q['map_rms']:.2f} nT,  nav {q['nav_rms']:.0f} +- {q['nav_sd']:.0f} m", fontsize=9)
    for x in ax.flatten():
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"beta={beta}, h={h:.0f} m, sigma={a.sigma} nT, {a.n * a.dx / 1000:.1f} km maps; errors are over {a.maps} maps", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.out}/magnav_maps{a.suffix}.png", dpi=130); plt.close(fig)
    return dict(true_map=ref, maps=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/vitxxl_b3.5_h200_D131072_s0_lines.json")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--dx", type=float, default=100.0)
    p.add_argument("--sigma0", type=float, default=50.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--h", type=float, default=200.0)
    p.add_argument("--spacings", default="4,8")
    p.add_argument("--betas", default="2.5,3.5")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--maps", type=int, default=3, help="independent maps for the survey comparison")
    p.add_argument("--nsteps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    a.betas = [float(b) for b in a.betas.split(",")]
    a.spacings = [int(x) for x in a.spacings.split(",")]
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    trials = placements(a.n, a.trials, a.nsteps, rng)
    a.suffix = "" if (a.ckpt and os.path.exists(a.ckpt) and os.path.exists(a.ckpt.replace(".json", ".pt"))) else "_nonetwork"
    res = dict(altitude=altitude_law(a, trials), maps=map_quality(a, trials, rng))
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/magnav{a.suffix}.json", "w"), indent=1)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()

"""Problem 2C / 3C probes.

For a given mask the optimal predictor is linear, m_H = W y_O with W from
floor.wiener_matrix. The probes test whether a trained network learned W
(jacobian probe), whether it is a linear map of its input at fixed mask
(linearity probe), and whether its tangent kernel moved during training
(NTK probe).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from .floor import patch_covariance, wiener_matrix
from .grf import sample_fields, add_noise, make_mask, observe
from .models import build


def load(run_json, device, which="final"):
    r = json.load(open(run_json))
    c = r["config"]
    model = build(c["model"], c["n"], c["width"], c["depth"], c["heads"], not c["no_altitude"],
                  c["mup"], c["base_width"], c["patch"]).to(device)
    pt = run_json.replace(".json", "_init.pt" if which == "init" else ".pt")
    if not os.path.exists(pt):
        raise SystemExit(f"missing weights {pt}; see docs/release.md for where to get them")
    model.load_state_dict(torch.load(pt, map_location=device))
    model.eval()
    return model, r


def batch(c, h, B, device, seed=1234):
    gen = torch.Generator(device=device).manual_seed(seed)
    clean = sample_fields(B, c["n"], c["beta"], h, c["dx"], c["sigma0"], gen, device)
    noisy = add_noise(clean, c["sigma"], gen)
    mask = make_mask(c["mask"], B, c["n"], c["ratio"], c["spacing"], gen, device)
    return clean, noisy, mask


def predict(model, c, noisy, mask, h):
    hmax = max(800.0, h)
    x = observe(noisy, mask, c["sigma0"])
    hn = torch.full((noisy.shape[0],), h / hmax, device=noisy.device)
    return model(x, hn) * c["sigma0"]


def jacobian(model, c, noisy, mask, h, pixels):
    """rows of d pred[pixel] / d noisy[observed] for one input, as (len(pixels), n*n) with zeros at hidden"""
    noisy = noisy.clone().requires_grad_(True)
    pred = predict(model, c, noisy[None], mask[None], h)[0]
    rows = []
    for (i, j) in pixels:
        g, = torch.autograd.grad(pred[i, j], noisy, retain_graph=True)
        rows.append(g.flatten())
    return torch.stack(rows)


def probe_jacobian(run, h, n_inputs, n_pixels, device, outdir):
    model, r = load(run, device)
    c = r["config"]
    n = c["n"]
    clean, noisy, mask = batch(c, h, n_inputs, device)
    K = patch_covariance(n, c["beta"], h, c["dx"], c["sigma0"], device)
    stats = []
    for b in range(n_inputs):
        hid = mask[b].flatten()
        W, _ = wiener_matrix(K, hid, c["sigma"])                      # |H| x |O|
        hid_idx = torch.nonzero(hid).flatten()
        rng = torch.Generator().manual_seed(b)
        pick = hid_idx[torch.randperm(len(hid_idx), generator=rng)[:n_pixels]]
        pixels = [(int(p) // n, int(p) % n) for p in pick]
        J = jacobian(model, c, noisy[b], mask[b], h, pixels).double()   # n_pixels x n^2
        Wfull = torch.zeros(len(pixels), n * n, dtype=torch.float64, device=device)
        for q, p in enumerate(pick):
            Wfull[q, ~hid] = W[(hid_idx == p).nonzero().item()]
        for q in range(len(pixels)):
            j, w = J[q], Wfull[q]
            stats.append(dict(rel_err=((j - w).norm() / w.norm()).item(),
                              cosine=((j @ w) / (j.norm() * w.norm())).item(),
                              gain_ratio=(j.sum() / w.sum()).item()))
        if b == 0:
            plot_rows(J[:3].cpu().numpy(), Wfull[:3].cpu().numpy(), pixels[:3], n,
                      os.path.join(outdir, f"jacobian_{os.path.basename(run)[:-5]}_h{h:.0f}.png"))
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out["n_rows"] = len(stats)
    print("jacobian vs wiener:", out)
    return out


def radial_profile(row, pixel, n, rmax):
    # mean weight at each integer distance from the target pixel, over observed pixels only
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d = np.round(np.sqrt((yy - pixel[0]) ** 2 + (xx - pixel[1]) ** 2)).astype(int).flatten()
    prof = []
    for r in range(rmax + 1):
        sel = (d == r) & (row != 0)
        prof.append(row[sel].mean() if sel.any() else np.nan)
    return np.array(prof)


def plot_rows(J, W, pixels, n, path, win=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(pixels)
    fig, ax = plt.subplots(k, 4, figsize=(13, 3.2 * k), squeeze=False)
    for q in range(k):
        i, j = pixels[q]
        y0, y1 = max(0, i - win), min(n, i + win + 1)
        x0, x1 = max(0, j - win), min(n, j + win + 1)
        v = max(abs(W[q]).max(), 1e-12)
        for col, (img, t) in enumerate([(J[q], "learned jacobian row"), (W[q], "exact wiener row"), (J[q] - W[q], "difference")]):
            ax[q, col].imshow(img.reshape(n, n)[y0:y1, x0:x1], cmap="RdBu_r", vmin=-v, vmax=v,
                              extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
            ax[q, col].plot(j, i, "k+", ms=10)
            ax[q, col].set_title(t, fontsize=9); ax[q, col].axis("off")
        pj, pw = radial_profile(J[q], (i, j), n, win), radial_profile(W[q], (i, j), n, win)
        ax[q, 3].plot(pw, "ks-", lw=2, ms=4, label="exact wiener")
        ax[q, 3].plot(pj, "o-", ms=4, label="learned")
        ax[q, 3].axhline(0, c="gray", lw=0.6)
        ax[q, 3].set_xlabel("distance from target pixel"); ax[q, 3].set_ylabel("mean weight")
        ax[q, 3].set_title("radial profile of the filter", fontsize=9); ax[q, 3].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


@torch.no_grad()
def probe_linearity(run, h, n_pairs, device):
    # a linear map satisfies f(a x1 + (1-a) x2) = a f(x1) + (1-a) f(x2) exactly (same mask)
    model, r = load(run, device)
    c = r["config"]
    clean, noisy, mask = batch(c, h, 2 * n_pairs, device)
    mask = mask[:1].expand(2 * n_pairs, -1, -1)
    x1, x2 = noisy[:n_pairs], noisy[n_pairs:]
    devs = []
    for a in [0.25, 0.5, 0.75]:
        mix = predict(model, c, a * x1 + (1 - a) * x2, mask[:n_pairs], h)
        lin = a * predict(model, c, x1, mask[:n_pairs], h) + (1 - a) * predict(model, c, x2, mask[:n_pairs], h)
        m = mask[:n_pairs]
        devs.append((((mix - lin) ** 2 * m).sum() / (lin ** 2 * m).sum()).sqrt().item())
    # affinity vs linearity: a linear map sends the zero input to zero; report |f(0)| / |f(x)|
    zero = predict(model, c, torch.zeros_like(x1), mask[:n_pairs], h)
    fx = predict(model, c, x1, mask[:n_pairs], h)
    m = mask[:n_pairs]
    offset = ((zero ** 2 * m).sum() / (fx ** 2 * m).sum()).sqrt().item()
    out = dict(rel_nonlinearity=float(np.mean(devs)), affine_offset_ratio=float(offset))
    print("linearity:", out)
    return out


def ntk(model, c, noisy, mask, h, pixels_per_input, sketch_dim=8192, seed=0):
    """Gram matrix of parameter gradients over (input, hidden pixel) outputs, through a
    count-sketch of the gradients: each parameter is hashed to one of sketch_dim buckets with a
    random sign, so inner products are estimated with relative error about 1/sqrt(sketch_dim)
    and 128 outputs of a 50M-parameter model fit in memory. The same hash is used for the
    initial and final weights so the two Grams are comparable."""
    params = list(model.parameters())
    P = sum(q.numel() for q in params)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, sketch_dim, (P,), generator=g).to(noisy.device)
    sign = (torch.randint(0, 2, (P,), generator=g).float() * 2 - 1).to(noisy.device)
    rows = []
    for b in range(noisy.shape[0]):
        pred = predict(model, c, noisy[b:b + 1], mask[b:b + 1], h)[0]
        n = pred.shape[0]
        hid = torch.nonzero(mask[b].flatten()).flatten()
        rng = torch.Generator().manual_seed(b)
        for q in hid[torch.randperm(len(hid), generator=rng)[:pixels_per_input]]:
            grads = torch.autograd.grad(pred[q // n, q % n], params, retain_graph=True, allow_unused=True)
            flat = torch.cat([(x if x is not None else torch.zeros_like(prm)).flatten() for x, prm in zip(grads, params)])
            rows.append(torch.zeros(sketch_dim, device=flat.device).index_add_(0, idx, flat * sign))
    S = torch.stack(rows)
    return S @ S.T


def ntk_stats(K0, K1):
    def centred(K):
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1.0 / n
        return H @ K @ H
    def cos(A, B):
        return ((A * B).sum() / (A.norm() * B.norm())).item()
    v0 = torch.linalg.eigh(K0)[1][:, -1]
    v1 = torch.linalg.eigh(K1)[1][:, -1]
    return dict(alignment=cos(K0, K1), centred_alignment=cos(centred(K0), centred(K1)),
                offdiag_alignment=cos(K0 - torch.diag(torch.diag(K0)), K1 - torch.diag(torch.diag(K1))),
                top_eigenvector_overlap=abs((v0 @ v1).item()),
                rel_change=((K1 - K0).norm() / K0.norm()).item(), scale_ratio=(K1.trace() / K0.trace()).item(),
                n_outputs=int(K0.shape[0]))


def probe_ntk(run, h, n_inputs, pixels_per_input, device):
    m0, r = load(run, device, "init")
    m1, _ = load(run, device, "final")
    c = r["config"]
    clean, noisy, mask = batch(c, h, n_inputs, device)
    K0 = ntk(m0, c, noisy, mask, h, pixels_per_input)
    K1 = ntk(m1, c, noisy, mask, h, pixels_per_input)
    out = ntk_stats(K0, K1)
    print("ntk init vs final:", out)
    return out


def probe_widths(pattern, h, device, outdir):
    # Problem 3C summary: loss vs width under SP and muP, plus NTK drift per width.
    # Needs initial and final weights for every width run; refuses to run otherwise so the
    # committed results/width_sweep.json (which has the NTK numbers) is never overwritten
    # with a loss-only version.
    files = sorted(glob.glob(pattern))
    missing = [f for f in files if not (os.path.exists(f.replace(".json", ".pt")) and os.path.exists(f.replace(".json", "_init.pt")))]
    if not files or missing:
        raise SystemExit(f"probes widths needs .pt and _init.pt for every width run; missing for {len(missing)} of {len(files)}. "
                         f"The committed numbers are in results/width_sweep.json; plot them with: python -m magscale.plots widths")
    rows = []
    for f in files:
        r = json.load(open(f))
        c = r["config"]
        row = dict(width=c["width"], par=("mup" if c["mup"] else "sp") + f" lr{c['lr']:g}" + (" wd0" if c.get("wd", 0.05) == 0 else ""),
                   run=os.path.basename(f)[:-5], loss=r["best_val"][str(float(h))], floor=r["floor"].get(str(float(h))))
        row.update(probe_ntk(f, h, 16, 8, device))
        rows.append(row)
        print(row)
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/width_sweep.json", "w"), indent=1)
    from .plots import widths
    widths("results/width_sweep.json", os.path.join(outdir, "width_sweep.png"))


@torch.no_grad()
def plot_recon(run, h, device, outdir, k=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model, r = load(run, device)
    c = r["config"]
    n = c["n"]
    clean, noisy, mask = batch(c, h, k, device, seed=7)
    pred = predict(model, c, noisy, mask, h)
    K = patch_covariance(n, c["beta"], h, c["dx"], c["sigma0"], device)
    fig, ax = plt.subplots(k, 5, figsize=(15, 3 * k), squeeze=False)
    for b in range(k):
        hid = mask[b].flatten()
        W, _ = wiener_matrix(K, hid, c["sigma"])
        opt = noisy[b].flatten().double().clone()
        opt[hid] = W @ noisy[b].flatten().double()[~hid]
        opt = opt.view(n, n).float()
        obs = noisy[b].clone(); obs[mask[b]] = float("nan")
        v = clean[b].abs().max().item()
        panels = [(clean[b], "truth"), (obs, "observed"), (pred[b], "network"), (opt, "exact wiener"),
                  ((pred[b] - clean[b]) * mask[b], "network error (hidden)")]
        for col, (img, t) in enumerate(panels):
            vv = v if col < 4 else 3 * c["sigma"]
            ax[b, col].imshow(img.cpu().numpy(), cmap="RdBu_r", vmin=-vv, vmax=vv)
            ax[b, col].set_title(t, fontsize=9); ax[b, col].axis("off")
        e_net = ((pred[b] - clean[b]) ** 2 * mask[b]).sum() / mask[b].sum()
        e_opt = ((opt - clean[b]) ** 2 * mask[b]).sum() / mask[b].sum()
        ax[b, 0].set_title(f"truth   (net {e_net:.2f} / optimal {e_opt:.2f} nT^2)", fontsize=9)
    fig.suptitle(f"{os.path.basename(run)[:-5]}  beta={c['beta']} h={h:.0f} m")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"recon_{os.path.basename(run)[:-5]}_h{h:.0f}.png"), dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["jacobian", "linearity", "ntk", "widths", "recon", "all"])
    p.add_argument("--run", help="runs/<run>.json (needs --save at training time)")
    p.add_argument("--pattern", default="runs/width*.json")
    p.add_argument("--h", type=float, default=200.0)
    p.add_argument("--n_inputs", type=int, default=4)
    p.add_argument("--n_pixels", type=int, default=4)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    if a.what in ("jacobian", "all"):
        res["jacobian"] = probe_jacobian(a.run, a.h, a.n_inputs, a.n_pixels, device, a.out)
    if a.what in ("linearity", "all"):
        res["linearity"] = probe_linearity(a.run, a.h, a.n_inputs, device)
    if a.what in ("ntk", "all"):
        res["ntk"] = probe_ntk(a.run, a.h, 16, 8, device)
    if a.what in ("recon", "all"):
        plot_recon(a.run, a.h, device, a.out)
    if a.what == "widths":
        probe_widths(a.pattern, a.h, device, a.out)
    if res:
        os.makedirs("results", exist_ok=True)
        json.dump(res, open(f"results/probes_{os.path.basename(a.run)[:-5]}_h{a.h:.0f}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

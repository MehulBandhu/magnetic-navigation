"""Problem 3B reference: a linear learner on the same task.

For a fixed mask the Bayes predictor is linear, so the kernel-regime baseline is
linear regression from observed pixels to hidden pixels, fitted from D patches.
Its learning curve L_lin(D) is what kernel/linear theory describes, and it can be
computed exactly for D up to millions on a CPU, because only the Gram matrices
X^T X (|O| x |O|) and X^T Y (|O| x |H|) are ever stored.

Two estimators per D: ridgeless (min-norm) and ridge with lambda chosen by the
exact excess risk (oracle), which is what optimally-regularised kernel theory
describes. Excess risk is exact, no test-set noise: E|(W - W_opt)^T x|^2.

python -m magscale.linear_ref --beta 3.5 --h 200 --out results/linear_b3.5_h200.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .grf import sample_fields, add_noise, make_mask
from .floor import patch_covariance, wiener_matrix


def gram(a, mask, D, seed, chunk=2048, device="cpu"):
    """accumulate X^T X and X^T Y over D patches with a fixed mask; X = noisy observed, Y = clean hidden"""
    o, hdn = ~mask.flatten(), mask.flatten()
    XtX = torch.zeros(int(o.sum()), int(o.sum()), dtype=torch.float64, device=device)
    XtY = torch.zeros(int(o.sum()), int(hdn.sum()), dtype=torch.float64, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    done = 0
    while done < D:
        b = min(chunk, D - done)
        clean = sample_fields(b, a.n, a.beta, a.h, a.dx, a.sigma0, gen, device).double()
        noisy = add_noise(clean, a.sigma, gen)
        X = noisy.view(b, -1)[:, o]
        Y = clean.view(b, -1)[:, hdn]
        XtX += X.T @ X
        XtY += X.T @ Y
        done += b
    return XtX, XtY


def test_error(W, a, mask, n_test, seed, device="cpu"):
    o, hdn = ~mask.flatten(), mask.flatten()
    gen = torch.Generator(device=device).manual_seed(seed)
    clean = sample_fields(n_test, a.n, a.beta, a.h, a.dx, a.sigma0, gen, device).double()
    noisy = add_noise(clean, a.sigma, gen)
    pred = noisy.view(n_test, -1)[:, o] @ W
    return ((pred - clean.view(n_test, -1)[:, hdn]) ** 2).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=3.5)
    p.add_argument("--h", type=float, default=200.0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--dx", type=float, default=100.0)
    p.add_argument("--sigma0", type=float, default=50.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--mask", default="random")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--spacing", type=int, default=4)
    p.add_argument("--D", default="256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_test", type=int, default=2048)
    p.add_argument("--out", default="results/linear.json")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ds = [int(x) for x in a.D.split(",")]

    gen = torch.Generator(device=device).manual_seed(1000 + a.seed)
    mask = make_mask(a.mask, 1, a.n, a.ratio, a.spacing, gen, device)[0]
    o, hdn = ~mask.flatten(), mask.flatten()
    K = patch_covariance(a.n, a.beta, a.h, a.dx, a.sigma0, device)
    W_opt, var = wiener_matrix(K, hdn, a.sigma)
    floor = var.mean().item()
    Sxx = K[o][:, o] + a.sigma ** 2 * torch.eye(int(o.sum()), dtype=K.dtype, device=device)
    nH = int(hdn.sum())

    def excess(W):
        # exact excess risk: E|(W - W_opt)^T x|^2 / |H|, no test-set noise at all
        d = W - W_opt.T
        return ((Sxx @ d) * d).sum().item() / nH
    err_opt = test_error(W_opt.T, a, mask, a.n_test, 77, device)
    n_obs = int(o.sum())
    print(f"beta {a.beta} h {a.h}: |O|={n_obs} |H|={int(hdn.sum())} floor {floor:.4f} (wiener on test {err_opt:.4f})")

    lams = 10.0 ** np.arange(-6, 6, 0.5)
    rows = []
    XtX, XtY, done = None, None, 0
    t0 = time.time()
    for D in Ds:
        # grow the Gram matrices: later D reuse the earlier patches, so curves are nested
        g1, g2 = gram(a, mask, D - done, 100 * a.seed + done, device=device)
        XtX, XtY = (g1, g2) if XtX is None else (XtX + g1, XtY + g2)
        done = D
        row = dict(D=D)
        # ridgeless / min-norm
        W = torch.linalg.lstsq(XtX, XtY, driver="gelsd").solution if D < n_obs else torch.linalg.solve(XtX, XtY)
        row["ridgeless"] = test_error(W, a, mask, a.n_test, 77, device)
        row["excess_ridgeless"] = excess(W)
        # oracle ridge: lambda chosen by the exact excess (W_opt is known), i.e. the best any
        # ridge estimator could do. This is the "optimally regularised" curve theory describes.
        best = None
        tr = XtX.trace().item() / n_obs
        for lam in lams:
            Wl = torch.linalg.solve(XtX + lam * tr * torch.eye(n_obs, dtype=XtX.dtype, device=device), XtY)
            v = excess(Wl)
            if best is None or v < best[0]:
                best = (v, lam, Wl)
        row["ridge"] = test_error(best[2], a, mask, a.n_test, 77, device)
        row["excess_ridge"] = excess(best[2])
        row["lambda"] = best[1]
        row["W_err"] = ((best[2] - W_opt.T).norm() / W_opt.norm()).item()   # how far from the Wiener matrix
        rows.append(row)
        print(f"D {D:8d}  excess ridgeless {row['excess_ridgeless']:10.4f}  ridge {row['excess_ridge']:9.5f}  "
              f"|W-Wopt|/|Wopt| {row['W_err']:.3f}  [{time.time()-t0:.0f}s]")

    # local exponent of the exact excess error, ridge estimator
    ex = np.array([r["excess_ridge"] for r in rows])
    Dn = np.array([r["D"] for r in rows], float)
    slopes = -np.diff(np.log(ex)) / np.diff(np.log(Dn))
    out = dict(config=vars(a), n_obs=n_obs, floor=floor, wiener_test=err_opt, rows=rows,
               local_delta=[dict(D_mid=float(math.sqrt(Dn[i] * Dn[i + 1])), delta=float(slopes[i])) for i in range(len(slopes))])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("local delta:", " ".join(f"{s:.2f}" for s in slopes))


if __name__ == "__main__":
    main()

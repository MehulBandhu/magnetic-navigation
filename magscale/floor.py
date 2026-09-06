"""Problem 2A: the best achievable masked reconstruction error.

The maps are Gaussian, so E[m_hidden | y_observed] is linear and its error is
the posterior variance. No estimator can achieve lower error than these values.
"""
import math
import torch

from .grf import amplitude, covariance_lags, make_mask


def patch_covariance(n, beta, h, dx=100.0, sigma0=50.0, device="cpu", gen_n=None):
    """n^2 x n^2 covariance of an n-patch cropped from a periodic generation grid of size
    gen_n (default 2n, which is what sample_fields uses for n-patches) (float64)"""
    ng = gen_n or 2 * n
    cov = covariance_lags(amplitude(ng, beta, h, dx, sigma0, device))
    iy, ix = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
    iy, ix = iy.flatten(), ix.flatten()
    dy = (iy[:, None] - iy[None, :]) % ng
    dxx = (ix[:, None] - ix[None, :]) % ng
    return cov[dy, dxx]


def wiener_matrix(K, hidden, sigma):
    """W with E[m_H | y_O] = W y_O, plus the posterior variance of each hidden pixel"""
    o = ~hidden
    Koo = K[o][:, o] + sigma ** 2 * torch.eye(int(o.sum()), dtype=K.dtype, device=K.device)
    Kho = K[hidden][:, o]
    L = torch.linalg.cholesky(Koo)
    W = torch.cholesky_solve(Kho.T, L).T            # Kho Koo^-1
    var = K[hidden, hidden] - (W * Kho).sum(1)      # diag(Khh - Kho Koo^-1 Koh)
    return W, var


def conditional_floor(K, hidden, sigma):
    return wiener_matrix(K, hidden, sigma)[1].mean()


def exact_floor(n, beta, h, sigma, mask_kind="random", ratio=0.75, spacing=4,
                dx=100.0, sigma0=50.0, n_masks=16, seed=0, device="cpu"):
    """mean posterior variance (nT^2) over hidden pixels, averaged over masks"""
    K = patch_covariance(n, beta, h, dx, sigma0, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    vals = []
    for _ in range(n_masks):
        m = make_mask(mask_kind, 1, n, ratio, spacing, gen, device)[0].flatten()
        vals.append(conditional_floor(K, m, sigma).item())
    vals = torch.tensor(vals)
    return vals.mean().item(), vals.std().item() / math.sqrt(n_masks)


def prior_variance(n, beta, h, dx=100.0, sigma0=50.0, device="cpu"):
    # per-pixel variance at altitude h; the trivial predictor (zero) scores this
    A = amplitude(2 * n, beta, h, dx, sigma0, device)
    return ((A ** 2).sum() / (2 * n) ** 2).item()


def dense_wiener(n, beta, h, sigma, dx=100.0, sigma0=50.0, device="cpu"):
    # every pixel observed: per-mode error A^2 s^2 / (A^2 + s^2), periodic grid
    A2 = amplitude(2 * n, beta, h, dx, sigma0, device) ** 2
    return ((A2 * sigma ** 2) / (A2 + sigma ** 2)).sum().item() / (2 * n) ** 2


def denoise_floor(K, sigma):
    # same thing on the finite patch: conditional variance given all n^2 noisy pixels
    N = K.shape[0]
    L = torch.linalg.cholesky(K + sigma ** 2 * torch.eye(N, dtype=K.dtype, device=K.device))
    return (K.diagonal() - (K * torch.cholesky_solve(K, L)).sum(1)).mean().item()


def effective_modes(n, beta, h, sigma, dx=100.0, sigma0=50.0, device="cpu"):
    # number of Fourier modes per patch whose signal power beats the noise.
    # this is the "effective rank" that Problem 3 needs.
    A2 = amplitude(2 * n, beta, h, dx, sigma0, device) ** 2
    return int((A2 > sigma ** 2).sum().item()) * n ** 2 / (2 * n) ** 2


def difficulty_table(betas, hs, sigma, n=64, mask_kind="random", ratio=0.75, spacing=4,
                     dx=100.0, sigma0=50.0, n_masks=8, device="cpu"):
    rows = []
    for b in betas:
        for h in hs:
            fl, err = exact_floor(n, b, h, sigma, mask_kind, ratio, spacing, dx, sigma0, n_masks, device=device)
            pv = prior_variance(n, b, h, dx, sigma0, device)
            rows.append(dict(beta=b, h=h, floor=fl, floor_err=err, prior_var=pv,
                             frac=fl / pv, modes=effective_modes(n, b, h, sigma, dx, sigma0, device)))
    return rows


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--betas", default="2.5,3.0,3.5,4.0")
    p.add_argument("--hs", default="0,100,200,400,800")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--mask", default="random")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--spacing", type=int, default=4)
    p.add_argument("--out", default="results/floor.json")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = difficulty_table([float(x) for x in a.betas.split(",")], [float(x) for x in a.hs.split(",")],
                            a.sigma, a.n, a.mask, a.ratio, a.spacing, device=dev)
    print(f"{'beta':>5} {'h[m]':>6} {'floor nT^2':>11} {'prior nT^2':>11} {'floor/prior':>11} {'modes':>7}")
    for r in rows:
        print(f"{r['beta']:5.2f} {r['h']:6.0f} {r['floor']:11.4f} {r['prior_var']:11.2f} {r['frac']:11.4f} {r['modes']:7.0f}")
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)

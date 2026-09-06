"""Problem 3C: test the deep-linear time-exponent derivation without an ODE solver.

Gradient flow on the population loss for a two-layer linear network with balanced small
initialisation decouples in the singular basis of the whitened cross-covariance (Saxe et al.
2014). Per mode with target singular value s and initial strength u0 the product of the two
layers follows the logistic

    u(t) = s / (1 + (s/u0 - 1) exp(-2 s t)),

and the excess loss is sum_k (u_k(t) - s_k)^2. The one-layer model on the same whitened inputs
converges at one rate for every mode; on raw inputs mode k converges at rate lambda_k.

Two settings are computed exactly here:
  unmasked periodic patch: modes are Fourier modes, s_k = A_k^2 / sqrt(A_k^2 + sigma^2)
  masked, whitened: modes are the singular vectors of K_HO (K_OO + sigma^2 I)^-1/2
The derivation predicts excess ~ t^-2(beta-2)/beta for the unmasked case above the noise floor.
The masked raw-input case (no whitening) is not decoupled and is left as a conjecture.

python -m magscale.deep_linear   ->  results/deep_linear.json, figures/deep_linear.png
"""
import json
import os

import numpy as np
import torch

from .grf import amplitude, random_mask
from .floor import patch_covariance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def two_layer_excess(s, ts, u0=1e-3, norm=1.0):
    # sum over modes of (u_k(t) - s_k)^2 with the logistic solution, divided by the number of
    # pixels the modes describe so the result is a per-pixel error in nT^2
    s = s[s > 0]
    out = []
    for t in ts:
        u = s / (1 + (s / u0 - 1) * np.exp(-2 * s * t))
        out.append(float(np.sum((u - s) ** 2)) / norm)
    return np.array(out)


def one_layer_excess(s, rates, ts, norm=1.0):
    # w_k(t) = s_k (1 - exp(-rate_k t)) in the whitened metric; excess = sum (w_k - s_k)^2
    return np.array([float(np.sum((s * np.exp(-rates * t)) ** 2)) / norm for t in ts])


def local_slope(ts, ex, lo, hi):
    m = (ts >= lo) & (ts <= hi) & (ex > 0)
    return float(-np.polyfit(np.log(ts[m]), np.log(ex[m]), 1)[0]) if m.sum() > 3 else float("nan")


def switch_time(s, u0=1e-3):
    # the logistic for mode k reaches half its target at t_k = log(s_k/u0) / (2 s_k)
    return np.log(s / u0) / (2 * s)


def power_law_window(s, u0=1e-3, lo_rank=16, hi_rank=1024):
    # the derivation counts modes learned in order of s; its power-law regime is where the
    # switching front sits between rank lo_rank and hi_rank (capped at the modes that carry
    # more than 1e-3 of the largest singular value)
    srt = np.sort(s)[::-1]
    hi_rank = min(hi_rank, int((srt > 1e-3 * srt[0]).sum()) - 1)
    return switch_time(srt[lo_rank], u0), switch_time(srt[hi_rank], u0), lo_rank, hi_rank


def spectrum_exponent(s, lo_rank, hi_rank):
    # c in s_rho ~ rho^-c over the same ranks; the counting argument then predicts a time
    # exponent (2c - 1) / c whatever produced the spectrum
    srt = np.sort(s)[::-1]
    r = np.arange(1, len(srt) + 1)
    m = (r >= lo_rank) & (r <= hi_rank)
    c = float(-np.polyfit(np.log(r[m]), np.log(srt[m]), 1)[0])
    return c, (2 * c - 1) / c


def unmasked(beta, h, n=64, dx=100.0, sigma0=50.0, sigma=1.0):
    A2 = (amplitude(2 * n, beta, h, dx, sigma0) ** 2).numpy().flatten()
    A2 = A2[A2 > 0]
    lam = A2 + sigma ** 2                       # input variance per mode (signal + noise)
    s = A2 / np.sqrt(lam)                       # whitened cross-covariance
    return s, lam


def masked_whitened(beta, h, n=64, sigma=1.0, ratio=0.75, seed=0):
    K = patch_covariance(n, beta, h)
    hid = random_mask(1, n, ratio, torch.Generator().manual_seed(1000 + seed))[0].flatten()
    Koo = K[~hid][:, ~hid] + sigma ** 2 * torch.eye(int((~hid).sum()), dtype=K.dtype)
    Kho = K[hid][:, ~hid]
    evals, evecs = torch.linalg.eigh(Koo)
    Sxx_inv_half = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    M = Kho @ Sxx_inv_half                      # whitened cross-covariance, |H| x |O|
    s = torch.linalg.svdvals(M).numpy()
    return s, evals.numpy()


def main():
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    ts = np.logspace(-3, 4, 120)
    res = {"unmasked": {}, "masked_whitened": {}}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # ground level: the spectrum is a pure power law up to the grid cutoff, which is the
    # regime the derivation describes. At altitude the exponential cutoff dominates and the
    # curves fall faster than any power law, as the write-up says they should.
    for beta in [2.5, 3.0, 3.5, 4.0]:
        s, lam = unmasked(beta, 0.0)
        norm = (2 * 64) ** 2                                          # modes of the generation grid -> per pixel
        ex2 = two_layer_excess(s, ts, norm=norm)
        ex1w = one_layer_excess(s, np.full_like(s, 2.0), ts, norm=norm)   # whitened: one rate for every mode
        lo, hi, r0, r1 = power_law_window(s)
        pred = 2 * (beta - 2) / beta
        c, pred_c = spectrum_exponent(s, r0, r1)
        slope2 = local_slope(ts, ex2, lo, hi)
        res["unmasked"][str(beta)] = dict(predicted_from_beta=pred, spectrum_exponent_c=c, predicted_from_spectrum=pred_c,
                                          two_layer_slope=slope2, window=[float(lo), float(hi)],
                                          one_layer_whitened_slope=local_slope(ts, ex1w, lo, hi))
        l, = ax[0].loglog(ts, ex2, label=f"beta={beta}: slope {slope2:.2f}; predicted {pred:.2f} (from beta), {pred_c:.2f} (from spectrum)")
        ax[0].loglog(ts, ex1w, ":", c=l.get_color(), lw=1)
        ax[0].axvspan(lo, hi, color=l.get_color(), alpha=0.05)
        print(f"unmasked beta {beta}, h=0: spectrum c {c:.2f} (beta/4 = {beta/4:.2f}); two-layer slope {slope2:.2f} in [{lo:.2g}, {hi:.2g}]; "
              f"predicted {pred:.2f} from beta, {pred_c:.2f} from the spectrum; one-layer whitened {res['unmasked'][str(beta)]['one_layer_whitened_slope']:.2f}")
    ax[0].set_ylim(1e-2, 5e3); ax[0].set_xlim(1e-3, 1e2)
    ax[0].set_xlabel("gradient-flow time (whitened units)"); ax[0].set_ylabel("excess per pixel [nT^2]")
    ax[0].set_title("unmasked patch, ground level; dotted: one-layer, whitened inputs; shaded: fitting window", fontsize=8); ax[0].legend(fontsize=7)

    for beta, h in [(3.5, 0.0), (3.5, 200.0), (3.5, 800.0)]:
        s, ev = masked_whitened(beta, h)
        ex2 = two_layer_excess(s, ts, norm=3072)                      # per hidden pixel
        lo, hi, r0, r1 = power_law_window(s, hi_rank=min(1000, len(s) - 1))
        c, pred_c = spectrum_exponent(s, r0, r1)
        slope = local_slope(ts, ex2, lo, hi)
        res["masked_whitened"][f"{beta}_{h:.0f}"] = dict(two_layer_slope=slope, spectrum_exponent_c=c, predicted_from_spectrum=pred_c,
                                                        predicted_from_beta=2 * (beta - 2) / beta, ranks=[r0, r1], n_modes=int(len(s)))
        l, = ax[1].loglog(ts, ex2, label=f"h={h:.0f} m: slope {slope:.2f}; from spectrum {pred_c:.2f}, from beta {2*(beta-2)/beta:.2f}")
        ax[1].axvspan(lo, hi, color=l.get_color(), alpha=0.05)
        print(f"masked whitened beta {beta} h {h:.0f}: spectrum c {c:.2f} over ranks {r0}-{r1}; two-layer slope {slope:.2f}; "
              f"predicted {pred_c:.2f} from the spectrum, {2*(beta-2)/beta:.2f} from beta alone")
    ax[1].set_ylim(1e-2, 5e3); ax[1].set_xlim(1e-3, 1e2)
    ax[1].set_xlabel("gradient-flow time (whitened units)"); ax[1].set_ylabel("excess per hidden pixel [nT^2]")
    ax[1].set_title("75% random mask, whitened inputs, beta = 3.5", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/deep_linear.png", dpi=130)
    json.dump(res, open("results/deep_linear.json", "w"), indent=1)


if __name__ == "__main__":
    main()

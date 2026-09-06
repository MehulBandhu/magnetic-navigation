"""Problem 2B: compute-optimal allocation from the committed loss curves, at no GPU cost.

Every run logs the validation loss every 250 steps. Treating each logged point as (N, S, loss)
with S = steps x batch (examples seen) gives a grid in N and S from which the additive form
excess = A N^-alpha + B S^-delta_S can be fitted; the compute-optimal allocation then follows
from C = 6 N S T (T = 256 tokens per example):

    N* ~ C^(delta_S / (alpha + delta_S)),   S* ~ C^(alpha / (alpha + delta_S)).

Caveat, quantified in the write-up: the loss at step s of a run with a longer cosine schedule is
not the loss of an s-step run. The long run at its step 8000 (0.876) beat the run whose schedule
ended at 8000 (1.104), a 21% effect, against an order-of-magnitude spread across sizes.

python -m magscale.compute_optimal  ->  results/compute_optimal.json, figures/compute_optimal.png
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .fit import is_grid_run, fit_additive, NAME  # noqa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKENS = 256


def curves(pattern="runs/*.json", min_D=4096, late_fraction=1 / 3):
    """(N, S, excess, run, batch) at every logged evaluation after the first third of each run;
    the early part of a cosine schedule is dominated by the warm-up and is not what the
    additive form describes"""
    pts = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        if not is_grid_run(f, r) or r["D_actual"] < min_D:
            continue
        m = NAME.match(os.path.basename(f))
        key = (float(m.group(2)), float(m.group(3)))
        h = str(key[1])
        fl = r["floor"][h]
        batch = r["config"]["batch"]
        for e in r["log"]:
            if e["step"] >= late_fraction * r["config"]["steps"]:
                pts[key].append((r["params"], e["step"] * batch, max(e["val"][h] - fl, 1e-6), os.path.basename(f), batch))
    return pts


def equal_step_slope(pts, step, sizes=None):
    """slope of log(excess) vs log N with every size read at the same step, from the logged
    curves; removes the step-count part of the size-dependent budget (the learning rate still
    differs by size)"""
    by_size = defaultdict(list)
    for N, S, L, name, batch in pts:
        if S == step * batch and (sizes is None or N in sizes):
            by_size[N].append(L)
    if len(by_size) < 3:
        return None
    Ns = np.array(sorted(by_size)); Ls = np.array([np.exp(np.mean(np.log(by_size[n]))) for n in Ns])
    return float(-np.polyfit(np.log(Ns), np.log(Ls), 1)[0]), int(len(Ns))


def main():
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    pts = curves()
    out = {}
    fig, axes = plt.subplots(1, len(pts), figsize=(5 * len(pts), 4), squeeze=False)
    for ax, key in zip(axes[0], sorted(pts)):
        N = np.array([p[0] for p in pts[key]], float); S = np.array([p[1] for p in pts[key]], float)
        L = np.array([p[2] for p in pts[key]]); names = [p[3] for p in pts[key]]
        fit, _ = fit_additive(N, S, L, 0.0)            # excess = A N^-alpha + B S^-delta_S
        a, d = fit["alpha"], fit["delta"]
        C = 6 * N * S * TOKENS
        # envelope: best excess reached at each compute level by any run
        bins = np.logspace(np.log10(C.min()), np.log10(C.max()), 25)
        env_C, env_L, env_N = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (C >= lo) & (C < hi)
            if m.any():
                i = np.argmin(L[m]); env_C.append(np.sqrt(lo * hi)); env_L.append(L[m][i]); env_N.append(N[m][i])
        # fitted optimum: minimise A N^-a + B (C/(6 N T))^-d over N at each C
        Cg = np.logspace(np.log10(C.min()), np.log10(C.max()), 40)
        Ng = np.logspace(np.log10(N.min()) - 0.5, np.log10(N.max()) + 0.5, 400)
        Nstar, Lstar = [], []
        for c in Cg:
            Ls = fit["A"] * Ng ** -a + fit["B"] * (c / (6 * Ng * TOKENS)) ** -d
            Nstar.append(Ng[np.argmin(Ls)]); Lstar.append(Ls.min())
        # the equal-step slopes need the early evaluations too, so they read the full curves
        full = curves(late_fraction=0.0)[key]
        eq4 = equal_step_slope(full, 4000)                                       # all seven sizes
        eq8 = equal_step_slope(full, 8000, sizes={n for n in np.unique(N) if n > 4e6})  # l, xl, xxl
        # model-free allocation: the size of the best run at each compute level along the envelope
        env = np.array([(c_, n_) for c_, l_, n_ in zip(env_C, env_L, env_N)])
        env_exp = float(np.polyfit(np.log(env[:, 0]), np.log(env[:, 1]), 1)[0]) if len(env) > 3 else float("nan")
        out[f"{key[0]}_{key[1]:.0f}"] = dict(beta=key[0], h=key[1], alpha=a, delta_S=d,
                                             N_exponent=d / (a + d), S_exponent=a / (a + d),
                                             envelope_N_exponent=env_exp,
                                             alpha_equal_steps_4000=eq4[0] if eq4 else None, n_sizes_4000=eq4[1] if eq4 else 0,
                                             alpha_equal_steps_8000_large=eq8[0] if eq8 else None, n_sizes_8000=eq8[1] if eq8 else 0,
                                             n_points=int(len(L)), n_runs=len(set(names)),
                                             envelope=[dict(C=float(c), excess=float(l), N=float(n)) for c, l, n in zip(env_C, env_L, env_N)],
                                             fitted_optimum=[dict(C=float(c), N=float(n), excess=float(l)) for c, n, l in zip(Cg, Nstar, Lstar)])
        print(f"beta {key[0]} h {key[1]:.0f}: fit on the last two thirds of each run: alpha {a:.2f}, delta_S {d:.2f}, N* ~ C^{d/(a+d):.2f}; "
              f"envelope (model-free): best N ~ C^{env_exp:.2f}  ({len(set(names))} runs, {len(L)} points)")
        print(f"   equal-step slope at 4000 steps, all sizes: {eq4[0] if eq4 else float('nan'):.2f};  at 8000 steps, l to xxl: {eq8[0] if eq8 else float('nan'):.2f}")
        for n in np.unique(N):
            m = N == n
            o = np.argsort(C[m])
            ax.loglog(C[m][o], L[m][o], "-", lw=0.8, alpha=0.6, label=f"N={n/1e6:.2f}M")
        ax.loglog(env_C, env_L, "k.-", lw=1.5, label="envelope (best at each C)")
        ax.loglog(Cg, Lstar, "k--", lw=1, label=f"fitted optimum, N* ~ C^{d/(a+d):.2f}; envelope: best N ~ C^{env_exp:.2f}")
        ax.set_xlabel("compute C = 6 N S T [FLOP]"); ax.set_ylabel("excess over floor [nT^2]")
        ax.set_title(f"beta={key[0]}, h={key[1]:.0f} m", fontsize=9); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig("figures/compute_optimal.png", dpi=130)
    json.dump(out, open("results/compute_optimal.json", "w"), indent=1)


if __name__ == "__main__":
    main()

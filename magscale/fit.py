"""Problem 2B/2C: scaling fits on the excess over the exact floor.

The grid does not follow a single power law in N and one in D (see results/table), so the
primary numbers are:
  alpha_global   OLS slope of log(excess) vs log N over all sizes, D >= 4096, seed-aware bootstrap
  alpha_small / alpha_large   the same slope below and above the knee between m (1.3M) and l (5M)
  knee           the excess ratio across that step
  delta_local    slope of log(excess) vs log D per model size, between consecutive D values
The additive form E + A N^-alpha + B D^-delta (Chinchilla) is fitted too, in log-excess with E
fixed to the exact floor, and a floor profile (refit at each candidate E on a grid) shows whether
the grid identifies the floor at all. The loss used per run is the best value on
a fixed validation set of 2048 patches over the logged evaluations; that carries a small optimistic
bias (of order 1%, the eval-to-eval scatter), stated in the write-up rather than corrected.

The alpha values are slopes under a fixed training protocol (steps and learning rate per size),
and most large runs were still improving at the end, so they mix capacity and optimisation.

python -m magscale.fit                # main grid
python -m magscale.fit --tag _fm      # fixed-mask study, with the linear reference at matched D
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

SIZES = ["xxs", "xs", "s", "m", "l", "xl", "xxl"]
SMALL, LARGE = ["xxs", "xs", "s", "m"], ["l", "xl", "xxl"]
NAME = re.compile(r"^vit(xxs|xs|s|m|l|xl|xxl)_b([\d.]+)_h(\d+)_D(\d+)_s(\d+)(_fm)?\.json$")
TOKENS = 256


def is_grid_run(path, r, tag=""):
    m = NAME.match(os.path.basename(path))
    if not m or (m.group(6) or "") != tag:
        return False
    c = r["config"]
    return (c["model"] == "vit" and not c.get("mup") and not c.get("no_altitude")
            and c.get("mask", "random") == "random" and "," not in str(c["h"])
            and bool(c.get("fixed_mask", False)) == (tag == "_fm"))


def converged(r, h):
    # loss still falling by more than 5% over the last quarter of training counts as not converged
    vals = [e["val"][h] for e in r["log"] if e["step"] > 0]
    q = max(1, len(vals) // 4)
    return abs(vals[-1] - vals[-q]) / max(vals[-q], 1e-12) < 0.05


def load_runs(pattern, tag=""):
    cells = defaultdict(lambda: defaultdict(list))   # (beta, h) -> (size, N, D) -> [excess]
    floors, unconv = {}, defaultdict(int)
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        if not is_grid_run(f, r, tag):
            continue
        m = NAME.match(os.path.basename(f))
        key = (float(m.group(2)), float(m.group(3)))
        h = str(key[1])
        ex = r["best_val"][h] - r["floor"][h]
        cells[key][(m.group(1), r["params"], r["D_actual"])].append(max(ex, 1e-6))
        floors[key] = r["floor"][h]
        unconv[key] += 0 if converged(r, h) else 1
    return cells, floors, unconv


def cell_arrays(cells):
    sz = np.array([k[0] for k in cells])
    N = np.array([k[1] for k in cells], float)
    D = np.array([k[2] for k in cells], float)
    L = np.array([np.mean(v) for v in cells.values()])
    vals = [list(v) for v in cells.values()]
    nseed = np.array([len(v) for v in cells.values()])
    # log-space scatter per cell: from seeds where available, else the median seed scatter
    sd = np.array([np.std(np.log(v), ddof=1) if len(v) > 1 else np.nan for v in cells.values()])
    fill = np.nanmedian(sd) if np.isfinite(sd).any() else 0.2
    sd = np.where(np.isfinite(sd), sd, fill)
    return sz, N, D, L, nseed, sd, vals


def ols_slope(x, y, w=None):
    w = np.ones_like(x) if w is None else w
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def slope_with_error(N, L, sd, nboot=400, seed=0, values=None):
    """slope of log(excess) vs log N across model sizes.

    Point estimate: each cell (size, D) is one equally weighted estimate (its mean over seeds);
    each size is the geometric mean of its cells; OLS across sizes.
    Uncertainty: cell-level bootstrap. In every replicate, a cell with several seeds is resampled
    among its seeds, a cell with one seed is perturbed in log space by the proxy scatter (median
    seed scatter over replicated cells), the size means and the slope are recomputed. A heuristic
    band, not a confidence interval from replicated protocols."""
    values = [[l] for l in L] if values is None else values
    sizes = np.unique(N)
    cells_by_size = {n: [(vals, s_) for n_, vals, s_ in zip(N, values, sd) if n_ == n] for n in sizes}
    x = np.log(sizes)

    def size_logmeans(rng=None):
        out = []
        for n in sizes:
            logs = []
            for vals, s_ in cells_by_size[n]:
                if rng is None:
                    logs.append(np.mean(np.log(vals)))
                elif len(vals) > 1:
                    logs.append(np.mean(np.log(rng.choice(vals, len(vals)))))
                else:
                    logs.append(math.log(vals[0]) + rng.normal(0, s_))
            out.append(np.mean(logs))
        return np.array(out)

    s = -ols_slope(x, size_logmeans())
    rng = np.random.default_rng(seed)
    bs = [-ols_slope(x, size_logmeans(rng)) for _ in range(nboot)]
    return s, float(np.std(bs))


def local_slopes(sz, N, D, L, sd=None, values=None, nboot=400, seed=0):
    """delta between consecutive dataset sizes, per model size, with a band from the same
    cell-level bootstrap as the alpha slopes (seeds resampled, singletons perturbed by the proxy)"""
    out = {}
    rng = np.random.default_rng(seed)
    for s in SIZES:
        m = np.where(sz == s)[0]
        if len(m) < 2:
            continue
        o = m[np.argsort(D[m])]
        segs = []
        for i, j in zip(o[:-1], o[1:]):
            dlogD = np.log(D[j]) - np.log(D[i])
            point = -(np.log(L[j]) - np.log(L[i])) / dlogD
            band = float("nan")
            if sd is not None:
                def draw(k):
                    v = values[k]
                    return np.mean(np.log(rng.choice(v, len(v)))) if len(v) > 1 else math.log(L[k]) + rng.normal(0, sd[k])
                band = float(np.std([-(draw(j) - draw(i)) / dlogD for _ in range(nboot)]))
            segs.append(dict(D_from=int(D[i]), D_to=int(D[j]), delta=float(point), err=band))
        out[s] = segs
    return out


def huber(x, d=1e-3):
    return np.where(np.abs(x) <= d, 0.5 * x ** 2, d * (np.abs(x) - 0.5 * d))


def fit_additive(N, D, L, E_fixed=None, E_floor=0.0):
    """excess = A N^-a + B D^-d (E fixed) or excess = shift + A N^-a + B D^-d (E free, shift in
    [-E_floor, ...] so the fitted floor can be below or above the exact one), Huber in log space"""
    free = E_fixed is None

    def pred(p):
        return np.exp(p[0]) * N ** -p[1] + np.exp(p[2]) * D ** -p[3]

    def obj(p):
        with np.errstate(all="ignore"):
            pr, tgt = (p[4] + pred(p), L) if free else (pred(p), L)
        if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
            return 1e6
        return 1e4 * float(huber(np.log(pr) - np.log(tgt)).sum())

    # a free floor may sit below the exact one (down to zero) as well as above it
    bounds = [(-20, 40), (0.01, 3), (-20, 40), (0.01, 3)] + ([(-E_floor, 0.999 * L.min())] if free else [])
    best = None
    for a in [0.2, 0.5, 1.0]:
        for d in [0.2, 0.5, 1.0]:
            x0 = [math.log(np.median(L) * np.median(N) ** a), a, math.log(np.median(L) * np.median(D) ** d), d]
            if free:
                x0.append(0.0)
            r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options=dict(ftol=1e-14, gtol=1e-10, maxiter=5000))
            if best is None or r.fun < best.fun:
                best = r
    r = minimize(obj, best.x, method="Nelder-Mead", bounds=bounds, options=dict(xatol=1e-7, fatol=1e-12, maxiter=20000))
    p = r.x if r.fun < best.fun else best.x
    out = dict(A=math.exp(p[0]), alpha=float(p[1]), B=math.exp(p[2]), delta=float(p[3]))
    if free:
        out["E_shift"] = float(p[4])   # fitted floor = E_exact + E_shift when the data are excesses
    return out, pred


def loo_additive(N, D, L, E_fixed):
    errs = []
    for i in range(len(N)):
        keep = np.arange(len(N)) != i
        fit, _ = fit_additive(N[keep], D[keep], L[keep], E_fixed)
        pr = fit["A"] * N[i] ** -fit["alpha"] + fit["B"] * D[i] ** -fit["delta"]
        errs.append(abs(math.log(pr) - math.log(L[i])))
    return float(np.mean(errs))


def size_means(N, L):
    sizes = np.unique(N)
    return sizes, np.array([np.exp(np.mean(np.log(L[N == n]))) for n in sizes])


def broken_pred(p, N):
    # A N^-a1 below the knee, continuous, A Nk^(a2-a1) N^-a2 above it
    A, a1, a2, Nk = math.exp(p[0]), p[1], p[2], math.exp(p[3])
    return np.where(N <= Nk, A * N ** -a1, A * Nk ** (a2 - a1) * N ** -a2)


def fit_broken(N, L):
    """excess = broken power law in N, fitted to the per-size means in log space"""
    sizes, means = size_means(N, L)
    x, y = sizes, np.log(means)

    def obj(p):
        with np.errstate(all="ignore"):
            pr = broken_pred(p, x)
        return 1e4 * float(huber(np.log(pr) - y).sum()) if np.all(np.isfinite(pr)) and np.all(pr > 0) else 1e6

    lo, hi = math.log(x.min()), math.log(x.max())
    best = None
    for a1 in [0.2, 0.5, 1.0]:
        for a2 in [0.2, 0.5, 1.0]:
            for nk in np.linspace(lo, hi, 7)[1:-1]:
                x0 = [math.log(means[0]) + a1 * math.log(x[0]), a1, a2, nk]
                r = minimize(obj, x0, method="L-BFGS-B", bounds=[(-20, 40), (0, 3), (0, 3), (lo, hi)],
                             options=dict(ftol=1e-14, gtol=1e-10, maxiter=5000))
                if best is None or r.fun < best.fun:
                    best = r
    p = best.x
    return dict(A=math.exp(p[0]), alpha_below=float(p[1]), alpha_above=float(p[2]), knee_N=float(math.exp(p[3])))


def loo_broken(N, L):
    # leave one model size out, fit, predict its mean
    sizes, means = size_means(N, L)
    errs = []
    for i, n in enumerate(sizes):
        keep = N != n
        p = fit_broken(N[keep], L[keep])
        pr = broken_pred([math.log(p["A"]), p["alpha_below"], p["alpha_above"], math.log(p["knee_N"])], np.array([n]))[0]
        errs.append(abs(math.log(pr) - math.log(means[i])))
    return float(np.mean(errs))


def loo_additive_sizes(N, D, L, E):
    # the additive form judged on the same footing: leave one model size out; the held-out
    # size's prediction is the geometric mean over its D values, matching how the observation
    # is aggregated
    sizes, means = size_means(N, L)
    errs = []
    for i, n in enumerate(sizes):
        keep = N != n
        fit, _ = fit_additive(N[keep], D[keep], L[keep], E)
        pr = np.exp(np.mean([math.log(fit["A"] * n ** -fit["alpha"] + fit["B"] * d ** -fit["delta"]) for d in D[N == n]]))
        errs.append(abs(math.log(pr) - math.log(means[i])))
    return float(np.mean(errs))


def fit_additive_total(N, D, L_total, E):
    """loss = E + A N^-a + B D^-d with E fixed, fitted and scored in log TOTAL loss. This is the
    objective that makes different candidate floors comparable: the earlier version fitted and
    scored log(loss - E), a quantity whose scale changes with E, which favoured small E."""
    def pred(p):
        return E + np.exp(p[0]) * N ** -p[1] + np.exp(p[2]) * D ** -p[3]

    def obj(p):
        with np.errstate(all="ignore"):
            pr = pred(p)
        if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
            return 1e6
        return 1e4 * float(huber(np.log(pr) - np.log(L_total)).sum())

    bounds = [(-20, 40), (0.01, 3), (-20, 40), (0.01, 3)]
    ex0 = np.maximum(L_total - E, 1e-3 * L_total)
    best = None
    for a in [0.2, 0.5, 1.0]:
        for d in [0.2, 0.5, 1.0]:
            x0 = [math.log(np.median(ex0) * np.median(N) ** a), a, math.log(np.median(ex0) * np.median(D) ** d), d]
            r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options=dict(ftol=1e-14, gtol=1e-10, maxiter=5000))
            if best is None or r.fun < best.fun:
                best = r
    r = minimize(obj, best.x, method="Nelder-Mead", bounds=bounds, options=dict(xatol=1e-7, fatol=1e-12, maxiter=20000))
    p = r.x if r.fun < best.fun else best.x
    pr = pred(p)
    return dict(A=math.exp(p[0]), alpha=float(p[1]), B=math.exp(p[2]), delta=float(p[3]),
                err=float(np.mean(np.abs(np.log(pr) - np.log(L_total)))))


def profile_floor(N, D, L_excess, floor, npts=31):
    """Is the floor identified by the grid? For each candidate floor E on a grid from 0 to
    1.5 x exact, fit loss = E + A N^-a + B D^-d and score it in log total loss, the same objective
    at every E. The minimum of the profile is the free-floor estimate; a flat profile means the
    data cannot tell the floors apart."""
    L_total = L_excess + floor
    grid = np.unique(np.concatenate([np.linspace(0, 1.5 * floor, npts), [floor]]))
    rows = []
    for E in grid:
        if E >= L_total.min():
            continue
        fit = fit_additive_total(N, D, L_total, float(E))
        rows.append(dict(E=float(E), err=fit["err"], alpha=fit["alpha"], delta=fit["delta"]))
    errs = np.array([r["err"] for r in rows]); Es = np.array([r["E"] for r in rows])
    best = int(np.argmin(errs))
    near = Es[errs <= errs.min() * 1.05]
    return dict(profile=rows, best_E=float(Es[best]), best_err=float(errs[best]),
                best_E_over_exact=float(Es[best] / floor),
                range_within_5pct=[float(near.min()), float(near.max())],
                err_at_exact=float(errs[np.argmin(np.abs(Es - floor))]))


def additive_bootstrap(N, D, L, sd, values, floor, nboot=100, seed=0):
    """bands on the additive fit's alpha and delta (E fixed to the exact floor) from the same
    cell-level bootstrap as the slopes: seeds resampled where they exist, singletons perturbed
    by the proxy scatter, the fit repeated. Heuristic, like the slope bands."""
    rng = np.random.default_rng(seed)
    al, de = [], []
    for _ in range(nboot):
        Lb = np.array([np.exp(np.mean(np.log(rng.choice(v, len(v))))) if len(v) > 1 else L[k] * math.exp(rng.normal(0, sd[k]))
                       for k, v in enumerate(values)])
        fit, _ = fit_additive(N, D, Lb, floor)
        al.append(fit["alpha"]); de.append(fit["delta"])
    return float(np.std(al)), float(np.std(de))


def linear_reference(beta, h):
    rs = [json.load(open(f)) for f in glob.glob(f"results/linear_b{beta}_h{h:.0f}_s*.json")]
    if not rs:
        return None
    Ds = [row["D"] for row in rs[0]["rows"]]
    ex = np.mean([[row["excess_ridge"] for row in r["rows"]] for r in rs], axis=0)
    return dict(zip(Ds, ex.tolist()))


def analyse(key, cells, floor, unconv, tag, nboot):
    sz, N, D, L, nseed, sd, vals = cell_arrays(cells)
    res = dict(beta=key[0], h=key[1], floor=floor, n_cells=len(L), n_runs=int(nseed.sum()), n_unconverged=unconv,
               cells=[dict(size=s, N=int(n), D=int(d), excess=float(l), seeds=int(k)) for s, n, d, l, k in zip(sz, N, D, L, nseed)])
    big = D >= 4096 if tag == "" else D >= 1024
    if big.sum() >= 3:
        sub = lambda m: (N[m], L[m], sd[m], [v for v, k in zip(vals, m) if k])
        res["alpha_global"], res["alpha_global_err"] = slope_with_error(*sub(big)[:3], nboot, values=sub(big)[3])
        ms = big & np.isin(sz, SMALL); ml = big & np.isin(sz, LARGE)
        if len(np.unique(N[ms])) >= 2:
            res["alpha_small"], res["alpha_small_err"] = slope_with_error(*sub(ms)[:3], nboot, values=sub(ms)[3])
        if len(np.unique(N[ml])) >= 2:
            res["alpha_large"], res["alpha_large_err"] = slope_with_error(*sub(ml)[:3], nboot, values=sub(ml)[3])
        # alternative functional form: a broken power law in N with a fitted knee, on the same
        # per-size means; compared with the additive form by leave-one-size-out error
        bp = fit_broken(N[big], L[big])
        bp["loo"] = loo_broken(N[big], L[big])
        res["broken_power_law"] = bp
        res["additive_loo_sizes"] = loo_additive_sizes(N[big], D[big], L[big], floor)
        em = L[big & (sz == "m")]; el = L[big & (sz == "l")]
        if len(em) and len(el):
            res["knee_m_over_l"] = float(em.mean() / el.mean())
    res["delta_local"] = local_slopes(sz, N, D, L, sd, vals, nboot)
    if len(np.unique(N)) >= 3 and len(np.unique(D)) >= 2:
        fx, _ = fit_additive(N, D, L, floor)
        fx["loo"] = loo_additive(N, D, L, floor)
        fx["delta_identified"] = bool(fx["delta"] < 2.99)      # 3 is the optimiser's upper bound
        fx["alpha_err"], fx["delta_err"] = additive_bootstrap(N, D, L, sd, vals, floor, nboot=min(nboot, 100))
        res["additive_fixed_floor"] = fx
        res["floor_profile"] = profile_floor(N, D, L, floor)
        # the same question restricted to the sizes above the knee
        ml = np.isin(sz, LARGE)
        if len(np.unique(N[ml])) >= 3:
            res["floor_profile_large"] = profile_floor(N[ml], D[ml], L[ml], floor)
        # no compute-optimal allocation is reported: compute here scales with N x steps at a
        # fixed batch, while D was varied independently, so C = 6 N D T does not describe the protocol
    if tag == "_fm":
        lin = linear_reference(key[0], key[1])
        if lin:
            res["linear_reference"] = {str(int(d)): lin[int(d)] for d in np.unique(D) if int(d) in lin}
    return res


def report(res, tag):
    print(f"\nbeta={res['beta']} h={res['h']:.0f} m {'FIXED MASK' if tag else ''}  floor {res['floor']:.4f}  "
          f"{res['n_runs']} runs in {res['n_cells']} cells, {res['n_unconverged']} still improving >5% at the end")
    if "alpha_global" in res:
        line = f"  alpha (fixed-budget protocol slope, size-level bootstrap): all sizes {res['alpha_global']:.2f} +- {res['alpha_global_err']:.2f}"
        if "alpha_small" in res:
            line += f"   below knee (xs..m) {res['alpha_small']:.2f} +- {res['alpha_small_err']:.2f}"
        if "alpha_large" in res:
            line += f"   above knee (l..xxl) {res['alpha_large']:.2f} +- {res['alpha_large_err']:.2f}"
        if "knee_m_over_l" in res:
            line += f"   knee m/l = {res['knee_m_over_l']:.1f}x"
        print(line)
    for s, segs in res["delta_local"].items():
        print(f"  delta {s:3s}: " + "  ".join(f"{g['D_from']}->{g['D_to']}: {g['delta']:5.2f}+-{g['err']:.2f}" for g in segs))
    if "broken_power_law" in res:
        b = res["broken_power_law"]
        print(f"  broken power law in N: {b['alpha_below']:.2f} below a knee at {b['knee_N']/1e6:.1f}M, {b['alpha_above']:.2f} above;"
              f"  leave-one-size-out log error {b['loo']:.2f} vs {res['additive_loo_sizes']:.2f} for the additive form")
    if "additive_fixed_floor" in res:
        fx, pf = res["additive_fixed_floor"], res["floor_profile"]
        print(f"  additive form, E fixed: alpha {fx['alpha']:.2f} +- {fx.get('alpha_err', float('nan')):.2f}  delta {fx['delta']:.2f} +- {fx.get('delta_err', float('nan')):.2f} (loo over cells {fx['loo']:.2f}; bands are the cell-level bootstrap)")
        print(f"  floor profile (log total loss, same objective at every E): best E {pf['best_E']:.3f} = {pf['best_E_over_exact']:.2f} x exact (err {pf['best_err']:.3f}), exact {res['floor']:.3f} (err {pf['err_at_exact']:.3f}); "
              f"within 5% of the minimum for E in [{pf['range_within_5pct'][0]:.3f}, {pf['range_within_5pct'][1]:.3f}]")
    if "linear_reference" in res:
        print("  linear learner at the same mask and D: " + "  ".join(f"D={d}: {v:.4f}" for d, v in res["linear_reference"].items()))


def plot_condition(res, tag, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = res["cells"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for Dv in sorted({c["D"] for c in cells}):
        pts = sorted([c for c in cells if c["D"] == Dv], key=lambda c: c["N"])
        ax[0].plot([c["N"] for c in pts], [c["excess"] for c in pts], "o-", ms=5, label=f"D={Dv}")
    if "alpha_global" in res:
        Ng = np.logspace(5, 7.7, 20)
        big = [c for c in cells if c["D"] >= (4096 if tag == "" else 1024)]
        ref = np.exp(np.mean([math.log(c["excess"]) + res["alpha_global"] * math.log(c["N"]) for c in big]))
        ax[0].plot(Ng, ref * Ng ** -res["alpha_global"], "k--", lw=1, label=f"slope {res['alpha_global']:.2f}")
    for s in SIZES:
        pts = sorted([c for c in cells if c["size"] == s], key=lambda c: c["D"])
        if len(pts) > 1:
            ax[1].plot([c["D"] for c in pts], [c["excess"] for c in pts], "s-", ms=5, label=f"{s} ({pts[0]['N']/1e6:.2f}M)")
    if "linear_reference" in res:
        ds = sorted(int(d) for d in res["linear_reference"])
        ax[1].plot(ds, [res["linear_reference"][str(d)] for d in ds], "k^--", label="linear regression, same mask")
    for a, xl in zip(ax[:2], ["parameters N", "fields D"]):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(xl); a.set_ylabel("excess over exact floor [nT^2]"); a.legend(fontsize=7)
    if "floor_profile" in res:
        pf = res["floor_profile"]
        ax[2].plot([r["E"] for r in pf["profile"]], [r["err"] for r in pf["profile"]], "o-", ms=3)
        ax[2].axvline(res["floor"], c="k", ls="--", lw=1, label="exact floor")
        ax[2].axhline(pf["best_err"] * 1.05, c="gray", lw=0.8, label="5% above the minimum")
        ax[2].set_xlabel("candidate floor E [nT^2]"); ax[2].set_ylabel("mean |log error| of the additive fit"); ax[2].legend(fontsize=7)
        ax[2].set_title("is the floor identified by the grid?", fontsize=9)
    fig.suptitle(f"beta={res['beta']}  h={res['h']:.0f} m  {'fixed mask' if tag else 'random mask, resampled every batch'}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_summary(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    rs = sorted([r for r in results if r["beta"] == 3.5 and "alpha_global" in r], key=lambda r: r["h"])
    if rs:
        hs = [r["h"] for r in rs]
        ax[0].errorbar(hs, [r["alpha_global"] for r in rs], [r["alpha_global_err"] for r in rs], marker="o", capsize=3, label="all sizes")
        ax[0].errorbar(hs, [r.get("alpha_large", np.nan) for r in rs], [r.get("alpha_large_err", 0) for r in rs], marker="s", capsize=3, label="l to xxl")
        ax[0].set_xlabel("altitude h [m]"); ax[0].set_ylabel("alpha (beta = 3.5)"); ax[0].legend(fontsize=8)
        for r in rs:
            segs = r["delta_local"].get("l", [])
            ax[1].plot([math.sqrt(g["D_from"] * g["D_to"]) for g in segs], [g["delta"] for g in segs], "o-", label=f"h={r['h']:.0f} m")
        ax[1].set_xscale("log"); ax[1].set_xlabel("fields D"); ax[1].set_ylabel("local delta, 5M model"); ax[1].axhline(0, c="gray", lw=0.8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/*.json")
    p.add_argument("--tag", default="", help="'' main grid, '_fm' fixed-mask study")
    p.add_argument("--nboot", type=int, default=400)
    p.add_argument("--out", default="results")
    p.add_argument("--figures", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figures, exist_ok=True)
    cells, floors, unconv = load_runs(a.runs, a.tag)
    print(f"{sum(sum(len(v) for v in c.values()) for c in cells.values())} runs selected")
    results = []
    for key in sorted(cells):
        res = analyse(key, cells[key], floors[key], unconv[key], a.tag, a.nboot)
        report(res, a.tag)
        plot_condition(res, a.tag, f"{a.figures}/scaling{a.tag}_b{key[0]}_h{key[1]:.0f}.png")
        results.append(res)
    json.dump(results, open(f"{a.out}/fit{a.tag}.json", "w"), indent=1)
    if a.tag == "" and results:
        plot_summary(results, f"{a.figures}/exponents_vs_h.png")


if __name__ == "__main__":
    main()

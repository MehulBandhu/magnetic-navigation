"""Figures from the run jsons that need no extra compute.

python -m magscale.plots curves    runs/vitxxl_b3.5_h200_D131072_s0_ref.json runs/..._long.json ...
python -m magscale.plots arch      --pattern "runs/*b3.5_h200_D131072*.json"     loss vs params, unet vs vit
python -m magscale.plots altitude  runs/vitxxl_b3.5_hmix_D131072_s0_alt.json runs/vitxxl_b3.5_hmix_D131072_s0_noalt.json
python -m magscale.plots all       (the three above with the default file names)
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    r = json.load(open(path))
    r["name"] = os.path.basename(path)[:-5]
    return r


def short(name):
    # strip the condition prefix so legends stay readable
    for tag in ["_ref", "_long", "_bigdata", "_noalt", "_lines", "_alt"]:
        if name.endswith(tag):
            return tag[1:]
    return name.split("_")[0]


def curves(paths, h, out):
    runs = [load(p) for p in paths]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in runs:
        steps = [e["step"] for e in r["log"] if e["step"] > 0]
        val = [e["val"][h] for e in r["log"] if e["step"] > 0]
        c = r["config"]
        label = f"{short(r['name'])}: {c['model']} {r['params']/1e6:.1f}M, D={r.get('D_actual', 0) or 'inf'}, {c['steps']} steps"
        ax.plot(steps, val, label=label)
    fl = runs[0]["floor"][h]
    ax.axhline(fl, ls="--", c="k", lw=1, label=f"exact floor {fl:.3f}")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("val loss on hidden pixels [nT^2]")
    c = runs[0]["config"]
    ax.set_title(f"beta={c['beta']}  h={float(h):.0f} m", fontsize=10)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def arch(pattern, h, out):
    runs = [load(p) for p in glob.glob(pattern)]
    runs = [r for r in runs if h in r["best_val"] and "hmix" not in r["name"]
            and r["config"].get("mask", "random") == "random" and not r["config"].get("no_altitude")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fl = runs[0]["floor"][h]
    for model, marker in [("unet", "s"), ("vit", "o")]:
        rs = sorted([r for r in runs if r["config"]["model"] == model], key=lambda r: r["params"])
        if not rs:
            continue
        ax.plot([r["params"] for r in rs], [r["best_val"][h] - fl for r in rs], marker, ms=7,
                label=model + " (label = steps)")
        for r in rs:
            ax.annotate(str(r["config"]["steps"]), (r["params"], r["best_val"][h] - fl), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("parameters"); ax.set_ylabel("L - floor  [nT^2]")
    c = runs[0]["config"]
    ax.set_title(f"same data (D={runs[0].get('D_actual')}), beta={c['beta']}, h={float(h):.0f} m", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def altitude(paths, out):
    runs = [load(p) for p in paths]
    hs = sorted(runs[0]["best_val"], key=float)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    w = 0.35
    x = np.arange(len(hs))
    for i, r in enumerate(runs):
        ex = [r["best_val"][h] - r["floor"][h] for h in hs]
        rel = [r["best_val"][h] / r["floor"][h] for h in hs]
        lab = "no altitude input" if r["config"]["no_altitude"] else "FiLM altitude conditioning"
        ax[0].bar(x + (i - 0.5) * w, ex, w, label=lab)
        ax[1].bar(x + (i - 0.5) * w, rel, w, label=lab)
    for a, yl in zip(ax, ["L - floor  [nT^2]", "L / floor"]):
        a.set_xticks(x); a.set_xticklabels([f"{float(h):.0f} m" for h in hs]); a.set_ylabel(yl); a.legend(fontsize=8)
    ax[0].set_yscale("log")
    ax[1].axhline(1, c="k", lw=0.8)
    fig.suptitle("one model trained on all three altitudes", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def linear(pattern, out):
    """Problem 3B reference: the linear learner's learning curves from results/linear_*.json.
    Left: excess risk vs D for beta=3.5 at every altitude, with the p/D law. Middle: local
    exponent vs D. Right: small-D exponent (D <= 1024, oracle ridge) over the (beta, h) grid."""
    files = glob.glob(pattern)
    runs = {}
    for f in files:
        r = json.load(open(f))
        c = r["config"]
        runs.setdefault((c["beta"], c["h"]), []).append(r)
    betas = sorted({k[0] for k in runs}); hs = sorted({k[1] for k in runs})

    def curve(rs, key):
        D = np.array([row["D"] for row in rs[0]["rows"]], float)
        ex = np.mean([[row[key] for row in r["rows"]] for r in rs], axis=0)
        return D, ex

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    b0 = 3.5 if 3.5 in betas else betas[-1]
    for h in hs:
        if (b0, h) not in runs:
            continue
        rs = runs[(b0, h)]
        D, ex = curve(rs, "excess_ridge")
        Dl, exl = curve(rs, "excess_ridgeless")
        l, = ax[0].loglog(D, ex, "o-", ms=4, label=f"h={h:.0f} m")
        ax[0].loglog(Dl[Dl > 1024], exl[Dl > 1024], "--", c=l.get_color(), lw=1)
        p = rs[0]["n_obs"]; fl = rs[0]["floor"]
        ax[0].loglog(D[D > 2 * p], fl * p / D[D > 2 * p], ":", c=l.get_color(), lw=1)
        loc = -np.diff(np.log(ex)) / np.diff(np.log(D))
        ax[1].semilogx(np.sqrt(D[1:] * D[:-1]), loc, "o-", ms=4, c=l.get_color(), label=f"h={h:.0f} m")
    ax[0].axvline(1024, c="gray", lw=0.8); ax[0].text(1100, ax[0].get_ylim()[1] * 0.5, "D = |O| = 1024", fontsize=7)
    ax[0].set_xlabel("training patches D"); ax[0].set_ylabel("exact excess risk [nT^2]")
    ax[0].set_title(f"linear learner, beta={b0}: oracle ridge (solid), ridgeless (dashed), floor x p/D (dotted)", fontsize=8)
    ax[0].legend(fontsize=7)
    ax[1].axhline(1, c="k", lw=0.8); ax[1].axvline(1024, c="gray", lw=0.8)
    ax[1].set_xlabel("D"); ax[1].set_ylabel("local exponent  -dlog(excess)/dlog(D)"); ax[1].set_title("delta along the curve", fontsize=9)
    ax[1].legend(fontsize=7)

    small = np.full((len(betas), len(hs)), np.nan)
    for i, b in enumerate(betas):
        for j, h in enumerate(hs):
            if (b, h) in runs:
                D, ex = curve(runs[(b, h)], "excess_ridge")
                m = (D >= 256) & (D <= 1024)
                small[i, j] = np.polyfit(np.log(D[m]), np.log(ex[m]), 1)[0] * -1
    im = ax[2].imshow(small, cmap="viridis", origin="lower", aspect="auto")
    ax[2].set_xticks(range(len(hs))); ax[2].set_xticklabels([f"{h:.0f}" for h in hs]); ax[2].set_xlabel("altitude h [m]")
    ax[2].set_yticks(range(len(betas))); ax[2].set_yticklabels([str(b) for b in betas]); ax[2].set_ylabel("beta")
    for i in range(len(betas)):
        for j in range(len(hs)):
            if np.isfinite(small[i, j]):
                ax[2].text(j, i, f"{small[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax[2].set_title("delta_linear for D in [256, 1024] (oracle ridge)", fontsize=9)
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("small-D delta_linear(beta, h):")
    print("beta\\h " + " ".join(f"{h:7.0f}" for h in hs))
    for i, b in enumerate(betas):
        print(f"{b:5.1f}  " + " ".join(f"{small[i, j]:7.2f}" for j in range(len(hs))))


def relabel(rows):
    # rows written by the current probe carry a `run` name and a wd0 label; an older
    # width_sweep.json labels the no-weight-decay run like the standard one, and the only
    # thing that distinguishes it is that it comes second in the sorted file order. Accept that
    # only when exactly one duplicated (width, par) pair exists.
    if all("run" in q for q in rows):
        for q in rows:
            if "wd0" in q["run"] and " wd0" not in q["par"]:
                q["par"] += " wd0"
        return rows
    seen, dup = set(), []
    for q in rows:
        key = (q["width"], q["par"])
        if key in seen:
            dup.append(q)
        seen.add(key)
    if len(dup) != 1:
        raise SystemExit(f"width_sweep.json has {len(dup)} duplicated (width, parametrisation) rows; expected exactly one (the wd0 run)")
    dup[0]["par"] += " wd0"
    return rows


def widths(path, out):
    """the four NTK/loss diagnostics vs width, from the numbers saved by `probes widths`"""
    rows = relabel(json.load(open(path)))
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    for par in sorted({q["par"] for q in rows}):
        rs = sorted([q for q in rows if q["par"] == par], key=lambda q: q["width"])
        w = [q["width"] for q in rs]
        ax[0].plot(w, [q["loss"] - q["floor"] for q in rs], "o-", label=par)
        if "alignment" in rs[0]:
            ax[1].plot(w, [q.get("centred_alignment", q["alignment"]) for q in rs], "o-", label=par)
            ax[2].plot(w, [q["rel_change"] for q in rs], "o-", label=par)
            ax[3].plot(w, [q["scale_ratio"] for q in rs], "o-" if " wd0" not in par else "x", label=par)
    for x in ax:
        x.set_xscale("log"); x.set_xlabel("width"); x.legend(fontsize=7)
    ax[0].set_yscale("log"); ax[0].set_ylabel("L - floor  [nT^2]")
    ax[1].set_ylabel("centred NTK alignment, init vs final")
    ax[2].set_yscale("log"); ax[2].set_ylabel("|NTK_final - NTK_init| / |NTK_init|")
    ax[3].set_yscale("log"); ax[3].set_ylabel("tr NTK_final / tr NTK_init"); ax[3].axhline(1, c="k", lw=0.8)
    fig.suptitle("a lazy model would have alignment 1, relative change 0 and scale ratio 1", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def altitude_difficulty(out):
    """the floor against altitude, absolute and relative to the field variance, next to the 38.5M
    model's excess, absolute and relative to the floor: easier in absolute terms, harder relative"""
    fl = json.load(open("results/floor.json"))
    fits = json.load(open("results/fit.json")) if os.path.exists("results/fit.json") else []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for beta in sorted({r["beta"] for r in fl}):
        rows = sorted([r for r in fl if r["beta"] == beta], key=lambda r: r["h"])
        hs = [max(r["h"], 30) for r in rows]
        l, = ax[0].loglog(hs, [r["floor"] for r in rows], "o-", label=f"beta={beta}: floor [nT^2]")
        ax[0].loglog(hs, [r["floor"] / r["prior_var"] for r in rows], "s--", c=l.get_color(), label=f"beta={beta}: floor / field variance")
    ax[0].set_xlabel("altitude h [m]  (0 plotted at 30)"); ax[0].legend(fontsize=6); ax[0].set_title("floor: falls with altitude, absolutely and relatively", fontsize=9)
    for a_ in ax:
        a_.set_xticks([30, 100, 200, 400, 800]); a_.set_xticklabels(["0", "100", "200", "400", "800"]); a_.minorticks_off()
    xxl = [(r["h"], min(c["excess"] for c in r["cells"] if c["size"] == "xxl"), r["floor"]) for r in fits if r["beta"] == 3.5 and any(c["size"] == "xxl" for c in r["cells"])]
    if xxl:
        xxl.sort()
        hs = [max(h, 30) for h, _, _ in xxl]
        ax[1].loglog(hs, [e for _, e, _ in xxl], "o-", label="38.5M model: excess over floor [nT^2]")
        ax[1].loglog(hs, [e / f for _, e, f in xxl], "s--", label="38.5M model: excess / floor")
        ax[1].set_xlabel("altitude h [m]  (0 plotted at 30)"); ax[1].legend(fontsize=7)
        ax[1].set_title("network at beta 3.5, grid budget: smaller absolute excess, larger relative", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["curves", "arch", "altitude", "linear", "widths", "difficulty", "all"])
    p.add_argument("paths", nargs="*")
    p.add_argument("--pattern", default="runs/*b3.5_h200_D131072*.json")
    p.add_argument("--h", default="200.0")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what in ("curves", "all"):
        paths = a.paths or ["runs/vitxxl_b3.5_h200_D131072_s0_ref.json", "runs/vitxxl_b3.5_h200_D131072_s0_long.json",
                            "runs/vitxxl_b3.5_h200_D1048576_s0_bigdata.json", "runs/unet64_b3.5_h200_D131072_s0.json"]
        curves([q for q in paths if os.path.exists(q)], a.h, f"{a.out}/gap_curves.png")
    if a.what in ("arch", "all"):
        arch(a.pattern, a.h, f"{a.out}/arch_vs_params.png")
    if a.what in ("linear", "all") and glob.glob("results/linear_*.json"):
        linear("results/linear_*.json", f"{a.out}/linear_reference.png")
    if a.what in ("difficulty", "all") and os.path.exists("results/floor.json"):
        altitude_difficulty(f"{a.out}/altitude_difficulty.png")
    if a.what in ("widths", "all") and os.path.exists("results/width_sweep.json"):
        widths("results/width_sweep.json", f"{a.out}/width_sweep.png")
    if a.what in ("altitude", "all"):
        paths = a.paths or ["runs/vitxxl_b3.5_hmix_D131072_s0_alt.json", "runs/vitxxl_b3.5_hmix_D131072_s0_noalt.json"]
        if all(os.path.exists(q) for q in paths):
            altitude(paths, f"{a.out}/altitude_ablation.png")
    print("figures in", a.out)


if __name__ == "__main__":
    main()

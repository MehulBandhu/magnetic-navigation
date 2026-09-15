"""Run the scripted demo and record it.

    python -m magsim.run_demo --map gawler --scenario full --record demo.rrd
    python -m magsim.run_demo --map synthetic --scenario calibration --spawn

Writes results/magsim_demo.json (the readouts of each part) and, for the navigation part,
results/magsim_nav.json and figures/magsim_demo.png (the still image on the front page).
"""
import argparse
import json
import os

import numpy as np

from . import scenario, world


def still_image(extras, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["true", "network", "wiener", "interp"]
    fig, ax = plt.subplots(len(extras), 5, figsize=(16, 3.6 * len(extras)), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.4]), squeeze=False)
    for row, (tag, (M, leg, nav)) in enumerate(extras.items()):
        ox, oy = M["origin_m"]; L = (M["n"] - 1) * M["dx_m"]
        v = np.abs(M["maps"]["true"]).max()
        for a, k in zip(ax[row, :4], keys):
            a.imshow(M["maps"][k], cmap="RdBu_r", vmin=-v, vmax=v, extent=[ox, ox + L, oy, oy + L], origin="lower")
            a.plot(leg["x_m"], leg["y_m"], "k-", lw=0.8)
            snaps = dict(nav[k]["particles"])
            for t, col in [(0, "orange"), (300, "gold"), (max(snaps), "red")]:
                tt = min(snaps, key=lambda s: abs(s - t))
                a.plot(snaps[tt][:, 0], snaps[tt][:, 1], ".", ms=1, color=col, alpha=0.5)
            a.set_title(f"{M['h_m']:.0f} m, {k}: err {np.sqrt(np.mean(nav[k]['error_m'][800:] ** 2)):.0f} m after the turn", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        t = leg["t_s"] - leg["t_s"][0]
        for k in keys:
            ax[row, 4].semilogy(t, np.maximum(nav[k]["error_m"], 0.1), label=k, lw=1)
        ax[row, 4].semilogy(t, np.maximum(nav["true"]["bound_m"], 0.1), "r--", lw=1, label="bound (true map)")
        ax[row, 4].set_xlabel("time on the leg [s]"); ax[row, 4].set_ylabel("position error [m]"); ax[row, 4].legend(fontsize=7)
        ax[row, 4].set_title(f"{M['h_m']:.0f} m over {M['dx_m']:.0f} m pixels", fontsize=9)
    fig.suptitle("Survey legs over the Gawler window: the particle cloud (orange: first reading, gold: 30 s, red: end) on the four maps", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="gawler", choices=["gawler", "synthetic"])
    p.add_argument("--scenario", default="full", choices=["full", "calibration", "interference", "navigation"])
    p.add_argument("--record", default=None, help="write the Rerun recording here (e.g. demo.rrd)")
    p.add_argument("--spawn", action="store_true", help="open the Rerun viewer")
    p.add_argument("--no-imu", action="store_true", help="give the estimator chain the true attitude")
    p.add_argument("--model", default="runs/corrector.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default="results/magsim_demo.json")
    a = p.parse_args()
    w = world.gawler() if a.map == "gawler" else world.synthetic(n=256, seed=a.seed)
    out, nav_extra = scenario.run(w, a.scenario, imu=not a.no_imu, model_path=a.model, record=a.record, spawn=a.spawn, seed=a.seed)
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    json.dump(out, open(a.results, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "part3_navigation"}, indent=1))
    if nav_extra:
        n3 = out["part3_navigation"]
        json.dump(dict(imu=not a.no_imu, spacing=4, sigma_nT=1.0, likelihood_sensor_rms_nT=n3["likelihood_sensor_rms_nT"],
                       legs={tag: {k: v for k, v in leg.items() if k != "tiles"} for tag, leg in n3["legs"].items()}), open("results/magsim_nav.json", "w"), indent=1)
        still_image(nav_extra, "figures/magsim_demo.png")
        for tag, leg in n3["legs"].items():
            print(f"leg at {leg['h_m']:.0f} m, {leg['dx_m']:.0f} m pixels, residual against the true map {leg['residual_rms_nT']:.2f} nT (slow part {leg['residual_slow_rms_nT']:.2f}, drift {leg['drift_nT_per_min']:.1f} nT per root minute)")
            for k, t in leg["maps"].items():
                print(f"  {t['label']:42s} map rms {t['map_rms_nT']:6.1f} nT   width {t['likelihood_sigma_nT']:5.1f}   error after the turn {t['error_after_turn_m']:7.1f} m (bound {t['bound_after_turn_m']:.1f})   from a 300 m box {t['error_after_turn_small_box_m']:7.1f} m (bound {t['bound_after_turn_small_box_m']:.1f})"
                      + (f"   white-noise reading {t['error_after_turn_white_noise_m']:.1f} m, with a constant offset {t['error_after_turn_white_noise_constant_offset_m']:.1f} m" if "error_after_turn_white_noise_m" in t else ""))
    print(f"wrote {a.results}" + (f" and {a.record}" if a.record else ""))


if __name__ == "__main__":
    main()

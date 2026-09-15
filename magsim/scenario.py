"""The scripted demo, about four minutes of simulated flight in four parts:

1. A survey fixed-wing flies the calibration box. The live Tolles-Lawson calibration shows
   the regressor rank climbing to 17 (16 band-passed) and the residual, the numbers of
   problem1/checks.py.
2. A multirotor (about 2700 nT of platform field, motors on) flies the box and then a free
   manoeuvre. The linear model calibrated on the box leaves tens of nT on the manoeuvre; the
   Gauss-Newton and lag baselines of problem1/edge_cases.py, fitted on the same box, and the
   corrector are shown as traces.
3. A survey leg over the Gawler window at 60 m with the fixed-wing, then the same leg at
   240 m over the same ground in 178 m pixels; the estimator chain is the calibration and
   the lag model (the corrector stays part 2's trace until its held-out table earns it a
   place in the chain). The particle filter runs on the true map and on the three reconstructions of maps.py; the
   position error of each is logged in metres next to the Cramer-Rao bound.
4. Throughout the leg, the tile under the aircraft: its Gaussian floor and the network's
   error on it, from the Part G evaluation run on the window's tiles.

The numbers the panels show go to results/magsim_demo.json and results/magsim_nav.json, from
which the README tables are rendered.
"""
import json
import math
import os

import numpy as np
import torch

from problem1.checks import B0
from problem1.edge_cases import gauss_newton, gn_predict, lag, rms
from magscale.grid_eval import evaluate

from . import aircraft, calibrate, corrector, maps, navigator, platform, sensor, viewer, world
from .train_corrector import HP_TAU_S, fit_lag_model

WINDOW = (1000, 1400, 256)         # rows, cols, size of the Gawler window used for the survey leg
CHECKS_LINE = "problem1/checks.py, check_tl_rank():  roll+pitch+yaw   raw rank 17  band-passed rank 16"


def estimator_inputs(w, flight, r, imu, rng):
    """what the estimator chain has: the attitude (true or from the IMU model), the direction
    cosines and the regressor built from it, and y = reading minus the ambient magnitude"""
    est = aircraft.imu(flight, rng) if imu else flight
    u, udot = aircraft.body_direction_cosines(est, w.incl_rad, w.decl_rad)
    A = calibrate.regressor(u, udot, r["ambient_nT"])
    return dict(est=est, u=u, udot=udot, A=A, y=r["reading_nT"] - r["ambient_nT"])


def log_flight_frames(flight, r, entity="world/vehicle", every=2):
    for t in range(0, len(flight["t_s"]), every):
        viewer.set_time(flight["t_s"][t])
        viewer.log_vehicle(entity, flight["x_m"][t], flight["y_m"][t], flight["h_m"][t], flight["roll_rad"][t], flight["pitch_rad"][t], flight["yaw_rad"][t], r["map_nT"][t])


def part_calibration(w, t0, imu, rng, log):
    """part 1: the fixed-wing box; returns the end time and the readouts"""
    v = platform.preset("survey_fixed_wing", seed=0)
    f = aircraft.fly(aircraft.calibration_box(leg_s=20.0, turn_s=5.0), *start_point(w, rng), 60.0, 30.0, 0.0)
    f = shift_time(f, t0)
    r = sensor.read(w, f, v, noise_nT=0.1, rng=rng)
    e = estimator_inputs(w, f, r, False, rng)          # the calibration panel reproduces the numerical check: true attitude, constant B0
    raw = calibrate.live(e["u"], e["udot"], e["y"], f["dt_s"])
    bp = calibrate.live(e["u"], e["udot"], e["y"], f["dt_s"], highpass_tau_s=HP_TAU_S)
    if log:
        viewer.log_text("text/overlay", f"Part 1: survey fixed-wing, calibration box. Live Tolles-Lawson calibration; the rank should reach 17 raw and 16 band-passed.\n{CHECKS_LINE}")
        log_flight_frames(f, r)
        for t in range(len(f["t_s"])):
            viewer.set_time(f["t_s"][t])
            viewer.log_scalar("plots/reading/raw", r["reading_nT"][t] - r["ambient_nT"][t])
            viewer.log_scalar("plots/reading/platform_first_order", r["first_order_nT"][t] - r["ambient_nT"][t])
            viewer.log_scalar("plots/calibration/raw_fit", raw["residual_nT"][t])
            viewer.log_scalar("plots/calibration/band_passed_fit", bp["residual_nT"][t])
            viewer.log_scalar("plots/rank/raw", raw["rank"][t])
            viewer.log_scalar("plots/rank/band_passed", bp["rank"][t])
        viewer.log_path("world/path/part1", f["x_m"], f["y_m"], r["map_nT"] * viewer.VERTICAL_M_PER_NT + f["h_m"])
    out = dict(rank_raw_end=int(raw["rank"][-1]), rank_band_passed_end=int(bp["rank"][-1]),
               time_to_rank_17_s=float(f["t_s"][np.argmax(raw["rank"] >= 17)] - t0), residual_rms_band_passed_nT=float(bp["rms_nT"][-1]),
               platform_field_mean_nT=float(np.linalg.norm(r["platform"]["Bp_nT"], axis=1).mean()))
    return f["t_s"][-1] + f["dt_s"], out


def part_interference(w, t0, imu, model, rng, log):
    """part 2: multirotor box then a free manoeuvre; baselines and corrector on the manoeuvre"""
    v = platform.preset("multirotor", seed=0)
    box = shift_time(aircraft.fly(aircraft.calibration_box(leg_s=20.0, turn_s=5.0), *start_point(w, rng), 60.0, 20.0, 0.0), t0)
    rb = sensor.read(w, box, v, noise_nT=0.1, rng=rng)
    eb = estimator_inputs(w, box, rb, imu, rng)
    hp = lambda x: calibrate.highpass(x, box["dt_s"], HP_TAU_S)
    coef = calibrate.fit(eb["A"], eb["y"], highpass_tau_s=HP_TAU_S, dt_s=box["dt_s"])
    live = calibrate.live(eb["u"], eb["udot"], eb["y"], box["dt_s"], B_nT=rb["ambient_nT"], highpass_tau_s=HP_TAU_S)
    # the physics baselines, fitted on the box as in train_corrector.baselines
    th_gn = gauss_newton(eb["u"], eb["udot"], rb["reading_nT"], iters=12, B=rb["ambient_nT"], return_theta=True, filt=hp)[1]
    tau, coef_lag = fit_lag_model(eb["u"], eb["udot"], eb["y"], rb["ambient_nT"], box["dt_s"])
    th_gnlag = gauss_newton(eb["u"], lag(eb["udot"], tau, box["dt_s"]), rb["reading_nT"], iters=12, B=rb["ambient_nT"], return_theta=True, filt=hp)[1]

    t1 = box["t_s"][-1] + box["dt_s"]
    man = shift_time(aircraft.fly([(15.0, math.radians(12), math.radians(8), math.radians(4)), (15.0, math.radians(8), math.radians(10), math.radians(-5)),
                                   (15.0, math.radians(14), math.radians(6), math.radians(3)), (15.0, math.radians(5), math.radians(9), 0.0)],
                                  box["x_m"][-1], box["y_m"][-1], 60.0, 20.0, box["yaw_rad"][-1]), t1)
    rm = sensor.read(w, man, v, noise_nT=0.1, rng=rng)
    em = estimator_inputs(w, man, rm, imu, rng)
    drift = rm["platform"]["drift_nT"]
    res_lin = em["y"] - calibrate.apply(coef, em["A"])
    res_gn = rm["reading_nT"] - gn_predict(th_gn, em["u"], em["udot"], B=rm["ambient_nT"])[0]
    res_lag = em["y"] - calibrate.apply(coef_lag, calibrate.regressor(em["u"], lag(em["udot"], tau, man["dt_s"]), rm["ambient_nT"]))
    res_gnlag = rm["reading_nT"] - gn_predict(th_gnlag, em["u"], lag(em["udot"], tau, man["dt_s"]), B=rm["ambient_nT"])[0]
    X = corrector.features(em["est"], em["A"], rm["platform"]["telemetry"], em["u"], em["udot"])
    res_corr = res_lin - corrector.predict(model, X, res_lin, False)
    res_lagcorr = res_lag - corrector.predict(model, X, res_lag, True)
    dm = lambda x: rms((x - drift) - (x - drift).mean())
    out = dict(platform_field_mean_nT=float(np.linalg.norm(rb["platform"]["Bp_nT"], axis=1).mean()), box_band_passed_rank=int(live["rank"][-1]),
               box_residual_rms_nT=float(live["rms_nT"][-1]), manoeuvre=dict(linear=dm(res_lin), gauss_newton=dm(res_gn), lag=dm(res_lag), gauss_newton_lag=dm(res_gnlag),
                                                                          corrector=dm(res_corr), lag_corrector=dm(res_lagcorr)))
    if log:
        m = out["manoeuvre"]
        viewer.log_text("text/overlay", f"Part 2: multirotor, about {out['platform_field_mean_nT']:.0f} nT of platform field with motors, battery and drift. Box: the linear model leaves {out['box_residual_rms_nT']:.1f} nT in band. "
                                        f"Then a free manoeuvre with the box's calibration applied: linear {m['linear']:.1f} nT, lag model {m['lag']:.1f}, Gauss-Newton + lag {m['gauss_newton_lag']:.1f}, "
                                        f"corrector after the linear model {m['corrector']:.1f}, after the lag model {m['lag_corrector']:.1f} nT (rms, drift and mean removed).")
        log_flight_frames(box, rb); log_flight_frames(man, rm)
        for t in range(len(box["t_s"])):
            viewer.set_time(box["t_s"][t])
            viewer.log_scalar("plots/reading/raw", rb["reading_nT"][t] - rb["ambient_nT"][t])
            viewer.log_scalar("plots/reading/platform_first_order", rb["first_order_nT"][t] - rb["ambient_nT"][t])
            viewer.log_scalar("plots/calibration/band_passed_fit", live["residual_nT"][t])
            viewer.log_scalar("plots/rank/band_passed", live["rank"][t])
        for t in range(len(man["t_s"])):
            viewer.set_time(man["t_s"][t])
            viewer.log_scalar("plots/reading/raw", rm["reading_nT"][t] - rm["ambient_nT"][t])
            viewer.log_scalar("plots/corrector/linear", res_lin[t] - drift[t])
            viewer.log_scalar("plots/corrector/gauss_newton", res_gn[t] - drift[t])
            viewer.log_scalar("plots/corrector/lag", res_lag[t] - drift[t])
            viewer.log_scalar("plots/corrector/gauss_newton_lag", res_gnlag[t] - drift[t])
            viewer.log_scalar("plots/corrector/corrector", res_corr[t] - drift[t])
            viewer.log_scalar("plots/corrector/lag_corrector", res_lagcorr[t] - drift[t])
        viewer.log_path("world/path/part2", np.concatenate([box["x_m"], man["x_m"]]), np.concatenate([box["y_m"], man["y_m"]]),
                        np.concatenate([rb["map_nT"], rm["map_nT"]]) * viewer.VERTICAL_M_PER_NT + 60.0)
    return man["t_s"][-1] + man["dt_s"], out


LABELS = dict(true="true map", network="network (lines-trained)", network_gen2="network (generator-trained, random masks)",
              wiener="Gaussian estimator, beta 3.5", interp="linear interpolation between lines")
COLOURS = dict(true=(40, 40, 40), network=(0, 130, 200), network_gen2=(120, 120, 220), wiener=(220, 120, 0), interp=(0, 160, 80))
WINDOW_240 = (150, 200, 256)       # the same ground in the 178 m pixels of the grid continued to 240 m (factor 4)


def survey_leg(w, t0, imu, rng, log, h_m, window, tag, tau, coef_lag, sensor_rms, vehicle, n_particles=10000):
    """one survey leg (straight, a turn, straight) at height h_m over the window, the
    estimator chain being the calibration and the lag model, the navigator on the five maps.
    Returns the end time, the results dict and (maps, leg, nav) for the still image."""
    M = maps.four_maps(w, *window, sigma_nT=1.0, seed=0, h_m=h_m)
    ox, oy = M["origin_m"]; L = (M["n"] - 1) * M["dx_m"]
    leg = shift_time(aircraft.fly([(60.0, math.radians(2), math.radians(1), 0.0), (10.0, math.radians(20), 0.0, math.pi / 2 / 10), (60.0, math.radians(2), math.radians(1), 0.0)],
                                  ox + 0.2 * L, oy + 0.25 * L, h_m, 30.0, math.radians(15)), t0)
    r = sensor.read(w, leg, vehicle, noise_nT=0.1, rng=rng)
    e = estimator_inputs(w, leg, r, imu, rng)
    # the navigator does not know the map value at its position, so the estimator's regressor
    # uses B0 here; the induced-term error that costs is part of what the navigator absorbs
    y_est = r["reading_nT"] - B0
    matched = y_est - calibrate.apply(coef_lag, calibrate.regressor(e["u"], lag(e["udot"], tau, leg["dt_s"])))
    truth = r["map_nT"]
    # the likelihood width: what the chain leaves on the box in band, its slow part on the
    # leg (the drift, the IMU bias walk and the induced term read with B0 instead of B move
    # the residual by a nT or two over ten seconds), and the map's rms error
    res = matched - truth
    slow_rms = float(np.std(np.convolve(res - res.mean(), np.ones(100) / 100, "same")))
    nav, results = {}, {}
    after = slice(800, None)
    rms_after = lambda a: float(np.sqrt(np.mean(a[after] ** 2)))
    for k, mp in M["maps"].items():
        map_err = M["rms_error_nT"].get(k, 0.0)
        sig = math.hypot(sensor_rms, slow_rms, map_err, 1.0)
        grad = float(math.hypot(*[np.gradient(mp, axis=ax).std() / M["dx_m"] for ax in (0, 1)]))
        run = lambda z, box, walk=0.05: navigator.run(mp, M["dx_m"], (ox, oy), leg["x_m"], leg["y_m"], z, sig, np.random.default_rng(3), n_particles=n_particles,
                                                      init_box_m=box, motion_sigma_m=0.05, bias_sigma_nT=1000.0, bias_walk_nT=walk)
        # the search from a 1.5 km box, and the tracking from a 300 m one
        nav[k] = run(matched, 1500.0)
        small = run(matched, 300.0)
        results[k] = dict(label=LABELS[k], map_rms_nT=float(map_err), likelihood_sigma_nT=float(sig), map_gradient_rms_nT_per_m=grad,
                          error_after_turn_m=rms_after(nav[k]["error_m"]), bound_after_turn_m=rms_after(nav[k]["bound_m"]),
                          error_after_turn_small_box_m=rms_after(small["error_m"]), bound_after_turn_small_box_m=rms_after(small["bound_m"]))
        if k == "true":
            # the same filter on a reading with white noise of the residual's rms in place of
            # the chain's residual: what separates the filter from the chain's slow errors
            z_white = truth + res.mean() + np.random.default_rng(5).normal(0, np.std(res), len(res))
            results[k]["error_after_turn_white_noise_m"] = rms_after(run(z_white, 1500.0)["error_m"])
            # and with the offset held constant, the case the bound describes
            results[k]["error_after_turn_white_noise_constant_offset_m"] = rms_after(run(z_white, 1500.0, 0.0)["error_m"])
    # the floor panel: the Part G evaluation on the window's own 64-pixel tiles (raster order)
    model_g, c = maps.load_model(maps.RUN, "cpu")
    rows = evaluate(M["maps"]["true"].astype(np.float32), h_m / M["dx_m"], model_g, c, maps.PATCH, 1.0, "cpu", 10 ** 6, torch.Generator().manual_seed(0))
    per_row = M["n"] // maps.PATCH
    def tile_of(x, y):
        i, j = int((y - oy) / M["dx_m"]) // maps.PATCH, int((x - ox) / M["dx_m"]) // maps.PATCH
        k = i * per_row + j
        return rows[k] if 0 <= i < per_row and 0 <= j < per_row and k < len(rows) else None
    out = dict(h_m=float(h_m), dx_m=float(M["dx_m"]), window=dict(row0=window[0], col0=window[1], n=window[2]), window_km=float(M["n"] * M["dx_m"] / 1000),
               residual_rms_nT=float(demeaned(res)), residual_slow_rms_nT=slow_rms, drift_nT_per_min=float(vehicle["drift_nT_per_min"]), maps=results,
               tiles=[dict(floor_predicted=t["floor_predicted"], mse_network=t["mse_network"], mse_wiener_beta35=t["mse_wiener_beta35"]) for t in rows])
    if log:
        viewer.log_text("text/overlay", f"Survey leg at {h_m:.0f} m over the Gawler window ({M['dx_m']:.0f} m pixels), fixed-wing, chain: calibration and lag model. The particle filter runs on the true map and on "
                                        "reconstructions from 25% survey lines; position error in metres next to the Cramer-Rao bound. The tile under the aircraft: its Gaussian floor and the network's error (Part G).\n"
                                        + "\n".join(f"{results[k]['label']}: map rms {results[k]['map_rms_nT']:.1f} nT, error after the turn {results[k]['error_after_turn_m']:.0f} m, bound {results[k]['bound_after_turn_m']:.0f} m" for k in results))
        yoff = 0.0 if tag == "60" else -1.15 * L
        for i, k in enumerate(["true", "network", "wiener", "interp", "network_gen2"]):
            viewer.log_map(f"world/maps{tag}/{k}", M["maps"][k], M["dx_m"], (ox + i * 1.15 * L, oy + yoff) if (i or tag != "60") else (ox, oy), z_offset_m=0.0 if (k == "true" and tag == "60") else -600.0)
            viewer.series_style(f"plots/position{tag}/{k}", LABELS[k], COLOURS[k])
        viewer.series_style(f"plots/position{tag}/bound_true_map", "Cramer-Rao bound, true map", (200, 0, 0))
        log_flight_frames(leg, r)
        snaps = {k: dict(nav[k]["particles"]) for k in nav}
        for t in range(len(leg["t_s"])):
            viewer.set_time(leg["t_s"][t])
            viewer.log_scalar("plots/reading/raw", y_est[t]); viewer.log_scalar("plots/reading/corrected", matched[t]); viewer.log_scalar("plots/reading/map", truth[t])
            viewer.log_scalar("plots/corrector/lag", matched[t] - truth[t])
            for k in nav:
                viewer.log_scalar(f"plots/position{tag}/{k}", nav[k]["error_m"][t])
                if t in snaps[k]:
                    viewer.log_particles(f"world/particles/{k}", snaps[k][t], h_m + 20.0 + 10.0 * list(nav).index(k), COLOURS[k])
            viewer.log_scalar(f"plots/position{tag}/bound_true_map", nav["true"]["bound_m"][t])
            tile = tile_of(leg["x_m"][t], leg["y_m"][t])
            if tile is not None:
                viewer.log_scalar("plots/floor/gaussian_floor", tile["floor_predicted"]); viewer.log_scalar("plots/floor/network_mse", tile["mse_network"]); viewer.log_scalar("plots/floor/gaussian_mse", tile["mse_wiener_beta35"])
        viewer.log_path(f"world/path/leg{tag}", leg["x_m"], leg["y_m"], truth * viewer.VERTICAL_M_PER_NT + h_m)
    return leg["t_s"][-1] + leg["dt_s"], out, (M, leg, nav)


def part_navigation(w, t0, imu, model, rng, log, n_particles=10000):
    """parts 3 and 4: a calibration box, then the survey leg at 60 m over the 45 m-pixel
    window and again at 240 m over the same ground in 178 m pixels (Part G's scale where the
    network reconstruction wins). The corrector is not in the navigation chain: until its
    held-out table puts it within a factor of two of the physics baselines the navigator
    runs on the calibration and the lag model, and the corrector stays part 2's trace."""
    v = platform.preset("survey_fixed_wing", seed=1)
    ox, oy = WINDOW[1] * w.dx_m, WINDOW[0] * w.dx_m
    box = shift_time(aircraft.fly(aircraft.calibration_box(leg_s=20.0, turn_s=5.0), ox - 4000.0, oy + 0.5 * (WINDOW[2] - 1) * w.dx_m, 60.0, 30.0, 0.0), t0)
    rb = sensor.read(w, box, v, noise_nT=0.1, rng=rng)
    eb = estimator_inputs(w, box, rb, imu, rng)
    tau, coef_lag = fit_lag_model(eb["u"], eb["udot"], eb["y"], rb["ambient_nT"], box["dt_s"])
    # the width of the navigator's likelihood comes from the calibration box: what the chain
    # leaves there in band, as an estimator would measure it before the leg
    res_box = eb["y"] - calibrate.apply(coef_lag, calibrate.regressor(eb["u"], lag(eb["udot"], tau, box["dt_s"]), rb["ambient_nT"]))
    sensor_rms = float(rms(calibrate.highpass(res_box, box["dt_s"], HP_TAU_S)))
    if log:
        log_flight_frames(box, rb)
    t = box["t_s"][-1] + box["dt_s"]
    legs, extras = {}, {}
    for tag, h_m, window in (("60", 60.0, WINDOW), ("240", 240.0, WINDOW_240)):
        t, legs[tag], extras[tag] = survey_leg(w, t, imu, rng, log, h_m, window, tag, tau, coef_lag, sensor_rms, v, n_particles)
    return t, dict(likelihood_sensor_rms_nT=sensor_rms, legs=legs), extras


def demeaned(x):
    return rms(x - x.mean())


def start_point(w, rng):
    ex, ey = w.extent_m
    return float(rng.uniform(0.2 * ex, 0.3 * ex)), float(rng.uniform(0.2 * ey, 0.3 * ey))


def shift_time(flight, t0):
    f = dict(flight); f["t_s"] = flight["t_s"] + t0
    return f


def run(w, scenario="full", imu=True, model_path="runs/corrector.pt", record=None, spawn=False, seed=0, log=True):
    rng = np.random.default_rng(seed)
    model = corrector.load(model_path)
    if log:
        rr = viewer.start(record=record, spawn=spawn)
        rr.send_blueprint(viewer.blueprint())
        viewer.log_map("world/map", w.grid_at(w.h0_m), w.dx_m, (0.0, 0.0), step=4 if w.grid_nT.shape[0] > 1000 else 1)
        for name, col in [("raw", (120, 120, 120)), ("platform_first_order", (200, 100, 0)), ("corrected", (0, 130, 200)), ("map", (0, 0, 0))]:
            viewer.series_style(f"plots/reading/{name}", name, col)
        for name, col in [("linear", (120, 120, 120)), ("gauss_newton", (200, 100, 0)), ("lag", (0, 160, 80)), ("gauss_newton_lag", (150, 0, 150)), ("corrector", (0, 130, 200)), ("lag_corrector", (200, 0, 60))]:
            viewer.series_style(f"plots/corrector/{name}", name, col)
    out, t = dict(scenario=scenario, imu=imu, map=w.name, model=model_path), 0.0
    if scenario in ("full", "calibration"):
        t, out["part1_calibration"] = part_calibration(w, t, imu, rng, log)
    if scenario in ("full", "interference"):
        t, out["part2_interference"] = part_interference(w, t, imu, model, rng, log)
    nav_extra = None
    if scenario in ("full", "navigation"):
        t, out["part3_navigation"], nav_extra = part_navigation(w, t, imu, model, rng, log)
    out["duration_s"] = float(t)
    return out, nav_extra

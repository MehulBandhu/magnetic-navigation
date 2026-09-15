"""Physical components of the flight simulator, each checked against a number the repository
already contains. CPU, a few seconds.
    python -m pytest -q tests/test_magsim.py
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem1.checks import regressors, numerical_rank, B0
from problem1.edge_cases import rms
from magsim import world, aircraft, platform, sensor


def _world():
    return world.synthetic(n=128, h0_m=60.0, dx_m=45.0, seed=0)


def _box_flight(h_m=60.0):
    # a 45 m grid of 128 pixels is 5.7 km across; a 30 m/s box of 60 s legs fits inside
    return aircraft.fly(aircraft.calibration_box(), x0_m=1500.0, y0_m=1500.0, h_m=h_m, speed_mps=30.0)


# ---------------------------------------------------------------- stage 1: world, aircraft, platform, sensor
def test_box_manoeuvre_rank_is_17_raw_16_band_passed():
    # the rank readout of problem1/checks.py, reproduced from the simulated attitude and rates
    f = _box_flight()
    u, udot = aircraft.body_direction_cosines(f, world.INCL, world.DECL)
    A = regressors(u, udot)
    assert numerical_rank(A)[0] == 17
    assert numerical_rank(A - A.mean(0), ref=A)[0] == 16


def test_straight_leg_rank_is_1_raw_0_band_passed():
    f = aircraft.fly(aircraft.straight_leg(60.0), x0_m=1000.0, y0_m=1000.0, heading0_rad=math.radians(37))
    u, udot = aircraft.body_direction_cosines(f, world.INCL, world.DECL)
    A = regressors(u, udot)
    assert numerical_rank(A)[0] == 1
    assert numerical_rank(A - A.mean(0), ref=A)[0] == 0


def test_reading_is_map_value_without_platform_field():
    w = _world()
    f = _box_flight()
    r = sensor.read(w, f, vehicle=None, noise_nT=0.0)
    expected = w.field(f["x_m"], f["y_m"], f["h_m"])
    assert np.max(np.abs(r["reading_nT"] - B0 - expected)) < 1e-6
    assert expected.std() > 1.0                      # the flight actually crosses structure


def test_second_order_term_matches_bp_perp_squared_over_2b():
    # the exact magnitude minus the first-order model is |Bp_perp|^2 / 2B (Problem 1A); the
    # next term is smaller by eta = |Bp| / B, under 1% at the fixed-wing scale
    w = _world()
    f = _box_flight()
    v = platform.preset("survey_fixed_wing", seed=0)
    r = sensor.read(w, f, vehicle=v, noise_nT=0.0)
    Bp = r["platform"]["Bp_nT"]
    dropped = r["exact_nT"] - r["first_order_nT"]
    predicted = platform.second_order_nT(r["u"], Bp, r["ambient_nT"])
    assert rms(dropped) > 0.3                         # the term is there to be seen
    assert abs(rms(dropped) / rms(predicted) - 1) < 0.01
    assert np.max(np.abs(dropped - predicted)) < 0.01 * np.max(np.abs(predicted))


def test_random_vehicle_telemetry_shapes():
    w = _world()
    f = aircraft.fly([(30.0, math.radians(10), math.radians(5), 0.0)], x0_m=1000.0, y0_m=1000.0)
    T = len(f["t_s"])
    assert T == 300
    for seed in range(3):
        v = platform.custom(seed)
        r = sensor.read(w, f, vehicle=v, noise_nT=0.1, rng=np.random.default_rng(seed))
        p = r["platform"]; tel = p["telemetry"]
        assert p["Bp_nT"].shape == (T, 3) and p["drift_nT"].shape == (T,)
        assert tel["motor_current_A"].shape == (T, platform.MAX_MOTORS)
        assert tel["motor_rpm"].shape == (T, platform.MAX_MOTORS)
        for k in ("battery_voltage_V", "battery_current_A", "throttle", "servo"):
            assert tel[k].shape == (T,), k
        assert 1 <= v["n_motors"] <= platform.MAX_MOTORS
        assert np.all(tel["motor_current_A"][:, v["n_motors"]:] == 0)
        assert np.all(tel["motor_current_A"][:, :v["n_motors"]] > 0)
        assert set(np.unique(tel["servo"])) <= {0, 1}
        assert r["reading_nT"].shape == (T,) and np.isfinite(r["reading_nT"]).all()


def test_platform_classes_have_the_edge_case_magnitudes():
    # mean |Bp| over the box for the three presets, against the class scales of edge_cases.py
    # (about 460, 930, 2700 nT in its table; the draw here differs, the order of magnitude must not)
    w = _world()
    f = _box_flight()
    mags = {}
    for name in ("survey_fixed_wing", "helicopter", "multirotor"):
        r = sensor.read(w, f, vehicle=platform.preset(name), noise_nT=0.0)
        mags[name] = np.linalg.norm(r["platform"]["Bp_nT"], axis=1).mean()
    assert 250 < mags["survey_fixed_wing"] < 700
    assert 500 < mags["helicopter"] < 1400
    assert 1800 < mags["multirotor"] < 4000


def test_world_samples_grid_and_gradient():
    w = _world()
    dx = w.dx_m
    # pixel centres return the grid at h0
    rows, cols = np.array([3, 40, 100]), np.array([7, 64, 120])
    assert np.allclose(w.field(cols * dx, rows * dx, w.h0_m), w.grid_nT[rows, cols])
    # the gradient agrees with a central difference of the sampled field
    x, y = 50.3 * dx, 61.7 * dx
    g = w.gradient(x, y, w.h0_m)
    eps = 0.5 * dx
    fd = np.array([(w.field(x + eps, y, w.h0_m) - w.field(x - eps, y, w.h0_m)) / (2 * eps),
                   (w.field(x, y + eps, w.h0_m) - w.field(x, y - eps, w.h0_m)) / (2 * eps)])
    assert np.allclose(g, fd, rtol=0.05, atol=1e-3 * np.abs(fd).max()), (g, fd)
    # continuation smooths: less variance 200 m higher, the gradient at least halved, and
    # no change below h0
    assert w.grid_at(w.h0_m + 200).std() < w.grid_at(w.h0_m).std()
    grad_rms = lambda h: math.hypot(*[g.std() for g in w.gradient_at(h)])
    assert grad_rms(w.h0_m + 200) < 0.5 * grad_rms(w.h0_m)
    assert np.array_equal(w.grid_at(10.0), w.grid_at(w.h0_m))


@pytest.mark.skipif(not os.path.exists("data/gawler_tmi.nc"), reason="Gawler grid is local only")
def test_gawler_world_matches_the_evaluated_grid():
    from magscale.grid_eval import load_grid
    ref = json.load(open("results/grid_gawler.json"))
    w = world.gawler()
    assert abs(w.dx_m - ref["dx"]) < 1.0 and w.h0_m == ref["h0"]
    g, _ = load_grid("data/gawler_tmi.nc")
    assert np.allclose(w.grid_at(w.h0_m), g)
    # the flown-altitude field is what the survey reads; sampling off the square works too
    assert np.isfinite(w.field(w.extent_m[0] * 0.9, w.extent_m[1] * 0.9, 60.0))


# ---------------------------------------------------------------- stage 2: calibration, corrector
@pytest.mark.skipif(not os.path.exists("data/gawler_tmi.nc"), reason="Gawler grid is local only")
def test_gawler_box_rank_under_its_own_ambient_direction():
    # inclination -65 and declination +7 degrees, not the checks.py direction: still 17 and 16
    w = world.gawler()
    f = aircraft.fly(aircraft.calibration_box(), x0_m=30000.0, y0_m=40000.0, h_m=60.0)
    u, udot = aircraft.body_direction_cosines(f, w.incl_rad, w.decl_rad)
    A = regressors(u, udot)
    assert numerical_rank(A)[0] == 17 and numerical_rank(A - A.mean(0), ref=A)[0] == 16


def test_imu_error_has_the_stated_size():
    f = _box_flight()
    est = aircraft.imu(f, np.random.default_rng(0), noise_deg=0.1, bias_deg=0.2)
    d = est["roll_rad"] - f["roll_rad"]
    assert abs(np.std(d - d.mean()) / math.radians(0.1) - 1) < 0.15      # white part
    assert 0 < abs(d.mean()) < math.radians(1.0)                           # a bias of a few tenths of a degree
    assert np.array_equal(f["roll_rad"], _box_flight()["roll_rad"])        # the true flight is untouched


def test_live_calibration_reaches_the_rank_and_a_small_residual():
    from magsim import calibrate
    w = _world()
    f = _box_flight()
    v = platform.preset("survey_fixed_wing", seed=0)
    v["drift_nT_per_min"] = 0.0
    r = sensor.read(w, f, vehicle=v, noise_nT=0.1, rng=np.random.default_rng(0))
    y = r["reading_nT"] - r["ambient_nT"]
    # the regressor with the constant B0 of the numerical checks: with the map value folded
    # into B the isotropic induced column is no longer constant and the band-passed rank is 17
    raw = calibrate.live(r["u"], r["udot"], y, f["dt_s"])
    hp = calibrate.live(r["u"], r["udot"], y, f["dt_s"], highpass_tau_s=10.0)
    assert raw["rank"][-1] == 17 and hp["rank"][-1] == 16
    assert raw["rank"][9] < 17                                             # the rank climbs during the box
    assert hp["rms_nT"][-1] < 1.5, hp["rms_nT"][-1]                        # what the linear model leaves on a fixed-wing


def test_corrector_is_small_and_its_windows_are_aligned():
    from magsim import corrector
    m = corrector.Corrector()
    assert m.n_params() < 200_000
    T = 400
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, corrector.N_BASE)).astype(np.float32)
    r = (5.0 * rng.normal(size=T) + 20.0).astype(np.float32)
    t = np.array([0, 5, 199, 399])
    W, level, scale = corrector.windows(X, r, True, t, m.window, m.gap)
    assert W.shape == (4, corrector.N_FEATURES, m.window) and level.shape == (4,) and np.all(scale > 0)
    assert np.allclose(W[:, :corrector.N_BASE, -1], X[t])                    # last column is the current step
    assert np.all(W[:, corrector.N_BASE, :] == 1.0)                         # the flag channel
    assert np.all(W[:, -1, m.window - m.gap:] == 0)                          # residual blanked over the gap
    assert np.allclose(W[0, :corrector.N_BASE, 0], X[0])                     # edge padded at the front
    assert abs(level[-1] - 20.0) < 2.0 and abs(scale[-1] - 5.0) < 1.5
    pred = corrector.predict(m, X, r, False)
    assert pred.shape == (T,) and np.isfinite(pred).all()


# ---------------------------------------------------------------- stage 3: maps, navigator
@pytest.mark.skipif(not os.path.exists("data/gawler_tmi.nc"), reason="Gawler grid is local only")
def test_part_g_tiles_reproduce_the_committed_json():
    # the maps module runs magscale.grid_eval.evaluate on the same grid with the same seed;
    # the first tiles must be the committed ones
    from magsim import maps
    rows = maps.part_g_tiles(world.gawler(), max_tiles=8)
    ref = maps.committed_tiles()
    for k in ("mse_network", "mse_wiener_beta35", "mse_interp", "floor_predicted", "beta"):
        a = np.array([r[k] for r in rows]); b = np.array([r[k] for r in ref[:len(rows)]])
        assert np.allclose(a, b, rtol=1e-4), k


def test_navigator_on_the_true_map_reaches_the_bound_after_the_first_turn():
    from magsim import navigator
    w = _world()
    sigma = 1.0
    L = w.extent_m[0]
    segs = [(50.0, 0.0, 0.0, 0.0), (10.0, math.radians(20), 0.0, math.pi / 2 / 10), (50.0, 0.0, 0.0, 0.0)]
    f = aircraft.fly(segs, 0.3 * L, 0.3 * L, w.h0_m, 30.0, math.radians(30))
    r = sensor.read(w, f, None, noise_nT=sigma, rng=np.random.default_rng(0))
    z = r["reading_nT"] - B0
    nav = navigator.run(w.grid_at(w.h0_m), w.dx_m, (0.0, 0.0), f["x_m"], f["y_m"], z, sigma, np.random.default_rng(1),
                        n_particles=3000, init_box_m=1000.0, motion_sigma_m=0.3)
    e, b = nav["error_m"], nav["bound_m"]
    after = slice(700, None)                                   # the second leg, 10 s into it
    assert b[0] > 100 and b[-1] < b[600] < b[0]                 # the bound falls as readings come in
    rms_err = math.sqrt(np.mean(e[after] ** 2)); rms_bound = math.sqrt(np.mean(b[after] ** 2))
    assert rms_err < 2 * rms_bound, (rms_err, rms_bound)
    assert nav["spread_m"][0] > 10 * nav["spread_m"][-1]        # the cloud contracts
    # with an unknown offset of hundreds of nT on the reading and the offset state on, the
    # filter still tracks from a 300 m box and recovers the offset (from a kilometre box the
    # search is ambiguous on a smooth map when the level is unknown: many positions fit with
    # a suitable offset, which is what the demo's 240 m leg shows)
    nav2 = navigator.run(w.grid_at(w.h0_m), w.dx_m, (0.0, 0.0), f["x_m"], f["y_m"], z + 300.0, sigma, np.random.default_rng(1),
                         n_particles=3000, init_box_m=300.0, motion_sigma_m=0.3, bias_sigma_nT=1000.0)
    e2, b2 = nav2["error_m"], nav2["bound_m"]
    assert math.sqrt(np.mean(e2[after] ** 2)) < 2 * math.sqrt(np.mean(b2[after] ** 2)), (e2[after], b2[after])
    assert abs(nav2["bias_est_nT"][-1] - 300.0) < 3 * sigma

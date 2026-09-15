"""Train the corrector on simulated vehicles and evaluate it against the physics baselines.

Protocol. Every vehicle flies a calibration box first; the linear Tolles-Lawson model and the
lag model (tau scanned) are fitted there (magsim.calibrate, high-passed so the drift stays
out) and applied to the vehicle's other flights: random manoeuvres and a survey pattern. The
residual each leaves is a target. The corrector is trained on the training vehicles' flights
and evaluated on held-out vehicles it has never seen, on their non-calibration flights. It is
shown either residual (a flag channel says which), so one model serves both chains:
calibration then corrector, or calibration, lag model, then corrector, which is the demo's.

The baselines from problem1.edge_cases are fitted on the same calibration box and applied to
the same flights: Gauss-Newton on the exact model, the lag model, both together, and, as the
physics-with-telemetry reference, Gauss-Newton with the battery current as an extra
regressor. The table in results/corrector.json gives residual rms after each, per platform
class, for vehicles with and without the motor, battery and servo fields, after removing the
flight mean and the thermal drift (no method can see the drift; the navigator absorbs it).

The loss is scale-free per vehicle: the error is divided by the rms of that flight's linear
residual before a log-cosh, so a helicopter's 50 nT rows weigh the same as a fixed-wing's 5.
Per-step features are precomputed once for every flight; windows are gathered per batch,
which costs about a millisecond.

    python -m magsim.train_corrector --vehicles 300 --heldout 45 --steps 40000
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from problem1.edge_cases import gauss_newton, gn_predict, lag, rms

from . import aircraft, calibrate, platform, sensor, world
from .platform import rigid_body_field, first_order_nT
from .corrector import Corrector, features, windows, predict, save, N_FEATURES, N_BASE

CLASS_NAMES = list(platform.CLASSES)
HP_TAU_S = 10.0        # the calibration fits are high-passed at this time constant (drift out, manoeuvres at 0.1 Hz kept)
METHODS = ["linear", "gauss_newton", "lag", "gauss_newton_lag", "gauss_newton_current", "corrector", "lag_corrector", "drift", "imu_floor"]


def demeaned_rms(x):
    """rms after removing the flight's mean: a constant is a bias the navigator absorbs"""
    return rms(x - x.mean())


# ------------------------------------------------------------------ flights
def random_manoeuvre(rng, duration_s=100.0):
    """segments of 8 to 25 s with random roll and pitch excitation and heading rate"""
    segs, t = [], 0.0
    while t < duration_s:
        d = float(rng.uniform(8.0, 25.0))
        segs.append((d, math.radians(rng.uniform(0, 15)), math.radians(rng.uniform(0, 10)), math.radians(rng.uniform(-6, 6))))
        t += d
    return segs


def survey_pattern(rng, leg_s=40.0):
    """two straight legs with mild excitation and a 90 degree turn between them"""
    a = math.radians(rng.uniform(1, 4))
    return [(leg_s, a, 0.5 * a, 0.0), (10.0, math.radians(20), 0.0, math.pi / 2 / 10.0), (leg_s, a, 0.5 * a, 0.0)]


def start_inside(w, rng, margin_m=2500.0):
    ex, ey = w.extent_m
    return float(rng.uniform(margin_m, ex - margin_m)), float(rng.uniform(margin_m, ey - margin_m))


def fit_lag_model(u, udot, y, B, dt):
    """the lag model on a calibration flight: tau by the coarse scan of edge_cases then a fine
    one, coefficients by the high-passed fit. Returns (tau, coef)."""
    def one(tt):
        A = calibrate.regressor(u, lag(udot, tt, dt), B)
        c = calibrate.fit(A, y, highpass_tau_s=HP_TAU_S, dt_s=dt)
        return rms(calibrate.highpass(y - calibrate.apply(c, A), dt, HP_TAU_S)), tt, c
    _, tau, _ = min(one(tt) for tt in np.logspace(-2.3, 1, 25))
    _, tau, coef = min(one(tt) for tt in tau * np.logspace(-0.15, 0.15, 13))
    return tau, coef


def simulate_vehicle(vehicle, w, rng, imu, n_random=2, noise_nT=0.1):
    """the calibration box and the evaluation flights of one vehicle. Returns dict with the
    fits from the box and a list of flight records, the box first, each holding what the
    corrector and the baselines need."""
    records, fits = [], None
    plans = [("box", aircraft.calibration_box())] + [(f"random{i}", random_manoeuvre(rng)) for i in range(n_random)] + [("survey", survey_pattern(rng))]
    for name, segs in plans:
        x0, y0 = start_inside(w, rng)
        flight = aircraft.fly(segs, x0, y0, h_m=60.0, speed_mps=30.0, heading0_rad=float(rng.uniform(0, 2 * math.pi)))
        r = sensor.read(w, flight, vehicle, noise_nT=noise_nT, rng=rng)
        est = aircraft.imu(flight, rng) if imu else flight
        u, udot = aircraft.body_direction_cosines(est, w.incl_rad, w.decl_rad)
        dt, B = flight["dt_s"], r["ambient_nT"]
        A = calibrate.regressor(u, udot, B)
        y = r["reading_nT"] - B
        if fits is None:                                   # the calibration, on the box
            tau, coef_lag = fit_lag_model(u, udot, y, B, dt)
            fits = dict(coef=calibrate.fit(A, y, highpass_tau_s=HP_TAU_S, dt_s=dt), tau=tau, coef_lag=coef_lag)
        residual_lin = y - calibrate.apply(fits["coef"], A)
        residual_lag = y - calibrate.apply(fits["coef_lag"], calibrate.regressor(u, lag(udot, fits["tau"], dt), B))
        # the targets exclude the thermal drift: no method can see it, the navigator absorbs it
        # as bias, and it would otherwise dominate every column of the table alike. The
        # residual the corrector is shown still contains it.
        drift = r["platform"]["drift_nT"]
        # what the IMU error alone costs: the true rigid-body field read through the estimated
        # attitude instead of the true one (zero without --imu)
        Bp_true = rigid_body_field(vehicle, r["u"], r["udot"], dt, B)
        imu_floor = demeaned_rms(first_order_nT(u, rigid_body_field(vehicle, u, udot, dt, B)) - first_order_nT(r["u"], Bp_true))
        records.append(dict(name=name, X=features(est, A, r["platform"]["telemetry"], u, udot),
                            residual_lin=residual_lin.astype(np.float32), residual_lag=residual_lag.astype(np.float32),
                            target_lin=(residual_lin - drift).astype(np.float32), target_lag=(residual_lag - drift).astype(np.float32),
                            rms_lin=max(demeaned_rms(residual_lin), 0.5), rms_lag=max(demeaned_rms(residual_lag), 0.5),
                            drift_nT=drift, imu_floor=imu_floor, u=u, udot=udot, y_nT=y, reading_nT=r["reading_nT"], ambient_nT=B, dt_s=dt,
                            battery_current_A=r["platform"]["telemetry"]["battery_current_A"]))
    return dict(vehicle=vehicle, fits=fits, records=records)


def make_vehicles(n, seed0, extras_fraction=0.7):
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        out.append(platform.custom(seed0 + i, scale=1.0, extras=bool(rng.uniform() < extras_fraction)))
    return out


def generate(vehicles, worlds, seed, imu):
    rng = np.random.default_rng(seed)
    return [simulate_vehicle(v, worlds[i % len(worlds)], rng, imu) for i, v in enumerate(vehicles)]


# ------------------------------------------------------------------ baselines
def baselines(entry):
    """fit each physics model on the calibration box, apply it to the other flights; returns
    {flight name: {method: residual rms nT}}"""
    box, fits = entry["records"][0], entry["fits"]
    dt, tau = box["dt_s"], fits["tau"]
    hp = lambda x: calibrate.highpass(x, dt, HP_TAU_S)
    gn = lambda udot_box, extra=None: gauss_newton(box["u"], udot_box, box["reading_nT"], iters=12, extra=extra, B=box["ambient_nT"], return_theta=True, filt=hp)[1]
    th_gn = gn(box["udot"])
    th_gnlag = gn(lag(box["udot"], tau, dt))
    cur_mean = box["battery_current_A"].mean()
    th_cur = gn(lag(box["udot"], tau, dt), box["battery_current_A"] - cur_mean)
    out = {}
    for rec in entry["records"][1:]:
        u, udot, B, d = rec["u"], rec["udot"], rec["ambient_nT"], rec["drift_nT"]
        cur = rec["battery_current_A"] - cur_mean
        out[rec["name"]] = {
            "linear": demeaned_rms(rec["target_lin"]),
            "gauss_newton": demeaned_rms(rec["reading_nT"] - gn_predict(th_gn, u, udot, B=B)[0] - d),
            "lag": demeaned_rms(rec["target_lag"]),
            "gauss_newton_lag": demeaned_rms(rec["reading_nT"] - gn_predict(th_gnlag, u, lag(udot, tau, dt), B=B)[0] - d),
            "gauss_newton_current": demeaned_rms(rec["reading_nT"] - gn_predict(th_cur, u, lag(udot, tau, dt), extra=cur, B=B)[0] - d),
            "drift": demeaned_rms(d), "imu_floor": rec["imu_floor"],
        }
    return out


# ------------------------------------------------------------------ training
def samples_of(data):
    """one sample per flight and residual variant: (X, residual shown, target, after_lag flag, flight rms)"""
    out = []
    for e in data:
        for rec in e["records"]:
            out.append((rec["X"], rec["residual_lin"], rec["target_lin"], False, rec["rms_lin"]))
            out.append((rec["X"], rec["residual_lag"], rec["target_lag"], True, rec["rms_lag"]))
    return out


def sample_batch(samples, lens, batch, window, gap, rng):
    """windows drawn with probability proportional to flight length; returns the inputs, the
    normalised targets and the loss weights scale_window / rms_flight"""
    fi = rng.choice(len(samples), batch, p=lens / lens.sum())
    xb, yb, wb = [], [], []
    for i in np.unique(fi):
        X, r, target, flag, frms = samples[i]
        t = rng.integers(gap, lens[i], int((fi == i).sum()))
        w, level, scale = windows(X, r, flag, t, window, gap)
        xb.append(w); yb.append((target[t] - level) / scale); wb.append(scale / frms)
    return torch.from_numpy(np.concatenate(xb)), torch.from_numpy(np.concatenate(yb).astype(np.float32)), torch.from_numpy(np.concatenate(wb).astype(np.float32))


def logcosh(x):
    return (x + torch.nn.functional.softplus(-2 * x) - math.log(2.0)).mean()


def train(model, samples, val_samples, steps, batch, lr, seed, log_every=100, eval_every=1000, loss_scale="flight", device="cpu",
          fresh=None):
    """fresh: optional (every_n_steps, function) that returns a new list of samples from newly
    drawn vehicles; with 300 fixed vehicles the held-out loss rises after about a thousand
    steps while the training loss falls (the model learns the vehicles, not how to identify
    one), so the long runs draw fresh vehicles as they go"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lens = np.array([len(s[2]) for s in samples])
    vlens = np.array([len(s[2]) for s in val_samples])
    vx, vy, vw = [t.to(device) for t in sample_batch(val_samples, vlens, 2048, model.window, model.gap, np.random.default_rng(seed + 7))]
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    log, ema, t0 = [], None, time.time()
    for step in range(1, steps + 1):
        if fresh is not None and step > 1 and step % fresh[0] == 1:
            t1 = time.time()
            samples = fresh[1](step)
            lens = np.array([len(s[2]) for s in samples])
            print(f"step {step:6d}  fresh vehicles drawn in {time.time() - t1:.0f} s", flush=True)
        model.train()
        xb, yb, wb = [t.to(device) for t in sample_batch(samples, lens, batch, model.window, model.gap, rng)]
        # loss_scale "flight": error over the flight's linear residual rms (scale-free per
        # vehicle); "window": over the window's own residual scale only, the earlier behaviour
        loss = logcosh((model(xb) - yb) * (wb if loss_scale == "flight" else 1.0))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % eval_every == 0 or step == steps or step == 1:
            model.eval()
            with torch.no_grad():
                vl = float(logcosh((torch.cat([model(vx[i:i + 512]) for i in range(0, len(vx), 512)]) - vy) * vw))
            log.append(dict(step=step, train=ema, heldout=vl, time_s=time.time() - t0))
            print(f"step {step:6d}  train {ema:.4f}  held-out {vl:.4f}  (log-cosh, error over the flight's linear residual rms)  {time.time() - t0:.0f} s", flush=True)
        elif step % log_every == 0:
            log.append(dict(step=step, train=ema, time_s=time.time() - t0))
    return log


def evaluate(model, heldout):
    """per held-out flight: baselines, the corrector after the linear model, and the corrector
    after the lag model; aggregated per class and extras group as the root mean square over
    flights of each flight's rms"""
    rows = []
    for e in heldout:
        b = baselines(e)
        for rec in e["records"][1:]:
            pred_lin = predict(model, rec["X"], rec["residual_lin"], False)
            pred_lag = predict(model, rec["X"], rec["residual_lag"], True)
            rows.append(dict(vehicle=e["vehicle"]["name"], platform_class=e["vehicle"]["platform_class"], extras=e["vehicle"]["extras"], flight=rec["name"],
                             corrector=demeaned_rms(rec["target_lin"] - pred_lin), lag_corrector=demeaned_rms(rec["target_lag"] - pred_lag), **b[rec["name"]]))
    table = {}
    for cls in CLASS_NAMES:
        for extras in (False, True):
            sel = [r for r in rows if r["platform_class"] == cls and r["extras"] == extras]
            if sel:
                table[f"{cls}:{'with' if extras else 'without'}_extras"] = dict(
                    flights=len(sel), vehicles=len({r["vehicle"] for r in sel}),
                    **{m: float(np.sqrt(np.mean([r[m] ** 2 for r in sel]))) for m in METHODS},
                    **{m + "_median": float(np.median([r[m] for r in sel])) for m in METHODS})
    return rows, table


def print_table(table):
    print(f"\n{'class':>18} {'extras':>8} {'n':>3} {'linear':>7} {'GN':>7} {'lag':>7} {'GN+lag':>7} {'GN+cur':>7} {'corr':>7} {'lag+corr':>8} {'drift':>6} {'imu':>5}   residual rms nT, held-out vehicles, flight mean and drift excluded")
    for k, r in table.items():
        cls, ex = k.split(":")
        print(f"{cls:>18} {ex[:-7]:>8} {r['flights']:3d} {r['linear']:7.2f} {r['gauss_newton']:7.2f} {r['lag']:7.2f} {r['gauss_newton_lag']:7.2f} {r['gauss_newton_current']:7.2f} "
              f"{r['corrector']:7.2f} {r['lag_corrector']:8.2f} {r['drift']:6.2f} {r['imu_floor']:5.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vehicles", type=int, default=300)
    p.add_argument("--heldout", type=int, default=45)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--gap", type=int, default=20)
    p.add_argument("--dilations", default="1,2,4,8,16")
    p.add_argument("--strides", default="2,2")
    p.add_argument("--ridge", type=int, default=0, help="dimension of the in-context least-squares basis (0: plain head)")
    p.add_argument("--imu", action="store_true", help="give the estimator chain IMU attitude (aircraft.imu) instead of the truth")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--threads", type=int, default=0, help="torch threads (0: all but one core)")
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--loss_scale", default="flight", choices=["flight", "window"])
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda when available); the evaluation runs on the cpu")
    p.add_argument("--fresh_every", type=int, default=0, help="draw a new set of --vehicles training vehicles every this many steps (0: fixed set)")
    p.add_argument("--out", default="runs/corrector.pt")
    p.add_argument("--results", default="results/corrector.json")
    a = p.parse_args()
    if os.path.abspath(a.results) == os.path.abspath(a.out.replace(".pt", ".json")):
        raise SystemExit("--results must not be the checkpoint's own json (--out with .json), which holds the model config")
    torch.set_num_threads(a.threads or max(1, os.cpu_count() - 1))

    t0 = time.time()
    worlds = [world.synthetic(n=256, h0_m=60.0, dx_m=45.0, seed=s) for s in range(a.worlds)]
    train_v = make_vehicles(a.vehicles, 1000 + a.seed * 100000)
    held_v = make_vehicles(a.heldout, 900000 + a.seed * 100000)
    data = generate(train_v, worlds, a.seed, a.imu)
    heldout = generate(held_v, worlds, a.seed + 1, a.imu)
    samples, val_samples = samples_of(data), samples_of(heldout)
    n_steps = sum(len(e["records"][0]["X"]) + sum(len(r["X"]) for r in e["records"][1:]) for e in data)
    print(f"data: {len(train_v)} training vehicles, {len(held_v)} held out, {n_steps} training steps of {N_BASE} features plus the two residual channels, {time.time() - t0:.0f} s")

    allX = np.concatenate([e["records"][i]["X"] for e in data for i in range(len(e["records"]))])
    mean = np.concatenate([allX.mean(0), [0.0, 0.0]])
    std = np.concatenate([allX.std(0) + 1e-6, [1.0, 1.0]])
    del allX
    model = Corrector(N_FEATURES, a.width, 5, tuple(int(d) for d in a.dilations.split(",")), a.window, a.gap, mean, std,
                      strides=tuple(int(x) for x in a.strides.split(",")) if a.strides else (), ridge_dim=a.ridge)
    print(f"model: {model.n_params()} parameters, window {a.window} steps, gap {a.gap} steps, {torch.get_num_threads()} threads")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"training on {device}")
    def fresh_samples(step):
        vs = make_vehicles(a.vehicles, 1000 + a.seed * 100000 + step * 1000)
        return samples_of(generate(vs, worlds, a.seed + step, a.imu))
    log = train(model, samples, val_samples, a.steps, a.batch, a.lr, a.seed, eval_every=a.eval_every, loss_scale=a.loss_scale, device=device,
                fresh=(a.fresh_every, fresh_samples) if a.fresh_every else None)
    model.cpu()
    rows, table = evaluate(model, heldout)
    print_table(table)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); os.makedirs(os.path.dirname(a.results), exist_ok=True)
    save(model, a.out, dict(args=vars(a), n_params=model.n_params(), n_training_steps_data=int(n_steps), log=log, table=table, wall_time_s=time.time() - t0))
    json.dump(dict(args=vars(a), n_params=model.n_params(), table=table, rows=rows, methods=METHODS, log=log, wall_time_s=time.time() - t0), open(a.results, "w"), indent=1)
    print(f"saved {a.out} and {a.results}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

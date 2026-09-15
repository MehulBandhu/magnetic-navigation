"""Particle filter over horizontal position on a given map, with the Cramer-Rao bound of
Problem 1C alongside.

Predict: each particle moves by the aircraft's dead-reckoned displacement plus noise. Weight:
Gaussian likelihood of the corrected scalar reading against the map at the particle. The
reading carries an unknown offset (the level the band-passed calibration never fits, the
thermal drift, what the corrector leaves as bias), so each particle also carries a scalar
Kalman estimate of that offset (mean and variance) and the likelihood is marginalised over
it, the Rao-Blackwellised form: a single reading then constrains nothing, the variation of
the field along the track constrains position, and the level settles the offset. Resample
systematically when the effective sample size drops, with a small roughening.

Bound: the Fisher information of Problem 1C, J = sum g g^T / sigma^2 over the readings so far
with g the map gradient at the true position (magscale.magnav.crb uses the same quantities),
with the offset as a third unknown and the initial box and offset prior as information.
sqrt(trace of the position block of J^-1) in metres is what a known-shape trajectory with no
motion noise could reach; the filter has motion noise and should approach the bound, not beat
it. The gradient comes from the map used for matching, so with a reconstructed map the bound
is the heuristic one of magnav.crb.
"""
import math

import numpy as np

from magscale.magnav import bilinear


def run(map_nT, dx_m, origin_m, x_true_m, y_true_m, reading_nT, sigma_nT, rng, n_particles=3000,
        init_box_m=1500.0, motion_sigma_m=0.3, bias_sigma_nT=0.0, bias_walk_nT=0.0, snapshot_every=10, reading_fn=None, temper=1.0, anneal_s=15.0, grid_init=True):
    """navigate along a flight. x_true, y_true (T,) in metres are the true track (the dead
    reckoning is their increments plus noise); reading_nT (T,) is the corrected anomaly
    reading, map_nT the map used for matching (n, n) with pixel size dx_m and its (0, 0)
    pixel at origin_m; bias_sigma_nT is the prior width of the reading's offset (0: no offset
    state) and bias_walk_nT its random walk per step. reading_fn, if given, replaces
    reading_nT: called as reading_fn(t, x_est, y_est, spread) with the filter's estimate from
    the previous step (the box centre at t = 0), so an estimator that needs the map at the
    estimated position, such as the corrector, can be closed in the loop. Returns dict of
    (T,) arrays: x_est_m, y_est_m, error_m, spread_m (sqrt of the trace of the particle
    covariance), bound_m, bias_est_nT, reading_nT (what was matched), and 'particles', a list
    of (t index, (k, 2) positions) snapshots every snapshot_every steps. temper scales the
    widening of the likelihood by the field change over the particle spacing, which is
    annealed from its initial value with the time constant anneal_s (at 10 Hz)."""
    T = len(x_true_m)
    n = map_nT.shape[0]
    ox, oy = origin_m
    gx = np.gradient(map_nT, axis=1) / dx_m
    gy = np.gradient(map_nT, axis=0) / dx_m

    def sample(m, x, y):
        return bilinear(m, (y - oy) / dx_m, (x - ox) / dx_m)

    # the initial cloud is a jittered grid over the box, so no part of it is left uncovered
    # by chance (a uniform draw leaves gaps of several spacings that a narrow likelihood
    # never recovers from)
    if grid_init:
        side = int(math.ceil(math.sqrt(n_particles)))
        gx0, gy0 = np.meshgrid(np.linspace(-init_box_m, init_box_m, side), np.linspace(-init_box_m, init_box_m, side))
        cell = 2 * init_box_m / side
        px = x_true_m[0] + gx0.ravel()[:n_particles] + rng.uniform(-0.5, 0.5, n_particles) * cell
        py = y_true_m[0] + gy0.ravel()[:n_particles] + rng.uniform(-0.5, 0.5, n_particles) * cell
    else:
        px = x_true_m[0] + rng.uniform(-init_box_m, init_box_m, n_particles)
        py = y_true_m[0] + rng.uniform(-init_box_m, init_box_m, n_particles)
    mb = np.zeros(n_particles)                                  # offset mean per particle, nT
    Pb = np.full(n_particles, float(bias_sigma_nT) ** 2)        # and its variance
    logw = np.zeros(n_particles)
    spread = init_box_m * math.sqrt(2.0 / 3.0)     # sqrt of the trace of the box's covariance
    d0 = 2.0 * init_box_m / math.sqrt(n_particles)  # the initial particle spacing
    anneal_steps = (anneal_s or 1.0) * 10.0
    # the information of the initial box (as a Gaussian of the same variance) and of the
    # offset prior; the offset is the third unknown of the bound
    J = np.diag([3.0 / init_box_m ** 2, 3.0 / init_box_m ** 2, 1.0 / bias_sigma_nT ** 2 if bias_sigma_nT > 0 else 1e12])
    out = dict(x_est_m=np.zeros(T), y_est_m=np.zeros(T), error_m=np.zeros(T), spread_m=np.zeros(T), bound_m=np.zeros(T), bias_est_nT=np.zeros(T),
               reading_nT=np.zeros(T), particles=[])
    xe, ye = float(x_true_m[0]), float(y_true_m[0])
    for t in range(T):
        z = reading_fn(t, xe, ye, spread) if reading_fn is not None else reading_nT[t]
        out["reading_nT"][t] = z
        if t > 0:
            ddx, ddy = x_true_m[t] - x_true_m[t - 1], y_true_m[t] - y_true_m[t - 1]
            px += ddx + rng.normal(0, motion_sigma_m, n_particles)
            py += ddy + rng.normal(0, motion_sigma_m, n_particles)
            Pb += bias_walk_nT ** 2
        inside = (px > ox) & (px < ox + (n - 1) * dx_m) & (py > oy) & (py < oy + (n - 1) * dx_m)
        # the cloud samples the map at a finite spacing d; a particle within d of the truth
        # has to survive, so the likelihood is widened by the field change over d, with the
        # gradient at the particle (the window's rms gradient can be ten times the local
        # one). The spacing is annealed in time from the initial one rather than read from
        # the cloud: a cloud split over several modes keeps a large spread even when each
        # mode is tight, and a width tied to the spread then never sharpens and the estimate
        # sits between the modes. After a few anneal times the width is sigma alone.
        d = d0 * math.exp(-t / anneal_steps) if anneal_s else 2.0 * spread / math.sqrt(n_particles)
        g_local = np.hypot(sample(gx, px, py), sample(gy, px, py))
        sigma_eff2 = sigma_nT ** 2 + (temper * g_local * d) ** 2
        innov = z - sample(map_nT, px, py) - mb
        S = Pb + sigma_eff2
        # the widening enters the exponent only: it is a heuristic, not part of the model, and
        # a per-particle width in the normalisation would favour particles in flat regions of
        # the map by a factor exp(3) per step whatever the data says
        logw += -0.5 * innov ** 2 / S - 0.5 * np.log(Pb + sigma_nT ** 2)
        logw[~inside] = -np.inf
        K = Pb / S
        mb += K * innov
        Pb *= 1.0 - K
        w = np.exp(logw - logw.max()); w /= w.sum()
        xe, ye = float(w @ px), float(w @ py)
        cov = np.cov(np.stack([px, py]), aweights=w) if w.max() < 1 else np.zeros((2, 2))
        out["x_est_m"][t], out["y_est_m"][t], out["bias_est_nT"][t] = xe, ye, float(w @ mb)
        out["error_m"][t] = math.hypot(xe - x_true_m[t], ye - y_true_m[t])
        spread = math.sqrt(max(np.trace(cov), 0.0))
        out["spread_m"][t] = spread
        g = np.array([sample(gx, x_true_m[t], y_true_m[t]), sample(gy, x_true_m[t], y_true_m[t]), 1.0])
        J = J + np.outer(g, g) / sigma_nT ** 2
        out["bound_m"][t] = math.sqrt(np.trace(np.linalg.inv(J)[:2, :2]))
        if t % snapshot_every == 0:
            out["particles"].append((t, np.stack([px, py], 1).copy()))
        if 1.0 / np.sum(w ** 2) < 0.5 * n_particles:
            # systematic resampling, then roughening at a fraction of the current spread
            idx = np.searchsorted(np.cumsum(w), (rng.uniform() + np.arange(n_particles)) / n_particles)
            idx = np.minimum(idx, n_particles - 1)
            px, py, mb, Pb = px[idx], py[idx], mb[idx], Pb[idx]
            sr = max(0.5 * d, motion_sigma_m)
            px += rng.normal(0, sr, n_particles); py += rng.normal(0, sr, n_particles)
            logw = np.zeros(n_particles)
    return out

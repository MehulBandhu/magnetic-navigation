"""Where Tolles-Lawson breaks and what fixes it. Reuses the manoeuvre and
regressor code from checks.py.

1. Eddy currents with a time constant: B_eddy = C * lag_tau(dB_e/dt). Standard TL
   assumes tau -> 0. Residual vs omega*tau, and two fixes (u_i u''_j terms, or a
   fitted lag).
2. Large platform fields: Gauss-Newton on the exact |B_e + B_p| instead of the
   first-order model.
3. Observability vs magnetic inclination for a four-heading box calibration and
   for a single heading.
4. Three platform classes (survey fixed-wing, helicopter, multirotor) with the
   assumption each one breaks, and the residual before and after the fix.

python -m problem1.edge_cases  ->  tables + figures/p1_edge_*.png
"""
import math
import os

import numpy as np

from problem1.checks import B0, INCL, manoeuvre, direction_cosines, regressors, numerical_rank

os.makedirs("figures", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lag(x, tau, dt):
    # first-order low pass, exact discretisation: y' = (x - y) / tau
    if tau <= 0:
        return x.copy()
    a = math.exp(-dt / tau)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * y[i - 1] + (1 - a) * x[i]
    return y


def platform_field(u, udot, dt, a, M, C, tau=0.0, extra=None):
    """vector platform field in the body frame. extra: (t x 3) field not tied to attitude"""
    Bp = a[None, :] + B0 * u @ M.T + B0 * lag(udot, tau, dt) @ C.T
    if extra is not None:
        Bp = Bp + extra
    return Bp


def exact_scalar(u, Bp):
    return np.linalg.norm(B0 * u + Bp, axis=1)


def fit_linear(A, y):
    return A @ np.linalg.lstsq(A, y, rcond=None)[0]


def gauss_newton(u, udot, m, iters=4, extra=None):
    """fit a, M, C (21 params) to exact scalar data by Gauss-Newton with column
    scaling and a min-norm step (the null modes stay at zero). Iteration 1 from
    zero is the linear TL fit; later iterations pick up the second-order term.
    extra: optional (T,) signal such as motor current; its body-frame field
    direction (3 params) is fitted too."""
    def predict(theta):
        a, M, C = theta[:3], theta[3:12].reshape(3, 3), theta[12:21].reshape(3, 3)
        Bp = a[None, :] + B0 * u @ M.T + B0 * udot @ C.T
        if extra is not None:
            Bp = Bp + extra[:, None] * theta[21:][None, :]
        tot = B0 * u + Bp
        return np.linalg.norm(tot, axis=1), tot

    theta = np.zeros(21 + (0 if extra is None else 3))
    pred, tot = predict(theta)
    for _ in range(iters):
        d = tot / pred[:, None]                                   # d m / d Bp
        cols = [d]
        cols += [B0 * u[:, [j]] * d[:, [i]] for i in range(3) for j in range(3)]      # M_ij
        cols += [B0 * udot[:, [j]] * d[:, [i]] for i in range(3) for j in range(3)]   # C_ij
        if extra is not None:
            cols.append(extra[:, None] * d)
        J = np.hstack(cols)
        sc = np.linalg.norm(J, axis=0)
        sc[sc == 0] = 1
        step = np.linalg.lstsq(J / sc, m - pred, rcond=1e-10)[0] / sc
        # backtracking: halve the step until the residual decreases
        for _ in range(20):
            new_pred, new_tot = predict(theta + step)
            if rms(m - new_pred) < rms(m - pred):
                break
            step = step / 2
        theta, pred, tot = theta + step, new_pred, new_tot
    return pred


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# 1 ------------------------------------------------------------------------
def eddy_lag():
    print("\n== 1. eddy currents with a time constant ==")
    t, ang, rate = manoeuvre("roll+pitch+yaw")
    dt = t[1] - t[0]
    u, udot = direction_cosines(t, ang, rate)
    uddot = np.gradient(udot, dt, axis=0)
    rng = np.random.default_rng(1)
    a = rng.normal(0, 200, 3)
    M = rng.normal(0, 1, (3, 3)) * 2e-3
    C = rng.normal(0, 1, (3, 3)) * 1e-2
    A = regressors(u, udot)
    A_acc = np.hstack([A, B0 * np.stack([u[:, i] * uddot[:, j] for i in range(3) for j in range(3)], 1)])
    omega = 2 * math.pi * 0.10
    taus = np.logspace(-2, 0.7, 12)
    print(f"{'tau [s]':>8} {'omega*tau':>10} {'standard TL':>12} {'+ u u_ddot':>11} {'fitted lag':>11}   residual rms nT")
    out = []
    for tau in taus:
        Bp = platform_field(u, udot, dt, a, M, C, tau)
        y = (u * Bp).sum(1)                       # first-order scalar, so only the lag is on trial here
        r_std = rms(y - fit_linear(A, y))
        r_acc = rms(y - fit_linear(A_acc, y))
        best = min((rms(y - fit_linear(regressors(u, lag(udot, tt, dt)), y)), tt) for tt in np.logspace(-2.3, 1, 40))
        out.append((r_std, r_acc, best[0]))
        print(f"{tau:8.3f} {omega*tau:10.3f} {r_std:12.3f} {r_acc:11.3f} {best[0]:11.3f}   (tau_hat {best[1]:.3f})")
    out = np.array(out)
    plt.figure(figsize=(5.5, 3.6))
    for i, lab in enumerate(["standard 18-term TL", "+9 terms u_i u''_j", "fitted first-order lag"]):
        plt.loglog(omega * taus, out[:, i], "o-", label=lab)
    plt.axhline(1, ls="--", c="k", lw=0.8); plt.xlabel("omega tau  (manoeuvre rate x eddy time constant)")
    plt.ylabel("residual rms [nT]"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("figures/p1_edge_eddy_lag.png", dpi=130)


# 2 ------------------------------------------------------------------------
def nonlinear_refit():
    print("\n== 2. large platform fields: linear TL vs Gauss-Newton on the exact model ==")
    t, ang, rate = manoeuvre("roll+pitch+yaw")
    dt = t[1] - t[0]
    u, udot = direction_cosines(t, ang, rate)
    A = regressors(u, udot)
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 3)
    M = rng.normal(0, 1, (3, 3)) * 3e-3
    C = rng.normal(0, 1, (3, 3)) * 5e-3
    print(f"{'|Bp| nT':>9} {'eta':>7} {'linear TL':>10} {'GN 3 it':>9} {'GN 30 it':>9}   residual rms nT (noise free)")
    rows = []
    for s in np.logspace(0, 1.6, 8):
        Bp = platform_field(u, udot, dt, 300 * s * a, s * M, s * C)
        m = exact_scalar(u, Bp)
        mag = np.linalg.norm(Bp, axis=1).mean()
        r_lin = rms(m - B0 - fit_linear(A, m - B0))
        r_gn2 = rms(m - gauss_newton(u, udot, m, iters=3))
        r_gn4 = rms(m - gauss_newton(u, udot, m, iters=30))
        rows.append((mag, r_lin, r_gn2, r_gn4))
        print(f"{mag:9.0f} {mag/B0:7.4f} {r_lin:10.3f} {r_gn2:9.3f} {r_gn4:9.3f}")
    rows = np.array(rows)
    plt.figure(figsize=(5.5, 3.6))
    plt.loglog(rows[:, 0], rows[:, 1], "o-", label="linear TL (first order)")
    plt.loglog(rows[:, 0], rows[:, 2], "s-", label="Gauss-Newton, 3 iterations")
    plt.loglog(rows[:, 0], np.maximum(rows[:, 3], 1e-3), "^-", label="Gauss-Newton, converged (clipped at 1e-3)")
    plt.axhline(1, ls="--", c="k", lw=0.8); plt.xlabel("mean |B_p| [nT]"); plt.ylabel("residual rms [nT]")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/p1_edge_nonlinear.png", dpi=130)


# 3 ------------------------------------------------------------------------
def box_pattern(incl, headings):
    # the classic calibration: roll, pitch, yaw excitations repeated on several headings
    blocks = []
    for psi0 in headings:
        t, ang, rate = manoeuvre("roll+pitch+yaw", T=60.0)
        ang = [ang[0], ang[1], ang[2] + psi0]
        blocks.append(regressors(*direction_cosines(t, ang, rate, incl=incl)))
    return np.vstack(blocks)


def latitude():
    print("\n== 3. observability vs magnetic inclination ==")
    incls = np.radians(np.linspace(-89, 89, 37))
    box = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    print(f"{'incl deg':>9} {'rank box':>9} {'cond box':>10} {'rank N only':>12} {'cond N only':>12}   (condition number of the rank-17 part)")
    res = []
    for inc in incls:
        r1, sv1, _ = numerical_rank(box_pattern(inc, box))
        r2, sv2, _ = numerical_rank(box_pattern(inc, [0.0]))
        c1, c2 = sv1[0] / sv1[16], sv2[0] / sv2[16]
        res.append((c1, c2))
        if abs(math.degrees(inc)) % 30 < 5 or abs(math.degrees(inc)) > 85:
            print(f"{math.degrees(inc):9.0f} {r1:9d} {c1:10.1e} {r2:12d} {c2:12.1e}")
    res = np.array(res)
    plt.figure(figsize=(5.5, 3.6))
    plt.semilogy(np.degrees(incls), res[:, 0], label="4-heading box pattern")
    plt.semilogy(np.degrees(incls), res[:, 1], label="single heading (north)")
    plt.axvline(math.degrees(INCL), ls=":", c="gray"); plt.xlabel("magnetic inclination [deg]")
    plt.ylabel("condition number, rank-17 part"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("figures/p1_edge_latitude.png", dpi=130)


# 4 ------------------------------------------------------------------------
PLATFORMS = {
    # name: (permanent nT, induced, eddy s, eddy tau s, motor-current term nT)
    "survey fixed-wing": (300, 4e-3, 3e-3, 0.03, 0),
    "helicopter": (600, 6e-3, 2e-2, 0.4, 0),
    "multirotor drone": (2500, 2e-3, 5e-4, 0.01, 400),
}


def platforms():
    print("\n== 4. three platform classes ==")
    t, ang, rate = manoeuvre("roll+pitch+yaw")
    dt = t[1] - t[0]
    u, udot = direction_cosines(t, ang, rate)
    A = regressors(u, udot)
    rng = np.random.default_rng(3)
    # throttle: what a multirotor does while manoeuvring, not a function of attitude
    throttle = 0.5 + 0.3 * np.cumsum(rng.normal(0, 1, len(t))) / math.sqrt(len(t))
    throttle = np.clip(throttle, 0.2, 1.0)
    names, results = [], []
    print(f"{'platform':>18} {'|Bp|':>6} {'standard TL':>12} {'+lag+GN':>9} {'+current':>9}   residual rms nT")
    for name, (perm, ind, eddy, tau, motor) in PLATFORMS.items():
        a = rng.normal(0, 1, 3); a = perm * a / np.linalg.norm(a)
        M = rng.normal(0, 1, (3, 3)) * ind
        C = rng.normal(0, 1, (3, 3)) * eddy
        extra = motor * (throttle - throttle.mean())[:, None] * np.array([0.3, 0.2, 0.93])[None, :]
        Bp = platform_field(u, udot, dt, a, M, C, tau, extra)
        m = exact_scalar(u, Bp)
        r_std = rms(m - B0 - fit_linear(A, m - B0))
        cur = throttle - throttle.mean()
        r_fix, best_tau = min((rms(m - gauss_newton(u, lag(udot, tt, dt), m, iters=12)), tt) for tt in np.logspace(-2.3, 1, 25))
        r_cur = min(rms(m - gauss_newton(u, lag(udot, tt, dt), m, iters=12, extra=cur)) for tt in np.logspace(-2.3, 1, 25))
        names.append(name); results.append((r_std, r_fix, r_cur))
        print(f"{name:>18} {np.linalg.norm(Bp, axis=1).mean():6.0f} {r_std:12.3f} {r_fix:9.3f} {r_cur:9.3f}   (tau_hat {best_tau:.2f} s)")
    results = np.array(results)
    x = np.arange(len(names)); w = 0.25
    plt.figure(figsize=(6.5, 3.6))
    for i, lab in enumerate(["standard TL", "+ eddy lag + nonlinear refit", "+ current regressor"]):
        plt.bar(x + (i - 1) * w, np.maximum(results[:, i], 1e-3), w, label=lab)
    plt.yscale("log"); plt.ylim(1e-3, 100); plt.xticks(x, names); plt.axhline(1, ls="--", c="k", lw=0.8)
    plt.ylabel("residual rms [nT]  (clipped at 1e-3)"); plt.legend(fontsize=8)
    plt.title("toy parameter study: coefficient scales are illustrative, not measured platforms", fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/p1_edge_platforms.png", dpi=130)


if __name__ == "__main__":
    eddy_lag()
    nonlinear_refit()
    latitude()
    platforms()
    print("\nfigures written to figures/p1_edge_*.png")

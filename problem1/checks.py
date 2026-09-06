"""Numerical companions to the Problem 1 derivations. Each check drops an
assumption the derivation made and shows what happens.

1. Tolles-Lawson rank for real manoeuvres (rank 17 / 16 band-passed / 1 straight and level),
   and the null vectors are the ones predicted (isotropic eddy, isotropic induced).
2. The first-order scalar linearisation: exact |Be + Bp| against the TL fit as the
   platform field grows. Residual goes like eta^2 B / 2 until it exceeds sensor noise.
3. Gradient power vs altitude from the actual map generator: slope beta - 4 in the
   middle regime, saturation from grid resolution below, exponential from map size above.
4. Vector vs scalar magnetometer with the full 3x2 gradient geometry and attitude-error
   covariance. No characteristic-scale approximation.

python -m problem1.checks   ->  prints tables, writes figures/p1_*.png
"""
import math
import os

import numpy as np
import torch

from magscale.grf import ground_amplitude, sample_fields

os.makedirs("figures", exist_ok=True)
B0 = 50_000.0
INCL, DECL = math.radians(70.0), math.radians(-2.0)   # roughly Scotland


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx          # body -> NED


def manoeuvre(kind, T=120.0, fs=10.0):
    """returns t and the three euler angles plus their analytic rates"""
    t = np.arange(0, T, 1 / fs)
    a = math.radians(15)
    amp = {"straight": (0, 0, 0), "roll": (a, 0, 0), "roll+pitch": (a, 0.7 * a, 0),
           "roll+pitch+yaw": (a, 0.7 * a, 2 * a)}[kind]
    freq = (0.10, 0.137, 0.061)
    ang = [A * np.sin(2 * math.pi * f * t) for A, f in zip(amp, freq)]
    rate = [A * 2 * math.pi * f * np.cos(2 * math.pi * f * t) for A, f in zip(amp, freq)]
    return t, ang, rate


def direction_cosines(t, ang, rate, finite_diff=False, incl=INCL, decl=DECL):
    u_ned = np.array([math.cos(incl) * math.cos(decl), math.cos(incl) * math.sin(decl), math.sin(incl)])
    roll, pitch, yaw = ang
    u = np.array([rot(r, p, y).T @ u_ned for r, p, y in zip(roll, pitch, yaw)])
    if finite_diff:
        return u, np.gradient(u, t, axis=0)
    # exact: u_body = R^T u_ned, so udot = -omega_body x u_body, omega from euler rates (ZYX)
    rd, pd, yd = rate
    w = np.stack([rd - yd * np.sin(pitch),
                  pd * np.cos(roll) + yd * np.cos(pitch) * np.sin(roll),
                  -pd * np.sin(roll) + yd * np.cos(pitch) * np.cos(roll)], 1)
    return u, -np.cross(w, u)


def regressors(u, udot, B=B0):
    ux, uy, uz = u.T
    dx, dy, dz = udot.T
    cols = [ux, uy, uz,
            B * ux * ux, B * ux * uy, B * ux * uz, B * uy * uy, B * uy * uz, B * uz * uz,
            B * ux * dx, B * ux * dy, B * ux * dz, B * uy * dx, B * uy * dy, B * uy * dz,
            B * uz * dx, B * uz * dy, B * uz * dz]
    return np.stack(cols, 1)


def numerical_rank(A, tol=1e-8, ref=None):
    # columns are scaled to unit norm so one tolerance means something; a column whose
    # norm is negligible relative to `ref` (the unfiltered matrix) is treated as zero
    s = np.linalg.norm(A, axis=0)
    ref_norm = np.linalg.norm(ref, axis=0) if ref is not None else s
    zero = s < 1e-10 * max(ref_norm.max(), 1e-300)
    if zero.all():
        return 0, np.zeros(A.shape[1]), np.eye(A.shape[1])
    s = np.where(zero, 1.0, s)
    An = np.where(zero[None, :], 0.0, A / s)                                   # column scaling so one tolerance means something
    U, sv, Vt = np.linalg.svd(An, full_matrices=False)
    r = int((sv > tol * sv[0]).sum())
    null = Vt[r:] / s                            # undo the scaling: null vectors of the original A
    return r, sv / sv[0], null


def check_tl_rank():
    print("\n== 1. Tolles-Lawson rank by manoeuvre ==")
    iso_eddy = np.zeros(18); iso_eddy[[9, 13, 17]] = 1
    iso_ind = np.zeros(18); iso_ind[[3, 6, 8]] = 1
    for kind in ["straight", "roll", "roll+pitch", "roll+pitch+yaw"]:
        t, ang, rate = manoeuvre(kind)
        A = regressors(*direction_cosines(t, ang, rate))
        rank, sv, null = numerical_rank(A)
        Abp = A - A.mean(0)                      # crude band-pass: kill the DC part
        rank_bp, _, null_bp = numerical_rank(Abp, ref=A)
        line = f"{kind:16s} raw rank {rank:2d}  band-passed rank {rank_bp:2d}"
        if kind != "straight":
            # projection of the predicted null modes onto the numerical null space
            P = null.T @ np.linalg.pinv(null.T)
            Pbp = null_bp.T @ np.linalg.pinv(null_bp.T)
            line += (f"   iso-eddy in null: {np.linalg.norm(P @ iso_eddy) / np.linalg.norm(iso_eddy):.4f}"
                     f"   iso-induced in bp-null: {np.linalg.norm(Pbp @ iso_ind) / np.linalg.norm(iso_ind):.4f}")
        print(line)
    print("singular values, roll+pitch+yaw, scaled columns:", np.array2string(sv, precision=2))
    t, ang, rate = manoeuvre("roll+pitch")
    _, sv_fd, _ = numerical_rank(regressors(*direction_cosines(t, ang, rate, finite_diff=True)))
    print(f"same manoeuvre with finite-difference udot at 10 Hz: smallest singular value {sv_fd[-1]:.1e} "
          f"instead of 0. Numerical derivatives break the exact dependency and leak into the isotropic eddy mode.")


def check_linearisation():
    print("\n== 2. first-order scalar model vs exact |Be + Bp| ==")
    t, ang, rate = manoeuvre("roll+pitch+yaw")
    u, udot = direction_cosines(t, ang, rate)
    A = regressors(u, udot)
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 3)                       # permanent, unit scale
    M = rng.normal(0, 1, (3, 3)) * 1e-2 / 3       # induced, dimensionless
    C = rng.normal(0, 1, (3, 3)) * 5e-3           # eddy, seconds
    print(f"{'eta':>8} {'lin residual rms':>17} {'|Bp_perp|^2/2B':>15} {'TL fit residual rms':>20}")
    etas, res_lin, res_fit = [], [], []
    for scale in np.logspace(-1, 1.5, 8):
        Bp = scale * (300 * a[None, :] + B0 * u @ M.T + B0 * udot @ C.T)
        eta = np.linalg.norm(Bp, axis=1).mean() / B0
        exact = np.linalg.norm(B0 * u + Bp, axis=1)
        first = B0 + (u * Bp).sum(1)
        beta_hat = np.linalg.lstsq(A, exact - B0, rcond=None)[0]
        e_lin = np.sqrt(np.mean((exact - first) ** 2))
        e_fit = np.sqrt(np.mean((exact - B0 - A @ beta_hat) ** 2))
        perp = np.sqrt(np.mean(((np.linalg.norm(Bp, axis=1) ** 2 - (u * Bp).sum(1) ** 2) / (2 * B0)) ** 2))
        etas.append(eta); res_lin.append(e_lin); res_fit.append(e_fit)
        print(f"{eta:8.4f} {e_lin:17.3f} {perp:15.3f} {e_fit:20.3f}   nT")
    slope = np.polyfit(np.log(etas), np.log(res_lin), 1)[0]
    print(f"log-log slope of linearisation residual vs eta: {slope:.2f} (2 expected)")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3.5))
    plt.loglog(etas, res_lin, "o-", label="exact - first order")
    plt.loglog(etas, res_fit, "s-", label="exact - best TL fit")
    plt.axhline(1.0, ls="--", c="k", label="1 nT sensor noise")
    plt.xlabel("eta = |Bp| / B"); plt.ylabel("residual rms [nT]"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/p1_linearisation.png", dpi=130)


def check_gradient_scaling(n=1024, dx=100.0, sigma0=50.0):
    print("\n== 3. gradient power vs altitude from the map generator ==")
    hs = np.logspace(0, 4, 25)                    # 1 m to 10 km
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 4))
    for beta in [2.5, 3.0, 3.5, 4.0]:
        A0, k = ground_amplitude(n, beta, dx, sigma0)
        # spectral: E|grad m|^2 = (1/n^2) sum_k k^2 A^2 exp(-2kh). Exact for the generator.
        G2 = np.array([((k ** 2 * A0 ** 2 * torch.exp(-2 * k * h)).sum() / n ** 2).item() for h in hs])
        # monte carlo with finite differences on the grid, as available from a gridded map
        mc = []
        for h in hs[::6]:
            f = sample_fields(4, n // 2, beta, float(h), dx, sigma0)
            gx, gy = torch.gradient(f, spacing=dx, dim=(1, 2))
            mc.append((gx ** 2 + gy ** 2).mean().item())
        mid = (hs > 3 * dx) & (hs < n * dx / 60)
        slope = np.polyfit(np.log(hs[mid]), np.log(G2[mid]), 1)[0]
        print(f"beta {beta}: mid-regime slope {slope:6.2f}   beta-4 = {beta-4:5.2f}   "
              f"gradient rms at h=0 {math.sqrt(G2[0]):.3f}, 200 m {math.sqrt(np.interp(200, hs, G2)):.3f} nT/m")
        plt.loglog(hs, np.sqrt(G2), label=f"beta={beta}, slope {slope/2:.2f}")
        plt.loglog(hs[::6], np.sqrt(mc), "k.", ms=4)
    plt.axvline(dx, ls=":", c="gray"); plt.axvline(n * dx / (2 * math.pi), ls=":", c="gray")
    plt.xlabel("altitude h [m]"); plt.ylabel("rms gradient [nT/m]  (dots: finite differences,\nbelow the line for h < dx because they cannot see sub-grid structure)")
    plt.title("sigma_pos ~ sigma / rms gradient ~ h^((4-beta)/2)", fontsize=9)
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/p1_gradient_scaling.png", dpi=130)


def vector_anomaly(T, dx):
    """consistent 3-component anomaly from a scalar anomaly map, plus its horizontal gradients.
    Potential field above the sources: b_hat = (-i kx, -i ky, |k|) Phi_hat, T_hat = f . b_hat."""
    n = T.shape[0]
    k1 = 2 * math.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2)
    f = np.array([math.cos(INCL) * math.cos(DECL), math.cos(INCL) * math.sin(DECL), -math.sin(INCL)])  # z up
    g = -1j * kx * f[0] - 1j * ky * f[1] + k * f[2]
    g[0, 0] = 1.0
    Th = np.fft.fft2(T)
    Ph = Th / g
    Ph[0, 0] = 0
    bh = np.stack([-1j * kx * Ph, -1j * ky * Ph, k * Ph])
    b = np.fft.ifft2(bh, axes=(1, 2)).real
    G = np.stack([np.fft.ifft2(1j * kx * bh, axes=(1, 2)).real, np.fft.ifft2(1j * ky * bh, axes=(1, 2)).real], -1)
    return b, G, f            # G: (3, n, n, 2)


def check_vector_vs_scalar(beta=3.0, h=200.0, sigma=1.0, dx=100.0, n=128):
    print("\n== 4. vector vs scalar magnetometer, full geometry ==")
    T = sample_fields(4, n, beta, h, dx, 50.0).numpy()
    Js, Jv_iso, Jv_head = [], [], []
    eps = np.logspace(-6, -2, 25)
    zhat = np.array([0, 0, 1.0])
    for m in T:
        b, G, f = vector_anomaly(m, dx)
        Gm = G.reshape(3, -1, 2).transpose(1, 0, 2)               # pixels x 3 x 2
        gs = np.einsum("pij,i->pj", Gm, f)                          # scalar gradient = G^T f
        Js.append(np.einsum("pi,pj->pij", gs, gs).mean(0) / sigma ** 2)
        Bx = np.array([[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]]) * B0
        rows_iso, rows_head = [], []
        for e in eps:
            for rows, Sth in [(rows_iso, e ** 2 * np.eye(3)), (rows_head, e ** 2 * np.outer(zhat, zhat))]:
                Sv = sigma ** 2 * np.eye(3) + Bx @ Sth @ Bx.T
                Si = np.linalg.inv(Sv)
                rows.append(np.einsum("pki,kl,plj->pij", Gm, Si, Gm).mean(0))
        Jv_iso.append(rows_iso); Jv_head.append(rows_head)
    Js = np.mean(Js, 0); Jv_iso = np.mean(Jv_iso, 0); Jv_head = np.mean(Jv_head, 0)
    ts = np.trace(Js)
    gain_iso = np.array([np.trace(J) for J in Jv_iso]) / ts
    gain_head = np.array([np.trace(J) for J in Jv_head]) / ts
    Gv2 = np.mean([np.sum(vector_anomaly(m, dx)[1] ** 2) / m.size for m in T])
    Gs2 = ts * sigma ** 2
    eps_formula = sigma / B0 * math.sqrt(max(Gv2 / Gs2 - 1, 0))
    half = 1 + 0.5 * (gain_iso[0] - 1)
    eps_half = np.interp(-half, -gain_iso, eps)
    print(f"ideal vector / scalar Fisher information (eps=0): {gain_iso[0]:.2f}")
    print(f"eps where the vector advantage halves (isotropic attitude error): {eps_half:.2e} rad = {math.degrees(eps_half):.4f} deg")
    print(f"characteristic scale sigma/B0 sqrt(Gv^2/Gs^2 - 1) from the derivation: {eps_formula:.2e} rad "
          f"(a scale, not a crossing: the full-covariance vector information never drops below scalar)")
    print(f"sigma/B0 = {sigma / B0:.2e} rad;  vector info never drops below scalar: min gain {gain_iso.min():.3f}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 3.5))
    plt.semilogx(eps, gain_iso, label="isotropic attitude error")
    plt.semilogx(eps, gain_head, label="heading error only")
    plt.axvline(sigma / B0, ls="--", c="k", label="sigma / B0")
    plt.axvline(eps_formula, ls=":", c="r", label="characteristic scale sigma/B0 sqrt(Gv^2/Gs^2-1)")
    plt.axhline(1, c="gray", lw=0.8)
    plt.xlabel("attitude error eps [rad]"); plt.ylabel("tr J_vector / tr J_scalar"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figures/p1_vector_vs_scalar.png", dpi=130)


if __name__ == "__main__":
    check_tl_rank()
    check_linearisation()
    check_gradient_scaling()
    check_vector_vs_scalar()
    print("\nfigures written to figures/p1_*.png")

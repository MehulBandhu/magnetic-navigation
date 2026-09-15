"""Live Tolles-Lawson calibration: the regressor of problem1.checks built row by row, the
least-squares fit refreshed every second, the numerical rank of the regressor so far and the
residual of the latest fit. The band-passed variant demeans the regressor and the data over
the rows seen so far, the crude band-pass of checks.py, which is what turns rank 17 into 16.

The calibration is what the demo's first panel shows. Its outputs are also the starting
point of the corrector: the coefficients fitted on the calibration box are applied to later
flights and the corrector predicts what they leave.
"""
import numpy as np

from problem1.checks import B0, regressors, numerical_rank
from problem1.edge_cases import lag


def regressor(u, udot, B_nT=B0):
    """(T, 18) Tolles-Lawson regressor from problem1.checks; B scalar or (T,) array"""
    if np.ndim(B_nT):
        return regressors(u, udot, B=np.reshape(B_nT, -1))
    return regressors(u, udot, B=B_nT)


def highpass(x, dt_s, tau_s=30.0):
    """first-order high-pass, x minus its lag (edge_cases.lag) with time constant tau_s, so a
    slow sensor drift stays out of a calibration fit; works on (T,) and (T, k)"""
    return x - lag(x, tau_s, dt_s)


def _lstsq(A, y):
    # columns scaled to unit norm and a relative tolerance, as numerical_rank does, so a
    # column that the band-pass has reduced to rounding noise gets a zero coefficient
    # instead of thousands of nT along a null direction
    sc = np.linalg.norm(A, axis=0)
    sc = np.where(sc < 1e-10 * max(sc.max(), 1e-300), 1.0, sc)
    return np.linalg.lstsq(A / sc, y, rcond=1e-8)[0] / sc


def fit(A, y_nT, demean=False, highpass_tau_s=None, dt_s=0.1):
    """least-squares coefficients (18,) of y on A; with demean, both are centred first and the
    mean of y is returned as the 19th entry so apply() can reproduce the level; with
    highpass_tau_s both are high-passed first and the 19th entry is zero (the level is not
    fitted, the navigator absorbs it)"""
    if highpass_tau_s:
        beta = _lstsq(highpass(A, dt_s, highpass_tau_s), highpass(y_nT, dt_s, highpass_tau_s))
        return np.concatenate([beta, [0.0]])
    if demean:
        beta = _lstsq(A - A.mean(0), y_nT - y_nT.mean())
        return np.concatenate([beta, [y_nT.mean() - A.mean(0) @ beta]])
    return np.concatenate([_lstsq(A, y_nT), [0.0]])


def apply(coef, A):
    """the linear model's prediction (T,) nT for coefficients from fit()"""
    return A @ coef[:18] + coef[18]


def live(u, udot, y_nT, dt_s, refit_every_s=1.0, demean=False, B_nT=B0, highpass_tau_s=None):
    """run the calibration causally along a flight. y_nT is the scalar reading minus the
    ambient magnitude (what the platform adds, plus drift and noise). Returns a dict of (T,)
    arrays: rank (of the rows so far, demeaned or high-passed when either is set),
    residual_nT (y minus the latest fit, causal), rms_nT (rms of the latest fit's residual
    over the rows it was fitted on, in the fitted band), and coef (T, 19), the coefficients
    in force at each step."""
    A = regressor(u, udot, B_nT)
    Ah = highpass(A, dt_s, highpass_tau_s) if highpass_tau_s else A
    yh = highpass(y_nT, dt_s, highpass_tau_s) if highpass_tau_s else y_nT
    T = len(y_nT)
    every = max(1, int(round(refit_every_s / dt_s)))
    rank = np.zeros(T, dtype=int)
    rms = np.zeros(T)
    coef = np.zeros((T, 19))
    c = np.zeros(19)
    r = 0
    fit_rms = 0.0
    for t in range(T):
        if t % every == every - 1 or t == T - 1:
            rows = A[: t + 1]
            if highpass_tau_s:
                r = numerical_rank(Ah[: t + 1], ref=rows)[0]
                c = fit(rows, y_nT[: t + 1], highpass_tau_s=highpass_tau_s, dt_s=dt_s)
                fit_rms = float(np.sqrt(np.mean((yh[: t + 1] - Ah[: t + 1] @ c[:18]) ** 2)))
            else:
                r = numerical_rank(rows - rows.mean(0), ref=rows)[0] if demean else numerical_rank(rows)[0]
                c = fit(rows, y_nT[: t + 1], demean)
                fit_rms = float(np.sqrt(np.mean((y_nT[: t + 1] - apply(c, rows)) ** 2)))
        rank[t] = r
        rms[t] = fit_rms
        coef[t] = c
    residual = y_nT - np.einsum("ti,ti->t", A, coef[:, :18]) - coef[:, 18]
    return dict(rank=rank, residual_nT=residual, rms_nT=rms, coef=coef, A=A)

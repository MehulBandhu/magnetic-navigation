"""Correctness checks for the pieces everything else rests on. Runs on CPU in about a minute.
    python -m pytest -q tests
"""
import math

import numpy as np
import torch

from magscale.grf import sample_fields, random_mask, line_mask, make_mask
from magscale.floor import patch_covariance, wiener_matrix, prior_variance, dense_wiener, denoise_floor, exact_floor


def test_masks_have_exact_ratio_and_spacing():
    m = random_mask(8, 32, 0.75, torch.Generator().manual_seed(0))
    assert (m.sum(dim=(1, 2)) == round(0.75 * 32 * 32)).all()
    l = line_mask(8, 32, 4, torch.Generator().manual_seed(0))
    assert torch.allclose(l.float().mean(dim=(1, 2)), torch.tensor(0.75))


def test_generator_matches_its_covariance():
    # the covariance matrix used by the floor must be the covariance of what the generator makes
    n, beta, h = 16, 3.0, 200.0
    gen = torch.Generator().manual_seed(1)
    f = sample_fields(4000, n, beta, h, gen=gen).double().view(4000, -1)
    emp = f.T @ f / f.shape[0]
    K = patch_covariance(n, beta, h)
    rel = (emp - K).norm() / K.norm()
    assert rel < 0.08, rel
    assert abs(emp.diagonal().mean().item() / prior_variance(n, beta, h) - 1) < 0.05


def test_ground_rms_is_sigma0():
    f = sample_fields(64, 32, 3.5, 0.0, sigma0=50.0, gen=torch.Generator().manual_seed(2))
    assert abs(f.std().item() / 50.0 - 1) < 0.05


def test_floor_is_below_prior_and_falls_with_altitude():
    f0 = exact_floor(32, 3.5, 0.0, 1.0, n_masks=2)[0]
    f8 = exact_floor(32, 3.5, 800.0, 1.0, n_masks=2)[0]
    assert 0 < f8 < f0 < prior_variance(32, 3.5, 0.0)


def test_dense_wiener_matches_finite_patch_denoising():
    n, beta, h, sigma = 32, 3.0, 200.0, 1.0
    K = patch_covariance(n, beta, h)
    a, b = dense_wiener(n, beta, h, sigma), denoise_floor(K, sigma)
    assert abs(a / b - 1) < 0.15, (a, b)


def test_wiener_matrix_is_the_conditional_mean():
    # E[(m_H - W y_O)^2] over samples must equal the analytic posterior variance
    n, beta, h, sigma = 16, 3.5, 200.0, 1.0
    K = patch_covariance(n, beta, h)
    gen = torch.Generator().manual_seed(3)
    hid = random_mask(1, n, 0.75, gen)[0].flatten()
    W, var = wiener_matrix(K, hid, sigma)
    clean = sample_fields(3000, n, beta, h, gen=gen).double().view(3000, -1)
    noisy = clean + sigma * torch.randn(clean.shape, generator=gen, dtype=torch.float64)
    pred = noisy[:, ~hid] @ W.T
    mse = ((pred - clean[:, hid]) ** 2).mean().item()
    assert abs(mse / var.mean().item() - 1) < 0.06, (mse, var.mean().item())


def test_fixed_mask_is_shared_with_linear_reference():
    # train.py --fixed_mask and linear_ref.py must draw the same mask for the same seed
    from magscale.train import fixed_mask
    class A: n = 64; mask = "random"; ratio = 0.75; spacing = 4; seed = 0
    m1 = fixed_mask(A, "cpu")
    gen = torch.Generator().manual_seed(1000 + A.seed)
    m2 = make_mask(A.mask, 1, A.n, A.ratio, A.spacing, gen, "cpu")[0]
    assert torch.equal(m1, m2)


def test_fit_recovers_known_exponents():
    import subprocess, sys
    subprocess.run([sys.executable, "scripts/check_fitter.py"], check=True)


def test_tolles_lawson_null_modes():
    # the isotropic eddy mode is invisible because u . udot = 0; the isotropic induced mode is a
    # constant offset that demeaning removes. Raw rank 17, band-passed rank 16 for a generic manoeuvre
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from problem1.checks import manoeuvre, direction_cosines, regressors, numerical_rank
    t, ang, rate = manoeuvre("roll+pitch+yaw")
    A = regressors(*direction_cosines(t, ang, rate))
    rank, _, null = numerical_rank(A)
    assert rank == 17
    iso_eddy = np.zeros(18); iso_eddy[[9, 13, 17]] = 1
    v = null[0] / np.linalg.norm(null[0])            # null rows are null vectors of A
    assert abs(v @ iso_eddy) / np.linalg.norm(iso_eddy) > 0.99
    rank_bp, _, _ = numerical_rank(A - A.mean(0), ref=A)
    assert rank_bp == 16
    # straight and level: raw rank 1, nothing left after demeaning
    t, ang, rate = manoeuvre("straight")
    A = regressors(*direction_cosines(t, ang, rate))
    assert numerical_rank(A)[0] == 1 and numerical_rank(A - A.mean(0), ref=A)[0] == 0


def test_gradient_power_exponent():
    # rms gradient of a k^-beta field falls as h^((beta-4)/2) between the grid and map cutoffs
    from magscale.grf import amplitude
    beta, n, dx = 2.5, 64, 100.0
    g = []
    for h in [200.0, 400.0]:
        A = amplitude(2 * n, beta, h, dx, 50.0)
        kx = torch.fft.fftfreq(2 * n, d=dx) * 2 * math.pi
        k2 = (kx[:, None] ** 2 + kx[None, :] ** 2)
        g.append(float((A ** 2 * k2).sum().sqrt()))
    slope = math.log(g[1] / g[0]) / math.log(2.0)
    assert abs(slope - (beta - 4) / 2) < 0.1, slope

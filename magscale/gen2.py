"""Generator v2: anomaly maps with the statistics real tiles have and Gaussian fields lack.

A tile from EMAG2 differs from a Gaussian random field in three measurable ways: its spectral
slope varies from tile to tile (1.7 to 5.4 on land, by province), its amplitude varies within a
tile (kurtosis 3.9 median, near 10 at the 90th percentile; 3 for Gaussian), and its texture has a
strike direction. The construction here reproduces all three and stays Gaussian conditional on
its latents, which is what keeps a floor computable:

    m_0(x) = s(x) g(x)          g: Gaussian field with slope beta and an anisotropic stretch
                                s: exp(sigma_s z(x) - sigma_s^2), z a smooth field with unit variance over the patch,
                                   so E[s^2] = 1 and kurtosis = 3 exp(4 sigma_s^2)
    m_h   = upward continuation of m_0 by exp(-kh), applied to the product (the product is the
            source field; continuation is linear, so m_h is still harmonic above ground)

Per patch: beta ~ U[beta_lo, beta_hi], sigma_s ~ U[mod_lo, mod_hi], stretch ~ U[1, aniso_max],
angle ~ U[0, pi). With mod_range (0.1, 0.6) the kurtosis runs from 3.1 to 12, which brackets the
EMAG2 land tiles.

Two references for any estimator on this data, both by Monte Carlo over validation patches:
  oracle   the conditional mean given the mask and all latents, including s(x): a Gaussian
           problem with covariance S K S, a lower bound on what any estimator can do
  gauss    the Gaussian estimator with the patch's beta and stretch but no modulation, the
           estimator the v1 study calls exact; on v2 data it is not, and its realised error is
           the upper reference
The covariance of the continued, modulated field is approximated by S K_h S (modulating at
altitude instead of at ground level); s varies over scales much longer than h, and the error of
the approximation is checked in tests. Generation is single precision; the reference floors are
computed in double. The runs in scripts/sweeps/gen2.txt were made with a double-precision
version of the generator, which is slower on a GPU and otherwise identical at these magnitudes.
"""
import math

import torch

from .grf import wavenumbers, covariance_lags
from .floor import wiener_matrix


def anisotropic_k(n, dx, stretch, angle, device="cpu"):
    """|k| with the plane stretched by `stretch` along `angle` and compressed across it, so
    the field's correlation length is longer along the strike. (batch,) parameters -> (batch, n, n)"""
    freqs = torch.fft.fftfreq(n, d=dx, device=device).float() * 2 * math.pi
    ky, kx = torch.meshgrid(freqs, freqs, indexing="ij")
    c, s = torch.cos(angle).float().view(-1, 1, 1), torch.sin(angle).float().view(-1, 1, 1)
    a = stretch.float().view(-1, 1, 1)
    along = kx * c + ky * s
    across = -kx * s + ky * c
    return torch.sqrt(along ** 2 / a + across ** 2 * a)


def ground_amplitude_v2(ng, beta, dx, sigma0, stretch, angle, device="cpu"):
    """per-patch Fourier filter (batch, ng, ng), each normalised to rms sigma0 at ground level"""
    k = anisotropic_k(ng, dx, stretch, angle, device)
    A2 = torch.zeros_like(k)
    nz = k > 0
    A2[nz] = k[nz] ** (-beta.float().view(-1, 1, 1).expand_as(k)[nz])
    A2 = A2 * (sigma0 ** 2 * ng ** 2 / A2.sum(dim=(1, 2), keepdim=True))
    return A2.sqrt()


def modulation(batch, ng, mod_sigma, gen, device="cpu", n=None):
    """s(x) = exp(sigma_s z - sigma_s^2) with z a smooth (slope 3) field normalised to unit
    variance over the central n x n crop, so sigma_s sets the within-patch kurtosis"""
    n = n or ng // 2
    k = wavenumbers(ng, 100.0, device).float()
    A2 = torch.zeros_like(k); A2[k > 0] = k[k > 0] ** -3.0
    z = torch.fft.ifft2(A2.sqrt() * torch.fft.fft2(torch.randn(batch, ng, ng, generator=gen, device=device, dtype=torch.float32))).real
    c = n // 2
    zc = z[:, c:c + n, c:c + n]
    z = (z - zc.mean(dim=(1, 2), keepdim=True)) / zc.std(dim=(1, 2), keepdim=True)
    ms = mod_sigma.float().view(-1, 1, 1)
    return torch.exp(ms * z - ms ** 2)


def draw_latents(batch, cfg, gen, device="cpu"):
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(batch, generator=gen, device=device)
    return dict(beta=u(*cfg["beta_range"]), mod_sigma=u(*cfg["mod_range"]),
                stretch=u(1.0, cfg["aniso_max"]), angle=u(0.0, math.pi))


def sample_fields_v2(batch, n, h, cfg, dx=100.0, sigma0=50.0, gen=None, device="cpu"):
    """n x n patches at altitude h (float or (batch,) tensor) from the scale-mixture generator.
    Returns clean (batch, n, n) float32 and the latents, with s cropped to the patch."""
    ng = 2 * n
    lat = draw_latents(batch, cfg, gen, device)
    A0 = ground_amplitude_v2(ng, lat["beta"], dx, sigma0, lat["stretch"], lat["angle"], device)
    w = torch.randn(batch, ng, ng, generator=gen, device=device, dtype=torch.float32)
    g = torch.fft.ifft2(A0 * torch.fft.fft2(w)).real
    s = modulation(batch, ng, lat["mod_sigma"], gen, device, n)
    m0 = s * g
    k = wavenumbers(ng, dx, device).double()
    hh = torch.as_tensor(h, dtype=torch.float32, device=device).reshape(-1, 1, 1)
    mh = torch.fft.ifft2(torch.exp(-k * hh) * torch.fft.fft2(m0)).real
    c = n // 2
    lat["s"] = s[:, c:c + n, c:c + n].float().contiguous()
    return mh[:, c:c + n, c:c + n].float().contiguous(), lat


def patch_cov_from_amplitude(n, A_h):
    """n^2 x n^2 covariance of an n-patch cropped from the 2n grid whose Fourier amplitude is A_h"""
    ng = A_h.shape[-1]
    cov = covariance_lags(A_h)
    iy, ix = torch.meshgrid(torch.arange(n, device=A_h.device), torch.arange(n, device=A_h.device), indexing="ij")
    iy, ix = iy.flatten(), ix.flatten()
    return cov[(iy[:, None] - iy[None, :]) % ng, (ix[:, None] - ix[None, :]) % ng]


def reference_floors(clean, noisy, mask, lat, h, dx=100.0, sigma0=50.0, sigma=1.0, max_patches=32):
    """Monte Carlo over the first max_patches of a batch: the oracle floor (posterior variance of
    the modulated Gaussian problem, s known) and the realised error of the Gaussian estimator
    with the patch's beta and stretch and no modulation. Returns dict of means, nT^2."""
    n = clean.shape[-1]; ng = 2 * n
    k = wavenumbers(ng, dx, clean.device).double()
    hh = torch.as_tensor(h, dtype=torch.float32, device=clean.device).reshape(-1, 1, 1)
    A0 = ground_amplitude_v2(ng, lat["beta"], dx, sigma0, lat["stretch"], lat["angle"], clean.device)
    A_h = (A0 * torch.exp(-k * hh)).double()
    oracle, gauss_err, gauss_pred = [], [], []
    for i in range(min(max_patches, clean.shape[0])):
        K = patch_cov_from_amplitude(n, A_h[i]).cpu()
        hid = mask[i].flatten().cpu()
        y = noisy[i].flatten().double().cpu(); m = clean[i].flatten().double().cpu()
        # the Gaussian estimator: the v1 "exact" one, wrong on this data
        W, var = wiener_matrix(K, hid, sigma)
        pred = W @ y[~hid]
        gauss_err.append(float(((pred - m[hid]) ** 2).mean())); gauss_pred.append(float(var.mean()))
        # the oracle: same problem with the modulation known
        sv = lat["s"][i].flatten().double().cpu()
        Ks = sv[:, None] * K * sv[None, :]
        _, var_o = wiener_matrix(Ks, hid, sigma)
        oracle.append(float(var_o.mean()))
    return dict(oracle=sum(oracle) / len(oracle), gauss_realised=sum(gauss_err) / len(gauss_err),
                gauss_predicted=sum(gauss_pred) / len(gauss_pred), n_patches=len(oracle))


def parse_cfg(a):
    lo, hi = (float(x) for x in a.beta_range.split(","))
    mlo, mhi = (float(x) for x in a.mod_range.split(","))
    return dict(beta_range=(lo, hi), mod_range=(mlo, mhi), aniso_max=float(a.aniso_max))

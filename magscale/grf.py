import math
import torch


def wavenumbers(n, dx, device="cpu"):
    # |k| in rad/m on an n x n grid, fft layout
    k1 = 2 * math.pi * torch.fft.fftfreq(n, d=dx, device=device)
    kx, ky = torch.meshgrid(k1, k1, indexing="ij")
    return torch.sqrt(kx ** 2 + ky ** 2)


def ground_amplitude(n, beta, dx=100.0, sigma0=50.0, device="cpu"):
    # Fourier filter A0(k) = c k^(-beta/2). field = ifft2(A * fft2(white)).real,
    # so per-pixel variance is sum(A^2)/n^2. c is set so the ground-level rms
    # is sigma0 nT on this grid. k=0 dropped: anomaly maps are zero mean by
    # construction once the IGRF is removed.
    k = wavenumbers(n, dx, device).double()
    A2 = torch.zeros_like(k)
    nz = k > 0
    A2[nz] = k[nz] ** (-beta)
    A2 = A2 * (sigma0 ** 2 * n ** 2 / A2.sum())
    return A2.sqrt(), k


def amplitude(n, beta, h, dx=100.0, sigma0=50.0, device="cpu"):
    # upward continuation to altitude h (metres): amplitude times exp(-k h)
    A0, k = ground_amplitude(n, beta, dx, sigma0, device)
    return A0 * torch.exp(-k * h)


def covariance_lags(A):
    # cov(lag) = E[m(x) m(x+lag)] = (1/n^2) sum_k A^2 e^{ik.lag}, which is
    # exactly what ifft2 computes. Periodic on the generation grid.
    return torch.fft.ifft2(A ** 2).real


def sample_fields(batch, n, beta, h, dx=100.0, sigma0=50.0, gen=None, device="cpu"):
    """n x n anomaly patches (nT) at altitude h, cropped from a 2n periodic grid.

    h may be a float or a (batch,) tensor of altitudes in metres.
    """
    ng = 2 * n
    A0, k = ground_amplitude(ng, beta, dx, sigma0, device)
    h = torch.as_tensor(h, dtype=torch.float64, device=device).reshape(-1, 1, 1)
    A = (A0 * torch.exp(-k * h)).float()
    w = torch.randn(batch, ng, ng, generator=gen, device=device)
    f = torch.fft.ifft2(A * torch.fft.fft2(w)).real
    c = n // 2
    return f[:, c:c + n, c:c + n].contiguous()


def random_mask(batch, n, ratio, gen=None, device="cpu"):
    # True = hidden. Exact hidden count so the floor is well defined.
    nh = round(ratio * n * n)
    r = torch.rand(batch, n * n, generator=gen, device=device)
    idx = r.argsort(dim=1)[:, :nh]
    m = torch.zeros(batch, n * n, dtype=torch.bool, device=device)
    m.scatter_(1, idx, True)
    return m.view(batch, n, n)


def line_mask(batch, n, spacing, gen=None, device="cpu"):
    # survey flight lines: every `spacing`-th row observed, random offset and
    # orientation; the rows between lines are hidden.
    off = torch.randint(0, spacing, (batch, 1), generator=gen, device=device)
    rows = torch.arange(n, device=device)[None, :]
    observed = (rows - off) % spacing == 0
    m = ~observed[:, :, None].expand(batch, n, n)
    swap = torch.rand(batch, generator=gen, device=device) < 0.5
    m = torch.where(swap[:, None, None], m.transpose(1, 2), m)
    return m.contiguous()


def make_mask(kind, batch, n, ratio=0.75, spacing=4, gen=None, device="cpu"):
    if kind == "random":
        return random_mask(batch, n, ratio, gen, device)
    if kind == "lines":
        return line_mask(batch, n, spacing, gen, device)
    raise ValueError(kind)


def add_noise(fields, sigma, gen=None):
    return fields + sigma * torch.randn(fields.shape, generator=gen, device=fields.device)


def observe(noisy, mask, scale):
    """model input: (B, 2, n, n) = [masked noisy values / scale, mask]"""
    x = torch.where(mask, torch.zeros_like(noisy), noisy) / scale
    return torch.stack([x, mask.float()], dim=1)

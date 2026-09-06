"""Two animations, written as GIFs into figures/.

upward     one ground-level field continued upward from 0 to 1000 m, with the exact floor and
           the number of modes above the noise at each altitude (Problems 1C, 2A, 3B in one picture)
navigate   a scalar magnetometer flown along a trajectory with one turn; the position likelihood
           over a search window is a ridge along the local gradient after one measurement (rank-one
           Fisher information) and becomes a spot once the trajectory has turned (Problem 1C)

python -m magscale.animate upward
python -m magscale.animate navigate
"""
import argparse
import math
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .grf import amplitude
from .floor import exact_floor, effective_modes
from .magnav import true_map, trajectory, bilinear


def frames_to_gif(frames, path, ms=120):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


def fig_to_image(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return Image.frombytes("RGB", (w, h), fig.canvas.tostring_rgb() if hasattr(fig.canvas, "tostring_rgb") else fig.canvas.buffer_rgba().tobytes()[::1]).convert("P", palette=Image.ADAPTIVE)


def render(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return Image.fromarray(buf.copy()).convert("P", palette=Image.ADAPTIVE)


def upward(beta=3.5, n=128, dx=100.0, sigma0=50.0, sigma=1.0, seed=0, out="figures/upward_continuation.gif"):
    # one white-noise realisation, continued to each altitude by e^-kh in Fourier space
    gen = torch.Generator().manual_seed(seed)
    noise = torch.fft.fft2(torch.randn(2 * n, 2 * n, generator=gen, dtype=torch.float64))
    hs = np.concatenate([np.zeros(6), np.linspace(0, 1000, 45)[1:], np.full(8, 1000.0)])
    floors, modes = {}, {}
    for h in sorted(set(np.round(hs, 0))):
        floors[h] = exact_floor(64, beta, float(h), sigma, n_masks=2)[0]
        modes[h] = effective_modes(64, beta, float(h), sigma, dx, sigma0)
    frames = []
    ground = None
    for h in hs:
        A = amplitude(2 * n, beta, float(h), dx, sigma0)
        f = torch.fft.ifft2(noise * A).real[n // 2:n // 2 + n, n // 2:n // 2 + n].numpy()
        if ground is None:
            ground = f.copy(); vmax = float(np.abs(ground).max())
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.6))
        ax[0].imshow(f, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[0].set_title("same colour scale as ground level: the field fades", fontsize=9)
        v = float(np.abs(f).max()) + 1e-9
        ax[1].imshow(f, cmap="RdBu_r", vmin=-v, vmax=v); ax[1].set_title("rescaled per frame: the field smooths", fontsize=9)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        hk = float(np.round(h, 0))
        fig.suptitle(f"beta = {beta}, 12.8 km of map, altitude h = {h:5.0f} m   |   rms {f.std():5.1f} nT   |   "
                     f"exact floor {floors[hk]:7.3f} nT^2   |   modes above 1 nT noise: {int(modes[hk]):4d} of 4096", fontsize=9)
        fig.tight_layout()
        frames.append(render(fig)); plt.close(fig)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames_to_gif(frames, out, ms=110)
    print("wrote", out, len(frames), "frames")


def navigate(beta=3.5, h=800.0, n=256, dx=100.0, sigma0=50.0, sigma=1.0, seed=3, nsteps=60, out="figures/navigation_likelihood.gif"):
    mp = true_map(n, beta, h, dx, sigma0, seed)
    rng = np.random.default_rng(seed)
    y0, x0 = n / 2 + 10, n / 2 - 20
    ys, xs = trajectory(nsteps, y0, x0, math.radians(20), nsteps // 2, math.radians(90))
    z = bilinear(mp, ys, xs) + sigma * rng.standard_normal(nsteps)
    # likelihood of a position offset (dy, dx) applied to the whole trajectory, accumulated
    half, step = 6.0, 0.25
    offs = np.arange(-half, half + step / 2, step)
    DY, DX = np.meshgrid(offs, offs, indexing="ij")
    frames = []
    for t in range(1, nsteps + 1):
        ll = np.zeros_like(DY)
        for k in range(t):
            pred = bilinear(mp, ys[k] + DY, xs[k] + DX)
            ll -= (pred - z[k]) ** 2 / (2 * sigma ** 2)
        post = np.exp(ll - ll.max())
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
        v = float(np.abs(mp).max())
        ax[0].imshow(mp, cmap="RdBu_r", vmin=-v, vmax=v, extent=[0, n * dx / 1000, n * dx / 1000, 0])
        ax[0].plot(xs[:t] * dx / 1000, ys[:t] * dx / 1000, "k-", lw=1.5); ax[0].plot(xs[t - 1] * dx / 1000, ys[t - 1] * dx / 1000, "ko", ms=5)
        ax[0].set_xlim((x0 - 40) * dx / 1000, (x0 + 40) * dx / 1000); ax[0].set_ylim((y0 + 40) * dx / 1000, (y0 - 40) * dx / 1000)
        ax[0].set_xlabel("km"); ax[0].set_ylabel("km"); ax[0].set_title(f"anomaly map at {h:.0f} m (beta {beta}); {t} of {nsteps} scalar measurements, 1 nT noise", fontsize=9)
        ax[1].imshow(post, cmap="magma", origin="lower", extent=[-half * dx, half * dx, -half * dx, half * dx], vmin=0, vmax=1)
        ax[1].plot(0, 0, "w+", ms=12, mew=1.5)
        ax[1].set_xlabel("east offset [m]"); ax[1].set_ylabel("north offset [m]")
        J = np.zeros((2, 2))
        for k in range(t):
            gy = (bilinear(mp, ys[k] + 0.5, xs[k]) - bilinear(mp, ys[k] - 0.5, xs[k])) / dx
            gx = (bilinear(mp, ys[k], xs[k] + 0.5) - bilinear(mp, ys[k], xs[k] - 0.5)) / dx
            J += np.outer([gy, gx], [gy, gx]) / sigma ** 2
        ev = np.linalg.eigvalsh(J)
        ratio = ev.min() / ev.max() if ev.max() > 0 else 0.0
        if ratio < 1e-3:
            info = "Fisher information effectively rank one: a ridge along the contour, across the gradient"
        else:
            info = f"both directions observed: CRB {math.sqrt(np.trace(np.linalg.inv(J))):.0f} m, eigenvalue ratio {ratio:.2f}"
        leg = "straight leg" if t <= nsteps // 2 else "after the 90 degree turn"
        ax[1].set_title(f"position likelihood over a 1.2 km window, {leg}\n{info}", fontsize=9)
        fig.tight_layout()
        frames.append(render(fig)); plt.close(fig)
    frames += [frames[-1]] * 10
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames_to_gif(frames, out, ms=90)
    print("wrote", out, len(frames), "frames")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["upward", "navigate", "all"])
    a = p.parse_args()
    if a.what in ("upward", "all"):
        upward()
    if a.what in ("navigate", "all"):
        navigate()


if __name__ == "__main__":
    main()

"""Transfer across scale and altitude on a real survey grid.

EMAG2 is 3.7 km pixels continued to 4 km, h/dx = 1.08. A national aeromagnetic grid at 80 m
pixels flown at 80 m clearance is h/dx = 1 at a physical scale fifty times smaller. The
synthetic study says only h/dx matters; this script tests that on real data, and uses the
decay model, upward continuation by exp(-kh), to place the same real ground at several
altitudes. Continuation from a draped survey is approximate over rough terrain.

python -m magscale.grid_eval --grid data/yilgarn_tmi_80m.tif --dx 80 --h0 80 --altitudes 0,240,720,3920 \\
       --run runs/vitxxl_gen2_hmix_D0_s0_72k.json --tag _yilgarn

For each altitude: square tiles of n pixels are cut from the continued grid (h/dx =
(h0 + dh)/dx), detrended, their slope, kurtosis and top-octave excess measured, rescaled to the
training amplitude, masked 75%, reconstructed by the network, the Gaussian estimator (fitted
beta and beta 3.5) and interpolation, with the error-concentration statistic. Results go to
results/grid<tag>.json with one summary per altitude; the table script picks them up.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from .emag2 import (detrend, fit_beta, Wiener, interpolate, load_model, DX)
from .floor import prior_variance
from .grf import random_mask, observe


def load_grid(path, dx_hint=None):
    """returns (grid, dx). A latitude-longitude netCDF (as the NCI subset service delivers) is
    resampled to square pixels: rows are kept, columns are stretched by 1 / cos(latitude) so
    that a pixel is the north-south cell size in both directions; dx is that size in metres."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nc":
        import netCDF4
        from scipy.ndimage import zoom
        ds = netCDF4.Dataset(path)
        names = list(ds.variables)
        var = [v for v in names if ds.variables[v].ndim == 2][0]
        g = np.array(ds.variables[var][:], dtype=np.float32)
        if hasattr(ds.variables[var], "_FillValue"):
            g[g == ds.variables[var]._FillValue] = np.nan
        g[np.abs(g) > 1e5] = np.nan
        latn = [v for v in names if v.lower().startswith("lat")]; lonn = [v for v in names if v.lower().startswith("lon")]
        if latn and lonn:
            lat = np.array(ds.variables[latn[0]][:], dtype=float); lon = np.array(ds.variables[lonn[0]][:], dtype=float)
            if lat[0] > lat[-1]:
                g = g[::-1]; lat = lat[::-1]
            dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
            dx = dlat * 111320.0
            ew = dlon * 111320.0 * math.cos(math.radians(float(np.mean(lat))))
            g = zoom(np.where(np.isnan(g), np.nanmean(g), g), (1.0, ew / dx), order=1)
            print(f"lat-lon grid: {dlat*3600:.1f} arcsec cells, {dx:.0f} m north-south, {ew:.0f} m east-west at {np.mean(lat):.1f} deg; columns resampled to {dx:.0f} m")
            return g.astype(np.float32), dx
        return g, dx_hint
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                g = src.read(1).astype(np.float32)
                nod = src.nodata
        except ImportError:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            g = np.array(Image.open(path), dtype=np.float32); nod = None
        if nod is not None:
            g[g == nod] = np.nan
    elif ext == ".npy":
        g = np.load(path).astype(np.float32)
    g[np.abs(g) > 1e5] = np.nan
    return g, dx_hint


def continue_upward(g, dx, dh):
    """upward continuation of a whole grid by dh metres, in double precision; NaNs are filled
    with the grid mean for the transform and masked again afterwards"""
    if dh <= 0:
        return g
    fill = np.where(np.isnan(g), np.nanmean(g), g).astype(np.float64)
    # the grid is not periodic; reflect-pad by a margin of several decay lengths so the
    # transform's wrap-around does not ring into the edges (a margin of 8 dh, at least 64 px)
    pad = int(min(max(64, 8 * dh / dx), min(fill.shape) // 2))
    fp = np.pad(fill, pad, mode="reflect")
    ny, nx = fp.shape
    ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi; kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    out = np.fft.ifft2(np.fft.fft2(fp) * np.exp(-k * dh)).real[pad:-pad, pad:-pad].astype(np.float32)
    out[np.isnan(g)] = np.nan
    return out


def block_mean(g, f):
    """resample by averaging f x f blocks (NaN if any pixel of the block is NaN)"""
    if f <= 1:
        return g
    ny, nx = (g.shape[0] // f) * f, (g.shape[1] // f) * f
    b = g[:ny, :nx].reshape(ny // f, f, nx // f, f)
    return b.mean(axis=(1, 3)).astype(np.float32)


def evaluate(grid, h_over_dx, model, c, n, sigma, device, max_tiles, gen):
    from .emag2 import H_EQ
    h_eq = h_over_dx * DX                          # the study's units: 100 m pixels
    train_hs = [float(x) for x in str(c["h"]).split(",")]
    h_in = torch.tensor([h_eq / max(800.0, max(train_hs))], device=device)
    sigma0 = c["sigma0"]
    wiener = Wiener(n, sigma, DX, h_eq)
    target_rms = math.sqrt(prior_variance(n, 3.5, h_eq, DX, 50.0))
    rows = []
    for i in range(0, grid.shape[0] - n + 1, n):
        for j in range(0, grid.shape[1] - n + 1, n):
            t = grid[i:i + n, j:j + n]
            if np.isnan(t).any():
                continue
            t = detrend(t.astype(np.float64))
            if t.std() < 0.5:
                continue
            # the fitting band ends where the continuation has attenuated the power by e^-6;
            # above that the spectrum is numerical noise
            fmax = int(min(24, max(6, 30 / h_over_dx)))
            beta, top = fit_beta(t, h_over_dx, fmax=fmax, return_excess=True)
            if h_over_dx > 3:
                top = float("nan")                 # the band above fmax is attenuated past float precision
            kurt = float(((t - t.mean()) ** 4).mean() / t.var() ** 2)
            t = t * (target_rms / t.std())
            clean = torch.tensor(t, dtype=torch.float32)
            noisy = clean + sigma * torch.randn(clean.shape, generator=gen)
            mask = random_mask(1, n, 0.75, gen)[0]; hid = mask.flatten()
            with torch.no_grad():
                net = (model(observe(noisy[None].to(device), mask[None].to(device), sigma0), h_in)[0] * sigma0).cpu()
            w, fl = wiener(noisy.double(), hid, beta, float(clean.var()))
            w35, _ = wiener(noisy.double(), hid, 3.5, float(clean.var()))
            wf = noisy.clone().flatten(); wf[hid] = torch.from_numpy(w).float(); wf = wf.view(n, n)
            w3 = noisy.clone().flatten(); w3[hid] = torch.from_numpy(w35).float(); w3 = w3.view(n, n)
            it = torch.from_numpy(interpolate(noisy.numpy(), mask.numpy()))
            err = lambda p: float(((p - clean) ** 2)[mask].mean())
            def top5(p):
                se = ((p - clean) ** 2)[mask].flatten().sort(descending=True).values
                return float(se[: max(1, int(0.05 * len(se)))].sum() / se.sum())
            rows.append(dict(beta=beta, top_octave_excess=top, kurtosis=kurt, mse_network=err(net), mse_wiener=err(wf),
                             mse_wiener_beta35=err(w3), floor_predicted=fl, mse_interp=err(it),
                             err_top5_gauss=top5(w3), err_top5_network=top5(net)))
            if len(rows) >= max_tiles:
                return rows
    return rows


def summarise(rows):
    a = lambda k: np.array([r[k] for r in rows])
    net, w35, wie, fl, it = a("mse_network"), a("mse_wiener_beta35"), a("mse_wiener"), a("floor_predicted"), a("mse_interp")
    return dict(tiles=len(rows), beta_median=float(np.median(a("beta"))), beta_q10=float(np.quantile(a("beta"), .1)), beta_q90=float(np.quantile(a("beta"), .9)),
                kurtosis_median=float(np.median(a("kurtosis"))), kurtosis_q90=float(np.quantile(a("kurtosis"), .9)),
                top_octave_excess_median=float(np.nanmedian(a("top_octave_excess"))),
                mse_network_median=float(np.median(net)), mse_wiener_beta35_median=float(np.median(w35)), mse_wiener_median=float(np.median(wie)),
                mse_interp_median=float(np.median(it)), wiener_over_own_floor_median=float(np.median(wie / fl)),
                network_over_wiener_beta35_median=float(np.median(net / w35)), frac_network_beats_wiener=float(np.mean(net < w35)),
                frac_network_beats_interp=float(np.mean(net < it)), err_top5_gauss_median=float(np.median(a("err_top5_gauss"))),
                err_top5_network_median=float(np.median(a("err_top5_network"))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--dx", type=float, default=None, help="pixel size in metres (read from the file for a lat-lon netCDF)")
    p.add_argument("--h0", type=float, required=True, help="survey height above ground in metres")
    p.add_argument("--altitudes", default="0", help="extra continuation heights in metres, comma separated; 0 = the grid as flown")
    p.add_argument("--keep_dx", action="store_true",
                   help="keep the pixel size after continuation (the altitude axis at fixed scale); by default the grid is "
                        "resampled so that h/dx stays what it was when flown (the scale axis at fixed h/dx)")
    p.add_argument("--run", default="runs/vitxxl_gen2_hmix_D0_s0_72k.json")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--max_tiles", type=int, default=400)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid, dx0 = load_grid(a.grid, a.dx)
    if dx0 is None:
        raise SystemExit("pixel size unknown: pass --dx")
    a.dx = dx0
    print(f"grid {grid.shape}, {np.isnan(grid).mean()*100:.1f}% missing, rms {np.nanstd(grid):.1f} nT, pixel {a.dx:.0f} m, flown at {a.h0:.0f} m")
    model, c = load_model(a.run, device)
    out = dict(grid=os.path.basename(a.grid), dx=a.dx, h0=a.h0, run=os.path.basename(a.run)[:-5], n=a.n, sigma=a.sigma, per_altitude={})
    gen = torch.Generator().manual_seed(0)
    for dh in [float(x) for x in a.altitudes.split(",")]:
        g = continue_upward(grid, a.dx, dh)
        dx = a.dx
        if not a.keep_dx and dh > 0:
            f = max(1, int(round((a.h0 + dh) / a.h0)))
            g = block_mean(g, f); dx = a.dx * f
        hdx = (a.h0 + dh) / dx
        rows = evaluate(g, hdx, model, c, a.n, a.sigma, device, a.max_tiles, gen)
        if len(rows) < 4:
            print(f"altitude {a.h0 + dh:.0f} m, pixel {dx:.0f} m: {len(rows)} tiles, too few (a {a.n}-pixel tile is {a.n * dx / 1000:.0f} km; the window is {grid.shape[1] * a.dx / 1000:.0f} km wide)")
            continue
        s = summarise(rows); s["h_over_dx"] = hdx; s["dh"] = dh; s["dx"] = dx
        out["per_altitude"][f"{a.h0 + dh:.0f}"] = dict(summary=s, tiles=rows)
        print(f"altitude {a.h0 + dh:.0f} m, pixel {dx:.0f} m (h/dx {hdx:.2f}, tile {a.n * dx / 1000:.1f} km): {s['tiles']} tiles; beta {s['beta_median']:.2f} ({s['beta_q10']:.2f}, {s['beta_q90']:.2f}); "
              f"kurtosis {s['kurtosis_median']:.2f} (q90 {s['kurtosis_q90']:.2f}); top-octave {s['top_octave_excess_median']:.2f}; "
              f"Gaussian / own floor {s['wiener_over_own_floor_median']:.2f}; network / Gaussian(3.5) {s['network_over_wiener_beta35_median']:.2f}; "
              f"beats interp {s['frac_network_beats_interp']:.2f}; worst-5% share {s['err_top5_gauss_median']:.2f}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/grid{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

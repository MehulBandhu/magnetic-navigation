"""Out-of-distribution test on a real grid: EMAG2 v3 (2-arc-minute Earth Magnetic Anomaly Grid).

The study's fields are scale-free, so the only quantity that carries over from the synthetic
setting is h / dx. EMAG2's upward-continued grid is at 4 km above sea level on 2-arc-minute
pixels (about 3.7 km), h / dx = 1.08, which in the study's units (dx = 100 m) is h = 108 m.

For each 64 x 64 tile in a region with survey coverage:
  1. remove the mean and a linear trend, estimate the spectral slope beta from the radial power
     spectrum (the model P(k) ~ k^-beta e^-2kh with h / dx fixed, beta fitted);
  2. rescale to the training amplitude, add 1 nT of noise, hide 75% of the pixels at random;
  3. reconstruct the hidden pixels with the trained network, with the exact Gaussian
     conditional mean for the tile's fitted beta (the estimator that would be optimal if the
     tile were a stationary Gaussian field), and with linear interpolation;
  4. record the hidden-pixel errors, and the Gaussian estimator's predicted floor for the tile,
     which is what its error would be if the tile were Gaussian and stationary.

python -m magscale.emag2 convert data/EMAG2_V3_20170530.csv      -> data/emag2_upcont.npy
python -m magscale.emag2 run                                      -> results/emag2.json, figures/emag2_*.png
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from scipy.ndimage import zoom

from .floor import patch_covariance, wiener_matrix, prior_variance
from .grf import random_mask, observe
from .models import build

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NLAT, NLON = 5400, 10800                      # 2-arc-minute global grid
H_OVER_DX = 4.0 / 3.706                       # 4 km continuation over the pixel size at the equator
DX, H_EQ = 100.0, 100.0 * 4.0 / 3.706          # the study's units
REGIONS = {
    "australia": dict(lat=(-40, -12), lon=(113, 154)),
    "north_america": dict(lat=(30, 55), lon=(-125, -70)),
}
MISSING = 99999.0


# ----------------------------------------------------------------------------- conversion
def convert(path, out="data/emag2_upcont.npy"):
    """Read whatever form of EMAG2 v3 is on disk and write the upward-continued grid as one
    float32 array indexed [lat, lon] with lat from -90 (row 0) and lon from -180 (column 0)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                g = src.read(1).astype(np.float32)
                if src.transform.e < 0:           # north-up rasters store row 0 at +90 latitude
                    g = g[::-1]
        except ImportError:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            g = np.array(Image.open(path), dtype=np.float32)[::-1]
    elif ext in (".nc", ".grd"):
        import netCDF4
        ds = netCDF4.Dataset(path)
        name = [v for v in ds.variables if ds.variables[v].ndim == 2][0]
        g = np.array(ds.variables[name][:], dtype=np.float32)
        lat = ds.variables[[v for v in ds.variables if v.lower().startswith("lat")][0]][:]
        if lat[0] > lat[-1]:
            g = g[::-1]
    else:
        # CSV: one row per cell. EMAG2 v3 columns are i, j, LON, LAT, SeaLevel, UpCont, Code, Error;
        # indices are recomputed from LON and LAT so the column order of i and j does not matter.
        import pandas as pd
        g = np.full((NLAT, NLON), np.nan, dtype=np.float32)
        code = np.full((NLAT, NLON), 999, dtype=np.int16)
        for chunk in pd.read_csv(path, header=None, chunksize=2_000_000, low_memory=False):
            num = chunk.apply(pd.to_numeric, errors="coerce")        # a header line, if any, becomes NaN and is dropped
            num = num.dropna(subset=[2, 3, 5])
            lon = num.iloc[:, 2].to_numpy(float); lat = num.iloc[:, 3].to_numpy(float)
            val = num.iloc[:, 5].to_numpy(float)
            # cell centres are at half-cell offsets, so floor, not round, gives the cell index
            ci = np.floor((lon + 180.0) * 30.0).astype(int) % NLON
            ri = np.clip(np.floor((lat + 90.0) * 30.0).astype(int), 0, NLAT - 1)
            g[ri, ci] = val
            if num.shape[1] > 6:
                code[ri, ci] = num.iloc[:, 6].fillna(999).to_numpy(int)
        np.save(out.replace("upcont", "code"), code)
        print(f"wrote {out.replace('upcont', 'code')}: source codes ({(code < 100).mean()*100:.1f}% of cells from land compilations)")
    g = g.astype(np.float32)
    g[np.abs(g) >= MISSING] = np.nan
    if g.shape != (NLAT, NLON):
        raise SystemExit(f"grid has shape {g.shape}, expected {(NLAT, NLON)}")
    np.save(out, g)
    print(f"wrote {out}: {g.shape}, {np.isnan(g).mean()*100:.1f}% missing, rms {np.nanstd(g):.1f} nT")


def region_slice(name):
    r = REGIONS[name]
    r0, r1 = int((r["lat"][0] + 90) * 30), int((r["lat"][1] + 90) * 30)
    c0, c1 = int((r["lon"][0] + 180) * 30), int((r["lon"][1] + 180) * 30)
    return slice(r0, r1), slice(c0, c1)


# ----------------------------------------------------------------------------- spectral slope
def detrend(t):
    n = t.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    A = np.stack([np.ones(n * n), x.ravel(), y.ravel()], 1)
    coef, *_ = np.linalg.lstsq(A, t.ravel(), rcond=None)
    return t - (A @ coef).reshape(n, n)


def fit_beta(t, h_over_dx=H_OVER_DX, fmin=3, fmax=24):
    """slope of the radial power spectrum with the continuation factor divided out:
    log P(k) + 2 k h = log C - beta log k, over integer radial frequencies fmin..fmax (cycles/tile)"""
    n = t.shape[0]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    P = np.abs(np.fft.fft2(t * w)) ** 2
    f = np.fft.fftfreq(n) * n
    fr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    bins = np.arange(fmin, fmax + 1)
    Pr = np.array([P[(fr >= b - 0.5) & (fr < b + 0.5)].mean() for b in bins])
    k = 2 * np.pi * bins / n                    # radians per pixel
    y = np.log(Pr) + 2 * k * h_over_dx
    slope, intercept = np.polyfit(np.log(k), y, 1)
    return float(-slope)


# ----------------------------------------------------------------------------- reconstructions
def detrend_projection(n):
    # the tiles are detrended (mean and plane removed); the same projection applied to the
    # stationary covariance gives the covariance of a detrended tile, K' = P K P
    y, x = np.mgrid[0:n, 0:n]
    A = np.stack([np.ones(n * n), x.ravel(), y.ravel()], 1)
    P = np.eye(n * n) - A @ np.linalg.solve(A.T @ A, A.T)
    return torch.tensor(P)


class Wiener:
    """exact conditional mean for a detrended tile treated as a Gaussian field with the fitted
    beta at the equivalent altitude; detrended covariances cached per beta bin and rescaled to
    the tile's sample variance (whose expectation under the model is trace(K')/n^2)"""
    def __init__(self, n, sigma, dx=DX, h=H_EQ):
        self.n, self.sigma, self.dx, self.h = n, sigma, dx, h
        self.P = detrend_projection(n)
        self.cache = {}

    def K(self, beta, detrended=True):
        b = round(max(1.5, min(6.5, beta)) * 10) / 10
        if (b, detrended) not in self.cache:
            K = patch_covariance(self.n, b, self.h, self.dx, 50.0)
            Kd = self.P @ K @ self.P
            # the stationary prior (what the network was trained on) is scaled so that its
            # detrended variance matches the tile, the same normalisation as the detrended one
            self.cache[(b, detrended)] = (Kd if detrended else K, float(Kd.diagonal().mean()))
        return self.cache[(b, detrended)]

    def __call__(self, noisy, hid, beta, tile_var, detrended=True):
        K, pv = self.K(beta, detrended)
        Ks = K * (tile_var / pv)                # match the tile's variance
        W, var = wiener_matrix(Ks, hid, self.sigma)
        pred = W @ noisy.flatten()[~hid]
        return pred.numpy(), float(var.mean())


def interpolate(noisy, mask):
    from scipy.interpolate import griddata
    n = noisy.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    obs = ~mask
    out = griddata((yy[obs], xx[obs]), noisy[obs], (yy, xx), method="linear")
    near = griddata((yy[obs], xx[obs]), noisy[obs], (yy, xx), method="nearest")
    out[np.isnan(out)] = near[np.isnan(out)]
    return out


def load_model(run, device):
    r = json.load(open(run))
    c = r["config"]
    model = build(c["model"], c["n"], c["width"], c["depth"], c["heads"], not c["no_altitude"], c["mup"], c["base_width"], c["patch"]).to(device)
    model.load_state_dict(torch.load(run.replace(".json", ".pt"), map_location=device))
    model.eval()
    return model, c


# ----------------------------------------------------------------------------- main run
def run(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = np.load(a.grid)
    code_path = a.grid.replace("upcont", "code")
    code = np.load(code_path) if os.path.exists(code_path) else None
    if code is None:
        print("no source-code grid found; tiles are not restricted to land surveys")
    model, c = load_model(a.run, device)
    train_hs = [float(x) for x in str(c["h"]).split(",")]
    h_in = torch.tensor([H_EQ / max(800.0, max(train_hs))], device=device)
    sigma0 = c["sigma0"]
    wiener = Wiener(a.n, a.sigma)
    gen = torch.Generator().manual_seed(0)
    rows, candidates, prev = [], [], None
    for region in a.regions.split(","):
        rs, cs = region_slice(region)
        sub = grid[rs, cs]
        subcode = code[rs, cs] if code is not None else None
        for i in range(0, sub.shape[0] - a.n + 1, a.n):
            # the grid is latitude-longitude: a pixel is 3.7 km north-south and 3.7 cos(lat) km
            # east-west. Take 64 / cos(lat) columns for 64 rows and resample the columns so the
            # tile is square in kilometres; the isotropic prior and the radial spectrum then apply
            lat = -90 + (rs.start + i + a.n / 2) / 30
            wcols = int(round(a.n / math.cos(math.radians(lat))))
            for j in range(0, sub.shape[1] - wcols + 1, wcols):
                raw = sub[i:i + a.n, j:j + wcols]
                if np.isnan(raw).any():
                    continue
                if subcode is not None:
                    cc = subcode[i:i + a.n, j:j + wcols]
                    if np.mean((cc > 0) & (cc < 100)) < 0.95:      # land compilations: codes 1 to 38 in EMAG2 v3
                        continue
                t = zoom(raw.astype(np.float64), (1.0, a.n / wcols), order=1)[:, :a.n]
                if t.shape != (a.n, a.n):
                    continue
                t = detrend(t)
                rms = float(t.std())
                if rms < 1.0:
                    continue
                beta = fit_beta(t)
                # rescale to the amplitude the network trained on at this altitude
                target_rms = math.sqrt(prior_variance(a.n, 3.5, H_EQ, DX, 50.0))
                scale = target_rms / rms
                clean = torch.tensor(t * scale, dtype=torch.float32)
                noisy = clean + a.sigma * torch.randn(clean.shape, generator=gen)
                mask = random_mask(1, a.n, 0.75, gen)[0]
                hid = mask.flatten()
                with torch.no_grad():
                    x = observe(noisy[None].to(device), mask[None].to(device), sigma0)
                    net = (model(x, h_in)[0] * sigma0).cpu()
                    # mixing test on real inputs: f(0.5 y1 + 0.5 y2) against 0.5 f(y1) + 0.5 f(y2), same mask
                    nonlin = float("nan")
                    if prev is not None:
                        y2 = prev
                        xm = observe((0.5 * noisy + 0.5 * y2)[None].to(device), mask[None].to(device), sigma0)
                        x2 = observe(y2[None].to(device), mask[None].to(device), sigma0)
                        f_mix = (model(xm, h_in)[0] * sigma0).cpu(); f2 = (model(x2, h_in)[0] * sigma0).cpu()
                        lin = 0.5 * net + 0.5 * f2
                        nonlin = float((((f_mix - lin) ** 2 * mask).sum() / ((lin ** 2) * mask).sum()).sqrt())
                    prev = noisy.clone()
                w_pred, floor_pred = wiener(noisy.double(), hid, beta, float(clean.var()))
                # the same estimator with the network's training beta: separates per-tile adaptivity
                # (which the network was never given) from everything else
                w35_pred, _ = wiener(noisy.double(), hid, 3.5, float(clean.var()))
                # and with the stationary (not detrended) covariance, the network's implicit prior
                w35s_pred, _ = wiener(noisy.double(), hid, 3.5, float(clean.var()), detrended=False)
                interp = torch.from_numpy(interpolate(noisy.numpy(), mask.numpy()))
                m = mask
                err = lambda p: float(((p - clean) ** 2)[m].mean())
                w_full = noisy.clone().flatten(); w_full[hid] = torch.from_numpy(w_pred).float(); w_full = w_full.view(a.n, a.n)
                w35_full = noisy.clone().flatten(); w35_full[hid] = torch.from_numpy(w35_pred).float(); w35_full = w35_full.view(a.n, a.n)
                w35s_full = noisy.clone().flatten(); w35s_full[hid] = torch.from_numpy(w35s_pred).float(); w35s_full = w35s_full.view(a.n, a.n)
                rec = dict(region=region, lat=float(lat), lon=float(-180 + (cs.start + j + wcols / 2) / 30),
                           beta=beta, rms_nT=rms, kurtosis=float(((t - t.mean()) ** 4).mean() / t.var() ** 2),
                           mse_network=err(net), mse_wiener=err(w_full), mse_wiener_beta35=err(w35_full), mse_wiener_beta35_stationary=err(w35s_full), floor_predicted=floor_pred, mse_interp=err(interp), nonlinearity=nonlin)
                rows.append(rec)
                candidates.append((clean, noisy, mask, net, w_full, interp, rec))
        print(f"{region}: {sum(r['region'] == region for r in rows)} tiles")
    # four example tiles spread across both regions, with a slope inside the assumed range
    ok = [x for x in candidates if 2.5 < x[6]["beta"] < 4.0]
    examples = [ok[k] for k in np.linspace(0, len(ok) - 1, 4).astype(int)] if len(ok) >= 4 else ok
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    summary = summarise(rows)
    json.dump(dict(run=os.path.basename(a.run)[:-5], h_equivalent_m=H_EQ, sigma=a.sigma, n=a.n, tiles=rows, summary=summary),
              open("results/emag2.json", "w"), indent=1)
    plot_beta_map(rows, grid, "figures/emag2_beta_map.png")
    plot_summary(rows, summary, "figures/emag2_summary.png")
    plot_examples(examples, "figures/emag2_tiles.png")
    for k, v in summary.items():
        print(f"{k}: " + ", ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in v.items()))
    # the figures depend on the map regions; the resampled tiles are plotted at their centre


def summarise(rows):
    out = {}
    for region in sorted({r["region"] for r in rows}) + ["all"]:
        rs = [r for r in rows if region == "all" or r["region"] == region]
        if not rs:
            continue
        b = np.array([r["beta"] for r in rs])
        net = np.array([r["mse_network"] for r in rs]); wie = np.array([r["mse_wiener"] for r in rs])
        fl = np.array([r["floor_predicted"] for r in rs]); it = np.array([r["mse_interp"] for r in rs])
        w35 = np.array([r["mse_wiener_beta35"] for r in rs]); w35s = np.array([r["mse_wiener_beta35_stationary"] for r in rs])
        nl = np.array([r["nonlinearity"] for r in rs if r.get("nonlinearity") == r.get("nonlinearity")])
        out[region] = dict(tiles=len(rs), nonlinearity_median=float(np.median(nl)) if len(nl) else float("nan"),
                           frac_beta_clipped=float(np.mean((b < 1.5) | (b > 6.5))), beta_median=float(np.median(b)), beta_q10=float(np.quantile(b, 0.1)), beta_q90=float(np.quantile(b, 0.9)),
                           frac_beta_in_2p5_4=float(np.mean((b >= 2.5) & (b <= 4.0))),
                           mse_network_median=float(np.median(net)), mse_wiener_median=float(np.median(wie)),
                           floor_predicted_median=float(np.median(fl)), mse_interp_median=float(np.median(it)),
                           wiener_over_own_floor_median=float(np.median(wie / fl)),
                           mse_wiener_beta35_median=float(np.median(w35)), mse_wiener_beta35_stationary_median=float(np.median(w35s)),
                           network_over_wiener_beta35_stationary_median=float(np.median(net / w35s)),
                           network_over_wiener_median=float(np.median(net / wie)),
                           network_over_wiener_beta35_median=float(np.median(net / w35)),
                           frac_network_beats_wiener=float(np.mean(net < wie)), frac_network_beats_interp=float(np.mean(net < it)))
    return out


def plot_beta_map(rows, grid, out):
    regions = sorted({r["region"] for r in rows})
    fig, axes = plt.subplots(1, len(regions), figsize=(6.5 * len(regions), 5), squeeze=False)
    for ax, region in zip(axes[0], regions):
        rs, cs = region_slice(region)
        sub = grid[rs, cs]
        ext = [-180 + cs.start / 30, -180 + cs.stop / 30, -90 + rs.start / 30, -90 + rs.stop / 30]
        v = np.nanpercentile(np.abs(sub), 98)
        ax.imshow(sub, origin="lower", extent=ext, cmap="gray", vmin=-v, vmax=v, aspect="auto")
        pts = [r for r in rows if r["region"] == region]
        sc = ax.scatter([r["lon"] for r in pts], [r["lat"] for r in pts], c=[r["beta"] for r in pts], s=22, cmap="viridis", vmin=2.0, vmax=4.5, edgecolors="none")
        ax.set_title(f"{region.replace('_', ' ')}: fitted beta per land-survey tile", fontsize=10)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.colorbar(sc, ax=axes[0].tolist(), label="beta", shrink=0.8)
    fig.suptitle("grey: EMAG2 v3 anomaly at 4 km; tiles are 64 x 64 after resampling to square 3.7 km pixels", fontsize=9)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_summary(rows, summary, out):
    b = np.array([r["beta"] for r in rows]); net = np.array([r["mse_network"] for r in rows])
    wie = np.array([r["mse_wiener"] for r in rows]); fl = np.array([r["floor_predicted"] for r in rows]); it = np.array([r["mse_interp"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(b, bins=30, color="#4a7c7c"); ax[0].axvspan(2.5, 4.0, color="orange", alpha=0.15, label="range assumed by the study")
    ax[0].set_xlabel("fitted beta per tile"); ax[0].set_ylabel("tiles"); ax[0].legend(fontsize=8)
    ax[1].loglog(wie, net, ".", ms=4, alpha=0.6); lim = [min(wie.min(), net.min()) * 0.8, max(wie.max(), net.max()) * 1.2]
    ax[1].plot(lim, lim, "k--", lw=1); ax[1].set_xlabel("Gaussian estimator error, fitted beta [nT^2]"); ax[1].set_ylabel("network error [nT^2]")
    ax[1].set_title("per tile; below the line the network is better", fontsize=9)
    ax[2].loglog(fl, wie, ".", ms=4, alpha=0.6, label="Gaussian estimator"); ax[2].loglog(fl, it, ".", ms=4, alpha=0.4, label="interpolation")
    ax[2].plot(lim, lim, "k--", lw=1); ax[2].set_xlabel("floor predicted from the tile's fitted beta [nT^2]"); ax[2].set_ylabel("realised error [nT^2]")
    ax[2].set_title("above the line: the tile is less Gaussian or stationary than its spectrum says", fontsize=9); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_examples(examples, out):
    if not examples:
        return
    fig, ax = plt.subplots(len(examples), 6, figsize=(17, 2.9 * len(examples)), squeeze=False)
    for r, (clean, noisy, mask, net, w, it, rec) in enumerate(examples):
        v = clean.abs().max().item()
        obs = noisy.clone(); obs[mask] = float("nan")
        panels = [("EMAG2 tile (rescaled)", clean), ("observed 25%", obs), (f"network {rec['mse_network']:.2f}", net),
                  (f"Gaussian, beta {rec['beta']:.2f}: {rec['mse_wiener']:.2f}", w), (f"interpolation {rec['mse_interp']:.2f}", it)]
        for i, (name, img) in enumerate(panels):
            ax[r, i].imshow(img, cmap="RdBu_r", vmin=-v, vmax=v); ax[r, i].set_title(name, fontsize=8)
        res = (net - clean) * mask; vr = float(res[mask].abs().quantile(0.95))
        ax[r, 5].imshow(res, cmap="RdBu_r", vmin=-vr, vmax=vr); ax[r, 5].set_title(f"network residual on hidden pixels (+-{vr:.1f} nT)", fontsize=8)
        ax[r, 0].set_ylabel(f"{rec['region'].replace('_', ' ')}\n{rec['lat']:.1f}, {rec['lon']:.1f}", fontsize=8)
        for x in ax[r]:
            x.set_xticks([]); x.set_yticks([])
    fig.suptitle("EMAG2 4 km grid, land-survey tiles of 64 pixels (about 240 km), 75% hidden, 1 nT noise; errors are hidden-pixel MSE in nT^2 after rescaling", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def calibrate(n=64, sigma=1.0):
    """the same pipeline on seamless Gaussian tiles at the equivalent altitude: what the
    realised / predicted ratio and the fitted beta look like when the model is exactly right"""
    from .grf import sample_fields
    gen = torch.Generator().manual_seed(1)
    wiener = Wiener(n, sigma)
    out = {}
    for beta in [2.5, 3.0, 3.5, 4.0]:
        f = sample_fields(1, 512, beta, H_EQ, DX, 50.0, gen)[0].numpy().astype(np.float64)
        ratios, betas = [], []
        for i in range(0, 512, n):
            for j in range(0, 512, n):
                t = detrend(f[i:i + n, j:j + n]); b = fit_beta(t); betas.append(b)
                clean = torch.tensor(t, dtype=torch.float32); noisy = clean + sigma * torch.randn(clean.shape, generator=gen)
                mask = random_mask(1, n, 0.75, gen)[0]; hid = mask.flatten()
                pred, fl = wiener(noisy.double(), hid, b, float(clean.var()))
                full = noisy.clone().flatten(); full[hid] = torch.from_numpy(pred).float(); full = full.view(n, n)
                ratios.append(float(((full - clean) ** 2)[mask].mean()) / fl)
        out[str(beta)] = dict(beta_fitted_median=float(np.median(betas)), beta_q10=float(np.quantile(betas, 0.1)), beta_q90=float(np.quantile(betas, 0.9)),
                              ratio_median=float(np.median(ratios)), ratio_q10=float(np.quantile(ratios, 0.1)), ratio_q90=float(np.quantile(ratios, 0.9)))
        print(f"true beta {beta}: fitted {out[str(beta)]['beta_fitted_median']:.2f}; realised / predicted floor {out[str(beta)]['ratio_median']:.2f} "
              f"(q10 {out[str(beta)]['ratio_q10']:.2f}, q90 {out[str(beta)]['ratio_q90']:.2f})")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/emag2_calibration.json", "w"), indent=1)


def synthetic(run, n=64, sigma=1.0):
    """the identical pipeline on Gaussian tiles at the EMAG2 equivalent altitude with spectral
    slopes on and off the network's training value: separates the effect of an unfamiliar
    spectrum from the effect of non-Gaussian structure"""
    from .grf import sample_fields
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, c = load_model(run, device)
    train_hs = [float(x) for x in str(c["h"]).split(",")]
    h_in = torch.tensor([H_EQ / max(800.0, max(train_hs))], device=device)
    sigma0 = c["sigma0"]; wiener = Wiener(n, sigma)
    gen = torch.Generator().manual_seed(2)
    out = {}
    target_rms = math.sqrt(prior_variance(n, 3.5, H_EQ, DX, 50.0))
    for beta in [2.5, 3.0, 3.5, 4.0, 4.5]:
        f = sample_fields(1, 512, beta, H_EQ, DX, 50.0, gen)[0].numpy().astype(np.float64)
        ratios, nets, w35s, nonlins = [], [], [], []
        prev = None
        for i in range(0, 512, n):
            for j in range(0, 512, n):
                t = detrend(f[i:i + n, j:j + n]); t = t * (target_rms / t.std())
                clean = torch.tensor(t, dtype=torch.float32); noisy = clean + sigma * torch.randn(clean.shape, generator=gen)
                mask = random_mask(1, n, 0.75, gen)[0]; hid = mask.flatten()
                with torch.no_grad():
                    net = (model(observe(noisy[None].to(device), mask[None].to(device), sigma0), h_in)[0] * sigma0).cpu()
                    if prev is not None:
                        f_mix = (model(observe((0.5 * noisy + 0.5 * prev)[None].to(device), mask[None].to(device), sigma0), h_in)[0] * sigma0).cpu()
                        f2 = (model(observe(prev[None].to(device), mask[None].to(device), sigma0), h_in)[0] * sigma0).cpu()
                        lin = 0.5 * net + 0.5 * f2
                        nonlins.append(float((((f_mix - lin) ** 2 * mask).sum() / ((lin ** 2) * mask).sum()).sqrt()))
                    prev = noisy.clone()
                w, _ = wiener(noisy.double(), hid, 3.5, float(clean.var()))
                full = noisy.clone().flatten(); full[hid] = torch.from_numpy(w).float(); full = full.view(n, n)
                e_net = float(((net - clean) ** 2)[mask].mean()); e_w = float(((full - clean) ** 2)[mask].mean())
                nets.append(e_net); w35s.append(e_w); ratios.append(e_net / e_w)
        out[str(beta)] = dict(mse_network_median=float(np.median(nets)), mse_wiener_beta35_median=float(np.median(w35s)), ratio_median=float(np.median(ratios)),
                              nonlinearity_median=float(np.median(nonlins)))
        print(f"synthetic Gaussian tiles, beta {beta}, h/dx 1.08: network {np.median(nets):.2f}, Gaussian(3.5) {np.median(w35s):.2f}, ratio {np.median(ratios):.2f}, "
              f"mixing-test nonlinearity {np.median(nonlins):.3f}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/emag2_synthetic.json", "w"), indent=1)


def replot(grid_path="data/emag2_upcont.npy"):
    """regenerate the map and summary figures from results/emag2.json (the example-tile figure
    needs the reconstructions and is only written by `run`)"""
    em = json.load(open("results/emag2.json"))
    grid = np.load(grid_path)
    plot_beta_map(em["tiles"], grid, "figures/emag2_beta_map.png")
    plot_summary(em["tiles"], em["summary"], "figures/emag2_summary.png")
    print("rewrote figures/emag2_beta_map.png and figures/emag2_summary.png")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replot")
    pc = sub.add_parser("convert"); pc.add_argument("path"); pc.add_argument("--out", default="data/emag2_upcont.npy")
    sub.add_parser("calibrate")
    ps = sub.add_parser("synthetic"); ps.add_argument("--run", default="runs/vitxxl_b3.5_hmix_D131072_s0_noalt.json")
    pr = sub.add_parser("run")
    pr.add_argument("--grid", default="data/emag2_upcont.npy")
    pr.add_argument("--run", default="runs/vitxxl_b3.5_hmix_D131072_s0_noalt.json",
                    help="checkpoint; the altitude-mixture model without the altitude input generalises across h")
    pr.add_argument("--regions", default="australia,north_america")
    pr.add_argument("--n", type=int, default=64)
    pr.add_argument("--sigma", type=float, default=1.0)
    a = p.parse_args()
    if a.cmd == "convert":
        convert(a.path, a.out)
    elif a.cmd == "calibrate":
        calibrate()
    elif a.cmd == "synthetic":
        synthetic(a.run)
    elif a.cmd == "replot":
        replot()
    else:
        run(a)


if __name__ == "__main__":
    main()

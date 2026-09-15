"""The world: a magnetic anomaly map and the field it produces at flight altitude.

The map is either the Gawler survey grid (data/gawler_tmi.nc, 45 m pixels, flown at 60 m) read
by magscale.grid_eval.load_grid, or one synthetic map from the generator of Part F
(magscale.gen2.sample_fields_v2). Either way the grid is valid at one height h0 and the field
at any higher h comes from upward continuation by exp(-k (h - h0)) with reflect padding
(magscale.grid_eval.continue_upward), then bilinear sampling (magscale.magnav.bilinear).

Simplifications, also listed in magsim/README.md: the grid value is taken as the scalar
anomaly along the ambient direction, so the earth field vector is (B0 + anomaly) times a fixed
unit vector; heights below h0 are clamped to h0 (no downward continuation); continuation of a
draped survey is approximate over rough terrain.
"""
import math

import numpy as np

from problem1.checks import B0, INCL, DECL
from magscale.grid_eval import load_grid, continue_upward
from magscale.magnav import bilinear


class World:
    """A grid of anomaly values in nT at height h0_m, with cached continuations. State is the
    map and the cache, which is why this is a class and not a set of functions."""

    def __init__(self, grid_nT, dx_m, h0_m, incl_rad=INCL, decl_rad=DECL, name="grid"):
        self.grid_nT = np.asarray(grid_nT, dtype=np.float64)
        self.dx_m = float(dx_m)
        self.h0_m = float(h0_m)
        self.incl_rad, self.decl_rad = float(incl_rad), float(decl_rad)
        self.name = name
        self._grids = {}          # rounded height in m -> continued grid, nT
        self._gradients = {}      # rounded height in m -> (d/dx, d/dy) in nT/m

    @property
    def extent_m(self):
        """(x_max, y_max): the far corner of the grid in metres"""
        return ((self.grid_nT.shape[1] - 1) * self.dx_m, (self.grid_nT.shape[0] - 1) * self.dx_m)

    def _key(self, h_m):
        # heights are rounded to the metre so a flight at nominally constant altitude reuses
        # one transform; below h0 the grid is used as it is (downward continuation is unstable)
        return int(round(max(float(h_m), self.h0_m)))

    def grid_at(self, h_m):
        """the anomaly grid continued to height h_m above ground, nT"""
        k = self._key(h_m)
        if k not in self._grids:
            self._grids[k] = continue_upward(self.grid_nT, self.dx_m, k - self.h0_m).astype(np.float64)
        return self._grids[k]

    def gradient_at(self, h_m):
        """(d/dx, d/dy) of the continued grid in nT/m, central differences as magnav.crb uses"""
        k = self._key(h_m)
        if k not in self._gradients:
            g = self.grid_at(k)
            self._gradients[k] = (np.gradient(g, axis=1) / self.dx_m, np.gradient(g, axis=0) / self.dx_m)
        return self._gradients[k]

    def _pixels(self, x_m, y_m):
        return np.asarray(y_m, dtype=np.float64) / self.dx_m, np.asarray(x_m, dtype=np.float64) / self.dx_m

    def field(self, x_m, y_m, h_m):
        """anomaly in nT at (x, y, h); x, y scalars or arrays, h a scalar or an array of the
        same shape (samples are grouped by rounded height so each transform runs once)"""
        row, col = self._pixels(x_m, y_m)
        h = np.broadcast_to(np.asarray(h_m, dtype=np.float64), row.shape)
        out = np.empty(row.shape)
        keys = np.vectorize(self._key)(h) if h.ndim else np.array(self._key(h))
        for k in np.unique(keys):
            sel = keys == k
            out[sel] = bilinear(self.grid_at(k), row[sel], col[sel])
        return out if out.ndim else float(out)

    def gradient(self, x_m, y_m, h_m):
        """horizontal gradient (..., 2) in nT/m: d/dx (east) and d/dy (north)"""
        row, col = self._pixels(x_m, y_m)
        h = np.broadcast_to(np.asarray(h_m, dtype=np.float64), row.shape)
        out = np.empty(row.shape + (2,))
        keys = np.vectorize(self._key)(h) if h.ndim else np.array(self._key(h))
        for k in np.unique(keys):
            sel = keys == k
            gx, gy = self.gradient_at(k)
            out[sel, 0] = bilinear(gx, row[sel], col[sel])
            out[sel, 1] = bilinear(gy, row[sel], col[sel])
        return out

    def ambient_direction_ned(self):
        """unit vector of the ambient field in NED, from the inclination and declination"""
        ci, si = np.cos(self.incl_rad), np.sin(self.incl_rad)
        return np.array([ci * np.cos(self.decl_rad), ci * np.sin(self.decl_rad), si])

    def ambient_nT(self, x_m, y_m, h_m):
        """scalar earth field at the point: B0 plus the anomaly, nT"""
        return B0 + self.field(x_m, y_m, h_m)


# the ambient direction over the Gawler window (30 S, 135 E): inclination about -65 degrees,
# declination about +7 degrees. The synthetic world keeps the checks.py direction so the rank
# tests are the numerical checks' numbers; the Gawler rank is asserted separately.
GAWLER_INCL, GAWLER_DECL = math.radians(-65.0), math.radians(7.0)


def gawler(path="data/gawler_tmi.nc", h0_m=60.0, window=None, incl_rad=GAWLER_INCL, decl_rad=GAWLER_DECL):
    """the Gawler survey grid as flown (60 m clearance) under its local ambient direction;
    window = (row0, col0, n) cuts a square"""
    grid, dx = load_grid(path)
    if window is not None:
        r0, c0, n = window
        grid = grid[r0:r0 + n, c0:c0 + n]
    grid = np.where(np.isnan(grid), np.nanmean(grid), grid)
    return World(grid, dx, h0_m, incl_rad, decl_rad, name="gawler")


def synthetic(n=256, h0_m=60.0, dx_m=45.0, cfg=None, seed=0, sigma0=50.0):
    """one map from the generator of Part F at height h0_m. The default ranges are those of
    the released generator-v2 runs (runs/vitxxl_gen2_hmix_D0_s0_72k.json)."""
    import torch
    from magscale.gen2 import sample_fields_v2
    cfg = cfg or dict(beta_range=(2.0, 5.0), mod_range=(0.2, 0.9), aniso_max=2.5)
    gen = torch.Generator().manual_seed(seed)
    clean, _ = sample_fields_v2(1, n, h0_m, cfg, dx=dx_m, sigma0=sigma0, gen=gen)
    return World(clean[0].double().numpy(), dx_m, h0_m, name=f"synthetic_s{seed}")

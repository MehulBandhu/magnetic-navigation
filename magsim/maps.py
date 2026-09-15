"""The maps to navigate on, all for the same ground: the true grid at the flown altitude, and
three reconstructions of it from 25% survey lines (every fourth row observed, the others
hidden): the generator-trained network of Part F (runs/vitxxl_gen2_hmix_D0_s0_72k, loaded by
magscale.emag2.load_model), the Gaussian estimator with the training prior (beta 3.5,
magscale.emag2.Wiener), and linear interpolation between lines (magscale.magnav.line_interp).

The reconstructions run on overlapping 64-pixel tiles with the centre-weighted blend of
magscale.magnav.tile_apply. Each tile is detrended and scaled the way the Part G evaluation
does it (magscale.grid_eval.evaluate), except that the plane and the scale are fitted on the
observed pixels only, since a navigator has no others. part_g_tiles() runs the Part G
evaluation itself on the same window so that its per-tile errors can be checked against
results/grid_gawler.json, which is what tests/test_magsim.py does.
"""
import json
import math

import numpy as np
import torch

from magscale.emag2 import DX, Wiener, load_model
from magscale.floor import prior_variance
from magscale.grf import observe
from magscale.grid_eval import evaluate, block_mean
from magscale.magnav import survey_mask, line_interp, tile_apply

RUN = "runs/vitxxl_gen2_hmix_D0_s0_72k.json"                 # generator-trained, random masks (Part F, G)
RUN_LINES = "runs/vitxxl_b3.5_h200_D131072_s0_lines.json"      # trained on survey lines (the magnav study)
PATCH = 64
SPACING = 4            # every fourth row is a survey line: 25% observed
BETA_PRIOR = 3.5       # the training prior of the Gaussian estimator column in Part G


def plane(tile, obs):
    """least-squares plane through the observed pixels, (n, n)"""
    n = tile.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    A = np.stack([np.ones(n * n), x.ravel(), y.ravel()], 1)
    coef = np.linalg.lstsq(A[obs.ravel()], tile.ravel()[obs.ravel()], rcond=None)[0]
    return (A @ coef).reshape(n, n)


def survey(true_nT, sigma_nT=1.0, seed=0, spacing=SPACING):
    """the survey: noisy readings on every spacing-th row; returns (noisy, mask) with mask
    True where hidden, the convention of magscale.magnav"""
    rng = np.random.default_rng(seed)
    noisy = true_nT + sigma_nT * rng.standard_normal(true_nT.shape)
    return noisy, survey_mask(true_nT.shape[0], spacing)


def network_map(noisy_nT, mask, h_over_dx, sigma_nT=1.0, run=RUN, device="cpu"):
    """the network's reconstruction of the whole map, (n, n) nT"""
    model, c = load_model(run, device)
    h_eq = h_over_dx * DX                                  # the study's units: 100 m pixels
    train_hs = [float(x) for x in str(c["h"]).split(",")]
    h_in = torch.tensor([h_eq / max(800.0, max(train_hs))], device=device)
    sigma0 = c["sigma0"]
    target_rms = math.sqrt(prior_variance(PATCH, BETA_PRIOR, h_eq, DX, 50.0))

    @torch.no_grad()
    def fn_network(blk, m):
        trend, t, scale = prepare(blk, m, target_rms)
        x = observe(torch.tensor(t, dtype=torch.float32)[None], torch.from_numpy(m)[None], sigma0)
        return model(x.to(device), h_in)[0].cpu().numpy().astype(np.float64) * sigma0 / scale + trend

    return tile_apply(noisy_nT, mask, fn_network, PATCH)


def prepare(blk, m, target_rms):
    """detrend and scale a tile on its observed pixels, as the Part G evaluation does on the
    whole tile; returns (trend, scaled detrended tile, scale)"""
    obs = ~m
    trend = plane(blk, obs)
    t = blk - trend
    scale = target_rms / max(t[obs].std(), 1e-6)
    return trend, t * scale, scale


def wiener_map(noisy_nT, mask, h_over_dx, sigma_nT=1.0):
    """the Gaussian estimator with the training prior (beta 3.5), tile by tile"""
    h_eq = h_over_dx * DX
    target_rms = math.sqrt(prior_variance(PATCH, BETA_PRIOR, h_eq, DX, 50.0))
    wiener = Wiener(PATCH, sigma_nT, DX, h_eq)

    def fn_wiener(blk, m):
        trend, t, scale = prepare(blk, m, target_rms)
        hid = torch.from_numpy(m.flatten())
        w, _ = wiener(torch.tensor(t, dtype=torch.float64), hid, BETA_PRIOR, float(t[~m].var()))
        out = t.copy().flatten(); out[m.flatten()] = w
        return out.reshape(PATCH, PATCH) / scale + trend
    return tile_apply(noisy_nT, mask, fn_wiener, PATCH)


def reconstruct(noisy_nT, mask, h_over_dx, sigma_nT=1.0, device="cpu"):
    """the reconstructions of the whole map, dict name -> (n, n) nT: 'network' is the
    lines-trained checkpoint, the one the repository's navigation study used on survey lines;
    'network_gen2' is the generator-trained model of Part G, which was trained on random
    masks and is out of its distribution on lines (its error is reported, not hidden);
    'wiener' the Gaussian estimator; 'interp' linear interpolation between lines"""
    return dict(network=network_map(noisy_nT, mask, h_over_dx, sigma_nT, RUN_LINES, device),
                network_gen2=network_map(noisy_nT, mask, h_over_dx, sigma_nT, RUN, device),
                wiener=wiener_map(noisy_nT, mask, h_over_dx, sigma_nT),
                interp=line_interp(noisy_nT, mask))


def scaled_grid(world, h_m):
    """the world's grid at height h_m with the pixel size coarsened so that h / dx stays what
    it was when flown, the scale axis of Part G (magscale.grid_eval.main): continued up by
    h - h0, then block-averaged by round(h / h0). Returns (grid, dx_m, factor)."""
    f = max(1, int(round(h_m / world.h0_m)))
    return block_mean(world.grid_at(h_m).astype(np.float32), f).astype(np.float64), world.dx_m * f, f


def four_maps(world, row0, col0, n=256, sigma_nT=1.0, seed=0, h_m=None):
    """the four maps of Part G for an n x n window of the world (plus the generator-trained
    network's map), at the flown height or at h_m over the correspondingly coarsened grid;
    row0, col0, n are in the pixels of that grid. Returns dict with 'true', 'network',
    'network_gen2', 'wiener', 'interp' (n, n) nT, the survey mask, the pixel size, the window
    origin in metres, the height and the rms error of each reconstruction on hidden pixels"""
    h_m = world.h0_m if h_m is None else h_m
    g, dx, f = scaled_grid(world, h_m)
    grid = g[row0:row0 + n, col0:col0 + n]
    noisy, mask = survey(grid, sigma_nT, seed)
    maps = dict(true=grid, **reconstruct(noisy, mask, h_m / dx, sigma_nT))
    err = {k: float(np.sqrt(np.mean((v - grid)[mask] ** 2))) for k, v in maps.items() if k != "true"}
    return dict(maps=maps, mask=mask, dx_m=dx, origin_m=(col0 * dx, row0 * dx), rms_error_nT=err, n=n, h_m=h_m, factor=f)


def part_g_tiles(world, max_tiles=50, n=PATCH, sigma_nT=1.0, run=RUN, device="cpu"):
    """the Part G evaluation (magscale.grid_eval.evaluate) on this world's grid, the same
    tiles in the same order with the same mask and noise generator seed as the committed run,
    so the per-tile numbers can be compared with results/grid_gawler.json"""
    model, c = load_model(run, device)
    gen = torch.Generator().manual_seed(0)
    return evaluate(world.grid_at(world.h0_m).astype(np.float32), world.h0_m / world.dx_m, model, c, n, sigma_nT, device, max_tiles, gen)


def committed_tiles(path="results/grid_gawler.json", altitude="60"):
    return json.load(open(path))["per_altitude"][altitude]["tiles"]

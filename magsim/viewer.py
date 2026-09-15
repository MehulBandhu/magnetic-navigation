"""Rerun logging for the demo: the map as a height-coloured mesh, the vehicle as a small body
with its attitude, the flown path, the particle cloud, and the time series (reading, platform
field, calibration residual and rank, corrector residual, position error and bound, the
current tile's floor next to the network's error). Everything is logged from Python on the
timeline "flight" in seconds; the recording is saved to an .rrd file or streamed to a viewer.

World frame: x east, y north, z up, metres. The map is drawn with its value as height, so the
anomaly reads as terrain; VERTICAL_M_PER_NT sets the exaggeration.
"""
import math

import numpy as np

from problem1.checks import rot

VERTICAL_M_PER_NT = 0.1
NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _rr():
    import rerun as rr
    return rr


def start(app="magsim", record=None, spawn=False):
    """initialise the recording; record: path of the .rrd to write; spawn: open a viewer"""
    rr = _rr()
    rr.init(app, spawn=spawn)
    if record:
        rr.save(record)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    return rr


def blueprint():
    """layout: the 3D world on the left, the time series on the right"""
    rr = _rr()
    import rerun.blueprint as rrb
    plots = [rrb.TimeSeriesView(origin=f"plots/{name}", name=title) for name, title in
             [("reading", "reading minus map, nT"), ("calibration", "calibration residual, nT"), ("rank", "regressor rank"),
              ("corrector", "residual after each estimator, nT"), ("position", "position error and bound, m"), ("floor", "tile floor and network error, nT^2")]]
    return rrb.Blueprint(rrb.Horizontal(rrb.Vertical(rrb.Spatial3DView(origin="world", name="Gawler survey grid"), rrb.TextDocumentView(origin="text", name="what is shown"), row_shares=[4, 1]),
                                        rrb.Grid(*plots, grid_columns=2), column_shares=[3, 2]))


def colours(values, vmin=None, vmax=None):
    """(n, 4) uint8 RGBA, a blue-white-red ramp centred on zero"""
    v = np.asarray(values, dtype=np.float64)
    lim = max(abs(vmin if vmin is not None else -np.nanstd(v) * 2.5), abs(vmax if vmax is not None else np.nanstd(v) * 2.5), 1e-9)
    x = np.clip(v / lim, -1, 1)
    r = np.where(x > 0, 255, 255 * (1 + x))
    b = np.where(x < 0, 255, 255 * (1 - x))
    g = 255 * (1 - np.abs(x))
    return np.stack([r, g, b, np.full_like(r, 255)], 1).astype(np.uint8)


def log_map(entity, map_nT, dx_m, origin_m, step=1, z_offset_m=0.0):
    """the map as a mesh, height = value * VERTICAL_M_PER_NT above z_offset"""
    rr = _rr()
    m = np.asarray(map_nT, dtype=np.float64)[::step, ::step]
    ny, nx = m.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    pos = np.stack([origin_m[0] + xs.ravel() * dx_m * step, origin_m[1] + ys.ravel() * dx_m * step, z_offset_m + m.ravel() * VERTICAL_M_PER_NT], 1)
    i = (ys[:-1, :-1] * nx + xs[:-1, :-1]).ravel()
    tri = np.concatenate([np.stack([i, i + 1, i + nx], 1), np.stack([i + 1, i + nx + 1, i + nx], 1)])
    rr.log(entity, rr.Mesh3D(vertex_positions=pos.astype(np.float32), triangle_indices=tri.astype(np.uint32), vertex_colors=colours(m.ravel())), static=True)


def set_time(t_s):
    _rr().set_time("flight", duration=float(t_s))


def log_vehicle(entity, x_m, y_m, h_m, roll_rad, pitch_rad, yaw_rad, map_value_nT=0.0, size_m=60.0):
    """the vehicle body (logged once as arrows along its axes) placed by a transform; the
    height is the map's drawn height at the vehicle plus its clearance"""
    rr = _rr()
    R = NED_TO_ENU @ rot(roll_rad, pitch_rad, yaw_rad)          # body (forward, right, down) -> ENU
    z = map_value_nT * VERTICAL_M_PER_NT + h_m
    rr.log(entity, rr.Transform3D(translation=[float(x_m), float(y_m), float(z)], mat3x3=R.astype(np.float32)))
    rr.log(entity + "/body", rr.Arrows3D(vectors=[[size_m, 0, 0], [0, 0.5 * size_m, 0], [0, 0, -0.3 * size_m]],
                                         colors=[[230, 60, 60], [60, 200, 60], [60, 60, 230]], radii=0.04 * size_m), static=True)


def log_path(entity, xs_m, ys_m, zs_m, colour=(30, 30, 30)):
    rr = _rr()
    pts = np.stack([xs_m, ys_m, zs_m], 1).astype(np.float32)
    rr.log(entity, rr.LineStrips3D([pts], colors=[colour], radii=4.0))


def log_particles(entity, pts_m, z_m, colour=(255, 140, 0)):
    rr = _rr()
    p = np.column_stack([pts_m, np.full(len(pts_m), z_m)]).astype(np.float32)
    rr.log(entity, rr.Points3D(p, colors=[colour], radii=6.0))


def log_scalar(entity, value):
    _rr().log(entity, _rr().Scalars(float(value)))


def series_style(entity, name, colour):
    rr = _rr()
    rr.log(entity, rr.SeriesLines(colors=[colour], names=[name]), static=True)


def log_text(entity, text):
    _rr().log(entity, _rr().TextDocument(text))

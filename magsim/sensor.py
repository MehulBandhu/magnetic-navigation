"""The scalar magnetometer: |B_earth + B_platform| plus drift and noise.

The earth field in the body frame is (B0 + anomaly) u, with u the ambient unit vector rotated
by the attitude (problem1.checks.direction_cosines). The reading is the magnitude of the exact
vector sum (problem1.edge_cases.exact_scalar), so the first-order Tolles-Lawson model is an
approximation the calibration has to live with; the dropped term is |Bp_perp|^2 / 2B.
"""
import numpy as np

from problem1.checks import B0
from problem1.edge_cases import exact_scalar

from .aircraft import body_direction_cosines
from .platform import interference, first_order_nT


def read(world, flight, vehicle=None, noise_nT=0.1, rng=None):
    """simulate the magnetometer along a flight. Returns a dict of (T,) arrays: reading_nT,
    map_nT (the anomaly a perfect sensor would read), ambient_nT (B0 + map), first_order_nT
    (ambient plus u . Bp, the linear model's reading), plus u and udot (T, 3) and, when a
    vehicle is given, its interference dict under 'platform'. With vehicle None the platform
    field is zero and reading_nT - B0 is the map value up to the noise."""
    # unseeded by default so repeated calls give independent noise; pass an rng to reproduce
    rng = rng if rng is not None else np.random.default_rng()
    u, udot = body_direction_cosines(flight, world.incl_rad, world.decl_rad)
    anomaly = world.field(flight["x_m"], flight["y_m"], flight["h_m"])
    ambient = B0 + anomaly
    out = dict(map_nT=anomaly, ambient_nT=ambient, u=u, udot=udot)
    if vehicle is None:
        Bp = np.zeros_like(u)
        offset = 0.0
    else:
        out["platform"] = interference(vehicle, flight, u, udot, ambient, rng)
        Bp = out["platform"]["Bp_nT"]
        offset = out["platform"]["drift_nT"]
    exact = exact_scalar(u, Bp, B=ambient)
    out["exact_nT"] = exact
    out["first_order_nT"] = ambient + first_order_nT(u, Bp)
    out["reading_nT"] = exact + offset + noise_nT * rng.normal(size=len(exact))
    return out

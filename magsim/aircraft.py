"""Kinematics of the scripted flight. No dynamics: attitude is prescribed, position follows
the heading at constant speed, height is constant.

A manoeuvre is a list of segments (duration_s, roll_amp_rad, pitch_amp_rad,
heading_rate_rad_s) at 10 Hz. Within a segment roll and pitch are sinusoids at the
frequencies of problem1.checks.manoeuvre (0.10 and 0.137 Hz) in global time, and the rates
are analytic. That matters: the Tolles-Lawson rank of 17 rests on u . du/dt = 0 holding
exactly, and finite-difference rates break it (checks.py demonstrates the leak).
"""
import math

import numpy as np

from problem1.checks import direction_cosines

ROLL_HZ, PITCH_HZ = 0.10, 0.137


def fly(segments, x0_m=0.0, y0_m=0.0, h_m=60.0, speed_mps=30.0, heading0_rad=0.0, fs_hz=10.0,
        roll_hz=ROLL_HZ, pitch_hz=PITCH_HZ):
    """returns a dict of (T,) arrays: t_s, x_m (east), y_m (north), h_m, roll_rad, pitch_rad,
    yaw_rad (heading from north towards east), roll_rate, pitch_rate, yaw_rate (rad/s), and
    the scalar dt_s. Position is integrated with the trapezoid rule along the heading."""
    dt = 1.0 / fs_hz
    t, roll, pitch, yaw, rr, pr, yr = [], [], [], [], [], [], []
    t0, psi = 0.0, float(heading0_rad)
    for dur, ra, pa, hr in segments:
        ts = t0 + np.arange(0, dur - 1e-9, dt)
        wr, wp = 2 * math.pi * roll_hz, 2 * math.pi * pitch_hz
        roll.append(ra * np.sin(wr * ts)); rr.append(ra * wr * np.cos(wr * ts))
        pitch.append(pa * np.sin(wp * ts)); pr.append(pa * wp * np.cos(wp * ts))
        yaw.append(psi + hr * (ts - t0)); yr.append(np.full_like(ts, hr))
        t.append(ts)
        psi += hr * dur; t0 += dur
    t, roll, pitch, yaw, rr, pr, yr = [np.concatenate(v) for v in (t, roll, pitch, yaw, rr, pr, yr)]
    vx, vy = speed_mps * np.sin(yaw), speed_mps * np.cos(yaw)
    x = x0_m + np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dt)])
    y = y0_m + np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dt)])
    return dict(t_s=t, x_m=x, y_m=y, h_m=np.full_like(t, float(h_m)), roll_rad=roll, pitch_rad=pitch,
                yaw_rad=yaw, roll_rate=rr, pitch_rate=pr, yaw_rate=yr, dt_s=dt)


def calibration_box(leg_s=60.0, turn_s=10.0, roll_amp_rad=math.radians(15), pitch_amp_rad=0.7 * math.radians(15),
                    bank_rad=math.radians(25)):
    """four excited legs on successive headings joined by 90 degree turns: the classic
    calibration pattern of problem1.edge_cases.box_pattern, as a segment list"""
    segs = []
    for _ in range(4):
        segs.append((leg_s, roll_amp_rad, pitch_amp_rad, 0.0))
        segs.append((turn_s, bank_rad, 0.0, 0.5 * math.pi / turn_s))
    return segs


def straight_leg(duration_s=60.0):
    """straight and level: constant attitude, zero rates"""
    return [(duration_s, 0.0, 0.0, 0.0)]


def body_direction_cosines(flight, incl_rad, decl_rad):
    """u (T, 3), the ambient unit vector in the body frame, and its exact derivative udot
    (T, 3) in 1/s, by problem1.checks.direction_cosines from the flight's attitude and rates"""
    ang = [flight["roll_rad"], flight["pitch_rad"], flight["yaw_rad"]]
    rate = [flight["roll_rate"], flight["pitch_rate"], flight["yaw_rate"]]
    return direction_cosines(flight["t_s"], ang, rate, incl=incl_rad, decl=decl_rad)


def imu(flight, rng, noise_deg=0.1, bias_deg=0.2, gyro_noise_deg_s=0.05, bias_walk_deg_per_sqrt_min=0.05):
    """the attitude an estimator would actually have: the true angles plus white noise of
    noise_deg and a slow bias (a random constant of bias_deg per axis plus a random walk), the
    rates plus gyro white noise. Returns a new flight dict; the true one is untouched. The
    simulator itself always knows the true attitude; this is an option for the estimator chain."""
    T = len(flight["t_s"])
    dt = flight["dt_s"]
    out = dict(flight)
    for ang, rate in (("roll_rad", "roll_rate"), ("pitch_rad", "pitch_rate"), ("yaw_rad", "yaw_rate")):
        bias = math.radians(bias_deg) * rng.normal() + np.cumsum(math.radians(bias_walk_deg_per_sqrt_min) * math.sqrt(dt / 60.0) * rng.normal(size=T))
        out[ang] = flight[ang] + bias + math.radians(noise_deg) * rng.normal(size=T)
        out[rate] = flight[rate] + math.radians(gyro_noise_deg_s) * rng.normal(size=T)
    return out

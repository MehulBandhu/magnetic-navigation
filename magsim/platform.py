"""The platform field: what the vehicle adds to the field at its own magnetometer.

Two parts. The rigid-body part is the 18-term Tolles-Lawson model of problem1 (permanent a,
induced M, eddy C) with the eddy time constant of problem1.edge_cases, computed by
edge_cases.platform_field. The rest is invented for this demonstration and calibrated against
nothing real: motor current loops at fixed offsets with random orientation, a battery dipole,
servo current pulses on a fixed-wing, and a slow thermal drift. Their magnitudes follow the
three platform classes of edge_cases.PLATFORMS (about 460, 930 and 2700 nT of rigid-body
field, and a motor term of 400 nT for the multirotor) so the numbers in the demo line up with
the tables in problem1/. The second-order term of the scalar reading is not modelled here; it
appears by itself in sensor.py, which takes the magnitude of the exact vector sum.

A vehicle is a plain dict of parameters. preset() gives the three classes with fixed seeds,
custom() draws a random vehicle so the corrector can be trained platform-agnostic.

Telemetry is what the vehicle logs anyway at 10 Hz: per-motor current in A and RPM (padded to
8 motors), battery voltage in V and current in A, throttle in [0, 1], a binary servo flag.
The drift is not observable from telemetry.
"""
import math

import numpy as np

from problem1.checks import B0
from problem1.edge_cases import PLATFORMS, platform_field, lag

MAX_MOTORS = 8

# class name here -> (row of edge_cases.PLATFORMS, motor count, full-throttle current per
# motor in A, battery field at full current in nT, servos)
CLASSES = {
    "survey_fixed_wing": ("survey fixed-wing", 1, 30.0, 10.0, True),
    "helicopter": ("helicopter", 1, 80.0, 50.0, False),
    "multirotor": ("multirotor drone", 6, 25.0, 150.0, False),
}


def _unit(v):
    return v / np.linalg.norm(v)


def _dipole_direction(rng):
    """direction at the sensor of the field of a dipole with random moment at a random offset
    0.2 to 1 m away; only the direction is kept, the magnitude is set by the class"""
    r = _unit(rng.normal(size=3)) * rng.uniform(0.2, 1.0)
    m = _unit(rng.normal(size=3))
    rhat = r / np.linalg.norm(r)
    return _unit(3 * (m @ rhat) * rhat - m)


def _draw(rng, platform_class, perm_nT, ind, eddy_s, tau_s, motor_nT, n_motors, i_max_A, battery_nT,
          servos, drift_nT_per_min, name):
    a = rng.normal(size=3); a = perm_nT * a / np.linalg.norm(a)
    M = rng.normal(size=(3, 3)) * ind
    C = rng.normal(size=(3, 3)) * eddy_s
    # each motor's field at full current is its share of the class motor term, with a spread
    share = motor_nT / max(n_motors, 1) * rng.uniform(0.7, 1.3, n_motors)
    motor_gain = np.stack([_dipole_direction(rng) * s / i_max_A for s in share]) if n_motors else np.zeros((0, 3))
    return dict(
        name=name, platform_class=platform_class,
        a_nT=a, M=M, C_s=C, tau_s=float(tau_s),
        n_motors=int(n_motors), motor_gain_nT_per_A=motor_gain, motor_max_A=float(i_max_A),
        pwm_hz=rng.uniform(50.0, 400.0, n_motors), motor_imbalance=rng.uniform(0.9, 1.1, n_motors),
        rpm_max=float(rng.uniform(4000.0, 12000.0)),
        battery_gain_nT_per_A=_dipole_direction(rng) * battery_nT / max(n_motors * i_max_A, 1.0),
        battery_V0=float(rng.uniform(22.0, 25.2)), battery_R_ohm=float(rng.uniform(0.01, 0.04)),
        battery_sag_V_per_s=float(rng.uniform(1e-4, 5e-4)), avionics_A=float(rng.uniform(0.5, 2.0)),
        servos=bool(servos), servo_gain_nT_per_A=_dipole_direction(rng) * rng.uniform(2.0, 8.0),
        servo_pulse_A=float(rng.uniform(1.0, 4.0)), servo_threshold_rad_s=float(rng.uniform(0.08, 0.15)),
        servo_tau_s=0.3,
        drift_nT_per_min=float(drift_nT_per_min),
    )


def preset(name, seed=0):
    """survey_fixed_wing, helicopter or multirotor with the coefficient scales of
    edge_cases.PLATFORMS (the motor term is zero for the first two there, and stays zero)"""
    row, n_motors, i_max, batt, servos = CLASSES[name]
    perm, ind, eddy, tau, motor = PLATFORMS[row]
    rng = np.random.default_rng(seed)
    return _draw(rng, name, perm, ind, eddy, tau, motor, n_motors, i_max, batt, servos,
                 rng.uniform(1.0, 5.0), f"{name}_s{seed}")


def custom(seed, scale=1.0, platform_class=None, extras=True):
    """a random vehicle: class drawn at random unless given, rigid-body magnitudes scaled by
    `scale` and a factor in [0.5, 1.5] about the class values, tau in [0.02, 1] s, 4 to 8
    motors on a multirotor, a small motor term even on the fixed-wing and helicopter. With
    extras False the motor, battery and servo fields are zero (the rigid body and the drift
    remain), the group the corrector table reports separately."""
    rng = np.random.default_rng(seed)
    platform_class = platform_class or list(CLASSES)[rng.integers(len(CLASSES))]
    row, n_motors, i_max, batt, servos = CLASSES[platform_class]
    perm, ind, eddy, _, motor = PLATFORMS[row]
    f = lambda: scale * rng.uniform(0.5, 1.5)
    if platform_class == "multirotor":
        n_motors = int(rng.integers(4, MAX_MOTORS + 1))
        motor = motor * f()
    else:
        motor = {"survey_fixed_wing": 50.0, "helicopter": 100.0}[platform_class] * rng.uniform(0.0, 1.0) * scale
    if not extras:
        motor, batt, servos = 0.0, 0.0, False
    v = _draw(rng, platform_class, perm * f(), ind * f(), eddy * f(), rng.uniform(0.02, 1.0), motor,
              n_motors, i_max * rng.uniform(0.7, 1.3), batt * f(), servos, rng.uniform(1.0, 5.0),
              f"custom_s{seed}")
    v["extras"] = bool(extras)
    return v


# ------------------------------------------------------------------ components
def rigid_body_field(vehicle, u, udot, dt_s, B_nT=B0):
    """(T, 3) nT: permanent, induced and lagged eddy terms, by edge_cases.platform_field"""
    return platform_field(u, udot, dt_s, vehicle["a_nT"], vehicle["M"], vehicle["C_s"], vehicle["tau_s"], B=B_nT)


def throttle_profile(vehicle, flight, rng):
    """(T,) in [0.2, 1]: the random walk of edge_cases.platforms, plus a lean term on a
    multirotor (it pitches to fly, and pitching costs thrust)"""
    T = len(flight["t_s"])
    walk = np.cumsum(rng.normal(0, 1, T)) / math.sqrt(T)
    if vehicle["platform_class"] == "multirotor":
        th = 0.5 + 0.3 * walk + 0.2 * np.abs(flight["pitch_rad"]) / math.radians(15)
    else:
        th = 0.65 + 0.08 * walk
    return np.clip(th, 0.2, 1.0)


def motor_currents(vehicle, throttle, t_s, rng):
    """(T, n_motors) A and (T, n_motors) RPM. Current goes as throttle^1.5 (propeller power)
    with a few percent of PWM ripple at each motor's own frequency; at 10 Hz the ripple is
    aliased, which is how a logged current looks. RPM goes as sqrt(throttle) with noise."""
    n = vehicle["n_motors"]
    base = vehicle["motor_max_A"] * throttle[:, None] ** 1.5 * vehicle["motor_imbalance"][None, :]
    ripple = 1 + 0.03 * np.sin(2 * math.pi * vehicle["pwm_hz"][None, :] * t_s[:, None] + rng.uniform(0, 2 * math.pi, n))
    current = base * ripple + 0.02 * vehicle["motor_max_A"] * rng.normal(size=(len(t_s), n))
    rpm = vehicle["rpm_max"] * np.sqrt(throttle)[:, None] * vehicle["motor_imbalance"][None, :] + 20.0 * rng.normal(size=(len(t_s), n))
    return np.maximum(current, 0.0), np.maximum(rpm, 0.0)


def battery(vehicle, motor_current_A, t_s):
    """total current (T,) A and terminal voltage (T,) V: open-circuit voltage minus the
    resistive drop minus a slow sag as the pack discharges"""
    i_total = motor_current_A.sum(1) + vehicle["avionics_A"]
    v = vehicle["battery_V0"] - vehicle["battery_R_ohm"] * i_total - vehicle["battery_sag_V_per_s"] * t_s
    return i_total, v


def servo_pulses(vehicle, flight, rng):
    """(T,) int flag and (T,) A: a control surface moves when the commanded rate is large,
    drawing a current pulse that decays with the servo time constant (edge_cases.lag)"""
    T = len(flight["t_s"])
    if not vehicle["servos"]:
        return np.zeros(T, dtype=int), np.zeros(T)
    moving = (np.abs(flight["roll_rate"]) > vehicle["servo_threshold_rad_s"]) | (np.abs(flight["pitch_rate"]) > vehicle["servo_threshold_rad_s"])
    flag = moving.astype(int)
    current = lag(vehicle["servo_pulse_A"] * flag.astype(float), vehicle["servo_tau_s"], flight["dt_s"])
    return flag, current


def drift(vehicle, t_s, rng):
    """(T,) nT: thermal offset as a random walk of drift_nT_per_min per root minute"""
    dt = t_s[1] - t_s[0] if len(t_s) > 1 else 0.1
    step = vehicle["drift_nT_per_min"] * math.sqrt(dt / 60.0)
    return np.cumsum(step * rng.normal(size=len(t_s)))


def interference(vehicle, flight, u, udot, B_nT=B0, rng=None):
    """everything the vehicle adds. Returns a dict with Bp_nT (T, 3) in the body frame and its
    parts rigid_nT, motor_nT, battery_nT, servo_nT (T, 3), drift_nT (T,) which is a scalar
    sensor offset and not part of the vector, and telemetry (padded to MAX_MOTORS)."""
    rng = rng if rng is not None else np.random.default_rng()
    T = len(flight["t_s"])
    rigid = rigid_body_field(vehicle, u, udot, flight["dt_s"], B_nT)
    throttle = throttle_profile(vehicle, flight, rng)
    i_motor, rpm = motor_currents(vehicle, throttle, flight["t_s"], rng)
    motor = i_motor @ vehicle["motor_gain_nT_per_A"]
    i_batt, v_batt = battery(vehicle, i_motor, flight["t_s"])
    batt = i_batt[:, None] * vehicle["battery_gain_nT_per_A"][None, :]
    flag, i_servo = servo_pulses(vehicle, flight, rng)
    servo = i_servo[:, None] * vehicle["servo_gain_nT_per_A"][None, :]
    pad = lambda x: np.pad(x, ((0, 0), (0, MAX_MOTORS - x.shape[1])))
    telemetry = dict(motor_current_A=pad(i_motor), motor_rpm=pad(rpm), battery_voltage_V=v_batt,
                     battery_current_A=i_batt, throttle=throttle, servo=flag)
    return dict(Bp_nT=rigid + motor + batt + servo, rigid_nT=rigid, motor_nT=motor, battery_nT=batt,
                servo_nT=servo, drift_nT=drift(vehicle, flight["t_s"], rng), telemetry=telemetry)


# ------------------------------------------------------------------ the two orders
def first_order_nT(u, Bp):
    """(T,) nT: the projection u . Bp, what the 18-term linear model describes"""
    return (u * Bp).sum(1)


def second_order_nT(u, Bp, B_nT=B0):
    """(T,) nT: |Bp_perp|^2 / 2B, the term the linear model drops (Problem 1A)"""
    return (np.sum(Bp ** 2, axis=1) - first_order_nT(u, Bp) ** 2) / (2 * B_nT)

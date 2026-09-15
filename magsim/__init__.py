"""Flight simulator for magnetic navigation, built on the results of this repository.

Three layers, described in SIM_SPEC.md. The world (world.py) is a magnetic anomaly map, real
or synthetic, continued to flight altitude. The vehicle is a scripted flight (aircraft.py), an
invented platform interference model (platform.py) and a scalar magnetometer that reads the
magnitude of the exact vector sum (sensor.py). The estimator chain (calibrate.py, corrector.py,
navigator.py) and the demo (scenario.py, viewer.py, run_demo.py) come in later stages.

Conventions: x east and y north in metres from the grid's (0, 0) pixel, h height above ground
in metres, roll, pitch, yaw in radians (ZYX, body to NED, problem1.checks.rot), field in nT,
time in seconds at 10 Hz. Every array carries its unit in its name or docstring.
"""

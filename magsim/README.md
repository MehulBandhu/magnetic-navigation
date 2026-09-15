# Flight simulator

A demonstration of the results in this repository: a scripted flight over a real magnetic
map with a magnetometer that reads what the vehicle does to the field, followed by the
estimator chain (Tolles-Lawson calibration, the lag model, a learned corrector for what the
physics leaves, a particle filter on maps reconstructed by the repository's network).
SIM_SPEC.md is the specification.

## What it simulates

- The world (`world.py`): the Gawler survey grid (data/gawler_tmi.nc, 45 m pixels, flown at
  60 m, read by `magscale.grid_eval.load_grid`) under its local ambient direction, or one map
  from the Part F generator (`magscale.gen2.sample_fields_v2`) under the direction of
  `problem1/checks.py`; continued to flight altitude by `magscale.grid_eval.continue_upward`
  and sampled bilinearly with `magscale.magnav.bilinear`, with the horizontal gradient for
  the Cramer-Rao bound.
- The aircraft (`aircraft.py`): position, attitude and analytic rates from a list of segments
  (duration, roll amplitude, pitch amplitude, heading rate) at 10 Hz, with the roll and pitch
  frequencies of `problem1.checks.manoeuvre`. `calibration_box` is the four-heading pattern
  of `problem1.edge_cases.box_pattern`. `imu` gives an estimator the attitude with 0.1
  degree white noise, a bias of a few tenths of a degree with a slow walk, and gyro noise.
- The platform (`platform.py`): the 18-term Tolles-Lawson field with an eddy time constant,
  by `problem1.edge_cases.platform_field`, plus motors, battery, servos and drift, with the
  telemetry a vehicle logs anyway (per-motor current and RPM, battery voltage and current,
  throttle, a servo flag).
- The sensor (`sensor.py`): the magnitude of the exact vector sum
  (`problem1.edge_cases.exact_scalar`) plus drift and white noise, so the second-order term
  |Bp_perp|^2 / 2B of Problem 1A is in the data.
- The calibration (`calibrate.py`): the regressor of `problem1.checks` built row by row, the
  fit refreshed every second, the numerical rank so far; band-passed variants by demeaning
  (as the numerical checks) or by a first-order high-pass (as the estimator chain uses, so
  the thermal drift stays out of the fit).
- The corrector (`corrector.py`, `train_corrector.py`): a causal 1-D convolutional network
  under 200k parameters that predicts the residual the linear model or the lag model leaves,
  from attitude, rates, the Tolles-Lawson regressors, the eddy regressors filtered at five
  time constants, the telemetry, and the last 20 s of the residual with the final 2 s
  blanked. Trained on hundreds of random vehicles, evaluated on held-out ones against the
  Gauss-Newton and lag baselines of `problem1/edge_cases.py`.
- The maps (`maps.py`): for one window of the Gawler grid, the true map and three
  reconstructions from 25% survey lines by the Part G code path: the network, the Gaussian
  estimator with the training prior (`magscale.emag2.Wiener`), and interpolation between
  lines (`magscale.magnav.line_interp`). `part_g_tiles` runs `magscale.grid_eval.evaluate`
  on the same grid, and the test asserts its per-tile numbers are those of
  results/grid_gawler.json.
- The navigator (`navigator.py`): a particle filter over position with a Rao-Blackwellised
  offset state, and the Cramer-Rao bound of Problem 1C from the map gradient along the
  track (the quantities of `magscale.magnav.crb`), with the offset as a third unknown.
- The viewer (`viewer.py`) logs to Rerun; the scenario (`scenario.py`) is the scripted
  four-part flight; `run_demo.py` records it.

## Conventions

x east and y north in metres from the grid's (0, 0) pixel; h height above ground in metres;
roll, pitch, yaw in radians, ZYX, body to NED (`problem1.checks.rot`); yaw is the heading from
north towards east; field in nT; time in seconds at 10 Hz.

## How to run

    python -m pytest -q tests/test_magsim.py
    python -m magsim.run_demo --map gawler --scenario full --record demo.rrd
    rerun demo.rrd

The demo needs data/gawler_tmi.nc (local), the two checkpoints runs/vitxxl_gen2_hmix_D0_s0_72k
and runs/vitxxl_b3.5_h200_D131072_s0_lines, and runs/corrector.pt. `--map synthetic` runs
the first two parts on a generated map; `--no-imu` gives the estimator chain the true
attitude. Everything runs on a laptop CPU in a few minutes.

## The corrector

Every vehicle flies a calibration box; the linear model and the lag model (tau scanned as in
`edge_cases.py`) are fitted there, high-passed at 10 s, and applied to the vehicle's other
flights. The corrector is trained on the residuals of 300 random vehicles and evaluated on
45 it has never seen. The baselines are fitted on the same box: Gauss-Newton on the exact
model, the lag model, both together, and Gauss-Newton with the battery current as an extra
regressor, the physics with telemetry. The metric is the residual rms per flight after
removing the flight's mean (a bias the navigator absorbs) and the true thermal drift (no
method can see it; its rms is a column of its own). The loss is scale-free per vehicle (the
error divided by the rms of the flight's linear residual, then log-cosh), so the helicopter's
rows do not dominate the fixed-wing's.

The tables below are rendered from results/corrector.json and results/magsim_nav.json by
`python scripts/render_tables.py`; the demo readouts come from results/magsim_demo.json.
Until the cluster run of scripts/sweeps/corrector.txt is fetched, the corrector table is
that of a 4000-step run on the Mac and stands in for it.

<!-- tables:start -->

### Residual after each estimator on held-out vehicles (45 vehicles, 300 in training, 156993 parameters, window 200 steps with a 20-step gap, attitude known; rms nT per flight after removing the flight mean and the thermal drift, root mean square over flights)

| platform class | motor, battery, servo fields | flights | linear TL | Gauss-Newton | lag model | GN + lag | GN + lag + battery current | corrector after linear | corrector after lag model | drift rms | IMU floor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| survey fixed wing | off | 15 | 5.92 | 5.91 | 0.74 | 0.71 | 0.73 | 2.76 | 1.93 | 2.00 | 0.00 |
| survey fixed wing | on | 30 | 9.09 | 9.02 | 1.82 | 1.81 | 1.76 | 3.34 | 1.87 | 1.60 | 0.00 |
| helicopter | off | 6 | 48.24 | 48.01 | 3.16 | 2.45 | 3.07 | 15.88 | 2.39 | 2.25 | 0.00 |
| helicopter | on | 39 | 36.60 | 36.54 | 3.42 | 3.41 | 1.01 | 13.22 | 2.87 | 2.08 | 0.00 |
| multirotor | off | 27 | 1.20 | 1.17 | 0.46 | 0.43 | 0.65 | 1.57 | 1.40 | 1.55 | 0.00 |
| multirotor | on | 18 | 19.69 | 19.68 | 20.40 | 20.38 | 4.62 | 9.28 | 10.22 | 1.37 | 0.00 |

The baselines are fitted on each vehicle's own calibration box (280 s) and applied to its other flights; the corrector has never seen the vehicle and infers it from the last window of telemetry and residual. The battery-current column is the physics with telemetry, what the corrector is meant to approach. The corrector after the lag model is the demo's chain: calibration, lag model, network for the rest. The IMU floor is the residual the attitude error alone leaves through the true rigid-body field, which no attitude-based method can remove.


### Navigation on the Gawler window at 60 m (11.4 km, 45 m pixels, survey lines every 4 rows, 1.0 nT survey noise, attitude from the IMU model; the chain leaves 2.58 nT against the true map, of which 2.38 nT varies over ten seconds or longer; the vehicle's drift is 3.0 nT per root minute)

| map | rms map error on hidden pixels nT | rms map gradient nT/m | likelihood width nT | position error after the first turn, 1.5 km initial box, m | bound m | from a 300 m box, m | bound m |
|---|---|---|---|---|---|---|---|
| true map | 0.0 | 1.752 | 2.8 | 10.2 | 3.2 | 14.6 | 3.2 |
| network (lines-trained) | 78.1 | 1.694 | 78.1 | 75.1 | 80.3 | 73.6 | 74.0 |
| network (generator-trained, random masks) | 238.4 | 2.705 | 238.4 | 63.6 | 118.1 | 60.7 | 102.0 |
| Gaussian estimator, beta 3.5 | 78.9 | 2.058 | 79.0 | 75.3 | 87.4 | 73.5 | 79.2 |
| linear interpolation between lines | 58.3 | 1.615 | 58.4 | 65.6 | 66.7 | 68.9 | 62.8 |

The same filter on the true map with a reading carrying white noise of the residual's rms instead of the chain's residual, from the 1.5 km box: 116.3 m after the turn with the offset allowed to walk as the drift does, 113.4 m with a constant offset, which is the case the bound of 3.2 m describes.

### Navigation on the Gawler window at 240 m (45.6 km, 178 m pixels, survey lines every 4 rows, 1.0 nT survey noise, attitude from the IMU model; the chain leaves 4.40 nT against the true map, of which 4.18 nT varies over ten seconds or longer; the vehicle's drift is 3.0 nT per root minute)

| map | rms map error on hidden pixels nT | rms map gradient nT/m | likelihood width nT | position error after the first turn, 1.5 km initial box, m | bound m | from a 300 m box, m | bound m |
|---|---|---|---|---|---|---|---|
| true map | 0.0 | 0.889 | 4.5 | 75.2 | 22.9 | 56.6 | 22.8 |
| network (lines-trained) | 179.1 | 0.774 | 179.2 | 286.2 | 782.3 | 29.0 | 232.8 |
| network (generator-trained, random masks) | 260.5 | 0.982 | 260.5 | 365.8 | 713.9 | 39.7 | 232.5 |
| Gaussian estimator, beta 3.5 | 112.9 | 0.783 | 113.0 | 282.4 | 398.4 | 54.7 | 210.4 |
| linear interpolation between lines | 117.8 | 0.717 | 117.9 | 341.4 | 668.8 | 39.2 | 223.5 |

The same filter on the true map with a reading carrying white noise of the residual's rms instead of the chain's residual, from the 1.5 km box: 99.4 m after the turn with the offset allowed to walk as the drift does, 92.4 m with a constant offset, which is the case the bound of 22.9 m describes.

The chain is the calibration and the lag model; the bound is the Cramer-Rao bound of Problem 1C along the flown track with the likelihood width used for matching (what the chain leaves on the calibration box, the slow part of its residual, and the map's rms error combined) and the offset as a third unknown; it assumes white sensor error, and the part of the chain's residual that varies over ten seconds is what keeps the realised error above it. A lower map rms does not by itself mean better navigation: the navigator uses the map's gradients, and interpolation between lines has the smallest rms error while flattening the gradient across lines, which is what a position fix needs.


### Demo readouts (scenario full, 620 s of flight, attitude from the IMU model)

- Part 1, fixed-wing box (752 nT of platform field): rank 17 raw and 16 band-passed at the end, 17 reached after 8 s; the linear model leaves 1.54 nT in band.
- Part 2, multirotor (2659 nT): on the box the linear model leaves 12.7 nT in band; on the free manoeuvre with the box's calibration: linear 21.2, Gauss-Newton 21.3, lag model 20.6, Gauss-Newton + lag 20.6, corrector after the linear model 29.0, after the lag model 25.3 nT (rms, drift and mean removed).
<!-- tables:end -->

## What the panels correspond to

- Part 1, rank 17 raw and 16 band-passed on the box: `problem1/checks.py`, `check_tl_rank`,
  and `tests/test_core.py::test_tolles_lawson_null_modes`.
- Part 2, the linear model leaving tens of nT on a 2700 nT platform and Gauss-Newton with
  the lag recovering it: `problem1/edge_cases.py`, sections 1, 2 and 4 (three platform
  classes).
- Part 3, navigation on the true map against the bound: Problem 1C and
  `magscale/magnav.py`; on reconstructed maps: docs/problem2.md Part G and
  results/grid_gawler.json (the tile numbers in the floor panel are those tiles).

## What it does not model

- Kinematics, not dynamics: attitude is prescribed, position follows the heading at constant
  speed, there is no wind and no turn dynamics. Attitude is known to the simulator; the
  estimator chain gets it through the IMU model when the option is on (the demo turns it
  on), whose white part leaves a floor no attitude-based method can remove.
- The Gawler world uses the inclination and declination of the window (about -65 and +7
  degrees); the synthetic world and the rank tests use the direction of `problem1/checks.py`.
  The magnitude is B0 plus the map value, taken as the scalar anomaly along the ambient
  direction; the anomaly's own vector components are dropped.
- Upward continuation of a draped survey is approximate over rough terrain; heights below
  the flown clearance are clamped to it (no downward continuation).
- The platform interference models beyond the Tolles-Lawson terms (motor current loops,
  battery dipole, servo pulses, thermal drift) are invented for the demonstration, with
  magnitudes chosen to match the three platform classes of `problem1/edge_cases.py`, and
  calibrated against nothing real. PWM ripple is generated at its own frequency and sampled
  at 10 Hz, so it appears aliased, as a logged current would.
- The sensor has white noise and a random-walk offset only: no heading error, no dead zone,
  no bandwidth limit.
- The calibration and the corrector's training data assume the map value along the
  calibration track is known (a calibration flight over a surveyed area). The corrector is
  not in the navigation chain: until its held-out table puts it within a factor of two of
  the physics baselines the navigator runs on the calibration and the lag model, and the
  corrector is part 2's before-and-after trace. The band-passed rank readout uses the
  constant-B0 regressor of the numerical checks; with the map value folded into B the
  isotropic induced column is no longer constant and the band-passed rank is 17.
- A lower map rms does not translate into better navigation: the navigator uses the map's
  gradients, and interpolation between lines has the smallest rms error on the 45 m grid
  while flattening the gradient across lines. The second leg, at 240 m over the same ground
  in 178 m pixels, is Part G's scale where the network reconstruction wins.
- The network map for the survey leg comes from the lines-trained checkpoint of the
  navigation study; the generator-trained model of Part G was trained on random masks and is
  out of its distribution on line masks, which its row in the table shows rather than hides.
- The bound is the static Cramer-Rao bound for a known trajectory shape; the filter has
  motion noise and a finite cloud, so it approaches the bound rather than reaching it.

## Running the corrector training on the cluster

The training data is generated on the node (synthetic maps, random vehicles), so the
checkout needs nothing beyond the repository. From the Mac:

    bash scripts/cuillin_submit.sh     # pushes the current commit, pulls it on Cuillin, submits scripts/sweeps/corrector.txt line 1
    bash scripts/cuillin_fetch.sh      # when the job is done: rsyncs results/corrector.json, runs/corrector.json, runs/corrector.pt and prints the table

Expected wall time for line 1 (300 vehicles, 40k steps, a fresh set of 300 vehicles drawn
every 2000 steps): about 20 regenerations of 90 s on the node's CPU plus a few minutes of
training on a 6000 Ada GPU and of evaluation, about 45 minutes in all; about three hours on
eight CPU cores. Line 2 is the same run with the IMU attitude model on (`--array=1-2` runs
both). The job skips itself if runs/corrector.pt already exists in the checkout on the
cluster. Fresh vehicles are drawn because, on a fixed set of 300, the held-out loss rises
after about a thousand steps while the training loss keeps falling: the model learns the
vehicles rather than how to identify one from its residual.

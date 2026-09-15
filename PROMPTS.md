# Prompts for Claude Code, one stage at a time

Run each in a fresh Claude Code session in the repository root, after reading the previous
stage's output and tests. Do not skip a review.

## Stage 1 (day 1): world, vehicle, platform, sensor

Read CLAUDE.md, SIM_SPEC.md, problem1/checks.py, problem1/edge_cases.py, magscale/grid_eval.py
and magscale/magnav.py. Build magsim/world.py, magsim/aircraft.py, magsim/platform.py and
magsim/sensor.py as specified, reusing the Tolles-Lawson regressor and the edge-case models from
problem1/ and the grid loader and continuation from magscale/grid_eval.py. Write
tests/test_magsim.py with: the calibration rank of a simulated box manoeuvre (17 raw, 16
band-passed) and of a straight leg (1, 0), reusing the regressor; the sensor reading equal to the
map value when the platform field is zero; the second-order term of the exact magnitude matching
|Bp_perp|^2 / 2B to 1%; a random vehicle's telemetry having the right shapes. Run the tests. Stop
and report: which existing functions you reused, what you had to change in them, and the test
output.

## Stage 2 (day 2): calibration and the corrector

Build magsim/calibrate.py (live least squares, rank, residual, band-passed variant) and
magsim/train_corrector.py (data generation over random vehicles and manoeuvres, the model in
magsim/corrector.py, training script with a --steps flag, evaluation on held-out vehicles
against the Gauss-Newton and lag baselines from problem1/edge_cases.py, results to
results/corrector.json). Add scripts/sweeps/corrector.txt with the one training command. Do not
run training; generate a small dataset, run 50 steps as a smoke test, and stop. Report the data
shapes, the parameter count, and the smoke-test loss going down.

## Stage 3 (day 3): navigator, maps, viewer

Build magsim/maps.py (the four maps of Part G for the same ground: true, network from 25% survey
lines, Gaussian estimator, interpolation; reuse magscale/magnav.py and the Part G evaluation
code; assert the reconstruction errors against results/grid_gawler.json), magsim/navigator.py
(particle filter with the Cramer-Rao bound alongside), and magsim/viewer.py (Rerun logging of
the map mesh, the vehicle, the path, the particles, and the time series listed in SIM_SPEC.md).
Test: the navigator on the true map with a clean sensor reaches the bound within a factor of
two after the first turn. Stop and report.

## Stage 4 (day 4): scenario, recording, docs

Build magsim/scenario.py and magsim/run_demo.py for the scripted four-minute flight in
SIM_SPEC.md, record demo.rrd, write magsim/README.md (what is simulated, what is not, the
corrector table rendered from results/corrector.json, the navigation-on-real-map table, how to
run), add scripts/render_tables.py support for the two json files, and one paragraph with one
still image for the front-page README under the post-submission section. Run the full test
suite. Stop and report what the README says the simulator does not model.

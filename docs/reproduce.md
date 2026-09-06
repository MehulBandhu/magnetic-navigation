# Reproduce

The front page has the three-command version. This page has everything else.

Five minutes, CPU only:

```
pip install -r requirements.txt
make test          # covariance, masks, floor, Wiener matrix, Tolles-Lawson null modes, gradient exponent, fitter: 10 checks
make demo          # one map, 75% hidden, interpolation vs exact oracle (vs network if weights present)
```

Checkpoint-independent figures, from the committed jsons, Five minutes, CPU only:

```
make figures       # Problem 1 checks, floors, fit, plots, the closed-form deep-linear test, the compute-optimal fit, docs/results_tables.md
```

`make figures` rewrites the committed figures, `results/fit*.json` and `docs/results_tables.md`.
Floors, slopes, knee ratios, local slopes, floor profiles, probes and navigation results
reproduce to the printed digits; the broken-power-law parameters and the leave-one-size-out
errors come from a non-convex multi-start fit and differ at the 10 to 25% level between scipy
versions, which is why the tables quote them to one decimal.

`make demo` and `make magnav` add a network column when the release checkpoints are in `runs/`;
without them they still run, and write `*_nonetwork` outputs so the committed versions are not
replaced. `make probes-core` (Jacobian, linearity, NTK on the two probed checkpoints) needs the
five files in `docs/release.md`, which are attached to the release at
https://github.com/MehulBandhu/magnetic-navigation/releases/tag/v0.1 with SHA-256 checksums.
`make probes-widths` needs the 30 width-run checkpoints, which are not distributed; it refuses to
run without them, and the numbers it produced are committed in `results/width_sweep.json`.

The full sweep, about 27 GPU-hours of recorded wall time (A100 for most of the main grid, RTX
6000 Ada and 2080 Ti on the Cuillin cluster for the rest; every run json records which, and
`docs/results_tables.md` has the breakdown), plus the CPU linear-reference jobs:

```
python scripts/sweep.py
bash scripts/run_list.sh scripts/sweeps/colab_main.txt        # 68 runs, about 11 A100-hours
bash scripts/run_list.sh scripts/sweeps/colab_smallD.txt      # 32 runs
bash scripts/run_list.sh scripts/sweeps/colab_fixedmask.txt   # 16 runs
bash scripts/run_list.sh scripts/sweeps/colab_seeds.txt       # 18 runs
bash scripts/run_list.sh scripts/sweeps/xxs.txt               # 20 runs, the 37k-parameter size
bash scripts/run_list.sh scripts/sweeps/knee.txt              # 5 runs, one factor at a time across the knee
bash scripts/run_list.sh scripts/sweeps/extra.txt             # 2 runs: no weight decay, noisy targets
bash scripts/run_list.sh scripts/sweeps/teacher.txt           # 4 runs: exact conditional mean as the target
bash scripts/run_list.sh scripts/sweeps/gap.txt               # 9 runs, saves weights
bash scripts/run_list.sh scripts/sweeps/width.txt             # 15 runs, saves weights
bash scripts/run_list.sh scripts/sweeps/altitude.txt          # 2 runs, the conditioning ablation
bash scripts/run_list.sh scripts/sweeps/linear.txt            # CPU, 40 jobs of 10 minutes
```

The EMAG2 test needs the 4.3 GB CSV from NOAA NCEI (EMAG2 v3, 2-arc-minute grid) in `data/`,
which is not committed; `python -m magscale.emag2 convert data/EMAG2_V3_20170530.csv` once, then
`make emag2` (calibration, synthetic control and the real tiles; about 15 minutes on a CPU).

`run_list.sh` skips a run whose json already exists in `runs/`, so lists can be interrupted and
resumed; to rerun an experiment from scratch, move the committed json aside or pass a different
`--out` to `magscale.train` (every line in `scripts/sweeps/*.txt` is a plain command). Every run writes one json
into `runs/`; all 191 are committed, so the fits and plots reproduce without a GPU. Every json
records the configuration, seed, floor, loss history, best loss, wall time and peak memory; the
runs made after the throughput timer and hardware fields were added also record the GPU,
precision and samples per second (the count is in docs/results_tables.md under Compute used).
`python scripts/env_info.py` prints the versions, GPU and commit for a new run.


## Checkpoints

The five files in `docs/release.md`, with SHA-256 checksums, are attached to the GitHub release
`v0.1`. Put them in `runs/` for `make probes-core`, the network column of `make demo` and
`make magnav`. The width-sweep checkpoints (30 files, 2.4 GB) are not distributed; their
numbers are committed in `results/width_sweep.json`.

## Compute used

About 27 GPU-hours of recorded wall time across 191 runs, plus 40 CPU jobs for the linear
reference; the breakdown by experiment, and the count of runs that record GPU, precision and
throughput, are generated into `docs/results_tables.md` under Compute used.

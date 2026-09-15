# Working in this repository

This is a research repository for magnetic navigation: reconstruction of magnetic anomaly maps
by neural networks, measured against an exact Gaussian floor, with real-data tests on EMAG2 and
on a 45 m survey grid. Read README.md, docs/problem2.md and figures/README.md before changing
anything. The current build is the flight simulator described in SIM_SPEC.md; work through it in
the stages listed in PROMPTS.md and stop at the end of each stage for review.

## Rules

- Reuse the existing modules. The Tolles-Lawson regressor lives in problem1/checks.py, its
  extensions in problem1/edge_cases.py, upward continuation and the grid loader in
  magscale/grid_eval.py, the Wiener estimator in magscale/floor.py, the trained networks in
  magscale/emag2.py (load_model), the survey-line masks and map matching in magscale/magnav.py.
  Import them; do not reimplement them. If one needs a small change to be reusable, make the
  change there and keep its tests passing.
- Every physical component gets a test in tests/test_magsim.py that checks a number the repository
  already contains: the calibration rank (17 raw, 16 band-passed, 1 straight, 0 straight and
  demeaned), the exact floor of a synthetic tile, the reconstruction error of the Gawler maps
  against results/grid_gawler.json, the Cramer-Rao bound of one reading.
- Run `python -m pytest -q tests` after every stage. Do not start any training run yourself;
  write the training script and the sweep line, and stop. Training happens on the cluster,
  launched by the user.
- Style: plain Python, functions over classes unless state is genuinely needed, comments that
  say why and read as written by a person (no "we", no "let's", no marketing), docstrings that
  say what the function computes and in what units. No em dashes anywhere, in code, comments,
  docs or commit messages. Sentence-case headings. No bold for emphasis in markdown.
- Units: nT for field, metres for distance, seconds for time, radians internally for angles.
  Every array that carries a physical quantity has its unit in the variable name or docstring.
- Simplifications are stated, not hidden. magsim/README.md lists what the simulator does not
  model. The platform interference models are invented for the demonstration and say so.
- Nothing is typed into a document that a script can generate. Numbers in docs come from json
  files in results/, rendered by scripts/render_tables.py or a sibling.
- Commit messages are one plain sentence describing the change.

## Environment

- Mac: `source .venv/bin/activate`; CPU only; the networks in runs/*.pt load and run on CPU.
- The Gawler grid is data/gawler_tmi.nc (local, ignored by git); EMAG2 is data/emag2_upcont.npy.
- Visualisation is Rerun (`pip install rerun-sdk`); log from Python, do not build a web front end.
- The cluster is Cuillin; job scripts are in scripts/slurm/, run lists in scripts/sweeps/.

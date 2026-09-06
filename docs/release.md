# Checkpoints

Model weights are not in the repository. The files below are attached to the GitHub release
`v0.1`; put them in `runs/` to run `make probes-core`, the network column of `make demo` and the
network column of `make magnav`.

| file | used by | sha256 |
|---|---|---|
| vitxxl_b3.5_h200_D131072_s0_ref.pt | jacobian / linearity probes at 8k steps, recon figure | d8bb3946d816cdb8fdebf77cada3512b5013b1b45fc42a916d2b4221219a5a6b |
| vitxxl_b3.5_h200_D131072_s0_ref_init.pt | NTK probe, initial weights | e8a7011de6edea1d4a1abfda563229c0bdc359972f564b57abeb3c02fd286dc8 |
| vitxxl_b3.5_h200_D131072_s0_long.pt | jacobian / linearity probes at 24k steps, demo | 9c11b957a46fcf46823ba811b57149dace0e83a898aace1c5c4d0c293f95488e |
| vitxxl_b3.5_h200_D131072_s0_long_init.pt | NTK probe, initial weights | 55fe058c5e3d6fe55fc5cb5cadba30441e64857badc3ad98d0bac6f81694166a |
| vitxxl_b3.5_h200_D131072_s0_lines.pt | navigation demo, network column | ae749765ba168b9ae57d535da2fc5abad9bc939e54ff32c3017049b2f19212f8 |

Verify after downloading with `shasum -a 256 runs/*.pt`.

The width-sweep NTK diagnostics were computed from the 15 width runs' initial and final weights
(30 files, 2.4 GB); those are not distributed. The numbers are committed in
`results/width_sweep.json` and the figure regenerates from them (`python -m magscale.plots widths`).

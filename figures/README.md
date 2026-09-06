# Figures

Each caption states what the figure shows about the corresponding question.

| file | part | content |
|---|---|---|
| `demo.png` | all | one patch: interpolation vs the exact conditional estimator vs the network; the estimator's error on a single patch scatters around the printed floor, which is its expectation over patches |
| `p1_linearisation.png` | 1A | the first-order scalar model is exact to 1 nT up to about 500 nT of platform field, then fails as eta^2 |
| `p1_gradient_scaling.png` | 1C(b) | the rms map gradient falls as h^((beta-4)/2) in the middle regime; finite grid and finite map bend it at both ends |
| `p1_vector_vs_scalar.png` | 1C(c) | vector information over scalar vs attitude error: the derived sigma/B0 scale is where the advantage halves, and it never drops below 1 |
| `p1_edge_eddy_lag.png` | 1 extra | standard TL exceeds 1 nT residual once omega*tau > 0.2; fitting the eddy time constant keeps the residual under 1 nT |
| `p1_edge_nonlinear.png` | 1 extra | linear TL crosses 1 nT at 3000 nT of platform field; Gauss-Newton on the exact model does not (toy, noise-free) |
| `p1_edge_latitude.png` | 1 extra | the box calibration is best conditioned at the magnetic equator and worst at the poles; a single heading is 1000x worse at the equator |
| `p1_edge_platforms.png` | 1 extra | toy platform classes: each breaks a different assumption; the multirotor requires a current regressor rather than a better attitude model |
| `recon_*_long_*.png` | 2 | the 24k-step network next to the exact conditional estimator: same structure, fine texture missing |
| `recon_*_ref_*.png` | 2 | the same patches with the 8k-step model, for comparison with the 24k-step one |
| `gap_curves.png` | 2C | the U-Net reaches 0.61 in 6k steps; the transformer needs 24k to reach 0.55 and is at 1.1 after 8k |
| `arch_vs_params.png` | 2C | U-Net excess falls as N^-0.71; the fully trained transformer lands near that line, the standard-budget one 18x above it |
| `altitude_ablation.png` | 2 setup | telling the model its altitude changes nothing measurable; absolute excess falls with h, relative excess rises |
| `jacobian_*_ref_h200.png` | 2C, 3 | at 8k steps the network has the filter's location but not its shape |
| `jacobian_*_long_h200.png` | 2C, 3 | at 24k steps the learned rows sit close to the exact Wiener rows, negative ring included (16% relative error over 16 sampled pixels) |
| `scaling_b*_h*.png` | 2B | excess over the floor against N (with the fitted protocol slope), against D per model size, and the floor profile: fit error against candidate floor, which falls toward E = 0 because the knee is easier to fit without a floor |
| `scaling_fm_b3.5_h200.png` | 3B | the fixed-mask block with linear regression on the same mask: overfitting at 256 fields, then flattening far above the linear learner |
| `exponents_vs_h.png` | 2B, 3B | protocol slope alpha against altitude (falls, contrary to the prediction) and the 5M model's local delta against D |
| `deep_linear.png` | 3C | closed-form gradient flow of the two-layer linear network: unmasked slopes 0.66 to 1.15 against 0.40 to 1.00 predicted (the offset is the dropped log factor); with a mask the whitened spectrum steepens and so does the exponent |
| `compute_optimal.png` | 2B | excess against compute along every logged curve, the envelope, and the fitted optimum N*(C); the fit is a description of this grid with a cosine-schedule caveat |
| `emag2_beta_map.png` | 2E | fitted spectral slope per land-survey tile (resampled to square pixels) over EMAG2's 4 km grid, Australia and North America: beta varies by province from below 2 to above 5 |
| `emag2_summary.png` | 2E | left: histogram of fitted beta against the assumed range; middle: network vs Gaussian-estimator error per tile (the network is above the line on all but one tile); right: realised vs predicted error of the Gaussian estimator, the distance above the line is how non-Gaussian the tile is |
| `emag2_tiles.png` | 2E | example real tiles: truth, observed 25%, network, Gaussian estimator with the fitted beta, interpolation, and the network's residual |
| `upward_continuation.gif` | 1C, 2A, 3B | one field continued upward from 0 to 1000 m with the exact floor and the mode count in the title: the field keeps its amplitude and loses its texture, the floor falls 280x |
| `navigation_likelihood.gif` | 1C | the position likelihood along a trajectory with one turn: a ridge along the contour (perpendicular to the gradient) after one measurement, rounding after the turn as the sampled gradients span both directions |
| `linear_reference.png` | 3B | the linear learner: spectrum-set exponent below ~1000 patches, collapse near the effective rank, exactly p/(D-p-1) beyond |
| `width_sweep.png` | 3C | four panels from `results/width_sweep.json`: loss, NTK alignment, relative change, trace ratio vs width. SP: direction freezes with width but scale drops 5x; muP-inspired: direction keeps moving, widest model wins |
| `magnav_altitude.png` | demo | the map-matching error lies on the Cramer-Rao bound (error bars: bootstrap over 40 trajectories); position error follows h^0.75 at beta 2.5, and at beta 3.5 the measured, bound, finite-map and asymptotic slopes separate for the reasons given in docs/problem2.md |
| `magnav_maps.png` | demo | map error becomes position error: true map, tiled Wiener, network and interpolation at 400 m and 800 m line spacing, mean and spread over three maps; the mean ordering is Wiener, network, interpolation, and the spread is too wide for every pairwise difference to be significant |

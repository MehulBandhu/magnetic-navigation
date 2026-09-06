# What went wrong, and what replaced it

Each item changed a number in the repository. The order is the order in which they happened.

| what failed | what replaced it |
|---|---|
| The scaling fit was done in log loss; with a fixed floor and large models close to it, their excess was invisible and the optimiser went to its bounds | The fit is on log excess over the exact floor |
| A free-floor point estimate that sat anywhere between zero and the exact value depending on the optimiser's path | A floor profile: the additive form refitted at each candidate floor; it prefers zero everywhere, because the step in N is easier to fit without a floor |
| The 25% difference between the 8k-step reference run and the 24k-step run at its own step 8000 was called run-to-run noise | It is a schedule effect (the reference had annealed, the long run had not); seeds were added and the two comparisons are kept apart |
| The step between 1.3M and 5M parameters was called a capability threshold | Five one-factor runs: width and training steps in equal parts, depth alone worse at a fixed budget, head dimension no effect |
| The NTK probe used eight outputs and cosine alignment only, which is dominated by the diagonal | 128 hidden-pixel outputs through a gradient sketch; centred alignment, top-eigenvector overlap, relative change and trace ratio |
| The claim that feature learning helps a little was written from the muP-inspired runs at one learning rate | Neither parametrisation wins at matched width under this budget; the muP-inspired runs at 4e-3 improve monotonically with width, SP's optimum drifts |
| The predicted altitude dependence of the parameter slope (steeper at altitude) | Wrong, and kept as wrong: the slope falls with altitude, and every run at 800 m was still improving when it stopped |
| The floor profile fitted and scored log(loss - E), whose scale changes with E, and concluded that a free fit puts the floor at zero | Fitted and scored in log total loss at every E; the free floor lands at 35% to 95% of the exact one on a flat profile, and the grid does not identify it |
| The fixed-mask transformer and the linear reference were described as sharing one mask; the masks were drawn on GPU and CPU from the same seed and differ | Described as different fixed-mask realisations of one protocol (floors 0.503 and 0.529) |
| The 8k-step reference run was described as the scaling grid's budget for the 38.5M model; the grid gave it 14k steps | Corrected everywhere; the reference is the gap study's intervention |
| Ridge regularisation for the linear reference was tuned on a small validation set, which double-counted noise | Oracle ridge from the exact excess risk |
| The tiled Wiener reconstruction for the navigation experiment used non-overlapping tiles, and looked worse than interpolation at the seams | Overlapping tiles with edge-weighted blending |
| A closing survey line added to the flight-line mask put the network out of distribution on the bottom tile (100 nT errors) | Regular line spacing only, matching the training pattern |
| The compute-optimal fit on all logged points was dominated by early training and returned exponents above 2 | Fit on the last two thirds of each run; the envelope exponent turned out to reflect the grid's own schedule (see docs/problem2.md) |
| EMAG2 tiles were cut on the latitude-longitude grid, so a 64-pixel tile at 45 N was 237 by 168 km and the isotropic prior was wrong by cos(latitude) | Each tile resampled to square 3.7 km pixels before anything else; North America's fitted slope moved from 4.1 to 3.5 |
| The EMAG2 tile selection included marine parts of the grid (ship tracks and a smooth model), adding a second population of tiles with slopes of 5 to 8 | Land aeromagnetic compilations only, by EMAG2's source-code column |
| The claim that the network is linear on its training distribution and not off it was written before it was measured | The mixing test on real tiles: 11 to 12% nonlinearity against 1 to 4% on Gaussian tiles |

# Problem 1: the physics of a magnetic fix

The derivations are handwritten, in `derivations.pdf` (18 pages, scanned). This page says
where each answer is, what it is, and what the numerical checks confirm; it also records four
corrections to the scan that were found after it was made.

## Where the answers are

| part | answer | pages |
|---|---|---|
| A, the scalar model | Delta m = a.u + B u^T M u + B u^T C du/dt with u the unit ambient field: 3 permanent, 6 induced (M symmetric), 9 eddy, 18 terms. Exact to second order in eta = \|B_p\|/B; the dropped term is \|B_p_perp\|^2 / 2B | 1 to 5 |
| B, rank | 17 raw: u.du/dt = 0 for any manoeuvre, so the isotropic eddy term c I is invisible. 16 after demeaning or band-passing: \|u\|^2 = 1 makes the isotropic induced term mu I a constant, degenerate with the DC level. 1 raw and 0 demeaned on a straight and level leg (u constant, du/dt = 0) | 6 to 9 |
| B, manoeuvre | Roll, pitch and heading excitation at distinct temporal frequencies; roll plus pitch reaches 17. Conditioning is best near the magnetic equator and worst near the poles | 9 to 10 |
| C(a), Cramer-Rao bound | J = g g^T / sigma^2 for one scalar measurement: rank one, position resolved along the local gradient only; the likelihood is a ridge along the contour. A trajectory whose sampled gradients span two directions makes J full rank | 11 to 13 |
| C(b), altitude law | E\|grad m_h\|^2 = (C / 2 pi) Gamma(4 - beta) (2h)^(beta - 4), so sigma_pos ~ h^gamma with gamma = (4 - beta)/2. The integral converges for beta < 4 between the map resolution (ultraviolet cutoff) and the map extent (infrared cutoff); at beta = 4 it is logarithmic, and above it the map extent sets the scale | 14 to 16 |
| C(c), vector vs scalar | Attitude error enters as [f]x Sigma_theta [f]x^T, and [f]x f = 0, so the ambient direction is an eigenvector with eigenvalue sigma^2 whatever the attitude covariance. The characteristic scale is eps* = (sigma / B_0) sqrt(G_v^2 / G_s^2 - 1), about 2e-5 rad (a thousandth of a degree) for 1 nT noise and a 50,000 nT field. It is a scale, not a crossing: the full-covariance vector information never falls below the scalar's, because the vector measurement contains the scalar projection | 17 to 18 |

## Corrections to the scan

- Page 18: 2 / 50,000 = 4e-5 rad, about 0.0023 degrees; the scan writes 4e-6, a factor of ten
  slip in the arithmetic. The numerical check below gives 2.0e-5 rad for the case it computes.
- Pages 17 to 18: the derivation presents eps* as the attitude accuracy below which a vector
  magnetometer adds information. The numerical companion shows that, with the attitude covariance
  modelled, the vector information never drops below the scalar's; eps* is the scale at which the
  advantage halves, and the statement in the table above is the one to use.
- Page 9: "two independent rotations" is a sufficient manoeuvre, not a minimality proof. The
  nondegeneracy condition is that the sampled u and du/dt span enough directions for the 17
  non-null columns to be independent; the roll-plus-pitch example in `checks.py` reaches rank 17
  and a single rotation does not (rank 10).
- Page 15: the three altitude regimes are named but not written out. Low altitude: the integral
  is set by the ultraviolet cutoff and saturates. Intermediate: the power law above. beta = 4:
  logarithmic in h. High altitude: the infrared cutoff enters and the law steepens; this is what
  bends the beta 3.5 slope in the navigation figure.

## Numerical checks

```
python -m problem1.checks        # rank by manoeuvre, second-order term, gradient power, vector vs scalar
python -m problem1.edge_cases    # eddy time constants, large platform fields, inclination, platform classes
```

`checks.py`: the rank of the 18-column regressor for four manoeuvres, raw and band-passed,
with the two predicted null vectors projected onto the computed null space (1.0000 for both);
the linearisation residual growing as eta^2 (slope 1.98); the slope of gradient power against
altitude (-1.51 measured against beta - 4 = -1.50 at beta 2.5, departing at beta 3.5 and 4 where
the finite map's cutoff enters); and the vector-versus-scalar Fisher information in full
geometry (ratio 2.13, halving at 2.0e-5 rad, never below 1). `edge_cases.py`: eddy currents with
a time constant, platform fields large enough to need Gauss-Newton on the exact model (about
3000 nT), observability against magnetic inclination (condition number 2.8 at the equator,
2.7e4 at the poles), and three platform classes. `tests/test_core.py` asserts the two null
modes, the ranks and the gradient exponent, and CI runs it on every push.

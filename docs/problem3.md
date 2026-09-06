# Problem 3: explaining the exponents

Claims are tagged at paragraph or argument-block level as [PROVEN] (a theorem exists for the
stated setting, with its assumptions given where used), [NUMERICAL] (this repository or the
cited experiments) or [CONJECTURE] (an argument without a proof).

Notation. Patch of n x n = 64 x 64 pixels, spacing dx = 100 m. Ground-level anomaly is a Gaussian
random field with power k^-beta; at altitude h the per-mode power is P_h(k) = C k^-beta e^-2kh.
Observation noise is white with variance sigma^2 = 1 nT^2. A patch has |O| = 1024 observed and
|H| = 3072 hidden pixels (75% random mask). For a given mask M the Bayes predictor is linear in
the observed values, m_H = W_M y_O with W_M = K_HO (K_OO + sigma^2 I)^-1, and its error is the
exact floor L_inf(beta, h, sigma). Across masks the predictor is the family {W_M}: the network,
which is given the mask as an input channel and sees a new mask every batch, has to represent
all of them. Everything in this document that says "linear" is conditional on a mask. The linear
reference below fixes one mask; the fixed-mask transformer study puts the network on that same
task so the two can be compared directly.

## A. What is known

**Kernel regression rates.** For kernel ridge regression with kernel eigenvalues
lambda_i ~ i^-a and a target whose power in the i-th eigenmode decays as c_i^2 ~ i^-b, the excess
risk after D samples is, up to constants and logs,

    noiseless, vanishing ridge:        D^-(b-1)            (needs a >= b-1)
    noisy target, optimal ridge:       D^-(b-1)/b

The first line is the "modes learned in order" picture: D samples pin down roughly the D largest
eigenmodes and the error is the target power left in the rest, sum_{i>D} c_i^2 ~ D^(1-b). The
second is the source/capacity rate of Caponnetto and De Vito (2007) [PROVEN, for kernel ridge
regression with i.i.d. samples, a target in the kernel's RKHS-power space with source exponent
r = (b - 1)/(2a), capacity exponent a > 1, bounded noise, and the ridge tuned as D^(-a/(2ar+1))]
rewritten in these exponents. Both rates assume the kernel eigenbasis and the target coefficients
are power laws in the same basis. Bordelon, Canatar and Pehlevan (2020) and Spigler, Geiger and Wyart (2020) give the
learning curve from the spectrum in closed form in a high-dimensional limit [PROVEN within that
limit, NUMERICAL otherwise]; Cui, Loureiro, Krzakala and Zdeborova (2021) give the crossover from
the noiseless to the noisy rate [PROVEN in the Gaussian design setting].

**NTK regime.** At infinite width under standard parametrization, gradient descent on a network is
kernel regression with the neural tangent kernel (Jacot et al. 2018) [PROVEN]. Chizat and Bach
(2019) characterise when finite networks are "lazy" (the kernel does not move) [PROVEN for the
scaling they define]. Under muP (Yang and Hu 2021) the infinite-width limit learns features and
the optimal learning rate transfers across width [PROVEN for the limit; transfer is NUMERICAL].

**Neural scaling laws.** L(N, D) = E + A N^-alpha + B D^-delta is empirical (Kaplan et al. 2020,
Hoffmann et al. 2022) [NUMERICAL]. Theory exists for models that are linear in their parameters:
Bahri et al. (2021) relate alpha and delta to the spectrum of the data in the variance-limited and
resolution-limited regimes [PROVEN for random-feature models, NUMERICAL for deep nets]; Maloney,
Roberts and Sully (2022) solve a random-feature model with joint N and D scaling [PROVEN within
the model]; Bordelon, Atanasov and Pehlevan (2024) add training time [PROVEN within a linear
random-feature model]; Lin, Wu, Kakade, Bartlett and Lee (2024) prove matching bounds for SGD on
linear regression with a power-law covariance [PROVEN]. Sharma and Kaplan (2020) and Michaud et
al. (2023) offer mechanisms for deep networks [CONJECTURE, with numerical support].

**Feature learning with rigorous content.** One gradient step on a two-layer network already
escapes the kernel regime for single-index targets (Ba et al. 2022) [PROVEN]. Deep linear
networks have exactly solvable gradient-flow dynamics when the input is whitened (Saxe, McClelland
and Ganguli 2014) [PROVEN] and under balanced initialisation more generally (Arora et al. 2018,
Gidel et al. 2019) [PROVEN].

**The smallest unanswered version of the question.** Take a two-layer network with a nonlinearity,
trained by gradient descent from a standard initialisation on inputs whose covariance spectrum is
a power law truncated by e^-2kh plus a white floor, with a linear target. What is the exponent of
the excess risk in D, and in training time, and is either different from kernel regression on the
same data? For the linear two-layer network the time exponent can be derived (part C); the D
exponent at convergence coincides with linear regression because the minimiser is the same. For
the nonlinear masked finite-width setting I did not find a theorem that applies.

## B. Kernel prediction for this data model

**Mode counting.** Modes are 2D wavevectors. Ranked by |k|, the rank of wavevector k is
rho ~ (L k / 2 pi)^2, so the per-mode target power c_rho^2 ~ rho^-beta/2, i.e. b = beta/2.
The kernel the linear learner uses is the input covariance, whose eigenvalues are
A(k)^2 + sigma^2: the same power law with a noise floor, so a = beta/2 as well and the
condition a >= b-1 holds. The transformer's lazy-limit kernel is its NTK, which is not this raw-
pixel kernel; the linear learner is the kernel-regime baseline for the task, and identifying the
transformer's NTK spectrum with it would need a separate argument that is not made here.
Plugging in:

    noiseless   delta = (beta - 2)/2  :  0.25, 0.50, 0.75, 1.00  for beta = 2.5, 3.0, 3.5, 4.0
    noisy       delta = (beta - 2)/beta:  0.20, 0.33, 0.43, 0.50

The dimension d = 2 enters through rho ~ k^d: in general delta_noiseless = (beta - d)/d. [PROVEN
in the kernel-regression setting, given the identification of a and b above]

**Which regime.** sigma = 1 nT is small next to the per-pixel variance (500 to 2500 nT^2) but not
next to the power in the modes that are being learned last. The crossover between the two rates
happens where the unlearned target power drops to the noise level, and it moves to smaller D as
h grows. Measured exponents between the two rows are therefore expected.

**The altitude term.** With modes ranked by |k| in two dimensions, rho ~ k^2, the per-mode power
at altitude is lambda_rho ~ rho^-beta/2 exp(-2h c sqrt(rho)) with c = 2 pi / L. Below the cutoff
this inherits the ground-level power law; above it the tail falls faster than any power law. So
e^-2kh does not give a different exponent, it ends the power-law regime. Define the effective rank
r(h) as the number of modes whose signal power exceeds the noise, A(k)^2 e^-2kh > sigma^2. Solving
for the cutoff wavenumber gives k*(h) ~ (1/2h) log(C / (sigma^2 k*^beta)), so r(h) ~ (L/h)^2 up
to a log. For these patches (from `floor.effective_modes`):

    h [m]     0     100    200    400    800
    beta=2.5  4096  2085   795    288    100
    beta=3.5  4074  880    425    187    73

Consequences [CONJECTURE, with the linear learner as NUMERICAL support]:

1. For D << r(h) the learning curve is the power law above.
2. For D ~ r(h) the excess collapses faster than any power law (local delta > 1): the learner
   runs out of signal modes to learn.
3. For D >> r(h) what remains is finite-dimensional estimation. For ordinary least squares with
   Gaussian inputs, independent noise and D > p + 1 the excess is (residual variance) x p / (D - p - 1),
   independent of the input spectrum [PROVEN, classical]; p/D is its large-D limit, so delta -> 1
   whatever beta and h.
4. The same three regimes apply to N: a model only needs to represent a rank-r(h) linear operator,
   so parameter scaling saturates at N ~ r(h) x (something modest), earlier at altitude.
5. Compute-optimal allocation: with L = E + A N^-alpha + B D^-delta and C = 6 N D T, the optimum
   is N ~ C^(delta/(alpha+delta)) and D ~ C^(alpha/(alpha+delta)). Where N saturates (large
   effective alpha) the split moves toward data, so at altitude the allocation would favour
   data over parameters. Untested here, because D stopped mattering above about a thousand
   fields (Part E, last row).

**Why the closed-form exponents are an approximation for this task.** The formulas assume the kernel eigenbasis
and the target coefficients follow power laws in the same basis. The input here is 1024 noisy
pixels on a mask, the target is 3072 other pixels, and the relevant spectra are those of
K_OO + sigma^2 I and of the cross-covariance K_HO, which are the field spectrum filtered through
the mask geometry. The closed-form delta values above are therefore a controlled approximation.
The kernel-regime prediction for this exact task is a computation: linear regression on the same
patches under one fixed mask, run exactly (`linear_ref.py`, exact excess risk, no test-set noise).
Result, exponent measured on D in [256, 1024] with oracle ridge:

    beta \ h    0      100    200    400    800
    2.5        0.59   1.14   1.43   1.10   0.81
    3.0        0.71   1.22   1.32   1.02   0.71
    3.5        0.86   1.24   1.14   0.93   0.72
    4.0        0.96   1.10   0.99   0.86   0.70

[NUMERICAL]. At ground level the exponent rises with beta as the theory orders it, sitting nearer
the noiseless row than the noisy one. At 100 to 200 m it exceeds 1: regime 2, the collapse, because
r(h) there is 400 to 2000, inside the D window. At 800 m it is back to 0.7: regime 3 begins below
D = 256 because r(800) is under 100. Above D ~ 2000 every curve has exponent close to 1 and the
excess follows the classical floor x p / (D - p - 1) with p = 1024 (0.190 measured against 0.265
for the large-D approximation floor x p / D at D = 2048, beta 3.5, 200 m; the two agree once
D is well above p).

**Prediction for the transformer, written before the fit.** Two things differ from the linear learner. First, in the main
grid the mask is resampled every epoch, so one field yields thousands of distinct (input, target)
pairs and the network has to represent the whole family {W_M}; the effective sample count is far
above the field count and the network sits in regime 3 (or past it) for any D above a few
thousand fields. Prediction: delta measured on D in [4096, 262144] is close to 0, and the loss
depends on D only through optimisation (fresh fields help SGD). [NUMERICAL: at every condition and
size, D = 4096, 32768 and 262144 show no stable monotonic trend; local slopes scatter around zero.] The fixed-mask study
(`scripts/sweeps/colab_fixedmask.txt`, D in {256, 1024, 4096, 32768}) removes this difference
and is the closest comparison; its mask was drawn on the GPU and the linear reference's on the
CPU from the same seed, so they are different fixed-mask realisations of the same protocol
(floors 0.503 and 0.529), and the comparison is between protocols, not on one mask. Second, the transformer starts from
random weights and is trained for a fixed budget, so its measured alpha is partly optimisation.
Predictions written down before the fit: delta on D in [256, 1024] follows the linear learner's
non-monotone pattern in h (about 0.9, above 1, 0.7 at 0, 200, 800 m for beta = 3.5); alpha is
larger at 800 m than at 0 m because fewer modes need representing; the fitted floor comes out
above the exact one when left free.

## C. Beyond kernels

**Experiments** [NUMERICAL]. Width sweep at depth 4, 4000 steps, beta = 3.5, h = 200 m:

    width           64     128    256    512    1024
    SP  loss        8.04   3.26   2.07   1.21   1.41
    SP  alignment   0.42   0.74   0.86   0.99   0.995
    muP loss        3.09   2.63   1.84   1.57   1.27
    muP alignment   0.63   0.64   0.77   0.76   0.80

The empirical NTK is the Gram matrix of parameter gradients over 128 (input, hidden pixel)
outputs, computed through a count-sketch of the gradients so that 50M-parameter models fit in
memory; alignment is the cosine between the centred Gram matrices at initialisation and after
training, and the top-eigenvector overlap, relative Frobenius change and trace ratio are stored
with it. Under SP the centred alignment rises with width from 0.32 to 0.99 and the top-eigenvector
overlap from 0.03 to 0.90, while the relative change stays near 0.85 and the trace ratio falls to
0.18: the kernel's direction stabilises with width while its scale drops five-fold. That is not
the lazy regime in the strict sense (kernel unchanged in direction and scale). Weight decay is
not the cause of the scale drop: the width-512 run without it has the same trace ratio (0.178
against 0.178) and the same alignment (0.971), at a slightly higher loss (excess 0.80 against
0.70). Under the muP-inspired scaling at 4e-3 the direction keeps moving at every width
(alignment 0.37 to 0.52, top-eigenvector overlap below 0.25) and the trace ratio rises to 1.8 at
width 1024, while the loss improves monotonically with width. The SP optimum drifts with width, so the widest SP model gets
worse. At matched width neither is better under this budget: the lowest excess in the sweep is
SP at width 512 (0.70), and muP at 4e-3 only wins at width 1024 (0.76 vs 0.89), where SP's
untuned rate has failed; SP's rate was never retuned per width, so the comparison is between one
parametrisation that transfers its rate and one that does not. The main-grid models (SP) are
closer to kernel regression than to feature learning, and the linear reference is the right first
comparison for their exponents, with the caveat above. Two more facts: the trained 38M
transformer is linear in its input to under 0.5% (mixing test at fixed mask: 0.5% at 8k steps,
0.15% at 24k, affine offset under 0.5%), and its Jacobian rows on 16 sampled pixels come within
16% of the exact Wiener rows, negative ring included, once trained to 24k steps (59% at 8k). Its
NTK moves like the wide SP models': alignment 0.95 to 0.96, top-eigenvector overlap 0.63, trace
ratio 0.10 to 0.12. On this target, feature learning
is available and is not required.

"muP-inspired" rather than muP: the implementation scales attention logits by 1/d instead of
1/sqrt(d), multiplies the head output by base_width/width, and scales the Adam learning rate of
hidden and output matrices by base_width/width. It does not use the mup package's base shapes,
per-layer initialisation rules or coordinate checks, so it is not validated as the maximal-update
parametrisation of Yang et al. What it does demonstrate is learning-rate transfer across a 16x
range of width at a fixed base rate, which is the property used here.

**A rigorous statement for a model that learns features** [PROVEN in the whitened, aligned
case]. Compare two models with the same minimiser: one-layer linear regression f(x) = W x, and
the two-layer linear network f(x) = W_2 W_1 x, both under gradient flow on the population squared
loss with a linear target. Write the singular value decomposition of the input-output
cross-covariance Sigma_yx = U S V^T. For whitened inputs (Sigma_xx = I) and an initialisation
that is balanced and aligned with the target's singular directions (W_1(0) = eps V^T,
W_2(0) = eps U up to the singular structure, as in Saxe et al. 2014), the dynamics decouple mode
by mode; from a generic small initialisation the modes first align and then follow the same
curves, and that alignment phase is not part of what is computed here (Dominé et al. 2024
separate the two cases).
For the one-layer model, mode i converges as

    u_i(t) = s_i (1 - e^-t),

a linear ODE with the same rate for every mode. For the two-layer network with small balanced
initialisation u_i(0) = eps (Saxe et al. 2014),

    u_i(t) = s_i / (1 + (s_i/eps - 1) e^(-2 s_i t)),

a sigmoid that switches on at t_i = log(s_i/eps) / (2 s_i). Modes are learned in order of s_i, at
times inversely proportional to s_i, whereas the one-layer model learns them all at once.

The time exponent, derived from that solution with the log factor dropped. Write the singular
values of the whitened cross-covariance as s_rho ~ rho^-c (c is a new exponent, not the b of
part B: after whitening, s_rho is the per-mode correlation between input and target, which for
the unmasked problem is A_rho^2 / sqrt(A_rho^2 + sigma^2), so c = beta/4 for the modes above the
noise floor). At time t the modes with s_rho t > 1 are learned, i.e. rho < rho*(t) ~ t^(1/c). The
loss contribution of an unlearned mode is its target power, A_rho^2 ~ rho^-beta/2, so

    L(t) - L_inf ~ sum_{rho > rho*} rho^-beta/2 ~ rho*^(1 - beta/2) ~ t^-(beta - 2)/(2c) = t^-2(beta - 2)/beta,

up to the log(s_rho/eps) correction, which for beta = 3.5 is t^-0.86. For a one-layer model on
non-whitened inputs with covariance eigenvalues ~ rho^-a the same counting gives
t^-(beta - 2)/(2a), and on whitened inputs no power law at all (every mode at the same rate). So
the two models reach the same minimiser along different laws, and the difference is entirely due
to the first layer moving. [The modal solution is PROVEN for the whitened, aligned, unmasked
setting; the exponent is a mode-counting ARGUMENT over a restricted spectral window that drops
logs, not a theorem about the training dynamics in general.]

**Test of the derivation** (`magscale/deep_linear.py`, `figures/deep_linear.png`,
`docs/results_tables.md`) [NUMERICAL]. With balanced initialisation the two-layer product per
mode is the logistic above, so the population gradient flow needs no solver: the excess is a sum
over modes of (u_k(t) - s_k)^2. Two settings are computed exactly. Unmasked, ground level: the
singular spectrum decays as rho^-c with c = 0.62 / 0.75 / 0.87 / 1.00 for beta = 2.5 / 3 / 3.5 / 4,
against beta/4 = 0.62 / 0.75 / 0.88 / 1.00, so the first step of the derivation is exact; the
two-layer excess falls with slope 0.66 / 0.83 / 0.99 / 1.15 in the window where the switching
front sits between rank 16 and rank 1024, against 0.40 / 0.67 / 0.86 / 1.00 predicted. The
ordering and spacing are right and the offset (0.15 to 0.26) has the size of the log factor the
counting step drops, although the test evaluates the assumed logistic solution rather than
independent training dynamics, and the offset has not been shown to arise from the log alone. The one-layer model on the same whitened inputs has no power law. Masked
and whitened (75% random mask, beta = 3.5): the singular values of K_HO (K_OO + sigma^2 I)^-1/2
decay faster than the unmasked spectrum, c = 1.21 / 1.59 / 2.44 at 0 / 200 / 800 m, and the
two-layer slope is 1.86 / 1.89 / 2.17. So the exponent does not survive the mask as written: the
mask changes the spectrum, and the counting rule applied to the masked spectrum, (2c - 1)/c =
1.17 / 1.37 / 1.59, gives the direction of the change and about two thirds of its size. What
remains untested is gradient flow on the raw masked inputs, where Sigma_xx and Sigma_yx do not
share singular vectors and the modes couple [CONJECTURE that the whitened result carries over
approximately].

## D. Things tried that did not work

- muP with the base learning rate copied from the SP runs (1e-3 at width 64): losses 3 to 10x
  worse than SP at every width. muP transfers the optimum, it does not create one; the base rate
  must be tuned at the base width. Fixed at 4e-3.
- Ridge regularisation tuned on a 512-patch validation set for the linear reference: at large D the
  differences between lambda values fall below the validation resolution and the curve goes flat.
  Replaced by oracle lambda (chosen by the exact excess), which is what the theory describes.
- Reconstructing a whole survey map tile by tile without overlap: the rows at each tile's edge had
  no survey line on one side and were extrapolated, making the Wiener filter look worse than
  linear interpolation at 800 m line spacing. Fixed with overlapping tiles and a centre-weighted blend.
- The Wiener prior built for 64-pixel training patches (128-pixel periodic grid) applied to a map
  generated on a 512-pixel grid: too little long-wavelength power in the prior, visible only when
  survey lines were far apart. Fixed by matching the prior to the map's grid.
- The altitude-conditioning ablation on a single altitude: the FiLM input is a constant, so the
  12% difference measured was seed scatter. Repeated on a three-altitude mixture: no effect.
- Fitting alpha from one seed: synthetic tests with 3% scatter give alpha 0.1 high with +-0.06
  bootstrap error; delta is unaffected. Seeds are the only fix and the budget did not allow them.
- Chinchilla additive fit with a free floor on a grid where D barely matters: the fit absorbs the
  D term into the floor; report the fixed-floor fit and show the free one as a warning.
- Calling the reference-versus-long-run difference at step 8000 seed scatter: the two runs
  were at different points of their learning-rate schedules, so it was a schedule effect. Scatter
  is measured with repeated seeds and nothing else.
- Reading the NTK from cosine alignment alone: it hid a five-fold drop in kernel scale. Report
  relative change and trace ratio with it.
- A linear-regression baseline on one fixed mask presented as the same task as the random-mask
  transformer: it is the same protocol only for the fixed-mask transformer study, which was added
  for that reason, and even there the two masks are different realisations (GPU and CPU draws
  from one seed).

## E. Predictions against measurements

    quantity                                        predicted before the fit         measured
    delta_linear, h = 0, beta = 3.5                 0.43 to 0.75                     0.86
    delta_linear vs h                               non-monotone, peak near r(h)     0.86, 1.24, 1.14, 0.93, 0.72 at 0..800 m
    delta_transformer, D >= 4096 (random mask)      about 0                          within +-0.1 of 0, every size and altitude
    delta_transformer, 5M model, 256 -> 1024, h=0   about 0.9                        0.55
    same, h = 200                                   above 1                          0.29
    same, h = 800                                   about 0.7                        0.59
    delta_transformer, fixed mask, 256 -> 1024      close to the linear 1.14         0.89 (5M), rising with size from 0.15
    alpha, h = 800 vs h = 0                         alpha(800) > alpha(0)            0.53 < 0.74: wrong
    free-floor E vs exact                           E_free > E_exact                 wrong in the other direction: the free floor lands at 35% to 95% of the exact one, on a profile too flat to identify it
    allocation at altitude favours data              yes                              untested: D stopped mattering above about 1000 fields
    linear learner beats transformer at same D      yes                              0.004 vs 0.045 at D = 131k (fixed vs random mask)

Three of these need comment. The transformer's small-D slopes with the resampled mask are
about half the linear learner's at the same altitudes and do not show the collapse above 1 at
200 m; the resampled mask multiplies the effective sample count, so the transformer is never as
data-starved as the linear learner at the same field count, and the non-monotone pattern is
washed out. On the fixed mask, where the protocols coincide, the slope is 0.89 against 1.14: closer,
and the gap is the transformer's optimisation budget (those runs were still improving). The
altitude prediction for alpha was wrong. The effective-rank argument says fewer modes need
representing at altitude, so parameter scaling should saturate earlier and the slope in the
resolved range should be steeper; the measured slope falls from 0.74 to 0.53 instead. Every run
at 800 m was still improving at its last step, and at 800 m the excess is measured against a
floor of 0.08 nT^2, so the fitted slope there is mostly a statement about how far a fixed budget
gets a model of each size, not about what it could represent. The measurement stands; the prediction was made for a converged protocol and this one was
not. A converged sweep would decide it.

## References

- Caponnetto, A. and De Vito, E. (2007). Optimal rates for the regularized least-squares algorithm. Found. Comput. Math. 7, 331.
- Bordelon, B., Canatar, A. and Pehlevan, C. (2020). Spectrum dependent learning curves in kernel regression and wide neural networks. ICML. arXiv:2002.02561.
- Spigler, S., Geiger, M. and Wyart, M. (2020). Asymptotic learning curves of kernel methods: empirical data versus teacher-student paradigm. J. Stat. Mech. arXiv:1905.10843.
- Cui, H., Loureiro, B., Krzakala, F. and Zdeborova, L. (2021). Generalization error rates in kernel regression: the crossover from the noiseless to noisy regime. NeurIPS. arXiv:2105.15004.
- Jacot, A., Gabriel, F. and Hongler, C. (2018). Neural tangent kernel: convergence and generalization in neural networks. NeurIPS. arXiv:1806.07572.
- Lee, J. et al. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. NeurIPS. arXiv:1902.06720.
- Chizat, L., Oyallon, E. and Bach, F. (2019). On lazy training in differentiable programming. NeurIPS. arXiv:1812.07956.
- Yang, G. and Hu, E. (2021). Feature learning in infinite-width neural networks. ICML. arXiv:2011.14522.
- Yang, G. et al. (2022). Tensor Programs V: tuning large neural networks via zero-shot hyperparameter transfer. arXiv:2203.03466.
- Kaplan, J. et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- Hoffmann, J. et al. (2022). Training compute-optimal large language models. arXiv:2203.15556.
- Bahri, Y., Dyer, E., Kaplan, J., Lee, J. and Sharma, U. (2021). Explaining neural scaling laws. arXiv:2102.06701.
- Sharma, U. and Kaplan, J. (2020). A neural scaling law from the dimension of the data manifold. arXiv:2004.10802.
- Maloney, A., Roberts, D. and Sully, J. (2022). A solvable model of neural scaling laws. arXiv:2210.16859.
- Michaud, E., Liu, Z., Girit, U. and Tegmark, M. (2023). The quantization model of neural scaling. NeurIPS. arXiv:2303.13506.
- Bordelon, B., Atanasov, A. and Pehlevan, C. (2024). A dynamical model of neural scaling laws. ICML. arXiv:2402.01092.
- Bordelon, B., Atanasov, A. and Pehlevan, C. (2024). How feature learning can improve neural scaling laws. arXiv:2409.17858.
- Canatar, A., Bordelon, B. and Pehlevan, C. (2021). Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature Communications 12, 2914. arXiv:2006.13198.
- Paquette, E., Paquette, C., Xiao, L. and Pennington, J. (2024). 4+3 phases of compute-optimal neural scaling laws. arXiv:2405.15074.
- Lin, L., Wu, J., Kakade, S., Bartlett, P. and Lee, J. (2024). Scaling laws in linear regression: compute, parameters, and data. arXiv:2406.08466.
- Ba, J. et al. (2022). High-dimensional asymptotics of feature learning: how one gradient step improves the representation. NeurIPS. arXiv:2205.01445.
- Saxe, A., McClelland, J. and Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR. arXiv:1312.6120.
- Arora, S., Cohen, N., Golowich, N. and Hu, W. (2018). A convergence analysis of gradient descent for deep linear neural networks. arXiv:1810.02281.
- Gidel, G., Bach, F. and Lacoste-Julien, S. (2019). Implicit regularization of discrete gradient dynamics in linear neural networks. NeurIPS. arXiv:1904.13262.
- Gnadt, A., Wollaber, A. and Nielsen, J. (2022). Derivation and extensions of the Tolles-Lawson model for aeromagnetic compensation. arXiv:2212.09899.
- Kaub, L. et al. (2021). Magnetic surveys with unmanned aerial systems: software for assessing and comparing the accuracy of different sensor systems, suspension designs and compensation methods. Geochem. Geophys. Geosyst. 22, e2021GC009745.


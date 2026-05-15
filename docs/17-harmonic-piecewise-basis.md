# Harmonic Piecewise Basis (HPB) — Iters 1 and 2

Parallel research line opened alongside KSQ/KSQP, testing whether an
offset-paired triangle/square basis organized by harmonic frequency can
serve as an MCC substrate primitive. Plan in
[`harmonic_piecewise_basis.md`](harmonic_piecewise_basis.md); this
write-up covers iter 1 (basis + XOR), iter 1.5 (parity sweep + 2D XOR
+ L2 finite-norm-basin diagnostic), and iter 2 (smoothing kernels +
smooth-function approximation).

Implementation in `sibarum.strnn.hpb` — same locality discipline as
KSQ and KSQP. Promotion to `mcc-core` is deferred until the
architecture-level claims survive further iters.

## Scope

**Iter 1 — basis correctness and minimal capacity test.** Build the
raw (δ-kernel) basis as a `PiecewisePolynomial` substrate over
`double[]`. Exact rationals deferred per the iter-1 plan decision;
double-precision is sufficient for the iter-1 falsification gates
because tri_k and sq_k have integer-rational coefficients that
round-trip exactly through double. Run T1 (basis spec), T3 (gradient
check), T5 (1D XOR).

**Iter 1.5 — stronger capacity tests after T5 was found structurally
trivial.** A single period of sq_1 evaluated at four equally-spaced
positions is a parity by construction, so T5 doesn't pressure the
basis. Added: T6 parity sweep for T ∈ {2, 4, 8}; 2D XOR over
independent inputs with three architectures (per-dim, joint
tensor-product, per-dim + hidden ReLU); long-run CE descent test to
diagnose the apparent "CE → machine zero" result on joint-lift; L2
finite-norm-basin test to falsify the doc's "exact-rational basin"
claim under the right loss function.

**Iter 2 — smoothing kernels.** `PiecewisePolynomial.antiderivative()`,
`SmoothedBasisElement` (δ/box/tent), `HpbRegressionModel`. T2
(derivative pairing under smoothing) and T7 (smooth-function
approximation with closed-form least squares).

The original plan's T8 (crossover), T9 (Fourier-class generalization),
T10 (PDE operator learning), and the trainable-kernel stretch (T14)
are deferred to iter 3+.

## Architecture

```
  x ∈ [0, 1)                                            ←  scalar input
    │
    │   per-frequency lift
    ▼
  features = [tri_1(x), sq_1(x), …, tri_K(x), sq_K(x)] ∈ R^{2K}
    │
    │   linear readout (the only learned parameters)
    ▼
  logits = W · features + b ∈ R^outDim
```

`tri_k` and `sq_k` are constructed as `PiecewisePolynomial` instances
with period `T = 1/k`, amplitude 1, and four pieces over one period.
`sq_k = d/dx tri_k` by construction — both as polynomial coefficients
on each piece and pointwise away from breakpoints.

The basis itself has no learned parameters. This matches the KSQ
philosophy (fixed structural substrate; learning constrained to a
small set of indexing parameters) and isolates the iter-1 question to
"does the basis carry XOR through a linear readout?"

## T1 — Basis correctness

`HpbBasisCorrectnessDemo`. For k ∈ {1, 2, 3}, checked:

- Piece-by-piece equality of `triK(k).derivative()` against `sqK(k)`
  in polynomial coefficients (coefficient tolerance 1e-12).
- Pointwise `tri_k` at the breakpoint anchors {0, T/4, T/2, 3T/4}
  against the §2.1 spec values {0, +1, 0, −1}.
- Pointwise `sq_k` at the mid-piece anchors {T/8, 3T/8, 5T/8, 7T/8}
  against the spec values {+4k, −4k, −4k, +4k}.
- Periodic-wrap identity: `tri_k(T + T/8) == tri_k(T/8)`.
- Dense-grid (N=1000, offset to miss breakpoints) pointwise
  `tri'(x) == sq(x)`.

Result: **3039 checks, 0 mismatches. PASS.** F1 does not fire on the
δ-kernel basis.

## T3 — Gradient check

`HpbGradientCheckDemo`. K=2, outDim=2 (so featDim=4, 4·2 + 2 = 10 params
per test point). Test points chosen mid-piece for both k=1 and k=2
(1/16, 5/16, 9/16, 13/16) so finite-difference perturbations of W and
b never cross a breakpoint. Central-difference eps = 1e-5, tolerance
1e-6 abs or 1e-4 rel.

Result: **40 parameter gradients checked, 0 mismatches, max abs err
7.9e-10. PASS.** F1 does not fire on the backward pass.

## T5 — XOR

`HpbXorDemo`. Encoding: the four XOR inputs are mapped to a single
scalar `x = (4·n + 1)/16` where `n = 2·b0 + b1 ∈ {0,1,2,3}`. Inputs
land at {1/16, 5/16, 9/16, 13/16} — mid-piece for k=1 and k=2, no
breakpoint collisions. Standard XOR labels.

10 seeds, 2000 epochs, lr=0.05, two-class cross-entropy.

| K | featDim | solved | final CE (range)        |
|---|---------|--------|--------------------------|
| 1 | 2       | 10/10  | 3.06e-4 .. 3.12e-4       |
| 2 | 4       | 10/10  | 2.87e-4 .. 3.12e-4       |

Result: **F2 does not fire.** XOR is solvable by the raw harmonic
piecewise basis with a single linear readout.

### Why K=1 already works

At the four encoded positions, `sq_1(x)` takes values `{+4, −4, −4, +4}`
— exactly the XOR sign pattern, scaled. The two output logits under
softmax-CE need to differ by a sign aligned with sq_1: with
`W[0][sq_1] − W[1][sq_1]` set to any positive multiple of −1, the
softmax separates the four points correctly. SGD finds an arrangement
of (W, b) consistent with this discrimination from every random seed;
the magnitude grows as long as training continues (separable data,
unbounded CE ray — see the Long-Run section below).

The choice of encoding is doing real work here: by placing the four
inputs at the four mid-pieces of the k=1 period, we set up a sq_1
feature that is exactly the XOR-discriminating signal. Other
encodings exist (e.g., per-dimension lift of two binary scalars) that
*cannot* solve XOR with a single linear readout — those would need
either joint lifting (tensor-product of per-dim harmonics) or a
hidden layer. Iter 1 deliberately picked the simplest encoding that
exposes the basis's capacity.

### What T5 actually tests — and what it doesn't

T5 as written falsifies a weaker claim than the doc's §5 F2 names. A
single period of a square wave evaluated at four equally-spaced
positions *is* a parity, by construction — any periodic feature with
two sign changes per period would solve this. The K=1 demo doesn't
distinguish HPB from generic Fourier features and doesn't pressure
the basis's capacity. The capacity test is **T=4 / T=8 parity**
(below), where one period of a square wave no longer suffices.

### CE doesn't reach machine zero

Final CE ≈ 3e-4, not 1e-16. This is the standard CE asymptote — driving
CE to machine zero requires logits → ±∞, which plain SGD on
unregularized CE approaches only at log scales. Accuracy is 4/4 on
every seed; the discriminating direction in W has been found, but the
ray along that direction is unbounded. The 2D XOR joint-lift result
(below) reaches CE = 1.1e-13 in 5000 epochs because its
max-margin-per-unit-‖W‖ is 4× larger than 1D K=1's — *not* because
joint has a finite-norm optimum (it doesn't; see the Long-Run
section).

## T6 — Parity sweep T ∈ {2, 4, 8}

`HpbParityDemo`. For each T, encode the 2^T input patterns as scalars
`x_n = (n + 0.5) / 2^T`, label by `popcount(n) mod 2`. Sweep K ∈
{1, 2, 4, 8, 16}; for each (T, K), 5 seeds, 5000 epochs, lr=0.05.

```
   T \ K     K=1     K=2     K=4     K=8     K=16
   --------------------------------------------------
   T=2       5/5     5/5     5/5     5/5     5/5
   T=4       0/5     0/5     0/5     5/5     5/5
   T=8       0/5     0/5     0/5     0/5     0/5
```

What this shows:

- **T=2 solves at K=1**, consistent with T5 — XOR's DFT support at
  N=4 is on frequency 1, which sq_1 captures.
- **T=4 needs K ≥ 8**, not K=4. The reading is DOF-counting, not
  Nyquist-coverage: parity at T=4 has zero DFT amplitude at the
  Nyquist freq 8 (verified directly), so freq 8 isn't the obstruction.
  But the K=4 basis aliases at the N=16 sample points — sq_3, sq_5,
  sq_7 share DFT support with sq_1, collapsing the effective basis
  rank. K=4 provides ~9 real DOFs (including bias) against parity's
  ~14 nonzero real DOFs at non-Nyquist freqs, so the LS projection has
  positive residual and the soft-label classification can't reach
  zero error. K=8 raises the feature count past N, making the system
  underdetermined-or-saturated; perfect fit becomes available and
  SGD finds it.
- **T=8 unsolved at K ≤ 16**. By the same counting, T=8 parity needs
  ~N = 256 real DOFs at non-zero freqs; K=16 supplies far less.
  K ≈ N/2 is the rough threshold — for T=8 that would be K=128, a
  stretch test deferred for compute.

**Cross-architecture comparison with KSQ.** KSQ's iter-6 hit a
degree-2 expressivity ceiling: T=4 and T=8 parity both collapsed to
predicting uniform regardless of magnitude. HPB does *not* have a
degree-2 ceiling — T=4 solves cleanly at K=8 (5/5 seeds, no tuning).
The cost is that K must grow with task degree, roughly at the rank-
saturation rate K ≈ N/2. HPB and KSQ failure modes are structurally
different: KSQ runs out of polynomial degree, HPB runs out of effective
feature rank under sample aliasing. The rank-saturation cost is a real
limitation, but a different one.

## 2D XOR over independent inputs

`Hpb2dXorDemo`. Three architectures on the same 4-point 2D XOR
problem (each bit b ∈ {0, 1} mapped to x ∈ {0.125, 0.625}; standard
XOR labels). 5 seeds, 5000 epochs, lr=0.05.

**A. Per-dim lift + linear readout.**

| K | featDim | solved | best CE |
|---|---------|--------|---------|
| 1 | 4 | 0/5 | 6.93e-1 |
| 2 | 8 | 0/5 | 1.52e+0 |
| 4 | 16 | 0/5 | 1.94e+0 |

Predicted to fail; failed at every K. The lift is linear in each
input dimension and a linear readout cannot multiply features across
dimensions, so no amount of K helps. CE at K=1 sits at log(2) — the
chance baseline; at K=2 and K=4, CE exceeds chance, indicating
saturating-wrong-direction logits. This confirms the iter-1 reading:
T5 worked because the encoding folded the 2D XOR into a single
period, not because per-dim lift composes across dimensions.

**B. Joint tensor-product lift + linear readout.**

| K | featDim | solved | best CE |
|---|---------|--------|---------|
| 1 | 4 | 5/5 | 1.11e-13 |
| 2 | 16 | 5/5 | 0.00e+0 |

K=1 (featDim = 4 = (2·1)²) suffices. The cross-feature
`sq_1(x_0) · sq_1(x_1)` takes values `{+16, −16, −16, +16}` at the
four XOR points — exactly the XOR sign pattern. The optimal logit-
difference direction puts a (signed) coefficient on this single
cross-feature; SGD finds an arrangement consistent with that
direction from every seed.

CE reaches 1.11e-13 at K=1 and exactly 0.0 at K=2 — but as the
Long-Run section below shows, this is **not** because joint K=1 has
a finite-norm optimum. Both joint K=1 and 1D K=1 are on unbounded
CE rays; joint just descends faster because its max-margin-per-‖W‖
is 4× larger and its parameter budget is slightly larger. The
"exact-rational basin" reading of F2 is properly tested under L2,
not CE (see "Exact-rational basin — verified under L2" below).

**C. Per-dim lift + hidden ReLU + linear readout.**

| K | hidden | solved | best CE |
|---|--------|--------|---------|
| 1 | 4 | 5/5 | 1.16e-4 |
| 1 | 8 | 5/5 | 1.05e-4 |
| 2 | 4 | 4/5 | 1.43e-4 |
| 2 | 8 | 5/5 | 1.33e-4 |

A standard 2-layer MLP on top of the per-dim lift solves 2D XOR
robustly. Baseline confirmation that HPB drops into existing MLP
architectures as a feature transform — the per-dim lift is not a
dead end, it just needs a non-linearity above it.

## Long-run CE descent — is the "exact-rational basin" real under CE?

The B (joint K=1) result reaching CE = 1.1e-13 vs T5's CE = 3e-4
initially looked like evidence for an "exact-rational basin" in the
joint architecture. `HpbXorLongRunDemo` tests this by running 1D
K=1 XOR out to 200,000 epochs and measuring CE descent rate:

```
   epochs        CE          ‖W‖   logit margin
   ------    ---------    ------   ------------
     2000    3.08e-04       1.73          7.69
     5000    1.23e-04       1.86          8.62
    20000    3.08e-05       2.07         10.02
    50000    1.23e-05       2.21         10.94
   100000    6.16e-06       2.32         11.64
   200000    3.08e-06       2.43         12.34
```

The empirical fit: **CE halves with every doubling of epochs**, and
logit margin grows like `0.7 · log₂(t)` — the classic logistic-
regression-on-separable-data behavior. CE follows `CE(t) ≈ 0.62/t`
(verified: at every checkpoint, `CE × t ≈ 0.62`).

Reaching CE = 1e-13 along this ray would require ~10¹³ epochs.
Reaching float64 underflow (margin ≈ 700) is computationally
inaccessible. **The 1D and joint K=1 architectures are not on
equivalent rays.** Joint K=1 reached CE = 1.1e-13 at 5000 epochs,
implying (if joint also follows 1/t descent) a constant
`C_joint ≈ 5.5e-10`. Ratio `C_1D / C_joint ≈ 0.62 / 5.5e-10 ≈ 10⁹`.

The mechanism is the L2-max-margin-per-unit-‖W‖ of each architecture:
1D K=1's max-margin direction (all weight on sq_1) yields margin ≈ 4
per ‖W‖; joint K=1's max-margin direction (all weight on sq·sq)
yields ≈ 16. Since CE ∝ exp(-margin), a 4× difference in margin-
per-norm compounds *exponentially* in the CE descent rate. Combined
with joint's slightly larger parameter budget (10 vs 6 params for
4 data points), the convergence rates diverge by many orders of
magnitude.

**What this rules out.** No architecture in iter 1 demonstrates a
finite-‖W‖ "exact basin" under cross-entropy. Every separable-data
+ CE setup gives an unbounded ray; only the rate constant differs.
The doc's §3 prediction of an "exact rational solution; loss reaches
machine zero" is a category error under CE: machine zero is reached
only by float64 underflow when the ‖W‖-along-margin product crosses
~700, which is rate-dependent, not a structural feature of the basin.

**The right test for the exact-rational basin claim is L2 against
soft or label targets**, where the optimum is at a specific finite
‖W‖. Under L2 against label targets {0, 1}, the 1D K=1 architecture
has a unique closed-form least-squares optimum at
`(w_tri, w_sq, b) = (0, -1/8, 1/2)` — a rational point with zero
residual (the labels happen to lie in the column-span of the
feature+bias matrix).

## Exact-rational basin — verified under L2

`HpbXorL2Demo` runs SGD on half-MSE for the 1D K=1 architecture
against the {0, 1} labels. All 5 seeds, at 5000 epochs and beyond:

```
   epochs    half-MSE         w_tri              w_sq               b              ‖W − W*‖
   ------   ---------    ----------------   ----------------   --------------     ----------
     1000   ~1e-13       ~+8e-7             -0.124999...       +0.50000000        ~8e-7
     5000   1.12e-31     -3.6e-17           -0.125000000000    +0.500000000000    5.0e-16
    20000   1.12e-31     -3.6e-17           -0.125000000000    +0.500000000000    5.0e-16
   100000   1.12e-31     -3.6e-17           -0.125000000000    +0.500000000000    5.0e-16
```

- **`w_sq = -0.125` exactly** in float64 (`-1/8` is exactly representable).
- **`b = 0.5` exactly** in float64.
- **`w_tri ≈ -3.6e-17`** — at the float64 precision floor; mathematically zero.
- **half-MSE = 1.12e-31** — machine zero squared, the precision floor for sums of squares of order 5e-16.
- **‖W − W*‖ ≈ 5e-16** — the float64 representation floor at this magnitude.

The basin is reached by 5000 epochs, then SGD stays there for the
remaining 95k epochs (no drift). The convergence is monotone in
‖W − W*‖ and quadratic near the optimum (L2 with full-rank features
is strictly convex).

**This validates the doc's §5 F2 claim under the right loss
function.** The exact-rational basin exists, lives at the rational
point predicted by closed-form linear algebra, and SGD reaches it
to float64 precision. The earlier CE-based confusion was an artifact
of CE not having a finite-norm minimum on separable data, not a
property of HPB.

**Why this matters for the framework's pitch.** The claim that HPB
has an exact-rational basin distinguishes it from generic Fourier
features (which would also reach this point under L2, with the same
rationality) and from ReLU MLPs (whose basin coordinates are
irrational in general). The exact-rational property is real and
testable, but the test must use a loss with a finite-norm minimum.
A future iter with a `Rational` substrate could verify the basin
*as an exact rational identity* rather than as a 5e-16 numerical
neighborhood; that is the natural follow-up if/when exact arithmetic
is introduced.

## Iter 2 — smoothing kernels (box, tent)

Iter 2 adds compact-support kernels and the closed-form convolution
machinery to test the doc's regularity-tuning claim (F3).

### Implementation

- `PiecewisePolynomial.antiderivative()` — antiderivative with
  drift-correction so the result is periodic even when the input has
  nonzero mean. Used to express `(f * box(w))(x)` as
  `(F(x+w/2) − F(x−w/2)) / w` and `(f * tent(w))(x)` as
  `(4/w²)·(G(x+w/2) − 2 G(x) + G(x−w/2))` (the doubled-box identity
  `tent(w) = box(w/2) * box(w/2)`).
- `SmoothedBasisElement` — wraps an original PWP plus its first/second
  antiderivatives, evaluates the kernel convolution analytically.
  Three factories: `delta(p)`, `box(p, w)`, `tent(p, w)`.
- `HpbRegressionModel` — scalar input → smoothed harmonic lift →
  linear readout → scalar prediction. Used by T7.

### T2 — derivative pairing under smoothing

The pairing `(tri_k * φ)' = sq_k * φ` follows from convolution
commuting with differentiation; numerically verifying it via central
difference is misleading because the truncation error
`O(ε² · f''')` scales unfavorably at high k under tent kernels. The
demo instead checks two analytic identities on the antiderivative
chain, pointwise on a dense grid:

- **Identity A** (box pairing): `sq_k.antiderivative() ≡ tri_k`.
- **Identity B** (tent pairing): `sq_k.antiderivative().antiderivative()
  ≡ tri_k.antiderivative()`.

Both identities reduce the pairing claim to function equality at the
antiderivative level (machine-exact, no derivative estimation). The
demo also samples `SmoothedBasisElement` against the closed-form
formula as a sanity check that the class produces the same values
the math predicts.

Result: **7000 pointwise checks, 0 mismatches, max abs err = 0.0e+00.**
The PWP coefficients are integer-rational multiples of 1/k, which
round-trip exactly through float64, so the identities hold bit-exact.

### T7 — smooth function approximation

Closed-form least squares (normal equations) on a smooth periodic
target:
`f(x) = exp(sin(2π x)) + sin(4π x) cos(6π x)`, sampled at N=64
points. Compares δ / box / tent kernels at K ∈ {2, 4, 8, 16} and
width fractions w_k / T_k ∈ {1/16, 1/8, 1/4}.

```
   w_k = T_k / 16
       K | delta MSE      | box MSE        | tent MSE       | best/delta
       --+----------------+----------------+----------------+-----------
       2 | 1.20e-01       | 1.19e-01       | 1.20e-01       |   0.99
       4 | 1.20e-01       | 1.19e-01       | 1.19e-01       |   0.99
       8 | 4.43e-03       | 2.64e-03       | 3.27e-03       |   0.60
      16 | 9.05e-04       | 5.33e-04       | 4.98e-04       |   0.55

   w_k = T_k / 8
       K | delta MSE      | box MSE        | tent MSE       | best/delta
       --+----------------+----------------+----------------+-----------
       2 | 1.20e-01       | 1.22e-01       | 1.20e-01       |   1.00
       4 | 1.20e-01       | 1.22e-01       | 1.20e-01       |   1.00
       8 | 4.43e-03       | 1.57e-03       | 2.26e-03       |   0.35
      16 | 9.05e-04       | 3.85e-04       | 4.54e-04       |   0.43

   w_k = T_k / 4
       K | delta MSE      | box MSE        | tent MSE       | best/delta
       --+----------------+----------------+----------------+-----------
       2 | 1.20e-01       | 1.29e-01       | 1.24e-01       |   1.03
       4 | 1.20e-01       | 1.28e-01       | 1.24e-01       |   1.03
       8 | 4.43e-03       | 2.52e-04       | 7.97e-04       |   0.06
      16 | 9.05e-04       | 1.57e-05       | 9.91e-05       |   0.02
```

What this shows:

- **K=2, K=4 are below-rank** for the target — all three kernels
  plateau at MSE ≈ 0.12 (about 1/3 of target variance), regardless
  of width. The basis isn't rich enough to capture f; smoothing the
  basis elements doesn't help when the limitation is dimensionality.
- **K=8 and K=16 show the kernel effect clearly.** At every tested
  width, both smoothed kernels beat δ. The advantage grows with
  width: at w_k = T_k/4 and K=16, the smoothed bases give MSE = 1.6e-5,
  a **57× reduction** vs the δ basis.
- **Box ≤ tent at most settings.** The doc's specific prediction
  `tent < box < δ` is wrong on this target. Box wins at every width
  for K=8, and at the larger widths for K=16; tent ties or slightly
  wins only at the smallest width (T_k/16) at K=16.

The intuition (predicted but not borne out): tent gives higher
continuity (C^1 vs C^0) and should match smooth targets better.
The empirical: box's slower frequency rolloff
(Fourier transform of box is `sinc(ω w / 2)`, tent is
`sinc²(ω w / 2)`) preserves more useful high-frequency content for
this target. Tent over-smooths.

**F3 is decisively refuted: smoothing buys substantial MSE reduction
on smooth targets.** Up to 57× at the tested settings. The
"tunable regularity" property is real and ablate-able. The doc's
specific kernel ordering is wrong but the broader claim holds.

The result has a natural follow-up question (deferred): what is the
optimal kernel-and-width as a function of target regularity? The
crossover-characterization test (T8 in the original plan) would
sweep target smoothness directly. With closed-form LS, that's a
cheap experiment — a candidate for iter 3.

## Falsification register

| Criterion | Status | Notes |
|-----------|--------|-------|
| F1 (implementation / gradient bug) | did not fire | T1 + T3 both clean |
| F2 (raw basis cannot solve XOR / parity / 2D XOR) | did not fire | T5 (1D XOR) 10/10 at K=1; T6 T=4 parity 5/5 at K=8; 2D XOR joint-lift K=1 solves with CE → underflow |
| F2′ (HPB has KSQ's degree-2 ceiling) | did not fire | T=4 parity solves at K=8; HPB's limit is rank saturation under sample aliasing (K ≈ N/2 threshold), not a polynomial-degree bound |
| F2″ ("exact-rational basin" claim under CE) | **category error** | CE has no finite-‖W‖ minimum on separable data; long-run test shows CE ∝ 1/t descent — the question is malformed under CE |
| F2‴ ("exact-rational basin" claim under L2) | did not fire | `HpbXorL2Demo`: 5/5 seeds reach (0, -1/8, 1/2) to float64 precision; half-MSE = 1.12e-31 (machine zero squared); basin verified at the closed-form rational point |
| F3 (smoothing buys nothing on smooth targets) | did not fire | T7: smoothed kernels give 2–57× MSE reduction vs δ on smooth target at K ≥ 8; effect strongest at moderate-to-large w_k |
| F3′ (predicted tent < box ordering) | **fired** | Box ≤ tent at most settings on the tested target; doc's specific kernel ordering claim is wrong, but the broader F3 claim holds. Tent over-smooths (faster Fourier rolloff than box). Empirical reading: kernel choice is target-dependent. |
| F4 (raw-vs-smooth crossover is monotone) | not tested | iter 3 — closed-form LS makes this cheap |
| F5 (FNO baseline beats HPB on operator learning) | not tested | iter 4+ |

## What iters 1, 1.5, and 2 demonstrated

- The δ-kernel basis is correct as constructed (T1).
- Linear-readout gradients through the lift are correct (T3).
- 1D-encoded XOR is solvable through a single linear readout (T5) —
  but this is structurally trivial (one period of sq_1 *is* a parity
  on 4 points by construction).
- T-bit parity is solvable at K roughly at the rank-saturation
  threshold ≈ N/2. T=2 at K=1, T=4 at K=8, T=8 not reached at K ≤ 16.
- HPB has a different expressivity profile than KSQ. KSQ's
  degree-2 bilinear ceiling kills T=4; HPB's K-scaling cost (rank
  saturation under sample aliasing) lets T=4 solve at K=8.
- Per-dim lift + linear cannot solve 2D XOR over independent inputs;
  joint tensor-product lift K=1 (featDim=4) solves it; per-dim lift
  + hidden ReLU + linear also solves it as a standard MLP-on-feature-
  lift.
- The "exact-rational basin" property is real and reachable, but
  only under a loss with a finite-norm optimum. Under L2 against
  label targets, 1D K=1 SGD converges to the exact rational point
  `(0, -1/8, 1/2)` at float64 precision across every seed. Under
  CE, no finite-norm optimum exists and the question is malformed.
- The smoothed basis (box, tent kernels) is correctly constructed —
  derivative pairing under smoothing verified exactly via analytic
  identities on the antiderivative chain (T2: 7000 checks, 0
  mismatches, max abs err 0.0).
- Smoothing buys substantial approximation power on smooth targets
  (T7: 2–57× MSE reduction vs δ kernel at K ≥ 8, scaling with
  smoothing width). The "tunable regularity" property is real and
  ablate-able under closed-form LS.
- The doc's specific kernel ordering `tent < box < δ` did not hold;
  empirically `box ≤ tent ≤ δ` at most settings on the tested
  smooth target (tent over-smooths because of sinc² Fourier rolloff
  vs box's sinc).

What's still not addressed:

- Whether the basis competes with Fourier features or FNO at
  function approximation scale (T9, T10 — deferred).
- Whether the K-scaling cost can be reduced (e.g., by dyadic
  frequency selection, or by a logarithmic-depth multi-resolution
  variant).
- Whether T=8 parity is solvable at K=128 — the predicted rank-
  saturation threshold; not yet run because compute cost is ~minutes
  per seed.
- Whether the exact-rational story is empirically realized in
  arithmetic (no `Rational` substrate yet — verified to float64
  precision only).
- The raw-vs-smoothed crossover characterization (T8) as a function
  of target regularity — natural iter-3 candidate, cheap under
  closed-form LS.

## Iter 3 candidates

Iter 2 closed: T2 passed exactly, T7's F3 didn't fire (smoothed
kernels give 2–57× MSE reduction over δ on smooth targets). Natural
next questions, in rough priority order:

1. **T8 — crossover characterization.** Parameterize a family of
   targets from "rational/discrete" (parity-like) to "smooth"
   (sinusoid-like) and find the crossover point where raw and
   smoothed bases tie. With closed-form LS this is a cheap sweep:
   evaluate residual MSE on each target for δ, box, tent at fixed
   K. The doc's F4 predicts a non-monotone crossover; iter 3 would
   either confirm or refute it.
2. **Why box beats tent on smooth targets** — a small analytic
   side-question. Box's Fourier transform is `sinc(ω w/2)`, tent's
   is `sinc²(ω w/2)`; tent attenuates high harmonics more
   aggressively. For a smooth target with non-trivial
   high-frequency content (like the `exp(sin)` series), preserving
   those harmonics matters. The empirical kernel ordering is
   target-dependent in a way the doc's F3 didn't capture.
3. **Adaptive width.** The width sweep showed w_k = T_k/4 is best
   at K=16 but the ordering reverses for K=2 (where any smoothing
   hurts). A per-K optimal width could be picked from the LS
   residual itself; not a "learned" hyperparameter but a
   target-conditioned one.
4. **T6 follow-up at K=128 for T=8 parity.** Iter-1.5's parity
   sweep flagged this as a rank-saturation-threshold stretch test
   (K ≈ N/2 = 128 for N=256), deferred for compute. With ~1–2
   minutes of CPU it's reachable.
5. **T10 — PDE operator learning (FNO comparison).** The
   highest-value but highest-cost item. Requires significant
   infrastructure (1D heat eq solver as supervision, FNO
   architecture as baseline). Likely iter 4 or 5.

## Files

- `strnn-model/src/main/java/sibarum/strnn/hpb/PiecewisePolynomial.java`
- `strnn-model/src/main/java/sibarum/strnn/hpb/HarmonicBasis.java`
- `strnn-model/src/main/java/sibarum/strnn/hpb/HpbModel.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbBasisCorrectnessDemo.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbGradientCheckDemo.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbXorDemo.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbParityDemo.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/Hpb2dXorDemo.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbXorLongRunDemo.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbXorL2Demo.java`
- `strnn-model/src/main/java/sibarum/strnn/hpb/SmoothedBasisElement.java`
- `strnn-model/src/main/java/sibarum/strnn/hpb/HpbRegressionModel.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbDerivativePairingDemo.java`
- `strnn-model/src/main/java/sibarum/strnn/demo/HpbSmoothApproximationDemo.java`

## Run

```
mvn -pl strnn-model compile
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbBasisCorrectnessDemo
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbGradientCheckDemo
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbXorDemo
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbParityDemo
java -cp strnn-model/target/classes sibarum.strnn.demo.Hpb2dXorDemo
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbXorLongRunDemo
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbXorL2Demo
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbDerivativePairingDemo
java -cp strnn-model/target/classes sibarum.strnn.demo.HpbSmoothApproximationDemo
```

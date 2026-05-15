# Harmonic Piecewise Basis: Implementation Guide and Testing Plan

## 1. Concept

A neural network basis built from offset-paired triangle and square waves, organized by harmonic frequency index, with a tunable smoothing kernel that controls regularity without breaking derivative pairing.

The three load-bearing properties:

1. **Exact derivative pairing.** For each frequency `k`, the basis contains a triangle wave `tri_k` and a square wave `sq_k` related by `d/dx tri_k = sq_k` (up to constants), with `sq_k` phase-shifted by a quarter period — the discrete analog of the `sin`/`cos` pairing.
2. **Tunable regularity via convolution.** A kernel `φ` smooths both halves of the basis uniformly. Because convolution commutes with differentiation, the pairing survives: `d/dx (tri_k ∗ φ) = sq_k ∗ φ`.
3. **Exact rational arithmetic.** All breakpoints, slopes, and peaks are rational. With piecewise-polynomial `φ`, the smoothed basis stays piecewise-polynomial with rational coefficients.

The piecewise-linear (unsmoothed) limit is ReLU-native: each basis element is representable as a small ReLU subnetwork with periodic boundary conditions.

## 2. Basis Construction

### 2.1 Geometry of a single (tri, sq) pair at frequency k

Let `T = 2π/k` be the period (or `T = 1/k` in unit-period convention — pick one and commit).

**Triangle `tri_k(x)`:** piecewise linear, peaks at `x = T/4 + nT/2` with alternating sign. On `[0, T/2]`, ramps up from 0 at `x=0` to peak `+a` at `x=T/4`, then down to 0 at `x=T/2`. On `[T/2, T]`, ramps down to `-a` at `x=3T/4`, then back to 0.

**Square `sq_k(x)`:** the literal derivative of `tri_k`. Piecewise constant with value `+slope` on `[0, T/4]`, `-slope` on `[T/4, T/2]`, `-slope` on `[T/2, 3T/4]`, `+slope` on `[3T/4, T]`. This is a square wave of frequency `2k` of the triangle — that's the correct sin/cos analog.

**Offset alignment:** the peak of the triangle sits at the center of the square's "high" plateau in the appropriate phase, by construction.

### 2.2 Choice of amplitude

Pick `a = 1/k` if you want the triangle to have Fourier-like decay (matching the natural `1/k²` spectral content of a triangle wave evaluated at frequency `k`). Pick `a = 1` if you want unit amplitude per basis element and let the readout weights handle the scaling. Recommend `a = 1` for training; reweight at inference if needed.

### 2.3 Smoothing kernel φ

Three canonical choices, all exact-rational:

- **δ (no smoothing):** raw piecewise-linear basis. ReLU-native.
- **box of width w:** smoothed basis is piecewise quadratic. `C^0` continuity at smoothed corners.
- **tent of width w:** smoothed basis is piecewise cubic. `C^1` continuity.

Higher-order: convolve box with itself n times to get order-(n+1) B-spline-like kernels, giving `C^{n-1}` regularity.

The smoothing width `w` should be a fraction of the basis period — e.g., `w = T/8` is a reasonable default. Note: `w → 0` recovers the δ limit; `w → T/2` smooths the basis element into near-zero. There's a sweet spot.

## 3. Implementation Plan

### 3.1 Java reference implementation (priority: high)

Build alongside the existing exact-arithmetic substrate. Required components:

- `PiecewisePolynomial`: rational-coefficient piecewise polynomial on a periodic domain, with breakpoints stored as exact rationals.
- `HarmonicBasisElement`: indexed by `(k, half)` where `half ∈ {tri, sq}`, parameterized by amplitude `a` and period `T`.
- `Kernel`: piecewise polynomial with compact support; convolution operator on `PiecewisePolynomial`.
- `Convolve(basis, kernel)`: produces the smoothed basis as a `PiecewisePolynomial`. Closed-form; no numerical integration.
- `Evaluate(basis, x)`: exact rational evaluation at rational `x`.
- `Derivative(basis)`: returns the derivative as another `PiecewisePolynomial`; for the unsmoothed basis, derivative of `tri_k` should equal `sq_k` up to numerical identity (verify).

### 3.2 Gradient verification (required before training)

Standard methodology — verify before any optimization.

For each `(k, half, kernel)` combination, check:
1. `d/dx (tri_k ∗ φ) = sq_k ∗ φ` symbolically — i.e., the result of convolving and then differentiating equals the result of differentiating and then convolving, exactly.
2. Numerical gradient of the basis element evaluated at random rational `x` agrees with analytic derivative to machine epsilon (in floating-point cross-check) or to exact zero (in rational arithmetic).
3. Backpropagated gradient through a one-layer network with this basis matches a finite-difference gradient to the same tolerance.

If any of these fail, the implementation is wrong before any architecture-level claims can be tested.

### 3.3 PyTorch wrapper for training (priority: medium)

Wrap the exact-arithmetic forward pass in a `torch.autograd.Function` that:
- Forward: evaluates the smoothed basis at floating-point inputs by converting to nearest rational, evaluating exactly, converting back.
- Backward: returns the analytic gradient computed exactly, converted to floating-point.

For training on consumer hardware, exact arithmetic throughout is too slow. Use exact arithmetic for: (a) initialization, (b) gradient verification, (c) post-training crystallization (anneal to exact rationals if the target supports it). Use float32 for the bulk of training.

### 3.4 Architecture variants to support

The basis is a primitive; multiple architectures use it. Build the basis first as a standalone module, then plug into:

- **Lifting layer:** `R → R^{2K}`, maps scalar `x` to `[tri_1(x), ..., tri_K(x), sq_1(x), ..., sq_K(x)]`. Linear readout follows.
- **Per-dimension lifting:** for input in `R^d`, apply the lifting independently to each dimension, concatenate. Standard Fourier-features pattern.
- **Activation replacement:** drop in for ReLU in an MLP, with frequency index tied to layer position or learned.
- **FNO-style operator layer:** lift, multiply by trainable kernel weights, project. The exact-arithmetic version of the FNO spectral convolution.

Recommend implementing in this order — each variant is a small extension of the previous.

## 4. Testing Plan

### 4.1 Sanity tests (gate further work)

Before any benchmark claims, verify:

**T1 — Basis correctness.** Evaluate `tri_k` and `sq_k` at dense grid; verify piecewise structure matches specification. Check derivative pairing `d/dx tri_k = sq_k` holds pointwise away from corners and in distribution at corners.

**T2 — Smoothing correctness.** For each kernel `φ`, verify `(tri_k ∗ φ)` has the expected regularity (continuity, smoothness order) and that the derivative pairing is preserved.

**T3 — Gradient correctness.** As described in §3.2. Required before any training experiment.

**T4 — Orthogonality check.** Compute `<tri_j, tri_k>` and `<sq_j, sq_k>` for `j ≠ k` on the fundamental period. If exact orthogonality holds, the basis is well-conditioned; if not, characterize the cross-talk. This determines whether the basis is a true frame or needs explicit Gram-matrix handling.

### 4.2 Capability tests (architecture-level claims)

**T5 — Exact XOR solve.** One-layer network, raw basis (`φ = δ`), trained on XOR. Predicted result: exact rational solution achievable; loss reaches machine zero. Compare against ReLU MLP baseline, which can solve XOR but typically doesn't reach exact zero.

**T6 — T=4 parity.** Direct analog of the existing KSQP parity test. Predicted result: raw basis with sufficient `K` can express parity exactly. If smoothed basis cannot, that quantifies the regularity-vs-discreteness tradeoff.

**T7 — Smooth function approximation.** Approximate a known smooth periodic function (e.g., `f(x) = exp(sin(x)) + sin(2x)cos(3x)`) with both raw and smoothed bases at varying `K`. Measure approximation error as function of `K`. Predicted result: smoothed basis converges faster on smooth targets; raw basis plateaus at piecewise-linear approximation error.

**T8 — Crossover characterization.** Construct a parameterized family of targets that interpolates between exact-rational (parity-like) and smooth (sinusoid-like). For each target in the family, compare raw vs. smoothed basis convergence. Predicted result: a crossover point where the bases tie; characterize its location as a function of target regularity.

### 4.3 Generalization tests

**T9 — Fourier-class generalization.** Train on a smooth periodic target, evaluate generalization to held-out points. Compare against (a) plain Fourier features + MLP, (b) ReLU MLP. Predicted result: smoothed basis matches Fourier generalization up to constants; raw basis matches ReLU MLP behavior. The interesting case is intermediate smoothing.

**T10 — PDE operator learning.** Pick a simple parametric PDE (1D heat equation, 1D Burgers' equation) with periodic boundary conditions. Build an FNO-style architecture using the harmonic piecewise basis as the lifting. Compare against vanilla FNO. Predicted result: comparable accuracy; the question is whether exact-arithmetic crystallization at the end of training gives an exact-rational solution where one exists.

### 4.4 Training dynamics tests

**T11 — Optimizer behavior on raw vs. smoothed basis.** Same target, same architecture, varying `φ`. Measure (a) training loss curve, (b) gradient norm over training, (c) parameter trajectory length. Predicted result: raw basis has higher gradient variance and rougher loss curve; smoothed basis is smoother. Use this to validate the "cliff vs. hill" framing.

**T12 — Regularity annealing.** Start with smoothed `φ`, anneal toward `φ = δ` during training. Compare against fixed-`φ` training at both endpoints. Predicted result: annealing matches or beats fixed-smooth on smooth targets and matches or beats fixed-raw on discrete targets. If both, the annealing protocol is a robust default.

**T13 — Layer-heterogeneous kernels.** Multi-layer architecture, different `φ` per layer. Initial setting: smoother kernels in early layers, sharper in later layers. Compare against uniform-`φ` baseline. Predicted result: heterogeneous wins on tasks with mixed structure (e.g., smooth input features → discrete output).

### 4.5 Trainable kernel test (stretch goal)

**T14 — Learnable φ.** Make `φ` a piecewise-linear bump with trainable rational breakpoints, one per layer. Train end-to-end. Predicted result: the learned `φ` reflects the regularity of the data in that layer. If the network is presented with smooth targets, `φ` should converge to a smooth bump; on discrete targets, toward a δ-like spike. This is a diagnostic: post-hoc inspection of learned `φ` tells you what regularity the data wanted.

## 5. Falsification Criteria

The construction should be considered falsified or significantly limited if:

- **F1:** Gradient verification (T3) fails — implementation bug, fix before proceeding.
- **F2:** XOR (T5) cannot be solved exactly with raw basis — the "exact-rational basin" claim is wrong; the basis is no better than a ReLU MLP at exact tasks.
- **F3:** Smooth-function approximation (T7) shows no advantage for smoothed basis over raw — the regularity knob is not buying what's claimed.
- **F4:** Crossover (T8) is monotone (one basis always wins) — the dual-regime framing is wrong; pick one regime and commit.
- **F5:** PDE operator learning (T10) is significantly worse than vanilla FNO — the basis is not competitive in the FNO niche, which is the strongest existing reference point.

Any of F1–F5 narrows the claim. F2 or F3 individually would be substantial — F2 means the discrete-task advantage is illusory; F3 means the smooth-task advantage is illusory. F4 or F5 are softer — they restrict the use case but don't kill the construction.

## 6. Notes on Existing Work

For positioning when writing up:

- **Fourier Neural Operators (Li et al. 2020):** the closest reference architecture. Uses floating-point DFT in the spectral lift. This construction differs by being exact-rational and ReLU-native at the sharp limit.
- **B-splines:** share the piecewise-polynomial structure and regularity tunability, but are not frequency-organized as a harmonic basis. Closer cousin than Fourier features in some ways.
- **Wavelets (Daubechies in particular):** share the (function, derivative) duality and tunable regularity via multiresolution analysis. Differ in being localized rather than periodic, and in having algebraic-irrational coefficients rather than rational.
- **Fourier features for coordinate networks (Tancik et al. 2020):** show that frequency-organized features are necessary for high-frequency content in coordinate MLPs. This construction is in the same family of methods but uses piecewise carriers.

The novel combination is: exact-rational + frequency-organized + tunable-regularity + derivative-paired. None of the existing references hit all four.

## 7. Open Design Questions

To resolve during implementation:

- **Q1:** Period parameterization — `2π/k` (Fourier convention) or `1/k` (unit convention)? Probably unit, since exact rationals.
- **Q2:** Amplitude parameterization — fixed at 1, fixed at `1/k`, or trainable? Probably trainable as part of readout weights, with basis at fixed amplitude.
- **Q3:** Frequency selection — `k = 1, 2, ..., K` (linear) or `k = 1, 2, 4, ..., 2^K` (dyadic)? Dyadic gives Walsh-like structure and aligns with multiresolution analysis; linear is the natural Fourier analog. Probably support both.
- **Q4:** Multi-dimensional input — per-dimension lifting (independent harmonics per axis) or joint (tensor product of harmonics across axes)? Per-dimension is cheaper; joint is more expressive. Start per-dimension.
- **Q5:** Phase parameter — fix phase by construction (as specified above) or make it trainable? Trainable phase adds a degree of freedom per basis element; may help on tasks with arbitrary periodic structure. Probably start fixed, add later if needed.

# KSQ Iter 7: The Elevator Mechanism (Corrected)

## Status

**This plan was not the direction taken.** The actual iter-7 work
landed on two different architectures:

- **`PowerLevelModel`** in `sibarum.strnn.ksq.elevator`: signed-power
  activation $y_i = \text{sign}(\ell_i) \cdot |\ell_i|^n$ with a
  continuous learnable scalar $n$. Polynomial degree of $S = Q^2$ is
  $2n$, continuous in $n$. Falsified in its free-$n$ form (T=4 / T=8
  fail because $n$ drifts down, not up); confirmed in its frozen-$n$
  form at $n=2$ (T=4 solves 10/10).
- **`KSQP`** in `sibarum.strnn.ksqp`: discrete-degree control via
  null-cone events on the split-quaternion parameters. Different
  algebra (4-component split quat with sandwich operator), different
  aggregation (split-quat product across tokens, or soft-attention KV
  pool). Works on XOR variants; fails on the 8-cluster circle.

See iter-7 section in `docs/16-ksq-substrate.md` for the consolidated
write-up. The plan in `docs/KSQP.md` is the proposal that became
`KSQP`.

The iterated-squaring picture below remains a coherent third
candidate that wasn't built. It supersedes the iter-6 plan
(`ksq_iter6_elevator_plan.md`), which proposed magnitude scaling as
the level-shifting mechanism. That mechanism was falsified empirically
— see `Iter 6 — the elevator arc, falsified` in
`docs/16-ksq-substrate.md`. The underlying picture (algebra with
parabolic channel adapting computation depth) survives in a corrected
operational form.

The communication error: "magnitude" was read as $\|\ell\|$ (vector norm), when the intended meaning was "exponent of the bilinear step" (the power $n$ in $Q^n$). Scalar amplitude doesn't change polynomial degree; iterated squaring does. The mechanism in this plan is iterated squaring, not magnitude scaling.

---

## Premise

Phase 11's KSQ implements a single bilinear step $S = Q \cdot Q$, fixed at degree 2 in the embedding $\ell$. The iter-6 falsification confirmed that no parametrization change to a *single* bilinear step can break the degree-2 ceiling — higher polynomial degree requires actual iterated application of the bilinear step.

**Iterated squaring is not chained sequence composition.** Iter 1 rejected chaining $Q$'s across sequence positions ($S_T = Q_T \cdots Q_1$) because that made the architecture transformer-shaped. Iterated squaring is different: it operates on a *single* $Q$ derived from a single pooled embedding, raising its polynomial degree by repeated self-multiplication. There is no sequence composition; the sequence is pooled to $\ell$ before any squaring happens.

The architecture's expressivity at depth $z$ (number of squarings) is degree $2^z$ in $\ell$:

| Depth $z$ | Operator | Polynomial degree in $\ell$ |
|-----------|----------|---------------------------|
| 0 | $Q$ | 1 (linear) |
| 1 | $Q^2$ | 2 (current iter-5 KSQ) |
| 2 | $Q^4 = (Q^2)^2$ | 4 |
| 3 | $Q^8 = ((Q^2)^2)^2$ | 8 |
| $z$ | $Q^{2^z}$ | $2^z$ |

The polynomial degree doubles per squaring step, so $z$ levels of iteration give degree-$2^z$ expressivity from a fixed architecture by varying only the iteration count.

The depth $z$ is determined per-input by the parabolic channel: $K_\infty$-aligned signal in the trained embedding drives additional squarings. Easy inputs use few squarings; hard inputs use many. The gradient flows through the depth-selection mechanism via differentiable mixture (or differentiable halting), so the depth selection is itself trained.

---

## What the iter-6 falsification established

Three results carry forward into iter 7:

**Result 1: $\rho_\infty(t)$ is a real signal.** Phase 0's diagnostic measurement under iter-5 KSQ confirmed that the gradient persistently aligns with $K_\infty$ during training (Outcome A; seed 7 zero sign-flips over 4000 epochs). The architecture has been asking for *something* in the parabolic direction. Iter 6 misidentified that something as "more magnitude"; iter 7 reads it as "more squarings."

**Result 2: Magnitude scaling doesn't change polynomial degree.** Phases 2–4 of iter 6 confirmed that increasing $\|\ell\|$ scales $Q^2$ quadratically but does not raise the polynomial degree of the readout features. T=4 and T=8 remained unsolvable across all magnitude regimes. The single bilinear step is fundamentally degree-2; no parametrization rescues it.

**Result 3: The fifth anchor ($K_{e_-}$) participates in trained basins.** Phase 3 added the placeholder $K_{e_-} = e_- = \bigl[\begin{smallmatrix}0&0\\0&1\end{smallmatrix}\bigr]$. It appeared in multiple trained basins at LR=0.1, indicating the architecture uses the extended anchor set. This survives into iter 7 unchanged; the anchor extension is independent of the depth mechanism.

**Bonus positive result from iter 6 Phase 1.** Unit-norm direction + scalar magnitude as a head feature gave 10/10 on XOR (vs. iter-5's 6/10 at λ=1.0). This wasn't elevator dynamics, but it was a real architectural improvement worth preserving. Phase 1's parametrization (unit-norm direction, magnitude exposed at the head) is retained in iter 7 alongside the iterated-squaring mechanism.

---

## The Four Mirror Calculus Regimes (Restated)

| Regime | Generator | $\delta$ | Vector | Role | Status |
|--------|-----------|----------|--------|------|--------|
| Hyperbolic | $j$ | $+1$ | $(1, 1)$ | Boost / polynomial growth | $K_j$ implemented |
| Parabolic | $\varepsilon$ | $0$ | $(0, 1)$ | Integration / level-raising | $K_\infty$ implemented |
| Elliptic | $\eta$ | $-1$ | $(-1, 1)$ | Rotation / oscillation | $K_i$ implemented |
| Projective | $k$ | $\omega$ | $(1, 0)$ | Differentiation / level-lowering | $K_{e_-}$ placeholder (iter 6 phase 3); true projective requires Traction |

The parabolic anchor's role under the corrected mechanism: the $K_\infty$ component of $Q$ controls how many squarings are applied. Strong $K_\infty$ alignment in the trained embedding triggers more iterations; weak alignment uses fewer. The anchor's role is *gating depth*, not *carrying level information directly*.

---

## The Iterated Squaring Architecture

### Forward pass

Given pooled embedding $\ell_{\text{seq}} \in \mathbb{R}^5$ (now 5-dimensional with the $K_{e_-}$ anchor included):

1. Split into direction and magnitude: $\hat\ell = \ell_{\text{seq}} / \|\ell_{\text{seq}}\|$, $r = \|\ell_{\text{seq}}\|$.
2. Lift: $Q_0 = \sum_i \hat\ell_i K_i$. (Direction selects the algebra element; magnitude $r$ is a separate head input.)
3. Iterate squaring up to maximum depth $z_{\max}$: $Q_{k+1} = Q_k \cdot Q_k$ for $k = 0, 1, \ldots, z_{\max} - 1$.
4. Compute depth-mixing weights $w_z$ from $\hat\ell$'s parabolic component (see "Depth selection" below).
5. Mixed output: $S = \sum_{z=0}^{z_{\max}} w_z \cdot Q_z$.
6. Project: $\beta_i = \langle S, K_i \rangle_F / \|K_i\|_F^2$ for each anchor.
7. Readout: $\text{logits} = W \cdot [\beta; r] + b$. (Magnitude $r$ is concatenated to $\beta$ as a head feature, per iter-6 Phase 1.)

The bilinear step is now $z_{\max} + 1$ matmuls instead of one. Cost scales linearly with $z_{\max}$.

### Depth selection

The depth-mixing weights $w_z$ are computed from the parabolic component of the trained embedding. The simplest form:

$$w_z = \text{softmax}_z(\gamma \cdot \langle \hat\ell, K_\infty^{\text{dir}}\rangle \cdot z + \text{bias}_z)$$

where $K_\infty^{\text{dir}}$ is the unit direction in $\hat\ell$-space corresponding to the parabolic anchor, $\gamma$ is a learned scalar (or fixed), and $\text{bias}_z$ are per-level learned offsets initialized to favor $z = 1$ (matching iter-5 behavior at initialization).

This is *differentiable end-to-end*. The depth selection is itself trained via the standard loss. No discrete depth choice, no straight-through estimator, no halting probability — just a soft mixture whose weights depend on the parabolic alignment of the embedding.

**Why this is differentiable:** $\hat\ell$ depends smoothly on $\ell_{\text{seq}}$ (modulo the unit-norm projection, which is smooth away from zero). The parabolic alignment is a smooth function of $\hat\ell$. The softmax over $z$ is smooth. The mixture is a smooth combination of the iterated $Q_z$ matrices. The gradient flows through everything.

### Alternative depth-selection mechanisms (deferred)

If the soft-mixture form proves too compute-wasteful at scale (since it computes all $Q_z$ for all inputs), the architecture can later be upgraded to:

**(a) Adaptive halting:** apply $Q_{k+1} = Q_k \cdot Q_k$ iteratively, with a per-step halting probability driven by the parabolic component. Stop when halting fires. Standard adaptive-computation-time training (PonderNet-style).

**(b) Discrete depth via Gumbel-softmax:** sample a depth at each forward pass, anneal the temperature toward hard sampling. Standard discrete-categorical-routing training.

**(c) Algebraic threshold:** when the parabolic component of $Q_k$ exceeds an algebra-determined threshold (related to null-cone proximity), increment $k$. No learned threshold; the algebra's geometry decides.

(c) is the most elegant but the hardest to implement. (a) is intermediate. The soft-mixture form is the simplest and is the recommended starting point. Upgrade only if needed.

### Backward pass

Standard autograd through the iterated bilinear steps and the mixture. The bilinear backward for $S = Q^2$ is:

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S} \cdot Q^\top + Q^\top \cdot \frac{\partial \mathcal{L}}{\partial S}$$

Iterating this through the squaring chain gives the gradient at $Q_0$, which then flows back through the lift and pooling to the embedding table.

**Gradient check requirement:** the new bilinear chain (multiple squarings + mixture) needs finite-difference verification before any training. This is iter-7's risky-piece in the gradient-check discipline. Expected max-abs-err at machine epsilon as in previous iterations.

---

## Predictions

### Primary predictions (T=2, T=4, T=8 parity)

**P1: T=2 XOR solves with effective $z \approx 1$.** The depth mixture concentrates on $w_1$ (corresponding to $Q^2$, the iter-5 operator). Mean $w_1 > 0.7$ across trained seeds. Solve rate ≥ iter-5 (i.e., ≥ 10/10 at favorable $(\lambda, \nu)$).

**P2: T=4 parity solves with effective $z \approx 2$.** The depth mixture shifts to concentrate on $w_2$ (corresponding to $Q^4$, degree-4 expressivity). Mean $w_2 > 0.5$ across trained seeds. **Solve rate substantially > 0** (vs. iter-6's 0/10 falsification).

**P3: T=8 parity solves with effective $z \approx 3$.** Mixture concentrates on $w_3$. **Solve rate substantially > 0**.

**P4: The mixture weights cluster by task.** Different seeds on the same task converge to similar mixture weights. The cluster's center varies systematically with task degree: T=2 → mean $z = 1$, T=4 → mean $z = 2$, T=8 → mean $z = 3$ (or close).

### Secondary predictions (training dynamics)

**P5: $\rho_\infty(t)$ during training shows level-transition signature.** For T=4 specifically:
- Early training: $\rho_\infty$ large positive (gradient asking for more squarings).
- Mid training: $\rho_\infty$ peaks around the time the mixture weight shifts from $w_1$-dominated to $w_2$-dominated.
- Late training: $\rho_\infty$ small (architecture has settled on the right depth).

For T=8, two such peaks corresponding to $w_1 \to w_2$ and $w_2 \to w_3$ transitions.

**P6: Ablating depth-mixing forces the falsification to recur.** If we set $w_z = \mathbb{1}[z = 1]$ (force $z = 1$ always, recovering iter-5), T=4 and T=8 become unsolvable again. This verifies that depth-adaptivity is what makes the higher-degree tasks solvable.

**P7: Per-token depth varies.** In multi-token sequences, different tokens within the same sequence can drive different effective depths. (This is implicit in the architecture — depth is a function of the *pooled* embedding, so tokens whose embeddings drive parabolic alignment when pooled produce higher depth.)

### Falsification criteria

The corrected mechanism is falsified if:

- **F1: T=4 parity remains unsolvable** with the iterated-squaring architecture across reasonable $z_{\max}$ values (4 or higher) and reasonable training budgets. The depth-adaptivity isn't the right mechanism; something else is needed.
- **F2: T=4 solves but mixture weights don't cluster at $z = 2$.** The architecture is solving T=4 through some artifact (extra parameter capacity, the magnitude head feature, etc.), not through actual iterated squaring. The depth mechanism isn't doing what it claims.
- **F3: The ablation (P6) doesn't recover the iter-6 falsification.** Forcing $z = 1$ should make T=4 unsolvable. If it remains solvable, the depth-adaptivity isn't the source of the improvement and the test isn't isolating what it claims to isolate.

Any of F1, F2, F3 would force structural rethink. F1 is the strongest falsifier; F2 and F3 are diagnostic confirmations that the right mechanism is being measured.

---

## Implementation

### Files

New package: `sibarum.strnn.ksq.iterated`. Mirror the existing `sibarum.strnn.ksq` layout.

| File | Purpose |
|------|---------|
| `iterated/KsqIteratedAnchors.java` | Five anchors (K_0, K_i, K_j, K_∞, K_e_-), Frobenius norms precomputed |
| `iterated/KsqIteratedEmbeddingTable.java` | Embedding `vocab → R^5` logits, unit-norm + magnitude split |
| `iterated/KsqIteratedDepthMixer.java` | Computes depth mixture weights $w_z$ from parabolic alignment |
| `iterated/KsqIteratedOutputHead.java` | Linear $[β; r] →$ logits, accommodating concatenated magnitude |
| `iterated/KsqIteratedModel.java` | Orchestrator: forward (with iterated squaring), backward (chain through multiple matmuls + mixture), regularizers |
| `demo/KsqIteratedGradientCheckDemo.java` | Finite-difference verification of CE, both regularizers, and the iterated-squaring chain |
| `demo/KsqIteratedParityDemo.java` | T=2, T=4, T=8 sweep with depth mixture reporting |
| `demo/KsqIteratedAblationDemo.java` | Forced-$z=1$ ablation to verify P6 |
| `demo/KsqIteratedRhoInfinityDemo.java` | Measures $\rho_\infty(t)$ during training, verifies P5 |

iter-5 KSQ in `sibarum.strnn.ksq` is unchanged. iter-6's `sibarum.strnn.ksq.elevator` is preserved as the falsification record.

### Pre-training checks

Before any training run:

1. **Gradient check.** Finite-difference verification of every backward pathway, target max-abs-err ≤ 1e-10. The risky-piece for iter 7 is the iterated-squaring chain — the bilinear backward composes with itself $z_{\max}$ times, and the mixture's backward propagates gradient to all $Q_z$ simultaneously. Test at $z_{\max} = 3, 4, 5$ to verify the chain-of-squarings is correctly differentiated.

2. **Initialization check.** Mixture weights $w_z$ should initialize to favor $z = 1$ (so the architecture starts at iter-5-equivalent behavior). Verify that initial forward passes on XOR produce iter-5-like outputs, then check that gradient flow can move the mixture away from $z = 1$ when the task demands.

3. **Ablation check.** Confirm that forcing $w_z = \mathbb{1}[z = 1]$ recovers iter-5 behavior exactly. This sanity-checks the implementation: if forced-$z=1$ produces different XOR results than iter-5, the implementation has a bug unrelated to the depth mechanism.

### Training protocol

For each task (T=2, T=4, T=8):

1. Run 10 seeds at the iter-5-favorable $(\lambda, \nu) = (0.0, 0.0)$ first. This is the cleanest test of whether the depth mechanism alone (no regularization) can solve higher-degree tasks.
2. Run the full $(\lambda, \nu)$ sweep if (1) shows positive results. The regularizers may need re-tuning under the new architecture.
3. For each seed, log:
   - Final solve rate (raw and non-trivial).
   - Final mixture weights $w_z$.
   - $\rho_\infty(t)$ trajectory across training.
   - Mean and variance of mixture weights across seeds.
4. Run the ablation (P6) on T=4: force $w_z = \mathbb{1}[z = 1]$ and verify failure recurs.

### Cost estimate

Implementation: ~1 week for the iterated-squaring architecture and the four new demos. Half of that is gradient-check work for the multi-matmul chain.

Training: each XOR sweep at iter-5 cost ~minutes per seed; iter-7's added cost is $z_{\max}$ × the bilinear step, so ~5× iter-5 at $z_{\max} = 5$. T=4 and T=8 add longer training trajectories. Estimate full prediction battery at a few hours of compute total, well within consumer-hardware budget.

---

## What's New vs. iter 6 plan

The iter-6 plan proposed magnitude scaling as the level mechanism. The iter-7 plan proposes iterated squaring with parabolic-driven depth mixing.

| Aspect | iter 6 plan | iter 7 plan |
|--------|-------------|-------------|
| Level mechanism | Magnitude $\|\ell\|$ scales output | Number of squarings of $Q$ |
| Polynomial degree | Capped at 2 (single bilinear step) | $2^z$ where $z$ is depth |
| Depth selection | Implicit via magnitude | Explicit via mixture weights, driven by parabolic alignment |
| Falsification mode | Magnitude grew but degree didn't | (To be tested — falsified if F1, F2, or F3 fires) |
| Implementation | Remove tanh, add magnitude-aware head | Iterate the bilinear step, mix over depths |
| Predicted T=4 result | Unsolvable (correct, falsified the plan) | Solvable (the actual claim being tested) |

The iter-6 phase-1 result (unit-norm direction + magnitude as head feature) is retained in iter 7 — it's a real improvement orthogonal to the depth mechanism, and there's no reason to discard it.

The fifth anchor ($K_{e_-}$) from iter-6 phase 3 is retained. The depth mechanism is independent of the anchor set; both improvements layer cleanly.

---

## Why This Survives the iter-6 Falsification

The iter-6 falsification showed that magnitude scaling is the wrong mechanism. It did *not* show that:

- Adaptive depth is the wrong concept. (The picture survives; only the mechanism was wrong.)
- The parabolic channel is uninvolved. (Phase 0's $\rho_\infty$ signal was real; iter 7 gives it a concrete operational role.)
- KSQ has a hard ceiling at degree 2. (Single bilinear step does; iterated squaring doesn't.)
- The substrate-as-primitive case is wrong. (Phase 11's organization/retention/attraction results are unchanged.)

What it did show: any mechanism that doesn't increase the *polynomial degree as a function of $\ell$* cannot raise the expressivity ceiling. Iterated squaring increases the polynomial degree explicitly ($Q^{2^z}$ is degree $2^z$ in $\ell$). The mechanism now matches what the falsification showed was necessary.

---

## Open Questions

1. **What's the right $z_{\max}$?** Too small caps expressivity; too large wastes compute. Start with $z_{\max} = 5$ (covering up to degree-32 polynomials) and adjust based on observed mixture distributions.

2. **Does the soft-mixture form give clean cluster results, or does it diffuse the depth across multiple $z$ values?** If the mixture stays spread across $z = 1, 2$ on a T=4 task rather than concentrating at $z = 2$, the prediction P4 fails partially. Worth measuring.

3. **What is the relationship between the depth mixer's parameters and the existing per-vocab / cross-vocab regularizers?** The mixer adds learnable scalars; these may interact with the regularizers in ways that change the optimal $(\lambda, \nu)$. Re-tuning may be needed.

4. **Does the discrete-depth variant (Gumbel-softmax or adaptive halting) behave differently from the soft mixture?** If so, that's diagnostic about whether the architecture wants soft routing or hard routing. Soft mixture first; discrete variants are deferred.

5. **How does the projective anchor (true one, via Traction) fit into the iterated-squaring picture?** Once the depth-adaptivity mechanism is verified, the projective channel should provide the level-*lowering* counterpart — but exactly how that interacts with iterated squaring (does it un-square? halt early? something else?) is an open question for iter 8+.

---

## Sequencing

In order:

1. **Implement iterated squaring with soft mixture.** ~1 week, including gradient-check work.
2. **Verify on T=2 XOR.** Confirm iter-5-equivalent behavior when mixture concentrates at $z = 1$. (P1)
3. **Run T=4 parity.** The make-or-break experiment. Either P2 confirms (architecture works) or F1 fires (mechanism is still wrong).
4. **If P2 confirms, run T=8 parity.** Confirm P3. Scaling to higher degrees.
5. **Ablation studies.** Force $z = 1$, verify P6. Confirms the depth mechanism is doing the work, not auxiliary capacity.
6. **$\rho_\infty(t)$ measurement.** Verify P5. Confirms the parabolic channel is the driver.
7. **Document.** Update `docs/16-ksq-substrate.md` with iter-7 results, alongside the iter-6 falsification.

If F1 fires (T=4 unsolvable with iterated squaring), the architecture's expressivity ceiling is deeper than the bilinear step's degree. Possible structural rethinks at that point: change the anchor set, change the readout, change the algebra entirely, or admit that single-pool single-bilinear architectures are degree-bounded regardless of iteration.

If P2 confirms but P4 fails (T=4 solves but mixture doesn't cluster at $z = 2$), the architecture is solving T=4 through some other mechanism than depth-adaptivity. Diagnostic work to identify the actual mechanism.

If P2 and P4 both confirm, the elevator picture is empirically grounded and the architecture is depth-adaptive. The morning's prediction lands, late by one iteration, but lands.

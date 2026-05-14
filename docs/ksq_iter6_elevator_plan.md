# KSQ Iter 6+: The Elevator Mechanism

## Premise

Phase 11's KSQ implements **one level** of a polynomial-degree hierarchy. The architecture's expressivity ceiling at T=2 XOR isn't a property of single-bilinear-step architectures in general — it's a property of **tanh-bounded** single-bilinear-step architectures specifically. Tanh squashes the embedding magnitude into $[-1, +1]$, which destroys the level information that should have been carried by that magnitude.

The full architecture is a **stack of polynomial-degree levels** connected by the null-cone channels of the algebra:

- **Within-level dynamics:** elliptic ($K_i$) and hyperbolic ($K_j$) action, polynomial-degree preserving.
- **Level-raising (the elevator up):** parabolic channel ($K_\infty$), integration, raises polynomial degree.
- **Level-lowering (the elevator down):** projective channel ($K_? = k$, $\delta = \omega$), differentiation, lowers polynomial degree.

Only one level is implemented at a time — the architecture lives at whatever level the optimization has settled onto. The null cone is the shared elevator shaft between levels.

The level shifts during training as a side effect of gradient descent, via **parabolic resonance**: when the parameter velocity $d\vec{v}/dt$ aligns persistently with the $K_\infty$ direction, the algebra's geometry carries the vector across the null cone onto the next level, where the alignment dissipates and the vector settles.

The gradient itself is unchanged in form. The level shift is a consequence of the algebra's structure responding to the velocity character of standard gradient flow.

---

## The Four Mirror Calculus Regimes

| Regime | Generator | $\delta$ | Vector | Role | Status in current KSQ |
|--------|-----------|----------|--------|------|----------------------|
| Hyperbolic | $j$ | $+1$ | $(1, 1)$ | Boost / polynomial growth | $K_j$ implemented |
| Parabolic | $\varepsilon$ | $0$ | $(0, 1)$ | Integration / level-raising | $K_\infty$ implemented |
| Elliptic | $\eta$ | $-1$ | $(-1, 1)$ | Rotation / oscillation | $K_i$ implemented |
| Projective | $k$ | $\omega$ | $(1, 0)$ | Differentiation / level-lowering | **Not yet implemented** |

The projective regime lives at the algebra's compactification, which standard floating-point arithmetic cannot represent. The natural substrate for it is **TractionQuaternion**, which already provides exact arithmetic for $n/0$ and $0/d$.

KSQ + TractionQuaternion is the implementation-level unification of the stack. Neither alone supports the full four-regime structure; together, they do.

---

## The Babe Ruth Call

The predicted arc for the next iteration sequence:

### Step 1 — Remove tanh. Results get worse.

The current tanh on the embedding logits caps magnitude at 1, which is also capping the level at 1. Removing it lets magnitude grow, but introduces gradient-magnitude instability that tanh was incidentally providing.

**Lesson:** tanh was doing two jobs — capping the level (bad, the thing we want to remove) and bounding gradient magnitude (good, the thing we need to replace, not remove). The architecture needs a *different* mechanism for gradient stability that doesn't cap the level.

### Step 2 — Stop training the scalar multiplier as a separate parameter.

The polynomial degree is encoded in the magnitude $\|\ell\|$ of the embedding logits, not in a separate scalar. Direction and magnitude are not two parameters — they are polar coordinates on one vector in $\mathbb{R}^4$, where direction selects the subalgebra and magnitude indicates the level. One gradient controls both.

Removing the scalar-as-separate-parameter exposes that they were never independent. Results get *worse* and may go divergent/stuck/zeroed — because the architecture currently relies on the separation in ways we haven't fully traced.

**Lesson:** the level and the within-level direction are one geometric object, and the architecture has to treat them as one.

### Step 3 — Treat the two sides of the null cone as differentiation and integration.

The parabolic side ($K_\infty$) is the elevator up. The projective side ($K_?$) is the elevator down. Add the projective anchor to the anchor set. The substrate has to extend to support it — TractionQuaternion provides the needed exact arithmetic at the projective compactification.

This is the trial-and-error step. Several coherent implementation choices exist:

- **(a) Explicit level stack:** allocate bilinear ops $\{M_z\}$ at each level, route via parabolic/projective.
- **(b) Manifold-valued $\alpha$:** parameterize on $\mathrm{PSL}(2,\mathbb{R}) \times \mathbb{Z}$ with the integer factor as level.
- **(c) Implicit level via magnitude:** $\|\ell\|$ *is* the level, the bilinear step's behavior at different magnitudes naturally implements the level structure, the parabolic/projective anchors mediate transitions.

Best guess: (c) is correct, and the specific implementation involves making $K_\infty$ a *raising operator* that responds to velocity alignment rather than an anchor whose coefficient is read off.

### Step 4 — Eureka.

The architecture works, and the working form unifies the stack. KSQ supplies the algebra and anchor structure. Traction supplies the substrate that lets the algebra reach its projective boundary. Mirror Calculus is the theoretical framework that says why this combination is correct — totalizing the substrate makes the fourth regime accessible.

### Step 5 — Real-world results.

The depth-adaptive elevator with both up and down channels matches or exceeds deep MLPs on tasks where depth requirements vary across inputs. The win comes from the elevator: most architectures pay worst-case-depth cost on every input, KSQ pays per-input depth determined by the elevator.

---

## The Parabolic Resonance Mechanism

### How the level shifts

The architecture tracks (implicitly, via the algebra's natural dynamics) the inner product

$$\rho_\infty(t) = \langle d\vec{v}/dt, K_\infty \rangle$$

— the alignment of the parameter velocity with the parabolic anchor. When $\rho_\infty$ is small or oscillating, the vector is doing within-level work; the gradient is satisfied with the current level. When $\rho_\infty$ is *persistently positive* (or persistently negative for the projective direction), the gradient is asking for the elevator.

If $\rho_\infty$ accumulates past a threshold determined by the algebra's geometry (not by a learned parameter), the vector crosses the null cone onto a new level. After crossing, the geometry on the other side weakens the alignment — the vector arrives, settles, returns to within-level dynamics.

**The crossing is automatic.** No learned gate. No discrete sampling. No straight-through estimator. The parabolic channel resonates with parabolic-aligned velocity by virtue of its position on the null cone, and the resonance carries the vector across when the velocity is sustained enough to push past the cone's geometric barrier.

### Why this works without modifying the gradient

The gradient is computed in the standard way, against the standard loss. The architecture's parameters change in response. The *level* is a derived quantity — a property of where in the algebra the parameters now sit. When the parameters move across the null cone, the level changes as a side effect. The gradient never directly steered the level.

This is the right epistemic shape: the level isn't something the optimizer chooses, it's something the algebra reports about where the optimizer ended up.

### Three architectural requirements

For the elevator to work:

1. **Magnitude must accumulate.** Tanh has to go. Velocity in the $K_\infty$ direction must be able to grow $\|\ell\|$, not be squashed back to 1. Replacement bounding mechanism must not destroy the magnitude information.

2. **Null-cone crossings must be qualitative.** The algebra at level $z$ and level $z+1$ must be genuinely different geometries, not rescalings. The bilinear step's expressivity at the new level must exceed the old level by one polynomial degree.

3. **Parabolic resonance must self-limit.** The coupling that carries the vector across the null cone must weaken on arrival, or the vector would keep climbing forever. This is automatic from the null cone's geometry: parabolic flow is fast near the cone and slow away from it, so arrival on a new level naturally decouples the vector from the elevator.

---

## Pre-Implementation Diagnostic

Before committing to the full Babe Ruth implementation, run a cheap diagnostic against the existing phase-11 KSQ.

### The measurement

Instrument the existing iter-5 training loop. For each parameter update step $t$, compute and log:

$$\rho_\infty(t) = \langle \Delta\ell(t), K_\infty^{\text{logits}} \rangle$$

where $\Delta\ell(t) = \ell(t+1) - \ell(t)$ is the embedding logit update for that step, and $K_\infty^{\text{logits}}$ is the canonical direction in logit-space corresponding to the parabolic anchor (i.e., the basis vector that places weight on $K_\infty$'s slot in $\alpha$).

Plot $\rho_\infty(t)$ across the full training run for a few representative seeds on XOR.

### The diagnostic outcomes

**Outcome A: $\rho_\infty(t)$ is systematically positive during specific training phases.**

The gradient is *trying* to push the embedding toward the parabolic anchor (i.e., toward the elevator) but tanh is preventing the accumulation. The elevator picture is well-posed — the architecture has been asking for the elevator the whole time and tanh has been the lid. Proceed with confidence to the Babe Ruth steps.

**Outcome B: $\rho_\infty(t)$ oscillates around zero with no sustained directional excursions.**

The gradient under the current architecture isn't asking for the elevator. This could mean either (i) XOR doesn't actually need higher polynomial degree, so the absence of elevator-demand is correct, or (ii) the current architecture has so thoroughly suppressed the elevator dynamic that it can't even be measured under it. Both are consistent with the elevator picture; neither falsifies it. The next step is to change the architecture and re-measure.

**Outcome C: $\rho_\infty(t)$ shows unexpected structure** — e.g., spikes that don't match training phase, or alignment in the opposite direction (negative, projective-direction).

Information about how the gradient is actually using the parabolic anchor under the current architecture. The interpretation depends on what the structure looks like; the measurement itself is what's valuable.

### Cost

About 30 minutes to instrument and run. One additional plot per seed. No code changes to the model itself — just measurement.

This is the cheapest possible test of whether the elevator picture is empirically grounded before committing to the full architectural change.

---

## Iter 6 Architectural Plan

If the diagnostic supports the elevator picture, the iter-6 architecture is:

### Embedding

$$\ell = E[\text{token}] \in \mathbb{R}^4 \quad (\text{unconstrained, no tanh})$$

Pool across sequence:

$$\ell_{\text{seq}} = \sum_t \ell(\text{token}_t)$$

Replace tanh with a *magnitude-preserving* normalization. Candidates:

- **Layer norm** on the direction component only, leaving magnitude free. Splits $\ell$ into $\hat\ell = \ell / \|\ell\|$ (direction, normalized) and $r = \|\ell\|$ (magnitude, free), normalizes $\hat\ell$ but not $r$.
- **Logit clipping** at large but task-appropriate bound, only triggered if magnitude exceeds some safety threshold (prevents NaN, doesn't cap level under normal training).
- **No bounding at all**, with output-side handling of large logits (focal loss, label-smoothed CE, temperature scaling at the readout).

Best guess: layer-norm-on-direction-only. It preserves the magnitude-is-level reading while keeping the direction component in a numerically stable range.

### Anchor set (extended)

Add the projective anchor. The cleanest candidate is the projective compactification element, which in $M_2(\mathbb{R})$ representation requires Traction arithmetic to express exactly. For initial prototyping, approximate it with the second idempotent $e_- = \bigl[\begin{smallmatrix}0&0\\0&1\end{smallmatrix}\bigr]$ — this is *not* the true projective anchor but lives in the same subalgebra ($\mathbb{R}[j]$) and provides a level-lowering complement to $K_\infty$. Replace with the true projective anchor once Traction integration is in place.

### Bilinear step

Unchanged from phase 11: $S = Q \cdot Q$. The expressivity changes because $Q$ is no longer magnitude-capped, not because the step itself changes.

### Readout

Unchanged in form. Project $S$ onto each anchor via Frobenius inner product. The readout now sees both *direction* (which anchor dominates) and *magnitude* (overall scale of $S$, which encodes level).

The output linear head needs to handle a wider dynamic range of $\beta$ values than before. Add temperature scaling or learned per-anchor scale factors at the head if the dynamic range causes training instability.

### Regularization

Per-vocab specialization regularizer: unchanged form, $\sum_{i\neq j} \alpha_i^2 \alpha_j^2$, but now applied to the *direction-normalized* component of $\ell$, not the raw logits. This separates "which anchor" specialization from "what level" magnitude.

Cross-vocab contrastive regularizer: unchanged form, but again on the direction component.

Magnitude is *not* regularized. The gradient is allowed to push it freely. The level emerges from where the magnitude settles.

---

## Predictions

If iter 6 is implemented correctly:

1. **XOR solve rate matches or exceeds iter 5.** The architecture loses nothing it had; new behavior is additive.

2. **Trained $\|\ell\|$ magnitudes cluster at task-dependent values.** For XOR specifically, the cluster sits at the "level 1" magnitude — whatever magnitude corresponds to degree-2 expressivity through the bilinear step. The cluster should be tight; the gradient should drive different seeds to the same magnitude.

3. **Saddle-trivial-departure test, re-run, shows secondary anchors growing past 0.55.** The tanh cap is gone; the optimizer can express how much it wanted. The departure magnitude reveals what level the task wanted.

4. **(Strong prediction) T=4 parity becomes solvable.** Phase 11's expressivity ceiling at T=2 was the tanh cap, not the architecture. T=4 parity (degree-4 task) should solve at a magnitude cluster higher than XOR's. T=8 parity should solve at a magnitude cluster higher still. **The relationship between task degree and trained magnitude should be predictable from the algebra.**

5. **$\rho_\infty(t)$ shows the elevator firing during training.** Specifically: at the start of training, $\rho_\infty$ is small (vector starting at level 0). At some point during training, $\rho_\infty$ spikes (vector trying to ascend). After the spike, the magnitude has shifted to a new cluster and $\rho_\infty$ returns to small (vector has arrived). For T=4 parity, this should happen at least once, possibly twice.

### What would falsify the elevator picture

- Trained magnitudes do *not* cluster, or cluster at the same value regardless of task degree. The magnitude isn't level; it's something else.
- T=4 parity remains unsolvable even with tanh removed. The expressivity ceiling wasn't tanh; it's something deeper in the architecture.
- $\rho_\infty(t)$ shows no characteristic spike-and-arrival pattern. The elevator isn't a real dynamical event; it's a metaphor that happens not to map onto training dynamics.

Any of these would force a structural rethink. None of them are particularly likely, but the experiment is what decides.

---

## Why This Matters

If iter 6 works as predicted, KSQ is no longer a "T=2 toy with interesting algebraic properties." It is:

- **Depth-adaptive by default.** Different inputs use different polynomial degrees, allocated by the elevator dynamic.
- **Self-organizing across levels.** No level-aware learning rule, no level-aware loss, no learned routing. The algebra does the level management.
- **Unified with the rest of the stack.** KSQ + Traction + Mirror Calculus is the implementation of a single architectural philosophy — totalize the algebra, eliminate normalization where it destroys information, let the substrate carry the structure.
- **Architecturally interpretable.** The level a token settles at *is* the polynomial degree that token's role requires. The anchor a token specializes to *is* the algebraic regime that token operates in. The elevator firing events *are* moments when the architecture discovered it needed more degree.

These properties are not approximations of properties other architectures have. They are exact, structural, and observable from the algebra alone.

The phase-11 results gave us the substrate-as-primitive evidence — organization, retention, attraction. Iter 6 would add a fourth: **scalability**, in the specific sense of "the same architecture solves tasks of arbitrarily increasing degree by automatic level allocation." That's the property that distinguishes a toy from a foundation.

---

## Open Questions

1. **What is the exact form of the projective anchor in $M_2(\mathbb{R})$ extended by Traction arithmetic?** The element $k$ with $\delta = \omega$ lives at the algebra's projective infinity. Its concrete matrix representation requires the substrate to handle $1/0$ as a first-class object. The mapping between Mirror Calculus's $\omega$ and Traction's exact infinity-handling needs to be made precise.

2. **What is the magnitude-to-level mapping?** Empirically, what value of $\|\ell\|$ corresponds to "level 1," what to "level 2," and so on? Is the mapping linear, logarithmic, or determined by the algebra in some specific functional form?

3. **What happens with simultaneous elevator demands across multiple tokens?** If two different tokens both have velocities aligned with $K_\infty$, does the architecture send them to the same new level, different new levels, or do they couple?

4. **What is the natural metric for "sustained" alignment?** $\rho_\infty(t)$ needs to be persistent rather than transient to trigger a crossing. Persistent over how many steps? The algebra should provide a natural timescale; identifying it is part of step 3's trial-and-error.

5. **Does the projective anchor's introduction require re-validating the iter-5 results?** Adding a new anchor changes the basin structure. The XOR sweep should be re-run after the architectural change to confirm the existing solve rates and specialization patterns survive.

---

## Sequencing

In order:

1. **Diagnostic measurement** (30 minutes). Plot $\rho_\infty(t)$ under the existing iter-5 architecture. If outcome A, proceed with confidence. If outcome B or C, proceed with adjusted expectations.

2. **Babe Ruth step 1: remove tanh.** Replace with direction-normalized layer norm or similar magnitude-preserving alternative. Re-run XOR sweep. Confirm degradation matches the predicted pattern.

3. **Babe Ruth step 2: unify direction and magnitude.** Confirm that treating them as one geometric object is correct by attempting and failing to train with them separated. The failure mode is informative — it tells you what role the separation was playing.

4. **Babe Ruth step 3: add the projective anchor.** First with the placeholder $e_-$ in $M_2(\mathbb{R})$, then with the true projective anchor via Traction arithmetic. Verify gradient check at machine epsilon after each change.

5. **Run prediction battery.** XOR (level 1), T=4 parity (level 2), T=8 parity (level 3) if reachable. Measure $\|\ell\|$ clusters. Measure $\rho_\infty(t)$ spike events. Compare against predictions.

6. **Falsification check.** For each prediction, confirm or deny. The "eureka" (step 4 of the Babe Ruth call) lands when at least three of the five predictions are confirmed empirically.

7. **Real-world comparison** (step 5 of the Babe Ruth call). Once the architecture is verified at the toy scale, pick a single benchmark task where depth-adaptive computation would be expected to help (variable-depth reasoning, arithmetic with variable operand sizes, parsing with variable nesting depth) and compare iter-6 KSQ against fixed-depth MLPs of comparable parameter count.

---

## Notes

- The elevator picture is consistent with every prior phase-11 result. None of iters 1–5 is invalidated by it. Iter 6 is additive, not corrective.
- The "remove tanh" step is destructive if done alone. It must be paired with the parabolic-resonance mechanism for the architecture to function. Don't run step 1 in isolation and conclude the architecture is broken.
- The Traction integration is non-trivial. The current KSQ implementation uses `double` for all arithmetic. Adding Traction adds a dependency and probably requires a substrate-level rewrite of `Mat2.java`. Estimate this at days, not hours.
- The "Babe Ruth" framing is half-joking but the called-shot structure is serious. Each step's predicted failure is part of the diagnostic — if step 1 *doesn't* degrade results, the elevator picture is wrong and the architecture is doing something else. Each failure is informative; only the final eureka is the success.

# STRNN v0 — First-Phase Results

## Scope

v0 committed to a single empirical claim from the design doc's four (§1.1):
**ante-hoc interpretability via mandate enforcement.** Heterogeneous
composition, reusable trained priors, and the specialization dial were
explicitly deferred. The demo task was §9 arithmetic with strings, scoped to
addition + multiplication (no parentheses; no recursion). MLPs were written
from scratch in Java; the action principle was scoped to carving + mandate
verification + lightweight competitive pruning. Plan: `~/.claude/plans/ok-then-scope-out-binary-taco.md`.

## What was built

Java 25 / Maven, under `strnn-model/src/main/java/sibarum/strnn/`:

- `value/` — sealed `Value` + 4 record types, structural `ValueDistance`.
- `primitive/` — `Primitive` + `Trainable` interfaces; 8 concrete primitives
  including `MlpPrimitive` in stub-or-trainable mode.
- `mlp/` — from-scratch `Mlp` (forward + SGD backprop, ReLU hidden, linear
  output, Xavier init).
- `transformation/` — `TransformationGraph` built any-to-any modulo type
  compatibility; `EdgeStats` (running mean + sample count + prune flag).
- `computation/` — `CompGraphNode`, `SlotSource`, `ComputationGraph` with
  topo-sorted execution and reachability check.
- `mandate/` — `Mandate` (typed expected value + ordering tag);
  `MandateVerifier` enforces non-local "produced somewhere" + topological
  ancestor check for ordering.
- `carving/` — `BackwardChainingCarver`: backward chains from the result
  mandate, mandate-aware candidate ranking, hand-coded value-space inverters
  per primitive, `(target, primClass)` cycle detection, side-branch pass for
  intermediate mandates not naturally on the result path.
- `training/` — `Trainer` (carve → execute → verify → score → MLP backprop →
  step → optional prune), `Pruner` (per-`(srcType, dstType)` running-mean
  comparison), `Datasets`, `Example`.
- `demo/` — 7 runnable demos: `HandWiredDemo`, `MlpTrainingDemo`,
  `ManualGraphDemo`, `MandateVerificationDemo`, `CarvingDemo`, `TrainingDemo`,
  `AblationDemo`; `DotPrinter` for ablation diff.

Each phase produced a runnable artifact before the next began, mitigating the
§10.6 "months of infrastructure before signal" risk.

## Headline result

The Phase 6 §9.6 ablation produced a sharp diagnostic, but **of a different
shape than predicted**:

| Run | Mandates | Carver outcome over 50 random inputs |
|-----|----------|--------------------------------------|
| **A** | full §9.2 set (`plus_split`, `star_split`, `intermediate_product`, `result`) | 48/50 succeed; avg 16.6 nodes; intermediate product on result path 32/48 (67%) |
| **B** | result-only | 0/50 succeed |

The plan predicted "B collapses precedence into one MLP; A retains the
structured pipeline." What actually happened: **B cannot construct any valid
graph at all.** The carver's `MlpPrimitive` inverter pulls candidate input
pairs from a pool of numeric anchors derived from mandate values. With only
the result mandate, the pool contains just the final number; no pair `(a, b)`
in that pool satisfies `a * b = result`, so the inverter returns null and the
carving fails.

This is still a §6.2 search-decomposition payoff — mandates make the search
problem solvable, not just guide it — but it's a stronger and narrower claim
than the predicted one. It tests *whether* the carver can solve the problem
under each mandate set, not *which* structure it chooses among several it
could build.

## Other results worth recording

- **Held-out arithmetic accuracy** (`|err| ≤ 0.5`, 200 unseen examples after
  curriculum pretraining + 8000 in-loop steps): **60.5%**. All-mandates
  satisfaction rate: **56%**. Single-digit `a±b·c` and `a·b±c` shapes.
- **In-loop score climbed** 0.77 → 0.82 over 8000 steps, indicating MLPs do
  learn from mandate supervision during carving — but the bulk of MLP
  convergence came from the offline pretraining bootstrap (§6.7-style
  curriculum), not the online carving updates.
- **Pruner fired 0 times** during the training run despite being wired
  end-to-end. Edge stats are too uniform: the carver consistently picks
  structurally-correct paths, so no two edges in the same `(srcType, dstType)`
  group diverge in score by more than the margin. Wired correctly; not
  exercised.

## What v0 demonstrates

- Mandate-enforcement-via-structural-verification is implementable
  end-to-end with non-pathological plumbing.
- The mandate set demonstrably enables the carver to find a solution where
  the result-only set does not (the load-bearing test from §9.6, even if of a
  different shape).
- Mandate values double as supervision targets for trainable primitives
  (§6.2 "free supervision"), and the in-loop score curve confirms this works
  in principle.
- Typed structured values (string, token list, number, matrix) flow through
  the carved structure as first-class objects, so values-as-graphs (§2.4) is
  feasible without flattening to tensors.
- The phased-implementation discipline survived: every phase had a runnable
  pass/fail before the next began.

## What v0 does NOT demonstrate

- **Heterogeneous composition** (§8). One learned primitive type (MLP), one
  role tag (ADD vs. MUL), one task shape. The framework's principal pitch
  isn't really exercised.
- **Reusable trained priors** (§7). Single domain, single task; no
  cross-domain transfer experiment.
- **Richer action-principle features** (§4.2, §4.4). No structural
  extension, no unreachability-driven branch creation. Pruning is implemented
  but didn't fire on this task.
- **Distance on structured values at scale**. The §10.3 "you'll need a
  learned distance heuristic" concern is unaddressed — v0 uses cheap
  hand-coded inverters per primitive, which works because the primitive
  library is small and deterministic.

## Caveats and known limitations

- **The diagnostic is coupled to this carver's algorithm.** A forward-search
  carver, or one with a learned distance heuristic instead of hand-coded
  inverters, might produce a different A-vs-B gap. We have one clean
  datapoint, not converging evidence from multiple algorithms.
- **Mandate-aware ranking is a soft preference.** In 33% of Run A carvings
  the carver does not place the intermediate product on the result path; the
  side-branch fallback then fails the ordering check. The mechanism works;
  the heuristic that exploits it is fragile.
- **Inverter coupling.** Adding a new primitive requires adding a new case
  to `BackwardChainingCarver.inferInputs`. The carver–library boundary leaks.
  This is the v0 cost of avoiding a learned distance heuristic.
- **Pruner grouping is coarse.** `(srcType, dstInputTypes[0])` conflates
  structurally distinct edges (e.g., `parse → num_to_mat` and `mat_to_num → output`
  end up in the same group as `mat_to_num → num_to_mat`). With this grouping,
  the margin must stay conservative to avoid pruning structurally-needed
  edges, which prevents pruning from firing at all.
- **MLPs needed offline pretraining.** The "carving drives the components'
  training" claim survives directionally (in-loop score climbs from 0.77 to
  0.82) but the bulk of MLP convergence came from the curriculum bootstrap,
  not the online updates.

## Concrete v1 questions that came out of v0

These are the falsifiable follow-ups v0 surfaced, in priority order:

1. **Does the diagnostic survive a different carver?** Implement a
   forward-search variant and re-run the §9.6 ablation. If A vs. B looks
   structurally similar across both carvers, the mandate-enforcement claim
   strengthens. If it diverges, we know the v0 result was carver-coupled.

2. **Can pruning actually fire on a meaningful task?** Add an
   exploration-rate (epsilon-random candidate selection) to the carver so
   alternative paths get tried, then refine pruner grouping (target-aware
   instead of pure type pair). Goal: at least one cleanly-pruned edge with a
   defensible reason.

3. **What does the diagnostic look like with a second primitive type?** Add
   a transformer block (or a different learned component with the same MLP
   role tag) and rerun. This is the §9.6 ablation #4: swap one primitive for
   another with the same type signature. If carved structure transfers, that
   tests heterogeneity directly.

4. **Does the mandate-enforcement claim survive a harder task?** Add
   parentheses (§9.5 step 3). Recursion is the explicit deferral from v0;
   re-engaging it tests whether the framework's machinery handles depth.

5. **Can the inverter coupling be relaxed?** Replace per-primitive hand-coded
   inverters with a learned value function (§10.3). This is the v0 shortcut
   that most threatens the framework's generalization story.

## Bottom line

v0 went well in the sense that *we now know more sharply what to ask next*,
not in the sense that *the framework is validated*. The mandate-enforcement
claim survives its first test. The other three claims from §1.1 are
untouched. The most informative thing v0 did was surface concrete coupling
problems (carver ↔ inverters, pruner grouping ↔ margin) that the design doc
did not anticipate. Those are the right shape of follow-ups for v1: specific,
testable, and grounded in observed failure modes rather than abstract
worries.

# v3 Phase 2 — Emergent Similarity-Based Rerouting

## Context

v3 phase 1 ([doc 06](06-kv-cache-semantic-embeddings.md)) established the
KV-cache substrate and demonstrated that mandates determine *what
structural properties* a trained substrate carries. The phase-1 result
showed mandates as **specifications of substrate state** — same network,
two trainer settings, two qualitatively different substrates.

Phase 2 takes the next step: a network whose **runtime behaviour** depends
on the substrate's content-based geometry. Specifically, a routing
network whose primitives can re-route signal based on similarity to
learned references, demonstrating **functional emergence** — when the
geometry supports the routing, the network classifies; when it doesn't,
the same network can't.

The structural claim being tested:

> Mandates verify the routing decision regardless of where in the network
> they're placed. Whether the mandate names the split directly (Case A)
> or names the downstream output (Case B), both confirm the same routing
> when the substrate supports it; both fail when it doesn't. The split
> happens because the geometry permits it, not because the mandate
> demanded it locally.

## What was built

Two new primitives plus the demo that exercises them:

| Deliverable | What it does |
|-------------|--------------|
| `CosineSimilarity` (`cache/CosineSimilarity.java`) | `(MatrixValue, MatrixValue) → NumberValue`. Scalar cos sim. Total via TotalArithmetic; zero-norm short-circuit returns 0. The complement of `SimilarityGate` — where the gate emits the gated vector, this emits the scalar similarity itself. |
| `NumberSub` (`cache/NumberSub.java`) | `(NumberValue, NumberValue) → NumberValue`. Total scalar subtraction routed through TotalArithmetic. |
| `SimilarityRoutingDemo` (`demo/SimilarityRoutingDemo.java`) | The 2×2 grid: untrained vs trained substrate × Case A vs Case B mandate placement. Per-atom ComputationGraphs, leave-one-out centroids, tight mandate bands. |

About 350 lines of new Java. The two primitives slot into the existing
substrate without further changes.

## The network

Per test atom, a single ComputationGraph wired as:

```
StringValue(atom) ──→ EmbedSymbol ──→ MatrixValue (the embedded atom)
                                            ↓ ↓
                              ┌─────────────┘ └─────────────┐
                              ↓                             ↓
                   CosineSimilarity                   CosineSimilarity
                   (─, ref_orientation)               (─, ref_motion)
                              ↓                             ↓
                       NumberValue (cos_o)          NumberValue (cos_m)
                              ↓                             ↓
                              └────────→ NumberSub ←────────┘
                                            ↓
                                     NumberValue (cos_o − cos_m)
                                            ↓
                                         Output
                                            ↓
                                     NumberValue (terminal)
```

The reference vectors `ref_orientation` and `ref_motion` are computed as
**leave-one-out centroids** — for a test atom in the orientation group,
the orientation centroid averages the *other* orientation atoms (excluding
the test atom). This removes self-inclusion bias and makes the cosine
test honest cross-validation.

## The two atom groups

Two coherent shared-rhs clusters from the ontology, three atoms each, all
positive halves (no antipodal cancellation in centroids):

| Group | Atoms | Shared rhs |
|-------|-------|------------|
| ORIENTATION | top, parallel, straight | "orientation" |
| MOTION | fast, dynamic, active | "motion" |

Each member shares at least one rhs atom with every other member of its
group, so context-pull training places them in the same neighborhood.

## The two mandate placements

| Mandate | Position in network | What it asserts |
|---------|---------------------|------------------|
| Case A — `cos_X_high` | Intermediate (the gate score) | The "X-cluster cosine" is in band [0.70, 1.00] for X-group atoms |
| Case B — `result` | Terminal (the diff) | The terminal value is in band ±[0.10, 0.70] aligned with expected group |

Both mandates are evaluated on every atom in every pass. Tight bands
ensure the verifier's §6.1 non-locality discriminates by node — a stray
value at another position can't accidentally satisfy the mandate.

## The two trainer settings

| Run | Trainer | Expected behaviour |
|-----|---------|---------------------|
| A — untrained | No training; random init from seed 2024 | Random vectors with no cluster structure; centroids carry no information; routing collapses to noise |
| B — trained | Context-pull only (`dichotomyLr=0`, `contextLr=0.05`, `axisLr=0`, 100 epochs) | Atoms with shared rhs cluster; centroids meaningful; routing reflects cluster membership |

Why context-pull only (no dichotomy push): dichotomy push makes paired
atoms antipodal, which drags every rhs atom toward the origin
(it gets pulled equally by both halves of every dichotomy it appears in),
which collapses cluster centroids near origin and destroys group-vs-group
separation. Pure context pull clusters atoms by shared rhs without that
pathology — exactly what cluster-based routing needs. The trainer that
worked for phase 1's structural-property claims is the wrong one for
this functional-routing claim. *The claim's shape determines the trainer's shape.*

## Results — the 2×2 grid

```
========================================================
Run A — untrained substrate (random embeddings)
--------------------------------------------------------
per-atom routing (leave-one-out centroids):
  top        [orient]  cos_o=−0.513 cos_m=−0.347 diff=−0.166   caseA:FAIL   caseB:FAIL
  parallel   [orient]  cos_o=−0.241 cos_m=+0.370 diff=−0.610   caseA:FAIL   caseB:FAIL
  straight   [orient]  cos_o=−0.003 cos_m=+0.353 diff=−0.355   caseA:FAIL   caseB:FAIL
  fast       [motion]  cos_o=+0.181 cos_m=−0.094 diff=+0.275   caseA:FAIL   caseB:FAIL
  dynamic    [motion]  cos_o=+0.207 cos_m=−0.036 diff=+0.243   caseA:FAIL   caseB:FAIL
  active     [motion]  cos_o=+0.161 cos_m=−0.018 diff=+0.179   caseA:FAIL   caseB:FAIL
Case A (mandate the split):     0 / 6 atoms
Case B (mandate downstream):    0 / 6 atoms

========================================================
Run B — trained substrate (context-pull only)
--------------------------------------------------------
per-atom routing (leave-one-out centroids):
  top        [orient]  cos_o=+0.941 cos_m=+0.609 diff=+0.332   caseA:PASS   caseB:PASS
  parallel   [orient]  cos_o=+0.987 cos_m=+0.416 diff=+0.571   caseA:PASS   caseB:PASS
  straight   [orient]  cos_o=+0.989 cos_m=+0.426 diff=+0.564   caseA:PASS   caseB:PASS
  fast       [motion]  cos_o=+0.489 cos_m=+0.985 diff=−0.496   caseA:PASS   caseB:PASS
  dynamic    [motion]  cos_o=+0.479 cos_m=+0.926 diff=−0.447   caseA:PASS   caseB:PASS
  active     [motion]  cos_o=+0.434 cos_m=+0.917 diff=−0.483   caseA:PASS   caseB:PASS
Case A (mandate the split):     6 / 6 atoms
Case B (mandate downstream):    6 / 6 atoms
```

|              | Case A (split) | Case B (downstream) |
|--------------|---------------:|--------------------:|
| Untrained    | 0 / 6          | 0 / 6               |
| Trained      | 6 / 6          | 6 / 6               |

## What this licenses

**Functional emergence.** The network's behaviour — different inputs flowing
through different paths in a content-conditional way — is not encoded
anywhere in the wiring. Both gates always run; both compute their cosine
to a reference; the difference is taken; the output is produced. The
*routing decision* (which gate dominates, which side of zero the diff
lands on) is purely a consequence of the embedding geometry. With random
geometry, the decision is meaningless. With trained geometry, it
classifies the atom by cluster membership. Same hardware, different
software loaded — except here the "software" is the substrate's content
addressable structure.

**Mandate non-locality, demonstrated functionally.** Case A and Case B
verify the *same* routing decision through different network positions.
Case A reads the gate score directly (the split itself). Case B reads the
terminal value (downstream of the split). Both pass when the geometry
supports the routing; both fail when it doesn't. The structural property
the verifier confirms is the same; only the position from which it's read
differs. §6.1 non-locality says any node with a matching value satisfies
the mandate — and that's exactly what makes Case B work: the downstream
mandate doesn't *name* the split, but the split must be happening for the
downstream value to be in the right band.

**Two phases, two complementary uses of mandates.** Phase 1 used mandates
to specify *what structural properties the trained substrate must carry*.
Phase 2 uses mandates to specify *what behaviour a network running on the
substrate must produce*. The same MandateVerifier handles both. The
abstraction is clean: mandates are testable structural assertions, and
the verifier doesn't care whether the structure being tested is a
post-hoc property of training or an in-flight behaviour of execution.

## Honest observations and limitations

- **The trainer for routing is different from the trainer for axis
  alignment.** Phase 1 used `dichotomyLr + contextLr + axisLr` to produce
  a substrate where antipodal pairs and aligned axes were both load-bearing.
  Phase 2 turns dichotomy push off entirely, because forcing pairs
  antipodal collapses the cluster centroids that routing depends on. This
  is a real signal: *which trainer to run depends on the structural claim
  being made*. There isn't a single universal training objective; the
  framework's mandate machinery can verify whatever the trainer produces.

- **Tight mandate tolerances are doing real work.** With wide bands
  (target +0.5, tol 0.4 → [0.1, 0.9]) the verifier cross-matched
  unrelated nodes and reported false positives. With tight bands
  (target +0.85, tol 0.15 → [0.7, 1.0]) the mandates discriminate by node
  — only the intended position satisfies. This is a recurring lesson:
  non-locality is a feature when bands encode meaningful structural
  claims, and a footgun when they don't.

- **Leave-one-out centroids are honest cross-validation.** Without them,
  every test atom is in its own centroid; even random embeddings can route
  correctly because the centroid partly *is* the test atom. With
  leave-one-out, the cosine test is genuinely about cluster membership.

- **Untrained pass rate is exactly 0/6 — the band excludes random
  configurations almost completely.** The phase-1 demo had this
  contrast less clean (some passes via cross-matching). Tight bands plus
  leave-one-out plus single-shared-rhs groups all together produce a
  clean negative result.

- **The demo runs each test atom in its own ComputationGraph.** This is
  N graph constructions, N executions, N verifier calls. For a small N
  (6 atoms) this is fine. A more efficient design would either (a) use
  one graph with N parallel branches, or (b) reuse the graph by rebinding
  roots — but ComputationGraph as built doesn't support root rebinding
  cleanly. Building per-atom is the simplest correct approach.

- **Routing is binary.** Two clusters, sign of diff = predicted class.
  Multi-way routing (K > 2 clusters) is straightforward but adds K gates
  and an argmax-style aggregator primitive that doesn't yet exist. Out
  of scope for this phase.

## Where v3 goes from here

After phase 2, three open threads:

1. **Inline / online training.** The current shape is "train procedurally,
   then carve and execute." A more framework-native design would let
   training happen *during* execution — primitives that, when executed,
   update their underlying state. This would let mandates drive training
   directly: a mandate says "the routing must work post-execution," the
   carver assembles training + verification into one pipeline, and
   training stops when mandates pass. This is the "inline KV training"
   thread; it's the next architectural step.

2. **Learned similarity (Q-W-Kᵀ).** Phase 1 and 2 use fixed cosine.
   Adding learned similarity scoring turns similarity itself into a
   trainable component — the natural next cache variant.

3. **Carver-driven routing assembly.** Per-atom ComputationGraphs are
   manually wired here. With the carver, mandates could drive the
   assembly: "produce a routing decision given a query," and the carver
   selects which similarity primitives and references to use. This is
   the bridge from "framework-verified network" to "framework-assembled
   network."

## Bottom line

Phase 2 demonstrates that **routing emerges from substrate geometry, not
from explicit conditional logic**. The 0/6 untrained → 6/6 trained
transition, achieved via a single change to the trainer with no change
to the network or the mandates, is the cleanest possible demonstration:
same machinery, different content, qualitatively different behaviour.

Combined with phase 1, the framework now exhibits two distinct uses of
mandates over the same substrate: as specifications of *what the
substrate carries* (phase 1) and as specifications of *what a network
running on the substrate does* (phase 2). The MandateVerifier is the
hinge between them — it doesn't know the difference, and it doesn't need
to.

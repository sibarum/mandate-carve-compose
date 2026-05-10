# v3 Phase 3 — Inline, Mandate-Driven Training

## Context

The prior three phases used mandates in three modes:

- **v0 / v1 / v2** — mandates as search-time constraints. The carver
  must produce a structure that satisfies them.
- **v3 P1** — mandates as specifications of structural properties a
  trained substrate must carry. Training is procedural and pre-run;
  the verifier checks post-hoc.
- **v3 P2** — mandates as specifications of network behaviour.
  Training is still procedural; the verifier checks that the carved
  network does the right routing at runtime.

In every case so far, training has been **out-of-band** — a separate
procedural step that happens before the framework's mandate-and-verify
machinery runs. Phase 3 collapses that separation: **the training loop
moves inside the carved graph, with the same mandate serving as both
the structural spec and the gradient target.**

## What was built

Two small additions, plus the demo that exercises them:

| Deliverable | What it does |
|-------------|--------------|
| `StringOutputPrimitive` (`primitive/`) | Pass-through terminal for `StringValue`. Mirrors `OutputPrimitive` (NumberValue) and `TreeOutputPrimitive` (ParseTreeValue) — the String analogue, needed for carvings whose terminal is a symbol. |
| `InlineTrainingDemo` (`demo/`) | Minimal inline-training demonstration. Carved graph + trainable bridge + mandate-driven training loop + same `MandateVerifier`. No pre-training, no separate trainer call. |

About 150 lines of new Java. The pattern reuses existing components
(`EmbedSymbol`, `LookupSymbol`, `VectorTransform`, `MandateVerifier`)
without modification — the framework's existing Trainable contract is
sufficient for one-trainable inline training.

## The carved graph

```
StringValue(A) ──→ EmbedSymbol ──→ MatrixValue (embed(A))
                                          ↓
                                   VectorTransform (trainable, W)
                                          ↓
                                   MatrixValue (W · embed(A))
                                          ↓
                                   LookupSymbol (cosine-nearest)
                                          ↓
                                   StringValue (the predicted output atom)
                                          ↓
                                   StringOutputPrimitive (terminal)
```

Three of these four primitives are static (the embedding table is
initialized lazily and frozen for this run; the lookup is deterministic
cosine-nearest; the terminal is pass-through). Only `VectorTransform`
is trainable. Its W starts random.

## The mandate

```java
MandateSet mandates = new MandateSet(List.of(
        Mandate.result(new StringValue("cold"), 0.0, 0)));
```

One mandate. It names the value the terminal *must* produce. In this
run the input atom is `"hot"` (root binding), and the expected output
is `"cold"`. Same mandate is used twice: once as the gradient target
for training (specifically, `embed("cold")` is the target vector for
the bridge), and once as the verifier's check after the loop finishes.

## The loop

```java
for (int step = 1; step <= maxSteps; step++) {
    cg.execute();
    bridge.backward(trainingTarget);  // trainingTarget = MatrixValue(embed("cold"))
    bridge.step(lr);

    if (step % checkEvery == 0) {
        cg.execute();
        VerificationReport rep = new MandateVerifier().verify(cg, mandates);
        if (rep.allSatisfied()) break;
    }
}
```

Forward → backward → step → verify. The carved graph, the
TransformationGraph, the MandateSet, and the MandateVerifier are
unchanged between iterations. The only thing that updates is the
bridge's W. When `MandateVerifier` reports PASS, the loop stops.

## Results

```
vocabulary: [hot, cold, warm, cool, fire, ice, burn, freeze]
mandate: input='hot'  →  output='cold'

before training:
  terminal = 'cool'

initial mandate verification: FAIL

inline training loop (mandate-driven, terminal: forward → backward → step):
  step   1: terminal='cool'  mandate:FAIL
  step  25: terminal='cold'  mandate:PASS

after training:
  terminal = 'cold'
  final mandate verification: PASS

Inline training: mandate satisfied in 25 steps.
```

**Twenty-five steps** to converge from random init. Initial terminal was
`'cool'` — the random W happened to map `embed("hot")` closest to
`embed("cool")` among the eight vocabulary atoms. After training,
`W · embed("hot") ≈ embed("cold")`, and `LookupSymbol` returns `"cold"`.

## What this licenses

**Mandates as the unifying contract for training and verification.**
The same `Mandate.result(StringValue("cold"), 0.0, 0)` serves
simultaneously as:

1. The **structural spec** the carving must satisfy.
2. The **training target** the bridge optimizes toward.
3. The **verification predicate** that determines when training stops.

This is the next escalation in mandate usage. v0/v1/v2 used them for
(1) only. v3 P1/P2 added (3). v3 P3 adds (2). One abstraction; three
roles; the same `MandateVerifier` doesn't care which role a particular
caller is using it for.

**Training is no longer a separate concern.** Previous phases had a
shape like:

```
parse → train_procedurally → build_carving → execute → verify
```

This phase has:

```
build_carving → loop[ forward → backward → step → verify ]
```

The training loop is *the* execution loop. The carved graph is what
runs forward; backward and step are the framework's response to the
verifier reporting FAIL. No external trainer, no pre-run setup, no
state to carry between phases.

**Static and trainable primitives coexist seamlessly.** The frozen
table, the deterministic lookup, and the pass-through terminal sit
right next to the trainable bridge in the same graph. The framework
makes no architectural distinction between them — training a primitive
is opt-in (`Trainable` interface), and the loop just iterates over the
trainables it finds.

## Honest observations and limitations

- **One trainable in this demo.** Gradient flow is local to
  `VectorTransform`. Extending to multiple chained trainables requires
  the framework to propagate `inputGradient` backward across primitive
  boundaries — that's the autograd-on-the-carving step we deferred.
  Each `Trainable.backward(target)` would need a counterpart that
  takes an output gradient and emits an input gradient.

- **No carver involvement yet.** The graph is hand-wired. The natural
  next step is: define a TransformationGraph with both deterministic
  paths and trainable bridges, let the carver assemble per-mandate,
  and run the inline training loop on whatever carving the carver
  picked. Edge stats then accumulate across mandate-driven carvings —
  the substrate learns which bridges are worth keeping.

- **Mandate-as-target derivation is hand-coded here.** The demo
  manually computes `embed("cold")` as the bridge's training target
  given the result mandate. A framework-native version would derive
  per-trainable targets from the mandate set automatically — given
  the carved structure and the mandates, figure out which intermediate
  position each trainable occupies, and read the appropriate
  expected-value-from-the-mandate as that trainable's target.

- **No "input mandate" yet.** The mandate names the result only; the
  input is a root binding. A symmetric design would let the mandate
  name both ends, and the carver derives the carving structure (and
  the trainables' targets) from that pair. Current shape works because
  the input is fixed at the root binding; the more general shape would
  treat both as mandates.

- **Discrete primitives in the path block backward.** `LookupSymbol`
  is the closest-symbol-by-cosine — discrete output, no real gradient.
  In this demo it doesn't matter because the trainable is *upstream*
  of the discrete operation and trains against a continuous target
  (`embed("cold")`). For a trainable *downstream* of a discrete
  primitive, we'd need a straight-through estimator or stop-gradient
  marker. Future work.

- **Convergence depends on vocabulary density.** With 8 atoms in 32D,
  initial W produces a clear nearest-neighbour distinction and training
  finds the target quickly. With dense vocabularies, the trainable
  needs more capacity (bigger W, maybe nonlinearities) and longer
  training. Same trade-off as in any neural mapper.

## Where v3 goes from here

This phase completes the KV-cache line at the level of "training works
inside the framework." The remaining v3 questions:

1. **Multi-trainable end-to-end.** Two or more trainables in a chain;
   the gradient propagates backward through all of them. This is the
   real architectural step — the framework's existing claim of "not
   differentiable end-to-end" gets relaxed *within a carved graph*.

2. **Mandate-driven carver assembly with trainables in the substrate.**
   The carver decides whether to use a trainable bridge or a
   deterministic alternative, given the mandate. Edge stats accumulate
   across many mandate-driven carvings to learn which combinations are
   worth assembling.

3. **Training the embedding table inline.** Right now the table is
   frozen during the inline run. With proper backward through
   `EmbedSymbol`, the table itself could update during training —
   the substrate co-adapts with the bridge.

4. **All four mandate roles in one demo.** Search constraint (v0–v2),
   structural-property assertion (v3 P1), behavioural assertion (v3 P2),
   training target (v3 P3) — combined in one carving. That's the
   "framework's claim, fully cashed" demo.

## Bottom line

Phase 3 lands the last piece of the mandate abstraction: **mandates
drive training, not just verify it**. The framework's machinery from
v0/v1/v2 (`MandateSet`, `MandateVerifier`, `Trainable`, `ComputationGraph`)
extended to handle this use case without architectural change — the
inline training loop is just a `while (!report.allSatisfied())` around
the existing forward/backward/step trio.

Combined with phases 1 and 2, mandates now play four distinct roles
over the same machinery:

| Phase | Mandate role | Demo |
|-------|--------------|------|
| v0–v2 | Search constraint — "the carver must produce a structure that emits this value" | `MandateVerificationDemo`, `SymbolicRewriteDemo` |
| v3 P1 | Structural-property assertion — "the trained substrate must carry this property" | `SemanticEmbeddingDemo` |
| v3 P2 | Behavioural assertion — "the network running on the substrate must produce this routing" | `SimilarityRoutingDemo` |
| v3 P3 | Training target — "this is the value training should drive the substrate toward" | `InlineTrainingDemo` |

Four roles, one machinery, one MandateVerifier that doesn't know the
difference. That's the framework's claim earned in full, at the level
of v0/v1/v2-style minimal demonstrations.

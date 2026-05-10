# v3 Phase 4 — Carver-Driven End-to-End

## Context

Phases 1, 2, and 3 hand-wired their ComputationGraphs. The carver
(written for v0 against the arithmetic primitive library) had not yet
been pointed at the KV-cache primitives. Phase 3 closed the
*training* loop inside the framework's mandate-and-verify machinery
but left the *carving* loop hand-bypassed.

This phase makes the carver assemble the entire end-to-end pipeline.
No hand-wired ComputationGraph; no per-trainable target derivation in
user code. The same `BackwardChainingCarver` that built arithmetic
pipelines from operator MLPs now builds KV-cache pipelines from
embedding tables, lookups, and a trainable vector bridge.

The scope was deliberately narrowed: **one trainable per carving**,
with deterministic primitives surrounding it. Multi-trainable
end-to-end requires gradient propagation across primitive boundaries
(autograd-on-the-carving), which remains the deferred architectural
step from doc 08. The phase-4 scope hits the "single-trainable
end-to-end" milestone cleanly, with all the carver-side machinery
needed for the harder case already in place.

## What was built

| Deliverable | What it does |
|-------------|--------------|
| Carver: `findOutputNode` generalisation | Recognises any `Terminal` whose `outputType()` matches the result mandate. Previously hard-coded to `OutputPrimitive` (Number) and `TreeOutputPrimitive` (Parse tree); now also picks up `StringOutputPrimitive` (String) and any future terminal. |
| Carver: KV-primitive inversion | `inferInputs` cases for `LookupSymbol` (target `String s` → `MatrixValue(table.embed(s))`), `EmbedSymbol` (target matrix → cosine-nearest symbol), `VectorTransform` (trainable: returns the forward-anchor for `MATRIX`). |
| Carver: forward-anchor pre-pass | Before backward chaining begins, the carver walks the substrate forward from the root input through every non-Terminal primitive (deterministic *and* trainable), recording one representative value per output type. Trainables don't need an exact inversion; their input slot can accept any type-compatible upstream value, and the anchor gives `solve()` a concrete value to recurse on so it terminates at the root. |
| `InlineTrainer` (`training/`) | Generic forward / backward / step / verify loop. Walks a `CarvingResult`, finds every `Trainable` placed by the carver, and uses `CarvingResult.simulatedValues()` as that trainable's target. Identity-dedups via `Trainable.trainableIdentity()`. About 100 lines. |
| `CarverEndToEndDemo` (`demo/`) | Builds a TransformationGraph from the KV primitives, supplies a single result mandate (`StringValue("cold")`) and root (`StringValue("hot")`), and prints the carved graph plus the training trace. No `new ComputationGraph(...)` anywhere in user code. |

About 200 lines of new Java plus 60 lines of carver extension. No
changes to `Trainable`, `MandateVerifier`, `ComputationGraph`, or the
existing primitives' contracts.

## The carved graph

The TransformationGraph the carver sees is any-to-any modulo type
compatibility, built by `TransformationGraphBuilder`:

```
output   --> embed     (terminal can feed back into embed; structurally allowed)
lookup   --> output
lookup   --> embed     (cycle-capable; the carver's path-block prevents revisit)
embed    --> lookup
embed    --> bridge
bridge   --> lookup
```

Six edges, four nodes. The carver is *not* told which path is correct.

Given:
- root input `StringValue("hot")`,
- result mandate `StringValue("cold")`,

the carver backward-chains:

```
terminal (string-output)  target='cold'
  └── lookup                target=embed("cold")  [inferred from target string via table]
        └── bridge          target=embed("cold")  [carver places the trainable here because
              │                                    the deterministic-only path from "hot"
              │                                    cannot reach "cold"]
              └── embed     target=embed("hot")   [forward anchor for MATRIX, computed
                    │                              from rootInput through the embed table]
                    └── ROOT  "hot"
```

Demo output (the printed carving):

```
carved computation graph:
  c3_embed [embed-symbol] slot0<-ROOT  simulated=matrix[dim=32]
  c2_bridge [vector-transform] slot0<-c3_embed  simulated=matrix[dim=32]
  c1_lookup [lookup-symbol] slot0<-c2_bridge  simulated='cold'
  c0_output [string-output] slot0<-c1_lookup
```

Four nodes, three slot wires, one root binding. The `simulated`
column is what the carver expected each node to produce — and that's
exactly what the trainer uses as the per-node training target.

## The training loop

`InlineTrainer.run()` does this, generically:

```java
ComputationGraph cg = carving.graph();
Map<CompGraphNode, Value> sim = carving.simulatedValues();
List<TrainableSlot> slots = collectTrainables(cg, sim);

for (int step = 1; step <= maxSteps; step++) {
    cg.execute();
    IdentityHashMap<Object, Boolean> seen = new IdentityHashMap<>();
    for (TrainableSlot s : slots) {
        if (seen.putIfAbsent(s.trainable.trainableIdentity(), Boolean.TRUE) != null) continue;
        s.trainable.backward(s.target);
        s.trainable.step(learningRate);
    }
    if (atCheckpoint(step)) {
        cg.execute();
        if (verifier.verify(cg, mandates).allSatisfied()) return success;
    }
}
```

No knowledge of which primitives are trainable. No hand-derived
targets. Each trainable's local target is the value the carver
simulated at that node. For the bridge, that's `embed("cold")`. For
the embedding (also `Trainable`, but with target equal to its current
output `embed("hot")`), the backward update is a no-op — the
gradient is zero because the value already matches.

This is why **single-trainable-per-path** works without autograd: each
trainable's target is *locally* derivable from the carver's simulated
value, and only one trainable in the chain has a non-trivial gradient
to apply.

## Results

```
vocabulary: [hot, cold, warm, cool, fire, ice, burn, freeze]
root input:        'hot'
mandate (result):  'cold'

transformation graph edges (substrate):
  output   --> embed
  lookup   --> output
  lookup   --> embed
  embed    --> lookup
  embed    --> bridge
  bridge   --> lookup

carved computation graph:
  c3_embed [embed-symbol] slot0<-ROOT  simulated=matrix[dim=32]
  c2_bridge [vector-transform] slot0<-c3_embed  simulated=matrix[dim=32]
  c1_lookup [lookup-symbol] slot0<-c2_bridge  simulated='cold'
  c0_output [string-output] slot0<-c1_lookup

inline training loop:
  step   0: terminal='cool'  mandate:FAIL
  step   1: terminal='cool'  mandate:FAIL
  step  25: terminal='cold'  mandate:PASS

Carver + InlineTrainer: mandate satisfied in 25 steps.
```

Twenty-five steps to convergence, same as doc 08's hand-wired demo.
The training behaviour is identical because the carved graph *is*
the same graph the previous phase hand-wired — the carver finds the
same minimal structure the human did, because it's the only path
that satisfies the mandate.

## What this licenses

**The carver handles trainables in the substrate.** Previously the
carver had been exercised with `MlpPrimitive`/`LearnedArithmetic`
(trainables surrounded by deterministic encoding/decoding glue) and
with `RewriteRulePrimitive` (deterministic). It now also handles
`VectorTransform` (trainable with arbitrary input shape) and the
KV primitives. The pattern that emerged — **trainable inversion uses
a forward anchor instead of an exact inverse** — is the right
abstraction: a trainable's input doesn't constrain the carving's
structure, only its type and the upstream chain back to the root.

**One signature, three programmes.** The `Trainable` interface
(`backward`, `step`, `trainableIdentity`) supports operator-MLP
training (v0/v1), embedding-table training (v3 P1), and now
carver-placed bridge training (v3 P4) without any change. The trainer
doesn't know which kind of network it's stepping; it only knows that
each trainable has a local target.

**The end-to-end pipeline now belongs to the framework.** Earlier
phases had user code constructing `ComputationGraph`s by hand.
Phase 4 reduces user code to: "here's the primitive library, here's
the mandate, here's the root — go." Everything else — graph
assembly, target derivation, training loop, verification — runs
inside the framework.

## Honest observations and limitations

- **One trainable in the gradient path.** Both `EmbedSymbol` and
  `VectorTransform` are `Trainable`, but the embedding table's target
  equals its current output, so its gradient is zero. The only
  trainable with non-trivial loss is the bridge. Multi-trainable
  with cross-primitive gradient flow is still deferred.

- **Forward-anchor pre-pass uses first-found-per-type.** If the
  substrate has two primitives producing `MATRIX` with different
  semantics (e.g., two embedding tables in a multi-KV design),
  the pre-pass picks one and the carver uses it for all trainable
  inversion. Disambiguating by source-node or by typed value would
  be a small but real extension.

- **The carver still doesn't do dim-checking.** `VectorTransform`'s
  inDim must match the upstream `MatrixValue`'s dim, or execution
  crashes. The carver checks `ValueType` equality but not vector
  dimensionality. Fine for this demo (everything is dim 32); will
  bite as soon as the substrate has multiple matrix sizes.

- **Mandate-to-target derivation is implicit, not explicit.** The
  carver's `simulatedValues` map is the bridge between mandates and
  training targets — but it's a consequence of the backward-chaining
  inversion, not a deliberately exposed mechanism. A future
  refactor would name the relation: "for each placed `Trainable`,
  derive its target from the mandate via simulation."

- **No edge-stat feedback from inline training yet.** The carver
  uses edge stats from prior runs to rank candidates. With inline
  training, every carve is also a training opportunity, but the
  trainer doesn't currently feed back a per-edge success score
  to the substrate. Adding it would let the carver learn which
  bridges are worth assembling over many mandate-driven sessions —
  the "training the substrate as a side-effect of solving tasks"
  story doc 08 hinted at.

## Where v3 goes from here

Phase 4 closes the no-hand-wiring milestone. Remaining:

1. **Multi-trainable end-to-end.** Autograd-on-the-carving:
   `Primitive.backward(outputGrad) → inputGrad` so gradients chain
   across primitive boundaries. The most architecturally consequential
   step.
2. **Multi-KV-per-symbol** (the design James raised). Every symbol
   carries multiple embeddings; the carver picks which view a given
   bridge attends to. Sidesteps disambiguation by making "all views"
   the only addressable unit.
3. **Inline-train + edge-stats feedback.** Each successful inline
   run updates the substrate's edge stats; the carver learns which
   trainables-in-the-substrate are worth keeping.

## Bottom line

The carver, the trainer, and the verifier — three pieces written for
three different phases — compose into a single pipeline driven by a
single mandate. No hand-wired graphs; no per-trainable target
derivation; no separate pre-training pass. The mandate is the input,
the carver assembles the structure, the inline trainer drives the
trainable to convergence, the verifier confirms the result. The
framework's `Mandate, Carve, Compose` slogan is now executable in
that order with no intermediate human glue.

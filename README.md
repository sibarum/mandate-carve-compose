# Mandate, Carve, Compose

A meta-architecture for orchestrating heterogeneous primitives — neural
networks, hand-coded operations, and symbolic rewrite rules — into
computation pipelines whose structure is itself learned and inspectable.

> Codebase identifier: **STRNN** (Symbolic Term Rewrite Neural Network) —
> the original aspirational name, preserved in package paths
> (`sibarum.strnn.*`). The project has outgrown its starting frame; the
> three-verb name describes what it has become.

## The three verbs

The framework is one substrate operated through three distinct moves.
Each is a real piece of code; each names a capability the framework
demonstrates.

- **Mandate** — what the engineer brings. A *mandate* is a typed
  specification of an intermediate or final value that the carved
  computation must produce *somewhere* along its execution. Mandates
  are non-local (they specify *what*, not *where*) and structurally
  enforced: a candidate computation graph either produces the
  mandated value or is rejected. Code: `sibarum.strnn.mandate.*`.
- **Carve** — what the framework does. The *carver* searches the
  transformation graph (the substrate of available primitives) for a
  computation graph that satisfies every mandate. It backward-chains
  from the result mandate, ranks candidates by accumulated edge
  statistics, and uses ε-greedy exploration to avoid lock-in.
  Code: `sibarum.strnn.carving.*`.
- **Compose** — what comes out. The carved computation graph is a
  DAG that composes heterogeneous primitives — MLPs, transformer
  blocks, deterministic operations, pattern-matched rewrite rules —
  through typed slot wiring. The same machinery dispatches all of
  them; the only thing that changes between primitives is what's
  inside the box. Code: `sibarum.strnn.computation.*`,
  `sibarum.strnn.primitive.*`, `sibarum.strnn.rewrite.*`.

The three verbs aren't a slogan layered on top — they map directly
to package boundaries and to the demos that exercise each capability
in isolation.

## What this is not

- Not a new neural network architecture. The framework sits *above*
  MLPs and transformers; they remain the workhorses for learned
  function approximation.
- Not a symbolic-AI system pretending to use neural nets. Both
  symbolic rewrite rules and learned components are first-class; the
  carver dispatches them through the same code path.
- Not differentiable end-to-end. The carving is discrete; gradients
  flow only inside individual learned primitives. The framework
  composes differentiable units without itself being differentiable.

## Project status

Implementation has progressed through four phases. All phases are
runnable; each closes with a written-up diagnostic.

- **v0** — substrate plus the structural-enforcement claim, demonstrated
  on a single arithmetic task.
- **v1** (Patterns A and B) — heterogeneous *learner* composition with
  two architectures (MLP, transformer); competitive selection at the
  same role; ε-greedy exploration; primitive-level pruning.
- **v2** — symbolic rewrite rules as first-class primitives. The dual
  claim (symbolic rewrite *and* heterogeneous composition under one
  carving substrate) is demonstrated on the simplest possible test.
- **v3 P1** — KV-cache foundation: symbol↔vector embeddings with
  bidirectional lookup, total componentwise arithmetic floor (no NaN),
  learnable vector ops, multi-objective semantic training, and an A/B
  diagnostic showing that a single mandate moves from FAIL to PASS
  *solely* because the trainer was given the matching objective. Same
  primitives, same scorers, same mandate set, two qualitatively
  different substrates.
- **v3 P2** — Emergent similarity-based rerouting. A network whose
  routing decisions depend on cosine similarity to learned cluster
  references. The 2×2 diagnostic (untrained vs trained substrate ×
  Case A mandate-on-split vs Case B mandate-downstream) goes from
  0/6 to 6/6 along both mandate placements when training is added.
  Demonstrates **functional emergence** — the routing isn't encoded
  in the wiring; it emerges from substrate geometry.
- **v3 P3** — Inline, mandate-driven training. No pre-training, no
  separate trainer call: the carved graph hosts the training loop
  directly. A single mandate (input atom A, output atom B) serves as
  the structural spec, the gradient target, and the verification
  predicate. Random W converges in 25 steps. Mandates now play four
  distinct roles over the same machinery — search constraint,
  structural-property assertion, behavioural assertion, and training
  target.
- **v3 P4** — Carver-driven end-to-end. The carver assembles the
  KV-cache pipeline itself from a primitive library, no hand-wired
  ComputationGraph. A generic `InlineTrainer` walks the carving,
  picks up every `Trainable`, and uses the carver's simulated values
  as per-trainable targets. Single mandate, single root input, single
  call: framework builds the graph, trains the bridge, verifies the
  result. Scope: one trainable per gradient path; multi-trainable
  end-to-end (autograd-on-the-carving) remains deferred.

All implementation is in Java 25. MLPs and transformer blocks are
written from scratch (~500 lines including backprop) with no
ML-library dependency. The carver, mandate verifier, pattern-matching
substrate, and KV-cache primitives are all hand-rolled.

## Documentation index

In dependency order — each builds on the previous:

| Doc                                                  | What it covers |
|------------------------------------------------------|----------------|
| [`initial-design-doc.md`](docs/initial-design-doc.md) | Original framework design — vocabulary, mechanism, claims |
| [`01-first-phase-results.md`](docs/01-first-phase-results.md) | v0 substrate, mandate enforcement, the §9.6 ablation |
| [`02-pattern-a-results.md`](docs/02-pattern-a-results.md) | MLP ↔ transformer drop-in swap; component-agnostic orchestration |
| [`03-pattern-b-results.md`](docs/03-pattern-b-results.md) | Competitive coexistence; the ε-greedy fix; primitive-level pruning |
| [`04-v2-plan-symbolic-rewriting.md`](docs/04-v2-plan-symbolic-rewriting.md) | v2 roadmap |
| [`05-v2-symbolic-rewriting-results.md`](docs/05-v2-symbolic-rewriting-results.md) | Symbolic-rewrite outcomes; the dual-claim diagnostic |
| [`06-kv-cache-semantic-embeddings.md`](docs/06-kv-cache-semantic-embeddings.md) | KV-cache foundation, semantic ontology, A/B mandate-vs-trainer diagnostic |
| [`07-similarity-based-rerouting.md`](docs/07-similarity-based-rerouting.md) | Emergent routing in a network, 2×2 (untrained/trained × split/downstream) diagnostic |
| [`08-inline-mandate-driven-training.md`](docs/08-inline-mandate-driven-training.md) | Mandates drive training inline; four mandate roles over one machinery |
| [`09-carver-driven-end-to-end.md`](docs/09-carver-driven-end-to-end.md) | Carver assembles the KV-cache pipeline; generic InlineTrainer; no hand-wired graph |

## What has been demonstrated

- **Mandate enforcement is structural, not loss-traded** ([01](docs/01-first-phase-results.md)).
  The §9.6 ablation showed that without mandates, the carver cannot
  even construct a valid graph for the arithmetic task; with full
  mandates, 96% of carvings succeed.

- **Carved orchestration is component-agnostic** ([02](docs/02-pattern-a-results.md)).
  Swapping MLPs for transformers at the same role and signature leaves
  the carved structure identical (within ~0.3 nodes per primitive
  class across 50 carvings). The carver does not know or care which
  underlying learner fills its slots.

- **Edge statistics correctly reflect primitive quality** ([03](docs/03-pattern-b-results.md)).
  When two architectures compete at the same role and one is
  deliberately undertrained, the framework identifies the weaker one
  and prunes it. The score gap (0.060) cleanly exceeds the prune
  margin (0.030).

- **Selection responds to evaluation when exploration is enabled** ([03](docs/03-pattern-b-results.md)).
  v0's pure-exploitation default produced edge-level lock-in. Adding
  ε-greedy candidate sampling (ε=0.1) restored quality-driven
  selection. The 10% exploration cost paid back in better outcomes.

- **The dual claim earns its name** ([05](docs/05-v2-symbolic-rewriting-results.md)).
  The same primitive library — including pure-symbolic rewrite rules
  (`IdentityZero`, `IdentityOne`, `Distribute`, `FactorCommon`) and a
  learned-leaf rule (`EvaluateBinaryOp` with MLP inside) — produces
  different carved orchestrations depending on which rules' patterns
  match the input. `"x + 0"` yields `identity-zero`; `"3 + 4"` yields
  `evaluate-add`. Same library, same carver, same seed.

- **Mandates determine what structural properties the substrate carries** ([06](docs/06-kv-cache-semantic-embeddings.md)).
  The A/B diagnostic on the trained KV-cache substrate runs the same
  pipeline twice — same primitives, same scorers, same mandate set,
  same seed — under two trainer settings. Run A (no axis-alignment
  objective): `axes_aligned` mandate FAILS, axis score 0.147 (random
  baseline). Run B (with axis-alignment objective): `axes_aligned`
  PASSES, axis score 0.818. Two qualitatively different substrates from
  one specification difference, with the verifier reporting honestly in
  both directions. Generalizes mandates from "search-time constraints"
  to "specifications of structural properties the substrate must carry."

- **Routing emerges from substrate geometry, not from wiring** ([07](docs/07-similarity-based-rerouting.md)).
  A network with two similarity gates and a difference node correctly
  classifies atoms by cluster membership only when the substrate is
  trained. 2×2 grid: untrained substrate scores 0/6 in both Case A
  (mandate the split) and Case B (mandate downstream); trained
  substrate scores 6/6 in both. Same network, same wiring, same
  mandate forms — only the substrate's training state changes.
  Demonstrates **functional emergence** plus a third use of mandates:
  specifying *what behaviour a network running on the substrate must
  produce*. Mandate non-locality means Case A and Case B verify the
  same routing decision through different positions in the graph.

- **Mandates drive training inline — no pre-training required** ([08](docs/08-inline-mandate-driven-training.md)).
  A carved graph hosts the training loop directly. One mandate
  (`input='hot', output='cold'`) serves as the structural spec, the
  gradient target for the trainable bridge, and the verification
  predicate. From random W, the bridge converges in 25 forward /
  backward / step iterations; the same `MandateVerifier` that v0–v2
  used confirms convergence. Static primitives (frozen table,
  deterministic lookup, terminal) coexist seamlessly with the
  trainable bridge in the same graph. Adds the fourth mandate role:
  *training target*.

- **No hand-wired graphs: the carver assembles end-to-end** ([09](docs/09-carver-driven-end-to-end.md)).
  The carver from v0 (originally built for arithmetic operator MLPs)
  is now exercised against the KV-cache primitives. Given a
  TransformationGraph of `EmbedSymbol`, `VectorTransform`,
  `LookupSymbol`, and `StringOutputPrimitive`, a single result
  mandate (`StringValue("cold")`), and a root input
  (`StringValue("hot")`), the carver assembles
  `embed → bridge → lookup → output` automatically — picking the
  trainable bridge as the only edge that can close the gap between
  unrelated symbols. A generic `InlineTrainer` (no knowledge of any
  specific primitive) iterates over whatever trainables the carver
  placed, using the carver's `simulatedValues` as per-trainable
  targets, and converges in 25 steps. Scope: one trainable per
  gradient path; the autograd-on-the-carving step that would let
  multiple trainables share gradient flow is still deferred, but
  every other piece of "no hand-wiring" now works end-to-end.

## Known limitations

These are real and worth being explicit about:

- **Rule application is root-only.** The v2 demo's rules apply at the
  root of the input tree. `Distribute` and `FactorCommon` are in the
  library but not exercised because composing them with `evaluate-add`
  would require descending into sub-trees. v3 should pick a strategy
  (descent primitive vs. built-in walking).

- **Carve-search tractability is untested at scale.** The largest
  primitive library tried is ~12 primitives with 28 type-compatible
  edges. Whether the carver's mandate-driven backward search remains
  tractable with 50+ rules — particularly conditional or non-linear
  ones — is unknown.

- **Pruning in the edge-pair sense is too coarse to fire.** The v0
  edge-pair pruner (group by `(srcType, dstInputTypes[0])`) conflates
  structurally distinct edges and never fires safely. The v2
  primitive-competition pruner works correctly but only on the
  narrower question of "which learner wins at this role."

- **Mandates are hand-specified.** No mechanism yet for deriving
  mandates from a higher-level specification. Real users would want
  to say "produce a normal form" rather than enumerate every
  intermediate tree shape.

- **Action-principle features beyond pruning are deferred.** §4.2
  (structural expansion when a region has high penalty pressure) and
  §4.4 (unreachability-driven graph extension) are not implemented.
  v0/v1/v2 use a fixed transformation graph; the framework's full
  self-extension story is unrealized.

## Next steps (v3 questions)

v3 phase 1 (KV-cache foundation) opened a parallel research line; phase 2
(similarity-based rerouting) demonstrated functional emergence on top of
that substrate. The remaining v3 questions split across the symbolic and
the cache lines:

**KV-cache line:**

1. **Multi-trainable end-to-end.** Phase 4 has one trainable per
   carved gradient path. Extending to chained trainables requires
   the framework to propagate `inputGradient` backward across primitive
   boundaries — autograd-on-the-carving. The framework's "not
   differentiable end-to-end" claim relaxes *within a carved graph*.

2. **Multiple KVs per symbol.** A single symbol carries multiple
   embeddings (orthogonal "views"), all retrieved together. The carver
   picks which view a given bridge attends to. Sidesteps the disambiguation
   question by making "all views" the only addressable unit.

3. **Learned similarity (Q-W-Kᵀ).** Phases 1–4 use fixed cosine.
   Adding a learned scoring function turns similarity itself into a
   trainable component — the natural next cache variant.

4. **Inline-train + edge-stats feedback.** Every successful inline
   training run is also a vote of confidence in the placed bridges.
   Feeding the trainer's success score back into `EdgeStats` would
   let the carver learn over many sessions which bridges are worth
   assembling — the "substrate learns from solving tasks" loop.

**Symbolic line:**

4. **Sub-tree rule application.** v2's rules apply at the root. Either
   add a `Recurse` primitive (orthogonal, adds rules) or build walking
   into each rule (simpler, less flexible).

5. **Rule-library scaling.** Run the carver against 20–50 rules covering
   associativity, commutativity, distributivity, and simplification.
   Instrument per-step search cost.

6. **Mandate derivation.** Specify abstract goals ("simplify", "evaluate
   fully", "produce canonical form") and derive concrete mandate sets
   from them — eliminating the hand-specification limitation.

7. **All primitive types in one carving.** Pattern A showed MLPs and
   transformers swap; v2 showed symbolic and learned coexist; v3 P1
   added embedding-space ops; v3 P2 added similarity-based routing.
   The next test is **all of them at once in one task** — a transformer
   that produces a parse tree, a symbolic rule library that simplifies
   it, an MLP that evaluates leaves, and an embedding cache that
   carries semantic context with similarity-based dispatch.

## Build and run

Java 25 (GraalVM tested), Maven multi-module:

```
mvn -pl strnn-model compile
```

Demos run as standalone classes. From the project root:

```
java -cp strnn-model/target/classes sibarum.strnn.demo.<DemoName>
```

Available demos, ordered by dependency:

| Demo                          | Phase    | What it shows                                      |
|-------------------------------|----------|----------------------------------------------------|
| `HandWiredDemo`               | v0 P0    | Type system + primitives carry arithmetic end-to-end |
| `MlpTrainingDemo`             | v0 P1    | From-scratch MLP converges on add/mul             |
| `ManualGraphDemo`             | v0 P2    | TransformationGraph + ComputationGraph + execution |
| `MandateVerificationDemo`     | v0 P3    | Structural mandate enforcement passes/fails as expected |
| `CarvingDemo`                 | v0 P4    | Carver builds the structured pipeline automatically |
| `TrainingDemo`                | v0 P5    | End-to-end training loop with edge stats           |
| `AblationDemo`                | v0 P6    | §9.6 diagnostic: with-mandates vs result-only      |
| `TransformerTrainingDemo`     | v1       | From-scratch transformer block                     |
| `PatternADemo`                | v1       | MLP ↔ transformer drop-in swap                     |
| `PatternBDemo`                | v1       | Competitive coexistence + ε-greedy + pruning       |
| `ParseTreeDemo`               | v2 P0    | Tree-valued type, parser, ValueDistance            |
| `PatternMatchingDemo`         | v2 P1    | Pattern-match + substitute round-trips             |
| `IdentityZeroDemo`            | v2 P2    | First symbolic rule end-to-end                     |
| `EvaluateBinaryOpDemo`        | v2 P3    | Learned MLP inside a symbolic rule                 |
| `TreeCarvingDemo`             | v2 P4    | Carver builds tree-shaped pipelines                |
| `SymbolicRewriteDemo`         | v2 P5    | The dual-claim diagnostic                          |
| `TotalArithmeticDemo`         | v3 P1    | Total componentwise arithmetic floor (no NaN, sign-preserving) |
| `SymbolEmbeddingDemo`         | v3 P1    | Symbol↔vector substrate (lazy init + cosine reverse lookup) |
| `EmbeddingTrainingDemo`       | v3 P1    | SGD through the embedding table; contrastive structure |
| `TrainedRecallDemo`           | v3 P1    | Composed proof: trained KV cache survives noise |
| `VectorOpsDemo`               | v3 P1    | Learnable transform, vector +/−/·, similarity gate |
| `SemanticParserDemo`          | v3 P1    | Parser for the six-operator semantic ontology |
| `SemanticEmbeddingDemo`       | v3 P1    | A/B mandate diagnostic: axes_aligned FAIL → PASS |
| `SimilarityRoutingDemo`       | v3 P2    | 2×2 routing diagnostic: untrained/trained × split/downstream |
| `InlineTrainingDemo`          | v3 P3    | Inline mandate-driven training: 25-step convergence from random W |
| `CarverEndToEndDemo`          | v3 P4    | Carver assembles KV pipeline; generic InlineTrainer; no hand-wired graph |

## Repository layout

```
strnn-model/src/main/java/sibarum/strnn/
├── value/         # sealed Value hierarchy (StringValue, NumberValue, MatrixValue, TokenListValue, ParseTreeValue)
├── primitive/     # Primitive interface, Trainable, LearnedArithmetic, Terminal markers
├── mlp/           # from-scratch Mlp with backprop
├── transformer/   # from-scratch Transformer block (single-head attention + FFN + residuals)
├── transformation/ # TransformationGraph (substrate), TransformationNode/Edge, EdgeStats
├── computation/   # CompGraphNode, SlotSource, ComputationGraph (per-task DAG with topo execution)
├── mandate/       # Mandate, MandateSet, MandateVerifier      ← Mandate
├── carving/       # BackwardChainingCarver — backward search with mandate-aware ranking + ε-greedy   ← Carve
├── rewrite/       # Pattern, Matcher, RewriteRulePrimitive, EvaluateBinaryOp, IdentityZero/One, Distribute, FactorCommon
├── training/      # Trainer, Pruner, Datasets, InlineTrainer
├── cache/         # KV-cache foundation: SymbolEmbeddingTable, EmbedSymbol/LookupSymbol, TotalArithmetic,
│   │             # VectorTransform/Add/Sub/Mul, SimilarityGate
│   └── semantic/  # parser + AST for the semantic ontology, multi-objective trainer, three scoring primitives
└── demo/          # all runnable demos                        ← Compose (the runnable demonstrations)
docs/              # design doc + phase result writeups
strnn-model/src/main/resources/sample-semantics.txt  # hand-crafted ontology used by v3 P1 demos
```

## Bottom line

The architecture's name describes what exists. **Mandate, Carve,
Compose** is one substrate operated through three distinct moves: the
engineer specifies what must hold, the framework searches for or trains
a structure that holds it, the result is a heterogeneous composition.
At the demonstrated level, this works for symbolic rewrite tasks, for
heterogeneous learner orchestration, for their union, and (as of v3 P1)
for trained vector-space substrates whose properties are themselves
mandate-specified.

The v3 P1 A/B result generalized what mandates *do*. They started as
search-time constraints (v0 / v1 / v2: which structure does the carver
pick?). They now also serve as specifications of structural properties
that must hold over a trained substrate (v3 P1: same primitives, two
trainer objectives, two qualitatively different substrates). The
MandateVerifier doesn't care which mode — it just confirms whether the
expected structure exists.

Whether the framework scales further — to large rule libraries, derived
mandates, learned similarity, all-primitive-types-in-one-carving — are
the open v3 questions. The discipline that produced v0 → v1 → v2 → v3 P1
was: pick one specific claim, build the smallest thing that tests it,
write up what was and was not shown. The remaining v3 phases should
follow the same pattern.

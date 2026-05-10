# Symbolic Term Rewrite Neural Network (STRNN)

A meta-architecture for orchestrating heterogeneous primitives — neural
networks, hand-coded operations, and symbolic rewrite rules — into
computation pipelines whose structure is itself learned and inspectable.

## What this is

STRNN is **not** a new neural network architecture competing with MLPs
or transformers. It is a **structural layer** that sits above them.
The framework's value comes from three properties:

1. **Specification-driven design.** Engineers specify what intermediate
   values must be produced and what type signatures must hold; a search
   procedure (the *carver*) discovers a working composition that
   satisfies the specification.

2. **Ante-hoc interpretability.** Mandated intermediate values are
   enforced *structurally* — either the carved graph produces the
   value somewhere, or it is rejected. This is verifiable by
   construction, not by post-hoc analysis of trained weights.

3. **Heterogeneous composition.** Different primitive types — MLPs,
   transformer blocks, deterministic operations, pattern-matched
   rewrite rules — coexist in one transformation graph and are
   dispatched by the same orchestration machinery.

Two graph objects underpin everything: a slowly-shaped **transformation
graph** (the substrate, an any-to-any-modulo-types library of available
primitives) and a fast **computation graph** (carved per-task from
fragments of the transformation graph). The full design rationale lives
in [`docs/initial-design-doc.md`](docs/initial-design-doc.md).

## Project status

The implementation has progressed through three roughly-equal phases:

- **v0** — substrate + interpretability claim, demonstrated on a single
  task (arithmetic over single-digit operands).
- **v1** (Patterns A and B) — heterogeneous composition with two
  learner types (MLP and transformer), competitive selection between
  alternatives at the same role, ε-greedy exploration, primitive-level
  pruning.
- **v2** — symbolic rewrite rules as first-class primitives;
  rule-based and learned-leaf primitives coexist under the same carver.

All implementation is in Java 25, pure pluggable code with no
ML-library dependency. MLPs and transformer blocks are written from
scratch (~500 lines including backprop). The carver, mandate verifier,
and pattern-matching substrate are all hand-rolled.

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

## What has been demonstrated

- **Mandate enforcement is structural, not loss-traded** ([01](docs/01-first-phase-results.md)).
  The §9.6 ablation showed that without mandates, the carver cannot
  even construct a valid graph for the arithmetic task; with full
  mandates, 96% of carvings succeed.

- **The orchestration is component-agnostic** ([02](docs/02-pattern-a-results.md)).
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
  v0's pure-exploitation default produced edge-level lock-in (the
  first edge to score above 0.5 dominated indefinitely). Adding
  ε-greedy candidate sampling (ε=0.1) restored quality-driven
  selection. The 10% exploration cost paid back in better outcomes.

- **The dual claim earns its name** ([05](docs/05-v2-symbolic-rewriting-results.md)).
  The same primitive library — including pure-symbolic rewrite rules
  (`IdentityZero`, `IdentityOne`, `Distribute`, `FactorCommon`) and a
  learned-leaf rule (`EvaluateBinaryOp` with MLP inside) — produces
  different carved orchestrations depending on which rules' patterns
  match the input. `"x + 0"` yields `identity-zero`; `"3 + 4"` yields
  `evaluate-add`. Same library, same carver, same seed.

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

In dependency order:

1. **Sub-tree rule application.** Without it, symbolic rules can only
   transform whole trees, which limits the framework to trivially-
   small expressions. Either add a `Recurse` primitive (orthogonal,
   adds rules) or build walking into each rule (simpler, less
   flexible).

2. **Rule-library scaling.** Run the existing carver against 20-50
   rules covering associativity, commutativity, distributivity, and
   simplification. Instrument per-step search cost. Measure where the
   `(target, primClass)` cycle key starts blocking legitimate
   reapplications.

3. **Mandate derivation.** Right now, every mandate is a fully-
   specified `(name, expectedValue, tolerance, ordering)` tuple. A
   useful tool would let users specify abstract goals ("simplify",
   "evaluate fully", "produce canonical form") and derive concrete
   mandate sets from them.

4. **Genuinely-different primitive types in one carving.** Pattern A
   showed MLPs and transformers transparently swap; v2 showed
   symbolic and learned coexist. The next test is **all of them at
   once in one task** — e.g., a transformer that produces a parse
   tree, a symbolic rule library that simplifies it, and an MLP that
   evaluates leaves. That exercises the full §1.2 / §8.1 pitch.

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

## Repository layout

```
strnn-model/src/main/java/sibarum/strnn/
├── value/         # sealed Value hierarchy (StringValue, NumberValue, MatrixValue, TokenListValue, ParseTreeValue)
├── primitive/     # Primitive interface, Trainable, LearnedArithmetic, Terminal markers
├── mlp/           # from-scratch Mlp with backprop
├── transformer/   # from-scratch Transformer block (single-head attention + FFN + residuals)
├── transformation/ # TransformationGraph (substrate), TransformationNode/Edge, EdgeStats
├── computation/   # CompGraphNode, SlotSource, ComputationGraph (per-task DAG with topo execution)
├── mandate/       # Mandate, MandateSet, MandateVerifier
├── carving/       # BackwardChainingCarver — backward search with mandate-aware ranking + ε-greedy
├── rewrite/       # Pattern, Matcher, RewriteRulePrimitive, EvaluateBinaryOp, IdentityZero/One, Distribute, FactorCommon
├── training/      # Trainer, Pruner, Datasets
└── demo/          # all runnable demos
docs/              # design doc + phase result writeups
```

## Bottom line

The architecture's name describes what exists, not just what was
intended. STRNN is — at the demonstrated level — a symbolic rewrite
system that doubles as a heterogeneous network composition platform,
sharing one orchestration substrate. Whether it scales to genuinely
hard symbolic tasks is the open question.

Whether the project keeps demonstrating its value depends on each next
deliverable being a falsifiable test rather than a layer of new
abstraction. The pattern from v0 → v1 → v2 has been: pick one specific
claim, build the smallest thing that tests it, write up what was and
was not shown. v3 should follow the same discipline.

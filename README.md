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
- **v3 P5** — Multi-head KV. Two parallel cache chains coexist in
  one substrate; the carver picks among them via edge stats; inline
  training tunes whichever chain was picked; chains retain
  independent specialization across mandate switches. Adds the
  `CachedItem` scaffold (reserves the type slot for `NetworkItem`
  later), source-aware forward anchors in the carver (per-source
  rather than per-type), and edge-stats feedback from the trainer
  (per-session reward → traced-edge updates). Three-session
  diagnostic: chain A trains hot→cold; chain B trains hot→warm;
  chain A re-verifies hot→cold in **zero** steps (no further
  training). The "multiple views, carver selects" plumbing for
  Key-Network is in place.
- **v3 P6** — **Key-Network.** The cache stores entire trained
  subgraphs as items, and the carver composes them. `NetworkItem`
  joins `EmbeddingItem` under `CachedItem`; `CachedNetworkPrimitive`
  exposes a stored carving to the outer carver as just another
  deterministic primitive. The carver's `inferInputs` for cached
  networks is forward-evaluation: try each anchor, see which
  produces the target. With two pre-trained inner networks
  (`hot→cold` and `cold→freeze`) and an outer mandate
  `result='freeze'` with root `'hot'`, the carver builds a two-step
  composed pipeline automatically — no shortcuts available because
  the inner vocabularies are disjoint at the input atom. The "values
  are networks" reframing of KV is real, and lives inside the
  framework as just one more variant of one more interface.
- **v3 P8** — **Carver composes from the cache.** The NetworkCache's
  inventory becomes the substrate. Hand `cache.primitives()` to a
  TransformationGraphBuilder; submit a mandate; the carver assembles
  a chain of cached networks of whatever depth is required. Three
  mandates over the same 3-network cache produce three different
  composed chains: `hot→cold` (1-step), `hot→freeze` (2-step via
  composition), `hot→ice` (3-step via composition). Adds BFS
  reachability to the carver's forward-anchor pre-pass (deterministic
  primitives only — Trainables get a single per-node anchor instead,
  since their forward produces unbounded continuous outputs) and
  two-phase regularized bridge training to `NetworkCache` (Phase 1:
  identity-on-vocabulary; Phase 2: positive shift with identity
  maintenance — blocks cross-cache shortcuts from rank-1 linear
  bridge extrapolation).
- **v3 P9** — **A substrate that builds itself.** Mandates drive
  the cache's growth. For each `(input, output)` mandate: try to
  carve from the current inventory; if it fails, BFS the cache from
  the mandate's input to find the deepest reachable atom, spawn one
  new bridge to close the gap, re-carve. Seven mandates over an
  initially empty cache produce 5 spawns + 2 reuses; the final
  inventory is a minimal spanning set that covers every mandate
  seen. Mandate 7 (`cold→ice`) was a reuse — the cache had already
  assembled `cold→freeze→ice` from prior spawns and the carver found
  it. The framework's most autonomous form: the user writes the
  mandate stream, the framework returns the carving and grows the
  substrate as needed. Mandate, Carve, Compose — over a substrate
  that didn't exist when the demo started.
- **v3 P7** — **NetworkCache: a cache that learns its own inventory.**
  `NetworkCache` is a stateful cache of `NetworkItem`s with
  spawn-on-demand training. Start empty; each `(input → output)` pair
  fed to it either matches an existing entry or triggers a fresh
  carving + inline-training run that produces a new stored network.
  Optional `maxNetworks` cap with eviction by success count, so the
  cache's contents drift toward whatever mappings the training data
  actually exercises. Two-run diagnostic: unbounded cache populates
  4 entries from 4 distinct mappings; bounded cache (max=3) handles 5
  spawns by evicting the 2 least-successful entries while preserving
  `hot` (which was re-fed three times). The cache is a coordinator on
  top of the existing carver + trainer; no framework changes needed.
- **v3 P10** — **Mandate-Carve-Compose as a methodology.** The
  framework leaves the toy substrate and tackles real-world NLP:
  build an entity span extractor for an Elden Ring lore corpus that
  generalizes to unseen items. Eight iterations on one task, each
  failure naming the next missing mandate — POS tagging
  (UD English EWT) → BIO structural mandate → Parquet pretraining
  → staged training → Lexicanum cross-entry labeling + density
  filter → POS-conditioned decode constraint. The artifact isn't a
  state-of-the-art model; it's the *worked example of mandate-driven
  AI development* — and the demonstrably-auditable system that falls
  out of it. Adds the `mcc-elden-ring` downstream-consumer module,
  the `store/` persistence-boundary interfaces (`NetworkStore`,
  `EntityStore<N,R>`, `VectorStore`), two new framework primitives
  (`TextTokenize`, `ClassifierHead`), and the first end-to-end
  multi-layer backprop chain in the framework (POS loss flows
  through the context encoder into the embedding table).
- **v3 P10 follow-up — POS layer improvement (iters 9–13).** The same
  diagnose-then-name-the-mandate loop applied to the POS layer
  itself, the bottleneck iter 8 identified. Added a dev-split + per-
  tag confusion matrix to `PosTrainer` first (no architecture change),
  then read what the diagnostic named: PROPN F1 = 0.328, 37% of gold
  PROPN being mis-tagged NOUN. Mandate: deterministic word-shape
  features (capitalization, all-caps, digit, hyphen, punct, length),
  which raised dev accuracy 86.1% → 89.2%. Wider context window
  *regressed* at the same MLP width — diagnostic: capacity-bound, not
  unhelpful — bigger MLP (hidden 64 → 128) plus more epochs took it
  to **90.9%**. The downstream re-run of `BioCrossEntryDemo` is the
  payoff: the upstream BIO tagger now emits only 44 *raw* spans on
  Ranni (down from 67), of which 43 are clean entities — the
  POS-conditioned decode constraint demotes 1 token where it
  previously demoted 22. **Improving the lower layer raised the
  ceiling for the upper layer automatically**, which is the layered-
  inspectability methodology paying off at the layer-stack level.
- **v3 P10 follow-up — span-boundary chase (iters 14–17).** Four
  iterations targeting the residual failure mode: multi-word entities
  with internal function words ("Remembrance of the Baleful Shadow")
  splitting at the article. **Iter 14** (Viterbi with learned bigram
  POS transitions): null result — every non-zero transition weight
  regressed; bigram transitions can't disambiguate semantic confusions
  like DET ADJ NOUN vs DET NOUN NOUN. **Iter 15** (book-title
  span-coherence pretraining from 1.48M Parquet rows): partial fix —
  `the` got pulled in but full entity still split at the next
  capitalized token. **Iter 16** (Viterbi over BIO with hand-set
  transitions): another near-null — the model takes the I→O→B escape
  path that first-order Markov can't see. **Iter 17** (replace the
  whole bootstrap chain with `elden_ring_final_train.jsonl` —
  11.7k Elden Ring Q&A pairs with canonical `entity_name` metadata):
  **the failure mode dissolves**. "Remembrance of the Baleful Shadow",
  "House Caria", and most multi-word lore entities now bound
  correctly as single spans, in one Stage-1 pretrain over
  domain-matched labels. The methodology lesson: iters 4 through 16
  were an honest series of *substitutes* for one missing ingredient
  (domain-matched labeled supervision); when the data arrived, no
  architectural change was needed. **The sharper rule: when a series
  of structural fixes makes only incremental progress, the
  diagnostic is often "we don't have the right supervision" — look
  for the data, not the architecture.** Infrastructure landed:
  `PosTransitions`, `BioTransitions`, generic `ViterbiDecoder`,
  `BookTitlesBioLoader`, `EldenJsonlBioLoader`, `ParquetPeekDemo`.
- **v3 P10 follow-up — typed-entity heads (iters 22–23).** After the
  iter 17–21 entity-tagger arc, the natural next move was lateral: add
  a third trainable layer that predicts `EntityType` for each detected
  span. **Iter 22** (one-hot output): three named layers (POS → BIO →
  Type), 99.1% training-set accuracy on hand-annotated, typed Ranni
  output reads naturally (`[CHARACTER] Rennala`, `[ARTIFACT] Rune of
  Death`, `[EVENT] Night of the Black Knives`). **The compositional
  claim worked — adding a layer cost ~250 lines and reused the prior
  layers unmodified.** **Iter 23** (substrate-authentic vector-anchor
  output): the seven `EntityType` labels registered as keys in their
  own `SymbolEmbeddingTable`, MLP outputs a 32-dim vector, inference
  is nearest-anchor cosine, both MLP and anchors trainable. **Result:
  anchor collapse** — all trained anchors drifted to the same point;
  training-set accuracy crashed from 99.1% to 29.6%. The diagnostic
  cleanly names the missing piece (a contrastive component to push
  different-class anchors apart). Iter 23 stays in the repo as the
  honest negative result: the substrate-authentic architecture is
  *correct as a principle* but needs an anti-collapse mandate to
  function. Both iters demonstrate the methodology generalizing across
  layers: each named mandate has an ablate-able effect, each failure
  names the next missing mandate, applied symmetrically at architecture
  *and* data-selection *and* now output-encoding layers.
- **v3 P10 follow-up — data-selection mandates (iters 18–21).**
  Iter 17 found the right supervision, but at 258k Stage-1 tokens.
  This sub-arc asks: *which subset of that signal is load-bearing,
  and what mandates select it?* **Iter 18** (3-corpus gradient with
  7-class output): +21 spans, but output classes collapsed under
  Stage 2 fine-tuning — *gradient training is a representation tool,
  not an output-class tool*. **Iter 19** (random budget-clip to ~80
  spans per source): **78× less training, comparable result** —
  established that most of the iter-17 signal wasn't load-bearing.
  **Iter 20** (naive failure-driven curation): backfired because the
  `two-word-propn` pattern flooded the budget (76/132 sentences) and
  starved the longer-multi-word patterns that the targeted failures
  actually needed. **Iter 21** (relaxed patterns + add book-titles
  as a fourth corpus): **215 spans (best result of the arc), with
  60× less training data than iter 17**. Book-titles, used
  pattern-specifically for its rich multi-function-word internal
  structure, supplied the supply iter 20 starved on. The methodology
  lesson: **data-selection mandates are the same kind of artifact as
  architecture mandates** — each is a named decision with an
  ablate-able effect; the right corpus for a pattern is the one that
  has the pattern; per-pattern balance matters more than richness.
  Infrastructure landed: `EntityClasses`, `KnowledgeGraphBioLoader`,
  `BioGradientSnapshotDemo`, `BioBudgetParitySnapshotDemo`,
  `BioTargetedCurationSnapshotDemo`, `BioRelaxedCurationSnapshotDemo`,
  `FullCorpusSnapshotDemo`, and side-by-side `snapshot-iter*.txt`
  artifacts for review.
- **v3 P11** — **KSQ: a fixed-algebra substrate.** Parallel research
  line opening alongside the MCC NLP arc. KSQ
  (Key-Split-Quaternion) is an architecture where the *substrate* is
  the split-quaternion algebra $\mathbb{H}_s \cong M_2(\mathbb{R})$
  with four hand-picked anchors (Möbius cardinals: $K_0$ identity,
  $K_i$ elliptic, $K_j$ hyperbolic, $K_\infty$ idempotent) as fixed
  structural constants, and learning is constrained to **two small
  embedding tables** — token → anchor coefficients and β → output
  logits. No learned weight matrices anywhere in the middle. The
  "computation" lives in the algebra (one bilinear step $Q^2 = Q
  \cdot Q$); the model's expressive job is to pick where in the
  algebra each input lands. Three architectural iterations, each
  named by a structural diagnostic: **iter 1** (chain $S_T = Q_T
  \cdots Q_1$): transformer-shaped, REJECTED — the algebra was being
  used as sequence-composition. **iter 2** (sum-pool token logits →
  softmax α → single Q → Q²): T=2 XOR converges, but the spec's
  predicted K_i specialization is unreachable (0/40 trials); the
  optimizer locks onto K_j because softmax α makes the cross-term
  $\beta_i = 2 \alpha_0 \alpha_i$ sign-bounded. **iter 3**
  (signed α via tanh + squared-product regularizer): K_i becomes
  accessible (0 → 19 occurrences), the K_i/K_j basin counts split
  exactly 5-5 at λ=0.1, and the architecture's elliptic/hyperbolic
  symmetry is restored. A new degenerate corner surfaces — saturated
  $K_i \leftrightarrow K_j$ paired specialization makes Q² = 0 via
  the anti-commutator $\{K_i, K_j\} = 0$ — names the next mandate
  (contrastive cross-vocab regularizer; deferred). Every iteration
  gradient-checked to machine epsilon before any training, so
  "architecture is wrong" stayed cleanly separated from "math is
  wrong." **iter 4** (re-align $K_\infty$ from $(1+k)/2$ to
  $(1+j)/2 = e_+$, the idempotent of the *same* $\mathbb{R}[j]$
  subalgebra that $K_j$ generates): the anchor set becomes
  intentionally linearly dependent ($K_0 + K_j = 2 K_\infty$,
  spanning 3 dims in $M_2(\mathbb{R})$ rather than 4), so $K_j$
  and $K_\infty$ basins now represent two natural *bases* of the
  same subalgebra. XOR solve rate preserved at 10/10 across λ ≤ 0.1;
  seed 3 lands cleanly in $K_j \to K_\infty$ (one token in
  generator basis, one in idempotent basis), the cleanest possible
  algebraic outcome for the inference-specialization collapse.
  **iter 5** (cross-vocab contrastive regularizer promoted from
  deferred to implemented after the same same-anchor collapse
  failure was named in two consecutive iterations):
  $\mathcal{R}_{\text{cross}} = \nu \sum_{v_1 \neq v_2}
  \langle\alpha(v_1), \alpha(v_2)\rangle^2$, gradient-checked at
  6.8e-11. λ=1.0 solve rate goes 6/10 → 9/10 with ν ≥ 0.1.
  Small regression at λ=0.1 (10/10 → 9/10) because the squared
  inner product penalizes anti-aligned same-anchor solutions
  (valid for XOR; the regularizer prefers truly orthogonal
  anchors). No single (λ, ν) achieves clean 10/10 across all
  three λ; names the iter-6 mandate (adaptive ν schedule: strong
  early, decay late). Adds `sibarum.strnn.ksq.*` (anchors, Mat2,
  embedding table, output head, model) and two demos.
  The methodology lesson
  (substrate-construction mandates are mandates too, with the same
  ablate-able-effect discipline as the architecture and data
  mandates from iter 1–23) extends MCC down a level: the algebra
  itself is a designed primitive whose choice has measurable
  consequences.

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
| [`10-multi-head-kv.md`](docs/10-multi-head-kv.md) | Multi-head KV; per-source forward anchors; edge-stats feedback; three-session diagnostic |
| [`11-key-network.md`](docs/11-key-network.md) | Key-Network: cached subgraphs as `CachedItem`; carver composes two stored networks |
| [`12-network-cache.md`](docs/12-network-cache.md) | NetworkCache: stateful cache of trained subgraphs; spawn-on-demand; eviction by success |
| [`13-carver-composes-from-cache.md`](docs/13-carver-composes-from-cache.md) | Carver composes from the cache's inventory; N-step composition via BFS reachability |
| [`14-grand-finale.md`](docs/14-grand-finale.md) | The grand finale: a substrate that builds itself from a stream of mandates |
| [`15-mandate-as-methodology.md`](docs/15-mandate-as-methodology.md) | Twenty-three iterations on real-world NLP; MCC as a development methodology applied at the architecture, data-selection, and output-encoding layers |
| [`16-ksq-substrate.md`](docs/16-ksq-substrate.md) | KSQ: a fixed split-quaternion algebra as substrate; three architectural iterations restoring elliptic/hyperbolic basin symmetry; mandates at the substrate-construction layer |

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

- **Multi-head KV with carver-driven selection** ([10](docs/10-multi-head-kv.md)).
  Two parallel KV chains in one substrate (separate
  `SymbolEmbeddingTable`s, `EmbedSymbol`s, `VectorTransform` bridges,
  `LookupSymbol`s, shared terminal). The carver routes through
  whichever chain edge stats prefer. Three-session diagnostic on the
  same substrate: chain A trains `hot→cold` in 25 steps; chain B
  trains `hot→warm` in 25 steps; chain A re-verifies `hot→cold` in
  **0** steps — the chains are genuinely independent, training one
  does not perturb the other. Adds source-aware forward anchors
  (`Map<TransformationNode, Value>` so multiple TG nodes producing
  the same `ValueType` keep distinct anchors per source), edge-stats
  feedback from `InlineTrainer` (reward → traced-edge updates), and
  the `CachedItem` scaffold that reserves the slot for `NetworkItem`
  — the actual Key-Network landing.

- **Key-Network: cached subgraphs composed by the carver** ([11](docs/11-key-network.md)).
  The cache stores trained subgraphs as items. `NetworkItem` joins
  `EmbeddingItem` under the sealed `CachedItem` interface;
  `CachedNetworkPrimitive` wraps a `NetworkItem` and presents it to
  the outer carver as a regular deterministic primitive. Two pre-
  trained inner networks (`hot→cold` and `cold→freeze`, with disjoint
  vocabularies at the input atom to prevent shortcuts) sit in an
  outer TransformationGraph; the outer mandate `result='freeze'` with
  root `'hot'` is satisfied by a two-step pipeline the carver
  composes: `ROOT → hotCold → coldFreeze → output`. The substrate
  returns *behaviours* now, not just vectors; the carver retrieves
  by structural fit to the mandate; composition uses the same
  Primitive/Carver machinery as anything else. No new abstraction
  layer was needed — one new `CachedItem` variant plus an
  `inferInputs` case in the carver.

- **A cache that learns its own inventory** ([12](docs/12-network-cache.md)).
  `NetworkCache` starts empty and spawns new `NetworkItem`s on demand:
  every `(input → output)` pair fed to it either matches an existing
  entry (counts as a successful use) or triggers a full carving +
  inline-training pass that produces and stores a new network.
  Optional bounded capacity with eviction by success count makes the
  cache's contents drift toward whatever mappings the training data
  actually exercises. The diagnostic runs an unbounded cache to 4
  spawns/4 entries, then a bounded cache (max=3) through 5 spawns
  with 2 evictions — `hot` is re-fed three times and survives, the
  least-used entries get dropped. The cache is a coordinator on top
  of the existing carver + trainer; "items are networks" is opaque
  to its interface but load-bearing for what it does.

- **Mandate-Carve-Compose as a development methodology** ([15](docs/15-mandate-as-methodology.md)).
  Eight iterations on real-world NLP — entity extraction over a hand-
  annotated Elden Ring corpus, generalizing to a held-out questline.
  Each iteration's failure mode named the next missing mandate: POS
  tagging (function words leaking as entities) → BIO structural
  mandate (binary too coarse) → Parquet pretraining (proper-noun
  generalization) → staged training (style transfer) → Lexicanum
  cross-entry labeling + density filter (lore-style register) →
  POS-conditioned decode constraint (cleanest output yet). The
  artifact isn't a state-of-the-art model; it's the *worked example*
  that MCC is a development methodology, not just a framework: a
  model designed to be auditable is necessarily a stack of small,
  named, individually-validated layers, where every failure is
  localizable. This is to AI roughly what TDD is to software
  engineering — designing for inspectability changes what you build.

- **A fixed-algebra substrate is a viable MCC primitive** ([16](docs/16-ksq-substrate.md)).
  KSQ uses the split-quaternion algebra
  $\mathbb{H}_s \cong M_2(\mathbb{R})$ as a fixed substrate with four
  Möbius-cardinal anchors, and constrains learning to two small
  embedding tables. No learned weight matrices anywhere in the middle;
  the model's expressive job is choosing where in the algebra each
  input lands. Five structural rewrites driven by named diagnostics:
  chain (transformer-shaped) → sum-pool + softmax α + $Q^2$ (K_i
  basin unreachable, 0/40) → sum-pool + tanh signed α + $Q^2$ (K_i
  becomes accessible, K_i/K_j basins split exactly 5-5 at λ=0.1) →
  re-align $K_\infty$ to the same $\mathbb{R}[j]$ subalgebra as
  $K_j$ (anchor set becomes intentionally linearly dependent;
  $K_j \leftrightarrow K_\infty$ basins now express two bases of the
  same subalgebra rather than two different subalgebras) →
  cross-vocab contrastive regularizer (same-anchor collapse mandate
  named twice in iters 3 and 4 promoted to implemented; λ=1.0
  goes 6/10 → 9/10). Each iteration was gradient-checked to machine
  epsilon *before* training, cleanly separating "math is wrong" from
  "architecture is wrong."
  Extends the methodology pattern (mandates with ablate-able effects)
  down to the substrate-construction layer: the algebra itself —
  including the precise *choice of anchor representatives*, not just
  the anchor count — is a designed primitive whose choice has
  measurable consequences for which basins the optimizer can find
  and what the basin structure means algebraically. **Solve-rate is
  insufficient alone**: a scalar-collapse "solution" (both tokens on
  K_0) passes XOR but fails the architectural claim, so the demo
  reports raw AND non-trivial-subalgebra solve rate. With a strict
  scalar-specialization classifier (K_0 dominant AND every other |α|
  < 0.5), the two metrics coincide in all 120 trials: at our scale
  the architecture never trivializes, even when basin-label
  shorthand suggested otherwise. **Saddle-vs-basin verification**
  goes further: initializing the embedding directly at scalar-trivial
  (token 0 → pure +K_0, token 1 → pure -K_0) and training under
  4 reg settings × 3 init magnitudes finds that 11/12 conditions
  LEAVE scalar-trivial — the K_j, K_∞ components grow from 0 (at
  init) to 0.20–0.58 magnitude. Scalar-trivial is a saddle, not a
  basin. The architecture prefers to use its substrate is now a
  tested dynamical property, not an inference from random-init
  outcomes alone.

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

**Beyond Key-Network:**

1. **Autograd across cached-network boundaries.** Phase 6 composes
   pre-trained inner networks; gradients don't flow from the outer
   mandate back through them. The deferred step now has concrete
   motivation: a `Primitive.backward(outputGrad) → inputGrad` contract
   would let the outer training loop tune inner networks even after
   they're cached. End-to-end differentiability *within a carved
   graph* relaxes the framework's standing "not differentiable
   end-to-end" claim.

2. **`NetworkCache` and explicit key lookup.** A first-class cache
   that stores many `NetworkItem`s and exposes them via a
   `LookupNetwork(key) → NetworkItem` primitive — analogous to how
   `LookupSymbol` retrieves a vector by key. The carver chooses
   *which* network to retrieve from a much larger substrate.

3. **Emergent head specialization (phase 5b).** Phase 5 hand-biases
   edge stats. The follow-up: run N sessions with random init and
   equal initial stats; watch heads specialize over a mandate
   sequence via edge-stat feedback alone. The infrastructure is
   already in place.

4. **Learned similarity (Q-W-Kᵀ).** Phases 1–5 use fixed cosine.
   Adding a learned scoring function turns similarity itself into a
   trainable component — natural next cache variant.

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

Java 25 (GraalVM tested), Maven multi-module. Framework:

```
mvn -pl strnn-model compile
```

Full project (framework + Elden Ring NLP module):

```
mvn clean install -DskipTests
```

Framework demos run as standalone classes:

```
java -cp strnn-model/target/classes sibarum.strnn.demo.<DemoName>
```

P10 NLP demos require external datasets to be downloaded into
`download/` (see `download-files.md` for attribution) and are run via
the `exec-maven-plugin`:

```
mvn -pl mcc-elden-ring exec:java \
  -Dexec.mainClass=sibarum.elden.demo.BioCrossEntryDemo \
  -Dexec.args="./download/en_ewt-ud-train.conllu ./download/Lexicanum_Warhammer_RAG-v1.12.txt"
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
| `MultiHeadCarvedDemo`         | v3 P5    | Two parallel KV chains; carver picks via edge stats; independent specialization |
| `KeyNetworkDemo`              | v3 P6    | Cached subgraphs as substrate items; carver composes two stored networks |
| `NetworkCacheTrainingDemo`    | v3 P7    | NetworkCache builds inventory from data; bounded variant evicts by success |
| `CarverFromCacheDemo`         | v3 P8    | Carver composes from cache inventory; N-step compositions per mandate |
| `AdaptiveCarverDemo`          | v3 P9    | **Grand finale**: substrate builds itself; mandates drive cache growth |
| `TokenizerDemo`               | v3 P10   | `TextTokenize` primitive across the full Elden Ring corpus (vocab=1,243) |
| `EmbeddingDemo`               | v3 P10   | Embedding substrate end-to-end: lookup stability, NN, gradient updates |
| `SpanTaggerTrainingDemo`      | v3 P10   | Binary span tagger trained on annotated items, F1=0.987 on training set |
| `RanniInferenceDemo`          | v3 P10   | Binary span tagger inference on held-out Ranni's questline |
| `PosTaggerTrainingDemo`       | v3 P10   | POS layer trained on UD English EWT; end-to-end backprop into embedding table |
| `PosAwareInferenceDemo`       | v3 P10   | Span tagger augmented with predicted POS as input features |
| `BioInferenceDemo`            | v3 P10   | BIO (3-class) structural mandate: multi-word entities cluster cleanly |
| `BioWithProperNounsDemo`      | v3 P10   | Staged training: Parquet (news NER) → fine-tune on Elden Ring |
| `BioWithLoreDemo`             | v3 P10   | Staged training with Parquet + Lexicanum (self-mention labeling) |
| `BioCrossEntryDemo`           | v3 P10   | **Iter 8 best**: Lexicanum cross-entry labeling + density filter + POS-conditioned decode (improved POS layer = iter 13) |
| `ParquetPeekDemo`             | v3 P10   | Schema + sample-row inspector for any Parquet file (DuckDB) |
| `BioWithBookTitlesDemo`       | v3 P10   | Iter 15: 3-stage BIO with book-title span-coherence pretraining |
| `BioWithViterbiDemo`          | v3 P10   | Iter 16: book-title pretrain + Viterbi over BIO with hand-set transitions |
| `BioWithEldenJsonlDemo`       | v3 P10   | **Iter 17 — boundary problem resolved**: direct supervision from `elden_ring_final_train.jsonl` (11.7k labeled Q&A pairs) |
| `FullCorpusSnapshotDemo`      | v3 P10   | Iter 17 inference written to `snapshot-iter17.txt` across all 5 storylines (44 items, 191 spans) |
| `BioGradientSnapshotDemo`     | v3 P10   | Iter 18: 7-class corpus-gradient BIO (normal/fantasy/eldenring); class signal collapses under Stage 2 fine-tune |
| `BioBudgetParitySnapshotDemo` | v3 P10   | Iter 19: 80-span-per-source budget; **78× less training, comparable result** to iter 17 |
| `BioTargetedCurationSnapshotDemo` | v3 P10 | Iter 20: failure-driven biased clip with narrow patterns; backfires — the `two-word-propn` pattern floods the budget |
| `BioRelaxedCurationSnapshotDemo` | v3 P10 | **Iter 21 — arc best**: relaxed patterns + book-titles as 4th corpus; 60× less training, +12.6% spans vs iter 17 |
| `BioWithTypedHeadsSnapshotDemo` | v3 P10 | **Iter 22**: typed-entity classifier (one-hot output) — three named layers stack cleanly |
| `BioWithVectorTypedHeadsSnapshotDemo` | v3 P10 | Iter 23: substrate-authentic vector-anchor type head; recorded *negative* result (anchor collapse names the missing contrastive mandate) |
| `KsqGradientCheckDemo`        | v3 P11   | Finite-difference verification of every KSQ gradient pathway (CE + per-vocab regularizer + cross-vocab regularizer) at machine epsilon |
| `KsqParityDemo`               | v3 P11   | **KSQ T=2 XOR sweep**: 10 seeds × (λ, ν) grid. Reports basin distribution AND non-trivial-subalgebra solve rate (architectural claim: algebra must actually be used, not collapsed to scalar). Demonstrates cross-vocab regularizer rescues λ=1.0 from 6/10 to 9/10 |
| `KsqScalarTrivialDemo`        | v3 P11   | **Saddle-vs-basin verification**: initialize at scalar-trivial (token 0 → +K_0, token 1 → -K_0), train under 4 reg settings × 3 init magnitudes. 11/12 conditions LEAVE the trivial config — scalar-trivial is a saddle, not a basin |

## Repository layout

```
strnn-model/src/main/java/sibarum/strnn/          ← THE FRAMEWORK
├── value/         # sealed Value hierarchy (StringValue, NumberValue, MatrixValue, TokenListValue, ParseTreeValue)
├── primitive/     # Primitive interface, Trainable, LearnedArithmetic, Terminal markers
│                  # + TextTokenize, ClassifierHead (P10)
├── mlp/           # from-scratch Mlp with backprop (backward returns input gradient — consumed by P10)
├── transformer/   # from-scratch Transformer block (single-head attention + FFN + residuals)
├── transformation/ # TransformationGraph (substrate), TransformationNode/Edge, EdgeStats
├── computation/   # CompGraphNode, SlotSource, ComputationGraph (per-task DAG with topo execution)
├── mandate/       # Mandate, MandateSet, MandateVerifier      ← Mandate
├── carving/       # BackwardChainingCarver — backward search with mandate-aware ranking + ε-greedy   ← Carve
├── rewrite/       # Pattern, Matcher, RewriteRulePrimitive, EvaluateBinaryOp, IdentityZero/One, Distribute, FactorCommon
├── training/      # Trainer, Pruner, Datasets, InlineTrainer
├── cache/         # KV-cache foundation: SymbolEmbeddingTable, EmbedSymbol/LookupSymbol, TotalArithmetic,
│   │             # VectorTransform/Add/Sub/Mul, SimilarityGate; CachedItem/NetworkItem/NetworkCache
│   └── semantic/  # parser + AST for the semantic ontology, multi-objective trainer, three scoring primitives
├── store/         # P10: NetworkStore, EntityStore<N,R>, VectorStore — persistence-boundary interfaces
├── ksq/           # P11: fixed-algebra substrate. KsqAnchors (4 Möbius cardinals in M_2(R)),
│                  # Mat2 (2x2 matmul/transpose helpers), KsqEmbeddingTable, KsqOutputHead,
│                  # KsqModel (sum-pool → tanh α → Q → Q² → β → head). Parallel research line.
└── demo/          # all runnable framework demos              ← Compose (the runnable demonstrations)

mcc-elden-ring/src/main/java/sibarum/elden/       ← P10 DOWNSTREAM CONSUMER
├── corpus/        # Item, ItemCategory, ShatteringEra, RannisQuestline, VolcanoManor, DungEaterQuestline, MillicentQuestline
├── annotation/    # AnnotatedItem, EntityType, EntitySpan, Relation, RelationType, AnnotationParser, ImplicitEntities + 4 storyline annotation files
├── graph/         # EntityGraph, EntityNode, EntityGraphBuilder
├── embedding/     # CorpusVocabulary, OffsetTokenizer, ContextEncoder
├── pos/           # ConlluParser, PosTagset, PosTrainer (end-to-end backprop)
├── data/          # ParquetBioLoader (DuckDB), LexicanumParser, LexicanumCrossEntryLabeler, SurfaceTrie
├── training/      # TaggingTrainingData, TaggerPipeline
└── demo/          # all NLP pipeline demos

docs/              # design doc + phase result writeups (01..16)
strnn-model/src/main/resources/sample-semantics.txt  # hand-crafted ontology used by v3 P1 demos
download/          # external datasets pulled in for P10 (UD CoNLL-U, Parquet, Lexicanum); .gitignored
download-files.md  # attribution paper-trail for downloaded datasets
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

**Phase 10 added a further claim**: MCC is not only a framework but a
*technique* — a discipline for building AI systems as stacks of small,
named, individually-validated layers where every failure is localizable
to a specific layer with a specific named mandate. This is to AI roughly
what TDD is to software engineering: designing for inspectability
changes what you build. The Elden Ring NLP pipeline in `mcc-elden-ring`
is the worked example. The methodology — diagnose, name the missing
mandate, add the layer, repeat — is the artifact that carries over to
whatever gets built on the framework next.

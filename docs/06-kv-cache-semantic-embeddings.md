# v3 Phase 1 — KV-Cache Foundation and Semantic Embeddings

## Context

v2 closed with the dual claim demonstrated on the simplest possible test:
symbolic rewrite rules and a learned-leaf rule coexisting in one carving
substrate. The substrate did its job structurally; what it did *not* yet
do was carry meaning beyond the bare-symbol level.

v3 starts a different line of work: turn the framework's content-addressable
machinery into a real **KV cache**, where (a) symbols can be embedded into
vectors, (b) those embeddings can be trained to encode semantic structure,
and (c) the structural properties we want them to carry are themselves
stated as mandates. The point of phase 1 is not to ship a useful tool —
it's to lay a substrate that downstream phases (learned similarity,
multi-cache architectures, carver integration) will build on.

## What was built

Six layers, each with its own diagnostic, capped by one composition that
puts everything together under mandate verification.

| Layer | Deliverable | Demo |
|-------|-------------|------|
| 0 | `SymbolEmbeddingTable` — lazy random init, cosine-nearest reverse lookup, in-place gradient updates. `EmbedSymbol` / `LookupSymbol` primitives over the existing `Trainable` contract. | `SymbolEmbeddingDemo` |
| 1 | Direct supervised training of embeddings via `EmbedSymbol.backward` + `step`; contrastive training via direct `table.update` calls. Both demonstrate SGD landing on the right table row. | `EmbeddingTrainingDemo` |
| 2 | Composed proof of concept: train N=20 symbols to orthogonal targets, sweep recall under additive noise. 100% recall through noise = 1.0·‖emb‖, graceful degradation past that. | `TrainedRecallDemo` |
| 3 | `TotalArithmetic` floor — total, sign-preserving, NaN-rejecting helpers for `+`, `−`, `·`, `/` on `double`. IEEE's NaN-producing edge cases overridden per a small dispatch table. | `TotalArithmeticDemo` |
| 4 | Vector ops layer: `VectorTransform` (learnable Wv), `VectorAdd` / `VectorSub` / `VectorMul` (componentwise via TotalArithmetic), `SimilarityGate` (soft cosine gating). All compose without NaN. | `VectorOpsDemo` |
| 5 | Semantic substrate: parser for `sample-semantics.txt` (six operators, 107 relations, 326 atoms); `SemanticTrainer` (multi-objective: dichotomy push + context pull); three scoring primitives. | `SemanticParserDemo`, `SemanticEmbeddingDemo` |

About 1500 lines of new Java across `cache/`, `cache/semantic/`, and the
demos. The arithmetic floor is independently testable; the vector ops
inherit totality by construction; the semantic embedding work composes the
table, the trainer, and the scorers under one carved (manually-wired)
ComputationGraph with mandates checking each structural claim.

## The total arithmetic floor

A precondition the user fixed early: zero tolerance for algebra-breaking
edge cases. Vector arithmetic must be total — no NaN, no undefined cases
leaking through componentwise operations.

The chosen encoding is `±∞` for the omega element with sign preserved,
accepting that the omega tower flattens (`2/0` and `5/0` collapse to the
same `+∞`; `ω·ω` cannot be distinguished from `ω`). For embedding-space
vectors, this is acceptable — the regime where coefficients matter is
already pathological.

IEEE 754 covers most cases correctly. The five overrides are:

| operation | IEEE result | total-algebra result |
|-----------|-------------|----------------------|
| `0 · ±∞` | NaN | ±1 (sign of the infinity) |
| `±∞ · 0` | NaN | ±1 |
| `±∞ + ∓∞` | NaN | 0 |
| `±∞ − ±∞` (same sign) | NaN | 0 |
| `0 / 0` | NaN | 1 |
| `±∞ / ±∞` | NaN | sign(a)·sign(b) |

NaN inputs are rejected at the boundary — they cannot enter the substrate.
`TotalArithmeticDemo` enumerates 68 (a, b) cases across the full
sign × magnitude grid, plus NaN-guard tests.

## The semantic substrate

`sample-semantics.txt` is a hand-crafted ontology: 107 dichotomies along
contextual axes, expressed in a six-operator language. Parsing it required
a small recursive-descent grammar with operator precedence (`:` tightest,
then `+`/`*`, then `&` loosest, with `(...)` overriding). Comma is
tolerated as `&` (one line uses comma where the surrounding lines use `&`).

After parsing, `SemanticTrainer` runs multi-objective gradient descent on
the embedding table:

- **Dichotomy push** — for each `(A | B)`, target embed(A) toward `−embed(B)` and
  vice versa. Drives `cos(A, B) → −1`.
- **Context pull** — for each rhs atom X (excluding the dichotomy anchors),
  pull both A and B toward X (and X toward both). Forms neighborhoods
  shared by all dichotomies whose rhs contains X.

The two objectives are run as alternating updates within each epoch.

Three scoring primitives convert the trained table into NumberValue
intermediates the mandate verifier can check:

- `DichotomyOppositionScorer` — average `cos(left, right)` over all
  dichotomies. Target: near `−1`.
- `ContextClusterScorer` — within-context cosine minus between-context
  cosine, restricted to rhs atoms. Two atoms are "within-context" iff they
  appear in at least one common relation's rhs. Target: positive margin.
- `AxisAlignmentScorer` — for each rhs atom appearing in 2+ dichotomies,
  compute the average pairwise `|cos|` of those dichotomies' axis vectors
  (`embed(left) − embed(right)`). Average across qualifying contexts.
  Target: substantially above the random baseline.

## The single composition

`SemanticEmbeddingDemo` does it all in one composed pipeline:

```
parse semantics → train table → carve [dichotomy → context → axis → output] → verify mandates
```

The four mandates encode the structural claims:

| Mandate | Target | Tolerance | Range |
|---------|--------|-----------|-------|
| `dichotomy_opposite` (intermediate) | −0.70 | 0.30 | `[−1.00, −0.40]` |
| `context_clustered` (intermediate) | +0.15 | 0.10 | `[+0.05, +0.25]` |
| `axes_aligned` (intermediate) | +0.40 | 0.10 | `[+0.30, +0.50]` |
| `result` | +0.40 | 0.10 | `[+0.30, +0.50]` |

A single `MandateVerifier.verify(cg, mandates)` call produces the report.

## Results

```
parsed 107 relations, 326 unique atoms
trained 80 epochs (dichotomyLr=0.050, contextLr=0.005)

produced scores:
  dichotomy_score (avg cos of dichotomy pairs)  = −0.9622
  context_score   (within − between cosine)     = +0.1668
  axis_score      (avg |cos| of shared axes)    = +0.1473
  terminal value  (passed through from axis)    = +0.1473

mandate verification:
  OK   dichotomy_opposite     @c0_dichotomy, value −0.9622, target −0.70 ± 0.30
  OK   context_clustered      @c1_context,    value +0.1668, target +0.15 ± 0.10
  FAIL axes_aligned           target +0.40 ± 0.10  (no node produced a matching value)
  FAIL result                 target +0.40 ± 0.10  (terminal did not match)

structural claims: 2 satisfied / 4 total
```

**Two pass, two fail.** This is the framework working, not the demo failing.

- **Dichotomy opposition is strongly trained.** `cos = −0.9622` means
  embed(A) and embed(B) are nearly antipodal across all 107 dichotomies.
  The dichotomy-push objective converges sharply.
- **Context clustering is positive but modest.** Within-context cosine
  exceeds between-context cosine by 0.17. The clusters form, but they're
  loose — atoms shared across many contexts (`physical`, `disposition`)
  get averaged across competing pulls.
- **Axis alignment is at the random baseline.** For `d = 32`, the expected
  `|cos|` of two random unit vectors is `√(2/(πd)) ≈ 0.141`. The trained
  value is `0.147` — barely above noise. The trainer has no objective for
  axis alignment, so the property is not produced.
- **The terminal mandate fails for the same reason.** It passes through
  the axis score, which doesn't meet the `+0.40` bar.

## What this licenses

The framework name earns another claim: **mandates as exposed structural
contracts**. We made four claims about the embedding space and the
verifier returned which two hold. The two FAILs aren't bugs — they're
real diagnostic information saying "the trainer doesn't optimize for this
property; if you want it, add an objective."

This matters because in the v0 / v1 / v2 work, mandates were used as
constraints the carver had to satisfy *during search*. Here they're used
differently — as **post-hoc structural assertions over a trained
substrate**. The MandateVerifier doesn't care which use case; it just
checks whether values matching the expected ranges appear at the right
positions in the executed graph. Both modes are valid; v3 phase 1
demonstrates the second one.

It also reveals a real structural insight about the ontology. "Shared rhs
atom" is not the same as "same axis." `(solid | fluid)` and `(hard | soft)`
both carry `physical` in their rhs, but they're orthogonal physical
properties (phase vs. deformation resistance). Expecting their axis
vectors to align is a stronger claim than the ontology actually asserts.
The FAIL on `axes_aligned` is honest evidence that this conflation was a
mistake to bake into the trainer.

## Known limitations

- **Axis alignment is not trained for.** The trainer has dichotomy push and
  context pull. Axis alignment would need a third objective: for relations
  sharing a context atom, pull their axis vectors toward parallelism (or
  anti-parallelism — direction is undirected for dichotomies). v3 phase 2
  is the natural place for this if axis alignment is a load-bearing claim.

- **The "shared rhs atom" axis grouping is too coarse.** As above:
  `physical` is shared by many dichotomies that aren't actually the same
  axis. A tighter grouping criterion (e.g., shared qualifier head, or
  shared full rhs subtree) would be more honest.

- **Verification is non-local.** §6.1 is doing what §6.1 says it does:
  any node whose value matches the mandate's range counts as satisfaction.
  When mandate tolerance bands overlap with multiple nodes' produced
  values, the verifier returns the first match in iteration order. This
  is correct framework behavior; the demo's job is to make tolerance bands
  meaningful enough that the cross-matching encodes a real structural
  property rather than coincidental overlap.

- **Cache is read-only against trained state.** The "KV cache" framing
  promises bidirectional memory: write (K, V) on demand, query later by
  similarity. v3 phase 1 has the substrate (table + embed/lookup) but no
  primitives for runtime writes during a carving — that's what later
  cache variants (Q-W-K, K-V-Q, etc.) would build on.

- **No carver integration yet.** The composition is manually wired. The
  carver could compose the same pipeline given `dichotomy_score`,
  `context_score`, `axis_score`, and `output` as primitives in a
  TransformationGraph plus appropriate mandates — but exercising that path
  was deferred since the manual wiring was already enough to demonstrate
  the mandate-verification claim.

## Where v3 goes from here

Three candidate next steps, in rough dependency order:

1. **Axis-alignment objective.** Add a third training objective that
   explicitly aligns axis vectors of dichotomies sharing a context atom.
   Expected outcome: the same mandate set goes from 2/4 to 4/4. If it
   doesn't, that's a real signal that "shared rhs atom → parallel axes"
   isn't structurally available even with explicit training, which would
   inform the next ontology refinement.

2. **Carver-driven query pipeline on the trained substrate.** Build a
   small task — e.g., "given an atom, produce its dichotomy partner" —
   and let the carver assemble `EmbedSymbol → vector op → LookupSymbol`
   into a query pipeline driven by mandates. This exercises the cache as
   actual content-addressable memory, not just a substrate.

3. **Learned similarity.** The KV cache currently uses fixed cosine. The
   user's stated intent is to add learned similarity (Q-W-Kᵀ-style scoring)
   as the next cache variant. This is where attention-as-primitive starts
   becoming attention-as-orchestration: similarity itself becomes a
   trainable component.

The user's question at this writeup pause: whether axis alignment is
worth chasing now, or whether it's better to move on to carver integration
or learned similarity first. v3 phase 1 leaves the choice clean — the
substrate works, mandates verify what they should, and the next move
depends on which structural claim is most valuable to make load-bearing
next.

## Bottom line

v3 phase 1 went well in the same sense v0 / v1 / v2 went well: we now know
more sharply what to ask next. Symbol embeddings, total componentwise
arithmetic, learnable transforms, soft gating, semantic training, and
mandate verification all compose without NaN, without algebra-breaking
edge cases, and with structural claims that hold or fail honestly. The
framework's mandate machinery handles a new use case (post-hoc structural
assertions over a trained substrate) without architectural change — which
is itself meaningful evidence that the mandate abstraction was the right
shape.

The KV cache is now a real substrate, not just an aspiration.

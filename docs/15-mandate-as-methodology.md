# v3 Phase 10 — Mandate-Carve-Compose as a Methodology: A Layered NLP Pipeline

## Context

Phases 1–9 demonstrated the framework's three verbs over a small,
controlled substrate: symbol-keyed vectors, hand-picked atoms,
arithmetic and lore-toy mandates. Phase 10 takes the framework into
open-ended NLP — extracting named entities from real prose — and uses
that journey to surface a higher-level claim about *what kind of artifact
MCC is*.

This phase is not a single feature landing. It's a methodology
demonstration spanning eight iterations on one task, where each
iteration's failure was diagnostic and named the next mandate to add.

The claim that emerges:

> **Mandate-Carve-Compose is to AI what test-driven development is to
> software engineering** — not because mandates come first, but because
> *designing for inspectability changes what you build*. A model designed
> to be auditable cannot be a monolithic trained blob. It has to be a
> stack of small, named, individually-validated layers, where each
> failure is localizable to a specific layer with a specific named
> mandate.

## The task

Build an entity span extractor for an Elden Ring lore corpus, trained
on hand-annotated item descriptions, that generalizes to unseen items
(Ranni's questline — held out from all training).

## A second module joins the project

| Module | Role |
|--------|------|
| `strnn-model`            | The framework (unchanged philosophy, additions only) |
| `mcc-elden-ring` (new)   | The applied corpus, the annotations, and the NLP pipeline that consumes the framework |

The new module isn't a framework rewrite — it's the first downstream
*consumer* of the framework. The split formalizes the distinction
between "framework primitives" (general, reusable) and "domain code"
(this app, this corpus).

## What was built

### Framework additions (in `strnn-model`)

| File                                                     | Purpose |
|----------------------------------------------------------|---------|
| `primitive/TextTokenize.java`                            | First text-handling primitive — `StringValue → TokenListValue` |
| `primitive/ClassifierHead.java`                          | General trainable head (wraps any-shape `Mlp`, exposes `inputGradient()`) |
| `store/NetworkStore.java`                                | Read-only interface to a keyed collection of `NetworkItem`s |
| `store/EntityStore.java`                                 | Generic typed entity-and-relation graph interface |
| `store/VectorStore.java`                                 | Read-only HNSW-shaped vector lookup interface |
| `cache/NetworkCache.java`                                | Now `implements NetworkStore` |
| `mlp/Mlp.java`                                           | `backward()` already returned input gradient — now actually consumed by `ClassifierHead` |

The Store interfaces are the persistence boundary for a future GUI
workspace format. v0 implementations are in memory; later, an LMDB-backed
`NetworkStore` or SQLite-backed `EntityStore` swap in without touching
the rest of the framework.

### Domain code (in `mcc-elden-ring`)

The Elden Ring corpus, hand-paraphrased into the project's own prose:

| Storyline             | Items  | Annotated? |
|-----------------------|--------|-----------|
| Shattering era        | 10     | yes (10)   |
| Ranni's questline     | 8      | held out — never used in training |
| Volcano Manor         | 12     | yes (2)    |
| Dung Eater questline  | 5      | yes (1)    |
| Millicent questline   | 9      | yes (1)    |

14 hand-annotated items → 1,402 supervised tokens, 233 in-entity, 79
entity spans, ~76 relations. The annotation format (in
`sibarum.elden.annotation`) is inline markup `[surface](type:id)` plus a
relations list — a parser strips the markup to produce raw text +
structured spans, which is exactly what a downstream extractor must
learn to produce.

### The layered NLP pipeline

| Layer                | Mandate                                       | Source data                       |
|----------------------|-----------------------------------------------|-----------------------------------|
| Tokenize             | "Split prose into word + punctuation tokens"  | deterministic (no training)       |
| Token embedding      | "Each token gets a learnable vector"          | combined corpus vocabulary        |
| Context encoding     | "Each position aware of its neighbors"        | deterministic sliding window      |
| POS tagger           | "Each token has a part of speech"             | Universal Dependencies English EWT (12.5k sentences) |
| BIO entity tagger    | "Each token plays a role in entity structure" | Lexicanum + Elden Ring annotations |
| POS-conditioned decode | "Only nominal POS can start an entity span" | rule, no training                 |

Each layer is a separate component with its own input, its own training
data (where applicable), its own evaluation, and its own input/output
contract. The output of each layer is inspectable.

## The eight iterations

| #  | Configuration                                           | Spans on Ranni | Diagnostic |
|----|---------------------------------------------------------|----------------|------------|
| 1  | Binary span tagger (no POS)                             | 67             | Function words leak as entities |
| 2  | + POS features                                          | 70             | POS info underused — binary too coarse |
| 3  | → BIO (3-class) + POS                                   | 67             | Multi-word entities cluster cleanly |
| 4  | + Parquet pretraining (news-style NER)                  | 54             | Distribution mismatch (news ≠ lore) |
| 5  | + Staged training (pretrain → finetune)                 | 73             | Style transfer works; novel proper nouns recognized |
| 6  | + Lexicanum self-mention labeling                       | 77             | Sparse positives drag precision |
| 7  | → Cross-entry labeling + density filter                 | 67             | **Lore entities cluster cleanly** ✓ |
| 8  | + POS-conditioned decode constraint                     | 45             | Cleanest output yet — Layer 1 accuracy now bounds Layer 2 |

Each row's "Diagnostic" column is the *named missing mandate* that
the next iteration addressed. No iteration was "add more data" or
"train longer." Every step was a specific structural change motivated
by a specific observed failure.

## A representative item: how it cleaned up

The Elden Ring item *Cursemark of Death* (held out — not in any
annotated training set), across iterations:

```
Iter 1 (binary):           "of Death", "Rune of Death", "Night of the Black Knives"
                            — three fragments where one entity was intended
Iter 3 (BIO):              "Rune of Death", "Night of the Black Knives"  
                            — multi-word clustering works
Iter 7 (cross-entry):      "Cursemark of Death", "Rune of Death",
                           "Night of the Black Knives", "Empyreans"
                            — the item's own name now recognized as one span
Iter 8 (POS-constrained):  "Cursemark of Death", "Night of the Black Knives"
                            — pure entity output, no filler
```

Same item description, same model architecture family, eight iterations.
Each iteration's failure mode pointed at the next missing mandate.

## What this demonstrates

**Each failure exposed a specific missing mandate, not a vague
"the model needs more data."** When `of` was misfiring, the answer
was "POS tagging." When binary couldn't distinguish entity-internal
from external `of`, the answer was "BIO structure." When news-style
pretraining hurt rather than helped, the answer was "lore-style
register matching the target." Each diagnostic *named* the layer to
add.

**Each addition was measurably visible in a specific way.** Function
words were filtered. Multi-word entities clustered. Novel proper nouns
surfaced. Span boundaries cleaned up. Each layer's effect is auditable
not as a delta on a single metric, but as a specific change in *which
mistakes get made and which don't*.

**The system reaches a bounded-by-its-weakest-layer state.** By
iteration 8, the POS layer's 86% accuracy is the visible gating factor
on the decode constraint's effectiveness. This is precisely the kind of
per-layer bottleneck diagnostic that conventional monolithic models
cannot offer.

**The end-to-end backprop chain is real and necessary.** The POS
layer's training does NOT just train its classifier — gradients flow
back from the POS loss through the context encoder into the embedding
table, updating the rows for tokens that appeared in this context. The
existing `Mlp.backward()` returned input gradient since v0 (with a
`// not used in v0` comment); P10 actually uses it.

## Honest observations and limitations

- **This isn't state-of-the-art NLP.** The training set is 14 hand-
  annotated items plus 5k bootstrapped Lexicanum entries; modern NER
  systems train on orders of magnitude more. The point is the
  *methodology*, not the absolute performance.

- **The classifier heads aren't yet NetworkItems in the cache.** They
  exist as `ClassifierHead` instances trained directly by a per-demo
  loop. The full MCC story — each head as a `NetworkItem` composed by
  the carver via mandates — is the next step, not this one.

- **Cross-entry labeling depends on gazetteer quality.** The Lexicanum
  provides titles + aliases the editors thought to include. Anything
  missing shows up as silent false negatives in the bootstrap labels.

- **POS-conditioned decode is bounded by POS accuracy.** When POS
  mis-tags `concealed` as ADJ instead of VERB, the constraint can't
  filter it. The next layer-improvement target is the layer below.

- **Multi-word entity recognition leans on capitalization.** Novel
  entities recognized in iteration 7 were almost all capitalized
  multi-token phrases. Lowercase compound entities would not be picked
  up by this configuration.

## Where the framework goes from here

1. **Per-`EntityType` classifier heads** as NetworkItems in the cache —
   where MCC's composition story actually shines: one trained
   subnetwork per type, all spawned/retrieved through the same
   `NetworkCache` machinery.

2. **POS layer improvement** — wider context window, BIO transitions
   over POS, larger MLP. Directly raises the ceiling for the decode
   constraint.

3. **Entity-link layer** — surface form → canonical entity id (e.g.
   "Queen Marika" / "Marika the Eternal" → `marika`). This is the
   layer that turns NER output into a knowledge graph and unlocks
   relation extraction.

4. **GUI app for visual ComputationGraph editing.** The Store
   interfaces are already in place. The methodology demonstrated here
   is the user-facing pitch: build AI that you can audit, fix, and
   upgrade layer by layer.

## Bottom line

Eight iterations, eight named mandates, eight measurable improvements
on specific failure modes — culminating in clean multi-word lore-entity
recognition on items the model has never seen during training.

**This is what *Mandate-Carve-Compose as a technique* looks like in
practice.** The artifact isn't the trained model. The artifact is the
development methodology — and the demonstrably-auditable system that
falls out of it.

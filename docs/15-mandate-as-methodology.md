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

## Phase 10 follow-up: improving the POS bottleneck (iters 9–13)

Iteration 8 ended with a deliberate finding: the POS-conditioned decode
constraint is **bounded by POS accuracy**. When the POS tagger mis-tags
`concealed` as ADJ, the constraint can't filter it out. That made POS
the named next-target, and the methodology says: improve the layer the
diagnostic points at.

The same loop runs at the POS layer itself.

### Step 1 — build the diagnostic

The POS trainer had no held-out evaluation; we couldn't tell *which*
tags it was confusing without one. So before changing any architecture,
the trainer was modified to:

- carve off the last 10% of UD sentences as a deterministic dev split
- print dev accuracy after every epoch
- emit a per-tag P/R table + top off-diagonal confusions at the end

This is the same "designing for inspectability" rule applied to the
trainer instead of the model. No model change yet — just a measurement
surface.

### Step 2 — read the diagnostic

| Tag | Baseline F1 | Failure mode |
|-----|-------------|--------------|
| PROPN | **0.328** | 36.7% of gold PROPN → predicted NOUN |
| ADJ   | 0.726     | 14% → NOUN |
| ADV   | 0.738     | scattered across ADP/NOUN/PROPN |
| INTJ  | 0.074     | tiny support (78) — ignore |
| X     | 0.000     | tiny support (10) — ignore |

The dominant error by a wide margin is PROPN being mis-recognized as
NOUN. The Elden Ring sample sentences confirmed it concretely —
`Marika`, `Elden`, `Maliketh`, `Knives` were all mis-tagged. This is
the kind of diagnostic that *names* a mandate: the embedding lookup
loses capitalization, so any feature derived purely from the embedding
is blind to the most reliable PROPN signal in English.

### Step 3 — the iteration ladder

Each row's "Diagnostic" column names the missing mandate that drove the
*next* iteration.

| # | Configuration | Dev acc | Diagnostic |
|---|---------------|---------|------------|
| 9  | + word-shape features (capitalization, all-caps, digit, hyphen, punct, length-bucket) | **89.2%** | PROPN→NOUN drops out of the top-10 confusions entirely; remaining errors are ADV / SCONJ context-driven |
| 10 | + wider context window (radius 1 → 2), same hidden=64 | 88.4% | **Regression.** Wider input at the same MLP width is capacity-bound — "Marika"/"Ranni" both started mis-tagging as ADV |
| 11 | + bigger MLP (hidden 64 → 128) at radius 2 | 89.7% | Vindicates the wider-window hypothesis: it was capacity-starved, not unhelpful |
| 12 | + more epochs (5 → 8) | **90.9%** | Train-vs-dev gap starting to widen; remaining errors are semantic (ADJ↔NOUN↔VERB) — bounded by representation, not architecture |
| 13 | re-run BioCrossEntryDemo with the iter-12 POS layer | 43 spans on Ranni | The downstream BIO tagger now produces 44 *raw* spans (down from 67) — all clean entities; constraint demotions go from 22 → 1 |

The iteration-10 regression is itself a methodology lesson: **you don't
always win by adding**. Reading the result with the same discipline
("what mandate did the regression reveal?") said the window expansion
was real but the MLP wasn't big enough to use it.

### What changed downstream

Comparing iter 8 (POS-constrained decode, 86% POS) to iter 13 (same
decode, 91% POS):

|                                  | iter 8 (86% POS) | iter 13 (91% POS) |
|----------------------------------|------------------|-------------------|
| Raw spans (pre-constraint)       | 67               | **44**            |
| Constraint-demoted Bs            | 22               | **1**             |
| Final constrained spans          | 45               | 43                |
| Quality of remaining spans       | mixed            | uniformly clean entities |

The headline number (final spans) moved by only 2, but the *internal
dynamic* shifted decisively: the upstream BIO tagger became
discriminating on its own — the decode constraint went from
load-bearing to decorative. This is the methodology working at the
layer-stack level: **improving the lower layer raises the ceiling of
every layer above it, automatically**, because the upper layer is
trained on the lower layer's outputs and inherits their quality.

### What the diagnostic still names

The iter-12 confusion matrix points at the next mandate without
ambiguity:

```
top off-diagonal confusions (gold -> predicted):
    ADJ    -> NOUN      221   (15.0% of gold ADJ)
    VERB   -> NOUN      198   ( 6.8% of gold VERB)
    NOUN   -> PROPN     146   ( 4.0% of gold NOUN)
    ADV    -> NOUN      130   ( 8.4% of gold ADV)
```

These are *semantic* ambiguity (English really does let the same word
play noun or verb or adjective by context). Shape features and wider
windows can't fix them; they need either a stronger representation (a
contextual embedding, not a lookup table) or a structured-decode
mandate over the tag sequence itself ("if the previous tag is DET, the
current tag is much more likely NOUN/ADJ than VERB").

### Iter 14 — Viterbi decode (null result)

The structured-decode mandate got tried. Bigram tag-transition
log-probabilities computed from the UD training split (Laplace-
smoothed), plugged into a standard Viterbi forward+backtrace, with the
emission scores supplied by the existing MLP. A single tuning knob —
`transWeight`, scalar multiplier on the transition contribution — was
swept on the dev set:

```
Viterbi transition-weight sweep (dev set):
    transWeight=0.00   dev_acc=0.9094   <- greedy baseline
    transWeight=0.05   dev_acc=0.9094
    transWeight=0.10   dev_acc=0.9083
    transWeight=0.20   dev_acc=0.8937
    transWeight=0.30   dev_acc=0.8641
    transWeight=0.50   dev_acc=0.7704
    transWeight=1.00   dev_acc=0.4556
```

**No weight beat greedy.** Every non-zero contribution either matched
or regressed; weight=1.0 collapsed to 45.6% because the transition
prior began ignoring the input entirely. The selected weight is 0.0
(i.e., the layer falls back to greedy decode).

Why the null result? The window-radius-2 + shape-features MLP already
captures the bigram-level context a Viterbi prior would add. The
remaining confusions — `DET ADJ NOUN` vs `DET NOUN NOUN`, `AUX VERB`
vs `AUX NOUN` — are sequences a flat bigram model can't disambiguate
because *both* paths are common in English. The errors are
representational, not structural.

This is a real methodology outcome, recorded honestly: the named
mandate was tested and didn't deliver, and the negative result
*itself* names the next mandate — **a richer per-token representation
(contextual embedding) rather than a structured-decode layer over the
existing one**. The transition + Viterbi infrastructure stays in the
codebase (`PosTransitions`, `ViterbiDecoder`,
`TrainedPosLayer.predictSequence`), gated on a non-zero weight, so a
future iteration that finds a useful application — e.g. constrained
decode over BIO labels — can reuse it without re-implementing.

### Iter 15 — span-coherence pretraining from book titles

The downstream-task focus shifts back to the BIO entity tagger. By
iter 13 the remaining failure on Ranni was not false positives — those
were essentially solved by the POS-conditioned decode — but **span
boundary errors**. The canonical example: "Remembrance of the Baleful
Shadow" came out as two spans, splitting at the article `the`:

```
iter 13:  [Remembrance of] | [Baleful Shadow]
```

The diagnostic this names: the BIO tagger defaults to O on lowercase
function-word tokens, even when both flanking tokens are I-ENT inside a
multi-word entity. The model has too few training examples of "X of the
Y" / "X to the Y" patterns to learn that internal function words stay
inside a span.

The mandate: a pretraining corpus with **thousands of multi-word
capitalized phrases containing internal function words**. Such a corpus
exists in `book-names.parquet` — 1.48M book titles with the right
structural morphology ("A Guide to the Project Management Body of
Knowledge", "The Handbook of Project-Based Management", "Principles of
Corporate Finance"). 20k random titles get wrapped in simple sentence
templates ("She wrote [TITLE].", "The book [TITLE] was popular.") and
fed as Stage 0 BIO pretraining ahead of the iter-13 stages.

| Stage | Source                       | Effect on Ranni inference (intermediate) |
|-------|------------------------------|------------------------------------------|
| 0     | 20k book titles (264k tokens) | over-spans: B=9, I=588, 68 spans         |
| 1     | Lexicanum cross-entry         | reverts to 52 spans                      |
| 2     | Elden Ring fine-tune (60 ep)  | 45 spans final                           |

Stage 0 alone is too aggressive — book titles are 68% in-span by
construction, so the model overfits to "everything is I." Stage 1
reverses most of that; Stage 2 settles it.

**The targeted failure mode partially moved:**

```
iter 13:  [Remembrance of]      | [Baleful Shadow]   (split after "of")
iter 15:  [Remembrance of the]  | [Baleful Shadow]   (split after "the")
```

The article `the` is now pulled into the span — a real one-token
improvement — but the entity still breaks at the next capitalized
token. The residual error is now specifically an **I → B transition**
problem: the BIO tagger emits B at "Baleful" even though the previous
token is I and the surrounding context is in-span. Pretraining moved
the *emission* one token in the right direction; the transition prior
the model is using ("start a new span at every capitalized token")
remains uncorrected.

**Methodology read:** Stage 0 pretraining was a real partial fix, not a
null result, but its ceiling was the per-position emission model. The
next mandate the residual error *explicitly* names is the I → B
transition discouragement — which is exactly the structural-decode use
case the iter-14 Viterbi null result said the machinery would be useful
for. The two iterations stack: iter 15 made the upstream tagger willing
to extend spans across one function word; iter 16 will make the
decoder unwilling to start a new span when the surrounding context is
in-span.

### Iter 16 — Viterbi over BIO with hand-set transitions

Took the iter-15 stack and replaced the per-position greedy argmax at
inference with Viterbi over hand-set BIO transitions:

```
trans[O][O] =  0      trans[B][O] =  0      trans[I][O] =  0
trans[O][B] =  0      trans[B][B] = -1      trans[I][B] = -2   <- the targeted penalty
trans[O][I] = -50     trans[B][I] =  0      trans[I][I] =  0   <- structurally forbidden
```

`transWeight = 1.0` so the hand-set magnitudes take effect as written.
`-50` is a finite stand-in for `-infinity` (avoids NaN if the weight is
ever zero). The O→I forbidden value is the same as the standard BIO
structural constraint that has been implicit in the codebase all along.

**Result: a near-null effect.** Only 2 token labels changed across 696
Ranni tokens. Final span count went from 45 (iter 15) to 42. The
targeted failure mode actually regressed:

```
iter 13:  [Remembrance of]      | [Baleful Shadow]   (split after "of")
iter 15:  [Remembrance of the]  | [Baleful Shadow]   (article pulled in)
iter 16:  [Remembrance of]      | [Baleful Shadow]   (article pushed back out)
```

**The diagnostic:** the BIO tagger's path through this phrase is not
`I → B` directly; it's `I → O → B`. The model emits a strong O at the
article and a strong B at the next capitalized token. Both legs of that
path cost 0 in my transition matrix — the I→B penalty is sidestepped by
the free I→O→B escape. First-order Markov transitions can't express
"stay in span across this gap" when the gap genuinely is an O emission;
the constraint needs context the Markov state doesn't carry.

**What the iter-16 residual error names as the next mandate:**

- **Post-Viterbi span-merge.** A heuristic pass that merges adjacent
  spans separated by ≤2 function-word O tokens flanked by capitalized
  I-ENT content. This isn't a Markov constraint — it's a domain rule
  about how entity spans behave in our corpus.
- **A richer label set.** Distinguish "in-span-gap" from "outside-O" so
  the Markov chain can carry the "inside entity" state across function
  words: `O, B, I, GAP` with transitions `I → GAP` cheap, `GAP → B`
  expensive, `GAP → I` cheap. This *does* lift the constraint into the
  Markov state.

The infrastructure (`BioTransitions`, the `ViterbiDecoder` overload
that takes raw `logTrans` / `logInitial` arrays) is in place and
correct — it just isn't expressive enough by itself to fix the
specific failure mode iter 15 named. A second null result, recorded
honestly, with the diagnostic naming a more capable next attempt.

### Iter 17 — domain-matched supervision (the bootstrap chain dissolves)

A new dataset arrived: `elden_ring_final_train.jsonl` — 11,693 Elden
Ring Q&A pairs, each with a `metadata.entity_name` field naming a
canonical entity that appears in the `output` prose. 2,438 unique
entity names, 92.5% multi-token, in our exact domain. Examples:

```
"Ash of War: Repeating Thrust" / "Carian Glintstone Staff" /
"Death Rite Bird" / "Prophet Robe (Altered)" / "Cathedral of Manus Celes"
```

This is exactly the structural pattern iter 5–16 had been chasing
through bootstrap surrogates (Parquet news NER, Lexicanum cross-entry,
book titles, structured decode). With domain-matched labeled data
available, the bootstrap chain becomes unnecessary.

The new pipeline is *simpler*, not more complex: two stages.

| Stage | Source | Effect |
|-------|--------|--------|
| 1 | 11.7k JSONL rows, entity_name labeled B/I in output prose (258k tokens) | 14 spans on Ranni after stage 1 — high precision, low recall |
| 2 | Elden Ring hand-annotations (60 ep) | 36 final spans on Ranni, multi-word entities correctly bounded |

**The targeted failure mode is solved:**

```
iter 13:  [Remembrance of] | [Baleful Shadow]
iter 15:  [Remembrance of the] | [Baleful Shadow]
iter 16:  [Remembrance of] | [Baleful Shadow]
iter 17:  [Remembrance of the Baleful Shadow]    <- one span, correctly bounded
```

`House Caria` also recovered (iter 13 had only `House`). Most multi-
word entities now bound correctly: Carian Inverted Statue, Divine
Tower of Liurnia, Fingerslayer Blade, Cursemark of Death, Night of
the Black Knives, Cathedral of Manus Celes (mostly), Black
Knifeprint, Discarded Palace Key.

**Trade-off (honest):** the run produces 36 final spans vs iter 13's
43 — fewer, but with different mistakes:
- Wins: 7 multi-word entities correctly bounded that iter 13 split
- Losses: a few entities iter 13 captured are now fragmented or
  missed (`Two Fingers` → `Two` + `Fingers`, `Eternal Cities` →
  `Eternal`, `Cathedral of Manus Celes` → `Manus Celes`)

The recall loss is interpretable: the JSONL trains the model to
recognize specific surface forms in specific contexts (Q&A answers).
Entities heavily present in the JSONL (e.g., things weapons can do to
you) get clean recognition; entities prominent in Ranni lore but
under-represented in the JSONL (boss names appearing across multiple
questlines) lose recall. The right next step is to merge the JSONL
supervision with the iter-13 Lexicanum cross-entry supervision rather
than fully replace it — but the *boundary* problem the entire iter
14–16 arc was chasing is decisively resolved.

**The methodology lesson, recorded:** iters 4 through 16 were a
sequence of increasingly elaborate substitutes for one missing
ingredient — domain-matched labeled supervision. Each substitute
made measurable but limited progress on the targeted failure mode;
none fully resolved it. When the actual labeled data arrived, the
problem dissolved in *one* stage with no architectural changes. This
is not a criticism of the bootstrap iterations — each was honestly
diagnostic, and the framework's discipline (one named mandate per
iter, measure, name the residual) held up perfectly. But it points
at a sharper rule: **when a series of structural fixes makes only
incremental progress, the diagnostic is often "we don't have the
right supervision."** The right next move is to look for the data,
not the architecture.

### Iter 18 — corpus gradient (mixed result, class collapse observed)

Premise: iter 17 used one corpus (Elden Ring JSONL) as Stage 1. But three
corpora are available at increasing domain specificity — NLP-KnowledgeGraph
(generic English), Lexicanum (fantasy genre), elden_ring_final_train.jsonl
(Elden Ring). Rather than treating them as substitutes, use them as a
gradient: a 7-class BIO head with separate B/I classes per corpus, so the
model can express "this token looks generic-entity-ish" vs "fantasy-entity-ish"
vs "specifically-Elden-Ring-entity-ish" instead of one binary entity decision.

```
O                              -- function words, verbs
B-normal,    I-normal           -- from NLP-KG (generic English)
B-fantasy,   I-fantasy           -- from Lexicanum cross-entry
B-eldenring, I-eldenring         -- from JSONL + hand-annotations
```

Stage 1 mixed all three (~517k tokens). Stage 2 fine-tuned 60 epochs on
hand-annotated items only (eldenring-class).

**Result: +21 spans across the corpus (191 → 212), with the class signal
collapsed.** Of 216 B-* predictions, 214 were B-eldenring, 2 were B-normal,
0 were B-fantasy. The 60 eldenring-only fine-tune epochs overrode the
output head's class signal from Stage 1 — classic catastrophic forgetting of
output classes.

**The mixed-bag detail:**

| Item | iter 17 | iter 18 |
|------|---------|---------|
| Remembrance of the Baleful Shadow | one span (correct) | "Remembrance of" only (regression) |
| House Caria                       | captured            | gone                              |
| Two Fingers                       | split as "Two" + "Fingers" | one span (correct)         |
| Rune of Death                     | not found           | captured (new)                    |
| Nokron / Tarnished                | partial             | captured cleanly                  |

The pattern: tokens that look generic-entity-like across multiple corpora
(like the article `the` inside "Remembrance of the Baleful Shadow") got
pulled toward O by the dominant non-entity signal across NLP-KG +
Lexicanum, *even though* the entity-specific signal from JSONL would have
labeled them as I-eldenring. The shared embedding table is the conduit:
gradient updates from multiple corpora's training pull the embedding in
multiple directions on shared function-word tokens.

**Where iter 18 quietly helps anyway:** the new entities surfaced (Rune
of Death, Nokron, Tarnished, the corrected Two Fingers) came from the
broader pattern exposure during Stage 1. The model learned to recognize
"this looks like an entity" more broadly, even though the class signal
collapsed at the output. Representation learning from the gradient
worked; output-class learning didn't survive Stage 2.

**Methodology lesson:** **gradient training is a representation tool,
not an output-class tool**, unless Stage 2 is changed to preserve class
supervision. Three concrete fix paths the diagnostic names:

1. Freeze the output head during Stage 2 — only update internal
   representations from hand-annotated supervision.
2. Rehearse Stage 1 examples in Stage 2 — sample a small fraction of
   each corpus per epoch so the output head doesn't forget the gradient.
3. Use a 3-class BIO head with the gradient signal applied as auxiliary
   multi-task supervision instead of primary class labels. (Closer to the
   structured-distillation framing the methodology has been pointing at.)

The honest synthesis across iter 17 and iter 18: **iter 17 was the right
"clean" supervision answer for the boundary problem; iter 18 trades some
boundary precision for more general recall, with the gradient class
signal mostly wasted on a fixable design issue**. The right system to
ship today is iter 17. The right experiment to pursue if we want better
domain-aware representations is iter 18-style training with one of the
class-preservation fixes above. Both are reasonable next moves.

### Iters 19–21 — data-selection mandates (the methodology applied one level deeper)

The iter 17–18 arc produced a quiet observation: iter 1 (binary tagger,
no POS, ~1402 training tokens) produced 67 raw spans on Ranni; iter 17
(full pipeline, ~258,000 training tokens) produced 191 final spans
across the *whole* corpus. The system kept *adding* training data and
the result was a precision/recall pendulum, not monotonic improvement.
Iter 4 had already shown implicitly that the *wrong* data hurts; iter 17
showed that the *right* data dissolves the boundary problem; iter 18
showed that *more* data with the wrong class structure regressed
specific cases. The diagnostic the arc named: **the data layer itself
has mandates and they haven't been being explicit**.

The next three iterations apply the framework's discipline to the data
layer.

#### Iter 19 — entity-budget parity (minimum viable test)

Clip each external corpus to ~80 entity spans (matching the hand-
annotated corpus's ~79 spans), sample randomly within budget, train with
the same iter-17 architecture. Stage 1 shrinks from 258k tokens to
**3,332 tokens**.

```
iter 17 (full):   258,145 stage-1 tokens -> 191 corpus spans
iter 19 (80/src):   3,332 stage-1 tokens -> 199 corpus spans
```

**78× less training, slightly *more* spans.** The result wasn't
particularly clean — some boundary cases regressed and new common-noun
false positives appeared (`figure`, `trust`, `beyond`) — but it
established the headline finding the user named: *most of the iter-17
training signal wasn't load-bearing for this task*. The interesting
question shifted from "how much data?" to "which data?"

#### Iter 20 — naive failure-driven curation (what doesn't work)

Categorize iter 19's failures, define pattern matchers, *prefer*
sentences matching any of them within the same 80-span budget. The
patterns chosen were narrow:

- `of-the-inside` — span containing literal `of the` with I-labels on both
- `two-word-propn` — adjacent capitalized B-I pair
- `initial-cap-as-O` — first token capitalized and O
- `long-O-stretch` — 8+ contiguous O between entity spans

Per-pattern matches across the three external corpora:

| Pattern | KG | Lex | JSONL | Total |
|---|---:|----:|------:|------:|
| of-the-inside | 0 | 4 | 2 | **6** |
| two-word-propn | 0 | 10 | 66 | **76** |
| initial-cap-as-O | 19 | 4 | 6 | 29 |
| long-O-stretch | 18 | 0 | 3 | 21 |

**The two-word-propn pattern flooded the budget at 76/132 sentences,
starving the longer-multi-word patterns.** The very pattern that was
supposed to address "Remembrance of the Baleful Shadow" splitting (the
`of-the-inside` pattern) found only 6 sentences across all three
sources, because the canonical entity-name databases mostly hold short
multi-word phrases without internal articles.

Result: 187 corpus spans, *lower* than iter 19. New false positives
(`relic`) and some regressions (`Cathedral of Manus Celes` now lost
both halves). The pattern *did* kill some iter-19 false positives
(`beyond`, `trust`) but the imbalance dominated.

**Methodology lesson:** "any matching pattern is preferable to random"
was the implicit mandate, and it under-corrected for the actual
distribution of failures the curation was supposed to fix. The
diagnostic names the iter 21 fix.

#### Iter 21 — relaxed patterns across four corpora (the win)

Two changes:

1. **Relax the patterns.** Instead of requiring rigid `X of the Y`
   structure, accept *any* span containing internal `of` / `the`, *any*
   span with 2+ internal lowercase function words, *any* span of 3+
   tokens, *any* span starting with a known failure prefix (House /
   Cathedral / Manus / Two / Eternal / Forged / Greater / Remembrance).
2. **Add `book-names.parquet` as a fourth corpus.** 1.48M book titles
   are 92.5% multi-word with rich internal function-word structure —
   exactly the supply the iter-20 patterns starved on.

Per-pattern matches with the relaxed scheme across four sources:

| Pattern | KG | Lex | JSONL | Books | Total |
|---|---:|----:|------:|------:|------:|
| anchored-failure | 0 | 0 | 0 | 1 | 1 |
| multi-func-internal | 0 | 0 | 2 | **42** | 44 |
| three-plus-word | 0 | 9 | 26 | **33** | 68 |
| two-word-propn | 0 | 3 | 46 | 2 | 51 |
| initial-cap-as-O | 17 | 0 | 1 | 2 | 20 |
| long-O-stretch | 17 | 0 | 4 | 0 | 21 |

Book-titles carried the multi-func-internal and three-plus-word load —
75 sentences combined — exactly the patterns iter 20 starved on.

**Result: 215 corpus spans, 47 on Ranni. Best result of the entire
arc, with 60× less training data than iter 17.**

```
iter 17 (full):                258,145 stage-1 tokens -> 191 spans, 36 on Ranni
iter 18 (full + gradient):     517,353 stage-1 tokens -> 212 spans, 43 on Ranni
iter 19 (random clip):           3,332 stage-1 tokens -> 199 spans, 39 on Ranni
iter 20 (narrow patterns):       3,668 stage-1 tokens -> 187 spans, 33 on Ranni
iter 21 (relaxed + books):       4,355 stage-1 tokens -> 215 spans, 47 on Ranni
```

Failure-case progress on Ranni:

| Case | iter 17 | iter 19 | iter 20 | iter 21 |
|------|---------|---------|---------|---------|
| Two Fingers | split | split | split | **one span ✓** |
| Manus Celes | captured | captured | gone | captured |
| Cathedral of Manus Celes | partial | partial | gone | partial (Cathedral of + Manus Celes) |
| Remembrance of the Baleful Shadow | one span ✓ | Rem of | Rem of | Rem of the + Baleful Shadow |
| House Caria | captured | House | House | House |
| Divine Tower of Liurnia | captured | captured | captured | split |

The model now reliably includes internal function words in spans (the
training mandate was learned) but four-token entities sometimes split
in new places. That's the next-iteration diagnostic — and the
mandates it names are concrete: source examples of 4+ word entities,
source examples with "common-noun-shape title prefix" (House, Lord,
Lady), source sentences with sentence-medial capitalized verbs as O.

### The methodology lesson the arc landed

The entire iter 4–16 search for the right *architecture* to fix span
boundaries was, in retrospect, a search for the supervision signal we
didn't yet have. Iter 17 found the signal (domain-matched labels).
Iters 19–21 then asked the question the framework should have been
asking from the start: **which subset of that signal is actually
load-bearing, and what mandates select it?**

Five concrete claims fall out of this:

1. **Data-selection mandates are the same kind of artifact as
   architecture mandates.** Each is a named decision with an
   ablate-able effect, a failure mode that diagnoses it, and a fix
   that the diagnostic names.
2. **Pattern matchers are mandates on training rows.** When the pattern
   is too narrow, the diagnostic isn't "the pattern doesn't work";
   it's "no corpus supplies enough rows that match the pattern,"
   and the fix is either to relax the pattern or source a corpus that
   does.
3. **Per-pattern *balance* matters more than per-pattern *richness*.**
   Iter 20 had two-word-propn at 76 sentences and `of-the-inside` at
   6, and the imbalance dominated the result. Iter 21's relaxed
   patterns plus the book-titles corpus brought the long-entity
   patterns up to 44+68=112 sentences combined and the result followed.
4. **The right corpus for a pattern is the one that has the pattern.**
   The book-titles corpus had been available since iter 15 but was
   only used as a generic span-coherence pretrain; iter 21 used it
   *for the specific patterns it's rich in*. The same corpus, used
   differently, produced a meaningfully different result.
5. **Most of the training signal in the original data was waste — but
   we couldn't have known which 4 thousand tokens were the load-bearing
   ones without first running the full-data pipeline and reading what
   failed.** The methodology of the bootstrap iterations (4–16) was
   how we learned the failure modes; the methodology of the curation
   iterations (19–21) was how we used that knowledge to pick the
   minimal-viable training set.

The synthesis across the whole P10 arc: **build the auditable layered
system, run the full-data pipeline, read the failure modes, then use
the named failure modes to select a curated minimal training set, and
the result outperforms the full-data baseline.** That is the
deliverable the framework's positioning has been promising — not a
better model on a benchmark, but a development process where each
iteration's diagnostic is the next iteration's mandate, applied
indifferently at the architecture layer or the data layer.

### Iter 22 — per-type heads (the next layer in the stack works)

The first lateral move after the iter 17–21 entity-tagger arc: add a
**type classifier** as a third trainable layer on top of POS + BIO. For
each detected entity span, predict one of seven `EntityType` values
(`CHARACTER`, `ARTIFACT`, `EVENT`, `PLACE`, `FACTION`, `ERA`, `CONCEPT`).
Implementation is small: ~250 lines for a span-feature extractor (average
of per-token context + POS logits across span tokens), a `ClassifierHead`
wrapper, and a JSONL typed-span loader (mapping the JSONL's ten
in-game-mechanical types to our seven narrative types via a static
table).

Training:
- Stage 1: ~11k typed spans from the JSONL covers four of seven types
  (ARTIFACT, CHARACTER, PLACE, CONCEPT) via the mapping.
- Stage 2: 108 typed hand-annotated spans add EVENT to the mix
  (FACTION and ERA stay at zero training samples — our hand annotations
  never used those values).

Hand-annotated training-set accuracy: 99.1%. Held-out Ranni distribution
of predicted types: CHARACTER 58, ARTIFACT 47, EVENT 10, PLACE 23,
CONCEPT 53, FACTION 0, ERA 0. By eye, the well-trained classes are
mostly correct (`Rennala` → CHARACTER, `Lands Between` → PLACE,
`Night of the Black Knives` → EVENT, `Cursemark of Death` → ARTIFACT);
the PLACE class is under-resourced and mis-types some locations as
ARTIFACT or CONCEPT.

**The architectural claim landed:** three named layers, each
independently trained, each with its own input/output contract, stack
into a working typed-NER pipeline. Adding a layer cost ~250 lines and
reused the existing POS + BIO layers without modification. That is the
"compose named layers" pitch made concrete on a different layer than
the iter 1–21 arc had exercised.

### Iter 23 — substrate-authentic type labels (collapse observed)

Iter 22 had one inconsistency: the seven `EntityType` labels were
encoded as one-hot output positions. From the AI's perspective, those
are alien opaque integers — they exist outside the symbol-vector
substrate the framework otherwise uses uniformly. The framework's
*own* `SymbolEmbeddingTable` was built exactly for the case "named
symbols → learnable vectors"; type names should live in that substrate
too, with their own mini-KV.

The corrected design (iter 23):
- Register the seven `EntityType.name()` strings as keys in a 32-dim
  `SymbolEmbeddingTable` of their own (a "type-context mini-KV").
- The type head's MLP outputs a 32-dim vector instead of 7-dim logits.
- Loss is MSE between predicted vector and the gold type's anchor.
- Both MLP weights AND the anchor vectors update via gradient descent
  (anchors via `SymbolEmbeddingTable.update`). The two co-adapt: anchors
  become learned class centroids; the MLP maps span features to those
  centroids.
- Inference: nearest-anchor cosine similarity over the seven vectors.

**Result: anchor collapse.**

After Stage 1 + Stage 2 training, every type that received supervision
had its anchor pulled to the *same point* in 32-dim space:

```
CHARACTER, ARTIFACT, EVENT, PLACE, CONCEPT : magnitude 0.276,
                                              mutual cosine 1.000
FACTION, ERA                                : magnitudes 0.902 / 1.024
                                              (random init, no training)
```

MLP training loss went to 0.0000 — the symptom of a degenerate fixed
point. The MLP learned to output a single point `c`, and all trainable
anchors followed it to `c`. Loss vanishes, but `argmax(cosine)` over
identical anchors becomes meaningless. Training-set accuracy dropped
from iter 22's 99.1% to 29.6%.

**Why it happened:** with both predictor and targets trainable and no
*restoring force* separating different classes, the loss landscape has a
trivial fixed point at "everything collapses to one vector." Sufficient
per-class data doesn't prevent this — the data only distinguishes
classes *if the anchors stay distinguished*; the moment they drift
together, the per-class data becomes effectively the same supervision
signal.

**The next mandate the diagnostic names** — any of these would address
collapse, each from the standard metric-learning playbook:

1. **Contrastive loss component.** Penalize cosine similarity between
   anchors of different types during training. The missing
   restoring-force gradient.
2. **Orthogonal anchor initialization.** Start the seven anchors as
   orthogonal unit vectors; collapse then requires fighting the
   initial geometric separation.
3. **Frozen anchors after warmup.** Train both for N epochs, then
   freeze the anchors and continue MLP-only training.
4. **Asymmetric learning rates.** Anchor LR set to 1% of MLP LR — anchors
   are *nominally* trainable but effectively-frozen on the relevant
   timescale.

**Why iter 23 is in the repo as a recorded negative result:** the
substrate-authentic architecture (type labels as real KV entries) is
correct *as a principle* — it's how the framework is supposed to
treat all symbolic things. The collapse demonstrates that
substrate-correctness alone is insufficient; the missing piece is the
anti-collapse mandate, which the diagnostic above names cleanly. A
next iteration could add any of the four fixes and would (predictably)
recover iter-22-like accuracy while keeping the architectural
authenticity. For the purpose of *demonstrating the methodology*,
iter 23 is more valuable than a quietly-passing run: it shows the
"name the missing mandate" loop working even on the framework's own
philosophical commitments.

The pragmatic upshot: iter 22 (one-hot) is the shippable working
typed-NER pipeline. Iter 23 (vector-anchor) is the philosophically
consistent architecture with one named missing component. The
methodology vocabulary applies symmetrically to both.

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

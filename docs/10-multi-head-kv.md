# v3 Phase 5 — Multi-Head KV (Key-Network preparation)

## Context

The framework has been converging on a Key-Network shape — a cache that
returns *behaviour* rather than vectors. Phase 4 closed the
carver-driven end-to-end milestone with a single KV chain. The next
load-bearing question is structural multiplicity: **can the carver
pick among parallel cache chains?** Without that, "the substrate has
multiple views" is just an architectural daydream.

Phase 5 lands the multi-head substrate. Two parallel KV chains
(separate `SymbolEmbeddingTable`s, separate `EmbedSymbol`s, separate
`VectorTransform` bridges, separate `LookupSymbol`s) coexist in one
transformation graph. The carver routes through one or the other per
mandate. Inline training tunes whichever chain the carver picked.
Independent chains retain their specialization across mandate
switches.

The scope was deliberately narrowed away from emergence: head
selection is **driven by hand-biased edge stats** in this phase, not
by accumulated learning over a multi-session loop. The mechanism for
substrate-learning (edge-stats feedback from `InlineTrainer`) is
wired in and exercised, but the diagnostic that *proves* heads
specialize via repeated use is deferred. This phase proves the
plumbing, not the emergence.

## What was built

| Deliverable | What it does |
|-------------|--------------|
| `cache/CachedItem` + `cache/EmbeddingItem` | Future-proofing scaffold. Sealed interface for items stored in a KV; today only `EmbeddingItem` (a vector). Tomorrow, `NetworkItem` drops in alongside, holding a carved subgraph instead of a vector — the actual Key-Network landing. |
| Source-aware forward anchors | `Map<TransformationNode, Value>` replaces `Map<ValueType, Value>` in the carver's pre-pass. Multi-head substrates (multiple TG nodes producing the same `ValueType`) keep distinct anchors per source. `inferInputs` now takes the candidate's `TransformationNode` so trainable inversion can pick the anchor produced by *its* TG-upstream, ranked by edge-stat score. |
| `EdgeStats.setMeanScore` / `reset` | Demo-time bias and reset. Lets a multi-session demo configure preferences without rebuilding the TG. |
| `InlineTrainer.Result.score(maxSteps)` + `applyEdgeFeedback` | Closes the loop: per-session reward (1.0 for PASS, decayed for FAIL) → `EdgeStats.update` on each traced edge. The substrate's preferences accumulate from successful carvings. |
| `MultiHeadCarvedDemo` | The diagnostic. Three sessions over one substrate: chain A trains "hot→cold"; chain B trains "hot→warm"; chain A re-verifies "hot→cold" still works without further training. |

About 250 lines of new code plus ~80 lines of carver refactor.

## The substrate

H = 2 heads, each a complete embedding-bridge-lookup chain:

```
                          ROOT (StringValue "hot")
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
            EmbedSymbol_A                   EmbedSymbol_B
            (table A)                       (table B)
                  │ MatrixValue                  │ MatrixValue
                  ▼                               ▼
            VectorTransform_A               VectorTransform_B
            (trainable bridge)              (trainable bridge)
                  │                               │
                  ▼                               ▼
            LookupSymbol_A                   LookupSymbol_B
            (table A nearest)               (table B nearest)
                  │ StringValue                  │ StringValue
                  └───────────────┬───────────────┘
                                  ▼
                       StringOutputPrimitive
                       (terminal, shared)
```

`TransformationGraphBuilder` wires every type-compatible
`(from, to)` pair — 22 edges total, including all the cross-head
combinations the carver doesn't want. Edge stats are what tell the
carver which combinations are good.

## The three-session diagnostic

### Session 1 — chain A trains "hot → cold"

```
bias chain A edges:  score = 1.0
bias chain B edges:  score = 0.0
mandate: result = StringValue("cold"), root = StringValue("hot")

carved chain:
  c4_embedA [embed-symbol]      (head A)  slot0<-ROOT
  c3_bridgeA [vector-transform] (head A)  slot0<-c4_embedA
  c1_lookupA [lookup-symbol]    (head A)  slot0<-c3_bridgeA
  c0_output  [string-output]              slot0<-c1_lookupA
routed through: head A (expected A)

inline training:
  step   0: terminal='cool'  mandate:FAIL
  step  25: terminal='cold'  mandate:PASS

session 1: PASS in 25 steps; edge feedback = 1.00
```

### Session 2 — chain B trains "hot → warm"

Reset all edge stats. Bias chain B = 1.0, chain A = 0.0.

```
mandate: result = StringValue("warm")

carved chain:
  c3_embedB [embed-symbol]      (head B)  slot0<-ROOT
  c2_bridgeB [vector-transform] (head B)  slot0<-c3_embedB
  c1_lookupB [lookup-symbol]    (head B)  slot0<-c2_bridgeB
  c0_output  [string-output]              slot0<-c1_lookupB
routed through: head B (expected B)

inline training:
  step   0: terminal='fire'  mandate:FAIL
  step  25: terminal='warm'  mandate:PASS

session 2: PASS in 25 steps; edge feedback = 1.00
```

Chain B's bridge starts at a different random init (seed 11 vs.
chain A's seed 7) and against a different embedding table — its
initial output is `fire`, not `cool`. Both converge in 25 steps.

### Session 3 — chain A still works

Reset edge stats. Bias chain A = 1.0 again. Run mandate 1 ("hot →
cold") through the inline trainer with the SAME `maxSteps = 500`.

```
inline training:
  step   0: terminal='cold'  mandate:PASS

session 3: PASS in 0 steps; edge feedback = 1.00
```

**Zero training steps required.** Chain A's bridge retained its
specialization through session 2's training of chain B. The chains
are genuinely independent — training one does not perturb the other,
because their bridges are separate `VectorTransform` instances and
their embedding tables are separate `SymbolEmbeddingTable` instances.

## What this licenses

**Structural multiplicity works.** The carver navigates a substrate
with parallel chains, picks one based on edge stats, and the inline
trainer drives whichever chain was picked. This is the multi-view
plumbing Key-Network needs.

**Per-source forward anchors are the right abstraction.** The
multi-head failure mode for the old `Map<ValueType, Value>` anchor
table was that `EmbedSymbol_A` and `EmbedSymbol_B` both produce
`MATRIX` and first-found-wins-and-discards-the-other. With
per-source-node anchors plus TG-aware bridge inversion (the bridge
picks its anchor from its highest-scoring TG-upstream), each head's
trainable gets paired with its own head's upstream. No new value
types; the topology disambiguates.

**Edge-stats feedback closes the substrate-learning loop.** Each
session's reward propagates back to its traced edges. After many
sessions, the substrate's edge stats encode "which carvings were
worth their compute." Phase 5 wires this in and exercises it; phase
5b (deferred) will demonstrate emergent head specialization driven
by accumulated edge-stat reward alone — without hand-bias.

**The `CachedItem` scaffold reserves the type slot for
`NetworkItem`.** Today the cache stores vectors. Tomorrow the cache
stores carved subgraphs. The interface holds the place without
requiring a refactor when the second variant lands.

## Honest observations and limitations

- **Head selection is hand-biased, not learned.** The carver's edge
  ranking is correct, but in this demo the rankings are demo-set,
  not earned. The "emergence" demo (5b) would start from random
  init and equal stats, run N sessions over a mandate sequence, and
  show that heads specialize via accumulated reward. Doable; not
  done yet.

- **Heads share a vocabulary in this demo.** Both tables embed all
  eight atoms. The interesting alternative — heads with *disjoint*
  vocabularies (one thermal, one directional) — would force
  routing by atom rather than by edge-stat bias, which is closer to
  how real-world multi-head memory works. Future demo.

- **The cross-head edges in the TG are vestigial.** `bridgeA →
  bridgeB`, `embedA → lookupB`, etc., all exist (any-to-any modulo
  type), but the carver never picks them because edge stats steer
  away. Pruning them explicitly would simplify the substrate at the
  cost of giving up the "the substrate is rich, the carver
  navigates" narrative.

- **`CachedItem` doesn't refactor `SymbolEmbeddingTable` yet.** The
  table still stores `double[]` directly. The interface exists to
  hold the type slot; the actual storage refactor happens when
  `NetworkItem` lands.

- **No multi-trainable end-to-end.** This phase, like 3 and 4, has
  exactly one trainable per gradient path (the bridge). Heads are
  parallel, not chained. Multi-trainable / autograd-on-carving
  remains deferred.

## Where v3 goes from here

Phase 5 is preparation. The next steps the design has been pointing
at:

1. **Multi-head emergence (5b).** Replace hand-bias with edge-stats
   feedback over a long mandate sequence. Heads specialize
   naturally; one ends up handling thermal-style mandates, the
   other handles directional-style. *The substrate learns the head
   assignment.*

2. **`NetworkItem` and `CachedNetworkPrimitive`.** A `CachedItem`
   variant that holds a frozen `CarvingResult` plus trained state.
   A `Primitive` that wraps it and exposes `(query input) →
   network output`. The carver can now use prior carvings as
   building blocks. *This is the Key-Network milestone.*

3. **Multi-trainable end-to-end.** When `NetworkItem`-bearing
   carvings are themselves wired into bigger carvings, gradient
   flow across the boundary becomes the question. Autograd on the
   carving graph; primitive-level `backward(outputGrad) →
   inputGrad`.

## Bottom line

Multi-head KV works end-to-end through the framework. The carver
picks among parallel cache chains; inline training tunes the chosen
chain; heads retain independence across mandate switches; the
substrate's edge stats accumulate the per-session reward. The
plumbing for Key-Network's "multiple views, carver selects" piece is
landed and exercised.

The framework's verbs continue to compose without architectural
revision. Mandates specify, the carver assembles, and the
composition runs — even when the substrate has multiplicity the
carver has to navigate. Each phase has earned a slightly more
expressive `Compose`, and the next phase finally upgrades what
`Value` can be: from vectors to networks.

# v3 Phase 6 — Key-Network

## Context

Phases 1 through 5 built up the substrate machinery: a KV cache, a
trainable bridge, a carver that assembles end-to-end pipelines, a
generic inline trainer, multi-head support. Every demo to this point
stored *vectors* in its caches and returned vectors from its lookups.

Phase 6 lands the architectural step the framework has been
converging on since the v3 P1 work: **the cache stores entire trained
subgraphs as items, and the carver composes them.** A stored item is
no longer a vector to be returned — it is a behaviour to be invoked.

The renamed shape: instead of Key-Value, **Key-Network**. The
substrate's stored items return functions parameterized by query;
the carver retrieves and composes them through the same machinery
that composes any other primitive.

## What was built

| Deliverable | What it does |
|-------------|--------------|
| `cache/NetworkItem` | A `CachedItem` variant that wraps a trained `ComputationGraph` plus the `RootBinding` that identifies its input slot. `execute(input)` rebinds the root and runs the inner graph, returning the terminal value. Multi-input cached networks are a straightforward extension; the simplest case (one input, one output) is enough for this demo. |
| `cache/CachedNetworkPrimitive` | A `Primitive` that wraps a `NetworkItem`. From the outer carver's perspective, this is just another deterministic primitive — declared input/output types match the inner network's signature. The carver places it like anything else. |
| `carving/RootBinding` (extracted) | Promoted from a private record inside `BackwardChainingCarver` to a public type. `CarvingResult.rootBindings()` exposes the carver's bindings so `NetworkItem` can rebind them between executions. |
| Carver `inferInputs` for `CachedNetworkPrimitive` | The inversion is forward-evaluation: try the root input first, then every per-source forward anchor of the right type. If any input produces the desired target when fed through the cached network, return it as the inferred input. This is what lets the carver *chain* cached networks. |
| `demo/KeyNetworkDemo` | Pre-train two inner networks (`hot→cold` and `cold→freeze`), each as a separate carving + inline training run; wrap each as `NetworkItem` + `CachedNetworkPrimitive`; build an outer TG containing both; submit a single result mandate (`freeze`) with root (`hot`); execute the composed pipeline. |

About 250 lines of new code plus a small carver refactor.

## The demo

### Pre-training: two cached subgraphs

Each inner network is a complete `embed → bridge → lookup → output`
carving trained inline against a single mandate.
The vocabularies are **deliberately disjoint** at the input atom:

```
hotCold inner network:
  vocabulary = [hot, cold, warm, cool, fire, ice, burn, freeze]
  mandate     = "hot" -> "cold"
  converges in 25 steps

coldFreeze inner network:
  vocabulary = [cold, freeze, ice, winter, snow, frost, chill, cool]    // no "hot"
  mandate     = "cold" -> "freeze"
  converges in 1 step
```

The vocabulary disjointness is load-bearing: it rules out the
single-step shortcut the linear bridge in `coldFreeze` would
otherwise find. A diagnostic confirms it:

```
shortcut check: coldFreeze.execute('hot') = 'frost'  (must NOT be 'freeze')
```

`coldFreeze` applied directly to `"hot"` lazy-initialises a random
`hot` embedding inside its own table; the bridge transforms it; the
lookup returns `frost`. The only path that produces `"freeze"` from
`"hot"` is the composed chain.

### Outer substrate

```
TransformationGraph (cached-network substrate):
  output       --> coldFreeze
  output       --> hotCold
  coldFreeze   --> output
  coldFreeze   --> hotCold
  hotCold      --> output
  hotCold      --> coldFreeze
```

Three nodes, two of which are cached networks. The
`TransformationGraphBuilder` wires every type-compatible edge.

### Outer mandate and carving

```
outer mandate: root='hot', result='freeze'

carved outer pipeline:
  c2_hotCold    [net_hot->cold]   slot0<-ROOT
  c1_coldFreeze [net_cold->freeze] slot0<-c2_hotCold
  c0_output     [string-output]    slot0<-c1_coldFreeze

executed pipeline -> terminal value: 'freeze'
outer mandate verification:        PASS
```

The carver backward-chained:

```
terminal needs "freeze"
  └── coldFreeze produces "freeze" from "cold"
        └── hotCold produces "cold" from "hot"
              └── root matches "hot"
```

Each `inferInputs` call for a cached network forward-evaluated it
against every available anchor and matched against the target. The
forward-anchor pre-pass (introduced in phase 5) populated each cached
network's output anchor based on root input — `hotCold` anchors to
`"cold"`, `coldFreeze` anchors to `"frost"` (its random output for
`"hot"`). When the carver searched for `"freeze"`, it found that
`coldFreeze` applied to `"cold"` (the anchor from `hotCold`) produced
the target. Two-step pipeline, automatic.

## What this licenses

**The cache stores behaviours.** A `CachedItem` is no longer
restricted to a stored vector — `NetworkItem` is alongside
`EmbeddingItem` under the same sealed interface. The same lookup
machinery (cache primitive ↔ retrieval primitive) generalises across
both: an embedding-style cache returns a vector via `LookupSymbol`;
a network-style cache returns a transformation via
`CachedNetworkPrimitive`. The interface is the only thing the rest of
the framework needs to know.

**The carver composes cached networks without architectural change.**
The existing backward-chaining machinery, the existing forward-anchor
pre-pass, the existing mandate verifier — all of them handle cached
networks the same way they handle any other deterministic primitive.
The inversion case in `inferInputs` is the entire framework support
for "values are networks." Twenty lines.

**Composition emerges from the carver's normal job.** The demo's
two-step chain wasn't programmed; it was *found*. The carver
searched for any path that produces `"freeze"` from `"hot"` and found
that only the composed chain works. If we'd added a third cached
network for `"freeze" → "winter"` and changed the mandate to
`"winter"`, the carver would have built a three-step chain through
all three. The substrate's behaviour is determined by what's in it,
not by what was wired in.

**Key-Network is unified with the rest of the framework.** Mandates
still specify what must hold. Carving still searches for a structure
that holds it. Composition still produces a `ComputationGraph` that
runs forward through typed slots. The only thing that changed is
what can live inside a primitive's box. The "three verbs over one
substrate" claim holds even when the substrate's items are themselves
networks.

## Honest observations and limitations

- **Inner networks are pre-trained and frozen.** The demo runs each
  inner training to completion, then treats the result as immutable
  for the rest of the run. A future variant would let inner networks
  continue training as the outer composition is exercised — which
  would close the loop on "stored networks improve from use."

- **Cached network inversion is forward-only.** `inferInputs` tries
  each available anchor against the cached primitive and checks
  whether the output matches the target. This works for the chained-
  composition case but doesn't help when the carver needs to *create*
  a new input value that's not already an anchor. For multi-input
  cached networks, or for cases where the desired input is itself a
  computed value, the inversion would need to be smarter.

- **No explicit "lookup by key" yet.** A `NetworkItem` is found by
  the carver via structural fit to the mandate, not by name. A more
  literal Key-Network design would have a `NetworkCache` primitive
  that, given a key (e.g. a symbol), returns the corresponding
  `NetworkItem`. This is a small extension and natural to build once
  there's a reason to (e.g. when the outer carving's mandate
  references the key by name).

- **No autograd across cached-network boundaries.** The inner
  networks have their own trainables, but no gradient flows from the
  outer carving back through them. This is the same deferred
  "autograd-on-the-carving" question — and now it has a sharper
  motivation: training a chain of cached networks end-to-end requires
  gradients to cross the `CachedNetworkPrimitive` boundary.

- **The two cached networks share neither a TransformationGraph nor
  an embedding table.** This is intentional (the shortcut-blocking
  rationale), but it does mean the "cache" in this phase is really
  two independent caches. A unified `NetworkCache` storing many
  `NetworkItem`s by key is the obvious refactor and would carry the
  Key-Network framing more literally.

## Where the framework goes from here

Phase 6 has landed the Key-Network shape. What's next:

1. **Autograd across cached-network boundaries.** The deferred step,
   now with concrete motivation. `Primitive.backward(outputGrad) →
   inputGrad` would let the outer carving's verification failures
   train inner networks even after they're cached.

2. **`NetworkCache` and explicit key lookup.** A first-class cache
   type that stores many `NetworkItem`s and exposes them via a
   `LookupNetwork(key) → NetworkItem` primitive. The carver could
   then choose *which* network to retrieve from a much larger
   substrate — analogous to how `LookupSymbol` chooses *which*
   vector to return.

3. **Inner-network learning during outer carving execution.** The
   inline trainer already runs over whatever trainables the carver
   placed. If cached networks expose their inner trainables to the
   outer trainer, the same machinery would tune them based on the
   outer mandate. This is the most direct way to get "the substrate
   learns from being used."

4. **Phase 5b emergence.** Independently of Key-Network: run the
   multi-head demo's substrate over many sessions with edge-stats
   feedback, no hand-bias, and watch heads specialize. The
   infrastructure is already in place.

## Bottom line

The framework has reached the shape it has been pointing at. The
three verbs — `Mandate`, `Carve`, `Compose` — operate on a substrate
whose stored items can be networks, and the carver composes those
networks the same way it composes any other primitive. A mandate
naming `"freeze"` over a substrate containing `hot→cold` and
`cold→freeze` produces a two-step composition, with no special
machinery beyond `inferInputs` for the cached-network case.

The original framework claims survive intact: ante-hoc
interpretability (the composed pipeline is structurally
inspectable), specification-driven design (the mandate specified
"freeze" and the substrate did the rest), and reusable trained
priors (the two inner networks were each pre-trained and reused as
substrate items). The framework didn't need a new abstraction layer
to host Key-Network. It just needed one new `CachedItem` variant and
one new `Primitive` that wraps it.

The verbs compose. The substrate is uniform. Mandates still mean
what they have always meant. The framework's most expressive form is
also its simplest.

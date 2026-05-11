# v3 Phase 9 — The Grand Finale: A Substrate That Builds Itself

## Context

The framework has been climbing a ladder of increasing autonomy:

- **v0** — substrate fixed; carver finds compositions; verifier confirms.
- **v3 P4** — carver assembles end-to-end; no hand-wired graphs.
- **v3 P6** — Key-Network: cached subgraphs as substrate items.
- **v3 P7** — NetworkCache spawns its own items from training data.
- **v3 P8** — carver composes from the cache's inventory.

Each phase pushed the framework one step further toward "the user
specifies, the framework figures everything else out." Phase 9 closes
the loop: **mandates drive the cache's growth.** A mandate the
substrate can't yet satisfy triggers the smallest extension that would
make it satisfiable. The cache learns its inventory by being asked.

## What was built

| Deliverable | What it does |
|-------------|--------------|
| `AdaptiveCarverDemo` | The resolver loop. For each mandate: try to carve from the cache's current inventory. If carve fails or verification fails, BFS the cache from the mandate's input to find the deepest reachable atom (the "frontier"), spawn a single new network from frontier to mandate's output, then re-carve. ~250 lines. |

No framework changes outside the new demo. The resolver loop is built
entirely on top of `NetworkCache`, the carver, and the inline trainer
from earlier phases.

## How `resolve(input, output)` works

```
resolve(cache, input, output):
  carving = tryCarve(cache, input, output)
  if carving != null and verify(carving) passes:
    return REUSE                                # cache already covers this

  reachable = cache.bfsFrom(input)             # which atoms can we reach?
  frontier  = reachable.frontier               # deepest atom on the BFS path

  if frontier == input:
    spawn (input, output)                       # cache had no outgoing edge from input
  else if output in reachable:
    spawn (input, output)                       # output reachable but carve failed; close gap directly
  else:
    spawn (frontier, output)                    # extend the existing chain by one bridge

  carving = tryCarve(cache, input, output)
  assert carving != null and verify(carving)
  return SPAWN
```

The BFS treats each `NetworkItem` as a directed edge from its trained
input atom to its trained output atom. The frontier is the deepest
atom reachable from the mandate's input via these edges.

The spawn is **minimal**: at most one new network per failed mandate.
Existing chains are preferred over fresh ones. Mandate ordering
matters — early mandates seed the cache; later ones reuse or extend
what's there.

## The mandate sequence

Seven mandates, designed to exercise the loop's behaviour:

```
1. hot  -> cold       spawn from empty
2. hot  -> freeze     extend chain (spawn cold -> freeze)
3. hot  -> ice        extend chain (spawn freeze -> ice)
4. hot  -> freeze     REUSE existing chain (mandate 1+2 already cover it)
5. warm -> cool       start parallel chain (spawn warm -> cool)
6. warm -> ice        extend parallel chain (spawn cool -> ice)
7. cold -> ice        REUSE existing (cold -> freeze -> ice already in cache)
```

## What the demo produces

```
mandate 1: 'hot' -> 'cold'
  SPAWN: cache reachable from 'hot' = [hot]; spawning 'hot' -> 'cold'
  PASS after spawn
  chain: net_hot_to_cold

mandate 2: 'hot' -> 'freeze'
  SPAWN: cache reachable from 'hot' = [hot, cold]; spawning 'cold' -> 'freeze'
  PASS after spawn
  chain: net_hot_to_cold -> net_cold_to_freeze

mandate 3: 'hot' -> 'ice'
  SPAWN: cache reachable from 'hot' = [hot, cold, freeze]; spawning 'freeze' -> 'ice'
  PASS after spawn
  chain: net_hot_to_cold -> net_cold_to_freeze -> net_freeze_to_ice

mandate 4: 'hot' -> 'freeze'
  REUSE: cache already covers this mandate
  chain: net_hot_to_cold -> net_cold_to_freeze

mandate 5: 'warm' -> 'cool'
  SPAWN: cache reachable from 'warm' = [warm]; spawning 'warm' -> 'cool'
  PASS after spawn
  chain: net_warm_to_cool

mandate 6: 'warm' -> 'ice'
  SPAWN: cache reachable from 'warm' = [warm, cool]; spawning 'cool' -> 'ice'
  PASS after spawn
  chain: net_warm_to_cool -> net_cool_to_ice

mandate 7: 'cold' -> 'ice'
  REUSE: cache already covers this mandate
  chain: net_cold_to_freeze -> net_freeze_to_ice
```

```
Summary
mandates submitted: 7
spawns triggered:   5
carves reused:      2
final cache size:   5 networks

final cache inventory:
  hot    -> cold     (success=2)
  cold   -> freeze   (success=2)
  freeze -> ice      (success=1)
  warm   -> cool     (success=1)
  cool   -> ice      (success=1)
```

## What this demonstrates

**A minimal spanning set, learned from data.** The 7 mandates needed
only 5 networks. The cache evolved exactly the bridges required to
cover every (input, output) pair seen, and no more. Mandate 7
(`cold → ice`) was satisfied by a chain assembled entirely from
bridges that prior mandates had triggered — the cache hadn't been
"told" about `cold → ice` directly, but the composition was already
in its inventory.

**The substrate's shape is a record of what was asked.** Mandate
ordering matters. If `hot → ice` had come first, the cache would
have spawned `hot → ice` as a single bridge. Following with
`hot → cold` would have spawned that too, and `cold → ice` would
have triggered a third spawn. The same 3 mandates in a different
order, same 3 spawns — but the inventory is different. Adaptation
isn't arbitrary; it's shaped by the order of demand.

**The same machinery handles everything.** Carver, inline trainer,
verifier — all unchanged from earlier phases. The resolver is a
~50-line loop on top of `cache.bfsFrom`, `cache.getOrTrain`,
`carver.carve`, and `MandateVerifier.verify`. Phase 9 isn't an
architectural addition; it's a policy expressed in framework
primitives.

**Mandate, Carve, Compose — over a substrate that didn't exist when
the demo started.** This is the framework's most autonomous form.
The user supplies a sequence of `(input, output)` pairs. The
framework returns the carving that satisfies each one, growing the
substrate when necessary. Specification-driven design taken to its
limit: the spec is the *only* thing the user writes.

## Honest observations and limitations

- **Shortest-extension is a heuristic, not optimal.** When two
  candidate spawns would equally close the gap, the BFS picks the
  frontier-extension. This minimizes the *current* spawn but may
  cause more spawns over the long run if a different choice would
  have been more reusable. A future variant could optimize over a
  predicted mandate distribution rather than greedily.

- **The cache grows monotonically.** Once a network is spawned, it
  stays unless evicted by the bounded-cache policy (Phase 7). There's
  no garbage collection for networks that turn out to be dead-ends.
  In a long-running adaptive setting, periodic pruning by success
  count would help.

- **Mandate-ordering sensitivity is real.** The demo's mandate order
  was chosen to exercise reuse cleanly. A pathological order can
  cause the cache to spawn far more than the minimum — e.g., if
  every mandate has a unique input atom, every mandate spawns its
  own chain.

- **The resolver is built on top of `NetworkCache`, not inside it.**
  The "framework primitive" for this loop doesn't yet exist as a
  named API. The natural refactor would be a `CacheBackedCarver` or
  `AdaptiveResolver` class that encapsulates the loop. Out of scope
  for this milestone — the demo is the proof.

- **No autograd across cached-network boundaries.** Each cached
  network's training was done at spawn-time and is frozen
  afterward. The outer mandate's verification doesn't propagate back
  through the cached primitives to refine them. Still the standing
  deferred step.

- **One trainable per gradient path inside each cached network.** The
  inner training is the simple bridge optimization from Phase 4.
  Multi-trainable end-to-end remains future work.

## Where the framework goes from here

The framework's core arc — Mandate, Carve, Compose — is now executable
end-to-end against a substrate that builds itself. The remaining work
is all extensions and refinements rather than load-bearing claims:

1. **Autograd across cached-network boundaries.** Gradient flow
   through `CachedNetworkPrimitive` into inner trainables. Would let
   the outer mandate's verification refine inner networks even after
   they're cached.

2. **A named `AdaptiveResolver` API.** The resolver loop in this
   demo deserves to be a first-class framework primitive.

3. **Learned similarity (Q-W-K^T).** Phases 1–9 use fixed cosine.

4. **Larger, noisier substrates.** All demos use small (~10 atom)
   vocabularies. Real-world scale (1k+ symbols, longer chains)
   would surface different bottlenecks.

5. **Phase 5b emergence.** Multi-head substrate, no hand-bias, edge
   stats specialize heads from random init alone.


## Bottom line

The framework's verbs compose over a substrate that didn't exist when
the user started. Each mandate either uses the substrate as-is, or
triggers the smallest extension that would make it usable. The cache's
inventory at any moment is a minimal record of what's been asked, and
the carver makes that record queryable. The user writes mandates; the
framework returns carvings.

This is what the original framework claims have been pointing at all
along: specification-driven design (mandates are the only spec), ante-
hoc interpretability (every cached network is structurally
inspectable), reusable trained priors (every network in the cache is
reused across mandates that need it), and a specialization dial (the
cache's inventory specializes to its mandate stream).

Nine phases. One substrate. Three verbs. The substrate builds itself.

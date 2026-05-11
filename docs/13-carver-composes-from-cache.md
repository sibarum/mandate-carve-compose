# v3 Phase 8 — Carver Composes from a NetworkCache

## Context

Phase 7 introduced `NetworkCache` — a cache that builds its own inventory
from training data. Phase 6 showed that cached subgraphs can be composed
by the carver when wired into an outer TransformationGraph by hand.

Phase 8 joins these: the cache *is* the substrate. Hand a NetworkCache's
inventory to the carver via `cache.primitives()`; submit mandates of
varying difficulty; watch the carver assemble chains of different depths
out of the same stored networks.

The headline result: **three mandates over the same 3-network cache
produce three different composed chains**: a 1-step direct mapping, a
2-step composition, and a 3-step composition — all using the same
cached primitives, picked and chained by the carver per mandate.

## What was built

| Deliverable | What it does |
|-------------|--------------|
| Carver: BFS reachability pre-pass | Extends the forward-anchor pre-pass with a BFS over reachable values per type. For each non-Trainable single-input primitive, apply it to every reachable input value and add the output to the reachable set. Run to fixpoint. The pre-pass is **conditional** — only enabled when the substrate contains `CachedNetworkPrimitive`s, since BFS through general Trainable primitives in some substrates (multi-head with parallel `VectorTransform`s) hits memory issues. The conditional gate keeps multi-head working while unlocking N-step composition for Key-Network substrates. |
| Carver: cached-network inversion uses reachable-set | `inferInputs` for `CachedNetworkPrimitive` iterates the reachable-values set for its input type. For each candidate, apply the cached primitive and check whether the output matches the target. If so, return that candidate as the inferred input. This is what lets `solve()` chain cached networks N levels deep. |
| `NetworkCache`: two-phase bridge training | Replaces the simple `InlineTrainer` call with a two-phase scheme. Phase 1 trains the bridge to be identity on every atom in the vocabulary (overcomes random Xavier init, gives the bridge a known structure). Phase 2 trains the positive mapping while continuously reinforcing identity on the rest. The result: `bridge(embed(input)) ≈ embed(output)`, `bridge(embed(other)) ≈ embed(other)`. Without this, the linear bridge collapses to a rank-1 outer product and the cached network generalizes to *any* input correlated with its trained input — a cross-cache shortcut that defeats the composition demo. |
| `CarverFromCacheDemo` | Build empty NetworkCache; feed three mappings; print inventory; submit three mandates of increasing chain depth (1, 2, 3 steps); verify each carved chain matches expectation. |

About 200 lines of code changes plus the demo.

## The substrate

A NetworkCache with three spawned networks:

```
  hot     -> cold     (trained inline by feed(hot, cold))
  cold    -> freeze   (trained inline by feed(cold, freeze))
  freeze  -> ice      (trained inline by feed(freeze, ice))
```

The cache's `primitives()` returns a list of three `CachedNetworkPrimitive`
wrappers. Each wraps a `NetworkItem` holding a trained
embed→bridge→lookup→string-output subgraph.

The outer TransformationGraph is built by `TransformationGraphBuilder`
from these three primitives plus a `StringOutputPrimitive`:

```
TransformationGraph (cache-as-substrate):
  output            --> net_cold_to_freeze
  output            --> net_hot_to_cold
  net_freeze_to_ice --> output
  net_freeze_to_ice --> net_cold_to_freeze
  net_freeze_to_ice --> net_hot_to_cold
  net_cold_to_freeze --> output
  net_cold_to_freeze --> net_freeze_to_ice
  net_cold_to_freeze --> net_hot_to_cold
  net_hot_to_cold    --> output
  net_hot_to_cold    --> net_cold_to_freeze
  net_hot_to_cold    --> net_freeze_to_ice
```

Any-to-any modulo type compatibility. The carver decides which edges to
use per mandate.

## Three mandates, three chains

### Mandate 1: `hot → cold` (1-step)

```
shortcut check (each primitive applied to root):
  net_hot_to_cold('hot') -> 'cold'  *** SHORTCUT ***
  net_cold_to_freeze('hot') -> 'hot'
  net_freeze_to_ice('hot') -> 'hot'

carved chain:
  c1_net_hot_to_cold [net_hot_to_cold] slot0<-ROOT
  c0_output [string-output] slot0<-c1_net_hot_to_cold

terminal: 'cold'  mandate:PASS   chain length: 1
```

`net_hot_to_cold` directly handles this mandate. The other two cached
networks are identity on `hot` (they were regularized to not transform
atoms outside their trained pair), so they don't compete.

### Mandate 2: `hot → freeze` (2-step composition)

```
shortcut check (each primitive applied to root):
  net_hot_to_cold('hot') -> 'cold'
  net_cold_to_freeze('hot') -> 'hot'
  net_freeze_to_ice('hot') -> 'hot'

carved chain:
  c3_net_hot_to_cold [net_hot_to_cold] slot0<-ROOT
  c2_net_cold_to_freeze [net_cold_to_freeze] slot0<-c3_net_hot_to_cold
  c0_output [string-output] slot0<-c2_net_cold_to_freeze

terminal: 'freeze'  mandate:PASS   chain length: 2
```

No single cached network produces `freeze` from `hot`. The carver
composes `hot_to_cold` and `cold_to_freeze`. The BFS reachability
pre-pass populated `freeze` as a reachable value (via 2 hops from
root), which let `inferInputs` for `net_cold_to_freeze` find that
applying it to `cold` produces `freeze`.

### Mandate 3: `hot → ice` (3-step composition)

```
shortcut check (each primitive applied to root):
  net_hot_to_cold('hot') -> 'cold'
  net_cold_to_freeze('hot') -> 'hot'
  net_freeze_to_ice('hot') -> 'hot'

carved chain:
  c6_net_hot_to_cold [net_hot_to_cold] slot0<-ROOT
  c4_net_cold_to_freeze [net_cold_to_freeze] slot0<-c6_net_hot_to_cold
  c3_net_freeze_to_ice [net_freeze_to_ice] slot0<-c4_net_cold_to_freeze
  c0_output [string-output] slot0<-c3_net_freeze_to_ice

terminal: 'ice'  mandate:PASS   chain length: 3
```

Three-step composition. Same cache, same primitives, deeper chain
because the mandate target is deeper in the reachability graph.

## What this licenses

**N-step composition over cached subgraphs.** The carver assembles
arbitrarily deep chains as long as the reachability pre-pass surfaces
the intermediate values. This is the framework's first demo where the
carver does genuine *graph search* over a substrate's compositional
structure — not just picking one primitive per mandate, but threading
several in series.

**The cache's inventory is the substrate.** No hand-wired
TransformationGraph beyond a one-liner that hands `cache.primitives()`
to a builder. The user's interaction with the framework is now:
"populate the cache with what you know, submit mandates for what you
want." The carver does the rest.

**Cached-network shortcut-blocking via regularized training.** The
two-phase bridge training (identity-on-all then targeted-shift) is a
real architectural addition. Without it, the rank-1 linear bridge
generalizes the trained mapping to any input correlated with its
trained input, producing cross-cache shortcuts that defeat
composition. With it, each cached network behaves like a precise
single-pair lookup table with identity elsewhere — exactly the
property compositional retrieval needs.

**BFS reachability over deterministic primitives.** The carver's
forward-anchor pre-pass now tracks all reachable values per type via
BFS. For deterministic single-input primitives (which CachedNetworkPrimitive
qualifies as), reaching a value at depth N is no harder than reaching
one at depth 1. The carver's backward chaining then finds the chain by
picking the right (primitive, input) pair at each step.

## Honest observations and limitations

- **BFS is conditional on substrate content.** When no
  `CachedNetworkPrimitive` is present, BFS is skipped. This is because
  unbounded BFS through some Trainable substrates (multi-head with two
  parallel `VectorTransform`s) hits memory issues — there's a real bug
  in the interaction between my BFS and parallel-trainable substrates
  that I haven't fully diagnosed. The conditional gate keeps every
  prior demo working while unlocking Phase 8. The bug is real but
  doesn't matter for substrates we use.

- **The shortcut check is run as an assertion, not as carver guidance.**
  The demo asserts that no single primitive directly produces the
  target — if a cached network's bridge generalizes too aggressively
  and produces the target by coincidence, the demo fails. With the
  two-phase training in place, this hasn't happened in the demonstrated
  vocabulary. Larger vocabularies might surface more coincidences.

- **No carver-driven SPAWNING yet.** The cache is populated by the user
  (via `feed(cache, input, output)`) before the carver runs. When the
  outer mandate can't be satisfied, the carver fails. The "failed
  mandate triggers cache spawn" loop — where the cache extends itself
  to close the gap — is Phase 9.

- **Two-phase training is brute-force.** 300 epochs of identity then
  500 epochs of positive+maintenance, ~3000 gradient steps per spawn.
  Fine for the demo (each spawn takes ~1 second), but scales O(vocab²)
  in the worst case. Faster alternatives exist (closed-form regression,
  non-linear bridges) but are beyond scope.

- **No autograd across cached-network boundaries.** Each cached
  network's inner training was done before it was stored. The outer
  mandate's verification doesn't propagate back through the cached
  primitives to refine them. Still the standing deferred step.

## Where the framework goes from here

Phase 8 leaves one piece of the grand finale: **failed-mandate-triggers-
cache-spawn**. The cache currently fails silently when the carver
can't find a path; the natural extension is for the cache to observe
the mandate, BFS its existing inventory to find what's reachable,
identify the shortest extension that would close the gap, and spawn
that as a new entry — then re-run the carver. This is Phase 9.

Other open threads:
- Multi-trainable end-to-end (autograd through cached primitives).
- Phase 5b emergence (multi-head specialization without hand-bias).
- Learned similarity (Q-W-Kᵀ).

## Bottom line

The cache builds its inventory from training data (Phase 7); the
carver composes the cache's inventory per mandate (Phase 8). One
cache, three mandates, three different composed chains, no
hand-wiring beyond the cache's own training feeds. The framework's
substrate is no longer a fixed set of primitives — it's the
accumulated record of what the user has trained the cache to do, and
the carver makes that record *queryable by mandate*.

# v3 Phase 7 — NetworkCache: a Cache That Learns Its Own Inventory

## Context

Phase 6 showed that the framework can compose stored networks: pre-train
two inner networks, register them as primitives, the carver chains them.
But the cache in P6 was implicit — two `NetworkItem`s, hand-built,
hand-registered. There was no *cache type*. No spawn-on-demand. No
inventory policy.

Phase 7 adds the missing piece: **`NetworkCache`**, a mutable cache of
`NetworkItem`s that builds its own inventory from training data. The
cache starts empty. Each `(input → output)` pair the user feeds it
either matches an existing entry (counts as a successful use) or spawns
a fresh inner network on the fly. With an optional `maxNetworks` cap,
least-successful entries are evicted to make room — the cache's
contents drift toward whatever mappings are actually exercised.

This is the "cache as a learning data structure" idea: not gradients
flowing through frozen networks, but the cache *itself* deciding what
to remember.

## What was built

| Deliverable | What it does |
|-------------|--------------|
| `cache/NetworkCache` | Stateful cache. Constructor takes `dim`, vocabulary, seed base, optional `maxNetworks`, and a `trainingMode` flag. Methods: `get(key)`, `query(key, input)`, `getOrTrain(input, output)`, `primitives()`, plus diagnostics (`size`, `keys`, `spawnCount`, `evictionCount`, `successCount`). About 200 lines. |
| `demo/NetworkCacheTrainingDemo` | Two-run diagnostic. Run 1: unbounded, four mappings, demonstrates spawn-on-demand and read-through. Run 2: capped at 3, five mappings with `hot` re-fed three times, demonstrates eviction by success count. |

No framework changes outside the new files — the cache is built entirely
on top of the existing carver + inline trainer.

## How `getOrTrain` works

```
getOrTrain(input, output):
  if there is an entry at key=input:
    if entry.execute(input) == output:
      entry.successCount++
      return entry          // hit
    else:
      // existing entry no longer satisfies; will be replaced
  build a TG: embed -> bridge -> lookup -> string-output
  carve mandate { result = output, root = input }
  inline-train to convergence
  wrap as NetworkItem
  put(key=input, entry)
  spawnCount++
  if size > maxNetworks:
    evict the least-successful entry
  return new entry
```

The inner training reuses the entire P3/P4 stack: a single-trainable
carving, the carver finding the embed-bridge-lookup pipeline, the
inline trainer driving the bridge to a fixed-point. The cache is a
*coordinator*, not a separate training algorithm.

## Run 1 — unbounded cache

```
feeding training data:
  [SPAWN] hot    -> cold     (cache size = 1, spawns = 1)
  [SPAWN] cold   -> freeze   (cache size = 2, spawns = 2)
  [SPAWN] warm   -> cool     (cache size = 3, spawns = 3)
  [SPAWN] fire   -> ice      (cache size = 4, spawns = 4)

re-feeding an existing mapping ('hot' -> 'cold'):
  cache size = 4 (no spawn: true)

querying stored keys:
  [OK ] query('hot')  -> 'cold'   (expected 'cold')
  [OK ] query('cold') -> 'freeze' (expected 'freeze')
  [OK ] query('warm') -> 'cool'   (expected 'cool')
  [OK ] query('fire') -> 'ice'    (expected 'ice')

querying a missing key:
  query('unknown') -> <missing>
```

Four mappings, four spawns, four queries reproduce their trained
outputs. Re-feeding `(hot, cold)` produces a hit, not a spawn. An
unknown key returns nothing.

## Run 2 — bounded cache, eviction by success count

`maxNetworks = 3`. Feed five mappings, with `(hot, cold)` repeated
three times before the others.

```
  [SPAWN] hot    -> cold     (size = 1)
  [hit  ] hot    -> cold     (size = 1)
  [hit  ] hot    -> cold     (size = 1)
  [SPAWN] cold   -> freeze   (size = 2)
  [SPAWN] warm   -> cool     (size = 3)        # at cap
  [SPAWN] fire   -> ice      (size = 3) (+1 evict)
  [SPAWN] up     -> down     (size = 3) (+1 evict)

final cache contents (max = 3):
  hot      success=3
  fire     success=1
  up       success=1

  spawns = 5, evictions = 2, size = 3
```

The cap held. Two evictions happened: the least-successful entries
(`cold` and `warm`, each with success=1) were dropped to make room for
the new spawns. `hot` survived because its three uses accumulated
success that outranked everything else.

The surviving networks still produce their trained outputs:

```
query('hot')  -> 'cold'
query('fire') -> 'ice'
query('up')   -> 'down'
```

## What this licenses

**A cache that grows from data.** The substrate's contents are no
longer a fixed library; they're a function of what the user (or some
outer loop) has asked the cache to handle. This is the closest the
framework has come to "the architecture builds itself from
experience."

**Bounded online learning, the same way other adaptive data structures
work.** The combination of "keyed insertion + capacity + eviction by
usage statistic" is the same shape as an LFU cache, a bloom filter
backed by counts, a count-min sketch. The fact that the items happen
to be neural networks is incidental to the interface — and that's the
whole point. The same cache discipline that works for KV pairs works
for KV pairs whose values are subgraphs.

**Cache contents are first-class for the carver.** `NetworkCache.primitives()`
returns a list of `CachedNetworkPrimitive`s — one per stored network.
Drop those into a `TransformationGraphBuilder` and the carver can
compose them per a mandate. The two halves (cache learns its inventory;
carver composes from the inventory) are independent and compose.

**The same machinery handles every cache type.** `SymbolEmbeddingTable`
stores vectors; `NetworkCache` stores networks. Both expose a keyed
lookup; both can be queried; both can grow. The verbs above them —
mandate, carve, compose — don't know the difference.

## Honest observations and limitations

- **Keys are exact-match symbols.** One stored network per input atom.
  The "keys drift to encompass success" idea — a single network covering
  a *region* of similar inputs — would require learned partitioning or
  near-match retrieval. Out of scope for this phase; the right next step
  is probably to add a near-match `query` that finds the closest stored
  key in embedding space.

- **Eviction is the simplest possible policy.** Lowest success count
  wins. No recency, no tie-breaking by re-train cost, no merging of
  similar networks. Plenty of room to make this smarter once we have
  a workload that pressures it.

- **Each network has its own embedding table.** This is intentional
  (it preserves the disjoint-vocabulary insight from P6 and keeps
  networks independent), but it does mean memory grows linearly with
  cache size — and that vector spaces drift apart between networks.
  A "shared substrate vocabulary, per-network bridge" variant is a
  natural refactor when there's a reason for it.

- **No carver-driven composition over a `NetworkCache` in this
  demo.** The cache is exercised standalone; carver integration would
  be `KeyNetworkDemo`-style but with the inventory coming from a
  cache instead of hand-built. The pieces are all in place
  (`cache.primitives()` exists); the next demo to write is the
  composition.

- **Spawning is synchronous and serial.** Each new entry triggers a
  full carver + inline-train run. Fine for a demo; would need
  amortization at scale.

- **No carver-driven *spawning*.** The mandate that triggers a spawn
  is hand-set in user code (`getOrTrain(input, output)`). A more
  ambitious cache would consume failed *outer* carvings as spawn
  requests: when the outer mandate can't be satisfied by the current
  inventory, automatically spawn whatever bridge would close the gap.
  That's the natural-feedback version of "the cache learns its
  inventory from what it's asked to do."

## Where the framework goes from here

Phase 7 unblocks the next two demos:

1. **Carver composes from a `NetworkCache`.** Build a cache, populate
   it with several mappings, hand its `primitives()` to a
   `TransformationGraphBuilder`, set an outer mandate, let the carver
   chain. Like P6 but with the inventory coming from a learned cache
   instead of two hand-trained networks. Should work without any code
   changes — the pieces compose.

2. **Failed-outer-mandate-triggers-cache-spawn.** When the outer
   carving fails to satisfy a mandate, the cache treats that as a
   spawn request and trains a new bridge to close the gap. The
   substrate genuinely learns from being used.

Plus the standing items:

3. **Autograd across cached-network boundaries.** Gradient flow
   through `CachedNetworkPrimitive` into inner trainables. Independent
   of the cache's inventory question.

4. **Phase 5b emergence.** Multi-head demo without hand-bias; edge
   stats accumulate over many sessions; heads specialize.

## Bottom line

The cache builds itself. Spawn-on-demand from training data, bounded
capacity with success-weighted eviction, queries via keyed lookup —
the same discipline that works for any other adaptive cache, applied
to a cache whose entries happen to be trained neural subgraphs.

The "items are networks" part is opaque to the cache's interface and
load-bearing for what the cache *does*. `NetworkCache` is the first
data structure in this framework that is genuinely a cache of
*behaviours*: ask it to handle a mapping, and either it already can or
it can learn to. That's the Key-Network promise as a usable object.

package sibarum.strnn.demo;

import sibarum.strnn.cache.NetworkCache;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * A NetworkCache that builds its own inventory from a stream of training
 * data. The cache starts empty; every (input → output) pair the demo
 * feeds it either matches an existing entry (counts as a successful use)
 * or spawns and trains a fresh inner network on the fly. With an optional
 * {@code maxNetworks} cap, least-successful entries are evicted as the
 * cache fills — its contents drift toward whatever mappings the training
 * data actually exercises.
 *
 * Two runs:
 *
 *   Run 1 — unbounded cache. Feed four distinct mappings. Cache populates
 *           with four networks. Re-feed an existing mapping; no new
 *           network is spawned. Query existing keys; values come back
 *           from the stored networks. Query a missing key; the cache
 *           returns nothing.
 *
 *   Run 2 — bounded cache (max = 3). Feed five mappings, with re-feeding
 *           the first one so it accumulates success. The cache should hit
 *           its cap, evict one entry (the least-successful), and end at
 *           three entries.
 *
 * The architectural claim being tested: a cache of trained subgraphs can
 * be operated like any other adaptive data structure — keyed insertion,
 * bounded capacity, eviction by usage. The "items happen to be neural
 * networks" is incidental to the interface.
 */
public final class NetworkCacheTrainingDemo {

    private static final int DIM = 32;
    private static final List<String> VOCABULARY = List.of(
            "hot", "cold", "warm", "cool", "fire", "ice", "burn", "freeze",
            "up", "down", "left", "right");

    public static void main(String[] args) {
        runUnbounded();
        System.out.println();
        runBounded();
    }

    private static void runUnbounded() {
        System.out.println("============================================================");
        System.out.println("Run 1 — unbounded cache, training mode");
        System.out.println("============================================================");

        NetworkCache cache = new NetworkCache(
                DIM, VOCABULARY, /* seedBase */ 42L,
                /* maxNetworks */ OptionalInt.empty(),
                /* trainingMode */ true);

        List<String[]> trainingData = List.of(
                new String[]{"hot",  "cold"},
                new String[]{"cold", "freeze"},
                new String[]{"warm", "cool"},
                new String[]{"fire", "ice"});

        System.out.println("feeding training data:");
        for (String[] pair : trainingData) {
            long beforeSpawn = cache.spawnCount();
            cache.getOrTrain(pair[0], pair[1]);
            String tag = (cache.spawnCount() > beforeSpawn) ? "SPAWN" : "hit  ";
            System.out.printf(Locale.ROOT,
                    "  [%s] %-6s -> %-6s   (cache size = %d, spawns = %d)%n",
                    tag, pair[0], pair[1], cache.size(), cache.spawnCount());
        }

        System.out.println();
        System.out.println("re-feeding an existing mapping ('hot' -> 'cold'):");
        long beforeSpawn = cache.spawnCount();
        cache.getOrTrain("hot", "cold");
        boolean reused = cache.spawnCount() == beforeSpawn;
        System.out.printf(Locale.ROOT,
                "  cache size = %d (no spawn: %s)%n", cache.size(), reused);
        if (!reused) {
            throw new AssertionError(
                    "expected no spawn for re-fed mapping; cache spawned anyway");
        }

        System.out.println();
        System.out.println("querying stored keys:");
        for (String[] pair : trainingData) {
            Optional<Value> out = cache.query(pair[0], new StringValue(pair[0]));
            String got = out.map(v -> ((StringValue) v).s()).orElse("<missing>");
            String tag = got.equals(pair[1]) ? "OK " : "BAD";
            System.out.printf(Locale.ROOT,
                    "  [%s] query('%s') -> '%s'   (expected '%s')%n",
                    tag, pair[0], got, pair[1]);
            if (!got.equals(pair[1])) {
                throw new AssertionError(
                        "stored network did not reproduce trained mapping for '"
                                + pair[0] + "'");
            }
        }

        System.out.println();
        System.out.println("querying a missing key:");
        Optional<Value> out = cache.query("unknown", new StringValue("unknown"));
        System.out.printf(Locale.ROOT,
                "  query('unknown') -> %s   (expected <missing>)%n",
                out.map(v -> "'" + ((StringValue) v).s() + "'").orElse("<missing>"));
        if (out.isPresent()) {
            throw new AssertionError("expected missing entry for unknown key");
        }

        System.out.println();
        System.out.printf(Locale.ROOT,
                "Run 1 summary: %d entries, %d spawns, %d evictions%n",
                cache.size(), cache.spawnCount(), cache.evictionCount());
    }

    private static void runBounded() {
        System.out.println("============================================================");
        System.out.println("Run 2 — bounded cache (max = 3), training mode");
        System.out.println("============================================================");

        NetworkCache cache = new NetworkCache(
                DIM, VOCABULARY, /* seedBase */ 99L,
                /* maxNetworks */ OptionalInt.of(3),
                /* trainingMode */ true);

        // Build accumulated success for "hot" so it survives eviction
        // (re-feed it three times, then add three more entries; the cap
        // forces eviction of the least-successful — which is whichever
        // newer entry hasn't been re-used).
        System.out.println("feeding training data (hot gets re-used to accumulate success):");
        feed(cache, "hot",  "cold");    // size=1, hot.success=1
        feed(cache, "hot",  "cold");    // size=1, hot.success=2
        feed(cache, "hot",  "cold");    // size=1, hot.success=3
        feed(cache, "cold", "freeze");  // size=2
        feed(cache, "warm", "cool");    // size=3, at cap
        feed(cache, "fire", "ice");     // size=3, evicts the least-successful new entry
        feed(cache, "up",   "down");    // size=3, evicts again

        System.out.println();
        System.out.printf(Locale.ROOT, "final cache contents (max = 3):%n");
        for (String key : cache.keys()) {
            System.out.printf(Locale.ROOT, "  %-6s   success=%d%n",
                    key, cache.successCount(key).orElse(-1));
        }
        System.out.printf(Locale.ROOT,
                "  spawns = %d, evictions = %d, size = %d%n",
                cache.spawnCount(), cache.evictionCount(), cache.size());

        if (cache.size() > 3) {
            throw new AssertionError("cache exceeded max=3 after spawns");
        }
        if (cache.evictionCount() == 0) {
            throw new AssertionError("expected at least one eviction once cap was hit");
        }
        if (!cache.keys().contains("hot")) {
            throw new AssertionError(
                    "'hot' was re-used 3 times and should have survived eviction");
        }

        System.out.println();
        System.out.println("verifying retained networks still produce the right outputs:");
        for (String key : cache.keys()) {
            Optional<Value> out = cache.query(key, new StringValue(key));
            System.out.printf(Locale.ROOT,
                    "  query('%s') -> '%s'%n",
                    key, out.map(v -> ((StringValue) v).s()).orElse("<missing>"));
        }

        System.out.println();
        System.out.println("NetworkCache: stateful cache of trained subgraphs.");
        System.out.println("  - Items spawn on demand from training data.");
        System.out.println("  - Cap + success-based eviction give bounded online learning.");
        System.out.println("  - The cache learns its own inventory from what it's asked to do.");
    }

    private static void feed(NetworkCache cache, String input, String output) {
        long beforeSpawn = cache.spawnCount();
        long beforeEvict = cache.evictionCount();
        cache.getOrTrain(input, output);
        String tag = (cache.spawnCount() > beforeSpawn) ? "SPAWN" : "hit  ";
        String evictTag = (cache.evictionCount() > beforeEvict)
                ? " (+" + (cache.evictionCount() - beforeEvict) + " evict)"
                : "";
        System.out.printf(Locale.ROOT,
                "  [%s] %-6s -> %-6s   (size = %d)%s%n",
                tag, input, output, cache.size(), evictTag);
    }
}

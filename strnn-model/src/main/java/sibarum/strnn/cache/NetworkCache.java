package sibarum.strnn.cache;

import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.primitive.StringOutputPrimitive;
import sibarum.strnn.training.InlineTrainer;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Mutable cache of {@link NetworkItem}s, keyed by input symbol. In training
 * mode, the cache spawns and trains a fresh {@code NetworkItem} on demand
 * whenever a key arrives that doesn't already have one (or whose existing
 * network doesn't produce the requested output). With an optional
 * {@code maxNetworks} cap, least-successful entries are evicted to make
 * room — the cache's inventory drifts toward whatever mappings the
 * training data actually exercises.
 *
 * The current design key-by-input-symbol is the simplest case: one stored
 * network per input atom. A richer variant would key by some learned
 * partition of input space (so a single network covers a region of similar
 * inputs), which is the "keys drift to encompass success" idea taken
 * further. Out of scope for this phase.
 *
 * Query (read-only) and getOrTrain (mutating) are the two access paths.
 * Both succeed when an existing network handles the request; only the
 * second can change the cache's contents.
 */
public final class NetworkCache {
    private final int dim;
    private final List<String> vocabulary;
    private final long seedBase;
    private final OptionalInt maxNetworks;
    private final boolean trainingMode;

    private final Map<String, Entry> entries = new LinkedHashMap<>();
    private long spawnCounter = 0;
    private long evictionCounter = 0;

    public NetworkCache(
            int dim,
            List<String> vocabulary,
            long seedBase,
            OptionalInt maxNetworks,
            boolean trainingMode) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be positive");
        if (vocabulary.isEmpty()) throw new IllegalArgumentException("vocabulary must be non-empty");
        this.dim = dim;
        this.vocabulary = List.copyOf(vocabulary);
        this.seedBase = seedBase;
        this.maxNetworks = maxNetworks;
        this.trainingMode = trainingMode;
    }

    public int size() {
        return entries.size();
    }

    public Set<String> keys() {
        return Collections.unmodifiableSet(entries.keySet());
    }

    public boolean isTrainingMode() {
        return trainingMode;
    }

    public OptionalInt maxNetworks() {
        return maxNetworks;
    }

    public long spawnCount() {
        return spawnCounter;
    }

    public long evictionCount() {
        return evictionCounter;
    }

    /** Read-only: return the NetworkItem keyed by {@code key}, or empty. */
    public Optional<NetworkItem> get(String key) {
        Entry e = entries.get(key);
        return e == null ? Optional.empty() : Optional.of(e.item);
    }

    /** Per-entry success count — useful for diagnostics. */
    public Optional<Integer> successCount(String key) {
        Entry e = entries.get(key);
        return e == null ? Optional.empty() : Optional.of(e.successCount);
    }

    /**
     * Execute the network keyed by {@code key} against the given input, if
     * one exists. Returns empty if there is no such entry. Increments the
     * entry's success count when called (querying counts as use).
     */
    public Optional<Value> query(String key, Value input) {
        Entry e = entries.get(key);
        if (e == null) return Optional.empty();
        Value out = e.item.execute(input);
        e.successCount++;
        return Optional.of(out);
    }

    /**
     * Spawn or retrieve the network for {@code inputAtom → outputAtom}. If
     * an entry already exists at {@code inputAtom} and it currently maps
     * {@code inputAtom} to {@code outputAtom}, it is reused (success
     * incremented). Otherwise a fresh network is trained, the existing
     * entry (if any) is replaced, and the cache's size is enforced.
     *
     * Only usable when the cache was constructed with trainingMode = true.
     */
    public NetworkItem getOrTrain(String inputAtom, String outputAtom) {
        if (!trainingMode) {
            throw new IllegalStateException(
                    "getOrTrain called on a read-only cache");
        }
        Entry existing = entries.get(inputAtom);
        if (existing != null) {
            StringValue out = (StringValue) existing.item.execute(new StringValue(inputAtom));
            if (out.s().equals(outputAtom)) {
                existing.successCount++;
                return existing.item;
            }
            // Existing entry no longer satisfies; replace it.
        }
        NetworkItem item = trainNew(inputAtom, outputAtom);
        Entry e = new Entry(item, 1);
        entries.put(inputAtom, e);
        spawnCounter++;
        enforceMax();
        return item;
    }

    /**
     * Snapshot the current entries as a list of {@link CachedNetworkPrimitive}s,
     * one per stored network. Useful for handing the cache's contents to a
     * carver as a substrate.
     */
    public List<CachedNetworkPrimitive> primitives() {
        List<CachedNetworkPrimitive> out = new ArrayList<>(entries.size());
        for (Map.Entry<String, Entry> e : entries.entrySet()) {
            out.add(new CachedNetworkPrimitive(
                    "net@" + e.getKey(), e.getValue().item));
        }
        return out;
    }

    private NetworkItem trainNew(String inputAtom, String outputAtom) {
        long s = seedBase + spawnCounter * 1009L;
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(dim, s);
        for (String sym : vocabulary) table.embed(sym);
        if (!vocabulary.contains(inputAtom)) table.embed(inputAtom);
        if (!vocabulary.contains(outputAtom)) table.embed(outputAtom);

        VectorTransform bridge = new VectorTransform(dim, dim, s + 31L);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("embed",  new EmbedSymbol(table))
                .addNode("bridge", bridge)
                .addNode("lookup", new LookupSymbol(table))
                .addNode("output", new StringOutputPrimitive())
                .build();

        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new StringValue(outputAtom), 0.0, 0)));

        CarvingResult carving = new BackwardChainingCarver(s + 13L)
                .carve(tg, mandates, new StringValue(inputAtom));
        if (carving == null) {
            throw new IllegalStateException(
                    "NetworkCache could not carve a network for '"
                            + inputAtom + "' -> '" + outputAtom + "'");
        }
        InlineTrainer.Result tr = new InlineTrainer(
                carving, mandates, /* lr */ 0.1, /* maxSteps */ 500, /* checkEvery */ 25).run();
        if (!tr.converged()) {
            throw new IllegalStateException(
                    "NetworkCache training did not converge for '"
                            + inputAtom + "' -> '" + outputAtom + "'");
        }
        return NetworkItem.fromCarving(carving, ValueType.STRING, ValueType.STRING);
    }

    private void enforceMax() {
        if (maxNetworks.isEmpty()) return;
        int max = maxNetworks.getAsInt();
        while (entries.size() > max) {
            String victim = leastSuccessfulKey();
            if (victim == null) break;
            entries.remove(victim);
            evictionCounter++;
        }
    }

    private String leastSuccessfulKey() {
        String worst = null;
        int worstCount = Integer.MAX_VALUE;
        for (Map.Entry<String, Entry> e : entries.entrySet()) {
            if (e.getValue().successCount < worstCount) {
                worstCount = e.getValue().successCount;
                worst = e.getKey();
            }
        }
        return worst;
    }

    private static final class Entry {
        final NetworkItem item;
        int successCount;

        Entry(NetworkItem item, int successCount) {
            this.item = item;
            this.successCount = successCount;
        }
    }
}

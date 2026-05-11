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
import java.util.Objects;
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
     * The (input atom -> output atom) mapping for each stored entry. Used
     * for inventory inspection and for BFS-style reasoning over the cache
     * (e.g., "what atoms are reachable from this root?").
     */
    public Map<String, String> mappings() {
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<String, Entry> e : entries.entrySet()) {
            out.put(e.getKey(), e.getValue().outputAtom);
        }
        return out;
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
        Entry e = new Entry(item, outputAtom, 1);
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
                    "net_" + e.getKey() + "_to_" + e.getValue().outputAtom,
                    e.getValue().item));
        }
        return out;
    }

    /**
     * BFS over the cache's (input -> output) graph from {@code root}.
     * Returns the set of atoms reachable from root by following any
     * number of cached mappings, plus the longest-path "frontier" atom
     * (the atom found at maximum BFS depth — useful when the cache
     * needs to spawn a new bridge to extend its reach).
     */
    public Reachability bfsFrom(String root) {
        Set<String> reachable = new java.util.LinkedHashSet<>();
        reachable.add(root);
        List<String> frontier = new ArrayList<>();
        frontier.add(root);
        String deepestNonRoot = null;
        while (!frontier.isEmpty()) {
            List<String> nextFrontier = new ArrayList<>();
            for (String atom : frontier) {
                Entry e = entries.get(atom);
                if (e == null) continue;
                String next = e.outputAtom;
                if (reachable.add(next)) {
                    nextFrontier.add(next);
                    deepestNonRoot = next;
                }
            }
            frontier = nextFrontier;
        }
        return new Reachability(reachable, deepestNonRoot == null ? root : deepestNonRoot);
    }

    /**
     * Result of a BFS over the cache. {@code reachable} is the set of
     * atoms reachable from the BFS root by following cached mappings.
     * {@code frontier} is the deepest atom in that set (or root itself
     * if root has no outgoing entry).
     */
    public record Reachability(Set<String> reachable, String frontier) {
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

        // Two-phase training that prevents the cross-cache shortcut:
        //
        // Phase 1 — pre-shape the bridge toward identity on every atom in
        //   the vocabulary. This overcomes the random Xavier init and
        //   gives the bridge a known starting structure (W ≈ I) before
        //   the targeted shift is applied. Without this phase, the
        //   subsequent positive training collapses W to a rank-1 outer
        //   product (output - 0) · input^T / |input|^2 and the bridge
        //   generalizes to *anything* aligned with the input atom — the
        //   shortcut behaviour that originally broke this demo.
        //
        // Phase 2 — train (input → output) while continuously reinforcing
        //   identity on the remaining vocab atoms. The bridge ends up as
        //   "identity + targeted shift on input": bridge(embed(input)) ≈
        //   embed(output), bridge(embed(other)) ≈ embed(other).
        List<String> identityAtoms = new ArrayList<>(vocabulary);
        identityAtoms.removeIf(a -> a.equals(inputAtom) || a.equals(outputAtom));

        // Phase 1: identity on every atom in vocab, including the input
        // and output atoms (the positive shift hasn't started yet).
        for (int epoch = 0; epoch < 300; epoch++) {
            for (String d : vocabulary) {
                bridge.apply(List.of(new sibarum.strnn.value.MatrixValue(table.embed(d))));
                bridge.backward(new sibarum.strnn.value.MatrixValue(table.embed(d)));
                bridge.step(0.1);
            }
        }

        // Phase 2: positive shift + maintained identity on non-trained atoms.
        for (int epoch = 0; epoch < 500; epoch++) {
            // Positive: input -> output
            bridge.apply(List.of(new sibarum.strnn.value.MatrixValue(table.embed(inputAtom))));
            bridge.backward(new sibarum.strnn.value.MatrixValue(table.embed(outputAtom)));
            bridge.step(0.1);
            // Maintain identity on the rest
            for (String d : identityAtoms) {
                bridge.apply(List.of(new sibarum.strnn.value.MatrixValue(table.embed(d))));
                bridge.backward(new sibarum.strnn.value.MatrixValue(table.embed(d)));
                bridge.step(0.05);
            }
        }

        // Verify the positive mapping holds end-to-end through the carving
        sibarum.strnn.value.Value finalCheck = carving.graph().execute();
        String got = ((StringValue) finalCheck).s();
        if (!got.equals(outputAtom)) {
            throw new IllegalStateException(
                    "NetworkCache training did not satisfy '"
                            + inputAtom + "' -> '" + outputAtom + "': got '" + got + "'");
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
        final String outputAtom;
        int successCount;

        Entry(NetworkItem item, String outputAtom, int successCount) {
            this.item = item;
            this.outputAtom = Objects.requireNonNull(outputAtom);
            this.successCount = successCount;
        }
    }
}

package sibarum.strnn.demo;

import sibarum.strnn.cache.CachedNetworkPrimitive;
import sibarum.strnn.cache.NetworkCache;
import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.StringOutputPrimitive;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalInt;

/**
 * The grand finale: a substrate that builds itself.
 *
 * Start with an empty NetworkCache. For each (input, output) mandate in a
 * sequence:
 *
 *   1. Try to carve a composition from the cache's current inventory.
 *   2. If the carver finds a valid chain that satisfies the mandate, use
 *      it — no spawn. The cache reused what it already had.
 *   3. If the carve fails (no chain from input to output in the current
 *      inventory), BFS the cache from the mandate's input to find the
 *      deepest reachable atom — the "frontier". Spawn a single new
 *      network that bridges from frontier to the mandate's output. Re-
 *      carve, which now succeeds because the new bridge closes the gap.
 *
 * After the sequence completes:
 *   - The cache's inventory is a minimal *spanning set* of bridges that
 *     covers every mandate seen.
 *   - With reuse, the cache will typically have fewer entries than the
 *     number of mandates submitted.
 *   - The cache learned which bridges are worth keeping by being asked.
 *
 * The architectural claim: this is the framework's first end-to-end
 * "substrate learns from use" loop. Mandates are the *input* to the cache,
 * not just a verification predicate on top of a pre-existing substrate.
 * The cache responds to each mandate by either using its existing
 * inventory (via the carver's composition) or extending its inventory
 * (via a minimal spawn). Either way, every mandate gets satisfied; the
 * substrate's contents drift toward whatever the mandate stream
 * exercises.
 *
 * Combined with phase 8's BFS reachability, the carver can compose
 * chains of arbitrary depth from whatever the cache contains. The
 * adaptive resolver in this demo orchestrates the spawn-when-needed
 * piece on top.
 */
public final class AdaptiveCarverDemo {

    private static final int DIM = 32;
    private static final List<String> VOCABULARY = List.of(
            "hot", "cold", "warm", "cool", "fire", "ice", "burn", "freeze",
            "up", "down", "left", "right", "x", "y", "z");

    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("AdaptiveCarverDemo: substrate builds itself from mandates");
        System.out.println("============================================================");

        NetworkCache cache = new NetworkCache(
                DIM, VOCABULARY,
                /* seedBase */ 42L,
                /* maxNetworks */ OptionalInt.empty(),
                /* trainingMode */ true);

        // Mandate sequence — a curriculum the substrate must adapt to.
        // Carefully designed to exercise:
        //   - Spawn from empty (mandate 1).
        //   - Extend an existing chain (mandates 2, 3).
        //   - Reuse an existing chain (mandate 4).
        //   - Start a parallel chain (mandate 5).
        //   - Extend the parallel chain (mandate 6).
        //   - Reuse with chain crossing (mandate 7 — uses both chains).
        List<String[]> mandates = List.of(
                new String[]{"hot",  "cold"},     // 1: spawn hot->cold
                new String[]{"hot",  "freeze"},   // 2: extend with cold->freeze
                new String[]{"hot",  "ice"},      // 3: extend with freeze->ice
                new String[]{"hot",  "freeze"},   // 4: REUSE (no spawn)
                new String[]{"warm", "cool"},     // 5: spawn warm->cool (new chain)
                new String[]{"warm", "ice"},      // 6: extend with cool->ice (or directly)
                new String[]{"cold", "ice"}       // 7: REUSE existing (cold->freeze->ice)
        );

        System.out.printf(Locale.ROOT, "%nmandate sequence (%d mandates):%n", mandates.size());
        for (int i = 0; i < mandates.size(); i++) {
            System.out.printf(Locale.ROOT, "  %d. %-6s -> %s%n",
                    i + 1, mandates.get(i)[0], mandates.get(i)[1]);
        }

        BackwardChainingCarver carver = new BackwardChainingCarver(2026L);
        int spawnCount = 0;
        int reuseCount = 0;

        for (int i = 0; i < mandates.size(); i++) {
            String[] m = mandates.get(i);
            ResolveResult result = resolve(cache, carver, m[0], m[1], i + 1);
            if (result.spawned) spawnCount++; else reuseCount++;
        }

        System.out.println();
        System.out.println("============================================================");
        System.out.println("Summary");
        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "mandates submitted: %d%n", mandates.size());
        System.out.printf(Locale.ROOT, "spawns triggered:   %d%n", spawnCount);
        System.out.printf(Locale.ROOT, "carves reused:      %d%n", reuseCount);
        System.out.printf(Locale.ROOT, "final cache size:   %d networks%n", cache.size());
        System.out.println();
        System.out.println("final cache inventory:");
        for (Map.Entry<String, String> e : cache.mappings().entrySet()) {
            System.out.printf(Locale.ROOT, "  %-6s -> %s   (success=%d)%n",
                    e.getKey(), e.getValue(),
                    cache.successCount(e.getKey()).orElse(-1));
        }

        if (cache.size() >= mandates.size()) {
            throw new AssertionError(
                    "expected fewer cached networks than mandates due to reuse; "
                            + "got " + cache.size() + " networks for " + mandates.size() + " mandates");
        }

        System.out.println();
        System.out.println("The substrate built itself.");
        System.out.println("  - Every mandate was satisfied.");
        System.out.println("  - Some triggered spawns (the cache didn't yet know how).");
        System.out.println("  - Some reused existing inventory (the cache already knew).");
        System.out.println("  - Final inventory is smaller than the mandate count: the cache");
        System.out.println("    learned a minimal spanning set of bridges.");
        System.out.println("  - Mandate ordering determines which bridges get added; the");
        System.out.println("    record of what was asked shaped the substrate.");
        System.out.println();
        System.out.println("Mandate, Carve, Compose — over a substrate that didn't exist");
        System.out.println("when the demo started talking to it.");
    }

    private record ResolveResult(boolean spawned, String chain) {
    }

    /**
     * Try to satisfy {@code input -> output} from the cache's current
     * inventory. If the carver can't build a valid chain, BFS the cache
     * to find the deepest atom reachable from {@code input}, spawn a new
     * bridge from that frontier to {@code output}, and re-carve.
     */
    private static ResolveResult resolve(
            NetworkCache cache,
            BackwardChainingCarver carver,
            String input,
            String output,
            int mandateNum) {
        System.out.println();
        System.out.println("------------------------------------------------------------");
        System.out.printf(Locale.ROOT,
                "mandate %d: '%s' -> '%s'%n", mandateNum, input, output);
        System.out.println("------------------------------------------------------------");

        CarvingResult carving = tryCarve(cache, carver, input, output);
        if (carving != null) {
            String chain = describeChain(carving);
            // Verify execution actually satisfies the mandate (the carver
            // may have found a chain that doesn't quite produce the target
            // due to internal misclassification; the verifier is the
            // authority).
            Value result = carving.graph().execute();
            VerificationReport rep = new MandateVerifier().verify(
                    carving.graph(),
                    new MandateSet(List.of(
                            Mandate.result(new StringValue(output), 0.0, 0))));
            if (rep.allSatisfied()) {
                System.out.printf(Locale.ROOT,
                        "  REUSE: cache already covers this mandate%n  chain: %s%n  terminal: '%s'%n",
                        chain, ((StringValue) result).s());
                cache.query(input, new StringValue(input));  // bump success
                return new ResolveResult(false, chain);
            }
            System.out.printf(Locale.ROOT,
                    "  carve succeeded structurally but verification failed (terminal='%s'); "
                            + "treating as spawn-required%n",
                    ((StringValue) result).s());
        }

        // Need to spawn. BFS the cache to find the frontier.
        NetworkCache.Reachability r = cache.bfsFrom(input);
        String frontier = r.frontier();
        String spawnInput, spawnOutput;
        if (frontier.equals(input) || r.reachable().contains(output)) {
            // Either the input has no outgoing entry in the cache (frontier == input)
            // or the output is already reachable but carve failed for some
            // other reason (e.g. the chain visited a node that produces
            // output via a non-trained mapping). Spawn a direct (input, output)
            // bridge to close the gap.
            spawnInput = input;
            spawnOutput = output;
        } else {
            // The cache reaches at least one new atom from input; extend
            // by one step from the deepest reachable atom.
            spawnInput = frontier;
            spawnOutput = output;
        }
        System.out.printf(Locale.ROOT,
                "  SPAWN: cache reachable from '%s' = %s; spawning '%s' -> '%s'%n",
                input, r.reachable(), spawnInput, spawnOutput);

        cache.getOrTrain(spawnInput, spawnOutput);

        carving = tryCarve(cache, carver, input, output);
        if (carving == null) {
            throw new AssertionError(
                    "carve still fails after spawning '" + spawnInput + "' -> '" + spawnOutput + "'");
        }
        String chain = describeChain(carving);
        Value result = carving.graph().execute();
        VerificationReport rep = new MandateVerifier().verify(
                carving.graph(),
                new MandateSet(List.of(
                        Mandate.result(new StringValue(output), 0.0, 0))));
        if (!rep.allSatisfied()) {
            throw new AssertionError(
                    "post-spawn verification failed; terminal='"
                            + ((StringValue) result).s() + "'");
        }
        System.out.printf(Locale.ROOT,
                "  PASS after spawn%n  chain: %s%n  terminal: '%s'%n",
                chain, ((StringValue) result).s());
        return new ResolveResult(true, chain);
    }

    private static CarvingResult tryCarve(
            NetworkCache cache,
            BackwardChainingCarver carver,
            String input,
            String output) {
        List<CachedNetworkPrimitive> prims = cache.primitives();
        if (prims.isEmpty()) return null;
        TransformationGraphBuilder builder = new TransformationGraphBuilder()
                .addNode("output", new StringOutputPrimitive());
        for (CachedNetworkPrimitive p : prims) {
            builder.addNode(p.name(), p);
        }
        TransformationGraph tg = builder.build();
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new StringValue(output), 0.0, 0)));
        return carver.carve(tg, mandates, new StringValue(input));
    }

    private static String describeChain(CarvingResult carving) {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (sibarum.strnn.computation.CompGraphNode n : carving.graph().topoOrder()) {
            String name = n.tNode().primitive().name();
            if (name.equals("string-output")) continue;
            if (!first) sb.append(" -> ");
            sb.append(name);
            first = false;
        }
        return sb.toString();
    }
}

package sibarum.strnn.demo;

import sibarum.strnn.cache.CachedNetworkPrimitive;
import sibarum.strnn.cache.NetworkCache;
import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.StringOutputPrimitive;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalInt;

/**
 * The carver composes cached subgraphs taken from a NetworkCache's
 * inventory. The cache is populated with single-step mappings; the
 * carver assembles whatever chain a mandate requires.
 *
 * Setup. An empty NetworkCache is populated with three mappings:
 *
 *   hot    -> cold
 *   cold   -> freeze
 *   freeze -> ice
 *
 * The cache uses a small distractor pool ("x", "y", "z") for the per-spawn
 * vocabularies. Each network's table is seeded with the distractors plus
 * its own input and output atom — so {hot, cold, x, y, z} for hotCold,
 * {cold, freeze, x, y, z} for coldFreeze, {freeze, ice, x, y, z} for
 * freezeIce. The disjoint-at-the-non-interface-atoms structure rules out
 * the linear-bridge shortcut from P6: if a cached network's bridge is fed
 * an atom it hasn't seen, lazy-init produces a random vector and lookup
 * resolves to one of the distractors, not to the mandate target.
 *
 * For each mandate, the demo:
 *   1. Builds a fresh outer TG from {@code cache.primitives()} plus a
 *      StringOutputPrimitive terminal.
 *   2. Performs a "shortcut check": applies every cached primitive to the
 *      root input and verifies none directly produces the mandate target
 *      (so any successful carving must be a composition).
 *   3. Submits the mandate to a {@link BackwardChainingCarver}.
 *   4. Executes the carving, verifies, and reports the carved chain.
 *
 * Three mandates exercise three depths of composition:
 *   - hot -> cold      (1-step, single primitive)
 *   - hot -> freeze    (2-step composition)
 *   - hot -> ice       (3-step composition)
 *
 * Same cache, same primitives, different chains.
 */
public final class CarverFromCacheDemo {

    private static final int DIM = 32;
    // The vocabulary passed to NetworkCache becomes both the pre-seeded
    // atoms in every spawn's table AND the regularization scope: each
    // spawn's bridge is trained to be identity on every vocab atom that
    // isn't its own (input, output) pair. Including the thermal atoms
    // (not just generic distractors) is what blocks cross-cache shortcuts
    // — coldFreeze applied to "hot" produces "hot" (identity), not "cold"
    // or "freeze", because coldFreeze's bridge was regularized to be
    // identity on "hot".
    private static final List<String> VOCABULARY = List.of(
            "hot", "cold", "warm", "cool", "fire", "ice", "burn", "freeze",
            "x", "y", "z");

    public static void main(String[] args) {
        // ---- Build and populate the cache ----
        NetworkCache cache = new NetworkCache(
                DIM, VOCABULARY,
                /* seedBase */ 42L,
                /* maxNetworks */ OptionalInt.empty(),
                /* trainingMode */ true);

        System.out.println("============================================================");
        System.out.println("CarverFromCacheDemo: cache becomes the substrate");
        System.out.println("============================================================");
        System.out.println();
        System.out.println("populating cache:");
        feed(cache, "hot",    "cold");
        feed(cache, "cold",   "freeze");
        feed(cache, "freeze", "ice");

        System.out.println();
        System.out.println("cache inventory:");
        for (Map.Entry<String, String> m : cache.mappings().entrySet()) {
            System.out.printf(Locale.ROOT, "  %-7s -> %s%n", m.getKey(), m.getValue());
        }

        // ---- Run mandates of increasing chain depth ----
        runMandate(cache, "hot",  "cold",   1);
        runMandate(cache, "hot",  "freeze", 2);
        runMandate(cache, "hot",  "ice",    3);

        System.out.println();
        System.out.println("CarverFromCacheDemo: same cache, three mandates, three different");
        System.out.println("composed chains — produced by the carver, not by the user.");
    }

    private static void feed(NetworkCache cache, String input, String output) {
        cache.getOrTrain(input, output);
        System.out.printf(Locale.ROOT, "  spawned net_%s_to_%s%n", input, output);
    }

    /**
     * Build a TG from the cache's primitives + a terminal, shortcut-check,
     * carve, execute, verify, and print the carved chain.
     */
    private static void runMandate(
            NetworkCache cache, String input, String output, int expectedDepth) {
        System.out.println();
        System.out.println("------------------------------------------------------------");
        System.out.printf(Locale.ROOT,
                "mandate: '%s' -> '%s'   (expecting %d-step composition)%n",
                input, output, expectedDepth);
        System.out.println("------------------------------------------------------------");

        List<CachedNetworkPrimitive> prims = cache.primitives();

        // Shortcut check: no single cached primitive should produce the target
        // directly from the root, otherwise the demo's claim collapses.
        System.out.println("shortcut check (each primitive applied to root):");
        for (CachedNetworkPrimitive p : prims) {
            Value out = p.apply(List.of(new StringValue(input)));
            String s = ((StringValue) out).s();
            boolean shortcut = s.equals(output);
            System.out.printf(Locale.ROOT, "  %s('%s') -> '%s'%s%n",
                    p.name(), input, s, shortcut ? "  *** SHORTCUT ***" : "");
            if (shortcut && expectedDepth > 1) {
                throw new AssertionError(
                        p.name() + " is a single-step shortcut for the mandate; "
                                + "any successful carving here may be coincidence");
            }
        }

        // Build outer TG: every cached primitive plus the terminal
        TransformationGraphBuilder builder = new TransformationGraphBuilder()
                .addNode("output", new StringOutputPrimitive());
        for (CachedNetworkPrimitive p : prims) {
            builder.addNode(p.name(), p);
        }
        TransformationGraph tg = builder.build();

        // Mandate + carve
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new StringValue(output), 0.0, 0)));
        BackwardChainingCarver carver = new BackwardChainingCarver(2026L + expectedDepth);
        CarvingResult carving = carver.carve(tg, mandates, new StringValue(input));
        if (carving == null) {
            throw new AssertionError("carver failed for '" + input + "' -> '" + output + "'");
        }

        // Print the carved chain (skip terminal duplicates)
        System.out.println("carved chain:");
        List<String> chainNodes = new ArrayList<>();
        for (CompGraphNode n : carving.graph().topoOrder()) {
            String primName = n.tNode().primitive().name();
            if (!primName.equals("string-output")) chainNodes.add(primName);
            StringBuilder sb = new StringBuilder("  ").append(n.id())
                    .append(" [").append(primName).append("]");
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource src = n.slot(i);
                if (src == null) sb.append(" slot").append(i).append("<-ROOT");
                else sb.append(" slot").append(i).append("<-").append(src.source().id());
            }
            System.out.println(sb);
        }

        // Execute and verify
        Value terminal = carving.graph().execute();
        VerificationReport rep = new MandateVerifier().verify(carving.graph(), mandates);
        System.out.printf(Locale.ROOT, "terminal: '%s'  mandate:%s   chain length: %d%n",
                ((StringValue) terminal).s(),
                rep.allSatisfied() ? "PASS" : "FAIL",
                chainNodes.size());

        if (!rep.allSatisfied()) {
            throw new AssertionError(
                    "mandate did not pass; terminal=" + terminal);
        }
        if (chainNodes.size() != expectedDepth) {
            System.out.printf(Locale.ROOT,
                    "  note: chain length %d differs from expected %d — the cache may%n"
                            + "        contain a shorter path the carver legitimately preferred.%n",
                    chainNodes.size(), expectedDepth);
        }
    }
}

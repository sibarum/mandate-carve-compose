package sibarum.strnn.demo;

import sibarum.strnn.cache.CachedNetworkPrimitive;
import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.LookupSymbol;
import sibarum.strnn.cache.NetworkItem;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.cache.VectorTransform;
import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.StringOutputPrimitive;
import sibarum.strnn.training.InlineTrainer;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;
import java.util.Locale;

/**
 * Key-Network landing: the cache stores entire trained subgraphs as items
 * and the carver composes them.
 *
 * Setup. Two inner networks are pre-trained inline, each on a single
 * mapping:
 *
 *   N_hot_to_cold : StringValue("hot")  -> StringValue("cold")
 *   N_cold_to_freeze : StringValue("cold") -> StringValue("freeze")
 *
 * Each is wrapped as a {@link NetworkItem} and exposed to the outer carver
 * as a {@link CachedNetworkPrimitive}. From the outer carver's point of
 * view, these are just two deterministic primitives, each with declared
 * input and output types {@code String -> String}.
 *
 * The outer mandate: root input "hot", result "freeze". The carver
 * backward-chains:
 *
 *   terminal needs "freeze"
 *     -> N_cold_to_freeze produces "freeze" from "cold"
 *        -> N_hot_to_cold produces "cold" from "hot"
 *           -> root matches "hot"
 *
 * The carver assembles a two-step pipeline composing the two cached
 * networks. Executing it produces "freeze".
 *
 * This is what Key-Network looks like in this framework: the substrate's
 * stored items are *behaviours*, the carver retrieves them by structural
 * fit to the mandate, and composition happens via the same machinery that
 * composes any other primitive. The same MandateVerifier confirms the
 * pipeline does what it's supposed to.
 */
public final class KeyNetworkDemo {

    private static final int DIM = 32;

    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("Key-Network: cached subgraphs composed by the carver");
        System.out.println("============================================================");

        // ---- Pre-train two inner networks ----
        // Vocabularies are deliberately structured so that
        // (a) hotCold has "hot" and "cold" (it has to);
        // (b) coldFreeze has "cold" and "freeze" but NOT "hot" — so
        //     applying coldFreeze directly to "hot" lazy-inits a fresh
        //     random embedding for "hot" inside coldFreeze's table, and
        //     its bridge (trained only on embed("cold")) produces an
        //     output unrelated to "freeze". This rules out the
        //     single-step shortcut the linear bridge would otherwise find,
        //     forcing the outer carver to compose the two networks to
        //     reach the result.
        List<String> hotColdVocab = List.of(
                "hot", "cold", "warm", "cool", "fire", "ice", "burn", "freeze");
        List<String> coldFreezeVocab = List.of(
                "cold", "freeze", "ice", "winter", "snow", "frost", "chill", "cool");

        NetworkItem netHotCold    = trainMapping("hot",  "cold",   42L,  7L,  hotColdVocab);
        NetworkItem netColdFreeze = trainMapping("cold", "freeze", 99L, 13L, coldFreezeVocab);

        // Diagnostic: confirm coldFreeze applied directly to "hot" does NOT
        // produce "freeze" — i.e., the shortcut path is unavailable.
        Value shortcutCheck = netColdFreeze.execute(new StringValue("hot"));
        System.out.printf(Locale.ROOT,
                "%nshortcut check: coldFreeze.execute('hot') = '%s'  (must NOT be 'freeze')%n",
                ((StringValue) shortcutCheck).s());
        if ("freeze".equals(((StringValue) shortcutCheck).s())) {
            throw new AssertionError(
                    "shortcut still available — coldFreeze maps 'hot' directly to 'freeze'; "
                            + "vocabulary or seeds need adjustment to force composition");
        }

        // ---- Outer substrate: two cached networks + a terminal ----
        CachedNetworkPrimitive primHotCold    = new CachedNetworkPrimitive("net_hot->cold",   netHotCold);
        CachedNetworkPrimitive primColdFreeze = new CachedNetworkPrimitive("net_cold->freeze", netColdFreeze);

        TransformationGraph outerTg = new TransformationGraphBuilder()
                .addNode("hotCold",    primHotCold)
                .addNode("coldFreeze", primColdFreeze)
                .addNode("output",     new StringOutputPrimitive())
                .build();

        System.out.println();
        System.out.println("outer substrate (cached networks as primitives):");
        outerTg.edges().forEach(e -> System.out.printf(Locale.ROOT,
                "  %-12s --> %s%n", e.from().id(), e.to().id()));

        // ---- Outer mandate: root "hot" -> result "freeze" ----
        MandateSet outerMandates = new MandateSet(List.of(
                Mandate.result(new StringValue("freeze"), 0.0, 0)));

        System.out.println();
        System.out.println("outer mandate: root='hot', result='freeze'");
        System.out.println("the only path that satisfies this is a composition: hot --> cold --> freeze");
        System.out.println();

        BackwardChainingCarver outerCarver = new BackwardChainingCarver(2026L);
        CarvingResult outerCarving = outerCarver.carve(
                outerTg, outerMandates, new StringValue("hot"));
        if (outerCarving == null) {
            throw new AssertionError("outer carver failed to compose cached networks");
        }

        System.out.println("carved outer pipeline:");
        for (CompGraphNode n : outerCarving.graph().topoOrder()) {
            StringBuilder sb = new StringBuilder("  ").append(n.id())
                    .append(" [").append(n.tNode().primitive().name()).append("]");
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource src = n.slot(i);
                if (src == null) sb.append(" slot").append(i).append("<-ROOT");
                else sb.append(" slot").append(i).append("<-").append(src.source().id());
            }
            System.out.println(sb);
        }

        // ---- Execute and verify ----
        Value terminal = outerCarving.graph().execute();
        VerificationReport rep = new MandateVerifier().verify(outerCarving.graph(), outerMandates);

        System.out.println();
        System.out.printf(Locale.ROOT, "executed pipeline -> terminal value: '%s'%n",
                ((StringValue) terminal).s());
        System.out.printf(Locale.ROOT, "outer mandate verification:        %s%n",
                rep.allSatisfied() ? "PASS" : "FAIL");

        if (!rep.allSatisfied()) {
            throw new AssertionError(
                    "outer mandate did not pass; terminal = " + terminal);
        }

        System.out.println();
        System.out.println("Key-Network landed.");
        System.out.println("  - The substrate's stored items were not vectors but trained subgraphs.");
        System.out.println("  - The carver retrieved them via structural fit to the mandate.");
        System.out.println("  - Composition happened through the same Primitive/Carver machinery");
        System.out.println("    that composes anything else; the same MandateVerifier confirmed it.");
    }

    /**
     * Pre-train an inner network that maps {@code inputAtom} to
     * {@code outputAtom} via embed-bridge-lookup. Returns a NetworkItem
     * wrapping the trained carving so the outer demo can use it as a
     * cache entry.
     */
    private static NetworkItem trainMapping(
            String inputAtom, String outputAtom, long tableSeed, long bridgeSeed,
            List<String> vocabulary) {
        System.out.println();
        System.out.printf(Locale.ROOT,
                "------ training inner network: '%s' -> '%s' (vocab=%s) ------%n",
                inputAtom, outputAtom, vocabulary);

        SymbolEmbeddingTable table = new SymbolEmbeddingTable(DIM, tableSeed);
        for (String s : vocabulary) table.embed(s);

        VectorTransform bridge = new VectorTransform(DIM, DIM, bridgeSeed);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("embed",  new EmbedSymbol(table))
                .addNode("bridge", bridge)
                .addNode("lookup", new LookupSymbol(table))
                .addNode("output", new StringOutputPrimitive())
                .build();

        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new StringValue(outputAtom), 0.0, 0)));

        CarvingResult carving = new BackwardChainingCarver(tableSeed + bridgeSeed)
                .carve(tg, mandates, new StringValue(inputAtom));
        if (carving == null) {
            throw new AssertionError(
                    "inner carver failed for '" + inputAtom + "' -> '" + outputAtom + "'");
        }

        InlineTrainer.Result trainResult = new InlineTrainer(
                carving, mandates,
                /* lr */ 0.1, /* maxSteps */ 500, /* checkEvery */ 25).run();
        if (!trainResult.converged()) {
            throw new AssertionError("inner training did not converge for '"
                    + inputAtom + "' -> '" + outputAtom + "'");
        }
        System.out.printf(Locale.ROOT, "  converged in %d step%s%n",
                trainResult.stepsTaken(), trainResult.stepsTaken() == 1 ? "" : "s");

        return NetworkItem.fromCarving(carving, ValueType.STRING, ValueType.STRING);
    }
}

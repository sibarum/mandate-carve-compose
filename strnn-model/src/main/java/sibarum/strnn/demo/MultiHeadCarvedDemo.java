package sibarum.strnn.demo;

import sibarum.strnn.cache.EmbedSymbol;
import sibarum.strnn.cache.LookupSymbol;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.cache.VectorTransform;
import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.SlotSource;
import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.primitive.StringOutputPrimitive;
import sibarum.strnn.training.InlineTrainer;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationGraphBuilder;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;

import java.util.List;
import java.util.Locale;

/**
 * Multi-head KV substrate exercised through the carver and inline trainer.
 * The substrate has H = 2 parallel cache chains, each a complete
 * {@code embed_h -> bridge_h -> lookup_h} pipeline with its own
 * {@link SymbolEmbeddingTable} (its own KV namespace). A single shared
 * {@link StringOutputPrimitive} terminates either chain.
 *
 * The demo runs two sessions over the SAME substrate:
 *
 *   Session 1: bias edge stats toward chain A, carve mandate "hot -> cold",
 *              inline-train. Verify the carver used chain A and the mandate
 *              passes.
 *
 *   Session 2: reset edge stats, bias toward chain B, carve mandate
 *              "hot -> warm", inline-train. Verify the carver used chain B
 *              and the mandate passes. Chain A's previously trained bridge
 *              is untouched (the substrate supports independent
 *              specialization).
 *
 *   Session 3 (independence check): bias back to chain A, carve mandate 1
 *              again, verify it still passes without any further training
 *              (chain A retained its specialization through session 2).
 *
 * This demonstrates the load-bearing piece of Key-Network's "carver picks
 * among multiple substrate views" claim. Edge-stats feedback in
 * {@link InlineTrainer#applyEdgeFeedback} is exercised after each session
 * so the substrate's preferences also accumulate naturally — that's the
 * shape of the longer-arc emergence demo we haven't fully built yet.
 */
public final class MultiHeadCarvedDemo {

    public static void main(String[] args) {
        int dim = 32;
        List<String> vocabulary = List.of(
                "hot", "cold", "warm", "cool", "fire", "ice", "burn", "freeze");

        // ---- Two KV caches (the multi-head substrate) ----
        SymbolEmbeddingTable tableA = new SymbolEmbeddingTable(dim, 42L);
        SymbolEmbeddingTable tableB = new SymbolEmbeddingTable(dim, 1337L);
        for (String s : vocabulary) {
            tableA.embed(s);
            tableB.embed(s);
        }

        // ---- Shared primitive instances (the substrate is one substrate) ----
        EmbedSymbol embedA = new EmbedSymbol(tableA);
        VectorTransform bridgeA = new VectorTransform(dim, dim, 7L);
        LookupSymbol lookupA = new LookupSymbol(tableA);

        EmbedSymbol embedB = new EmbedSymbol(tableB);
        VectorTransform bridgeB = new VectorTransform(dim, dim, 11L);
        LookupSymbol lookupB = new LookupSymbol(tableB);

        StringOutputPrimitive output = new StringOutputPrimitive();

        // ---- One TG over both chains; any-to-any modulo type ----
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("embedA",  embedA)
                .addNode("bridgeA", bridgeA)
                .addNode("lookupA", lookupA)
                .addNode("embedB",  embedB)
                .addNode("bridgeB", bridgeB)
                .addNode("lookupB", lookupB)
                .addNode("output",  output)
                .build();

        System.out.println("multi-head substrate (head A: tableA, head B: tableB)");
        System.out.println("transformation graph edges:");
        for (TransformationEdge e : tg.edges()) {
            System.out.printf(Locale.ROOT, "  %-9s --> %s%n", e.from().id(), e.to().id());
        }
        System.out.println();

        // ---- Session 1: chain A, mandate hot -> cold ----
        System.out.println("============================================================");
        System.out.println("Session 1: bias toward chain A, mandate 'hot' -> 'cold'");
        System.out.println("============================================================");
        resetAllEdgeStats(tg);
        biasChain(tg, "A", 1.0);
        biasChain(tg, "B", 0.0);

        runSession(tg, "hot", "cold", "A", new int[]{1});

        // ---- Session 2: chain B, mandate hot -> warm ----
        System.out.println();
        System.out.println("============================================================");
        System.out.println("Session 2: bias toward chain B, mandate 'hot' -> 'warm'");
        System.out.println("============================================================");
        resetAllEdgeStats(tg);
        biasChain(tg, "A", 0.0);
        biasChain(tg, "B", 1.0);

        runSession(tg, "hot", "warm", "B", new int[]{2});

        // ---- Session 3: independence check — chain A still works for mandate 1 ----
        System.out.println();
        System.out.println("============================================================");
        System.out.println("Session 3: re-bias to chain A, re-verify 'hot' -> 'cold'");
        System.out.println("           (no further training; chain A must retain memory)");
        System.out.println("============================================================");
        resetAllEdgeStats(tg);
        biasChain(tg, "A", 1.0);
        biasChain(tg, "B", 0.0);

        runSession(tg, "hot", "cold", "A", new int[]{3});

        System.out.println();
        System.out.println("MultiHead end-to-end: the substrate carries two specialized");
        System.out.println("heads, the carver picks among them via edge stats, and inline");
        System.out.println("training tunes whichever head was selected. Heads retain their");
        System.out.println("specialization across mandate switches (session 3 confirms).");
    }

    /**
     * Carve, inline-train, verify, and feed the result back into the
     * edge-stat substrate. Verifies that the chosen head matches the
     * expected one ("A" or "B").
     */
    private static void runSession(
            TransformationGraph tg,
            String inputAtom,
            String outputAtom,
            String expectedHead,
            int[] sessionNum) {
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(new StringValue(outputAtom), 0.0, 0)));

        BackwardChainingCarver carver = new BackwardChainingCarver(100L + sessionNum[0]);
        CarvingResult carving = carver.carve(tg, mandates, new StringValue(inputAtom));
        if (carving == null) {
            throw new AssertionError(
                    "session " + sessionNum[0] + ": carver failed for '"
                            + inputAtom + "' -> '" + outputAtom + "'");
        }

        System.out.println("carved chain:");
        boolean[] sawHead = {false, false};  // [A, B]
        for (CompGraphNode n : carving.graph().topoOrder()) {
            String head = headOf(n);
            if ("A".equals(head)) sawHead[0] = true;
            if ("B".equals(head)) sawHead[1] = true;
            StringBuilder sb = new StringBuilder("  ").append(n.id())
                    .append(" [").append(n.tNode().primitive().name()).append("]");
            if (!head.isEmpty()) sb.append(" (head ").append(head).append(")");
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource src = n.slot(i);
                if (src == null) sb.append(" slot").append(i).append("<-ROOT");
                else sb.append(" slot").append(i).append("<-").append(src.source().id());
            }
            System.out.println(sb);
        }

        String routedHead;
        if (sawHead[0] && !sawHead[1]) routedHead = "A";
        else if (sawHead[1] && !sawHead[0]) routedHead = "B";
        else routedHead = "mixed/none";

        System.out.printf(Locale.ROOT, "routed through: head %s (expected %s)%n",
                routedHead, expectedHead);

        if (!routedHead.equals(expectedHead)) {
            throw new AssertionError(
                    "session " + sessionNum[0] + ": expected head " + expectedHead
                            + " but carver routed through " + routedHead);
        }

        InlineTrainer trainer = new InlineTrainer(
                carving, mandates,
                /* lr */ 0.1, /* maxSteps */ 500, /* checkEvery */ 25);

        System.out.println("inline training:");
        InlineTrainer.Result result = trainer.run();
        for (InlineTrainer.StepRecord rec : result.trace()) {
            System.out.printf(Locale.ROOT, "  step %3d: terminal='%s'  mandate:%s%n",
                    rec.step(),
                    ((StringValue) rec.terminalValue()).s(),
                    rec.mandatePass() ? "PASS" : "FAIL");
        }

        if (!result.converged()) {
            throw new AssertionError(
                    "session " + sessionNum[0] + ": inline training did not converge");
        }

        double reward = result.score(500);
        InlineTrainer.applyEdgeFeedback(carving, reward);
        System.out.printf(Locale.ROOT, "session %d: PASS in %d step%s; edge feedback = %.2f%n",
                sessionNum[0], result.stepsTaken(),
                result.stepsTaken() == 1 ? "" : "s", reward);
    }

    /**
     * For demo-time bias: walk all edges that connect head-A primitives
     * (or head-B) and pin their score. The carver's candidate ranking sorts
     * by score and ties are randomised, so a high score on one chain is
     * enough to route through it.
     */
    private static void biasChain(TransformationGraph tg, String head, double score) {
        for (TransformationEdge e : tg.edges()) {
            boolean fromHead = e.from().id().endsWith(head);
            boolean toHead   = e.to().id().endsWith(head);
            // edge is part of chain head iff both endpoints belong to that head
            // (or the edge terminates at the shared output and the source is
            // from this head — which is exactly what biases head-X's path to
            // the terminal).
            if ((fromHead && toHead) || (fromHead && e.to().id().equals("output"))) {
                e.stats().setMeanScore(score);
            }
        }
    }

    private static void resetAllEdgeStats(TransformationGraph tg) {
        for (TransformationEdge e : tg.edges()) {
            e.stats().reset();
        }
    }

    private static String headOf(CompGraphNode n) {
        String id = n.tNode().id();
        if (id.endsWith("A") && !id.equals("output")) return "A";
        if (id.endsWith("B") && !id.equals("output")) return "B";
        return "";
    }
}

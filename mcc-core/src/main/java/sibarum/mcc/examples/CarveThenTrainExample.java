package sibarum.mcc.examples;

import sibarum.mcc.carving.BackwardChainingCarver;
import sibarum.mcc.carving.CarvingResult;
import sibarum.mcc.embedding.Embed;
import sibarum.mcc.embedding.SymbolEmbeddingTable;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.mandate.Mandate;
import sibarum.mcc.mandate.MandateSet;
import sibarum.mcc.mandate.MandateVerifier;
import sibarum.mcc.op.Terminal;
import sibarum.mcc.training.Corpus;
import sibarum.mcc.training.Example;
import sibarum.mcc.training.GraphTrainer;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.StringValue;
import sibarum.mcc.value.ValueType;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * End-to-end Mandate-Carve-Compose example.
 *
 * <ol>
 *   <li><b>Mandate</b>: declare a target terminal value.</li>
 *   <li><b>Carve</b>: backward-chain from the mandate through the
 *       substrate; produce a runnable {@link ComputationGraph} plus
 *       per-node simulated values.</li>
 *   <li><b>Compose / Train</b>: the carved graph routes
 *       {@code Embed("alpha") → Terminal}. Gradient flow on the
 *       Embed table converges its stored vector to the mandate
 *       target.</li>
 * </ol>
 *
 * <p>Run with:
 * <pre>
 *   mvn exec:java -pl mcc-core -Dexec.mainClass=sibarum.mcc.examples.CarveThenTrainExample
 * </pre>
 */
public final class CarveThenTrainExample {

    public static void main(String[] args) {
        // 1. Substrate: Embed (str → matrix) + Terminal (matrix passthrough).
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(4, 42L);
        Embed embed = new Embed(table);
        StringValue rootSymbol = new StringValue("alpha");
        embed.apply(List.of(rootSymbol));   // seed "alpha" into the table

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("emb", embed)
                .addNode("term", new Terminal(ValueType.MATRIX))
                .build();

        // 2. Mandate: the desired terminal value.
        MatrixValue target = new MatrixValue(new double[] { 0.5, -0.5, 0.5, -0.5 });
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(target, 0.01, /* ordering */ 10)
        ));

        // 3. Carve.
        BackwardChainingCarver carver = new BackwardChainingCarver(0L);
        CarvingResult carving = carver.carve(tg, mandates, rootSymbol);
        if (carving == null) {
            System.err.println("carving failed");
            System.exit(1);
        }
        ComputationGraph carved = carving.graph();
        System.out.println("carved graph with " + carved.nodes().size() + " nodes:");
        for (var n : carved.topoOrder()) {
            System.out.println("  " + n.id() + " (" + n.tNode().primitive().name() + ")");
        }

        // 4. Verify before training (will fail — the table's random init
        //    almost certainly does not match the mandate target).
        carved.execute();
        boolean preTrain = new MandateVerifier().verify(carved, mandates).allSatisfied();
        System.out.println("pre-train mandate satisfied:  " + preTrain);

        // 5. Train.
        GraphTrainer trainer = new GraphTrainer(
                carved,
                (g, ex) -> { /* roots stay bound from carving */ },
                /* lr */ 0.1,
                /* stepEveryExample */ true
        );
        Corpus corpus = new SingleExampleCorpus(target);
        for (int e = 0; e < 200; e++) {
            trainer.trainEpoch(corpus);
        }

        // 6. Re-verify.
        carved.execute();
        boolean postTrain = new MandateVerifier().verify(carved, mandates).allSatisfied();
        System.out.println("post-train mandate satisfied: " + postTrain);

        double[] finalVec = table.rawVector("alpha");
        System.out.printf("alpha's trained vector: [%+.4f, %+.4f, %+.4f, %+.4f]%n",
                finalVec[0], finalVec[1], finalVec[2], finalVec[3]);
        System.out.printf("              target:   [%+.4f, %+.4f, %+.4f, %+.4f]%n",
                target.data()[0], target.data()[1], target.data()[2], target.data()[3]);
    }

    private record SingleExampleCorpus(MatrixValue target) implements Corpus {
        @Override public String name() { return "single-target"; }
        @Override public long size() { return 1; }
        @Override
        public Iterator<Example> stream() {
            return List.of(new Example("only", Map.of(), target)).iterator();
        }
    }
}

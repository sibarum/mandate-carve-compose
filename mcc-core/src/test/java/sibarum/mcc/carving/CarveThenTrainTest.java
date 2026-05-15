package sibarum.mcc.carving;

import org.junit.jupiter.api.Test;
import sibarum.mcc.embedding.Embed;
import sibarum.mcc.embedding.SymbolEmbeddingTable;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.mandate.Mandate;
import sibarum.mcc.mandate.MandateSet;
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

import static org.junit.jupiter.api.Assertions.*;

/**
 * Closes the Mandate-Carve-Compose loop end-to-end:
 *
 * <ol>
 *   <li><b>Mandate</b>: specify the desired terminal value.</li>
 *   <li><b>Carve</b>: {@link BackwardChainingCarver} builds a graph from
 *       the substrate that <em>could</em> produce that value if the
 *       Embed primitive's parameters were the target.</li>
 *   <li><b>Compose</b>: train the carved graph against the result
 *       mandate. Since the carver's chosen Embed → Terminal chain
 *       can route any vector to the terminal, gradient flow on the
 *       Embed table converges its stored vector to the mandate target.</li>
 * </ol>
 */
class CarveThenTrainTest {

    @Test
    void carveThenTrainConvergesEmbeddingTowardMandateTarget() {
        // Pre-populate the table so the carver has a symbol to invert to.
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(4, 42L);
        Embed embed = new Embed(table);
        StringValue rootSymbol = new StringValue("alpha");
        embed.apply(List.of(rootSymbol));  // ensures "alpha" exists in the table

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("emb", embed)
                .addNode("term", new Terminal(ValueType.MATRIX))
                .build();

        // Pick a target that does NOT match alpha's current embedding —
        // we want training to move the embedding toward it.
        MatrixValue target = new MatrixValue(new double[] { 0.5, -0.5, 0.5, -0.5 });
        MandateSet mandates = new MandateSet(List.of(
                Mandate.result(target, 0.01, 10)
        ));

        BackwardChainingCarver carver = new BackwardChainingCarver(0L);
        CarvingResult carving = carver.carve(tg, mandates, rootSymbol);
        assertNotNull(carving, "carver should produce a graph: Terminal ← Embed ← root('alpha')");

        ComputationGraph carved = carving.graph();
        // Snapshot the initial Embed vector for alpha.
        double[] before = table.rawVector("alpha").clone();
        double initialL2 = l2(before, target.data());

        // Train the carved graph against the result mandate's target.
        // Root is already bound by the carver — the rootBinder is a no-op.
        GraphTrainer trainer = new GraphTrainer(
                carved,
                (g, ex) -> { /* roots stay bound from carving */ },
                0.1,
                /* stepEveryExample */ true
        );
        Corpus corpus = new SingleExampleCorpus(rootSymbol, target);

        double lossLast = trainer.trainEpoch(corpus);
        for (int i = 0; i < 100; i++) lossLast = trainer.trainEpoch(corpus);

        double[] after = table.rawVector("alpha");
        double finalL2 = l2(after, target.data());
        assertTrue(finalL2 < initialL2 * 0.05,
                "embedding must converge toward mandate target; initial L2 "
                        + initialL2 + ", final L2 " + finalL2);
        assertTrue(Double.isFinite(lossLast), "final loss must be finite: " + lossLast);
    }

    private static double l2(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    /** A corpus that yields the same {target} every example. */
    private record SingleExampleCorpus(StringValue rootSymbol, MatrixValue target) implements Corpus {
        @Override public String name() { return "single"; }
        @Override public long size() { return 1; }
        @Override
        public Iterator<Example> stream() {
            return List.of(new Example("only", Map.of(), target)).iterator();
        }
    }
}

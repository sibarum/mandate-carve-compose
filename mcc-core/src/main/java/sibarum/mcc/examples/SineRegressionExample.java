package sibarum.mcc.examples;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.op.block.MlpBlock;
import sibarum.mcc.serialization.Exporter;
import sibarum.mcc.training.Corpus;
import sibarum.mcc.training.Example;
import sibarum.mcc.training.GraphTrainer;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * End-to-end example: build a 2-layer MLP, train it on synthetic
 * regression ({@code y = sin(x)}), export to {@code sine.mcc}.
 *
 * <p>Run with:
 * <pre>
 *   mvn exec:java -pl mcc-core -Dexec.mainClass=sibarum.mcc.examples.SineRegressionExample
 * </pre>
 *
 * <p>Reload the exported model with {@code mcc-runtime}'s
 * {@code Component.load(Path.of("sine.mcc"))}.
 */
public final class SineRegressionExample {

    public static void main(String[] args) throws IOException {
        // 1. Substrate: a single MLP block.
        MlpBlock mlp = new MlpBlock(new int[] { 1, 32, 32, 1 }, 7L);
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("mlp", mlp)
                .build();

        // 2. ComputationGraph with the MLP as both root-bearing and terminal.
        CompGraphNode node = new CompGraphNode("c-mlp", tg.node("mlp"));
        ComputationGraph cg = new ComputationGraph(List.of(node), node);

        // 3. Train against y = sin(x), x ∈ [-π, π].
        Corpus corpus = new SineCorpus(512, 0L);
        GraphTrainer trainer = new GraphTrainer(
                cg,
                (g, ex) -> g.bindRoot(node, 0, (MatrixValue) ex.inputs().get("x")),
                /* lr */ 0.02,
                /* stepEveryExample */ true
        );

        double loss0 = trainer.trainEpoch(corpus);
        System.out.printf("epoch  0   loss = %.6f%n", loss0);
        double loss = loss0;
        for (int e = 1; e <= 200; e++) {
            loss = trainer.trainEpoch(corpus);
            if (e % 20 == 0) System.out.printf("epoch %3d  loss = %.6f%n", e, loss);
        }
        System.out.printf("trained 200 epochs;  loss0 = %.4f → final = %.4f (%.2f%% reduction)%n",
                loss0, loss, 100.0 * (1.0 - loss / loss0));

        // 4. Export.
        Path outDir = Path.of("sine.mcc");
        new Exporter().export(cg,
                List.of(new Exporter.RootInput("x", node, 0)),
                outDir);
        System.out.println("exported → " + outDir.toAbsolutePath());

        // 5. Smoke-test inference at a few points (using the in-memory model;
        //    cross-module load lives in mcc-runtime's tests).
        double[] xs = { -Math.PI, -Math.PI / 2, 0.0, Math.PI / 2, Math.PI };
        for (double x : xs) {
            cg.bindRoot(node, 0, new MatrixValue(new double[] { x }));
            Value out = cg.execute();
            double yPred = ((MatrixValue) out).data()[0];
            double yTrue = Math.sin(x);
            System.out.printf("  x = %+5.2f   pred = %+.4f   true = %+.4f   err = %+.4f%n",
                    x, yPred, yTrue, yPred - yTrue);
        }
    }

    /** Uniform samples of y = sin(x) on [-π, π]. */
    private record SineCorpus(long size, long seed) implements Corpus {
        @Override public String name() { return "sin(x)"; }

        @Override
        public Iterator<Example> stream() {
            Random rng = new Random(seed);
            List<Example> all = new ArrayList<>((int) size);
            for (long i = 0; i < size; i++) {
                double x = rng.nextDouble() * 2 * Math.PI - Math.PI;
                MatrixValue xv = new MatrixValue(new double[] { x });
                MatrixValue yv = new MatrixValue(new double[] { Math.sin(x) });
                all.add(new Example("ex-" + i, Map.of("x", xv), yv));
            }
            return all.iterator();
        }
    }
}

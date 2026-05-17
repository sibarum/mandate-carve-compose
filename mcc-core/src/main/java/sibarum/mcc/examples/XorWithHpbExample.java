package sibarum.mcc.examples;

import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.op.HarmonicLift;
import sibarum.mcc.op.Linear;
import sibarum.mcc.op.advanced.SmoothedBasisElement.Kernel;
import sibarum.mcc.serialization.Exporter;
import sibarum.mcc.training.Corpus;
import sibarum.mcc.training.Example;
import sibarum.mcc.training.GraphTrainer;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * End-to-end example for {@link HarmonicLift}: train 1D-encoded XOR
 * with a δ-kernel basis lift feeding a {@link Linear} readout, export
 * the result. Demonstrates that the basis has a finite-norm
 * "exact-rational basin" under L2: the closed-form least-squares
 * optimum at {@code (w_tri, w_sq, b) = (0, -1/8, 1/2)} is reached by
 * SGD to float64 precision in a few thousand epochs.
 *
 * <p>The four XOR inputs are encoded as scalars
 * {@code x = (4·n + 1)/16} with {@code n = 2·b0 + b1 ∈ {0,1,2,3}},
 * placing inputs at {1/16, 5/16, 9/16, 13/16} — mid-piece for k=1 and
 * k=2, no breakpoint collisions. Labels are the standard XOR truth
 * table; this is a 1D-encoded version of XOR, not 2D over independent
 * inputs (the per-dim lift + linear readout cannot solve genuine 2D
 * XOR; that needs joint lifting or a hidden layer).
 *
 * <p>Run with:
 * <pre>
 *   mvn exec:java -pl mcc-core -Dexec.mainClass=sibarum.mcc.examples.XorWithHpbExample
 * </pre>
 */
public final class XorWithHpbExample {

    public static void main(String[] args) throws IOException {
        int K = 1;
        HarmonicLift lift = new HarmonicLift(K, /*inputDim*/ 1, Kernel.DELTA, 0.0);
        Linear readout = new Linear(1, 2 * K, /*withBias*/ true, 7L);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("lift", lift)
                .addNode("readout", readout)
                .build();

        CompGraphNode liftN = new CompGraphNode("c-lift", tg.node("lift"));
        CompGraphNode readN = new CompGraphNode("c-readout", tg.node("readout"));
        readN.wire(0, new SlotSource(liftN, tg.edge(tg.node("lift"), tg.node("readout"))));
        ComputationGraph cg = new ComputationGraph(List.of(liftN, readN), readN);

        Corpus corpus = new XorCorpus();
        GraphTrainer trainer = new GraphTrainer(
                cg,
                (g, ex) -> g.bindRoot(liftN, 0, (MatrixValue) ex.inputs().get("x")),
                /* lr */ 0.05,
                /* stepEveryExample */ true
        );

        double loss0 = trainer.trainEpoch(corpus);
        System.out.printf("epoch    0:  half-MSE = %.6e%n", loss0);
        double loss = loss0;
        int[] checkpoints = { 100, 500, 1000, 5000 };
        int last = 0;
        for (int target : checkpoints) {
            for (int e = last; e < target; e++) loss = trainer.trainEpoch(corpus);
            last = target;
            System.out.printf("epoch %4d:  half-MSE = %.6e%n", target, loss);
        }

        System.out.println();
        System.out.println("Predicted closed-form L2 optimum: (w_tri, w_sq, b) = (0, -1/8, 1/2)");
        System.out.printf("Trained readout weights:  w = [%+.10f, %+.10f]   b = [%+.10f]%n",
                readout.weights()[0][0], readout.weights()[0][1], readout.biases()[0]);

        double[] xs = { 1.0 / 16, 5.0 / 16, 9.0 / 16, 13.0 / 16 };
        double[] ys = { 0.0, 1.0, 1.0, 0.0 };
        System.out.println();
        System.out.println("Final predictions:");
        for (int i = 0; i < xs.length; i++) {
            cg.bindRoot(liftN, 0, new MatrixValue(new double[] { xs[i] }));
            Value y = cg.execute();
            double pred = ((MatrixValue) y).data()[0];
            System.out.printf("  x = %.4f   pred = %+.10f   target = %.0f   err = %+.2e%n",
                    xs[i], pred, ys[i], pred - ys[i]);
        }

        Path outDir = Path.of("xor-hpb.mcc");
        new Exporter().export(cg,
                List.of(new Exporter.RootInput("x", liftN, 0)),
                outDir);
        System.out.println();
        System.out.println("exported → " + outDir.toAbsolutePath());
    }

    private static final class XorCorpus implements Corpus {
        @Override public String name() { return "xor-1d-encoded"; }
        @Override public long size() { return 4; }

        @Override
        public Iterator<Example> stream() {
            double[] xs = { 1.0 / 16, 5.0 / 16, 9.0 / 16, 13.0 / 16 };
            double[] ys = { 0.0, 1.0, 1.0, 0.0 };
            List<Example> all = new java.util.ArrayList<>(4);
            for (int i = 0; i < 4; i++) {
                MatrixValue xv = new MatrixValue(new double[] { xs[i] });
                MatrixValue yv = new MatrixValue(new double[] { ys[i] });
                all.add(new Example("ex-" + i, Map.of("x", xv), yv));
            }
            return all.iterator();
        }
    }
}

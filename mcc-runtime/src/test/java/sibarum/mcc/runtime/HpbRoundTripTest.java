package sibarum.mcc.runtime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import sibarum.mcc.graph.CompGraphNode;
import sibarum.mcc.graph.ComputationGraph;
import sibarum.mcc.graph.SlotSource;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.graph.substrate.TransformationGraphBuilder;
import sibarum.mcc.op.HarmonicLift;
import sibarum.mcc.op.Linear;
import sibarum.mcc.op.advanced.SmoothedBasisElement.Kernel;
import sibarum.mcc.serialization.Exporter;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.Value;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Export → reload round-trip for a graph built from a {@link HarmonicLift}
 * (parameterless, config-only) feeding a trained {@link Linear} readout.
 *
 * <p>Confirms that:
 * <ol>
 *   <li>The BuiltinPrimitives registry reconstructs {@code HarmonicLift}
 *       from its serialized config (K, inputDim, kernel, widthFrac).</li>
 *   <li>Linear's trained parameters survive the round-trip.</li>
 *   <li>Inference parity holds at machine precision for every kernel
 *       choice (δ, box, tent).</li>
 * </ol>
 */
class HpbRoundTripTest {

    @TempDir
    Path tempDir;

    @Test
    void deltaKernelRoundTrips() throws IOException {
        runRoundTrip(Kernel.DELTA, 0.0);
    }

    @Test
    void boxKernelRoundTrips() throws IOException {
        runRoundTrip(Kernel.BOX, 0.125);
    }

    @Test
    void tentKernelRoundTrips() throws IOException {
        runRoundTrip(Kernel.TENT, 0.125);
    }

    private void runRoundTrip(Kernel kernel, double widthFrac) throws IOException {
        int K = 2;
        int inputDim = 1;
        int outDim = 1;
        HarmonicLift lift = new HarmonicLift(K, inputDim, kernel, widthFrac);
        Linear readout = new Linear(outDim, 2 * K, true, 13L);

        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("lift", lift)
                .addNode("readout", readout)
                .build();
        CompGraphNode liftN = new CompGraphNode("c-lift", tg.node("lift"));
        CompGraphNode readN = new CompGraphNode("c-readout", tg.node("readout"));
        readN.wire(0, new SlotSource(liftN, tg.edge(tg.node("lift"), tg.node("readout"))));
        ComputationGraph cg = new ComputationGraph(List.of(liftN, readN), readN);

        double[][] testInputs = { { 0.1 }, { 0.3 }, { 0.55 }, { 0.82 } };
        double[][] inMemoryOutputs = new double[testInputs.length][];
        for (int i = 0; i < testInputs.length; i++) {
            cg.bindRoot(liftN, 0, new MatrixValue(testInputs[i]));
            Value y = cg.execute();
            inMemoryOutputs[i] = ((MatrixValue) y).data().clone();
        }

        Path componentDir = tempDir.resolve("hpb-" + kernel.name() + ".mcc");
        new Exporter().export(cg,
                List.of(new Exporter.RootInput("x", liftN, 0)),
                componentDir);
        assertTrue(Files.exists(componentDir.resolve("graph.json")));
        assertTrue(Files.exists(componentDir.resolve("params.bin")));

        Component component = Component.load(componentDir);
        for (int i = 0; i < testInputs.length; i++) {
            MatrixValue x = new MatrixValue(testInputs[i]);
            MatrixValue y = (MatrixValue) component.infer(Map.of("x", x));
            assertArrayEquals(inMemoryOutputs[i], y.data(), 1e-12,
                    "round-trip mismatch at input " + i + " under kernel " + kernel);
        }
    }
}

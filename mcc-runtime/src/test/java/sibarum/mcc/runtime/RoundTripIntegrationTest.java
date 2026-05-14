package sibarum.mcc.runtime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 6 — MVP integration gate.
 *
 * <ol>
 *   <li>Build a 2-layer MLP wired via {@link ComputationGraph}.</li>
 *   <li>Train it with {@link GraphTrainer} on synthetic regression.</li>
 *   <li>Export to {@code component.mcc} via {@link Exporter}.</li>
 *   <li>Reload via {@link Component}, run inference on a held-out set,
 *       and assert outputs match the in-memory model at 1e-12.</li>
 *   <li>Mutate {@code params.bin} by one byte, reload, assert SHA-256
 *       verification fails.</li>
 * </ol>
 *
 * <p>The {@link Component#load} path uses the runtime's
 * {@link sibarum.mcc.primitive.BuiltinPrimitives#defaults} registry —
 * no training or carving code is touched on the inference path.
 */
class RoundTripIntegrationTest {

    @TempDir
    Path tempDir;

    @Test
    void trainExportReloadInferenceParityAndTamperDetection() throws IOException {
        // 1. Build the graph: a single MlpBlock.
        MlpBlock mlp = new MlpBlock(new int[] { 2, 16, 16, 2 }, 7L);
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("mlp", mlp)
                .build();
        CompGraphNode node = new CompGraphNode("c-mlp", tg.node("mlp"));
        ComputationGraph cg = new ComputationGraph(List.of(node), node);

        // 2. Train for 50 epochs.
        Corpus train = new SyntheticCorpus(256, 0L);
        GraphTrainer trainer = new GraphTrainer(
                cg,
                (g, ex) -> {
                    MatrixValue x = (MatrixValue) ex.inputs().get("x");
                    g.bindRoot(g.nodes().getFirst(), 0, x);
                },
                0.05,
                /* stepEveryExample */ true
        );
        double loss0 = trainer.trainEpoch(train);
        double lossLast = loss0;
        for (int i = 0; i < 50; i++) lossLast = trainer.trainEpoch(train);
        assertTrue(lossLast < loss0 * 0.1,
                "training did not converge enough: loss0=" + loss0 + " final=" + lossLast);

        // 3. Snapshot in-memory inference outputs on a held-out set.
        List<double[]> heldOutInputs = new ArrayList<>();
        List<double[]> inMemoryOutputs = new ArrayList<>();
        Random rng = new Random(99L);
        for (int i = 0; i < 32; i++) {
            double x0 = rng.nextDouble() * 2 - 1;
            double x1 = rng.nextDouble() * 2 - 1;
            heldOutInputs.add(new double[] { x0, x1 });
            cg.bindRoot(node, 0, new MatrixValue(new double[] { x0, x1 }));
            Value y = cg.execute();
            inMemoryOutputs.add(((MatrixValue) y).data());
        }

        // 4. Export.
        Path componentDir = tempDir.resolve("model.mcc");
        Exporter exporter = new Exporter();
        exporter.export(cg,
                List.of(new Exporter.RootInput("x", node, 0)),
                componentDir);

        assertTrue(Files.exists(componentDir.resolve("graph.json")));
        assertTrue(Files.exists(componentDir.resolve("params.bin")));

        // 5. Reload in the runtime and run inference.
        Component component = Component.load(componentDir);
        for (int i = 0; i < heldOutInputs.size(); i++) {
            MatrixValue x = new MatrixValue(heldOutInputs.get(i));
            MatrixValue y = (MatrixValue) component.infer(Map.of("x", x));
            assertArrayEquals(inMemoryOutputs.get(i), y.data(), 1e-12,
                    "round-trip output mismatch at index " + i);
        }

        // 6. Tamper test: flip one byte in the params blob, expect SHA-256 mismatch on reload.
        Path paramsPath = componentDir.resolve("params.bin");
        byte[] bytes = Files.readAllBytes(paramsPath);
        // Pick a byte in the body (after the 16-byte header) and flip a bit.
        bytes[24] ^= (byte) 0x01;
        Files.write(paramsPath, bytes);

        IOException ex = assertThrows(IOException.class, () -> Component.load(componentDir));
        assertTrue(ex.getMessage().contains("SHA-256 mismatch"),
                "expected SHA-256 mismatch error, got: " + ex.getMessage());
    }

    @Test
    void schemaVersionAndDefaultRegistryWork() throws IOException {
        // Sanity: a minimal export-then-load cycle for a stateless graph (just an Add).
        // Even the simplest case should round-trip cleanly.
        TransformationGraph tg = new TransformationGraphBuilder()
                .addNode("add", new sibarum.mcc.op.Add())
                .build();
        CompGraphNode n = new CompGraphNode("c-add", tg.node("add"));
        ComputationGraph cg = new ComputationGraph(List.of(n), n);

        // Bind so terminal can execute (we won't execute pre-export, but the export
        // doesn't need bindings — only inference does).
        Path dir = tempDir.resolve("trivial.mcc");
        new Exporter().export(cg, List.of(
                new Exporter.RootInput("a", n, 0),
                new Exporter.RootInput("b", n, 1)
        ), dir);

        Component c = Component.load(dir);
        Value out = c.infer(Map.of(
                "a", new MatrixValue(new double[] { 1, 2, 3 }),
                "b", new MatrixValue(new double[] { 10, 20, 30 })
        ));
        assertArrayEquals(new double[] { 11, 22, 33 }, ((MatrixValue) out).data(), 1e-12);
    }

    /** y = [x0·x1, x0+x1] on uniform x ∈ [-1, 1]². */
    private static final class SyntheticCorpus implements Corpus {
        private final long size;
        private final long seed;

        SyntheticCorpus(long size, long seed) {
            this.size = size;
            this.seed = seed;
        }

        @Override public String name() { return "synthetic-2x2"; }
        @Override public long size() { return size; }

        @Override
        public Iterator<Example> stream() {
            Random rng = new Random(seed);
            List<Example> all = new ArrayList<>((int) size);
            for (long i = 0; i < size; i++) {
                double x0 = rng.nextDouble() * 2 - 1;
                double x1 = rng.nextDouble() * 2 - 1;
                MatrixValue x = new MatrixValue(new double[] { x0, x1 });
                MatrixValue y = new MatrixValue(new double[] { x0 * x1, x0 + x1 });
                all.add(new Example("ex-" + i, Map.of("x", x), y));
            }
            return all.iterator();
        }
    }
}

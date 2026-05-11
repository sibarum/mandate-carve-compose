package sibarum.strnn.training;

import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.Trainable;
import sibarum.strnn.value.Value;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Drives a forward / backward / step loop on a carved ComputationGraph until
 * the supplied MandateSet verifies — or until the step budget is exhausted.
 *
 * The carver's {@link CarvingResult#simulatedValues()} map supplies each
 * Trainable's target: whatever value the carver simulated at the node
 * hosting the Trainable is what the Trainable must learn to produce. This is
 * the mandate-as-training-target idea generalised: the carver decomposes the
 * result mandate into per-node simulated values, and each Trainable's local
 * loss is "your output should equal that simulated value".
 *
 * Identity-dedup is via {@link Trainable#trainableIdentity}: two
 * CompGraphNodes that share the same underlying network (e.g. two
 * EmbedSymbol primitives over the same SymbolEmbeddingTable) get one
 * backward + step per loop iteration, not two.
 */
public final class InlineTrainer {
    private final CarvingResult carving;
    private final MandateSet mandates;
    private final double learningRate;
    private final int maxSteps;
    private final int checkEvery;

    public InlineTrainer(
            CarvingResult carving,
            MandateSet mandates,
            double learningRate,
            int maxSteps,
            int checkEvery) {
        this.carving = carving;
        this.mandates = mandates;
        this.learningRate = learningRate;
        this.maxSteps = maxSteps;
        this.checkEvery = checkEvery;
    }

    public Result run() {
        ComputationGraph cg = carving.graph();
        Map<CompGraphNode, Value> sim = carving.simulatedValues();
        List<TrainableSlot> slots = collectTrainables(cg, sim);

        cg.execute();
        VerificationReport initial = new MandateVerifier().verify(cg, mandates);
        List<StepRecord> trace = new ArrayList<>();
        trace.add(new StepRecord(0, cg.terminal().producedValue(), initial.allSatisfied()));
        if (initial.allSatisfied()) {
            return new Result(true, 0, trace, initial);
        }

        for (int step = 1; step <= maxSteps; step++) {
            cg.execute();
            // backward + step on each unique trainable (deduped by identity)
            IdentityHashMap<Object, Boolean> seen = new IdentityHashMap<>();
            for (TrainableSlot s : slots) {
                if (seen.putIfAbsent(s.trainable.trainableIdentity(), Boolean.TRUE) != null) continue;
                s.trainable.backward(s.target);
                s.trainable.step(learningRate);
            }

            boolean atCheckpoint = (step == 1) || (step % checkEvery == 0) || (step == maxSteps);
            if (atCheckpoint) {
                cg.execute();
                VerificationReport rep = new MandateVerifier().verify(cg, mandates);
                trace.add(new StepRecord(step, cg.terminal().producedValue(), rep.allSatisfied()));
                if (rep.allSatisfied()) {
                    return new Result(true, step, trace, rep);
                }
            }
        }

        cg.execute();
        VerificationReport finalRep = new MandateVerifier().verify(cg, mandates);
        return new Result(finalRep.allSatisfied(), maxSteps, trace, finalRep);
    }

    private static List<TrainableSlot> collectTrainables(
            ComputationGraph cg, Map<CompGraphNode, Value> sim) {
        List<TrainableSlot> out = new ArrayList<>();
        for (CompGraphNode n : cg.nodes()) {
            if (n.tNode().primitive() instanceof Trainable t) {
                Value target = sim.get(n);
                if (target == null) {
                    throw new IllegalStateException(
                            "no simulated value for trainable node " + n.id()
                                    + "; carver did not place a target");
                }
                out.add(new TrainableSlot(n, t, target));
            }
        }
        return out;
    }

    private record TrainableSlot(CompGraphNode node, Trainable trainable, Value target) {
    }

    public record StepRecord(int step, Value terminalValue, boolean mandatePass) {
    }

    public record Result(
            boolean converged,
            int stepsTaken,
            List<StepRecord> trace,
            VerificationReport finalReport) {

        /**
         * Per-session reward derived from convergence. Demos that want to
         * close the loop "successful carvings reinforce their edges" call
         * {@code applyEdgeFeedback(carving, result.score())} after training.
         * 1.0 for PASS, scaled-down for FAIL by how far from the budget the
         * run got.
         */
        public double score(int maxSteps) {
            if (converged) return 1.0;
            if (maxSteps <= 0) return 0.0;
            return Math.max(0.0, 1.0 - (double) stepsTaken / maxSteps) * 0.25;
        }
    }

    /**
     * Apply the trainer's reward to every edge the carver traced when
     * assembling the graph. Closes the loop: carvings that succeed reinforce
     * the edges they used; carvings that fail leave the substrate's edge
     * stats nearly unchanged (a small penalty proportional to how short of
     * convergence they came).
     */
    public static void applyEdgeFeedback(
            sibarum.strnn.carving.CarvingResult carving, double reward) {
        for (sibarum.strnn.transformation.TransformationEdge e : carving.tracedEdges()) {
            e.stats().update(reward);
        }
    }
}

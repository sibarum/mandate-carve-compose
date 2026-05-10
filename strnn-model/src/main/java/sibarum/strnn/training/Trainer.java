package sibarum.strnn.training;

import sibarum.strnn.carving.BackwardChainingCarver;
import sibarum.strnn.carving.CarvingResult;
import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.mandate.MandateVerifier;
import sibarum.strnn.mandate.VerificationReport;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.primitive.Trainable;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.value.Value;

import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Objects;
import java.util.Set;

/**
 * Orchestrates one training step per Example: carve, execute, verify, compute
 * terminal score, accumulate edge stats, run MLP backprop using simulated
 * values as supervision targets, step MLPs.
 *
 * The trainer assumes each Mlp instance is invoked at most once per Example
 * (true for v0 since each carved graph has one mlp_add and one mlp_mul). If
 * future demos invoke the same Mlp multiple times per execution, per-call
 * activation caching will be needed in Mlp.
 */
public final class Trainer {
    private final TransformationGraph tg;
    private final BackwardChainingCarver carver;
    private final MandateVerifier verifier = new MandateVerifier();
    private final double mlpLr;
    private final Pruner pruner;
    private final long pruneEvery;
    private long stepCount = 0;
    private int totalPruned = 0;
    private int lastPrunedCount = 0;
    private long lastPrunedAtStep = -1;

    public Trainer(TransformationGraph tg, long carverSeed, double mlpLr, Pruner pruner, long pruneEvery) {
        this(tg, carverSeed, mlpLr, pruner, pruneEvery, 0.0);
    }

    public Trainer(TransformationGraph tg, long carverSeed, double mlpLr, Pruner pruner, long pruneEvery, double explorationEpsilon) {
        this.tg = tg;
        this.carver = new BackwardChainingCarver(carverSeed, /*budget=*/200, explorationEpsilon);
        this.mlpLr = mlpLr;
        this.pruner = pruner;
        this.pruneEvery = pruneEvery;
    }

    public StepResult step(Example ex) {
        stepCount++;
        CarvingResult result = carver.carve(tg, ex.mandates(), ex.payload());
        if (result == null) {
            propagateFailure(ex);
            return StepResult.carveFailed(ex);
        }

        ComputationGraph cg = result.graph();
        Value executed;
        try {
            executed = cg.execute();
        } catch (RuntimeException re) {
            propagateScore(result, 0.0);
            return new StepResult(ex, result, null, 0.0, false, re.getMessage(), 0);
        }

        VerificationReport report = verifier.verify(cg, ex.mandates());
        double score = report.fractionSatisfied();

        // result is non-null at this point (early-returned above). Restate for the IDE.
        Objects.requireNonNull(result);
        runMlpBackpropAndStep(cg, result);
        propagateScore(result, score);

        int prunedThisStep = 0;
        if (pruner != null && pruneEvery > 0 && stepCount % pruneEvery == 0) {
            prunedThisStep = pruner.prune(tg);
            if (prunedThisStep > 0) {
                totalPruned += prunedThisStep;
                lastPrunedCount = prunedThisStep;
                lastPrunedAtStep = stepCount;
            }
        }

        return new StepResult(ex, result, executed, score, report.allSatisfied(), null, prunedThisStep);
    }

    public int totalPruned() {
        return totalPruned;
    }

    public int lastPrunedCount() {
        return lastPrunedCount;
    }

    public long lastPrunedAtStep() {
        return lastPrunedAtStep;
    }

    private void runMlpBackpropAndStep(ComputationGraph cg, CarvingResult result) {
        for (CompGraphNode n : cg.topoOrder()) {
            Primitive p = n.tNode().primitive();
            if (!(p instanceof Trainable t)) continue;
            if (t.trainableIdentity() == null) continue;
            Value target = result.simulatedValues().get(n);
            if (target == null) continue;
            t.backward(target);
        }
        IdentityHashMap<Object, Boolean> stepped = new IdentityHashMap<>();
        for (CompGraphNode n : cg.topoOrder()) {
            Primitive p = n.tNode().primitive();
            if (p instanceof Trainable t
                    && t.trainableIdentity() != null
                    && !stepped.containsKey(t.trainableIdentity())) {
                t.step(mlpLr);
                stepped.put(t.trainableIdentity(), Boolean.TRUE);
            }
        }
    }

    private void propagateScore(CarvingResult result, double score) {
        Set<TransformationEdge> seen = new HashSet<>();
        for (TransformationEdge e : result.tracedEdges()) {
            if (seen.add(e)) e.stats().update(score);
        }
    }

    private void propagateFailure(Example ex) {
        // No traced edges available since carving failed; nothing to update here.
    }

    public record StepResult(
            Example example,
            CarvingResult carving,
            Value executedResult,
            double score,
            boolean allSatisfied,
            String error,
            int prunedThisStep) {
        public static StepResult carveFailed(Example ex) {
            return new StepResult(ex, null, null, 0.0, false, "carve_failed", 0);
        }
    }
}

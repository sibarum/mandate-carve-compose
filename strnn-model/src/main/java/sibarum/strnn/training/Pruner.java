package sibarum.strnn.training;

import sibarum.strnn.primitive.LearnedArithmetic;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.transformation.TransformationNode;
import sibarum.strnn.value.ValueType;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Lightweight pruner per plan §5: when two TransformationEdges share the same
 * (srcType, dstType) pair, both have at least minSamples observations, and
 * one's mean score lags the other's by at least margin, prune the worse.
 *
 * Operates on already-collected EdgeStats — no extra bookkeeping during the
 * carving / training loop. Designed to be called periodically.
 */
public final class Pruner {
    private final long minSamples;
    private final double margin;

    public Pruner(long minSamples, double margin) {
        this.minSamples = minSamples;
        this.margin = margin;
    }

    public int prune(TransformationGraph tg) {
        Map<TypePair, List<TransformationEdge>> byPair = new HashMap<>();
        for (TransformationEdge e : tg.edges()) {
            if (e.stats().isPruned()) continue;
            TypePair p = new TypePair(e.from().outputType(),
                    e.to().inputTypes().getFirst()); // representative input type
            // Better: group by (srcType, every input type the dst accepts).
            // For v0 we just use the source-type and a digest of dst inputs.
            byPair.computeIfAbsent(p, k -> new java.util.ArrayList<>()).add(e);
        }

        int pruned = 0;
        for (List<TransformationEdge> group : byPair.values()) {
            if (group.size() < 2) continue;
            TransformationEdge best = null;
            for (TransformationEdge e : group) {
                if (e.stats().samples() < minSamples) continue;
                if (best == null || e.stats().score() > best.stats().score()) best = e;
            }
            if (best == null) continue;
            for (TransformationEdge e : group) {
                if (e == best || e.stats().isPruned()) continue;
                if (e.stats().samples() < minSamples) continue;
                if (best.stats().score() - e.stats().score() >= margin) {
                    e.stats().prune();
                    pruned++;
                }
            }
        }
        return pruned;
    }

    /**
     * Pattern B: primitive-level competition. When multiple LearnedArithmetic
     * primitives share a role (e.g. one MlpPrimitive(MUL) and one
     * TransformerPrimitive(MUL) both fill the MUL slot), aggregate each
     * primitive's average outgoing-edge score across all its outgoing edges
     * with at least minSamples each. If the best primitive's aggregate beats
     * any competitor's by at least margin, prune all of the loser's outgoing
     * edges. Returns the number of edges pruned.
     *
     * Decoupled from {@link #prune(TransformationGraph)} because the grouping
     * criterion is different: edge-pair grouping looks at structural slots,
     * primitive-competition grouping looks at functional alternatives at the
     * same role.
     */
    public int prunePrimitiveCompetition(TransformationGraph tg) {
        Map<MlpRole, List<TransformationNode>> byRole = new EnumMap<>(MlpRole.class);
        for (TransformationNode n : tg.nodes()) {
            if (n.primitive() instanceof LearnedArithmetic la) {
                byRole.computeIfAbsent(la.role(), k -> new ArrayList<>()).add(n);
            }
        }

        int pruned = 0;
        for (Map.Entry<MlpRole, List<TransformationNode>> entry : byRole.entrySet()) {
            List<TransformationNode> competitors = entry.getValue();
            if (competitors.size() < 2) continue;

            Map<TransformationNode, Double> avgScore = new HashMap<>();
            Map<TransformationNode, Long> totalSamples = new HashMap<>();
            for (TransformationNode tn : competitors) {
                double sum = 0;
                long n = 0;
                long samples = 0;
                for (TransformationEdge e : tg.outgoing(tn)) {
                    if (e.stats().isPruned()) continue;
                    if (e.stats().samples() < minSamples) continue;
                    sum += e.stats().score();
                    samples += e.stats().samples();
                    n++;
                }
                if (n == 0) continue;
                avgScore.put(tn, sum / n);
                totalSamples.put(tn, samples);
            }
            if (avgScore.size() < 2) continue;

            TransformationNode best = null;
            for (TransformationNode tn : avgScore.keySet()) {
                if (best == null || avgScore.get(tn) > avgScore.get(best)) best = tn;
            }
            for (TransformationNode tn : competitors) {
                if (tn == best) continue;
                Double s = avgScore.get(tn);
                if (s == null) continue;
                if (avgScore.get(best) - s >= margin) {
                    for (TransformationEdge e : tg.outgoing(tn)) {
                        if (!e.stats().isPruned()) {
                            e.stats().prune();
                            pruned++;
                        }
                    }
                    for (TransformationEdge e : tg.incoming(tn)) {
                        if (!e.stats().isPruned()) {
                            e.stats().prune();
                            pruned++;
                        }
                    }
                }
            }
        }
        return pruned;
    }

    private record TypePair(ValueType from, ValueType to) {
    }
}

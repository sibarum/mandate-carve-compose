package sibarum.strnn.training;

import sibarum.strnn.transformation.TransformationEdge;
import sibarum.strnn.transformation.TransformationGraph;
import sibarum.strnn.value.ValueType;

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

    private record TypePair(ValueType from, ValueType to) {
    }
}

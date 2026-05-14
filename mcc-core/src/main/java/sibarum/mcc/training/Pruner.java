package sibarum.mcc.training;

import sibarum.mcc.graph.substrate.TransformationEdge;
import sibarum.mcc.graph.substrate.TransformationGraph;
import sibarum.mcc.value.ValueType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Edge-pair pruner: when two {@link TransformationEdge}s share the
 * same {@code (srcType, dstType)} pair, both have at least
 * {@code minSamples} observations, and one's mean score lags the
 * other's by at least {@code margin}, prune the worse.
 *
 * <p>Operates on already-collected {@code EdgeStats} — no extra
 * bookkeeping during the training loop. Designed to be called
 * periodically.
 *
 * <p>The strnn-model also had a primitive-role-aware pruner; that
 * relied on an arithmetic-task-specific role marker and isn't ported.
 * A generic role-aware variant can be added when the GUI introduces
 * primitive roles.
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
            // Representative input type at the destination — group by
            // (src output, first dst input type). A more refined grouping
            // could enumerate every type-compatible dst slot.
            TypePair p = new TypePair(e.from().outputType(),
                    e.to().inputTypes().getFirst());
            byPair.computeIfAbsent(p, k -> new ArrayList<>()).add(e);
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

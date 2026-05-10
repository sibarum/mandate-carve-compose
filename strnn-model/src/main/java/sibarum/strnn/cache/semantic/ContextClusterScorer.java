package sibarum.strnn.cache.semantic;

import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Scoring primitive: within-context cosine similarity minus between-context
 * cosine similarity, restricted to the rhs descriptor atoms (the ones that
 * name the semantic axes themselves: {@code direction}, {@code physical},
 * {@code intelligence}, etc.).
 *
 * Dichotomy anchors {@code (A | B)} are intentionally excluded — they're
 * forced near {@code cos = −1} by the dichotomy-push objective and would
 * bias within-context similarity downward. The structural claim being tested
 * here is "rhs atoms appearing together cluster," not "dichotomy halves
 * cluster with their context."
 *
 * Within: average cosine over every pair of rhs atoms that appear in at
 *   least one common relation.
 * Between: average cosine over every pair of rhs atoms that share zero
 *   relations.
 * Score: within − between. Positive means the embedding space carries the
 * context structure.
 */
public final class ContextClusterScorer implements Primitive {
    private final SymbolEmbeddingTable table;
    private final List<SemRelation> relations;

    public ContextClusterScorer(SymbolEmbeddingTable table, List<SemRelation> relations) {
        this.table = table;
        this.relations = relations;
    }

    @Override
    public String name() {
        return "context-cluster-margin";
    }

    @Override
    public List<ValueType> inputTypes() {
        return List.of(ValueType.NUMBER);
    }

    @Override
    public ValueType outputType() {
        return ValueType.NUMBER;
    }

    @Override
    public Value apply(List<Value> inputs) {
        // Map atom -> set of relation indices in whose rhs it appears.
        Map<String, Set<Integer>> atomToRelations = new HashMap<>();
        for (int i = 0; i < relations.size(); i++) {
            for (String atom : SemanticTrainer.collectRhsAtoms(relations.get(i).rhs())) {
                atomToRelations.computeIfAbsent(atom, k -> new HashSet<>()).add(i);
            }
        }
        String[] atoms = atomToRelations.keySet().toArray(new String[0]);

        double withinSum = 0.0;
        int withinCount = 0;
        double betweenSum = 0.0;
        int betweenCount = 0;
        for (int i = 0; i < atoms.length; i++) {
            Set<Integer> ri = atomToRelations.get(atoms[i]);
            double[] ei = table.embed(atoms[i]);
            for (int j = i + 1; j < atoms.length; j++) {
                Set<Integer> rj = atomToRelations.get(atoms[j]);
                double c = cosine(ei, table.embed(atoms[j]));
                if (intersects(ri, rj)) {
                    withinSum += c;
                    withinCount++;
                } else {
                    betweenSum += c;
                    betweenCount++;
                }
            }
        }

        double within = withinCount == 0 ? 0.0 : withinSum / withinCount;
        double between = betweenCount == 0 ? 0.0 : betweenSum / betweenCount;
        return new NumberValue(within - between);
    }

    private static boolean intersects(Set<Integer> a, Set<Integer> b) {
        Set<Integer> small = a.size() <= b.size() ? a : b;
        Set<Integer> big = small == a ? b : a;
        for (Integer x : small) if (big.contains(x)) return true;
        return false;
    }

    private static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0.0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}

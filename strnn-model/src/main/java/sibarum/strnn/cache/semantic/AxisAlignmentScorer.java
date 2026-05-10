package sibarum.strnn.cache.semantic;

import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Scoring primitive: how well do dichotomies sharing a context atom produce
 * parallel "axis vectors"?
 *
 * For each rhs atom X that appears in two or more relations, compute the
 * axis vector {@code embed(left) − embed(right)} for every relation
 * containing X, then take the average pairwise {@code |cos|} of those axes.
 * High {@code |cos|} means the axes are parallel (or anti-parallel) — the
 * dichotomies share the same direction in embedding space.
 *
 * The final score is the mean of those per-context averages. Direction-of-axis
 * is intentionally ignored ({@code |cos|}, not {@code cos}) because the
 * file's syntax doesn't fix which atom in a dichotomy is "positive": (up | down)
 * and (down | up) are the same axis, sign-flipped. Aligned in either sign is
 * the structural property we care about.
 */
public final class AxisAlignmentScorer implements Primitive {
    private final SymbolEmbeddingTable table;
    private final List<SemRelation> relations;

    public AxisAlignmentScorer(SymbolEmbeddingTable table, List<SemRelation> relations) {
        this.table = table;
        this.relations = relations;
    }

    @Override
    public String name() {
        return "axis-alignment";
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
        Map<String, List<int[]>> byContext = groupByContext(relations);

        double sum = 0.0;
        int contextsCounted = 0;
        for (Map.Entry<String, List<int[]>> e : byContext.entrySet()) {
            List<int[]> idxs = e.getValue();
            if (idxs.size() < 2) continue;

            double[][] axes = new double[idxs.size()][];
            for (int i = 0; i < idxs.size(); i++) {
                SemRelation r = relations.get(idxs.get(i)[0]);
                double[] eL = table.embed(r.lhs().left());
                double[] eR = table.embed(r.lhs().right());
                axes[i] = sub(eL, eR);
            }

            double pairwise = 0.0;
            int pairs = 0;
            for (int i = 0; i < axes.length; i++) {
                for (int j = i + 1; j < axes.length; j++) {
                    pairwise += Math.abs(cosine(axes[i], axes[j]));
                    pairs++;
                }
            }
            if (pairs > 0) {
                sum += pairwise / pairs;
                contextsCounted++;
            }
        }

        return new NumberValue(contextsCounted == 0 ? 0.0 : sum / contextsCounted);
    }

    private static Map<String, List<int[]>> groupByContext(List<SemRelation> relations) {
        Map<String, List<int[]>> out = new HashMap<>();
        for (int i = 0; i < relations.size(); i++) {
            Set<String> rhsAtoms = SemanticTrainer.collectRhsAtoms(relations.get(i).rhs());
            for (String atom : rhsAtoms) {
                out.computeIfAbsent(atom, k -> new ArrayList<>()).add(new int[]{i});
            }
        }
        return out;
    }

    private static double[] sub(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] - b[i];
        return r;
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

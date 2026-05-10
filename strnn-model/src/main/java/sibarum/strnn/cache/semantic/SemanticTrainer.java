package sibarum.strnn.cache.semantic;

import sibarum.strnn.cache.SymbolEmbeddingTable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Multi-objective trainer for the symbol embedding table given a parsed
 * semantic ontology.
 *
 * Three objectives, each applied per epoch:
 *
 *   1) Dichotomy push — for each {@code (A | B)}, pull A toward {@code −B}
 *      and vice versa. Drives {@code cos(A, B) → −1}.
 *
 *   2) Context pull — for each atom X appearing anywhere in the relation's
 *      rhs, pull both A and B toward X (and X toward both). Builds the
 *      neighborhood structure.
 *
 *   3) Axis alignment (optional) — for each rhs atom appearing in 2+
 *      dichotomies, pull those dichotomies' axis vectors toward parallel
 *      (sign-aware: dichotomies' direction is undirected, so anti-parallel
 *      counts as aligned). Endpoints move symmetrically so the dichotomy's
 *      centroid is preserved while the axis rotates.
 *
 * Push and pull use the same gradient pattern as
 * {@code EmbedSymbol.backward(target)}: target {@code v} gives
 * {@code grad = u − v}, and target {@code −v} gives {@code grad = u + v}.
 * Updating with {@code emb -= lr * grad} therefore moves toward the target.
 *
 * Set {@code axisLr = 0} to skip axis alignment — used by the negative-result
 * arm of {@code SemanticEmbeddingDemo}, which leaves alignment unmandated to
 * surface the framework's diagnostic FAIL on a structural property the
 * trainer doesn't optimize.
 */
public final class SemanticTrainer {

    private SemanticTrainer() {
    }

    public static void train(SymbolEmbeddingTable table,
                             List<SemRelation> relations,
                             double dichotomyLr,
                             double contextLr,
                             int epochs) {
        train(table, relations, dichotomyLr, contextLr, /*axisLr=*/0.0, epochs);
    }

    public static void train(SymbolEmbeddingTable table,
                             List<SemRelation> relations,
                             double dichotomyLr,
                             double contextLr,
                             double axisLr,
                             int epochs) {
        for (SemRelation r : relations) {
            table.embed(r.lhs().left());
            table.embed(r.lhs().right());
            for (String atom : collectRhsAtoms(r.rhs())) {
                table.embed(atom);
            }
        }

        int dim = table.dim();
        double[] tmp = new double[dim];

        // Pre-compute shared-context groups once; topology doesn't change during training.
        Map<String, List<Integer>> sharedContext = axisLr > 0.0
                ? buildSharedContextGroups(relations)
                : null;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // 1) Dichotomy push  +  2) Context pull
            for (SemRelation r : relations) {
                String leftS = r.lhs().left();
                String rightS = r.lhs().right();

                double[] eL = table.embed(leftS);
                double[] eR = table.embed(rightS);
                add(eL, eR, tmp);
                table.update(leftS, tmp, dichotomyLr);
                add(eR, eL, tmp);
                table.update(rightS, tmp, dichotomyLr);

                for (String atom : collectRhsAtoms(r.rhs())) {
                    if (atom.equals(leftS) || atom.equals(rightS)) continue;
                    double[] eX = table.embed(atom);
                    sub(table.embed(leftS), eX, tmp);
                    table.update(leftS, tmp, contextLr);
                    sub(eX, table.embed(leftS), tmp);
                    table.update(atom, tmp, contextLr);
                    sub(table.embed(rightS), eX, tmp);
                    table.update(rightS, tmp, contextLr);
                    sub(eX, table.embed(rightS), tmp);
                    table.update(atom, tmp, contextLr);
                }
            }

            // 3) Axis alignment (optional)
            if (axisLr > 0.0) {
                applyAxisAlignment(table, relations, sharedContext, axisLr, dim);
            }
        }
    }

    private static void applyAxisAlignment(SymbolEmbeddingTable table,
                                           List<SemRelation> relations,
                                           Map<String, List<Integer>> sharedContext,
                                           double axisLr,
                                           int dim) {
        double[] axisI = new double[dim];
        double[] axisJ = new double[dim];
        double[] deltaI = new double[dim];
        double[] deltaJ = new double[dim];
        double[] gradHalf = new double[dim];

        for (Map.Entry<String, List<Integer>> entry : sharedContext.entrySet()) {
            List<Integer> group = entry.getValue();
            if (group.size() < 2) continue;

            for (int a = 0; a < group.size(); a++) {
                SemRelation ri = relations.get(group.get(a));
                String li = ri.lhs().left();
                String rri = ri.lhs().right();
                computeAxis(table, li, rri, axisI);

                for (int b = a + 1; b < group.size(); b++) {
                    SemRelation rj = relations.get(group.get(b));
                    String lj = rj.lhs().left();
                    String rrj = rj.lhs().right();
                    computeAxis(table, lj, rrj, axisJ);

                    double dotIJ = dot(axisI, axisJ);
                    double s = dotIJ >= 0.0 ? 1.0 : -1.0;

                    // delta_i = s * axis_j - axis_i (target axis_i toward s * axis_j)
                    for (int k = 0; k < dim; k++) {
                        deltaI[k] = s * axisJ[k] - axisI[k];
                        deltaJ[k] = s * axisI[k] - axisJ[k];
                    }

                    // To shift left_i by +delta_i/2 with table.update (which subtracts lr*grad),
                    // pass grad = -delta_i/2.
                    halfNeg(deltaI, gradHalf);
                    table.update(li, gradHalf, axisLr);
                    halfPos(deltaI, gradHalf);
                    table.update(rri, gradHalf, axisLr);

                    halfNeg(deltaJ, gradHalf);
                    table.update(lj, gradHalf, axisLr);
                    halfPos(deltaJ, gradHalf);
                    table.update(rrj, gradHalf, axisLr);

                    // Refresh axis_i since left_i / right_i just moved.
                    computeAxis(table, li, rri, axisI);
                }
            }
        }
    }

    private static Map<String, List<Integer>> buildSharedContextGroups(List<SemRelation> relations) {
        Map<String, List<Integer>> out = new HashMap<>();
        for (int i = 0; i < relations.size(); i++) {
            for (String atom : collectRhsAtoms(relations.get(i).rhs())) {
                out.computeIfAbsent(atom, k -> new ArrayList<>()).add(i);
            }
        }
        return out;
    }

    private static void computeAxis(SymbolEmbeddingTable table, String left, String right, double[] out) {
        double[] eL = table.embed(left);
        double[] eR = table.embed(right);
        for (int i = 0; i < out.length; i++) out[i] = eL[i] - eR[i];
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    private static void halfNeg(double[] in, double[] out) {
        for (int i = 0; i < in.length; i++) out[i] = -in[i] * 0.5;
    }

    private static void halfPos(double[] in, double[] out) {
        for (int i = 0; i < in.length; i++) out[i] = in[i] * 0.5;
    }

    public static Set<String> collectRhsAtoms(SemExpr e) {
        Set<String> out = new HashSet<>();
        collect(e, out);
        return out;
    }

    private static void collect(SemExpr e, Set<String> out) {
        switch (e) {
            case SemExpr.Atom a -> out.add(a.name());
            case SemExpr.Qualified q -> {
                collect(q.head(), out);
                collect(q.facet(), out);
            }
            case SemExpr.Composition c -> {
                collect(c.left(), out);
                collect(c.right(), out);
            }
            case SemExpr.Conjunction c -> {
                collect(c.left(), out);
                collect(c.right(), out);
            }
            case SemExpr.Union u -> {
                for (SemExpr m : u.members()) collect(m, out);
            }
        }
    }

    private static void add(double[] a, double[] b, double[] out) {
        for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    }

    private static void sub(double[] a, double[] b, double[] out) {
        for (int i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    }
}

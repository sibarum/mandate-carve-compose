package sibarum.strnn.cache.semantic;

import sibarum.strnn.cache.SymbolEmbeddingTable;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Multi-objective trainer for the symbol embedding table given a parsed
 * semantic ontology.
 *
 * Two objectives, each applied to every relation per epoch:
 *
 *   1) Dichotomy push — for each {@code (A | B)}, pull A toward {@code −B}
 *      and vice versa. Drives {@code cos(A, B) → −1}.
 *
 *   2) Context pull — for each atom X appearing anywhere in the relation's
 *      rhs, pull both A and B toward X (and X toward both). Builds the
 *      neighborhood structure: dichotomies sharing a context atom become
 *      neighbors, with their "axis" running through the shared neighborhood.
 *
 * Push and pull use the same gradient pattern as
 * {@code EmbedSymbol.backward(target)}: target {@code v} gives
 * {@code grad = u − v}, and target {@code −v} gives {@code grad = u + v}.
 * Updating with {@code emb -= lr * grad} therefore moves toward the target.
 */
public final class SemanticTrainer {

    private SemanticTrainer() {
    }

    public static void train(SymbolEmbeddingTable table,
                             List<SemRelation> relations,
                             double dichotomyLr,
                             double contextLr,
                             int epochs) {
        // Force lazy init of every atom up front so initial state doesn't
        // depend on training order.
        for (SemRelation r : relations) {
            table.embed(r.lhs().left());
            table.embed(r.lhs().right());
            for (String atom : collectRhsAtoms(r.rhs())) {
                table.embed(atom);
            }
        }

        int dim = table.dim();
        double[] tmp = new double[dim];

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (SemRelation r : relations) {
                String leftS = r.lhs().left();
                String rightS = r.lhs().right();

                // 1) Dichotomy push
                double[] eL = table.embed(leftS);
                double[] eR = table.embed(rightS);
                add(eL, eR, tmp);
                table.update(leftS, tmp, dichotomyLr);
                add(eR, eL, tmp);
                table.update(rightS, tmp, dichotomyLr);

                // 2) Context pull
                for (String atom : collectRhsAtoms(r.rhs())) {
                    if (atom.equals(leftS) || atom.equals(rightS)) continue;
                    double[] eX = table.embed(atom);
                    // Pull L toward X
                    sub(table.embed(leftS), eX, tmp);
                    table.update(leftS, tmp, contextLr);
                    sub(eX, table.embed(leftS), tmp);
                    table.update(atom, tmp, contextLr);
                    // Pull R toward X
                    sub(table.embed(rightS), eX, tmp);
                    table.update(rightS, tmp, contextLr);
                    sub(eX, table.embed(rightS), tmp);
                    table.update(atom, tmp, contextLr);
                }
            }
        }
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

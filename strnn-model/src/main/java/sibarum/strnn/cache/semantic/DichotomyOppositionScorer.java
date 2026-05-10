package sibarum.strnn.cache.semantic;

import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueType;

import java.util.List;

/**
 * Scoring primitive: averages {@code cos(embed(left), embed(right))} across
 * every dichotomy in the relation list. After training, this should approach
 * {@code −1} (opposites). Mandate verifier matches this against the target
 * via {@link sibarum.strnn.value.ValueDistance}.
 *
 * Holds the table and relations by reference so it can re-score whenever
 * {@code apply} runs — the score reflects the table's current state. Takes a
 * NumberValue trigger as its formal input so the primitive fits the standard
 * primitive contract; the trigger's value is ignored.
 */
public final class DichotomyOppositionScorer implements Primitive {
    private final SymbolEmbeddingTable table;
    private final List<SemRelation> relations;

    public DichotomyOppositionScorer(SymbolEmbeddingTable table, List<SemRelation> relations) {
        this.table = table;
        this.relations = relations;
    }

    @Override
    public String name() {
        return "dichotomy-opposition";
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
        double sum = 0.0;
        int n = 0;
        for (SemRelation r : relations) {
            double[] a = table.embed(r.lhs().left());
            double[] b = table.embed(r.lhs().right());
            sum += cosine(a, b);
            n++;
        }
        return new NumberValue(n == 0 ? 0.0 : sum / n);
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

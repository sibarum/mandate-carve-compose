package sibarum.elden.demo;

import sibarum.elden.annotation.EntityType;
import sibarum.elden.data.TypedSpanExample;
import sibarum.elden.pos.PosTagset;
import sibarum.elden.pos.PosTrainer.TrainedPosLayer;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/**
 * Iter 23: typed-entity classifier using the framework's own symbol substrate
 * for output labels.
 *
 * Where {@link EntityTypeClassifier} encoded the 7 EntityType labels as
 * one-hot positions (an alien encoding pasted onto the side of the network),
 * this version registers each EntityType name as a key in a
 * {@link SymbolEmbeddingTable} of its own — a "mini-KV" for the
 * type-classification context. The MLP now outputs a 32-dim vector (not
 * 7-dim logits); training drives that vector toward the gold type's anchor,
 * and the anchor is itself updated toward the prediction so the two
 * co-adapt. Inference is "cosine-similarity-nearest over the 7 anchors."
 *
 * Both the MLP weights AND the anchor vectors are trainable. Anchors that
 * receive training signal drift to become learned class centroids in the
 * 32-dim space; anchors for types with no training data (FACTION, ERA in
 * our corpus) stay near their random initialization.
 */
public final class VectorEntityTypeClassifier {

    private final TrainedPosLayer pos;
    private final ClassifierHead head;
    private final SymbolEmbeddingTable typeAnchors;
    private final int contextDim;
    private final int posDim;
    private final int featureDim;
    private final int embedDim;

    public VectorEntityTypeClassifier(TrainedPosLayer pos, int hiddenDim, int embedDim, long seed) {
        this.pos = pos;
        this.contextDim = pos.contextDim;
        this.posDim = PosTagset.size();
        this.featureDim = contextDim + posDim;
        this.embedDim = embedDim;

        Mlp mlp = new Mlp(new int[]{featureDim, hiddenDim, embedDim}, seed);
        this.head = new ClassifierHead("entity-type-vector-head", mlp);

        // The type-context mini-KV. Different seed from the token KV so the
        // initial anchor distribution is independent of the token distribution.
        this.typeAnchors = new SymbolEmbeddingTable(embedDim, seed + 1);
        for (EntityType t : EntityType.values()) {
            typeAnchors.embed(t.name());
        }
    }

    /** Average of per-token (context + POS-logits) across the span. */
    public double[] spanFeatures(List<String> tokens, int spanStart, int spanEnd) {
        double[] sum = new double[featureDim];
        int n = spanEnd - spanStart + 1;
        for (int i = spanStart; i <= spanEnd; i++) {
            double[] ctx = pos.contextOf(tokens, i);
            double[] posLogits = ((MatrixValue) pos.classifier.apply(
                    List.of(new MatrixValue(ctx)))).data();
            for (int j = 0; j < contextDim; j++) sum[j] += ctx[j];
            for (int j = 0; j < posDim; j++) sum[contextDim + j] += posLogits[j];
        }
        for (int j = 0; j < featureDim; j++) sum[j] /= n;
        return sum;
    }

    /**
     * Train one phase. Both MLP weights and type anchors update each step.
     * For a gold type T with anchor `t` and predicted vector `p`:
     *   - MLP gradient pulls `p` toward `t`        (via head.backward(t))
     *   - anchor gradient pulls `t` toward `p`     (via typeAnchors.update)
     * The two co-adapt: anchors become learned centroids of their class.
     */
    public void train(List<TypedSpanExample> examples, int epochs, double lr,
                       Random rng, String label) {
        if (examples.isEmpty()) {
            System.out.println("  [" + label + "] empty training set, skipping");
            return;
        }

        // Pre-compute span features once (they don't change between epochs;
        // the POS layer is frozen during type-head training).
        double[][] features = new double[examples.size()][];
        String[] typeNames = new String[examples.size()];
        for (int i = 0; i < examples.size(); i++) {
            TypedSpanExample ex = examples.get(i);
            features[i] = spanFeatures(ex.tokens(), ex.spanStart(), ex.spanEnd());
            typeNames[i] = ex.type().name();
        }

        List<Integer> order = new ArrayList<>(examples.size());
        for (int i = 0; i < examples.size(); i++) order.add(i);

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(order, rng);
            double totalLoss = 0;
            int correct = 0;
            for (int k : order) {
                // Read the current anchor (may have moved since last epoch).
                double[] anchor = typeAnchors.embed(typeNames[k]).clone();

                // Forward: MLP produces a 32-dim vector.
                double[] predicted = ((MatrixValue) head.apply(
                        List.of(new MatrixValue(features[k])))).data();

                // Loss: MSE(predicted, anchor).
                double exampleLoss = 0;
                for (int i = 0; i < embedDim; i++) {
                    double d = predicted[i] - anchor[i];
                    exampleLoss += d * d;
                }
                totalLoss += exampleLoss;

                // Inference check: is the nearest anchor to `predicted` the gold one?
                Optional<String> best = typeAnchors.nearest(predicted);
                if (best.isPresent() && best.get().equals(typeNames[k])) correct++;

                // MLP backward: target is the anchor; gradients pull predicted -> anchor.
                head.backward(new MatrixValue(anchor));
                head.step(lr);

                // Anchor update: gradient is dL/d(anchor) = 2*(anchor - predicted),
                // SymbolEmbeddingTable.update subtracts lr * gradient, so anchor moves
                // by `+2*lr*(predicted - anchor)` per step. Anchor drifts toward predicted.
                double[] anchorGrad = new double[embedDim];
                for (int i = 0; i < embedDim; i++) {
                    anchorGrad[i] = 2.0 * (anchor[i] - predicted[i]);
                }
                typeAnchors.update(typeNames[k], anchorGrad, lr);
            }
            if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 5 == 0) {
                System.out.printf("  [%s] epoch %2d  loss=%.4f  accuracy=%.3f%n",
                        label, epoch + 1, totalLoss / examples.size(),
                        (double) correct / examples.size());
            }
        }
    }

    /** Predict the {@link EntityType} for one span via nearest-anchor cosine. */
    public EntityType predict(List<String> tokens, int spanStart, int spanEnd) {
        double[] predicted = predictVector(tokens, spanStart, spanEnd);
        Optional<String> best = typeAnchors.nearest(predicted);
        return best.map(EntityType::valueOf).orElse(EntityType.values()[0]);
    }

    /** Raw 32-dim prediction (the point in type-space). */
    public double[] predictVector(List<String> tokens, int spanStart, int spanEnd) {
        double[] feat = spanFeatures(tokens, spanStart, spanEnd);
        return ((MatrixValue) head.apply(List.of(new MatrixValue(feat)))).data();
    }

    /** Top-k nearest anchors by cosine similarity, for confidence display. */
    public List<TypeScore> topK(List<String> tokens, int spanStart, int spanEnd, int k) {
        double[] predicted = predictVector(tokens, spanStart, spanEnd);
        List<TypeScore> scores = new ArrayList<>();
        for (EntityType t : EntityType.values()) {
            double[] anchor = typeAnchors.embed(t.name());
            scores.add(new TypeScore(t, cosineSim(predicted, anchor)));
        }
        scores.sort((a, b) -> Double.compare(b.score(), a.score()));
        return scores.subList(0, Math.min(k, scores.size()));
    }

    /** Top-1 accuracy on a held-out set; returns (correct, total). */
    public int[] evaluate(List<TypedSpanExample> examples) {
        int correct = 0;
        for (TypedSpanExample ex : examples) {
            EntityType p = predict(ex.tokens(), ex.spanStart(), ex.spanEnd());
            if (p == ex.type()) correct++;
        }
        return new int[]{correct, examples.size()};
    }

    /** Per-type P/R/F1 from a confusion matrix. */
    public String evaluatePerTypeReport(List<TypedSpanExample> examples) {
        int n = EntityType.values().length;
        int[][] cm = new int[n][n];
        for (TypedSpanExample ex : examples) {
            int gold = ex.type().ordinal();
            EntityType pred = predict(ex.tokens(), ex.spanStart(), ex.spanEnd());
            cm[gold][pred.ordinal()]++;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("    %-10s %6s %6s %6s %8s%n", "type", "P", "R", "F1", "support"));
        for (int g = 0; g < n; g++) {
            int tp = cm[g][g];
            int row = 0, col = 0;
            for (int j = 0; j < n; j++) { row += cm[g][j]; col += cm[j][g]; }
            if (row == 0 && col == 0) continue;
            double p = col > 0 ? (double) tp / col : 0;
            double r = row > 0 ? (double) tp / row : 0;
            double f1 = (p + r) > 0 ? 2 * p * r / (p + r) : 0;
            sb.append(String.format("    %-10s %6.3f %6.3f %6.3f %8d%n",
                    EntityType.values()[g], p, r, f1, row));
        }
        return sb.toString();
    }

    /** Inspection: pairwise cosine sim of the 7 anchors + their magnitudes. */
    public String anchorGeometryReport() {
        EntityType[] types = EntityType.values();
        int n = types.length;
        double[][] anchors = new double[n][];
        for (int i = 0; i < n; i++) anchors[i] = typeAnchors.embed(types[i].name());

        StringBuilder sb = new StringBuilder();
        sb.append("anchor magnitudes (||v||):\n");
        for (int i = 0; i < n; i++) {
            sb.append(String.format("    %-10s : %6.3f%n", types[i], norm(anchors[i])));
        }
        sb.append("pairwise cosine similarity (- = orthogonal, 1 = identical):\n");
        sb.append(String.format("    %-10s", ""));
        for (EntityType t : types) sb.append(String.format(" %8s", abbrev(t)));
        sb.append('\n');
        for (int i = 0; i < n; i++) {
            sb.append(String.format("    %-10s", types[i]));
            for (int j = 0; j < n; j++) {
                sb.append(String.format(" %8.3f", cosineSim(anchors[i], anchors[j])));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    private static String abbrev(EntityType t) {
        String n = t.name();
        return n.substring(0, Math.min(8, n.length()));
    }

    private static double cosineSim(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double denom = Math.sqrt(na) * Math.sqrt(nb);
        return denom == 0 ? 0 : dot / denom;
    }

    private static double norm(double[] v) {
        double s = 0;
        for (double x : v) s += x * x;
        return Math.sqrt(s);
    }

    public record TypeScore(EntityType type, double score) {}
}

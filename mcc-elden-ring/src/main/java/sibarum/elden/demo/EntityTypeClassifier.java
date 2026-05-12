package sibarum.elden.demo;

import sibarum.elden.annotation.EntityType;
import sibarum.elden.data.TypedSpanExample;
import sibarum.elden.pos.PosTrainer.TrainedPosLayer;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Type classifier head: given an entity span and surrounding context, predicts
 * one of the seven {@link EntityType} values.
 *
 * Stacks compositionally on top of the iter-17 entity tagger:
 *
 * <pre>
 *   POS layer  -> context vectors + POS logits (iter 13 architecture)
 *   BIO tagger -> B/I/O labels per token        (iter 17 supervision)
 *   THIS layer -> EntityType per detected span   (iter 22 — NEW)
 * </pre>
 *
 * Each predicted span (a B + I* run) is fed into this classifier as a single
 * feature vector: the average of (context + POS logits) across its tokens.
 * The MLP then maps that 207-dim feature to a 7-way distribution over types.
 *
 * Training data: ~11k typed (span, type) pairs from
 * {@code elden_ring_final_train.jsonl} (Stage 1), plus ~70 typed examples
 * from hand-annotated items (Stage 2). The JSONL provides supervision for
 * four of seven types (ARTIFACT, CHARACTER, PLACE, CONCEPT); the remaining
 * three (EVENT, FACTION, ERA) come from hand-annotated only.
 */
public final class EntityTypeClassifier {

    public static final int NUM_TYPES = EntityType.values().length;

    private final TrainedPosLayer pos;
    private final ClassifierHead head;
    private final int contextDim;
    private final int posDim;
    private final int featureDim;

    public EntityTypeClassifier(TrainedPosLayer pos, int hiddenDim, long seed) {
        this.pos = pos;
        this.contextDim = pos.contextDim;
        this.posDim = sibarum.elden.pos.PosTagset.size();
        this.featureDim = contextDim + posDim;
        Mlp mlp = new Mlp(new int[]{featureDim, hiddenDim, NUM_TYPES}, seed);
        this.head = new ClassifierHead("entity-type-head", mlp);
    }

    /** Compute span features: average of per-token (context + POS logits). */
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

    /** Train one phase (epochs of MSE-on-one-hot SGD). */
    public void train(List<TypedSpanExample> examples, int epochs, double lr, Random rng, String label) {
        if (examples.isEmpty()) {
            System.out.println("  [" + label + "] empty training set, skipping");
            return;
        }
        // Pre-compute features once.
        double[][] features = new double[examples.size()][];
        int[] targets = new int[examples.size()];
        for (int i = 0; i < examples.size(); i++) {
            TypedSpanExample ex = examples.get(i);
            features[i] = spanFeatures(ex.tokens(), ex.spanStart(), ex.spanEnd());
            targets[i] = ex.type().ordinal();
        }

        List<Integer> order = new ArrayList<>(examples.size());
        for (int i = 0; i < examples.size(); i++) order.add(i);
        double[] target = new double[NUM_TYPES];

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(order, rng);
            double totalLoss = 0;
            int correct = 0;
            for (int k : order) {
                Arrays.fill(target, 0.0);
                target[targets[k]] = 1.0;
                double[] logits = ((MatrixValue) head.apply(
                        List.of(new MatrixValue(features[k])))).data();
                for (int i = 0; i < NUM_TYPES; i++) {
                    double d = logits[i] - target[i];
                    totalLoss += d * d;
                }
                if (argmax(logits) == targets[k]) correct++;
                head.backward(new MatrixValue(target));
                head.step(lr);
            }
            if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 5 == 0) {
                System.out.printf("  [%s] epoch %2d  loss=%.4f  accuracy=%.3f%n",
                        label, epoch + 1, totalLoss / examples.size(),
                        (double) correct / examples.size());
            }
        }
    }

    /** Predict the {@link EntityType} for one span. */
    public EntityType predict(List<String> tokens, int spanStart, int spanEnd) {
        double[] feat = spanFeatures(tokens, spanStart, spanEnd);
        double[] logits = ((MatrixValue) head.apply(List.of(new MatrixValue(feat)))).data();
        return EntityType.values()[argmax(logits)];
    }

    /** Full per-type logits for a span (for confidence reporting). */
    public double[] predictLogits(List<String> tokens, int spanStart, int spanEnd) {
        double[] feat = spanFeatures(tokens, spanStart, spanEnd);
        return ((MatrixValue) head.apply(List.of(new MatrixValue(feat)))).data();
    }

    /** Evaluate top-1 accuracy on a held-out set; returns (correct, total). */
    public int[] evaluate(List<TypedSpanExample> examples) {
        int correct = 0;
        for (TypedSpanExample ex : examples) {
            EntityType p = predict(ex.tokens(), ex.spanStart(), ex.spanEnd());
            if (p == ex.type()) correct++;
        }
        return new int[]{correct, examples.size()};
    }

    /** Evaluate per-type precision/recall. Returns a string report. */
    public String evaluatePerTypeReport(List<TypedSpanExample> examples) {
        int n = NUM_TYPES;
        int[][] cm = new int[n][n];  // [gold][pred]
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

    private static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}

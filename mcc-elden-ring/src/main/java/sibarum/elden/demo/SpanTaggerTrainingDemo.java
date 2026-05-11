package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.embedding.ContextEncoder;
import sibarum.elden.embedding.CorpusVocabulary;
import sibarum.elden.training.TaggingTrainingData;
import sibarum.elden.training.TaggingTrainingData.Example;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;
import sibarum.strnn.value.Value;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * End-to-end Part 1 training demo for the binary span tagger.
 *
 * Pipeline:
 *   1. Build corpus vocabulary, initialize embedding table.
 *   2. Build context encoder (sliding window, radius=1).
 *   3. Gather all annotated items across the 4 storylines.
 *   4. Extract per-token training examples (context vector + in-entity label).
 *   5. Train a ClassifierHead (96 -> 32 -> 1) with MSE loss.
 *   6. Evaluate: precision / recall / F1 on the positive class
 *      (token is part of an entity).
 *   7. Print a sample inference trace on one item.
 */
public final class SpanTaggerTrainingDemo {

    private static final int EMBED_DIM = 32;
    private static final int WINDOW_RADIUS = 1;
    private static final int HIDDEN_DIM = 32;
    private static final long SEED = 42L;
    private static final int EPOCHS = 60;
    private static final double LR = 0.01;
    private static final double THRESHOLD = 0.5;

    public static void main(String[] args) {
        System.out.println("Span tagger training demo");
        System.out.println("=========================");
        System.out.println();

        // Vocabulary + embedding table.
        var vocab = CorpusVocabulary.tokens();
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(EMBED_DIM, SEED);
        for (String t : vocab) table.embed(t);
        System.out.println("vocabulary: " + vocab.size() + " tokens, dim=" + EMBED_DIM);

        // Context encoder.
        ContextEncoder ctx = new ContextEncoder(table, WINDOW_RADIUS);
        int contextDim = ctx.contextDim();
        System.out.println("context dim (window=" + WINDOW_RADIUS + "): " + contextDim);

        // Annotated items across all storylines.
        List<AnnotatedItem> items = new ArrayList<>();
        items.addAll(ShatteringEraAnnotations.items());
        items.addAll(MillicentAnnotations.items());
        items.addAll(DungEaterAnnotations.items());
        items.addAll(VolcanoManorAnnotations.items());
        System.out.println("annotated items: " + items.size());

        // Training examples.
        List<Example> examples = TaggingTrainingData.extractAll(items, ctx);
        long positives = examples.stream().filter(e -> e.label() > 0.5).count();
        long negatives = examples.size() - positives;
        System.out.printf("training examples: %d  (%d positive / %d negative, %.1f%% positive)%n",
                examples.size(), positives, negatives, 100.0 * positives / examples.size());
        System.out.println();

        // Classifier head: 96 -> 32 -> 1
        Mlp mlp = new Mlp(new int[]{contextDim, HIDDEN_DIM, 1}, SEED);
        ClassifierHead tagger = new ClassifierHead("span-tagger", mlp);

        // Train.
        System.out.println("training (epochs=" + EPOCHS + ", lr=" + LR + "):");
        Random rng = new Random(SEED);
        List<Example> shuffled = new ArrayList<>(examples);
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            Collections.shuffle(shuffled, rng);
            double totalLoss = 0;
            for (Example ex : shuffled) {
                MatrixValue input = new MatrixValue(ex.contextVector());
                MatrixValue target = new MatrixValue(new double[]{ex.label()});
                Value out = tagger.apply(List.of(input));
                double pred = ((MatrixValue) out).data()[0];
                totalLoss += (pred - ex.label()) * (pred - ex.label());
                tagger.backward(target);
                tagger.step(LR);
            }
            if (epoch == 0 || epoch == EPOCHS - 1 || (epoch + 1) % 10 == 0) {
                System.out.printf("  epoch %2d  avg loss = %.4f%n", epoch + 1, totalLoss / examples.size());
            }
        }
        System.out.println();

        // Evaluate.
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (Example ex : examples) {
            double pred = ((MatrixValue) tagger.apply(List.of(new MatrixValue(ex.contextVector())))).data()[0];
            boolean predPos = pred >= THRESHOLD;
            boolean truePos = ex.label() > 0.5;
            if (predPos && truePos) tp++;
            else if (predPos && !truePos) fp++;
            else if (!predPos && truePos) fn++;
            else tn++;
        }
        double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
        double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0;
        double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
        double accuracy = (double) (tp + tn) / examples.size();
        System.out.println("evaluation (training set):");
        System.out.printf("  TP=%d  FP=%d  FN=%d  TN=%d%n", tp, fp, fn, tn);
        System.out.printf("  precision=%.3f  recall=%.3f  F1=%.3f  accuracy=%.3f%n", precision, recall, f1, accuracy);
        System.out.println();

        // Sample inference trace.
        AnnotatedItem sample = items.getFirst();
        System.out.println("sample inference on '" + sample.itemId() + "':");
        List<Example> sampleExamples = TaggingTrainingData.extract(sample, ctx);
        for (Example ex : sampleExamples) {
            double pred = ((MatrixValue) tagger.apply(List.of(new MatrixValue(ex.contextVector())))).data()[0];
            boolean predPos = pred >= THRESHOLD;
            boolean truePos = ex.label() > 0.5;
            String marker = predPos == truePos ? " " : "X";
            System.out.printf("  %s pred=%.2f  truth=%s  '%s'%n",
                    marker, pred, truePos ? "1" : "0", ex.token());
        }
    }
}

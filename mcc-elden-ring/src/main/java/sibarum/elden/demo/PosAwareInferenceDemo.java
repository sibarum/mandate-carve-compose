package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.EntitySpan;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.embedding.CorpusVocabulary;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;
import sibarum.elden.pos.ConlluParser;
import sibarum.elden.pos.ConlluParser.Sentence;
import sibarum.elden.pos.PosTagset;
import sibarum.elden.pos.PosTrainer;
import sibarum.elden.pos.PosTrainer.TrainedPosLayer;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * The full layered story end-to-end:
 *   Layer 1 (POS): trained on Universal Dependencies English EWT.
 *   Layer 2 (entity span tagger): trained on the annotated Elden Ring corpus,
 *     consuming Layer 1's predictions as additional input features.
 *
 * Runs inference on Ranni's questline (held out from training) so we can
 * compare directly with {@link RanniInferenceDemo}'s output — same items,
 * same task, the only difference is the POS layer in front.
 */
public final class PosAwareInferenceDemo {

    private static final int EMBED_DIM = 32;
    private static final int WINDOW_RADIUS = 1;
    private static final int POS_HIDDEN_DIM = 64;
    private static final int ENTITY_HIDDEN_DIM = 32;
    private static final long SEED = 42L;
    private static final int POS_EPOCHS = 5;
    private static final int ENTITY_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double ENTITY_LR = 0.01;
    private static final double THRESHOLD = 0.5;

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: PosAwareInferenceDemo <path-to-en_ewt-ud-train.conllu>");
            System.exit(1);
        }

        // ============================================================
        // Layer 1: POS tagger trained on UD English EWT.
        // ============================================================
        System.out.println("=== Layer 1: POS tagger on UD English EWT ===");
        List<Sentence> sentences = ConlluParser.parse(Path.of(args[0]));
        TrainedPosLayer pos = PosTrainer.train(
                sentences,
                CorpusVocabulary.tokens(),
                EMBED_DIM, WINDOW_RADIUS, POS_HIDDEN_DIM,
                SEED, POS_EPOCHS, POS_LR);
        int contextDim = pos.contextDim;
        int posDim = PosTagset.size();
        int entityInputDim = contextDim + posDim;
        System.out.println();

        // ============================================================
        // Layer 2: entity span tagger.
        // Input = [context (96)] ++ [POS logits (17)] = 113 dim.
        // ============================================================
        System.out.println("=== Layer 2: entity span tagger ===");
        List<AnnotatedItem> annotated = new ArrayList<>();
        annotated.addAll(ShatteringEraAnnotations.items());
        annotated.addAll(MillicentAnnotations.items());
        annotated.addAll(DungEaterAnnotations.items());
        annotated.addAll(VolcanoManorAnnotations.items());

        // Build training examples.
        List<double[]> inputs = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        for (AnnotatedItem item : annotated) {
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.rawText());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            for (int i = 0; i < toks.size(); i++) {
                inputs.add(featureFor(pos, forms, i, contextDim, posDim));
                labels.add(inEntity(toks.get(i), item.spans()) ? 1.0 : 0.0);
            }
        }
        long pos1 = labels.stream().filter(d -> d > 0.5).count();
        System.out.printf("entity training examples: %d  (%d positive / %d negative)%n",
                inputs.size(), pos1, inputs.size() - pos1);

        // Train.
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, 1}, SEED);
        ClassifierHead tagger = new ClassifierHead("entity-span(pos-aware)", mlp);
        Random rng = new Random(SEED);
        Integer[] idx = new Integer[inputs.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        List<Integer> order = new ArrayList<>(List.of(idx));
        for (int epoch = 0; epoch < ENTITY_EPOCHS; epoch++) {
            Collections.shuffle(order, rng);
            double totalLoss = 0;
            for (int k : order) {
                tagger.apply(List.of(new MatrixValue(inputs.get(k))));
                tagger.backward(new MatrixValue(new double[]{labels.get(k)}));
                double pred = ((MatrixValue) tagger.apply(List.of(new MatrixValue(inputs.get(k))))).data()[0];
                double err = pred - labels.get(k);
                totalLoss += err * err;
                tagger.step(ENTITY_LR);
            }
            if (epoch == 0 || epoch == ENTITY_EPOCHS - 1 || (epoch + 1) % 10 == 0) {
                System.out.printf("  epoch %2d  avg loss = %.4f%n", epoch + 1, totalLoss / inputs.size());
            }
        }

        // Training-set metrics.
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (int k = 0; k < inputs.size(); k++) {
            double p = ((MatrixValue) tagger.apply(List.of(new MatrixValue(inputs.get(k))))).data()[0];
            boolean predPos = p >= THRESHOLD;
            boolean truePos = labels.get(k) > 0.5;
            if (predPos && truePos) tp++;
            else if (predPos && !truePos) fp++;
            else if (!predPos && truePos) fn++;
            else tn++;
        }
        double prec = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
        double rec = tp + fn > 0 ? (double) tp / (tp + fn) : 0;
        double f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
        System.out.printf("training-set: P=%.3f  R=%.3f  F1=%.3f  (TP=%d FP=%d FN=%d TN=%d)%n",
                prec, rec, f1, tp, fp, fn, tn);
        System.out.println();

        // ============================================================
        // Inference on Ranni's questline (unseen).
        // ============================================================
        System.out.println("=== Inference on Ranni's questline (unseen) ===");
        List<Item> ranniItems = RannisQuestline.items();
        int totalTokens = 0, totalPositive = 0, totalSpans = 0;

        for (Item item : ranniItems) {
            System.out.println();
            System.out.println("---------------- " + item.name() + " ----------------");
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.description());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            boolean[] positive = new boolean[toks.size()];
            double[] scores = new double[toks.size()];

            for (int i = 0; i < toks.size(); i++) {
                double[] feat = featureFor(pos, forms, i, contextDim, posDim);
                scores[i] = ((MatrixValue) tagger.apply(List.of(new MatrixValue(feat)))).data()[0];
                positive[i] = scores[i] >= THRESHOLD;
            }
            totalTokens += toks.size();
            for (boolean p : positive) if (p) totalPositive++;

            // Aggregate consecutive positives into spans.
            List<String> spans = new ArrayList<>();
            StringBuilder cur = new StringBuilder();
            for (int i = 0; i < toks.size(); i++) {
                if (positive[i]) {
                    if (cur.length() > 0) cur.append(' ');
                    cur.append(toks.get(i).text());
                } else if (cur.length() > 0) {
                    spans.add(cur.toString());
                    cur.setLength(0);
                }
            }
            if (cur.length() > 0) spans.add(cur.toString());
            totalSpans += spans.size();

            System.out.println("  predicted spans (" + spans.size() + "):");
            for (String s : spans) System.out.println("    • " + s);
        }

        System.out.println();
        System.out.println("=== Corpus-level summary (Ranni, unseen) ===");
        System.out.println("items inferred:           " + ranniItems.size());
        System.out.println("total tokens:             " + totalTokens);
        System.out.println("predicted positive:       " + totalPositive);
        System.out.printf ("positive rate:            %.1f%%%n", 100.0 * totalPositive / totalTokens);
        System.out.println("predicted spans total:    " + totalSpans);
        System.out.println();
        System.out.println("Compare against RanniInferenceDemo (no POS): 13.2% positive, 67 spans.");
    }

    private static boolean inEntity(OffsetToken t, List<EntitySpan> spans) {
        for (EntitySpan s : spans) {
            if (t.startChar() >= s.start() && t.endChar() <= s.end()) return true;
        }
        return false;
    }

    private static double[] featureFor(TrainedPosLayer pos, List<String> forms, int position,
                                        int contextDim, int posDim) {
        double[] ctx = pos.contextOf(forms, position);
        double[] logits = ((MatrixValue) pos.classifier.apply(List.of(new MatrixValue(ctx)))).data();
        double[] feat = new double[contextDim + posDim];
        System.arraycopy(ctx, 0, feat, 0, contextDim);
        System.arraycopy(logits, 0, feat, contextDim, posDim);
        return feat;
    }
}

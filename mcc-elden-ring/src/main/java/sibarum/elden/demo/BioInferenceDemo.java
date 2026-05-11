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
 * Full layered pipeline with BIO entity tagging instead of binary.
 *
 *   Layer 1: POS tagger (UD English EWT).
 *   Layer 2: 3-class BIO tagger (B-ENT / I-ENT / O), trained on the
 *            annotated Elden Ring corpus, consuming POS features.
 *
 * BIO labeling rule:
 *   - O      = token not inside any entity span
 *   - B-ENT  = token IS inside an entity span AND the previous token
 *              is NOT in the same span
 *   - I-ENT  = token IS inside an entity span AND the previous token
 *              IS in the same span
 *
 * Inference decoder: walk tokens left-to-right. B starts a new span,
 * I continues the current span (or starts one if there is no active span
 * — noise tolerance for orphan I predictions), O ends any active span.
 *
 * Compare against {@link RanniInferenceDemo} (binary, no POS) and
 * {@link PosAwareInferenceDemo} (binary, with POS) to see whether the BIO
 * structural mandate gives POS a chance to actually help.
 */
public final class BioInferenceDemo {

    private static final int LABEL_O = 0;
    private static final int LABEL_B = 1;
    private static final int LABEL_I = 2;
    private static final int NUM_LABELS = 3;

    private static final int EMBED_DIM = 32;
    private static final int WINDOW_RADIUS = 1;
    private static final int POS_HIDDEN_DIM = 64;
    private static final int ENTITY_HIDDEN_DIM = 32;
    private static final long SEED = 42L;
    private static final int POS_EPOCHS = 5;
    private static final int ENTITY_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double ENTITY_LR = 0.01;

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: BioInferenceDemo <path-to-en_ewt-ud-train.conllu>");
            System.exit(1);
        }

        // ============================================================
        // Layer 1: POS tagger.
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
        // Layer 2: BIO tagger.
        // ============================================================
        System.out.println("=== Layer 2: BIO entity tagger ===");
        List<AnnotatedItem> annotated = new ArrayList<>();
        annotated.addAll(ShatteringEraAnnotations.items());
        annotated.addAll(MillicentAnnotations.items());
        annotated.addAll(DungEaterAnnotations.items());
        annotated.addAll(VolcanoManorAnnotations.items());

        // Extract (feature, bioLabel) examples.
        List<double[]> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        int[] labelCounts = new int[NUM_LABELS];
        for (AnnotatedItem item : annotated) {
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.rawText());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int[] bio = computeBio(toks, item.spans());
            for (int i = 0; i < toks.size(); i++) {
                inputs.add(featureFor(pos, forms, i, contextDim, posDim));
                labels.add(bio[i]);
                labelCounts[bio[i]]++;
            }
        }
        System.out.printf("entity training examples: %d  (O=%d, B=%d, I=%d)%n",
                inputs.size(), labelCounts[LABEL_O], labelCounts[LABEL_B], labelCounts[LABEL_I]);

        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);

        Random rng = new Random(SEED);
        List<Integer> order = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) order.add(i);

        double[] target = new double[NUM_LABELS];
        for (int epoch = 0; epoch < ENTITY_EPOCHS; epoch++) {
            Collections.shuffle(order, rng);
            double totalLoss = 0;
            int correct = 0;
            for (int k : order) {
                int trueLabel = labels.get(k);
                java.util.Arrays.fill(target, 0.0);
                target[trueLabel] = 1.0;

                double[] logits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(inputs.get(k))))).data();
                for (int i = 0; i < NUM_LABELS; i++) {
                    double d = logits[i] - target[i];
                    totalLoss += d * d;
                }
                int pred = argmax(logits);
                if (pred == trueLabel) correct++;

                tagger.backward(new MatrixValue(target));
                tagger.step(ENTITY_LR);
            }
            if (epoch == 0 || epoch == ENTITY_EPOCHS - 1 || (epoch + 1) % 10 == 0) {
                System.out.printf("  epoch %2d  loss=%.4f  accuracy=%.3f%n",
                        epoch + 1, totalLoss / inputs.size(), (double) correct / inputs.size());
            }
        }

        // Per-class evaluation on training set.
        int[][] confusion = new int[NUM_LABELS][NUM_LABELS];
        for (int k = 0; k < inputs.size(); k++) {
            double[] logits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(inputs.get(k))))).data();
            int pred = argmax(logits);
            int truth = labels.get(k);
            confusion[truth][pred]++;
        }
        System.out.println("training-set confusion matrix (rows=truth, cols=pred):");
        System.out.println("           pred=O   pred=B   pred=I");
        String[] names = {"truth=O", "truth=B", "truth=I"};
        for (int t = 0; t < NUM_LABELS; t++) {
            System.out.printf("  %-7s  %6d   %6d   %6d%n", names[t],
                    confusion[t][0], confusion[t][1], confusion[t][2]);
        }
        System.out.println();

        // ============================================================
        // Inference on Ranni's questline.
        // ============================================================
        System.out.println("=== Inference on Ranni's questline (unseen) ===");
        List<Item> ranniItems = RannisQuestline.items();
        int totalTokens = 0, totalBStart = 0, totalISpans = 0;
        int[] predLabelCounts = new int[NUM_LABELS];

        for (Item item : ranniItems) {
            System.out.println();
            System.out.println("---------------- " + item.name() + " ----------------");
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.description());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int[] predLabels = new int[toks.size()];
            for (int i = 0; i < toks.size(); i++) {
                double[] feat = featureFor(pos, forms, i, contextDim, posDim);
                double[] logits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(feat)))).data();
                predLabels[i] = argmax(logits);
                predLabelCounts[predLabels[i]]++;
            }
            totalTokens += toks.size();

            // Decode BIO into spans.
            List<String> spans = decodeBioSpans(toks, predLabels);
            totalBStart += spans.size();

            System.out.println("  predicted spans (" + spans.size() + "):");
            for (String s : spans) System.out.println("    • " + s);
        }

        System.out.println();
        System.out.println("=== Corpus-level summary (Ranni, unseen) ===");
        System.out.printf("items inferred:         %d%n", ranniItems.size());
        System.out.printf("total tokens:           %d%n", totalTokens);
        System.out.printf("predicted labels:       O=%d  B=%d  I=%d%n",
                predLabelCounts[LABEL_O], predLabelCounts[LABEL_B], predLabelCounts[LABEL_I]);
        System.out.printf("predicted spans:        %d%n", totalBStart);
        System.out.println();
        System.out.println("Compare:");
        System.out.println("  RanniInferenceDemo  (binary, no POS): 13.2%, 67 spans");
        System.out.println("  PosAwareInferenceDemo (binary + POS): 13.8%, 70 spans");
    }

    /** Compute B/I/O labels for tokens in an item given its EntitySpans. */
    static int[] computeBio(List<OffsetToken> toks, List<EntitySpan> spans) {
        int n = toks.size();
        int[] spanOf = new int[n];
        java.util.Arrays.fill(spanOf, -1);
        for (int i = 0; i < n; i++) {
            OffsetToken t = toks.get(i);
            for (int j = 0; j < spans.size(); j++) {
                EntitySpan s = spans.get(j);
                if (t.startChar() >= s.start() && t.endChar() <= s.end()) {
                    spanOf[i] = j;
                    break;
                }
            }
        }
        int[] bio = new int[n];
        for (int i = 0; i < n; i++) {
            if (spanOf[i] < 0) {
                bio[i] = LABEL_O;
            } else if (i > 0 && spanOf[i - 1] == spanOf[i]) {
                bio[i] = LABEL_I;
            } else {
                bio[i] = LABEL_B;
            }
        }
        return bio;
    }

    /** Decode a B/I/O label sequence into surface-text spans. */
    static List<String> decodeBioSpans(List<OffsetToken> toks, int[] labels) {
        List<String> spans = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < toks.size(); i++) {
            int lab = labels[i];
            if (lab == LABEL_B) {
                if (cur.length() > 0) {
                    spans.add(cur.toString());
                    cur.setLength(0);
                }
                cur.append(toks.get(i).text());
            } else if (lab == LABEL_I) {
                if (cur.length() > 0) cur.append(' ');
                cur.append(toks.get(i).text());
            } else {
                if (cur.length() > 0) {
                    spans.add(cur.toString());
                    cur.setLength(0);
                }
            }
        }
        if (cur.length() > 0) spans.add(cur.toString());
        return spans;
    }

    private static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
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

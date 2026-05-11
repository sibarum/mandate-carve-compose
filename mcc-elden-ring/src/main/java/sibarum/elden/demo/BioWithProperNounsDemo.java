package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.data.ParquetBioLoader;
import sibarum.elden.data.ParquetBioLoader.Sentence;
import sibarum.elden.embedding.CorpusVocabulary;
import sibarum.elden.embedding.OffsetTokenizer;
import sibarum.elden.embedding.OffsetTokenizer.OffsetToken;
import sibarum.elden.pos.ConlluParser;
import sibarum.elden.pos.PosTagset;
import sibarum.elden.pos.PosTrainer;
import sibarum.elden.pos.PosTrainer.TrainedPosLayer;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;

import java.io.IOException;
import java.nio.file.Path;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.OptionalInt;
import java.util.Random;
import java.util.Set;

/**
 * Extends {@link BioInferenceDemo} by folding a HuggingFace proper-noun NER
 * Parquet dataset into the training corpus. The expectation: real-world
 * sentence contexts featuring thousands of proper nouns teach the model the
 * "capitalized rare noun in entity-like context" generalization that 14
 * Elden Ring items can't.
 *
 * Layers unchanged:
 *   1. POS (UD English EWT)
 *   2. BIO entity tagger (3-class) — now trained on Elden Ring annotations
 *      PLUS the Parquet sentences.
 *
 * Usage:
 *   BioWithProperNounsDemo <ud-conllu-path> <proper-nouns-parquet-path> [max-sentences]
 */
public final class BioWithProperNounsDemo {

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
    private static final int PRETRAIN_EPOCHS = 3;   // stage 1: parquet (proper-noun structure)
    private static final int FINETUNE_EPOCHS = 60;  // stage 2: Elden Ring (item-description style)
    private static final double POS_LR = 0.01;
    private static final double PRETRAIN_LR = 0.01;
    private static final double FINETUNE_LR = 0.01;
    private static final int DEFAULT_MAX_SENTENCES = 5000;

    public static void main(String[] args) throws IOException, SQLException, ClassNotFoundException {
        if (args.length < 2) {
            System.err.println("Usage: BioWithProperNounsDemo <ud-conllu-path> <proper-nouns-parquet-path> [max-sentences]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path parquet = Path.of(args[1]);
        int maxSentences = args.length >= 3 ? Integer.parseInt(args[2]) : DEFAULT_MAX_SENTENCES;

        // ============================================================
        // Schema check on the Parquet file.
        // ============================================================
        System.out.println("=== Parquet schema check ===");
        ParquetBioLoader.discoverSchema(parquet);
        System.out.println();

        // ============================================================
        // Load Parquet sentences.
        // ============================================================
        System.out.println("=== Loading proper-noun NER sentences ===");
        List<Sentence> parquetSentences = ParquetBioLoader.load(parquet, OptionalInt.of(maxSentences));
        long parquetTokens = parquetSentences.stream().mapToLong(s -> s.tokens().size()).sum();
        long parquetPositives = parquetSentences.stream()
                .flatMapToInt(s -> java.util.stream.IntStream.of(s.bioLabels()))
                .filter(t -> t != LABEL_O).count();
        System.out.printf("loaded %d sentences  (%d tokens, %d entity tokens, %.1f%% positive)%n",
                parquetSentences.size(), parquetTokens, parquetPositives,
                100.0 * parquetPositives / parquetTokens);
        // Label distribution sanity check.
        int[] dist = new int[NUM_LABELS + 2];
        for (Sentence s : parquetSentences) for (int t : s.bioLabels()) {
            int idx = Math.min(t, NUM_LABELS + 1);
            dist[idx]++;
        }
        System.out.printf("label dist: O=%d  B=%d  I=%d  other=%d%n", dist[0], dist[1], dist[2], dist[3]);
        System.out.println();

        // ============================================================
        // Build combined vocabulary (UD + Elden Ring + Parquet).
        // ============================================================
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence s : parquetSentences) extraVocab.addAll(s.tokens());
        System.out.println("extra vocab (Elden Ring + Parquet): " + extraVocab.size() + " tokens");

        // ============================================================
        // Layer 1: POS tagger.
        // ============================================================
        System.out.println();
        System.out.println("=== Layer 1: POS tagger ===");
        TrainedPosLayer pos = PosTrainer.train(
                ConlluParser.parse(conllu), extraVocab,
                EMBED_DIM, WINDOW_RADIUS, POS_HIDDEN_DIM,
                SEED, POS_EPOCHS, POS_LR);
        int contextDim = pos.contextDim;
        int posDim = PosTagset.size();
        int entityInputDim = contextDim + posDim;

        // ============================================================
        // Layer 2: BIO tagger — two staged training phases.
        //   Stage 1 (PRETRAIN) mandate: learn generic proper-noun structure
        //                                from Parquet sentences.
        //   Stage 2 (FINETUNE) mandate: adapt to Elden Ring item-description
        //                                style from the hand-annotated corpus.
        // ============================================================

        // Extract Parquet examples.
        List<double[]> parquetInputs = new ArrayList<>();
        List<Integer> parquetLabels = new ArrayList<>();
        for (Sentence s : parquetSentences) {
            List<String> forms = s.tokens();
            int[] bio = s.bioLabels();
            for (int i = 0; i < forms.size(); i++) {
                int lab = bio[i];
                if (lab > LABEL_I) lab = LABEL_O;
                parquetInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                parquetLabels.add(lab);
            }
        }

        // Extract Elden Ring examples.
        List<double[]> eldenInputs = new ArrayList<>();
        List<Integer> eldenLabels = new ArrayList<>();
        List<AnnotatedItem> annotated = new ArrayList<>();
        annotated.addAll(ShatteringEraAnnotations.items());
        annotated.addAll(MillicentAnnotations.items());
        annotated.addAll(DungEaterAnnotations.items());
        annotated.addAll(VolcanoManorAnnotations.items());
        for (AnnotatedItem item : annotated) {
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.rawText());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int[] bio = BioInferenceDemo.computeBio(toks, item.spans());
            for (int i = 0; i < toks.size(); i++) {
                eldenInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                eldenLabels.add(bio[i]);
            }
        }
        System.out.printf("training examples: Parquet=%d, Elden Ring=%d%n",
                parquetInputs.size(), eldenInputs.size());
        System.out.println();

        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        // ------------ Stage 1: pretrain on Parquet ------------
        System.out.println("=== Layer 2 / Stage 1: pretrain on proper-noun Parquet ===");
        trainPhase(tagger, parquetInputs, parquetLabels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        System.out.println();
        runInferenceSummary(tagger, pos, contextDim, posDim, "after Stage 1 (Parquet only)");
        System.out.println();

        // ------------ Stage 2: fine-tune on Elden Ring ------------
        System.out.println("=== Layer 2 / Stage 2: fine-tune on Elden Ring annotations ===");
        trainPhase(tagger, eldenInputs, eldenLabels, FINETUNE_EPOCHS, FINETUNE_LR, rng);
        System.out.println();

        // ============================================================
        // Inference after both stages.
        // ============================================================
        System.out.println("=== Inference on Ranni's questline (after both stages) ===");
        List<Item> ranniItems = RannisQuestline.items();
        int totalTokens = 0;
        int[] predLabelCounts = new int[NUM_LABELS];
        int totalSpans = 0;
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

            List<String> spans = BioInferenceDemo.decodeBioSpans(toks, predLabels);
            totalSpans += spans.size();

            System.out.println("  predicted spans (" + spans.size() + "):");
            for (String s : spans) System.out.println("    • " + s);
        }
        System.out.println();
        System.out.println("=== Final corpus-level summary (Ranni, unseen) ===");
        System.out.printf("items inferred:      %d%n", ranniItems.size());
        System.out.printf("total tokens:        %d%n", totalTokens);
        System.out.printf("predicted labels:    O=%d  B=%d  I=%d%n",
                predLabelCounts[LABEL_O], predLabelCounts[LABEL_B], predLabelCounts[LABEL_I]);
        System.out.printf("predicted spans:     %d%n", totalSpans);
        System.out.println();
        System.out.println("Compare prior runs:");
        System.out.println("  RanniInferenceDemo  (binary, no POS):    13.2%, 67 spans");
        System.out.println("  PosAwareInferenceDemo (binary + POS):    13.8%, 70 spans");
        System.out.println("  BioInferenceDemo    (BIO + POS):         12.5%, 67 spans");
    }

    private static void trainPhase(ClassifierHead tagger, List<double[]> inputs, List<Integer> labels,
                                    int epochs, double lr, Random rng) {
        List<Integer> order = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) order.add(i);
        double[] target = new double[NUM_LABELS];
        for (int epoch = 0; epoch < epochs; epoch++) {
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
                if (argmax(logits) == trueLabel) correct++;
                tagger.backward(new MatrixValue(target));
                tagger.step(lr);
            }
            if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 10 == 0) {
                System.out.printf("  epoch %2d  loss=%.4f  accuracy=%.3f%n",
                        epoch + 1, totalLoss / inputs.size(), (double) correct / inputs.size());
            }
        }
    }

    private static void runInferenceSummary(ClassifierHead tagger, TrainedPosLayer pos,
                                              int contextDim, int posDim, String stageLabel) {
        List<Item> ranniItems = RannisQuestline.items();
        int totalTokens = 0;
        int[] predLabelCounts = new int[NUM_LABELS];
        int totalSpans = 0;
        for (Item item : ranniItems) {
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
            totalSpans += BioInferenceDemo.decodeBioSpans(toks, predLabels).size();
        }
        System.out.printf("Ranni summary [%s]: tokens=%d, O=%d B=%d I=%d, spans=%d%n",
                stageLabel, totalTokens,
                predLabelCounts[LABEL_O], predLabelCounts[LABEL_B], predLabelCounts[LABEL_I],
                totalSpans);
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

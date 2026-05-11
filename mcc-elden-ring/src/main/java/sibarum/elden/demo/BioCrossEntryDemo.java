package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.data.LexicanumCrossEntryLabeler;
import sibarum.elden.data.LexicanumCrossEntryLabeler.Result;
import sibarum.elden.data.LexicanumParser;
import sibarum.elden.data.LexicanumParser.Entry;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Focused lore-prose training: scan the Lexicanum for entities (titles +
 * aliases), label all cross-entry mentions in every entry's prose, filter
 * to only the entity-dense passages, and train BIO on that focused subset
 * before fine-tuning on Elden Ring annotations.
 *
 *   Layer 1 (POS):    UD English EWT
 *   Layer 2 / Stage 1 pretrain: Lexicanum cross-entry mentions, filtered
 *                                by density (only entity-rich prose).
 *   Layer 2 / Stage 2 fine-tune: Elden Ring hand annotations.
 *
 * Parquet is dropped — the news-style proper-noun distribution mismatched
 * lore prose, and Lexicanum's gazetteer-driven labeling produces denser,
 * stylistically-matched supervision.
 *
 * Usage:
 *   BioCrossEntryDemo <ud-conllu> <lexicanum-txt> [min-density] [min-entity-tokens]
 */
public final class BioCrossEntryDemo {

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
    private static final int PRETRAIN_EPOCHS = 3;
    private static final int FINETUNE_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double PRETRAIN_LR = 0.01;
    private static final double FINETUNE_LR = 0.01;
    private static final double DEFAULT_MIN_DENSITY = 0.10;
    private static final int DEFAULT_MIN_ENTITY_TOKENS = 5;

    public static void main(String[] args) throws IOException, ClassNotFoundException, java.sql.SQLException {
        if (args.length < 2) {
            System.err.println("Usage: BioCrossEntryDemo <ud-conllu> <lexicanum-txt> [min-density] [min-entity-tokens]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path lex = Path.of(args[1]);
        double minDensity = args.length >= 3 ? Double.parseDouble(args[2]) : DEFAULT_MIN_DENSITY;
        int minEntityTokens = args.length >= 4 ? Integer.parseInt(args[3]) : DEFAULT_MIN_ENTITY_TOKENS;

        // ============================================================
        // Load + label Lexicanum.
        // ============================================================
        System.out.println("=== Loading Lexicanum ===");
        List<Entry> entries = LexicanumParser.parse(lex);
        System.out.println("entries parsed: " + entries.size());

        System.out.println("=== Cross-entry labeling (minDensity=" + minDensity
                + ", minEntityTokens=" + minEntityTokens + ") ===");
        Result labeled = LexicanumCrossEntryLabeler.label(entries, minDensity, minEntityTokens);
        var s = labeled.stats;
        System.out.printf("surface forms in dictionary:    %,d%n", s.totalSurfaceForms());
        System.out.printf("labeled sentences (all):        %,d (%,d tokens, %,d entity, %.1f%%)%n",
                s.labeledSentences(), s.totalTokens(), s.entityTokens(),
                100.0 * s.entityTokens() / Math.max(s.totalTokens(), 1));
        long keptTokens = labeled.sentences.stream().mapToLong(x -> x.tokens().size()).sum();
        System.out.printf("after density filter:           %,d sentences (%,d tokens, %,d entity, %.1f%%)%n",
                s.sentencesAfterFilter(), keptTokens, s.entityTokensAfterFilter(),
                100.0 * s.entityTokensAfterFilter() / Math.max(keptTokens, 1));
        System.out.println();

        // ============================================================
        // Build combined vocab.
        // ============================================================
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence sent : labeled.sentences) extraVocab.addAll(sent.tokens());
        System.out.println("extra vocab (Elden Ring + Lexicanum filtered): " + extraVocab.size());
        System.out.println();

        // ============================================================
        // Layer 1: POS.
        // ============================================================
        System.out.println("=== Layer 1: POS tagger ===");
        TrainedPosLayer pos = PosTrainer.train(
                ConlluParser.parse(conllu), extraVocab,
                EMBED_DIM, WINDOW_RADIUS, POS_HIDDEN_DIM,
                SEED, POS_EPOCHS, POS_LR);
        int contextDim = pos.contextDim;
        int posDim = PosTagset.size();
        int entityInputDim = contextDim + posDim;
        System.out.println();

        // ============================================================
        // Extract training examples.
        // ============================================================
        List<double[]> pretrainInputs = new ArrayList<>();
        List<Integer> pretrainLabels = new ArrayList<>();
        for (Sentence sent : labeled.sentences) {
            List<String> forms = sent.tokens();
            int[] bio = sent.bioLabels();
            for (int i = 0; i < forms.size(); i++) {
                int lab = bio[i];
                if (lab > LABEL_I) lab = LABEL_O;
                pretrainInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                pretrainLabels.add(lab);
            }
        }

        List<double[]> finetuneInputs = new ArrayList<>();
        List<Integer> finetuneLabels = new ArrayList<>();
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
                finetuneInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                finetuneLabels.add(bio[i]);
            }
        }
        System.out.printf("training: pretrain=%,d, finetune=%,d%n",
                pretrainInputs.size(), finetuneInputs.size());
        System.out.println();

        // ============================================================
        // Train.
        // ============================================================
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 1: pretrain on entity-dense Lexicanum subset ===");
        trainPhase(tagger, pretrainInputs, pretrainLabels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        runInferenceSummary(tagger, pos, contextDim, posDim, "after Stage 1");
        System.out.println();

        System.out.println("=== Stage 2: fine-tune on Elden Ring annotations ===");
        trainPhase(tagger, finetuneInputs, finetuneLabels, FINETUNE_EPOCHS, FINETUNE_LR, rng);
        System.out.println();

        // ============================================================
        // Inference on Ranni's questline.
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
            for (String span : spans) System.out.println("    • " + span);
        }
        System.out.println();
        System.out.println("=== Final corpus-level summary (Ranni, unseen) ===");
        System.out.printf("items inferred:    %d%n", ranniItems.size());
        System.out.printf("total tokens:      %d%n", totalTokens);
        System.out.printf("predicted labels:  O=%d  B=%d  I=%d%n",
                predLabelCounts[LABEL_O], predLabelCounts[LABEL_B], predLabelCounts[LABEL_I]);
        System.out.printf("predicted spans:   %d%n", totalSpans);
        System.out.println();
        System.out.println("Compare prior runs:");
        System.out.println("  RanniInferenceDemo  (binary, no POS):                 13.2%, 67 spans");
        System.out.println("  PosAwareInferenceDemo (binary + POS):                 13.8%, 70 spans");
        System.out.println("  BioInferenceDemo    (BIO + POS):                      12.5%, 67 spans");
        System.out.println("  BioWithProperNounsDemo (Parquet stage 1):             ~10.5%, 54 spans");
        System.out.println("  BioWithLoreDemo (Parquet + self-mention Lexicanum):    ~14%, 77 spans");
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
                                              int contextDim, int posDim, String label) {
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
        System.out.printf("Ranni [%s]: tokens=%d, O=%d B=%d I=%d, spans=%d%n",
                label, totalTokens,
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

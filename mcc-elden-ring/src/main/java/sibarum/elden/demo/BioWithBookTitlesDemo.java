package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.data.BookTitlesBioLoader;
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
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.OptionalInt;
import java.util.Random;
import java.util.Set;

/**
 * Iter 15: span-coherence pretraining from a corpus of multi-word book titles.
 *
 * Three-stage training:
 *   Stage 0: 20k book titles (sampled from book-names.parquet) — teaches the
 *     BIO tagger that multi-word capitalized phrases with internal function
 *     words ("of the", "for the", "to the") stay as one coherent span.
 *   Stage 1: Lexicanum cross-entry pretraining (identical to iter 13).
 *   Stage 2: Elden Ring fine-tune (identical to iter 13).
 *
 * The diagnostic the new stage targets: iter 13's tagger splits
 * "Remembrance of the Baleful Shadow" at "the" because article-tokens default
 * to O. Book-title pretraining shows tens of thousands of "X of the Y" /
 * "X to the Y" / "X for the Y" patterns labeled as one span — exactly the
 * pattern Elden Ring entities follow.
 *
 * Usage:
 *   BioWithBookTitlesDemo &lt;ud-conllu&gt; &lt;lexicanum-txt&gt; &lt;book-titles-parquet&gt;
 *                          [book-limit] [min-density] [min-entity-tokens]
 */
public final class BioWithBookTitlesDemo {

    private static final int LABEL_O = 0;
    private static final int LABEL_B = 1;
    private static final int LABEL_I = 2;
    private static final int NUM_LABELS = 3;

    private static final int EMBED_DIM = 32;
    private static final int WINDOW_RADIUS = 2;
    private static final int[] POS_HIDDEN = {128};
    private static final boolean POS_SHAPE_FEATURES = true;
    private static final int ENTITY_HIDDEN_DIM = 32;
    private static final long SEED = 42L;
    private static final int POS_EPOCHS = 8;
    private static final int BOOK_EPOCHS = 2;
    private static final int PRETRAIN_EPOCHS = 3;
    private static final int FINETUNE_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double BOOK_LR = 0.01;
    private static final double PRETRAIN_LR = 0.01;
    private static final double FINETUNE_LR = 0.01;
    private static final double DEFAULT_MIN_DENSITY = 0.10;
    private static final int DEFAULT_MIN_ENTITY_TOKENS = 5;
    private static final int DEFAULT_BOOK_LIMIT = 20000;

    private static final Set<String> STOP_POS_FOR_B = Set.of(
            "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART",
            "PRON", "PUNCT", "SCONJ", "SYM", "VERB");

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 3) {
            System.err.println("Usage: BioWithBookTitlesDemo <ud-conllu> <lexicanum-txt> <book-titles-parquet>"
                    + " [book-limit] [min-density] [min-entity-tokens]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path lex = Path.of(args[1]);
        Path books = Path.of(args[2]);
        int bookLimit = args.length >= 4 ? Integer.parseInt(args[3]) : DEFAULT_BOOK_LIMIT;
        double minDensity = args.length >= 5 ? Double.parseDouble(args[4]) : DEFAULT_MIN_DENSITY;
        int minEntityTokens = args.length >= 6 ? Integer.parseInt(args[5]) : DEFAULT_MIN_ENTITY_TOKENS;

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
        // Stage 0 data: book-title sentences with BIO labels.
        // ============================================================
        System.out.println("=== Loading book titles (limit=" + bookLimit + ") ===");
        List<Sentence> bookSents = BookTitlesBioLoader.load(books, OptionalInt.of(bookLimit), SEED);
        System.out.println();

        // ============================================================
        // Combined vocab (pre-registers all tokens for deterministic init).
        // ============================================================
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence sent : labeled.sentences) extraVocab.addAll(sent.tokens());
        for (Sentence sent : bookSents) extraVocab.addAll(sent.tokens());
        System.out.println("extra vocab (Elden Ring + Lexicanum filtered + book titles): " + extraVocab.size());
        System.out.println();

        // ============================================================
        // Layer 1: POS tagger.
        // ============================================================
        System.out.println("=== Layer 1: POS tagger ===");
        TrainedPosLayer pos = PosTrainer.train(
                ConlluParser.parse(conllu), extraVocab,
                EMBED_DIM, WINDOW_RADIUS, POS_HIDDEN,
                SEED, POS_EPOCHS, POS_LR, POS_SHAPE_FEATURES);
        int contextDim = pos.contextDim;
        int posDim = PosTagset.size();
        int entityInputDim = contextDim + posDim;
        System.out.println();

        // ============================================================
        // Pre-compute features per stage.
        // ============================================================
        List<double[]> bookInputs = new ArrayList<>();
        List<Integer> bookLabels = new ArrayList<>();
        for (Sentence sent : bookSents) {
            List<String> forms = sent.tokens();
            int[] bio = sent.bioLabels();
            for (int i = 0; i < forms.size(); i++) {
                int lab = bio[i];
                if (lab > LABEL_I) lab = LABEL_O;
                bookInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                bookLabels.add(lab);
            }
        }

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
        System.out.printf("training: book-titles=%,d, lexicanum-pretrain=%,d, finetune=%,d%n",
                bookInputs.size(), pretrainInputs.size(), finetuneInputs.size());
        System.out.println();

        // ============================================================
        // Train.
        // ============================================================
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 0: book-title span-coherence pretraining ===");
        trainPhase(tagger, bookInputs, bookLabels, BOOK_EPOCHS, BOOK_LR, rng);
        runInferenceSummary(tagger, pos, contextDim, posDim, "after Stage 0");
        System.out.println();

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
        System.out.println("=== Inference on Ranni's questline (after all three stages) ===");
        List<Item> ranniItems = RannisQuestline.items();
        int totalTokens = 0;
        int[] rawLabelCounts = new int[NUM_LABELS];
        int[] constrainedLabelCounts = new int[NUM_LABELS];
        int totalRawSpans = 0;
        int totalConstrainedSpans = 0;
        int totalBDemoted = 0;
        for (Item item : ranniItems) {
            System.out.println();
            System.out.println("---------------- " + item.name() + " ----------------");
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.description());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int[] posPreds = new int[toks.size()];
            int[] rawBio = new int[toks.size()];
            for (int i = 0; i < toks.size(); i++) {
                double[] ctx = pos.contextOf(forms, i);
                double[] posLogits = ((MatrixValue) pos.classifier.apply(List.of(new MatrixValue(ctx)))).data();
                posPreds[i] = argmax(posLogits);
                double[] feat = new double[contextDim + posDim];
                System.arraycopy(ctx, 0, feat, 0, contextDim);
                System.arraycopy(posLogits, 0, feat, contextDim, posDim);
                double[] bioLogits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(feat)))).data();
                rawBio[i] = argmax(bioLogits);
                rawLabelCounts[rawBio[i]]++;
            }
            int[] constrained = posConstrainedDecode(rawBio, posPreds);
            for (int b : constrained) constrainedLabelCounts[b]++;
            int demoted = 0;
            for (int i = 0; i < rawBio.length; i++) {
                if (rawBio[i] == LABEL_B && constrained[i] != LABEL_B) demoted++;
            }
            totalBDemoted += demoted;

            totalTokens += toks.size();
            List<String> rawSpans = BioInferenceDemo.decodeBioSpans(toks, rawBio);
            List<String> constrainedSpans = BioInferenceDemo.decodeBioSpans(toks, constrained);
            totalRawSpans += rawSpans.size();
            totalConstrainedSpans += constrainedSpans.size();

            System.out.printf("  raw spans:         %d (B demoted by POS constraint: %d)%n",
                    rawSpans.size(), demoted);
            System.out.println("  constrained spans (" + constrainedSpans.size() + "):");
            for (String span : constrainedSpans) System.out.println("    • " + span);
        }
        System.out.println();
        System.out.println("=== Final corpus-level summary (Ranni, unseen) ===");
        System.out.printf("items inferred:           %d%n", ranniItems.size());
        System.out.printf("total tokens:             %d%n", totalTokens);
        System.out.printf("raw labels:               O=%d  B=%d  I=%d%n",
                rawLabelCounts[LABEL_O], rawLabelCounts[LABEL_B], rawLabelCounts[LABEL_I]);
        System.out.printf("constrained labels:       O=%d  B=%d  I=%d%n",
                constrainedLabelCounts[LABEL_O], constrainedLabelCounts[LABEL_B], constrainedLabelCounts[LABEL_I]);
        System.out.printf("B demoted by POS rule:    %d%n", totalBDemoted);
        System.out.printf("raw spans:                %d%n", totalRawSpans);
        System.out.printf("constrained spans:        %d%n", totalConstrainedSpans);
        System.out.println();
        System.out.println("Compare iter 13 (POS-constrained, no book-titles): 43 spans");
        System.out.println("  known boundary errors in iter 13:");
        System.out.println("    Remembrance of  |  Baleful Shadow      (one entity split in two)");
        System.out.println("    Ranni  |  Ranni's                       (possessive duplicated)");
        System.out.println("    Blaidd's                                 (possessive truncated)");
        System.out.println("    House                                    (likely truncated from House Caria)");
    }

    private static int[] posConstrainedDecode(int[] rawBio, int[] posPreds) {
        int[] out = new int[rawBio.length];
        for (int i = 0; i < rawBio.length; i++) {
            int raw = rawBio[i];
            String posTag = PosTagset.tagAt(posPreds[i]);
            if (raw == LABEL_B) {
                out[i] = STOP_POS_FOR_B.contains(posTag) ? LABEL_O : LABEL_B;
            } else if (raw == LABEL_I) {
                if (i > 0 && (out[i - 1] == LABEL_B || out[i - 1] == LABEL_I)) {
                    out[i] = LABEL_I;
                } else {
                    out[i] = LABEL_O;
                }
            } else {
                out[i] = LABEL_O;
            }
        }
        return out;
    }

    private static void trainPhase(ClassifierHead tagger, List<double[]> inputs, List<Integer> labels,
                                    int epochs, double lr, Random rng) {
        if (inputs.isEmpty()) {
            System.out.println("  (empty training set, skipping)");
            return;
        }
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

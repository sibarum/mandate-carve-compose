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
import sibarum.elden.pos.ViterbiDecoder;
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
 * Iter 16: same three-stage training as iter 15 (book-titles → Lexicanum →
 * Elden Ring), but at inference the BIO emissions are decoded with Viterbi
 * over a hand-set transition matrix ({@link BioTransitions}) instead of
 * per-position greedy argmax.
 *
 * The targeted failure mode: iter 15 still split "Remembrance of the Baleful
 * Shadow" at the article. The diagnostic was that the BIO tagger emits B at
 * every capitalized token even when the previous token was I. Hand-set
 * transitions penalize I → B specifically, encouraging I-continuation when
 * the surrounding context is in-span.
 *
 * Usage:
 *   BioWithViterbiDemo &lt;ud-conllu&gt; &lt;lexicanum-txt&gt; &lt;book-titles-parquet&gt;
 *                        [book-limit] [min-density] [min-entity-tokens]
 */
public final class BioWithViterbiDemo {

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
    private static final double VITERBI_WEIGHT = 1.0;

    private static final Set<String> STOP_POS_FOR_B = Set.of(
            "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART",
            "PRON", "PUNCT", "SCONJ", "SYM", "VERB");

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 3) {
            System.err.println("Usage: BioWithViterbiDemo <ud-conllu> <lexicanum-txt> <book-titles-parquet>"
                    + " [book-limit] [min-density] [min-entity-tokens]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path lex = Path.of(args[1]);
        Path books = Path.of(args[2]);
        int bookLimit = args.length >= 4 ? Integer.parseInt(args[3]) : DEFAULT_BOOK_LIMIT;
        double minDensity = args.length >= 5 ? Double.parseDouble(args[4]) : DEFAULT_MIN_DENSITY;
        int minEntityTokens = args.length >= 6 ? Integer.parseInt(args[5]) : DEFAULT_MIN_ENTITY_TOKENS;

        System.out.println("=== Loading Lexicanum ===");
        List<Entry> entries = LexicanumParser.parse(lex);
        Result labeled = LexicanumCrossEntryLabeler.label(entries, minDensity, minEntityTokens);
        System.out.printf("Lexicanum: %d entries -> %,d labeled sentences after density filter%n",
                entries.size(), labeled.stats.sentencesAfterFilter());

        System.out.println("=== Loading book titles (limit=" + bookLimit + ") ===");
        List<Sentence> bookSents = BookTitlesBioLoader.load(books, OptionalInt.of(bookLimit), SEED);
        System.out.println();

        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence sent : labeled.sentences) extraVocab.addAll(sent.tokens());
        for (Sentence sent : bookSents) extraVocab.addAll(sent.tokens());
        System.out.println("extra vocab: " + extraVocab.size());
        System.out.println();

        System.out.println("=== Layer 1: POS tagger ===");
        TrainedPosLayer pos = PosTrainer.train(
                ConlluParser.parse(conllu), extraVocab,
                EMBED_DIM, WINDOW_RADIUS, POS_HIDDEN,
                SEED, POS_EPOCHS, POS_LR, POS_SHAPE_FEATURES);
        int contextDim = pos.contextDim;
        int posDim = PosTagset.size();
        int entityInputDim = contextDim + posDim;
        System.out.println();

        // ---------- Pre-compute features per stage ----------
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

        // ---------- Train ----------
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 0: book-title span-coherence pretraining ===");
        trainPhase(tagger, bookInputs, bookLabels, BOOK_EPOCHS, BOOK_LR, rng);
        System.out.println();

        System.out.println("=== Stage 1: pretrain on entity-dense Lexicanum subset ===");
        trainPhase(tagger, pretrainInputs, pretrainLabels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        System.out.println();

        System.out.println("=== Stage 2: fine-tune on Elden Ring annotations ===");
        trainPhase(tagger, finetuneInputs, finetuneLabels, FINETUNE_EPOCHS, FINETUNE_LR, rng);
        System.out.println();

        // ---------- Inference on Ranni's questline with Viterbi over BIO ----------
        double[][] bioTrans = BioTransitions.logTrans();
        double[] bioInit = BioTransitions.logInitial();

        System.out.println("=== Inference on Ranni's questline (Viterbi over BIO, transWeight="
                + VITERBI_WEIGHT + ") ===");
        List<Item> ranniItems = RannisQuestline.items();
        int totalTokens = 0;
        int[] greedyLabelCounts = new int[NUM_LABELS];
        int[] viterbiLabelCounts = new int[NUM_LABELS];
        int[] constrainedLabelCounts = new int[NUM_LABELS];
        int totalGreedySpans = 0;
        int totalViterbiSpans = 0;
        int totalConstrainedSpans = 0;
        int totalBDemoted = 0;
        int totalViterbiBChanges = 0;

        for (Item item : ranniItems) {
            System.out.println();
            System.out.println("---------------- " + item.name() + " ----------------");
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.description());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int T = toks.size();
            int[] posPreds = new int[T];
            double[][] bioEmissions = new double[T][NUM_LABELS];
            int[] greedyBio = new int[T];

            for (int i = 0; i < T; i++) {
                double[] ctx = pos.contextOf(forms, i);
                double[] posLogits = ((MatrixValue) pos.classifier.apply(List.of(new MatrixValue(ctx)))).data();
                posPreds[i] = argmax(posLogits);
                double[] feat = new double[contextDim + posDim];
                System.arraycopy(ctx, 0, feat, 0, contextDim);
                System.arraycopy(posLogits, 0, feat, contextDim, posDim);
                double[] bioLogits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(feat)))).data();
                bioEmissions[i] = bioLogits;
                greedyBio[i] = argmax(bioLogits);
                greedyLabelCounts[greedyBio[i]]++;
            }

            int[] viterbiBio = ViterbiDecoder.decode(bioEmissions, bioTrans, bioInit, VITERBI_WEIGHT);
            for (int b : viterbiBio) viterbiLabelCounts[b]++;
            int viterbiChanges = 0;
            for (int i = 0; i < T; i++) {
                if (greedyBio[i] != viterbiBio[i]) viterbiChanges++;
            }
            totalViterbiBChanges += viterbiChanges;

            int[] constrained = posConstrainedDecode(viterbiBio, posPreds);
            for (int b : constrained) constrainedLabelCounts[b]++;
            int demoted = 0;
            for (int i = 0; i < viterbiBio.length; i++) {
                if (viterbiBio[i] == LABEL_B && constrained[i] != LABEL_B) demoted++;
            }
            totalBDemoted += demoted;

            totalTokens += T;
            List<String> greedySpans = BioInferenceDemo.decodeBioSpans(toks, greedyBio);
            List<String> viterbiSpans = BioInferenceDemo.decodeBioSpans(toks, viterbiBio);
            List<String> constrainedSpans = BioInferenceDemo.decodeBioSpans(toks, constrained);
            totalGreedySpans += greedySpans.size();
            totalViterbiSpans += viterbiSpans.size();
            totalConstrainedSpans += constrainedSpans.size();

            System.out.printf("  greedy spans:      %d%n", greedySpans.size());
            System.out.printf("  viterbi spans:     %d (token changes from greedy: %d)%n",
                    viterbiSpans.size(), viterbiChanges);
            System.out.printf("  constrained spans: %d (B demoted by POS: %d)%n",
                    constrainedSpans.size(), demoted);
            System.out.println("  final spans (" + constrainedSpans.size() + "):");
            for (String span : constrainedSpans) System.out.println("    • " + span);
        }
        System.out.println();
        System.out.println("=== Final corpus-level summary (Ranni, unseen) ===");
        System.out.printf("items inferred:           %d%n", ranniItems.size());
        System.out.printf("total tokens:             %d%n", totalTokens);
        System.out.printf("greedy   labels:          O=%d  B=%d  I=%d  (spans=%d)%n",
                greedyLabelCounts[LABEL_O], greedyLabelCounts[LABEL_B], greedyLabelCounts[LABEL_I],
                totalGreedySpans);
        System.out.printf("viterbi  labels:          O=%d  B=%d  I=%d  (spans=%d, changes=%d)%n",
                viterbiLabelCounts[LABEL_O], viterbiLabelCounts[LABEL_B], viterbiLabelCounts[LABEL_I],
                totalViterbiSpans, totalViterbiBChanges);
        System.out.printf("constrained labels:       O=%d  B=%d  I=%d  (spans=%d)%n",
                constrainedLabelCounts[LABEL_O], constrainedLabelCounts[LABEL_B], constrainedLabelCounts[LABEL_I],
                totalConstrainedSpans);
        System.out.printf("B demoted by POS rule:    %d%n", totalBDemoted);
        System.out.println();
        System.out.println("Compare prior runs:");
        System.out.println("  iter 13 (no book titles, greedy BIO):     43 spans, 22 B demotions");
        System.out.println("  iter 15 (book titles, greedy BIO):        45 spans, 0 B demotions");
        System.out.println("  iter 16 (book titles + Viterbi BIO):      " + totalConstrainedSpans + " spans");
        System.out.println("  targeted failure: \"Remembrance of the Baleful Shadow\"");
        System.out.println("    iter 13: [Remembrance of] | [Baleful Shadow]");
        System.out.println("    iter 15: [Remembrance of the] | [Baleful Shadow]");
        System.out.println("    iter 16: see above");
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
        if (inputs.isEmpty()) { System.out.println("  (empty, skipping)"); return; }
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

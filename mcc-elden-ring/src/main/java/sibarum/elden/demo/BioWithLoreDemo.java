package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.data.LexicanumBioLabeler;
import sibarum.elden.data.LexicanumParser;
import sibarum.elden.data.LexicanumParser.Entry;
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
 * Full multi-source layered pipeline:
 *
 *   Layer 1 (POS):    Universal Dependencies English EWT
 *   Layer 2 / Stage 1 pretrain: Parquet (news-style proper nouns)
 *                              + Lexicanum (Warhammer lore-style prose,
 *                                bootstrapped BIO from entry headers)
 *   Layer 2 / Stage 2 fine-tune: Elden Ring hand-annotations
 *
 * The Stage-1 mix gives the BIO tagger exposure to TWO writing styles
 * (news + lore) before specialization, so it can generalize lore-prose
 * entity patterns to Elden Ring text — exactly the "diverse data for
 * generalization" point.
 *
 * Usage:
 *   BioWithLoreDemo <ud-conllu> <parquet> <lexicanum-txt> [parquet-cap]
 */
public final class BioWithLoreDemo {

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
    private static final int PRETRAIN_EPOCHS = 2;
    private static final int FINETUNE_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double PRETRAIN_LR = 0.01;
    private static final double FINETUNE_LR = 0.01;
    private static final int DEFAULT_PARQUET_CAP = 2000;

    public static void main(String[] args) throws IOException, SQLException, ClassNotFoundException {
        if (args.length < 3) {
            System.err.println("Usage: BioWithLoreDemo <ud-conllu> <parquet> <lexicanum-txt> [parquet-cap]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path parquet = Path.of(args[1]);
        Path lex = Path.of(args[2]);
        int parquetCap = args.length >= 4 ? Integer.parseInt(args[3]) : DEFAULT_PARQUET_CAP;

        // ============================================================
        // Load all sources.
        // ============================================================
        System.out.println("=== Loading sources ===");
        List<Sentence> parquetSents = ParquetBioLoader.load(parquet, OptionalInt.of(parquetCap));
        long pTokens = parquetSents.stream().mapToLong(s -> s.tokens().size()).sum();
        System.out.printf("Parquet:    %d sentences, %d tokens%n", parquetSents.size(), pTokens);

        List<Entry> lexEntries = LexicanumParser.parse(lex);
        System.out.printf("Lexicanum:  %d entries%n", lexEntries.size());

        List<Sentence> lexSents = LexicanumBioLabeler.label(lexEntries);
        long lTokens = lexSents.stream().mapToLong(s -> s.tokens().size()).sum();
        long lEntityTokens = lexSents.stream()
                .flatMapToInt(s -> java.util.stream.IntStream.of(s.bioLabels()))
                .filter(t -> t != LABEL_O).count();
        System.out.printf("Lexicanum:  %d sentences, %d tokens, %d entity tokens (%.1f%% positive)%n",
                lexSents.size(), lTokens, lEntityTokens, 100.0 * lEntityTokens / Math.max(lTokens, 1));

        // Sample first 3 entries: show titles and entity counts only (no prose).
        System.out.println("First 3 Lexicanum entries (titles + label coverage):");
        for (int i = 0; i < Math.min(3, lexEntries.size()); i++) {
            Entry e = lexEntries.get(i);
            int spans = countSpans(lexSents.get(i).bioLabels());
            System.out.printf("  [%s] %-30s  spans=%d  aliases=%d%n",
                    e.type(), e.title(), spans, e.aliases().size());
        }
        System.out.println();

        // ============================================================
        // Build combined vocab.
        // ============================================================
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence s : parquetSents) extraVocab.addAll(s.tokens());
        for (Sentence s : lexSents) extraVocab.addAll(s.tokens());
        System.out.println("extra vocab (Elden Ring + Parquet + Lexicanum): " + extraVocab.size());
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

        // ============================================================
        // Extract training examples.
        // ============================================================
        System.out.println();
        List<double[]> pretrainInputs = new ArrayList<>();
        List<Integer> pretrainLabels = new ArrayList<>();
        addPretokenizedExamples(parquetSents, pos, contextDim, posDim, pretrainInputs, pretrainLabels);
        int parquetTrainSize = pretrainInputs.size();
        addPretokenizedExamples(lexSents, pos, contextDim, posDim, pretrainInputs, pretrainLabels);
        int lexTrainSize = pretrainInputs.size() - parquetTrainSize;

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
        System.out.printf("training examples: pretrain=%d (parquet=%d, lex=%d), finetune=%d%n",
                pretrainInputs.size(), parquetTrainSize, lexTrainSize, finetuneInputs.size());
        System.out.println();

        // ============================================================
        // Train.
        // ============================================================
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 1: pretrain on Parquet + Lexicanum (diverse styles) ===");
        trainPhase(tagger, pretrainInputs, pretrainLabels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        runInferenceSummary(tagger, pos, contextDim, posDim, "after Stage 1 (Parquet + Lexicanum)");
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
            for (String s : spans) System.out.println("    • " + s);
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
    }

    private static void addPretokenizedExamples(List<Sentence> sents, TrainedPosLayer pos,
                                                  int contextDim, int posDim,
                                                  List<double[]> inputs, List<Integer> labels) {
        for (Sentence s : sents) {
            List<String> forms = s.tokens();
            int[] bio = s.bioLabels();
            for (int i = 0; i < forms.size(); i++) {
                int lab = bio[i];
                if (lab > LABEL_I) lab = LABEL_O;
                inputs.add(featureFor(pos, forms, i, contextDim, posDim));
                labels.add(lab);
            }
        }
    }

    private static int countSpans(int[] bio) {
        int spans = 0;
        for (int i = 0; i < bio.length; i++) {
            if (bio[i] == LABEL_B) spans++;
        }
        return spans;
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
        System.out.printf("Ranni summary [%s]: tokens=%d, O=%d B=%d I=%d, spans=%d%n",
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

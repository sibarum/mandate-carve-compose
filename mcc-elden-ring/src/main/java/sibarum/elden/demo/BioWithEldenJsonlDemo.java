package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.data.EldenJsonlBioLoader;
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
 * Iter 17: replace the Lexicanum + book-titles bootstrap chain with
 * domain-matched BIO supervision from a JSONL corpus of Elden Ring Q&amp;A
 * pairs annotated with canonical {@code entity_name} metadata.
 *
 * Two-stage:
 *   Stage 1: 11.7k Elden Ring Q&amp;A answers, each row's prose labeled by
 *     finding all token-aligned occurrences of {@code metadata.entity_name}.
 *   Stage 2: Elden Ring hand-annotated fine-tune (same as prior iters).
 *
 * Hypothesis: the iter 4-16 bootstrap chain was a substitute for exactly this
 * signal. Replacing it with domain-matched supervision should dissolve the
 * span-boundary failure modes (Remembrance of the Baleful Shadow, Ranni /
 * Ranni's, etc.) because the model now sees thousands of multi-word Elden
 * Ring entities with internal function words labeled correctly in context.
 *
 * Usage:
 *   BioWithEldenJsonlDemo &lt;ud-conllu&gt; &lt;jsonl&gt;
 */
public final class BioWithEldenJsonlDemo {

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
    private static final int PRETRAIN_EPOCHS = 3;
    private static final int FINETUNE_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double PRETRAIN_LR = 0.01;
    private static final double FINETUNE_LR = 0.01;

    private static final Set<String> STOP_POS_FOR_B = Set.of(
            "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART",
            "PRON", "PUNCT", "SCONJ", "SYM", "VERB");

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 2) {
            System.err.println("Usage: BioWithEldenJsonlDemo <ud-conllu> <jsonl>");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path jsonl = Path.of(args[1]);

        System.out.println("=== Loading Elden Ring JSONL ===");
        EldenJsonlBioLoader.Result loaded = EldenJsonlBioLoader.load(jsonl, OptionalInt.empty());
        System.out.println();

        // Combined vocab.
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence sent : loaded.sentences) extraVocab.addAll(sent.tokens());
        System.out.println("extra vocab (Elden Ring corpus + JSONL): " + extraVocab.size());
        System.out.println();

        // Layer 1: POS.
        System.out.println("=== Layer 1: POS tagger ===");
        TrainedPosLayer pos = PosTrainer.train(
                ConlluParser.parse(conllu), extraVocab,
                EMBED_DIM, WINDOW_RADIUS, POS_HIDDEN,
                SEED, POS_EPOCHS, POS_LR, POS_SHAPE_FEATURES);
        int contextDim = pos.contextDim;
        int posDim = PosTagset.size();
        int entityInputDim = contextDim + posDim;
        System.out.println();

        // Per-stage features.
        List<double[]> pretrainInputs = new ArrayList<>();
        List<Integer> pretrainLabels = new ArrayList<>();
        for (Sentence sent : loaded.sentences) {
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
        System.out.printf("training: jsonl-pretrain=%,d, finetune=%,d%n",
                pretrainInputs.size(), finetuneInputs.size());
        System.out.println();

        // Train.
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 1: pretrain on Elden Ring JSONL ===");
        trainPhase(tagger, pretrainInputs, pretrainLabels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        runInferenceSummary(tagger, pos, contextDim, posDim, "after Stage 1");
        System.out.println();

        System.out.println("=== Stage 2: fine-tune on hand-annotated items ===");
        trainPhase(tagger, finetuneInputs, finetuneLabels, FINETUNE_EPOCHS, FINETUNE_LR, rng);
        System.out.println();

        // Inference on Ranni.
        System.out.println("=== Inference on Ranni's questline (after both stages) ===");
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
        System.out.println("Compare prior runs (constrained final span count):");
        System.out.println("  iter 13 (POS-constrained, Lexicanum):              43 spans");
        System.out.println("  iter 15 (book titles + Lexicanum):                 45 spans");
        System.out.println("  iter 16 (book titles + Lexicanum + Viterbi BIO):   42 spans");
        System.out.println("  iter 17 (JSONL direct supervision):                " + totalConstrainedSpans + " spans");
        System.out.println("  watch for: \"Remembrance of the Baleful Shadow\" as ONE span");
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

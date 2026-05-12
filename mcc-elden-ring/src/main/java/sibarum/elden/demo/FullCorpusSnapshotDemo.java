package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.corpus.DungEaterQuestline;
import sibarum.elden.corpus.Item;
import sibarum.elden.corpus.MillicentQuestline;
import sibarum.elden.corpus.RannisQuestline;
import sibarum.elden.corpus.ShatteringEra;
import sibarum.elden.corpus.VolcanoManor;
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
import java.io.PrintWriter;
import java.nio.file.Path;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Random;
import java.util.Set;

/**
 * Snapshot of what the current pipeline (iter 17) produces on the full
 * hand-curated Elden Ring corpus. Trains the iter-17 stack, then runs
 * inference against every item across all five storylines:
 *
 *   1. Shattering Era       (10/10 annotated — trained)
 *   2. Ranni's Questline    ( 0/ 8 annotated — fully held out)
 *   3. Volcano Manor        ( 2/12 annotated — partially trained)
 *   4. Dung Eater Questline ( 1/ 5 annotated — partially trained)
 *   5. Millicent Questline  ( 1/ 9 annotated — partially trained)
 *
 * Output: a single text file with per-item token tables, predicted spans,
 * and per-storyline / overall aggregates. The file is intended for human
 * review of where the system is strong vs weak, not for automated metrics.
 *
 * Usage:
 *   FullCorpusSnapshotDemo &lt;ud-conllu&gt; &lt;jsonl&gt; [output-file]
 *
 * Default output file: snapshot-iter17.txt at the working directory root.
 */
public final class FullCorpusSnapshotDemo {

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

    /** Each storyline: display name + items + training-state caption. */
    private record Storyline(String name, List<Item> items, String trainingCaption) {}

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 2) {
            System.err.println("Usage: FullCorpusSnapshotDemo <ud-conllu> <jsonl> [output-file]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path jsonl = Path.of(args[1]);
        Path outPath = Path.of(args.length >= 3 ? args[2] : "snapshot-iter17.txt");

        // ============================================================
        // Train iter-17 pipeline.
        // ============================================================
        System.out.println("=== Loading Elden Ring JSONL ===");
        EldenJsonlBioLoader.Result loaded = EldenJsonlBioLoader.load(jsonl, OptionalInt.empty());
        System.out.println();

        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence sent : loaded.sentences) extraVocab.addAll(sent.tokens());
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

        // Stage 1 features (JSONL).
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

        // Stage 2 features (hand-annotated).
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

        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 1: pretrain on Elden Ring JSONL ===");
        trainPhase(tagger, pretrainInputs, pretrainLabels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        System.out.println("=== Stage 2: fine-tune on hand-annotated items ===");
        trainPhase(tagger, finetuneInputs, finetuneLabels, FINETUNE_EPOCHS, FINETUNE_LR, rng);
        System.out.println();

        // ============================================================
        // Snapshot: run all five storylines and write to file.
        // ============================================================
        List<Storyline> storylines = List.of(
                new Storyline("Shattering Era", ShatteringEra.items(),
                        "10 of 10 items annotated -- training data (sanity check)"),
                new Storyline("Ranni's Questline", RannisQuestline.items(),
                        "0 of 8 items annotated -- FULLY HELD OUT (real test)"),
                new Storyline("Volcano Manor", VolcanoManor.items(),
                        "2 of 12 items annotated -- partial training (interpolation)"),
                new Storyline("Dung Eater Questline", DungEaterQuestline.items(),
                        "1 of 5 items annotated -- partial training (interpolation)"),
                new Storyline("Millicent Questline", MillicentQuestline.items(),
                        "1 of 9 items annotated -- partial training (interpolation)")
        );

        Map<String, int[]> perStoryStats = new LinkedHashMap<>();  // {tokens, B, I, spans, items}
        int grandTokens = 0, grandB = 0, grandI = 0, grandSpans = 0, grandItems = 0;

        try (PrintWriter out = new PrintWriter(outPath.toFile(), java.nio.charset.StandardCharsets.UTF_8)) {
            out.println("====================================================================");
            out.println("Iter-17 pipeline snapshot: full Elden Ring hand-curated corpus");
            out.println("====================================================================");
            out.println();
            out.println("Pipeline:");
            out.println("  POS layer:    UD English EWT, embed=32 window=2 hidden=128 epochs=8");
            out.println("                + word-shape features, 90.9% dev accuracy");
            out.println("  Stage 1 BIO:  elden_ring_final_train.jsonl, 11.7k labeled prose passages");
            out.println("  Stage 2 BIO:  hand-annotated items (Shattering+Volcano+DungEater+Millicent)");
            out.println();
            out.println("Decode:");
            out.println("  Greedy argmax per token over (BIO logits).");
            out.println("  POS-conditioned constraint: B demoted to O if POS in {ADP, ADV, AUX,");
            out.println("    CCONJ, DET, INTJ, PART, PRON, PUNCT, SCONJ, SYM, VERB}; I retained");
            out.println("    only if previous constrained label is B or I.");
            out.println();

            for (Storyline story : storylines) {
                int sTokens = 0, sB = 0, sI = 0, sSpans = 0, sItems = 0;
                out.println();
                out.println("====================================================================");
                out.println("STORYLINE: " + story.name());
                out.println("  " + story.trainingCaption());
                out.println("====================================================================");

                for (Item item : story.items()) {
                    sItems++; grandItems++;
                    List<OffsetToken> toks = OffsetTokenizer.tokenize(item.description());
                    List<String> forms = toks.stream().map(OffsetToken::text).toList();
                    int T = toks.size();
                    int[] posPreds = new int[T];
                    int[] rawBio = new int[T];
                    for (int i = 0; i < T; i++) {
                        double[] ctx = pos.contextOf(forms, i);
                        double[] posLogits = ((MatrixValue) pos.classifier.apply(List.of(new MatrixValue(ctx)))).data();
                        posPreds[i] = argmax(posLogits);
                        double[] feat = new double[contextDim + posDim];
                        System.arraycopy(ctx, 0, feat, 0, contextDim);
                        System.arraycopy(posLogits, 0, feat, contextDim, posDim);
                        double[] bioLogits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(feat)))).data();
                        rawBio[i] = argmax(bioLogits);
                    }
                    int[] constrained = posConstrainedDecode(rawBio, posPreds);
                    for (int b : constrained) {
                        if (b == LABEL_B) { sB++; grandB++; }
                        else if (b == LABEL_I) { sI++; grandI++; }
                    }
                    sTokens += T; grandTokens += T;

                    List<String> spans = BioInferenceDemo.decodeBioSpans(toks, constrained);
                    sSpans += spans.size(); grandSpans += spans.size();

                    out.println();
                    out.println("---- " + item.name() + " (id: " + item.id() + ", "
                            + item.category().name().toLowerCase() + ") ----");
                    out.println("Description:");
                    out.println("  " + item.description());
                    out.println();
                    out.println("Token-level analysis:");
                    out.printf("   %-3s %-20s %-7s %-9s %-9s%n",
                            "#", "TOKEN", "POS", "BIO_RAW", "BIO_FINAL");
                    for (int i = 0; i < T; i++) {
                        out.printf("   %-3d %-20s %-7s %-9s %-9s%n",
                                i,
                                truncate("'" + toks.get(i).text() + "'", 20),
                                PosTagset.tagAt(posPreds[i]),
                                bioName(rawBio[i]),
                                bioName(constrained[i]));
                    }
                    out.println();
                    out.println("Predicted entity spans (" + spans.size() + "):");
                    if (spans.isEmpty()) {
                        out.println("  (none)");
                    } else {
                        for (String s : spans) out.println("  - " + s);
                    }
                }

                perStoryStats.put(story.name(), new int[]{sTokens, sB, sI, sSpans, sItems});
                out.println();
                out.println("---- " + story.name() + " summary ----");
                out.printf("  items: %d   tokens: %d   B: %d   I: %d   spans: %d%n",
                        sItems, sTokens, sB, sI, sSpans);
            }

            out.println();
            out.println("====================================================================");
            out.println("Overall snapshot summary");
            out.println("====================================================================");
            out.printf("  storylines:  %d%n", storylines.size());
            out.printf("  items:       %d%n", grandItems);
            out.printf("  tokens:      %d%n", grandTokens);
            out.printf("  B labels:    %d%n", grandB);
            out.printf("  I labels:    %d%n", grandI);
            out.printf("  total spans: %d%n", grandSpans);
            out.printf("  avg span length: %.2f tokens%n",
                    grandB == 0 ? 0 : (grandB + grandI) / (double) grandB);
            out.println();
            out.println("Per-storyline:");
            for (var e : perStoryStats.entrySet()) {
                int[] s = e.getValue();
                out.printf("  %-25s items=%2d  tokens=%4d  spans=%3d  (B=%3d, I=%3d)%n",
                        e.getKey(), s[4], s[0], s[3], s[1], s[2]);
            }
        }

        System.out.println();
        System.out.println("Snapshot written to: " + outPath.toAbsolutePath());
        System.out.printf("  %d storylines, %d items, %d tokens, %d spans%n",
                storylines.size(), grandItems, grandTokens, grandSpans);
        for (var e : perStoryStats.entrySet()) {
            int[] s = e.getValue();
            System.out.printf("  %-25s items=%2d  tokens=%4d  spans=%3d%n",
                    e.getKey(), s[4], s[0], s[3]);
        }
    }

    private static String bioName(int label) {
        return switch (label) { case LABEL_B -> "B"; case LABEL_I -> "I"; default -> "O"; };
    }

    private static String truncate(String s, int max) {
        if (s.length() <= max) return s;
        return s.substring(0, max - 1) + "...";
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
            if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 20 == 0) {
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

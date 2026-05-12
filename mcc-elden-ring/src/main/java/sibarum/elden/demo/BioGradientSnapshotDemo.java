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
import sibarum.elden.data.EntityClasses;
import sibarum.elden.data.KnowledgeGraphBioLoader;
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
 * Iter 18: corpus-gradient BIO training. A single 7-class head learns BIO
 * spans from three corpora at different domain specificity:
 *
 *   B-normal / I-normal      (1, 2)  -- NLP-KnowledgeGraph (generic English)
 *   B-fantasy / I-fantasy    (3, 4)  -- Lexicanum (fantasy genre)
 *   B-eldenring / I-eldenring (5, 6) -- JSONL + hand-annotated (Elden Ring)
 *
 * At inference the union of any (B-* + I-*) run is an entity span; the
 * predicted class tells how domain-specific the model thinks it is.
 *
 * Comparison anchor: snapshot-iter17.txt (44 items, 191 spans, single
 * eldenring class only).
 *
 * Usage:
 *   BioGradientSnapshotDemo &lt;ud-conllu&gt; &lt;jsonl&gt; &lt;lexicanum&gt; &lt;kg-csv&gt;
 *                            [kg-limit] [output-file]
 */
public final class BioGradientSnapshotDemo {

    private static final int NUM_LABELS = EntityClasses.NUM_CLASSES;
    private static final int O = EntityClasses.O;
    private static final int LABEL_B = 1;  // 3-class BIO for the span decoder
    private static final int LABEL_I = 2;

    private static final int EMBED_DIM = 32;
    private static final int WINDOW_RADIUS = 2;
    private static final int[] POS_HIDDEN = {128};
    private static final boolean POS_SHAPE_FEATURES = true;
    private static final int ENTITY_HIDDEN_DIM = 64;  // up from 32 to handle 7 classes
    private static final long SEED = 42L;
    private static final int POS_EPOCHS = 8;
    private static final int PRETRAIN_EPOCHS = 3;
    private static final int FINETUNE_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double PRETRAIN_LR = 0.01;
    private static final double FINETUNE_LR = 0.01;
    private static final int DEFAULT_KG_LIMIT = 10000;
    private static final double DEFAULT_MIN_DENSITY = 0.10;
    private static final int DEFAULT_MIN_ENTITY_TOKENS = 5;

    private static final Set<String> STOP_POS_FOR_B = Set.of(
            "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART",
            "PRON", "PUNCT", "SCONJ", "SYM", "VERB");

    private record Storyline(String name, List<Item> items, String trainingCaption) {}

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 4) {
            System.err.println("Usage: BioGradientSnapshotDemo <ud-conllu> <jsonl> <lexicanum> <kg-csv>"
                    + " [kg-limit] [output-file]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path jsonl = Path.of(args[1]);
        Path lex = Path.of(args[2]);
        Path kg = Path.of(args[3]);
        int kgLimit = args.length >= 5 ? Integer.parseInt(args[4]) : DEFAULT_KG_LIMIT;
        Path outPath = Path.of(args.length >= 6 ? args[5] : "snapshot-iter18.txt");

        // ============================================================
        // Load all three corpora at their respective domain levels.
        // ============================================================
        System.out.println("=== Loading NLP-KnowledgeGraph (normal class) ===");
        List<Sentence> normalSents = KnowledgeGraphBioLoader.load(kg, OptionalInt.of(kgLimit), SEED);
        System.out.println();

        System.out.println("=== Loading Lexicanum (fantasy class) ===");
        List<Entry> entries = LexicanumParser.parse(lex);
        Result labeled = LexicanumCrossEntryLabeler.label(entries, DEFAULT_MIN_DENSITY, DEFAULT_MIN_ENTITY_TOKENS);
        List<Sentence> fantasySents = new ArrayList<>();
        for (Sentence s : labeled.sentences) {
            fantasySents.add(EntityClasses.remap(s, EntityClasses.B_FANTASY, EntityClasses.I_FANTASY));
        }
        System.out.printf("Lexicanum: %,d entries -> %,d labeled sentences (remapped to fantasy class)%n",
                entries.size(), fantasySents.size());
        System.out.println();

        System.out.println("=== Loading Elden Ring JSONL (eldenring class) ===");
        EldenJsonlBioLoader.Result jsonResult = EldenJsonlBioLoader.load(jsonl, OptionalInt.empty());
        List<Sentence> erSents = new ArrayList<>();
        for (Sentence s : jsonResult.sentences) {
            erSents.add(EntityClasses.remap(s, EntityClasses.B_ELDENRING, EntityClasses.I_ELDENRING));
        }
        System.out.println();

        // ============================================================
        // Vocab over all three corpora + Elden Ring item descriptions.
        // ============================================================
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence s : normalSents) extraVocab.addAll(s.tokens());
        for (Sentence s : fantasySents) extraVocab.addAll(s.tokens());
        for (Sentence s : erSents) extraVocab.addAll(s.tokens());
        System.out.println("extra vocab (Elden Ring + normal + fantasy + eldenring): " + extraVocab.size());
        System.out.println();

        // ============================================================
        // Layer 1: POS tagger (unchanged from iter 17).
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
        // Build per-stage features.
        // Stage 1 = all three corpora mixed (gradient pretrain).
        // Stage 2 = Elden Ring hand-annotations only (calibration fine-tune).
        // ============================================================
        List<double[]> stage1Inputs = new ArrayList<>();
        List<Integer> stage1Labels = new ArrayList<>();
        appendStageData(pos, contextDim, posDim, normalSents, stage1Inputs, stage1Labels);
        appendStageData(pos, contextDim, posDim, fantasySents, stage1Inputs, stage1Labels);
        appendStageData(pos, contextDim, posDim, erSents, stage1Inputs, stage1Labels);

        List<double[]> stage2Inputs = new ArrayList<>();
        List<Integer> stage2Labels = new ArrayList<>();
        List<AnnotatedItem> annotated = new ArrayList<>();
        annotated.addAll(ShatteringEraAnnotations.items());
        annotated.addAll(MillicentAnnotations.items());
        annotated.addAll(DungEaterAnnotations.items());
        annotated.addAll(VolcanoManorAnnotations.items());
        for (AnnotatedItem item : annotated) {
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.rawText());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int[] bio3 = BioInferenceDemo.computeBio(toks, item.spans());
            for (int i = 0; i < toks.size(); i++) {
                int lab;
                if (bio3[i] == LABEL_B) lab = EntityClasses.B_ELDENRING;
                else if (bio3[i] == LABEL_I) lab = EntityClasses.I_ELDENRING;
                else lab = O;
                stage2Inputs.add(featureFor(pos, forms, i, contextDim, posDim));
                stage2Labels.add(lab);
            }
        }
        System.out.printf("training: stage1-mix=%,d, stage2-finetune=%,d%n",
                stage1Inputs.size(), stage2Inputs.size());
        System.out.println();

        // ============================================================
        // Train 7-class head.
        // ============================================================
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-gradient-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 1: pretrain on three-corpus gradient ===");
        trainPhase(tagger, stage1Inputs, stage1Labels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        System.out.println("=== Stage 2: fine-tune on hand-annotated items ===");
        trainPhase(tagger, stage2Inputs, stage2Labels, FINETUNE_EPOCHS, FINETUNE_LR, rng);
        System.out.println();

        // ============================================================
        // Snapshot: run all five storylines, write to file.
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

        Map<String, int[]> perStoryStats = new LinkedHashMap<>();
        int grandTokens = 0, grandSpans = 0, grandItems = 0;
        int[] grandClassCounts = new int[NUM_LABELS];

        try (PrintWriter out = new PrintWriter(outPath.toFile(), java.nio.charset.StandardCharsets.UTF_8)) {
            out.println("====================================================================");
            out.println("Iter-18 corpus-gradient pipeline snapshot");
            out.println("====================================================================");
            out.println();
            out.println("Pipeline:");
            out.println("  POS layer:    UD English EWT, embed=32 window=2 hidden=128 epochs=8");
            out.println("                + word-shape features, 90.9% dev accuracy");
            out.println("  Stage 1 BIO:  mixed corpus gradient");
            out.printf ("                    normal  (NLP-KG)        %,d sentences%n", normalSents.size());
            out.printf ("                    fantasy (Lexicanum)     %,d sentences%n", fantasySents.size());
            out.printf ("                    eldenring (JSONL)        %,d sentences%n", erSents.size());
            out.println("  Stage 2 BIO:  hand-annotated items, eldenring class only");
            out.println();
            out.println("Output: 7-class BIO {O, B-N, I-N, B-F, I-F, B-E, I-E}");
            out.println("        spans collapse any B-* + I-* into a positive entity recognition;");
            out.println("        the class label records which corpus the model thinks it resembles.");
            out.println();

            for (Storyline story : storylines) {
                int sTokens = 0, sSpans = 0, sItems = 0;
                int[] sClassCounts = new int[NUM_LABELS];
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
                    int[] rawCls = new int[T];
                    for (int i = 0; i < T; i++) {
                        double[] ctx = pos.contextOf(forms, i);
                        double[] posLogits = ((MatrixValue) pos.classifier.apply(List.of(new MatrixValue(ctx)))).data();
                        posPreds[i] = argmax(posLogits);
                        double[] feat = new double[contextDim + posDim];
                        System.arraycopy(ctx, 0, feat, 0, contextDim);
                        System.arraycopy(posLogits, 0, feat, contextDim, posDim);
                        double[] bioLogits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(feat)))).data();
                        rawCls[i] = argmax(bioLogits);
                        sClassCounts[rawCls[i]]++;
                        grandClassCounts[rawCls[i]]++;
                    }

                    // Collapse to 3-class BIO for span decoding + POS constraint.
                    int[] rawBio3 = new int[T];
                    for (int i = 0; i < T; i++) rawBio3[i] = EntityClasses.toBio3(rawCls[i]);
                    int[] constrained = posConstrainedDecode(rawBio3, posPreds);

                    // Spans + their dominant class.
                    List<int[]> spanRanges = findSpanRanges(constrained);
                    sSpans += spanRanges.size(); grandSpans += spanRanges.size();
                    sTokens += T; grandTokens += T;

                    out.println();
                    out.println("---- " + item.name() + " (id: " + item.id() + ", "
                            + item.category().name().toLowerCase() + ") ----");
                    out.println("Description:");
                    out.println("  " + item.description());
                    out.println();
                    out.println("Token-level analysis:");
                    out.printf("   %-3s %-20s %-7s %-7s %-9s%n",
                            "#", "TOKEN", "POS", "BIO-7", "BIO-FIN");
                    for (int i = 0; i < T; i++) {
                        out.printf("   %-3d %-20s %-7s %-7s %-9s%n",
                                i,
                                truncate("'" + toks.get(i).text() + "'", 20),
                                PosTagset.tagAt(posPreds[i]),
                                EntityClasses.shortName(rawCls[i]),
                                bio3Name(constrained[i]));
                    }
                    out.println();
                    out.println("Predicted entity spans (" + spanRanges.size() + "):");
                    if (spanRanges.isEmpty()) {
                        out.println("  (none)");
                    } else {
                        for (int[] range : spanRanges) {
                            StringBuilder sb = new StringBuilder();
                            for (int i = range[0]; i <= range[1]; i++) {
                                if (i > range[0]) sb.append(' ');
                                sb.append(toks.get(i).text());
                            }
                            // Determine dominant class for the span.
                            int domClass = dominantClass(rawCls, range[0], range[1]);
                            out.printf("  - [%s] %s%n", EntityClasses.domain(domClass), sb);
                        }
                    }
                }

                perStoryStats.put(story.name(), new int[]{sTokens, sSpans, sItems,
                        sClassCounts[EntityClasses.B_NORMAL],
                        sClassCounts[EntityClasses.B_FANTASY],
                        sClassCounts[EntityClasses.B_ELDENRING]});
                out.println();
                out.println("---- " + story.name() + " summary ----");
                out.printf("  items: %d   tokens: %d   spans: %d%n", sItems, sTokens, sSpans);
                out.printf("  B-class breakdown:  B-normal=%d  B-fantasy=%d  B-eldenring=%d%n",
                        sClassCounts[EntityClasses.B_NORMAL],
                        sClassCounts[EntityClasses.B_FANTASY],
                        sClassCounts[EntityClasses.B_ELDENRING]);
            }

            out.println();
            out.println("====================================================================");
            out.println("Overall snapshot summary");
            out.println("====================================================================");
            out.printf("  storylines:  %d%n", storylines.size());
            out.printf("  items:       %d%n", grandItems);
            out.printf("  tokens:      %d%n", grandTokens);
            out.printf("  total spans: %d%n", grandSpans);
            out.println();
            out.println("Class predictions (raw, before POS-constrained collapse):");
            out.printf("    O           : %d%n", grandClassCounts[O]);
            out.printf("    B-normal    : %d   (I-normal=%d)%n",
                    grandClassCounts[EntityClasses.B_NORMAL], grandClassCounts[EntityClasses.I_NORMAL]);
            out.printf("    B-fantasy   : %d   (I-fantasy=%d)%n",
                    grandClassCounts[EntityClasses.B_FANTASY], grandClassCounts[EntityClasses.I_FANTASY]);
            out.printf("    B-eldenring : %d   (I-eldenring=%d)%n",
                    grandClassCounts[EntityClasses.B_ELDENRING], grandClassCounts[EntityClasses.I_ELDENRING]);
            out.println();
            out.println("Per-storyline:");
            for (var e : perStoryStats.entrySet()) {
                int[] s = e.getValue();
                out.printf("  %-25s items=%2d tokens=%4d spans=%3d  (B-N=%2d, B-F=%2d, B-E=%2d)%n",
                        e.getKey(), s[2], s[0], s[1], s[3], s[4], s[5]);
            }
            out.println();
            out.println("Compare to iter 17 (single eldenring class, 44 items, 191 spans).");
        }

        System.out.println("Snapshot written to: " + outPath.toAbsolutePath());
        System.out.printf("  %d storylines, %d items, %d tokens, %d spans%n",
                storylines.size(), grandItems, grandTokens, grandSpans);
        System.out.printf("  B-normal=%d, B-fantasy=%d, B-eldenring=%d%n",
                grandClassCounts[EntityClasses.B_NORMAL],
                grandClassCounts[EntityClasses.B_FANTASY],
                grandClassCounts[EntityClasses.B_ELDENRING]);
    }

    private static void appendStageData(TrainedPosLayer pos, int contextDim, int posDim,
                                          List<Sentence> sents,
                                          List<double[]> inputs, List<Integer> labels) {
        for (Sentence sent : sents) {
            List<String> forms = sent.tokens();
            int[] bio = sent.bioLabels();
            for (int i = 0; i < forms.size(); i++) {
                inputs.add(featureFor(pos, forms, i, contextDim, posDim));
                labels.add(bio[i]);
            }
        }
    }

    private static List<int[]> findSpanRanges(int[] bio3) {
        List<int[]> out = new ArrayList<>();
        int start = -1;
        for (int i = 0; i < bio3.length; i++) {
            if (bio3[i] == LABEL_B) {
                if (start >= 0) out.add(new int[]{start, i - 1});
                start = i;
            } else if (bio3[i] == LABEL_I) {
                if (start < 0) start = i;
            } else {
                if (start >= 0) { out.add(new int[]{start, i - 1}); start = -1; }
            }
        }
        if (start >= 0) out.add(new int[]{start, bio3.length - 1});
        return out;
    }

    private static int dominantClass(int[] rawCls, int from, int to) {
        int[] counts = new int[NUM_LABELS];
        for (int i = from; i <= to; i++) counts[rawCls[i]]++;
        // Prefer entity classes over O; among entity classes pick max.
        int best = O, bestCount = -1;
        for (int c = 1; c < NUM_LABELS; c++) {
            if (counts[c] > bestCount) { bestCount = counts[c]; best = c; }
        }
        return best;
    }

    private static String bio3Name(int label) {
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
                out[i] = STOP_POS_FOR_B.contains(posTag) ? 0 : LABEL_B;
            } else if (raw == LABEL_I) {
                if (i > 0 && (out[i - 1] == LABEL_B || out[i - 1] == LABEL_I)) {
                    out[i] = LABEL_I;
                } else {
                    out[i] = 0;
                }
            } else {
                out[i] = 0;
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

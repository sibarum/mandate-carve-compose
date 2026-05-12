package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.EntitySpan;
import sibarum.elden.annotation.EntityType;
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
import sibarum.elden.data.EldenJsonlTypedLoader;
import sibarum.elden.data.ParquetBioLoader.Sentence;
import sibarum.elden.data.TypedSpanExample;
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
 * Iter 22: per-type heads. Stacks a typed-entity classifier on top of the
 * iter-17 entity tagger.
 *
 * Three layers, three named mandates, three separately-trainable heads:
 *
 * <ol>
 *   <li>POS layer (UD English EWT, iter 13)            -> 17-way POS</li>
 *   <li>BIO tagger (iter 17 JSONL pretrain + fine-tune) -> 3-way O/B/I</li>
 *   <li>NEW Type head (iter 22, this demo)              -> 7-way EntityType</li>
 * </ol>
 *
 * The type head is trained on (span, type) pairs:
 *   Stage 1: ~11k typed spans from elden_ring_final_train.jsonl
 *            (covers ARTIFACT, CHARACTER, PLACE, CONCEPT)
 *   Stage 2: ~70 typed spans from hand-annotated items
 *            (adds EVENT, FACTION, ERA)
 *
 * At inference, every span detected by the BIO tagger gets a predicted
 * EntityType printed alongside it. The snapshot file records typed spans
 * across all 5 storylines for review.
 *
 * Usage:
 *   BioWithTypedHeadsSnapshotDemo &lt;ud-conllu&gt; &lt;jsonl&gt; [output-file]
 */
public final class BioWithTypedHeadsSnapshotDemo {

    private static final int LABEL_O = 0;
    private static final int LABEL_B = 1;
    private static final int LABEL_I = 2;
    private static final int NUM_BIO_LABELS = 3;

    private static final int EMBED_DIM = 32;
    private static final int WINDOW_RADIUS = 2;
    private static final int[] POS_HIDDEN = {128};
    private static final boolean POS_SHAPE_FEATURES = true;
    private static final int BIO_HIDDEN_DIM = 32;
    private static final int TYPE_HIDDEN_DIM = 32;
    private static final long SEED = 42L;
    private static final int POS_EPOCHS = 8;
    private static final int BIO_PRETRAIN_EPOCHS = 3;
    private static final int BIO_FINETUNE_EPOCHS = 60;
    private static final int TYPE_PRETRAIN_EPOCHS = 3;
    private static final int TYPE_FINETUNE_EPOCHS = 60;
    private static final double POS_LR = 0.01;
    private static final double BIO_LR = 0.01;
    private static final double TYPE_LR = 0.01;

    private static final Set<String> STOP_POS_FOR_B = Set.of(
            "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART",
            "PRON", "PUNCT", "SCONJ", "SYM", "VERB");

    private record Storyline(String name, List<Item> items, String trainingCaption) {}

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 2) {
            System.err.println("Usage: BioWithTypedHeadsSnapshotDemo <ud-conllu> <jsonl> [output-file]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path jsonl = Path.of(args[1]);
        Path outPath = Path.of(args.length >= 3 ? args[2] : "snapshot-iter22.txt");

        // ============================================================
        // Load JSONL twice: once for BIO supervision, once for typed.
        // ============================================================
        System.out.println("=== Loading JSONL for BIO supervision ===");
        EldenJsonlBioLoader.Result bioLoaded = EldenJsonlBioLoader.load(jsonl, OptionalInt.empty());
        System.out.println();

        System.out.println("=== Loading JSONL for typed-span supervision ===");
        List<TypedSpanExample> typedJsonl = EldenJsonlTypedLoader.load(jsonl);
        System.out.println();

        // Vocab + POS.
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence s : bioLoaded.sentences) extraVocab.addAll(s.tokens());
        for (TypedSpanExample ex : typedJsonl) extraVocab.addAll(ex.tokens());
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

        // ============================================================
        // Layer 2: BIO tagger (iter 17 style).
        // ============================================================
        List<double[]> bioPretrainInputs = new ArrayList<>();
        List<Integer> bioPretrainLabels = new ArrayList<>();
        for (Sentence sent : bioLoaded.sentences) {
            List<String> forms = sent.tokens();
            int[] bio = sent.bioLabels();
            for (int i = 0; i < forms.size(); i++) {
                int lab = bio[i];
                if (lab > LABEL_I) lab = LABEL_O;
                bioPretrainInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                bioPretrainLabels.add(lab);
            }
        }

        List<AnnotatedItem> annotated = new ArrayList<>();
        annotated.addAll(ShatteringEraAnnotations.items());
        annotated.addAll(MillicentAnnotations.items());
        annotated.addAll(DungEaterAnnotations.items());
        annotated.addAll(VolcanoManorAnnotations.items());

        List<double[]> bioFinetuneInputs = new ArrayList<>();
        List<Integer> bioFinetuneLabels = new ArrayList<>();
        for (AnnotatedItem item : annotated) {
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.rawText());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int[] bio = BioInferenceDemo.computeBio(toks, item.spans());
            for (int i = 0; i < toks.size(); i++) {
                bioFinetuneInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                bioFinetuneLabels.add(bio[i]);
            }
        }

        Mlp bioMlp = new Mlp(new int[]{entityInputDim, BIO_HIDDEN_DIM, NUM_BIO_LABELS}, SEED);
        ClassifierHead bioTagger = new ClassifierHead("bio-tagger", bioMlp);
        Random bioRng = new Random(SEED);

        System.out.println("=== Layer 2 / BIO Stage 1: pretrain on JSONL ===");
        trainBioPhase(bioTagger, bioPretrainInputs, bioPretrainLabels, BIO_PRETRAIN_EPOCHS, BIO_LR, bioRng);
        System.out.println("=== Layer 2 / BIO Stage 2: fine-tune on hand-annotated ===");
        trainBioPhase(bioTagger, bioFinetuneInputs, bioFinetuneLabels, BIO_FINETUNE_EPOCHS, BIO_LR, bioRng);
        System.out.println();

        // ============================================================
        // Layer 3: Type classifier.
        // ============================================================
        // Build typed hand-annotated examples from annotated items.
        List<TypedSpanExample> typedHand = new ArrayList<>();
        for (AnnotatedItem item : annotated) {
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.rawText());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            for (EntitySpan span : item.spans()) {
                int[] range = charRangeToTokenRange(toks, span.start(), span.end());
                if (range == null) continue;
                typedHand.add(new TypedSpanExample(forms, range[0], range[1], span.type()));
            }
        }
        Map<EntityType, Integer> typedHandPerType = new LinkedHashMap<>();
        for (TypedSpanExample ex : typedHand) {
            typedHandPerType.merge(ex.type(), 1, Integer::sum);
        }
        System.out.printf("Typed hand-annotated supervision: %d spans%n", typedHand.size());
        for (var e : typedHandPerType.entrySet()) {
            System.out.printf("    %-12s : %d%n", e.getKey(), e.getValue());
        }
        System.out.println();

        EntityTypeClassifier typeClassifier = new EntityTypeClassifier(pos, TYPE_HIDDEN_DIM, SEED);

        Random typeRng = new Random(SEED);
        System.out.println("=== Layer 3 / Type Stage 1: pretrain on JSONL typed spans ===");
        typeClassifier.train(typedJsonl, TYPE_PRETRAIN_EPOCHS, TYPE_LR, typeRng, "type-pretrain");
        System.out.println("=== Layer 3 / Type Stage 2: fine-tune on hand-annotated typed spans ===");
        typeClassifier.train(typedHand, TYPE_FINETUNE_EPOCHS, TYPE_LR, typeRng, "type-finetune");
        System.out.println();

        // Quick sanity eval on hand-annotated (training-set metric only).
        int[] handEval = typeClassifier.evaluate(typedHand);
        System.out.printf("Type-head training-set accuracy on hand-annotated: %d/%d = %.3f%n",
                handEval[0], handEval[1], (double) handEval[0] / handEval[1]);
        System.out.println();
        System.out.println("Per-type metrics on hand-annotated (training set):");
        System.out.print(typeClassifier.evaluatePerTypeReport(typedHand));
        System.out.println();

        // ============================================================
        // Snapshot: full corpus with typed spans.
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

        Map<String, int[]> perStoryStats = new LinkedHashMap<>();  // {tokens, spans, items}
        int[] grandTypeCounts = new int[EntityType.values().length];
        int grandTokens = 0, grandSpans = 0, grandItems = 0;

        try (PrintWriter out = new PrintWriter(outPath.toFile(), java.nio.charset.StandardCharsets.UTF_8)) {
            out.println("====================================================================");
            out.println("Iter-22 typed-entity snapshot (3-layer pipeline)");
            out.println("====================================================================");
            out.println();
            out.println("Pipeline:");
            out.println("  Layer 1: POS tagger     (UD English EWT, 90.9% dev acc)");
            out.println("  Layer 2: BIO tagger     (iter 17: JSONL pretrain + hand-annotated fine-tune)");
            out.println("  Layer 3: Type classifier (NEW): EntityType per detected span");
            out.println("             features = average over span tokens of (context + POS logits)");
            out.println("             7 classes: CHARACTER, ARTIFACT, EVENT, PLACE, FACTION, ERA, CONCEPT");
            out.println();
            out.println("Type-head training:");
            out.printf ("  Stage 1: %,d JSONL typed spans (4 of 7 types covered)%n", typedJsonl.size());
            out.printf ("  Stage 2: %,d hand-annotated typed spans (introduces EVENT, FACTION, ERA)%n",
                    typedHand.size());
            out.println();

            for (Storyline story : storylines) {
                int sTokens = 0, sSpans = 0, sItems = 0;
                int[] sTypeCounts = new int[EntityType.values().length];
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
                        double[] bioLogits = ((MatrixValue) bioTagger.apply(List.of(new MatrixValue(feat)))).data();
                        rawBio[i] = argmax(bioLogits);
                    }
                    int[] constrained = posConstrainedDecode(rawBio, posPreds);
                    sTokens += T; grandTokens += T;

                    // Decode spans + predict types.
                    List<int[]> spanRanges = decodeSpanRanges(constrained);
                    sSpans += spanRanges.size(); grandSpans += spanRanges.size();

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
                    out.println("Predicted typed entity spans (" + spanRanges.size() + "):");
                    if (spanRanges.isEmpty()) {
                        out.println("  (none)");
                    } else {
                        for (int[] range : spanRanges) {
                            EntityType type = typeClassifier.predict(forms, range[0], range[1]);
                            sTypeCounts[type.ordinal()]++;
                            grandTypeCounts[type.ordinal()]++;
                            StringBuilder sb = new StringBuilder();
                            for (int i = range[0]; i <= range[1]; i++) {
                                if (i > range[0]) sb.append(' ');
                                sb.append(toks.get(i).text());
                            }
                            out.printf("  - [%-9s] %s%n", type, sb);
                        }
                    }
                }

                perStoryStats.put(story.name(), new int[]{sTokens, sSpans, sItems});
                out.println();
                out.println("---- " + story.name() + " summary ----");
                out.printf("  items: %d   tokens: %d   spans: %d%n", sItems, sTokens, sSpans);
                out.println("  type distribution:");
                for (EntityType t : EntityType.values()) {
                    if (sTypeCounts[t.ordinal()] > 0) {
                        out.printf("    %-10s : %d%n", t, sTypeCounts[t.ordinal()]);
                    }
                }
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
            out.println("Type distribution (across full corpus):");
            for (EntityType t : EntityType.values()) {
                out.printf("    %-10s : %d%n", t, grandTypeCounts[t.ordinal()]);
            }
            out.println();
            out.println("Per-storyline:");
            for (var e : perStoryStats.entrySet()) {
                int[] s = e.getValue();
                out.printf("  %-25s items=%2d  tokens=%4d  spans=%3d%n",
                        e.getKey(), s[2], s[0], s[1]);
            }
            out.println();
            out.println("Compare prior runs (BIO span counts, untyped):");
            out.println("  iter 17 (JSONL pretrain): 191 spans (same BIO pipeline)");
            out.println("  iter 22 (this run):      " + grandSpans + " spans, plus type label per span");
        }

        System.out.println("Snapshot written to: " + outPath.toAbsolutePath());
        System.out.printf("  %d items, %d tokens, %d spans%n", grandItems, grandTokens, grandSpans);
        System.out.println("Type distribution:");
        for (EntityType t : EntityType.values()) {
            System.out.printf("    %-10s : %d%n", t, grandTypeCounts[t.ordinal()]);
        }
    }

    /** Decode contiguous (B + I*) runs in a constrained BIO sequence. */
    private static List<int[]> decodeSpanRanges(int[] bio) {
        List<int[]> out = new ArrayList<>();
        int start = -1;
        for (int i = 0; i < bio.length; i++) {
            if (bio[i] == LABEL_B) {
                if (start >= 0) out.add(new int[]{start, i - 1});
                start = i;
            } else if (bio[i] == LABEL_I) {
                if (start < 0) start = i;
            } else {
                if (start >= 0) { out.add(new int[]{start, i - 1}); start = -1; }
            }
        }
        if (start >= 0) out.add(new int[]{start, bio.length - 1});
        return out;
    }

    /** Map a character range to the inclusive token-index range it covers. */
    private static int[] charRangeToTokenRange(List<OffsetToken> toks, int charStart, int charEnd) {
        int startIdx = -1, endIdx = -1;
        for (int i = 0; i < toks.size(); i++) {
            OffsetToken t = toks.get(i);
            if (t.startChar() >= charStart && t.endChar() <= charEnd) {
                if (startIdx < 0) startIdx = i;
                endIdx = i;
            }
        }
        if (startIdx < 0 || endIdx < 0) return null;
        return new int[]{startIdx, endIdx};
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

    private static void trainBioPhase(ClassifierHead tagger, List<double[]> inputs, List<Integer> labels,
                                       int epochs, double lr, Random rng) {
        if (inputs.isEmpty()) { System.out.println("  (empty, skipping)"); return; }
        List<Integer> order = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) order.add(i);
        double[] target = new double[NUM_BIO_LABELS];
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(order, rng);
            double totalLoss = 0;
            int correct = 0;
            for (int k : order) {
                int trueLabel = labels.get(k);
                java.util.Arrays.fill(target, 0.0);
                target[trueLabel] = 1.0;
                double[] logits = ((MatrixValue) tagger.apply(List.of(new MatrixValue(inputs.get(k))))).data();
                for (int i = 0; i < NUM_BIO_LABELS; i++) {
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

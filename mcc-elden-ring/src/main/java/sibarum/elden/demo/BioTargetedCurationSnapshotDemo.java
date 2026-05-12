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
import java.util.function.Predicate;

/**
 * Iter 20: <b>failure-driven biased budget-clip</b>.
 *
 * Same 80-span budget per source as iter 19, BUT instead of random sampling
 * from each external corpus, sentences that exhibit one of the named iter-19
 * failure-mode patterns are preferentially selected. The remaining budget is
 * filled randomly. Total training-data volume is unchanged from iter 19.
 *
 * The four target patterns, each one a "mandate" derived from iter 19's
 * observed failures:
 *
 * <ol>
 *   <li><b>Sentence-initial capitalized O.</b> Teaches the model that a
 *       capitalized first token can be background — addresses iter 19's
 *       false positive on `Forged` and similar tokens.</li>
 *   <li><b>X of the Y inside one span.</b> Teaches the model to carry I-ENT
 *       across `of the` — addresses iter 19's split of
 *       "Remembrance of the Baleful Shadow" at the article.</li>
 *   <li><b>Two-word PROPN sequence as one span.</b> Teaches the model that
 *       adjacent capitalized tokens often form one entity — addresses
 *       iter 19's truncation of "House Caria" to "House".</li>
 *   <li><b>Long O-stretch within a sentence containing at least one
 *       entity.</b> Teaches the model that nominal-looking tokens can be O
 *       when they're not entities — addresses false positives on common
 *       nouns like `figure`, `trust`, `impression`.</li>
 * </ol>
 *
 * Usage:
 *   BioTargetedCurationSnapshotDemo &lt;ud-conllu&gt; &lt;jsonl&gt; &lt;lexicanum&gt; &lt;kg-csv&gt;
 *                                    [span-budget] [output-file]
 */
public final class BioTargetedCurationSnapshotDemo {

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
    private static final int DEFAULT_SPAN_BUDGET = 80;
    private static final double DEFAULT_MIN_DENSITY = 0.10;
    private static final int DEFAULT_MIN_ENTITY_TOKENS = 5;
    private static final int KG_LOAD_LIMIT = 500;  // larger to give the pattern filter room

    private static final Set<String> STOP_POS_FOR_B = Set.of(
            "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART",
            "PRON", "PUNCT", "SCONJ", "SYM", "VERB");

    private record Storyline(String name, List<Item> items, String trainingCaption) {}

    /** The four targeted patterns, named after the iter-19 failure they address. */
    private static final List<Predicate<Sentence>> TARGETED_PATTERNS = List.of(
            BioTargetedCurationSnapshotDemo::matchOfTheInsideSpan,
            BioTargetedCurationSnapshotDemo::matchTwoWordPropn,
            BioTargetedCurationSnapshotDemo::matchInitialCapAsO,
            BioTargetedCurationSnapshotDemo::matchLongOStretch
    );
    private static final List<String> TARGETED_PATTERN_NAMES = List.of(
            "of-the-inside",
            "two-word-propn",
            "initial-cap-as-O",
            "long-O-stretch"
    );

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 4) {
            System.err.println("Usage: BioTargetedCurationSnapshotDemo <ud-conllu> <jsonl> <lexicanum> <kg-csv>"
                    + " [span-budget] [output-file]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path jsonl = Path.of(args[1]);
        Path lex = Path.of(args[2]);
        Path kg = Path.of(args[3]);
        int budget = args.length >= 5 ? Integer.parseInt(args[4]) : DEFAULT_SPAN_BUDGET;
        Path outPath = Path.of(args.length >= 6 ? args[5] : "snapshot-iter20.txt");

        Random clipRng = new Random(SEED);

        // ============================================================
        // Load + targeted-clip each external corpus.
        // ============================================================
        System.out.println("=== Loading + targeted-clipping (budget=" + budget + " spans each) ===");

        List<Sentence> kgRaw = KnowledgeGraphBioLoader.load(kg, OptionalInt.of(KG_LOAD_LIMIT), SEED);
        List<Sentence> kgClipped = targetedBudgetClip(kgRaw, budget, clipRng, "knowledge-graph");

        List<Entry> entries = LexicanumParser.parse(lex);
        Result lexResult = LexicanumCrossEntryLabeler.label(entries, DEFAULT_MIN_DENSITY, DEFAULT_MIN_ENTITY_TOKENS);
        List<Sentence> lexClipped = targetedBudgetClip(lexResult.sentences, budget, clipRng, "lexicanum");

        EldenJsonlBioLoader.Result jsonResult = EldenJsonlBioLoader.load(jsonl, OptionalInt.empty());
        List<Sentence> erClipped = targetedBudgetClip(jsonResult.sentences, budget, clipRng, "elden-ring-jsonl");

        System.out.println();

        // Combined Stage 1 corpus.
        List<Sentence> stage1 = new ArrayList<>();
        stage1.addAll(kgClipped);
        stage1.addAll(lexClipped);
        stage1.addAll(erClipped);

        // ============================================================
        // Vocab.
        // ============================================================
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence s : stage1) extraVocab.addAll(s.tokens());
        System.out.println("extra vocab: " + extraVocab.size());
        System.out.println();

        // ============================================================
        // Layer 1: POS tagger (unchanged).
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

        // Stage 1 features.
        List<double[]> pretrainInputs = new ArrayList<>();
        List<Integer> pretrainLabels = new ArrayList<>();
        for (Sentence sent : stage1) {
            List<String> forms = sent.tokens();
            int[] bio = sent.bioLabels();
            for (int i = 0; i < forms.size(); i++) {
                int lab = bio[i];
                if (lab > LABEL_I) lab = LABEL_O;
                pretrainInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                pretrainLabels.add(lab);
            }
        }

        // Stage 2 features (full hand-annotated).
        List<double[]> finetuneInputs = new ArrayList<>();
        List<Integer> finetuneLabels = new ArrayList<>();
        List<AnnotatedItem> annotated = new ArrayList<>();
        annotated.addAll(ShatteringEraAnnotations.items());
        annotated.addAll(MillicentAnnotations.items());
        annotated.addAll(DungEaterAnnotations.items());
        annotated.addAll(VolcanoManorAnnotations.items());
        int handSpans = 0;
        for (AnnotatedItem item : annotated) {
            List<OffsetToken> toks = OffsetTokenizer.tokenize(item.rawText());
            List<String> forms = toks.stream().map(OffsetToken::text).toList();
            int[] bio = BioInferenceDemo.computeBio(toks, item.spans());
            for (int i = 0; i < toks.size(); i++) {
                finetuneInputs.add(featureFor(pos, forms, i, contextDim, posDim));
                finetuneLabels.add(bio[i]);
                if (bio[i] == LABEL_B) handSpans++;
            }
        }
        System.out.printf("training: stage1-mix=%,d tokens (%,d sentences), stage2-finetune=%,d tokens (%d spans)%n",
                pretrainInputs.size(), stage1.size(), finetuneInputs.size(), handSpans);
        System.out.println();

        // Train.
        Mlp mlp = new Mlp(new int[]{entityInputDim, ENTITY_HIDDEN_DIM, NUM_LABELS}, SEED);
        ClassifierHead tagger = new ClassifierHead("bio-tagger", mlp);
        Random rng = new Random(SEED);

        System.out.println("=== Stage 1: pretrain on failure-targeted budget-clipped corpus ===");
        trainPhase(tagger, pretrainInputs, pretrainLabels, PRETRAIN_EPOCHS, PRETRAIN_LR, rng);
        System.out.println("=== Stage 2: fine-tune on hand-annotated items ===");
        trainPhase(tagger, finetuneInputs, finetuneLabels, FINETUNE_EPOCHS, FINETUNE_LR, rng);
        System.out.println();

        // ============================================================
        // Snapshot.
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
        int grandTokens = 0, grandB = 0, grandI = 0, grandSpans = 0, grandItems = 0;

        try (PrintWriter out = new PrintWriter(outPath.toFile(), java.nio.charset.StandardCharsets.UTF_8)) {
            out.println("====================================================================");
            out.println("Iter-20 failure-driven curation snapshot");
            out.println("====================================================================");
            out.println();
            out.println("Pipeline:");
            out.println("  Same architecture as iters 17 and 19 (3-class BIO, two-stage,");
            out.println("  hidden=32). Same span budget per source (" + budget + ").");
            out.println("  THE ONLY CHANGE from iter 19: which 80 spans get selected.");
            out.println();
            out.println("Mandates on training data (each addresses an iter-19 failure):");
            out.println("  1. of-the-inside    : prefer sentences with 'X of the Y' inside one span");
            out.println("                        (addresses 'Remembrance of the Baleful Shadow' split)");
            out.println("  2. two-word-propn   : prefer sentences with adjacent capitalized B-I spans");
            out.println("                        (addresses 'House Caria' truncation)");
            out.println("  3. initial-cap-as-O : prefer sentences where token 0 is capitalized O");
            out.println("                        (addresses sentence-start verb mis-tags)");
            out.println("  4. long-O-stretch   : prefer sentences with 8+ contiguous O tokens between spans");
            out.println("                        (addresses common-noun false positives like 'figure', 'trust')");
            out.println();
            out.printf("  knowledge-graph (normal)  : %,d sentences after targeted clip%n", kgClipped.size());
            out.printf("  lexicanum (fantasy)       : %,d sentences after targeted clip%n", lexClipped.size());
            out.printf("  elden-ring-jsonl          : %,d sentences after targeted clip%n", erClipped.size());
            out.printf("  hand-annotated (Stage 2)  : %d items, %d spans (full set)%n",
                    annotated.size(), handSpans);
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
            out.println();
            out.println("Compare prior runs:");
            out.println("  iter 17 (full JSONL pretrain,  258k tokens) : 191 spans");
            out.println("  iter 18 (3-corpus gradient,    517k tokens) : 212 spans");
            out.println("  iter 19 (random budget-clip,   3.3k tokens) : 199 spans");
            out.println("  iter 20 (failure-targeted clip, this run)   : " + grandSpans + " spans");
        }

        System.out.println("Snapshot written to: " + outPath.toAbsolutePath());
        System.out.printf("  %d items, %d tokens, %d spans (avg %.2f tokens/span)%n",
                grandItems, grandTokens, grandSpans,
                grandB == 0 ? 0 : (grandB + grandI) / (double) grandB);
        for (var e : perStoryStats.entrySet()) {
            int[] s = e.getValue();
            System.out.printf("  %-25s items=%2d  tokens=%4d  spans=%3d%n",
                    e.getKey(), s[4], s[0], s[3]);
        }
    }

    /**
     * Targeted budget clip: prefer sentences matching any of {@link
     * #TARGETED_PATTERNS}, fall back to random for the remainder.
     */
    private static List<Sentence> targetedBudgetClip(List<Sentence> sents, int spanBudget,
                                                       Random rng, String label) {
        // Separate by which patterns each sentence matches.
        int[] perPatternKept = new int[TARGETED_PATTERNS.size()];
        List<Sentence> targeted = new ArrayList<>();
        List<Sentence> other = new ArrayList<>();
        for (Sentence s : sents) {
            int matchIdx = -1;
            for (int i = 0; i < TARGETED_PATTERNS.size(); i++) {
                if (TARGETED_PATTERNS.get(i).test(s)) { matchIdx = i; break; }
            }
            if (matchIdx >= 0) {
                targeted.add(s);
            } else {
                other.add(s);
            }
        }
        Collections.shuffle(targeted, rng);
        Collections.shuffle(other, rng);

        List<Sentence> out = new ArrayList<>();
        int spanCount = 0;
        int targetedKept = 0;

        for (Sentence s : targeted) {
            if (spanCount >= spanBudget) break;
            int spans = 0;
            for (int b : s.bioLabels()) if (b == LABEL_B) spans++;
            if (spans == 0) continue;
            out.add(s);
            spanCount += spans;
            targetedKept++;
            // attribute which pattern (first match wins, consistent with classification above)
            for (int i = 0; i < TARGETED_PATTERNS.size(); i++) {
                if (TARGETED_PATTERNS.get(i).test(s)) { perPatternKept[i]++; break; }
            }
        }
        for (Sentence s : other) {
            if (spanCount >= spanBudget) break;
            int spans = 0;
            for (int b : s.bioLabels()) if (b == LABEL_B) spans++;
            if (spans == 0) continue;
            out.add(s);
            spanCount += spans;
        }

        System.out.printf("  targeted-clip [%-18s] kept %,d spans across %,d sents (%,d targeted, %,d random)%n",
                label, spanCount, out.size(), targetedKept, out.size() - targetedKept);
        for (int i = 0; i < TARGETED_PATTERNS.size(); i++) {
            System.out.printf("       pattern %-18s : %d sents%n", TARGETED_PATTERN_NAMES.get(i), perPatternKept[i]);
        }
        return out;
    }

    // -------- pattern matchers --------

    /** "X of the Y" appears inside a single span (I-labels on of and the). */
    private static boolean matchOfTheInsideSpan(Sentence s) {
        List<String> toks = s.tokens();
        int[] bio = s.bioLabels();
        for (int i = 0; i + 2 < toks.size(); i++) {
            if (bio[i] == LABEL_O || bio[i + 1] != LABEL_I || bio[i + 2] != LABEL_I) continue;
            if ("of".equalsIgnoreCase(toks.get(i + 1)) && "the".equalsIgnoreCase(toks.get(i + 2))) return true;
        }
        return false;
    }

    /** Two adjacent capitalized tokens labeled B-I (a two-word PROPN span). */
    private static boolean matchTwoWordPropn(Sentence s) {
        List<String> toks = s.tokens();
        int[] bio = s.bioLabels();
        for (int i = 0; i + 1 < toks.size(); i++) {
            if (bio[i] != LABEL_B || bio[i + 1] != LABEL_I) continue;
            String t1 = toks.get(i);
            String t2 = toks.get(i + 1);
            if (t1.isEmpty() || t2.isEmpty()) continue;
            if (Character.isUpperCase(t1.charAt(0)) && Character.isUpperCase(t2.charAt(0))) return true;
        }
        return false;
    }

    /** Sentence-initial capitalized O (teaches the model that token 0 isn't always B). */
    private static boolean matchInitialCapAsO(Sentence s) {
        if (s.tokens().isEmpty()) return false;
        String first = s.tokens().get(0);
        return !first.isEmpty() && Character.isUpperCase(first.charAt(0)) && s.bioLabels()[0] == LABEL_O;
    }

    /** 8+ contiguous O tokens within a sentence that also has at least one entity span. */
    private static boolean matchLongOStretch(Sentence s) {
        int[] bio = s.bioLabels();
        int run = 0, maxRun = 0;
        boolean hasEntity = false;
        for (int b : bio) {
            if (b == LABEL_O) { run++; if (run > maxRun) maxRun = run; }
            else { run = 0; if (b == LABEL_B) hasEntity = true; }
        }
        return hasEntity && maxRun >= 8;
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

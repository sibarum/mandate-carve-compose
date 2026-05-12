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
import sibarum.elden.data.BookTitlesBioLoader;
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
 * Iter 21: relaxed-pattern failure-driven curation across four corpora.
 *
 * Iter 20's patterns were too narrow — "X of the Y" found only 6 sentences
 * across all sources, and two-word PROPN dominated the budget at 76/132.
 * Iter 21 fixes both:
 *
 * <ul>
 *   <li><b>Relaxed patterns:</b> match any span containing internal
 *       lowercase 'of' / 'the' / 'in' / etc., spans with 2+ internal function
 *       words, spans of 3+ tokens, and spans whose first token matches a
 *       known failure prefix (House / Cathedral / Manus / Two / Eternal).</li>
 *   <li><b>Fourth corpus:</b> {@code book-names.parquet} added to the source
 *       pool, since book titles are 92.5% multi-word with rich internal
 *       function-word structure.</li>
 * </ul>
 *
 * Same 80 spans-per-source budget. Total stage 1 = ~320 spans across 4
 * sources (still ~80x smaller than iter 17).
 *
 * Usage:
 *   BioRelaxedCurationSnapshotDemo &lt;ud-conllu&gt; &lt;jsonl&gt; &lt;lexicanum&gt; &lt;kg-csv&gt;
 *                                   &lt;book-titles-parquet&gt;
 *                                   [span-budget] [output-file]
 */
public final class BioRelaxedCurationSnapshotDemo {

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
    private static final int KG_LOAD_LIMIT = 1000;
    private static final int BOOKS_LOAD_LIMIT = 2000;

    private static final Set<String> STOP_POS_FOR_B = Set.of(
            "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART",
            "PRON", "PUNCT", "SCONJ", "SYM", "VERB");

    private static final Set<String> FUNCTION_WORDS = Set.of(
            "of", "the", "in", "for", "to", "by", "with", "and", "or", "on", "at", "from");
    private static final Set<String> FAILURE_PREFIXES = Set.of(
            "House", "Cathedral", "Manus", "Two", "Eternal", "Forged", "Greater", "Remembrance");

    private record Storyline(String name, List<Item> items, String trainingCaption) {}

    /** Relaxed patterns. First match wins for per-pattern attribution. */
    private static final List<Predicate<Sentence>> PATTERNS = List.of(
            BioRelaxedCurationSnapshotDemo::matchAnchoredFailure,
            BioRelaxedCurationSnapshotDemo::matchMultiFuncInternal,
            BioRelaxedCurationSnapshotDemo::matchThreePlusWordSpan,
            BioRelaxedCurationSnapshotDemo::matchInternalOf,
            BioRelaxedCurationSnapshotDemo::matchInternalThe,
            BioRelaxedCurationSnapshotDemo::matchTwoWordPropn,
            BioRelaxedCurationSnapshotDemo::matchInitialCapAsO,
            BioRelaxedCurationSnapshotDemo::matchLongOStretch
    );
    private static final List<String> PATTERN_NAMES = List.of(
            "anchored-failure",
            "multi-func-internal",
            "three-plus-word",
            "internal-of",
            "internal-the",
            "two-word-propn",
            "initial-cap-as-O",
            "long-O-stretch"
    );

    public static void main(String[] args) throws IOException, ClassNotFoundException, SQLException {
        if (args.length < 5) {
            System.err.println("Usage: BioRelaxedCurationSnapshotDemo <ud-conllu> <jsonl> <lexicanum> <kg-csv>"
                    + " <book-titles-parquet> [span-budget] [output-file]");
            System.exit(1);
        }
        Path conllu = Path.of(args[0]);
        Path jsonl = Path.of(args[1]);
        Path lex = Path.of(args[2]);
        Path kg = Path.of(args[3]);
        Path books = Path.of(args[4]);
        int budget = args.length >= 6 ? Integer.parseInt(args[5]) : DEFAULT_SPAN_BUDGET;
        Path outPath = Path.of(args.length >= 7 ? args[6] : "snapshot-iter21.txt");

        Random clipRng = new Random(SEED);

        // ============================================================
        // Load + relaxed-pattern targeted-clip each corpus.
        // ============================================================
        System.out.println("=== Loading + relaxed-pattern clip (budget=" + budget + " spans each, 4 corpora) ===");

        List<Sentence> kgRaw = KnowledgeGraphBioLoader.load(kg, OptionalInt.of(KG_LOAD_LIMIT), SEED);
        List<Sentence> kgClipped = targetedBudgetClip(kgRaw, budget, clipRng, "knowledge-graph");

        List<Entry> entries = LexicanumParser.parse(lex);
        Result lexResult = LexicanumCrossEntryLabeler.label(entries, DEFAULT_MIN_DENSITY, DEFAULT_MIN_ENTITY_TOKENS);
        List<Sentence> lexClipped = targetedBudgetClip(lexResult.sentences, budget, clipRng, "lexicanum");

        EldenJsonlBioLoader.Result jsonResult = EldenJsonlBioLoader.load(jsonl, OptionalInt.empty());
        List<Sentence> erClipped = targetedBudgetClip(jsonResult.sentences, budget, clipRng, "elden-ring-jsonl");

        List<Sentence> bookRaw = BookTitlesBioLoader.load(books, OptionalInt.of(BOOKS_LOAD_LIMIT), SEED);
        List<Sentence> bookClipped = targetedBudgetClip(bookRaw, budget, clipRng, "book-titles");

        System.out.println();

        List<Sentence> stage1 = new ArrayList<>();
        stage1.addAll(kgClipped);
        stage1.addAll(lexClipped);
        stage1.addAll(erClipped);
        stage1.addAll(bookClipped);

        // ============================================================
        // Vocab + POS.
        // ============================================================
        Set<String> extraVocab = new LinkedHashSet<>(CorpusVocabulary.tokens());
        for (Sentence s : stage1) extraVocab.addAll(s.tokens());
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

        // Stage 2 features (hand-annotated).
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

        System.out.println("=== Stage 1: pretrain on relaxed-pattern budget-clipped corpus ===");
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
            out.println("Iter-21 relaxed-curation snapshot (4 corpora, 80 spans each)");
            out.println("====================================================================");
            out.println();
            out.println("Pipeline:");
            out.println("  Same architecture as iter 17 / 19 / 20.");
            out.println("  Stage 1: 4 corpora, " + budget + " spans each, relaxed pattern matching:");
            out.println("    - anchored-failure   : span starts with House/Cathedral/Manus/Two/Eternal/etc.");
            out.println("    - multi-func-internal: span contains 2+ lowercase function words inside");
            out.println("    - three-plus-word    : span of 3+ tokens");
            out.println("    - internal-of        : any span with lowercase 'of' inside");
            out.println("    - internal-the       : any span with lowercase 'the' inside");
            out.println("    - two-word-propn     : adjacent capitalized B-I pair");
            out.println("    - initial-cap-as-O   : sentence-initial capitalized token labeled O");
            out.println("    - long-O-stretch     : sentence with 8+ contiguous O between spans");
            out.println();
            out.printf("  knowledge-graph        : %,d sentences kept%n", kgClipped.size());
            out.printf("  lexicanum              : %,d sentences kept%n", lexClipped.size());
            out.printf("  elden-ring-jsonl       : %,d sentences kept%n", erClipped.size());
            out.printf("  book-titles            : %,d sentences kept%n", bookClipped.size());
            out.printf("  hand-annotated (Stage 2): %d items, %d spans (full set)%n",
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
            out.println("Compare prior:");
            out.println("  iter 17 (full JSONL pretrain,   258k tokens) : 191 spans");
            out.println("  iter 18 (3-corpus gradient,     517k tokens) : 212 spans");
            out.println("  iter 19 (random budget-clip,    3.3k tokens) : 199 spans");
            out.println("  iter 20 (targeted budget-clip,  3.7k tokens) : 187 spans");
            out.println("  iter 21 (relaxed+4-corpus,      this run)    : " + grandSpans + " spans");
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

    private static List<Sentence> targetedBudgetClip(List<Sentence> sents, int spanBudget,
                                                       Random rng, String label) {
        int[] perPatternKept = new int[PATTERNS.size()];
        List<Sentence> targeted = new ArrayList<>();
        List<Sentence> other = new ArrayList<>();
        for (Sentence s : sents) {
            int matchIdx = -1;
            for (int i = 0; i < PATTERNS.size(); i++) {
                if (PATTERNS.get(i).test(s)) { matchIdx = i; break; }
            }
            if (matchIdx >= 0) targeted.add(s); else other.add(s);
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
            for (int i = 0; i < PATTERNS.size(); i++) {
                if (PATTERNS.get(i).test(s)) { perPatternKept[i]++; break; }
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
        System.out.printf("  relaxed-clip [%-18s] kept %,d spans across %,d sents (%,d targeted, %,d random)%n",
                label, spanCount, out.size(), targetedKept, out.size() - targetedKept);
        for (int i = 0; i < PATTERNS.size(); i++) {
            if (perPatternKept[i] > 0) {
                System.out.printf("       pattern %-22s : %d sents%n",
                        PATTERN_NAMES.get(i), perPatternKept[i]);
            }
        }
        return out;
    }

    // -------- relaxed pattern matchers --------

    /** Span starts with one of the iter-19/20 failure prefixes (House, Cathedral, etc.). */
    private static boolean matchAnchoredFailure(Sentence s) {
        List<String> toks = s.tokens();
        int[] bio = s.bioLabels();
        for (int i = 0; i + 1 < toks.size(); i++) {
            if (bio[i] != LABEL_B) continue;
            if (!FAILURE_PREFIXES.contains(toks.get(i))) continue;
            if (bio[i + 1] == LABEL_I) return true;
        }
        return false;
    }

    /** A span contains 2+ lowercase function-word tokens labeled I. */
    private static boolean matchMultiFuncInternal(Sentence s) {
        List<String> toks = s.tokens();
        int[] bio = s.bioLabels();
        int funcInside = 0;
        int run = 0;
        for (int i = 0; i < toks.size(); i++) {
            if (bio[i] == LABEL_O) { run = 0; funcInside = 0; continue; }
            run++;
            if (FUNCTION_WORDS.contains(toks.get(i).toLowerCase())) funcInside++;
            if (funcInside >= 2) return true;
        }
        return false;
    }

    /** A B-I+ span of total length 3+. */
    private static boolean matchThreePlusWordSpan(Sentence s) {
        int[] bio = s.bioLabels();
        int run = 0;
        for (int b : bio) {
            if (b == LABEL_B) run = 1;
            else if (b == LABEL_I) run++;
            else { if (run >= 3) return true; run = 0; }
        }
        return run >= 3;
    }

    /** A span contains the lowercase token 'of' labeled I. */
    private static boolean matchInternalOf(Sentence s) {
        List<String> toks = s.tokens();
        int[] bio = s.bioLabels();
        for (int i = 0; i < toks.size(); i++) {
            if (bio[i] == LABEL_I && "of".equalsIgnoreCase(toks.get(i))) return true;
        }
        return false;
    }

    /** A span contains the lowercase token 'the' labeled I. */
    private static boolean matchInternalThe(Sentence s) {
        List<String> toks = s.tokens();
        int[] bio = s.bioLabels();
        for (int i = 0; i < toks.size(); i++) {
            if (bio[i] == LABEL_I && "the".equalsIgnoreCase(toks.get(i))) return true;
        }
        return false;
    }

    private static boolean matchTwoWordPropn(Sentence s) {
        List<String> toks = s.tokens();
        int[] bio = s.bioLabels();
        for (int i = 0; i + 1 < toks.size(); i++) {
            if (bio[i] != LABEL_B || bio[i + 1] != LABEL_I) continue;
            String t1 = toks.get(i), t2 = toks.get(i + 1);
            if (t1.isEmpty() || t2.isEmpty()) continue;
            if (Character.isUpperCase(t1.charAt(0)) && Character.isUpperCase(t2.charAt(0))) return true;
        }
        return false;
    }

    private static boolean matchInitialCapAsO(Sentence s) {
        if (s.tokens().isEmpty()) return false;
        String first = s.tokens().get(0);
        return !first.isEmpty() && Character.isUpperCase(first.charAt(0)) && s.bioLabels()[0] == LABEL_O;
    }

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

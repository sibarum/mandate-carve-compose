package sibarum.elden.pos;

import sibarum.elden.pos.ConlluParser.Sentence;
import sibarum.elden.pos.ConlluParser.TaggedToken;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * End-to-end trainer for the POS layer. Trains a {@link ClassifierHead} to
 * predict universal POS tags from sliding-window context vectors, AND
 * propagates gradients back through the context window into the
 * {@link SymbolEmbeddingTable} so the embeddings themselves learn POS-relevant
 * features.
 *
 * Backprop chain per token:
 *   classifier loss
 *     -> ClassifierHead.backward()           [updates MLP weights]
 *     -> classifier.inputGradient()          [gradient w.r.t. context vector]
 *     -> slice into per-position chunks      [each chunk = grad w.r.t. one token]
 *     -> SymbolEmbeddingTable.update(token, embed-portion, lr)
 *
 * For per-position word-shape features (if enabled) the gradient slice has the
 * embedding portion routed back into the table and the shape-feature portion
 * discarded — shape features are deterministic, not learnable.
 *
 * Eval discipline: the trainer holds out the last 10% of sentences as a dev
 * split, reports dev accuracy after every epoch (so overfitting is visible),
 * and at the end prints per-tag precision / recall plus the top off-diagonal
 * confusions — the diagnostic that names which structural mandate to add next.
 */
public final class PosTrainer {

    /** Fraction of input sentences reserved for evaluation (taken from the tail). */
    public static final double DEV_FRACTION = 0.10;

    /** Word-shape feature dimensionality appended per token if shape features are on. */
    public static final int SHAPE_DIM = 6;

    public static final class TrainedPosLayer {
        public final SymbolEmbeddingTable table;
        public final ClassifierHead classifier;
        public final int embedDim;
        public final int windowRadius;
        public final int perTokenDim;   // embedDim, or embedDim + SHAPE_DIM if shape features on
        public final int contextDim;
        public final boolean useShapeFeatures;
        /** Bigram transitions for Viterbi decode; null if greedy-only. */
        public PosTransitions transitions;
        /** Weight applied to transition log-probs during Viterbi decode. */
        public double viterbiWeight;

        TrainedPosLayer(SymbolEmbeddingTable table, ClassifierHead classifier,
                        int embedDim, int windowRadius, boolean useShapeFeatures) {
            this.table = table;
            this.classifier = classifier;
            this.embedDim = embedDim;
            this.windowRadius = windowRadius;
            this.useShapeFeatures = useShapeFeatures;
            this.perTokenDim = embedDim + (useShapeFeatures ? SHAPE_DIM : 0);
            this.contextDim = (2 * windowRadius + 1) * perTokenDim;
        }

        /** Build the context vector for a position in a sequence. */
        public double[] contextOf(List<String> sentence, int position) {
            return buildContext(sentence, position, table, windowRadius, embedDim, perTokenDim, contextDim, useShapeFeatures);
        }

        /** Compute emission logits for one position. */
        public double[] emissionsAt(List<String> sentence, int position) {
            double[] ctx = contextOf(sentence, position);
            return ((MatrixValue) classifier.apply(List.of(new MatrixValue(ctx)))).data();
        }

        /** Greedy argmax POS tag for one position (no transitions). */
        public String predict(List<String> sentence, int position) {
            double[] logits = emissionsAt(sentence, position);
            int best = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[best]) best = i;
            }
            return PosTagset.tagAt(best);
        }

        /**
         * Sequence prediction. Uses Viterbi if {@link #transitions} is set;
         * falls back to per-position greedy argmax otherwise. Returns one
         * tag-index per token.
         */
        public int[] predictSequence(List<String> sentence) {
            int T = sentence.size();
            int K = PosTagset.size();
            double[][] emissions = new double[T][K];
            for (int i = 0; i < T; i++) emissions[i] = emissionsAt(sentence, i);
            if (transitions != null) {
                return ViterbiDecoder.decode(emissions, transitions, viterbiWeight);
            }
            int[] out = new int[T];
            for (int i = 0; i < T; i++) {
                int best = 0;
                for (int k = 1; k < K; k++) if (emissions[i][k] > emissions[i][best]) best = k;
                out[i] = best;
            }
            return out;
        }
    }

    public static final class EvalResult {
        public final double accuracy;
        public final int[][] confusion;  // [gold][pred]
        public final int totalExamples;

        EvalResult(double accuracy, int[][] confusion, int totalExamples) {
            this.accuracy = accuracy;
            this.confusion = confusion;
            this.totalExamples = totalExamples;
        }
    }

    /**
     * Train with default architecture (embedding-only per-token features, no
     * word-shape features). Held-out dev split is taken from the tail 10% of
     * {@code sentences}.
     */
    public static TrainedPosLayer train(List<Sentence> sentences,
                                         Set<String> additionalVocab,
                                         int embedDim,
                                         int windowRadius,
                                         int hiddenDim,
                                         long seed,
                                         int epochs,
                                         double lr) {
        return train(sentences, additionalVocab, embedDim, windowRadius,
                new int[]{hiddenDim}, seed, epochs, lr, false);
    }

    /**
     * Train with explicit hidden-layer shape, optional word-shape features.
     * {@code hiddenSizes} is the list of hidden-layer widths between input
     * and the 17-way output; pass {@code {64}} for one hidden layer of 64,
     * or {@code {128, 64}} for two hidden layers.
     */
    public static TrainedPosLayer train(List<Sentence> sentences,
                                         Set<String> additionalVocab,
                                         int embedDim,
                                         int windowRadius,
                                         int[] hiddenSizes,
                                         long seed,
                                         int epochs,
                                         double lr,
                                         boolean useShapeFeatures) {

        // ----- 90 / 10 train / dev split -----
        int devCount = (int) Math.round(sentences.size() * DEV_FRACTION);
        if (devCount < 1) devCount = 1;
        int trainEnd = sentences.size() - devCount;
        List<Sentence> trainSents = sentences.subList(0, trainEnd);
        List<Sentence> devSents = sentences.subList(trainEnd, sentences.size());

        // ----- Vocab: every UD form (train + dev) plus extras. -----
        // Dev tokens get random embeddings; they're never gradient-updated
        // because dev sentences are never iterated for training. This is the
        // intended inference-time behavior for novel tokens.
        Set<String> vocab = new LinkedHashSet<>();
        for (Sentence s : sentences) for (TaggedToken t : s.tokens()) vocab.add(t.form());
        vocab.addAll(additionalVocab);

        SymbolEmbeddingTable table = new SymbolEmbeddingTable(embedDim, seed);
        for (String tok : vocab) table.embed(tok);

        int perTokenDim = embedDim + (useShapeFeatures ? SHAPE_DIM : 0);
        int contextDim = (2 * windowRadius + 1) * perTokenDim;
        int[] mlpSizes = new int[hiddenSizes.length + 2];
        mlpSizes[0] = contextDim;
        System.arraycopy(hiddenSizes, 0, mlpSizes, 1, hiddenSizes.length);
        mlpSizes[mlpSizes.length - 1] = PosTagset.size();

        Mlp mlp = new Mlp(mlpSizes, seed);
        ClassifierHead classifier = new ClassifierHead("pos-tagger", mlp);
        TrainedPosLayer layer = new TrainedPosLayer(table, classifier, embedDim, windowRadius, useShapeFeatures);

        // Flatten train sentences into (sentence, position, tag-class) triples.
        List<int[]> indices = new ArrayList<>();
        for (int si = 0; si < trainSents.size(); si++) {
            List<TaggedToken> toks = trainSents.get(si).tokens();
            for (int pi = 0; pi < toks.size(); pi++) {
                int cls = PosTagset.indexOf(toks.get(pi).upos());
                if (cls < 0) continue;
                indices.add(new int[]{si, pi, cls});
            }
        }
        System.out.println("POS training examples: " + indices.size()
                + " across " + trainSents.size() + " train sentences"
                + " (held-out dev sentences: " + devSents.size() + ","
                + " vocab=" + vocab.size() + ", contextDim=" + contextDim
                + ", arch=" + Arrays.toString(mlpSizes)
                + ", shapeFeats=" + useShapeFeatures + ")");

        Random rng = new Random(seed);
        double[] target = new double[PosTagset.size()];

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(indices, rng);
            double totalLoss = 0;
            int correct = 0;
            for (int[] ix : indices) {
                Sentence s = trainSents.get(ix[0]);
                int pos = ix[1];
                int cls = ix[2];

                List<String> sentenceForms = formsOf(s);
                double[] ctx = buildContext(sentenceForms, pos, table, windowRadius,
                        embedDim, perTokenDim, contextDim, useShapeFeatures);

                // Forward.
                double[] logits = ((MatrixValue) classifier.apply(List.of(new MatrixValue(ctx)))).data();

                // Loss + argmax accuracy on the training example.
                Arrays.fill(target, 0.0);
                target[cls] = 1.0;
                for (int i = 0; i < logits.length; i++) {
                    double d = logits[i] - target[i];
                    totalLoss += d * d;
                }
                int best = 0;
                for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
                if (best == cls) correct++;

                classifier.backward(new MatrixValue(target));
                classifier.step(lr);

                double[] gradCtx = classifier.inputGradient();
                propagateToEmbeddings(gradCtx, sentenceForms, pos, table,
                        windowRadius, embedDim, perTokenDim, lr);
            }
            double avgLoss = totalLoss / indices.size();
            double trainAcc = (double) correct / indices.size();

            EvalResult dev = evaluate(layer, devSents);
            System.out.printf("  epoch %2d  loss=%.4f  train_acc=%.3f  dev_acc=%.3f%n",
                    epoch + 1, avgLoss, trainAcc, dev.accuracy);
        }

        // Final report: per-tag P/R and top confusions on the dev set (greedy).
        System.out.println();
        EvalResult finalEval = evaluate(layer, devSents);
        printConfusionReport(finalEval);

        // Bigram transitions from the train split + Viterbi weight sweep on dev.
        layer.transitions = PosTransitions.fromSentences(trainSents);
        double[] sweep = {0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0};
        System.out.println();
        System.out.println("Viterbi transition-weight sweep (dev set):");
        double bestWeight = 0.0;
        double bestAcc = finalEval.accuracy;
        for (double w : sweep) {
            layer.viterbiWeight = w;
            EvalResult r = evaluateSequence(layer, devSents);
            String marker = r.accuracy > bestAcc ? "  <- best" : "";
            System.out.printf("    transWeight=%.2f   dev_acc=%.4f%s%n", w, r.accuracy, marker);
            if (r.accuracy > bestAcc) { bestAcc = r.accuracy; bestWeight = w; }
        }
        layer.viterbiWeight = bestWeight;
        System.out.printf("Selected transWeight=%.2f (dev_acc=%.4f, vs greedy %.4f).%n",
                bestWeight, bestAcc, finalEval.accuracy);
        if (bestWeight > 0.0) {
            EvalResult finalViterbi = evaluateSequence(layer, devSents);
            System.out.println();
            printConfusionReport(finalViterbi);
        }

        return layer;
    }

    /**
     * Evaluate the layer in sequence (Viterbi) mode using the layer's current
     * {@code transitions} and {@code viterbiWeight}. Falls back to per-position
     * greedy if transitions are null.
     */
    public static EvalResult evaluateSequence(TrainedPosLayer layer, List<Sentence> sentences) {
        int n = PosTagset.size();
        int[][] cm = new int[n][n];
        int correct = 0, total = 0;
        for (Sentence s : sentences) {
            List<String> forms = formsOf(s);
            int[] pred = layer.predictSequence(forms);
            for (int i = 0; i < forms.size(); i++) {
                int gold = PosTagset.indexOf(s.tokens().get(i).upos());
                if (gold < 0) continue;
                int p = pred[i];
                cm[gold][p]++;
                if (p == gold) correct++;
                total++;
            }
        }
        double acc = total == 0 ? 0.0 : (double) correct / total;
        return new EvalResult(acc, cm, total);
    }

    /** Evaluate a trained layer on a list of sentences. */
    public static EvalResult evaluate(TrainedPosLayer layer, List<Sentence> sentences) {
        int n = PosTagset.size();
        int[][] cm = new int[n][n];
        int correct = 0, total = 0;
        for (Sentence s : sentences) {
            List<String> forms = formsOf(s);
            for (int i = 0; i < forms.size(); i++) {
                int gold = PosTagset.indexOf(s.tokens().get(i).upos());
                if (gold < 0) continue;
                double[] ctx = layer.contextOf(forms, i);
                double[] logits = ((MatrixValue) layer.classifier.apply(List.of(new MatrixValue(ctx)))).data();
                int pred = 0;
                for (int k = 1; k < logits.length; k++) if (logits[k] > logits[pred]) pred = k;
                cm[gold][pred]++;
                if (pred == gold) correct++;
                total++;
            }
        }
        double acc = total == 0 ? 0.0 : (double) correct / total;
        return new EvalResult(acc, cm, total);
    }

    /** Print per-tag P/R and the worst off-diagonal confusions. */
    public static void printConfusionReport(EvalResult r) {
        int n = PosTagset.size();
        System.out.println("dev set per-tag metrics (n=" + r.totalExamples
                + ", overall acc=" + String.format("%.3f", r.accuracy) + "):");
        System.out.printf("    %-7s %6s %6s %6s %8s%n", "tag", "P", "R", "F1", "support");
        for (int g = 0; g < n; g++) {
            int tp = r.confusion[g][g];
            int rowSum = 0, colSum = 0;
            for (int j = 0; j < n; j++) { rowSum += r.confusion[g][j]; colSum += r.confusion[j][g]; }
            if (rowSum == 0 && colSum == 0) continue;
            double prec = colSum > 0 ? (double) tp / colSum : 0;
            double rec = rowSum > 0 ? (double) tp / rowSum : 0;
            double f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
            System.out.printf("    %-7s %6.3f %6.3f %6.3f %8d%n",
                    PosTagset.tagAt(g), prec, rec, f1, rowSum);
        }

        System.out.println("top off-diagonal confusions (gold -> predicted):");
        List<int[]> pairs = new ArrayList<>();
        for (int g = 0; g < n; g++) {
            for (int p = 0; p < n; p++) {
                if (g != p && r.confusion[g][p] > 0) {
                    pairs.add(new int[]{g, p, r.confusion[g][p]});
                }
            }
        }
        pairs.sort(Comparator.<int[]>comparingInt(a -> a[2]).reversed());
        int limit = Math.min(10, pairs.size());
        for (int k = 0; k < limit; k++) {
            int[] row = pairs.get(k);
            int goldTotal = 0;
            for (int j = 0; j < n; j++) goldTotal += r.confusion[row[0]][j];
            double pct = goldTotal > 0 ? 100.0 * row[2] / goldTotal : 0;
            System.out.printf("    %-6s -> %-6s   %4d   (%5.1f%% of gold %s)%n",
                    PosTagset.tagAt(row[0]), PosTagset.tagAt(row[1]), row[2],
                    pct, PosTagset.tagAt(row[0]));
        }
    }

    private static List<String> formsOf(Sentence s) {
        List<String> out = new ArrayList<>(s.tokens().size());
        for (TaggedToken t : s.tokens()) out.add(t.form());
        return out;
    }

    /**
     * Build the per-position context vector. For each of (2 * radius + 1) slots
     * the per-token chunk is [ embedDim values from the table | optional SHAPE_DIM
     * deterministic features ]. Out-of-range slots contribute zeros.
     */
    private static double[] buildContext(List<String> sentence, int position,
                                          SymbolEmbeddingTable table, int windowRadius,
                                          int embedDim, int perTokenDim, int contextDim,
                                          boolean useShapeFeatures) {
        double[] out = new double[contextDim];
        for (int offset = -windowRadius; offset <= windowRadius; offset++) {
            int idx = position + offset;
            int slot = offset + windowRadius;
            if (idx >= 0 && idx < sentence.size()) {
                String tok = sentence.get(idx);
                double[] emb = table.embed(tok);
                System.arraycopy(emb, 0, out, slot * perTokenDim, embedDim);
                if (useShapeFeatures) {
                    double[] shape = shapeFeatures(tok);
                    System.arraycopy(shape, 0, out, slot * perTokenDim + embedDim, SHAPE_DIM);
                }
            }
        }
        return out;
    }

    /**
     * Six orthogonal word-shape signals. The mandate behind each is to give the
     * classifier a hard, deterministic feature for a property the embedding
     * alone struggles to encode (e.g. capitalization is identity-erased once
     * the token-to-vector lookup happens — "Apple" and "apple" share an
     * embedding slot if we lowercase, or sit at unrelated random points if we
     * don't, but neither lookup tells the classifier "this is capitalized").
     *
     * Index | Mandate
     *   0   | first character is uppercase   (PROPN vs NOUN)
     *   1   | every character is uppercase   (acronyms / SYM)
     *   2   | contains a digit               (NUM)
     *   3   | contains a hyphen              (compound adjectives)
     *   4   | is pure punctuation            (PUNCT)
     *   5   | length bucket (log scale)      (DET / ADP tend short)
     */
    static double[] shapeFeatures(String tok) {
        double[] f = new double[SHAPE_DIM];
        if (tok.isEmpty()) return f;
        char c0 = tok.charAt(0);
        boolean firstUpper = Character.isUpperCase(c0);
        boolean allUpper = true;
        boolean anyDigit = false;
        boolean anyHyphen = false;
        boolean allPunct = true;
        for (int i = 0; i < tok.length(); i++) {
            char ch = tok.charAt(i);
            if (Character.isLetter(ch)) {
                allPunct = false;
                if (!Character.isUpperCase(ch)) allUpper = false;
            } else {
                allUpper = false;
                if (Character.isDigit(ch)) { anyDigit = true; allPunct = false; }
                else if (ch == '-') anyHyphen = true;
                else if (Character.isLetterOrDigit(ch)) allPunct = false;
            }
        }
        f[0] = firstUpper ? 1.0 : 0.0;
        f[1] = allUpper && tok.length() > 1 ? 1.0 : 0.0;
        f[2] = anyDigit ? 1.0 : 0.0;
        f[3] = anyHyphen ? 1.0 : 0.0;
        f[4] = allPunct ? 1.0 : 0.0;
        f[5] = Math.log(1 + tok.length()) / Math.log(20);  // ~0..1 range
        return f;
    }

    /**
     * Backprop the context-gradient back into the embedding table. Only the
     * embedding portion of each per-token chunk is routed back — the shape
     * portion (if present) is deterministic and has no learnable parameters.
     */
    private static void propagateToEmbeddings(double[] gradCtx, List<String> sentence,
                                                int position, SymbolEmbeddingTable table,
                                                int windowRadius, int embedDim,
                                                int perTokenDim, double lr) {
        if (gradCtx == null) return;
        for (int offset = -windowRadius; offset <= windowRadius; offset++) {
            int idx = position + offset;
            int slot = offset + windowRadius;
            if (idx < 0 || idx >= sentence.size()) continue;
            double[] tokenGrad = new double[embedDim];
            System.arraycopy(gradCtx, slot * perTokenDim, tokenGrad, 0, embedDim);
            table.update(sentence.get(idx), tokenGrad, lr);
        }
    }
}

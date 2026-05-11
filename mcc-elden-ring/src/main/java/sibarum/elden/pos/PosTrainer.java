package sibarum.elden.pos;

import sibarum.elden.pos.ConlluParser.Sentence;
import sibarum.elden.pos.ConlluParser.TaggedToken;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;

import java.util.ArrayList;
import java.util.Collections;
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
 *     -> classifier.inputGradient()          [gradient w.r.t. 96-dim context vector]
 *     -> slice into per-position chunks      [each chunk = grad w.r.t. one token's embedding]
 *     -> SymbolEmbeddingTable.update(token, chunk, lr)
 *
 * This is the framework's first multi-layer trained chain where gradients
 * flow back into a KV table — the realization of the multi-KV story
 * discussed in the design phase.
 */
public final class PosTrainer {

    public static final class TrainedPosLayer {
        public final SymbolEmbeddingTable table;
        public final ClassifierHead classifier;
        public final int embedDim;
        public final int windowRadius;
        public final int contextDim;

        TrainedPosLayer(SymbolEmbeddingTable table, ClassifierHead classifier,
                        int embedDim, int windowRadius, int contextDim) {
            this.table = table;
            this.classifier = classifier;
            this.embedDim = embedDim;
            this.windowRadius = windowRadius;
            this.contextDim = contextDim;
        }

        /** Build the context vector for a position in a sequence. */
        public double[] contextOf(List<String> sentence, int position) {
            double[] out = new double[contextDim];
            for (int offset = -windowRadius; offset <= windowRadius; offset++) {
                int idx = position + offset;
                int slot = offset + windowRadius;
                if (idx >= 0 && idx < sentence.size()) {
                    double[] emb = table.embed(sentence.get(idx));
                    System.arraycopy(emb, 0, out, slot * embedDim, embedDim);
                }
            }
            return out;
        }

        /** Predict POS tag for one token at the given position. */
        public String predict(List<String> sentence, int position) {
            double[] ctx = contextOf(sentence, position);
            double[] logits = ((MatrixValue) classifier.apply(List.of(new MatrixValue(ctx)))).data();
            int best = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[best]) best = i;
            }
            return PosTagset.tagAt(best);
        }
    }

    public static TrainedPosLayer train(List<Sentence> sentences,
                                         Set<String> additionalVocab,
                                         int embedDim,
                                         int windowRadius,
                                         int hiddenDim,
                                         long seed,
                                         int epochs,
                                         double lr) {
        // Vocabulary: every UD form plus any extra vocab (e.g. Elden Ring tokens
        // we'll want to apply the trained tagger to later).
        Set<String> vocab = new LinkedHashSet<>();
        for (Sentence s : sentences) for (TaggedToken t : s.tokens()) vocab.add(t.form());
        vocab.addAll(additionalVocab);

        SymbolEmbeddingTable table = new SymbolEmbeddingTable(embedDim, seed);
        for (String tok : vocab) table.embed(tok);

        int contextDim = (2 * windowRadius + 1) * embedDim;
        Mlp mlp = new Mlp(new int[]{contextDim, hiddenDim, PosTagset.size()}, seed);
        ClassifierHead classifier = new ClassifierHead("pos-tagger", mlp);

        // Flatten sentences into (sentence, position, tag-class) triples.
        List<int[]> indices = new ArrayList<>();  // [sentenceIdx, position, classIdx]
        for (int si = 0; si < sentences.size(); si++) {
            List<TaggedToken> toks = sentences.get(si).tokens();
            for (int pi = 0; pi < toks.size(); pi++) {
                int cls = PosTagset.indexOf(toks.get(pi).upos());
                if (cls < 0) continue;
                indices.add(new int[]{si, pi, cls});
            }
        }
        System.out.println("POS training examples: " + indices.size()
                + " across " + sentences.size() + " sentences"
                + " (vocab=" + vocab.size() + ", contextDim=" + contextDim + ")");

        Random rng = new Random(seed);
        double[] target = new double[PosTagset.size()];

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(indices, rng);
            double totalLoss = 0;
            int correct = 0;
            for (int[] ix : indices) {
                Sentence s = sentences.get(ix[0]);
                int pos = ix[1];
                int cls = ix[2];

                List<String> sentenceForms = formsOf(s);
                double[] ctx = buildContext(sentenceForms, pos, table, windowRadius, embedDim, contextDim);

                // Forward.
                double[] logits = ((MatrixValue) classifier.apply(List.of(new MatrixValue(ctx)))).data();

                // Loss + argmax accuracy.
                java.util.Arrays.fill(target, 0.0);
                target[cls] = 1.0;
                for (int i = 0; i < logits.length; i++) {
                    double d = logits[i] - target[i];
                    totalLoss += d * d;
                }
                int best = 0;
                for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
                if (best == cls) correct++;

                // Backward through the classifier.
                classifier.backward(new MatrixValue(target));

                // Step the classifier weights.
                classifier.step(lr);

                // Propagate gradient back into the embedding table.
                double[] gradCtx = classifier.inputGradient();
                propagateToEmbeddings(gradCtx, sentenceForms, pos, table, windowRadius, embedDim, lr);
            }
            double avgLoss = totalLoss / indices.size();
            double acc = (double) correct / indices.size();
            if (epoch == 0 || (epoch + 1) % 1 == 0) {
                System.out.printf("  epoch %2d  loss=%.4f  accuracy=%.3f%n", epoch + 1, avgLoss, acc);
            }
        }

        return new TrainedPosLayer(table, classifier, embedDim, windowRadius, contextDim);
    }

    private static List<String> formsOf(Sentence s) {
        List<String> out = new ArrayList<>(s.tokens().size());
        for (TaggedToken t : s.tokens()) out.add(t.form());
        return out;
    }

    private static double[] buildContext(List<String> sentence, int position,
                                          SymbolEmbeddingTable table, int windowRadius,
                                          int embedDim, int contextDim) {
        double[] out = new double[contextDim];
        for (int offset = -windowRadius; offset <= windowRadius; offset++) {
            int idx = position + offset;
            int slot = offset + windowRadius;
            if (idx >= 0 && idx < sentence.size()) {
                double[] emb = table.embed(sentence.get(idx));
                System.arraycopy(emb, 0, out, slot * embedDim, embedDim);
            }
        }
        return out;
    }

    private static void propagateToEmbeddings(double[] gradCtx, List<String> sentence,
                                                int position, SymbolEmbeddingTable table,
                                                int windowRadius, int embedDim, double lr) {
        if (gradCtx == null) return;
        for (int offset = -windowRadius; offset <= windowRadius; offset++) {
            int idx = position + offset;
            int slot = offset + windowRadius;
            if (idx < 0 || idx >= sentence.size()) continue;
            double[] tokenGrad = new double[embedDim];
            System.arraycopy(gradCtx, slot * embedDim, tokenGrad, 0, embedDim);
            table.update(sentence.get(idx), tokenGrad, lr);
        }
    }
}

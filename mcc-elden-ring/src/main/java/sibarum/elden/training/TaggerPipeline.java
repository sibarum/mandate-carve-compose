package sibarum.elden.training;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.embedding.ContextEncoder;
import sibarum.elden.embedding.CorpusVocabulary;
import sibarum.elden.training.TaggingTrainingData.Example;
import sibarum.strnn.cache.SymbolEmbeddingTable;
import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.ClassifierHead;
import sibarum.strnn.value.MatrixValue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Reusable setup + training pipeline for the binary span tagger. Holds the
 * embedding table, context encoder, and trained classifier head together so
 * inference code can reuse the trained state without re-running training.
 */
public final class TaggerPipeline {

    public final SymbolEmbeddingTable table;
    public final ContextEncoder ctx;
    public final ClassifierHead tagger;
    public final int contextDim;
    public final double threshold;

    private TaggerPipeline(SymbolEmbeddingTable table, ContextEncoder ctx, ClassifierHead tagger, int contextDim, double threshold) {
        this.table = table;
        this.ctx = ctx;
        this.tagger = tagger;
        this.contextDim = contextDim;
        this.threshold = threshold;
    }

    public static TaggerPipeline trainDefault() {
        return train(32, 1, 32, 42L, 60, 0.01, 0.5);
    }

    public static TaggerPipeline train(int embedDim, int windowRadius, int hiddenDim, long seed, int epochs, double lr, double threshold) {
        var vocab = CorpusVocabulary.tokens();
        SymbolEmbeddingTable table = new SymbolEmbeddingTable(embedDim, seed);
        for (String t : vocab) table.embed(t);

        ContextEncoder ctx = new ContextEncoder(table, windowRadius);
        int contextDim = ctx.contextDim();

        List<AnnotatedItem> items = new ArrayList<>();
        items.addAll(ShatteringEraAnnotations.items());
        items.addAll(MillicentAnnotations.items());
        items.addAll(DungEaterAnnotations.items());
        items.addAll(VolcanoManorAnnotations.items());

        List<Example> examples = TaggingTrainingData.extractAll(items, ctx);

        Mlp mlp = new Mlp(new int[]{contextDim, hiddenDim, 1}, seed);
        ClassifierHead tagger = new ClassifierHead("span-tagger", mlp);

        Random rng = new Random(seed);
        List<Example> shuffled = new ArrayList<>(examples);
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(shuffled, rng);
            for (Example ex : shuffled) {
                tagger.apply(List.of(new MatrixValue(ex.contextVector())));
                tagger.backward(new MatrixValue(new double[]{ex.label()}));
                tagger.step(lr);
            }
        }

        return new TaggerPipeline(table, ctx, tagger, contextDim, threshold);
    }

    public double score(double[] contextVector) {
        var out = (MatrixValue) tagger.apply(List.of(new MatrixValue(contextVector)));
        return out.data()[0];
    }
}

package sibarum.mcc.primitive;

import sibarum.mcc.op.Add;
import sibarum.mcc.op.Concat;
import sibarum.mcc.op.CosineSimilarity;
import sibarum.mcc.op.CrossProduct3;
import sibarum.mcc.op.DotProduct;
import sibarum.mcc.op.Linear;
import sibarum.mcc.op.Magnitude;
import sibarum.mcc.op.Mul;
import sibarum.mcc.op.Normalize;
import sibarum.mcc.op.Relu;
import sibarum.mcc.op.Sigmoid;
import sibarum.mcc.op.SimilarityGate;
import sibarum.mcc.op.Slice;
import sibarum.mcc.op.Softmax;
import sibarum.mcc.op.Sub;
import sibarum.mcc.op.Tanh;
import sibarum.mcc.op.block.MlpBlock;

import java.util.List;
import java.util.Map;

/**
 * Static registry of mcc-core's built-in primitives. The importer
 * uses this to reconstruct primitives from their serialized names +
 * configs.
 *
 * <p>Stateless primitives ignore their config map. Parameterized
 * primitives read structural parameters (sizes, seed, etc.) from it.
 * Note: the {@code seed} a Trainable was constructed with affects
 * <em>initialization only</em>; after parameters are loaded from a
 * params blob, the seed is irrelevant.
 */
public final class BuiltinPrimitives {

    private BuiltinPrimitives() {}

    public static PrimitiveRegistry defaults() {
        PrimitiveRegistry r = new PrimitiveRegistry();

        // Stateless ops: ignore config.
        r.register("add", cfg -> new Add());
        r.register("sub", cfg -> new Sub());
        r.register("mul", cfg -> new Mul());
        r.register("dot-product", cfg -> new DotProduct());
        r.register("cosine-similarity", cfg -> new CosineSimilarity());
        r.register("similarity-gate", cfg -> new SimilarityGate());
        r.register("softmax", cfg -> new Softmax());
        r.register("magnitude", cfg -> new Magnitude());
        r.register("normalize", cfg -> new Normalize());
        r.register("cross-product-3", cfg -> new CrossProduct3());
        r.register("relu", cfg -> new Relu());
        r.register("sigmoid", cfg -> new Sigmoid());
        r.register("tanh", cfg -> new Tanh());
        r.register("concat", cfg -> new Concat());

        // Slice has a structural config.
        r.register("slice", cfg -> {
            int from = ((Number) requireField(cfg, "from")).intValue();
            int to = ((Number) requireField(cfg, "to")).intValue();
            return new Slice(from, to);
        });

        // Trainables: structural config selects shape; parameters are restored
        // from the params blob after construction.
        r.register("linear", cfg -> {
            int outDim = ((Number) requireField(cfg, "outDim")).intValue();
            int inDim = ((Number) requireField(cfg, "inDim")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new Linear(outDim, inDim, true, seed);
        });
        r.register("linear-no-bias", cfg -> {
            int outDim = ((Number) requireField(cfg, "outDim")).intValue();
            int inDim = ((Number) requireField(cfg, "inDim")).intValue();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new Linear(outDim, inDim, false, seed);
        });
        r.register("mlp-block", cfg -> {
            @SuppressWarnings("unchecked")
            List<Number> rawSizes = (List<Number>) requireField(cfg, "sizes");
            int[] sizes = rawSizes.stream().mapToInt(Number::intValue).toArray();
            long seed = ((Number) cfg.getOrDefault("seed", 0L)).longValue();
            return new MlpBlock(sizes, seed);
        });

        return r;
    }

    private static Object requireField(Map<String, Object> cfg, String key) {
        Object v = cfg.get(key);
        if (v == null) throw new IllegalArgumentException("primitive config missing field: " + key);
        return v;
    }
}

package sibarum.strnn.demo;

import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.transformer.Transformer;

import java.util.Random;

/**
 * Shared offline pretraining for the v0 demos. Trains a fresh Mlp on add or
 * multiply over single-digit pairs, in the same encoding NumberToMatrix /
 * MatrixToNumber use. Lives in the demo package because both TrainingDemo and
 * AblationDemo bootstrap MLPs the same way.
 */
public final class TrainingDemoBootstrap {
    private TrainingDemoBootstrap() {
    }

    public static void pretrain(Mlp mlp, MlpRole role, int epochs, int batch, double lr) {
        pretrainImpl(mlp, role, epochs, batch, lr, false);
    }

    public static void pretrainSilent(Mlp mlp, MlpRole role, int epochs, int batch, double lr) {
        pretrainImpl(mlp, role, epochs, batch, lr, true);
    }

    private static void pretrainImpl(Mlp mlp, MlpRole role, int epochs, int batch, double lr, boolean silent) {
        Random rng = new Random(role == MlpRole.ADD ? 7L : 13L);
        double scale = NumberToMatrix.SCALE;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double sum = 0;
            for (int i = 0; i < batch; i++) {
                int a = rng.nextInt(10);
                int b = rng.nextInt(10);
                double[] in = {a / scale, b / scale};
                double targetReal = role == MlpRole.ADD ? (a + b) : (a * b);
                double[] target = {targetReal / scale};
                double[] out = mlp.forward(in);
                double diff = out[0] - target[0];
                sum += diff * diff;
                mlp.backward(target);
            }
            mlp.step(lr / batch);
            if (!silent && epoch % Math.max(1, epochs / 8) == 0) {
                System.out.printf("  [%s] epoch %d  loss %.5f%n",
                        role.name(), epoch, sum / batch);
            }
        }
    }

    public static void pretrainTransformer(Transformer t, MlpRole role, int epochs, int batch, double lr) {
        pretrainTransformerImpl(t, role, epochs, batch, lr, false);
    }

    public static void pretrainTransformerSilent(Transformer t, MlpRole role, int epochs, int batch, double lr) {
        pretrainTransformerImpl(t, role, epochs, batch, lr, true);
    }

    private static void pretrainTransformerImpl(Transformer t, MlpRole role, int epochs, int batch, double lr, boolean silent) {
        Random rng = new Random(role == MlpRole.ADD ? 7L : 13L);
        double scale = NumberToMatrix.SCALE;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double sum = 0;
            for (int i = 0; i < batch; i++) {
                int a = rng.nextInt(10);
                int b = rng.nextInt(10);
                double[] in = {a / scale, b / scale};
                double targetReal = role == MlpRole.ADD ? (a + b) : (a * b);
                double[] target = {targetReal / scale};
                double[] out = t.forward(in);
                double diff = out[0] - target[0];
                sum += diff * diff;
                t.backward(target);
            }
            t.step(lr / batch);
            if (!silent && epoch % Math.max(1, epochs / 8) == 0) {
                System.out.printf("  [TFM/%s] epoch %d  loss %.5f%n",
                        role.name(), epoch, sum / batch);
            }
        }
    }
}

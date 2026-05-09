package sibarum.strnn.demo;

import sibarum.strnn.mlp.Mlp;
import sibarum.strnn.primitive.NumberToMatrix;

import java.util.Random;

/**
 * Phase 1 standalone test: train two tiny MLPs (one for add, one for multiply)
 * on single-digit pairs, in the same normalized encoding the rest of the
 * pipeline uses. Verifies &lt;1% mean absolute error in the un-scaled output.
 *
 * Encoding is consistent with NumberToMatrix.SCALE: input is [a/scale, b/scale]
 * and target is [(a OP b) / scale]. For multiplication targets can exceed 1.0
 * (e.g. 9*9/10 = 8.1) so the linear output layer is necessary.
 */
public final class MlpTrainingDemo {
    private static final double SCALE = NumberToMatrix.SCALE;
    private static final int BATCH = 32;
    private static final double LR_ADD = 0.01;
    private static final double LR_MUL = 0.004;
    private static final int EPOCHS_ADD = 4000;
    private static final int EPOCHS_MUL = 12000;

    static void main(String[] args) {
        Mlp addMlp = new Mlp(new int[]{2, 32, 1}, 42L);
        Mlp mulMlp = new Mlp(new int[]{2, 128, 64, 1}, 1337L);
        Random rng = new Random(7L);

        trainOp(addMlp, rng, "ADD", LR_ADD, true, EPOCHS_ADD);
        trainOp(mulMlp, rng, "MUL", LR_MUL, false, EPOCHS_MUL);

        double addMae = evalOp(addMlp, true);
        double mulMae = evalOp(mulMlp, false);
        System.out.printf("ADD final MAE: %.4f%n", addMae);
        System.out.printf("MUL final MAE: %.4f%n", mulMae);

        if (addMae > 0.1) throw new AssertionError("ADD did not converge: " + addMae);
        if (mulMae > 0.5) throw new AssertionError("MUL did not converge: " + mulMae);
        System.out.println("Phase 1 MLP training OK.");
    }

    private static void trainOp(Mlp mlp, Random rng, String label, double lr, boolean isAdd, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double sumLoss = 0.0;
            for (int i = 0; i < BATCH; i++) {
                int a = rng.nextInt(10);
                int b = rng.nextInt(10);
                double[] in = new double[]{a / SCALE, b / SCALE};
                double targetVal = isAdd ? (a + b) : (a * b);
                double[] target = new double[]{targetVal / SCALE};
                double[] out = mlp.forward(in);
                double diff = out[0] - target[0];
                sumLoss += diff * diff;
                mlp.backward(target);
            }
            mlp.step(lr / BATCH);
            if (epoch % 1000 == 0) {
                System.out.printf("[%s] epoch %d  batch-loss %.5f%n",
                        label, epoch, sumLoss / BATCH);
            }
        }
    }

    private static double evalOp(Mlp mlp, boolean isAdd) {
        double total = 0.0;
        int count = 0;
        for (int a = 0; a < 10; a++) {
            for (int b = 0; b < 10; b++) {
                double[] in = new double[]{a / SCALE, b / SCALE};
                double predScaled = mlp.forward(in)[0];
                double pred = predScaled * SCALE;
                double truth = isAdd ? (a + b) : (a * b);
                total += Math.abs(pred - truth);
                count++;
            }
        }
        return total / count;
    }
}

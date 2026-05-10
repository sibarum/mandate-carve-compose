package sibarum.strnn.demo;

import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.transformer.Transformer;

import java.util.Random;

/**
 * Standalone test analogous to MlpTrainingDemo: train two transformers (add
 * and multiply) on single-digit pairs and verify both reach reasonable error
 * before we trust them inside the carving demos.
 *
 * Encoding matches NumberToMatrix.SCALE so the transformer slots into the
 * pipeline at the same place an MLP would.
 */
public final class TransformerTrainingDemo {
    private static final double SCALE = NumberToMatrix.SCALE;
    private static final int BATCH = 32;

    public static void main(String[] args) {
        Transformer addTfm = new Transformer(
                /*seqLen=*/2, /*dIn=*/1, /*dModel=*/16, /*dFf=*/64, /*dOut=*/1, 42L);
        Transformer mulTfm = new Transformer(
                /*seqLen=*/2, /*dIn=*/1, /*dModel=*/32, /*dFf=*/128, /*dOut=*/1, 1337L);

        Random rng = new Random(7L);
        train(addTfm, rng, "ADD", true, /*epochs=*/4000, /*lr=*/0.005);
        train(mulTfm, rng, "MUL", false, /*epochs=*/20000, /*lr=*/0.0015);

        double addMae = eval(addTfm, true);
        double mulMae = eval(mulTfm, false);
        System.out.printf("ADD final MAE: %.4f%n", addMae);
        System.out.printf("MUL final MAE: %.4f%n", mulMae);

        if (addMae > 0.2) throw new AssertionError("ADD did not converge: " + addMae);
        if (mulMae > 0.8) throw new AssertionError("MUL did not converge: " + mulMae);
        System.out.println("Transformer training OK.");
    }

    private static void train(Transformer t, Random rng, String label, boolean isAdd, int epochs, double lr) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double sum = 0.0;
            for (int i = 0; i < BATCH; i++) {
                int a = rng.nextInt(10);
                int b = rng.nextInt(10);
                double[] in = new double[]{a / SCALE, b / SCALE};
                double targetReal = isAdd ? (a + b) : (a * b);
                double[] target = new double[]{targetReal / SCALE};
                double[] out = t.forward(in);
                double diff = out[0] - target[0];
                sum += diff * diff;
                t.backward(target);
            }
            t.step(lr / BATCH);
            if (epoch % Math.max(1, epochs / 8) == 0) {
                System.out.printf("[%s] epoch %d  loss %.5f%n", label, epoch, sum / BATCH);
            }
        }
    }

    private static double eval(Transformer t, boolean isAdd) {
        double total = 0.0;
        int count = 0;
        for (int a = 0; a < 10; a++) {
            for (int b = 0; b < 10; b++) {
                double[] in = new double[]{a / SCALE, b / SCALE};
                double pred = t.forward(in)[0] * SCALE;
                double truth = isAdd ? (a + b) : (a * b);
                total += Math.abs(pred - truth);
                count++;
            }
        }
        return total / count;
    }
}

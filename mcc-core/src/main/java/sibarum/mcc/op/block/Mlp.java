package sibarum.mcc.op.block;

import java.util.Arrays;
import java.util.Random;

/**
 * From-scratch tiny MLP: any number of fully-connected layers, ReLU
 * hidden, linear output. MSE loss against a target during
 * {@link #backward}. Plain SGD on {@link #step}. Caches activations
 * from the most recent {@link #forward} pass so {@code backward} can
 * use them directly.
 *
 * <p>This is the raw network. {@link MlpBlock} wraps it as a
 * Trainable Primitive for use in a graph.
 */
public final class Mlp {
    private final int[] sizes;
    private final double[][][] weights;
    private final double[][] biases;
    private final double[][][] gradWeights;
    private final double[][] gradBiases;

    private double[][] preActivations;
    private double[][] activations;
    private double[] cachedInput;

    public Mlp(int[] sizes, long seed) {
        if (sizes.length < 2) {
            throw new IllegalArgumentException("MLP needs at least input and output layer");
        }
        this.sizes = sizes.clone();
        int numLayers = sizes.length - 1;
        this.weights = new double[numLayers][][];
        this.biases = new double[numLayers][];
        this.gradWeights = new double[numLayers][][];
        this.gradBiases = new double[numLayers][];

        Random rng = new Random(seed);
        for (int l = 0; l < numLayers; l++) {
            int in = sizes[l];
            int out = sizes[l + 1];
            weights[l] = new double[out][in];
            biases[l] = new double[out];
            gradWeights[l] = new double[out][in];
            gradBiases[l] = new double[out];
            double bound = Math.sqrt(6.0 / (in + out));
            for (int i = 0; i < out; i++) {
                for (int j = 0; j < in; j++) {
                    weights[l][i][j] = (rng.nextDouble() * 2 - 1) * bound;
                }
            }
        }
    }

    public int inputDim() { return sizes[0]; }
    public int outputDim() { return sizes[sizes.length - 1]; }
    public int[] sizes() { return sizes.clone(); }

    public double[] forward(double[] input) {
        if (input.length != sizes[0]) {
            throw new IllegalArgumentException(
                    "expected input dim " + sizes[0] + ", got " + input.length);
        }
        cachedInput = input.clone();
        int numLayers = sizes.length - 1;
        preActivations = new double[numLayers][];
        activations = new double[numLayers][];
        double[] cur = input;
        for (int l = 0; l < numLayers; l++) {
            double[] z = matMulAdd(weights[l], cur, biases[l]);
            preActivations[l] = z;
            double[] a = (l == numLayers - 1) ? z.clone() : relu(z);
            activations[l] = a;
            cur = a;
        }
        return cur.clone();
    }

    /**
     * Computes gradients given a target output. Accumulates into
     * {@code gradWeights}/{@code gradBiases}. Returns {@code dInput}
     * so callers can chain gradients further back if needed.
     */
    public double[] backward(double[] target) {
        if (target.length != outputDim()) {
            throw new IllegalArgumentException(
                    "expected target dim " + outputDim() + ", got " + target.length);
        }
        int numLayers = sizes.length - 1;
        double[] output = activations[numLayers - 1];
        double[] delta = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            delta[i] = output[i] - target[i];
        }

        for (int l = numLayers - 1; l >= 0; l--) {
            double[] prevAct = (l == 0) ? cachedInput : activations[l - 1];
            for (int i = 0; i < delta.length; i++) {
                gradBiases[l][i] += delta[i];
                for (int j = 0; j < prevAct.length; j++) {
                    gradWeights[l][i][j] += delta[i] * prevAct[j];
                }
            }

            if (l > 0) {
                double[] dPrev = new double[prevAct.length];
                for (int j = 0; j < prevAct.length; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < delta.length; i++) {
                        sum += weights[l][i][j] * delta[i];
                    }
                    double z = preActivations[l - 1][j];
                    dPrev[j] = (z > 0.0) ? sum : 0.0;
                }
                delta = dPrev;
            } else {
                double[] dInput = new double[cachedInput.length];
                for (int j = 0; j < cachedInput.length; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < delta.length; i++) {
                        sum += weights[l][i][j] * delta[i];
                    }
                    dInput[j] = sum;
                }
                return dInput;
            }
        }
        return new double[cachedInput.length];
    }

    public void step(double lr) {
        int numLayers = sizes.length - 1;
        for (int l = 0; l < numLayers; l++) {
            for (int i = 0; i < weights[l].length; i++) {
                biases[l][i] -= lr * gradBiases[l][i];
                gradBiases[l][i] = 0.0;
                for (int j = 0; j < weights[l][i].length; j++) {
                    weights[l][i][j] -= lr * gradWeights[l][i][j];
                    gradWeights[l][i][j] = 0.0;
                }
            }
        }
    }

    public void zeroGrad() {
        int numLayers = sizes.length - 1;
        for (int l = 0; l < numLayers; l++) {
            Arrays.fill(gradBiases[l], 0.0);
            for (int i = 0; i < gradWeights[l].length; i++) {
                Arrays.fill(gradWeights[l][i], 0.0);
            }
        }
    }

    /** Parameter accessors for serialization / introspection. */
    public double[][][] weights() { return weights; }
    public double[][] biases() { return biases; }

    private static double[] matMulAdd(double[][] w, double[] x, double[] b) {
        int out = w.length;
        int in = x.length;
        double[] result = new double[out];
        for (int i = 0; i < out; i++) {
            double s = b[i];
            for (int j = 0; j < in; j++) {
                s += w[i][j] * x[j];
            }
            result[i] = s;
        }
        return result;
    }

    private static double[] relu(double[] z) {
        double[] a = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            a[i] = Math.max(0.0, z[i]);
        }
        return a;
    }
}

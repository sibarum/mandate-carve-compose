package sibarum.strnn.ksq.elevator;

import java.util.Random;

/**
 * Output head for elevator KSQ. Linear map β → logits, where β ∈ R^n is
 * the anchor projection of Q². In the final iter-6 architecture
 * magnitude flows into Q directly via α = ℓ (Phase 2), so the head only
 * sees the readout-projected β; the inputDim matches the anchor count.
 *
 * <p>The Phase 1 prototype concatenated the pooled-embedding magnitude
 * r alongside β as an extra head feature (inputDim = n + 1); that
 * variant isn't retained in code — only Phase 2's pure-β input is.
 */
public final class ElevatorOutputHead {

    private final int outDim;
    private final int inputDim;
    private final double[][] weights;
    private final double[] biases;
    private final double[][] gradWeights;
    private final double[] gradBiases;

    private double[] cachedInput;

    public ElevatorOutputHead(int outDim, int inputDim, long seed) {
        if (outDim <= 0) throw new IllegalArgumentException("outDim must be positive: " + outDim);
        if (inputDim <= 0) throw new IllegalArgumentException("inputDim must be positive: " + inputDim);
        this.outDim = outDim;
        this.inputDim = inputDim;
        this.weights = new double[outDim][inputDim];
        this.biases = new double[outDim];
        this.gradWeights = new double[outDim][inputDim];
        this.gradBiases = new double[outDim];

        Random rng = new Random(seed);
        double bound = Math.sqrt(6.0 / (inputDim + outDim));
        for (int k = 0; k < outDim; k++) {
            for (int i = 0; i < inputDim; i++) {
                weights[k][i] = (rng.nextDouble() * 2.0 - 1.0) * bound;
            }
        }
    }

    public int outDim() { return outDim; }
    public int inputDim() { return inputDim; }

    public double[] forward(double[] input) {
        if (input.length != inputDim) {
            throw new IllegalArgumentException(
                    "input dim " + input.length + " != inputDim " + inputDim);
        }
        cachedInput = input.clone();
        double[] logits = new double[outDim];
        for (int k = 0; k < outDim; k++) {
            double s = biases[k];
            for (int i = 0; i < inputDim; i++) {
                s += weights[k][i] * input[i];
            }
            logits[k] = s;
        }
        return logits;
    }

    public double[] backward(double[] dLogits) {
        if (dLogits.length != outDim) {
            throw new IllegalArgumentException(
                    "dLogits dim " + dLogits.length + " != outDim " + outDim);
        }
        if (cachedInput == null) {
            throw new IllegalStateException("forward must be called before backward");
        }
        for (int k = 0; k < outDim; k++) {
            gradBiases[k] += dLogits[k];
            for (int i = 0; i < inputDim; i++) {
                gradWeights[k][i] += dLogits[k] * cachedInput[i];
            }
        }
        double[] dInput = new double[inputDim];
        for (int i = 0; i < inputDim; i++) {
            double s = 0.0;
            for (int k = 0; k < outDim; k++) {
                s += weights[k][i] * dLogits[k];
            }
            dInput[i] = s;
        }
        return dInput;
    }

    public void step(double lr) {
        for (int k = 0; k < outDim; k++) {
            biases[k] -= lr * gradBiases[k];
            gradBiases[k] = 0.0;
            for (int i = 0; i < inputDim; i++) {
                weights[k][i] -= lr * gradWeights[k][i];
                gradWeights[k][i] = 0.0;
            }
        }
    }

    public void zeroGrad() {
        for (int k = 0; k < outDim; k++) {
            gradBiases[k] = 0.0;
            for (int i = 0; i < inputDim; i++) {
                gradWeights[k][i] = 0.0;
            }
        }
    }

    public double[][] weights() { return weights; }
    public double[] biases() { return biases; }
    public double[][] gradWeights() { return gradWeights; }
    public double[] gradBiases() { return gradBiases; }
}

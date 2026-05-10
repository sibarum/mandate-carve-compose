package sibarum.strnn.transformer;

import java.util.Random;

/**
 * From-scratch tiny transformer block: input projection -> single-head
 * self-attention -> position-wise FFN -> mean-pool -> output projection.
 * Pure Java, no external libraries.
 *
 * Designed as a drop-in alternative to {@link sibarum.strnn.mlp.Mlp} for
 * tasks of shape (Din,) per token, S tokens per example, Dout scalar output.
 * For the v0 STRNN demo we use S=2 (the composed matrix [a/scale, b/scale])
 * and Dout=1. Layer norm and residual connections are omitted to keep the
 * implementation small; what matters for the §9.6 ablation is that the
 * architecture is structurally distinct from the MLP, not that it is a
 * complete reference transformer.
 *
 * Loss: MSE against a target. backward(target) accumulates gradients;
 * step(lr) applies them.
 */
public final class Transformer {
    private final int seqLen;
    private final int dIn;
    private final int dModel;
    private final int dFf;
    private final int dOut;

    private final double[][] Win;   // (dIn, dModel)
    private final double[][] Wq;    // (dModel, dModel)
    private final double[][] Wk;    // (dModel, dModel)
    private final double[][] Wv;    // (dModel, dModel)
    private final double[][] W1;    // (dModel, dFf)
    private final double[] b1;      // (dFf,)
    private final double[][] W2;    // (dFf, dModel)
    private final double[] b2;      // (dModel,)
    private final double[][] Wo;    // (dModel, dOut)
    private final double[] bo;      // (dOut,)

    private final double[][] gWin, gWq, gWk, gWv, gW1, gW2, gWo;
    private final double[] gb1, gb2, gbo;

    // forward cache
    private double[][] x;          // (S, Din)
    private double[][] xp;         // (S, Dm)
    private double[][] Q, K, V;    // (S, Dm)
    private double[][] scores;     // (S, S)
    private double[][] attn;       // (S, S)
    private double[][] attnOut;    // (S, Dm)
    private double[][] hiddenPre;  // (S, Df)
    private double[][] hidden;     // (S, Df)
    private double[][] ffnOut;     // (S, Dm)
    private double[] pooled;       // (Dm,)
    private double[] y;            // (Dout,)

    public Transformer(int seqLen, int dIn, int dModel, int dFf, int dOut, long seed) {
        this.seqLen = seqLen;
        this.dIn = dIn;
        this.dModel = dModel;
        this.dFf = dFf;
        this.dOut = dOut;
        Random rng = new Random(seed);
        this.Win = xavier(dIn, dModel, rng);
        this.Wq = xavier(dModel, dModel, rng);
        this.Wk = xavier(dModel, dModel, rng);
        this.Wv = xavier(dModel, dModel, rng);
        this.W1 = xavier(dModel, dFf, rng);
        this.b1 = new double[dFf];
        this.W2 = xavier(dFf, dModel, rng);
        this.b2 = new double[dModel];
        this.Wo = xavier(dModel, dOut, rng);
        this.bo = new double[dOut];

        this.gWin = new double[dIn][dModel];
        this.gWq = new double[dModel][dModel];
        this.gWk = new double[dModel][dModel];
        this.gWv = new double[dModel][dModel];
        this.gW1 = new double[dModel][dFf];
        this.gb1 = new double[dFf];
        this.gW2 = new double[dFf][dModel];
        this.gb2 = new double[dModel];
        this.gWo = new double[dModel][dOut];
        this.gbo = new double[dOut];
    }

    public int inputDim() {
        return seqLen * dIn;
    }

    public int outputDim() {
        return dOut;
    }

    /**
     * Accepts a flat double[] of size seqLen*dIn (row-major: token by token)
     * and returns a flat double[] of size dOut.
     */
    public double[] forward(double[] flatInput) {
        if (flatInput.length != seqLen * dIn) {
            throw new IllegalArgumentException("expected " + (seqLen * dIn) + " inputs");
        }
        x = new double[seqLen][dIn];
        for (int i = 0; i < seqLen; i++) {
            System.arraycopy(flatInput, i * dIn, x[i], 0, dIn);
        }

        xp = matmul(x, Win);                   // (S, Dm)
        Q = matmul(xp, Wq);                    // (S, Dm)
        K = matmul(xp, Wk);                    // (S, Dm)
        V = matmul(xp, Wv);                    // (S, Dm)

        scores = new double[seqLen][seqLen];
        double scale = 1.0 / Math.sqrt(dModel);
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double s = 0;
                for (int k = 0; k < dModel; k++) s += Q[i][k] * K[j][k];
                scores[i][j] = s * scale;
            }
        }

        attn = softmaxRows(scores);

        attnOut = matmul(attn, V);             // (S, Dm)
        // Residual: attnOut += xp
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) attnOut[i][k] += xp[i][k];
        }

        hiddenPre = addRowVec(matmul(attnOut, W1), b1);   // (S, Df)
        hidden = relu(hiddenPre);                          // (S, Df)
        ffnOut = addRowVec(matmul(hidden, W2), b2);        // (S, Dm)
        // Residual: ffnOut += attnOut
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) ffnOut[i][k] += attnOut[i][k];
        }

        pooled = new double[dModel];
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) pooled[k] += ffnOut[i][k];
        }
        for (int k = 0; k < dModel; k++) pooled[k] /= seqLen;

        y = new double[dOut];
        for (int o = 0; o < dOut; o++) {
            double s = bo[o];
            for (int k = 0; k < dModel; k++) s += pooled[k] * Wo[k][o];
            y[o] = s;
        }
        return y.clone();
    }

    public void backward(double[] target) {
        double[] dY = new double[dOut];
        for (int o = 0; o < dOut; o++) dY[o] = y[o] - target[o];

        double[] dPooled = new double[dModel];
        for (int o = 0; o < dOut; o++) {
            gbo[o] += dY[o];
            for (int k = 0; k < dModel; k++) {
                gWo[k][o] += pooled[k] * dY[o];
                dPooled[k] += dY[o] * Wo[k][o];
            }
        }

        double[][] dFfnOut = new double[seqLen][dModel];
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) dFfnOut[i][k] = dPooled[k] / seqLen;
        }

        double[][] dHidden = new double[seqLen][dFf];
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) {
                gb2[k] += dFfnOut[i][k];
                for (int h = 0; h < dFf; h++) {
                    gW2[h][k] += hidden[i][h] * dFfnOut[i][k];
                    dHidden[i][h] += dFfnOut[i][k] * W2[h][k];
                }
            }
        }

        double[][] dHiddenPre = new double[seqLen][dFf];
        for (int i = 0; i < seqLen; i++) {
            for (int h = 0; h < dFf; h++) {
                dHiddenPre[i][h] = hiddenPre[i][h] > 0 ? dHidden[i][h] : 0.0;
            }
        }

        // dFfnOut flows through both the FFN branch AND the residual into attnOut.
        // The FFN branch contributes to dAttnOut via dHidden -> W1 -> attnOut;
        // the residual contributes dAttnOut += dFfnOut directly.
        double[][] dAttnOut = new double[seqLen][dModel];
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) dAttnOut[i][k] = dFfnOut[i][k];
        }
        for (int i = 0; i < seqLen; i++) {
            for (int h = 0; h < dFf; h++) {
                gb1[h] += dHiddenPre[i][h];
                for (int k = 0; k < dModel; k++) {
                    gW1[k][h] += attnOut[i][k] * dHiddenPre[i][h];
                    dAttnOut[i][k] += dHiddenPre[i][h] * W1[k][h];
                }
            }
        }

        // dV = attn^T @ dAttnOut
        double[][] dV = new double[seqLen][dModel];
        for (int j = 0; j < seqLen; j++) {
            for (int k = 0; k < dModel; k++) {
                double s = 0;
                for (int i = 0; i < seqLen; i++) s += attn[i][j] * dAttnOut[i][k];
                dV[j][k] = s;
            }
        }
        // dAttn[i,j] = sum_k V[j,k] * dAttnOut[i,k]
        double[][] dAttn = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double s = 0;
                for (int k = 0; k < dModel; k++) s += V[j][k] * dAttnOut[i][k];
                dAttn[i][j] = s;
            }
        }
        // softmax backward (row-wise)
        double[][] dScores = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            double sumDot = 0;
            for (int l = 0; l < seqLen; l++) sumDot += dAttn[i][l] * attn[i][l];
            for (int j = 0; j < seqLen; j++) {
                dScores[i][j] = attn[i][j] * (dAttn[i][j] - sumDot);
            }
        }

        double scale = 1.0 / Math.sqrt(dModel);
        // dScoresScaled = dScores * scale
        // dQ[i,k] = sum_j dScoresScaled[i,j] * K[j,k]
        // dK[j,k] = sum_i dScoresScaled[i,j] * Q[i,k]
        double[][] dQ = new double[seqLen][dModel];
        double[][] dK = new double[seqLen][dModel];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double s = dScores[i][j] * scale;
                for (int k = 0; k < dModel; k++) {
                    dQ[i][k] += s * K[j][k];
                    dK[j][k] += s * Q[i][k];
                }
            }
        }

        // dWq = xp^T @ dQ; dWk = xp^T @ dK; dWv = xp^T @ dV
        // dXp also gets the residual contribution from dAttnOut.
        double[][] dXp = new double[seqLen][dModel];
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) dXp[i][k] = dAttnOut[i][k];
        }
        for (int i = 0; i < seqLen; i++) {
            for (int k = 0; k < dModel; k++) {
                for (int m = 0; m < dModel; m++) {
                    gWq[k][m] += xp[i][k] * dQ[i][m];
                    gWk[k][m] += xp[i][k] * dK[i][m];
                    gWv[k][m] += xp[i][k] * dV[i][m];
                    dXp[i][k] += dQ[i][m] * Wq[k][m]
                            + dK[i][m] * Wk[k][m]
                            + dV[i][m] * Wv[k][m];
                }
            }
        }

        // dWin = x^T @ dXp
        for (int i = 0; i < seqLen; i++) {
            for (int din = 0; din < dIn; din++) {
                for (int m = 0; m < dModel; m++) {
                    gWin[din][m] += x[i][din] * dXp[i][m];
                }
            }
        }
    }

    public void step(double lr) {
        applyAndClear(Win, gWin, lr);
        applyAndClear(Wq, gWq, lr);
        applyAndClear(Wk, gWk, lr);
        applyAndClear(Wv, gWv, lr);
        applyAndClear(W1, gW1, lr);
        applyAndClear(W2, gW2, lr);
        applyAndClear(Wo, gWo, lr);
        applyAndClearVec(b1, gb1, lr);
        applyAndClearVec(b2, gb2, lr);
        applyAndClearVec(bo, gbo, lr);
    }

    private static void applyAndClear(double[][] w, double[][] g, double lr) {
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++) {
                w[i][j] -= lr * g[i][j];
                g[i][j] = 0.0;
            }
        }
    }

    private static void applyAndClearVec(double[] b, double[] g, double lr) {
        for (int i = 0; i < b.length; i++) {
            b[i] -= lr * g[i];
            g[i] = 0.0;
        }
    }

    private static double[][] matmul(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        int p = a[0].length;
        double[][] r = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < p; k++) {
                double aik = a[i][k];
                for (int j = 0; j < m; j++) r[i][j] += aik * b[k][j];
            }
        }
        return r;
    }

    private static double[][] addRowVec(double[][] m, double[] v) {
        for (double[] row : m) {
            for (int j = 0; j < row.length; j++) row[j] += v[j];
        }
        return m;
    }

    private static double[][] relu(double[][] z) {
        double[][] a = new double[z.length][z[0].length];
        for (int i = 0; i < z.length; i++) {
            for (int j = 0; j < z[0].length; j++) a[i][j] = Math.max(0.0, z[i][j]);
        }
        return a;
    }

    private static double[][] softmaxRows(double[][] m) {
        int rows = m.length;
        int cols = m[0].length;
        double[][] r = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) if (m[i][j] > max) max = m[i][j];
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                r[i][j] = Math.exp(m[i][j] - max);
                sum += r[i][j];
            }
            for (int j = 0; j < cols; j++) r[i][j] /= sum;
        }
        return r;
    }

    private static double[][] xavier(int in, int out, Random rng) {
        double bound = Math.sqrt(6.0 / (in + out));
        double[][] w = new double[in][out];
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                w[i][j] = (rng.nextDouble() * 2 - 1) * bound;
            }
        }
        return w;
    }
}

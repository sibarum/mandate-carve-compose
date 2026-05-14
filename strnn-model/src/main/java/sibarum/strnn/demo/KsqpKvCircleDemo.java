package sibarum.strnn.demo;

import sibarum.strnn.ksqp.KsqpKvModel;
import sibarum.strnn.ksqp.SplitQuat;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

/**
 * Step 2 of the isolated-test sequence: scale M from 4 to 8 on a richer
 * cluster pattern, holding the rest of the architecture and the
 * hyperparameters constant.
 *
 * <p>Task: 8 clusters arranged at 45° increments on the unit circle.
 * Labels alternate around the circle, giving a 4-vs-4 binary task where
 * every pair of adjacent clusters has opposite labels. Linear in raw 2D
 * space? No — alternating labels around a circle are NOT linearly
 * separable, so the architecture has to use the per-entry transformations
 * to distinguish, not just route by location.
 *
 * <p>Compared to {@link KsqpKvXorDemo}: M=8 entries (vs 4), 8 clusters
 * (vs 4), trainable stored keys (matches step 1's outcome). Everything
 * else — temperature, noise σ, samples per cluster, learning rate,
 * epochs, seeds — identical.
 *
 * <p>What this tests: does the architecture handle 8-way attention?
 * Failure modes to watch for:
 * <ul>
 *   <li>Attention spreading uniformly across all 8 (τ=1.0 too soft for M=8?)</li>
 *   <li>Mode collapse — multiple keys converging to the same location</li>
 *   <li>Loss-landscape pathology — some clusters never solved</li>
 * </ul>
 */
public final class KsqpKvCircleDemo {

    private static final int OUT_DIM = 2;
    private static final int QUERY_DIM = 2;
    private static final int M = 8;
    private static final int EPOCHS = 2000;
    private static final double LR = 0.1;
    private static final double TEMPERATURE = 1.0;
    private static final double NOISE_SIGMA = 0.3;
    private static final int TRAIN_PER_CLUSTER = 8;
    private static final int VAL_PER_CLUSTER = 25;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };

    private static final double[][] CLUSTERS = buildCircleClusters(M);
    private static final int[] CLUSTER_LABELS = buildAlternatingLabels(M);

    private static double[][] buildCircleClusters(int n) {
        double[][] out = new double[n][2];
        for (int i = 0; i < n; i++) {
            double theta = 2.0 * Math.PI * i / n;
            out[i][0] = Math.cos(theta);
            out[i][1] = Math.sin(theta);
        }
        return out;
    }

    private static int[] buildAlternatingLabels(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = i & 1;
        return out;
    }

    public static void main(String[] args) throws IOException {
        String runTag = args.length > 0 ? args[0] : "kv-circle-8";
        Path outDir = Paths.get("ksqp-data", runTag);
        Files.createDirectories(outDir);
        Path summaryPath = outDir.resolve("summary.csv");
        Path lossPath = outDir.resolve("loss.csv");
        Path trajPath = outDir.resolve("trajectories.csv");
        Path eventsPath = outDir.resolve("events.csv");
        Path attnPath = outDir.resolve("attention_samples.csv");
        Path keysPath = outDir.resolve("stored_keys.csv");
        Path clustersPath = outDir.resolve("clusters.csv");

        System.out.println("=== KSQP-KV — circle-8 parity (step 2: scale M to 8) ===");
        System.out.println("  task: " + M + " clusters at 45° increments on unit circle, alternating labels");
        System.out.println("  arch: query → softmax-attention(d=‖q-k_v‖²/τ) over " + M + " TRAINABLE prototypes; per-entry lift→P_p→sandwich; pool→head");
        System.out.println("  event map: N flips + → -  ⇒ p++   |   - → +  ⇒ p--");
        System.out.println("  τ=" + TEMPERATURE + "   σ_noise=" + NOISE_SIGMA
                + "   train/cluster=" + TRAIN_PER_CLUSTER + "   val/cluster=" + VAL_PER_CLUSTER
                + "   LR=" + LR + "   epochs=" + EPOCHS);
        System.out.println("  writing raw data → " + outDir.toAbsolutePath());
        System.out.println();
        System.out.printf("  %5s | %7s | %5s | %-9s | %-6s | %-7s%n",
                "seed", "train", "val", "ce", "events", "mean p");
        System.out.println("  -----+---------+-------+-----------+--------+---------");

        // Log cluster positions once at the top of the run.
        try (BufferedWriter w = Files.newBufferedWriter(clustersPath)) {
            w.write("cluster_id,x,y,label\n");
            for (int c = 0; c < M; c++) {
                w.write(String.format("%d,%.6f,%.6f,%d%n", c, CLUSTERS[c][0], CLUSTERS[c][1], CLUSTER_LABELS[c]));
            }
        }

        int solved = 0;
        int generalizes = 0;
        int totalEvents = 0;

        try (BufferedWriter summary = Files.newBufferedWriter(summaryPath);
             BufferedWriter loss = Files.newBufferedWriter(lossPath);
             BufferedWriter traj = Files.newBufferedWriter(trajPath);
             BufferedWriter events = Files.newBufferedWriter(eventsPath);
             BufferedWriter attn = Files.newBufferedWriter(attnPath);
             BufferedWriter keys = Files.newBufferedWriter(keysPath)) {

            summary.write("seed,train_acc,val_acc,ce,events,mean_p,outcome\n");
            loss.write("seed,epoch,ce,train_acc,val_acc\n");
            traj.write("seed,epoch,entry,sq0,sq1,sq2,sq3,N,p\n");
            events.write("seed,epoch,entry,prevP,newP,normBefore,normAfter,sq0_at,sq1_at,sq2_at,sq3_at\n");
            // attention header is wide (M=8): w0..w7.
            StringBuilder attnHeader = new StringBuilder("seed,q_x,q_y,label");
            for (int v = 0; v < M; v++) attnHeader.append(",w").append(v);
            attnHeader.append(",pred\n");
            attn.write(attnHeader.toString());
            keys.write("seed,epoch,entry,k_x,k_y\n");

            for (long seed : SEEDS) {
                RunResult r = runOne(seed, traj, loss, events, attn, keys);
                totalEvents += r.eventCount;
                String outcome = r.valAcc >= 0.90 ? "solved"
                        : r.trainAcc >= 0.95 && r.valAcc >= 0.80 ? "ok"
                        : r.diverged ? "diverged"
                        : "stuck";
                if ("solved".equals(outcome)) solved++;
                if ("solved".equals(outcome) || "ok".equals(outcome)) generalizes++;

                double meanP = 0.0;
                for (int v : r.finalP) meanP += v;
                meanP /= r.finalP.length;

                System.out.printf("  %5d | %5.3f   | %5.3f | %.5f  | %6d | %5.2f%n",
                        seed, r.trainAcc, r.valAcc, r.ce, r.eventCount, meanP);
                summary.write(String.format("%d,%.6f,%.6f,%.6f,%d,%.6f,%s%n",
                        seed, r.trainAcc, r.valAcc, r.ce, r.eventCount, meanP, outcome));
            }
        }

        System.out.println();
        System.out.println("Summary across " + SEEDS.length + " seeds:");
        System.out.println("  solved (val ≥ 90%):       " + solved + "/" + SEEDS.length);
        System.out.println("  generalizes (val ≥ 80%):  " + generalizes + "/" + SEEDS.length);
        System.out.println("  total null-cone events:   " + totalEvents);
        System.out.println();
        System.out.println("  summary:    " + summaryPath.toAbsolutePath());
        System.out.println("  loss:       " + lossPath.toAbsolutePath());
        System.out.println("  traj:       " + trajPath.toAbsolutePath());
        System.out.println("  events:     " + eventsPath.toAbsolutePath());
        System.out.println("  attention:  " + attnPath.toAbsolutePath());
        System.out.println("  keys:       " + keysPath.toAbsolutePath());
        System.out.println("  clusters:   " + clustersPath.toAbsolutePath());
    }

    private record Sample(double[] q, int label) {}

    private record RunResult(double trainAcc, double valAcc, double ce, boolean diverged,
                             int[] finalP, int eventCount) {}

    private static RunResult runOne(long seed,
                                    BufferedWriter traj, BufferedWriter loss,
                                    BufferedWriter eventsOut, BufferedWriter attnOut,
                                    BufferedWriter keysOut)
            throws IOException {
        Random trainRng = new Random(seed * 1_000_003L);
        Random valRng = new Random(seed * 1_000_009L + 7L);
        Sample[] train = sampleClusters(trainRng, TRAIN_PER_CLUSTER);
        Sample[] val = sampleClusters(valRng, VAL_PER_CLUSTER);

        KsqpKvModel model = new KsqpKvModel(OUT_DIM, QUERY_DIM, M, seed,
                KsqpKvModel.DEFAULT_SQ_INIT_A0, KsqpKvModel.DEFAULT_SQ_INIT_NOISE,
                TEMPERATURE, KsqpKvModel.DEFAULT_STORED_KEY_INIT_RANGE);

        int eventCount = 0;
        boolean diverged = false;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int v = 0; v < M; v++) {
                double[] s = model.sq(v);
                traj.write(String.format("%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%d%n",
                        seed, epoch, v, s[0], s[1], s[2], s[3], SplitQuat.norm(s), model.p(v)));
                double[] k = model.storedKey(v);
                keysOut.write(String.format("%d,%d,%d,%.8e,%.8e%n", seed, epoch, v, k[0], k[1]));
            }

            model.zeroGrad();
            double epochCe = 0.0;
            int trainCorrect = 0;
            for (Sample ex : train) {
                double[] logits = model.forward(ex.q);
                if (argmax(logits) == ex.label) trainCorrect++;
                epochCe += model.crossEntropyLoss(ex.label);
                model.backward(ex.label);
            }
            epochCe /= train.length;
            double trainAcc = trainCorrect / (double) train.length;

            model.step(LR / train.length);
            List<KsqpKvModel.EventRecord> evs = model.detectEvents(epoch);
            for (KsqpKvModel.EventRecord ev : evs) {
                double[] s = ev.sqAtEvent();
                eventsOut.write(String.format("%d,%d,%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e%n",
                        seed, ev.epoch(), ev.entryId(), ev.prevP(), ev.newP(),
                        ev.normBefore(), ev.normAfter(), s[0], s[1], s[2], s[3]));
            }
            eventCount += evs.size();

            double valAcc = (epoch % 50 == 0 || epoch == EPOCHS - 1) ? evaluate(model, val) : Double.NaN;
            loss.write(String.format("%d,%d,%.8e,%.6f,%s%n",
                    seed, epoch, epochCe, trainAcc, Double.isNaN(valAcc) ? "" : String.format("%.6f", valAcc)));

            if (epoch > 0 && epoch % 200 == 0) {
                for (int v = 0; v < M; v++) {
                    double m = magnitude(model.sq(v));
                    if (!Double.isFinite(m) || m > 1e8) { diverged = true; break; }
                }
                if (diverged) break;
            }
        }

        double trainAcc = evaluate(model, train);
        double valAcc = evaluate(model, val);
        double ce = 0.0;
        for (Sample ex : val) {
            model.forward(ex.q);
            ce += model.crossEntropyLoss(ex.label);
        }
        ce /= val.length;

        for (Sample ex : val) {
            double[] logits = model.forward(ex.q);
            double[] w = model.lastWeights();
            StringBuilder row = new StringBuilder();
            row.append(seed).append(',')
                    .append(String.format("%.6f", ex.q[0])).append(',')
                    .append(String.format("%.6f", ex.q[1])).append(',')
                    .append(ex.label);
            for (int v = 0; v < M; v++) row.append(',').append(String.format("%.6f", w[v]));
            row.append(',').append(argmax(logits)).append('\n');
            attnOut.write(row.toString());
        }

        int[] finalP = new int[M];
        for (int v = 0; v < M; v++) finalP[v] = model.p(v);
        return new RunResult(trainAcc, valAcc, ce, diverged, finalP, eventCount);
    }

    private static Sample[] sampleClusters(Random rng, int perCluster) {
        Sample[] out = new Sample[M * perCluster];
        int idx = 0;
        for (int c = 0; c < M; c++) {
            for (int i = 0; i < perCluster; i++) {
                double[] q = new double[QUERY_DIM];
                for (int j = 0; j < QUERY_DIM; j++) {
                    q[j] = CLUSTERS[c][j] + rng.nextGaussian() * NOISE_SIGMA;
                }
                out[idx++] = new Sample(q, CLUSTER_LABELS[c]);
            }
        }
        return out;
    }

    private static double evaluate(KsqpKvModel model, Sample[] data) {
        int correct = 0;
        for (Sample ex : data) {
            double[] logits = model.forward(ex.q);
            if (argmax(logits) == ex.label) correct++;
        }
        return correct / (double) data.length;
    }

    private static double magnitude(double[] v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return Math.sqrt(s);
    }

    private static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}

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
 * Smallest real KV toy: continuous XOR-2D classification with a soft-
 * attention KV cache of 4 stored prototypes at the corners of the unit
 * square. Each training/validation query is a noisy 2D point sampled
 * near one of the four corners; XOR labels on the sign bits of the
 * underlying corner.
 *
 * <p>This isolates "does similarity-based lookup work" from "can the
 * model learn prototype positions" by freezing the stored keys at the
 * corner locations. sq, P_d, and the head are gradient-trained. p is
 * discrete and event-driven, exactly as in the indexed-KV XOR.
 *
 * <p>Train set is sampled once per seed (32 noisy queries, 8 per corner)
 * and used for all epochs. Validation set is a held-out 100 queries
 * (25 per corner) sampled with a different RNG offset. Report
 * train acc, val acc, events, final p per entry.
 */
public final class KsqpKvXorDemo {

    private static final int OUT_DIM = 2;
    private static final int QUERY_DIM = 2;
    private static final int EPOCHS = 2000;
    private static final double LR = 0.1;
    private static final double TEMPERATURE = 1.0;
    private static final double NOISE_SIGMA = 0.3;
    private static final int TRAIN_PER_CLUSTER = 8;
    private static final int VAL_PER_CLUSTER = 25;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };

    // 4 corner prototypes; XOR of sign bits gives labels (1,1)→0, (1,-1)→1, (-1,-1)→0, (-1,1)→1.
    private static final double[][] CORNERS = {
            { 1.0,  1.0 },
            { 1.0, -1.0 },
            {-1.0, -1.0 },
            {-1.0,  1.0 }
    };
    private static final int[] CORNER_LABELS = { 0, 1, 0, 1 };

    public static void main(String[] args) throws IOException {
        String runTag = args.length > 0 ? args[0] : "kv-xor-2d";
        boolean trainKeys = false;
        for (int i = 1; i < args.length; i++) {
            if ("--train-keys".equals(args[i])) trainKeys = true;
        }
        Path outDir = Paths.get("ksqp-data", runTag);
        Files.createDirectories(outDir);
        Path summaryPath = outDir.resolve("summary.csv");
        Path lossPath = outDir.resolve("loss.csv");
        Path trajPath = outDir.resolve("trajectories.csv");
        Path eventsPath = outDir.resolve("events.csv");
        Path attnPath = outDir.resolve("attention_samples.csv");
        Path keysPath = outDir.resolve("stored_keys.csv");

        System.out.println("=== KSQP-KV — continuous XOR-2D with soft-attention KV ===");
        System.out.println("  arch: query → softmax-attention(d=‖q-k_v‖²/τ) over " + CORNERS.length
                + (trainKeys ? " TRAINABLE prototypes" : " frozen prototypes")
                + "; per-entry lift→P_p→sandwich; pool→head");
        System.out.println("  event map: N flips + → -  ⇒ p++   |   - → +  ⇒ p--");
        System.out.println("  τ=" + TEMPERATURE + "   σ_noise=" + NOISE_SIGMA
                + "   train/cluster=" + TRAIN_PER_CLUSTER + "   val/cluster=" + VAL_PER_CLUSTER
                + "   LR=" + LR + "   epochs=" + EPOCHS
                + "   storedKeys=" + (trainKeys
                    ? "trainable (init range ±" + KsqpKvModel.DEFAULT_STORED_KEY_INIT_RANGE + ")"
                    : "frozen at corners"));
        System.out.println("  writing raw data → " + outDir.toAbsolutePath());
        System.out.println();
        System.out.printf("  %5s | %8s | %7s | %-9s | %-6s | %-19s | %s%n",
                "seed", "train", "val", "ce", "events", "final p (4 entries)", "final N (4 entries)");
        System.out.println("  -----+----------+---------+-----------+--------+---------------------+--------------------------");

        int solved = 0;       // val ≥ 90%
        int generalizes = 0;  // train ≥ 95% AND val ≥ 80%
        int totalEvents = 0;

        try (BufferedWriter summary = Files.newBufferedWriter(summaryPath);
             BufferedWriter loss = Files.newBufferedWriter(lossPath);
             BufferedWriter traj = Files.newBufferedWriter(trajPath);
             BufferedWriter events = Files.newBufferedWriter(eventsPath);
             BufferedWriter attn = Files.newBufferedWriter(attnPath);
             BufferedWriter keys = Files.newBufferedWriter(keysPath)) {

            summary.write("seed,train_acc,val_acc,ce,events,p0,p1,p2,p3,N0,N1,N2,N3,outcome\n");
            loss.write("seed,epoch,ce,train_acc,val_acc\n");
            traj.write("seed,epoch,entry,sq0,sq1,sq2,sq3,N,p\n");
            events.write("seed,epoch,entry,prevP,newP,normBefore,normAfter,sq0_at,sq1_at,sq2_at,sq3_at\n");
            attn.write("seed,q_x,q_y,label,w0,w1,w2,w3,pred\n");
            keys.write("seed,epoch,entry,k_x,k_y\n");

            for (long seed : SEEDS) {
                RunResult r = runOne(seed, trainKeys, traj, loss, events, attn, keys);
                totalEvents += r.eventCount;
                String outcome = r.valAcc >= 0.90 ? "solved"
                        : r.trainAcc >= 0.95 && r.valAcc >= 0.80 ? "ok"
                        : r.diverged ? "diverged"
                        : "stuck";
                if ("solved".equals(outcome)) solved++;
                if ("solved".equals(outcome) || "ok".equals(outcome)) generalizes++;

                System.out.printf("  %5d | %4.2f/4   | %5.2f   | %.5f  | %6d | (%d, %d, %d, %d)        | (%+.3f, %+.3f, %+.3f, %+.3f)%n",
                        seed, r.trainAcc * 4, r.valAcc, r.ce, r.eventCount,
                        r.finalP[0], r.finalP[1], r.finalP[2], r.finalP[3],
                        r.finalN[0], r.finalN[1], r.finalN[2], r.finalN[3]);
                summary.write(String.format("%d,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%s%n",
                        seed, r.trainAcc, r.valAcc, r.ce, r.eventCount,
                        r.finalP[0], r.finalP[1], r.finalP[2], r.finalP[3],
                        r.finalN[0], r.finalN[1], r.finalN[2], r.finalN[3], outcome));
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
    }

    private record Sample(double[] q, int label) {}

    private record RunResult(double trainAcc, double valAcc, double ce, boolean diverged,
                             int[] finalP, double[] finalN, int eventCount) {}

    private static RunResult runOne(long seed, boolean trainKeys,
                                    BufferedWriter traj, BufferedWriter loss,
                                    BufferedWriter eventsOut, BufferedWriter attnOut,
                                    BufferedWriter keysOut)
            throws IOException {
        // Sample train / val sets with deterministic RNG derived from seed.
        Random trainRng = new Random(seed * 1_000_003L);
        Random valRng = new Random(seed * 1_000_009L + 7L);
        Sample[] train = sampleClusters(trainRng, TRAIN_PER_CLUSTER);
        Sample[] val = sampleClusters(valRng, VAL_PER_CLUSTER);

        KsqpKvModel model = trainKeys
                ? new KsqpKvModel(OUT_DIM, QUERY_DIM, CORNERS.length, seed,
                        KsqpKvModel.DEFAULT_SQ_INIT_A0, KsqpKvModel.DEFAULT_SQ_INIT_NOISE,
                        TEMPERATURE, KsqpKvModel.DEFAULT_STORED_KEY_INIT_RANGE)
                : new KsqpKvModel(OUT_DIM, QUERY_DIM, CORNERS, seed,
                        KsqpKvModel.DEFAULT_SQ_INIT_A0, KsqpKvModel.DEFAULT_SQ_INIT_NOISE, TEMPERATURE);

        int eventCount = 0;
        boolean diverged = false;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Snapshot sq/p at start of epoch.
            for (int v = 0; v < CORNERS.length; v++) {
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

            // Compute val acc every 50 epochs (cheap, but no need to log every step).
            double valAcc = (epoch % 50 == 0 || epoch == EPOCHS - 1) ? evaluate(model, val) : Double.NaN;
            loss.write(String.format("%d,%d,%.8e,%.6f,%s%n",
                    seed, epoch, epochCe, trainAcc, Double.isNaN(valAcc) ? "" : String.format("%.6f", valAcc)));

            if (epoch > 0 && epoch % 200 == 0) {
                for (int v = 0; v < CORNERS.length; v++) {
                    double m = magnitude(model.sq(v));
                    if (!Double.isFinite(m) || m > 1e8) { diverged = true; break; }
                }
                if (diverged) break;
            }
        }

        // Final eval and attention snapshot.
        double trainAcc = evaluate(model, train);
        double valAcc = evaluate(model, val);
        double ce = 0.0;
        for (Sample ex : val) {
            model.forward(ex.q);
            ce += model.crossEntropyLoss(ex.label);
        }
        ce /= val.length;

        // Sample a handful of attention distributions for inspection.
        for (Sample ex : val) {
            double[] logits = model.forward(ex.q);
            double[] w = model.lastWeights();
            attnOut.write(String.format("%d,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%d%n",
                    seed, ex.q[0], ex.q[1], ex.label,
                    w[0], w[1], w[2], w[3], argmax(logits)));
        }

        int[] finalP = new int[CORNERS.length];
        double[] finalN = new double[CORNERS.length];
        for (int v = 0; v < CORNERS.length; v++) {
            finalP[v] = model.p(v);
            finalN[v] = SplitQuat.norm(model.sq(v));
        }
        return new RunResult(trainAcc, valAcc, ce, diverged, finalP, finalN, eventCount);
    }

    private static Sample[] sampleClusters(Random rng, int perCluster) {
        Sample[] out = new Sample[CORNERS.length * perCluster];
        int idx = 0;
        for (int c = 0; c < CORNERS.length; c++) {
            for (int i = 0; i < perCluster; i++) {
                double[] q = new double[QUERY_DIM];
                for (int j = 0; j < QUERY_DIM; j++) {
                    q[j] = CORNERS[c][j] + rng.nextGaussian() * NOISE_SIGMA;
                }
                out[idx++] = new Sample(q, CORNER_LABELS[c]);
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

package sibarum.strnn.demo;

import sibarum.strnn.ksqp.KsqpModel;
import sibarum.strnn.ksqp.SplitQuat;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * KSQP XOR data-gathering run on the correct architecture: per-token
 * fixed random k, lift to degree-p monomials, learned projection P_d
 * to ℝ⁴, conjugate sandwich y = sq · x · sq̄, concat per-token outputs,
 * linear head. Discrete p moves on null-cone sign-flips of sq.
 *
 * <p>This restarts the up/down mapping A/B fresh — the prior runs
 * measured the broken architecture and don't carry over.
 */
public final class KsqpXorDemo {

    private static final int VOCAB = 2;
    private static final int OUT_DIM = 2;
    private static final int SEQ_LEN = 2;
    private static final int N = 4;
    private static final int EPOCHS = 2000;
    private static final double LR = 0.1;
    private static final long[] SEEDS = { 1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L };

    // XOR: 2-token sequences, label = bit_0 XOR bit_1.
    private static final int[][] XS = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    private static final int[] YS = { 0, 1, 1, 0 };

    public static void main(String[] args) throws IOException {
        String runTag = args.length > 0 ? args[0] : "run";
        Path outDir = Paths.get("ksqp-data", runTag);
        Files.createDirectories(outDir);
        Path trajPath = outDir.resolve("trajectories.csv");
        Path lossPath = outDir.resolve("loss.csv");
        Path eventsPath = outDir.resolve("events.csv");
        Path summaryPath = outDir.resolve("summary.csv");

        System.out.println("=== KSQP — XOR with null-cone-event degree control (correct arch) ===");
        System.out.println("  arch: per-token  lift_p(k_v) → P_p → sandwich(sq_v) → split-quat product → head");
        System.out.println("  event map: N flips + → -  ⇒ p++   |   - → +  ⇒ p--");
        System.out.println("  p ∈ [" + KsqpModel.P_MIN + ", " + KsqpModel.P_MAX + "]   "
                + "p_init = " + KsqpModel.P_INIT
                + "   n=" + N + "   seqLen=" + SEQ_LEN
                + "   sq init: a_0=" + KsqpModel.DEFAULT_SQ_INIT_A0
                + " ± " + KsqpModel.DEFAULT_SQ_INIT_NOISE
                + "   LR=" + LR + "   epochs=" + EPOCHS);
        System.out.println("  writing raw data → " + outDir.toAbsolutePath());
        System.out.println();
        System.out.printf("  %5s | %7s | %-9s | %-6s | %-15s | %-9s | %s%n",
                "seed", "acc", "ce", "events", "final p (t0,t1)", "outcome", "final N (t0,t1)");
        System.out.println("  -----+---------+-----------+--------+-----------------+-----------+-----------------");

        int solved = 0;
        int diverged = 0;
        int allZero = 0;
        int totalEvents = 0;

        try (BufferedWriter traj = Files.newBufferedWriter(trajPath);
             BufferedWriter loss = Files.newBufferedWriter(lossPath);
             BufferedWriter events = Files.newBufferedWriter(eventsPath);
             BufferedWriter summary = Files.newBufferedWriter(summaryPath)) {

            traj.write("seed,epoch,token,sq0,sq1,sq2,sq3,N,p\n");
            loss.write("seed,epoch,ce\n");
            events.write("seed,epoch,token,prevP,newP,normBefore,normAfter,sq0_at,sq1_at,sq2_at,sq3_at\n");
            summary.write("seed,correct,ce,events,p0,p1,N0,N1,outcome\n");

            for (long seed : SEEDS) {
                RunResult r = runOne(seed, traj, loss, events);
                totalEvents += r.eventCount;

                String outcome;
                if (r.diverged) { outcome = "diverged"; diverged++; }
                else if (r.allZero) { outcome = "all-zero"; allZero++; }
                else if (r.correct == XS.length) { outcome = "solved"; solved++; }
                else outcome = "stuck";

                System.out.printf("  %5d | %3d/%-3d | %.5f  | %6d | (%d, %d)           | %-9s | (%+.4f, %+.4f)%n",
                        seed, r.correct, XS.length, r.ce, r.eventCount,
                        r.finalP[0], r.finalP[1], outcome, r.finalN[0], r.finalN[1]);
                summary.write(String.format("%d,%d,%.6f,%d,%d,%d,%.6f,%.6f,%s%n",
                        seed, r.correct, r.ce, r.eventCount,
                        r.finalP[0], r.finalP[1], r.finalN[0], r.finalN[1], outcome));
            }
        }

        System.out.println();
        System.out.println("Summary across " + SEEDS.length + " seeds:");
        System.out.println("  solved:    " + solved + "/" + SEEDS.length);
        System.out.println("  diverged:  " + diverged + "/" + SEEDS.length);
        System.out.println("  all-zero:  " + allZero + "/" + SEEDS.length);
        System.out.println("  stuck:     " + (SEEDS.length - solved - diverged - allZero) + "/" + SEEDS.length);
        System.out.println("  total null-cone events: " + totalEvents);
        System.out.println();
        System.out.println("  trajectories: " + trajPath.toAbsolutePath());
        System.out.println("  loss:         " + lossPath.toAbsolutePath());
        System.out.println("  events:       " + eventsPath.toAbsolutePath());
        System.out.println("  summary:      " + summaryPath.toAbsolutePath());
    }

    private static RunResult runOne(long seed, BufferedWriter traj, BufferedWriter loss,
                                    BufferedWriter eventsOut) throws IOException {
        KsqpModel model = new KsqpModel(VOCAB, OUT_DIM, SEQ_LEN, N, seed);
        int eventCount = 0;
        boolean diverged = false;

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Snapshot sq and p BEFORE this epoch's update, so trajectory at
            // epoch t reflects the state used in that epoch's forward pass.
            for (int v = 0; v < VOCAB; v++) {
                double[] q = model.sq(v);
                traj.write(String.format("%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%d%n",
                        seed, epoch, v, q[0], q[1], q[2], q[3],
                        SplitQuat.norm(q), model.p(v)));
            }

            model.zeroGrad();
            double epochCe = 0.0;
            for (int i = 0; i < XS.length; i++) {
                model.forward(XS[i]);
                epochCe += model.crossEntropyLoss(YS[i]);
                model.backward(YS[i]);
            }
            epochCe /= XS.length;
            loss.write(String.format("%d,%d,%.8e%n", seed, epoch, epochCe));

            model.step(LR / XS.length);
            List<KsqpModel.EventRecord> evs = model.detectEvents(epoch);
            for (KsqpModel.EventRecord ev : evs) {
                double[] s = ev.sqAtEvent();
                eventsOut.write(String.format("%d,%d,%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e%n",
                        seed, ev.epoch(), ev.tokenId(), ev.prevP(), ev.newP(),
                        ev.normBefore(), ev.normAfter(), s[0], s[1], s[2], s[3]));
            }
            eventCount += evs.size();

            if (epoch > 0 && epoch % 200 == 0) {
                double m0 = magnitude(model.sq(0));
                double m1 = magnitude(model.sq(1));
                if (!Double.isFinite(m0) || !Double.isFinite(m1) || m0 > 1e8 || m1 > 1e8) {
                    diverged = true;
                    break;
                }
            }
        }

        int correct = 0;
        double ce = 0.0;
        for (int i = 0; i < XS.length; i++) {
            double[] logits = model.forward(XS[i]);
            if (argmax(logits) == YS[i]) correct++;
            ce += model.crossEntropyLoss(YS[i]);
        }
        ce /= XS.length;

        double[] sq0 = model.sq(0);
        double[] sq1 = model.sq(1);
        boolean allZero = magnitude(sq0) < 1e-6 && magnitude(sq1) < 1e-6;

        return new RunResult(correct, ce, diverged, allZero,
                new int[] { model.p(0), model.p(1) },
                new double[] { SplitQuat.norm(sq0), SplitQuat.norm(sq1) },
                eventCount);
    }

    private record RunResult(int correct, double ce, boolean diverged, boolean allZero,
                             int[] finalP, double[] finalN,
                             int eventCount) {}

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

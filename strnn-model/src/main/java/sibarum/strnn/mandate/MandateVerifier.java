package sibarum.strnn.mandate;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;

import java.util.HashMap;
import java.util.Map;

/**
 * §6.4 structural verifier. After ComputationGraph.execute() has populated
 * each node's produced value, verify(...) checks each mandate against every
 * node and enforces ordering constraints via topological reachability.
 *
 * &quot;Somewhere&quot; (§6.1) is implemented literally: any node in the executed DAG
 * counts. The result mandate is exempt — it is matched specifically against
 * the terminal node, since that is the value the computation actually returns.
 */
public final class MandateVerifier {

    public VerificationReport verify(ComputationGraph cg, MandateSet mandates) {
        VerificationReport report = new VerificationReport();
        Map<Mandate, CompGraphNode> producedBy = new HashMap<>();

        for (Mandate m : mandates.mandates()) {
            CompGraphNode found = null;
            if (m.isResult()) {
                if (m.matches(cg.terminal().producedValue())) {
                    found = cg.terminal();
                }
            } else {
                for (CompGraphNode n : cg.nodes()) {
                    if (n.producedValue() != null && m.matches(n.producedValue())) {
                        found = n;
                        break;
                    }
                }
            }
            if (found == null) {
                String reason = m.isResult()
                        ? "terminal node did not produce expected value"
                        : "no node produced a value matching this mandate";
                report.record(m, VerificationReport.Outcome.failed(reason));
            } else {
                producedBy.put(m, found);
                report.record(m, VerificationReport.Outcome.satisfied(found));
            }
        }

        for (int i = 0; i < mandates.mandates().size(); i++) {
            Mandate mi = mandates.mandates().get(i);
            VerificationReport.Outcome outI = report.get(mi);
            if (!outI.satisfied()) continue;
            for (int j = i + 1; j < mandates.mandates().size(); j++) {
                Mandate mj = mandates.mandates().get(j);
                if (mj.ordering() <= mi.ordering()) continue;
                VerificationReport.Outcome outJ = report.get(mj);
                if (!outJ.satisfied()) continue;
                if (!cg.reaches(producedBy.get(mi), producedBy.get(mj))) {
                    report.record(mj, VerificationReport.Outcome.failed(
                            "ordering violation: '" + mi.name() + "' (ord=" + mi.ordering()
                                    + ") was produced by " + producedBy.get(mi).id()
                                    + " which does not topologically precede " + producedBy.get(mj).id()
                                    + " producing '" + mj.name() + "' (ord=" + mj.ordering() + ")"));
                }
            }
        }

        return report;
    }
}

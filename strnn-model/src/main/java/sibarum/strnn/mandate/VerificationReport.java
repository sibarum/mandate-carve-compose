package sibarum.strnn.mandate;

import sibarum.strnn.computation.CompGraphNode;

import java.util.LinkedHashMap;
import java.util.Map;

public final class VerificationReport {
    private final Map<Mandate, Outcome> outcomes = new LinkedHashMap<>();

    public void record(Mandate mandate, Outcome outcome) {
        outcomes.put(mandate, outcome);
    }

    public Outcome get(Mandate mandate) {
        return outcomes.get(mandate);
    }

    public boolean allSatisfied() {
        for (Outcome o : outcomes.values()) if (!o.satisfied()) return false;
        return true;
    }

    public int satisfiedCount() {
        int c = 0;
        for (Outcome o : outcomes.values()) if (o.satisfied()) c++;
        return c;
    }

    public int total() {
        return outcomes.size();
    }

    public Map<Mandate, Outcome> outcomes() {
        return outcomes;
    }

    public double fractionSatisfied() {
        return total() == 0 ? 1.0 : (double) satisfiedCount() / total();
    }

    public record Outcome(boolean satisfied, CompGraphNode producedBy, String reason) {
        public static Outcome satisfied(CompGraphNode by) {
            return new Outcome(true, by, null);
        }

        public static Outcome failed(String reason) {
            return new Outcome(false, null, reason);
        }
    }
}

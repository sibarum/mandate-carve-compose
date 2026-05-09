package sibarum.strnn.mandate;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class MandateSet {
    private final List<Mandate> mandates;

    public MandateSet(List<Mandate> mandates) {
        List<Mandate> sorted = new ArrayList<>(mandates);
        sorted.sort(Comparator.comparingInt(Mandate::ordering));
        this.mandates = List.copyOf(sorted);
    }

    public List<Mandate> mandates() {
        return mandates;
    }

    public Mandate result() {
        for (Mandate m : mandates) if (m.isResult()) return m;
        return null;
    }

    public int size() {
        return mandates.size();
    }
}

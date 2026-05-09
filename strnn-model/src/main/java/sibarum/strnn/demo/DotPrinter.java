package sibarum.strnn.demo;

import sibarum.strnn.computation.CompGraphNode;
import sibarum.strnn.computation.ComputationGraph;
import sibarum.strnn.computation.SlotSource;

/**
 * Renders a ComputationGraph in Graphviz DOT format. Used for the §9.6
 * ablation diff: dump A (full mandate set) and B (result-only) and compare
 * their structures by eye or by simple node-count metrics.
 */
public final class DotPrinter {
    private DotPrinter() {
    }

    public static String toDot(ComputationGraph cg, String name) {
        StringBuilder sb = new StringBuilder();
        sb.append("digraph ").append(name).append(" {\n");
        sb.append("  rankdir=LR;\n");
        sb.append("  node [shape=box, style=rounded, fontname=\"Consolas\"];\n");
        for (CompGraphNode n : cg.topoOrder()) {
            String label = n.id() + "\\n" + n.tNode().primitive().name();
            String value = n.producedValue() == null ? "" : "\\n=" + escape(String.valueOf(n.producedValue()));
            String style = (n == cg.terminal()) ? ", color=darkgreen, penwidth=2" : "";
            sb.append("  ").append(n.id())
                    .append(" [label=\"").append(label).append(value).append("\"")
                    .append(style).append("];\n");
        }
        for (CompGraphNode n : cg.topoOrder()) {
            for (int i = 0; i < n.slotCount(); i++) {
                SlotSource s = n.slot(i);
                if (s == null) continue;
                sb.append("  ").append(s.source().id()).append(" -> ").append(n.id())
                        .append(" [label=\"slot").append(i).append("\"];\n");
            }
        }
        sb.append("}\n");
        return sb.toString();
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }
}

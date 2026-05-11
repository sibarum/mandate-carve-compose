package sibarum.elden.demo;

import sibarum.elden.annotation.AnnotatedItem;
import sibarum.elden.annotation.DungEaterAnnotations;
import sibarum.elden.annotation.ImplicitEntities;
import sibarum.elden.annotation.MillicentAnnotations;
import sibarum.elden.annotation.RelationType;
import sibarum.elden.annotation.ShatteringEraAnnotations;
import sibarum.elden.annotation.VolcanoManorAnnotations;
import sibarum.elden.graph.EntityGraph;
import sibarum.elden.graph.EntityGraphBuilder;
import sibarum.elden.graph.EntityNode;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class GraphDemo {

    public static void main(String[] args) {
        List<AnnotatedItem> all = new ArrayList<>();
        all.addAll(ShatteringEraAnnotations.items());
        all.addAll(MillicentAnnotations.items());
        all.addAll(DungEaterAnnotations.items());
        all.addAll(VolcanoManorAnnotations.items());

        EntityGraph graph = EntityGraphBuilder.build(all, ImplicitEntities.all());

        System.out.println("Cross-item entity graph");
        System.out.println("=======================");
        System.out.println("entities: " + graph.entities().size());
        System.out.println("relations: " + graph.relations().size());
        System.out.println();

        System.out.println("entities by type:");
        graph.typeCounts().forEach((t, c) -> System.out.printf("  %-12s %d%n", t, c));
        long untypedCount = graph.untypedEntities().size();
        System.out.printf("  %-12s %d%n", "(untyped)", untypedCount);
        System.out.println();

        System.out.println("relations by type:");
        graph.relationCounts().forEach((t, c) -> System.out.printf("  %-14s %d%n", t, c));
        System.out.println();

        System.out.println("untyped entities (declared only in relations):");
        for (String id : graph.untypedEntities()) {
            System.out.println("  " + id);
        }
        System.out.println();

        System.out.println("top 10 most-connected entities:");
        for (EntityNode node : graph.mostConnected(10)) {
            System.out.printf("  %-32s type=%-10s spans=%d relations=%d surfaces=%s%n",
                    node.id(),
                    node.type().map(Enum::name).orElse("?"),
                    node.spanOccurrences(),
                    node.relationOccurrences(),
                    node.surfaceForms());
        }
        System.out.println();

        System.out.println("causal closure from 'night_of_black_knives' (CAUSED only):");
        Set<String> reachable = graph.reachableFrom("night_of_black_knives", Set.of(RelationType.CAUSED));
        for (String id : reachable) {
            System.out.println("  " + id);
        }
    }
}

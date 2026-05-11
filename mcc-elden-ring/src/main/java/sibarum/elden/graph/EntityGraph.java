package sibarum.elden.graph;

import sibarum.elden.annotation.EntityType;
import sibarum.elden.annotation.Relation;
import sibarum.elden.annotation.RelationType;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * Consolidated entity-and-relation graph built from all annotated items. Carries
 * the union of every entity ever named and every relation ever declared, plus
 * lightweight query methods for inspection.
 */
public record EntityGraph(
        Map<String, EntityNode> entities,
        List<Relation> relations
) {

    public List<String> untypedEntities() {
        return entities.values().stream()
                .filter(e -> e.type().isEmpty())
                .map(EntityNode::id)
                .sorted()
                .toList();
    }

    public Map<EntityType, Long> typeCounts() {
        Map<EntityType, Long> counts = new TreeMap<>();
        for (EntityNode node : entities.values()) {
            node.type().ifPresent(t -> counts.merge(t, 1L, Long::sum));
        }
        return counts;
    }

    public Map<RelationType, Long> relationCounts() {
        Map<RelationType, Long> counts = new TreeMap<>();
        for (Relation r : relations) {
            counts.merge(r.type(), 1L, Long::sum);
        }
        return counts;
    }

    public List<Relation> outgoing(String entityId) {
        return relations.stream()
                .filter(r -> r.subjectId().equals(entityId))
                .toList();
    }

    public List<Relation> incoming(String entityId) {
        return relations.stream()
                .filter(r -> r.objectId().equals(entityId))
                .toList();
    }

    public List<EntityNode> mostConnected(int n) {
        return entities.values().stream()
                .sorted(Comparator.<EntityNode>comparingInt(e -> e.spanOccurrences() + e.relationOccurrences()).reversed())
                .limit(n)
                .toList();
    }

    public Set<String> reachableFrom(String startId, Set<RelationType> via) {
        Map<String, List<Relation>> adj = relations.stream()
                .filter(r -> via.contains(r.type()))
                .collect(Collectors.groupingBy(Relation::subjectId));
        Set<String> visited = new java.util.LinkedHashSet<>();
        java.util.Deque<String> frontier = new java.util.ArrayDeque<>();
        frontier.push(startId);
        while (!frontier.isEmpty()) {
            String here = frontier.pop();
            if (!visited.add(here)) continue;
            for (Relation r : adj.getOrDefault(here, List.of())) {
                frontier.push(r.objectId());
            }
        }
        return visited;
    }
}

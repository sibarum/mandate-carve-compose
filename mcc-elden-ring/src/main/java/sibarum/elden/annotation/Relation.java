package sibarum.elden.annotation;

public record Relation(
        String subjectId,
        RelationType type,
        String objectId
) {}

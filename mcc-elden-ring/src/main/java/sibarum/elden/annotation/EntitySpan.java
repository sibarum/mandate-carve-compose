package sibarum.elden.annotation;

public record EntitySpan(
        int start,
        int end,
        String surface,
        EntityType type,
        String entityId
) {}

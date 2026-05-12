package sibarum.elden.data;

import sibarum.elden.annotation.EntityType;

import java.util.List;

/**
 * One typed-span training example for the entity-type classifier head.
 *
 * @param tokens     full sentence tokens (context window comes from here)
 * @param spanStart  index of the first token in the entity span
 * @param spanEnd    index of the last token in the entity span (inclusive)
 * @param type       gold {@link EntityType} for this span
 */
public record TypedSpanExample(
        List<String> tokens,
        int spanStart,
        int spanEnd,
        EntityType type
) {}

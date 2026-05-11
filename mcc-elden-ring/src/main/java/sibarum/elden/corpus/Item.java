package sibarum.elden.corpus;

import java.util.Optional;

public record Item(
        String id,
        String name,
        ItemCategory category,
        String description,
        Optional<String> location
) {
    public static Item of(String id, String name, ItemCategory category, String description) {
        return new Item(id, name, category, description, Optional.empty());
    }

    public static Item of(String id, String name, ItemCategory category, String description, String location) {
        return new Item(id, name, category, description, Optional.of(location));
    }
}

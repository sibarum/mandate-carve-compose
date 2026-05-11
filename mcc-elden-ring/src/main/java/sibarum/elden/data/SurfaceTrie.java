package sibarum.elden.data;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Token-level trie for longest-match surface-form lookup over a token sequence.
 * Keys are lowercased so the matcher is case-insensitive.
 *
 * Insertion is O(surface length); longest-match lookup at a position is O(walk
 * depth), independent of the dictionary size.
 */
public final class SurfaceTrie {

    private final Map<String, SurfaceTrie> children = new HashMap<>();
    private boolean isEnd = false;
    private int length = 0;

    /** Insert one surface form (already token-split) into the trie. */
    public void insert(List<String> tokens) {
        if (tokens.isEmpty()) return;
        SurfaceTrie cur = this;
        for (String t : tokens) {
            String key = t.toLowerCase(Locale.ROOT);
            cur = cur.children.computeIfAbsent(key, k -> new SurfaceTrie());
        }
        // Keep the longer surface if there's a collision.
        if (!cur.isEnd || cur.length < tokens.size()) {
            cur.isEnd = true;
            cur.length = tokens.size();
        }
    }

    /**
     * Length of the longest surface form matching at position {@code start}.
     * Returns 0 if no surface form starts at this position.
     */
    public int longestMatch(List<String> tokens, int start) {
        SurfaceTrie cur = this;
        int matched = 0;
        for (int i = start; i < tokens.size(); i++) {
            String key = tokens.get(i).toLowerCase(Locale.ROOT);
            SurfaceTrie child = cur.children.get(key);
            if (child == null) break;
            cur = child;
            if (cur.isEnd) matched = cur.length;
        }
        return matched;
    }
}

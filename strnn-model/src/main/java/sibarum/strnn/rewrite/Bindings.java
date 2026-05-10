package sibarum.strnn.rewrite;

import sibarum.strnn.value.ParseTreeValue;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Hole-name → bound-subtree map produced by Matcher.match. Immutable view
 * exposed to callers; internally uses a HashMap.
 */
public final class Bindings {
    private final Map<String, ParseTreeValue> map;

    private Bindings(Map<String, ParseTreeValue> map) {
        this.map = map;
    }

    public static Bindings empty() {
        return new Bindings(Map.of());
    }

    public Optional<ParseTreeValue> get(String name) {
        return Optional.ofNullable(map.get(name));
    }

    public ParseTreeValue require(String name) {
        ParseTreeValue v = map.get(name);
        if (v == null) {
            throw new IllegalStateException("no binding for hole '" + name + "'");
        }
        return v;
    }

    public Map<String, ParseTreeValue> asMap() {
        return Map.copyOf(map);
    }

    public boolean has(String name) {
        return map.containsKey(name);
    }

    public int size() {
        return map.size();
    }

    @Override
    public String toString() {
        return map.toString();
    }

    /** Mutable builder used internally by Matcher. */
    public static final class Builder {
        private final Map<String, ParseTreeValue> map = new HashMap<>();

        public boolean bind(String name, ParseTreeValue tree) {
            ParseTreeValue existing = map.get(name);
            if (existing == null) {
                map.put(name, tree);
                return true;
            }
            return existing.equals(tree);
        }

        public Bindings build() {
            return new Bindings(Map.copyOf(map));
        }
    }
}

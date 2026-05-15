package sibarum.mcc.serialization;

import java.util.List;
import java.util.Map;

/**
 * Minimal reflection-free JSON writer. Accepts {@code null},
 * {@link Boolean}, {@link Number}, {@link String}, {@code int[]},
 * {@code long[]}, {@code double[]}, {@link List}, and {@link Map}
 * (string keys only). Indentation is configurable.
 *
 * <p>Reflection-free by design — works in GraalVM native-image without
 * registered reachability metadata. The mcc-core schema is small and
 * stable enough that hand-rolling the marshallers is the right
 * trade-off vs. Jackson.
 */
public final class JsonWriter {

    private final StringBuilder out = new StringBuilder();
    private final boolean indent;
    private int level;

    public JsonWriter() {
        this(true);
    }

    public JsonWriter(boolean indent) {
        this.indent = indent;
    }

    public String result() {
        return out.toString();
    }

    public void writeValue(Object v) {
        if (v == null) {
            out.append("null");
        } else if (v instanceof Boolean b) {
            out.append(b ? "true" : "false");
        } else if (v instanceof Number n) {
            writeNumber(n);
        } else if (v instanceof String s) {
            writeString(s);
        } else if (v instanceof Map<?, ?> m) {
            writeObject(m);
        } else if (v instanceof List<?> l) {
            writeArray(l);
        } else if (v instanceof int[] arr) {
            writeIntArray(arr);
        } else if (v instanceof long[] arr) {
            writeLongArray(arr);
        } else if (v instanceof double[] arr) {
            writeDoubleArray(arr);
        } else {
            throw new IllegalArgumentException(
                    "JsonWriter: unsupported value type " + v.getClass().getName());
        }
    }

    private void writeNumber(Number n) {
        if (n instanceof Double || n instanceof Float) {
            double d = n.doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                throw new IllegalArgumentException(
                        "JsonWriter: non-finite numeric value " + d + " is not valid JSON");
            }
            out.append(d);
        } else {
            out.append(n);
        }
    }

    private void writeString(String s) {
        out.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"'  -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
    }

    private void writeObject(Map<?, ?> m) {
        if (m.isEmpty()) {
            out.append("{}");
            return;
        }
        out.append('{');
        level++;
        boolean first = true;
        for (Map.Entry<?, ?> e : m.entrySet()) {
            if (!first) out.append(',');
            first = false;
            newline();
            if (!(e.getKey() instanceof String key)) {
                throw new IllegalArgumentException("JsonWriter: map keys must be String");
            }
            writeString(key);
            out.append(indent ? ": " : ":");
            writeValue(e.getValue());
        }
        level--;
        newline();
        out.append('}');
    }

    private void writeArray(List<?> l) {
        if (l.isEmpty()) {
            out.append("[]");
            return;
        }
        out.append('[');
        level++;
        boolean first = true;
        for (Object v : l) {
            if (!first) out.append(',');
            first = false;
            newline();
            writeValue(v);
        }
        level--;
        newline();
        out.append(']');
    }

    private void writeIntArray(int[] arr) {
        out.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) out.append(indent ? ", " : ",");
            out.append(arr[i]);
        }
        out.append(']');
    }

    private void writeLongArray(long[] arr) {
        out.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) out.append(indent ? ", " : ",");
            out.append(arr[i]);
        }
        out.append(']');
    }

    private void writeDoubleArray(double[] arr) {
        out.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) out.append(indent ? ", " : ",");
            double d = arr[i];
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                throw new IllegalArgumentException(
                        "JsonWriter: non-finite double in array: " + d);
            }
            out.append(d);
        }
        out.append(']');
    }

    private void newline() {
        if (!indent) return;
        out.append('\n');
        for (int i = 0; i < level; i++) out.append("  ");
    }
}

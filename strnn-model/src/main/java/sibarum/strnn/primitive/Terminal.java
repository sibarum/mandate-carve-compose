package sibarum.strnn.primitive;

/**
 * Marker for primitives whose only legitimate position is the terminal of a
 * computation graph. The carver excludes these from non-terminal candidate
 * consideration so they don't appear as passthrough nodes in the middle of
 * a carved chain. Both OutputPrimitive (NumberValue terminal) and
 * TreeOutputPrimitive (ParseTreeValue terminal) implement this.
 */
public interface Terminal extends Primitive {
}

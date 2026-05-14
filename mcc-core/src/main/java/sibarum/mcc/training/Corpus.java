package sibarum.mcc.training;

import java.util.Iterator;

/**
 * SPI for an iterable training/validation dataset. Implementations
 * decide where examples come from (generated, file-backed, sampled).
 * Calling {@link #stream} returns a fresh iterator each time, allowing
 * multi-epoch passes.
 *
 * <p>Implementations should be deterministic given the same seed when
 * sampling is involved, so training runs are reproducible.
 */
public interface Corpus {

    /** Human-readable name for logging / export metadata. */
    String name();

    /** Total number of examples in one pass, or {@code -1} if unknown / unbounded. */
    long size();

    /** Fresh iterator over the corpus. May be called repeatedly for multi-epoch training. */
    Iterator<Example> stream();
}

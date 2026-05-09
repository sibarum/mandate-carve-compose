package sibarum.strnn.training;

import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.value.Value;

/**
 * One training instance: input value, full mandate specification (which
 * already names the result mandate), and a human-readable label for logging.
 */
public record Example(String label, Value payload, MandateSet mandates) {
}

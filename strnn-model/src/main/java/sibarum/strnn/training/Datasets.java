package sibarum.strnn.training;

import sibarum.strnn.mandate.Mandate;
import sibarum.strnn.mandate.MandateSet;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.TokenListValue;

import java.util.List;
import java.util.Random;

/**
 * Synthetic generator for the v0 demo task. Produces single-digit arithmetic
 * expressions of the form &quot;a+b*c&quot; or &quot;a*b+c&quot;, each annotated with the full
 * §9.2 mandate set: token splits at both delimiters, the multiplication
 * intermediate, and the final result.
 */
public final class Datasets {
    public enum Shape {
        ADD_THEN_MUL, MUL_THEN_ADD
    }

    public static Example generate(Random rng, Shape shape) {
        int a = rng.nextInt(10);
        int b = rng.nextInt(10);
        int c = rng.nextInt(10);
        return shapeExample(shape, a, b, c);
    }

    public static Example shapeExample(Shape shape, int a, int b, int c) {
        return switch (shape) {
            case ADD_THEN_MUL -> addThenMul(a, b, c);
            case MUL_THEN_ADD -> mulThenAdd(a, b, c);
        };
    }

    public static Example addThenMul(int a, int b, int c) {
        String expr = a + "+" + b + "*" + c;
        int product = b * c;
        int result = a + product;
        MandateSet ms = new MandateSet(List.of(
                Mandate.intermediate("plus_split",
                        new TokenListValue(List.of(Integer.toString(a), b + "*" + c)), 0.0, 0),
                Mandate.intermediate("star_split",
                        new TokenListValue(List.of(Integer.toString(b), Integer.toString(c))), 0.0, 1),
                Mandate.intermediate("intermediate_product",
                        new NumberValue(product), 0.5, 2),
                Mandate.result(new NumberValue(result), 0.5, 3)));
        return new Example(expr, new StringValue(expr), ms);
    }

    public static Example mulThenAdd(int a, int b, int c) {
        String expr = a + "*" + b + "+" + c;
        int product = a * b;
        int result = product + c;
        MandateSet ms = new MandateSet(List.of(
                Mandate.intermediate("plus_split",
                        new TokenListValue(List.of(a + "*" + b, Integer.toString(c))), 0.0, 0),
                Mandate.intermediate("star_split",
                        new TokenListValue(List.of(Integer.toString(a), Integer.toString(b))), 0.0, 1),
                Mandate.intermediate("intermediate_product",
                        new NumberValue(product), 0.5, 2),
                Mandate.result(new NumberValue(result), 0.5, 3)));
        return new Example(expr, new StringValue(expr), ms);
    }

    @SuppressWarnings("DataFlowIssue")
    public static Example resultOnly(Example src) {
        Mandate result = src.mandates().result();
        return new Example(src.label(), src.payload(),
                new MandateSet(List.of(result)));
    }

    private Datasets() {
    }
}

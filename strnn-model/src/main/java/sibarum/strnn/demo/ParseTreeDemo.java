package sibarum.strnn.demo;

import sibarum.strnn.primitive.ParseExpression;
import sibarum.strnn.value.Operator;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;
import sibarum.strnn.value.ValueDistance;

import java.util.List;

/**
 * v2 Phase 0 smoke test: ParseTreeValue + ParseExpression primitive.
 * Confirms parsing is correct, structural equality works, and ValueDistance
 * handles tree-shaped values.
 */
public final class ParseTreeDemo {

    public static void main(String[] args) {
        ParseExpression parser = new ParseExpression();

        // Round-trip 1: "(2 + 3) * 4" -> ((2+3)*4)
        Value parsed1 = parser.apply(List.of(new StringValue("(2 + 3) * 4")));
        ParseTreeValue expected1 = ParseTreeValue.mul(
                ParseTreeValue.add(ParseTreeValue.lit(2), ParseTreeValue.lit(3)),
                ParseTreeValue.lit(4));
        assertEquals(expected1, parsed1, "case 1");
        System.out.println("(2 + 3) * 4  -> " + parsed1);

        // Round-trip 2: precedence "2 + 3 * 4" -> (2 + (3*4))
        Value parsed2 = parser.apply(List.of(new StringValue("2 + 3 * 4")));
        ParseTreeValue expected2 = ParseTreeValue.add(
                ParseTreeValue.lit(2),
                ParseTreeValue.mul(ParseTreeValue.lit(3), ParseTreeValue.lit(4)));
        assertEquals(expected2, parsed2, "case 2");
        System.out.println("2 + 3 * 4    -> " + parsed2);

        // Round-trip 3: variables and mixed ops "(a + 0) * 1"
        Value parsed3 = parser.apply(List.of(new StringValue("(a + 0) * 1")));
        ParseTreeValue expected3 = ParseTreeValue.mul(
                ParseTreeValue.add(ParseTreeValue.var("a"), ParseTreeValue.lit(0)),
                ParseTreeValue.lit(1));
        assertEquals(expected3, parsed3, "case 3");
        System.out.println("(a + 0) * 1  -> " + parsed3);

        // ValueDistance check
        ParseTreeValue same = ParseTreeValue.lit(7);
        ParseTreeValue alsoSame = new ParseTreeValue.Literal(7);
        ParseTreeValue diff = ParseTreeValue.lit(8);
        if (ValueDistance.distance(same, alsoSame) != 0.0) {
            throw new AssertionError("expected distance 0 for identical literals");
        }
        if (ValueDistance.distance(same, diff) != 1.0) {
            throw new AssertionError("expected distance 1 for different literals");
        }
        if (!ValueDistance.matches(same, alsoSame, 0.0)) {
            throw new AssertionError("expected matches=true for identical literals");
        }

        // Cross-type: matching against a non-tree
        if (ValueDistance.matches(same, new sibarum.strnn.value.NumberValue(7.0), 0.0)) {
            throw new AssertionError("expected matches=false across types");
        }

        // Size sanity
        if (parsed1 instanceof ParseTreeValue p1 && p1.size() != 5) {
            throw new AssertionError("expected size 5 for (2+3)*4, got " + p1.size());
        }

        // Operator round-trip
        if (Operator.fromChar('+') != Operator.ADD) throw new AssertionError("operator add");

        System.out.println("\nv2 Phase 0 OK.");
    }

    private static void assertEquals(Object expected, Object actual, String label) {
        if (!expected.equals(actual)) {
            throw new AssertionError(label + ": expected " + expected + " but got " + actual);
        }
    }
}

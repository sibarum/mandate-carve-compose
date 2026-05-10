package sibarum.strnn.demo;

import sibarum.strnn.rewrite.Bindings;
import sibarum.strnn.rewrite.Matcher;
import sibarum.strnn.rewrite.Pattern;
import sibarum.strnn.value.ParseTreeValue;

import java.util.Optional;

/**
 * v2 Phase 1 smoke test: pattern matching and substitution. Five cases:
 *   1. Linear match succeeds, all holes bind.
 *   2. Non-matching pattern fails cleanly.
 *   3. Non-linear pattern (?a + ?a) succeeds when both occurrences are equal.
 *   4. Non-linear pattern fails when occurrences differ.
 *   5. Substitution round-trip: distribute (?a + ?b) * ?c -&gt; ?a*?c + ?b*?c.
 */
public final class PatternMatchingDemo {

    public static void main(String[] args) {
        // Case 1: linear match
        Pattern p1 = Pattern.mul(
                Pattern.add(Pattern.hole("a"), Pattern.hole("b")),
                Pattern.hole("c"));
        ParseTreeValue t1 = ParseTreeValue.mul(
                ParseTreeValue.add(ParseTreeValue.lit(2), ParseTreeValue.lit(3)),
                ParseTreeValue.lit(4));
        Optional<Bindings> b1 = Matcher.match(p1, t1);
        if (b1.isEmpty()) throw new AssertionError("case 1 should match");
        System.out.println("case 1 (linear match): " + b1.get());
        assertSame(b1.get().require("a"), ParseTreeValue.lit(2));
        assertSame(b1.get().require("b"), ParseTreeValue.lit(3));
        assertSame(b1.get().require("c"), ParseTreeValue.lit(4));

        // Case 2: non-matching pattern
        Pattern p2 = Pattern.add(Pattern.hole("x"), Pattern.hole("y"));
        ParseTreeValue t2 = ParseTreeValue.mul(ParseTreeValue.lit(5), ParseTreeValue.lit(6));
        Optional<Bindings> b2 = Matcher.match(p2, t2);
        if (b2.isPresent()) throw new AssertionError("case 2 should not match");
        System.out.println("case 2 (no match):     ok");

        // Case 3: non-linear, equal subtrees
        Pattern p3 = Pattern.add(Pattern.hole("a"), Pattern.hole("a"));
        ParseTreeValue t3 = ParseTreeValue.add(ParseTreeValue.lit(7), ParseTreeValue.lit(7));
        Optional<Bindings> b3 = Matcher.match(p3, t3);
        if (b3.isEmpty()) throw new AssertionError("case 3 should match");
        System.out.println("case 3 (non-linear ok): " + b3.get());

        // Case 4: non-linear, different subtrees
        Pattern p4 = Pattern.add(Pattern.hole("a"), Pattern.hole("a"));
        ParseTreeValue t4 = ParseTreeValue.add(ParseTreeValue.lit(7), ParseTreeValue.lit(8));
        Optional<Bindings> b4 = Matcher.match(p4, t4);
        if (b4.isPresent()) throw new AssertionError("case 4 should not match");
        System.out.println("case 4 (non-linear bad): ok");

        // Case 5: distribute substitution
        Pattern lhs = Pattern.mul(
                Pattern.add(Pattern.hole("a"), Pattern.hole("b")),
                Pattern.hole("c"));
        Pattern rhs = Pattern.add(
                Pattern.mul(Pattern.hole("a"), Pattern.hole("c")),
                Pattern.mul(Pattern.hole("b"), Pattern.hole("c")));
        Optional<Bindings> b5 = Matcher.match(lhs, t1);
        if (b5.isEmpty()) throw new AssertionError("case 5 should match");
        ParseTreeValue rewritten = Matcher.substitute(rhs, b5.get());
        ParseTreeValue expected = ParseTreeValue.add(
                ParseTreeValue.mul(ParseTreeValue.lit(2), ParseTreeValue.lit(4)),
                ParseTreeValue.mul(ParseTreeValue.lit(3), ParseTreeValue.lit(4)));
        if (!rewritten.equals(expected)) {
            throw new AssertionError("case 5 substitution mismatch: " + rewritten);
        }
        System.out.println("case 5 (distribute):   " + t1 + "  ->  " + rewritten);

        System.out.println("\nv2 Phase 1 OK.");
    }

    private static void assertSame(ParseTreeValue actual, ParseTreeValue expected) {
        if (!actual.equals(expected)) {
            throw new AssertionError("expected " + expected + " but got " + actual);
        }
    }
}

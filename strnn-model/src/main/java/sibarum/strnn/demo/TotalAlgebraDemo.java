package sibarum.strnn.demo;

import sibarum.strnn.primitive.ParseExpression;
import sibarum.strnn.rewrite.DivByZero;
import sibarum.strnn.rewrite.IdentityZero;
import sibarum.strnn.rewrite.Reversible;
import sibarum.strnn.rewrite.RewriteRulePrimitive;
import sibarum.strnn.rewrite.ZeroDivZero;
import sibarum.strnn.rewrite.ZeroTimesOmega;
import sibarum.strnn.value.Operator;
import sibarum.strnn.value.ParseTreeValue;
import sibarum.strnn.value.StringValue;

import java.util.List;
import java.util.Optional;

/**
 * v3 Phase 0: smallest possible runnable artifact for the total, reversible
 * algebra. No carver, no mandates — just the rules-substrate directly,
 * exercising the identities that distinguish this algebra from the standard one:
 *
 *   1) ?a / 0 → ?a · ω  (no undefined; division by zero is multiplication by ω,
 *                          and coefficients carry through ω as for any variable)
 *   2) 0 / 0 → 1        (no indeterminate)
 *   3) 0 · ω → 1        (structurally the same identity as 0/0 = 1)
 *   4) 0 · x stays      (no annihilation; there is no 0·x → 0 rule)
 *   5) 2·ω, ω+2, ω·ω    (irreducible — ω is a variable-like leaf, not flattened)
 *   6) round-trip       (rules marked Reversible recover the input via inferInput)
 *
 * Diagnostics (4) and (5) are the load-bearing ones: standard CAS bakes in
 * "0*x → 0" and treats ω-arithmetic as undefined or NaN. Here the rule library
 * has neither annihilator nor a normalizer that flattens products involving ω.
 * Diagnostic (6) shows that reversibility is opt-in via the marker interface
 * and verifiable at runtime; rules that don't claim it are free to be
 * destructive.
 */
public final class TotalAlgebraDemo {

    public static void main(String[] args) {
        ParseExpression parser = new ParseExpression();

        DivByZero divByZero = new DivByZero();
        ZeroDivZero zeroDivZero = new ZeroDivZero();
        ZeroTimesOmega zeroTimesOmega = new ZeroTimesOmega();

        List<RewriteRulePrimitive> totalAlgebraRules = List.of(divByZero, zeroDivZero, zeroTimesOmega);

        System.out.println("=== Diagnostic 1: ?a / 0 → ?a · ω  (general; coefficients carry) ===");
        // Three inputs covering literal, larger literal, and variable operands.
        // The rule treats ω like any variable would be treated; the ?a hole
        // captures whatever sits in the dividend slot and threads it through.
        List<String> dividends = List.of("1/0", "2/0", "x/0");
        for (String src : dividends) {
            ParseTreeValue input = parse(parser, src);
            require(divByZero.applies(input), "DivByZero must match " + src);
            ParseTreeValue out = (ParseTreeValue) divByZero.apply(List.of(input));
            ParseTreeValue.BinaryOp bo = (ParseTreeValue.BinaryOp) out;
            require(bo.op() == Operator.MUL && bo.right() instanceof ParseTreeValue.Omega,
                    "expected ?a · ω; got " + out);
            System.out.printf("  %-6s → %s%n", input, out);
        }
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 2: 0 / 0 → 1 ===");
        ParseTreeValue zeroOverZero = parse(parser, "0/0");
        System.out.println("  input:  " + zeroOverZero);
        require(zeroDivZero.applies(zeroOverZero), "ZeroDivZero must match 0/0");
        ParseTreeValue rewritten2 = (ParseTreeValue) zeroDivZero.apply(List.of(zeroOverZero));
        System.out.println("  output: " + rewritten2);
        require(rewritten2 instanceof ParseTreeValue.Literal lit && lit.value() == 1.0,
                "0/0 must rewrite to Literal(1); got: " + rewritten2);
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 3: 0 · ω → 1  (the same identity as 0/0 = 1) ===");
        ParseTreeValue zeroTimesOmegaTree = new ParseTreeValue.BinaryOp(
                Operator.MUL, ParseTreeValue.lit(0), ParseTreeValue.omega());
        System.out.println("  input:  " + zeroTimesOmegaTree);
        require(zeroTimesOmega.applies(zeroTimesOmegaTree), "ZeroTimesOmega must match 0·ω");
        ParseTreeValue rewritten3 = (ParseTreeValue) zeroTimesOmega.apply(List.of(zeroTimesOmegaTree));
        System.out.println("  output: " + rewritten3);
        require(rewritten3 instanceof ParseTreeValue.Literal lit && lit.value() == 1.0,
                "0·ω must rewrite to Literal(1); got: " + rewritten3);
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 4: 0 · x stays formal (no annihilation rule) ===");
        ParseTreeValue zeroTimesX = parse(parser, "0*x");
        System.out.println("  input:  " + zeroTimesX);
        for (RewriteRulePrimitive rule : totalAlgebraRules) {
            if (rule.applies(zeroTimesX)) {
                throw new AssertionError("annihilation leak: rule '" + rule.name()
                        + "' fired on 0*x — total algebra forbids this");
            }
        }
        // Cross-check: even with v2's identity rules in scope, 0·x does not
        // collapse — IdentityZero is `?a + 0`, not `?a · 0`, and there is no
        // multiplicative annihilator rule in the library.
        require(!new IdentityZero().applies(zeroTimesX),
                "IdentityZero must not match 0*x");
        System.out.println("  none of the total-algebra rules apply; tree is preserved");
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 5: reversibility is opt-in and round-trips ===");
        // For each rule that claims the Reversible marker, applying the rule
        // and then asking inferInput to invert must recover the input. Rules
        // that don't claim the marker are free to be destructive — the demo
        // simply doesn't test them here.
        List<RoundTrip> roundTrips = List.of(
                new RoundTrip(divByZero,      parse(parser, "x/0")),
                new RoundTrip(divByZero,      parse(parser, "2/0")),
                new RoundTrip(zeroDivZero,    parse(parser, "0/0")),
                new RoundTrip(zeroTimesOmega, new ParseTreeValue.BinaryOp(
                        Operator.MUL, ParseTreeValue.lit(0), ParseTreeValue.omega())));
        for (RoundTrip rt : roundTrips) {
            require(rt.rule instanceof Reversible,
                    "rule '" + rt.rule.name() + "' must claim the Reversible marker for round-trip");
            ParseTreeValue forward = (ParseTreeValue) rt.rule.apply(List.of(rt.input));
            Optional<ParseTreeValue> back = rt.rule.inferInput(forward);
            require(back.isPresent(),
                    "inferInput failed on " + forward + " for rule " + rt.rule.name());
            require(rt.input.equals(back.get()),
                    "round-trip mismatch for " + rt.rule.name() + ": " + rt.input + " → " + forward + " → " + back.get());
            System.out.printf("  [%s] %s → %s → %s%n",
                    rt.rule.name(), rt.input, forward, back.get());
        }
        System.out.println("  PASS");

        System.out.println("\n=== Diagnostic 6: Z-graded irreducibles preserved ===");
        // Elements of the integer-graded tower above and below the reals are
        // irreducible: 2·ω is at index +1 (with coefficient 2), ω+2 is a sum
        // across two distinct rungs, and ω·ω is at index +2 (Cantor's ω²). No
        // rule in the current library should fire on any of them.
        ParseTreeValue twoOmega = new ParseTreeValue.BinaryOp(
                Operator.MUL, ParseTreeValue.lit(2), ParseTreeValue.omega());
        ParseTreeValue omegaPlusTwo = new ParseTreeValue.BinaryOp(
                Operator.ADD, ParseTreeValue.omega(), ParseTreeValue.lit(2));
        ParseTreeValue omegaSquared = new ParseTreeValue.BinaryOp(
                Operator.MUL, ParseTreeValue.omega(), ParseTreeValue.omega());

        List<ParseTreeValue> irreducibles = List.of(twoOmega, omegaPlusTwo, omegaSquared);
        for (ParseTreeValue tree : irreducibles) {
            for (RewriteRulePrimitive rule : totalAlgebraRules) {
                if (rule.applies(tree)) {
                    throw new AssertionError(
                            "irreducible '" + tree + "' was reduced by rule '" + rule.name()
                                    + "' — total algebra preserves the Z-graded tower");
                }
            }
            System.out.println("  " + tree + " is irreducible (no rule applies)");
        }
        System.out.println("  PASS");

        System.out.println("\nv3 Phase 0: total algebra rule library "
                + "(div-by-zero general, zero-div-zero, zero-times-omega), "
                + "annihilation forbidden, ω-tower preserved, reversibility opt-in.");
    }

    private record RoundTrip(RewriteRulePrimitive rule, ParseTreeValue input) {
    }

    private static ParseTreeValue parse(ParseExpression parser, String s) {
        return (ParseTreeValue) parser.apply(List.of(new StringValue(s)));
    }

    private static void require(boolean cond, String msg) {
        if (!cond) {
            throw new AssertionError(msg);
        }
    }
}

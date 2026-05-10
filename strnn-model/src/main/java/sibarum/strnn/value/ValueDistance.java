package sibarum.strnn.value;

public final class ValueDistance {
    private ValueDistance() {}

    public static double distance(Value a, Value b) {
        if (a.type() != b.type()) return Double.POSITIVE_INFINITY;
        return switch (a) {
            case NumberValue na -> Math.abs(na.n() - ((NumberValue) b).n());
            case MatrixValue ma -> frobenius(ma.data(), ((MatrixValue) b).data());
            case StringValue sa -> sa.s().equals(((StringValue) b).s()) ? 0.0 : 1.0;
            case TokenListValue ta -> ta.tokens().equals(((TokenListValue) b).tokens()) ? 0.0 : 1.0;
            case ParseTreeValue pa -> pa.equals(b) ? 0.0 : 1.0;
        };
    }

    private static double frobenius(double[] x, double[] y) {
        if (x.length != y.length) return Double.POSITIVE_INFINITY;
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - y[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    public static boolean matches(Value expected, Value actual, double tolerance) {
        if (expected.type() != actual.type()) return false;
        return switch (expected) {
            case NumberValue ne -> Math.abs(ne.n() - ((NumberValue) actual).n()) <= tolerance;
            case MatrixValue me -> frobenius(me.data(), ((MatrixValue) actual).data()) <= tolerance;
            case StringValue se -> se.s().equals(((StringValue) actual).s());
            case TokenListValue te -> te.tokens().equals(((TokenListValue) actual).tokens());
            case ParseTreeValue pe -> treeMatches(pe, (ParseTreeValue) actual, tolerance);
        };
    }

    private static boolean treeMatches(ParseTreeValue expected, ParseTreeValue actual, double tolerance) {
        if (expected instanceof ParseTreeValue.Literal le && actual instanceof ParseTreeValue.Literal la) {
            return Math.abs(le.value() - la.value()) <= tolerance;
        }
        if (expected instanceof ParseTreeValue.Variable ve && actual instanceof ParseTreeValue.Variable va) {
            return ve.name().equals(va.name());
        }
        if (expected instanceof ParseTreeValue.Omega && actual instanceof ParseTreeValue.Omega) {
            return true;
        }
        if (expected instanceof ParseTreeValue.BinaryOp be && actual instanceof ParseTreeValue.BinaryOp ba) {
            return be.op() == ba.op()
                    && treeMatches(be.left(), ba.left(), tolerance)
                    && treeMatches(be.right(), ba.right(), tolerance);
        }
        return false;
    }
}

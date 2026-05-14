package sibarum.mcc.value;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ValueDistanceTest {

    private static final double TOL = 1e-12;

    @Test
    void differentTypesReturnInfinity() {
        Value a = new NumberValue(1.0);
        Value b = new StringValue("1.0");
        assertEquals(Double.POSITIVE_INFINITY, ValueDistance.distance(a, b));
        assertFalse(ValueDistance.matches(a, b, 0.0));
    }

    @Test
    void numberDistance() {
        Value a = new NumberValue(1.0);
        Value b = new NumberValue(1.25);
        assertEquals(0.25, ValueDistance.distance(a, b), TOL);
        assertTrue(ValueDistance.matches(a, b, 0.5));
        assertFalse(ValueDistance.matches(a, b, 0.1));
    }

    @Test
    void matrixDistanceFrobenius() {
        Value a = new MatrixValue(new double[] { 0.0, 0.0 });
        Value b = new MatrixValue(new double[] { 3.0, 4.0 });
        assertEquals(5.0, ValueDistance.distance(a, b), TOL);
    }

    @Test
    void matrixShapeMismatchInfinity() {
        Value a = new MatrixValue(new double[] { 1.0, 2.0 });
        Value b = new MatrixValue(new double[] { 1.0, 2.0, 3.0 });
        assertEquals(Double.POSITIVE_INFINITY, ValueDistance.distance(a, b));
    }

    @Test
    void ternionDistance() {
        Value a = new TernionValue(0.0, 0.0, 0.0);
        Value b = new TernionValue(2.0, 3.0, 6.0);
        assertEquals(7.0, ValueDistance.distance(a, b), TOL);
    }

    @Test
    void quaternionDistance() {
        Value a = new QuaternionValue(1.0, 0.0, 0.0, 0.0);
        Value b = new QuaternionValue(0.0, 1.0, 0.0, 0.0);
        assertEquals(Math.sqrt(2.0), ValueDistance.distance(a, b), TOL);
    }

    @Test
    void tensorSameShapeFrobenius() {
        Value a = new TensorValue(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        Value b = new TensorValue(new int[] { 2, 2 }, new double[] { 1, 2, 3, 6 });
        assertEquals(2.0, ValueDistance.distance(a, b), TOL);
    }

    @Test
    void tensorShapeMismatchInfinity() {
        Value a = new TensorValue(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        Value b = new TensorValue(new int[] { 4 },    new double[] { 1, 2, 3, 4 });
        assertEquals(Double.POSITIVE_INFINITY, ValueDistance.distance(a, b));
        assertFalse(ValueDistance.matches(a, b, 1.0));
    }

    @Test
    void stringMatchesExact() {
        Value a = new StringValue("abc");
        Value b = new StringValue("abc");
        Value c = new StringValue("abd");
        assertTrue(ValueDistance.matches(a, b, 0.0));
        assertFalse(ValueDistance.matches(a, c, 0.999));
    }
}

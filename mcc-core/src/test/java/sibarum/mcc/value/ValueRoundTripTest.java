package sibarum.mcc.value;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ValueRoundTripTest {

    @Test
    void stringValueRoundTrip() {
        StringValue a = new StringValue("hello");
        StringValue b = new StringValue("hello");
        assertEquals(a, b);
        assertEquals(ValueType.STRING, a.type());
    }

    @Test
    void numberValueRoundTrip() {
        NumberValue a = new NumberValue(3.14);
        NumberValue b = new NumberValue(3.14);
        assertEquals(a, b);
        assertEquals(ValueType.NUMBER, a.type());
    }

    @Test
    void matrixValueCopiesInput() {
        double[] data = { 1.0, 2.0, 3.0 };
        MatrixValue mv = new MatrixValue(data);
        data[0] = 999.0;
        assertEquals(1.0, mv.data()[0], 0.0,
                "MatrixValue must defensively copy its data on construction");
    }

    @Test
    void matrixValueEqualsByContents() {
        MatrixValue a = new MatrixValue(new double[] { 1.0, 2.0 });
        MatrixValue b = new MatrixValue(new double[] { 1.0, 2.0 });
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    void ternionValueRoundTrip() {
        TernionValue a = new TernionValue(1.0, 2.0, 3.0);
        TernionValue b = new TernionValue(new double[] { 1.0, 2.0, 3.0 });
        assertEquals(a, b);
        assertArrayEquals(new double[] { 1.0, 2.0, 3.0 }, a.toArray());
        assertEquals(ValueType.TERNION, a.type());
    }

    @Test
    void ternionValueRejectsWrongLength() {
        assertThrows(IllegalArgumentException.class,
                () -> new TernionValue(new double[] { 1.0, 2.0 }));
        assertThrows(IllegalArgumentException.class,
                () -> new TernionValue(new double[] { 1.0, 2.0, 3.0, 4.0 }));
    }

    @Test
    void quaternionValueRoundTrip() {
        QuaternionValue a = new QuaternionValue(1.0, 0.0, 0.0, 0.0);
        QuaternionValue b = new QuaternionValue(new double[] { 1.0, 0.0, 0.0, 0.0 });
        assertEquals(a, b);
        assertArrayEquals(new double[] { 1.0, 0.0, 0.0, 0.0 }, a.toArray());
        assertEquals(ValueType.QUATERNION, a.type());
    }

    @Test
    void quaternionValueRejectsWrongLength() {
        assertThrows(IllegalArgumentException.class,
                () -> new QuaternionValue(new double[] { 1.0, 0.0, 0.0 }));
    }

    @Test
    void tensorValueShapeVolume() {
        assertEquals(1, TensorValue.volume(new int[0]));
        assertEquals(6, TensorValue.volume(new int[] { 2, 3 }));
        assertEquals(24, TensorValue.volume(new int[] { 2, 3, 4 }));
    }

    @Test
    void tensorValueRoundTrip() {
        double[] flat = { 1, 2, 3, 4, 5, 6 };
        TensorValue t = new TensorValue(new int[] { 2, 3 }, flat);
        assertEquals(2, t.rank());
        assertEquals(6, t.size());
        assertEquals(2, t.dim(0));
        assertEquals(3, t.dim(1));
        assertEquals(ValueType.TENSOR, t.type());
    }

    @Test
    void tensorValueCopiesShapeAndData() {
        int[] shape = { 2, 2 };
        double[] data = { 1, 2, 3, 4 };
        TensorValue t = new TensorValue(shape, data);
        shape[0] = 999;
        data[0] = 999.0;
        assertEquals(2, t.shape()[0]);
        assertEquals(1.0, t.data()[0]);
    }

    @Test
    void tensorValueShapeMismatchThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> new TensorValue(new int[] { 2, 3 }, new double[] { 1, 2, 3 }));
    }

    @Test
    void tensorScalarConstructor() {
        TensorValue s = TensorValue.scalar(7.0);
        assertEquals(0, s.rank());
        assertEquals(1, s.size());
        assertEquals(7.0, s.data()[0]);
    }

    @Test
    void tensorVectorConstructor() {
        TensorValue v = TensorValue.vector(1, 2, 3);
        assertEquals(1, v.rank());
        assertEquals(3, v.dim(0));
    }

    @Test
    void tensorEqualsBySharedShapeAndData() {
        TensorValue a = new TensorValue(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        TensorValue b = new TensorValue(new int[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
        TensorValue c = new TensorValue(new int[] { 4 }, new double[] { 1, 2, 3, 4 });
        assertEquals(a, b);
        assertNotEquals(a, c, "Same data, different shape, must not be equal");
    }
}

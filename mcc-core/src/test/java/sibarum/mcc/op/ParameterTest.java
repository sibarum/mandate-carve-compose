package sibarum.mcc.op;

import org.junit.jupiter.api.Test;
import sibarum.mcc.primitive.Parameterized.NamedTensor;
import sibarum.mcc.value.MatrixValue;
import sibarum.mcc.value.NumberValue;
import sibarum.mcc.value.QuaternionValue;
import sibarum.mcc.value.TensorValue;
import sibarum.mcc.value.TernionValue;
import sibarum.mcc.value.Value;
import sibarum.mcc.value.ValueType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ParameterTest {

    private static final double TOL = 1e-12;

    @Test
    void vectorParameterProducesMatrixValue() {
        Parameter p = new Parameter(ValueType.MATRIX, new int[] { 4 }, 42L);
        Value out = p.apply(List.of());
        assertInstanceOf(MatrixValue.class, out);
        assertEquals(4, ((MatrixValue) out).data().length);
    }

    @Test
    void ternionParameterRequiresShape3() {
        assertThrows(IllegalArgumentException.class,
                () -> new Parameter(ValueType.TERNION, new int[] { 4 }, 0L));
        Parameter p = new Parameter(ValueType.TERNION, new int[] { 3 }, 0L);
        assertInstanceOf(TernionValue.class, p.apply(List.of()));
    }

    @Test
    void quaternionParameterRequiresShape4() {
        Parameter p = new Parameter(ValueType.QUATERNION, new int[] { 4 }, 0L);
        assertInstanceOf(QuaternionValue.class, p.apply(List.of()));
    }

    @Test
    void tensorParameterPreservesShape() {
        Parameter p = new Parameter(ValueType.TENSOR, new int[] { 2, 3 }, 0L);
        TensorValue t = (TensorValue) p.apply(List.of());
        assertArrayEquals(new int[] { 2, 3 }, t.shape());
        assertEquals(6, t.data().length);
    }

    @Test
    void numberParameterScalar() {
        Parameter p = new Parameter(ValueType.NUMBER, new int[0], 0L);
        Value out = p.apply(List.of());
        assertInstanceOf(NumberValue.class, out);
    }

    @Test
    void backwardAccumulatesGradient() {
        Parameter p = new Parameter(ValueType.MATRIX, new int[] { 3 }, 42L);
        double[] before = ((MatrixValue) p.apply(List.of())).data().clone();
        p.backward(new MatrixValue(new double[] { 1.0, 1.0, 1.0 }));
        p.step(0.1);
        double[] after = ((MatrixValue) p.apply(List.of())).data();
        for (int i = 0; i < 3; i++) {
            assertEquals(before[i] - 0.1, after[i], TOL,
                    "step should subtract lr * gradient at index " + i);
        }
    }

    @Test
    void rejectsInputs() {
        Parameter p = new Parameter(ValueType.MATRIX, new int[] { 2 }, 0L);
        assertThrows(IllegalArgumentException.class,
                () -> p.apply(List.of(new MatrixValue(new double[] { 1.0 }))));
    }

    @Test
    void backwardReturnsEmptyInputGradients() {
        Parameter p = new Parameter(ValueType.MATRIX, new int[] { 2 }, 0L);
        p.apply(List.of());
        List<Value> grads = p.backward(new MatrixValue(new double[] { 1.0, 2.0 }));
        assertTrue(grads.isEmpty(),
                "Parameter has no inputs so backward should return an empty list");
    }

    @Test
    void parametersAndLoadParametersRoundTrip() {
        Parameter src = new Parameter(ValueType.TENSOR, new int[] { 2, 2 }, 42L);
        double[] before = ((TensorValue) src.apply(List.of())).data().clone();

        // Snapshot parameters.
        List<NamedTensor> snapshot = src.parameters();
        assertEquals(1, snapshot.size());
        assertEquals("data", snapshot.getFirst().name());

        // Fresh parameter, different seed → different initial values.
        Parameter dst = new Parameter(ValueType.TENSOR, new int[] { 2, 2 }, 999L);
        double[] dstBefore = ((TensorValue) dst.apply(List.of())).data().clone();
        // Sanity: different seeds usually produce different inits.
        boolean differBefore = false;
        for (int i = 0; i < before.length; i++) {
            if (Math.abs(before[i] - dstBefore[i]) > 1e-9) { differBefore = true; break; }
        }
        assertTrue(differBefore);

        Map<String, NamedTensor> map = new HashMap<>();
        for (NamedTensor t : snapshot) map.put(t.name(), t);
        dst.loadParameters(map);

        double[] after = ((TensorValue) dst.apply(List.of())).data();
        assertArrayEquals(before, after, TOL, "loadParameters must restore exact bytes");
    }

    @Test
    void configRoundTrips() {
        Parameter p = new Parameter(ValueType.TENSOR, new int[] { 3, 4 }, 0L);
        Map<String, Object> cfg = p.config();
        assertEquals("TENSOR", cfg.get("outputType"));
        @SuppressWarnings("unchecked")
        List<Number> shape = (List<Number>) cfg.get("shape");
        assertEquals(2, shape.size());
        assertEquals(3, shape.get(0).intValue());
        assertEquals(4, shape.get(1).intValue());
    }

    @Test
    void gradientCheckAgainstFiniteDifferences() {
        // Loss L = ⟨gradOut, output⟩. Analytic ∂L/∂param = gradOut.
        Parameter p = new Parameter(ValueType.MATRIX, new int[] { 3 }, 13L);
        double[] gradOut = { 0.4, -0.7, 0.2 };
        MatrixValue out = (MatrixValue) p.apply(List.of());
        double[] outBefore = out.data().clone();
        p.backward(new MatrixValue(gradOut));

        // Finite-difference each parameter component.
        double eps = 1e-5;
        for (int i = 0; i < 3; i++) {
            double orig = p.rawData()[i];
            p.rawData()[i] = orig + eps;
            double lPlus = dot(((MatrixValue) p.apply(List.of())).data(), gradOut);
            p.rawData()[i] = orig - eps;
            double lMinus = dot(((MatrixValue) p.apply(List.of())).data(), gradOut);
            p.rawData()[i] = orig;
            double numeric = (lPlus - lMinus) / (2.0 * eps);
            assertEquals(gradOut[i], numeric, 1e-9, "param gradient at " + i);
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
    }
}

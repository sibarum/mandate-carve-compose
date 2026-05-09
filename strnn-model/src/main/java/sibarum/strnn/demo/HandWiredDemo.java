package sibarum.strnn.demo;

import sibarum.strnn.primitive.ComposeMatrices;
import sibarum.strnn.primitive.MlpPrimitive;
import sibarum.strnn.primitive.MlpRole;
import sibarum.strnn.primitive.NumberToMatrix;
import sibarum.strnn.primitive.MatrixToNumber;
import sibarum.strnn.primitive.OutputPrimitive;
import sibarum.strnn.primitive.ParseNumber;
import sibarum.strnn.primitive.Primitive;
import sibarum.strnn.primitive.SplitStringAt;
import sibarum.strnn.primitive.TokenAt;
import sibarum.strnn.value.NumberValue;
import sibarum.strnn.value.StringValue;
import sibarum.strnn.value.Value;

import java.util.List;

/**
 * Phase 0: hand-wired pipeline. Proves the value/primitive system can carry
 * the arithmetic computation end-to-end without any graph machinery.
 *
 * Pipeline for "3+4*5":
 *   split('+')           : "3+4*5"  -> ["3", "4*5"]
 *   tokenAt(0)           : ["3", "4*5"] -> "3"
 *   tokenAt(1)           : ["3", "4*5"] -> "4*5"
 *   split('*')           : "4*5"    -> ["4", "5"]
 *   tokenAt(0/1)         : ["4", "5"] -> "4" / "5"
 *   parseNumber          : "3"/"4"/"5" -> 3.0/4.0/5.0
 *   numberToMatrix       : 4.0 -> [0.4]; 5.0 -> [0.5]
 *   composeMatrices      : ([0.4], [0.5]) -> [0.4, 0.5]
 *   mlp(MUL)             : [0.4, 0.5] -> [2.0]   (i.e. 20/10)
 *   matrixToNumber       : [2.0] -> 20.0
 *   numberToMatrix x2    : 3 and 20 -> [0.3] and [2.0]
 *   composeMatrices      : -> [0.3, 2.0]
 *   mlp(ADD)             : [0.3, 2.0] -> [2.3]
 *   matrixToNumber       : -> 23.0
 *   output               : 23.0
 */
public final class HandWiredDemo {
    static void main(String[] args) {
        Primitive splitPlus = new SplitStringAt('+');
        Primitive splitStar = new SplitStringAt('*');
        Primitive tokenAt0 = new TokenAt(0);
        Primitive tokenAt1 = new TokenAt(1);
        Primitive parse = new ParseNumber();
        Primitive numToMat = new NumberToMatrix();
        Primitive compose = new ComposeMatrices();
        Primitive mlpAdd = new MlpPrimitive(MlpRole.ADD);
        Primitive mlpMul = new MlpPrimitive(MlpRole.MUL);
        Primitive matToNum = new MatrixToNumber();
        Primitive output = new OutputPrimitive();

        Value input = new StringValue("3+4*5");

        Value plusSplit = splitPlus.apply(List.of(input));        // ["3","4*5"]
        Value left = tokenAt0.apply(List.of(plusSplit));          // "3"
        Value right = tokenAt1.apply(List.of(plusSplit));         // "4*5"

        Value starSplit = splitStar.apply(List.of(right));        // ["4","5"]
        Value mulLeft = tokenAt0.apply(List.of(starSplit));       // "4"
        Value mulRight = tokenAt1.apply(List.of(starSplit));      // "5"

        Value n4 = parse.apply(List.of(mulLeft));                 // 4.0
        Value n5 = parse.apply(List.of(mulRight));                // 5.0
        Value m4 = numToMat.apply(List.of(n4));                   // [0.4]
        Value m5 = numToMat.apply(List.of(n5));                   // [0.5]
        Value m45 = compose.apply(List.of(m4, m5));               // [0.4,0.5]
        Value mProd = mlpMul.apply(List.of(m45));                 // [2.0]
        Value nProd = matToNum.apply(List.of(mProd)); // 20.0

        Value n3 = parse.apply(List.of(left));                    // 3.0
        Value m3 = numToMat.apply(List.of(n3));                   // [0.3]
        Value mProdEnc = numToMat.apply(List.of(nProd));          // [2.0]
        Value mAdd = compose.apply(List.of(m3, mProdEnc));        // [0.3,2.0]
        Value mSum = mlpAdd.apply(List.of(mAdd));                 // [2.3]
        Value nSum = matToNum.apply(List.of(mSum));               // 23.0

        Value result = output.apply(List.of(nSum));

        double answer = ((NumberValue) result).n();
        System.out.println("3+4*5 = " + answer);

        if (Math.abs(answer - 23.0) > 1e-6) {
            throw new AssertionError("Expected 23.0, got " + answer);
        }
        System.out.println("Phase 0 hand-wired pipeline OK.");
    }
}

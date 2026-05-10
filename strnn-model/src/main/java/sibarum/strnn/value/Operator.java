package sibarum.strnn.value;

public enum Operator {
    ADD("+"),
    SUB("-"),
    MUL("*"),
    DIV("/");

    private final String symbol;

    Operator(String symbol) {
        this.symbol = symbol;
    }

    public String symbol() {
        return symbol;
    }

    public double apply(double left, double right) {
        return switch (this) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
        };
    }

    public static Operator fromChar(char c) {
        return switch (c) {
            case '+' -> ADD;
            case '-' -> SUB;
            case '*' -> MUL;
            case '/' -> DIV;
            default -> throw new IllegalArgumentException("not an operator: " + c);
        };
    }
}

package wikiedits;

public class Tuple6<A, B, C, D, E, F> {
    private final A element1;
    private final B element2;
    private final C element3;
    private final D element4;
    private final E element5;
    private final F element6;

    public Tuple6(A element1, B element2, C element3, D element4, E element5, F element6) {
        this.element1 = element1;
        this.element2 = element2;
        this.element3 = element3;
        this.element4 = element4;
        this.element5 = element5;
        this.element6 = element6;
    }

    public A getElement1() {
        return element1;
    }

    public B getElement2() {
        return element2;
    }

    public C getElement3() {
        return element3;
    }

    public D getElement4() {
        return element4;
    }

    public E getElement5() {
        return element5;
    }

    public F getElement6() {
        return element6;
    }

    @Override
    public String toString() {
        return "(" + element1 + ", " + element2 + ", " + element3 + ", " +
               element4 + ", " + element5 + ", " + element6 + ")";
    }
}

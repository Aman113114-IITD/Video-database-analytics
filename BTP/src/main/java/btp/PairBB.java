package btp;

import java.util.Objects;

public class PairBB {
	private Event<Integer, Integer, Integer, String, Float, Float, Float, Float> object1;
	private Event<Integer, Integer, Integer, String, Float, Float, Float, Float> object2;
	public PairBB(Event<Integer, Integer, Integer, String, Float, Float, Float, Float> object1, Event<Integer, Integer, Integer, String, Float, Float, Float, Float> object2) {
    	this.object1 = object1;
    	this.object2 = object2;
	}
	public Event<Integer, Integer, Integer, String, Float, Float, Float, Float> getobject1() {
        return object1;
    }
	public Event<Integer, Integer, Integer, String, Float, Float, Float, Float> getobject2() {
        return object2;
    }
	@Override
    public String toString() {
        return "PairBB{" +
                "object1=" + object1 +
                ", object2=" + object2 +
                '}';
    }
}

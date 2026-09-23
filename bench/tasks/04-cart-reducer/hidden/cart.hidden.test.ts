import { it, expect } from "vitest";
import { cartReducer, cartTotalCents, emptyCart, type CartState } from "../src/cart";

const tee = { id: "tee", name: "T-shirt", priceCents: 1500 };
const mug = { id: "mug", name: "Mug", priceCents: 899 };
const deepFreeze = (s: CartState) => Object.freeze({ items: Object.freeze(s.items.map((i) => Object.freeze({ ...i }))) }) as CartState;
const base = deepFreeze({ items: [{ ...tee, qty: 1 }, { ...mug, qty: 2 }] });

it("merges with custom qty", () => expect(cartReducer(base, { type: "add", item: tee, qty: 3 }).items[0].qty).toBe(4));
it("adds new line", () => expect(cartReducer(base, { type: "add", item: { id: "cap", name: "Cap", priceCents: 1000 } }).items).toHaveLength(3));
it("setQty updates", () => expect(cartReducer(base, { type: "setQty", id: "mug", qty: 5 }).items[1].qty).toBe(5));
it("setQty 0 removes", () => expect(cartReducer(base, { type: "setQty", id: "mug", qty: 0 }).items.map((i) => i.id)).toEqual(["tee"]));
it("setQty negative removes", () => expect(cartReducer(base, { type: "setQty", id: "tee", qty: -1 }).items.map((i) => i.id)).toEqual(["mug"]));
it("remove", () => expect(cartReducer(base, { type: "remove", id: "tee" }).items.map((i) => i.id)).toEqual(["mug"]));
it("clear", () => expect(cartReducer(base, { type: "clear" } as any)).toEqual({ items: [] }));
it("total uses qty", () => expect(cartTotalCents(base)).toBe(1500 + 2 * 899));
it("does not mutate (frozen state)", () => {
  expect(() => cartReducer(base, { type: "add", item: tee })).not.toThrow();
  expect(base.items[0].qty).toBe(1);
});
it("returns new object", () => expect(cartReducer(base, { type: "add", item: tee })).not.toBe(base));

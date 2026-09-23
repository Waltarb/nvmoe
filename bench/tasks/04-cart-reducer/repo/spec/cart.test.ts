import { it, expect } from "vitest";
import { cartReducer, cartTotalCents, emptyCart } from "../src/cart";

const tee = { id: "tee", name: "T-shirt", priceCents: 1500 };

it("adding the same item twice increases qty", () => {
  const s = cartReducer(cartReducer(emptyCart, { type: "add", item: tee }), { type: "add", item: tee });
  expect(s.items).toHaveLength(1);
  expect(s.items[0].qty).toBe(2);
});

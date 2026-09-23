# Cart reducer: bugs + a new action

The shopping cart reducer in `src/cart.ts` has problems:

1. Adding an item whose `id` is already in the cart creates a duplicate line. It should increase that line's `qty` instead (by `action.qty`, default 1).
2. `setQty` with a quantity of 0 or less should remove the line.
3. `cartTotalCents` ignores quantity.
4. Add a new action `{ type: "clear" }` that empties the cart.

The reducer is used with React `useReducer`, so it must never mutate the incoming state.

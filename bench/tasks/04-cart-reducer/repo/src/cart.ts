export interface CartItem {
  id: string;
  name: string;
  priceCents: number;
  qty: number;
}

export interface CartState {
  items: CartItem[];
}

export type CartAction =
  | { type: "add"; item: Omit<CartItem, "qty">; qty?: number }
  | { type: "remove"; id: string }
  | { type: "setQty"; id: string; qty: number };

export const emptyCart: CartState = { items: [] };

export function cartReducer(state: CartState, action: CartAction): CartState {
  switch (action.type) {
    case "add":
      return { items: [...state.items, { ...action.item, qty: action.qty ?? 1 }] };
    case "remove":
      return { items: state.items.filter((i) => i.id !== action.id) };
    case "setQty":
      return { items: state.items.map((i) => (i.id === action.id ? { ...i, qty: action.qty } : i)) };
  }
}

export function cartTotalCents(state: CartState): number {
  return state.items.reduce((sum, i) => sum + i.priceCents, 0);
}

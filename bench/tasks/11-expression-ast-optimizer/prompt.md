# Feature: Algebraic and Boolean Expression AST Optimizer

Implement `optimizeAST(node: ASTNode): ASTNode` in `src/optimizer.ts` that simplifies expression ASTs to a fixpoint.

## Node Types
```ts
export type LiteralNode = { type: "literal"; value: number | boolean };
export type IdentifierNode = { type: "identifier"; name: string };
export type UnaryNode = { type: "unary"; op: "!" | "-"; argument: ASTNode };
export type BinaryNode = {
  type: "binary";
  op: "+" | "-" | "*" | "/" | "==" | "!=" | "&&" | "||";
  left: ASTNode;
  right: ASTNode;
};
export type ASTNode = LiteralNode | IdentifierNode | UnaryNode | BinaryNode;
```

## Optimization Rules
The optimizer must simplify subtrees recursively until no more rules match (fixpoint).

1. **Constant Folding**:
   - `!literal(bool)` -> `literal(!bool)`
   - `-literal(num)` -> `literal(-num)`
   - `literal(num1) [+, -, *] literal(num2)` -> folded `literal(result)` (do not fold division by zero)
   - `literal(n1) / literal(n2)` when `n2 !== 0` -> `literal(n1 / n2)`
   - `literal(a) == literal(b)` -> `literal(a === b)`
   - `literal(a) != literal(b)` -> `literal(a !== b)`
   - `literal(b1) && literal(b2)` -> `literal(b1 && b2)`
   - `literal(b1) || literal(b2)` -> `literal(b1 || b2)`

2. **Identities & Annihilators**:
   - `x + 0` or `0 + x` -> `x`
   - `x - 0` -> `x`
   - `x * 1` or `1 * x` -> `x`
   - `x * 0` or `0 * x` -> `literal(0)`
   - `x && true` or `true && x` -> `x`
   - `x && false` or `false && x` -> `literal(false)`
   - `x || false` or `false || x` -> `x`
   - `x || true` or `true || x` -> `literal(true)`

3. **Double Negation**:
   - `!(!x)` -> `x`
   - `-(-x)` -> `x`

4. **De Morgan's Laws**:
   - `!(A && B)` -> `(!A) || (!B)`
   - `!(A || B)` -> `(!A) && (!B)`
   *(Note: when applying De Morgan's, simplify any resulting double negations, e.g. `!(x && !y)` -> `!x || y`).*

5. **Structural Equality**:
   - `isEqual(x, x)` for `x == x` -> `literal(true)`
   - `isEqual(x, x)` for `x != x` -> `literal(false)`

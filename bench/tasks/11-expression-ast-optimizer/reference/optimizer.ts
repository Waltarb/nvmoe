import type { ASTNode } from "../src/types";

function areNodesEqual(a: ASTNode, b: ASTNode): boolean {
  if (a.type !== b.type) return false;
  if (a.type === "literal" && b.type === "literal") {
    return a.value === b.value;
  }
  if (a.type === "identifier" && b.type === "identifier") {
    return a.name === b.name;
  }
  if (a.type === "unary" && b.type === "unary") {
    return a.op === b.op && areNodesEqual(a.argument, b.argument);
  }
  if (a.type === "binary" && b.type === "binary") {
    return a.op === b.op && areNodesEqual(a.left, b.left) && areNodesEqual(a.right, b.right);
  }
  return false;
}

function optimizeStep(node: ASTNode): ASTNode {
  if (node.type === "literal" || node.type === "identifier") {
    return node;
  }

  if (node.type === "unary") {
    const arg = optimizeStep(node.argument);

    // Double negation: !(!x) -> x, -(-x) -> x
    if (arg.type === "unary" && arg.op === node.op) {
      return arg.argument;
    }

    // Constant folding
    if (arg.type === "literal") {
      if (node.op === "!" && typeof arg.value === "boolean") {
        return { type: "literal", value: !arg.value };
      }
      if (node.op === "-" && typeof arg.value === "number") {
        return { type: "literal", value: -arg.value };
      }
    }

    // De Morgan's laws: !(A && B) -> !A || !B, !(A || B) -> !A && !B
    if (node.op === "!" && arg.type === "binary" && (arg.op === "&&" || arg.op === "||")) {
      const newOp = arg.op === "&&" ? "||" : "&&";
      const leftNeg: ASTNode = optimizeStep({ type: "unary", op: "!", argument: arg.left });
      const rightNeg: ASTNode = optimizeStep({ type: "unary", op: "!", argument: arg.right });
      return { type: "binary", op: newOp, left: leftNeg, right: rightNeg };
    }

    return { type: "unary", op: node.op, argument: arg };
  }

  if (node.type === "binary") {
    const left = optimizeStep(node.left);
    const right = optimizeStep(node.right);
    const op = node.op;

    // Structural equality: x == x -> true, x != x -> false
    if (op === "==" && areNodesEqual(left, right)) {
      return { type: "literal", value: true };
    }
    if (op === "!=" && areNodesEqual(left, right)) {
      return { type: "literal", value: false };
    }

    // Constant folding
    if (left.type === "literal" && right.type === "literal") {
      const lv = left.value;
      const rv = right.value;

      if (typeof lv === "number" && typeof rv === "number") {
        if (op === "+") return { type: "literal", value: lv + rv };
        if (op === "-") return { type: "literal", value: lv - rv };
        if (op === "*") return { type: "literal", value: lv * rv };
        if (op === "/") {
          if (rv !== 0) return { type: "literal", value: lv / rv };
        }
        if (op === "==") return { type: "literal", value: lv === rv };
        if (op === "!=") return { type: "literal", value: lv !== rv };
      }

      if (typeof lv === "boolean" && typeof rv === "boolean") {
        if (op === "&&") return { type: "literal", value: lv && rv };
        if (op === "||") return { type: "literal", value: lv || rv };
        if (op === "==") return { type: "literal", value: lv === rv };
        if (op === "!=") return { type: "literal", value: lv !== rv };
      }
    }

    // Identities & Annihilators:
    // + 0
    if (op === "+") {
      if (right.type === "literal" && right.value === 0) return left;
      if (left.type === "literal" && left.value === 0) return right;
    }

    // - 0
    if (op === "-") {
      if (right.type === "literal" && right.value === 0) return left;
    }

    // * 1, * 0
    if (op === "*") {
      if (right.type === "literal" && right.value === 1) return left;
      if (left.type === "literal" && left.value === 1) return right;
      if (right.type === "literal" && right.value === 0) return { type: "literal", value: 0 };
      if (left.type === "literal" && left.value === 0) return { type: "literal", value: 0 };
    }

    // && true / false
    if (op === "&&") {
      if (right.type === "literal" && right.value === true) return left;
      if (left.type === "literal" && left.value === true) return right;
      if (right.type === "literal" && right.value === false) return { type: "literal", value: false };
      if (left.type === "literal" && left.value === false) return { type: "literal", value: false };
    }

    // || true / false
    if (op === "||") {
      if (right.type === "literal" && right.value === false) return left;
      if (left.type === "literal" && left.value === false) return right;
      if (right.type === "literal" && right.value === true) return { type: "literal", value: true };
      if (left.type === "literal" && left.value === true) return { type: "literal", value: true };
    }

    return { type: "binary", op, left, right };
  }

  return node;
}

export function optimizeAST(node: ASTNode): ASTNode {
  let current = node;
  for (let i = 0; i < 20; i++) {
    const next = optimizeStep(current);
    if (areNodesEqual(current, next)) {
      return next;
    }
    current = next;
  }
  return current;
}

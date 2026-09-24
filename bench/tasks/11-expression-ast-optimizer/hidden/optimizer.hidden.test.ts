import { describe, it, expect } from "vitest";
import { optimizeAST } from "../src/optimizer";
import type { ASTNode } from "../src/types";

describe("AST Optimizer (hidden)", () => {
  it("applies De Morgan's law: !(A && B) -> (!A) || (!B)", () => {
    const ast: ASTNode = {
      type: "unary",
      op: "!",
      argument: {
        type: "binary",
        op: "&&",
        left: { type: "identifier", name: "A" },
        right: { type: "identifier", name: "B" },
      },
    };
    expect(optimizeAST(ast)).toEqual({
      type: "binary",
      op: "||",
      left: { type: "unary", op: "!", argument: { type: "identifier", name: "A" } },
      right: { type: "unary", op: "!", argument: { type: "identifier", name: "B" } },
    });
  });

  it("applies De Morgan's with nested negation: !(x && !y) -> (!x) || y", () => {
    const ast: ASTNode = {
      type: "unary",
      op: "!",
      argument: {
        type: "binary",
        op: "&&",
        left: { type: "identifier", name: "x" },
        right: { type: "unary", op: "!", argument: { type: "identifier", name: "y" } },
      },
    };
    expect(optimizeAST(ast)).toEqual({
      type: "binary",
      op: "||",
      left: { type: "unary", op: "!", argument: { type: "identifier", name: "x" } },
      right: { type: "identifier", name: "y" },
    });
  });

  it("simplifies annihilators (x && false -> false, x || true -> true, x * 0 -> 0)", () => {
    const ast1: ASTNode = {
      type: "binary",
      op: "&&",
      left: { type: "identifier", name: "p" },
      right: { type: "literal", value: false },
    };
    expect(optimizeAST(ast1)).toEqual({ type: "literal", value: false });

    const ast2: ASTNode = {
      type: "binary",
      op: "||",
      left: { type: "literal", value: true },
      right: { type: "identifier", name: "q" },
    };
    expect(optimizeAST(ast2)).toEqual({ type: "literal", value: true });

    const ast3: ASTNode = {
      type: "binary",
      op: "*",
      left: { type: "identifier", name: "num" },
      right: { type: "literal", value: 0 },
    };
    expect(optimizeAST(ast3)).toEqual({ type: "literal", value: 0 });
  });

  it("reduces structural equality for compound nodes", () => {
    const compound: ASTNode = {
      type: "binary",
      op: "+",
      left: { type: "identifier", name: "a" },
      right: { type: "identifier", name: "b" },
    };
    const eqNode: ASTNode = {
      type: "binary",
      op: "==",
      left: compound,
      right: compound,
    };
    expect(optimizeAST(eqNode)).toEqual({ type: "literal", value: true });

    const neqNode: ASTNode = {
      type: "binary",
      op: "!=",
      left: compound,
      right: compound,
    };
    expect(optimizeAST(neqNode)).toEqual({ type: "literal", value: false });
  });

  it("does not fold division by zero", () => {
    const divZero: ASTNode = {
      type: "binary",
      op: "/",
      left: { type: "literal", value: 10 },
      right: { type: "literal", value: 0 },
    };
    expect(optimizeAST(divZero)).toEqual(divZero);
  });

  it("reaches fixpoint across deeply nested reductions", () => {
    // !(!(!(!(a + 0)))) * 1 -> a
    const ast: ASTNode = {
      type: "binary",
      op: "*",
      left: {
        type: "unary",
        op: "!",
        argument: {
          type: "unary",
          op: "!",
          argument: {
            type: "unary",
            op: "!",
            argument: {
              type: "unary",
              op: "!",
              argument: {
                type: "binary",
                op: "+",
                left: { type: "identifier", name: "a" },
                right: { type: "literal", value: 0 },
              },
            },
          },
        },
      },
      right: { type: "literal", value: 1 },
    };

    expect(optimizeAST(ast)).toEqual({ type: "identifier", name: "a" });
  });
});

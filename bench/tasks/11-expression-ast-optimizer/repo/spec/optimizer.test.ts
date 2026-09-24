import { describe, it, expect } from "vitest";
import { optimizeAST } from "../src/optimizer";
import type { ASTNode } from "../src/types";

describe("AST Optimizer (visible)", () => {
  it("folds constant numeric arithmetic", () => {
    // 2 + 3
    const ast: ASTNode = {
      type: "binary",
      op: "+",
      left: { type: "literal", value: 2 },
      right: { type: "literal", value: 3 },
    };
    expect(optimizeAST(ast)).toEqual({ type: "literal", value: 5 });
  });

  it("simplifies identity element x + 0 -> x", () => {
    const ast: ASTNode = {
      type: "binary",
      op: "+",
      left: { type: "identifier", name: "x" },
      right: { type: "literal", value: 0 },
    };
    expect(optimizeAST(ast)).toEqual({ type: "identifier", name: "x" });
  });

  it("eliminates double negation !(!x) -> x", () => {
    const ast: ASTNode = {
      type: "unary",
      op: "!",
      argument: {
        type: "unary",
        op: "!",
        argument: { type: "identifier", name: "isValid" },
      },
    };
    expect(optimizeAST(ast)).toEqual({ type: "identifier", name: "isValid" });
  });
});

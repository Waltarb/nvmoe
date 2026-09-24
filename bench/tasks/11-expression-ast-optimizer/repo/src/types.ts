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

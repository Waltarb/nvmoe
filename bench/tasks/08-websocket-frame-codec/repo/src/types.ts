export enum Opcode {
  CONTINUATION = 0x0,
  TEXT = 0x1,
  BINARY = 0x2,
  CLOSE = 0x8,
  PING = 0x9,
  PONG = 0xA,
}

export interface WebSocketFrame {
  fin: boolean;
  rsv1?: boolean;
  rsv2?: boolean;
  rsv3?: boolean;
  opcode: Opcode | number;
  payload: Uint8Array;
}

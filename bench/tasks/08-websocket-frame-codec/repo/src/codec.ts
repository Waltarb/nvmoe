import { Opcode, type WebSocketFrame } from "./types";

export function encodeFrame(frame: WebSocketFrame, maskKey?: Uint8Array): Uint8Array {
  throw new Error("Not implemented");
}

export class FrameDecoder {
  get bufferedBytes(): number {
    return 0;
  }

  push(chunk: Uint8Array): WebSocketFrame[] {
    return [];
  }
}

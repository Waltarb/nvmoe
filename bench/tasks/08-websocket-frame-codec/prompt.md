# Feature: RFC 6455 WebSocket Binary Frame Codec

Implement `encodeFrame` and `FrameDecoder` in `src/codec.ts` to encode and stream-decode binary WebSocket frames according to RFC 6455.

## Data Structures
```ts
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
```

## Functions & Classes

### 1. `encodeFrame(frame: WebSocketFrame, maskKey?: Uint8Array): Uint8Array`
- Builds a valid RFC 6455 binary frame:
  - Byte 0: `(fin ? 0x80 : 0) | ((rsv1 ? 1 : 0) << 6) | ((rsv2 ? 1 : 0) << 5) | ((rsv3 ? 1 : 0) << 4) | (opcode & 0x0F)`
  - Length encoding in Byte 1 + extension:
    - If `payload.length <= 125`: Byte 1 contains length.
    - If `126 <= payload.length <= 65535`: Byte 1 length field is `126`, followed by 2 bytes (16-bit uint, big-endian).
    - If `payload.length >= 65536`: Byte 1 length field is `127`, followed by 8 bytes (64-bit uint, big-endian).
  - Masking:
    - If `maskKey` is provided (must be 4 bytes): set mask bit (0x80) on Byte 1, append the 4-byte mask key, and XOR-mask each byte of the payload (`payload[i] ^ maskKey[i % 4]`).
    - If `maskKey` is not provided: mask bit is 0, payload is unmasked.

### 2. `class FrameDecoder`
- Streaming frame parser that handles chunked streams split across arbitrary byte boundaries:
  - `push(chunk: Uint8Array): WebSocketFrame[]`:
    - Buffers incoming bytes and parses as many complete frames as available.
    - Unmasks payload if mask bit was set.
    - Preserves unconsumed partial frame bytes for subsequent `push()` calls.
  - `get bufferedBytes(): number`: returns number of unprocessed bytes currently buffered.
  - Validation rules:
    - If any RSV bit (RSV1, RSV2, RSV3) is 1, throws `new Error("RSV bits must be 0")`.
    - Control frames (opcodes >= 0x8: CLOSE, PING, PONG) MUST have `fin === true` and `payload.length <= 125`. If a control frame has `fin === false` or payload length > 125, throws `new Error("Invalid control frame")`.

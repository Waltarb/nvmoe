import { describe, it, expect } from "vitest";
import { encodeFrame, FrameDecoder } from "../src/codec";
import { Opcode } from "../src/types";

describe("WebSocket Frame Codec (visible)", () => {
  it("encodes and decodes a simple unmasked text frame", () => {
    const textBytes = new TextEncoder().encode("Hello, World!");
    const frame = {
      fin: true,
      opcode: Opcode.TEXT,
      payload: textBytes,
    };

    const encoded = encodeFrame(frame);
    const decoder = new FrameDecoder();
    const decoded = decoder.push(encoded);

    expect(decoded.length).toBe(1);
    expect(decoded[0].fin).toBe(true);
    expect(decoded[0].opcode).toBe(Opcode.TEXT);
    expect(new TextDecoder().decode(decoded[0].payload)).toBe("Hello, World!");
  });

  it("encodes and decodes a masked frame", () => {
    const data = new Uint8Array([1, 2, 3, 4, 5]);
    const maskKey = new Uint8Array([0x12, 0x34, 0x56, 0x78]);
    const frame = {
      fin: true,
      opcode: Opcode.BINARY,
      payload: data,
    };

    const encoded = encodeFrame(frame, maskKey);
    const decoder = new FrameDecoder();
    const decoded = decoder.push(encoded);

    expect(decoded.length).toBe(1);
    expect(decoded[0].opcode).toBe(Opcode.BINARY);
    expect(Array.from(decoded[0].payload)).toEqual([1, 2, 3, 4, 5]);
  });
});

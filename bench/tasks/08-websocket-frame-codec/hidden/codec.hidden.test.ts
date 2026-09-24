import { describe, it, expect } from "vitest";
import { encodeFrame, FrameDecoder } from "../src/codec";
import { Opcode } from "../src/types";

describe("WebSocket Frame Codec (hidden)", () => {
  it("encodes and decodes 16-bit length payloads (> 125 bytes)", () => {
    const payload = new Uint8Array(1000);
    for (let i = 0; i < 1000; i++) payload[i] = i % 256;

    const frame = { fin: true, opcode: Opcode.BINARY, payload };
    const encoded = encodeFrame(frame);
    expect(encoded[1] & 0x7F).toBe(126);

    const decoder = new FrameDecoder();
    const [res] = decoder.push(encoded);
    expect(res.payload.length).toBe(1000);
    expect(res.payload[999]).toBe(999 % 256);
  });

  it("encodes and decodes 64-bit length payloads (> 65535 bytes)", () => {
    const payload = new Uint8Array(70000);
    payload[0] = 0xAA;
    payload[69999] = 0xBB;

    const frame = { fin: true, opcode: Opcode.BINARY, payload };
    const encoded = encodeFrame(frame);
    expect(encoded[1] & 0x7F).toBe(127);

    const decoder = new FrameDecoder();
    const [res] = decoder.push(encoded);
    expect(res.payload.length).toBe(70000);
    expect(res.payload[0]).toBe(0xAA);
    expect(res.payload[69999]).toBe(0xBB);
  });

  it("handles byte-by-byte streaming fragmentation", () => {
    const payload = new Uint8Array([10, 20, 30, 40, 50]);
    const maskKey = new Uint8Array([1, 2, 3, 4]);
    const encoded = encodeFrame({ fin: true, opcode: Opcode.BINARY, payload }, maskKey);

    const decoder = new FrameDecoder();
    const frames = [];

    // Push 1 byte at a time
    for (let i = 0; i < encoded.length; i++) {
      const slice = encoded.subarray(i, i + 1);
      const res = decoder.push(slice);
      frames.push(...res);
    }

    expect(frames.length).toBe(1);
    expect(Array.from(frames[0].payload)).toEqual([10, 20, 30, 40, 50]);
    expect(decoder.bufferedBytes).toBe(0);
  });

  it("rejects non-zero RSV bits", () => {
    const badFrame = new Uint8Array([
      0b11000001, // FIN=1, RSV1=1, Opcode=TEXT
      0,          // Mask=0, Len=0
    ]);

    const decoder = new FrameDecoder();
    expect(() => decoder.push(badFrame)).toThrow(/RSV bits must be 0/);
  });

  it("rejects fragmented control frames (FIN=0 for Ping)", () => {
    const badPing = new Uint8Array([
      0b00001001, // FIN=0, Opcode=PING (0x9)
      0,
    ]);

    const decoder = new FrameDecoder();
    expect(() => decoder.push(badPing)).toThrow(/Invalid control frame/);
  });

  it("rejects control frames exceeding 125 bytes", () => {
    const badControl = new Uint8Array([
      0b10001001, // FIN=1, Opcode=PING
      126,        // Len=126 (> 125)
      0, 130,
    ]);

    const decoder = new FrameDecoder();
    expect(() => decoder.push(badControl)).toThrow(/Invalid control frame/);
  });

  it("supports interleaved control frames between message fragments", () => {
    // Fragment 1: Text, FIN=0
    const f1 = encodeFrame({ fin: false, opcode: Opcode.TEXT, payload: new TextEncoder().encode("Part1") });
    // Interleaved Ping: FIN=1
    const ping = encodeFrame({ fin: true, opcode: Opcode.PING, payload: new Uint8Array([1]) });
    // Fragment 2: Continuation, FIN=1
    const f2 = encodeFrame({ fin: true, opcode: Opcode.CONTINUATION, payload: new TextEncoder().encode("Part2") });

    const decoder = new FrameDecoder();
    const stream = new Uint8Array(f1.length + ping.length + f2.length);
    stream.set(f1, 0);
    stream.set(ping, f1.length);
    stream.set(f2, f1.length + ping.length);

    const received = decoder.push(stream);
    expect(received.length).toBe(3);
    expect(received[0].opcode).toBe(Opcode.TEXT);
    expect(received[0].fin).toBe(false);
    expect(received[1].opcode).toBe(Opcode.PING);
    expect(received[1].fin).toBe(true);
    expect(received[2].opcode).toBe(Opcode.CONTINUATION);
    expect(received[2].fin).toBe(true);
  });
});

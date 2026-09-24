import { Opcode, type WebSocketFrame } from "../src/types";

export function encodeFrame(frame: WebSocketFrame, maskKey?: Uint8Array): Uint8Array {
  const isMasked = maskKey !== undefined;
  if (isMasked && maskKey.length !== 4) {
    throw new Error("Mask key must be 4 bytes");
  }

  const payloadLen = frame.payload.length;
  let headerLen = 2;
  let lenIndicator = 0;

  if (payloadLen <= 125) {
    lenIndicator = payloadLen;
  } else if (payloadLen <= 65535) {
    lenIndicator = 126;
    headerLen += 2;
  } else {
    lenIndicator = 127;
    headerLen += 8;
  }

  if (isMasked) {
    headerLen += 4;
  }

  const out = new Uint8Array(headerLen + payloadLen);
  let byte0 = (frame.opcode & 0x0f);
  if (frame.fin) byte0 |= 0x80;
  if (frame.rsv1) byte0 |= 0x40;
  if (frame.rsv2) byte0 |= 0x20;
  if (frame.rsv3) byte0 |= 0x10;
  out[0] = byte0;

  let byte1 = lenIndicator;
  if (isMasked) byte1 |= 0x80;
  out[1] = byte1;

  let offset = 2;
  if (lenIndicator === 126) {
    const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
    view.setUint16(offset, payloadLen, false);
    offset += 2;
  } else if (lenIndicator === 127) {
    const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
    view.setBigUint64(offset, BigInt(payloadLen), false);
    offset += 8;
  }

  if (isMasked) {
    out.set(maskKey, offset);
    offset += 4;

    for (let i = 0; i < payloadLen; i++) {
      out[offset + i] = frame.payload[i] ^ maskKey[i % 4];
    }
  } else {
    out.set(frame.payload, offset);
  }

  return out;
}

export class FrameDecoder {
  private buffer: Uint8Array = new Uint8Array(0);

  get bufferedBytes(): number {
    return this.buffer.length;
  }

  push(chunk: Uint8Array): WebSocketFrame[] {
    if (this.buffer.length === 0) {
      this.buffer = chunk;
    } else {
      const merged = new Uint8Array(this.buffer.length + chunk.length);
      merged.set(this.buffer, 0);
      merged.set(chunk, this.buffer.length);
      this.buffer = merged;
    }

    const frames: WebSocketFrame[] = [];
    let offset = 0;

    while (this.buffer.length - offset >= 2) {
      const byte0 = this.buffer[offset];
      const byte1 = this.buffer[offset + 1];

      const fin = (byte0 & 0x80) !== 0;
      const rsv1 = (byte0 & 0x40) !== 0;
      const rsv2 = (byte0 & 0x20) !== 0;
      const rsv3 = (byte0 & 0x10) !== 0;
      const opcode = byte0 & 0x0f;

      if (rsv1 || rsv2 || rsv3) {
        throw new Error("RSV bits must be 0");
      }

      const masked = (byte1 & 0x80) !== 0;
      const rawLen = byte1 & 0x7f;

      let curOffset = offset + 2;
      let payloadLen = 0;

      if (rawLen <= 125) {
        payloadLen = rawLen;
      } else if (rawLen === 126) {
        if (this.buffer.length - curOffset < 2) break;
        const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + curOffset, 2);
        payloadLen = view.getUint16(0, false);
        curOffset += 2;
      } else {
        if (this.buffer.length - curOffset < 8) break;
        const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + curOffset, 8);
        const bigLen = view.getBigUint64(0, false);
        payloadLen = Number(bigLen);
        curOffset += 8;
      }

      const isControl = opcode >= 0x8;
      if (isControl) {
        if (!fin || payloadLen > 125) {
          throw new Error("Invalid control frame");
        }
      }

      let maskKey: Uint8Array | null = null;
      if (masked) {
        if (this.buffer.length - curOffset < 4) break;
        maskKey = this.buffer.subarray(curOffset, curOffset + 4);
        curOffset += 4;
      }

      if (this.buffer.length - curOffset < payloadLen) break;

      let payload: Uint8Array;
      if (masked && maskKey) {
        payload = new Uint8Array(payloadLen);
        for (let i = 0; i < payloadLen; i++) {
          payload[i] = this.buffer[curOffset + i] ^ maskKey[i % 4];
        }
      } else {
        payload = this.buffer.slice(curOffset, curOffset + payloadLen);
      }

      curOffset += payloadLen;
      offset = curOffset;

      frames.push({
        fin,
        rsv1,
        rsv2,
        rsv3,
        opcode,
        payload,
      });
    }

    if (offset > 0) {
      this.buffer = this.buffer.slice(offset);
    }

    return frames;
  }
}

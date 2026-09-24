import { describe, it, expect, vi } from "vitest";
import { TypedEmitter } from "../src/emitter";

interface AppEvents {
  "order:created": { id: string; amount: number };
  "order:cancelled": { id: string };
  "payment:processed": { id: string };
}

describe("TypedEmitter (hidden)", () => {
  it("stops propagation when requested by handler", async () => {
    const emitter = new TypedEmitter<AppEvents>();
    const secondFn = vi.fn();

    emitter.on("order:created", (_payload, ctx) => {
      ctx.stopPropagation();
    });
    emitter.on("order:created", secondFn);

    const res = await emitter.emit("order:created", { id: "101", amount: 50 });
    expect(res.handledCount).toBe(1);
    expect(res.stopped).toBe(true);
    expect(secondFn).not.toHaveBeenCalled();
  });

  it("captures handler errors and continues remaining handlers", async () => {
    const emitter = new TypedEmitter<AppEvents>();
    const err = new Error("handler failure");
    const secondFn = vi.fn();

    emitter.on("order:cancelled", () => {
      throw err;
    });
    emitter.on("order:cancelled", secondFn);

    const res = await emitter.emit("order:cancelled", { id: "99" });
    expect(res.handledCount).toBe(2);
    expect(res.errors).toEqual([err]);
    expect(secondFn).toHaveBeenCalled();
  });

  it("invokes once() listeners at most once", async () => {
    const emitter = new TypedEmitter<AppEvents>();
    const fn = vi.fn();

    emitter.once("order:created", fn);
    expect(emitter.listenerCount("order:created")).toBe(1);

    await emitter.emit("order:created", { id: "1", amount: 10 });
    expect(fn).toHaveBeenCalledTimes(1);
    expect(emitter.listenerCount("order:created")).toBe(0);

    await emitter.emit("order:created", { id: "2", amount: 20 });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("unsubscribes cleanly via returned function", async () => {
    const emitter = new TypedEmitter<AppEvents>();
    const fn = vi.fn();

    const unsub = emitter.on("payment:processed", fn);
    unsub();

    await emitter.emit("payment:processed", { id: "p1" });
    expect(fn).not.toHaveBeenCalled();
    expect(emitter.listenerCount()).toBe(0);
  });

  it("matches prefix wildcards strictly on namespace delimiter", async () => {
    const emitter = new TypedEmitter<{
      "user:add": any;
      "user_edit": any;
    }>();
    const prefixHandler = vi.fn();

    emitter.on("user:*", prefixHandler);

    await emitter.emit("user:add", {});
    expect(prefixHandler).toHaveBeenCalledTimes(1);

    await emitter.emit("user_edit", {});
    expect(prefixHandler).toHaveBeenCalledTimes(1); // should NOT match user_edit
  });

  it("handles removeAllListeners for specific and all events", async () => {
    const emitter = new TypedEmitter<AppEvents>();
    emitter.on("order:created", () => {});
    emitter.on("order:cancelled", () => {});
    emitter.on("*", () => {});

    expect(emitter.listenerCount()).toBe(3);

    emitter.removeAllListeners("order:created");
    expect(emitter.listenerCount("order:created")).toBe(0);
    expect(emitter.listenerCount()).toBe(2);

    emitter.removeAllListeners();
    expect(emitter.listenerCount()).toBe(0);
  });
});

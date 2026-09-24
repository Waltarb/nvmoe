import { describe, it, expect, vi } from "vitest";
import { TypedEmitter } from "../src/emitter";

interface AppEvents {
  "user:login": { username: string };
  "user:logout": { username: string };
}

describe("TypedEmitter (visible)", () => {
  it("emits and handles typed events", async () => {
    const emitter = new TypedEmitter<AppEvents>();
    const fn = vi.fn();

    emitter.on("user:login", fn);
    const res = await emitter.emit("user:login", { username: "alice" });

    expect(res.handledCount).toBe(1);
    expect(res.stopped).toBe(false);
    expect(fn).toHaveBeenCalledWith({ username: "alice" }, expect.anything());
  });

  it("handles wildcard listeners in tier order", async () => {
    const emitter = new TypedEmitter<AppEvents>();
    const order: string[] = [];

    emitter.on("*", () => { order.push("global"); });
    emitter.on("user:*", () => { order.push("prefix"); });
    emitter.on("user:login", () => { order.push("exact"); });

    await emitter.emit("user:login", { username: "alice" });

    expect(order).toEqual(["exact", "prefix", "global"]);
  });
});

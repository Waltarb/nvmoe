import type { EventMap, EventHandler, EmitResult } from "./types";

export class TypedEmitter<T extends EventMap> {
  on<K extends string>(event: K, handler: EventHandler<any>): () => void {
    return () => {};
  }

  once<K extends string>(event: K, handler: EventHandler<any>): () => void {
    return () => {};
  }

  async emit<K extends keyof T & string>(event: K, payload: T[K]): Promise<EmitResult> {
    return { event, handledCount: 0, stopped: false, errors: [] };
  }

  listenerCount(event?: string): number {
    return 0;
  }

  removeAllListeners(event?: string): void {
    // Not implemented
  }
}

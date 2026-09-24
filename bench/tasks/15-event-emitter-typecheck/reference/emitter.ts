import type { EventMap, EventHandler, EmitResult, EventContext } from "../src/types";

interface ListenerEntry {
  eventPattern: string;
  handler: EventHandler<any>;
  once: boolean;
}

export class TypedEmitter<T extends EventMap> {
  private listeners: ListenerEntry[] = [];

  on<K extends string>(event: K, handler: EventHandler<any>): () => void {
    const entry: ListenerEntry = { eventPattern: event, handler, once: false };
    this.listeners.push(entry);
    return () => {
      const idx = this.listeners.indexOf(entry);
      if (idx !== -1) {
        this.listeners.splice(idx, 1);
      }
    };
  }

  once<K extends string>(event: K, handler: EventHandler<any>): () => void {
    const entry: ListenerEntry = { eventPattern: event, handler, once: true };
    this.listeners.push(entry);
    return () => {
      const idx = this.listeners.indexOf(entry);
      if (idx !== -1) {
        this.listeners.splice(idx, 1);
      }
    };
  }

  async emit<K extends keyof T & string>(event: K, payload: T[K]): Promise<EmitResult> {
    const exact: ListenerEntry[] = [];
    const prefix: ListenerEntry[] = [];
    const global: ListenerEntry[] = [];

    for (const entry of this.listeners) {
      if (entry.eventPattern === event) {
        exact.push(entry);
      } else if (entry.eventPattern.endsWith(":*")) {
        const pre = entry.eventPattern.slice(0, -2);
        if (event.startsWith(pre + ":")) {
          prefix.push(entry);
        }
      } else if (entry.eventPattern === "*") {
        global.push(entry);
      }
    }

    const toRun = [...exact, ...prefix, ...global];
    let stopped = false;
    let handledCount = 0;
    const errors: Error[] = [];

    const ctx: EventContext = {
      event,
      stopPropagation: () => {
        stopped = true;
      },
      get isPropagationStopped() {
        return stopped;
      },
    };

    for (const entry of toRun) {
      if (stopped) break;

      handledCount++;
      if (entry.once) {
        const idx = this.listeners.indexOf(entry);
        if (idx !== -1) {
          this.listeners.splice(idx, 1);
        }
      }

      try {
        await entry.handler(payload, ctx);
      } catch (err: any) {
        errors.push(err instanceof Error ? err : new Error(String(err)));
      }
    }

    return {
      event,
      handledCount,
      stopped,
      errors,
    };
  }

  listenerCount(event?: string): number {
    if (event === undefined) {
      return this.listeners.length;
    }
    return this.listeners.filter((l) => l.eventPattern === event).length;
  }

  removeAllListeners(event?: string): void {
    if (event === undefined) {
      this.listeners = [];
    } else {
      this.listeners = this.listeners.filter((l) => l.eventPattern !== event);
    }
  }
}

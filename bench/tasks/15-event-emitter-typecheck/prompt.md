# Feature: Strictly-Typed Hierarchical Event Emitter with Wildcards

Implement `TypedEmitter<T>` in `src/emitter.ts` providing strongly typed events, namespace wildcards (`prefix:*`), async handler sequencing, and cancellation propagation.

## Interface & Types
```ts
export type EventMap = Record<string, unknown>;

export interface EventContext {
  readonly event: string;
  stopPropagation(): void;
  readonly isPropagationStopped: boolean;
}

export type EventHandler<TPayload> = (payload: TPayload, ctx: EventContext) => void | Promise<void>;

export interface EmitResult {
  event: string;
  handledCount: number;
  stopped: boolean;
  errors: Error[];
}
```

## `TypedEmitter<T extends EventMap>`
- `on(event, handler): () => void`: Registers a handler. Returns an unsubscribe function.
- `once(event, handler): () => void`: Registers a single-shot handler. Unregisters after first invocation.
- `emit<K extends keyof T & string>(event: K, payload: T[K]): Promise<EmitResult>`:
  - Invokes matching handlers sequentially in the following tier order:
    1. Exact event listeners (`event`).
    2. Prefix wildcard listeners (`prefix:*` where `event.startsWith(prefix + ":")`).
    3. Global wildcard listeners (`*`).
  - Within each tier, handlers run in registration order.
  - Passes `payload` and an `EventContext` to each handler.
  - If a handler calls `ctx.stopPropagation()`, ceases invoking further handlers and sets `stopped: true`.
  - If a handler throws or returns a rejected Promise:
    - Captures the Error into `result.errors`.
    - Continues executing remaining handlers (unless propagation was stopped).
  - Returns `EmitResult { event, handledCount, stopped, errors }`.
- `listenerCount(event?: string): number`: Returns count of registered listeners for a specific event or pattern, or total listeners if omitted.
- `removeAllListeners(event?: string): void`: Removes listeners for `event`, or all listeners if omitted.

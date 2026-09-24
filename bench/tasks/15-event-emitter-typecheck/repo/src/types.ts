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

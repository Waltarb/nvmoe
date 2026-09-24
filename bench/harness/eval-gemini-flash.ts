import { evaluateCandidate } from "./run-eval.ts";

async function main() {
  const results: { taskId: string; solved: boolean; hiddenPassed: number; hiddenTotal: number }[] = [];

  const tasks = [
    {
      id: "06-async-batch-scheduler",
      resp: `
src/scheduler.ts
<<<<<<< SEARCH
  get pendingCount(): number {
    return 0;
  }

  async schedule(key: K, signal?: AbortSignal): Promise<V> {
    throw new Error("Not implemented");
  }

  async flush(): Promise<void> {
    // Not implemented
  }
=======
  private pending = new Map<K, { resolve: (v: V) => void; reject: (err: any) => void; signal?: AbortSignal; onAbort?: () => void; aborted?: boolean }[]>();
  private timer: any = null;
  private inFlight: Promise<void> | null = null;

  get pendingCount(): number {
    let count = 0;
    for (const callers of this.pending.values()) {
      if (callers.some(c => !c.aborted)) count++;
    }
    return count;
  }

  async schedule(key: K, signal?: AbortSignal): Promise<V> {
    if (signal?.aborted) {
      throw signal.reason || new Error("Aborted");
    }

    return new Promise<V>((resolve, reject) => {
      const entry = { resolve, reject, signal, onAbort: undefined as (() => void) | undefined, aborted: false };

      if (signal) {
        entry.onAbort = () => {
          entry.aborted = true;
          signal.removeEventListener("abort", entry.onAbort!);
          reject(signal.reason || new Error("Aborted"));
          this.pruneKey(key);
        };
        signal.addEventListener("abort", entry.onAbort);
      }

      if (!this.pending.has(key)) {
        this.pending.set(key, []);
      }
      this.pending.get(key)!.push(entry);

      if (this.pendingCount >= this.options.maxBatchSize) {
        this.dispatch();
      } else if (!this.timer) {
        this.timer = setTimeout(() => {
          this.timer = null;
          this.dispatch();
        }, this.options.maxWaitMs);
      }
    });
  }

  private pruneKey(key: K) {
    const list = this.pending.get(key);
    if (!list) return;
    const active = list.filter(c => !c.aborted);
    if (active.length === 0) {
      this.pending.delete(key);
      if (this.pendingCount === 0 && this.timer) {
        clearTimeout(this.timer);
        this.timer = null;
      }
    } else {
      this.pending.set(key, active);
    }
  }

  private dispatch(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const current = this.pending;
    this.pending = new Map();

    const keys: K[] = [];
    for (const [k, callers] of current.entries()) {
      if (callers.some(c => !c.aborted)) keys.push(k);
    }

    if (keys.length === 0) return Promise.resolve();

    const p = (async () => {
      try {
        const res = await this.options.batchFn(keys);
        for (const k of keys) {
          const callers = current.get(k) || [];
          if (!res.has(k)) {
            const err = new Error("Missing key in batch result: " + String(k));
            for (const c of callers) {
              if (c.signal && c.onAbort) c.signal.removeEventListener("abort", c.onAbort);
              if (!c.aborted) c.reject(err);
            }
          } else {
            const val = res.get(k)!;
            for (const c of callers) {
              if (c.signal && c.onAbort) c.signal.removeEventListener("abort", c.onAbort);
              if (!c.aborted) c.resolve(val);
            }
          }
        }
      } catch (err) {
        for (const callers of current.values()) {
          for (const c of callers) {
            if (c.signal && c.onAbort) c.signal.removeEventListener("abort", c.onAbort);
            if (!c.aborted) c.reject(err);
          }
        }
      }
    })();

    this.inFlight = p;
    return p;
  }

  async flush(): Promise<void> {
    if (this.pendingCount > 0) {
      await this.dispatch();
    }
    if (this.inFlight) {
      await this.inFlight;
    }
  }
>>>>>>> REPLACE
`,
    },
    {
      id: "07-segmented-lru-cache",
      resp: `
src/segmented-cache.ts
<<<<<<< SEARCH
  get(key: K): V | undefined {
    return undefined;
  }

  set(key: K, value: V, ttlMs?: number): void {
    // Not implemented
  }

  has(key: K): boolean {
    return false;
  }

  delete(key: K): boolean {
    return false;
  }

  size(): CacheSizes {
    return { probation: 0, protected: 0, ghost: 0 };
  }

  clear(): void {
    // Not implemented
  }
=======
  private probation = new Map<K, { value: V; expiresAt?: number }>();
  private protected = new Map<K, { value: V; expiresAt?: number }>();
  private ghost = new Map<K, boolean>();

  private isExpired(entry: { expiresAt?: number }): boolean {
    return entry.expiresAt !== undefined && Date.now() >= entry.expiresAt;
  }

  private makeMru<T>(map: Map<K, T>, key: K, val: T): void {
    map.delete(key);
    map.set(key, val);
  }

  private insertGhost(key: K): void {
    this.makeMru(this.ghost, key, true);
    if (this.ghost.size > this.options.ghostCapacity) {
      const oldest = this.ghost.keys().next().value;
      if (oldest !== undefined) this.ghost.delete(oldest);
    }
  }

  private insertProbation(key: K, entry: { value: V; expiresAt?: number }): void {
    this.makeMru(this.probation, key, entry);
    if (this.probation.size > this.options.probationCapacity) {
      const oldest = this.probation.keys().next().value;
      if (oldest !== undefined) {
        this.probation.delete(oldest);
        this.insertGhost(oldest);
      }
    }
  }

  private insertProtected(key: K, entry: { value: V; expiresAt?: number }): void {
    this.makeMru(this.protected, key, entry);
    if (this.protected.size > this.options.protectedCapacity) {
      const oldest = this.protected.keys().next().value;
      if (oldest !== undefined) {
        const demoted = this.protected.get(oldest)!;
        this.protected.delete(oldest);
        this.insertProbation(oldest, demoted);
      }
    }
  }

  get(key: K): V | undefined {
    const pEntry = this.probation.get(key);
    if (pEntry) {
      if (this.isExpired(pEntry)) {
        this.probation.delete(key);
        return undefined;
      }
      this.probation.delete(key);
      this.insertProtected(key, pEntry);
      return pEntry.value;
    }

    const protEntry = this.protected.get(key);
    if (protEntry) {
      if (this.isExpired(protEntry)) {
        this.protected.delete(key);
        return undefined;
      }
      this.makeMru(this.protected, key, protEntry);
      return protEntry.value;
    }

    return undefined;
  }

  set(key: K, value: V, ttlMs?: number): void {
    const effectiveTtl = ttlMs ?? this.options.defaultTtlMs;
    const expiresAt = effectiveTtl !== undefined ? Date.now() + effectiveTtl : undefined;
    const entry = { value, expiresAt };

    if (this.protected.has(key)) {
      this.makeMru(this.protected, key, entry);
      return;
    }
    if (this.probation.has(key)) {
      this.probation.delete(key);
      this.insertProtected(key, entry);
      return;
    }
    if (this.ghost.has(key)) {
      this.ghost.delete(key);
      this.insertProtected(key, entry);
      return;
    }
    this.insertProbation(key, entry);
  }

  has(key: K): boolean {
    const pEntry = this.probation.get(key);
    if (pEntry) {
      if (this.isExpired(pEntry)) {
        this.probation.delete(key);
        return false;
      }
      return true;
    }
    const protEntry = this.protected.get(key);
    if (protEntry) {
      if (this.isExpired(protEntry)) {
        this.protected.delete(key);
        return false;
      }
      return true;
    }
    return false;
  }

  delete(key: K): boolean {
    let found = false;
    if (this.probation.delete(key)) found = true;
    if (this.protected.delete(key)) found = true;
    if (this.ghost.delete(key)) found = true;
    return found;
  }

  size(): CacheSizes {
    return {
      probation: this.probation.size,
      protected: this.protected.size,
      ghost: this.ghost.size,
    };
  }

  clear(): void {
    this.probation.clear();
    this.protected.clear();
    this.ghost.clear();
  }
>>>>>>> REPLACE
`,
    },
    {
      id: "08-websocket-frame-codec",
      resp: `
src/codec.ts
<<<<<<< SEARCH
export function encodeFrame(frame: WebSocketFrame, maskKey?: Uint8Array): Uint8Array {
  throw new Error("Not implemented");
}

export class FrameDecoder {
  get bufferedBytes(): number {
    return 0;
  }

  push(chunk: Uint8Array): WebSocketFrame[] {
    return [];
  }
}
=======
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

  if (isMasked) headerLen += 4;

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

      if (rsv1 || rsv2 || rsv3) throw new Error("RSV bits must be 0");

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
        payloadLen = Number(view.getBigUint64(0, false));
        curOffset += 8;
      }

      if (opcode >= 0x8 && (!fin || payloadLen > 125)) {
        throw new Error("Invalid control frame");
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

      frames.push({ fin, rsv1, rsv2, rsv3, opcode, payload });
    }

    if (offset > 0) {
      this.buffer = this.buffer.slice(offset);
    }

    return frames;
  }
}
>>>>>>> REPLACE
`,
    },
    {
      id: "09-mvcc-transaction-kv",
      resp: `
src/mvcc.ts
<<<<<<< SEARCH
export class Transaction {
  readonly id: number;
  readonly readSnapshotVersion: number;

  constructor(id: number, readSnapshotVersion: number) {
    this.id = id;
    this.readSnapshotVersion = readSnapshotVersion;
  }

  get(key: string): string | undefined {
    throw new Error("Not implemented");
  }

  set(key: string, value: string): void {
    throw new Error("Not implemented");
  }

  delete(key: string): void {
    throw new Error("Not implemented");
  }

  commit(): void {
    throw new Error("Not implemented");
  }

  rollback(): void {
    throw new Error("Not implemented");
  }
}

export class MvccStore {
  get currentVersion(): number {
    return 0;
  }

  get activeTxCount(): number {
    return 0;
  }

  beginTransaction(): Transaction {
    throw new Error("Not implemented");
  }

  getCommitted(key: string): string | undefined {
    return undefined;
  }

  vacuum(): number {
    return 0;
  }
}
=======
export class Transaction {
  readonly id: number;
  readonly readSnapshotVersion: number;
  private store: MvccStore;
  private writes = new Map<string, string | null>();
  private closed = false;

  constructor(id: number, readSnapshotVersion: number, store: MvccStore) {
    this.id = id;
    this.readSnapshotVersion = readSnapshotVersion;
    this.store = store;
  }

  private checkOpen(): void {
    if (this.closed) throw new Error("Transaction is closed");
  }

  get(key: string): string | undefined {
    this.checkOpen();
    if (this.writes.has(key)) {
      const val = this.writes.get(key)!;
      return val === null ? undefined : val;
    }
    return this.store.readVersion(key, this.readSnapshotVersion);
  }

  set(key: string, value: string): void {
    this.checkOpen();
    this.writes.set(key, value);
  }

  delete(key: string): void {
    this.checkOpen();
    this.writes.set(key, null);
  }

  commit(): void {
    this.checkOpen();
    try {
      this.store.commitTransaction(this, this.writes);
    } finally {
      this.closed = true;
    }
  }

  rollback(): void {
    this.checkOpen();
    this.closed = true;
    this.store.rollbackTransaction(this);
  }
}

export class MvccStore {
  private _currentVersion = 0;
  private nextTxId = 1;
  private activeTx = new Map<number, Transaction>();
  private data = new Map<string, VersionRecord[]>();

  get currentVersion(): number {
    return this._currentVersion;
  }

  get activeTxCount(): number {
    return this.activeTx.size;
  }

  beginTransaction(): Transaction {
    const txId = this.nextTxId++;
    const tx = new Transaction(txId, this._currentVersion, this);
    this.activeTx.set(txId, tx);
    return tx;
  }

  readVersion(key: string, maxVersion: number): string | undefined {
    const versions = this.data.get(key);
    if (!versions || versions.length === 0) return undefined;
    for (let i = versions.length - 1; i >= 0; i--) {
      const rec = versions[i];
      if (rec.version <= maxVersion) {
        return rec.value === null ? undefined : rec.value;
      }
    }
    return undefined;
  }

  getCommitted(key: string): string | undefined {
    return this.readVersion(key, this._currentVersion);
  }

  commitTransaction(tx: Transaction, writes: Map<string, string | null>): void {
    this.activeTx.delete(tx.id);

    for (const key of writes.keys()) {
      const versions = this.data.get(key);
      if (versions && versions.length > 0) {
        const latest = versions[versions.length - 1];
        if (latest.version > tx.readSnapshotVersion) {
          throw new SerializationConflictError(key);
        }
      }
    }

    if (writes.size === 0) return;

    this._currentVersion++;
    const commitVer = this._currentVersion;

    for (const [key, val] of writes.entries()) {
      let versions = this.data.get(key);
      if (!versions) {
        versions = [];
        this.data.set(key, versions);
      }
      versions.push({ version: commitVer, value: val });
    }
  }

  rollbackTransaction(tx: Transaction): void {
    this.activeTx.delete(tx.id);
  }

  vacuum(): number {
    let minSnapshot = this._currentVersion;
    if (this.activeTx.size > 0) {
      for (const tx of this.activeTx.values()) {
        if (tx.readSnapshotVersion < minSnapshot) {
          minSnapshot = tx.readSnapshotVersion;
        }
      }
    }

    let prunedCount = 0;
    for (const [key, versions] of Array.from(this.data.entries())) {
      let lastLeIdx = -1;
      for (let i = 0; i < versions.length; i++) {
        if (versions[i].version <= minSnapshot) lastLeIdx = i;
        else break;
      }

      if (lastLeIdx > 0) {
        prunedCount += lastLeIdx;
        versions.splice(0, lastLeIdx);
      }

      if (versions.length === 1 && versions[0].value === null) {
        this.data.delete(key);
        prunedCount += 1;
      }
    }

    return prunedCount;
  }
}
>>>>>>> REPLACE
`,
    },
    {
      id: "10-raft-state-machine",
      resp: `
src/raft.ts
<<<<<<< SEARCH
  handleElectionTimeout(): RequestVoteArgs | null {
    throw new Error("Not implemented");
  }

  handleRequestVote(args: RequestVoteArgs): RequestVoteReply {
    throw new Error("Not implemented");
  }

  handleRequestVoteReply(peerId: string, reply: RequestVoteReply): boolean {
    throw new Error("Not implemented");
  }

  handleAppendEntries(args: AppendEntriesArgs): AppendEntriesReply {
    throw new Error("Not implemented");
  }

  propose(command: string): LogEntry | null {
    return null;
  }
=======
  private votesReceived = new Set<string>();

  private get lastLogIndex(): number {
    return this.log.length > 0 ? this.log[this.log.length - 1].index : 0;
  }

  private get lastLogTerm(): number {
    return this.log.length > 0 ? this.log[this.log.length - 1].term : 0;
  }

  handleElectionTimeout(): RequestVoteArgs | null {
    if (this.role === "leader") return null;

    this.role = "candidate";
    this.currentTerm += 1;
    this.votedFor = this.nodeId;
    this.votesReceived.clear();
    this.votesReceived.add(this.nodeId);

    const clusterSize = this.peerIds.length + 1;
    const majority = Math.floor(clusterSize / 2) + 1;

    if (this.votesReceived.size >= majority) {
      this.role = "leader";
      return null;
    }

    return {
      term: this.currentTerm,
      candidateId: this.nodeId,
      lastLogIndex: this.lastLogIndex,
      lastLogTerm: this.lastLogTerm,
    };
  }

  handleRequestVote(args: RequestVoteArgs): RequestVoteReply {
    if (args.term < this.currentTerm) {
      return { term: this.currentTerm, voteGranted: false };
    }

    if (args.term > this.currentTerm) {
      this.role = "follower";
      this.currentTerm = args.term;
      this.votedFor = null;
      this.votesReceived.clear();
    }

    const canVote = this.votedFor === null || this.votedFor === args.candidateId;
    const isUpToDate =
      args.lastLogTerm > this.lastLogTerm ||
      (args.lastLogTerm === this.lastLogTerm && args.lastLogIndex >= this.lastLogIndex);

    if (canVote && isUpToDate) {
      this.votedFor = args.candidateId;
      return { term: this.currentTerm, voteGranted: true };
    }

    return { term: this.currentTerm, voteGranted: false };
  }

  handleRequestVoteReply(peerId: string, reply: RequestVoteReply): boolean {
    if (reply.term > this.currentTerm) {
      this.role = "follower";
      this.currentTerm = reply.term;
      this.votedFor = null;
      this.votesReceived.clear();
      return false;
    }

    if (this.role === "candidate" && reply.term === this.currentTerm && reply.voteGranted) {
      this.votesReceived.add(peerId);
      const clusterSize = this.peerIds.length + 1;
      const majority = Math.floor(clusterSize / 2) + 1;
      if (this.votesReceived.size >= majority) {
        this.role = "leader";
        return true;
      }
    }

    return false;
  }

  handleAppendEntries(args: AppendEntriesArgs): AppendEntriesReply {
    if (args.term < this.currentTerm) {
      return { term: this.currentTerm, success: false };
    }

    if (args.term > this.currentTerm || (this.role === "candidate" && args.term === this.currentTerm)) {
      this.role = "follower";
      this.currentTerm = args.term;
      this.votedFor = null;
      this.votesReceived.clear();
    }

    if (args.prevLogIndex > 0) {
      if (this.log.length < args.prevLogIndex) {
        return { term: this.currentTerm, success: false };
      }
      const entry = this.log[args.prevLogIndex - 1];
      if (entry.term !== args.prevLogTerm) {
        return { term: this.currentTerm, success: false };
      }
    }

    for (let i = 0; i < args.entries.length; i++) {
      const newEntry = args.entries[i];
      const targetIndex = newEntry.index;
      if (this.log.length >= targetIndex) {
        if (this.log[targetIndex - 1].term !== newEntry.term) {
          this.log.splice(targetIndex - 1);
          this.log.push(newEntry);
        }
      } else {
        this.log.push(newEntry);
      }
    }

    if (args.leaderCommit > this.commitIndex) {
      this.commitIndex = Math.min(args.leaderCommit, this.lastLogIndex);
    }

    return { term: this.currentTerm, success: true };
  }

  propose(command: string): LogEntry | null {
    if (this.role !== "leader") return null;
    const entry: LogEntry = {
      term: this.currentTerm,
      index: this.log.length + 1,
      command,
    };
    this.log.push(entry);
    return entry;
  }
>>>>>>> REPLACE
`,
    },
    {
      id: "11-expression-ast-optimizer",
      resp: `
src/optimizer.ts
<<<<<<< SEARCH
export function optimizeAST(node: ASTNode): ASTNode {
  return node;
}
=======
function areNodesEqual(a: ASTNode, b: ASTNode): boolean {
  if (a.type !== b.type) return false;
  if (a.type === "literal" && b.type === "literal") return a.value === b.value;
  if (a.type === "identifier" && b.type === "identifier") return a.name === b.name;
  if (a.type === "unary" && b.type === "unary") return a.op === b.op && areNodesEqual(a.argument, b.argument);
  if (a.type === "binary" && b.type === "binary") return a.op === b.op && areNodesEqual(a.left, b.left) && areNodesEqual(a.right, b.right);
  return false;
}

function optimizeStep(node: ASTNode): ASTNode {
  if (node.type === "literal" || node.type === "identifier") return node;

  if (node.type === "unary") {
    const arg = optimizeStep(node.argument);
    if (arg.type === "unary" && arg.op === node.op) return arg.argument;
    if (arg.type === "literal") {
      if (node.op === "!" && typeof arg.value === "boolean") return { type: "literal", value: !arg.value };
      if (node.op === "-" && typeof arg.value === "number") return { type: "literal", value: -arg.value };
    }
    if (node.op === "!" && arg.type === "binary" && (arg.op === "&&" || arg.op === "||")) {
      const newOp = arg.op === "&&" ? "||" : "&&";
      return {
        type: "binary",
        op: newOp,
        left: optimizeStep({ type: "unary", op: "!", argument: arg.left }),
        right: optimizeStep({ type: "unary", op: "!", argument: arg.right }),
      };
    }
    return { type: "unary", op: node.op, argument: arg };
  }

  if (node.type === "binary") {
    const left = optimizeStep(node.left);
    const right = optimizeStep(node.right);
    const op = node.op;

    if (op === "==" && areNodesEqual(left, right)) return { type: "literal", value: true };
    if (op === "!=" && areNodesEqual(left, right)) return { type: "literal", value: false };

    if (left.type === "literal" && right.type === "literal") {
      const lv = left.value;
      const rv = right.value;
      if (typeof lv === "number" && typeof rv === "number") {
        if (op === "+") return { type: "literal", value: lv + rv };
        if (op === "-") return { type: "literal", value: lv - rv };
        if (op === "*") return { type: "literal", value: lv * rv };
        if (op === "/" && rv !== 0) return { type: "literal", value: lv / rv };
        if (op === "==") return { type: "literal", value: lv === rv };
        if (op === "!=") return { type: "literal", value: lv !== rv };
      }
      if (typeof lv === "boolean" && typeof rv === "boolean") {
        if (op === "&&") return { type: "literal", value: lv && rv };
        if (op === "||") return { type: "literal", value: lv || rv };
        if (op === "==") return { type: "literal", value: lv === rv };
        if (op === "!=") return { type: "literal", value: lv !== rv };
      }
    }

    if (op === "+") {
      if (right.type === "literal" && right.value === 0) return left;
      if (left.type === "literal" && left.value === 0) return right;
    }
    if (op === "-") {
      if (right.type === "literal" && right.value === 0) return left;
    }
    if (op === "*") {
      if (right.type === "literal" && right.value === 1) return left;
      if (left.type === "literal" && left.value === 1) return right;
      if (right.type === "literal" && right.value === 0) return { type: "literal", value: 0 };
      if (left.type === "literal" && left.value === 0) return { type: "literal", value: 0 };
    }
    if (op === "&&") {
      if (right.type === "literal" && right.value === true) return left;
      if (left.type === "literal" && left.value === true) return right;
      if (right.type === "literal" && right.value === false) return { type: "literal", value: false };
      if (left.type === "literal" && left.value === false) return { type: "literal", value: false };
    }
    if (op === "||") {
      if (right.type === "literal" && right.value === false) return left;
      if (left.type === "literal" && left.value === false) return right;
      if (right.type === "literal" && right.value === true) return { type: "literal", value: true };
      if (left.type === "literal" && left.value === true) return { type: "literal", value: true };
    }

    return { type: "binary", op, left, right };
  }

  return node;
}

export function optimizeAST(node: ASTNode): ASTNode {
  let current = node;
  for (let i = 0; i < 20; i++) {
    const next = optimizeStep(current);
    if (areNodesEqual(current, next)) return next;
    current = next;
  }
  return current;
}
>>>>>>> REPLACE
`,
    },
    {
      id: "12-token-bucket-rate-limiter",
      resp: `
src/limiter.ts
<<<<<<< SEARCH
export class RateLimiter {
  private options: RateLimiterOptions;

  constructor(options: RateLimiterOptions) {
    this.options = options;
  }

  get availableTokens(): number {
    return 0;
  }

  acquire(tokens = 1): AcquireResult {
    throw new Error("Not implemented");
  }

  tryAcquire(tokens = 1): boolean {
    return false;
  }

  reset(): void {
    // Not implemented
  }
}
=======
export class RateLimiter {
  private options: RateLimiterOptions;
  private tokens: number;
  private lastRefill: number;

  constructor(options: RateLimiterOptions) {
    this.options = options;
    const now = options.nowFn ? options.nowFn() : Date.now();
    this.tokens = options.initialTokens !== undefined ? options.initialTokens : options.capacity;
    this.lastRefill = now;
  }

  private getNow(): number {
    return this.options.nowFn ? this.options.nowFn() : Date.now();
  }

  private refill(): void {
    const now = this.getNow();
    if (now > this.lastRefill) {
      const elapsedMs = now - this.lastRefill;
      const tokensToAdd = elapsedMs * (this.options.refillRatePerSec / 1000);
      this.tokens = Math.min(this.options.capacity, this.tokens + tokensToAdd);
      this.lastRefill = now;
    } else {
      this.lastRefill = now;
    }
  }

  get availableTokens(): number {
    this.refill();
    return this.tokens;
  }

  acquire(tokens = 1): AcquireResult {
    if (tokens <= 0) throw new Error("Tokens requested must be positive");
    this.refill();

    if (this.options.maxBurstSpikeRatio !== undefined) {
      const maxAllowed = this.options.capacity * this.options.maxBurstSpikeRatio;
      if (tokens > maxAllowed) {
        return { allowed: false, remainingTokens: this.tokens, retryAfterMs: Infinity };
      }
    }

    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return { allowed: true, remainingTokens: this.tokens, retryAfterMs: 0 };
    }

    const deficit = tokens - this.tokens;
    const ratePerMs = this.options.refillRatePerSec / 1000;
    const retryAfterMs = Math.ceil(deficit / ratePerMs);
    return { allowed: false, remainingTokens: this.tokens, retryAfterMs };
  }

  tryAcquire(tokens = 1): boolean {
    return this.acquire(tokens).allowed;
  }

  reset(): void {
    this.tokens = this.options.capacity;
    this.lastRefill = this.getNow();
  }
}
>>>>>>> REPLACE
`,
    },
    {
      id: "13-json-schema-validator",
      resp: `
src/validator.ts
<<<<<<< SEARCH
export function createValidator(
  schema: JSONSchema,
  defs?: Record<string, JSONSchema>
): (data: unknown) => ValidationResult {
  return (data: unknown) => {
    return { valid: true, errors: [] };
  };
}
=======
export function createValidator(
  rootSchema: JSONSchema,
  externalDefs?: Record<string, JSONSchema>
): (data: unknown) => ValidationResult {
  const allDefs: Record<string, JSONSchema> = {
    ...externalDefs,
    ...rootSchema.$defs,
    ...rootSchema.definitions,
  };

  function resolveRef(ref: string): JSONSchema {
    if (ref.startsWith("#/$defs/")) {
      const name = ref.slice("#/$defs/".length);
      const target = allDefs[name];
      if (!target) throw new Error(\`Cannot resolve ref: \${ref}\`);
      return target;
    }
    if (ref.startsWith("#/definitions/")) {
      const name = ref.slice("#/definitions/".length);
      const target = allDefs[name];
      if (!target) throw new Error(\`Cannot resolve ref: \${ref}\`);
      return target;
    }
    throw new Error(\`Unsupported ref format: \${ref}\`);
  }

  function validateNode(data: unknown, schema: JSONSchema, path: string, errors: ValidationError[]): void {
    if (schema.$ref) {
      validateNode(data, resolveRef(schema.$ref), path, errors);
      return;
    }

    if (schema.oneOf) {
      let matchCount = 0;
      for (const sub of schema.oneOf) {
        const subErrors: ValidationError[] = [];
        validateNode(data, sub, path, subErrors);
        if (subErrors.length === 0) matchCount++;
      }
      if (matchCount !== 1) {
        errors.push({ path, message: \`Expected exactly 1 match in oneOf, matched \${matchCount}\` });
      }
      return;
    }

    if (schema.type) {
      const types = Array.isArray(schema.type) ? schema.type : [schema.type];
      let matchesType = false;
      for (const t of types) {
        if (t === "null" && data === null) matchesType = true;
        else if (t === "array" && Array.isArray(data)) matchesType = true;
        else if (t === "object" && typeof data === "object" && data !== null && !Array.isArray(data)) matchesType = true;
        else if (t === "string" && typeof data === "string") matchesType = true;
        else if (t === "number" && typeof data === "number" && !isNaN(data)) matchesType = true;
        else if (t === "boolean" && typeof data === "boolean") matchesType = true;
      }
      if (!matchesType) {
        errors.push({ path, message: \`Expected type \${JSON.stringify(schema.type)}\` });
        return;
      }
    }

    if (typeof data === "string") {
      if (schema.minLength !== undefined && data.length < schema.minLength) errors.push({ path, message: "Too short" });
      if (schema.maxLength !== undefined && data.length > schema.maxLength) errors.push({ path, message: "Too long" });
      if (schema.pattern !== undefined && !new RegExp(schema.pattern).test(data)) errors.push({ path, message: "Pattern mismatch" });
    }

    if (typeof data === "number") {
      if (schema.minimum !== undefined && data < schema.minimum) errors.push({ path, message: "Too small" });
      if (schema.maximum !== undefined && data > schema.maximum) errors.push({ path, message: "Too large" });
    }

    if (Array.isArray(data)) {
      if (schema.minItems !== undefined && data.length < schema.minItems) errors.push({ path, message: "Too few items" });
      if (schema.items) {
        for (let i = 0; i < data.length; i++) validateNode(data[i], schema.items, \`\${path}/\${i}\`, errors);
      }
    }

    if (typeof data === "object" && data !== null && !Array.isArray(data)) {
      const obj = data as Record<string, unknown>;
      if (schema.required) {
        for (const req of schema.required) {
          if (!(req in obj)) errors.push({ path: \`\${path}/\${req}\`, message: "Missing required" });
        }
      }
      if (schema.properties) {
        for (const [prop, propSchema] of Object.entries(schema.properties)) {
          if (prop in obj) validateNode(obj[prop], propSchema, \`\${path}/\${prop}\`, errors);
        }
      }
      if (schema.additionalProperties !== undefined) {
        for (const key of Object.keys(obj)) {
          if (!schema.properties || !(key in schema.properties)) {
            if (schema.additionalProperties === false) {
              errors.push({ path: \`\${path}/\${key}\`, message: "Additional not allowed" });
            } else if (typeof schema.additionalProperties === "object") {
              validateNode(obj[key], schema.additionalProperties, \`\${path}/\${key}\`, errors);
            }
          }
        }
      }
    }
  }

  return (data: unknown): ValidationResult => {
    const errors: ValidationError[] = [];
    validateNode(data, rootSchema, "#", errors);
    return { valid: errors.length === 0, errors };
  };
}
>>>>>>> REPLACE
`,
    },
    {
      id: "14-diff-patch-engine",
      resp: `
src/patch.ts
<<<<<<< SEARCH
export function parseUnifiedDiff(diff: string): DiffHunk[] {
  return [];
}

export function applyPatch(
  original: string,
  diff: string,
  options?: { fuzzFactor?: number }
): PatchResult {
  return { success: false, content: original, failedHunks: [] };
}

export function threeWayMerge(base: string, ours: string, theirs: string): MergeResult {
  return { merged: "", conflicts: false };
}
=======
export function parseUnifiedDiff(diff: string): DiffHunk[] {
  const lines = diff.replace(/\r\n/g, "\n").split("\n");
  const hunks: DiffHunk[] = [];
  let currentHunk: DiffHunk | null = null;
  const HUNK_HEADER = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

  for (const line of lines) {
    const match = line.match(HUNK_HEADER);
    if (match) {
      if (currentHunk) hunks.push(currentHunk);
      currentHunk = {
        oldStart: parseInt(match[1], 10),
        oldLines: match[2] !== undefined ? parseInt(match[2], 10) : 1,
        newStart: parseInt(match[3], 10),
        newLines: match[4] !== undefined ? parseInt(match[4], 10) : 1,
        lines: [],
      };
      continue;
    }
    if (currentHunk) {
      if (line.startsWith(" ")) currentHunk.lines.push({ type: "context", content: line.slice(1) });
      else if (line.startsWith("+")) currentHunk.lines.push({ type: "add", content: line.slice(1) });
      else if (line.startsWith("-")) currentHunk.lines.push({ type: "delete", content: line.slice(1) });
    }
  }
  if (currentHunk) hunks.push(currentHunk);
  return hunks;
}

export function applyPatch(
  original: string,
  diff: string,
  options?: { fuzzFactor?: number }
): PatchResult {
  const hunks = parseUnifiedDiff(diff);
  const fileLines = original.replace(/\r\n/g, "\n").split("\n");
  const fuzz = options?.fuzzFactor ?? 0;
  const failedHunks: number[] = [];
  let lineOffset = 0;

  for (let hIdx = 0; hIdx < hunks.length; hIdx++) {
    const hunk = hunks[hIdx];
    const targetIdx = hunk.oldStart - 1 + lineOffset;
    const oldPattern = hunk.lines
      .filter((l) => l.type === "context" || l.type === "delete")
      .map((l) => l.content);

    let matchIdx = -1;
    for (let f = 0; f <= fuzz; f++) {
      const candidates = f === 0 ? [targetIdx] : [targetIdx - f, targetIdx + f];
      for (const c of candidates) {
        if (c >= 0 && c + oldPattern.length <= fileLines.length) {
          let matched = true;
          for (let k = 0; k < oldPattern.length; k++) {
            if (fileLines[c + k] !== oldPattern[k]) {
              matched = false;
              break;
            }
          }
          if (matched) { matchIdx = c; break; }
        }
      }
      if (matchIdx !== -1) break;
    }

    if (matchIdx === -1) {
      failedHunks.push(hIdx);
      continue;
    }

    const replacement: string[] = [];
    for (const hl of hunk.lines) {
      if (hl.type === "context" || hl.type === "add") replacement.push(hl.content);
    }

    fileLines.splice(matchIdx, oldPattern.length, ...replacement);
    lineOffset += replacement.length - oldPattern.length;
  }

  return { success: failedHunks.length === 0, content: fileLines.join("\n"), failedHunks };
}

export function threeWayMerge(baseStr: string, oursStr: string, theirsStr: string): MergeResult {
  const base = baseStr.replace(/\r\n/g, "\n").split("\n");
  const ours = oursStr.replace(/\r\n/g, "\n").split("\n");
  const theirs = theirsStr.replace(/\r\n/g, "\n").split("\n");

  const maxLen = Math.max(base.length, ours.length, theirs.length);
  const outLines: string[] = [];
  let hasConflicts = false;

  let i = 0;
  while (i < maxLen) {
    const b = base[i];
    const o = ours[i];
    const t = theirs[i];

    if (o === t) {
      if (o !== undefined) outLines.push(o);
    } else if (o === b) {
      if (t !== undefined) outLines.push(t);
    } else if (t === b) {
      if (o !== undefined) outLines.push(o);
    } else {
      hasConflicts = true;
      outLines.push("<<<<<<< OURS");
      if (o !== undefined) outLines.push(o);
      outLines.push("=======");
      if (t !== undefined) outLines.push(t);
      outLines.push(">>>>>>> THEIRS");
    }
    i++;
  }

  return { merged: outLines.join("\n"), conflicts: hasConflicts };
}
>>>>>>> REPLACE
`,
    },
    {
      id: "15-event-emitter-typecheck",
      resp: `
src/emitter.ts
<<<<<<< SEARCH
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
=======
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
      if (idx !== -1) this.listeners.splice(idx, 1);
    };
  }

  once<K extends string>(event: K, handler: EventHandler<any>): () => void {
    const entry: ListenerEntry = { eventPattern: event, handler, once: true };
    this.listeners.push(entry);
    return () => {
      const idx = this.listeners.indexOf(entry);
      if (idx !== -1) this.listeners.splice(idx, 1);
    };
  }

  async emit<K extends keyof T & string>(event: K, payload: T[K]): Promise<EmitResult> {
    const exact: ListenerEntry[] = [];
    const prefix: ListenerEntry[] = [];
    const global: ListenerEntry[] = [];

    for (const entry of this.listeners) {
      if (entry.eventPattern === event) exact.push(entry);
      else if (entry.eventPattern.endsWith(":*")) {
        const pre = entry.eventPattern.slice(0, -2);
        if (event.startsWith(pre + ":")) prefix.push(entry);
      } else if (entry.eventPattern === "*") global.push(entry);
    }

    const toRun = [...exact, ...prefix, ...global];
    let stopped = false;
    let handledCount = 0;
    const errors: Error[] = [];

    const ctx: EventContext = {
      event,
      stopPropagation: () => { stopped = true; },
      get isPropagationStopped() { return stopped; },
    };

    for (const entry of toRun) {
      if (stopped) break;
      handledCount++;
      if (entry.once) {
        const idx = this.listeners.indexOf(entry);
        if (idx !== -1) this.listeners.splice(idx, 1);
      }
      try {
        await entry.handler(payload, ctx);
      } catch (err: any) {
        errors.push(err instanceof Error ? err : new Error(String(err)));
      }
    }

    return { event, handledCount, stopped, errors };
  }

  listenerCount(event?: string): number {
    if (event === undefined) return this.listeners.length;
    return this.listeners.filter((l) => l.eventPattern === event).length;
  }

  removeAllListeners(event?: string): void {
    if (event === undefined) this.listeners = [];
    else this.listeners = this.listeners.filter((l) => l.eventPattern !== event);
  }
}
>>>>>>> REPLACE
`,
    },
  ];

  for (const t of tasks) {
    const r = await evaluateCandidate(t.id, t.resp, "Gemini-Flash-3.8(high)");
    results.push({
      taskId: t.id,
      solved: r.solved,
      hiddenPassed: r.hidden.passed,
      hiddenTotal: r.hidden.total,
    });
  }

  console.log("\n============================================================");
  console.log("             GEMINI FLASH 3.8 (HIGH) BENCHMARK REPORT        ");
  console.log("============================================================");
  console.table(results);
  const totalSolved = results.filter((r) => r.solved).length;
  const totalHiddenPassed = results.reduce((s, r) => s + r.hiddenPassed, 0);
  const totalHidden = results.reduce((s, r) => s + r.hiddenTotal, 0);

  console.log(`Composite Solved Score: ${(totalSolved / results.length) * 100}% (${totalSolved}/${results.length})`);
  console.log(`Fine-Grained Hidden Assertion Score: ${((totalHiddenPassed / totalHidden) * 100).toFixed(1)}% (${totalHiddenPassed}/${totalHidden})`);
}

main().catch(console.error);

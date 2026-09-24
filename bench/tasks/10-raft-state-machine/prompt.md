# Feature: Raft Consensus State Machine

Implement `RaftNode` in `src/raft.ts` modeling core Raft election safety, term rollover, and log replication.

## Types
```ts
export type Role = "follower" | "candidate" | "leader";

export interface LogEntry {
  term: number;
  index: number;
  command: string;
}

export interface RequestVoteArgs {
  term: number;
  candidateId: string;
  lastLogIndex: number;
  lastLogTerm: number;
}

export interface RequestVoteReply {
  term: number;
  voteGranted: boolean;
}

export interface AppendEntriesArgs {
  term: number;
  leaderId: string;
  prevLogIndex: number;
  prevLogTerm: number;
  entries: LogEntry[];
  leaderCommit: number;
}

export interface AppendEntriesReply {
  term: number;
  success: boolean;
}
```

## `RaftNode(nodeId: string, peerIds: string[])`
- Starts with `role = "follower"`, `currentTerm = 0`, `votedFor = null`, `commitIndex = 0`, empty log `[]`.
- Total cluster size = `peerIds.length + 1`. Majority = `Math.floor(clusterSize / 2) + 1`.

### Invariants:
1. `handleElectionTimeout(): RequestVoteArgs | null`:
   - If already leader, do nothing and return `null`.
   - Transitions to `"candidate"`, increments `currentTerm += 1`, votes for self (`votedFor = nodeId`).
   - If single-node cluster (0 peers), immediately transitions to `"leader"` and returns `null`.
   - Returns `RequestVoteArgs` with the candidate's last log index (0 if empty) and last log term (0 if empty).
2. `handleRequestVote(args: RequestVoteArgs): RequestVoteReply`:
   - If `args.term < currentTerm`, return `{ term: currentTerm, voteGranted: false }`.
   - If `args.term > currentTerm`, step down to `"follower"`, `currentTerm = args.term`, `votedFor = null`.
   - Grant vote iff:
     - (`votedFor === null || votedFor === args.candidateId`), AND
     - Candidate's log is up-to-date: `args.lastLogTerm > myLastLogTerm` OR (`args.lastLogTerm === myLastLogTerm && args.lastLogIndex >= myLastLogIndex`).
   - If vote is granted: record `votedFor = args.candidateId` and return `{ term: currentTerm, voteGranted: true }`. Otherwise return `voteGranted: false`.
3. `handleRequestVoteReply(peerId: string, reply: RequestVoteReply): boolean`:
   - If `reply.term > currentTerm`: step down to `"follower"`, `currentTerm = reply.term`, `votedFor = null`, return `false`.
   - If `role === "candidate"` and `reply.term === currentTerm` and `reply.voteGranted`:
     - Tally vote for `peerId`.
     - If votes count reaches majority: transition to `"leader"`, return `true` (became leader).
   - Return `false`.
4. `handleAppendEntries(args: AppendEntriesArgs): AppendEntriesReply`:
   - If `args.term < currentTerm`: return `{ term: currentTerm, success: false }`.
   - If `args.term > currentTerm` or (`role === "candidate"` and `args.term === currentTerm`):
     - Step down to `"follower"`, `currentTerm = args.term`, `votedFor = null`.
   - Log Matching Property:
     - If `args.prevLogIndex > 0`: node must have an entry at `args.prevLogIndex` with `term === args.prevLogTerm`. If not, return `{ term: currentTerm, success: false }`.
   - If an existing entry conflicts with a new entry (same index, different term): delete the existing entry and all entries that follow it.
   - Append any new entries not already in the log.
   - If `args.leaderCommit > commitIndex`:
     - Update `commitIndex = Math.min(args.leaderCommit, lastLogIndex)`.
   - Return `{ term: currentTerm, success: true }`.
5. `propose(command: string): LogEntry | null`:
   - If not leader, returns `null`.
   - Appends `{ term: currentTerm, index: log.length + 1, command }` to own log and returns it.

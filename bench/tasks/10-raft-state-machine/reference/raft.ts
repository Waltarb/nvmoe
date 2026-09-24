import type {
  Role,
  LogEntry,
  RequestVoteArgs,
  RequestVoteReply,
  AppendEntriesArgs,
  AppendEntriesReply,
} from "../src/types";

export class RaftNode {
  readonly nodeId: string;
  readonly peerIds: string[];
  role: Role = "follower";
  currentTerm = 0;
  votedFor: string | null = null;
  log: LogEntry[] = [];
  commitIndex = 0;
  private votesReceived = new Set<string>();

  constructor(nodeId: string, peerIds: string[]) {
    this.nodeId = nodeId;
    this.peerIds = peerIds;
  }

  private get lastLogIndex(): number {
    return this.log.length > 0 ? this.log[this.log.length - 1].index : 0;
  }

  private get lastLogTerm(): number {
    return this.log.length > 0 ? this.log[this.log.length - 1].term : 0;
  }

  handleElectionTimeout(): RequestVoteArgs | null {
    if (this.role === "leader") {
      return null;
    }

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

    const canVoteForCandidate = this.votedFor === null || this.votedFor === args.candidateId;
    const isCandidateLogUpToDate =
      args.lastLogTerm > this.lastLogTerm ||
      (args.lastLogTerm === this.lastLogTerm && args.lastLogIndex >= this.lastLogIndex);

    if (canVoteForCandidate && isCandidateLogUpToDate) {
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
    if (this.role !== "leader") {
      return null;
    }
    const entry: LogEntry = {
      term: this.currentTerm,
      index: this.log.length + 1,
      command,
    };
    this.log.push(entry);
    return entry;
  }
}

import type {
  Role,
  LogEntry,
  RequestVoteArgs,
  RequestVoteReply,
  AppendEntriesArgs,
  AppendEntriesReply,
} from "./types";

export class RaftNode {
  readonly nodeId: string;
  readonly peerIds: string[];
  role: Role = "follower";
  currentTerm = 0;
  votedFor: string | null = null;
  log: LogEntry[] = [];
  commitIndex = 0;

  constructor(nodeId: string, peerIds: string[]) {
    this.nodeId = nodeId;
    this.peerIds = peerIds;
  }

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
}

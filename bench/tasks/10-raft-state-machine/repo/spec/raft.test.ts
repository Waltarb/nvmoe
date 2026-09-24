import { describe, it, expect } from "vitest";
import { RaftNode } from "../src/raft";

describe("Raft Node (visible)", () => {
  it("starts election on election timeout", () => {
    const node = new RaftNode("node1", ["node2", "node3"]);
    const args = node.handleElectionTimeout();

    expect(node.role).toBe("candidate");
    expect(node.currentTerm).toBe(1);
    expect(node.votedFor).toBe("node1");
    expect(args).toEqual({
      term: 1,
      candidateId: "node1",
      lastLogIndex: 0,
      lastLogTerm: 0,
    });
  });

  it("becomes leader when receiving majority votes", () => {
    const node = new RaftNode("node1", ["node2", "node3"]);
    node.handleElectionTimeout();

    const becameLeader = node.handleRequestVoteReply("node2", { term: 1, voteGranted: true });
    expect(becameLeader).toBe(true);
    expect(node.role).toBe("leader");

    const entry = node.propose("set x=1");
    expect(entry).toEqual({ term: 1, index: 1, command: "set x=1" });
  });
});

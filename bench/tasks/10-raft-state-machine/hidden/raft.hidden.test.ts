import { describe, it, expect } from "vitest";
import { RaftNode } from "../src/raft";

describe("Raft Node (hidden)", () => {
  it("rejects vote requests with smaller term", () => {
    const node = new RaftNode("n1", ["n2"]);
    node.currentTerm = 5;

    const reply = node.handleRequestVote({
      term: 4,
      candidateId: "n2",
      lastLogIndex: 0,
      lastLogTerm: 0,
    });

    expect(reply.voteGranted).toBe(false);
    expect(reply.term).toBe(5);
  });

  it("steps down when receiving higher term in AppendEntries", () => {
    const node = new RaftNode("n1", ["n2", "n3"]);
    node.handleElectionTimeout(); // term 1, candidate

    const reply = node.handleAppendEntries({
      term: 3,
      leaderId: "n2",
      prevLogIndex: 0,
      prevLogTerm: 0,
      entries: [],
      leaderCommit: 0,
    });

    expect(reply.success).toBe(true);
    expect(node.role).toBe("follower");
    expect(node.currentTerm).toBe(3);
  });

  it("does not grant second vote in same term to another candidate", () => {
    const node = new RaftNode("n1", ["n2", "n3"]);
    node.currentTerm = 1;
    node.votedFor = "n2";

    const reply = node.handleRequestVote({
      term: 1,
      candidateId: "n3",
      lastLogIndex: 0,
      lastLogTerm: 0,
    });

    expect(reply.voteGranted).toBe(false);
  });

  it("rejects candidate whose log is less up-to-date (term priority)", () => {
    const node = new RaftNode("n1", ["n2"]);
    node.currentTerm = 2;
    node.log = [{ term: 2, index: 1, command: "cmd" }];

    // Candidate has more entries (len 5) but lower term (1)
    const reply = node.handleRequestVote({
      term: 2,
      candidateId: "n2",
      lastLogIndex: 5,
      lastLogTerm: 1,
    });

    expect(reply.voteGranted).toBe(false);
  });

  it("rejects AppendEntries when prevLogIndex does not match", () => {
    const node = new RaftNode("n1", ["n2"]);
    node.currentTerm = 1;
    node.log = [{ term: 1, index: 1, command: "cmd1" }];

    const reply = node.handleAppendEntries({
      term: 1,
      leaderId: "n2",
      prevLogIndex: 2, // node only has index 1!
      prevLogTerm: 1,
      entries: [{ term: 1, index: 3, command: "cmd3" }],
      leaderCommit: 0,
    });

    expect(reply.success).toBe(false);
    expect(node.log.length).toBe(1);
  });

  it("truncates conflicting uncommitted entries and appends new ones", () => {
    const node = new RaftNode("n1", ["n2"]);
    node.currentTerm = 2;
    node.log = [
      { term: 1, index: 1, command: "c1" },
      { term: 1, index: 2, command: "c2" },
      { term: 1, index: 3, command: "c3-stale" },
    ];

    const reply = node.handleAppendEntries({
      term: 2,
      leaderId: "n2",
      prevLogIndex: 2,
      prevLogTerm: 1,
      entries: [
        { term: 2, index: 3, command: "c3-new" },
        { term: 2, index: 4, command: "c4-new" },
      ],
      leaderCommit: 3,
    });

    expect(reply.success).toBe(true);
    expect(node.log.length).toBe(4);
    expect(node.log[2].command).toBe("c3-new");
    expect(node.log[3].command).toBe("c4-new");
    expect(node.commitIndex).toBe(3);
  });

  it("elects leader immediately in single-node cluster", () => {
    const node = new RaftNode("solo", []);
    const args = node.handleElectionTimeout();

    expect(node.role).toBe("leader");
    expect(node.currentTerm).toBe(1);
    expect(args).toBeNull();
  });
});

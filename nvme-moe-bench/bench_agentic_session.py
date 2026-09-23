#!/usr/bin/env python3
"""Phase B (RESULTS_AND_FINDINGS.md #23): measures prefill's real share of wall-clock
time in a representative multi-turn agentic session, rather than the single-shot prompts
used everywhere else in this project's benchmarks.

Each turn appends the previous turn's assistant response plus a synthetic tool-call/
tool-result exchange to the conversation, so context grows the way a real agentic loop's
does. Per turn: TTFT is that turn's prefill time (the KV cache already holds everything
before this turn, so TTFT reflects processing only the newly appended tokens); everything
after the first streamed token is decode for that turn.
"""
import json
import time
import sys
import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-nvfp4-ftw"

SYSTEM = (
    "You are a coding assistant with access to tools: read_file(path), "
    "run_tests(target), grep_code(pattern, path), write_file(path, content). "
    "When you need a tool, respond with a single line: TOOL_CALL: <name>(<args as JSON>). "
    "Otherwise answer directly."
)

# Each entry: (user turn text, synthetic tool name+result to append after the assistant's
# first reply, to simulate the tool round-trip before the NEXT user turn).
TURNS = [
    ("There's a bug in the palindrome checker in utils/strings.py -- can you find it?",
     "read_file(\"utils/strings.py\")",
     "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n\n"
     "def longest_palindrome(s):\n    n = len(s)\n    best = ''\n    for i in range(n):\n"
     "        for j in range(i, n):\n            sub = s[i:j+1]\n            if is_palindrome(sub) and len(sub) > len(best):\n"
     "                best = sub\n    return best\n"),
    ("Now run the test suite for that module and show me failures.",
     "run_tests(\"tests/test_strings.py\")",
     "..F..F....\n======================\nFAIL: test_longest_palindrome_mixed_case\n"
     "AssertionError: expected 'racecar', got ''\nFAIL: test_longest_palindrome_empty\n"
     "AssertionError: expected '', got IndexError: string index out of range\n"
     "Ran 10 tests in 0.042s\nFAILED (failures=2)\n"),
    ("Grep the codebase for other callers of longest_palindrome so I know what else might break.",
     "grep_code(\"longest_palindrome\", \".\")",
     "cli/format.py:12:    result = longest_palindrome(user_input)\n"
     "api/handlers.py:88:    return {'longest': longest_palindrome(text)}\n"
     "tests/test_strings.py:5:from utils.strings import longest_palindrome\n"),
    ("Ok, fix the bug (empty string case) and write the corrected file.",
     "write_file(\"utils/strings.py\", ...)",
     "OK, wrote 14 lines to utils/strings.py.\n"),
    ("Run the tests again to confirm it's fixed.",
     "run_tests(\"tests/test_strings.py\")",
     "..........\nRan 10 tests in 0.038s\nOK\n"),
]


def stream_turn(messages, max_tokens=150):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload, stream=True, timeout=300)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)
    ttft = None
    last = t0
    text_parts = []
    n_tokens = 0
    for line in resp.iter_lines():
        if not line:
            continue
        d = line.decode("utf-8")
        if not d.startswith("data: "):
            continue
        s = d[6:].strip()
        if s == "[DONE]":
            break
        now = time.perf_counter()
        try:
            chunk = json.loads(s)
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("reasoning_content") or delta.get("content")
            if text:
                text_parts.append(text)
                n_tokens += 1
                if ttft is None:
                    ttft = now - t0
                last = now
        except Exception:
            pass
    total = last - t0
    decode_time = total - (ttft or 0.0)
    return {
        "ttft": ttft or 0.0,
        "decode_time": decode_time,
        "total": total,
        "n_tokens": n_tokens,
        "text": "".join(text_parts),
    }


def run_session():
    messages = [{"role": "system", "content": SYSTEM}]
    results = []
    for i, (user_text, tool_call, tool_result) in enumerate(TURNS):
        messages.append({"role": "user", "content": user_text})
        print(f"\n=== Turn {i+1}: {user_text[:60]}...")
        r = stream_turn(messages)
        print(f"    TTFT (prefill): {r['ttft']*1000:.1f}ms | decode: {r['decode_time']*1000:.1f}ms "
              f"for {r['n_tokens']} tokens | context growing")
        results.append(r)
        # Append the assistant's reply, then a synthetic tool round-trip, before the next turn.
        messages.append({"role": "assistant", "content": r["text"] or f"TOOL_CALL: {tool_call}"})
        messages.append({"role": "user", "content": f"[tool result for {tool_call}]\n{tool_result}"})
        time.sleep(1)

    total_prefill = sum(r["ttft"] for r in results)
    total_decode = sum(r["decode_time"] for r in results)
    total = total_prefill + total_decode
    print("\n" + "=" * 60)
    print(f"Session totals over {len(results)} turns:")
    print(f"  Total prefill (TTFT sum): {total_prefill*1000:.1f}ms")
    print(f"  Total decode:             {total_decode*1000:.1f}ms")
    print(f"  Prefill share:            {total_prefill/total*100:.1f}%")
    print(f"  Decode share:             {total_decode/total*100:.1f}%")
    for i, r in enumerate(results):
        print(f"  Turn {i+1}: ttft={r['ttft']*1000:.0f}ms decode={r['decode_time']*1000:.0f}ms "
              f"tokens={r['n_tokens']}")
    print("=" * 60)


if __name__ == "__main__":
    run_session()

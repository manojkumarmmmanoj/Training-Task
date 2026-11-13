import json

train = [json.loads(line) for line in open("candidate_pack/data/train_cleaned.jsonl", encoding="utf-8")]
dev = [json.loads(line) for line in open("candidate_pack/data/dev.jsonl", encoding="utf-8")]
test = [json.loads(line) for line in open("candidate_pack/data/test_inputs.jsonl", encoding="utf-8")]

def stats(name, rows, is_test=False):
    sys_lens = [len(r["messages"][0]["content"].split()) for r in rows]
    user_lens = [len(r["messages"][1]["content"].split()) for r in rows]
    user_chars = [len(r["messages"][1]["content"]) for r in rows]
    print(f"=== {name} ===")
    print(f"Count: {len(rows)}")
    print(f"User words: min={min(user_lens)}, max={max(user_lens)}, avg={sum(user_lens)/len(user_lens):.1f}")
    print(f"User chars: min={min(user_chars)}, max={max(user_chars)}, avg={sum(user_chars)/len(user_chars):.1f}")
    if not is_test:
        asst_lens = [len(r["messages"][2]["content"].split()) for r in rows]
        asst_chars = [len(r["messages"][2]["content"]) for r in rows]
        print(f"Asst chars: min={min(asst_chars)}, max={max(asst_chars)}, avg={sum(asst_chars)/len(asst_chars):.1f}")
        total_approx_tokens = [int((u + a + 15) * 1.3) for u, a in zip(user_lens, asst_lens)]
        print(f"Approx total tokens: min={min(total_approx_tokens)}, max={max(total_approx_tokens)}, avg={sum(total_approx_tokens)/len(total_approx_tokens):.1f}")

stats("Train Cleaned", train)
stats("Dev", dev)
stats("Test", test, is_test=True)


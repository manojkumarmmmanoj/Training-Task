import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("--- 1. JSON Parse Errors ---")
parse_errors = []
for r in train_rows:
    c = r["messages"][-1]["content"]
    try:
        json.loads(c)
    except Exception as e:
        parse_errors.append((r["id"], str(e), c, r["messages"][1]["content"]))

print(f"Count of parse errors: {len(parse_errors)}")
for pid, err, c, user_msg in parse_errors[:5]:
    print(f"ID: {pid} | Err: {err} | Truncated content: {c}")
    print(f"User msg: {user_msg[:100]}...\n")


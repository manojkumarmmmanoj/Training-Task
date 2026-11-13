import json
import re
import csv
from datetime import datetime

# Load data
train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

# Setup cleaning log
cleaning_log = [] # (id, action, reason)

# Sets of IDs for different issues
# 1. Truncated JSON
truncated_ids = set()
for r in train_rows:
    c = r["messages"][-1]["content"]
    try:
        json.loads(c)
    except:
        truncated_ids.add(r["id"])

# 2. Gibberish tickets
gibberish_ids = set()
for r in train_rows:
    user_msg = r["messages"][1]["content"]
    lines = [l.strip() for l in user_msg.split("\n") if l.strip() and not l.startswith("Received:") and not l.startswith("Source:")]
    text = " ".join(lines).lower()
    if text in ["test", "test test 123", "asdf asdf", "asdf", "hello", "foo bar", "testing"] or len(text) < 15:
        gibberish_ids.add(r["id"])

print(f"Truncated: {len(truncated_ids)}")
print(f"Gibberish: {len(gibberish_ids)}")

drop_ids = truncated_ids | gibberish_ids
print(f"Total dropped: {len(drop_ids)}")

for rid in sorted(truncated_ids):
    cleaning_log.append((rid, "drop", "truncated assistant JSON output / incomplete label"))

for rid in sorted(gibberish_ids):
    cleaning_log.append((rid, "drop", "synthetic test placeholder or gibberish input with hallucinated label"))

print(f"Total dropped logged: {len(cleaning_log)}")


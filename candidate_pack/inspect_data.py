import json
import re
from collections import Counter, defaultdict

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_rows = load("candidate_pack/data/train.jsonl")
dev_rows = load("candidate_pack/data/dev.jsonl")
test_rows = load("candidate_pack/data/test_inputs.jsonl")

print(f"Train count: {len(train_rows)}")
print(f"Dev count: {len(dev_rows)}")
print(f"Test count: {len(test_rows)}")

# Check duplicate IDs
train_ids = [r["id"] for r in train_rows]
print(f"Unique train IDs: {len(set(train_ids))}")
dup_train_ids = [k for k, v in Counter(train_ids).items() if v > 1]
print(f"Duplicate IDs: {dup_train_ids}")

# Check system prompts
sys_prompts_train = Counter(r["messages"][0]["content"] for r in train_rows)
print(f"Train system prompts: {sys_prompts_train}")
sys_prompts_dev = Counter(r["messages"][0]["content"] for r in dev_rows)
print(f"Dev system prompts: {sys_prompts_dev}")
sys_prompts_test = Counter(r["messages"][0]["content"] for r in test_rows)
print(f"Test system prompts: {sys_prompts_test}")

# Check number of messages
print("Train msg counts:", Counter(len(r["messages"]) for r in train_rows))
print("Dev msg counts:", Counter(len(r["messages"]) for r in dev_rows))
print("Test msg counts:", Counter(len(r["messages"]) for r in test_rows))


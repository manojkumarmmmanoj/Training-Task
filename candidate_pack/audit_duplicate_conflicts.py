import json
from collections import Counter

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

user_msgs = [r["messages"][1]["content"].strip() for r in train_rows]
dup_msgs = [msg for msg, count in Counter(user_msgs).items() if count > 1]

conflicting_dups = []
identical_dups = []

for msg in dup_msgs:
    matching = [r for r in train_rows if r["messages"][1]["content"].strip() == msg]
    labels = [r["messages"][-1]["content"] for r in matching]
    if len(set(labels)) > 1:
        conflicting_dups.append((matching, labels))
    else:
        identical_dups.append(matching)

print(f"Identical duplicate pairs: {len(identical_dups)}")
print(f"Conflicting duplicate pairs: {len(conflicting_dups)}")

for matching, labels in conflicting_dups:
    print(f"\nConflict on msg: {matching[0]['messages'][1]['content'][:80]}...")
    for r in matching:
        print(f"  ID: {r['id']} | Label: {r['messages'][-1]['content']}")


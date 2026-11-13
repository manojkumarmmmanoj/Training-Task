import json
from collections import Counter

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

user_msgs = [r["messages"][1]["content"].strip() for r in train_rows]
print(f"Total train tickets: {len(user_msgs)}")
print(f"Unique train tickets: {len(set(user_msgs))}")

dup_msgs = [msg for msg, count in Counter(user_msgs).items() if count > 1]
print(f"Duplicate ticket contents count: {len(dup_msgs)}")

# Let's inspect the duplicate tickets and their labels!
for msg in dup_msgs[:10]:
    matching_rows = [r for r in train_rows if r["messages"][1]["content"].strip() == msg]
    print(f"\n--- Duplicated message ({len(matching_rows)} times) ---")
    print(f"Snippet: {msg[:100]}...")
    for r in matching_rows:
        print(f"  ID: {r['id']} | Label: {r['messages'][-1]['content']}")


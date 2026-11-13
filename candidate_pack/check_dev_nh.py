import json
import re

dev_rows = [json.loads(line) for line in open("candidate_pack/data/dev.jsonl", encoding="utf-8")]

print("=== CHECKING DEV ROWS FOR NEEDS_HUMAN ===")
dev_nh_true_non_p1 = []
for r in dev_rows:
    c = json.loads(r["messages"][-1]["content"])
    if c["priority"] != "P1" and c["needs_human"] is True:
        dev_nh_true_non_p1.append((r["id"], c["category"], c["priority"], r["messages"][1]["content"]))

print(f"Dev non-P1 tickets with needs_human=True: {len(dev_nh_true_non_p1)}")
for d in dev_nh_true_non_p1:
    print(f"ID: {d[0]} | cat: {d[1]} | pri: {d[2]}")
    lines = [l.strip() for l in d[3].split('\n') if l.strip() and not l.startswith("Received:") and not l.startswith("Source:") and not l.startswith("Subject:")]
    print(f"   Body: {' '.join(lines)}")


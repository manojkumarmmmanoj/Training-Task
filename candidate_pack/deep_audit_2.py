import json
from collections import Counter

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

prio_rows = []
for r in train_rows:
    c = r["messages"][-1]["content"]
    try:
        data = json.loads(c)
        if "prio" in data:
            prio_rows.append((r["id"], data, r["messages"][1]["content"]))
    except:
        pass

print(f"Total prio rows: {len(prio_rows)}")
prio_vals = Counter(d["prio"] for _, d, _ in prio_rows)
print(f"prio values: {prio_vals}")

# Also check dates in prio rows
date_formats = Counter()
for _, d, _ in prio_rows:
    if d.get("txn_date"):
        if "/" in d["txn_date"]:
            date_formats["slash"] += 1
        elif "-" in d["txn_date"]:
            date_formats["dash"] += 1
print(f"Date formats in prio rows: {date_formats}")


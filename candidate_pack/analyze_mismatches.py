import json
import re
from datetime import datetime, timedelta
from collections import Counter

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

dev_rows = load("candidate_pack/data/dev.jsonl")

# Let's inspect where our rule matcher differed from dev gold labels!
with open("candidate_pack/dev_rule_preds.jsonl", encoding="utf-8") as f:
    preds = {p["id"]: json.loads(p["output"]) for p in [json.loads(line) for line in f]}

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]

diffs = []
for r in dev_rows:
    gold = json.loads(r["messages"][-1]["content"])
    pred = preds.get(r["id"])
    mismatches = {}
    for k in KEYS:
        if gold[k] != pred.get(k):
            mismatches[k] = {"gold": gold[k], "pred": pred.get(k)}
    if mismatches:
        diffs.append((r["id"], mismatches, r["messages"][1]["content"]))

print(f"Total dev rows with at least one mismatch: {len(diffs)} / {len(dev_rows)}")
# What fields mismatched most?
field_mismatch_counts = Counter()
for _, m, _ in diffs:
    for k in m:
        field_mismatch_counts[k] += 1

print("Field mismatch counts on dev:")
for k, c in field_mismatch_counts.most_common():
    print(f"  {k}: {c}")

print("\nSample 5 category mismatches:")
for rid, m, text in [d for d in diffs if "category" in d[1]][:5]:
    print(f"ID: {rid} | Gold: {m['category']['gold']} | Pred: {m['category']['pred']}")
    print(f"Text: {text.strip()[:140]}...\n")

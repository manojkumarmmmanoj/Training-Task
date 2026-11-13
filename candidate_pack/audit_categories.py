import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

fraud_keywords = [
    "didn't make", "did not make", "unfamiliar", "fraud", "unrecognised", "unrecognized",
    "unauthorized", "unauthorised", "scam", "maine nahi kiya", "kisi aur ne", "chori",
    "stolen", "didn't do", "did not do", "not my transaction", "not me", "haven't made",
    "never made", "without my permission", "without my consent"
]

print("=== CHECKING CATEGORY ANOMALIES ===")
potential_fraud_misclassified = []
false_fraud = []

for r in train_rows:
    body = r["messages"][1]["content"].lower()
    c = r["messages"][-1]["content"]
    try:
        data = json.loads(c)
    except:
        continue
    cat = data.get("category")
    
    # Check if text strongly indicates fraud
    has_fraud_kw = any(kw in body for kw in fraud_keywords)
    
    if has_fraud_kw and cat != "fraud":
        # Filter out if it's a false match like "how to report fraud" or "is this fraud"
        potential_fraud_misclassified.append((r["id"], cat, r["messages"][1]["content"]))
        
    if cat == "fraud" and not has_fraud_kw:
        false_fraud.append((r["id"], r["messages"][1]["content"]))

print(f"Tickets with fraud keywords but cat != fraud: {len(potential_fraud_misclassified)}")
for m in potential_fraud_misclassified:
    print(f"ID: {m[0]} | Cat: {m[1]}")
    lines = [l for l in m[2].split('\n') if l.strip() and not l.startswith("Received:") and not l.startswith("Source:")]
    print(f"  Body: {' '.join(lines)[:140]}")

print(f"\nTickets labelled fraud without standard keywords: {len(false_fraud)}")
for f in false_fraud:
    print(f"ID: {f[0]}")
    lines = [l for l in f[1].split('\n') if l.strip() and not l.startswith("Received:") and not l.startswith("Source:")]
    print(f"  Body: {' '.join(lines)[:140]}")


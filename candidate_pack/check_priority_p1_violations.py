import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]
legal_re = re.compile(r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.IGNORECASE)

print("=== 1. FRAUD BUT NOT P1 ===")
for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    if c.get("category") == "fraud" and c.get("priority") != "P1":
        print(f"ID: {r['id']} | pri: {c.get('priority')} | amt: {c.get('amount_paise')}")
        print("  Msg:", r["messages"][1]["content"][:120].strip())

print("\n=== 2. LARGE AMOUNT (>=50k) BUT NOT P1 ===")
for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    amt = c.get("amount_paise")
    if amt is not None and amt >= 5000000 and c.get("priority") != "P1":
        print(f"ID: {r['id']} | pri: {c.get('priority')} | amt: {amt}")
        print("  Msg:", r["messages"][1]["content"][:120].strip())

print("\n=== 3. LEGAL THREAT BUT NOT P1 ===")
for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    msg = r["messages"][1]["content"]
    if bool(legal_re.search(msg)) and c.get("priority") != "P1":
        print(f"ID: {r['id']} | pri: {c.get('priority')} | cat: {c.get('category')}")
        print("  Msg:", msg[:120].strip())


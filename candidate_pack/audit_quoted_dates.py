import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING FOR QUOTED EMAIL DATES (LINES STARTING WITH '>') ===")
quoted_date_errors = []

for r in train_rows:
    body = r["messages"][1]["content"]
    c = r["messages"][-1]["content"]
    try:
        data = json.loads(c)
    except:
        continue
    dt = data.get("txn_date")
    if dt is None:
        continue
        
    lines = body.split("\n")
    quoted_lines = [l for l in lines if l.startswith(">")]
    non_quoted_lines = [l for l in lines if not l.startswith(">")]
    
    # If there are quoted lines, check if dt appears ONLY in quoted lines or is derived from quoted lines
    if quoted_lines:
        quoted_text = "\n".join(quoted_lines)
        non_quoted_text = "\n".join(non_quoted_lines)
        quoted_date_errors.append((r["id"], dt, quoted_text, non_quoted_text))

print(f"Tickets with quoted lines: {len(quoted_date_errors)}")
for q in quoted_date_errors[:10]:
    print(f"ID: {q[0]} | txn_date: {q[1]}")
    print(f"  Quoted text: {q[2]}")
    print(f"  Non-quoted text: {q[3][:100]}...")


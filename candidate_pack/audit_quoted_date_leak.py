import json

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING IF TXN_DATE COMES ONLY FROM QUOTED LINES ===")
quoted_date_leaks = []

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
    
    if quoted_lines:
        non_quoted_text = "\n".join(non_quoted_lines)
        # Check if non_quoted_text mentions any date or relative date
        # (e.g. yesterday, ago, kal, parso, or any month or digits)
        quoted_date_leaks.append((r["id"], dt, non_quoted_text, "\n".join(quoted_lines[:3])))

print(f"Total rows with quoted lines and non-null txn_date: {len(quoted_date_leaks)}")
for q in quoted_date_leaks:
    # Let's inspect the non-quoted text
    print(f"ID: {q[0]} | txn_date: {q[1]}")
    body_lines = [l for l in q[2].split('\n') if l.strip() and not l.startswith("Received:") and not l.startswith("Source:") and not l.startswith("Subject:")]
    print(f"  Body: {' '.join(body_lines)}")


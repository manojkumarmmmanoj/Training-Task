import json

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING P2 ELIGIBLE ROWS LABELED P3 ===")
for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    pri = c.get("priority")
    cat = c.get("category")
    amt = c.get("amount_paise")
    if pri == "P3":
        if (cat in ("payment_failed", "refund") and amt is not None and amt >= 500000 and amt < 5000000) or (cat == "account_access"):
            print(f"ID: {r['id']} | cat: {cat} | amt: {amt} | nh: {c.get('needs_human')}")
            lines = [l for l in r['messages'][1]['content'].split('\n') if l.strip() and not l.startswith("Received:") and not l.startswith("Source:")]
            print(f"   Body: {' '.join(lines)[:100]}")


import json

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

for rid, d, msg in prio_rows[:10]:
    print(f"ID: {rid} | prio: {d['prio']} | category: {d['category']} | amount: {d['amount_paise']} | date: {d['txn_date']}")
    print(f"Ticket: {msg.strip()[:150]}...")
    print("-" * 50)


import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

txid_rows = []
for r in train_rows:
    c = r["messages"][-1]["content"]
    try:
        data = json.loads(c)
        txid = data.get("txn_id")
        if txid is not None and not re.match(r"^NP\d{10}$", txid):
            txid_rows.append((r["id"], txid, r["messages"][1]["content"]))
    except:
        pass

print(f"Total invalid txid rows: {len(txid_rows)}")
for rid, txid, msg in txid_rows:
    print(f"{rid}: txid='{txid}'")
    m = re.search(r"np\s*[- ]?\s*(\d{10})", msg, re.IGNORECASE)
    print(f"   In ticket: {m.group(0) if m else 'NOT FOUND'}")


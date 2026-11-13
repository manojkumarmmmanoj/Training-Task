import json
import re
import csv
from datetime import datetime
from collections import Counter

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
ALLOWED_CATEGORIES = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
ALLOWED_PRIORITIES = {"P1", "P2", "P3"}
ALLOWED_CHANNELS = {"upi", "card", "netbanking", "wallet", None}
ALLOWED_LANGUAGES = {"en", "hi-en"}

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

# 1. Truncated rows (25)
truncated_ids = set()
for r in train_rows:
    try:
        json.loads(r["messages"][-1]["content"])
    except:
        truncated_ids.add(r["id"])

# 2. Gibberish rows (10)
gibberish_ids = set()
for r in train_rows:
    user_msg = r["messages"][1]["content"]
    lines = [l.strip() for l in user_msg.split("\n") if l.strip() and not l.startswith("Received:") and not l.startswith("Source:")]
    text = " ".join(lines).lower()
    if text in ["test", "test test 123", "asdf asdf", "asdf", "hello", "foo bar", "testing"] or len(text) < 15:
        gibberish_ids.add(r["id"])

# Known Rupee scale errors (14)
rupee_fix_map = {
    "tr-00144": 16700,
    "tr-00167": 26700,
    "tr-00187": 22000000,
    "tr-00199": 67400,
    "tr-00222": 59450,
    "tr-00328": 66750,
    "tr-00407": 35900,
    "tr-00560": 110000,
    "tr-00622": 370200,
    "tr-00688": 20900,
    "tr-00691": 67800,
    "tr-00795": 3430000,
    "tr-00920": 85300,
    "tr-01311": 49799,
}

# 30 erroneous P1 rows with needs_human=False that should be P3
erroneous_p1_ids = {
    'tr-00018', 'tr-00023', 'tr-00059', 'tr-00146', 'tr-00225', 'tr-00306',
    'tr-00309', 'tr-00312', 'tr-00380', 'tr-00413', 'tr-00452', 'tr-00485',
    'tr-00503', 'tr-00505', 'tr-00508', 'tr-00523', 'tr-00533', 'tr-00689',
    'tr-00717', 'tr-00745', 'tr-00757', 'tr-00782', 'tr-00802', 'tr-00879',
    'tr-00933', 'tr-00967', 'tr-01048', 'tr-01146', 'tr-01222', 'tr-01407'
}

# 15 severe rows (fraud, >=50k, legal) mislabeled as P3 that should be P1
mislabeled_p1_ids = {
    'tr-00045', 'tr-00638', 'tr-01019', 'tr-01370', # fraud
    'tr-00212', 'tr-00427', 'tr-01023', 'tr-01319', # amount >= 50k
    'tr-00184', 'tr-00424', 'tr-00852', 'tr-00890', 'tr-01017', 'tr-01086', 'tr-01403' # legal threats
}

# Eligible P2 rows mislabeled as P3 (amount >= 5k or account_access)
mislabeled_p2_ids = {
    'tr-00129', 'tr-00154', 'tr-00220', 'tr-00294', 'tr-00355',
    'tr-00567', 'tr-00648', 'tr-00706', 'tr-00707', 'tr-00746',
    'tr-00932', 'tr-01109', 'tr-01131', 'tr-01305'
}

cleaning_records = {} # rid -> {"action": "...", "reasons": []}

for rid in sorted(truncated_ids):
    cleaning_records[rid] = {"action": "drop", "reasons": ["truncated assistant JSON output / incomplete label"]}

for rid in sorted(gibberish_ids):
    cleaning_records[rid] = {"action": "drop", "reasons": ["synthetic test placeholder or gibberish input with hallucinated label"]}

cleaned_train_rows = []

for r in train_rows:
    rid = r["id"]
    if rid in cleaning_records and cleaning_records[rid]["action"] == "drop":
        continue
        
    c = r["messages"][-1]["content"]
    data = json.loads(c)
    reasons = []
    
    # 1. Check legacy prio key
    if "prio" in data:
        p_val = data.pop("prio")
        p_map = {"high": "P1", "medium": "P2", "low": "P3"}
        data["priority"] = p_map.get(p_val, p_val)
        reasons.append(f"migrated legacy key 'prio' ({p_val}) to 'priority' ({data['priority']})")
        
    # 2. Check date format DD/MM/YYYY
    if data.get("txn_date") and "/" in data["txn_date"]:
        old_dt = data["txn_date"]
        d_obj = datetime.strptime(old_dt, "%d/%m/%Y")
        data["txn_date"] = d_obj.strftime("%Y-%m-%d")
        reasons.append(f"converted txn_date from {old_dt} to ISO {data['txn_date']}")
        
    # 3. Check unnormalized txn_id
    if data.get("txn_id") and not re.match(r"^NP\d{10}$", data["txn_id"]):
        old_txid = data["txn_id"]
        data["txn_id"] = re.sub(r"[^A-Za-z0-9]", "", old_txid).upper()
        assert re.match(r"^NP\d{10}$", data["txn_id"]), f"Failed to normalize {old_txid}"
        reasons.append(f"normalized txn_id from {old_txid} to {data['txn_id']}")
        
    # 4. Check rupee scale errors
    if rid in rupee_fix_map:
        old_amt = data["amount_paise"]
        data["amount_paise"] = rupee_fix_map[rid]
        reasons.append(f"corrected rupee-to-paise scale error from {old_amt} to {data['amount_paise']}")
        
    # 5. Check erroneous P1 rows
    if rid in erroneous_p1_ids:
        data["priority"] = "P3"
        reasons.append("fixed priority from P1 to P3 (low severity, amount < 5k, not fraud, no legal threat)")
        
    # 6. Check mislabeled P1 rows
    if rid in mislabeled_p1_ids:
        data["priority"] = "P1"
        data["needs_human"] = True
        reasons.append("fixed priority from P3 to P1 (qualifies for P1 under schema rules: fraud/amount>=50k/legal threat)")
        
    # 7. Check mislabeled P2 rows
    if rid in mislabeled_p2_ids:
        data["priority"] = "P2"
        reasons.append("fixed priority from P3 to P2 (qualifies for P2 under schema rules: amount>=5k or account_access)")

    # Enforce exact key ordering according to schema
    cleaned_obj = {k: data[k] for k in KEYS}
    
    if reasons:
        cleaning_records[rid] = {"action": "fix", "reasons": reasons}
        
    # New row
    new_r = {
        "id": rid,
        "messages": [
            r["messages"][0],
            r["messages"][1],
            {"role": "assistant", "content": json.dumps(cleaned_obj)}
        ]
    }
    cleaned_train_rows.append(new_r)

print(f"Original train rows: {len(train_rows)}")
print(f"Cleaned train rows: {len(cleaned_train_rows)}")
print(f"Total dropped: {len([r for r in cleaning_records.values() if r['action'] == 'drop'])}")
print(f"Total fixed: {len([r for r in cleaning_records.values() if r['action'] == 'fix'])}")
print(f"Total logged in cleaning_log: {len(cleaning_records)}")

# Write cleaning_log.csv
with open("candidate_pack/cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "action", "reason"])
    for rid in sorted(cleaning_records):
        rec = cleaning_records[rid]
        writer.writerow([rid, rec["action"], "; ".join(rec["reasons"])])

print("Wrote candidate_pack/cleaning_log.csv")

# Write cleaned train data
with open("candidate_pack/data/train_cleaned.jsonl", "w", encoding="utf-8") as f:
    for r in cleaned_train_rows:
        f.write(json.dumps(r) + "\n")

print("Wrote candidate_pack/data/train_cleaned.jsonl")


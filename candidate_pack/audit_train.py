import json
import re
from collections import Counter

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
ALLOWED_CATEGORIES = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
ALLOWED_PRIORITIES = {"P1", "P2", "P3"}
ALLOWED_CHANNELS = {"upi", "card", "netbanking", "wallet", None}
ALLOWED_LANGUAGES = {"en", "hi-en"}

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

issues = []

for r in train_rows:
    row_id = r["id"]
    content = r["messages"][-1]["content"]
    
    # 1. JSON parsing
    try:
        data = json.loads(content)
    except Exception as e:
        issues.append((row_id, "json_parse_error", str(e), content))
        continue
        
    if not isinstance(data, dict):
        issues.append((row_id, "not_dict", type(data).__name__, content))
        continue
        
    # 2. Keys check
    if list(data.keys()) != KEYS:
        issues.append((row_id, "keys_mismatch", f"Keys: {list(data.keys())}", content))
        
    # 3. Type & allowed values
    for k in ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]:
        if k not in data:
            issues.append((row_id, f"missing_key_{k}", None, content))
            
    cat = data.get("category")
    pri = data.get("priority")
    amt = data.get("amount_paise")
    dt = data.get("txn_date")
    txid = data.get("txn_id")
    chn = data.get("channel")
    lang = data.get("language")
    nh = data.get("needs_human")
    
    if cat not in ALLOWED_CATEGORIES:
        issues.append((row_id, "invalid_category", cat, content))
        
    if pri not in ALLOWED_PRIORITIES:
        issues.append((row_id, "invalid_priority", pri, content))
        
    if amt is not None:
        if not isinstance(amt, int) or isinstance(amt, bool):
            issues.append((row_id, "invalid_amount_type", f"{amt} ({type(amt).__name__})", content))
        elif amt < 0:
            issues.append((row_id, "negative_amount", amt, content))
            
    if dt is not None:
        if not isinstance(dt, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", dt):
            issues.append((row_id, "invalid_date_format", dt, content))
            
    if txid is not None:
        if not isinstance(txid, str) or not re.match(r"^NP\d{10}$", txid):
            issues.append((row_id, "invalid_txid_format", txid, content))
            
    if chn not in ALLOWED_CHANNELS:
        issues.append((row_id, "invalid_channel", chn, content))
        
    if lang not in ALLOWED_LANGUAGES:
        issues.append((row_id, "invalid_language", lang, content))
        
    if not isinstance(nh, bool):
        issues.append((row_id, "invalid_needs_human_type", f"{nh} ({type(nh).__name__})", content))
        
    # 4. Transaction fields for non-transaction categories
    if cat in {"account_access", "kyc", "other"}:
        non_null_fields = []
        if amt is not None: non_null_fields.append(f"amount_paise={amt}")
        if dt is not None: non_null_fields.append(f"txn_date={dt}")
        if txid is not None: non_null_fields.append(f"txn_id={txid}")
        if chn is not None: non_null_fields.append(f"channel={chn}")
        if non_null_fields:
            issues.append((row_id, "non_transaction_category_has_txn_fields", f"{cat}: {', '.join(non_null_fields)}", content))

print(f"Total basic structural/schema issues found: {len(issues)}")
issue_types = Counter(x[1] for x in issues)
for it, c in issue_types.items():
    print(f"  {it}: {c}")

print("\nSample issues:")
for x in issues[:25]:
    print(x)

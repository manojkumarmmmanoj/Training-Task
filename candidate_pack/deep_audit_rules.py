import json
import re
from datetime import datetime, timedelta

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_rows = load("candidate_pack/data/train.jsonl")

# Let's inspect priority vs schema rules:
# Priority rule:
# 1. P1 if:
#    - category == 'fraud'
#    - amount_paise >= 5000000 (disputed amount >= 50,000)
#    - customer makes legal threat (RBI, ombudsman, consumer court, lawyer, legal notice, legal action)
# 2. P2 if not P1 and:
#    - category in ('payment_failed', 'refund') and amount_paise >= 500000
#    - category == 'account_access'
# 3. P3 otherwise.

# Also needs_human rule:
# true if priority is P1 OR customer says contacted support before ("third time", "again and again", "baar baar", "pehle bhi", etc.)

legal_threat_pattern = re.compile(r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.IGNORECASE)

print("=== CHECKING PRIORITY & NEEDS_HUMAN CONSISTENCY IN TRAIN ===")
priority_anomalies = []
needs_human_anomalies = []

for r in train_rows:
    row_id = r["id"]
    content = r["messages"][-1]["content"]
    user_msg = r["messages"][1]["content"]
    try:
        data = json.loads(content)
    except:
        continue
    
    # Map prio to priority if present for this analysis
    if "prio" in data:
        p_map = {"high": "P1", "medium": "P2", "low": "P3"}
        pri = p_map.get(data["prio"])
    else:
        pri = data.get("priority")
        
    cat = data.get("category")
    amt = data.get("amount_paise")
    nh = data.get("needs_human")
    
    # Check legal threat
    has_legal = bool(legal_threat_pattern.search(user_msg))
    is_fraud = (cat == "fraud")
    is_large = (amt is not None and amt >= 5000000)
    
    # Expected priority:
    if is_fraud or is_large or has_legal:
        expected_pri = "P1"
    elif (cat in ("payment_failed", "refund") and amt is not None and amt >= 500000) or (cat == "account_access"):
        expected_pri = "P2"
    else:
        expected_pri = "P3"
        
    if pri != expected_pri:
        priority_anomalies.append({
            "id": row_id,
            "actual_pri": pri,
            "expected_pri": expected_pri,
            "cat": cat,
            "amt": amt,
            "has_legal": has_legal,
            "ticket": user_msg
        })
        
    # Check needs_human:
    # If priority is P1, needs_human MUST be True!
    if pri == "P1" and nh is not True:
        needs_human_anomalies.append({
            "id": row_id,
            "issue": "P1_but_needs_human_false",
            "pri": pri,
            "nh": nh,
            "ticket": user_msg
        })

print(f"Total priority anomalies against strict rule: {len(priority_anomalies)}")
print(f"Total P1 but needs_human False: {len(needs_human_anomalies)}")

for a in priority_anomalies[:15]:
    print(f"ID: {a['id']} | Actual: {a['actual_pri']} | Expected: {a['expected_pri']} | Cat: {a['cat']} | Amt: {a['amt']} | Legal: {a['has_legal']}")
    first_line = a['ticket'].split('\n')[0]
    body = [line for line in a['ticket'].split('\n') if line and not line.startswith("Received:") and not line.startswith("Source:")][:2]
    print(f"   Body: {' '.join(body)[:120]}")


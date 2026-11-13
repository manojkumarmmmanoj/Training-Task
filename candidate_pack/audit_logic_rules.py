import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

# Regex for legal threat
legal_re = re.compile(r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.IGNORECASE)

# Regex for repeated contact
repeated_re = re.compile(r"\b(again and again|third time|fourth time|multiple times|several times|baar baar|pehle bhi|already contacted|already written|already raised|keep raising|contacting you again|pichla ticket|earlier ticket)\b", re.IGNORECASE)

# Let's inspect all rows (excluding truncated and gibberish)
p1_nh_false = []
fraud_not_p1 = []
large_amt_not_p1 = []
legal_not_p1 = []
p2_eligible_is_p3 = []
repeated_nh_false = []
p3_no_rep_is_nh_true = []

for r in train_rows:
    row_id = r["id"]
    content = r["messages"][-1]["content"]
    user_msg = r["messages"][1]["content"]
    
    try:
        data = json.loads(content)
    except:
        continue
        
    lines = [l.strip() for l in user_msg.split("\n") if l.strip() and not l.startswith("Received:") and not l.startswith("Source:")]
    text = " ".join(lines).lower()
    if text in ["test", "test test 123", "asdf asdf", "asdf", "hello", "foo bar", "testing"] or len(text) < 15:
        continue
        
    pri = data.get("priority")
    if "prio" in data:
        pri = {"high": "P1", "medium": "P2", "low": "P3"}.get(data["prio"])
        
    cat = data.get("category")
    amt = data.get("amount_paise")
    nh = data.get("needs_human")
    
    # 1. P1 but needs_human is False
    if pri == "P1" and nh is False:
        p1_nh_false.append((row_id, pri, nh, cat, amt, user_msg))
        
    # 2. Fraud but priority not P1
    if cat == "fraud" and pri != "P1":
        fraud_not_p1.append((row_id, pri, cat, amt, user_msg))
        
    # 3. Disputed amount >= 50,000 (5,000,000 paise) but not P1
    if amt is not None and amt >= 5000000 and pri != "P1":
        large_amt_not_p1.append((row_id, pri, cat, amt, user_msg))
        
    # 4. Legal threat but not P1
    if bool(legal_re.search(user_msg)) and pri != "P1":
        legal_not_p1.append((row_id, pri, cat, amt, user_msg))
        
    # 5. Should be P2 but is P3
    if pri == "P3":
        if (cat in ("payment_failed", "refund") and amt is not None and amt >= 500000) or (cat == "account_access"):
            p2_eligible_is_p3.append((row_id, pri, cat, amt, user_msg))
            
    # 6. Repeated contact but needs_human is False
    if nh is False and bool(repeated_re.search(user_msg)):
        repeated_nh_false.append((row_id, pri, nh, user_msg))

print(f"1. P1 but needs_human False: {len(p1_nh_false)}")
print(f"2. Fraud but priority not P1: {len(fraud_not_p1)}")
print(f"3. Large amount (>=50k) but not P1: {len(large_amt_not_p1)}")
print(f"4. Legal threat but not P1: {len(legal_not_p1)}")
print(f"5. Should be P2 but is P3: {len(p2_eligible_is_p3)}")
print(f"6. Repeated contact but needs_human False: {len(repeated_nh_false)}")


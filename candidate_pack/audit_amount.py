import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING AMOUNT_PAISE ANOMALIES ===")
amount_issues = []

for r in train_rows:
    row_id = r["id"]
    content = r["messages"][-1]["content"]
    user_msg = r["messages"][1]["content"]
    try:
        data = json.loads(content)
    except:
        continue
        
    cat = data.get("category")
    amt = data.get("amount_paise")
    
    # Non-transaction categories MUST have amount_paise = None
    if cat in ("account_access", "kyc", "other"):
        if amt is not None:
            amount_issues.append((row_id, "non_txn_cat_has_amount", amt, user_msg))
            
    # If transaction category, check if amount matches something in the text
    # Look for account balance distractors
    bal_match = re.search(r"(?:account\s*balance|balance\s*is|balance\s*:|balance\s*of)\s*(?:is|rs\.?|inr|₹)?\s*([0-9,]+(?:\.[0-9]+)?)", user_msg, re.IGNORECASE)
    if bal_match:
        bal_str = bal_match.group(1).replace(",", "")
        try:
            bal_paise = int(round(float(bal_str) * 100))
            if amt == bal_paise:
                amount_issues.append((row_id, "balance_used_as_amount", amt, user_msg))
        except:
            pass

print(f"Total amount issues found: {len(amount_issues)}")
for issue in amount_issues[:15]:
    print(issue[0], issue[1], issue[2])
    print("  Ticket:", issue[3].strip()[:140])


import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING FOR RUPEES VS PAISE CONVERSION ERRORS (MISSED *100) ===")
rupee_errors = []

for r in train_rows:
    body = r["messages"][1]["content"]
    c = r["messages"][-1]["content"]
    try:
        data = json.loads(c)
    except:
        continue
    amt = data.get("amount_paise")
    if amt is None:
        continue
        
    # Check for 'k' in body
    # e.g. 1.1k -> 1100 rupees -> should be 110000 paise, but was 1100
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", body, re.IGNORECASE)
    if k_match:
        val = float(k_match.group(1)) * 1000
        # If amt == val (rupees without *100)
        if int(val) == amt:
            rupee_errors.append((r["id"], "k_missed_100", amt, int(val * 100), body))
            
    # Check for 'lakh' in body
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*lakh\b", body, re.IGNORECASE)
    if lakh_match:
        val = float(lakh_match.group(1)) * 100000
        if int(val) == amt:
            rupee_errors.append((r["id"], "lakh_missed_100", amt, int(val * 100), body))
            
    # Also check plain rupees: what if rupees was e.g. 1500 and amt was 1500?
    plain_rs_match = re.findall(r"(?:rs\.?|inr|₹)\s*(\d+(?:\.\d+)?)\b", body, re.IGNORECASE)
    for m in plain_rs_match:
        try:
            rs_val = float(m)
            if rs_val > 100 and int(rs_val) == amt:
                rupee_errors.append((r["id"], "rupees_missed_100", amt, int(rs_val * 100), body))
        except:
            pass

print(f"Total rupee conversion errors: {len(rupee_errors)}")
for err in rupee_errors:
    print(f"ID: {err[0]} | Type: {err[1]} | Current amt: {err[2]} | Should be: {err[3]}")


import json
import re
from datetime import datetime, timedelta

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING RELATIVE DATE CALCULATIONS ===")
date_calc_errors = []

for r in train_rows:
    body = r["messages"][1]["content"]
    c = r["messages"][-1]["content"]
    try:
        data = json.loads(c)
    except:
        continue
    dt = data.get("txn_date")
    
    # Extract Received date: e.g. "Received: 2026-08-14 (Fri)"
    m_rec = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", body)
    if not m_rec:
        continue
    rec_date = datetime.strptime(m_rec.group(1), "%Y-%m-%d").date()
    
    # Check relative expressions:
    # 1. "today" / "aaj" -> rec_date
    # 2. "yesterday" / "kal" -> can be rec_date - 1 day
    # 3. "day before yesterday" / "parso" -> rec_date - 2 days
    # 4. "N days ago" / "N din pehle" -> rec_date - N days
    
    # Let's check "N days ago" / "N din pehle"
    m_days = re.search(r"(\d+)\s*(?:days?\s*ago|din\s*pehle)", body, re.IGNORECASE)
    if m_days:
        n_days = int(m_days.group(1))
        expected_date = (rec_date - timedelta(days=n_days)).strftime("%Y-%m-%d")
        if dt is not None and dt != expected_date:
            # Check if there is an explicit date elsewhere in the body
            date_calc_errors.append((r["id"], "n_days_ago_mismatch", n_days, dt, expected_date, body))
            
    m_dby = re.search(r"\b(day before yesterday|parso)\b", body, re.IGNORECASE)
    if m_dby and not m_days:
        expected_date = (rec_date - timedelta(days=2)).strftime("%Y-%m-%d")
        if dt is not None and dt != expected_date:
            date_calc_errors.append((r["id"], "day_before_yesterday_mismatch", 2, dt, expected_date, body))
            
    m_yest = re.search(r"\b(yesterday)\b", body, re.IGNORECASE)
    if m_yest and not m_dby and not m_days:
        expected_date = (rec_date - timedelta(days=1)).strftime("%Y-%m-%d")
        if dt is not None and dt != expected_date:
            date_calc_errors.append((r["id"], "yesterday_mismatch", 1, dt, expected_date, body))

print(f"Total date calculation mismatches: {len(date_calc_errors)}")
for err in date_calc_errors:
    print(f"ID: {err[0]} | Type: {err[1]} | Label dt: {err[3]} | Expected from rel: {err[4]}")
    lines = [l for l in err[5].split('\n') if l and not l.startswith("Received:") and not l.startswith("Source:")]
    print(f"  Body: {' '.join(lines)[:120]}")


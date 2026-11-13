import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING ALL AMOUNT_PAISE IN TRAIN ===")
unmatched_amounts = []

for r in train_rows:
    row_id = r["id"]
    content = r["messages"][-1]["content"]
    user_msg = r["messages"][1]["content"]
    try:
        data = json.loads(content)
    except:
        continue
        
    amt = data.get("amount_paise")
    if amt is None:
        continue
        
    # Check if this paise amount can be derived from the text
    # e.g. amt / 100 in rupees
    rupees = amt / 100.0
    # Search for rupees in user_msg
    # Examples: 4700 -> 4.7k, 4700, 4,700, etc.
    # Let's search if digits of amt appear in msg
    rupees_int = int(rupees) if rupees.is_integer() else None
    
    # Check strings
    found = False
    clean_msg = user_msg.replace(",", "")
    if str(amt) in clean_msg:
        found = True
    elif rupees_int and str(rupees_int) in clean_msg:
        found = True
    elif f"{rupees:.2f}" in clean_msg or f"{rupees:.1f}" in clean_msg:
        found = True
    elif f"{rupees/1000:.1f}k" in clean_msg.lower() or f"{rupees/1000:.2f}k" in clean_msg.lower() or f"{int(rupees/1000)}k" in clean_msg.lower():
        found = True
    elif f"{rupees/100000:.1f} lakh" in clean_msg.lower() or f"{rupees/100000:.2f} lakh" in clean_msg.lower() or f"{int(rupees/100000)} lakh" in clean_msg.lower():
        found = True
        
    if not found:
        unmatched_amounts.append((row_id, amt, rupees, user_msg))

print(f"Unmatched amounts count: {len(unmatched_amounts)}")
for u in unmatched_amounts[:15]:
    print(f"ID: {u[0]} | Paise: {u[1]} | Rupees: {u[2]}")
    lines = [l for l in u[3].split('\n') if l and not l.startswith("Received:") and not l.startswith("Source:")]
    print(f"   Msg: {' '.join(lines)[:120]}")


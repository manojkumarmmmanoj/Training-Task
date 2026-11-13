import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

target_ids = {'tr-00018', 'tr-00023', 'tr-00059', 'tr-00146', 'tr-00225', 'tr-00306', 'tr-00309', 'tr-00312', 'tr-00380', 'tr-00413', 'tr-00452', 'tr-00485', 'tr-00503', 'tr-00505', 'tr-00508', 'tr-00523', 'tr-00533', 'tr-00689', 'tr-00717', 'tr-00745', 'tr-00757', 'tr-00782', 'tr-00802', 'tr-00879', 'tr-00933', 'tr-00967', 'tr-01048', 'tr-01146', 'tr-01222', 'tr-01407'}

legal_re = re.compile(r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.IGNORECASE)

for r in train_rows:
    if r["id"] in target_ids:
        msg = r["messages"][1]["content"]
        c = json.loads(r["messages"][-1]["content"])
        amt = c.get("amount_paise")
        has_legal = bool(legal_re.search(msg))
        print(f"ID: {r['id']} | Cat: {c['category']} | Amt: {amt} | Legal: {has_legal}")


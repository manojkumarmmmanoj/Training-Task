import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

support_contact_patterns = [
    r"again and again",
    r"third time",
    r"second complaint",
    r"written \d+ times",
    r"already written",
    r"contacted you before",
    r"contacting you again",
    r"baar baar",
    r"pehle bhi",
    r"already raised",
    r"keep raising this",
    r"teesri baar",
    r"dusri baar"
]
sc_re = re.compile("|".join(support_contact_patterns), re.IGNORECASE)

false_nh_true = []
for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    body = r["messages"][1]["content"]
    nh = c.get("needs_human")
    pri = c.get("priority")
    if "prio" in c:
        pri = {"high": "P1", "medium": "P2", "low": "P3"}.get(c["prio"])
        
    if nh is True and pri != "P1":
        has_contact = bool(sc_re.search(body))
        if not has_contact:
            false_nh_true.append((r["id"], pri, c.get("category"), body))

print(f"Non-P1 tickets with needs_human=True but no standard repeated contact phrase: {len(false_nh_true)}")
for m in false_nh_true[:10]:
    print(f"ID: {m[0]} | pri: {m[1]} | cat: {m[2]}")
    lines = [l.strip() for l in m[3].split('\n') if l.strip() and not l.startswith("Received:") and not l.startswith("Source:") and not l.startswith("Subject:")]
    print(f"   Body: {' '.join(lines)[:100]}")


import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

patterns = [
    r"again and again",
    r"third time",
    r"second complaint",
    r"written \d+ times",
    r"contacted you before",
    r"contacting you again",
    r"baar baar",
    r"pehle bhi",
    r"multiple times",
    r"several times"
]
combined_re = re.compile("|".join(patterns), re.IGNORECASE)

missed_nh = []
for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    body = r["messages"][1]["content"]
    nh = c.get("needs_human")
    if bool(combined_re.search(body)) and nh is False:
        missed_nh.append((r["id"], c.get("priority"), c.get("category"), body))

print(f"Tickets with repeated contact phrases but needs_human=False: {len(missed_nh)}")
for m in missed_nh[:10]:
    print(f"ID: {m[0]} | pri: {m[1]} | cat: {m[2]}")
    lines = [l.strip() for l in m[3].split('\n') if l.strip() and not l.startswith("Received:") and not l.startswith("Source:") and not l.startswith("Subject:")]
    print(f"   Body: {' '.join(lines)[:100]}")


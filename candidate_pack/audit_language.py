import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

hindi_words = {
    "karo", "hai", "hua", "nahi", "tha", "thi", "the", "mera", "meri", "mere",
    "dhanyavaad", "raha", "rahi", "hoon", "paise", "kat", "gaye", "gaya", "gayi",
    "bheje", "chuka", "chuki", "bhi", "pehle", "baar", "kya", "bhai", "namaste",
    "ji", "aap", "dost", "rakam", "saamaan", "lekin", "wapas", "mile", "din",
    "decline", "karun", "pado", "pade", "rupaye", "rupiya", "teesri", "dusri", "pichla"
}

lang_errors = []

for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    body = r["messages"][1]["content"].lower()
    lang = c.get("language")
    
    # Tokenize words
    words = set(re.findall(r"\b[a-z]+\b", body))
    hi_overlap = words & hindi_words
    
    # If substantial Hindi words present (>= 3 words) but marked 'en'
    if len(hi_overlap) >= 3 and lang == "en":
        lang_errors.append((r["id"], "hi_words_in_en", hi_overlap, r["messages"][1]["content"]))
    # If 0 Hindi words present and marked 'hi-en'
    elif len(hi_overlap) == 0 and lang == "hi-en":
        # Check if really English
        lang_errors.append((r["id"], "no_hi_words_in_hien", hi_overlap, r["messages"][1]["content"]))

print(f"Potential language classification issues: {len(lang_errors)}")
for l in lang_errors:
    print(f"ID: {l[0]} | Type: {l[1]} | Words: {l[2]}")
    lines = [line.strip() for line in l[3].split('\n') if line.strip() and not line.startswith("Received:") and not line.startswith("Source:") and not line.startswith("Subject:")]
    print(f"  Body: {' '.join(lines)[:120]}")


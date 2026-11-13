import json
import re

test_rows = [json.loads(line) for line in open("candidate_pack/data/test_inputs.jsonl", encoding="utf-8")]

print(f"Total test rows: {len(test_rows)}")
print(f"First test row ID: {test_rows[0]['id']}, last: {test_rows[-1]['id']}")

# Check sources in test
sources = [re.search(r"Source:\s*([a-z-]+)", r["messages"][1]["content"]).group(1) for r in test_rows]
print("Test sources distribution:")
from collections import Counter
print(Counter(sources))

# Check hinglish vs english in test
hindi_words = {"karo", "hai", "hua", "nahi", "tha", "mera", "meri", "mere", "dhanyavaad", "raha", "rahi", "hoon", "paise", "kat", "gaye", "bheje", "chuka", "bhi", "pehle", "baar", "kya", "bhai", "namaste", "ji", "dost", "rakam", "saamaan", "lekin"}
hi_count = sum(1 for r in test_rows if len(set(re.findall(r"\b[a-z]+\b", r["messages"][1]["content"].lower())) & hindi_words) >= 2)
print(f"Approx Hinglish in test: {hi_count} ({hi_count/len(test_rows):.1%})")


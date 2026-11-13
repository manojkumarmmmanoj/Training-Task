import json
import re

test_rows = [json.loads(line) for line in open("candidate_pack/data/test_inputs.jsonl", encoding="utf-8")]

print(f"Total test rows: {len(test_rows)}")
# Check all rows have system and user messages
for i, r in enumerate(test_rows):
    assert len(r["messages"]) == 2
    assert r["messages"][0]["role"] == "system"
    assert r["messages"][1]["role"] == "user"
    assert r["messages"][0]["content"] == "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."

print("All 400 test inputs have valid system and user messages!")


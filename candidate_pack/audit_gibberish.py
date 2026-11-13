import json

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

print("=== CHECKING FOR GIBBERISH / SYNTHETIC TEST / SHORT TICKETS ===")
gibberish = []
for r in train_rows:
    body = r["messages"][1]["content"]
    # Look at the actual message lines
    lines = [l.strip() for l in body.split("\n") if l.strip() and not l.startswith("Received:") and not l.startswith("Source:")]
    text = " ".join(lines).lower()
    if text in ["test", "test test 123", "asdf asdf", "asdf", "hello", "foo bar", "testing"] or len(text) < 15:
        gibberish.append((r["id"], text, r["messages"][-1]["content"]))

print(f"Total gibberish tickets found: {len(gibberish)}")
for g in gibberish:
    print(f"ID: {g[0]} | Text: '{g[1]}' | Label: {g[2]}")


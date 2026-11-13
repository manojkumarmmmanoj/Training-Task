import json

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]
dev_rows = [json.loads(line) for line in open("candidate_pack/data/dev.jsonl", encoding="utf-8")]

train_map = {r["messages"][1]["content"].strip(): r for r in train_rows}
dev_map = {r["messages"][1]["content"].strip(): r for r in dev_rows}

overlap = set(train_map.keys()) & set(dev_map.keys())
print(f"Overlap count: {len(overlap)}")

for text in list(overlap)[:5]:
    tr = train_map[text]
    dv = dev_map[text]
    print(f"Train ID: {tr['id']} | Dev ID: {dv['id']}")
    print(f"Train content: {tr['messages'][-1]['content']}")
    print(f"Dev content:   {dv['messages'][-1]['content']}")
    print(f"Keys in train row: {list(tr.keys())}")
    print(f"Keys in dev row:   {list(dv.keys())}")
    print("-" * 50)


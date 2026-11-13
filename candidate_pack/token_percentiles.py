import json

train = [json.loads(line) for line in open("candidate_pack/data/train_cleaned.jsonl", encoding="utf-8")]

total_words = [len(r["messages"][1]["content"].split()) + len(r["messages"][2]["content"].split()) + 15 for r in train]
total_words.sort()

n = len(total_words)
p50 = total_words[int(0.50 * n)]
p90 = total_words[int(0.90 * n)]
p95 = total_words[int(0.95 * n)]
p98 = total_words[int(0.98 * n)]
p99 = total_words[int(0.99 * n)]

print(f"50th percentile total words: {p50} (~{int(p50*1.3)} tokens)")
print(f"90th percentile total words: {p90} (~{int(p90*1.3)} tokens)")
print(f"95th percentile total words: {p95} (~{int(p95*1.3)} tokens)")
print(f"98th percentile total words: {p98} (~{int(p98*1.3)} tokens)")
print(f"99th percentile total words: {p99} (~{int(p99*1.3)} tokens)")
print(f"Max total words: {total_words[-1]} (~{int(total_words[-1]*1.3)} tokens)")

over_512 = sum(1 for w in total_words if w * 1.3 > 512)
over_1024 = sum(1 for w in total_words if w * 1.3 > 1024)
print(f"Rows exceeding 512 tokens: {over_512} ({over_512/n:.1%})")
print(f"Rows exceeding 1024 tokens: {over_1024} ({over_1024/n:.1%})")


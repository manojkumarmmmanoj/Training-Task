import json
import re

train_rows = [json.loads(line) for line in open("candidate_pack/data/train.jsonl", encoding="utf-8")]

channel_patterns = {
    "upi": re.compile(r"\bupi\b", re.IGNORECASE),
    "card": re.compile(r"\b(card|credit card|debit card|visa|mastercard)\b", re.IGNORECASE),
    "netbanking": re.compile(r"\b(netbanking|net banking|internet banking)\b", re.IGNORECASE),
    "wallet": re.compile(r"\bwallet\b", re.IGNORECASE),
}

channel_issues = []

for r in train_rows:
    try:
        c = json.loads(r["messages"][-1]["content"])
    except:
        continue
    cat = c.get("category")
    # For non-transaction categories, channel is always null
    if cat in ("account_access", "kyc", "other"):
        continue
        
    chn = c.get("channel")
    body = r["messages"][1]["content"]
    
    # Check if channel in label is in body
    if chn is not None:
        patt = channel_patterns.get(chn)
        if patt and not patt.search(body):
            channel_issues.append((r["id"], "channel_not_in_text", chn, body))
            
    # Check if a channel is in body but label is None
    if chn is None:
        found = []
        for ch, patt in channel_patterns.items():
            if patt.search(body):
                found.append(ch)
        if found:
            # Let's see if it's really about payment channel or card delivery/activation
            channel_issues.append((r["id"], "text_has_channel_but_label_null", found, body))

print(f"Channel issues: {len(channel_issues)}")
not_in_text = [x for x in channel_issues if x[1] == "channel_not_in_text"]
print(f"Channel in label but not in text: {len(not_in_text)}")
for x in not_in_text:
    print(x[0], x[1], x[2])


import json
from collections import Counter

dev_rows = [json.loads(line) for line in open("candidate_pack/data/dev.jsonl", encoding="utf-8")]

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
ALLOWED_CATEGORIES = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
ALLOWED_PRIORITIES = {"P1", "P2", "P3"}
ALLOWED_CHANNELS = {"upi", "card", "netbanking", "wallet", None}
ALLOWED_LANGUAGES = {"en", "hi-en"}

print("Checking dev labels against schema...")
cat_counts = Counter()
priority_counts = Counter()
channel_counts = Counter()
lang_counts = Counter()
needs_human_counts = Counter()

for r in dev_rows:
    asst = json.loads(r["messages"][-1]["content"])
    assert list(asst.keys()) == KEYS, f"Keys mismatch: {list(asst.keys())}"
    cat_counts[asst["category"]] += 1
    priority_counts[asst["priority"]] += 1
    channel_counts[asst["channel"]] += 1
    lang_counts[asst["language"]] += 1
    needs_human_counts[asst["needs_human"]] += 1
    
    # Check types
    assert asst["category"] in ALLOWED_CATEGORIES
    assert asst["priority"] in ALLOWED_PRIORITIES
    assert asst["amount_paise"] is None or (isinstance(asst["amount_paise"], int) and asst["amount_paise"] > 0)
    assert asst["txn_date"] is None or isinstance(asst["txn_date"], str)
    assert asst["txn_id"] is None or isinstance(asst["txn_id"], str)
    assert asst["channel"] in ALLOWED_CHANNELS
    assert asst["language"] in ALLOWED_LANGUAGES
    assert isinstance(asst["needs_human"], bool)
    
    # Check transaction fields for account_access, kyc, other
    if asst["category"] in {"account_access", "kyc", "other"}:
        assert asst["amount_paise"] is None, f"{r['id']} category {asst['category']} has amount_paise {asst['amount_paise']}"
        assert asst["txn_date"] is None, f"{r['id']} category {asst['category']} has txn_date {asst['txn_date']}"
        assert asst["txn_id"] is None, f"{r['id']} category {asst['category']} has txn_id {asst['txn_id']}"
        assert asst["channel"] is None, f"{r['id']} category {asst['category']} has channel {asst['channel']}"
        
    # Check needs_human rule for P1:
    if asst["priority"] == "P1":
        assert asst["needs_human"] is True, f"{r['id']} P1 but needs_human is {asst['needs_human']}"

print("All 200 dev rows adhere strictly to schema!")
print(f"Categories: {cat_counts}")
print(f"Priorities: {priority_counts}")
print(f"Channels: {channel_counts}")
print(f"Languages: {lang_counts}")
print(f"Needs human: {needs_human_counts}")


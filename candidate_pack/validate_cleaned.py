import json
import re
from collections import Counter

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
ALLOWED_CATEGORIES = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
ALLOWED_PRIORITIES = {"P1", "P2", "P3"}
ALLOWED_CHANNELS = {"upi", "card", "netbanking", "wallet", None}
ALLOWED_LANGUAGES = {"en", "hi-en"}

rows = [json.loads(line) for line in open("candidate_pack/data/train_cleaned.jsonl", encoding="utf-8")]

print(f"Validating {len(rows)} cleaned rows...")
for r in rows:
    c = r["messages"][-1]["content"]
    data = json.loads(c)
    assert list(data.keys()) == KEYS, f"Keys order error in {r['id']}: {list(data.keys())}"
    assert data["category"] in ALLOWED_CATEGORIES, f"Bad cat in {r['id']}"
    assert data["priority"] in ALLOWED_PRIORITIES, f"Bad pri in {r['id']}"
    assert data["amount_paise"] is None or (isinstance(data["amount_paise"], int) and not isinstance(data["amount_paise"], bool)), f"Bad amt in {r['id']}"
    assert data["txn_date"] is None or re.match(r"^\d{4}-\d{2}-\d{2}$", data["txn_date"]), f"Bad date in {r['id']}: {data['txn_date']}"
    assert data["txn_id"] is None or re.match(r"^NP\d{10}$", data["txn_id"]), f"Bad txid in {r['id']}: {data['txn_id']}"
    assert data["channel"] in ALLOWED_CHANNELS, f"Bad channel in {r['id']}: {data['channel']}"
    assert data["language"] in ALLOWED_LANGUAGES, f"Bad lang in {r['id']}"
    assert isinstance(data["needs_human"], bool), f"Bad nh in {r['id']}"
    
    # Non-transaction categories MUST have null transaction fields
    if data["category"] in {"account_access", "kyc", "other"}:
        assert data["amount_paise"] is None, f"{r['id']} has amount for {data['category']}"
        assert data["txn_date"] is None, f"{r['id']} has date for {data['category']}"
        assert data["txn_id"] is None, f"{r['id']} has txid for {data['category']}"
        assert data["channel"] is None, f"{r['id']} has channel for {data['category']}"
        
    # P1 MUST have needs_human == True
    if data["priority"] == "P1":
        assert data["needs_human"] is True, f"{r['id']} P1 but needs_human is {data['needs_human']}"

print("All 1,410 cleaned training rows passed 100% of schema checks!")


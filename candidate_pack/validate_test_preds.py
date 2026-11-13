import json
import re

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
ALLOWED_CATEGORIES = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
ALLOWED_PRIORITIES = {"P1", "P2", "P3"}
ALLOWED_CHANNELS = {"upi", "card", "netbanking", "wallet", None}
ALLOWED_LANGUAGES = {"en", "hi-en"}

rows = [json.loads(line) for line in open("predictions_test.jsonl", encoding="utf-8")]
assert len(rows) == 400, f"Expected 400 rows, got {len(rows)}"

for r in rows:
    assert "id" in r and r["id"].startswith("ts-"), f"Invalid id: {r.get('id')}"
    assert "output" in r, f"Missing output in {r['id']}"
    
    # Check output JSON
    data = json.loads(r["output"].strip())
    assert list(data.keys()) == KEYS, f"Keys order mismatch in {r['id']}: {list(data.keys())}"
    assert data["category"] in ALLOWED_CATEGORIES, f"Bad cat in {r['id']}"
    assert data["priority"] in ALLOWED_PRIORITIES, f"Bad prio in {r['id']}"
    assert data["amount_paise"] is None or isinstance(data["amount_paise"], int), f"Bad amt in {r['id']}"
    assert data["txn_date"] is None or re.match(r"^\d{4}-\d{2}-\d{2}$", data["txn_date"]), f"Bad date in {r['id']}"
    assert data["txn_id"] is None or re.match(r"^NP\d{10}$", data["txn_id"]), f"Bad txid in {r['id']}"
    assert data["channel"] in ALLOWED_CHANNELS, f"Bad channel in {r['id']}"
    assert data["language"] in ALLOWED_LANGUAGES, f"Bad lang in {r['id']}"
    assert isinstance(data["needs_human"], bool), f"Bad nh in {r['id']}"
    
    # Null constraint for non-transaction categories
    if data["category"] in ("account_access", "kyc", "other"):
        assert data["amount_paise"] is None
        assert data["txn_date"] is None
        assert data["txn_id"] is None
        assert data["channel"] is None
        
    # P1 constraint
    if data["priority"] == "P1":
        assert data["needs_human"] is True

print("All 400 predictions in predictions_test.jsonl passed 100% of strict validation checks!")


import json
import re
from datetime import datetime, timedelta

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

dev_rows = load("candidate_pack/data/dev.jsonl")

# Schema definitions
KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]

# Regex patterns
legal_re = re.compile(r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b", re.IGNORECASE)
repeated_contact_re = re.compile(
    r"\b(again and again|third time|fourth time|second complaint|written \d+ times|contacted (?:you|support)|contacting you again|baar baar|pehle bhi|already raised|keep raising this|teesri baar|dusri baar)\b",
    re.IGNORECASE
)

# Fraud keywords (must indicate unauthorized transaction)
fraud_phrases = [
    r"didn't make", r"did not make", r"didn't do", r"did not do",
    r"unfamiliar", r"\bfraud\b", r"unrecognised", r"unrecognized",
    r"unauthorized", r"unauthorised", r"someone (?:has|had|might have|else|stole)",
    r"kisi aur ne", r"chori", r"not my transaction", r"not mine", r"does not belong to me",
    r"haven't made", r"never made", r"never submitted", r"without my (?:permission|consent)",
    r"maine (?:koi )?payment nahi kiya", r"maine nahi kiya", r"jaan bujh ke nahi",
    r"jaan nahi", r"anjaan", r"stranger", r"strange cash withdrawal", r"strange payment",
    r"random withdrawal", r"compromised"
]
fraud_re = re.compile("|".join(fraud_phrases), re.IGNORECASE)

# KYC phrases
kyc_phrases = [
    r"\bkyc\b", r"proof of identity", r"identity (?:check|verification)",
    r"verify(?:ing)? my identity", r"verify me", r"aadhaar", r"\bpan\b",
    r"id verification", r"type of id", r"identi(?:fy|fication)"
]
kyc_re = re.compile("|".join(kyc_phrases), re.IGNORECASE)

# Account access phrases
access_phrases = [
    r"passcode", r"unblock (?:my )?(?:pin|card|account)", r"reset (?:my )?(?:pin|passcode|password)",
    r"blocked (?:pin|account)", r"locked account", r"\botp\b", r"sign in", r"log in", r"login",
    r"cannot get in", r"wrong pin"
]
access_re = re.compile("|".join(access_phrases), re.IGNORECASE)

# Offers phrases
offers_phrases = [
    r"cashback", r"promo code", r"coupon", r"scratch card", r"referral", r"reward point",
    r"joining bonus", r"bonus"
]
offers_re = re.compile("|".join(offers_phrases), re.IGNORECASE)

# Refund phrases
refund_phrases = [
    r"\brefund\b", r"refunded", r"charged twice", r"duplicate (?:payment|charge|card)",
    r"twice", r"double charge", r"money back", r"paise wapas", r"wapas kar",
    r"return(?:ed)? (?:an item|a purchase|a product|saamaan)", r"cancel(?:led)? (?:order|purchase)"
]
refund_re = re.compile("|".join(refund_phrases), re.IGNORECASE)

# Payment failed phrases
failed_phrases = [
    r"failed", r"decline", r"declined", r"stuck", r"didn't go through", r"did not go through",
    r"didn't work", r"won't work", r"not working", r"not successful", r"unsuccessful",
    r"top ?up (?:failed|didn't|hasn't)", r"transfer (?:failed|didn't|not)",
    r"fail ho gaya", r"nahi mile", r"kat gaye", r"nahi hua"
]
failed_re = re.compile("|".join(failed_phrases), re.IGNORECASE)

print("Rule matcher initialized.")


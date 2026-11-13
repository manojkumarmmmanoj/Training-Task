import json
import re
from datetime import datetime, timedelta

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12
}

def extract_triage(user_content):
    lines = user_content.split("\n")
    rec_match = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", user_content)
    rec_date = datetime.strptime(rec_match.group(1), "%Y-%m-%d").date() if rec_match else None
    
    # Filter out quoted lines
    body_lines = [l for l in lines if not l.startswith(">")]
    body = "\n".join(body_lines)
    body_lower = body.lower()
    
    # 1. Language
    hi_words = {
        "karo", "hai", "hua", "nahi", "tha", "thi", "the", "mera", "meri", "mere",
        "dhanyavaad", "raha", "rahi", "hoon", "paise", "kat", "gaye", "gaya", "gayi",
        "bheje", "chuka", "chuki", "bhi", "pehle", "baar", "kya", "bhai", "namaste",
        "ji", "dost", "rakam", "saamaan", "lekin", "wapas", "mile", "karun", "pado",
        "pade", "rupaye", "rupiya", "teesri", "dusri", "pichla", "anjaan", "shukriya"
    }
    tokens = set(re.findall(r"\b[a-z]+\b", body_lower))
    lang = "hi-en" if len(tokens & hi_words) >= 2 else "en"
    
    # 2. Txn ID
    # Look for NP followed by 10 digits
    txid = None
    np_matches = re.findall(r"\bnp\s*[- ]?\s*(?:\d\s*[- ]?){10}\b", body, re.IGNORECASE)
    if np_matches:
        cand = re.sub(r"[^A-Za-z0-9]", "", np_matches[0]).upper()
        if re.match(r"^NP\d{10}$", cand):
            txid = cand

    # 3. Channel
    # Check mentions, ignoring "scratch card", "id card", "aadhaar", "pan card"
    channel = None
    # Mask distractors
    masked_body = re.sub(r"\b(?:scratch\s*card|identity\s*card|id\s*card|aadhaar\s*card|pan\s*card)\b", "____", body_lower)
    
    if re.search(r"\bupi\b", masked_body):
        channel = "upi"
    elif re.search(r"\b(netbanking|internet banking|net banking)\b", masked_body):
        channel = "netbanking"
    elif re.search(r"\bwallet\b", masked_body):
        channel = "wallet"
    elif re.search(r"\b(card|credit card|debit card|visa|mastercard)\b", masked_body):
        channel = "card"

    # 4. Date extraction
    dt_str = None
    if rec_date:
        # Check relative dates
        # "N days ago" or "N din pehle"
        m_days = re.search(r"(\d+)\s*(?:days?\s*ago|din\s*pehle)", masked_body)
        m_dby = re.search(r"\b(day before yesterday|parso)\b", masked_body)
        m_yest = re.search(r"\b(yesterday|kal(?:\s*ki\s*baat)?)\b", masked_body)
        m_today = re.search(r"\b(today|aaj)\b", masked_body)
        
        # Check explicit dates in unquoted text
        # Patterns:
        # YYYY-MM-DD
        m_iso = re.search(r"\b(202\d-\d{2}-\d{2})\b", masked_body)
        # DD/MM/YYYY or DD-MM-YYYY
        m_dmy = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](202\d)\b", masked_body)
        # DD-MM-YY
        m_dmy2 = re.search(r"\b(\d{1,2})-(\d{1,2})-(26)\b", masked_body)
        # DD(th)? Month (YYYY)?
        month_patt = r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|september|oct|nov|dec)[a-z]*\s*(202\d)?\b"
        m_txt1 = re.search(month_patt, masked_body)
        # Month DD(th)? (YYYY)?
        month_patt2 = r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|september|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(202\d))?\b"
        m_txt2 = re.search(month_patt2, masked_body)

        if m_iso:
            dt_str = m_iso.group(1)
        elif m_dmy:
            day, month, yr = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
            dt_str = f"{yr:04d}-{month:02d}-{day:02d}"
        elif m_dmy2:
            day, month, yr = int(m_dmy2.group(1)), int(m_dmy2.group(2)), 2000 + int(m_dmy2.group(3))
            dt_str = f"{yr:04d}-{month:02d}-{day:02d}"
        elif m_txt1:
            day = int(m_txt1.group(1))
            m_name = m_txt1.group(2).lower()[:3]
            month = MONTHS[m_name]
            yr = int(m_txt1.group(3)) if m_txt1.group(3) else rec_date.year
            cand_dt = datetime(yr, month, day).date()
            if cand_dt > rec_date:
                cand_dt = datetime(yr - 1, month, day).date()
            dt_str = cand_dt.strftime("%Y-%m-%d")
        elif m_txt2:
            m_name = m_txt2.group(1).lower()[:3]
            month = MONTHS[m_name]
            day = int(m_txt2.group(2))
            yr = int(m_txt2.group(3)) if m_txt2.group(3) else rec_date.year
            cand_dt = datetime(yr, month, day).date()
            if cand_dt > rec_date:
                cand_dt = datetime(yr - 1, month, day).date()
            dt_str = cand_dt.strftime("%Y-%m-%d")
        elif m_days:
            n = int(m_days.group(1))
            dt_str = (rec_date - timedelta(days=n)).strftime("%Y-%m-%d")
        elif m_dby:
            dt_str = (rec_date - timedelta(days=2)).strftime("%Y-%m-%d")
        elif m_yest:
            dt_str = (rec_date - timedelta(days=1)).strftime("%Y-%m-%d")
        elif m_today:
            dt_str = rec_date.strftime("%Y-%m-%d")

    # 5. Amount paise
    # Mask out distractors: balance, limit, fee, ticket number, phone model
    clean_text = masked_body
    # Mask balance
    clean_text = re.sub(r"(?:balance|limit|fee|charge\s*of)\s*(?:is|of|:)?\s*(?:rs\.?|inr|₹)?\s*([0-9,]+(?:\.[0-9]+)?)", "____", clean_text)
    clean_text = re.sub(r"(?:ticket|ticket number|ref)\s*#?\s*[0-9]+", "____", clean_text)
    clean_text = re.sub(r"\b(?:redmi note \d+|pixel \d+[a-z]?|oneplus \d+|iphone \d+|samsung [a-z0-9]+|vivo [a-z0-9]+|realme [0-9]+)\b", "____", clean_text)
    clean_text = re.sub(r"\bversion\s*[0-9.]+\b", "____", clean_text)
    
    amount_paise = None
    # Lakh
    m_lakh = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*lakh\b", clean_text)
    # k
    m_k = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*k\b", clean_text)
    # Rupees / INR / ₹ / rupaye
    m_rs = re.search(r"(?:rs\.?|inr|₹)\s*([0-9,]+(?:\.[0-9]+)?)\b", clean_text)
    m_rs_post = re.search(r"\b([0-9,]+(?:\.[0-9]+)?)\s*(?:rupees|rupaye|kat gaye|tha|ka tha)\b", clean_text)
    
    if m_lakh:
        val = float(m_lakh.group(1).replace(",", ""))
        amount_paise = int(round(val * 100000 * 100))
    elif m_k:
        val = float(m_k.group(1).replace(",", ""))
        amount_paise = int(round(val * 1000 * 100))
    elif m_rs:
        val = float(m_rs.group(1).replace(",", ""))
        amount_paise = int(round(val * 100))
    elif m_rs_post:
        val = float(m_rs_post.group(1).replace(",", ""))
        amount_paise = int(round(val * 100))

    # 6. Category
    # Fraud rule: wins over everything!
    fraud_patt = r"\b(didn't make|did not make|didn't do|did not do|unfamiliar|fraud|unrecognised|unrecognized|unauthorized|unauthorised|someone (?:has|had|might have|else|stole)|kisi aur ne|chori|not my transaction|not mine|does not belong to me|haven't made|never made|never submitted|without my (?:permission|consent)|maine (?:koi )?payment nahi kiya|maine nahi kiya|jaan bujh ke nahi|anjaan|strange cash withdrawal|strange payment|random withdrawal|compromised|withdrawal.*not remember)\b"
    is_fraud = bool(re.search(fraud_patt, masked_body))
    
    kyc_patt = r"\b(kyc|proof of identity|identity (?:check|verification)|verify(?:ing)? my identity|verify me|aadhaar|pan\b|id verification|type of id|identi(?:fy|fication))\b"
    is_kyc = bool(re.search(kyc_patt, masked_body))
    
    access_patt = r"\b(passcode|unblock (?:my )?(?:pin|card|account)|reset (?:my )?(?:pin|passcode|password)|blocked (?:pin|account)|locked account|\botp\b|sign in|log in|login|cannot get in|wrong pin)\b"
    is_access = bool(re.search(access_patt, masked_body))
    
    offers_patt = r"\b(cashback|promo code|coupon|scratch card|referral|reward point|joining bonus)\b"
    is_offers = bool(re.search(offers_patt, masked_body))
    
    refund_patt = r"\b(refund|refunded|charged twice|duplicate (?:payment|charge|card)|double charge|money back|paise wapas|wapas kar|return(?:ed)? (?:an item|a purchase|a product|saamaan)|cancel(?:led)? (?:order|purchase))\b"
    is_refund = bool(re.search(refund_patt, masked_body))
    
    failed_patt = r"\b(failed|decline|declined|stuck|didn't go through|did not go through|didn't work|won't work|not working|not successful|unsuccessful|top ?up (?:failed|didn't|hasn't)|transfer (?:failed|didn't|not)|fail ho gaya|nahi mile|nahi hua)\b"
    is_failed = bool(re.search(failed_patt, masked_body))
    
    if is_fraud:
        cat = "fraud"
    elif is_kyc:
        cat = "kyc"
    elif is_access:
        cat = "account_access"
    elif is_offers:
        cat = "offers"
    elif is_refund:
        cat = "refund"
    elif is_failed:
        cat = "payment_failed"
    else:
        cat = "other"

    # Transaction fields constraint
    if cat in ("account_access", "kyc", "other"):
        amount_paise = None
        dt_str = None
        txid = None
        channel = None

    # 7. Priority
    legal_patt = r"\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b"
    has_legal = bool(re.search(legal_patt, masked_body))
    
    if cat == "fraud" or (amount_paise is not None and amount_paise >= 5000000) or has_legal:
        priority = "P1"
    elif (cat in ("payment_failed", "refund") and amount_paise is not None and amount_paise >= 500000) or (cat == "account_access"):
        priority = "P2"
    else:
        priority = "P3"

    # 8. Needs Human
    rep_contact_patt = r"\b(again and again|third time|fourth time|second complaint|written \d+ times|contacted (?:you|support)|contacting you again|baar baar|pehle bhi|already raised|keep raising this|teesri baar|dusri baar)\b"
    has_rep_contact = bool(re.search(rep_contact_patt, masked_body))
    
    needs_human = True if (priority == "P1" or has_rep_contact) else False

    return {
        "category": cat,
        "priority": priority,
        "amount_paise": amount_paise,
        "txn_date": dt_str,
        "txn_id": txid,
        "channel": channel,
        "language": lang,
        "needs_human": needs_human
    }

dev_rows = [json.loads(line) for line in open("candidate_pack/data/dev.jsonl", encoding="utf-8")]
preds = []
for r in dev_rows:
    triage = extract_triage(r["messages"][1]["content"])
    preds.append({"id": r["id"], "output": json.dumps(triage)})

with open("candidate_pack/dev_rule_preds.jsonl", "w", encoding="utf-8") as f:
    for p in preds:
        f.write(json.dumps(p) + "\n")

print(f"Generated {len(preds)} predictions on dev.")


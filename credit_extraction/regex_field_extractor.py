"""
Zero-dependency, non-LLM extractor for the same field schema
credit_extraction_common.py's SYSTEM_PROMPT asks Qwen to produce --
keyword-proximity regex instead of a model. Pure `re`, no download, no
GPU, millisecond latency, and it fails by returning `None` rather than
inventing a plausible-looking value, which is the core reason it exists
alongside the LLM (see hybrid_extraction_pipeline.py's docstring).

Deliberately scoped to what regex is actually good at: numbers and enums
anchored by an explicit keyword in the same sentence ("credit score is
712", "run my own landscaping business"). It will miss phrasing it
wasn't written for -- that's the tradeoff for having zero training cost
and zero hallucination risk, and it's why this is paired with the LLM
rather than used alone.
"""
import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Requires an explicit currency marker ($ or a trailing "k") so a bare
# number like "2 years" in the same sentence is never mistaken for a
# dollar amount.
_AMOUNT_RE = re.compile(r"\$\s*([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k)?\b|\b(\d+(?:\.\d+)?)\s*k\b", re.I)


def _split_sentences(text: str):
    return _SENTENCE_SPLIT_RE.split(text)


def _parse_amount(sentence: str, start: int = 0):
    """Find a dollar amount at or after `start` in `sentence`. Anchoring
    forward from a keyword's position (rather than searching the whole
    sentence) matters when a sentence mentions more than one dollar
    figure -- e.g. "income $65,000 a year ... looking to borrow $15,000"
    has no sentence-ending punctuation until the very end, so without
    this anchor, searching for the loan amount would find the income
    figure instead, since it appears first."""
    match = _AMOUNT_RE.search(sentence, start)
    if not match:
        return None
    if match.group(1) is not None:
        value = float(match.group(1).replace(",", ""))
        if match.group(2):
            value *= 1000
    else:
        value = float(match.group(3)) * 1000
    return round(value)


def extract_age(text: str):
    for pattern in (
        r"\b(\d{1,3})\s*[-\s]?years?\s*old\b",
        r"\bage[:\s]+(\d{1,3})\b",
        r"\bI(?:'m|\s+am)\s+(\d{1,3})\b",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            age = int(match.group(1))
            if 18 <= age <= 100:
                return age
    return None


def extract_credit_score(text: str):
    for sentence in _split_sentences(text):
        if re.search(r"credit\s*score|fico|credit\s*rating", sentence, re.I):
            match = re.search(r"\b([2-8]\d{2})\b", sentence)
            if match:
                score = int(match.group(1))
                if 300 <= score <= 850:
                    return score
    return None


def extract_annual_income_usd(text: str):
    for sentence in _split_sentences(text):
        match = re.search(r"\b(income|salary|earn|make|bring in)\b", sentence, re.I)
        if match:
            amount = _parse_amount(sentence, match.end())
            if amount is not None and 5_000 <= amount <= 1_000_000:
                return amount
    return None


def extract_debt_to_income_ratio_percent(text: str):
    for sentence in _split_sentences(text):
        if re.search(r"debt[- ]to[- ]income|\bDTI\b", sentence, re.I):
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", sentence, re.I)
            if match:
                value = float(match.group(1))
                if 0 <= value <= 100:
                    return value
    return None


def extract_employment_status(text: str):
    if re.search(r"\bretired\b|\bpension\b", text, re.I):
        return "retired"
    if re.search(r"\bself[- ]employed\b|\bfreelance\b|\brun my own\b|\bown business\b", text, re.I):
        return "self_employed"
    if re.search(r"\bpart[- ]time\b", text, re.I):
        return "employed_part_time"
    if re.search(r"\bfull[- ]time\b|\b9-to-5\b", text, re.I):
        return "employed_full_time"
    return None


_EMPLOYMENT_CONTEXT_RE = re.compile(
    r"\b(job|work|business|employed|self-employed|full-time|full time|part-time|part time|freelance)\b", re.I
)


def extract_current_employment_duration_months(text: str):
    for sentence in _split_sentences(text):
        if re.search(r"\bretired\b", sentence, re.I):
            continue
        if not _EMPLOYMENT_CONTEXT_RE.search(sentence):
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*years?\b", sentence, re.I)
        if match:
            return round(float(match.group(1)) * 12)
        match = re.search(r"(\d+)\s*months?\b", sentence, re.I)
        if match:
            return int(match.group(1))
    return None


def extract_residency_status(text: str):
    if re.search(r"\bpermanent resident\b|\bgreen card\b", text, re.I):
        return "Permanent_Resident"
    if re.search(r"\bUS citizen\b|\bcitizen of the US\b|\bcitizen here\b", text, re.I):
        return "US_Citizen"
    return None


def extract_has_bankruptcy_recent(text: str):
    match = re.search(r"bankrupt\w*\s+(?:about\s+)?(\d+)\s*years?\s*(?:ago|back)\b", text, re.I)
    if match:
        return int(match.group(1)) <= 7
    if re.search(r"\bno\b[^.!?]*bankrupt|\bnever\b[^.!?]*bankrupt", text, re.I):
        return False
    return None  # mentioned with no timeframe, or not mentioned at all -- don't guess either way


def extract_has_verifiable_bank_account(text: str):
    if re.search(r"\bdon'?t have (?:a )?(?:verifiable )?bank account\b|\bno (?:verifiable )?bank account\b", text, re.I):
        return False
    if re.search(r"\bverifiable bank account\b|\bchecking account\b|\bbank account\b", text, re.I):
        return True
    return None


def extract_requested_amount_usd(text: str):
    for sentence in _split_sentences(text):
        match = re.search(r"\b(borrow|loan|requesting|asking for|take out)\b", sentence, re.I)
        if match:
            amount = _parse_amount(sentence, match.end())
            if amount is not None and 500 <= amount <= 500_000:
                return amount
    return None


def extract_fields_regex(text: str) -> dict:
    """Same schema shape as the LLM extractor's output -- a field this
    couldn't confidently pin down is `None`, exactly like the LLM's
    `null`, so both extractors' outputs can be merged field-by-field."""
    return {
        "applicant": {
            "age": extract_age(text),
            "credit_score": extract_credit_score(text),
            "annual_income_usd": extract_annual_income_usd(text),
            "debt_to_income_ratio_percent": extract_debt_to_income_ratio_percent(text),
            "employment_status": extract_employment_status(text),
            "current_employment_duration_months": extract_current_employment_duration_months(text),
            "residency_status": extract_residency_status(text),
            "has_bankruptcy_recent": extract_has_bankruptcy_recent(text),
            "has_verifiable_bank_account": extract_has_verifiable_bank_account(text),
        },
        "loan_application": {
            "requested_amount_usd": extract_requested_amount_usd(text),
        },
    }

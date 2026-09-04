"""
Synthesize labeled (text, JSON-fields) pairs for
credit_extraction/credit_fields_finetune_qwen.py -- one free-text applicant
statement per row, paired with the exact JSON label that script trains
Qwen2.5-3B-Instruct to reproduce.

    python data_generator/generate_credit_extraction_data.py

Writes data/credit_extraction_data.json: a JSON array, one compact
object per line, each `{"text": ..., "label": {...}}`. Every field is
included independently at random (and left out -> labeled null when
excluded) so the model sees realistic partial statements, not just
fully-populated ones -- that's the input shape CreditRuleChecker
actually has to tolerate, since it does no input validation of its own
(see the comment on encode_with_completion_mask in
credit_fields_finetune_qwen.py).

Two deliberately "hard" cases are baked into the generator rather than
left to chance:
  - residency_status sometimes gets a phrase ("on an H-1B visa") that
    is truthful but doesn't match either enum value in
    RULE-RESIDENCY-001 -- correct label is null, not a guess.
  - has_bankruptcy_recent is derived from *how many years ago* a
    filing happened, mirroring RULE-BANKRUPTCY-001's "within the last
    7 years" wording -- so "bankruptcy 9 years ago" must label False,
    not True, even though bankruptcy is mentioned.

Deterministic: reruns with the same COUNT and SEED reproduce the same
file byte-for-byte.
"""
import json
import random
from pathlib import Path

COUNT = 10_000
SEED = 1337
# Resolved relative to this file (not the current working directory) so
# this always writes to the same place regardless of where it's invoked
# from -- see credit_extraction_common.py's ADAPTER_DIR for the same
# pattern, adopted after a cwd-relative path there caused a real bug.
# parent.parent because this file lives in data_generator/ and the
# dataset lives in data/, both direct children of the repo root.
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "credit_extraction_data.json"

JOBS = [
    "nurse", "teacher", "software engineer", "electrician", "accountant",
    "chef", "truck driver", "graphic designer", "sales associate",
    "mechanic", "warehouse worker", "barista", "construction worker",
    "customer service rep", "dental hygienist", "paralegal",
]
BUSINESSES = [
    "landscaping", "consulting", "photography", "catering", "cleaning",
    "bakery", "auto repair", "tutoring", "e-commerce", "construction",
]
COMPANIES = [
    "a manufacturing plant", "a hospital", "a tech startup",
    "a retail store", "a law firm", "a school district",
    "a logistics company", "a restaurant",
]
BANKS = [
    "Chase", "Wells Fargo", "Bank of America", "Citibank", "Capital One",
    "US Bank", "PNC", "TD Bank", "Ally Bank", "a local credit union",
]
OTHER_RESIDENCY_PHRASES = [
    "I'm on an H-1B work visa",
    "I'm here on a student visa",
    "I have a pending green card application",
    "I'm not a US citizen or permanent resident",
    "I'm on a temporary work permit",
]

EMPLOYMENT_STATUS_VALUES = ["employed_full_time", "employed_part_time", "self_employed", "retired"]
RESIDENCY_STATUS_VALUES = ["US_Citizen", "Permanent_Resident"]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def gen_age(rng):
    return clamp(round(rng.gauss(38, 13)), 18, 80)


def gen_credit_score(rng):
    return clamp(round(rng.gauss(680, 90)), 300, 850)


def gen_income(rng):
    return clamp(round(rng.gauss(60000, 25000) / 500) * 500, 12000, 250000)


def gen_dti(rng):
    value = clamp(rng.gauss(30, 15), 0, 70)
    return round(value, 1) if rng.random() < 0.3 else round(value)


def gen_duration_months(rng):
    return clamp(round(rng.gauss(30, 30)), 1, 360)


def gen_loan_amount(rng):
    return clamp(round(rng.gauss(18000, 10000) / 500) * 500, 1000, 100000)


def phrase_age(rng, age):
    return rng.choice([
        f"I'm {age} years old", f"I am {age}", f"{age}-year-old applicant",
        f"Age: {age}",
    ])


def phrase_credit_score(rng, score):
    return rng.choice([
        f"my credit score is {score}", f"my FICO score is {score}",
        f"credit rating of {score}", f"my score's about {score}",
        f"my credit score is around {score}",
    ])


def phrase_income(rng, income):
    formatted = f"${income:,}" if rng.random() < 0.7 else f"${income // 1000}k"
    return rng.choice([
        f"I make {formatted} a year", f"annual income is {formatted}",
        f"I earn about {formatted} a year", f"salary is {formatted}",
        f"I bring in {formatted}/year",
    ])


def phrase_dti(rng, dti):
    return rng.choice([
        f"my debt-to-income ratio is {dti}%", f"DTI is around {dti}%",
        f"debt to income sits at {dti}%", f"my DTI is {dti} percent",
    ])


def phrase_employment(rng, status, duration_months):
    duration_phrase = ""
    if duration_months is not None:
        if duration_months >= 12 and rng.random() < 0.6:
            years = round(duration_months / 12, 1)
            duration_phrase = f", been there {years} years" if rng.random() < 0.5 else f" for the last {years} years"
        else:
            duration_phrase = f", about {duration_months} months in" if rng.random() < 0.5 else f" for {duration_months} months"

    if status == "employed_full_time":
        base = rng.choice([
            f"I work full-time as a {rng.choice(JOBS)}",
            f"I have a full-time job at {rng.choice(COMPANIES)}",
            "I'm employed full time",
            f"I work a 9-to-5 as a {rng.choice(JOBS)}",
        ])
    elif status == "employed_part_time":
        base = rng.choice([
            f"I work part-time as a {rng.choice(JOBS)}",
            f"I pick up part-time shifts at {rng.choice(COMPANIES)}",
            "I'm employed part time",
        ])
    elif status == "self_employed":
        base = rng.choice([
            f"I run my own {rng.choice(BUSINESSES)} business",
            f"I freelance as a {rng.choice(JOBS)}",
            "I'm self-employed",
        ])
    else:  # retired
        base = rng.choice([
            "I'm retired", f"I've been retired since {rng.randint(2005, 2023)}",
            "I'm living off my pension",
        ])
        duration_phrase = ""  # "current employment duration" doesn't apply to retirees

    return base + duration_phrase


def phrase_residency(rng, value):
    if value == "US_Citizen":
        return rng.choice(["I'm a US citizen", "US citizen here", "I'm a citizen of the US"])
    if value == "Permanent_Resident":
        return rng.choice(["I'm a permanent resident", "I've been a permanent resident for years", "green card holder"])
    return rng.choice(OTHER_RESIDENCY_PHRASES)


def phrase_bankruptcy(rng, years_ago):
    if years_ago is None:
        return rng.choice(["no bankruptcies", "never filed for bankruptcy", "no bankruptcy history"])
    return rng.choice([
        f"I filed for bankruptcy {years_ago} years ago",
        f"I declared bankruptcy about {years_ago} years back",
    ])


def phrase_bank_account(rng, has_account):
    if has_account:
        return rng.choice([
            f"I have a verifiable bank account at {rng.choice(BANKS)}",
            f"I have a checking account with {rng.choice(BANKS)}",
            "I have an active, verifiable bank account",
        ])
    return rng.choice(["I don't have a bank account on file", "no verifiable bank account yet"])


def phrase_loan_amount(rng, amount):
    formatted = f"${amount:,}"
    return rng.choice([
        f"looking to borrow {formatted}", f"requesting a {formatted} loan",
        f"I want to take out {formatted}", f"asking for {formatted}",
    ])


def generate_one(rng):
    applicant = {}
    fragments = []

    age = None
    if rng.random() < 0.55:
        age = gen_age(rng)
        applicant["age"] = age
        fragments.append(phrase_age(rng, age))
    else:
        applicant["age"] = None

    if rng.random() < 0.85:
        score = gen_credit_score(rng)
        applicant["credit_score"] = score
        fragments.append(phrase_credit_score(rng, score))
    else:
        applicant["credit_score"] = None

    if rng.random() < 0.7:
        income = gen_income(rng)
        applicant["annual_income_usd"] = income
        fragments.append(phrase_income(rng, income))
    else:
        applicant["annual_income_usd"] = None

    if rng.random() < 0.5:
        dti = gen_dti(rng)
        applicant["debt_to_income_ratio_percent"] = dti
        fragments.append(phrase_dti(rng, dti))
    else:
        applicant["debt_to_income_ratio_percent"] = None

    if rng.random() < 0.85:
        status = rng.choices(EMPLOYMENT_STATUS_VALUES, weights=[0.45, 0.15, 0.15, 0.25])[0]
        applicant["employment_status"] = status
        duration = None
        if status != "retired" and rng.random() < 0.6:
            duration = gen_duration_months(rng)
        applicant["current_employment_duration_months"] = duration
        fragments.append(phrase_employment(rng, status, duration))
    else:
        applicant["employment_status"] = None
        applicant["current_employment_duration_months"] = None

    if rng.random() < 0.6:
        residency = rng.choices(
            [*RESIDENCY_STATUS_VALUES, "other"], weights=[0.55, 0.25, 0.20]
        )[0]
        applicant["residency_status"] = None if residency == "other" else residency
        fragments.append(phrase_residency(rng, residency))
    else:
        applicant["residency_status"] = None

    if rng.random() < 0.55:
        mention_recent = rng.random() < 0.2
        years_ago = rng.randint(1, 7) if mention_recent else rng.randint(1, 15)
        has_filed = rng.random() < 0.25
        if has_filed:
            applicant["has_bankruptcy_recent"] = years_ago <= 7
            fragments.append(phrase_bankruptcy(rng, years_ago))
        else:
            applicant["has_bankruptcy_recent"] = False
            fragments.append(phrase_bankruptcy(rng, None))
    else:
        applicant["has_bankruptcy_recent"] = None

    if rng.random() < 0.6:
        has_account = rng.random() < 0.85
        applicant["has_verifiable_bank_account"] = has_account
        fragments.append(phrase_bank_account(rng, has_account))
    else:
        applicant["has_verifiable_bank_account"] = None

    loan_application = {}
    if rng.random() < 0.75:
        amount = gen_loan_amount(rng)
        loan_application["requested_amount_usd"] = amount
        fragments.append(phrase_loan_amount(rng, amount))
    else:
        loan_application["requested_amount_usd"] = None

    rng.shuffle(fragments)
    if not fragments:
        fragments.append("No further financial details provided.")
    fragments = [f[0].upper() + f[1:] for f in fragments]
    text = ". ".join(fragments) + "."

    return {
        "text": text,
        "label": {"applicant": applicant, "loan_application": loan_application},
    }


def main():
    rng = random.Random(SEED)
    records = [generate_one(rng) for _ in range(COUNT)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, record in enumerate(records):
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            suffix = ",\n" if i < len(records) - 1 else "\n"
            f.write("  " + line + suffix)
        f.write("]\n")

    print(f"Wrote {len(records)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

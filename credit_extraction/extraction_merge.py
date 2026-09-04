"""
Field-level merge policy for combining the LLM extractor
(credit_extraction_pipeline.py) with the regex extractor
(regex_field_extractor.py). See hybrid_extraction_pipeline.py's
docstring for the reasoning behind which extractor wins where.
"""

# Numeric fields with an unambiguous keyword anchor ("credit score is
# X", "$X a year") -- the regex extractor's answer wins here whenever it
# found one, since the LLM's flexibility buys nothing on these and its
# hallucination risk (see credit_extraction_pipeline.py's docstring) is
# pure downside. Every other field defaults to the LLM, falling back to
# regex only when the LLM returned null.
REGEX_PREFERRED_FIELDS = {
    "applicant.age",
    "applicant.credit_score",
    "applicant.annual_income_usd",
    "applicant.debt_to_income_ratio_percent",
    "loan_application.requested_amount_usd",
}


def merge_records(regex_record: dict, llm_record: dict):
    """Merge two records sharing the applicant/loan_application schema.

    Returns (merged, sources, conflicts):
      - merged: the combined record, ready for CreditRuleChecker.check()
      - sources: {"applicant.credit_score": "regex", ...} -- which
        extractor's value was used for each field, or None if neither
        extractor found one
      - conflicts: [(field_path, regex_value, llm_value), ...] for every
        field where both extractors found a value and disagreed -- a
        free consistency check a single-extractor pipeline can't get
    """
    merged = {"applicant": {}, "loan_application": {}}
    sources = {}
    conflicts = []

    for section in ("applicant", "loan_application"):
        regex_section = (regex_record or {}).get(section) or {}
        llm_section = (llm_record or {}).get(section) or {}

        for field in set(regex_section) | set(llm_section):
            path = f"{section}.{field}"
            regex_value = regex_section.get(field)
            llm_value = llm_section.get(field)

            if regex_value is not None and llm_value is not None and regex_value != llm_value:
                conflicts.append((path, regex_value, llm_value))

            if path in REGEX_PREFERRED_FIELDS:
                value, source = (regex_value, "regex") if regex_value is not None else (llm_value, "llm")
            else:
                value, source = (llm_value, "llm") if llm_value is not None else (regex_value, "regex")

            merged[section][field] = value
            sources[path] = source if value is not None else None

    return merged, sources, conflicts

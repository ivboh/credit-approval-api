"""
Hybrid extraction: run the Qwen LLM extractor
(credit_extraction_pipeline.py, unmodified) and the zero-dependency
regex extractor (regex_field_extractor.py) on the same free-text
statement, merge their per-field outputs (extraction_merge.py), and only
then hand the merged record to CreditRuleChecker.

    pip install transformers torch peft
    python credit_extraction/credit_fields_finetune_qwen.py    # optional: trains the adapter this reuses
    python credit_extraction/hybrid_extraction_pipeline.py

Why merge instead of picking one extractor: the LLM and a
keyword-proximity regex fail in different, mostly uncorrelated ways. The
LLM can invent a plausible-looking value for something the text never
said -- see credit_extraction_pipeline.py's docstring: before
fine-tuning, it hallucinated `"residency_status": "US_Citizen"` for a
statement that never mentioned citizenship. A regex either matches a
real pattern or returns null; it never fabricates. But regex can't make
the semantic calls the LLM gets for free, like inferring
`"self_employed"` from "I run my own landscaping business" without the
enum string ever appearing in the text.

Merge policy (REGEX_PREFERRED_FIELDS in extraction_merge.py): numeric
fields with an unambiguous keyword anchor (credit_score,
annual_income_usd, debt_to_income_ratio_percent, requested_amount_usd,
age) default to the regex answer when it found one -- the LLM's
flexibility buys nothing there, so its hallucination risk is pure
downside. Every other field defaults to the LLM, falling back to regex
only if the LLM returned null. Whenever both extractors found a value
and disagree, that's logged as a CONFLICT -- an independent consistency
check a single-extractor pipeline never gets for free.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from credit_extraction_pipeline import APPLICANT_STATEMENTS, extract_applicant_record, load_model  # noqa: E402
from extraction_merge import merge_records  # noqa: E402
from regex_field_extractor import extract_fields_regex  # noqa: E402

# decision_engine/ lives one directory up from credit_extraction/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decision_engine.deterministic_checks import CreditRuleChecker  # noqa: E402

EMPTY_RECORD = {"applicant": {}, "loan_application": {}}


def main():
    model, tokenizer, device = load_model()
    checker = CreditRuleChecker()  # loads decision_engine/rules/credit_rules.json

    for text in APPLICANT_STATEMENTS:
        print(f"\n{text!r}")

        llm_record = extract_applicant_record(model, tokenizer, device, text)
        regex_record = extract_fields_regex(text)
        print(f"  llm:    {json.dumps(llm_record)}")
        print(f"  regex:  {json.dumps(regex_record)}")

        merged, sources, conflicts = merge_records(regex_record, llm_record or EMPTY_RECORD)
        print(f"  merged: {json.dumps(merged)}")
        for path, source in sorted(sources.items()):
            if source is not None:
                print(f"    {path} <- {source}")
        for path, regex_value, llm_value in conflicts:
            print(f"    CONFLICT {path}: regex={regex_value!r} vs llm={llm_value!r} -- used {sources[path]}")

        result = checker.check(merged)
        status = "HARD_FAIL" if result.hard_fail else ("FLAGGED" if result.reasons else "CLEAN")
        print(f"  decision: {status}")
        for reason in result.reasons:
            print(f"    - {reason}")


if __name__ == "__main__":
    main()

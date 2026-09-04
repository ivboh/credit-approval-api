"""
Runs the hybrid extraction pipeline (LLM + regex, merged --
hybrid_extraction_pipeline.py) over a sample of data/val.json's rows --
the held-out split data_generator/train_test_splitter.py produced, which
credit_fields_finetune_qwen.py never trains on, so LLM accuracy here
isn't inflated by the model having memorized the answer. Writes one
JSON object per row to export/hybrid_pipeline_val_results.jsonl (text,
both extractors' raw output, the merged record, which extractor
supplied each field, every conflict, and the CreditRuleChecker decision
under the true label / LLM alone / merged), then prints a summary:
per-field accuracy for each of the three (LLM alone, regex alone,
merged) against ground truth, and how often each one's final decision
matches the decision the true label itself would produce.

    python evaluation/run_hybrid_on_dataset.py [N]

Defaults to the first 100 rows (~17s per LLM call -> ~30 min for 100).
Pass a different N as the sole CLI argument to change the sample size.

evaluate_extraction_accuracy.py re-scores this run's output file without
re-running any extraction -- run this first, then that.
"""
import json
import sys
from pathlib import Path

# credit_extraction_pipeline.py and its sibling modules live in
# credit_extraction/, not in this file's own directory (evaluation/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "credit_extraction"))
from credit_extraction_pipeline import extract_applicant_record, load_model  # noqa: E402
from extraction_merge import merge_records  # noqa: E402
from regex_field_extractor import extract_fields_regex  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decision_engine.deterministic_checks import CreditRuleChecker  # noqa: E402

DATA_PATH = Path(__file__).parent.parent / "data" / "val.json"
RESULTS_PATH = Path(__file__).parent.parent / "export" / "hybrid_pipeline_val_results.jsonl"
DEFAULT_SAMPLE_SIZE = 100
EMPTY_RECORD = {"applicant": {}, "loan_application": {}}

FIELD_PATHS = [
    "applicant.age",
    "applicant.credit_score",
    "applicant.annual_income_usd",
    "applicant.debt_to_income_ratio_percent",
    "applicant.employment_status",
    "applicant.current_employment_duration_months",
    "applicant.residency_status",
    "applicant.has_bankruptcy_recent",
    "applicant.has_verifiable_bank_account",
    "loan_application.requested_amount_usd",
]


def get_field(record, path):
    section, field = path.split(".", 1)
    return (record or {}).get(section, {}).get(field)


def decision_for(checker, record):
    result = checker.check(record)
    if result.hard_fail:
        return "HARD_FAIL"
    if result.reasons:
        return "FLAGGED"
    return "CLEAN"


def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE_SIZE

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    sample = records[:sample_size]
    n = len(sample)

    model, tokenizer, device = load_model()
    checker = CreditRuleChecker()  # loads decision_engine/rules/credit_rules.json

    field_correct = {"llm": {p: 0 for p in FIELD_PATHS}, "regex": {p: 0 for p in FIELD_PATHS}, "merged": {p: 0 for p in FIELD_PATHS}}
    decision_agree = {"llm": 0, "merged": 0}
    conflict_counts = {}
    total_conflicts = 0

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as out:
        for i, record in enumerate(sample):
            text = record["text"]
            true_label = record["label"]

            llm_extraction = extract_applicant_record(model, tokenizer, device, text) or EMPTY_RECORD
            regex_extraction = extract_fields_regex(text)
            merged, sources, conflicts = merge_records(regex_extraction, llm_extraction)

            for path, regex_value, llm_value in conflicts:
                conflict_counts[path] = conflict_counts.get(path, 0) + 1
                total_conflicts += 1

            for path in FIELD_PATHS:
                expected = get_field(true_label, path)
                if get_field(llm_extraction, path) == expected:
                    field_correct["llm"][path] += 1
                if get_field(regex_extraction, path) == expected:
                    field_correct["regex"][path] += 1
                if get_field(merged, path) == expected:
                    field_correct["merged"][path] += 1

            decision_true = decision_for(checker, true_label)
            decision_llm = decision_for(checker, llm_extraction)
            decision_merged = decision_for(checker, merged)
            if decision_llm == decision_true:
                decision_agree["llm"] += 1
            if decision_merged == decision_true:
                decision_agree["merged"] += 1

            out.write(json.dumps({
                "index": i,
                "text": text,
                "true_label": true_label,
                "llm_extraction": llm_extraction,
                "regex_extraction": regex_extraction,
                "merged": merged,
                "sources": sources,
                "conflicts": conflicts,
                "decision_true": decision_true,
                "decision_llm": decision_llm,
                "decision_merged": decision_merged,
            }) + "\n")
            out.flush()

            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{n}] done", flush=True)

    print(f"\n=== Summary over {n} examples ===")
    print(f"Total conflicts: {total_conflicts}")
    for path, count in sorted(conflict_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {path}: {count} conflicts")

    print("\nPer-field exact-match accuracy vs ground truth:")
    for path in FIELD_PATHS:
        llm_acc = field_correct["llm"][path] / n
        regex_acc = field_correct["regex"][path] / n
        merged_acc = field_correct["merged"][path] / n
        print(f"  {path}: llm={llm_acc:.1%}  regex={regex_acc:.1%}  merged={merged_acc:.1%}")

    print("\nFinal-decision agreement with the ground-truth-based decision:")
    print(f"  llm-only:        {decision_agree['llm']}/{n} ({decision_agree['llm'] / n:.1%})")
    print(f"  merged (hybrid): {decision_agree['merged']}/{n} ({decision_agree['merged'] / n:.1%})")

    print(f"\nFull per-example results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

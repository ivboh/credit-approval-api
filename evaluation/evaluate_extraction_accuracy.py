"""
Compares the LLM extractor's and the regex extractor's already-computed
output -- export/hybrid_pipeline_val_results.jsonl, produced by
evaluation/run_hybrid_on_dataset.py against data/val.json -- back
against the ground truth in data/val.json itself, and reports per-field
accuracy for each extractor alone.

    python evaluation/run_hybrid_on_dataset.py       # writes export/hybrid_pipeline_val_results.jsonl
    python evaluation/evaluate_extraction_accuracy.py

This script does no extraction itself -- no model load, no `.generate()`
call, no regex re-run -- it only re-scores output that already exists.
That makes it near-instant to rerun after changing how the numbers are
sliced or reported, without paying for another ~17s-per-row LLM pass.
The actual field-accuracy numbers it computes are identical to what
run_hybrid_on_dataset.py already prints for "llm" and "regex" at the end
of its own run; this exists to re-derive them later without rerunning
extraction, and to isolate LLM-alone/regex-alone accuracy from that
script's broader summary (which also covers the merged/hybrid result,
conflicts, and decision agreement).

data/val.json and the results file are matched up by position (row 0
with row 0, row 1 with row 1, ...), not by any id, since neither file
carries one -- this only works because run_hybrid_on_dataset.py reads
data/val.json top-to-bottom with no shuffling in between. If the
results file has fewer rows than data/val.json (e.g. it was run with a
sample size N), only the first N rows of data/val.json are compared;
if it has more, something is inconsistent and this raises rather than
silently comparing mismatched rows.
"""
import json
from pathlib import Path

VAL_PATH = Path(__file__).resolve().parent.parent / "data" / "val.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "export" / "hybrid_pipeline_val_results.jsonl"

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


def main():
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        result_rows = [json.loads(line) for line in f]

    n = len(result_rows)
    if n > len(val_records):
        raise ValueError(
            f"{RESULTS_PATH} has {n} rows but {VAL_PATH} only has {len(val_records)} -- "
            "these don't look like they came from the same val.json; rerun run_hybrid_on_dataset.py."
        )

    field_correct = {"llm": {p: 0 for p in FIELD_PATHS}, "regex": {p: 0 for p in FIELD_PATHS}}
    label_mismatches = 0

    for val_record, result_row in zip(val_records[:n], result_rows):
        true_label = val_record["label"]
        # Sanity check: the results file carries its own copy of the true
        # label (written at extraction time) -- if it disagrees with
        # data/val.json's current content, the two files no longer
        # correspond row-for-row (e.g. val.json was regenerated since).
        if true_label != result_row["true_label"]:
            label_mismatches += 1

        llm_extraction = result_row["llm_extraction"]
        regex_extraction = result_row["regex_extraction"]

        for path in FIELD_PATHS:
            expected = get_field(true_label, path)
            if get_field(llm_extraction, path) == expected:
                field_correct["llm"][path] += 1
            if get_field(regex_extraction, path) == expected:
                field_correct["regex"][path] += 1

    if label_mismatches:
        print(
            f"WARNING: {label_mismatches}/{n} rows' ground-truth labels differ between "
            f"{VAL_PATH} and {RESULTS_PATH} -- results below may not be trustworthy.\n"
        )

    total_llm = sum(field_correct["llm"].values())
    total_regex = sum(field_correct["regex"].values())
    total_fields = n * len(FIELD_PATHS)

    print(f"=== Accuracy vs ground truth over {n} rows ({VAL_PATH.name}) ===")
    print(f"(scored from {RESULTS_PATH.name}, no extraction re-run)\n")
    print(f"{'Field':<42}{'LLM':>10}{'Regex':>10}")
    for path in FIELD_PATHS:
        llm_acc = field_correct["llm"][path] / n
        regex_acc = field_correct["regex"][path] / n
        print(f"{path:<42}{llm_acc:>10.1%}{regex_acc:>10.1%}")
    print(f"{'-' * 62}")
    print(f"{'OVERALL (all fields)':<42}{total_llm / total_fields:>10.1%}{total_regex / total_fields:>10.1%}")


if __name__ == "__main__":
    main()

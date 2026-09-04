"""
Splits data/credit_extraction_data.json into train/val subsets and
writes each to its own file in data/ -- factored out of
credit_extraction/credit_fields_finetune_qwen.py so the split is a
separate, inspectable step rather than something recomputed silently
every time the fine-tuning script runs.

    python data_generator/generate_credit_extraction_data.py   # if data/credit_extraction_data.json doesn't exist yet
    python data_generator/train_test_splitter.py

Writes data/train.json and data/val.json, same schema and same compact
one-record-per-line format as credit_extraction_data.json (see that
generator's docstring for why: git-friendly diffs, more compact than
pretty-printed JSON).

Deterministic: the shuffle uses a fixed seed, so rerunning this
reproduces the same split every time -- reshuffling only happens if you
change SPLIT_SEED or TRAIN_FRACTION yourself. Splitting is independent
of generation: regenerating credit_extraction_data.json (e.g. with a
different COUNT) means rerunning this script too, but editing the split
fraction doesn't require regenerating the underlying dataset.
"""
import json
import random
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "credit_extraction_data.json"
TRAIN_PATH = Path(__file__).parent.parent / "data" / "train.json"
VAL_PATH = Path(__file__).parent.parent / "data" / "val.json"

TRAIN_FRACTION = 0.9
SPLIT_SEED = 42  # fixed so the split is reproducible across runs


def _write_records(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, record in enumerate(records):
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            suffix = ",\n" if i < len(records) - 1 else "\n"
            f.write("  " + line + suffix)
        f.write("]\n")


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(records)

    split_idx = int(len(records) * TRAIN_FRACTION)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    _write_records(TRAIN_PATH, train_records)
    _write_records(VAL_PATH, val_records)

    print(f"Wrote {len(train_records)} train examples to {TRAIN_PATH}")
    print(f"Wrote {len(val_records)} val examples to {VAL_PATH}")


if __name__ == "__main__":
    main()

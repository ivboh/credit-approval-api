"""
LoRA fine-tune Qwen2.5-3B-Instruct to extract every field
`CreditRuleChecker` needs (decision_engine/rules/credit_rules.json)
directly out of a free-text applicant statement, as a single JSON object
-- no per-field regex/NER, no LLM call at inference time against a
hosted API (the fine-tuned adapter runs locally).

    pip install transformers torch peft accelerate
    python data_generator/generate_credit_extraction_data.py   # writes the 10k-example dataset once
    python data_generator/train_test_splitter.py               # splits it into data/train.json + data/val.json
    python credit_extraction/credit_fields_finetune_qwen.py

Requires a CUDA GPU with >=16GB VRAM to get through 9,000 training rows
in reasonable time. On CPU this will run but is impractically slow for
the full dataset -- set COUNT lower in generate_credit_extraction_data.py
and/or lower EPOCHS below to smoke-test the mechanics instead.

Training data comes from data/train.json and data/val.json (90/10 split
of the 10,000 synthetic rows generate_credit_extraction_data.py writes
to data/credit_extraction_data.json -- see data_generator/train_test_splitter.py
for the split itself, and generate_credit_extraction_data.py's docstring
for how fields are sampled and phrased). It's still synthetic data with
a limited set of phrasing templates, not real applicant text -- treat
this as demonstrating the fine-tuning mechanics at a realistic *scale*,
not as production-ready training data.

Saves the trained LoRA adapter to qwen_credit_lora_adapter/
(ADAPTER_DIR in credit_extraction_common.py) -- that's where
credit_extraction/credit_extraction_pipeline.py loads it from to wire the model's
extraction into CreditRuleChecker, the deterministic implementation of
credit_rules.json. The prompt/schema constants live in
credit_extraction_common.py rather than being copied into both files, so
a prompt change here can't silently drift out of sync with what the
pipeline script serves at inference time.

Design notes:
  - Target schema mirrors the record shape `CreditRuleChecker.check()`
    expects (see its docstring in decision_engine/deterministic_checks.py)
    -- an "applicant" object with every `applicant.*` field referenced
    by a rule's "field" in decision_engine/rules/credit_rules.json, plus
    "loan_application".
  - A field the input text never mentions is labeled `null` rather than
    guessed -- CreditRuleChecker has no input validation of its own
    (see decision_engine/deterministic_checks.py), so downstream code
    must treat `null` as "needs a human/another source", not silently
    coerce it.
  - `employment_status` / `residency_status` are constrained to the enum
    values from RULE-EMPLOY-001 / RULE-RESIDENCY-001 in the ruleset --
    the prompt spells them out so the model has somewhere to land instead
    of inventing a phrase CreditRuleChecker's `in` operator won't match.
  - Loss is computed only over the assistant's JSON completion, not the
    prompt -- otherwise the model would spend capacity learning to
    reproduce the (fixed) instructions instead of the extraction itself.
"""
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from credit_extraction_common import ADAPTER_DIR, MODEL_NAME, SYSTEM_PROMPT, build_prompt_text

TRAIN_PATH = Path(__file__).parent.parent / "data" / "train.json"
VAL_PATH = Path(__file__).parent.parent / "data" / "val.json"

BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512  # SYSTEM_PROMPT + longest applicant text + JSON completion + ChatML markers ~= 400-420 tokens
EVAL_SAMPLE_SIZE = 50  # val rows to actually run generation on (all 1,000 would be slow on CPU)


def load_examples(path: Path):
    """Load a data_generator/train_test_splitter.py output file --
    already-split train or val examples, not the raw combined dataset."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [(r["text"], r["label"]) for r in records]


TRAIN_EXAMPLES = load_examples(TRAIN_PATH)
VALIDATE_EXAMPLES = load_examples(VAL_PATH)


def build_example_text(tokenizer, applicant_text: str, label: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": applicant_text},
        {"role": "assistant", "content": json.dumps(label)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def encode_with_completion_mask(tokenizer, applicant_text: str, label: dict):
    """Tokenize prompt+completion together, but mask the prompt tokens'
    labels with -100 so the loss only trains the JSON completion."""
    prompt = build_prompt_text(tokenizer, applicant_text)
    full = build_example_text(tokenizer, applicant_text, label)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


class CreditExtractionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        input_ids, labels = encode_with_completion_mask(self.tokenizer, text, label)
        return {"input_ids": input_ids[: self.max_length], "labels": labels[: self.max_length]}


def collate_batch(batch, pad_id):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def evaluate(model, tokenizer, examples, device, sample_size=EVAL_SAMPLE_SIZE):
    """Run generation on a sample of the labeled val set and report
    exact-match accuracy per field, plus how often the model even
    produced valid JSON."""
    model.eval()
    sample = examples[:sample_size]
    field_correct, field_total, json_valid = {}, {}, 0

    for text, label in sample:
        prompt = build_prompt_text(tokenizer, text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        completion = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            parsed = json.loads(completion)
            json_valid += 1
        except json.JSONDecodeError:
            parsed = {}

        for section in ("applicant", "loan_application"):
            for field, expected in label.get(section, {}).items():
                key = f"{section}.{field}"
                predicted = parsed.get(section, {}).get(field) if isinstance(parsed, dict) else None
                field_total[key] = field_total.get(key, 0) + 1
                if predicted == expected:
                    field_correct[key] = field_correct.get(key, 0) + 1

    print(f"\nValid JSON output: {json_valid}/{len(sample)}")
    print("Per-field exact-match accuracy (sampled val):")
    for key in sorted(field_total):
        correct = field_correct.get(key, 0)
        total = field_total[key]
        print(f"  {key}: {correct / total:.1%} ({correct}/{total})")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_dataset = CreditExtractionDataset(TRAIN_EXAMPLES, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"training on {len(TRAIN_EXAMPLES)} examples, holding out {len(VALIDATE_EXAMPLES)}")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)}: loss={outputs.loss.item():.4f}")
        print(f"epoch {epoch}: avg_loss={total_loss / len(train_loader):.4f}")

    evaluate(model, tokenizer, VALIDATE_EXAMPLES, device)

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nsaved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()

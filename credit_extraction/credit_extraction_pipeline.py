"""
Deterministic layer on top of the Qwen2.5-3B extractor from
credit_fields_finetune_qwen.py: a free-text applicant statement goes in,
the model extracts fields as JSON, and CreditRuleChecker -- the existing,
declarative implementation of decision_engine/rules/credit_rules.json
(see decision_engine/deterministic_checks.py) -- evaluates that JSON
against every rule to produce an auditable decision.

    pip install transformers torch peft
    python credit_extraction/credit_fields_finetune_qwen.py    # optional: trains qwen_credit_lora_adapter/
    python credit_extraction/credit_extraction_pipeline.py

If qwen_credit_lora_adapter/ doesn't exist locally (e.g. a fresh clone
that hasn't run the fine-tuning script), load_model() downloads it from
the Hugging Face Hub instead (HF_ADAPTER_REPO below) and caches it at
that same local path so later runs don't re-download. If that download
also fails (offline, repo not found, etc.), this falls back to the bare
Qwen2.5-3B-Instruct model -- the explicit JSON schema in the prompt
still gets you *valid* JSON most of the time, but expect materially
worse field accuracy than the fine-tuned adapter, especially on the
messier phrasing patterns credit_extraction_data.json was built to
cover.

Why the model doesn't just decide approve/reject/review by itself: its
job here is narrowly "read the numbers off the page." It never sees a
threshold, an operator, or an action_on_fail value -- every actual
underwriting decision (what counts as too little income, whether a
failure is a hard reject or a review flag) is enforced by
CreditRuleChecker against credit_rules.json. A bad extraction can
misread a field, but it can't silently redefine policy, and a policy
change is a JSON edit to the ruleset rather than a retrain.

Expect most statements below to come back HARD_FAIL, including ones a
human would call "clean": the model emits `null` for anything the text
doesn't mention, and CreditRuleChecker's rule engine treats a missing
field as a failed check for every REJECT-severity rule (see
_evaluate_rule's `if actual is None` branch in
decision_engine/deterministic_checks.py) -- it fails closed rather than
assuming an unmentioned field would have passed. That's the correct,
already-existing behavior of the deterministic layer, not a bug in this
pipeline: real free text is rarely exhaustive, and an underwriting
system shouldn't approve an application on missing data.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from credit_extraction_common import ADAPTER_DIR, MODEL_NAME, build_prompt_text

# Where load_model() downloads the adapter from if it's not present at
# ADAPTER_DIR -- e.g. on a fresh clone that hasn't run
# credit_fields_finetune_qwen.py locally yet.
HF_ADAPTER_REPO = "msxhuang68/credit-extraction-lora"

# decision_engine/ lives one directory up from credit_extraction/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decision_engine.deterministic_checks import CreditRuleChecker  # noqa: E402

APPLICANT_STATEMENTS = [
    "I'm 34 and my FICO score is 712. I make $65,000 a year working "
    "full-time as a nurse, been there 2 years. I'm a US citizen, no "
    "bankruptcies, and I have a checking account with Chase. Looking "
    "to borrow $15,000.",
    "29 years old, credit score around 610. My debt-to-income is "
    "pretty high, about 55%. I'm employed full time, salary is "
    "$42k. Requesting a $10,000 loan.",
    "I just started a new job 5 months ago, full-time, credit score "
    "690, income $40,000/year. DTI is 30%. Permanent resident. "
    "Asking for $25,000.",
    "I run my own landscaping business, been doing it for 3 years. "
    "Credit score is 740. I filed for bankruptcy 4 years ago though. "
    "I'm a US citizen with a verifiable bank account.",
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    from peft import PeftModel

    adapter_path = Path(ADAPTER_DIR)
    if adapter_path.exists():
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        print(f"loaded fine-tuned LoRA adapter from {adapter_path}")
    else:
        print(f"no local adapter at {adapter_path} -- downloading {HF_ADAPTER_REPO} from the Hugging Face Hub")
        try:
            # PeftModel.from_pretrained accepts a Hub repo id directly, no
            # separate download step needed.
            model = PeftModel.from_pretrained(base_model, HF_ADAPTER_REPO)
            model.save_pretrained(str(adapter_path))  # cache locally so the next run skips the download
            print(f"downloaded adapter from {HF_ADAPTER_REPO}, cached at {adapter_path}")
        except Exception as e:
            model = base_model
            print(
                f"could not download {HF_ADAPTER_REPO} ({e}) -- using base {MODEL_NAME} "
                "(run credit_fields_finetune_qwen.py to train one locally instead)"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def extract_applicant_record(model, tokenizer, device, text: str):
    """Run the model on one free-text statement and parse its completion
    as JSON. Returns None if the model didn't return valid JSON -- the
    LLM's output is untrusted input from here on, so this is the one
    place in the pipeline that validates rather than trusts its input.
    """
    prompt = build_prompt_text(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    completion = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    try:
        return json.loads(completion)
    except json.JSONDecodeError:
        return None


def main():
    model, tokenizer, device = load_model()
    checker = CreditRuleChecker()  # loads decision_engine/rules/credit_rules.json

    for text in APPLICANT_STATEMENTS:
        print(f"\n{text!r}")

        record = extract_applicant_record(model, tokenizer, device, text)
        if record is None:
            print("  -> model did not return valid JSON; routing to manual review")
            continue
        print(f"  extracted: {json.dumps(record)}")

        result = checker.check(record)
        status = "HARD_FAIL" if result.hard_fail else ("FLAGGED" if result.reasons else "CLEAN")
        print(f"  decision: {status}")
        for reason in result.reasons:
            print(f"    - {reason}")


if __name__ == "__main__":
    main()

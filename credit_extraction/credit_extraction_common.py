"""
Shared prompt/schema definitions for the Qwen2.5-3B credit-field
extractor, used by both the fine-tuning script
(credit_fields_finetune_qwen.py) and the inference pipeline
(credit_extraction_pipeline.py) so the two can't drift apart --
training against one prompt and then serving completions built from a
second, slightly-different copy of "the same" prompt is a classic way
to quietly tank real-world accuracy relative to eval numbers.
"""
from pathlib import Path

# Qwen2.5-3B-Instruct is the recommended size for real extraction quality
# (see the fine-tuning script's docstring), but it needs ~12GB RAM in fp32
# on CPU. Pinned to 0.5B here for local smoke-testing on machines without a
# GPU or with limited RAM -- swap back to the 3B checkpoint for real runs.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Where credit_fields_finetune_qwen.py saves its LoRA adapter, and where
# credit_extraction_pipeline.py looks for one to load. Resolved relative to
# this file (not the current working directory) so both scripts agree on
# the same path regardless of where they're invoked from. parent.parent
# because qwen_credit_lora_adapter/ lives at the repo root, one level up
# from this file's own credit_extraction/ directory.
ADAPTER_DIR = Path(__file__).parent.parent / "qwen_credit_lora_adapter"

# decision_engine/rules/credit_rules.json > RULE-EMPLOY-001 / RULE-RESIDENCY-001 > value
EMPLOYMENT_STATUS_VALUES = ["employed_full_time", "employed_part_time", "self_employed", "retired"]
RESIDENCY_STATUS_VALUES = ["US_Citizen", "Permanent_Resident"]

SYSTEM_PROMPT = f"""You extract loan applicant fields from free text into JSON.
Output ONLY a JSON object with this exact shape (no prose, no markdown fences):
{{
  "applicant": {{
    "age": <int or null>,
    "credit_score": <int 300-850 or null>,
    "annual_income_usd": <int or null>,
    "debt_to_income_ratio_percent": <number or null>,
    "employment_status": <one of {EMPLOYMENT_STATUS_VALUES} or null>,
    "current_employment_duration_months": <int or null>,
    "residency_status": <one of {RESIDENCY_STATUS_VALUES} or null>,
    "has_bankruptcy_recent": <true/false or null>,
    "has_verifiable_bank_account": <true/false or null>
  }},
  "loan_application": {{
    "requested_amount_usd": <int or null>
  }}
}}
If a field is not mentioned in the text, use null for it. Do not guess."""


def build_prompt_text(tokenizer, applicant_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": applicant_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

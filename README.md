# Credit Approval API

An underwriting pipeline that turns a free-text loan application into an
auditable **APPROVE / FLAG_REVIEW / REJECT** decision:

```
free-text applicant statement
        │
        ▼
 field extraction (LLM + regex, merged)   <- credit_extraction/
        │  ("age": 34, "credit_score": 712, ...)
        ▼
 declarative rule engine                  <- decision_engine/
        │  (decision_engine/rules/credit_rules.json)
        ▼
 decision + reasons (HARD_FAIL / FLAGGED / CLEAN)
```

The rule engine never sees raw text and the extractor never sees a
threshold — this split means a policy change is a JSON edit to the
ruleset, not a retrain, and every decision comes with a human-readable
reason trail back to the specific rule that fired.

## Repository layout

```
decision_engine/
  rules/credit_rules.json       # the credit loan ruleset (see "1. Dataset" below)
  deterministic_checks.py       # CreditRuleChecker: evaluates a record against the ruleset
  models.py                     # RuleOutcome / DeterministicCheckResult dataclasses

data_generator/
  generate_credit_extraction_data.py   # synthesizes the (text -> label) training dataset
  train_test_splitter.py               # splits it into data/train.json + data/val.json

data/
  credit_extraction_data.json   # 10,000 synthetic (text, label) pairs
  train.json / val.json         # 90/10 split of the above

credit_extraction/
  credit_extraction_common.py       # shared prompt/schema (used by training + inference)
  credit_fields_finetune_qwen.py    # LoRA fine-tunes Qwen2.5 on data/train.json
  credit_extraction_pipeline.py     # LLM-only inference -> CreditRuleChecker
  regex_field_extractor.py          # zero-dependency regex extractor (same schema)
  extraction_merge.py               # per-field merge policy for LLM + regex
  hybrid_extraction_pipeline.py     # LLM + regex merged -> CreditRuleChecker

evaluation/
  run_hybrid_on_dataset.py          # runs the hybrid pipeline over data/val.json
  evaluate_extraction_accuracy.py   # re-scores that output, no re-run needed

qwen_credit_lora_adapter/       # trained LoRA adapter (checked in / downloadable)
demo.py                         # FastAPI demo: text in, decision + reasons out
```

## Setup

Requires **Python 3.11.5** (the pinned `torch==2.14.0`/`transformers==5.16.1`
versions in `requirements.txt` need Python 3.9+; older interpreters like
the Anaconda base env's 3.7 won't work).

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`decision_engine/` itself is pure standard library. `requirements.txt`
covers everything under `credit_extraction/` (Qwen2.5 inference, LoRA
fine-tuning, `demo.py`'s FastAPI server): `torch`, `transformers`,
`peft`, `accelerate`, `fastapi`, `uvicorn`.

`credit_extraction_common.py` pins the base model to
`Qwen/Qwen2.5-0.5B-Instruct` for local, GPU-free smoke-testing; swap it
to `Qwen/Qwen2.5-3B-Instruct` for realistic extraction quality (needs a
CUDA GPU to fine-tune in reasonable time — see below).

---

## 1. Dataset

### 1a. The rules dataset

[`decision_engine/rules/credit_rules.json`](decision_engine/rules/credit_rules.json)
is the sample dataset of credit loan rules — a personal-loan
underwriting policy as a flat JSON array. Each entry is one rule that
must be satisfied:

```json
{
  "id": "RULE-CREDIT-001",
  "name": "Minimum Credit Score",
  "field": "applicant.credit_score",
  "operator": ">=",
  "value": 670,
  "action_on_fail": "REJECT",
  "severity": "CRITICAL",
  "group": "Creditworthiness"
}
```

- **field** — a dotted path into the applicant record (e.g.
  `applicant.credit_score`, `loan_application.requested_amount_usd`).
- **operator** — one of `>=`, `<=`, `>`, `<`, `==`, `!=`, `is`, `in`.
- **value** — the threshold/set to compare against (some rules instead
  use `value_field_multiplier` + `multiplier_value`, e.g. "requested
  amount must not exceed 0.5x annual income").
- **action_on_fail** — `REJECT` (hard fail) or `FLAG_REVIEW` (surfaced
  as a reason but doesn't block).
- **severity** / **group** — metadata for reporting, not evaluated.

The 10 shipped rules cover age, credit score, income, debt-to-income
ratio, employment status/duration, residency, bankruptcy history, loan
amount vs. income, and bank account verifiability.
`decision_engine/deterministic_checks.py` (`CreditRuleChecker`) is a
**domain-agnostic engine**: it only understands this schema, not credit
specifically, so swapping in a different product's or jurisdiction's
ruleset is a JSON edit, not a code change.

### 1b. The extraction training dataset

Because the rules operate on structured fields but real applications
arrive as free text, a second dataset teaches an LLM to bridge the two:
[`data/credit_extraction_data.json`](data/credit_extraction_data.json)
— 10,000 synthetic `{"text": ..., "label": {...}}` pairs, one applicant
statement paired with the exact JSON the rule engine needs.

```json
{
  "text": "I'm 34 and my FICO score is 712. I make $65,000 a year working full-time as a nurse.",
  "label": {
    "applicant": {
      "age": 34, "credit_score": 712, "annual_income_usd": 65000,
      "debt_to_income_ratio_percent": null, "employment_status": "employed_full_time",
      "current_employment_duration_months": null, "residency_status": null,
      "has_bankruptcy_recent": null, "has_verifiable_bank_account": null
    },
    "loan_application": { "requested_amount_usd": null }
  }
}
```

Generate it (already checked in, but reproducible byte-for-byte with a
fixed seed):

```bash
python data_generator/generate_credit_extraction_data.py
```

---

## 2. Tasks

### 2a. Data Preparation

**Script:** [`data_generator/generate_credit_extraction_data.py`](data_generator/generate_credit_extraction_data.py) + [`data_generator/train_test_splitter.py`](data_generator/train_test_splitter.py)

```bash
python data_generator/generate_credit_extraction_data.py   # writes data/credit_extraction_data.json
python data_generator/train_test_splitter.py               # writes data/train.json + data/val.json
```

**What it does:**

1. **Loads/builds the dataset.** Each of the 10,000 rows is generated
   field-by-field: every applicant field (age, credit score, income,
   DTI, employment, residency, bankruptcy, bank account) and the
   requested loan amount is independently included or omitted at
   random, then phrased into natural language from a pool of templates
   ("my FICO score is 712" / "credit rating of 712" / "my score's about
   712"), shuffled into a paragraph. Omitted fields are labeled `null`.
   Two deliberately hard cases are baked in rather than left to chance:
   a residency phrase that's truthful but doesn't match either policy
   enum (correct label: `null`, not a guess), and bankruptcy recency
   computed from *years-ago* to mirror the rule's "within 7 years"
   wording (bankruptcy 9 years ago must label `False`).
2. **Preprocessing / tokenization for LLM training** happens in
   [`credit_extraction/credit_fields_finetune_qwen.py`](credit_extraction/credit_fields_finetune_qwen.py):
   - Each `(text, label)` pair is rendered through Qwen's chat template
     as a 3-turn conversation (`system` prompt with the exact target
     JSON schema, `user` = the applicant text, `assistant` = the label
     as compact JSON), then tokenized.
   - **Completion-only loss masking:** the prompt tokens' labels are
     set to `-100` (`encode_with_completion_mask`) so the model is
     trained to produce the JSON completion, not to reproduce the
     (fixed) instructions — otherwise it spends capacity memorizing the
     system prompt instead of learning extraction.
   - **Special tokens:** handled by `tokenizer.apply_chat_template`,
     which inserts Qwen's ChatML role/turn markers automatically — no
     manual `[CLS]`/`[SEP]`-style token insertion is needed for a
     decoder-only chat model.
   - **Truncation:** `MAX_SEQ_LENGTH = 512` (system prompt + longest
     applicant text + JSON completion + ChatML markers tops out around
     400–420 tokens, so 512 leaves headroom without wasting compute on
     unreachable padding).
   - **Padding:** dynamic, per-batch — `collate_batch` pads every
     example in a batch up to that batch's own longest sequence (not a
     fixed global length), padding `input_ids` with the tokenizer's pad
     (or eos) id, `attention_mask` with 0s, and `labels` with `-100` so
     padding never contributes to the loss.
3. **Train/validation split:**
   [`data_generator/train_test_splitter.py`](data_generator/train_test_splitter.py)
   shuffles with a fixed seed (`SPLIT_SEED = 42`) and splits 90/10
   (`TRAIN_FRACTION = 0.9`) → 9,000 train rows / 1,000 validation rows,
   written to `data/train.json` / `data/val.json`.

**Rationale for these choices:**

- *Synthetic, template-based generation* gives exact, noise-free ground
  truth at zero labeling cost and lets specific hard cases (the
  visa-status trap, the bankruptcy-recency trap) be guaranteed present
  rather than hoped for — at the cost of limited phrasing diversity
  relative to real applicant text (see Limitations below).
- *Independently-sampled field presence* (each field included ~50–85%
  of the time, tuned per field) matches the actual input shape
  `CreditRuleChecker` has to tolerate: real free text is rarely
  exhaustive, and the checker does no input validation of its own — a
  missing field must train the model to say `null`, not to
  hallucinate a plausible value.
- *Completion-only loss* is the standard fix for instruction-tuning
  chat models on a fixed-format prompt; without it, loss on the long,
  repeated system prompt would dominate gradient signal over the much
  shorter JSON target.
- *Dynamic per-batch padding* (vs. padding every example to a single
  global max) avoids wasting compute on padding tokens across batches
  where most examples are much shorter than the longest one in the
  whole dataset.
- *Fixed seeds* throughout (`SEED = 1337` for generation, `SPLIT_SEED =
  42` for the split) make both steps fully reproducible — rerunning
  either script produces byte-identical output.

### 2b. Model Selection and Training

**Script:** [`credit_extraction/credit_fields_finetune_qwen.py`](credit_extraction/credit_fields_finetune_qwen.py)

```bash
pip install transformers torch peft accelerate
python data_generator/generate_credit_extraction_data.py   # if data/credit_extraction_data.json doesn't exist yet
python data_generator/train_test_splitter.py
python credit_extraction/credit_fields_finetune_qwen.py
```

**Model:** [Qwen2.5-Instruct](https://huggingface.co/Qwen) from Hugging
Face Transformers, as the base LLM (`MODEL_NAME` in
`credit_extraction_common.py` — pinned to the 0.5B checkpoint for
CPU-friendly smoke tests; use the 3B checkpoint for real training runs).
Chosen because it's a small, instruction-tuned, JSON-capable open model
that fine-tunes cheaply with LoRA and runs locally at inference time —
no hosted API call needed once trained.

**Training approach:** [LoRA](https://arxiv.org/abs/2106.09685)
(`peft.LoraConfig`, rank 16, alpha 32, targeting the attention
projections `q_proj`/`k_proj`/`v_proj`/`o_proj`) rather than full
fine-tuning — a few million trainable parameters instead of billions,
so it fits on a single consumer GPU and trains an epoch over 9,000 rows
quickly. 2 epochs, batch size 4, learning rate 2e-4, AdamW.

Needs a CUDA GPU with ≥16GB VRAM for the full dataset in reasonable
time; on CPU it still runs, just impractically slowly — lower `COUNT`
in the data generator and/or `EPOCHS` to smoke-test the mechanics
instead. The trained adapter is saved to `qwen_credit_lora_adapter/`.

**Why an LLM at all, rather than a classifier:** the task isn't
classification, it's *structured extraction* from unconstrained natural
language into a fixed schema — an LLM with a JSON-schema prompt handles
arbitrary phrasing far more robustly than a rule-based NER model would,
while still being cheap to fine-tune via LoRA.

### 2c. Evaluation and Analysis

**Scripts:** [`evaluation/run_hybrid_on_dataset.py`](evaluation/run_hybrid_on_dataset.py), [`evaluation/evaluate_extraction_accuracy.py`](evaluation/evaluate_extraction_accuracy.py)

```bash
python evaluation/run_hybrid_on_dataset.py [N]      # default N=100 rows of data/val.json
python evaluation/evaluate_extraction_accuracy.py    # re-scores the output above, no re-run
```

`run_hybrid_on_dataset.py` runs extraction over a sample of the
**held-out validation set** (`data/val.json` — never trained on, so
accuracy isn't inflated by memorization), for three variants: the LLM
alone, the regex extractor alone, and the two merged. For each row it
writes the text, every extractor's raw output, the merged record, which
extractor supplied each field, any LLM/regex disagreements, and the
`CreditRuleChecker` decision under the true label vs. the LLM's
extraction vs. the merged extraction. It then prints:

- **Per-field exact-match accuracy** against ground truth, for LLM
  alone, regex alone, and merged.
- **Final-decision agreement** — how often the LLM-only and merged
  pipelines land on the same `HARD_FAIL` / `FLAGGED` / `CLEAN` decision
  as the ground-truth label would produce. This is the metric that
  actually matters end-to-end: a field-level miss on a field no rule
  depends on doesn't change the decision, so decision agreement can
  exceed raw per-field accuracy.

`evaluate_extraction_accuracy.py` re-derives the LLM-alone/regex-alone
numbers from the results file already written above, with no model load
and no regex re-run — useful for re-slicing results without paying for
another full pass.

#### Sample applications and adjudication output

Run either pipeline directly to see individual decisions with full
rationale:

```bash
python credit_extraction/credit_extraction_pipeline.py       # LLM-only
python credit_extraction/hybrid_extraction_pipeline.py       # LLM + regex, merged (recommended)
python demo.py                                                # FastAPI UI/API wrapping the hybrid pipeline
```

Each of `credit_extraction_pipeline.py` / `hybrid_extraction_pipeline.py`
ships four example applicant statements
(`APPLICANT_STATEMENTS`) and prints, per statement: the extracted
fields, the decision (`HARD_FAIL` / `FLAGGED` / `CLEAN`), and the
**adjudication rationale** — every rule that failed, in plain English,
e.g.:

```
'29 years old, credit score around 610. My debt-to-income is pretty
high, about 55%. I'm employed full time, salary is $42k. Requesting a
$10,000 loan.'
  extracted: {"applicant": {"age": 29, "credit_score": 610, ...}}
  decision: HARD_FAIL
    - Minimum Credit Score: applicant.credit_score=610 fails (>= 670)
    - Maximum Debt-to-Income Ratio (DTI): applicant.debt_to_income_ratio_percent=55 fails (<= 40)
```

This is the adjudication decision plus rationale requested by the
assignment: the decision label comes from `CreditRuleChecker`
(`decision_engine/deterministic_checks.py`), and the rationale is the
specific, named rule(s) that failed and by how much — not a free-text
LLM explanation, so it's exactly reproducible from the ruleset and
auditable by a human underwriter. `demo.py` exposes the same thing over
HTTP (`POST /decide {"text": "..."}`) plus a minimal browser form.

#### Strengths and weaknesses

**Strengths**

- **Auditability.** The decision is never an LLM's free-text opinion —
  it's a deterministic evaluation of named, versioned rules. Every
  rejection cites the exact rule id, field, operator, and threshold
  that failed.
- **Fails closed on missing data.** A field the model never saw in the
  text is labeled `null`, and every `REJECT`-severity rule treats a
  missing field as a failed check (`decision_engine/deterministic_checks.py`,
  `_evaluate_rule`) — the system never approves an application on
  data it doesn't actually have.
- **Policy changes don't require a retrain.** Because the LLM only
  extracts fields and never sees a threshold, changing "minimum credit
  score 670 → 650" is a one-line JSON edit to `credit_rules.json`.
- **Hybrid extraction catches hallucination.** The regex extractor
  never invents a value — it either matches an explicit keyword-anchored
  pattern or returns `None`. Before fine-tuning, the base LLM
  hallucinated `"residency_status": "US_Citizen"` for text that never
  mentioned citizenship; merging with regex (preferring regex for
  unambiguous numeric fields, see `extraction_merge.py`) and logging
  every disagreement as a `CONFLICT` gives a free, independent
  consistency check a single-extractor pipeline doesn't get.

**Weaknesses**

- **Synthetic training data.** `credit_extraction_data.json` comes from
  a fixed set of phrasing templates, not real applicant statements —
  this demonstrates the fine-tuning mechanics at realistic *scale*, not
  production-ready training data. Expect a real-world accuracy drop on
  phrasing patterns the templates never covered.
- **Regex extractor is brittle by design.** It only catches
  keyword-anchored patterns it was explicitly written for ("credit
  score is 712"); any other phrasing of the same fact is silently
  missed (returns `None`), trading recall for zero hallucination risk.
- **Small base model.** `Qwen2.5-0.5B-Instruct` is pinned for local,
  GPU-free testing; the 3B checkpoint (recommended in
  `credit_fields_finetune_qwen.py`'s docstring) gives materially better
  real extraction quality but needs a GPU to fine-tune practically.
- **Most realistic statements will fail closed.** Real free text is
  rarely exhaustive, so most inputs come back `HARD_FAIL` purely from
  missing fields on `REJECT`-severity rules — correct behavior for an
  underwriting system, but it means a low "clean approval rate" on
  casual test input isn't itself a defect.
- **CPU latency.** Without a GPU, a single extraction takes several
  seconds to tens of seconds (`demo.py`'s docstring), which is the
  dominant cost in `evaluate_extraction_accuracy.py`'s ~17s/row figure
  for `run_hybrid_on_dataset.py`.

---

## Design notes

- **Separation of concerns:** `decision_engine/` is domain-agnostic and
  has zero ML dependencies (pure stdlib) — it evaluates any ruleset
  matching the schema in `decision_engine/rules/credit_rules.json`
  against any record. `credit_extraction/` is the only place that talks
  to a model, and its sole job is producing that record from text.
- **Shared prompt/schema source of truth:**
  `credit_extraction/credit_extraction_common.py` is imported by both
  the fine-tuning script and every inference pipeline, so training and
  serving can't silently drift onto two different prompts.
- **Adapter distribution:** if `qwen_credit_lora_adapter/` isn't present
  locally (e.g. a fresh clone that hasn't run the fine-tuning script),
  `credit_extraction_pipeline.py`'s `load_model()` downloads it from the
  Hugging Face Hub (`msxhuang68/credit-extraction-lora`) and caches it
  locally; if that also fails, it falls back to the bare base model with
  materially worse field accuracy.

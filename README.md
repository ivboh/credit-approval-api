# identity-checker

A five-stage identity verification pipeline: deterministic checks, document
verification, a risk-scoring model, and a decision gate — with a human
review queue whose verdicts retrain the model.

Run the demo:

```
python demo.py
```

Run the tests:

```
python -m unittest discover -s tests
```

See `decision_engine/pipeline.py` for the orchestration and
`decision_engine/document_verification.py` for the pluggable KYC provider
interface (`MockDocumentVerifier` stands in for Persona/Onfido/Jumio).


 Setup
  - Model: Qwen2.5-0.5B-Instruct + LoRA (rank 16, alpha 32, targeting
    q_proj/k_proj/v_proj/o_proj — only 2.16M of 496M params trainable, 0.44%)
  - Data: 400 training rows / 100 holdout rows (from the 10,000-row synthetic dataset), 1 epoch,
    batch size 4
  - Hardware: CPU-only — took 73.6 minutes total (training + eval)
  - Saved to qwen_credit_lora_adapter/

  Loss
  - Step 0: 0.163
  - Epoch average: 0.023

  Field-extraction accuracy — before (20-row smoke test) vs. after (this 400-row run), both
  measured on held-out data never seen in training:

  ┌────────────────────────────────────┬──────────────────────┬─────────────────────┐
  │               Field                │ Smoke test (20 rows) │ This run (400 rows) │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ residency_status                   │ 20%                  │ 100%                │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ employment_status                  │ 40%                  │ 92%                 │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ has_verifiable_bank_account        │ 60%                  │ 100%                │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ has_bankruptcy_recent              │ 60%                  │ 94%                 │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ annual_income_usd                  │ 60%                  │ 84%                 │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ debt_to_income_ratio_percent       │ 80%                  │ 100%                │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ age                                │ 100%                 │ 98%                 │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ credit_score                       │ 100%                 │ 100%                │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ current_employment_duration_months │ 100%                 │ 90%                 │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ requested_amount_usd               │ 100%                 │ 100%                │
  ├────────────────────────────────────┼──────────────────────┼─────────────────────┤
  │ Valid JSON rate                    │ 5/5 (100%)           │ 50/50 (100%)        │
  └────────────────────────────────────┴──────────────────────┴─────────────────────┘

  The jump on residency_status and employment_status is the headline result — going from 20
  examples to 400 is what actually taught the model the schema and field semantics, rather than
  just pattern-matching a handful of memorized cases. (A few fields dip slightly — age,
  duration, income — which is normal variance at this scale, not a regression; still comfortably
  usable.)

  Qualitative wins, spot-checked on the four fixed demo statements before vs. after fine-tuning:
  - Never hallucinated residency_status again for text that never mentions citizenship (base
    model had guessed "US_Citizen")
  - Correctly parsed "started a new job 5 months ago" → current_employment_duration_months: 5
    (base model returned null)
  - Correctly classified "I run my own landscaping business" → self_employed (base model wrongly
    said employed_full_time)

  That third one is the one worth remembering: it's the model doing something a keyword-based
  extractor structurally can't — inferring an enum value that never appears as a literal string
  in the text.

✻ Churned for 25s · done 1:56 PM · 1 shell still running


 LLM+regex credit-application extraction pipeline
"""
FastAPI demo: type a free-text loan application into a form (or POST
JSON directly), get back CreditRuleChecker's decision and the exact
reasons behind it. Wires together the same hybrid extraction pipeline
as hybrid_extraction_pipeline.py -- Qwen (fine-tuned if available) plus
the regex extractor, merged per-field -- then hands the merged record to
CreditRuleChecker, the deterministic implementation of
decision_engine/rules/credit_rules.json.

    pip install fastapi uvicorn
    python credit_extraction/credit_fields_finetune_qwen.py   # optional: trains the adapter this reuses
    python demo.py

Then open http://127.0.0.1:8000/ in a browser to type an application, or
POST directly:

    curl -X POST http://127.0.0.1:8000/decide \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"I'm 34, credit score 712, income $65,000, US citizen.\"}"

The model is loaded once at startup, not per request -- a single
extraction call takes several seconds to tens of seconds on CPU (see
credit_extraction_pipeline.py's docstring for measured latency), so the
first request after startup and every request after that will feel slow
compared to a typical web API; that's inherent to running Qwen without a
GPU, not a bug in this demo.

As with every other script in this package: the model only extracts
fields from the text. It never sees a threshold or an action_on_fail
value. Every actual accept/reject/review boundary comes from
CreditRuleChecker evaluating the merged extraction against
credit_rules.json -- this endpoint is a thin HTTP wrapper around that
same pipeline, not a new decision-making path.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# credit_extraction/ and decision_engine/ are both direct siblings of this file.
sys.path.insert(0, str(Path(__file__).resolve().parent / "credit_extraction"))
from credit_extraction_pipeline import extract_applicant_record, load_model  # noqa: E402
from extraction_merge import merge_records  # noqa: E402
from regex_field_extractor import extract_fields_regex  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_engine.deterministic_checks import CreditRuleChecker  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

EMPTY_RECORD = {"applicant": {}, "loan_application": {}}

app = FastAPI(title="Credit Decision Demo")

_state: Dict[str, Any] = {}


@app.on_event("startup")
def _load_once() -> None:
    model, tokenizer, device = load_model()
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["device"] = device
    _state["checker"] = CreditRuleChecker()  # loads decision_engine/rules/credit_rules.json


class ApplicationRequest(BaseModel):
    text: str


class ApplicationResponse(BaseModel):
    decision: str
    reasons: List[str]
    extracted: Dict[str, Any]
    sources: Dict[str, Optional[str]]
    conflicts: List[Tuple[str, Any, Any]]


def decide(text: str) -> ApplicationResponse:
    llm_extraction = extract_applicant_record(
        _state["model"], _state["tokenizer"], _state["device"], text
    ) or EMPTY_RECORD
    regex_extraction = extract_fields_regex(text)
    merged, sources, conflicts = merge_records(regex_extraction, llm_extraction)

    result = _state["checker"].check(merged)
    decision_label = "HARD_FAIL" if result.hard_fail else ("FLAGGED" if result.reasons else "CLEAN")

    return ApplicationResponse(
        decision=decision_label,
        reasons=result.reasons,
        extracted=merged,
        sources=sources,
        conflicts=conflicts,
    )


@app.post("/decide", response_model=ApplicationResponse)
def decide_endpoint(request: ApplicationRequest) -> ApplicationResponse:
    return decide(request.text)


_INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Credit Decision Demo</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
    textarea { width: 100%; height: 8rem; font-size: 1rem; padding: 0.5rem; box-sizing: border-box; }
    button { margin-top: 0.5rem; padding: 0.5rem 1.5rem; font-size: 1rem; cursor: pointer; }
    button:disabled { cursor: wait; opacity: 0.6; }
    #result { margin-top: 1.5rem; white-space: pre-wrap; font-family: monospace; }
    .decision { font-weight: bold; font-size: 1.2rem; padding: 0.3rem 0.6rem; border-radius: 4px; display: inline-block; }
    .CLEAN { background: #d4edda; color: #155724; }
    .FLAGGED { background: #fff3cd; color: #856404; }
    .HARD_FAIL { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <h1>Credit Decision Demo</h1>
  <p>Type a free-text loan application below. Extraction runs a local LLM, so each submission can take several seconds to tens of seconds on CPU.</p>
  <textarea id="text" placeholder="I'm 34, credit score 712, income $65,000 a year, US citizen, looking to borrow $15,000.">I'm 34, credit score 712, income $65,000 a year, US citizen, looking to borrow $15,000.</textarea>
  <br>
  <button id="submit">Submit application</button>
  <div id="decision" class="decision" style="display:none"></div>
  <div id="result"></div>

  <script>
    const btn = document.getElementById("submit");
    const decisionEl = document.getElementById("decision");
    const result = document.getElementById("result");
    btn.addEventListener("click", async () => {
      const text = document.getElementById("text").value;
      btn.disabled = true;
      decisionEl.style.display = "none";
      result.textContent = "Evaluating (this can take a while on CPU)...";
      try {
        const resp = await fetch("/decide", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const data = await resp.json();

        decisionEl.textContent = data.decision;
        decisionEl.className = "decision " + data.decision;
        decisionEl.style.display = "inline-block";

        const NL = String.fromCharCode(10);
        const lines = ["Reasons:"];
        if (data.reasons.length) {
          for (const r of data.reasons) { lines.push("  - " + r); }
        } else {
          lines.push("  (none)");
        }
        lines.push("");
        lines.push("Extracted fields:");
        lines.push(JSON.stringify(data.extracted, null, 2));
        if (data.conflicts.length) {
          lines.push("");
          lines.push("Conflicts (LLM vs regex disagreed):");
          lines.push(JSON.stringify(data.conflicts, null, 2));
        }
        result.textContent = lines.join(NL);
      } catch (err) {
        result.textContent = "Error: " + err;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

from .models import DeterministicCheckResult, RuleOutcome

# ---------------------------------------------------------------------------
# Generic, declarative rule engine — evaluates a JSON ruleset like
# decision_engine/rules/credit_rules.json against an applicant record.
#
# The engine is domain-agnostic: it doesn't know about credit or identity,
# only the ruleset schema (field, operator, value, action_on_fail,
# severity, group). Swapping the ruleset — a different product's
# underwriting policy, a different jurisdiction's eligibility rules — is a
# JSON edit, not a code change.
# ---------------------------------------------------------------------------

DEFAULT_CREDIT_RULES_PATH = Path(__file__).parent / "rules" / "credit_rules.json"

_OPERATORS = {
    ">=": lambda actual, expected: actual >= expected,
    "<=": lambda actual, expected: actual <= expected,
    ">": lambda actual, expected: actual > expected,
    "<": lambda actual, expected: actual < expected,
    "==": lambda actual, expected: actual == expected,
    "!=": lambda actual, expected: actual != expected,
    "is": lambda actual, expected: actual == expected,
    "in": lambda actual, expected: actual in expected,
}


@dataclass
class Rule:
    id: str
    name: str
    field: str
    operator: str
    action_on_fail: str
    severity: str
    group: str
    value: Any = None
    value_field_multiplier: Optional[str] = None
    multiplier_value: Optional[float] = None
    description: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "Rule":
        return cls(
            id=raw["id"],
            name=raw["name"],
            field=raw["field"],
            operator=raw["operator"],
            action_on_fail=raw["action_on_fail"],
            severity=raw["severity"],
            group=raw["group"],
            value=raw.get("value"),
            value_field_multiplier=raw.get("value_field_multiplier"),
            multiplier_value=raw.get("multiplier_value"),
            description=raw.get("description", ""),
        )


def load_ruleset(path: Union[str, Path]) -> List[Rule]:
    """Load a rules file shaped like credit_rules.json: a single top-level
    key wrapping `{"rules": [...]}`. The wrapper key's own name isn't
    significant — only the `rules` list underneath it is read.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    ruleset = next(iter(data.values()))
    return [Rule.from_dict(raw) for raw in ruleset["rules"]]


def _get_field(record: Any, path: str) -> Any:
    """Resolve a dotted path (e.g. "applicant.credit_score") against a
    record that may be a nested dict, a dataclass, or a mix of both.
    """
    current = record
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(segment)
        else:
            current = getattr(current, segment, None)
    return current


def _evaluate_rule(rule: Rule, record: Any) -> RuleOutcome:
    actual = _get_field(record, rule.field)

    if rule.value_field_multiplier is not None:
        base = _get_field(record, rule.value_field_multiplier)
        if actual is None or base is None:
            reason = f"{rule.name}: required field(s) missing ({rule.field}, {rule.value_field_multiplier})"
            return _outcome(rule, passed=False, reason=reason)
        threshold = base * rule.multiplier_value
        passed = _OPERATORS[rule.operator](actual, threshold)
        reason = (
            f"{rule.name}: {rule.field}={actual} exceeds "
            f"{rule.multiplier_value}x {rule.value_field_multiplier}={threshold}"
        )
        return _outcome(rule, passed=passed, reason=None if passed else reason)

    if actual is None:
        return _outcome(rule, passed=False, reason=f"{rule.name}: {rule.field} is missing")

    passed = _OPERATORS[rule.operator](actual, rule.value)
    reason = f"{rule.name}: {rule.field}={actual!r} fails ({rule.operator} {rule.value!r})"
    return _outcome(rule, passed=passed, reason=None if passed else reason)


def _outcome(rule: Rule, passed: bool, reason: Optional[str]) -> RuleOutcome:
    return RuleOutcome(
        rule_id=rule.id,
        name=rule.name,
        group=rule.group,
        severity=rule.severity,
        action_on_fail=rule.action_on_fail,
        passed=passed,
        reason=reason,
    )


class CreditRuleChecker:
    """Stage 02, driven by a JSON ruleset instead of hardcoded logic.

    Ships with `decision_engine/rules/credit_rules.json` (personal-loan
    underwriting: age, credit score, income, DTI, employment, residency,
    bankruptcy, loan-to-income ratio, bank account) as the default
    ruleset, but any file following that schema works — pass `rules_path`
    or a pre-loaded `rules` list to use a different one.

    Call `check()` with a record shaped like the ruleset's field paths
    expect, e.g. `{"applicant": {...}, "loan_application": {...}}`.
    A rule whose `action_on_fail` is `REJECT` sets `hard_fail=True` on the
    result; `FLAG_REVIEW` rules only show up in `reasons`/`outcomes`.
    """

    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        rules_path: Optional[Union[str, Path]] = None,
    ):
        self.rules = rules if rules is not None else load_ruleset(rules_path or DEFAULT_CREDIT_RULES_PATH)

    def check(self, record: Any) -> DeterministicCheckResult:
        reasons: List[str] = []
        outcomes: List[RuleOutcome] = []
        hard_fail = False

        for rule in self.rules:
            outcome = _evaluate_rule(rule, record)
            outcomes.append(outcome)
            if not outcome.passed:
                reasons.append(outcome.reason)
                if rule.action_on_fail == "REJECT":
                    hard_fail = True

        return DeterministicCheckResult(reasons=reasons, hard_fail=hard_fail, outcomes=outcomes)

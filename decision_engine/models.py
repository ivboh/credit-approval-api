from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RuleOutcome:
    """Per-rule result from a declarative ruleset (see
    `deterministic_checks.CreditRuleChecker`) — kept alongside the plain
    `reasons` strings so a caller can distinguish which rule fired, at
    what severity, and whether it was a hard reject or a review flag.
    """

    rule_id: str
    name: str
    group: str
    severity: str
    action_on_fail: str
    passed: bool
    reason: Optional[str] = None


@dataclass
class DeterministicCheckResult:
    """`CreditRuleChecker.check()`'s return type. `hard_fail` means at
    least one fired rule has `action_on_fail: REJECT`; a rule with
    `FLAG_REVIEW` only shows up in `reasons`/`outcomes`.
    """

    reasons: List[str] = field(default_factory=list)
    hard_fail: bool = False
    outcomes: List[RuleOutcome] = field(default_factory=list)

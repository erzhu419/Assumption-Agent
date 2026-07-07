"""Typed Hypothesis object for Exp 82 v2 ablation.

Mirrors MC-WM's `mc_wm/self_audit/hypothesis.py` design but adapted to the
problem-solving domain: each Hypothesis is a falsifiable claim that adding a
specific structure (feature / constraint / decomposition / verification /
hp_change) to the solver pipeline produces a measurable correctness gain on a
problem subset.

Persistence: append-only JSONL at phase six/exp82/hypotheses.jsonl
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


HYPO_DIR = Path(__file__).parent
HYPO_LOG = HYPO_DIR / "hypotheses.jsonl"


KINDS = ("feature", "constraint", "decomposition", "verification", "hp_change")


@dataclass
class Hypothesis:
    """One falsifiable claim about a candidate (wisdom-derived) structure.

    Fields:
        hid:               12-char hex id, assigned at creation.
        seed_cid:          which v1 wisdom (e.g. WCAND10) this was derived from.
        kind:              one of KINDS.
        claim:             one-sentence English-or-Chinese description of the
                           hypothesis ("inject 'list assumptions before answering'
                           as decomposition step 1 because seed wisdom signals
                           'investigation-first'").
        expr:              kind-specific actionable payload.
                           - feature:        regex / keyword list / classifier id
                           - constraint:    predicate (regex / required-substring) + retry policy
                           - decomposition: an ordered list of step descriptions
                           - verification:  a post-answer check instruction
                           - hp_change:     dict like {"temperature": 0.0, "max_tokens": 2000}
        trigger_subset:    list of pids where seed_cid was labelled SHOULD_FIRE
                           in exp17 (the cell on which we A/B test).
        outside_subset:    list of pids where NO_FIRE — for specificity check.
        expected_metric:   "correctness" (graded vs gold) for this experiment.
        expected_direction: "increase".
        expected_min_delta: 0.05 (5pp on trigger_subset).
        evidence:          dict filled in by the evaluator after A/B test:
                           {
                               "n_trigger": int,
                               "base_correct": int,
                               "ext_correct": int,
                               "generic_correct": int (optional),
                               "delta_ext_base": float,
                               "delta_ext_generic": float (optional),
                               "outside_n": int,
                               "outside_delta_ext_base": float,
                               "judge_wr_ext_vs_base": float (optional),
                           }
        decision:          "accepted" | "rejected" | "deferred"
        failure_reason:    "insignificant" | "destabilising" | "outside_drop" | "trigger_miss" | None
        timestamp:         ISO-8601.
    """

    seed_cid: str
    kind: str
    claim: str
    expr: object  # str or list or dict, kind-dependent
    trigger_subset: list = field(default_factory=list)
    outside_subset: list = field(default_factory=list)

    expected_metric: str = "correctness"
    expected_direction: str = "increase"
    expected_min_delta: float = 0.05

    hid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    evidence: dict = field(default_factory=dict)
    decision: str = "deferred"
    failure_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; must be one of {KINDS}")

    def record_outcome(self, evidence: dict, decision: str,
                       failure_reason: Optional[str] = None) -> None:
        """Set post-test fields in-place."""
        self.evidence.update(evidence)
        self.decision = decision
        self.failure_reason = failure_reason

    def persist(self, path: Path = HYPO_LOG) -> None:
        """Append one JSONL line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(self), ensure_ascii=False) + "\n")

    def to_dict(self) -> dict:
        return asdict(self)


def load_all(path: Path = HYPO_LOG) -> list[Hypothesis]:
    """Read JSONL log and rehydrate Hypothesis objects."""
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Hypothesis(**d))
    return out


def accept_decide(evidence: dict, min_delta: float = 0.05,
                    outside_drop_tol: float = -0.05) -> tuple[str, Optional[str]]:
    """Apply the v2 acceptance rule.

    accepted iff:
      - delta_ext_base >= min_delta on trigger subset
      - outside_delta_ext_base >= outside_drop_tol (no big drop outside)

    Returns (decision, failure_reason).
    """
    d = evidence.get("delta_ext_base")
    if d is None:
        return "deferred", "trigger_miss"
    out = evidence.get("outside_delta_ext_base", 0.0)
    if d < min_delta:
        return "rejected", "insignificant"
    if out < outside_drop_tol:
        return "rejected", "outside_drop"
    return "accepted", None


if __name__ == "__main__":
    # smoke test
    h = Hypothesis(
        seed_cid="WCAND10",
        kind="feature",
        claim="problem text mentions 'symptoms' or 'multiple variables tangled'",
        expr={"keywords_zh": ["症状", "纠缠", "模糊", "多个"], "keywords_en": []},
        trigger_subset=["engineering_0199", "business_0062"],
        outside_subset=["mathematics_0043"],
    )
    print(json.dumps(h.to_dict(), ensure_ascii=False, indent=2))
    print("kind valid?", h.kind in KINDS)

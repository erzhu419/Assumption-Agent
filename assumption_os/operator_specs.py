"""Compile retrieved assumptions into answer-time operator constraints.

The graph stores assumptions as auditable claims.  Solver prompts need a more
procedural shape: trigger, steps, output slots, negative controls, and checks.
This module keeps that conversion deterministic and small so experiments can
measure whether assumptions were merely retrieved or actually made executable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .schema import AssumptionNode, AssumptionType


@dataclass(frozen=True)
class OperatorSpec:
    source_id: str
    source_type: str
    source_claim: str
    trigger_conditions: list[str]
    execution_steps: list[str]
    required_output_slots: list[str]
    negative_controls: list[str]
    verifier_checks: list[str]
    fallback_policy: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorGateDecision:
    enabled: bool
    status: str
    reason: str
    domain: str
    allowed_domains: list[str]
    skipped_domains: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OPERATOR_SOURCE_TYPES = {
    AssumptionType.METHOD.value,
    AssumptionType.HARNESS.value,
    AssumptionType.RETRIEVAL.value,
    AssumptionType.WORLD_MODEL.value,
    AssumptionType.EVALUATOR.value,
    AssumptionType.ALIGNMENT.value,
    AssumptionType.FRAMEWORK.value,
    AssumptionType.FRAMEWORK_BRANCH.value,
    AssumptionType.STRATEGY.value,
}


def operator_gate_decision(
    domain: str,
    *,
    enabled: bool = True,
    allowed_domains: set[str] | None = None,
    skipped_domains: set[str] | None = None,
) -> OperatorGateDecision:
    allowed = _normalize_domain_set(allowed_domains)
    skipped = _normalize_domain_set(skipped_domains)
    dom = str(domain or "").strip()
    if not enabled:
        return OperatorGateDecision(
            enabled=False,
            status="disabled",
            reason="disabled_by_cli",
            domain=dom,
            allowed_domains=sorted(allowed),
            skipped_domains=sorted(skipped),
        )
    if dom in skipped:
        return OperatorGateDecision(
            enabled=False,
            status="skipped",
            reason="domain_in_operator_skip_list",
            domain=dom,
            allowed_domains=sorted(allowed),
            skipped_domains=sorted(skipped),
        )
    if allowed and dom not in allowed:
        return OperatorGateDecision(
            enabled=False,
            status="not_selected",
            reason="domain_not_in_operator_allow_list",
            domain=dom,
            allowed_domains=sorted(allowed),
            skipped_domains=sorted(skipped),
        )
    return OperatorGateDecision(
        enabled=True,
        status="enabled",
        reason="domain_selected",
        domain=dom,
        allowed_domains=sorted(allowed),
        skipped_domains=sorted(skipped),
    )


def build_operator_specs(nodes: Iterable[AssumptionNode], *, max_specs: int = 2) -> list[OperatorSpec]:
    specs: list[OperatorSpec] = []
    for node in nodes:
        source_type = str(getattr(node.type, "value", node.type))
        if source_type not in OPERATOR_SOURCE_TYPES and not _explicit_operator_payload(node):
            continue
        spec = operator_spec_from_node(node)
        if spec:
            specs.append(spec)
        if len(specs) >= max_specs:
            break
    return specs


def operator_spec_from_node(node: AssumptionNode) -> OperatorSpec | None:
    explicit = _explicit_operator_payload(node)
    formal_expr = _formal_expr(node)
    trigger_conditions = _dedupe_nonempty([
        *_list_field(explicit, "trigger_conditions"),
        *_list_field(explicit, "trigger"),
        *node.context_conditions[:4],
    ])
    execution_steps = _dedupe_nonempty([
        *_list_field(explicit, "execution_steps"),
        *_list_field(explicit, "steps"),
        *_list_field(formal_expr, "steps"),
    ])
    required_output_slots = _dedupe_nonempty([
        *_list_field(explicit, "required_output_slots"),
        *_list_field(explicit, "output_slots"),
        *_list_field(formal_expr, "required_output_slots"),
    ])
    negative_controls = _dedupe_nonempty([
        *_list_field(explicit, "negative_controls"),
        *_list_field(explicit, "anti_patterns"),
        *[f"Avoid: {risk}" for risk in node.risk_predictions[:3]],
    ])
    verifier_checks = _dedupe_nonempty([
        *_list_field(explicit, "verifier_checks"),
        *_list_field(explicit, "verifiers"),
        *_list_field(formal_expr, "verifier_checks"),
        *([str(formal_expr["instruction"])] if formal_expr.get("instruction") else []),
        *node.verifiers[:4],
    ])
    fallback_policy = str(
        explicit.get("fallback_policy")
        or explicit.get("fallback")
        or formal_expr.get("fallback_policy")
        or ""
    ).strip()

    heuristic = _heuristic_operator(node)
    trigger_conditions = trigger_conditions or heuristic["trigger_conditions"]
    execution_steps = execution_steps or heuristic["execution_steps"]
    required_output_slots = required_output_slots or heuristic["required_output_slots"]
    negative_controls = negative_controls or heuristic["negative_controls"]
    verifier_checks = verifier_checks or heuristic["verifier_checks"]
    fallback_policy = fallback_policy or heuristic["fallback_policy"]

    if not execution_steps and not required_output_slots and not verifier_checks:
        return None
    return OperatorSpec(
        source_id=node.id,
        source_type=str(getattr(node.type, "value", node.type)),
        source_claim=node.claim,
        trigger_conditions=trigger_conditions[:5],
        execution_steps=execution_steps[:6],
        required_output_slots=required_output_slots[:6],
        negative_controls=negative_controls[:4],
        verifier_checks=verifier_checks[:5],
        fallback_policy=fallback_policy,
        confidence=float(getattr(node, "confidence", 0.0) or 0.0),
    )


def format_operator_specs(specs: list[OperatorSpec], *, max_specs: int = 2) -> str:
    if not specs:
        return ""
    lines = [
        "## Operatorized Assumption Constraints",
        "These are execution constraints, not background context. Use only operators whose trigger fits the problem. If an operator is used, satisfy its required slots in the answer instead of merely naming the source assumption.",
    ]
    for index, spec in enumerate(specs[:max_specs], start=1):
        lines.append(f"\n[OP{index}] {spec.source_id} ({spec.source_type}, confidence={spec.confidence:.2f})")
        if spec.trigger_conditions:
            lines.append("Trigger: " + "; ".join(spec.trigger_conditions))
        if spec.execution_steps:
            lines.append("Execution steps:")
            lines.extend(f"  {step_index}. {step}" for step_index, step in enumerate(spec.execution_steps, start=1))
        if spec.required_output_slots:
            lines.append("Required answer slots: " + "; ".join(spec.required_output_slots))
        if spec.negative_controls:
            lines.append("Negative controls: " + "; ".join(spec.negative_controls))
        if spec.verifier_checks:
            lines.append("Verifier checks: " + "; ".join(spec.verifier_checks))
        if spec.fallback_policy:
            lines.append("Fallback policy: " + spec.fallback_policy)
    return "\n".join(lines).strip()


def operator_trace_summary(specs: list[OperatorSpec]) -> dict[str, Any]:
    return {
        "operator_count": len(specs),
        "operator_source_ids": [spec.source_id for spec in specs],
        "operator_source_types": [spec.source_type for spec in specs],
        "operator_specs": [spec.to_dict() for spec in specs],
        "required_slot_count": sum(len(spec.required_output_slots) for spec in specs),
        "verifier_check_count": sum(len(spec.verifier_checks) for spec in specs),
    }


def _explicit_operator_payload(node: AssumptionNode) -> dict[str, Any]:
    payload = node.payload if isinstance(node.payload, dict) else {}
    operator = payload.get("operator_spec")
    if isinstance(operator, dict):
        return operator
    if any(
        key in payload
        for key in (
            "trigger_conditions",
            "execution_steps",
            "required_output_slots",
            "negative_controls",
            "verifier_checks",
            "fallback_policy",
        )
    ):
        return payload
    return {}


def _formal_expr(node: AssumptionNode) -> dict[str, Any]:
    formal = node.formal_form if isinstance(node.formal_form, dict) else {}
    expr = formal.get("expr")
    return expr if isinstance(expr, dict) else {}


def _list_field(mapping: dict[str, Any], key: str) -> list[str]:
    if not mapping:
        return []
    value = mapping.get(key)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dedupe_nonempty(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _heuristic_operator(node: AssumptionNode) -> dict[str, Any]:
    text = " ".join([
        node.id,
        str(getattr(node.type, "value", node.type)),
        str(getattr(node.kind, "value", node.kind)),
        node.claim,
        " ".join(node.context_conditions),
        " ".join(node.tags),
    ]).lower()
    if _contains_any(text, ["control variable", "controlled variable", "控制变量", "causal", "因果", "ablation", "单因素", "对照"]):
        return {
            "trigger_conditions": ["Causal attribution is uncertain or several factors could explain the outcome."],
            "execution_steps": [
                "List the candidate variables or causes before recommending an action.",
                "Choose one variable or intervention to change.",
                "State which variables, data slice, environment, or baseline stay fixed.",
                "Define the metric and control comparison.",
                "Name the confound or dependency that would falsify the conclusion.",
            ],
            "required_output_slots": [
                "variable_or_cause_changed",
                "variables_held_constant",
                "control_or_baseline",
                "observed_metric",
                "decision_rule",
            ],
            "negative_controls": ["Do not recommend changing many causal factors at once unless dependencies are stated first."],
            "verifier_checks": ["The answer contains a changed variable, held-constant variables, a metric, a control, and a decision rule."],
            "fallback_policy": "If variables are coupled, switch to a dependency-aware intervention plan.",
        }
    if _contains_any(text, ["incremental", "strangler", "legacy", "重构", "迁移", "替换", "adapter", "适配", "mvp"]):
        return {
            "trigger_conditions": ["A system must change without breaking existing behavior or ownership boundaries."],
            "execution_steps": [
                "Name the preserved behavior, interface, or user workflow.",
                "Define the smallest reversible slice or adapter boundary.",
                "Add characterization tests, telemetry, or acceptance checks before replacement.",
                "Plan rollout, rollback, and the stopping condition for the next slice.",
            ],
            "required_output_slots": [
                "preserved_behavior",
                "incremental_slice",
                "adapter_or_boundary",
                "acceptance_metric",
                "rollback_path",
            ],
            "negative_controls": ["Do not propose a full rewrite before preserving behavior and rollback criteria."],
            "verifier_checks": ["The answer includes a reversible slice, boundary, metric, and rollback path."],
            "fallback_policy": "If the boundary is unknown, run a discovery spike before committing to migration.",
        }
    if _contains_any(text, ["morphism", "isomorphism", "analogy", "类比", "structure", "pattern", "框架", "generalization", "泛化"]):
        return {
            "trigger_conditions": ["A prior framework or pattern is being transferred to a new problem."],
            "execution_steps": [
                "Map source roles to target roles explicitly.",
                "State the invariant that must be preserved.",
                "State the limiting case where the old framework should reappear.",
                "Identify a negative control where the transfer should fail.",
            ],
            "required_output_slots": [
                "source_roles",
                "target_roles",
                "preserved_invariant",
                "limiting_case",
                "negative_control",
            ],
            "negative_controls": ["Do not transfer surface vocabulary without role and invariant preservation."],
            "verifier_checks": ["The answer gives role mapping, invariant, limiting case, and a failure case."],
            "fallback_policy": "If role mapping is weak, treat the analogy as inspiration rather than evidence.",
        }
    if str(getattr(node.type, "value", node.type)) == AssumptionType.RETRIEVAL.value or _contains_any(text, ["retrieval", "evidence", "source", "检索", "证据"]):
        return {
            "trigger_conditions": ["The answer depends on external evidence, precedent, or source grounding."],
            "execution_steps": [
                "Separate answer-bearing evidence from merely topical evidence.",
                "Tie each decisive claim to the exact entity, scope, date, or relation requested.",
                "State what evidence would overturn the answer.",
            ],
            "required_output_slots": [
                "decisive_evidence",
                "entity_scope_boundary",
                "answer_bearing_relation",
                "overturn_condition",
            ],
            "negative_controls": ["Do not use generic topical retrieval as support for the final answer."],
            "verifier_checks": ["Evidence is answer-bearing rather than merely related."],
            "fallback_policy": "If evidence is generic or conflicting, abstain or narrow the claim.",
        }
    return {
        "trigger_conditions": ["The retrieved assumption directly constrains the current problem."],
        "execution_steps": [
            "State the concrete trigger from the problem.",
            "Apply the assumption as a decision constraint or execution step.",
            "Name the verifier or failure condition that would change the answer.",
        ],
        "required_output_slots": [
            "trigger",
            "applied_constraint",
            "evidence_or_boundary",
            "verifier_or_failure_condition",
        ],
        "negative_controls": ["Do not merely restate the assumption as advice."],
        "verifier_checks": ["The answer changes content or structure because of the assumption."],
        "fallback_policy": "If the trigger does not fit, ignore the assumption rather than forcing it.",
    }


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _normalize_domain_set(values: set[str] | None) -> set[str]:
    out = {str(value).strip() for value in (values or set()) if str(value).strip()}
    if {value.lower() for value in out} & {"*", "all"}:
        return set()
    return out

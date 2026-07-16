"""Finite TRAIN-only evaluator formation and prospective search utility.

Evaluator rules rank typed retriever programs using only a declared formation
block.  A fixed official-label anchor may then accept or reject the challenger,
and a third formation/measurement pair measures whether the new evaluator
actually improves candidate search.  The official support objective remains
independent and is never replaced by the learned evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from ..archive import AnchorScore, wilson_lower_bound
from ..models import stable_hash


VERSION = "musique_evaluator_coevolution_v1"
FOLD_COUNT = 4
INCUMBENT_RULE_ID = "micro_recall_v1"
ANCHOR_CONFIDENCE = 0.9


class EvaluatorFormationError(RuntimeError):
    """Raised when evaluator formation or anchor custody is ambiguous."""


@dataclass(frozen=True)
class ItemRetrievalEvidence:
    item_commitment_sha256: str
    support_hits: int
    support_total: int
    invalid: bool
    retrieval_sha256: str

    @property
    def recall(self) -> Fraction:
        return Fraction(self.support_hits, self.support_total)

    def validate(self) -> "ItemRetrievalEvidence":
        for value, field in (
            (self.item_commitment_sha256, "item commitment"),
            (self.retrieval_sha256, "retrieval"),
        ):
            if (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise EvaluatorFormationError(f"{field} must be lowercase sha256")
        if (
            type(self.support_hits) is not int
            or type(self.support_total) is not int
            or self.support_total <= 0
            or self.support_hits < 0
            or self.support_hits > self.support_total
            or type(self.invalid) is not bool
        ):
            raise EvaluatorFormationError("retrieval evidence counts are invalid")
        if self.invalid and self.support_hits != 0:
            raise EvaluatorFormationError("invalid retrieval may not receive support hits")
        return self


@dataclass(frozen=True)
class ProgramRetrievalEvidence:
    program_sha256: str
    program_length: int
    items: tuple[ItemRetrievalEvidence, ...]

    def validate(self) -> "ProgramRetrievalEvidence":
        if (
            len(self.program_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.program_sha256
            )
        ):
            raise EvaluatorFormationError("program hash must be lowercase sha256")
        if type(self.program_length) is not int or self.program_length <= 0:
            raise EvaluatorFormationError("program length must be positive")
        if len(self.items) < FOLD_COUNT:
            raise EvaluatorFormationError("too few evaluator formation items")
        commitments = []
        for item in self.items:
            item.validate()
            commitments.append(item.item_commitment_sha256)
        if len(set(commitments)) != len(commitments):
            raise EvaluatorFormationError("duplicate item commitment")
        return self


@dataclass(frozen=True)
class EvaluatorRule:
    id: str
    micro_weight: int = 0
    macro_weight: int = 0
    complete_weight: int = 0
    worst_weight: int = 0
    invalid_penalty: int = 1

    def validate(self) -> "EvaluatorRule":
        weights = (
            self.micro_weight,
            self.macro_weight,
            self.complete_weight,
            self.worst_weight,
        )
        if not self.id or any(type(value) is not int or value < 0 for value in weights):
            raise EvaluatorFormationError("evaluator rule weights are invalid")
        if sum(weights) <= 0 or self.invalid_penalty < 0:
            raise EvaluatorFormationError("evaluator rule is empty or has negative penalty")
        return self

    @property
    def complexity(self) -> int:
        return sum(
            value != 0
            for value in (
                self.micro_weight,
                self.macro_weight,
                self.complete_weight,
                self.worst_weight,
                self.invalid_penalty,
            )
        )

    @property
    def rule_sha256(self) -> str:
        return stable_hash(asdict(self.validate()))

    def score(
        self,
        evidence: ProgramRetrievalEvidence,
        positions: Sequence[int],
    ) -> Fraction:
        self.validate()
        evidence.validate()
        if not positions or any(
            type(position) is not int
            or position < 0
            or position >= len(evidence.items)
            for position in positions
        ):
            raise EvaluatorFormationError("evaluator positions are invalid")
        rows = [evidence.items[position] for position in positions]
        total_hits = sum(row.support_hits for row in rows)
        total_labels = sum(row.support_total for row in rows)
        micro = Fraction(total_hits, total_labels)
        macro = sum((row.recall for row in rows), Fraction()) / len(rows)
        complete = Fraction(sum(row.support_hits == row.support_total for row in rows), len(rows))
        worst = min(row.recall for row in rows)
        invalid = sum(row.invalid for row in rows)
        return (
            self.micro_weight * micro
            + self.macro_weight * macro
            + self.complete_weight * complete
            + self.worst_weight * worst
            - self.invalid_penalty * invalid
        )


def evaluator_rules() -> tuple[EvaluatorRule, ...]:
    """Return the complete fixed evaluator DSL in deterministic order."""

    return (
        EvaluatorRule(INCUMBENT_RULE_ID, micro_weight=1),
        EvaluatorRule("macro_recall_v1", macro_weight=1),
        EvaluatorRule("complete_chain_v1", complete_weight=1),
        EvaluatorRule("micro_macro_v1", micro_weight=1, macro_weight=1),
        EvaluatorRule("micro_complete_v1", micro_weight=1, complete_weight=1),
        EvaluatorRule("macro_complete_v1", macro_weight=1, complete_weight=1),
        EvaluatorRule(
            "micro_macro_complete_v1",
            micro_weight=1,
            macro_weight=1,
            complete_weight=1,
        ),
        EvaluatorRule("micro_worst_v1", micro_weight=1, worst_weight=1),
        EvaluatorRule("macro_worst_v1", macro_weight=1, worst_weight=1),
        EvaluatorRule(
            "balanced_chain_v1",
            micro_weight=2,
            macro_weight=1,
            complete_weight=1,
            worst_weight=1,
        ),
    )


def _validate_evidence_set(
    evidences: Sequence[ProgramRetrievalEvidence],
) -> tuple[ProgramRetrievalEvidence, ...]:
    normalized = tuple(evidence.validate() for evidence in evidences)
    if not normalized:
        raise EvaluatorFormationError("candidate evidence set is empty")
    program_hashes = [row.program_sha256 for row in normalized]
    if len(set(program_hashes)) != len(program_hashes):
        raise EvaluatorFormationError("duplicate program evidence")
    item_order = tuple(
        item.item_commitment_sha256 for item in normalized[0].items
    )
    if any(
        tuple(item.item_commitment_sha256 for item in row.items) != item_order
        for row in normalized[1:]
    ):
        raise EvaluatorFormationError("candidate item sets or order differ")
    return normalized


def _evidence_set_sha256(
    evidences: Sequence[ProgramRetrievalEvidence],
) -> str:
    normalized = _validate_evidence_set(evidences)
    return stable_hash(
        [
            {
                "program_sha256": program.program_sha256,
                "program_length": program.program_length,
                "items": [
                    {
                        "item_commitment_sha256": item.item_commitment_sha256,
                        "support_hits": item.support_hits,
                        "support_total": item.support_total,
                        "invalid": item.invalid,
                        "retrieval_sha256": item.retrieval_sha256,
                    }
                    for item in program.items
                ],
            }
            for program in normalized
        ]
    )


def _verify_formation_receipt(
    receipt: Mapping[str, Any],
    evidence: Sequence[ProgramRetrievalEvidence],
) -> tuple[EvaluatorRule, EvaluatorRule, str]:
    body = dict(receipt)
    declared = body.pop("formation_sha256", None)
    if (
        receipt.get("schema") != f"{VERSION}_formation_receipt"
        or receipt.get("partition") != "A_form"
        or stable_hash(body) != declared
        or receipt.get("anchor_accessed") is not False
        or receipt.get("measurement_accessed") is not False
        or receipt.get("evidence_set_sha256")
        != _evidence_set_sha256(evidence)
    ):
        raise EvaluatorFormationError("evaluator formation receipt mismatch")
    rules = {rule.id: rule for rule in evaluator_rules()}
    try:
        incumbent = rules[str(receipt["incumbent_rule"]["id"])]
        challenger = rules[str(receipt["challenger_rule"]["id"])]
    except (KeyError, TypeError) as exc:
        raise EvaluatorFormationError("evaluator rule identity is invalid") from exc
    if (
        dict(receipt["incumbent_rule"]) != asdict(incumbent)
        or dict(receipt["challenger_rule"]) != asdict(challenger)
        or incumbent.id != INCUMBENT_RULE_ID
        or challenger.id == incumbent.id
    ):
        raise EvaluatorFormationError("evaluator rule payload drifted")
    incumbent_program = select_program(incumbent, evidence)
    challenger_program = select_program(challenger, evidence)
    if (
        receipt.get("incumbent_selected_program_sha256")
        != incumbent_program.program_sha256
        or receipt.get("challenger_selected_program_sha256")
        != challenger_program.program_sha256
    ):
        raise EvaluatorFormationError("evaluator-selected program drifted")
    return incumbent, challenger, str(declared)


def select_program(
    rule: EvaluatorRule,
    evidences: Sequence[ProgramRetrievalEvidence],
    positions: Sequence[int] | None = None,
) -> ProgramRetrievalEvidence:
    candidates = _validate_evidence_set(evidences)
    selected_positions = (
        tuple(range(len(candidates[0].items)))
        if positions is None
        else tuple(positions)
    )
    return min(
        candidates,
        key=lambda row: (
            -rule.score(row, selected_positions),
            sum(row.items[position].invalid for position in selected_positions),
            row.program_length,
            row.program_sha256,
        ),
    )


def _crossfit_rule(
    rule: EvaluatorRule,
    evidences: Sequence[ProgramRetrievalEvidence],
) -> dict[str, Any]:
    candidates = _validate_evidence_set(evidences)
    item_count = len(candidates[0].items)
    held_hits = 0
    held_total = 0
    folds: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        held_positions = tuple(
            position for position in range(item_count) if position % FOLD_COUNT == fold
        )
        fit_positions = tuple(
            position for position in range(item_count) if position not in set(held_positions)
        )
        selected = select_program(rule, candidates, fit_positions)
        fold_hits = sum(selected.items[position].support_hits for position in held_positions)
        fold_total = sum(selected.items[position].support_total for position in held_positions)
        held_hits += fold_hits
        held_total += fold_total
        folds.append(
            {
                "fold": fold,
                "fit_count": len(fit_positions),
                "held_count": len(held_positions),
                "selected_program_sha256": selected.program_sha256,
                "held_support_hits": fold_hits,
                "held_support_total": fold_total,
            }
        )
    return {
        "rule_id": rule.id,
        "rule_sha256": rule.rule_sha256,
        "rule_complexity": rule.complexity,
        "held_support_hits": held_hits,
        "held_support_total": held_total,
        "folds": folds,
    }


def form_evaluator_challenger(
    formation_evidence: Sequence[ProgramRetrievalEvidence],
) -> dict[str, Any]:
    """Select one non-incumbent rule using A_form evidence only."""

    candidates = _validate_evidence_set(formation_evidence)
    rules = evaluator_rules()
    incumbent = next(rule for rule in rules if rule.id == INCUMBENT_RULE_ID)
    rows = [_crossfit_rule(rule, candidates) for rule in rules]
    challenger_rows = [row for row in rows if row["rule_id"] != incumbent.id]
    selected_row = min(
        challenger_rows,
        key=lambda row: (
            -Fraction(row["held_support_hits"], row["held_support_total"]),
            row["rule_complexity"],
            row["rule_sha256"],
        ),
    )
    challenger = next(
        rule for rule in rules if rule.id == selected_row["rule_id"]
    )
    incumbent_program = select_program(incumbent, candidates)
    challenger_program = select_program(challenger, candidates)
    body: dict[str, Any] = {
        "schema": f"{VERSION}_formation_receipt",
        "partition": "A_form",
        "fold_policy": "private_block_position_modulo_4_v1",
        "candidate_program_count": len(candidates),
        "evidence_set_sha256": _evidence_set_sha256(candidates),
        "evaluator_rule_count": len(rules),
        "incumbent_rule": asdict(incumbent),
        "challenger_rule": asdict(challenger),
        "incumbent_selected_program_sha256": incumbent_program.program_sha256,
        "challenger_selected_program_sha256": challenger_program.program_sha256,
        "crossfit": rows,
        "anchor_accessed": False,
        "measurement_accessed": False,
        "model_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    body["formation_sha256"] = stable_hash(body)
    return body


def freeze_prospective_search_formation(
    *,
    formation_evidence: Sequence[ProgramRetrievalEvidence],
    evaluator_formation_evidence: Sequence[ProgramRetrievalEvidence],
    evaluator_formation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze both old/new evaluator selections on F3 before M3 opens."""

    formation = _validate_evidence_set(formation_evidence)
    incumbent, challenger, evaluator_formation_hash = _verify_formation_receipt(
        evaluator_formation_receipt,
        evaluator_formation_evidence,
    )
    old_program = select_program(incumbent, formation)
    challenger_program = select_program(challenger, formation)
    body: dict[str, Any] = {
        "schema": f"{VERSION}_search_formation_receipt",
        "partition": "F3",
        "evaluator_formation_sha256": evaluator_formation_hash,
        "evidence_set_sha256": _evidence_set_sha256(formation),
        "candidate_program_count": len(formation),
        "incumbent_rule_id": incumbent.id,
        "challenger_rule_id": challenger.id,
        "incumbent_selected_program_sha256": old_program.program_sha256,
        "challenger_selected_program_sha256": challenger_program.program_sha256,
        "anchor_accessed": False,
        "measurement_accessed": False,
        "model_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    body["search_formation_sha256"] = stable_hash(body)
    return body


def compare_on_fixed_anchor(
    *,
    formation_evidence: Sequence[ProgramRetrievalEvidence],
    anchor_evidence: Sequence[ProgramRetrievalEvidence],
    formation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen incumbent/challenger selections on A_hold."""

    formation = _validate_evidence_set(formation_evidence)
    anchor = _validate_evidence_set(anchor_evidence)
    if set(row.program_sha256 for row in formation) != set(
        row.program_sha256 for row in anchor
    ):
        raise EvaluatorFormationError("formation and anchor program sets differ")
    incumbent, challenger, declared = _verify_formation_receipt(
        formation_receipt,
        formation,
    )
    incumbent_program = select_program(incumbent, formation)
    challenger_program = select_program(challenger, formation)
    anchor_by_program = {row.program_sha256: row for row in anchor}
    incumbent_anchor = anchor_by_program[incumbent_program.program_sha256]
    challenger_anchor = anchor_by_program[challenger_program.program_sha256]
    incumbent_hits = sum(row.support_hits for row in incumbent_anchor.items)
    challenger_hits = sum(row.support_hits for row in challenger_anchor.items)
    total = sum(row.support_total for row in incumbent_anchor.items)
    if total != sum(row.support_total for row in challenger_anchor.items):
        raise EvaluatorFormationError("anchor label totals differ")
    incumbent_score = AnchorScore(
        evaluator_id=incumbent.id,
        anchor_manifest_hash=stable_hash(
            [row.item_commitment_sha256 for row in incumbent_anchor.items]
        ),
        successes=incumbent_hits,
        total=total,
    )
    challenger_score = AnchorScore(
        evaluator_id=challenger.id,
        anchor_manifest_hash=incumbent_score.anchor_manifest_hash,
        successes=challenger_hits,
        total=total,
    )
    incumbent_lower = wilson_lower_bound(
        incumbent_hits, total, ANCHOR_CONFIDENCE
    )
    challenger_lower = wilson_lower_bound(
        challenger_hits, total, ANCHOR_CONFIDENCE
    )
    promoted = challenger_lower > incumbent_lower
    body: dict[str, Any] = {
        "schema": f"{VERSION}_anchor_result",
        "partition": "A_hold",
        "formation_sha256": declared,
        "anchor_evidence_set_sha256": _evidence_set_sha256(anchor),
        "anchor_manifest_sha256": incumbent_score.anchor_manifest_hash,
        "anchor_item_count": len(incumbent_anchor.items),
        "anchor_support_total": total,
        "incumbent_rule_id": incumbent.id,
        "challenger_rule_id": challenger.id,
        "incumbent_selected_program_sha256": incumbent_program.program_sha256,
        "challenger_selected_program_sha256": challenger_program.program_sha256,
        "incumbent_support_hits": incumbent_hits,
        "challenger_support_hits": challenger_hits,
        "confidence": ANCHOR_CONFIDENCE,
        "incumbent_lower_bound": incumbent_lower,
        "challenger_lower_bound": challenger_lower,
        "challenger_promoted": promoted,
        "transition_policy": "strict_wilson_lower_bound_improvement_v1",
        "official_support_objective_replaced": False,
        "model_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    body["anchor_result_sha256"] = stable_hash(body)
    return body


def measure_prospective_search_utility(
    *,
    formation_evidence: Sequence[ProgramRetrievalEvidence],
    measurement_evidence: Sequence[ProgramRetrievalEvidence],
    evaluator_formation_evidence: Sequence[ProgramRetrievalEvidence],
    anchor_evidence: Sequence[ProgramRetrievalEvidence],
    evaluator_formation_receipt: Mapping[str, Any],
    search_formation_receipt: Mapping[str, Any],
    anchor_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare old/new evaluator selections on untouched F3 -> M3."""

    formation = _validate_evidence_set(formation_evidence)
    measurement = _validate_evidence_set(measurement_evidence)
    if set(row.program_sha256 for row in formation) != set(
        row.program_sha256 for row in measurement
    ):
        raise EvaluatorFormationError("F3 and M3 program sets differ")
    expected_anchor = compare_on_fixed_anchor(
        formation_evidence=evaluator_formation_evidence,
        anchor_evidence=anchor_evidence,
        formation_receipt=evaluator_formation_receipt,
    )
    if dict(anchor_result) != expected_anchor:
        raise EvaluatorFormationError(
            "anchor result differs from fixed A_hold evidence"
        )
    anchor_body = dict(anchor_result)
    anchor_hash = anchor_body.pop("anchor_result_sha256", None)
    if stable_hash(anchor_body) != anchor_hash:
        raise EvaluatorFormationError("anchor result hash mismatch")
    incumbent, challenger, evaluator_formation_hash = _verify_formation_receipt(
        evaluator_formation_receipt,
        evaluator_formation_evidence,
    )
    if anchor_result.get("formation_sha256") != evaluator_formation_hash:
        raise EvaluatorFormationError("anchor used a different evaluator formation")
    search_body = dict(search_formation_receipt)
    search_hash = search_body.pop("search_formation_sha256", None)
    if (
        search_formation_receipt.get("schema")
        != f"{VERSION}_search_formation_receipt"
        or search_formation_receipt.get("partition") != "F3"
        or stable_hash(search_body) != search_hash
        or search_formation_receipt.get("evaluator_formation_sha256")
        != evaluator_formation_hash
        or search_formation_receipt.get("evidence_set_sha256")
        != _evidence_set_sha256(formation)
        or search_formation_receipt.get("anchor_accessed") is not False
        or search_formation_receipt.get("measurement_accessed") is not False
    ):
        raise EvaluatorFormationError("F3 search formation receipt mismatch")
    old_program = select_program(incumbent, formation)
    active_rule = (
        challenger if anchor_result.get("challenger_promoted") is True else incumbent
    )
    active_program = select_program(active_rule, formation)
    if (
        search_formation_receipt.get("incumbent_selected_program_sha256")
        != old_program.program_sha256
        or search_formation_receipt.get("challenger_selected_program_sha256")
        != select_program(challenger, formation).program_sha256
    ):
        raise EvaluatorFormationError("F3 selected program binding drifted")
    measurement_by_program = {row.program_sha256: row for row in measurement}
    old_measurement = measurement_by_program[old_program.program_sha256]
    active_measurement = measurement_by_program[active_program.program_sha256]
    old_hits = sum(row.support_hits for row in old_measurement.items)
    active_hits = sum(row.support_hits for row in active_measurement.items)
    total = sum(row.support_total for row in old_measurement.items)
    if total != sum(row.support_total for row in active_measurement.items):
        raise EvaluatorFormationError("M3 support totals differ")
    body: dict[str, Any] = {
        "schema": f"{VERSION}_prospective_search_utility",
        "formation_partition": "F3",
        "measurement_partition": "M3",
        "anchor_result_sha256": anchor_hash,
        "search_formation_sha256": search_hash,
        "measurement_evidence_set_sha256": _evidence_set_sha256(measurement),
        "incumbent_rule_id": incumbent.id,
        "active_rule_id": active_rule.id,
        "incumbent_selected_program_sha256": old_program.program_sha256,
        "active_selected_program_sha256": active_program.program_sha256,
        "support_total": total,
        "incumbent_support_hits": old_hits,
        "active_support_hits": active_hits,
        "net_support_hits": active_hits - old_hits,
        "evaluator_transition_had_positive_search_utility": (
            anchor_result.get("challenger_promoted") is True
            and active_hits > old_hits
        ),
        "model_calls": 0,
        "online_evaluator_calls": 0,
        "raw_content_persisted": False,
    }
    body["search_utility_sha256"] = stable_hash(body)
    return body

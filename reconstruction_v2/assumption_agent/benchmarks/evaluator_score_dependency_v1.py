"""Prospective score-dependency ledger for evaluator epoch changes.

The production archive already invalidates scores tagged with a replaced
evaluator epoch.  This module makes that dependency explicit for fresh studies
and, importantly, distinguishes evaluator-derived scores from fixed external
objectives.  It is wiring, not an additional promotion or performance gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from ..models import stable_hash


LEDGER_SCHEMA = "evaluator_score_dependency_ledger_v1"
EVALUATOR_DEPENDENT = "evaluator_epoch"
INDEPENDENT_OBJECTIVE = "independent_objective"
_DEPENDENCY_KINDS = frozenset((EVALUATOR_DEPENDENT, INDEPENDENT_OBJECTIVE))


class ScoreDependencyError(RuntimeError):
    """Raised when score provenance or epoch invalidation is ambiguous."""


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScoreDependencyError(f"{field} must be lowercase sha256")
    return value


@dataclass(frozen=True)
class CachedOutputBinding:
    """Hash-only identity of immutable outputs that may be re-evaluated."""

    producer_execution_sha256: str
    output_set_sha256: str
    item_set_sha256: str
    item_count: int
    raw_outputs_persisted_publicly: bool = False

    def validate(self) -> "CachedOutputBinding":
        _require_sha256(
            self.producer_execution_sha256, "producer execution sha256"
        )
        _require_sha256(self.output_set_sha256, "output set sha256")
        _require_sha256(self.item_set_sha256, "item set sha256")
        if type(self.item_count) is not int or self.item_count <= 0:
            raise ScoreDependencyError("cached output item count must be positive")
        if self.raw_outputs_persisted_publicly is not False:
            raise ScoreDependencyError("cached raw outputs may not be public")
        return self

    @property
    def binding_sha256(self) -> str:
        self.validate()
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class DependentScoreRecord:
    id: str
    archive_node_id: str
    metric: str
    successes: int
    total: int
    item_set_sha256: str
    evidence_sha256: str
    dependency_kind: str
    evaluator_epoch_id: str | None
    cached_output_binding_sha256: str | None
    valid: bool = True
    invalidation_reason: str = ""

    def validate(self) -> "DependentScoreRecord":
        if not self.id or not self.archive_node_id or not self.metric:
            raise ScoreDependencyError("score identity fields must be non-empty")
        if (
            type(self.successes) is not int
            or type(self.total) is not int
            or self.total <= 0
            or self.successes < 0
            or self.successes > self.total
        ):
            raise ScoreDependencyError("score counts are invalid")
        _require_sha256(self.item_set_sha256, "score item set sha256")
        _require_sha256(self.evidence_sha256, "score evidence sha256")
        if self.dependency_kind not in _DEPENDENCY_KINDS:
            raise ScoreDependencyError("unknown score dependency kind")
        if self.dependency_kind == EVALUATOR_DEPENDENT:
            if not self.evaluator_epoch_id:
                raise ScoreDependencyError(
                    "evaluator-dependent score requires an epoch"
                )
            if self.cached_output_binding_sha256 is not None:
                _require_sha256(
                    self.cached_output_binding_sha256,
                    "cached output binding sha256",
                )
        elif (
            self.evaluator_epoch_id is not None
            or self.cached_output_binding_sha256 is not None
        ):
            raise ScoreDependencyError(
                "independent score may not claim evaluator/cache dependency"
            )
        if self.dependency_kind == INDEPENDENT_OBJECTIVE and not self.valid:
            raise ScoreDependencyError(
                "independent objective may not be epoch-invalidated"
            )
        if self.valid and self.invalidation_reason:
            raise ScoreDependencyError(
                "valid score may not have an invalidation reason"
            )
        if not self.valid and not self.invalidation_reason:
            raise ScoreDependencyError(
                "invalid score requires an invalidation reason"
            )
        return self


class ScoreDependencyLedger:
    """Append-only score provenance plus selective epoch invalidation."""

    def __init__(self) -> None:
        self.records: dict[str, DependentScoreRecord] = {}
        self.transitions: list[dict[str, Any]] = []

    def record(
        self,
        *,
        archive_node_id: str,
        metric: str,
        successes: int,
        total: int,
        item_set_sha256: str,
        evidence_sha256: str,
        dependency_kind: str,
        evaluator_epoch_id: str | None = None,
        cached_outputs: CachedOutputBinding | None = None,
    ) -> DependentScoreRecord:
        cache_hash = (
            cached_outputs.validate().binding_sha256
            if cached_outputs is not None
            else None
        )
        identity = {
            "archive_node_id": archive_node_id,
            "metric": metric,
            "item_set_sha256": item_set_sha256,
            "evidence_sha256": evidence_sha256,
            "dependency_kind": dependency_kind,
            "evaluator_epoch_id": evaluator_epoch_id,
            "cached_output_binding_sha256": cache_hash,
        }
        record = DependentScoreRecord(
            id=f"dependent_score_{stable_hash(identity)[:20]}",
            archive_node_id=archive_node_id,
            metric=metric,
            successes=successes,
            total=total,
            item_set_sha256=item_set_sha256,
            evidence_sha256=evidence_sha256,
            dependency_kind=dependency_kind,
            evaluator_epoch_id=evaluator_epoch_id,
            cached_output_binding_sha256=cache_hash,
        ).validate()
        if any(
            record.dependency_kind == EVALUATOR_DEPENDENT
            and row["old_epoch_id"] == record.evaluator_epoch_id
            for row in self._validated_transitions()
        ):
            raise ScoreDependencyError(
                "cannot append a score for an evaluator epoch already replaced"
            )
        existing = self.records.get(record.id)
        if existing is not None and existing != record:
            raise ScoreDependencyError("dependent score ID collision")
        self.records[record.id] = record
        return record

    def transition_epoch(
        self,
        *,
        old_epoch_id: str,
        new_epoch_id: str,
        transition_evidence_sha256: str,
    ) -> tuple[str, ...]:
        if not old_epoch_id or not new_epoch_id or old_epoch_id == new_epoch_id:
            raise ScoreDependencyError("epoch transition identity is invalid")
        _require_sha256(
            transition_evidence_sha256, "transition evidence sha256"
        )
        if any(
            row["old_epoch_id"] == old_epoch_id
            for row in self.transitions
        ):
            raise ScoreDependencyError("old evaluator epoch was already replaced")
        invalidated: list[str] = []
        for record_id, record in tuple(self.records.items()):
            if (
                record.valid
                and record.dependency_kind == EVALUATOR_DEPENDENT
                and record.evaluator_epoch_id == old_epoch_id
            ):
                self.records[record_id] = replace(
                    record,
                    valid=False,
                    invalidation_reason=(
                        f"evaluator_epoch_replaced:{old_epoch_id}->{new_epoch_id}"
                    ),
                )
                invalidated.append(record_id)
        transition = {
            "old_epoch_id": old_epoch_id,
            "new_epoch_id": new_epoch_id,
            "transition_evidence_sha256": transition_evidence_sha256,
            "invalidated_score_record_ids": sorted(invalidated),
        }
        transition["transition_sha256"] = stable_hash(transition)
        self.transitions.append(transition)
        self._validated_transitions()
        return tuple(sorted(invalidated))

    def _validated_transitions(self) -> tuple[dict[str, Any], ...]:
        expected_keys = {
            "old_epoch_id",
            "new_epoch_id",
            "transition_evidence_sha256",
            "invalidated_score_record_ids",
            "transition_sha256",
        }
        replaced_epochs: set[str] = set()
        referenced_records: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for raw in self.transitions:
            if not isinstance(raw, Mapping) or set(raw) != expected_keys:
                raise ScoreDependencyError("epoch transition row is malformed")
            row = dict(raw)
            declared = row.pop("transition_sha256", None)
            if stable_hash(row) != declared:
                raise ScoreDependencyError("epoch transition hash mismatch")
            old_epoch_id = row.get("old_epoch_id")
            new_epoch_id = row.get("new_epoch_id")
            if (
                not isinstance(old_epoch_id, str)
                or not old_epoch_id
                or not isinstance(new_epoch_id, str)
                or not new_epoch_id
                or old_epoch_id == new_epoch_id
                or old_epoch_id in replaced_epochs
            ):
                raise ScoreDependencyError("epoch transition identity is invalid")
            _require_sha256(
                row.get("transition_evidence_sha256"),
                "transition evidence sha256",
            )
            record_ids = row.get("invalidated_score_record_ids")
            if (
                not isinstance(record_ids, list)
                or any(not isinstance(value, str) or not value for value in record_ids)
                or record_ids != sorted(set(record_ids))
            ):
                raise ScoreDependencyError(
                    "invalidated score record IDs are malformed"
                )
            for record_id in record_ids:
                record = self.records.get(record_id)
                expected_reason = (
                    f"evaluator_epoch_replaced:{old_epoch_id}->{new_epoch_id}"
                )
                if (
                    record is None
                    or record.dependency_kind != EVALUATOR_DEPENDENT
                    or record.evaluator_epoch_id != old_epoch_id
                    or record.valid
                    or record.invalidation_reason != expected_reason
                    or record_id in referenced_records
                ):
                    raise ScoreDependencyError(
                        "epoch transition references an invalid score record"
                    )
                referenced_records.add(record_id)
            replaced_epochs.add(old_epoch_id)
            normalized.append(dict(raw))
        invalid_records = {
            record_id
            for record_id, record in self.records.items()
            if not record.valid
        }
        if invalid_records != referenced_records:
            raise ScoreDependencyError(
                "invalid score records are not closed by epoch transitions"
            )
        return tuple(normalized)

    def to_dict(self) -> dict[str, Any]:
        transitions = self._validated_transitions()
        body: dict[str, Any] = {
            "schema": LEDGER_SCHEMA,
            "records": {
                key: asdict(value.validate())
                for key, value in sorted(self.records.items())
            },
            "transitions": [dict(row) for row in transitions],
            "raw_outputs_persisted_publicly": False,
        }
        body["ledger_sha256"] = stable_hash(body)
        return body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreDependencyLedger":
        body = dict(value)
        declared = body.pop("ledger_sha256", None)
        if value.get("schema") != LEDGER_SCHEMA or stable_hash(body) != declared:
            raise ScoreDependencyError("score dependency ledger hash mismatch")
        records = value.get("records")
        transitions = value.get("transitions")
        if not isinstance(records, Mapping) or not isinstance(transitions, list):
            raise ScoreDependencyError("score dependency ledger is malformed")
        ledger = cls()
        for key, row in records.items():
            if not isinstance(row, Mapping):
                raise ScoreDependencyError("score dependency row is malformed")
            record = DependentScoreRecord(**dict(row)).validate()
            if key != record.id:
                raise ScoreDependencyError("score dependency key mismatch")
            ledger.records[record.id] = record
        ledger.transitions = [dict(row) for row in transitions]
        if ledger.to_dict() != dict(value):
            raise ScoreDependencyError("score dependency ledger round trip drifted")
        return ledger


def cached_rescore_receipt(
    *,
    cached_outputs: CachedOutputBinding,
    prior_epoch_id: str,
    new_epoch_id: str,
    evaluator_implementation_sha256: str,
    score_evidence_sha256: str,
    item_result_hashes: Sequence[str],
) -> dict[str, Any]:
    """Bind a re-score to immutable outputs without rerunning their producer."""

    binding = cached_outputs.validate()
    if not prior_epoch_id or not new_epoch_id or prior_epoch_id == new_epoch_id:
        raise ScoreDependencyError("cached re-score epoch identity is invalid")
    _require_sha256(
        evaluator_implementation_sha256, "evaluator implementation sha256"
    )
    _require_sha256(score_evidence_sha256, "score evidence sha256")
    if len(item_result_hashes) != binding.item_count:
        raise ScoreDependencyError("cached re-score item count mismatch")
    normalized_hashes = [
        _require_sha256(value, "cached item result hash")
        for value in item_result_hashes
    ]
    if stable_hash(normalized_hashes) != binding.output_set_sha256:
        raise ScoreDependencyError("cached output set hash mismatch")
    body: dict[str, Any] = {
        "cached_output_binding_sha256": binding.binding_sha256,
        "prior_epoch_id": prior_epoch_id,
        "new_epoch_id": new_epoch_id,
        "evaluator_implementation_sha256": evaluator_implementation_sha256,
        "score_evidence_sha256": score_evidence_sha256,
        "item_count": binding.item_count,
        "producer_rerun_calls": 0,
        "raw_outputs_persisted_publicly": False,
    }
    body["receipt_sha256"] = stable_hash(body)
    return body

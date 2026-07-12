from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .evaluation import PromotionDecision
from .events import Event, EventSink, NullEventSink
from .models import HypothesisProgram, HypothesisStatus, stable_hash


class ArchiveNodeStatus(str, Enum):
    CANDIDATE = "candidate"
    INCUMBENT = "incumbent"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class ArchiveNode:
    id: str
    parent_id: str | None
    active_hypothesis_ids: tuple[str, ...]
    evaluator_epoch_id: str
    runtime_version: str
    generation: int
    status: ArchiveNodeStatus = ArchiveNodeStatus.CANDIDATE

    @property
    def payload_hash(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class ScoreRecord:
    id: str
    archive_node_id: str
    split: str
    evaluator_epoch_id: str
    metric: str
    successes: int
    total: int
    item_set_hash: str
    valid: bool = True
    invalidation_reason: str = ""


@dataclass(frozen=True)
class EvaluatorSpec:
    id: str
    version: str
    implementation_hash: str
    criteria_hash: str
    anchor_manifest_hash: str


@dataclass(frozen=True)
class EvaluatorEpoch:
    id: str
    index: int
    evaluator: EvaluatorSpec
    parent_epoch_id: str | None = None


@dataclass(frozen=True)
class AnchorScore:
    evaluator_id: str
    anchor_manifest_hash: str
    successes: int
    total: int

    def lower_bound(self, confidence: float) -> float:
        return wilson_lower_bound(self.successes, self.total, confidence)


@dataclass(frozen=True)
class EvaluatorTransition:
    promoted: bool
    incumbent_epoch_id: str
    next_epoch: EvaluatorEpoch
    incumbent_lower_bound: float
    challenger_lower_bound: float
    invalidated_score_record_ids: tuple[str, ...]
    reason: str


class PolicyArchive:
    def __init__(self, *, event_sink: EventSink | None = None) -> None:
        self.hypotheses: dict[str, HypothesisProgram] = {}
        self.nodes: dict[str, ArchiveNode] = {}
        self.score_records: dict[str, ScoreRecord] = {}
        self.incumbent_id: str | None = None
        self.event_sink = event_sink or NullEventSink()

    def register_hypothesis(self, program: HypothesisProgram, *, trace_id: str = "archive") -> None:
        if program.id in self.hypotheses and self.hypotheses[program.id].payload_hash != program.payload_hash:
            raise ValueError(f"hypothesis ID collision: {program.id}")
        self.hypotheses[program.id] = program
        self.event_sink.emit(
            Event(
                event="archive_hypothesis_registered",
                stage="archive",
                trace_id=trace_id,
                payload={
                    "hypothesis_id": program.id,
                    "hypothesis_hash": program.payload_hash,
                    "kind": program.kind.value,
                    "status": program.status.value,
                    "evaluator_epoch": program.evaluator_epoch,
                },
            )
        )

    def set_hypothesis_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        *,
        trace_id: str = "archive",
    ) -> None:
        if hypothesis_id not in self.hypotheses:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")
        current = self.hypotheses[hypothesis_id]
        self.hypotheses[hypothesis_id] = replace(current, status=status)
        self.event_sink.emit(
            Event(
                event="archive_hypothesis_status_changed",
                stage="archive",
                trace_id=trace_id,
                payload={
                    "hypothesis_id": hypothesis_id,
                    "previous_status": current.status.value,
                    "next_status": status.value,
                    "hypothesis_hash": self.hypotheses[hypothesis_id].payload_hash,
                },
            )
        )

    def create_node(
        self,
        *,
        active_hypothesis_ids: tuple[str, ...],
        evaluator_epoch_id: str,
        runtime_version: str,
        parent_id: str | None = None,
        trace_id: str = "archive",
    ) -> ArchiveNode:
        missing = sorted(set(active_hypothesis_ids) - set(self.hypotheses))
        if missing:
            raise KeyError(f"archive node references unknown hypotheses: {missing}")
        generation = self.nodes[parent_id].generation + 1 if parent_id else 0
        node_id = f"node_{stable_hash({'parent': parent_id, 'hypotheses': active_hypothesis_ids, 'epoch': evaluator_epoch_id, 'runtime': runtime_version})[:16]}"
        node = ArchiveNode(
            id=node_id,
            parent_id=parent_id,
            active_hypothesis_ids=tuple(sorted(active_hypothesis_ids)),
            evaluator_epoch_id=evaluator_epoch_id,
            runtime_version=runtime_version,
            generation=generation,
        )
        self.nodes[node.id] = node
        self.event_sink.emit(
            Event(
                event="archive_node_created",
                stage="archive",
                trace_id=trace_id,
                payload={"archive_node": node.to_dict(), "archive_node_hash": node.payload_hash},
            )
        )
        return node

    def record_score(
        self,
        *,
        archive_node_id: str,
        split: str,
        evaluator_epoch_id: str,
        metric: str,
        successes: int,
        total: int,
        item_ids: tuple[str, ...],
        valid: bool = True,
        invalidation_reason: str = "",
    ) -> ScoreRecord:
        if archive_node_id not in self.nodes:
            raise KeyError(f"unknown archive node: {archive_node_id}")
        if valid and invalidation_reason:
            raise ValueError("valid score record cannot have an invalidation reason")
        if not valid and not invalidation_reason:
            raise ValueError("invalid score record requires an invalidation reason")
        record_id = f"score_{stable_hash({'node': archive_node_id, 'split': split, 'epoch': evaluator_epoch_id, 'metric': metric, 'items': item_ids})[:16]}"
        record = ScoreRecord(
            id=record_id,
            archive_node_id=archive_node_id,
            split=split,
            evaluator_epoch_id=evaluator_epoch_id,
            metric=metric,
            successes=successes,
            total=total,
            item_set_hash=stable_hash({"item_ids": sorted(item_ids)}),
            valid=valid,
            invalidation_reason=invalidation_reason,
        )
        self.score_records[record.id] = record
        return record

    def apply_promotion(
        self,
        *,
        candidate_node_id: str,
        decision: PromotionDecision,
        trace_id: str = "archive",
    ) -> ArchiveNode:
        candidate = self.nodes[candidate_node_id]
        if decision.allowed:
            if self.incumbent_id and self.incumbent_id in self.nodes:
                self.nodes[self.incumbent_id] = replace(
                    self.nodes[self.incumbent_id], status=ArchiveNodeStatus.SUPERSEDED
                )
            candidate = replace(candidate, status=ArchiveNodeStatus.INCUMBENT)
            self.incumbent_id = candidate.id
            for hypothesis_id in candidate.active_hypothesis_ids:
                self.hypotheses[hypothesis_id] = replace(
                    self.hypotheses[hypothesis_id], status=HypothesisStatus.PROMOTED
                )
        else:
            candidate = replace(candidate, status=ArchiveNodeStatus.REJECTED)
            parent_ids = (
                set(self.nodes[candidate.parent_id].active_hypothesis_ids)
                if candidate.parent_id and candidate.parent_id in self.nodes
                else set()
            )
            for hypothesis_id in set(candidate.active_hypothesis_ids) - parent_ids:
                self.hypotheses[hypothesis_id] = replace(
                    self.hypotheses[hypothesis_id],
                    status=HypothesisStatus.REJECTED,
                )
        self.nodes[candidate.id] = candidate
        self.event_sink.emit(
            Event(
                event="archive_promotion_applied",
                stage="archive.promotion",
                trace_id=trace_id,
                payload={
                    "candidate_node_id": candidate.id,
                    "archive_status": candidate.status.value,
                    "promotion_allowed": decision.allowed,
                    "promotion_blockers": list(decision.blockers),
                    "incumbent_id": self.incumbent_id,
                },
            )
        )
        return candidate

    def invalidate_evaluator_epoch(self, epoch_id: str) -> tuple[str, ...]:
        invalidated: list[str] = []
        for record_id, record in list(self.score_records.items()):
            if record.valid and record.evaluator_epoch_id == epoch_id:
                self.score_records[record_id] = replace(
                    record,
                    valid=False,
                    invalidation_reason="evaluator_epoch_replaced",
                )
                invalidated.append(record_id)
        return tuple(sorted(invalidated))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypotheses": {key: value.to_dict() for key, value in sorted(self.hypotheses.items())},
            "nodes": {key: value.to_dict() for key, value in sorted(self.nodes.items())},
            "score_records": {key: asdict(value) for key, value in sorted(self.score_records.items())},
            "incumbent_id": self.incumbent_id,
            "archive_hash": stable_hash(
                {
                    "hypotheses": {key: value.payload_hash for key, value in sorted(self.hypotheses.items())},
                    "nodes": {key: value.payload_hash for key, value in sorted(self.nodes.items())},
                    "scores": {key: asdict(value) for key, value in sorted(self.score_records.items())},
                    "incumbent_id": self.incumbent_id,
                }
            ),
            "raw_content_persisted": False,
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class EvaluatorEpochController:
    def __init__(
        self,
        initial_epoch: EvaluatorEpoch,
        *,
        confidence: float = 0.9,
        minimum_lower_bound_gain: float = 0.0,
        event_sink: EventSink | None = None,
    ) -> None:
        self.current = initial_epoch
        self.confidence = confidence
        self.minimum_lower_bound_gain = minimum_lower_bound_gain
        self.event_sink = event_sink or NullEventSink()

    def consider_challenger(
        self,
        challenger: EvaluatorSpec,
        *,
        incumbent_score: AnchorScore,
        challenger_score: AnchorScore,
        archive: PolicyArchive,
        trace_id: str = "evaluator_transition",
    ) -> EvaluatorTransition:
        if incumbent_score.evaluator_id != self.current.evaluator.id:
            raise ValueError("incumbent anchor score does not match the frozen evaluator")
        if challenger_score.evaluator_id != challenger.id:
            raise ValueError("challenger anchor score does not match the challenger evaluator")
        if incumbent_score.anchor_manifest_hash != challenger_score.anchor_manifest_hash:
            raise ValueError("evaluator comparison must use the same fixed anchor")
        if challenger.anchor_manifest_hash != challenger_score.anchor_manifest_hash:
            raise ValueError("challenger was not scored on its declared anchor")
        if incumbent_score.total != challenger_score.total:
            raise ValueError("evaluator anchor comparisons require matched rows")
        incumbent_lower = incumbent_score.lower_bound(self.confidence)
        challenger_lower = challenger_score.lower_bound(self.confidence)
        promoted = challenger_lower > incumbent_lower + self.minimum_lower_bound_gain
        old_epoch = self.current
        if promoted:
            invalidated = archive.invalidate_evaluator_epoch(old_epoch.id)
            next_epoch = EvaluatorEpoch(
                id=f"eval_epoch_{old_epoch.index + 1}_{stable_hash({'challenger': asdict(challenger), 'parent': old_epoch.id})[:10]}",
                index=old_epoch.index + 1,
                evaluator=challenger,
                parent_epoch_id=old_epoch.id,
            )
            self.current = next_epoch
            reason = "challenger_anchor_lower_bound_improved"
        else:
            invalidated = ()
            next_epoch = old_epoch
            reason = "incumbent_retained_on_tie_or_lower_bound"
        transition = EvaluatorTransition(
            promoted=promoted,
            incumbent_epoch_id=old_epoch.id,
            next_epoch=next_epoch,
            incumbent_lower_bound=incumbent_lower,
            challenger_lower_bound=challenger_lower,
            invalidated_score_record_ids=invalidated,
            reason=reason,
        )
        self.event_sink.emit(
            Event(
                event="evaluator_epoch_transition_decided",
                stage="evaluator_epoch",
                trace_id=trace_id,
                payload={
                    "promoted": promoted,
                    "incumbent_epoch_id": old_epoch.id,
                    "next_epoch_id": next_epoch.id,
                    "incumbent_evaluator_id": old_epoch.evaluator.id,
                    "challenger_evaluator_id": challenger.id,
                    "anchor_manifest_hash": challenger.anchor_manifest_hash,
                    "incumbent_lower_bound": incumbent_lower,
                    "challenger_lower_bound": challenger_lower,
                    "invalidated_score_record_ids": list(invalidated),
                    "reason": reason,
                },
            )
        )
        return transition


def wilson_lower_bound(successes: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 0.0
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    z_score = statistics.NormalDist().inv_cdf(confidence)
    proportion = successes / total
    denominator = 1.0 + z_score**2 / total
    centre = proportion + z_score**2 / (2.0 * total)
    margin = z_score * math.sqrt(
        (proportion * (1.0 - proportion) + z_score**2 / (4.0 * total)) / total
    )
    return (centre - margin) / denominator

"""Prospective, dataset-independent L4 operator-retention measurement.

The protocol freezes two operator slots (``P`` retained and ``Q`` novel),
executes each slot exactly once per fresh item, and then derives the four
counterfactual arms ``empty``, ``P``, ``Q``, and ``P+Q`` from those immutable
rankings.  Arm construction therefore cannot change after support labels are
scored.  Retrieval is evaluated locally with support recall@5 and deterministic
reciprocal-rank fusion (RRF).

This module is a measurement boundary, not a promotion gate.  It binds the
existing :class:`~assumption_agent.archive.PolicyArchive` incumbent/evaluator
epoch, but deliberately does not introduce a second archive or threshold
system.  Dataset adapters only need to provide opaque item payloads, corpus
document IDs, support-document IDs, and frozen ranked operators.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from fractions import Fraction
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..archive import ArchiveNodeStatus, PolicyArchive
from ..models import HypothesisStatus, stable_hash


PROTOCOL_VERSION = "l4_typed_operator_retention_protocol_v1"
ARM_IDS = ("empty", "P", "Q", "P_plus_Q")
SLOT_IDS = ("P", "Q")
ARM_SLOT_MAP: Mapping[str, tuple[str, ...]] = {
    "empty": (),
    "P": ("P",),
    "Q": ("Q",),
    "P_plus_Q": ("P", "Q"),
}
TOP_K = 5
RRF_CONSTANT = 60
PRIMARY_METRIC = "offline_micro_support_recall_at_5"
RETENTION_ESTIMAND = "Y(P_plus_Q)-Y(Q)"
NOVELTY_ESTIMAND = "Y(P_plus_Q)-Y(P)"
FORGETTING_ESTIMAND = (
    "support_hits(P)_not_in_support_hits(P_plus_Q)/all_supports"
)
FRESH_ROOT_POLICY = "new_bound_root_exclusive_create_no_replay_v1"
RANKING_REUSE_POLICY = "one_P_and_one_Q_call_per_item_shared_across_arms_v1"
FAILURE_FILENAME = "retention.failure.json"
FREEZE_FILENAME = "retention.pre_run_freeze.json"
CONSUMPTION_FILENAME = "retention.authorization.consumed.json"
REPORT_FILENAME = "retention.aggregate.report.json"


class L4RetentionProtocolError(RuntimeError):
    """The frozen L4 measurement contract was not satisfied."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, field_name: str) -> str:
    if not _is_sha256(value):
        raise L4RetentionProtocolError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return str(value)


def _canonical_json_clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise L4RetentionProtocolError(
            "operator input must be one finite JSON object"
        ) from exc
    if not isinstance(decoded, dict):
        raise L4RetentionProtocolError(
            "operator input must be one JSON object"
        )
    return decoded


def _require_nonempty_unique_strings(
    values: Sequence[str], field_name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(values)
    if (
        (not allow_empty and not normalized)
        or any(not isinstance(value, str) or not value for value in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise L4RetentionProtocolError(
            f"{field_name} must contain unique non-empty strings"
        )
    return normalized


def _canonical_new_root(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name in {"", ".", ".."}:
        raise L4RetentionProtocolError("execution root is invalid")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise L4RetentionProtocolError(
            "execution root parent is unavailable"
        ) from exc
    if not parent.is_dir():
        raise L4RetentionProtocolError(
            "execution root parent is not a directory"
        )
    return parent / candidate.name


def execution_root_binding_hash(path: str | Path) -> str:
    root = _canonical_new_root(path)
    return stable_hash({"absolute_execution_root": str(root)})


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class OperatorQuery:
    """Gold-free input passed to a ranked operator."""

    item_id: str
    document_ids: tuple[str, ...]
    payload: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class RetentionItem:
    """One private item used to freeze and score the L4 measurement."""

    item_id: str
    block_id: str
    document_ids: tuple[str, ...]
    support_document_ids: tuple[str, ...] = field(repr=False)
    operator_input: Mapping[str, Any] = field(repr=False, compare=False)
    baseline_ranked_document_ids: tuple[str, ...] = ()

    def verify(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise L4RetentionProtocolError("item ID is missing")
        if not isinstance(self.block_id, str) or not self.block_id:
            raise L4RetentionProtocolError("block ID is missing")
        documents = _require_nonempty_unique_strings(
            self.document_ids, "document IDs"
        )
        supports = _require_nonempty_unique_strings(
            self.support_document_ids, "support-document IDs"
        )
        baseline = _require_nonempty_unique_strings(
            self.baseline_ranked_document_ids,
            "baseline ranking",
            allow_empty=True,
        )
        if not set(supports).issubset(documents):
            raise L4RetentionProtocolError(
                "support-document IDs escaped the frozen corpus"
            )
        if not set(baseline).issubset(documents):
            raise L4RetentionProtocolError(
                "baseline ranking escaped the frozen corpus"
            )
        _canonical_json_clone(self.operator_input)

    @property
    def item_id_hash(self) -> str:
        return stable_hash({"item_id": self.item_id})

    @property
    def block_id_hash(self) -> str:
        return stable_hash({"block_id": self.block_id})

    def binding_payload(self) -> dict[str, Any]:
        self.verify()
        body = {
            "item_id_hash": self.item_id_hash,
            "block_id_hash": self.block_id_hash,
            "operator_input_hash": stable_hash(
                _canonical_json_clone(self.operator_input)
            ),
            "document_order_hash": stable_hash(
                {"document_ids": list(self.document_ids)}
            ),
            "document_count": len(self.document_ids),
            "support_set_hash": stable_hash(
                {"support_document_ids": sorted(self.support_document_ids)}
            ),
            "support_count": len(self.support_document_ids),
            "baseline_ranking_hash": stable_hash(
                {
                    "baseline_ranked_document_ids": list(
                        self.baseline_ranked_document_ids
                    )
                }
            ),
            "baseline_ranking_count": len(
                self.baseline_ranked_document_ids
            ),
        }
        return {**body, "item_binding_hash": stable_hash(body)}

    def gold_free_query(self) -> OperatorQuery:
        self.verify()
        return OperatorQuery(
            item_id=self.item_id,
            document_ids=tuple(self.document_ids),
            payload=_canonical_json_clone(self.operator_input),
        )


@dataclass(frozen=True)
class ItemManifestRow:
    item_id_hash: str
    block_id_hash: str
    item_binding_hash: str
    support_count: int
    document_count: int
    baseline_ranking_count: int

    @classmethod
    def from_item(cls, item: RetentionItem) -> "ItemManifestRow":
        binding = item.binding_payload()
        return cls(
            item_id_hash=binding["item_id_hash"],
            block_id_hash=binding["block_id_hash"],
            item_binding_hash=binding["item_binding_hash"],
            support_count=binding["support_count"],
            document_count=binding["document_count"],
            baseline_ranking_count=binding["baseline_ranking_count"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id_hash": self.item_id_hash,
            "block_id_hash": self.block_id_hash,
            "item_binding_hash": self.item_binding_hash,
            "support_count": self.support_count,
            "document_count": self.document_count,
            "baseline_ranking_count": self.baseline_ranking_count,
        }


@dataclass(frozen=True)
class FixedItemBlockManifest:
    """Hash-safe commitment to one fresh, fixed item/block set."""

    dataset_binding_hash: str
    fresh_partition_hash: str
    item_rows: tuple[ItemManifestRow, ...]
    block_rows: tuple[Mapping[str, Any], ...]

    @classmethod
    def freeze(
        cls,
        items: Sequence[RetentionItem],
        *,
        dataset_binding_hash: str,
        fresh_partition_hash: str,
    ) -> "FixedItemBlockManifest":
        _require_sha256(dataset_binding_hash, "dataset binding")
        _require_sha256(fresh_partition_hash, "fresh partition")
        if not items:
            raise L4RetentionProtocolError(
                "fixed item/block manifest cannot be empty"
            )
        if len({item.item_id for item in items}) != len(items):
            raise L4RetentionProtocolError("fixed item IDs are not unique")
        rows = tuple(
            sorted(
                (ItemManifestRow.from_item(item) for item in items),
                key=lambda row: row.item_id_hash,
            )
        )
        if len({row.item_binding_hash for row in rows}) != len(rows):
            raise L4RetentionProtocolError(
                "fixed item bindings are not unique"
            )
        block_rows: list[dict[str, Any]] = []
        for block_hash in sorted({row.block_id_hash for row in rows}):
            members = tuple(
                row.item_id_hash
                for row in rows
                if row.block_id_hash == block_hash
            )
            block_rows.append(
                {
                    "block_id_hash": block_hash,
                    "item_count": len(members),
                    "item_set_hash": stable_hash(
                        {"item_id_hashes": list(members)}
                    ),
                }
            )
        manifest = cls(
            dataset_binding_hash=dataset_binding_hash,
            fresh_partition_hash=fresh_partition_hash,
            item_rows=rows,
            block_rows=tuple(block_rows),
        )
        manifest.verify()
        return manifest

    def verify(self) -> None:
        _require_sha256(self.dataset_binding_hash, "dataset binding")
        _require_sha256(self.fresh_partition_hash, "fresh partition")
        if not self.item_rows or not self.block_rows:
            raise L4RetentionProtocolError("fixed manifest is empty")
        item_hashes = tuple(row.item_id_hash for row in self.item_rows)
        if (
            item_hashes != tuple(sorted(item_hashes))
            or len(set(item_hashes)) != len(item_hashes)
        ):
            raise L4RetentionProtocolError(
                "fixed manifest item ordering drifted"
            )
        for row in self.item_rows:
            for value in (
                row.item_id_hash,
                row.block_id_hash,
                row.item_binding_hash,
            ):
                _require_sha256(value, "fixed manifest row hash")
            if row.support_count <= 0 or row.document_count < row.support_count:
                raise L4RetentionProtocolError(
                    "fixed manifest row counts are invalid"
                )
        expected_blocks = sorted({row.block_id_hash for row in self.item_rows})
        observed_blocks = [str(row.get("block_id_hash") or "") for row in self.block_rows]
        if observed_blocks != expected_blocks:
            raise L4RetentionProtocolError(
                "fixed block manifest does not cover the item set exactly"
            )
        for block in self.block_rows:
            block_hash = str(block["block_id_hash"])
            members = tuple(
                row.item_id_hash
                for row in self.item_rows
                if row.block_id_hash == block_hash
            )
            if (
                block.get("item_count") != len(members)
                or block.get("item_set_hash")
                != stable_hash({"item_id_hashes": list(members)})
            ):
                raise L4RetentionProtocolError(
                    "fixed block manifest content drifted"
                )

    @property
    def manifest_hash(self) -> str:
        return stable_hash(
            {
                "dataset_binding_hash": self.dataset_binding_hash,
                "fresh_partition_hash": self.fresh_partition_hash,
                "item_rows": [row.to_dict() for row in self.item_rows],
                "block_rows": [dict(row) for row in self.block_rows],
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        self.verify()
        return {
            "manifest_hash": self.manifest_hash,
            "dataset_binding_hash": self.dataset_binding_hash,
            "fresh_partition_hash": self.fresh_partition_hash,
            "item_count": len(self.item_rows),
            "block_count": len(self.block_rows),
            "total_support_count": sum(
                row.support_count for row in self.item_rows
            ),
            "item_binding_set_hash": stable_hash(
                {
                    "item_binding_hashes": [
                        row.item_binding_hash for row in self.item_rows
                    ]
                }
            ),
            "block_manifest_hash": stable_hash(
                [dict(row) for row in self.block_rows]
            ),
            "raw_item_or_block_ids_persisted": False,
        }

    def require_exact_items(self, items: Sequence[RetentionItem]) -> None:
        observed = FixedItemBlockManifest.freeze(
            items,
            dataset_binding_hash=self.dataset_binding_hash,
            fresh_partition_hash=self.fresh_partition_hash,
        )
        if observed != self or observed.manifest_hash != self.manifest_hash:
            raise L4RetentionProtocolError(
                "execution items differ from the fixed item/block manifest"
            )


@dataclass(frozen=True)
class ArchiveEpochBinding:
    """Binding from the existing archive to retained P and candidate Q."""

    archive_hash: str
    incumbent_node_id: str
    incumbent_node_hash: str
    evaluator_epoch_id: str
    p_hypothesis_id: str
    p_hypothesis_hash: str
    q_hypothesis_id: str
    q_hypothesis_hash: str

    @classmethod
    def from_archive(
        cls,
        archive: PolicyArchive,
        *,
        p_hypothesis_id: str,
        q_hypothesis_id: str,
    ) -> "ArchiveEpochBinding":
        if archive.incumbent_id is None:
            raise L4RetentionProtocolError(
                "retention requires an existing incumbent archive node"
            )
        incumbent = archive.nodes.get(archive.incumbent_id)
        if (
            incumbent is None
            or incumbent.status is not ArchiveNodeStatus.INCUMBENT
        ):
            raise L4RetentionProtocolError(
                "archive incumbent node is unavailable"
            )
        if p_hypothesis_id not in incumbent.active_hypothesis_ids:
            raise L4RetentionProtocolError(
                "P is not retained by the incumbent archive node"
            )
        if q_hypothesis_id in incumbent.active_hypothesis_ids:
            raise L4RetentionProtocolError(
                "Q already belongs to the retained incumbent"
            )
        if p_hypothesis_id == q_hypothesis_id:
            raise L4RetentionProtocolError("P and Q must be distinct")
        try:
            p_program = archive.hypotheses[p_hypothesis_id]
            q_program = archive.hypotheses[q_hypothesis_id]
        except KeyError as exc:
            raise L4RetentionProtocolError(
                "archive does not contain both frozen hypotheses"
            ) from exc
        if p_program.status is not HypothesisStatus.PROMOTED:
            raise L4RetentionProtocolError(
                "P must be the promoted retained hypothesis"
            )
        if q_program.status not in {
            HypothesisStatus.CANDIDATE,
            HypothesisStatus.SHADOW,
        }:
            raise L4RetentionProtocolError(
                "Q must be a prospective candidate or shadow hypothesis"
            )
        if (
            not incumbent.evaluator_epoch_id
            or p_program.evaluator_epoch != incumbent.evaluator_epoch_id
            or q_program.evaluator_epoch != incumbent.evaluator_epoch_id
        ):
            raise L4RetentionProtocolError(
                "P, Q, and the incumbent must share one evaluator epoch"
            )
        archive_payload = archive.to_dict()
        binding = cls(
            archive_hash=str(archive_payload["archive_hash"]),
            incumbent_node_id=incumbent.id,
            incumbent_node_hash=incumbent.payload_hash,
            evaluator_epoch_id=incumbent.evaluator_epoch_id,
            p_hypothesis_id=p_hypothesis_id,
            p_hypothesis_hash=p_program.payload_hash,
            q_hypothesis_id=q_hypothesis_id,
            q_hypothesis_hash=q_program.payload_hash,
        )
        binding.verify()
        return binding

    def verify(self) -> None:
        for value in (
            self.archive_hash,
            self.incumbent_node_hash,
            self.p_hypothesis_hash,
            self.q_hypothesis_hash,
        ):
            _require_sha256(value, "archive/epoch binding hash")
        for value in (
            self.incumbent_node_id,
            self.evaluator_epoch_id,
            self.p_hypothesis_id,
            self.q_hypothesis_id,
        ):
            if not isinstance(value, str) or not value:
                raise L4RetentionProtocolError(
                    "archive/epoch binding identity is missing"
                )
        if self.p_hypothesis_id == self.q_hypothesis_id:
            raise L4RetentionProtocolError("P and Q must be distinct")

    def safe_payload(self) -> dict[str, Any]:
        self.verify()
        return {
            "archive_hash": self.archive_hash,
            "incumbent_node_id_hash": stable_hash(
                {"node_id": self.incumbent_node_id}
            ),
            "incumbent_node_hash": self.incumbent_node_hash,
            "evaluator_epoch_id_hash": stable_hash(
                {"evaluator_epoch_id": self.evaluator_epoch_id}
            ),
            "p_hypothesis_id_hash": stable_hash(
                {"hypothesis_id": self.p_hypothesis_id}
            ),
            "p_hypothesis_hash": self.p_hypothesis_hash,
            "q_hypothesis_id_hash": stable_hash(
                {"hypothesis_id": self.q_hypothesis_id}
            ),
            "q_hypothesis_hash": self.q_hypothesis_hash,
            "raw_archive_id_text_persisted": False,
        }


@dataclass(frozen=True)
class FrozenOperatorSlot:
    slot_id: str
    operator_id: str
    implementation_hash: str
    hypothesis_id: str
    hypothesis_hash: str
    candidate_budget: int

    def verify(self) -> None:
        if self.slot_id not in SLOT_IDS:
            raise L4RetentionProtocolError("operator slot is not P or Q")
        if not isinstance(self.operator_id, str) or not self.operator_id:
            raise L4RetentionProtocolError("operator ID is missing")
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id:
            raise L4RetentionProtocolError("operator hypothesis ID is missing")
        _require_sha256(self.implementation_hash, "operator implementation")
        _require_sha256(self.hypothesis_hash, "operator hypothesis")
        if not isinstance(self.candidate_budget, int) or self.candidate_budget < TOP_K:
            raise L4RetentionProtocolError(
                "operator candidate budget must cover recall@5"
            )

    def safe_payload(self) -> dict[str, Any]:
        self.verify()
        return {
            "slot_id": self.slot_id,
            "operator_id_hash": stable_hash(
                {"operator_id": self.operator_id}
            ),
            "implementation_hash": self.implementation_hash,
            "hypothesis_id_hash": stable_hash(
                {"hypothesis_id": self.hypothesis_id}
            ),
            "hypothesis_hash": self.hypothesis_hash,
            "candidate_budget": self.candidate_budget,
        }


class RankedOperator(Protocol):
    operator_id: str
    implementation_hash: str

    def retrieve(
        self, query: OperatorQuery, *, candidate_budget: int
    ) -> Sequence[str]:
        """Return a unique, ordered subset of ``query.document_ids``."""


OperatorFactory = Callable[[FrozenOperatorSlot], RankedOperator]


@dataclass(frozen=True)
class L4RetentionPlan:
    manifest: FixedItemBlockManifest
    archive_epoch: ArchiveEpochBinding
    operator_slots: tuple[FrozenOperatorSlot, FrozenOperatorSlot]
    authorization_hash: str
    execution_root_hash: str
    baseline_candidate_budget: int = TOP_K

    @classmethod
    def freeze(
        cls,
        *,
        manifest: FixedItemBlockManifest,
        archive_epoch: ArchiveEpochBinding,
        operator_slots: Sequence[FrozenOperatorSlot],
        authorization_hash: str,
        execution_root: str | Path,
        baseline_candidate_budget: int = TOP_K,
    ) -> "L4RetentionPlan":
        slots = tuple(operator_slots)
        if len(slots) != 2:
            raise L4RetentionProtocolError(
                "L4 requires exactly two frozen operator slots"
            )
        plan = cls(
            manifest=manifest,
            archive_epoch=archive_epoch,
            operator_slots=(slots[0], slots[1]),
            authorization_hash=_require_sha256(
                authorization_hash, "execution authorization"
            ),
            execution_root_hash=execution_root_binding_hash(execution_root),
            baseline_candidate_budget=baseline_candidate_budget,
        )
        plan.verify()
        return plan

    def verify(self) -> None:
        self.manifest.verify()
        self.archive_epoch.verify()
        _require_sha256(self.authorization_hash, "execution authorization")
        _require_sha256(self.execution_root_hash, "execution root binding")
        if (
            not isinstance(self.baseline_candidate_budget, int)
            or self.baseline_candidate_budget < 0
        ):
            raise L4RetentionProtocolError(
                "baseline candidate budget is invalid"
            )
        slots = tuple(self.operator_slots)
        if tuple(slot.slot_id for slot in slots) != SLOT_IDS:
            raise L4RetentionProtocolError(
                "operator slots must be frozen in P,Q order"
            )
        for slot in slots:
            slot.verify()
        if slots[0].candidate_budget != slots[1].candidate_budget:
            raise L4RetentionProtocolError(
                "P and Q must receive the same candidate budget"
            )
        if (
            slots[0].hypothesis_id != self.archive_epoch.p_hypothesis_id
            or slots[0].hypothesis_hash
            != self.archive_epoch.p_hypothesis_hash
            or slots[1].hypothesis_id != self.archive_epoch.q_hypothesis_id
            or slots[1].hypothesis_hash
            != self.archive_epoch.q_hypothesis_hash
        ):
            raise L4RetentionProtocolError(
                "operator slots drifted from the archive/epoch binding"
            )

    def safe_payload(self) -> dict[str, Any]:
        self.verify()
        per_slot_budget = self.operator_slots[0].candidate_budget
        arm_plan = [
            {"arm_id": arm, "active_slots": list(ARM_SLOT_MAP[arm])}
            for arm in ARM_IDS
        ]
        body = {
            "version": PROTOCOL_VERSION,
            "manifest": self.manifest.safe_payload(),
            "archive_epoch": self.archive_epoch.safe_payload(),
            "operator_slots": [
                slot.safe_payload() for slot in self.operator_slots
            ],
            "authorization_hash": self.authorization_hash,
            "execution_root_hash": self.execution_root_hash,
            "arm_plan": arm_plan,
            "arm_plan_hash": stable_hash(arm_plan),
            "operator_slot_count": 2,
            "ranking_reuse_policy": RANKING_REUSE_POLICY,
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "rrf_constant": RRF_CONSTANT,
                "tie_break": "document_id_lexical_ascending",
                "top_k": TOP_K,
            },
            "budget": {
                "same_item_manifest_for_all_arms": True,
                "reserved_operator_slots_per_item_for_all_arms": 2,
                "candidate_budget_per_operator_slot": per_slot_budget,
                "baseline_candidate_budget": self.baseline_candidate_budget,
                "final_ranking_budget_per_arm": TOP_K,
                "operator_work_unit_count": 2
                * len(self.manifest.item_rows),
            },
            "evaluation": {
                "primary_metric": PRIMARY_METRIC,
                "retention_estimand": RETENTION_ESTIMAND,
                "novelty_estimand": NOVELTY_ESTIMAND,
                "forgetting_estimand": FORGETTING_ESTIMAND,
                "support_labels_visible_to_operators": False,
                "score_after_all_operator_terminals": True,
                "online_evaluator_calls": 0,
            },
            "execution_policy": {
                "fresh_root_policy": FRESH_ROOT_POLICY,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
                "outcome_dependent_arm_changes": 0,
            },
            "raw_content_persisted": False,
        }
        return body

    @property
    def plan_hash(self) -> str:
        return stable_hash(self.safe_payload())


def deterministic_rrf(
    rankings: Sequence[Sequence[str]],
    *,
    top_k: int = TOP_K,
    rrf_constant: int = RRF_CONSTANT,
) -> tuple[str, ...]:
    """Fuse frozen rankings with exact rational scores and lexical ties."""

    if top_k != TOP_K:
        raise L4RetentionProtocolError("primary protocol top_k must remain 5")
    if rrf_constant != RRF_CONSTANT:
        raise L4RetentionProtocolError(
            "primary protocol RRF constant drifted"
        )
    scores: dict[str, Fraction] = {}
    for ranking in rankings:
        normalized = _require_nonempty_unique_strings(
            tuple(ranking), "RRF source ranking", allow_empty=True
        )
        for rank, document_id in enumerate(normalized, start=1):
            scores[document_id] = scores.get(document_id, Fraction()) + Fraction(
                1, rrf_constant + rank
            )
    ordered = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return tuple(ordered[:top_k])


@dataclass(frozen=True)
class _WorkUnit:
    item: RetentionItem = field(repr=False)
    slot: FrozenOperatorSlot

    @property
    def key(self) -> tuple[str, str]:
        return (self.item.item_id_hash, self.slot.slot_id)


def _validated_ranking(
    value: Sequence[str],
    *,
    work: _WorkUnit,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise L4RetentionProtocolError(
            "operator returned a scalar instead of a ranking"
        )
    ranking = _require_nonempty_unique_strings(
        tuple(value), "operator ranking", allow_empty=True
    )
    if len(ranking) > work.slot.candidate_budget:
        raise L4RetentionProtocolError(
            "operator exceeded its frozen candidate budget"
        )
    if not set(ranking).issubset(work.item.document_ids):
        raise L4RetentionProtocolError(
            "operator ranking escaped the frozen item corpus"
        )
    return ranking


def _aggregate_arm(
    rows: Sequence[Mapping[str, Any]], arm_id: str
) -> dict[str, Any]:
    support_total = sum(int(row["support_total"]) for row in rows)
    support_hits = sum(int(row["support_hits"]) for row in rows)
    macro = sum(
        Fraction(int(row["support_hits"]), int(row["support_total"]))
        for row in rows
    ) / len(rows)
    closure = [
        {
            "item_id_hash": row["item_id_hash"],
            "ranking_hash": row["ranking_hash"],
            "support_hits": row["support_hits"],
            "support_total": row["support_total"],
        }
        for row in rows
    ]
    return {
        "arm_id": arm_id,
        "item_count": len(rows),
        "support_hit_count": support_hits,
        "support_total": support_total,
        "micro_support_recall_at_5": float(
            Fraction(support_hits, support_total)
        ),
        "macro_support_recall_at_5": float(macro),
        "items_with_any_support_hit": sum(
            int(row["support_hits"] > 0) for row in rows
        ),
        "ranking_score_closure_hash": stable_hash(closure),
    }


def _score_all_arms(
    *,
    plan: L4RetentionPlan,
    items: Sequence[RetentionItem],
    rankings: Mapping[tuple[str, str], tuple[str, ...]],
) -> dict[str, Any]:
    # Gold is first dereferenced here, after all operator futures have joined.
    item_rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: value.item_id_hash):
        if len(item.baseline_ranked_document_ids) > plan.baseline_candidate_budget:
            raise L4RetentionProtocolError(
                "baseline ranking exceeded its frozen candidate budget"
            )
        baseline = tuple(item.baseline_ranked_document_ids)
        arm_rankings: dict[str, tuple[str, ...]] = {}
        arm_hits: dict[str, frozenset[str]] = {}
        for arm_id in ARM_IDS:
            sources: list[tuple[str, ...]] = []
            if baseline:
                sources.append(baseline)
            for slot_id in ARM_SLOT_MAP[arm_id]:
                sources.append(rankings[(item.item_id_hash, slot_id)])
            fused = deterministic_rrf(sources)
            arm_rankings[arm_id] = fused
            arm_hits[arm_id] = frozenset(fused).intersection(
                item.support_document_ids
            )
        item_rows.append(
            {
                "item_id_hash": item.item_id_hash,
                "block_id_hash": item.block_id_hash,
                "support_total": len(item.support_document_ids),
                "arms": {
                    arm_id: {
                        "support_hits": len(arm_hits[arm_id]),
                        "ranking_hash": stable_hash(
                            {"ranked_document_ids": list(arm_rankings[arm_id])}
                        ),
                    }
                    for arm_id in ARM_IDS
                },
                "hit_sets": arm_hits,
            }
        )

    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        aggregate_rows = [
            {
                "item_id_hash": row["item_id_hash"],
                "support_total": row["support_total"],
                **row["arms"][arm_id],
            }
            for row in item_rows
        ]
        arm_metrics[arm_id] = _aggregate_arm(aggregate_rows, arm_id)

    support_total = arm_metrics["P_plus_Q"]["support_total"]
    y = {
        arm_id: Fraction(
            arm_metrics[arm_id]["support_hit_count"], support_total
        )
        for arm_id in ARM_IDS
    }
    retained_added = 0
    retained_displaced = 0
    novel_added = 0
    forgotten = 0
    retention_gain_items = 0
    retention_harm_items = 0
    retention_tie_items = 0
    novelty_gain_items = 0
    forgetting_items = 0
    for row in item_rows:
        hits = row["hit_sets"]
        retained_added += len(hits["P_plus_Q"] - hits["Q"])
        retained_displaced += len(hits["Q"] - hits["P_plus_Q"])
        novel_added += len(hits["P_plus_Q"] - hits["P"])
        lost = len(hits["P"] - hits["P_plus_Q"])
        forgotten += lost
        forgetting_items += int(lost > 0)
        q_count = len(hits["Q"])
        pq_count = len(hits["P_plus_Q"])
        retention_gain_items += int(pq_count > q_count)
        retention_harm_items += int(pq_count < q_count)
        retention_tie_items += int(pq_count == q_count)
        novelty_gain_items += int(len(hits["P_plus_Q"] - hits["P"]) > 0)

    retention_delta = y["P_plus_Q"] - y["Q"]
    novelty_delta = y["P_plus_Q"] - y["P"]
    novelty_rate = Fraction(novel_added, support_total)
    forgetting_rate = Fraction(forgotten, support_total)
    if retention_delta != Fraction(retained_added - retained_displaced, support_total):
        raise L4RetentionProtocolError("retention decomposition drifted")
    if novelty_delta != novelty_rate - forgetting_rate:
        raise L4RetentionProtocolError("novelty/forgetting decomposition drifted")

    block_aggregates: list[dict[str, Any]] = []
    for block_id_hash in sorted(
        {str(row["block_id_hash"]) for row in item_rows}
    ):
        block_items = [
            row for row in item_rows if row["block_id_hash"] == block_id_hash
        ]
        block_arms: dict[str, Any] = {}
        for arm_id in ARM_IDS:
            block_arms[arm_id] = _aggregate_arm(
                [
                    {
                        "item_id_hash": row["item_id_hash"],
                        "support_total": row["support_total"],
                        **row["arms"][arm_id],
                    }
                    for row in block_items
                ],
                arm_id,
            )
        block_aggregates.append(
            {
                "block_id_hash": block_id_hash,
                "item_count": len(block_items),
                "arm_metrics": block_arms,
            }
        )

    return {
        "primary_metric": PRIMARY_METRIC,
        "arm_metrics": arm_metrics,
        "retention": {
            "estimand": RETENTION_ESTIMAND,
            "delta": float(retention_delta),
            "retained_support_added_count": retained_added,
            "q_support_displaced_count": retained_displaced,
            "paired_gain_item_count": retention_gain_items,
            "paired_harm_item_count": retention_harm_items,
            "paired_tie_item_count": retention_tie_items,
        },
        "novelty": {
            "estimand": NOVELTY_ESTIMAND,
            "net_delta": float(novelty_delta),
            "new_support_coverage_rate": float(novelty_rate),
            "new_support_coverage_count": novel_added,
            "items_with_new_support_coverage": novelty_gain_items,
            "q_over_empty_delta": float(y["Q"] - y["empty"]),
        },
        "forgetting": {
            "estimand": FORGETTING_ESTIMAND,
            "support_forgetting_rate": float(forgetting_rate),
            "forgotten_support_count": forgotten,
            "item_count_with_forgetting": forgetting_items,
        },
        "block_aggregates": block_aggregates,
        "scored_item_closure_hash": stable_hash(
            [
                {
                    "item_id_hash": row["item_id_hash"],
                    "block_id_hash": row["block_id_hash"],
                    "support_total": row["support_total"],
                    "arms": row["arms"],
                }
                for row in item_rows
            ]
        ),
        "raw_item_document_or_support_ids_persisted": False,
    }


def execute_l4_retention_plan(
    *,
    plan: L4RetentionPlan,
    items: Sequence[RetentionItem],
    operator_factory: OperatorFactory,
    execution_root: str | Path,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Consume one frozen authorization and write one aggregate-only report.

    Preflight mismatches happen before root creation.  Once the root is
    created, any exception leaves a terminal failure receipt and the root can
    never be reused by this API.  Operator failures are not retried.
    """

    plan.verify()
    plan.manifest.require_exact_items(items)
    if any(
        len(item.baseline_ranked_document_ids)
        > plan.baseline_candidate_budget
        for item in items
    ):
        raise L4RetentionProtocolError(
            "baseline ranking exceeded its frozen candidate budget"
        )
    root = _canonical_new_root(execution_root)
    if execution_root_binding_hash(root) != plan.execution_root_hash:
        raise L4RetentionProtocolError(
            "execution root differs from the frozen root binding"
        )
    if root.exists() or root.is_symlink():
        raise L4RetentionProtocolError(
            "fresh execution root already exists; replay is forbidden"
        )
    work_units = tuple(
        _WorkUnit(item=item, slot=slot)
        for item in sorted(items, key=lambda value: value.item_id_hash)
        for slot in plan.operator_slots
    )
    if max_workers is None:
        max_workers = len(work_units)
    if (
        not isinstance(max_workers, int)
        or max_workers <= 0
        or max_workers > len(work_units)
    ):
        raise L4RetentionProtocolError(
            "operator concurrency must be between one and work-unit count"
        )

    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise L4RetentionProtocolError(
            "fresh execution root already exists; replay is forbidden"
        ) from exc
    stage = "freeze"
    attempted = 0
    completed = 0
    counter_lock = threading.Lock()
    try:
        freeze_body = {
            **plan.safe_payload(),
            "plan_hash": plan.plan_hash,
        }
        _write_json_exclusive(root / FREEZE_FILENAME, freeze_body)
        stage = "authorization_consumption"
        consumption_body = {
            "version": PROTOCOL_VERSION,
            "authorization_hash": plan.authorization_hash,
            "plan_hash": plan.plan_hash,
            "execution_root_hash": plan.execution_root_hash,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {
            **consumption_body,
            "consumption_hash": stable_hash(consumption_body),
        }
        _write_json_exclusive(root / CONSUMPTION_FILENAME, consumption)

        stage = "operator_materialization"
        materialized: list[tuple[_WorkUnit, RankedOperator]] = []
        for work in work_units:
            operator = operator_factory(work.slot)
            if (
                getattr(operator, "operator_id", None)
                != work.slot.operator_id
                or getattr(operator, "implementation_hash", None)
                != work.slot.implementation_hash
                or not callable(getattr(operator, "retrieve", None))
            ):
                raise L4RetentionProtocolError(
                    "materialized operator differs from its frozen slot"
                )
            materialized.append((work, operator))

        stage = "operator_execution"

        def run_one(
            pair: tuple[_WorkUnit, RankedOperator]
        ) -> tuple[tuple[str, str], tuple[str, ...]]:
            nonlocal attempted, completed
            work, operator = pair
            with counter_lock:
                attempted += 1
            value = operator.retrieve(
                work.item.gold_free_query(),
                candidate_budget=work.slot.candidate_budget,
            )
            ranking = _validated_ranking(value, work=work)
            with counter_lock:
                completed += 1
            return work.key, ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="l4-retention",
        ) as executor:
            futures = [executor.submit(run_one, pair) for pair in materialized]
            terminal_rows = [future.result() for future in futures]
        if attempted != len(work_units) or completed != len(work_units):
            raise L4RetentionProtocolError(
                "operator work-unit closure is incomplete"
            )
        rankings = dict(terminal_rows)
        if len(rankings) != len(work_units):
            raise L4RetentionProtocolError(
                "operator work-unit keys are not one-to-one"
            )

        stage = "offline_support_scoring"
        measurement = _score_all_arms(
            plan=plan,
            items=items,
            rankings=rankings,
        )
        ranking_receipt_rows = [
            {
                "item_id_hash": item_hash,
                "slot_id": slot_id,
                "ranking_hash": stable_hash(
                    {"ranked_document_ids": list(ranking)}
                ),
                "ranking_count": len(ranking),
            }
            for (item_hash, slot_id), ranking in sorted(rankings.items())
        ]
        report_body: dict[str, Any] = {
            "version": PROTOCOL_VERSION,
            "valid": True,
            "plan_hash": plan.plan_hash,
            "manifest_hash": plan.manifest.manifest_hash,
            "archive_hash": plan.archive_epoch.archive_hash,
            "evaluator_epoch_id_hash": stable_hash(
                {
                    "evaluator_epoch_id": plan.archive_epoch.evaluator_epoch_id
                }
            ),
            "measurement": measurement,
            "execution": {
                "fresh_root_policy": FRESH_ROOT_POLICY,
                "ranking_reuse_policy": RANKING_REUSE_POLICY,
                "operator_work_unit_count": len(work_units),
                "operator_call_count": attempted,
                "operator_terminal_count": completed,
                "configured_maximum_concurrency": max_workers,
                "ranking_receipt_set_hash": stable_hash(
                    ranking_receipt_rows
                ),
                "all_operator_terminals_joined_before_gold_scoring": True,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
                "outcome_dependent_arm_changes": 0,
            },
            "model_calls": 0,
            "online_evaluator_calls": 0,
            "offline_evaluator_calls": len(items) * len(ARM_IDS),
            "sealed_or_test_content_accessed": False,
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_hash": stable_hash(report_body)}
        stage = "report_persistence"
        _write_json_exclusive(root / REPORT_FILENAME, report)
        persisted = json.loads((root / REPORT_FILENAME).read_text("utf-8"))
        declared = persisted.pop("report_hash", None)
        if declared != stable_hash(persisted):
            raise L4RetentionProtocolError(
                "persisted aggregate report hash drifted"
            )
        return report
    except Exception as exc:
        failure_body = {
            "version": PROTOCOL_VERSION,
            "valid": False,
            "plan_hash": plan.plan_hash,
            "manifest_hash": plan.manifest.manifest_hash,
            "failure_stage": stage,
            "error_type_hash": stable_hash(
                {"error_type": type(exc).__name__}
            ),
            "operator_work_unit_count": len(work_units),
            "operator_attempt_count": attempted,
            "operator_terminal_count": completed,
            "authorization_consumed": (root / CONSUMPTION_FILENAME).is_file(),
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        failure = {**failure_body, "failure_hash": stable_hash(failure_body)}
        try:
            _write_json_exclusive(root / FAILURE_FILENAME, failure)
        except Exception:
            pass
        raise L4RetentionProtocolError(
            "frozen L4 retention run failed and cannot be replayed"
        ) from exc

"""Eager-submit three-arm scheduler for ERASER Evidence Inference.

Semantic preparation happens once before either pool opens and executes no
action.  Agent and RAW are separate logical tasks in one local pool; official
HippoRAG uses a separate pool.  Every one of the ``3 * n`` futures is submitted
before the scheduler reads the first result.  Post-join work is limited to
binding the already fixed actions and selected-pair measurements into the
exact feature bridge and content-free seals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from assumption_agent.benchmarks import (
    eraser_evidence_inference_exact_feature_bridge_v1 as feature_bridge,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_local_runtime_v1 as local_runtime,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_e3_runner_v1 as runner,
)


VERSION = "eraser_evidence_inference_three_arm_scheduler_v1"
BLOCK_ORDER = runner.BLOCK_ORDER
ANCHOR_BLOCKS = ("A_hold", "M_search")
LOCAL_WORKER_CAP = 32
HIPPORAG_WORKER_CAP = 32


class EraserEvidenceInferenceThreeArmSchedulerError(RuntimeError):
    """A barrier, independent arm, feature, seal, or archive drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "scheduler payload is not exact JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_json_text(value: Mapping[str, object]) -> str:
    return _canonical_bytes(dict(value)).decode("ascii")


def _parse_json_object(value: str, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            f"{field} is not canonical JSON text"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            f"{field} is unreadable"
        ) from exc
    if not isinstance(parsed, dict) or _canonical_json_text(parsed) != value:
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            f"{field} is not a canonical JSON object"
        )
    return parsed


def _self_hashed(body: Mapping[str, object], field: str) -> dict[str, object]:
    if field in body:
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "self-hash field already exists"
        )
    payload = dict(body)
    payload[field] = stable_hash(payload)
    return payload


def _arm_receipt(
    *,
    arm: str,
    block: str,
    item_count: int,
    retrieval_matrix_sha256: str,
    item_commitment_set_sha256: str,
) -> dict[str, object]:
    body = {
        "schema": f"{VERSION}_{arm}_arm_receipt",
        "version": VERSION,
        "arm": arm,
        "block": block,
        "item_count": item_count,
        "retrieval_matrix_sha256": retrieval_matrix_sha256,
        "item_commitment_set_sha256": item_commitment_set_sha256,
        "canonical_order": "ascending_item_commitment_sha256",
        "independent_logical_future": True,
        "labels_gold_evaluator_source_or_network_calls": 0,
        "raw_content_persisted": False,
    }
    return _self_hashed(body, f"{arm}_arm_receipt_sha256")


@dataclass(frozen=True)
class HippoArmSeal:
    """All-block ordinal-only Hippo arm seal owned by this scheduler."""

    block: str
    rows: tuple[runner.HippoRetrieval, ...]
    retrieval_matrix_sha256: str
    item_commitment_set_sha256: str
    hipporag_arm_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCK_ORDER
            or not isinstance(self.rows, tuple)
            or len(self.rows) != runner.BLOCK_COUNTS[self.block]
            or any(not isinstance(row, runner.HippoRetrieval) for row in self.rows)
            or self.rows
            != tuple(sorted(self.rows, key=lambda row: row.item_commitment_sha256))
            or len({row.item_commitment_sha256 for row in self.rows})
            != len(self.rows)
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduler Hippo arm matrix drifted"
            )
        matrix = stable_hash([row.payload() for row in self.rows])
        item_set = stable_hash(
            [row.item_commitment_sha256 for row in self.rows]
        )
        receipt = _arm_receipt(
            arm="hipporag",
            block=self.block,
            item_count=len(self.rows),
            retrieval_matrix_sha256=matrix,
            item_commitment_set_sha256=item_set,
        )
        if (
            self.retrieval_matrix_sha256 != matrix
            or self.item_commitment_set_sha256 != item_set
            or self.hipporag_arm_receipt_sha256
            != receipt["hipporag_arm_receipt_sha256"]
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduler Hippo arm seal binding drifted"
            )

    @property
    def receipt(self) -> dict[str, object]:
        return _arm_receipt(
            arm="hipporag",
            block=self.block,
            item_count=len(self.rows),
            retrieval_matrix_sha256=self.retrieval_matrix_sha256,
            item_commitment_set_sha256=self.item_commitment_set_sha256,
        )

    @property
    def by_item(self) -> Mapping[str, runner.HippoRetrieval]:
        return {row.item_commitment_sha256: row for row in self.rows}


def seal_hippo_arm(
    *, block: str, rows: Sequence[runner.HippoRetrieval]
) -> HippoArmSeal:
    if block not in BLOCK_ORDER or any(
        not isinstance(row, runner.HippoRetrieval) for row in rows
    ):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "scheduler Hippo arm inputs are invalid"
        )
    canonical = tuple(sorted(rows, key=lambda row: row.item_commitment_sha256))
    matrix = stable_hash([row.payload() for row in canonical])
    item_set = stable_hash([row.item_commitment_sha256 for row in canonical])
    receipt = _arm_receipt(
        arm="hipporag",
        block=block,
        item_count=len(canonical),
        retrieval_matrix_sha256=matrix,
        item_commitment_set_sha256=item_set,
    )
    return HippoArmSeal(
        block=block,
        rows=canonical,
        retrieval_matrix_sha256=matrix,
        item_commitment_set_sha256=item_set,
        hipporag_arm_receipt_sha256=str(
            receipt["hipporag_arm_receipt_sha256"]
        ),
    )


@dataclass(frozen=True)
class RawArmSeal:
    """All-block independent RAW/R0 seal owned by this scheduler."""

    block: str
    rows: tuple[runner.RawRetrieval, ...]
    retrieval_matrix_sha256: str
    item_commitment_set_sha256: str
    raw_arm_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCK_ORDER
            or not isinstance(self.rows, tuple)
            or len(self.rows) != runner.BLOCK_COUNTS[self.block]
            or any(not isinstance(row, runner.RawRetrieval) for row in self.rows)
            or self.rows
            != tuple(sorted(self.rows, key=lambda row: row.item_commitment_sha256))
            or len({row.item_commitment_sha256 for row in self.rows})
            != len(self.rows)
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduler RAW arm matrix drifted"
            )
        matrix = stable_hash([row.payload() for row in self.rows])
        item_set = stable_hash(
            [row.item_commitment_sha256 for row in self.rows]
        )
        receipt = _arm_receipt(
            arm="raw",
            block=self.block,
            item_count=len(self.rows),
            retrieval_matrix_sha256=matrix,
            item_commitment_set_sha256=item_set,
        )
        if (
            self.retrieval_matrix_sha256 != matrix
            or self.item_commitment_set_sha256 != item_set
            or self.raw_arm_receipt_sha256 != receipt["raw_arm_receipt_sha256"]
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduler RAW arm seal binding drifted"
            )

    @property
    def receipt(self) -> dict[str, object]:
        return _arm_receipt(
            arm="raw",
            block=self.block,
            item_count=len(self.rows),
            retrieval_matrix_sha256=self.retrieval_matrix_sha256,
            item_commitment_set_sha256=self.item_commitment_set_sha256,
        )

    @property
    def by_item(self) -> Mapping[str, runner.RawRetrieval]:
        return {row.item_commitment_sha256: row for row in self.rows}


def seal_raw_arm(
    *, block: str, rows: Sequence[runner.RawRetrieval]
) -> RawArmSeal:
    if block not in BLOCK_ORDER or any(
        not isinstance(row, runner.RawRetrieval) for row in rows
    ):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "scheduler RAW arm inputs are invalid"
        )
    canonical = tuple(sorted(rows, key=lambda row: row.item_commitment_sha256))
    matrix = stable_hash([row.payload() for row in canonical])
    item_set = stable_hash([row.item_commitment_sha256 for row in canonical])
    receipt = _arm_receipt(
        arm="raw",
        block=block,
        item_count=len(canonical),
        retrieval_matrix_sha256=matrix,
        item_commitment_set_sha256=item_set,
    )
    return RawArmSeal(
        block=block,
        rows=canonical,
        retrieval_matrix_sha256=matrix,
        item_commitment_set_sha256=item_set,
        raw_arm_receipt_sha256=str(receipt["raw_arm_receipt_sha256"]),
    )


@dataclass(frozen=True)
class ScheduledItemArtifact:
    """All three arms and the exact label-free feature build for one item."""

    block: str
    agent: local_runtime.AgentExecutionArtifact
    raw: local_runtime.RawExecutionArtifact
    hippo: local_runtime.HippoExecutionArtifact
    selected_pair_receipt: feature_bridge.SelectedPairSemanticReceipt
    exact_feature_build: feature_bridge.ExactDifferenceTraceBuild

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ORDER:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduled item block is invalid"
            )
        if (
            not isinstance(self.agent, local_runtime.AgentExecutionArtifact)
            or not isinstance(self.raw, local_runtime.RawExecutionArtifact)
            or not isinstance(self.hippo, local_runtime.HippoExecutionArtifact)
            or not isinstance(
                self.selected_pair_receipt,
                feature_bridge.SelectedPairSemanticReceipt,
            )
            or not isinstance(
                self.exact_feature_build,
                feature_bridge.ExactDifferenceTraceBuild,
            )
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduled item contains a foreign artifact type"
            )
        commitment = self.agent.item_commitment_sha256
        if (
            self.raw.item_commitment_sha256 != commitment
            or self.hippo.item_commitment_sha256 != commitment
            or self.hippo.block != self.block
            or self.raw.graph_sha256 != self.agent.graph_sha256
            or self.raw.semantic_tensor_sha256
            != self.agent.semantic_tensor_sha256
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "three-arm item binding drifted"
            )
        # RAW is a separate future and must reconstruct the same deterministic
        # R0 action without sharing the Agent action object.
        if self.raw.r0_action is self.agent.r0_action:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "RAW reused the Agent R0 action object"
            )
        if (
            self.raw.r0_action.trace_sha256
            != self.agent.r0_action.trace_sha256
            or self.raw.r0_action.behavior_sha256
            != self.agent.r0_action.behavior_sha256
            or self.raw.top5 != self.agent.r0_action.output_top5
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "independent RAW did not reconstruct Agent R0 exactly"
            )
        trace = self.exact_feature_build.difference_trace
        receipt = self.exact_feature_build.feature_receipt
        if (
            trace.item_commitment_sha256 != commitment
            or trace.r0_action_trace_sha256
            != self.agent.r0_action.trace_sha256
            or trace.r7_action_trace_sha256
            != self.agent.r7_action.trace_sha256
            or receipt.selected_pair_semantic_receipt_sha256
            != self.selected_pair_receipt.receipt_sha256
            or self.selected_pair_receipt.pair_rows != self.agent.pair_rows
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "exact feature build drifted from the Agent future"
            )
        try:
            feature_bridge.verify_exact_difference_trace_build(
                self.exact_feature_build
            )
        except feature_bridge.EraserExactFeatureBridgeError as exc:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "exact feature build verification failed"
            ) from exc

    @property
    def item_commitment_sha256(self) -> str:
        return self.agent.item_commitment_sha256

    @property
    def difference_trace(self) -> runner.DifferenceTrace:
        return self.exact_feature_build.difference_trace

    def payload(self) -> dict[str, object]:
        return {
            "item_commitment_sha256": self.item_commitment_sha256,
            "agent": self.agent.payload(),
            "raw": self.raw.payload(),
            "hipporag": self.hippo.payload(),
            "selected_pair_semantic_receipt": (
                self.selected_pair_receipt.payload()
            ),
            "difference_trace": self.difference_trace.payload(),
            "exact_feature_computation_receipt": (
                self.exact_feature_build.feature_receipt.payload()
            ),
            "independent_RAW_matches_Agent_R0": True,
            "RAW_action_object_reused": False,
        }


def _block_archive_body(
    *,
    block: str,
    items: Sequence[ScheduledItemArtifact],
    feature_seal: runner.FeatureSeal,
    hippo_arm_seal: HippoArmSeal,
    raw_arm_seal: RawArmSeal,
    hippo_retrieval_seal: runner.HippoRetrievalSeal | None,
    raw_retrieval_seal: runner.RawRetrievalSeal | None,
    local_workers: int,
    hippo_workers: int,
    global_item_count: int,
    global_submitted_count: int,
) -> dict[str, object]:
    return {
        "schema": f"{VERSION}_block_archive",
        "version": VERSION,
        "status": "complete_offline_three_arm_block",
        "block": block,
        "item_count": len(items),
        "logical_agent_raw_hipporag_task_count": 3 * len(items),
        "global_item_count_at_submission_barrier": global_item_count,
        "global_submitted_task_count_before_first_result": (
            global_submitted_count
        ),
        "global_first_result_read_after_submit_count": global_submitted_count,
        "all_3n_tasks_submitted_before_first_result": True,
        "local_agent_raw_pool_max_workers": local_workers,
        "official_hipporag_pool_max_workers": hippo_workers,
        "feature_receipt": feature_seal.receipt,
        "hipporag_arm_receipt": hippo_arm_seal.receipt,
        "raw_arm_receipt": raw_arm_seal.receipt,
        "anchor_hipporag_retrieval_matrix_sha256": (
            None
            if hippo_retrieval_seal is None
            else hippo_retrieval_seal.retrieval_matrix_sha256
        ),
        "anchor_raw_retrieval_receipt_sha256": (
            None
            if raw_retrieval_seal is None
            else raw_retrieval_seal.raw_retrieval_receipt_sha256
        ),
        "item_commitment_set_sha256": feature_seal.item_commitment_set_sha256,
        "items": [item.payload() for item in items],
        "full_square_pair_scan_performed": False,
        "labels_gold_utility_evaluator_or_source_accessed": False,
        "online_evaluator_calls": 0,
        "external_network_calls": 0,
        "raw_content_persisted": False,
    }


@dataclass(frozen=True)
class BlockThreeArmArtifact:
    block: str
    items: tuple[ScheduledItemArtifact, ...]
    feature_seal: runner.FeatureSeal
    hippo_arm_seal: HippoArmSeal
    raw_arm_seal: RawArmSeal
    hippo_retrieval_seal: runner.HippoRetrievalSeal | None
    raw_retrieval_seal: runner.RawRetrievalSeal | None
    local_pool_max_workers: int
    hippo_pool_max_workers: int
    global_item_count: int
    global_submitted_task_count: int
    archive_json: str
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCK_ORDER
            or not isinstance(self.items, tuple)
            or len(self.items) != runner.BLOCK_COUNTS[self.block]
            or any(
                not isinstance(item, ScheduledItemArtifact)
                or item.block != self.block
                for item in self.items
            )
            or self.items
            != tuple(
                sorted(
                    self.items,
                    key=lambda item: item.item_commitment_sha256,
                )
            )
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "block item artifact registry drifted"
            )
        if (
            not isinstance(self.feature_seal, runner.FeatureSeal)
            or not isinstance(self.hippo_arm_seal, HippoArmSeal)
            or not isinstance(self.raw_arm_seal, RawArmSeal)
            or self.feature_seal.block != self.block
            or self.hippo_arm_seal.block != self.block
            or self.raw_arm_seal.block != self.block
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "block feature or Hippo seal drifted"
            )
        commitments = tuple(item.item_commitment_sha256 for item in self.items)
        if (
            commitments != self.feature_seal.item_commitments
            or commitments
            != tuple(
                row.item_commitment_sha256
                for row in self.hippo_arm_seal.rows
            )
            or commitments
            != tuple(row.item_commitment_sha256 for row in self.raw_arm_seal.rows)
            or self.hippo_arm_seal.item_commitment_set_sha256
            != self.feature_seal.item_commitment_set_sha256
            or self.raw_arm_seal.item_commitment_set_sha256
            != self.feature_seal.item_commitment_set_sha256
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "block seal item registry differs from the three arms"
            )
        hippo_by_item = self.hippo_arm_seal.by_item
        raw_by_item = self.raw_arm_seal.by_item
        if any(
            hippo_by_item[item.item_commitment_sha256].sentence_count
            != item.difference_trace.sentence_count
            or hippo_by_item[item.item_commitment_sha256].top5 != item.hippo.top5
            or raw_by_item[item.item_commitment_sha256].sentence_count
            != item.difference_trace.sentence_count
            or raw_by_item[item.item_commitment_sha256].top5 != item.raw.top5
            or raw_by_item[item.item_commitment_sha256].action_trace_sha256
            != item.raw.r0_action.trace_sha256
            for item in self.items
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduler arm seal rows drifted from future artifacts"
            )
        if self.block in ANCHOR_BLOCKS:
            if (
                not isinstance(
                    self.hippo_retrieval_seal, runner.HippoRetrievalSeal
                )
                or not isinstance(
                    self.raw_retrieval_seal, runner.RawRetrievalSeal
                )
                or self.hippo_retrieval_seal.rows != self.hippo_arm_seal.rows
                or self.raw_retrieval_seal.rows != self.raw_arm_seal.rows
                or self.hippo_retrieval_seal.retrieval_matrix_sha256
                != self.hippo_arm_seal.retrieval_matrix_sha256
                or self.raw_retrieval_seal.retrieval_matrix_sha256
                != self.raw_arm_seal.retrieval_matrix_sha256
            ):
                raise EraserEvidenceInferenceThreeArmSchedulerError(
                    "anchor runner seals drifted from scheduler arm seals"
                )
        elif (
            self.hippo_retrieval_seal is not None
            or self.raw_retrieval_seal is not None
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "pre-anchor block unexpectedly contains runner anchor seals"
            )
        expected_archive = _self_hashed(
            _block_archive_body(
                block=self.block,
                items=self.items,
                feature_seal=self.feature_seal,
                hippo_arm_seal=self.hippo_arm_seal,
                raw_arm_seal=self.raw_arm_seal,
                hippo_retrieval_seal=self.hippo_retrieval_seal,
                raw_retrieval_seal=self.raw_retrieval_seal,
                local_workers=self.local_pool_max_workers,
                hippo_workers=self.hippo_pool_max_workers,
                global_item_count=self.global_item_count,
                global_submitted_count=self.global_submitted_task_count,
            ),
            "archive_sha256",
        )
        if _parse_json_object(self.archive_json, "block archive") != expected_archive:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "block archive binding drifted"
            )
        expected_receipt = _self_hashed(
            {
                "schema": f"{VERSION}_block_receipt",
                "version": VERSION,
                "status": "complete_offline_three_arm_block",
                "block": self.block,
                "item_count": len(self.items),
                "logical_agent_raw_hipporag_task_count": 3 * len(self.items),
                "global_submitted_task_count_before_first_result": (
                    self.global_submitted_task_count
                ),
                "all_3n_tasks_submitted_before_first_result": True,
                "archive_sha256": expected_archive["archive_sha256"],
                "feature_receipt_sha256": (
                    self.feature_seal.feature_receipt_sha256
                ),
                "hipporag_arm_receipt_sha256": (
                    self.hippo_arm_seal.hipporag_arm_receipt_sha256
                ),
                "raw_arm_receipt_sha256": (
                    self.raw_arm_seal.raw_arm_receipt_sha256
                ),
                "anchor_hipporag_retrieval_matrix_sha256": (
                    None
                    if self.hippo_retrieval_seal is None
                    else self.hippo_retrieval_seal.retrieval_matrix_sha256
                ),
                "anchor_raw_retrieval_receipt_sha256": (
                    None
                    if self.raw_retrieval_seal is None
                    else self.raw_retrieval_seal.raw_retrieval_receipt_sha256
                ),
                "item_commitment_set_sha256": (
                    self.feature_seal.item_commitment_set_sha256
                ),
                "labels_evaluator_source_or_network_calls": 0,
            },
            "receipt_sha256",
        )
        if _parse_json_object(self.receipt_json, "block receipt") != expected_receipt:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "block receipt binding drifted"
            )

    @property
    def archive_payload(self) -> dict[str, Any]:
        return _parse_json_object(self.archive_json, "block archive")

    @property
    def receipt(self) -> dict[str, Any]:
        return _parse_json_object(self.receipt_json, "block receipt")


@dataclass(frozen=True)
class ThreeArmScheduleArtifact:
    blocks: tuple[BlockThreeArmArtifact, ...]
    total_item_count: int
    local_pool_max_workers: int
    hippo_pool_max_workers: int
    submitted_task_count: int
    receipt_json: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.blocks, tuple)
            or not self.blocks
            or any(not isinstance(row, BlockThreeArmArtifact) for row in self.blocks)
            or tuple(row.block for row in self.blocks)
            != tuple(block for block in runner.BLOCK_ORDER if block in self.by_block)
            or self.total_item_count
            != sum(len(row.items) for row in self.blocks)
            or self.local_pool_max_workers
            != min(LOCAL_WORKER_CAP, 2 * self.total_item_count)
            or self.hippo_pool_max_workers
            != min(HIPPORAG_WORKER_CAP, self.total_item_count)
            or self.submitted_task_count != 3 * self.total_item_count
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "schedule accounting drifted"
            )
        body = {
            "schema": f"{VERSION}_schedule_receipt",
            "version": VERSION,
            "status": "complete_offline_three_arm_schedule",
            "blocks": [row.block for row in self.blocks],
            "total_item_count": self.total_item_count,
            "logical_agent_raw_hipporag_task_count": self.submitted_task_count,
            "submitted_task_count_before_first_result": self.submitted_task_count,
            "all_3n_tasks_submitted_before_first_result": True,
            "local_agent_raw_pool_max_workers": self.local_pool_max_workers,
            "official_hipporag_pool_max_workers": self.hippo_pool_max_workers,
            "block_receipt_sha256s": [
                row.receipt["receipt_sha256"] for row in self.blocks
            ],
            "labels_evaluator_source_or_network_calls": 0,
            "full_square_pair_scan_performed": False,
        }
        expected = _self_hashed(body, "schedule_receipt_sha256")
        if _parse_json_object(self.receipt_json, "schedule receipt") != expected:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "schedule receipt binding drifted"
            )

    @property
    def by_block(self) -> Mapping[str, BlockThreeArmArtifact]:
        return {row.block: row for row in self.blocks}

    @property
    def receipt(self) -> dict[str, Any]:
        return _parse_json_object(self.receipt_json, "schedule receipt")


@dataclass(frozen=True)
class _SubmittedItem:
    block: str
    prepared: local_runtime.PreparedItemArtifact
    agent_future: Any
    raw_future: Any
    hippo_future: Any


def _normalize_inputs(
    items_by_block: Mapping[str, Sequence[local_runtime.ItemTextView]],
) -> dict[str, tuple[local_runtime.ItemTextView, ...]]:
    if not isinstance(items_by_block, Mapping) or not items_by_block:
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "scheduler item registry is empty or invalid"
        )
    if any(block not in BLOCK_ORDER for block in items_by_block):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "scheduler block is outside the frozen four-block registry"
        )
    result: dict[str, tuple[local_runtime.ItemTextView, ...]] = {}
    commitments: list[str] = []
    for block in runner.BLOCK_ORDER:
        if block not in items_by_block:
            continue
        raw_rows = items_by_block[block]
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduler block rows are not a sequence"
            )
        rows = tuple(raw_rows)
        if (
            len(rows) != runner.BLOCK_COUNTS[block]
            or any(not isinstance(row, local_runtime.ItemTextView) for row in rows)
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "scheduler block item count or view type drifted"
            )
        result[block] = rows
        commitments.extend(row.item_commitment_sha256 for row in rows)
    if len(commitments) != len(set(commitments)):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "scheduler item commitment is duplicated"
        )
    return result


def _cancel_submitted(rows: Sequence[_SubmittedItem]) -> None:
    for row in rows:
        for future in (row.agent_future, row.raw_future, row.hippo_future):
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()


def run_three_arm_schedule(
    *,
    items_by_block: Mapping[str, Sequence[local_runtime.ItemTextView]],
    runtime_bundle: object,
    local_executor_factory: Callable[..., Any] = ThreadPoolExecutor,
    hippo_executor_factory: Callable[..., Any] = ThreadPoolExecutor,
) -> ThreeArmScheduleArtifact:
    """Prepare once, submit all ``3 * n`` tasks, then join and seal."""

    normalized = _normalize_inputs(items_by_block)
    prepare = getattr(runtime_bundle, "prepare", None)
    hippo_gateway = getattr(runtime_bundle, "hippo", None)
    prepare_blocks = getattr(hippo_gateway, "prepare_blocks", None)
    retrieve_artifact = getattr(hippo_gateway, "retrieve_artifact", None)
    if (
        not callable(prepare)
        or not callable(prepare_blocks)
        or not callable(retrieve_artifact)
        or not callable(local_executor_factory)
        or not callable(hippo_executor_factory)
    ):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "runtime bundle or executor factory is incomplete"
        )
    try:
        prepared_batch = prepare(normalized)
    except Exception as exc:
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "semantic preparation failed before scheduler entry"
        ) from exc
    if not isinstance(prepared_batch, local_runtime.PreparedBatchArtifact):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "runtime returned no typed prepared batch"
        )
    expected_views = {
        (block, view.item_commitment_sha256): view
        for block, rows in normalized.items()
        for view in rows
    }
    if (
        len(prepared_batch.items) != len(expected_views)
        or any(
            expected_views.get((item.block, item.item_commitment_sha256))
            != item.view
            for item in prepared_batch.items
        )
    ):
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "prepared batch differs from scheduler input"
        )
    blocks = tuple(block for block in runner.BLOCK_ORDER if block in normalized)
    try:
        prepare_blocks(blocks)
    except Exception as exc:
        raise EraserEvidenceInferenceThreeArmSchedulerError(
            "HippoRAG block-parent preparation failed"
        ) from exc

    item_count = len(prepared_batch.items)
    local_workers = min(LOCAL_WORKER_CAP, 2 * item_count)
    hippo_workers = min(HIPPORAG_WORKER_CAP, item_count)
    submitted: list[_SubmittedItem] = []
    results: list[tuple[str, local_runtime.PreparedItemArtifact, Any, Any, Any]] = []
    try:
        with ExitStack() as stack:
            local_pool = stack.enter_context(
                local_executor_factory(max_workers=local_workers)
            )
            hippo_pool = stack.enter_context(
                hippo_executor_factory(max_workers=hippo_workers)
            )
            # Submission loop contains no result read, join, feature build, or
            # action execution performed by this thread.
            for prepared in prepared_batch.items:
                submitted.append(
                    _SubmittedItem(
                        block=prepared.block,
                        prepared=prepared,
                        agent_future=local_pool.submit(
                            local_runtime.execute_agent, prepared
                        ),
                        raw_future=local_pool.submit(
                            local_runtime.execute_raw, prepared
                        ),
                        hippo_future=hippo_pool.submit(
                            retrieve_artifact,
                            block=prepared.block,
                            view=prepared.view,
                        ),
                    )
                )
            submitted_count = 3 * len(submitted)
            if submitted_count != 3 * item_count:
                raise EraserEvidenceInferenceThreeArmSchedulerError(
                    "three-arm submitted task count drifted"
                )
            # This is the first location at which Future.result is called.
            for row in submitted:
                results.append(
                    (
                        row.block,
                        row.prepared,
                        row.agent_future.result(),
                        row.raw_future.result(),
                        row.hippo_future.result(),
                    )
                )
    except BaseException:
        _cancel_submitted(submitted)
        raise

    by_block: dict[str, list[ScheduledItemArtifact]] = {
        block: [] for block in blocks
    }
    for block, prepared, agent, raw, hippo in results:
        if (
            not isinstance(agent, local_runtime.AgentExecutionArtifact)
            or not isinstance(raw, local_runtime.RawExecutionArtifact)
            or not isinstance(hippo, local_runtime.HippoExecutionArtifact)
        ):
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "three-arm future returned a foreign artifact"
            )
        try:
            pair_receipt = feature_bridge.build_selected_pair_semantic_receipt(
                graph=prepared.graph,
                semantic_tensor=prepared.semantic_tensor,
                r0_top5=agent.r0_action.output_top5,
                r7_top5=agent.r7_action.output_top5,
                pair_rows=agent.pair_rows,
            )
            exact_build = feature_bridge.build_exact_difference_trace(
                item_commitment_sha256=prepared.item_commitment_sha256,
                graph=prepared.graph,
                semantic_tensor=prepared.semantic_tensor,
                r0_action=agent.r0_action,
                r7_action=agent.r7_action,
                selected_pair_semantic_receipt=pair_receipt,
            )
            by_block[block].append(
                ScheduledItemArtifact(
                    block=block,
                    agent=agent,
                    raw=raw,
                    hippo=hippo,
                    selected_pair_receipt=pair_receipt,
                    exact_feature_build=exact_build,
                )
            )
        except (
            feature_bridge.EraserExactFeatureBridgeError,
            runner.EraserEvidenceInferenceRunnerError,
            local_runtime.EraserEvidenceInferenceLocalRuntimeError,
        ) as exc:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "three-arm exact feature binding failed"
            ) from exc

    block_artifacts: list[BlockThreeArmArtifact] = []
    for block in blocks:
        items = tuple(
            sorted(
                by_block[block],
                key=lambda item: item.item_commitment_sha256,
            )
        )
        try:
            feature_seal = runner.seal_feature_matrix(
                block=block,
                traces=tuple(item.difference_trace for item in items),
            )
            hippo_rows = tuple(
                runner.HippoRetrieval(
                    item_commitment_sha256=item.item_commitment_sha256,
                    sentence_count=item.difference_trace.sentence_count,
                    top5=item.hippo.top5,
                )
                for item in items
            )
            raw_rows = tuple(
                runner.RawRetrieval(
                    item_commitment_sha256=item.item_commitment_sha256,
                    sentence_count=item.difference_trace.sentence_count,
                    top5=item.raw.top5,
                    action_trace_sha256=item.raw.r0_action.trace_sha256,
                )
                for item in items
            )
            hippo_arm_seal = seal_hippo_arm(block=block, rows=hippo_rows)
            raw_arm_seal = seal_raw_arm(block=block, rows=raw_rows)
            hippo_seal = (
                runner.seal_hippo_retrievals(block=block, rows=hippo_rows)
                if block in ANCHOR_BLOCKS
                else None
            )
            raw_seal = (
                runner.seal_raw_retrievals(block=block, rows=raw_rows)
                if block in ANCHOR_BLOCKS
                else None
            )
        except runner.EraserEvidenceInferenceRunnerError as exc:
            raise EraserEvidenceInferenceThreeArmSchedulerError(
                "block feature or retrieval seal failed"
            ) from exc
        archive = _self_hashed(
            _block_archive_body(
                block=block,
                items=items,
                feature_seal=feature_seal,
                hippo_arm_seal=hippo_arm_seal,
                raw_arm_seal=raw_arm_seal,
                hippo_retrieval_seal=hippo_seal,
                raw_retrieval_seal=raw_seal,
                local_workers=local_workers,
                hippo_workers=hippo_workers,
                global_item_count=item_count,
                global_submitted_count=submitted_count,
            ),
            "archive_sha256",
        )
        receipt = _self_hashed(
            {
                "schema": f"{VERSION}_block_receipt",
                "version": VERSION,
                "status": "complete_offline_three_arm_block",
                "block": block,
                "item_count": len(items),
                "logical_agent_raw_hipporag_task_count": 3 * len(items),
                "global_submitted_task_count_before_first_result": submitted_count,
                "all_3n_tasks_submitted_before_first_result": True,
                "archive_sha256": archive["archive_sha256"],
                "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
                "hipporag_arm_receipt_sha256": (
                    hippo_arm_seal.hipporag_arm_receipt_sha256
                ),
                "raw_arm_receipt_sha256": raw_arm_seal.raw_arm_receipt_sha256,
                "anchor_hipporag_retrieval_matrix_sha256": (
                    None
                    if hippo_seal is None
                    else hippo_seal.retrieval_matrix_sha256
                ),
                "anchor_raw_retrieval_receipt_sha256": (
                    None
                    if raw_seal is None
                    else raw_seal.raw_retrieval_receipt_sha256
                ),
                "item_commitment_set_sha256": (
                    feature_seal.item_commitment_set_sha256
                ),
                "labels_evaluator_source_or_network_calls": 0,
            },
            "receipt_sha256",
        )
        block_artifacts.append(
            BlockThreeArmArtifact(
                block=block,
                items=items,
                feature_seal=feature_seal,
                hippo_arm_seal=hippo_arm_seal,
                raw_arm_seal=raw_arm_seal,
                hippo_retrieval_seal=hippo_seal,
                raw_retrieval_seal=raw_seal,
                local_pool_max_workers=local_workers,
                hippo_pool_max_workers=hippo_workers,
                global_item_count=item_count,
                global_submitted_task_count=submitted_count,
                archive_json=_canonical_json_text(archive),
                receipt_json=_canonical_json_text(receipt),
            )
        )
    schedule_body = {
        "schema": f"{VERSION}_schedule_receipt",
        "version": VERSION,
        "status": "complete_offline_three_arm_schedule",
        "blocks": [row.block for row in block_artifacts],
        "total_item_count": item_count,
        "logical_agent_raw_hipporag_task_count": submitted_count,
        "submitted_task_count_before_first_result": submitted_count,
        "all_3n_tasks_submitted_before_first_result": True,
        "local_agent_raw_pool_max_workers": local_workers,
        "official_hipporag_pool_max_workers": hippo_workers,
        "block_receipt_sha256s": [
            row.receipt["receipt_sha256"] for row in block_artifacts
        ],
        "labels_evaluator_source_or_network_calls": 0,
        "full_square_pair_scan_performed": False,
    }
    return ThreeArmScheduleArtifact(
        blocks=tuple(block_artifacts),
        total_item_count=item_count,
        local_pool_max_workers=local_workers,
        hippo_pool_max_workers=hippo_workers,
        submitted_task_count=submitted_count,
        receipt_json=_canonical_json_text(
            _self_hashed(schedule_body, "schedule_receipt_sha256")
        ),
    )


__all__ = [
    "ANCHOR_BLOCKS",
    "BLOCK_ORDER",
    "BlockThreeArmArtifact",
    "EraserEvidenceInferenceThreeArmSchedulerError",
    "HIPPORAG_WORKER_CAP",
    "HippoArmSeal",
    "LOCAL_WORKER_CAP",
    "RawArmSeal",
    "ScheduledItemArtifact",
    "ThreeArmScheduleArtifact",
    "VERSION",
    "run_three_arm_schedule",
    "seal_hippo_arm",
    "seal_raw_arm",
    "stable_hash",
]

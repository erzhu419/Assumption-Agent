"""Capability-ordered one-shot controller for the HybridQA P6/E2 study.

The production lifecycle consumes only the packs created by
``hybridqa_direct_acquisition_v1``.  It persists a complete label-free action
archive before opening each corresponding label pack, freezes both F_search
policies before scoring A_hold, and does not invoke the M_search pack loader
before a typed A_hold score seal proves promotion.  Production execution uses
only the frozen local MiniLM and official HippoRAG runtimes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Protocol

from assumption_agent.benchmarks import hybridqa_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks import hybridqa_local_runtime_v1 as local_runtime
from assumption_agent.benchmarks import hybridqa_query_anchored_formal_runner_v1 as runner
from assumption_agent.benchmarks import hybridqa_query_anchored_operator_v1 as operator
from replication_runtime.multihoprag_minilm_v1 import adapter as minilm_adapter


VERSION = "hybridqa_p6_e2_formal_controller_v1"
CONTROLLER_ROOT_RELATIVE = acquisition.FORMAL_ROOT_RELATIVE / "controller"
MARKER_FILENAME = "lifecycle.one_shot_marker.json"
FAILURE_FILENAME = "lifecycle.terminal_failure.json"
RESULT_FILENAME = "lifecycle.terminal_result.json"
BLOCK_ORDER = acquisition.BLOCK_ORDER
BLOCK_COUNTS = acquisition.BLOCK_COUNTS
FAMILIES = acquisition.FAMILIES
LOCAL_WORKER_CAP = 32

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class HybridQaFormalControllerError(RuntimeError):
    """A pack, capability order, runtime, archive or one-shot rule drifted."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HybridQaFormalControllerError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise HybridQaFormalControllerError(f"{field} is not a lowercase SHA-256")
    return value


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HybridQaFormalControllerError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    if not isinstance(payload, Mapping):
        raise HybridQaFormalControllerError("self-hashed payload is not an object")
    body = dict(payload)
    declared = _require_sha256(body.pop(field, None), field)
    if stable_hash(body) != declared:
        raise HybridQaFormalControllerError(f"{field} self-hash drifted")
    return declared


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assert_private_regular(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HybridQaFormalControllerError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HybridQaFormalControllerError(f"{field} is unsafe")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise HybridQaFormalControllerError(f"{field} is not private")


def _assert_private_directory(
    path: Path, field: str, *, expected_mode: int | None = None
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HybridQaFormalControllerError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HybridQaFormalControllerError(f"{field} is unsafe")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise HybridQaFormalControllerError(f"{field} is not private")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise HybridQaFormalControllerError(f"{field} mode drifted")


def _read_private_bytes(
    path: Path | str, *, field: str, dir_fd: int | None = None
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise HybridQaFormalControllerError(f"{field} is unsafe")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            return handle.read()
    except HybridQaFormalControllerError:
        raise
    except OSError as exc:
        raise HybridQaFormalControllerError(f"{field} is unreadable") from exc


def _load_canonical_object(
    path: Path | str,
    *,
    field: str,
    expected_file_sha256: str | None = None,
    private: bool = True,
    dir_fd: int | None = None,
) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise HybridQaFormalControllerError(f"{field} is unsafe")
        if private and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise HybridQaFormalControllerError(f"{field} is not private")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            raw = handle.read()
        value = json.loads(raw.decode("ascii"))
    except HybridQaFormalControllerError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridQaFormalControllerError(f"{field} is invalid") from exc
    digest = _sha256_bytes(raw)
    if expected_file_sha256 is not None and digest != _require_sha256(
        expected_file_sha256, f"{field} expected file hash"
    ):
        raise HybridQaFormalControllerError(f"{field} file hash drifted")
    if not isinstance(value, dict) or raw != _canonical_bytes(value, newline=True):
        raise HybridQaFormalControllerError(f"{field} is not canonical JSON")
    return value, digest


def _write_exclusive(path: Path, payload: Mapping[str, Any], *, mode: int) -> str:
    if mode not in {0o600, 0o644}:
        raise HybridQaFormalControllerError("artifact mode is invalid")
    raw = _canonical_bytes(payload, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise HybridQaFormalControllerError("exclusive artifact creation failed") from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        parent_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return _sha256_bytes(raw)


def _canonical_project(project_root: str | Path) -> Path:
    try:
        project = Path(project_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HybridQaFormalControllerError("project root is unavailable") from exc
    if project.is_symlink() or not project.is_dir():
        raise HybridQaFormalControllerError("project root is unsafe")
    return project


@dataclass(frozen=True)
class CorpusPack:
    articles: tuple[dict[str, object], ...]
    minilm_articles: tuple[minilm_adapter.ArticleText, ...]
    graph: operator.TypedCorpusGraph
    pack_sha256: str
    duplicate_text_group_count: int
    duplicate_text_unit_count: int
    acquisition_receipt_sha256: str


@dataclass(frozen=True)
class ViewItem:
    item_commitment_sha256: str
    question: str
    question_postag: str


@dataclass(frozen=True)
class BlockView:
    block: str
    items: tuple[ViewItem, ...]
    view_sha256: str


@dataclass(frozen=True)
class LabelRow:
    item_commitment_sha256: str
    family: str
    gold_indices: tuple[int, ...]


@dataclass(frozen=True)
class LabelPack:
    block: str
    by_commitment: Mapping[str, LabelRow]
    label_pack_sha256: str


@dataclass(frozen=True)
class ArchiveCapability:
    """A durable archive seal that must exist before the label capability."""

    block: str
    receipt_json: str

    def __post_init__(self) -> None:
        try:
            receipt = json.loads(self.receipt_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise HybridQaFormalControllerError("archive capability is invalid") from exc
        if (
            not isinstance(receipt, dict)
            or _canonical_bytes(receipt).decode("ascii") != self.receipt_json
            or set(receipt)
            != {
                "schema",
                "version",
                "block",
                "acquisition_receipt_sha256",
                "corpus_pack_sha256",
                "block_view_sha256",
                "graph_sha256",
                "item_commitment_set_sha256",
                "execution_receipt_sha256",
                "archive_file_sha256",
                "archive_sha256",
                "feature_receipt_sha256",
                "label_pack_opened_before_seal",
                "archive_seal_sha256",
            }
            or receipt.get("block") != self.block
            or receipt.get("schema")
            != f"{VERSION}_label_free_archive_seal"
            or receipt.get("version") != VERSION
            or receipt.get("label_pack_opened_before_seal") is not False
        ):
            raise HybridQaFormalControllerError("archive capability schema drifted")
        verify_self_hash(receipt, "archive_seal_sha256")

    @property
    def receipt(self) -> dict[str, Any]:
        value = json.loads(self.receipt_json)
        assert isinstance(value, dict)
        return value

    def verify_durable(
        self,
        controller_root: Path,
        *,
        acquisition_receipt_sha256: str,
        corpus_pack_sha256: str,
        graph_sha256: str,
        block_view_sha256: str,
        feature_seal: runner.FeatureSeal,
    ) -> None:
        if (
            not isinstance(feature_seal, runner.FeatureSeal)
            or feature_seal.block != self.block
        ):
            raise HybridQaFormalControllerError(
                "archive feature capability drifted"
            )
        expected_binding = {
            "acquisition_receipt_sha256": acquisition_receipt_sha256,
            "corpus_pack_sha256": corpus_pack_sha256,
            "graph_sha256": graph_sha256,
            "block_view_sha256": block_view_sha256,
            "item_commitment_set_sha256": (
                feature_seal.item_commitment_set_sha256
            ),
            "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
        }
        if any(
            self.receipt.get(field) != _require_sha256(value, field)
            for field, value in expected_binding.items()
        ):
            raise HybridQaFormalControllerError("archive seal binding drifted")
        observed, _ = _load_canonical_object(
            controller_root / f"{self.block}.archive_seal.json",
            field=f"{self.block} archive seal",
            private=True,
        )
        verify_self_hash(observed, "archive_seal_sha256")
        if observed != self.receipt:
            raise HybridQaFormalControllerError("archive capability is not durable")
        archive, _ = _load_canonical_object(
            controller_root / f"{self.block}.label_free_archive.json",
            field=f"{self.block} archive",
            expected_file_sha256=str(observed.get("archive_file_sha256")),
            private=True,
        )
        verify_self_hash(archive, "archive_sha256")
        expected_archive_keys = {
            "schema",
            "version",
            "block",
            "item_count",
            *expected_binding,
            "execution_receipt",
            "feature_receipt",
            "hipporag_top5",
            "items",
            "complete_action_and_feature_traces_persisted",
            "raw_query_corpus_label_family_gold_or_utility_persisted",
            "online_evaluator_calls",
            "archive_sha256",
        }
        if (
            set(archive) != expected_archive_keys
            or archive.get("schema") != f"{VERSION}_label_free_archive"
            or archive.get("version") != VERSION
            or archive.get("block") != self.block
            or archive.get("item_count") != BLOCK_COUNTS[self.block]
            or archive.get("archive_sha256") != observed.get("archive_sha256")
            or archive.get("complete_action_and_feature_traces_persisted")
            is not True
            or archive.get(
                "raw_query_corpus_label_family_gold_or_utility_persisted"
            )
            is not False
            or archive.get("online_evaluator_calls") != 0
            or any(
                archive.get(field) != value
                for field, value in expected_binding.items()
            )
        ):
            raise HybridQaFormalControllerError("archive capability content drifted")
        execution = archive.get("execution_receipt")
        expected_execution_keys = {
            "schema",
            "version",
            "block",
            "item_count",
            "acquisition_receipt_sha256",
            "corpus_pack_sha256",
            "block_view_sha256",
            "item_commitment_set_sha256",
            "feature_receipt_sha256",
            "graph_sha256",
            "minilm_index_sha256",
            "hipporag_build_receipt_sha256",
            "combined_hipporag_retrieval_receipt_sha256",
            "combined_query_schedule_sha256",
            "projected_hipporag_top5_sha256",
            "local_worker_cap",
            "all_item_matrices_eagerly_submitted_before_join",
            "local_and_official_jobs_submitted_before_join",
            "labels_family_gold_or_utility_accessed",
            "raw_query_or_corpus_text_persisted",
            "online_evaluator_calls",
            "execution_receipt_sha256",
        }
        if not isinstance(execution, Mapping) or set(execution) != expected_execution_keys:
            raise HybridQaFormalControllerError("archive execution receipt drifted")
        verify_self_hash(execution, "execution_receipt_sha256")
        if (
            execution.get("schema")
            != f"{VERSION}_label_free_execution_receipt"
            or execution.get("version") != VERSION
            or execution.get("block") != self.block
            or execution.get("item_count") != BLOCK_COUNTS[self.block]
            or execution.get("local_worker_cap") != LOCAL_WORKER_CAP
            or execution.get(
                "all_item_matrices_eagerly_submitted_before_join"
            )
            is not True
            or execution.get("local_and_official_jobs_submitted_before_join")
            is not True
            or execution.get("labels_family_gold_or_utility_accessed") is not False
            or execution.get("raw_query_or_corpus_text_persisted") is not False
            or execution.get("online_evaluator_calls") != 0
            or execution.get("execution_receipt_sha256")
            != observed.get("execution_receipt_sha256")
            or any(
                execution.get(field) != value
                for field, value in expected_binding.items()
                if field != "feature_receipt_sha256"
            )
            or execution.get("feature_receipt_sha256")
            != feature_seal.feature_receipt_sha256
            or any(
                not _require_sha256(execution.get(field), field)
                for field in (
                    "minilm_index_sha256",
                    "hipporag_build_receipt_sha256",
                    "combined_hipporag_retrieval_receipt_sha256",
                    "combined_query_schedule_sha256",
                    "projected_hipporag_top5_sha256",
                )
            )
        ):
            raise HybridQaFormalControllerError(
                "archive execution semantics drifted"
            )
        if archive.get("feature_receipt") != feature_seal.receipt:
            raise HybridQaFormalControllerError("archive feature receipt drifted")

        commitments = tuple(feature_seal.item_commitments)
        hippo_rows = archive.get("hipporag_top5")
        if (
            not isinstance(hippo_rows, list)
            or len(hippo_rows) != BLOCK_COUNTS[self.block]
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not isinstance(row[1], list)
                or len(row[1]) != 5
                or len(set(row[1])) != 5
                or any(type(index) is not int or not 0 <= index < 609 for index in row[1])
                for row in hippo_rows
            )
            or tuple(row[0] for row in hippo_rows) != commitments
            or execution.get("projected_hipporag_top5_sha256")
            != stable_hash(hippo_rows)
        ):
            raise HybridQaFormalControllerError("archive HippoRAG matrix drifted")
        item_rows = archive.get("items")
        if (
            not isinstance(item_rows, list)
            or len(item_rows) != BLOCK_COUNTS[self.block]
            or any(
                not isinstance(row, Mapping)
                or set(row)
                != {"item_commitment_sha256", "actions", "feature_traces"}
                for row in item_rows
            )
            or tuple(row["item_commitment_sha256"] for row in item_rows)
            != commitments
        ):
            raise HybridQaFormalControllerError("archive item matrix drifted")
        traces_by_item = {
            commitment: [
                trace.payload()
                for trace in feature_seal.traces
                if trace.item_commitment_sha256 == commitment
            ]
            for commitment in commitments
        }
        action_keys = {
            "recipe_id",
            "output_top5",
            "retained_raw_top3",
            "selection_steps",
            "raw_dense_order_sha256",
            "graph_sha256",
            "query_sha256",
            "semantic_tensor_sha256",
            "reachability_sha256",
            "candidate_scan_sha256",
            "candidate_universe_size",
            "candidate_score_evaluations",
            "semantic_cell_scan_count",
            "hipporag_candidate_or_feature_count",
            "trace_sha256",
        }
        for row in item_rows:
            commitment = row["item_commitment_sha256"]
            actions = row["actions"]
            feature_traces = row["feature_traces"]
            if (
                feature_traces != traces_by_item[commitment]
                or not isinstance(actions, list)
                or len(actions) != len(runner.RECIPE_IDS)
                or any(
                    not isinstance(action, Mapping)
                    or set(action) != action_keys
                    for action in actions
                )
                or tuple(action["recipe_id"] for action in actions)
                != runner.RECIPE_IDS
                or any(
                    action["graph_sha256"] != graph_sha256
                    or action["trace_sha256"]
                    != feature_trace["behavior_sha256"]
                    or not isinstance(action["output_top5"], list)
                    or len(action["output_top5"]) != 5
                    or len(set(action["output_top5"])) != 5
                    for action, feature_trace in zip(
                        actions, feature_traces, strict=True
                    )
                )
            ):
                raise HybridQaFormalControllerError(
                    "archive action/feature matrix drifted"
                )


def _verify_source_qualification_receipt(value: object) -> None:
    source = acquisition.source_qualification
    if not isinstance(value, Mapping):
        raise HybridQaFormalControllerError(
            "embedded source qualification receipt is absent"
        )
    try:
        code_path = Path(source.__file__).resolve(strict=True)
        expected_code_sha256 = _sha256_bytes(code_path.read_bytes())
        source.verify_qualification_receipt(
            value,
            expected_qualification_code_sha256=expected_code_sha256,
        )
    except (
        OSError,
        RuntimeError,
        source.HybridQaSourceQualificationError,
    ) as exc:
        raise HybridQaFormalControllerError(
            "embedded source qualification contract drifted"
        ) from exc


def _validate_gold_topology(
    *, family: str, gold_indices: Sequence[int], graph: operator.TypedCorpusGraph
) -> None:
    try:
        units = tuple(graph.units[index] for index in gold_indices)
    except (IndexError, TypeError) as exc:
        raise HybridQaFormalControllerError("gold topology index drifted") from exc
    tables = {unit.table_key for unit in units}
    rows = tuple(unit for unit in units if unit.unit_type == "table_row")
    passages = tuple(
        unit for unit in units if unit.unit_type == "linked_passage"
    )
    if len(tables) != 1:
        raise HybridQaFormalControllerError("gold topology crosses tables")
    if family == "TABLE_ONLY":
        valid = len(units) == 1 and len(rows) == 1 and not passages
    elif family == "PASSAGE_ONLY":
        valid = len(units) == 2 and len(rows) == 1 and len(passages) == 1
    elif family == "DUAL_TABLE_PASSAGE":
        valid = (
            len(units) in {2, 3}
            and len(rows) in {1, 2}
            and len(passages) == 1
        )
    else:
        valid = False
    if not valid:
        raise HybridQaFormalControllerError("gold family topology drifted")
    if passages:
        target = passages[0].link_target_keys[0]
        if not any(target in row.link_target_keys for row in rows):
            raise HybridQaFormalControllerError(
                "gold row/passage link topology drifted"
            )


class AcquisitionBoundary:
    """The only production object allowed to open acquisition private packs."""

    def __init__(self, project: Path, *, expected_freeze_sha256: str) -> None:
        self.project = _canonical_project(project)
        self.root = self.project / acquisition.ACQUISITION_RELATIVE
        expected_freeze_sha256 = _require_sha256(
            expected_freeze_sha256, "expected implementation freeze"
        )
        _assert_private_directory(
            self.project / acquisition.FORMAL_ROOT_RELATIVE,
            "formal attempt root",
            expected_mode=0o700,
        )
        _assert_private_directory(
            self.root, "acquisition root", expected_mode=0o500
        )
        if os.path.lexists(self.root / acquisition.FAILURE_FILENAME):
            raise HybridQaFormalControllerError(
                "acquisition terminal failure coexists with completion"
            )
        public, _file_sha = _load_canonical_object(
            self.root / acquisition.PUBLIC_FILENAME,
            field="acquisition public receipt",
            private=True,
        )
        verify_self_hash(public, "acquisition_receipt_sha256")
        expected_public_keys = {
            "schema",
            "version",
            "status",
            "design_sha256",
            "implementation_freeze_sha256",
            "source_qualification_receipt",
            "selection_secret_commitment_sha256",
            "selection_secret_persisted_publicly",
            "candidate_counts_by_family",
            "typed_exclusion_counts",
            "block_counts",
            "per_family_quota",
            "selected_question_count",
            "selected_table_count",
            "question_and_table_disjoint",
            "corpus_unit_count",
            "corpus_unit_type_counts",
            "private_pack_file_sha256s",
            "F_search_label_pack_created",
            "raw_question_answer_table_or_unit_identity_persisted_publicly",
            "online_evaluator_calls",
            "retry_replay_or_resample",
            "acquisition_receipt_sha256",
        }
        candidate_counts = public.get("candidate_counts_by_family")
        exclusions = public.get("typed_exclusion_counts")
        unit_counts = public.get("corpus_unit_type_counts")
        secret_commitment = public.get("selection_secret_commitment_sha256")
        if (
            set(public) != expected_public_keys
            or public.get("schema") != f"{acquisition.VERSION}_public_receipt"
            or public.get("version") != acquisition.VERSION
            or public.get("status") != "formal_acquisition_complete"
            or public.get("design_sha256") != acquisition.DESIGN_SHA256
            or public.get("implementation_freeze_sha256")
            != expected_freeze_sha256
            or public.get("selection_secret_persisted_publicly") is not False
            or not isinstance(secret_commitment, str)
            or _HEX64.fullmatch(secret_commitment) is None
            or public.get("block_counts") != dict(BLOCK_COUNTS)
            or public.get("per_family_quota")
            != dict(acquisition.PER_FAMILY_QUOTA)
            or public.get("question_and_table_disjoint") is not True
            or public.get("F_search_label_pack_created") is not False
            or public.get("selected_question_count") != sum(BLOCK_COUNTS.values())
            or public.get("selected_table_count") != sum(BLOCK_COUNTS.values())
            or public.get("corpus_unit_count") != acquisition.CORPUS_UNIT_COUNT
            or public.get(
                "raw_question_answer_table_or_unit_identity_persisted_publicly"
            )
            is not False
            or public.get("online_evaluator_calls") != 0
            or public.get("retry_replay_or_resample") != 0
            or not isinstance(candidate_counts, Mapping)
            or set(candidate_counts) != set(FAMILIES)
            or any(
                type(count) is not int or count <= 0
                for count in candidate_counts.values()
            )
            or not isinstance(exclusions, Mapping)
            or any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count < 0
                for reason, count in exclusions.items()
            )
            or not isinstance(unit_counts, Mapping)
            or set(unit_counts) != {"table_row", "linked_passage"}
            or any(
                type(count) is not int or count <= 0
                for count in unit_counts.values()
            )
            or sum(unit_counts.values()) != acquisition.CORPUS_UNIT_COUNT
        ):
            raise HybridQaFormalControllerError("acquisition public contract drifted")
        _verify_source_qualification_receipt(
            public.get("source_qualification_receipt")
        )
        marker, _ = _load_canonical_object(
            self.root / acquisition.MARKER_FILENAME,
            field="acquisition one-shot marker",
            private=True,
        )
        verify_self_hash(marker, "marker_sha256")
        if (
            set(marker)
            != {
                "schema",
                "version",
                "status",
                "design_sha256",
                "implementation_freeze_sha256",
                "source_validation_completed",
                "selection_secret_created",
                "marker_sha256",
            }
            or marker.get("schema")
            != f"{acquisition.VERSION}_one_shot_marker"
            or marker.get("version") != acquisition.VERSION
            or marker.get("status") != "formal_attempt_started"
            or marker.get("design_sha256") != acquisition.DESIGN_SHA256
            or marker.get("implementation_freeze_sha256")
            != expected_freeze_sha256
            or marker.get("source_validation_completed") is not False
            or marker.get("selection_secret_created") is not False
        ):
            raise HybridQaFormalControllerError("acquisition marker drifted")
        hashes = public.get("private_pack_file_sha256s")
        if not isinstance(hashes, Mapping):
            raise HybridQaFormalControllerError("private pack hash map is absent")
        expected_names = {
            acquisition.CORPUS_FILENAME,
            *(f"{block}.view.private.json" for block in BLOCK_ORDER),
            *(
                f"{block}.labels.sealed.json"
                for block in ("A_form", "A_hold", "M_search")
            ),
        }
        if set(hashes) != expected_names or any(
            not isinstance(value, str) or _HEX64.fullmatch(value) is None
            for value in hashes.values()
        ):
            raise HybridQaFormalControllerError("private pack file set drifted")
        if os.path.lexists(self.root / "F_search.labels.sealed.json"):
            raise HybridQaFormalControllerError("F_search label pack exists")
        expected_root_names = {
            acquisition.MARKER_FILENAME,
            acquisition.SECRET_FILENAME,
            acquisition.PUBLIC_FILENAME,
            *expected_names,
        }
        try:
            root_names = {entry.name for entry in os.scandir(self.root)}
        except OSError as exc:
            raise HybridQaFormalControllerError(
                "acquisition root cannot be enumerated"
            ) from exc
        if root_names != expected_root_names:
            raise HybridQaFormalControllerError("acquisition root file set drifted")
        _assert_private_regular(
            self.root / acquisition.SECRET_FILENAME,
            "selection/fold secret",
        )
        self.public = public
        self.file_hashes = dict(hashes)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._root_fd = os.open(self.root, flags)
            root_metadata = os.fstat(self._root_fd)
        except OSError as exc:
            raise HybridQaFormalControllerError(
                "sealed acquisition root cannot be held"
            ) from exc
        self._root_identity = (root_metadata.st_dev, root_metadata.st_ino)
        self._expected_root_names = frozenset(expected_root_names)
        self._assert_root_intact()
        self._corpus_binding: tuple[str, str] | None = None
        self._view_bindings: dict[str, tuple[str, tuple[str, ...]]] = {}

    def _assert_root_intact(self) -> None:
        try:
            held = os.fstat(self._root_fd)
            path_metadata = self.root.lstat()
            names = set(os.listdir(self._root_fd))
        except OSError as exc:
            raise HybridQaFormalControllerError(
                "sealed acquisition root changed"
            ) from exc
        if (
            (held.st_dev, held.st_ino) != self._root_identity
            or (path_metadata.st_dev, path_metadata.st_ino) != self._root_identity
            or not stat.S_ISDIR(held.st_mode)
            or stat.S_IMODE(held.st_mode) != 0o500
            or names != set(self._expected_root_names)
            or acquisition.FAILURE_FILENAME in names
            or "F_search.labels.sealed.json" in names
        ):
            raise HybridQaFormalControllerError(
                "sealed acquisition root identity drifted"
            )

    def close(self) -> None:
        descriptor = getattr(self, "_root_fd", None)
        if isinstance(descriptor, int) and descriptor >= 0:
            try:
                os.close(descriptor)
            finally:
                self._root_fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    @property
    def acquisition_receipt_sha256(self) -> str:
        return _require_sha256(
            self.public.get("acquisition_receipt_sha256"),
            "acquisition receipt",
        )

    def _load_pack(self, filename: str, *, self_hash_field: str) -> dict[str, Any]:
        self._assert_root_intact()
        value, _digest = _load_canonical_object(
            filename,
            field=filename,
            expected_file_sha256=self.file_hashes[filename],
            private=True,
            dir_fd=self._root_fd,
        )
        verify_self_hash(value, self_hash_field)
        return value

    def load_corpus(self) -> CorpusPack:
        pack = self._load_pack(
            acquisition.CORPUS_FILENAME, self_hash_field="corpus_pack_sha256"
        )
        expected_keys = {
            "schema",
            "version",
            "unit_count",
            "shared_arm_document_serialization",
            "duplicate_text_group_count",
            "duplicate_text_unit_count",
            "duplicate_expansion_delegated_to_frozen_official_HippoRAG_adapter",
            "units",
            "corpus_pack_sha256",
        }
        units = pack.get("units")
        if (
            set(pack) != expected_keys
            or pack.get("schema") != f"{acquisition.VERSION}_corpus_pack"
            or pack.get("version") != acquisition.VERSION
            or pack.get("unit_count") != acquisition.CORPUS_UNIT_COUNT
            or pack.get("shared_arm_document_serialization")
            != "title_plus_two_LF_plus_body"
            or pack.get(
                "duplicate_expansion_delegated_to_frozen_official_HippoRAG_adapter"
            )
            is not True
            or not isinstance(units, list)
            or len(units) != acquisition.CORPUS_UNIT_COUNT
        ):
            raise HybridQaFormalControllerError("corpus pack envelope drifted")
        articles: list[dict[str, object]] = []
        minilm_articles: list[minilm_adapter.ArticleText] = []
        atomic_units: list[operator.AtomicUnit] = []
        documents: list[str] = []
        for position, raw in enumerate(units):
            if not isinstance(raw, Mapping) or set(raw) != {
                "idx",
                "unit_type",
                "title",
                "body",
                "sidecar",
            }:
                raise HybridQaFormalControllerError("corpus unit schema drifted")
            sidecar = raw.get("sidecar")
            if not isinstance(sidecar, Mapping) or set(sidecar) != {
                "table_key",
                "row_ordinal",
                "link_target_keys",
            }:
                raise HybridQaFormalControllerError("corpus sidecar schema drifted")
            title = raw.get("title")
            body = raw.get("body")
            index = raw.get("idx")
            unit_type = raw.get("unit_type")
            table_key = sidecar.get("table_key")
            row_ordinal = sidecar.get("row_ordinal")
            links = sidecar.get("link_target_keys")
            if (
                type(index) is not int
                or index != position
                or not isinstance(unit_type, str)
                or unit_type not in {"table_row", "linked_passage"}
                or not isinstance(title, str)
                or not title.strip()
                or not isinstance(body, str)
                or not body.strip()
                or not isinstance(table_key, str)
                or not table_key
                or (
                    unit_type == "table_row"
                    and (type(row_ordinal) is not int or row_ordinal < 0)
                )
                or (unit_type == "linked_passage" and row_ordinal is not None)
                or not isinstance(links, list)
                or any(not isinstance(link, str) for link in links)
            ):
                raise HybridQaFormalControllerError("corpus text/ordinal drifted")
            article = {"idx": position, "title": title, "body": body}
            articles.append(article)
            minilm_articles.append(minilm_adapter.ArticleText(position, title, body))
            try:
                atomic_units.append(
                    operator.AtomicUnit(
                        position,
                        unit_type,
                        table_key,
                        row_ordinal,
                        tuple(links),
                    )
                )
            except operator.HybridQaOperatorError as exc:
                raise HybridQaFormalControllerError("corpus sidecar drifted") from exc
            documents.append(title + "\n\n" + body)
        multiplicity = Counter(documents)
        duplicate_groups = sum(count > 1 for count in multiplicity.values())
        duplicate_units = sum(count for count in multiplicity.values() if count > 1)
        if (
            pack.get("duplicate_text_group_count") != duplicate_groups
            or pack.get("duplicate_text_unit_count") != duplicate_units
        ):
            raise HybridQaFormalControllerError("corpus duplicate aggregate drifted")
        try:
            graph = operator.build_typed_graph(atomic_units)
        except operator.HybridQaOperatorError as exc:
            raise HybridQaFormalControllerError("typed graph formation failed") from exc
        corpus = CorpusPack(
            tuple(articles),
            tuple(minilm_articles),
            graph,
            str(pack["corpus_pack_sha256"]),
            duplicate_groups,
            duplicate_units,
            self.acquisition_receipt_sha256,
        )
        self._corpus_binding = (corpus.pack_sha256, corpus.graph.graph_sha256)
        return corpus

    def load_view(
        self,
        block: str,
        *,
        a_hold_authorization: runner.AnchorScoreSeal | None = None,
        controller_root: Path | None = None,
    ) -> BlockView:
        if block not in BLOCK_COUNTS:
            raise HybridQaFormalControllerError("view block is invalid")
        if block == "M_search":
            self._verify_promoted_authorization(
                a_hold_authorization=a_hold_authorization,
                controller_root=controller_root,
            )
        elif a_hold_authorization is not None or controller_root is not None:
            raise HybridQaFormalControllerError(
                "pre-promotion view received an extraneous capability"
            )
        filename = f"{block}.view.private.json"
        pack = self._load_pack(filename, self_hash_field="block_view_sha256")
        rows = pack.get("items")
        if (
            set(pack)
            != {
                "schema",
                "version",
                "block",
                "item_count",
                "items",
                "labels_family_gold_or_table_included",
                "block_view_sha256",
            }
            or pack.get("schema") != f"{acquisition.VERSION}_block_view"
            or pack.get("version") != acquisition.VERSION
            or pack.get("block") != block
            or pack.get("item_count") != BLOCK_COUNTS[block]
            or pack.get("labels_family_gold_or_table_included") is not False
            or not isinstance(rows, list)
            or len(rows) != BLOCK_COUNTS[block]
        ):
            raise HybridQaFormalControllerError("block view envelope drifted")
        items: list[ViewItem] = []
        for ordinal, raw in enumerate(rows):
            if not isinstance(raw, Mapping) or set(raw) != {
                "item_commitment_sha256",
                "question",
                "question_postag",
            }:
                raise HybridQaFormalControllerError("block view item schema drifted")
            commitment = _require_sha256(
                raw.get("item_commitment_sha256"), "item commitment"
            )
            question = raw.get("question")
            postag = raw.get("question_postag")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(postag, str)
                or not postag.strip()
            ):
                raise HybridQaFormalControllerError("block view text drifted")
            if commitment != acquisition.item_commitment(
                block=block,
                ordinal=ordinal,
                question=question,
                question_postag=postag,
            ):
                raise HybridQaFormalControllerError(
                    "block view item commitment drifted"
                )
            items.append(ViewItem(commitment, question, postag))
        if len({item.item_commitment_sha256 for item in items}) != len(items):
            raise HybridQaFormalControllerError("block view commitment duplicated")
        view = BlockView(block, tuple(items), str(pack["block_view_sha256"]))
        self._view_bindings[block] = (
            view.view_sha256,
            tuple(sorted(item.item_commitment_sha256 for item in view.items)),
        )
        return view

    def _verified_controller_root(self, controller_root: Path | None) -> Path:
        if not isinstance(controller_root, Path):
            raise HybridQaFormalControllerError("controller root capability is absent")
        try:
            observed = controller_root.resolve(strict=True)
            expected = (self.project / CONTROLLER_ROOT_RELATIVE).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HybridQaFormalControllerError(
                "controller root capability is unavailable"
            ) from exc
        if observed != expected:
            raise HybridQaFormalControllerError("controller root capability drifted")
        _assert_private_directory(
            observed, "controller root", expected_mode=0o700
        )
        return observed

    def _verify_promoted_authorization(
        self,
        *,
        a_hold_authorization: runner.AnchorScoreSeal | None,
        controller_root: Path | None,
        policy_seal: runner.PolicySeal | None = None,
    ) -> Path:
        if (
            not isinstance(a_hold_authorization, runner.AnchorScoreSeal)
            or a_hold_authorization.block != "A_hold"
            or not a_hold_authorization.evaluator_promoted
            or (
                policy_seal is not None
                and a_hold_authorization.policies.policy_receipt_sha256
                != policy_seal.policy_receipt_sha256
            )
        ):
            raise HybridQaFormalControllerError(
                "M_search lacks a promoted policy-matched A_hold score seal"
            )
        root = self._verified_controller_root(controller_root)
        self._verify_durable_policy(
            root=root,
            policy_seal=a_hold_authorization.policies,
        )
        persisted, _ = _load_canonical_object(
            root / "A_hold.score_receipt.json",
            field="durable A_hold score receipt",
            private=True,
        )
        verify_self_hash(persisted, "score_receipt_sha256")
        if persisted != a_hold_authorization.receipt:
            raise HybridQaFormalControllerError(
                "A_hold promotion capability is not durable"
            )
        return root

    @staticmethod
    def _verify_durable_policy(
        *, root: Path, policy_seal: runner.PolicySeal
    ) -> None:
        persisted, _ = _load_canonical_object(
            root / "F_search.policy_receipt.json",
            field="durable F_search policy receipt",
            private=True,
        )
        verify_self_hash(persisted, "policy_receipt_sha256")
        if persisted != policy_seal.receipt:
            raise HybridQaFormalControllerError(
                "F_search policy capability is not durable"
            )

    def load_labels(
        self,
        block: str,
        *,
        expected_view: BlockView,
        corpus: CorpusPack,
        archive_capability: ArchiveCapability,
        feature_seal: runner.FeatureSeal,
        controller_root: Path,
        policy_seal: runner.PolicySeal | None = None,
        a_hold_authorization: runner.AnchorScoreSeal | None = None,
    ) -> LabelPack:
        if block not in {"A_form", "A_hold", "M_search"}:
            raise HybridQaFormalControllerError("label capability is invalid")
        if expected_view.block != block:
            raise HybridQaFormalControllerError("label/view block binding drifted")
        root = self._verified_controller_root(controller_root)
        expected_view_binding = (
            expected_view.view_sha256,
            tuple(
                sorted(
                    item.item_commitment_sha256 for item in expected_view.items
                )
            ),
        )
        if self._view_bindings.get(block) != expected_view_binding:
            raise HybridQaFormalControllerError("view capability was not issued")
        if (
            not isinstance(corpus, CorpusPack)
            or self._corpus_binding
            != (corpus.pack_sha256, corpus.graph.graph_sha256)
        ):
            raise HybridQaFormalControllerError("corpus capability was not issued")
        try:
            recomputed_graph_sha256 = operator.recompute_graph_sha256(corpus.graph)
        except operator.HybridQaOperatorError as exc:
            raise HybridQaFormalControllerError("corpus graph capability drifted") from exc
        if recomputed_graph_sha256 != corpus.graph.graph_sha256:
            raise HybridQaFormalControllerError("corpus graph capability drifted")
        if (
            not isinstance(archive_capability, ArchiveCapability)
            or archive_capability.block != block
            or not isinstance(feature_seal, runner.FeatureSeal)
            or feature_seal.block != block
            or archive_capability.receipt.get("feature_receipt_sha256")
            != feature_seal.feature_receipt_sha256
            or set(feature_seal.item_commitments)
            != set(expected_view_binding[1])
        ):
            raise HybridQaFormalControllerError(
                "late-label archive/feature capability drifted"
            )
        archive_capability.verify_durable(
            root,
            acquisition_receipt_sha256=self.acquisition_receipt_sha256,
            corpus_pack_sha256=corpus.pack_sha256,
            graph_sha256=corpus.graph.graph_sha256,
            block_view_sha256=expected_view.view_sha256,
            feature_seal=feature_seal,
        )
        if block == "A_form":
            if policy_seal is not None or a_hold_authorization is not None:
                raise HybridQaFormalControllerError(
                    "A_form label open received an extraneous capability"
                )
        elif block == "A_hold":
            if (
                not isinstance(policy_seal, runner.PolicySeal)
                or a_hold_authorization is not None
            ):
                raise HybridQaFormalControllerError(
                    "A_hold labels require the frozen F policy"
                )
            self._verify_durable_policy(root=root, policy_seal=policy_seal)
        else:
            if not isinstance(policy_seal, runner.PolicySeal):
                raise HybridQaFormalControllerError(
                    "M_search labels require the frozen F policy"
                )
            self._verify_durable_policy(root=root, policy_seal=policy_seal)
            self._verify_promoted_authorization(
                a_hold_authorization=a_hold_authorization,
                controller_root=root,
                policy_seal=policy_seal,
            )
        pack = self._load_pack(
            f"{block}.labels.sealed.json", self_hash_field="label_pack_sha256"
        )
        rows = pack.get("items")
        if (
            set(pack)
            != {
                "schema",
                "version",
                "block",
                "item_count",
                "block_view_sha256",
                "corpus_pack_sha256",
                "items",
                "label_pack_sha256",
            }
            or pack.get("schema") != f"{acquisition.VERSION}_label_pack"
            or pack.get("version") != acquisition.VERSION
            or pack.get("block") != block
            or pack.get("item_count") != BLOCK_COUNTS[block]
            or pack.get("block_view_sha256") != expected_view.view_sha256
            or pack.get("corpus_pack_sha256") != corpus.pack_sha256
            or not isinstance(rows, list)
            or len(rows) != BLOCK_COUNTS[block]
        ):
            raise HybridQaFormalControllerError("label pack envelope drifted")
        labels: dict[str, LabelRow] = {}
        family_counts: Counter[str] = Counter()
        for raw in rows:
            if not isinstance(raw, Mapping) or set(raw) != {
                "item_commitment_sha256",
                "family",
                "gold_indices",
            }:
                raise HybridQaFormalControllerError("label row schema drifted")
            commitment = _require_sha256(
                raw.get("item_commitment_sha256"), "label item commitment"
            )
            family = raw.get("family")
            gold = raw.get("gold_indices")
            if (
                family not in FAMILIES
                or not isinstance(gold, list)
                or not 1 <= len(gold) <= 3
                or any(type(index) is not int or not 0 <= index < 609 for index in gold)
                or gold != sorted(set(gold))
                or commitment in labels
            ):
                raise HybridQaFormalControllerError("label row content drifted")
            _validate_gold_topology(
                family=family,
                gold_indices=gold,
                graph=corpus.graph,
            )
            labels[commitment] = LabelRow(commitment, family, tuple(gold))
            family_counts[family] += 1
        expected_commitments = {
            item.item_commitment_sha256 for item in expected_view.items
        }
        expected_per_family = acquisition.PER_FAMILY_QUOTA[block]
        if set(labels) != expected_commitments or family_counts != Counter(
            {family: expected_per_family for family in FAMILIES}
        ):
            raise HybridQaFormalControllerError("label/view or family binding drifted")
        return LabelPack(block, labels, str(pack["label_pack_sha256"]))

    def _load_fold_secret(self) -> bytes:
        self._assert_root_intact()
        secret = _read_private_bytes(
            acquisition.SECRET_FILENAME,
            field="selection/fold secret",
            dir_fd=self._root_fd,
        )
        if len(secret) != 32:
            raise HybridQaFormalControllerError("selection/fold secret length drifted")
        commitment = acquisition._selection_secret_commitment(secret)
        if self.public.get("selection_secret_commitment_sha256") != commitment:
            raise HybridQaFormalControllerError("selection/fold secret commitment drifted")
        return secret

    def fit_e2(
        self,
        *,
        feature_seal: runner.FeatureSeal,
        utilities: Mapping[tuple[str, str], Any],
    ) -> runner.E2FitSeal:
        secret = self._load_fold_secret()
        try:
            return runner.fit_e2(
                feature_seal=feature_seal,
                utilities=utilities,
                fold_secret=secret,
            )
        finally:
            del secret


class HippoGateway(Protocol):
    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]: ...

    def retrieve(self, *, block: str, queries: Sequence[str]) -> object: ...


@dataclass(frozen=True)
class PreparedCorpus:
    corpus: CorpusPack
    embedding_index: minilm_adapter.CorpusEmbeddingIndex
    hippo_build_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class BlockExecution:
    block: str
    view: BlockView
    items: tuple[runner.ItemExecution, ...]
    hippo_top5_by_commitment: Mapping[str, tuple[int, ...]]
    feature_seal: runner.FeatureSeal
    execution_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.block != self.view.block
            or self.block not in BLOCK_COUNTS
            or len(self.items) != BLOCK_COUNTS[self.block]
            or len(self.hippo_top5_by_commitment) != BLOCK_COUNTS[self.block]
            or not isinstance(self.feature_seal, runner.FeatureSeal)
            or self.feature_seal.block != self.block
        ):
            raise HybridQaFormalControllerError("block execution shape drifted")
        item_commitments = {
            item.item_commitment_sha256 for item in self.items
        }
        if (
            len(item_commitments) != len(self.items)
            or item_commitments
            != {item.item_commitment_sha256 for item in self.view.items}
            or item_commitments != set(self.hippo_top5_by_commitment)
            or item_commitments != set(self.feature_seal.item_commitments)
        ):
            raise HybridQaFormalControllerError(
                "block execution commitment alignment drifted"
            )


def _prepare_corpus(
    *,
    corpus: CorpusPack,
    encoder: runner.Encoder,
    hippo: HippoGateway,
) -> PreparedCorpus:
    """Build the two independent corpus-wide indices concurrently."""

    with ThreadPoolExecutor(max_workers=2) as pool:
        minilm_future = pool.submit(
            minilm_adapter.build_corpus_embedding_index,
            articles=corpus.minilm_articles,
            encoder=encoder,
        )
        hippo_future = pool.submit(hippo.build, corpus.articles)
        embedding_index = minilm_future.result()
        hippo_receipt = hippo_future.result()
    if not isinstance(hippo_receipt, Mapping):
        raise HybridQaFormalControllerError("HippoRAG build receipt drifted")
    if embedding_index.article_count != acquisition.CORPUS_UNIT_COUNT:
        raise HybridQaFormalControllerError("MiniLM index count drifted")
    if (
        hippo_receipt.get("corpus_count") != acquisition.CORPUS_UNIT_COUNT
        or hippo_receipt.get("duplicate_text_group_count")
        != corpus.duplicate_text_group_count
        or hippo_receipt.get("duplicate_text_unit_count")
        != corpus.duplicate_text_unit_count
        or hippo_receipt.get("index_call_count") != 1
    ):
        raise HybridQaFormalControllerError("HippoRAG build/corpus binding drifted")
    return PreparedCorpus(
        corpus=corpus,
        embedding_index=embedding_index,
        hippo_build_receipt=dict(hippo_receipt),
    )


def _execute_local_views(
    *,
    views: Sequence[BlockView],
    prepared: PreparedCorpus,
    encoder: runner.Encoder,
) -> dict[str, tuple[runner.ItemExecution, ...]]:
    rows = tuple(
        runner.BulkQueryInput(
            item.item_commitment_sha256,
            item.question,
            item.question_postag,
        )
        for view in views
        for item in view.items
    )
    tensors = runner.build_query_semantic_tensors_bulk(
        rows=rows,
        index=prepared.embedding_index,
        encoder=encoder,
    )
    # Eager submission saturates the frozen 32-worker cap.  Each task executes
    # all four recipes from the same verified item tensor, preserving the
    # operator's shared preparation while avoiding serial item execution.
    with ThreadPoolExecutor(
        max_workers=min(LOCAL_WORKER_CAP, len(rows))
    ) as pool:
        futures = {
            row.item_commitment_sha256: pool.submit(
                runner.execute_item,
                item_commitment_sha256=row.item_commitment_sha256,
                graph=prepared.corpus.graph,
                tensor=tensors[row.item_commitment_sha256],
            )
            for row in rows
        }
        completed = {
            commitment: future.result()
            for commitment, future in futures.items()
        }
    return {
        view.block: tuple(
            sorted(
                (
                    completed[item.item_commitment_sha256]
                    for item in view.items
                ),
                key=lambda item: item.item_commitment_sha256,
            )
        )
        for view in views
    }


def _execute_views_with_hippo(
    *,
    views: Sequence[BlockView],
    prepared: PreparedCorpus,
    encoder: runner.Encoder,
    hippo: HippoGateway,
    hippo_stage: str,
) -> dict[str, BlockExecution]:
    canonical_views = tuple(views)
    if not canonical_views or len({view.block for view in canonical_views}) != len(
        canonical_views
    ):
        raise HybridQaFormalControllerError("execution view set drifted")
    queries = tuple(item.question for view in canonical_views for item in view.items)
    schedule = tuple(
        (view.block, item.item_commitment_sha256)
        for view in canonical_views
        for item in view.items
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        local_future = pool.submit(
            _execute_local_views,
            views=canonical_views,
            prepared=prepared,
            encoder=encoder,
        )
        hippo_future = pool.submit(
            hippo.retrieve,
            block=hippo_stage,
            queries=queries,
        )
        local_by_block = local_future.result()
        hippo_batch = hippo_future.result()
    indices = getattr(hippo_batch, "indices", None)
    hippo_receipt = getattr(hippo_batch, "receipt", None)
    if (
        not isinstance(indices, tuple)
        or len(indices) != len(schedule)
        or not isinstance(hippo_receipt, Mapping)
        or hippo_receipt.get("query_count") != len(schedule)
        or hippo_receipt.get("index_call_count") != 0
    ):
        raise HybridQaFormalControllerError("combined HippoRAG retrieval drifted")
    hippo_by_block: dict[str, dict[str, tuple[int, ...]]] = {
        view.block: {} for view in canonical_views
    }
    for (block, commitment), top5 in zip(schedule, indices, strict=True):
        try:
            row = runner.HippoRetrieval(commitment, tuple(top5))
        except (TypeError, runner.HybridQaFormalRunnerError) as exc:
            raise HybridQaFormalControllerError("HippoRAG top five drifted") from exc
        hippo_by_block[block][commitment] = row.top5
    combined_receipt_sha = _require_sha256(
        hippo_receipt.get("receipt_sha256"), "HippoRAG retrieval receipt"
    )
    schedule_sha = stable_hash([[block, commitment] for block, commitment in schedule])
    output: dict[str, BlockExecution] = {}
    for view in canonical_views:
        items = local_by_block[view.block]
        traces = tuple(
            trace for item in items for trace in item.recipe_traces
        )
        feature_seal = runner.seal_feature_matrix(
            block=view.block, traces=traces
        )
        receipt_body = {
            "schema": f"{VERSION}_label_free_execution_receipt",
            "version": VERSION,
            "block": view.block,
            "item_count": len(view.items),
            "acquisition_receipt_sha256": (
                prepared.corpus.acquisition_receipt_sha256
            ),
            "corpus_pack_sha256": prepared.corpus.pack_sha256,
            "block_view_sha256": view.view_sha256,
            "item_commitment_set_sha256": stable_hash(
                sorted(item.item_commitment_sha256 for item in view.items)
            ),
            "feature_receipt_sha256": feature_seal.feature_receipt_sha256,
            "graph_sha256": prepared.corpus.graph.graph_sha256,
            "minilm_index_sha256": prepared.embedding_index.index_sha256,
            "hipporag_build_receipt_sha256": _require_sha256(
                prepared.hippo_build_receipt.get("receipt_sha256"),
                "HippoRAG build receipt",
            ),
            "combined_hipporag_retrieval_receipt_sha256": combined_receipt_sha,
            "combined_query_schedule_sha256": schedule_sha,
            "projected_hipporag_top5_sha256": stable_hash(
                [
                    [commitment, list(hippo_by_block[view.block][commitment])]
                    for commitment in sorted(hippo_by_block[view.block])
                ]
            ),
            "local_worker_cap": LOCAL_WORKER_CAP,
            "all_item_matrices_eagerly_submitted_before_join": True,
            "local_and_official_jobs_submitted_before_join": True,
            "labels_family_gold_or_utility_accessed": False,
            "raw_query_or_corpus_text_persisted": False,
            "online_evaluator_calls": 0,
        }
        execution_receipt = _self_hashed(
            receipt_body, "execution_receipt_sha256"
        )
        output[view.block] = BlockExecution(
            block=view.block,
            view=view,
            items=items,
            hippo_top5_by_commitment=hippo_by_block[view.block],
            feature_seal=feature_seal,
            execution_receipt=execution_receipt,
        )
    return output


def _action_payload(trace: operator.ActionTrace) -> dict[str, Any]:
    try:
        operator.verify_action_trace(trace)
    except operator.HybridQaOperatorError as exc:
        raise HybridQaFormalControllerError("archive action trace drifted") from exc
    return {
        "recipe_id": trace.recipe_id,
        "output_top5": list(trace.output_top5),
        "retained_raw_top3": list(trace.retained_raw_top3),
        "selection_steps": [
            [
                step.output_slot,
                step.selected_unit_ordinal,
                step.disposition,
                step.residual_facet_coverage_gain_int,
                step.direct_anchor,
                step.path_length,
                step.path_strength_int,
            ]
            for step in trace.selection_steps
        ],
        "raw_dense_order_sha256": trace.raw_dense_order_sha256,
        "graph_sha256": trace.graph_sha256,
        "query_sha256": trace.query_sha256,
        "semantic_tensor_sha256": trace.semantic_tensor_sha256,
        "reachability_sha256": trace.reachability_sha256,
        "candidate_scan_sha256": trace.candidate_scan_sha256,
        "candidate_universe_size": trace.candidate_universe_size,
        "candidate_score_evaluations": trace.candidate_score_evaluations,
        "semantic_cell_scan_count": trace.semantic_cell_scan_count,
        "hipporag_candidate_or_feature_count": (
            trace.hipporag_candidate_or_feature_count
        ),
        "trace_sha256": trace.trace_sha256,
    }


def _archive_payload(execution: BlockExecution) -> dict[str, Any]:
    items = sorted(execution.items, key=lambda item: item.item_commitment_sha256)
    body = {
        "schema": f"{VERSION}_label_free_archive",
        "version": VERSION,
        "block": execution.block,
        "item_count": len(items),
        "acquisition_receipt_sha256": execution.execution_receipt[
            "acquisition_receipt_sha256"
        ],
        "corpus_pack_sha256": execution.execution_receipt[
            "corpus_pack_sha256"
        ],
        "block_view_sha256": execution.view.view_sha256,
        "graph_sha256": execution.execution_receipt["graph_sha256"],
        "item_commitment_set_sha256": execution.feature_seal.item_commitment_set_sha256,
        "feature_receipt_sha256": execution.feature_seal.feature_receipt_sha256,
        "execution_receipt": dict(execution.execution_receipt),
        "feature_receipt": execution.feature_seal.receipt,
        "hipporag_top5": [
            [
                commitment,
                list(execution.hippo_top5_by_commitment[commitment]),
            ]
            for commitment in sorted(execution.hippo_top5_by_commitment)
        ],
        "items": [
            {
                "item_commitment_sha256": item.item_commitment_sha256,
                "actions": [_action_payload(trace) for trace in item.action_traces],
                "feature_traces": [
                    trace.payload() for trace in item.recipe_traces
                ],
            }
            for item in items
        ],
        "complete_action_and_feature_traces_persisted": True,
        "raw_query_corpus_label_family_gold_or_utility_persisted": False,
        "online_evaluator_calls": 0,
    }
    return _self_hashed(body, "archive_sha256")


def _persist_and_verify_archive(
    *, controller_root: Path, execution: BlockExecution
) -> ArchiveCapability:
    archive = _archive_payload(execution)
    archive_path = controller_root / f"{execution.block}.label_free_archive.json"
    file_sha = _write_exclusive(archive_path, archive, mode=0o600)
    seal_body = {
        "schema": f"{VERSION}_label_free_archive_seal",
        "version": VERSION,
        "block": execution.block,
        "acquisition_receipt_sha256": execution.execution_receipt[
            "acquisition_receipt_sha256"
        ],
        "corpus_pack_sha256": execution.execution_receipt[
            "corpus_pack_sha256"
        ],
        "block_view_sha256": execution.view.view_sha256,
        "graph_sha256": execution.execution_receipt["graph_sha256"],
        "item_commitment_set_sha256": execution.feature_seal.item_commitment_set_sha256,
        "execution_receipt_sha256": execution.execution_receipt[
            "execution_receipt_sha256"
        ],
        "archive_file_sha256": file_sha,
        "archive_sha256": archive["archive_sha256"],
        "feature_receipt_sha256": execution.feature_seal.feature_receipt_sha256,
        "label_pack_opened_before_seal": False,
    }
    seal = _self_hashed(seal_body, "archive_seal_sha256")
    seal_path = controller_root / f"{execution.block}.archive_seal.json"
    _write_exclusive(seal_path, seal, mode=0o600)
    observed, observed_file_sha = _load_canonical_object(
        archive_path,
        field=f"{execution.block} archive",
        expected_file_sha256=file_sha,
        private=True,
    )
    verify_self_hash(observed, "archive_sha256")
    if observed_file_sha != seal["archive_file_sha256"] or observed != archive:
        raise HybridQaFormalControllerError("durable archive verification drifted")
    capability = ArchiveCapability(
        block=execution.block,
        receipt_json=_canonical_bytes(seal).decode("ascii"),
    )
    capability.verify_durable(
        controller_root,
        acquisition_receipt_sha256=execution.execution_receipt[
            "acquisition_receipt_sha256"
        ],
        corpus_pack_sha256=execution.execution_receipt["corpus_pack_sha256"],
        graph_sha256=execution.execution_receipt["graph_sha256"],
        block_view_sha256=execution.view.view_sha256,
        feature_seal=execution.feature_seal,
    )
    return capability


def _a_form_utilities(
    execution: BlockExecution, labels: LabelPack
) -> dict[tuple[str, str], Any]:
    if execution.block != labels.block or execution.block != "A_form":
        raise HybridQaFormalControllerError("A_form utility binding drifted")
    utilities: dict[tuple[str, str], Any] = {}
    for item in execution.items:
        label = labels.by_commitment.get(item.item_commitment_sha256)
        if not isinstance(label, LabelRow):
            raise HybridQaFormalControllerError("A_form label join drifted")
        for recipe_id, output in item.outputs.items():
            utilities[(item.item_commitment_sha256, recipe_id)] = runner.item_utility(
                output, label.gold_indices
            )[0]
    if len(utilities) != BLOCK_COUNTS["A_form"] * len(runner.RECIPE_IDS):
        raise HybridQaFormalControllerError("A_form utility matrix drifted")
    return utilities


def _anchor_labels(labels: LabelPack) -> tuple[runner.AnchorLabel, ...]:
    return tuple(
        runner.AnchorLabel(
            row.item_commitment_sha256,
            row.gold_indices,
            row.family,
        )
        for row in sorted(
            labels.by_commitment.values(),
            key=lambda value: value.item_commitment_sha256,
        )
    )


def _anchor_hippo_seal(execution: BlockExecution) -> runner.HippoRetrievalSeal:
    return runner.seal_hippo_retrievals(
        block=execution.block,
        rows=tuple(
            runner.HippoRetrieval(commitment, top5)
            for commitment, top5 in execution.hippo_top5_by_commitment.items()
        ),
    )


def _promotion_payload(
    *, hold_score: runner.AnchorScoreSeal, policies: runner.PolicySeal
) -> dict[str, Any]:
    if (
        not isinstance(hold_score, runner.AnchorScoreSeal)
        or hold_score.block != "A_hold"
        or not hold_score.evaluator_promoted
        or not isinstance(policies, runner.PolicySeal)
        or hold_score.policies.policy_receipt_sha256
        != policies.policy_receipt_sha256
    ):
        raise HybridQaFormalControllerError(
            "promotion receipt lacks a promoted policy-matched A_hold score"
        )
    body = {
        "schema": f"{VERSION}_promotion_receipt",
        "version": VERSION,
        "status": "A_hold_evaluator_promoted_M_search_authorized",
        "A_hold_score_receipt_sha256": hold_score.score_receipt_sha256,
        "policy_receipt_sha256": policies.policy_receipt_sha256,
        "M_search_authorized": True,
        "raw_content_or_item_level_utility_persisted": False,
        "online_evaluator_calls": 0,
    }
    return _self_hashed(body, "promotion_receipt_sha256")


def _persist_receipt(
    controller_root: Path,
    filename: str,
    receipt: Mapping[str, Any],
    *,
    self_hash_field: str,
) -> str:
    verify_self_hash(receipt, self_hash_field)
    file_sha = _write_exclusive(
        controller_root / filename, dict(receipt), mode=0o600
    )
    observed, observed_sha = _load_canonical_object(
        controller_root / filename,
        field=filename,
        expected_file_sha256=file_sha,
        private=True,
    )
    verify_self_hash(observed, self_hash_field)
    if observed != dict(receipt) or observed_sha != file_sha:
        raise HybridQaFormalControllerError("persisted receipt verification drifted")
    return file_sha


def _terminal_failure(
    controller_root: Path,
    *,
    stage: str,
    exc: BaseException,
) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure",
        "version": VERSION,
        "status": "terminal_no_retry_replay_resample_or_threshold_change",
        "failure_stage": stage,
        "exception_class": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8", errors="replace")
        ).hexdigest(),
        "item_level_result_or_raw_content_persisted_publicly": False,
        "online_evaluator_calls": 0,
    }
    try:
        _write_exclusive(
            controller_root / FAILURE_FILENAME,
            _self_hashed(body, "failure_sha256"),
            mode=0o600,
        )
    except BaseException:
        pass


def run_formal_lifecycle(project_root: str | Path) -> dict[str, Any]:
    """Run the only production HybridQA lifecycle without retry or injection."""

    project = _canonical_project(project_root)
    stage = "prerequisite_verification"
    # Recompute every implementation file hash before any output, pack read,
    # model load or formal inference.
    try:
        freeze = acquisition._verify_implementation_freeze(project)
    except Exception as exc:
        raise HybridQaFormalControllerError(
            "implementation freeze verification failed"
        ) from exc
    boundary = AcquisitionBoundary(
        project,
        expected_freeze_sha256=_require_sha256(
            freeze.get("freeze_sha256"), "implementation freeze"
        ),
    )
    config = local_runtime.default_formal_runtime_config(project)
    try:
        preflight = local_runtime.preflight_formal_runtime_config(config)
    except BaseException:
        boundary.close()
        raise
    controller_root = project / CONTROLLER_ROOT_RELATIVE
    if os.path.lexists(controller_root):
        boundary.close()
        raise HybridQaFormalControllerError(
            "formal controller root already exists and is nonreusable"
        )
    try:
        controller_root.mkdir(mode=0o700)
    except OSError as exc:
        boundary.close()
        raise HybridQaFormalControllerError(
            "formal controller root creation failed"
        ) from exc
    marker_body = {
        "schema": f"{VERSION}_one_shot_marker",
        "version": VERSION,
        "status": "formal_lifecycle_started",
        "design_sha256": acquisition.DESIGN_SHA256,
        "implementation_freeze_sha256": freeze["freeze_sha256"],
        "acquisition_receipt_sha256": boundary.acquisition_receipt_sha256,
        "runtime_preflight_sha256": stable_hash(preflight),
        "retry_replay_resample_or_threshold_change_authorized": False,
        "M_search_authorized_at_marker": False,
    }
    marker = _self_hashed(marker_body, "marker_sha256")
    _write_exclusive(
        controller_root / MARKER_FILENAME, marker, mode=0o600
    )

    archive_seals: dict[str, ArchiveCapability] = {}
    m_score: runner.AnchorScoreSeal | None = None
    promotion_payload: dict[str, Any] | None = None
    try:
        stage = "offline_runtime_open_and_canary"
        runtime = local_runtime.open_runtime(config)

        stage = "private_corpus_and_allowed_view_open"
        corpus = boundary.load_corpus()
        views = (
            boundary.load_view("A_form"),
            boundary.load_view("F_search"),
            boundary.load_view("A_hold"),
        )

        stage = "parallel_corpus_index_build"
        prepared = _prepare_corpus(
            corpus=corpus,
            encoder=runtime.encoder,
            hippo=runtime.hippo,
        )

        stage = "parallel_A_form_F_search_A_hold_label_free_execution"
        initial = _execute_views_with_hippo(
            views=views,
            prepared=prepared,
            encoder=runtime.encoder,
            hippo=runtime.hippo,
            hippo_stage="A_form_F_search_A_hold",
        )

        stage = "durable_initial_label_free_archives"
        # All three available blocks are durable before the first late label
        # open.  Their archives contain complete action/features and official
        # top fives but no raw query/corpus text or label-derived value.
        for block in ("A_form", "F_search", "A_hold"):
            archive_seals[block] = _persist_and_verify_archive(
                controller_root=controller_root,
                execution=initial[block],
            )

        stage = "A_form_late_label_open_and_E2_fit"
        a_labels = boundary.load_labels(
            "A_form",
            expected_view=initial["A_form"].view,
            corpus=corpus,
            archive_capability=archive_seals["A_form"],
            feature_seal=initial["A_form"].feature_seal,
            controller_root=controller_root,
        )
        utilities = _a_form_utilities(initial["A_form"], a_labels)
        fit = boundary.fit_e2(
            feature_seal=initial["A_form"].feature_seal,
            utilities=utilities,
        )
        _persist_receipt(
            controller_root,
            "A_form.e2_fit_receipt.json",
            fit.receipt,
            self_hash_field="fit_receipt_sha256",
        )
        del utilities, a_labels

        stage = "F_search_policy_freeze"
        policies = runner.freeze_f_policies(
            feature_seal=initial["F_search"].feature_seal,
            fit_seal=fit,
        )
        _persist_receipt(
            controller_root,
            "F_search.policy_receipt.json",
            policies.receipt,
            self_hash_field="policy_receipt_sha256",
        )

        stage = "A_hold_late_label_open_and_primary_score"
        hold_labels = boundary.load_labels(
            "A_hold",
            expected_view=initial["A_hold"].view,
            corpus=corpus,
            archive_capability=archive_seals["A_hold"],
            feature_seal=initial["A_hold"].feature_seal,
            controller_root=controller_root,
            policy_seal=policies,
        )
        hold_score = runner.score_anchor(
            block="A_hold",
            items=initial["A_hold"].items,
            labels=_anchor_labels(hold_labels),
            anchor_feature_seal=initial["A_hold"].feature_seal,
            hippo_retrieval_seal=_anchor_hippo_seal(initial["A_hold"]),
            policy_seal=policies,
        )
        _persist_receipt(
            controller_root,
            "A_hold.score_receipt.json",
            hold_score.receipt,
            self_hash_field="score_receipt_sha256",
        )
        del hold_labels

        if hold_score.evaluator_promoted:
            stage = "A_hold_promotion_receipt"
            promotion_payload = _promotion_payload(
                hold_score=hold_score,
                policies=policies,
            )
            _persist_receipt(
                controller_root,
                "A_hold.promotion_receipt.json",
                promotion_payload,
                self_hash_field="promotion_receipt_sha256",
            )

            stage = "M_search_post_promotion_view_open_and_label_free_execution"
            m_view = boundary.load_view(
                "M_search",
                a_hold_authorization=hold_score,
                controller_root=controller_root,
            )
            m_execution = _execute_views_with_hippo(
                views=(m_view,),
                prepared=prepared,
                encoder=runtime.encoder,
                hippo=runtime.hippo,
                hippo_stage="M_search",
            )["M_search"]
            archive_seals["M_search"] = _persist_and_verify_archive(
                controller_root=controller_root,
                execution=m_execution,
            )

            stage = "M_search_late_label_open_and_L5_score"
            m_labels = boundary.load_labels(
                "M_search",
                expected_view=m_view,
                corpus=corpus,
                archive_capability=archive_seals["M_search"],
                feature_seal=m_execution.feature_seal,
                controller_root=controller_root,
                policy_seal=policies,
                a_hold_authorization=hold_score,
            )
            m_score = runner.score_anchor(
                block="M_search",
                items=m_execution.items,
                labels=_anchor_labels(m_labels),
                anchor_feature_seal=m_execution.feature_seal,
                hippo_retrieval_seal=_anchor_hippo_seal(m_execution),
                policy_seal=policies,
                a_hold_authorization=hold_score,
            )
            _persist_receipt(
                controller_root,
                "M_search.score_receipt.json",
                m_score.receipt,
                self_hash_field="score_receipt_sha256",
            )
            terminal_status = "valid_promoted_M_search_completed"
        else:
            terminal_status = "valid_A_hold_nonpromotion_M_search_unopened"

        stage = "terminal_result_persistence"
        result_body = {
            "schema": f"{VERSION}_terminal_result",
            "version": VERSION,
            "status": terminal_status,
            "design_sha256": acquisition.DESIGN_SHA256,
            "implementation_freeze_sha256": freeze["freeze_sha256"],
            "acquisition_receipt_sha256": boundary.acquisition_receipt_sha256,
            "marker_sha256": marker["marker_sha256"],
            "runtime_preflight_sha256": stable_hash(preflight),
            "corpus_pack_sha256": corpus.pack_sha256,
            "typed_graph_sha256": corpus.graph.graph_sha256,
            "minilm_index_sha256": prepared.embedding_index.index_sha256,
            "official_hipporag_build_receipt": dict(
                prepared.hippo_build_receipt
            ),
            "archive_seals": {
                block: archive_seals[block].receipt
                for block in sorted(archive_seals)
            },
            "E2_fit_receipt": fit.receipt,
            "F_policy_receipt": policies.receipt,
            "A_hold_score_receipt": hold_score.receipt,
            "M_search_score_receipt": m_score.receipt if m_score else None,
            "promotion_receipt": promotion_payload,
            "M_search_private_view_or_label_opened": m_score is not None,
            "logical_initial_RAW_HippoRAG_Agent_work_units": 3
            * (
                BLOCK_COUNTS["A_form"]
                + BLOCK_COUNTS["F_search"]
                + BLOCK_COUNTS["A_hold"]
            ),
            "logical_M_search_RAW_HippoRAG_Agent_work_units": (
                3 * BLOCK_COUNTS["M_search"] if m_score else 0
            ),
            "retry_replay_or_resample": 0,
            "online_or_Ruoli_evaluator_calls": 0,
            "item_level_results_or_raw_content_persisted_publicly": False,
        }
        result = _self_hashed(result_body, "result_sha256")
        _write_exclusive(
            controller_root / RESULT_FILENAME, result, mode=0o600
        )
        boundary.close()
        return result
    except BaseException as exc:
        _terminal_failure(controller_root, stage=stage, exc=exc)
        boundary.close()
        if isinstance(exc, HybridQaFormalControllerError):
            raise
        raise HybridQaFormalControllerError(
            f"formal lifecycle failed terminally at {stage}"
        ) from exc


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_formal_lifecycle(arguments.project)
    return 0


def main() -> int:
    from assumption_agent.benchmarks import hybridqa_isolated_bootstrap_v1 as bootstrap

    target = "assumption_agent.benchmarks.hybridqa_p6_e2_formal_controller_v1"
    bootstrap.reexec_isolated(target, tuple(os.sys.argv[1:]))
    bootstrap.assert_isolated(target)
    return _main(os.sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AcquisitionBoundary",
    "ArchiveCapability",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "BlockExecution",
    "BlockView",
    "CONTROLLER_ROOT_RELATIVE",
    "CorpusPack",
    "HybridQaFormalControllerError",
    "LabelPack",
    "LabelRow",
    "LOCAL_WORKER_CAP",
    "PreparedCorpus",
    "VERSION",
    "ViewItem",
    "run_formal_lifecycle",
    "stable_hash",
    "verify_self_hash",
]

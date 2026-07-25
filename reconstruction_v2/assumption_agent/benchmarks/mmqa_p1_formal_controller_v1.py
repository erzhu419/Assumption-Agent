"""Source-blind one-shot lifecycle controller for MMQA P1.

The controller reads only the selector's anonymous action packs.  It never
opens the trusted source ledger and cannot read a gold pack directly.  Gold is
released by the selector's archive-bound one-shot opener only after the whole
corresponding label-free action archive is canonical, durable, and sealed.

Local coordinates and candidate-restricted official HippoRAG are injected as
block-batch executors.  They receive opaque work IDs and anonymous content,
never source IDs, family/type, answers, support, or gold.  This generic module
does not load a model, read a formal source, access a network, retry a call, or
perform online evaluation.

The lifecycle is fixed: A_form actions -> late gold -> neutral-aware five-fold
OOF audit -> one full E5 fit; label-free F_search behavior trace (never gold,
never a gate); A_hold four-arm actions -> late gold -> promotion/reality; and,
only after promotion, M_search actions -> late gold -> L5.  Any exception
consumes the lifecycle and emits only a hash-safe terminal receipt.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Protocol

import numpy as np

from . import mmqa_p1_action_integration_v1 as integration
from . import mmqa_p1_local_action_executor_v1 as local_executor
from . import mmqa_p1_private_selection_v1 as selection
from . import mmqa_p1_typed_proof_e5_core_v1 as core


VERSION = "mmqa_p1_formal_controller_v1"
STUDY_ID = selection.STUDY_ID
STUDY_DESIGN_SELF_SHA256 = selection.STUDY_DESIGN_SELF_SHA256
SOURCE_CUSTODY_SELF_SHA256 = selection.SOURCE_CUSTODY_SELF_SHA256

BLOCK_ORDER = selection.BLOCK_ORDER
BLOCK_ITEM_COUNTS = selection.BLOCK_ITEM_COUNTS
FAMILIES = selection.FAMILIES

ATTEMPT_MARKER_FILENAME = "lifecycle.one_shot.private.json"
FINAL_RECEIPT_FILENAME = "lifecycle.final.safe.private.json"
FAILURE_FILENAME = "lifecycle.terminal_failure.private.json"
FULL_MODEL_FILENAME = "e5.full.model.private.json"
STAGE_ACTION_ARCHIVE_FILENAME = "action.archive.private.json"
STAGE_SCORE_FILENAME = "offline.score.private.json"
GOLD_AUTHORIZATION_FILENAME = "gold.open.authorization.private.json"
OOF_AUDIT_FILENAME = "oof.audit.private.json"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID = re.compile(r"mmqa-work-v1-[0-9a-f]{64}\Z")


class MmqaP1FormalControllerError(RuntimeError):
    """The source-blind one-shot lifecycle or a sealed binding drifted."""


class CoordinateProvider(Protocol):
    """One label-free block batch; implementations may batch/parallelize."""

    def __call__(
        self,
        *,
        block: str,
        items: Mapping[str, integration.AnonymousWorkItem],
    ) -> Mapping[
        str,
        Sequence[integration.UnitCoordinates | Mapping[str, object]],
    ]: ...


class HippoExecutor(Protocol):
    """One A_hold candidate-restricted official-HippoRAG batch."""

    def __call__(
        self,
        *,
        block: str,
        payloads: Mapping[
            str, local_executor.CandidateRestrictedHippoRAGPayload
        ],
    ) -> Mapping[str, Sequence[int]]: ...


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1FormalControllerError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value, newline=False)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    if field in body:
        raise MmqaP1FormalControllerError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    if not isinstance(value, Mapping):
        raise MmqaP1FormalControllerError("self-hashed value is not an object")
    body = dict(value)
    claimed = body.pop(field, None)
    if not isinstance(claimed, str) or _HEX64.fullmatch(claimed) is None:
        raise MmqaP1FormalControllerError("self-hash is absent or invalid")
    if not hmac.compare_digest(stable_hash(body), claimed):
        raise MmqaP1FormalControllerError("self-hash drifted")
    return claimed


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise MmqaP1FormalControllerError("durable directory is unavailable") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise MmqaP1FormalControllerError("durable path is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise MmqaP1FormalControllerError("directory parent is unavailable")
            missing.append(cursor)
            cursor = cursor.parent
            continue
        except OSError as exc:
            raise MmqaP1FormalControllerError("directory cannot be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MmqaP1FormalControllerError("directory path is unsafe")
        break
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
            os.chmod(directory, 0o700)
        except OSError as exc:
            raise MmqaP1FormalControllerError("directory cannot be created") from exc
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _create_one_shot_root(path: Path) -> None:
    _ensure_private_directory(path.parent)
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
    except FileExistsError as exc:
        raise MmqaP1FormalControllerError(
            "controller root already exists; replay is forbidden"
        ) from exc
    except OSError as exc:
        raise MmqaP1FormalControllerError("controller root cannot be created") from exc
    _fsync_directory(path)
    _fsync_directory(path.parent)


def _write_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish canonical mode-0600 bytes with no overwrite path."""

    raw = _canonical_bytes(value)
    _ensure_private_directory(path.parent)
    staging = path.with_name(f".{path.name}.part")
    if path.exists() or path.is_symlink() or staging.exists() or staging.is_symlink():
        raise MmqaP1FormalControllerError("one-shot output already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(staging, flags, 0o600)
    except OSError as exc:
        raise MmqaP1FormalControllerError("one-shot staging output failed") from exc
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        os.link(staging, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        os.unlink(staging)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise MmqaP1FormalControllerError("one-shot output publication failed") from exc
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise MmqaP1FormalControllerError("sealed output mode or type drifted")
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode_octal": "0600",
    }


def _read_canonical_private(path: Path, *, label: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise MmqaP1FormalControllerError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
        ):
            raise MmqaP1FormalControllerError(f"{label} is not sealed mode-0600")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MmqaP1FormalControllerError(f"{label} changed while read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MmqaP1FormalControllerError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise MmqaP1FormalControllerError(f"{label} is not canonical")
    return value


def _stage_root(control_root: Path, block: str) -> Path:
    if block not in BLOCK_ORDER:
        raise MmqaP1FormalControllerError("stage block is invalid")
    return control_root / "stages" / block


def _ranking_payload(value: integration.ActionRanking | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "policy_id": value.policy_id,
        "top5_ordinals": list(value.top5_ordinals),
        "selected_bundle_ordinals": (
            None
            if value.selected_bundle_ordinals is None
            else list(value.selected_bundle_ordinals)
        ),
        "selected_bundle_energy_float64_hex": (
            None
            if value.selected_bundle_energy is None
            else value.selected_bundle_energy.hex()
        ),
    }


def _coordinate_payload(value: integration.UnitCoordinates) -> dict[str, Any]:
    return {
        "ordinal": value.ordinal,
        "minilm_similarity_float64_hex": value.minilm_similarity.hex(),
        "cross_encoder_relevance_float64_hex": value.cross_encoder_relevance.hex(),
        "entity_anchor": value.entity_anchor,
        "relation_anchor": value.relation_anchor,
        "numeric_or_temporal_anchor": value.numeric_or_temporal_anchor,
    }


def _score_payload(value: integration.OfflineRankingScore) -> dict[str, Any]:
    return {
        "ndcg_at_5_float64_hex": value.ndcg_at_5.hex(),
        "integer_utility": value.integer_utility,
        "recall_at_5_float64_hex": value.recall_at_5.hex(),
        "connected_gold_row_text_pair_recovered": (
            value.connected_gold_row_text_pair_recovered
        ),
    }


def _paired_payload(value: core.PairedUtilitySummary) -> dict[str, Any]:
    return {
        "item_count": value.item_count,
        "total_integer_delta": value.total_integer_delta,
        "gains": value.gains,
        "harms": value.harms,
        "ties": value.ties,
        "exact_one_sided_p": {
            "numerator": value.exact_one_sided_p.numerator,
            "denominator": value.exact_one_sided_p.denominator,
        },
        "positive_total": value.positive_total,
        "tail_at_most_0_10": value.tail_at_most_alpha,
        "passed": value.passed,
    }


def _model_payload(model: core.E5Model) -> dict[str, Any]:
    if not isinstance(model, core.E5Model):
        raise MmqaP1FormalControllerError("full E5 model type drifted")
    payload = model.payload()
    if payload.get("forbidden_id_or_family_features") != sorted(core.FORBIDDEN_FEATURES):
        raise MmqaP1FormalControllerError("E5 forbidden-feature binding drifted")
    return payload


@dataclass(frozen=True)
class LoadedAction:
    work_id: str
    work_item: integration.AnonymousWorkItem


@dataclass(frozen=True)
class WorkExecution:
    work_id: str
    work_item: integration.AnonymousWorkItem
    coordinates: tuple[integration.UnitCoordinates, ...]
    actions: integration.IntegratedActions
    bundle_first_top5: tuple[tuple[core.ProofBundle, tuple[int, ...]], ...]
    hipporag_top5: tuple[int, ...] | None = None
    hipporag_binding: Mapping[str, object] | None = None


@dataclass(frozen=True)
class StageExecution:
    block: str
    action_pack_sha256: str
    works: tuple[WorkExecution, ...]
    archive: Mapping[str, Any]
    archive_path: Path
    archive_file_sha256: str


def _bundle_contains_exact_gold_pair(
    bundle: core.ProofBundle,
    exact_gold_pairs: Sequence[integration.ExactRowTextLink],
) -> bool:
    if not isinstance(bundle, core.ProofBundle):
        raise MmqaP1FormalControllerError(
            "A_form target bundle type drifted"
        )
    members = set(bundle.node_ordinals)
    return any(
        pair.row_ordinal in members and pair.text_ordinal in members
        for pair in exact_gold_pairs
    )


def _positive_exact_gold_bundles(
    bundles: Sequence[core.ProofBundle],
    exact_gold_pairs: Sequence[integration.ExactRowTextLink],
) -> tuple[core.ProofBundle, ...]:
    checked = tuple(bundles)
    pairs = tuple(exact_gold_pairs)
    if (
        not checked
        or len(set(checked)) != len(checked)
        or not pairs
        or not all(
            isinstance(pair, integration.ExactRowTextLink)
            for pair in pairs
        )
    ):
        raise MmqaP1FormalControllerError(
            "A_form exact-gold target registry drifted"
        )
    return tuple(
        bundle
        for bundle in checked
        if _bundle_contains_exact_gold_pair(bundle, pairs)
    )


@dataclass(frozen=True)
class TrainingSlate:
    work: WorkExecution
    fold: int
    gold_evidence: tuple[int, ...]
    exact_gold_pairs: tuple[integration.ExactRowTextLink, ...]
    positive_exact_gold_bundles: tuple[core.ProofBundle, ...]
    maximum_bundle_first_integer_utility_audit_only: int
    neutral_no_exact_gold_bundle: bool

    def __post_init__(self) -> None:
        positives = tuple(self.positive_exact_gold_bundles)
        if not isinstance(self.work, WorkExecution):
            raise MmqaP1FormalControllerError(
                "A_form exact-gold training target drifted"
            )
        expected = _positive_exact_gold_bundles(
            self.work.actions.bundles, self.exact_gold_pairs
        )
        if (
            type(self.fold) is not int
            or not 0 <= self.fold < 5
            or type(self.maximum_bundle_first_integer_utility_audit_only)
            is not int
            or type(self.neutral_no_exact_gold_bundle) is not bool
            or len(set(positives)) != len(positives)
            or positives != expected
            or not set(positives).issubset(self.work.actions.bundles)
            or any(
                not _bundle_contains_exact_gold_pair(
                    bundle, self.exact_gold_pairs
                )
                for bundle in positives
            )
            or self.neutral_no_exact_gold_bundle != (not positives)
        ):
            raise MmqaP1FormalControllerError(
                "A_form exact-gold training target drifted"
            )
        object.__setattr__(
            self, "positive_exact_gold_bundles", positives
        )


@dataclass(frozen=True)
class AHoldOutcome:
    promotion: core.PromotionDecision
    promotion_sha256: str
    reality_primary_passed: bool
    e5_vs_raw: core.PairedUtilitySummary
    e5_vs_hipporag: core.PairedUtilitySummary
    family_delta_e5_minus_raw: Mapping[str, int]
    family_delta_e5_minus_hipporag: Mapping[str, int]
    score_binding: Mapping[str, Any]


def _load_selection_receipt(
    selection_root: Path, expected_acquisition_sha256: str
) -> dict[str, Any]:
    if _HEX64.fullmatch(expected_acquisition_sha256) is None:
        raise MmqaP1FormalControllerError("selection acquisition binding is invalid")
    try:
        value = selection._load_public_receipt(Path(selection_root))  # noqa: SLF001
    except selection.MmqaP1PrivateSelectionError as exc:
        raise MmqaP1FormalControllerError("selection receipt validation failed") from exc
    if not hmac.compare_digest(
        str(value.get("acquisition_sha256")), expected_acquisition_sha256
    ):
        raise MmqaP1FormalControllerError("selection acquisition binding drifted")
    binding = value.get("binding_self_sha256")
    if (
        not isinstance(binding, Mapping)
        or binding.get("source_custody") != SOURCE_CUSTODY_SELF_SHA256
        or binding.get("study_design") != STUDY_DESIGN_SELF_SHA256
    ):
        raise MmqaP1FormalControllerError("selection study binding drifted")
    return value


def _convert_action_item(value: object) -> LoadedAction:
    if not isinstance(value, Mapping) or set(value) != {
        "work_id",
        "question",
        "nodes",
        "edges",
    }:
        raise MmqaP1FormalControllerError("selector action item shape drifted")
    work_id = value.get("work_id")
    nodes = value.get("nodes")
    edges = value.get("edges")
    if (
        not isinstance(work_id, str)
        or _WORK_ID.fullmatch(work_id) is None
        or not isinstance(nodes, list)
        or not nodes
        or not isinstance(edges, list)
        or not edges
    ):
        raise MmqaP1FormalControllerError("selector action item value drifted")
    rows: list[dict[str, object]] = []
    texts: list[dict[str, object]] = []
    node_type: dict[int, str] = {}
    for expected_ordinal, node in enumerate(nodes):
        if not isinstance(node, Mapping) or set(node) != {
            "ordinal",
            "node_type",
            "content",
        }:
            raise MmqaP1FormalControllerError("selector action node shape drifted")
        ordinal = node.get("ordinal")
        kind = node.get("node_type")
        content = node.get("content")
        if (
            type(ordinal) is not int
            or ordinal != expected_ordinal
            or kind not in {core.ROW, core.TEXT}
            or not isinstance(content, str)
        ):
            raise MmqaP1FormalControllerError("selector action node value drifted")
        node_type[ordinal] = str(kind)
        projection = {"ordinal": ordinal, "serialized_content": content}
        (rows if kind == core.ROW else texts).append(projection)
    directed: set[tuple[int, int, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping) or set(edge) != {
            "source_ordinal",
            "target_ordinal",
            "edge_type",
        }:
            raise MmqaP1FormalControllerError("selector action edge shape drifted")
        source = edge.get("source_ordinal")
        target = edge.get("target_ordinal")
        kind = edge.get("edge_type")
        if type(source) is not int or type(target) is not int:
            raise MmqaP1FormalControllerError("selector action edge ordinal drifted")
        expected_kind = (
            core.ROW_TO_TEXT
            if node_type.get(source) == core.ROW and node_type.get(target) == core.TEXT
            else core.TEXT_TO_ROW
            if node_type.get(source) == core.TEXT and node_type.get(target) == core.ROW
            else None
        )
        if kind != expected_kind or (source, target, str(kind)) in directed:
            raise MmqaP1FormalControllerError("selector typed edge drifted")
        directed.add((source, target, str(kind)))
    links: list[dict[str, int]] = []
    for source, target, kind in sorted(directed):
        if kind != core.ROW_TO_TEXT:
            continue
        if (target, source, core.TEXT_TO_ROW) not in directed:
            raise MmqaP1FormalControllerError("selector edge lacks exact reverse")
        links.append({"row_ordinal": source, "text_ordinal": target})
    if len(directed) != 2 * len(links):
        raise MmqaP1FormalControllerError("selector edges are not exactly bidirectional")
    try:
        work_item = integration.validate_anonymous_work_item(
            {
                "schema": integration.ANONYMOUS_WORK_ITEM_SCHEMA,
                "question": value.get("question"),
                "rows": rows,
                "texts": texts,
                "exact_row_text_links": links,
            }
        )
    except integration.MmqaP1ActionIntegrationError as exc:
        raise MmqaP1FormalControllerError(
            "selector action cannot form AnonymousWorkItem"
        ) from exc
    return LoadedAction(work_id=work_id, work_item=work_item)


def _load_action_pack(
    selection_root: Path,
    receipt: Mapping[str, Any],
    *,
    block: str,
) -> tuple[dict[str, Any], tuple[LoadedAction, ...], Mapping[str, Any]]:
    try:
        binding = selection._pack_binding(receipt, block=block, role="action")  # noqa: SLF001
        value = selection._read_bound_pack(  # noqa: SLF001
            Path(selection_root), binding=binding, label=f"{block} action pack"
        )
    except selection.MmqaP1PrivateSelectionError as exc:
        raise MmqaP1FormalControllerError(f"{block} action pack read failed") from exc
    expected_fields = {
        "schema",
        "version",
        "study_id",
        "block",
        "item_count",
        "item_exact_fields",
        "source_identifier_family_exact_type_answer_support_or_metadata_included",
        "items",
        "action_pack_sha256",
    }
    try:
        semantic = selection.verify_self_hash(value, "action_pack_sha256")
    except selection.MmqaP1PrivateSelectionError as exc:
        raise MmqaP1FormalControllerError("action pack semantic hash failed") from exc
    items = value.get("items")
    if (
        set(value) != expected_fields
        or value.get("schema") != f"{selection.VERSION}_label_free_action_pack_v1"
        or value.get("version") != selection.VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("block") != block
        or value.get("item_count") != BLOCK_ITEM_COUNTS[block]
        or value.get("item_exact_fields") != ["work_id", "question", "nodes", "edges"]
        or value.get(
            "source_identifier_family_exact_type_answer_support_or_metadata_included"
        )
        is not False
        or not isinstance(items, list)
        or len(items) != BLOCK_ITEM_COUNTS[block]
        or semantic != binding.get("semantic_sha256")
    ):
        raise MmqaP1FormalControllerError(f"{block} action pack contract drifted")
    loaded = tuple(_convert_action_item(item) for item in items)
    work_ids = tuple(item.work_id for item in loaded)
    if len(set(work_ids)) != len(work_ids):
        raise MmqaP1FormalControllerError("action pack work IDs are duplicated")
    return value, loaded, binding


def _bundle_top5_rows(
    actions: integration.IntegratedActions,
) -> tuple[tuple[core.ProofBundle, tuple[int, ...]], ...]:
    rows: list[tuple[core.ProofBundle, tuple[int, ...]]] = []
    for bundle in actions.bundles:
        # This is intentionally the integration policy helper rather than a
        # controller reimplementation.  The pre-gold archive therefore binds
        # the exact same bundle-first + cross-encoder-remainder policy used by
        # formal E0/E5 action formation.
        try:
            top5 = integration._bundle_first_top5(  # noqa: SLF001
                actions.core_closure.graph, bundle
            )
        except integration.MmqaP1ActionIntegrationError as exc:
            raise MmqaP1FormalControllerError(
                "bundle-first top-five formation failed"
            ) from exc
        rows.append((bundle, top5))
    if len(rows) != len(actions.bundles) or not rows:
        raise MmqaP1FormalControllerError("bundle-first policy registry drifted")
    return tuple(rows)


def _validate_provider_batch(
    loaded: Sequence[LoadedAction],
    raw: Mapping[
        str, Sequence[integration.UnitCoordinates | Mapping[str, object]]
    ],
) -> dict[str, tuple[integration.UnitCoordinates, ...]]:
    if not isinstance(raw, Mapping):
        raise MmqaP1FormalControllerError("coordinate provider batch is not a mapping")
    expected = {item.work_id for item in loaded}
    if set(raw) != expected:
        raise MmqaP1FormalControllerError("coordinate provider work-ID set drifted")
    by_work = {item.work_id: item.work_item for item in loaded}
    result: dict[str, tuple[integration.UnitCoordinates, ...]] = {}
    for work_id in expected:
        try:
            result[work_id] = integration.validate_unit_coordinates(
                by_work[work_id], raw[work_id]
            )
        except integration.MmqaP1ActionIntegrationError as exc:
            raise MmqaP1FormalControllerError(
                "coordinate provider returned a malformed anonymous vector"
            ) from exc
    return result


def _validate_hippo_batch(
    works: Sequence[WorkExecution],
    raw: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    if not isinstance(raw, Mapping):
        raise MmqaP1FormalControllerError("HippoRAG batch is not a mapping")
    expected = {work.work_id for work in works}
    if set(raw) != expected:
        raise MmqaP1FormalControllerError("HippoRAG work-ID set drifted")
    by_work = {work.work_id: work for work in works}
    result: dict[str, tuple[int, ...]] = {}
    for work_id in expected:
        value = raw[work_id]
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise MmqaP1FormalControllerError("HippoRAG ranking is not an array")
        checked = tuple(value)
        closure = frozenset(by_work[work_id].actions.shared_closure.ordinals)
        if (
            len(checked) != core.TOP_K
            or len(set(checked)) != core.TOP_K
            or any(type(ordinal) is not int for ordinal in checked)
            or not set(checked).issubset(closure)
        ):
            raise MmqaP1FormalControllerError(
                "HippoRAG ranking escaped the candidate-restricted closure"
            )
        result[work_id] = checked
    return result


def _stage_archive_item(work: WorkExecution) -> dict[str, Any]:
    return {
        "work_id": work.work_id,
        "anonymous_projection_sha256": work.work_item.anonymous_projection_sha256,
        "coordinates": [_coordinate_payload(row) for row in work.coordinates],
        "coordinate_vector_sha256": stable_hash(
            [_coordinate_payload(row) for row in work.coordinates]
        ),
        "closure_ordinal_bytes_sha256": work.actions.shared_closure.ordinal_bytes_sha256,
        "action_feature_archive": work.actions.action_feature_archive.payload(),
        "action_feature_archive_sha256": hashlib.sha256(
            work.actions.action_feature_archive.canonical_bytes()
        ).hexdigest(),
        "E0": _ranking_payload(work.actions.e0_ranking),
        "E5": _ranking_payload(work.actions.e5_ranking),
        "RAW": _ranking_payload(work.actions.raw_ranking),
        "sealed_bundle_first_top5": [
            {
                "bundle_ordinals": list(bundle.node_ordinals),
                "top5_ordinals": list(top5),
            }
            for bundle, top5 in work.bundle_first_top5
        ],
        "sealed_bundle_first_top5_sha256": stable_hash(
            [
                {
                    "bundle_ordinals": list(bundle.node_ordinals),
                    "top5_ordinals": list(top5),
                }
                for bundle, top5 in work.bundle_first_top5
            ]
        ),
        "HippoRAG_top5_ordinals": (
            None if work.hipporag_top5 is None else list(work.hipporag_top5)
        ),
        "HippoRAG_payload_binding": (
            None if work.hipporag_binding is None else dict(work.hipporag_binding)
        ),
        "gold_family_type_answer_support_or_source_ID_read_count": 0,
    }


def _materialize_stage(
    *,
    block: str,
    selection_root: Path,
    selection_receipt: Mapping[str, Any],
    control_root: Path,
    coordinate_provider: CoordinateProvider,
    e5_model: core.E5Model | None,
    hippo_executor: HippoExecutor | None,
) -> StageExecution:
    action_pack, loaded, action_binding = _load_action_pack(
        selection_root, selection_receipt, block=block
    )
    provider_items = {row.work_id: row.work_item for row in loaded}
    try:
        raw_coordinates = coordinate_provider(block=block, items=provider_items)
    except Exception as exc:
        raise MmqaP1FormalControllerError(
            f"{block} coordinate provider batch failed"
        ) from exc
    coordinates = _validate_provider_batch(loaded, raw_coordinates)
    works: list[WorkExecution] = []
    for row in loaded:
        try:
            actions = integration.form_actions(
                row.work_item,
                coordinates[row.work_id],
                e5_model=e5_model,
            )
        except (integration.MmqaP1ActionIntegrationError, core.MmqaP1CoreError) as exc:
            raise MmqaP1FormalControllerError(
                f"{block} anonymous action formation failed"
            ) from exc
        if (block == "A_form") != (actions.e5_ranking is None):
            raise MmqaP1FormalControllerError(f"{block} E5 presence drifted")
        works.append(
            WorkExecution(
                work_id=row.work_id,
                work_item=row.work_item,
                coordinates=coordinates[row.work_id],
                actions=actions,
                bundle_first_top5=_bundle_top5_rows(actions),
            )
        )

    hippo_batch_calls = 0
    if block == "A_hold":
        if hippo_executor is None:
            raise MmqaP1FormalControllerError("A_hold requires a HippoExecutor")
        payloads: dict[str, local_executor.CandidateRestrictedHippoRAGPayload] = {}
        for work in works:
            try:
                payloads[work.work_id] = (
                    local_executor.build_candidate_restricted_hipporag_payload(
                        work.actions
                    )
                )
            except local_executor.MmqaP1LocalActionExecutorError as exc:
                raise MmqaP1FormalControllerError(
                    "candidate-restricted HippoRAG payload failed"
                ) from exc
        try:
            raw_hippo = hippo_executor(block=block, payloads=payloads)
        except Exception as exc:
            raise MmqaP1FormalControllerError("A_hold HippoRAG batch failed") from exc
        hippo = _validate_hippo_batch(works, raw_hippo)
        hippo_batch_calls = 1
        works = [
            WorkExecution(
                work_id=work.work_id,
                work_item=work.work_item,
                coordinates=work.coordinates,
                actions=work.actions,
                bundle_first_top5=work.bundle_first_top5,
                hipporag_top5=hippo[work.work_id],
                hipporag_binding=payloads[work.work_id].anonymous_binding(),
            )
            for work in works
        ]
    elif hippo_executor is not None and block not in {"A_form", "F_search", "M_search"}:
        raise MmqaP1FormalControllerError("unexpected HippoRAG stage")

    archive_body = {
        "schema": f"{VERSION}_stage_action_archive_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "selection_acquisition_sha256": selection_receipt["acquisition_sha256"],
        "block": block,
        "status": "all_label_free_actions_complete_and_sealed_before_gold",
        "action_pack_sha256": action_pack["action_pack_sha256"],
        "action_pack_file_sha256": action_binding["file_sha256"],
        "item_count": len(works),
        "coordinate_provider_block_batch_call_count": 1,
        "hipporag_block_batch_call_count": hippo_batch_calls,
        "E5_model_sha256": (
            None if e5_model is None else stable_hash(_model_payload(e5_model))
        ),
        "items": [_stage_archive_item(work) for work in works],
        "gold_open_count_before_archive": 0,
        "online_evaluator_call_count": 0,
        "retry_replay_resample_count": 0,
    }
    archive = self_hashed(archive_body, "archive_sha256")
    path = _stage_root(control_root, block) / STAGE_ACTION_ARCHIVE_FILENAME
    binding = _write_once(path, archive)
    verified = _read_canonical_private(path, label=f"{block} action archive")
    if (
        verified != archive
        or verify_self_hash(verified, "archive_sha256") != archive["archive_sha256"]
    ):
        raise MmqaP1FormalControllerError(f"{block} sealed archive reread drifted")
    return StageExecution(
        block=block,
        action_pack_sha256=str(action_pack["action_pack_sha256"]),
        works=tuple(works),
        archive=archive,
        archive_path=path,
        archive_file_sha256=str(binding["file_sha256"]),
    )


def _open_stage_gold(
    *,
    selection_root: Path,
    control_root: Path,
    stage: StageExecution,
    promotion_sha256: str | None = None,
    promotion_receipt_path: Path | None = None,
    promotion_action_archive_path: Path | None = None,
) -> dict[str, Any]:
    if stage.block == "F_search":
        raise MmqaP1FormalControllerError("F_search has no gold-open path")
    if not stage.archive_path.exists():
        raise MmqaP1FormalControllerError("action archive is absent before gold open")
    authorization_path = (
        _stage_root(control_root, stage.block) / GOLD_AUTHORIZATION_FILENAME
    )
    try:
        authorization = selection.write_block_gold_open_authorization(
            authorization_path,
            output_root=selection_root,
            block=stage.block,
            action_archive_sha256s=(stage.archive_file_sha256,),
            action_archive_paths=(stage.archive_path,),
            promotion_sha256=promotion_sha256,
            promotion_receipt_path=promotion_receipt_path,
            promotion_action_archive_path=promotion_action_archive_path,
        )
        gold = selection.open_block_gold(
            output_root=selection_root,
            block=stage.block,
            authorization_path=authorization_path,
            expected_authorization_sha256=str(
                authorization["authorization_sha256"]
            ),
        )
    except selection.MmqaP1PrivateSelectionError as exc:
        raise MmqaP1FormalControllerError(
            f"{stage.block} archive-bound gold open failed"
        ) from exc
    return gold


def _validated_gold_items(
    gold_pack: Mapping[str, Any],
    stage: StageExecution,
    *,
    selection_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    items = gold_pack.get("items")
    if not isinstance(items, list) or len(items) != len(stage.works):
        raise MmqaP1FormalControllerError("opened gold item count drifted")
    expected_ids = [work.work_id for work in stage.works]
    observed_ids = [row.get("work_id") if isinstance(row, Mapping) else None for row in items]
    if observed_ids != expected_ids:
        raise MmqaP1FormalControllerError("opened gold/action order binding drifted")
    if stage.block == "A_form":
        contract = selection_receipt.get("selection_contract")
        oof = contract.get("A_form_five_fold_OOF") if isinstance(contract, Mapping) else None
        if (
            not isinstance(oof, Mapping)
            or oof.get("fold_count") != 5
            or oof.get("component_atomic") is not True
            or oof.get("secret_HMAC_ordered_deterministic_balancing") is not True
        ):
            raise MmqaP1FormalControllerError("A_form OOF public binding drifted")
        commitment_rows = [
            {"work_id": row["work_id"], "oof_fold": row["oof_fold"]}
            for row in items
            if isinstance(row, Mapping)
        ]
        fold_counts = Counter(row["oof_fold"] for row in items if isinstance(row, Mapping))
        if (
            stable_hash(commitment_rows) != oof.get("assignment_commitment_sha256")
            or {str(key): value for key, value in sorted(fold_counts.items())}
            != oof.get("fold_sizes")
            or set(fold_counts) != set(range(5))
        ):
            raise MmqaP1FormalControllerError("A_form OOF assignment binding drifted")
    elif stage.block in {"A_hold", "M_search"}:
        counts = Counter(
            row.get("evaluation_family")
            for row in items
            if isinstance(row, Mapping)
        )
        if counts != Counter(
            {family: selection.BLOCK_QUOTA_PER_FAMILY[stage.block] for family in FAMILIES}
        ):
            raise MmqaP1FormalControllerError("late evaluation-family quota drifted")
    return tuple(dict(row) for row in items if isinstance(row, Mapping))


def _gold_projection(
    work: WorkExecution, row: Mapping[str, Any]
) -> tuple[tuple[int, ...], tuple[integration.ExactRowTextLink, ...]]:
    gold_rows = row.get("gold_row_ordinals")
    gold_texts = row.get("gold_text_ordinals")
    raw_pairs = row.get("exact_gold_pairs")
    if (
        not isinstance(gold_rows, list)
        or not isinstance(gold_texts, list)
        or not isinstance(raw_pairs, list)
    ):
        raise MmqaP1FormalControllerError("late gold ordinal arrays drifted")
    gold = tuple((*gold_rows, *gold_texts))
    if (
        not gold
        or len(set(gold)) != len(gold)
        or any(type(value) is not int or value < 0 for value in gold)
    ):
        raise MmqaP1FormalControllerError("late gold evidence drifted")
    pairs: list[integration.ExactRowTextLink] = []
    for raw in raw_pairs:
        if not isinstance(raw, Mapping) or set(raw) != {
            "row_ordinal",
            "text_ordinal",
        }:
            raise MmqaP1FormalControllerError("late exact gold pair shape drifted")
        try:
            pairs.append(
                integration.ExactRowTextLink(
                    raw.get("row_ordinal"),  # type: ignore[arg-type]
                    raw.get("text_ordinal"),  # type: ignore[arg-type]
                )
            )
        except integration.MmqaP1ActionIntegrationError as exc:
            raise MmqaP1FormalControllerError("late exact gold pair drifted") from exc
    # score_late_gold validates the original item universe and structural link.
    try:
        integration.score_late_gold(
            work.actions, gold, exact_gold_pairs=tuple(pairs)
        )
    except integration.MmqaP1ActionIntegrationError as exc:
        raise MmqaP1FormalControllerError("late gold projection failed") from exc
    return gold, tuple(pairs)


def _prepare_training_slates(
    stage: StageExecution,
    gold_items: Sequence[Mapping[str, Any]],
) -> tuple[TrainingSlate, ...]:
    if stage.block != "A_form" or len(stage.works) != len(gold_items):
        raise MmqaP1FormalControllerError("A_form training alignment drifted")
    slates: list[TrainingSlate] = []
    for work, row in zip(stage.works, gold_items, strict=True):
        fold = row.get("oof_fold")
        if type(fold) is not int or not 0 <= fold < 5:
            raise MmqaP1FormalControllerError("A_form OOF fold drifted")
        gold, pairs = _gold_projection(work, row)
        utilities = tuple(
            core.integer_binary_evidence_utility(top5, gold)
            for _bundle, top5 in work.bundle_first_top5
        )
        maximum = max(utilities)
        positives = _positive_exact_gold_bundles(
            work.actions.bundles,
            pairs,
        )
        slates.append(
            TrainingSlate(
                work=work,
                fold=fold,
                gold_evidence=gold,
                exact_gold_pairs=pairs,
                positive_exact_gold_bundles=positives,
                maximum_bundle_first_integer_utility_audit_only=(
                    maximum
                ),
                neutral_no_exact_gold_bundle=not positives,
            )
        )
    if Counter(slate.fold for slate in slates).keys() != set(range(5)):
        raise MmqaP1FormalControllerError("A_form OOF folds are incomplete")
    return tuple(slates)


def _training_slate_key(slate: TrainingSlate) -> tuple[object, ...]:
    gold = set(slate.positive_exact_gold_bundles)
    return tuple(
        coordinate.hex()
        for bundle in slate.work.actions.bundles
        for coordinate in core.bundle_feature_vector(
            slate.work.actions.core_closure.graph, bundle
        )
    ) + tuple(int(bundle in gold) for bundle in slate.work.actions.bundles)


def _fit_prepared_slates(slates: Sequence[TrainingSlate]) -> core.E5Model:
    """Fit exact-gold multi-positive targets and retain empty-target slates.

    Every positive is a sealed bundle containing at least one late-opened exact
    gold row-text pair.  Multiple such bundles enter the unchanged log-sum-exp
    marginal likelihood together.  A source-valid item whose frozen closure
    contains no exact-gold bundle contributes no conditional term, while its
    label-free bundle rows remain in the preregistered population scaler.
    """

    if (
        isinstance(slates, (str, bytes))
        or not isinstance(slates, Sequence)
        or not slates
        or not all(isinstance(row, TrainingSlate) for row in slates)
    ):
        raise MmqaP1FormalControllerError("E5 fit requires prepared A_form slates")
    checked = tuple(sorted(slates, key=_training_slate_key))
    raw_slates: list[np.ndarray] = []
    gold_indices: list[np.ndarray] = []
    for slate in checked:
        graph = slate.work.actions.core_closure.graph
        bundles = tuple(slate.work.actions.bundles)
        if (
            not bundles
            or not set(slate.positive_exact_gold_bundles).issubset(
                bundles
            )
            or any(
                not _bundle_contains_exact_gold_pair(
                    bundle, slate.exact_gold_pairs
                )
                for bundle in slate.positive_exact_gold_bundles
            )
        ):
            raise MmqaP1FormalControllerError("prepared bundle slate drifted")
        raw_slates.append(
            np.asarray(
                [core.bundle_feature_vector(graph, bundle) for bundle in bundles],
                dtype=np.float64,
            )
        )
        gold_set = set(slate.positive_exact_gold_bundles)
        gold_indices.append(
            np.asarray(
                [index for index, bundle in enumerate(bundles) if bundle in gold_set],
                dtype=np.int64,
            )
        )
    all_features = np.vstack(raw_slates)
    means = np.mean(all_features, axis=0, dtype=np.float64)
    stds = np.std(all_features, axis=0, ddof=0, dtype=np.float64)
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    standardized = tuple((slate - means) / safe_stds for slate in raw_slates)
    for slate in standardized:
        slate[:, stds == 0.0] = 0.0
    informative = tuple(
        (features, gold)
        for features, gold in zip(
            standardized, gold_indices, strict=True
        )
        if gold.size > 0
    )

    def objective_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            return core._conditional_loss_gradient(  # noqa: SLF001
                beta,
                tuple(features for features, _gold in informative),
                tuple(gold for _features, gold in informative),
            )
        except core.MmqaP1CoreError as exc:
            raise MmqaP1FormalControllerError("E5 objective failed") from exc

    try:
        beta, objective, iterations, converged = core._numpy_lbfgs(  # noqa: SLF001
            objective_gradient,
            len(core.FEATURE_ORDER),
            max_iter=core.E5_MAX_ITER,
        )
        objective, gradient = objective_gradient(beta)
    except core.MmqaP1CoreError as exc:
        raise MmqaP1FormalControllerError("E5 deterministic fit failed") from exc
    if not converged and float(np.max(np.abs(gradient))) > 1.0e-6:
        raise MmqaP1FormalControllerError("E5 deterministic L-BFGS did not converge")
    try:
        return core.E5Model(
            population_mean=tuple(float(value) for value in means),
            population_std=tuple(float(value) for value in stds),
            coefficients=tuple(float(value) for value in beta),
            training_item_count=len(checked),
            training_bundle_count=sum(len(value) for value in raw_slates),
            solver="numpy_deterministic_lbfgs_m10_v1",
            iterations=iterations,
            converged=converged,
            objective=float(objective),
        )
    except core.MmqaP1CoreError as exc:
        raise MmqaP1FormalControllerError("fitted E5 model drifted") from exc


def _verify_reformed_actions(
    sealed: WorkExecution, reformed: integration.IntegratedActions
) -> None:
    if (
        reformed.work_item.anonymous_projection_sha256
        != sealed.work_item.anonymous_projection_sha256
        or reformed.shared_closure.ordinal_bytes_sha256
        != sealed.actions.shared_closure.ordinal_bytes_sha256
        or tuple(reformed.bundles) != tuple(sealed.actions.bundles)
        or reformed.e0_ranking.top5_ordinals
        != sealed.actions.e0_ranking.top5_ordinals
        or reformed.raw_ranking.top5_ordinals
        != sealed.actions.raw_ranking.top5_ordinals
        or reformed.e5_ranking is None
    ):
        raise MmqaP1FormalControllerError(
            "post-gold E5 action escaped the sealed label-free slate"
        )


def _run_oof_audit(
    slates: Sequence[TrainingSlate], *, control_root: Path
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    private_folds: list[dict[str, Any]] = []
    all_e5: list[int] = []
    all_e0: list[int] = []
    safe_folds: dict[str, Any] = {}
    for held_fold in range(5):
        train = tuple(row for row in slates if row.fold != held_fold)
        held = tuple(row for row in slates if row.fold == held_fold)
        if not train or not held:
            raise MmqaP1FormalControllerError("OOF train/held partition is empty")
        model = _fit_prepared_slates(train)
        item_rows: list[dict[str, Any]] = []
        fold_e5: list[int] = []
        fold_e0: list[int] = []
        for slate in held:
            try:
                actions = integration.form_actions(
                    slate.work.work_item,
                    slate.work.coordinates,
                    e5_model=model,
                )
                _verify_reformed_actions(slate.work, actions)
                scores = integration.score_late_gold(
                    actions,
                    slate.gold_evidence,
                    exact_gold_pairs=slate.exact_gold_pairs,
                )
            except (integration.MmqaP1ActionIntegrationError, core.MmqaP1CoreError) as exc:
                raise MmqaP1FormalControllerError("OOF held-fold scoring failed") from exc
            if scores.e5 is None:
                raise MmqaP1FormalControllerError("OOF E5 score is absent")
            fold_e5.append(scores.e5.integer_utility)
            fold_e0.append(scores.e0.integer_utility)
            item_rows.append(
                {
                    "work_id": slate.work.work_id,
                    "fold": held_fold,
                    "E0": _score_payload(scores.e0),
                    "E5": _score_payload(scores.e5),
                    "neutral_no_exact_gold_bundle": (
                        slate.neutral_no_exact_gold_bundle
                    ),
                }
            )
        summary = core.paired_utility_summary(fold_e5, fold_e0)
        model_payload = _model_payload(model)
        private_folds.append(
            {
                "held_fold": held_fold,
                "train_item_count": len(train),
                "held_item_count": len(held),
                "train_exact_positive_slate_count": sum(
                    bool(row.positive_exact_gold_bundles)
                    for row in train
                ),
                "train_no_exact_positive_omitted_conditional_slate_count": sum(
                    row.neutral_no_exact_gold_bundle for row in train
                ),
                "model": model_payload,
                "model_sha256": stable_hash(model_payload),
                "items": item_rows,
                "E5_minus_E0": _paired_payload(summary),
            }
        )
        safe_folds[str(held_fold)] = {
            "train_item_count": len(train),
            "held_item_count": len(held),
            "train_exact_positive_slate_count": sum(
                bool(row.positive_exact_gold_bundles) for row in train
            ),
            "train_no_exact_positive_omitted_conditional_slate_count": sum(
                row.neutral_no_exact_gold_bundle for row in train
            ),
            "model_sha256": stable_hash(model_payload),
            "E5_minus_E0": _paired_payload(summary),
        }
        all_e5.extend(fold_e5)
        all_e0.extend(fold_e0)
    aggregate = core.paired_utility_summary(all_e5, all_e0)
    body = {
        "schema": f"{VERSION}_A_form_five_fold_OOF_audit_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "completed_diagnostic_only_no_tuning_selection_or_gate",
        "fold_count": 5,
        "item_count": len(slates),
        "exact_positive_slate_count": sum(
            bool(row.positive_exact_gold_bundles) for row in slates
        ),
        "no_exact_positive_omitted_conditional_slate_count": sum(
            row.neutral_no_exact_gold_bundle for row in slates
        ),
        "folds": private_folds,
        "aggregate_E5_minus_E0": _paired_payload(aggregate),
        "full_model_affected_by_OOF_outcome": False,
        "family_type_source_ID_or_evaluation_stratum_used_by_model": False,
    }
    value = self_hashed(body, "audit_sha256")
    binding = _write_once(
        _stage_root(control_root, "A_form") / OOF_AUDIT_FILENAME, value
    )
    safe = {
        "status": body["status"],
        "fold_count": 5,
        "item_count": len(slates),
        "exact_positive_slate_count": body[
            "exact_positive_slate_count"
        ],
        "no_exact_positive_omitted_conditional_slate_count": body[
            "no_exact_positive_omitted_conditional_slate_count"
        ],
        "folds": safe_folds,
        "aggregate_E5_minus_E0": _paired_payload(aggregate),
        "audit_sha256": value["audit_sha256"],
        "file_sha256": binding["file_sha256"],
    }
    return safe, {**binding, "semantic_sha256": value["audit_sha256"]}


def _seal_full_model(
    slates: Sequence[TrainingSlate], *, control_root: Path
) -> tuple[core.E5Model, Mapping[str, Any]]:
    model = _fit_prepared_slates(slates)
    payload = _model_payload(model)
    target_commitment = stable_hash(
        [
            {
                "sealed_slate_sha256": stable_hash(
                    [
                        {
                            "bundle_ordinals": list(bundle.node_ordinals),
                            "top5_ordinals": list(top5),
                        }
                        for bundle, top5 in slate.work.bundle_first_top5
                    ]
                ),
                "positive_exact_gold_bundle_ordinals": [
                    list(bundle.node_ordinals)
                    for bundle in slate.positive_exact_gold_bundles
                ],
                "neutral_no_exact_gold_bundle": (
                    slate.neutral_no_exact_gold_bundle
                ),
            }
            for slate in slates
        ]
    )
    body = {
        "schema": f"{VERSION}_full_E5_model_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "single_full_E5_fit_sealed_after_fixed_OOF_audit",
        "model": payload,
        "model_sha256": stable_hash(payload),
        "training_item_count": len(slates),
        "population_scaler_slate_count": len(slates),
        "population_scaler_bundle_count": sum(
            len(row.work.actions.bundles) for row in slates
        ),
        "exact_positive_slate_count": sum(
            bool(row.positive_exact_gold_bundles) for row in slates
        ),
        "no_exact_positive_omitted_conditional_slate_count": sum(
            row.neutral_no_exact_gold_bundle for row in slates
        ),
        "training_target_commitment_sha256": target_commitment,
        "OOF_outcomes_used_for_tuning_selection_rejection_or_model_change": False,
        "family_type_source_ID_or_evaluation_stratum_used_by_model": False,
    }
    value = self_hashed(body, "model_pack_sha256")
    binding = _write_once(control_root / FULL_MODEL_FILENAME, value)
    return model, {
        **binding,
        "semantic_sha256": value["model_pack_sha256"],
        "model_sha256": body["model_sha256"],
    }


def _sealed_gold_open_binding(
    *, control_root: Path, block: str
) -> Mapping[str, Any]:
    """Return the already-consumed authorization binding without touching gold."""

    path = _stage_root(control_root, block) / GOLD_AUTHORIZATION_FILENAME
    value = _read_canonical_private(path, label=f"{block} gold authorization")
    try:
        semantic = selection.verify_self_hash(value, "authorization_sha256")
    except selection.MmqaP1PrivateSelectionError as exc:
        raise MmqaP1FormalControllerError(
            f"{block} gold authorization binding drifted"
        ) from exc
    raw = _canonical_bytes(value)
    return {
        "semantic_sha256": semantic,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode_octal": "0600",
    }


def _score_a_form_targets(
    *,
    stage: StageExecution,
    gold_pack: Mapping[str, Any],
    slates: Sequence[TrainingSlate],
    control_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Seal the complete private training target audit before any model fit."""

    if (
        stage.block != "A_form"
        or len(stage.works) != len(slates)
        or any(slate.work is not work for slate, work in zip(slates, stage.works, strict=True))
    ):
        raise MmqaP1FormalControllerError("A_form target audit alignment drifted")
    item_rows: list[dict[str, Any]] = []
    e0_utilities: list[int] = []
    raw_utilities: list[int] = []
    for slate in slates:
        try:
            scores = integration.score_late_gold(
                slate.work.actions,
                slate.gold_evidence,
                exact_gold_pairs=slate.exact_gold_pairs,
            )
        except integration.MmqaP1ActionIntegrationError as exc:
            raise MmqaP1FormalControllerError("A_form offline scoring failed") from exc
        if scores.e5 is not None or scores.hipporag is not None:
            raise MmqaP1FormalControllerError("A_form acquired a forbidden scored arm")
        bundle_rows = []
        for bundle, top5 in slate.work.bundle_first_top5:
            bundle_rows.append(
                {
                    "bundle_ordinals": list(bundle.node_ordinals),
                    "top5_ordinals": list(top5),
                    "integer_utility": core.integer_binary_evidence_utility(
                        top5, slate.gold_evidence
                    ),
                    "top5_recovers_connected_exact_gold_pair": any(
                        pair.row_ordinal in top5 and pair.text_ordinal in top5
                        for pair in slate.exact_gold_pairs
                    ),
                    "bundle_contains_exact_gold_pair": (
                        _bundle_contains_exact_gold_pair(
                            bundle, slate.exact_gold_pairs
                        )
                    ),
                    "positive_exact_gold_bundle_target": (
                        bundle in slate.positive_exact_gold_bundles
                    ),
                }
            )
        item_rows.append(
            {
                "work_id": slate.work.work_id,
                "oof_fold": slate.fold,
                "late_gold_evidence_ordinals": list(slate.gold_evidence),
                "late_exact_gold_pairs": [
                    {
                        "row_ordinal": pair.row_ordinal,
                        "text_ordinal": pair.text_ordinal,
                    }
                    for pair in slate.exact_gold_pairs
                ],
                "E0": _score_payload(scores.e0),
                "RAW": _score_payload(scores.raw),
                "sealed_bundle_first_candidates": bundle_rows,
                "maximum_bundle_first_integer_utility_audit_only": (
                    slate.maximum_bundle_first_integer_utility_audit_only
                ),
                "positive_exact_gold_bundle_ordinals": [
                    list(bundle.node_ordinals)
                    for bundle in slate.positive_exact_gold_bundles
                ],
                "neutral_no_exact_gold_bundle": (
                    slate.neutral_no_exact_gold_bundle
                ),
            }
        )
        e0_utilities.append(scores.e0.integer_utility)
        raw_utilities.append(scores.raw.integer_utility)
    authorization = _sealed_gold_open_binding(
        control_root=control_root, block="A_form"
    )
    body = {
        "schema": f"{VERSION}_A_form_offline_target_audit_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "late_gold_scored_against_presealed_actions_targets_fixed",
        "block": "A_form",
        "action_archive_sha256": stage.archive["archive_sha256"],
        "action_archive_file_sha256": stage.archive_file_sha256,
        "gold_authorization": dict(authorization),
        "gold_pack_sha256": gold_pack["gold_pack_sha256"],
        "item_count": len(slates),
        "items": item_rows,
        "aggregate_E0_integer_utility": sum(e0_utilities),
        "aggregate_RAW_integer_utility": sum(raw_utilities),
        "population_scaler_slate_count": len(slates),
        "population_scaler_bundle_count": sum(
            len(row.work.actions.bundles) for row in slates
        ),
        "exact_positive_slate_count": sum(
            bool(row.positive_exact_gold_bundles) for row in slates
        ),
        "no_exact_positive_omitted_conditional_slate_count": sum(
            row.neutral_no_exact_gold_bundle for row in slates
        ),
        "target_rule": (
            "all_presealed_bundles_containing_at_least_one_late_exact_gold_"
            "row_text_pair_are_positive_multiple_positives_use_logsumexp_"
            "bundle_first_nDCG_is_audit_only"
        ),
        "post_gold_action_reformation_count": 0,
        "online_evaluator_call_count": 0,
        "tuning_or_gate_count": 0,
    }
    value = self_hashed(body, "score_sha256")
    binding = _write_once(
        _stage_root(control_root, "A_form") / STAGE_SCORE_FILENAME, value
    )
    safe = {
        "status": body["status"],
        "item_count": len(slates),
        "aggregate_E0_integer_utility": body["aggregate_E0_integer_utility"],
        "aggregate_RAW_integer_utility": body["aggregate_RAW_integer_utility"],
        "exact_positive_slate_count": body[
            "exact_positive_slate_count"
        ],
        "no_exact_positive_omitted_conditional_slate_count": body[
            "no_exact_positive_omitted_conditional_slate_count"
        ],
        "score_sha256": value["score_sha256"],
        "file_sha256": binding["file_sha256"],
    }
    return safe, {**binding, "semantic_sha256": value["score_sha256"]}


def _family_delta(
    *,
    families: Sequence[str],
    challenger: Sequence[int],
    incumbent: Sequence[int],
) -> dict[str, int]:
    if (
        len(families) != len(challenger)
        or len(families) != len(incumbent)
        or any(family not in FAMILIES for family in families)
    ):
        raise MmqaP1FormalControllerError("late family utility vectors drifted")
    return {
        family: sum(
            left - right
            for observed, left, right in zip(
                families, challenger, incumbent, strict=True
            )
            if observed == family
        )
        for family in FAMILIES
    }


def _score_a_hold(
    *,
    stage: StageExecution,
    gold_pack: Mapping[str, Any],
    gold_items: Sequence[Mapping[str, Any]],
    control_root: Path,
) -> AHoldOutcome:
    if stage.block != "A_hold" or len(stage.works) != len(gold_items):
        raise MmqaP1FormalControllerError("A_hold score alignment drifted")
    private_items: list[dict[str, Any]] = []
    e0_utilities: list[int] = []
    e5_utilities: list[int] = []
    raw_utilities: list[int] = []
    hippo_utilities: list[int] = []
    families: list[str] = []
    for work, gold_row in zip(stage.works, gold_items, strict=True):
        gold, pairs = _gold_projection(work, gold_row)
        family = gold_row.get("evaluation_family")
        if family not in FAMILIES or work.hipporag_top5 is None:
            raise MmqaP1FormalControllerError("A_hold late scoring stratum drifted")
        try:
            scores = integration.score_late_gold(
                work.actions,
                gold,
                exact_gold_pairs=pairs,
                hipporag_top5_ordinals=work.hipporag_top5,
            )
        except integration.MmqaP1ActionIntegrationError as exc:
            raise MmqaP1FormalControllerError("A_hold offline scoring failed") from exc
        if scores.e5 is None or scores.hipporag is None:
            raise MmqaP1FormalControllerError("A_hold four-arm score is incomplete")
        families.append(str(family))
        e0_utilities.append(scores.e0.integer_utility)
        e5_utilities.append(scores.e5.integer_utility)
        raw_utilities.append(scores.raw.integer_utility)
        hippo_utilities.append(scores.hipporag.integer_utility)
        private_items.append(
            {
                "work_id": work.work_id,
                "evaluation_family": family,
                "late_gold_evidence_ordinals": list(gold),
                "late_exact_gold_pairs": [
                    {
                        "row_ordinal": pair.row_ordinal,
                        "text_ordinal": pair.text_ordinal,
                    }
                    for pair in pairs
                ],
                "E0": _score_payload(scores.e0),
                "E5": _score_payload(scores.e5),
                "RAW": _score_payload(scores.raw),
                "official_candidate_restricted_HippoRAG": _score_payload(
                    scores.hipporag
                ),
            }
        )
    try:
        promotion = core.decide_a_hold_promotion(e5_utilities, e0_utilities)
        e5_vs_raw = core.paired_utility_summary(e5_utilities, raw_utilities)
        e5_vs_hippo = core.paired_utility_summary(e5_utilities, hippo_utilities)
    except core.MmqaP1CoreError as exc:
        raise MmqaP1FormalControllerError("A_hold paired decision failed") from exc
    family_raw = _family_delta(
        families=families, challenger=e5_utilities, incumbent=raw_utilities
    )
    family_hippo = _family_delta(
        families=families, challenger=e5_utilities, incumbent=hippo_utilities
    )
    reality = (
        e5_vs_raw.passed
        and e5_vs_hippo.passed
        and all(value > 0 for value in family_raw.values())
        and all(value > 0 for value in family_hippo.values())
    )
    authorization = _sealed_gold_open_binding(
        control_root=control_root, block="A_hold"
    )
    body = {
        "schema": f"{VERSION}_A_hold_offline_four_arm_score_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": promotion.status,
        "block": "A_hold",
        "action_archive_sha256": stage.archive["archive_sha256"],
        "action_archive_file_sha256": stage.archive_file_sha256,
        "gold_authorization": dict(authorization),
        "gold_pack_sha256": gold_pack["gold_pack_sha256"],
        "item_count": len(private_items),
        "items": private_items,
        "promotion_E5_minus_E0": _paired_payload(promotion.comparison),
        "promoted": promotion.promoted,
        "M_search_authorized": promotion.m_search_authorized,
        "reality_primary": {
            "passed": reality,
            "E5_minus_RAW": _paired_payload(e5_vs_raw),
            "E5_minus_official_candidate_restricted_HippoRAG": _paired_payload(
                e5_vs_hippo
            ),
            "family_delta_E5_minus_RAW": family_raw,
            "family_delta_E5_minus_official_candidate_restricted_HippoRAG": (
                family_hippo
            ),
            "rule": (
                "both_aggregate_positive_and_exact_tail_at_most_0_10_and_"
                "every_family_delta_strictly_positive"
            ),
            "used_as_promotion_gate": False,
        },
        "late_family_used_for_offline_stratified_scoring_only": True,
        "post_gold_action_reformation_count": 0,
        "online_evaluator_call_count": 0,
        "retry_replay_resample_count": 0,
    }
    value = self_hashed(body, "score_sha256")
    binding = _write_once(
        _stage_root(control_root, "A_hold") / STAGE_SCORE_FILENAME, value
    )
    return AHoldOutcome(
        promotion=promotion,
        promotion_sha256=str(value["score_sha256"]),
        reality_primary_passed=reality,
        e5_vs_raw=e5_vs_raw,
        e5_vs_hipporag=e5_vs_hippo,
        family_delta_e5_minus_raw=family_raw,
        family_delta_e5_minus_hipporag=family_hippo,
        score_binding={
            **binding,
            "semantic_sha256": value["score_sha256"],
        },
    )


def _score_m_search(
    *,
    stage: StageExecution,
    gold_pack: Mapping[str, Any],
    gold_items: Sequence[Mapping[str, Any]],
    promotion: core.PromotionDecision,
    promotion_sha256: str,
    control_root: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if (
        stage.block != "M_search"
        or not promotion.promoted
        or len(stage.works) != len(gold_items)
        or _HEX64.fullmatch(promotion_sha256) is None
    ):
        raise MmqaP1FormalControllerError("M_search score authorization drifted")
    private_items: list[dict[str, Any]] = []
    e0_utilities: list[int] = []
    e5_utilities: list[int] = []
    families: list[str] = []
    for work, gold_row in zip(stage.works, gold_items, strict=True):
        gold, pairs = _gold_projection(work, gold_row)
        family = gold_row.get("evaluation_family")
        if family not in FAMILIES:
            raise MmqaP1FormalControllerError("M_search late family drifted")
        try:
            scores = integration.score_late_gold(
                work.actions, gold, exact_gold_pairs=pairs
            )
        except integration.MmqaP1ActionIntegrationError as exc:
            raise MmqaP1FormalControllerError("M_search offline scoring failed") from exc
        if scores.e5 is None or scores.hipporag is not None:
            raise MmqaP1FormalControllerError("M_search scored-arm set drifted")
        families.append(str(family))
        e0_utilities.append(scores.e0.integer_utility)
        e5_utilities.append(scores.e5.integer_utility)
        private_items.append(
            {
                "work_id": work.work_id,
                "evaluation_family": family,
                "late_gold_evidence_ordinals": list(gold),
                "late_exact_gold_pairs": [
                    {
                        "row_ordinal": pair.row_ordinal,
                        "text_ordinal": pair.text_ordinal,
                    }
                    for pair in pairs
                ],
                "E0": _score_payload(scores.e0),
                "E5": _score_payload(scores.e5),
            }
        )
    try:
        core_decision = core.decide_m_search(
            promotion, e5_utilities, e0_utilities
        )
    except core.MmqaP1CoreError as exc:
        raise MmqaP1FormalControllerError("M_search paired decision failed") from exc
    if core_decision.comparison is None:
        raise MmqaP1FormalControllerError("authorized M_search lacks comparison")
    family_delta = _family_delta(
        families=families, challenger=e5_utilities, incumbent=e0_utilities
    )
    family_nonnegative = all(value >= 0 for value in family_delta.values())
    positive_family_count = sum(value > 0 for value in family_delta.values())
    l5_passed = (
        core_decision.comparison.passed
        and family_nonnegative
        and positive_family_count >= 2
    )
    authorization = _sealed_gold_open_binding(
        control_root=control_root, block="M_search"
    )
    body = {
        "schema": f"{VERSION}_M_search_offline_L5_score_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": (
            "valid_L5_improvement_with_family_retention"
            if l5_passed
            else "valid_no_L5_improvement_with_family_retention"
        ),
        "block": "M_search",
        "action_archive_sha256": stage.archive["archive_sha256"],
        "action_archive_file_sha256": stage.archive_file_sha256,
        "gold_authorization": dict(authorization),
        "A_hold_promotion_sha256": promotion_sha256,
        "gold_pack_sha256": gold_pack["gold_pack_sha256"],
        "item_count": len(private_items),
        "items": private_items,
        "aggregate_E5_minus_E0": _paired_payload(core_decision.comparison),
        "family_delta_E5_minus_E0": family_delta,
        "all_family_deltas_nonnegative": family_nonnegative,
        "strictly_positive_family_count": positive_family_count,
        "L5_passed": l5_passed,
        "rule": (
            "aggregate_positive_and_exact_tail_at_most_0_10_all_family_"
            "deltas_nonnegative_at_least_two_strictly_positive"
        ),
        "late_family_used_for_offline_stratified_scoring_only": True,
        "post_gold_action_reformation_count": 0,
        "online_evaluator_call_count": 0,
        "retry_replay_resample_count": 0,
    }
    value = self_hashed(body, "score_sha256")
    binding = _write_once(
        _stage_root(control_root, "M_search") / STAGE_SCORE_FILENAME, value
    )
    safe = {
        "status": body["status"],
        "item_count": len(private_items),
        "aggregate_E5_minus_E0": body["aggregate_E5_minus_E0"],
        "family_delta_E5_minus_E0": family_delta,
        "all_family_deltas_nonnegative": family_nonnegative,
        "strictly_positive_family_count": positive_family_count,
        "L5_passed": l5_passed,
        "score_sha256": value["score_sha256"],
        "file_sha256": binding["file_sha256"],
    }
    return safe, {**binding, "semantic_sha256": value["score_sha256"]}


def _f_search_behavior(stage: StageExecution) -> dict[str, Any]:
    if stage.block != "F_search":
        raise MmqaP1FormalControllerError("F_search behavior stage drifted")
    same = 0
    for work in stage.works:
        if work.actions.e5_ranking is None:
            raise MmqaP1FormalControllerError("F_search E5 trace is absent")
        same += (
            work.actions.e5_ranking.top5_ordinals
            == work.actions.e0_ranking.top5_ordinals
        )
    return {
        "status": "label_free_behavior_trace_complete_never_scored_never_gate",
        "item_count": len(stage.works),
        "E5_equal_E0_top5_count": same,
        "E5_changed_from_E0_top5_count": len(stage.works) - same,
        "action_archive_sha256": stage.archive["archive_sha256"],
        "action_archive_file_sha256": stage.archive_file_sha256,
        "gold_opened": False,
        "used_as_gate": False,
    }


def _assert_no_gold_marker(selection_root: Path, block: str) -> None:
    path = Path(selection_root) / selection.GOLD_OPEN_MARKER_FILENAMES[block]
    if path.exists() or path.is_symlink():
        raise MmqaP1FormalControllerError(
            f"{block} gold marker exists outside the current lifecycle"
        )


def _assert_safe_receipt(value: Mapping[str, Any]) -> None:
    try:
        selection._assert_public_safe(value)  # noqa: SLF001
    except selection.MmqaP1PrivateSelectionError as exc:
        raise MmqaP1FormalControllerError("final safe receipt leaked item data") from exc


def _terminal_failure(
    *,
    control_root: Path,
    phase: str,
    exc: BaseException,
) -> None:
    path = control_root / FAILURE_FILENAME
    if path.exists() or path.is_symlink():
        return
    message = f"{type(exc).__module__}.{type(exc).__qualname__}:{exc}"
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "terminal_consumed_no_retry_replay_or_resample",
        "failed_phase": phase,
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "exception_message_sha256": hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest(),
        "retry_replay_resample_authorized": False,
        "online_evaluator_call_count": 0,
    }
    _write_once(path, self_hashed(body, "failure_sha256"))


class FormalController:
    """Execute one generic source-blind lifecycle against injected runtimes."""

    def __init__(
        self,
        *,
        selection_root: str | Path,
        control_root: str | Path,
        expected_selection_acquisition_sha256: str,
        coordinate_provider: CoordinateProvider,
        hippo_executor: HippoExecutor,
    ) -> None:
        self.selection_root = Path(selection_root)
        self.control_root = Path(control_root)
        self.expected_selection_acquisition_sha256 = (
            expected_selection_acquisition_sha256
        )
        self.coordinate_provider = coordinate_provider
        self.hippo_executor = hippo_executor

    def run(self) -> dict[str, Any]:
        """Consume exactly one lifecycle root and return its hash-safe receipt."""

        phase = "create_one_shot_controller_root"
        root_created = False
        try:
            _create_one_shot_root(self.control_root)
            root_created = True
            phase = "seal_lifecycle_attempt"
            attempt = self_hashed(
                {
                    "schema": f"{VERSION}_one_shot_attempt_v1",
                    "version": VERSION,
                    "study_id": STUDY_ID,
                    "status": "one_shot_lifecycle_consumed_before_any_stage",
                    "expected_selection_acquisition_sha256": (
                        self.expected_selection_acquisition_sha256
                    ),
                    "block_order": list(BLOCK_ORDER),
                    "retry_replay_resample_authorized": False,
                    "online_evaluator_authorized": False,
                    "trusted_source_ledger_read_authorized": False,
                },
                "attempt_sha256",
            )
            attempt_binding = _write_once(
                self.control_root / ATTEMPT_MARKER_FILENAME, attempt
            )

            phase = "validate_fresh_selection_and_public_receipt"
            for block in BLOCK_ORDER:
                _assert_no_gold_marker(self.selection_root, block)
            selection_receipt = _load_selection_receipt(
                self.selection_root,
                self.expected_selection_acquisition_sha256,
            )

            phase = "A_form_seal_all_label_free_actions"
            a_form = _materialize_stage(
                block="A_form",
                selection_root=self.selection_root,
                selection_receipt=selection_receipt,
                control_root=self.control_root,
                coordinate_provider=self.coordinate_provider,
                e5_model=None,
                hippo_executor=None,
            )
            phase = "A_form_archive_bound_gold_open"
            a_form_gold = _open_stage_gold(
                selection_root=self.selection_root,
                control_root=self.control_root,
                stage=a_form,
            )
            phase = "A_form_fix_targets_and_offline_score"
            a_form_gold_items = _validated_gold_items(
                a_form_gold, a_form, selection_receipt=selection_receipt
            )
            slates = _prepare_training_slates(a_form, a_form_gold_items)
            a_form_safe, a_form_score_binding = _score_a_form_targets(
                stage=a_form,
                gold_pack=a_form_gold,
                slates=slates,
                control_root=self.control_root,
            )

            phase = "A_form_fixed_five_fold_OOF_audit"
            oof_safe, oof_binding = _run_oof_audit(
                slates, control_root=self.control_root
            )
            phase = "seal_one_full_E5_model"
            e5_model, full_model_binding = _seal_full_model(
                slates, control_root=self.control_root
            )

            phase = "F_search_seal_label_free_behavior_trace"
            f_search = _materialize_stage(
                block="F_search",
                selection_root=self.selection_root,
                selection_receipt=selection_receipt,
                control_root=self.control_root,
                coordinate_provider=self.coordinate_provider,
                e5_model=e5_model,
                hippo_executor=None,
            )
            f_behavior = _f_search_behavior(f_search)
            _assert_no_gold_marker(self.selection_root, "F_search")

            phase = "A_hold_seal_all_four_arm_actions"
            a_hold = _materialize_stage(
                block="A_hold",
                selection_root=self.selection_root,
                selection_receipt=selection_receipt,
                control_root=self.control_root,
                coordinate_provider=self.coordinate_provider,
                e5_model=e5_model,
                hippo_executor=self.hippo_executor,
            )
            phase = "A_hold_archive_bound_gold_open"
            a_hold_gold = _open_stage_gold(
                selection_root=self.selection_root,
                control_root=self.control_root,
                stage=a_hold,
            )
            phase = "A_hold_offline_promotion_and_reality_score"
            a_hold_gold_items = _validated_gold_items(
                a_hold_gold, a_hold, selection_receipt=selection_receipt
            )
            a_hold_outcome = _score_a_hold(
                stage=a_hold,
                gold_pack=a_hold_gold,
                gold_items=a_hold_gold_items,
                control_root=self.control_root,
            )

            m_stage_safe: dict[str, Any]
            m_score_binding: Mapping[str, Any] | None
            if a_hold_outcome.promotion.promoted:
                phase = "M_search_seal_E0_E5_actions_after_promotion"
                m_search = _materialize_stage(
                    block="M_search",
                    selection_root=self.selection_root,
                    selection_receipt=selection_receipt,
                    control_root=self.control_root,
                    coordinate_provider=self.coordinate_provider,
                    e5_model=e5_model,
                    hippo_executor=None,
                )
                phase = "M_search_archive_and_promotion_bound_gold_open"
                m_gold = _open_stage_gold(
                    selection_root=self.selection_root,
                    control_root=self.control_root,
                    stage=m_search,
                    promotion_sha256=a_hold_outcome.promotion_sha256,
                    promotion_receipt_path=(
                        _stage_root(self.control_root, "A_hold")
                        / STAGE_SCORE_FILENAME
                    ),
                    promotion_action_archive_path=a_hold.archive_path,
                )
                phase = "M_search_offline_L5_score"
                m_gold_items = _validated_gold_items(
                    m_gold, m_search, selection_receipt=selection_receipt
                )
                m_stage_safe, m_score_binding = _score_m_search(
                    stage=m_search,
                    gold_pack=m_gold,
                    gold_items=m_gold_items,
                    promotion=a_hold_outcome.promotion,
                    promotion_sha256=a_hold_outcome.promotion_sha256,
                    control_root=self.control_root,
                )
                m_stage_safe = {
                    **m_stage_safe,
                    "authorized": True,
                    "gold_opened": True,
                    "action_archive_sha256": m_search.archive["archive_sha256"],
                    "action_archive_file_sha256": m_search.archive_file_sha256,
                }
            else:
                phase = "verify_M_search_remains_permanently_sealed"
                _assert_no_gold_marker(self.selection_root, "M_search")
                m_score_binding = None
                m_stage_safe = {
                    "status": "sealed_after_A_hold_valid_nonpromotion",
                    "authorized": False,
                    "gold_opened": False,
                    "action_archive_created": False,
                    "L5_passed": False,
                }

            phase = "seal_hash_safe_final_receipt"
            _assert_no_gold_marker(self.selection_root, "F_search")
            a_hold_safe = {
                "status": a_hold_outcome.promotion.status,
                "item_count": len(a_hold.works),
                "promotion_E5_minus_E0": _paired_payload(
                    a_hold_outcome.promotion.comparison
                ),
                "promoted": a_hold_outcome.promotion.promoted,
                "M_search_authorized": (
                    a_hold_outcome.promotion.m_search_authorized
                ),
                "reality_primary_passed": (
                    a_hold_outcome.reality_primary_passed
                ),
                "E5_minus_RAW": _paired_payload(a_hold_outcome.e5_vs_raw),
                "E5_minus_official_candidate_restricted_HippoRAG": (
                    _paired_payload(a_hold_outcome.e5_vs_hipporag)
                ),
                "family_delta_E5_minus_RAW": dict(
                    a_hold_outcome.family_delta_e5_minus_raw
                ),
                "family_delta_E5_minus_official_candidate_restricted_HippoRAG": (
                    dict(a_hold_outcome.family_delta_e5_minus_hipporag)
                ),
                "action_archive_sha256": a_hold.archive["archive_sha256"],
                "action_archive_file_sha256": a_hold.archive_file_sha256,
                "score_sha256": a_hold_outcome.score_binding[
                    "semantic_sha256"
                ],
                "score_file_sha256": a_hold_outcome.score_binding[
                    "file_sha256"
                ],
            }
            final_body = {
                "schema": f"{VERSION}_hash_safe_final_receipt_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": (
                    "lifecycle_complete_promoted_M_scored"
                    if a_hold_outcome.promotion.promoted
                    else "lifecycle_complete_valid_nonpromotion_M_sealed"
                ),
                "selection_acquisition_sha256": selection_receipt[
                    "acquisition_sha256"
                ],
                "attempt": {
                    "semantic_sha256": attempt["attempt_sha256"],
                    **attempt_binding,
                },
                "A_form": {
                    **a_form_safe,
                    "action_archive_sha256": a_form.archive["archive_sha256"],
                    "action_archive_file_sha256": a_form.archive_file_sha256,
                    "score_file_sha256": a_form_score_binding["file_sha256"],
                    "OOF_audit": oof_safe,
                    "OOF_audit_file_sha256": oof_binding["file_sha256"],
                    "full_E5_model_sha256": full_model_binding["model_sha256"],
                    "full_E5_model_pack_sha256": full_model_binding[
                        "semantic_sha256"
                    ],
                    "full_E5_model_file_sha256": full_model_binding[
                        "file_sha256"
                    ],
                },
                "F_search": f_behavior,
                "A_hold": a_hold_safe,
                "M_search": m_stage_safe,
                "invariants": {
                    "trusted_source_ledger_read_count": 0,
                    "formal_source_read_count": 0,
                    "online_evaluator_call_count": 0,
                    "retry_replay_resample_count": 0,
                    "F_search_gold_open_count": 0,
                    "F_search_used_as_gate": False,
                    "OOF_used_for_tuning_selection_or_gate": False,
                    "late_family_used_by_E5_model": False,
                    "M_search_opened_only_after_A_hold_promotion": True,
                    "all_controller_outputs_mode_0600": True,
                },
            }
            if m_score_binding is not None:
                final_body["M_search"]["score_file_sha256"] = m_score_binding[
                    "file_sha256"
                ]
            final = self_hashed(final_body, "final_sha256")
            _assert_safe_receipt(final)
            _write_once(self.control_root / FINAL_RECEIPT_FILENAME, final)
            return final
        except Exception as exc:
            if root_created:
                try:
                    _terminal_failure(
                        control_root=self.control_root, phase=phase, exc=exc
                    )
                except Exception:
                    # The root is already consumed even if a damaged filesystem
                    # prevents the secondary terminal receipt from being sealed.
                    pass
            if isinstance(exc, MmqaP1FormalControllerError):
                raise
            raise MmqaP1FormalControllerError(
                f"one-shot formal lifecycle failed during {phase}"
            ) from exc


def run_lifecycle(
    *,
    selection_root: str | Path,
    control_root: str | Path,
    expected_selection_acquisition_sha256: str,
    coordinate_provider: CoordinateProvider,
    hippo_executor: HippoExecutor,
) -> dict[str, Any]:
    """Convenience entry point for one generic injected-runtime lifecycle."""

    return FormalController(
        selection_root=selection_root,
        control_root=control_root,
        expected_selection_acquisition_sha256=(
            expected_selection_acquisition_sha256
        ),
        coordinate_provider=coordinate_provider,
        hippo_executor=hippo_executor,
    ).run()


__all__ = [
    "VERSION",
    "STUDY_ID",
    "ATTEMPT_MARKER_FILENAME",
    "FINAL_RECEIPT_FILENAME",
    "FAILURE_FILENAME",
    "FULL_MODEL_FILENAME",
    "STAGE_ACTION_ARCHIVE_FILENAME",
    "STAGE_SCORE_FILENAME",
    "GOLD_AUTHORIZATION_FILENAME",
    "OOF_AUDIT_FILENAME",
    "MmqaP1FormalControllerError",
    "CoordinateProvider",
    "HippoExecutor",
    "FormalController",
    "run_lifecycle",
    "stable_hash",
    "self_hashed",
    "verify_self_hash",
]

"""Executable, tamper-evident evidence store for the frozen HoVer lifecycle.

This module writes evidence; it does not score outcomes or introduce another
acceptance gate.  Every writer is O_EXCL/fsync, every loader revalidates the
current committed acquisition binding and all derived typed receipts, and the
only late-outcome input is the controller's exact A_hold report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import hover_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
    ACTION_IDS,
    INTEGER_SCALE,
    ActionTrace,
    CausalSignature,
    CoverageSignature,
    EvaluationObservation,
    MultiHopRAGTypedOperatorV2Error,
    PolicySelection,
    VERSION as TYPED_CORE_VERSION,
    item_utility,
    normalize_text,
    paired_utility_summary,
    policies_identifiable,
    recompute_action_trace_sha256,
    recompute_policy_selection_sha256,
    select_global_policy,
)


VERSION = "hover_lifecycle_store_v1"
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_COUNTS = {"A_form": 48, "F_search": 36, "A_hold": 30, "M_search": 30}
CORPUS_SIZE = 609
AGENT_ACTION_IDS = tuple(ACTION_IDS)

PRIVATE_ROOT_RELATIVE = Path("artifacts/hover_joint_graph_formal_v1")
STAGE_OUTPUT_ARCHIVE_RELATIVES = {
    block: PRIVATE_ROOT_RELATIVE / f"{block}.stage_output.private.json"
    for block in BLOCK_ORDER
}
ACTION_SEAL_RELATIVES = {
    block: Path("manifests") / f"hover_{block.casefold()}_action_seal_v1.json"
    for block in ("A_form", "A_hold", "M_search")
}
A_FORM_EVALUATOR_FREEZE_RELATIVE = Path(
    "manifests/hover_a_form_evaluator_freeze_v1.json"
)
F_POLICY_FREEZE_RELATIVE = Path("manifests/hover_f_search_policy_freeze_v1.json")
PROMOTION_RELATIVE = Path("manifests/hover_a_hold_promotion_v1.json")

STAGE_RUNTIME_BINDING_KEYS = (
    "preparation_sha256",
    "graph_sha256",
    "embedding_index_sha256",
    "ner_runtime_receipt_sha256",
    "ner_entity_matrix_sha256",
    "hippo_build_receipt_sha256",
    "hippo_retrieval_receipt_sha256",
    "execution_matrix_sha256",
)
EVALUATOR_DEFINITIONS = {
    "E0_INDEPENDENT_V2": (
        "independent_metadata_coverage_then_dense_relevance_then_redundancy"
    ),
    "E1_CAUSAL_NECESSITY_V2": (
        "leave_one_out_necessity_then_typed_path_connectivity_then_E0"
    ),
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SYNTHETIC_SENTINEL = ".hover_lifecycle_store_synthetic_test_root"


class HoVerLifecycleStoreError(RuntimeError):
    """A persistence, ordering, acquisition, or receipt invariant drifted."""


@dataclass(frozen=True)
class StageAcquisitionContext:
    """The only acquisition information the evidence store is allowed to use."""

    acquisition_sha256: str
    acquisition_file_sha256: str
    acquisition_git_head: str
    acquisition_git_blob_sha1: str
    corpus_file_sha256: str
    corpus_semantic_sha256: str
    block: str
    view_file_sha256: str
    view_semantic_sha256: str
    view_items: tuple[Mapping[str, Any], ...]
    f_search_label_pack_created: bool
    # Synthetic sentinel tests may carry an already committed label payload
    # through the single acquisition seam.  Production always leaves this
    # unset and performs the late open only after the A_hold action seal.
    late_labels: Mapping[str, Any] | None = None


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
        raise HoVerLifecycleStoreError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HoVerLifecycleStoreError(f"{field} is not a SHA256")
    return value


def _require_sha1(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise HoVerLifecycleStoreError(f"{field} is not a Git SHA1")
    return value


def _with_self_hash(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HoVerLifecycleStoreError("self-hash field already exists")
    return {**dict(body), field: stable_hash(dict(body))}


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, schema: str
) -> str:
    declared = _require_sha256(payload.get(field), field)
    body = dict(payload)
    del body[field]
    if payload.get("schema") != schema or not hmac.compare_digest(
        stable_hash(body), declared
    ):
        raise HoVerLifecycleStoreError(f"{field} self-hash mismatch")
    return declared


def _strict_json(raw: bytes, field: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise HoVerLifecycleStoreError(f"{field} has duplicate JSON keys")
            result[key] = value
        return result

    def reject_float(value: str) -> None:
        raise HoVerLifecycleStoreError(f"{field} contains a non-integer number: {value}")

    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_float=reject_float,
            parse_constant=reject_float,
        )
    except HoVerLifecycleStoreError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoVerLifecycleStoreError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise HoVerLifecycleStoreError(f"{field} JSON root is not an object")
    return value


def _project_root(project: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(project)))
    try:
        info = root.lstat()
    except OSError as exc:
        raise HoVerLifecycleStoreError("project root is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise HoVerLifecycleStoreError("project root must be a real directory")
    return root


def _ensure_parent(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise HoVerLifecycleStoreError("output escaped the project root") from exc
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            try:
                cursor.mkdir(mode=0o700)
            except OSError as exc:
                raise HoVerLifecycleStoreError("output parent creation failed") from exc
            info = cursor.lstat()
        except OSError as exc:
            raise HoVerLifecycleStoreError("output parent is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HoVerLifecycleStoreError("output parent is not a real directory")


def _reject_unsafe_existing_ancestors(root: Path, path: Path) -> None:
    """Reject a pre-existing symlink/non-directory without creating anything."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise HoVerLifecycleStoreError("lifecycle path escaped the project root") from exc
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise HoVerLifecycleStoreError("lifecycle path ancestor is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HoVerLifecycleStoreError("lifecycle path ancestor is unsafe")


def _write_json_exclusive(
    *, root: Path, relative: Path, payload: Mapping[str, Any], mode: int
) -> None:
    if mode not in {0o600, 0o644}:
        raise HoVerLifecycleStoreError("output mode is invalid")
    path = root / relative
    _ensure_parent(root, path)
    raw = _canonical_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise HoVerLifecycleStoreError(
            f"exclusive lifecycle output already exists or is unsafe: {relative}"
        ) from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_json_file(
    *, root: Path, relative: Path, field: str, mode: int
) -> tuple[dict[str, Any], bytes]:
    path = root / relative
    _reject_unsafe_existing_ancestors(root, path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise HoVerLifecycleStoreError(f"{field} is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != mode
    ):
        raise HoVerLifecycleStoreError(f"{field} file type or mode drifted")
    raw = path.read_bytes()
    payload = _strict_json(raw, field)
    if raw != _canonical_bytes(payload) + b"\n":
        raise HoVerLifecycleStoreError(f"{field} bytes are not canonical")
    return payload, raw


def lifecycle_output_paths(project: Path) -> tuple[Path, ...]:
    """List every path this module may create, without touching the filesystem."""

    root = _project_root(project)
    relatives = (
        *(STAGE_OUTPUT_ARCHIVE_RELATIVES[block] for block in BLOCK_ORDER),
        *(ACTION_SEAL_RELATIVES[block] for block in ("A_form", "A_hold", "M_search")),
        A_FORM_EVALUATOR_FREEZE_RELATIVE,
        F_POLICY_FREEZE_RELATIVE,
        PROMOTION_RELATIVE,
    )
    return tuple(root / relative for relative in relatives)


def preflight_lifecycle_outputs_absent(project: Path) -> tuple[Path, ...]:
    """Read-only preflight: return the complete path set or reject any occupant."""

    root = _project_root(project)
    paths = lifecycle_output_paths(root)
    for path in paths:
        _reject_unsafe_existing_ancestors(root, path)
    occupied = [path.relative_to(root).as_posix() for path in paths if os.path.lexists(path)]
    if occupied:
        raise HoVerLifecycleStoreError(
            "lifecycle output preflight found existing paths: " + ", ".join(occupied)
        )
    return paths


def _load_stage_acquisition_context(
    *, project: Path, block: str
) -> StageAcquisitionContext:
    """Load and rebind one committed acquisition stage (the sole test seam)."""

    if block not in BLOCK_ORDER:
        raise HoVerLifecycleStoreError("acquisition block is invalid")
    receipt, binding = acquisition.load_formal_committed_acquisition_receipt(project)
    corpus = binding.get("corpus")
    blocks = binding.get("blocks")
    if not isinstance(corpus, Mapping) or not isinstance(blocks, Mapping):
        raise HoVerLifecycleStoreError("acquisition commitment binding drifted")
    row = blocks.get(block)
    if not isinstance(row, Mapping) or not isinstance(row.get("view"), Mapping):
        raise HoVerLifecycleStoreError("acquisition block binding drifted")
    view_binding = row["view"]
    corpus_view = acquisition.load_corpus_view(project=project)
    view = acquisition.load_block_view(project=project, expected_block=block)
    if (
        corpus_view.get("corpus_view_sha256") != corpus.get("semantic_sha256")
        or view.get("block_view_sha256") != view_binding.get("semantic_sha256")
        or view.get("item_count") != BLOCK_COUNTS[block]
        or not isinstance(view.get("items"), list)
    ):
        raise HoVerLifecycleStoreError("loaded acquisition view binding drifted")
    f_row = blocks.get("F_search")
    if not isinstance(f_row, Mapping) or not isinstance(f_row.get("labels"), Mapping):
        raise HoVerLifecycleStoreError("F_search acquisition label binding drifted")
    return StageAcquisitionContext(
        acquisition_sha256=_require_sha256(
            receipt.get("acquisition_sha256"), "acquisition receipt"
        ),
        acquisition_file_sha256=_require_sha256(
            binding.get("receipt_file_sha256"), "acquisition receipt file"
        ),
        acquisition_git_head=_require_sha1(
            binding.get("receipt_git_head"), "acquisition Git HEAD"
        ),
        acquisition_git_blob_sha1=_require_sha1(
            binding.get("receipt_git_blob_sha1"), "acquisition Git blob"
        ),
        corpus_file_sha256=_require_sha256(
            corpus.get("file_sha256"), "corpus view file"
        ),
        corpus_semantic_sha256=_require_sha256(
            corpus.get("semantic_sha256"), "corpus view semantic"
        ),
        block=block,
        view_file_sha256=_require_sha256(
            view_binding.get("file_sha256"), "block view file"
        ),
        view_semantic_sha256=_require_sha256(
            view_binding.get("semantic_sha256"), "block view semantic"
        ),
        view_items=tuple(dict(item) for item in view["items"]),
        f_search_label_pack_created=f_row["labels"].get("created") is True,
    )


def _validate_context(context: StageAcquisitionContext, *, block: str) -> None:
    if not isinstance(context, StageAcquisitionContext) or context.block != block:
        raise HoVerLifecycleStoreError("stage acquisition context identity drifted")
    for field, value in (
        ("acquisition", context.acquisition_sha256),
        ("acquisition file", context.acquisition_file_sha256),
        ("corpus file", context.corpus_file_sha256),
        ("corpus semantic", context.corpus_semantic_sha256),
        ("view file", context.view_file_sha256),
        ("view semantic", context.view_semantic_sha256),
    ):
        _require_sha256(value, field)
    _require_sha1(context.acquisition_git_head, "acquisition Git HEAD")
    _require_sha1(context.acquisition_git_blob_sha1, "acquisition Git blob")
    if (
        len(context.view_items) != BLOCK_COUNTS[block]
        or context.f_search_label_pack_created
    ):
        raise HoVerLifecycleStoreError("stage acquisition cardinality/label drifted")
    for ordinal, item in enumerate(context.view_items):
        if (
            not isinstance(item, Mapping)
            or item.get("block") != block
            or item.get("ordinal") != ordinal
            or not isinstance(item.get("claim"), str)
            or not item["claim"].strip()
        ):
            raise HoVerLifecycleStoreError("stage claim view row drifted")


def _number(value: Fraction | int) -> int | list[int]:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if type(value) is not int:
        raise HoVerLifecycleStoreError("typed numeric value drifted")
    return value


def _decode_number(value: object, field: str) -> Fraction | int:
    if type(value) is int:
        return value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(type(part) is int for part in value)
        and value[1] > 0
    ):
        result = Fraction(value[0], value[1])
        if value != [result.numerator, result.denominator]:
            raise HoVerLifecycleStoreError(f"{field} fraction is noncanonical")
        return result
    raise HoVerLifecycleStoreError(f"{field} typed number drifted")


def _top5(value: object, field: str) -> tuple[int, int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 5
        or any(type(index) is not int or not 0 <= index < CORPUS_SIZE for index in value)
        or len(set(value)) != 5
    ):
        raise HoVerLifecycleStoreError(f"{field} top5 drifted")
    return tuple(value)  # type: ignore[return-value]


def _dense_relevance(value: object) -> tuple[tuple[int, ...], str]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != CORPUS_SIZE
        or any(
            type(score) is not int or not -INTEGER_SCALE <= score <= INTEGER_SCALE
            for score in value
        )
    ):
        raise HoVerLifecycleStoreError("dense relevance vector drifted")
    rows = tuple(value)
    return rows, stable_hash({"integer_scale": INTEGER_SCALE, "values": list(rows)})


def _trace_body(trace: ActionTrace) -> dict[str, Any]:
    return {
        "action_id": trace.action_id,
        "causal": [
            trace.causal.necessary_count,
            _number(trace.causal.necessary_fraction),
            _number(trace.causal.minimum_leave_one_out_loss),
            _number(trace.causal.minimum_replacement_loss),
            _number(trace.causal.path_connectivity),
        ],
        "core": list(trace.core),
        "core_quality": [_number(row) for row in trace.core_quality],
        "coverage": [trace.coverage.covered, trace.coverage.total],
        "coverage_slot_keys": list(trace.coverage.slot_keys),
        "coverage_covered_slot_keys": list(trace.coverage.covered_slot_keys),
        "e0": [_number(row) for row in trace.e0_key],
        "e1": [_number(row) for row in trace.e1_key],
        "extension_scan_count": trace.extension_scan_count,
        "graph_sha256": trace.graph_sha256,
        "output_top5": list(trace.output_top5),
        "ordered_pair_scan_count": trace.ordered_pair_scan_count,
        "plan_sha256": trace.plan_sha256,
        "query_sha256": trace.query_sha256,
        "relevance_sha256": trace.relevance_sha256,
        "version": TYPED_CORE_VERSION,
    }


def _encode_trace(trace: ActionTrace) -> dict[str, Any]:
    if not isinstance(trace, ActionTrace):
        raise HoVerLifecycleStoreError("Agent trace has the wrong type")
    observed = recompute_action_trace_sha256(trace)
    if not hmac.compare_digest(observed, trace.trace_sha256):
        raise HoVerLifecycleStoreError("Agent trace self-recomputation failed")
    envelope = {
        "action_id": trace.action_id,
        "terminal": True,
        "trace": _trace_body(trace),
        "trace_sha256": observed,
    }
    _decode_trace(envelope, expected_action_id=trace.action_id)
    return envelope


_TRACE_KEYS = {
    "action_id",
    "causal",
    "core",
    "core_quality",
    "coverage",
    "coverage_slot_keys",
    "coverage_covered_slot_keys",
    "e0",
    "e1",
    "extension_scan_count",
    "graph_sha256",
    "output_top5",
    "ordered_pair_scan_count",
    "plan_sha256",
    "query_sha256",
    "relevance_sha256",
    "version",
}


def _decode_trace(value: object, *, expected_action_id: str) -> ActionTrace:
    if not isinstance(value, Mapping) or set(value) != {
        "action_id",
        "terminal",
        "trace",
        "trace_sha256",
    }:
        raise HoVerLifecycleStoreError("Agent trace envelope drifted")
    receipt = value.get("trace")
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != _TRACE_KEYS
        or value.get("action_id") != expected_action_id
        or value.get("terminal") is not True
        or receipt.get("action_id") != expected_action_id
        or receipt.get("version") != TYPED_CORE_VERSION
    ):
        raise HoVerLifecycleStoreError("Agent trace identity drifted")
    output = _top5(receipt.get("output_top5"), "Agent")
    core = receipt.get("core")
    coverage = receipt.get("coverage")
    slot_keys = receipt.get("coverage_slot_keys")
    covered_keys = receipt.get("coverage_covered_slot_keys")
    causal = receipt.get("causal")
    if (
        not isinstance(core, list)
        or len(core) != 4
        or any(type(row) is not int or not 0 <= row < CORPUS_SIZE for row in core)
        or len(set(core)) != 4
        or tuple(core) != output[:4]
        or not isinstance(coverage, list)
        or len(coverage) != 2
        or not isinstance(causal, list)
        or len(causal) != 5
    ):
        raise HoVerLifecycleStoreError("Agent trace core/signature shape drifted")
    covered, total = coverage
    if (
        type(covered) is not int
        or type(total) is not int
        or total <= 0
        or not 0 <= covered <= total
        or not isinstance(slot_keys, list)
        or len(slot_keys) != total
        or any(not isinstance(key, str) or not key for key in slot_keys)
        or len(set(slot_keys)) != total
        or not isinstance(covered_keys, list)
        or len(covered_keys) != covered
        or len(set(covered_keys)) != covered
        or any(key not in slot_keys for key in covered_keys)
    ):
        raise HoVerLifecycleStoreError("Agent trace coverage drifted")
    causal_values = (
        causal[0],
        *(_decode_number(row, "causal") for row in causal[1:]),
    )
    if (
        type(causal_values[0]) is not int
        or not 0 <= causal_values[0] <= 4
        or any(not isinstance(row, Fraction) for row in causal_values[1:])
        or causal_values[1] != Fraction(causal_values[0], 4)
    ):
        raise HoVerLifecycleStoreError("Agent causal signature drifted")
    quality = receipt.get("core_quality")
    raw_e0 = receipt.get("e0")
    raw_e1 = receipt.get("e1")
    if (
        not isinstance(quality, list)
        or not quality
        or not isinstance(raw_e0, list)
        or len(raw_e0) != 3
        or not isinstance(raw_e1, list)
        or len(raw_e1) != 6
    ):
        raise HoVerLifecycleStoreError("Agent evaluator key shape drifted")
    e0 = tuple(_decode_number(row, "E0") for row in raw_e0)
    e1 = tuple(_decode_number(row, "E1") for row in raw_e1)
    if e1 != (causal_values[1], causal_values[2], causal_values[4], *e0):
        raise HoVerLifecycleStoreError("Agent E1 key algebra drifted")
    if (
        receipt.get("ordered_pair_scan_count") != CORPUS_SIZE * (CORPUS_SIZE - 1)
        or receipt.get("extension_scan_count")
        != (CORPUS_SIZE - 2) + (CORPUS_SIZE - 3)
    ):
        raise HoVerLifecycleStoreError("Agent exhaustive scan count drifted")
    trace = ActionTrace(
        action_id=expected_action_id,
        output_top5=output,
        core=tuple(core),  # type: ignore[arg-type]
        core_quality=tuple(_decode_number(row, "core quality") for row in quality),
        coverage=CoverageSignature(
            covered=covered,
            total=total,
            value=Fraction(covered, total),
            slot_keys=tuple(slot_keys),
            covered_slot_keys=tuple(covered_keys),
        ),
        causal=CausalSignature(
            necessary_count=causal_values[0],
            necessary_fraction=causal_values[1],  # type: ignore[arg-type]
            minimum_leave_one_out_loss=causal_values[2],  # type: ignore[arg-type]
            minimum_replacement_loss=causal_values[3],  # type: ignore[arg-type]
            path_connectivity=causal_values[4],  # type: ignore[arg-type]
        ),
        e0_key=e0,
        e1_key=e1,
        ordered_pair_scan_count=receipt["ordered_pair_scan_count"],
        extension_scan_count=receipt["extension_scan_count"],
        graph_sha256=_require_sha256(receipt.get("graph_sha256"), "trace graph"),
        plan_sha256=_require_sha256(receipt.get("plan_sha256"), "trace plan"),
        query_sha256=_require_sha256(receipt.get("query_sha256"), "trace query"),
        relevance_sha256=_require_sha256(
            receipt.get("relevance_sha256"), "trace relevance"
        ),
        trace_sha256=_require_sha256(value.get("trace_sha256"), "typed trace"),
    )
    if not hmac.compare_digest(
        recompute_action_trace_sha256(trace), trace.trace_sha256
    ):
        raise HoVerLifecycleStoreError("typed ActionTrace receipt drifted")
    return trace


def _method_output(method: str, top5: Sequence[int]) -> dict[str, Any]:
    output = list(_top5(top5, method))
    return _with_self_hash(
        {"method": method, "terminal": True, "output_top5": output},
        "output_sha256",
    )


def _validate_method_output(value: object, *, method: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "method",
        "terminal",
        "output_top5",
        "output_sha256",
    }:
        raise HoVerLifecycleStoreError(f"{method} output envelope drifted")
    body = dict(value)
    declared = _require_sha256(body.pop("output_sha256"), f"{method} output")
    if not hmac.compare_digest(stable_hash(body), declared):
        raise HoVerLifecycleStoreError(f"{method} output self-hash drifted")
    _top5(value.get("output_top5"), method)
    if value.get("method") != method or value.get("terminal") is not True:
        raise HoVerLifecycleStoreError(f"{method} output is not terminal")
    return dict(value)


def build_stage_output_record(
    *,
    block: str,
    ordinal: int,
    view_sha256: str,
    dense_relevance_ints: Sequence[int],
    raw_top5: Sequence[int],
    hipporag_top5: Sequence[int],
    action_traces: Sequence[ActionTrace],
) -> dict[str, Any]:
    """Build one complete, label-free RAW/HippoRAG/six-action record."""

    if block not in BLOCK_ORDER or type(ordinal) is not int or ordinal < 0:
        raise HoVerLifecycleStoreError("stage record identity drifted")
    _require_sha256(view_sha256, "stage view item")
    relevance, relevance_sha = _dense_relevance(dense_relevance_ints)
    raw = _top5(raw_top5, "RAW")
    expected_raw = tuple(
        sorted(range(CORPUS_SIZE), key=lambda index: (-relevance[index], index))[:5]
    )
    if raw != expected_raw:
        raise HoVerLifecycleStoreError("RAW top5 differs from dense relevance")
    traces = [_encode_trace(trace) for trace in action_traces]
    if [row["action_id"] for row in traces] != list(AGENT_ACTION_IDS):
        raise HoVerLifecycleStoreError("Agent action registry/order drifted")
    payload = _with_self_hash(
        {
            "schema": f"hover_{block.casefold()}_stage_output_record_v1",
            "version": VERSION,
            "block": block,
            "ordinal": ordinal,
            "view_sha256": view_sha256,
            "dense_relevance_ints": list(relevance),
            "relevance_sha256": relevance_sha,
            "raw_output": _method_output("RAW", raw),
            "hipporag_output": _method_output("HippoRAG", hipporag_top5),
            "agent_action_traces": traces,
        },
        "record_sha256",
    )
    _validate_stage_record(payload, block=block, ordinal=ordinal, view_sha256=view_sha256)
    return payload


def _query_sha256(claim: str) -> str:
    return hashlib.sha256(normalize_text(claim).encode("utf-8")).hexdigest()


def _validate_stage_record(
    value: object,
    *,
    block: str,
    ordinal: int,
    view_sha256: str,
    query_sha256: str | None = None,
    graph_sha256: str | None = None,
) -> tuple[dict[str, Any], str, str, tuple[str, ...], tuple[ActionTrace, ...]]:
    expected = {
        "schema",
        "version",
        "block",
        "ordinal",
        "view_sha256",
        "dense_relevance_ints",
        "relevance_sha256",
        "raw_output",
        "hipporag_output",
        "agent_action_traces",
        "record_sha256",
    }
    schema = f"hover_{block.casefold()}_stage_output_record_v1"
    if not isinstance(value, Mapping) or set(value) != expected:
        raise HoVerLifecycleStoreError("stage record schema drifted")
    _verify_self_hash(value, field="record_sha256", schema=schema)
    if (
        value.get("version") != VERSION
        or value.get("block") != block
        or value.get("ordinal") != ordinal
        or value.get("view_sha256") != view_sha256
    ):
        raise HoVerLifecycleStoreError("stage record binding drifted")
    relevance, relevance_sha = _dense_relevance(value.get("dense_relevance_ints"))
    if value.get("relevance_sha256") != relevance_sha:
        raise HoVerLifecycleStoreError("stage relevance receipt drifted")
    raw = _validate_method_output(value.get("raw_output"), method="RAW")
    hippo = _validate_method_output(value.get("hipporag_output"), method="HippoRAG")
    expected_raw = tuple(
        sorted(range(CORPUS_SIZE), key=lambda index: (-relevance[index], index))[:5]
    )
    if tuple(raw["output_top5"]) != expected_raw:
        raise HoVerLifecycleStoreError("archived RAW top5 is not reproducible")
    envelopes = value.get("agent_action_traces")
    if not isinstance(envelopes, list) or len(envelopes) != len(AGENT_ACTION_IDS):
        raise HoVerLifecycleStoreError("complete six-action trace matrix is absent")
    traces = tuple(
        _decode_trace(envelope, expected_action_id=action_id)
        for action_id, envelope in zip(AGENT_ACTION_IDS, envelopes, strict=True)
    )
    inputs = {
        (trace.graph_sha256, trace.plan_sha256, trace.query_sha256, trace.relevance_sha256)
        for trace in traces
    }
    if len(inputs) != 1:
        raise HoVerLifecycleStoreError("six actions do not share one observation input")
    observed_graph, _plan, observed_query, observed_relevance = next(iter(inputs))
    if (
        observed_relevance != relevance_sha
        or (query_sha256 is not None and observed_query != query_sha256)
        or (graph_sha256 is not None and observed_graph != graph_sha256)
    ):
        raise HoVerLifecycleStoreError("stage record input cross-binding drifted")
    return (
        dict(value),
        str(raw["output_sha256"]),
        str(hippo["output_sha256"]),
        tuple(trace.trace_sha256 for trace in traces),
        traces,
    )


def validate_stage_output_record(
    *,
    record: Mapping[str, Any],
    block: str,
    ordinal: int,
    view_sha256: str,
    expected_query_sha256: str | None = None,
    expected_graph_sha256: str | None = None,
) -> dict[str, Any]:
    """Public strict validator for one canonical terminal stage record."""

    checked, _raw, _hippo, _trace_hashes, _traces = _validate_stage_record(
        record,
        block=block,
        ordinal=ordinal,
        view_sha256=view_sha256,
        query_sha256=expected_query_sha256,
        graph_sha256=expected_graph_sha256,
    )
    return checked


def _runtime_binding(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(STAGE_RUNTIME_BINDING_KEYS):
        raise HoVerLifecycleStoreError("stage runtime binding schema drifted")
    return {
        field: _require_sha256(value.get(field), f"stage runtime {field}")
        for field in STAGE_RUNTIME_BINDING_KEYS
    }


def _archive_schema(block: str) -> str:
    return f"hover_{block.casefold()}_stage_output_archive_v1"


def _prior_stage_authorization(project: Path, block: str) -> None:
    if block == "F_search":
        load_a_form_evaluator_freeze(project=project)
    elif block == "A_hold":
        load_f_search_policy_freeze(project=project)
    elif block == "M_search":
        load_a_hold_promotion(project=project)


def _validate_archive_payload(
    *, payload: Mapping[str, Any], context: StageAcquisitionContext, block: str
) -> dict[str, Any]:
    expected = {
        "schema",
        "version",
        "block",
        "acquisition_sha256",
        "acquisition_file_sha256",
        "acquisition_git_head",
        "acquisition_git_blob_sha1",
        "corpus_view_file_sha256",
        "corpus_view_semantic_sha256",
        "block_view_file_sha256",
        "block_view_semantic_sha256",
        "stage_runtime_binding",
        "stage_runtime_binding_sha256",
        "item_count",
        "raw_output_set_sha256",
        "hipporag_output_set_sha256",
        "agent_complete_six_action_trace_matrix_sha256",
        "all_three_methods_terminal",
        "gold_or_hop_fields_included",
        "records",
        "stage_output_archive_sha256",
    }
    if set(payload) != expected:
        raise HoVerLifecycleStoreError(f"{block} archive schema drifted")
    _verify_self_hash(
        payload, field="stage_output_archive_sha256", schema=_archive_schema(block)
    )
    runtime = _runtime_binding(payload.get("stage_runtime_binding"))
    bindings = (
        ("acquisition_sha256", context.acquisition_sha256),
        ("acquisition_file_sha256", context.acquisition_file_sha256),
        ("acquisition_git_head", context.acquisition_git_head),
        ("acquisition_git_blob_sha1", context.acquisition_git_blob_sha1),
        ("corpus_view_file_sha256", context.corpus_file_sha256),
        ("corpus_view_semantic_sha256", context.corpus_semantic_sha256),
        ("block_view_file_sha256", context.view_file_sha256),
        ("block_view_semantic_sha256", context.view_semantic_sha256),
    )
    records = payload.get("records")
    if (
        payload.get("version") != VERSION
        or payload.get("block") != block
        or any(payload.get(field) != expected_value for field, expected_value in bindings)
        or payload.get("stage_runtime_binding_sha256") != stable_hash(runtime)
        or payload.get("item_count") != BLOCK_COUNTS[block]
        or payload.get("all_three_methods_terminal") is not True
        or payload.get("gold_or_hop_fields_included") is not False
        or not isinstance(records, list)
        or len(records) != BLOCK_COUNTS[block]
    ):
        raise HoVerLifecycleStoreError(f"{block} archive binding drifted")
    raw_hashes: list[str] = []
    hippo_hashes: list[str] = []
    trace_matrix: list[list[str]] = []
    record_hashes: set[str] = set()
    for ordinal, (record, view_item) in enumerate(
        zip(records, context.view_items, strict=True)
    ):
        checked, raw_sha, hippo_sha, trace_hashes, _traces = _validate_stage_record(
            record,
            block=block,
            ordinal=ordinal,
            view_sha256=stable_hash(dict(view_item)),
            query_sha256=_query_sha256(str(view_item["claim"])),
            graph_sha256=runtime["graph_sha256"],
        )
        raw_hashes.append(raw_sha)
        hippo_hashes.append(hippo_sha)
        trace_matrix.append(list(trace_hashes))
        record_hashes.add(str(checked["record_sha256"]))
    if (
        len(record_hashes) != BLOCK_COUNTS[block]
        or payload.get("raw_output_set_sha256") != stable_hash(raw_hashes)
        or payload.get("hipporag_output_set_sha256") != stable_hash(hippo_hashes)
        or payload.get("agent_complete_six_action_trace_matrix_sha256")
        != stable_hash(trace_matrix)
    ):
        raise HoVerLifecycleStoreError(f"{block} archive derived commitments drifted")
    return dict(payload)


def load_stage_output_archive(
    *, project: Path, block: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if block not in BLOCK_ORDER:
        raise HoVerLifecycleStoreError("stage archive block is invalid")
    root = _project_root(project)
    _prior_stage_authorization(root, block)
    context = _load_stage_acquisition_context(project=root, block=block)
    _validate_context(context, block=block)
    payload, raw = _read_json_file(
        root=root,
        relative=STAGE_OUTPUT_ARCHIVE_RELATIVES[block],
        field=f"{block} stage archive",
        mode=0o600,
    )
    checked = _validate_archive_payload(payload=payload, context=context, block=block)
    return checked, {
        "file_sha256": _sha256_bytes(raw),
        "semantic_sha256": checked["stage_output_archive_sha256"],
        "byte_size": len(raw),
        "mode": "0600",
    }


def create_stage_output_archive_once(
    *,
    project: Path,
    block: str,
    records: Sequence[Mapping[str, Any]],
    stage_runtime_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if block not in BLOCK_ORDER:
        raise HoVerLifecycleStoreError("stage archive block is invalid")
    root = _project_root(project)
    _prior_stage_authorization(root, block)
    context = _load_stage_acquisition_context(project=root, block=block)
    _validate_context(context, block=block)
    runtime = _runtime_binding(stage_runtime_binding)
    if len(records) != BLOCK_COUNTS[block]:
        raise HoVerLifecycleStoreError("stage archive record count drifted")
    exact_records: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    hippo_hashes: list[str] = []
    trace_matrix: list[list[str]] = []
    for ordinal, (record, view_item) in enumerate(
        zip(records, context.view_items, strict=True)
    ):
        checked, raw_sha, hippo_sha, trace_hashes, _traces = _validate_stage_record(
            record,
            block=block,
            ordinal=ordinal,
            view_sha256=stable_hash(dict(view_item)),
            query_sha256=_query_sha256(str(view_item["claim"])),
            graph_sha256=runtime["graph_sha256"],
        )
        exact_records.append(checked)
        raw_hashes.append(raw_sha)
        hippo_hashes.append(hippo_sha)
        trace_matrix.append(list(trace_hashes))
    payload = _with_self_hash(
        {
            "schema": _archive_schema(block),
            "version": VERSION,
            "block": block,
            "acquisition_sha256": context.acquisition_sha256,
            "acquisition_file_sha256": context.acquisition_file_sha256,
            "acquisition_git_head": context.acquisition_git_head,
            "acquisition_git_blob_sha1": context.acquisition_git_blob_sha1,
            "corpus_view_file_sha256": context.corpus_file_sha256,
            "corpus_view_semantic_sha256": context.corpus_semantic_sha256,
            "block_view_file_sha256": context.view_file_sha256,
            "block_view_semantic_sha256": context.view_semantic_sha256,
            "stage_runtime_binding": runtime,
            "stage_runtime_binding_sha256": stable_hash(runtime),
            "item_count": BLOCK_COUNTS[block],
            "raw_output_set_sha256": stable_hash(raw_hashes),
            "hipporag_output_set_sha256": stable_hash(hippo_hashes),
            "agent_complete_six_action_trace_matrix_sha256": stable_hash(trace_matrix),
            "all_three_methods_terminal": True,
            "gold_or_hop_fields_included": False,
            "records": exact_records,
        },
        "stage_output_archive_sha256",
    )
    _validate_archive_payload(payload=payload, context=context, block=block)
    _write_json_exclusive(
        root=root,
        relative=STAGE_OUTPUT_ARCHIVE_RELATIVES[block],
        payload=payload,
        mode=0o600,
    )
    return load_stage_output_archive(project=root, block=block)


def _observations_from_archive(
    archive: Mapping[str, Any], *, block: str
) -> tuple[EvaluationObservation, ...]:
    records = archive.get("records")
    if not isinstance(records, list) or len(records) != BLOCK_COUNTS[block]:
        raise HoVerLifecycleStoreError("policy archive trace matrix drifted")
    observations: list[EvaluationObservation] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("agent_action_traces"), list
        ):
            raise HoVerLifecycleStoreError("policy record trace matrix drifted")
        traces = {
            action_id: _decode_trace(envelope, expected_action_id=action_id)
            for action_id, envelope in zip(
                AGENT_ACTION_IDS, record["agent_action_traces"], strict=True
            )
        }
        observations.append(EvaluationObservation(traces_by_action=traces))
    return tuple(observations)


def _recompute_policies(
    archive: Mapping[str, Any], *, block: str
) -> tuple[PolicySelection, PolicySelection, bool]:
    observations = _observations_from_archive(archive, block=block)
    try:
        e0 = select_global_policy(
            evaluator_id="E0_INDEPENDENT_V2", observations=observations
        )
        e1 = select_global_policy(
            evaluator_id="E1_CAUSAL_NECESSITY_V2", observations=observations
        )
        identifiable = policies_identifiable(e0, e1, observations)
    except MultiHopRAGTypedOperatorV2Error as exc:
        raise HoVerLifecycleStoreError("typed policy recomputation failed") from exc
    return e0, e1, identifiable


def _encode_policy(policy: PolicySelection) -> dict[str, Any]:
    if (
        not isinstance(policy, PolicySelection)
        or policy.selection_sha256 != recompute_policy_selection_sha256(policy)
    ):
        raise HoVerLifecycleStoreError("PolicySelection receipt drifted")
    return {
        "typed_core_version": TYPED_CORE_VERSION,
        "evaluator_id": policy.evaluator_id,
        "action_id": policy.action_id,
        "observation_count": policy.observation_count,
        "macro_key": [_number(row) for row in policy.macro_key],
        "per_action_macro_keys": [
            [action_id, [_number(row) for row in key]]
            for action_id, key in policy.per_action_macro_keys
        ],
        "input_receipt_sha256": policy.input_receipt_sha256,
        "selection_sha256": policy.selection_sha256,
    }


def _decode_policy(value: object, *, evaluator_id: str) -> PolicySelection:
    if not isinstance(value, Mapping) or set(value) != {
        "typed_core_version",
        "evaluator_id",
        "action_id",
        "observation_count",
        "macro_key",
        "per_action_macro_keys",
        "input_receipt_sha256",
        "selection_sha256",
    }:
        raise HoVerLifecycleStoreError("PolicySelection envelope drifted")
    macro = value.get("macro_key")
    per_action = value.get("per_action_macro_keys")
    if (
        value.get("typed_core_version") != TYPED_CORE_VERSION
        or value.get("evaluator_id") != evaluator_id
        or value.get("action_id") not in AGENT_ACTION_IDS
        or type(value.get("observation_count")) is not int
        or value["observation_count"] <= 0
        or not isinstance(macro, list)
        or not isinstance(per_action, list)
        or len(per_action) != len(AGENT_ACTION_IDS)
    ):
        raise HoVerLifecycleStoreError("PolicySelection identity drifted")
    decoded_rows: list[tuple[str, tuple[Fraction, ...]]] = []
    for expected_action, row in zip(AGENT_ACTION_IDS, per_action, strict=True):
        if (
            not isinstance(row, list)
            or len(row) != 2
            or row[0] != expected_action
            or not isinstance(row[1], list)
        ):
            raise HoVerLifecycleStoreError("per-action policy key drifted")
        values = tuple(_decode_number(part, "policy key") for part in row[1])
        if any(not isinstance(part, Fraction) for part in values):
            raise HoVerLifecycleStoreError("policy keys must be exact fractions")
        decoded_rows.append((expected_action, values))  # type: ignore[arg-type]
    macro_values = tuple(_decode_number(part, "policy macro") for part in macro)
    if any(not isinstance(part, Fraction) for part in macro_values):
        raise HoVerLifecycleStoreError("policy macro must be exact fractions")
    policy = PolicySelection(
        evaluator_id=evaluator_id,
        action_id=str(value["action_id"]),
        observation_count=value["observation_count"],
        macro_key=macro_values,  # type: ignore[arg-type]
        per_action_macro_keys=tuple(decoded_rows),
        input_receipt_sha256=_require_sha256(
            value.get("input_receipt_sha256"), "policy input"
        ),
        selection_sha256=_require_sha256(
            value.get("selection_sha256"), "policy selection"
        ),
    )
    if policy.selection_sha256 != recompute_policy_selection_sha256(policy):
        raise HoVerLifecycleStoreError("PolicySelection self-hash mismatch")
    return policy


def _archive_binding(
    *, project: Path, block: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_stage_output_archive(project=project, block=block)


def _policy_prerequisite(
    project: Path, block: str
) -> tuple[str, str | None, str | None, str | None]:
    if block == "A_form":
        return "all_six_actions_no_F_policy", None, None, None
    if block == "A_hold":
        freeze = load_f_search_policy_freeze(project=project)
        return (
            "frozen_F_E0_and_E1_policies",
            str(freeze["policy_freeze_sha256"]),
            str(freeze["e0_policy"]["selection_sha256"]),
            str(freeze["e1_policy"]["selection_sha256"]),
        )
    promotion = load_a_hold_promotion(project=project)
    return (
        "promoted_E1_and_counterfactual_E0_policies",
        str(promotion["promotion_sha256"]),
        str(promotion["e0_policy_sha256"]),
        str(promotion["e1_policy_sha256"]),
    )


def _action_seal_schema(block: str) -> str:
    return f"hover_{block.casefold()}_action_seal_v1"


def _validate_action_seal(
    *, payload: Mapping[str, Any], project: Path, block: str
) -> dict[str, Any]:
    expected = {
        "schema",
        "version",
        "status",
        "block",
        "acquisition_sha256",
        "stage_output_archive_file_sha256",
        "stage_output_archive_semantic_sha256",
        "raw_output_set_sha256",
        "hipporag_output_set_sha256",
        "agent_complete_six_action_trace_matrix_sha256",
        "policy_stage",
        "policy_authorization_sha256",
        "e0_policy_sha256",
        "e1_policy_sha256",
        "label_pack_opened_before_action_seal",
        "created_with_O_EXCL_and_fsync",
        "same_block_replay_authorized",
        "action_seal_sha256",
    }
    if set(payload) != expected:
        raise HoVerLifecycleStoreError(f"{block} action seal schema drifted")
    _verify_self_hash(payload, field="action_seal_sha256", schema=_action_seal_schema(block))
    archive, binding = _archive_binding(project=project, block=block)
    stage, authorization, e0, e1 = _policy_prerequisite(project, block)
    if (
        payload.get("version") != VERSION
        or payload.get("status") != f"{block}_all_methods_terminal"
        or payload.get("block") != block
        or payload.get("acquisition_sha256") != archive["acquisition_sha256"]
        or payload.get("stage_output_archive_file_sha256") != binding["file_sha256"]
        or payload.get("stage_output_archive_semantic_sha256") != binding["semantic_sha256"]
        or payload.get("raw_output_set_sha256") != archive["raw_output_set_sha256"]
        or payload.get("hipporag_output_set_sha256") != archive["hipporag_output_set_sha256"]
        or payload.get("agent_complete_six_action_trace_matrix_sha256")
        != archive["agent_complete_six_action_trace_matrix_sha256"]
        or payload.get("policy_stage") != stage
        or payload.get("policy_authorization_sha256") != authorization
        or payload.get("e0_policy_sha256") != e0
        or payload.get("e1_policy_sha256") != e1
        or payload.get("label_pack_opened_before_action_seal") is not False
        or payload.get("created_with_O_EXCL_and_fsync") is not True
        or payload.get("same_block_replay_authorized") is not False
    ):
        raise HoVerLifecycleStoreError(f"{block} action seal binding drifted")
    return dict(payload)


def load_action_seal(*, project: Path, block: str) -> dict[str, Any]:
    if block not in ACTION_SEAL_RELATIVES:
        raise HoVerLifecycleStoreError("block has no action seal")
    root = _project_root(project)
    payload, _raw = _read_json_file(
        root=root,
        relative=ACTION_SEAL_RELATIVES[block],
        field=f"{block} action seal",
        mode=0o644,
    )
    return _validate_action_seal(payload=payload, project=root, block=block)


def create_action_seal_once(*, project: Path, block: str) -> dict[str, Any]:
    if block not in ACTION_SEAL_RELATIVES:
        raise HoVerLifecycleStoreError("block has no action seal")
    root = _project_root(project)
    archive, binding = _archive_binding(project=root, block=block)
    stage, authorization, e0, e1 = _policy_prerequisite(root, block)
    payload = _with_self_hash(
        {
            "schema": _action_seal_schema(block),
            "version": VERSION,
            "status": f"{block}_all_methods_terminal",
            "block": block,
            "acquisition_sha256": archive["acquisition_sha256"],
            "stage_output_archive_file_sha256": binding["file_sha256"],
            "stage_output_archive_semantic_sha256": binding["semantic_sha256"],
            "raw_output_set_sha256": archive["raw_output_set_sha256"],
            "hipporag_output_set_sha256": archive["hipporag_output_set_sha256"],
            "agent_complete_six_action_trace_matrix_sha256": archive[
                "agent_complete_six_action_trace_matrix_sha256"
            ],
            "policy_stage": stage,
            "policy_authorization_sha256": authorization,
            "e0_policy_sha256": e0,
            "e1_policy_sha256": e1,
            "label_pack_opened_before_action_seal": False,
            "created_with_O_EXCL_and_fsync": True,
            "same_block_replay_authorized": False,
        },
        "action_seal_sha256",
    )
    _validate_action_seal(payload=payload, project=root, block=block)
    _write_json_exclusive(
        root=root, relative=ACTION_SEAL_RELATIVES[block], payload=payload, mode=0o644
    )
    return load_action_seal(project=root, block=block)


def _freeze_schema(kind: str) -> str:
    return f"hover_{kind}_v1"


def _validate_evaluator_freeze(
    *, payload: Mapping[str, Any], project: Path
) -> dict[str, Any]:
    expected = {
        "schema",
        "version",
        "status",
        "acquisition_sha256",
        "a_form_archive_file_sha256",
        "a_form_archive_semantic_sha256",
        "a_form_action_seal_sha256",
        "complete_a_form_trace_matrix_sha256",
        "typed_core_version",
        "fixed_evaluator_definitions",
        "e0_policy",
        "e1_policy",
        "policies_identifiable",
        "selection_purpose",
        "A_form_labels_opened_before_freeze",
        "same_stage_replay_or_reselection_authorized",
        "evaluator_freeze_sha256",
    }
    if set(payload) != expected:
        raise HoVerLifecycleStoreError("A_form evaluator freeze schema drifted")
    _verify_self_hash(
        payload,
        field="evaluator_freeze_sha256",
        schema=_freeze_schema("a_form_evaluator_freeze"),
    )
    archive, binding = load_stage_output_archive(project=project, block="A_form")
    seal = load_action_seal(project=project, block="A_form")
    expected_e0, expected_e1, identifiable = _recompute_policies(
        archive, block="A_form"
    )
    e0 = _decode_policy(payload.get("e0_policy"), evaluator_id="E0_INDEPENDENT_V2")
    e1 = _decode_policy(
        payload.get("e1_policy"), evaluator_id="E1_CAUSAL_NECESSITY_V2"
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "A_form_prelabel_evaluators_frozen"
        or payload.get("acquisition_sha256") != archive["acquisition_sha256"]
        or payload.get("a_form_archive_file_sha256") != binding["file_sha256"]
        or payload.get("a_form_archive_semantic_sha256") != binding["semantic_sha256"]
        or payload.get("a_form_action_seal_sha256") != seal["action_seal_sha256"]
        or payload.get("complete_a_form_trace_matrix_sha256")
        != archive["agent_complete_six_action_trace_matrix_sha256"]
        or payload.get("typed_core_version") != TYPED_CORE_VERSION
        or payload.get("fixed_evaluator_definitions") != EVALUATOR_DEFINITIONS
        or e0 != expected_e0
        or e1 != expected_e1
        or payload.get("policies_identifiable") is not identifiable
        or payload.get("selection_purpose") != "diagnostic_only_not_F_policy"
        or payload.get("A_form_labels_opened_before_freeze") is not False
        or payload.get("same_stage_replay_or_reselection_authorized") is not False
    ):
        raise HoVerLifecycleStoreError("A_form evaluator freeze binding drifted")
    return dict(payload)


def load_a_form_evaluator_freeze(*, project: Path) -> dict[str, Any]:
    root = _project_root(project)
    payload, _raw = _read_json_file(
        root=root,
        relative=A_FORM_EVALUATOR_FREEZE_RELATIVE,
        field="A_form evaluator freeze",
        mode=0o644,
    )
    return _validate_evaluator_freeze(payload=payload, project=root)


def create_a_form_evaluator_freeze_once(
    *, project: Path, e0_policy: PolicySelection, e1_policy: PolicySelection
) -> dict[str, Any]:
    root = _project_root(project)
    archive, binding = load_stage_output_archive(project=root, block="A_form")
    seal = load_action_seal(project=root, block="A_form")
    expected_e0, expected_e1, identifiable = _recompute_policies(
        archive, block="A_form"
    )
    if e0_policy != expected_e0 or e1_policy != expected_e1:
        raise HoVerLifecycleStoreError(
            "supplied A_form diagnostic policies differ from canonical evidence"
        )
    payload = _with_self_hash(
        {
            "schema": _freeze_schema("a_form_evaluator_freeze"),
            "version": VERSION,
            "status": "A_form_prelabel_evaluators_frozen",
            "acquisition_sha256": archive["acquisition_sha256"],
            "a_form_archive_file_sha256": binding["file_sha256"],
            "a_form_archive_semantic_sha256": binding["semantic_sha256"],
            "a_form_action_seal_sha256": seal["action_seal_sha256"],
            "complete_a_form_trace_matrix_sha256": archive[
                "agent_complete_six_action_trace_matrix_sha256"
            ],
            "typed_core_version": TYPED_CORE_VERSION,
            "fixed_evaluator_definitions": EVALUATOR_DEFINITIONS,
            "e0_policy": _encode_policy(e0_policy),
            "e1_policy": _encode_policy(e1_policy),
            "policies_identifiable": identifiable,
            "selection_purpose": "diagnostic_only_not_F_policy",
            "A_form_labels_opened_before_freeze": False,
            "same_stage_replay_or_reselection_authorized": False,
        },
        "evaluator_freeze_sha256",
    )
    _validate_evaluator_freeze(payload=payload, project=root)
    _write_json_exclusive(
        root=root,
        relative=A_FORM_EVALUATOR_FREEZE_RELATIVE,
        payload=payload,
        mode=0o644,
    )
    return load_a_form_evaluator_freeze(project=root)


def _validate_f_policy_freeze(
    *, payload: Mapping[str, Any], project: Path
) -> dict[str, Any]:
    expected = {
        "schema",
        "version",
        "status",
        "acquisition_sha256",
        "f_archive_file_sha256",
        "f_archive_semantic_sha256",
        "complete_f_trace_matrix_sha256",
        "typed_core_version",
        "fixed_evaluator_definitions",
        "e0_policy",
        "e1_policy",
        "e0_action_id",
        "e0_policy_sha256",
        "e1_action_id",
        "e1_policy_sha256",
        "policies_identifiable",
        "F_search_label_pack_exists",
        "same_stage_replay_or_reselection_authorized",
        "policy_freeze_sha256",
    }
    if set(payload) != expected:
        raise HoVerLifecycleStoreError("F_search policy freeze schema drifted")
    _verify_self_hash(
        payload,
        field="policy_freeze_sha256",
        schema=_freeze_schema("f_search_policy_freeze"),
    )
    archive, binding = load_stage_output_archive(project=project, block="F_search")
    expected_e0, expected_e1, identifiable = _recompute_policies(
        archive, block="F_search"
    )
    e0 = _decode_policy(payload.get("e0_policy"), evaluator_id="E0_INDEPENDENT_V2")
    e1 = _decode_policy(
        payload.get("e1_policy"), evaluator_id="E1_CAUSAL_NECESSITY_V2"
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "F_search_terminal_policies_frozen"
        or payload.get("acquisition_sha256") != archive["acquisition_sha256"]
        or payload.get("f_archive_file_sha256") != binding["file_sha256"]
        or payload.get("f_archive_semantic_sha256") != binding["semantic_sha256"]
        or payload.get("complete_f_trace_matrix_sha256")
        != archive["agent_complete_six_action_trace_matrix_sha256"]
        or payload.get("typed_core_version") != TYPED_CORE_VERSION
        or payload.get("fixed_evaluator_definitions") != EVALUATOR_DEFINITIONS
        or e0 != expected_e0
        or e1 != expected_e1
        or payload.get("e0_action_id") != e0.action_id
        or payload.get("e0_policy_sha256") != e0.selection_sha256
        or payload.get("e1_action_id") != e1.action_id
        or payload.get("e1_policy_sha256") != e1.selection_sha256
        or identifiable is not True
        or payload.get("policies_identifiable") is not True
        or payload.get("F_search_label_pack_exists") is not False
        or payload.get("same_stage_replay_or_reselection_authorized") is not False
    ):
        raise HoVerLifecycleStoreError("F_search policy freeze binding drifted")
    return dict(payload)


def load_f_search_policy_freeze(*, project: Path) -> dict[str, Any]:
    root = _project_root(project)
    payload, _raw = _read_json_file(
        root=root,
        relative=F_POLICY_FREEZE_RELATIVE,
        field="F_search policy freeze",
        mode=0o644,
    )
    return _validate_f_policy_freeze(payload=payload, project=root)


def create_f_search_policy_freeze_once(
    *, project: Path, e0_policy: PolicySelection, e1_policy: PolicySelection
) -> dict[str, Any]:
    """Persist supplied typed selections only if archive recomputation is exact."""

    root = _project_root(project)
    archive, binding = load_stage_output_archive(project=root, block="F_search")
    expected_e0, expected_e1, identifiable = _recompute_policies(
        archive, block="F_search"
    )
    if e0_policy != expected_e0 or e1_policy != expected_e1 or not identifiable:
        raise HoVerLifecycleStoreError("supplied F policies differ from canonical evidence")
    payload = _with_self_hash(
        {
            "schema": _freeze_schema("f_search_policy_freeze"),
            "version": VERSION,
            "status": "F_search_terminal_policies_frozen",
            "acquisition_sha256": archive["acquisition_sha256"],
            "f_archive_file_sha256": binding["file_sha256"],
            "f_archive_semantic_sha256": binding["semantic_sha256"],
            "complete_f_trace_matrix_sha256": archive[
                "agent_complete_six_action_trace_matrix_sha256"
            ],
            "typed_core_version": TYPED_CORE_VERSION,
            "fixed_evaluator_definitions": EVALUATOR_DEFINITIONS,
            "e0_policy": _encode_policy(e0_policy),
            "e1_policy": _encode_policy(e1_policy),
            "e0_action_id": e0_policy.action_id,
            "e0_policy_sha256": e0_policy.selection_sha256,
            "e1_action_id": e1_policy.action_id,
            "e1_policy_sha256": e1_policy.selection_sha256,
            "policies_identifiable": True,
            "F_search_label_pack_exists": False,
            "same_stage_replay_or_reselection_authorized": False,
        },
        "policy_freeze_sha256",
    )
    _validate_f_policy_freeze(payload=payload, project=root)
    _write_json_exclusive(
        root=root, relative=F_POLICY_FREEZE_RELATIVE, payload=payload, mode=0o644
    )
    return load_f_search_policy_freeze(project=root)


_A_HOLD_REPORT_KEYS = {
    "primary_passed",
    "promoted",
    "promotion_delta_total",
    "promotion_signflip_p",
    "e0_minus_hippo_delta_total",
    "e0_minus_hippo_signflip_p",
    "e0_minus_hippo_stratum_deltas",
    "e0_minus_raw_delta_total",
    "e0_minus_raw_signflip_p",
    "e0_complete_count",
    "raw_complete_count",
}


def _canonical_a_hold_report(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _A_HOLD_REPORT_KEYS:
        raise HoVerLifecycleStoreError("A_hold outcome report schema drifted")
    report = dict(value)
    if (
        type(report["primary_passed"]) is not bool
        or type(report["promoted"]) is not bool
        or type(report["e0_complete_count"]) is not int
        or type(report["raw_complete_count"]) is not int
        or not 0 <= report["e0_complete_count"] <= BLOCK_COUNTS["A_hold"]
        or not 0 <= report["raw_complete_count"] <= BLOCK_COUNTS["A_hold"]
    ):
        raise HoVerLifecycleStoreError("A_hold outcome scalar drifted")
    for field in (
        "promotion_delta_total",
        "promotion_signflip_p",
        "e0_minus_hippo_delta_total",
        "e0_minus_hippo_signflip_p",
        "e0_minus_raw_delta_total",
        "e0_minus_raw_signflip_p",
    ):
        if not isinstance(_decode_number(report[field], field), Fraction):
            raise HoVerLifecycleStoreError(f"{field} must be an exact fraction")
    strata = report["e0_minus_hippo_stratum_deltas"]
    if not isinstance(strata, Mapping) or set(strata) != {"2_hop", "3_hop", "4_hop"}:
        raise HoVerLifecycleStoreError("A_hold hop-stratum report drifted")
    for hop, value_row in strata.items():
        if not isinstance(_decode_number(value_row, str(hop)), Fraction):
            raise HoVerLifecycleStoreError("A_hold stratum delta is not exact")
    # Re-encoding proves the input already uses canonical reduced fractions.
    _canonical_bytes(report)
    return report


def _load_joined_late_labels(
    *, project: Path, context: StageAcquisitionContext, expected_block: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Late-open and one-to-one join one committed held-out utility mapping."""

    if expected_block not in {"A_hold", "M_search"} or context.block != expected_block:
        raise HoVerLifecycleStoreError("late-label context block drifted")
    labels = (
        dict(context.late_labels)
        if context.late_labels is not None
        else acquisition.load_block_labels(
            project=project, expected_block=expected_block
        )
    )
    try:
        acquisition.validate_block_labels(labels, expected_block=expected_block)
    except Exception as exc:
        raise HoVerLifecycleStoreError(
            f"committed {expected_block} late-label pack failed validation"
        ) from exc
    items = labels.get("items")
    if not isinstance(items, list) or len(items) != BLOCK_COUNTS[expected_block]:
        raise HoVerLifecycleStoreError(
            f"{expected_block} late-label cardinality drifted"
        )
    expected_keys = {
        "schema",
        "block",
        "ordinal",
        "view_sha256",
        "identity_commitment_sha256",
        "source_record_commitment_sha256",
        "hop_stratum",
        "gold_article_ids",
    }
    identities: set[str] = set()
    source_records: set[str] = set()
    joined_view_hashes: set[str] = set()
    strata: Counter[str] = Counter()
    joined: list[tuple[str, tuple[int, ...]]] = []
    for ordinal, (label, view_item) in enumerate(
        zip(items, context.view_items, strict=True)
    ):
        if not isinstance(label, Mapping) or set(label) != expected_keys:
            raise HoVerLifecycleStoreError(
                f"{expected_block} late-label row schema drifted"
            )
        view_sha = _require_sha256(label.get("view_sha256"), "late-label view")
        identity = _require_sha256(
            label.get("identity_commitment_sha256"), "late-label identity"
        )
        source = _require_sha256(
            label.get("source_record_commitment_sha256"), "late-label source"
        )
        stratum = label.get("hop_stratum")
        gold = label.get("gold_article_ids")
        expected_gold_count = (
            int(str(stratum)[0])
            if stratum in {"2_hop", "3_hop", "4_hop"}
            else -1
        )
        if (
            label.get("block") != expected_block
            or label.get("ordinal") != ordinal
            or view_sha != stable_hash(dict(view_item))
            or not isinstance(gold, list)
            or len(gold) != expected_gold_count
            or gold != sorted(set(gold))
            or any(
                type(article_id) is not int
                or not 0 <= article_id < CORPUS_SIZE
                for article_id in gold
            )
            or identity in identities
            or source in source_records
            or view_sha in joined_view_hashes
        ):
            raise HoVerLifecycleStoreError(
                f"{expected_block} late-label join drifted"
            )
        identities.add(identity)
        source_records.add(source)
        joined_view_hashes.add(view_sha)
        strata[str(stratum)] += 1
        joined.append((str(stratum), tuple(gold)))
    quota = BLOCK_COUNTS[expected_block] // 3
    if strata != Counter(
        {"2_hop": quota, "3_hop": quota, "4_hop": quota}
    ):
        raise HoVerLifecycleStoreError(
            f"{expected_block} formal hop quotas drifted"
        )
    return tuple(joined)


def _recompute_a_hold_report_from_evidence(
    *,
    project: Path,
    freeze: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the controller report without trusting any reported number."""

    context = _load_stage_acquisition_context(project=project, block="A_hold")
    _validate_context(context, block="A_hold")
    if archive.get("acquisition_sha256") != context.acquisition_sha256:
        raise HoVerLifecycleStoreError("A_hold report acquisition binding drifted")
    joined = _load_joined_late_labels(
        project=project, context=context, expected_block="A_hold"
    )
    e0_policy = _decode_policy(
        freeze.get("e0_policy"), evaluator_id="E0_INDEPENDENT_V2"
    )
    e1_policy = _decode_policy(
        freeze.get("e1_policy"), evaluator_id="E1_CAUSAL_NECESSITY_V2"
    )
    if (
        e0_policy.action_id == e1_policy.action_id
        or freeze.get("e0_action_id") != e0_policy.action_id
        or freeze.get("e1_action_id") != e1_policy.action_id
    ):
        raise HoVerLifecycleStoreError("frozen A_hold policy identity drifted")
    records = archive.get("records")
    if not isinstance(records, list) or len(records) != len(joined):
        raise HoVerLifecycleStoreError("A_hold archived evidence count drifted")
    e0_values: list[Fraction] = []
    e1_values: list[Fraction] = []
    hippo_values: list[Fraction] = []
    raw_values: list[Fraction] = []
    stratum_deltas = {
        "2_hop": Fraction(0),
        "3_hop": Fraction(0),
        "4_hop": Fraction(0),
    }
    e0_complete = 0
    raw_complete = 0
    for ordinal, (record, (stratum, gold)) in enumerate(
        zip(records, joined, strict=True)
    ):
        if not isinstance(record, Mapping) or record.get("ordinal") != ordinal:
            raise HoVerLifecycleStoreError("A_hold archived record order drifted")
        envelopes = record.get("agent_action_traces")
        if not isinstance(envelopes, list) or len(envelopes) != len(AGENT_ACTION_IDS):
            raise HoVerLifecycleStoreError("A_hold six-action matrix drifted")
        traces = {
            action_id: _decode_trace(envelope, expected_action_id=action_id)
            for action_id, envelope in zip(
                AGENT_ACTION_IDS, envelopes, strict=True
            )
        }
        raw_output = _validate_method_output(
            record.get("raw_output"), method="RAW"
        )["output_top5"]
        hippo_output = _validate_method_output(
            record.get("hipporag_output"), method="HippoRAG"
        )["output_top5"]
        e0_output = traces[e0_policy.action_id].output_top5
        e1_output = traces[e1_policy.action_id].output_top5
        try:
            e0_value = item_utility(e0_output, gold)
            e1_value = item_utility(e1_output, gold)
            hippo_value = item_utility(hippo_output, gold)
            raw_value = item_utility(raw_output, gold)
        except MultiHopRAGTypedOperatorV2Error as exc:
            raise HoVerLifecycleStoreError("A_hold utility recomputation failed") from exc
        e0_values.append(e0_value)
        e1_values.append(e1_value)
        hippo_values.append(hippo_value)
        raw_values.append(raw_value)
        stratum_deltas[stratum] += e0_value - hippo_value
        e0_complete += int(set(gold) <= set(e0_output))
        raw_complete += int(set(gold) <= set(raw_output))
    try:
        promotion = paired_utility_summary(e1_values, e0_values)
        primary = paired_utility_summary(e0_values, hippo_values)
        raw_boundary = paired_utility_summary(e0_values, raw_values)
    except MultiHopRAGTypedOperatorV2Error as exc:
        raise HoVerLifecycleStoreError("A_hold paired summary failed") from exc
    return _canonical_a_hold_report(
        {
            "primary_passed": primary.delta_total > 0
            and primary.exact_one_sided_p <= Fraction(1, 10)
            and all(value > 0 for value in stratum_deltas.values()),
            "promoted": promotion.delta_total > 0
            and promotion.exact_one_sided_p <= Fraction(1, 10),
            "promotion_delta_total": _number(promotion.delta_total),
            "promotion_signflip_p": _number(promotion.exact_one_sided_p),
            "e0_minus_hippo_delta_total": _number(primary.delta_total),
            "e0_minus_hippo_signflip_p": _number(primary.exact_one_sided_p),
            "e0_minus_hippo_stratum_deltas": {
                stratum: _number(stratum_deltas[stratum])
                for stratum in ("2_hop", "3_hop", "4_hop")
            },
            "e0_minus_raw_delta_total": _number(raw_boundary.delta_total),
            "e0_minus_raw_signflip_p": _number(
                raw_boundary.exact_one_sided_p
            ),
            "e0_complete_count": e0_complete,
            "raw_complete_count": raw_complete,
        }
    )


def recompute_a_hold_outcome_report(*, project: Path) -> dict[str, Any]:
    """Read sealed evidence and independently reproduce the exact A_hold report."""

    root = _project_root(project)
    freeze = load_f_search_policy_freeze(project=root)
    archive, _binding = load_stage_output_archive(project=root, block="A_hold")
    # This must succeed before the function reaches the late-label loader.
    load_action_seal(project=root, block="A_hold")
    return _recompute_a_hold_report_from_evidence(
        project=root, freeze=freeze, archive=archive
    )


def _validate_promotion(
    *, payload: Mapping[str, Any], project: Path
) -> dict[str, Any]:
    expected = {
        "schema",
        "version",
        "status",
        "acquisition_sha256",
        "f_policy_freeze_sha256",
        "a_hold_archive_file_sha256",
        "a_hold_archive_semantic_sha256",
        "a_hold_action_seal_sha256",
        "e0_action_id",
        "e0_policy_sha256",
        "e1_action_id",
        "e1_policy_sha256",
        "outcome_report",
        "outcome_report_sha256",
        "challenger_promoted",
        "outcome_used_to_change_action_evaluator_threshold_or_cohort",
        "same_source_replay_authorized",
        "promotion_sha256",
    }
    if set(payload) != expected:
        raise HoVerLifecycleStoreError("A_hold promotion schema drifted")
    _verify_self_hash(
        payload, field="promotion_sha256", schema=_freeze_schema("a_hold_promotion")
    )
    freeze = load_f_search_policy_freeze(project=project)
    archive, binding = load_stage_output_archive(project=project, block="A_hold")
    seal = load_action_seal(project=project, block="A_hold")
    report = _canonical_a_hold_report(payload.get("outcome_report"))
    recomputed_report = _recompute_a_hold_report_from_evidence(
        project=project, freeze=freeze, archive=archive
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "A_hold_challenger_promoted"
        or payload.get("acquisition_sha256") != archive["acquisition_sha256"]
        or payload.get("f_policy_freeze_sha256") != freeze["policy_freeze_sha256"]
        or payload.get("a_hold_archive_file_sha256") != binding["file_sha256"]
        or payload.get("a_hold_archive_semantic_sha256") != binding["semantic_sha256"]
        or payload.get("a_hold_action_seal_sha256") != seal["action_seal_sha256"]
        or payload.get("e0_action_id") != freeze["e0_action_id"]
        or payload.get("e0_policy_sha256") != freeze["e0_policy_sha256"]
        or payload.get("e1_action_id") != freeze["e1_action_id"]
        or payload.get("e1_policy_sha256") != freeze["e1_policy_sha256"]
        or payload.get("outcome_report_sha256") != stable_hash(report)
        or not hmac.compare_digest(
            _canonical_bytes(report), _canonical_bytes(recomputed_report)
        )
        or report["promoted"] is not True
        or payload.get("challenger_promoted") is not True
        or payload.get("outcome_used_to_change_action_evaluator_threshold_or_cohort")
        is not False
        or payload.get("same_source_replay_authorized") is not False
    ):
        raise HoVerLifecycleStoreError("A_hold promotion binding drifted")
    return dict(payload)


def load_a_hold_promotion(*, project: Path) -> dict[str, Any]:
    root = _project_root(project)
    payload, _raw = _read_json_file(
        root=root,
        relative=PROMOTION_RELATIVE,
        field="A_hold promotion",
        mode=0o644,
    )
    return _validate_promotion(payload=payload, project=root)


def create_a_hold_promotion_once(
    *, project: Path, outcome_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Authorize M only from the controller's frozen, exact positive report."""

    root = _project_root(project)
    report = _canonical_a_hold_report(outcome_report)
    freeze = load_f_search_policy_freeze(project=root)
    archive, binding = load_stage_output_archive(project=root, block="A_hold")
    seal = load_action_seal(project=root, block="A_hold")
    recomputed_report = _recompute_a_hold_report_from_evidence(
        project=root, freeze=freeze, archive=archive
    )
    if not hmac.compare_digest(
        _canonical_bytes(report), _canonical_bytes(recomputed_report)
    ):
        raise HoVerLifecycleStoreError(
            "controller A_hold report differs from sealed evidence recomputation"
        )
    if report["promoted"] is not True:
        raise HoVerLifecycleStoreError("nonpromotion cannot authorize M_search")
    payload = _with_self_hash(
        {
            "schema": _freeze_schema("a_hold_promotion"),
            "version": VERSION,
            "status": "A_hold_challenger_promoted",
            "acquisition_sha256": archive["acquisition_sha256"],
            "f_policy_freeze_sha256": freeze["policy_freeze_sha256"],
            "a_hold_archive_file_sha256": binding["file_sha256"],
            "a_hold_archive_semantic_sha256": binding["semantic_sha256"],
            "a_hold_action_seal_sha256": seal["action_seal_sha256"],
            "e0_action_id": freeze["e0_action_id"],
            "e0_policy_sha256": freeze["e0_policy_sha256"],
            "e1_action_id": freeze["e1_action_id"],
            "e1_policy_sha256": freeze["e1_policy_sha256"],
            "outcome_report": report,
            "outcome_report_sha256": stable_hash(report),
            "challenger_promoted": True,
            "outcome_used_to_change_action_evaluator_threshold_or_cohort": False,
            "same_source_replay_authorized": False,
        },
        "promotion_sha256",
    )
    _validate_promotion(payload=payload, project=root)
    _write_json_exclusive(
        root=root, relative=PROMOTION_RELATIVE, payload=payload, mode=0o644
    )
    return load_a_hold_promotion(project=root)


_M_SEARCH_REPORT_KEYS = {
    "l5_passed",
    "l5_delta_total",
    "l5_signflip_p",
    "e1_minus_hippo_delta_total",
    "e1_minus_hippo_signflip_p",
    "e1_minus_hippo_stratum_deltas",
    "e1_minus_raw_delta_total",
    "e1_minus_raw_signflip_p",
    "e1_complete_count",
    "raw_complete_count",
}


def _canonical_m_search_report(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _M_SEARCH_REPORT_KEYS:
        raise HoVerLifecycleStoreError("M_search outcome report schema drifted")
    report = dict(value)
    if (
        type(report["l5_passed"]) is not bool
        or type(report["e1_complete_count"]) is not int
        or type(report["raw_complete_count"]) is not int
        or not 0 <= report["e1_complete_count"] <= BLOCK_COUNTS["M_search"]
        or not 0 <= report["raw_complete_count"] <= BLOCK_COUNTS["M_search"]
    ):
        raise HoVerLifecycleStoreError("M_search outcome scalar drifted")
    for field in (
        "l5_delta_total",
        "l5_signflip_p",
        "e1_minus_hippo_delta_total",
        "e1_minus_hippo_signflip_p",
        "e1_minus_raw_delta_total",
        "e1_minus_raw_signflip_p",
    ):
        if not isinstance(_decode_number(report[field], field), Fraction):
            raise HoVerLifecycleStoreError(f"{field} must be an exact fraction")
    strata = report["e1_minus_hippo_stratum_deltas"]
    if not isinstance(strata, Mapping) or set(strata) != {
        "2_hop",
        "3_hop",
        "4_hop",
    }:
        raise HoVerLifecycleStoreError("M_search hop-stratum report drifted")
    for hop, value_row in strata.items():
        if not isinstance(_decode_number(value_row, str(hop)), Fraction):
            raise HoVerLifecycleStoreError("M_search stratum delta is not exact")
    _canonical_bytes(report)
    return report


def _recompute_m_search_report_from_evidence(
    *,
    project: Path,
    freeze: Mapping[str, Any],
    promotion: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute M/L5 from frozen policies, sealed outputs, and late labels."""

    context = _load_stage_acquisition_context(project=project, block="M_search")
    _validate_context(context, block="M_search")
    if archive.get("acquisition_sha256") != context.acquisition_sha256:
        raise HoVerLifecycleStoreError("M_search report acquisition binding drifted")
    e0_policy = _decode_policy(
        freeze.get("e0_policy"), evaluator_id="E0_INDEPENDENT_V2"
    )
    e1_policy = _decode_policy(
        freeze.get("e1_policy"), evaluator_id="E1_CAUSAL_NECESSITY_V2"
    )
    if (
        e0_policy.action_id == e1_policy.action_id
        or promotion.get("e0_action_id") != e0_policy.action_id
        or promotion.get("e0_policy_sha256") != e0_policy.selection_sha256
        or promotion.get("e1_action_id") != e1_policy.action_id
        or promotion.get("e1_policy_sha256") != e1_policy.selection_sha256
    ):
        raise HoVerLifecycleStoreError("promoted M_search policy binding drifted")
    joined = _load_joined_late_labels(
        project=project, context=context, expected_block="M_search"
    )
    records = archive.get("records")
    if not isinstance(records, list) or len(records) != len(joined):
        raise HoVerLifecycleStoreError("M_search archived evidence count drifted")
    e0_values: list[Fraction] = []
    e1_values: list[Fraction] = []
    hippo_values: list[Fraction] = []
    raw_values: list[Fraction] = []
    stratum_deltas = {
        "2_hop": Fraction(0),
        "3_hop": Fraction(0),
        "4_hop": Fraction(0),
    }
    e1_complete = 0
    raw_complete = 0
    for ordinal, (record, (stratum, gold)) in enumerate(
        zip(records, joined, strict=True)
    ):
        if not isinstance(record, Mapping) or record.get("ordinal") != ordinal:
            raise HoVerLifecycleStoreError("M_search archived record order drifted")
        envelopes = record.get("agent_action_traces")
        if not isinstance(envelopes, list) or len(envelopes) != len(AGENT_ACTION_IDS):
            raise HoVerLifecycleStoreError("M_search six-action matrix drifted")
        traces = {
            action_id: _decode_trace(envelope, expected_action_id=action_id)
            for action_id, envelope in zip(
                AGENT_ACTION_IDS, envelopes, strict=True
            )
        }
        raw_output = _validate_method_output(
            record.get("raw_output"), method="RAW"
        )["output_top5"]
        hippo_output = _validate_method_output(
            record.get("hipporag_output"), method="HippoRAG"
        )["output_top5"]
        e0_output = traces[e0_policy.action_id].output_top5
        e1_output = traces[e1_policy.action_id].output_top5
        try:
            e0_value = item_utility(e0_output, gold)
            e1_value = item_utility(e1_output, gold)
            hippo_value = item_utility(hippo_output, gold)
            raw_value = item_utility(raw_output, gold)
        except MultiHopRAGTypedOperatorV2Error as exc:
            raise HoVerLifecycleStoreError("M_search utility recomputation failed") from exc
        e0_values.append(e0_value)
        e1_values.append(e1_value)
        hippo_values.append(hippo_value)
        raw_values.append(raw_value)
        stratum_deltas[stratum] += e1_value - hippo_value
        e1_complete += int(set(gold) <= set(e1_output))
        raw_complete += int(set(gold) <= set(raw_output))
    try:
        l5 = paired_utility_summary(e1_values, e0_values)
        hippo_summary = paired_utility_summary(e1_values, hippo_values)
        raw_summary = paired_utility_summary(e1_values, raw_values)
    except MultiHopRAGTypedOperatorV2Error as exc:
        raise HoVerLifecycleStoreError("M_search paired summary failed") from exc
    return _canonical_m_search_report(
        {
            "l5_passed": l5.delta_total > 0
            and l5.exact_one_sided_p <= Fraction(1, 10),
            "l5_delta_total": _number(l5.delta_total),
            "l5_signflip_p": _number(l5.exact_one_sided_p),
            "e1_minus_hippo_delta_total": _number(
                hippo_summary.delta_total
            ),
            "e1_minus_hippo_signflip_p": _number(
                hippo_summary.exact_one_sided_p
            ),
            "e1_minus_hippo_stratum_deltas": {
                stratum: _number(stratum_deltas[stratum])
                for stratum in ("2_hop", "3_hop", "4_hop")
            },
            "e1_minus_raw_delta_total": _number(raw_summary.delta_total),
            "e1_minus_raw_signflip_p": _number(
                raw_summary.exact_one_sided_p
            ),
            "e1_complete_count": e1_complete,
            "raw_complete_count": raw_complete,
        }
    )


def recompute_m_search_outcome_report(*, project: Path) -> dict[str, Any]:
    """Read sealed M evidence and independently reproduce its exact report."""

    root = _project_root(project)
    freeze = load_f_search_policy_freeze(project=root)
    promotion = load_a_hold_promotion(project=root)
    archive, _binding = load_stage_output_archive(project=root, block="M_search")
    # The late-label path is not reached until all M actions are sealed.
    load_action_seal(project=root, block="M_search")
    return _recompute_m_search_report_from_evidence(
        project=root,
        freeze=freeze,
        promotion=promotion,
        archive=archive,
    )


def validate_m_search_outcome_report(
    *,
    project: Path,
    outcome_report: Mapping[str, Any],
    l5_passed: bool,
) -> dict[str, Any]:
    """Require the controller's M report and separate L5 flag to match evidence."""

    if type(l5_passed) is not bool:
        raise HoVerLifecycleStoreError("controller L5 flag is not boolean")
    reported = _canonical_m_search_report(outcome_report)
    recomputed = recompute_m_search_outcome_report(project=project)
    if (
        not hmac.compare_digest(
            _canonical_bytes(reported), _canonical_bytes(recomputed)
        )
        or l5_passed is not recomputed["l5_passed"]
    ):
        raise HoVerLifecycleStoreError(
            "controller M_search report differs from sealed evidence recomputation"
        )
    return reported


# Compatibility names used by the controller adapter while keeping the public
# terminology explicit: the A_form receipt freezes evaluators and their
# descriptive selections, not the later F policies.
load_a_form_policy_freeze = load_a_form_evaluator_freeze
create_a_form_policy_freeze_once = create_a_form_evaluator_freeze_once
load_committed_promotion_authorization = load_a_hold_promotion


__all__ = [
    "ACTION_SEAL_RELATIVES",
    "AGENT_ACTION_IDS",
    "A_FORM_EVALUATOR_FREEZE_RELATIVE",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "CORPUS_SIZE",
    "EVALUATOR_DEFINITIONS",
    "F_POLICY_FREEZE_RELATIVE",
    "HoVerLifecycleStoreError",
    "PROMOTION_RELATIVE",
    "PRIVATE_ROOT_RELATIVE",
    "STAGE_OUTPUT_ARCHIVE_RELATIVES",
    "STAGE_RUNTIME_BINDING_KEYS",
    "StageAcquisitionContext",
    "VERSION",
    "build_stage_output_record",
    "create_a_form_evaluator_freeze_once",
    "create_a_form_policy_freeze_once",
    "create_a_hold_promotion_once",
    "create_action_seal_once",
    "create_f_search_policy_freeze_once",
    "create_stage_output_archive_once",
    "lifecycle_output_paths",
    "load_a_form_evaluator_freeze",
    "load_a_form_policy_freeze",
    "load_a_hold_promotion",
    "load_action_seal",
    "load_committed_promotion_authorization",
    "load_f_search_policy_freeze",
    "load_stage_output_archive",
    "preflight_lifecycle_outputs_absent",
    "recompute_a_hold_outcome_report",
    "recompute_m_search_outcome_report",
    "stable_hash",
    "validate_m_search_outcome_report",
    "validate_stage_output_record",
]

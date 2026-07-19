"""Single preregistered crash recovery for the interrupted ERASER EI study.

This module is deliberately outside the v1 implementation freeze.  It does
not resume or reuse any partial HippoRAG work.  Instead it verifies the frozen
v1 implementation and the preregistered incident, atomically archives the
entire interrupted formal root, clones only the five byte-identical base
acquisition custody files into a fresh canonical root, and invokes the frozen
v1 lifecycle through two guarded call-site substitutions:

* the source qualifier returns the already archived aggregate qualification;
* ``acquire_once`` verifies the cloned acquisition state without running its
  original body, generating a secret, or selecting a cohort.

The scheduler's physical HippoRAG worker cap is temporarily reduced to the
preregistered recovery-only value of two.  The complete ``3 * n`` future
submission barrier and every retrieval/scoring rule remain unchanged.  All
three substituted symbols are restored in ``finally``.  Any failure after the
v2 marker is terminal; there is no second recovery attempt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import (
    eraser_evidence_inference_direct_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_formal_controller_v1 as formal_controller,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_local_runtime_v1 as local_runtime,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_source_qualification_v1 as source_qualification,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_three_arm_scheduler_v1 as scheduler,
)


VERSION = "eraser_evidence_inference_crash_recovery_v2"

INCIDENT_RELATIVE = Path(
    "manifests/"
    "eraser_evidence_inference_formal_v1_hard_interruption_incident_v1.json"
)
RECOVERY_DESIGN_RELATIVE = Path(
    "manifests/eraser_evidence_inference_crash_recovery_design_v2.json"
)
RECOVERY_IMPLEMENTATION_FREEZE_RELATIVE = Path(
    "manifests/eraser_evidence_inference_crash_recovery_implementation_freeze_v2.json"
)
BASE_IMPLEMENTATION_FREEZE_RELATIVE = (
    formal_controller.FULL_IMPLEMENTATION_FREEZE_RELATIVE
)
BASE_DESIGN_RELATIVE = formal_controller.DESIGN_RELATIVE

RECOVERY_ROOT_RELATIVE = Path(
    "artifacts/eraser_evidence_inference_r7_e3_crash_recovery_v2"
)
ARCHIVE_DESTINATION_RELATIVE = RECOVERY_ROOT_RELATIVE / "v1_crash_archive"
CANONICAL_FORMAL_ROOT_RELATIVE = local_runtime.FORMAL_ROOT_RELATIVE
RECOVERY_CONTROLLER_DIRECTORY = "controller"
CANONICAL_ACQUISITION_DIRECTORY = formal_controller.ACQUISITION_DIRECTORY

MARKER_FILENAME = "recovery.one_shot_marker.json"
ARCHIVE_RECEIPT_FILENAME = "recovery.atomic_archive.receipt.json"
CLONE_RECEIPT_FILENAME = "recovery.base_acquisition_clone.receipt.json"
PREFLIGHT_FILENAME = "runtime.preflight.receipt.json"
RESULT_FILENAME = "recovery.terminal_result.json"
FAILURE_FILENAME = "recovery.terminal_failure.json"

EXPECTED_INCIDENT_SHA256 = (
    "292808da871299732989115936e82256453317cbf9470de675e5e0c5fa73c5cb"
)
EXPECTED_INCIDENT_FILE_SHA256 = (
    "e8181ea5df2d2a0687fb167784c2c89963212d71dbd4d29141bdedee578bcb45"
)
EXPECTED_RECOVERY_DESIGN_SHA256 = (
    "deb7d31a36bbed8dbd2a10f1e68bad0a89192c4b0addbac122035a6297c834b8"
)
EXPECTED_RECOVERY_DESIGN_FILE_SHA256 = (
    "941d8a36ff86b0f0d2d03cf249fc09ad72d55dd9adc36670376def3c3e3ae219"
)
EXPECTED_BASE_DESIGN_SHA256 = acquisition.FORMAL_DESIGN_SHA256
EXPECTED_BASE_DESIGN_FILE_SHA256 = (
    "22e166b4b749cad7c7280dcaf39cb86fdf9e4b30884597203d28a3bf06435954"
)
EXPECTED_BASE_IMPLEMENTATION_FREEZE_SHA256 = (
    "d76397b6bbcb0ecd306b6a619ccd6120939dfc4c8deaff60e133687a54993fd4"
)
EXPECTED_BASE_IMPLEMENTATION_FREEZE_FILE_SHA256 = (
    "d57809aab6f9f2613b98c5f075ba252a380cdf1154866f67b75b3bebb40d564b"
)
EXPECTED_CRASH_TREE_SHA256 = (
    "cf1a864955889d7723a23e669c74eb3dc3478e4e23b2e8f33fb92a4096bfb436"
)
EXPECTED_TREE_ENTRY_COUNT = 499
EXPECTED_TREE_DIRECTORY_COUNT = 358
EXPECTED_TREE_REGULAR_FILE_COUNT = 141
EXPECTED_TREE_REGULAR_FILE_BYTES = 12_393_656
EXPECTED_TREE_CANONICAL_BYTES = 155_661

BASE_HIPPORAG_WORKER_CAP = 32
RECOVERY_HIPPORAG_WORKER_CAP = 2
COPY_CHUNK_BYTES = 1024 * 1024

CLONED_ACQUISITION_RELATIVE_PATHS = (
    "acquisition.marker.private.json",
    "assignment.private.json",
    "acquisition.receipt.json",
    "views/A_form.private.json",
    "views/F_search.private.json",
)

RECOVERY_IMPLEMENTATION_ROLE_PATHS = {
    "crash_recovery_controller": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_crash_recovery_v2.py"
    ),
    "test_crash_recovery_controller": (
        "tests/test_eraser_evidence_inference_crash_recovery_v2.py"
    ),
}


class EraserEvidenceInferenceCrashRecoveryError(RuntimeError):
    """A preregistration, custody, archive, replay, or terminal edge drifted."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery payload is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    try:
        return formal_controller._require_sha256(value, field)
    except formal_controller.EraserEvidenceInferenceFormalControllerError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} is not a lowercase SHA256"
        ) from exc


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery self-hash field already exists"
        )
    return {**dict(body), field: stable_hash(dict(body))}


def _verify_self_hash(
    payload: Mapping[str, Any], *, schema: str, field: str
) -> str:
    body = dict(payload)
    declared = _require_sha256(body.pop(field, None), field)
    if payload.get("schema") != schema or not hmac.compare_digest(
        stable_hash(body), declared
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} self-hash drifted"
        )
    return declared


def _regular_nonsymlink(path: Path, field: str, *, mode: int | None = None) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} is unavailable"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} is not a safe regular file"
        )
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} mode drifted"
        )
    return metadata


def _private_directory(path: Path, field: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} is not a private 0700 directory"
        )
    return metadata


def _sha256_regular_file(
    path: Path, field: str, *, required_mode: int | None = None
) -> str:
    _regular_nonsymlink(path, field, mode=required_mode)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} cannot be hashed"
        ) from exc
    return digest.hexdigest()


def _read_json(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    _regular_nonsymlink(path, field)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} is not strict JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} is not a JSON object"
        )
    return payload, raw


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery relative path is invalid"
        )
    path = PurePosixPath(value)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or any(part == ".." for part in parts):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery relative path is unsafe"
        )
    return PurePosixPath(*parts).as_posix()


def _canonical_project(project_root: str | Path) -> Path:
    try:
        return formal_controller._canonical_project(project_root)
    except formal_controller.EraserEvidenceInferenceFormalControllerError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "project root is unavailable"
        ) from exc


@dataclass(frozen=True)
class CrashTreeSnapshot:
    tree_sha256: str
    canonical_json_byte_count: int
    descendant_entry_count: int
    descendant_regular_file_count: int
    descendant_directory_count: int
    descendant_regular_file_total_bytes: int
    root_device: int
    root_inode: int

    def __post_init__(self) -> None:
        _require_sha256(self.tree_sha256, "crash tree")
        for value, field in (
            (self.canonical_json_byte_count, "tree canonical byte count"),
            (self.descendant_entry_count, "tree entry count"),
            (self.descendant_regular_file_count, "tree file count"),
            (self.descendant_directory_count, "tree directory count"),
            (self.descendant_regular_file_total_bytes, "tree file bytes"),
            (self.root_device, "tree root device"),
            (self.root_inode, "tree root inode"),
        ):
            if type(value) is not int or value < 0:
                raise EraserEvidenceInferenceCrashRecoveryError(
                    f"{field} is invalid"
                )

    def aggregate_payload(self) -> dict[str, Any]:
        return {
            "tree_sha256": self.tree_sha256,
            "canonical_json_byte_count": self.canonical_json_byte_count,
            "descendant_entry_count": self.descendant_entry_count,
            "descendant_regular_file_count": self.descendant_regular_file_count,
            "descendant_directory_count": self.descendant_directory_count,
            "descendant_regular_file_total_bytes": (
                self.descendant_regular_file_total_bytes
            ),
        }


def snapshot_private_tree(root: Path) -> CrashTreeSnapshot:
    """Hash the interrupted tree as bytes/metadata without parsing payloads."""

    root_meta = _private_directory(root, "interrupted formal root")
    rows: list[dict[str, Any]] = []
    try:
        descendants = tuple(root.rglob("*"))
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "interrupted tree cannot be enumerated"
        ) from exc
    for path in descendants:
        try:
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            relative.encode("utf-8", errors="strict")
        except (OSError, ValueError, UnicodeEncodeError) as exc:
            raise EraserEvidenceInferenceCrashRecoveryError(
                "interrupted tree entry cannot be inspected"
            ) from exc
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            entry_type = "directory"
            file_sha256: str | None = None
        elif stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            entry_type = "regular_file"
            file_sha256 = _sha256_regular_file(path, "interrupted tree file")
        else:
            raise EraserEvidenceInferenceCrashRecoveryError(
                "interrupted tree contains a symlink or special file"
            )
        rows.append(
            {
                "relative_path": relative,
                "type": entry_type,
                "mode": stat.S_IMODE(metadata.st_mode),
                "size": metadata.st_size,
                "file_sha256": file_sha256,
            }
        )
    rows.sort(key=lambda row: str(row["relative_path"]).encode("utf-8"))
    raw = canonical_bytes(rows)
    return CrashTreeSnapshot(
        tree_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_json_byte_count=len(raw),
        descendant_entry_count=len(rows),
        descendant_regular_file_count=sum(
            row["type"] == "regular_file" for row in rows
        ),
        descendant_directory_count=sum(
            row["type"] == "directory" for row in rows
        ),
        descendant_regular_file_total_bytes=sum(
            int(row["size"])
            for row in rows
            if row["type"] == "regular_file"
        ),
        root_device=root_meta.st_dev,
        root_inode=root_meta.st_ino,
    )


@dataclass(frozen=True)
class RecoveryPrerequisites:
    base_freeze: Mapping[str, Any]
    base_freeze_file_sha256: str
    recovery_freeze: Mapping[str, Any]
    recovery_freeze_file_sha256: str
    incident: Mapping[str, Any]
    design: Mapping[str, Any]
    crash_tree: CrashTreeSnapshot

    def __post_init__(self) -> None:
        _require_sha256(
            self.base_freeze_file_sha256, "base freeze prerequisite file"
        )
        _require_sha256(
            self.recovery_freeze_file_sha256,
            "recovery freeze prerequisite file",
        )


def verify_recovery_implementation_freeze(
    *, project: Path, freeze_path: Path
) -> dict[str, Any]:
    """Verify the v2 controller/test freeze created after synthetic tests."""

    payload, raw = _read_json(freeze_path, "recovery implementation freeze")
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_implementation_freeze",
        field="implementation_freeze_sha256",
    )
    rows = payload.get("implementation_binding")
    files = rows.get("files") if isinstance(rows, Mapping) else None
    if (
        payload.get("version") != "v2"
        or payload.get("status")
        != "frozen_before_recovery_marker_or_archive_transition"
        or payload.get("recovery_design_sha256")
        != EXPECTED_RECOVERY_DESIGN_SHA256
        or payload.get("recovery_design_file_sha256")
        != EXPECTED_RECOVERY_DESIGN_FILE_SHA256
        or payload.get("incident_sha256") != EXPECTED_INCIDENT_SHA256
        or payload.get("incident_file_sha256")
        != EXPECTED_INCIDENT_FILE_SHA256
        or payload.get("base_design_sha256")
        != EXPECTED_BASE_DESIGN_SHA256
        or payload.get("base_design_file_sha256")
        != EXPECTED_BASE_DESIGN_FILE_SHA256
        or payload.get("base_implementation_freeze_sha256")
        != EXPECTED_BASE_IMPLEMENTATION_FREEZE_SHA256
        or payload.get("base_implementation_freeze_file_sha256")
        != EXPECTED_BASE_IMPLEMENTATION_FREEZE_FILE_SHA256
        or payload.get("required_role_registry")
        != list(RECOVERY_IMPLEMENTATION_ROLE_PATHS)
        or not isinstance(files, list)
        or len(files) != len(RECOVERY_IMPLEMENTATION_ROLE_PATHS)
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery implementation freeze semantics drifted"
        )
    observed: list[str] = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "role",
            "relative_path",
            "sha256",
        }:
            raise EraserEvidenceInferenceCrashRecoveryError(
                "recovery implementation file row drifted"
            )
        role = row.get("role")
        if role not in RECOVERY_IMPLEMENTATION_ROLE_PATHS or role in observed:
            raise EraserEvidenceInferenceCrashRecoveryError(
                "recovery implementation role drifted"
            )
        relative = _safe_relative_path(row.get("relative_path"))
        expected_relative = RECOVERY_IMPLEMENTATION_ROLE_PATHS[str(role)]
        digest = _require_sha256(row.get("sha256"), "recovery frozen file")
        if (
            relative != expected_relative
            or _sha256_regular_file(
                project / relative, f"recovery frozen {role}"
            )
            != digest
        ):
            raise EraserEvidenceInferenceCrashRecoveryError(
                "recovery implementation role/path/hash drifted"
            )
        observed.append(str(role))
    if observed != list(RECOVERY_IMPLEMENTATION_ROLE_PATHS):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery implementation role order drifted"
        )
    tests = payload.get("synthetic_test_receipt")
    if (
        not isinstance(tests, Mapping)
        or type(tests.get("collected_case_count")) is not int
        or tests.get("collected_case_count", 0) <= 0
        or tests.get("passed_case_count") != tests.get("collected_case_count")
        or tests.get("real_source_or_benchmark_item_read") is not False
        or tests.get("model_inference_calls") != 0
        or tests.get("online_or_network_calls") != 0
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery implementation synthetic test receipt drifted"
        )
    if declared != payload["implementation_freeze_sha256"]:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery implementation freeze identity drifted"
        )
    if hashlib.sha256(raw).hexdigest() != _sha256_regular_file(
        freeze_path, "recovery implementation freeze"
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery implementation freeze changed while verifying"
        )
    return payload


def _load_preregistered_manifests(
    project: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    incident_path = project / INCIDENT_RELATIVE
    design_path = project / RECOVERY_DESIGN_RELATIVE
    if (
        _sha256_regular_file(incident_path, "incident manifest")
        != EXPECTED_INCIDENT_FILE_SHA256
        or _sha256_regular_file(design_path, "recovery design")
        != EXPECTED_RECOVERY_DESIGN_FILE_SHA256
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "incident or recovery design file binding drifted"
        )
    incident, _incident_raw = _read_json(incident_path, "incident manifest")
    design, _design_raw = _read_json(design_path, "recovery design")
    if (
        _verify_self_hash(
            incident,
            schema="eraser_evidence_inference_formal_v1_hard_interruption_incident_v1",
            field="incident_sha256",
        )
        != EXPECTED_INCIDENT_SHA256
        or _verify_self_hash(
            design,
            schema="eraser_evidence_inference_crash_recovery_design_v2",
            field="recovery_design_sha256",
        )
        != EXPECTED_RECOVERY_DESIGN_SHA256
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "incident or recovery design identity drifted"
        )
    return incident, design


def _verify_preregistered_semantics(
    *, incident: Mapping[str, Any], design: Mapping[str, Any]
) -> None:
    binding = design.get("binding")
    authorization = design.get("authorization")
    custody = design.get("base_acquisition_custody")
    guarded = design.get("guarded_frozen_lifecycle_replay")
    parallel = design.get("parallel_execution")
    transition = design.get("root_transition")
    incident_binding = incident.get("binding")
    snapshot = incident.get("crash_snapshot")
    tree = snapshot.get("tree_digest") if isinstance(snapshot, Mapping) else None
    absence = incident.get("observed_absence")
    if (
        design.get("status")
        != "preregistered_result_blind_single_crash_recovery_before_any_recovery_action"
        or incident.get("status")
        != "recorded_post_assignment_pre_label_pre_score_hard_interruption"
        or not all(
            isinstance(value, Mapping)
            for value in (
                binding,
                authorization,
                custody,
                guarded,
                parallel,
                transition,
                incident_binding,
                snapshot,
                tree,
                absence,
            )
        )
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery preregistration shape drifted"
        )
    assert isinstance(binding, Mapping)
    assert isinstance(authorization, Mapping)
    assert isinstance(custody, Mapping)
    assert isinstance(guarded, Mapping)
    assert isinstance(parallel, Mapping)
    assert isinstance(transition, Mapping)
    assert isinstance(incident_binding, Mapping)
    assert isinstance(snapshot, Mapping)
    assert isinstance(tree, Mapping)
    assert isinstance(absence, Mapping)
    if (
        binding.get("incident_sha256") != EXPECTED_INCIDENT_SHA256
        or binding.get("incident_file_sha256")
        != EXPECTED_INCIDENT_FILE_SHA256
        or binding.get("base_design_sha256")
        != acquisition.FORMAL_DESIGN_SHA256
        or binding.get("base_implementation_freeze_sha256")
        != incident_binding.get("implementation_freeze_sha256")
        or transition.get("crash_tree_sha256")
        != EXPECTED_CRASH_TREE_SHA256
        or transition.get("canonical_formal_root_relative")
        != CANONICAL_FORMAL_ROOT_RELATIVE.as_posix()
        or transition.get("archive_destination_relative")
        != ARCHIVE_DESTINATION_RELATIVE.as_posix()
        or transition.get("old_root_transition")
        != "single_same_filesystem_os_rename_then_fsync_both_parents"
        or transition.get("residual_stage_or_cache_reused") is not False
        or parallel.get("hipporag_physical_worker_cap")
        != RECOVERY_HIPPORAG_WORKER_CAP
        or parallel.get("all_initial_3n_logical_futures_submitted_before_first_result")
        is not True
        or guarded.get("original_acquire_once_body_called") is not False
        or guarded.get("original_qualification_builder_called") is not False
        or guarded.get("runtime_symbol_and_worker_cap_restoration_required_in_finally")
        is not True
        or authorization.get("recovery_attempt_index") != 1
        or authorization.get("new_secret_generation_authorized") is not False
        or authorization.get("partial_work_cache_action_or_result_reuse_authorized")
        is not False
        or authorization.get("qualification_original_body_call_count_authorized")
        != 0
        or authorization.get("acquire_once_original_body_call_count_authorized")
        != 0
        or authorization.get("recovery_after_v2_marker_failure_authorized")
        is not False
        or custody.get("cloned_relative_paths")
        != list(CLONED_ACQUISITION_RELATIVE_PATHS)
        or custody.get("exact_clone_file_count")
        != len(CLONED_ACQUISITION_RELATIVE_PATHS)
        or custody.get("source_qualification_reexecution") is not False
        or custody.get("source_resampling_or_member_selection") is not False
        or tree.get("tree_sha256") != EXPECTED_CRASH_TREE_SHA256
        or tree.get("schema") != "sorted_descendant_entry_rows_v1"
        or snapshot.get("descendant_entry_count") != EXPECTED_TREE_ENTRY_COUNT
        or snapshot.get("descendant_regular_file_count")
        != EXPECTED_TREE_REGULAR_FILE_COUNT
        or snapshot.get("descendant_directory_count")
        != EXPECTED_TREE_DIRECTORY_COUNT
        or snapshot.get("descendant_regular_file_total_bytes")
        != EXPECTED_TREE_REGULAR_FILE_BYTES
        or snapshot.get("canonical_tree_json_byte_count")
        != EXPECTED_TREE_CANONICAL_BYTES
        or snapshot.get("single_item_output_or_result_count") != 0
        or absence.get("persisted_schedule_receipt_count") != 0
        or absence.get("persisted_three_arm_archive_or_receipt_count") != 0
        or absence.get("public_or_private_scored_outcome_count") != 0
        or absence.get("terminal_failure_receipt_count") != 0
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery preregistration semantics drifted"
        )


def verify_recovery_prerequisites(*, project: Path) -> RecoveryPrerequisites:
    """Verify every immutable prerequisite before the v2 marker or archive."""

    recovery_root = project / RECOVERY_ROOT_RELATIVE
    if os.path.lexists(recovery_root):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery root already exists; a second attempt is forbidden"
        )
    base_freeze_path = project / BASE_IMPLEMENTATION_FREEZE_RELATIVE
    try:
        base_freeze = formal_controller.verify_full_implementation_freeze(
            project=project,
            freeze_path=base_freeze_path,
        )
    except formal_controller.EraserEvidenceInferenceFormalControllerError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "base v1 full implementation freeze drifted"
        ) from exc
    base_freeze_file_sha256 = _sha256_regular_file(
        base_freeze_path, "base implementation freeze"
    )
    if (
        base_freeze.get("implementation_freeze_sha256")
        != EXPECTED_BASE_IMPLEMENTATION_FREEZE_SHA256
        or base_freeze_file_sha256
        != EXPECTED_BASE_IMPLEMENTATION_FREEZE_FILE_SHA256
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "base v1 implementation freeze identity drifted"
        )
    incident, design = _load_preregistered_manifests(project)
    _verify_preregistered_semantics(incident=incident, design=design)
    base_design_path = project / BASE_DESIGN_RELATIVE
    base_design, base_design_raw = _read_json(base_design_path, "base design")
    try:
        base_design_sha256 = formal_controller._verify_self_hash(
            base_design,
            schema=None,
            field="design_sha256",
        )
    except formal_controller.EraserEvidenceInferenceFormalControllerError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "base design self-hash drifted"
        ) from exc
    if (
        base_design_sha256 != EXPECTED_BASE_DESIGN_SHA256
        or hashlib.sha256(base_design_raw).hexdigest()
        != EXPECTED_BASE_DESIGN_FILE_SHA256
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "base design file binding drifted"
        )
    recovery_freeze_path = project / RECOVERY_IMPLEMENTATION_FREEZE_RELATIVE
    recovery_freeze = verify_recovery_implementation_freeze(
        project=project,
        freeze_path=recovery_freeze_path,
    )
    recovery_freeze_file_sha256 = _sha256_regular_file(
        recovery_freeze_path, "recovery implementation freeze"
    )
    # No interrupted private tree byte is opened before both implementation
    # freezes and their exact file identities have passed.
    crash_tree = snapshot_private_tree(project / CANONICAL_FORMAL_ROOT_RELATIVE)
    if crash_tree.aggregate_payload() != {
        "tree_sha256": EXPECTED_CRASH_TREE_SHA256,
        "canonical_json_byte_count": EXPECTED_TREE_CANONICAL_BYTES,
        "descendant_entry_count": EXPECTED_TREE_ENTRY_COUNT,
        "descendant_regular_file_count": EXPECTED_TREE_REGULAR_FILE_COUNT,
        "descendant_directory_count": EXPECTED_TREE_DIRECTORY_COUNT,
        "descendant_regular_file_total_bytes": EXPECTED_TREE_REGULAR_FILE_BYTES,
    }:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "interrupted v1 tree differs from the preregistered crash snapshot"
        )
    base_binding = design["binding"]
    incident_binding = incident["binding"]
    if (
        base_freeze.get("implementation_freeze_sha256")
        != base_binding["base_implementation_freeze_sha256"]
        or base_freeze_file_sha256
        != base_binding["base_implementation_freeze_file_sha256"]
        or incident_binding.get("private_assignment_sha256")
        != design["base_acquisition_custody"]["same_private_assignment_sha256"]
        or incident_binding.get("public_acquisition_receipt_sha256")
        != design["base_acquisition_custody"]["same_public_receipt_sha256"]
        or incident_binding.get("selection_secret_commitment_sha256")
        != design["base_acquisition_custody"][
            "same_selection_secret_commitment_sha256"
        ]
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "base freeze or acquisition preregistration binding drifted"
        )
    return RecoveryPrerequisites(
        base_freeze=base_freeze,
        base_freeze_file_sha256=base_freeze_file_sha256,
        recovery_freeze=recovery_freeze,
        recovery_freeze_file_sha256=recovery_freeze_file_sha256,
        incident=incident,
        design=design,
        crash_tree=crash_tree,
    )


def _ensure_new_private_directory(path: Path, field: str) -> None:
    if os.path.lexists(path):
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} already exists"
        )
    try:
        os.mkdir(path, 0o700)
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} cannot be created"
        ) from exc
    _private_directory(path, field)


def _persist_typed(
    *, path: Path, payload: Mapping[str, Any], schema: str, field: str
) -> formal_controller.PersistedArtifact:
    _verify_self_hash(payload, schema=schema, field=field)
    try:
        return formal_controller._persist_typed_artifact(
            path=path,
            payload=payload,
            schema=schema,
            field=field,
            expected_sha256=str(payload[field]),
        )
    except formal_controller.EraserEvidenceInferenceFormalControllerError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "recovery typed artifact cannot be persisted"
        ) from exc


def _recovery_marker_payload(
    *,
    project: Path,
    prerequisites: RecoveryPrerequisites,
) -> dict[str, Any]:
    return _self_hashed(
        {
            "schema": f"{VERSION}_one_shot_marker",
            "version": "v2",
            "status": "crash_recovery_started_failure_is_terminal",
            "recovery_attempt_index": 1,
            "project_identity_sha256": hashlib.sha256(
                os.fsencode(project)
            ).hexdigest(),
            "incident_sha256": EXPECTED_INCIDENT_SHA256,
            "incident_file_sha256": EXPECTED_INCIDENT_FILE_SHA256,
            "recovery_design_sha256": EXPECTED_RECOVERY_DESIGN_SHA256,
            "recovery_design_file_sha256": EXPECTED_RECOVERY_DESIGN_FILE_SHA256,
            "base_implementation_freeze_sha256": prerequisites.base_freeze[
                "implementation_freeze_sha256"
            ],
            "base_implementation_freeze_file_sha256": (
                prerequisites.base_freeze_file_sha256
            ),
            "recovery_implementation_freeze_sha256": prerequisites.recovery_freeze[
                "implementation_freeze_sha256"
            ],
            "recovery_implementation_freeze_file_sha256": (
                prerequisites.recovery_freeze_file_sha256
            ),
            "crash_tree": prerequisites.crash_tree.aggregate_payload(),
            "archive_destination_relative": ARCHIVE_DESTINATION_RELATIVE.as_posix(),
            "cloned_acquisition_relative_paths": list(
                CLONED_ACQUISITION_RELATIVE_PATHS
            ),
            "hipporag_physical_worker_cap": RECOVERY_HIPPORAG_WORKER_CAP,
            "qualification_original_body_call_count_authorized": 0,
            "acquire_once_original_body_call_count_authorized": 0,
            "new_secret_generation_resample_or_source_selection_authorized": False,
            "partial_work_cache_action_or_result_reuse_authorized": False,
            "second_recovery_attempt_authorized": False,
            "test_access_authorized": False,
            "online_or_external_evaluation_authorized": False,
        },
        "recovery_marker_sha256",
    )


def _fsync_directory(path: Path, field: str) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            f"{field} cannot be fsynced"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_archive_interrupted_root(
    *,
    project: Path,
    recovery_root: Path,
    snapshot: CrashTreeSnapshot,
) -> dict[str, Any]:
    source = project / CANONICAL_FORMAL_ROOT_RELATIVE
    destination = project / ARCHIVE_DESTINATION_RELATIVE
    source_meta = _private_directory(source, "interrupted formal root")
    recovery_meta = _private_directory(recovery_root, "recovery root")
    if (
        source_meta.st_dev != recovery_meta.st_dev
        or source_meta.st_dev != snapshot.root_device
        or source_meta.st_ino != snapshot.root_inode
        or os.path.lexists(destination)
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "atomic archive precondition drifted"
        )
    try:
        os.rename(source, destination)
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "single atomic interrupted-root archive transition failed"
        ) from exc
    destination_meta = _private_directory(destination, "archived interrupted root")
    if (
        destination_meta.st_dev != snapshot.root_device
        or destination_meta.st_ino != snapshot.root_inode
        or os.path.lexists(source)
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "atomic archive inode identity drifted"
        )
    _fsync_directory(source.parent, "archive source parent")
    _fsync_directory(destination.parent, "archive destination parent")
    post_rename = snapshot_private_tree(destination)
    if (
        post_rename.aggregate_payload() != snapshot.aggregate_payload()
        or post_rename.root_device != snapshot.root_device
        or post_rename.root_inode != snapshot.root_inode
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "post-rename archived tree differs from the preregistered snapshot"
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_atomic_archive_receipt",
            "version": "v2",
            "status": "interrupted_v1_root_atomically_archived_once",
            "source_relative": CANONICAL_FORMAL_ROOT_RELATIVE.as_posix(),
            "destination_relative": ARCHIVE_DESTINATION_RELATIVE.as_posix(),
            "tree": post_rename.aggregate_payload(),
            "root_device": snapshot.root_device,
            "root_inode": snapshot.root_inode,
            "same_filesystem_atomic_os_rename_count": 1,
            "both_parent_directories_fsynced": True,
            "post_rename_tree_verified_before_clone_or_qualification_read": True,
            "archive_mutation_after_transition_authorized": False,
            "partial_work_cache_action_or_result_reuse": False,
        },
        "archive_receipt_sha256",
    )


def _clone_file_raw_exclusive(source: Path, destination: Path) -> dict[str, Any]:
    source_meta = _regular_nonsymlink(source, "base acquisition clone source", mode=0o600)
    if os.path.lexists(destination):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "base acquisition clone destination already exists"
        )
    source_flags = os.O_RDONLY
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        destination_flags |= os.O_NOFOLLOW
    source_fd: int | None = None
    destination_fd: int | None = None
    source_digest = hashlib.sha256()
    destination_digest = hashlib.sha256()
    byte_count = 0
    try:
        source_fd = os.open(source, source_flags)
        destination_fd = os.open(destination, destination_flags, 0o600)
        os.fchmod(destination_fd, 0o600)
        while True:
            chunk = os.read(source_fd, COPY_CHUNK_BYTES)
            if not chunk:
                break
            source_digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("zero-length clone write")
                destination_digest.update(view[:written])
                byte_count += written
                view = view[written:]
        os.fsync(destination_fd)
    except OSError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "raw O_EXCL acquisition clone failed"
        ) from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
    destination_meta = _regular_nonsymlink(
        destination, "base acquisition clone destination", mode=0o600
    )
    if (
        byte_count != source_meta.st_size
        or destination_meta.st_size != source_meta.st_size
        or source_digest.hexdigest() != destination_digest.hexdigest()
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "raw acquisition clone byte identity drifted"
        )
    return {
        "size": byte_count,
        "file_sha256": source_digest.hexdigest(),
        "source_mode": stat.S_IMODE(source_meta.st_mode),
        "destination_mode": stat.S_IMODE(destination_meta.st_mode),
    }


def _recreate_and_clone_base_acquisition(
    *, project: Path, recovery_root: Path
) -> tuple[Path, dict[str, Any]]:
    del recovery_root
    formal_root = project / CANONICAL_FORMAL_ROOT_RELATIVE
    _ensure_new_private_directory(formal_root, "fresh canonical formal root")
    acquisition_root = formal_root / CANONICAL_ACQUISITION_DIRECTORY
    _ensure_new_private_directory(acquisition_root, "fresh acquisition root")
    views_root = acquisition_root / "views"
    _ensure_new_private_directory(views_root, "fresh acquisition views root")
    archive_acquisition = (
        project / ARCHIVE_DESTINATION_RELATIVE / CANONICAL_ACQUISITION_DIRECTORY
    )
    rows: list[dict[str, Any]] = []
    for relative in CLONED_ACQUISITION_RELATIVE_PATHS:
        source = archive_acquisition / PurePosixPath(relative)
        destination = acquisition_root / PurePosixPath(relative)
        binding = _clone_file_raw_exclusive(source, destination)
        rows.append({"relative_path": relative, **binding})
    _fsync_directory(views_root, "fresh acquisition views root")
    _fsync_directory(acquisition_root, "fresh acquisition root")
    _fsync_directory(formal_root, "fresh canonical formal root")
    receipt = _self_hashed(
        {
            "schema": f"{VERSION}_base_acquisition_clone_receipt",
            "version": "v2",
            "status": "exact_five_base_acquisition_files_cloned",
            "clone_method": "raw_bytes_O_EXCL_mode_0600_fsync",
            "source_root_relative": (
                ARCHIVE_DESTINATION_RELATIVE / CANONICAL_ACQUISITION_DIRECTORY
            ).as_posix(),
            "destination_root_relative": (
                CANONICAL_FORMAL_ROOT_RELATIVE
                / CANONICAL_ACQUISITION_DIRECTORY
            ).as_posix(),
            "files": rows,
            "exact_clone_file_count": len(rows),
            "controller_selection_secret_parsed": False,
            "new_secret_generation_resample_or_source_selection": False,
            "partial_work_cache_action_or_result_reused": False,
        },
        "clone_receipt_sha256",
    )
    return acquisition_root, receipt


def _load_archived_qualification(
    *, project: Path, prerequisites: RecoveryPrerequisites
) -> dict[str, Any]:
    path = (
        project
        / ARCHIVE_DESTINATION_RELATIVE
        / formal_controller.CONTROLLER_DIRECTORY
        / formal_controller.QUALIFICATION_FILENAME
    )
    payload, raw = _read_json(path, "archived aggregate qualification")
    declared = _verify_self_hash(
        payload,
        schema=source_qualification.SCHEMA,
        field="qualification_sha256",
    )
    incident_binding = prerequisites.incident["binding"]
    if (
        declared != incident_binding["qualification_sha256"]
        or hashlib.sha256(raw).hexdigest()
        != incident_binding["qualification_file_sha256"]
        or payload.get("status") != "passed_source_qualification_no_selection"
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "archived aggregate qualification binding drifted"
        )
    return payload


@dataclass
class _SubstitutionAudit:
    qualification_shim_call_count: int = 0
    acquisition_shim_call_count: int = 0
    original_qualification_body_call_count: int = 0
    original_acquire_once_body_call_count: int = 0
    symbols_restored: bool = False


def _run_guarded_frozen_lifecycle(
    *,
    project: Path,
    controller_root: Path,
    acquisition_root: Path,
    runtime_config: local_runtime.FormalRuntimeConfig,
    recovery_marker: formal_controller.PersistedArtifact,
    preflight_artifact: formal_controller.PersistedArtifact,
    archived_qualification: Mapping[str, Any],
    stage_state: dict[str, str],
) -> tuple[dict[str, Any], _SubstitutionAudit]:
    """Invoke the frozen v1 orchestration with two exact guarded shims."""

    original_qualification = source_qualification.build_formal_qualification
    original_acquire_once = acquisition.acquire_once
    original_worker_cap = scheduler.HIPPORAG_WORKER_CAP
    audit = _SubstitutionAudit()
    expected_qualification_path = controller_root / formal_controller.QUALIFICATION_FILENAME

    def qualification_shim(call_project: str | Path) -> dict[str, Any]:
        if _canonical_project(call_project) != project:
            raise EraserEvidenceInferenceCrashRecoveryError(
                "guarded qualification call project drifted"
            )
        audit.qualification_shim_call_count += 1
        if audit.qualification_shim_call_count != 1:
            raise EraserEvidenceInferenceCrashRecoveryError(
                "guarded qualification call count drifted"
            )
        return dict(archived_qualification)

    def acquisition_shim(**kwargs: Any) -> dict[str, Any]:
        expected = {
            "archive_path": project / formal_controller.ARCHIVE_RELATIVE,
            "prompt_sidecar_path": project / formal_controller.PROMPT_SIDECAR_RELATIVE,
            "qualification_receipt_path": expected_qualification_path,
            "design_path": project / BASE_DESIGN_RELATIVE,
            "implementation_freeze_path": project / BASE_IMPLEMENTATION_FREEZE_RELATIVE,
            "project_root": project,
            "acquisition_root": acquisition_root,
            "selection_secret": None,
            "enforce_formal_design_identity": True,
        }
        if set(kwargs) != set(expected) or any(
            kwargs[key] != value for key, value in expected.items()
        ):
            raise EraserEvidenceInferenceCrashRecoveryError(
                "guarded acquisition call arguments drifted"
            )
        audit.acquisition_shim_call_count += 1
        if audit.acquisition_shim_call_count != 1:
            raise EraserEvidenceInferenceCrashRecoveryError(
                "guarded acquisition call count drifted"
            )
        return acquisition.verify_acquisition_state(
            acquisition_root=acquisition_root,
            qualification_receipt_path=expected_qualification_path,
            design_path=project / BASE_DESIGN_RELATIVE,
            implementation_freeze_path=(
                project / BASE_IMPLEMENTATION_FREEZE_RELATIVE
            ),
            project_root=project,
            enforce_formal_design_identity=True,
        )

    if original_worker_cap != BASE_HIPPORAG_WORKER_CAP:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "frozen scheduler base worker cap drifted before recovery substitution"
        )
    try:
        # The guard begins before the first global mutation.  Even a failure
        # during a later assignment restores every original symbol/value.
        source_qualification.build_formal_qualification = qualification_shim
        acquisition.acquire_once = acquisition_shim
        scheduler.HIPPORAG_WORKER_CAP = RECOVERY_HIPPORAG_WORKER_CAP
        inner = formal_controller._run_started_lifecycle(
            project=project,
            controller_root=controller_root,
            acquisition_root=acquisition_root,
            runtime_config=runtime_config,
            lifecycle_marker=recovery_marker,
            preflight_artifact=preflight_artifact,
            stage_state=stage_state,
        )
    finally:
        source_qualification.build_formal_qualification = original_qualification
        acquisition.acquire_once = original_acquire_once
        scheduler.HIPPORAG_WORKER_CAP = original_worker_cap
        audit.symbols_restored = (
            source_qualification.build_formal_qualification
            is original_qualification
            and acquisition.acquire_once is original_acquire_once
            and scheduler.HIPPORAG_WORKER_CAP == original_worker_cap
        )
    if (
        audit.qualification_shim_call_count != 1
        or audit.acquisition_shim_call_count != 1
        or audit.original_qualification_body_call_count != 0
        or audit.original_acquire_once_body_call_count != 0
        or not audit.symbols_restored
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "guarded lifecycle substitution audit drifted"
        )
    if not isinstance(inner, dict):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "frozen lifecycle returned no terminal mapping"
        )
    return inner, audit


def _terminal_result_payload(
    *,
    recovery_marker: formal_controller.PersistedArtifact,
    prerequisites: RecoveryPrerequisites,
    archive_artifact: formal_controller.PersistedArtifact,
    clone_artifact: formal_controller.PersistedArtifact,
    preflight_artifact: formal_controller.PersistedArtifact,
    inner: Mapping[str, Any],
    inner_path: Path,
    audit: _SubstitutionAudit,
) -> dict[str, Any]:
    try:
        inner_sha = formal_controller._verify_self_hash(
            inner,
            schema=f"{formal_controller.VERSION}_terminal_result",
            field="terminal_result_sha256",
        )
    except formal_controller.EraserEvidenceInferenceFormalControllerError as exc:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "inner frozen lifecycle terminal binding drifted"
        ) from exc
    inner_file_sha = _sha256_regular_file(
        inner_path, "inner frozen lifecycle terminal", required_mode=0o600
    )
    if inner_file_sha != hashlib.sha256(
        formal_controller.canonical_bytes(dict(inner))
    ).hexdigest():
        raise EraserEvidenceInferenceCrashRecoveryError(
            "inner frozen lifecycle terminal file drifted"
        )
    claims = inner.get("claims")
    if not isinstance(claims, Mapping):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "inner frozen lifecycle claims are absent"
        )
    inner_status = inner.get("status")
    promoted = inner_status == "complete_promoted_untouched_M_search_measured"
    if inner_status not in {
        "complete_promoted_untouched_M_search_measured",
        "complete_nonpromotion_M_search_unopened",
    }:
        raise EraserEvidenceInferenceCrashRecoveryError(
            "inner frozen lifecycle terminal status drifted"
        )
    m_fields = (
        "M_search_label_free_view_sha256",
        "M_search_schedule_receipt_sha256",
        "M_search_label_capability_sha256",
        "M_search_label_pack_sha256",
        "M_search_score_receipt_sha256",
        "M_search_score_receipt",
    )
    if (
        inner.get("M_search_opened_without_promotion") is not False
        or (not promoted and any(inner.get(field) is not None for field in m_fields))
        or bool(claims.get("evaluator_promoted")) is not promoted
    ):
        raise EraserEvidenceInferenceCrashRecoveryError(
            "inner frozen lifecycle M_search branch drifted"
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_terminal_result",
            "version": "v2",
            "status": (
                "complete_crash_replay_promoted_M_search_measured"
                if promoted
                else "complete_crash_replay_nonpromotion_M_search_unopened"
            ),
            "recovery_marker_sha256": recovery_marker.self_sha256,
            "incident_sha256": EXPECTED_INCIDENT_SHA256,
            "recovery_design_sha256": EXPECTED_RECOVERY_DESIGN_SHA256,
            "recovery_implementation_freeze_sha256": prerequisites.recovery_freeze[
                "implementation_freeze_sha256"
            ],
            "archive_receipt_sha256": archive_artifact.self_sha256,
            "clone_receipt_sha256": clone_artifact.self_sha256,
            "runtime_preflight_receipt_sha256": preflight_artifact.self_sha256,
            "inner_v1_shaped_terminal_sha256": inner_sha,
            "inner_v1_shaped_terminal_file_sha256": inner_file_sha,
            "inner_v1_shaped_terminal_semantics": (
                "frozen_lifecycle_replay_component_not_pristine_v1_completion"
            ),
            "pristine_v1_completion_claim_allowed": False,
            "recovery_result_interpretation": (
                "single_preregistered_post_assignment_pre_label_pre_score_"
                "crash_recovery_with_identical_cohort_and_frozen_scoring"
            ),
            "claims": dict(claims),
            "M_search_opened": promoted,
            "M_search_opened_without_promotion": False,
            "same_private_assignment_sha256": prerequisites.design[
                "base_acquisition_custody"
            ]["same_private_assignment_sha256"],
            "same_selection_secret_commitment_sha256": prerequisites.design[
                "base_acquisition_custody"
            ]["same_selection_secret_commitment_sha256"],
            "qualification_shim_call_count": audit.qualification_shim_call_count,
            "acquisition_shim_call_count": audit.acquisition_shim_call_count,
            "qualification_original_body_call_count": (
                audit.original_qualification_body_call_count
            ),
            "acquire_once_original_body_call_count": (
                audit.original_acquire_once_body_call_count
            ),
            "substituted_symbols_and_worker_cap_restored": audit.symbols_restored,
            "hipporag_physical_worker_cap_during_replay": (
                RECOVERY_HIPPORAG_WORKER_CAP
            ),
            "partial_work_cache_action_or_result_reused": False,
            "new_secret_generation_resample_or_source_selection": False,
            "online_evaluator_calls": 0,
            "external_network_calls": 0,
            "test_access_authorized_or_performed": False,
        },
        "recovery_terminal_result_sha256",
    )


def _persist_terminal_failure(
    *,
    controller_root: Path,
    recovery_marker_sha256: str,
    stage: str,
    error: BaseException,
) -> None:
    failure_path = controller_root / FAILURE_FILENAME
    result_path = controller_root / RESULT_FILENAME
    if os.path.lexists(failure_path) or os.path.lexists(result_path):
        return
    message = str(error).encode("utf-8", errors="backslashreplace")
    payload = _self_hashed(
        {
            "schema": f"{VERSION}_terminal_failure",
            "version": "v2",
            "status": "terminal_crash_recovery_failure_no_further_attempt",
            "recovery_marker_sha256": recovery_marker_sha256,
            "incident_sha256": EXPECTED_INCIDENT_SHA256,
            "recovery_design_sha256": EXPECTED_RECOVERY_DESIGN_SHA256,
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(message).hexdigest(),
            "exception_message_persisted": False,
            "retry_replay_resample_replacement_or_secret_rotation_authorized": False,
            "test_access_authorized": False,
            "online_or_external_evaluation_calls": 0,
        },
        "recovery_terminal_failure_sha256",
    )
    try:
        _persist_typed(
            path=failure_path,
            payload=payload,
            schema=f"{VERSION}_terminal_failure",
            field="recovery_terminal_failure_sha256",
        )
    except EraserEvidenceInferenceCrashRecoveryError:
        return


def run_crash_recovery(*, project_root: str | Path) -> dict[str, Any]:
    """Execute the sole preregistered recovery attempt."""

    project = _canonical_project(project_root)
    prerequisites = verify_recovery_prerequisites(project=project)
    recovery_root = project / RECOVERY_ROOT_RELATIVE
    recovery_controller = recovery_root / RECOVERY_CONTROLLER_DIRECTORY
    _ensure_new_private_directory(recovery_root, "recovery root")
    _ensure_new_private_directory(recovery_controller, "recovery controller")
    marker_payload = _recovery_marker_payload(
        project=project, prerequisites=prerequisites
    )
    marker_artifact = _persist_typed(
        path=recovery_controller / MARKER_FILENAME,
        payload=marker_payload,
        schema=f"{VERSION}_one_shot_marker",
        field="recovery_marker_sha256",
    )
    stage_state = {"name": "atomic_archive_interrupted_v1_root"}
    try:
        archive_payload = _atomic_archive_interrupted_root(
            project=project,
            recovery_root=recovery_root,
            snapshot=prerequisites.crash_tree,
        )
        archive_artifact = _persist_typed(
            path=recovery_controller / ARCHIVE_RECEIPT_FILENAME,
            payload=archive_payload,
            schema=f"{VERSION}_atomic_archive_receipt",
            field="archive_receipt_sha256",
        )

        stage_state["name"] = "recreate_and_clone_base_acquisition"
        acquisition_root, clone_payload = _recreate_and_clone_base_acquisition(
            project=project,
            recovery_root=recovery_root,
        )
        clone_artifact = _persist_typed(
            path=recovery_controller / CLONE_RECEIPT_FILENAME,
            payload=clone_payload,
            schema=f"{VERSION}_base_acquisition_clone_receipt",
            field="clone_receipt_sha256",
        )

        stage_state["name"] = "offline_runtime_preflight"
        runtime_config = local_runtime.default_formal_runtime_config(project)
        raw_preflight = local_runtime.preflight_formal_runtime_config(runtime_config)
        preflight_payload = formal_controller._runtime_preflight_receipt(
            raw_preflight
        )
        preflight_artifact = formal_controller._persist_typed_artifact(
            path=recovery_controller / PREFLIGHT_FILENAME,
            payload=preflight_payload,
            schema=f"{formal_controller.VERSION}_runtime_preflight_receipt",
            field="runtime_preflight_receipt_sha256",
            expected_sha256=preflight_payload[
                "runtime_preflight_receipt_sha256"
            ],
        )

        stage_state["name"] = "load_archived_qualification"
        archived_qualification = _load_archived_qualification(
            project=project, prerequisites=prerequisites
        )

        stage_state["name"] = "guarded_frozen_lifecycle_replay"
        inner, audit = _run_guarded_frozen_lifecycle(
            project=project,
            controller_root=recovery_controller,
            acquisition_root=acquisition_root,
            runtime_config=runtime_config,
            recovery_marker=marker_artifact,
            preflight_artifact=preflight_artifact,
            archived_qualification=archived_qualification,
            stage_state=stage_state,
        )

        stage_state["name"] = "v2_terminal_wrap"
        inner_path = recovery_controller / formal_controller.RESULT_FILENAME
        result_payload = _terminal_result_payload(
            recovery_marker=marker_artifact,
            prerequisites=prerequisites,
            archive_artifact=archive_artifact,
            clone_artifact=clone_artifact,
            preflight_artifact=preflight_artifact,
            inner=inner,
            inner_path=inner_path,
            audit=audit,
        )
        _persist_typed(
            path=recovery_controller / RESULT_FILENAME,
            payload=result_payload,
            schema=f"{VERSION}_terminal_result",
            field="recovery_terminal_result_sha256",
        )
        return result_payload
    except BaseException as exc:
        _persist_terminal_failure(
            controller_root=recovery_controller,
            recovery_marker_sha256=marker_artifact.self_sha256,
            stage=stage_state["name"],
            error=exc,
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise EraserEvidenceInferenceCrashRecoveryError(
            "crash recovery failed terminally"
        ) from exc


def _public_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "recovery_terminal_result_sha256": result.get(
            "recovery_terminal_result_sha256"
        ),
        "claims": result.get("claims"),
        "pristine_v1_completion_claim_allowed": result.get(
            "pristine_v1_completion_claim_allowed"
        ),
        "online_evaluator_calls": result.get("online_evaluator_calls"),
        "external_network_calls": result.get("external_network_calls"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_crash_recovery(project_root=arguments.project_root)
    except EraserEvidenceInferenceCrashRecoveryError:
        print(
            json.dumps(
                {"status": "terminal_error_see_private_recovery_failure_receipt"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(_public_summary(result), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_DESTINATION_RELATIVE",
    "CANONICAL_FORMAL_ROOT_RELATIVE",
    "CLONED_ACQUISITION_RELATIVE_PATHS",
    "CrashTreeSnapshot",
    "EraserEvidenceInferenceCrashRecoveryError",
    "RECOVERY_HIPPORAG_WORKER_CAP",
    "RECOVERY_IMPLEMENTATION_FREEZE_RELATIVE",
    "RECOVERY_ROOT_RELATIVE",
    "VERSION",
    "canonical_bytes",
    "main",
    "run_crash_recovery",
    "snapshot_private_tree",
    "stable_hash",
    "verify_recovery_implementation_freeze",
    "verify_recovery_prerequisites",
]

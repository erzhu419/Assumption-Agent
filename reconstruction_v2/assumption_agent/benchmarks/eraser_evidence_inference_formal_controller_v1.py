"""One-shot formal controller for the frozen ERASER EI R7/E3 study.

The controller verifies the full implementation freeze before source access,
runs the aggregate-only qualifier once, delegates the sole private cohort
assignment to ``direct_acquisition``, and executes each available block through
the eager ``3 * n`` offline scheduler.  Gold is opened only after the matching
three-arm archive and exact feature matrix are durably sealed.  ``M_search`` is
never inspected unless the typed A_hold score object promotes E3.

There is no retry or recovery path.  Once the one-shot marker exists, any
failure is terminal for this source epoch.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from assumption_agent.benchmarks import (
    eraser_evidence_inference_direct_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_local_runtime_v1 as local_runtime,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_e3_runner_v1 as runner,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_source_qualification_v1 as source_qualification,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_three_arm_scheduler_v1 as scheduler,
)


VERSION = "eraser_evidence_inference_formal_controller_v1"
FORMAL_ROOT_RELATIVE = local_runtime.FORMAL_ROOT_RELATIVE
CONTROLLER_DIRECTORY = "controller"
ACQUISITION_DIRECTORY = "acquisition"
QUALIFICATION_FILENAME = "source_qualification.receipt.json"
MARKER_FILENAME = "lifecycle.one_shot_marker.json"
FAILURE_FILENAME = "lifecycle.terminal_failure.json"
RESULT_FILENAME = "lifecycle.terminal_result.json"
FULL_IMPLEMENTATION_FREEZE_RELATIVE = Path(
    "manifests/eraser_evidence_inference_full_implementation_freeze_v1.json"
)
DESIGN_RELATIVE = Path(
    "manifests/eraser_evidence_inference_r7_e3_design_v1.json"
)
ARCHIVE_RELATIVE = source_qualification.FORMAL_ARCHIVE_RELATIVE_PATH
PROMPT_SIDECAR_RELATIVE = (
    source_qualification.FORMAL_PROMPT_SIDECAR_RELATIVE_PATH
)

EXPECTED_ROLE_PATHS = {
    "source_qualifier": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_source_qualification_v1.py"
    ),
    "direct_acquisition": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_direct_acquisition_v1.py"
    ),
    "local_runtime": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_local_runtime_v1.py"
    ),
    "three_arm_scheduler": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_three_arm_scheduler_v1.py"
    ),
    "r7_operator": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_r7_operator_v1.py"
    ),
    "exact_feature_bridge": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_exact_feature_bridge_v1.py"
    ),
    "e3_runner": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_r7_e3_runner_v1.py"
    ),
    "formal_controller": (
        "assumption_agent/benchmarks/"
        "eraser_evidence_inference_formal_controller_v1.py"
    ),
    "test_source_qualifier": (
        "tests/test_eraser_evidence_inference_source_qualification_v1.py"
    ),
    "test_direct_acquisition": (
        "tests/test_eraser_evidence_inference_direct_acquisition_v1.py"
    ),
    "test_local_runtime": (
        "tests/test_eraser_evidence_inference_local_runtime_v1.py"
    ),
    "test_three_arm_scheduler": (
        "tests/test_eraser_evidence_inference_three_arm_scheduler_v1.py"
    ),
    "test_r7_operator": (
        "tests/test_eraser_evidence_inference_r7_operator_v1.py"
    ),
    "test_exact_feature_bridge": (
        "tests/test_eraser_evidence_inference_exact_feature_bridge_v1.py"
    ),
    "test_e3_runner": (
        "tests/test_eraser_evidence_inference_r7_e3_runner_v1.py"
    ),
    "hipporag_freeze_manifest": (
        "manifests/eraser_evidence_inference_hipporag_implementation_freeze_v1.json"
    ),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class EraserEvidenceInferenceFormalControllerError(RuntimeError):
    """A freeze, capability, execution, label order, or one-shot edge drifted."""


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
        raise EraserEvidenceInferenceFormalControllerError(
            "controller payload is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EraserEvidenceInferenceFormalControllerError(
            f"{field} is not a lowercase SHA256"
        )
    return value


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise EraserEvidenceInferenceFormalControllerError(
            "self-hash field already exists"
        )
    return {**dict(body), field: stable_hash(dict(body))}


def _verify_self_hash(
    payload: Mapping[str, Any], *, schema: str | None, field: str
) -> str:
    body = dict(payload)
    declared = _require_sha256(body.pop(field, None), field)
    if (schema is not None and payload.get("schema") != schema) or not hmac.compare_digest(
        stable_hash(body), declared
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            f"{field} self-hash drifted"
        )
    return declared


def _canonical_project(project_root: str | Path) -> Path:
    try:
        lexical = Path(project_root).expanduser().absolute()
        if lexical.is_symlink():
            raise EraserEvidenceInferenceFormalControllerError(
                "project root is a symlink"
            )
        project = lexical.resolve(strict=True)
    except (
        EraserEvidenceInferenceFormalControllerError,
        local_runtime.EraserEvidenceInferenceLocalRuntimeError,
    ):
        raise
    except (OSError, RuntimeError) as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            "project root is unavailable"
        ) from exc
    if not project.is_dir():
        raise EraserEvidenceInferenceFormalControllerError(
            "project root is not a directory"
        )
    return project


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise EraserEvidenceInferenceFormalControllerError(
            "freeze path is invalid"
        )
    path = PurePosixPath(value)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or any(part == ".." for part in parts):
        raise EraserEvidenceInferenceFormalControllerError(
            "freeze path is unsafe"
        )
    return PurePosixPath(*parts).as_posix()


def _regular_nonsymlink(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            f"{field} is unavailable"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EraserEvidenceInferenceFormalControllerError(
            f"{field} is not a safe regular file"
        )


def _sha256_file(path: Path, field: str) -> str:
    _regular_nonsymlink(path, field)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            f"{field} cannot be hashed"
        ) from exc
    return digest.hexdigest()


def _strict_json_file(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    _regular_nonsymlink(path, field)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            f"{field} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise EraserEvidenceInferenceFormalControllerError(
            f"{field} is not an object"
        )
    return value, raw


def verify_full_implementation_freeze(
    *, project: Path, freeze_path: Path
) -> dict[str, Any]:
    """Verify the exact role-to-path registry before any source access."""

    payload, _raw = _strict_json_file(freeze_path, "full implementation freeze")
    _verify_self_hash(
        payload,
        schema=acquisition.IMPLEMENTATION_FREEZE_SCHEMA,
        field=acquisition.IMPLEMENTATION_FREEZE_SELF_HASH_FIELD,
    )
    binding = payload.get("implementation_binding")
    rows = binding.get("files") if isinstance(binding, Mapping) else None
    required_roles = list(acquisition.REQUIRED_IMPLEMENTATION_ROLE_REGISTRY)
    if (
        payload.get("status")
        != "frozen_before_source_qualification_or_private_assignment"
        or payload.get("design_sha256") != acquisition.FORMAL_DESIGN_SHA256
        or payload.get("required_role_registry") != required_roles
        or set(EXPECTED_ROLE_PATHS) != set(required_roles)
        or not isinstance(rows, list)
        or len(rows) != len(required_roles)
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "full implementation freeze semantics drifted"
        )
    observed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "role",
            "sha256",
        }:
            raise EraserEvidenceInferenceFormalControllerError(
                "full implementation freeze file row drifted"
            )
        role = row.get("role")
        relative = _safe_relative(row.get("relative_path"))
        digest = _require_sha256(row.get("sha256"), "frozen file hash")
        if (
            role not in EXPECTED_ROLE_PATHS
            or relative != EXPECTED_ROLE_PATHS[str(role)]
            or str(role) in observed
            or _sha256_file(project / relative, f"frozen {role}") != digest
        ):
            raise EraserEvidenceInferenceFormalControllerError(
                "full implementation freeze role/path/hash drifted"
            )
        observed[str(role)] = digest
    if tuple(observed) != tuple(required_roles):
        raise EraserEvidenceInferenceFormalControllerError(
            "full implementation freeze role order drifted"
        )
    tests = payload.get("synthetic_test_receipt")
    if (
        not isinstance(tests, Mapping)
        or type(tests.get("collected_case_count")) is not int
        or tests.get("collected_case_count", 0) <= 0
        or tests.get("passed_case_count") != tests.get("collected_case_count")
        or tests.get("real_source_or_benchmark_item_read") is not False
        or tests.get("online_or_network_calls") != 0
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "full implementation freeze test receipt drifted"
        )
    return payload


def _ensure_private_directory(path: Path, *, create: bool) -> None:
    if create:
        try:
            path.mkdir(mode=0o700)
        except OSError as exc:
            raise EraserEvidenceInferenceFormalControllerError(
                "private controller directory cannot be created"
            ) from exc
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            "private controller directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "private controller directory mode drifted"
        )


def _write_exclusive(
    path: Path, payload: Mapping[str, Any], *, mode: int = 0o600
) -> str:
    if mode not in {0o600, 0o644}:
        raise EraserEvidenceInferenceFormalControllerError(
            "artifact mode is invalid"
        )
    raw = canonical_bytes(dict(payload))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            "exclusive controller artifact write failed"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _persist_self_hashed(
    controller_root: Path,
    filename: str,
    payload: Mapping[str, Any],
    *,
    schema: str,
    field: str,
) -> str:
    _verify_self_hash(payload, schema=schema, field=field)
    return _write_exclusive(controller_root / filename, payload)


def _view_to_runtime_rows(
    view: Mapping[str, Any], *, block: str
) -> tuple[local_runtime.ItemTextView, ...]:
    if (
        view.get("block") != block
        or view.get("item_count") != runner.BLOCK_COUNTS[block]
        or not isinstance(view.get("items"), list)
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "verified label-free view shape drifted"
        )
    rows: list[local_runtime.ItemTextView] = []
    for ordinal, item in enumerate(view["items"]):
        if (
            not isinstance(item, Mapping)
            or set(item) != {
                "block_ordinal",
                "item_commitment_sha256",
                "payload",
            }
            or item.get("block_ordinal") != ordinal
            or not isinstance(item.get("payload"), Mapping)
        ):
            raise EraserEvidenceInferenceFormalControllerError(
                "verified view row drifted"
            )
        payload = item["payload"]
        ico = payload.get("official_ico")
        sentences = payload.get("sentence_tokens")
        if (
            set(payload) != {"query", "official_ico", "sentence_tokens"}
            or not isinstance(ico, Mapping)
            or set(ico) != {"Intervention", "Comparator", "Outcome"}
            or not isinstance(sentences, list)
        ):
            raise EraserEvidenceInferenceFormalControllerError(
                "verified exact text view drifted"
            )
        try:
            row = local_runtime.ItemTextView(
                item_commitment_sha256=str(item["item_commitment_sha256"]),
                query=payload["query"],
                intervention=ico["Intervention"],
                comparator=ico["Comparator"],
                outcome=ico["Outcome"],
                official_tokenized_sentences=tuple(
                    tuple(sentence) for sentence in sentences
                ),
            )
        except (KeyError, TypeError, local_runtime.EraserEvidenceInferenceLocalRuntimeError) as exc:
            raise EraserEvidenceInferenceFormalControllerError(
                "verified view cannot form an exact runtime item"
            ) from exc
        rows.append(row)
    return tuple(rows)


def _labels_from_pack(
    pack: Mapping[str, Any], *, block: str
) -> tuple[runner.AnchorLabel, ...]:
    if (
        block not in {"A_form", "A_hold", "M_search"}
        or pack.get("block") != block
        or pack.get("item_count") != runner.BLOCK_COUNTS[block]
        or not isinstance(pack.get("items"), list)
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "verified label pack shape drifted"
        )
    try:
        return tuple(
            runner.AnchorLabel(
                item_commitment_sha256=row["item_commitment_sha256"],
                gold_ordinals=tuple(row["flattened_gold_sentence_ordinals"]),
                family=row["family"],
            )
            for row in pack["items"]
        )
    except (KeyError, TypeError, runner.EraserEvidenceInferenceRunnerError) as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            "verified label pack cannot form exact labels"
        ) from exc


@dataclass(frozen=True)
class PersistedBlockExecution:
    block: str
    artifact: scheduler.BlockThreeArmArtifact
    archive_file_sha256: str
    receipt_file_sha256: str

    @property
    def execution_seal_sha256(self) -> str:
        return _require_sha256(
            self.artifact.receipt["receipt_sha256"],
            f"{self.block} execution seal",
        )


@dataclass(frozen=True)
class PersistedArtifact:
    path: Path
    self_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.self_sha256, "persisted artifact self hash")
        _require_sha256(self.file_sha256, "persisted artifact file hash")
        if not isinstance(self.path, Path):
            raise EraserEvidenceInferenceFormalControllerError(
                "persisted artifact path type drifted"
            )


@dataclass(frozen=True)
class PersistedScheduleExecution:
    artifact: scheduler.ThreeArmScheduleArtifact
    schedule_receipt: PersistedArtifact
    blocks: Mapping[str, PersistedBlockExecution]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact, scheduler.ThreeArmScheduleArtifact)
            or tuple(self.blocks) != tuple(row.block for row in self.artifact.blocks)
            or self.schedule_receipt.self_sha256
            != self.artifact.receipt["schedule_receipt_sha256"]
        ):
            raise EraserEvidenceInferenceFormalControllerError(
                "persisted schedule binding drifted"
            )


def _persist_typed_artifact(
    *,
    path: Path,
    payload: Mapping[str, Any],
    schema: str,
    field: str,
    expected_sha256: str | None = None,
) -> PersistedArtifact:
    declared = _verify_self_hash(payload, schema=schema, field=field)
    if expected_sha256 is not None and not hmac.compare_digest(
        declared, _require_sha256(expected_sha256, field)
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "typed artifact differs from its in-memory seal"
        )
    file_sha = _write_exclusive(path, payload)
    return PersistedArtifact(
        path=path,
        self_sha256=declared,
        file_sha256=file_sha,
    )


def _persist_schedule(
    *, controller_root: Path, schedule: scheduler.ThreeArmScheduleArtifact
) -> PersistedScheduleExecution:
    schedule_receipt = _persist_typed_artifact(
        path=(
            controller_root
            / f"{'_'.join(row.block for row in schedule.blocks)}.schedule.receipt.json"
        ),
        payload=schedule.receipt,
        schema=f"{scheduler.VERSION}_schedule_receipt",
        field="schedule_receipt_sha256",
        expected_sha256=schedule.receipt["schedule_receipt_sha256"],
    )
    result: dict[str, PersistedBlockExecution] = {}
    for row in schedule.blocks:
        archive_file = _write_exclusive(
            controller_root / f"{row.block}.three_arm.archive.private.json",
            row.archive_payload,
        )
        receipt_file = _write_exclusive(
            controller_root / f"{row.block}.three_arm.receipt.json",
            row.receipt,
        )
        result[row.block] = PersistedBlockExecution(
            block=row.block,
            artifact=row,
            archive_file_sha256=archive_file,
            receipt_file_sha256=receipt_file,
        )
    return PersistedScheduleExecution(
        artifact=schedule,
        schedule_receipt=schedule_receipt,
        blocks=result,
    )


def _a_form_utility_deltas(
    *, execution: scheduler.BlockThreeArmArtifact, labels: Sequence[runner.AnchorLabel]
) -> dict[str, Fraction]:
    if execution.block != "A_form":
        raise EraserEvidenceInferenceFormalControllerError(
            "A_form utility input block drifted"
        )
    by_item = {row.item_commitment_sha256: row for row in labels}
    if len(by_item) != len(labels) or set(by_item) != set(
        execution.feature_seal.item_commitments
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "A_form label/feature alignment drifted"
        )
    deltas: dict[str, Fraction] = {}
    for trace in execution.feature_seal.traces:
        gold = by_item[trace.item_commitment_sha256].gold_ordinals
        r0 = runner.item_utility(trace.r0_top5, gold)[0]
        r7 = runner.item_utility(trace.r7_top5, gold)[0]
        deltas[trace.item_commitment_sha256] = r7 - r0
    return deltas


def _write_capability(
    controller_root: Path,
    filename: str,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> Path:
    _require_sha256(payload.get(field), field)
    path = controller_root / filename
    _write_exclusive(path, payload)
    return path


def _private_payload_file_binding(
    *, path: Path, payload: Mapping[str, Any], field: str
) -> dict[str, str]:
    return {
        field: _require_sha256(payload.get(field), field),
        f"{field.removesuffix('_sha256')}_file_sha256": _sha256_file(
            path, f"private {field}"
        ),
    }


def _self_hashed_file_binding(
    *, path: Path, schema: str, field: str
) -> dict[str, str]:
    payload, _raw = _strict_json_file(path, field)
    return {
        field: _verify_self_hash(payload, schema=schema, field=field),
        f"{field.removesuffix('_sha256')}_file_sha256": _sha256_file(
            path, field
        ),
    }


def _artifact_binding(value: PersistedArtifact) -> dict[str, str]:
    return {
        "self_sha256": value.self_sha256,
        "file_sha256": value.file_sha256,
    }


def _schedule_binding(
    value: PersistedScheduleExecution,
) -> dict[str, Any]:
    return {
        "schedule_receipt": _artifact_binding(value.schedule_receipt),
        "blocks": {
            block: {
                "archive_sha256": row.artifact.archive_payload[
                    "archive_sha256"
                ],
                "archive_file_sha256": row.archive_file_sha256,
                "execution_receipt_sha256": row.execution_seal_sha256,
                "execution_receipt_file_sha256": row.receipt_file_sha256,
                "feature_receipt_sha256": (
                    row.artifact.feature_seal.feature_receipt_sha256
                ),
                "hipporag_arm_receipt_sha256": (
                    row.artifact.hippo_arm_seal.hipporag_arm_receipt_sha256
                ),
                "raw_arm_receipt_sha256": (
                    row.artifact.raw_arm_seal.raw_arm_receipt_sha256
                ),
            }
            for block, row in value.blocks.items()
        },
    }


def _runtime_preflight_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        payload.get("schema") != local_runtime.PREFLIGHT_SCHEMA
        or payload.get("model_inference_calls") != 0
        or payload.get("benchmark_source_or_private_pack_reads") != 0
        or payload.get("external_network_calls") != 0
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "offline runtime preflight boundary drifted"
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_runtime_preflight_receipt",
            "version": VERSION,
            "status": "offline_runtime_preflight_passed_before_source",
            "runtime_preflight": dict(payload),
            "model_inference_calls": 0,
            "benchmark_source_or_private_pack_reads": 0,
            "external_network_calls": 0,
        },
        "runtime_preflight_receipt_sha256",
    )


def _create_fresh_formal_roots(
    *, project: Path
) -> tuple[Path, Path, Path]:
    formal_root = project / FORMAL_ROOT_RELATIVE
    if os.path.lexists(formal_root):
        raise EraserEvidenceInferenceFormalControllerError(
            "formal source epoch root already exists"
        )
    try:
        os.mkdir(formal_root, 0o700)
    except OSError as exc:
        raise EraserEvidenceInferenceFormalControllerError(
            "formal source epoch root cannot be created"
        ) from exc
    _ensure_private_directory(formal_root, create=False)
    controller_root = formal_root / CONTROLLER_DIRECTORY
    acquisition_root = formal_root / ACQUISITION_DIRECTORY
    _ensure_private_directory(controller_root, create=True)
    return formal_root, controller_root, acquisition_root


def _lifecycle_marker(
    *,
    project: Path,
    freeze: Mapping[str, Any],
    freeze_path: Path,
    design_path: Path,
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    design, _raw = _strict_json_file(design_path, "formal design")
    design_sha = _verify_self_hash(
        design,
        schema=None,
        field="design_sha256",
    )
    if design_sha != acquisition.FORMAL_DESIGN_SHA256:
        raise EraserEvidenceInferenceFormalControllerError(
            "formal design identity drifted"
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_one_shot_marker",
            "version": VERSION,
            "status": "formal_source_epoch_started_failure_is_terminal",
            "project_identity_sha256": hashlib.sha256(
                os.fsencode(project)
            ).hexdigest(),
            "design_sha256": design_sha,
            "design_file_sha256": _sha256_file(design_path, "formal design"),
            "implementation_freeze_sha256": freeze[
                acquisition.IMPLEMENTATION_FREEZE_SELF_HASH_FIELD
            ],
            "implementation_freeze_file_sha256": _sha256_file(
                freeze_path, "full implementation freeze"
            ),
            "runtime_preflight_receipt_sha256": preflight_receipt[
                "runtime_preflight_receipt_sha256"
            ],
            "qualification_run_count_authorized": 1,
            "private_selection_secret_generation_count_authorized": 1,
            "retry_replay_resample_replacement_or_secret_rotation_authorized": False,
            "test_access_authorized": False,
            "online_evaluation_authorized": False,
        },
        "lifecycle_marker_sha256",
    )


def _persist_label_capability(
    *,
    controller_root: Path,
    block: str,
    public: Mapping[str, Any],
    view: Mapping[str, Any],
    execution: PersistedBlockExecution,
) -> PersistedArtifact:
    capability = acquisition.build_label_capability(
        block=block,
        private_assignment_sha256=str(public["private_assignment_sha256"]),
        public_receipt_sha256=str(public["public_receipt_sha256"]),
        label_free_view_sha256=str(view["label_free_view_sha256"]),
        three_arm_execution_seal_sha256=execution.execution_seal_sha256,
        feature_seal_sha256=(
            execution.artifact.feature_seal.feature_receipt_sha256
        ),
    )
    return _persist_typed_artifact(
        path=controller_root / f"{block}.label.capability.json",
        payload=capability,
        schema=f"{acquisition.VERSION}_label_capability",
        field="label_capability_sha256",
        expected_sha256=capability["label_capability_sha256"],
    )


def _materialize_labels(
    *,
    project: Path,
    acquisition_root: Path,
    block: str,
    capability: PersistedArtifact,
) -> tuple[
    dict[str, Any],
    tuple[runner.AnchorLabel, ...],
    str,
    dict[str, Any],
]:
    pack = acquisition.materialize_label_pack_once(
        archive_path=project / ARCHIVE_RELATIVE,
        prompt_sidecar_path=project / PROMPT_SIDECAR_RELATIVE,
        acquisition_root=acquisition_root,
        block=block,
        label_capability_path=capability.path,
    )
    label_path = acquisition_root / "labels" / f"{block}.private.json"
    pack_sha = _require_sha256(pack.get("label_pack_sha256"), "label pack")
    if _sha256_file(label_path, f"{block} label pack") != hashlib.sha256(
        canonical_bytes(pack)
    ).hexdigest():
        raise EraserEvidenceInferenceFormalControllerError(
            "materialized label pack file binding drifted"
        )
    label_state = acquisition.load_verified_label_state(
        acquisition_root=acquisition_root,
        block=block,
        label_capability_path=capability.path,
    )
    if (
        label_state.get("label_pack_sha256") != pack_sha
        or label_state.get("label_capability_sha256") != capability.self_sha256
        or label_state.get("label_capability_file_sha256")
        != capability.file_sha256
        or label_state.get(
            "upstream_typed_artifact_content_verified_by_acquisition"
        )
        is not False
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "controller/acquisition persisted label authorization chain drifted"
        )
    _require_sha256(
        label_state.get("label_stage_marker_sha256"),
        f"{block} label stage marker",
    )
    return pack, _labels_from_pack(pack, block=block), pack_sha, label_state


def _promotion_decision(
    *, score: runner.AnchorScoreSeal, policy: runner.PolicySeal
) -> dict[str, Any]:
    if score.block != "A_hold":
        raise EraserEvidenceInferenceFormalControllerError(
            "promotion decision source block drifted"
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_a_hold_promotion_decision",
            "version": VERSION,
            "status": (
                "evaluator_promoted_M_search_authorized"
                if score.evaluator_promoted
                else "evaluator_not_promoted_M_search_remains_unopened"
            ),
            "a_hold_score_receipt_sha256": score.score_receipt_sha256,
            "f_search_policy_receipt_sha256": policy.policy_receipt_sha256,
            "evaluator_promoted": score.evaluator_promoted,
            "decision_rule": (
                "typed_A_hold_score_projection_without_additional_gate"
            ),
            "M_search_materialization_authorized": score.evaluator_promoted,
            "new_threshold_seed_candidate_feature_or_family_added": False,
        },
        "promotion_decision_sha256",
    )


def _terminal_result_payload(
    *,
    lifecycle_marker: PersistedArtifact,
    preflight: PersistedArtifact,
    qualification: PersistedArtifact,
    public: Mapping[str, Any],
    initial_schedule: PersistedScheduleExecution,
    a_form_capability: PersistedArtifact,
    a_form_label_pack_sha256: str,
    fit: PersistedArtifact,
    policy: PersistedArtifact,
    f_policy: PersistedArtifact,
    a_hold_view_sha256: str,
    a_hold_schedule: PersistedScheduleExecution,
    a_hold_capability: PersistedArtifact,
    a_hold_label_pack_sha256: str,
    a_hold_score: runner.AnchorScoreSeal,
    a_hold_score_artifact: PersistedArtifact,
    decision: PersistedArtifact,
    promotion: PersistedArtifact | None,
    m_view_sha256: str | None,
    m_schedule: PersistedScheduleExecution | None,
    m_capability: PersistedArtifact | None,
    m_label_pack_sha256: str | None,
    m_score: runner.AnchorScoreSeal | None,
    m_score_artifact: PersistedArtifact | None,
    postflight: PersistedArtifact,
    private_materialized_file_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    promoted = a_hold_score.evaluator_promoted
    if promoted is not (m_score is not None):
        raise EraserEvidenceInferenceFormalControllerError(
            "terminal promotion/measurement state drifted"
        )
    hold_receipt = a_hold_score.receipt
    m_receipt = None if m_score is None else m_score.receipt
    claims = {
        "A_hold_real_domain_primary_passed": hold_receipt[
            "A_hold_real_domain_primary_passed"
        ],
        "A_hold_RAW_block_passed": hold_receipt["RAW_block_passed"],
        "evaluator_promoted": promoted,
        "M_L5_passed": None if m_receipt is None else m_receipt["M_L5_passed"],
        "cross_relation_stability_passed": (
            None
            if m_receipt is None
            else m_receipt["cross_relation_stability_passed"]
        ),
        "RAW_advantage_overcome": (
            None if m_receipt is None else m_receipt["RAW_advantage_overcome"]
        ),
    }
    claims["total_goal_evidence_passed"] = bool(
        claims["A_hold_real_domain_primary_passed"]
        and claims["A_hold_RAW_block_passed"]
        and claims["evaluator_promoted"]
        and claims["M_L5_passed"]
        and claims["cross_relation_stability_passed"]
        and claims["RAW_advantage_overcome"]
    )
    body: dict[str, Any] = {
        "schema": f"{VERSION}_terminal_result",
        "version": VERSION,
        "status": (
            "complete_promoted_untouched_M_search_measured"
            if promoted
            else "complete_nonpromotion_M_search_unopened"
        ),
        "lifecycle_marker_sha256": lifecycle_marker.self_sha256,
        "runtime_preflight_receipt_sha256": preflight.self_sha256,
        "qualification_sha256": qualification.self_sha256,
        "private_assignment_sha256": public["private_assignment_sha256"],
        "public_acquisition_receipt_sha256": public["public_receipt_sha256"],
        "initial_schedule_receipt_sha256": (
            initial_schedule.schedule_receipt.self_sha256
        ),
        "A_form_label_capability_sha256": a_form_capability.self_sha256,
        "A_form_label_pack_sha256": a_form_label_pack_sha256,
        "E3_fit_receipt_sha256": fit.self_sha256,
        "F_search_policy_receipt_sha256": policy.self_sha256,
        "F_policy_authorization_seal_sha256": f_policy.self_sha256,
        "A_hold_label_free_view_sha256": a_hold_view_sha256,
        "A_hold_schedule_receipt_sha256": (
            a_hold_schedule.schedule_receipt.self_sha256
        ),
        "A_hold_label_capability_sha256": a_hold_capability.self_sha256,
        "A_hold_label_pack_sha256": a_hold_label_pack_sha256,
        "A_hold_score_receipt_sha256": a_hold_score_artifact.self_sha256,
        "A_hold_score_receipt": hold_receipt,
        "promotion_decision_sha256": decision.self_sha256,
        "promotion_authorization_seal_sha256": (
            None if promotion is None else promotion.self_sha256
        ),
        "M_search_label_free_view_sha256": m_view_sha256,
        "M_search_schedule_receipt_sha256": (
            None if m_schedule is None else m_schedule.schedule_receipt.self_sha256
        ),
        "M_search_label_capability_sha256": (
            None if m_capability is None else m_capability.self_sha256
        ),
        "M_search_label_pack_sha256": m_label_pack_sha256,
        "M_search_score_receipt_sha256": (
            None if m_score_artifact is None else m_score_artifact.self_sha256
        ),
        "M_search_score_receipt": m_receipt,
        "postflight_state_sha256": postflight.self_sha256,
        "artifact_file_bindings": {
            "lifecycle_marker": _artifact_binding(lifecycle_marker),
            "runtime_preflight": _artifact_binding(preflight),
            "qualification": _artifact_binding(qualification),
            "initial_schedule": _schedule_binding(initial_schedule),
            "A_form_label_capability": _artifact_binding(a_form_capability),
            "E3_fit": _artifact_binding(fit),
            "F_search_policy": _artifact_binding(policy),
            "F_policy_authorization": _artifact_binding(f_policy),
            "A_hold_schedule": _schedule_binding(a_hold_schedule),
            "A_hold_label_capability": _artifact_binding(a_hold_capability),
            "A_hold_score": _artifact_binding(a_hold_score_artifact),
            "promotion_decision": _artifact_binding(decision),
            "promotion_authorization": (
                None if promotion is None else _artifact_binding(promotion)
            ),
            "M_search_schedule": (
                None if m_schedule is None else _schedule_binding(m_schedule)
            ),
            "M_search_label_capability": (
                None if m_capability is None else _artifact_binding(m_capability)
            ),
            "M_search_score": (
                None
                if m_score_artifact is None
                else _artifact_binding(m_score_artifact)
            ),
            "postflight": _artifact_binding(postflight),
            "private_materialized_files": {
                key: dict(value)
                for key, value in private_materialized_file_bindings.items()
            },
        },
        "claims": claims,
        "all_available_blocks_executed_with_3n_submission_barrier": True,
        "F_search_label_pack_created_or_opened": False,
        "M_search_opened_without_promotion": False,
        "online_evaluator_calls": 0,
        "external_network_calls": 0,
        "private_item_content_or_item_level_utility_persisted": False,
    }
    return _self_hashed(body, "terminal_result_sha256")


def _persist_terminal_failure(
    *,
    controller_root: Path,
    lifecycle_marker_sha256: str,
    stage: str,
    error: BaseException,
) -> None:
    failure_path = controller_root / FAILURE_FILENAME
    result_path = controller_root / RESULT_FILENAME
    if os.path.lexists(failure_path) or os.path.lexists(result_path):
        return
    message = str(error).encode("utf-8", errors="backslashreplace")
    receipt = _self_hashed(
        {
            "schema": f"{VERSION}_terminal_failure",
            "version": VERSION,
            "status": "terminal_source_epoch_failure_no_retry",
            "lifecycle_marker_sha256": lifecycle_marker_sha256,
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(message).hexdigest(),
            "exception_message_persisted": False,
            "retry_replay_resample_replacement_or_secret_rotation_authorized": False,
            "test_access_authorized": False,
            "online_evaluator_calls": 0,
        },
        "terminal_failure_sha256",
    )
    try:
        _write_exclusive(failure_path, receipt)
    except EraserEvidenceInferenceFormalControllerError:
        # The epoch is already terminal.  Do not turn failure reporting into a
        # recovery, overwrite, or second execution path.
        return


def _run_started_lifecycle(
    *,
    project: Path,
    controller_root: Path,
    acquisition_root: Path,
    runtime_config: local_runtime.FormalRuntimeConfig,
    lifecycle_marker: PersistedArtifact,
    preflight_artifact: PersistedArtifact,
    stage_state: dict[str, str],
) -> dict[str, Any]:
    archive_path = project / ARCHIVE_RELATIVE
    sidecar_path = project / PROMPT_SIDECAR_RELATIVE
    design_path = project / DESIGN_RELATIVE
    freeze_path = project / FULL_IMPLEMENTATION_FREEZE_RELATIVE

    stage_state["name"] = "source_qualification"
    qualification_payload = source_qualification.build_formal_qualification(project)
    qualification_sha = _verify_self_hash(
        qualification_payload,
        schema=source_qualification.SCHEMA,
        field="qualification_sha256",
    )
    qualification_artifact = _persist_typed_artifact(
        path=controller_root / QUALIFICATION_FILENAME,
        payload=qualification_payload,
        schema=source_qualification.SCHEMA,
        field="qualification_sha256",
        expected_sha256=qualification_sha,
    )
    if qualification_payload.get("status") != "passed_source_qualification_no_selection":
        raise EraserEvidenceInferenceFormalControllerError(
            "formal source qualification terminated without selection"
        )

    stage_state["name"] = "private_assignment_and_initial_views"
    public = acquisition.acquire_once(
        archive_path=archive_path,
        prompt_sidecar_path=sidecar_path,
        qualification_receipt_path=qualification_artifact.path,
        design_path=design_path,
        implementation_freeze_path=freeze_path,
        project_root=project,
        acquisition_root=acquisition_root,
        selection_secret=None,
        enforce_formal_design_identity=True,
    )
    _require_sha256(public.get("private_assignment_sha256"), "private assignment")
    _require_sha256(public.get("public_receipt_sha256"), "public acquisition")

    a_form_view = acquisition.load_verified_block_view(
        acquisition_root=acquisition_root,
        block="A_form",
        authorization_path=None,
    )
    f_search_view = acquisition.load_verified_block_view(
        acquisition_root=acquisition_root,
        block="F_search",
        authorization_path=None,
    )
    a_form_rows = _view_to_runtime_rows(a_form_view, block="A_form")
    f_search_rows = _view_to_runtime_rows(f_search_view, block="F_search")

    stage_state["name"] = "initial_A_form_F_search_three_arm_execution"
    runtime_bundle = local_runtime.open_runtime(runtime_config)
    initial = scheduler.run_three_arm_schedule(
        items_by_block={
            "A_form": a_form_rows,
            "F_search": f_search_rows,
        },
        runtime_bundle=runtime_bundle,
    )
    initial_persisted = _persist_schedule(
        controller_root=controller_root,
        schedule=initial,
    )
    a_execution = initial_persisted.blocks["A_form"]
    f_execution = initial_persisted.blocks["F_search"]

    stage_state["name"] = "A_form_label_open_and_E3_fit"
    a_form_capability = _persist_label_capability(
        controller_root=controller_root,
        block="A_form",
        public=public,
        view=a_form_view,
        execution=a_execution,
    )
    (
        a_form_pack,
        a_form_labels,
        a_form_pack_sha,
        a_form_label_state,
    ) = _materialize_labels(
        project=project,
        acquisition_root=acquisition_root,
        block="A_form",
        capability=a_form_capability,
    )
    fold_key = acquisition.derive_a_form_fold_key(
        acquisition_root=acquisition_root
    )
    utility_deltas = _a_form_utility_deltas(
        execution=a_execution.artifact,
        labels=a_form_labels,
    )
    try:
        fit_seal = runner.fit_e3(
            feature_seal=a_execution.artifact.feature_seal,
            utility_deltas=utility_deltas,
            fold_secret=fold_key,
        )
    finally:
        del utility_deltas
        del fold_key
        del a_form_labels
        del a_form_pack
    fit_artifact = _persist_typed_artifact(
        path=controller_root / "A_form.e3_fit.receipt.json",
        payload=fit_seal.receipt,
        schema=f"{runner.VERSION}_e3_fit_receipt",
        field="fit_receipt_sha256",
        expected_sha256=fit_seal.fit_receipt_sha256,
    )

    stage_state["name"] = "F_search_policy_freeze"
    policy_seal = runner.freeze_f_policy(
        feature_seal=f_execution.artifact.feature_seal,
        fit_seal=fit_seal,
    )
    policy_artifact = _persist_typed_artifact(
        path=controller_root / "F_search.policy.receipt.json",
        payload=policy_seal.receipt,
        schema=f"{runner.VERSION}_policy_receipt",
        field="policy_receipt_sha256",
        expected_sha256=policy_seal.policy_receipt_sha256,
    )

    f_policy_payload = acquisition.build_f_policy_seal(
        private_assignment_sha256=str(public["private_assignment_sha256"]),
        public_receipt_sha256=str(public["public_receipt_sha256"]),
        a_form_label_free_view_sha256=str(
            a_form_view["label_free_view_sha256"]
        ),
        f_search_label_free_view_sha256=str(
            f_search_view["label_free_view_sha256"]
        ),
        a_form_three_arm_execution_seal_sha256=(
            a_execution.execution_seal_sha256
        ),
        f_search_three_arm_execution_seal_sha256=(
            f_execution.execution_seal_sha256
        ),
        a_form_feature_seal_sha256=(
            a_execution.artifact.feature_seal.feature_receipt_sha256
        ),
        f_search_feature_seal_sha256=(
            f_execution.artifact.feature_seal.feature_receipt_sha256
        ),
        a_form_label_pack_sha256=a_form_pack_sha,
        a_form_label_capability_sha256=str(
            a_form_label_state["label_capability_sha256"]
        ),
        a_form_label_capability_file_sha256=str(
            a_form_label_state["label_capability_file_sha256"]
        ),
        a_form_label_stage_marker_sha256=str(
            a_form_label_state["label_stage_marker_sha256"]
        ),
        e3_fit_receipt_sha256=fit_seal.fit_receipt_sha256,
        f_search_policy_receipt_sha256=policy_seal.policy_receipt_sha256,
    )
    f_policy_artifact = _persist_typed_artifact(
        path=controller_root / "F_search.policy.authorization.json",
        payload=f_policy_payload,
        schema=f"{acquisition.VERSION}_f_policy_seal",
        field="f_policy_seal_sha256",
        expected_sha256=f_policy_payload["f_policy_seal_sha256"],
    )

    stage_state["name"] = "A_hold_view_and_three_arm_execution"
    acquisition.materialize_late_view_once(
        archive_path=archive_path,
        prompt_sidecar_path=sidecar_path,
        acquisition_root=acquisition_root,
        block="A_hold",
        authorization_path=f_policy_artifact.path,
    )
    a_hold_view = acquisition.load_verified_block_view(
        acquisition_root=acquisition_root,
        block="A_hold",
        authorization_path=f_policy_artifact.path,
    )
    a_hold_rows = _view_to_runtime_rows(a_hold_view, block="A_hold")
    a_hold_schedule = scheduler.run_three_arm_schedule(
        items_by_block={"A_hold": a_hold_rows},
        runtime_bundle=runtime_bundle,
    )
    a_hold_persisted = _persist_schedule(
        controller_root=controller_root,
        schedule=a_hold_schedule,
    )
    hold_execution = a_hold_persisted.blocks["A_hold"]
    if (
        hold_execution.artifact.hippo_retrieval_seal is None
        or hold_execution.artifact.raw_retrieval_seal is None
    ):
        raise EraserEvidenceInferenceFormalControllerError(
            "A_hold scheduler did not return both independent baseline seals"
        )
    stage_state["name"] = "A_hold_label_open_and_score"
    a_hold_capability = _persist_label_capability(
        controller_root=controller_root,
        block="A_hold",
        public=public,
        view=a_hold_view,
        execution=hold_execution,
    )
    (
        a_hold_pack,
        a_hold_labels,
        a_hold_pack_sha,
        a_hold_label_state,
    ) = _materialize_labels(
        project=project,
        acquisition_root=acquisition_root,
        block="A_hold",
        capability=a_hold_capability,
    )
    try:
        hold_score = runner.score_anchor(
            block="A_hold",
            labels=a_hold_labels,
            anchor_feature_seal=hold_execution.artifact.feature_seal,
            hippo_retrieval_seal=(
                hold_execution.artifact.hippo_retrieval_seal
            ),
            raw_retrieval_seal=hold_execution.artifact.raw_retrieval_seal,
            policy_seal=policy_seal,
            a_hold_authorization=None,
        )
    finally:
        del a_hold_labels
        del a_hold_pack
    hold_score_artifact = _persist_typed_artifact(
        path=controller_root / "A_hold.score.receipt.json",
        payload=hold_score.receipt,
        schema=f"{runner.VERSION}_A_hold_score_receipt",
        field="score_receipt_sha256",
        expected_sha256=hold_score.score_receipt_sha256,
    )
    decision_payload = _promotion_decision(score=hold_score, policy=policy_seal)
    decision_artifact = _persist_typed_artifact(
        path=controller_root / "A_hold.promotion.decision.json",
        payload=decision_payload,
        schema=f"{VERSION}_a_hold_promotion_decision",
        field="promotion_decision_sha256",
        expected_sha256=decision_payload["promotion_decision_sha256"],
    )

    promotion_artifact: PersistedArtifact | None = None
    m_view_sha: str | None = None
    m_persisted: PersistedScheduleExecution | None = None
    m_capability: PersistedArtifact | None = None
    m_pack_sha: str | None = None
    m_label_state: dict[str, Any] | None = None
    m_score: runner.AnchorScoreSeal | None = None
    m_score_artifact: PersistedArtifact | None = None
    late_authorizations: dict[str, Path] = {
        "A_hold": f_policy_artifact.path
    }
    label_capabilities: dict[str, Path] = {
        "A_form": a_form_capability.path,
        "A_hold": a_hold_capability.path,
    }
    if hold_score.evaluator_promoted:
        stage_state["name"] = "promoted_M_search_view_execution_and_score"
        promotion_payload = acquisition.build_a_hold_promotion_seal(
            private_assignment_sha256=str(public["private_assignment_sha256"]),
            public_receipt_sha256=str(public["public_receipt_sha256"]),
            a_hold_label_free_view_sha256=str(
                a_hold_view["label_free_view_sha256"]
            ),
            f_policy_seal_sha256=f_policy_artifact.self_sha256,
            a_hold_three_arm_execution_seal_sha256=(
                hold_execution.execution_seal_sha256
            ),
            a_hold_feature_seal_sha256=(
                hold_execution.artifact.feature_seal.feature_receipt_sha256
            ),
            a_hold_label_pack_sha256=a_hold_pack_sha,
            a_hold_label_capability_sha256=str(
                a_hold_label_state["label_capability_sha256"]
            ),
            a_hold_label_capability_file_sha256=str(
                a_hold_label_state["label_capability_file_sha256"]
            ),
            a_hold_label_stage_marker_sha256=str(
                a_hold_label_state["label_stage_marker_sha256"]
            ),
            a_hold_score_receipt_sha256=hold_score.score_receipt_sha256,
            promotion_decision_sha256=decision_artifact.self_sha256,
        )
        promotion_artifact = _persist_typed_artifact(
            path=controller_root / "A_hold.promotion.authorization.json",
            payload=promotion_payload,
            schema=f"{acquisition.VERSION}_a_hold_promotion_seal",
            field="a_hold_promotion_seal_sha256",
            expected_sha256=promotion_payload[
                "a_hold_promotion_seal_sha256"
            ],
        )
        acquisition.materialize_late_view_once(
            archive_path=archive_path,
            prompt_sidecar_path=sidecar_path,
            acquisition_root=acquisition_root,
            block="M_search",
            authorization_path=promotion_artifact.path,
        )
        m_view = acquisition.load_verified_block_view(
            acquisition_root=acquisition_root,
            block="M_search",
            authorization_path=promotion_artifact.path,
        )
        m_view_sha = _require_sha256(
            m_view.get("label_free_view_sha256"), "M_search view"
        )
        m_rows = _view_to_runtime_rows(m_view, block="M_search")
        m_schedule = scheduler.run_three_arm_schedule(
            items_by_block={"M_search": m_rows},
            runtime_bundle=runtime_bundle,
        )
        m_persisted = _persist_schedule(
            controller_root=controller_root,
            schedule=m_schedule,
        )
        m_execution = m_persisted.blocks["M_search"]
        if (
            m_execution.artifact.hippo_retrieval_seal is None
            or m_execution.artifact.raw_retrieval_seal is None
        ):
            raise EraserEvidenceInferenceFormalControllerError(
                "M_search scheduler did not return both independent baseline seals"
            )
        m_capability = _persist_label_capability(
            controller_root=controller_root,
            block="M_search",
            public=public,
            view=m_view,
            execution=m_execution,
        )
        m_pack, m_labels, m_pack_sha, m_label_state = _materialize_labels(
            project=project,
            acquisition_root=acquisition_root,
            block="M_search",
            capability=m_capability,
        )
        try:
            m_score = runner.score_anchor(
                block="M_search",
                labels=m_labels,
                anchor_feature_seal=m_execution.artifact.feature_seal,
                hippo_retrieval_seal=(
                    m_execution.artifact.hippo_retrieval_seal
                ),
                raw_retrieval_seal=m_execution.artifact.raw_retrieval_seal,
                policy_seal=policy_seal,
                a_hold_authorization=hold_score,
            )
        finally:
            del m_labels
            del m_pack
        m_score_artifact = _persist_typed_artifact(
            path=controller_root / "M_search.score.receipt.json",
            payload=m_score.receipt,
            schema=f"{runner.VERSION}_M_search_score_receipt",
            field="score_receipt_sha256",
            expected_sha256=m_score.score_receipt_sha256,
        )
        late_authorizations["M_search"] = promotion_artifact.path
        label_capabilities["M_search"] = m_capability.path

    stage_state["name"] = "postflight_and_terminal_result"
    base_postflight = acquisition.verify_acquisition_state(
        acquisition_root=acquisition_root,
        qualification_receipt_path=qualification_artifact.path,
        design_path=design_path,
        implementation_freeze_path=freeze_path,
        project_root=project,
        enforce_formal_design_identity=True,
    )
    if hold_score.evaluator_promoted:
        full_postflight: Mapping[str, Any] = (
            acquisition.verify_full_acquisition_state(
                acquisition_root=acquisition_root,
                late_authorization_paths=late_authorizations,
                label_capability_paths=label_capabilities,
            )
        )
    else:
        # Deliberately do not call the generic full verifier here because it
        # stats M_search paths.  Nonpromotion keeps M wholly untouched.
        acquisition.load_verified_block_view(
            acquisition_root=acquisition_root,
            block="A_hold",
            authorization_path=f_policy_artifact.path,
        )
        acquisition.load_verified_label_state(
            acquisition_root=acquisition_root,
            block="A_form",
            label_capability_path=a_form_capability.path,
        )
        acquisition.load_verified_label_state(
            acquisition_root=acquisition_root,
            block="A_hold",
            label_capability_path=a_hold_capability.path,
        )
        full_postflight = {
            "verified_view_blocks": ["A_form", "F_search", "A_hold"],
            "verified_label_blocks": ["A_form", "A_hold"],
            "M_search_path_stat_open_or_materialization_calls": 0,
            "reason": "typed_A_hold_nonpromotion",
        }
    postflight_payload = _self_hashed(
        {
            "schema": f"{VERSION}_postflight_receipt",
            "version": VERSION,
            "status": "all_materialized_stages_verified",
            "base_acquisition_receipt_sha256": base_postflight[
                "public_receipt_sha256"
            ],
            "full_stage_verification": dict(full_postflight),
            "M_search_opened": hold_score.evaluator_promoted,
            "test_access_authorized_or_performed": False,
        },
        "postflight_receipt_sha256",
    )
    postflight_artifact = _persist_typed_artifact(
        path=controller_root / "lifecycle.postflight.receipt.json",
        payload=postflight_payload,
        schema=f"{VERSION}_postflight_receipt",
        field="postflight_receipt_sha256",
        expected_sha256=postflight_payload["postflight_receipt_sha256"],
    )

    private_bindings: dict[str, Mapping[str, str]] = {
        # The controller never parses the source-epoch marker because it owns
        # the unique selection secret.  Direct acquisition has already
        # verified its self hash; the public projection plus streamed file hash
        # is sufficient for the terminal ledger without exposing that secret.
        "source_epoch_marker": {
            "source_epoch_marker_sha256": str(
                public["source_epoch_marker_sha256"]
            ),
            "source_epoch_marker_file_sha256": _sha256_file(
                acquisition_root / "acquisition.marker.private.json",
                "source epoch marker",
            ),
        },
        "private_assignment": {
            "private_assignment_sha256": str(
                public["private_assignment_sha256"]
            ),
            "private_assignment_file_sha256": str(
                public["private_assignment_file_sha256"]
            ),
        },
        "public_acquisition_receipt": {
            "public_receipt_sha256": str(public["public_receipt_sha256"]),
            "public_receipt_file_sha256": _sha256_file(
                acquisition_root / "acquisition.receipt.json",
                "public acquisition receipt",
            ),
        },
        "A_form_label_free_view": _private_payload_file_binding(
            path=acquisition_root / "views" / "A_form.private.json",
            payload=a_form_view,
            field="label_free_view_sha256",
        ),
        "F_search_label_free_view": _private_payload_file_binding(
            path=acquisition_root / "views" / "F_search.private.json",
            payload=f_search_view,
            field="label_free_view_sha256",
        ),
        "A_form_label_pack": {
            "label_pack_sha256": a_form_pack_sha,
            "label_pack_file_sha256": _sha256_file(
                acquisition_root / "labels" / "A_form.private.json",
                "A_form label pack",
            ),
        },
        "A_form_acquisition_owned_label_chain": {
            "label_capability_sha256": str(
                a_form_label_state["label_capability_sha256"]
            ),
            "label_capability_file_sha256": str(
                a_form_label_state["label_capability_file_sha256"]
            ),
            "label_stage_marker_sha256": str(
                a_form_label_state["label_stage_marker_sha256"]
            ),
            "label_stage_marker_file_sha256": _sha256_file(
                acquisition_root
                / "stage_markers"
                / "label.A_form.private.json",
                "A_form label stage marker",
            ),
        },
        "A_hold_label_free_view": _private_payload_file_binding(
            path=acquisition_root / "views" / "A_hold.private.json",
            payload=a_hold_view,
            field="label_free_view_sha256",
        ),
        "A_hold_view_stage_marker": _self_hashed_file_binding(
            path=(
                acquisition_root
                / "stage_markers"
                / "view.A_hold.private.json"
            ),
            schema=f"{acquisition.VERSION}_stage_marker",
            field="stage_marker_sha256",
        ),
        "A_hold_label_pack": {
            "label_pack_sha256": a_hold_pack_sha,
            "label_pack_file_sha256": _sha256_file(
                acquisition_root / "labels" / "A_hold.private.json",
                "A_hold label pack",
            ),
        },
        "A_hold_acquisition_owned_label_chain": {
            "label_capability_sha256": str(
                a_hold_label_state["label_capability_sha256"]
            ),
            "label_capability_file_sha256": str(
                a_hold_label_state["label_capability_file_sha256"]
            ),
            "label_stage_marker_sha256": str(
                a_hold_label_state["label_stage_marker_sha256"]
            ),
            "label_stage_marker_file_sha256": _sha256_file(
                acquisition_root
                / "stage_markers"
                / "label.A_hold.private.json",
                "A_hold label stage marker",
            ),
        },
    }
    if hold_score.evaluator_promoted:
        if m_view_sha is None or m_pack_sha is None or m_label_state is None:
            raise EraserEvidenceInferenceFormalControllerError(
                "promoted private M_search bindings are absent"
            )
        private_bindings["M_search_label_free_view"] = {
            "label_free_view_sha256": m_view_sha,
            "label_free_view_file_sha256": _sha256_file(
                acquisition_root / "views" / "M_search.private.json",
                "M_search label-free view",
            ),
        }
        private_bindings["M_search_view_stage_marker"] = (
            _self_hashed_file_binding(
                path=(
                    acquisition_root
                    / "stage_markers"
                    / "view.M_search.private.json"
                ),
                schema=f"{acquisition.VERSION}_stage_marker",
                field="stage_marker_sha256",
            )
        )
        private_bindings["M_search_label_pack"] = {
            "label_pack_sha256": m_pack_sha,
            "label_pack_file_sha256": _sha256_file(
                acquisition_root / "labels" / "M_search.private.json",
                "M_search label pack",
            ),
        }
        private_bindings["M_search_acquisition_owned_label_chain"] = {
            "label_capability_sha256": str(
                m_label_state["label_capability_sha256"]
            ),
            "label_capability_file_sha256": str(
                m_label_state["label_capability_file_sha256"]
            ),
            "label_stage_marker_sha256": str(
                m_label_state["label_stage_marker_sha256"]
            ),
            "label_stage_marker_file_sha256": _sha256_file(
                acquisition_root
                / "stage_markers"
                / "label.M_search.private.json",
                "M_search label stage marker",
            ),
        }

    result = _terminal_result_payload(
        lifecycle_marker=lifecycle_marker,
        preflight=preflight_artifact,
        qualification=qualification_artifact,
        public=public,
        initial_schedule=initial_persisted,
        a_form_capability=a_form_capability,
        a_form_label_pack_sha256=a_form_pack_sha,
        fit=fit_artifact,
        policy=policy_artifact,
        f_policy=f_policy_artifact,
        a_hold_view_sha256=str(a_hold_view["label_free_view_sha256"]),
        a_hold_schedule=a_hold_persisted,
        a_hold_capability=a_hold_capability,
        a_hold_label_pack_sha256=a_hold_pack_sha,
        a_hold_score=hold_score,
        a_hold_score_artifact=hold_score_artifact,
        decision=decision_artifact,
        promotion=promotion_artifact,
        m_view_sha256=m_view_sha,
        m_schedule=m_persisted,
        m_capability=m_capability,
        m_label_pack_sha256=m_pack_sha,
        m_score=m_score,
        m_score_artifact=m_score_artifact,
        postflight=postflight_artifact,
        private_materialized_file_bindings=private_bindings,
    )
    _persist_typed_artifact(
        path=controller_root / RESULT_FILENAME,
        payload=result,
        schema=f"{VERSION}_terminal_result",
        field="terminal_result_sha256",
        expected_sha256=result["terminal_result_sha256"],
    )
    return result


def run_formal_study(*, project_root: str | Path) -> dict[str, Any]:
    """Execute the frozen study once; any post-marker failure is terminal."""

    project = _canonical_project(project_root)
    freeze_path = project / FULL_IMPLEMENTATION_FREEZE_RELATIVE
    design_path = project / DESIGN_RELATIVE
    freeze = verify_full_implementation_freeze(
        project=project,
        freeze_path=freeze_path,
    )
    runtime_config = local_runtime.default_formal_runtime_config(project)
    raw_preflight = local_runtime.preflight_formal_runtime_config(runtime_config)
    preflight_payload = _runtime_preflight_receipt(raw_preflight)
    lifecycle_payload = _lifecycle_marker(
        project=project,
        freeze=freeze,
        freeze_path=freeze_path,
        design_path=design_path,
        preflight_receipt=preflight_payload,
    )
    _formal_root, controller_root, acquisition_root = _create_fresh_formal_roots(
        project=project
    )
    lifecycle_artifact = _persist_typed_artifact(
        path=controller_root / MARKER_FILENAME,
        payload=lifecycle_payload,
        schema=f"{VERSION}_one_shot_marker",
        field="lifecycle_marker_sha256",
        expected_sha256=lifecycle_payload["lifecycle_marker_sha256"],
    )
    stage_state = {"name": "persist_runtime_preflight"}
    try:
        preflight_artifact = _persist_typed_artifact(
            path=controller_root / "runtime.preflight.receipt.json",
            payload=preflight_payload,
            schema=f"{VERSION}_runtime_preflight_receipt",
            field="runtime_preflight_receipt_sha256",
            expected_sha256=preflight_payload[
                "runtime_preflight_receipt_sha256"
            ],
        )
        return _run_started_lifecycle(
            project=project,
            controller_root=controller_root,
            acquisition_root=acquisition_root,
            runtime_config=runtime_config,
            lifecycle_marker=lifecycle_artifact,
            preflight_artifact=preflight_artifact,
            stage_state=stage_state,
        )
    except BaseException as exc:
        _persist_terminal_failure(
            controller_root=controller_root,
            lifecycle_marker_sha256=lifecycle_artifact.self_sha256,
            stage=stage_state["name"],
            error=exc,
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise EraserEvidenceInferenceFormalControllerError(
            "formal lifecycle failed terminally"
        ) from exc


def _public_stdout_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "terminal_result_sha256": result.get("terminal_result_sha256"),
        "claims": result.get("claims"),
        "online_evaluator_calls": result.get("online_evaluator_calls"),
        "external_network_calls": result.get("external_network_calls"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    arguments = parser.parse_args(argv)
    try:
        result = run_formal_study(project_root=arguments.project_root)
    except EraserEvidenceInferenceFormalControllerError:
        print(
            json.dumps(
                {"status": "terminal_error_see_private_failure_receipt"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(_public_stdout_summary(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

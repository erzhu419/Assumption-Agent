"""One-shot A_hold and M3 runners for evaluator co-evolution.

Freeze builders have no measurement-block path parameter and therefore cannot
open A_hold or M3.  They bind the public acquisition commitments, the exact
A_form/F3 evidence and candidate set, the live implementation, a fresh formal
root, and a one-shot authorization.  Clean module-CLI execution consumes that
root before opening the exact measurement block.

Each formal run executes the complete fixed typed-program/item grid with
gold-free retrieval inputs at maximum local concurrency.  Official support
labels are consulted only after every retrieval terminal has joined.  A_hold
recomputes the strict 0.9 Wilson transition.  M3 first re-verifies that result
from the exact A_hold private evidence, then measures prospective search
utility.  No model, generator, network, online judge, retry, replay, resample,
or callable/result injection surface exists.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from .musique_evaluator_coevolution_v1 import (
    ANCHOR_CONFIDENCE,
    compare_on_fixed_anchor,
    measure_prospective_search_utility,
)
from . import musique_evaluator_stage_formation_v1 as formation
from .musique_recursive_study_blocks_v1 import (
    load_measurement_block_after_freeze,
    load_study_acquisition_binding,
)


VERSION = "musique_evaluator_measurement_runner_v1"
FREEZE_SCHEMA = f"{VERSION}_pre_run_freeze"
REPORT_SCHEMA = f"{VERSION}_aggregate_report"
FAILURE_SCHEMA = f"{VERSION}_failure"
CONSUMPTION_SCHEMA = f"{VERSION}_authorization_consumption"
IMPLEMENTATION_SCHEMA = f"{VERSION}_implementation"
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/musique_evaluator_measurement_runner_v1.py",
    "assumption_agent/benchmarks/musique_evaluator_stage_formation_v1.py",
    "assumption_agent/benchmarks/musique_evaluator_coevolution_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
)
MEASUREMENT_STAGES = ("A_hold", "M3")
ITEM_COUNT = 12
CONSUMPTION_FILENAME = "authorization.consumed.json"
PRIVATE_EVIDENCE_FILENAME = {
    "A_hold": "a_hold.private.evidence.json",
    "M3": "m3.private.evidence.json",
}
REPORT_FILENAME = {
    "A_hold": "a_hold.aggregate.report.json",
    "M3": "m3.aggregate.report.json",
}
FAILURE_FILENAME = {
    "A_hold": "a_hold.failure.json",
    "M3": "m3.failure.json",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class MuSiQueEvaluatorMeasurementError(RuntimeError):
    """A frozen evaluator measurement failed closed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueEvaluatorMeasurementError(
            "required evaluator measurement file is unavailable"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueEvaluatorMeasurementError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _absolute_no_symlink(path: str | Path, field_name: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueEvaluatorMeasurementError(
                f"{field_name} contains a symlink component"
            )
    return candidate


def _canonical_new_root(path: str | Path) -> Path:
    candidate = _absolute_no_symlink(path, "evaluator formal execution root")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise MuSiQueEvaluatorMeasurementError(
            "evaluator execution-root parent is unavailable"
        ) from exc
    if not parent.is_dir():
        raise MuSiQueEvaluatorMeasurementError(
            "evaluator execution-root parent is not a directory"
        )
    return parent / candidate.name


def _root_binding_hash(path: str | Path) -> str:
    return stable_hash({"absolute_execution_root": str(_canonical_new_root(path))})


def _read_json_object(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = _absolute_no_symlink(path, field_name)
    if not candidate.is_file():
        raise MuSiQueEvaluatorMeasurementError(f"{field_name} is unavailable")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueEvaluatorMeasurementError(f"{field_name} is invalid") from exc
    if not isinstance(value, dict):
        raise MuSiQueEvaluatorMeasurementError(
            f"{field_name} must contain one object"
        )
    return value, raw


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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


def _require_private_execution_boundary(path: Path) -> None:
    """Reject an in-repository execution root unless Git ignores it.

    A_hold and M3 persist item-level support evidence below the formal root.
    Keeping the path out of public receipts is not sufficient if that root can
    subsequently be staged by accident.
    """

    try:
        formation._require_private_cache_boundary(
            path / PRIVATE_EVIDENCE_FILENAME["A_hold"]
        )
        formation._require_private_cache_boundary(
            path / PRIVATE_EVIDENCE_FILENAME["M3"]
        )
    except formation.MuSiQueEvaluatorStageFormationError as exc:
        raise MuSiQueEvaluatorMeasurementError(
            "evaluator execution root must be external or git-ignored"
        ) from exc


def current_evaluator_measurement_implementation_binding(
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueEvaluatorMeasurementError(
                f"evaluator measurement implementation is missing: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def _validate_implementation_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "files",
        "set_sha256",
    }:
        raise MuSiQueEvaluatorMeasurementError(
            "evaluator measurement implementation binding is malformed"
        )
    if value.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueEvaluatorMeasurementError(
            "evaluator measurement implementation schema drifted"
        )
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(IMPLEMENTATION_RELATIVE_FILES):
        raise MuSiQueEvaluatorMeasurementError(
            "evaluator measurement implementation set drifted"
        )
    rows: list[dict[str, str]] = []
    for expected, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row.get("path") != expected
        ):
            raise MuSiQueEvaluatorMeasurementError(
                "evaluator measurement implementation row drifted"
            )
        rows.append(
            {
                "path": expected,
                "sha256": _require_sha256(
                    row.get("sha256"), "implementation file hash"
                ),
            }
        )
    set_hash = _require_sha256(value.get("set_sha256"), "implementation set hash")
    if stable_hash(rows) != set_hash:
        raise MuSiQueEvaluatorMeasurementError(
            "evaluator measurement implementation set hash drifted"
        )
    return {"schema": IMPLEMENTATION_SCHEMA, "files": rows, "set_sha256": set_hash}


def _file_binding(path: str | Path, semantic_sha256: str) -> dict[str, str]:
    candidate = _absolute_no_symlink(path, "bound evaluator artifact")
    return {
        "file_sha256": _sha256_file(candidate),
        "semantic_sha256": _require_sha256(semantic_sha256, "artifact semantic hash"),
    }


def _a_form_binding(
    *,
    private_path: str | Path,
    public_path: str | Path,
    cache: Mapping[str, Any],
    public: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "private_evidence": _file_binding(private_path, cache["cache_sha256"]),
        "public_receipt": _file_binding(public_path, public["receipt_sha256"]),
        "evidence_set_sha256": cache["evidence_set_sha256"],
        "candidate_set_binding": cache["candidate_set_binding"],
        "source_binding": cache["source_binding"],
        "core_formation_sha256": public["core_receipt"]["formation_sha256"],
    }


def _f3_binding(
    *,
    private_path: str | Path,
    public_path: str | Path,
    cache: Mapping[str, Any],
    public: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "private_evidence": _file_binding(private_path, cache["cache_sha256"]),
        "public_receipt": _file_binding(public_path, public["receipt_sha256"]),
        "evidence_set_sha256": cache["evidence_set_sha256"],
        "candidate_set_binding": cache["candidate_set_binding"],
        "source_binding": cache["source_binding"],
        "core_search_formation_sha256": public["core_receipt"][
            "search_formation_sha256"
        ],
    }


def _load_a_form(
    *,
    private_path: str | Path,
    public_path: str | Path,
    project_root: str | Path,
):
    evidence, cache, public = formation.load_a_form_bundle(
        private_evidence_path=private_path,
        public_receipt_path=public_path,
        verify_live=True,
        project_root=project_root,
    )
    return evidence, cache, public, _a_form_binding(
        private_path=private_path,
        public_path=public_path,
        cache=cache,
        public=public,
    )


def _load_f3(
    *,
    private_path: str | Path,
    public_path: str | Path,
    a_form_private_path: str | Path,
    a_form_public_path: str | Path,
    project_root: str | Path,
):
    evidence, cache, public = formation.load_f3_bundle(
        private_evidence_path=private_path,
        public_receipt_path=public_path,
        a_form_private_evidence_path=a_form_private_path,
        a_form_public_receipt_path=a_form_public_path,
        verify_live=True,
        project_root=project_root,
    )
    return evidence, cache, public, _f3_binding(
        private_path=private_path,
        public_path=public_path,
        cache=cache,
        public=public,
    )


def _execution_contract(stage: str, item_count: int) -> dict[str, Any]:
    candidate_count = formation.candidate_set_binding()["candidate_count"]
    work_units = candidate_count * item_count
    return {
        "measurement_stage": stage,
        "candidate_count": candidate_count,
        "item_count": item_count,
        "work_unit_count": work_units,
        "maximum_local_concurrency": work_units,
        "top_k": formation.candidate_set_binding()["top_k"],
        "all_work_units_submitted_before_support_scoring": True,
        "all_terminals_joined_before_support_scoring": True,
        "formal_entry": "clean_python_module_cli_v1",
        "generator_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def _build_freeze(
    *,
    stage: str,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
    f3_private_evidence_path: str | Path | None = None,
    f3_public_receipt_path: str | Path | None = None,
    a_hold_private_evidence_path: str | Path | None = None,
    a_hold_report_path: str | Path | None = None,
) -> dict[str, Any]:
    if stage not in MEASUREMENT_STAGES:
        raise MuSiQueEvaluatorMeasurementError("unknown evaluator measurement stage")
    project = Path(project_root).resolve(strict=True)
    _require_private_execution_boundary(_canonical_new_root(execution_root))
    acquisition = load_study_acquisition_binding(acquisition_receipt_path)
    commitment = acquisition.commitment_for(stage)
    a_evidence, a_cache, a_public, a_binding = _load_a_form(
        private_path=a_form_private_evidence_path,
        public_path=a_form_public_receipt_path,
        project_root=project,
    )
    if (
        a_cache["source_binding"].get("acquisition_sha256")
        != acquisition.acquisition_sha256
        or a_cache["source_binding"].get("private_pack_sha256")
        != acquisition.private_pack_sha256
    ):
        raise MuSiQueEvaluatorMeasurementError(
            "A_form differs from the measurement acquisition"
        )

    f3_binding: Mapping[str, Any] | None = None
    anchor_binding: Mapping[str, Any] | None = None
    if stage == "M3":
        if any(
            value is None
            for value in (
                f3_private_evidence_path,
                f3_public_receipt_path,
                a_hold_private_evidence_path,
                a_hold_report_path,
            )
        ):
            raise MuSiQueEvaluatorMeasurementError(
                "M3 freeze requires exact F3 and A_hold artifacts"
            )
        _f3_evidence, f3_cache, _f3_public, f3_binding = _load_f3(
            private_path=f3_private_evidence_path,
            public_path=f3_public_receipt_path,
            a_form_private_path=a_form_private_evidence_path,
            a_form_public_path=a_form_public_receipt_path,
            project_root=project,
        )
        if f3_cache["source_binding"].get("acquisition_sha256") != (
            acquisition.acquisition_sha256
        ):
            raise MuSiQueEvaluatorMeasurementError(
                "F3 differs from the M3 acquisition"
            )
        _anchor_evidence, _anchor_cache, _anchor_report, anchor_binding = (
            load_and_reverify_a_hold_artifacts(
                private_evidence_path=a_hold_private_evidence_path,
                report_path=a_hold_report_path,
                a_form_private_evidence_path=a_form_private_evidence_path,
                a_form_public_receipt_path=a_form_public_receipt_path,
                project_root=project,
            )
        )
        anchor_commitment = acquisition.commitment_for("A_hold")
        expected_anchor_source = {
            "acquisition_sha256": acquisition.acquisition_sha256,
            "acquisition_file_sha256": acquisition.acquisition_file_sha256,
            "private_pack_sha256": acquisition.private_pack_sha256,
            "measurement_block_id_hash": stable_hash({"block": "A_hold"}),
            "measurement_block_file_sha256": anchor_commitment.file_sha256,
            "measurement_item_commitment_set_sha256": (
                anchor_commitment.item_commitment_set_sha256
            ),
            "measurement_item_count": anchor_commitment.count,
        }
        if (
            anchor_binding["source_binding"].get("acquisition_sha256")
            != acquisition.acquisition_sha256
            or anchor_binding["source_binding"].get("private_pack_sha256")
            != acquisition.private_pack_sha256
            or _anchor_report.get("source_binding") != expected_anchor_source
        ):
            raise MuSiQueEvaluatorMeasurementError(
                "A_hold differs from the M3 acquisition"
            )

    body: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "decision": f"authorize_exact_{stage}_offline_evaluator_measurement_once",
        "measurement_stage": stage,
        "implementation": current_evaluator_measurement_implementation_binding(project),
        "authorization_hash": _require_sha256(
            authorization_hash, f"{stage} execution authorization"
        ),
        "execution_root_hash": _root_binding_hash(execution_root),
        "source_binding": {
            "acquisition_sha256": acquisition.acquisition_sha256,
            "acquisition_file_sha256": acquisition.acquisition_file_sha256,
            "private_pack_sha256": acquisition.private_pack_sha256,
            "measurement_block_id_hash": stable_hash({"block": stage}),
            "measurement_block_file_sha256": commitment.file_sha256,
            "measurement_item_commitment_set_sha256": (
                commitment.item_commitment_set_sha256
            ),
            "measurement_item_count": commitment.count,
        },
        "a_form_binding": a_binding,
        "f3_binding": f3_binding,
        "a_hold_binding": anchor_binding,
        "candidate_set_binding": formation.candidate_set_binding(),
        "execution_contract": _execution_contract(stage, commitment.count),
        "ordering": {
            "measurement_block_rows_read_while_freezing": 0,
            "measurement_support_labels_read_while_freezing": 0,
            "pre_run_freeze_complete_before_measurement_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_hash": stable_hash(body)}
    formation._assert_public_safe(freeze)
    destination = _absolute_no_symlink(output_path, f"{stage} freeze output")
    if destination.exists():
        raise MuSiQueEvaluatorMeasurementError(
            f"{stage} freeze output already exists"
        )
    _write_json_exclusive(destination, freeze)
    return freeze


def build_a_hold_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze A_hold without accepting or reading an A_hold path."""

    return _build_freeze(
        stage="A_hold",
        project_root=project_root,
        acquisition_receipt_path=acquisition_receipt_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        execution_root=execution_root,
        authorization_hash=authorization_hash,
        output_path=output_path,
    )


def build_m3_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f3_private_evidence_path: str | Path,
    f3_public_receipt_path: str | Path,
    a_hold_private_evidence_path: str | Path,
    a_hold_report_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze M3 after re-verifying exact F3 and A_hold evidence."""

    return _build_freeze(
        stage="M3",
        project_root=project_root,
        acquisition_receipt_path=acquisition_receipt_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f3_private_evidence_path=f3_private_evidence_path,
        f3_public_receipt_path=f3_public_receipt_path,
        a_hold_private_evidence_path=a_hold_private_evidence_path,
        a_hold_report_path=a_hold_report_path,
        execution_root=execution_root,
        authorization_hash=authorization_hash,
        output_path=output_path,
    )


def _load_freeze(path: str | Path, expected_stage: str) -> tuple[dict[str, Any], str]:
    payload, raw = _read_json_object(path, f"{expected_stage} pre-run freeze")
    body = dict(payload)
    declared = _require_sha256(body.pop("freeze_hash", None), "freeze hash")
    expected_keys = {
        "a_form_binding",
        "a_hold_binding",
        "authorization_hash",
        "candidate_set_binding",
        "decision",
        "execution_contract",
        "execution_root_hash",
        "f3_binding",
        "freeze_hash",
        "implementation",
        "measurement_stage",
        "ordering",
        "raw_content_persisted",
        "schema",
        "source_binding",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != FREEZE_SCHEMA
        or payload.get("measurement_stage") != expected_stage
        or payload.get("decision")
        != f"authorize_exact_{expected_stage}_offline_evaluator_measurement_once"
        or stable_hash(body) != declared
        or payload.get("raw_content_persisted") is not False
        or payload.get("candidate_set_binding") != formation.candidate_set_binding()
    ):
        raise MuSiQueEvaluatorMeasurementError(
            f"{expected_stage} pre-run freeze drifted"
        )
    contract = payload.get("execution_contract")
    ordering = payload.get("ordering")
    if (
        not isinstance(contract, Mapping)
        or contract != _execution_contract(expected_stage, ITEM_COUNT)
        or not isinstance(ordering, Mapping)
        or ordering.get("measurement_block_rows_read_while_freezing") != 0
        or ordering.get("measurement_support_labels_read_while_freezing") != 0
        or ordering.get("pre_run_freeze_complete_before_measurement_open") is not True
        or (expected_stage == "A_hold" and (
            payload.get("f3_binding") is not None
            or payload.get("a_hold_binding") is not None
        ))
        or (expected_stage == "M3" and (
            not isinstance(payload.get("f3_binding"), Mapping)
            or not isinstance(payload.get("a_hold_binding"), Mapping)
        ))
    ):
        raise MuSiQueEvaluatorMeasurementError(
            f"{expected_stage} frozen execution contract drifted"
        )
    _validate_implementation_binding(payload.get("implementation"))
    _require_sha256(payload.get("authorization_hash"), "authorization hash")
    _require_sha256(payload.get("execution_root_hash"), "execution root hash")
    return payload, _sha256_bytes(raw)


def _anchor_report_body(
    *,
    freeze: Mapping[str, Any],
    freeze_file_hash: str,
    cache: Mapping[str, Any],
    cache_file_hash: str,
    core_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "valid": True,
        "measurement_stage": "A_hold",
        "freeze_hash": freeze["freeze_hash"],
        "freeze_file_sha256": freeze_file_hash,
        "source_binding": freeze["source_binding"],
        "a_form_binding": freeze["a_form_binding"],
        "candidate_set_binding": freeze["candidate_set_binding"],
        "private_evidence_binding": {
            "file_sha256": cache_file_hash,
            "cache_sha256": cache["cache_sha256"],
            "evidence_set_sha256": cache["evidence_set_sha256"],
            "private_path_persisted_publicly": False,
            "item_level_evidence_persisted_publicly": False,
        },
        "core_result": dict(core_result),
        "transition_verification": {
            "confidence": ANCHOR_CONFIDENCE,
            "policy": "strict_wilson_lower_bound_improvement_v1",
            "recomputed_from_exact_A_hold_evidence": True,
            "official_support_objective_replaced": False,
        },
        "execution": cache["execution"],
        "raw_content_persisted": False,
    }


def load_and_reverify_a_hold_artifacts(
    *,
    private_evidence_path: str | Path,
    report_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    project_root: str | Path,
):
    """Recompute the anchor transition from exact cached A_hold evidence."""

    a_evidence, _a_cache, a_public, a_binding = _load_a_form(
        private_path=a_form_private_evidence_path,
        public_path=a_form_public_receipt_path,
        project_root=project_root,
    )
    anchor_evidence, anchor_cache, cache_file_hash = formation.load_private_evidence_cache(
        private_evidence_path, expected_stage="A_hold"
    )
    report, report_raw = _read_json_object(report_path, "A_hold aggregate report")
    report_body = dict(report)
    declared = _require_sha256(report_body.pop("report_hash", None), "A_hold report hash")
    expected_core = compare_on_fixed_anchor(
        formation_evidence=a_evidence,
        anchor_evidence=anchor_evidence,
        formation_receipt=a_public["core_receipt"],
    )
    expected_report_keys = {
        "a_form_binding",
        "candidate_set_binding",
        "core_result",
        "execution",
        "freeze_file_sha256",
        "freeze_hash",
        "measurement_stage",
        "private_evidence_binding",
        "raw_content_persisted",
        "report_hash",
        "schema",
        "source_binding",
        "transition_verification",
        "valid",
    }
    source = report.get("source_binding")
    expected_transition = {
        "confidence": ANCHOR_CONFIDENCE,
        "policy": "strict_wilson_lower_bound_improvement_v1",
        "recomputed_from_exact_A_hold_evidence": True,
        "official_support_objective_replaced": False,
    }
    expected_source_keys = {
        "acquisition_file_sha256",
        "acquisition_sha256",
        "measurement_block_file_sha256",
        "measurement_block_id_hash",
        "measurement_item_commitment_set_sha256",
        "measurement_item_count",
        "private_pack_sha256",
    }
    _require_sha256(report.get("freeze_hash"), "A_hold report freeze hash")
    _require_sha256(
        report.get("freeze_file_sha256"), "A_hold report freeze file hash"
    )
    if (
        set(report) != expected_report_keys
        or report.get("schema") != REPORT_SCHEMA
        or report.get("measurement_stage") != "A_hold"
        or report.get("valid") is not True
        or stable_hash(report_body) != declared
        or report.get("raw_content_persisted") is not False
        or report.get("a_form_binding") != a_binding
        or report.get("candidate_set_binding") != formation.candidate_set_binding()
        or report.get("core_result") != expected_core
        or report.get("execution") != anchor_cache.get("execution")
        or report.get("transition_verification") != expected_transition
        or not isinstance(source, Mapping)
        or set(source) != expected_source_keys
        or source.get("acquisition_sha256")
        != anchor_cache["source_binding"].get("acquisition_sha256")
        or source.get("private_pack_sha256")
        != anchor_cache["source_binding"].get("private_pack_sha256")
        or source.get("measurement_block_id_hash")
        != anchor_cache["source_binding"].get("block_id_hash")
        or source.get("measurement_block_file_sha256")
        != anchor_cache["source_binding"].get("file_sha256")
        or source.get("measurement_item_commitment_set_sha256")
        != anchor_cache["source_binding"].get("item_commitment_set_sha256")
        or source.get("measurement_item_count")
        != anchor_cache["source_binding"].get("item_count")
    ):
        raise MuSiQueEvaluatorMeasurementError(
            "A_hold report differs from exact anchor evidence"
        )
    evidence_binding = report.get("private_evidence_binding")
    if (
        not isinstance(evidence_binding, Mapping)
        or set(evidence_binding) != {
            "cache_sha256",
            "evidence_set_sha256",
            "file_sha256",
            "item_level_evidence_persisted_publicly",
            "private_path_persisted_publicly",
        }
        or evidence_binding.get("file_sha256") != cache_file_hash
        or evidence_binding.get("cache_sha256") != anchor_cache["cache_sha256"]
        or evidence_binding.get("evidence_set_sha256")
        != anchor_cache["evidence_set_sha256"]
        or evidence_binding.get("private_path_persisted_publicly") is not False
        or evidence_binding.get("item_level_evidence_persisted_publicly") is not False
        or not isinstance(report.get("transition_verification"), Mapping)
    ):
        raise MuSiQueEvaluatorMeasurementError(
            "A_hold evidence binding or Wilson transition drifted"
        )
    formation._assert_public_safe(report)
    binding = {
        "private_evidence": _file_binding(
            private_evidence_path, anchor_cache["cache_sha256"]
        ),
        "public_report": {
            "file_sha256": _sha256_bytes(report_raw),
            "semantic_sha256": declared,
        },
        "evidence_set_sha256": anchor_cache["evidence_set_sha256"],
        "anchor_result_sha256": expected_core["anchor_result_sha256"],
        "source_binding": anchor_cache["source_binding"],
        "strict_wilson_transition_reverified": True,
    }
    return anchor_evidence, anchor_cache, report, binding


def _verify_freeze_inputs(
    *,
    freeze: Mapping[str, Any],
    stage: str,
    project: Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f3_private_evidence_path: str | Path | None,
    f3_public_receipt_path: str | Path | None,
    a_hold_private_evidence_path: str | Path | None,
    a_hold_report_path: str | Path | None,
):
    acquisition = load_study_acquisition_binding(acquisition_receipt_path)
    commitment = acquisition.commitment_for(stage)
    source = freeze.get("source_binding")
    expected_source = {
        "acquisition_sha256": acquisition.acquisition_sha256,
        "acquisition_file_sha256": acquisition.acquisition_file_sha256,
        "private_pack_sha256": acquisition.private_pack_sha256,
        "measurement_block_id_hash": stable_hash({"block": stage}),
        "measurement_block_file_sha256": commitment.file_sha256,
        "measurement_item_commitment_set_sha256": (
            commitment.item_commitment_set_sha256
        ),
        "measurement_item_count": commitment.count,
    }
    if source != expected_source:
        raise MuSiQueEvaluatorMeasurementError(
            f"{stage} source binding drifted"
        )
    a_evidence, a_cache, a_public, a_binding = _load_a_form(
        private_path=a_form_private_evidence_path,
        public_path=a_form_public_receipt_path,
        project_root=project,
    )
    if freeze.get("a_form_binding") != a_binding:
        raise MuSiQueEvaluatorMeasurementError(
            f"{stage} A_form binding drifted"
        )
    f3_bundle = None
    anchor_bundle = None
    if stage == "M3":
        if any(value is None for value in (
            f3_private_evidence_path,
            f3_public_receipt_path,
            a_hold_private_evidence_path,
            a_hold_report_path,
        )):
            raise MuSiQueEvaluatorMeasurementError(
                "formal M3 requires exact frozen F3 and A_hold artifacts"
            )
        f3_bundle = _load_f3(
            private_path=f3_private_evidence_path,
            public_path=f3_public_receipt_path,
            a_form_private_path=a_form_private_evidence_path,
            a_form_public_path=a_form_public_receipt_path,
            project_root=project,
        )
        if freeze.get("f3_binding") != f3_bundle[3]:
            raise MuSiQueEvaluatorMeasurementError("M3 F3 binding drifted")
        anchor_bundle = load_and_reverify_a_hold_artifacts(
            private_evidence_path=a_hold_private_evidence_path,
            report_path=a_hold_report_path,
            a_form_private_evidence_path=a_form_private_evidence_path,
            a_form_public_receipt_path=a_form_public_receipt_path,
            project_root=project,
        )
        if freeze.get("a_hold_binding") != anchor_bundle[3]:
            raise MuSiQueEvaluatorMeasurementError("M3 A_hold binding drifted")
        anchor_commitment = acquisition.commitment_for("A_hold")
        if anchor_bundle[2].get("source_binding") != {
            "acquisition_sha256": acquisition.acquisition_sha256,
            "acquisition_file_sha256": acquisition.acquisition_file_sha256,
            "private_pack_sha256": acquisition.private_pack_sha256,
            "measurement_block_id_hash": stable_hash({"block": "A_hold"}),
            "measurement_block_file_sha256": anchor_commitment.file_sha256,
            "measurement_item_commitment_set_sha256": (
                anchor_commitment.item_commitment_set_sha256
            ),
            "measurement_item_count": anchor_commitment.count,
        }:
            raise MuSiQueEvaluatorMeasurementError(
                "M3 A_hold source binding drifted"
            )
    return acquisition, a_evidence, a_cache, a_public, f3_bundle, anchor_bundle


def _execute_formal(
    *,
    stage: str,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    measurement_block_path: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    execution_root: str | Path,
    f3_private_evidence_path: str | Path | None = None,
    f3_public_receipt_path: str | Path | None = None,
    a_hold_private_evidence_path: str | Path | None = None,
    a_hold_report_path: str | Path | None = None,
) -> dict[str, Any]:
    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise MuSiQueEvaluatorMeasurementError(
            "formal evaluator measurement is available only through the clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_freeze(pre_run_freeze_path, stage)
    if freeze["implementation"] != current_evaluator_measurement_implementation_binding(
        project
    ):
        raise MuSiQueEvaluatorMeasurementError(
            "live evaluator measurement implementation drifted"
        )
    root = _canonical_new_root(execution_root)
    if freeze.get("execution_root_hash") != _root_binding_hash(root):
        raise MuSiQueEvaluatorMeasurementError(
            f"{stage} execution-root binding drifted"
        )
    if root.exists() or root.is_symlink():
        raise MuSiQueEvaluatorMeasurementError(
            f"fresh {stage} execution root already exists; replay is forbidden"
        )
    _require_private_execution_boundary(root)
    (
        _acquisition,
        a_evidence,
        _a_cache,
        a_public,
        f3_bundle,
        anchor_bundle,
    ) = _verify_freeze_inputs(
        freeze=freeze,
        stage=stage,
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f3_private_evidence_path=f3_private_evidence_path,
        f3_public_receipt_path=f3_public_receipt_path,
        a_hold_private_evidence_path=a_hold_private_evidence_path,
        a_hold_report_path=a_hold_report_path,
    )
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise MuSiQueEvaluatorMeasurementError(
            f"fresh {stage} execution root already exists; replay is forbidden"
        ) from exc

    current_stage = "authorization_consumption"
    try:
        consumption_body = {
            "schema": CONSUMPTION_SCHEMA,
            "measurement_stage": stage,
            "authorization_hash": freeze["authorization_hash"],
            "freeze_hash": freeze["freeze_hash"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_hash": freeze["execution_root_hash"],
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {
            **consumption_body,
            "consumption_sha256": stable_hash(consumption_body),
        }
        _write_json_exclusive(root / CONSUMPTION_FILENAME, consumption)

        current_stage = f"exact_{stage}_open_after_freeze"
        block = load_measurement_block_after_freeze(
            block_path=measurement_block_path,
            acquisition_receipt_path=acquisition_receipt_path,
            measurement_freeze_path=pre_run_freeze_path,
            expected_block=stage,
        )
        if len(block.items) != ITEM_COUNT:
            raise MuSiQueEvaluatorMeasurementError(
                f"exact {stage} item count drifted"
            )

        current_stage = "maximum_concurrency_gold_free_retrieval"
        measurement_evidence, execution = formation.build_program_retrieval_evidence(
            block
        )
        if execution != {
            **execution,
            "all_terminals_joined_before_support_scoring": True,
        }:
            raise MuSiQueEvaluatorMeasurementError(
                "measurement evidence was scored before full terminal join"
            )
        current_stage = "private_evidence_persistence"
        cache = formation._cache_payload(
            stage=stage,
            block=block,
            evidences=measurement_evidence,
            execution=execution,
        )
        private_evidence_path = root / PRIVATE_EVIDENCE_FILENAME[stage]
        _write_json_exclusive(private_evidence_path, cache)
        cache_file_hash = _sha256_file(private_evidence_path)

        current_stage = "offline_official_support_evaluation"
        if stage == "A_hold":
            core_result = compare_on_fixed_anchor(
                formation_evidence=a_evidence,
                anchor_evidence=measurement_evidence,
                formation_receipt=a_public["core_receipt"],
            )
            report_body = _anchor_report_body(
                freeze=freeze,
                freeze_file_hash=freeze_file_hash,
                cache=cache,
                cache_file_hash=cache_file_hash,
                core_result=core_result,
            )
        else:
            if f3_bundle is None or anchor_bundle is None:
                raise MuSiQueEvaluatorMeasurementError(
                    "M3 frozen evidence bundles are unavailable"
                )
            f3_evidence, _f3_cache, f3_public, _f3_binding_value = f3_bundle
            anchor_evidence, _anchor_cache, anchor_report, _anchor_binding_value = (
                anchor_bundle
            )
            # The core utility function independently recomputes the anchor
            # result from the exact A_hold evidence before scoring M3.
            core_result = measure_prospective_search_utility(
                formation_evidence=f3_evidence,
                measurement_evidence=measurement_evidence,
                evaluator_formation_evidence=a_evidence,
                anchor_evidence=anchor_evidence,
                evaluator_formation_receipt=a_public["core_receipt"],
                search_formation_receipt=f3_public["core_receipt"],
                anchor_result=anchor_report["core_result"],
            )
            report_body = {
                "schema": REPORT_SCHEMA,
                "valid": True,
                "measurement_stage": "M3",
                "freeze_hash": freeze["freeze_hash"],
                "freeze_file_sha256": freeze_file_hash,
                "source_binding": freeze["source_binding"],
                "a_form_binding": freeze["a_form_binding"],
                "f3_binding": freeze["f3_binding"],
                "a_hold_binding": freeze["a_hold_binding"],
                "candidate_set_binding": freeze["candidate_set_binding"],
                "private_evidence_binding": {
                    "file_sha256": cache_file_hash,
                    "cache_sha256": cache["cache_sha256"],
                    "evidence_set_sha256": cache["evidence_set_sha256"],
                    "private_path_persisted_publicly": False,
                    "item_level_evidence_persisted_publicly": False,
                },
                "core_result": core_result,
                "anchor_reverification": {
                    "anchor_result_sha256": anchor_report["core_result"][
                        "anchor_result_sha256"
                    ],
                    "strict_wilson_transition_recomputed_from_exact_A_hold_evidence": True,
                    "completed_before_M3_utility_evaluation": True,
                },
                "execution": execution,
                "raw_content_persisted": False,
            }
        report = {**report_body, "report_hash": stable_hash(report_body)}
        formation._assert_public_safe(report)
        current_stage = "aggregate_report_persistence"
        report_path = root / REPORT_FILENAME[stage]
        _write_json_exclusive(report_path, report)
        persisted, _raw = _read_json_object(report_path, f"persisted {stage} report")
        persisted_body = dict(persisted)
        persisted_hash = persisted_body.pop("report_hash", None)
        if persisted_hash != stable_hash(persisted_body):
            raise MuSiQueEvaluatorMeasurementError(
                f"persisted {stage} report hash drifted"
            )
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "valid": False,
            "measurement_stage": stage,
            "freeze_hash": freeze["freeze_hash"],
            "failure_stage": current_stage,
            "error_type_hash": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (root / CONSUMPTION_FILENAME).is_file(),
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        failure = {**failure_body, "failure_hash": stable_hash(failure_body)}
        try:
            _write_json_exclusive(root / FAILURE_FILENAME[stage], failure)
        except Exception:
            pass
        raise MuSiQueEvaluatorMeasurementError(
            f"formal {stage} run failed and cannot be replayed"
        ) from exc


def execute_a_hold_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    a_hold_block_path: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    """Consume the exact A_hold authorization; no result injection exists."""

    return _execute_formal(
        stage="A_hold",
        project_root=project_root,
        pre_run_freeze_path=pre_run_freeze_path,
        measurement_block_path=a_hold_block_path,
        acquisition_receipt_path=acquisition_receipt_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        execution_root=execution_root,
    )


def execute_m3_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    m3_block_path: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f3_private_evidence_path: str | Path,
    f3_public_receipt_path: str | Path,
    a_hold_private_evidence_path: str | Path,
    a_hold_report_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    """Consume exact M3 authorization after re-verifying A_hold evidence."""

    return _execute_formal(
        stage="M3",
        project_root=project_root,
        pre_run_freeze_path=pre_run_freeze_path,
        measurement_block_path=m3_block_path,
        acquisition_receipt_path=acquisition_receipt_path,
        a_form_private_evidence_path=a_form_private_evidence_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f3_private_evidence_path=f3_private_evidence_path,
        f3_public_receipt_path=f3_public_receipt_path,
        a_hold_private_evidence_path=a_hold_private_evidence_path,
        a_hold_report_path=a_hold_report_path,
        execution_root=execution_root,
    )


def formal_signatures_have_no_injection_surface() -> bool:
    forbidden = {
        "candidate_programs",
        "evidence",
        "operator",
        "operator_factory",
        "result_injection",
        "results",
        "retriever",
        "runner",
    }
    return all(
        forbidden.isdisjoint(inspect.signature(function).parameters)
        for function in (execute_a_hold_formal, execute_m3_formal)
    )


__all__ = [
    "MuSiQueEvaluatorMeasurementError",
    "build_a_hold_pre_run_freeze",
    "build_m3_pre_run_freeze",
    "execute_a_hold_formal",
    "execute_m3_formal",
    "formal_signatures_have_no_injection_surface",
    "load_and_reverify_a_hold_artifacts",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_a = subparsers.add_parser("freeze-a-hold")
    execute_a = subparsers.add_parser("execute-a-hold")
    freeze_m3 = subparsers.add_parser("freeze-m3")
    execute_m3 = subparsers.add_parser("execute-m3")

    for command in (freeze_a, execute_a, freeze_m3, execute_m3):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--acquisition-receipt", type=Path, required=True)
        command.add_argument("--a-form-private-evidence", type=Path, required=True)
        command.add_argument("--a-form-public-receipt", type=Path, required=True)
        command.add_argument("--execution-root", type=Path, required=True)
    for command in (freeze_a, freeze_m3):
        command.add_argument("--authorization-hash", required=True)
        command.add_argument("--output", type=Path, required=True)
    for command in (freeze_m3, execute_m3):
        command.add_argument("--f3-private-evidence", type=Path, required=True)
        command.add_argument("--f3-public-receipt", type=Path, required=True)
        command.add_argument("--a-hold-private-evidence", type=Path, required=True)
        command.add_argument("--a-hold-report", type=Path, required=True)
    execute_a.add_argument("--a-hold-block", type=Path, required=True)
    execute_a.add_argument("--pre-run-freeze", type=Path, required=True)
    execute_m3.add_argument("--m3-block", type=Path, required=True)
    execute_m3.add_argument("--pre-run-freeze", type=Path, required=True)
    arguments = parser.parse_args(argv)
    common = {
        "project_root": arguments.project_root,
        "acquisition_receipt_path": arguments.acquisition_receipt,
        "a_form_private_evidence_path": arguments.a_form_private_evidence,
        "a_form_public_receipt_path": arguments.a_form_public_receipt,
        "execution_root": arguments.execution_root,
    }
    if arguments.command == "freeze-a-hold":
        build_a_hold_pre_run_freeze(
            **common,
            authorization_hash=arguments.authorization_hash,
            output_path=arguments.output,
        )
        return 0
    if arguments.command == "freeze-m3":
        build_m3_pre_run_freeze(
            **common,
            f3_private_evidence_path=arguments.f3_private_evidence,
            f3_public_receipt_path=arguments.f3_public_receipt,
            a_hold_private_evidence_path=arguments.a_hold_private_evidence,
            a_hold_report_path=arguments.a_hold_report,
            authorization_hash=arguments.authorization_hash,
            output_path=arguments.output,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        if arguments.command == "execute-a-hold":
            execute_a_hold_formal(
                **common,
                pre_run_freeze_path=arguments.pre_run_freeze,
                a_hold_block_path=arguments.a_hold_block,
            )
        else:
            execute_m3_formal(
                **common,
                pre_run_freeze_path=arguments.pre_run_freeze,
                m3_block_path=arguments.m3_block,
                f3_private_evidence_path=arguments.f3_private_evidence,
                f3_public_receipt_path=arguments.f3_public_receipt,
                a_hold_private_evidence_path=arguments.a_hold_private_evidence,
                a_hold_report_path=arguments.a_hold_report,
            )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

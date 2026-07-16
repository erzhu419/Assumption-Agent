"""One-shot MuSiQue M2 recursive-retention and official-core runner.

The freeze command cannot express an M2 path.  It binds the public acquisition
commitment for M2, the exact F1/P and F2/Q formation artifacts, the successful
positive M1 promotion chain for P, a fresh v2 official-HippoRAG filesystem
attestation, one authorization, one fresh execution root, and the complete
implementation closure.  Only the clean ``execute`` module command can open
the exact M2 file, and it does so after authorization consumption.

Canonical RAW is a deterministic, zero-call first-five baseline.  P, Q, and
official HippoRAG contribute exactly ``3 * n`` retrieval calls in one
maximum-width executor.  Callables see only :class:`RetrievalStudyItem`, which
contains no support or answer labels.  After every retrieval terminal joins
and a fresh runtime postflight succeeds, local official-support scoring derives
the frozen L4 ``empty/P/Q/P+Q`` RRF arms and a homologous direct top-five
RAW/P/Q/official-HippoRAG comparison.  There is no generator, network judge,
online evaluator, retry, replay, resample, callback, or result-injection API.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from .l4_retention_protocol_v1 import (
    ARM_IDS as L4_ARM_IDS,
    PRIMARY_METRIC as L4_PRIMARY_METRIC,
    RANKING_REUSE_POLICY,
    RRF_CONSTANT,
    TOP_K,
    RetentionItem,
    _score_all_arms,
    deterministic_rrf,
)
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .musique_recursive_study_blocks_v1 import (
    RetrievalStudyItem,
    StudyAcquisitionBinding,
    StudyItem,
    load_measurement_block_after_freeze,
    load_study_acquisition_binding,
    load_study_frozen_program,
)
from .musique_typed_retriever_formation_v1 import (
    TypedRetrievalProgram,
    retrieve as typed_retrieve,
)


RUNNER_VERSION = "musique_generation_two_m2_recursive_retention_runner_v1"
FREEZE_SCHEMA = "musique_generation_two_m2_pre_run_freeze_v1"
REPORT_SCHEMA = "musique_generation_two_m2_aggregate_report_v1"
FAILURE_SCHEMA = "musique_generation_two_m2_failure_v1"
M1_FREEZE_SCHEMA = "musique_generation_one_m1_pre_run_freeze_v1"
M1_REPORT_SCHEMA = "musique_generation_one_m1_aggregate_report_v1"
M1_PROMOTION_POLICY = (
    "promote_frozen_P_iff_total_support_hits_P_minus_canonical_RAW_is_strictly_positive_v1"
)
M1_POSITIVE_DISPOSITION = "promote_P_to_retained_generation_one"
M2_ITEM_COUNT = 12
RETRIEVAL_COMPONENT_IDS = ("P", "Q", "official_HippoRAG")
COMPARISON_ARM_IDS = (
    "canonical_RAW",
    "recursive_typed_retrieval",
    "official_HippoRAG",
)
WORK_UNIT_COUNT = len(RETRIEVAL_COMPONENT_IDS) * M2_ITEM_COUNT
MAXIMUM_CONCURRENCY = WORK_UNIT_COUNT
FREEZE_DECISION = (
    "authorize_exact_M2_recursive_retention_once_after_positive_M1_and_full_pre_run_freeze"
)
CONSUMPTION_FILENAME = "m2.execution.authorization.consumed.json"
REPORT_FILENAME = "m2.aggregate.report.json"
FAILURE_FILENAME = "m2.failure.json"
IMPLEMENTATION_SCHEMA = "musique_generation_two_m2_implementation_v1"
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/musique_m2_retention_runner_v1.py",
    "assumption_agent/benchmarks/l4_retention_protocol_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_recursive_study_acquisition_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
)
_CLEAN_MODULE_CLI_ACTIVE = False
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MuSiQueM2RunnerError(RuntimeError):
    """The frozen M2 formal contract failed closed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueM2RunnerError("required frozen file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueM2RunnerError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _absolute_no_symlink(path: str | Path, field_name: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueM2RunnerError(
                f"{field_name} contains a symlink component"
            )
    return candidate


def _canonical_new_root(path: str | Path) -> Path:
    candidate = _absolute_no_symlink(path, "M2 execution root")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise MuSiQueM2RunnerError(
            "M2 execution root parent is unavailable"
        ) from exc
    if not parent.is_dir():
        raise MuSiQueM2RunnerError("M2 execution root parent is not a directory")
    return parent / candidate.name


def _root_binding_hash(path: str | Path) -> str:
    return stable_hash(
        {"absolute_execution_root": str(_canonical_new_root(path))}
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
        allow_nan=False,
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


def _read_json_object(
    path: str | Path, field_name: str
) -> tuple[dict[str, Any], bytes]:
    candidate = _absolute_no_symlink(path, field_name)
    if not candidate.is_file():
        raise MuSiQueM2RunnerError(f"{field_name} is unavailable")
    try:
        descriptor = os.open(
            candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueM2RunnerError(f"{field_name} is invalid") from exc
    if not isinstance(payload, dict):
        raise MuSiQueM2RunnerError(f"{field_name} must contain one object")
    return payload, raw


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answers"',
        '"corpus"',
        '"item_id"',
        '"normalized_answers"',
        '"paragraph_text"',
        '"private_root"',
        '"question"',
        '"source_row_sha256"',
        '"support_indices"',
    )
    if any(token in serialized for token in forbidden):
        raise MuSiQueM2RunnerError(
            "public M2 artifact contains private content or locator keys"
        )


def current_m2_implementation_binding(
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueM2RunnerError(
                f"M2 implementation file is missing: {relative}"
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
        raise MuSiQueM2RunnerError("M2 implementation binding is malformed")
    if value.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueM2RunnerError("M2 implementation schema drifted")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(
        IMPLEMENTATION_RELATIVE_FILES
    ):
        raise MuSiQueM2RunnerError("M2 implementation set drifted")
    normalized: list[dict[str, str]] = []
    for relative, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row.get("path") != relative
        ):
            raise MuSiQueM2RunnerError("M2 implementation row drifted")
        normalized.append(
            {
                "path": relative,
                "sha256": _require_sha256(
                    row.get("sha256"), "M2 implementation file hash"
                ),
            }
        )
    set_hash = _require_sha256(
        value.get("set_sha256"), "M2 implementation set hash"
    )
    if stable_hash(normalized) != set_hash:
        raise MuSiQueM2RunnerError("M2 implementation set hash drifted")
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": normalized,
        "set_sha256": set_hash,
    }


@dataclass(frozen=True)
class OfficialRuntimePaths:
    runtime_python: Path = field(repr=False)
    local_llm_model: Path = field(repr=False)
    local_embedding_model: Path = field(repr=False)
    base_binding_receipt_path: Path = field(repr=False)
    attestation_receipt_path: Path = field(repr=False)

    def verify(self) -> None:
        if not self.runtime_python.absolute().is_file():
            raise MuSiQueM2RunnerError("runtime Python is unavailable")
        for path in (self.local_llm_model, self.local_embedding_model):
            if not path.resolve(strict=True).is_dir():
                raise MuSiQueM2RunnerError("local runtime asset is unavailable")
        for path in (
            self.base_binding_receipt_path,
            self.attestation_receipt_path,
        ):
            if path.is_symlink() or not path.is_file():
                raise MuSiQueM2RunnerError(
                    "official runtime receipt is unavailable"
                )


def _runtime_paths(
    *,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> OfficialRuntimePaths:
    paths = OfficialRuntimePaths(
        runtime_python=Path(runtime_python).absolute(),
        local_llm_model=Path(local_llm_model).resolve(strict=True),
        local_embedding_model=Path(local_embedding_model).resolve(strict=True),
        base_binding_receipt_path=_absolute_no_symlink(
            base_binding_receipt_path, "base binding receipt"
        ),
        attestation_receipt_path=_absolute_no_symlink(
            attestation_receipt_path, "v2 attestation receipt"
        ),
    )
    paths.verify()
    return paths


def _prepare_runtime(
    *, project: Path, runtime: OfficialRuntimePaths
) -> PreparedFormalRuntimeV2:
    return prepare_formal_runtime_v2(
        project_root=project,
        attestation_receipt_path=runtime.attestation_receipt_path,
        base_binding_receipt_path=runtime.base_binding_receipt_path,
        runtime_python=runtime.runtime_python,
        local_llm_model=runtime.local_llm_model,
        local_embedding_model=runtime.local_embedding_model,
    )


def _program_binding(
    *,
    formation_receipt_path: Path,
    frozen_program_path: Path,
    project: Path,
    expected_block: str,
) -> tuple[TypedRetrievalProgram, dict[str, Any]]:
    program, receipt, envelope = load_study_frozen_program(
        frozen_program_path=frozen_program_path,
        formation_receipt_path=formation_receipt_path,
        verify_live=True,
        implementation_root=project,
    )
    expected_block_hash = stable_hash({"block": expected_block})
    if (
        receipt.get("formation_block_id_hash") != expected_block_hash
        or envelope.get("formation_block_id_hash") != expected_block_hash
    ):
        raise MuSiQueM2RunnerError(
            f"frozen program was not formed on exact {expected_block}"
        )
    binding = {
        "formation_receipt_file_sha256": _sha256_file(
            formation_receipt_path
        ),
        "formation_receipt_hash": _require_sha256(
            receipt.get("receipt_hash"), "formation receipt hash"
        ),
        "frozen_program_file_sha256": _sha256_file(frozen_program_path),
        "frozen_program_envelope_hash": _require_sha256(
            envelope.get("envelope_hash"), "frozen program envelope hash"
        ),
        "program_hash": program.program_hash,
        "formed_on_block_id_hash": expected_block_hash,
    }
    return program, binding


def _load_self_hashed(
    *, path: str | Path, field_name: str, hash_key: str
) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_json_object(path, field_name)
    body = dict(payload)
    declared = _require_sha256(body.pop(hash_key, None), f"{field_name} hash")
    if stable_hash(body) != declared:
        raise MuSiQueM2RunnerError(f"{field_name} self-hash drifted")
    return payload, raw


def _load_positive_m1_chain(
    *,
    m1_pre_run_freeze_path: str | Path,
    m1_promotion_report_path: str | Path,
    acquisition: StudyAcquisitionBinding,
    p_binding: Mapping[str, Any],
) -> dict[str, Any]:
    freeze, freeze_raw = _load_self_hashed(
        path=m1_pre_run_freeze_path,
        field_name="M1 pre-run freeze",
        hash_key="freeze_hash",
    )
    source = freeze.get("source_binding")
    frozen_p = freeze.get("p_operator_binding")
    m1_contract = freeze.get("execution_contract")
    m1_ordering = freeze.get("ordering")
    m1_commitment = acquisition.commitment_for("M1")
    if (
        set(freeze)
        != {
            "authorization_hash",
            "decision",
            "execution_contract",
            "execution_root_hash",
            "freeze_hash",
            "implementation",
            "official_runtime_binding",
            "ordering",
            "p_operator_binding",
            "raw_content_persisted",
            "schema",
            "source_binding",
        }
        or freeze.get("schema") != M1_FREEZE_SCHEMA
        or freeze.get("decision")
        != "authorize_exact_M1_retrieval_only_once_after_full_pre_run_freeze"
        or freeze.get("raw_content_persisted") is not False
        or not isinstance(source, Mapping)
        or not isinstance(frozen_p, Mapping)
        or not isinstance(m1_contract, Mapping)
        or not isinstance(m1_ordering, Mapping)
        or dict(frozen_p) != dict(p_binding)
        or source.get("acquisition_sha256") != acquisition.acquisition_sha256
        or source.get("acquisition_file_sha256")
        != acquisition.acquisition_file_sha256
        or source.get("private_pack_sha256") != acquisition.private_pack_sha256
        or source.get("measurement_block_id_hash")
        != stable_hash({"block": "M1"})
        or source.get("measurement_block_file_sha256")
        != m1_commitment.file_sha256
        or source.get("measurement_item_commitment_set_sha256")
        != m1_commitment.item_commitment_set_sha256
        or source.get("measurement_item_count") != m1_commitment.count
        or m1_contract.get("arms")
        != ["canonical_RAW", "frozen_P", "official_HippoRAG"]
        or m1_contract.get("item_count") != M2_ITEM_COUNT
        or m1_contract.get("work_unit_count") != 3 * M2_ITEM_COUNT
        or m1_contract.get("maximum_concurrency") != 3 * M2_ITEM_COUNT
        or m1_contract.get("promotion_policy") != M1_PROMOTION_POLICY
        or any(
            m1_contract.get(field) != 0
            for field in (
                "generator_calls",
                "external_network_calls",
                "online_evaluator_calls",
                "retries",
                "replays",
                "resamples",
            )
        )
        or m1_ordering.get("measurement_block_rows_read_while_freezing") != 0
        or m1_ordering.get("measurement_support_labels_read_while_freezing")
        != 0
        or m1_ordering.get("pre_run_freeze_complete_before_measurement_open")
        is not True
    ):
        raise MuSiQueM2RunnerError("M1 freeze does not bind the retained P chain")

    report, report_raw = _load_self_hashed(
        path=m1_promotion_report_path,
        field_name="positive M1 promotion report",
        hash_key="report_hash",
    )
    measurement = report.get("measurement")
    if not isinstance(measurement, Mapping):
        raise MuSiQueM2RunnerError("M1 promotion measurement is malformed")
    disposition = measurement.get("promotion_disposition")
    paired = measurement.get("paired_P_minus_RAW")
    arm_metrics = measurement.get("arm_metrics")
    m1_execution = report.get("execution")
    m1_runtime = report.get("runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (
            disposition,
            paired,
            arm_metrics,
            m1_execution,
            m1_runtime,
        )
    ):
        raise MuSiQueM2RunnerError("M1 promotion evidence is malformed")
    p_metric = arm_metrics.get("frozen_P")
    raw_metric = arm_metrics.get("canonical_RAW")
    if not isinstance(p_metric, Mapping) or not isinstance(raw_metric, Mapping):
        raise MuSiQueM2RunnerError("M1 promotion arm evidence is malformed")
    p_hits = p_metric.get("support_hit_count")
    raw_hits = raw_metric.get("support_hit_count")
    net = paired.get("net_support_hit_count")
    if (
        set(report)
        != {
            "execution",
            "freeze_file_sha256",
            "freeze_hash",
            "measurement",
            "measurement_block_file_sha256",
            "measurement_block_id_hash",
            "raw_content_persisted",
            "report_hash",
            "runtime",
            "schema",
            "sealed_or_test_content_accessed",
            "valid",
        }
        or report.get("schema") != M1_REPORT_SCHEMA
        or report.get("valid") is not True
        or report.get("raw_content_persisted") is not False
        or report.get("sealed_or_test_content_accessed") is not False
        or report.get("freeze_hash") != freeze.get("freeze_hash")
        or report.get("freeze_file_sha256") != _sha256_bytes(freeze_raw)
        or report.get("measurement_block_id_hash")
        != stable_hash({"block": "M1"})
        or report.get("measurement_block_file_sha256")
        != m1_commitment.file_sha256
        or type(p_hits) is not int
        or type(raw_hits) is not int
        or type(net) is not int
        or net != p_hits - raw_hits
        or net <= 0
        or disposition.get("policy") != M1_PROMOTION_POLICY
        or disposition.get("positive_net_support") is not True
        or disposition.get("disposition") != M1_POSITIVE_DISPOSITION
        or disposition.get("archive_mutated_by_runner") is not False
        or m1_execution.get("item_count") != M2_ITEM_COUNT
        or m1_execution.get("work_unit_count") != 3 * M2_ITEM_COUNT
        or m1_execution.get("retrieval_call_count") != 3 * M2_ITEM_COUNT
        or m1_execution.get("retrieval_terminal_count") != 3 * M2_ITEM_COUNT
        or m1_execution.get("configured_maximum_concurrency")
        != 3 * M2_ITEM_COUNT
        or m1_execution.get("all_terminals_joined_before_support_scoring")
        is not True
        or any(
            m1_execution.get(field) != 0
            for field in (
                "generator_calls",
                "external_network_calls",
                "online_evaluator_calls",
                "retries",
                "replays",
                "resamples",
            )
        )
        or m1_runtime.get("formal_entry_executable_identity_probe_calls") != 0
        or m1_runtime.get("official_arm_terminal_count") != M2_ITEM_COUNT
        or m1_runtime.get("postflight_fresh_filesystem_attestation") is not True
    ):
        raise MuSiQueM2RunnerError(
            "M1 did not positively promote the exact retained P program"
        )
    binding = {
        "m1_freeze_hash": freeze["freeze_hash"],
        "m1_freeze_file_sha256": _sha256_bytes(freeze_raw),
        "m1_report_hash": report["report_hash"],
        "m1_report_file_sha256": _sha256_bytes(report_raw),
        "p_program_hash": p_binding["program_hash"],
        "positive_net_support_hit_count": net,
        "promotion_policy": M1_PROMOTION_POLICY,
        "disposition": M1_POSITIVE_DISPOSITION,
    }
    return binding


def _runtime_binding(
    *, prepared: PreparedFormalRuntimeV2, runtime: OfficialRuntimePaths
) -> dict[str, Any]:
    safe = prepared.safe_binding
    if safe.get("formal_entry_executable_identity_probe_calls") != 0:
        raise MuSiQueM2RunnerError(
            "prospective v2 attestation attempted an executable probe"
        )
    return {
        "prepared_safe_binding": safe,
        "base_binding_file_sha256": _sha256_file(
            runtime.base_binding_receipt_path
        ),
        "attestation_file_sha256": _sha256_file(
            runtime.attestation_receipt_path
        ),
        "pre_run_fresh_filesystem_attestation": True,
        "formal_entry_uses_filesystem_attestation_only": True,
        "post_run_fresh_filesystem_attestation_before_scoring": True,
    }


def build_m2_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    m1_pre_run_freeze_path: str | Path,
    m1_promotion_report_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze M2 completely without accepting or opening an M2 path."""

    project = Path(project_root).resolve(strict=True)
    acquisition_path = _absolute_no_symlink(
        acquisition_receipt_path, "study acquisition receipt"
    )
    p_receipt_path = _absolute_no_symlink(
        p_formation_receipt_path, "F1/P formation receipt"
    )
    p_program_path = _absolute_no_symlink(
        p_frozen_program_path, "F1/P frozen program"
    )
    q_receipt_path = _absolute_no_symlink(
        q_formation_receipt_path, "F2/Q formation receipt"
    )
    q_program_path = _absolute_no_symlink(
        q_frozen_program_path, "F2/Q frozen program"
    )
    acquisition = load_study_acquisition_binding(acquisition_path)
    m2_commitment = acquisition.commitment_for("M2")
    _p_program, p_binding = _program_binding(
        formation_receipt_path=p_receipt_path,
        frozen_program_path=p_program_path,
        project=project,
        expected_block="F1",
    )
    _q_program, q_binding = _program_binding(
        formation_receipt_path=q_receipt_path,
        frozen_program_path=q_program_path,
        project=project,
        expected_block="F2",
    )
    promotion_binding = _load_positive_m1_chain(
        m1_pre_run_freeze_path=m1_pre_run_freeze_path,
        m1_promotion_report_path=m1_promotion_report_path,
        acquisition=acquisition,
        p_binding=p_binding,
    )
    runtime = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = _prepare_runtime(project=project, runtime=runtime)
    body: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "decision": FREEZE_DECISION,
        "implementation": current_m2_implementation_binding(project),
        "authorization_hash": _require_sha256(
            authorization_hash, "M2 execution authorization"
        ),
        "execution_root_hash": _root_binding_hash(execution_root),
        "source_binding": {
            "acquisition_sha256": acquisition.acquisition_sha256,
            "acquisition_file_sha256": acquisition.acquisition_file_sha256,
            "private_pack_sha256": acquisition.private_pack_sha256,
            "measurement_block_id_hash": stable_hash({"block": "M2"}),
            "measurement_block_file_sha256": m2_commitment.file_sha256,
            "measurement_item_commitment_set_sha256": (
                m2_commitment.item_commitment_set_sha256
            ),
            "measurement_item_count": m2_commitment.count,
        },
        "recursive_operator_binding": {
            "P": p_binding,
            "Q": q_binding,
            "lineage": "P_formed_on_F1_promoted_on_M1_then_Q_formed_on_F2_v1",
            "P_and_Q_calls_reused_across_all_L4_arms": True,
        },
        "positive_m1_promotion_binding": promotion_binding,
        "official_runtime_binding": _runtime_binding(
            prepared=prepared, runtime=runtime
        ),
        "execution_contract": {
            "retrieval_components": list(RETRIEVAL_COMPONENT_IDS),
            "zero_call_baseline": "canonical_RAW_first_five_corpus_indices_v1",
            "direct_comparison_arms": list(COMPARISON_ARM_IDS),
            "derived_l4_arms": list(L4_ARM_IDS),
            "item_count": M2_ITEM_COUNT,
            "top_k": TOP_K,
            "rrf_constant": RRF_CONSTANT,
            "retrieval_work_unit_count": WORK_UNIT_COUNT,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "formal_entry": "clean_python_module_cli_v1",
            "all_retrieval_terminals_joined_before_support_scoring": True,
            "fresh_runtime_postflight_before_support_scoring": True,
            "primary_metric": L4_PRIMARY_METRIC,
            "ranking_reuse_policy": RANKING_REUSE_POLICY,
            "generator_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "outcome_dependent_arm_changes": 0,
        },
        "ordering": {
            "measurement_block_rows_read_while_freezing": 0,
            "measurement_support_labels_read_while_freezing": 0,
            "pre_run_freeze_complete_before_measurement_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_hash": stable_hash(body)}
    _assert_public_safe(freeze)
    output = _absolute_no_symlink(output_path, "M2 pre-run freeze output")
    if output.exists():
        raise MuSiQueM2RunnerError("M2 pre-run freeze output already exists")
    _write_json_exclusive(output, freeze)
    return freeze


def _load_freeze(path: str | Path) -> tuple[dict[str, Any], str]:
    payload, raw = _load_self_hashed(
        path=path, field_name="M2 pre-run freeze", hash_key="freeze_hash"
    )
    expected_keys = {
        "authorization_hash",
        "decision",
        "execution_contract",
        "execution_root_hash",
        "freeze_hash",
        "implementation",
        "official_runtime_binding",
        "ordering",
        "positive_m1_promotion_binding",
        "raw_content_persisted",
        "recursive_operator_binding",
        "schema",
        "source_binding",
    }
    contract = payload.get("execution_contract")
    ordering = payload.get("ordering")
    if (
        set(payload) != expected_keys
        or payload.get("schema") != FREEZE_SCHEMA
        or payload.get("decision") != FREEZE_DECISION
        or payload.get("raw_content_persisted") is not False
        or not isinstance(contract, Mapping)
        or not isinstance(ordering, Mapping)
        or contract.get("retrieval_components")
        != list(RETRIEVAL_COMPONENT_IDS)
        or contract.get("direct_comparison_arms") != list(COMPARISON_ARM_IDS)
        or contract.get("derived_l4_arms") != list(L4_ARM_IDS)
        or contract.get("item_count") != M2_ITEM_COUNT
        or contract.get("top_k") != TOP_K
        or contract.get("rrf_constant") != RRF_CONSTANT
        or contract.get("retrieval_work_unit_count") != WORK_UNIT_COUNT
        or contract.get("maximum_concurrency") != MAXIMUM_CONCURRENCY
        or contract.get("formal_entry") != "clean_python_module_cli_v1"
        or contract.get("primary_metric") != L4_PRIMARY_METRIC
        or contract.get("ranking_reuse_policy") != RANKING_REUSE_POLICY
        or contract.get("zero_call_baseline")
        != "canonical_RAW_first_five_corpus_indices_v1"
        or contract.get(
            "all_retrieval_terminals_joined_before_support_scoring"
        )
        is not True
        or contract.get("fresh_runtime_postflight_before_support_scoring")
        is not True
        or any(
            contract.get(field) != 0
            for field in (
                "generator_calls",
                "external_network_calls",
                "online_evaluator_calls",
                "retries",
                "replays",
                "resamples",
                "outcome_dependent_arm_changes",
            )
        )
        or ordering.get("measurement_block_rows_read_while_freezing") != 0
        or ordering.get("measurement_support_labels_read_while_freezing") != 0
        or ordering.get("pre_run_freeze_complete_before_measurement_open")
        is not True
    ):
        raise MuSiQueM2RunnerError("M2 pre-run freeze drifted")
    _require_sha256(payload.get("authorization_hash"), "M2 authorization hash")
    _require_sha256(payload.get("execution_root_hash"), "M2 root binding hash")
    _validate_implementation_binding(payload.get("implementation"))
    _assert_public_safe(payload)
    return payload, _sha256_bytes(raw)


def _p_retrieve(
    program: TypedRetrievalProgram, item: RetrievalStudyItem
) -> tuple[int, ...]:
    return typed_retrieve(program, item.question, item.corpus)


def _q_retrieve(
    program: TypedRetrievalProgram, item: RetrievalStudyItem
) -> tuple[int, ...]:
    return typed_retrieve(program, item.question, item.corpus)


def _official_retrieve(
    item: RetrievalStudyItem,
    runtime: PreparedFormalRuntimeV2,
    work_root: Path,
) -> tuple[int, ...]:
    return runtime.retrieve(
        question=item.question,
        paragraphs=item.hipporag_paragraphs(),
        work_root=work_root,
    )


@dataclass(frozen=True)
class _WorkUnit:
    ordinal: int
    component_id: str
    item: RetrievalStudyItem = field(repr=False)

    @property
    def key(self) -> tuple[int, str]:
        return self.ordinal, self.component_id


def _validate_ranking(
    value: Sequence[int], *, item: RetrievalStudyItem
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise MuSiQueM2RunnerError("retrieval output is not an index sequence")
    try:
        ranking = tuple(value)
    except TypeError as exc:
        raise MuSiQueM2RunnerError(
            "retrieval output is not an index sequence"
        ) from exc
    if (
        len(ranking) != TOP_K
        or len(set(ranking)) != TOP_K
        or any(
            type(index) is not int or not 0 <= index < len(item.corpus)
            for index in ranking
        )
    ):
        raise MuSiQueM2RunnerError(
            "retrieval output violates exact top-five idx contract"
        )
    return ranking


def _document_id(index: int) -> str:
    return f"document_{index:03d}"


@dataclass(frozen=True)
class _L4ScoringPlan:
    baseline_candidate_budget: int = TOP_K


def _aggregate_component(
    *,
    arm_id: str,
    items: Sequence[StudyItem],
    rankings: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item, ranking in zip(items, rankings):
        supports = frozenset(item.support_indices)
        rows.append(
            {
                "item_id_hash": item.item_id_hash,
                "support_total": len(supports),
                "support_hits": len(supports.intersection(ranking)),
                "ranking_hash": stable_hash(
                    {"retrieved_indices": list(ranking)}
                ),
            }
        )
    support_total = sum(row["support_total"] for row in rows)
    support_hits = sum(row["support_hits"] for row in rows)
    return {
        "arm_id": arm_id,
        "item_count": len(rows),
        "support_hit_count": support_hits,
        "support_total": support_total,
        "support_recall_at_5": float(Fraction(support_hits, support_total)),
        "items_with_any_support_hit": sum(
            row["support_hits"] > 0 for row in rows
        ),
        "ranking_score_closure_hash": stable_hash(rows),
    }


def _score_measurement(
    *,
    items: Sequence[StudyItem],
    rankings: Mapping[tuple[int, str], tuple[int, ...]],
) -> dict[str, Any]:
    """First and only support-label dereference point, after postflight."""

    retention_items: list[RetentionItem] = []
    l4_rankings: dict[tuple[str, str], tuple[str, ...]] = {}
    direct_rankings: dict[str, list[tuple[int, ...]]] = {
        arm_id: [] for arm_id in COMPARISON_ARM_IDS
    }
    for ordinal, item in enumerate(items):
        document_ids = tuple(_document_id(paragraph.idx) for paragraph in item.corpus)
        index_by_document_id = {
            document_id: paragraph.idx
            for document_id, paragraph in zip(document_ids, item.corpus)
        }
        raw = tuple(paragraph.idx for paragraph in item.corpus[:TOP_K])
        p = rankings[(ordinal, "P")]
        q = rankings[(ordinal, "Q")]
        official = rankings[(ordinal, "official_HippoRAG")]
        retention = RetentionItem(
            item_id=item.item_id,
            block_id="M2",
            document_ids=document_ids,
            support_document_ids=tuple(
                _document_id(index) for index in item.support_indices
            ),
            operator_input={"schema": "gold_free_scoring_placeholder_v1"},
            baseline_ranked_document_ids=tuple(
                _document_id(index) for index in raw
            ),
        )
        retention_items.append(retention)
        l4_rankings[(retention.item_id_hash, "P")] = tuple(
            _document_id(index) for index in p
        )
        l4_rankings[(retention.item_id_hash, "Q")] = tuple(
            _document_id(index) for index in q
        )
        recursive = tuple(
            index_by_document_id[document_id]
            for document_id in deterministic_rrf(
                (
                    tuple(_document_id(index) for index in raw),
                    l4_rankings[(retention.item_id_hash, "P")],
                    l4_rankings[(retention.item_id_hash, "Q")],
                )
            )
        )
        direct_rankings["canonical_RAW"].append(raw)
        direct_rankings["recursive_typed_retrieval"].append(recursive)
        direct_rankings["official_HippoRAG"].append(official)

    l4 = _score_all_arms(
        plan=_L4ScoringPlan(),
        items=retention_items,
        rankings=l4_rankings,
    )
    direct = {
        arm_id: _aggregate_component(
            arm_id=arm_id,
            items=items,
            rankings=direct_rankings[arm_id],
        )
        for arm_id in COMPARISON_ARM_IDS
    }
    official_hits = direct["official_HippoRAG"]["support_hit_count"]
    support_total = direct["official_HippoRAG"]["support_total"]
    official_comparisons = {
        arm_id: {
            "support_hit_delta_official_minus_comparator": (
                official_hits - metric["support_hit_count"]
            ),
            "support_recall_delta_official_minus_comparator": float(
                Fraction(
                    official_hits - metric["support_hit_count"], support_total
                )
            ),
        }
        for arm_id, metric in direct.items()
        if arm_id != "official_HippoRAG"
    }
    return {
        "primary_metric": L4_PRIMARY_METRIC,
        "l4_recursive_retention": l4,
        "homologous_direct_top5_comparison": {
            "same_exact_M2_items": True,
            "same_official_support_labels": True,
            "same_top_k": TOP_K,
            "recursive_typed_retrieval_is_exact_l4_P_plus_Q_ranking": True,
            "arm_metrics": direct,
            "official_HippoRAG_comparisons": official_comparisons,
        },
        "support_scoring_after_all_terminals_and_runtime_postflight": True,
        "raw_item_document_or_support_ids_persisted": False,
    }


def execute_m2_retention_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    m2_block_path: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    q_formation_receipt_path: str | Path,
    q_frozen_program_path: str | Path,
    m1_pre_run_freeze_path: str | Path,
    m1_promotion_report_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    """Consume the sole M2 authorization; no callable/result injection exists."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise MuSiQueM2RunnerError(
            "formal M2 execution is available only through the clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_freeze(pre_run_freeze_path)
    if freeze["implementation"] != current_m2_implementation_binding(project):
        raise MuSiQueM2RunnerError("live M2 implementation drifted")
    root = _canonical_new_root(execution_root)
    if freeze.get("execution_root_hash") != _root_binding_hash(root):
        raise MuSiQueM2RunnerError("M2 execution root binding drifted")
    if root.exists() or root.is_symlink():
        raise MuSiQueM2RunnerError(
            "fresh M2 execution root already exists; replay is forbidden"
        )

    acquisition_path = _absolute_no_symlink(
        acquisition_receipt_path, "study acquisition receipt"
    )
    p_receipt_path = _absolute_no_symlink(
        p_formation_receipt_path, "F1/P formation receipt"
    )
    p_program_path = _absolute_no_symlink(
        p_frozen_program_path, "F1/P frozen program"
    )
    q_receipt_path = _absolute_no_symlink(
        q_formation_receipt_path, "F2/Q formation receipt"
    )
    q_program_path = _absolute_no_symlink(
        q_frozen_program_path, "F2/Q frozen program"
    )
    acquisition = load_study_acquisition_binding(acquisition_path)
    m2_commitment = acquisition.commitment_for("M2")
    source = freeze.get("source_binding")
    if (
        not isinstance(source, Mapping)
        or source.get("acquisition_sha256") != acquisition.acquisition_sha256
        or source.get("acquisition_file_sha256")
        != acquisition.acquisition_file_sha256
        or source.get("private_pack_sha256") != acquisition.private_pack_sha256
        or source.get("measurement_block_id_hash")
        != stable_hash({"block": "M2"})
        or source.get("measurement_block_file_sha256")
        != m2_commitment.file_sha256
        or source.get("measurement_item_commitment_set_sha256")
        != m2_commitment.item_commitment_set_sha256
        or source.get("measurement_item_count") != M2_ITEM_COUNT
    ):
        raise MuSiQueM2RunnerError("M2 source binding drifted")
    p_program, p_binding = _program_binding(
        formation_receipt_path=p_receipt_path,
        frozen_program_path=p_program_path,
        project=project,
        expected_block="F1",
    )
    q_program, q_binding = _program_binding(
        formation_receipt_path=q_receipt_path,
        frozen_program_path=q_program_path,
        project=project,
        expected_block="F2",
    )
    recursive = freeze.get("recursive_operator_binding")
    expected_recursive = {
        "P": p_binding,
        "Q": q_binding,
        "lineage": "P_formed_on_F1_promoted_on_M1_then_Q_formed_on_F2_v1",
        "P_and_Q_calls_reused_across_all_L4_arms": True,
    }
    if recursive != expected_recursive:
        raise MuSiQueM2RunnerError("M2 recursive operator binding drifted")
    promotion = _load_positive_m1_chain(
        m1_pre_run_freeze_path=m1_pre_run_freeze_path,
        m1_promotion_report_path=m1_promotion_report_path,
        acquisition=acquisition,
        p_binding=p_binding,
    )
    if freeze.get("positive_m1_promotion_binding") != promotion:
        raise MuSiQueM2RunnerError("M2 positive M1 promotion binding drifted")
    runtime = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = _prepare_runtime(project=project, runtime=runtime)
    safe_runtime_binding = prepared.safe_binding
    if freeze.get("official_runtime_binding") != _runtime_binding(
        prepared=prepared, runtime=runtime
    ):
        raise MuSiQueM2RunnerError("M2 official runtime binding drifted")

    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise MuSiQueM2RunnerError(
            "fresh M2 execution root already exists; replay is forbidden"
        ) from exc
    stage = "authorization_consumption"
    attempted = 0
    completed = 0
    counter_lock = threading.Lock()
    try:
        consumption_body = {
            "schema": f"{RUNNER_VERSION}_authorization_consumption",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_hash": freeze["freeze_hash"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_hash": freeze["execution_root_hash"],
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {
            **consumption_body,
            "consumption_hash": stable_hash(consumption_body),
        }
        _write_json_exclusive(root / CONSUMPTION_FILENAME, consumption)

        stage = "exact_M2_open_after_freeze"
        block = load_measurement_block_after_freeze(
            block_path=m2_block_path,
            acquisition_receipt_path=acquisition_path,
            measurement_freeze_path=pre_run_freeze_path,
            expected_block="M2",
        )
        if len(block.items) != M2_ITEM_COUNT:
            raise MuSiQueM2RunnerError("exact M2 item count drifted")
        work_units = tuple(
            _WorkUnit(
                ordinal=ordinal,
                component_id=component_id,
                item=item.retrieval_view(),
            )
            for ordinal, item in enumerate(block.items)
            for component_id in RETRIEVAL_COMPONENT_IDS
        )
        if len(work_units) != WORK_UNIT_COUNT:
            raise MuSiQueM2RunnerError("M2 work-unit grid drifted")

        stage = "retrieval_execution"

        def run_one(work: _WorkUnit) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            with counter_lock:
                attempted += 1
            if work.component_id == "P":
                value = _p_retrieve(p_program, work.item)
            elif work.component_id == "Q":
                value = _q_retrieve(q_program, work.item)
            elif work.component_id == "official_HippoRAG":
                value = _official_retrieve(
                    work.item,
                    prepared,
                    root / f"official_item_{work.ordinal:02d}",
                )
            else:  # pragma: no cover - frozen constant grid
                raise MuSiQueM2RunnerError("unknown frozen M2 component")
            ranking = _validate_ranking(value, item=work.item)
            with counter_lock:
                completed += 1
            return work.key, ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAXIMUM_CONCURRENCY,
            thread_name_prefix="musique-m2",
        ) as executor:
            futures = [executor.submit(run_one, work) for work in work_units]
            terminal_rows = [future.result() for future in futures]
        if attempted != WORK_UNIT_COUNT or completed != WORK_UNIT_COUNT:
            raise MuSiQueM2RunnerError("M2 terminal closure is incomplete")
        rankings = dict(terminal_rows)
        if len(rankings) != WORK_UNIT_COUNT:
            raise MuSiQueM2RunnerError("M2 terminal keys are not one-to-one")

        stage = "fresh_runtime_postflight_before_scoring"
        postflight_binding = prepared.fresh_reverify()
        if postflight_binding != safe_runtime_binding:
            raise MuSiQueM2RunnerError(
                "M2 runtime drifted between preflight and postflight"
            )

        stage = "offline_support_scoring_after_join"
        measurement = _score_measurement(items=block.items, rankings=rankings)
        ranking_receipts = [
            {
                "ordinal_hash": stable_hash({"ordinal": ordinal}),
                "component_id": component_id,
                "ranking_hash": stable_hash(
                    {"retrieved_indices": list(ranking)}
                ),
            }
            for (ordinal, component_id), ranking in sorted(rankings.items())
        ]
        report_body: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "valid": True,
            "freeze_hash": freeze["freeze_hash"],
            "freeze_file_sha256": freeze_file_hash,
            "measurement_block_id_hash": stable_hash({"block": "M2"}),
            "measurement_block_file_sha256": block.file_sha256,
            "measurement": measurement,
            "execution": {
                "retrieval_component_ids": list(RETRIEVAL_COMPONENT_IDS),
                "zero_call_baseline_id": "canonical_RAW",
                "item_count": len(block.items),
                "retrieval_work_unit_count": WORK_UNIT_COUNT,
                "retrieval_call_count": attempted,
                "retrieval_terminal_count": completed,
                "canonical_raw_derivation_count": len(block.items),
                "configured_maximum_concurrency": MAXIMUM_CONCURRENCY,
                "all_retrieval_terminals_joined_before_support_scoring": True,
                "ranking_receipt_set_hash": stable_hash(ranking_receipts),
                "generator_calls": 0,
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
                "outcome_dependent_arm_changes": 0,
            },
            "runtime": {
                "attestation_receipt_sha256": safe_runtime_binding[
                    "attestation_receipt_sha256"
                ],
                "formal_entry_executable_identity_probe_calls": 0,
                "official_arm_terminal_count": sum(
                    component_id == "official_HippoRAG"
                    for _ordinal, component_id in rankings
                ),
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight_binding[
                    "binding_sha256"
                ],
            },
            "sealed_or_test_content_accessed": False,
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_hash": stable_hash(report_body)}
        _assert_public_safe(report)
        stage = "aggregate_report_persistence"
        _write_json_exclusive(root / REPORT_FILENAME, report)
        persisted = json.loads((root / REPORT_FILENAME).read_text("utf-8"))
        declared = persisted.pop("report_hash", None)
        if declared != stable_hash(persisted):
            raise MuSiQueM2RunnerError("persisted M2 report hash drifted")
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "valid": False,
            "freeze_hash": freeze["freeze_hash"],
            "failure_stage": stage,
            "error_type_hash": stable_hash(
                {"error_type": type(exc).__name__}
            ),
            "authorization_consumed": (
                root / CONSUMPTION_FILENAME
            ).is_file(),
            "retrieval_work_unit_count": WORK_UNIT_COUNT,
            "retrieval_attempt_count": attempted,
            "retrieval_terminal_count": completed,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        failure = {**failure_body, "failure_hash": stable_hash(failure_body)}
        try:
            _assert_public_safe(failure)
            _write_json_exclusive(root / FAILURE_FILENAME, failure)
        except Exception:
            pass
        raise MuSiQueM2RunnerError(
            "formal M2 run failed and cannot be replayed"
        ) from exc


__all__ = [
    "MuSiQueM2RunnerError",
    "build_m2_pre_run_freeze",
    "execute_m2_retention_formal",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Clean module-CLI entry with no callable or result injection surface."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    execute = subparsers.add_parser("execute")
    for command in (freeze, execute):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--acquisition-receipt", type=Path, required=True)
        command.add_argument("--p-formation-receipt", type=Path, required=True)
        command.add_argument("--p-frozen-program", type=Path, required=True)
        command.add_argument("--q-formation-receipt", type=Path, required=True)
        command.add_argument("--q-frozen-program", type=Path, required=True)
        command.add_argument("--m1-pre-run-freeze", type=Path, required=True)
        command.add_argument("--m1-promotion-report", type=Path, required=True)
        command.add_argument("--runtime-python", type=Path, required=True)
        command.add_argument("--local-llm-model", type=Path, required=True)
        command.add_argument("--local-embedding-model", type=Path, required=True)
        command.add_argument("--base-binding-receipt", type=Path, required=True)
        command.add_argument("--attestation-receipt", type=Path, required=True)
        command.add_argument("--execution-root", type=Path, required=True)
    freeze.add_argument("--authorization-hash", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute.add_argument("--pre-run-freeze", type=Path, required=True)
    execute.add_argument("--m2-block", type=Path, required=True)
    arguments = parser.parse_args(argv)
    common = {
        "project_root": arguments.project_root,
        "acquisition_receipt_path": arguments.acquisition_receipt,
        "p_formation_receipt_path": arguments.p_formation_receipt,
        "p_frozen_program_path": arguments.p_frozen_program,
        "q_formation_receipt_path": arguments.q_formation_receipt,
        "q_frozen_program_path": arguments.q_frozen_program,
        "m1_pre_run_freeze_path": arguments.m1_pre_run_freeze,
        "m1_promotion_report_path": arguments.m1_promotion_report,
        "runtime_python": arguments.runtime_python,
        "local_llm_model": arguments.local_llm_model,
        "local_embedding_model": arguments.local_embedding_model,
        "base_binding_receipt_path": arguments.base_binding_receipt,
        "attestation_receipt_path": arguments.attestation_receipt,
        "execution_root": arguments.execution_root,
    }
    if arguments.command == "freeze":
        build_m2_pre_run_freeze(
            **common,
            authorization_hash=arguments.authorization_hash,
            output_path=arguments.output,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_m2_retention_formal(
            **common,
            pre_run_freeze_path=arguments.pre_run_freeze,
            m2_block_path=arguments.m2_block,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

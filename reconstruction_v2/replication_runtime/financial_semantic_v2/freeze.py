from __future__ import annotations

"""Preregister and freeze the project-authored SEC-13F replication."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    FRESH_SPLIT_RELATIVE_PATH,
    V320_PROTOCOL_RELATIVE_PATH,
    _provider_selection_receipt,
)
from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol

from .pack import (
    Sec13FSource,
    derive_selection_seed,
    payload_hash,
    read_json,
    sha256_file,
    verify_measurement_view,
    write_json,
)
from .materialize import MATERIALIZATION_VERSION
from .plan import (
    FixedPeriodOutTreatmentV2,
    MeasurementTargetV2,
    build_measurement_plan_v2,
)
from .prewarm import (
    OFFLINE_VERIFIER_PROFILE_ID,
    OFFLINE_VERIFIER_REQUIREMENTS,
    PREWARM_VERSION,
)
from .treatment import (
    FixedFinancialCandidateIdentityV1,
    build_replication_evaluator_binding_v1,
    load_fixed_financial_candidate_identity_v1,
    validate_replication_evaluator_binding_v1,
)


PREREGISTRATION_VERSION = (
    "financial_semantic_sec13f_period_out_preregistration_v1"
)
ACQUISITION_RECEIPT_VERSION = (
    "financial_semantic_sec13f_period_out_acquisition_v1"
)
EXECUTION_FREEZE_VERSION = (
    "financial_semantic_sec13f_period_out_execution_freeze_v1"
)
STUDY_ID = "financial-semantic-sec13f-period-out-v1"
SELECTION_SEED = (
    "assumption-agent-financial-semantic-period-out-v1-20260716"
)

SOURCE_CLOSURE_VERSION = (
    "financial_semantic_sec13f_runtime_source_closure_v2"
)
STAGE_CHAIN_VERSION = (
    "financial_semantic_sec13f_execution_stage_chain_v1"
)

PREVIOUS_URL = (
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
    "01sep2025-30nov2025_form13f.zip"
)
CURRENT_URL = (
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
    "01dec2025-28feb2026_form13f.zip"
)

_RUNTIME_SOURCE_ROOTS = (
    "assumption_agent",
    "candidates/financial_semantic_operator_v1",
    "replication_runtime",
)

# These immutable inputs are read directly while constructing or validating
# the period-out plan, but live outside the recursive source roots above.
_PREREG_BOUND_PATHS = (
    FRESH_SPLIT_RELATIVE_PATH,
    "artifacts/financial_semantic_fresh_v1_plus_actual01/"
    "fresh_paired.recovered.report.json",
    "manifests/financial_analysis_6_sealed_contamination_receipt_v1.json",
    "manifests/financial_distilbert_qa_runtime_asset_v1.json",
    "manifests/financial_semantic_operator_asset_v1.json",
    "manifests/financial_semantic_treatment_freeze_v1.json",
    "manifests/semantic_assignment_minilm_runtime_asset_v1.json",
    V320_PROTOCOL_RELATIVE_PATH,
)

_TRANSIENT_SOURCE_DIRS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_TRANSIENT_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


class PeriodOutFreezeError(PermissionError):
    """A preregistered or finalized replication binding drifted."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _git_commit(project: Path) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PeriodOutFreezeError("project Git commit is unavailable") from exc
    value = completed.stdout.strip().lower()
    if _GIT_COMMIT.fullmatch(value) is None:
        raise PeriodOutFreezeError("project Git commit is malformed")
    return value


def _git_relative_path(project: Path, relative: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_root = Path(completed.stdout.strip()).resolve(strict=True)
        prefix = project.relative_to(git_root)
    except (OSError, subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        raise PeriodOutFreezeError("project Git worktree is unavailable") from exc
    return (prefix / relative).as_posix()


def _preregistration_commit_binding(
    project: Path,
    prereg_path: Path,
    prereg_relative: str,
) -> str:
    repo_relative = _git_relative_path(project, prereg_relative)
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "status",
                "--porcelain=v1",
                "--",
                prereg_relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        commit_result = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "log",
                "-1",
                "--format=%H",
                "--",
                prereg_relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PeriodOutFreezeError(
            "preregistration Git binding is unavailable"
        ) from exc
    commit = commit_result.stdout.strip().lower()
    if status.stdout.strip() or _GIT_COMMIT.fullmatch(commit) is None:
        raise PeriodOutFreezeError(
            "preregistration must be committed before archive acquisition"
        )
    try:
        blob = subprocess.run(
            ["git", "-C", str(project), "show", f"{commit}:{repo_relative}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PeriodOutFreezeError(
            "committed preregistration blob is unavailable"
        ) from exc
    if hashlib.sha256(blob).hexdigest() != sha256_file(prereg_path):
        raise PeriodOutFreezeError(
            "committed preregistration differs from the local artifact"
        )
    return commit


def _verify_preregistration_commit_binding(
    project: Path,
    prereg_path: Path,
    prereg_relative: str,
    commit: object,
) -> None:
    if _GIT_COMMIT.fullmatch(str(commit or "")) is None:
        raise PeriodOutFreezeError("preregistration commit is malformed")
    repo_relative = _git_relative_path(project, prereg_relative)
    try:
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "show",
                f"{commit}:{repo_relative}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PeriodOutFreezeError(
            "frozen preregistration commit is unavailable"
        ) from exc
    if hashlib.sha256(blob).hexdigest() != sha256_file(prereg_path):
        raise PeriodOutFreezeError(
            "frozen preregistration commit does not contain the bound file"
        )


def _project_path(project: Path, relative: str) -> Path:
    raw = Path(relative)
    if not relative or raw.is_absolute() or ".." in raw.parts:
        raise PeriodOutFreezeError("project-relative path is unsafe")
    path = (project / raw).resolve(strict=True)
    try:
        path.relative_to(project)
    except ValueError as exc:
        raise PeriodOutFreezeError("project-relative path escaped root") from exc
    return path


def _runtime_source_files(project: Path) -> tuple[tuple[str, Path], ...]:
    found: dict[str, Path] = {}
    for relative_root in _RUNTIME_SOURCE_ROOTS:
        root = _project_path(project, relative_root)
        if root.is_symlink() or not root.is_dir():
            raise PeriodOutFreezeError("runtime source root is unavailable")
        for candidate in sorted(
            root.rglob("*"),
            key=lambda value: value.relative_to(project).as_posix(),
        ):
            local_parts = candidate.relative_to(root).parts
            if any(part in _TRANSIENT_SOURCE_DIRS for part in local_parts):
                continue
            if candidate.is_symlink():
                raise PeriodOutFreezeError(
                    "runtime source closure contains a symbolic link"
                )
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise PeriodOutFreezeError(
                    "runtime source closure contains a special file"
                )
            if candidate.suffix in _TRANSIENT_SOURCE_SUFFIXES:
                continue
            relative = candidate.relative_to(project).as_posix()
            found[relative] = candidate
    for relative in _PREREG_BOUND_PATHS:
        path = _project_path(project, relative)
        if path.is_symlink() or not path.is_file():
            raise PeriodOutFreezeError("bound runtime artifact is unavailable")
        found[relative] = path
    if not found:
        raise PeriodOutFreezeError("runtime source closure is empty")
    return tuple(sorted(found.items()))


def _runtime_source_closure(project: Path) -> dict[str, Any]:
    rows = [
        {"relative_path": relative, "file_sha256": sha256_file(path)}
        for relative, path in _runtime_source_files(project)
    ]
    body = {
        "closure_version": SOURCE_CLOSURE_VERSION,
        "scope_policy": "recursive_all_regular_files_plus_bound_artifacts_v2",
        "runtime_scope_paths": list(_RUNTIME_SOURCE_ROOTS),
        "bound_artifact_paths": sorted(_PREREG_BOUND_PATHS),
        "files": rows,
        "file_count": len(rows),
        "file_set_hash": payload_hash(rows),
    }
    return {
        **body,
        "source_commit": _git_commit(project),
        "closure_hash": payload_hash(body),
    }


def _validate_source_closure_shape(closure: Mapping[str, Any]) -> None:
    expected_fields = {
        "closure_version",
        "scope_policy",
        "runtime_scope_paths",
        "bound_artifact_paths",
        "files",
        "file_count",
        "file_set_hash",
        "source_commit",
        "closure_hash",
    }
    if set(closure) != expected_fields:
        raise PeriodOutFreezeError("runtime source closure fields drifted")
    rows = closure.get("files")
    if not isinstance(rows, list) or not rows:
        raise PeriodOutFreezeError("runtime source closure is empty")
    normalized: list[dict[str, str]] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "file_sha256"}
            or not isinstance(row.get("relative_path"), str)
            or not row["relative_path"]
            or Path(str(row["relative_path"])).is_absolute()
            or ".." in Path(str(row["relative_path"])).parts
            or not _is_sha256(row.get("file_sha256"))
        ):
            raise PeriodOutFreezeError("runtime source closure row is malformed")
        normalized.append(
            {
                "relative_path": str(row["relative_path"]),
                "file_sha256": str(row["file_sha256"]),
            }
        )
    body = {
        "closure_version": SOURCE_CLOSURE_VERSION,
        "scope_policy": "recursive_all_regular_files_plus_bound_artifacts_v2",
        "runtime_scope_paths": list(_RUNTIME_SOURCE_ROOTS),
        "bound_artifact_paths": sorted(_PREREG_BOUND_PATHS),
        "files": normalized,
        "file_count": len(normalized),
        "file_set_hash": payload_hash(normalized),
    }
    if (
        rows != sorted(normalized, key=lambda row: row["relative_path"])
        or len({row["relative_path"] for row in normalized}) != len(normalized)
        or any(closure.get(key) != value for key, value in body.items())
        or _GIT_COMMIT.fullmatch(str(closure.get("source_commit") or ""))
        is None
        or closure.get("closure_hash") != payload_hash(body)
    ):
        raise PeriodOutFreezeError("runtime source closure identity drifted")


def _validate_runtime_source_closure(
    project: Path,
    closure: Mapping[str, Any],
) -> None:
    _validate_source_closure_shape(closure)
    current = _runtime_source_closure(project)
    stable_fields = set(closure) - {"source_commit"}
    if any(current.get(key) != closure.get(key) for key in stable_fields):
        raise PeriodOutFreezeError("runtime source closure changed after freeze")
    commit = str(closure["source_commit"])
    scopes = [
        *[str(value) for value in closure["runtime_scope_paths"]],
        *[str(value) for value in closure["bound_artifact_paths"]],
    ]
    try:
        subprocess.run(
            ["git", "-C", str(project), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
        )
        difference = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "diff",
                "--quiet",
                commit,
                "--",
                *scopes,
            ],
            check=False,
            capture_output=True,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *scopes,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PeriodOutFreezeError(
            "runtime source closure could not be checked against Git"
        ) from exc
    if difference.returncode not in {0, 1}:
        raise PeriodOutFreezeError("runtime source comparison failed")
    if difference.returncode != 0:
        raise PeriodOutFreezeError("runtime scope differs from source commit")
    if status.stdout.strip():
        raise PeriodOutFreezeError("runtime scope has tracked or untracked drift")


def _relative_artifact(project: Path, path: str | Path) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        relative = resolved.relative_to(project).as_posix()
    except ValueError as exc:
        raise PeriodOutFreezeError(
            "execution artifact must be within the project"
        ) from exc
    return resolved, relative


def _preregistered_policy_sections() -> dict[str, Any]:
    return {
        "period_data": {
            "source_page": (
                "https://www.sec.gov/data-research/sec-markets-data/"
                "form-13f-data-sets"
            ),
            "source_policy": "official_sec_form_13f_quarterly_flattened_v1",
            "download_authorized_after_preregistration_only": True,
            "pack_period_labels": {
                "previous": "2025 Q3",
                "current": "2025 Q4",
                "labels_are_report_quarters": True,
                "labels_are_archive_calendar_windows": False,
            },
            "expected_snapshot_report_dates": {
                "previous": "2025-09-30",
                "current": "2025-12-31",
            },
            "container_roots": {
                "previous": "/root/2025-q2",
                "current": "/root/2025-q3",
                "aliases_are_legacy_candidate_compatibility_locations": True,
                "aliases_are_report_quarter_labels": False,
                "aliases_are_archive_calendar_window_labels": False,
            },
            "archives": [
                {
                    "role": "previous",
                    "calendar_window": "2025-09-01/2025-11-30",
                    "url": PREVIOUS_URL,
                    "expected_content_length": 85618099,
                    "expected_last_modified": "Wed, 03 Dec 2025",
                },
                {
                    "role": "current",
                    "calendar_window": "2025-12-01/2026-02-28",
                    "url": CURRENT_URL,
                    "expected_content_length": 90264650,
                    "expected_last_modified": "Tue, 03 Mar 2026",
                },
            ],
            "archive_resampling_authorized": False,
            "required_tables": {
                "COVERPAGE.tsv": [
                    "ACCESSION_NUMBER",
                    "FILINGMANAGER_NAME",
                    "REPORTCALENDARORQUARTER",
                    "REPORTTYPE",
                ],
                "INFOTABLE.tsv": [
                    "ACCESSION_NUMBER",
                    "NAMEOFISSUER",
                    "TITLEOFCLASS",
                    "CUSIP",
                    "VALUE",
                ],
            },
        },
        "pack": {
            "selection_seed": SELECTION_SEED,
            "selection_policy": (
                "sha256_ranked_period_out_4fold_x2_plus_4sealed_v1"
            ),
            "measurement_count": 8,
            "measurement_fold_count": 4,
            "measurement_items_per_fold": 2,
            "sealed_count": 4,
            "templates": {
                "measurement": {"four_question": 4, "three_question": 4},
                "sealed": {"four_question": 2, "three_question": 2},
            },
            "independent_oracles": [
                "sec13f_pandas_chunked_v1",
                "sec13f_stdlib_streaming_v1",
            ],
            "oracle_candidate_code_imported": False,
            "cross_oracle_exact_agreement_required": True,
            "sealed_full_pack_and_gold_private": True,
            "measurement_view_redacts_sealed_query_entity_instruction": True,
            "resplit_authorized": False,
        },
        "execution": {
            "model": "gpt-5.4-mini",
            "agent_id": "codex",
            "max_steps": 100,
            "measurement_pairs": 8,
            "raw_calls": 8,
            "candidate_calls": 8,
            "physical_calls": 16,
            "outer_workers": 16,
            "model_inference_slots": 16,
            "all_futures_submitted_before_results_read": True,
            "retries": 0,
            "resampling_authorized": False,
            "mid_batch_provider_switch_authorized": False,
            "provider_policy": (
                "plus_first_pro_only_after_complete_unavailability_receipt"
            ),
            "independent_agent_trajectories": True,
            "future_terminal_auditor": "codex_ordered_terminal_event_audit_v2",
            "model_replay_after_crash_authorized": False,
        },
        "offline_evaluation": {
            "profile_id": "common-pytest-ctrf-py312-v1",
            "policy": "family_profile_readonly_volume_v3",
            "formal_verifier_network": "none",
            "online_judge_calls": 0,
            "online_evaluation_fallback_authorized": False,
            "preparation_downloads_and_image_builds_allowed": True,
        },
        "hipporag": {
            "official": False,
            "status": "not_applicable_nonexecuted",
            "reason": "no_isomorphic_executable_file_task_adapter",
            "execution_count": 0,
            "proxy_substitution_authorized": False,
        },
        "reporting_and_claims": {
            "per_item_gain_harm_tie": True,
            "per_fold_descriptive_results": True,
            "aggregate_raw_candidate_delta": True,
            "performance_thresholds_bound": False,
            "performance_gate": False,
            "promotion_authorized": False,
            "incumbent_update_authorized": False,
            "claim_scope": "project_authored_sec13f_period_out_replication_only",
            "official_skilllearnbench_score": False,
            "sealed_execution_authorized": False,
        },
    }


def build_preregistration_v1(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    candidate = load_fixed_financial_candidate_identity_v1(project)
    parent_treatment_path = project / (
        "manifests/financial_semantic_treatment_freeze_v1.json"
    )
    fresh_report_path = project / (
        "artifacts/financial_semantic_fresh_v1_plus_actual01/"
        "fresh_paired.recovered.report.json"
    )
    contamination_path = project / (
        "manifests/financial_analysis_6_sealed_contamination_receipt_v1.json"
    )
    parent_treatment = read_json(parent_treatment_path)
    fresh_report = read_json(fresh_report_path)
    contamination = read_json(contamination_path)
    if (
        parent_treatment.get("manifest_hash") != candidate.parent_manifest_hash
        or fresh_report.get("report_hash")
        != "e6bc247ec318311429ac937e267f85473137f76532a79ea3e8f24d09d069d389"
        or contamination.get("manifest_hash")
        != "60e84c312c547f64b26b9d8b22be33fc2545d50ac15697d2281cfd80afd76493"
        or contamination.get("future_sealed_or_promotion_use_authorized")
        is not False
    ):
        raise PeriodOutFreezeError("parent evidence boundary drifted")
    closure = _runtime_source_closure(project)
    body: dict[str, Any] = {
        "manifest_version": PREREGISTRATION_VERSION,
        "study_id": STUDY_ID,
        "parent_evidence": {
            "parent_treatment": {
                "relative_path": parent_treatment_path.relative_to(project).as_posix(),
                "file_sha256": sha256_file(parent_treatment_path),
                "manifest_hash": candidate.parent_manifest_hash,
            },
            "first_fresh_measurement": {
                "relative_path": fresh_report_path.relative_to(project).as_posix(),
                "file_sha256": sha256_file(fresh_report_path),
                "report_hash": fresh_report["report_hash"],
                "claim_scope": "single_preregistered_unit_level_gain_only",
            },
            "old_financial_6_contamination": {
                "relative_path": contamination_path.relative_to(project).as_posix(),
                "file_sha256": sha256_file(contamination_path),
                "manifest_hash": contamination["manifest_hash"],
                "future_use_authorized": False,
            },
        },
        "candidate": {
            **candidate.safe_payload(project_root=project),
            "candidate_modified": False,
            "new_period_out_treatment_id_required": True,
        },
        **_preregistered_policy_sections(),
        "source_closure": closure,
        "secret_value_persisted": False,
    }
    return {**body, "manifest_hash": payload_hash(body)}


def validate_preregistration_v1(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> str:
    expected_fields = {
        "manifest_version",
        "study_id",
        "parent_evidence",
        "candidate",
        *set(_preregistered_policy_sections()),
        "source_closure",
        "secret_value_persisted",
        "manifest_hash",
    }
    body = dict(payload)
    declared = body.pop("manifest_hash", None)
    if (
        set(payload) != expected_fields
        or
        payload.get("manifest_version") != PREREGISTRATION_VERSION
        or payload.get("study_id") != STUDY_ID
        or not _is_sha256(declared)
        or declared != payload_hash(body)
        or any(
            payload.get(key) != value
            for key, value in _preregistered_policy_sections().items()
        )
        or payload.get("secret_value_persisted") is not False
    ):
        raise PeriodOutFreezeError("period-out preregistration drifted")

    parent = payload.get("parent_evidence")
    candidate_payload = payload.get("candidate")
    if not isinstance(parent, Mapping) or set(parent) != {
        "parent_treatment",
        "first_fresh_measurement",
        "old_financial_6_contamination",
    }:
        raise PeriodOutFreezeError("parent evidence binding is malformed")
    expected_parent_shapes = {
        "parent_treatment": {
            "relative_path",
            "file_sha256",
            "manifest_hash",
        },
        "first_fresh_measurement": {
            "relative_path",
            "file_sha256",
            "report_hash",
            "claim_scope",
        },
        "old_financial_6_contamination": {
            "relative_path",
            "file_sha256",
            "manifest_hash",
            "future_use_authorized",
        },
    }
    for label, fields in expected_parent_shapes.items():
        row = parent.get(label)
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or not isinstance(row.get("relative_path"), str)
            or Path(str(row["relative_path"])).is_absolute()
            or ".." in Path(str(row["relative_path"])).parts
            or not _is_sha256(row.get("file_sha256"))
        ):
            raise PeriodOutFreezeError("parent evidence row is malformed")
    if (
        parent["parent_treatment"]["relative_path"]
        != "manifests/financial_semantic_treatment_freeze_v1.json"
        or parent["first_fresh_measurement"]["relative_path"]
        != (
            "artifacts/financial_semantic_fresh_v1_plus_actual01/"
            "fresh_paired.recovered.report.json"
        )
        or parent["first_fresh_measurement"]["report_hash"]
        != "e6bc247ec318311429ac937e267f85473137f76532a79ea3e8f24d09d069d389"
        or parent["first_fresh_measurement"]["claim_scope"]
        != "single_preregistered_unit_level_gain_only"
        or parent["old_financial_6_contamination"]["relative_path"]
        != "manifests/financial_analysis_6_sealed_contamination_receipt_v1.json"
        or parent["old_financial_6_contamination"]["manifest_hash"]
        != "60e84c312c547f64b26b9d8b22be33fc2545d50ac15697d2281cfd80afd76493"
        or parent["old_financial_6_contamination"]["future_use_authorized"]
        is not False
    ):
        raise PeriodOutFreezeError("parent evidence boundary drifted")

    candidate_fields = {
        "parent_manifest_hash",
        "candidate_id",
        "candidate_manifest_hash",
        "recipe_id",
        "program_set_hash",
        "parent_treatment_id",
        "external_skill_source_receipt_hash",
        "candidate_skill_source",
        "operator_asset_path",
        "minilm_runtime_asset_path",
        "qa_runtime_asset_path",
        "candidate_recipe_and_source_identity_reused_exactly",
        "candidate_modified",
        "new_period_out_treatment_id_required",
    }
    if not isinstance(candidate_payload, Mapping) or set(candidate_payload) != (
        candidate_fields
    ):
        raise PeriodOutFreezeError("candidate preregistration is malformed")
    candidate_hash_fields = (
        "parent_manifest_hash",
        "candidate_id",
        "candidate_manifest_hash",
        "recipe_id",
        "program_set_hash",
        "parent_treatment_id",
        "external_skill_source_receipt_hash",
    )
    candidate_paths = (
        "candidate_skill_source",
        "operator_asset_path",
        "minilm_runtime_asset_path",
        "qa_runtime_asset_path",
    )
    if (
        not all(_is_sha256(candidate_payload.get(key)) for key in candidate_hash_fields)
        or candidate_payload.get("program_set_hash")
        != payload_hash({"recipe_ids": [candidate_payload.get("recipe_id")]})
        or any(
            not isinstance(candidate_payload.get(key), str)
            or Path(str(candidate_payload[key])).is_absolute()
            or ".." in Path(str(candidate_payload[key])).parts
            for key in candidate_paths
        )
        or candidate_payload.get("candidate_recipe_and_source_identity_reused_exactly")
        is not True
        or candidate_payload.get("candidate_modified") is not False
        or candidate_payload.get("new_period_out_treatment_id_required") is not True
        or parent["parent_treatment"]["manifest_hash"]
        != candidate_payload.get("parent_manifest_hash")
    ):
        raise PeriodOutFreezeError("candidate preregistration drifted")

    closure = payload.get("source_closure")
    if not isinstance(closure, Mapping):
        raise PeriodOutFreezeError("runtime source closure is missing")
    _validate_source_closure_shape(closure)
    if project_root is not None:
        project = Path(project_root).expanduser().resolve(strict=True)
        _validate_runtime_source_closure(
            project,
            closure,
        )
        candidate = load_fixed_financial_candidate_identity_v1(project)
        expected_candidate = {
            **candidate.safe_payload(project_root=project),
            "candidate_modified": False,
            "new_period_out_treatment_id_required": True,
        }
        if candidate_payload != expected_candidate:
            raise PeriodOutFreezeError("live candidate differs from preregistration")
        for row in parent.values():
            evidence_path = _project_path(project, str(row["relative_path"]))
            if sha256_file(evidence_path) != row["file_sha256"]:
                raise PeriodOutFreezeError(
                    "parent evidence file differs from preregistration"
                )
    return str(declared)


def build_acquisition_receipt_v1(
    *,
    project_root: str | Path,
    preregistration_path: str | Path,
    previous_archive: str | Path,
    current_archive: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    prereg_path, prereg_relative = _relative_artifact(
        project, preregistration_path
    )
    preregistration = read_json(prereg_path)
    prereg_hash = validate_preregistration_v1(
        preregistration,
        project_root=project,
    )
    prereg_commit = _preregistration_commit_binding(
        project,
        prereg_path,
        prereg_relative,
    )
    paths = {
        "previous": Path(previous_archive).expanduser().resolve(strict=True),
        "current": Path(current_archive).expanduser().resolve(strict=True),
    }
    registered = {
        row["role"]: row for row in preregistration["period_data"]["archives"]
    }
    prereg_ctime_ns = prereg_path.stat().st_ctime_ns
    archive_ctimes: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for role in ("previous", "current"):
        path = paths[role]
        expected = registered[role]
        if path.is_symlink() or not path.is_file():
            raise PeriodOutFreezeError("acquired archive is not a regular file")
        if path.stat().st_size != expected["expected_content_length"]:
            raise PeriodOutFreezeError("acquired archive length drifted")
        archive_ctimes[role] = path.stat().st_ctime_ns
        if archive_ctimes[role] < prereg_ctime_ns:
            raise PeriodOutFreezeError(
                "SEC archive existed locally before preregistration"
            )
        source = Sec13FSource.open(path)
        rows.append(
            {
                "role": role,
                "source_url": expected["url"],
                "calendar_window": expected["calendar_window"],
                "expected_last_modified": expected["expected_last_modified"],
                "archive_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "coverpage_sha256": source.coverpage_sha256,
                "infotable_sha256": source.infotable_sha256,
                "source_fingerprint": source.source_fingerprint,
                "source_path_persisted": False,
            }
        )
    body = {
        "receipt_version": ACQUISITION_RECEIPT_VERSION,
        "study_id": STUDY_ID,
        "preregistration": {
            "relative_path": prereg_relative,
            "file_sha256": sha256_file(prereg_path),
            "manifest_hash": prereg_hash,
            "committed_at_git_commit": prereg_commit,
        },
        "acquisition_order": {
            "policy": "preregistration_file_precedes_local_archive_inodes_v1",
            "preregistration_file_ctime_ns": prereg_ctime_ns,
            "archive_file_ctime_ns_by_role": archive_ctimes,
            "preregistration_preceded_all_local_archives": True,
        },
        "archives": rows,
        "archive_set_hash": payload_hash(rows),
        "source_roles": ["previous", "current"],
        "resampling_used": False,
        "model_calls": 0,
        "online_judge_calls": 0,
        "secret_value_persisted": False,
    }
    return {**body, "receipt_hash": payload_hash(body)}


def validate_acquisition_receipt_v1(
    payload: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    project_root: str | Path | None = None,
    preregistration_path: str | Path | None = None,
) -> str:
    expected_fields = {
        "receipt_version",
        "study_id",
        "preregistration",
        "acquisition_order",
        "archives",
        "archive_set_hash",
        "source_roles",
        "resampling_used",
        "model_calls",
        "online_judge_calls",
        "secret_value_persisted",
        "receipt_hash",
    }
    body = dict(payload)
    declared = body.pop("receipt_hash", None)
    prereg_hash = validate_preregistration_v1(
        preregistration,
        project_root=project_root,
    )
    binding = payload.get("preregistration")
    order = payload.get("acquisition_order")
    rows = payload.get("archives")
    if (
        set(payload) != expected_fields
        or payload.get("receipt_version") != ACQUISITION_RECEIPT_VERSION
        or payload.get("study_id") != STUDY_ID
        or not _is_sha256(declared)
        or declared != payload_hash(body)
        or not isinstance(binding, Mapping)
        or set(binding)
        != {
            "relative_path",
            "file_sha256",
            "manifest_hash",
            "committed_at_git_commit",
        }
        or not isinstance(binding.get("relative_path"), str)
        or Path(str(binding.get("relative_path") or "")).is_absolute()
        or ".." in Path(str(binding.get("relative_path") or "")).parts
        or not _is_sha256(binding.get("file_sha256"))
        or binding.get("manifest_hash") != prereg_hash
        or _GIT_COMMIT.fullmatch(
            str(binding.get("committed_at_git_commit") or "")
        )
        is None
        or not isinstance(order, Mapping)
        or set(order)
        != {
            "policy",
            "preregistration_file_ctime_ns",
            "archive_file_ctime_ns_by_role",
            "preregistration_preceded_all_local_archives",
        }
        or order.get("policy")
        != "preregistration_file_precedes_local_archive_inodes_v1"
        or order.get("preregistration_preceded_all_local_archives") is not True
        or isinstance(order.get("preregistration_file_ctime_ns"), bool)
        or not isinstance(order.get("preregistration_file_ctime_ns"), int)
        or order["preregistration_file_ctime_ns"] < 0
        or not isinstance(rows, list)
        or len(rows) != 2
        or payload.get("source_roles") != ["previous", "current"]
        or payload.get("resampling_used") is not False
        or payload.get("model_calls") != 0
        or payload.get("online_judge_calls") != 0
        or payload.get("secret_value_persisted") is not False
    ):
        raise PeriodOutFreezeError("acquisition receipt drifted")
    assert isinstance(order, Mapping)
    assert isinstance(rows, list)
    registered = {
        str(row["role"]): row
        for row in preregistration["period_data"]["archives"]
    }
    normalized_rows: list[dict[str, Any]] = []
    expected_row_fields = {
        "role",
        "source_url",
        "calendar_window",
        "expected_last_modified",
        "archive_sha256",
        "size_bytes",
        "coverpage_sha256",
        "infotable_sha256",
        "source_fingerprint",
        "source_path_persisted",
    }
    for expected_role, row in zip(("previous", "current"), rows):
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise PeriodOutFreezeError("acquisition archive row is malformed")
        registered_row = registered[expected_role]
        expected_fingerprint = payload_hash(
            {
                "source_policy": preregistration["period_data"]["source_policy"],
                "coverpage_sha256": row.get("coverpage_sha256"),
                "infotable_sha256": row.get("infotable_sha256"),
            }
        )
        if (
            row.get("role") != expected_role
            or row.get("source_url") != registered_row["url"]
            or row.get("calendar_window") != registered_row["calendar_window"]
            or row.get("expected_last_modified")
            != registered_row["expected_last_modified"]
            or row.get("size_bytes") != registered_row["expected_content_length"]
            or isinstance(row.get("size_bytes"), bool)
            or not all(
                _is_sha256(row.get(key))
                for key in (
                    "archive_sha256",
                    "coverpage_sha256",
                    "infotable_sha256",
                    "source_fingerprint",
                )
            )
            or row.get("source_fingerprint") != expected_fingerprint
            or row.get("source_path_persisted") is not False
        ):
            raise PeriodOutFreezeError("acquisition archive binding drifted")
        normalized_rows.append(dict(row))
    ctimes = order.get("archive_file_ctime_ns_by_role")
    prereg_ctime = order["preregistration_file_ctime_ns"]
    if (
        not isinstance(ctimes, Mapping)
        or set(ctimes) != {"previous", "current"}
        or any(
            isinstance(ctimes.get(role), bool)
            or not isinstance(ctimes.get(role), int)
            or ctimes[role] < prereg_ctime
            for role in ("previous", "current")
        )
        or rows[0]["source_fingerprint"] == rows[1]["source_fingerprint"]
        or payload.get("archive_set_hash") != payload_hash(normalized_rows)
    ):
        raise PeriodOutFreezeError("acquisition ordering or source set drifted")
    if project_root is not None:
        if preregistration_path is None:
            raise PeriodOutFreezeError(
                "live acquisition validation requires preregistration path"
            )
        project = Path(project_root).expanduser().resolve(strict=True)
        prereg_path, prereg_relative = _relative_artifact(
            project, preregistration_path
        )
        if (
            binding["relative_path"] != prereg_relative
            or binding["file_sha256"] != sha256_file(prereg_path)
        ):
            raise PeriodOutFreezeError(
                "preregistration file binding changed after acquisition"
            )
        _verify_preregistration_commit_binding(
            project,
            prereg_path,
            prereg_relative,
            binding["committed_at_git_commit"],
        )
    return str(declared)


def _verify_self_hashed_payload(
    payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or declared != payload_hash(body):
        raise PeriodOutFreezeError(f"{label} self hash mismatch")
    return str(declared)


def _acquisition_rows_by_role(
    acquisition: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = acquisition.get("archives")
    if not isinstance(rows, list):
        raise PeriodOutFreezeError("acquisition source rows are unavailable")
    result = {
        str(row.get("role")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(result) != {"previous", "current"}:
        raise PeriodOutFreezeError("acquisition source roles drifted")
    return result


def _validate_view_stage_binding(
    *,
    preregistration: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    view: Mapping[str, Any],
) -> None:
    period = preregistration["period_data"]
    acquired = _acquisition_rows_by_role(acquisition)
    sources = view.get("sources")
    if not isinstance(sources, Mapping):
        raise PeriodOutFreezeError("measurement view sources are unavailable")
    labels = period["pack_period_labels"]
    for role in ("previous", "current"):
        source = sources.get(role)
        row = acquired[role]
        if (
            not isinstance(source, Mapping)
            or source.get("period_label") != labels[role]
            or source.get("source_policy") != period["source_policy"]
            or any(
                source.get(field) != row[field]
                for field in (
                    "coverpage_sha256",
                    "infotable_sha256",
                    "source_fingerprint",
                )
            )
            or source.get("source_path_persisted") is not False
        ):
            raise PeriodOutFreezeError(
                "measurement view source differs from acquisition"
            )
    expected_roots = {
        role: period["container_roots"][role]
        for role in ("previous", "current")
    }
    if (
        view.get("container_roots") != expected_roots
        or view.get("snapshot_report_dates")
        != period["expected_snapshot_report_dates"]
    ):
        raise PeriodOutFreezeError(
            "measurement view period labels or compatibility roots drifted"
        )
    derived_seed = derive_selection_seed(
        preregistration_seed=str(preregistration["pack"]["selection_seed"]),
        previous_source_fingerprint=str(
            acquired["previous"]["source_fingerprint"]
        ),
        current_source_fingerprint=str(
            acquired["current"]["source_fingerprint"]
        ),
    )
    if view.get("selection_seed_commitment_hash") != payload_hash(
        {"selection_seed": derived_seed}
    ):
        raise PeriodOutFreezeError(
            "measurement view selection seed was not preregistered"
        )


def _validate_materialization_stage(
    payload: Mapping[str, Any],
    *,
    view: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> str:
    declared = _verify_self_hashed_payload(
        payload,
        field="materialization_hash",
        label="materialization",
    )
    expected_fields = {
        "materialization_version",
        "project_authored_extension",
        "official_skilllearnbench_score",
        "private_pack_hash",
        "measurement_view_hash",
        "measurement_gold_hash",
        "previous_archive_sha256",
        "current_archive_sha256",
        "period_source_receipts",
        "period_aliases",
        "item_count",
        "items",
        "item_set_hash",
        "benchmark_tree_hash",
        "sealed_task_count_materialized",
        "sealed_content_accessed_by_measurement_root",
        "sealed_content_persisted",
        "sealed_gold_accessed",
        "model_calls",
        "online_judge_calls",
        "secret_value_persisted",
        "materialization_hash",
    }
    acquired = _acquisition_rows_by_role(acquisition)
    items = payload.get("items")
    view_items = view.get("measurement_items")
    if (
        set(payload) != expected_fields
        or payload.get("materialization_version") != MATERIALIZATION_VERSION
        or payload.get("project_authored_extension") is not True
        or payload.get("official_skilllearnbench_score") is not False
        or payload.get("private_pack_hash") != view.get("private_pack_hash")
        or payload.get("measurement_view_hash")
        != view.get("measurement_view_hash")
        or not _is_sha256(payload.get("measurement_gold_hash"))
        or payload.get("previous_archive_sha256")
        != acquired["previous"]["archive_sha256"]
        or payload.get("current_archive_sha256")
        != acquired["current"]["archive_sha256"]
        or payload.get("period_source_receipts")
        != {
            "previous": {
                "container_alias": "/root/2025-q2",
                "archive_sha256": acquired["previous"]["archive_sha256"],
                "coverpage_sha256": acquired["previous"]["coverpage_sha256"],
                "infotable_sha256": acquired["previous"]["infotable_sha256"],
                "source_fingerprint": acquired["previous"]["source_fingerprint"],
                "source_path_persisted": False,
            },
            "current": {
                "container_alias": "/root/2025-q3",
                "archive_sha256": acquired["current"]["archive_sha256"],
                "coverpage_sha256": acquired["current"]["coverpage_sha256"],
                "infotable_sha256": acquired["current"]["infotable_sha256"],
                "source_fingerprint": acquired["current"]["source_fingerprint"],
                "source_path_persisted": False,
            },
        }
        or payload.get("period_aliases")
        != {
            "previous": "/root/2025-q2",
            "current": "/root/2025-q3",
            "aliases_are_calendar_labels": False,
        }
        or payload.get("item_count") != 8
        or not isinstance(items, list)
        or len(items) != 8
        or not isinstance(view_items, list)
        or len(view_items) != 8
        or payload.get("item_set_hash") != payload_hash(items)
        or not _is_sha256(payload.get("benchmark_tree_hash"))
        or payload.get("sealed_task_count_materialized") != 0
        or payload.get("sealed_content_accessed_by_measurement_root") is not False
        or payload.get("sealed_content_persisted") is not False
        or payload.get("sealed_gold_accessed") is not False
        or payload.get("model_calls") != 0
        or payload.get("online_judge_calls") != 0
        or payload.get("secret_value_persisted") is not False
    ):
        raise PeriodOutFreezeError("measurement materialization drifted")
    expected_by_id = {
        str(item["item_id"]): item
        for item in view_items
        if isinstance(item, Mapping)
    }
    observed_ids: list[str] = []
    item_fields = {
        "item_id",
        "item_id_hash",
        "fold",
        "template",
        "instruction_sha256",
        "task_toml_sha256",
        "environment_tree_hash",
        "tests_tree_hash",
        "expected_output_sha256",
        "answers_hash",
        "raw_content_persisted_in_report",
    }
    for row in items:
        if not isinstance(row, Mapping) or set(row) != item_fields:
            raise PeriodOutFreezeError("materialization item row is malformed")
        item_id = str(row.get("item_id") or "")
        expected = expected_by_id.get(item_id)
        if (
            expected is None
            or row.get("item_id_hash") != payload_hash({"item_id": item_id})
            or row.get("fold") != expected.get("fold")
            or row.get("template") != expected.get("template")
            or row.get("instruction_sha256")
            != expected.get("instruction_sha256")
            or row.get("raw_content_persisted_in_report") is not False
            or not all(
                _is_sha256(row.get(field))
                for field in (
                    "item_id_hash",
                    "instruction_sha256",
                    "task_toml_sha256",
                    "environment_tree_hash",
                    "tests_tree_hash",
                    "expected_output_sha256",
                    "answers_hash",
                )
            )
        ):
            raise PeriodOutFreezeError("materialization item binding drifted")
        observed_ids.append(item_id)
    if observed_ids != [str(item["item_id"]) for item in view_items]:
        raise PeriodOutFreezeError("materialization item order drifted")
    return declared


def _validate_prewarm_stage(
    payload: Mapping[str, Any],
    *,
    view: Mapping[str, Any],
    materialization: Mapping[str, Any],
    prewarm_path: str | Path,
) -> str:
    declared = _verify_self_hashed_payload(
        payload,
        field="prewarm_hash",
        label="prewarm",
    )
    preparation = payload.get("preparation_rows")
    formal = payload.get("formal_cache_rows")
    offline_preparation = payload.get("offline_verifier_preparation")
    expected_fields = {
        "prewarm_version",
        "measurement_view_hash",
        "materialization_hash",
        "benchmark_tree_hash",
        "item_count",
        "preparation_rows",
        "preparation_row_set_hash",
        "formal_cache_rows",
        "formal_cache_row_set_hash",
        "unique_image_id_hash",
        "unique_cache_key_hash",
        "offline_verifier_profile_id",
        "offline_verifier_profile_hash",
        "offline_verifier_requirements",
        "offline_verifier_requirements_hash",
        "offline_verifier_runtime_key",
        "offline_verifier_preparation",
        "formal_execution_cache_only",
        "formal_image_cache_only",
        "formal_offline_verifier_cache_only",
        "preparation_network_allowed",
        "formal_verifier_network",
        "model_calls",
        "online_judge_calls",
        "sealed_task_count",
        "sealed_content_accessed",
        "secret_value_persisted",
        "prewarm_hash",
    }
    expected_ids = [
        str(row["item_id"])
        for row in view.get("measurement_items", [])
        if isinstance(row, Mapping)
    ]
    if (
        set(payload) != expected_fields
        or payload.get("prewarm_version") != PREWARM_VERSION
        or payload.get("measurement_view_hash")
        != view.get("measurement_view_hash")
        or payload.get("materialization_hash")
        != materialization.get("materialization_hash")
        or payload.get("benchmark_tree_hash")
        != materialization.get("benchmark_tree_hash")
        or payload.get("item_count") != 8
        or not isinstance(preparation, list)
        or not isinstance(formal, list)
        or len(preparation) != 8
        or len(formal) != 8
        or payload.get("preparation_row_set_hash") != payload_hash(preparation)
        or payload.get("formal_cache_row_set_hash") != payload_hash(formal)
        or not _is_sha256(payload.get("unique_image_id_hash"))
        or not _is_sha256(payload.get("unique_cache_key_hash"))
        or payload.get("offline_verifier_profile_id")
        != OFFLINE_VERIFIER_PROFILE_ID
        or not _is_sha256(payload.get("offline_verifier_profile_hash"))
        or payload.get("offline_verifier_requirements")
        != list(OFFLINE_VERIFIER_REQUIREMENTS)
        or payload.get("offline_verifier_requirements_hash")
        != payload_hash(list(OFFLINE_VERIFIER_REQUIREMENTS))
        or not isinstance(payload.get("offline_verifier_runtime_key"), str)
        or not payload["offline_verifier_runtime_key"]
        or not isinstance(offline_preparation, Mapping)
        or set(offline_preparation)
        != {
            "relative_path",
            "file_sha256",
            "receipt_hash",
            "network_allowed_only_during_preparation",
            "docker_install_network",
            "probe_passed",
        }
        or offline_preparation.get("relative_path")
        != "offline-verifier.preparation.json"
        or not _is_sha256(offline_preparation.get("file_sha256"))
        or not _is_sha256(offline_preparation.get("receipt_hash"))
        or offline_preparation.get(
            "network_allowed_only_during_preparation"
        )
        is not True
        or offline_preparation.get("docker_install_network") != "none"
        or offline_preparation.get("probe_passed") is not True
        or payload.get("formal_execution_cache_only") is not True
        or payload.get("formal_image_cache_only") is not True
        or payload.get("formal_offline_verifier_cache_only") is not True
        or payload.get("preparation_network_allowed") is not True
        or payload.get("formal_verifier_network") != "none"
        or payload.get("model_calls") != 0
        or payload.get("online_judge_calls") != 0
        or payload.get("sealed_task_count") != 0
        or payload.get("sealed_content_accessed") is not False
        or payload.get("secret_value_persisted") is not False
    ):
        raise PeriodOutFreezeError("measurement prewarm drifted")
    preparation_fields = {
        "item_id",
        "item_id_hash",
        "cache_key",
        "environment_hash",
        "source_environment_hash",
        "image_id",
        "agent_runtime_key",
        "agent_runtime_version",
        "prepared_before_formal_cache_check",
    }
    formal_fields = {
        "item_id",
        "item_id_hash",
        "cache_key",
        "environment_hash",
        "source_environment_hash",
        "image_id",
        "agent_runtime_key",
        "agent_runtime_version",
        "prebuilt_cache_reused",
        "offline_verifier_profile_id",
        "offline_verifier_profile_hash",
        "offline_verifier_runtime_key",
        "offline_verifier_runtime_reused",
        "verifier_runtime_network",
    }
    for rows, formal_rows in ((preparation, False), (formal, True)):
        if [str(row.get("item_id")) for row in rows if isinstance(row, Mapping)] != (
            expected_ids
        ):
            raise PeriodOutFreezeError("prewarm item order drifted")
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or set(row) != (formal_fields if formal_rows else preparation_fields)
                or row.get("item_id_hash")
                != payload_hash({"item_id": row.get("item_id")})
                or not all(
                    _is_sha256(row.get(field))
                    for field in (
                        "cache_key",
                        "environment_hash",
                        "source_environment_hash",
                        "agent_runtime_key",
                    )
                )
                or not isinstance(row.get("image_id"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", row["image_id"])
                is None
                or not isinstance(row.get("agent_runtime_version"), str)
                or not row["agent_runtime_version"]
                or (
                    not formal_rows
                    and row.get("prepared_before_formal_cache_check") is not True
                )
                or (
                    formal_rows
                    and (
                        row.get("prebuilt_cache_reused") is not True
                        or row.get("offline_verifier_profile_id")
                        != OFFLINE_VERIFIER_PROFILE_ID
                        or row.get("offline_verifier_profile_hash")
                        != payload.get("offline_verifier_profile_hash")
                        or row.get("offline_verifier_runtime_key")
                        != payload.get("offline_verifier_runtime_key")
                        or row.get("offline_verifier_runtime_reused") is not True
                        or row.get("verifier_runtime_network") != "none"
                    )
                )
            ):
                raise PeriodOutFreezeError("prewarm item binding drifted")
    image_ids = {str(row["image_id"]) for row in formal}
    cache_keys = {str(row["cache_key"]) for row in formal}
    runtime_keys = {str(row["offline_verifier_runtime_key"]) for row in formal}
    if (
        len(image_ids) != 1
        or len(cache_keys) != 1
        or runtime_keys != {str(payload["offline_verifier_runtime_key"])}
        or payload.get("unique_image_id_hash")
        != payload_hash({"image_id": next(iter(image_ids), "")})
        or payload.get("unique_cache_key_hash")
        != payload_hash({"cache_key": next(iter(cache_keys), "")})
    ):
        raise PeriodOutFreezeError("prewarm shared runtime binding drifted")
    source_path = Path(prewarm_path).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise PeriodOutFreezeError("prewarm report is not a regular file")
    prewarm_file = source_path.resolve(strict=True)
    if read_json(prewarm_file) != dict(payload):
        raise PeriodOutFreezeError("prewarm report payload differs from file")
    receipt_path = prewarm_file.parent / str(
        offline_preparation["relative_path"]
    )
    if (
        receipt_path.parent != prewarm_file.parent
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or sha256_file(receipt_path) != offline_preparation["file_sha256"]
    ):
        raise PeriodOutFreezeError(
            "offline verifier preparation file binding drifted"
        )
    receipt = read_json(receipt_path)
    receipt_hash = _verify_self_hashed_payload(
        receipt,
        field="receipt_hash",
        label="offline verifier preparation receipt",
    )
    if (
        receipt_hash != offline_preparation["receipt_hash"]
        or receipt.get("report_version")
        != "offline_verifier_preparation_receipt_v2"
        or receipt.get("policy") != OFFLINE_VERIFIER_POLICY_VERSION
        or receipt.get("profile_id") != OFFLINE_VERIFIER_PROFILE_ID
        or receipt.get("profile_hash")
        != payload.get("offline_verifier_profile_hash")
        or receipt.get("runtime_key")
        != payload.get("offline_verifier_runtime_key")
        or receipt.get("base_image_id") != next(iter(image_ids))
        or receipt.get("python_version") != "3.12"
        or receipt.get("python_abi") != "cp312"
        or receipt.get("docker_install_network") != "none"
        or receipt.get("probe_passed") is not True
        or receipt.get("raw_content_persisted") is not False
    ):
        raise PeriodOutFreezeError(
            "offline verifier preparation receipt drifted"
        )
    return declared


def _validate_protocol_binding(
    preregistration: Mapping[str, Any],
    protocol: PaperProtocol,
) -> None:
    execution = preregistration["execution"]
    if (
        protocol.payload.get("agent_id") != execution["agent_id"]
        or protocol.payload.get("model") != execution["model"]
        or protocol.payload.get("max_steps") != execution["max_steps"]
        or not _is_sha256(protocol.codex_agent_execution_policy.policy_hash)
    ):
        raise PeriodOutFreezeError(
            "paper protocol differs from preregistered execution"
        )


def _build_stage_chain(
    *,
    preregistration_hash: str,
    acquisition_hash: str,
    measurement_view_hash: str,
    materialization_hash: str,
    prewarm_hash: str,
    provider_receipt_hash: str,
    plan_hash: str,
) -> dict[str, Any]:
    identities = (
        ("preregistration", preregistration_hash),
        ("acquisition", acquisition_hash),
        ("measurement_view", measurement_view_hash),
        ("materialization", materialization_hash),
        ("prewarm", prewarm_hash),
        ("provider_selection", provider_receipt_hash),
        ("execution_plan", plan_hash),
    )
    if not all(_is_sha256(identity) for _, identity in identities):
        raise PeriodOutFreezeError("execution stage identity is malformed")
    rows: list[dict[str, Any]] = []
    parent: str | None = None
    for stage, identity in identities:
        row_body = {
            "stage": stage,
            "identity_hash": identity,
            "parent_identity_hash": parent,
        }
        row = {**row_body, "stage_binding_hash": payload_hash(row_body)}
        rows.append(row)
        parent = identity
    body = {"chain_version": STAGE_CHAIN_VERSION, "stages": rows}
    return {**body, "stage_chain_hash": payload_hash(body)}


def build_execution_freeze_v1(
    *,
    project_root: str | Path,
    preregistration_path: str | Path,
    acquisition_receipt_path: str | Path,
    measurement_view_path: str | Path,
    materialization_report_path: str | Path,
    prewarm_path: str | Path,
    provider_selection_path: str | Path,
    selected_canary_path: str | Path,
    provider_label: str = "plus",
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    candidate = load_fixed_financial_candidate_identity_v1(project)
    prereg_path, prereg_relative = _relative_artifact(project, preregistration_path)
    prereg = read_json(prereg_path)
    prereg_hash = validate_preregistration_v1(prereg, project_root=project)
    acquisition_path, acquisition_relative = _relative_artifact(
        project, acquisition_receipt_path
    )
    acquisition = read_json(acquisition_path)
    acquisition_hash = validate_acquisition_receipt_v1(
        acquisition,
        preregistration=prereg,
        project_root=project,
        preregistration_path=prereg_path,
    )
    view_path, view_relative = _relative_artifact(project, measurement_view_path)
    view = verify_measurement_view(read_json(view_path))
    _validate_view_stage_binding(
        preregistration=prereg,
        acquisition=acquisition,
        view=view,
    )
    materialization_path, materialization_relative = _relative_artifact(
        project, materialization_report_path
    )
    materialization = read_json(materialization_path)
    materialization_hash = _validate_materialization_stage(
        materialization,
        view=view,
        acquisition=acquisition,
    )
    prewarm_file, prewarm_relative = _relative_artifact(project, prewarm_path)
    prewarm = read_json(prewarm_file)
    prewarm_hash = _validate_prewarm_stage(
        prewarm,
        view=view,
        materialization=materialization,
        prewarm_path=prewarm_file,
    )
    selection_path, selection_relative = _relative_artifact(
        project, provider_selection_path
    )
    canary_path, canary_relative = _relative_artifact(
        project, selected_canary_path
    )
    if provider_label != "plus":
        raise PeriodOutFreezeError(
            "this finalized batch selected Plus; Pro needs full failure evidence"
        )
    provider_receipt = _provider_selection_receipt(
        provider_label="plus",
        provider_selection_receipt_path=selection_path,
        selected_canary_report_path=canary_path,
        plus_canary_report_path=canary_path,
        plus_transport_failure_receipt_path=None,
        plus_failure_event_ledger_path=None,
        plus_expected_canary_report_path=None,
    )
    provider_receipt_hash = payload_hash(provider_receipt)
    closure = prereg["source_closure"]
    binding = build_replication_evaluator_binding_v1(
        candidate=candidate,
        preregistration_hash=prereg_hash,
        runtime_source_closure_hash=closure["closure_hash"],
        pack_commitment_hash=str(view["private_pack_hash"]),
    )
    validate_replication_evaluator_binding_v1(
        binding,
        candidate=candidate,
        require_pack_commitment=True,
    )
    treatment = {
        **binding,
        "evaluator_epoch": (
            "financial-semantic-sec13f-periodout-"
            + str(view["private_pack_hash"])[:12]
        ),
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
    }
    protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
    _validate_protocol_binding(prereg, protocol)
    targets = tuple(
        MeasurementTargetV2(
            item_id=str(item["item_id"]),
            fold_id=f"measurement-fold-{int(item['fold'])}",
        )
        for item in view["measurement_items"]
    )
    plan = build_measurement_plan_v2(
        targets=targets,
        manifest_hash=str(view["measurement_view_hash"]),
        evaluator_epoch=treatment["evaluator_epoch"],
        treatment=FixedPeriodOutTreatmentV2(
            recipe_id=candidate.recipe_id,
            program_set_hash=candidate.program_set_hash,
            period_out_treatment_id=binding["period_out_treatment_id"],
            external_skill_source_receipt_hash=(
                candidate.external_skill_source_receipt_hash
            ),
            candidate_skill_source=candidate.candidate_skill_source,
        ),
        agent_id=str(protocol.payload["agent_id"]),
        model=str(protocol.payload["model"]),
        max_steps=int(protocol.payload["max_steps"]),
        codex_agent_execution_policy_hash=(
            protocol.codex_agent_execution_policy.policy_hash
        ),
    )
    stage_chain = _build_stage_chain(
        preregistration_hash=prereg_hash,
        acquisition_hash=acquisition_hash,
        measurement_view_hash=str(view["measurement_view_hash"]),
        materialization_hash=materialization_hash,
        prewarm_hash=prewarm_hash,
        provider_receipt_hash=provider_receipt_hash,
        plan_hash=plan.plan_hash,
    )
    body = {
        "manifest_version": EXECUTION_FREEZE_VERSION,
        "study_id": STUDY_ID,
        "preregistration": {
            "relative_path": prereg_relative,
            "file_sha256": sha256_file(prereg_path),
            "manifest_hash": prereg_hash,
        },
        "acquisition": {
            "relative_path": acquisition_relative,
            "file_sha256": sha256_file(acquisition_path),
            "receipt_hash": acquisition_hash,
            "archive_set_hash": acquisition["archive_set_hash"],
        },
        "measurement_view": {
            "relative_path": view_relative,
            "file_sha256": sha256_file(view_path),
            "measurement_view_hash": view["measurement_view_hash"],
            "private_pack_hash": view["private_pack_hash"],
            "measurement_count": 8,
            "sealed_commitments": view["sealed_item_commitments"],
            "sealed_content_persisted": False,
            "period_binding_hash": payload_hash(
                {
                    "period_labels": prereg["period_data"][
                        "pack_period_labels"
                    ],
                    "snapshot_report_dates": view[
                        "snapshot_report_dates"
                    ],
                    "container_roots": view["container_roots"],
                }
            ),
        },
        "materialization": {
            "relative_path": materialization_relative,
            "file_sha256": sha256_file(materialization_path),
            "materialization_hash": materialization_hash,
            "benchmark_tree_hash": materialization["benchmark_tree_hash"],
            "sealed_task_count_materialized": 0,
        },
        "prewarm": {
            "relative_path": prewarm_relative,
            "file_sha256": sha256_file(prewarm_file),
            "prewarm_hash": prewarm_hash,
            "formal_execution_cache_only": True,
            "formal_verifier_network": "none",
        },
        "provider_selection": {
            "provider_label": provider_label,
            "selection_relative_path": selection_relative,
            "selection_file_sha256": sha256_file(selection_path),
            "selected_canary_relative_path": canary_relative,
            "selected_canary_file_sha256": sha256_file(canary_path),
            "verified_receipt": provider_receipt,
            "verified_receipt_hash": provider_receipt_hash,
            "fixed_for_complete_batch": True,
            "mid_batch_switch_authorized": False,
        },
        "candidate": candidate.safe_payload(project_root=project),
        "treatment": treatment,
        "plan": {"safe_payload": plan.safe_payload(), "plan_hash": plan.plan_hash},
        "stage_chain": stage_chain,
        "execution": {
            "physical_calls": 16,
            "outer_workers": 16,
            "model_inference_slots": 16,
            "all_futures_submitted_before_results_read": True,
            "retries": 0,
            "model_replay_authorized": False,
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "performance_gate_bound": False,
            "promotion_authorized": False,
            "sealed_execution_authorized": False,
        },
        "source_closure": closure,
        "secret_value_persisted": False,
    }
    return {**body, "manifest_hash": payload_hash(body)}


def validate_execution_freeze_v1(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> FixedFinancialCandidateIdentityV1:
    project = Path(project_root).expanduser().resolve(strict=True)
    expected_fields = {
        "manifest_version",
        "study_id",
        "preregistration",
        "acquisition",
        "measurement_view",
        "materialization",
        "prewarm",
        "provider_selection",
        "candidate",
        "treatment",
        "plan",
        "stage_chain",
        "execution",
        "source_closure",
        "secret_value_persisted",
        "manifest_hash",
    }
    expected_execution = {
        "physical_calls": 16,
        "outer_workers": 16,
        "model_inference_slots": 16,
        "all_futures_submitted_before_results_read": True,
        "retries": 0,
        "model_replay_authorized": False,
        "offline_evaluation_only": True,
        "online_judge_calls": 0,
        "performance_gate_bound": False,
        "promotion_authorized": False,
        "sealed_execution_authorized": False,
    }
    body = dict(payload)
    declared = body.pop("manifest_hash", None)
    if (
        set(payload) != expected_fields
        or payload.get("manifest_version") != EXECUTION_FREEZE_VERSION
        or payload.get("study_id") != STUDY_ID
        or not _is_sha256(declared)
        or declared != payload_hash(body)
        or payload.get("execution") != expected_execution
        or payload.get("secret_value_persisted") is not False
    ):
        raise PeriodOutFreezeError("execution freeze drifted")
    closure = payload.get("source_closure")
    if not isinstance(closure, Mapping):
        raise PeriodOutFreezeError("execution source closure is malformed")
    _validate_runtime_source_closure(project, closure)

    section_shapes = {
        "preregistration": {
            "relative_path",
            "file_sha256",
            "manifest_hash",
        },
        "acquisition": {
            "relative_path",
            "file_sha256",
            "receipt_hash",
            "archive_set_hash",
        },
        "measurement_view": {
            "relative_path",
            "file_sha256",
            "measurement_view_hash",
            "private_pack_hash",
            "measurement_count",
            "sealed_commitments",
            "sealed_content_persisted",
            "period_binding_hash",
        },
        "materialization": {
            "relative_path",
            "file_sha256",
            "materialization_hash",
            "benchmark_tree_hash",
            "sealed_task_count_materialized",
        },
        "prewarm": {
            "relative_path",
            "file_sha256",
            "prewarm_hash",
            "formal_execution_cache_only",
            "formal_verifier_network",
        },
    }
    resolved: dict[str, Path] = {}
    for section, fields in section_shapes.items():
        row = payload.get(section)
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or not isinstance(row.get("relative_path"), str)
            or not _is_sha256(row.get("file_sha256"))
        ):
            raise PeriodOutFreezeError(f"{section} binding is malformed")
        file_path = _project_path(project, str(row["relative_path"]))
        if sha256_file(file_path) != row["file_sha256"]:
            raise PeriodOutFreezeError(f"{section} file hash drifted")
        resolved[section] = file_path

    prereg = read_json(resolved["preregistration"])
    prereg_hash = validate_preregistration_v1(prereg, project_root=project)
    prereg_binding = payload["preregistration"]
    if (
        prereg_binding["manifest_hash"] != prereg_hash
        or closure != prereg["source_closure"]
    ):
        raise PeriodOutFreezeError(
            "execution freeze does not reuse the preregistered source closure"
        )
    acquisition = read_json(resolved["acquisition"])
    acquisition_hash = validate_acquisition_receipt_v1(
        acquisition,
        preregistration=prereg,
        project_root=project,
        preregistration_path=resolved["preregistration"],
    )
    acquisition_binding = payload["acquisition"]
    if (
        acquisition_binding["receipt_hash"] != acquisition_hash
        or acquisition_binding["archive_set_hash"]
        != acquisition["archive_set_hash"]
    ):
        raise PeriodOutFreezeError("execution acquisition binding drifted")

    view = verify_measurement_view(read_json(resolved["measurement_view"]))
    _validate_view_stage_binding(
        preregistration=prereg,
        acquisition=acquisition,
        view=view,
    )
    view_binding = payload["measurement_view"]
    expected_period_binding = payload_hash(
        {
            "period_labels": prereg["period_data"]["pack_period_labels"],
            "snapshot_report_dates": view["snapshot_report_dates"],
            "container_roots": view["container_roots"],
        }
    )
    if (
        view_binding["measurement_view_hash"]
        != view["measurement_view_hash"]
        or view_binding["private_pack_hash"] != view["private_pack_hash"]
        or view_binding["measurement_count"] != 8
        or view_binding["sealed_commitments"]
        != view["sealed_item_commitments"]
        or view_binding["sealed_content_persisted"] is not False
        or view_binding["period_binding_hash"] != expected_period_binding
    ):
        raise PeriodOutFreezeError("execution measurement-view binding drifted")

    materialization = read_json(resolved["materialization"])
    materialization_hash = _validate_materialization_stage(
        materialization,
        view=view,
        acquisition=acquisition,
    )
    materialization_binding = payload["materialization"]
    if (
        materialization_binding["materialization_hash"]
        != materialization_hash
        or materialization_binding["benchmark_tree_hash"]
        != materialization["benchmark_tree_hash"]
        or materialization_binding["sealed_task_count_materialized"] != 0
    ):
        raise PeriodOutFreezeError("execution materialization binding drifted")
    prewarm = read_json(resolved["prewarm"])
    prewarm_hash = _validate_prewarm_stage(
        prewarm,
        view=view,
        materialization=materialization,
        prewarm_path=resolved["prewarm"],
    )
    prewarm_binding = payload["prewarm"]
    if (
        prewarm_binding["prewarm_hash"] != prewarm_hash
        or prewarm_binding["formal_execution_cache_only"] is not True
        or prewarm_binding["formal_verifier_network"] != "none"
    ):
        raise PeriodOutFreezeError("execution prewarm binding drifted")

    candidate = load_fixed_financial_candidate_identity_v1(project)
    expected_candidate = candidate.safe_payload(project_root=project)
    if payload.get("candidate") != expected_candidate:
        raise PeriodOutFreezeError("live candidate differs from execution freeze")
    treatment = payload.get("treatment")
    if not isinstance(treatment, Mapping):
        raise PeriodOutFreezeError("execution treatment is malformed")
    expected_binding = build_replication_evaluator_binding_v1(
        candidate=candidate,
        preregistration_hash=prereg_hash,
        runtime_source_closure_hash=str(closure["closure_hash"]),
        pack_commitment_hash=str(view["private_pack_hash"]),
    )
    validate_replication_evaluator_binding_v1(
        expected_binding,
        candidate=candidate,
        require_pack_commitment=True,
    )
    expected_treatment = {
        **expected_binding,
        "evaluator_epoch": (
            "financial-semantic-sec13f-periodout-"
            + str(view["private_pack_hash"])[:12]
        ),
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
    }
    if treatment != expected_treatment:
        raise PeriodOutFreezeError("execution treatment cross-binding drifted")

    provider = payload.get("provider_selection")
    provider_fields = {
        "provider_label",
        "selection_relative_path",
        "selection_file_sha256",
        "selected_canary_relative_path",
        "selected_canary_file_sha256",
        "verified_receipt",
        "verified_receipt_hash",
        "fixed_for_complete_batch",
        "mid_batch_switch_authorized",
    }
    if (
        not isinstance(provider, Mapping)
        or set(provider) != provider_fields
        or provider.get("provider_label") != "plus"
        or provider.get("fixed_for_complete_batch") is not True
        or provider.get("mid_batch_switch_authorized") is not False
        or not _is_sha256(provider.get("selection_file_sha256"))
        or not _is_sha256(provider.get("selected_canary_file_sha256"))
        or not _is_sha256(provider.get("verified_receipt_hash"))
    ):
        raise PeriodOutFreezeError("provider selection drifted")
    selection = _project_path(project, str(provider["selection_relative_path"]))
    canary = _project_path(project, str(provider["selected_canary_relative_path"]))
    verified_provider = _provider_selection_receipt(
        provider_label="plus",
        provider_selection_receipt_path=selection,
        selected_canary_report_path=canary,
        plus_canary_report_path=canary,
        plus_transport_failure_receipt_path=None,
        plus_failure_event_ledger_path=None,
        plus_expected_canary_report_path=None,
    )
    if (
        sha256_file(selection) != provider["selection_file_sha256"]
        or sha256_file(canary) != provider["selected_canary_file_sha256"]
        or verified_provider != provider["verified_receipt"]
        or payload_hash(verified_provider) != provider["verified_receipt_hash"]
    ):
        raise PeriodOutFreezeError("provider evidence changed after freeze")

    protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
    _validate_protocol_binding(prereg, protocol)
    try:
        from .runner import build_plan_from_freeze_v1

        plan = build_plan_from_freeze_v1(
            measurement_view=view,
            execution_freeze=payload,
            candidate=candidate,
            protocol=protocol,
        )
    except Exception as exc:
        raise PeriodOutFreezeError(
            "execution plan differs from final freeze"
        ) from exc
    expected_chain = _build_stage_chain(
        preregistration_hash=prereg_hash,
        acquisition_hash=acquisition_hash,
        measurement_view_hash=str(view["measurement_view_hash"]),
        materialization_hash=materialization_hash,
        prewarm_hash=prewarm_hash,
        provider_receipt_hash=str(provider["verified_receipt_hash"]),
        plan_hash=plan.plan_hash,
    )
    if payload.get("stage_chain") != expected_chain:
        raise PeriodOutFreezeError("execution stage chain drifted")
    return candidate


def load_execution_freeze_v1(
    path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[dict[str, Any], FixedFinancialCandidateIdentityV1]:
    payload = read_json(path)
    candidate = validate_execution_freeze_v1(
        payload,
        project_root=project_root,
    )
    return payload, candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prereg = commands.add_parser("preregister")
    prereg.add_argument("--project-root", type=Path, required=True)
    prereg.add_argument("--output", type=Path, required=True)
    acquisition = commands.add_parser("acquisition")
    acquisition.add_argument("--project-root", type=Path, required=True)
    acquisition.add_argument("--preregistration", type=Path, required=True)
    acquisition.add_argument("--previous-archive", type=Path, required=True)
    acquisition.add_argument("--current-archive", type=Path, required=True)
    acquisition.add_argument("--output", type=Path, required=True)
    execution = commands.add_parser("execution-freeze")
    execution.add_argument("--project-root", type=Path, required=True)
    execution.add_argument("--preregistration", type=Path, required=True)
    execution.add_argument("--acquisition-receipt", type=Path, required=True)
    execution.add_argument("--measurement-view", type=Path, required=True)
    execution.add_argument("--materialization-report", type=Path, required=True)
    execution.add_argument("--prewarm", type=Path, required=True)
    execution.add_argument("--provider-selection", type=Path, required=True)
    execution.add_argument("--selected-canary", type=Path, required=True)
    execution.add_argument("--provider-label", choices=("plus",), default="plus")
    execution.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-execution-freeze")
    verify.add_argument("--project-root", type=Path, required=True)
    verify.add_argument("--execution-freeze", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preregister":
        payload = build_preregistration_v1(args.project_root)
    elif args.command == "acquisition":
        payload = build_acquisition_receipt_v1(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
            previous_archive=args.previous_archive,
            current_archive=args.current_archive,
        )
    elif args.command == "execution-freeze":
        payload = build_execution_freeze_v1(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
            acquisition_receipt_path=args.acquisition_receipt,
            measurement_view_path=args.measurement_view,
            materialization_report_path=args.materialization_report,
            prewarm_path=args.prewarm,
            provider_selection_path=args.provider_selection,
            selected_canary_path=args.selected_canary,
            provider_label=args.provider_label,
        )
    else:
        payload, _candidate = load_execution_freeze_v1(
            args.execution_freeze,
            project_root=args.project_root,
        )
        print(
            json.dumps(
                {
                    "manifest_hash": payload["manifest_hash"],
                    "study_id": payload["study_id"],
                    "verified": True,
                },
                sort_keys=True,
            )
        )
        return 0
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "hash": payload.get("manifest_hash")
                or payload.get("receipt_hash")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

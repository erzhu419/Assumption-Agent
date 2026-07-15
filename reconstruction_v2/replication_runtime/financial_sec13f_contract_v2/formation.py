from __future__ import annotations

"""Fresh 2025-Q4 to 2026-Q1 SEC-13F pack formation.

This module deliberately stops at a private public-pack plus its redacted
measurement view.  It has no command or helper that forms oracle output or
gold.  The previous-period archive is inherited from the already frozen
period-out acquisition, while only the genuinely new current archive is
subject to the preregistration/ctime ordering rule.
"""

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from replication_runtime.financial_semantic_v2 import oracle_pandas
from replication_runtime.financial_semantic_v2 import oracle_streaming
from replication_runtime.financial_semantic_v2.freeze import (
    validate_acquisition_receipt_v1 as validate_prior_acquisition_receipt_v1,
)
from replication_runtime.financial_semantic_v2.pack import (
    Sec13FSource,
    build_measurement_view,
    build_public_pack,
    payload_hash,
    read_json,
    sha256_file,
    verify_measurement_view,
    verify_public_pack,
    write_json,
)


PREREGISTRATION_VERSION = (
    "financial_sec13f_contract_v2_fresh_preregistration_v1"
)
ACQUISITION_RECEIPT_VERSION = (
    "financial_sec13f_contract_v2_fresh_acquisition_v1"
)
PACK_FORMATION_RECEIPT_VERSION = (
    "financial_sec13f_contract_v2_fresh_pack_formation_v1"
)
STUDY_ID = "financial-sec13f-contract-v2-fresh-2025q4-to-2026q1"

CANDIDATE_COMMIT = "7738b348abc06d319f337c9a925dda692e980349"
PRIOR_ACQUISITION_RECEIPT_HASH = (
    "f077083290dbec3f6caeb4ba6368be5294f3b705f302f19feaba01149cc6df1c"
)
PRIOR_ACQUISITION_RELATIVE = (
    "manifests/financial_semantic_sec13f_period_out_acquisition_v1.json"
)
PRIOR_PREREGISTRATION_RELATIVE = (
    "manifests/financial_semantic_sec13f_period_out_preregistration_v1.json"
)
PRIOR_MEASUREMENT_VIEW_RELATIVE = (
    "manifests/financial_semantic_sec13f_period_out_measurement_view_v1.json"
)
PRIOR_MEASUREMENT_VIEW_COMMIT = "c12f79e9bc314ae3d23567030fa364c5b4f95089"
PRIOR_MEASUREMENT_VIEW_FILE_SHA256 = (
    "cf7244579dc42cc8e0f5c85c76f914e8192b1392d92c19d784f5a1c2054cf5d2"
)

CURRENT_URL = (
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
    "01mar2026-31may2026_form13f.zip"
)
CURRENT_CALENDAR_WINDOW = "2026-03-01/2026-05-31"
PREVIOUS_PERIOD_LABEL = "2025 Q4"
CURRENT_PERIOD_LABEL = "2026 Q1"
EXPECTED_PREVIOUS_REPORT_DATE = "2025-12-31"
EXPECTED_CURRENT_REPORT_DATE = "2026-03-31"
# Stable compatibility aliases used by the frozen task Dockerfile.  They are
# logical roles, not calendar labels; the period labels and source hashes bind
# the actual 2025-Q4/2026-Q1 data.
PREVIOUS_CONTAINER_ROOT = "/root/2025-q2"
CURRENT_CONTAINER_ROOT = "/root/2025-q3"
SELECTION_SEED = (
    "assumption-agent-financial-sec13f-contract-v2-fresh-20260716"
)

FROZEN_ORACLE_IDS = (
    oracle_pandas.ORACLE_ID,
    oracle_streaming.ORACLE_ID,
)

_CANDIDATE_PATHS = (
    "assumption_agent/benchmarks/financial_sec13f_contract_operator_v2.py",
    "candidates/financial_sec13f_contract_operator_v2/SKILL.md",
    "manifests/financial_sec13f_contract_operator_v2_consumed_regression_v1.json",
    "manifests/financial_sec13f_public_contract_asset_v2.json",
    "tests/test_financial_sec13f_contract_operator_v2.py",
)
_FORMATION_SOURCE_PATHS = (
    "replication_runtime/financial_sec13f_contract_v2/formation.py",
    "replication_runtime/financial_semantic_v2/pack.py",
    "replication_runtime/financial_semantic_v2/oracle_pandas.py",
    "replication_runtime/financial_semantic_v2/oracle_streaming.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


class FreshFormationError(PermissionError):
    """A frozen fresh-period formation boundary drifted."""


@dataclass(frozen=True)
class FreshPackArtifactsV1:
    """Private pack, safe measurement view, and answer-free receipt."""

    private_pack: dict[str, Any]
    measurement_view: dict[str, Any]
    formation_receipt: dict[str, Any]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _verify_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or declared != payload_hash(body):
        raise FreshFormationError(f"{label} self hash mismatch")
    return str(declared)


def _project_path(project: Path, relative: str) -> Path:
    raw = Path(relative)
    if not relative or raw.is_absolute() or ".." in raw.parts:
        raise FreshFormationError("project-relative path is unsafe")
    path = (project / raw).resolve(strict=True)
    try:
        path.relative_to(project)
    except ValueError as exc:
        raise FreshFormationError("project-relative path escaped root") from exc
    return path


def _git_root(project: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(completed.stdout.strip()).resolve(strict=True)
    except (OSError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise FreshFormationError("project Git worktree is unavailable") from exc


def _repo_relative(project: Path, relative: str) -> str:
    try:
        prefix = project.relative_to(_git_root(project))
    except ValueError as exc:
        raise FreshFormationError("project is outside its Git worktree") from exc
    return (prefix / relative).as_posix()


def _git_blob(project: Path, commit: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "show",
                f"{commit}:{_repo_relative(project, relative)}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshFormationError("frozen Git blob is unavailable") from exc


def _candidate_commit_binding(project: Path) -> dict[str, Any]:
    try:
        resolved = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "rev-parse",
                "--verify",
                f"{CANDIDATE_COMMIT}^{{commit}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshFormationError("frozen candidate commit is unavailable") from exc
    if resolved != CANDIDATE_COMMIT:
        raise FreshFormationError("candidate commit identity drifted")

    rows: list[dict[str, str]] = []
    for relative in _CANDIDATE_PATHS:
        live = _project_path(project, relative)
        if live.is_symlink() or not live.is_file():
            raise FreshFormationError("frozen candidate file is unavailable")
        blob_sha256 = hashlib.sha256(
            _git_blob(project, CANDIDATE_COMMIT, relative)
        ).hexdigest()
        live_sha256 = sha256_file(live)
        if live_sha256 != blob_sha256:
            raise FreshFormationError(
                "live candidate differs from frozen candidate commit"
            )
        rows.append(
            {
                "relative_path": relative,
                "file_sha256": live_sha256,
            }
        )
    asset = read_json(
        _project_path(
            project,
            "manifests/financial_sec13f_public_contract_asset_v2.json",
        )
    )
    if not all(
        _is_sha256(asset.get(field))
        for field in ("manifest_hash", "candidate_id", "operator_source_sha256")
    ):
        raise FreshFormationError("frozen candidate asset is malformed")
    return {
        "commit": CANDIDATE_COMMIT,
        "commit_role": "candidate_frozen_before_fresh_preregistration_v1",
        "files": rows,
        "file_count": len(rows),
        "file_set_hash": payload_hash(rows),
        "candidate_id": asset["candidate_id"],
        "asset_manifest_hash": asset["manifest_hash"],
        "operator_source_sha256": asset["operator_source_sha256"],
    }


def _formation_source_closure(project: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for relative in _FORMATION_SOURCE_PATHS:
        path = _project_path(project, relative)
        if path.is_symlink() or not path.is_file():
            raise FreshFormationError("formation source file is unavailable")
        rows.append(
            {
                "relative_path": relative,
                "file_sha256": sha256_file(path),
            }
        )
    body = {
        "closure_version": "financial_sec13f_formation_source_closure_v1",
        "files": rows,
        "file_count": len(rows),
        "file_set_hash": payload_hash(rows),
    }
    return {**body, "closure_hash": payload_hash(body)}


def _load_prior_evidence(project: Path) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    prior_prereg_path = _project_path(project, PRIOR_PREREGISTRATION_RELATIVE)
    prior_acquisition_path = _project_path(project, PRIOR_ACQUISITION_RELATIVE)
    prior_view_path = _project_path(project, PRIOR_MEASUREMENT_VIEW_RELATIVE)
    prior_prereg = read_json(prior_prereg_path)
    prior_acquisition = read_json(prior_acquisition_path)
    live_view_sha256 = sha256_file(prior_view_path)
    frozen_view_sha256 = hashlib.sha256(
        _git_blob(
            project,
            PRIOR_MEASUREMENT_VIEW_COMMIT,
            PRIOR_MEASUREMENT_VIEW_RELATIVE,
        )
    ).hexdigest()
    if (
        live_view_sha256 != PRIOR_MEASUREMENT_VIEW_FILE_SHA256
        or frozen_view_sha256 != PRIOR_MEASUREMENT_VIEW_FILE_SHA256
    ):
        raise FreshFormationError(
            "prior measurement view differs from its authoritative Git blob"
        )
    prior_view = verify_measurement_view(read_json(prior_view_path))
    validated_hash = validate_prior_acquisition_receipt_v1(
        prior_acquisition,
        preregistration=prior_prereg,
    )
    if validated_hash != PRIOR_ACQUISITION_RECEIPT_HASH:
        raise FreshFormationError("prior acquisition receipt identity drifted")
    rows = prior_acquisition.get("archives")
    if not isinstance(rows, list) or len(rows) != 2:
        raise FreshFormationError("prior acquisition source rows are malformed")
    by_role = {
        str(row.get("role")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_role) != {"previous", "current"}:
        raise FreshFormationError("prior acquisition source roles drifted")
    view_sources = prior_view.get("sources")
    if not isinstance(view_sources, Mapping) or set(view_sources) != {
        "previous",
        "current",
    }:
        raise FreshFormationError("prior measurement source set drifted")
    for role in ("previous", "current"):
        view_row = view_sources.get(role)
        acquisition_row = by_role[role]
        if not isinstance(view_row, Mapping) or any(
            view_row.get(field) != acquisition_row.get(field)
            for field in (
                "coverpage_sha256",
                "infotable_sha256",
                "source_fingerprint",
                "source_path_persisted",
            )
        ):
            raise FreshFormationError(
                "prior measurement view is not bound to prior acquisition"
            )
    return prior_acquisition, dict(by_role["current"]), prior_view


def build_preregistration_v1(project_root: str | Path) -> dict[str, Any]:
    """Bind the frozen candidate and exact new URL before acquisition."""

    project = Path(project_root).expanduser().resolve(strict=True)
    prior_acquisition, inherited_previous, prior_view = _load_prior_evidence(
        project
    )
    prior_acquisition_path = _project_path(
        project, PRIOR_ACQUISITION_RELATIVE
    )
    prior_view_path = _project_path(project, PRIOR_MEASUREMENT_VIEW_RELATIVE)
    body: dict[str, Any] = {
        "manifest_version": PREREGISTRATION_VERSION,
        "study_id": STUDY_ID,
        "candidate_freeze": _candidate_commit_binding(project),
        "formation_source_closure": _formation_source_closure(project),
        "inherited_previous": {
            "prior_acquisition_relative_path": PRIOR_ACQUISITION_RELATIVE,
            "prior_acquisition_file_sha256": sha256_file(
                prior_acquisition_path
            ),
            "prior_acquisition_receipt_hash": prior_acquisition[
                "receipt_hash"
            ],
            "prior_archive_role": "current",
            "archive_receipt": inherited_previous,
            "live_archive_hash_required_at_acquisition": True,
            "archive_ctime_constrained": False,
        },
        "prior_commitment_view": {
            "relative_path": PRIOR_MEASUREMENT_VIEW_RELATIVE,
            "file_sha256": sha256_file(prior_view_path),
            "authoritative_git_commit": PRIOR_MEASUREMENT_VIEW_COMMIT,
            "authoritative_blob_sha256": PRIOR_MEASUREMENT_VIEW_FILE_SHA256,
            "measurement_view_hash": prior_view["measurement_view_hash"],
            "source_fingerprints": {
                role: prior_view["sources"][role]["source_fingerprint"]
                for role in ("previous", "current")
            },
            "measurement_items_readable": True,
            "sealed_commitments_readable": True,
            "sealed_content_persisted": False,
            "prior_private_pack_accessed": False,
            "prior_sealed_content_accessed": False,
        },
        "period_data": {
            "source_policy": (
                "official_sec_form_13f_quarterly_flattened_v1"
            ),
            "previous_period_label": PREVIOUS_PERIOD_LABEL,
            "current_period_label": CURRENT_PERIOD_LABEL,
            "expected_report_dates": {
                "previous": EXPECTED_PREVIOUS_REPORT_DATE,
                "current": EXPECTED_CURRENT_REPORT_DATE,
            },
            "container_roots": {
                "previous": PREVIOUS_CONTAINER_ROOT,
                "current": CURRENT_CONTAINER_ROOT,
            },
            "current_archive": {
                "url": CURRENT_URL,
                "calendar_window": CURRENT_CALENDAR_WINDOW,
                "download_authorized_after_preregistration_only": True,
                "content_length_bound_at_preregistration": False,
                "last_modified_bound_at_preregistration": False,
                "resampling_authorized": False,
            },
            "acquisition_order": {
                "policy": (
                    "current_archive_ctime_not_before_preregistration_v1"
                ),
                "previous_archive_ctime_constrained": False,
                "current_archive_ctime_constrained": True,
            },
        },
        "pack": {
            "selection_seed": SELECTION_SEED,
            "implementation": (
                "replication_runtime.financial_semantic_v2.pack"
            ),
            "measurement_count": 8,
            "measurement_fold_count": 4,
            "measurement_items_per_fold": 2,
            "sealed_count": 4,
            "prior_query_and_instruction_commitments_must_be_disjoint": True,
            "resplit_authorized": False,
        },
        "oracles": {
            "implementation_modules": [
                "replication_runtime.financial_semantic_v2.oracle_pandas",
                "replication_runtime.financial_semantic_v2.oracle_streaming",
            ],
            "oracle_ids": list(FROZEN_ORACLE_IDS),
            "calls_during_pack_formation": 0,
            "gold_formation_authorized_in_this_stage": False,
        },
        "evidence_boundary": {
            "prior_private_pack_accessed": False,
            "prior_sealed_content_accessed": False,
            "gold_formed": False,
            "model_calls": 0,
            "online_judge_calls": 0,
            "secret_value_persisted": False,
        },
    }
    return {**body, "manifest_hash": payload_hash(body)}


def validate_preregistration_v1(
    value: Mapping[str, Any], *, project_root: str | Path | None = None
) -> str:
    expected_fields = {
        "manifest_version",
        "study_id",
        "candidate_freeze",
        "formation_source_closure",
        "inherited_previous",
        "prior_commitment_view",
        "period_data",
        "pack",
        "oracles",
        "evidence_boundary",
        "manifest_hash",
    }
    if set(value) != expected_fields:
        raise FreshFormationError("fresh preregistration fields drifted")
    declared = _verify_self_hash(
        value, field="manifest_hash", label="fresh preregistration"
    )
    candidate = value.get("candidate_freeze")
    formation_closure = value.get("formation_source_closure")
    inherited = value.get("inherited_previous")
    prior_view = value.get("prior_commitment_view")
    period = value.get("period_data")
    pack = value.get("pack")
    oracles = value.get("oracles")
    boundary = value.get("evidence_boundary")
    candidate_fields = {
        "commit",
        "commit_role",
        "files",
        "file_count",
        "file_set_hash",
        "candidate_id",
        "asset_manifest_hash",
        "operator_source_sha256",
    }
    closure_fields = {
        "closure_version",
        "files",
        "file_count",
        "file_set_hash",
        "closure_hash",
    }
    inherited_fields = {
        "prior_acquisition_relative_path",
        "prior_acquisition_file_sha256",
        "prior_acquisition_receipt_hash",
        "prior_archive_role",
        "archive_receipt",
        "live_archive_hash_required_at_acquisition",
        "archive_ctime_constrained",
    }
    prior_view_fields = {
        "relative_path",
        "file_sha256",
        "authoritative_git_commit",
        "authoritative_blob_sha256",
        "measurement_view_hash",
        "source_fingerprints",
        "measurement_items_readable",
        "sealed_commitments_readable",
        "sealed_content_persisted",
        "prior_private_pack_accessed",
        "prior_sealed_content_accessed",
    }
    period_fields = {
        "source_policy",
        "previous_period_label",
        "current_period_label",
        "expected_report_dates",
        "container_roots",
        "current_archive",
        "acquisition_order",
    }
    pack_fields = {
        "selection_seed",
        "implementation",
        "measurement_count",
        "measurement_fold_count",
        "measurement_items_per_fold",
        "sealed_count",
        "prior_query_and_instruction_commitments_must_be_disjoint",
        "resplit_authorized",
    }
    oracle_fields = {
        "implementation_modules",
        "oracle_ids",
        "calls_during_pack_formation",
        "gold_formation_authorized_in_this_stage",
    }
    candidate_rows = candidate.get("files") if isinstance(candidate, Mapping) else None
    closure_rows = (
        formation_closure.get("files")
        if isinstance(formation_closure, Mapping)
        else None
    )
    if (
        value.get("manifest_version") != PREREGISTRATION_VERSION
        or value.get("study_id") != STUDY_ID
        or not isinstance(candidate, Mapping)
        or set(candidate) != candidate_fields
        or candidate.get("commit") != CANDIDATE_COMMIT
        or candidate.get("commit_role")
        != "candidate_frozen_before_fresh_preregistration_v1"
        or not isinstance(candidate_rows, list)
        or len(candidate_rows) != len(_CANDIDATE_PATHS)
        or candidate_rows
        != [
            {
                "relative_path": relative,
                "file_sha256": candidate_rows[index].get("file_sha256"),
            }
            for index, relative in enumerate(_CANDIDATE_PATHS)
            if isinstance(candidate_rows[index], Mapping)
            and set(candidate_rows[index]) == {"relative_path", "file_sha256"}
            and _is_sha256(candidate_rows[index].get("file_sha256"))
        ]
        or candidate.get("file_count") != len(_CANDIDATE_PATHS)
        or candidate.get("file_set_hash") != payload_hash(candidate_rows)
        or not all(
            _is_sha256(candidate.get(field))
            for field in (
                "candidate_id",
                "asset_manifest_hash",
                "operator_source_sha256",
            )
        )
        or not isinstance(formation_closure, Mapping)
        or set(formation_closure) != closure_fields
        or formation_closure.get("closure_version")
        != "financial_sec13f_formation_source_closure_v1"
        or not isinstance(closure_rows, list)
        or len(closure_rows) != len(_FORMATION_SOURCE_PATHS)
        or closure_rows
        != [
            {
                "relative_path": relative,
                "file_sha256": closure_rows[index].get("file_sha256"),
            }
            for index, relative in enumerate(_FORMATION_SOURCE_PATHS)
            if isinstance(closure_rows[index], Mapping)
            and set(closure_rows[index]) == {"relative_path", "file_sha256"}
            and _is_sha256(closure_rows[index].get("file_sha256"))
        ]
        or formation_closure.get("file_count")
        != len(_FORMATION_SOURCE_PATHS)
        or formation_closure.get("file_set_hash") != payload_hash(closure_rows)
        or formation_closure.get("closure_hash")
        != payload_hash(
            {
                "closure_version": formation_closure.get("closure_version"),
                "files": closure_rows,
                "file_count": formation_closure.get("file_count"),
                "file_set_hash": formation_closure.get("file_set_hash"),
            }
        )
        or not isinstance(inherited, Mapping)
        or set(inherited) != inherited_fields
        or inherited.get("prior_acquisition_relative_path")
        != PRIOR_ACQUISITION_RELATIVE
        or inherited.get("prior_acquisition_receipt_hash")
        != PRIOR_ACQUISITION_RECEIPT_HASH
        or inherited.get("prior_archive_role") != "current"
        or inherited.get("live_archive_hash_required_at_acquisition") is not True
        or inherited.get("archive_ctime_constrained") is not False
        or not _is_sha256(inherited.get("prior_acquisition_file_sha256"))
        or not isinstance(prior_view, Mapping)
        or set(prior_view) != prior_view_fields
        or prior_view.get("relative_path") != PRIOR_MEASUREMENT_VIEW_RELATIVE
        or prior_view.get("file_sha256")
        != PRIOR_MEASUREMENT_VIEW_FILE_SHA256
        or prior_view.get("authoritative_git_commit")
        != PRIOR_MEASUREMENT_VIEW_COMMIT
        or prior_view.get("authoritative_blob_sha256")
        != PRIOR_MEASUREMENT_VIEW_FILE_SHA256
        or not _is_sha256(prior_view.get("measurement_view_hash"))
        or not isinstance(prior_view.get("source_fingerprints"), Mapping)
        or set(prior_view["source_fingerprints"])
        != {"previous", "current"}
        or not all(
            _is_sha256(value)
            for value in prior_view["source_fingerprints"].values()
        )
        or prior_view.get("sealed_content_persisted") is not False
        or prior_view.get("measurement_items_readable") is not True
        or prior_view.get("sealed_commitments_readable") is not True
        or prior_view.get("prior_private_pack_accessed") is not False
        or prior_view.get("prior_sealed_content_accessed") is not False
        or not isinstance(period, Mapping)
        or set(period) != period_fields
        or period.get("source_policy")
        != "official_sec_form_13f_quarterly_flattened_v1"
        or period.get("previous_period_label") != PREVIOUS_PERIOD_LABEL
        or period.get("current_period_label") != CURRENT_PERIOD_LABEL
        or period.get("expected_report_dates")
        != {
            "previous": EXPECTED_PREVIOUS_REPORT_DATE,
            "current": EXPECTED_CURRENT_REPORT_DATE,
        }
        or period.get("container_roots")
        != {
            "previous": PREVIOUS_CONTAINER_ROOT,
            "current": CURRENT_CONTAINER_ROOT,
        }
        or period.get("current_archive")
        != {
            "url": CURRENT_URL,
            "calendar_window": CURRENT_CALENDAR_WINDOW,
            "download_authorized_after_preregistration_only": True,
            "content_length_bound_at_preregistration": False,
            "last_modified_bound_at_preregistration": False,
            "resampling_authorized": False,
        }
        or period.get("acquisition_order")
        != {
            "policy": "current_archive_ctime_not_before_preregistration_v1",
            "previous_archive_ctime_constrained": False,
            "current_archive_ctime_constrained": True,
        }
        or not isinstance(pack, Mapping)
        or set(pack) != pack_fields
        or pack.get("selection_seed") != SELECTION_SEED
        or pack.get("implementation")
        != "replication_runtime.financial_semantic_v2.pack"
        or pack.get("measurement_count") != 8
        or pack.get("measurement_fold_count") != 4
        or pack.get("measurement_items_per_fold") != 2
        or pack.get("sealed_count") != 4
        or pack.get("prior_query_and_instruction_commitments_must_be_disjoint")
        is not True
        or pack.get("resplit_authorized") is not False
        or not isinstance(oracles, Mapping)
        or set(oracles) != oracle_fields
        or oracles.get("implementation_modules")
        != [
            "replication_runtime.financial_semantic_v2.oracle_pandas",
            "replication_runtime.financial_semantic_v2.oracle_streaming",
        ]
        or oracles.get("oracle_ids") != list(FROZEN_ORACLE_IDS)
        or oracles.get("calls_during_pack_formation") != 0
        or oracles.get("gold_formation_authorized_in_this_stage") is not False
        or boundary
        != {
            "prior_private_pack_accessed": False,
            "prior_sealed_content_accessed": False,
            "gold_formed": False,
            "model_calls": 0,
            "online_judge_calls": 0,
            "secret_value_persisted": False,
        }
    ):
        raise FreshFormationError("fresh preregistration drifted")
    archive_receipt = inherited.get("archive_receipt")
    if (
        not isinstance(archive_receipt, Mapping)
        or set(archive_receipt)
        != {
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
        or archive_receipt.get("role") != "current"
        or archive_receipt.get("source_path_persisted") is not False
        or isinstance(archive_receipt.get("size_bytes"), bool)
        or not isinstance(archive_receipt.get("size_bytes"), int)
        or archive_receipt["size_bytes"] <= 0
        or not all(
            _is_sha256(archive_receipt.get(field))
            for field in (
                "archive_sha256",
                "coverpage_sha256",
                "infotable_sha256",
                "source_fingerprint",
            )
        )
    ):
        raise FreshFormationError("inherited previous archive is malformed")
    if project_root is not None:
        project = Path(project_root).expanduser().resolve(strict=True)
        expected = build_preregistration_v1(project)
        if dict(value) != expected:
            raise FreshFormationError(
                "fresh preregistration differs from live frozen inputs"
            )
    return declared


def _committed_preregistration(
    project: Path, prereg_path: Path, prereg_relative: str
) -> str:
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
        ).stdout.strip()
        commit = subprocess.run(
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
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshFormationError(
            "preregistration Git binding is unavailable"
        ) from exc
    if status or _GIT_COMMIT.fullmatch(commit) is None:
        raise FreshFormationError(
            "preregistration must be committed before current acquisition"
        )
    if hashlib.sha256(
        _git_blob(project, commit, prereg_relative)
    ).hexdigest() != sha256_file(prereg_path):
        raise FreshFormationError(
            "committed preregistration differs from the live file"
        )
    return commit


def _source_row(
    *, role: str, source: Sec13FSource, archive: Path, source_url: str
) -> dict[str, Any]:
    return {
        "role": role,
        "source_url": source_url,
        "archive_sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
        "coverpage_sha256": source.coverpage_sha256,
        "infotable_sha256": source.infotable_sha256,
        "source_fingerprint": source.source_fingerprint,
        "source_path_persisted": False,
    }


def build_acquisition_receipt_v1(
    *,
    project_root: str | Path,
    preregistration_path: str | Path,
    previous_archive: str | Path,
    current_archive: str | Path,
    current_last_modified: str | None = None,
) -> dict[str, Any]:
    """Bind inherited bytes and the one newly acquired current archive."""

    project = Path(project_root).expanduser().resolve(strict=True)
    prereg_path = Path(preregistration_path).expanduser().resolve(strict=True)
    try:
        prereg_relative = prereg_path.relative_to(project).as_posix()
    except ValueError as exc:
        raise FreshFormationError(
            "preregistration must be inside the project"
        ) from exc
    preregistration = read_json(prereg_path)
    prereg_hash = validate_preregistration_v1(
        preregistration, project_root=project
    )
    prereg_commit = _committed_preregistration(
        project, prereg_path, prereg_relative
    )

    previous_path = Path(previous_archive).expanduser().resolve(strict=True)
    current_path = Path(current_archive).expanduser().resolve(strict=True)
    for path in (previous_path, current_path):
        if path.is_symlink() or not path.is_file():
            raise FreshFormationError("SEC archive is not a regular file")
    prereg_ctime_ns = prereg_path.stat().st_ctime_ns
    current_ctime_ns = current_path.stat().st_ctime_ns
    if current_ctime_ns < prereg_ctime_ns:
        raise FreshFormationError(
            "new current archive existed before preregistration"
        )
    previous_source = Sec13FSource.open(previous_path)
    current_source = Sec13FSource.open(current_path)
    previous_row = _source_row(
        role="previous",
        source=previous_source,
        archive=previous_path,
        source_url=str(
            preregistration["inherited_previous"]["archive_receipt"][
                "source_url"
            ]
        ),
    )
    inherited_row = preregistration["inherited_previous"]["archive_receipt"]
    inherited_projection = {
        key: inherited_row[key]
        for key in (
            "source_url",
            "archive_sha256",
            "size_bytes",
            "coverpage_sha256",
            "infotable_sha256",
            "source_fingerprint",
            "source_path_persisted",
        )
    }
    if {
        key: previous_row[key]
        for key in inherited_projection
    } != inherited_projection:
        raise FreshFormationError(
            "previous archive differs from inherited prior acquisition"
        )
    current_row = _source_row(
        role="current",
        source=current_source,
        archive=current_path,
        source_url=CURRENT_URL,
    )
    current_row["calendar_window"] = CURRENT_CALENDAR_WINDOW
    current_row["observed_last_modified"] = (
        str(current_last_modified).strip()
        if current_last_modified is not None
        else None
    )
    if previous_row["source_fingerprint"] == current_row["source_fingerprint"]:
        raise FreshFormationError("fresh previous and current sources are identical")
    rows = [previous_row, current_row]
    body: dict[str, Any] = {
        "receipt_version": ACQUISITION_RECEIPT_VERSION,
        "study_id": STUDY_ID,
        "preregistration": {
            "relative_path": prereg_relative,
            "file_sha256": sha256_file(prereg_path),
            "manifest_hash": prereg_hash,
            "committed_at_git_commit": prereg_commit,
        },
        "acquisition_order": {
            "policy": "current_archive_ctime_not_before_preregistration_v1",
            "preregistration_file_ctime_ns": prereg_ctime_ns,
            "current_archive_file_ctime_ns": current_ctime_ns,
            "current_archive_not_older_than_preregistration": True,
            "previous_archive_ctime_observed": False,
            "previous_archive_ctime_constrained": False,
        },
        "archives": rows,
        "archive_set_hash": payload_hash(rows),
        "previous_inherited_from_receipt_hash": (
            PRIOR_ACQUISITION_RECEIPT_HASH
        ),
        "resampling_used": False,
        "model_calls": 0,
        "online_judge_calls": 0,
        "secret_value_persisted": False,
    }
    return {**body, "receipt_hash": payload_hash(body)}


def _validate_source_row(row: object, *, role: str) -> Mapping[str, Any]:
    if not isinstance(row, Mapping) or row.get("role") != role:
        raise FreshFormationError("fresh acquisition source row is malformed")
    required = {
        "role",
        "source_url",
        "archive_sha256",
        "size_bytes",
        "coverpage_sha256",
        "infotable_sha256",
        "source_fingerprint",
        "source_path_persisted",
    }
    if role == "current":
        required |= {"calendar_window", "observed_last_modified"}
    if (
        set(row) != required
        or isinstance(row.get("size_bytes"), bool)
        or not isinstance(row.get("size_bytes"), int)
        or row["size_bytes"] <= 0
        or not all(
            _is_sha256(row.get(field))
            for field in (
                "archive_sha256",
                "coverpage_sha256",
                "infotable_sha256",
                "source_fingerprint",
            )
        )
        or row.get("source_path_persisted") is not False
        or row.get("source_fingerprint")
        != payload_hash(
            {
                "source_policy": (
                    "official_sec_form_13f_quarterly_flattened_v1"
                ),
                "coverpage_sha256": row.get("coverpage_sha256"),
                "infotable_sha256": row.get("infotable_sha256"),
            }
        )
        or (
            role == "current"
            and row.get("observed_last_modified") is not None
            and (
                not isinstance(row.get("observed_last_modified"), str)
                or not str(row["observed_last_modified"]).strip()
            )
        )
    ):
        raise FreshFormationError("fresh acquisition source row drifted")
    return row


def validate_acquisition_receipt_v1(
    value: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    project_root: str | Path | None = None,
    preregistration_path: str | Path | None = None,
    previous_archive: str | Path | None = None,
    current_archive: str | Path | None = None,
) -> str:
    expected_fields = {
        "receipt_version",
        "study_id",
        "preregistration",
        "acquisition_order",
        "archives",
        "archive_set_hash",
        "previous_inherited_from_receipt_hash",
        "resampling_used",
        "model_calls",
        "online_judge_calls",
        "secret_value_persisted",
        "receipt_hash",
    }
    if set(value) != expected_fields:
        raise FreshFormationError("fresh acquisition fields drifted")
    declared = _verify_self_hash(
        value, field="receipt_hash", label="fresh acquisition"
    )
    prereg_hash = validate_preregistration_v1(
        preregistration, project_root=project_root
    )
    binding = value.get("preregistration")
    order = value.get("acquisition_order")
    rows = value.get("archives")
    if not isinstance(rows, list) or len(rows) != 2:
        raise FreshFormationError("fresh acquisition sources are malformed")
    previous_row = _validate_source_row(rows[0], role="previous")
    current_row = _validate_source_row(rows[1], role="current")
    inherited = preregistration["inherited_previous"]["archive_receipt"]
    inherited_projection = {
        key: inherited[key]
        for key in (
            "source_url",
            "archive_sha256",
            "size_bytes",
            "coverpage_sha256",
            "infotable_sha256",
            "source_fingerprint",
            "source_path_persisted",
        )
    }
    if (
        value.get("receipt_version") != ACQUISITION_RECEIPT_VERSION
        or value.get("study_id") != STUDY_ID
        or not isinstance(binding, Mapping)
        or set(binding)
        != {
            "relative_path",
            "file_sha256",
            "manifest_hash",
            "committed_at_git_commit",
        }
        or not isinstance(binding.get("relative_path"), str)
        or not binding.get("relative_path")
        or Path(str(binding["relative_path"])).is_absolute()
        or ".." in Path(str(binding["relative_path"])).parts
        or binding.get("manifest_hash") != prereg_hash
        or not _is_sha256(binding.get("file_sha256"))
        or _GIT_COMMIT.fullmatch(
            str(binding.get("committed_at_git_commit") or "")
        )
        is None
        or not isinstance(order, Mapping)
        or set(order)
        != {
            "policy",
            "preregistration_file_ctime_ns",
            "current_archive_file_ctime_ns",
            "current_archive_not_older_than_preregistration",
            "previous_archive_ctime_observed",
            "previous_archive_ctime_constrained",
        }
        or order.get("policy")
        != "current_archive_ctime_not_before_preregistration_v1"
        or order.get("current_archive_not_older_than_preregistration") is not True
        or order.get("previous_archive_ctime_observed") is not False
        or order.get("previous_archive_ctime_constrained") is not False
        or isinstance(order.get("preregistration_file_ctime_ns"), bool)
        or not isinstance(order.get("preregistration_file_ctime_ns"), int)
        or order["preregistration_file_ctime_ns"] < 0
        or isinstance(order.get("current_archive_file_ctime_ns"), bool)
        or not isinstance(order.get("current_archive_file_ctime_ns"), int)
        or order["current_archive_file_ctime_ns"]
        < order["preregistration_file_ctime_ns"]
        or {
            key: previous_row[key]
            for key in inherited_projection
        }
        != inherited_projection
        or current_row.get("source_url") != CURRENT_URL
        or current_row.get("calendar_window") != CURRENT_CALENDAR_WINDOW
        or previous_row.get("source_fingerprint")
        == current_row.get("source_fingerprint")
        or value.get("archive_set_hash") != payload_hash(rows)
        or value.get("previous_inherited_from_receipt_hash")
        != PRIOR_ACQUISITION_RECEIPT_HASH
        or value.get("resampling_used") is not False
        or value.get("model_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("secret_value_persisted") is not False
    ):
        raise FreshFormationError("fresh acquisition receipt drifted")
    if project_root is not None:
        if preregistration_path is None:
            raise FreshFormationError(
                "live acquisition validation requires preregistration path"
            )
        project = Path(project_root).expanduser().resolve(strict=True)
        prereg_path = Path(preregistration_path).expanduser().resolve(
            strict=True
        )
        try:
            relative = prereg_path.relative_to(project).as_posix()
        except ValueError as exc:
            raise FreshFormationError(
                "preregistration must be inside the project"
            ) from exc
        if (
            binding.get("relative_path") != relative
            or binding.get("file_sha256") != sha256_file(prereg_path)
            or _committed_preregistration(project, prereg_path, relative)
            != binding.get("committed_at_git_commit")
            or prereg_path.stat().st_ctime_ns
            != order["preregistration_file_ctime_ns"]
        ):
            raise FreshFormationError(
                "live preregistration binding differs from acquisition"
            )
    live_pairs = (
        ("previous", previous_archive, previous_row),
        ("current", current_archive, current_row),
    )
    if any(path is not None for _, path, _ in live_pairs):
        if not all(path is not None for _, path, _ in live_pairs):
            raise FreshFormationError(
                "live acquisition validation requires both archives"
            )
        for role, path, row in live_pairs:
            assert path is not None
            archive = Path(path).expanduser().resolve(strict=True)
            source = Sec13FSource.open(archive)
            if (
                sha256_file(archive) != row["archive_sha256"]
                or archive.stat().st_size != row["size_bytes"]
                or source.coverpage_sha256 != row["coverpage_sha256"]
                or source.infotable_sha256 != row["infotable_sha256"]
                or source.source_fingerprint != row["source_fingerprint"]
            ):
                raise FreshFormationError(
                    f"live {role} archive differs from acquisition"
                )
        assert current_archive is not None
        live_current_ctime_ns = Path(current_archive).expanduser().resolve(
            strict=True
        ).stat().st_ctime_ns
        if (
            live_current_ctime_ns != order["current_archive_file_ctime_ns"]
            or live_current_ctime_ns < order["preregistration_file_ctime_ns"]
        ):
            raise FreshFormationError(
                "live current archive violates preregistration order"
            )
        # Deliberately no previous-archive ctime read or comparison here.
    return declared


def _commitments_from_prior_view(
    prior_view: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    verified = verify_measurement_view(prior_view)
    instruction_hashes: list[str] = []
    query_hashes: list[str] = []
    for item in verified["measurement_items"]:
        instruction_hashes.append(str(item["instruction_sha256"]))
        query_hashes.append(payload_hash(item["query"]))
    for row in verified["sealed_item_commitments"]:
        instruction_hashes.append(str(row["instruction_sha256"]))
        query_hashes.append(str(row["query_commitment_hash"]))
    if (
        len(instruction_hashes) != 12
        or len(query_hashes) != 12
        or len(set(instruction_hashes)) != len(instruction_hashes)
        or len(set(query_hashes)) != len(query_hashes)
    ):
        raise FreshFormationError("prior commitment set is ambiguous")
    return set(instruction_hashes), set(query_hashes)


def assert_no_prior_commitment_collision_v1(
    *, new_pack: Mapping[str, Any], prior_measurement_view: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on any old/new query or instruction commitment overlap."""

    verified_pack = verify_public_pack(new_pack)
    prior_instructions, prior_queries = _commitments_from_prior_view(
        prior_measurement_view
    )
    new_instruction_rows = [
        str(item["instruction_sha256"]) for item in verified_pack["items"]
    ]
    new_query_rows = [
        payload_hash(item["query"]) for item in verified_pack["items"]
    ]
    if (
        len(set(new_instruction_rows)) != len(new_instruction_rows)
        or len(set(new_query_rows)) != len(new_query_rows)
    ):
        raise FreshFormationError("new pack contains duplicate commitments")
    instruction_collisions = set(new_instruction_rows) & prior_instructions
    query_collisions = set(new_query_rows) & prior_queries
    if instruction_collisions or query_collisions:
        raise FreshFormationError(
            "fresh pack collides with a prior query or instruction commitment"
        )
    audit_body = {
        "policy": "old_new_query_instruction_commitments_disjoint_v1",
        "prior_instruction_commitment_count": len(prior_instructions),
        "prior_query_commitment_count": len(prior_queries),
        "new_instruction_commitment_count": len(new_instruction_rows),
        "new_query_commitment_count": len(new_query_rows),
        "instruction_collision_count": 0,
        "query_collision_count": 0,
        "prior_private_pack_accessed": False,
        "prior_sealed_content_accessed": False,
    }
    return {**audit_body, "audit_hash": payload_hash(audit_body)}


def build_collision_checked_pack_v1(
    *,
    previous_source: str | Path | Sec13FSource,
    current_source: str | Path | Sec13FSource,
    prior_measurement_view: Mapping[str, Any],
    preregistration_seed: str = SELECTION_SEED,
    previous_period_label: str = PREVIOUS_PERIOD_LABEL,
    current_period_label: str = CURRENT_PERIOD_LABEL,
    previous_container_root: str = PREVIOUS_CONTAINER_ROOT,
    current_container_root: str = CURRENT_CONTAINER_ROOT,
    expected_previous_report_date: str = EXPECTED_PREVIOUS_REPORT_DATE,
    expected_current_report_date: str = EXPECTED_CURRENT_REPORT_DATE,
) -> FreshPackArtifactsV1:
    """Reuse the frozen v2 pack builder without evaluating either partition."""

    pack = build_public_pack(
        previous_source=previous_source,
        current_source=current_source,
        previous_period_label=previous_period_label,
        current_period_label=current_period_label,
        preregistration_seed=preregistration_seed,
        previous_container_root=previous_container_root,
        current_container_root=current_container_root,
    )
    if pack["snapshot_report_dates"] != {
        "previous": expected_previous_report_date,
        "current": expected_current_report_date,
    }:
        raise FreshFormationError("fresh pack report dates drifted")
    collision_audit = assert_no_prior_commitment_collision_v1(
        new_pack=pack,
        prior_measurement_view=prior_measurement_view,
    )
    measurement_view = build_measurement_view(pack)
    receipt_body: dict[str, Any] = {
        "receipt_version": PACK_FORMATION_RECEIPT_VERSION,
        "study_id": STUDY_ID,
        "private_pack_hash": pack["pack_hash"],
        "measurement_view_hash": measurement_view["measurement_view_hash"],
        "prior_measurement_view_hash": prior_measurement_view[
            "measurement_view_hash"
        ],
        "preregistration_hash": None,
        "acquisition_receipt_hash": None,
        "input_binding_complete": False,
        "collision_audit": collision_audit,
        "frozen_pack_implementation": (
            "replication_runtime.financial_semantic_v2.pack"
        ),
        "frozen_oracle_ids_reserved_for_later_measurement_gold": list(
            FROZEN_ORACLE_IDS
        ),
        "oracle_calls": 0,
        "gold_formed": False,
        "prior_private_pack_accessed": False,
        "prior_sealed_content_accessed": False,
        "model_calls": 0,
        "network_calls": 0,
        "online_judge_calls": 0,
        "secret_value_persisted": False,
    }
    receipt = {
        **receipt_body,
        "receipt_hash": payload_hash(receipt_body),
    }
    return FreshPackArtifactsV1(
        private_pack=pack,
        measurement_view=measurement_view,
        formation_receipt=receipt,
    )


def validate_pack_formation_receipt_v1(
    value: Mapping[str, Any],
    *,
    private_pack: Mapping[str, Any],
    measurement_view: Mapping[str, Any],
    prior_measurement_view: Mapping[str, Any],
    preregistration_hash: str,
    acquisition_receipt_hash: str,
) -> str:
    """Validate the exact, fully input-bound production receipt schema."""

    expected_fields = {
        "receipt_version",
        "study_id",
        "private_pack_hash",
        "measurement_view_hash",
        "prior_measurement_view_hash",
        "preregistration_hash",
        "acquisition_receipt_hash",
        "input_binding_complete",
        "collision_audit",
        "frozen_pack_implementation",
        "frozen_oracle_ids_reserved_for_later_measurement_gold",
        "oracle_calls",
        "gold_formed",
        "prior_private_pack_accessed",
        "prior_sealed_content_accessed",
        "model_calls",
        "network_calls",
        "online_judge_calls",
        "secret_value_persisted",
        "receipt_hash",
    }
    if set(value) != expected_fields:
        raise FreshFormationError("production formation receipt fields drifted")
    declared = _verify_self_hash(
        value, field="receipt_hash", label="production formation receipt"
    )
    verified_pack = verify_public_pack(private_pack)
    verified_view = verify_measurement_view(
        measurement_view, private_pack=verified_pack
    )
    verified_prior_view = verify_measurement_view(prior_measurement_view)
    collision = value.get("collision_audit")
    collision_fields = {
        "policy",
        "prior_instruction_commitment_count",
        "prior_query_commitment_count",
        "new_instruction_commitment_count",
        "new_query_commitment_count",
        "instruction_collision_count",
        "query_collision_count",
        "prior_private_pack_accessed",
        "prior_sealed_content_accessed",
        "audit_hash",
    }
    if not isinstance(collision, Mapping) or set(collision) != collision_fields:
        raise FreshFormationError("formation collision audit fields drifted")
    collision_body = dict(collision)
    collision_hash = collision_body.pop("audit_hash", None)
    expected_new_count = len(verified_pack["items"])
    if (
        not _is_sha256(preregistration_hash)
        or not _is_sha256(acquisition_receipt_hash)
        or collision_hash != payload_hash(collision_body)
        or collision.get("policy")
        != "old_new_query_instruction_commitments_disjoint_v1"
        or collision.get("prior_instruction_commitment_count") != 12
        or collision.get("prior_query_commitment_count") != 12
        or collision.get("new_instruction_commitment_count")
        != expected_new_count
        or collision.get("new_query_commitment_count") != expected_new_count
        or collision.get("instruction_collision_count") != 0
        or collision.get("query_collision_count") != 0
        or collision.get("prior_private_pack_accessed") is not False
        or collision.get("prior_sealed_content_accessed") is not False
        or value.get("receipt_version") != PACK_FORMATION_RECEIPT_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("private_pack_hash") != verified_pack["pack_hash"]
        or value.get("measurement_view_hash")
        != verified_view["measurement_view_hash"]
        or value.get("prior_measurement_view_hash")
        != verified_prior_view["measurement_view_hash"]
        or value.get("preregistration_hash") != preregistration_hash
        or value.get("acquisition_receipt_hash")
        != acquisition_receipt_hash
        or value.get("input_binding_complete") is not True
        or value.get("frozen_pack_implementation")
        != "replication_runtime.financial_semantic_v2.pack"
        or value.get("frozen_oracle_ids_reserved_for_later_measurement_gold")
        != list(FROZEN_ORACLE_IDS)
        or value.get("oracle_calls") != 0
        or value.get("gold_formed") is not False
        or value.get("prior_private_pack_accessed") is not False
        or value.get("prior_sealed_content_accessed") is not False
        or value.get("model_calls") != 0
        or value.get("network_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("secret_value_persisted") is not False
    ):
        raise FreshFormationError("production formation receipt drifted")
    return declared


def _archive_file_identity(path: str | Path) -> dict[str, Any]:
    archive = Path(path).expanduser().resolve(strict=True)
    if archive.is_symlink() or not archive.is_file():
        raise FreshFormationError("SEC archive is not a regular file")
    return {
        "archive_sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
    }


def form_fresh_pack_v1(
    *,
    project_root: str | Path,
    preregistration_path: str | Path,
    prior_measurement_view_path: str | Path,
    preregistration: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    previous_archive: str | Path,
    current_archive: str | Path,
    prior_measurement_view: Mapping[str, Any],
) -> FreshPackArtifactsV1:
    """Form the production pack after validating all frozen local inputs."""

    project = Path(project_root).expanduser().resolve(strict=True)
    prereg_path = Path(preregistration_path).expanduser().resolve(strict=True)
    if read_json(prereg_path) != dict(preregistration):
        raise FreshFormationError(
            "live preregistration content differs from formation input"
        )
    prereg_hash = validate_preregistration_v1(
        preregistration, project_root=project
    )
    acquisition_hash = validate_acquisition_receipt_v1(
        acquisition,
        preregistration=preregistration,
        project_root=project,
        preregistration_path=prereg_path,
        previous_archive=previous_archive,
        current_archive=current_archive,
    )
    prior_view_path = Path(prior_measurement_view_path).expanduser().resolve(
        strict=True
    )
    authoritative_prior_view_path = _project_path(
        project, PRIOR_MEASUREMENT_VIEW_RELATIVE
    )
    if (
        prior_view_path != authoritative_prior_view_path
        or sha256_file(prior_view_path)
        != PRIOR_MEASUREMENT_VIEW_FILE_SHA256
        or read_json(prior_view_path) != dict(prior_measurement_view)
    ):
        raise FreshFormationError(
            "prior commitment view is not the authoritative Git-bound file"
        )
    verified_prior_view = verify_measurement_view(prior_measurement_view)
    prior_binding = preregistration["prior_commitment_view"]
    if (
        verified_prior_view["measurement_view_hash"]
        != prior_binding["measurement_view_hash"]
        or sha256_file(prior_view_path) != prior_binding["file_sha256"]
        or {
            role: verified_prior_view["sources"][role]["source_fingerprint"]
            for role in ("previous", "current")
        }
        != prior_binding["source_fingerprints"]
    ):
        raise FreshFormationError("prior commitment view binding drifted")
    rows = acquisition["archives"]
    archive_paths = (previous_archive, current_archive)
    before_identities = [
        _archive_file_identity(path) for path in archive_paths
    ]
    for identity, row in zip(before_identities, rows):
        if identity != {
            "archive_sha256": row["archive_sha256"],
            "size_bytes": row["size_bytes"],
        }:
            raise FreshFormationError(
                "archive identity drifted before pack formation"
            )
    artifacts = build_collision_checked_pack_v1(
        previous_source=previous_archive,
        current_source=current_archive,
        prior_measurement_view=verified_prior_view,
    )
    after_identities = [
        _archive_file_identity(path) for path in archive_paths
    ]
    if after_identities != before_identities:
        raise FreshFormationError(
            "archive identity changed during pack formation"
        )
    if (
        validate_acquisition_receipt_v1(
            acquisition,
            preregistration=preregistration,
            project_root=project,
            preregistration_path=prereg_path,
            previous_archive=previous_archive,
            current_archive=current_archive,
        )
        != acquisition_hash
    ):
        raise FreshFormationError(
            "acquisition identity changed during pack formation"
        )
    for role, row in zip(("previous", "current"), rows):
        source = artifacts.private_pack["sources"][role]
        if any(
            source[field] != row[field]
            for field in (
                "coverpage_sha256",
                "infotable_sha256",
                "source_fingerprint",
                "source_path_persisted",
            )
        ):
            raise FreshFormationError(
                "formed pack source differs from acquisition"
            )
    receipt_body = dict(artifacts.formation_receipt)
    receipt_body.pop("receipt_hash")
    receipt_body["preregistration_hash"] = prereg_hash
    receipt_body["acquisition_receipt_hash"] = acquisition_hash
    receipt_body["input_binding_complete"] = True
    receipt = {
        **receipt_body,
        "receipt_hash": payload_hash(receipt_body),
    }
    validate_pack_formation_receipt_v1(
        receipt,
        private_pack=artifacts.private_pack,
        measurement_view=artifacts.measurement_view,
        prior_measurement_view=verified_prior_view,
        preregistration_hash=prereg_hash,
        acquisition_receipt_hash=acquisition_hash,
    )
    return FreshPackArtifactsV1(
        private_pack=artifacts.private_pack,
        measurement_view=artifacts.measurement_view,
        formation_receipt=receipt,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preregister/acquire/form the fresh SEC-13F contract pack."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preregister = commands.add_parser("preregister")
    preregister.add_argument("--project-root", type=Path, required=True)
    preregister.add_argument("--output", type=Path, required=True)

    acquire = commands.add_parser("acquire")
    acquire.add_argument("--project-root", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--previous", type=Path, required=True)
    acquire.add_argument("--current", type=Path, required=True)
    acquire.add_argument("--current-last-modified")
    acquire.add_argument("--output", type=Path, required=True)

    form = commands.add_parser("form-pack")
    form.add_argument("--project-root", type=Path, required=True)
    form.add_argument("--preregistration", type=Path, required=True)
    form.add_argument("--acquisition", type=Path, required=True)
    form.add_argument("--prior-measurement-view", type=Path, required=True)
    form.add_argument("--previous", type=Path, required=True)
    form.add_argument("--current", type=Path, required=True)
    form.add_argument("--private-pack-output", type=Path, required=True)
    form.add_argument("--measurement-view-output", type=Path, required=True)
    form.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preregister":
        write_json(args.output, build_preregistration_v1(args.project_root))
        return 0
    if args.command == "acquire":
        receipt = build_acquisition_receipt_v1(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
            previous_archive=args.previous,
            current_archive=args.current,
            current_last_modified=args.current_last_modified,
        )
        write_json(args.output, receipt)
        return 0
    artifacts = form_fresh_pack_v1(
        project_root=args.project_root,
        preregistration_path=args.preregistration,
        prior_measurement_view_path=args.prior_measurement_view,
        preregistration=read_json(args.preregistration),
        acquisition=read_json(args.acquisition),
        previous_archive=args.previous,
        current_archive=args.current,
        prior_measurement_view=read_json(args.prior_measurement_view),
    )
    write_json(args.private_pack_output, artifacts.private_pack)
    write_json(args.measurement_view_output, artifacts.measurement_view)
    write_json(args.receipt_output, artifacts.formation_receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())

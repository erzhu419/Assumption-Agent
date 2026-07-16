"""Acquire the fresh MuSiQue residual cohort for the portfolio evaluator.

The earlier recursive MuSiQue study used ranks ``[0, 96)`` under a private,
already-fixed HMAC ordering of structurally eligible official DEV rows.  This
module reuses that exact secret and domain and allocates the mechanical next
window ``[96, 264)``.  No earlier private block is needed to prove or execute
the disjoint allocation.

Preregistration is zero-row: it reads committed public lineage, the selection
secret, and implementation bytes, but neither the official archive nor an old
private block.  Acquisition is one shot.  All output directories and atomic
persistence primitives are exercised before the durable consumption marker;
only then may the immutable source archive be opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Mapping, Sequence
import zipfile

from ..models import stable_hash
from . import musique_recursive_study_acquisition_v1 as prior
from .hotpot_family_out_acquisition_v1 import committed_public_file_receipt
from .musique_official_core_comparison_v1 import (
    OFFICIAL_SOURCE_COMMIT,
    _assert_git_ignored_private_path,
    _canonical_bytes,
    _iter_source_rows,
    _normalize_source_row,
    _read_selection_secret,
    _selection_secret_commitment,
    _sha256_bytes,
    _sha256_file,
    official_source_receipt,
)


VERSION = "musique_evaluator_portfolio_acquisition_v1"
PREREGISTRATION_SCHEMA = f"{VERSION}_preregistration"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition"
PRIVATE_ROW_SCHEMA = f"{VERSION}_private_row"
PRIVATE_LOCATOR_SCHEMA = f"{VERSION}_private_locator"
CONSUMPTION_SCHEMA = f"{VERSION}_consumption"

BLOCK_ORDER = (
    "A_form_0",
    "A_form_1",
    "F_search_0",
    "F_search_1",
    "A_hold",
    "M_search",
)
BLOCK_COUNTS = {
    "A_form_0": 24,
    "A_form_1": 24,
    "F_search_0": 24,
    "F_search_1": 24,
    "A_hold": 48,
    "M_search": 24,
}
SELECTED_COUNT = sum(BLOCK_COUNTS.values())
PREVIOUS_RANK_WINDOW_START = 0
PREVIOUS_RANK_WINDOW_STOP = prior.SELECTED_COUNT
RANK_WINDOW_START = prior.SELECTED_COUNT
RANK_WINDOW_STOP = RANK_WINDOW_START + SELECTED_COUNT
SELECTION_DOMAIN_SEPARATOR = prior.VERSION

TOP_K = prior.TOP_K
OFFICIAL_ARCHIVE_SHA256 = prior.OFFICIAL_ARCHIVE_SHA256
OFFICIAL_DEV_MEMBER_BASENAME = prior.OFFICIAL_DEV_MEMBER_BASENAME
OFFICIAL_DEV_MEMBER_SHA256 = (
    "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
)
EXPECTED_ELIGIBLE_ROWS = 594

PRIOR_ACQUISITION_RELATIVE = (
    "manifests/musique_recursive_evaluator_study_v1_acquisition.json"
)
PRIOR_PREREGISTRATION_RELATIVE = (
    "manifests/musique_recursive_evaluator_study_v1_preregistration.json"
)
PRIOR_PREREGISTRATION_FILE_SHA256 = (
    "caf401166db501d72465482f0626ad28ca49c7f4b5540fa8d71db204099bccc1"
)
PRIOR_PREREGISTRATION_SHA256 = (
    "680775501d5058fa7db0aeda67d75fc2e72cdbe43b057faa8578d72610465247"
)
PRIOR_ACQUISITION_FILE_SHA256 = (
    "a0c71304acf406ad4b7963b2e0f6e4ab0d19cbc413e0034067edf609ee32ccbf"
)
PRIOR_ACQUISITION_SHA256 = (
    "1d2215fe8ed06fc29ac0dc63eddae99e9d76f82b22516f6cd03660edbdd3c354"
)
PORTFOLIO_DESIGN_RELATIVE = "manifests/musique_evaluator_portfolio_design_v1.json"
PORTFOLIO_DESIGN_SCHEMA = "musique_evaluator_portfolio_design_v1"
CONSUMPTION_RELATIVE = (
    "artifacts/musique_evaluator_portfolio_acquisition_v1/"
    "authorization.consumed.json"
)
PREREGISTRATION_RELATIVE = (
    "manifests/musique_evaluator_portfolio_acquisition_v1_preregistration.json"
)
ACQUISITION_RELATIVE = (
    "manifests/musique_evaluator_portfolio_acquisition_v1_acquisition.json"
)

PRIOR_LINEAGE_RELATIVE_FILES = (
    (
        "retained_P_formation_receipt",
        "manifests/musique_recursive_study_f1_formation_v1/formation.receipt.json",
    ),
    (
        "retained_P_frozen_program",
        "manifests/musique_recursive_study_f1_formation_v1/frozen_program.json",
    ),
    (
        "retained_P_measurement_freeze",
        "manifests/musique_recursive_study_m1_pre_run_freeze_v1.json",
    ),
    (
        "retained_P_measurement_report",
        "manifests/musique_recursive_study_m1_aggregate_report_v1.json",
    ),
    (
        "prior_L5_terminal_disposition",
        "manifests/musique_recursive_evaluator_coevolution_disposition_v1.json",
    ),
)

# The runner is intentionally prospective: preregistration is unavailable
# until both modules and the fixed design are clean committed HEAD blobs.
IMPLEMENTATION_RELATIVE_FILES = tuple(
    dict.fromkeys(
        (
            *prior.IMPLEMENTATION_RELATIVE_FILES,
            "assumption_agent/benchmarks/hotpot_family_out_acquisition_v1.py",
            "assumption_agent/benchmarks/hotpot_family_out_runner_v1.py",
            "assumption_agent/benchmarks/hotpot_evaluator_coevolution_v2.py",
            "assumption_agent/benchmarks/hotpot_evaluator_portfolio_coevolution_v1.py",
            "assumption_agent/benchmarks/hotpot_recursive_l4_v1.py",
            "assumption_agent/benchmarks/musique_evaluator_portfolio_acquisition_v1.py",
            "assumption_agent/benchmarks/musique_evaluator_portfolio_coevolution_v1.py",
        )
    )
)

PRIVATE_BLOCK_ROW_KEYS = frozenset(
    {
        "schema",
        "block",
        "item_id",
        "question",
        "corpus",
        "answers",
        "normalized_answers",
        "support_indices",
        "source_row_sha256",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MIN_FREE_BYTES = 256 * 1024 * 1024


class MuSiQuePortfolioAcquisitionError(RuntimeError):
    """Raised when the fixed residual selection or custody contract drifts."""


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    count: int
    file_sha256: str
    item_commitment_set_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQuePortfolioAcquisitionError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _read_json_object(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise MuSiQuePortfolioAcquisitionError(f"{field} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQuePortfolioAcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise MuSiQuePortfolioAcquisitionError(f"{field} must be one object")
    return value, raw


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
        '"selection_secret_path"',
        '"support_indices"',
    )
    if any(token in serialized for token in forbidden):
        raise MuSiQuePortfolioAcquisitionError(
            "public artifact contains private content or a private locator"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    """Persist bytes atomically without ever replacing an existing path."""

    if not path.parent.is_dir() or path.parent.is_symlink():
        raise MuSiQuePortfolioAcquisitionError("output parent is unavailable")
    nonce = os.urandom(12).hex()
    temporary = path.parent / f".{path.name}.{nonce}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    mode: int,
) -> None:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = stable_hash(body)
    raw = (json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_write_exclusive(path, raw, mode=mode)


def _write_jsonl_exclusive(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _atomic_write_exclusive(path, raw, mode=0o600)
    return _sha256_bytes(raw), stable_hash([stable_hash(row) for row in rows])


def _committed_binding(*, project: Path, path: Path, field: str) -> dict[str, Any]:
    try:
        receipt = committed_public_file_receipt(project=project, path=path)
    except Exception as exc:
        raise MuSiQuePortfolioAcquisitionError(
            f"{field} must be the clean tracked HEAD blob"
        ) from exc
    file_sha256 = receipt["preregistration_file_sha256"]
    if file_sha256 != receipt["preregistration_head_blob_sha256"]:
        raise MuSiQuePortfolioAcquisitionError(f"{field} HEAD binding drifted")
    return {
        "file_sha256": file_sha256,
        "head_blob_sha256": file_sha256,
        "clean_tracked_HEAD_blob": True,
    }


def _canonical_public_path(
    *, project: Path, supplied: Path, canonical_relative: str, field: str
) -> Path:
    root = project.resolve(strict=True)
    candidate = supplied if supplied.is_absolute() else root / supplied
    candidate = candidate.absolute()
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQuePortfolioAcquisitionError(
                f"{field} may not traverse a symbolic link"
            )
    expected = root / canonical_relative
    try:
        actual = candidate.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MuSiQuePortfolioAcquisitionError(
            f"canonical {field} is unavailable"
        ) from exc
    if actual != canonical:
        raise MuSiQuePortfolioAcquisitionError(
            f"{field} must use its fixed canonical path"
        )
    return canonical


def implementation_binding(project: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQuePortfolioAcquisitionError(
                f"implementation file missing or symlinked: {relative}"
            )
        custody = _committed_binding(
            project=project, path=path, field=f"implementation {relative}"
        )
        live = _sha256_file(path)
        if custody["file_sha256"] != live:
            raise MuSiQuePortfolioAcquisitionError(
                f"implementation file drifted: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": live,
                "head_blob_sha256": live,
                "clean_tracked_HEAD_blob": True,
            }
        )
    return {"files": rows, "set_sha256": stable_hash(rows)}


def portfolio_design_binding(project: Path) -> dict[str, Any]:
    path = project / PORTFOLIO_DESIGN_RELATIVE
    payload, raw = _read_json_object(path, "portfolio design")
    body = dict(payload)
    declared = _require_sha256(body.pop("design_sha256", None), "design hash")
    if (
        payload.get("schema") != PORTFOLIO_DESIGN_SCHEMA
        or stable_hash(body) != declared
        or payload.get("raw_content_persisted") is not False
    ):
        raise MuSiQuePortfolioAcquisitionError("portfolio design drifted")
    custody = _committed_binding(project=project, path=path, field="portfolio design")
    file_sha256 = _sha256_bytes(raw)
    if custody["file_sha256"] != file_sha256:
        raise MuSiQuePortfolioAcquisitionError("portfolio design custody drifted")
    return {
        "relative_path": PORTFOLIO_DESIGN_RELATIVE,
        "schema": PORTFOLIO_DESIGN_SCHEMA,
        "design_file_sha256": file_sha256,
        "design_sha256": declared,
        "committed_custody": custody,
    }


def prior_study_lineage_binding(project: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for role, relative in PRIOR_LINEAGE_RELATIVE_FILES:
        path = project / relative
        payload, raw = _read_json_object(path, role)
        custody = _committed_binding(project=project, path=path, field=role)
        file_sha256 = _sha256_bytes(raw)
        if custody["file_sha256"] != file_sha256:
            raise MuSiQuePortfolioAcquisitionError(f"{role} custody drifted")
        if role == "retained_P_measurement_report":
            promotion = payload.get("measurement", {}).get("promotion_disposition", {})
            if (
                payload.get("schema") != "musique_generation_one_m1_aggregate_report_v1"
                or payload.get("valid") is not True
                or promotion.get("disposition")
                != "promote_P_to_retained_generation_one"
            ):
                raise MuSiQuePortfolioAcquisitionError(
                    "retained P promotion lineage drifted"
                )
        if role == "prior_L5_terminal_disposition" and (
            payload.get("schema")
            != "musique_recursive_evaluator_coevolution_disposition_v1"
            or payload.get("decision")
            != "no_evaluator_transition_no_positive_search_utility"
        ):
            raise MuSiQuePortfolioAcquisitionError("prior L5 disposition drifted")
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "file_sha256": file_sha256,
                "committed_custody": custody,
            }
        )
    return {
        "files": rows,
        "set_sha256": stable_hash(rows),
        "retained_P_fixed_before_residual_selection": True,
        "prior_L5_outcomes_not_used_for_rank_selection": True,
    }


def _load_prior_acquisition_binding(
    *, project: Path, supplied_path: Path, secret: bytes
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _canonical_public_path(
        project=project,
        supplied=supplied_path,
        canonical_relative=PRIOR_ACQUISITION_RELATIVE,
        field="prior acquisition receipt",
    )
    payload, raw = _read_json_object(path, "prior acquisition receipt")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "prior acquisition hash"
    )
    commitments = payload.get("commitments")
    counts = payload.get("counts")
    source = payload.get("source")
    if (
        _sha256_bytes(raw) != PRIOR_ACQUISITION_FILE_SHA256
        or payload.get("schema") != prior.ACQUISITION_SCHEMA
        or stable_hash(body) != declared
        or declared != PRIOR_ACQUISITION_SHA256
        or not isinstance(commitments, Mapping)
        or commitments.get("selection_secret_commitment_sha256")
        != _selection_secret_commitment(secret)
        or not isinstance(counts, Mapping)
        or counts.get("selected_rows") != PREVIOUS_RANK_WINDOW_STOP
        or counts.get("eligible_rows") != EXPECTED_ELIGIBLE_ROWS
        or not isinstance(source, Mapping)
        or source.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256
        or source.get("official_dev_member_sha256")
        != OFFICIAL_DEV_MEMBER_SHA256
        or source.get("source_split") != "official_dev"
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "prior acquisition ordering, source, or secret drifted"
        )
    custody = _committed_binding(
        project=project, path=path, field="prior acquisition receipt"
    )
    return payload, {
        "relative_path": PRIOR_ACQUISITION_RELATIVE,
        "schema": payload["schema"],
        "acquisition_file_sha256": _sha256_bytes(raw),
        "acquisition_sha256": declared,
        "committed_custody": custody,
        "selected_rank_window_start_inclusive": PREVIOUS_RANK_WINDOW_START,
        "selected_rank_window_stop_exclusive": PREVIOUS_RANK_WINDOW_STOP,
        "selected_count": PREVIOUS_RANK_WINDOW_STOP,
        "selection_domain_separator": SELECTION_DOMAIN_SEPARATOR,
        "selection_secret_commitment_sha256": commitments[
            "selection_secret_commitment_sha256"
        ],
        "private_block_files_opened": 0,
    }


def _load_prior_preregistration_binding(
    *, project: Path, supplied_path: Path, secret: bytes
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the exact prior normalizer/selection implementation closure."""

    path = _canonical_public_path(
        project=project,
        supplied=supplied_path,
        canonical_relative=PRIOR_PREREGISTRATION_RELATIVE,
        field="prior preregistration",
    )
    payload, raw = _read_json_object(path, "prior preregistration")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("preregistration_sha256", None), "prior preregistration hash"
    )
    selection = payload.get("selection")
    source = payload.get("source")
    live_prior_implementation = prior._implementation_binding(project)
    if (
        _sha256_bytes(raw) != PRIOR_PREREGISTRATION_FILE_SHA256
        or payload.get("schema") != prior.PREREGISTRATION_SCHEMA
        or stable_hash(body) != declared
        or declared != PRIOR_PREREGISTRATION_SHA256
        or payload.get("implementation") != live_prior_implementation
        or not isinstance(selection, Mapping)
        or selection.get("selected_count") != PREVIOUS_RANK_WINDOW_STOP
        or selection.get("selection_secret_commitment_sha256")
        != _selection_secret_commitment(secret)
        or selection.get("algorithm")
        != "ascending_hmac_sha256_private_secret_and_official_item_id_v1"
        or not isinstance(source, Mapping)
        or source.get("official_archive_sha256") != OFFICIAL_ARCHIVE_SHA256
        or source.get("source_split") != "official_dev"
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "prior preregistration dependency closure drifted"
        )
    custody = _committed_binding(
        project=project, path=path, field="prior preregistration"
    )
    return payload, {
        "relative_path": PRIOR_PREREGISTRATION_RELATIVE,
        "schema": payload["schema"],
        "preregistration_file_sha256": _sha256_bytes(raw),
        "preregistration_sha256": declared,
        "implementation_set_sha256": live_prior_implementation["set_sha256"],
        "selection_secret_commitment_sha256": selection[
            "selection_secret_commitment_sha256"
        ],
        "selection_domain_separator": SELECTION_DOMAIN_SEPARATOR,
        "normalizer": payload["eligibility"]["normalizer"],
        "committed_custody": custody,
        "dataset_rows_read": 0,
    }


def _selection_key(item_id: str, secret: bytes) -> str:
    """Use the exact prior version/domain; a new domain would not continue ranks."""

    try:
        return prior._selection_key(item_id, secret)
    except Exception as exc:
        raise MuSiQuePortfolioAcquisitionError(
            "prior selection implementation rejected the fixed secret"
        ) from exc


def _assemble_preregistration(
    *,
    source_repository: object,
    source_license: object,
    secret_commitment: str,
    prior_preregistration_binding: Mapping[str, Any],
    prior_acquisition_binding: Mapping[str, Any],
    design_binding: Mapping[str, Any],
    retained_p_lineage: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the single exact preregistration shape used by build and audit."""

    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "decision": "residual_acquisition_only_no_formation_or_measurement_authority",
        "source": {
            "repository": source_repository,
            "commit": OFFICIAL_SOURCE_COMMIT,
            "license": source_license,
            "official_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "official_dev_member_sha256": OFFICIAL_DEV_MEMBER_SHA256,
            "archive_member_basename": OFFICIAL_DEV_MEMBER_BASENAME,
            "source_split": "official_dev",
            "canonical_prior_source_binding_reused": True,
        },
        "eligibility": {
            "normalizer": "musique_official_core_comparison_v1._normalize_source_row",
            "identical_to_prior_acquisition": True,
            "expected_eligible_rows": EXPECTED_ELIGIBLE_ROWS,
            "answerable": True,
            "minimum_paragraph_count": TOP_K,
            "minimum_supporting_paragraph_count": 2,
            "minimum_non_supporting_paragraph_count": 1,
            "minimum_distinct_normalized_answers": 2,
        },
        "prior_preregistration_binding": dict(prior_preregistration_binding),
        "prior_acquisition_binding": dict(prior_acquisition_binding),
        "portfolio_design_binding": dict(design_binding),
        "retained_P_lineage": dict(retained_p_lineage),
        "selection": {
            "algorithm": (
                "exact_prior_ascending_HMAC_SHA256_order_take_mechanical_"
                "continuation_window_v1"
            ),
            "domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": secret_commitment,
            "selection_secret_reused_from_prior_study": True,
            "selection_secret_persisted_publicly": False,
            "previous_rank_window_start_inclusive": PREVIOUS_RANK_WINDOW_START,
            "previous_rank_window_stop_exclusive": PREVIOUS_RANK_WINDOW_STOP,
            "rank_window_start_inclusive": RANK_WINDOW_START,
            "rank_window_stop_exclusive": RANK_WINDOW_STOP,
            "rank_window_disjoint_from_prior_by_construction": True,
            "selected_count": SELECTED_COUNT,
            "block_order": list(BLOCK_ORDER),
            "block_counts": dict(BLOCK_COUNTS),
            "replacement": False,
            "manual_or_outcome_conditioned_selection": False,
        },
        "access_contract": {
            "all_six_blocks_formed_together": True,
            "committed_preregistration_precedes_consumption_marker": True,
            "persistence_parents_and_atomic_rename_canaries_precede_marker": True,
            "pack_root_created_and_fsynced_before_marker": True,
            "one_shot_marker_precedes_source_archive_open": True,
            "prior_private_pack_parameter_accepted": False,
            "prior_private_block_rows_opened": 0,
            "measurement_requires_separate_pre_run_freeze": True,
            "retry_replay_resample": 0,
        },
        "study_contract": {
            "evaluator_formation_partitions": ["A_form_0", "A_form_1"],
            "search_formation_partitions": ["F_search_0", "F_search_1"],
            "evaluator_anchor_partition": "A_hold",
            "search_measurement_partition": "M_search",
            "primary_metric": "official_support_recall_at_5",
            "offline_evaluation_only": True,
            "online_evaluator_calls": 0,
            "study_level_answer_generator_calls": 0,
        },
        "implementation": dict(implementation),
        "claim_boundary": {
            "cross_family_evaluator_confirmation_only": True,
            "answer_generation_claim": False,
            "performance_claim_before_measurement": False,
            "public_benchmark_pretraining_cannot_be_excluded": True,
        },
        "safety": {
            "dataset_rows_read": 0,
            "source_archive_opened": False,
            "prior_private_block_rows_read": 0,
            "model_calls": 0,
            "network_calls": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
        },
    }
    _assert_public_safe(payload)
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def build_preregistration(
    *,
    project: Path,
    official_repository: Path,
    selection_secret_path: Path,
    prior_preregistration_path: Path,
    prior_acquisition_receipt_path: Path,
) -> dict[str, Any]:
    """Build the complete preregistration without opening any dataset row."""

    root = project.resolve(strict=True)
    secret = _read_selection_secret(project=root, path=selection_secret_path)
    source_receipt = official_source_receipt(official_repository)
    if source_receipt.get("commit") != OFFICIAL_SOURCE_COMMIT:
        raise MuSiQuePortfolioAcquisitionError("official source commit drifted")
    _prior_prereg, prior_prereg_binding = _load_prior_preregistration_binding(
        project=root, supplied_path=prior_preregistration_path, secret=secret
    )
    prior_receipt, prior_binding = _load_prior_acquisition_binding(
        project=root, supplied_path=prior_acquisition_receipt_path, secret=secret
    )
    if (
        prior_receipt.get("ordering", {}).get("preregistration_sha256")
        != prior_prereg_binding["preregistration_sha256"]
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "prior acquisition does not descend from the canonical preregistration"
        )
    return _assemble_preregistration(
        source_repository=source_receipt["repository"],
        source_license=source_receipt["license"],
        secret_commitment=_selection_secret_commitment(secret),
        prior_preregistration_binding=prior_prereg_binding,
        prior_acquisition_binding=prior_binding,
        design_binding=portfolio_design_binding(root),
        retained_p_lineage=prior_study_lineage_binding(root),
        implementation=implementation_binding(root),
    )


def verify_preregistration(
    *,
    path: Path,
    project: Path,
    official_repository: Path,
    selection_secret_path: Path,
    prior_preregistration_path: Path,
    prior_acquisition_receipt_path: Path,
) -> dict[str, Any]:
    payload, _raw = _read_json_object(path, "portfolio preregistration")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("preregistration_sha256", None), "preregistration hash"
    )
    if payload.get("schema") != PREREGISTRATION_SCHEMA or stable_hash(body) != declared:
        raise MuSiQuePortfolioAcquisitionError("preregistration self-hash drifted")
    expected = build_preregistration(
        project=project,
        official_repository=official_repository,
        selection_secret_path=selection_secret_path,
        prior_preregistration_path=prior_preregistration_path,
        prior_acquisition_receipt_path=prior_acquisition_receipt_path,
    )
    if payload != expected:
        raise MuSiQuePortfolioAcquisitionError(
            "preregistration differs from the complete live protocol"
        )
    return payload


def _persistence_canary(directory: Path) -> None:
    """Exercise the exact formal hardlink/fsync primitive, then clean it up."""

    if directory.is_symlink() or not directory.is_dir():
        raise MuSiQuePortfolioAcquisitionError("persistence directory is unsafe")
    token = os.urandom(12).hex()
    target = directory / f".{VERSION}.{token}.canary"
    expected = b"portfolio-acquisition-persistence-canary\n"
    try:
        _atomic_write_exclusive(target, expected, mode=0o600)
        if (
            target.read_bytes() != expected
            or stat.S_IMODE(target.stat().st_mode) & 0o077
        ):
            raise MuSiQuePortfolioAcquisitionError(
                "persistence canary verification failed"
            )
    finally:
        target.unlink(missing_ok=True)
        _fsync_directory(directory)


def _preflight_persistence(
    *,
    pack_root: Path,
    locator: Path,
    consumption_path: Path,
    public_receipt_path: Path,
) -> None:
    """Eliminate foreseeable persistence failures before authorization burn."""

    if consumption_path.exists():
        raise FileExistsError("portfolio acquisition authorization was already consumed")
    if pack_root.exists() or locator.exists() or public_receipt_path.exists():
        raise FileExistsError("portfolio acquisition output already exists")
    paths = (pack_root, locator, consumption_path, public_receipt_path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise MuSiQuePortfolioAcquisitionError(
                    "private and public output paths must be disjoint"
                )

    directories = {
        pack_root.parent,
        locator.parent,
        consumption_path.parent,
        public_receipt_path.parent,
    }
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise MuSiQuePortfolioAcquisitionError("output parent is unsafe")
    pack_root_created = False
    try:
        os.mkdir(pack_root, 0o700)
        pack_root_created = True
        os.chmod(pack_root, 0o700)
        if stat.S_IMODE(pack_root.stat().st_mode) != 0o700:
            raise MuSiQuePortfolioAcquisitionError(
                "private pack root permissions cannot be restricted"
            )
        _fsync_directory(pack_root.parent)
        for directory in {*directories, pack_root}:
            _persistence_canary(directory)
        # Check every output filesystem explicitly; paths may cross mount points.
        for output_directory in (
            pack_root,
            locator.parent,
            consumption_path.parent,
            public_receipt_path.parent,
        ):
            if shutil.disk_usage(output_directory).free < _MIN_FREE_BYTES:
                raise MuSiQuePortfolioAcquisitionError(
                    "insufficient free space for one-shot acquisition"
                )
    except BaseException:
        if pack_root_created:
            try:
                pack_root.rmdir()
                _fsync_directory(pack_root.parent)
            except OSError:
                pass
        raise


def load_private_block(
    path: str | Path,
    *,
    commitment: BlockCommitment,
    expected_block: str | None = None,
) -> tuple[dict[str, Any], ...]:
    block = commitment.block if expected_block is None else expected_block
    if block != commitment.block or block not in BLOCK_ORDER:
        raise MuSiQuePortfolioAcquisitionError("private block identity drifted")
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise MuSiQuePortfolioAcquisitionError("private block is unavailable")
    raw = candidate.read_bytes()
    if _sha256_bytes(raw) != commitment.file_sha256 or not raw.endswith(b"\n"):
        raise MuSiQuePortfolioAcquisitionError("private block file hash drifted")
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise MuSiQuePortfolioAcquisitionError("private row is malformed")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQuePortfolioAcquisitionError("private block JSONL is invalid") from exc
    if (
        len(rows) != commitment.count
        or commitment.count != BLOCK_COUNTS[block]
        or any(set(row) != PRIVATE_BLOCK_ROW_KEYS for row in rows)
        or any(row.get("schema") != PRIVATE_ROW_SCHEMA for row in rows)
        or any(row.get("block") != block for row in rows)
        or b"".join(_canonical_bytes(row) + b"\n" for row in rows) != raw
        or stable_hash([stable_hash(row) for row in rows])
        != commitment.item_commitment_set_sha256
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "private block schema, count, or commitment drifted"
        )
    ids = [row["item_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise MuSiQuePortfolioAcquisitionError("private block IDs are not unique")
    return tuple(rows)


def _valid_committed_custody(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {"clean_tracked_HEAD_blob", "file_sha256", "head_blob_sha256"}
        and value.get("clean_tracked_HEAD_blob") is True
        and _SHA256_RE.fullmatch(str(value.get("file_sha256"))) is not None
        and value.get("head_blob_sha256") == value.get("file_sha256")
    )


def _valid_implementation_shape(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"files", "set_sha256"}:
        return False
    rows = value.get("files")
    if not isinstance(rows, list) or value.get("set_sha256") != stable_hash(rows):
        return False
    if [row.get("path") for row in rows if isinstance(row, Mapping)] != list(
        IMPLEMENTATION_RELATIVE_FILES
    ):
        return False
    return all(
        isinstance(row, Mapping)
        and set(row)
        == {
            "clean_tracked_HEAD_blob",
            "head_blob_sha256",
            "path",
            "sha256",
        }
        and row.get("clean_tracked_HEAD_blob") is True
        and row.get("head_blob_sha256") == row.get("sha256")
        and _SHA256_RE.fullmatch(str(row.get("sha256"))) is not None
        for row in rows
    )


def _valid_lineage_shape(value: object) -> bool:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "files",
            "prior_L5_outcomes_not_used_for_rank_selection",
            "retained_P_fixed_before_residual_selection",
            "set_sha256",
        }
        or value.get("retained_P_fixed_before_residual_selection") is not True
        or value.get("prior_L5_outcomes_not_used_for_rank_selection") is not True
    ):
        return False
    rows = value.get("files")
    if not isinstance(rows, list) or value.get("set_sha256") != stable_hash(rows):
        return False
    expected = list(PRIOR_LINEAGE_RELATIVE_FILES)
    if [
        (row.get("role"), row.get("relative_path"))
        for row in rows
        if isinstance(row, Mapping)
    ] != expected:
        return False
    return all(
        isinstance(row, Mapping)
        and set(row)
        == {"committed_custody", "file_sha256", "relative_path", "role"}
        and _SHA256_RE.fullmatch(str(row.get("file_sha256"))) is not None
        and _valid_committed_custody(row.get("committed_custody"))
        and row["committed_custody"]["file_sha256"] == row["file_sha256"]
        for row in rows
    )


def _valid_design_binding_shape(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "committed_custody",
            "design_file_sha256",
            "design_sha256",
            "relative_path",
            "schema",
        }
        and value.get("relative_path") == PORTFOLIO_DESIGN_RELATIVE
        and value.get("schema") == PORTFOLIO_DESIGN_SCHEMA
        and _SHA256_RE.fullmatch(str(value.get("design_file_sha256"))) is not None
        and _SHA256_RE.fullmatch(str(value.get("design_sha256"))) is not None
        and _valid_committed_custody(value.get("committed_custody"))
        and value["committed_custody"]["file_sha256"]
        == value["design_file_sha256"]
    )


def _valid_prior_acquisition_binding_shape(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "acquisition_file_sha256",
            "acquisition_sha256",
            "committed_custody",
            "private_block_files_opened",
            "relative_path",
            "schema",
            "selected_count",
            "selected_rank_window_start_inclusive",
            "selected_rank_window_stop_exclusive",
            "selection_domain_separator",
            "selection_secret_commitment_sha256",
        }
        and value.get("relative_path") == PRIOR_ACQUISITION_RELATIVE
        and value.get("schema") == prior.ACQUISITION_SCHEMA
        and value.get("acquisition_file_sha256") == PRIOR_ACQUISITION_FILE_SHA256
        and value.get("acquisition_sha256") == PRIOR_ACQUISITION_SHA256
        and value.get("selected_count") == PREVIOUS_RANK_WINDOW_STOP
        and value.get("selected_rank_window_start_inclusive")
        == PREVIOUS_RANK_WINDOW_START
        and value.get("selected_rank_window_stop_exclusive")
        == PREVIOUS_RANK_WINDOW_STOP
        and value.get("selection_domain_separator") == SELECTION_DOMAIN_SEPARATOR
        and _SHA256_RE.fullmatch(
            str(value.get("selection_secret_commitment_sha256"))
        )
        is not None
        and value.get("private_block_files_opened") == 0
        and _valid_committed_custody(value.get("committed_custody"))
        and value["committed_custody"]["file_sha256"]
        == value["acquisition_file_sha256"]
    )


def _valid_prior_preregistration_binding_shape(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "committed_custody",
            "dataset_rows_read",
            "implementation_set_sha256",
            "normalizer",
            "preregistration_file_sha256",
            "preregistration_sha256",
            "relative_path",
            "schema",
            "selection_domain_separator",
            "selection_secret_commitment_sha256",
        }
        and value.get("relative_path") == PRIOR_PREREGISTRATION_RELATIVE
        and value.get("schema") == prior.PREREGISTRATION_SCHEMA
        and value.get("preregistration_file_sha256")
        == PRIOR_PREREGISTRATION_FILE_SHA256
        and value.get("preregistration_sha256") == PRIOR_PREREGISTRATION_SHA256
        and _SHA256_RE.fullmatch(str(value.get("implementation_set_sha256")))
        is not None
        and value.get("normalizer")
        == "musique_official_core_comparison_v1._normalize_source_row"
        and value.get("selection_domain_separator") == SELECTION_DOMAIN_SEPARATOR
        and _SHA256_RE.fullmatch(
            str(value.get("selection_secret_commitment_sha256"))
        )
        is not None
        and value.get("dataset_rows_read") == 0
        and _valid_committed_custody(value.get("committed_custody"))
        and value["committed_custody"]["file_sha256"]
        == value["preregistration_file_sha256"]
    )


def load_acquisition_binding(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[BlockCommitment, ...]]:
    receipt, _raw = _read_json_object(Path(path), "portfolio acquisition receipt")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "acquisition hash"
    )
    if receipt.get("schema") != ACQUISITION_SCHEMA or stable_hash(body) != declared:
        raise MuSiQuePortfolioAcquisitionError("acquisition receipt self-hash drifted")
    rows = receipt.get("commitments", {}).get("block_files")
    if not isinstance(rows, list) or len(rows) != len(BLOCK_ORDER):
        raise MuSiQuePortfolioAcquisitionError("block commitments are malformed")
    blocks: list[BlockCommitment] = []
    for expected, row in zip(BLOCK_ORDER, rows):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"block", "count", "file_sha256", "item_commitment_set_sha256"}
            or row.get("block") != expected
            or row.get("count") != BLOCK_COUNTS[expected]
        ):
            raise MuSiQuePortfolioAcquisitionError("block commitment drifted")
        blocks.append(
            BlockCommitment(
                block=expected,
                count=BLOCK_COUNTS[expected],
                file_sha256=_require_sha256(row.get("file_sha256"), "block hash"),
                item_commitment_set_sha256=_require_sha256(
                    row.get("item_commitment_set_sha256"), "item commitment set"
                ),
            )
        )
    expected_top_level = {
        "acquisition_sha256",
        "commitments",
        "counts",
        "decision",
        "implementation",
        "portfolio_design_binding",
        "preregistration_custody",
        "preregistration_sha256",
        "prior_acquisition_binding",
        "prior_preregistration_binding",
        "prospective_ordering",
        "retained_P_lineage",
        "safety",
        "schema",
        "selection_continuity",
        "source",
    }
    counts = receipt.get("counts")
    selection = receipt.get("selection_continuity")
    commitments = receipt.get("commitments")
    ordering = receipt.get("prospective_ordering")
    source = receipt.get("source")
    prior_binding = receipt.get("prior_acquisition_binding")
    prior_prereg_binding = receipt.get("prior_preregistration_binding")
    implementation = receipt.get("implementation")
    implementation_rows = (
        implementation.get("files") if isinstance(implementation, Mapping) else None
    )
    lineage = receipt.get("retained_P_lineage")
    lineage_rows = lineage.get("files") if isinstance(lineage, Mapping) else None
    design = receipt.get("portfolio_design_binding")
    safety = receipt.get("safety")
    if (
        set(receipt) != expected_top_level
        or receipt.get("decision")
        != "fresh_residual_six_block_pack_formed_no_measurement_authority"
        or _SHA256_RE.fullmatch(str(receipt.get("preregistration_sha256"))) is None
        or not _valid_committed_custody(receipt.get("preregistration_custody"))
        or not isinstance(counts, Mapping)
        or set(counts)
        != {
            "blocks",
            "eligible_rows",
            "rank_window_start_inclusive",
            "rank_window_stop_exclusive",
            "selected_previous_rank_window_overlap",
            "selected_rows",
            "source_rows",
        }
        or counts.get("blocks") != BLOCK_COUNTS
        or counts.get("eligible_rows") != EXPECTED_ELIGIBLE_ROWS
        or counts.get("selected_rows") != SELECTED_COUNT
        or counts.get("rank_window_start_inclusive") != RANK_WINDOW_START
        or counts.get("rank_window_stop_exclusive") != RANK_WINDOW_STOP
        or counts.get("selected_previous_rank_window_overlap") != 0
        or type(counts.get("source_rows")) is not int
        or counts["source_rows"] < counts["eligible_rows"]
        or not isinstance(selection, Mapping)
        or set(selection)
        != {
            "prior_private_block_rows_opened",
            "prior_public_outcomes_used_for_rank_selection",
            "previous_rank_window_start_inclusive",
            "previous_rank_window_stop_exclusive",
            "rank_window_disjoint_from_prior_by_construction",
            "rank_window_start_inclusive",
            "rank_window_stop_exclusive",
            "same_selection_secret_as_prior",
            "selection_domain_separator",
            "selection_secret_commitment_sha256",
        }
        or selection.get("selection_domain_separator")
        != SELECTION_DOMAIN_SEPARATOR
        or selection.get("same_selection_secret_as_prior") is not True
        or selection.get("previous_rank_window_start_inclusive")
        != PREVIOUS_RANK_WINDOW_START
        or selection.get("previous_rank_window_stop_exclusive")
        != PREVIOUS_RANK_WINDOW_STOP
        or selection.get("rank_window_start_inclusive") != RANK_WINDOW_START
        or selection.get("rank_window_stop_exclusive") != RANK_WINDOW_STOP
        or selection.get("rank_window_disjoint_from_prior_by_construction")
        is not True
        or selection.get("prior_private_block_rows_opened") != 0
        or selection.get("prior_public_outcomes_used_for_rank_selection") is not False
        or _SHA256_RE.fullmatch(
            str(selection.get("selection_secret_commitment_sha256"))
        )
        is None
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "archive_sha256",
            "dataset",
            "official_dev_member_sha256",
            "repository_commit",
            "source_split",
        }
        or source.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256
        or source.get("dataset") != "MuSiQue-Answerable v1.0"
        or source.get("official_dev_member_sha256")
        != OFFICIAL_DEV_MEMBER_SHA256
        or source.get("repository_commit") != OFFICIAL_SOURCE_COMMIT
        or source.get("source_split") != "official_dev"
        or not _valid_prior_acquisition_binding_shape(prior_binding)
        or not _valid_prior_preregistration_binding_shape(prior_prereg_binding)
        or prior_binding.get("selection_secret_commitment_sha256")
        != prior_prereg_binding.get("selection_secret_commitment_sha256")
        or selection.get("selection_secret_commitment_sha256")
        != prior_binding.get("selection_secret_commitment_sha256")
        or not _valid_design_binding_shape(design)
        or not _valid_implementation_shape(implementation)
        or not _valid_lineage_shape(lineage)
        or not isinstance(ordering, Mapping)
        or set(ordering)
        != {
            "acquisition_consumption_file_sha256",
            "acquisition_consumption_sha256",
            "consumption_persisted_before_source_archive_open",
            "pack_root_created_before_consumption",
            "persistence_preflight_complete_before_consumption",
            "preregistration_committed_before_consumption",
            "retry_replay_resample_authorized",
            "source_rows_opened_before_consumption",
        }
        or ordering.get("preregistration_committed_before_consumption") is not True
        or ordering.get("persistence_preflight_complete_before_consumption") is not True
        or ordering.get("pack_root_created_before_consumption") is not True
        or ordering.get("consumption_persisted_before_source_archive_open") is not True
        or ordering.get("source_rows_opened_before_consumption") != 0
        or ordering.get("retry_replay_resample_authorized") is not False
        or _SHA256_RE.fullmatch(
            str(ordering.get("acquisition_consumption_file_sha256"))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(ordering.get("acquisition_consumption_sha256"))
        )
        is None
        or not isinstance(commitments, Mapping)
        or set(commitments)
        != {
            "block_files",
            "item_ids_persisted_publicly",
            "private_locator_file_sha256",
            "private_pack_sha256",
            "private_paths_persisted_publicly",
            "private_row_key_set_sha256",
            "selection_secret_commitment_sha256",
        }
        or commitments.get("private_pack_sha256")
        != stable_hash([block.to_dict() for block in blocks])
        or commitments.get("selection_secret_commitment_sha256")
        != selection.get("selection_secret_commitment_sha256")
        or commitments.get("private_row_key_set_sha256")
        != stable_hash(sorted(PRIVATE_BLOCK_ROW_KEYS))
        or commitments.get("item_ids_persisted_publicly") is not False
        or commitments.get("private_paths_persisted_publicly") is not False
        or _SHA256_RE.fullmatch(
            str(commitments.get("private_locator_file_sha256"))
        )
        is None
        or not isinstance(safety, Mapping)
        or safety
        != {
            "formation_executed": False,
            "measurement_executed": False,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "prior_private_block_rows_read": 0,
            "scores_computed": 0,
        }
    ):
        raise MuSiQuePortfolioAcquisitionError("acquisition receipt contract drifted")
    _assert_public_safe(receipt)
    return receipt, tuple(blocks)


def _load_preregistration_live(
    *, project: Path, selection_secret_path: Path
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    """Load the exact clean canonical preregistration and rebuild its meaning."""

    root = project.resolve(strict=True)
    canonical = _canonical_public_path(
        project=root,
        supplied=root / PREREGISTRATION_RELATIVE,
        canonical_relative=PREREGISTRATION_RELATIVE,
        field="portfolio preregistration",
    )
    payload, raw = _read_json_object(canonical, "portfolio preregistration")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("preregistration_sha256", None), "preregistration hash"
    )
    if payload.get("schema") != PREREGISTRATION_SCHEMA or stable_hash(body) != declared:
        raise MuSiQuePortfolioAcquisitionError(
            "canonical preregistration self-hash drifted"
        )
    custody = _committed_binding(
        project=root, path=canonical, field="portfolio preregistration"
    )
    if custody["file_sha256"] != _sha256_bytes(raw):
        raise MuSiQuePortfolioAcquisitionError(
            "canonical preregistration file custody drifted"
        )
    secret = _read_selection_secret(project=root, path=selection_secret_path)
    secret_commitment = _selection_secret_commitment(secret)
    prior_prereg, prior_prereg_binding = _load_prior_preregistration_binding(
        project=root,
        supplied_path=root / PRIOR_PREREGISTRATION_RELATIVE,
        secret=secret,
    )
    prior_receipt, prior_acquisition_binding = _load_prior_acquisition_binding(
        project=root,
        supplied_path=root / PRIOR_ACQUISITION_RELATIVE,
        secret=secret,
    )
    if (
        prior_receipt.get("ordering", {}).get("preregistration_sha256")
        != prior_prereg_binding["preregistration_sha256"]
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "canonical prior lineage is inconsistent"
        )
    prior_source = prior_prereg.get("source")
    if not isinstance(prior_source, Mapping):
        raise MuSiQuePortfolioAcquisitionError(
            "canonical prior source binding is malformed"
        )
    expected = _assemble_preregistration(
        source_repository=prior_source.get("repository"),
        source_license=prior_source.get("license"),
        secret_commitment=secret_commitment,
        prior_preregistration_binding=prior_prereg_binding,
        prior_acquisition_binding=prior_acquisition_binding,
        design_binding=portfolio_design_binding(root),
        retained_p_lineage=prior_study_lineage_binding(root),
        implementation=implementation_binding(root),
    )
    if payload != expected:
        raise MuSiQuePortfolioAcquisitionError(
            "canonical preregistration differs from its complete live closure"
        )
    return payload, raw, custody, secret


def _load_consumption_marker_live(
    *, project: Path
) -> tuple[dict[str, Any], bytes]:
    root = project.resolve(strict=True)
    marker_path = _assert_git_ignored_private_path(
        project=root, path=root / CONSUMPTION_RELATIVE, require_file=True
    )
    if stat.S_IMODE(marker_path.stat().st_mode) & 0o077:
        raise MuSiQuePortfolioAcquisitionError(
            "acquisition consumption marker permissions are too broad"
        )
    marker, raw = _read_json_object(marker_path, "acquisition consumption marker")
    body = dict(marker)
    declared = _require_sha256(
        body.pop("consumption_sha256", None), "consumption marker hash"
    )
    expected_keys = {
        "consumption_sha256",
        "persistence_preflight_complete",
        "portfolio_design_file_sha256",
        "preregistration_file_sha256",
        "preregistration_sha256",
        "prior_acquisition_file_sha256",
        "prior_preregistration_file_sha256",
        "prior_private_block_rows_opened",
        "private_locator_path_hash",
        "private_root_path_hash",
        "public_receipt_path_hash",
        "retry_replay_resample_authorized",
        "schema",
        "selection_secret_commitment_sha256",
        "source_archive_opened_before_consumption",
        "source_archive_sha256",
        "source_member_sha256",
    }
    if (
        set(marker) != expected_keys
        or marker.get("schema") != CONSUMPTION_SCHEMA
        or stable_hash(body) != declared
        or marker.get("persistence_preflight_complete") is not True
        or marker.get("source_archive_opened_before_consumption") is not False
        or marker.get("prior_private_block_rows_opened") != 0
        or marker.get("retry_replay_resample_authorized") is not False
        or any(
            _SHA256_RE.fullmatch(str(marker.get(field))) is None
            for field in (
                "portfolio_design_file_sha256",
                "preregistration_file_sha256",
                "preregistration_sha256",
                "prior_acquisition_file_sha256",
                "prior_preregistration_file_sha256",
                "private_locator_path_hash",
                "private_root_path_hash",
                "public_receipt_path_hash",
                "selection_secret_commitment_sha256",
                "source_archive_sha256",
                "source_member_sha256",
            )
        )
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "acquisition consumption marker contract drifted"
        )
    return marker, raw


def load_acquisition_binding_live(
    *,
    project: Path,
    path: str | Path,
    selection_secret_path: Path,
) -> tuple[dict[str, Any], tuple[BlockCommitment, ...]]:
    """Verify the complete committed acquisition/preregistration/marker chain.

    This is the only loader suitable for formal downstream formation or
    measurement.  It reads no benchmark row and no prior private block.
    """

    root = project.resolve(strict=True)
    canonical = _canonical_public_path(
        project=root,
        supplied=Path(path),
        canonical_relative=ACQUISITION_RELATIVE,
        field="portfolio acquisition receipt",
    )
    receipt, blocks = load_acquisition_binding(canonical)
    receipt_raw = canonical.read_bytes()
    receipt_custody = _committed_binding(
        project=root, path=canonical, field="portfolio acquisition receipt"
    )
    if receipt_custody["file_sha256"] != _sha256_bytes(receipt_raw):
        raise MuSiQuePortfolioAcquisitionError(
            "canonical acquisition file custody drifted"
        )
    prereg, prereg_raw, prereg_custody, secret = _load_preregistration_live(
        project=root, selection_secret_path=selection_secret_path
    )
    marker, marker_raw = _load_consumption_marker_live(project=root)
    secret_commitment = _selection_secret_commitment(secret)
    receipt_selection = receipt["selection_continuity"]
    prereg_selection = prereg["selection"]
    receipt_source = receipt["source"]
    prereg_source = prereg["source"]
    receipt_ordering = receipt["prospective_ordering"]

    continuity_values = {
        secret_commitment,
        receipt_selection["selection_secret_commitment_sha256"],
        receipt["commitments"]["selection_secret_commitment_sha256"],
        prereg_selection["selection_secret_commitment_sha256"],
        receipt["prior_acquisition_binding"][
            "selection_secret_commitment_sha256"
        ],
        receipt["prior_preregistration_binding"][
            "selection_secret_commitment_sha256"
        ],
        marker["selection_secret_commitment_sha256"],
    }
    if (
        receipt["preregistration_sha256"] != prereg["preregistration_sha256"]
        or receipt["preregistration_custody"] != prereg_custody
        or receipt["preregistration_custody"]["file_sha256"]
        != _sha256_bytes(prereg_raw)
        or receipt["prior_preregistration_binding"]
        != prereg["prior_preregistration_binding"]
        or receipt["prior_acquisition_binding"]
        != prereg["prior_acquisition_binding"]
        or receipt["portfolio_design_binding"]
        != prereg["portfolio_design_binding"]
        or receipt["retained_P_lineage"] != prereg["retained_P_lineage"]
        or receipt["implementation"] != prereg["implementation"]
        or len(continuity_values) != 1
        or receipt_selection["selection_domain_separator"]
        != prereg_selection["domain_separator"]
        or receipt_selection["previous_rank_window_start_inclusive"]
        != prereg_selection["previous_rank_window_start_inclusive"]
        or receipt_selection["previous_rank_window_stop_exclusive"]
        != prereg_selection["previous_rank_window_stop_exclusive"]
        or receipt_selection["rank_window_start_inclusive"]
        != prereg_selection["rank_window_start_inclusive"]
        or receipt_selection["rank_window_stop_exclusive"]
        != prereg_selection["rank_window_stop_exclusive"]
        or receipt["counts"]["selected_rows"]
        != prereg_selection["selected_count"]
        or receipt["counts"]["blocks"] != prereg_selection["block_counts"]
        or receipt["counts"]["eligible_rows"]
        != prereg["eligibility"]["expected_eligible_rows"]
        or receipt_source["repository_commit"] != prereg_source["commit"]
        or receipt_source["archive_sha256"]
        != prereg_source["official_archive_sha256"]
        or receipt_source["official_dev_member_sha256"]
        != prereg_source["official_dev_member_sha256"]
        or receipt_source["source_split"] != prereg_source["source_split"]
        or marker["preregistration_sha256"]
        != prereg["preregistration_sha256"]
        or marker["preregistration_file_sha256"] != _sha256_bytes(prereg_raw)
        or marker["source_archive_sha256"]
        != prereg_source["official_archive_sha256"]
        or marker["source_member_sha256"]
        != prereg_source["official_dev_member_sha256"]
        or marker["prior_acquisition_file_sha256"]
        != prereg["prior_acquisition_binding"]["acquisition_file_sha256"]
        or marker["prior_preregistration_file_sha256"]
        != prereg["prior_preregistration_binding"][
            "preregistration_file_sha256"
        ]
        or marker["portfolio_design_file_sha256"]
        != prereg["portfolio_design_binding"]["design_file_sha256"]
        or marker["public_receipt_path_hash"]
        != stable_hash({"absolute_public_receipt": str(canonical)})
        or receipt_ordering["acquisition_consumption_file_sha256"]
        != _sha256_bytes(marker_raw)
        or receipt_ordering["acquisition_consumption_sha256"]
        != marker["consumption_sha256"]
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "canonical acquisition, preregistration, marker, source, or secret continuity drifted"
        )
    return receipt, blocks


def acquire_private_blocks(
    *,
    project: Path,
    preregistration_path: Path,
    official_repository: Path,
    selection_secret_path: Path,
    prior_preregistration_path: Path,
    prior_acquisition_receipt_path: Path,
    source_archive_path: Path,
    private_root: Path,
    private_locator_path: Path,
    public_receipt_path: Path,
) -> dict[str, Any]:
    """Consume the authorization and allocate exact ranks ``[96, 264)``."""

    root = project.resolve(strict=True)
    preregistration = verify_preregistration(
        path=preregistration_path,
        project=root,
        official_repository=official_repository,
        selection_secret_path=selection_secret_path,
        prior_preregistration_path=prior_preregistration_path,
        prior_acquisition_receipt_path=prior_acquisition_receipt_path,
    )
    preregistration_custody = _committed_binding(
        project=root, path=preregistration_path, field="portfolio preregistration"
    )
    secret = _read_selection_secret(project=root, path=selection_secret_path)
    source = _assert_git_ignored_private_path(
        project=root, path=source_archive_path, require_file=True
    )
    pack_root = _assert_git_ignored_private_path(
        project=root, path=private_root, require_file=False
    )
    locator = _assert_git_ignored_private_path(
        project=root, path=private_locator_path, require_file=None
    )
    consumption_path = _assert_git_ignored_private_path(
        project=root,
        path=root / CONSUMPTION_RELATIVE,
        require_file=None,
    )
    public_receipt = public_receipt_path.absolute()
    try:
        public_receipt.relative_to(root)
    except ValueError as exc:
        raise MuSiQuePortfolioAcquisitionError(
            "public receipt must be inside the project"
        ) from exc
    _preflight_persistence(
        pack_root=pack_root,
        locator=locator,
        consumption_path=consumption_path,
        public_receipt_path=public_receipt,
    )

    consumption_body = {
        "schema": CONSUMPTION_SCHEMA,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preregistration_file_sha256": preregistration_custody["file_sha256"],
        "source_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "source_member_sha256": OFFICIAL_DEV_MEMBER_SHA256,
        "selection_secret_commitment_sha256": _selection_secret_commitment(
            secret
        ),
        "prior_acquisition_file_sha256": preregistration[
            "prior_acquisition_binding"
        ]["acquisition_file_sha256"],
        "prior_preregistration_file_sha256": preregistration[
            "prior_preregistration_binding"
        ]["preregistration_file_sha256"],
        "portfolio_design_file_sha256": preregistration[
            "portfolio_design_binding"
        ]["design_file_sha256"],
        "private_root_path_hash": stable_hash(
            {"absolute_private_root": str(pack_root)}
        ),
        "private_locator_path_hash": stable_hash(
            {"absolute_private_locator": str(locator)}
        ),
        "public_receipt_path_hash": stable_hash(
            {"absolute_public_receipt": str(public_receipt)}
        ),
        "persistence_preflight_complete": True,
        "source_archive_opened_before_consumption": False,
        "prior_private_block_rows_opened": 0,
        "retry_replay_resample_authorized": False,
    }
    try:
        _write_json_exclusive(
            consumption_path,
            consumption_body,
            hash_field="consumption_sha256",
            mode=0o600,
        )
    except BaseException:
        if not consumption_path.exists():
            try:
                pack_root.rmdir()
                _fsync_directory(pack_root.parent)
            except OSError:
                pass
        raise
    consumption_raw = consumption_path.read_bytes()

    # The immutable source is first opened only after the durable one-shot marker.
    if _sha256_file(source) != OFFICIAL_ARCHIVE_SHA256:
        raise MuSiQuePortfolioAcquisitionError("official archive hash drifted")
    with zipfile.ZipFile(source) as archive:
        member = prior._find_exact_member(archive)
        source_raw = archive.read(member)
    if _sha256_bytes(source_raw) != OFFICIAL_DEV_MEMBER_SHA256:
        raise MuSiQuePortfolioAcquisitionError(
            "official DEV member hash drifted"
        )
    eligible: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_rows = 0
    for raw_row in _iter_source_rows(source_raw):
        source_rows += 1
        row = _normalize_source_row(raw_row)
        if row is None:
            continue
        item_id = str(row["item_id"])
        if item_id in seen_ids:
            raise MuSiQuePortfolioAcquisitionError(
                "duplicate eligible official item ID"
            )
        seen_ids.add(item_id)
        eligible.append(row)
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS or len(eligible) < RANK_WINDOW_STOP:
        raise MuSiQuePortfolioAcquisitionError("eligible official DEV rows drifted")
    eligible.sort(
        key=lambda row: (
            _selection_key(str(row["item_id"]), secret),
            str(row["item_id"]),
        )
    )
    previous_ids = {
        row["item_id"]
        for row in eligible[
            PREVIOUS_RANK_WINDOW_START:PREVIOUS_RANK_WINDOW_STOP
        ]
    }
    selected = eligible[RANK_WINDOW_START:RANK_WINDOW_STOP]
    if (
        len(selected) != SELECTED_COUNT
        or any(row["item_id"] in previous_ids for row in selected)
    ):
        raise MuSiQuePortfolioAcquisitionError(
            "mechanical continuation disjointness failed"
        )

    block_commitments: list[BlockCommitment] = []
    offset = 0
    for block in BLOCK_ORDER:
        count = BLOCK_COUNTS[block]
        rows = tuple(
            {
                "schema": PRIVATE_ROW_SCHEMA,
                "block": block,
                **row,
            }
            for row in selected[offset : offset + count]
        )
        offset += count
        if len(rows) != count or any(set(row) != PRIVATE_BLOCK_ROW_KEYS for row in rows):
            raise MuSiQuePortfolioAcquisitionError("private row schema drifted")
        file_hash, item_set_hash = _write_jsonl_exclusive(
            pack_root / f"{block}.jsonl", rows
        )
        block_commitments.append(
            BlockCommitment(
                block=block,
                count=count,
                file_sha256=file_hash,
                item_commitment_set_sha256=item_set_hash,
            )
        )
    if offset != SELECTED_COUNT:
        raise MuSiQuePortfolioAcquisitionError("block allocation drifted")

    locator_body = {
        "schema": PRIVATE_LOCATOR_SCHEMA,
        "private_root": str(pack_root),
        "blocks": [
            {**row.to_dict(), "relative_file": f"{row.block}.jsonl"}
            for row in block_commitments
        ],
        "private_pack_sha256": stable_hash(
            [row.to_dict() for row in block_commitments]
        ),
        "selection_secret_included": False,
    }
    _write_json_exclusive(
        locator, locator_body, hash_field="locator_sha256", mode=0o600
    )
    locator_raw = locator.read_bytes()
    receipt: dict[str, Any] = {
        "schema": ACQUISITION_SCHEMA,
        "decision": "fresh_residual_six_block_pack_formed_no_measurement_authority",
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preregistration_custody": preregistration_custody,
        "source": {
            "repository_commit": OFFICIAL_SOURCE_COMMIT,
            "dataset": "MuSiQue-Answerable v1.0",
            "source_split": "official_dev",
            "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "official_dev_member_sha256": OFFICIAL_DEV_MEMBER_SHA256,
        },
        "counts": {
            "source_rows": source_rows,
            "eligible_rows": len(eligible),
            "rank_window_start_inclusive": RANK_WINDOW_START,
            "rank_window_stop_exclusive": RANK_WINDOW_STOP,
            "selected_rows": len(selected),
            "selected_previous_rank_window_overlap": sum(
                row["item_id"] in previous_ids for row in selected
            ),
            "blocks": dict(BLOCK_COUNTS),
        },
        "prior_preregistration_binding": preregistration[
            "prior_preregistration_binding"
        ],
        "prior_acquisition_binding": preregistration[
            "prior_acquisition_binding"
        ],
        "portfolio_design_binding": preregistration["portfolio_design_binding"],
        "retained_P_lineage": preregistration["retained_P_lineage"],
        "selection_continuity": {
            "selection_domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "same_selection_secret_as_prior": True,
            "previous_rank_window_start_inclusive": PREVIOUS_RANK_WINDOW_START,
            "previous_rank_window_stop_exclusive": PREVIOUS_RANK_WINDOW_STOP,
            "rank_window_start_inclusive": RANK_WINDOW_START,
            "rank_window_stop_exclusive": RANK_WINDOW_STOP,
            "rank_window_disjoint_from_prior_by_construction": True,
            "prior_private_block_rows_opened": 0,
            "prior_public_outcomes_used_for_rank_selection": False,
        },
        "implementation": preregistration["implementation"],
        "prospective_ordering": {
            "preregistration_committed_before_consumption": True,
            "persistence_preflight_complete_before_consumption": True,
            "pack_root_created_before_consumption": True,
            "consumption_persisted_before_source_archive_open": True,
            "source_rows_opened_before_consumption": 0,
            "acquisition_consumption_file_sha256": _sha256_bytes(
                consumption_raw
            ),
            "acquisition_consumption_sha256": json.loads(consumption_raw)[
                "consumption_sha256"
            ],
            "retry_replay_resample_authorized": False,
        },
        "commitments": {
            "block_files": [row.to_dict() for row in block_commitments],
            "private_pack_sha256": stable_hash(
                [row.to_dict() for row in block_commitments]
            ),
            "private_locator_file_sha256": _sha256_bytes(locator_raw),
            "private_row_key_set_sha256": stable_hash(
                sorted(PRIVATE_BLOCK_ROW_KEYS)
            ),
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "safety": {
            "formation_executed": False,
            "measurement_executed": False,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "scores_computed": 0,
            "prior_private_block_rows_read": 0,
        },
    }
    _assert_public_safe(receipt)
    return receipt


__all__ = [
    "ACQUISITION_SCHEMA",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "BlockCommitment",
    "MuSiQuePortfolioAcquisitionError",
    "PREREGISTRATION_SCHEMA",
    "PRIVATE_BLOCK_ROW_KEYS",
    "PRIVATE_LOCATOR_SCHEMA",
    "PREVIOUS_RANK_WINDOW_STOP",
    "RANK_WINDOW_START",
    "RANK_WINDOW_STOP",
    "SELECTED_COUNT",
    "SELECTION_DOMAIN_SEPARATOR",
    "acquire_private_blocks",
    "build_preregistration",
    "implementation_binding",
    "load_acquisition_binding",
    "load_acquisition_binding_live",
    "load_private_block",
    "portfolio_design_binding",
    "prior_study_lineage_binding",
    "verify_preregistration",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    acquire = subparsers.add_parser("acquire")
    for command in (preregister, acquire):
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--official-repository", type=Path, required=True)
        command.add_argument("--selection-secret", type=Path, required=True)
        command.add_argument("--prior-preregistration", type=Path, required=True)
        command.add_argument("--prior-acquisition-receipt", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--source-archive", type=Path, required=True)
    acquire.add_argument("--private-root", type=Path, required=True)
    acquire.add_argument("--private-locator", type=Path, required=True)
    arguments = parser.parse_args(argv)

    root = arguments.project.resolve(strict=True)
    expected_output = root / (
        PREREGISTRATION_RELATIVE
        if arguments.command == "preregister"
        else ACQUISITION_RELATIVE
    )
    if arguments.output.resolve(strict=False) != expected_output.resolve(strict=False):
        raise MuSiQuePortfolioAcquisitionError(
            "production CLI output must use the fixed canonical manifest path"
        )
    if arguments.command == "acquire" and arguments.preregistration.resolve(
        strict=True
    ) != (root / PREREGISTRATION_RELATIVE).resolve(strict=True):
        raise MuSiQuePortfolioAcquisitionError(
            "production CLI must use the fixed canonical preregistration"
        )
    if arguments.output.exists():
        raise FileExistsError("public portfolio-acquisition output already exists")
    common = {
        "project": arguments.project,
        "official_repository": arguments.official_repository,
        "selection_secret_path": arguments.selection_secret,
        "prior_preregistration_path": arguments.prior_preregistration,
        "prior_acquisition_receipt_path": arguments.prior_acquisition_receipt,
    }
    if arguments.command == "preregister":
        payload = build_preregistration(**common)
        _write_json_exclusive(
            arguments.output,
            payload,
            hash_field="preregistration_sha256",
            mode=0o644,
        )
        return 0
    payload = acquire_private_blocks(
        **common,
        preregistration_path=arguments.preregistration,
        source_archive_path=arguments.source_archive,
        private_root=arguments.private_root,
        private_locator_path=arguments.private_locator,
        public_receipt_path=arguments.output,
    )
    _write_json_exclusive(
        arguments.output,
        payload,
        hash_field="acquisition_sha256",
        mode=0o644,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

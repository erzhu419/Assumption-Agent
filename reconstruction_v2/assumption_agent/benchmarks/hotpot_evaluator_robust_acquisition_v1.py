"""Prospective acquisition for the final Hotpot evaluator-coevolution study.

This protocol reuses the already-fixed recursive-study selection secret and
the exact recursive-study HMAC ordering.  After excluding the original twelve
family-out items, the earlier study occupied ranks ``[0, 156)``; this study
atomically allocates ranks ``[156, 324)``.  Consequently the new blocks are
disjoint by construction without opening any earlier private block, including
the earlier ``M_search`` block.

Preregistration reads only committed public artifacts, the private selection
secret, and live implementation/lineage bytes.  Acquisition persists a
one-shot marker before opening either the original twelve-item private pack or
the immutable source parquet.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from . import hotpot_family_out_acquisition_v1 as original
from . import hotpot_recursive_acquisition_v1 as v2
from .musique_official_core_comparison_v1 import (
    _assert_git_ignored_private_path,
    _read_selection_secret,
    _selection_secret_commitment,
)


VERSION = "hotpot_evaluator_robust_acquisition_v1"
PREREGISTRATION_SCHEMA = f"{VERSION}_preregistration"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition"
PRIVATE_LOCATOR_SCHEMA = f"{VERSION}_private_locator"
ACQUISITION_CONSUMPTION_SCHEMA = f"{VERSION}_acquisition_consumption"

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

# The first study used the exact top 156 ranks.  These constants are semantic
# protocol constants rather than values inferred from any private old block.
PREVIOUS_RANK_WINDOW_START = 0
PREVIOUS_RANK_WINDOW_STOP = v2.SELECTED_COUNT
RANK_WINDOW_START = v2.SELECTED_COUNT
RANK_WINDOW_STOP = RANK_WINDOW_START + SELECTED_COUNT
SELECTION_DOMAIN_SEPARATOR = v2.VERSION

PRIVATE_BLOCK_ROW_KEYS = v2.PRIVATE_BLOCK_ROW_KEYS
TOP_K = v2.TOP_K
PRIOR_SAMPLE_COUNT = v2.PRIOR_SAMPLE_COUNT

SOURCE_URL = v2.SOURCE_URL
HF_REPOSITORY = v2.HF_REPOSITORY
HF_REPOSITORY_COMMIT = v2.HF_REPOSITORY_COMMIT
SOURCE_SHA256 = v2.SOURCE_SHA256
SOURCE_SIZE = v2.SOURCE_SIZE
SOURCE_ROW_COUNT = v2.SOURCE_ROW_COUNT
EXPECTED_SOURCE_FIELDS = v2.EXPECTED_SOURCE_FIELDS

V2_FINAL_DISPOSITION_SCHEMA = "hotpot_recursive_study_v1_final_disposition"
V2_ACQUISITION_RELATIVE = (
    "manifests/hotpot_recursive_study_v1_acquisition.json"
)
V2_ACQUISITION_FILE_SHA256 = (
    "2d907a7214d547c7dcae99aba8e38a260cf4e21aeed5d33ff74fd220fc2e4dd6"
)
V2_ACQUISITION_SHA256 = (
    "ebd6e89fc73a8232ccb718a4def55c2c1a896330a9fa154211066680b23acce4"
)
V2_FINAL_DISPOSITION_RELATIVE = (
    "manifests/hotpot_recursive_study_v1_final_disposition.json"
)
V2_FINAL_DISPOSITION_FILE_SHA256 = (
    "631c80917688fd38762579b7bf9f65546c70d213a46e9edd95a05b56610a2949"
)
V2_FINAL_DISPOSITION_SHA256 = (
    "487831a0ec75d796e7c1a28e22f498fb7b65151c5b164b577bcbb6b960941aef"
)
V2_SELECTION_IMPLEMENTATION_RELATIVE = (
    "assumption_agent/benchmarks/hotpot_recursive_acquisition_v1.py"
)
PORTFOLIO_DESIGN_RELATIVE = (
    "manifests/hotpot_evaluator_portfolio_design_v1.json"
)
PORTFOLIO_DESIGN_SCHEMA = "hotpot_evaluator_portfolio_design_v1"
PORTFOLIO_DESIGN_SHA256 = (
    "3ed3811d8c14856c2586c14307318b9e4203cc70b26a9eae86e8f5091917d37d"
)
PORTFOLIO_DESIGN_FILE_SHA256 = (
    "1cb6d552c077e940320a2b6432ae2c700b024c2d2051a5bb2f9ccd23329aba43"
)
ACQUISITION_CONSUMPTION_RELATIVE = (
    "artifacts/hotpot_evaluator_robust_acquisition_v1/authorization.consumed.json"
)

# Preregistration deliberately cannot be formed until the portfolio evaluator
# exists; its exact bytes are part of the prospective implementation closure.
IMPLEMENTATION_RELATIVE_FILES = tuple(
    dict.fromkeys(
        (
            *v2.IMPLEMENTATION_RELATIVE_FILES,
            "assumption_agent/benchmarks/hotpot_evaluator_robust_acquisition_v1.py",
            "assumption_agent/benchmarks/hotpot_evaluator_portfolio_coevolution_v1.py",
        )
    )
)
RETAINED_P_LINEAGE_RELATIVE_FILES = v2.RETAINED_P_LINEAGE_RELATIVE_FILES

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ORIGINAL_ROW_KEYS = v2._PRIOR_ROW_KEYS


class HotpotEvaluatorRobustAcquisitionError(RuntimeError):
    """The fixed source, public lineage, rank window, or custody drifted."""


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    count: int
    file_sha256: str
    item_commitment_set_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HotpotEvaluatorRobustAcquisitionError("required file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HotpotEvaluatorRobustAcquisitionError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _read_json_object(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HotpotEvaluatorRobustAcquisitionError(f"{field_name} is unavailable")
    raw = candidate.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotEvaluatorRobustAcquisitionError(f"{field_name} is invalid") from exc
    if not isinstance(payload, dict):
        raise HotpotEvaluatorRobustAcquisitionError(
            f"{field_name} must be one object"
        )
    return payload, raw


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"context"',
        '"corpus"',
        '"item_id"',
        '"paragraph_text"',
        '"private_pack_path"',
        '"private_root"',
        '"question"',
        '"selection_secret_path"',
        '"support_indices"',
    )
    if any(token in serialized for token in forbidden):
        raise HotpotEvaluatorRobustAcquisitionError(
            "public artifact contains private content or private paths"
        )


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
    raw = json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_exclusive(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    digest = hashlib.sha256()
    commitments: list[str] = []
    with os.fdopen(descriptor, "wb") as handle:
        for row in rows:
            raw = _canonical_bytes(row) + b"\n"
            handle.write(raw)
            digest.update(raw)
            commitments.append(stable_hash(row))
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest(), stable_hash(commitments)


def implementation_binding(project: Path) -> dict[str, Any]:
    """Bind every implementation file only if it is the clean HEAD blob."""

    root = project.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotEvaluatorRobustAcquisitionError(
                f"implementation file missing or symlinked: {relative}"
            )
        live_sha256 = _sha256_file(path)
        custody = _committed_public_binding(
            project=root, path=path, field_name=f"implementation file {relative}"
        )
        if (
            custody["file_sha256"] != live_sha256
            or custody["head_blob_sha256"] != live_sha256
        ):
            raise HotpotEvaluatorRobustAcquisitionError(
                f"implementation file HEAD binding drifted: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": live_sha256,
                "head_blob_sha256": live_sha256,
                "clean_tracked_HEAD_blob": True,
            }
        )
    return {"files": rows, "set_sha256": stable_hash(rows)}


def retained_p_lineage_binding(project: Path) -> dict[str, Any]:
    """Bind every retained-P artifact only if it is the clean HEAD blob."""

    root = project.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for role, relative in RETAINED_P_LINEAGE_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotEvaluatorRobustAcquisitionError(
                f"retained-P lineage file missing or symlinked: {relative}"
            )
        live_sha256 = _sha256_file(path)
        custody = _committed_public_binding(
            project=root, path=path, field_name=f"retained-P file {relative}"
        )
        if (
            custody["file_sha256"] != live_sha256
            or custody["head_blob_sha256"] != live_sha256
        ):
            raise HotpotEvaluatorRobustAcquisitionError(
                f"retained-P file HEAD binding drifted: {relative}"
            )
        rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": live_sha256,
                "head_blob_sha256": live_sha256,
                "clean_tracked_HEAD_blob": True,
            }
        )
    return {
        "files": rows,
        "set_sha256": stable_hash(rows),
        "fixed_before_fresh_block_selection": True,
    }


def _committed_public_binding(
    *, project: Path, path: Path, field_name: str
) -> dict[str, Any]:
    try:
        receipt = original.committed_public_file_receipt(project=project, path=path)
    except Exception as exc:
        raise HotpotEvaluatorRobustAcquisitionError(
            f"{field_name} must be a clean tracked HEAD blob"
        ) from exc
    # Binding the current repository tip would make this change when the new
    # preregistration itself advances HEAD.  The equal tracked/live blob hashes
    # are the stable committed-file custody fact required by this protocol.
    return {
        "file_sha256": receipt["preregistration_file_sha256"],
        "head_blob_sha256": receipt["preregistration_head_blob_sha256"],
        "clean_tracked_HEAD_blob": True,
    }


def _canonical_public_artifact(
    *,
    project: Path,
    supplied_path: Path,
    canonical_relative: str,
    field_name: str,
) -> Path:
    """Reject even committed substitutes for a named historical artifact."""

    root = project.resolve(strict=True)
    candidate = supplied_path if supplied_path.is_absolute() else root / supplied_path
    candidate = candidate.absolute()
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise HotpotEvaluatorRobustAcquisitionError(
                f"{field_name} may not traverse a symlink"
            )
    expected = root / canonical_relative
    try:
        candidate_resolved = candidate.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HotpotEvaluatorRobustAcquisitionError(
            f"canonical {field_name} is unavailable"
        ) from exc
    if candidate_resolved != expected_resolved:
        raise HotpotEvaluatorRobustAcquisitionError(
            f"{field_name} must use the fixed canonical path"
        )
    return expected_resolved


def _legacy_retained_p_binding(
    retained_p: Mapping[str, Any],
) -> dict[str, Any]:
    rows = retained_p.get("files")
    if not isinstance(rows, list):
        raise HotpotEvaluatorRobustAcquisitionError(
            "retained-P clean closure is malformed"
        )
    legacy_rows = [
        {
            "role": row["role"],
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in rows
        if isinstance(row, Mapping)
        and set(row)
        == {
            "role",
            "path",
            "sha256",
            "head_blob_sha256",
            "clean_tracked_HEAD_blob",
        }
    ]
    if len(legacy_rows) != len(rows):
        raise HotpotEvaluatorRobustAcquisitionError(
            "retained-P clean closure row drifted"
        )
    return {
        "files": legacy_rows,
        "set_sha256": stable_hash(legacy_rows),
        "fixed_before_fresh_block_selection": True,
    }


def portfolio_design_binding(project: Path) -> dict[str, Any]:
    """Validate and bind the committed final portfolio design."""

    root = project.resolve(strict=True)
    path = root / PORTFOLIO_DESIGN_RELATIVE
    payload, raw = _read_json_object(path, "portfolio evaluator design")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("design_sha256", None), "portfolio design hash"
    )
    expected_keys = {
        "claim_boundary",
        "cohort_contract",
        "design_evidence",
        "design_sha256",
        "execution_contract",
        "mechanism",
        "promotion_contract",
        "raw_content_persisted",
        "schema",
        "status",
        "terminal_policy",
    }
    file_sha256 = _sha256_bytes(raw)
    if (
        set(payload) != expected_keys
        or payload.get("schema") != PORTFOLIO_DESIGN_SCHEMA
        or stable_hash(body) != declared
        or declared != PORTFOLIO_DESIGN_SHA256
        or file_sha256 != PORTFOLIO_DESIGN_FILE_SHA256
        or payload.get("status")
        != "single_final_same_source_confirmatory_mechanism_fixed_before_new_cohort"
        or payload.get("raw_content_persisted") is not False
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "committed portfolio evaluator design drifted"
        )
    custody = _committed_public_binding(
        project=root, path=path, field_name="portfolio evaluator design"
    )
    if (
        custody.get("file_sha256") != file_sha256
        or custody.get("head_blob_sha256") != file_sha256
        or custody.get("clean_tracked_HEAD_blob") is not True
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "portfolio design committed custody drifted"
        )
    return {
        "relative_path": PORTFOLIO_DESIGN_RELATIVE,
        "schema": PORTFOLIO_DESIGN_SCHEMA,
        "design_file_sha256": file_sha256,
        "design_sha256": declared,
        "committed_custody": custody,
    }


def _load_v2_public_binding(
    *,
    project: Path,
    path: Path,
    secret: bytes,
    retained_p: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    canonical = _canonical_public_artifact(
        project=project,
        supplied_path=path,
        canonical_relative=V2_ACQUISITION_RELATIVE,
        field_name="v2 acquisition receipt",
    )
    raw = canonical.read_bytes()
    if _sha256_bytes(raw) != V2_ACQUISITION_FILE_SHA256:
        raise HotpotEvaluatorRobustAcquisitionError(
            "fixed canonical v2 acquisition file hash drifted"
        )
    try:
        receipt, blocks = v2.load_acquisition_binding(canonical)
    except Exception as exc:
        raise HotpotEvaluatorRobustAcquisitionError(
            "fixed v2 acquisition receipt drifted"
        ) from exc
    custody = _committed_public_binding(
        project=project, path=canonical, field_name="v2 acquisition receipt"
    )
    commitments = receipt.get("commitments")
    prior_exclusion = receipt.get("prior_exclusion")
    implementation = receipt.get("implementation")
    live_v2_implementation = v2.implementation_binding(project)
    legacy_retained_p = _legacy_retained_p_binding(retained_p)
    implementation_files = (
        implementation.get("files")
        if isinstance(implementation, Mapping)
        else None
    )
    selection_implementation_rows = [
        row
        for row in implementation_files or ()
        if isinstance(row, Mapping)
        and row.get("path") == V2_SELECTION_IMPLEMENTATION_RELATIVE
    ]
    if (
        len(blocks) != len(v2.BLOCK_ORDER)
        or sum(row.count for row in blocks) != PREVIOUS_RANK_WINDOW_STOP
        or receipt.get("acquisition_sha256") != V2_ACQUISITION_SHA256
        or receipt.get("counts", {}).get("selected_rows")
        != PREVIOUS_RANK_WINDOW_STOP
        or implementation != live_v2_implementation
        or len(selection_implementation_rows) != 1
        or selection_implementation_rows[0].get("sha256")
        != _sha256_file(project / V2_SELECTION_IMPLEMENTATION_RELATIVE)
        or not isinstance(commitments, Mapping)
        or commitments.get("selection_secret_commitment_sha256")
        != _selection_secret_commitment(secret)
        or receipt.get("retained_P_lineage") != legacy_retained_p
        or not isinstance(prior_exclusion, Mapping)
        or prior_exclusion.get("excluded_prior_item_count") != PRIOR_SAMPLE_COUNT
        or prior_exclusion.get("selected_prior_item_overlap_count") != 0
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "v2 ordering, secret, exclusion, or retained-P binding drifted"
        )
    binding = {
        "schema": receipt["schema"],
        "relative_path": V2_ACQUISITION_RELATIVE,
        "acquisition_file_sha256": _sha256_bytes(raw),
        "acquisition_sha256": receipt["acquisition_sha256"],
        "committed_custody": custody,
        "selected_rank_window_start_inclusive": PREVIOUS_RANK_WINDOW_START,
        "selected_rank_window_stop_exclusive": PREVIOUS_RANK_WINDOW_STOP,
        "selected_count": PREVIOUS_RANK_WINDOW_STOP,
        "selection_secret_commitment_sha256": commitments[
            "selection_secret_commitment_sha256"
        ],
        "selection_domain_separator": SELECTION_DOMAIN_SEPARATOR,
        "private_block_files_opened": 0,
        "private_M_search_content_opened": False,
        "private_M_search_outcome_opened": False,
    }
    return receipt, raw, binding


def _load_final_disposition_binding(
    *, project: Path, path: Path, v2_acquisition_file_sha256: str
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    canonical = _canonical_public_artifact(
        project=project,
        supplied_path=path,
        canonical_relative=V2_FINAL_DISPOSITION_RELATIVE,
        field_name="v2 final disposition",
    )
    payload, raw = _read_json_object(canonical, "v2 final disposition")
    if _sha256_bytes(raw) != V2_FINAL_DISPOSITION_FILE_SHA256:
        raise HotpotEvaluatorRobustAcquisitionError(
            "fixed canonical v2 final disposition file hash drifted"
        )
    body = dict(payload)
    declared = _require_sha256(
        body.pop("disposition_sha256", None), "v2 final disposition hash"
    )
    bindings = payload.get("bindings")
    l5 = payload.get("L5")
    terminal = payload.get("terminal_policy")
    expected_keys = {
        "L4",
        "L5",
        "bindings",
        "disposition_sha256",
        "external_network_calls",
        "online_evaluator_calls",
        "raw_content_persisted",
        "schema",
        "status",
        "terminal_policy",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != V2_FINAL_DISPOSITION_SCHEMA
        or stable_hash(body) != declared
        or declared != V2_FINAL_DISPOSITION_SHA256
        or payload.get("status") != "L4_narrow_positive_L5_no_promotion_terminal"
        or payload.get("external_network_calls") != 0
        or payload.get("online_evaluator_calls") != 0
        or payload.get("raw_content_persisted") is not False
        or not isinstance(bindings, Mapping)
        or bindings.get("acquisition_file_sha256")
        != v2_acquisition_file_sha256
        or not isinstance(l5, Mapping)
        or l5.get("M_search_authorized") is not False
        or l5.get("M_search_opened") is not False
        or l5.get("challenger_promoted") is not False
        or l5.get("evaluator_coevolution_achieved") is not False
        or not isinstance(terminal, Mapping)
        or terminal.get("future_L5_requires_new_mechanism_and_new_cohort")
        is not True
        or terminal.get("same_anchor_retry_replay_resample") is not False
        or terminal.get("same_anchor_challenger_substitution") is not False
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "v2 final disposition is not the fixed terminal public record"
        )
    custody = _committed_public_binding(
        project=project, path=canonical, field_name="v2 final disposition"
    )
    binding = {
        "schema": payload["schema"],
        "relative_path": V2_FINAL_DISPOSITION_RELATIVE,
        "disposition_file_sha256": _sha256_bytes(raw),
        "disposition_sha256": declared,
        "committed_custody": custody,
        "previous_M_search_authorized": False,
        "previous_M_search_opened": False,
        "future_study_requires_new_mechanism_and_new_cohort": True,
        "public_aggregate_outcomes_used_for_rank_selection": False,
    }
    return payload, raw, binding


def _selection_key(item_id: str, secret: bytes) -> str:
    """Return the exact v2 HMAC key; do not introduce a new domain."""

    try:
        return v2._selection_key(item_id, secret)
    except Exception as exc:
        raise HotpotEvaluatorRobustAcquisitionError(
            "fixed v2 selection-key implementation rejected the secret"
        ) from exc


def build_preregistration(
    *,
    project: Path,
    selection_secret_path: Path,
    v2_acquisition_receipt_path: Path,
    v2_final_disposition_path: Path,
) -> dict[str, Any]:
    """Build the zero-row/private-block-access preregistration."""

    root = project.resolve(strict=True)
    secret = _read_selection_secret(project=root, path=selection_secret_path)
    retained_p = retained_p_lineage_binding(root)
    portfolio_design = portfolio_design_binding(root)
    v2_receipt, _v2_raw, v2_binding = _load_v2_public_binding(
        project=root,
        path=v2_acquisition_receipt_path,
        secret=secret,
        retained_p=retained_p,
    )
    _disposition, _disposition_raw, disposition_binding = (
        _load_final_disposition_binding(
            project=root,
            path=v2_final_disposition_path,
            v2_acquisition_file_sha256=v2_binding["acquisition_file_sha256"],
        )
    )
    original_exclusion = v2_receipt["prior_exclusion"]
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "decision": "acquisition_only_no_formation_measurement_or_scoring_authority",
        "source": {
            "canonical_role": "fixed_HF_HotpotQA_distractor_validation_parquet",
            "url": SOURCE_URL,
            "hf_repository": HF_REPOSITORY,
            "hf_repository_commit": HF_REPOSITORY_COMMIT,
            "file_sha256": SOURCE_SHA256,
            "file_size": SOURCE_SIZE,
            "row_count": SOURCE_ROW_COUNT,
            "expected_source_field_order_sha256": stable_hash(
                list(EXPECTED_SOURCE_FIELDS)
            ),
            "label_provenance": (
                "source_provided_supporting_facts_from_fixed_HF_parquet"
            ),
            "original_CMU_JSON_byte_or_row_equivalence_claim": False,
            "original_CMU_JSON_label_equivalence_claim": False,
        },
        "eligibility": {
            "normalizer": "hotpot_family_out_acquisition_v1._normalize_source_row",
            "nonempty_globally_unique_source_id": True,
            "question_nonempty": True,
            "minimum_unique_nonempty_context_titles": TOP_K,
            "duplicate_context_titles_eligible": False,
            "context_sentences_nonempty": True,
            "support_title_and_sentence_index_valid_in_source": True,
            "exact_unique_source_provided_support_title_count": 2,
            "answer_type_level_text_or_score_filtering": False,
        },
        "original_twelve_exclusion": {
            "prior_acquisition_file_sha256": original_exclusion[
                "prior_acquisition_file_sha256"
            ],
            "prior_acquisition_sha256": original_exclusion[
                "prior_acquisition_sha256"
            ],
            "prior_private_pack_file_sha256": original_exclusion[
                "prior_private_pack_file_sha256"
            ],
            "prior_item_commitment_set_sha256": original_exclusion[
                "prior_item_commitment_set_sha256"
            ],
            "exact_item_count": PRIOR_SAMPLE_COUNT,
            "private_pack_opened_during_preregistration": False,
            "all_exact_original_ids_excluded_before_HMAC_ranking": True,
            "outcomes_used_for_selection": False,
        },
        "v2_public_binding": v2_binding,
        "v2_final_disposition_binding": disposition_binding,
        "portfolio_design_binding": portfolio_design,
        "selection": {
            "algorithm": (
                "exclude_exact_original_twelve_then_exact_v2_ascending_"
                "HMAC_SHA256_order_take_rank_window_v1"
            ),
            "domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "selection_secret_reused_from_v2": True,
            "selection_secret_persisted_publicly": False,
            "previous_rank_window_start_inclusive": PREVIOUS_RANK_WINDOW_START,
            "previous_rank_window_stop_exclusive": PREVIOUS_RANK_WINDOW_STOP,
            "rank_window_start_inclusive": RANK_WINDOW_START,
            "rank_window_stop_exclusive": RANK_WINDOW_STOP,
            "rank_window_disjoint_from_v2_by_construction": True,
            "selected_count": SELECTED_COUNT,
            "block_order": list(BLOCK_ORDER),
            "block_counts": dict(BLOCK_COUNTS),
            "replacement": False,
            "manual_or_outcome_conditioned_selection": False,
        },
        "access_contract": {
            "all_six_blocks_formed_together": True,
            "committed_preregistration_precedes_marker": True,
            "one_shot_marker_precedes_original_twelve_pack_open": True,
            "one_shot_marker_precedes_source_open": True,
            "v2_private_pack_path_parameter_accepted": False,
            "v2_private_block_files_opened": 0,
            "previous_M_search_content_opened": False,
            "previous_M_search_outcome_opened": False,
            "measurement_requires_separate_pre_run_freeze": True,
            "retry_replay_resample": 0,
        },
        "study_contract": {
            "evaluator_formation_partitions": ["A_form_0", "A_form_1"],
            "search_formation_partitions": ["F_search_0", "F_search_1"],
            "evaluator_anchor_partition": "A_hold",
            "search_measurement_partition": "M_search",
            "primary_metric": "source_provided_support_recall_at_5",
            "offline_evaluation_only": True,
            "online_evaluator_calls": 0,
            "study_level_answer_generator_calls": 0,
        },
        "retained_P_lineage": retained_p,
        "implementation": implementation_binding(root),
        "acquisition_runtime": original.acquisition_runtime_binding(),
        "claim_boundary": {
            "fixed_HF_conversion_not_original_CMU_JSON": True,
            "source_provided_label_claim_only": True,
            "answer_generation_claim": False,
            "performance_claim_before_measurement": False,
        },
        "safety": {
            "source_rows_read": 0,
            "original_twelve_private_pack_rows_read": 0,
            "v2_private_block_rows_read": 0,
            "previous_M_search_private_rows_read": 0,
            "model_calls": 0,
            "network_calls": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
        },
    }
    _assert_public_safe(payload)
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def verify_preregistration(
    *,
    path: Path,
    project: Path,
    selection_secret_path: Path,
    v2_acquisition_receipt_path: Path,
    v2_final_disposition_path: Path,
) -> dict[str, Any]:
    payload, _raw = _read_json_object(path, "robust-evaluator preregistration")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("preregistration_sha256", None), "preregistration hash"
    )
    if payload.get("schema") != PREREGISTRATION_SCHEMA or stable_hash(body) != declared:
        raise HotpotEvaluatorRobustAcquisitionError(
            "preregistration self-hash drifted"
        )
    expected = build_preregistration(
        project=project,
        selection_secret_path=selection_secret_path,
        v2_acquisition_receipt_path=v2_acquisition_receipt_path,
        v2_final_disposition_path=v2_final_disposition_path,
    )
    if payload != expected:
        raise HotpotEvaluatorRobustAcquisitionError(
            "preregistration differs from the complete live protocol"
        )
    return payload


def _read_original_private_ids_after_marker(
    *,
    project: Path,
    original_private_pack_path: Path,
    original_binding: Mapping[str, Any],
    consumption_path: Path,
) -> frozenset[str]:
    if not consumption_path.is_file():
        raise HotpotEvaluatorRobustAcquisitionError(
            "one-shot marker must exist before original private pack open"
        )
    pack = _assert_git_ignored_private_path(
        project=project, path=original_private_pack_path, require_file=True
    )
    raw = pack.read_bytes()
    if (
        _sha256_bytes(raw) != original_binding.get("prior_private_pack_file_sha256")
        or not raw
        or not raw.endswith(b"\n")
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "exact original twelve-item private pack drifted"
        )
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            if not line:
                raise HotpotEvaluatorRobustAcquisitionError(
                    "original private pack contains a blank row"
                )
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise HotpotEvaluatorRobustAcquisitionError(
                    "original private pack row is malformed"
                )
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotEvaluatorRobustAcquisitionError(
            "original private pack JSONL is invalid"
        ) from exc
    if len(rows) != PRIOR_SAMPLE_COUNT:
        raise HotpotEvaluatorRobustAcquisitionError(
            "original private pack count drifted"
        )
    if b"".join(_canonical_bytes(row) + b"\n" for row in rows) != raw:
        raise HotpotEvaluatorRobustAcquisitionError(
            "original private pack is not canonical JSONL"
        )
    if stable_hash([stable_hash(row) for row in rows]) != original_binding.get(
        "prior_item_commitment_set_sha256"
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "original private pack item commitments drifted"
        )
    item_ids: list[str] = []
    for row in rows:
        if (
            set(row) != _ORIGINAL_ROW_KEYS
            or row.get("schema") != original.PRIVATE_ROW_SCHEMA
            or not isinstance(row.get("item_id"), str)
            or not row["item_id"]
        ):
            raise HotpotEvaluatorRobustAcquisitionError(
                "original private pack row schema drifted"
            )
        item_ids.append(row["item_id"])
    if len(set(item_ids)) != PRIOR_SAMPLE_COUNT:
        raise HotpotEvaluatorRobustAcquisitionError(
            "original private pack IDs are not unique"
        )
    return frozenset(item_ids)


def _normalized_source_row(raw: object) -> dict[str, Any] | None:
    return v2._normalized_source_row(raw)


def load_private_block(
    path: str | Path,
    *,
    commitment: BlockCommitment,
    expected_block: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Strictly load one new private block against its public commitment."""

    block = commitment.block if expected_block is None else expected_block
    if block != commitment.block or block not in BLOCK_ORDER:
        raise HotpotEvaluatorRobustAcquisitionError("private block identity drifted")
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HotpotEvaluatorRobustAcquisitionError("private block is unavailable")
    raw = candidate.read_bytes()
    if _sha256_bytes(raw) != commitment.file_sha256 or not raw.endswith(b"\n"):
        raise HotpotEvaluatorRobustAcquisitionError("private block file hash drifted")
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise HotpotEvaluatorRobustAcquisitionError(
                    "private block row is malformed"
                )
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotEvaluatorRobustAcquisitionError(
            "private block JSONL is invalid"
        ) from exc
    if (
        len(rows) != commitment.count
        or commitment.count != BLOCK_COUNTS[block]
        or any(set(row) != PRIVATE_BLOCK_ROW_KEYS for row in rows)
        or any(row.get("block") != block for row in rows)
        or b"".join(_canonical_bytes(row) + b"\n" for row in rows) != raw
        or stable_hash([stable_hash(row) for row in rows])
        != commitment.item_commitment_set_sha256
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "private block schema, count, or commitment drifted"
        )
    item_ids = [row["item_id"] for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise HotpotEvaluatorRobustAcquisitionError(
            "private block IDs are not unique"
        )
    return tuple(rows)


def _clean_closure_is_structurally_valid(
    binding: object, *, retained: bool
) -> bool:
    if not isinstance(binding, Mapping):
        return False
    expected_binding_keys = (
        {"files", "set_sha256", "fixed_before_fresh_block_selection"}
        if retained
        else {"files", "set_sha256"}
    )
    rows = binding.get("files")
    if (
        set(binding) != expected_binding_keys
        or not isinstance(rows, list)
        or (retained and binding.get("fixed_before_fresh_block_selection") is not True)
        or binding.get("set_sha256") != stable_hash(rows)
    ):
        return False
    expected_row_keys = (
        {
            "role",
            "path",
            "sha256",
            "head_blob_sha256",
            "clean_tracked_HEAD_blob",
        }
        if retained
        else {
            "path",
            "sha256",
            "head_blob_sha256",
            "clean_tracked_HEAD_blob",
        }
    )
    return all(
        isinstance(row, Mapping)
        and set(row) == expected_row_keys
        and isinstance(row.get("path"), str)
        and (not retained or isinstance(row.get("role"), str))
        and _SHA256_RE.fullmatch(str(row.get("sha256"))) is not None
        and row.get("head_blob_sha256") == row.get("sha256")
        and row.get("clean_tracked_HEAD_blob") is True
        for row in rows
    )


def _validate_acquisition_receipt(
    receipt: Mapping[str, Any], blocks: tuple[BlockCommitment, ...]
) -> None:
    expected_keys = {
        "acquisition_runtime",
        "acquisition_sha256",
        "commitments",
        "counts",
        "decision",
        "implementation",
        "portfolio_design_binding",
        "original_twelve_exclusion",
        "preregistration_custody",
        "preregistration_sha256",
        "prospective_ordering",
        "retained_P_lineage",
        "safety",
        "schema",
        "selection_continuity",
        "source",
        "v2_final_disposition_binding",
        "v2_public_binding",
    }
    counts = receipt.get("counts")
    commitments = receipt.get("commitments")
    continuity = receipt.get("selection_continuity")
    ordering = receipt.get("prospective_ordering")
    source = receipt.get("source")
    portfolio_design = receipt.get("portfolio_design_binding")
    safety = receipt.get("safety")
    v2_public = receipt.get("v2_public_binding")
    v2_disposition = receipt.get("v2_final_disposition_binding")
    implementation = receipt.get("implementation")
    retained_p = receipt.get("retained_P_lineage")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != ACQUISITION_SCHEMA
        or receipt.get("decision")
        != "fresh_rank_window_six_block_pack_formed_no_measurement_authority"
        or source
        != {
            "canonical_role": "fixed_HF_HotpotQA_distractor_validation_parquet",
            "file_sha256": SOURCE_SHA256,
            "file_size": SOURCE_SIZE,
            "hf_repository_commit": HF_REPOSITORY_COMMIT,
            "row_count": SOURCE_ROW_COUNT,
            "label_provenance": (
                "source_provided_supporting_facts_from_fixed_HF_parquet"
            ),
            "original_CMU_JSON_byte_or_row_equivalence_claim": False,
            "original_CMU_JSON_label_equivalence_claim": False,
        }
        or not isinstance(portfolio_design, Mapping)
        or set(portfolio_design)
        != {
            "relative_path",
            "schema",
            "design_file_sha256",
            "design_sha256",
            "committed_custody",
        }
        or portfolio_design.get("relative_path") != PORTFOLIO_DESIGN_RELATIVE
        or portfolio_design.get("schema") != PORTFOLIO_DESIGN_SCHEMA
        or portfolio_design.get("design_file_sha256")
        != PORTFOLIO_DESIGN_FILE_SHA256
        or portfolio_design.get("design_sha256") != PORTFOLIO_DESIGN_SHA256
        or portfolio_design.get("committed_custody")
        != {
            "file_sha256": PORTFOLIO_DESIGN_FILE_SHA256,
            "head_blob_sha256": PORTFOLIO_DESIGN_FILE_SHA256,
            "clean_tracked_HEAD_blob": True,
        }
        or not isinstance(v2_public, Mapping)
        or set(v2_public)
        != {
            "schema",
            "relative_path",
            "acquisition_file_sha256",
            "acquisition_sha256",
            "committed_custody",
            "selected_rank_window_start_inclusive",
            "selected_rank_window_stop_exclusive",
            "selected_count",
            "selection_secret_commitment_sha256",
            "selection_domain_separator",
            "private_block_files_opened",
            "private_M_search_content_opened",
            "private_M_search_outcome_opened",
        }
        or v2_public.get("schema") != v2.ACQUISITION_SCHEMA
        or v2_public.get("relative_path") != V2_ACQUISITION_RELATIVE
        or v2_public.get("acquisition_file_sha256")
        != V2_ACQUISITION_FILE_SHA256
        or v2_public.get("acquisition_sha256") != V2_ACQUISITION_SHA256
        or v2_public.get("committed_custody")
        != {
            "file_sha256": V2_ACQUISITION_FILE_SHA256,
            "head_blob_sha256": V2_ACQUISITION_FILE_SHA256,
            "clean_tracked_HEAD_blob": True,
        }
        or v2_public.get("selected_rank_window_start_inclusive")
        != PREVIOUS_RANK_WINDOW_START
        or v2_public.get("selected_rank_window_stop_exclusive")
        != PREVIOUS_RANK_WINDOW_STOP
        or v2_public.get("selected_count") != PREVIOUS_RANK_WINDOW_STOP
        or _SHA256_RE.fullmatch(
            str(v2_public.get("selection_secret_commitment_sha256"))
        )
        is None
        or v2_public.get("selection_domain_separator")
        != SELECTION_DOMAIN_SEPARATOR
        or v2_public.get("private_block_files_opened") != 0
        or v2_public.get("private_M_search_content_opened") is not False
        or v2_public.get("private_M_search_outcome_opened") is not False
        or not isinstance(v2_disposition, Mapping)
        or set(v2_disposition)
        != {
            "schema",
            "relative_path",
            "disposition_file_sha256",
            "disposition_sha256",
            "committed_custody",
            "previous_M_search_authorized",
            "previous_M_search_opened",
            "future_study_requires_new_mechanism_and_new_cohort",
            "public_aggregate_outcomes_used_for_rank_selection",
        }
        or v2_disposition.get("schema") != V2_FINAL_DISPOSITION_SCHEMA
        or v2_disposition.get("relative_path")
        != V2_FINAL_DISPOSITION_RELATIVE
        or v2_disposition.get("disposition_file_sha256")
        != V2_FINAL_DISPOSITION_FILE_SHA256
        or v2_disposition.get("disposition_sha256")
        != V2_FINAL_DISPOSITION_SHA256
        or v2_disposition.get("committed_custody")
        != {
            "file_sha256": V2_FINAL_DISPOSITION_FILE_SHA256,
            "head_blob_sha256": V2_FINAL_DISPOSITION_FILE_SHA256,
            "clean_tracked_HEAD_blob": True,
        }
        or v2_disposition.get("previous_M_search_authorized") is not False
        or v2_disposition.get("previous_M_search_opened") is not False
        or v2_disposition.get(
            "future_study_requires_new_mechanism_and_new_cohort"
        )
        is not True
        or v2_disposition.get("public_aggregate_outcomes_used_for_rank_selection")
        is not False
        or not _clean_closure_is_structurally_valid(
            implementation, retained=False
        )
        or not _clean_closure_is_structurally_valid(
            retained_p, retained=True
        )
        or not isinstance(counts, Mapping)
        or counts.get("source_rows") != SOURCE_ROW_COUNT
        or counts.get("selected_rows") != SELECTED_COUNT
        or counts.get("rank_window_start_inclusive") != RANK_WINDOW_START
        or counts.get("rank_window_stop_exclusive") != RANK_WINDOW_STOP
        or counts.get("original_ids_present_and_structurally_eligible")
        != PRIOR_SAMPLE_COUNT
        or counts.get("selected_original_id_overlap") != 0
        or counts.get("selected_previous_rank_window_overlap") != 0
        or not isinstance(commitments, Mapping)
        or commitments.get("private_row_key_set_sha256")
        != stable_hash(sorted(PRIVATE_BLOCK_ROW_KEYS))
        or commitments.get("private_pack_sha256")
        != stable_hash([row.to_dict() for row in blocks])
        or commitments.get("item_ids_persisted_publicly") is not False
        or commitments.get("private_paths_persisted_publicly") is not False
        or commitments.get("selection_secret_commitment_sha256")
        != v2_public.get("selection_secret_commitment_sha256")
        or not isinstance(continuity, Mapping)
        or continuity.get("selection_secret_commitment_sha256")
        != v2_public.get("selection_secret_commitment_sha256")
        or continuity.get("selection_domain_separator")
        != SELECTION_DOMAIN_SEPARATOR
        or continuity.get("same_selection_secret_as_v2") is not True
        or continuity.get("rank_window_disjoint_from_v2_by_construction") is not True
        or continuity.get("v2_private_block_files_opened") != 0
        or continuity.get("previous_M_search_content_opened") is not False
        or continuity.get("previous_M_search_outcome_opened") is not False
        or continuity.get("previous_public_outcomes_used_for_selection") is not False
        or not isinstance(ordering, Mapping)
        or ordering.get("preregistration_committed_before_marker") is not True
        or ordering.get("marker_persisted_before_original_twelve_pack_open")
        is not True
        or ordering.get("marker_persisted_before_source_open") is not True
        or ordering.get("original_private_rows_opened_before_marker") != 0
        or ordering.get("source_rows_opened_before_marker") != 0
        or ordering.get("retry_replay_resample_authorized") is not False
        or safety
        != {
            "formation_executed": False,
            "measurement_executed": False,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "scores_computed": 0,
            "v2_private_block_rows_read": 0,
        }
    ):
        raise HotpotEvaluatorRobustAcquisitionError("acquisition receipt drifted")
    if not isinstance(commitments.get("block_files"), list):
        raise HotpotEvaluatorRobustAcquisitionError("block commitments are malformed")
    if commitments["block_files"] != [row.to_dict() for row in blocks]:
        raise HotpotEvaluatorRobustAcquisitionError("block commitment drifted")
    for value, field_name in (
        (receipt.get("preregistration_sha256"), "preregistration hash"),
        (commitments.get("private_pack_sha256"), "private pack hash"),
        (commitments.get("private_locator_file_sha256"), "private locator hash"),
        (commitments.get("private_row_key_set_sha256"), "private row key-set hash"),
        (
            commitments.get("selection_secret_commitment_sha256"),
            "selection secret commitment",
        ),
        (
            ordering.get("acquisition_consumption_file_sha256"),
            "consumption file hash",
        ),
        (
            ordering.get("acquisition_consumption_sha256"),
            "consumption semantic hash",
        ),
    ):
        _require_sha256(value, field_name)
    for binding_name in (
        "v2_public_binding",
        "v2_final_disposition_binding",
        "retained_P_lineage",
        "implementation",
        "preregistration_custody",
        "original_twelve_exclusion",
    ):
        if not isinstance(receipt.get(binding_name), Mapping):
            raise HotpotEvaluatorRobustAcquisitionError(
                f"{binding_name} is malformed"
            )
    _assert_public_safe(receipt)


def load_acquisition_binding(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[BlockCommitment, ...]]:
    """Load and strictly validate the public acquisition receipt."""

    receipt, _raw = _read_json_object(path, "robust-evaluator acquisition receipt")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "acquisition receipt hash"
    )
    if receipt.get("schema") != ACQUISITION_SCHEMA or stable_hash(body) != declared:
        raise HotpotEvaluatorRobustAcquisitionError(
            "acquisition receipt self-hash drifted"
        )
    rows = receipt.get("commitments", {}).get("block_files")
    if not isinstance(rows, list) or len(rows) != len(BLOCK_ORDER):
        raise HotpotEvaluatorRobustAcquisitionError("block commitments are malformed")
    blocks: list[BlockCommitment] = []
    for expected, row in zip(BLOCK_ORDER, rows):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"block", "count", "file_sha256", "item_commitment_set_sha256"}
            or row.get("block") != expected
            or row.get("count") != BLOCK_COUNTS[expected]
        ):
            raise HotpotEvaluatorRobustAcquisitionError("block commitment drifted")
        blocks.append(
            BlockCommitment(
                block=expected,
                count=BLOCK_COUNTS[expected],
                file_sha256=_require_sha256(row.get("file_sha256"), "block file hash"),
                item_commitment_set_sha256=_require_sha256(
                    row.get("item_commitment_set_sha256"),
                    "block item commitment set",
                ),
            )
        )
    _validate_acquisition_receipt(receipt, tuple(blocks))
    return receipt, tuple(blocks)


def acquire_private_blocks(
    *,
    project: Path,
    preregistration_path: Path,
    selection_secret_path: Path,
    v2_acquisition_receipt_path: Path,
    v2_final_disposition_path: Path,
    original_private_pack_path: Path,
    source_parquet_path: Path,
    private_root: Path,
    private_locator_path: Path,
) -> dict[str, Any]:
    """Consume one authorization and allocate exact HMAC ranks [156, 324)."""

    root = project.resolve(strict=True)
    preregistration = verify_preregistration(
        path=preregistration_path,
        project=root,
        selection_secret_path=selection_secret_path,
        v2_acquisition_receipt_path=v2_acquisition_receipt_path,
        v2_final_disposition_path=v2_final_disposition_path,
    )
    preregistration_custody = _committed_public_binding(
        project=root,
        path=preregistration_path,
        field_name="robust-evaluator preregistration",
    )
    secret = _read_selection_secret(project=root, path=selection_secret_path)
    source = _assert_git_ignored_private_path(
        project=root, path=source_parquet_path, require_file=None
    )
    pack_root = _assert_git_ignored_private_path(
        project=root, path=private_root, require_file=False
    )
    locator = _assert_git_ignored_private_path(
        project=root, path=private_locator_path, require_file=None
    )
    consumption_path = _assert_git_ignored_private_path(
        project=root,
        path=root / ACQUISITION_CONSUMPTION_RELATIVE,
        require_file=None,
    )
    if consumption_path.exists():
        raise FileExistsError("robust-evaluator acquisition was already consumed")
    if pack_root.exists() or locator.exists():
        raise FileExistsError("robust-evaluator private output already exists")
    if (
        locator == consumption_path
        or pack_root in locator.parents
        or locator in pack_root.parents
        or pack_root == consumption_path
        or pack_root in consumption_path.parents
        or consumption_path in pack_root.parents
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "private locator and pack root must be separate"
        )
    if not hmac.compare_digest(
        _selection_secret_commitment(secret),
        preregistration["selection"]["selection_secret_commitment_sha256"],
    ):
        raise HotpotEvaluatorRobustAcquisitionError("selection secret drifted")
    if preregistration.get("acquisition_runtime") != original.acquisition_runtime_binding():
        raise HotpotEvaluatorRobustAcquisitionError("acquisition runtime drifted")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - workspace dependency
        raise HotpotEvaluatorRobustAcquisitionError("pyarrow is unavailable") from exc

    original_binding = preregistration["original_twelve_exclusion"]
    consumption_body = {
        "schema": ACQUISITION_CONSUMPTION_SCHEMA,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "source_file_sha256": SOURCE_SHA256,
        "portfolio_design_file_sha256": preregistration[
            "portfolio_design_binding"
        ]["design_file_sha256"],
        "portfolio_design_sha256": preregistration["portfolio_design_binding"][
            "design_sha256"
        ],
        "original_private_pack_file_sha256": original_binding[
            "prior_private_pack_file_sha256"
        ],
        "v2_acquisition_file_sha256": preregistration["v2_public_binding"][
            "acquisition_file_sha256"
        ],
        "v2_final_disposition_file_sha256": preregistration[
            "v2_final_disposition_binding"
        ]["disposition_file_sha256"],
        "private_root_path_hash": stable_hash(
            {"absolute_private_root": str(pack_root)}
        ),
        "private_locator_path_hash": stable_hash(
            {"absolute_private_locator": str(locator)}
        ),
        "original_private_rows_opened_before_consumption": 0,
        "source_rows_opened_before_consumption": 0,
        "v2_private_block_rows_opened": 0,
        "retry_replay_resample_authorized": False,
    }
    _write_json_exclusive(
        consumption_path,
        consumption_body,
        hash_field="consumption_sha256",
        mode=0o600,
    )
    consumption_raw = consumption_path.read_bytes()

    # Both private inputs are first opened only after the marker is durable.
    original_ids = _read_original_private_ids_after_marker(
        project=root,
        original_private_pack_path=original_private_pack_path,
        original_binding=original_binding,
        consumption_path=consumption_path,
    )
    source = _assert_git_ignored_private_path(
        project=root, path=source, require_file=True
    )
    if source.stat().st_size != SOURCE_SIZE or _sha256_file(source) != SOURCE_SHA256:
        raise HotpotEvaluatorRobustAcquisitionError("fixed source identity drifted")
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != SOURCE_ROW_COUNT:
        raise HotpotEvaluatorRobustAcquisitionError("source row count drifted")
    if tuple(parquet.schema_arrow.names) != EXPECTED_SOURCE_FIELDS:
        raise HotpotEvaluatorRobustAcquisitionError("source schema drifted")
    source_rows = parquet.read().to_pylist()
    normalized = [_normalized_source_row(row) for row in source_rows]
    id_counts = Counter(
        row.get("id")
        for row in source_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("id"), str)
        and row.get("id").strip()
    )
    eligible_before_exclusion = [
        row
        for row in normalized
        if row is not None and id_counts[row["item_id"]] == 1
    ]
    eligible_ids = {row["item_id"] for row in eligible_before_exclusion}
    if original_ids - eligible_ids:
        raise HotpotEvaluatorRobustAcquisitionError(
            "an original excluded ID is absent from the structurally eligible source"
        )
    eligible = [
        row
        for row in eligible_before_exclusion
        if row["item_id"] not in original_ids
    ]
    if len(eligible) < RANK_WINDOW_STOP:
        raise HotpotEvaluatorRobustAcquisitionError(
            "insufficient eligible rows for the fixed next-rank window"
        )
    eligible.sort(
        key=lambda row: (
            _selection_key(str(row["item_id"]), secret),
            str(row["item_id"]),
        )
    )
    previous_rank_ids = {
        row["item_id"]
        for row in eligible[PREVIOUS_RANK_WINDOW_START:PREVIOUS_RANK_WINDOW_STOP]
    }
    selected = eligible[RANK_WINDOW_START:RANK_WINDOW_STOP]
    if (
        len(selected) != SELECTED_COUNT
        or any(row["item_id"] in original_ids for row in selected)
        or any(row["item_id"] in previous_rank_ids for row in selected)
    ):
        raise HotpotEvaluatorRobustAcquisitionError(
            "fixed rank-window disjointness failed"
        )

    os.mkdir(pack_root, 0o700)
    block_commitments: list[BlockCommitment] = []
    offset = 0
    for block in BLOCK_ORDER:
        count = BLOCK_COUNTS[block]
        rows = tuple(
            {"block": block, **row} for row in selected[offset : offset + count]
        )
        offset += count
        if len(rows) != count or any(set(row) != PRIVATE_BLOCK_ROW_KEYS for row in rows):
            raise HotpotEvaluatorRobustAcquisitionError(
                "private block row schema drifted"
            )
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
        raise HotpotEvaluatorRobustAcquisitionError(
            "private block allocation drifted"
        )
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
        "decision": (
            "fresh_rank_window_six_block_pack_formed_no_measurement_authority"
        ),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preregistration_custody": preregistration_custody,
        "source": {
            "canonical_role": "fixed_HF_HotpotQA_distractor_validation_parquet",
            "file_sha256": SOURCE_SHA256,
            "file_size": SOURCE_SIZE,
            "hf_repository_commit": HF_REPOSITORY_COMMIT,
            "row_count": SOURCE_ROW_COUNT,
            "label_provenance": (
                "source_provided_supporting_facts_from_fixed_HF_parquet"
            ),
            "original_CMU_JSON_byte_or_row_equivalence_claim": False,
            "original_CMU_JSON_label_equivalence_claim": False,
        },
        "counts": {
            "source_rows": len(source_rows),
            "structurally_valid_rows": sum(row is not None for row in normalized),
            "eligible_unique_id_rows_before_original_exclusion": len(
                eligible_before_exclusion
            ),
            "original_ids_present_and_structurally_eligible": len(
                original_ids.intersection(eligible_ids)
            ),
            "eligible_after_original_exclusion": len(eligible),
            "rank_window_start_inclusive": RANK_WINDOW_START,
            "rank_window_stop_exclusive": RANK_WINDOW_STOP,
            "selected_rows": len(selected),
            "selected_original_id_overlap": sum(
                row["item_id"] in original_ids for row in selected
            ),
            "selected_previous_rank_window_overlap": sum(
                row["item_id"] in previous_rank_ids for row in selected
            ),
        },
        "original_twelve_exclusion": {
            **original_binding,
            "private_pack_opened_after_marker": True,
            "excluded_item_count": len(original_ids),
            "selected_item_overlap_count": 0,
            "outcomes_used_for_selection": False,
        },
        "v2_public_binding": preregistration["v2_public_binding"],
        "v2_final_disposition_binding": preregistration[
            "v2_final_disposition_binding"
        ],
        "portfolio_design_binding": preregistration["portfolio_design_binding"],
        "selection_continuity": {
            "selection_domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "same_selection_secret_as_v2": True,
            "previous_rank_window_start_inclusive": PREVIOUS_RANK_WINDOW_START,
            "previous_rank_window_stop_exclusive": PREVIOUS_RANK_WINDOW_STOP,
            "rank_window_start_inclusive": RANK_WINDOW_START,
            "rank_window_stop_exclusive": RANK_WINDOW_STOP,
            "rank_window_disjoint_from_v2_by_construction": True,
            "v2_private_block_files_opened": 0,
            "previous_M_search_content_opened": False,
            "previous_M_search_outcome_opened": False,
            "previous_public_outcomes_used_for_selection": False,
        },
        "implementation": preregistration["implementation"],
        "retained_P_lineage": preregistration["retained_P_lineage"],
        "acquisition_runtime": original.acquisition_runtime_binding(),
        "prospective_ordering": {
            "preregistration_committed_before_marker": True,
            "marker_persisted_before_original_twelve_pack_open": True,
            "marker_persisted_before_source_open": True,
            "original_private_rows_opened_before_marker": 0,
            "source_rows_opened_before_marker": 0,
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
            "v2_private_block_rows_read": 0,
        },
    }
    _assert_public_safe(receipt)
    return receipt


__all__ = [
    "ACQUISITION_SCHEMA",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "BlockCommitment",
    "HotpotEvaluatorRobustAcquisitionError",
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
    "load_private_block",
    "portfolio_design_binding",
    "retained_p_lineage_binding",
    "verify_preregistration",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    acquire = subparsers.add_parser("acquire")
    for command in (preregister, acquire):
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--selection-secret", type=Path, required=True)
        command.add_argument("--v2-acquisition-receipt", type=Path, required=True)
        command.add_argument("--v2-final-disposition", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--original-private-pack", type=Path, required=True)
    acquire.add_argument("--source-parquet", type=Path, required=True)
    acquire.add_argument("--private-root", type=Path, required=True)
    acquire.add_argument("--private-locator", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.output.exists():
        raise FileExistsError("public robust-evaluator output already exists")
    common = {
        "project": arguments.project,
        "selection_secret_path": arguments.selection_secret,
        "v2_acquisition_receipt_path": arguments.v2_acquisition_receipt,
        "v2_final_disposition_path": arguments.v2_final_disposition,
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
        original_private_pack_path=arguments.original_private_pack,
        source_parquet_path=arguments.source_parquet,
        private_root=arguments.private_root,
        private_locator_path=arguments.private_locator,
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

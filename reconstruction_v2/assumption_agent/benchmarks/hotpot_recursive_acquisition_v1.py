"""Prospective HotpotQA recursive-retention and evaluator-study acquisition.

The canonical source is the same immutable Hugging Face Parquet used by the
closed twelve-item family-out run.  That conversion supplies the support
labels used here; this protocol makes no byte, row, or label-equivalence claim
about the original CMU JSON.  Preregistration opens neither source rows nor the
prior private pack.  A single acquisition marker is persisted before either
is opened, after which the exact prior twelve-item pack is verified and its IDs
are excluded from a fresh private-HMAC selection.
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
from . import hotpot_family_out_acquisition_v1 as prior
from .musique_official_core_comparison_v1 import (
    SELECTION_SECRET_BYTES,
    _assert_git_ignored_private_path,
    _read_selection_secret,
    _selection_secret_commitment,
    generate_selection_secret,
)


VERSION = "hotpot_recursive_retention_evaluator_v1"
PREREGISTRATION_SCHEMA = f"{VERSION}_preregistration"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition"
PRIVATE_BLOCK_ROW_SCHEMA = f"{VERSION}_private_block_row"
PRIVATE_LOCATOR_SCHEMA = f"{VERSION}_private_locator"
ACQUISITION_CONSUMPTION_SCHEMA = f"{VERSION}_acquisition_consumption"

BLOCK_ORDER = ("F_Q", "M_L4", "A_form", "A_hold", "F_search", "M_search")
BLOCK_COUNTS = {
    "F_Q": 36,
    "M_L4": 24,
    "A_form": 24,
    "A_hold": 24,
    "F_search": 24,
    "M_search": 24,
}
SELECTED_COUNT = sum(BLOCK_COUNTS.values())
PRIVATE_BLOCK_ROW_KEYS = frozenset(
    {
        "block",
        "corpus",
        "item_id",
        "question",
        "source_row_sha256",
        "support_indices",
    }
)
TOP_K = prior.TOP_K
PRIOR_SAMPLE_COUNT = prior.SAMPLE_COUNT

SOURCE_URL = prior.SOURCE_URL
HF_REPOSITORY = prior.HF_REPOSITORY
HF_REPOSITORY_COMMIT = prior.HF_REPOSITORY_COMMIT
SOURCE_SHA256 = prior.SOURCE_SHA256
SOURCE_SIZE = prior.SOURCE_SIZE
SOURCE_ROW_COUNT = prior.SOURCE_ROW_COUNT
EXPECTED_SOURCE_FIELDS = prior.EXPECTED_SOURCE_FIELDS

ACQUISITION_CONSUMPTION_RELATIVE = (
    "artifacts/hotpot_recursive_acquisition_v1/authorization.consumed.json"
)

# The future L4 and evaluator implementations are deliberately prospective
# dependencies: preregistration fails until both exist and their exact bytes
# are included in this closure.
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/archive.py",
    "assumption_agent/benchmarks/hotpot_family_out_acquisition_v1.py",
    "assumption_agent/benchmarks/hotpot_recursive_acquisition_v1.py",
    "assumption_agent/benchmarks/hotpot_recursive_l4_v1.py",
    "assumption_agent/benchmarks/hotpot_evaluator_coevolution_v2.py",
    "assumption_agent/benchmarks/hotpot_family_out_runner_v1.py",
    "assumption_agent/benchmarks/l4_retention_protocol_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/benchmarks/musique_m1_retrieval_runner_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)

# P is not chosen after the fresh Hotpot blocks have been formed.  These four
# already-public artifacts identify the exact retained generation-one program
# and its positive M1 lineage before acquisition authority is consumed.
RETAINED_P_LINEAGE_RELATIVE_FILES = (
    (
        "P_formation_receipt",
        "manifests/musique_recursive_study_f1_formation_v1/formation.receipt.json",
    ),
    (
        "P_frozen_program",
        "manifests/musique_recursive_study_f1_formation_v1/frozen_program.json",
    ),
    (
        "M1_pre_run_freeze",
        "manifests/musique_recursive_study_m1_pre_run_freeze_v1.json",
    ),
    (
        "M1_positive_promotion_report",
        "manifests/musique_recursive_study_m1_aggregate_report_v1.json",
    ),
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_PRIOR_ROW_KEYS = frozenset(
    {
        "corpus",
        "item_id",
        "question",
        "schema",
        "source_row_sha256",
        "support_indices",
    }
)


class HotpotRecursiveAcquisitionError(RuntimeError):
    """The recursive-study source, exclusion, or custody contract drifted."""


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
        raise HotpotRecursiveAcquisitionError("required file is unavailable")
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
        raise HotpotRecursiveAcquisitionError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


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
        raise HotpotRecursiveAcquisitionError(
            "public recursive-study artifact contains private content or paths"
        )


def implementation_binding(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotRecursiveAcquisitionError(
                f"implementation file missing or symlinked: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {"files": rows, "set_sha256": stable_hash(rows)}


def retained_p_lineage_binding(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    rows: list[dict[str, str]] = []
    for role, relative in RETAINED_P_LINEAGE_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotRecursiveAcquisitionError(
                f"retained-P lineage file missing or symlinked: {relative}"
            )
        rows.append(
            {"role": role, "path": relative, "sha256": _sha256_file(path)}
        )
    return {
        "files": rows,
        "set_sha256": stable_hash(rows),
        "fixed_before_fresh_block_selection": True,
    }


def load_acquisition_binding(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[BlockCommitment, ...]]:
    """Load a public receipt and return its exact ordered block commitments."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HotpotRecursiveAcquisitionError("acquisition receipt is unavailable")
    raw = candidate.read_bytes()
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotRecursiveAcquisitionError("acquisition receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise HotpotRecursiveAcquisitionError("acquisition receipt must be one object")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "acquisition receipt hash"
    )
    if (
        receipt.get("schema") != ACQUISITION_SCHEMA
        or stable_hash(body) != declared
    ):
        raise HotpotRecursiveAcquisitionError("acquisition receipt self-hash drifted")
    commitments = receipt.get("commitments")
    rows = commitments.get("block_files") if isinstance(commitments, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(BLOCK_ORDER):
        raise HotpotRecursiveAcquisitionError("block commitments are malformed")
    blocks: list[BlockCommitment] = []
    for expected, row in zip(BLOCK_ORDER, rows):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"block", "count", "file_sha256", "item_commitment_set_sha256"}
            or row.get("block") != expected
            or row.get("count") != BLOCK_COUNTS[expected]
        ):
            raise HotpotRecursiveAcquisitionError("block commitment drifted")
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
    _validate_acquisition_receipt(receipt, raw, tuple(blocks))
    return receipt, tuple(blocks)


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


def _read_json_object(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise HotpotRecursiveAcquisitionError(f"{field_name} is unavailable")
    raw = candidate.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotRecursiveAcquisitionError(f"{field_name} is invalid") from exc
    if not isinstance(payload, dict):
        raise HotpotRecursiveAcquisitionError(f"{field_name} must be one object")
    return payload, raw


def _prior_acquisition_binding(path: str | Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    payload, raw = _read_json_object(path, "prior family-out acquisition receipt")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "prior acquisition hash"
    )
    expected_keys = {
        "acquisition_runtime",
        "acquisition_sha256",
        "commitments",
        "counts",
        "decision",
        "preregistration_custody",
        "preregistration_sha256",
        "prospective_ordering",
        "safety",
        "schema",
        "source",
    }
    source = payload.get("source")
    counts = payload.get("counts")
    commitments = payload.get("commitments")
    ordering = payload.get("prospective_ordering")
    custody = payload.get("preregistration_custody")
    runtime = payload.get("acquisition_runtime")
    if (
        set(payload) != expected_keys
        or payload.get("schema") != prior.ACQUISITION_SCHEMA
        or payload.get("decision")
        != "fresh_family_out_pack_formed_measurement_not_authorized"
        or stable_hash(body) != declared
        or source
        != {
            "file_sha256": SOURCE_SHA256,
            "file_size": SOURCE_SIZE,
            "hf_repository_commit": HF_REPOSITORY_COMMIT,
            "original_CMU_JSON_equivalence_claim": False,
            "row_count": SOURCE_ROW_COUNT,
        }
        or not isinstance(counts, Mapping)
        or set(counts)
        != {
            "eligible_unique_id_rows",
            "selected_rows",
            "source_rows",
            "structurally_valid_rows",
        }
        or counts.get("source_rows") != SOURCE_ROW_COUNT
        or counts.get("selected_rows") != PRIOR_SAMPLE_COUNT
        or type(counts.get("eligible_unique_id_rows")) is not int
        or type(counts.get("structurally_valid_rows")) is not int
        or not (
            PRIOR_SAMPLE_COUNT
            <= counts["eligible_unique_id_rows"]
            <= counts["structurally_valid_rows"]
            <= SOURCE_ROW_COUNT
        )
        or not isinstance(commitments, Mapping)
        or set(commitments)
        != {
            "item_commitment_set_sha256",
            "item_ids_persisted_publicly",
            "private_pack_file_sha256",
            "private_paths_persisted_publicly",
            "selection_secret_commitment_sha256",
        }
        or commitments.get("item_ids_persisted_publicly") is not False
        or commitments.get("private_paths_persisted_publicly") is not False
        or payload.get("safety")
        != {
            "measurement_executed": False,
            "model_calls": 0,
            "online_evaluator_calls": 0,
            "scores_computed": 0,
        }
        or not isinstance(ordering, Mapping)
        or set(ordering)
        != {
            "acquisition_consumed_before_source_row_open",
            "acquisition_consumption_file_sha256",
            "acquisition_consumption_sha256",
            "preregistration_committed_before_source_row_open",
            "retry_replay_resample_authorized",
            "source_rows_opened_before_consumption",
        }
        or ordering.get("preregistration_committed_before_source_row_open") is not True
        or ordering.get("acquisition_consumed_before_source_row_open") is not True
        or ordering.get("source_rows_opened_before_consumption") != 0
        or ordering.get("retry_replay_resample_authorized") is not False
        or not isinstance(custody, Mapping)
        or set(custody)
        != {
            "preregistration_file_sha256",
            "preregistration_head_blob_sha256",
            "repository_commit",
        }
        or custody.get("preregistration_file_sha256")
        != custody.get("preregistration_head_blob_sha256")
        or not isinstance(custody.get("repository_commit"), str)
        or _GIT_COMMIT_RE.fullmatch(custody["repository_commit"]) is None
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {"pyarrow_version", "python_implementation", "python_version"}
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise HotpotRecursiveAcquisitionError("prior acquisition receipt drifted")
    for value, name in (
        (payload.get("preregistration_sha256"), "prior preregistration hash"),
        (commitments.get("private_pack_file_sha256"), "prior pack file hash"),
        (
            commitments.get("item_commitment_set_sha256"),
            "prior item commitment set",
        ),
        (
            commitments.get("selection_secret_commitment_sha256"),
            "prior selection secret commitment",
        ),
        (
            ordering.get("acquisition_consumption_file_sha256"),
            "prior consumption file hash",
        ),
        (
            ordering.get("acquisition_consumption_sha256"),
            "prior consumption semantic hash",
        ),
    ):
        _require_sha256(value, name)
    _assert_public_safe(payload)
    binding = {
        "acquisition_file_sha256": _sha256_bytes(raw),
        "acquisition_sha256": declared,
        "private_pack_file_sha256": commitments["private_pack_file_sha256"],
        "item_commitment_set_sha256": commitments[
            "item_commitment_set_sha256"
        ],
        "selected_item_count": PRIOR_SAMPLE_COUNT,
        "item_ids_persisted_publicly": False,
        "private_paths_persisted_publicly": False,
    }
    return payload, raw, binding


def _selection_key(item_id: str, secret: bytes) -> str:
    if len(secret) != SELECTION_SECRET_BYTES:
        raise HotpotRecursiveAcquisitionError("selection secret length drifted")
    return hmac.new(
        secret,
        f"{VERSION}:{item_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_preregistration(
    *,
    project: Path,
    selection_secret_path: Path,
    prior_acquisition_receipt_path: Path,
) -> dict[str, Any]:
    """Build a zero-row-access preregistration for all six private blocks."""

    root = project.resolve(strict=True)
    secret = _read_selection_secret(project=root, path=selection_secret_path)
    _prior_receipt, _prior_raw, prior_binding = _prior_acquisition_binding(
        prior_acquisition_receipt_path
    )
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
        "prior_exclusion": {
            **prior_binding,
            "private_pack_opened_during_preregistration": False,
            "all_exact_prior_pack_ids_excluded_before_new_HMAC_ranking": True,
            "prior_outcomes_used_for_selection": False,
        },
        "selection": {
            "algorithm": (
                "exclude_exact_prior_pack_ids_then_ascending_HMAC_SHA256_"
                "new_private_secret_and_source_id_v1"
            ),
            "domain_separator": VERSION,
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "selection_secret_persisted_publicly": False,
            "selected_count": SELECTED_COUNT,
            "block_order": list(BLOCK_ORDER),
            "block_counts": dict(BLOCK_COUNTS),
            "replacement": False,
            "manual_or_outcome_conditioned_selection": False,
        },
        "access_contract": {
            "formation_blocks": ["F_Q", "A_form", "F_search"],
            "measurement_blocks": ["M_L4", "A_hold", "M_search"],
            "all_six_blocks_formed_together": True,
            "one_shot_marker_precedes_prior_private_pack_open": True,
            "one_shot_marker_precedes_source_row_open": True,
            "committed_preregistration_precedes_marker": True,
            "measurement_requires_separate_pre_run_freeze": True,
            "retry_replay_resample": 0,
        },
        "study_contract": {
            "F_Q_item_count": BLOCK_COUNTS["F_Q"],
            "M_L4_item_count": BLOCK_COUNTS["M_L4"],
            "evaluator_formation_partition": "A_form",
            "evaluator_anchor_partition": "A_hold",
            "search_formation_partition": "F_search",
            "search_measurement_partition": "M_search",
            "primary_metric": "source_provided_support_recall_at_5",
            "offline_evaluation_only": True,
            "online_evaluator_calls": 0,
            "study_level_answer_generator_calls": 0,
            "official_arm_internal_frozen_local_LLM_OpenIE_retained": True,
        },
        "claim_boundary": {
            "source": "fixed_HF_conversion_not_original_CMU_JSON",
            "source_provided_label_claim_only": True,
            "original_CMU_JSON_equivalence_claim": False,
            "answer_generation_claim": False,
            "performance_claim_before_measurement": False,
        },
        "retained_P_lineage": retained_p_lineage_binding(root),
        "implementation": implementation_binding(root),
        "acquisition_runtime": prior.acquisition_runtime_binding(),
        "safety": {
            "source_rows_read": 0,
            "prior_private_pack_rows_read": 0,
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
    prior_acquisition_receipt_path: Path,
) -> dict[str, Any]:
    payload, _raw = _read_json_object(path, "recursive-study preregistration")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("preregistration_sha256", None), "preregistration hash"
    )
    if (
        payload.get("schema") != PREREGISTRATION_SCHEMA
        or stable_hash(body) != declared
    ):
        raise HotpotRecursiveAcquisitionError("preregistration self-hash drifted")
    expected = build_preregistration(
        project=project,
        selection_secret_path=selection_secret_path,
        prior_acquisition_receipt_path=prior_acquisition_receipt_path,
    )
    if payload != expected:
        raise HotpotRecursiveAcquisitionError(
            "preregistration differs from the complete live protocol"
        )
    return payload


def _read_prior_private_ids_after_marker(
    *,
    project: Path,
    prior_private_pack_path: Path,
    prior_binding: Mapping[str, Any],
    consumption_path: Path,
) -> frozenset[str]:
    """Open and verify the exact prior pack only after marker persistence."""

    if not consumption_path.is_file():
        raise HotpotRecursiveAcquisitionError(
            "one-shot marker must exist before prior private pack open"
        )
    pack = _assert_git_ignored_private_path(
        project=project, path=prior_private_pack_path, require_file=True
    )
    raw = pack.read_bytes()
    if (
        _sha256_bytes(raw) != prior_binding.get("private_pack_file_sha256")
        or not raw
        or not raw.endswith(b"\n")
    ):
        raise HotpotRecursiveAcquisitionError("exact prior private pack drifted")
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            if not line:
                raise HotpotRecursiveAcquisitionError(
                    "prior private pack contains a blank row"
                )
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise HotpotRecursiveAcquisitionError(
                    "prior private pack row is malformed"
                )
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotRecursiveAcquisitionError(
            "prior private pack JSONL is invalid"
        ) from exc
    if len(rows) != PRIOR_SAMPLE_COUNT or len(rows) != prior_binding.get(
        "selected_item_count"
    ):
        raise HotpotRecursiveAcquisitionError("prior private pack count drifted")
    if b"".join(_canonical_bytes(row) + b"\n" for row in rows) != raw:
        raise HotpotRecursiveAcquisitionError(
            "prior private pack is not canonical JSONL"
        )
    if stable_hash([stable_hash(row) for row in rows]) != prior_binding.get(
        "item_commitment_set_sha256"
    ):
        raise HotpotRecursiveAcquisitionError(
            "prior private pack item commitments drifted"
        )
    item_ids: list[str] = []
    for row in rows:
        if (
            set(row) != _PRIOR_ROW_KEYS
            or row.get("schema") != prior.PRIVATE_ROW_SCHEMA
            or not isinstance(row.get("item_id"), str)
            or not row["item_id"]
        ):
            raise HotpotRecursiveAcquisitionError(
                "prior private pack row schema drifted"
            )
        item_ids.append(row["item_id"])
    if len(set(item_ids)) != PRIOR_SAMPLE_COUNT:
        raise HotpotRecursiveAcquisitionError("prior private pack IDs are not unique")
    return frozenset(item_ids)


def _normalized_source_row(raw: object) -> dict[str, Any] | None:
    normalized = prior._normalize_source_row(raw)
    if normalized is None:
        return None
    return {
        "item_id": normalized["item_id"],
        "question": normalized["question"],
        "corpus": normalized["corpus"],
        "support_indices": normalized["support_indices"],
        "source_row_sha256": normalized["source_row_sha256"],
    }


def _validate_acquisition_receipt(
    receipt: Mapping[str, Any],
    raw: bytes,
    blocks: tuple[BlockCommitment, ...],
) -> None:
    del raw  # The loader already binds and verifies the exact self-hashed bytes.
    expected_keys = {
        "acquisition_runtime",
        "acquisition_sha256",
        "commitments",
        "counts",
        "decision",
        "implementation",
        "preregistration_custody",
        "preregistration_sha256",
        "prior_exclusion",
        "prospective_ordering",
        "retained_P_lineage",
        "safety",
        "schema",
        "source",
    }
    source = receipt.get("source")
    counts = receipt.get("counts")
    commitments = receipt.get("commitments")
    exclusion = receipt.get("prior_exclusion")
    ordering = receipt.get("prospective_ordering")
    custody = receipt.get("preregistration_custody")
    runtime = receipt.get("acquisition_runtime")
    implementation = receipt.get("implementation")
    implementation_files = (
        implementation.get("files")
        if isinstance(implementation, Mapping)
        else None
    )
    retained_p = receipt.get("retained_P_lineage")
    retained_p_files = (
        retained_p.get("files") if isinstance(retained_p, Mapping) else None
    )
    if (
        set(receipt) != expected_keys
        or receipt.get("decision")
        != "fresh_six_block_pack_formed_no_formation_measurement_or_scoring_authority"
        or source
        != {
            "canonical_role": "fixed_HF_HotpotQA_distractor_validation_parquet",
            "file_sha256": SOURCE_SHA256,
            "file_size": SOURCE_SIZE,
            "hf_repository_commit": HF_REPOSITORY_COMMIT,
            "label_provenance": (
                "source_provided_supporting_facts_from_fixed_HF_parquet"
            ),
            "original_CMU_JSON_byte_or_row_equivalence_claim": False,
            "original_CMU_JSON_label_equivalence_claim": False,
            "row_count": SOURCE_ROW_COUNT,
        }
        or not isinstance(counts, Mapping)
        or set(counts)
        != {
            "eligible_after_prior_exclusion",
            "eligible_unique_id_rows_before_prior_exclusion",
            "prior_ids_present_and_structurally_eligible",
            "selected_prior_id_overlap",
            "selected_rows",
            "source_rows",
            "structurally_valid_rows",
        }
        or counts.get("source_rows") != SOURCE_ROW_COUNT
        or counts.get("selected_rows") != SELECTED_COUNT
        or counts.get("prior_ids_present_and_structurally_eligible")
        != PRIOR_SAMPLE_COUNT
        or counts.get("selected_prior_id_overlap") != 0
        or any(
            type(counts.get(key)) is not int
            for key in (
                "eligible_after_prior_exclusion",
                "eligible_unique_id_rows_before_prior_exclusion",
                "structurally_valid_rows",
            )
        )
        or not (
            SELECTED_COUNT
            <= counts["eligible_after_prior_exclusion"]
            == counts["eligible_unique_id_rows_before_prior_exclusion"]
            - PRIOR_SAMPLE_COUNT
            <= counts["structurally_valid_rows"]
            <= SOURCE_ROW_COUNT
        )
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
        or commitments.get("item_ids_persisted_publicly") is not False
        or commitments.get("private_paths_persisted_publicly") is not False
        or commitments.get("private_row_key_set_sha256")
        != stable_hash(sorted(PRIVATE_BLOCK_ROW_KEYS))
        or commitments.get("private_pack_sha256")
        != stable_hash([block.to_dict() for block in blocks])
        or not isinstance(exclusion, Mapping)
        or exclusion.get("excluded_prior_item_count") != PRIOR_SAMPLE_COUNT
        or exclusion.get("selected_prior_item_overlap_count") != 0
        or exclusion.get("prior_private_pack_opened_after_marker") is not True
        or exclusion.get("prior_outcomes_used_for_selection") is not False
        or not isinstance(ordering, Mapping)
        or ordering.get("preregistration_committed_before_marker") is not True
        or ordering.get("marker_persisted_before_prior_private_pack_open") is not True
        or ordering.get("marker_persisted_before_source_row_open") is not True
        or ordering.get("source_rows_opened_before_marker") != 0
        or ordering.get("prior_private_rows_opened_before_marker") != 0
        or ordering.get("retry_replay_resample_authorized") is not False
        or not isinstance(custody, Mapping)
        or set(custody)
        != {
            "preregistration_file_sha256",
            "preregistration_head_blob_sha256",
            "repository_commit",
        }
        or custody.get("preregistration_file_sha256")
        != custody.get("preregistration_head_blob_sha256")
        or not isinstance(custody.get("repository_commit"), str)
        or _GIT_COMMIT_RE.fullmatch(custody["repository_commit"]) is None
        or not isinstance(runtime, Mapping)
        or set(runtime) != {
            "pyarrow_version",
            "python_implementation",
            "python_version",
        }
        or any(not isinstance(value, str) or not value for value in runtime.values())
        or not isinstance(implementation, Mapping)
        or set(implementation) != {"files", "set_sha256"}
        or not isinstance(implementation_files, list)
        or len(implementation_files) != len(IMPLEMENTATION_RELATIVE_FILES)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row.get("path") != relative
            or _SHA256_RE.fullmatch(str(row.get("sha256"))) is None
            for relative, row in zip(
                IMPLEMENTATION_RELATIVE_FILES, implementation_files
            )
        )
        or implementation.get("set_sha256")
        != stable_hash(implementation_files)
        or not isinstance(retained_p, Mapping)
        or set(retained_p)
        != {"files", "fixed_before_fresh_block_selection", "set_sha256"}
        or retained_p.get("fixed_before_fresh_block_selection") is not True
        or not isinstance(retained_p_files, list)
        or len(retained_p_files) != len(RETAINED_P_LINEAGE_RELATIVE_FILES)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "role", "sha256"}
            or row.get("role") != role
            or row.get("path") != relative
            or _SHA256_RE.fullmatch(str(row.get("sha256"))) is None
            for (role, relative), row in zip(
                RETAINED_P_LINEAGE_RELATIVE_FILES, retained_p_files
            )
        )
        or retained_p.get("set_sha256") != stable_hash(retained_p_files)
        or receipt.get("safety")
        != {
            "formation_executed": False,
            "measurement_executed": False,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "scores_computed": 0,
        }
    ):
        raise HotpotRecursiveAcquisitionError("acquisition receipt drifted")
    expected_exclusion_keys = {
        "excluded_prior_item_count",
        "prior_acquisition_file_sha256",
        "prior_acquisition_sha256",
        "prior_item_commitment_set_sha256",
        "prior_outcomes_used_for_selection",
        "prior_private_pack_file_sha256",
        "prior_private_pack_opened_after_marker",
        "selected_prior_item_overlap_count",
    }
    expected_ordering_keys = {
        "acquisition_consumption_file_sha256",
        "acquisition_consumption_sha256",
        "marker_persisted_before_prior_private_pack_open",
        "marker_persisted_before_source_row_open",
        "preregistration_committed_before_marker",
        "prior_private_rows_opened_before_marker",
        "retry_replay_resample_authorized",
        "source_rows_opened_before_marker",
    }
    if set(exclusion) != expected_exclusion_keys or set(ordering) != expected_ordering_keys:
        raise HotpotRecursiveAcquisitionError(
            "exclusion or prospective ordering schema drifted"
        )
    for value, name in (
        (receipt.get("preregistration_sha256"), "preregistration hash"),
        (
            commitments.get("private_locator_file_sha256"),
            "private locator file hash",
        ),
        (commitments.get("private_pack_sha256"), "private pack hash"),
        (
            commitments.get("private_row_key_set_sha256"),
            "private row key-set hash",
        ),
        (
            commitments.get("selection_secret_commitment_sha256"),
            "selection secret commitment",
        ),
        (exclusion.get("prior_acquisition_file_sha256"), "prior receipt file hash"),
        (exclusion.get("prior_acquisition_sha256"), "prior acquisition hash"),
        (
            exclusion.get("prior_item_commitment_set_sha256"),
            "prior item commitment set",
        ),
        (
            exclusion.get("prior_private_pack_file_sha256"),
            "prior private pack file hash",
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
        _require_sha256(value, name)
    _assert_public_safe(receipt)


def acquire_private_blocks(
    *,
    project: Path,
    preregistration_path: Path,
    selection_secret_path: Path,
    prior_acquisition_receipt_path: Path,
    prior_private_pack_path: Path,
    source_parquet_path: Path,
    private_root: Path,
    private_locator_path: Path,
) -> dict[str, Any]:
    """Consume one authorization and form all six disjoint private blocks."""

    root = project.resolve(strict=True)
    preregistration = verify_preregistration(
        path=preregistration_path,
        project=root,
        selection_secret_path=selection_secret_path,
        prior_acquisition_receipt_path=prior_acquisition_receipt_path,
    )
    preregistration_custody = prior.committed_public_file_receipt(
        project=root, path=preregistration_path
    )
    _prior_receipt, _prior_raw, prior_binding = _prior_acquisition_binding(
        prior_acquisition_receipt_path
    )
    source = _assert_git_ignored_private_path(
        project=root, path=source_parquet_path, require_file=True
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
    if pack_root.exists() or locator.exists():
        raise FileExistsError("recursive-study private output already exists")
    if consumption_path.exists():
        raise FileExistsError("recursive-study acquisition was already consumed")
    if (
        locator == consumption_path
        or pack_root in locator.parents
        or locator in pack_root.parents
        or pack_root == consumption_path
        or pack_root in consumption_path.parents
        or consumption_path in pack_root.parents
    ):
        raise HotpotRecursiveAcquisitionError(
            "private locator and pack root must be separate"
        )
    secret = _read_selection_secret(project=root, path=selection_secret_path)
    if not hmac.compare_digest(
        _selection_secret_commitment(secret),
        preregistration["selection"]["selection_secret_commitment_sha256"],
    ):
        raise HotpotRecursiveAcquisitionError("selection secret drifted")
    if preregistration.get("acquisition_runtime") != prior.acquisition_runtime_binding():
        raise HotpotRecursiveAcquisitionError("acquisition runtime drifted")
    if source.stat().st_size != SOURCE_SIZE or _sha256_file(source) != SOURCE_SHA256:
        raise HotpotRecursiveAcquisitionError("fixed source identity drifted")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - workspace dependency
        raise HotpotRecursiveAcquisitionError("pyarrow is unavailable") from exc

    consumption_body = {
        "schema": ACQUISITION_CONSUMPTION_SCHEMA,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "source_file_sha256": SOURCE_SHA256,
        "prior_acquisition_file_sha256": prior_binding[
            "acquisition_file_sha256"
        ],
        "prior_private_pack_file_sha256": prior_binding[
            "private_pack_file_sha256"
        ],
        "private_root_path_hash": stable_hash(
            {"absolute_private_root": str(pack_root)}
        ),
        "private_locator_path_hash": stable_hash(
            {"absolute_private_locator": str(locator)}
        ),
        "prior_private_rows_opened_before_consumption": 0,
        "source_rows_opened_before_consumption": 0,
        "retry_replay_resample_authorized": False,
    }
    _write_json_exclusive(
        consumption_path,
        consumption_body,
        hash_field="consumption_sha256",
        mode=0o600,
    )
    consumption_raw = consumption_path.read_bytes()

    prior_ids = _read_prior_private_ids_after_marker(
        project=root,
        prior_private_pack_path=prior_private_pack_path,
        prior_binding=prior_binding,
        consumption_path=consumption_path,
    )
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != SOURCE_ROW_COUNT:
        raise HotpotRecursiveAcquisitionError("source row count drifted")
    if tuple(parquet.schema_arrow.names) != EXPECTED_SOURCE_FIELDS:
        raise HotpotRecursiveAcquisitionError("source schema drifted")
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
    if prior_ids - eligible_ids:
        raise HotpotRecursiveAcquisitionError(
            "an exact prior ID is absent from the structurally eligible source"
        )
    eligible = [
        row for row in eligible_before_exclusion if row["item_id"] not in prior_ids
    ]
    if len(eligible) < SELECTED_COUNT:
        raise HotpotRecursiveAcquisitionError(
            "insufficient eligible rows after exact prior-pack exclusion"
        )
    eligible.sort(
        key=lambda row: (
            _selection_key(str(row["item_id"]), secret),
            str(row["item_id"]),
        )
    )
    selected = eligible[:SELECTED_COUNT]
    if any(row["item_id"] in prior_ids for row in selected):
        raise HotpotRecursiveAcquisitionError("prior ID exclusion failed")

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
            raise HotpotRecursiveAcquisitionError("private block row schema drifted")
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
        raise HotpotRecursiveAcquisitionError("private block allocation drifted")
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
        locator,
        locator_body,
        hash_field="locator_sha256",
        mode=0o600,
    )
    locator_raw = locator.read_bytes()
    receipt: dict[str, Any] = {
        "schema": ACQUISITION_SCHEMA,
        "decision": (
            "fresh_six_block_pack_formed_no_formation_measurement_or_scoring_authority"
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
            "eligible_unique_id_rows_before_prior_exclusion": len(
                eligible_before_exclusion
            ),
            "prior_ids_present_and_structurally_eligible": len(
                prior_ids.intersection(eligible_ids)
            ),
            "eligible_after_prior_exclusion": len(eligible),
            "selected_rows": len(selected),
            "selected_prior_id_overlap": sum(
                row["item_id"] in prior_ids for row in selected
            ),
        },
        "prior_exclusion": {
            "prior_acquisition_file_sha256": prior_binding[
                "acquisition_file_sha256"
            ],
            "prior_acquisition_sha256": prior_binding["acquisition_sha256"],
            "prior_private_pack_file_sha256": prior_binding[
                "private_pack_file_sha256"
            ],
            "prior_item_commitment_set_sha256": prior_binding[
                "item_commitment_set_sha256"
            ],
            "excluded_prior_item_count": len(prior_ids),
            "selected_prior_item_overlap_count": 0,
            "prior_private_pack_opened_after_marker": True,
            "prior_outcomes_used_for_selection": False,
        },
        "implementation": preregistration["implementation"],
        "retained_P_lineage": preregistration["retained_P_lineage"],
        "acquisition_runtime": prior.acquisition_runtime_binding(),
        "prospective_ordering": {
            "preregistration_committed_before_marker": True,
            "marker_persisted_before_prior_private_pack_open": True,
            "marker_persisted_before_source_row_open": True,
            "prior_private_rows_opened_before_marker": 0,
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
            "scores_computed": 0,
            "online_evaluator_calls": 0,
        },
    }
    _assert_public_safe(receipt)
    return receipt


__all__ = [
    "ACQUISITION_SCHEMA",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "BlockCommitment",
    "HotpotRecursiveAcquisitionError",
    "PREREGISTRATION_SCHEMA",
    "PRIVATE_BLOCK_ROW_KEYS",
    "PRIVATE_BLOCK_ROW_SCHEMA",
    "PRIVATE_LOCATOR_SCHEMA",
    "RETAINED_P_LINEAGE_RELATIVE_FILES",
    "SELECTED_COUNT",
    "acquire_private_blocks",
    "build_preregistration",
    "generate_selection_secret",
    "implementation_binding",
    "load_acquisition_binding",
    "retained_p_lineage_binding",
    "verify_preregistration",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    secret = subparsers.add_parser("generate-secret")
    preregister = subparsers.add_parser("preregister")
    acquire = subparsers.add_parser("acquire")

    secret.add_argument("--project", type=Path, required=True)
    secret.add_argument("--output", type=Path, required=True)
    for command in (preregister, acquire):
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--selection-secret", type=Path, required=True)
        command.add_argument(
            "--prior-acquisition-receipt", type=Path, required=True
        )
        command.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--prior-private-pack", type=Path, required=True)
    acquire.add_argument("--source-parquet", type=Path, required=True)
    acquire.add_argument("--private-root", type=Path, required=True)
    acquire.add_argument("--private-locator", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "generate-secret":
        commitment = generate_selection_secret(
            project=arguments.project, output=arguments.output
        )
        print(json.dumps({"selection_secret_commitment_sha256": commitment}))
        return 0
    if arguments.output.exists():
        raise FileExistsError("public recursive-study output already exists")
    common = {
        "project": arguments.project,
        "selection_secret_path": arguments.selection_secret,
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
        prior_private_pack_path=arguments.prior_private_pack,
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

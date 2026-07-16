"""Pre-register and acquire a fresh MuSiQue recursive-study pack.

The prior six-item study came from MuSiQue's official TRAIN split and is closed
without an efficacy result.  This version uses the disjoint official DEV split,
forms eight private HMAC blocks before any formation or measurement, and keeps
all questions, labels, IDs, and paths outside public receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import zipfile

from ..models import stable_hash
from .musique_official_core_comparison_v1 import (
    OFFICIAL_SOURCE_COMMIT,
    SELECTION_SECRET_BYTES,
    _assert_git_ignored_private_path,
    _canonical_bytes,
    _iter_source_rows,
    _normalize_source_row,
    _read_selection_secret,
    _selection_secret_commitment,
    _sha256_bytes,
    _sha256_file,
    _write_jsonl_exclusive,
    generate_selection_secret,
    official_source_receipt,
)


VERSION = "musique_recursive_evaluator_study_v1"
PREREGISTRATION_SCHEMA = f"{VERSION}_preregistration"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition"
PRIVATE_ROW_SCHEMA = f"{VERSION}_private_row"
OFFICIAL_DEV_MEMBER_BASENAME = "musique_ans_v1.0_dev.jsonl"
OFFICIAL_ARCHIVE_SHA256 = (
    "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
)
BLOCK_ORDER = ("F1", "M1", "F2", "M2", "F3", "M3", "A_form", "A_hold")
BLOCK_COUNT = 12
BLOCK_COUNTS = {name: BLOCK_COUNT for name in BLOCK_ORDER}
SELECTED_COUNT = len(BLOCK_ORDER) * BLOCK_COUNT
TOP_K = 5
RRF_K = 60

# This is a fixed transitive research-protocol closure.  Prospective runtime
# attestation and L4 protocol files are added here before the preregistration is
# ever instantiated; the builder fails if any declared file is missing.
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/archive.py",
    "assumption_agent/benchmarks/evaluator_score_dependency_v1.py",
    "assumption_agent/benchmarks/l4_retention_protocol_v1.py",
    "assumption_agent/benchmarks/musique_evaluator_coevolution_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_m1_retrieval_runner_v1.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_acquisition_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)


class MuSiQueRecursiveAcquisitionError(RuntimeError):
    """Raised when the fresh-study custody contract cannot be audited."""


def _implementation_binding(project: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueRecursiveAcquisitionError(
                f"implementation file missing or symlinked: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {"files": rows, "set_sha256": stable_hash(rows)}


def _selection_key(item_id: str, secret: bytes) -> str:
    if len(secret) != SELECTION_SECRET_BYTES:
        raise MuSiQueRecursiveAcquisitionError("selection secret length mismatch")
    return hmac.new(
        secret,
        f"{VERSION}:{item_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _find_exact_member(archive: zipfile.ZipFile) -> str:
    matches = sorted(
        name
        for name in archive.namelist()
        if Path(name).name == OFFICIAL_DEV_MEMBER_BASENAME and not name.endswith("/")
    )
    if len(matches) != 1:
        raise MuSiQueRecursiveAcquisitionError(
            "official DEV archive member is missing or ambiguous"
        )
    return matches[0]


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    mode: int = 0o600,
) -> None:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = stable_hash(body)
    raw = json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_safe_public_payload(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answers"',
        '"corpus"',
        '"item_id"',
        '"normalized_answers"',
        '"paragraph_text"',
        '"question"',
        '"support_indices"',
        '"private_root"',
        '"selection_secret_path"',
    )
    if any(value in serialized for value in forbidden):
        raise MuSiQueRecursiveAcquisitionError(
            "unsafe content or locator in public payload"
        )


def build_preregistration(
    *,
    project: Path,
    official_repository: Path,
    selection_secret_path: Path,
) -> dict[str, Any]:
    project = project.resolve(strict=True)
    source = official_source_receipt(official_repository)
    if source.get("commit") != OFFICIAL_SOURCE_COMMIT:
        raise MuSiQueRecursiveAcquisitionError("official source commit mismatch")
    secret = _read_selection_secret(
        project=project,
        path=selection_secret_path,
    )
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "decision": "acquisition_only_no_formation_measurement_or_scoring_authority",
        "source": {
            "repository": source["repository"],
            "commit": source["commit"],
            "license": source["license"],
            "official_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "archive_member_basename": OFFICIAL_DEV_MEMBER_BASENAME,
            "source_split": "official_dev",
            "prior_closed_cohort_source_split": "official_train",
            "split_disjoint_from_prior_closed_cohort": True,
        },
        "eligibility": {
            "normalizer": "musique_official_core_comparison_v1._normalize_source_row",
            "answerable": True,
            "minimum_paragraph_count": TOP_K,
            "minimum_supporting_paragraph_count": 2,
            "minimum_non_supporting_paragraph_count": 1,
            "minimum_distinct_normalized_answers": 2,
            "paragraph_idx_namespace": "official_contiguous_zero_based_idx",
        },
        "selection": {
            "algorithm": "ascending_hmac_sha256_private_secret_and_official_item_id_v1",
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "selection_secret_persisted_publicly": False,
            "selected_count": SELECTED_COUNT,
            "block_order": list(BLOCK_ORDER),
            "block_counts": BLOCK_COUNTS,
            "manual_item_selection": False,
            "outcome_conditioned_selection": False,
        },
        "access_contract": {
            "formation_blocks": ["F1", "F2", "F3", "A_form"],
            "measurement_blocks": ["M1", "M2", "M3", "A_hold"],
            "formation_blocks_may_open_only_after_acquisition": True,
            "measurement_blocks_may_open_only_after_complete_pre_run_freeze": True,
            "all_eight_blocks_formed_before_any_formation_or_measurement": True,
            "retry_replay_resample": 0,
        },
        "l4_protocol": {
            "arms": ["empty", "P", "Q", "P_plus_Q"],
            "same_item_set_and_budget": True,
            "fixed_operator_slots": 2,
            "fusion": {"algorithm": "reciprocal_rank_fusion", "k": RRF_K},
            "top_k": TOP_K,
            "primary_metric": "official_support_recall_at_5",
            "retention_estimand": "Y(P_plus_Q)-Y(Q)",
            "novelty_estimand": "Y(P_plus_Q)-Y(P)",
            "forgetting_estimand": (
                "support_hits(P)_not_in_support_hits(P_plus_Q)/all_supports"
            ),
            "repair_participates": False,
        },
        "l5_protocol": {
            "challenger_formation_partition": "A_form_only",
            "anchor_partition": "A_hold_untouched_until_challenger_freeze",
            "anchor_source": "official_third_party_support_labels",
            "anchor_confidence": 0.9,
            "anchor_transition_policy": (
                "strict_wilson_lower_bound_improvement_v1"
            ),
            "finite_evaluator_dsl": True,
            "selective_score_dependency_invalidation": True,
            "independent_official_objective_preserved": True,
            "cached_output_reevaluation_without_agent_rerun": True,
            "prospective_search_utility_partition": "F3_to_M3",
        },
        "comparison": {
            "homologous_retrieval_arms": [
                "canonical_order_raw",
                "recursive_typed_retrieval",
                "official_hipporag_retrieve_only",
            ],
            "all_arms_share_question_corpus_top_k_and_offline_labels": True,
            "maximum_parallelism": "all_precommitted_item_arm_units",
            "online_evaluator_calls": 0,
            "generator_calls_in_primary_retrieval_study": 0,
        },
        "claim_boundary": {
            "primary_scope": "MuSiQue_official_DEV_private_HMAC_retrieval_only",
            "public_benchmark_pretraining_cannot_be_excluded": True,
            "answer_generation_claim": False,
            "family_out_claim": False,
            "performance_claim_before_measurement": False,
        },
        "implementation": _implementation_binding(project),
        "safety": {
            "dataset_rows_read": 0,
            "model_calls": 0,
            "network_calls": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
            "prior_closed_cohort_accessed": False,
        },
    }
    _assert_safe_public_payload(payload)
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def _verify_preregistration(
    path: Path,
    project: Path,
    *,
    official_repository: Path,
    selection_secret_path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    declared = body.pop("preregistration_sha256", None)
    if (
        payload.get("schema") != PREREGISTRATION_SCHEMA
        or stable_hash(body) != declared
    ):
        raise MuSiQueRecursiveAcquisitionError(
            "preregistration self-hash mismatch"
        )
    expected = build_preregistration(
        project=project,
        official_repository=official_repository,
        selection_secret_path=selection_secret_path,
    )
    if payload != expected:
        raise MuSiQueRecursiveAcquisitionError(
            "preregistration differs from the complete live protocol"
        )
    return payload


def acquire_private_pack(
    *,
    project: Path,
    preregistration_path: Path,
    official_repository: Path,
    source_archive: Path,
    private_root: Path,
    private_locator_path: Path,
    selection_secret_path: Path,
) -> dict[str, Any]:
    project = project.resolve(strict=True)
    preregistration = _verify_preregistration(
        preregistration_path,
        project,
        official_repository=official_repository,
        selection_secret_path=selection_secret_path,
    )
    preregistration_mtime_ns = preregistration_path.stat().st_mtime_ns
    source_archive = _assert_git_ignored_private_path(
        project=project, path=source_archive, require_file=True
    )
    private_root = _assert_git_ignored_private_path(
        project=project, path=private_root, require_file=False
    )
    private_locator_path = _assert_git_ignored_private_path(
        project=project, path=private_locator_path, require_file=False
    )
    secret = _read_selection_secret(
        project=project, path=selection_secret_path
    )
    if not hmac.compare_digest(
        _selection_secret_commitment(secret),
        str(
            preregistration["selection"][
                "selection_secret_commitment_sha256"
            ]
        ),
    ):
        raise MuSiQueRecursiveAcquisitionError(
            "selection secret does not match preregistration"
        )
    if _sha256_file(source_archive) != OFFICIAL_ARCHIVE_SHA256:
        raise MuSiQueRecursiveAcquisitionError("official archive hash mismatch")
    if private_root.exists() or private_locator_path.exists():
        raise FileExistsError("private output already exists")
    private_root.mkdir(parents=True, mode=0o700)
    if stat.S_IMODE(private_root.stat().st_mode) != 0o700:
        os.chmod(private_root, 0o700)

    with zipfile.ZipFile(source_archive) as archive:
        member = _find_exact_member(archive)
        source_raw = archive.read(member)
    source_rows = 0
    eligible: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_row in _iter_source_rows(source_raw):
        source_rows += 1
        row = _normalize_source_row(raw_row)
        if row is None:
            continue
        item_id = str(row["item_id"])
        if item_id in seen_ids:
            raise MuSiQueRecursiveAcquisitionError(
                "duplicate eligible official item ID"
            )
        seen_ids.add(item_id)
        eligible.append(row)
    if len(eligible) < SELECTED_COUNT:
        raise MuSiQueRecursiveAcquisitionError(
            "insufficient eligible official DEV rows"
        )
    eligible.sort(
        key=lambda row: (
            _selection_key(str(row["item_id"]), secret),
            str(row["item_id"]),
        )
    )
    selected = eligible[:SELECTED_COUNT]

    block_files: list[dict[str, Any]] = []
    cursor = 0
    for block_name in BLOCK_ORDER:
        rows = [
            {
                "schema": PRIVATE_ROW_SCHEMA,
                "block": block_name,
                **row,
            }
            for row in selected[cursor : cursor + BLOCK_COUNT]
        ]
        cursor += BLOCK_COUNT
        path = private_root / f"{block_name}.jsonl"
        file_sha256 = _write_jsonl_exclusive(path, rows)
        block_files.append(
            {
                "block": block_name,
                "count": len(rows),
                "file_sha256": file_sha256,
                "item_commitment_set_sha256": stable_hash(
                    [stable_hash(row) for row in rows]
                ),
            }
        )
    private_pack_sha256 = stable_hash(block_files)
    preregistration_preceded_blocks = all(
        preregistration_mtime_ns
        < (private_root / f"{block_name}.jsonl").stat().st_mtime_ns
        for block_name in BLOCK_ORDER
    )
    if not preregistration_preceded_blocks:
        raise MuSiQueRecursiveAcquisitionError(
            "local filesystem does not show preregistration before block formation"
        )
    locator_payload = {
        "schema": f"{VERSION}_private_locator",
        "private_root": str(private_root),
        "private_pack_sha256": private_pack_sha256,
        "block_files": block_files,
        "selection_secret_included": False,
    }
    _write_json_exclusive(
        private_locator_path,
        locator_payload,
        hash_field="locator_sha256",
    )

    receipt: dict[str, Any] = {
        "schema": ACQUISITION_SCHEMA,
        "decision": "fresh_private_pack_formed_no_formation_or_measurement_authority",
        "source": {
            "repository": preregistration["source"]["repository"],
            "commit": preregistration["source"]["commit"],
            "dataset": "MuSiQue-Answerable v1.0",
            "source_split": "official_dev",
            "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "official_dev_member_sha256": _sha256_bytes(source_raw),
            "split_disjoint_from_prior_official_train_cohort": True,
        },
        "counts": {
            "source_rows": source_rows,
            "eligible_rows": len(eligible),
            "selected_rows": len(selected),
            "blocks": BLOCK_COUNTS,
            "oracle_disagreements": 0,
        },
        "commitments": {
            "private_pack_sha256": private_pack_sha256,
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "block_files": block_files,
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "ordering": {
            "preregistration_sha256": preregistration[
                "preregistration_sha256"
            ],
            "all_eight_blocks_formed_together": True,
            "formation_or_measurement_before_pack_complete": False,
            "preregistration_preceded_block_files_local_mtime": True,
            "ordering_evidence_scope": "local_filesystem_only",
        },
        "private_boundary": {
            "source_archive_git_ignored": True,
            "selection_secret_git_ignored": True,
            "private_pack_git_ignored": True,
            "private_locator_git_ignored": True,
            "secret_free_private_locator_formed": True,
            "private_locator_path_persisted_publicly": False,
        },
        "safety": {
            "model_calls": 0,
            "network_calls_during_acquisition": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
            "prior_closed_cohort_accessed": False,
            "measurement_blocks_scored": 0,
        },
    }
    _assert_safe_public_payload(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    secret = subparsers.add_parser("generate-secret")
    secret.add_argument("--project", type=Path, required=True)
    secret.add_argument("--output", type=Path, required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--project", type=Path, required=True)
    preregister.add_argument("--official-repository", type=Path, required=True)
    preregister.add_argument("--selection-secret", type=Path, required=True)
    preregister.add_argument("--output", type=Path, required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--project", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--official-repository", type=Path, required=True)
    acquire.add_argument("--source-archive", type=Path, required=True)
    acquire.add_argument("--private-root", type=Path, required=True)
    acquire.add_argument("--private-locator", type=Path, required=True)
    acquire.add_argument("--selection-secret", type=Path, required=True)
    acquire.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "generate-secret":
        commitment = generate_selection_secret(
            project=arguments.project,
            output=arguments.output,
        )
        print(json.dumps({"selection_secret_commitment_sha256": commitment}))
        return 0
    if arguments.command == "preregister":
        payload = build_preregistration(
            project=arguments.project,
            official_repository=arguments.official_repository,
            selection_secret_path=arguments.selection_secret,
        )
        _write_json_exclusive(
            arguments.output,
            payload,
            hash_field="preregistration_sha256",
        )
        return 0
    receipt = acquire_private_pack(
        project=arguments.project,
        preregistration_path=arguments.preregistration,
        official_repository=arguments.official_repository,
        source_archive=arguments.source_archive,
        private_root=arguments.private_root,
        private_locator_path=arguments.private_locator,
        selection_secret_path=arguments.selection_secret,
    )
    _write_json_exclusive(
        arguments.output,
        receipt,
        hash_field="acquisition_sha256",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""Finite execution-freeze builder for the preregistered replication-B run.

This module is deliberately separate from :mod:`.freeze`.  The historical
``financial_sec13f_contract_execution_freeze_v2`` validator therefore keeps
its original study-specific semantics, while this builder emits a
runner-compatible manifest with an additional replication-B profile marker.

The builder accepts no substitutable study inputs: preregistration,
acquisition, formation, measurement view, materialization, prewarm, paper
protocol, and Plus-provider evidence all have fixed project-relative paths.
It never opens a private pack or a gold file and never parses expected-output
bytes.  The inherited hygienic tree validator may hash verifier files to bind
the already-materialized benchmark, but no verifier payload is copied into the
freeze.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
)
from assumption_agent.models import stable_hash

from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    read_json,
    verify_measurement_view,
    write_json,
)

from . import freeze as _base
from .provider import build_execution_provider_binding_v1
from .treatment import (
    FixedContractCandidateV2,
    build_evaluation_treatment_v2,
    load_fixed_contract_candidate_v2,
    validate_evaluation_treatment_v2,
)


ContractFreezeError = _base.ContractFreezeError

# ``runner.py`` intentionally accepts this stable wire version.  The distinct
# profile field below prevents the historical study validator from accepting a
# replication-B freeze and vice versa.
EXECUTION_FREEZE_VERSION = _base.EXECUTION_FREEZE_VERSION
REPLICATION_B_EXECUTION_FREEZE_PROFILE_VERSION = (
    "financial_sec13f_contract_v2_replication_b_execution_freeze_v1"
)
REPLICATION_B_STUDY_ID = (
    "financial-sec13f-contract-v2-replication-b-2025q4-to-2026q1"
)
REPLICATION_B_PREREGISTRATION_VERSION = (
    "financial_sec13f_contract_v2_replication_b_preregistration_v1"
)
REPLICATION_B_FORMATION_VERSION = (
    "financial_sec13f_contract_v2_replication_b_pack_formation_v1"
)

PREREGISTRATION_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_replication_b_preregistration_v1.json"
)
ACQUISITION_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_fresh_acquisition_v1.json"
)
FORMATION_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_replication_b_pack_formation_v1.json"
)
MEASUREMENT_VIEW_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_replication_b_measurement_view_v1.json"
)
BENCHMARK_ROOT_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_replication_b_private/"
    "measurement_benchmark"
)
MATERIALIZATION_RELATIVE_PATH = (
    BENCHMARK_ROOT_RELATIVE_PATH + "/measurement.materialization.json"
)
PREWARM_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_replication_b_private/"
    "prewarm/measurement.prewarm.json"
)
PLUS_IDENTITY_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_fresh_private/"
    "provider/plus/identity.sidecar.json"
)
PLUS_CANARY_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_fresh_private/"
    "provider/plus/canary.report.json"
)
PLUS_EVENTS_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_fresh_private/"
    "provider/plus/canary.events.jsonl"
)
PLUS_SELECTION_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_fresh_private/"
    "provider/selection.receipt.json"
)

PREREGISTRATION_FILE_SHA256 = (
    "7eb0a5b0f8c67c6ac0a1335a4ea933e4040d3f36691a40709686b11324f6c81d"
)
PREREGISTRATION_MANIFEST_HASH = (
    "1be0c692feeaf16b2769eb928e0236f1a7038535759067ccfeb3c21c5663aab7"
)
ACQUISITION_FILE_SHA256 = (
    "0d5629a8fe7360b76bc7343aa0064d58743a8ce0fb9552ff505705b1b2806e39"
)
ACQUISITION_RECEIPT_HASH = (
    "0f19907600a5e1eb38e987f6ccbb3e28d2285de72eaafc0e19fa505432e815ee"
)
ARCHIVE_SET_HASH = (
    "d7261ed659f54408600d422a58996d65030826b10828cb1a4d0064f834ca966d"
)
FORMATION_FILE_SHA256 = (
    "5316f31bc12ba4a3a7685733dc8c335be185bd8c1324aa2b197954ef50fa2316"
)
FORMATION_RECEIPT_HASH = (
    "a7ed34d3f0ce783b0884bfed8d58c4acedc4b335a5578c238f9baa2eb40cc3f1"
)
MEASUREMENT_VIEW_FILE_SHA256 = (
    "d986cdf5bb41041c55b4074f7ecb2212b8b36cd3a6e0882c99b7f7b58696e8f3"
)
MEASUREMENT_VIEW_HASH = (
    "fcc14ddb0a171685a74715e0a0297bb5e1bb39a334df2a1f2d3527cfd7c0b61f"
)
PRIVATE_PACK_HASH = (
    "160f6c8112b9a211dc0173fc2c546ba60694a3b71bcaa1d228afbd08b2b0ba7e"
)
SELECTION_SEED = (
    "assumption-agent-financial-sec13f-contract-v2-replication-b-20260716"
)
FIX_COMMIT = "8f1ea16293b387ae3dc3b1e514ab08241e579dbc"
FIXED_RUNNER_SHA256 = (
    "80638b1f9cc3692cbc7cba534afe6cf154cd9d07f4cdac79094cfa68ae55afd9"
)
CANDIDATE_ID = (
    "5230094b7838e6a542db6c1fb7d7e067a76820282e8e56a5f38071a8735f19ad"
)
CANDIDATE_ASSET_MANIFEST_HASH = (
    "aff2f3e2424d948d5047d92befed7578e245c3d6baf5946c256cd498c8e2ae51"
)
CANDIDATE_OPERATOR_SOURCE_SHA256 = (
    "882dac414dd30b9df88ea4130fdfd0d774db58f7c347104d4fe997e9a618887d"
)
FAILED_STUDY_AMENDMENT_HASH = (
    "2c990218f0927eda1ba42f977e5d60a17eeffb5dd1b5c24090cd712b32dd308b"
)

_EXPECTED_PROTOCOL_ID = (
    "assumption-agent-v2-skilllearn-paper-v3.20-offline86-ruoli-gpt54mini"
)
_EXPECTED_MODEL = "gpt-5.4-mini"
_EXPECTED_AGENT = "codex"
_EXPECTED_MAX_STEPS = 100

__all__ = [
    "ContractFreezeError",
    "EXECUTION_FREEZE_VERSION",
    "REPLICATION_B_EXECUTION_FREEZE_PROFILE_VERSION",
    "REPLICATION_B_STUDY_ID",
    "build_replication_b_execution_freeze_v1",
    "load_replication_b_execution_freeze_v1",
    "validate_replication_b_execution_freeze_v1",
]


def _expect_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractFreezeError(f"{label} is malformed")
    return value


def _fixed_committed_json(
    project: Path,
    *,
    relative_path: str,
    expected_file_sha256: str,
    label: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    binding = _base._committed_file_binding(
        project,
        relative_path,
        label=label,
    )
    if (
        binding.get("relative_path") != relative_path
        or binding.get("file_sha256") != expected_file_sha256
    ):
        raise ContractFreezeError(f"{label} fixed identity drifted")
    payload = read_json(
        _base._project_path(project, relative_path, label=label)
    )
    return binding, payload


def _validate_replication_b_preregistration(
    value: Mapping[str, Any],
) -> str:
    expected_fields = {
        "analysis_policy",
        "candidate_freeze",
        "evidence_boundary",
        "exclusion_commitment_views",
        "failed_study_amendment",
        "formation_source_closure",
        "infrastructure_fix",
        "manifest_hash",
        "manifest_version",
        "measurement_execution",
        "pack",
        "preregistered_at_git_commit",
        "purpose",
        "source_acquisition",
        "study_id",
    }
    declared = _base._require_self_hash(
        value,
        field="manifest_hash",
        label="replication-B preregistration",
    )
    expected_analysis = {
        "controls_before_valid_measurement_authorized": False,
        "descriptive_paired_fold_report_required": True,
        "family_out_before_valid_measurement_authorized": False,
        "invalid_pair_replacement_authorized": False,
        "performance_gate_bound": False,
        "performance_threshold_bound": False,
        "promotion_authorized_by_preregistration": False,
        "promotion_requires_separate_post_measurement_decision": True,
        "resampling_authorized": False,
        "sealed_before_promotion_authorized": False,
    }
    expected_execution = {
        "all_futures_submitted_before_results_read": True,
        "arms": ["raw", "candidate"],
        "hipporag_proxy_substitution_used": False,
        "model_inference_slots": 16,
        "model_replay_authorized": False,
        "official_hipporag": False,
        "offline_evaluation_only": True,
        "online_judge_calls": 0,
        "outer_workers": 16,
        "physical_calls": 16,
        "provider_policy": (
            "plus_fixed_if_complete_response_already_bound_otherwise_"
            "pro_only_after_plus_unavailable"
        ),
        "retries": 0,
    }
    expected_boundary = {
        "gold_formed": False,
        "model_calls": 0,
        "new_measurement_items_read": False,
        "new_pack_formed": False,
        "new_sealed_content_read": False,
        "online_judge_calls": 0,
        "secret_value_persisted": False,
    }
    expected_candidate = {
        "asset_manifest_hash": CANDIDATE_ASSET_MANIFEST_HASH,
        "candidate_commit": "7738b348abc06d319f337c9a925dda692e980349",
        "candidate_content_changed_after_failed_study": False,
        "candidate_id": CANDIDATE_ID,
        "operator_source_sha256": CANDIDATE_OPERATOR_SOURCE_SHA256,
    }
    source = _expect_mapping(
        value.get("source_acquisition"),
        label="replication-B source acquisition",
    )
    pack = _expect_mapping(value.get("pack"), label="replication-B pack")
    failure = _expect_mapping(
        value.get("failed_study_amendment"),
        label="failed-study amendment",
    )
    fix = _expect_mapping(
        value.get("infrastructure_fix"), label="infrastructure fix"
    )
    exclusions = value.get("exclusion_commitment_views")
    if (
        set(value) != expected_fields
        or declared != PREREGISTRATION_MANIFEST_HASH
        or value.get("manifest_version")
        != REPLICATION_B_PREREGISTRATION_VERSION
        or value.get("study_id") != REPLICATION_B_STUDY_ID
        or value.get("purpose")
        != "fixed_candidate_untouched_remeasurement_after_infrastructure_failure_v1"
        or value.get("analysis_policy") != expected_analysis
        or value.get("measurement_execution") != expected_execution
        or value.get("evidence_boundary") != expected_boundary
        or value.get("candidate_freeze") != expected_candidate
        or source.get("relative_path") != ACQUISITION_RELATIVE_PATH
        or source.get("file_sha256") != ACQUISITION_FILE_SHA256
        or source.get("receipt_hash") != ACQUISITION_RECEIPT_HASH
        or source.get("archive_set_hash") != ARCHIVE_SET_HASH
        or source.get("archive_content_changed") is not False
        or source.get("redownload_required") is not False
        or pack.get("measurement_count") != 8
        or pack.get("measurement_fold_count") != 4
        or pack.get("measurement_items_per_fold") != 2
        or pack.get("sealed_count") != 4
        or pack.get("selection_seed") != SELECTION_SEED
        or pack.get("resplit_authorized") is not False
        or pack.get("collision_with_every_exclusion_view_forbidden") is not True
        or failure.get("amendment_hash") != FAILED_STUDY_AMENDMENT_HASH
        or failure.get("performance_result_reused") is not False
        or failure.get("prior_model_calls_replayed") is not False
        or fix.get("fix_commit") != FIX_COMMIT
        or fix.get("fix_scope") != "bound_planner_asset_path_wiring_only_v1"
        or fix.get("candidate_asset_changed") is not False
        or fix.get("candidate_operator_source_changed") is not False
        or not isinstance(exclusions, list)
        or len(exclusions) != 2
        or any(
            not isinstance(row, Mapping)
            or row.get("private_pack_accessed") is not False
            or row.get("sealed_content_accessed") is not False
            or row.get("measurement_item_count") != 8
            or row.get("sealed_commitment_count") != 4
            for row in exclusions or ()
        )
    ):
        raise ContractFreezeError("replication-B preregistration drifted")
    return declared


def _validate_inherited_acquisition(
    value: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
) -> str:
    declared = _base._require_self_hash(
        value,
        field="receipt_hash",
        label="inherited SEC acquisition receipt",
    )
    prereg_source = _expect_mapping(
        preregistration.get("source_acquisition"),
        label="preregistered source acquisition",
    )
    prereg_archives = prereg_source.get("archives")
    observed_archives = value.get("archives")
    identity_fields = (
        "role",
        "archive_sha256",
        "size_bytes",
        "coverpage_sha256",
        "infotable_sha256",
        "source_fingerprint",
        "source_url",
        "source_path_persisted",
    )
    def identities(rows: object) -> list[dict[str, Any]]:
        if not isinstance(rows, list) or len(rows) != 2:
            raise ContractFreezeError("SEC acquisition archive set is malformed")
        if not all(isinstance(row, Mapping) for row in rows):
            raise ContractFreezeError("SEC acquisition archive row is malformed")
        return [
            {field: row.get(field) for field in identity_fields}
            for row in rows
        ]

    if (
        declared != ACQUISITION_RECEIPT_HASH
        or value.get("archive_set_hash") != ARCHIVE_SET_HASH
        or prereg_source.get("receipt_hash") != declared
        or prereg_source.get("archive_set_hash") != ARCHIVE_SET_HASH
        or identities(observed_archives) != identities(prereg_archives)
        or value.get("model_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("resampling_used") is not False
        or value.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("inherited SEC acquisition drifted")
    return declared


def _validate_view_identity(value: Mapping[str, Any]) -> None:
    items = value.get("measurement_items")
    folds = (
        [int(row.get("fold")) for row in items]
        if isinstance(items, list)
        and all(isinstance(row, Mapping) for row in items)
        else []
    )
    if (
        value.get("measurement_view_hash") != MEASUREMENT_VIEW_HASH
        or value.get("private_pack_hash") != PRIVATE_PACK_HASH
        or value.get("measurement_item_count") != 8
        or value.get("sealed_item_count") != 4
        or len(items or ()) != 8
        or folds != [0, 0, 1, 1, 2, 2, 3, 3]
        or value.get("ground_truth_persisted") is not False
        or value.get("sealed_content_persisted") is not False
        or value.get("model_calls") != 0
        or value.get("network_calls") != 0
    ):
        raise ContractFreezeError("replication-B measurement view drifted")


def _validate_collision_audit(
    value: Mapping[str, Any],
    *,
    exclusions: Mapping[str, Mapping[str, Any]],
) -> None:
    body = dict(value)
    declared = body.pop("binding_hash", None)
    collision = _expect_mapping(
        value.get("collision_audit"), label="collision audit"
    )
    collision_body = dict(collision)
    audit_hash = collision_body.pop("audit_hash", None)
    exclusion_hash = value.get("exclusion_measurement_view_hash")
    excluded = exclusions.get(str(exclusion_hash))
    if (
        declared != payload_hash(body)
        or audit_hash != payload_hash(collision_body)
        or excluded is None
        or value.get("exclusion_view_relative_path")
        != excluded.get("relative_path")
        or value.get("exclusion_view_file_sha256")
        != excluded.get("file_sha256")
        or collision.get("policy")
        != "old_new_query_instruction_commitments_disjoint_v1"
        or collision.get("prior_instruction_commitment_count") != 12
        or collision.get("prior_query_commitment_count") != 12
        or collision.get("new_instruction_commitment_count") != 12
        or collision.get("new_query_commitment_count") != 12
        or collision.get("instruction_collision_count") != 0
        or collision.get("query_collision_count") != 0
        or collision.get("prior_private_pack_accessed") is not False
        or collision.get("prior_sealed_content_accessed") is not False
    ):
        raise ContractFreezeError("replication-B collision audit drifted")


def _validate_replication_b_formation(
    value: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    acquisition_receipt_hash: str,
    measurement_view: Mapping[str, Any],
) -> str:
    expected_fields = {
        "all_prior_query_and_instruction_commitments_disjoint",
        "exclusion_collision_audit_set_hash",
        "exclusion_collision_audits",
        "formation_order",
        "formation_source_closure",
        "gold_formed",
        "measurement_count",
        "measurement_view_file_sha256",
        "measurement_view_hash",
        "model_calls",
        "network_calls",
        "new_sealed_content_persisted_in_receipt",
        "online_judge_calls",
        "oracle_calls",
        "preregistration",
        "prior_private_pack_accessed",
        "prior_sealed_content_accessed",
        "private_pack_file_sha256",
        "private_pack_hash",
        "private_pack_path_persisted",
        "receipt_hash",
        "receipt_version",
        "sealed_commitment_count",
        "secret_value_persisted",
        "selection_seed",
        "source_acquisition",
        "study_id",
    }
    declared = _base._require_self_hash(
        value,
        field="receipt_hash",
        label="replication-B formation receipt",
    )
    prereg_binding = _expect_mapping(
        value.get("preregistration"), label="formation preregistration binding"
    )
    acquisition = _expect_mapping(
        value.get("source_acquisition"),
        label="formation acquisition binding",
    )
    formation_order = _expect_mapping(
        value.get("formation_order"), label="formation order"
    )
    audits = value.get("exclusion_collision_audits")
    prereg_exclusions = preregistration.get("exclusion_commitment_views")
    exclusions = {
        str(row.get("measurement_view_hash")): row
        for row in prereg_exclusions or ()
        if isinstance(row, Mapping)
    }
    if not isinstance(audits, list) or len(audits) != 2:
        raise ContractFreezeError("replication-B collision audit set drifted")
    for audit in audits:
        _validate_collision_audit(
            _expect_mapping(audit, label="collision audit row"),
            exclusions=exclusions,
        )
    prereg_ctime = formation_order.get("preregistration_file_ctime_ns")
    pack_ctime = formation_order.get("private_pack_file_ctime_ns")
    if (
        set(value) != expected_fields
        or declared != FORMATION_RECEIPT_HASH
        or value.get("receipt_version") != REPLICATION_B_FORMATION_VERSION
        or value.get("study_id") != REPLICATION_B_STUDY_ID
        or value.get("measurement_view_hash") != MEASUREMENT_VIEW_HASH
        or value.get("measurement_view_file_sha256")
        != MEASUREMENT_VIEW_FILE_SHA256
        or value.get("private_pack_hash") != PRIVATE_PACK_HASH
        or value.get("measurement_count") != 8
        or value.get("sealed_commitment_count") != 4
        or value.get("selection_seed") != SELECTION_SEED
        or measurement_view.get("measurement_view_hash")
        != value.get("measurement_view_hash")
        or measurement_view.get("private_pack_hash")
        != value.get("private_pack_hash")
        or prereg_binding.get("relative_path") != PREREGISTRATION_RELATIVE_PATH
        or prereg_binding.get("file_sha256") != PREREGISTRATION_FILE_SHA256
        or prereg_binding.get("manifest_hash")
        != PREREGISTRATION_MANIFEST_HASH
        or acquisition.get("relative_path") != ACQUISITION_RELATIVE_PATH
        or acquisition.get("file_sha256") != ACQUISITION_FILE_SHA256
        or acquisition.get("receipt_hash") != acquisition_receipt_hash
        or acquisition.get("archive_set_hash") != ARCHIVE_SET_HASH
        or acquisition.get("redownloaded") is not False
        or value.get("formation_source_closure")
        != preregistration.get("formation_source_closure")
        or value.get("all_prior_query_and_instruction_commitments_disjoint")
        is not True
        or value.get("exclusion_collision_audit_set_hash")
        != payload_hash(audits)
        or formation_order.get("policy")
        != "replication_preregistration_inode_precedes_private_pack_v1"
        or formation_order.get("formation_after_preregistration") is not True
        or not isinstance(prereg_ctime, int)
        or isinstance(prereg_ctime, bool)
        or not isinstance(pack_ctime, int)
        or isinstance(pack_ctime, bool)
        or pack_ctime <= prereg_ctime
        or value.get("gold_formed") is not False
        or value.get("oracle_calls") != 0
        or value.get("model_calls") != 0
        or value.get("network_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("private_pack_path_persisted") is not False
        or value.get("prior_private_pack_accessed") is not False
        or value.get("prior_sealed_content_accessed") is not False
        or value.get("new_sealed_content_persisted_in_receipt") is not False
        or value.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("replication-B formation receipt drifted")
    return declared


def _validate_candidate_against_preregistration(
    candidate: FixedContractCandidateV2,
    preregistration: Mapping[str, Any],
) -> None:
    frozen = _expect_mapping(
        preregistration.get("candidate_freeze"),
        label="preregistered candidate",
    )
    if (
        candidate.candidate_id != frozen.get("candidate_id")
        or candidate.asset_manifest_hash != frozen.get("asset_manifest_hash")
        or candidate.operator_source_sha256
        != frozen.get("operator_source_sha256")
    ):
        raise ContractFreezeError("fixed candidate differs from preregistration")


def _validate_infrastructure_fix_in_source_closure(
    source_closure: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> None:
    rows = source_closure.get("files")
    by_path = {
        str(row.get("relative_path")): row.get("file_sha256")
        for row in rows or ()
        if isinstance(row, Mapping)
    }
    fix = _expect_mapping(
        preregistration.get("infrastructure_fix"), label="infrastructure fix"
    )
    changed = fix.get("changed_files")
    expected_after = {
        str(row.get("relative_path")): row.get("after_sha256")
        for row in changed or ()
        if isinstance(row, Mapping)
    }
    runner_path = "replication_runtime/financial_sec13f_contract_v2/runner.py"
    if (
        fix.get("fix_commit") != FIX_COMMIT
        or expected_after.get(runner_path) != FIXED_RUNNER_SHA256
        or by_path.get(runner_path) != FIXED_RUNNER_SHA256
    ):
        raise ContractFreezeError(
            "replication-B infrastructure fix is absent from source closure"
        )


def _validate_plus_provider(
    provider: Mapping[str, Any],
    *,
    protocol_binding: Mapping[str, Any],
) -> None:
    if (
        provider.get("provider_label") != "plus"
        or provider.get("model") != _EXPECTED_MODEL
        or provider.get("model") != protocol_binding.get("model")
        or provider.get("api_origin") != "https://ruoli.dev"
        or provider.get("plus_transport_failure_before_pro_selection")
        is not False
        or provider.get("selected_provider_fixed_for_complete_batch")
        is not True
        or provider.get("mid_batch_provider_switch_authorized") is not False
        or provider.get("mid_batch_retry_authorized") is not False
        or provider.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("fixed Plus provider binding drifted")


def _assert_exact_replication_b_plan(safe: Mapping[str, Any]) -> None:
    work_units = safe.get("work_units")
    if not isinstance(work_units, list) or len(work_units) != 16:
        raise ContractFreezeError("replication-B plan is not a 16-unit grid")
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for row in work_units:
        if not isinstance(row, Mapping):
            raise ContractFreezeError("replication-B work unit is malformed")
        pairs.setdefault(str(row.get("pair_id")), []).append(row)
    if (
        safe.get("physical_work_unit_count") != 16
        or safe.get("measurement_pair_count") != 8
        or safe.get("raw_execution_count") != 8
        or safe.get("candidate_execution_count") != 8
        or safe.get("official_hipporag") is not False
        or safe.get("official_hipporag_execution_count") != 0
        or safe.get("projection_count") != 0
        or safe.get("maximum_workers") != 16
        or safe.get("retry_count") != 0
        or safe.get("retry_policy") != "none"
        or safe.get("descriptive_only") is not True
        or safe.get("performance_gate_bound") is not False
        or safe.get("promotion_authorized") is not False
        or len(pairs) != 8
        or any(
            {str(row.get("arm")) for row in rows} != {"raw", "candidate"}
            or len(rows) != 2
            or any(row.get("retry_count") != 0 for row in rows)
            or any(row.get("raw_content_persisted") is not False for row in rows)
            or any(
                row.get("candidate_source_required")
                is not (str(row.get("arm")) == "candidate")
                for row in rows
            )
            for rows in pairs.values()
        )
    ):
        raise ContractFreezeError("replication-B exact paired plan drifted")


def _replication_binding(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    fix = _expect_mapping(
        preregistration.get("infrastructure_fix"), label="infrastructure fix"
    )
    failure = _expect_mapping(
        preregistration.get("failed_study_amendment"),
        label="failed-study amendment",
    )
    return {
        "purpose": preregistration["purpose"],
        "failed_study_amendment_hash": failure["amendment_hash"],
        "infrastructure_fix_commit": fix["fix_commit"],
        "infrastructure_fix_scope": fix["fix_scope"],
        "prior_performance_result_reused": False,
        "prior_model_calls_replayed": False,
        "invalid_pair_replacement_authorized": False,
        "resampling_authorized": False,
    }


def _assemble_replication_b_execution_freeze_v1(
    *,
    project: Path,
    provider_env_file: str | Path,
    source_closure: Mapping[str, Any],
) -> tuple[dict[str, Any], FixedContractCandidateV2]:
    prereg_binding, preregistration = _fixed_committed_json(
        project,
        relative_path=PREREGISTRATION_RELATIVE_PATH,
        expected_file_sha256=PREREGISTRATION_FILE_SHA256,
        label="replication-B preregistration",
    )
    preregistration_hash = _validate_replication_b_preregistration(
        preregistration
    )
    acquisition_binding, acquisition = _fixed_committed_json(
        project,
        relative_path=ACQUISITION_RELATIVE_PATH,
        expected_file_sha256=ACQUISITION_FILE_SHA256,
        label="inherited acquisition receipt",
    )
    acquisition_hash = _validate_inherited_acquisition(
        acquisition,
        preregistration=preregistration,
    )
    view_binding, raw_view = _fixed_committed_json(
        project,
        relative_path=MEASUREMENT_VIEW_RELATIVE_PATH,
        expected_file_sha256=MEASUREMENT_VIEW_FILE_SHA256,
        label="replication-B measurement view",
    )
    view = verify_measurement_view(raw_view)
    _validate_view_identity(view)
    formation_binding, formation = _fixed_committed_json(
        project,
        relative_path=FORMATION_RELATIVE_PATH,
        expected_file_sha256=FORMATION_FILE_SHA256,
        label="replication-B formation receipt",
    )
    formation_hash = _validate_replication_b_formation(
        formation,
        preregistration=preregistration,
        acquisition_receipt_hash=acquisition_hash,
        measurement_view=view,
    )

    materialization, materialization_binding, _, _ = (
        _base._validate_materialization_and_tree(
            project=project,
            benchmark_root=BENCHMARK_ROOT_RELATIVE_PATH,
            materialization_report_path=MATERIALIZATION_RELATIVE_PATH,
            measurement_view=view,
        )
    )
    prewarm_file, prewarm_relative = _base._relative_artifact(
        project,
        PREWARM_RELATIVE_PATH,
        label="replication-B prewarm report",
    )
    _, prewarm_binding = _base._validate_prewarm(
        prewarm_path=prewarm_file,
        measurement_view=view,
        materialization=materialization,
        benchmark_tree_hash=str(materialization_binding["benchmark_tree_hash"]),
    )
    prewarm_binding["relative_path"] = prewarm_relative

    protocol, protocol_binding = _base._paper_protocol_binding(
        project,
        V320_PROTOCOL_RELATIVE_PATH,
    )
    if (
        protocol_binding.get("protocol_id") != _EXPECTED_PROTOCOL_ID
        or protocol_binding.get("agent_id") != _EXPECTED_AGENT
        or protocol_binding.get("model") != _EXPECTED_MODEL
        or protocol_binding.get("max_steps") != _EXPECTED_MAX_STEPS
    ):
        raise ContractFreezeError("replication-B paper protocol drifted")

    candidate = load_fixed_contract_candidate_v2(project)
    _validate_candidate_against_preregistration(candidate, preregistration)
    candidate_payload = candidate.safe_payload(project)
    _validate_infrastructure_fix_in_source_closure(
        source_closure,
        preregistration,
    )
    evaluation = build_evaluation_treatment_v2(
        candidate=candidate,
        execution_source_closure_hash=str(source_closure["closure_hash"]),
        measurement_view_hash=str(view["measurement_view_hash"]),
        benchmark_tree_hash=str(materialization_binding["benchmark_tree_hash"]),
    )
    validate_evaluation_treatment_v2(evaluation, candidate=candidate)
    evaluator_epoch = (
        "financial-sec13f-contract-v2-replication-b-"
        + str(view["private_pack_hash"])[:12]
    )
    treatment = {
        "evaluation_binding": evaluation,
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
        "evaluator_epoch": evaluator_epoch,
    }

    provider = build_execution_provider_binding_v1(
        project_root=project,
        provider_label="plus",
        identity_sidecar_path=PLUS_IDENTITY_RELATIVE_PATH,
        selected_canary_report_path=PLUS_CANARY_RELATIVE_PATH,
        selected_event_ledger_path=PLUS_EVENTS_RELATIVE_PATH,
        selection_receipt_path=PLUS_SELECTION_RELATIVE_PATH,
        env_file=provider_env_file,
    )
    _base._validate_provider_paths_and_binding(
        project,
        provider,
        env_file=provider_env_file,
    )
    _validate_plus_provider(provider, protocol_binding=protocol_binding)

    plan_set_hash, typed_plan_set = _base._typed_plan_set(
        measurement_view=view,
        candidate=candidate,
    )
    plan = _base._build_measurement_plan(
        measurement_view=view,
        candidate=candidate,
        evaluation=evaluation,
        protocol=protocol,
        evaluator_epoch=evaluator_epoch,
    )
    safe_plan = plan.safe_payload()
    _assert_exact_replication_b_plan(safe_plan)

    body = {
        "manifest_version": EXECUTION_FREEZE_VERSION,
        "freeze_profile_version": (
            REPLICATION_B_EXECUTION_FREEZE_PROFILE_VERSION
        ),
        "study_id": REPLICATION_B_STUDY_ID,
        "replication_binding": _replication_binding(preregistration),
        "preregistration": {
            **prereg_binding,
            "manifest_hash": preregistration_hash,
            "manifest_version": REPLICATION_B_PREREGISTRATION_VERSION,
        },
        "acquisition": {
            **acquisition_binding,
            "receipt_hash": acquisition_hash,
            "receipt_version": acquisition["receipt_version"],
            "archive_set_hash": acquisition["archive_set_hash"],
        },
        "formation": {
            **formation_binding,
            "receipt_hash": formation_hash,
            "receipt_version": REPLICATION_B_FORMATION_VERSION,
            "private_pack_hash": PRIVATE_PACK_HASH,
            "private_pack_accessed_by_freeze": False,
        },
        "measurement_view": {
            **view_binding,
            "measurement_view_hash": MEASUREMENT_VIEW_HASH,
            "private_pack_hash": PRIVATE_PACK_HASH,
            "measurement_count": 8,
            "sealed_commitment_count": 4,
            "sealed_content_accessed": False,
        },
        "materialization": materialization_binding,
        "prewarm": prewarm_binding,
        "paper_protocol": protocol_binding,
        "provider": provider,
        "candidate": candidate_payload,
        "treatment": treatment,
        "precomputed_plan_set_hash": plan_set_hash,
        "typed_plan_set": typed_plan_set,
        "plan": {"plan_hash": plan.plan_hash, "safe_payload": safe_plan},
        "execution_source_closure": dict(source_closure),
        "execution": _base._execution_policy(),
        "private_pack_accessed": False,
        "gold_artifact_accessed": False,
        "expected_output_content_accessed": False,
        "sealed_content_accessed": False,
        "secret_value_persisted": False,
    }
    _base._assert_no_secret_or_raw_payload(body)
    return {**body, "manifest_hash": payload_hash(body)}, candidate


def build_replication_b_execution_freeze_v1(
    *,
    project_root: str | Path,
    provider_env_file: str | Path,
) -> dict[str, Any]:
    """Build the one fixed replication-B freeze without executing a model."""

    project = Path(project_root).expanduser().resolve(strict=True)
    source_closure = _base.build_execution_source_closure_v2(project)
    payload, _ = _assemble_replication_b_execution_freeze_v1(
        project=project,
        provider_env_file=provider_env_file,
        source_closure=source_closure,
    )
    return payload


def _expected_top_level_fields() -> set[str]:
    return {
        "manifest_version",
        "freeze_profile_version",
        "study_id",
        "replication_binding",
        "preregistration",
        "acquisition",
        "formation",
        "measurement_view",
        "materialization",
        "prewarm",
        "paper_protocol",
        "provider",
        "candidate",
        "treatment",
        "precomputed_plan_set_hash",
        "typed_plan_set",
        "plan",
        "execution_source_closure",
        "execution",
        "private_pack_accessed",
        "gold_artifact_accessed",
        "expected_output_content_accessed",
        "sealed_content_accessed",
        "secret_value_persisted",
        "manifest_hash",
    }


def validate_replication_b_execution_freeze_v1(
    value: Mapping[str, Any],
    *,
    project_root: str | Path,
    provider_env_file: str | Path,
) -> FixedContractCandidateV2:
    """Recompute every live binding and reject any self-consistent drift."""

    project = Path(project_root).expanduser().resolve(strict=True)
    body = dict(value)
    declared = body.pop("manifest_hash", None)
    if (
        set(value) != _expected_top_level_fields()
        or value.get("manifest_version") != EXECUTION_FREEZE_VERSION
        or value.get("freeze_profile_version")
        != REPLICATION_B_EXECUTION_FREEZE_PROFILE_VERSION
        or value.get("study_id") != REPLICATION_B_STUDY_ID
        or not _base._is_sha256(declared)
        or declared != payload_hash(body)
        or value.get("execution") != _base._execution_policy()
        or value.get("private_pack_accessed") is not False
        or value.get("gold_artifact_accessed") is not False
        or value.get("expected_output_content_accessed") is not False
        or value.get("sealed_content_accessed") is not False
        or value.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("replication-B execution freeze drifted")
    _base._assert_no_secret_or_raw_payload(body)
    source = _expect_mapping(
        value.get("execution_source_closure"),
        label="replication-B execution source closure",
    )
    _base.validate_execution_source_closure_v2(source, project_root=project)
    expected, candidate = _assemble_replication_b_execution_freeze_v1(
        project=project,
        provider_env_file=provider_env_file,
        source_closure=copy.deepcopy(dict(source)),
    )
    if dict(value) != expected:
        raise ContractFreezeError(
            "replication-B execution bindings changed after freeze"
        )
    return candidate


def load_replication_b_execution_freeze_v1(
    path: str | Path,
    *,
    project_root: str | Path,
    provider_env_file: str | Path,
) -> tuple[dict[str, Any], FixedContractCandidateV2]:
    freeze_path = Path(path).expanduser()
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise ContractFreezeError("replication-B execution freeze is unavailable")
    payload = read_json(freeze_path)
    candidate = validate_replication_b_execution_freeze_v1(
        payload,
        project_root=project_root,
        provider_env_file=provider_env_file,
    )
    return payload, candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("execution-freeze")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--provider-env-file", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-execution-freeze")
    verify.add_argument("--project-root", type=Path, required=True)
    verify.add_argument("--provider-env-file", type=Path, required=True)
    verify.add_argument("--execution-freeze", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "execution-freeze":
        payload = build_replication_b_execution_freeze_v1(
            project_root=args.project_root,
            provider_env_file=args.provider_env_file,
        )
        write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "manifest_hash": payload["manifest_hash"],
                    "provider_label": payload["provider"]["provider_label"],
                    "physical_calls": payload["execution"]["physical_calls"],
                    "study_id": payload["study_id"],
                },
                sort_keys=True,
            )
        )
        return 0
    payload, _ = load_replication_b_execution_freeze_v1(
        args.execution_freeze,
        project_root=args.project_root,
        provider_env_file=args.provider_env_file,
    )
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "validated": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

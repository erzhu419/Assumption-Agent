from __future__ import annotations

"""Finite execution freeze for the preregistered SEC-13F replication C."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
)

from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    read_json,
    verify_measurement_view,
    write_json,
)

from . import freeze as _base
from . import replication_b_freeze as _b
from .provider import build_execution_provider_binding_v1
from .treatment import (
    FixedContractCandidateV2,
    build_evaluation_treatment_v2,
    load_fixed_contract_candidate_v2,
    validate_evaluation_treatment_v2,
)


ContractFreezeError = _base.ContractFreezeError
EXECUTION_FREEZE_VERSION = _base.EXECUTION_FREEZE_VERSION
PROFILE_VERSION = "financial_sec13f_contract_v2_replication_c_execution_freeze_v1"
STUDY_ID = "financial-sec13f-contract-v2-replication-c-2025q4-to-2026q1"
PREREGISTRATION_VERSION = (
    "financial_sec13f_contract_v2_replication_c_preregistration_v1"
)
FORMATION_VERSION = (
    "financial_sec13f_contract_v2_replication_c_pack_formation_v1"
)

PREREGISTRATION_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_replication_c_preregistration_v1.json"
)
PREREGISTRATION_FILE_SHA256 = (
    "95a1bee00e75864d78d73aa88112629f277637d7e4fae008e2b05e200a462601"
)
PREREGISTRATION_HASH = (
    "2a9bee40bbcda9454712d7b046670591facb7c2bf1de459685019badcc2cd68b"
)
ACQUISITION_RELATIVE_PATH = _b.ACQUISITION_RELATIVE_PATH
ACQUISITION_FILE_SHA256 = _b.ACQUISITION_FILE_SHA256
ACQUISITION_HASH = _b.ACQUISITION_RECEIPT_HASH
ARCHIVE_SET_HASH = _b.ARCHIVE_SET_HASH
FORMATION_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_replication_c_pack_formation_v1.json"
)
FORMATION_FILE_SHA256 = (
    "3712486f9fe0078d52f5511dd428981dde7970729819cbdc188bd8d0897e4acb"
)
FORMATION_HASH = (
    "4e8c02d76c8083c083792f17fbae83723deec0d31277e90da679d62b2deebc71"
)
MEASUREMENT_VIEW_RELATIVE_PATH = (
    "manifests/financial_sec13f_contract_v2_replication_c_measurement_view_v1.json"
)
MEASUREMENT_VIEW_FILE_SHA256 = (
    "ad589f62d3f2ecd2202b84850716e083e6daedc122fa81c174bdd3c3c30e25c9"
)
MEASUREMENT_VIEW_HASH = (
    "02a1c2bb10f1517d5f63c53afccdd6b388b84687486593fe9bc7d376272ac7d8"
)
PRIVATE_PACK_HASH = (
    "ad3b0852e40fcf364f04bfaf97888b90ee342b1e8db273611bc14f04cac2741b"
)
SELECTION_SEED = (
    "assumption-agent-financial-sec13f-contract-v2-replication-c-20260716"
)
BENCHMARK_ROOT_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_replication_c_private/measurement_benchmark"
)
MATERIALIZATION_RELATIVE_PATH = (
    BENCHMARK_ROOT_RELATIVE_PATH + "/measurement.materialization.json"
)
PREWARM_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_replication_c_private/"
    "prewarm/measurement.prewarm.json"
)
PLUS_IDENTITY_RELATIVE_PATH = _b.PLUS_IDENTITY_RELATIVE_PATH
PLUS_CANARY_RELATIVE_PATH = _b.PLUS_CANARY_RELATIVE_PATH
PLUS_EVENTS_RELATIVE_PATH = _b.PLUS_EVENTS_RELATIVE_PATH
PLUS_SELECTION_RELATIVE_PATH = _b.PLUS_SELECTION_RELATIVE_PATH
B_INTERRUPTION_AMENDMENT_HASH = (
    "e1d7a2aef485855720f278c3da2d55f2841c97ca718a8cbbd498ef3cfc3bf223"
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractFreezeError(f"{label} is malformed")
    return value


def _validate_preregistration(value: Mapping[str, Any]) -> str:
    declared = _base._require_self_hash(
        value, field="manifest_hash", label="replication-C preregistration"
    )
    analysis = _mapping(value.get("analysis_policy"), "analysis policy")
    execution = _mapping(value.get("measurement_execution"), "measurement execution")
    pack = _mapping(value.get("pack"), "pack policy")
    candidate = _mapping(value.get("candidate_freeze"), "candidate freeze")
    boundary = _mapping(value.get("evidence_boundary"), "evidence boundary")
    interruption = _mapping(
        value.get("replication_b_interruption_amendment"),
        "replication-B interruption amendment",
    )
    source = _mapping(value.get("source_acquisition"), "source acquisition")
    exclusions = value.get("exclusion_commitment_views")
    if (
        declared != PREREGISTRATION_HASH
        or value.get("manifest_version") != PREREGISTRATION_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("purpose")
        != "fixed_candidate_second_untouched_replication_after_replication_b_interruption_v1"
        or analysis.get("performance_gate_bound") is not False
        or analysis.get("performance_threshold_bound") is not False
        or analysis.get("promotion_authorized_by_preregistration") is not False
        or analysis.get("resampling_authorized") is not False
        or analysis.get("invalid_pair_replacement_authorized") is not False
        or execution.get("physical_calls") != 16
        or execution.get("outer_workers") != 16
        or execution.get("model_inference_slots") != 16
        or execution.get("arms") != ["raw", "candidate"]
        or execution.get("retries") != 0
        or execution.get("model_replay_authorized") is not False
        or execution.get("offline_evaluation_only") is not True
        or execution.get("online_judge_calls") != 0
        or pack.get("selection_seed") != SELECTION_SEED
        or pack.get("measurement_count") != 8
        or pack.get("measurement_fold_count") != 4
        or pack.get("measurement_items_per_fold") != 2
        or pack.get("sealed_count") != 4
        or pack.get("exclusion_view_count") != 3
        or pack.get("resplit_authorized") is not False
        or pack.get("collision_with_every_exclusion_view_forbidden") is not True
        or candidate.get("candidate_id") != _b.CANDIDATE_ID
        or candidate.get("asset_manifest_hash") != _b.CANDIDATE_ASSET_MANIFEST_HASH
        or candidate.get("operator_source_sha256") != _b.CANDIDATE_OPERATOR_SOURCE_SHA256
        or candidate.get("candidate_content_changed_after_replication_b") is not False
        or interruption.get("amendment_hash") != B_INTERRUPTION_AMENDMENT_HASH
        or interruption.get("prior_model_calls_replayed") is not False
        or interruption.get("prior_claims_retried") is not False
        or interruption.get("performance_result_reused") is not False
        or source.get("relative_path") != ACQUISITION_RELATIVE_PATH
        or source.get("file_sha256") != ACQUISITION_FILE_SHA256
        or source.get("receipt_hash") != ACQUISITION_HASH
        or source.get("archive_set_hash") != ARCHIVE_SET_HASH
        or boundary.get("gold_formed") is not False
        or boundary.get("model_calls") != 0
        or boundary.get("new_pack_formed") is not False
        or boundary.get("new_sealed_content_read") is not False
        or boundary.get("replication_b_trace_content_read") is not False
        or not isinstance(exclusions, list)
        or len(exclusions) != 3
        or any(
            not isinstance(row, Mapping)
            or row.get("measurement_item_count") != 8
            or row.get("sealed_commitment_count") != 4
            or row.get("private_pack_accessed") is not False
            or row.get("sealed_content_accessed") is not False
            for row in exclusions or ()
        )
    ):
        raise ContractFreezeError("replication-C preregistration drifted")
    return declared


def _validate_view(value: Mapping[str, Any]) -> None:
    items = value.get("measurement_items")
    folds = (
        [int(row.get("fold")) for row in items]
        if isinstance(items, list) and all(isinstance(row, Mapping) for row in items)
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
        raise ContractFreezeError("replication-C measurement view drifted")


def _validate_formation(
    value: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    measurement_view: Mapping[str, Any],
) -> str:
    declared = _base._require_self_hash(
        value, field="receipt_hash", label="replication-C formation receipt"
    )
    prereg = _mapping(value.get("preregistration"), "formation preregistration")
    acquisition = _mapping(value.get("source_acquisition"), "formation acquisition")
    audits = value.get("exclusion_collision_audits")
    frozen_exclusions = preregistration.get("exclusion_commitment_views")
    if not isinstance(audits, list) or len(audits) != 3:
        raise ContractFreezeError("replication-C collision audit set drifted")
    expected = {
        str(row["measurement_view_hash"]): row
        for row in frozen_exclusions or ()
        if isinstance(row, Mapping)
    }
    for row in audits:
        if not isinstance(row, Mapping):
            raise ContractFreezeError("replication-C collision audit is malformed")
        body = dict(row)
        binding_hash = body.pop("binding_hash", None)
        collision = _mapping(row.get("collision_audit"), "collision audit")
        collision_body = dict(collision)
        audit_hash = collision_body.pop("audit_hash", None)
        prior = expected.get(str(row.get("exclusion_measurement_view_hash")))
        if (
            binding_hash != payload_hash(body)
            or audit_hash != payload_hash(collision_body)
            or prior is None
            or row.get("exclusion_view_relative_path") != prior.get("relative_path")
            or row.get("exclusion_view_file_sha256") != prior.get("file_sha256")
            or collision.get("instruction_collision_count") != 0
            or collision.get("query_collision_count") != 0
            or collision.get("prior_private_pack_accessed") is not False
            or collision.get("prior_sealed_content_accessed") is not False
        ):
            raise ContractFreezeError("replication-C collision audit drifted")
    if (
        declared != FORMATION_HASH
        or value.get("receipt_version") != FORMATION_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("selection_seed") != SELECTION_SEED
        or value.get("private_pack_hash") != PRIVATE_PACK_HASH
        or value.get("measurement_view_hash") != MEASUREMENT_VIEW_HASH
        or measurement_view.get("measurement_view_hash") != MEASUREMENT_VIEW_HASH
        or prereg.get("relative_path") != PREREGISTRATION_RELATIVE_PATH
        or prereg.get("file_sha256") != PREREGISTRATION_FILE_SHA256
        or prereg.get("manifest_hash") != PREREGISTRATION_HASH
        or acquisition.get("relative_path") != ACQUISITION_RELATIVE_PATH
        or acquisition.get("file_sha256") != ACQUISITION_FILE_SHA256
        or acquisition.get("receipt_hash") != ACQUISITION_HASH
        or acquisition.get("archive_set_hash") != ARCHIVE_SET_HASH
        or value.get("measurement_count") != 8
        or value.get("sealed_commitment_count") != 4
        or value.get("formation_after_preregistration") is not True
        or value.get("all_prior_query_and_instruction_commitments_disjoint") is not True
        or value.get("exclusion_collision_audit_set_hash") != payload_hash(audits)
        or value.get("gold_formed") is not False
        or value.get("oracle_calls") != 0
        or value.get("model_calls") != 0
        or value.get("network_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("private_pack_path_persisted") is not False
        or value.get("private_pack_content_persisted_in_receipt") is not False
        or value.get("new_sealed_content_persisted_in_receipt") is not False
        or value.get("prior_private_pack_accessed") is not False
        or value.get("prior_sealed_content_accessed") is not False
        or value.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("replication-C formation receipt drifted")
    return declared


def _assemble(
    *,
    project: Path,
    provider_env_file: str | Path,
    source_closure: Mapping[str, Any],
) -> tuple[dict[str, Any], FixedContractCandidateV2]:
    prereg_binding, preregistration = _b._fixed_committed_json(
        project,
        relative_path=PREREGISTRATION_RELATIVE_PATH,
        expected_file_sha256=PREREGISTRATION_FILE_SHA256,
        label="replication-C preregistration",
    )
    preregistration_hash = _validate_preregistration(preregistration)
    acquisition_binding, acquisition = _b._fixed_committed_json(
        project,
        relative_path=ACQUISITION_RELATIVE_PATH,
        expected_file_sha256=ACQUISITION_FILE_SHA256,
        label="inherited acquisition receipt",
    )
    acquisition_hash = _b._validate_inherited_acquisition(
        acquisition, preregistration=preregistration
    )
    view_binding, raw_view = _b._fixed_committed_json(
        project,
        relative_path=MEASUREMENT_VIEW_RELATIVE_PATH,
        expected_file_sha256=MEASUREMENT_VIEW_FILE_SHA256,
        label="replication-C measurement view",
    )
    view = verify_measurement_view(raw_view)
    _validate_view(view)
    formation_binding, formation = _b._fixed_committed_json(
        project,
        relative_path=FORMATION_RELATIVE_PATH,
        expected_file_sha256=FORMATION_FILE_SHA256,
        label="replication-C formation receipt",
    )
    formation_hash = _validate_formation(
        formation, preregistration=preregistration, measurement_view=view
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
        project, PREWARM_RELATIVE_PATH, label="replication-C prewarm report"
    )
    _, prewarm_binding = _base._validate_prewarm(
        prewarm_path=prewarm_file,
        measurement_view=view,
        materialization=materialization,
        benchmark_tree_hash=str(materialization_binding["benchmark_tree_hash"]),
    )
    prewarm_binding["relative_path"] = prewarm_relative
    protocol, protocol_binding = _base._paper_protocol_binding(
        project, V320_PROTOCOL_RELATIVE_PATH
    )
    if (
        protocol_binding.get("protocol_id") != _b._EXPECTED_PROTOCOL_ID
        or protocol_binding.get("agent_id") != _b._EXPECTED_AGENT
        or protocol_binding.get("model") != _b._EXPECTED_MODEL
        or protocol_binding.get("max_steps") != _b._EXPECTED_MAX_STEPS
    ):
        raise ContractFreezeError("replication-C paper protocol drifted")

    candidate = load_fixed_contract_candidate_v2(project)
    _b._validate_candidate_against_preregistration(candidate, preregistration)
    _b._validate_infrastructure_fix_in_source_closure(
        source_closure, preregistration
    )
    evaluation = build_evaluation_treatment_v2(
        candidate=candidate,
        execution_source_closure_hash=str(source_closure["closure_hash"]),
        measurement_view_hash=str(view["measurement_view_hash"]),
        benchmark_tree_hash=str(materialization_binding["benchmark_tree_hash"]),
    )
    validate_evaluation_treatment_v2(evaluation, candidate=candidate)
    evaluator_epoch = "financial-sec13f-contract-v2-replication-c-" + PRIVATE_PACK_HASH[:12]
    treatment = {
        "evaluation_binding": evaluation,
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "external_skill_source_receipt_hash": candidate.external_skill_source_receipt_hash,
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
        project, provider, env_file=provider_env_file
    )
    _b._validate_plus_provider(provider, protocol_binding=protocol_binding)
    plan_set_hash, typed_plan_set = _base._typed_plan_set(
        measurement_view=view, candidate=candidate
    )
    plan = _base._build_measurement_plan(
        measurement_view=view,
        candidate=candidate,
        evaluation=evaluation,
        protocol=protocol,
        evaluator_epoch=evaluator_epoch,
    )
    safe_plan = plan.safe_payload()
    _b._assert_exact_replication_b_plan(safe_plan)
    replication_binding = {
        "purpose": preregistration["purpose"],
        "replication_b_interruption_amendment_hash": B_INTERRUPTION_AMENDMENT_HASH,
        "prior_performance_result_reused": False,
        "prior_claims_retried": False,
        "prior_model_calls_replayed": False,
        "invalid_pair_replacement_authorized": False,
        "resampling_authorized": False,
    }
    body = {
        "manifest_version": EXECUTION_FREEZE_VERSION,
        "freeze_profile_version": PROFILE_VERSION,
        "study_id": STUDY_ID,
        "replication_binding": replication_binding,
        "preregistration": {
            **prereg_binding,
            "manifest_hash": preregistration_hash,
            "manifest_version": PREREGISTRATION_VERSION,
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
            "receipt_version": FORMATION_VERSION,
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
        "candidate": candidate.safe_payload(project),
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


def build_replication_c_execution_freeze_v1(
    *, project_root: str | Path, provider_env_file: str | Path
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    source = _base.build_execution_source_closure_v2(project)
    payload, _ = _assemble(
        project=project,
        provider_env_file=provider_env_file,
        source_closure=source,
    )
    return payload


def _fields() -> set[str]:
    return {
        "manifest_version", "freeze_profile_version", "study_id",
        "replication_binding", "preregistration", "acquisition", "formation",
        "measurement_view", "materialization", "prewarm", "paper_protocol",
        "provider", "candidate", "treatment", "precomputed_plan_set_hash",
        "typed_plan_set", "plan", "execution_source_closure", "execution",
        "private_pack_accessed", "gold_artifact_accessed",
        "expected_output_content_accessed", "sealed_content_accessed",
        "secret_value_persisted", "manifest_hash",
    }


def validate_replication_c_execution_freeze_v1(
    value: Mapping[str, Any],
    *,
    project_root: str | Path,
    provider_env_file: str | Path,
) -> FixedContractCandidateV2:
    project = Path(project_root).expanduser().resolve(strict=True)
    body = dict(value)
    declared = body.pop("manifest_hash", None)
    if (
        set(value) != _fields()
        or value.get("manifest_version") != EXECUTION_FREEZE_VERSION
        or value.get("freeze_profile_version") != PROFILE_VERSION
        or value.get("study_id") != STUDY_ID
        or not _base._is_sha256(declared)
        or declared != payload_hash(body)
        or value.get("execution") != _base._execution_policy()
        or value.get("private_pack_accessed") is not False
        or value.get("gold_artifact_accessed") is not False
        or value.get("expected_output_content_accessed") is not False
        or value.get("sealed_content_accessed") is not False
        or value.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("replication-C execution freeze drifted")
    _base._assert_no_secret_or_raw_payload(body)
    source = _mapping(value.get("execution_source_closure"), "source closure")
    _base.validate_execution_source_closure_v2(source, project_root=project)
    expected, candidate = _assemble(
        project=project,
        provider_env_file=provider_env_file,
        source_closure=copy.deepcopy(dict(source)),
    )
    if dict(value) != expected:
        raise ContractFreezeError("replication-C execution bindings changed")
    return candidate


def load_replication_c_execution_freeze_v1(
    path: str | Path,
    *,
    project_root: str | Path,
    provider_env_file: str | Path,
) -> tuple[dict[str, Any], FixedContractCandidateV2]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ContractFreezeError("replication-C execution freeze is unavailable")
    payload = read_json(source)
    candidate = validate_replication_c_execution_freeze_v1(
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
        payload = build_replication_c_execution_freeze_v1(
            project_root=args.project_root,
            provider_env_file=args.provider_env_file,
        )
        write_json(args.output, payload)
        print(json.dumps({
            "manifest_hash": payload["manifest_hash"],
            "provider_label": payload["provider"]["provider_label"],
            "physical_calls": payload["execution"]["physical_calls"],
            "study_id": payload["study_id"],
        }, sort_keys=True))
        return 0
    payload, _ = load_replication_c_execution_freeze_v1(
        args.execution_freeze,
        project_root=args.project_root,
        provider_env_file=args.provider_env_file,
    )
    print(json.dumps({"manifest_hash": payload["manifest_hash"], "validated": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

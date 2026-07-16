from __future__ import annotations

"""Build the separate finite execution freeze for Replication-C sealed test."""

import argparse
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    SharedFinancialSec13FContractPlannerV2,
)
from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    sha256_file,
    verify_measurement_view,
    write_json,
)
from replication_runtime.financial_semantic_v2.plan import FixedPeriodOutTreatmentV2

from .sealed_access import ACCESS_VERSION, validate_sealed_authorization_v1
from .freeze import (
    _git_blob,
    build_execution_source_closure_v2,
    validate_execution_source_closure_v2,
)
from .sealed_materialize import MATERIALIZATION_VERSION
from .sealed_plan import SealedTargetV1, build_sealed_plan_v1
from .sealed_prepare import PREPARATION_VERSION, verify_sealed_payload_v1
from .sealed_prewarm import PREWARM_VERSION
from .treatment import FixedContractCandidateV2, load_fixed_contract_candidate_v2


FREEZE_VERSION = "financial_sec13f_replication_c_sealed_execution_freeze_v1"
STUDY_ID = "financial-sec13f-contract-v2-replication-c-sealed-20260716"
SOURCE_CLOSURE_VERSION = (
    "financial_sec13f_replication_c_sealed_execution_source_closure_v1"
)
SUPPLEMENTAL_SOURCE_RELATIVE_PATHS = (
    "scripts/launch_tmux_detached_sealed_once.py",
    "scripts/launch_detached_formal_once.py",
)


class SealedFreezeError(PermissionError):
    """The separate sealed freeze failed closed."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def build_sealed_source_closure_v1(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    runtime = build_execution_source_closure_v2(project)
    rows = []
    for relative in SUPPLEMENTAL_SOURCE_RELATIVE_PATHS:
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise SealedFreezeError(f"sealed source is unavailable: {relative}")
        file_sha = sha256_file(path)
        try:
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(project),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    relative,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            blob = _git_blob(project, str(runtime["source_commit"]), relative)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SealedFreezeError("sealed supplemental Git binding failed") from exc
        if status.stdout.strip() or hashlib.sha256(blob).hexdigest() != file_sha:
            raise SealedFreezeError(
                f"sealed supplemental source is not committed and clean: {relative}"
            )
        rows.append({"relative_path": relative, "file_sha256": file_sha})
    body = {
        "closure_version": SOURCE_CLOSURE_VERSION,
        "scope_policy": "entire_runtime_closure_plus_committed_launchers_v1",
        "runtime_source_closure": runtime,
        "supplemental_files": rows,
        "supplemental_file_count": len(rows),
        "supplemental_file_set_hash": payload_hash(rows),
        "source_commit": runtime["source_commit"],
    }
    return {**body, "closure_hash": payload_hash(body)}


def validate_sealed_source_closure_v1(
    value: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> str:
    """Validate the frozen source commit without rebinding it to current HEAD."""

    project = Path(project_root).expanduser().resolve(strict=True)
    body = dict(value)
    declared = body.pop("closure_hash", None)
    runtime = value.get("runtime_source_closure")
    rows = value.get("supplemental_files")
    if (
        not _is_sha256(declared)
        or declared != payload_hash(body)
        or value.get("closure_version") != SOURCE_CLOSURE_VERSION
        or value.get("scope_policy")
        != "entire_runtime_closure_plus_committed_launchers_v1"
        or not isinstance(runtime, Mapping)
        or not isinstance(rows, list)
        or value.get("source_commit") != runtime.get("source_commit")
        or value.get("supplemental_file_count") != len(rows)
        or value.get("supplemental_file_set_hash") != payload_hash(rows)
    ):
        raise SealedFreezeError("sealed source closure identity drifted")
    try:
        validate_execution_source_closure_v2(runtime, project_root=project)
    except Exception as exc:
        raise SealedFreezeError("sealed runtime source closure drifted") from exc
    expected_paths = list(SUPPLEMENTAL_SOURCE_RELATIVE_PATHS)
    if [row.get("relative_path") for row in rows if isinstance(row, Mapping)] != expected_paths:
        raise SealedFreezeError("sealed supplemental source set drifted")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "file_sha256"}
            or not _is_sha256(row.get("file_sha256"))
        ):
            raise SealedFreezeError("sealed supplemental source row drifted")
        relative = str(row["relative_path"])
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise SealedFreezeError("sealed supplemental source is unavailable")
        try:
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(project),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    relative,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            blob = _git_blob(project, str(value["source_commit"]), relative)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SealedFreezeError("sealed supplemental Git binding failed") from exc
        if (
            status.stdout.strip()
            or sha256_file(path) != row["file_sha256"]
            or hashlib.sha256(blob).hexdigest() != row["file_sha256"]
        ):
            raise SealedFreezeError("sealed supplemental source changed after freeze")
    return str(declared)


def _validated_access_journal_binding(
    preparation: Mapping[str, Any],
    *,
    authorization_hash: str,
    private_pack_hash: str,
) -> dict[str, Any]:
    access = preparation.get("access_journal")
    expected_fields = {
        "access_version",
        "authorization_manifest_hash",
        "private_pack_hash",
        "claim_hash",
        "claim_file_sha256",
        "completion_hash",
        "completion_file_sha256",
        "access_claimed_before_path_probe",
        "access_completed",
        "raw_file_sha256_matches_precommit",
        "verified_public_pack_hash_matches_commitment",
        "private_path_persisted",
        "private_content_persisted",
    }
    if (
        not isinstance(access, Mapping)
        or set(access) != expected_fields
        or access.get("access_version") != ACCESS_VERSION
        or access.get("authorization_manifest_hash") != authorization_hash
        or access.get("private_pack_hash") != private_pack_hash
        or any(
            not _is_sha256(access.get(field))
            for field in (
                "claim_hash",
                "claim_file_sha256",
                "completion_hash",
                "completion_file_sha256",
            )
        )
        or access.get("access_claimed_before_path_probe") is not True
        or access.get("access_completed") is not True
        or access.get("raw_file_sha256_matches_precommit") is not True
        or access.get("verified_public_pack_hash_matches_commitment") is not True
        or access.get("private_path_persisted") is not False
        or access.get("private_content_persisted") is not False
        or preparation.get("private_pack_hash") != private_pack_hash
    ):
        raise SealedFreezeError("sealed access journal binding drifted")
    return dict(access)


def _self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or declared != payload_hash(body):
        raise SealedFreezeError(f"{label} self hash drifted")
    return str(declared)


def _build_treatment(
    *,
    candidate: FixedContractCandidateV2,
    source_closure_hash: str,
    sealed_payload_hash: str,
    benchmark_tree_hash: str,
) -> FixedPeriodOutTreatmentV2:
    identity = stable_hash(
        {
            "treatment_version": "financial_sec13f_replication_c_sealed_treatment_v1",
            "candidate_id": candidate.candidate_id,
            "recipe_id": candidate.recipe_id,
            "program_set_hash": candidate.program_set_hash,
            "external_skill_source_receipt_hash": candidate.external_skill_source_receipt_hash,
            "source_closure_hash": source_closure_hash,
            "sealed_payload_hash": sealed_payload_hash,
            "benchmark_tree_hash": benchmark_tree_hash,
            "operator_is_candidate_content": True,
            "candidate_changed_after_promotion": False,
        }
    )
    return FixedPeriodOutTreatmentV2(
        recipe_id=candidate.recipe_id,
        program_set_hash=candidate.program_set_hash,
        period_out_treatment_id=identity,
        external_skill_source_receipt_hash=candidate.external_skill_source_receipt_hash,
        candidate_skill_source=candidate.candidate_skill_source,
    )


def build_sealed_execution_freeze_v1(
    *,
    project_root: str | Path,
    measurement_view: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preparation: Mapping[str, Any],
    sealed_payload: Mapping[str, Any],
    materialization: Mapping[str, Any],
    prewarm: Mapping[str, Any],
    provider_binding: Mapping[str, Any],
    candidate: FixedContractCandidateV2,
    evaluator_epoch: str = "sec13f-replication-c-sealed-v1",
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    view = verify_measurement_view(measurement_view)
    payload = verify_sealed_payload_v1(sealed_payload, measurement_view=view)
    authorization_hash = validate_sealed_authorization_v1(
        authorization,
        expected_study_id=STUDY_ID,
        expected_private_pack_hash=str(view["private_pack_hash"]),
        expected_measurement_view_hash=str(view["measurement_view_hash"]),
        expected_candidate_id=candidate.candidate_id,
    )
    preparation_hash = _self_hash(preparation, "preparation_hash", "sealed preparation")
    materialization_hash = _self_hash(materialization, "materialization_hash", "sealed materialization")
    prewarm_hash = _self_hash(prewarm, "prewarm_hash", "sealed prewarm")
    access_journal = _validated_access_journal_binding(
        preparation,
        authorization_hash=authorization_hash,
        private_pack_hash=str(view["private_pack_hash"]),
    )
    if (
        preparation.get("preparation_version") != PREPARATION_VERSION
        or preparation.get("authorization_hash") != authorization_hash
        or preparation.get("sealed_payload_hash") != payload["sealed_payload_hash"]
        or preparation.get("sealed_item_count") != 4
        or preparation.get("cross_oracle_agreement") is not True
        or materialization.get("materialization_version") != MATERIALIZATION_VERSION
        or materialization.get("sealed_payload_hash") != payload["sealed_payload_hash"]
        or materialization.get("sealed_gold_hash")
        != preparation.get("sealed_gold_hash")
        or materialization.get("item_count") != 4
        or prewarm.get("prewarm_version") != PREWARM_VERSION
        or prewarm.get("sealed_payload_hash") != payload["sealed_payload_hash"]
        or prewarm.get("materialization_hash") != materialization_hash
        or prewarm.get("benchmark_tree_hash")
        != materialization.get("benchmark_tree_hash")
        or prewarm.get("item_count") != 4
        or prewarm.get("formal_execution_cache_only") is not True
        or provider_binding.get("provider_label") != "plus"
        or not _is_sha256(provider_binding.get("binding_hash"))
    ):
        raise SealedFreezeError("sealed execution inputs drifted")
    protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
    if protocol.payload.get("model") != provider_binding.get("model"):
        raise SealedFreezeError("Plus model differs from frozen protocol")
    source_closure = build_sealed_source_closure_v1(project)
    treatment = _build_treatment(
        candidate=candidate,
        source_closure_hash=source_closure["closure_hash"],
        sealed_payload_hash=payload["sealed_payload_hash"],
        benchmark_tree_hash=str(materialization["benchmark_tree_hash"]),
    )
    targets = tuple(
        SealedTargetV1(str(item["item_id"]), int(item["replicate"]))
        for item in payload["sealed_items"]
    )
    plan = build_sealed_plan_v1(
        targets=targets,
        manifest_hash=str(payload["sealed_payload_hash"]),
        evaluator_epoch=evaluator_epoch,
        treatment=treatment,
        agent_id=str(protocol.payload["agent_id"]),
        model=str(protocol.payload["model"]),
        max_steps=int(protocol.payload["max_steps"]),
        codex_agent_execution_policy_hash=protocol.codex_agent_execution_policy.policy_hash,
    )
    planner = SharedFinancialSec13FContractPlannerV2(asset_path=candidate.operator_asset_path)
    plan_receipts = []
    for item in payload["sealed_items"]:
        contract_plan, extraction = planner.build(str(item["instruction"]))
        plan_receipts.append(
            {
                "item_id_hash": payload_hash({"item_id": item["item_id"]}),
                "instruction_sha256": item["instruction_sha256"],
                "plan_hash": contract_plan["plan_hash"],
                "extraction_receipt_hash": extraction["receipt_hash"],
                "raw_plan_persisted": False,
            }
        )
    plan_set_hash = stable_hash(plan_receipts)
    body = {
        "manifest_version": FREEZE_VERSION,
        "study_id": STUDY_ID,
        "authorization_hash": authorization_hash,
        "sealed_access": access_journal,
        "private_pack_hash": view["private_pack_hash"],
        "measurement_view_hash": view["measurement_view_hash"],
        "sealed_payload_hash": payload["sealed_payload_hash"],
        "sealed_gold_hash": preparation["sealed_gold_hash"],
        "preparation_hash": preparation_hash,
        "materialization_hash": materialization_hash,
        "benchmark_tree_hash": materialization["benchmark_tree_hash"],
        "prewarm_hash": prewarm_hash,
        "candidate": candidate.safe_payload(project),
        "provider": dict(provider_binding),
        "execution_source_closure": source_closure,
        "treatment": {
            "recipe_id": treatment.recipe_id,
            "program_set_hash": treatment.program_set_hash,
            "period_out_treatment_id": treatment.period_out_treatment_id,
            "external_skill_source_receipt_hash": treatment.external_skill_source_receipt_hash,
            "evaluator_epoch": evaluator_epoch,
        },
        "plan": {"plan_hash": plan.plan_hash, "safe_payload": plan.safe_payload()},
        "precomputed_plan_receipts": plan_receipts,
        "precomputed_plan_set_hash": plan_set_hash,
        "execution_policy": {
            "sealed_pair_count": 4,
            "physical_model_calls": 8,
            "outer_workers": 8,
            "model_inference_slots": 8,
            "provider_label": "plus",
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "retry_count": 0,
            "replay_authorized": False,
            "resampling_authorized": False,
            "provider_switch_authorized": False,
            "recovery_authorized": False,
            "failure_disposition": "executed_incomplete_no_retry",
            "performance_gate_bound": False,
            "single_execution": True,
        },
        "sealed_content_persisted_in_freeze": False,
        "gold_content_persisted_in_freeze": False,
        "secret_value_persisted": False,
    }
    return {**body, "manifest_hash": payload_hash(body)}


def validate_sealed_execution_freeze_v1(
    value: Mapping[str, Any],
    *,
    project_root: str | Path,
    candidate: FixedContractCandidateV2,
) -> str:
    manifest_hash = _self_hash(value, "manifest_hash", "sealed execution freeze")
    policy = value.get("execution_policy")
    source = value.get("execution_source_closure")
    if (
        value.get("manifest_version") != FREEZE_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("candidate") != candidate.safe_payload(project_root)
        or not _is_sha256(value.get("sealed_gold_hash"))
        or not isinstance(value.get("sealed_access"), Mapping)
        or value.get("sealed_access", {}).get("access_claimed_before_path_probe") is not True
        or value.get("sealed_access", {}).get("access_completed") is not True
        or value.get("provider", {}).get("provider_label") != "plus"
        or not isinstance(policy, Mapping)
        or policy.get("physical_model_calls") != 8
        or policy.get("outer_workers") != 8
        or policy.get("model_inference_slots") != 8
        or policy.get("offline_evaluation_only") is not True
        or policy.get("online_judge_calls") != 0
        or policy.get("retry_count") != 0
        or policy.get("replay_authorized") is not False
        or policy.get("resampling_authorized") is not False
        or policy.get("provider_switch_authorized") is not False
        or policy.get("failure_disposition") != "executed_incomplete_no_retry"
        or value.get("sealed_content_persisted_in_freeze") is not False
        or not isinstance(source, Mapping)
    ):
        raise SealedFreezeError("sealed execution freeze drifted")
    validate_sealed_source_closure_v1(source, project_root=project_root)
    return manifest_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--sealed-payload", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--prewarm", type=Path, required=True)
    parser.add_argument("--provider-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from replication_runtime.financial_semantic_v2.pack import read_json

    args = _parser().parse_args(argv)
    project = args.project_root.expanduser().resolve(strict=True)
    freeze = build_sealed_execution_freeze_v1(
        project_root=project,
        measurement_view=read_json(args.measurement_view),
        authorization=read_json(args.authorization),
        preparation=read_json(args.preparation),
        sealed_payload=read_json(args.sealed_payload),
        materialization=read_json(args.materialization),
        prewarm=read_json(args.prewarm),
        provider_binding=read_json(args.provider_binding),
        candidate=load_fixed_contract_candidate_v2(project),
    )
    write_json(args.output, freeze)
    print(freeze["manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

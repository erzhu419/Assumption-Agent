from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import HypothesisProgram, HypothesisStatus, SplitName, stable_hash
from ..splits import BenchmarkItem, SplitManifest
from ..typed_operator_grammar import (
    TypedProgramBindingRegistry,
    freeze_typed_recipe_selection_snapshot,
    freeze_typed_selection_snapshot_ledger,
)
from .runtime_profile_injection import (
    RUNTIME_PROFILE_PROMPT_CONTAINER_PATH,
    RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
    RuntimeProfileInjectionError,
    RuntimeProfilePromptCapsule,
    RuntimeProfilePromptInjectionReceipt,
    VerifiedRuntimeProfile,
    bind_verified_runtime_profile_prompt,
    build_runtime_profile_prompt_capsule,
    verify_runtime_profile_prompt_injection_receipt,
)
from .skilllearn_compiler import SkillLearnProgramCompiler
from .skilllearn_lifecycle import (
    PortableTaskCapabilityRuntimeContext,
    SkillLearnSubprocessBackend,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .typed_portable_integration import (
    _docker_container_name_present,
    _docker_stdout,
    _project_portable_graph,
    _verify_historical_feasibility,
    reconstruct_current_full_graph_material,
)
from .typed_task_capability import PORTABLE_TASK_CAPABILITY_COMPILER_VERSION


TYPED_PROFILE_INJECTION_INTEGRATION_VERSION = (
    "v320_train_typed_profile_prompt_injection_integration_v1"
)
TYPED_PROFILE_INJECTION_RESULT_RECEIPT_VERSION = (
    "typed_profile_prompt_injection_result_receipt_v1"
)
TYPED_PROFILE_INJECTION_EVALUATOR_EPOCH = (
    "typed-profile-prompt-injection-integration-v1"
)

_PROGRAM_IDS = (
    "typed-policy-2b57dbf27b03b04011",
    "typed-policy-ac642731b6a72c2d97",
    "typed-policy-da248585fcf8685dc3",
)
_ACCEPTANCE_KEYS = (
    "historical_source_binding_passed",
    "train_only_scope_passed",
    "production_train_compile_passed",
    "verified_profile_materialization_passed",
    "runtime_profile_prompt_injection_passed",
    "pre_agent_launch_receipt_binding_passed",
    "fixed_tamper_probes_passed",
    "offline_non_scoring_boundary_passed",
)
_BOUNDARY_CONTRACT: Mapping[str, Any] = {
    "split_scope": "train_only",
    "source_agent_trials_reexecuted": False,
    "validation_public_input_accessed": False,
    "validation_outcome_accessed": False,
    "test_split_accessed": False,
    "sealed_split_accessed": False,
    "live_model_invoked": False,
    "task_backend_run_task_invoked": False,
    "live_evaluator_invoked": False,
    "verifier_invoked": False,
    "score_computed": False,
    "promotion_policy_evaluated": False,
    "container_runtime_invoked": True,
    "container_network": "none",
    "secret_value_persisted": False,
    "raw_content_persisted": False,
}


@dataclass(frozen=True)
class _CanaryExecution:
    row: Mapping[str, Any]
    capsule: RuntimeProfilePromptCapsule = field(
        compare=False,
        repr=False,
    )
    receipt: RuntimeProfilePromptInjectionReceipt
    run_template_before: str = field(compare=False, repr=False)
    run_template_after: str = field(compare=False, repr=False)
    public_instruction: str = field(compare=False, repr=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PermissionError(f"expected one JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_project_path(project_root: Path, raw: Any, *, directory: bool) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PermissionError("integration path is malformed")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("integration path escaped the project")
    path = (project_root / relative).resolve(strict=True)
    path.relative_to(project_root)
    if directory and not path.is_dir():
        raise PermissionError("integration directory is missing")
    if not directory and not path.is_file():
        raise PermissionError("integration file is missing")
    return path


def _implementation_rows(
    project_root: Path,
    paths: Sequence[str],
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for raw in paths:
        path = _safe_project_path(project_root, raw, directory=False)
        rows.append({"path": raw, "sha256": _sha256_file(path)})
    return tuple(rows)


def _read_preregistration(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    project_root = resolved.parent.parent.resolve(strict=True)
    payload = _read_json(resolved)
    if payload.get("integration_policy") != (
        TYPED_PROFILE_INJECTION_INTEGRATION_VERSION
    ):
        raise PermissionError("profile injection integration policy drifted")
    if payload.get("decision_budget") != 1:
        raise PermissionError("profile injection decision budget drifted")
    if tuple(payload.get("acceptance_predicates") or ()) != _ACCEPTANCE_KEYS:
        raise PermissionError("profile injection predicates drifted")
    if payload.get("boundary_contract") != dict(_BOUNDARY_CONTRACT):
        raise PermissionError("profile injection boundary drifted")
    implementation_paths = payload.get("implementation_files")
    if not isinstance(implementation_paths, list) or not all(
        isinstance(value, str) for value in implementation_paths
    ):
        raise PermissionError("profile injection implementation set is malformed")
    rows = _implementation_rows(project_root, implementation_paths)
    if stable_hash({"files": list(rows)}) != payload.get(
        "implementation_file_set_hash"
    ):
        raise PermissionError("profile injection implementation drifted")
    for key in (
        "benchmark_root",
        "manifest",
        "historical_typed_portable_preregistration",
        "historical_v320_archive",
        "historical_v320_report",
        "source_prewarm_receipt",
    ):
        expected_sha = payload.get(f"{key}_sha256")
        source = _safe_project_path(
            project_root,
            payload[key],
            directory=(key == "benchmark_root"),
        )
        if key != "benchmark_root" and _sha256_file(source) != expected_sha:
            raise PermissionError(f"profile injection source drifted: {key}")
    if payload.get("source_typed_snapshot_ledger_hash") != (
        "d560903a5df0da0a464b3636ef2f80bd86cba3f5230de53f5da6f3acc4597bbf"
    ):
        raise PermissionError("profile injection ledger commitment drifted")
    programs = payload.get("programs")
    if not isinstance(programs, list) or tuple(
        sorted(str(row.get("program_id") or "") for row in programs)
    ) != _PROGRAM_IDS:
        raise PermissionError("profile injection program bundle drifted")
    canaries = payload.get("train_canaries")
    if not isinstance(canaries, list) or len(canaries) != 3:
        raise PermissionError("profile injection canary set drifted")
    if stable_hash({"rows": canaries}) != payload.get("train_canary_set_hash"):
        raise PermissionError("profile injection canary hash drifted")
    prewarm = _read_json(
        _safe_project_path(
            project_root,
            payload["source_prewarm_receipt"],
            directory=False,
        )
    )
    prewarm_by_item_hash = {
        str(row.get("item_id_hash") or ""): row
        for row in prewarm.get("items", [])
        if isinstance(row, Mapping)
    }
    for canary in canaries:
        prewarm_row = prewarm_by_item_hash.get(str(canary["item_id_hash"]))
        if not isinstance(prewarm_row, Mapping) or any(
            prewarm_row.get(key) != canary.get(key)
            for key in (
                "family_hash",
                "prebuilt_image_id",
                "prebuilt_image_key",
                "task_input_closure_hash",
                "task_input_integrity_receipt_hash",
                "task_input_integrity_container_network",
            )
        ):
            raise PermissionError(
                "profile injection canary is not bound to the prewarm receipt"
            )
    return project_root, payload


def _restore_v320_program_bundle(
    *,
    project_root: Path,
    preregistration: Mapping[str, Any],
) -> tuple[
    SplitManifest,
    tuple[HypothesisProgram, ...],
    TypedProgramBindingRegistry,
    dict[str, Any],
]:
    historical_path = _safe_project_path(
        project_root,
        preregistration["historical_typed_portable_preregistration"],
        directory=False,
    )
    historical = _read_json(historical_path)
    verified_historical = _verify_historical_feasibility(
        historical,
        preregistration_path=historical_path,
    )
    commitments = dict(historical["full_graph_commitments"])
    commitments.update(
        {
            "manifest_hash": historical["manifest_hash"],
            "source_train_receipt_hash": historical[
                "source_train_receipt_hash"
            ],
        }
    )
    material = reconstruct_current_full_graph_material(
        root=project_root / historical["benchmark_root"],
        manifest_path=project_root / historical["manifest"],
        source_run_root=project_root / historical["source_run_root"],
        source_train_receipt=(
            project_root / historical["source_train_receipt"]
        ),
        commitments=commitments,
    )
    projected_graphs = tuple(
        _project_portable_graph(row) for row in material.full_graphs
    )
    snapshots = tuple(
        freeze_typed_recipe_selection_snapshot(row)
        for row in projected_graphs
    )
    graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": row.graph.target_family_hash,
                    "graph_hash": row.expected_graph_hash,
                    "availability_error_hash": None,
                }
                for row in snapshots
            ]
        }
    )
    model_catalog_set_hash = stable_hash(
        {
            "catalog_hashes": [
                row.expected_model_catalog_hash for row in snapshots
            ]
        }
    )
    manifest = SplitManifest.read(project_root / historical["manifest"])
    ledger = freeze_typed_selection_snapshot_ledger(
        snapshots,
        feasibility_preregistration_hash=(
            verified_historical.preregistration_hash
        ),
        feasibility_result_receipt_sha256=(
            verified_historical.result_receipt_file_sha256
        ),
        feasibility_decision_hash=verified_historical.decision_hash,
        feasibility_report_hash=verified_historical.report_hash,
        manifest_hash=manifest.manifest_hash,
        source_train_receipt_hash=material.evidence.source_train_receipt_hash,
        expected_graph_set_hash=graph_set_hash,
        expected_model_catalog_set_hash=model_catalog_set_hash,
        expected_target_family_hashes=tuple(
            row.target_family_hash for row in projected_graphs
        ),
    )
    if ledger.ledger_hash != preregistration[
        "source_typed_snapshot_ledger_hash"
    ]:
        raise PermissionError("profile injection typed ledger drifted")

    archive = _read_json(
        _safe_project_path(
            project_root,
            preregistration["historical_v320_archive"],
            directory=False,
        )
    )
    registry = TypedProgramBindingRegistry(snapshot_ledger=ledger)
    programs: list[HypothesisProgram] = []
    declared = {
        str(row["program_id"]): dict(row)
        for row in preregistration["programs"]
    }
    for program_id in _PROGRAM_IDS:
        program = HypothesisProgram.from_dict(archive["hypotheses"][program_id])
        binding = registry.restore_safe_payload(
            program,
            archive["typed_bindings"][program_id],
        )
        recipe = registry.require_bound_recipe(program).recipe
        row = declared[program_id]
        if (
            binding.binding_hash != row["binding_hash"]
            or recipe.recipe_id != row["recipe_id"]
            or program.trigger.all_of[0].value != row["family"]
        ):
            raise PermissionError("profile injection G2 bundle drifted")
        programs.append(program)
    source_receipt = {
        "historical_preregistration_sha256": _sha256_file(historical_path),
        "historical_result_receipt_sha256": (
            verified_historical.result_receipt_file_sha256
        ),
        "v320_archive_sha256": _sha256_file(
            _safe_project_path(
                project_root,
                preregistration["historical_v320_archive"],
                directory=False,
            )
        ),
        "v320_report_sha256": _sha256_file(
            _safe_project_path(
                project_root,
                preregistration["historical_v320_report"],
                directory=False,
            )
        ),
        "snapshot_ledger_hash": ledger.ledger_hash,
        "program_binding_set_hash": stable_hash(
            {
                "bindings": [
                    registry.require(program).safe_payload()
                    for program in programs
                ]
            }
        ),
        "historical_performance_reused": False,
        "raw_content_persisted": False,
    }
    return manifest, tuple(programs), registry, source_receipt


def _canary_items(
    *,
    project_root: Path,
    preregistration: Mapping[str, Any],
    manifest: SplitManifest,
) -> tuple[BenchmarkItem, ...]:
    benchmark_root = _safe_project_path(
        project_root,
        preregistration["benchmark_root"],
        directory=True,
    )
    items: list[BenchmarkItem] = []
    for row in preregistration["train_canaries"]:
        item_id = str(row["item_id"])
        family = str(row["family"])
        if (
            item_id not in manifest.train_ids
            or item_id in manifest.validation_ids
            or item_id in manifest.test_ids
            or manifest.family_by_id.get(item_id) != family
            or row["item_id_hash"] != stable_hash({"item_id": item_id})
            or row["family_hash"] != stable_hash({"family": family})
        ):
            raise PermissionError("profile injection canary is not TRAIN-only")
        instruction = benchmark_root / "tasks" / family / item_id / "instruction.md"
        dockerfile = (
            benchmark_root
            / "tasks"
            / family
            / item_id
            / "environment"
            / "Dockerfile"
        )
        if (
            _sha256_file(instruction) != row["public_instruction_sha256"]
            or _sha256_file(dockerfile) != row["dockerfile_sha256"]
        ):
            raise PermissionError("profile injection public input drifted")
        items.append(
            BenchmarkItem(
                id=item_id,
                family=family,
                features={
                    "benchmark": "skilllearnbench",
                    "family": family,
                    "train_profile_injection_canary": True,
                },
                content_ref=instruction.relative_to(benchmark_root).as_posix(),
                verifier_ref_hash=stable_hash(
                    {"item_id_hash": row["item_id_hash"], "accessed": False}
                ),
            )
        )
    return tuple(items)


def _run_canary(
    *,
    benchmark_root: Path,
    compile_result: Any,
    program_by_family: Mapping[str, HypothesisProgram],
    manifest_hash: str,
    canary: Mapping[str, Any],
    ordinal: int,
) -> _CanaryExecution:
    item_id = str(canary["item_id"])
    family = str(canary["family"])
    backend = SkillLearnSubprocessBackend(
        benchmark_root,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
    )
    runner = backend._load_runner()
    source = compile_result.source_for(item_id)
    receipt = compile_result.source_receipt_for(item_id)
    if source is None or len(receipt.portable_capability_metadata_file_hashes) != 1:
        raise PermissionError("profile injection canary compile route is missing")
    program = program_by_family[family]
    request = SkillLearnTrialRequest(
        item_id=item_id,
        family=family,
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_ON,
        evaluator_epoch=TYPED_PROFILE_INJECTION_EVALUATOR_EPOCH,
        pair_id=f"typed-profile-injection-canary-{ordinal}",
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash=manifest_hash,
        program_id=program.id,
        program_set_hash=compile_result.program_set_hash,
        treatment_hash=compile_result.treatment_hash_for(item_id),
        compile_manifest_hash=compile_result.manifest_hash,
        skill_source_receipt_hash=receipt.receipt_hash,
        compile_root=compile_result.output_root,
        typed_binding_set_hash=compile_result.typed_binding_set_hash,
        typed_snapshot_hashes=compile_result.typed_snapshot_hashes,
        typed_snapshot_ledger_hash=compile_result.typed_snapshot_ledger_hash,
        portable_capability_compiler_mode=(
            compile_result.portable_capability_compiler_mode
        ),
        portable_capability_role_spec_set_hash=(
            compile_result.portable_capability_role_spec_set_hash
        ),
        portable_capability_role_spec_hashes=(
            receipt.portable_capability_role_spec_hashes
        ),
        portable_capability_delivery_mode=(
            RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
        ),
    )
    context = backend._load_portable_task_capability_context(
        request=request,
        source_receipt=receipt,
        compile_root=compile_result.output_root,
    )
    if not isinstance(context, PortableTaskCapabilityRuntimeContext):
        raise PermissionError("profile injection runtime context is missing")

    expected_image_id = str(canary["prebuilt_image_id"])
    if _docker_stdout(
        runner.subprocess,
        ["docker", "image", "inspect", "--format", "{{.Id}}", expected_image_id],
    ).strip() != expected_image_id:
        raise PermissionError("profile injection image identity drifted")
    labels = json.loads(
        _docker_stdout(
            runner.subprocess,
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                expected_image_id,
            ],
        ).strip()
    )
    if not isinstance(labels, Mapping) or labels.get(
        "org.assumption-agent.prebuild.key"
    ) != canary["prebuilt_image_key"]:
        raise PermissionError("profile injection image key drifted")
    closure_hash = canary.get("task_input_closure_hash")
    if closure_hash is not None and labels.get(
        "org.assumption-agent.prebuild.task-input-closure"
    ) != closure_hash:
        raise PermissionError("profile injection task closure drifted")

    container_name = f"aa-profile-injection-{item_id[-1]}-{request.request_hash[:10]}"
    if _docker_container_name_present(runner.subprocess, container_name):
        raise PermissionError("profile injection container already exists")
    created = False
    execution: _CanaryExecution | None = None
    try:
        _docker_stdout(
            runner.subprocess,
            [
                "docker",
                "create",
                "--name",
                container_name,
                "--network",
                "none",
                "--entrypoint",
                "/bin/sh",
                expected_image_id,
                "-c",
                "while :; do sleep 60; done",
            ],
        )
        created = True
        _docker_stdout(runner.subprocess, ["docker", "start", container_name])
        dockerfile = (
            benchmark_root
            / "tasks"
            / family
            / item_id
            / "environment"
            / "Dockerfile"
        )
        copies = runner._parse_skill_copies(dockerfile)
        if not copies:
            raise PermissionError("profile injection route is missing")
        run_template_before = str(runner.get_agent("codex")["run"])
        runner._assumption_v2_task_capability_context = context
        runner._inject_skills_runtime(container_name, source, copies)
        effects = runner._assumption_v2_task_capability_effects
        injection = runner._assumption_v2_runtime_profile_injection_receipt
        run_template_after = str(runner.get_agent("codex")["run"])
        if (
            not isinstance(effects, tuple)
            or len(effects) != 1
            or not isinstance(injection, RuntimeProfilePromptInjectionReceipt)
        ):
            raise PermissionError("profile injection receipt is missing")
        profiles = tuple(
            VerifiedRuntimeProfile(
                metadata_hash=row.metadata_hash,
                item_id_hash=row.item_id_hash,
                role_spec_hash=row.role_spec_hash,
                effect_receipt_hash=row.effect_receipt_hash,
                output_sha256=row.output_sha256,
                profile_bytes=row.profile_bytes,
            )
            for row in effects
        )
        capsule = build_runtime_profile_prompt_capsule(
            request_hash=context.request_hash,
            context_hash=context.context_hash,
            source_receipt_hash=context.source_receipt_hash,
            typed_binding_set_hash=context.typed_binding_set_hash,
            public_instruction_hash=context.public_instruction_hash,
            profiles=profiles,
        )
        verify_runtime_profile_prompt_injection_receipt(
            injection,
            capsule=capsule,
            run_template_before=run_template_before,
            run_template_after=run_template_after,
            public_instruction=context.public_instruction,
        )

        with tempfile.TemporaryDirectory(
            prefix="typed-profile-launch-readback-",
        ) as temporary:
            temporary_root = Path(temporary)
            instruction = temporary_root / "instruction.txt"
            instruction.write_text(context.public_instruction, encoding="utf-8")
            _docker_stdout(
                runner.subprocess,
                [
                    "docker",
                    "cp",
                    str(instruction),
                    f"{container_name}:/tmp/instruction.txt",
                ],
            )
            effective_path = "/tmp/assumption-v2-effective-prompt.txt"
            _docker_stdout(
                runner.subprocess,
                [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    (
                        "printf '%s' \"$(cat /tmp/instruction.txt "
                        f"{RUNTIME_PROFILE_PROMPT_CONTAINER_PATH})\" > "
                        f"{effective_path}"
                    ),
                ],
            )
            effective = temporary_root / "effective-prompt.txt"
            _docker_stdout(
                runner.subprocess,
                [
                    "docker",
                    "cp",
                    f"{container_name}:{effective_path}",
                    str(effective),
                ],
            )
            effective_sha = _sha256_file(effective)
        network_mode = json.loads(
            _docker_stdout(
                runner.subprocess,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{json .HostConfig.NetworkMode}}",
                    container_name,
                ],
            ).strip()
        )
        container_image = _docker_stdout(
            runner.subprocess,
            ["docker", "inspect", "--format", "{{.Image}}", container_name],
        ).strip()
        if (
            network_mode != "none"
            or container_image != expected_image_id
            or effective_sha != injection.effective_prompt_sha256
        ):
            raise PermissionError("profile injection launch binding drifted")
        row = {
            "family_hash": canary["family_hash"],
            "item_id_hash": canary["item_id_hash"],
            "prebuilt_image_id": expected_image_id,
            "prebuilt_image_key": canary["prebuilt_image_key"],
            "request_hash": request.request_hash,
            "runtime_context_hash": context.context_hash,
            "source_receipt_hash": receipt.receipt_hash,
            "effect_hashes": [row.effect_hash for row in effects],
            "effect_receipt_hashes": [
                row.effect_receipt_hash for row in effects
            ],
            "profile_output_sha256s": [row.output_sha256 for row in effects],
            "capsule_hash": capsule.capsule_hash,
            "fragment_sha256": capsule.fragment_sha256,
            "injection_receipt_hash": injection.receipt_hash,
            "effective_prompt_sha256": injection.effective_prompt_sha256,
            "run_template_before_hash": injection.run_template_before_hash,
            "run_template_after_hash": injection.run_template_after_hash,
            "container_network": "none",
            "exact_image_verified": True,
            "profile_materialized_before_prompt_injection": True,
            "profile_present_in_effective_launch_prompt": True,
            "semantic_consumption_claimed": False,
            "task_effect_attributed": False,
            "agent_started": False,
            "model_invoked": False,
            "task_backend_run_task_invoked": False,
            "raw_content_persisted": False,
        }
        execution = _CanaryExecution(
            row=row,
            capsule=capsule,
            receipt=injection,
            run_template_before=run_template_before,
            run_template_after=run_template_after,
            public_instruction=context.public_instruction,
        )
    finally:
        runner._assumption_v2_task_capability_context = None
        runner._assumption_v2_task_capability_effects = None
        runner._assumption_v2_task_capability_agent_payloads = None
        runner._assumption_v2_runtime_profile_injection_receipt = None
        runner._assumption_v2_installed_skill_receipt = None
        cleanup_failed = False
        if created:
            try:
                _docker_stdout(
                    runner.subprocess,
                    ["docker", "rm", "-f", container_name],
                )
            except RuntimeError:
                cleanup_failed = True
        if cleanup_failed or _docker_container_name_present(
            runner.subprocess,
            container_name,
        ):
            raise RuntimeError("typed_profile_injection_cleanup_not_verified")
    if execution is None:
        raise RuntimeError("typed_profile_injection_canary_receipt_missing")
    return replace(
        execution,
        row={**dict(execution.row), "container_cleanup_verified": True},
    )


def _fixed_tamper_probes(
    executions: Sequence[_CanaryExecution],
) -> dict[str, bool]:
    if len(executions) != 3:
        raise PermissionError("profile injection tamper canaries are incomplete")
    first, second = executions[0], executions[1]
    profile = first.capsule.profiles[0]

    profile_mutation = False
    try:
        VerifiedRuntimeProfile(
            metadata_hash=profile.metadata_hash,
            item_id_hash=profile.item_id_hash,
            role_spec_hash=profile.role_spec_hash,
            effect_receipt_hash=profile.effect_receipt_hash,
            output_sha256=profile.output_sha256,
            profile_bytes=profile.profile_bytes + b" ",
        )
    except RuntimeProfileInjectionError:
        profile_mutation = True

    fragment_mutation = False
    try:
        bind_verified_runtime_profile_prompt(
            first.capsule,
            container_readback=first.capsule.fragment_bytes + b" ",
            run_template=first.run_template_before,
            public_instruction=first.public_instruction,
        )
    except RuntimeProfileInjectionError:
        fragment_mutation = True

    cross_item_swap = False
    try:
        bind_verified_runtime_profile_prompt(
            first.capsule,
            container_readback=second.capsule.fragment_bytes,
            run_template=first.run_template_before,
            public_instruction=first.public_instruction,
        )
    except RuntimeProfileInjectionError:
        cross_item_swap = True

    source_effect_rebinding = False
    try:
        verify_runtime_profile_prompt_injection_receipt(
            replace(first.receipt, source_receipt_hash="0" * 64),
            capsule=first.capsule,
            run_template_before=first.run_template_before,
            run_template_after=first.run_template_after,
            public_instruction=first.public_instruction,
        )
    except RuntimeProfileInjectionError:
        source_effect_rebinding = True
    return {
        "profile_byte_mutation": profile_mutation,
        "fragment_strip_or_mutation": fragment_mutation,
        "cross_item_capsule_swap": cross_item_swap,
        "source_effect_receipt_rebinding": source_effect_rebinding,
    }


def _artifact_paths(
    project_root: Path,
    preregistration: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path]:
    artifacts = preregistration.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PermissionError("profile injection artifact paths are malformed")
    paths: list[Path] = []
    for key in ("report", "events", "decision_lock", "result_receipt"):
        raw = artifacts.get(key)
        if not isinstance(raw, str) or not raw:
            raise PermissionError("profile injection artifact path is missing")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("profile injection artifact path escaped")
        path = (project_root / relative).resolve()
        path.relative_to(project_root)
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


def run_typed_profile_injection_integration(
    *,
    preregistration_path: str | Path,
) -> dict[str, Any]:
    project_root, preregistration = _read_preregistration(
        preregistration_path
    )
    report_path, events_path, lock_path, result_path = _artifact_paths(
        project_root,
        preregistration,
    )
    if any(path.exists() for path in (report_path, events_path, lock_path, result_path)):
        raise PermissionError("profile injection decision already exists")
    for path in (report_path, events_path, lock_path, result_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    manifest, programs, registry, source_binding = _restore_v320_program_bundle(
        project_root=project_root,
        preregistration=preregistration,
    )
    items = _canary_items(
        project_root=project_root,
        preregistration=preregistration,
        manifest=manifest,
    )
    benchmark_root = _safe_project_path(
        project_root,
        preregistration["benchmark_root"],
        directory=True,
    )
    compiler = SkillLearnProgramCompiler(
        typed_program_registry=registry,
        require_typed_bindings=True,
        portable_capability_compiler_mode=(
            PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
    )
    canary_rows = tuple(preregistration["train_canaries"])
    program_by_family = {
        str(program.trigger.all_of[0].value): program for program in programs
    }
    with tempfile.TemporaryDirectory(
        prefix="typed-profile-injection-compile-",
    ) as temporary:
        compile_result = compiler.compile(
            programs=programs,
            items=items,
            split_manifest=manifest,
            output_root=temporary,
            method_name="typed-profile-injection-integration",
            allowed_statuses={HypothesisStatus.SHADOW},
            target_item_ids=tuple(row.id for row in items),
            target_split="train",
            trace_id="typed-profile-injection:compile",
        )
        compile_manifest = _read_json(
            compile_result.output_root / "compile_manifest.json"
        )
        def execute_canary_pass() -> tuple[_CanaryExecution, ...]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        _run_canary,
                        benchmark_root=benchmark_root,
                        compile_result=compile_result,
                        program_by_family=program_by_family,
                        manifest_hash=manifest.manifest_hash,
                        canary=canary,
                        ordinal=index,
                    )
                    for index, canary in enumerate(canary_rows, start=1)
                ]
                return tuple(future.result() for future in futures)

        executions = execute_canary_pass()
        replay_executions = execute_canary_pass()
    executions = tuple(
        sorted(executions, key=lambda row: str(row.row["item_id_hash"]))
    )
    replay_executions = tuple(
        sorted(
            replay_executions,
            key=lambda row: str(row.row["item_id_hash"]),
        )
    )
    exact_replay_verified = [dict(row.row) for row in executions] == [
        dict(row.row) for row in replay_executions
    ]
    if not exact_replay_verified:
        raise RuntimeError("profile injection exact runtime replay drifted")
    tamper = _fixed_tamper_probes(executions)
    compile_receipt = {
        "compiler_class": type(compiler).__name__,
        "compiler_mode": PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
        "target_split": "train",
        "program_count": len(programs),
        "item_count": len(items),
        "compile_manifest_hash": stable_hash(compile_manifest),
        "typed_binding_set_hash": compile_result.typed_binding_set_hash,
        "typed_snapshot_ledger_hash": (
            compile_result.typed_snapshot_ledger_hash
        ),
        "portable_role_spec_set_hash": (
            compile_result.portable_capability_role_spec_set_hash
        ),
        "delivery_mode": RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
        "raw_content_persisted": False,
    }
    runtime_rows = [dict(row.row) for row in executions]
    runtime_receipt = {
        "canary_count": len(runtime_rows),
        "canary_set_hash": stable_hash({"canaries": runtime_rows}),
        "canaries": runtime_rows,
        "parallel_workers": 3,
        "exact_replay_canary_count": len(replay_executions),
        "exact_replay_verified": exact_replay_verified,
        "container_network": "none",
        "container_cleanup_verified": all(
            row["container_cleanup_verified"] is True for row in runtime_rows
        ),
        "agent_started": False,
        "model_invoked": False,
        "task_backend_run_task_invoked": False,
        "evaluator_invoked": False,
        "verifier_invoked": False,
        "raw_content_persisted": False,
    }
    acceptance = {
        "historical_source_binding_passed": (
            source_binding["snapshot_ledger_hash"]
            == preregistration["source_typed_snapshot_ledger_hash"]
            and len(programs) == 3
        ),
        "train_only_scope_passed": (
            len(items) == 3
            and all(item.id in manifest.train_ids for item in items)
            and all(item.id not in manifest.validation_ids for item in items)
            and all(item.id not in manifest.test_ids for item in items)
        ),
        "production_train_compile_passed": (
            compile_receipt["target_split"] == "train"
            and compile_receipt["program_count"] == 3
            and compile_receipt["item_count"] == 3
            and compile_receipt["typed_snapshot_ledger_hash"]
            == preregistration["source_typed_snapshot_ledger_hash"]
        ),
        "verified_profile_materialization_passed": all(
            row["profile_materialized_before_prompt_injection"] is True
            and len(row["effect_receipt_hashes"]) == 1
            for row in runtime_rows
        ),
        "runtime_profile_prompt_injection_passed": all(
            row["profile_present_in_effective_launch_prompt"] is True
            and row["fragment_sha256"]
            and row["injection_receipt_hash"]
            for row in runtime_rows
        ),
        "pre_agent_launch_receipt_binding_passed": all(
            row["agent_started"] is False
            and row["model_invoked"] is False
            and row["run_template_before_hash"]
            != row["run_template_after_hash"]
            and row["effective_prompt_sha256"]
            for row in runtime_rows
        ) and exact_replay_verified,
        "fixed_tamper_probes_passed": all(tamper.values()),
        "offline_non_scoring_boundary_passed": (
            runtime_receipt["container_network"] == "none"
            and runtime_receipt["container_cleanup_verified"] is True
            and runtime_receipt["model_invoked"] is False
            and runtime_receipt["task_backend_run_task_invoked"] is False
            and runtime_receipt["evaluator_invoked"] is False
            and runtime_receipt["verifier_invoked"] is False
        ),
    }
    integration_passed = all(acceptance.values())
    decision = {
        "integration_policy": TYPED_PROFILE_INJECTION_INTEGRATION_VERSION,
        "decision_ordinal": 1,
        "acceptance": acceptance,
        "source_binding": source_binding,
        "compiler_provenance": compile_receipt,
        "runtime": runtime_receipt,
        "tamper_probes": tamper,
        "offline_boundary_contract": dict(_BOUNDARY_CONTRACT),
        "integration_passed": integration_passed,
        "development_task_execution_authorized": False,
        "promotion_gate_or_score": None,
        "semantic_model_consumption_claimed": False,
        "task_utility_claimed": False,
        "raw_content_persisted": False,
    }
    decision_hash = stable_hash(decision)
    report_without_hash = {
        **decision,
        "decision_budget": 1,
        "decision_hash": decision_hash,
        "exact_replay_verified": True,
        "preregistration_sha256": _sha256_file(
            Path(preregistration_path).expanduser().resolve(strict=True)
        ),
    }
    # Canonical serialization must also be stable after the exact second
    # no-model/no-score Docker replay above.
    if stable_hash(report_without_hash) != stable_hash(
        json.loads(json.dumps(report_without_hash, sort_keys=True))
    ):
        raise RuntimeError("profile injection exact replay drifted")
    report = {
        **report_without_hash,
        "report_hash": stable_hash(report_without_hash),
    }
    events = [
        {
            "event": "typed_profile_injection_integration_started",
            "decision_ordinal": 1,
            "decision_budget": 1,
            "raw_content_persisted": False,
        },
        *[
            {
                "event": "typed_profile_injection_canary_completed",
                **row,
            }
            for row in runtime_rows
        ],
        {
            "event": "typed_profile_injection_integration_completed",
            "decision_hash": decision_hash,
            "report_hash": report["report_hash"],
            "integration_passed": integration_passed,
            "raw_content_persisted": False,
        },
    ]
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    _write_json(report_path, report)
    lock = {
        "lock_version": "typed_profile_injection_decision_lock_v1",
        "decision_ordinal": 1,
        "decision_hash": decision_hash,
        "report_hash": report["report_hash"],
        "integration_passed": integration_passed,
        "terminal": True,
        "raw_content_persisted": False,
    }
    _write_json(lock_path, lock)
    result = {
        "result_receipt_version": (
            TYPED_PROFILE_INJECTION_RESULT_RECEIPT_VERSION
        ),
        "integration_policy": TYPED_PROFILE_INJECTION_INTEGRATION_VERSION,
        "decision_budget": 1,
        "decision_ordinal": 1,
        "decision_hash": decision_hash,
        "report_hash": report["report_hash"],
        "integration_passed": integration_passed,
        "exact_replay_verified": True,
        "canonical_artifacts": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in (report_path, events_path, lock_path)
        ],
        "source_binding": source_binding,
        "compiler_provenance": compile_receipt,
        "runtime_canary_set_hash": runtime_receipt["canary_set_hash"],
        "acceptance": acceptance,
        "development_task_execution_authorized": False,
        "semantic_model_consumption_claimed": False,
        "task_utility_claimed": False,
        "raw_content_persisted": False,
    }
    result["result_receipt_hash"] = stable_hash(result)
    _write_json(result_path, result)
    if not integration_passed:
        raise RuntimeError("typed profile injection integration failed")
    return report


def verify_existing_typed_profile_injection(
    *,
    preregistration_path: str | Path,
) -> dict[str, Any]:
    project_root, preregistration = _read_preregistration(
        preregistration_path
    )
    report_path, events_path, lock_path, result_path = _artifact_paths(
        project_root,
        preregistration,
    )
    report = _read_json(report_path)
    lock = _read_json(lock_path)
    result = _read_json(result_path)
    if (
        report.get("report_hash")
        != stable_hash({k: v for k, v in report.items() if k != "report_hash"})
        or lock.get("report_hash") != report.get("report_hash")
        or lock.get("decision_hash") != report.get("decision_hash")
        or result.get("report_hash") != report.get("report_hash")
        or result.get("decision_hash") != report.get("decision_hash")
        or result.get("integration_passed") is not True
        or report.get("integration_passed") is not True
        or result.get("exact_replay_verified") is not True
        or set(report.get("acceptance") or {}) != set(_ACCEPTANCE_KEYS)
        or not all((report.get("acceptance") or {}).values())
    ):
        raise PermissionError("profile injection completed receipt drifted")
    expected_artifacts = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in (report_path, events_path, lock_path)
    ]
    result_hash = result.pop("result_receipt_hash", None)
    if (
        result.get("canonical_artifacts") != expected_artifacts
        or result_hash != stable_hash(result)
    ):
        raise PermissionError("profile injection result receipt drifted")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the TRAIN-only non-scoring runtime profile "
            "prompt-injection integration."
        )
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "manifests/skilllearn_typed_profile_injection_integration_v1.json"
        ),
    )
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        report = verify_existing_typed_profile_injection(
            preregistration_path=args.preregistration,
        )
    else:
        report = run_typed_profile_injection_integration(
            preregistration_path=args.preregistration,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

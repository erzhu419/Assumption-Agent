from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks.financial_sec13f_contract_operator_v2 import (
    NUMERIC_ENGINE,
    OPERATOR_VERSION,
    QUERY_RECEIPT_VERSION,
    payload_hash,
)
from assumption_agent.benchmarks.offline_verifier import (
    offline_verifier_profile_for_family,
)
from assumption_agent.models import stable_hash
from replication_runtime.financial_sec13f_contract_v2.controls import (
    CONTROL_STAGE_ORDER_V1,
    ControlTargetBindingV1,
    authorize_control_execution_once_v1,
    build_control_plan_v1,
    initialize_control_state_v1,
)
from replication_runtime.financial_sec13f_contract_v2.controls_formal import (
    CONTROLS_EXECUTION_FREEZE_VERSION,
    CONTROLS_FORMAL_RUNTIME_VERSION,
    ControlsFormalRuntimeError,
    FrozenOperatorOnlyBackendV1,
    OperatorOnlySharedRuntimeV1,
    _control_work_rows,
    _project_file_binding,
    validate_controls_execution_freeze_v1,
)
from replication_runtime.financial_semantic_v2.durable_state import (
    load_durable_stage_chain_v2,
)


def _hash(label: str) -> str:
    return stable_hash({"formal-controls-test": label})


def _targets() -> tuple[ControlTargetBindingV1, ...]:
    return tuple(
        ControlTargetBindingV1(
            item_id=f"formal-control-{index}",
            fold_id=f"measurement-fold-{index // 2}",
            prior_pair_id=f"prior-pair-{index}",
            prior_raw_observation_hash=_hash(f"raw-{index}"),
            prior_candidate_observation_hash=_hash(f"full-{index}"),
            prior_raw_success=False,
            prior_candidate_success=True,
            candidate_output_sha256=_hash(f"output-{index}"),
            typed_plan_hash=_hash(f"plan-{index}"),
            extraction_receipt_hash=_hash(f"extraction-{index}"),
        )
        for index in range(8)
    )


def _plan() -> Any:
    recipe = _hash("recipe")
    return build_control_plan_v1(
        targets=_targets(),
        controls_preregistration_hash=_hash("prereg"),
        prior_measurement_report_hash=_hash("report"),
        prior_measurement_plan_hash=_hash("prior-plan"),
        evaluator_epoch="formal-controls-test-v1",
        candidate_recipe_id=recipe,
        candidate_program_set_hash=stable_hash({"recipe_ids": [recipe]}),
        candidate_treatment_hash=_hash("full-treatment"),
        skill_only_treatment_hash=_hash("skill-treatment"),
        operator_only_treatment_hash=_hash("operator-treatment"),
        external_skill_source_receipt_hash=_hash("skill-source"),
        candidate_skill_source=Path("/tmp/frozen-skill"),
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        codex_agent_execution_policy_hash=_hash("execution-policy"),
    )


def _prewarm(plan: Any) -> dict[str, Any]:
    rows = []
    for index, target in enumerate(
        sorted(_targets(), key=lambda row: row.item_id_hash)
    ):
        rows.append(
            {
                "agent_runtime_key": _hash(f"agent-runtime-{index}"),
                "agent_runtime_version": "codex-cli test",
                "cache_key": _hash(f"cache-{index}"),
                "environment_hash": _hash(f"environment-{index}"),
                "image_id": "sha256:" + _hash(f"image-{index}"),
                "item_id": target.item_id,
                "item_id_hash": target.item_id_hash,
                "offline_verifier_profile_hash": _hash("profile"),
                "offline_verifier_profile_id": "common-pytest-ctrf-py312-v1",
                "offline_verifier_runtime_key": _hash("runtime"),
                "offline_verifier_runtime_reused": True,
                "prebuilt_cache_reused": True,
                "source_environment_hash": _hash(f"environment-{index}"),
                "verifier_runtime_network": "none",
            }
        )
    return {
        "formal_cache_rows": rows,
        "formal_cache_row_set_hash": payload_hash(rows),
        "formal_execution_cache_only": True,
        "formal_image_cache_only": True,
        "formal_offline_verifier_cache_only": True,
        "formal_verifier_network": "none",
        "model_calls": 0,
        "online_judge_calls": 0,
    }


def _query_receipt(
    work: Any,
    *,
    output_sha: str,
    pre_output_exists: bool = False,
) -> dict[str, Any]:
    asset = {
        "candidate_id": _hash("candidate"),
        "manifest_hash": _hash("asset"),
        "contract_hash": _hash("contract"),
        "operator_source_sha256": _hash("operator-source"),
    }
    input_rows = [
        {
            "role": role,
            "table": table,
            "size_bytes": index + 1,
            "file_sha256": _hash(f"input-{index}"),
        }
        for index, (role, table) in enumerate(
            (
                ("previous", "COVERPAGE.tsv"),
                ("previous", "INFOTABLE.tsv"),
                ("current", "COVERPAGE.tsv"),
                ("current", "INFOTABLE.tsv"),
            )
        )
    ]
    body = {
        "receipt_version": QUERY_RECEIPT_VERSION,
        "operator_version": OPERATOR_VERSION,
        "candidate_id": asset["candidate_id"],
        "asset_manifest_hash": asset["manifest_hash"],
        "contract_hash": asset["contract_hash"],
        "operator_source_sha256": asset["operator_source_sha256"],
        "plan_hash": work.target.typed_plan_hash,
        "numeric_engine": NUMERIC_ENGINE,
        "input_file_receipts": input_rows,
        "input_set_hash": payload_hash(input_rows),
        "pre_output_exists": pre_output_exists,
        "pre_output_sha256": (
            _hash("unexpected-pre-output") if pre_output_exists else None
        ),
        "post_output_sha256": output_sha,
        "output_changed": True,
        "answer_key_set_hash": _hash("answer-key-set"),
        "answers_payload_persisted_in_receipt": False,
        "raw_entity_persisted_in_receipt": False,
        "network_calls": 0,
        "model_calls": 0,
        "verifier_content_accessed": False,
        "gold_content_accessed": False,
        "pack_content_accessed": False,
    }
    return {**body, "receipt_hash": payload_hash(body)}


class _FakeDelegate:
    def __init__(
        self,
        *,
        state_root: Path,
        verifier_command: str,
        image_id: str,
    ) -> None:
        self.state_root = state_root
        self.verifier_command = verifier_command
        self.image_id = image_id
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], **_: Any) -> Any:
        argv = tuple(str(value) for value in command)
        self.commands.append(argv)
        if argv[:2] == ("docker", "run"):
            return SimpleNamespace(returncode=0, stdout="a" * 64, stderr="")
        if argv[:3] == ("docker", "inspect", "--format"):
            value = self.image_id if "{{.Image}}" in argv else "none"
            return SimpleNamespace(returncode=0, stdout=value + "\n", stderr="")
        if argv[:2] == ("docker", "exec") and argv[-1] == self.verifier_command:
            verifier = self.state_root / "operator_only_trial" / "verifier"
            (verifier / "reward.txt").write_text("1\n", encoding="utf-8")
            (verifier / "ctrf.json").write_text(
                json.dumps(
                    {
                        "results": {
                            "summary": {
                                "tests": 1,
                                "passed": 1,
                                "failed": 0,
                                "skipped": 0,
                                "pending": 0,
                                "other": 0,
                            },
                            "tests": [{"name": "test_answer_contract"}],
                        }
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _mock_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mismatch: bool = False,
    pre_output_exists: bool = False,
) -> tuple[FrozenOperatorOnlyBackendV1, Any, Path, _FakeDelegate]:
    plan = _plan()
    work = next(row for row in plan.work_units if row.arm == "operator_only")
    project = tmp_path / "project"
    task = (
        project
        / "benchmark"
        / "tasks"
        / "financial-analysis"
        / work.target.item_id
    )
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text("public instruction\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text("pytest --ctrf\n", encoding="utf-8")
    asset_path = project / "asset.json"
    asset_path.write_text("{}\n", encoding="utf-8")
    state = tmp_path / "durable"
    initialize_control_state_v1(state_root=state, work=work)
    authorize_control_execution_once_v1(state_root=state, work=work)
    profile = offline_verifier_profile_for_family("financial-analysis")
    assert profile is not None
    image_id = "sha256:" + _hash("image-live")
    delegate = _FakeDelegate(
        state_root=state,
        verifier_command=profile.verifier_command,
        image_id=image_id,
    )
    shared = OperatorOnlySharedRuntimeV1(
        project_root=project,
        benchmark_root=project / "benchmark",
        asset_path=asset_path,
        prebuilt_cache=object(),
        offline_verifier_cache=object(),
        runner=object(),
        prewarm=_prewarm(plan),
        expected_program_id=plan.candidate_recipe_id,
        expected_treatment_hash=plan.operator_only_treatment_hash,
        expected_external_skill_source_receipt_hash=(
            plan.external_skill_source_receipt_hash
        ),
        docker_delegate=delegate,
    )
    backend = FrozenOperatorOnlyBackendV1(work, shared=shared)
    asset = {
        "candidate_id": _hash("candidate"),
        "manifest_hash": _hash("asset"),
        "contract_hash": _hash("contract"),
        "operator_source_sha256": _hash("operator-source"),
    }
    bound = SimpleNamespace(asset=asset)
    monkeypatch.setattr(
        backend,
        "_frozen_planner",
        MethodType(
            lambda self, **kwargs: (
                bound,
                {"plan_hash": work.target.typed_plan_hash},
                {"receipt_hash": work.target.extraction_receipt_hash},
            ),
            backend,
        ),
    )
    image = SimpleNamespace(
        tag="frozen:test",
        cache_key=_hash("cache-live"),
        environment_hash=_hash("environment-live"),
        source_environment_hash=_hash("environment-live"),
        image_id=image_id,
        reused=True,
    )
    runtime = SimpleNamespace(
        profile=profile,
        runtime_key=_hash("runtime-live"),
        volume_name="frozen-offline-volume",
        base_image_id=image.image_id,
        reused=True,
    )
    monkeypatch.setattr(
        backend,
        "_ensure_local_runtime",
        MethodType(
            lambda self, **kwargs: (image, runtime, _prewarm(plan)["formal_cache_rows"][0]),
            backend,
        ),
    )
    output = _hash("different-output") if mismatch else work.target.candidate_output_sha256
    query_receipt = _query_receipt(
        work,
        output_sha=output,
        pre_output_exists=pre_output_exists,
    )
    contract_state = SimpleNamespace(runtime_evidence=None)

    class _ProductionHook:
        _contract_local = SimpleNamespace(state=contract_state)

        def _execute_contract_plan_before_verifier_v2(self, **kwargs: Any) -> None:
            kwargs["delegate"].commands.append(("PRODUCTION_OPERATOR",))
            contract_state.runtime_evidence = {
                "query_receipt": query_receipt,
                "evidence_hash": _hash("operator-evidence"),
                "online_calls": 0,
                "executed_before_verifier_materialization": True,
            }

    monkeypatch.setattr(
        backend,
        "_production_operator_backend",
        MethodType(
            lambda self, **kwargs: (_ProductionHook(), contract_state),
            backend,
        ),
    )
    return backend, work, state, delegate


def test_operator_only_orders_operator_before_tests_and_exact_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, work, state, delegate = _mock_backend(tmp_path, monkeypatch)
    result = backend.run_control(
        work=work,
        state_root=state,
        trace_id="formal-operator-only-test",
    )
    assert result.valid and result.success
    assert result.model_calls == 0 and result.operator_calls == 1
    assert result.output_sha256 == work.target.candidate_output_sha256
    run = next(row for row in delegate.commands if row[:2] == ("docker", "run"))
    assert "--network" in run and run[run.index("--network") + 1] == "none"
    assert work.target.item_id not in run
    assert delegate.image_id in run
    assert not any("API_KEY=" in token for token in run)
    assert any(
        row[:2] == ("docker", "exec")
        and row[-1] == "test ! -e /root/answers.json"
        for row in delegate.commands
    )
    operator_index = delegate.commands.index(("PRODUCTION_OPERATOR",))
    test_copy_index = next(
        index
        for index, row in enumerate(delegate.commands)
        if row[:2] == ("docker", "cp") and row[-1].endswith(":/tests")
    )
    verifier_index = next(
        index
        for index, row in enumerate(delegate.commands)
        if row[:2] == ("docker", "exec") and "--ctrf" in row[-1]
    )
    assert operator_index < test_copy_index < verifier_index
    chain = load_durable_stage_chain_v2(
        state,
        stage_order=CONTROL_STAGE_ORDER_V1,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request_hash,
    )
    assert [row.stage for row in chain] == list(CONTROL_STAGE_ORDER_V1)
    assert chain[3].payload["agent_started"] is False
    assert chain[4].payload["persisted_before_verifier"] is True


def test_operator_only_output_mismatch_fails_before_test_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, work, state, delegate = _mock_backend(
        tmp_path, monkeypatch, mismatch=True
    )
    with pytest.raises(ControlsFormalRuntimeError, match="differs from replication C"):
        backend.run_control(
            work=work,
            state_root=state,
            trace_id="formal-operator-only-mismatch",
        )
    assert not any(
        row[:2] == ("docker", "cp") and row[-1].endswith(":/tests")
        for row in delegate.commands
    )
    chain = load_durable_stage_chain_v2(
        state,
        stage_order=CONTROL_STAGE_ORDER_V1,
        work_unit_hash=work.work_unit_hash,
        request_hash=work.request_hash,
    )
    assert [row.stage for row in chain] == list(CONTROL_STAGE_ORDER_V1[:4])


def test_operator_only_rejects_any_preexisting_agent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, work, state, delegate = _mock_backend(
        tmp_path,
        monkeypatch,
        pre_output_exists=True,
    )
    with pytest.raises(ControlsFormalRuntimeError, match="causal boundary drifted"):
        backend.run_control(
            work=work,
            state_root=state,
            trace_id="formal-operator-only-pre-output",
        )
    assert not any(
        row[:2] == ("docker", "cp") and row[-1].endswith(":/tests")
        for row in delegate.commands
    )


def _freeze_payload(plan: Any) -> dict[str, Any]:
    work_rows = _control_work_rows(plan)
    prewarm = _prewarm(plan)
    launcher_rows = [
        {
            "role": role,
            "relative_path": relative,
            "file_sha256": _hash(role),
            "committed_at_git_commit": "a" * 40,
        }
        for role, relative in (
            ("controls_launcher", "scripts/launch_tmux_detached_controls_once.py"),
            ("base_launcher", "scripts/launch_detached_formal_once.py"),
        )
    ]
    provider = {
        "api_origin": "https://ruoli.dev",
        "model": "gpt-5.4-mini",
        "provider_label": "plus",
        "pro_fallback_authorized": False,
        "provider_binding_hash_from_replication_c": _hash("provider"),
        "inherit_replication_c_execution_policy_hash": _hash("policy"),
        "secret_value_persisted": False,
    }
    body = {
        "manifest_version": CONTROLS_EXECUTION_FREEZE_VERSION,
        "study_id": "formal-controls-test",
        "controls_formal_runtime_version": CONTROLS_FORMAL_RUNTIME_VERSION,
        "preregistration": {},
        "replication_c_execution_freeze": {},
        "execution_source_closure": {"closure_hash": _hash("closure")},
        "launcher_source_closure": {
            "files": launcher_rows,
            "file_count": 2,
            "file_set_hash": payload_hash(launcher_rows),
        },
        "plan": {
            "plan_hash": plan.plan_hash,
            "work_unit_count": 16,
            "work_units": work_rows,
            "work_unit_set_hash": payload_hash(work_rows),
            "raw_content_persisted": False,
        },
        "runtime_identity": {
            "expected_program_id": plan.candidate_recipe_id,
            "expected_program_set_hash": plan.candidate_program_set_hash,
            "skill_only_treatment_hash": plan.skill_only_treatment_hash,
            "operator_only_treatment_hash": plan.operator_only_treatment_hash,
            "external_skill_source_receipt_hash": (
                plan.external_skill_source_receipt_hash
            ),
        },
        "candidate": {},
        "cohort": {},
        "provider": provider,
        "prewarm": {
            "formal_cache_rows": prewarm["formal_cache_rows"],
            "formal_execution_cache_only": True,
            "formal_verifier_network": "none",
        },
        "prior_measurement_reuse": {
            "reused_observation_count": 16,
            "executions_performed": 0,
            "model_calls_performed": 0,
            "operator_calls_performed": 0,
            "offline_verifier_calls_performed": 0,
            "completed_arm_reexecution_authorized": False,
        },
        "execution": {
            "physical_work_units": 16,
            "skill_only_work_units": 8,
            "operator_only_work_units": 8,
            "maximum_concurrent_work_units": 16,
            "maximum_concurrent_model_calls": 8,
            "new_model_calls": 8,
            "operator_calls": 8,
            "offline_verifier_calls": 16,
            "all_futures_submitted_before_results_read": True,
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "provider_retry_authorized": False,
            "model_replay_authorized": False,
            "operator_replay_authorized": False,
            "verifier_replay_authorized": False,
            "invalid_item_replacement_authorized": False,
            "resampling_authorized": False,
            "retry_count": 0,
        },
        "evidence_boundary": {
            "answer_payload_content_accessed": False,
            "sealed_content_accessed": False,
        },
        "analysis_policy": {
            "controls_are_mechanism_characterization": True,
            "performance_gate_bound": False,
            "numeric_performance_threshold_bound": False,
            "promotion_gate_reopened": False,
            "candidate_mutation_authorized": False,
            "sealed_evaluation_authorized": False,
        },
    }
    return {**body, "manifest_hash": payload_hash(body)}


def test_execution_freeze_binds_exact_grid_without_performance_gate() -> None:
    plan = _plan()
    freeze = _freeze_payload(plan)
    assert validate_controls_execution_freeze_v1(
        freeze, expected_plan=plan
    ) == freeze["manifest_hash"]
    tampered = copy.deepcopy(freeze)
    tampered["execution"]["maximum_concurrent_work_units"] = 15
    body = dict(tampered)
    body.pop("manifest_hash")
    tampered["manifest_hash"] = payload_hash(body)
    with pytest.raises(ControlsFormalRuntimeError, match="policy"):
        validate_controls_execution_freeze_v1(tampered, expected_plan=plan)

    tampered_plan = copy.deepcopy(freeze)
    tampered_plan["plan"]["plan_hash"] = _hash("forged-plan")
    body = dict(tampered_plan)
    body.pop("manifest_hash")
    tampered_plan["manifest_hash"] = payload_hash(body)
    with pytest.raises(ControlsFormalRuntimeError, match="plan differs"):
        validate_controls_execution_freeze_v1(
            tampered_plan,
            expected_plan=plan,
        )

    tampered_runtime = copy.deepcopy(freeze)
    tampered_runtime["runtime_identity"][
        "operator_only_treatment_hash"
    ] = _hash("forged-operator-treatment")
    body = dict(tampered_runtime)
    body.pop("manifest_hash")
    tampered_runtime["manifest_hash"] = payload_hash(body)
    with pytest.raises(ControlsFormalRuntimeError, match="plan differs"):
        validate_controls_execution_freeze_v1(
            tampered_runtime,
            expected_plan=plan,
        )

    tampered_authorization = copy.deepcopy(freeze)
    first_skill = next(
        row
        for row in tampered_authorization["plan"]["work_units"]
        if row["arm"] == "skill_only"
    )
    first_skill["model_call_authorization_count"] = 0
    tampered_authorization["plan"]["work_unit_set_hash"] = payload_hash(
        tampered_authorization["plan"]["work_units"]
    )
    body = dict(tampered_authorization)
    body.pop("manifest_hash")
    tampered_authorization["manifest_hash"] = payload_hash(body)
    with pytest.raises(ControlsFormalRuntimeError, match="work set"):
        validate_controls_execution_freeze_v1(
            tampered_authorization,
            expected_plan=plan,
        )


def test_committed_binding_uses_repo_relative_path_for_nested_project(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    project = repository / "nested-project"
    project.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Formal Test"],
        check=True,
    )
    artifact = project / "binding.json"
    artifact.write_text('{"safe": true}\n', encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "add", "nested-project/binding.json"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    binding = _project_file_binding(
        project,
        artifact,
        label="nested committed fixture",
        committed=True,
    )
    assert binding["relative_path"] == "binding.json"
    assert len(binding["committed_at_git_commit"]) == 40

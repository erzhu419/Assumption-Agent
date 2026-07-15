from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from assumption_agent.benchmarks.execution_contract_integration_v2 import (
    EXECUTION_CONTRACT_BUNDLE_FILENAME,
    ExecutionContractSubprocessBackendV2,
    build_execution_contract_compile_bundle_v2,
)
from assumption_agent.benchmarks.execution_contract_prompt_v2 import (
    EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH,
    EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION,
    bind_execution_contract_prompt_v2,
    build_execution_contract_prompt_capsule_v2,
)
from assumption_agent.benchmarks.runtime_profile_injection import (
    VerifiedRuntimeProfile,
)
from assumption_agent.benchmarks.skilllearn_compiler import (
    SkillLearnProgramCompiler,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    PortableTaskCapabilityRuntimeContext,
    SkillLearnAgentTerminalError,
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.benchmarks.typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.models import (
    HypothesisStatus,
    ResidualExample,
    SplitName,
    stable_hash,
)
from assumption_agent.typed_execution_contract import (
    TypedExecutionContractRegistry,
    derive_train_execution_contract,
)
from tests.test_portable_capability_compiler import (
    STOCK_FAMILY,
    TRAIN_LITERAL,
    _bound_program,
    _items_and_manifest,
)
from tests.test_portable_capability_runtime import (
    BENCHMARK_ROOT,
    INPUT_LOCATOR,
    FakeDockerSubprocess,
    _copy_skills_to_dest,
)


ITEM_ID = f"{STOCK_FAMILY}-3"


def _compiled_bundle(tmp_path: Path):
    program, typed_registry = _bound_program()
    items, manifest = _items_and_manifest()
    compiler = SkillLearnProgramCompiler(
        typed_program_registry=typed_registry,
        require_typed_bindings=True,
        portable_capability_compiler_mode=(
            PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
    )
    compiled = compiler.compile(
        programs=(program,),
        items=items,
        split_manifest=manifest,
        output_root=tmp_path / "base-compile",
        allowed_statuses={HypothesisStatus.CANDIDATE},
        target_item_ids=manifest.train_ids,
        target_split="train",
    )
    bound = typed_registry.require_bound_recipe(program)
    residuals = tuple(
        ResidualExample(
            transition_id=f"train-transition-{index}",
            task_id=f"train-support-{index}",
            family=STOCK_FAMILY,
            split=SplitName.TRAIN,
            features={"family": STOCK_FAMILY},
            failure_type="policy_off_failure",
            evaluator_feedback=(),
            baseline_success=False,
            context={},
        )
        for index in range(2)
    )
    contract = derive_train_execution_contract(
        graph=bound.snapshot.graph,
        recipe_id=bound.recipe.recipe_id,
        residuals=residuals,
    )
    contract_registry = TypedExecutionContractRegistry()
    contract_registry.register(contract, graph=bound.snapshot.graph)
    bundle = build_execution_contract_compile_bundle_v2(
        base_compile_result=compiled,
        programs=(program,),
        items=items,
        typed_program_registry=typed_registry,
        execution_contract_registry=contract_registry,
        output_root=tmp_path / "contract-bundle",
    )
    return compiled, bundle, contract, manifest


def _request_and_context(tmp_path: Path, *, sink: MemoryEventSink):
    compiled, bundle, contract, manifest = _compiled_bundle(tmp_path)
    source = compiled.source_for(ITEM_ID)
    assert source is not None
    source_receipt = compiled.source_receipt_for(ITEM_ID)
    role_hashes = compiled.item_portable_capability_role_spec_hashes[
        stable_hash({"item_id": ITEM_ID})
    ]
    request = SkillLearnTrialRequest(
        item_id=ITEM_ID,
        family=STOCK_FAMILY,
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_ON,
        evaluator_epoch="execution-contract-v2-fixture",
        pair_id="execution-contract-v2-pair",
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash=manifest.manifest_hash,
        program_id=compiled.hypothesis_ids[0],
        program_set_hash=compiled.program_set_hash,
        treatment_hash=compiled.treatment_hash_for(ITEM_ID),
        compile_manifest_hash=compiled.manifest_hash,
        skill_source_receipt_hash=source_receipt.receipt_hash,
        compile_root=compiled.output_root,
        typed_binding_set_hash=compiled.typed_binding_set_hash,
        typed_snapshot_hashes=compiled.typed_snapshot_hashes,
        typed_snapshot_ledger_hash=compiled.typed_snapshot_ledger_hash,
        portable_capability_compiler_mode=(
            compiled.portable_capability_compiler_mode
        ),
        portable_capability_role_spec_set_hash=(
            compiled.portable_capability_role_spec_set_hash
        ),
        portable_capability_role_spec_hashes=role_hashes,
        portable_capability_delivery_mode="",
    )
    backend = ExecutionContractSubprocessBackendV2(
        BENCHMARK_ROOT,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
        event_sink=sink,
        execution_contract_bundle=bundle,
    )
    base_context = backend._load_portable_task_capability_context(
        request=request,
        source_receipt=source_receipt,
        compile_root=compiled.output_root,
    )
    assert isinstance(base_context, PortableTaskCapabilityRuntimeContext)
    return (
        compiled,
        bundle,
        contract,
        request,
        backend,
        base_context,
        source,
    )


def _fake_runner(
    tmp_path: Path,
    *,
    delegate: FakeDockerSubprocess,
) -> ModuleType:
    runner = ModuleType("fake_execution_contract_v2_runner")
    runner.subprocess = delegate
    runner._copy_skills_to_dest = _copy_skills_to_dest
    agent = {
        "run": 'codex exec --json --model {model} -- "$(cat {instruction_file})"'
    }
    runner.get_agent = lambda agent_id: agent if agent_id == "codex" else None

    def inject(container_name, skill_source_dir, copies):
        assert container_name == "trial"
        for source_pattern, destination in copies:
            assert source_pattern == "skills"
            assert _copy_skills_to_dest(
                Path(skill_source_dir),
                delegate.container_path(destination),
            )

    runner._inject_skills_runtime = inject
    return runner


def _container_delegate(tmp_path: Path) -> FakeDockerSubprocess:
    container_root = tmp_path / "container"
    input_path = container_root / "root/data/stock-descriptions.tsv"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        "ticker\tname\tsector\tmarketCap\n"
        "AAA\tAlpha\tTechnology\t1000\n"
        "BBB\tBeta\tFinance\t2000\n",
        encoding="utf-8",
    )
    return FakeDockerSubprocess(container_root)


def test_companion_bundle_preserves_v1_and_delivers_contract_prompt(
    tmp_path: Path,
) -> None:
    sink = MemoryEventSink()
    (
        compiled,
        bundle,
        contract,
        request,
        backend,
        context,
        source,
    ) = _request_and_context(tmp_path, sink=sink)
    delegate = _container_delegate(tmp_path)
    runner = _fake_runner(tmp_path, delegate=delegate)
    backend._install_treatment_receipt_adapter(runner)
    runner._assumption_v2_task_capability_context = context

    runner._inject_skills_runtime(
        "trial",
        source,
        [("skills", "/root/.codex/skills")],
    )

    bundle.verify()
    receipt = runner._assumption_v2_execution_contract_prompt_receipt
    assert receipt.request_hash == request.request_hash
    assert receipt.bundle_manifest_hash == bundle.manifest_hash
    assert receipt.contract_hashes == (contract.contract_hash,)
    assert receipt.safe_payload()[
        "execution_contracts_present_in_effective_launch_prompt"
    ] is True
    assert receipt.safe_payload()["semantic_consumption_claimed"] is False
    assert receipt.safe_payload()["runtime_enforcement_claimed"] is False
    assert runner.get_agent("codex")["run"].count(
        EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH
    ) == 1
    fragment = delegate.container_path(
        EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH
    ).read_text(encoding="utf-8")
    assert "observable_interaction_postcondition" in fragment
    assert "observable visible state change" in fragment
    assert "bounded_repair" in fragment
    assert "max_search_evaluations" in fragment
    assert TRAIN_LITERAL not in fragment
    assert INPUT_LOCATOR not in fragment
    bundle_text = (bundle.root / EXECUTION_CONTRACT_BUNDLE_FILENAME).read_text()
    assert TRAIN_LITERAL not in bundle_text
    assert STOCK_FAMILY not in bundle_text
    assert ITEM_ID not in bundle_text
    assert compiled.manifest_hash == bundle.manifest[
        "base_compile_manifest_hash"
    ]
    assert bundle.manifest["frozen_v1_files_modified"] is False
    injected = [
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_pre_agent_execution_contract_prompt_v2_injected"
    ]
    assert len(injected) == 1
    assert injected[0]["payload"]["delivery_policy"] == (
        EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
    )


def test_prompt_keeps_each_profile_bound_to_its_contract(
    tmp_path: Path,
) -> None:
    sink = MemoryEventSink()
    (
        _compiled,
        bundle,
        contract_a,
        request,
        _backend,
        context,
        _source,
    ) = _request_and_context(tmp_path, sink=sink)
    contract_b = replace(
        contract_a,
        recipe_id="recipe_" + stable_hash({"recipe": "second"})[:20],
    )
    assert not contract_b.validate_closed()
    profile_bytes = b'{"value":"verified"}\n'
    output_hash = hashlib.sha256(profile_bytes).hexdigest()

    def profile(metadata_hash: str) -> VerifiedRuntimeProfile:
        return VerifiedRuntimeProfile(
            metadata_hash=metadata_hash,
            item_id_hash=stable_hash({"item_id": request.item_id}),
            role_spec_hash=stable_hash({"role": metadata_hash}),
            effect_receipt_hash=stable_hash({"effect": metadata_hash}),
            output_sha256=output_hash,
            profile_bytes=profile_bytes,
        )

    profile_low = profile("0" * 64)
    profile_high = profile("f" * 64)
    capsule = build_execution_contract_prompt_capsule_v2(
        request_hash=request.request_hash,
        base_runtime_context_hash=context.context_hash,
        source_receipt_hash=context.source_receipt_hash,
        typed_binding_set_hash=context.typed_binding_set_hash,
        public_instruction_hash=context.public_instruction_hash,
        bundle_manifest_hash=bundle.manifest_hash,
        profiles=(profile_high, profile_low),
        contracts=(contract_b, contract_a),
    )
    assert capsule.profiles == (profile_low, profile_high)
    assert capsule.contracts == (contract_a, contract_b)
    assert [
        row["execution_contract_hash"]
        for row in capsule.profile_contract_bindings
    ] == [contract_a.contract_hash, contract_b.contract_hash]
    bound = bind_execution_contract_prompt_v2(
        capsule,
        container_readback=capsule.fragment_bytes,
        run_template=(
            'codex exec --json --model {model} -- "$(cat {instruction_file})"'
        ),
        public_instruction=context.public_instruction,
    )
    assert bound.receipt.profile_count == 2
    assert bound.receipt.contract_hashes == tuple(
        sorted((contract_a.contract_hash, contract_b.contract_hash))
    )
    assert len(bound.receipt.profile_contract_binding_hashes) == 2


def test_expected_route_missing_prompt_receipt_invalidates_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = MemoryEventSink()
    (
        _compiled,
        _bundle,
        _contract,
        request,
        backend,
        _context,
        source,
    ) = _request_and_context(tmp_path, sink=sink)

    def run_without_receipt(
        _backend: SkillLearnSubprocessBackend,
        current_request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        assert skill_source_dir == source
        assert trace_id == "missing-receipt"
        return SkillLearnTrialObservation(
            request=current_request,
            success=True,
            score=1.0,
            metrics={"evaluation_valid": 1.0},
            total_tokens=10,
            steps=1,
            duration_seconds=0.1,
            provider_fingerprint="provider",
            fairness_fingerprint="fairness",
        )

    monkeypatch.setattr(SkillLearnSubprocessBackend, "run", run_without_receipt)
    observation = backend.run(
        request,
        skill_source_dir=source,
        trace_id="missing-receipt",
    )
    assert observation.success is False
    assert observation.valid is False
    assert observation.error_type == (
        "execution_contract_prompt_delivery_missing"
    )
    assert observation.metrics == {"evaluation_valid": 0.0}
    blocked = [
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_execution_contract_trial_blocked_missing_receipt_v2"
    ]
    assert len(blocked) == 1
    completed = [
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_execution_contract_trial_completed_v2"
    ]
    assert len(completed) == 1
    assert completed[0]["payload"]["observation_hash"] == (
        observation.observation_hash
    )
    assert completed[0]["payload"]["valid"] is False
    assert completed[0]["payload"][
        "supersedes_frozen_v1_trial_completed"
    ] is True
    assert getattr(backend._execution_contract_local, "context", None) is None
    assert getattr(backend._execution_contract_local, "receipt", None) is None


def test_successful_run_overlays_verified_v2_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = MemoryEventSink()
    (
        _compiled,
        bundle,
        _contract,
        request,
        backend,
        base_context,
        source,
    ) = _request_and_context(tmp_path, sink=sink)
    delegate = _container_delegate(tmp_path)
    runner = _fake_runner(tmp_path, delegate=delegate)
    backend._install_treatment_receipt_adapter(runner)
    runner._assumption_v2_task_capability_context = base_context
    runner._inject_skills_runtime(
        "trial",
        source,
        [("skills", "/root/.codex/skills")],
    )
    runtime_context = backend._execution_contract_local.context
    receipt = runner._assumption_v2_execution_contract_prompt_receipt
    assert receipt.bundle_manifest_hash == bundle.manifest_hash

    def run_with_receipt(
        _backend: SkillLearnSubprocessBackend,
        current_request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        assert skill_source_dir == source
        assert trace_id == "verified-receipt"
        backend._execution_contract_local.context = runtime_context
        backend._execution_contract_local.receipt = receipt
        return SkillLearnTrialObservation(
            request=current_request,
            success=True,
            score=1.0,
            metrics={"evaluation_valid": 1.0},
            total_tokens=10,
            steps=1,
            duration_seconds=0.1,
            provider_fingerprint="provider",
            fairness_fingerprint="fairness",
        )

    monkeypatch.setattr(SkillLearnSubprocessBackend, "run", run_with_receipt)
    observation = backend.run(
        request,
        skill_source_dir=source,
        trace_id="verified-receipt",
    )
    assert observation.valid is True
    assert observation.success is True
    assert observation.runtime_profile_prompt_delivery_policy == (
        EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
    )
    assert observation.runtime_profile_prompt_injection_receipt_hash == (
        receipt.receipt_hash
    )
    completed = [
        row
        for row in sink.events
        if row["event"]
        == "skilllearn_execution_contract_trial_completed_v2"
    ]
    assert len(completed) == 1
    assert completed[0]["payload"]["prompt_receipt_valid"] is True
    assert completed[0]["payload"]["observation_hash"] == (
        observation.observation_hash
    )


def test_policy_off_run_remains_frozen_v1_passthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = MemoryEventSink()
    (
        _compiled,
        _bundle,
        _contract,
        policy_on_request,
        backend,
        _context,
        _source,
    ) = _request_and_context(tmp_path, sink=sink)
    request = replace(policy_on_request, variant=TrialVariant.POLICY_OFF)
    expected = SkillLearnTrialObservation(
        request=request,
        success=True,
        score=1.0,
        metrics={"evaluation_valid": 1.0},
        total_tokens=10,
        steps=1,
        duration_seconds=0.1,
        provider_fingerprint="provider",
        fairness_fingerprint="fairness",
    )

    def run_policy_off(
        _backend: SkillLearnSubprocessBackend,
        current_request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        assert current_request == request
        assert skill_source_dir is None
        assert trace_id == "policy-off"
        return expected

    monkeypatch.setattr(SkillLearnSubprocessBackend, "run", run_policy_off)
    observation = backend.run(
        request,
        skill_source_dir=None,
        trace_id="policy-off",
    )
    assert observation == expected
    assert not [
        row
        for row in sink.events
        if row["event"].startswith("skilllearn_execution_contract_trial_")
    ]


def test_companion_bundle_and_prompt_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    sink = MemoryEventSink()
    (
        _compiled,
        bundle,
        _contract,
        _request,
        backend,
        context,
        source,
    ) = _request_and_context(tmp_path, sink=sink)
    payload = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    with pytest.raises(TypeError, match="manifest is immutable"):
        bundle.manifest["item_routes"].clear()
    payload["runtime_enforcement_claimed"] = True
    bundle.manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        bundle.verify()
    with pytest.raises(PermissionError):
        bundle.has_item(ITEM_ID)

    fresh_root = tmp_path / "fresh"
    fresh_sink = MemoryEventSink()
    (
        _compiled,
        _bundle,
        _contract,
        _request,
        fresh_backend,
        fresh_context,
        fresh_source,
    ) = _request_and_context(fresh_root, sink=fresh_sink)

    class TamperPromptReadback(FakeDockerSubprocess):
        def _copy_from_container(self, source_value, destination):
            super()._copy_from_container(source_value, destination)
            if EXECUTION_CONTRACT_PROMPT_CONTAINER_PATH in source_value:
                path = Path(destination)
                path.write_bytes(path.read_bytes() + b"tampered\n")

    normal_delegate = _container_delegate(fresh_root)
    delegate = TamperPromptReadback(normal_delegate.container_root)
    runner = _fake_runner(fresh_root, delegate=delegate)
    fresh_backend._install_treatment_receipt_adapter(runner)
    runner._assumption_v2_task_capability_context = fresh_context
    with pytest.raises(
        SkillLearnAgentTerminalError,
        match="execution_contract_prompt_delivery_invalid",
    ):
        runner._inject_skills_runtime(
            "trial",
            fresh_source,
            [("skills", "/root/.codex/skills")],
        )
    assert runner._assumption_v2_execution_contract_prompt_receipt is None
    blocked = [
        row
        for row in fresh_sink.events
        if row["event"]
        == "skilllearn_trial_blocked_invalid_execution_contract_prompt_v2"
    ]
    assert len(blocked) == 1

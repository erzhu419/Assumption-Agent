from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from assumption_agent.benchmarks.execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
    ExecutionContractSubprocessBackendV2,
    ExecutionContractTrialEvidenceV2,
)
from assumption_agent.benchmarks.execution_contract_prompt_v2 import (
    EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION,
    ExecutionContractPromptInjectionReceiptV2,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnProviderCircuit,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.benchmarks.train_outcome_production_runner_v2 import (
    PROVIDER_MODEL_CAPACITY_ERROR_TYPE,
    ProductionTrainCandidateRunnerV2,
    ProductionTrainProviderCapacityError,
    ProductionTrainRunnerError,
    classify_v2_provider_capacity_terminal,
)
from assumption_agent.benchmarks.train_outcome_ranker_v2 import (
    FrozenRawTrainBaselineSetV2,
    TrainCandidateSpecV2,
    TrainCandidateWorkUnitV2,
    TrainOutcomeRankerV2,
)
from assumption_agent.models import SplitName, stable_hash
from tests.test_execution_contract_integration_v2 import _compiled_bundle
from tests.test_portable_capability_runtime import BENCHMARK_ROOT


EVALUATOR_EPOCH = "production-train-runner-v2-fixture"
SOURCE_RECEIPT_HASH = stable_hash({"source": "production-raw-train"})
PROVIDER_FINGERPRINT = "production-provider-fingerprint"
FAIRNESS_FINGERPRINT = "production-fairness-fingerprint"
OFFLINE_PROFILE = "production-offline-profile"
OFFLINE_RUNTIME = "production-offline-runtime"


def _digest(label: str) -> str:
    return stable_hash({"production-fixture": label})


def _baseline_set(manifest) -> FrozenRawTrainBaselineSetV2:
    observations = []
    for index, item_id in enumerate(manifest.train_ids):
        request = SkillLearnTrialRequest(
            item_id=item_id,
            family="stock-data-visualization",
            split=SplitName.TRAIN,
            variant=TrialVariant.POLICY_OFF,
            evaluator_epoch=EVALUATOR_EPOCH,
            pair_id=_digest(f"raw-pair:{item_id}")[:20],
            repeat=1,
            agent_id="codex",
            model="gpt-5.4-mini",
            max_steps=100,
            manifest_hash=manifest.manifest_hash,
        )
        success = index % 2 == 0
        observations.append(
            SkillLearnTrialObservation(
                request=request,
                success=success,
                score=float(success),
                metrics={
                    "evaluation_valid": 1.0,
                    "task_success": float(success),
                },
                total_tokens=10,
                steps=1,
                duration_seconds=0.1,
                provider_fingerprint=PROVIDER_FINGERPRINT,
                fairness_fingerprint=FAIRNESS_FINGERPRINT,
                upstream_result_hash=_digest(f"raw-result:{item_id}"),
                offline_verifier_profile_id=OFFLINE_PROFILE,
                offline_verifier_runtime_key=OFFLINE_RUNTIME,
            )
        )
    return FrozenRawTrainBaselineSetV2.from_observations(
        observations,
        manifest_hash=manifest.manifest_hash,
        evaluator_epoch=EVALUATOR_EPOCH,
        source_train_receipt_hash=SOURCE_RECEIPT_HASH,
        expected_item_ids=manifest.train_ids,
    )


def _prompt_receipt(
    work: TrainCandidateWorkUnitV2,
    request: SkillLearnTrialRequest,
) -> ExecutionContractPromptInjectionReceiptV2:
    route = work.candidate.route_for_item_hash(
        work.baseline.item_id_hash
    )
    profile_count = len(route.profile_contract_binding_hashes)
    return ExecutionContractPromptInjectionReceiptV2(
        capsule_hash=_digest(f"capsule:{work.work_unit_hash}"),
        request_hash=request.request_hash,
        base_runtime_context_hash=_digest(
            f"base-context:{work.work_unit_hash}"
        ),
        source_receipt_hash=route.source_receipt_hash,
        typed_binding_set_hash=work.candidate.typed_binding_set_hash,
        public_instruction_hash=_digest(
            f"public-instruction:{work.work_unit_hash}"
        ),
        bundle_manifest_hash=(
            work.candidate.compile_bundle_manifest_hash
        ),
        profile_set_hash=_digest(f"profiles:{work.work_unit_hash}"),
        profile_count=profile_count,
        effect_receipt_hashes=tuple(
            _digest(f"effect:{work.work_unit_hash}:{index}")
            for index in range(profile_count)
        ),
        profile_output_sha256s=tuple(
            _digest(f"profile-output:{work.work_unit_hash}:{index}")
            for index in range(profile_count)
        ),
        contract_set_hash=route.prompt_contract_set_hash,
        contract_hashes=route.prompt_contract_hashes,
        profile_contract_binding_set_hash=(
            route.profile_contract_binding_set_hash
        ),
        profile_contract_binding_hashes=(
            route.profile_contract_binding_hashes
        ),
        fragment_sha256=_digest(f"fragment:{work.work_unit_hash}"),
        fragment_size=2048,
        container_path_hash=_digest("container-path"),
        container_readback_sha256=_digest(
            f"container-readback:{work.work_unit_hash}"
        ),
        run_template_before_hash=_digest("template-before"),
        run_template_after_hash=_digest("template-after"),
        effective_prompt_sha256=_digest(
            f"effective-prompt:{work.work_unit_hash}"
        ),
    )


class FakeProductionBackend(ExecutionContractSubprocessBackendV2):
    def __init__(
        self,
        *,
        work: TrainCandidateWorkUnitV2,
        bundle: ExecutionContractCompileBundleV2,
        barrier: threading.Barrier | None = None,
        record_upstream: bool = False,
        route_expected: bool = True,
    ) -> None:
        super().__init__(
            BENCHMARK_ROOT,
            model="gpt-5.4-mini",
            provider_mode="openai_compatible",
            record_upstream=record_upstream,
            execution_contract_bundle=bundle,
        )
        self.work = work
        self.barrier = barrier
        self.route_expected = route_expected
        self.calls = 0
        self.requests: list[SkillLearnTrialRequest] = []
        self.sources: list[Path | None] = []

    def run_with_evidence(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> ExecutionContractTrialEvidenceV2:
        self.calls += 1
        self.requests.append(request)
        self.sources.append(skill_source_dir)
        assert trace_id.startswith("production-test:")
        assert request.program_id is None
        assert skill_source_dir is not None
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        receipt = _prompt_receipt(self.work, request)
        observation = SkillLearnTrialObservation(
            request=request,
            success=True,
            score=1.0,
            metrics={"evaluation_valid": 1.0, "task_success": 1.0},
            total_tokens=20,
            steps=2,
            duration_seconds=0.2,
            provider_fingerprint=PROVIDER_FINGERPRINT,
            fairness_fingerprint=FAIRNESS_FINGERPRINT,
            upstream_result_hash=_digest(
                f"candidate-result:{request.request_hash}"
            ),
            raw_trial_artifacts_persisted=self.record_upstream,
            offline_verifier_profile_id=OFFLINE_PROFILE,
            offline_verifier_runtime_key=OFFLINE_RUNTIME,
            installed_skill_source_receipt_hash=(
                request.skill_source_receipt_hash
            ),
            runtime_profile_prompt_delivery_policy=(
                EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
            ),
            runtime_profile_prompt_injection_receipt_hash=(
                receipt.receipt_hash
            ),
            runtime_profile_effective_prompt_sha256=(
                receipt.effective_prompt_sha256
            ),
        )
        return ExecutionContractTrialEvidenceV2(
            observation=observation,
            prompt_receipt=receipt,
            execution_backend_instance_hash=(
                self.execution_backend_instance_hash
            ),
            contract_route_expected=self.route_expected,
            prompt_receipt_valid=True,
        )


def _fixture(tmp_path: Path):
    compiled, bundle, _contract, manifest = _compiled_bundle(tmp_path)
    candidate = TrainCandidateSpecV2.from_verified_bundle(
        candidate_id="production-candidate",
        bundle=bundle,
        static_complexity=3,
    )
    baseline_set = _baseline_set(manifest)
    return compiled, bundle, candidate, baseline_set


def test_production_runner_builds_exact_requests_and_runs_flat_grid(
    tmp_path: Path,
) -> None:
    compiled, bundle, candidate, baseline_set = _fixture(tmp_path)
    barrier = threading.Barrier(len(candidate.item_routes))
    created: list[FakeProductionBackend] = []
    created_lock = threading.Lock()

    def factory(work, current_bundle):
        backend = FakeProductionBackend(
            work=work,
            bundle=current_bundle,
            barrier=barrier,
        )
        with created_lock:
            created.append(backend)
        return backend

    runner = ProductionTrainCandidateRunnerV2(
        baseline_set=baseline_set,
        candidate_bundles={candidate.candidate_hash: bundle},
        backend_factory=factory,
        trace_prefix="production-test",
    )
    result = TrainOutcomeRankerV2().rank(
        baseline_set=baseline_set,
        candidates=(candidate,),
        runner=runner,
    )

    assert len(result.outcomes) == len(baseline_set.rows)
    assert len(result.run_results) == len(candidate.item_routes)
    assert result.maximum_concurrent_runner_calls == len(
        candidate.item_routes
    )
    assert runner.retained_backend_count == len(candidate.item_routes)
    assert len(runner.backend_instance_hashes) == len(candidate.item_routes)
    assert len(created) == len(candidate.item_routes)
    assert all(backend.calls == 1 for backend in created)
    for backend in created:
        request = backend.requests[0]
        item_hash = stable_hash({"item_id": request.item_id})
        route = candidate.route_for_item_hash(item_hash)
        assert request.program_id is None
        assert request.compile_root == compiled.output_root
        assert request.program_set_hash == candidate.program_set_hash
        assert request.compile_manifest_hash == compiled.manifest_hash
        assert request.treatment_hash == route.treatment_hash
        assert request.skill_source_receipt_hash == route.source_receipt_hash
        assert request.typed_binding_set_hash == (
            candidate.typed_binding_set_hash
        )
        assert request.typed_snapshot_hashes == (
            candidate.typed_snapshot_hashes
        )
        assert request.typed_snapshot_ledger_hash == (
            candidate.typed_snapshot_ledger_hash
        )
        assert request.portable_capability_role_spec_hashes == (
            compiled.item_portable_capability_role_spec_hashes[item_hash]
        )


def test_backend_reuse_or_raw_recording_fails_before_second_model_call(
    tmp_path: Path,
) -> None:
    _compiled, bundle, candidate, baseline_set = _fixture(tmp_path)
    works = tuple(
        TrainCandidateWorkUnitV2(candidate, baseline)
        for baseline in baseline_set.rows
    )
    assert all(work.candidate_active for work in works)

    raw_backend = FakeProductionBackend(
        work=works[0],
        bundle=bundle,
        record_upstream=True,
    )
    raw_runner = ProductionTrainCandidateRunnerV2(
        baseline_set=baseline_set,
        candidate_bundles={candidate.candidate_hash: bundle},
        backend_factory=lambda _work, _bundle: raw_backend,
        trace_prefix="production-test",
    )
    with pytest.raises(ProductionTrainRunnerError):
        raw_runner(works[0])
    assert raw_backend.calls == 0

    shared = FakeProductionBackend(work=works[0], bundle=bundle)
    shared_runner = ProductionTrainCandidateRunnerV2(
        baseline_set=baseline_set,
        candidate_bundles={candidate.candidate_hash: bundle},
        backend_factory=lambda _work, _bundle: shared,
        trace_prefix="production-test",
    )
    shared_runner(works[0])
    with pytest.raises(ProductionTrainRunnerError):
        shared_runner(works[1])
    assert shared.calls == 1


def test_unbound_execution_evidence_fails_closed(tmp_path: Path) -> None:
    _compiled, bundle, candidate, baseline_set = _fixture(tmp_path)
    work = TrainCandidateWorkUnitV2(candidate, baseline_set.rows[0])

    def factory(current_work, current_bundle):
        return FakeProductionBackend(
            work=current_work,
            bundle=current_bundle,
            route_expected=False,
        )

    runner = ProductionTrainCandidateRunnerV2(
        baseline_set=baseline_set,
        candidate_bundles={candidate.candidate_hash: bundle},
        backend_factory=factory,
        trace_prefix="production-test",
    )
    with pytest.raises(ProductionTrainRunnerError):
        runner(work)


@pytest.mark.parametrize(
    "message",
    (
        "Selected model is at capacity. Please try a different model.",
        "The selected model is at capacity. Please try again later.",
        "Model is at capacity. Please try again later.",
    ),
)
def test_v2_exact_capacity_terminals_are_nonfatal(message: str) -> None:
    top_level = json.dumps({"type": "error", "message": message})
    nested = json.dumps(
        {"type": "turn.failed", "error": {"message": message}}
    )

    assert (
        classify_v2_provider_capacity_terminal(top_level, nested)
        == PROVIDER_MODEL_CAPACITY_ERROR_TYPE
    )
    circuit = SkillLearnProviderCircuit()
    assert circuit.open(PROVIDER_MODEL_CAPACITY_ERROR_TYPE) is False
    assert circuit.error_type is None


@pytest.mark.parametrize(
    "stream",
    (
        "Selected model is at capacity. Please try a different model.",
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": (
                        "Selected model is at capacity. "
                        "Please try a different model."
                    ),
                },
            }
        ),
        json.dumps(
            {
                "type": "turn.failed",
                "error": {
                    "message": (
                        "Selected model is at capacity because this request "
                        "violated a policy."
                    )
                },
            }
        ),
        json.dumps(
            {
                "type": "turn.failed",
                "error": {"message": "Model is not available."},
            }
        ),
    ),
)
def test_v2_capacity_classifier_rejects_nonexact_or_nonterminal_text(
    stream: str,
) -> None:
    assert classify_v2_provider_capacity_terminal(stream) is None


def test_production_runner_surfaces_capacity_before_frozen_boundary(
    tmp_path: Path,
) -> None:
    _compiled, bundle, candidate, baseline_set = _fixture(tmp_path)
    work = TrainCandidateWorkUnitV2(candidate, baseline_set.rows[0])
    created: list[FakeProductionBackend] = []

    class CapacityBackend(FakeProductionBackend):
        def run_with_evidence(
            self,
            request: SkillLearnTrialRequest,
            *,
            skill_source_dir: Path | None,
            trace_id: str,
        ) -> ExecutionContractTrialEvidenceV2:
            evidence = super().run_with_evidence(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            assert self.trials_dir is not None
            trace = self.trials_dir / "attempt" / "agent" / "codex.txt"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {
                            "message": (
                                "Selected model is at capacity. "
                                "Please try a different model."
                            )
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return replace(
                evidence,
                observation=replace(
                    evidence.observation,
                    success=False,
                    score=0.0,
                    metrics={"evaluation_valid": 0.0},
                    error_type="codex_turn_failed",
                ),
            )

    def factory(current_work, current_bundle):
        backend = CapacityBackend(
            work=current_work,
            bundle=current_bundle,
        )
        backend.trials_dir = tmp_path / "capacity-trials"
        created.append(backend)
        return backend

    runner = ProductionTrainCandidateRunnerV2(
        baseline_set=baseline_set,
        candidate_bundles={candidate.candidate_hash: bundle},
        backend_factory=factory,
        trace_prefix="production-test",
    )
    with pytest.raises(ProductionTrainProviderCapacityError) as caught:
        runner(work)

    assert caught.value.error_type == PROVIDER_MODEL_CAPACITY_ERROR_TYPE
    assert caught.value.request_hash == created[0].requests[0].request_hash
    assert caught.value.work_unit_hash == work.work_unit_hash
    assert created[0].provider_circuit.error_type is None

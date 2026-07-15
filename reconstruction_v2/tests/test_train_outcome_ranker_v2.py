from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import pytest

from assumption_agent.benchmarks.execution_contract_prompt_v2 import (
    EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION,
    ExecutionContractPromptInjectionReceiptV2,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.benchmarks.train_outcome_ranker_v2 import (
    OFFLINE_TRAIN_EVALUATION_MODE,
    FrozenRawTrainBaselineSetV2,
    TrainCandidateItemRouteV2,
    TrainCandidateRunResultV2,
    TrainCandidateSpecV2,
    TrainCandidateWorkUnitV2,
    TrainOutcomeRankerV2,
    TrainOutcomeRankingError,
    TrainProfileContractBindingV2,
)
from assumption_agent.models import SplitName, stable_hash
from tests.test_execution_contract_integration_v2 import _compiled_bundle


MANIFEST_HASH = stable_hash({"manifest": "train-outcome-v2"})
SOURCE_TRAIN_RECEIPT_HASH = stable_hash({"source": "frozen-raw-train"})
EVALUATOR_EPOCH = "offline-train-evaluator-v2"
FAMILY = "synthetic_family"
ITEM_IDS = tuple(f"train-item-{index}" for index in range(4))
PROVIDER_FINGERPRINT = "provider-route-gpt-5.4-mini"
FAIRNESS_FINGERPRINT = "same-model-same-action-budget"
OFFLINE_VERIFIER_PROFILE_ID = "local-family-test-verifier"
OFFLINE_VERIFIER_RUNTIME_KEY = "local-content-addressed-verifier-runtime"


def _digest(label: str) -> str:
    return stable_hash({"fixture": label})


def _request(
    *,
    item_id: str,
    variant: TrialVariant,
    program_set_hash: str = "",
    compile_manifest_hash: str = "",
    treatment_hash: str = "",
    skill_source_receipt_hash: str = "",
    typed_binding_set_hash: str = "",
    typed_snapshot_hashes: tuple[str, ...] = (),
    typed_snapshot_ledger_hash: str = "",
    split: SplitName = SplitName.TRAIN,
    candidate_id: str = "raw",
) -> SkillLearnTrialRequest:
    return SkillLearnTrialRequest(
        item_id=item_id,
        family=FAMILY,
        split=split,
        variant=variant,
        evaluator_epoch=EVALUATOR_EPOCH,
        pair_id=_digest(f"pair:{candidate_id}:{item_id}"),
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash=MANIFEST_HASH,
        program_set_hash=program_set_hash,
        compile_manifest_hash=compile_manifest_hash,
        treatment_hash=treatment_hash,
        skill_source_receipt_hash=skill_source_receipt_hash,
        typed_binding_set_hash=typed_binding_set_hash,
        typed_snapshot_hashes=typed_snapshot_hashes,
        typed_snapshot_ledger_hash=typed_snapshot_ledger_hash,
    )


def _observation(
    request: SkillLearnTrialRequest,
    *,
    success: bool,
    score: float,
    valid: bool = True,
    total_tokens: int = 10,
    prompt_receipt: ExecutionContractPromptInjectionReceiptV2 | None = None,
) -> SkillLearnTrialObservation:
    return SkillLearnTrialObservation(
        request=request,
        success=(success if valid else False),
        score=(score if valid else 0.0),
        metrics={
            "evaluation_valid": float(valid),
            "task_success": float(success if valid else False),
        },
        total_tokens=total_tokens,
        steps=1,
        duration_seconds=0.1,
        provider_fingerprint=PROVIDER_FINGERPRINT,
        fairness_fingerprint=FAIRNESS_FINGERPRINT,
        error_type=None if valid else "synthetic_candidate_invalid",
        upstream_result_hash=_digest(
            f"upstream:{request.request_hash}:{success}:{valid}:{score}"
        ),
        offline_verifier_profile_id=OFFLINE_VERIFIER_PROFILE_ID,
        offline_verifier_runtime_key=OFFLINE_VERIFIER_RUNTIME_KEY,
        runtime_profile_prompt_delivery_policy=(
            EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
            if prompt_receipt is not None
            else ""
        ),
        runtime_profile_prompt_injection_receipt_hash=(
            prompt_receipt.receipt_hash
            if prompt_receipt is not None
            else ""
        ),
        runtime_profile_effective_prompt_sha256=(
            prompt_receipt.effective_prompt_sha256
            if prompt_receipt is not None
            else ""
        ),
    )


def _baseline_set(
    successes: tuple[bool, ...] = (True, False, False, True),
) -> FrozenRawTrainBaselineSetV2:
    observations = tuple(
        _observation(
            _request(item_id=item_id, variant=TrialVariant.POLICY_OFF),
            success=success,
            score=float(success),
        )
        for item_id, success in zip(ITEM_IDS, successes, strict=True)
    )
    return FrozenRawTrainBaselineSetV2.from_observations(
        tuple(reversed(observations)),
        manifest_hash=MANIFEST_HASH,
        evaluator_epoch=EVALUATOR_EPOCH,
        source_train_receipt_hash=SOURCE_TRAIN_RECEIPT_HASH,
        expected_item_ids=ITEM_IDS,
    )


def _candidate(name: str, *, complexity: int) -> TrainCandidateSpecV2:
    contract_hash = _digest(f"contract:{name}")
    prompt_contract_set_hash = stable_hash(
        {"execution_contract_hashes": [contract_hash]}
    )
    program_set_hash = _digest(f"program-set:{name}")
    base_compile_manifest_hash = _digest(f"base-compile:{name}")
    typed_binding_set_hash = _digest(f"typed-binding-set:{name}")
    typed_snapshot_hashes = (_digest(f"typed-snapshot:{name}"),)
    typed_snapshot_ledger_hash = _digest(
        f"typed-snapshot-ledger:{name}"
    )
    item_routes = tuple(
        sorted(
            (
                TrainCandidateItemRouteV2(
                    item_id_hash=stable_hash({"item_id": item_id}),
                    item_route_hash=_digest(f"route:{name}:{item_id}"),
                    treatment_hash=_digest(
                        f"treatment:{name}:{item_id}"
                    ),
                    source_receipt_hash=_digest(
                        f"source:{name}:{item_id}"
                    ),
                    prompt_contract_hashes=(contract_hash,),
                    prompt_contract_set_hash=prompt_contract_set_hash,
                    profile_contract_bindings=(
                        TrainProfileContractBindingV2(
                            metadata_hash=_digest(
                                f"metadata:{name}:{item_id}"
                            ),
                            execution_contract_hash=contract_hash,
                        ),
                    ),
                )
                for item_id in ITEM_IDS
            ),
            key=lambda row: row.item_id_hash,
        )
    )
    candidate_behavior_hash = stable_hash(
        {
            "program_set_hash": program_set_hash,
            "base_compile_manifest_hash": base_compile_manifest_hash,
            "typed_binding_set_hash": typed_binding_set_hash,
            "execution_contract_set_hash": prompt_contract_set_hash,
            "item_routes": [row.safe_payload() for row in item_routes],
        }
    )
    return TrainCandidateSpecV2(
        candidate_id=name,
        candidate_behavior_hash=candidate_behavior_hash,
        program_set_hash=program_set_hash,
        base_compile_manifest_hash=base_compile_manifest_hash,
        typed_binding_set_hash=typed_binding_set_hash,
        typed_snapshot_hashes=typed_snapshot_hashes,
        typed_snapshot_ledger_hash=typed_snapshot_ledger_hash,
        compile_bundle_manifest_hash=_digest(f"bundle:{name}"),
        execution_contract_set_hash=prompt_contract_set_hash,
        item_routes=item_routes,
        static_complexity=complexity,
    )


def _prompt_receipt(
    work: TrainCandidateWorkUnitV2,
    request: SkillLearnTrialRequest,
) -> ExecutionContractPromptInjectionReceiptV2:
    route = work.candidate.route_for_item_hash(work.baseline.item_id_hash)
    effective_prompt_hash = _digest(
        f"effective-prompt:{work.work_unit_hash}"
    )
    return ExecutionContractPromptInjectionReceiptV2(
        capsule_hash=_digest(f"capsule:{work.work_unit_hash}"),
        request_hash=request.request_hash,
        base_runtime_context_hash=_digest(
            f"base-context:{work.work_unit_hash}"
        ),
        source_receipt_hash=route.source_receipt_hash,
        typed_binding_set_hash=work.candidate.typed_binding_set_hash,
        public_instruction_hash=_digest(
            f"instruction:{work.baseline.item_id}"
        ),
        bundle_manifest_hash=(
            work.candidate.compile_bundle_manifest_hash
        ),
        profile_set_hash=_digest(f"profiles:{work.work_unit_hash}"),
        profile_count=1,
        effect_receipt_hashes=(
            _digest(f"effect:{work.work_unit_hash}"),
        ),
        profile_output_sha256s=(
            _digest(f"profile-output:{work.work_unit_hash}"),
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
        fragment_size=1024,
        container_path_hash=_digest("container-path"),
        container_readback_sha256=_digest(
            f"readback:{work.work_unit_hash}"
        ),
        run_template_before_hash=_digest("template-before"),
        run_template_after_hash=_digest("template-after"),
        effective_prompt_sha256=effective_prompt_hash,
    )


SyntheticOutcome = tuple[bool, float, bool, int]


class SyntheticCandidateRunner:
    def __init__(
        self,
        outcomes: dict[str, dict[str, SyntheticOutcome]],
        *,
        barrier: threading.Barrier | None = None,
        reverse_delay: bool = False,
    ) -> None:
        self.outcomes = outcomes
        self.barrier = barrier
        self.reverse_delay = reverse_delay
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.calls: list[tuple[str, str, TrialVariant]] = []
        self.finished: list[str] = []

    def __call__(
        self,
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=10)
            delay_bucket = int(work.work_unit_hash[:2], 16) % 5
            if self.reverse_delay:
                delay_bucket = 4 - delay_bucket
            time.sleep(delay_bucket * 0.001)
            success, score, valid, total_tokens = self.outcomes[
                work.candidate.candidate_id
            ][work.baseline.item_id]
            route = work.candidate.route_for_item_hash(
                work.baseline.item_id_hash
            )
            request = _request(
                item_id=work.baseline.item_id,
                variant=TrialVariant.POLICY_ON,
                program_set_hash=work.candidate.program_set_hash,
                compile_manifest_hash=(
                    work.candidate.base_compile_manifest_hash
                ),
                treatment_hash=route.treatment_hash,
                skill_source_receipt_hash=route.source_receipt_hash,
                typed_binding_set_hash=(
                    work.candidate.typed_binding_set_hash
                ),
                typed_snapshot_hashes=(
                    work.candidate.typed_snapshot_hashes
                ),
                typed_snapshot_ledger_hash=(
                    work.candidate.typed_snapshot_ledger_hash
                ),
                candidate_id=work.candidate.candidate_id,
            )
            prompt_receipt = _prompt_receipt(work, request) if valid else None
            observation = _observation(
                request,
                success=success,
                score=score,
                valid=valid,
                total_tokens=total_tokens,
                prompt_receipt=prompt_receipt,
            )
            with self._lock:
                self.calls.append(
                    (
                        work.candidate.candidate_hash,
                        work.baseline.item_id,
                        request.variant,
                    )
                )
            result = TrainCandidateRunResultV2.from_observation(
                work,
                observation,
                execution_backend_instance_hash=_digest(
                    f"backend-instance:{work.work_unit_hash}"
                ),
                prompt_receipt=prompt_receipt,
            )
            with self._lock:
                self.finished.append(work.work_unit_hash)
            return result
        finally:
            with self._lock:
                self.active -= 1


def _utility_fixture():
    cheap_bad = _candidate("cheap_bad", complexity=0)
    middle = _candidate("middle", complexity=10)
    expensive_good = _candidate("expensive_good", complexity=900_000)
    outcomes: dict[str, dict[str, SyntheticOutcome]] = {
        "cheap_bad": {
            ITEM_IDS[0]: (False, 0.0, True, 5),
            ITEM_IDS[1]: (False, 0.0, True, 5),
            ITEM_IDS[2]: (False, 0.0, True, 5),
            ITEM_IDS[3]: (True, 1.0, True, 5),
        },
        "middle": {
            ITEM_IDS[0]: (True, 1.0, True, 10),
            ITEM_IDS[1]: (True, 1.0, True, 10),
            ITEM_IDS[2]: (False, 0.0, True, 10),
            ITEM_IDS[3]: (True, 1.0, True, 10),
        },
        "expensive_good": {
            item_id: (True, 1.0, True, 20) for item_id in ITEM_IDS
        },
    }
    return (cheap_bad, middle, expensive_good), outcomes


def test_candidate_identity_is_derived_from_verified_bundle(
    tmp_path: Path,
) -> None:
    compiled, bundle, _contract, _manifest = _compiled_bundle(tmp_path)
    candidate = TrainCandidateSpecV2.from_verified_bundle(
        candidate_id="bundle-candidate",
        bundle=bundle,
        static_complexity=7,
    )

    candidate.verify()
    assert candidate.program_set_hash == compiled.program_set_hash
    assert candidate.base_compile_manifest_hash == compiled.manifest_hash
    assert candidate.compile_bundle_manifest_hash == bundle.manifest_hash
    assert tuple(row.item_id_hash for row in candidate.item_routes) == tuple(
        sorted(bundle.manifest["item_routes"])
    )


def test_actual_utility_wins_and_all_candidate_train_units_run_in_parallel() -> None:
    baseline_set = _baseline_set()
    candidates, outcomes = _utility_fixture()
    work_count = len(candidates) * len(ITEM_IDS)
    runner = SyntheticCandidateRunner(
        outcomes,
        barrier=threading.Barrier(work_count),
    )
    result = TrainOutcomeRankerV2().rank(
        baseline_set=baseline_set,
        candidates=tuple(reversed(candidates)),
        runner=runner,
    )

    assert result.ordered_candidate_hashes == (
        candidates[2].candidate_hash,
        candidates[1].candidate_hash,
        candidates[0].candidate_hash,
    )
    assert result.top_candidate_hash == candidates[2].candidate_hash
    assert result.effective_worker_count == work_count
    assert result.maximum_concurrent_runner_calls == work_count
    assert runner.maximum_active == work_count
    assert len(runner.calls) == work_count
    assert len(set((candidate_hash, item_id) for candidate_hash, item_id, _ in runner.calls)) == work_count
    assert all(variant is TrialVariant.POLICY_ON for _, _, variant in runner.calls)
    payload = result.to_dict()
    assert payload["baseline_execution_count"] == 0
    assert payload["candidate_execution_count"] == work_count
    assert payload["evaluation_mode"] == OFFLINE_TRAIN_EVALUATION_MODE
    assert payload["online_judge_calls"] == 0
    assert payload["validation_accessed"] is False
    assert payload["test_accessed"] is False
    assert payload["promotion_gate_applied"] is False
    assert payload["promotion_authorized"] is False


def test_completion_order_and_input_order_do_not_change_ranking_receipt() -> None:
    baseline_set = _baseline_set()
    candidates, outcomes = _utility_fixture()
    forward_runner = SyntheticCandidateRunner(outcomes)
    reverse_runner = SyntheticCandidateRunner(outcomes, reverse_delay=True)

    first = TrainOutcomeRankerV2().rank(
        baseline_set=baseline_set,
        candidates=candidates,
        runner=forward_runner,
    )
    second = TrainOutcomeRankerV2(max_workers=1).rank(
        baseline_set=baseline_set,
        candidates=tuple(reversed(candidates)),
        runner=reverse_runner,
    )

    assert first.ranking_hash == second.ranking_hash
    assert first.outcome_set_hash == second.outcome_set_hash
    assert first.ordered_candidate_hashes == second.ordered_candidate_hashes
    assert first.safe_payload() == second.safe_payload()


def test_invalid_outcome_is_ranked_last_and_complexity_is_only_a_tiebreak() -> None:
    baseline_set = _baseline_set((False, False, False, False))
    low = _candidate("tie_low", complexity=1)
    high = _candidate("tie_high", complexity=100)
    invalid = _candidate("invalid", complexity=0)
    identical = {item_id: (True, 1.0, True, 10) for item_id in ITEM_IDS}
    outcomes = {
        "tie_low": dict(identical),
        "tie_high": dict(identical),
        "invalid": {
            **identical,
            ITEM_IDS[0]: (False, 0.0, False, 1),
        },
    }
    result = TrainOutcomeRankerV2().rank(
        baseline_set=baseline_set,
        candidates=(invalid, high, low),
        runner=SyntheticCandidateRunner(outcomes),
    )

    assert result.ordered_candidate_hashes == (
        low.candidate_hash,
        high.candidate_hash,
        invalid.candidate_hash,
    )
    aggregates = {row.candidate_hash: row for row in result.aggregates}
    assert aggregates[invalid.candidate_hash].invalid_count == 1
    assert aggregates[low.candidate_hash].ranking_key[:-2] == (
        aggregates[high.candidate_hash].ranking_key[:-2]
    )


def test_invalid_on_raw_success_counts_as_regression_before_recovery() -> None:
    baseline_set = _baseline_set((True, False, False, False))
    loses_success = _candidate("loses_success", complexity=0)
    preserves_success = _candidate("preserves_success", complexity=100)
    outcomes = {
        "loses_success": {
            ITEM_IDS[0]: (False, 0.0, False, 1),
            ITEM_IDS[1]: (True, 1.0, True, 10),
            ITEM_IDS[2]: (True, 1.0, True, 10),
            ITEM_IDS[3]: (True, 1.0, True, 10),
        },
        "preserves_success": {
            ITEM_IDS[0]: (True, 1.0, True, 10),
            ITEM_IDS[1]: (False, 0.0, False, 1),
            ITEM_IDS[2]: (False, 0.0, True, 10),
            ITEM_IDS[3]: (False, 0.0, True, 10),
        },
    }
    result = TrainOutcomeRankerV2().rank(
        baseline_set=baseline_set,
        candidates=(loses_success, preserves_success),
        runner=SyntheticCandidateRunner(outcomes),
    )
    aggregates = {row.candidate_hash: row for row in result.aggregates}

    assert aggregates[loses_success.candidate_hash].invalid_count == 1
    assert aggregates[preserves_success.candidate_hash].invalid_count == 1
    assert aggregates[loses_success.candidate_hash].regression_count == 1
    assert aggregates[preserves_success.candidate_hash].regression_count == 0
    assert result.top_candidate_hash == preserves_success.candidate_hash


def test_raw_baseline_rejects_non_train_invalid_and_duplicate_evidence() -> None:
    valid = _observation(
        _request(item_id=ITEM_IDS[0], variant=TrialVariant.POLICY_OFF),
        success=False,
        score=0.0,
    )
    validation = replace(
        valid,
        request=replace(valid.request, split=SplitName.VALIDATION),
    )
    policy_on = replace(
        valid,
        request=replace(valid.request, variant=TrialVariant.POLICY_ON),
    )
    invalid = replace(
        valid,
        metrics={"evaluation_valid": 0.0},
        error_type="offline_verifier_invalid",
    )
    for observation in (validation, policy_on, invalid):
        with pytest.raises(TrainOutcomeRankingError):
            FrozenRawTrainBaselineSetV2.from_observations(
                (observation,),
                manifest_hash=MANIFEST_HASH,
                evaluator_epoch=EVALUATOR_EPOCH,
                source_train_receipt_hash=SOURCE_TRAIN_RECEIPT_HASH,
                expected_item_ids=(ITEM_IDS[0],),
            )
    with pytest.raises(TrainOutcomeRankingError):
        FrozenRawTrainBaselineSetV2.from_observations(
            (valid, valid),
            manifest_hash=MANIFEST_HASH,
            evaluator_epoch=EVALUATOR_EPOCH,
            source_train_receipt_hash=SOURCE_TRAIN_RECEIPT_HASH,
            expected_item_ids=(ITEM_IDS[0],),
        )
    with pytest.raises(TrainOutcomeRankingError):
        FrozenRawTrainBaselineSetV2.from_observations(
            (valid,),
            manifest_hash=MANIFEST_HASH,
            evaluator_epoch=EVALUATOR_EPOCH,
            source_train_receipt_hash=SOURCE_TRAIN_RECEIPT_HASH,
            expected_item_ids=ITEM_IDS,
        )


def test_online_or_cross_split_candidate_receipts_fail_closed() -> None:
    baseline_set = _baseline_set((False, False, False, False))
    candidate = _candidate("candidate", complexity=1)
    outcomes = {
        "candidate": {
            item_id: (True, 1.0, True, 10) for item_id in ITEM_IDS
        }
    }
    normal_runner = SyntheticCandidateRunner(outcomes)

    def online_runner(
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        result = normal_runner(work)
        return replace(
            result,
            offline_evaluation=replace(
                result.offline_evaluation,
                evaluation_mode="online_judge",
                network_fallback_used=True,
                online_judge_calls=1,
            ),
        )

    with pytest.raises(TrainOutcomeRankingError):
        TrainOutcomeRankerV2().rank(
            baseline_set=baseline_set,
            candidates=(candidate,),
            runner=online_runner,
        )

    def wrong_bundle_runner(
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        result = normal_runner(work)
        assert result.prompt_receipt is not None
        return replace(
            result,
            prompt_receipt=replace(
                result.prompt_receipt,
                bundle_manifest_hash=_digest("wrong-bundle"),
            ),
        )

    with pytest.raises(TrainOutcomeRankingError):
        TrainOutcomeRankerV2().rank(
            baseline_set=baseline_set,
            candidates=(candidate,),
            runner=wrong_bundle_runner,
        )

    def binding_tamper_runner(
        field_name: str,
    ):
        def run(
            work: TrainCandidateWorkUnitV2,
        ) -> TrainCandidateRunResultV2:
            result = normal_runner(work)
            assert result.prompt_receipt is not None
            bad_receipt = replace(
                result.prompt_receipt,
                **{field_name: _digest(f"wrong:{field_name}")},
            )
            bad_observation = replace(
                result.observation,
                runtime_profile_prompt_injection_receipt_hash=(
                    bad_receipt.receipt_hash
                ),
            )
            return TrainCandidateRunResultV2.from_observation(
                work,
                bad_observation,
                execution_backend_instance_hash=(
                    result.execution_backend_instance_hash
                ),
                prompt_receipt=bad_receipt,
            )

        return run

    for field_name in (
        "typed_binding_set_hash",
        "profile_contract_binding_set_hash",
    ):
        with pytest.raises(TrainOutcomeRankingError):
            TrainOutcomeRankerV2().rank(
                baseline_set=baseline_set,
                candidates=(candidate,),
                runner=binding_tamper_runner(field_name),
            )

    def validation_runner(
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        route = candidate.route_for_item_hash(work.baseline.item_id_hash)
        request = _request(
            item_id=work.baseline.item_id,
            variant=TrialVariant.POLICY_ON,
            program_set_hash=candidate.program_set_hash,
            compile_manifest_hash=candidate.base_compile_manifest_hash,
            treatment_hash=route.treatment_hash,
            skill_source_receipt_hash=route.source_receipt_hash,
            typed_binding_set_hash=candidate.typed_binding_set_hash,
            typed_snapshot_hashes=candidate.typed_snapshot_hashes,
            typed_snapshot_ledger_hash=(
                candidate.typed_snapshot_ledger_hash
            ),
            split=SplitName.VALIDATION,
            candidate_id=candidate.candidate_id,
        )
        prompt_receipt = _prompt_receipt(work, request)
        observation = _observation(
            request,
            success=True,
            score=1.0,
            prompt_receipt=prompt_receipt,
        )
        return TrainCandidateRunResultV2.from_observation(
            work,
            observation,
            execution_backend_instance_hash=_digest(
                f"validation-backend:{work.work_unit_hash}"
            ),
            prompt_receipt=prompt_receipt,
        )

    with pytest.raises(TrainOutcomeRankingError):
        TrainOutcomeRankerV2().rank(
            baseline_set=baseline_set,
            candidates=(candidate,),
            runner=validation_runner,
        )


def test_missing_prompt_runner_exception_and_result_tamper_fail_closed() -> None:
    baseline_set = _baseline_set((False, False, False, False))
    low = _candidate("low", complexity=1)
    high = _candidate("high", complexity=2)
    outcomes = {
        name: {
            item_id: (True, 1.0, True, 10) for item_id in ITEM_IDS
        }
        for name in ("low", "high")
    }

    def missing_prompt(
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        route = work.candidate.route_for_item_hash(
            work.baseline.item_id_hash
        )
        request = _request(
            item_id=work.baseline.item_id,
            variant=TrialVariant.POLICY_ON,
            program_set_hash=work.candidate.program_set_hash,
            compile_manifest_hash=(
                work.candidate.base_compile_manifest_hash
            ),
            treatment_hash=route.treatment_hash,
            skill_source_receipt_hash=route.source_receipt_hash,
            typed_binding_set_hash=(
                work.candidate.typed_binding_set_hash
            ),
            typed_snapshot_hashes=(
                work.candidate.typed_snapshot_hashes
            ),
            typed_snapshot_ledger_hash=(
                work.candidate.typed_snapshot_ledger_hash
            ),
            candidate_id=work.candidate.candidate_id,
        )
        observation = _observation(
            request,
            success=True,
            score=1.0,
            prompt_receipt=None,
        )
        return TrainCandidateRunResultV2.from_observation(
            work,
            observation,
            execution_backend_instance_hash=_digest(
                f"missing-prompt-backend:{work.work_unit_hash}"
            ),
            prompt_receipt=None,
        )

    with pytest.raises(TrainOutcomeRankingError):
        TrainOutcomeRankerV2().rank(
            baseline_set=baseline_set,
            candidates=(low,),
            runner=missing_prompt,
        )

    def exploding_runner(
        _work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        raise RuntimeError("synthetic runner crash")

    with pytest.raises(
        TrainOutcomeRankingError,
        match="failed without a valid receipt",
    ):
        TrainOutcomeRankerV2().rank(
            baseline_set=baseline_set,
            candidates=(low,),
            runner=exploding_runner,
        )

    normal_runner = SyntheticCandidateRunner(outcomes)

    def shared_backend_runner(
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        return replace(
            normal_runner(work),
            execution_backend_instance_hash=_digest("shared-backend"),
        )

    with pytest.raises(
        TrainOutcomeRankingError,
        match="backend instances drifted",
    ):
        TrainOutcomeRankerV2().rank(
            baseline_set=baseline_set,
            candidates=(low,),
            runner=shared_backend_runner,
        )

    result = TrainOutcomeRankerV2().rank(
        baseline_set=baseline_set,
        candidates=(high, low),
        runner=SyntheticCandidateRunner(outcomes),
    )
    tampered = replace(
        result,
        ordered_candidate_hashes=tuple(
            reversed(result.ordered_candidate_hashes)
        ),
    )
    with pytest.raises(TrainOutcomeRankingError):
        tampered.verify()

    forged_outcomes = (
        replace(result.outcomes[0], work_unit_hash=_digest("forged-work")),
        *result.outcomes[1:],
    )
    forged_aggregates = tuple(
        replace(
            aggregate,
            outcome_hashes=tuple(
                row.outcome_hash
                for row in forged_outcomes
                if row.candidate_hash == aggregate.candidate_hash
            ),
        )
        for aggregate in result.aggregates
    )
    forged = replace(
        result,
        outcomes=forged_outcomes,
        aggregates=forged_aggregates,
        outcome_set_hash=stable_hash(
            {
                "outcome_hashes": [
                    row.outcome_hash for row in forged_outcomes
                ]
            }
        ),
    )
    with pytest.raises(TrainOutcomeRankingError):
        forged.verify()

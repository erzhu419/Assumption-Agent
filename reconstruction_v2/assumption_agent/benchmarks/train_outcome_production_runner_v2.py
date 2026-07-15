from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import threading

from ..models import SplitName, stable_hash
from .execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
    ExecutionContractSubprocessBackendV2,
    ExecutionContractTrialEvidenceV2,
)
from .skilllearn_lifecycle import (
    SkillLearnTrialRequest,
    TrialVariant,
)
from .train_outcome_ranker_v2 import (
    FrozenRawTrainBaselineSetV2,
    TrainCandidateRunResultV2,
    TrainCandidateSpecV2,
    TrainCandidateWorkUnitV2,
)


TRAIN_OUTCOME_PRODUCTION_RUNNER_VERSION = (
    "execution_contract_train_candidate_runner_v2"
)


class ProductionTrainRunnerError(PermissionError):
    """A production candidate run crossed its frozen TRAIN binding."""


ProductionBackendFactoryV2 = Callable[
    [TrainCandidateWorkUnitV2, ExecutionContractCompileBundleV2],
    ExecutionContractSubprocessBackendV2,
]


class ProductionTrainCandidateRunnerV2:
    """Build and execute one fully bound policy-on TRAIN work unit.

    The outer ``TrainOutcomeRankerV2`` owns flat candidate-by-item
    parallelism.  This adapter deliberately requests a new backend from the
    factory for every work unit and retains every instance until the complete
    ranking call is no longer using the adapter.
    """

    def __init__(
        self,
        *,
        baseline_set: FrozenRawTrainBaselineSetV2,
        candidate_bundles: Mapping[str, ExecutionContractCompileBundleV2],
        backend_factory: ProductionBackendFactoryV2,
        trace_prefix: str = "train-outcome-production-v2",
    ) -> None:
        baseline_set.verify()
        if not isinstance(trace_prefix, str) or not trace_prefix:
            raise ValueError("production TRAIN trace prefix is empty")
        if not callable(backend_factory):
            raise TypeError("production TRAIN backend factory is not callable")
        bundles = dict(candidate_bundles)
        if not bundles:
            raise ValueError("production TRAIN candidate bundle map is empty")
        for candidate_hash, bundle in bundles.items():
            if (
                not isinstance(candidate_hash, str)
                or len(candidate_hash) != 64
                or not isinstance(bundle, ExecutionContractCompileBundleV2)
            ):
                raise TypeError(
                    "production TRAIN candidate bundle map is invalid"
                )
            bundle.verify()
        self.baseline_set = baseline_set
        self._candidate_bundles = bundles
        self._backend_factory = backend_factory
        self.trace_prefix = trace_prefix
        self._backend_lock = threading.Lock()
        self._retained_backends: list[
            ExecutionContractSubprocessBackendV2
        ] = []
        self._backend_instance_hashes: set[str] = set()

    @property
    def retained_backend_count(self) -> int:
        with self._backend_lock:
            return len(self._retained_backends)

    @property
    def backend_instance_hashes(self) -> tuple[str, ...]:
        with self._backend_lock:
            return tuple(sorted(self._backend_instance_hashes))

    def _bundle_for(
        self,
        work: TrainCandidateWorkUnitV2,
    ) -> ExecutionContractCompileBundleV2:
        try:
            bundle = self._candidate_bundles[work.candidate.candidate_hash]
        except KeyError as exc:
            raise ProductionTrainRunnerError(
                "production TRAIN work has no candidate bundle"
            ) from exc
        reconstructed = TrainCandidateSpecV2.from_verified_bundle(
            candidate_id=work.candidate.candidate_id,
            bundle=bundle,
            static_complexity=work.candidate.static_complexity,
        )
        if reconstructed.safe_payload() != work.candidate.safe_payload():
            raise ProductionTrainRunnerError(
                "production TRAIN candidate does not match its bundle"
            )
        return bundle

    def _request_for(
        self,
        work: TrainCandidateWorkUnitV2,
        bundle: ExecutionContractCompileBundleV2,
    ) -> tuple[SkillLearnTrialRequest, Path]:
        baseline = work.baseline.observation
        baseline_request = baseline.request
        if (
            baseline_request.split is not SplitName.TRAIN
            or baseline_request.variant is not TrialVariant.POLICY_OFF
            or work.baseline.item_id_hash
            not in self.baseline_set.train_item_hashes
            or work.baseline.baseline_evidence_hash
            not in {
                row.baseline_evidence_hash
                for row in self.baseline_set.rows
            }
        ):
            raise ProductionTrainRunnerError(
                "production TRAIN work is outside the frozen baseline"
            )
        compiled = bundle.base_compile_result
        route = work.candidate.route_for_item_hash(
            work.baseline.item_id_hash
        )
        source = compiled.source_for(work.baseline.item_id)
        if source is None:
            raise ProductionTrainRunnerError(
                "production TRAIN item has no compiled source"
            )
        source_receipt = compiled.source_receipt_for(
            work.baseline.item_id
        )
        role_hashes = tuple(
            compiled.item_portable_capability_role_spec_hashes.get(
                work.baseline.item_id_hash,
                (),
            )
        )
        if (
            compiled.manifest_hash
            != work.candidate.base_compile_manifest_hash
            or compiled.program_set_hash
            != work.candidate.program_set_hash
            or compiled.typed_binding_set_hash
            != work.candidate.typed_binding_set_hash
            or compiled.typed_snapshot_hashes
            != work.candidate.typed_snapshot_hashes
            or compiled.typed_snapshot_ledger_hash
            != work.candidate.typed_snapshot_ledger_hash
            or compiled.treatment_hash_for(work.baseline.item_id)
            != route.treatment_hash
            or source_receipt.receipt_hash != route.source_receipt_hash
            or source_receipt.portable_capability_role_spec_hashes
            != role_hashes
            or not compiled.portable_capability_compiler_mode
            or not role_hashes
        ):
            raise ProductionTrainRunnerError(
                "production TRAIN compiled request provenance drifted"
            )
        pair_id = stable_hash(
            {
                "runner_policy": TRAIN_OUTCOME_PRODUCTION_RUNNER_VERSION,
                "candidate_hash": work.candidate.candidate_hash,
                "item_id_hash": work.baseline.item_id_hash,
                "baseline_evidence_hash": (
                    work.baseline.baseline_evidence_hash
                ),
            }
        )[:20]
        request = SkillLearnTrialRequest(
            item_id=work.baseline.item_id,
            family=work.baseline.family,
            split=SplitName.TRAIN,
            variant=TrialVariant.POLICY_ON,
            evaluator_epoch=self.baseline_set.evaluator_epoch,
            pair_id=pair_id,
            repeat=baseline_request.repeat,
            agent_id=baseline_request.agent_id,
            model=baseline_request.model,
            max_steps=baseline_request.max_steps,
            manifest_hash=self.baseline_set.manifest_hash,
            codex_agent_execution_policy_hash=(
                baseline_request.codex_agent_execution_policy_hash
            ),
            # A bundle can combine several local policies.  None preserves the
            # frozen lifecycle meaning "no single program asserted"; the full
            # program-set, treatment, source, metadata, and contract routes are
            # still verified exactly by the v2 sidecar.
            program_id=None,
            program_set_hash=work.candidate.program_set_hash,
            treatment_hash=route.treatment_hash,
            compile_manifest_hash=(
                work.candidate.base_compile_manifest_hash
            ),
            skill_source_receipt_hash=route.source_receipt_hash,
            compile_root=compiled.output_root,
            typed_binding_set_hash=work.candidate.typed_binding_set_hash,
            typed_snapshot_hashes=work.candidate.typed_snapshot_hashes,
            typed_snapshot_ledger_hash=(
                work.candidate.typed_snapshot_ledger_hash
            ),
            portable_capability_compiler_mode=(
                compiled.portable_capability_compiler_mode
            ),
            portable_capability_role_spec_set_hash=(
                compiled.portable_capability_role_spec_set_hash
            ),
            portable_capability_role_spec_hashes=role_hashes,
            portable_capability_delivery_mode="",
        )
        if request.request_hash == baseline_request.request_hash:
            raise ProductionTrainRunnerError(
                "production TRAIN candidate collapsed to its RAW request"
            )
        return request, source

    def _register_backend(
        self,
        backend: ExecutionContractSubprocessBackendV2,
        bundle: ExecutionContractCompileBundleV2,
        work: TrainCandidateWorkUnitV2,
    ) -> str:
        baseline_request = work.baseline.observation.request
        if (
            not isinstance(backend, ExecutionContractSubprocessBackendV2)
            or backend.execution_contract_bundle is not bundle
            or backend.execution_contract_bundle.manifest_hash
            != bundle.manifest_hash
            or backend.agent_id != baseline_request.agent_id
            or backend.model != baseline_request.model
            or backend.max_steps != baseline_request.max_steps
            or backend.codex_agent_execution_policy_hash
            != baseline_request.codex_agent_execution_policy_hash
            or backend.record_upstream is not False
        ):
            raise ProductionTrainRunnerError(
                "production TRAIN backend crossed its frozen configuration"
            )
        backend_hash = backend.execution_backend_instance_hash
        with self._backend_lock:
            if backend_hash in self._backend_instance_hashes:
                raise ProductionTrainRunnerError(
                    "production TRAIN backend instance was reused"
                )
            self._backend_instance_hashes.add(backend_hash)
            self._retained_backends.append(backend)
        return backend_hash

    def __call__(
        self,
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2:
        bundle = self._bundle_for(work)
        request, source = self._request_for(work, bundle)
        backend = self._backend_factory(work, bundle)
        backend_hash = self._register_backend(backend, bundle, work)
        evidence = backend.run_with_evidence(
            request,
            skill_source_dir=source,
            trace_id=f"{self.trace_prefix}:{work.work_unit_hash[:20]}",
        )
        if (
            not isinstance(evidence, ExecutionContractTrialEvidenceV2)
            or evidence.execution_backend_instance_hash != backend_hash
            or evidence.observation.request != request
            or evidence.contract_route_expected is not True
        ):
            raise ProductionTrainRunnerError(
                "production TRAIN backend returned unbound evidence"
            )
        evidence.verify()
        result = TrainCandidateRunResultV2.from_observation(
            work,
            evidence.observation,
            execution_backend_instance_hash=backend_hash,
            prompt_receipt=evidence.prompt_receipt,
        )
        result.verify(work, self.baseline_set)
        return result

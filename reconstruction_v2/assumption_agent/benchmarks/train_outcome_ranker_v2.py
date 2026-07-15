from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field, replace
import json
import math
import re
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..models import SplitName, stable_hash
from .execution_contract_prompt_v2 import (
    EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION,
    ExecutionContractPromptInjectionReceiptV2,
)
from .execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
)
from .skilllearn_lifecycle import (
    SkillLearnTrialObservation,
    TrialVariant,
)
from .typed_task_capability import (
    validate_compiled_portable_task_capability,
)


FROZEN_RAW_TRAIN_BASELINE_VERSION = (
    "frozen_policy_off_train_outcomes_v2"
)
TRAIN_OUTCOME_RANKING_VERSION = (
    "actual_train_policy_on_outcome_ranking_v2"
)
OFFLINE_TRAIN_EVALUATION_MODE = "offline_post_agent_verifier"
SCORE_UNIT_SCALE = 1_000_000
COST_UNIT_SCALE = 1_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TrainOutcomeRankingError(PermissionError):
    """A TRAIN outcome crossed a frozen evidence or execution boundary."""


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TrainOutcomeRankingError(f"{label} is not a sha256 digest")
    return value


def _score_units(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainOutcomeRankingError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise TrainOutcomeRankingError(f"{label} is outside [0, 1]")
    return int(round(number * SCORE_UNIT_SCALE))


def _cost_units(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainOutcomeRankingError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise TrainOutcomeRankingError(f"{label} is not a finite cost")
    return int(round(number * COST_UNIT_SCALE))


def _evaluation_valid(observation: SkillLearnTrialObservation) -> bool:
    value = observation.metrics.get("evaluation_valid", 0.0)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 1.0
    )


def _snapshot_observation(
    observation: SkillLearnTrialObservation,
) -> SkillLearnTrialObservation:
    return replace(
        observation,
        metrics=MappingProxyType(dict(observation.metrics)),
        proposal_action_trace=MappingProxyType(
            dict(observation.proposal_action_trace)
        ),
    )


def _execution_boundary_payload(
    observation: SkillLearnTrialObservation,
) -> dict[str, Any]:
    request = observation.request
    return {
        "agent_id_hash": stable_hash({"agent_id": request.agent_id}),
        "model_hash": stable_hash({"model": request.model}),
        "max_steps": request.max_steps,
        "codex_agent_execution_policy_hash": (
            request.codex_agent_execution_policy_hash
        ),
        "provider_fingerprint_hash": stable_hash(
            {"provider_fingerprint": observation.provider_fingerprint}
        ),
        "fairness_fingerprint_hash": stable_hash(
            {"fairness_fingerprint": observation.fairness_fingerprint}
        ),
        "offline_verifier_profile_id_hash": stable_hash(
            {
                "offline_verifier_profile_id": (
                    observation.offline_verifier_profile_id
                )
            }
        ),
        "offline_verifier_runtime_key_hash": stable_hash(
            {
                "offline_verifier_runtime_key": (
                    observation.offline_verifier_runtime_key
                )
            }
        ),
    }


@dataclass(frozen=True)
class FrozenRawTrainOutcomeV2:
    observation: SkillLearnTrialObservation = field(
        compare=False,
        repr=False,
    )
    item_id_hash: str
    family_hash: str
    score_units: int
    execution_boundary_hash: str
    baseline_evidence_hash: str

    @classmethod
    def from_observation(
        cls,
        observation: SkillLearnTrialObservation,
        *,
        manifest_hash: str,
        evaluator_epoch: str,
    ) -> "FrozenRawTrainOutcomeV2":
        snapshot = _snapshot_observation(observation)
        request = snapshot.request
        if (
            request.split is not SplitName.TRAIN
            or request.variant is not TrialVariant.POLICY_OFF
            or request.manifest_hash != manifest_hash
            or request.evaluator_epoch != evaluator_epoch
            or not snapshot.valid
            or not _evaluation_valid(snapshot)
            or snapshot.raw_trial_artifacts_persisted
        ):
            raise TrainOutcomeRankingError(
                "RAW baseline is not valid frozen TRAIN policy-off evidence"
            )
        _require_sha256(
            snapshot.upstream_result_hash,
            "RAW baseline upstream result hash",
        )
        item_id_hash = stable_hash({"item_id": request.item_id})
        family_hash = stable_hash({"family": request.family})
        score_units = _score_units(snapshot.score, "RAW baseline score")
        execution_boundary_hash = stable_hash(
            _execution_boundary_payload(snapshot)
        )
        evidence_payload = {
            "item_id_hash": item_id_hash,
            "family_hash": family_hash,
            "observation_hash": snapshot.observation_hash,
            "score_units": score_units,
            "success": snapshot.success,
            "execution_boundary_hash": execution_boundary_hash,
            "split": SplitName.TRAIN.value,
            "variant": TrialVariant.POLICY_OFF.value,
        }
        return cls(
            observation=snapshot,
            item_id_hash=item_id_hash,
            family_hash=family_hash,
            score_units=score_units,
            execution_boundary_hash=execution_boundary_hash,
            baseline_evidence_hash=stable_hash(evidence_payload),
        )

    @property
    def item_id(self) -> str:
        return self.observation.request.item_id

    @property
    def family(self) -> str:
        return self.observation.request.family

    @property
    def success(self) -> bool:
        return self.observation.success

    def safe_payload(self) -> dict[str, Any]:
        return {
            "evidence_version": FROZEN_RAW_TRAIN_BASELINE_VERSION,
            "item_id_hash": self.item_id_hash,
            "family_hash": self.family_hash,
            "observation_hash": self.observation.observation_hash,
            "score_units": self.score_units,
            "success": self.success,
            "execution_boundary_hash": self.execution_boundary_hash,
            "baseline_evidence_hash": self.baseline_evidence_hash,
            "split": SplitName.TRAIN.value,
            "variant": TrialVariant.POLICY_OFF.value,
            "raw_item_family_or_observation_persisted": False,
        }

    def verify(self, *, manifest_hash: str, evaluator_epoch: str) -> None:
        reconstructed = type(self).from_observation(
            self.observation,
            manifest_hash=manifest_hash,
            evaluator_epoch=evaluator_epoch,
        )
        if self.safe_payload() != reconstructed.safe_payload():
            raise TrainOutcomeRankingError("RAW baseline evidence drifted")


@dataclass(frozen=True)
class FrozenRawTrainBaselineSetV2:
    manifest_hash: str
    evaluator_epoch: str = field(compare=False, repr=False)
    source_train_receipt_hash: str
    train_item_hashes: tuple[str, ...]
    rows: tuple[FrozenRawTrainOutcomeV2, ...]
    evaluation_mode: str = OFFLINE_TRAIN_EVALUATION_MODE
    network_fallback_allowed: bool = False

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[SkillLearnTrialObservation],
        *,
        manifest_hash: str,
        evaluator_epoch: str,
        source_train_receipt_hash: str,
        expected_item_ids: Sequence[str],
    ) -> "FrozenRawTrainBaselineSetV2":
        _require_sha256(manifest_hash, "TRAIN manifest hash")
        _require_sha256(source_train_receipt_hash, "source TRAIN receipt hash")
        if not isinstance(evaluator_epoch, str) or not evaluator_epoch:
            raise TrainOutcomeRankingError("evaluator epoch is empty")
        rows = tuple(
            sorted(
                (
                    FrozenRawTrainOutcomeV2.from_observation(
                        row,
                        manifest_hash=manifest_hash,
                        evaluator_epoch=evaluator_epoch,
                    )
                    for row in observations
                ),
                key=lambda row: row.item_id_hash,
            )
        )
        expected_hashes = tuple(
            sorted(
                stable_hash({"item_id": value})
                for value in expected_item_ids
            )
        )
        if (
            not expected_hashes
            or len(set(expected_hashes)) != len(expected_hashes)
            or expected_hashes != tuple(row.item_id_hash for row in rows)
        ):
            raise TrainOutcomeRankingError(
                "RAW baseline does not exactly cover frozen TRAIN items"
            )
        result = cls(
            manifest_hash=manifest_hash,
            evaluator_epoch=evaluator_epoch,
            source_train_receipt_hash=source_train_receipt_hash,
            train_item_hashes=expected_hashes,
            rows=rows,
        )
        result.verify()
        return result

    @property
    def baseline_set_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "baseline_policy": FROZEN_RAW_TRAIN_BASELINE_VERSION,
            "manifest_hash": self.manifest_hash,
            "evaluator_epoch_hash": stable_hash(
                {"evaluator_epoch": self.evaluator_epoch}
            ),
            "source_train_receipt_hash": self.source_train_receipt_hash,
            "train_item_hashes": list(self.train_item_hashes),
            "train_item_set_hash": stable_hash(
                {"train_item_hashes": list(self.train_item_hashes)}
            ),
            "evaluation_mode": self.evaluation_mode,
            "network_fallback_allowed": self.network_fallback_allowed,
            "rows": [row.safe_payload() for row in self.rows],
            "row_count": len(self.rows),
            "row_set_hash": stable_hash(
                {
                    "baseline_evidence_hashes": [
                        row.baseline_evidence_hash for row in self.rows
                    ]
                }
            ),
            "new_baseline_executions": 0,
            "online_judge_calls": 0,
            "validation_accessed": False,
            "test_accessed": False,
            "raw_content_persisted": False,
        }

    def verify(self) -> None:
        _require_sha256(self.manifest_hash, "TRAIN manifest hash")
        _require_sha256(
            self.source_train_receipt_hash,
            "source TRAIN receipt hash",
        )
        if (
            self.evaluation_mode != OFFLINE_TRAIN_EVALUATION_MODE
            or self.network_fallback_allowed is not False
            or not self.rows
            or self.train_item_hashes
            != tuple(sorted(set(self.train_item_hashes)))
            or self.train_item_hashes
            != tuple(row.item_id_hash for row in self.rows)
            or tuple(sorted(self.rows, key=lambda row: row.item_id_hash))
            != self.rows
            or len({row.item_id_hash for row in self.rows}) != len(self.rows)
        ):
            raise TrainOutcomeRankingError(
                "frozen RAW TRAIN baseline set is not canonical"
            )
        for row in self.rows:
            row.verify(
                manifest_hash=self.manifest_hash,
                evaluator_epoch=self.evaluator_epoch,
            )


@dataclass(frozen=True)
class TrainProfileContractBindingV2:
    metadata_hash: str
    execution_contract_hash: str

    @property
    def binding_hash(self) -> str:
        return stable_hash(
            {
                "metadata_hash": self.metadata_hash,
                "execution_contract_hash": self.execution_contract_hash,
            }
        )

    def safe_payload(self) -> dict[str, str]:
        return {
            "metadata_hash": self.metadata_hash,
            "execution_contract_hash": self.execution_contract_hash,
            "binding_hash": self.binding_hash,
        }

    def verify(self) -> None:
        _require_sha256(self.metadata_hash, "profile metadata hash")
        _require_sha256(
            self.execution_contract_hash,
            "profile execution-contract hash",
        )


@dataclass(frozen=True)
class TrainCandidateItemRouteV2:
    item_id_hash: str
    item_route_hash: str
    treatment_hash: str
    source_receipt_hash: str
    prompt_contract_hashes: tuple[str, ...]
    prompt_contract_set_hash: str
    profile_contract_bindings: tuple[TrainProfileContractBindingV2, ...]

    @property
    def profile_contract_binding_hashes(self) -> tuple[str, ...]:
        return tuple(row.binding_hash for row in self.profile_contract_bindings)

    @property
    def profile_contract_binding_set_hash(self) -> str:
        return stable_hash(
            {
                "profile_contract_bindings": [
                    row.safe_payload()
                    for row in self.profile_contract_bindings
                ]
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "item_id_hash": self.item_id_hash,
            "item_route_hash": self.item_route_hash,
            "treatment_hash": self.treatment_hash,
            "source_receipt_hash": self.source_receipt_hash,
            "prompt_contract_hashes": list(self.prompt_contract_hashes),
            "prompt_contract_set_hash": self.prompt_contract_set_hash,
            "profile_contract_binding_hashes": list(
                self.profile_contract_binding_hashes
            ),
            "profile_contract_bindings": [
                row.safe_payload()
                for row in self.profile_contract_bindings
            ],
            "profile_contract_binding_set_hash": (
                self.profile_contract_binding_set_hash
            ),
        }

    def verify(self) -> None:
        for value, label in (
            (self.item_id_hash, "candidate route item hash"),
            (self.item_route_hash, "candidate item-route hash"),
            (self.treatment_hash, "candidate treatment hash"),
            (self.source_receipt_hash, "candidate source-receipt hash"),
            (
                self.prompt_contract_set_hash,
                "candidate prompt-contract-set hash",
            ),
            (
                self.profile_contract_binding_set_hash,
                "candidate profile-contract-binding-set hash",
            ),
        ):
            _require_sha256(value, label)
        if (
            not self.prompt_contract_hashes
            or self.prompt_contract_hashes
            != tuple(sorted(set(self.prompt_contract_hashes)))
            or any(
                not _SHA256.fullmatch(value)
                for value in self.prompt_contract_hashes
            )
            or self.prompt_contract_set_hash
            != stable_hash(
                {
                    "execution_contract_hashes": list(
                        self.prompt_contract_hashes
                    )
                }
            )
        ):
            raise TrainOutcomeRankingError(
                "candidate prompt contract route drifted"
            )
        if (
            not self.profile_contract_bindings
            or self.profile_contract_bindings
            != tuple(
                sorted(
                    self.profile_contract_bindings,
                    key=lambda row: row.metadata_hash,
                )
            )
            or len(
                {row.metadata_hash for row in self.profile_contract_bindings}
            )
            != len(self.profile_contract_bindings)
        ):
            raise TrainOutcomeRankingError(
                "candidate profile-contract binding route drifted"
            )
        for binding in self.profile_contract_bindings:
            binding.verify()


@dataclass(frozen=True)
class TrainCandidateSpecV2:
    candidate_id: str = field(compare=False, repr=False)
    candidate_behavior_hash: str
    program_set_hash: str
    base_compile_manifest_hash: str
    typed_binding_set_hash: str
    typed_snapshot_hashes: tuple[str, ...]
    typed_snapshot_ledger_hash: str
    compile_bundle_manifest_hash: str
    execution_contract_set_hash: str
    item_routes: tuple[TrainCandidateItemRouteV2, ...]
    static_complexity: int

    @classmethod
    def from_verified_bundle(
        cls,
        *,
        candidate_id: str,
        bundle: ExecutionContractCompileBundleV2,
        static_complexity: int,
    ) -> "TrainCandidateSpecV2":
        bundle.verify()
        contract_rows_by_program = {
            str(row["program_id_hash"]): row
            for row in bundle.manifest["contract_rows"]
        }
        routes_list: list[TrainCandidateItemRouteV2] = []
        compile_root = bundle.base_compile_result.output_root.resolve(
            strict=True
        )
        for item_hash, route in sorted(
            bundle.manifest["item_routes"].items()
        ):
            contract_hash_by_program = dict(
                zip(
                    route["program_id_hashes"],
                    route["execution_contract_hashes"],
                    strict=True,
                )
            )
            metadata_paths = bundle.base_compile_result.item_portable_capability_metadata_paths.get(
                item_hash,
                (),
            )
            bindings: list[TrainProfileContractBindingV2] = []
            for metadata_path in metadata_paths:
                if metadata_path.is_symlink() or not metadata_path.is_file():
                    raise TrainOutcomeRankingError(
                        "candidate portable metadata is missing or linked"
                    )
                resolved_metadata_path = metadata_path.resolve(strict=True)
                try:
                    resolved_metadata_path.relative_to(compile_root)
                except ValueError as exc:
                    raise TrainOutcomeRankingError(
                        "candidate portable metadata escaped compile root"
                    ) from exc
                try:
                    raw_metadata = json.loads(
                        resolved_metadata_path.read_text(encoding="utf-8")
                    )
                    metadata = validate_compiled_portable_task_capability(
                        raw_metadata
                    )
                    contract_hash = contract_hash_by_program[
                        metadata.program_id_hash
                    ]
                    contract_row = contract_rows_by_program[
                        metadata.program_id_hash
                    ]
                except (
                    KeyError,
                    OSError,
                    PermissionError,
                    UnicodeError,
                    ValueError,
                ) as exc:
                    raise TrainOutcomeRankingError(
                        "candidate portable metadata binding is invalid"
                    ) from exc
                if (
                    metadata.item_id_hash != item_hash
                    or metadata.typed_binding_hash
                    != contract_row["typed_binding_hash"]
                    or metadata.bound_recipe_hash
                    != contract_row["bound_recipe_hash"]
                ):
                    raise TrainOutcomeRankingError(
                        "candidate portable metadata crossed bundle binding"
                    )
                bindings.append(
                    TrainProfileContractBindingV2(
                        metadata_hash=metadata.metadata_hash,
                        execution_contract_hash=str(contract_hash),
                    )
                )
            bindings_tuple = tuple(
                sorted(bindings, key=lambda row: row.metadata_hash)
            )
            if not bindings_tuple:
                raise TrainOutcomeRankingError(
                    "candidate item route has no portable metadata binding"
                )
            prompt_contract_hashes = tuple(
                sorted(set(route["execution_contract_hashes"]))
            )
            routes_list.append(
                TrainCandidateItemRouteV2(
                    item_id_hash=str(item_hash),
                    item_route_hash=str(route["item_route_hash"]),
                    treatment_hash=str(route["base_treatment_hash"]),
                    source_receipt_hash=str(
                        route["base_source_receipt_hash"]
                    ),
                    prompt_contract_hashes=prompt_contract_hashes,
                    prompt_contract_set_hash=stable_hash(
                        {
                            "execution_contract_hashes": list(
                                prompt_contract_hashes
                            )
                        }
                    ),
                    profile_contract_bindings=bindings_tuple,
                )
            )
        routes = tuple(routes_list)
        program_set_hash = str(bundle.manifest["base_program_set_hash"])
        base_compile_manifest_hash = str(
            bundle.manifest["base_compile_manifest_hash"]
        )
        execution_contract_set_hash = str(
            bundle.manifest["execution_contract_set_hash"]
        )
        typed_binding_set_hash = str(
            bundle.manifest["base_typed_binding_set_hash"]
        )
        typed_snapshot_hashes = tuple(
            bundle.manifest["base_typed_snapshot_hashes"]
        )
        typed_snapshot_ledger_hash = str(
            bundle.manifest["base_typed_snapshot_ledger_hash"]
        )
        candidate_behavior_hash = stable_hash(
            {
                "program_set_hash": program_set_hash,
                "base_compile_manifest_hash": base_compile_manifest_hash,
                "typed_binding_set_hash": typed_binding_set_hash,
                "execution_contract_set_hash": execution_contract_set_hash,
                "item_routes": [row.safe_payload() for row in routes],
            }
        )
        result = cls(
            candidate_id=candidate_id,
            candidate_behavior_hash=candidate_behavior_hash,
            program_set_hash=program_set_hash,
            base_compile_manifest_hash=base_compile_manifest_hash,
            typed_binding_set_hash=typed_binding_set_hash,
            typed_snapshot_hashes=typed_snapshot_hashes,
            typed_snapshot_ledger_hash=typed_snapshot_ledger_hash,
            compile_bundle_manifest_hash=bundle.manifest_hash,
            execution_contract_set_hash=execution_contract_set_hash,
            item_routes=routes,
            static_complexity=static_complexity,
        )
        result.verify()
        return result

    @property
    def candidate_id_hash(self) -> str:
        return stable_hash({"candidate_id": self.candidate_id})

    @property
    def candidate_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "candidate_id_hash": self.candidate_id_hash,
            "candidate_behavior_hash": self.candidate_behavior_hash,
            "program_set_hash": self.program_set_hash,
            "base_compile_manifest_hash": self.base_compile_manifest_hash,
            "typed_binding_set_hash": self.typed_binding_set_hash,
            "typed_snapshot_hashes": list(self.typed_snapshot_hashes),
            "typed_snapshot_ledger_hash": (
                self.typed_snapshot_ledger_hash
            ),
            "compile_bundle_manifest_hash": self.compile_bundle_manifest_hash,
            "execution_contract_set_hash": self.execution_contract_set_hash,
            "item_routes": [row.safe_payload() for row in self.item_routes],
            "item_route_set_hash": stable_hash(
                {"item_routes": [row.safe_payload() for row in self.item_routes]}
            ),
            "static_complexity": self.static_complexity,
            "raw_candidate_content_persisted": False,
        }

    def verify(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise TrainOutcomeRankingError("candidate identity is empty")
        for value, label in (
            (self.candidate_behavior_hash, "candidate behavior hash"),
            (self.program_set_hash, "candidate program-set hash"),
            (
                self.base_compile_manifest_hash,
                "candidate base-compile hash",
            ),
            (
                self.typed_binding_set_hash,
                "candidate typed-binding-set hash",
            ),
            (
                self.typed_snapshot_ledger_hash,
                "candidate typed-snapshot-ledger hash",
            ),
            (
                self.compile_bundle_manifest_hash,
                "candidate compile-bundle hash",
            ),
            (
                self.execution_contract_set_hash,
                "candidate execution-contract-set hash",
            ),
        ):
            _require_sha256(value, label)
        if (
            not self.typed_snapshot_hashes
            or self.typed_snapshot_hashes
            != tuple(sorted(set(self.typed_snapshot_hashes)))
            or any(
                not _SHA256.fullmatch(value)
                for value in self.typed_snapshot_hashes
            )
        ):
            raise TrainOutcomeRankingError(
                "candidate typed snapshot set is not canonical"
            )
        if (
            not self.item_routes
            or self.item_routes
            != tuple(sorted(self.item_routes, key=lambda row: row.item_id_hash))
            or len({row.item_id_hash for row in self.item_routes})
            != len(self.item_routes)
        ):
            raise TrainOutcomeRankingError(
                "candidate item routes are not canonical"
            )
        for route in self.item_routes:
            route.verify()
        if self.candidate_behavior_hash != stable_hash(
            {
                "program_set_hash": self.program_set_hash,
                "base_compile_manifest_hash": self.base_compile_manifest_hash,
                "typed_binding_set_hash": self.typed_binding_set_hash,
                "execution_contract_set_hash": (
                    self.execution_contract_set_hash
                ),
                "item_routes": [
                    row.safe_payload() for row in self.item_routes
                ],
            }
        ):
            raise TrainOutcomeRankingError(
                "candidate behavior hash is not bundle-derived"
            )
        if (
            isinstance(self.static_complexity, bool)
            or not isinstance(self.static_complexity, int)
            or self.static_complexity < 0
        ):
            raise TrainOutcomeRankingError(
                "candidate static complexity must be a nonnegative integer"
            )

    def route_for_item_hash(
        self,
        item_id_hash: str,
    ) -> TrainCandidateItemRouteV2:
        for route in self.item_routes:
            if route.item_id_hash == item_id_hash:
                return route
        raise TrainOutcomeRankingError(
            "candidate bundle has no frozen route for TRAIN item"
        )


@dataclass(frozen=True)
class TrainCandidateWorkUnitV2:
    candidate: TrainCandidateSpecV2 = field(compare=False, repr=False)
    baseline: FrozenRawTrainOutcomeV2 = field(compare=False, repr=False)

    @property
    def work_unit_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "ranking_policy": TRAIN_OUTCOME_RANKING_VERSION,
            "candidate_hash": self.candidate.candidate_hash,
            "item_id_hash": self.baseline.item_id_hash,
            "family_hash": self.baseline.family_hash,
            "baseline_evidence_hash": self.baseline.baseline_evidence_hash,
            "split": SplitName.TRAIN.value,
            "variant": TrialVariant.POLICY_ON.value,
            "raw_item_family_or_candidate_persisted": False,
        }


@dataclass(frozen=True)
class OfflineTrainEvaluationReceiptV2:
    work_unit_hash: str
    request_hash: str
    observation_hash: str
    upstream_result_hash: str
    offline_verifier_profile_id_hash: str
    offline_verifier_runtime_key_hash: str
    evaluation_valid: bool
    evaluation_mode: str = OFFLINE_TRAIN_EVALUATION_MODE
    network_fallback_used: bool = False
    online_judge_calls: int = 0
    validation_accessed: bool = False
    test_accessed: bool = False

    @classmethod
    def from_observation(
        cls,
        work: TrainCandidateWorkUnitV2,
        observation: SkillLearnTrialObservation,
    ) -> "OfflineTrainEvaluationReceiptV2":
        _require_sha256(
            observation.upstream_result_hash,
            "offline upstream result hash",
        )
        return cls(
            work_unit_hash=work.work_unit_hash,
            request_hash=observation.request.request_hash,
            observation_hash=observation.observation_hash,
            upstream_result_hash=observation.upstream_result_hash,
            offline_verifier_profile_id_hash=stable_hash(
                {
                    "offline_verifier_profile_id": (
                        observation.offline_verifier_profile_id
                    )
                }
            ),
            offline_verifier_runtime_key_hash=stable_hash(
                {
                    "offline_verifier_runtime_key": (
                        observation.offline_verifier_runtime_key
                    )
                }
            ),
            evaluation_valid=(observation.valid and _evaluation_valid(observation)),
        )

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "evaluation_mode": self.evaluation_mode,
            "work_unit_hash": self.work_unit_hash,
            "request_hash": self.request_hash,
            "observation_hash": self.observation_hash,
            "upstream_result_hash": self.upstream_result_hash,
            "offline_verifier_profile_id_hash": (
                self.offline_verifier_profile_id_hash
            ),
            "offline_verifier_runtime_key_hash": (
                self.offline_verifier_runtime_key_hash
            ),
            "evaluation_valid": self.evaluation_valid,
            "network_fallback_used": self.network_fallback_used,
            "online_judge_calls": self.online_judge_calls,
            "validation_accessed": self.validation_accessed,
            "test_accessed": self.test_accessed,
            "raw_evaluator_content_persisted": False,
        }

    def verify(
        self,
        work: TrainCandidateWorkUnitV2,
        observation: SkillLearnTrialObservation,
    ) -> None:
        for value, label in (
            (self.work_unit_hash, "offline work-unit hash"),
            (self.request_hash, "offline request hash"),
            (self.observation_hash, "offline observation hash"),
            (self.upstream_result_hash, "offline upstream result hash"),
            (
                self.offline_verifier_profile_id_hash,
                "offline verifier profile hash",
            ),
            (
                self.offline_verifier_runtime_key_hash,
                "offline verifier runtime hash",
            ),
        ):
            _require_sha256(value, label)
        if (
            self.evaluation_mode != OFFLINE_TRAIN_EVALUATION_MODE
            or self.network_fallback_used is not False
            or self.online_judge_calls != 0
            or self.validation_accessed is not False
            or self.test_accessed is not False
            or self.work_unit_hash != work.work_unit_hash
            or self.request_hash != observation.request.request_hash
            or self.observation_hash != observation.observation_hash
            or self.upstream_result_hash != observation.upstream_result_hash
            or self.offline_verifier_profile_id_hash
            != stable_hash(
                {
                    "offline_verifier_profile_id": (
                        observation.offline_verifier_profile_id
                    )
                }
            )
            or self.offline_verifier_runtime_key_hash
            != stable_hash(
                {
                    "offline_verifier_runtime_key": (
                        observation.offline_verifier_runtime_key
                    )
                }
            )
            or self.evaluation_valid
            != (observation.valid and _evaluation_valid(observation))
        ):
            raise TrainOutcomeRankingError(
                "offline TRAIN evaluation receipt drifted"
            )


@dataclass(frozen=True)
class TrainCandidateRunResultV2:
    work_unit_hash: str
    candidate_hash: str
    execution_backend_instance_hash: str
    compile_bundle_manifest_hash: str
    execution_contract_set_hash: str
    offline_evaluation: OfflineTrainEvaluationReceiptV2
    prompt_receipt: ExecutionContractPromptInjectionReceiptV2 | None = field(
        compare=False,
        repr=False,
    )
    observation: SkillLearnTrialObservation = field(
        compare=False,
        repr=False,
    )

    @classmethod
    def from_observation(
        cls,
        work: TrainCandidateWorkUnitV2,
        observation: SkillLearnTrialObservation,
        *,
        execution_backend_instance_hash: str,
        prompt_receipt: ExecutionContractPromptInjectionReceiptV2 | None,
    ) -> "TrainCandidateRunResultV2":
        snapshot = _snapshot_observation(observation)
        return cls(
            work_unit_hash=work.work_unit_hash,
            candidate_hash=work.candidate.candidate_hash,
            execution_backend_instance_hash=(
                execution_backend_instance_hash
            ),
            compile_bundle_manifest_hash=(
                work.candidate.compile_bundle_manifest_hash
            ),
            execution_contract_set_hash=(
                work.candidate.execution_contract_set_hash
            ),
            offline_evaluation=(
                OfflineTrainEvaluationReceiptV2.from_observation(
                    work,
                    snapshot,
                )
            ),
            prompt_receipt=prompt_receipt,
            observation=snapshot,
        )

    @property
    def run_receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "ranking_policy": TRAIN_OUTCOME_RANKING_VERSION,
            "work_unit_hash": self.work_unit_hash,
            "candidate_hash": self.candidate_hash,
            "execution_backend_instance_hash": (
                self.execution_backend_instance_hash
            ),
            "compile_bundle_manifest_hash": (
                self.compile_bundle_manifest_hash
            ),
            "execution_contract_set_hash": self.execution_contract_set_hash,
            "request_hash": self.observation.request.request_hash,
            "observation_hash": self.observation.observation_hash,
            "prompt_delivery_policy": (
                self.observation.runtime_profile_prompt_delivery_policy
                or None
            ),
            "prompt_injection_receipt_hash": (
                self.prompt_receipt.receipt_hash
                if self.prompt_receipt is not None
                else None
            ),
            "prompt_receipt_bundle_manifest_hash": (
                self.prompt_receipt.bundle_manifest_hash
                if self.prompt_receipt is not None
                else None
            ),
            "prompt_receipt_contract_set_hash": (
                self.prompt_receipt.contract_set_hash
                if self.prompt_receipt is not None
                else None
            ),
            "effective_prompt_sha256": (
                self.observation.runtime_profile_effective_prompt_sha256
                or None
            ),
            "offline_evaluation_receipt_hash": (
                self.offline_evaluation.receipt_hash
            ),
            "evaluation_valid": (
                self.observation.valid
                and _evaluation_valid(self.observation)
            ),
            "raw_observation_or_evaluator_content_persisted": False,
        }

    def verify(
        self,
        work: TrainCandidateWorkUnitV2,
        baseline_set: FrozenRawTrainBaselineSetV2,
    ) -> None:
        observation = self.observation
        request = observation.request
        baseline = work.baseline.observation
        route = work.candidate.route_for_item_hash(work.baseline.item_id_hash)
        _require_sha256(
            self.execution_backend_instance_hash,
            "execution backend instance hash",
        )
        if (
            self.work_unit_hash != work.work_unit_hash
            or self.candidate_hash != work.candidate.candidate_hash
            or self.compile_bundle_manifest_hash
            != work.candidate.compile_bundle_manifest_hash
            or self.execution_contract_set_hash
            != work.candidate.execution_contract_set_hash
        ):
            raise TrainOutcomeRankingError("candidate run binding drifted")
        if (
            request.split is not SplitName.TRAIN
            or request.variant is not TrialVariant.POLICY_ON
            or request.item_id != work.baseline.item_id
            or request.family != work.baseline.family
            or request.manifest_hash != baseline_set.manifest_hash
            or request.evaluator_epoch != baseline_set.evaluator_epoch
            or request.program_set_hash != work.candidate.program_set_hash
            or request.compile_manifest_hash
            != work.candidate.base_compile_manifest_hash
            or request.typed_binding_set_hash
            != work.candidate.typed_binding_set_hash
            or request.typed_snapshot_hashes
            != work.candidate.typed_snapshot_hashes
            or request.typed_snapshot_ledger_hash
            != work.candidate.typed_snapshot_ledger_hash
            or request.treatment_hash != route.treatment_hash
            or request.skill_source_receipt_hash != route.source_receipt_hash
            or request.agent_id != baseline.request.agent_id
            or request.model != baseline.request.model
            or request.max_steps != baseline.request.max_steps
            or request.codex_agent_execution_policy_hash
            != baseline.request.codex_agent_execution_policy_hash
            or observation.provider_fingerprint
            != baseline.provider_fingerprint
            or observation.fairness_fingerprint
            != baseline.fairness_fingerprint
            or observation.offline_verifier_profile_id
            != baseline.offline_verifier_profile_id
            or observation.offline_verifier_runtime_key
            != baseline.offline_verifier_runtime_key
            or observation.raw_trial_artifacts_persisted
        ):
            raise TrainOutcomeRankingError(
                "candidate run crossed a frozen TRAIN execution boundary"
            )
        evaluation_valid = _evaluation_valid(observation)
        if observation.valid != evaluation_valid:
            raise TrainOutcomeRankingError(
                "candidate observation validity and offline metric disagree"
            )
        prompt_declared = bool(
            observation.runtime_profile_prompt_delivery_policy
            or observation.runtime_profile_prompt_injection_receipt_hash
            or observation.runtime_profile_effective_prompt_sha256
            or self.prompt_receipt is not None
        )
        if observation.valid and not prompt_declared:
            raise TrainOutcomeRankingError(
                "valid candidate lacks v2 execution-contract delivery"
            )
        if prompt_declared:
            receipt = self.prompt_receipt
            if (
                not isinstance(
                    receipt,
                    ExecutionContractPromptInjectionReceiptV2,
                )
                or observation.runtime_profile_prompt_delivery_policy
                != EXECUTION_CONTRACT_PROMPT_DELIVERY_VERSION
                or receipt.request_hash != request.request_hash
                or receipt.bundle_manifest_hash
                != work.candidate.compile_bundle_manifest_hash
                or receipt.source_receipt_hash
                != request.skill_source_receipt_hash
                or receipt.typed_binding_set_hash
                != work.candidate.typed_binding_set_hash
                or receipt.contract_set_hash
                != route.prompt_contract_set_hash
                or receipt.contract_set_hash
                != stable_hash(
                    {
                        "execution_contract_hashes": list(
                            receipt.contract_hashes
                        )
                    }
                )
                or receipt.receipt_hash
                != observation.runtime_profile_prompt_injection_receipt_hash
                or receipt.effective_prompt_sha256
                != observation.runtime_profile_effective_prompt_sha256
                or receipt.profile_count <= 0
                or len(receipt.effect_receipt_hashes)
                != receipt.profile_count
                or len(receipt.profile_output_sha256s)
                != receipt.profile_count
                or len(receipt.profile_contract_binding_hashes)
                != receipt.profile_count
                or receipt.profile_contract_binding_set_hash
                != route.profile_contract_binding_set_hash
                or receipt.profile_contract_binding_hashes
                != route.profile_contract_binding_hashes
            ):
                raise TrainOutcomeRankingError(
                    "execution-contract prompt receipt crossed its bundle route"
                )
            _require_sha256(
                observation.runtime_profile_prompt_injection_receipt_hash,
                "execution-contract prompt receipt hash",
            )
            _require_sha256(
                observation.runtime_profile_effective_prompt_sha256,
                "effective prompt hash",
            )
        self.offline_evaluation.verify(work, observation)


class TrainCandidateRunnerV2(Protocol):
    def __call__(
        self,
        work: TrainCandidateWorkUnitV2,
    ) -> TrainCandidateRunResultV2: ...


@dataclass(frozen=True)
class TrainOutcomeRowV2:
    work_unit_hash: str
    candidate_hash: str
    item_id_hash: str
    baseline_evidence_hash: str
    run_receipt_hash: str
    observation_hash: str
    valid: bool
    baseline_success: bool
    candidate_success: bool
    regression: bool
    recovery: bool
    baseline_score_units: int
    candidate_score_units: int
    score_delta_units: int
    candidate_cost_units: int

    @classmethod
    def from_result(
        cls,
        work: TrainCandidateWorkUnitV2,
        result: TrainCandidateRunResultV2,
    ) -> "TrainOutcomeRowV2":
        observation = result.observation
        valid = observation.valid and _evaluation_valid(observation)
        candidate_success = valid and observation.success
        baseline_success = work.baseline.success
        baseline_score_units = work.baseline.score_units
        candidate_score_units = (
            _score_units(observation.score, "candidate score") if valid else 0
        )
        return cls(
            work_unit_hash=work.work_unit_hash,
            candidate_hash=work.candidate.candidate_hash,
            item_id_hash=work.baseline.item_id_hash,
            baseline_evidence_hash=work.baseline.baseline_evidence_hash,
            run_receipt_hash=result.run_receipt_hash,
            observation_hash=observation.observation_hash,
            valid=valid,
            baseline_success=baseline_success,
            candidate_success=candidate_success,
            regression=(baseline_success and not candidate_success),
            recovery=(valid and not baseline_success and candidate_success),
            baseline_score_units=baseline_score_units,
            candidate_score_units=candidate_score_units,
            score_delta_units=(
                candidate_score_units - baseline_score_units
            ),
            candidate_cost_units=_cost_units(
                observation.cost_units,
                "candidate cost",
            ),
        )

    @property
    def outcome_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "ranking_policy": TRAIN_OUTCOME_RANKING_VERSION,
            "work_unit_hash": self.work_unit_hash,
            "candidate_hash": self.candidate_hash,
            "item_id_hash": self.item_id_hash,
            "baseline_evidence_hash": self.baseline_evidence_hash,
            "run_receipt_hash": self.run_receipt_hash,
            "observation_hash": self.observation_hash,
            "valid": self.valid,
            "baseline_success": self.baseline_success,
            "candidate_success": self.candidate_success,
            "regression": self.regression,
            "recovery": self.recovery,
            "baseline_score_units": self.baseline_score_units,
            "candidate_score_units": self.candidate_score_units,
            "score_delta_units": self.score_delta_units,
            "candidate_cost_units": self.candidate_cost_units,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class TrainCandidateAggregateV2:
    candidate_hash: str
    static_complexity: int
    outcome_hashes: tuple[str, ...]
    invalid_count: int
    regression_count: int
    recovery_count: int
    candidate_success_count: int
    score_delta_units: int
    total_cost_units: int

    @property
    def ranking_key(self) -> tuple[int, int, int, int, int, int, int, str]:
        return (
            self.invalid_count,
            self.regression_count,
            -self.recovery_count,
            -self.candidate_success_count,
            -self.score_delta_units,
            self.total_cost_units,
            self.static_complexity,
            self.candidate_hash,
        )

    @property
    def aggregate_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "candidate_hash": self.candidate_hash,
            "outcome_hashes": list(self.outcome_hashes),
            "invalid_count": self.invalid_count,
            "regression_count": self.regression_count,
            "recovery_count": self.recovery_count,
            "candidate_success_count": self.candidate_success_count,
            "score_delta_units": self.score_delta_units,
            "total_cost_units": self.total_cost_units,
            "static_complexity": self.static_complexity,
            "ranking_key": list(self.ranking_key),
            "actual_outcomes_precede_static_tiebreaks": True,
        }


def _aggregate_candidate(
    candidate: TrainCandidateSpecV2,
    rows: Sequence[TrainOutcomeRowV2],
) -> TrainCandidateAggregateV2:
    ordered = tuple(sorted(rows, key=lambda row: row.item_id_hash))
    if not ordered or any(
        row.candidate_hash != candidate.candidate_hash for row in ordered
    ):
        raise TrainOutcomeRankingError("candidate outcome aggregation drifted")
    return TrainCandidateAggregateV2(
        candidate_hash=candidate.candidate_hash,
        static_complexity=candidate.static_complexity,
        outcome_hashes=tuple(row.outcome_hash for row in ordered),
        invalid_count=sum(not row.valid for row in ordered),
        regression_count=sum(row.regression for row in ordered),
        recovery_count=sum(row.recovery for row in ordered),
        candidate_success_count=sum(row.candidate_success for row in ordered),
        score_delta_units=sum(row.score_delta_units for row in ordered),
        total_cost_units=sum(row.candidate_cost_units for row in ordered),
    )


@dataclass(frozen=True)
class TrainOutcomeRankingResultV2:
    baseline_set: FrozenRawTrainBaselineSetV2 = field(
        compare=False,
        repr=False,
    )
    candidates: tuple[TrainCandidateSpecV2, ...] = field(
        compare=False,
        repr=False,
    )
    work_units: tuple[TrainCandidateWorkUnitV2, ...] = field(
        compare=False,
        repr=False,
    )
    run_results: tuple[TrainCandidateRunResultV2, ...] = field(
        compare=False,
        repr=False,
    )
    outcomes: tuple[TrainOutcomeRowV2, ...]
    aggregates: tuple[TrainCandidateAggregateV2, ...]
    ordered_candidate_hashes: tuple[str, ...]
    candidate_set_hash: str
    work_unit_set_hash: str
    outcome_set_hash: str
    effective_worker_count: int
    maximum_concurrent_runner_calls: int = field(compare=False)

    @property
    def top_candidate_hash(self) -> str:
        return self.ordered_candidate_hashes[0]

    @property
    def ranking_hash(self) -> str:
        return stable_hash(self.safe_payload())

    @property
    def concurrency_receipt_hash(self) -> str:
        return stable_hash(
            {
                "ranking_hash": self.ranking_hash,
                "work_unit_count": len(self.work_units),
                "effective_worker_count": self.effective_worker_count,
                "maximum_concurrent_runner_calls": (
                    self.maximum_concurrent_runner_calls
                ),
                "runner_call_concurrency_only": True,
                "distinct_backend_instance_count": len(
                    {
                        row.execution_backend_instance_hash
                        for row in self.run_results
                    }
                ),
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "ranking_policy": TRAIN_OUTCOME_RANKING_VERSION,
            "baseline_set_hash": self.baseline_set.baseline_set_hash,
            "candidate_set_hash": self.candidate_set_hash,
            "work_unit_set_hash": self.work_unit_set_hash,
            "outcome_set_hash": self.outcome_set_hash,
            "aggregates": [row.safe_payload() for row in self.aggregates],
            "aggregate_hashes": [
                row.aggregate_hash for row in self.aggregates
            ],
            "ordered_candidate_hashes": list(self.ordered_candidate_hashes),
            "top_candidate_hash": self.top_candidate_hash,
            "candidate_count": len(self.candidates),
            "train_item_count": len(self.baseline_set.rows),
            "candidate_execution_count": len(self.work_units),
            "baseline_execution_count": 0,
            "run_receipt_hashes": [
                row.run_receipt_hash for row in self.run_results
            ],
            "run_receipt_set_hash": stable_hash(
                {
                    "run_receipt_hashes": [
                        row.run_receipt_hash for row in self.run_results
                    ]
                }
            ),
            "execution_backend_instance_set_hash": stable_hash(
                {
                    "execution_backend_instance_hashes": sorted(
                        row.execution_backend_instance_hash
                        for row in self.run_results
                    )
                }
            ),
            "evaluation_mode": OFFLINE_TRAIN_EVALUATION_MODE,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "validation_accessed": False,
            "test_accessed": False,
            "promotion_gate_applied": False,
            "promotion_authorized": False,
            "ranking_only_not_a_promotion_decision": True,
            "raw_content_persisted": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.safe_payload(),
            "ranking_hash": self.ranking_hash,
            "effective_worker_count": self.effective_worker_count,
            "maximum_concurrent_runner_calls": (
                self.maximum_concurrent_runner_calls
            ),
            "runner_call_concurrency_only": True,
            "concurrency_receipt_hash": self.concurrency_receipt_hash,
        }

    def verify(self) -> None:
        self.baseline_set.verify()
        if not self.candidates:
            raise TrainOutcomeRankingError("candidate set is invalid")
        for candidate in self.candidates:
            candidate.verify()
            if tuple(
                route.item_id_hash for route in candidate.item_routes
            ) != self.baseline_set.train_item_hashes:
                raise TrainOutcomeRankingError(
                    "candidate bundle does not exactly cover frozen TRAIN"
                )
        expected_candidates = tuple(
            sorted(self.candidates, key=lambda row: row.candidate_hash)
        )
        if (
            expected_candidates != self.candidates
            or len({row.candidate_hash for row in self.candidates})
            != len(self.candidates)
            or len({row.candidate_behavior_hash for row in self.candidates})
            != len(self.candidates)
            or self.candidate_set_hash
            != stable_hash(
                {
                    "candidate_hashes": [
                        row.candidate_hash for row in self.candidates
                    ]
                }
            )
        ):
            raise TrainOutcomeRankingError("candidate set receipt drifted")
        expected_work_units = tuple(
            TrainCandidateWorkUnitV2(candidate, baseline)
            for candidate in self.candidates
            for baseline in self.baseline_set.rows
        )
        if (
            tuple(row.work_unit_hash for row in expected_work_units)
            != tuple(row.work_unit_hash for row in self.work_units)
            or self.work_unit_set_hash
            != stable_hash(
                {
                    "work_unit_hashes": [
                        row.work_unit_hash for row in self.work_units
                    ]
                }
            )
        ):
            raise TrainOutcomeRankingError("candidate-by-TRAIN grid drifted")
        if (
            len(self.run_results) != len(expected_work_units)
            or tuple(row.work_unit_hash for row in self.run_results)
            != tuple(row.work_unit_hash for row in expected_work_units)
            or len(
                {
                    row.execution_backend_instance_hash
                    for row in self.run_results
                }
            )
            != len(self.run_results)
        ):
            raise TrainOutcomeRankingError(
                "TRAIN run receipt grid or backend instances drifted"
            )
        for work, run_result in zip(
            expected_work_units,
            self.run_results,
            strict=True,
        ):
            run_result.verify(work, self.baseline_set)
        expected_outcome_bindings = {
            work.work_unit_hash: (
                work.candidate.candidate_hash,
                work.baseline.item_id_hash,
                work.baseline.baseline_evidence_hash,
            )
            for work in expected_work_units
        }
        expected_outcomes = tuple(
            TrainOutcomeRowV2.from_result(work, run_result)
            for work, run_result in zip(
                expected_work_units,
                self.run_results,
                strict=True,
            )
        )
        if (
            len(self.outcomes) != len(self.work_units)
            or self.outcomes != expected_outcomes
            or tuple(
                (row.candidate_hash, row.item_id_hash)
                for row in self.outcomes
            )
            != tuple(
                sorted(
                    (
                        (row.candidate_hash, row.item_id_hash)
                        for row in self.outcomes
                    )
                )
            )
            or len({row.work_unit_hash for row in self.outcomes})
            != len(self.outcomes)
            or set(row.work_unit_hash for row in self.outcomes)
            != set(expected_outcome_bindings)
            or any(
                expected_outcome_bindings[row.work_unit_hash]
                != (
                    row.candidate_hash,
                    row.item_id_hash,
                    row.baseline_evidence_hash,
                )
                for row in self.outcomes
            )
            or any(
                outcome.run_receipt_hash != run_result.run_receipt_hash
                for outcome, run_result in zip(
                    self.outcomes,
                    self.run_results,
                    strict=True,
                )
            )
            or self.outcome_set_hash
            != stable_hash(
                {"outcome_hashes": [row.outcome_hash for row in self.outcomes]}
            )
        ):
            raise TrainOutcomeRankingError("TRAIN outcome set drifted")
        expected_aggregates = tuple(
            _aggregate_candidate(
                candidate,
                tuple(
                    row
                    for row in self.outcomes
                    if row.candidate_hash == candidate.candidate_hash
                ),
            )
            for candidate in self.candidates
        )
        expected_order = tuple(
            row.candidate_hash
            for row in sorted(
                expected_aggregates,
                key=lambda row: row.ranking_key,
            )
        )
        if (
            expected_aggregates != self.aggregates
            or expected_order != self.ordered_candidate_hashes
            or self.effective_worker_count <= 0
            or self.effective_worker_count > len(self.work_units)
            or self.maximum_concurrent_runner_calls <= 0
            or self.maximum_concurrent_runner_calls
            > self.effective_worker_count
        ):
            raise TrainOutcomeRankingError("TRAIN ranking receipt drifted")


class TrainOutcomeRankerV2:
    """Rank candidates by actual TRAIN utility without creating a new gate.

    The runner must be safe for concurrent calls.  For the v2 subprocess
    backend, callers should supply independent backend instances per worker;
    one backend instance deliberately serializes its mutable frozen-v1 runner.
    """

    def __init__(self, *, max_workers: int | None = None) -> None:
        if max_workers is not None and (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("TRAIN ranker max_workers must be positive")
        self.max_workers = max_workers

    def rank(
        self,
        *,
        baseline_set: FrozenRawTrainBaselineSetV2,
        candidates: Sequence[TrainCandidateSpecV2],
        runner: TrainCandidateRunnerV2 | Callable[
            [TrainCandidateWorkUnitV2], TrainCandidateRunResultV2
        ],
    ) -> TrainOutcomeRankingResultV2:
        baseline_set.verify()
        ordered_candidates = tuple(
            sorted(candidates, key=lambda row: row.candidate_hash)
        )
        if not ordered_candidates:
            raise TrainOutcomeRankingError("TRAIN ranker has no candidates")
        for candidate in ordered_candidates:
            candidate.verify()
            if tuple(
                route.item_id_hash for route in candidate.item_routes
            ) != baseline_set.train_item_hashes:
                raise TrainOutcomeRankingError(
                    "candidate bundle does not exactly cover frozen TRAIN"
                )
        if len({row.candidate_hash for row in ordered_candidates}) != len(
            ordered_candidates
        ) or len(
            {row.candidate_behavior_hash for row in ordered_candidates}
        ) != len(ordered_candidates):
            raise TrainOutcomeRankingError("TRAIN candidates are not unique")
        work_units = tuple(
            TrainCandidateWorkUnitV2(candidate, baseline)
            for candidate in ordered_candidates
            for baseline in baseline_set.rows
        )
        effective_workers = min(
            self.max_workers or len(work_units),
            len(work_units),
        )
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()

        def invoke(
            work: TrainCandidateWorkUnitV2,
        ) -> TrainCandidateRunResultV2:
            nonlocal active, maximum_active
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                result = runner(work)
                if not isinstance(result, TrainCandidateRunResultV2):
                    raise TrainOutcomeRankingError(
                        "TRAIN candidate runner returned an unknown result"
                    )
                result.verify(work, baseline_set)
                return result
            except TrainOutcomeRankingError:
                raise
            except Exception as exc:
                raise TrainOutcomeRankingError(
                    "TRAIN candidate work unit failed without a valid receipt"
                ) from exc
            finally:
                with active_lock:
                    active -= 1

        results_by_work_hash: dict[str, TrainCandidateRunResultV2] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="train-outcome-v2",
        ) as executor:
            futures = {
                executor.submit(invoke, work): work for work in work_units
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    work = futures[future]
                    result = future.result()
                    if result.work_unit_hash in results_by_work_hash:
                        raise TrainOutcomeRankingError(
                            "TRAIN runner duplicated a work-unit receipt"
                        )
                    results_by_work_hash[work.work_unit_hash] = result
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        if set(results_by_work_hash) != {
            row.work_unit_hash for row in work_units
        }:
            raise TrainOutcomeRankingError(
                "TRAIN candidate-by-item result grid is incomplete"
            )
        run_results = tuple(
            results_by_work_hash[work.work_unit_hash]
            for work in work_units
        )
        outcomes = tuple(
            sorted(
                (
                    TrainOutcomeRowV2.from_result(
                        work,
                        results_by_work_hash[work.work_unit_hash],
                    )
                    for work in work_units
                ),
                key=lambda row: (row.candidate_hash, row.item_id_hash),
            )
        )
        aggregates = tuple(
            _aggregate_candidate(
                candidate,
                tuple(
                    row
                    for row in outcomes
                    if row.candidate_hash == candidate.candidate_hash
                ),
            )
            for candidate in ordered_candidates
        )
        ordered_candidate_hashes = tuple(
            row.candidate_hash
            for row in sorted(aggregates, key=lambda row: row.ranking_key)
        )
        result = TrainOutcomeRankingResultV2(
            baseline_set=baseline_set,
            candidates=ordered_candidates,
            work_units=work_units,
            run_results=run_results,
            outcomes=outcomes,
            aggregates=aggregates,
            ordered_candidate_hashes=ordered_candidate_hashes,
            candidate_set_hash=stable_hash(
                {
                    "candidate_hashes": [
                        row.candidate_hash for row in ordered_candidates
                    ]
                }
            ),
            work_unit_set_hash=stable_hash(
                {
                    "work_unit_hashes": [
                        row.work_unit_hash for row in work_units
                    ]
                }
            ),
            outcome_set_hash=stable_hash(
                {"outcome_hashes": [row.outcome_hash for row in outcomes]}
            ),
            effective_worker_count=effective_workers,
            maximum_concurrent_runner_calls=maximum_active,
        )
        result.verify()
        return result

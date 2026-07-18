"""One-shot lifecycle controller for the frozen HoVer joint-graph study.

The controller owns ordering and durable terminal receipts, while acquisition,
gold-free execution, and local runtimes remain replaceable boundaries.  The
formal entrypoint has no injectable dependencies.  Synthetic tests use the
separate sentinel-gated entrypoint and never open the formal source pack.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import hover_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks import hover_implementation_freeze_v1 as implementation_freeze
from assumption_agent.benchmarks import hover_isolated_bootstrap_v1 as isolated_bootstrap
from assumption_agent.benchmarks import hover_joint_graph_formal_runner_v1 as runner
from assumption_agent.benchmarks import hover_lifecycle_store_v1 as lifecycle_store
from assumption_agent.benchmarks import hover_local_runtime_v1 as local_runtime
from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import PolicySelection
from replication_runtime.qasper_minilm_v1.binding import OfflineMiniLMEncoder


VERSION = "hover_joint_graph_formal_controller_v1"
RESULT_SCHEMA = f"{VERSION}_terminal_result"
MARKER_SCHEMA = f"{VERSION}_one_shot_marker"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure"
DESCRIPTIVE_SCHEMA = f"{VERSION}_A_form_descriptive"

FORMAL_ROOT_RELATIVE = "artifacts/hover_joint_graph_formal_v1"
FORMAL_OUTPUT_PATHS = None  # initialized after LifecycleOutputPaths is defined
SYNTHETIC_ROOT_RELATIVE = "artifacts/hover_joint_graph_synthetic_test_only"
SYNTHETIC_SENTINEL = ".hover_joint_graph_synthetic_lifecycle_test_root"
SYNTHETIC_SENTINEL_CONTENT = "offline_synthetic_no_formal_capabilities_v1\n"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")

FormalRuntimeConfig = local_runtime.FormalRuntimeConfig
OfficialHippoGateway = local_runtime.OfficialHippoGateway
OfflineNERJSONLClient = local_runtime.OfflineNERJSONLClient


class HoVerFormalControllerError(RuntimeError):
    """A lifecycle ordering, binding, or durable-output invariant drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HoVerFormalControllerError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HoVerFormalControllerError(f"{field} is not a SHA256")
    return value


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HoVerFormalControllerError("self-hash field already exists")
    return {**body, field: stable_hash(body)}


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = _require_sha256(payload.get(field), field)
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise HoVerFormalControllerError(f"{field} self-hash mismatch")
    return declared


def _fraction(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise HoVerFormalControllerError("exact statistic is not a Fraction")
    return [value.numerator, value.denominator]


@dataclass(frozen=True)
class LifecycleOutputPaths:
    marker: str
    failure: str
    result: str
    a_form_descriptive: str


FORMAL_OUTPUT_PATHS = LifecycleOutputPaths(
    marker=f"{FORMAL_ROOT_RELATIVE}/runner.one_shot_marker.json",
    failure=f"{FORMAL_ROOT_RELATIVE}/runner.terminal_failure.json",
    result=f"{FORMAL_ROOT_RELATIVE}/formal_result.json",
    a_form_descriptive=f"{FORMAL_ROOT_RELATIVE}/A_form.descriptive.json",
)


SYNTHETIC_OUTPUT_PATHS = LifecycleOutputPaths(
    marker=f"{SYNTHETIC_ROOT_RELATIVE}/runner.one_shot_marker.json",
    failure=f"{SYNTHETIC_ROOT_RELATIVE}/runner.terminal_failure.json",
    result=f"{SYNTHETIC_ROOT_RELATIVE}/result.json",
    a_form_descriptive=f"{SYNTHETIC_ROOT_RELATIVE}/A_form.descriptive.json",
)


@dataclass(frozen=True)
class PrerequisiteBinding:
    implementation_freeze_sha256: str
    acquisition_receipt_sha256: str
    verified_git_head: str

    def validate(self) -> "PrerequisiteBinding":
        _require_sha256(
            self.implementation_freeze_sha256, "implementation freeze"
        )
        _require_sha256(self.acquisition_receipt_sha256, "acquisition receipt")
        if _GIT_SHA1.fullmatch(self.verified_git_head) is None:
            raise HoVerFormalControllerError("prerequisite Git HEAD is invalid")
        return self


@dataclass(frozen=True)
class LifecycleArtifact:
    kind: str
    block: str | None
    receipt_sha256: str
    payload: Mapping[str, Any]

    def validate(self, *, kind: str, block: str | None) -> "LifecycleArtifact":
        if self.kind != kind or self.block != block or not isinstance(
            self.payload, Mapping
        ):
            raise HoVerFormalControllerError("lifecycle artifact identity drifted")
        _require_sha256(self.receipt_sha256, f"{kind} receipt")
        return self


@dataclass(frozen=True)
class PolicyHandle:
    evaluator_id: str
    action_id: str
    selection_sha256: str
    runtime_policy: object


@dataclass(frozen=True)
class PolicyPair:
    e0: PolicyHandle
    e1: PolicyHandle
    identifiable: bool

    def public_payload(self) -> dict[str, Any]:
        for label, policy in (("E0", self.e0), ("E1", self.e1)):
            if not policy.action_id or not policy.evaluator_id:
                raise HoVerFormalControllerError(f"{label} policy identity drifted")
            _require_sha256(policy.selection_sha256, f"{label} policy")
        return {
            "e0_action_id": self.e0.action_id,
            "e0_policy_sha256": self.e0.selection_sha256,
            "e1_action_id": self.e1.action_id,
            "e1_policy_sha256": self.e1.selection_sha256,
            "policies_identifiable": self.identifiable,
        }


@dataclass(frozen=True)
class AHoldOutcome:
    primary_passed: bool
    promoted: bool
    report: Mapping[str, Any]


@dataclass(frozen=True)
class MSearchOutcome:
    l5_passed: bool
    report: Mapping[str, Any]


class LifecycleAcquisition(Protocol):
    """Only boundary allowed to know acquisition files or capability APIs."""

    def verify_prerequisites(self, *, project: Path) -> PrerequisiteBinding: ...

    def preflight_outputs(
        self,
        *,
        project: Path,
        config: FormalRuntimeConfig,
        output_paths: LifecycleOutputPaths,
    ) -> Mapping[str, Any]: ...

    def assert_repository_stable(
        self, *, project: Path, prerequisites: PrerequisiteBinding
    ) -> None: ...

    def load_corpus_view(self, *, project: Path) -> Mapping[str, Any]: ...

    def load_block_view(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]: ...

    def load_block_labels(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]: ...

    def archive_stage(
        self, *, project: Path, prepared: object, stage: object
    ) -> LifecycleArtifact: ...

    def seal_stage(
        self, *, project: Path, block: str, archive: LifecycleArtifact
    ) -> LifecycleArtifact: ...

    def freeze_a_form_evaluators(
        self,
        *,
        project: Path,
        policies: PolicyPair,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> LifecycleArtifact: ...

    def freeze_f_policies(
        self,
        *,
        project: Path,
        policies: PolicyPair,
        archive: LifecycleArtifact,
    ) -> LifecycleArtifact: ...

    def validate_a_hold_outcome(
        self,
        *,
        project: Path,
        outcome: AHoldOutcome,
        policy_freeze: LifecycleArtifact,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> None: ...

    def validate_m_search_outcome(
        self,
        *,
        project: Path,
        outcome: MSearchOutcome,
        policy_freeze: LifecycleArtifact,
        promotion: LifecycleArtifact,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> None: ...

    def authorize_promotion(
        self,
        *,
        project: Path,
        outcome: AHoldOutcome,
        policy_freeze: LifecycleArtifact,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> LifecycleArtifact: ...


class RuntimeFactory(Protocol):
    def create_encoder(self, config: FormalRuntimeConfig) -> object: ...

    def create_hippo(self, config: FormalRuntimeConfig) -> object: ...

    def create_ner_context(
        self, config: FormalRuntimeConfig
    ) -> AbstractContextManager[object]: ...


class JointGraphCore(Protocol):
    def prepare(
        self,
        *,
        corpus_view: Mapping[str, Any],
        encoder: object,
        ner: object,
        hippo: object,
        config: FormalRuntimeConfig,
    ) -> object: ...

    def execute(
        self,
        *,
        block: str,
        view: Mapping[str, Any],
        prepared: object,
        encoder: object,
        ner: object,
        hippo: object,
        config: FormalRuntimeConfig,
    ) -> object: ...

    def descriptive(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        policies: PolicyPair,
    ) -> Mapping[str, Any]: ...

    def select_label_free_policies(
        self, *, stage: object, expected_block: str
    ) -> PolicyPair: ...

    def select_f_policies(self, *, stage: object) -> PolicyPair: ...

    def assess_a_hold(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        f_stage: object,
        policies: PolicyPair,
    ) -> AHoldOutcome: ...

    def assess_m_search(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        f_stage: object,
        policies: PolicyPair,
    ) -> MSearchOutcome: ...


@dataclass(frozen=True)
class RunnerCoreAdapter:
    """Thin adapter over the committed HoVer prepare/execute/score core."""

    executor_factory: runner.ExecutorFactory = runner.spawn_process_pool_executor

    def prepare(
        self,
        *,
        corpus_view: Mapping[str, Any],
        encoder: object,
        ner: object,
        hippo: object,
        config: FormalRuntimeConfig,
    ) -> runner.PreparedCorpus:
        return runner.prepare_offline_corpus(
            corpus_view=corpus_view,
            encoder=encoder,  # type: ignore[arg-type]
            ner=ner,  # type: ignore[arg-type]
            hippo=hippo,  # type: ignore[arg-type]
            ner_batch_size=config.ner_batch_size,
            formal_shape=True,
        )

    def execute(
        self,
        *,
        block: str,
        view: Mapping[str, Any],
        prepared: object,
        encoder: object,
        ner: object,
        hippo: object,
        config: FormalRuntimeConfig,
    ) -> runner.StageExecution:
        if not isinstance(prepared, runner.PreparedCorpus):
            raise HoVerFormalControllerError("prepared corpus type drifted")
        return runner.execute_gold_free_stage(
            block=block,
            view=view,
            prepared=prepared,
            encoder=encoder,  # type: ignore[arg-type]
            ner=ner,  # type: ignore[arg-type]
            hippo=hippo,  # type: ignore[arg-type]
            ner_batch_size=config.ner_batch_size,
            local_worker_cap=config.local_worker_cap,
            formal_shape=True,
            executor_factory=self.executor_factory,
            agent_pycache_root=config.hippo_work_root / "agent_pycache",
        )

    def descriptive(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        policies: PolicyPair,
    ) -> Mapping[str, Any]:
        if not isinstance(stage, runner.StageExecution):
            raise HoVerFormalControllerError("descriptive stage type drifted")
        return runner.descriptive_stage_scores(
            stage=stage,
            labels=labels,
            e0_policy=policies.e0.runtime_policy,  # type: ignore[arg-type]
            e1_policy=policies.e1.runtime_policy,  # type: ignore[arg-type]
        )

    def select_label_free_policies(
        self, *, stage: object, expected_block: str
    ) -> PolicyPair:
        if not isinstance(stage, runner.StageExecution):
            raise HoVerFormalControllerError("label-free stage type drifted")
        e0, e1, identifiable = runner.select_label_free_policies(
            stage=stage, expected_block=expected_block
        )
        return PolicyPair(
            e0=PolicyHandle(e0.evaluator_id, e0.action_id, e0.selection_sha256, e0),
            e1=PolicyHandle(e1.evaluator_id, e1.action_id, e1.selection_sha256, e1),
            identifiable=identifiable,
        )

    def select_f_policies(self, *, stage: object) -> PolicyPair:
        return self.select_label_free_policies(
            stage=stage, expected_block="F_search"
        )

    def assess_a_hold(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        f_stage: object,
        policies: PolicyPair,
    ) -> AHoldOutcome:
        if not isinstance(stage, runner.StageExecution) or not isinstance(
            f_stage, runner.StageExecution
        ):
            raise HoVerFormalControllerError("A_hold stage type drifted")
        assessment = runner.decide_a_hold_promotion(
            stage=stage,
            labels=labels,
            f_stage=f_stage,
            e0_policy=policies.e0.runtime_policy,  # type: ignore[arg-type]
            e1_policy=policies.e1.runtime_policy,  # type: ignore[arg-type]
        )
        report = {
            "primary_passed": assessment.primary_passed,
            "promoted": assessment.promoted,
            "promotion_delta_total": _fraction(assessment.promotion_delta_total),
            "promotion_signflip_p": _fraction(assessment.promotion_signflip_p),
            "e0_minus_hippo_delta_total": _fraction(
                assessment.e0_minus_hippo_delta_total
            ),
            "e0_minus_hippo_signflip_p": _fraction(
                assessment.e0_minus_hippo_signflip_p
            ),
            "e0_minus_hippo_stratum_deltas": {
                name: _fraction(value)
                for name, value in assessment.e0_minus_hippo_stratum_deltas
            },
            "e0_minus_raw_delta_total": _fraction(
                assessment.e0_minus_raw_delta_total
            ),
            "e0_minus_raw_signflip_p": _fraction(
                assessment.e0_minus_raw_signflip_p
            ),
            "e0_complete_count": assessment.e0_complete_count,
            "raw_complete_count": assessment.raw_complete_count,
        }
        return AHoldOutcome(
            primary_passed=assessment.primary_passed,
            promoted=assessment.promoted,
            report=report,
        )

    def assess_m_search(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        f_stage: object,
        policies: PolicyPair,
    ) -> MSearchOutcome:
        if not isinstance(stage, runner.StageExecution) or not isinstance(
            f_stage, runner.StageExecution
        ):
            raise HoVerFormalControllerError("M stage type drifted")
        assessment = runner.assess_m_search(
            stage=stage,
            labels=labels,
            f_stage=f_stage,
            e0_policy=policies.e0.runtime_policy,  # type: ignore[arg-type]
            e1_policy=policies.e1.runtime_policy,  # type: ignore[arg-type]
        )
        report = {
            "l5_passed": assessment.l5_passed,
            "l5_delta_total": _fraction(assessment.l5_delta_total),
            "l5_signflip_p": _fraction(assessment.l5_signflip_p),
            "e1_minus_hippo_delta_total": _fraction(
                assessment.e1_minus_hippo_delta_total
            ),
            "e1_minus_hippo_signflip_p": _fraction(
                assessment.e1_minus_hippo_signflip_p
            ),
            "e1_minus_hippo_stratum_deltas": {
                name: _fraction(value)
                for name, value in assessment.e1_minus_hippo_stratum_deltas
            },
            "e1_minus_raw_delta_total": _fraction(
                assessment.e1_minus_raw_delta_total
            ),
            "e1_minus_raw_signflip_p": _fraction(
                assessment.e1_minus_raw_signflip_p
            ),
            "e1_complete_count": assessment.e1_complete_count,
            "raw_complete_count": assessment.raw_complete_count,
        }
        return MSearchOutcome(l5_passed=assessment.l5_passed, report=report)


def _find_receipt_sha256(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, str) and _SHA256.fullmatch(value):
            return value
    raise HoVerFormalControllerError("acquisition receipt hash field is absent")


def _artifact(
    *,
    kind: str,
    block: str | None,
    payload: Mapping[str, Any],
    binding: Mapping[str, Any] | None = None,
) -> LifecycleArtifact:
    body = {
        "kind": kind,
        "block": block,
        "payload_sha256": stable_hash(dict(payload)),
        "binding": {} if binding is None else dict(binding),
    }
    return LifecycleArtifact(
        kind=kind,
        block=block,
        receipt_sha256=stable_hash(body),
        payload=dict(payload),
    )


class ModuleAcquisitionAdapter:
    """Bind direct acquisition, implementation freeze, and lifecycle storage."""

    def __init__(self) -> None:
        self._verified_git_head: str | None = None
        self._implementation_freeze_sha256: str | None = None

    def verify_prerequisites(self, *, project: Path) -> PrerequisiteBinding:
        isolated_bootstrap.assert_isolated(
            "assumption_agent.benchmarks.hover_joint_graph_formal_controller_v1"
        )
        implementation = implementation_freeze.verify_committed_implementation_freeze(
            project
        )
        implementation_freeze.import_and_verify_frozen_python_roles(
            project=project, implementation_receipt=implementation
        )
        receipt, binding = acquisition.load_formal_committed_acquisition_receipt(
            project
        )
        if not isinstance(implementation, Mapping) or not isinstance(
            receipt, Mapping
        ) or not isinstance(binding, Mapping):
            raise HoVerFormalControllerError("committed prerequisite receipt drifted")
        implementation_head = implementation.get("verified_git_head")
        acquisition_head = binding.get("receipt_git_head")
        if (
            not isinstance(implementation_head, str)
            or _GIT_SHA1.fullmatch(implementation_head) is None
            or acquisition_head != implementation_head
        ):
            raise HoVerFormalControllerError(
                "implementation and acquisition Git HEAD differ"
            )
        binding = PrerequisiteBinding(
            implementation_freeze_sha256=_find_receipt_sha256(
                implementation,
                ("implementation_freeze_sha256", "freeze_sha256"),
            ),
            acquisition_receipt_sha256=_find_receipt_sha256(
                receipt, ("acquisition_sha256", "acquisition_receipt_sha256")
            ),
            verified_git_head=implementation_head,
        ).validate()
        self._verified_git_head = implementation_head
        self._implementation_freeze_sha256 = binding.implementation_freeze_sha256
        return binding

    def assert_repository_stable(
        self, *, project: Path, prerequisites: PrerequisiteBinding
    ) -> None:
        prerequisites.validate()
        if self._verified_git_head != prerequisites.verified_git_head:
            raise HoVerFormalControllerError("adapter prerequisite HEAD drifted")
        verified = implementation_freeze.verify_committed_implementation_freeze(
            project
        )
        if (
            verified.get("verified_git_head") != prerequisites.verified_git_head
            or verified.get(implementation_freeze.HASH_FIELD)
            != self._implementation_freeze_sha256
        ):
            raise HoVerFormalControllerError(
                "implementation closure changed after prerequisite verification"
            )

    def preflight_outputs(
        self,
        *,
        project: Path,
        config: FormalRuntimeConfig,
        output_paths: LifecycleOutputPaths,
    ) -> Mapping[str, Any]:
        private_pack = acquisition.preflight_formal_private_pack_files(project)
        lifecycle_paths = lifecycle_store.preflight_lifecycle_outputs_absent(project)
        controller_paths = preflight_controller_output_paths(
            project=project, config=config, output_paths=output_paths
        )
        return {
            "lifecycle_output_count": len(lifecycle_paths),
            "controller_output_count": len(controller_paths),
            "private_pack_file_count": private_pack["private_pack_file_count"],
            "private_pack_json_payloads_decoded": 0,
            "all_outcome_paths_absent": True,
        }

    def load_corpus_view(self, *, project: Path) -> Mapping[str, Any]:
        return acquisition.load_corpus_view(project=project)

    def load_block_view(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        return acquisition.load_block_view(
            project=project, expected_block=expected_block
        )

    def load_block_labels(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        if expected_block == "F_search":
            raise HoVerFormalControllerError("F_search has no utility label pack")
        return acquisition.load_block_labels(
            project=project, expected_block=expected_block
        )

    def archive_stage(
        self, *, project: Path, prepared: object, stage: object
    ) -> LifecycleArtifact:
        if not isinstance(prepared, runner.PreparedCorpus) or not isinstance(
            stage, runner.StageExecution
        ):
            raise HoVerFormalControllerError("stage archive input type drifted")
        view_items = stage.view.get("items")
        if not isinstance(view_items, list) or len(view_items) != len(stage.items):
            raise HoVerFormalControllerError("stage/view archive binding drifted")
        records = tuple(
            lifecycle_store.build_stage_output_record(
                block=stage.block,
                ordinal=item.ordinal,
                view_sha256=stable_hash(view_items[item.ordinal]),
                dense_relevance_ints=item.query_feature.dense_relevance_ints,
                raw_top5=item.raw_top5,
                hipporag_top5=item.hippo_top5,
                action_traces=item.traces,
            )
            for item in stage.items
        )
        runtime_binding = {
            "preparation_sha256": prepared.preparation_sha256,
            "graph_sha256": stage.graph_sha256,
            "embedding_index_sha256": stage.embedding_index_sha256,
            "ner_runtime_receipt_sha256": prepared.ner_runtime_receipt_sha256,
            "ner_entity_matrix_sha256": prepared.ner_entity_matrix_sha256,
            "hippo_build_receipt_sha256": stage.hippo_build_receipt_sha256,
            "hippo_retrieval_receipt_sha256": (
                stage.hippo_retrieval_receipt_sha256
            ),
            "execution_matrix_sha256": stage.execution_matrix_sha256,
        }
        created = lifecycle_store.create_stage_output_archive_once(
            project=project,
            block=stage.block,
            records=records,
            stage_runtime_binding=runtime_binding,
        )
        payload, binding = created
        if not isinstance(payload, Mapping) or not isinstance(binding, Mapping):
            raise HoVerFormalControllerError("stage archive capability drifted")
        return _artifact(
            kind="stage_archive",
            block=stage.block,
            payload=payload,
            binding=binding,
        )

    def seal_stage(
        self, *, project: Path, block: str, archive: LifecycleArtifact
    ) -> LifecycleArtifact:
        archive.validate(kind="stage_archive", block=block)
        payload = lifecycle_store.create_action_seal_once(
            project=project, block=block
        )
        if not isinstance(payload, Mapping):
            raise HoVerFormalControllerError("action seal capability drifted")
        return _artifact(kind="action_seal", block=block, payload=payload)

    def freeze_a_form_evaluators(
        self,
        *,
        project: Path,
        policies: PolicyPair,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> LifecycleArtifact:
        archive.validate(kind="stage_archive", block="A_form")
        seal.validate(kind="action_seal", block="A_form")
        if not isinstance(policies.e0.runtime_policy, PolicySelection) or not isinstance(
            policies.e1.runtime_policy, PolicySelection
        ):
            raise HoVerFormalControllerError("A_form policy type drifted")
        payload = lifecycle_store.create_a_form_evaluator_freeze_once(
            project=project,
            e0_policy=policies.e0.runtime_policy,
            e1_policy=policies.e1.runtime_policy,
        )
        public = policies.public_payload()
        if (
            payload.get("selection_purpose") != "diagnostic_only_not_F_policy"
            or payload.get("policies_identifiable") is not policies.identifiable
            or payload.get("e0_policy", {}).get("selection_sha256")
            != public["e0_policy_sha256"]
            or payload.get("e1_policy", {}).get("selection_sha256")
            != public["e1_policy_sha256"]
        ):
            raise HoVerFormalControllerError("A_form evaluator freeze drifted")
        return _artifact(
            kind="evaluator_freeze", block="A_form", payload=payload
        )

    def freeze_f_policies(
        self,
        *,
        project: Path,
        policies: PolicyPair,
        archive: LifecycleArtifact,
    ) -> LifecycleArtifact:
        archive.validate(kind="stage_archive", block="F_search")
        if not policies.identifiable:
            raise HoVerFormalControllerError("unidentifiable policies cannot freeze")
        if not isinstance(policies.e0.runtime_policy, PolicySelection) or not isinstance(
            policies.e1.runtime_policy, PolicySelection
        ):
            raise HoVerFormalControllerError("F policy type drifted")
        payload = lifecycle_store.create_f_search_policy_freeze_once(
            project=project,
            e0_policy=policies.e0.runtime_policy,
            e1_policy=policies.e1.runtime_policy,
        )
        public = policies.public_payload()
        if not isinstance(payload, Mapping) or any(
            payload.get(field) != public[field]
            for field in (
                "e0_action_id",
                "e0_policy_sha256",
                "e1_action_id",
                "e1_policy_sha256",
            )
        ):
            raise HoVerFormalControllerError("F policy freeze binding drifted")
        return _artifact(
            kind="policy_freeze", block="F_search", payload=payload
        )

    def validate_a_hold_outcome(
        self,
        *,
        project: Path,
        outcome: AHoldOutcome,
        policy_freeze: LifecycleArtifact,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> None:
        policy_freeze.validate(kind="policy_freeze", block="F_search")
        archive.validate(kind="stage_archive", block="A_hold")
        seal.validate(kind="action_seal", block="A_hold")
        recomputed = lifecycle_store.recompute_a_hold_outcome_report(
            project=project
        )
        if (
            not isinstance(outcome.report, Mapping)
            or _canonical_bytes(dict(outcome.report))
            != _canonical_bytes(recomputed)
            or recomputed.get("primary_passed") is not outcome.primary_passed
            or recomputed.get("promoted") is not outcome.promoted
        ):
            raise HoVerFormalControllerError(
                "A_hold controller outcome differs from sealed evidence"
            )

    def authorize_promotion(
        self,
        *,
        project: Path,
        outcome: AHoldOutcome,
        policy_freeze: LifecycleArtifact,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> LifecycleArtifact:
        if not outcome.promoted:
            raise HoVerFormalControllerError("nonpromotion cannot authorize M")
        policy_freeze.validate(kind="policy_freeze", block="F_search")
        archive.validate(kind="stage_archive", block="A_hold")
        seal.validate(kind="action_seal", block="A_hold")
        payload = lifecycle_store.create_a_hold_promotion_once(
            project=project, outcome_report=outcome.report
        )
        if not isinstance(payload, Mapping):
            raise HoVerFormalControllerError("promotion capability drifted")
        return _artifact(
            kind="promotion_authorization", block="A_hold", payload=payload
        )

    def validate_m_search_outcome(
        self,
        *,
        project: Path,
        outcome: MSearchOutcome,
        policy_freeze: LifecycleArtifact,
        promotion: LifecycleArtifact,
        archive: LifecycleArtifact,
        seal: LifecycleArtifact,
    ) -> None:
        policy_freeze.validate(kind="policy_freeze", block="F_search")
        promotion.validate(
            kind="promotion_authorization", block="A_hold"
        )
        archive.validate(kind="stage_archive", block="M_search")
        seal.validate(kind="action_seal", block="M_search")
        lifecycle_store.validate_m_search_outcome_report(
            project=project,
            outcome_report=outcome.report,
            l5_passed=outcome.l5_passed,
        )


class DefaultLocalRuntimeFactory:
    """Reuses the committed local-only MiniLM, NER, and Hippo gateways."""

    def create_encoder(self, config: FormalRuntimeConfig) -> object:
        return OfflineMiniLMEncoder(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )

    def create_hippo(self, config: FormalRuntimeConfig) -> object:
        return OfficialHippoGateway(
            runtime_python=config.hippo_runtime_python,
            local_llm_model=config.hippo_llm_model,
            local_embedding_model=config.hippo_embedding_model,
            base_binding_receipt_path=config.hippo_base_binding_receipt,
            attestation_receipt_path=config.hippo_attestation_receipt,
            stage_root=config.hippo_stage_root,
            work_root=config.hippo_work_root,
        )

    def create_ner_context(
        self, config: FormalRuntimeConfig
    ) -> AbstractContextManager[object]:
        return OfflineNERJSONLClient(
            project_root=config.project,
            asset_manifest_path=config.ner_asset_manifest,
            model_root=config.ner_model_root,
        )


def default_formal_runtime_config(project: Path) -> FormalRuntimeConfig:
    return local_runtime.default_formal_runtime_config(project)


def preflight_formal_runtime_config(
    config: FormalRuntimeConfig,
) -> Mapping[str, Any]:
    if not isinstance(config, FormalRuntimeConfig):
        raise HoVerFormalControllerError("formal runtime config type drifted")
    project = config.project.resolve(strict=True)
    if config != default_formal_runtime_config(project):
        raise HoVerFormalControllerError("formal runtime config is not canonical")
    return local_runtime.preflight_formal_runtime_config(config)


def _reject_unsafe_output_ancestors(*, project: Path, path: Path) -> None:
    try:
        relative = path.absolute().relative_to(project)
    except ValueError as exc:
        raise HoVerFormalControllerError("formal output escaped project root") from exc
    cursor = project
    for part in relative.parts[:-1]:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise HoVerFormalControllerError("formal output ancestor is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HoVerFormalControllerError("formal output ancestor is unsafe")


def preflight_controller_output_paths(
    *,
    project: Path,
    config: FormalRuntimeConfig,
    output_paths: LifecycleOutputPaths,
) -> tuple[Path, ...]:
    """Prepare only the empty Hippo work parent; reject every other residue."""

    root = project.resolve(strict=True)
    if config != default_formal_runtime_config(root):
        raise HoVerFormalControllerError("output preflight config drifted")
    owned = tuple(
        root / relative
        for relative in (
            output_paths.marker,
            output_paths.failure,
            output_paths.result,
            output_paths.a_form_descriptive,
        )
    )
    if len(set(owned)) != len(owned):
        raise HoVerFormalControllerError("controller output paths overlap")
    for path in (*owned, config.hippo_stage_root, config.hippo_work_root):
        _reject_unsafe_output_ancestors(project=root, path=path)
    occupied = [
        path
        for path in (*owned, config.hippo_stage_root, config.hippo_work_root)
        if os.path.lexists(path)
    ]
    if occupied:
        raise HoVerFormalControllerError("formal controller output already exists")

    formal_root = root / FORMAL_ROOT_RELATIVE
    if os.path.lexists(formal_root):
        info = formal_root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HoVerFormalControllerError("formal root is unsafe")
        unexpected = list(formal_root.iterdir())
        if unexpected:
            raise HoVerFormalControllerError("formal root contains residual outputs")
    return (*owned, config.hippo_stage_root, config.hippo_work_root)


def ensure_durable_output_directory(*, project: Path, directory: Path) -> Path:
    """Create an output directory chain and fsync every parent entry."""

    root = project.resolve(strict=True)
    try:
        relative = directory.absolute().relative_to(root)
    except ValueError as exc:
        raise HoVerFormalControllerError(
            "durable output directory escaped project"
        ) from exc
    cursor = root
    for part in relative.parts:
        child = cursor / part
        if not os.path.lexists(child):
            try:
                child.mkdir(mode=0o700)
            except OSError as exc:
                raise HoVerFormalControllerError(
                    "durable output directory creation failed"
                ) from exc
        try:
            metadata = child.lstat()
        except OSError as exc:
            raise HoVerFormalControllerError(
                "durable output directory is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise HoVerFormalControllerError(
                "durable output directory is unsafe"
            )
        try:
            descriptor = os.open(
                cursor,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise HoVerFormalControllerError(
                "durable output directory fsync failed"
            ) from exc
        cursor = child
    return cursor


def prepare_hippo_work_root(config: FormalRuntimeConfig) -> Path:
    """Create the sole empty runtime work root after the cohort marker."""

    project = config.project.resolve(strict=True)
    if config != default_formal_runtime_config(project):
        raise HoVerFormalControllerError("Hippo work-root config drifted")
    path = config.hippo_work_root
    _reject_unsafe_output_ancestors(project=project, path=path)
    try:
        path.mkdir(mode=0o700, parents=True)
    except OSError as exc:
        raise HoVerFormalControllerError(
            "Hippo query work root cannot be prepared"
        ) from exc
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise HoVerFormalControllerError("Hippo query work root is unsafe")
    return path


def write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, mode: int
) -> str:
    """Write canonical JSON using O_EXCL and reject a symlink final path."""

    if mode not in {0o600, 0o644}:
        raise HoVerFormalControllerError("output mode is invalid")
    absolute = Path(os.path.abspath(os.fspath(path)))
    absolute.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    raw = _canonical_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags, mode)
    except OSError as exc:
        raise HoVerFormalControllerError(
            "exclusive output already exists or is unsafe"
        ) from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        info = absolute.stat(follow_symlinks=False)
        # DrvFS and other mounted filesystems can ignore chmod bits.  O_EXCL,
        # O_NOFOLLOW, and the regular-file check remain authoritative there.
        if not stat.S_ISREG(info.st_mode):
            raise HoVerFormalControllerError("exclusive output type drifted")
        parent_descriptor = os.open(
            absolute.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return hashlib.sha256(raw).hexdigest()


def _one_shot_marker_payload(
    prerequisites: PrerequisiteBinding,
) -> dict[str, Any]:
    prerequisites.validate()
    return _self_hashed(
        {
            "schema": MARKER_SCHEMA,
            "version": VERSION,
            "phase": "formal_A_form_F_A_hold_M_one_shot",
            "implementation_freeze_sha256": (
                prerequisites.implementation_freeze_sha256
            ),
            "acquisition_receipt_sha256": (
                prerequisites.acquisition_receipt_sha256
            ),
            "verified_git_head": prerequisites.verified_git_head,
            "replay_retry_resample_replacement_authorized": False,
        },
        "marker_sha256",
    )


def consume_one_shot_marker(
    *, path: Path, prerequisites: PrerequisiteBinding
) -> dict[str, Any]:
    marker = _one_shot_marker_payload(prerequisites)
    write_json_exclusive(path, marker, mode=0o600)
    return marker


def _write_terminal_failure(
    *, path: Path, marker_sha256: str, stage: str, exc: BaseException
) -> None:
    payload = _self_hashed(
        {
            "schema": FAILURE_SCHEMA,
            "version": VERSION,
            "status": "terminal_cohort_burned_no_replay",
            "marker_sha256": _require_sha256(marker_sha256, "marker"),
            "failure_stage": stage,
            "exception_type_sha256": hashlib.sha256(
                f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
            ).hexdigest(),
            "private_item_or_label_content_included": False,
            "replay_retry_resample_replacement_authorized": False,
        },
        "failure_sha256",
    )
    try:
        write_json_exclusive(path, payload, mode=0o644)
    except BaseException:
        pass


def _run_lifecycle_core(
    config: FormalRuntimeConfig,
    *,
    acquisition_adapter: LifecycleAcquisition,
    runtime_factory: RuntimeFactory,
    core: JointGraphCore,
    output_paths: LifecycleOutputPaths,
) -> dict[str, Any]:
    if not isinstance(config, FormalRuntimeConfig):
        raise HoVerFormalControllerError("runtime config type drifted")
    if (
        config.local_worker_cap != runner.LOCAL_CONCURRENCY_CAP
        or config.ner_batch_size != runner.DEFAULT_NER_BATCH_SIZE
    ):
        raise HoVerFormalControllerError("runtime concurrency binding drifted")
    project = config.project.resolve(strict=True)
    prerequisites = acquisition_adapter.verify_prerequisites(project=project).validate()
    acquisition_adapter.assert_repository_stable(
        project=project, prerequisites=prerequisites
    )
    acquisition_adapter.preflight_outputs(
        project=project, config=config, output_paths=output_paths
    )
    acquisition_adapter.assert_repository_stable(
        project=project, prerequisites=prerequisites
    )
    marker_path = project / output_paths.marker
    ensure_durable_output_directory(
        project=project,
        directory=marker_path.parent,
    )
    marker_existed_before = os.path.lexists(marker_path)
    marker_sha256 = str(
        _one_shot_marker_payload(prerequisites)["marker_sha256"]
    )
    try:
        marker = consume_one_shot_marker(
            path=marker_path,
            prerequisites=prerequisites,
        )
    except BaseException as exc:
        if not marker_existed_before and os.path.lexists(marker_path):
            _write_terminal_failure(
                path=project / output_paths.failure,
                marker_sha256=marker_sha256,
                stage="one_shot_marker_consumption",
                exc=exc,
            )
        raise
    if marker.get("marker_sha256") != marker_sha256:
        raise HoVerFormalControllerError("one-shot marker identity drifted")
    failure_stage = "runtime_initialization"
    artifacts: dict[str, str] = {}

    def record(name: str, artifact: LifecycleArtifact, kind: str, block: str | None) -> None:
        artifact.validate(kind=kind, block=block)
        artifacts[name] = artifact.receipt_sha256

    def finish(status: str, body: Mapping[str, Any]) -> dict[str, Any]:
        acquisition_adapter.assert_repository_stable(
            project=project, prerequisites=prerequisites
        )
        result = _self_hashed(
            {
                "schema": RESULT_SCHEMA,
                "version": VERSION,
                "status": status,
                "marker_sha256": marker_sha256,
                "implementation_freeze_sha256": (
                    prerequisites.implementation_freeze_sha256
                ),
                "acquisition_receipt_sha256": (
                    prerequisites.acquisition_receipt_sha256
                ),
                "verified_git_head": prerequisites.verified_git_head,
                "artifact_receipt_sha256s": dict(sorted(artifacts.items())),
                **dict(body),
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "same_source_replay_authorized": False,
            },
            "result_sha256",
        )
        write_json_exclusive(project / output_paths.result, result, mode=0o644)
        return result

    try:
        if output_paths == FORMAL_OUTPUT_PATHS:
            failure_stage = "Hippo_work_root_initialization"
            prepare_hippo_work_root(config)
        failure_stage = "runtime_initialization"
        encoder = runtime_factory.create_encoder(config)
        hippo = runtime_factory.create_hippo(config)
        ner_context = runtime_factory.create_ner_context(config)
        with ExitStack() as runtime_stack:
            ner = runtime_stack.enter_context(ner_context)
            failure_stage = "corpus_preparation"
            corpus_view = acquisition_adapter.load_corpus_view(project=project)
            prepared = core.prepare(
                corpus_view=corpus_view,
                encoder=encoder,
                ner=ner,
                hippo=hippo,
                config=config,
            )
            acquisition_adapter.assert_repository_stable(
                project=project, prerequisites=prerequisites
            )

            def execute_archive(block: str) -> tuple[object, LifecycleArtifact]:
                nonlocal failure_stage
                acquisition_adapter.assert_repository_stable(
                    project=project, prerequisites=prerequisites
                )
                failure_stage = f"{block}_claim_view_and_gold_free_execution"
                view = acquisition_adapter.load_block_view(
                    project=project, expected_block=block
                )
                stage = core.execute(
                    block=block,
                    view=view,
                    prepared=prepared,
                    encoder=encoder,
                    ner=ner,
                    hippo=hippo,
                    config=config,
                )
                acquisition_adapter.assert_repository_stable(
                    project=project, prerequisites=prerequisites
                )
                failure_stage = f"{block}_canonical_archive"
                archive = acquisition_adapter.archive_stage(
                    project=project, prepared=prepared, stage=stage
                )
                record(f"{block}_archive", archive, "stage_archive", block)
                return stage, archive

            a_form_stage, a_form_archive = execute_archive("A_form")
            failure_stage = "A_form_action_seal"
            a_form_seal = acquisition_adapter.seal_stage(
                project=project, block="A_form", archive=a_form_archive
            )
            record("A_form_seal", a_form_seal, "action_seal", "A_form")
            failure_stage = "A_form_prelabel_evaluator_selection_and_freeze"
            a_form_policies = core.select_label_free_policies(
                stage=a_form_stage, expected_block="A_form"
            )
            a_form_evaluator_freeze = acquisition_adapter.freeze_a_form_evaluators(
                project=project,
                policies=a_form_policies,
                archive=a_form_archive,
                seal=a_form_seal,
            )
            record(
                "A_form_evaluator_freeze",
                a_form_evaluator_freeze,
                "evaluator_freeze",
                "A_form",
            )
            failure_stage = "A_form_late_descriptive"
            acquisition_adapter.assert_repository_stable(
                project=project, prerequisites=prerequisites
            )
            a_form_labels = acquisition_adapter.load_block_labels(
                project=project, expected_block="A_form"
            )
            descriptive_report = core.descriptive(
                stage=a_form_stage,
                labels=a_form_labels,
                policies=a_form_policies,
            )
            if not isinstance(descriptive_report, Mapping):
                raise HoVerFormalControllerError("A_form descriptive report drifted")
            descriptive_receipt = _self_hashed(
                {
                    "schema": DESCRIPTIVE_SCHEMA,
                    "version": VERSION,
                    "status": "late_descriptive_only_no_policy_change",
                    "marker_sha256": marker_sha256,
                    "A_form_archive_receipt_sha256": (
                        a_form_archive.receipt_sha256
                    ),
                    "A_form_seal_receipt_sha256": a_form_seal.receipt_sha256,
                    "A_form_evaluator_freeze_receipt_sha256": (
                        a_form_evaluator_freeze.receipt_sha256
                    ),
                    "report": dict(descriptive_report),
                    "labels_opened_after_action_seal": True,
                    "labels_opened_after_evaluator_freeze": True,
                    "policy_or_threshold_changed": False,
                },
                "descriptive_receipt_sha256",
            )
            write_json_exclusive(
                project / output_paths.a_form_descriptive,
                descriptive_receipt,
                mode=0o600,
            )
            artifacts["A_form_descriptive"] = descriptive_receipt[
                "descriptive_receipt_sha256"
            ]

            f_stage, f_archive = execute_archive("F_search")
            failure_stage = "F_search_label_free_policy_selection"
            policies = core.select_f_policies(stage=f_stage)
            policy_payload = policies.public_payload()
            if not policies.identifiable:
                failure_stage = "runtime_shutdown"
                runtime_stack.close()
                return finish(
                    "valid_F_search_nonidentifiable_A_hold_and_M_unopened",
                    {
                        **policy_payload,
                        "F_search_policy_freeze_created": False,
                        "A_hold_view_or_labels_opened": False,
                        "M_search_view_or_labels_opened": False,
                    },
                )

            failure_stage = "F_search_policy_freeze"
            policy_freeze = acquisition_adapter.freeze_f_policies(
                project=project, policies=policies, archive=f_archive
            )
            record(
                "F_search_policy_freeze",
                policy_freeze,
                "policy_freeze",
                "F_search",
            )

            a_hold_stage, a_hold_archive = execute_archive("A_hold")
            failure_stage = "A_hold_action_seal"
            a_hold_seal = acquisition_adapter.seal_stage(
                project=project, block="A_hold", archive=a_hold_archive
            )
            record("A_hold_seal", a_hold_seal, "action_seal", "A_hold")
            failure_stage = "A_hold_late_labels_primary_and_promotion"
            acquisition_adapter.assert_repository_stable(
                project=project, prerequisites=prerequisites
            )
            a_hold_labels = acquisition_adapter.load_block_labels(
                project=project, expected_block="A_hold"
            )
            a_hold_outcome = core.assess_a_hold(
                stage=a_hold_stage,
                labels=a_hold_labels,
                f_stage=f_stage,
                policies=policies,
            )
            if not isinstance(a_hold_outcome.report, Mapping):
                raise HoVerFormalControllerError("A_hold outcome report drifted")
            failure_stage = "A_hold_sealed_evidence_recomputation"
            acquisition_adapter.validate_a_hold_outcome(
                project=project,
                outcome=a_hold_outcome,
                policy_freeze=policy_freeze,
                archive=a_hold_archive,
                seal=a_hold_seal,
            )
            if not a_hold_outcome.promoted:
                failure_stage = "runtime_shutdown"
                runtime_stack.close()
                return finish(
                    "valid_A_hold_nonpromotion_M_unopened",
                    {
                        **policy_payload,
                        "A_hold_primary_passed": a_hold_outcome.primary_passed,
                        "A_hold_promotion": dict(a_hold_outcome.report),
                        "M_search_view_or_labels_opened": False,
                    },
                )

            failure_stage = "A_hold_promotion_authorization"
            promotion = acquisition_adapter.authorize_promotion(
                project=project,
                outcome=a_hold_outcome,
                policy_freeze=policy_freeze,
                archive=a_hold_archive,
                seal=a_hold_seal,
            )
            record(
                "A_hold_promotion",
                promotion,
                "promotion_authorization",
                "A_hold",
            )

            m_stage, m_archive = execute_archive("M_search")
            failure_stage = "M_search_action_seal"
            m_seal = acquisition_adapter.seal_stage(
                project=project, block="M_search", archive=m_archive
            )
            record("M_search_seal", m_seal, "action_seal", "M_search")
            failure_stage = "M_search_late_labels_and_L5"
            acquisition_adapter.assert_repository_stable(
                project=project, prerequisites=prerequisites
            )
            m_labels = acquisition_adapter.load_block_labels(
                project=project, expected_block="M_search"
            )
            m_outcome = core.assess_m_search(
                stage=m_stage,
                labels=m_labels,
                f_stage=f_stage,
                policies=policies,
            )
            if not isinstance(m_outcome.report, Mapping):
                raise HoVerFormalControllerError("M_search outcome report drifted")
            failure_stage = "M_search_sealed_evidence_recomputation"
            acquisition_adapter.validate_m_search_outcome(
                project=project,
                outcome=m_outcome,
                policy_freeze=policy_freeze,
                promotion=promotion,
                archive=m_archive,
                seal=m_seal,
            )
            failure_stage = "runtime_shutdown"
            runtime_stack.close()
            return finish(
                "formal_M_search_complete",
                {
                    **policy_payload,
                    "A_hold_primary_passed": a_hold_outcome.primary_passed,
                    "A_hold_promotion": dict(a_hold_outcome.report),
                    "M_search_L5": dict(m_outcome.report),
                    "L5_passed": m_outcome.l5_passed,
                },
            )
    except BaseException as exc:
        _write_terminal_failure(
            path=project / output_paths.failure,
            marker_sha256=marker_sha256,
            stage=failure_stage,
            exc=exc,
        )
        raise


def run_formal_lifecycle(config: FormalRuntimeConfig) -> dict[str, Any]:
    """Run only the non-injectable, locally attested formal lifecycle."""

    preflight_formal_runtime_config(config)
    return _run_lifecycle_core(
        config,
        acquisition_adapter=ModuleAcquisitionAdapter(),
        runtime_factory=DefaultLocalRuntimeFactory(),
        core=RunnerCoreAdapter(),
        output_paths=FORMAL_OUTPUT_PATHS,
    )


def run_synthetic_lifecycle(
    config: FormalRuntimeConfig,
    *,
    acquisition_adapter: LifecycleAcquisition,
    runtime_factory: RuntimeFactory,
    core: JointGraphCore,
) -> dict[str, Any]:
    """Sentinel-gated lifecycle used only with fake capabilities in tests."""

    project = config.project.resolve(strict=True)
    sentinel = project / SYNTHETIC_SENTINEL
    if (
        sentinel.is_symlink()
        or not sentinel.is_file()
        or sentinel.read_text(encoding="ascii") != SYNTHETIC_SENTINEL_CONTENT
    ):
        raise HoVerFormalControllerError("synthetic lifecycle sentinel is absent")
    if any(
        (project / relative).exists() or (project / relative).is_symlink()
        for relative in (
            FORMAL_OUTPUT_PATHS.marker,
            FORMAL_OUTPUT_PATHS.failure,
            FORMAL_OUTPUT_PATHS.result,
            FORMAL_OUTPUT_PATHS.a_form_descriptive,
        )
    ):
        raise HoVerFormalControllerError(
            "synthetic lifecycle cannot coexist with formal outputs"
        )
    return _run_lifecycle_core(
        config,
        acquisition_adapter=acquisition_adapter,
        runtime_factory=runtime_factory,
        core=core,
        output_paths=SYNTHETIC_OUTPUT_PATHS,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    isolated_bootstrap.reexec_isolated(
        "assumption_agent.benchmarks.hover_joint_graph_formal_controller_v1",
        raw_arguments,
    )
    arguments = _parser().parse_args(raw_arguments)
    project = arguments.project.resolve(strict=True)
    result = run_formal_lifecycle(default_formal_runtime_config(project))
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_sha256": result["result_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "AHoldOutcome",
    "DefaultLocalRuntimeFactory",
    "FORMAL_OUTPUT_PATHS",
    "FormalRuntimeConfig",
    "HoVerFormalControllerError",
    "JointGraphCore",
    "LifecycleAcquisition",
    "LifecycleArtifact",
    "LifecycleOutputPaths",
    "MSearchOutcome",
    "ModuleAcquisitionAdapter",
    "OfflineNERJSONLClient",
    "OfficialHippoGateway",
    "PolicyHandle",
    "PolicyPair",
    "PrerequisiteBinding",
    "RunnerCoreAdapter",
    "RuntimeFactory",
    "SYNTHETIC_OUTPUT_PATHS",
    "SYNTHETIC_SENTINEL",
    "SYNTHETIC_SENTINEL_CONTENT",
    "VERSION",
    "consume_one_shot_marker",
    "default_formal_runtime_config",
    "ensure_durable_output_directory",
    "main",
    "preflight_controller_output_paths",
    "preflight_formal_runtime_config",
    "run_formal_lifecycle",
    "run_synthetic_lifecycle",
    "stable_hash",
    "verify_self_hash",
    "write_json_exclusive",
]


if __name__ == "__main__":
    from assumption_agent.benchmarks import (
        hover_joint_graph_formal_controller_v1 as _canonical,
    )

    raise SystemExit(_canonical.main())

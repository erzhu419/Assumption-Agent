"""One-shot lifecycle controller for the frozen HoVer joint-graph study.

The controller owns ordering and durable terminal receipts, while acquisition,
gold-free execution, and local runtimes remain replaceable boundaries.  The
formal entrypoint has no injectable dependencies.  Synthetic tests use the
separate sentinel-gated entrypoint and never open the formal source pack.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import hover_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks import hover_joint_graph_formal_runner_v1 as runner
from assumption_agent.benchmarks import multihoprag_joint_graph_formal_runner_v1 as legacy
from replication_runtime.qasper_minilm_v1.binding import OfflineMiniLMEncoder
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)


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

FormalRuntimeConfig = legacy.FormalRuntimeConfig
OfficialHippoGateway = legacy.OfficialHippoGateway
OfflineNERJSONLClient = legacy.OfflineNERJSONLClient


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

    def validate(self) -> "PrerequisiteBinding":
        _require_sha256(
            self.implementation_freeze_sha256, "implementation freeze"
        )
        _require_sha256(self.acquisition_receipt_sha256, "acquisition receipt")
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

    def freeze_f_policies(
        self,
        *,
        project: Path,
        policies: PolicyPair,
        archive: LifecycleArtifact,
    ) -> LifecycleArtifact: ...

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
        self, *, stage: object, labels: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

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
        )

    def descriptive(
        self, *, stage: object, labels: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(stage, runner.StageExecution):
            raise HoVerFormalControllerError("descriptive stage type drifted")
        return runner.descriptive_stage_scores(stage=stage, labels=labels)

    def select_f_policies(self, *, stage: object) -> PolicyPair:
        if not isinstance(stage, runner.StageExecution):
            raise HoVerFormalControllerError("F stage type drifted")
        e0, e1, identifiable = runner.select_f_policies(f_stage=stage)
        return PolicyPair(
            e0=PolicyHandle(e0.evaluator_id, e0.action_id, e0.selection_sha256, e0),
            e1=PolicyHandle(e1.evaluator_id, e1.action_id, e1.selection_sha256, e1),
            identifiable=identifiable,
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
    """Central translation point for the still-evolving acquisition module."""

    @staticmethod
    def _callable(name: str) -> Any:
        function = getattr(acquisition, name, None)
        if not callable(function):
            raise HoVerFormalControllerError(
                f"HoVer acquisition capability {name} is unavailable"
            )
        return function

    def verify_prerequisites(self, *, project: Path) -> PrerequisiteBinding:
        implementation = self._callable("verify_committed_implementation_freeze")(
            project
        )
        loaded = self._callable("load_committed_acquisition_receipt")(project)
        receipt = loaded[0] if isinstance(loaded, tuple) else loaded
        if not isinstance(implementation, Mapping) or not isinstance(
            receipt, Mapping
        ):
            raise HoVerFormalControllerError("committed prerequisite receipt drifted")
        return PrerequisiteBinding(
            implementation_freeze_sha256=_find_receipt_sha256(
                implementation,
                ("implementation_freeze_sha256", "freeze_sha256"),
            ),
            acquisition_receipt_sha256=_find_receipt_sha256(
                receipt, ("acquisition_sha256", "acquisition_receipt_sha256")
            ),
        ).validate()

    def load_corpus_view(self, *, project: Path) -> Mapping[str, Any]:
        return self._callable("load_corpus_view")(project=project)

    def load_block_view(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        return self._callable("load_block_view")(
            project=project, expected_block=expected_block
        )

    def load_block_labels(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        if expected_block == "F_search":
            raise HoVerFormalControllerError("F_search has no utility label pack")
        return self._callable("load_block_labels")(
            project=project, expected_block=expected_block
        )

    def archive_stage(
        self, *, project: Path, prepared: object, stage: object
    ) -> LifecycleArtifact:
        if not isinstance(prepared, runner.PreparedCorpus) or not isinstance(
            stage, runner.StageExecution
        ):
            raise HoVerFormalControllerError("stage archive input type drifted")
        builder = self._callable("build_stage_output_record")
        view_items = stage.view.get("items")
        if not isinstance(view_items, list) or len(view_items) != len(stage.items):
            raise HoVerFormalControllerError("stage/view archive binding drifted")
        records = tuple(
            builder(
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
        created = self._callable("create_stage_output_archive_once")(
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
        payload = self._callable("create_action_seal_once")(
            project=project, block=block
        )
        if not isinstance(payload, Mapping):
            raise HoVerFormalControllerError("action seal capability drifted")
        return _artifact(kind="action_seal", block=block, payload=payload)

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
        payload = self._callable("create_f_search_policy_freeze_once")(
            project=project
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
        payload = self._callable("create_a_hold_promotion_once")(
            project=project
        )
        if not isinstance(payload, Mapping):
            raise HoVerFormalControllerError("promotion capability drifted")
        return _artifact(
            kind="promotion_authorization", block="A_hold", payload=payload
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
    root = project.resolve(strict=True)
    base = legacy.default_formal_runtime_config(root)
    return replace(
        base,
        hippo_stage_root=root / FORMAL_ROOT_RELATIVE / "official_hipporag_stage",
        hippo_work_root=root / FORMAL_ROOT_RELATIVE / "hipporag_query_work",
        local_worker_cap=runner.LOCAL_CONCURRENCY_CAP,
        ner_batch_size=runner.DEFAULT_NER_BATCH_SIZE,
    )


def preflight_formal_runtime_config(
    config: FormalRuntimeConfig,
) -> Mapping[str, Any]:
    if not isinstance(config, FormalRuntimeConfig):
        raise HoVerFormalControllerError("formal runtime config type drifted")
    project = config.project.resolve(strict=True)
    if config != default_formal_runtime_config(project):
        raise HoVerFormalControllerError("formal runtime config is not canonical")
    return verify_formal_runtime_attestation_v3(
        project_root=project,
        attestation_receipt_path=config.hippo_attestation_receipt,
        base_binding_receipt_path=config.hippo_base_binding_receipt,
        runtime_python=config.hippo_runtime_python,
        local_llm_model=config.hippo_llm_model,
        local_embedding_model=config.hippo_embedding_model,
    )


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
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return hashlib.sha256(raw).hexdigest()


def consume_one_shot_marker(
    *, path: Path, prerequisites: PrerequisiteBinding
) -> dict[str, Any]:
    prerequisites.validate()
    marker = _self_hashed(
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
            "replay_retry_resample_replacement_authorized": False,
        },
        "marker_sha256",
    )
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
    marker = consume_one_shot_marker(
        path=project / output_paths.marker,
        prerequisites=prerequisites,
    )
    marker_sha256 = str(marker["marker_sha256"])
    failure_stage = "runtime_initialization"
    artifacts: dict[str, str] = {}

    def record(name: str, artifact: LifecycleArtifact, kind: str, block: str | None) -> None:
        artifact.validate(kind=kind, block=block)
        artifacts[name] = artifact.receipt_sha256

    def finish(status: str, body: Mapping[str, Any]) -> dict[str, Any]:
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
        encoder = runtime_factory.create_encoder(config)
        hippo = runtime_factory.create_hippo(config)
        ner_context = runtime_factory.create_ner_context(config)
        with ner_context as ner:
            failure_stage = "corpus_preparation"
            corpus_view = acquisition_adapter.load_corpus_view(project=project)
            prepared = core.prepare(
                corpus_view=corpus_view,
                encoder=encoder,
                ner=ner,
                hippo=hippo,
                config=config,
            )

            def execute_archive(block: str) -> tuple[object, LifecycleArtifact]:
                nonlocal failure_stage
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
            failure_stage = "A_form_late_descriptive"
            a_form_labels = acquisition_adapter.load_block_labels(
                project=project, expected_block="A_form"
            )
            descriptive_report = core.descriptive(
                stage=a_form_stage, labels=a_form_labels
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
                    "report": dict(descriptive_report),
                    "labels_opened_after_action_seal": True,
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
            if not a_hold_outcome.promoted:
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
    "preflight_formal_runtime_config",
    "run_formal_lifecycle",
    "run_synthetic_lifecycle",
    "stable_hash",
    "verify_self_hash",
    "write_json_exclusive",
]

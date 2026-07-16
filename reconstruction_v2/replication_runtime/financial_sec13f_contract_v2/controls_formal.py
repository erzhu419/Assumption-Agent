from __future__ import annotations

"""Frozen formal runtime for the post-promotion SEC-13F controls.

The control planner and aggregate report live in :mod:`controls`.  This module
owns the narrower execution boundary that must not be shared with the model
arms: an operator-only work unit starts a fresh cached task image with no
network and no model credentials, invokes the already frozen public planner
and operator, materializes the frozen tests only after the output exists, and
then runs the already cached offline verifier.

Only hash-only receipts leave this module.  In particular, the generated
answer file, expected output, raw instruction, typed plan, and verifier source
are never copied into a durable receipt.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    financial_sec13f_contract_integration_v2 as _integration,
)
from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    FinancialSec13FContractSubprocessBackendV2,
    SharedFinancialSec13FContractPlannerV2,
)
from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_MOUNT,
    SkillLearnOfflineVerifierRuntimeCache,
    offline_verifier_profile_for_family,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
    _inspect_verifier_execution_receipt,
)
from assumption_agent.events import NullEventSink
from assumption_agent.models import stable_hash

from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
    transition_durable_stage_v2,
)
from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    sha256_file,
)

from .runner import BoundContractPlannerV2


CONTROLS_FORMAL_RUNTIME_VERSION = (
    "financial_sec13f_contract_controls_formal_runtime_v1"
)
CONTROLS_EXECUTION_FREEZE_VERSION = (
    "financial_sec13f_contract_v2_controls_execution_freeze_v1"
)
OPERATOR_ONLY_EXECUTION_RECEIPT_FILENAME = (
    "operator_only.execution.receipt.json"
)
OPERATOR_ONLY_CAUSAL_RECEIPT_FILENAME = "operator_only.causal.receipt.json"
FINANCIAL_FAMILY = "financial-analysis"
CONTROL_LAUNCHER_RELATIVE_PATH = "scripts/launch_tmux_detached_controls_once.py"
BASE_LAUNCHER_RELATIVE_PATH = "scripts/launch_detached_formal_once.py"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answers_payload",
        "entity",
        "expected_answer",
        "expected_output",
        "gold",
        "gold_payload",
        "instruction",
        "output_payload",
        "query",
        "raw_answer",
        "raw_entity",
        "raw_instruction",
        "raw_plan",
        "sealed_payload",
        "trace",
        "trajectory",
    }
)
_MODEL_SECRET_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ASSUMPTION_V2_API_KEY",
    "ASSUMPTION_V2_API_BASE",
    "GPT5_API_KEY",
    "GPT5_BASE_URL",
    "RUOLI_API_KEY",
    "RUOLI_BASE_URL",
)


class ControlsFormalRuntimeError(RuntimeError):
    """A frozen control input or operator-only execution failed closed."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ControlsFormalRuntimeError(f"{label} is not a SHA-256 identity")
    return str(value)


def _reject_raw_payload(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or key.casefold() in _FORBIDDEN_PAYLOAD_KEYS:
                raise ControlsFormalRuntimeError(
                    "formal control evidence contains forbidden raw payload"
                )
            _reject_raw_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_raw_payload(nested)


def _safe_item_path(benchmark_root: Path, item_id: object) -> Path:
    if not isinstance(item_id, str) or not item_id or Path(item_id).name != item_id:
        raise ControlsFormalRuntimeError("operator-only item id is malformed")
    task_root = (benchmark_root / "tasks" / FINANCIAL_FAMILY).resolve(strict=True)
    task = task_root / item_id
    if task.is_symlink() or not task.is_dir():
        raise ControlsFormalRuntimeError("operator-only task is not regular")
    resolved = task.resolve(strict=True)
    if resolved.parent != task_root:
        raise ControlsFormalRuntimeError("operator-only task escaped its cohort")
    instruction = resolved / "instruction.md"
    tests = resolved / "tests"
    if instruction.is_symlink() or not instruction.is_file():
        raise ControlsFormalRuntimeError("operator-only instruction is not regular")
    if tests.is_symlink() or not tests.is_dir():
        raise ControlsFormalRuntimeError("operator-only tests are not regular")
    return resolved


def _completed_returncode(result: object) -> int:
    value = getattr(result, "returncode", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return 1
    return value


def _run_checked(
    delegate: Any,
    command: Sequence[str],
    *,
    label: str,
) -> Any:
    completed = delegate.run(
        [str(value) for value in command],
        check=False,
        capture_output=True,
        text=True,
    )
    if _completed_returncode(completed) != 0:
        raise ControlsFormalRuntimeError(f"{label} failed")
    return completed


def _prewarm_rows(prewarm: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _reject_raw_payload(prewarm)
    rows = prewarm.get("formal_cache_rows")
    if (
        prewarm.get("formal_execution_cache_only") is not True
        or prewarm.get("formal_image_cache_only") is not True
        or prewarm.get("formal_offline_verifier_cache_only") is not True
        or prewarm.get("formal_verifier_network") != "none"
        or prewarm.get("model_calls") != 0
        or prewarm.get("online_judge_calls") != 0
        or not isinstance(rows, list)
        or len(rows) != 8
    ):
        raise ControlsFormalRuntimeError("frozen control prewarm policy drifted")
    by_item: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ControlsFormalRuntimeError("frozen prewarm row is malformed")
        item_id = row.get("item_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in by_item
            or row.get("item_id_hash") != stable_hash({"item_id": item_id})
            or not _is_sha256(row.get("cache_key"))
            or not _is_sha256(row.get("environment_hash"))
            or not _is_sha256(row.get("source_environment_hash"))
            or not isinstance(row.get("image_id"), str)
            or not str(row["image_id"]).startswith("sha256:")
            or not _is_sha256(row.get("offline_verifier_profile_hash"))
            or not _is_sha256(row.get("offline_verifier_runtime_key"))
            or row.get("prebuilt_cache_reused") is not True
            or row.get("offline_verifier_runtime_reused") is not True
            or row.get("verifier_runtime_network") != "none"
        ):
            raise ControlsFormalRuntimeError("frozen prewarm row drifted")
        by_item[item_id] = row
    return by_item


@dataclass(frozen=True)
class OperatorOnlySharedRuntimeV1:
    """Shared immutable cache handles for eight independent clean workers."""

    project_root: Path
    benchmark_root: Path
    asset_path: Path
    prebuilt_cache: Any
    offline_verifier_cache: Any
    runner: ModuleType | Any
    prewarm: Mapping[str, Any]
    expected_program_id: str
    expected_treatment_hash: str
    expected_external_skill_source_receipt_hash: str
    docker_delegate: Any

    def __post_init__(self) -> None:
        for label, value in (
            ("expected program id", self.expected_program_id),
            ("expected treatment hash", self.expected_treatment_hash),
            (
                "expected external skill source receipt hash",
                self.expected_external_skill_source_receipt_hash,
            ),
        ):
            _require_sha256(value, label)
        project = self.project_root.expanduser().resolve(strict=True)
        benchmark = self.benchmark_root.expanduser().resolve(strict=True)
        asset = self.asset_path.expanduser().resolve(strict=True)
        try:
            benchmark.relative_to(project)
            asset.relative_to(project)
        except ValueError as exc:
            raise ControlsFormalRuntimeError(
                "operator-only frozen input escaped the project"
            ) from exc
        if not benchmark.is_dir() or not asset.is_file():
            raise ControlsFormalRuntimeError(
                "operator-only frozen project input is missing"
            )
        _prewarm_rows(self.prewarm)


def prepare_operator_only_shared_runtime_v1(
    *,
    project_root: str | Path,
    benchmark_root: str | Path,
    asset_path: str | Path,
    prewarm: Mapping[str, Any],
    expected_program_id: str,
    expected_treatment_hash: str,
    expected_external_skill_source_receipt_hash: str,
    event_sink: Any | None = None,
) -> OperatorOnlySharedRuntimeV1:
    """Open only local cache handles; do not start a model or formal worker."""

    project = Path(project_root).expanduser().resolve(strict=True)
    benchmark = Path(benchmark_root).expanduser().resolve(strict=True)
    sink = event_sink or NullEventSink()
    prebuilt = SkillLearnPrebuiltImageCache(
        benchmark,
        cache_only=True,
        event_sink=sink,
    )
    offline = SkillLearnOfflineVerifierRuntimeCache(event_sink=sink)
    loader = SkillLearnSubprocessBackend(
        benchmark,
        agent_id="codex",
        provider_mode="openai_compatible",
        record_upstream=False,
        prebuilt_cache=prebuilt,
        event_sink=sink,
    )
    runner = loader._load_runner()
    return OperatorOnlySharedRuntimeV1(
        project_root=project,
        benchmark_root=benchmark,
        asset_path=Path(asset_path),
        prebuilt_cache=prebuilt,
        offline_verifier_cache=offline,
        runner=runner,
        prewarm=dict(prewarm),
        expected_program_id=expected_program_id,
        expected_treatment_hash=expected_treatment_hash,
        expected_external_skill_source_receipt_hash=(
            expected_external_skill_source_receipt_hash
        ),
        docker_delegate=runner.subprocess,
    )


class FrozenOperatorOnlyBackendV1:
    """One clean-container, zero-model, zero-network operator control."""

    def __init__(self, work: Any, *, shared: OperatorOnlySharedRuntimeV1) -> None:
        self.work = work
        self.shared = shared

    @staticmethod
    def _transition(
        *,
        state_root: Path,
        work: Any,
        stage: str,
        payload: Mapping[str, Any],
    ) -> Any:
        from .controls import CONTROL_STAGE_ORDER_V1

        chain = load_durable_stage_chain_v2(
            state_root,
            stage_order=CONTROL_STAGE_ORDER_V1,
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request_hash,
        )
        return transition_durable_stage_v2(
            state_root,
            stage_order=CONTROL_STAGE_ORDER_V1,
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request_hash,
            stage=stage,
            predecessor_stage_hash=chain[-1].stage_hash if chain else None,
            payload=dict(payload),
        )

    def _validate_work(self, work: Any) -> None:
        target = getattr(work, "target", None)
        if (
            work is not self.work
            or getattr(work, "arm", None) != "operator_only"
            or getattr(work, "request", None) is not None
            or getattr(work, "skill_source_dir", None) is not None
            or not _is_sha256(getattr(work, "work_unit_hash", None))
            or not _is_sha256(getattr(work, "request_hash", None))
            or target is None
            or not _is_sha256(getattr(target, "typed_plan_hash", None))
            or not _is_sha256(getattr(target, "extraction_receipt_hash", None))
            or not _is_sha256(getattr(target, "candidate_output_sha256", None))
        ):
            raise ControlsFormalRuntimeError(
                "operator-only work identity drifted before execution"
            )

    def _frozen_planner(
        self,
        *,
        instruction: str,
        target: Any,
    ) -> tuple[BoundContractPlannerV2, Mapping[str, Any], Mapping[str, Any]]:
        shared_planner = SharedFinancialSec13FContractPlannerV2(
            asset_path=self.shared.asset_path
        )
        plan, extraction = shared_planner.build(instruction)
        if (
            plan.get("plan_hash") != target.typed_plan_hash
            or extraction.get("receipt_hash") != target.extraction_receipt_hash
            or plan.get("instruction_sha256")
            != hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        ):
            raise ControlsFormalRuntimeError(
                "operator-only planner differs from the frozen C plan"
            )
        bound = BoundContractPlannerV2(
            shared=shared_planner,
            instruction_sha256=str(plan["instruction_sha256"]),
            plan=plan,
            extraction_receipt=extraction,
        )
        return bound, plan, extraction

    def _production_operator_backend(
        self,
        *,
        bound: BoundContractPlannerV2,
        work: Any,
        plan: Mapping[str, Any],
        extraction: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """Bind the exact production hook without constructing an agent backend."""

        backend = object.__new__(FinancialSec13FContractSubprocessBackendV2)
        backend.planner = bound
        backend.agent_id = "operator-only-control"
        backend.model = "none"
        backend.max_steps = 0
        backend.codex_agent_execution_policy_hash = stable_hash(
            {
                "policy": CONTROLS_FORMAL_RUNTIME_VERSION,
                "model_enabled": False,
                "network": "none",
            }
        )
        backend.expected_program_id = self.shared.expected_program_id
        backend.expected_treatment_hash = self.shared.expected_treatment_hash
        backend.expected_external_skill_source_receipt_hash = (
            self.shared.expected_external_skill_source_receipt_hash
        )
        backend._contract_local = threading.local()
        backend._contract_evidence_lock = threading.Lock()
        backend._contract_runtime_evidence = []
        backend.event_sink = NullEventSink()
        state = _integration._ContractRunStateV2(
            request_hash=work.request_hash,
            plan=plan,
            extraction_receipt=extraction,
        )
        backend._contract_local.state = state
        return backend, state

    def _ensure_local_runtime(
        self,
        *,
        item_id: str,
        trace_id: str,
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        rows = _prewarm_rows(self.shared.prewarm)
        expected = rows.get(item_id)
        if expected is None:
            raise ControlsFormalRuntimeError("operator-only item lacks prewarm binding")
        image = self.shared.prebuilt_cache.ensure(
            family=FINANCIAL_FAMILY,
            item_id=item_id,
            agent_id="codex",
            runner=self.shared.runner,
            trace_id=f"{trace_id}:prebuilt",
        )
        profile = offline_verifier_profile_for_family(FINANCIAL_FAMILY)
        if profile is None:
            raise ControlsFormalRuntimeError("offline verifier profile disappeared")
        runtime = self.shared.offline_verifier_cache.ensure(
            profile=profile,
            base_image_tag=image.tag,
            base_image_id=image.image_id,
            delegate=self.shared.docker_delegate,
            trace_id=f"{trace_id}:offline-verifier",
        )
        if (
            image.cache_key != expected.get("cache_key")
            or image.environment_hash != expected.get("environment_hash")
            or image.source_environment_hash
            != expected.get("source_environment_hash")
            or image.image_id != expected.get("image_id")
            or image.reused is not True
            or profile.profile_id != expected.get("offline_verifier_profile_id")
            or profile.profile_hash != expected.get("offline_verifier_profile_hash")
            or runtime.runtime_key
            != expected.get("offline_verifier_runtime_key")
            or runtime.base_image_id != image.image_id
            or runtime.reused is not True
        ):
            raise ControlsFormalRuntimeError(
                "operator-only local runtime differs from prewarm"
            )
        return image, runtime, expected

    @staticmethod
    def _container_name(work_hash: str) -> str:
        return f"sec13f-control-op-{work_hash[:24]}"

    def _execute_operator_and_verifier(
        self,
        *,
        work: Any,
        state_root: Path,
        trace_id: str,
    ) -> Mapping[str, Any]:
        target = work.target
        item_id = target.item_id
        task = _safe_item_path(self.shared.benchmark_root, item_id)
        instruction = (task / "instruction.md").read_text(encoding="utf-8")
        bound, plan, extraction = self._frozen_planner(
            instruction=instruction,
            target=target,
        )
        image, runtime, prewarm_row = self._ensure_local_runtime(
            item_id=item_id,
            trace_id=trace_id,
        )
        backend, contract_state = self._production_operator_backend(
            bound=bound,
            work=work,
            plan=plan,
            extraction=extraction,
        )

        trial = state_root / "operator_only_trial"
        if trial.exists() or trial.is_symlink():
            raise ControlsFormalRuntimeError(
                "operator-only trial root already exists; replay is forbidden"
            )
        verifier_dir = trial / "verifier"
        verifier_dir.mkdir(parents=True)
        container = self._container_name(work.work_unit_hash)
        delegate = self.shared.docker_delegate
        started = False
        try:
            started_result = _run_checked(
                delegate,
                [
                    "docker",
                    "run",
                    "--detach",
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--name",
                    container,
                    "-v",
                    f"{trial.resolve(strict=True)}:/logs",
                    "-v",
                    f"{runtime.volume_name}:{OFFLINE_VERIFIER_MOUNT}:ro",
                    # Address the immutable prewarmed image identity directly.
                    # A mutable tag could be retargeted between cache validation
                    # and container creation.
                    image.image_id,
                    "sh",
                    "-c",
                    "trap 'exit 0' TERM INT; while :; do sleep 86400; done",
                ],
                label="operator-only clean container start",
            )
            container_id = str(getattr(started_result, "stdout", "") or "").strip()
            if not re.fullmatch(r"[0-9a-f]{12,64}", container_id.lower()):
                raise ControlsFormalRuntimeError(
                    "operator-only container id is malformed"
                )
            started = True
            network = _run_checked(
                delegate,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.HostConfig.NetworkMode}}",
                    container,
                ],
                label="operator-only network inspection",
            )
            if str(getattr(network, "stdout", "") or "").strip() != "none":
                raise ControlsFormalRuntimeError(
                    "operator-only container network is not none"
                )
            running_image = _run_checked(
                delegate,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.Image}}",
                    container,
                ],
                label="operator-only image identity inspection",
            )
            if (
                str(getattr(running_image, "stdout", "") or "").strip()
                != image.image_id
            ):
                raise ControlsFormalRuntimeError(
                    "operator-only running image differs from prewarm"
                )
            secret_check = " && ".join(
                f'test -z "${{{name}:-}}"' for name in _MODEL_SECRET_NAMES
            )
            _run_checked(
                delegate,
                ["docker", "exec", container, "sh", "-lc", secret_check],
                label="operator-only model secret absence check",
            )
            _run_checked(
                delegate,
                ["docker", "exec", container, "sh", "-lc", "test ! -e /tests"],
                label="pre-operator verifier absence check",
            )
            _run_checked(
                delegate,
                [
                    "docker",
                    "exec",
                    container,
                    "sh",
                    "-lc",
                    "test ! -e /root/answers.json",
                ],
                label="operator-only clean output absence check",
            )

            self._transition(
                state_root=state_root,
                work=work,
                stage="agent_completed",
                payload={
                    "arm": "operator_only",
                    "applicable": False,
                    "agent_enabled": False,
                    "agent_started": False,
                    "model_calls": 0,
                    "network_calls": 0,
                    "operator_started": False,
                },
            )

            # This is the production planner/operator hook used in replication C.
            # No agent command exists in this execution path.
            backend._execute_contract_plan_before_verifier_v2(
                delegate=delegate,
                container_name=container,
            )
            evidence = contract_state.runtime_evidence
            if not isinstance(evidence, Mapping):
                raise ControlsFormalRuntimeError(
                    "operator-only production evidence is missing"
                )
            query_receipt = evidence.get("query_receipt")
            if not isinstance(query_receipt, Mapping):
                raise ControlsFormalRuntimeError(
                    "operator-only query receipt is missing"
                )
            # Delay the import so this independent module can be imported while
            # controls.py is being assembled, without a circular import.
            from .controls import validate_operator_only_query_receipt_v1

            validated_query = validate_operator_only_query_receipt_v1(
                query_receipt,
                expected_plan_hash=target.typed_plan_hash,
                expected_candidate_id=bound.asset["candidate_id"],
                expected_asset_manifest_hash=bound.asset["manifest_hash"],
                expected_contract_hash=bound.asset["contract_hash"],
                expected_operator_source_sha256=bound.asset[
                    "operator_source_sha256"
                ],
            )
            observed_output_sha = validated_query["post_output_sha256"]
            output_hash_match = observed_output_sha == target.candidate_output_sha256
            if not output_hash_match:
                raise ControlsFormalRuntimeError(
                    "operator-only output differs from replication C"
                )
            if (
                evidence.get("online_calls") != 0
                or evidence.get("executed_before_verifier_materialization") is not True
                or validated_query.get("model_calls") != 0
                or validated_query.get("network_calls") != 0
                or validated_query.get("pre_output_exists") is not False
                or validated_query.get("pre_output_sha256") is not None
                or validated_query.get("verifier_content_accessed") is not False
                or validated_query.get("gold_content_accessed") is not False
                or validated_query.get("pack_content_accessed") is not False
            ):
                raise ControlsFormalRuntimeError(
                    "operator-only causal boundary drifted"
                )
            causal_receipt = atomic_write_hashed_json_v2(
                state_root / OPERATOR_ONLY_CAUSAL_RECEIPT_FILENAME,
                {
                    "runtime_version": CONTROLS_FORMAL_RUNTIME_VERSION,
                    "arm": "operator_only",
                    "work_unit_hash": work.work_unit_hash,
                    "request_hash": work.request_hash,
                    "typed_plan_hash": target.typed_plan_hash,
                    "extraction_receipt_hash": target.extraction_receipt_hash,
                    "query_receipt_hash": validated_query["receipt_hash"],
                    "operator_runtime_evidence_hash": evidence[
                        "evidence_hash"
                    ],
                    "post_output_sha256": observed_output_sha,
                    "replication_c_output_sha256": (
                        target.candidate_output_sha256
                    ),
                    "candidate_output_hash_match": True,
                    "operator_calls": 1,
                    "model_calls": 0,
                    "network_calls": 0,
                    "tests_materialized": False,
                    "persisted_before_verifier": True,
                    "answers_payload_persisted": False,
                    "raw_plan_persisted": False,
                },
                hash_field="receipt_hash",
            )
            self._transition(
                state_root=state_root,
                work=work,
                stage="operator_completed",
                payload={
                    "arm": "operator_only",
                    "applicable": True,
                    "operator_calls": 1,
                    "model_calls": 0,
                    "network_calls": 0,
                    "post_output_sha256": observed_output_sha,
                    "candidate_output_hash_match": True,
                    "causal_receipt_hash": causal_receipt["receipt_hash"],
                    "persisted_before_verifier": True,
                    "tests_materialized": False,
                },
            )
            _run_checked(
                delegate,
                ["docker", "exec", container, "sh", "-lc", "test ! -e /tests"],
                label="post-operator verifier absence check",
            )
            _run_checked(
                delegate,
                ["docker", "exec", container, "mkdir", "-p", "/tests"],
                label="operator-only verifier directory creation",
            )
            _run_checked(
                delegate,
                ["docker", "cp", f"{task / 'tests'}/.", f"{container}:/tests"],
                label="operator-only post-output verifier materialization",
            )
            verifier = _run_checked(
                delegate,
                [
                    "docker",
                    "exec",
                    container,
                    "sh",
                    "-lc",
                    runtime.profile.verifier_command,
                ],
                label="operator-only frozen offline verifier",
            )
            reward_path = verifier_dir / "reward.txt"
            ctrf_path = verifier_dir / "ctrf.json"
            if (
                reward_path.is_symlink()
                or not reward_path.is_file()
                or ctrf_path.is_symlink()
                or not ctrf_path.is_file()
            ):
                raise ControlsFormalRuntimeError(
                    "operator-only offline verifier artifacts are incomplete"
                )
            reward_text = reward_path.read_text(encoding="utf-8").strip()
            if reward_text not in {"0", "1"}:
                raise ControlsFormalRuntimeError(
                    "operator-only offline reward is malformed"
                )
            verifier_receipt = _inspect_verifier_execution_receipt(
                test_script=task / "tests" / "test.sh",
                verifier_dir=verifier_dir,
                result={
                    "verifier_exit": _completed_returncode(verifier),
                    "reward": int(reward_text),
                },
                offline_verifier_profile=runtime.profile,
            )
            if not verifier_receipt.valid or verifier_receipt.test_count <= 0:
                raise ControlsFormalRuntimeError(
                    "operator-only verifier execution receipt is invalid"
                )
            network_after = _run_checked(
                delegate,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.HostConfig.NetworkMode}}",
                    container,
                ],
                label="post-verifier network inspection",
            )
            if str(getattr(network_after, "stdout", "") or "").strip() != "none":
                raise ControlsFormalRuntimeError(
                    "operator-only container network changed"
                )
            self._transition(
                state_root=state_root,
                work=work,
                stage="verifier_completed",
                payload={
                    "arm": "operator_only",
                    "offline": True,
                    "offline_verifier_calls": 1,
                    "online_judge_calls": 0,
                    "operator_calls_before_verifier": 1,
                    "candidate_output_hash_match": True,
                    "verifier_execution_receipt_hash": (
                        verifier_receipt.receipt_hash
                    ),
                    "verifier_test_count": verifier_receipt.test_count,
                    "reward": int(reward_text),
                },
            )
            receipt = {
                "runtime_version": CONTROLS_FORMAL_RUNTIME_VERSION,
                "arm": "operator_only",
                "work_unit_hash": work.work_unit_hash,
                "request_hash": work.request_hash,
                "item_id_hash": stable_hash({"item_id": item_id}),
                "fold_id_hash": stable_hash({"fold_id": target.fold_id}),
                "typed_plan_hash": target.typed_plan_hash,
                "extraction_receipt_hash": target.extraction_receipt_hash,
                "query_receipt_hash": validated_query["receipt_hash"],
                "operator_runtime_evidence_hash": evidence["evidence_hash"],
                "post_output_sha256": observed_output_sha,
                "replication_c_output_sha256": target.candidate_output_sha256,
                "candidate_output_hash_match": True,
                "prebuilt_cache_key": image.cache_key,
                "prebuilt_image_id": image.image_id,
                "prebuilt_cache_reused": True,
                "offline_verifier_profile_id": runtime.profile.profile_id,
                "offline_verifier_profile_hash": runtime.profile.profile_hash,
                "offline_verifier_runtime_key": runtime.runtime_key,
                "offline_verifier_runtime_reused": True,
                "prewarm_row_hash": payload_hash(dict(prewarm_row)),
                "container_network_before_operator": "none",
                "container_network_after_verifier": "none",
                "model_secret_env_present": False,
                "agent_started": False,
                "model_calls": 0,
                "network_calls": 0,
                "operator_calls": 1,
                "offline_verifier_calls": 1,
                "online_judge_calls": 0,
                "tests_present_before_operator": False,
                "tests_materialized_after_operator": True,
                "original_online_test_script_executed": False,
                "verifier_command_hash": stable_hash(
                    {"command": runtime.profile.verifier_command}
                ),
                "verifier_process_returncode": _completed_returncode(verifier),
                "verifier_execution_receipt_hash": (
                    verifier_receipt.receipt_hash
                ),
                "verifier_test_count": verifier_receipt.test_count,
                "reward": int(reward_text),
                "ctrf_file_sha256": sha256_file(ctrf_path),
                "answers_payload_persisted": False,
                "expected_output_persisted": False,
                "raw_instruction_persisted": False,
                "raw_plan_persisted": False,
                "gold_content_persisted": False,
                "sealed_content_accessed": False,
                "retry_count": 0,
                "replay_authorized": False,
            }
            _reject_raw_payload(receipt)
            return receipt
        finally:
            backend._contract_local.state = None
            if started:
                removed = delegate.run(
                    ["docker", "rm", "--force", container],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if _completed_returncode(removed) != 0:
                    raise ControlsFormalRuntimeError(
                        "operator-only clean container removal failed"
                    )

    def run_control(
        self,
        *,
        work: Any,
        state_root: Path,
        trace_id: str,
    ) -> Any:
        self._validate_work(work)
        state = Path(state_root).expanduser().resolve(strict=True)
        receipt_body = self._execute_operator_and_verifier(
            work=work,
            state_root=state,
            trace_id=trace_id,
        )
        receipt = atomic_write_hashed_json_v2(
            state / OPERATOR_ONLY_EXECUTION_RECEIPT_FILENAME,
            receipt_body,
            hash_field="receipt_hash",
        )
        observation_body = {
            "controls_formal_runtime_version": CONTROLS_FORMAL_RUNTIME_VERSION,
            "arm": "operator_only",
            "work_unit_hash": work.work_unit_hash,
            "request_hash": work.request_hash,
            "execution_receipt_hash": receipt["receipt_hash"],
            "valid": True,
            "success": bool(receipt["reward"] == 1),
            "score": float(receipt["reward"]),
            "model_calls": 0,
            "operator_calls": 1,
            "online_judge_calls": 0,
            "output_sha256": receipt["post_output_sha256"],
            "candidate_output_hash_match": True,
        }
        self._transition(
            state_root=state,
            work=work,
            stage="observation_finalized",
            payload={
                "arm": "operator_only",
                "observation_hash": stable_hash(observation_body),
                "observation_receipt_hash": receipt["receipt_hash"],
                "valid": True,
                "success": observation_body["success"],
                "score": observation_body["score"],
                "model_calls": 0,
                "operator_calls": 1,
                "online_judge_calls": 0,
                "candidate_output_hash_match": True,
            },
        )
        from .controls import ControlBackendResultV1

        return ControlBackendResultV1(
            arm="operator_only",
            work_unit_hash=work.work_unit_hash,
            request_hash=work.request_hash,
            observation_hash=stable_hash(observation_body),
            valid=True,
            success=observation_body["success"],
            score=observation_body["score"],
            model_calls=0,
            operator_calls=1,
            online_judge_calls=0,
            output_sha256=receipt["post_output_sha256"],
            candidate_output_hash_match=True,
        )


def operator_only_backend_factory_v1(
    shared: OperatorOnlySharedRuntimeV1,
) -> Callable[[Any], FrozenOperatorOnlyBackendV1]:
    return lambda work: FrozenOperatorOnlyBackendV1(work, shared=shared)


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ControlsFormalRuntimeError(f"{label} is not a regular file")
    try:
        value = json.loads(unresolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlsFormalRuntimeError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ControlsFormalRuntimeError(f"{label} is not a JSON object")
    _reject_raw_payload(value)
    return value


def _project_file_binding(
    project: Path,
    path: Path,
    *,
    label: str,
    committed: bool,
) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ControlsFormalRuntimeError(f"{label} is not a regular file")
    resolved = unresolved.resolve(strict=True)
    try:
        relative = resolved.relative_to(project).as_posix()
    except ValueError as exc:
        raise ControlsFormalRuntimeError(f"{label} escaped the project") from exc
    binding: dict[str, Any] = {
        "relative_path": relative,
        "file_sha256": sha256_file(resolved),
    }
    if not committed:
        return binding
    completed = subprocess.run(
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
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "log",
            "-1",
            "--format=%H",
            "--",
            relative,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if completed.stdout.strip() or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ControlsFormalRuntimeError(f"{label} is not committed and clean")
    # ``project`` may itself be a nested directory inside the Git worktree
    # (as reconstruction_v2 is in this repository).  Reuse the freeze module's
    # worktree-relative conversion rather than treating a project-relative
    # path as a repository-root path.
    from .freeze import _git_blob

    blob = _git_blob(project, commit, relative)
    if hashlib.sha256(blob).hexdigest() != binding["file_sha256"]:
        raise ControlsFormalRuntimeError(f"{label} Git blob drifted")
    binding["committed_at_git_commit"] = commit
    return binding


def _self_hash(value: Mapping[str, Any], *, field: str, label: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or payload_hash(body) != declared:
        raise ControlsFormalRuntimeError(f"{label} self hash drifted")
    return str(declared)


def _control_work_rows(plan: Any) -> list[dict[str, Any]]:
    works = getattr(plan, "work_units", None)
    if not isinstance(works, (tuple, list)) or len(works) != 16:
        raise ControlsFormalRuntimeError(
            "controls execution freeze requires exactly sixteen work units"
        )
    rows: list[dict[str, Any]] = []
    for work in works:
        target = getattr(work, "target", None)
        arm = getattr(work, "arm", None)
        request = getattr(work, "request", None)
        if arm not in {"skill_only", "operator_only"} or target is None:
            raise ControlsFormalRuntimeError("control work arm is malformed")
        work_hash = _require_sha256(
            getattr(work, "work_unit_hash", None), "control work unit hash"
        )
        request_hash = _require_sha256(
            getattr(work, "request_hash", None), "control request hash"
        )
        item_id = getattr(target, "item_id", None)
        fold_id = getattr(target, "fold_id", None)
        if not isinstance(item_id, str) or not isinstance(fold_id, str):
            raise ControlsFormalRuntimeError("control target identity is malformed")
        if arm == "skill_only":
            if request is None or getattr(work, "skill_source_dir", None) is None:
                raise ControlsFormalRuntimeError(
                    "skill-only work lost its frozen model request"
                )
            if getattr(request, "request_hash", None) != request_hash:
                raise ControlsFormalRuntimeError(
                    "skill-only request hash differs from work unit"
                )
        elif request is not None or getattr(work, "skill_source_dir", None) is not None:
            raise ControlsFormalRuntimeError(
                "operator-only work unexpectedly contains a model request"
            )
        row = {
            "arm": arm,
            "work_unit_hash": work_hash,
            "request_hash": request_hash,
            "item_id_hash": stable_hash({"item_id": item_id}),
            "fold_id_hash": stable_hash({"fold_id": fold_id}),
            "prior_pair_id_hash": stable_hash(
                {"pair_id": getattr(target, "prior_pair_id", "")}
            ),
            "typed_plan_hash": _require_sha256(
                getattr(target, "typed_plan_hash", None), "typed plan hash"
            ),
            "extraction_receipt_hash": _require_sha256(
                getattr(target, "extraction_receipt_hash", None),
                "extraction receipt hash",
            ),
            "replication_c_candidate_output_sha256": _require_sha256(
                getattr(target, "candidate_output_sha256", None),
                "replication C candidate output SHA",
            ),
            "model_call_authorization_count": 1 if arm == "skill_only" else 0,
            "operator_call_authorization_count": (
                1 if arm == "operator_only" else 0
            ),
            "offline_verifier_authorization_count": 1,
            "retry_count": 0,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["item_id_hash"], row["arm"]))
    if (
        sum(row["arm"] == "skill_only" for row in rows) != 8
        or sum(row["arm"] == "operator_only" for row in rows) != 8
        or len({row["work_unit_hash"] for row in rows}) != 16
        or len({row["request_hash"] for row in rows}) != 16
        or len({row["item_id_hash"] for row in rows}) != 8
    ):
        raise ControlsFormalRuntimeError("frozen control work set drifted")
    return rows


def _validate_provider_binding(value: Mapping[str, Any]) -> None:
    _reject_raw_payload(value)
    if (
        value.get("provider_label") != "plus"
        or value.get("model") != "gpt-5.4-mini"
        or value.get("api_origin") != "https://ruoli.dev"
        or value.get("pro_fallback_authorized") is not False
        or value.get("secret_value_persisted") is not False
        or not _is_sha256(value.get("provider_binding_hash_from_replication_c"))
        or not _is_sha256(
            value.get("inherit_replication_c_execution_policy_hash")
        )
    ):
        raise ControlsFormalRuntimeError("controls provider binding drifted")


def _validate_reuse_receipt(value: Mapping[str, Any]) -> str:
    _reject_raw_payload(value)
    declared = _self_hash(
        value,
        field="receipt_hash",
        label="prior measurement reuse receipt",
    )
    if (
        value.get("reused_observation_count") != 16
        or value.get("executions_performed") != 0
        or value.get("model_calls_performed") != 0
        or value.get("operator_calls_performed") not in {None, 0}
        or value.get("offline_verifier_calls_performed") not in {None, 0}
        or value.get("retry_count") not in {None, 0}
        or value.get("replay_authorized") not in {None, False}
    ):
        raise ControlsFormalRuntimeError(
            "prior measurement reuse receipt authorizes reexecution"
        )
    return declared


def build_controls_execution_freeze_v1(
    *,
    project_root: str | Path,
    plan: Any,
    preregistration_path: str | Path,
    replication_c_execution_freeze_path: str | Path,
    prewarm_path: str | Path,
    reuse_receipt_path: str | Path,
    source_closure: Mapping[str, Any],
    study_id: str = "financial-sec13f-contract-v2-post-promotion-controls-20260716",
) -> dict[str, Any]:
    """Build the finite hash-only execution freeze after source commit.

    This function deliberately has no credential argument and performs no
    model, operator, verifier, pack, gold, sealed, or trace access.
    """

    from .freeze import validate_execution_source_closure_v2

    project = Path(project_root).expanduser().resolve(strict=True)
    prereg_path = Path(preregistration_path)
    c_freeze_path = Path(replication_c_execution_freeze_path)
    warm_path = Path(prewarm_path)
    reuse_path = Path(reuse_receipt_path)
    prereg = _read_json_mapping(prereg_path, "controls preregistration")
    c_freeze = _read_json_mapping(c_freeze_path, "replication C execution freeze")
    prewarm = _read_json_mapping(warm_path, "controls prewarm")
    reuse = _read_json_mapping(reuse_path, "prior measurement reuse receipt")
    prereg_hash = _self_hash(
        prereg, field="manifest_hash", label="controls preregistration"
    )
    c_freeze_hash = _self_hash(
        c_freeze, field="manifest_hash", label="replication C execution freeze"
    )
    prewarm_hash = _self_hash(prewarm, field="prewarm_hash", label="prewarm")
    reuse_hash = _validate_reuse_receipt(reuse)
    closure_hash = validate_execution_source_closure_v2(
        source_closure,
        project_root=project,
    )
    if prereg.get("manifest_version") != (
        "financial_sec13f_contract_v2_controls_preregistration_v1"
    ):
        raise ControlsFormalRuntimeError("controls preregistration version drifted")
    prereg_freeze = prereg.get("freeze_bindings")
    if not isinstance(prereg_freeze, Mapping):
        raise ControlsFormalRuntimeError("controls preregistration bindings are missing")
    frozen_c = prereg_freeze.get("replication_c_execution_freeze")
    provider = prereg_freeze.get("provider")
    candidate = prereg_freeze.get("candidate")
    cohort = prereg_freeze.get("cohort")
    measurement = prereg_freeze.get("measurement_report")
    if not all(
        isinstance(value, Mapping)
        for value in (frozen_c, provider, candidate, cohort, measurement)
    ):
        raise ControlsFormalRuntimeError(
            "controls preregistration closure is incomplete"
        )
    assert isinstance(frozen_c, Mapping)
    assert isinstance(provider, Mapping)
    if (
        frozen_c.get("freeze_hash") != c_freeze_hash
        or frozen_c.get("file_sha256") != sha256_file(c_freeze_path)
        or c_freeze.get("prewarm", {}).get("prewarm_hash") != prewarm_hash
        or c_freeze.get("prewarm", {}).get("file_sha256")
        != sha256_file(warm_path)
    ):
        raise ControlsFormalRuntimeError(
            "controls inherited replication C runtime binding drifted"
        )
    _validate_provider_binding(provider)
    prewarm_rows = _prewarm_rows(prewarm)
    work_rows = _control_work_rows(plan)
    if {row["item_id_hash"] for row in work_rows} != {
        str(row["item_id_hash"]) for row in prewarm_rows.values()
    }:
        raise ControlsFormalRuntimeError(
            "control work items differ from the frozen prewarm cohort"
        )
    plan_hash = getattr(plan, "plan_hash", None)
    if not _is_sha256(plan_hash):
        safe = getattr(plan, "safe_payload", None)
        safe_payload = safe() if callable(safe) else None
        if not isinstance(safe_payload, Mapping):
            raise ControlsFormalRuntimeError("control plan has no safe identity")
        _reject_raw_payload(safe_payload)
        plan_hash = payload_hash(dict(safe_payload))
    skill_only_treatment_hashes = {
        getattr(work, "treatment_hash", None)
        for work in plan.work_units
        if getattr(work, "arm", None) == "skill_only"
    }
    operator_only_treatment_hashes = {
        getattr(work, "treatment_hash", None)
        for work in plan.work_units
        if getattr(work, "arm", None) == "operator_only"
    }
    if (
        len(skill_only_treatment_hashes) != 1
        or len(operator_only_treatment_hashes) != 1
    ):
        raise ControlsFormalRuntimeError("control treatment identities drifted")
    runtime_identity = {
        "expected_program_id": _require_sha256(
            getattr(plan, "candidate_recipe_id", None),
            "operator-only expected program id",
        ),
        "expected_program_set_hash": _require_sha256(
            getattr(plan, "candidate_program_set_hash", None),
            "controls expected program set hash",
        ),
        "skill_only_treatment_hash": _require_sha256(
            next(iter(skill_only_treatment_hashes)),
            "skill-only treatment hash",
        ),
        "operator_only_treatment_hash": _require_sha256(
            next(iter(operator_only_treatment_hashes)),
            "operator-only treatment hash",
        ),
        "external_skill_source_receipt_hash": _require_sha256(
            getattr(plan, "external_skill_source_receipt_hash", None),
            "external skill source receipt hash",
        ),
    }
    launcher_rows = [
        {
            "role": role,
            **_project_file_binding(
                project,
                project / relative,
                label=f"{role} launcher source",
                committed=True,
            ),
        }
        for role, relative in (
            ("controls_launcher", CONTROL_LAUNCHER_RELATIVE_PATH),
            ("base_launcher", BASE_LAUNCHER_RELATIVE_PATH),
        )
    ]
    body: dict[str, Any] = {
        "manifest_version": CONTROLS_EXECUTION_FREEZE_VERSION,
        "study_id": study_id,
        "controls_formal_runtime_version": CONTROLS_FORMAL_RUNTIME_VERSION,
        "preregistration": {
            **_project_file_binding(
                project,
                prereg_path,
                label="controls preregistration",
                committed=True,
            ),
            "manifest_hash": prereg_hash,
        },
        "replication_c_execution_freeze": {
            **_project_file_binding(
                project,
                c_freeze_path,
                label="replication C execution freeze",
                committed=True,
            ),
            "freeze_hash": c_freeze_hash,
        },
        "execution_source_closure": dict(source_closure),
        "launcher_source_closure": {
            "files": launcher_rows,
            "file_count": 2,
            "file_set_hash": payload_hash(launcher_rows),
        },
        "plan": {
            "plan_hash": str(plan_hash),
            "work_unit_count": 16,
            "work_units": work_rows,
            "work_unit_set_hash": payload_hash(work_rows),
            "raw_content_persisted": False,
        },
        "runtime_identity": runtime_identity,
        "candidate": dict(candidate),
        "cohort": dict(cohort),
        "provider": dict(provider),
        "prewarm": {
            **_project_file_binding(
                project,
                warm_path,
                label="controls prewarm",
                committed=False,
            ),
            "prewarm_hash": prewarm_hash,
            "formal_cache_row_set_hash": prewarm[
                "formal_cache_row_set_hash"
            ],
            "formal_cache_rows": [dict(row) for row in prewarm_rows.values()],
            "formal_execution_cache_only": True,
            "formal_verifier_network": "none",
        },
        "prior_measurement_reuse": {
            **_project_file_binding(
                project,
                reuse_path,
                label="prior measurement reuse receipt",
                committed=False,
            ),
            "receipt_hash": reuse_hash,
            "measurement_report_hash": measurement.get("report_hash"),
            "reused_observation_count": 16,
            "reused_raw_record_count": 8,
            "reused_full_record_count": 8,
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
            "expected_output_content_accessed": False,
            "gold_artifact_content_accessed": False,
            "private_pack_content_accessed": False,
            "sealed_content_accessed": False,
            "trajectory_or_trace_content_accessed": False,
            "model_calls_to_build_freeze": 0,
            "network_calls_to_build_freeze": 0,
            "online_judge_calls_to_build_freeze": 0,
            "secret_value_persisted": False,
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
    _reject_raw_payload(body)
    manifest = {**body, "manifest_hash": payload_hash(body)}
    validate_controls_execution_freeze_v1(
        manifest,
        expected_plan=plan,
        project_root=project,
        validate_live_files=True,
    )
    return manifest


def validate_controls_execution_freeze_v1(
    value: Mapping[str, Any],
    *,
    expected_plan: Any | None = None,
    project_root: str | Path | None = None,
    validate_live_files: bool = False,
) -> str:
    """Validate the finite control execution contract without opening outcomes."""

    _reject_raw_payload(value)
    declared = _self_hash(
        value,
        field="manifest_hash",
        label="controls execution freeze",
    )
    plan = value.get("plan")
    execution = value.get("execution")
    boundary = value.get("evidence_boundary")
    analysis = value.get("analysis_policy")
    provider = value.get("provider")
    prewarm = value.get("prewarm")
    reuse = value.get("prior_measurement_reuse")
    closure = value.get("execution_source_closure")
    launcher_closure = value.get("launcher_source_closure")
    runtime_identity = value.get("runtime_identity")
    if not all(
        isinstance(row, Mapping)
        for row in (
            plan,
            execution,
            boundary,
            analysis,
            provider,
            prewarm,
            reuse,
            closure,
            launcher_closure,
            runtime_identity,
        )
    ):
        raise ControlsFormalRuntimeError("controls execution freeze is incomplete")
    assert isinstance(plan, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(boundary, Mapping)
    assert isinstance(analysis, Mapping)
    assert isinstance(provider, Mapping)
    assert isinstance(prewarm, Mapping)
    assert isinstance(reuse, Mapping)
    assert isinstance(closure, Mapping)
    assert isinstance(launcher_closure, Mapping)
    assert isinstance(runtime_identity, Mapping)
    launcher_rows = launcher_closure.get("files")
    if (
        not isinstance(launcher_rows, list)
        or len(launcher_rows) != 2
        or launcher_closure.get("file_count") != 2
        or launcher_closure.get("file_set_hash") != payload_hash(launcher_rows)
        or {row.get("role") for row in launcher_rows if isinstance(row, Mapping)}
        != {"controls_launcher", "base_launcher"}
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "role",
                "relative_path",
                "file_sha256",
                "committed_at_git_commit",
            }
            or not isinstance(row.get("relative_path"), str)
            or Path(str(row.get("relative_path"))).is_absolute()
            or ".." in Path(str(row.get("relative_path"))).parts
            or not _is_sha256(row.get("file_sha256"))
            or re.fullmatch(
                r"[0-9a-f]{40}",
                str(row.get("committed_at_git_commit") or ""),
            )
            is None
            for row in launcher_rows
        )
    ):
        raise ControlsFormalRuntimeError("controls launcher source closure drifted")
    rows = plan.get("work_units")
    if (
        value.get("manifest_version") != CONTROLS_EXECUTION_FREEZE_VERSION
        or value.get("controls_formal_runtime_version")
        != CONTROLS_FORMAL_RUNTIME_VERSION
        or not isinstance(rows, list)
        or len(rows) != 16
        or plan.get("work_unit_count") != 16
        or plan.get("work_unit_set_hash") != payload_hash(rows)
        or plan.get("raw_content_persisted") is not False
        or sum(row.get("arm") == "skill_only" for row in rows) != 8
        or sum(row.get("arm") == "operator_only" for row in rows) != 8
        or len({row.get("work_unit_hash") for row in rows}) != 16
        or len({row.get("request_hash") for row in rows}) != 16
        or len({row.get("item_id_hash") for row in rows}) != 8
        or any(
            not isinstance(row, Mapping)
            or not _is_sha256(row.get("work_unit_hash"))
            or not _is_sha256(row.get("request_hash"))
            or not _is_sha256(row.get("item_id_hash"))
            or not _is_sha256(row.get("typed_plan_hash"))
            or not _is_sha256(row.get("extraction_receipt_hash"))
            or not _is_sha256(
                row.get("replication_c_candidate_output_sha256")
            )
            or row.get("model_call_authorization_count")
            != (1 if row.get("arm") == "skill_only" else 0)
            or row.get("operator_call_authorization_count")
            != (1 if row.get("arm") == "operator_only" else 0)
            or row.get("offline_verifier_authorization_count") != 1
            or row.get("retry_count") != 0
            or row.get("replay_authorized") is not False
            for row in rows
        )
    ):
        raise ControlsFormalRuntimeError("frozen control work set is malformed")
    runtime_identity_fields = (
        "expected_program_id",
        "expected_program_set_hash",
        "skill_only_treatment_hash",
        "operator_only_treatment_hash",
        "external_skill_source_receipt_hash",
    )
    if set(runtime_identity) != set(runtime_identity_fields):
        raise ControlsFormalRuntimeError("controls runtime identity fields drifted")
    for field in runtime_identity_fields:
        _require_sha256(runtime_identity.get(field), f"runtime identity {field}")
    if (
        execution.get("physical_work_units") != 16
        or execution.get("skill_only_work_units") != 8
        or execution.get("operator_only_work_units") != 8
        or execution.get("maximum_concurrent_work_units") != 16
        or execution.get("maximum_concurrent_model_calls") != 8
        or execution.get("new_model_calls") != 8
        or execution.get("operator_calls") != 8
        or execution.get("offline_verifier_calls") != 16
        or execution.get("all_futures_submitted_before_results_read") is not True
        or execution.get("offline_evaluation_only") is not True
        or execution.get("online_judge_calls") != 0
        or execution.get("provider_retry_authorized") is not False
        or execution.get("model_replay_authorized") is not False
        or execution.get("operator_replay_authorized") is not False
        or execution.get("verifier_replay_authorized") is not False
        or execution.get("invalid_item_replacement_authorized") is not False
        or execution.get("resampling_authorized") is not False
        or execution.get("retry_count") != 0
        or any(value is not False for value in boundary.values() if isinstance(value, bool))
        or any(
            analysis.get(field) is not False
            for field in (
                "performance_gate_bound",
                "numeric_performance_threshold_bound",
                "promotion_gate_reopened",
                "candidate_mutation_authorized",
                "sealed_evaluation_authorized",
            )
        )
        or analysis.get("controls_are_mechanism_characterization") is not True
        or reuse.get("reused_observation_count") != 16
        or reuse.get("executions_performed") != 0
        or reuse.get("model_calls_performed") != 0
        or reuse.get("operator_calls_performed") != 0
        or reuse.get("offline_verifier_calls_performed") != 0
        or reuse.get("completed_arm_reexecution_authorized") is not False
    ):
        raise ControlsFormalRuntimeError("controls execution policy drifted")
    _validate_provider_binding(provider)
    prewarm_rows = _prewarm_rows(
        {
            "formal_cache_rows": prewarm.get("formal_cache_rows"),
            "formal_execution_cache_only": prewarm.get(
                "formal_execution_cache_only"
            ),
            "formal_image_cache_only": True,
            "formal_offline_verifier_cache_only": True,
            "formal_verifier_network": prewarm.get("formal_verifier_network"),
            "model_calls": 0,
            "online_judge_calls": 0,
        }
    )
    if {row.get("item_id_hash") for row in rows} != {
        row.get("item_id_hash") for row in prewarm_rows.values()
    }:
        raise ControlsFormalRuntimeError("freeze prewarm cohort drifted")
    if expected_plan is not None:
        expected_runtime_identity = {
            "expected_program_id": getattr(
                expected_plan, "candidate_recipe_id", None
            ),
            "expected_program_set_hash": getattr(
                expected_plan, "candidate_program_set_hash", None
            ),
            "skill_only_treatment_hash": getattr(
                expected_plan, "skill_only_treatment_hash", None
            ),
            "operator_only_treatment_hash": getattr(
                expected_plan, "operator_only_treatment_hash", None
            ),
            "external_skill_source_receipt_hash": getattr(
                expected_plan, "external_skill_source_receipt_hash", None
            ),
        }
        if (
            rows != _control_work_rows(expected_plan)
            or plan.get("plan_hash") != getattr(expected_plan, "plan_hash", None)
            or dict(runtime_identity) != expected_runtime_identity
        ):
            raise ControlsFormalRuntimeError("live control plan differs from freeze")
    if validate_live_files:
        if project_root is None:
            raise ControlsFormalRuntimeError(
                "live file validation requires a project root"
            )
        from .freeze import validate_execution_source_closure_v2

        project = Path(project_root).expanduser().resolve(strict=True)
        validate_execution_source_closure_v2(closure, project_root=project)
        for label, committed in (
            ("preregistration", True),
            ("replication_c_execution_freeze", True),
            ("prewarm", False),
            ("prior_measurement_reuse", False),
        ):
            binding = value.get(label)
            if not isinstance(binding, Mapping):
                raise ControlsFormalRuntimeError(f"{label} binding is missing")
            relative = binding.get("relative_path")
            expected_sha = binding.get("file_sha256")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not _is_sha256(expected_sha)
            ):
                raise ControlsFormalRuntimeError(f"{label} binding is malformed")
            path = project / relative
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
                raise ControlsFormalRuntimeError(f"{label} live file drifted")
            if committed and not re.fullmatch(
                r"[0-9a-f]{40}", str(binding.get("committed_at_git_commit") or "")
            ):
                raise ControlsFormalRuntimeError(f"{label} commit is malformed")
        for row in launcher_rows:
            assert isinstance(row, Mapping)
            relative = str(row["relative_path"])
            path = project / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != row["file_sha256"]
            ):
                raise ControlsFormalRuntimeError(
                    "controls launcher live source drifted"
                )
    return declared


def write_controls_execution_freeze_v1(
    output_path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write one new freeze file; replacement is intentionally forbidden."""

    validate_controls_execution_freeze_v1(payload)
    output = Path(output_path).expanduser()
    if output.exists() or output.is_symlink():
        raise ControlsFormalRuntimeError(
            "controls execution freeze already exists; overwrite is forbidden"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ControlsFormalRuntimeError("controls freeze temporary path exists")
    rendered = json.dumps(
        dict(payload), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def add_controls_formal_cli_arguments_v1(parser: argparse.ArgumentParser) -> None:
    """Add freeze-only arguments to the controls CLI without owning its main."""

    parser.add_argument("--controls-execution-freeze-output", type=Path)
    parser.add_argument("--controls-preregistration", type=Path)
    parser.add_argument("--replication-c-execution-freeze", type=Path)
    parser.add_argument("--controls-prewarm", type=Path)
    parser.add_argument("--controls-reuse-receipt", type=Path)


def build_controls_execution_freeze_from_cli_v1(
    args: argparse.Namespace,
    *,
    project_root: str | Path,
    plan: Any,
) -> dict[str, Any] | None:
    """CLI adapter callable by ``controls.main`` after plan/reuse formation."""

    output = getattr(args, "controls_execution_freeze_output", None)
    if output is None:
        return None
    required = {
        "controls_preregistration": getattr(args, "controls_preregistration", None),
        "replication_c_execution_freeze": getattr(
            args, "replication_c_execution_freeze", None
        ),
        "controls_prewarm": getattr(args, "controls_prewarm", None),
        "controls_reuse_receipt": getattr(args, "controls_reuse_receipt", None),
    }
    if any(value is None for value in required.values()):
        raise ControlsFormalRuntimeError(
            "controls freeze CLI inputs are incomplete"
        )
    from .freeze import build_execution_source_closure_v2

    closure = build_execution_source_closure_v2(project_root)
    manifest = build_controls_execution_freeze_v1(
        project_root=project_root,
        plan=plan,
        preregistration_path=required["controls_preregistration"],
        replication_c_execution_freeze_path=required[
            "replication_c_execution_freeze"
        ],
        prewarm_path=required["controls_prewarm"],
        reuse_receipt_path=required["controls_reuse_receipt"],
        source_closure=closure,
    )
    write_controls_execution_freeze_v1(output, manifest)
    return manifest


__all__ = [
    "CONTROLS_EXECUTION_FREEZE_VERSION",
    "CONTROLS_FORMAL_RUNTIME_VERSION",
    "OPERATOR_ONLY_EXECUTION_RECEIPT_FILENAME",
    "ControlsFormalRuntimeError",
    "FrozenOperatorOnlyBackendV1",
    "OperatorOnlySharedRuntimeV1",
    "add_controls_formal_cli_arguments_v1",
    "build_controls_execution_freeze_from_cli_v1",
    "build_controls_execution_freeze_v1",
    "operator_only_backend_factory_v1",
    "prepare_operator_only_shared_runtime_v1",
    "validate_controls_execution_freeze_v1",
    "write_controls_execution_freeze_v1",
]

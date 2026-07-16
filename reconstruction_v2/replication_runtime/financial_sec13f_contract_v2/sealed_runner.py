from __future__ import annotations

"""Execute the frozen four-pair Replication-C sealed test exactly once."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence

from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    SharedFinancialSec13FContractPlannerV2,
)
from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
    _configure_environment,
)
from assumption_agent.benchmarks.offline_verifier import (
    SkillLearnOfflineVerifierRuntimeCache,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnModelInferenceLimiter,
    SkillLearnProviderCircuit,
    SkillLearnTrialObservation,
)
from assumption_agent.events import JsonlEventSink
from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
    read_hashed_json_v2,
)
from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    read_json,
)
from replication_runtime.financial_semantic_v2.recovery import (
    MODEL_EXECUTION_CLAIM_FILENAME,
    SEMANTIC_EVIDENCE_FILENAME,
)
from replication_runtime.financial_semantic_v2 import runner as _legacy
from replication_runtime.financial_semantic_v2.plan import FixedPeriodOutTreatmentV2

from .backends import (
    DurableFinancialSec13FContractBackendV2,
    DurableRawSubprocessBackendV2,
    backend_runtime_identity_v2,
    future_terminal_semantics_v2,
)
from .provider import (
    load_provider_environment_v1,
    validate_execution_provider_binding_v1,
)
from .runner import (
    BoundContractPlannerV2,
    ContractRecoveryBoundBackendV2,
    _assert_no_raw_contract_payload_v2,
)
from .sealed_freeze import validate_sealed_execution_freeze_v1
from .sealed_materialize import (
    MATERIALIZATION_REPORT_NAME,
    sealed_benchmark_tree_receipt_v1,
)
from .sealed_plan import (
    SealedTargetV1,
    build_sealed_plan_v1,
    execute_sealed_plan_v1,
)
from .sealed_prepare import verify_sealed_payload_v1
from .sealed_prewarm import PREWARM_VERSION
from .treatment import FixedContractCandidateV2, load_fixed_contract_candidate_v2


RUNNER_VERSION = "financial_sec13f_replication_c_sealed_runner_v1"
REPORT_FILENAME = "sealed.report.json"
FAILURE_FILENAME = "sealed.failure.json"
EVENTS_FILENAME = "sealed.events.jsonl"
NETWORK_ISOLATION_FILENAME = "sealed.verifier_network_isolation.json"
NETWORK_ISOLATION_VERSION = (
    "financial_sec13f_replication_c_sealed_verifier_network_isolation_v1"
)


class SealedRunnerError(RuntimeError):
    """The sealed formal execution failed closed."""


class _SealedVerifierIsolationCoordinatorV1:
    """Join all eight post-agent/disconnected containers before any verifier."""

    def __init__(self, participant_count: int = 8) -> None:
        if participant_count != 8:
            raise SealedRunnerError("sealed verifier barrier must contain eight units")
        self.participant_count = participant_count
        self._barrier = threading.Barrier(participant_count)
        self._lock = threading.Lock()
        self._ready: set[str] = set()
        self._released: set[str] = set()

    def await_all(self, work_unit_hash: str) -> None:
        with self._lock:
            if work_unit_hash in self._ready:
                raise SealedRunnerError("sealed verifier barrier replay is forbidden")
            self._ready.add(work_unit_hash)
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError as exc:
            raise SealedRunnerError("sealed verifier barrier aborted") from exc
        with self._lock:
            self._released.add(work_unit_hash)

    def abort(self) -> None:
        self._barrier.abort()

    @property
    def complete(self) -> bool:
        with self._lock:
            return len(self._ready) == len(self._released) == self.participant_count


class _SealedOfflineVerifierIsolationProxyV1:
    """Run the operator/checkpoint, disconnect, join, then expose tests."""

    def __init__(
        self,
        delegate: Any,
        *,
        backend: Any,
        network_name: str,
        coordinator: _SealedVerifierIsolationCoordinatorV1,
    ) -> None:
        if not network_name.strip():
            raise SealedRunnerError("sealed verifier network name is missing")
        self.delegate = delegate
        self.backend = backend
        self.network_name = network_name
        self.coordinator = coordinator
        self._seen = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    @property
    def _docker_proxy(self) -> Any:
        # RAW may expose the Docker verifier proxy directly, while candidate
        # exposes its contract-operator proxy whose delegate is Docker.  Use
        # instance-owned fields so proxy __getattr__ forwarding cannot make an
        # operator proxy look like Docker.
        direct = vars(self.delegate)
        if "_verifier_sources" in direct and "egress_policy" in direct:
            value = self.delegate
        else:
            value = direct.get("delegate")
        if (
            value is None
            or not callable(getattr(value, "run", None))
            or "_verifier_sources" not in vars(value)
            or "egress_policy" not in vars(value)
        ):
            raise SealedRunnerError("sealed verifier Docker proxy is unavailable")
        return value

    @property
    def _host_delegate(self) -> Any:
        value = getattr(self._docker_proxy, "delegate", None)
        if value is None or not callable(getattr(value, "run", None)):
            raise SealedRunnerError("sealed verifier host delegate is unavailable")
        return value

    def _run_host(self, command: Sequence[str], *, label: str) -> Any:
        completed = self._host_delegate.run(
            [str(value) for value in command],
            check=False,
            capture_output=True,
            text=True,
        )
        returncode = getattr(completed, "returncode", None)
        if isinstance(returncode, bool) or returncode != 0:
            raise SealedRunnerError(label)
        return completed

    def _inspect_empty(self, container: str, *, phase: str) -> str:
        inspected = self._run_host(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .NetworkSettings.Networks}}",
                container,
            ],
            label=f"sealed verifier {phase} network inspection failed",
        )
        raw = str(getattr(inspected, "stdout", "") or "").strip()
        try:
            networks = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SealedRunnerError(
                f"sealed verifier {phase} network inspection is malformed"
            ) from exc
        if networks != {}:
            raise SealedRunnerError(
                f"sealed verifier container remained network-attached {phase}"
            )
        return stable_hash(networks)

    def _run_post_agent_operator(self, container: str) -> None:
        if self.backend.durable_arm == "raw":
            self.backend._checkpoint_raw_before_verifier_v2()
        elif self.backend.durable_arm == "candidate":
            self.backend._execute_contract_plan_before_verifier_v2(
                delegate=self._docker_proxy,
                container_name=container,
            )
        else:
            raise SealedRunnerError("sealed verifier received an unknown arm")

    def run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        command = list(args) if isinstance(args, (list, tuple)) else args
        if not (
            isinstance(command, list)
            and len(command) >= 4
            and command[:2] == ["docker", "exec"]
            and "/tests/test.sh" in {str(value) for value in command[3:]}
        ):
            return self.delegate.run(command, *positional, **kwargs)
        if self._seen:
            raise SealedRunnerError("sealed offline verifier replay is forbidden")
        self._seen = True
        container = str(command[2])
        self._run_post_agent_operator(container)
        self._run_host(
            [
                "docker",
                "network",
                "disconnect",
                "--force",
                self.network_name,
                container,
            ],
            label="sealed verifier network disconnect failed",
        )
        before_hash = self._inspect_empty(container, phase="before")
        # Every participant reaches this point only after its agent exited,
        # its arm-local operator/checkpoint completed, and its network was
        # removed.  The inner Docker proxy (which copies /tests) is not called
        # until all eight participants have joined.
        self.coordinator.await_all(self.backend.durable_work_unit_hash)
        try:
            return self._docker_proxy.run(command, *positional, **kwargs)
        finally:
            after_hash = self._inspect_empty(container, phase="after")
            receipt = atomic_write_hashed_json_v2(
                self.backend.durable_state_root / NETWORK_ISOLATION_FILENAME,
                {
                    "receipt_version": NETWORK_ISOLATION_VERSION,
                    "work_unit_hash": self.backend.durable_work_unit_hash,
                    "request_hash": self.backend.durable_request_hash,
                    "arm": self.backend.durable_arm,
                    "container_name_hash": stable_hash(
                        {"container_name": container}
                    ),
                    "disconnected_network_name_hash": stable_hash(
                        {"network_name": self.network_name}
                    ),
                    "before_networks_hash": before_hash,
                    "after_networks_hash": after_hash,
                    "before_attached_network_count": 0,
                    "after_attached_network_count": 0,
                    "agent_exited_before_disconnect": True,
                    "operator_or_raw_checkpoint_completed_before_disconnect": True,
                    "tests_materialized_before_disconnect": False,
                    "all_agents_exited_and_disconnected_before_any_verifier_materialized": True,
                    "barrier_participant_count": 8,
                    "verifier_network": "none",
                    "online_judge_calls": 0,
                    "raw_network_payload_persisted": False,
                    "sealed_content_persisted": False,
                },
                hash_field="receipt_hash",
                refuse_existing=True,
            )
            self.backend._sealed_network_isolation_receipt_hash = receipt[
                "receipt_hash"
            ]


class _SealedNetworkIsolationBackendMixinV1:
    def __init__(
        self,
        *args: Any,
        sealed_isolation_coordinator: _SealedVerifierIsolationCoordinatorV1,
        **kwargs: Any,
    ) -> None:
        self._sealed_isolation_coordinator = sealed_isolation_coordinator
        self._sealed_network_isolation_receipt_hash: str | None = None
        super().__init__(*args, **kwargs)

    @contextmanager
    def _verifier_isolation(
        self,
        runner: Any,
        *,
        egress_policy: Any,
        **kwargs: Any,
    ) -> Iterator[None]:
        with super()._verifier_isolation(
            runner,
            egress_policy=egress_policy,
            **kwargs,
        ):
            operator_proxy = runner.subprocess
            runner.subprocess = _SealedOfflineVerifierIsolationProxyV1(
                operator_proxy,
                backend=self,
                network_name=egress_policy.network_name,
                coordinator=self._sealed_isolation_coordinator,
            )
            try:
                yield
            finally:
                runner.subprocess = operator_proxy

    def run(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().run(*args, **kwargs)
        except BaseException:
            self._sealed_isolation_coordinator.abort()
            raise


class _SealedRawBackendV1(
    _SealedNetworkIsolationBackendMixinV1,
    DurableRawSubprocessBackendV2,
):
    pass


class _SealedCandidateBackendV1(
    _SealedNetworkIsolationBackendMixinV1,
    DurableFinancialSec13FContractBackendV2,
):
    pass


def _safe_failure_snapshot(worker_root: Path) -> dict[str, Any]:
    if not worker_root.exists():
        return {
            "work_root_present": False,
            "model_execution_claim_count": 0,
            "semantic_evidence_receipt_count": 0,
            "raw_content_persisted": False,
        }
    return {
        "work_root_present": True,
        "model_execution_claim_count": sum(
            1 for _ in worker_root.rglob(MODEL_EXECUTION_CLAIM_FILENAME)
        ),
        "semantic_evidence_receipt_count": sum(
            1 for _ in worker_root.rglob(SEMANTIC_EVIDENCE_FILENAME)
        ),
        "raw_content_persisted": False,
    }


def _sealed_descriptive_results(execution: Any) -> dict[str, Any]:
    """Summarize only valid pairs; invalid outcomes never become gains/ties."""

    pairs: list[dict[str, Any]] = []
    for pair in execution.pair_results:
        raw = pair.raw_observation
        candidate = pair.candidate_observation
        if not isinstance(raw, SkillLearnTrialObservation) or not isinstance(
            candidate, SkillLearnTrialObservation
        ):
            raise SealedRunnerError("sealed backend returned an unknown observation")
        valid = raw.valid and candidate.valid
        if valid:
            raw_success: bool | None = raw.success
            candidate_success: bool | None = candidate.success
            delta: int | None = int(candidate.success) - int(raw.success)
            relation = "gain" if delta > 0 else "harm" if delta < 0 else "tie"
        else:
            raw_success = candidate_success = None
            delta = None
            relation = "invalid"
        pairs.append(
            {
                "item_id_hash": stable_hash({"item_id": pair.target.item_id}),
                "sealed_replicate": pair.target.replicate,
                "pair_id": pair.pair_id,
                "raw_observation_hash": raw.observation_hash,
                "candidate_observation_hash": candidate.observation_hash,
                "raw_valid": raw.valid,
                "candidate_valid": candidate.valid,
                "raw_success": raw_success,
                "candidate_success": candidate_success,
                "delta": delta,
                "relation": relation,
            }
        )
    valid_rows = [row for row in pairs if row["relation"] != "invalid"]
    body = {
        "pairs": pairs,
        "pair_set_hash": stable_hash(pairs),
        "valid_pair_count": len(valid_rows),
        "invalid_pair_count": len(pairs) - len(valid_rows),
        "raw_successes": sum(row["raw_success"] is True for row in valid_rows),
        "candidate_successes": sum(
            row["candidate_success"] is True for row in valid_rows
        ),
        "net_delta": sum(int(row["delta"]) for row in valid_rows),
        "gain_count": sum(row["relation"] == "gain" for row in valid_rows),
        "harm_count": sum(row["relation"] == "harm" for row in valid_rows),
        "tie_count": sum(row["relation"] == "tie" for row in valid_rows),
    }
    return body


def _sealed_terminal_disposition_v1(results: Mapping[str, Any]) -> str:
    if (
        results.get("valid_pair_count") == 4
        and results.get("invalid_pair_count") == 0
    ):
        return "executed_complete"
    return "executed_incomplete_no_retry"


def _validate_inputs(
    *,
    benchmark: Path,
    payload: Mapping[str, Any],
    prewarm: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    materialization = read_json(benchmark / MATERIALIZATION_REPORT_NAME)
    materialization_body = dict(materialization)
    materialization_hash = materialization_body.pop("materialization_hash", None)
    prewarm_body = dict(prewarm)
    prewarm_hash = prewarm_body.pop("prewarm_hash", None)
    rows = prewarm.get("formal_cache_rows")
    items = payload["sealed_items"]
    tree = sealed_benchmark_tree_receipt_v1(benchmark)
    if (
        materialization_hash != payload_hash(materialization_body)
        or materialization_hash != freeze.get("materialization_hash")
        or materialization.get("sealed_payload_hash") != payload["sealed_payload_hash"]
        or materialization.get("sealed_gold_hash") != freeze.get("sealed_gold_hash")
        or materialization.get("benchmark_tree_hash") != tree["tree_hash"]
        or prewarm_hash != payload_hash(prewarm_body)
        or prewarm_hash != freeze.get("prewarm_hash")
        or prewarm.get("prewarm_version") != PREWARM_VERSION
        or prewarm.get("sealed_payload_hash") != payload["sealed_payload_hash"]
        or prewarm.get("benchmark_tree_hash") != tree["tree_hash"]
        or prewarm.get("formal_execution_cache_only") is not True
        or prewarm.get("formal_image_cache_only") is not True
        or prewarm.get("formal_offline_verifier_cache_only") is not True
        or prewarm.get("formal_verifier_network") != "none"
        or prewarm.get("model_calls") != 0
        or prewarm.get("online_judge_calls") != 0
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise SealedRunnerError("sealed frozen input drifted")
    result: dict[str, Mapping[str, Any]] = {}
    for item, row in zip(items, rows or ()):
        item_id = str(item["item_id"])
        if (
            not isinstance(row, Mapping)
            or row.get("item_id_hash") != payload_hash({"item_id": item_id})
        ):
            raise SealedRunnerError("sealed prewarm item order drifted")
        result[item_id] = row
    return result


def _sealed_network_isolation_evidence_v1(
    *,
    worker_root: Path,
    plan: Any,
    backends: Mapping[str, ContractRecoveryBoundBackendV2],
    coordinator: _SealedVerifierIsolationCoordinatorV1,
) -> list[dict[str, Any]]:
    if not coordinator.complete:
        raise SealedRunnerError("sealed verifier isolation barrier is incomplete")
    rows: list[dict[str, Any]] = []
    work_by_hash = {work.work_unit_hash: work for work in plan.work_units}
    for work_hash in sorted(work_by_hash):
        work = work_by_hash[work_hash]
        wrapper = backends.get(work_hash)
        if wrapper is None:
            raise SealedRunnerError("sealed network backend evidence is missing")
        delegate = wrapper.delegate
        path = worker_root / work_hash / "durable" / NETWORK_ISOLATION_FILENAME
        receipt = read_hashed_json_v2(path, hash_field="receipt_hash")
        empty_hash = stable_hash({})
        if (
            receipt.get("receipt_version") != NETWORK_ISOLATION_VERSION
            or receipt.get("work_unit_hash") != work_hash
            or receipt.get("request_hash") != work.request.request_hash
            or receipt.get("arm") != work.arm
            or receipt.get("before_networks_hash") != empty_hash
            or receipt.get("after_networks_hash") != empty_hash
            or receipt.get("before_attached_network_count") != 0
            or receipt.get("after_attached_network_count") != 0
            or receipt.get("agent_exited_before_disconnect") is not True
            or receipt.get(
                "operator_or_raw_checkpoint_completed_before_disconnect"
            )
            is not True
            or receipt.get("tests_materialized_before_disconnect") is not False
            or receipt.get(
                "all_agents_exited_and_disconnected_before_any_verifier_materialized"
            )
            is not True
            or receipt.get("barrier_participant_count") != 8
            or receipt.get("verifier_network") != "none"
            or receipt.get("online_judge_calls") != 0
            or receipt.get("raw_network_payload_persisted") is not False
            or receipt.get("sealed_content_persisted") is not False
            or getattr(delegate, "_sealed_network_isolation_receipt_hash", None)
            != receipt.get("receipt_hash")
        ):
            raise SealedRunnerError("sealed network isolation receipt drifted")
        rows.append(
            {
                "work_unit_hash": work_hash,
                "arm": work.arm,
                "receipt_hash": receipt["receipt_hash"],
                "before_networks_hash": receipt["before_networks_hash"],
                "after_networks_hash": receipt["after_networks_hash"],
                "raw_content_persisted": False,
            }
        )
    if (
        len(rows) != 8
        or len({row["receipt_hash"] for row in rows}) != 8
        or sum(row["arm"] == "raw" for row in rows) != 4
        or sum(row["arm"] == "candidate" for row in rows) != 4
    ):
        raise SealedRunnerError("sealed network isolation receipt set drifted")
    return rows


def run_sealed_v1(
    *,
    project_root: str | Path,
    benchmark_root: str | Path,
    measurement_view: Mapping[str, Any],
    sealed_payload: Mapping[str, Any],
    prewarm: Mapping[str, Any],
    execution_freeze: Mapping[str, Any],
    candidate: FixedContractCandidateV2,
    env_file: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    project = Path(project_root).expanduser().resolve(strict=True)
    benchmark = Path(benchmark_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, mode=0o700)
    worker_root = destination / "worker_state"
    execution_started = False
    model_calls_claimed = 0
    try:
        freeze_hash = validate_sealed_execution_freeze_v1(
            execution_freeze,
            project_root=project,
            candidate=candidate,
        )
        payload = verify_sealed_payload_v1(
            sealed_payload,
            measurement_view=measurement_view,
        )
        if (
            payload["sealed_payload_hash"] != execution_freeze.get("sealed_payload_hash")
            or execution_freeze.get("execution_policy", {}).get("physical_model_calls") != 8
        ):
            raise SealedRunnerError("sealed payload differs from freeze")
        try:
            provider = validate_execution_provider_binding_v1(
                execution_freeze["provider"],
                project_root=project,
                env_file=env_file,
            )
            loaded_provider = load_provider_environment_v1(env_file)
        except Exception as exc:
            raise SealedRunnerError("Plus provider identity failed closed") from exc
        if (
            provider.get("provider_label") != "plus"
            or loaded_provider.get("api_key_hmac_sha256")
            != provider.get("api_key_hmac_sha256")
            or loaded_provider.get("model") != provider.get("model")
            or loaded_provider.get("api_origin") != provider.get("api_origin")
        ):
            raise SealedRunnerError("current provider differs from frozen Plus")
        prewarm_rows = _validate_inputs(
            benchmark=benchmark,
            payload=payload,
            prewarm=prewarm,
            freeze=execution_freeze,
        )
        protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
        if (
            protocol.payload.get("model") != provider.get("model")
            or protocol.payload.get("provider_endpoint_origin") != provider.get("api_origin")
        ):
            raise SealedRunnerError("provider differs from frozen paper protocol")
        _configure_environment(protocol)
        treatment_section = execution_freeze["treatment"]
        treatment = FixedPeriodOutTreatmentV2(
            recipe_id=candidate.recipe_id,
            program_set_hash=candidate.program_set_hash,
            period_out_treatment_id=str(treatment_section["period_out_treatment_id"]),
            external_skill_source_receipt_hash=candidate.external_skill_source_receipt_hash,
            candidate_skill_source=candidate.candidate_skill_source,
        )
        targets = tuple(
            SealedTargetV1(str(item["item_id"]), int(item["replicate"]))
            for item in payload["sealed_items"]
        )
        plan = build_sealed_plan_v1(
            targets=targets,
            manifest_hash=str(payload["sealed_payload_hash"]),
            evaluator_epoch=str(treatment_section["evaluator_epoch"]),
            treatment=treatment,
            agent_id=str(protocol.payload["agent_id"]),
            model=str(protocol.payload["model"]),
            max_steps=int(protocol.payload["max_steps"]),
            codex_agent_execution_policy_hash=protocol.codex_agent_execution_policy.policy_hash,
        )
        if execution_freeze.get("plan") != {
            "plan_hash": plan.plan_hash,
            "safe_payload": plan.safe_payload(),
        }:
            raise SealedRunnerError("sealed plan differs from freeze")
        planner = SharedFinancialSec13FContractPlannerV2(
            asset_path=candidate.operator_asset_path
        )
        precomputed: dict[str, BoundContractPlannerV2] = {}
        precomputed_receipts: dict[str, Mapping[str, Any]] = {}
        receipt_rows = []
        for item in payload["sealed_items"]:
            item_id = str(item["item_id"])
            contract_plan, extraction = planner.build(str(item["instruction"]))
            bound = BoundContractPlannerV2(
                shared=planner,
                instruction_sha256=str(item["instruction_sha256"]),
                plan=contract_plan,
                extraction_receipt=extraction,
            )
            precomputed[item_id] = bound
            receipt = {
                "item_id_hash": payload_hash({"item_id": item_id}),
                "instruction_sha256": item["instruction_sha256"],
                "plan_hash": contract_plan["plan_hash"],
                "extraction_receipt_hash": extraction["receipt_hash"],
                "raw_plan_persisted": False,
            }
            receipt_rows.append(receipt)
            precomputed_receipts[item_id] = {
                "applicable": True,
                "instruction_sha256": item["instruction_sha256"],
                "plan_hash": contract_plan["plan_hash"],
                "extraction_receipt_hash": extraction["receipt_hash"],
                "planner_hash": bound.planner_hash,
                "raw_plan_persisted": False,
                "model_calls": 0,
                "online_calls": 0,
            }
        if (
            receipt_rows != execution_freeze.get("precomputed_plan_receipts")
            or stable_hash(receipt_rows) != execution_freeze.get("precomputed_plan_set_hash")
        ):
            raise SealedRunnerError("sealed contract plans differ from freeze")
        _legacy._write_or_verify_hashed_json_v2(
            destination / "execution.plan.json",
            {
                "runner_version": RUNNER_VERSION,
                "execution_freeze_hash": freeze_hash,
                "plan_hash": plan.plan_hash,
                "safe_payload": plan.safe_payload(),
                "precomputed_plan_set_hash": execution_freeze["precomputed_plan_set_hash"],
                "physical_work_unit_count": 8,
                "sealed_content_persisted": False,
                "raw_plan_persisted": False,
                "model_calls_before_batch": 0,
            },
            hash_field="plan_receipt_hash",
        )
        for work in plan.work_units:
            semantic = (
                precomputed_receipts[work.target.item_id]
                if work.arm == "candidate"
                else {"applicable": False, "arm": "raw", "model_calls": 0, "online_calls": 0}
            )
            _legacy._ensure_work_state_v2(
                state_root=worker_root / work.work_unit_hash / "durable",
                work=work,
                planned_payload={
                    **work.safe_payload(),
                    "trial_id": work.request.trial_id,
                    "execution_freeze_hash": freeze_hash,
                    "model_calls": 0,
                    "retry_count": 0,
                },
                semantic_plan_payload=semantic,
            )
        event_sink = JsonlEventSink(destination / EVENTS_FILENAME)
        cache, offline_cache = _legacy._verify_formal_local_cache(
            benchmark_root=benchmark,
            item_ids=[target.item_id for target in targets],
            prewarm_rows=prewarm_rows,
            event_sink=event_sink,
        )
        if cache.cache_only is not True or not isinstance(
            offline_cache, SkillLearnOfflineVerifierRuntimeCache
        ):
            raise SealedRunnerError("sealed execution is not cache-only")
        _legacy._write_or_verify_hashed_json_v2(
            destination / "batch.started.json",
            {
                "runner_version": RUNNER_VERSION,
                "execution_freeze_hash": freeze_hash,
                "provider_binding_hash": provider["binding_hash"],
                "provider_label": "plus",
                "plan_hash": plan.plan_hash,
                "physical_work_unit_count": 8,
                "all_futures_required": True,
                "retry_authorized": False,
                "replay_authorized": False,
                "resampling_authorized": False,
                "provider_switch_authorized": False,
                "cache_only": True,
                "offline_judge_only": True,
            },
            hash_field="batch_start_hash",
        )
        execution_started = True
        limiter = SkillLearnModelInferenceLimiter(8)
        isolation_coordinator = _SealedVerifierIsolationCoordinatorV1()
        backends: dict[str, ContractRecoveryBoundBackendV2] = {}

        def backend_factory(work: Any) -> ContractRecoveryBoundBackendV2:
            state_root = worker_root / work.work_unit_hash / "durable"
            common = {
                "agent_id": work.request.agent_id,
                "model": work.request.model,
                "max_steps": work.request.max_steps,
                "provider_mode": "openai_compatible",
                "trials_dir": worker_root / work.work_unit_hash / "trials",
                "record_upstream": True,
                "prebuilt_cache": cache,
                "offline_verifier_cache": offline_cache,
                "provider_circuit": SkillLearnProviderCircuit(),
                "model_inference_limiter": limiter,
                "codex_agent_execution_policy": protocol.codex_agent_execution_policy,
                "event_sink": event_sink,
                "durable_state_root": state_root,
                "durable_work_unit_hash": work.work_unit_hash,
                "durable_request_hash": work.request.request_hash,
            }
            wrapper_kwargs: dict[str, Any] = {}
            if work.arm == "candidate":
                delegate = _SealedCandidateBackendV1(
                    benchmark,
                    sealed_isolation_coordinator=isolation_coordinator,
                    planner=precomputed[work.target.item_id],
                    expected_program_id=candidate.recipe_id,
                    expected_program_set_hash=candidate.program_set_hash,
                    expected_treatment_hash=treatment.period_out_treatment_id,
                    expected_external_skill_source_receipt_hash=candidate.external_skill_source_receipt_hash,
                    expected_precomputed_plan_hash=str(precomputed_receipts[work.target.item_id]["plan_hash"]),
                    **common,
                )
                wrapper_kwargs = {
                    "expected_plan_hash": str(precomputed_receipts[work.target.item_id]["plan_hash"]),
                    "expected_program_id": candidate.recipe_id,
                    "expected_treatment_hash": treatment.period_out_treatment_id,
                    "expected_external_source_receipt_hash": candidate.external_skill_source_receipt_hash,
                }
            else:
                delegate = _SealedRawBackendV1(
                    benchmark,
                    sealed_isolation_coordinator=isolation_coordinator,
                    **common,
                )
            wrapper = ContractRecoveryBoundBackendV2(
                delegate=delegate,
                work=work,
                state_root=state_root,
                trial_root=_legacy._trial_root_for_work_v2(worker_root, work),
                expected_process_scope=str(protocol.codex_agent_execution_policy.action_budget_process_scope),
                **wrapper_kwargs,
            )
            wrapper.inspect_recovery()
            backends[work.work_unit_hash] = wrapper
            return wrapper

        with future_terminal_semantics_v2():
            execution = execute_sealed_plan_v1(plan=plan, backend_factory=backend_factory)
        model_calls_claimed = sum(wrapper.backend_called for wrapper in backends.values())
        if model_calls_claimed != 8:
            raise SealedRunnerError("sealed execution did not consume exactly eight calls")
        descriptive = _sealed_descriptive_results(execution)
        network_isolation_evidence = _sealed_network_isolation_evidence_v1(
            worker_root=worker_root,
            plan=plan,
            backends=backends,
            coordinator=isolation_coordinator,
        )
        semantic_evidence = []
        for work_hash, wrapper in sorted(backends.items()):
            decision = wrapper.inspect_recovery()
            if not decision.completed or decision.model_calls_accounted != 1 or decision.model_replay_authorized:
                raise SealedRunnerError("sealed durable state is incomplete")
            work = wrapper.work
            evidence_path = worker_root / work_hash / "durable" / SEMANTIC_EVIDENCE_FILENAME
            if work.arm == "candidate":
                receipt = read_hashed_json_v2(evidence_path, hash_field="receipt_hash")
                evidence = receipt.get("evidence")
                if (
                    not isinstance(evidence, Mapping)
                    or evidence.get("plan_hash") != precomputed_receipts[work.target.item_id]["plan_hash"]
                    or evidence.get("program_id") != candidate.recipe_id
                    or evidence.get("treatment_hash") != treatment.period_out_treatment_id
                    or evidence.get("online_calls") != 0
                    or evidence.get("answers_payload_persisted") is not False
                ):
                    raise SealedRunnerError("sealed candidate evidence drifted")
                _assert_no_raw_contract_payload_v2(evidence)
                semantic_evidence.append({"work_unit_hash": work_hash, "evidence": dict(evidence)})
            elif evidence_path.exists() or evidence_path.is_symlink():
                raise SealedRunnerError("sealed RAW emitted candidate evidence")
        if len(semantic_evidence) != 4:
            raise SealedRunnerError("sealed candidate evidence cardinality drifted")
        if sealed_benchmark_tree_receipt_v1(benchmark)["tree_hash"] != execution_freeze["benchmark_tree_hash"]:
            raise SealedRunnerError("sealed benchmark changed during execution")
        observations = [row.observation for row in execution.work_results]
        if not all(
            isinstance(row, SkillLearnTrialObservation)
            and row.raw_trial_artifacts_persisted
            for row in observations
        ):
            raise SealedRunnerError("sealed trial artifact closure is incomplete")
        artifact_closure = _legacy._artifact_closure(worker_root)
        terminal_disposition = _sealed_terminal_disposition_v1(descriptive)
        if terminal_disposition != "executed_complete":
            incomplete = {
                "runner_version": RUNNER_VERSION,
                "terminal_report_kind": "sealed_invalid_pair_disposition_v1",
                "execution_completed": False,
                "physical_execution_completed": True,
                "evidence_valid": False,
                "disposition": terminal_disposition,
                "execution_freeze_hash": freeze_hash,
                "authorization_hash": execution_freeze["authorization_hash"],
                "sealed_payload_hash": payload["sealed_payload_hash"],
                "plan_hash": plan.plan_hash,
                "results": descriptive,
                "semantic_runtime_evidence_set_hash": stable_hash(
                    semantic_evidence
                ),
                "semantic_runtime_evidence_persisted_in_terminal_report": False,
                "verifier_network_isolation_receipts": network_isolation_evidence,
                "verifier_network_isolation_receipt_set_hash": stable_hash(
                    network_isolation_evidence
                ),
                "verifier_network_isolation_receipt_count": 8,
                "all_agents_exited_and_disconnected_before_any_verifier_materialized": True,
                "verifier_network_before_and_after": "none",
                "worker_artifact_closure": artifact_closure,
                "physical_model_call_count": 8,
                "raw_model_call_count": 4,
                "candidate_model_call_count": 4,
                "operator_call_count": 4,
                "offline_verifier_call_count": 8,
                "retry_authorized": False,
                "replay_authorized": False,
                "resampling_authorized": False,
                "provider_switch_authorized": False,
                "online_judge_calls": 0,
                "performance_gate_applied": False,
                "sealed_content_persisted_in_report": False,
                "gold_content_persisted_in_report": False,
                "answers_payload_persisted": False,
                "raw_plan_persisted": False,
                "secret_value_persisted": False,
            }
            return atomic_write_hashed_json_v2(
                destination / FAILURE_FILENAME,
                incomplete,
                hash_field="report_hash",
            )
        body = {
            "runner_version": RUNNER_VERSION,
            "execution_completed": True,
            "physical_execution_completed": True,
            "evidence_valid": True,
            "disposition": terminal_disposition,
            "execution_freeze_hash": freeze_hash,
            "authorization_hash": execution_freeze["authorization_hash"],
            "sealed_payload_hash": payload["sealed_payload_hash"],
            "plan_hash": plan.plan_hash,
            "plan": plan.safe_payload(),
            "execution": execution.safe_payload(),
            "results": descriptive,
            "semantic_runtime_evidence": semantic_evidence,
            "semantic_runtime_evidence_set_hash": stable_hash(semantic_evidence),
            "verifier_network_isolation_receipts": network_isolation_evidence,
            "verifier_network_isolation_receipt_set_hash": stable_hash(
                network_isolation_evidence
            ),
            "verifier_network_isolation_receipt_count": 8,
            "all_agents_exited_and_disconnected_before_any_verifier_materialized": True,
            "verifier_network_before_and_after": "none",
            "worker_artifact_closure": artifact_closure,
            "physical_model_call_count": 8,
            "raw_model_call_count": 4,
            "candidate_model_call_count": 4,
            "operator_call_count": 4,
            "offline_verifier_call_count": 8,
            "model_calls_this_invocation": 8,
            "model_inference_slot_limit": 8,
            "maximum_concurrent_model_calls": limiter.maximum_active,
            "maximum_workers": 8,
            "all_futures_submitted_before_results_read": True,
            "independent_provider_circuit_count": 8,
            "provider_label": "plus",
            "provider_binding_hash": provider["binding_hash"],
            "retry_count": 0,
            "model_replay_count": 0,
            "resampling_used": False,
            "mid_batch_provider_switch_used": False,
            "offline_evaluation_only": True,
            "offline_judge_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "official_hipporag": False,
            "hipporag_status": "not_applicable_nonexecuted",
            "official_hipporag_execution_count": 0,
            "performance_gate_applied": False,
            "single_execution": True,
            "backend_runtime_identity": backend_runtime_identity_v2(),
            "sealed_content_persisted_in_report": False,
            "gold_content_persisted_in_report": False,
            "answers_payload_persisted": False,
            "raw_plan_persisted": False,
            "secret_value_persisted": False,
        }
        return atomic_write_hashed_json_v2(
            destination / REPORT_FILENAME, body, hash_field="report_hash"
        )
    except Exception as exc:
        durable_snapshot = _safe_failure_snapshot(worker_root)
        failure = {
            "runner_version": RUNNER_VERSION,
            "execution_completed": False,
            "execution_started": execution_started,
            "disposition": "executed_incomplete_no_retry",
            "model_execution_claim_count": durable_snapshot[
                "model_execution_claim_count"
            ],
            "retry_authorized": False,
            "replay_authorized": False,
            "resampling_authorized": False,
            "provider_switch_authorized": False,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "durable_snapshot": durable_snapshot,
            "raw_error_persisted": False,
            "sealed_content_persisted": False,
            "secret_value_persisted": False,
        }
        try:
            atomic_write_hashed_json_v2(
                destination / FAILURE_FILENAME, failure, hash_field="report_hash"
            )
        except FileExistsError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--sealed-payload", type=Path, required=True)
    parser.add_argument("--prewarm", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project = args.project_root.expanduser().resolve(strict=True)
    candidate = load_fixed_contract_candidate_v2(project)
    # The committed public freeze is loaded and authenticated before the
    # private prepared sealed payload is opened by this formal entry point.
    execution_freeze = read_json(args.execution_freeze)
    validate_sealed_execution_freeze_v1(
        execution_freeze,
        project_root=project,
        candidate=candidate,
    )
    report = run_sealed_v1(
        project_root=project,
        benchmark_root=args.benchmark_root,
        measurement_view=read_json(args.measurement_view),
        sealed_payload=read_json(args.sealed_payload),
        prewarm=read_json(args.prewarm),
        execution_freeze=execution_freeze,
        candidate=candidate,
        env_file=args.env_file,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "evidence_valid": report["evidence_valid"],
                "disposition": report["disposition"],
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("disposition") == "executed_complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())

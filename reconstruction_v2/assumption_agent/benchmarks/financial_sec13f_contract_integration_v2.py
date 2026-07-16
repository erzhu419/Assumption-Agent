from __future__ import annotations

"""Post-agent, pre-verifier integration for the public SEC-13F contract."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Iterator, Mapping, Sequence
import re

from ..events import Event
from ..models import stable_hash
from .financial_sec13f_contract_operator_v2 import (
    EXTRACTION_RECEIPT_VERSION,
    OPERATOR_VERSION,
    QUERY_RECEIPT_VERSION,
    build_contract_plan_v2,
    load_contract_asset_v2,
    payload_hash,
)
from .skilllearn_lifecycle import (
    DockerEgressPolicy,
    SkillLearnAgentTerminalError,
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .offline_verifier import OfflineVerifierRuntime


INTEGRATION_VERSION = "financial_sec13f_contract_post_agent_pre_verifier_v2"
FINANCIAL_FAMILY = "financial-analysis"
_CONTAINER_OPERATOR = "/tmp/assumption_sec13f_contract_operator_v2.py"
_CONTAINER_ASSET = "/tmp/assumption_sec13f_contract_asset_v2.json"
_CONTAINER_PLAN = "/tmp/assumption_sec13f_contract_plan_v2.json"
_CONTAINER_RECEIPT = "/tmp/assumption_sec13f_contract_receipt_v2.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FinancialSec13FContractIntegrationError(RuntimeError):
    """The isolated typed operator did not satisfy its frozen boundary."""


def _remove_ephemeral_host_root_v2(root: Path) -> None:
    """Delete and verify the host plan root without a fail-open path."""

    try:
        if root.exists() or root.is_symlink():
            shutil.rmtree(root)
    except Exception as exc:
        raise FinancialSec13FContractIntegrationError(
            "ephemeral host plan cleanup failed"
        ) from exc
    if root.exists() or root.is_symlink():
        raise FinancialSec13FContractIntegrationError(
            "ephemeral host plan cleanup was not confirmed"
        )


@dataclass
class _ContractRunStateV2:
    request_hash: str
    plan: Mapping[str, Any]
    extraction_receipt: Mapping[str, Any]
    verifier_triggered: bool = False
    runtime_evidence: Mapping[str, Any] | None = None


class SharedFinancialSec13FContractPlannerV2:
    """One immutable public-contract asset shared by item-local planners."""

    def __init__(self, *, asset_path: str | Path) -> None:
        self.asset_path = Path(asset_path).expanduser().resolve(strict=True)
        self.asset = load_contract_asset_v2(self.asset_path)

    @property
    def planner_hash(self) -> str:
        return stable_hash(
            {
                "integration_version": INTEGRATION_VERSION,
                "operator_version": OPERATOR_VERSION,
                "candidate_id": self.asset["candidate_id"],
                "asset_manifest_hash": self.asset["manifest_hash"],
                "contract_hash": self.asset["contract_hash"],
                "template_grammar_hash": self.asset["template_grammar_hash"],
                "operator_source_sha256": self.asset[
                    "operator_source_sha256"
                ],
            }
        )

    def build(self, instruction: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return build_contract_plan_v2(instruction, self.asset)


class _ContractVerifierProxyV2:
    def __init__(self, delegate: Any, *, backend: Any) -> None:
        self.delegate = delegate
        self.backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        command = list(args) if isinstance(args, (list, tuple)) else args
        if (
            isinstance(command, list)
            and len(command) >= 4
            and command[:2] == ["docker", "exec"]
            and "/tests/test.sh" in {str(value) for value in command[3:]}
        ):
            self.backend._execute_contract_plan_before_verifier_v2(
                delegate=self.delegate,
                container_name=str(command[2]),
            )
        return self.delegate.run(command, *positional, **kwargs)


class FinancialSec13FContractSubprocessBackendV2(SkillLearnSubprocessBackend):
    """Execute one transient typed plan after the agent and before tests."""

    def __init__(
        self,
        *args: Any,
        planner: SharedFinancialSec13FContractPlannerV2,
        expected_program_id: str,
        expected_treatment_hash: str,
        expected_external_skill_source_receipt_hash: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        for label, value in (
            ("expected_program_id", expected_program_id),
            ("expected_treatment_hash", expected_treatment_hash),
            (
                "expected_external_skill_source_receipt_hash",
                expected_external_skill_source_receipt_hash,
            ),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{label} must be an opaque SHA-256 identity")
        self.planner = planner
        self.expected_program_id = expected_program_id
        self.expected_treatment_hash = expected_treatment_hash
        self.expected_external_skill_source_receipt_hash = (
            expected_external_skill_source_receipt_hash
        )
        self._contract_local = threading.local()
        self._contract_evidence_lock = threading.Lock()
        self._contract_runtime_evidence: list[Mapping[str, Any]] = []

    @property
    def financial_backend_instance_hash(self) -> str:
        return stable_hash(
            {
                "integration_version": INTEGRATION_VERSION,
                "planner_hash": self.planner.planner_hash,
                "agent_id": self.agent_id,
                "model": self.model,
                "max_steps": self.max_steps,
                "codex_agent_execution_policy_hash": (
                    self.codex_agent_execution_policy_hash
                ),
                "expected_program_id": self.expected_program_id,
                "expected_treatment_hash": self.expected_treatment_hash,
                "expected_external_skill_source_receipt_hash": (
                    self.expected_external_skill_source_receipt_hash
                ),
            }
        )

    @property
    def financial_runtime_evidence(self) -> tuple[Mapping[str, Any], ...]:
        with self._contract_evidence_lock:
            return tuple(self._contract_runtime_evidence)

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        self._contract_local.state = None
        expected = (
            request.variant is TrialVariant.POLICY_ON
            and request.family == FINANCIAL_FAMILY
            and request.program_id == self.expected_program_id
            and request.treatment_hash == self.expected_treatment_hash
            and request.external_skill_source_receipt_hash
            == self.expected_external_skill_source_receipt_hash
        )
        if not expected:
            self.event_sink.emit(
                Event(
                    event="financial_sec13f_contract_identity_rejected_v2",
                    stage="benchmark.skilllearn.financial_sec13f_contract_v2",
                    trace_id=request.request_hash[:20],
                    payload={
                        "request_hash": request.request_hash,
                        "program_id_hash": stable_hash(
                            {"program_id": request.program_id}
                        ),
                        "treatment_hash_matches": (
                            request.treatment_hash
                            == self.expected_treatment_hash
                        ),
                        "external_source_receipt_matches": (
                            request.external_skill_source_receipt_hash
                            == self.expected_external_skill_source_receipt_hash
                        ),
                        "agent_started": False,
                        "online_calls": 0,
                    },
                )
            )
            return self._local_error(
                request, "financial_sec13f_contract_identity_mismatch"
            )
        instruction_path = (
            self.benchmark_root
            / "tasks"
            / request.family
            / request.item_id
            / "instruction.md"
        )
        try:
            instruction = instruction_path.read_text(encoding="utf-8")
            plan, extraction_receipt = self.planner.build(instruction)
            if (
                plan.get("candidate_id")
                != self.planner.asset["candidate_id"]
                or plan.get("asset_manifest_hash")
                != self.planner.asset["manifest_hash"]
                or plan.get("contract_hash")
                != self.planner.asset["contract_hash"]
                or plan.get("template_grammar_hash")
                != self.planner.asset["template_grammar_hash"]
                or plan.get("operator_source_sha256")
                != self.planner.asset["operator_source_sha256"]
            ):
                raise FinancialSec13FContractIntegrationError(
                    "contract plan binding drifted"
                )
            extraction_body = dict(extraction_receipt)
            extraction_hash = extraction_body.pop("receipt_hash", None)
            if (
                extraction_hash != payload_hash(extraction_body)
                or extraction_receipt.get("receipt_version")
                != EXTRACTION_RECEIPT_VERSION
                or extraction_receipt.get("plan_hash")
                != plan.get("plan_hash")
                or extraction_receipt.get("candidate_id")
                != self.planner.asset["candidate_id"]
                or extraction_receipt.get("asset_manifest_hash")
                != self.planner.asset["manifest_hash"]
                or extraction_receipt.get("raw_instruction_persisted")
                is not False
                or extraction_receipt.get("raw_entity_persisted_in_receipt")
                is not False
                or extraction_receipt.get("model_calls") != 0
                or extraction_receipt.get("online_calls") != 0
            ):
                raise FinancialSec13FContractIntegrationError(
                    "contract extraction receipt drifted"
                )
            self._contract_local.state = _ContractRunStateV2(
                request_hash=request.request_hash,
                plan=plan,
                extraction_receipt=extraction_receipt,
            )
            self.event_sink.emit(
                Event(
                    event="financial_sec13f_contract_plan_built_v2",
                    stage="benchmark.skilllearn.financial_sec13f_contract_v2",
                    trace_id=request.request_hash[:20],
                    payload={
                        "request_hash": request.request_hash,
                        "candidate_id": plan["candidate_id"],
                        "asset_manifest_hash": plan["asset_manifest_hash"],
                        "plan_hash": plan["plan_hash"],
                        "extraction_receipt_hash": extraction_receipt[
                            "receipt_hash"
                        ],
                        "instruction_sha256": plan["instruction_sha256"],
                        "raw_instruction_persisted": False,
                        "raw_entity_persisted_in_event": False,
                        "online_calls": 0,
                    },
                )
            )
        except Exception as error:
            self.event_sink.emit(
                Event(
                    event="financial_sec13f_contract_plan_failed_v2",
                    stage="benchmark.skilllearn.financial_sec13f_contract_v2",
                    trace_id=request.request_hash[:20],
                    payload={
                        "request_hash": request.request_hash,
                        "error_type": type(error).__name__,
                        "agent_started": False,
                        "online_calls": 0,
                    },
                )
            )
            return self._local_error(
                request, "financial_sec13f_contract_pre_agent_invalid"
            )
        try:
            observation = super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            state = getattr(self._contract_local, "state", None)
            if (
                observation.valid
                and (
                    not isinstance(state, _ContractRunStateV2)
                    or not isinstance(state.runtime_evidence, Mapping)
                )
            ):
                return replace(
                    observation,
                    success=False,
                    score=0.0,
                    metrics={"evaluation_valid": 0.0},
                    error_type="financial_sec13f_contract_receipt_missing",
                )
            return observation
        finally:
            self._contract_local.state = None

    @staticmethod
    def _run_checked(delegate: Any, command: Sequence[str]) -> Any:
        result = delegate.run(list(command), capture_output=True, text=True)
        if int(getattr(result, "returncode", 1)) != 0:
            raise FinancialSec13FContractIntegrationError(
                "contract container command failed"
            )
        return result

    def _execute_contract_plan_before_verifier_v2(
        self,
        *,
        delegate: Any,
        container_name: str,
    ) -> None:
        state = getattr(self._contract_local, "state", None)
        if not isinstance(state, _ContractRunStateV2):
            return
        if state.verifier_triggered:
            if state.runtime_evidence is None:
                raise SkillLearnAgentTerminalError(
                    "financial_sec13f_contract_runtime_receipt_invalid"
                )
            return
        state.verifier_triggered = True
        root = Path(tempfile.mkdtemp(prefix="financial-sec13f-contract-v2-"))
        pending_evidence: dict[str, Any] | None = None
        try:
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    state.plan,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            operator_source = Path(__file__).resolve().with_name(
                "financial_sec13f_contract_operator_v2.py"
            )
            asset_path = self.planner.asset_path
            expected_source_hash = str(
                self.planner.asset["operator_source_sha256"]
            )
            expected_asset_file_hash = hashlib.sha256(
                asset_path.read_bytes()
            ).hexdigest()
            expected_plan_file_hash = hashlib.sha256(
                plan_path.read_bytes()
            ).hexdigest()
            if hashlib.sha256(operator_source.read_bytes()).hexdigest() != (
                expected_source_hash
            ):
                raise FinancialSec13FContractIntegrationError(
                    "contract operator changed after freeze"
                )
            self._run_checked(
                delegate,
                [
                    "docker",
                    "exec",
                    container_name,
                    "rm",
                    "-f",
                    _CONTAINER_OPERATOR,
                    _CONTAINER_ASSET,
                    _CONTAINER_PLAN,
                    _CONTAINER_RECEIPT,
                ],
            )
            for source, destination in (
                (operator_source, _CONTAINER_OPERATOR),
                (asset_path, _CONTAINER_ASSET),
                (plan_path, _CONTAINER_PLAN),
            ):
                self._run_checked(
                    delegate,
                    [
                        "docker",
                        "cp",
                        str(source),
                        f"{container_name}:{destination}",
                    ],
                )
            readback = self._run_checked(
                delegate,
                [
                    "docker",
                    "exec",
                    container_name,
                    "sha256sum",
                    _CONTAINER_OPERATOR,
                    _CONTAINER_ASSET,
                    _CONTAINER_PLAN,
                ],
            )
            readback_tokens = str(getattr(readback, "stdout", "")).split()
            observed_hashes = readback_tokens[0::2]
            if observed_hashes != [
                expected_source_hash,
                expected_asset_file_hash,
                expected_plan_file_hash,
            ]:
                raise FinancialSec13FContractIntegrationError(
                    "container contract inputs drifted"
                )
            observed_source_hash = observed_hashes[0]
            self._run_checked(
                delegate,
                [
                    "docker",
                    "exec",
                    container_name,
                    "python3",
                    "-B",
                    _CONTAINER_OPERATOR,
                    "execute",
                    "--asset",
                    _CONTAINER_ASSET,
                    "--plan",
                    _CONTAINER_PLAN,
                    "--previous-root",
                    str(state.plan["previous_root"]),
                    "--current-root",
                    str(state.plan["current_root"]),
                    "--output",
                    "/root/answers.json",
                    "--receipt-output",
                    _CONTAINER_RECEIPT,
                ],
            )
            receipt_path = root / "query.receipt.json"
            self._run_checked(
                delegate,
                [
                    "docker",
                    "cp",
                    f"{container_name}:{_CONTAINER_RECEIPT}",
                    str(receipt_path),
                ],
            )
            output_readback = self._run_checked(
                delegate,
                [
                    "docker",
                    "exec",
                    container_name,
                    "sha256sum",
                    "/root/answers.json",
                ],
            )
            observed_output_hash = str(
                getattr(output_readback, "stdout", "")
            ).split()[0]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise FinancialSec13FContractIntegrationError(
                    "contract query receipt is malformed"
                )
            body = dict(receipt)
            declared = body.pop("receipt_hash", None)
            if (
                declared != payload_hash(body)
                or receipt.get("receipt_version") != QUERY_RECEIPT_VERSION
                or receipt.get("plan_hash") != state.plan.get("plan_hash")
                or receipt.get("candidate_id")
                != self.planner.asset.get("candidate_id")
                or receipt.get("asset_manifest_hash")
                != self.planner.asset.get("manifest_hash")
                or receipt.get("contract_hash")
                != self.planner.asset.get("contract_hash")
                or receipt.get("operator_source_sha256")
                != expected_source_hash
                or receipt.get("post_output_sha256")
                != observed_output_hash
                or receipt.get("answers_payload_persisted_in_receipt")
                is not False
                or receipt.get("raw_entity_persisted_in_receipt") is not False
                or receipt.get("network_calls") != 0
                or receipt.get("model_calls") != 0
                or receipt.get("verifier_content_accessed") is not False
                or receipt.get("gold_content_accessed") is not False
                or receipt.get("pack_content_accessed") is not False
            ):
                raise FinancialSec13FContractIntegrationError(
                    "contract query receipt failed closed"
                )
            pending_evidence = {
                "runtime_version": INTEGRATION_VERSION,
                "request_hash": state.request_hash,
                "candidate_id": self.planner.asset["candidate_id"],
                "asset_manifest_hash": self.planner.asset["manifest_hash"],
                "contract_hash": self.planner.asset["contract_hash"],
                "planner_hash": self.planner.planner_hash,
                "backend_instance_hash": self.financial_backend_instance_hash,
                "plan_hash": state.plan["plan_hash"],
                "extraction_receipt_hash": state.extraction_receipt[
                    "receipt_hash"
                ],
                "extraction_receipt": dict(state.extraction_receipt),
                "query_receipt_hash": receipt["receipt_hash"],
                "query_receipt": receipt,
                "output_sha256": observed_output_hash,
                "answers_payload_persisted": False,
                "operator_source_sha256": expected_source_hash,
                "program_id": self.expected_program_id,
                "treatment_hash": self.expected_treatment_hash,
                "external_skill_source_receipt_hash": (
                    self.expected_external_skill_source_receipt_hash
                ),
                "container_operator_readback_sha256": observed_source_hash,
                "container_asset_readback_sha256": observed_hashes[1],
                "container_plan_readback_sha256": observed_hashes[2],
                "executed_after_agent_exit": True,
                "executed_before_verifier_materialization": True,
                "online_calls": 0,
                "raw_instruction_persisted": False,
                "raw_entity_persisted_in_durable_evidence": False,
                "gold_content_accessed": False,
                "pack_content_accessed": False,
            }
        except SkillLearnAgentTerminalError:
            raise
        except Exception as error:
            self.event_sink.emit(
                Event(
                    event="financial_sec13f_contract_runtime_failed_v2",
                    stage="benchmark.skilllearn.financial_sec13f_contract_v2",
                    trace_id=state.request_hash[:20],
                    payload={
                        "request_hash": state.request_hash,
                        "error_type": type(error).__name__,
                        "verifier_materialized": False,
                        "online_calls": 0,
                    },
                )
            )
            raise SkillLearnAgentTerminalError(
                "financial_sec13f_contract_runtime_receipt_invalid"
            ) from error
        finally:
            cleanup_error: Exception | None = None
            try:
                self._run_checked(
                    delegate,
                    [
                        "docker",
                        "exec",
                        container_name,
                        "rm",
                        "-f",
                        _CONTAINER_OPERATOR,
                        _CONTAINER_ASSET,
                        _CONTAINER_PLAN,
                        _CONTAINER_RECEIPT,
                    ],
                )
            except Exception as error:
                cleanup_error = error
            # The host directory contains the ephemeral typed plan and receipt.
            # Remove it even when container cleanup fails; nesting this call in
            # the Docker try block would leave raw plan material in /tmp.
            try:
                _remove_ephemeral_host_root_v2(root)
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
            if cleanup_error is not None:
                raise SkillLearnAgentTerminalError(
                    "financial_sec13f_contract_runtime_cleanup_invalid"
                ) from cleanup_error
        if pending_evidence is None:
            raise SkillLearnAgentTerminalError(
                "financial_sec13f_contract_runtime_receipt_invalid"
            )
        pending_evidence["ephemeral_plan_deleted_before_verifier"] = True
        pending_evidence["evidence_hash"] = stable_hash(pending_evidence)
        state.runtime_evidence = pending_evidence
        with self._contract_evidence_lock:
            self._contract_runtime_evidence.append(pending_evidence)
        self.event_sink.emit(
            Event(
                event="financial_sec13f_contract_executed_v2",
                stage="benchmark.skilllearn.financial_sec13f_contract_v2",
                trace_id=state.request_hash[:20],
                payload=dict(pending_evidence),
            )
        )

    @contextmanager
    def _verifier_isolation(
        self,
        runner: Any,
        *,
        agent_runtime_volume: str | None = None,
        egress_policy: DockerEgressPolicy,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
        trace_id: str = "financial-sec13f-contract-verifier-isolation",
    ) -> Iterator[None]:
        with super()._verifier_isolation(
            runner,
            agent_runtime_volume=agent_runtime_volume,
            egress_policy=egress_policy,
            offline_verifier_runtime=offline_verifier_runtime,
            trace_id=trace_id,
        ):
            base_proxy = runner.subprocess
            runner.subprocess = _ContractVerifierProxyV2(
                base_proxy,
                backend=self,
            )
            try:
                yield
            finally:
                runner.subprocess = base_proxy

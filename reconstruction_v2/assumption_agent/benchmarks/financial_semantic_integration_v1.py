from __future__ import annotations

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
from .financial_semantic_operator_v1 import (
    FINANCIAL_QUERY_RECEIPT_VERSION,
    OfflineFinancialQA,
    build_financial_semantic_plan,
    load_financial_semantic_asset,
)
from .semantic_assignment_operator_v1 import OfflineMiniLMEncoder
from .skilllearn_lifecycle import (
    DockerEgressPolicy,
    SkillLearnAgentTerminalError,
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .offline_verifier import OfflineVerifierRuntime


FINANCIAL_SEMANTIC_INTEGRATION_VERSION = (
    "financial_semantic_post_agent_pre_verifier_integration_v1"
)
FINANCIAL_SEMANTIC_FAMILY = "financial-analysis"
_CONTAINER_OPERATOR = "/tmp/assumption_financial_semantic_operator_v1.py"
_CONTAINER_PLAN = "/tmp/assumption_financial_semantic_plan_v1.json"
_CONTAINER_RECEIPT = "/tmp/assumption_financial_semantic_receipt_v1.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FinancialSemanticIntegrationError(RuntimeError):
    pass


@dataclass
class _FinancialRunStateV1:
    request_hash: str
    plan: Mapping[str, Any]
    extraction_receipt: Mapping[str, Any]
    verifier_triggered: bool = False
    runtime_evidence: Mapping[str, Any] | None = None


class SharedFinancialSemanticPlannerV1:
    """One frozen offline semantic runtime shared by independent backends."""

    def __init__(
        self,
        *,
        asset_path: str | Path,
        minilm_runtime_asset_path: str | Path,
        minilm_snapshot_root: str | Path,
        qa_runtime_asset_path: str | Path,
        qa_snapshot_root: str | Path,
    ) -> None:
        self.asset = load_financial_semantic_asset(
            asset_path,
            minilm_runtime_asset_path=minilm_runtime_asset_path,
            qa_runtime_asset_path=qa_runtime_asset_path,
        )
        self.encoder = OfflineMiniLMEncoder(
            runtime_asset_path=minilm_runtime_asset_path,
            snapshot_root=minilm_snapshot_root,
        )
        self.qa = OfflineFinancialQA(
            runtime_asset_path=qa_runtime_asset_path,
            snapshot_root=qa_snapshot_root,
        )
        self._lock = threading.Lock()

    @property
    def planner_hash(self) -> str:
        return stable_hash(
            {
                "integration_version": FINANCIAL_SEMANTIC_INTEGRATION_VERSION,
                "candidate_id": self.asset["candidate_id"],
                "candidate_manifest_hash": self.asset["manifest_hash"],
                "operator_source_sha256": self.asset[
                    "operator_source_sha256"
                ],
            }
        )

    def build(self, instruction: str) -> tuple[dict[str, Any], dict[str, Any]]:
        # PyTorch modules are read-only here, but the lock also makes the
        # deterministic inference order explicit under a 3*n trial runner.
        with self._lock:
            return build_financial_semantic_plan(
                instruction=instruction,
                asset=self.asset,
                encoder=self.encoder,
                qa=self.qa,
                minilm_runtime_receipt=self.encoder.runtime_receipt,
                qa_runtime_receipt=self.qa.runtime_receipt,
            )


class _FinancialVerifierProxyV1:
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
            self.backend._execute_financial_plan_before_verifier_v1(
                delegate=self.delegate,
                container_name=str(command[2]),
            )
        return self.delegate.run(command, *positional, **kwargs)


class FinancialSemanticSubprocessBackendV1(SkillLearnSubprocessBackend):
    """Apply a frozen semantic plan after the agent and before offline tests."""

    def __init__(
        self,
        *args: Any,
        planner: SharedFinancialSemanticPlannerV1,
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
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be an opaque sha256 identity")
        self.planner = planner
        self.expected_program_id = expected_program_id
        self.expected_treatment_hash = expected_treatment_hash
        self.expected_external_skill_source_receipt_hash = (
            expected_external_skill_source_receipt_hash
        )
        self._financial_local = threading.local()
        self._financial_evidence_lock = threading.Lock()
        self._financial_runtime_evidence: list[Mapping[str, Any]] = []

    @property
    def financial_backend_instance_hash(self) -> str:
        return stable_hash(
            {
                "integration_version": FINANCIAL_SEMANTIC_INTEGRATION_VERSION,
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
        with self._financial_evidence_lock:
            return tuple(self._financial_runtime_evidence)

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        self._financial_local.state = None
        expected = (
            request.variant is TrialVariant.POLICY_ON
            and request.family == FINANCIAL_SEMANTIC_FAMILY
            and request.program_id == self.expected_program_id
            and request.treatment_hash == self.expected_treatment_hash
            and request.external_skill_source_receipt_hash
            == self.expected_external_skill_source_receipt_hash
        )
        if not expected:
            self.event_sink.emit(
                Event(
                    event="financial_semantic_treatment_identity_rejected_v1",
                    stage="benchmark.skilllearn.financial_semantic_v1",
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
                request, "financial_semantic_treatment_identity_mismatch"
            )
        if expected:
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
                expected_bindings = {
                    "candidate_id": self.planner.asset["candidate_id"],
                    "candidate_manifest_hash": self.planner.asset[
                        "manifest_hash"
                    ],
                    "minilm_runtime_asset_manifest_hash": self.planner.asset[
                        "minilm_runtime_asset_manifest_hash"
                    ],
                    "qa_runtime_asset_manifest_hash": self.planner.asset[
                        "qa_runtime_asset_manifest_hash"
                    ],
                    "operator_source_sha256": self.planner.asset[
                        "operator_source_sha256"
                    ],
                }
                if any(
                    plan.get(key) != value
                    for key, value in expected_bindings.items()
                ):
                    raise FinancialSemanticIntegrationError(
                        "financial semantic plan binding drifted"
                    )
                extraction_body = dict(extraction_receipt)
                extraction_hash = extraction_body.pop("receipt_hash", None)
                if (
                    extraction_hash != stable_hash(extraction_body)
                    or extraction_receipt.get("plan_hash")
                    != plan.get("plan_hash")
                    or extraction_receipt.get("candidate_id")
                    != self.planner.asset["candidate_id"]
                    or extraction_receipt.get("candidate_manifest_hash")
                    != self.planner.asset["manifest_hash"]
                    or extraction_receipt.get("online_calls") != 0
                    or extraction_receipt.get(
                        "operator_created_raw_instruction_artifact"
                    )
                    is not False
                ):
                    raise FinancialSemanticIntegrationError(
                        "financial semantic extraction receipt drifted"
                    )
                self._financial_local.state = _FinancialRunStateV1(
                    request_hash=request.request_hash,
                    plan=plan,
                    extraction_receipt=extraction_receipt,
                )
                self.event_sink.emit(
                    Event(
                        event="financial_semantic_plan_built_before_agent_v1",
                        stage="benchmark.skilllearn.financial_semantic_v1",
                        trace_id=request.request_hash[:20],
                        payload={
                            "request_hash": request.request_hash,
                            "candidate_id": plan["candidate_id"],
                            "candidate_manifest_hash": plan[
                                "candidate_manifest_hash"
                            ],
                            "plan_hash": plan["plan_hash"],
                            "extraction_receipt_hash": extraction_receipt[
                                "receipt_hash"
                            ],
                            "instruction_sha256": plan[
                                "instruction_sha256"
                            ],
                            "raw_instruction_persisted": False,
                            "online_calls": 0,
                        },
                    )
                )
            except Exception as error:
                self.event_sink.emit(
                    Event(
                        event="financial_semantic_plan_build_failed_v1",
                        stage="benchmark.skilllearn.financial_semantic_v1",
                        trace_id=request.request_hash[:20],
                        payload={
                            "request_hash": request.request_hash,
                            "error_type": type(error).__name__,
                            "agent_started": False,
                            "online_calls": 0,
                            "raw_instruction_persisted": False,
                        },
                    )
                )
                return self._local_error(
                    request, "financial_semantic_pre_agent_invalid"
                )
        try:
            observation = super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            state = getattr(self._financial_local, "state", None)
            if (
                expected
                and observation.valid
                and (
                    not isinstance(state, _FinancialRunStateV1)
                    or not isinstance(state.runtime_evidence, Mapping)
                )
            ):
                return replace(
                    observation,
                    success=False,
                    score=0.0,
                    metrics={"evaluation_valid": 0.0},
                    error_type="financial_semantic_runtime_receipt_missing",
                )
            return observation
        finally:
            self._financial_local.state = None

    @staticmethod
    def _run_checked(delegate: Any, command: Sequence[str]) -> Any:
        result = delegate.run(list(command), capture_output=True, text=True)
        if int(getattr(result, "returncode", 1)) != 0:
            raise FinancialSemanticIntegrationError(
                "financial semantic container command failed"
            )
        return result

    def _execute_financial_plan_before_verifier_v1(
        self,
        *,
        delegate: Any,
        container_name: str,
    ) -> None:
        state = getattr(self._financial_local, "state", None)
        if not isinstance(state, _FinancialRunStateV1):
            return
        if state.verifier_triggered:
            if state.runtime_evidence is None:
                raise SkillLearnAgentTerminalError(
                    "financial_semantic_runtime_receipt_invalid"
                )
            return
        state.verifier_triggered = True
        root = Path(tempfile.mkdtemp(prefix="financial-semantic-runtime-v1-"))
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
                "financial_semantic_operator_v1.py"
            )
            expected_source_hash = str(
                self.planner.asset["operator_source_sha256"]
            )
            if hashlib.sha256(operator_source.read_bytes()).hexdigest() != (
                expected_source_hash
            ):
                raise FinancialSemanticIntegrationError(
                    "financial semantic source changed after freeze"
                )
            self._run_checked(
                delegate,
                [
                    "docker",
                    "cp",
                    str(operator_source),
                    f"{container_name}:{_CONTAINER_OPERATOR}",
                ],
            )
            self._run_checked(
                delegate,
                [
                    "docker",
                    "cp",
                    str(plan_path),
                    f"{container_name}:{_CONTAINER_PLAN}",
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
                ],
            )
            observed_source_hash = str(getattr(readback, "stdout", "")).split()[
                0
            ]
            if observed_source_hash != expected_source_hash:
                raise FinancialSemanticIntegrationError(
                    "financial semantic container source drifted"
                )
            self._run_checked(
                delegate,
                [
                    "docker",
                    "exec",
                    container_name,
                    "python3",
                    _CONTAINER_OPERATOR,
                    "execute",
                    "--plan",
                    _CONTAINER_PLAN,
                    "--q2-root",
                    "/root/2025-q2",
                    "--q3-root",
                    "/root/2025-q3",
                    "--output",
                    "/root/answers.json",
                    "--receipt-output",
                    _CONTAINER_RECEIPT,
                ],
            )
            receipt_path = root / "query.receipt.json"
            answers_path = root / "answers.json"
            self._run_checked(
                delegate,
                [
                    "docker",
                    "cp",
                    f"{container_name}:{_CONTAINER_RECEIPT}",
                    str(receipt_path),
                ],
            )
            self._run_checked(
                delegate,
                [
                    "docker",
                    "cp",
                    f"{container_name}:/root/answers.json",
                    str(answers_path),
                ],
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            answers_payload = json.loads(answers_path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise FinancialSemanticIntegrationError(
                    "financial semantic query receipt is malformed"
                )
            if not isinstance(answers_payload, dict):
                raise FinancialSemanticIntegrationError(
                    "financial semantic answers artifact is malformed"
                )
            answers_file_sha256 = hashlib.sha256(
                answers_path.read_bytes()
            ).hexdigest()
            declared = receipt.get("receipt_hash")
            body = dict(receipt)
            body.pop("receipt_hash", None)
            if (
                declared != stable_hash(body)
                or receipt.get("receipt_version")
                != FINANCIAL_QUERY_RECEIPT_VERSION
                or receipt.get("plan_hash") != state.plan.get("plan_hash")
                or receipt.get("candidate_id")
                != self.planner.asset.get("candidate_id")
                or receipt.get("candidate_manifest_hash")
                != self.planner.asset.get("manifest_hash")
                or receipt.get("minilm_runtime_asset_manifest_hash")
                != self.planner.asset.get(
                    "minilm_runtime_asset_manifest_hash"
                )
                or receipt.get("qa_runtime_asset_manifest_hash")
                != self.planner.asset.get("qa_runtime_asset_manifest_hash")
                or receipt.get("operator_source_sha256")
                != expected_source_hash
                or receipt.get("output_sha256") != answers_file_sha256
                or receipt.get("network_calls") != 0
                or receipt.get("verifier_content_accessed") is not False
            ):
                raise FinancialSemanticIntegrationError(
                    "financial semantic query receipt failed closed"
                )
            evidence: dict[str, Any] = {
                "runtime_version": FINANCIAL_SEMANTIC_INTEGRATION_VERSION,
                "request_hash": state.request_hash,
                "candidate_id": self.planner.asset["candidate_id"],
                "candidate_manifest_hash": self.planner.asset[
                    "manifest_hash"
                ],
                "planner_hash": self.planner.planner_hash,
                "backend_instance_hash": self.financial_backend_instance_hash,
                "plan_hash": state.plan["plan_hash"],
                "extraction_receipt_hash": state.extraction_receipt[
                    "receipt_hash"
                ],
                "extraction_receipt": dict(state.extraction_receipt),
                "query_receipt_hash": receipt["receipt_hash"],
                "query_receipt": receipt,
                "output_sha256": receipt["output_sha256"],
                "answers_file_sha256": answers_file_sha256,
                "answers_payload": answers_payload,
                "operator_source_sha256": expected_source_hash,
                "program_id": self.expected_program_id,
                "treatment_hash": self.expected_treatment_hash,
                "external_skill_source_receipt_hash": (
                    self.expected_external_skill_source_receipt_hash
                ),
                "container_operator_readback_sha256": observed_source_hash,
                "executed_after_agent_exit": True,
                "executed_before_verifier_materialization": True,
                "online_calls": 0,
                "raw_instruction_persisted": False,
            }
            evidence["evidence_hash"] = stable_hash(evidence)
            state.runtime_evidence = evidence
            with self._financial_evidence_lock:
                self._financial_runtime_evidence.append(evidence)
            self.event_sink.emit(
                Event(
                    event="financial_semantic_plan_executed_before_verifier_v1",
                    stage="benchmark.skilllearn.financial_semantic_v1",
                    trace_id=state.request_hash[:20],
                    payload=dict(evidence),
                )
            )
        except SkillLearnAgentTerminalError:
            raise
        except Exception as error:
            self.event_sink.emit(
                Event(
                    event="financial_semantic_runtime_failed_v1",
                    stage="benchmark.skilllearn.financial_semantic_v1",
                    trace_id=state.request_hash[:20],
                    payload={
                        "request_hash": state.request_hash,
                        "error_type": type(error).__name__,
                        "verifier_materialized": False,
                        "online_calls": 0,
                        "raw_instruction_persisted": False,
                    },
                )
            )
            raise SkillLearnAgentTerminalError(
                "financial_semantic_runtime_receipt_invalid"
            ) from error
        finally:
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
                        _CONTAINER_PLAN,
                        _CONTAINER_RECEIPT,
                    ],
                )
            except Exception:
                if state.runtime_evidence is not None:
                    raise SkillLearnAgentTerminalError(
                        "financial_semantic_runtime_cleanup_invalid"
                    )
            shutil.rmtree(root, ignore_errors=True)

    @contextmanager
    def _verifier_isolation(
        self,
        runner: Any,
        *,
        agent_runtime_volume: str | None = None,
        egress_policy: DockerEgressPolicy,
        offline_verifier_runtime: OfflineVerifierRuntime | None = None,
        trace_id: str = "financial-semantic-verifier-isolation",
    ) -> Iterator[None]:
        with super()._verifier_isolation(
            runner,
            agent_runtime_volume=agent_runtime_volume,
            egress_policy=egress_policy,
            offline_verifier_runtime=offline_verifier_runtime,
            trace_id=trace_id,
        ):
            base_proxy = runner.subprocess
            runner.subprocess = _FinancialVerifierProxyV1(
                base_proxy,
                backend=self,
            )
            try:
                yield
            finally:
                runner.subprocess = base_proxy

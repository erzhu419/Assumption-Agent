from __future__ import annotations

"""Run eight fresh RAW/contract-candidate pairs with sixteen workers."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
    _configure_environment,
)
from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    SharedFinancialSec13FContractPlannerV2,
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
    sha256_file,
    verify_measurement_view,
)
from replication_runtime.financial_semantic_v2.plan import (
    FixedPeriodOutTreatmentV2,
    MeasurementPlanV2,
    MeasurementTargetV2,
    MeasurementWorkUnitV2,
    build_measurement_plan_v2,
    execute_measurement_plan_v2,
)
from replication_runtime.financial_semantic_v2.recovery import (
    MODEL_EXECUTION_CLAIM_FILENAME,
    SEMANTIC_EVIDENCE_FILENAME,
)
from replication_runtime.financial_semantic_v2 import runner as _legacy

from .backends import (
    DurableFinancialSec13FContractBackendV2,
    DurableRawSubprocessBackendV2,
    backend_runtime_identity_v2,
    future_terminal_semantics_v2,
)
from .hygienic_materialize import (
    MATERIALIZATION_REPORT_NAME,
    measurement_benchmark_tree_receipt_v2,
)
from .hygienic_prewarm import PREWARM_VERSION
from .provider import (
    load_provider_environment_v1,
    validate_execution_provider_binding_v1,
)
from .treatment import (
    FixedContractCandidateV2,
    load_fixed_contract_candidate_v2,
    validate_evaluation_treatment_v2,
)


RUNNER_VERSION = "financial_sec13f_contract_fresh_runner_v2"
EXECUTION_FREEZE_VERSION = "financial_sec13f_contract_execution_freeze_v2"
REPORT_FILENAME = "measurement.report.json"
FAILURE_FILENAME = "measurement.failure.json"
EVENTS_FILENAME = "measurement.events.jsonl"

_FORBIDDEN_DURABLE_CONTRACT_KEYS = frozenset(
    {
        "answers",
        "answers_payload",
        "entity",
        "entity_normalized",
        "entity_raw",
        "expected_output",
        "gold_payload",
        "instruction",
        "operations",
        "plan",
    }
)


class ContractRunnerError(RuntimeError):
    """The fresh contract execution boundary failed closed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_hashed_payload(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or declared != payload_hash(body):
        raise ContractRunnerError(f"{label} self hash mismatch")
    return declared


def _disable_bytecode_writes_v2() -> None:
    """Keep runtime imports from mutating the frozen benchmark tree.

    The isolated contract operator is separately launched with ``python3 -B``;
    this switch covers dynamic imports performed by the host-side benchmark
    loader and verifier preflight.
    """

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if (
        sys.dont_write_bytecode is not True
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ContractRunnerError("bytecode write suppression is unavailable")


def _require_benchmark_tree_hash_v2(
    benchmark_root: Path,
    *,
    expected_tree_hash: object,
    stage: str,
) -> str:
    if not _is_sha256(expected_tree_hash):
        raise ContractRunnerError("frozen benchmark tree hash is malformed")
    actual = measurement_benchmark_tree_receipt_v2(benchmark_root)
    if actual.get("tree_hash") != expected_tree_hash:
        raise ContractRunnerError(f"measurement benchmark tree drifted at {stage}")
    return str(actual["tree_hash"])


def _assert_no_raw_contract_payload_v2(value: Any) -> None:
    """Reject raw typed plans, entities, instructions, or answer payloads.

    Hash-only receipts use explicit flags such as
    ``answers_payload_persisted_in_receipt``; those names remain allowed.  The
    forbidden set targets payload-bearing field names exactly and is applied
    recursively before evidence is copied into a durable aggregate report.
    """

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in _FORBIDDEN_DURABLE_CONTRACT_KEYS:
                raise ContractRunnerError(
                    "durable contract evidence contains raw payload content"
                )
            _assert_no_raw_contract_payload_v2(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_raw_contract_payload_v2(nested)


def _require_regular_output_tree_v2(root: Path) -> None:
    """Reject resume roots that could redirect durable writes."""

    if root.is_symlink() or not root.is_dir():
        raise ContractRunnerError("output root is not a regular directory")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ContractRunnerError(
                "output resume tree contains a link or special file"
            )


def _require_frozen_inputs_v2(
    *,
    project_root: Path,
    benchmark_root: Path,
    measurement_view_path: Path,
    prewarm_path: Path,
    execution_freeze: Mapping[str, Any],
) -> None:
    project = project_root.resolve(strict=True)

    def require_file(section: str, supplied: Path) -> Path:
        binding = execution_freeze.get(section)
        if not isinstance(binding, Mapping):
            raise ContractRunnerError(f"{section} freeze binding is missing")
        relative = binding.get("relative_path")
        expected_hash = binding.get("file_sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_sha256(expected_hash)
        ):
            raise ContractRunnerError(f"{section} freeze binding is malformed")
        expected = project / relative
        if expected.is_symlink() or not expected.is_file():
            raise ContractRunnerError(f"{section} frozen input is not regular")
        resolved_expected = expected.resolve(strict=True)
        try:
            resolved_expected.relative_to(project)
        except ValueError as exc:
            raise ContractRunnerError(f"{section} escaped the project") from exc
        unresolved_supplied = supplied.expanduser()
        if unresolved_supplied.is_symlink() or not unresolved_supplied.is_file():
            raise ContractRunnerError(f"{section} supplied input is not regular")
        resolved_supplied = unresolved_supplied.resolve(strict=True)
        if (
            resolved_supplied != resolved_expected
            or sha256_file(resolved_supplied) != expected_hash
        ):
            raise ContractRunnerError(f"{section} supplied input drifted")
        return resolved_expected

    require_file("measurement_view", measurement_view_path)
    require_file("prewarm", prewarm_path)
    materialization = require_file(
        "materialization",
        benchmark_root / MATERIALIZATION_REPORT_NAME,
    )
    supplied_benchmark = benchmark_root.expanduser()
    if supplied_benchmark.is_symlink() or not supplied_benchmark.is_dir():
        raise ContractRunnerError("measurement benchmark is not a regular tree")
    if supplied_benchmark.resolve(strict=True) != materialization.parent:
        raise ContractRunnerError("measurement benchmark differs from freeze")
    _require_benchmark_tree_hash_v2(
        supplied_benchmark,
        expected_tree_hash=execution_freeze["materialization"].get(
            "benchmark_tree_hash"
        ),
        stage="initial input validation",
    )


class BoundContractPlannerV2:
    """Immutable item-local view over one in-memory contract plan."""

    def __init__(
        self,
        *,
        shared: SharedFinancialSec13FContractPlannerV2,
        instruction_sha256: str,
        plan: Mapping[str, Any],
        extraction_receipt: Mapping[str, Any],
    ) -> None:
        if not _is_sha256(instruction_sha256):
            raise ContractRunnerError("instruction hash is malformed")
        self.asset = shared.asset
        self._instruction_sha256 = instruction_sha256
        self._plan = copy.deepcopy(dict(plan))
        self._receipt = copy.deepcopy(dict(extraction_receipt))
        if (
            self._plan.get("instruction_sha256") != instruction_sha256
            or self._receipt.get("plan_hash") != self._plan.get("plan_hash")
            or not _is_sha256(self._plan.get("plan_hash"))
            or not _is_sha256(self._receipt.get("receipt_hash"))
        ):
            raise ContractRunnerError("precomputed contract plan is inconsistent")
        self._planner_hash = stable_hash(
            {
                "policy": "item_local_precomputed_sec13f_contract_v2",
                "shared_planner_hash": shared.planner_hash,
                "instruction_sha256": instruction_sha256,
                "plan_hash": self._plan["plan_hash"],
                "extraction_receipt_hash": self._receipt["receipt_hash"],
            }
        )

    @property
    def planner_hash(self) -> str:
        return self._planner_hash

    def build(self, instruction: str) -> tuple[dict[str, Any], dict[str, Any]]:
        observed = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        if observed != self._instruction_sha256:
            raise ContractRunnerError("runtime instruction drifted")
        return copy.deepcopy(self._plan), copy.deepcopy(self._receipt)


class ContractRecoveryBoundBackendV2(_legacy.RecoveryBoundBackendV2):
    """Reuse the audited no-replay wrapper with the new candidate type."""

    def __init__(
        self,
        *,
        delegate: Any,
        work: MeasurementWorkUnitV2,
        state_root: Path,
        trial_root: Path,
        expected_process_scope: str,
        expected_plan_hash: str | None = None,
        expected_program_id: str | None = None,
        expected_treatment_hash: str | None = None,
        expected_external_source_receipt_hash: str | None = None,
    ) -> None:
        if work.arm == "raw":
            if not isinstance(delegate, DurableRawSubprocessBackendV2) or any(
                value is not None
                for value in (
                    expected_plan_hash,
                    expected_program_id,
                    expected_treatment_hash,
                    expected_external_source_receipt_hash,
                )
            ):
                raise ContractRunnerError("RAW work crossed candidate state")
        elif work.arm == "candidate":
            if (
                not isinstance(delegate, DurableFinancialSec13FContractBackendV2)
                or not all(
                    _is_sha256(value)
                    for value in (
                        expected_plan_hash,
                        expected_program_id,
                        expected_treatment_hash,
                        expected_external_source_receipt_hash,
                    )
                )
            ):
                raise ContractRunnerError("candidate work crossed RAW state")
        else:
            raise ContractRunnerError("unknown work arm")
        self.delegate = delegate
        self.work = work
        self.state_root = state_root.resolve()
        self.trial_root = trial_root.resolve()
        self.expected_process_scope = str(expected_process_scope)
        self.expected_plan_hash = expected_plan_hash
        self.expected_program_id = expected_program_id
        self.expected_treatment_hash = expected_treatment_hash
        self.expected_external_source_receipt_hash = (
            expected_external_source_receipt_hash
        )
        self._run_lock = threading.Lock()
        self._entered = False
        self._backend_called = False
        self._last_decision = None


def _prewarm_by_item_v2(
    *,
    prewarm: Mapping[str, Any],
    measurement_view_hash: str,
    benchmark_tree_hash: str,
    expected_item_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    if (
        prewarm.get("prewarm_version") != PREWARM_VERSION
        or prewarm.get("measurement_view_hash") != measurement_view_hash
        or prewarm.get("benchmark_tree_hash") != benchmark_tree_hash
        or prewarm.get("pre_prewarm_tree_hash") != benchmark_tree_hash
        or prewarm.get("post_prewarm_tree_hash") != benchmark_tree_hash
        or prewarm.get("benchmark_tree_unchanged") is not True
        or prewarm.get("python_dont_write_bytecode") is not True
        or prewarm.get("python_dont_write_bytecode_env") != "1"
        or prewarm.get("formal_execution_cache_only") is not True
        or prewarm.get("formal_image_cache_only") is not True
        or prewarm.get("formal_offline_verifier_cache_only") is not True
        or prewarm.get("formal_verifier_network") != "none"
        or prewarm.get("model_calls") != 0
        or prewarm.get("online_judge_calls") != 0
        or prewarm.get("sealed_task_count") != 0
        or prewarm.get("sealed_content_accessed") is not False
        or prewarm.get("secret_value_persisted") is not False
    ):
        raise ContractRunnerError("prewarm policy drifted")
    _verify_hashed_payload(prewarm, field="prewarm_hash", label="prewarm")
    rows = prewarm.get("formal_cache_rows")
    expected_ids = tuple(str(item_id) for item_id in expected_item_ids)
    if (
        len(expected_ids) != 8
        or len(set(expected_ids)) != 8
        or not isinstance(rows, list)
        or len(rows) != 8
        or prewarm.get("item_count") != 8
        or prewarm.get("formal_cache_row_set_hash") != payload_hash(rows)
    ):
        raise ContractRunnerError("prewarm rows drifted")
    result = {
        str(row.get("item_id")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(result) != 8 or set(result) != set(expected_ids):
        raise ContractRunnerError("prewarm item identities drifted")
    return result


def build_plan_from_freeze_v2(
    *,
    measurement_view: Mapping[str, Any],
    execution_freeze: Mapping[str, Any],
    candidate: FixedContractCandidateV2,
    protocol: PaperProtocol,
) -> MeasurementPlanV2:
    view = verify_measurement_view(measurement_view)
    treatment_section = execution_freeze.get("treatment")
    expected_treatment_fields = {
        "evaluation_binding",
        "recipe_id",
        "program_set_hash",
        "external_skill_source_receipt_hash",
        "evaluator_epoch",
    }
    if (
        not isinstance(treatment_section, Mapping)
        or set(treatment_section) != expected_treatment_fields
    ):
        raise ContractRunnerError("execution freeze treatment is missing")
    evaluation = treatment_section.get("evaluation_binding")
    if not isinstance(evaluation, Mapping):
        raise ContractRunnerError("evaluation treatment binding is missing")
    validate_evaluation_treatment_v2(evaluation, candidate=candidate)
    source_section = execution_freeze.get("execution_source_closure")
    materialization_section = execution_freeze.get("materialization")
    if (
        not isinstance(source_section, Mapping)
        or not isinstance(materialization_section, Mapping)
        or evaluation.get("measurement_view_hash")
        != view.get("measurement_view_hash")
        or evaluation.get("execution_source_closure_hash")
        != source_section.get("closure_hash")
        or evaluation.get("benchmark_tree_hash")
        != materialization_section.get("benchmark_tree_hash")
        or treatment_section.get("recipe_id") != candidate.recipe_id
        or treatment_section.get("program_set_hash")
        != candidate.program_set_hash
        or treatment_section.get("external_skill_source_receipt_hash")
        != candidate.external_skill_source_receipt_hash
        or not isinstance(treatment_section.get("evaluator_epoch"), str)
        or not str(treatment_section.get("evaluator_epoch")).strip()
    ):
        raise ContractRunnerError("execution freeze candidate identity drifted")
    targets = tuple(
        MeasurementTargetV2(
            item_id=str(item["item_id"]),
            fold_id=f"measurement-fold-{int(item['fold'])}",
        )
        for item in view["measurement_items"]
    )
    plan = build_measurement_plan_v2(
        targets=targets,
        manifest_hash=str(view["measurement_view_hash"]),
        evaluator_epoch=str(treatment_section["evaluator_epoch"]),
        treatment=FixedPeriodOutTreatmentV2(
            recipe_id=candidate.recipe_id,
            program_set_hash=candidate.program_set_hash,
            period_out_treatment_id=str(
                evaluation["period_out_treatment_id"]
            ),
            external_skill_source_receipt_hash=(
                candidate.external_skill_source_receipt_hash
            ),
            candidate_skill_source=candidate.candidate_skill_source,
        ),
        agent_id=str(protocol.payload["agent_id"]),
        model=str(protocol.payload["model"]),
        max_steps=int(protocol.payload["max_steps"]),
        codex_agent_execution_policy_hash=(
            protocol.codex_agent_execution_policy.policy_hash
        ),
    )
    frozen = execution_freeze.get("plan")
    if (
        not isinstance(frozen, Mapping)
        or set(frozen) != {"plan_hash", "safe_payload"}
        or frozen.get("plan_hash") != plan.plan_hash
        or frozen.get("safe_payload") != plan.safe_payload()
    ):
        raise ContractRunnerError("execution plan differs from freeze")
    return plan


def _validate_execution_freeze_v2(
    freeze: Mapping[str, Any],
    *,
    project_root: Path,
    candidate: FixedContractCandidateV2,
    env_file: str | Path,
) -> tuple[str, dict[str, Any]]:
    if freeze.get("manifest_version") != EXECUTION_FREEZE_VERSION:
        raise ContractRunnerError("execution freeze version drifted")
    manifest_hash = _verify_hashed_payload(
        freeze, field="manifest_hash", label="execution freeze"
    )
    candidate_payload = freeze.get("candidate")
    if candidate_payload != candidate.safe_payload(project_root):
        raise ContractRunnerError("execution freeze candidate drifted")
    provider = freeze.get("provider")
    if not isinstance(provider, Mapping):
        raise ContractRunnerError("execution freeze provider drifted")
    try:
        verified_provider = validate_execution_provider_binding_v1(
            provider,
            project_root=project_root,
            env_file=env_file,
        )
    except Exception as exc:
        raise ContractRunnerError(
            "execution freeze provider identity drifted"
        ) from exc
    source = freeze.get("execution_source_closure")
    if not isinstance(source, Mapping) or not _is_sha256(
        source.get("closure_hash")
    ):
        raise ContractRunnerError("execution source closure is missing")
    materialization = freeze.get("materialization")
    if (
        not isinstance(materialization, Mapping)
        or not _is_sha256(materialization.get("benchmark_tree_hash"))
        or not _is_sha256(freeze.get("precomputed_plan_set_hash"))
    ):
        raise ContractRunnerError("execution freeze measurement binding drifted")
    return manifest_hash, verified_provider


def run_measurement_v2(
    *,
    project_root: str | Path,
    benchmark_root: str | Path,
    measurement_view_path: str | Path,
    prewarm_path: str | Path,
    execution_freeze: Mapping[str, Any],
    candidate: FixedContractCandidateV2,
    env_file: str | Path,
    output_root: str | Path,
    recover_only: bool = False,
) -> dict[str, Any]:
    _disable_bytecode_writes_v2()
    project = Path(project_root).expanduser().resolve(strict=True)
    try:
        loaded_provider = load_provider_environment_v1(env_file)
    except Exception as exc:
        raise ContractRunnerError("provider env loading failed closed") from exc
    benchmark_input = Path(benchmark_root).expanduser()
    view_input = Path(measurement_view_path).expanduser()
    prewarm_input = Path(prewarm_path).expanduser()
    freeze_hash, verified_provider = _validate_execution_freeze_v2(
        execution_freeze,
        project_root=project,
        candidate=candidate,
        env_file=env_file,
    )
    if (
        loaded_provider.get("api_key_hmac_sha256")
        != verified_provider.get("api_key_hmac_sha256")
        or loaded_provider.get("model") != verified_provider.get("model")
        or loaded_provider.get("api_origin")
        != verified_provider.get("api_origin")
    ):
        raise ContractRunnerError("current provider identity differs from freeze")
    provider_label = str(verified_provider["provider_label"])
    _require_frozen_inputs_v2(
        project_root=project,
        benchmark_root=benchmark_input,
        measurement_view_path=view_input,
        prewarm_path=prewarm_input,
        execution_freeze=execution_freeze,
    )
    benchmark = benchmark_input.resolve(strict=True)
    view_path = view_input.resolve(strict=True)
    prewarm_file = prewarm_input.resolve(strict=True)
    unresolved_destination = Path(output_root).expanduser()
    if unresolved_destination.is_symlink():
        raise FileExistsError(unresolved_destination)
    destination = unresolved_destination.resolve()
    preexisting = destination.exists()
    if preexisting:
        if not destination.is_dir():
            raise FileExistsError(destination)
        _require_regular_output_tree_v2(destination)
        for name in (
            "execution.plan.json",
            "batch.started.json",
            REPORT_FILENAME,
            FAILURE_FILENAME,
            EVENTS_FILENAME,
        ):
            if (destination / name).is_symlink():
                raise ContractRunnerError("output control receipt is symlinked")
        if (destination / REPORT_FILENAME).exists():
            raise FileExistsError(destination / REPORT_FILENAME)
        existing = {path.name for path in destination.iterdir()}
        if existing and "execution.plan.json" not in existing:
            raise ContractRunnerError("nonempty output lacks resume marker")
    else:
        destination.mkdir(parents=True)
    prior_failure = (destination / FAILURE_FILENAME).is_file()
    execution_started = False
    try:
        protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
        if (
            verified_provider.get("model") != protocol.payload.get("model")
            or verified_provider.get("api_origin")
            != protocol.payload.get("provider_endpoint_origin")
        ):
            raise ContractRunnerError(
                "provider identity differs from frozen protocol"
            )
        _configure_environment(protocol)
        view = verify_measurement_view(read_json(view_path))
        prewarm = read_json(prewarm_file)
        expected_tree_hash = str(
            execution_freeze["materialization"]["benchmark_tree_hash"]
        )
        measurement_item_ids = [
            str(item["item_id"]) for item in view["measurement_items"]
        ]
        prewarm_rows = _prewarm_by_item_v2(
            prewarm=prewarm,
            measurement_view_hash=str(view["measurement_view_hash"]),
            benchmark_tree_hash=expected_tree_hash,
            expected_item_ids=measurement_item_ids,
        )
        plan = build_plan_from_freeze_v2(
            measurement_view=view,
            execution_freeze=execution_freeze,
            candidate=candidate,
            protocol=protocol,
        )
        event_sink = JsonlEventSink(destination / EVENTS_FILENAME)
        planner = SharedFinancialSec13FContractPlannerV2(
            asset_path=candidate.operator_asset_path
        )
        instruction_by_item = {
            str(item["item_id"]): str(item["instruction"])
            for item in view["measurement_items"]
        }
        instruction_hash_by_item = {
            str(item["item_id"]): str(item["instruction_sha256"])
            for item in view["measurement_items"]
        }
        precomputed: dict[str, BoundContractPlannerV2] = {}
        precomputed_receipts: dict[str, Mapping[str, Any]] = {}
        for item_id in sorted(instruction_by_item):
            contract_plan, extraction = planner.build(
                instruction_by_item[item_id]
            )
            bound = BoundContractPlannerV2(
                shared=planner,
                instruction_sha256=instruction_hash_by_item[item_id],
                plan=contract_plan,
                extraction_receipt=extraction,
            )
            precomputed[item_id] = bound
            receipt = {
                "applicable": True,
                "instruction_sha256": instruction_hash_by_item[item_id],
                "plan_hash": contract_plan["plan_hash"],
                "extraction_receipt_hash": extraction["receipt_hash"],
                "planner_hash": bound.planner_hash,
                "raw_plan_persisted": False,
                "model_calls": 0,
                "online_calls": 0,
            }
            _assert_no_raw_contract_payload_v2(receipt)
            precomputed_receipts[item_id] = receipt
        plan_set_hash = stable_hash(
            {
                key: dict(value)
                for key, value in sorted(precomputed_receipts.items())
            }
        )
        frozen_plan_set_hash = execution_freeze.get(
            "precomputed_plan_set_hash"
        )
        if frozen_plan_set_hash != plan_set_hash:
            raise ContractRunnerError("precomputed contract plans differ from freeze")
        _legacy._write_or_verify_hashed_json_v2(
            destination / "execution.plan.json",
            {
                "runner_version": RUNNER_VERSION,
                "execution_freeze_hash": freeze_hash,
                "provider_binding_hash": verified_provider["binding_hash"],
                "provider_label": provider_label,
                "plan_hash": plan.plan_hash,
                "safe_payload": plan.safe_payload(),
                "precomputed_plan_set_hash": plan_set_hash,
                "physical_work_unit_count": 16,
                "raw_plan_persisted": False,
                "model_calls_before_batch": 0,
                "sealed_content_accessed": False,
            },
            hash_field="plan_receipt_hash",
        )
        worker_root = destination / "worker_state"
        if worker_root.is_symlink() or (
            worker_root.exists() and not worker_root.is_dir()
        ):
            raise ContractRunnerError("worker state root is not a regular tree")
        for work in plan.work_units:
            state = worker_root / work.work_unit_hash / "durable"
            semantic = (
                precomputed_receipts[work.target.item_id]
                if work.arm == "candidate"
                else {
                    "applicable": False,
                    "arm": "raw",
                    "model_calls": 0,
                    "online_calls": 0,
                }
            )
            _legacy._ensure_work_state_v2(
                state_root=state,
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
        if recover_only:
            return _legacy.recover_measurement_artifacts_without_model_v2(
                destination=destination,
                worker_root=worker_root,
                plan=plan,
                precomputed_receipts=precomputed_receipts,
                candidate=candidate,  # structural protocol: same required fields
                expected_process_scope=str(
                    protocol.codex_agent_execution_policy.action_budget_process_scope
                ),
                execution_freeze_hash=freeze_hash,
            )
        cache, offline_cache = _legacy._verify_formal_local_cache(
            benchmark_root=benchmark,
            item_ids=[
                work.target.item_id
                for work in plan.work_units
                if work.arm == "raw"
            ],
            prewarm_rows=prewarm_rows,
            event_sink=event_sink,
        )
        if cache.cache_only is not True or not isinstance(
            offline_cache, SkillLearnOfflineVerifierRuntimeCache
        ):
            raise ContractRunnerError("formal run requires cache-only evaluation")
        _require_benchmark_tree_hash_v2(
            benchmark,
            expected_tree_hash=expected_tree_hash,
            stage="cache preflight completion",
        )
        _legacy._write_or_verify_hashed_json_v2(
            destination / "batch.started.json",
            {
                "runner_version": RUNNER_VERSION,
                "execution_freeze_hash": freeze_hash,
                "provider_binding_hash": verified_provider["binding_hash"],
                "provider_label": provider_label,
                "plan_hash": plan.plan_hash,
                "physical_work_unit_count": 16,
                "all_futures_required": True,
                "retry_authorized": False,
                "model_replay_authorized": False,
                "cache_only": True,
                "offline_judge_only": True,
            },
            hash_field="batch_start_hash",
        )
        execution_started = True
        limiter = SkillLearnModelInferenceLimiter(16)
        backends: dict[str, ContractRecoveryBoundBackendV2] = {}

        def backend_factory(work: MeasurementWorkUnitV2) -> Any:
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
                "codex_agent_execution_policy": (
                    protocol.codex_agent_execution_policy
                ),
                "event_sink": event_sink,
                "durable_state_root": state_root,
                "durable_work_unit_hash": work.work_unit_hash,
                "durable_request_hash": work.request.request_hash,
            }
            wrapper_kwargs: dict[str, Any] = {}
            if work.arm == "candidate":
                delegate = DurableFinancialSec13FContractBackendV2(
                    benchmark,
                    planner=precomputed[work.target.item_id],
                    expected_program_id=candidate.recipe_id,
                    expected_program_set_hash=candidate.program_set_hash,
                    expected_treatment_hash=(
                        plan.treatment.period_out_treatment_id
                    ),
                    expected_external_skill_source_receipt_hash=(
                        candidate.external_skill_source_receipt_hash
                    ),
                    expected_precomputed_plan_hash=str(
                        precomputed_receipts[work.target.item_id]["plan_hash"]
                    ),
                    **common,
                )
                wrapper_kwargs = {
                    "expected_plan_hash": str(
                        precomputed_receipts[work.target.item_id]["plan_hash"]
                    ),
                    "expected_program_id": candidate.recipe_id,
                    "expected_treatment_hash": (
                        plan.treatment.period_out_treatment_id
                    ),
                    "expected_external_source_receipt_hash": (
                        candidate.external_skill_source_receipt_hash
                    ),
                }
            else:
                delegate = DurableRawSubprocessBackendV2(benchmark, **common)
            wrapper = ContractRecoveryBoundBackendV2(
                delegate=delegate,
                work=work,
                state_root=state_root,
                trial_root=_legacy._trial_root_for_work_v2(worker_root, work),
                expected_process_scope=str(
                    protocol.codex_agent_execution_policy.action_budget_process_scope
                ),
                **wrapper_kwargs,
            )
            wrapper.inspect_recovery()
            backends[work.work_unit_hash] = wrapper
            return wrapper

        with future_terminal_semantics_v2():
            execution = execute_measurement_plan_v2(
                plan=plan,
                backend_factory=backend_factory,
            )
        descriptive = _legacy._descriptive_results(execution)
        observations = [row.observation for row in execution.work_results]
        semantic_evidence: list[dict[str, Any]] = []
        final_decisions = []
        work_by_hash = {
            work.work_unit_hash: work for work in plan.work_units
        }
        for work_hash, wrapper in sorted(backends.items()):
            work = work_by_hash[work_hash]
            decision = wrapper.inspect_recovery()
            if (
                not decision.completed
                or decision.model_calls_accounted != 1
                or decision.model_replay_authorized
            ):
                raise ContractRunnerError("final durable state is incomplete")
            final_decisions.append(decision)
            state_root = worker_root / work_hash / "durable"
            claim_path = state_root / MODEL_EXECUTION_CLAIM_FILENAME
            if claim_path.is_symlink() or not claim_path.is_file():
                raise ContractRunnerError("completed work lacks model claim")
            evidence_path = state_root / SEMANTIC_EVIDENCE_FILENAME
            live_rows = tuple(
                getattr(wrapper.delegate, "financial_runtime_evidence", ())
            )
            if work.arm == "candidate":
                receipt = read_hashed_json_v2(
                    evidence_path, hash_field="receipt_hash"
                )
                evidence = receipt.get("evidence")
                if not isinstance(evidence, Mapping):
                    raise ContractRunnerError("candidate evidence is unavailable")
                evidence_body = dict(evidence)
                evidence_hash = evidence_body.pop("evidence_hash", None)
                if (
                    set(receipt)
                    != {
                        "request_hash",
                        "evidence",
                        "evidence_hash",
                        "persisted_before_verifier",
                        "raw_plan_persisted",
                        "answers_payload_persisted",
                        "receipt_hash",
                    }
                    or receipt.get("request_hash")
                    != work.request.request_hash
                    or receipt.get("evidence_hash") != evidence_hash
                    or receipt.get("persisted_before_verifier") is not True
                    or receipt.get("raw_plan_persisted") is not False
                    or receipt.get("answers_payload_persisted") is not False
                    or stable_hash(evidence_body) != evidence_hash
                    or evidence.get("plan_hash")
                    != precomputed_receipts[work.target.item_id]["plan_hash"]
                    or evidence.get("program_id") != candidate.recipe_id
                    or evidence.get("treatment_hash")
                    != plan.treatment.period_out_treatment_id
                    or evidence.get("external_skill_source_receipt_hash")
                    != candidate.external_skill_source_receipt_hash
                    or evidence.get("answers_payload_persisted") is not False
                    or evidence.get("raw_instruction_persisted") is not False
                    or evidence.get("raw_entity_persisted_in_durable_evidence")
                    is not False
                    or evidence.get("online_calls") != 0
                ):
                    raise ContractRunnerError("candidate evidence identity drifted")
                _assert_no_raw_contract_payload_v2(evidence)
                if wrapper.backend_called and (
                    len(live_rows) != 1
                    or dict(live_rows[0]) != dict(evidence)
                ):
                    raise ContractRunnerError("live/durable evidence differs")
                semantic_evidence.append(
                    {"work_unit_hash": work_hash, "evidence": dict(evidence)}
                )
            elif evidence_path.exists() or evidence_path.is_symlink() or live_rows:
                raise ContractRunnerError("RAW emitted candidate evidence")
        if len(final_decisions) != 16:
            raise ContractRunnerError("final recovery cardinality drifted")
        _require_benchmark_tree_hash_v2(
            benchmark,
            expected_tree_hash=expected_tree_hash,
            stage="measurement execution completion",
        )
        model_calls_this_invocation = sum(
            wrapper.backend_called for wrapper in backends.values()
        )
        closure = _legacy._artifact_closure(worker_root)
        body = {
            "runner_version": RUNNER_VERSION,
            "execution_completed": True,
            "evidence_valid": (
                descriptive["invalid_pair_count"] == 0
                and len(semantic_evidence) == 8
                and all(
                    isinstance(row, SkillLearnTrialObservation)
                    and row.raw_trial_artifacts_persisted
                    for row in observations
                )
            ),
            "execution_freeze_hash": freeze_hash,
            "measurement_view_hash": view["measurement_view_hash"],
            "prewarm_hash": prewarm["prewarm_hash"],
            "plan_hash": plan.plan_hash,
            "plan": plan.safe_payload(),
            "execution": execution.safe_payload(),
            "results": descriptive,
            "semantic_runtime_evidence": semantic_evidence,
            "semantic_runtime_evidence_set_hash": stable_hash(
                semantic_evidence
            ),
            "worker_artifact_closure": closure,
            "physical_model_call_count": 16,
            "raw_model_call_count": 8,
            "candidate_model_call_count": 8,
            "model_calls_this_invocation": model_calls_this_invocation,
            "recovered_work_count": 16 - model_calls_this_invocation,
            "model_replay_count": 0,
            "recovery_only_invocation": False,
            "resume_invocation": preexisting,
            "prior_failure_receipt_present": prior_failure,
            "model_inference_slot_limit": 16,
            "maximum_concurrent_model_calls": limiter.maximum_active,
            "all_futures_submitted_before_results_read": True,
            "independent_agent_trajectories": True,
            "independent_provider_circuit_count": 16,
            "retry_count": 0,
            "resampling_used": False,
            "mid_batch_provider_switch_used": False,
            "provider_label": provider_label,
            "provider_binding_hash": verified_provider["binding_hash"],
            "provider_identity_sidecar_hash": verified_provider[
                "identity_sidecar_hash"
            ],
            "provider_selection_receipt_hash": verified_provider[
                "selection_receipt_hash"
            ],
            "provider_api_key_commitment_version": verified_provider[
                "api_key_commitment_version"
            ],
            "provider_api_key_hmac_sha256": verified_provider[
                "api_key_hmac_sha256"
            ],
            "plus_transport_failure_before_pro_selection": (
                verified_provider[
                    "plus_transport_failure_before_pro_selection"
                ]
            ),
            "official_hipporag": False,
            "hipporag_status": "not_applicable_nonexecuted",
            "official_hipporag_execution_count": 0,
            "hipporag_proxy_substitution_used": False,
            "offline_evaluation_only": True,
            "benchmark_tree_rehashed_after_execution": True,
            "benchmark_tree_hash": expected_tree_hash,
            "python_dont_write_bytecode": sys.dont_write_bytecode,
            "python_dont_write_bytecode_env": os.environ.get(
                "PYTHONDONTWRITEBYTECODE"
            ),
            "isolated_contract_operator_python_flag": "-B",
            "prebuilt_cache_only": cache.cache_only,
            "offline_judge_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "performance_gate_applied": False,
            "performance_thresholds_bound": False,
            "promotion_authorized": False,
            "incumbent_update_authorized": False,
            "sealed_content_accessed": False,
            "project_authored_period_out_extension": True,
            "official_skilllearnbench_score": False,
            "backend_runtime_identity": backend_runtime_identity_v2(),
            "answers_payload_persisted": False,
            "raw_plan_persisted": False,
            "secret_value_persisted": False,
        }
        return atomic_write_hashed_json_v2(
            destination / REPORT_FILENAME,
            body,
            hash_field="report_hash",
        )
    except Exception as exc:
        failure = {
            "runner_version": RUNNER_VERSION,
            "execution_completed": False,
            "execution_started": execution_started,
            "model_replay_authorized": False,
            "model_replay_count": 0,
            "recovery_only_invocation": recover_only,
            "resume_invocation": preexisting,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "raw_error_persisted": False,
            "secret_value_persisted": False,
        }
        try:
            atomic_write_hashed_json_v2(
                destination / FAILURE_FILENAME,
                failure,
                hash_field="report_hash",
            )
        except FileExistsError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--prewarm", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recover-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project = args.project_root.expanduser().resolve(strict=True)
    candidate = load_fixed_contract_candidate_v2(project)
    freeze = read_json(args.execution_freeze)
    report = run_measurement_v2(
        project_root=project,
        benchmark_root=args.benchmark_root,
        measurement_view_path=args.measurement_view,
        prewarm_path=args.prewarm,
        execution_freeze=freeze,
        candidate=candidate,
        env_file=args.env_file,
        output_root=args.output_root,
        recover_only=args.recover_only,
    )
    if report.get("report_type") == "artifact_recovery_only":
        print(
            json.dumps(
                {
                    "report_hash": report["report_hash"],
                    "recovery_completed": report["recovery_completed"],
                    "completed_work_unit_count": report[
                        "completed_work_unit_count"
                    ],
                    "unresolved_work_unit_count": report[
                        "unresolved_work_unit_count"
                    ],
                    "model_calls_this_invocation": 0,
                },
                sort_keys=True,
            )
        )
        return 0 if report["recovery_completed"] else 2
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "evidence_valid": report["evidence_valid"],
                "raw_successes": report["results"]["raw_successes"],
                "candidate_successes": report["results"][
                    "candidate_successes"
                ],
                "net_delta": report["results"]["net_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

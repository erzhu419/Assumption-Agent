from __future__ import annotations

"""No-model Docker diagnostic for the SEC-13F durable operator boundary.

This executable deliberately uses a synthetic four-file SEC fixture.  It does
not open a benchmark, measurement view, pack, oracle, gold file, or verifier.
The only repository data input is the public frozen contract asset.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import MethodType
from typing import Any, Mapping, Sequence
import uuid


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from assumption_agent.benchmarks import financial_sec13f_contract_operator_v2 as op
from assumption_agent.benchmarks import financial_sec13f_contract_integration_v2 as integration
from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    SharedFinancialSec13FContractPlannerV2,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import SplitName, stable_hash
from replication_runtime.financial_sec13f_contract_v2.backends import (
    DurableFinancialSec13FContractBackendV2,
    initialize_work_state_v2,
)
from replication_runtime.financial_sec13f_contract_v2.runner import (
    BoundContractPlannerV2,
)


DIAGNOSTIC_VERSION = (
    "financial_sec13f_contract_durable_operator_docker_diagnostic_v1"
)
ASSET_PATH = (
    PROJECT / "manifests" / "financial_sec13f_public_contract_asset_v2.json"
)
DEFAULT_IMAGE = "python:3.11-slim"
PREVIOUS_ROOT = "/root/diagnostic-previous"
CURRENT_ROOT = "/root/diagnostic-current"


class DiagnosticError(RuntimeError):
    """The isolated no-model diagnostic failed closed."""


class _EventSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)


class _DockerDelegate:
    """Exact subprocess surface consumed by the production integration."""

    def run(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        argv = [str(value) for value in command]
        if not argv or argv[0] != "docker":
            raise DiagnosticError("diagnostic delegate rejected a non-Docker command")
        return subprocess.run(argv, check=False, **kwargs)


def _run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(value) for value in command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DiagnosticError(
            f"diagnostic command failed: {command[0]} {command[1]}"
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_instruction() -> str:
    return (
        "You are a financial analyst comparing official SEC Form 13F data "
        "for diagnostic-current against diagnostic-previous. The previous "
        f"data is in `{PREVIOUS_ROOT}` and current data is in `{CURRENT_ROOT}`.\n\n"
        f"Frozen data semantics: {op._semantic_contract_text()}\n\n"
        "Questions:\n\n"
        "1. What is the current-period AUM of Alpha Capital?\n\n"
        "2. How many stock rows are held by Alpha Capital in the current period?\n\n"
        "3. What are the top 5 CUSIPs with increased investment by Beta "
        "Partners from the previous period to the current period, ranked by "
        "dollar-value increase?\n\n"
        "4. Which top 3 fund managers hold ACME, Inc. in the current period, "
        "ranked by aggregate position value?\n\n"
        "Write `/root/answers.json` with keys `q1_answer`, `q2_answer`, "
        "`q3_answer`, and `q4_answer` in that order. q1 and q2 are numbers; "
        "q3 and q4 are ordered JSON arrays.\n"
    )


def _write_synthetic_tables(root: Path) -> tuple[Path, Path]:
    previous = root / "previous"
    current = root / "current"
    previous.mkdir()
    current.mkdir()
    cover_header = (
        "ACCESSION_NUMBER\tREPORTCALENDARORQUARTER\tREPORTTYPE\t"
        "FILINGMANAGER_NAME\n"
    )
    info_header = "ACCESSION_NUMBER\tNAMEOFISSUER\tTITLEOFCLASS\tCUSIP\tVALUE\n"
    (previous / "COVERPAGE.tsv").write_text(
        cover_header
        + "P-ALPHA\t31-DEC-2025\t13F-HR\tAlpha Capital\n"
        + "P-BETA\t31-DEC-2025\t13F-HR\tBeta Partners\n",
        encoding="utf-8",
    )
    (current / "COVERPAGE.tsv").write_text(
        cover_header
        + "C-ALPHA\t31-MAR-2026\t13F-HR\tAlpha Capital\n"
        + "C-BETA\t31-MAR-2026\t13F-HR\tBeta Partners\n"
        + "C-GAMMA\t31-MAR-2026\t13F-HR\tGamma Advisors\n"
        + "C-DELTA\t31-MAR-2026\t13F-HR\tDelta Management\n",
        encoding="utf-8",
    )
    (previous / "INFOTABLE.tsv").write_text(
        info_header
        + "P-ALPHA\tACME, Inc.\tCOM\t999999999\t800\n"
        + "P-BETA\tIssuer One\tCOM\t111111111\t100\n"
        + "P-BETA\tIssuer Two\tCOM\t222222222\t50\n",
        encoding="utf-8",
    )
    (current / "INFOTABLE.tsv").write_text(
        info_header
        + "C-ALPHA\tACME, Inc.\tCOM\t999999999\t1000\n"
        + "C-BETA\tACME, Inc.\tCOM\t999999999\t500\n"
        + "C-BETA\tIssuer One\tCOM\t111111111\t150\n"
        + "C-BETA\tIssuer Two\tCOM\t222222222\t40\n"
        + "C-BETA\tIssuer Three\tCOM\t333333333\t60\n"
        + "C-GAMMA\tACME, Inc.\tCOM\t999999999\t700\n"
        + "C-DELTA\tACME, Inc.\tCOM\t999999999\t300\n",
        encoding="utf-8",
    )
    return previous, current


def _diagnostic_agent_completion(
    self: DurableFinancialSec13FContractBackendV2,
    request: SkillLearnTrialRequest,
    *,
    reconciled_after_backend_return: bool,
) -> dict[str, Any]:
    if request.request_hash != self.durable_request_hash:
        raise DiagnosticError("synthetic agent boundary request drifted")
    return {
        "arm": "candidate",
        "diagnostic_synthetic_agent_boundary": True,
        "model_calls": 0,
        "network_calls": 0,
        "reconciled_after_backend_return": reconciled_after_backend_return,
        "raw_trace_persisted_in_stage": False,
    }


def _copy_tables_into_container(
    *, container_name: str, previous: Path, current: Path
) -> None:
    _run_checked(
        [
            "docker",
            "exec",
            container_name,
            "mkdir",
            "-p",
            PREVIOUS_ROOT,
            CURRENT_ROOT,
        ]
    )
    for source_root, destination_root in (
        (previous, PREVIOUS_ROOT),
        (current, CURRENT_ROOT),
    ):
        for table in ("COVERPAGE.tsv", "INFOTABLE.tsv"):
            _run_checked(
                [
                    "docker",
                    "cp",
                    str(source_root / table),
                    f"{container_name}:{destination_root}/{table}",
                ]
            )


def run_diagnostic(*, image: str = DEFAULT_IMAGE) -> dict[str, Any]:
    """Run the real Docker operator hook without an agent or verifier."""

    if not shutil.which("docker"):
        raise DiagnosticError("Docker CLI is unavailable")
    image_info = _run_checked(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    )
    image_id = image_info.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise DiagnosticError("cached Docker image identity is malformed")

    shared = SharedFinancialSec13FContractPlannerV2(asset_path=ASSET_PATH)
    instruction = _synthetic_instruction()
    plan, extraction_receipt = shared.build(instruction)
    bound = BoundContractPlannerV2(
        shared=shared,
        instruction_sha256=plan["instruction_sha256"],
        plan=plan,
        extraction_receipt=extraction_receipt,
    )
    if bound.asset_path != ASSET_PATH.resolve(strict=True):
        raise DiagnosticError("bound planner did not retain the frozen asset path")
    rebuilt_plan, rebuilt_receipt = bound.build(instruction)
    if rebuilt_plan != plan or rebuilt_receipt != extraction_receipt:
        raise DiagnosticError("bound planner changed its precomputed payload")

    request_hash_seed = stable_hash(
        {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "plan_hash": plan["plan_hash"],
            "image_id": image_id,
        }
    )
    program_id = stable_hash({"diagnostic": "program"})
    program_set_hash = stable_hash({"diagnostic": "program-set"})
    treatment_hash = stable_hash({"diagnostic": "treatment"})
    source_receipt_hash = stable_hash({"diagnostic": "source-receipt"})
    execution_policy_hash = stable_hash({"diagnostic": "execution-policy"})

    backend = object.__new__(DurableFinancialSec13FContractBackendV2)
    backend.planner = bound
    backend.expected_precomputed_plan_hash = plan["plan_hash"]
    backend.expected_program_id = program_id
    backend.expected_program_set_hash = program_set_hash
    backend.expected_treatment_hash = treatment_hash
    backend.expected_external_skill_source_receipt_hash = source_receipt_hash
    backend.agent_id = "offline-diagnostic"
    backend.model = "none"
    backend.max_steps = 0
    backend.codex_agent_execution_policy_hash = execution_policy_hash
    backend.durable_arm = "candidate"
    backend._contract_local = threading.local()
    backend._contract_evidence_lock = threading.Lock()
    backend._contract_runtime_evidence = []
    backend.event_sink = _EventSink()
    backend._agent_completion_payload = MethodType(
        _diagnostic_agent_completion, backend
    )

    request = SkillLearnTrialRequest(
        item_id="synthetic-sec13f-durable-operator-diagnostic",
        family="financial-analysis",
        split=SplitName.VALIDATION,
        variant=TrialVariant.POLICY_ON,
        evaluator_epoch=DIAGNOSTIC_VERSION,
        pair_id="synthetic-sec13f-durable-operator-diagnostic-pair",
        repeat=0,
        agent_id=backend.agent_id,
        model=backend.model,
        max_steps=backend.max_steps,
        manifest_hash=request_hash_seed,
        codex_agent_execution_policy_hash=execution_policy_hash,
        program_id=program_id,
        program_set_hash=program_set_hash,
        treatment_hash=treatment_hash,
        external_skill_source_receipt_hash=source_receipt_hash,
    )
    backend.durable_request_hash = request.request_hash
    backend.durable_work_unit_hash = stable_hash(
        {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "request_hash": request.request_hash,
        }
    )
    backend._active_request = request

    container_name = f"sec13f-contract-diagnostic-{uuid.uuid4().hex[:12]}"
    delegate = _DockerDelegate()
    with tempfile.TemporaryDirectory(
        prefix="financial-sec13f-contract-durable-diagnostic-"
    ) as temporary:
        temporary_root = Path(temporary)
        previous, current = _write_synthetic_tables(temporary_root)
        backend.durable_state_root = temporary_root / "durable"
        initialize_work_state_v2(
            state_root=backend.durable_state_root,
            work_unit_hash=backend.durable_work_unit_hash,
            request_hash=request.request_hash,
            planned_payload={
                "arm": "candidate",
                "diagnostic": True,
                "model_calls": 0,
            },
            semantic_plan_payload={
                "applicable": True,
                "diagnostic": True,
                "plan_hash": plan["plan_hash"],
                "model_calls": 0,
            },
        )
        state = integration._ContractRunStateV2(
            request_hash=request.request_hash,
            plan=plan,
            extraction_receipt=extraction_receipt,
        )
        backend._contract_local.state = state
        try:
            _run_checked(
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
                    container_name,
                    image,
                    "sleep",
                    "300",
                ]
            )
            network_mode = _run_checked(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.HostConfig.NetworkMode}}",
                    container_name,
                ]
            ).stdout.strip()
            if network_mode != "none":
                raise DiagnosticError("diagnostic container is not network-isolated")
            _copy_tables_into_container(
                container_name=container_name,
                previous=previous,
                current=current,
            )
            backend._execute_contract_plan_before_verifier_v2(
                delegate=delegate,
                container_name=container_name,
            )
            chain = backend._durable_chain()
            stages = [row.stage for row in chain]
            if stages != [
                "planned",
                "semantic_plan_ready",
                "agent_completed",
                "operator_completed",
            ]:
                raise DiagnosticError("durable operator stage chain is incomplete")
            evidence = state.runtime_evidence
            if not isinstance(evidence, Mapping):
                raise DiagnosticError("runtime evidence was not produced")
            if (
                evidence.get("online_calls") != 0
                or evidence.get("gold_content_accessed") is not False
                or evidence.get("pack_content_accessed") is not False
                or evidence.get("executed_before_verifier_materialization")
                is not True
            ):
                raise DiagnosticError("runtime evidence crossed the diagnostic boundary")
            evidence_receipt = (
                backend.durable_state_root / "semantic_runtime_evidence.json"
            )
            if not evidence_receipt.is_file():
                raise DiagnosticError("durable runtime evidence receipt is missing")
            event_names = [
                str(getattr(event, "event", ""))
                for event in backend.event_sink.events
            ]
            if "financial_sec13f_contract_executed_v2" not in event_names:
                raise DiagnosticError("production operator event was not emitted")
            stage_hashes = [row.stage_hash for row in chain]
            receipt = {
                "diagnostic_version": DIAGNOSTIC_VERSION,
                "status": "passed",
                "fixture_provenance": "inline_synthetic_only_v1",
                "repository_data_inputs": [
                    str(ASSET_PATH.relative_to(PROJECT))
                ],
                "measurement_item_content_accessed": False,
                "benchmark_pack_accessed": False,
                "gold_or_oracle_accessed": False,
                "verifier_invoked": False,
                "model_calls": 0,
                "online_calls": 0,
                "container_network_mode": network_mode,
                "cached_image_id": image_id,
                "instruction_sha256": plan["instruction_sha256"],
                "plan_hash": plan["plan_hash"],
                "bound_planner_hash": bound.planner_hash,
                "bound_asset_file_sha256": _sha256_file(bound.asset_path),
                "durable_stage_names": stages,
                "durable_stage_set_hash": stable_hash(stage_hashes),
                "runtime_evidence_hash": evidence["evidence_hash"],
                "runtime_evidence_receipt_sha256": _sha256_file(
                    evidence_receipt
                ),
                "production_operator_event_emitted": True,
                "raw_instruction_persisted": False,
                "answers_payload_persisted": False,
            }
            receipt["receipt_hash"] = stable_hash(receipt)
            return receipt
        finally:
            backend._active_request = None
            subprocess.run(
                ["docker", "rm", "--force", container_name],
                check=False,
                capture_output=True,
                text=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_diagnostic(image=args.image)
    rendered = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

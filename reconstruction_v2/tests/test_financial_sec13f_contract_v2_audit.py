from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from assumption_agent.benchmarks import financial_sec13f_contract_operator_v2 as op
from assumption_agent.benchmarks import financial_sec13f_contract_integration_v2 as integration
from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    INTEGRATION_VERSION,
    SharedFinancialSec13FContractPlannerV2,
)
from assumption_agent.models import stable_hash
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnAgentTerminalError,
)
from replication_runtime.financial_sec13f_contract_v2 import formation
from replication_runtime.financial_sec13f_contract_v2.backends import (
    DurableFinancialSec13FContractBackendV2,
    FinancialSemanticReplicationBackendError,
)


PROJECT = Path(__file__).resolve().parents[1]


def _hash(label: str) -> str:
    return stable_hash({"financial-sec13f-contract-v2-audit": label})


def _runtime_fixture() -> tuple[
    DurableFinancialSec13FContractBackendV2, SimpleNamespace, dict
]:
    planner = SharedFinancialSec13FContractPlannerV2(
        asset_path=(
            PROJECT
            / "manifests"
            / "financial_sec13f_public_contract_asset_v2.json"
        )
    )
    instruction = (
        "You are a financial analyst comparing official SEC Form 13F data "
        "for 2026Q1 against 2025Q4. The previous data is in "
        "`/root/2025-q2` and current data is in `/root/2025-q3`.\n\n"
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
    plan, extraction = planner.build(instruction)
    input_receipts = [
        {
            "role": role,
            "table": table,
            "size_bytes": index + 1,
            "file_sha256": _hash(f"{role}-{table}"),
        }
        for index, (role, table) in enumerate(
            (
                ("previous", "COVERPAGE.tsv"),
                ("previous", "INFOTABLE.tsv"),
                ("current", "COVERPAGE.tsv"),
                ("current", "INFOTABLE.tsv"),
            )
        )
    ]
    output_sha256 = _hash("output")
    query_body = {
        "receipt_version": op.QUERY_RECEIPT_VERSION,
        "operator_version": op.OPERATOR_VERSION,
        "candidate_id": planner.asset["candidate_id"],
        "asset_manifest_hash": planner.asset["manifest_hash"],
        "contract_hash": planner.asset["contract_hash"],
        "operator_source_sha256": planner.asset["operator_source_sha256"],
        "plan_hash": plan["plan_hash"],
        "numeric_engine": op.NUMERIC_ENGINE,
        "input_file_receipts": input_receipts,
        "input_set_hash": op.payload_hash(input_receipts),
        "pre_output_exists": False,
        "pre_output_sha256": None,
        "post_output_sha256": output_sha256,
        "output_changed": True,
        "answer_key_set_hash": op.payload_hash(
            ["q1_answer", "q2_answer", "q3_answer", "q4_answer"]
        ),
        "answers_payload_persisted_in_receipt": False,
        "raw_entity_persisted_in_receipt": False,
        "network_calls": 0,
        "model_calls": 0,
        "verifier_content_accessed": False,
        "gold_content_accessed": False,
        "pack_content_accessed": False,
    }
    query = {**query_body, "receipt_hash": op.payload_hash(query_body)}

    backend = object.__new__(DurableFinancialSec13FContractBackendV2)
    backend.planner = planner
    backend.durable_request_hash = _hash("request")
    backend.expected_precomputed_plan_hash = plan["plan_hash"]
    backend.expected_program_id = _hash("program")
    backend.expected_treatment_hash = _hash("treatment")
    backend.expected_external_skill_source_receipt_hash = _hash("source")
    backend.agent_id = "codex"
    backend.model = "offline-test-model"
    backend.max_steps = 100
    backend.codex_agent_execution_policy_hash = _hash("execution-policy")
    state = SimpleNamespace(
        request_hash=backend.durable_request_hash,
        plan=plan,
    )
    plan_bytes = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    evidence_body = {
        "runtime_version": INTEGRATION_VERSION,
        "request_hash": backend.durable_request_hash,
        "candidate_id": planner.asset["candidate_id"],
        "asset_manifest_hash": planner.asset["manifest_hash"],
        "contract_hash": planner.asset["contract_hash"],
        "planner_hash": planner.planner_hash,
        "backend_instance_hash": backend.financial_backend_instance_hash,
        "plan_hash": plan["plan_hash"],
        "extraction_receipt_hash": extraction["receipt_hash"],
        "extraction_receipt": extraction,
        "query_receipt_hash": query["receipt_hash"],
        "query_receipt": query,
        "output_sha256": output_sha256,
        "answers_payload_persisted": False,
        "operator_source_sha256": planner.asset["operator_source_sha256"],
        "program_id": backend.expected_program_id,
        "treatment_hash": backend.expected_treatment_hash,
        "external_skill_source_receipt_hash": (
            backend.expected_external_skill_source_receipt_hash
        ),
        "container_operator_readback_sha256": planner.asset[
            "operator_source_sha256"
        ],
        "container_asset_readback_sha256": op.sha256_file(
            planner.asset_path
        ),
        "container_plan_readback_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "executed_after_agent_exit": True,
        "executed_before_verifier_materialization": True,
        "online_calls": 0,
        "raw_instruction_persisted": False,
        "raw_entity_persisted_in_durable_evidence": False,
        "ephemeral_plan_deleted_before_verifier": True,
        "gold_content_accessed": False,
        "pack_content_accessed": False,
    }
    evidence = {**evidence_body, "evidence_hash": stable_hash(evidence_body)}
    return backend, state, evidence


def test_durable_runtime_evidence_recomputes_hash_and_exact_schema() -> None:
    backend, state, evidence = _runtime_fixture()
    assert backend._validated_runtime_evidence_v2(
        evidence, state=state
    ) == evidence

    wrong_hash = copy.deepcopy(evidence)
    wrong_hash["evidence_hash"] = "0" * 64
    with pytest.raises(
        FinancialSemanticReplicationBackendError, match="evidence hash drifted"
    ):
        backend._validated_runtime_evidence_v2(wrong_hash, state=state)

    extra_field = copy.deepcopy(evidence)
    extra_field["debug_payload"] = "not part of the frozen schema"
    body = dict(extra_field)
    body.pop("evidence_hash")
    extra_field["evidence_hash"] = stable_hash(body)
    with pytest.raises(
        FinancialSemanticReplicationBackendError, match="schema drifted"
    ):
        backend._validated_runtime_evidence_v2(extra_field, state=state)


def test_durable_runtime_evidence_recursively_rejects_raw_answers() -> None:
    backend, state, evidence = _runtime_fixture()
    nested_raw = copy.deepcopy(evidence)
    nested_raw["query_receipt"]["diagnostic"] = {
        "answers_payload": {"q1_answer": 123}
    }
    query_body = dict(nested_raw["query_receipt"])
    query_body.pop("receipt_hash")
    nested_raw["query_receipt"]["receipt_hash"] = op.payload_hash(query_body)
    nested_raw["query_receipt_hash"] = nested_raw["query_receipt"][
        "receipt_hash"
    ]
    body = dict(nested_raw)
    body.pop("evidence_hash")
    nested_raw["evidence_hash"] = stable_hash(body)

    with pytest.raises(
        FinancialSemanticReplicationBackendError, match="forbidden raw content"
    ):
        backend._validated_runtime_evidence_v2(nested_raw, state=state)


def test_final_formation_requires_the_live_preregistration_content(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    preregistration_path = project / "fresh-preregistration.json"
    formation.write_json(preregistration_path, {"manifest_hash": "a" * 64})

    with pytest.raises(
        formation.FreshFormationError, match="live preregistration content"
    ):
        formation.form_fresh_pack_v1(
            project_root=project,
            preregistration_path=preregistration_path,
            prior_measurement_view_path=tmp_path / "prior-view.json",
            preregistration={"manifest_hash": "b" * 64},
            acquisition={},
            previous_archive=tmp_path / "previous.zip",
            current_archive=tmp_path / "current.zip",
            prior_measurement_view={},
        )


def test_final_formation_forwards_live_paths_to_both_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    preregistration_path = project / "fresh-preregistration.json"
    preregistration = {"manifest_hash": "a" * 64}
    formation.write_json(preregistration_path, preregistration)
    calls: dict[str, object] = {}

    def _validate_preregistration(value: object, *, project_root: Path) -> str:
        calls["prereg_project"] = project_root
        return "a" * 64

    def _validate_acquisition(
        value: object,
        *,
        preregistration: object,
        project_root: Path,
        preregistration_path: Path,
        previous_archive: Path,
        current_archive: Path,
    ) -> str:
        calls.update(
            {
                "acquisition_project": project_root,
                "preregistration_path": preregistration_path,
            }
        )
        raise formation.FreshFormationError("stop after live validation")

    monkeypatch.setattr(
        formation, "validate_preregistration_v1", _validate_preregistration
    )
    monkeypatch.setattr(
        formation, "validate_acquisition_receipt_v1", _validate_acquisition
    )
    with pytest.raises(formation.FreshFormationError, match="stop after"):
        formation.form_fresh_pack_v1(
            project_root=project,
            preregistration_path=preregistration_path,
            prior_measurement_view_path=tmp_path / "prior-view.json",
            preregistration=preregistration,
            acquisition={},
            previous_archive=tmp_path / "previous.zip",
            current_archive=tmp_path / "current.zip",
            prior_measurement_view={},
        )
    assert calls == {
        "prereg_project": project.resolve(),
        "acquisition_project": project.resolve(),
        "preregistration_path": preregistration_path.resolve(),
    }


def test_host_plan_cleanup_verifies_that_the_root_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ephemeral-plan"
    root.mkdir()
    (root / "plan.json").write_text('{"entity":"raw"}\n', encoding="utf-8")

    with monkeypatch.context() as context:
        context.setattr(integration.shutil, "rmtree", lambda path: None)
        with pytest.raises(
            integration.FinancialSec13FContractIntegrationError,
            match="not confirmed",
        ):
            integration._remove_ephemeral_host_root_v2(root)
    assert root.is_dir()
    integration._remove_ephemeral_host_root_v2(root)
    assert not root.exists()


def test_cleanup_failure_never_publishes_deleted_plan_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, base_state, evidence = _runtime_fixture()
    state = integration._ContractRunStateV2(
        request_hash=backend.durable_request_hash,
        plan=dict(base_state.plan),
        extraction_receipt=evidence["extraction_receipt"],
    )
    backend._contract_local = threading.local()
    backend._contract_local.state = state
    backend._contract_evidence_lock = threading.Lock()
    backend._contract_runtime_evidence = []

    class _Sink:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            self.events.append(event)

    sink = _Sink()
    backend.event_sink = sink
    query_receipt = evidence["query_receipt"]
    output_sha256 = evidence["output_sha256"]

    class _Delegate:
        def __init__(self) -> None:
            self.container_files: dict[str, bytes] = {}

        def run(self, args: list[str], **kwargs: object) -> SimpleNamespace:
            if args[:2] == ["docker", "cp"]:
                source, destination = args[2], args[3]
                if source.startswith("fixture:"):
                    Path(destination).write_text(
                        json.dumps(query_receipt), encoding="utf-8"
                    )
                else:
                    container_path = destination.split(":", 1)[1]
                    self.container_files[container_path] = Path(source).read_bytes()
                return SimpleNamespace(returncode=0, stdout="")
            if "sha256sum" in args:
                paths = args[args.index("sha256sum") + 1 :]
                if paths == ["/root/answers.json"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=f"{output_sha256}  /root/answers.json\n",
                    )
                stdout = "".join(
                    f"{hashlib.sha256(self.container_files[path]).hexdigest()}  {path}\n"
                    for path in paths
                )
                return SimpleNamespace(returncode=0, stdout=stdout)
            return SimpleNamespace(returncode=0, stdout="")

    host_root = tmp_path / "host-plan-root"
    host_root.mkdir()
    monkeypatch.setattr(
        integration.tempfile,
        "mkdtemp",
        lambda prefix: str(host_root),
    )

    def _cleanup_failure(root: Path) -> None:
        raise integration.FinancialSec13FContractIntegrationError(
            "simulated cleanup failure"
        )

    monkeypatch.setattr(
        integration, "_remove_ephemeral_host_root_v2", _cleanup_failure
    )
    with pytest.raises(
        SkillLearnAgentTerminalError, match="runtime_cleanup_invalid"
    ):
        integration.FinancialSec13FContractSubprocessBackendV2._execute_contract_plan_before_verifier_v2(
            backend,
            delegate=_Delegate(),
            container_name="fixture",
        )
    assert state.runtime_evidence is None
    assert backend._contract_runtime_evidence == []
    assert not any(
        getattr(event, "event", None)
        == "financial_sec13f_contract_executed_v2"
        for event in sink.events
    )
    assert (host_root / "plan.json").is_file()

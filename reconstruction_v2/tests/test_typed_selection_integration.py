from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from assumption_agent.archive import PolicyArchive
from assumption_agent.benchmarks.skilllearn_compiler import (
    SkillLearnProgramCompiler,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnCounterfactualRunner,
    SkillLearnEvolutionHarness,
    SkillLearnExternalEvaluator,
)
from assumption_agent.benchmarks import skilllearn_experiment
from assumption_agent.benchmarks import paper_protocol as paper_protocol_module
from assumption_agent.benchmarks.paper_protocol import (
    PaperProtocol,
    authorize_typed_selection_execution,
)
from assumption_agent.benchmarks.typed_selection_integration import (
    TYPED_SELECTION_INTEGRATION_VERSION,
    _validate_completed_result_artifacts,
)
from assumption_agent.benchmarks.typed_portable_integration import (
    TYPED_PORTABLE_INTEGRATION_VERSION,
)
from assumption_agent.benchmarks.typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.evolution import (
    EvolutionKernel,
    TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION,
)
from assumption_agent.models import (
    HypothesisKind,
    HypothesisProgram,
    HypothesisStatus,
    ResidualExample,
    SplitName,
    stable_hash,
)
from assumption_agent.proposer import StructuredHypothesisProposer
from assumption_agent.splits import BenchmarkItem, SplitManifest
from assumption_agent.typed_operator_grammar import (
    PrimitiveRef,
    TYPED_SELECTION_FREEZE_AUTHORIZATION_VERSION,
    TrialTraceEvidence,
    TypedProgramBindingRegistry,
    TypedSelectionExecutionAuthorization,
    TypedSelectionFreezeAuthorization,
    build_family_capability_graph,
    freeze_typed_selection_snapshot_ledger,
    freeze_typed_recipe_selection_snapshot,
    validate_typed_selection_history_payloads,
)
from assumption_agent.validation import (
    CheckResult,
    RecursiveValidationEngine,
    ValidationContext,
)


class _RecipeSelectionModel:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = dict(payload)
        self.requests.append(request)
        enum = request["output_schema"]["properties"]["recipe_id"]["enum"]
        return {"recipe_id": enum[0]}


class _RootOnlyFailureCheck:
    name = "forced_root_only_failure"

    def __init__(self, root_id: str) -> None:
        self.root_id = root_id

    def evaluate(
        self,
        program: HypothesisProgram,
        context: ValidationContext,
    ) -> CheckResult:
        passed = program.id != self.root_id
        return CheckResult(
            check=self.name,
            passed=passed,
            reason="alternate_recipe_selected" if passed else "root_recipe_rejected",
            evidence={"program_hash": program.payload_hash},
        )


def _snapshot_fixture():
    residuals: list[ResidualExample] = []
    action_profiles: dict[str, Mapping[str, Any]] = {}
    trial_evidence: dict[str, TrialTraceEvidence] = {}
    snapshots = []
    for family_index, family in enumerate(("family-a", "family-b", "family-c"), start=1):
        path = f"/root/task-{family_index}/input-{family_index}.csv"
        profile = {
            "runtime_environment": {
                "declared_task_local_paths": [path],
            },
            "action_trace": {},
        }
        profile_hash = stable_hash(profile)
        action_profiles[profile_hash] = profile
        family_residuals: list[ResidualExample] = []
        for item_index in range(2):
            item_id = f"{family}-item-{item_index + 1}"
            residual = ResidualExample(
                transition_id=f"transition-{family_index}-{item_index + 1}",
                task_id=item_id,
                family=family,
                split=SplitName.TRAIN,
                features={"family": family},
                failure_type="synthetic_train_failure",
                evaluator_feedback="",
                baseline_success=False,
                context={"action_context_profile_hash": profile_hash},
            )
            family_residuals.append(residual)
            residuals.append(residual)
            trial_hash = stable_hash({"item_id": item_id})
            trial_evidence[trial_hash] = TrialTraceEvidence(
                trial_id_hash=trial_hash,
                family_hash=stable_hash({"family": family}),
                trace_hash=stable_hash({"trace": item_id}),
                action_budget_receipt_hash=stable_hash({"receipt": item_id}),
                action_event_hash=stable_hash({"events": item_id}),
                baseline_success=False,
                action_budget_limit=100,
                trace_complete=True,
                action_start_count=1,
                command_span_count=0,
                discarded_command_count=0,
                changed_artifacts=(PrimitiveRef("artifact", path),),
                spans=(),
            )
        graph = build_family_capability_graph(
            target_family=family,
            failures=family_residuals,
            action_profiles=action_profiles,
            trial_evidence=trial_evidence,
        )
        snapshots.append(freeze_typed_recipe_selection_snapshot(graph))
    return tuple(residuals), action_profiles, tuple(snapshots)


def _snapshot_ledger(snapshots, *, manifest_hash: str | None = None):
    target_hashes = tuple(
        row.graph.target_family_hash for row in snapshots
    )
    graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": row.graph.target_family_hash,
                    "graph_hash": row.expected_graph_hash,
                    "availability_error_hash": None,
                }
                for row in snapshots
            ]
        }
    )
    catalog_set_hash = stable_hash(
        {
            "catalog_hashes": [
                row.expected_model_catalog_hash for row in snapshots
            ]
        }
    )
    return freeze_typed_selection_snapshot_ledger(
        snapshots,
        feasibility_preregistration_hash=stable_hash(
            {"synthetic": "preregistration"}
        ),
        feasibility_result_receipt_sha256=stable_hash(
            {"synthetic": "result-receipt"}
        ),
        feasibility_decision_hash=stable_hash(
            {"synthetic": "decision"}
        ),
        feasibility_report_hash=stable_hash({"synthetic": "report"}),
        manifest_hash=(
            manifest_hash
            if manifest_hash is not None
            else stable_hash({"synthetic": "manifest"})
        ),
        source_train_receipt_hash=stable_hash(
            {"synthetic": "source-receipt"}
        ),
        expected_graph_set_hash=graph_set_hash,
        expected_model_catalog_set_hash=catalog_set_hash,
        expected_target_family_hashes=target_hashes,
    )


def _context(residuals, action_profiles, snapshots) -> ValidationContext:
    ledger = _snapshot_ledger(snapshots)
    return ValidationContext(
        evaluator_epoch="typed-integration-epoch-v1",
        residuals=tuple(residuals),
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        trigger_feature_catalog={
            "family": {"allowed_operators": ["eq"]},
        },
        allowed_runtime_kinds=frozenset(
            {HypothesisKind.TASK, HypothesisKind.POLICY}
        ),
        allowed_action_operations=frozenset(
            {"execute_step", "check_condition", "produce_artifact"}
        ),
        action_semantics="skilllearn_prompt_directive_lowering_v2",
        external_evidence_is_hidden=True,
        action_design_profiles=action_profiles,
        typed_selection_snapshots=tuple(snapshots),
        typed_selection_ledger_hash=ledger.ledger_hash,
    )


def _kernel(proposer, snapshots, sink):
    ledger = _snapshot_ledger(snapshots)
    return EvolutionKernel(
        proposer=proposer,
        validator=RecursiveValidationEngine((), proposer=proposer, event_sink=sink),
        counterfactual_runner=SimpleNamespace(
            evaluator=SimpleNamespace(epoch="typed-integration-epoch-v1")
        ),
        promotion_gate=SimpleNamespace(spec=SimpleNamespace(metric="task_success")),
        archive=PolicyArchive(event_sink=sink),
        split_guard=SimpleNamespace(authorize=lambda *_: None),
        proposal_candidates_per_generation=3,
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        typed_selection_snapshots=snapshots,
        typed_selection_ledger=ledger,
        event_sink=sink,
    )


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_cli_requires_formal_integration_authority(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    manifests = project_root / "manifests"
    artifacts = project_root / "artifacts"
    manifests.mkdir(parents=True)
    source_run_root = artifacts / "source-run"
    source_run_root.mkdir(parents=True)
    protocol_path = manifests / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    preregistration = manifests / "integration-prereg.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    source_receipt = manifests / "source-receipt.json"
    source_receipt.write_text("{}\n", encoding="utf-8")
    authorization = manifests / "integration-result.json"
    ledger_hash = stable_hash({"production": "snapshot-ledger"})
    authorization.write_text("{}\n", encoding="utf-8")
    @dataclass(frozen=True)
    class FakeLedger:
        production_snapshot_ledger: Any
        freeze_authorization: Any = None

    fake_ledger = FakeLedger(
        production_snapshot_ledger=SimpleNamespace(ledger_hash=ledger_hash)
    )
    monkeypatch.setattr(
        skilllearn_experiment,
        "load_frozen_typed_selection_ledger",
        lambda **_: fake_ledger,
    )
    verified_receipt = {
        "source_binding": {
            "source_run_root": "artifacts/source-run",
            "source_train_receipt": "manifests/source-receipt.json",
            "source_train_receipt_file_sha256": _file_sha256(
                source_receipt
            ),
            "production_snapshot_ledger_hash": ledger_hash,
        },
        "fresh_development_protocol_freeze_eligible": True,
    }
    monkeypatch.setattr(
        skilllearn_experiment,
        "verify_typed_selection_integration_result_receipt",
        lambda **_: verified_receipt,
    )
    source = {
        "preregistration": "manifests/integration-prereg.json",
        "preregistration_file_sha256": _file_sha256(preregistration),
        "source_run_root": "artifacts/source-run",
        "source_train_receipt": "manifests/source-receipt.json",
        "source_train_receipt_file_sha256": _file_sha256(source_receipt),
        "integration_result_receipt": "manifests/integration-result.json",
        "integration_result_receipt_file_sha256": _file_sha256(
            authorization
        ),
        "snapshot_ledger_hash": ledger_hash,
    }
    protocol = PaperProtocol(
        path=protocol_path,
        payload={"protocol_version": "3.18.0"},
    )

    loaded = skilllearn_experiment._load_typed_selection_for_execution(
        root=project_root,
        manifest_path=manifests / "manifest.json",
        protocol=protocol,
        execution_contract={"typed_selection_snapshot_source": source},
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
    )
    assert loaded.production_snapshot_ledger is (
        fake_ledger.production_snapshot_ledger
    )
    assert loaded.freeze_authorization is not None

    monkeypatch.setattr(
        skilllearn_experiment,
        "verify_typed_selection_integration_result_receipt",
        lambda **_: {
            **verified_receipt,
            "fresh_development_protocol_freeze_eligible": False,
        },
    )
    with pytest.raises(PermissionError, match="protocol freeze"):
        skilllearn_experiment._load_typed_selection_for_execution(
            root=project_root,
            manifest_path=manifests / "manifest.json",
            protocol=protocol,
            execution_contract={
                "typed_selection_snapshot_source": source
            },
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
        )


def test_v320_production_loader_requires_portable_integration_authority(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    manifests = project_root / "manifests"
    artifacts = project_root / "artifacts"
    manifests.mkdir(parents=True)
    source_run_root = artifacts / "source-run"
    source_run_root.mkdir(parents=True)
    protocol_path = manifests / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    preregistration = manifests / "portable-prereg.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    source_receipt = manifests / "source-receipt.json"
    source_receipt.write_text("{}\n", encoding="utf-8")
    authorization = manifests / "portable-result.json"
    authorization.write_text("{}\n", encoding="utf-8")
    ledger_hash = stable_hash({"portable": "projected-ledger"})

    @dataclass(frozen=True)
    class FakeLedger:
        production_snapshot_ledger: Any
        freeze_authorization: Any = None

    fake_ledger = FakeLedger(
        production_snapshot_ledger=SimpleNamespace(ledger_hash=ledger_hash)
    )
    monkeypatch.setattr(
        skilllearn_experiment,
        "load_frozen_portable_typed_selection_ledger",
        lambda **_: fake_ledger,
    )
    monkeypatch.setattr(
        skilllearn_experiment,
        "load_frozen_typed_selection_ledger",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("v3.20 used the legacy typed ledger loader")
        ),
    )
    verified_receipt = {
        "source_binding": {
            "source_run_root": "artifacts/source-run",
            "source_train_receipt": "manifests/source-receipt.json",
            "source_train_receipt_file_sha256": _file_sha256(
                source_receipt
            ),
            "projected_snapshot_ledger_hash": ledger_hash,
        },
        "fresh_development_protocol_freeze_eligible": True,
    }
    monkeypatch.setattr(
        skilllearn_experiment,
        "verify_typed_portable_integration_result_receipt",
        lambda **_: verified_receipt,
    )
    monkeypatch.setattr(
        skilllearn_experiment,
        "verify_typed_selection_integration_result_receipt",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("v3.20 used the legacy integration verifier")
        ),
    )
    source = {
        "preregistration": "manifests/portable-prereg.json",
        "preregistration_file_sha256": _file_sha256(preregistration),
        "source_run_root": "artifacts/source-run",
        "source_train_receipt": "manifests/source-receipt.json",
        "source_train_receipt_file_sha256": _file_sha256(source_receipt),
        "integration_result_receipt": "manifests/portable-result.json",
        "integration_result_receipt_file_sha256": _file_sha256(
            authorization
        ),
        "snapshot_ledger_hash": ledger_hash,
    }
    protocol = PaperProtocol(
        path=protocol_path,
        payload={"protocol_version": "3.20.0"},
    )

    loaded = skilllearn_experiment._load_typed_selection_for_execution(
        root=project_root,
        manifest_path=manifests / "manifest.json",
        protocol=protocol,
        execution_contract={"typed_selection_snapshot_source": source},
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
    )
    assert loaded.production_snapshot_ledger is (
        fake_ledger.production_snapshot_ledger
    )
    assert loaded.freeze_authorization is not None

    with pytest.raises(PermissionError, match="diagnostic policy"):
        skilllearn_experiment._load_typed_selection_for_execution(
            root=project_root,
            manifest_path=manifests / "manifest.json",
            protocol=protocol,
            execution_contract={"typed_selection_snapshot_source": source},
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            integration_diagnostic_policy=(
                TYPED_SELECTION_INTEGRATION_VERSION
            ),
        )
    diagnostic = skilllearn_experiment._load_typed_selection_for_execution(
        root=project_root,
        manifest_path=manifests / "manifest.json",
        protocol=protocol,
        execution_contract={"typed_selection_snapshot_source": source},
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        integration_diagnostic_policy=TYPED_PORTABLE_INTEGRATION_VERSION,
    )
    assert diagnostic.freeze_authorization is None


def test_result_receipt_validation_binds_events_lock_and_compiler_evidence() -> None:
    preregistration = {"acceptance": {"all_passed": True}}
    compiler_commitment = {
        "compile_manifest_hash": stable_hash({"compile": "manifest"}),
        "compiled_binding_set_hash": stable_hash({"bindings": ["a"]}),
        "compiler_event_set_hash": stable_hash({"events": ["compile"]}),
        "compiler_event_path_normalization": (
            "item_hypothesis_content_route_v1"
        ),
        "compiler_binding_coverage_hash": stable_hash(
            {"bindings": ["a"]}
        ),
        "runtime_source_receipt_count": 38,
        "runtime_source_receipt_set_hash": stable_hash(
            {"receipts": ["a"]}
        ),
        "runtime_source_routed_count": 7,
        "runtime_source_no_skill_count": 31,
    }
    predecision = [
        {
            "event": "typed_compiler_binding_verified",
            "payload": dict(compiler_commitment),
        }
    ]
    report = {
        "integration_policy": TYPED_SELECTION_INTEGRATION_VERSION,
        "integration_passed": True,
        "decision_hash": stable_hash({"decision": "passed"}),
        "acceptance": {"all_passed": True},
        "fresh_development_protocol_freeze_eligible_if_passed": True,
        "development_task_execution_currently_authorized": False,
        "compiler_provenance": dict(compiler_commitment),
        "offline_boundary_contract": {
            "predecision_event_count": 1,
            "predecision_event_counts": {
                "typed_compiler_binding_verified": 1
            },
            "predecision_event_set_hash": stable_hash(
                {"events": predecision}
            ),
        },
    }
    report["report_hash"] = stable_hash(report)
    completed = {
        "event": "typed_selection_integration_completed",
        "payload": {
            "decision_hash": report["decision_hash"],
            "report_hash": report["report_hash"],
            "integration_passed": True,
        },
    }
    decision_lock = {
        "lock_version": "typed_selection_integration_decision_lock_v1",
        "decision_ordinal": 1,
        "state": "completed",
        "preregistration_hash": stable_hash(preregistration),
        "decision_hash": report["decision_hash"],
        "report_hash": report["report_hash"],
        "integration_passed": True,
        "raw_content_persisted": False,
    }

    _validate_completed_result_artifacts(
        preregistration=preregistration,
        report=report,
        events=(*predecision, completed),
        decision_lock=decision_lock,
    )

    drifted = json.loads(json.dumps(report))
    drifted["compiler_provenance"]["compiler_event_set_hash"] = (
        stable_hash({"events": ["forged"]})
    )
    drifted.pop("report_hash")
    drifted["report_hash"] = stable_hash(drifted)
    drifted_lock = {
        **decision_lock,
        "report_hash": drifted["report_hash"],
    }
    drifted_completed = {
        **completed,
        "payload": {
            **completed["payload"],
            "report_hash": drifted["report_hash"],
        },
    }
    with pytest.raises(PermissionError, match="compiler event commitment"):
        _validate_completed_result_artifacts(
            preregistration=preregistration,
            report=drifted,
            events=(*predecision, drifted_completed),
            decision_lock=drifted_lock,
        )


def test_production_kernel_uses_only_typed_recipe_selection_and_exact_replay() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    sink = MemoryEventSink()
    registry = TypedProgramBindingRegistry()
    proposer = StructuredHypothesisProposer(
        model,
        event_sink=sink,
        typed_program_registry=registry,
    )
    kernel = _kernel(proposer, snapshots, sink)
    context = _context(residuals, profiles, snapshots)

    programs = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-production-root",
    )
    replayed = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-production-root-replay",
    )

    assert replayed == programs
    assert len(model.requests) == 3
    assert registry.binding_count() == 3
    assert all(set(row["output_schema"]) == {
        "type", "additionalProperties", "required", "properties"
    } for row in model.requests)
    assert all(row["output_schema"]["additionalProperties"] is False for row in model.requests)
    assert all(row["output_schema"]["required"] == ["recipe_id"] for row in model.requests)
    assert all(set(row["output_schema"]["properties"]) == {"recipe_id"} for row in model.requests)
    assert all("residuals" not in row for row in model.requests)
    for program, request in zip(programs, model.requests):
        binding = registry.require(program)
        assert binding.request_hash == stable_hash(request)
        assert binding.response_hash == stable_hash(
            {"recipe_id": binding.recipe_id}
        )
    assert {
        row["event"] for row in sink.events
    } >= {
        "proposal_typed_recipe_plan_created",
        "typed_recipe_selection_requested",
        "typed_recipe_selection_materialized",
        "typed_recipe_selection_replayed",
        "proposal_typed_recipes_completed",
    }


def test_typed_kernel_and_registry_require_one_external_snapshot_ledger() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    proposer = StructuredHypothesisProposer(_RecipeSelectionModel())
    with pytest.raises(ValueError, match="ledger must be paired"):
        EvolutionKernel(
            proposer=proposer,
            validator=RecursiveValidationEngine((), proposer=proposer),
            counterfactual_runner=SimpleNamespace(
                evaluator=SimpleNamespace(
                    epoch="typed-integration-epoch-v1"
                )
            ),
            promotion_gate=SimpleNamespace(
                spec=SimpleNamespace(metric="task_success")
            ),
            archive=PolicyArchive(),
            split_guard=SimpleNamespace(authorize=lambda *_: None),
            proposal_candidates_per_generation=3,
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            typed_selection_snapshots=snapshots,
        )

    ledger = _snapshot_ledger(snapshots)
    registry = TypedProgramBindingRegistry(snapshot_ledger=ledger)
    tampered = replace(
        ledger,
        feasibility_report_hash=stable_hash({"tampered": "report"}),
    )
    with pytest.raises(PermissionError, match="binding conflict"):
        registry.bind_snapshot_ledger(tampered)

    context = _context(residuals, profiles, snapshots)
    assert context.typed_selection_ledger_hash == ledger.ledger_hash


def test_recursive_repair_reselects_within_same_frozen_snapshot() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    sink = MemoryEventSink()
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    kernel = _kernel(proposer, snapshots, sink)
    context = _context(residuals, profiles, snapshots)
    root = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-repair-root",
    )[0]
    validator = RecursiveValidationEngine(
        (_RootOnlyFailureCheck(root.id),),
        proposer=proposer,
        event_sink=sink,
    )

    tree = validator.validate(root, context, trace_id="typed-repair")

    assert tree.accepted_program is not None
    assert tree.recursion_depth == 1
    assert tree.accepted_program.parent_id == root.id
    assert len(model.requests) == 4
    repair_request = model.requests[-1]
    assert repair_request["request_kind"] == "select_typed_repair_recipe"
    assert set(repair_request["output_schema"]["properties"]) == {"recipe_id"}
    root_binding = proposer.typed_program_registry.require(root)
    assert repair_request["selection_scope"] == {
        "selection_round": 2,
        "excluded_recipe_ids": [root_binding.recipe_id],
        "excluded_recipe_set_hash": stable_hash(
            {"recipe_ids": [root_binding.recipe_id]}
        ),
        "excluded_recipe_count": 1,
    }
    assert root.id != tree.accepted_program.id
    proposer.typed_program_registry.require(tree.accepted_program)


def test_diagnostic_typed_ledger_cannot_execute_evolution() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    kernel = _kernel(
        StructuredHypothesisProposer(_RecipeSelectionModel()),
        snapshots,
        MemoryEventSink(),
    )

    with pytest.raises(
        PermissionError,
        match="protocol-lock task execution authorization",
    ):
        kernel.evolve_once(
            residuals=residuals,
            validation_tasks=(),
            validation_context=_context(residuals, profiles, snapshots),
            trace_id="diagnostic-ledger-execution-rejected",
        )


def test_typed_task_authority_is_factory_only_and_required_by_runner(
    tmp_path,
) -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    ledger = _snapshot_ledger(snapshots)
    with pytest.raises(PermissionError, match="factory-only"):
        TypedSelectionExecutionAuthorization(
            authorization_policy="forged",
            protocol_id="forged",
            protocol_hash="0" * 64,
            protocol_lock_hash="1" * 64,
            manifest_hash="2" * 64,
            snapshot_ledger_hash=ledger.ledger_hash,
            freeze_authorization_hash="3" * 64,
            task_execution_authorized=True,
        )

    diagnostic_proposer = StructuredHypothesisProposer(
        _RecipeSelectionModel()
    )
    diagnostic_program = _kernel(
        diagnostic_proposer,
        snapshots,
        MemoryEventSink(),
    ).propose_candidates(
        residuals,
        validation_context=_context(residuals, profiles, snapshots),
        trace_id="diagnostic-untyped-compiler-bypass",
    )[0]
    with pytest.raises(
        PermissionError,
        match="receipt-bound typed compiler",
    ):
        SkillLearnProgramCompiler().require_program_bindings(
            (diagnostic_program,)
        )

    compiler = SkillLearnProgramCompiler(
        typed_program_registry=TypedProgramBindingRegistry(
            snapshot_ledger=ledger
        ),
        require_typed_bindings=True,
    )
    with pytest.raises(
        PermissionError,
        match="protocol-lock task authority",
    ):
        SkillLearnCounterfactualRunner(
            adapter=SimpleNamespace(),
            manifest=SimpleNamespace(
                manifest_hash=stable_hash({"manifest": "synthetic"})
            ),
            guard=SimpleNamespace(),
            backend=SimpleNamespace(),
            evaluator=SkillLearnExternalEvaluator(
                "typed-integration-epoch-v1"
            ),
            compiler=compiler,
            output_root=tmp_path / "counterfactual",
        )


def test_validated_protocol_lock_issues_sealed_typed_task_authority(
    tmp_path,
    monkeypatch,
) -> None:
    _, _, snapshots = _snapshot_fixture()
    ledger = _snapshot_ledger(snapshots)
    freeze_authorization = TypedSelectionFreezeAuthorization(
        authorization_policy=(
            TYPED_SELECTION_FREEZE_AUTHORIZATION_VERSION
        ),
        result_receipt_stable_hash="0" * 64,
        result_receipt_file_sha256="1" * 64,
        source_binding_hash="2" * 64,
        snapshot_ledger_hash=ledger.ledger_hash,
        fresh_development_protocol_freeze_eligible=True,
        development_task_execution_authorized=False,
    )
    protocol_hash = stable_hash({"protocol": "typed-v318"})
    protocol = SimpleNamespace(
        payload={"protocol_version": "3.18.0"},
        id="typed-v318",
        protocol_hash=protocol_hash,
    )
    manifest_hash = stable_hash({"manifest": "typed-v318"})
    manifest = SimpleNamespace(manifest_hash=manifest_hash)
    lock_hash = stable_hash({"lock": "claim-eligible"})
    lock = {
        "typed_selection_freeze_authorization": (
            freeze_authorization.safe_payload()
        )
    }
    monkeypatch.setattr(
        paper_protocol_module,
        "validate_protocol_lock_for_execution",
        lambda *args, **kwargs: lock_hash,
    )

    authorization = authorize_typed_selection_execution(
        protocol,
        lock,
        manifest,
        tmp_path,
        tmp_path,
        ledger=ledger,
        freeze_authorization=freeze_authorization,
    )

    assert authorization.validate_for(
        ledger,
        freeze_authorization,
        manifest_hash=manifest_hash,
        protocol_hash=protocol_hash,
    ) == ()
    assert authorization.task_execution_authorized is True


def test_live_harness_binds_typed_ledger_before_runner_construction(
    tmp_path,
    monkeypatch,
) -> None:
    _, _, snapshots = _snapshot_fixture()
    manifest_hash = stable_hash({"manifest": "typed-harness"})
    ledger = _snapshot_ledger(
        snapshots,
        manifest_hash=manifest_hash,
    )
    freeze_authorization = TypedSelectionFreezeAuthorization(
        authorization_policy=(
            TYPED_SELECTION_FREEZE_AUTHORIZATION_VERSION
        ),
        result_receipt_stable_hash="0" * 64,
        result_receipt_file_sha256="1" * 64,
        source_binding_hash="2" * 64,
        snapshot_ledger_hash=ledger.ledger_hash,
        fresh_development_protocol_freeze_eligible=True,
        development_task_execution_authorized=False,
    )
    protocol_hash = stable_hash({"protocol": "typed-harness"})
    protocol = SimpleNamespace(
        payload={"protocol_version": "3.18.0"},
        id="typed-harness",
        protocol_hash=protocol_hash,
    )
    manifest = SimpleNamespace(manifest_hash=manifest_hash)
    lock_hash = stable_hash({"lock": "typed-harness"})
    lock = {
        "typed_selection_freeze_authorization": (
            freeze_authorization.safe_payload()
        )
    }
    monkeypatch.setattr(
        paper_protocol_module,
        "validate_protocol_lock_for_execution",
        lambda *args, **kwargs: lock_hash,
    )
    execution_authorization = authorize_typed_selection_execution(
        protocol,
        lock,
        manifest,
        tmp_path,
        tmp_path,
        ledger=ledger,
        freeze_authorization=freeze_authorization,
    )
    assert ledger.manifest_hash == manifest.manifest_hash
    assert execution_authorization.validate_for(
        ledger,
        freeze_authorization,
        manifest_hash=manifest_hash,
        protocol_hash=protocol_hash,
    ) == ()
    proposer = StructuredHypothesisProposer(_RecipeSelectionModel())
    adapter = SimpleNamespace(discover=lambda: ())

    harness = SkillLearnEvolutionHarness(
        adapter=adapter,
        manifest=manifest,
        guard=SimpleNamespace(),
        backend=SimpleNamespace(),
        proposer=proposer,
        validator=RecursiveValidationEngine((), proposer=proposer),
        promotion_gate=SimpleNamespace(
            spec=SimpleNamespace(metric="task_success")
        ),
        archive=PolicyArchive(),
        evaluator_epoch="typed-live-harness-epoch-v1",
        output_root=tmp_path / "compiled",
        proposal_candidates_per_generation=3,
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        typed_selection_snapshots=snapshots,
        typed_selection_ledger=ledger,
        typed_selection_freeze_authorization=freeze_authorization,
        typed_selection_execution_authorization=(
            execution_authorization
        ),
        portable_capability_compiler_mode=(
            PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
    )

    assert harness.compiler.typed_program_registry is (
        proposer.typed_program_registry
    )
    assert harness.kernel.proposer.typed_program_registry is (
        proposer.typed_program_registry
    )
    assert harness.compiler.require_typed_bindings is True
    assert harness.compiler.portable_capability_compiler_mode == (
        PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
    )
    assert (
        harness.compiler.typed_program_registry.require_snapshot_ledger()
        == ledger
    )
    assert harness.counterfactual_runner.typed_selection_execution_authorization is (
        execution_authorization
    )
    assert harness.kernel.typed_selection_execution_authorization is (
        execution_authorization
    )
    assert harness.kernel.typed_selection_freeze_authorization is (
        freeze_authorization
    )


def test_attempt_history_excludes_recipe_without_hypothesis_archival() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    proposer = StructuredHypothesisProposer(model)
    kernel = _kernel(proposer, snapshots, MemoryEventSink())
    context = _context(residuals, profiles, snapshots)

    first = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-attempt-history-generation-1",
    )
    kernel.record_typed_selection_attempts(
        first,
        trace_id="typed-attempt-history-generation-1:attempts",
    )
    assert kernel.archive.hypotheses == {}

    second = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-attempt-history-generation-2",
    )

    assert len(model.requests) == 6
    for prior, current in zip(first, second):
        prior_binding = proposer.typed_program_registry.require(prior)
        current_binding = proposer.typed_program_registry.require(current)
        assert current_binding.recipe_id != prior_binding.recipe_id
        assert current_binding.excluded_recipe_ids == (
            prior_binding.recipe_id,
        )
        assert current_binding.selection_round == 2


def test_history_only_binding_rejects_self_rehashed_forgery() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    proposer = StructuredHypothesisProposer(_RecipeSelectionModel())
    kernel = _kernel(proposer, snapshots, MemoryEventSink())
    programs = kernel.propose_candidates(
        residuals,
        validation_context=_context(residuals, profiles, snapshots),
        trace_id="typed-history-forgery-root",
    )
    kernel.record_typed_selection_attempts(
        programs,
        trace_id="typed-history-forgery-attempts",
    )
    history = {
        key: dict(value)
        for key, value in kernel.archive.typed_selection_history.items()
    }
    old_key = sorted(history)[0]
    forged = history.pop(old_key)
    forged["program_id"] = "forged-program-id"
    forged["program_identity_hash"] = stable_hash(
        {
            "program_id": forged["program_id"],
            "evaluator_epoch": forged["evaluator_epoch"],
            "program_executable_hash": forged["program_executable_hash"],
        }
    )
    forged.pop("binding_hash")
    forged["binding_hash"] = stable_hash(forged)
    history[forged["binding_hash"]] = forged

    with pytest.raises(PermissionError, match="not canonical"):
        validate_typed_selection_history_payloads(
            history,
            snapshot_ledger=_snapshot_ledger(snapshots),
        )


def test_root_selection_excludes_archived_bound_recipes_across_generations() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    sink = MemoryEventSink()
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    kernel = _kernel(proposer, snapshots, sink)
    context = _context(residuals, profiles, snapshots)

    generations: list[tuple[HypothesisProgram, ...]] = []
    for generation_index in range(3):
        programs = kernel.propose_candidates(
            residuals,
            validation_context=context,
            trace_id=f"typed-generation-{generation_index + 1}",
        )
        generations.append(programs)
        kernel.record_typed_selection_attempts(
            programs,
            trace_id=f"typed-generation-{generation_index + 1}:attempts",
        )
        for program in programs:
            kernel.archive.register_hypothesis(
                program,
                typed_binding=(
                    proposer.typed_program_registry.safe_binding(program)
                ),
            )

    assert len(model.requests) == 9
    for slot_index, snapshot in enumerate(snapshots):
        bindings = [
            proposer.typed_program_registry.require(
                generation[slot_index]
            )
            for generation in generations
        ]
        assert len({row.recipe_id for row in bindings}) == 3
        assert [row.selection_round for row in bindings] == [1, 2, 3]
        assert bindings[0].excluded_recipe_ids == ()
        assert bindings[1].excluded_recipe_ids == (
            bindings[0].recipe_id,
        )
        assert set(bindings[2].excluded_recipe_ids) == {
            bindings[0].recipe_id,
            bindings[1].recipe_id,
        }
        assert all(
            row.snapshot_hash == snapshot.snapshot_hash for row in bindings
        )

    before = len(model.requests)
    with pytest.raises(PermissionError, match="no untried recipe"):
        kernel.propose_candidates(
            residuals,
            validation_context=context,
            trace_id="typed-generation-exhausted",
        )
    assert len(model.requests) == before


def test_gen2_root_repair_inherits_archived_recipe_exclusions() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    proposer = StructuredHypothesisProposer(model)
    kernel = _kernel(proposer, snapshots, MemoryEventSink())
    context = _context(residuals, profiles, snapshots)

    generation_1 = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-inherited-exclusions-generation-1",
    )
    kernel.record_typed_selection_attempts(
        generation_1,
        trace_id="typed-inherited-exclusions-generation-1:attempts",
    )
    for program in generation_1:
        kernel.archive.register_hypothesis(
            program,
            typed_binding=(
                proposer.typed_program_registry.safe_binding(program)
            ),
        )
    generation_2 = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-inherited-exclusions-generation-2",
    )
    generation_1_binding = proposer.typed_program_registry.require(
        generation_1[0]
    )
    generation_2_binding = proposer.typed_program_registry.require(
        generation_2[0]
    )

    repaired = proposer.revise(
        generation_2[0],
        failed_checks=(
            {"check": "synthetic_check", "reason": "try another recipe"},
        ),
        residuals=residuals,
        depth=1,
        typed_recipe_snapshot=snapshots[0],
        trace_id="typed-inherited-exclusions-repair",
    )
    repaired_binding = proposer.typed_program_registry.require(repaired)

    assert generation_2_binding.excluded_recipe_ids == (
        generation_1_binding.recipe_id,
    )
    assert set(repaired_binding.excluded_recipe_ids) == {
        generation_1_binding.recipe_id,
        generation_2_binding.recipe_id,
    }
    assert repaired_binding.recipe_id not in {
        generation_1_binding.recipe_id,
        generation_2_binding.recipe_id,
    }
    assert repaired_binding.selection_round == 3
    assert model.requests[-1]["selection_scope"]["selection_round"] == 3
    assert set(
        model.requests[-1]["selection_scope"]["excluded_recipe_ids"]
    ) == {
        generation_1_binding.recipe_id,
        generation_2_binding.recipe_id,
    }


def test_binding_receipts_restore_and_reject_forged_hashes() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    proposer = StructuredHypothesisProposer(_RecipeSelectionModel())
    kernel = _kernel(proposer, snapshots, MemoryEventSink())
    context = _context(residuals, profiles, snapshots)
    root = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-binding-restore-root",
    )[0]
    child = proposer.revise(
        root,
        failed_checks=(
            {"check": "shape_check", "reason": "missing output table"},
        ),
        residuals=residuals,
        depth=1,
        typed_recipe_snapshot=snapshots[0],
        trace_id="typed-binding-restore-child",
    )
    root_payload = proposer.typed_program_registry.safe_binding(root)
    child_payload = proposer.typed_program_registry.safe_binding(child)
    assert child_payload["failed_checks"] == [
        {"check": "shape_check", "reason": "missing output table"}
    ]

    restored = TypedProgramBindingRegistry(
        snapshot_ledger=_snapshot_ledger(snapshots)
    )
    archived_root = replace(root, status=HypothesisStatus.SHADOW)
    assert restored.restore_safe_payload(
        archived_root,
        root_payload,
    ).safe_payload() == root_payload
    assert restored.restore_safe_payload(
        child,
        child_payload,
    ).safe_payload() == child_payload

    forged_request = TypedProgramBindingRegistry(
        snapshot_ledger=_snapshot_ledger(snapshots)
    )
    with pytest.raises(PermissionError, match="request receipt"):
        forged_request.restore_safe_payload(
            root,
            {
                **root_payload,
                "request_hash": stable_hash({"forged": "request"}),
            },
        )
    assert forged_request.binding_count() == 0

    forged_response = TypedProgramBindingRegistry(
        snapshot_ledger=_snapshot_ledger(snapshots)
    )
    with pytest.raises(PermissionError, match="response receipt"):
        forged_response.restore_safe_payload(
            root,
            {
                **root_payload,
                "response_hash": stable_hash({"forged": "response"}),
            },
        )
    assert forged_response.binding_count() == 0


def test_typed_repair_excludes_entire_bound_recipe_lineage() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    proposer = StructuredHypothesisProposer(model)
    kernel = _kernel(proposer, snapshots, MemoryEventSink())
    context = _context(residuals, profiles, snapshots)
    root = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-lineage-root",
    )[0]
    first_child = proposer.revise(
        root,
        failed_checks=(),
        residuals=residuals,
        depth=1,
        typed_recipe_snapshot=snapshots[0],
        trace_id="typed-lineage-repair-1",
    )
    second_child = proposer.revise(
        first_child,
        failed_checks=(),
        residuals=residuals,
        depth=2,
        typed_recipe_snapshot=snapshots[0],
        trace_id="typed-lineage-repair-2",
    )

    root_binding = proposer.typed_program_registry.require(root)
    first_binding = proposer.typed_program_registry.require(first_child)
    second_binding = proposer.typed_program_registry.require(second_child)
    assert len(
        {
            root_binding.recipe_id,
            first_binding.recipe_id,
            second_binding.recipe_id,
        }
    ) == 3
    assert first_binding.lineage_recipe_ids == (root_binding.recipe_id,)
    assert second_binding.lineage_recipe_ids == (
        root_binding.recipe_id,
        first_binding.recipe_id,
    )
    assert set(model.requests[-1]["selection_scope"]["excluded_recipe_ids"]) == {
        root_binding.recipe_id,
        first_binding.recipe_id,
    }

    before = len(model.requests)
    with pytest.raises(PermissionError, match="no untried recipe"):
        proposer.revise(
            second_child,
            failed_checks=(),
            residuals=residuals,
            depth=3,
            typed_recipe_snapshot=snapshots[0],
            trace_id="typed-lineage-repair-exhausted",
        )
    assert len(model.requests) == before

    shortened_lineage = replace(second_child, lineage=(first_child.id,))
    with pytest.raises(PermissionError, match="does not validate"):
        proposer.typed_program_registry.require(shortened_lineage)


def test_typed_shared_candidate_and_compiler_provenance_fail_closed(tmp_path) -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    sink = MemoryEventSink()
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    kernel = _kernel(proposer, snapshots, sink)
    context = _context(residuals, profiles, snapshots)
    programs = kernel.propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-shared",
    )

    kernel._require_typed_candidate_batch(programs)
    with pytest.raises(PermissionError, match="cardinality"):
        kernel._require_typed_candidate_batch(programs[:2])
    with pytest.raises(PermissionError, match="order or coverage"):
        kernel._require_typed_candidate_batch(tuple(reversed(programs)))
    mutated = replace(
        programs[0],
        statement="model-authored free-text bypass",
    )
    with pytest.raises(PermissionError, match="binding is missing"):
        proposer.typed_program_registry.require(mutated)
    with pytest.raises(PermissionError):
        kernel._require_typed_candidate_batch((mutated, *programs[1:]))

    compiler = SkillLearnProgramCompiler(
        typed_program_registry=proposer.typed_program_registry,
        require_typed_bindings=True,
    )
    train_ids = ("train-a", "train-b", "train-c")
    family_by_id = {
        "train-a": "family-a",
        "train-b": "family-b",
        "train-c": "family-c",
        "validation": "validation-family",
        "test": "test-family",
    }
    manifest = SplitManifest(
        benchmark="synthetic",
        protocol="instance_holdout",
        seed="typed-compiler",
        train_ids=train_ids,
        validation_ids=("validation",),
        test_ids=("test",),
        family_by_id=family_by_id,
    )
    items = tuple(
        BenchmarkItem(
            id=item_id,
            family=family_by_id[item_id],
            features={"family": family_by_id[item_id]},
            content_ref="synthetic",
            verifier_ref_hash=stable_hash({"verifier": item_id}),
        )
        for item_id in train_ids
    )
    compiled = compiler.compile(
        programs=programs,
        items=items,
        split_manifest=manifest,
        output_root=tmp_path,
        allowed_statuses={HypothesisStatus.CANDIDATE},
        target_item_ids=train_ids,
        target_split="train",
    )
    assert compiled.hypothesis_ids == tuple(sorted(row.id for row in programs))
    expected_ledger_hash = _snapshot_ledger(snapshots).ledger_hash
    assert compiled.typed_binding_hashes
    assert compiled.typed_binding_set_hash
    assert compiled.typed_snapshot_ledger_hash == expected_ledger_hash
    compile_manifest = json.loads(
        (compiled.output_root / "compile_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert compile_manifest["typed_binding_set_hash"] == (
        compiled.typed_binding_set_hash
    )
    assert compile_manifest["typed_snapshot_ledger_hash"] == (
        expected_ledger_hash
    )
    assert {
        row["binding_hash"] for row in compile_manifest["typed_binding_rows"]
    } == set(compiled.typed_binding_hashes)
    with pytest.raises(PermissionError, match="binding is missing"):
        compiler.compile(
            programs=(mutated, *programs[1:]),
            items=items,
            split_manifest=manifest,
            output_root=tmp_path,
            method_name="forged",
            allowed_statuses={HypothesisStatus.CANDIDATE},
            target_item_ids=train_ids,
            target_split="train",
        )

    promoted = replace(programs[0], status=HypothesisStatus.PROMOTED)
    assert (
        proposer.typed_program_registry.require(promoted).recipe_id
        == proposer.typed_program_registry.require(programs[0]).recipe_id
    )


def test_typed_repair_never_falls_back_for_an_unbound_parent() -> None:
    residuals, profiles, snapshots = _snapshot_fixture()
    model = _RecipeSelectionModel()
    sink = MemoryEventSink()
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    context = _context(residuals, profiles, snapshots)
    root = _kernel(proposer, snapshots, sink).propose_candidates(
        residuals,
        validation_context=context,
        trace_id="typed-unbound-source",
    )[0]
    with pytest.raises(PermissionError, match="generic free-text repair"):
        proposer.revise(
            root,
            failed_checks=(),
            residuals=residuals,
            depth=1,
            trace_id="typed-generic-repair-bypass",
        )
    unbound = replace(root, id="unbound-program")
    before = len(model.requests)
    with pytest.raises(PermissionError, match="unbound typed parent"):
        proposer.revise(
            unbound,
            failed_checks=(),
            residuals=residuals,
            depth=1,
            trace_id="typed-unbound-generic-repair-bypass",
        )
    assert len(model.requests) == before
    with pytest.raises(PermissionError, match="binding is missing"):
        proposer.select_typed_recipe(
            snapshot=snapshots[0],
            evaluator_epoch=root.evaluator_epoch,
            parent=unbound,
            depth=1,
            trace_id="typed-direct-unbound-parent",
        )
    assert len(model.requests) == before

    forged_child = replace(
        root,
        id="forged-repair",
        parent_id=unbound.id,
        lineage=(unbound.id,),
    )
    root_binding = proposer.typed_program_registry.require(root)
    with pytest.raises(PermissionError, match="requires a bound parent"):
        proposer.typed_program_registry.register(
            root,
            snapshot=snapshots[0],
            recipe_id=root_binding.recipe_id,
            request_kind="select_typed_repair_recipe",
            request_hash=stable_hash({"request": "parentless-repair"}),
            response_hash=stable_hash({"response": "parentless-repair"}),
            selection_round=2,
        )
    with pytest.raises(PermissionError, match="binding is missing"):
        proposer.typed_program_registry.register(
            forged_child,
            snapshot=snapshots[0],
            recipe_id=root_binding.recipe_id,
            request_kind="select_typed_repair_recipe",
            request_hash=stable_hash({"request": "forged"}),
            response_hash=stable_hash({"response": "forged"}),
            selection_round=2,
            excluded_recipe_ids=(),
            parent=unbound,
        )

    tree = RecursiveValidationEngine(
        (_RootOnlyFailureCheck(unbound.id),),
        proposer=proposer,
        event_sink=sink,
    ).validate(unbound, context, trace_id="typed-unbound")

    assert tree.accepted_program is None
    assert tree.nodes[0].terminal_reason == "typed_snapshot_unbound"
    assert len(model.requests) == before
    assert any(
        row["event"] == "typed_repair_blocked_unbound_parent"
        for row in sink.events
    )

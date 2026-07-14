from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import typed_portable_integration as portable
from assumption_agent.benchmarks.typed_task_capability import (
    project_portable_family_capability_graph,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.models import ResidualExample, SplitName, stable_hash
from assumption_agent.splits import SplitManifest
from assumption_agent.typed_operator_grammar import (
    ArtifactFormat,
    ArtifactSpec,
    FamilyCapabilityGraph,
    _artifact_id,
    _artifact_rank,
    _build_recipe,
    _capability_spec,
    _workflows_for_format,
    freeze_typed_recipe_selection_snapshot,
    freeze_typed_selection_snapshot_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = (
    REPO_ROOT
    / "reference"
    / "self_evo_continual_20260707"
    / "repos"
    / "SkillLearnBench"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "manifests"
    / "skilllearnbench_instance_holdout_offline_ready_v1.json"
)
FAMILIES = (
    "stock-data-visualization",
    "temperature-simulation",
    "organize-messy-files",
)
CANARY_IDS = (
    "stock-data-visualization-3",
    "temperature-simulation-1",
    "organize-messy-files-3",
)
IMAGE_IDS = (
    "sha256:85e70a1043a5a1c7d8b24306ce34a21c75d35a8906ff433362932f3944d5b83b",
    "sha256:f0e4e68439d8eda48162616712bfdc0fcfa7faddb275c183107be6fcb9575db6",
    "sha256:be341fcb2644c835ae279ebffabd1d0ab6e0f32e80b81cb9007d3a9456bb98cc",
)
IMAGE_KEYS = (
    "29a2b128f134614f65722ac49c7118137ba5355d996bc66fc744be8ca16f8895",
    "5630c42fd77721f6ea450d94342168767f2bd15ce30063159b79ad13f0f82986",
    "d91fbc9b9d38770e56ea81ea23b06f77c26fe9ed18826ac5c799246345c37c3b",
)
CLOSURE_HASHES = (
    "4955fd717d54962445bf43088595523962a2e931a377c2fc463beee2f08f27c1",
    None,
    "0ae82e6b22ffa08320b0d6c2e3c4e58439be710f038dbba5b871f24ca028e8d1",
)
INTEGRITY_HASHES = (
    "3df6370f11c78cdf6633e262a82512189d67dfd3a668089130761496ffe220c6",
    None,
    "a223f1f79bc654f7c9428b49a798a98400b2e486747fad81bfb4aececcc7f83a",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_graph(
    family: str,
    *,
    portable_format: ArtifactFormat,
    alias_count: int,
    unsupported_format: ArtifactFormat | None,
) -> FamilyCapabilityGraph:
    extension = {
        ArtifactFormat.TABULAR: "csv",
        ArtifactFormat.OFFICE_DOCUMENT: "docx",
        ArtifactFormat.WEB_ASSET: "html",
        ArtifactFormat.CONFIGURATION: "nml",
        ArtifactFormat.TEXT: "txt",
    }
    locators = [
        (f"/root/input-{index}.{extension[portable_format]}", portable_format)
        for index in range(alias_count + 1)
    ]
    if unsupported_format is not None:
        locators.append(
            (
                f"/root/unsupported.{extension[unsupported_format]}",
                unsupported_format,
            )
        )
    artifacts = []
    for index, (locator, artifact_format) in enumerate(locators, start=1):
        provenance = f"{index:x}" * 64
        relations = ("declared_task_local_path",)
        artifacts.append(
            ArtifactSpec(
                artifact_id=_artifact_id(
                    target_family=family,
                    locator=locator,
                    artifact_format=artifact_format,
                    support_count=2,
                    evidence_relations=relations,
                    provenance_hash=provenance,
                ),
                locator=locator,
                format=artifact_format,
                support_count=2,
                evidence_relations=relations,
                provenance_hash=provenance,
            )
        )
    ordered = tuple(sorted(artifacts, key=_artifact_rank))
    capabilities = tuple(
        _capability_spec(target_family=family, artifact=artifact)
        for artifact in ordered
    )
    recipes = tuple(
        sorted(
            (
                _build_recipe(
                    target_family=family,
                    artifact=artifact,
                    capability=capability,
                    workflow=workflow,
                )
                for artifact, capability in zip(ordered, capabilities)
                for workflow in _workflows_for_format(artifact.format)
            ),
            key=lambda row: row.recipe_id,
        )
    )
    graph = FamilyCapabilityGraph(
        target_family=family,
        source_evidence_hash="f" * 64,
        artifacts=ordered,
        capabilities=capabilities,
        recipes=recipes,
    )
    assert graph.validate() == ()
    return graph


def _graphs_and_ledger():
    full_graphs = (
        _full_graph(
            FAMILIES[0],
            portable_format=ArtifactFormat.TABULAR,
            alias_count=1,
            unsupported_format=ArtifactFormat.WEB_ASSET,
        ),
        _full_graph(
            FAMILIES[1],
            portable_format=ArtifactFormat.TABULAR,
            alias_count=2,
            unsupported_format=ArtifactFormat.CONFIGURATION,
        ),
        _full_graph(
            FAMILIES[2],
            portable_format=ArtifactFormat.OFFICE_DOCUMENT,
            alias_count=5,
            unsupported_format=None,
        ),
    )
    projected = tuple(
        project_portable_family_capability_graph(graph)
        for graph in full_graphs
    )
    snapshots = tuple(
        freeze_typed_recipe_selection_snapshot(graph) for graph in projected
    )
    graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": snapshot.graph.target_family_hash,
                    "graph_hash": snapshot.expected_graph_hash,
                    "availability_error_hash": None,
                }
                for snapshot in snapshots
            ]
        }
    )
    catalog_set_hash = stable_hash(
        {
            "catalog_hashes": [
                snapshot.expected_model_catalog_hash
                for snapshot in snapshots
            ]
        }
    )
    snapshot_ledger = freeze_typed_selection_snapshot_ledger(
        snapshots,
        feasibility_preregistration_hash="1" * 64,
        feasibility_result_receipt_sha256="2" * 64,
        feasibility_decision_hash="3" * 64,
        feasibility_report_hash="4" * 64,
        manifest_hash="5" * 64,
        source_train_receipt_hash="6" * 64,
        expected_graph_set_hash=graph_set_hash,
        expected_model_catalog_set_hash=catalog_set_hash,
        expected_target_family_hashes=tuple(
            graph.target_family_hash for graph in projected
        ),
    )
    residuals = tuple(
        ResidualExample(
            transition_id=f"transition-{index}",
            task_id=f"task-{index}",
            family=family,
            split=SplitName.TRAIN,
            features={"family": family},
            failure_type="fixture_failure",
            evaluator_feedback=(),
            baseline_success=False,
        )
        for index, family in enumerate(FAMILIES, start=1)
    )
    ledger = portable.FrozenTypedSelectionLedger(
        evidence=SimpleNamespace(
            residuals=residuals,
            action_profiles={},
        ),
        trials=(),
        snapshots=snapshots,
        production_snapshot_ledger=snapshot_ledger,
        upstream_binding_hash="7" * 64,
        trial_evidence_hash="8" * 64,
        graph_set_hash=graph_set_hash,
        model_catalog_set_hash=catalog_set_hash,
    )
    return full_graphs, projected, ledger


def _canary_rows() -> tuple[dict[str, object], ...]:
    rows = []
    for family, item_id, image_id, image_key, closure, integrity in zip(
        FAMILIES,
        CANARY_IDS,
        IMAGE_IDS,
        IMAGE_KEYS,
        CLOSURE_HASHES,
        INTEGRITY_HASHES,
    ):
        item_root = BENCHMARK_ROOT / "tasks" / family / item_id
        rows.append(
            {
                "family": family,
                "family_hash": stable_hash({"family": family}),
                "item_id": item_id,
                "item_id_hash": stable_hash({"item_id": item_id}),
                "prebuilt_image_id": image_id,
                "prebuilt_image_key": image_key,
                "task_input_closure_hash": closure,
                "task_input_integrity_receipt_hash": integrity,
                "task_input_integrity_container_network": (
                    "none" if integrity is not None else None
                ),
                "public_instruction_sha256": _sha256(
                    item_root / "instruction.md"
                ),
                "dockerfile_sha256": _sha256(
                    item_root / "environment" / "Dockerfile"
                ),
            }
        )
    return tuple(rows)


def test_projection_binds_honest_evidence_roles_selector_context_and_aliases():
    full_graphs, projected, _ = _graphs_and_ledger()

    rows = tuple(
        portable._projection_row(full, child)
        for full, child in zip(full_graphs, projected)
    )

    assert [row["projected_recipe_count"] for row in rows] == [3, 3, 3]
    assert [row["behavioral_alias_artifact_count"] for row in rows] == [1, 2, 5]
    assert all(row["projected_artifact_count"] == 1 for row in rows)
    assert all(row["behavioral_aliases_deduplicated"] is True for row in rows)
    assert all(
        row["capability_execution_covers_full_recipe_operator_plan"] is False
        for row in rows
    )
    assert all(row["pre_agent_sidecar_is_task_effect"] is False for row in rows)
    for graph in projected:
        portable._verify_portable_evidence_catalog_semantics(graph)
        catalog = graph.model_catalog()
        assert catalog["portable_selector_context"]["target_family"] == (
            graph.target_family
        )
        assert all(
            capability["capability_implementation_verified"] is False
            for capability in catalog["capabilities"]
        )


def test_projection_semantics_rejects_full_operator_overclaim():
    _, projected, _ = _graphs_and_ledger()
    catalog = projected[0].model_catalog()
    catalog["portable_capability_projection"] = {
        **catalog["portable_capability_projection"],
        "capability_execution_covers_full_recipe_operator_plan": True,
    }

    class Overclaim:
        target_family = projected[0].target_family

        @staticmethod
        def model_catalog():
            return catalog

    with pytest.raises(PermissionError, match="overclaims"):
        portable._verify_portable_evidence_catalog_semantics(Overclaim())


def test_real_proposer_kernel_selects_three_slots_and_deduplicated_treatments():
    full_graphs, _, ledger = _graphs_and_ledger()
    programs, proposer, receipt, tamper = (
        portable._exercise_opaque_production_selection(
            ledger=ledger,
            full_graphs=full_graphs,
            event_sink=MemoryEventSink(),
            trace_id="typed-portable-fixture",
        )
    )

    assert len(programs) == 3
    assert receipt["request_count"] == 6
    assert receipt["live_model_invoked"] is False
    assert receipt["behavioral_alias_recipe_ids_expressible"] is False
    assert receipt["projected_behavioral_alias_artifact_count"] == 8
    assert receipt["generation_two_changed_fixed_plan_treatment_per_slot"] is True
    assert all(tamper.values())
    assert len(
        [proposer.typed_program_registry.require(program) for program in programs]
    ) == 3


def test_all_three_bindings_compile_with_mocked_private_docker_seam(
    monkeypatch: pytest.MonkeyPatch,
):
    full_graphs, _, ledger = _graphs_and_ledger()
    programs, proposer, _, _ = portable._exercise_opaque_production_selection(
        ledger=ledger,
        full_graphs=full_graphs,
        event_sink=MemoryEventSink(),
        trace_id="typed-portable-compile-fixture",
    )
    canaries = _canary_rows()

    def fake_runtime(**kwargs):
        assert len(kwargs["programs"]) == 3
        assert kwargs["manifest_hash"] == SplitManifest.read(
            MANIFEST_PATH
        ).manifest_hash
        return tuple(
            {
                "family_hash": row["family_hash"],
                "item_id_hash": row["item_id_hash"],
                "artifact_evidence_sidecar_verified": True,
                "recipe_operator_effect_claimed": False,
                "task_effect_claimed": False,
                "container_cleanup_verified": True,
                "passed": True,
            }
            for row in kwargs["canaries"]
        )

    monkeypatch.setattr(
        portable,
        "_run_pre_agent_docker_canaries",
        fake_runtime,
    )
    preregistration = {
        "runtime_canaries": {
            "rows": list(canaries),
            "set_hash": stable_hash({"rows": list(canaries)}),
        }
    }
    compiler, runtime, mutation_rejected = (
        portable._compile_full_bundle_and_run_canaries(
            root=BENCHMARK_ROOT,
            manifest_path=MANIFEST_PATH,
            programs=programs,
            proposer=proposer,
            preregistration=preregistration,
            event_sink=MemoryEventSink(),
            trace_id="typed-portable-compile-fixture",
        )
    )

    assert compiler["program_count"] == 3
    assert compiler["all_three_typed_plan_bindings_compiled"] is True
    assert compiler["recipe_operator_execution_compiled"] is False
    assert compiler["source_receipt_count"] == 3
    assert runtime["canary_count"] == 3
    assert runtime["container_cleanup_verified"] is True
    assert runtime["agent_started"] is False
    assert mutation_rejected is True


def test_historical_feasibility_is_hash_bound_evidence_not_current_authority(
    tmp_path: Path,
):
    project = tmp_path / "project"
    manifests = project / "manifests"
    artifacts = project / "artifacts" / "old"
    manifests.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    new_preregistration = manifests / "new.json"
    new_preregistration.write_text("{}\n", encoding="utf-8")
    old_preregistration = manifests / "old.json"
    old_payload = {
        "historical_policy": "old-only",
        "expected_implementation_file_set_hash": "0" * 64,
    }
    old_preregistration.write_text(
        json.dumps(old_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision_hash = stable_hash({"decision": "passed"})
    report_payload = {
        "feasibility_passed": True,
        "decision_hash": decision_hash,
    }
    report_payload["report_hash"] = stable_hash(report_payload)
    report_path = artifacts / "report.json"
    report_path.write_text(
        json.dumps(report_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    events_path = artifacts / "events.jsonl"
    event = {
        "event": "typed_operator_feasibility_completed",
        "stage": "fixture",
        "trace_id": "fixture",
        "payload": {
            "decision_hash": decision_hash,
            "report_hash": report_payload["report_hash"],
        },
    }
    events_path.write_text(json.dumps(event, sort_keys=True) + "\n")
    lock_payload = {
        "state": "completed",
        "preregistration_hash": stable_hash(old_payload),
        "decision_hash": decision_hash,
        "report_hash": report_payload["report_hash"],
    }
    lock_path = artifacts / "lock.json"
    lock_path.write_text(json.dumps(lock_payload, sort_keys=True) + "\n")
    result_payload = {
        "feasibility_passed": True,
        "exact_replay_verified": True,
        "decision_hash": decision_hash,
        "report_hash": report_payload["report_hash"],
        "canonical_artifacts": {
            "report": "artifacts/old/report.json",
            "report_sha256": _sha256(report_path),
            "events": "artifacts/old/events.jsonl",
            "events_sha256": _sha256(events_path),
            "decision_lock": "artifacts/old/lock.json",
            "decision_lock_sha256": _sha256(lock_path),
        },
    }
    result_path = manifests / "old-result.json"
    result_path.write_text(
        json.dumps(result_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = {
        "preregistration": {
            "path": "manifests/old.json",
            "sha256": _sha256(old_preregistration),
            "stable_hash": stable_hash(old_payload),
        },
        "result_receipt": {
            "path": "manifests/old-result.json",
            "sha256": _sha256(result_path),
            "stable_hash": stable_hash(result_payload),
        },
        "report": {
            "path": "artifacts/old/report.json",
            "sha256": _sha256(report_path),
        },
        "events": {
            "path": "artifacts/old/events.jsonl",
            "sha256": _sha256(events_path),
        },
        "decision_lock": {
            "path": "artifacts/old/lock.json",
            "sha256": _sha256(lock_path),
        },
        "decision_hash": decision_hash,
        "report_hash": report_payload["report_hash"],
    }

    evidence = portable._verify_historical_feasibility(
        {"historical_feasibility": rows},
        preregistration_path=new_preregistration,
    )

    assert evidence.report_hash == report_payload["report_hash"]
    assert evidence.safe_payload()["historical_evidence_only"] is True
    assert evidence.safe_payload()["current_execution_authority_granted"] is False
    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="file hash drifted"):
        portable._verify_historical_feasibility(
            {"historical_feasibility": rows},
            preregistration_path=new_preregistration,
        )


def test_explicit_implementation_binding_fails_closed_on_tamper(tmp_path: Path):
    assert {
        "assumption_agent/benchmarks/skilllearn_experiment.py",
        "assumption_agent/benchmarks/paper_protocol.py",
    } <= portable._REQUIRED_IMPLEMENTATION_FILES
    project = tmp_path / "project"
    manifests = project / "manifests"
    manifests.mkdir(parents=True)
    rows = []
    for relative in sorted(portable._REQUIRED_IMPLEMENTATION_FILES):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
        rows.append({"path": relative, "sha256": "0" * 64})
    preregistration_path = manifests / "portable.json"
    preregistration_path.write_text(
        json.dumps({"implementation_files": rows}) + "\n",
        encoding="utf-8",
    )

    binding = portable.build_implementation_file_binding(
        preregistration_path
    )
    assert len(binding["implementation_files"]) == len(
        portable._REQUIRED_IMPLEMENTATION_FILES
    )
    payload = {
        "implementation_files": binding["implementation_files"],
        "expected_implementation_file_set_hash": binding[
            "expected_implementation_file_set_hash"
        ],
    }
    assert portable._implementation_file_set_hash(
        payload,
        preregistration_path=preregistration_path,
    ) == binding["expected_implementation_file_set_hash"]
    first = project / binding["implementation_files"][0]["path"]
    first.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="file hash drifted"):
        portable._implementation_file_set_hash(
            payload,
            preregistration_path=preregistration_path,
        )


def test_v320_diagnostic_production_loader_needs_no_result_receipt_and_rejects_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from assumption_agent.benchmarks import skilllearn_experiment
    from assumption_agent.benchmarks.paper_protocol import PaperProtocol
    from assumption_agent.evolution import (
        TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION,
    )

    project = tmp_path / "project"
    manifests = project / "manifests"
    source_root = project / "artifacts" / "source"
    manifests.mkdir(parents=True)
    source_root.mkdir(parents=True)
    preregistration = manifests / "portable.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    source_receipt = manifests / "source.json"
    source_receipt.write_text("{}\n", encoding="utf-8")
    projected_hash = "a" * 64
    fake_ledger = SimpleNamespace(
        production_snapshot_ledger=SimpleNamespace(
            ledger_hash=projected_hash
        ),
        freeze_authorization=None,
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
            AssertionError("v3.20 traversed the legacy ledger loader")
        ),
    )
    source = {
        "preregistration": "manifests/portable.json",
        "preregistration_file_sha256": _sha256(preregistration),
        "source_run_root": "artifacts/source",
        "source_train_receipt": "manifests/source.json",
        "source_train_receipt_file_sha256": _sha256(source_receipt),
        "snapshot_ledger_hash": projected_hash,
    }
    assert set(source) == portable._DIAGNOSTIC_TYPED_SOURCE_FIELDS
    protocol = PaperProtocol(
        path=manifests / "protocol.json",
        payload={"protocol_version": "3.20.0"},
    )

    loaded = skilllearn_experiment._load_typed_selection_for_execution(
        root=project,
        manifest_path=manifests / "manifest.json",
        protocol=protocol,
        execution_contract={"typed_selection_snapshot_source": source},
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        integration_diagnostic_policy=(
            portable.TYPED_PORTABLE_INTEGRATION_VERSION
        ),
    )
    assert loaded is fake_ledger
    assert not (manifests / "integration-result.json").exists()

    with pytest.raises(PermissionError, match="diagnostic policy"):
        skilllearn_experiment._load_typed_selection_for_execution(
            root=project,
            manifest_path=manifests / "manifest.json",
            protocol=protocol,
            execution_contract={"typed_selection_snapshot_source": source},
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            integration_diagnostic_policy=(
                portable.TYPED_SELECTION_INTEGRATION_VERSION
            ),
        )
    with pytest.raises(PermissionError, match="snapshot ledger drifted"):
        skilllearn_experiment._load_typed_selection_for_execution(
            root=project,
            manifest_path=manifests / "manifest.json",
            protocol=protocol,
            execution_contract={
                "typed_selection_snapshot_source": {
                    **source,
                    "snapshot_ledger_hash": "b" * 64,
                }
            },
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            integration_diagnostic_policy=(
                portable.TYPED_PORTABLE_INTEGRATION_VERSION
            ),
        )


def test_docker_container_name_probe_is_read_only_and_fail_closed():
    class Delegate:
        def __init__(self, *, stdout: str = "", returncode: int = 0):
            self.stdout = stdout
            self.returncode = returncode
            self.commands = []

        def run(self, command, **kwargs):
            self.commands.append(list(command))
            return SimpleNamespace(
                returncode=self.returncode,
                stdout=self.stdout,
            )

    absent = Delegate(stdout="")
    assert portable._docker_container_name_present(absent, "probe") is False
    assert absent.commands[0][1:4] == ["container", "ls", "--all"]
    assert "rm" not in absent.commands[0]

    present = Delegate(stdout="probe\n")
    assert portable._docker_container_name_present(present, "probe") is True

    failed = Delegate(returncode=1)
    with pytest.raises(RuntimeError, match="docker_command_failed"):
        portable._docker_container_name_present(failed, "probe")


def test_completed_artifact_validation_requires_exact_event_and_lock_replay():
    acceptance = {key: True for key in portable._ACCEPTANCE_PREDICATES}
    preregistration = {"acceptance": acceptance}
    projected_ledger_hash = "b" * 64
    production_loader = {
        "protocol_version": "3.20.0",
        "integration_diagnostic_policy": (
            portable.TYPED_PORTABLE_INTEGRATION_VERSION
        ),
        "legacy_integration_policy": (
            portable.TYPED_SELECTION_INTEGRATION_VERSION
        ),
        "protocol_contract_hash": "d" * 64,
        "diagnostic_source_fields": sorted(
            portable._DIAGNOSTIC_TYPED_SOURCE_FIELDS
        ),
        "diagnostic_source_field_set_hash": stable_hash(
            {
                "fields": sorted(
                    portable._DIAGNOSTIC_TYPED_SOURCE_FIELDS
                )
            }
        ),
        "diagnostic_source_omits_result_receipt_fields": True,
        "integration_result_receipt_required": False,
        "integration_result_receipt_used": False,
        "same_projected_ledger_loaded": True,
        "projected_snapshot_ledger_hash": projected_ledger_hash,
        "legacy_full_snapshot_ledger_hash": "c" * 64,
        "legacy_integration_policy_rejected": True,
        "legacy_full_snapshot_ledger_rejected": True,
        "freeze_authorization_present": False,
        "diagnostic_freeze_authority_rejected": True,
        "fresh_development_protocol_freeze_eligible": False,
        "development_task_execution_authorized": False,
    }
    predecision = {
        "event": "typed_portable_production_authorization_loader_verified",
        "stage": "fixture",
        "trace_id": "fixture",
        "payload": dict(production_loader),
    }
    runtime_canaries = [
        {
            "item_id_hash": str(index) * 64,
            "container_cleanup_verified": True,
        }
        for index in range(1, 4)
    ]
    report = {
        "integration_policy": portable.TYPED_PORTABLE_INTEGRATION_VERSION,
        "integration_passed": True,
        "acceptance": acceptance,
        "fresh_development_protocol_freeze_eligible_if_passed": True,
        "development_task_execution_currently_authorized": False,
        "decision_hash": "a" * 64,
        "portable_projection": {
            "projected_snapshot_ledger_hash": projected_ledger_hash,
        },
        "production_authorization_loader": production_loader,
        "pre_agent_runtime": {
            "container_cleanup_verified": True,
            "canaries": runtime_canaries,
            "canary_set_hash": stable_hash(
                {"canaries": runtime_canaries}
            ),
        },
        "offline_boundary_contract": {
            "issues": [],
            "predecision_event_count": 1,
            "predecision_event_counts": {
                "typed_portable_production_authorization_loader_verified": 1
            },
            "predecision_event_set_hash": stable_hash(
                {"events": [predecision]}
            ),
        },
    }
    report["report_hash"] = stable_hash(report)
    completion = {
        "event": "typed_portable_integration_completed",
        "stage": "fixture",
        "trace_id": "fixture",
        "payload": {
            "decision_hash": report["decision_hash"],
            "report_hash": report["report_hash"],
            "integration_passed": True,
        },
    }
    preregistration_hash = stable_hash(preregistration)
    lock = portable._completed_decision_lock(
        report=report,
        preregistration_hash=preregistration_hash,
    )

    portable._validate_completed_result_artifacts(
        preregistration=preregistration,
        report=report,
        events=[predecision, completion],
        decision_lock=lock,
    )
    with pytest.raises(PermissionError, match="event commitment drifted"):
        portable._validate_completed_result_artifacts(
            preregistration=preregistration,
            report=report,
            events=[{**predecision, "trace_id": "tampered"}, completion],
            decision_lock=lock,
        )
    with pytest.raises(PermissionError, match="decision lock"):
        portable._validate_completed_result_artifacts(
            preregistration=preregistration,
            report=report,
            events=[predecision, completion],
            decision_lock={**lock, "state": "reserved"},
        )

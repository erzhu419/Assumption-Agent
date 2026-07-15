from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks.skilllearn_compiler import (
    SkillLearnProgramCompiler,
)
from assumption_agent.benchmarks.typed_task_capability import (
    PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
    portable_role_spec_for_bound_recipe,
    validate_compiled_portable_task_capability,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.models import HypothesisStatus, stable_hash
from assumption_agent.splits import BenchmarkItem, SplitManifest
from assumption_agent.typed_operator_grammar import (
    ArtifactFormat,
    ArtifactSpec,
    FamilyCapabilityGraph,
    TypedProgramBindingRegistry,
    WorkflowKind,
    _artifact_id,
    _build_recipe,
    _capability_spec,
    _workflows_for_format,
    canonical_typed_recipe_selection_request,
    canonical_typed_recipe_selection_response,
    freeze_typed_recipe_selection_snapshot,
    freeze_typed_selection_snapshot_ledger,
    materialize_recipe_selection,
)


STOCK_FAMILY = "stock-data-visualization"
TRAIN_LITERAL = "/root/data/stock-descriptions.csv"


def _graph(
    *,
    family: str = STOCK_FAMILY,
    locator: str = TRAIN_LITERAL,
    artifact_format: ArtifactFormat = ArtifactFormat.TABULAR,
) -> FamilyCapabilityGraph:
    provenance_hash = "a" * 64
    relations = ("declared_task_local_path",)
    artifact = ArtifactSpec(
        artifact_id=_artifact_id(
            target_family=family,
            locator=locator,
            artifact_format=artifact_format,
            support_count=2,
            evidence_relations=relations,
            provenance_hash=provenance_hash,
        ),
        locator=locator,
        format=artifact_format,
        support_count=2,
        evidence_relations=relations,
        provenance_hash=provenance_hash,
    )
    capability = _capability_spec(
        target_family=family,
        artifact=artifact,
    )
    graph = FamilyCapabilityGraph(
        target_family=family,
        source_evidence_hash="b" * 64,
        artifacts=(artifact,),
        capabilities=(capability,),
        recipes=tuple(
            sorted(
                (
                    _build_recipe(
                        target_family=family,
                        artifact=artifact,
                        capability=capability,
                        workflow=workflow,
                    )
                    for workflow in _workflows_for_format(artifact.format)
                ),
                key=lambda row: row.recipe_id,
            )
        ),
    )
    assert graph.validate() == ()
    return graph


def _bound_program(
    *,
    family: str = STOCK_FAMILY,
    locator: str = TRAIN_LITERAL,
    artifact_format: ArtifactFormat = ArtifactFormat.TABULAR,
    workflow: WorkflowKind = WorkflowKind.BUILD_VISUALIZATION,
):
    graph = _graph(
        family=family,
        locator=locator,
        artifact_format=artifact_format,
    )
    recipe = next(
        row
        for row in graph.recipes
        if row.workflow is workflow
    )
    snapshot = freeze_typed_recipe_selection_snapshot(graph)
    graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": graph.target_family_hash,
                    "graph_hash": snapshot.expected_graph_hash,
                    "availability_error_hash": None,
                }
            ]
        }
    )
    catalog_set_hash = stable_hash(
        {"catalog_hashes": [snapshot.expected_model_catalog_hash]}
    )
    ledger = freeze_typed_selection_snapshot_ledger(
        (snapshot,),
        feasibility_preregistration_hash=stable_hash({"fixture": "prereg"}),
        feasibility_result_receipt_sha256=stable_hash({"fixture": "result"}),
        feasibility_decision_hash=stable_hash({"fixture": "decision"}),
        feasibility_report_hash=stable_hash({"fixture": "report"}),
        manifest_hash=stable_hash({"fixture": "manifest"}),
        source_train_receipt_hash=stable_hash({"fixture": "source"}),
        expected_graph_set_hash=graph_set_hash,
        expected_model_catalog_set_hash=catalog_set_hash,
        expected_target_family_hashes=(graph.target_family_hash,),
    )
    evaluator_epoch = "portable-compiler-fixture-v1"
    program = materialize_recipe_selection(
        {"recipe_id": recipe.recipe_id},
        graph=graph,
        evaluator_epoch=evaluator_epoch,
        expected_graph_hash=snapshot.expected_graph_hash,
        expected_model_catalog_hash=snapshot.expected_model_catalog_hash,
    )
    request = canonical_typed_recipe_selection_request(
        snapshot=snapshot,
        snapshot_ledger=ledger,
        evaluator_epoch=evaluator_epoch,
        selection_round=1,
    )
    response = canonical_typed_recipe_selection_response(recipe.recipe_id)
    registry = TypedProgramBindingRegistry(snapshot_ledger=ledger)
    registry.register(
        program,
        snapshot=snapshot,
        recipe_id=recipe.recipe_id,
        request_kind=request["request_kind"],
        request_hash=stable_hash(request),
        response_hash=stable_hash(response),
        selection_round=1,
    )
    return program, registry


def _items_and_manifest(family: str = STOCK_FAMILY):
    train_ids = (f"{family}-1", f"{family}-3")
    family_by_id = {
        train_ids[0]: family,
        train_ids[1]: family,
        "validation-item": "validation-family",
        "test-item": "test-family",
    }
    manifest = SplitManifest(
        benchmark="synthetic",
        protocol="instance_holdout",
        seed="portable-compiler",
        train_ids=train_ids,
        validation_ids=("validation-item",),
        test_ids=("test-item",),
        family_by_id=family_by_id,
    )
    items = tuple(
        BenchmarkItem(
            id=item_id,
            family=family,
            features={"family": family},
            content_ref=f"public://{item_id}",
            verifier_ref_hash=stable_hash({"verifier": item_id}),
        )
        for item_id in train_ids
    )
    return items, manifest


def _compile(
    tmp_path: Path,
    *,
    family: str = STOCK_FAMILY,
    locator: str = TRAIN_LITERAL,
    artifact_format: ArtifactFormat = ArtifactFormat.TABULAR,
    workflow: WorkflowKind = WorkflowKind.BUILD_VISUALIZATION,
    portable: bool = True,
):
    program, registry = _bound_program(
        family=family,
        locator=locator,
        artifact_format=artifact_format,
        workflow=workflow,
    )
    items, manifest = _items_and_manifest(family)
    sink = MemoryEventSink()
    compiler = SkillLearnProgramCompiler(
        event_sink=sink,
        typed_program_registry=registry,
        require_typed_bindings=True,
        portable_capability_compiler_mode=(
            PORTABLE_TASK_CAPABILITY_COMPILER_VERSION if portable else None
        ),
    )
    result = compiler.compile(
        programs=(program,),
        items=items,
        split_manifest=manifest,
        output_root=tmp_path,
        allowed_statuses={HypothesisStatus.CANDIDATE},
        target_item_ids=manifest.train_ids,
        target_split="train",
    )
    return result, sink, manifest


def test_portable_compiler_emits_locator_free_bound_metadata_and_receipts(
    tmp_path: Path,
) -> None:
    result, sink, manifest = _compile(tmp_path)

    all_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(result.output_root.rglob("*"))
        if path.is_file()
    )
    event_text = json.dumps(sink.events, sort_keys=True)
    assert TRAIN_LITERAL not in all_text
    assert TRAIN_LITERAL not in event_text
    assert "task_declared_primary_input" in all_text
    assert "/root/.assumption-v2/capabilities/portable-" in all_text

    compile_manifest = json.loads(
        (result.output_root / "compile_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert compile_manifest["portable_capability_compiler_mode"] == (
        PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
    )
    rows = compile_manifest["portable_capability_role_spec_rows"]
    assert len(rows) == 2
    assert compile_manifest["source_artifact_locators_persisted"] is False
    for row in rows:
        metadata = validate_compiled_portable_task_capability(
            json.loads((result.output_root / row["metadata_path"]).read_text())
        )
        assert metadata.role_spec.role_spec_hash == row["role_spec_hash"]
        assert metadata.item_id_hash == row["item_id_hash"]

    for item_id in manifest.train_ids:
        receipt = result.source_receipt_for(item_id)
        assert receipt.portable_capability_compiler_mode == (
            PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        )
        assert len(receipt.portable_capability_role_spec_hashes) == 1
        assert len(receipt.portable_capability_metadata_file_hashes) == 1
        assert receipt.to_dict()[
            "portable_capability_role_spec_set_hash"
        ] == result.portable_capability_role_spec_set_hash

    portable_events = [
        row
        for row in sink.events
        if row["event"] == "skilllearn_skill_compiled"
    ]
    assert len(portable_events) == 2
    assert all(
        row["payload"]["source_artifact_locator_disclosed"] is False
        for row in portable_events
    )


def test_portable_compiler_metadata_mutation_fails_source_receipt(
    tmp_path: Path,
) -> None:
    result, _, manifest = _compile(tmp_path)
    metadata_path = next(
        iter(
            result.item_portable_capability_metadata_paths[
                stable_hash({"item_id": manifest.train_ids[0]})
            ]
        )
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["source_artifact_locator_disclosed"] = True
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        result.source_receipt_for(manifest.train_ids[0])


def test_portable_compiler_fails_closed_without_prompt_fallback(
    tmp_path: Path,
) -> None:
    unsupported_family = "unsupported-tabular-family"
    with pytest.raises(
        PermissionError,
        match="family has no supported portable task capability",
    ):
        _compile(
            tmp_path,
            family=unsupported_family,
            locator="/root/data/unsupported.csv",
        )
    assert not (tmp_path / "assumption-agent-v2").exists()

    with pytest.raises(ValueError, match="requires typed bindings"):
        SkillLearnProgramCompiler(
            portable_capability_compiler_mode=(
                PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
            )
        )
    with pytest.raises(ValueError, match="unsupported"):
        SkillLearnProgramCompiler(
            portable_capability_compiler_mode="future-mode",
        )


def test_portable_role_spec_rejects_tampered_bound_snapshot_hash() -> None:
    program, registry = _bound_program()
    bound = registry.require_bound_recipe(program)
    tampered = replace(
        bound,
        binding=replace(bound.binding, snapshot_hash="f" * 64),
    )

    with pytest.raises(PermissionError, match="not canonical"):
        portable_role_spec_for_bound_recipe(tampered)


def test_portable_mode_off_preserves_legacy_compiler_bytes(
    tmp_path: Path,
) -> None:
    first, _, _ = _compile(tmp_path / "first", portable=False)
    second, _, _ = _compile(tmp_path / "second", portable=False)

    first_files = {
        path.relative_to(first.output_root).as_posix(): path.read_bytes()
        for path in first.output_root.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.output_root).as_posix(): path.read_bytes()
        for path in second.output_root.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    assert first.manifest_hash == second.manifest_hash
    assert first.treatment_hash == second.treatment_hash
    manifest = json.loads(first_files["compile_manifest.json"])
    assert not any(key.startswith("portable_capability") for key in manifest)
    assert "source_artifact_locators_persisted" not in manifest


@pytest.mark.parametrize(
    (
        "family",
        "locator",
        "artifact_format",
        "workflow",
        "expected_role",
    ),
    (
        (
            "temperature-simulation",
            "/root/field_temp_oxy.csv",
            ArtifactFormat.TABULAR,
            WorkflowKind.DERIVE_TASK_OUTPUT,
            "task_declared_primary_input",
        ),
        (
            "organize-messy-files",
            "/root/paper_file_1.docx",
            ArtifactFormat.OFFICE_DOCUMENT,
            WorkflowKind.ORGANIZE_COLLECTION,
            "task_declared_office_document_collection",
        ),
    ),
)
def test_portable_compiler_preserves_selected_workflow_without_family_aliasing(
    tmp_path: Path,
    family: str,
    locator: str,
    artifact_format: ArtifactFormat,
    workflow: WorkflowKind,
    expected_role: str,
) -> None:
    result, _, _ = _compile(
        tmp_path,
        family=family,
        locator=locator,
        artifact_format=artifact_format,
        workflow=workflow,
    )
    skill_text = "\n".join(
        path.read_text(encoding="utf-8") for path in result.skill_paths
    )
    graph = _graph(
        family=family,
        locator=locator,
        artifact_format=artifact_format,
    )
    selected = next(row for row in graph.recipes if row.workflow is workflow)

    assert "Complete the visualization" not in skill_text
    assert "Portable current-item tabular workflow" not in skill_text
    assert f"Selected workflow: `{workflow.value}`" in skill_text
    assert f"`{workflow.value}` workflow" in skill_text
    assert "supplies read-only artifact evidence only" in skill_text
    assert "remaining workflow operators are agent-executed" in skill_text
    assert "do not substitute model-authored primitives" not in skill_text
    assert expected_role in skill_text
    for node in selected.nodes:
        assert node.kind.value in skill_text
    if family == "organize-messy-files":
        assert "declared tabular input" not in skill_text
        assert "visualization requested" not in skill_text

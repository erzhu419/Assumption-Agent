from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks.typed_task_capability import (
    INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY,
    PROFILE_DELIMITED_TABLE_CAPABILITY,
    TASK_DECLARED_OFFICE_COLLECTION_ROLE,
    TASK_DECLARED_PRIMARY_INPUT_ROLE,
    OfficeCollectionCapabilityEffectReceipt,
    PortableFamilyCapabilityGraph,
    PortableArtifactResolutionError,
    PortableArtifactRoleSpec,
    RestrictedCapabilityExecutionError,
    execute_restricted_task_capability,
    portable_role_spec_for_recipe,
    project_portable_family_capability_graph,
    resolve_portable_artifact_role,
    verify_task_capability_effect,
)
from assumption_agent.benchmarks.train_proposal_diagnostic import (
    reconstruct_v315_train_evidence,
)
from assumption_agent.benchmarks.typed_selection_integration import (
    _extract_all_train_trials,
)
from assumption_agent.events import NullEventSink
from assumption_agent.splits import SplitManifest
from assumption_agent.typed_operator_grammar import (
    MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
    ArtifactFormat,
    ArtifactSpec,
    FamilyCapabilityGraph,
    WorkflowKind,
    _artifact_id,
    _artifact_rank,
    _build_recipe,
    _capability_spec,
    _workflows_for_format,
    build_family_capability_graph,
    canonical_recipe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = (
    REPO_ROOT
    / "reference"
    / "self_evo_continual_20260707"
    / "repos"
    / "SkillLearnBench"
)
STOCK_FAMILY = "stock-data-visualization"
TEMPERATURE_FAMILY = "temperature-simulation"
ORGANIZE_FAMILY = "organize-messy-files"
TRAIN_LITERAL = "/root/data/stock-descriptions.csv"


def _stock_graph() -> FamilyCapabilityGraph:
    provenance_hash = "a" * 64
    relations = ("declared_task_local_path",)
    artifact = ArtifactSpec(
        artifact_id=_artifact_id(
            target_family=STOCK_FAMILY,
            locator=TRAIN_LITERAL,
            artifact_format=ArtifactFormat.TABULAR,
            support_count=2,
            evidence_relations=relations,
            provenance_hash=provenance_hash,
        ),
        locator=TRAIN_LITERAL,
        format=ArtifactFormat.TABULAR,
        support_count=2,
        evidence_relations=relations,
        provenance_hash=provenance_hash,
    )
    capability = _capability_spec(
        target_family=STOCK_FAMILY,
        artifact=artifact,
    )
    recipes = tuple(
        sorted(
            (
                _build_recipe(
                    target_family=STOCK_FAMILY,
                    artifact=artifact,
                    capability=capability,
                    workflow=workflow,
                )
                for workflow in _workflows_for_format(artifact.format)
            ),
            key=lambda row: row.recipe_id,
        )
    )
    graph = FamilyCapabilityGraph(
        target_family=STOCK_FAMILY,
        source_evidence_hash="b" * 64,
        artifacts=(artifact,),
        capabilities=(capability,),
        recipes=recipes,
    )
    assert graph.validate() == ()
    return graph


def _mixed_graph(
    family: str,
    artifacts_by_locator: dict[str, ArtifactFormat],
) -> FamilyCapabilityGraph:
    artifacts = []
    for index, (locator, artifact_format) in enumerate(
        artifacts_by_locator.items(),
        start=1,
    ):
        provenance_hash = f"{index:x}" * 64
        relations = ("declared_task_local_path",)
        artifacts.append(
            ArtifactSpec(
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
        )
    ordered_artifacts = tuple(sorted(artifacts, key=_artifact_rank))
    capabilities = tuple(
        _capability_spec(target_family=family, artifact=artifact)
        for artifact in ordered_artifacts
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
                for artifact, capability in zip(
                    ordered_artifacts,
                    capabilities,
                )
                for workflow in _workflows_for_format(artifact.format)
            ),
            key=lambda row: row.recipe_id,
        )
    )
    graph = FamilyCapabilityGraph(
        target_family=family,
        source_evidence_hash="f" * 64,
        artifacts=ordered_artifacts,
        capabilities=capabilities,
        recipes=recipes,
    )
    assert graph.validate() == ()
    return graph


def _portable_graph_for_family(family: str) -> PortableFamilyCapabilityGraph:
    if family == STOCK_FAMILY:
        graph = _mixed_graph(
            family,
            {
                "/root/data/stock-descriptions.csv": ArtifactFormat.TABULAR,
                "/root/output/index.html": ArtifactFormat.WEB_ASSET,
            },
        )
    elif family == TEMPERATURE_FAMILY:
        graph = _mixed_graph(
            family,
            {
                "/root/field_temp_oxy.csv": ArtifactFormat.TABULAR,
                "/root/glm3.nml": ArtifactFormat.CONFIGURATION,
            },
        )
    elif family == ORGANIZE_FAMILY:
        graph = _mixed_graph(
            family,
            {
                "/root/paper_file_1.docx": ArtifactFormat.OFFICE_DOCUMENT,
                "/root/readme.txt": ArtifactFormat.TEXT,
            },
        )
    else:
        raise AssertionError(f"unknown fixture family: {family}")
    return project_portable_family_capability_graph(graph)


def _visualization_role() -> PortableArtifactRoleSpec:
    graph = _stock_graph()
    recipe = next(
        row
        for row in graph.recipes
        if row.workflow is WorkflowKind.BUILD_VISUALIZATION
    )
    return portable_role_spec_for_recipe(graph, recipe.recipe_id)


def _mirror_stock_item(tmp_path: Path, item_id: str) -> tuple[Path, str]:
    item_root = BENCHMARK_ROOT / "tasks" / STOCK_FAMILY / item_id
    instruction = (item_root / "instruction.md").read_text(encoding="utf-8")
    source_data = item_root / "environment" / "data"
    runtime_root = tmp_path / item_id
    runtime_data = runtime_root / "data"
    runtime_data.mkdir(parents=True)
    for source in source_data.glob("stock-descriptions.*"):
        shutil.copy2(source, runtime_data / source.name)
    return runtime_root, instruction


def test_current_item_role_executes_tsv_and_emits_exact_effect_receipt(
    tmp_path: Path,
) -> None:
    spec = _visualization_role()
    runtime_root, instruction = _mirror_stock_item(
        tmp_path,
        "stock-data-visualization-3",
    )

    spec_text = json.dumps(spec.safe_payload(), sort_keys=True)
    assert TRAIN_LITERAL not in spec_text
    assert spec.role == TASK_DECLARED_PRIMARY_INPUT_ROLE
    assert spec.capability == PROFILE_DELIMITED_TABLE_CAPABILITY

    binding = resolve_portable_artifact_role(
        spec,
        item_id="stock-data-visualization-3",
        public_instruction=instruction,
        runtime_root=runtime_root,
    )
    binding_text = json.dumps(binding.safe_payload(), sort_keys=True)
    assert binding.container_locator == "/root/data/stock-descriptions.tsv"
    assert TRAIN_LITERAL not in binding_text
    assert binding.container_locator not in binding_text

    receipt = execute_restricted_task_capability(
        binding,
        runtime_root=runtime_root,
    )
    verified = verify_task_capability_effect(
        receipt,
        runtime_root=runtime_root,
    )

    assert verified == receipt.safe_payload()
    assert receipt.record_count == 50
    assert receipt.column_count == 14
    assert receipt.minimum_record_width == 14
    assert receipt.maximum_record_width == 14
    assert receipt.input_before_sha256 == receipt.input_after_sha256
    assert receipt.output_host_path.is_file()
    assert receipt.agent_payload["profile_locator"].startswith(
        "/root/.assumption-v2/capabilities/"
    )
    assert receipt.agent_payload["source_artifact_locator_disclosed"] is False
    receipt_text = json.dumps(receipt.safe_payload(), sort_keys=True)
    assert TRAIN_LITERAL not in receipt_text
    assert binding.container_locator not in receipt_text


def test_frozen_train_graph_executes_current_item_offline(
    tmp_path: Path,
) -> None:
    manifest = SplitManifest.read(
        REPO_ROOT
        / "manifests"
        / "skilllearnbench_instance_holdout_offline_ready_v1.json"
    )
    source_run_root = (
        REPO_ROOT
        / "artifacts"
        / "paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01"
    )
    evidence = reconstruct_v315_train_evidence(
        root=BENCHMARK_ROOT,
        manifest=manifest,
        source_run_root=source_run_root,
        source_train_receipt=(
            REPO_ROOT
            / "manifests"
            / "skilllearn_v315_train_source_provenance_receipt_v1.json"
        ),
        event_sink=NullEventSink(),
    )
    trials = _extract_all_train_trials(
        evidence=evidence,
        source_run_root=source_run_root,
    )
    graph = build_family_capability_graph(
        target_family=STOCK_FAMILY,
        failures=evidence.failures,
        action_profiles=evidence.action_profiles,
        trial_evidence={row.trial_id_hash: row for row in trials},
        minimum_support=2,
        maximum_artifacts=MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
    )
    assert (
        graph.graph_hash
        == "658db644dfdd99a29c243e836a699fa84e7d6d2bdd4f1cf938a231d8010074e8"
    )
    recipe = canonical_recipe(graph)
    assert recipe.recipe_id == "recipe_e46897fbd1773d6c2549"
    spec = portable_role_spec_for_recipe(graph, recipe.recipe_id)
    runtime_root, instruction = _mirror_stock_item(
        tmp_path,
        "stock-data-visualization-3",
    )
    binding = resolve_portable_artifact_role(
        spec,
        item_id="stock-data-visualization-3",
        public_instruction=instruction,
        runtime_root=runtime_root,
    )
    receipt = execute_restricted_task_capability(
        binding,
        runtime_root=runtime_root,
    )

    assert binding.binding_hash == (
        "63e7a9c6ea1e56b560e756c05d2216429006a0335ad27c2a90d8643cb015852b"
    )
    assert receipt.receipt_hash == (
        "c45bd4eb65b6be0e18155539734c535320480d03fdb9c7c700901020186744f8"
    )
    assert receipt.input_before_sha256 == (
        "dc9ffc578317b476d01f8133919a95479e05d94b022bd699ff9f2dddeaf65215"
    )
    assert receipt.record_count == 50
    assert receipt.column_count == 14
    assert verify_task_capability_effect(
        receipt,
        runtime_root=runtime_root,
    )["receipt_hash"] == receipt.receipt_hash


def test_same_typed_role_resolves_each_current_item_not_train_literal(
    tmp_path: Path,
) -> None:
    spec = _visualization_role()
    csv_root, csv_instruction = _mirror_stock_item(
        tmp_path,
        "stock-data-visualization-1",
    )
    tsv_root, tsv_instruction = _mirror_stock_item(
        tmp_path,
        "stock-data-visualization-3",
    )

    csv_binding = resolve_portable_artifact_role(
        spec,
        item_id="stock-data-visualization-1",
        public_instruction=csv_instruction,
        runtime_root=csv_root,
    )
    tsv_binding = resolve_portable_artifact_role(
        spec,
        item_id="stock-data-visualization-3",
        public_instruction=tsv_instruction,
        runtime_root=tsv_root,
    )
    csv_receipt = execute_restricted_task_capability(
        csv_binding,
        runtime_root=csv_root,
    )
    tsv_receipt = execute_restricted_task_capability(
        tsv_binding,
        runtime_root=tsv_root,
    )

    assert csv_binding.container_locator.endswith(".csv")
    assert tsv_binding.container_locator.endswith(".tsv")
    assert csv_binding.spec.role_spec_hash == tsv_binding.spec.role_spec_hash
    assert csv_binding.binding_hash != tsv_binding.binding_hash
    assert csv_receipt.delimiter_kind == "comma"
    assert tsv_receipt.delimiter_kind == "tab"
    assert csv_receipt.record_count == tsv_receipt.record_count == 50


def test_role_resolution_and_capability_fail_closed_on_tamper(
    tmp_path: Path,
) -> None:
    spec = _visualization_role()
    runtime_root = tmp_path / "runtime"
    data = runtime_root / "data"
    data.mkdir(parents=True)
    (data / "one.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (data / "two.tsv").write_text("a\tb\n1\t2\n", encoding="utf-8")

    with pytest.raises(
        PortableArtifactResolutionError,
        match="exactly one",
    ):
        resolve_portable_artifact_role(
            spec,
            item_id="ambiguous-item",
            public_instruction=(
                "Use /root/data/one.csv and /root/data/two.tsv as inputs."
            ),
            runtime_root=runtime_root,
        )

    escape_root = tmp_path / "escape-runtime"
    escape_root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n", encoding="utf-8")
    (escape_root / "data").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(PortableArtifactResolutionError, match="found 0"):
        resolve_portable_artifact_role(
            spec,
            item_id="symlink-item",
            public_instruction="Use /root/data/outside.csv.",
            runtime_root=escape_root,
        )

    single_root = tmp_path / "single-runtime"
    (single_root / "data").mkdir(parents=True)
    source = single_root / "data" / "one.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    binding = resolve_portable_artifact_role(
        spec,
        item_id="single-item",
        public_instruction="Use /root/data/one.csv.",
        runtime_root=single_root,
    )
    source.write_text("a,b\n3,4\n", encoding="utf-8")
    with pytest.raises(
        RestrictedCapabilityExecutionError,
        match="changed before",
    ):
        execute_restricted_task_capability(
            binding,
            runtime_root=single_root,
        )

    forged_spec = replace(spec, capability="arbitrary_shell_command")
    with pytest.raises(
        PortableArtifactResolutionError,
        match="closed registry",
    ):
        resolve_portable_artifact_role(
            forged_spec,
            item_id="single-item",
            public_instruction="Use /root/data/one.csv.",
            runtime_root=single_root,
        )


def test_effect_verification_detects_output_mutation(tmp_path: Path) -> None:
    spec = _visualization_role()
    runtime_root, instruction = _mirror_stock_item(
        tmp_path,
        "stock-data-visualization-3",
    )
    binding = resolve_portable_artifact_role(
        spec,
        item_id="stock-data-visualization-3",
        public_instruction=instruction,
        runtime_root=runtime_root,
    )
    receipt = execute_restricted_task_capability(
        binding,
        runtime_root=runtime_root,
    )
    receipt.output_host_path.write_bytes(
        receipt.output_host_path.read_bytes() + b" "
    )
    with pytest.raises(PermissionError, match="output hash"):
        verify_task_capability_effect(
            receipt,
            runtime_root=runtime_root,
        )


def test_effect_output_is_create_only(tmp_path: Path) -> None:
    spec = _visualization_role()
    runtime_root, instruction = _mirror_stock_item(
        tmp_path,
        "stock-data-visualization-1",
    )
    binding = resolve_portable_artifact_role(
        spec,
        item_id="stock-data-visualization-1",
        public_instruction=instruction,
        runtime_root=runtime_root,
    )
    execute_restricted_task_capability(binding, runtime_root=runtime_root)
    with pytest.raises(
        RestrictedCapabilityExecutionError,
        match="already exists",
    ):
        execute_restricted_task_capability(binding, runtime_root=runtime_root)


@pytest.mark.parametrize(
    ("family", "expected_format", "expected_capability"),
    (
        (
            STOCK_FAMILY,
            ArtifactFormat.TABULAR,
            PROFILE_DELIMITED_TABLE_CAPABILITY,
        ),
        (
            TEMPERATURE_FAMILY,
            ArtifactFormat.TABULAR,
            PROFILE_DELIMITED_TABLE_CAPABILITY,
        ),
        (
            ORGANIZE_FAMILY,
            ArtifactFormat.OFFICE_DOCUMENT,
            INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY,
        ),
    ),
)
def test_portable_projection_exposes_only_complete_executable_recipe_sets(
    family: str,
    expected_format: ArtifactFormat,
    expected_capability: str,
) -> None:
    projected = _portable_graph_for_family(family)

    assert isinstance(projected, PortableFamilyCapabilityGraph)
    assert projected.validate() == ()
    assert {row.format for row in projected.artifacts} == {expected_format}
    assert project_portable_family_capability_graph(projected) is projected
    for recipe in projected.recipes:
        spec = portable_role_spec_for_recipe(projected, recipe.recipe_id)
        assert spec.artifact_format is expected_format
        assert spec.capability == expected_capability

    for catalog in (projected.model_catalog(), projected.safe_payload()):
        commitment = catalog["portable_capability_projection"]
        assert all(
            row["capability_implementation_verified"] is False
            and row["runtime_agent_argument_surface_restricted"] is False
            and row[
                "pre_agent_artifact_evidence_capability_verified"
            ] is True
            and row[
                "pre_agent_evidence_argument_surface_restricted"
            ] is True
            and row["runtime_agent_capability_invocation_available"]
            is False
            and row["full_recipe_operator_execution_by_capability"]
            is False
            for row in catalog["capabilities"]
        )
        assert commitment[
            "all_selectable_recipes_artifact_evidence_backed"
        ] is True
        assert commitment[
            "pre_agent_evidence_argument_surface_restricted"
        ] is True
        assert commitment[
            "capability_execution_covers_full_recipe_operator_plan"
        ] is False
        assert commitment["non_access_operators_are_fixed_agent_plan"] is True
        assert commitment["model_authored_locator_allowed"] is False
        assert commitment[
            "model_authored_capability_arguments_allowed"
        ] is False
    selector_context = projected.model_catalog()[
        "portable_selector_context"
    ]
    assert selector_context["target_family"] == family
    assert selector_context["target_family_scope"] == (
        "frozen_train_routing_label"
    )
    assert selector_context["selector_output_fields"] == ["recipe_id"]
    assert selector_context["opaque_recipe_id_only_output"] is True


def test_portable_projection_fails_closed_for_unsupported_family_and_format(
) -> None:
    unsupported_family = _mixed_graph(
        "unsupported-family",
        {"/root/input.csv": ArtifactFormat.TABULAR},
    )
    unsupported_format = _mixed_graph(
        STOCK_FAMILY,
        {"/root/input.json": ArtifactFormat.STRUCTURED_RECORD},
    )
    for graph in (unsupported_family, unsupported_format):
        with pytest.raises(PermissionError, match="no complete portable"):
            project_portable_family_capability_graph(graph)


def test_frozen_train_catalogs_project_to_executable_three_family_surface(
) -> None:
    manifest = SplitManifest.read(
        REPO_ROOT
        / "manifests"
        / "skilllearnbench_instance_holdout_offline_ready_v1.json"
    )
    source_run_root = (
        REPO_ROOT
        / "artifacts"
        / "paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01"
    )
    evidence = reconstruct_v315_train_evidence(
        root=BENCHMARK_ROOT,
        manifest=manifest,
        source_run_root=source_run_root,
        source_train_receipt=(
            REPO_ROOT
            / "manifests"
            / "skilllearn_v315_train_source_provenance_receipt_v1.json"
        ),
        event_sink=NullEventSink(),
    )
    trials = _extract_all_train_trials(
        evidence=evidence,
        source_run_root=source_run_root,
    )
    trial_map = {row.trial_id_hash: row for row in trials}
    expected = {
        STOCK_FAMILY: (ArtifactFormat.TABULAR, 1, 3, 1),
        TEMPERATURE_FAMILY: (ArtifactFormat.TABULAR, 1, 3, 2),
        ORGANIZE_FAMILY: (ArtifactFormat.OFFICE_DOCUMENT, 1, 3, 5),
    }
    for family, (
        expected_format,
        expected_artifact_count,
        expected_recipe_count,
        expected_alias_count,
    ) in expected.items():
        graph = build_family_capability_graph(
            target_family=family,
            failures=evidence.failures,
            action_profiles=evidence.action_profiles,
            trial_evidence=trial_map,
            minimum_support=2,
            maximum_artifacts=MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
        )
        projected = project_portable_family_capability_graph(graph)

        assert projected.validate() == ()
        assert len(projected.artifacts) == expected_artifact_count
        assert len(projected.recipes) == expected_recipe_count
        assert projected.behavioral_alias_count == expected_alias_count
        projection = projected.model_catalog()[
            "portable_capability_projection"
        ]
        assert projection["behavioral_aliases_deduplicated"] is True
        assert projection[
            "diversity_counts_behavioral_signature_only"
        ] is True
        assert projection[
            "behavioral_alias_artifact_count"
        ] == expected_alias_count
        assert {row.format for row in projected.artifacts} == {
            expected_format
        }
        assert all(
            portable_role_spec_for_recipe(projected, recipe.recipe_id)
            for recipe in projected.recipes
        )


def test_current_temperature_item_resolves_existing_delimited_role(
    tmp_path: Path,
) -> None:
    graph = _portable_graph_for_family(TEMPERATURE_FAMILY)
    spec = portable_role_spec_for_recipe(
        graph,
        graph.recipes[0].recipe_id,
    )
    item_root = (
        BENCHMARK_ROOT
        / "tasks"
        / TEMPERATURE_FAMILY
        / "temperature-simulation-1"
    )
    instruction = (item_root / "instruction.md").read_text(encoding="utf-8")
    runtime_root = tmp_path / "temperature-simulation-1"
    runtime_root.mkdir()
    shutil.copy2(
        item_root / "environment" / "field_temp_oxy.csv",
        runtime_root / "field_temp_oxy.csv",
    )
    source = runtime_root / "field_temp_oxy.csv"
    before = source.read_bytes()

    binding = resolve_portable_artifact_role(
        spec,
        item_id="temperature-simulation-1",
        public_instruction=instruction,
        runtime_root=runtime_root,
    )
    receipt = execute_restricted_task_capability(
        binding,
        runtime_root=runtime_root,
    )

    assert binding.container_locator == "/root/field_temp_oxy.csv"
    assert receipt.input_before_sha256 == receipt.input_after_sha256
    assert source.read_bytes() == before
    assert verify_task_capability_effect(
        receipt,
        runtime_root=runtime_root,
    )["receipt_hash"] == receipt.receipt_hash


def test_current_organize_item_resolves_and_inventories_collection_read_only(
    tmp_path: Path,
) -> None:
    graph = _portable_graph_for_family(ORGANIZE_FAMILY)
    spec = portable_role_spec_for_recipe(
        graph,
        graph.recipes[0].recipe_id,
    )
    assert spec.role == TASK_DECLARED_OFFICE_COLLECTION_ROLE
    item_root = (
        BENCHMARK_ROOT
        / "tasks"
        / ORGANIZE_FAMILY
        / "organize-messy-files-3"
    )
    instruction = (item_root / "instruction.md").read_text(encoding="utf-8")
    runtime_root = tmp_path / "organize-messy-files-3"
    collection = runtime_root / "papers" / "all"
    collection.mkdir(parents=True)
    for source in sorted((item_root / "environment").glob("*.docx")):
        shutil.copy2(source, collection / source.name)
    for source in sorted((item_root / "environment").glob("*.pptx")):
        shutil.copy2(source, collection / source.name)
    (collection / "sample.pdf").write_bytes(b"%PDF-1.4\nfixture\n")
    before = {
        path.name: path.read_bytes()
        for path in sorted(collection.iterdir())
    }

    binding = resolve_portable_artifact_role(
        spec,
        item_id="organize-messy-files-3",
        public_instruction=instruction,
        runtime_root=runtime_root,
    )
    receipt = execute_restricted_task_capability(
        binding,
        runtime_root=runtime_root,
    )

    assert isinstance(receipt, OfficeCollectionCapabilityEffectReceipt)
    assert binding.container_locator == "/root/papers/all"
    assert binding.container_locator not in json.dumps(
        binding.safe_payload(), sort_keys=True
    )
    assert receipt.input_before_sha256 == receipt.input_after_sha256
    assert receipt.input_entry_count_before == len(before)
    assert {
        path.name: path.read_bytes()
        for path in sorted(collection.iterdir())
    } == before
    profile = json.loads(receipt.output_host_path.read_text(encoding="utf-8"))
    assert profile["file_count"] == len(before)
    assert all(
        set(row) == {"filename", "extension", "size"}
        for row in profile["files"]
    )
    assert profile["document_content_profiled"] is False
    assert verify_task_capability_effect(
        receipt,
        runtime_root=runtime_root,
    )["receipt_hash"] == receipt.receipt_hash


def test_office_collection_discovery_fails_on_ambiguity_and_links(
    tmp_path: Path,
) -> None:
    graph = _portable_graph_for_family(ORGANIZE_FAMILY)
    spec = portable_role_spec_for_recipe(
        graph,
        graph.recipes[0].recipe_id,
    )
    ambiguous_root = tmp_path / "ambiguous"
    for name in ("one", "two"):
        collection = ambiguous_root / name
        collection.mkdir(parents=True)
        (collection / f"{name}.pdf").write_bytes(b"fixture")
    with pytest.raises(PortableArtifactResolutionError, match="exactly one"):
        resolve_portable_artifact_role(
            spec,
            item_id="organize-messy-files-3",
            public_instruction="Organize the current document collection.",
            runtime_root=ambiguous_root,
        )

    linked_root = tmp_path / "linked"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "paper.pdf").write_bytes(b"fixture")
    linked_root.mkdir()
    (linked_root / "papers").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PortableArtifactResolutionError, match="exactly one"):
        resolve_portable_artifact_role(
            spec,
            item_id="organize-messy-files-3",
            public_instruction="Organize the current document collection.",
            runtime_root=linked_root,
        )

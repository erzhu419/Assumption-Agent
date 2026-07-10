from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks import SkillLearnBenchAdapter, SkillLearnProgramCompiler
from assumption_agent.models import HypothesisProgram
from assumption_agent.splits import (
    AccessPhase,
    SplitAccessGuard,
    build_family_out_manifest,
    build_instance_holdout_manifest,
)


BENCH_ROOT = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "self_evo_continual_20260707"
    / "repos"
    / "SkillLearnBench"
)


def test_local_skilllearnbench_inventory_and_sealed_access() -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.discover()
    summary = adapter.inventory_summary()

    assert len(items) == 100
    assert summary["family_count"] == 20
    assert summary["all_verifier_refs_hashed"] is True
    assert summary["verifier_content_exposed"] is False

    manifest = build_instance_holdout_manifest(
        items,
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-instance-holdout",
    )
    assert manifest.validate() == []
    guard = SplitAccessGuard(manifest)
    train_id = manifest.train_ids[0]
    test_id = manifest.test_ids[0]

    assert adapter.load_instruction(train_id, phase=AccessPhase.PROPOSAL, guard=guard).strip()
    with pytest.raises(PermissionError):
        adapter.load_instruction(test_id, phase=AccessPhase.PROPOSAL, guard=guard)
    with pytest.raises(PermissionError):
        adapter.load_instruction(test_id, phase=AccessPhase.FINAL_REPORT, guard=guard)

    guard.freeze_archive()
    assert adapter.load_instruction(test_id, phase=AccessPhase.FINAL_REPORT, guard=guard).strip()
    assert guard.test_accessed is True


def test_skilllearnbench_family_out_manifest_has_disjoint_families() -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    manifest = build_family_out_manifest(
        adapter.discover(),
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-family-out",
    )

    train_families = {manifest.family_by_id[item_id] for item_id in manifest.train_ids}
    validation_families = {manifest.family_by_id[item_id] for item_id in manifest.validation_ids}
    test_families = {manifest.family_by_id[item_id] for item_id in manifest.test_ids}

    assert manifest.validate() == []
    assert train_families.isdisjoint(validation_families)
    assert train_families.isdisjoint(test_families)
    assert validation_families.isdisjoint(test_families)


def test_credential_independent_subset_excludes_complete_required_env_family() -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.credential_independent_items()
    summary = adapter.credential_independent_summary()

    assert len(items) == 95
    assert {item.family for item in items} == {
        item.family for item in adapter.discover()
    } - {"github-repo-analytics"}
    assert summary == {
        "policy": "exclude_external_credentials_by_family_v1",
        "eligible_instance_count": 95,
        "excluded_instance_count": 5,
        "excluded_families": ["github-repo-analytics"],
        "excluded_required_env_names": ["GH_TOKEN"],
        "secret_value_persisted": False,
    }


def test_promoted_task_hypothesis_compiles_to_skilllearn_skill(tmp_path: Path) -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.discover()
    manifest = build_instance_holdout_manifest(
        items,
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-instance-holdout",
    )
    program = HypothesisProgram.from_dict(
        {
            "id": "hyp-file-organization-workflow",
            "kind": "task",
            "statement": "Classify every source file before moving it into a stable output taxonomy.",
            "trigger": {
                "all_of": [
                    {"key": "family", "op": "eq", "value": "organize-messy-files"},
                ]
            },
            "anti_trigger": {"any_of": [{"key": "read_only", "op": "eq", "value": True}]},
            "action_graph": [
                {
                    "id": "inventory",
                    "operation": "execute_step",
                    "target": "inventory_sources",
                    "value": "Inventory every input file and extract enough content to classify it.",
                },
                {
                    "id": "classify",
                    "operation": "execute_step",
                    "target": "classify_sources",
                    "value": "Assign one deterministic destination category to each file.",
                    "depends_on": ["inventory"],
                },
                {
                    "id": "audit",
                    "operation": "check_condition",
                    "target": "all_sources_accounted_for",
                    "value": "The output count and source count must match before completion.",
                    "depends_on": ["classify"],
                },
            ],
            "expected_effect": {
                "metric": "task_success",
                "minimum_delta": 0.05,
                "maximum_harm_rate": 0.05,
                "maximum_cost_ratio": 1.5,
            },
            "verifier": {
                "checks": ["all_sources_accounted_for", "output_taxonomy_is_stable"],
                "required_evidence": ["source_inventory", "destination_inventory"],
                "anchor_id": "skilllearn_external_task_verifier",
                "repair_on_failure": True,
                "max_repair_depth": 2,
            },
            "evaluator_epoch": "epoch-0",
            "fallback": "preserve_baseline",
            "status": "promoted",
        }
    )

    result = SkillLearnProgramCompiler().compile(
        programs=(program,),
        items=items,
        split_manifest=manifest,
        output_root=tmp_path,
    )

    matching_ids = tuple(
        item_id
        for item_id in manifest.train_ids
        if manifest.family_by_id[item_id] == "organize-messy-files"
    )
    assert result.family_count == 1
    assert len(result.skill_paths) == len(matching_ids)
    assert all(result.source_for(item_id) is not None for item_id in matching_ids)
    skill_text = result.skill_paths[0].read_text(encoding="utf-8")
    assert "items" in result.skill_paths[0].parts
    assert "execute_step `inventory_sources`" in skill_text
    assert "all_sources_accounted_for" in skill_text
    assert "Preserve the baseline workflow" in skill_text

    routed_payload = program.to_dict()
    routed_payload["id"] = "hyp-item-routed-workflow"
    routed_payload["trigger"] = {
        "all_of": [
            {"key": "family", "op": "eq", "value": "organize-messy-files"},
            {"key": "route_variant", "op": "eq", "value": "selected"},
        ],
        "any_of": [],
        "none_of": [],
    }
    routed_program = HypothesisProgram.from_dict(routed_payload)
    routed_items = tuple(
        replace(
            item,
            features={
                **dict(item.features),
                "route_variant": (
                    "selected" if item.id == matching_ids[0] else "not-selected"
                ),
            },
        )
        for item in items
    )
    routed = SkillLearnProgramCompiler().compile(
        programs=(routed_program,),
        items=routed_items,
        split_manifest=manifest,
        output_root=tmp_path,
        method_name="item-routed",
    )

    assert routed.source_for(matching_ids[0]) is not None
    assert all(routed.source_for(item_id) is None for item_id in matching_ids[1:])

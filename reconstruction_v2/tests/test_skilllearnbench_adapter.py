from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks import SkillLearnBenchAdapter, SkillLearnProgramCompiler
from assumption_agent.benchmarks.skilllearn_compiler import (
    SKILL_ACTION_LOWERING_VERSION,
    SKILL_FALLBACK_SEMANTICS_VERSION,
    SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
)
from assumption_agent.models import HypothesisProgram
from assumption_agent.splits import (
    AccessPhase,
    SplitAccessGuard,
    SplitManifest,
    build_family_out_manifest,
    build_instance_holdout_manifest,
)
from assumption_agent.validation import RuntimeActionCheck, ValidationContext


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


def test_offline_ready_subset_is_frozen_without_resplitting() -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    ready = adapter.offline_ready_items()
    summary = adapter.offline_ready_summary()

    assert len(ready) == 86
    assert len({item.family for item in ready}) == 16
    assert summary["policy"] == (
        "exclude_external_credentials_and_offline_blockers_v1"
    )
    assert summary["offline_blocked_item_ids"] == [
        "weighted-gdp-calculation-2"
    ]
    assert summary["offline_blocked_families"] == [
        "fix-security-bug",
        "nlp-paper-reproduction",
        "python-scala-translation",
    ]

    root = Path(__file__).resolve().parents[1] / "manifests"
    old = SplitManifest.read(
        root / "skilllearnbench_instance_holdout_credential_independent_v1.json"
    )
    frozen = SplitManifest.read(
        root / "skilllearnbench_instance_holdout_offline_ready_v1.json"
    )
    ready_ids = {item.id for item in ready}
    assert frozen.train_ids == tuple(
        item_id for item_id in old.train_ids if item_id in ready_ids
    )
    assert frozen.validation_ids == tuple(
        item_id for item_id in old.validation_ids if item_id in ready_ids
    )
    assert frozen.test_ids == tuple(
        item_id for item_id in old.test_ids if item_id in ready_ids
    )


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
                    "id": "evidence",
                    "operation": "request_evidence",
                    "target": "local_file_inventory",
                    "value": "Record the task-local source and destination counts.",
                    "depends_on": ["classify"],
                },
                {
                    "id": "artifact",
                    "operation": "produce_artifact",
                    "target": "organized_output_tree",
                    "value": "Create the requested stable directory taxonomy.",
                    "depends_on": ["evidence"],
                },
                {
                    "id": "audit",
                    "operation": "check_condition",
                    "target": "all_sources_accounted_for",
                    "value": "The output count and source count must match before completion.",
                    "depends_on": ["artifact"],
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
    assert "**Agent instruction:** Inventory every input file" in skill_text
    assert "**Agent-local self-check:**" in skill_text
    assert "benchmark verifier runs after the agent exits" in skill_text
    assert "skilllearn_external_task_verifier" not in skill_text
    assert "source_inventory" not in skill_text
    assert "Minimum held-out delta" not in skill_text
    assert "post-hoc baseline output" in skill_text
    compile_manifest = (
        result.output_root / "compile_manifest.json"
    ).read_text(encoding="utf-8")
    assert SKILL_ACTION_LOWERING_VERSION in compile_manifest
    assert SKILL_FALLBACK_SEMANTICS_VERSION in compile_manifest

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


@pytest.mark.parametrize(
    "operation",
    (
        "enable_lane",
        "disable_lane",
        "prioritize_lane",
        "set_parameter",
        "require_verifier",
        "abstain",
    ),
)
def test_skilllearn_compiler_rejects_actions_without_backend_lowering(
    tmp_path: Path,
    operation: str,
) -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.discover()
    manifest = build_instance_holdout_manifest(
        items,
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-instance-holdout",
    )
    program = HypothesisProgram.from_dict(
        {
            "id": f"unsupported-{operation}",
            "kind": "policy",
            "statement": "This program deliberately exercises an unsupported backend action.",
            "trigger": {
                "all_of": [
                    {"key": "benchmark", "op": "eq", "value": "skilllearnbench"}
                ]
            },
            "anti_trigger": {},
            "action_graph": [
                {
                    "id": "unsupported",
                    "operation": operation,
                    "target": "skilllearn_external_task_verifier",
                    "value": "not lowerable by the SkillLearn prompt backend",
                }
            ],
            "expected_effect": {
                "metric": "task_success",
                "minimum_delta": 0.0,
                "maximum_harm_rate": 0.05,
                "maximum_cost_ratio": 1.5,
            },
            "verifier": {
                "checks": ["paired_validation"],
                "required_evidence": ["policy_off_outcome", "policy_on_outcome"],
                "anchor_id": "skilllearn_external_task_verifier",
                "repair_on_failure": False,
                "max_repair_depth": 0,
            },
            "evaluator_epoch": "epoch-0",
            "fallback": "preserve_baseline",
            "status": "promoted",
        }
    )

    with pytest.raises(ValueError, match="without a backend lowering"):
        SkillLearnProgramCompiler().compile(
            programs=(program,),
            items=items,
            split_manifest=manifest,
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("target", "value"),
    (
        ("skilllearn_external_task_verifier", "Collect local evidence."),
        ("local_evidence", "Read the paired_policy_on_outcome."),
    ),
)
def test_hidden_external_evidence_is_rejected_by_validation_and_compiler(
    tmp_path: Path,
    target: str,
    value: str,
) -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.discover()
    manifest = build_instance_holdout_manifest(
        items,
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-instance-holdout",
    )
    payload = _static_skill_program_payload()
    payload["action_graph"] = [
        {
            "id": "hidden-evidence",
            "operation": "request_evidence",
            "target": target,
            "value": value,
        }
    ]
    program = HypothesisProgram.from_dict(payload)
    context = ValidationContext(
        evaluator_epoch=program.evaluator_epoch,
        residuals=(),
        available_lanes=frozenset(),
        baseline_lane="skilllearn_incumbent",
        allowed_action_operations=SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        external_evidence_is_hidden=True,
    )

    check = RuntimeActionCheck().evaluate(program, context)

    assert check.passed is False
    assert check.evidence["hidden_external_reference_action_ids"] == [
        "hidden-evidence"
    ]
    with pytest.raises(ValueError, match="hidden external evaluation evidence"):
        SkillLearnProgramCompiler().compile(
            programs=(program,),
            items=items,
            split_manifest=manifest,
            output_root=tmp_path,
        )


def test_compile_manifest_binds_rendered_content_and_replaces_stale_tree(
    tmp_path: Path,
) -> None:
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    items = adapter.discover()
    manifest = build_instance_holdout_manifest(
        items,
        benchmark="skilllearnbench",
        seed="skilllearnbench-v2-instance-holdout",
    )
    first = HypothesisProgram.from_dict(_static_skill_program_payload())
    compiler = SkillLearnProgramCompiler()
    first_result = compiler.compile(
        programs=(first,),
        items=items,
        split_manifest=manifest,
        output_root=tmp_path,
        method_name="atomic-content-bound",
    )
    stale = first_result.output_root / "stale-skill" / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    changed_payload = _static_skill_program_payload()
    changed_payload["statement"] += " Rendered statement revision."
    changed = HypothesisProgram.from_dict(changed_payload)

    second_result = compiler.compile(
        programs=(changed,),
        items=items,
        split_manifest=manifest,
        output_root=tmp_path,
        method_name="atomic-content-bound",
    )
    manifest_payload = json.loads(
        (second_result.output_root / "compile_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert first_result.manifest_hash != second_result.manifest_hash
    assert first_result.treatment_hash != second_result.treatment_hash
    assert stale.exists() is False
    assert manifest_payload["skill_content_hashes"]
    assert manifest_payload["program_set_hash"] == second_result.program_set_hash
    assert manifest_payload["treatment_hash"] == second_result.treatment_hash


def _static_skill_program_payload() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "baselines" / "static_generic_program.json").read_text(
            encoding="utf-8"
        )
    )

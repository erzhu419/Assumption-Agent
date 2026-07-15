from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

import pytest

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    ACTIVE_FRESH_ITEM_ID,
    EXPECTED_INACTIVE_PROJECTION_COUNT,
    EXPECTED_PHYSICAL_WORK_UNIT_COUNT,
    EXPECTED_RAW_EXECUTION_COUNT,
    EXPECTED_SEMANTIC_EXECUTION_COUNT,
    FinancialSemanticFreshRunnerError,
    FreshSplitMetadataV1,
    FrozenFinancialTreatmentV1,
    build_fresh_execution_plan_v1,
    execute_fresh_plan_v1,
    load_fresh_split_metadata_v1,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialObservation,
    TrialVariant,
)
from assumption_agent.models import stable_hash


def _frozen_inputs(tmp_path: Path):
    item_ids = tuple(
        sorted(
            (
                ACTIVE_FRESH_ITEM_ID,
                "enterprise-information-search-6",
                "offer-letter-generator-5",
                "organize-messy-files-4",
                "stock-data-visualization-2",
                "temperature-simulation-4",
                "travel-planning-5",
                "video-object-counting-2",
                "weighted-gdp-calculation-1",
            )
        )
    )
    split_hash = "1" * 64
    split = FreshSplitMetadataV1(
        manifest_hash=split_hash,
        item_ids=item_ids,
        family_by_id={
            item_id: item_id.rsplit("-", 1)[0] for item_id in item_ids
        },
    )
    recipe_id = "2" * 64
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: frozen-financial\n---\nFrozen candidate.\n",
        encoding="utf-8",
    )
    from assumption_agent.benchmarks.skilllearn_compiler import (
        verify_skill_source_tree,
    )

    receipt = verify_skill_source_tree(source)
    treatment = FrozenFinancialTreatmentV1(
        manifest_hash="3" * 64,
        recipe_id=recipe_id,
        program_set_hash=stable_hash({"recipe_ids": [recipe_id]}),
        treatment_id="4" * 64,
        candidate_id="5" * 64,
        candidate_manifest_hash="6" * 64,
        external_skill_source_receipt_hash=receipt.receipt_hash,
        candidate_skill_source="candidate",
        fresh_item_id=ACTIVE_FRESH_ITEM_ID,
        fresh_split_manifest_hash=split_hash,
        evaluator_epoch=f"financial-semantic-fresh-{split_hash[:12]}",
        operator_asset_path="operator.json",
        minilm_runtime_asset_path="minilm.json",
        qa_runtime_asset_path="qa.json",
    )
    return split, treatment, source


def _plan(tmp_path: Path):
    split, treatment, source = _frozen_inputs(tmp_path)
    return build_fresh_execution_plan_v1(
        split=split,
        treatment=treatment,
        candidate_skill_source=source,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        codex_agent_execution_policy_hash="7" * 64,
    )


def _observation(request, *, success: bool = True):
    return SkillLearnTrialObservation(
        request=request,
        success=success,
        score=float(success),
        metrics={"evaluation_valid": 1.0},
        total_tokens=17,
        steps=3,
        duration_seconds=0.1,
        provider_fingerprint="provider",
        fairness_fingerprint="fairness",
        error_type=None,
    )


def test_frozen_fresh_split_metadata_loads_without_task_content() -> None:
    project = Path(__file__).resolve().parents[1]
    split = load_fresh_split_metadata_v1(
        project / "manifests/skilllearn_fresh_provenance_split_v1.json"
    )
    assert len(split.item_ids) == 9
    assert ACTIVE_FRESH_ITEM_ID in split.item_ids
    assert split.family_by_id[ACTIVE_FRESH_ITEM_ID] == "financial-analysis"


def test_plan_has_nine_raw_one_semantic_and_no_fake_hipporag(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    assert len(plan.physical_work_units) == EXPECTED_PHYSICAL_WORK_UNIT_COUNT
    assert (
        sum(row.arm == "raw" for row in plan.physical_work_units)
        == EXPECTED_RAW_EXECUTION_COUNT
    )
    assert (
        sum(row.arm == "semantic" for row in plan.physical_work_units)
        == EXPECTED_SEMANTIC_EXECUTION_COUNT
    )
    assert not any("hippo" in row.arm for row in plan.physical_work_units)
    assert plan.safe_payload()["official_hipporag_status"] == (
        "not_applicable_nonexecuted"
    )
    assert plan.safe_payload()["official_hipporag_execution_count"] == 0

    for item_id in plan.split.item_ids:
        raw = plan.raw_requests_by_item[item_id]
        candidate = plan.candidate_requests_by_item[item_id]
        assert raw.variant is TrialVariant.POLICY_OFF
        assert candidate.variant is TrialVariant.POLICY_ON
        assert candidate.program_id == plan.treatment.recipe_id
        assert candidate.treatment_hash == plan.treatment.treatment_id
        assert candidate.external_skill_source_receipt_hash == (
            plan.treatment.external_skill_source_receipt_hash
        )
        assert item_id not in raw.pair_id
        assert item_id not in candidate.to_dict().values()


def test_execution_submits_all_ten_and_projects_only_inactive_routes(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    barrier = threading.Barrier(EXPECTED_PHYSICAL_WORK_UNIT_COUNT)
    created: list[object] = []
    created_lock = threading.Lock()

    class Backend:
        def __init__(self, *, semantic: bool) -> None:
            self.semantic = semantic
            self.financial_runtime_evidence = (
                ({"evidence_hash": "8" * 64},) if semantic else ()
            )

        def run(self, request, *, skill_source_dir, trace_id):
            barrier.wait(timeout=5)
            if self.semantic:
                assert request.item_id == ACTIVE_FRESH_ITEM_ID
                assert skill_source_dir is not None
            else:
                assert skill_source_dir is None
            return _observation(request)

    def factory(work):
        backend = Backend(semantic=work.arm == "semantic")
        with created_lock:
            created.append(backend)
        return backend

    result = execute_fresh_plan_v1(
        plan=plan,
        backend_factory=factory,
        max_workers=EXPECTED_PHYSICAL_WORK_UNIT_COUNT,
    )

    assert len(created) == EXPECTED_PHYSICAL_WORK_UNIT_COUNT
    assert len({id(value) for value in created}) == len(created)
    assert result.maximum_concurrent_calls == EXPECTED_PHYSICAL_WORK_UNIT_COUNT
    assert len(result.physical_results) == EXPECTED_PHYSICAL_WORK_UNIT_COUNT
    assert len(result.inactive_projections) == (
        EXPECTED_INACTIVE_PROJECTION_COUNT
    )
    assert all(row.raw_success == row.projected_success for row in result.inactive_projections)
    assert all(
        row.raw_error_type == row.projected_error_type
        for row in result.inactive_projections
    )
    assert sum(
        len(row.semantic_runtime_evidence)
        for row in result.physical_results
    ) == 1

    raw_by_hash = {
        stable_hash({"item_id": row.work.item_id}): row.observation
        for row in result.physical_results
        if row.work.arm == "raw"
    }
    candidate_by_hash = {
        stable_hash({"item_id": item_id}): request
        for item_id, request in plan.candidate_requests_by_item.items()
        if item_id != ACTIVE_FRESH_ITEM_ID
    }
    for projection in result.inactive_projections:
        expected = raw_by_hash[projection.item_id_hash].as_variant(
            candidate_by_hash[projection.item_id_hash]
        )
        assert projection.projected_observation_hash == (
            expected.observation_hash
        )


def test_runner_rejects_serial_or_reused_backend_execution(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(
        FinancialSemanticFreshRunnerError,
        match="one worker per physical work unit",
    ):
        execute_fresh_plan_v1(
            plan=plan,
            backend_factory=lambda work: object(),
            max_workers=9,
        )

    shared = object()
    with pytest.raises(
        FinancialSemanticFreshRunnerError,
        match="reused an instance",
    ):
        execute_fresh_plan_v1(
            plan=plan,
            backend_factory=lambda work: shared,
            max_workers=10,
        )


def test_plan_fails_closed_if_candidate_identity_changes(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    item_id = next(
        value
        for value in plan.split.item_ids
        if value != ACTIVE_FRESH_ITEM_ID
    )
    changed = dict(plan.candidate_requests_by_item)
    changed[item_id] = replace(
        changed[item_id],
        treatment_hash="9" * 64,
    )
    drifted = replace(plan, candidate_requests_by_item=changed)
    with pytest.raises(
        FinancialSemanticFreshRunnerError,
        match="request identity drifted",
    ):
        drifted.verify()

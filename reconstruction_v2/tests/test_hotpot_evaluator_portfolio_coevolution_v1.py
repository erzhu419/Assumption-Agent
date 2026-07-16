from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    hotpot_evaluator_portfolio_coevolution_v1 as study,
)
from assumption_agent.models import stable_hash


def _digest(label: str) -> str:
    return stable_hash({"synthetic": label})


def _ranking_for(program_ordinal: int, item_ordinal: int) -> tuple[int, ...]:
    family = program_ordinal % len(study.CAPABILITY_FAMILIES)
    patterns = (
        (2, 3, 4, 5, 6),
        (0, 2, 3, 4, 5),
        (1, 2, 3, 4, 5),
        (0, 1, 5, 6, 7),
        (1, 0, 6, 7, 8),
        (3, 4, 5, 6, 7),
    )
    ranking = patterns[family]
    if item_ordinal % 3 == 1:
        ranking = ranking[1:] + ranking[:1]
    return ranking


def _grid(
    *, environment_ids: tuple[str, str] = study.A_FORM_ENVIRONMENTS
) -> tuple[study.FormationGridEvidence, str]:
    items = tuple(
        tuple(
            study.GridItemEvidence(
                item_commitment_sha256=_digest(f"item-{environment}-{ordinal}"),
                p_ranking=(0, 5, 6, 7, 8),
                support_indices=(0, 1),
            )
            for ordinal in range(study.FORMATION_ENV_ITEM_COUNT)
        )
        for environment in environment_ids
    )
    retained_hash = _digest("program-0")
    programs = []
    for ordinal in range(study.CANDIDATE_COUNT):
        family = study.CAPABILITY_FAMILIES[
            ordinal % len(study.CAPABILITY_FAMILIES)
        ]
        if ordinal == 0:
            rankings = tuple(
                tuple((0, 1, 2, 3, 4) for _item in range(24))
                for _environment in range(2)
            )
        else:
            rankings = tuple(
                tuple(_ranking_for(ordinal, item) for item in range(24))
                for _environment in range(2)
            )
        programs.append(
            study.ProgramGridEvidence(
                program_sha256=_digest(f"program-{ordinal}"),
                program_length=1 + ordinal % 4,
                seed_algorithm=family[0],
                expansion_mode=family[1],
                q_rankings=rankings,
            )
        )
    grid = study.FormationGridEvidence(
        environment_ids=environment_ids,
        items=items,
        programs=tuple(programs),
    ).validate(environment_ids)
    return grid, retained_hash


def test_portfolio_core_excludes_even_highest_scoring_retained_p() -> None:
    grid, retained = _grid()
    core = study.form_portfolio_policies_from_evidence(
        grid,
        expected_environment_ids=study.A_FORM_ENVIRONMENTS,
        retained_p_program_sha256=retained,
    )
    assert core["retained_P_program_sha256"] == retained
    assert core["retained_P_excluded_from_portfolios"] is True
    assert retained not in core["incumbent"]["program_sha256s"]
    assert retained not in core["challenger"]["program_sha256s"]
    assert retained not in core["incumbent_shortlist_program_sha256s"]
    assert retained not in core["challenger_shortlist_program_sha256s"]
    assert len(core["incumbent"]["program_sha256s"]) == 2
    assert len(core["challenger"]["program_sha256s"]) == 2
    assert core["identical_action_has_no_runner_up_or_fallback"] is True
    assert core["logical_retrieval_calls_per_compared_arm"] == 3


def test_behavior_deduplication_does_not_use_program_identity_or_gold() -> None:
    grid, _retained = _grid()
    first = grid.programs[3]
    second = replace(
        first,
        program_sha256=_digest("different-program-id"),
        program_length=first.program_length + 9,
    )
    assert study._program_behavior_sha256(grid, first) == study._program_behavior_sha256(
        grid, second
    )
    changed_gold = study.FormationGridEvidence(
        environment_ids=grid.environment_ids,
        items=tuple(
            tuple(replace(item, support_indices=(2, 3)) for item in environment)
            for environment in grid.items
        ),
        programs=grid.programs,
    )
    assert study._program_behavior_sha256(
        changed_gold, first
    ) == study._program_behavior_sha256(grid, first)


def test_portfolio_assessment_has_exact_eight_cell_challenger_grid() -> None:
    grid, retained = _grid()
    assessment, _shortlist = study.select_portfolio(
        grid,
        policy_id=study.CHALLENGER_POLICY_ID,
        retained_p_program_sha256=retained,
    )
    assert len(assessment.cell_net) == 8
    assert len(assessment.cell_gain) == 8
    assert len(assessment.cell_harm) == 8
    assert len(assessment.cell_combined_hits) == 8
    assert assessment.program_sha256s[0] != assessment.program_sha256s[1]
    assert assessment.families[0] != assessment.families[1]
    assert study.challenger_key(assessment)[0] == assessment.invalid_count


def test_exact_magnitude_sign_flip_and_alpha_are_frozen() -> None:
    promoted = study.exact_paired_sign_flip([1, 1, 1, 1])
    rejected = study.exact_paired_sign_flip([1, 1, 1])
    assert promoted["p_value_numerator"] == 1
    assert promoted["p_value_denominator"] == 16
    assert promoted["promoted"] is True
    assert rejected["p_value_numerator"] == 1
    assert rejected["p_value_denominator"] == 8
    assert rejected["promoted"] is False
    assert promoted["alpha_numerator"] == 1
    assert promoted["alpha_denominator"] == 10
    with pytest.raises(study.HotpotEvaluatorPortfolioError):
        study.exact_paired_sign_flip([])


def test_archive_transition_invalidates_only_dependent_score() -> None:
    promoted = study._archive_transition(
        anchor_manifest_sha256=_digest("anchor"),
        incumbent_hits=50,
        challenger_hits=60,
        support_total=96,
        item_count=48,
        promoted=True,
    )
    assert promoted["promoted"] is True
    assert promoted["selective_invalidation_performed"] is True
    assert promoted["dependent_score_valid_after_transition"] is False
    assert promoted["independent_source_score_valid_after_transition"] is True
    assert promoted["invalidated_score_record_ids"] == [
        promoted["dependent_score_record_id"]
    ]
    rejected = study._archive_transition(
        anchor_manifest_sha256=_digest("anchor-2"),
        incumbent_hits=50,
        challenger_hits=49,
        support_total=96,
        item_count=48,
        promoted=False,
    )
    assert rejected["selective_invalidation_performed"] is False
    assert rejected["invalidated_score_record_ids"] == []
    assert rejected["dependent_score_valid_after_transition"] is True


def test_design_work_grids_and_equal_compute_contract_are_exact() -> None:
    assert study.FORMATION_ENV_WORK_UNIT_COUNT == 2040
    assert study.FORMATION_WORK_UNIT_COUNT == 4080
    assert study.FORMATION_MAXIMUM_CONCURRENCY == 2040
    assert study.ANCHOR_COMPONENT_IDS == (
        "incumbent_P",
        "incumbent_Q1",
        "incumbent_Q2",
        "challenger_P",
        "challenger_Q1",
        "challenger_Q2",
    )
    assert study.ANCHOR_WORK_UNIT_COUNT == 288
    assert study.SEARCH_WORK_UNIT_COUNT == 192
    anchor = study._anchor_execution_contract()
    search = study._search_execution_contract()
    assert anchor["logical_retrieval_calls_per_compared_arm_item"] == 3
    assert anchor["sole_promotion_criterion"] is True
    assert search["logical_retrieval_calls_per_primary_arm_item"] == 3
    assert search["physical_component_ids"] == list(study.SEARCH_COMPONENT_IDS)
    assert "retained_P" in search["derived_arms"]
    assert "active_portfolio_minus_retained_P" in search["secondary_comparisons"]


def test_formal_freezes_have_no_measurement_path_or_injection_surface() -> None:
    assert study.formal_signatures_have_no_injection_surface() is True
    anchor_parameters = inspect.signature(
        study.build_a_hold_pre_run_freeze
    ).parameters
    search_parameters = inspect.signature(
        study.build_m_search_pre_run_freeze
    ).parameters
    assert "a_hold_block_path" not in anchor_parameters
    assert "m_search_block_path" not in search_parameters
    with pytest.raises(study.HotpotEvaluatorPortfolioError):
        study.execute_a_hold_formal(
            project_root="unused",
            pre_run_freeze_path="unused",
            acquisition_receipt_path="unused",
            a_hold_block_path="unused",
            p_formation_receipt_path="unused",
            p_frozen_program_path="unused",
            m1_freeze_path="unused",
            m1_report_path="unused",
            old_final_disposition_path="unused",
            a_form_private_cache_path="unused",
            a_form_public_receipt_path="unused",
            f_search_private_cache_path="unused",
            f_search_public_receipt_path="unused",
            execution_root="unused",
        )


def test_public_safety_rejects_private_content_and_paths() -> None:
    with pytest.raises(study.HotpotEvaluatorPortfolioError):
        study._assert_public_safe({"question": "secret"})
    with pytest.raises(study.HotpotEvaluatorPortfolioError):
        study._assert_public_safe({"path": "/tmp/private"})
    study._assert_public_safe(
        {
            "program_sha256s": [_digest("a"), _digest("b")],
            "raw_content_persisted": False,
        }
    )


def _items(count: int, *, prefix: str) -> tuple[study.l4.RecursiveItem, ...]:
    corpus = tuple(
        study.l4.RetrievalParagraph(index, f"title-{index}", f"text-{index}")
        for index in range(10)
    )
    return tuple(
        study.l4.RecursiveItem(
            item_id=f"{prefix}-{ordinal}",
            question=f"question {ordinal}",
            corpus=corpus,
            support_indices=(0, 1),
            row_commitment_sha256=_digest(f"{prefix}-row-{ordinal}"),
        )
        for ordinal in range(count)
    )


def test_formation_marker_is_one_shot_and_binds_both_outputs(tmp_path: Path) -> None:
    acquisition_hash = _digest("acquisition-file")
    private_output = tmp_path / "private.json"
    public_output = tmp_path / "public.json"
    binding = study._write_formation_marker(
        project=tmp_path,
        stage="A_form",
        acquisition_file_sha256=acquisition_hash,
        output_cache_path=private_output,
        output_receipt_path=public_output,
    )
    assert binding["marker_written_before_both_private_environment_blocks_open"]
    assert binding["private_block_rows_opened_before_marker"] == 0
    assert study._load_formation_marker(
        project=tmp_path,
        stage="A_form",
        acquisition_file_sha256=acquisition_hash,
        private_cache_path=private_output,
        public_receipt_path=public_output,
    ) == binding
    with pytest.raises(
        study.HotpotEvaluatorPortfolioError, match="already consumed"
    ):
        study._write_formation_marker(
            project=tmp_path,
            stage="A_form",
            acquisition_file_sha256=acquisition_hash,
            output_cache_path=private_output,
            output_receipt_path=public_output,
        )


def test_two_environment_grid_joins_both_barriers_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programs = study.fixed_programs()[:2]
    monkeypatch.setattr(study, "CANDIDATE_COUNT", 2)
    monkeypatch.setattr(study, "FORMATION_ENV_ITEM_COUNT", 2)
    monkeypatch.setattr(study, "FORMATION_ITEM_COUNT", 4)
    monkeypatch.setattr(study, "FORMATION_ENV_WORK_UNIT_COUNT", 6)
    monkeypatch.setattr(study, "FORMATION_WORK_UNIT_COUNT", 12)
    monkeypatch.setattr(study, "FORMATION_MAXIMUM_CONCURRENCY", 6)
    monkeypatch.setattr(study, "fixed_programs", lambda: programs)
    calls: list[str] = []

    def ranking(program: object, item: object) -> tuple[int, ...]:
        calls.append(getattr(program, "program_hash", "P"))
        return (0, 2, 3, 4, 5)

    monkeypatch.setattr(study.l4, "_ranking", ranking)
    grid, execution = study._evaluate_formation_grid(
        p_program=programs[0],
        environment_ids=study.A_FORM_ENVIRONMENTS,
        environments=(_items(2, prefix="env0"), _items(2, prefix="env1")),
    )
    assert len(calls) == 12
    assert grid.environment_ids == study.A_FORM_ENVIRONMENTS
    assert execution["physical_work_unit_count"] == 12
    assert execution["retrieval_attempt_count"] == 12
    assert execution["retrieval_terminal_count"] == 12
    assert execution["environment_barrier_count"] == 2
    assert execution["environment_barrier_party_count"] == 6
    assert execution["all_terminals_joined_before_support_scoring"] is True


def test_same_action_formation_prevents_a_hold_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p_program = SimpleNamespace(program_hash=_digest("P"))
    a_public = {"formation_core": {"measurable_contrast": False}}
    f_public = {"formation_core": {"measurable_contrast": True}}
    monkeypatch.setattr(
        study,
        "_artifact_bundles",
        lambda **_kwargs: (
            {}, b"{}", {}, p_program, {}, a_public, f_public, {}, {}
        ),
    )
    with pytest.raises(
        study.HotpotEvaluatorPortfolioError,
        match="A_hold must remain unopened",
    ):
        study.build_a_hold_pre_run_freeze(
            project_root=tmp_path,
            acquisition_receipt_path="unused",
            p_formation_receipt_path="unused",
            p_frozen_program_path="unused",
            m1_freeze_path="unused",
            m1_report_path="unused",
            old_final_disposition_path="unused",
            a_form_private_cache_path="unused",
            a_form_public_receipt_path="unused",
            f_search_private_cache_path="unused",
            f_search_public_receipt_path="unused",
            execution_root=tmp_path / "never-opened",
            authorization_hash=_digest("auth"),
            output_path=tmp_path / "freeze.json",
        )
    assert not (tmp_path / "freeze.json").exists()


def test_unpromoted_anchor_refuses_m_search_before_runtime_or_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        study,
        "load_and_reverify_a_hold",
        lambda **_kwargs: (
            {
                "evaluator_epoch_transition": {
                    "promoted": False,
                    "selective_invalidation_performed": False,
                    "independent_source_record_retained": True,
                }
            },
            {"anchor": _digest("anchor")},
        ),
    )
    monkeypatch.setattr(
        study,
        "_artifact_bundles",
        lambda **_kwargs: calls.append("source-opened"),
    )
    with pytest.raises(
        study.HotpotEvaluatorPortfolioError,
        match="M_search must remain unopened",
    ):
        study.build_m_search_pre_run_freeze(
            project_root=tmp_path,
            acquisition_receipt_path="unused",
            p_formation_receipt_path="unused",
            p_frozen_program_path="unused",
            m1_freeze_path="unused",
            m1_report_path="unused",
            old_final_disposition_path="unused",
            a_form_private_cache_path="unused",
            a_form_public_receipt_path="unused",
            f_search_private_cache_path="unused",
            f_search_public_receipt_path="unused",
            a_hold_pre_run_freeze_path="unused",
            a_hold_private_evidence_path="unused",
            a_hold_report_path="unused",
            capability_receipt_path="unused",
            runtime_python="unused",
            local_llm_model="unused",
            local_embedding_model="unused",
            base_binding_receipt_path="unused",
            attestation_receipt_path="unused",
            execution_root=tmp_path / "never-opened",
            authorization_hash=_digest("auth"),
            output_path=tmp_path / "m-freeze.json",
        )
    assert calls == []
    assert not (tmp_path / "m-freeze.json").exists()


def test_live_acquisition_design_mismatch_is_rejected_and_exact_binding_inherits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {
        "A_form_0": 24,
        "A_form_1": 24,
        "F_search_0": 24,
        "F_search_1": 24,
        "A_hold": 48,
        "M_search": 24,
    }
    implementation = {"set_sha256": _digest("implementation")}
    lineage = {"set_sha256": _digest("lineage")}
    live_design = {
        "schema": "hotpot_evaluator_portfolio_design_v1",
        "design_sha256": _digest("design"),
    }
    receipt = {
        "acquisition_sha256": _digest("acquisition"),
        "implementation": implementation,
        "retained_P_lineage": lineage,
        "portfolio_design_binding": {**live_design, "design_sha256": _digest("tamper")},
    }
    rows = tuple(
        SimpleNamespace(
            block=block,
            count=count,
            file_sha256=_digest(f"file-{block}"),
            item_commitment_set_sha256=_digest(f"items-{block}"),
        )
        for block, count in counts.items()
    )
    fake = SimpleNamespace(
        BLOCK_COUNTS=counts,
        BLOCK_ORDER=tuple(counts),
        load_acquisition_binding=lambda _path: (receipt, rows),
        implementation_binding=lambda _project: implementation,
        retained_p_lineage_binding=lambda _project: lineage,
        portfolio_design_binding=lambda _project: live_design,
    )
    monkeypatch.setattr(study, "_acquisition_module", lambda: fake)
    receipt_path = tmp_path / "acquisition.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(study.HotpotEvaluatorPortfolioError, match="design drifted"):
        study._load_acquisition_live(project=tmp_path, path=receipt_path)

    receipt["portfolio_design_binding"] = live_design
    loaded, raw, commitments = study._load_acquisition_live(
        project=tmp_path, path=receipt_path
    )
    source = study._formation_source_binding(
        receipt=loaded,
        receipt_raw=raw,
        commitments=commitments,
        environment_ids=study.A_FORM_ENVIRONMENTS,
    )
    assert source["portfolio_design_binding"] == live_design


def test_distinct_hash_with_exact_retained_p_behavior_is_excluded() -> None:
    grid, retained = _grid()
    alias = replace(
        grid.programs[1],
        program_sha256=_digest("distinct-P-behavior-alias"),
        q_rankings=tuple(
            tuple(item.p_ranking for item in environment)
            for environment in grid.items
        ),
    )
    aliased_grid = study.FormationGridEvidence(
        environment_ids=grid.environment_ids,
        items=grid.items,
        programs=(grid.programs[0], alias, *grid.programs[2:]),
    ).validate(grid.environment_ids)
    canonical = study.canonical_behavior_programs(
        aliased_grid, retained_p_program_sha256=retained
    )
    assert retained not in {program.program_sha256 for program in canonical}
    assert alias.program_sha256 not in {
        program.program_sha256 for program in canonical
    }
    core = study.form_portfolio_policies_from_evidence(
        aliased_grid,
        expected_environment_ids=study.A_FORM_ENVIRONMENTS,
        retained_p_program_sha256=retained,
    )
    selected = {
        *core["incumbent"]["program_sha256s"],
        *core["challenger"]["program_sha256s"],
        *core["incumbent_shortlist_program_sha256s"],
        *core["challenger_shortlist_program_sha256s"],
    }
    assert alias.program_sha256 not in selected
    assert core["retained_P_behavior_sha256"] == study._retained_p_behavior_sha256(
        aliased_grid
    )
    assert core["retained_P_hash_and_behavior_class_excluded"] is True


def test_rehashed_search_freeze_transition_tamper_is_rejected() -> None:
    transition = study._archive_transition(
        anchor_manifest_sha256=_digest("transition-anchor"),
        incumbent_hits=50,
        challenger_hits=60,
        support_total=96,
        item_count=48,
        promoted=True,
    )
    exact = study._promoted_transition_binding(transition)
    study._assert_search_transition_binding(exact, transition)
    tampered = {**exact, "active_evaluator_id": "tampered_evaluator"}
    body = {
        "schema": study.SEARCH_FREEZE_SCHEMA,
        "evaluator_epoch_transition": tampered,
    }
    rehashed = {**body, "freeze_sha256": stable_hash(body)}
    assert rehashed["freeze_sha256"] == stable_hash(body)
    with pytest.raises(
        study.HotpotEvaluatorPortfolioError,
        match="evaluator transition drifted",
    ):
        study._assert_search_transition_binding(
            rehashed["evaluator_epoch_transition"], transition
        )

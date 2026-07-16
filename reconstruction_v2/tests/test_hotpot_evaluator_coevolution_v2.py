from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks import hotpot_evaluator_coevolution_v2 as study
from assumption_agent.models import stable_hash
from assumption_agent.benchmarks.musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
)


def _program(
    name: str,
    *,
    q_hits: int,
    combined_hits: int,
    item_count: int = 8,
) -> study.ProgramEvidence:
    return study.ProgramEvidence(
        program_sha256=stable_hash({"program": name}),
        program_length=4,
        rows=tuple(
            study.ItemProgramEvidence(
                item_commitment_sha256=stable_hash({"item": ordinal}),
                q_direct_hits=q_hits,
                combined_hits=combined_hits,
                support_total=2,
                combined_complete=int(combined_hits == 2),
                combined_coverage=int(combined_hits > 0),
                retained_added=int(combined_hits > 0),
                novelty_added=int(combined_hits == 2),
                retained_displaced=0,
                invalid=False,
                action_sha256=stable_hash(
                    {
                        "item": ordinal,
                        "invalid": False,
                        "q_hits": q_hits,
                        "combined_hits": combined_hits,
                    }
                ),
            )
            for ordinal in range(item_count)
        ),
    ).validate()


def _formation_evidence() -> tuple[study.ProgramEvidence, ...]:
    # The incumbent Q-direct rule selects direct-Q.  All generic combined
    # rules select combo-Q, so the challenger has a genuinely different action.
    return (
        _program("retained-P", q_hits=0, combined_hits=0),
        _program("direct-Q", q_hits=2, combined_hits=1),
        _program("combo-Q", q_hits=0, combined_hits=2),
    )


def test_fixed_policy_formation_is_action_distinct_and_search_is_prospective() -> None:
    evidence = _formation_evidence()
    p_hash = evidence[0].program_sha256
    receipt = study.form_challenger_from_evidence(
        evidence, retained_p_program_sha256=p_hash
    )
    assert receipt["partition"] == "A_form"
    assert receipt["incumbent_policy"]["id"] == study.INCUMBENT_POLICY_ID
    assert receipt["challenger_policy"]["id"] != study.INCUMBENT_POLICY_ID
    assert (
        receipt["incumbent_selected_program_sha256"]
        != receipt["challenger_selected_program_sha256"]
    )
    assert (
        receipt["incumbent_selected_action_sha256"]
        != receipt["challenger_selected_action_sha256"]
    )
    assert receipt["action_identical_policy_excluded_from_challenger"] is True
    assert all(
        row["held_support_total"] == 16 for row in receipt["crossfit"]
    )
    frozen = study.freeze_search_choices_from_evidence(
        search_evidence=evidence,
        a_form_evidence=evidence,
        formation_receipt=receipt,
        retained_p_program_sha256=p_hash,
    )
    assert frozen["measurable_contrast"] is True
    assert frozen["anchor_accessed"] is False
    assert frozen["search_measurement_accessed"] is False


def test_identical_f_search_actions_close_before_anchor_without_fallback() -> None:
    formation = _formation_evidence()
    p_hash = formation[0].program_sha256
    receipt = study.form_challenger_from_evidence(
        formation, retained_p_program_sha256=p_hash
    )
    # On F_search both objectives choose the same program.  This is not a
    # performance failure; it makes the prospective causal contrast undefined.
    search = (
        _program("retained-P", q_hits=0, combined_hits=0),
        _program("direct-Q", q_hits=2, combined_hits=2),
        _program("combo-Q", q_hits=0, combined_hits=0),
    )
    frozen = study.freeze_search_choices_from_evidence(
        search_evidence=search,
        a_form_evidence=formation,
        formation_receipt=receipt,
        retained_p_program_sha256=p_hash,
    )
    assert frozen["measurable_contrast"] is False
    assert frozen["identical_action_has_no_fallback"] is True


def test_distinct_program_ids_with_identical_behavior_are_not_a_challenger() -> None:
    p = _program("retained-P", q_hits=0, combined_hits=0)
    first = _program("first-Q", q_hits=1, combined_hits=1)
    second = _program("second-Q", q_hits=1, combined_hits=1)
    assert first.program_sha256 != second.program_sha256
    assert study._program_action_sha256(first) == study._program_action_sha256(second)
    with pytest.raises(
        study.HotpotEvaluatorCoevolutionError,
        match="no action-distinct evaluator challenger",
    ):
        study.form_challenger_from_evidence(
            (p, first, second), retained_p_program_sha256=p.program_sha256
        )


def test_f_search_rejects_distinct_programs_with_identical_observed_actions() -> None:
    formation = _formation_evidence()
    p_hash = formation[0].program_sha256
    receipt = study.form_challenger_from_evidence(
        formation, retained_p_program_sha256=p_hash
    )
    direct = _program("direct-Q", q_hits=2, combined_hits=1)
    combo = _program("combo-Q", q_hits=0, combined_hits=2)
    combo = replace(
        combo,
        rows=tuple(
            replace(row, action_sha256=direct.rows[index].action_sha256)
            for index, row in enumerate(combo.rows)
        ),
    )
    assert direct.program_sha256 != combo.program_sha256
    assert study._program_action_sha256(direct) == study._program_action_sha256(combo)
    frozen = study.freeze_search_choices_from_evidence(
        search_evidence=(formation[0], direct, combo),
        a_form_evidence=formation,
        formation_receipt=receipt,
        retained_p_program_sha256=p_hash,
    )
    assert (
        frozen["incumbent_selected_program_sha256"]
        != frozen["challenger_selected_program_sha256"]
    )
    assert (
        frozen["incumbent_selected_action_sha256"]
        == frozen["challenger_selected_action_sha256"]
    )
    assert frozen["measurable_contrast"] is False
    assert frozen["behavior_distinct_required"] is True


def test_exact_paired_transition_promotes_and_selectively_invalidates() -> None:
    positive = study.exact_paired_sign_flip([1, 1, 1, 1, 0, 0, 0, 0])
    assert positive["p_value_numerator"] == 1
    assert positive["p_value_denominator"] == 16
    assert positive["promoted"] is True
    negative = study.exact_paired_sign_flip([1, 1, 1, 0, 0, 0, 0, 0])
    assert negative["p_value_numerator"] == 1
    assert negative["p_value_denominator"] == 8
    assert negative["promoted"] is False

    incumbent, challenger = study.evaluator_policies()[:2]
    transition = study._archive_transition(
        incumbent_policy=incumbent,
        challenger_policy=challenger,
        anchor_manifest_sha256=stable_hash({"anchor": "fresh"}),
        incumbent_hits=20,
        challenger_hits=24,
        support_total=48,
        item_count=24,
        promoted=positive["promoted"],
    )
    assert transition["promoted"] is True
    assert transition["selective_invalidation_performed"] is True
    assert transition["dependent_score_valid_after_transition"] is False
    assert transition["independent_source_score_valid_after_transition"] is True
    assert transition["independent_source_record_retained"] is True
    assert transition["invalidated_score_record_ids"] == [
        transition["dependent_score_record_id"]
    ]


def test_formal_surface_is_fixed_width_clean_cli_and_freezes_cannot_open_rows() -> None:
    assert study.formal_signatures_have_no_injection_surface() is True
    assert study.CANDIDATE_COUNT == 84
    assert study.ANCHOR_WORK_UNIT_COUNT == 72
    assert study.ANCHOR_MAXIMUM_CONCURRENCY == 72
    assert study.SEARCH_WORK_UNIT_COUNT == 96
    assert study.SEARCH_MAXIMUM_CONCURRENCY == 96
    assert study.SEARCH_ARM_IDS == (
        "canonical_RAW",
        "incumbent_combined",
        "active_combined",
        "official_HippoRAG",
    )
    assert "a_hold_block_path" not in inspect.signature(
        study.build_a_hold_pre_run_freeze
    ).parameters
    assert "m_search_block_path" not in inspect.signature(
        study.build_m_search_pre_run_freeze
    ).parameters
    assert study._anchor_execution_contract()["sole_promotion_criterion"] is True
    assert study._search_execution_contract()["retries"] == 0
    assert study._search_execution_contract()["replays"] == 0
    assert study._search_execution_contract()["resamples"] == 0


def test_combined_evidence_is_exact_P_plus_Q_without_RAW(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programs = study.fixed_programs()[:2]
    p_program, q_program = programs
    p_ranking = (0, 1, 2, 3, 4)
    q_ranking = (0, 1, 5, 2, 4)
    corpus = tuple(
        RetrievalParagraph(index, f"title-{index}", f"body-{index}")
        for index in range(6)
    )
    items = tuple(
        study.l4.RecursiveItem(
            item_id=f"private-{ordinal}",
            question="synthetic",
            corpus=corpus,
            support_indices=(0, 5),
            row_commitment_sha256=stable_hash({"row": ordinal}),
        )
        for ordinal in range(4)
    )

    monkeypatch.setattr(study, "fixed_programs", lambda: programs)

    def ranking(program, _item):
        return p_ranking if program.program_hash == p_program.program_hash else q_ranking

    monkeypatch.setattr(study.l4, "_ranking", ranking)
    evidence, execution = study._evaluate_program_grid(
        p_program=p_program, items=items
    )
    q_evidence = next(
        row for row in evidence if row.program_sha256 == q_program.program_hash
    )
    # P+Q RRF includes support document 5.  Adding canonical RAW would displace
    # it, so two hits here is an exact regression for the prospectively changed
    # P+Q semantics.
    assert [row.combined_hits for row in q_evidence.rows] == [2, 2, 2, 2]
    assert [row.novelty_added for row in q_evidence.rows] == [1, 1, 1, 1]
    assert execution["candidate_retrieval_terminal_count"] == 8


def test_fixed_formation_marker_is_exclusive_before_private_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / "fixed.marker.json"
    monkeypatch.setattr(
        study, "_formation_marker_path", lambda _project, _stage: marker_path
    )
    source = {"acquisition_sha256": stable_hash({"source": 1})}
    lineage = {"lineage_sha256": stable_hash({"lineage": 1})}
    implementation = {"set_sha256": stable_hash({"implementation": 1})}
    marker, raw = study._consume_formation_once(
        project=tmp_path,
        stage="A_form",
        source_binding=source,
        lineage_binding=lineage,
        implementation=implementation,
        private_cache_output_path=tmp_path / "private.json",
        public_receipt_output_path=tmp_path / "public.json",
    )
    binding = study._formation_consumption_binding(marker, raw)
    assert binding["private_block_rows_opened_before_marker"] == 0
    assert binding["retry_replay_resample_authorized"] is False
    with pytest.raises(FileExistsError):
        study._consume_formation_once(
            project=tmp_path,
            stage="A_form",
            source_binding=source,
            lineage_binding=lineage,
            implementation=implementation,
            private_cache_output_path=tmp_path / "private-2.json",
            public_receipt_output_path=tmp_path / "public-2.json",
        )


def test_formal_cache_rejects_reordered_candidate_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programs = study.fixed_programs()[:2]
    monkeypatch.setattr(study, "fixed_programs", lambda: programs)
    monkeypatch.setattr(study, "CANDIDATE_COUNT", 2)
    first = replace(
        _program("first", q_hits=1, combined_hits=1, item_count=24),
        program_sha256=programs[0].program_hash,
        program_length=programs[0].program_length,
    )
    second = replace(
        _program("second", q_hits=0, combined_hits=2, item_count=24),
        program_sha256=programs[1].program_hash,
        program_length=programs[1].program_length,
    )
    source = {
        "acquisition_file_sha256": stable_hash({"acquisition_file": 1}),
        "acquisition_sha256": stable_hash({"acquisition": 1}),
        "block_file_sha256": stable_hash({"block_file": 1}),
        "block_id_sha256": stable_hash({"block": "A_form"}),
        "item_commitment_set_sha256": stable_hash(
            [row.item_commitment_sha256 for row in first.rows]
        ),
        "item_count": 24,
    }
    execution = {
        "candidate_program_count": 2,
        "item_count": 24,
        "shared_retained_P_retrieval_call_count": 24,
        "candidate_retrieval_work_unit_count": 48,
        "candidate_retrieval_attempt_count": 48,
        "candidate_retrieval_terminal_count": 48,
        "configured_candidate_maximum_concurrency": 48,
        "all_candidate_terminals_joined_before_support_scoring": True,
        "invalid_terminal_count": 0,
        "model_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }
    cache = study._cache_payload(
        stage="A_form",
        source_binding=source,
        lineage_binding={"retained_P": {"program_hash": first.program_sha256}},
        evidence=(second, first),
        execution=execution,
        implementation={"set_sha256": stable_hash({"implementation": 1})},
        formation_consumption_binding={
            "marker_sha256": stable_hash({"marker": 1})
        },
    )
    path = tmp_path / "reordered.cache.json"
    study._write_json_exclusive(path, cache)
    with pytest.raises(
        study.HotpotEvaluatorCoevolutionError,
        match="private evidence cache drifted",
    ):
        study._load_cache(path, expected_stage="A_form")

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    musique_evaluator_portfolio_coevolution_v1 as study,
)
from assumption_agent.models import stable_hash


def _digest(label: str) -> str:
    return stable_hash({"synthetic": label})


def _grid(
    *, environment_ids: tuple[str, str] = study.A_FORM_ENVIRONMENTS
) -> tuple[study.FormationGridEvidence, str]:
    items = tuple(
        tuple(
            study.GridItemEvidence(
                item_commitment_sha256=_digest(f"item-{environment}-{ordinal}"),
                p_ranking=(0, 6, 7, 8, 9),
                # Deliberately variable: no Hotpot exact-two-support assumption.
                support_indices=tuple(range(2 + ordinal % 3)),
            )
            for ordinal in range(study.FORMATION_ENV_ITEM_COUNT)
        )
        for environment in environment_ids
    )
    patterns = (
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (0, 2, 4, 6, 8),
        (2, 3, 4, 5, 6),
        (0, 3, 5, 7, 9),
        (1, 0, 6, 7, 8),
    )
    programs = []
    retained = _digest("retained-P")
    for ordinal in range(study.CANDIDATE_COUNT):
        family = study.CAPABILITY_FAMILIES[ordinal % len(study.CAPABILITY_FAMILIES)]
        ranking = patterns[ordinal % len(patterns)]
        programs.append(
            study.ProgramGridEvidence(
                program_sha256=_digest(f"program-{ordinal}"),
                program_length=1 + ordinal % 4,
                seed_algorithm=family[0],
                expansion_mode=family[1],
                q_rankings=tuple(
                    tuple(ranking for _ in range(study.FORMATION_ENV_ITEM_COUNT))
                    for _ in range(study.FORMATION_ENV_COUNT)
                ),
            )
        )
    grid = study.FormationGridEvidence(
        environment_ids=environment_ids,
        items=items,
        programs=tuple(programs),
    ).validate(environment_ids)
    return grid, retained


def test_fixed_objective_accepts_variable_musique_support_counts() -> None:
    grid, retained = _grid()
    core = study.form_portfolio_policies_from_evidence(
        grid,
        expected_environment_ids=study.A_FORM_ENVIRONMENTS,
        retained_p_program_sha256=retained,
    )
    assert core["source_family"] == "MuSiQue_official_DEV_residual"
    assert core["variable_support_count_supported"] is True
    assert len(core["incumbent"]["program_sha256s"]) == 2
    assert len(core["challenger"]["program_sha256s"]) == 2
    assert core["logical_retrieval_calls_per_compared_arm"] == 3
    assert core["fixed_cell_count"] == 8
    assert core["identical_action_has_no_runner_up_or_fallback"] is True


def test_behavior_deduplication_remains_gold_free_and_program_id_free() -> None:
    grid, _retained = _grid()
    first = grid.programs[3]
    alias = replace(
        first,
        program_sha256=_digest("identity-alias"),
        program_length=first.program_length + 7,
    )
    assert study._program_behavior_sha256(grid, first) == study._program_behavior_sha256(
        grid, alias
    )
    changed_gold = study.FormationGridEvidence(
        environment_ids=grid.environment_ids,
        items=tuple(
            tuple(replace(item, support_indices=(4, 5, 6, 7)) for item in env)
            for env in grid.items
        ),
        programs=grid.programs,
    )
    assert study._program_behavior_sha256(
        changed_gold, first
    ) == study._program_behavior_sha256(grid, first)


def test_exact_work_grids_equal_compute_and_l5_criterion() -> None:
    assert study.FORMATION_ENV_WORK_UNIT_COUNT == 2040
    assert study.FORMATION_WORK_UNIT_COUNT == 4080
    assert study.FORMATION_MAXIMUM_CONCURRENCY == 2040
    assert study.ANCHOR_WORK_UNIT_COUNT == 288
    assert study.SEARCH_WORK_UNIT_COUNT == 192
    anchor = study._anchor_execution_contract()
    search = study._search_execution_contract()
    assert anchor["logical_retrieval_calls_per_compared_arm_item"] == 3
    assert anchor["sole_promotion_criterion"] is True
    assert search["logical_retrieval_calls_per_primary_arm_item"] == 3
    assert search["M_search_does_not_change_evaluator_epoch"] is True
    assert study._l5_achievement(
        {"net_support_hit_count": 1, "paired_test": {"promoted": True}}
    ) is True
    assert study._l5_achievement(
        {"net_support_hit_count": 0, "paired_test": {"promoted": True}}
    ) is False
    assert study._l5_achievement(
        {"net_support_hit_count": 2, "paired_test": {"promoted": False}}
    ) is False


def test_exact_magnitude_sign_flip_alpha_is_inherited() -> None:
    accepted = study.exact_paired_sign_flip([1, 1, 1, 1])
    rejected = study.exact_paired_sign_flip([1, 1, 1])
    assert accepted["p_value_numerator"] == 1
    assert accepted["p_value_denominator"] == 16
    assert accepted["promoted"] is True
    assert rejected["p_value_denominator"] == 8
    assert rejected["promoted"] is False
    assert accepted["alpha_numerator"] == 1
    assert accepted["alpha_denominator"] == 10
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError):
        study.exact_paired_sign_flip([])


def _items(count: int, *, prefix: str) -> tuple[study.StudyItem, ...]:
    corpus = tuple(
        study.RetrievalParagraph(idx, f"title {idx}", f"text {idx}")
        for idx in range(10)
    )
    return tuple(
        study.StudyItem(
            view=study.RetrievalItem(
                question=f"question {ordinal}",
                corpus=corpus,
                item_commitment_sha256=_digest(f"{prefix}-{ordinal}"),
            ),
            support_indices=tuple(range(2 + ordinal % 3)),
        )
        for ordinal in range(count)
    )


def test_two_environment_barriers_join_before_support_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programs = study.fixed_programs()[:2]
    for module in (study, study.frozen_core):
        monkeypatch.setattr(module, "CANDIDATE_COUNT", 2)
        monkeypatch.setattr(module, "FORMATION_ENV_ITEM_COUNT", 2)
        monkeypatch.setattr(module, "FORMATION_ITEM_COUNT", 4)
    monkeypatch.setattr(study, "FORMATION_ENV_WORK_UNIT_COUNT", 6)
    monkeypatch.setattr(study, "FORMATION_WORK_UNIT_COUNT", 12)
    monkeypatch.setattr(study, "FORMATION_MAXIMUM_CONCURRENCY", 6)
    monkeypatch.setattr(study, "fixed_programs", lambda: programs)
    calls: list[str] = []

    def ranking(program: object, _item: object) -> tuple[int, ...]:
        calls.append(getattr(program, "program_hash", "P"))
        return (0, 1, 2, 3, 4)

    monkeypatch.setattr(study, "_ranking", ranking)
    grid, execution = study._evaluate_formation_grid(
        p_program=programs[0],
        environment_ids=study.A_FORM_ENVIRONMENTS,
        environments=(_items(2, prefix="a"), _items(2, prefix="b")),
    )
    assert len(calls) == 12
    assert grid.environment_ids == study.A_FORM_ENVIRONMENTS
    assert execution["physical_work_unit_count"] == 12
    assert execution["retrieval_terminal_count"] == 12
    assert execution["environment_barrier_count"] == 2
    assert execution["environment_barrier_party_count"] == 6
    assert execution["all_terminals_joined_before_support_scoring"] is True


def test_formation_marker_is_one_shot_and_parent_is_precreated(tmp_path: Path) -> None:
    binding = study._write_formation_marker(
        project=tmp_path,
        stage="A_form",
        acquisition_file_sha256=_digest("acquisition"),
        output_cache_path=tmp_path / "private.json",
        output_receipt_path=tmp_path / "public.json",
    )
    assert binding["marker_written_before_both_private_environment_blocks_open"]
    assert (tmp_path / study.A_FORM_CONSUMPTION_RELATIVE).is_file()
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError, match="already consumed"):
        study._write_formation_marker(
            project=tmp_path,
            stage="A_form",
            acquisition_file_sha256=_digest("acquisition"),
            output_cache_path=tmp_path / "private.json",
            output_receipt_path=tmp_path / "public.json",
        )


def test_persistence_preflight_creates_parent_and_leaves_no_canary(tmp_path: Path) -> None:
    output = tmp_path / "new" / "nested" / "receipt.json"
    study._prepare_output_parent(output)
    assert output.parent.is_dir()
    assert not output.exists()
    assert list(output.parent.iterdir()) == []
    output.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError, match="already exists"):
        study._prepare_output_parent(output)


def test_atomic_json_persistence_is_exclusive_and_cleans_failed_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "receipt.json"
    study._write_json_exclusive(output, {"value": 1})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        study._write_json_exclusive(output, {"value": 2})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))

    failed = tmp_path / "failed.json"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic link failure")

    monkeypatch.setattr(study.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic link failure"):
        study._write_json_exclusive(failed, {"value": 3})
    assert not failed.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_live_acquisition_rejects_dirty_implementation_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {
        "A_form_0": 24, "A_form_1": 24, "F_search_0": 24,
        "F_search_1": 24, "A_hold": 48, "M_search": 24,
    }
    implementation = {"set_sha256": _digest("implementation")}
    lineage = {"set_sha256": _digest("lineage")}
    design = {"design_sha256": _digest("design")}
    receipt = {
        "acquisition_sha256": _digest("acquisition"),
        "implementation": {"set_sha256": _digest("dirty")},
        "retained_P_lineage": lineage,
        "portfolio_design_binding": design,
    }
    rows = tuple(
        SimpleNamespace(
            block=block, count=count,
            file_sha256=_digest(f"file-{block}"),
            item_commitment_set_sha256=_digest(f"items-{block}"),
        )
        for block, count in counts.items()
    )
    fake = SimpleNamespace(
        BLOCK_COUNTS=counts,
        BLOCK_ORDER=tuple(counts),
        load_acquisition_binding_live=lambda **_kwargs: (receipt, rows),
        implementation_binding=lambda _project: implementation,
        prior_study_lineage_binding=lambda _project: lineage,
    )
    monkeypatch.setattr(study, "_acquisition_module", lambda: fake)
    monkeypatch.setattr(study, "_live_design_binding", lambda _project: design)
    path = tmp_path / "acquisition.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError, match="implementation"):
        study._load_acquisition_live(
            project=tmp_path, path=path, selection_secret_path=tmp_path / "secret"
        )
    receipt["implementation"] = implementation
    receipt["retained_P_lineage"] = {"set_sha256": _digest("tamper")}
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError, match="lineage"):
        study._load_acquisition_live(
            project=tmp_path, path=path, selection_secret_path=tmp_path / "secret"
        )


def test_runner_delegates_to_strict_canonical_live_acquisition_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def reject_forged(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        raise RuntimeError("synthetic forged alternate receipt")

    fake = SimpleNamespace(load_acquisition_binding_live=reject_forged)
    monkeypatch.setattr(study, "_acquisition_module", lambda: fake)
    alternate = tmp_path / "alternate-rehashed.json"
    alternate.write_text("{}\n", encoding="utf-8")
    secret = tmp_path / "selection.key"
    with pytest.raises(
        study.MuSiQueEvaluatorPortfolioError, match="canonical live acquisition"
    ):
        study._load_acquisition_live(
            project=tmp_path, path=alternate, selection_secret_path=secret
        )
    assert calls == [
        {"project": tmp_path, "path": alternate, "selection_secret_path": secret}
    ]


def test_freeze_signatures_have_no_measurement_or_injection_surface() -> None:
    assert study.formal_signatures_have_no_injection_surface() is True
    for function in (
        study.build_a_hold_pre_run_freeze,
        study.execute_a_hold_formal,
        study.build_m_search_pre_run_freeze,
        study.execute_m_search_formal,
    ):
        assert "selection_secret_path" in inspect.signature(function).parameters
    assert "a_hold_block_path" not in inspect.signature(
        study.build_a_hold_pre_run_freeze
    ).parameters
    assert "m_search_block_path" not in inspect.signature(
        study.build_m_search_pre_run_freeze
    ).parameters
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError, match="clean module CLI"):
        study.execute_a_hold_formal(
            project_root="unused", pre_run_freeze_path="unused",
            acquisition_receipt_path="unused", a_hold_block_path="unused",
            selection_secret_path="unused",
            p_formation_receipt_path="unused", p_frozen_program_path="unused",
            m1_freeze_path="unused", m1_report_path="unused",
            a_form_private_cache_path="unused", a_form_public_receipt_path="unused",
            f_search_private_cache_path="unused", f_search_public_receipt_path="unused",
            execution_root="unused",
        )


def test_public_safety_rejects_private_content_and_locator() -> None:
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError):
        study._assert_public_safe({"question": "secret"})
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError):
        study._assert_public_safe({"path": "/tmp/private"})
    study._assert_public_safe(
        {"program_sha256s": [_digest("a"), _digest("b")], "raw_content_persisted": False}
    )


def test_same_action_prevents_anchor_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = SimpleNamespace(program_hash=_digest("P"))
    monkeypatch.setattr(
        study,
        "_artifact_bundles",
        lambda **_kwargs: (
            {}, b"{}", {}, p, {},
            {"formation_core": {"measurable_contrast": False}},
            {"formation_core": {"measurable_contrast": True}}, {}, {},
        ),
    )
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError, match="must remain unopened"):
        study.build_a_hold_pre_run_freeze(
            project_root=tmp_path, acquisition_receipt_path="unused",
            selection_secret_path="unused",
            p_formation_receipt_path="unused", p_frozen_program_path="unused",
            m1_freeze_path="unused", m1_report_path="unused",
            a_form_private_cache_path="unused", a_form_public_receipt_path="unused",
            f_search_private_cache_path="unused", f_search_public_receipt_path="unused",
            execution_root=tmp_path / "never", authorization_hash=_digest("auth"),
            output_path=tmp_path / "freeze.json",
        )
    assert not (tmp_path / "freeze.json").exists()


def test_unpromoted_anchor_refuses_search_before_runtime_or_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        study,
        "load_and_reverify_a_hold",
        lambda **_kwargs: (
            {"evaluator_epoch_transition": {
                "promoted": False, "selective_invalidation_performed": False,
                "independent_source_record_retained": True,
            }},
            {},
        ),
    )
    monkeypatch.setattr(study, "_artifact_bundles", lambda **_kwargs: calls.append("opened"))
    with pytest.raises(study.MuSiQueEvaluatorPortfolioError, match="must remain unopened"):
        study.build_m_search_pre_run_freeze(
            project_root=tmp_path, acquisition_receipt_path="unused",
            selection_secret_path="unused",
            p_formation_receipt_path="unused", p_frozen_program_path="unused",
            m1_freeze_path="unused", m1_report_path="unused",
            a_form_private_cache_path="unused", a_form_public_receipt_path="unused",
            f_search_private_cache_path="unused", f_search_public_receipt_path="unused",
            a_hold_pre_run_freeze_path="unused", a_hold_private_evidence_path="unused",
            a_hold_report_path="unused", capability_receipt_path="unused",
            runtime_python="unused", local_llm_model="unused",
            local_embedding_model="unused", base_binding_receipt_path="unused",
            attestation_receipt_path="unused", execution_root=tmp_path / "never",
            authorization_hash=_digest("auth"), output_path=tmp_path / "search.json",
        )
    assert calls == []


def test_archive_transition_selectively_invalidates_only_on_promotion() -> None:
    promoted = study._archive_transition(
        anchor_manifest_sha256=_digest("anchor"), incumbent_hits=80,
        challenger_hits=90, support_total=140, item_count=48, promoted=True,
    )
    assert promoted["selective_invalidation_performed"] is True
    assert promoted["dependent_score_valid_after_transition"] is False
    assert promoted["independent_source_score_valid_after_transition"] is True
    rejected = study._archive_transition(
        anchor_manifest_sha256=_digest("anchor-2"), incumbent_hits=80,
        challenger_hits=79, support_total=140, item_count=48, promoted=False,
    )
    assert rejected["invalidated_score_record_ids"] == []
    assert rejected["dependent_score_valid_after_transition"] is True

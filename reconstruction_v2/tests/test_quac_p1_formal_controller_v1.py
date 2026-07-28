from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools

import pytest

from assumption_agent.benchmarks import quac_p1_formal_controller_v1 as core
from assumption_agent.benchmarks import quac_rjmc_evaluator_v1 as evaluator


def _id(value: object) -> str:
    return hashlib.sha256(repr(value).encode("ascii")).hexdigest()


def _unit(index: int) -> evaluator.EvidenceUnit:
    return evaluator.EvidenceUnit(
        unit_id=_id(("unit", index)),
        node_features=(0.1 * index, 0.0, 0.0, 0.0),
        dialogue_facets=(int(index == 0), int(index == 1), 0, 0),
    )


def _graph(count: int = 11) -> evaluator.RelationalGraph:
    return evaluator.RelationalGraph(
        units=tuple(
            sorted(
                (_unit(index) for index in range(count)),
                key=lambda unit: unit.unit_id,
            )
        ),
        edges=(),
    )


def _action(item_id: str, *, e1_complete: bool = True) -> core.ActionRow:
    units = tuple(_id(("unit", index)) for index in range(5))
    qprev = _id("qprev")
    qcurr = _id("qcurr")
    return core.ActionRow(
        item_id=item_id,
        E0=units,
        E1=(
            (qprev, qcurr, *units[2:])
            if e1_complete
            else units
        ),
        RAW=units,
        official_HippoRAG=(qprev, *units[1:]),
    )


def _labels(count_per_family: int = 32) -> tuple[core.LateLabelRow, ...]:
    return tuple(
        core.LateLabelRow(
            item_id=_id(("item", family, index)),
            family=family,
            previous_qrel=(_id("qprev"),),
            current_qrel=(_id("qcurr"),),
        )
        for family in core.FAMILY_ORDER
        for index in range(count_per_family)
    )


def _measurement_corpus() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *(_id(("unit", index)) for index in range(5)),
                _id("qprev"),
                _id("qcurr"),
            }
        )
    )


def _a_form_corpus() -> tuple[str, ...]:
    return tuple(sorted(_id(("unit", index)) for index in range(5)))


def _sealed_actions(
    *,
    block: str,
    labels: tuple[core.LateLabelRow, ...],
    corpus: tuple[str, ...],
) -> core.SealedStageActions:
    return core.SealedStageActions(
        block=block,
        corpus_unit_ids_sha256=core.stable_hash(list(corpus)),
        rows=tuple(
            sorted(
                (_action(label.item_id) for label in labels),
                key=lambda row: row.item_id,
            )
        ),
    )


def test_graph_bound_is_eleven_units_six_candidates_and_181_states() -> None:
    item = core.LabelFreeGraphItem(
        item_id=_id("bounded"),
        fold=0,
        graph=_graph(11),
        raw_top5=tuple(_id(("unit", index)) for index in range(5)),
    )
    assert len(
        evaluator.enumerate_complete_states(
            item.graph, raw_top5=item.raw_top5
        )
    ) == 181
    with pytest.raises(core.QuacP1FormalControllerError, match="bounded"):
        core.LabelFreeGraphItem(
            item_id=_id("unbounded"),
            fold=0,
            graph=_graph(12),
            raw_top5=tuple(_id(("unit", index)) for index in range(5)),
        )


def test_two_role_utility_allows_one_window_to_satisfy_both_roles() -> None:
    same = _id("same")
    elsewhere = _id("elsewhere")
    outside = _id("outside")
    selected = (same, *tuple(_id(("other", index)) for index in range(4)))
    assert core.two_role_utility(
        selected,
        previous_qrel=(same,),
        current_qrel=(same,),
    ) == 4
    assert core.two_role_utility(
        selected,
        previous_qrel=(same,),
        current_qrel=(elsewhere,),
    ) == 1
    assert core.two_role_utility(
        selected,
        previous_qrel=(outside,),
        current_qrel=(elsewhere,),
    ) == 0


def test_exact_sign_flip_dynamic_program_matches_small_bruteforce() -> None:
    for deltas in (
        (0, 0, 0),
        (1, 1, -1),
        (4, -2, 1, 0),
        (-4, -3, 2),
    ):
        result = core.exact_magnitude_preserving_sign_flip(deltas)
        magnitudes = [abs(value) for value in deltas if value]
        signed = [
            sum(sign * magnitude for sign, magnitude in zip(signs, magnitudes))
            for signs in itertools.product((-1, 1), repeat=len(magnitudes))
        ]
        brute = Fraction(
            sum(value >= sum(deltas) for value in signed),
            len(signed),
        )
        assert result.p == brute
    assert core.exact_magnitude_preserving_sign_flip((0, 0)).p == 1


def test_a_form_requires_exact_balanced_folds_and_uses_all_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(5)
    items = tuple(
        core.LabelFreeGraphItem(
            item_id=_id(("train", index)),
            fold=(0 if index < 39 else
                  1 if index < 78 else
                  2 if index < 116 else
                  3 if index < 154 else 4),
            graph=graph,
            raw_top5=tuple(_id(("unit", unit)) for unit in range(5)),
        )
        for index in range(192)
    )
    labels = tuple(
        core.LateLabelRow(
            item_id=item.item_id,
            family=core.FAMILY_ORDER[index % 3],
            previous_qrel=(_id(("unit", 0)),),
            current_qrel=(_id(("unit", 1)),),
        )
        for index, item in enumerate(items)
    )
    observed = {}

    def fake_fit(rows, *, config):
        observed["rows"] = tuple(rows)
        observed["config"] = config
        return "fit-once"

    monkeypatch.setattr(evaluator, "fit_component_jackknife", fake_fit)
    corpus = _a_form_corpus()
    assert core.fit_a_form_once(
        items,
        labels,
        block_corpus_unit_ids=corpus,
    ) == "fit-once"
    assert len(observed["rows"]) == 192
    assert all(row.utility == (4,) for row in observed["rows"])
    assert observed["config"] == evaluator.FitConfig()

    bad = list(items)
    bad[-1] = core.LabelFreeGraphItem(
        item_id=bad[-1].item_id,
        fold=3,
        graph=graph,
        raw_top5=bad[-1].raw_top5,
    )
    with pytest.raises(core.QuacP1FormalControllerError, match="fold balance"):
        core.fit_a_form_once(
            bad,
            labels,
            block_corpus_unit_ids=corpus,
        )


def test_four_arm_barrier_precedes_late_scoring_and_primary_is_conjunction() -> None:
    labels = _labels()
    corpus = _measurement_corpus()
    actions = _sealed_actions(
        block="A_hold",
        labels=labels,
        corpus=corpus,
    )
    score = core.score_sealed_stage(
        actions,
        labels,
        block_corpus_unit_ids=corpus,
    )
    assert score.item_count == 96
    assert score.promotion is True
    assert score.reality_primary is True
    assert score.comparison("RAW").all_families_positive is True
    assert score.comparison("official_HippoRAG").all_families_positive is True
    assert score.comparison("RAW").exact.p <= Fraction(1, 10)
    assert "paired_deltas" not in score.safe_payload()["comparisons"][0]
    assert (
        "paired_delta_sha256"
        not in score.safe_payload()["comparisons"][0]
    )
    assert "private_item_score_sha256" not in score.safe_payload()

    missing = labels[:-1]
    with pytest.raises(core.QuacP1FormalControllerError, match="do not match"):
        core.score_sealed_stage(
            actions,
            missing,
            block_corpus_unit_ids=corpus,
        )


def test_m_presence_is_controlled_only_by_promotion() -> None:
    labels = _labels()
    corpus = _measurement_corpus()
    hold_actions = _sealed_actions(
        block="A_hold",
        labels=labels,
        corpus=corpus,
    )
    hold = core.score_sealed_stage(
        hold_actions,
        labels,
        block_corpus_unit_ids=corpus,
    )
    m_actions = _sealed_actions(
        block="M_search",
        labels=labels,
        corpus=corpus,
    )
    m = core.score_sealed_stage(
        m_actions,
        labels,
        block_corpus_unit_ids=corpus,
    )
    terminal = core.safe_terminal(
        a_hold=hold,
        m_search=m,
        model_parameter_sha256="a" * 64,
        action_commitments={
            "A_form_label_free_actions": "c" * 64,
            "A_hold_four_arm_actions": hold_actions.action_sha256,
            "M_search_four_arm_actions": m_actions.action_sha256,
        },
        runtime_commitments={
            "A_form_runtime": "d" * 64,
            "A_hold_runtime": "e" * 64,
            "M_search_runtime": "f" * 64,
        },
        M_materialization_count_before_promotion=0,
    )
    assert terminal["A_hold_promotion"] is True
    assert terminal["M_search_opened"] is True
    assert terminal["M_search_L5"] is True
    assert terminal["total_goal_success"] is True
    assert terminal["execution_design_self_sha256"] == (
        "def417300b3c25f127517eef1cdd61760757762f08cc5a9b9877b261036dace2"
    )

    with pytest.raises(
        core.QuacP1FormalControllerError, match="does not equal promotion"
    ):
        core.safe_terminal(
            a_hold=hold,
            m_search=None,
            model_parameter_sha256="a" * 64,
            action_commitments={},
            runtime_commitments={},
            M_materialization_count_before_promotion=0,
        )
    with pytest.raises(
        core.QuacP1FormalControllerError, match="valid terminal inputs"
    ):
        core.safe_terminal(
            a_hold=hold,
            m_search=m,
            model_parameter_sha256="a" * 64,
            action_commitments={},
            runtime_commitments={},
            M_materialization_count_before_promotion=1,
        )

    valid_actions = {
        "A_form_label_free_actions": "c" * 64,
        "A_hold_four_arm_actions": hold_actions.action_sha256,
        "M_search_four_arm_actions": m_actions.action_sha256,
    }
    valid_runtime = {
        "A_form_runtime": "d" * 64,
        "A_hold_runtime": "e" * 64,
        "M_search_runtime": "f" * 64,
    }
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="model parameter commitment",
    ):
        core.safe_terminal(
            a_hold=hold,
            m_search=m,
            model_parameter_sha256="G" * 64,
            action_commitments=valid_actions,
            runtime_commitments=valid_runtime,
            M_materialization_count_before_promotion=0,
        )
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="action commitment registry",
    ):
        core.safe_terminal(
            a_hold=hold,
            m_search=m,
            model_parameter_sha256="a" * 64,
            action_commitments={
                **valid_actions,
                "unexpected": "0" * 64,
            },
            runtime_commitments=valid_runtime,
            M_materialization_count_before_promotion=0,
        )
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="M_search_runtime commitment",
    ):
        core.safe_terminal(
            a_hold=hold,
            m_search=m,
            model_parameter_sha256="a" * 64,
            action_commitments=valid_actions,
            runtime_commitments={
                **valid_runtime,
                "M_search_runtime": "not-a-hash",
            },
            M_materialization_count_before_promotion=0,
        )


def test_nonpromotion_terminal_forbids_all_m_search_commitments() -> None:
    labels = _labels()
    corpus = _measurement_corpus()
    hold_actions = core.SealedStageActions(
        block="A_hold",
        corpus_unit_ids_sha256=core.stable_hash(list(corpus)),
        rows=tuple(
            sorted(
                (
                    _action(label.item_id, e1_complete=False)
                    for label in labels
                ),
                key=lambda row: row.item_id,
            )
        ),
    )
    hold = core.score_sealed_stage(
        hold_actions,
        labels,
        block_corpus_unit_ids=corpus,
    )
    assert hold.promotion is False
    actions = {
        "A_form_label_free_actions": "c" * 64,
        "A_hold_four_arm_actions": hold_actions.action_sha256,
    }
    runtimes = {
        "A_form_runtime": "d" * 64,
        "A_hold_runtime": "e" * 64,
    }
    terminal = core.safe_terminal(
        a_hold=hold,
        m_search=None,
        model_parameter_sha256="a" * 64,
        action_commitments=actions,
        runtime_commitments=runtimes,
        M_materialization_count_before_promotion=0,
    )
    assert terminal["status"] == "VALID_NONPROMOTION_M_UNOPENED"
    assert terminal["M_search_opened"] is False
    assert terminal["M_search"] is None

    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="action commitment registry",
    ):
        core.safe_terminal(
            a_hold=hold,
            m_search=None,
            model_parameter_sha256="a" * 64,
            action_commitments={
                **actions,
                "M_search_four_arm_actions": "f" * 64,
            },
            runtime_commitments=runtimes,
            M_materialization_count_before_promotion=0,
        )


def test_formal_measurement_registry_rejects_tiny_or_quota_drifted_rows() -> None:
    corpus = _measurement_corpus()
    tiny_labels = _labels(4)
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="exact formal registry",
    ):
        _sealed_actions(
            block="A_hold",
            labels=tiny_labels,
            corpus=corpus,
        )

    labels = list(_labels())
    labels[0] = core.LateLabelRow(
        item_id=labels[0].item_id,
        family="MAYBE_FOLLOW",
        previous_qrel=labels[0].previous_qrel,
        current_qrel=labels[0].current_qrel,
    )
    actions = _sealed_actions(
        block="A_hold",
        labels=tuple(labels),
        corpus=corpus,
    )
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="family quota",
    ):
        core.score_sealed_stage(
            actions,
            labels,
            block_corpus_unit_ids=corpus,
        )


def test_scoring_binds_actions_qrels_and_commitment_to_complete_corpus() -> None:
    labels = _labels()
    complete_corpus = _measurement_corpus()
    actions = _sealed_actions(
        block="A_hold",
        labels=labels,
        corpus=complete_corpus,
    )
    wrong_corpus = tuple(
        unit_id
        for unit_id in complete_corpus
        if unit_id != _id("qprev")
    )
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="do not match",
    ):
        core.score_sealed_stage(
            actions,
            labels,
            block_corpus_unit_ids=wrong_corpus,
        )

    wrongly_bound_actions = _sealed_actions(
        block="A_hold",
        labels=labels,
        corpus=wrong_corpus,
    )
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="escaped",
    ):
        core.score_sealed_stage(
            wrongly_bound_actions,
            labels,
            block_corpus_unit_ids=wrong_corpus,
        )


def test_a_form_rejects_graph_or_qrel_outside_complete_corpus() -> None:
    graph = _graph(5)
    items = tuple(
        core.LabelFreeGraphItem(
            item_id=_id(("train-corpus", index)),
            fold=(
                0 if index < 39 else
                1 if index < 78 else
                2 if index < 116 else
                3 if index < 154 else 4
            ),
            graph=graph,
            raw_top5=tuple(
                _id(("unit", unit)) for unit in range(5)
            ),
        )
        for index in range(192)
    )
    labels = tuple(
        core.LateLabelRow(
            item_id=item.item_id,
            family=core.FAMILY_ORDER[index % 3],
            previous_qrel=(_id(("unit", 0)),),
            current_qrel=(_id(("unit", 1)),),
        )
        for index, item in enumerate(items)
    )
    incomplete_corpus = tuple(
        sorted(
            {
                *(
                    unit_id
                    for unit_id in _a_form_corpus()
                    if unit_id != _id(("unit", 0))
                ),
                _id("unrelated-corpus-unit"),
            }
        )
    )
    with pytest.raises(
        core.QuacP1FormalControllerError,
        match="escaped",
    ):
        core.fit_a_form_once(
            items,
            labels,
            block_corpus_unit_ids=incomplete_corpus,
        )

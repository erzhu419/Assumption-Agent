from __future__ import annotations

import copy

import pytest

from assumption_agent.benchmarks.musique_evaluator_coevolution_v1 import (
    EvaluatorFormationError,
    ItemRetrievalEvidence,
    ProgramRetrievalEvidence,
    compare_on_fixed_anchor,
    form_evaluator_challenger,
    freeze_prospective_search_formation,
    measure_prospective_search_utility,
)
from assumption_agent.models import stable_hash


def _program(name: str, hits: list[int], totals: list[int]) -> ProgramRetrievalEvidence:
    return ProgramRetrievalEvidence(
        program_sha256=stable_hash({"program": name}),
        program_length=5,
        items=tuple(
            ItemRetrievalEvidence(
                item_commitment_sha256=stable_hash({"item": index}),
                support_hits=hit,
                support_total=total,
                invalid=False,
                retrieval_sha256=stable_hash(
                    {"program": name, "item": index, "hit": hit}
                ),
            )
            for index, (hit, total) in enumerate(zip(hits, totals))
        ),
    )


def _evidence() -> tuple[ProgramRetrievalEvidence, ...]:
    # Micro recall overweights the first high-cardinality item.  Macro recall
    # selects the consistently useful program and therefore wins cross-fit.
    totals = [10, 1, 1, 1, 10, 1, 1, 1]
    bursty = _program("bursty", [10, 0, 0, 0, 10, 0, 0, 0], totals)
    stable = _program("stable", [5, 1, 1, 1, 5, 1, 1, 1], totals)
    return (bursty, stable)


def _held_evidence() -> tuple[ProgramRetrievalEvidence, ...]:
    totals = [2] * 8
    bursty = _program("bursty", [2, 0, 0, 0, 2, 0, 0, 0], totals)
    stable = _program("stable", [1, 2, 2, 2, 1, 2, 2, 2], totals)
    return (bursty, stable)


def test_challenger_is_formed_without_anchor_access() -> None:
    receipt = form_evaluator_challenger(_evidence())
    assert receipt["partition"] == "A_form"
    assert receipt["anchor_accessed"] is False
    assert receipt["challenger_rule"]["id"] != receipt["incumbent_rule"]["id"]
    assert receipt["model_calls"] == 0


def test_anchor_transition_and_g3_search_utility_are_prospective() -> None:
    formation = _evidence()
    receipt = form_evaluator_challenger(formation)
    anchor = compare_on_fixed_anchor(
        formation_evidence=formation,
        anchor_evidence=_held_evidence(),
        formation_receipt=receipt,
    )
    assert anchor["challenger_promoted"] is True
    assert anchor["official_support_objective_replaced"] is False
    search_formation = freeze_prospective_search_formation(
        formation_evidence=formation,
        evaluator_formation_evidence=formation,
        evaluator_formation_receipt=receipt,
    )
    utility = measure_prospective_search_utility(
        formation_evidence=formation,
        measurement_evidence=_held_evidence(),
        evaluator_formation_evidence=formation,
        anchor_evidence=_held_evidence(),
        evaluator_formation_receipt=receipt,
        search_formation_receipt=search_formation,
        anchor_result=anchor,
    )
    assert utility["active_support_hits"] > utility["incumbent_support_hits"]
    assert utility["evaluator_transition_had_positive_search_utility"] is True
    assert utility["model_calls"] == 0


def test_rehashed_rule_or_search_receipt_tamper_fails_closed() -> None:
    formation = _evidence()
    receipt = form_evaluator_challenger(formation)
    tampered = copy.deepcopy(receipt)
    tampered["challenger_rule"]["macro_weight"] += 1
    body = dict(tampered)
    body.pop("formation_sha256")
    tampered["formation_sha256"] = stable_hash(body)
    with pytest.raises(EvaluatorFormationError, match="rule payload"):
        compare_on_fixed_anchor(
            formation_evidence=formation,
            anchor_evidence=_held_evidence(),
            formation_receipt=tampered,
        )

    anchor = compare_on_fixed_anchor(
        formation_evidence=formation,
        anchor_evidence=_held_evidence(),
        formation_receipt=receipt,
    )
    search = freeze_prospective_search_formation(
        formation_evidence=formation,
        evaluator_formation_evidence=formation,
        evaluator_formation_receipt=receipt,
    )
    search["challenger_selected_program_sha256"] = "0" * 64
    search_body = dict(search)
    search_body.pop("search_formation_sha256")
    search["search_formation_sha256"] = stable_hash(search_body)
    with pytest.raises(EvaluatorFormationError, match="selected program"):
        measure_prospective_search_utility(
            formation_evidence=formation,
            measurement_evidence=_held_evidence(),
            evaluator_formation_evidence=formation,
            anchor_evidence=_held_evidence(),
            evaluator_formation_receipt=receipt,
            search_formation_receipt=search,
            anchor_result=anchor,
        )

    forged_anchor = dict(anchor)
    forged_anchor["challenger_promoted"] = False
    anchor_body = dict(forged_anchor)
    anchor_body.pop("anchor_result_sha256")
    forged_anchor["anchor_result_sha256"] = stable_hash(anchor_body)
    with pytest.raises(EvaluatorFormationError, match="fixed A_hold"):
        measure_prospective_search_utility(
            formation_evidence=formation,
            measurement_evidence=_held_evidence(),
            evaluator_formation_evidence=formation,
            anchor_evidence=_held_evidence(),
            evaluator_formation_receipt=receipt,
            search_formation_receipt=freeze_prospective_search_formation(
                formation_evidence=formation,
                evaluator_formation_evidence=formation,
                evaluator_formation_receipt=receipt,
            ),
            anchor_result=forged_anchor,
        )

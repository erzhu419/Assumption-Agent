from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Decimal
import inspect

import pytest

from assumption_agent.benchmarks import wikisql_uao_policy_v1 as policy
from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality


HEADERS = ("Name", "Score")
TYPES = ("text", "real")


def _serialize(name: str, score: int) -> str:
    return (
        f'column[0] "Name" (text) = {name}\n'
        f'column[1] "Score" (real) = {score}'
    )


def _label_free(
    *,
    question: str,
    scores: tuple[int, ...],
    raw_rows: tuple[int, ...],
    prefix: str,
) -> policy.LabelFreeItem:
    raw = raw_rows + (None,) * (policy.TOP_K - len(raw_rows))
    return policy.LabelFreeItem(
        question=question,
        headers=HEADERS,
        types=TYPES,
        serialized_rows=tuple(
            _serialize(f"{prefix}_{index}", score)
            for index, score in enumerate(scores)
        ),
        raw_top5=raw,
    )


def _training_items() -> tuple[policy.TrainingItem, ...]:
    result: list[policy.TrainingItem] = []
    # EQ examples have one target outside the deliberately unhelpful RAW five.
    for index in range(policy.A_FORM_QUOTA_PER_FAMILY):
        target = 1_000 + index
        scores = (10, 20, 30, 40, 50, 60, target, 2_000 + index)
        result.append(
            policy.TrainingItem(
                item=_label_free(
                    question=f"Which Name has Score equal to {target} for eq{index}?",
                    scores=scores,
                    raw_rows=(0, 1, 2, 3, 4),
                    prefix=f"eq{index}",
                ),
                gold_row_ids=(6,),
                family="EQ",
                fold_index=index % policy.CROSS_FOLDS,
            )
        )
    # GT examples have two targets.  Question-specific numeric interaction is
    # necessary because lexical RAW excludes both satisfying rows.
    for index in range(policy.A_FORM_QUOTA_PER_FAMILY):
        threshold = 500 + index
        scores = (10, 20, 30, 35, 40, 42, threshold + 10, threshold + 20)
        result.append(
            policy.TrainingItem(
                item=_label_free(
                    question=(
                        f"Which Name has Score greater than {threshold} "
                        f"for gt{index}?"
                    ),
                    scores=scores,
                    raw_rows=(0, 1, 2, 3, 4),
                    prefix=f"gt{index}",
                ),
                gold_row_ids=(6, 7),
                family="GT",
                fold_index=index % policy.CROSS_FOLDS,
            )
        )
    # LT examples place the satisfying rows outside a different RAW five.
    for index in range(policy.A_FORM_QUOTA_PER_FAMILY):
        threshold = 500 + index
        scores = (
            threshold - 20,
            threshold - 10,
            threshold + 10,
            threshold + 20,
            threshold + 30,
            threshold + 40,
            threshold + 50,
            threshold + 60,
        )
        result.append(
            policy.TrainingItem(
                item=_label_free(
                    question=(
                        f"Which Name has Score less than {threshold} "
                        f"for lt{index}?"
                    ),
                    scores=scores,
                    raw_rows=(2, 3, 4, 5, 6),
                    prefix=f"lt{index}",
                ),
                gold_row_ids=(0, 1),
                family="LT",
                fold_index=index % policy.CROSS_FOLDS,
            )
        )
    return tuple(result)


@pytest.fixture(scope="module")
def formation() -> policy.PolicyFormation:
    return policy.fit_uao_policy(_training_items())


def test_typed_normalization_and_eq_gt_lt_anchor_extraction() -> None:
    assert policy.normalize_text("  Ａcme   SCORE ") == "acme score"
    assert policy.normalize_number("$1,250.50") == pytest.approx(1250.5)
    assert policy.normalize_number("(42)") == pytest.approx(-42.0)
    assert policy.normalize_number("not-a-number") is None
    assert policy.extract_anchors("score equal to 50").comparator == "EQ"
    assert policy.extract_anchors("score greater than 50").comparator == "GT"
    assert policy.extract_anchors("score below 50").comparator == "LT"


def test_shared_parser_accepts_marker_text_in_header_and_cell() -> None:
    table = reality.WikiSQLTable(
        table_id="marker",
        header=('Name " (text) = literal', "Score"),
        types=("text", "real"),
        rows=(
            ('value " (text) = payload', 1),
            ("second", 2),
            ("third", 3),
            ("fourth", 4),
            ("fifth", 5),
        ),
    )
    documents = reality.validated_retrieval_documents(table)
    item = policy.LabelFreeItem(
        question="Which value has score one?",
        headers=table.header,
        types=table.types,
        serialized_rows=documents,
        raw_top5=(0, 1, 2, 3, 4),
    )
    assert item.serialized_rows == documents
    assert policy._parse_serialized_row(  # noqa: SLF001
        documents[0],
        table.header,
        table.types,
    ).values[0] == 'value " (text) = payload'


def test_expected_utility_conditions_hits_and_complete_on_nonempty_relevance() -> None:
    # Two independent p=1/2 rows, selecting only row 0:
    #   E[hits; A] = 1/2
    #   P(complete; A) = P(row1=0) - P(all=0) = 1/2 - 1/4
    #   P(A) = 1 - 1/4
    # so E[hits + complete | A] = (1/2 + 1/2 - 1/4)/(3/4) = 1.
    # The previous mixed conditional/unconditional formula returned 5/6.
    assert policy._learned_expected_utility(
        (0,), (500_000, 500_000)
    ) == Decimal(1)

    # Selecting both rows has conditional expected hits 4/3 and complete 1.
    assert policy._learned_expected_utility(
        (0, 1), (500_000, 500_000)
    ) == Decimal(7) / Decimal(3)
    assert policy._learned_expected_utility((0,), (0, 0)) == Decimal(0)


def test_four_complete_fixed_claim_recipes() -> None:
    assert tuple(row.operator_template for row in policy.CLAIM_RECIPES) == (
        "T02",
        "T05",
        "T08",
        "T18",
    )
    assert len({row.claim_id for row in policy.CLAIM_RECIPES}) == 4
    assert all(row.feature_names and row.description for row in policy.CLAIM_RECIPES)


def test_four_fold_receipts_select_exactly_two_prediction_distinct_claims(
    formation: policy.PolicyFormation,
) -> None:
    assert len(formation.probe_receipts) == 4
    for receipt in formation.probe_receipts:
        assert len(receipt.fold_receipts) == policy.CROSS_FOLDS
        assert (
            receipt.support_count
            + receipt.counter_count
            + receipt.neutral_count
            == policy.A_FORM_QUOTA_PER_FAMILY * len(policy.FAMILY_ORDER)
        )
        assert all(
            dict(fold.family_counts)
            == {
                family: policy.A_FORM_QUOTA_PER_FAMILY // policy.CROSS_FOLDS
                for family in policy.FAMILY_ORDER
            }
            for fold in receipt.fold_receipts
        )
        assert receipt.safe_receipt()["self_sha256"] == policy.canonical_sha256(
            receipt.payload()
        )
        assert receipt.payload()["train_only"] is True
        assert receipt.payload()["heldout_access_count"] == 0
    selected = formation.policy.selected_claim_ids
    assert len(selected) == policy.SELECTED_CLAIM_COUNT == 2
    by_id = {row.claim_id: row for row in formation.probe_receipts}
    assert by_id[selected[0]].prediction_vector != by_id[selected[1]].prediction_vector
    selection = formation.claim_selection_receipt
    selected_second = next(
        row for row in selection.candidates if row.claim_id == selected[1]
    )
    assert (
        selected_second.hamming_from_first
        >= policy.ACTION_VECTOR_MIN_HAMMING
    )
    assert selected_second.adjusted_score == (
        selected_second.base_score
        - policy.ACTION_REDUNDANCY_PENALTY_PER_MATCH
        * selected_second.redundant_action_count
    )
    assert (
        selection.safe_receipt()["self_sha256"]
        == selection.receipt_sha256
    )

    calibration = formation.no_op_calibration_receipt
    assert tuple(
        row.margin_threshold for row in calibration.threshold_evaluations
    ) == policy.NO_OP_MARGIN_GRID
    assert calibration.selected_margin_threshold == formation.policy.margin_threshold
    assert formation.policy.margin_threshold > 0
    assert all(
        dict(rows)
        == {
            family: policy.A_FORM_QUOTA_PER_FOLD_FAMILY
            for family in policy.FAMILY_ORDER
        }
        for rows in calibration.fold_family_counts
    )
    assert calibration.safe_receipt()["self_sha256"] == calibration.receipt_sha256
    safe = formation.policy.safe_receipt()
    assert safe["self_sha256"] == policy.canonical_sha256(
        {key: value for key, value in safe.items() if key != "self_sha256"}
    )


def test_formation_is_permutation_invariant(
    formation: policy.PolicyFormation,
) -> None:
    reverse = policy.fit_uao_policy(tuple(reversed(_training_items())))
    assert reverse.policy.policy_sha256 == formation.policy.policy_sha256
    assert reverse.policy.safe_receipt() == formation.policy.safe_receipt()
    assert tuple(row.safe_receipt() for row in reverse.probe_receipts) == tuple(
        row.safe_receipt() for row in formation.probe_receipts
    )
    assert (
        reverse.claim_selection_receipt.safe_receipt()
        == formation.claim_selection_receipt.safe_receipt()
    )
    assert (
        reverse.no_op_calibration_receipt.safe_receipt()
        == formation.no_op_calibration_receipt.safe_receipt()
    )


def test_sealed_fold_indices_are_consumed_not_recomputed() -> None:
    rows = list(_training_items())
    # Moving one EQ item creates a 15/17 sealed fold imbalance.  A commitment-
    # based fold recomputation would silently ignore this mutation; formation
    # must instead reject it.
    rows[0] = replace(rows[0], fold_index=(rows[0].fold_index + 1) % 4)
    with pytest.raises(
        policy.WikiSQLUAOPolicyError,
        match="16 items per family per fold",
    ):
        policy.fit_uao_policy(tuple(rows))


def test_claim_diversity_uses_action_hamming_not_probability_hashes(
    formation: policy.PolicyFormation,
) -> None:
    receipts = list(formation.probe_receipts)
    first = sorted(
        receipts,
        key=lambda row: (-row.selection_score, row.claim_id),
    )[0]
    duplicate_index = next(
        index for index, row in enumerate(receipts)
        if row.claim_id != first.claim_id
    )
    duplicate_id = receipts[duplicate_index].claim_id
    # Make a competing claim's OOF selected-top5/no-op vector exactly equal.
    # Different fitted probabilities are intentionally irrelevant here.
    receipts[duplicate_index] = replace(
        receipts[duplicate_index],
        prediction_vector=first.prediction_vector,
    )
    selected, selection_receipt = policy._select_claims(tuple(receipts))
    duplicate = next(
        row for row in selection_receipt.candidates
        if row.claim_id == duplicate_id
    )
    assert duplicate.hamming_from_first == 0
    assert duplicate.eligible is False
    assert duplicate_id not in selected


def test_heldout_apply_signature_is_label_and_sql_free() -> None:
    parameters = set(inspect.signature(policy.apply_uao_policy).parameters)
    assert parameters == {
        "policy",
        "question",
        "headers",
        "types",
        "serialized_rows",
        "raw_top5",
        "embeddings",
    }
    assert not parameters.intersection(
        {
            "gold",
            "gold_rows",
            "gold_row_ids",
            "family",
            "sql",
            "aggregation",
            "conditions",
            "answer",
        }
    )


def test_byte_exact_no_op_and_score_tie_invariance(
    formation: policy.PolicyFormation,
) -> None:
    item = _label_free(
        question="Which Name has Score equal to 30?",
        scores=(10, 20, 30, 40, 50),
        raw_rows=(4, 3, 2, 1, 0),
        prefix="short",
    )
    result = policy.apply_uao_policy(
        formation.policy,
        question=item.question,
        headers=item.headers,
        types=item.types,
        serialized_rows=item.serialized_rows,
        raw_top5=item.raw_top5,
    )
    # All physical rows already fit in RAW.  Learned ties or re-ordering have
    # zero expected set gain and must preserve the exact caller tuple object.
    assert result is item.raw_top5
    assert result == (4, 3, 2, 1, 0)


def test_positive_train_oof_threshold_blocks_scored_but_harmful_candidate(
    formation: policy.PolicyFormation,
) -> None:
    item = _label_free(
        question="Which Name has Score exactly 73?",
        scores=(10, 11, 12, 13, 14, 73),
        raw_rows=(0, 1, 2, 3, 4),
        prefix="weak_margin",
    )
    prepared = policy._prepare(item)
    scores = policy._probability_scores(
        formation.policy.model,
        prepared,
        lambda value, row_index: policy._union_features(
            value,
            row_index,
            formation.policy.selected_claim_ids,
        ),
    )
    candidate, margin = policy._candidate_and_margin(item.raw_top5, scores)
    assert candidate != item.raw_top5
    assert 0 < margin <= formation.policy.margin_threshold
    # This test-only late label makes the scored swap genuinely harmful:
    # candidate drops row 0, while RAW contains and completes the gold set.
    gold = (0,)
    assert policy._utility(candidate, gold) < policy._utility(item.raw_top5, gold)

    action = policy.apply_uao_policy(
        formation.policy,
        question=item.question,
        headers=item.headers,
        types=item.types,
        serialized_rows=item.serialized_rows,
        raw_top5=item.raw_top5,
    )
    assert action is item.raw_top5

    reconstructed = policy.compiled_policy_from_private_payload(
        formation.policy.content_addressed_private_payload()
    )
    assert reconstructed.margin_threshold == formation.policy.margin_threshold
    rebuilt_action = policy.apply_uao_policy(
        reconstructed,
        question=item.question,
        headers=item.headers,
        types=item.types,
        serialized_rows=item.serialized_rows,
        raw_top5=item.raw_top5,
    )
    assert rebuilt_action is item.raw_top5


def test_synthetic_eq_gt_lt_actions_improve_over_raw(
    formation: policy.PolicyFormation,
) -> None:
    cases = (
        (
            _label_free(
                question="Which Name has Score exactly 73 in heldout EQ?",
                scores=(10, 20, 30, 40, 50, 60, 73, 90),
                raw_rows=(0, 1, 2, 3, 4),
                prefix="hold_eq",
            ),
            (6,),
        ),
        (
            _label_free(
                question="Which Name has Score above 55 in heldout GT?",
                scores=(10, 20, 30, 40, 45, 50, 65, 75),
                raw_rows=(0, 1, 2, 3, 4),
                prefix="hold_gt",
            ),
            (6, 7),
        ),
        (
            _label_free(
                question="Which Name has Score under 35 in heldout LT?",
                scores=(15, 25, 40, 50, 60, 70, 80, 90),
                raw_rows=(2, 3, 4, 5, 6),
                prefix="hold_lt",
            ),
            (0, 1),
        ),
    )
    for item, gold in cases:
        action = policy.apply_uao_policy(
            formation.policy,
            question=item.question,
            headers=item.headers,
            types=item.types,
            serialized_rows=item.serialized_rows,
            raw_top5=item.raw_top5,
        )
        raw_hits = len(set(item.raw_top5).intersection(gold))
        action_hits = len(set(action).intersection(gold))
        assert action_hits > raw_hits


def test_precomputed_embedding_contract_is_deterministic() -> None:
    rows = tuple(_serialize(f"dense_{index}", score) for index, score in enumerate((1, 2, 3, 4, 5)))
    embeddings = policy.PrecomputedEmbeddings(
        model_sha256="a" * 64,
        question=(1.0, 0.0),
        rows=((1.0, 0.0), (0.8, 0.2), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
    )
    first = policy.LabelFreeItem(
        question="dense fixture",
        headers=HEADERS,
        types=TYPES,
        serialized_rows=rows,
        raw_top5=(0, 1, 2, 3, 4),
        embeddings=embeddings,
    )
    second = policy.LabelFreeItem(
        question="dense fixture",
        headers=HEADERS,
        types=TYPES,
        serialized_rows=rows,
        raw_top5=(0, 1, 2, 3, 4),
        embeddings=embeddings,
    )
    assert first.commitment_sha256 == second.commitment_sha256

    payload = embeddings.content_addressed_payload()
    reconstructed = policy.precomputed_embeddings_from_payload(payload)
    assert reconstructed == embeddings
    assert reconstructed.embeddings_sha256 == payload["self_sha256"]

    tampered = copy.deepcopy(payload)
    tampered["question"][0] = 9.0
    with pytest.raises(policy.WikiSQLUAOPolicyError, match="content hash mismatch"):
        policy.precomputed_embeddings_from_payload(tampered)
    extra = {**payload, "unexpected": True}
    with pytest.raises(policy.WikiSQLUAOPolicyError, match="missing or extra"):
        policy.precomputed_embeddings_from_payload(extra)


def test_compiled_policy_process_boundary_round_trip_and_tamper_rejection(
    formation: policy.PolicyFormation,
) -> None:
    payload = formation.policy.content_addressed_private_payload()
    reconstructed = policy.compiled_policy_from_private_payload(payload)
    assert reconstructed == formation.policy
    assert reconstructed.policy_sha256 == payload["self_sha256"]

    heldout = _label_free(
        question="Which Name has Score greater than 50 at process boundary?",
        scores=(10, 20, 30, 40, 45, 48, 60, 70),
        raw_rows=(0, 1, 2, 3, 4),
        prefix="process",
    )
    arguments = {
        "question": heldout.question,
        "headers": heldout.headers,
        "types": heldout.types,
        "serialized_rows": heldout.serialized_rows,
        "raw_top5": heldout.raw_top5,
    }
    assert policy.apply_uao_policy(reconstructed, **arguments) == policy.apply_uao_policy(
        formation.policy, **arguments
    )

    tampered = copy.deepcopy(payload)
    tampered["model"]["coefficients"][0] += 1.0
    with pytest.raises(policy.WikiSQLUAOPolicyError, match="content hash mismatch"):
        policy.compiled_policy_from_private_payload(tampered)
    extra = {**payload, "answer": "forbidden"}
    with pytest.raises(policy.WikiSQLUAOPolicyError, match="missing or extra"):
        policy.compiled_policy_from_private_payload(extra)

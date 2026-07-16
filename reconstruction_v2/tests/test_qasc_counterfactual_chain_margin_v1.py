from __future__ import annotations

from dataclasses import replace

import pytest

from assumption_agent.benchmarks.qasc_counterfactual_chain_margin_v1 import (
    LABEL_SCHEMA,
    VIEW_SCHEMA,
    QASCCounterfactualError,
    ScoredRecipeItem,
    assign_hmac_folds,
    bridge_tokens,
    build_first_stage_plan,
    build_second_stage_plan,
    consume_stage_scores,
    execute_recipes,
    invalid_scored_item,
    load_label_envelope,
    load_retrieval_view,
    recipe_registry,
    score_recipe_action,
    select_formation_recipes,
    stable_sha256,
    validate_view_label_binding,
)


def _view_row(*, block: str = "A_form") -> dict[str, object]:
    source_member = "DEV" if block == "M_search" else "TRAIN"
    documents = [
        {
            "doc_id": index,
            "text": (
                f"document {index} unique{index} common material"
                if index
                else "the quartz rarealpha common material unique0"
            ),
            "bm25_score_int": 1000 - index,
        }
        for index in range(32)
    ]
    return {
        "schema": VIEW_SCHEMA,
        "block": block,
        "source_member": source_member,
        "formatted_question": "Which statement completes the chain?",
        "choices": [
            {"label": chr(ord("A") + index), "text": f"option{index} matter"}
            for index in range(8)
        ],
        "documents": documents,
        "raw_ranking": [0, 1, 2, 3, 4],
    }


def _label_row(view, *, answer: str = "A", fact1: int = 1, fact2: int = 0):
    return {
        "schema": LABEL_SCHEMA,
        "block": view.block,
        "source_member": view.source_member,
        "identity_commitment_sha256": stable_sha256(
            {"view": view.view_sha256, "answer": answer}
        ),
        "view_sha256": view.view_sha256,
        "answerKey": answer,
        "gold_document_ids": sorted([fact1, fact2]),
        "fact1_document_id": fact1,
        "fact2_document_id": fact2,
    }


def _zero_scorer(pairs):
    return [0] * len(pairs)


def _choice_scorer(pairs):
    scores = []
    for premise, hypothesis in pairs:
        doc_id = int(premise.split()[1]) if premise.startswith("document ") else 0
        choice_bonus = 0
        for choice_index in range(8):
            if f"option{choice_index}" in hypothesis:
                choice_bonus = choice_index * 10000
        scores.append(choice_bonus - doc_id)
    return scores


def _valid_item(
    identity: str,
    recipe_id: str,
    *,
    support: int,
    auc2: int,
    top1: bool,
    gold_pair: bool,
    top5: tuple[int, ...],
) -> ScoredRecipeItem:
    complete = support == 2
    return ScoredRecipeItem(
        identity_commitment_sha256=identity,
        view_sha256=stable_sha256({"view": identity}),
        recipe_id=recipe_id,
        invalid=False,
        support_hits_at_5=support,
        complete=complete,
        U=support + int(complete),
        auc2=auc2,
        top1=top1,
        gold_pair=gold_pair,
        ordered_top5=top5,
        action_sha256=stable_sha256({"action": identity, "recipe": recipe_id}),
    )


def test_label_free_view_schema_and_full_view_hash_are_strict() -> None:
    row = _view_row()
    view = load_retrieval_view(row)
    assert view.view_sha256 == stable_sha256(row)
    assert set(view.to_mapping()) == {
        "schema",
        "block",
        "source_member",
        "formatted_question",
        "choices",
        "documents",
        "raw_ranking",
    }

    leaked = dict(row, identity_commitment_sha256="0" * 64)
    with pytest.raises(QASCCounterfactualError, match="view keys"):
        load_retrieval_view(leaked)
    duplicate_choice = _view_row()
    duplicate_choice["choices"][1]["text"] = duplicate_choice["choices"][0]["text"]
    with pytest.raises(QASCCounterfactualError, match="texts must be unique"):
        load_retrieval_view(duplicate_choice)


def test_label_binding_is_authorized_late_and_preserves_sorted_gold_ids() -> None:
    view = load_retrieval_view(_view_row())
    label = load_label_envelope(_label_row(view, fact1=7, fact2=2))
    assert label.gold_document_ids == (2, 7)
    assert label.fact1_document_id == 7
    validate_view_label_binding(view, label)
    with pytest.raises(QASCCounterfactualError, match="do not bind"):
        validate_view_label_binding(view, replace(label, view_sha256="0" * 64))


def test_recipe_registry_is_exact_cartesian_16_and_rejects_duplicate_subset() -> None:
    recipes = recipe_registry()
    assert len(recipes) == len({recipe.recipe_id for recipe in recipes}) == 16
    assert {
        (recipe.first_query, recipe.bridge_budget, recipe.second_query, recipe.aggregation)
        for recipe in recipes
    } == {
        (first, budget, second, aggregation)
        for first in ("stem", "stem_choice")
        for budget in (2, 4)
        for second in ("choice_bridge", "stem_choice_bridge")
        for aggregation in ("bottleneck_rank", "sum_rank")
    }
    view = load_retrieval_view(_view_row())
    duplicate_ids = [recipe.recipe_id for recipe in recipes] + [recipes[0].recipe_id]
    with pytest.raises(QASCCounterfactualError, match="duplicate"):
        build_first_stage_plan(view, duplicate_ids)


def test_bridge_tokens_use_local_idf_stopwords_and_first_position() -> None:
    view = load_retrieval_view(_view_row())
    assert bridge_tokens(view, 0, 2) == ("quartz", "rarealpha")
    assert "the" not in bridge_tokens(view, 0, 4)


def test_two_stage_plans_deduplicate_pairs_and_preserve_conceptual_compute() -> None:
    view = load_retrieval_view(_view_row())
    recipes = recipe_registry()
    subset = [recipes[0].recipe_id, recipes[1].recipe_id]
    first = build_first_stage_plan(view, subset)
    assert first.conceptual_request_count == 2 * 8 * 32
    assert len(first.requests) == 8 * 32
    assert len(first.pairs) == 32
    assert len(first.pairs_sha256) == 64

    first_scores = [0] * len(first.pairs)
    second = build_second_stage_plan(view, subset, first_scores, first)
    assert second.conceptual_request_count == 2 * 8 * 4 * 31
    assert len(second.requests) == 8 * 4 * 31
    assert len(second.pairs) <= len(second.requests)
    assert len(second.pairs_sha256) == 64


def test_independent_oracle_for_score_tuples_pair_choice_rrf_and_full_counts() -> None:
    view = load_retrieval_view(_view_row())
    recipes = recipe_registry()
    bottleneck = next(
        recipe
        for recipe in recipes
        if recipe.first_query == "stem"
        and recipe.bridge_budget == 2
        and recipe.second_query == "choice_bridge"
        and recipe.aggregation == "bottleneck_rank"
    )
    sum_rank = next(
        recipe
        for recipe in recipes
        if recipe.first_query == "stem"
        and recipe.bridge_budget == 2
        and recipe.second_query == "choice_bridge"
        and recipe.aggregation == "sum_rank"
    )
    subset = [sum_rank.recipe_id, bottleneck.recipe_id]
    first = build_first_stage_plan(view, subset)
    first_scores: list[int | None] = [None] * len(first.pairs)
    for request in first.requests:
        value = 100 - request.doc_id
        old = first_scores[request.pair_index]
        assert old is None or old == value
        first_scores[request.pair_index] = value
    assert all(value is not None for value in first_scores)
    first_vector = [int(value) for value in first_scores]

    second = build_second_stage_plan(view, subset, first_vector, first)
    second_scores: list[int | None] = [None] * len(second.pairs)
    for request in second.requests:
        value = 1000 - request.second_doc_id
        old = second_scores[request.pair_index]
        assert old is None or old == value
        second_scores[request.pair_index] = value
    assert all(value is not None for value in second_scores)
    second_vector = [int(value) for value in second_scores]

    actions = {
        action.recipe_id: action
        for action in consume_stage_scores(
            view,
            first,
            first_vector,
            second,
            second_vector,
            subset,
        )
    }
    sum_action = actions[sum_rank.recipe_id]
    bottleneck_action = actions[bottleneck.recipe_id]
    assert sum_action.choice_paths[0].score == (-2, -1, 1099, 100)
    assert bottleneck_action.choice_paths[0].score == (-1, -2, 100, 1099)
    assert all(path.selected_pair == (0, 1) for path in sum_action.choice_paths)
    assert all(path.selected_pair == (0, 1) for path in bottleneck_action.choice_paths)
    assert sum_action.predicted_choice_label == bottleneck_action.predicted_choice_label == "A"
    assert sum_action.ordered_top5 == bottleneck_action.ordered_top5 == (0, 1, 2, 3, 4)

    full_first = build_first_stage_plan(view)
    assert full_first.conceptual_request_count == 16 * 8 * 32
    assert len(full_first.requests) == 2 * 8 * 32
    assert len(full_first.pairs) == 32 + 8 * 32
    full_first_scores = [0] * len(full_first.pairs)
    full_second = build_second_stage_plan(view, None, full_first_scores, full_first)
    assert full_second.conceptual_request_count == 16 * 8 * 4 * 31
    assert len(full_second.requests) == 2 * 2 * 2 * 8 * 4 * 31
    assert len(full_second.pairs) <= len(full_second.requests)


def test_score_vectors_fail_closed_on_missing_or_noninteger_values() -> None:
    view = load_retrieval_view(_view_row())
    recipe_id = recipe_registry()[0].recipe_id
    first = build_first_stage_plan(view, [recipe_id])
    with pytest.raises(QASCCounterfactualError, match="one integer"):
        build_second_stage_plan(view, [recipe_id], [0] * (len(first.pairs) - 1), first)
    with pytest.raises(QASCCounterfactualError, match="one integer"):
        build_second_stage_plan(view, [recipe_id], [False] * len(first.pairs), first)


def test_all_ties_use_document_pair_and_source_choice_order() -> None:
    view = load_retrieval_view(_view_row())
    action = execute_recipes(view, _zero_scorer, [recipe_registry()[0].recipe_id])[0]
    assert action.predicted_choice_label == "A"
    assert all(path.selected_pair == (0, 1) for path in action.choice_paths)
    assert action.ordered_top5 == (0, 1, 2, 3, 4)

    label = load_label_envelope(_label_row(view, answer="A", fact1=1, fact2=0))
    scored = score_recipe_action(view, action, label)
    assert (scored.auc2, scored.top1, scored.gold_pair) == (7, False, True)
    assert (scored.support_hits_at_5, scored.complete, scored.U) == (2, True, 3)


def test_choice_margins_drive_strict_counterfactual_metrics_without_label_input() -> None:
    view = load_retrieval_view(_view_row())
    action = execute_recipes(view, _choice_scorer, [recipe_registry()[1].recipe_id])[0]
    assert action.predicted_choice_label == "H"
    label = load_label_envelope(_label_row(view, answer="H", fact1=0, fact2=1))
    scored = score_recipe_action(view, action, label)
    assert scored.auc2 == 14
    assert scored.top1 is True
    assert scored.gold_pair is True


def test_action_hash_and_label_binding_tamper_fail_closed() -> None:
    view = load_retrieval_view(_view_row())
    action = execute_recipes(view, _zero_scorer, [recipe_registry()[0].recipe_id])[0]
    label = load_label_envelope(_label_row(view))
    with pytest.raises(QASCCounterfactualError, match="hash"):
        score_recipe_action(view, replace(action, action_sha256="0" * 64), label)


def test_hmac_fold_assignment_is_deterministic_and_exactly_balanced() -> None:
    identities = [stable_sha256({"item": index}) for index in range(64)]
    first = assign_hmac_folds(identities, b"x" * 32, block="A_form")
    second = assign_hmac_folds(list(reversed(identities)), b"x" * 32, block="A_form")
    assert first == second
    assert [sum(fold == index for fold in first.values()) for index in range(4)] == [16] * 4
    assert first != assign_hmac_folds(identities, b"y" * 32, block="A_form")


def test_formation_selects_support_incumbent_and_counterfactual_challenger() -> None:
    recipes = recipe_registry()
    identities = [stable_sha256({"formation": index}) for index in range(64)]
    folds = assign_hmac_folds(identities, b"z" * 32, block="A_form")
    support_winner = recipes[0].recipe_id
    counterfactual_winner = recipes[1].recipe_id
    evidence: dict[str, list[ScoredRecipeItem]] = {}
    for recipe in recipes:
        if recipe.recipe_id == support_winner:
            metrics = dict(support=2, auc2=0, top1=False, gold_pair=False, top5=(0, 1, 2, 3, 4))
        elif recipe.recipe_id == counterfactual_winner:
            metrics = dict(support=1, auc2=14, top1=True, gold_pair=True, top5=(5, 6, 7, 8, 9))
        else:
            metrics = dict(support=0, auc2=0, top1=False, gold_pair=False, top5=(10, 11, 12, 13, 14))
        evidence[recipe.recipe_id] = [
            _valid_item(identity, recipe.recipe_id, **metrics) for identity in identities
        ]
    selection = select_formation_recipes(evidence, folds)
    assert selection.incumbent_recipe_id == support_winner
    assert selection.challenger_recipe_id == counterfactual_winner
    assert selection.same_behavior is False

    evidence[counterfactual_winner] = [
        replace(item, ordered_top5=(0, 1, 2, 3, 4))
        for item in evidence[counterfactual_winner]
    ]
    assert select_formation_recipes(evidence, folds).same_behavior is True


def test_invalid_formation_items_are_zeroed_and_ranked_before_metrics() -> None:
    recipe_id = recipe_registry()[0].recipe_id
    invalid = invalid_scored_item(
        identity_commitment_sha256=stable_sha256({"item": 0}),
        view_sha256=stable_sha256({"view": 0}),
        recipe_id=recipe_id,
    )
    assert invalid.invalid is True and invalid.U == invalid.auc2 == 0
    malformed = replace(invalid, support_hits_at_5=2)

    identities = [stable_sha256({"bad": index}) for index in range(64)]
    folds = assign_hmac_folds(identities, b"b" * 32, block="F_search")
    evidence = {
        recipe.recipe_id: [
            _valid_item(
                identity,
                recipe.recipe_id,
                support=0,
                auc2=0,
                top1=False,
                gold_pair=False,
                top5=(0, 1, 2, 3, 4),
            )
            for identity in identities
        ]
        for recipe in recipe_registry()
    }
    evidence[recipe_id][0] = malformed
    with pytest.raises(QASCCounterfactualError, match="zero metrics"):
        select_formation_recipes(evidence, folds)

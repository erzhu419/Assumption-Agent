from __future__ import annotations

import pytest

from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import entailmentbank_proof_retrieval_core_v1 as core
from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_controller_v1 as controller,
)


def _item(index: int) -> core.LabelFreeItem:
    return core.LabelFreeItem(
        f"{index + 1:064x}",
        f"private question {index}",
        f"private answer {index}",
        f"private hypothesis {index} alpha",
        tuple(f"private fact {index} {ordinal} alpha" for ordinal in range(25)),
    )


def _tensor(index: int) -> core.ItemTensor:
    rows = []
    for ordinal in range(25):
        descending = round((24 - ordinal) * core.INTEGER_SCALE / 24)
        ascending = round(ordinal * core.INTEGER_SCALE / 24)
        rows.append(
            (
                descending,
                ascending,
                ascending,
                descending,
                descending,
                ascending,
                descending,
                (ordinal % 5) * 100_000,
            )
        )
    return core.ItemTensor(
        _item(index).item_commitment_sha256,
        tuple(rows),
        core.build_pair_token_f1(_item(index).node_texts),
    )


def _g_model() -> core.QuantizedRidgeModel:
    tensors = tuple(_tensor(index) for index in range(6))
    labels = tuple(
        core.ItemLabel(_item(index).item_commitment_sha256, "TWO_LEAF", (0, 1))
        for index in range(6)
    )
    return core.fit_g_model(tensors, labels)


def _policies() -> controller.FrozenPolicies:
    g_model = _g_model()
    rows = [tuple(index + offset for offset in range(16)) for index in range(32)]
    targets = [index % 6 for index in range(32)]
    e1 = core.fit_e1_model(rows, targets)
    return controller.FrozenPolicies(
        g_model,
        e1,
        core.RECIPE_REGISTRY[0].recipe_id,
        core.RECIPE_REGISTRY[-1].recipe_id,
        {recipe.recipe_id: 0 for recipe in core.RECIPE_REGISTRY},
        {recipe.recipe_id: 0 for recipe in core.RECIPE_REGISTRY},
    )


def _action(recipe_id: str, commitment: str, ordinals: tuple[int, ...]) -> core.Action:
    body = {
        "schema": f"{core.VERSION}_action",
        "recipe_id": recipe_id,
        "item_commitment_sha256": commitment,
        "selected_ordinals": list(ordinals),
    }
    return core.Action(recipe_id, commitment, ordinals, acquisition.stable_hash(body))


def _measurement_fixture(*, equal_q: bool = False):
    items = tuple(_item(index) for index in range(30))
    labels = []
    actions = {}
    q0_recipe = core.RECIPE_REGISTRY[0].recipe_id
    q1_recipe = core.RECIPE_REGISTRY[-1].recipe_id
    for index, item in enumerate(items):
        if index < 10:
            family, gold = "TWO_LEAF", (10, 11)
        elif index < 20:
            family, gold = "THREE_LEAF", (10, 11, 12)
        else:
            family, gold = "FOUR_FIVE_LEAF", (10, 11, 12, 13)
        labels.append(core.ItemLabel(item.item_commitment_sha256, family, gold))
        q0_ordinals = (20, 21, 22, 23, 24)
        q1_ordinals = q0_ordinals if equal_q else (10, 11, 12, 13, 14)
        actions[item.item_commitment_sha256] = {
            "Q0": _action(q0_recipe, item.item_commitment_sha256, q0_ordinals),
            "Q1": _action(q1_recipe, item.item_commitment_sha256, q1_ordinals),
            "official_HippoRAG": (10, 15, 16, 17, 18),
            "RAW": (0, 1, 2, 3, 4),
        }
    return items, tuple(labels), actions


def test_a_and_f_recipe_matrices_fit_e1_and_freeze_two_global_policies() -> None:
    g_model = _g_model()
    a_items = tuple(_item(100 + index) for index in range(36))
    a_tensors = tuple(_tensor(100 + index) for index in range(36))
    f_items = tuple(_item(200 + index) for index in range(30))
    f_tensors = tuple(_tensor(200 + index) for index in range(30))
    a_actions, a_features = controller.build_action_feature_matrix(
        a_items, a_tensors, g_model
    )
    f_actions, f_features = controller.build_action_feature_matrix(
        f_items, f_tensors, g_model
    )
    a_labels = []
    for index, item in enumerate(a_items):
        family_index = index // 12
        gold = ((0, 1), (0, 1, 2), (0, 1, 2, 3))[family_index]
        a_labels.append(core.ItemLabel(item.item_commitment_sha256, core.FAMILY_ORDER[family_index], gold))
    pack = controller.action_feature_pack(
        block="A_form",
        items=a_items,
        actions=a_actions,
        features=a_features,
        g_model_sha256=g_model.model_sha256,
    )
    assert acquisition.verify_self_hash(pack, "action_feature_pack_sha256")
    e1, totals = controller.fit_e1_from_a_form(
        items=a_items,
        labels=tuple(a_labels),
        actions=a_actions,
        features=a_features,
    )
    policies = controller.freeze_policies(
        g_model=g_model, e1_model=e1, f_features=f_features
    )
    assert set(totals) == set(core.RECIPE_BY_ID)
    assert policies.q0_recipe_id in core.RECIPE_BY_ID
    assert policies.q1_recipe_id in core.RECIPE_BY_ID
    assert len(f_actions) == 30


def test_measurement_submits_and_returns_all_three_arms_with_q0_retained(tmp_path) -> None:
    items = tuple(_item(300 + index) for index in range(30))
    tensors = tuple(_tensor(300 + index) for index in range(30))
    calls = []

    def official(item: core.LabelFreeItem, work_root):
        calls.append((item.item_commitment_sha256, work_root.name))
        return (20, 21, 22, 23, 24)

    policies = _policies()
    actions = controller.execute_measurement_actions(
        block="A_hold",
        items=items,
        tensors=tensors,
        policies=policies,
        official_runner=official,
        official_work_parent=tmp_path / "official",
        official_workers=4,
    )
    assert len(actions) == len(calls) == 30
    assert all(set(row) == {"Q0", "Q1", "official_HippoRAG", "RAW"} for row in actions.values())
    pack = controller.measurement_action_pack(
        block="A_hold", items=items, policies=policies, actions=actions
    )
    assert pack["all_3_times_n_tasks_submitted_before_any_result_join"] is True
    assert acquisition.verify_self_hash(pack, "measurement_action_pack_sha256")


def test_ahold_promotion_and_untouched_m_l5_require_exact_fixed_conditions() -> None:
    items, labels, actions = _measurement_fixture()
    a_hold = controller.score_measurement(
        block="A_hold", items=items, labels=labels, actions=actions
    )
    assert a_hold["evaluator_promoted"] is True
    primary = a_hold["paired_comparisons"]["Q1_minus_Q0"]
    assert primary["net_difference"] > 0
    assert 10 * primary["exact_one_sided_signflip"]["tail_numerator"] <= primary[
        "exact_one_sided_signflip"
    ]["tail_denominator"]
    m_score = controller.score_measurement(
        block="M_search", items=items, labels=labels, actions=actions
    )
    assert m_score["M_search_L5_success"] is True
    public = controller.public_score_result(
        score=m_score,
        action_pack_sha256="a" * 64,
        label_pack_sha256="b" * 64,
    )
    assert "private_item_scores" not in public
    assert public["M_search_L5_success"] is True


def test_nonpromotion_is_terminal_when_q1_equals_q0() -> None:
    items, labels, actions = _measurement_fixture(equal_q=True)
    score = controller.score_measurement(
        block="A_hold", items=items, labels=labels, actions=actions
    )
    assert score["evaluator_promoted"] is False
    assert score["paired_comparisons"]["Q1_minus_Q0"]["net_difference"] == 0
    public = controller.public_score_result(
        score=score,
        action_pack_sha256="a" * 64,
        label_pack_sha256="b" * 64,
    )
    assert public["status"] == "evaluator_not_promoted_terminal_without_M_search"


def test_measurement_score_fails_closed_on_family_or_commitment_drift() -> None:
    items, labels, actions = _measurement_fixture()
    drifted = list(labels)
    drifted[0] = core.ItemLabel(items[0].item_commitment_sha256, "THREE_LEAF", (0, 1, 2))
    with pytest.raises(controller.EntailmentBankControllerError, match="family balance"):
        controller.score_measurement(
            block="A_hold", items=items, labels=tuple(drifted), actions=actions
        )
    reordered = list(labels)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(controller.EntailmentBankControllerError, match="commitment"):
        controller.score_measurement(
            block="A_hold", items=items, labels=tuple(reordered), actions=actions
        )

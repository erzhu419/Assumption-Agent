from __future__ import annotations

from fractions import Fraction
import json

import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_e3_runner_v1 as runner,
)


def _trace(prefix: str, index: int, dense_delta: int) -> runner.DifferenceTrace:
    item = runner.stable_hash([prefix, index])
    r0 = (0, 1, 2, 3, 4)
    r7 = (5, 1, 2, 3, 4)
    features = {name: Fraction(0) for name in runner.FEATURE_ORDER}
    features["outside_RAW5_sentence_count"] = Fraction(1)
    features["dense_relevance_mass_delta"] = Fraction(dense_delta)
    return runner.DifferenceTrace.from_mapping(
        item_commitment_sha256=item,
        sentence_count=10,
        r0_action_trace_sha256=runner.stable_hash(["R0 action", prefix, index]),
        r7_action_trace_sha256=runner.stable_hash(["R7 action", prefix, index]),
        r0_top5=r0,
        r7_top5=r7,
        features=features,
    )


def _matrix(prefix: str, count: int, *, offset: int = 0):
    return tuple(_trace(prefix, index, offset + index) for index in range(count))


def _fit_and_policy() -> tuple[runner.E3FitSeal, runner.PolicySeal]:
    a_rows = _matrix("A", runner.BLOCK_COUNTS["A_form"])
    a_seal = runner.seal_feature_matrix(block="A_form", traces=a_rows)
    utilities = {
        trace.item_commitment_sha256: Fraction(-1 if index < 24 else 1)
        for index, trace in enumerate(a_seal.traces)
    }
    fit = runner.fit_e3(
        feature_seal=a_seal,
        utility_deltas=utilities,
        fold_secret=b"F" * 32,
    )
    f_seal = runner.seal_feature_matrix(
        block="F_search",
        traces=_matrix("F", runner.BLOCK_COUNTS["F_search"], offset=100),
    )
    return fit, runner.freeze_f_policy(feature_seal=f_seal, fit_seal=fit)


def _anchor_inputs(block: str, prefix: str):
    rows = _matrix(prefix, runner.BLOCK_COUNTS[block], offset=200)
    features = runner.seal_feature_matrix(block=block, traces=rows)
    labels = tuple(
        runner.AnchorLabel(
            trace.item_commitment_sha256,
            (5,),
            runner.FAMILIES[index // 10],
        )
        for index, trace in enumerate(features.traces)
    )
    hippo = runner.seal_hippo_retrievals(
        block=block,
        rows=tuple(
            runner.HippoRetrieval(
                trace.item_commitment_sha256, trace.sentence_count, trace.r0_top5
            )
            for trace in reversed(features.traces)
        ),
    )
    return features, labels, hippo


def test_exact_difference_receipt_keeps_action_and_behavior_hashes_independent() -> None:
    rows = _matrix("feature", runner.BLOCK_COUNTS["A_form"])
    seal = runner.seal_feature_matrix(block="A_form", traces=rows)
    receipt = seal.receipt

    assert receipt["trace_count"] == 48
    assert receipt["feature_basis"] == "one_exact_R7_minus_R0_vector_per_item"
    assert receipt["fixed_R7_minus_R0_feature_order"] == list(runner.FEATURE_ORDER)
    assert receipt["action_and_behavior_sha256_equality_required"] is False
    assert receipt["action_trace_matrix_sha256"] != receipt["behavior_matrix_sha256"]
    assert rows[0].r0_action_trace_sha256 != rows[0].r0_behavior_sha256
    assert all(isinstance(value, Fraction) for value in rows[0].features)

    identical_outputs = runner.DifferenceTrace.from_mapping(
        item_commitment_sha256="d" * 64,
        sentence_count=10,
        r0_action_trace_sha256="e" * 64,
        r7_action_trace_sha256="f" * 64,
        r0_top5=(0, 1, 2, 3, 4),
        r7_top5=(0, 1, 2, 3, 4),
        features={name: 0 for name in runner.FEATURE_ORDER},
    )
    assert identical_outputs.r0_action_trace_sha256 != (
        identical_outputs.r7_action_trace_sha256
    )
    assert identical_outputs.r0_behavior_sha256 == identical_outputs.r7_behavior_sha256
    assert identical_outputs.behavior_distinct is False

    with pytest.raises(
        runner.EraserEvidenceInferenceRunnerError,
        match="exact integer or Fraction",
    ):
        runner.DifferenceTrace.from_mapping(
            item_commitment_sha256="a" * 64,
            sentence_count=10,
            r0_action_trace_sha256="b" * 64,
            r7_action_trace_sha256="c" * 64,
            r0_top5=(0, 1, 2, 3, 4),
            r7_top5=(5, 1, 2, 3, 4),
            features={
                name: (0.5 if index == 0 else 0)
                for index, name in enumerate(runner.FEATURE_ORDER)
            },
        )

    source = rows[0]
    with pytest.raises(
        runner.EraserEvidenceInferenceRunnerError,
        match="independently recomputed",
    ):
        runner.DifferenceTrace(
            item_commitment_sha256=source.item_commitment_sha256,
            sentence_count=source.sentence_count,
            r0_action_trace_sha256=source.r0_action_trace_sha256,
            r7_action_trace_sha256=source.r7_action_trace_sha256,
            r0_behavior_sha256="f" * 64,
            r7_behavior_sha256=source.r7_behavior_sha256,
            r0_top5=source.r0_top5,
            r7_top5=source.r7_top5,
            features=source.features,
        )


def test_fit_is_lambda_one_no_intercept_population_scaled_and_crossfit_descriptive() -> None:
    fit, policy = _fit_and_policy()
    receipt = fit.receipt

    assert receipt["observation_count"] == 48
    assert receipt["ridge_lambda"] == "1"
    assert receipt["intercept"] is False
    assert receipt["crossfit_descriptive_only"] is True
    assert [row["held_item_count"] for row in receipt["crossfit"]] == [12] * 4
    assert [row["fit_item_count"] for row in receipt["crossfit"]] == [36] * 4
    assert "fold_secret_sha256" not in receipt
    assert receipt["final_fit_count"] == 1
    assert policy.receipt["item_count"] == 36
    assert policy.receipt["E0_routing"] == "always_R0_DENSE5"
    assert sum(policy.receipt["E3_route_counts"].values()) == 36
    assert policy.receipt["E3_route_counts"][runner.RECIPE_IDS[1]] == 36


def test_flattened_union_utility_does_not_best_group_maximize() -> None:
    assert runner.item_utility((0, 1, 2, 3, 4), (0, 4)) == (Fraction(2), True)
    assert runner.item_utility((0, 1, 2, 3, 4), (0, 4, 8)) == (
        Fraction(2, 3),
        False,
    )
    assert runner.item_utility((0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 8)) == (
        Fraction(5, 6),
        False,
    )


def test_a_hold_promotion_primary_family_raw_and_hippo_decisions() -> None:
    _fit, policy = _fit_and_policy()
    features, labels, _hippo = _anchor_inputs("A_hold", "AH")
    # Give HippoRAG one complete item in each relation family so that the
    # three pairwise aggregate families are observably different.
    hippo = runner.seal_hippo_retrievals(
        block="A_hold",
        rows=tuple(
            runner.HippoRetrieval(
                trace.item_commitment_sha256,
                trace.sentence_count,
                trace.r7_top5 if index in {0, 10, 20} else trace.r0_top5,
            )
            for index, trace in enumerate(features.traces)
        ),
    )
    score = runner.score_anchor(
        block="A_hold",
        labels=tuple(reversed(labels)),
        anchor_feature_seal=features,
        hippo_retrieval_seal=hippo,
        policy_seal=policy,
    )
    receipt = score.receipt

    assert receipt["logical_RAW_HippoRAG_Agent_work_units"] == 90
    assert receipt["E3_minus_E0"]["observed_net_U"] == {
        "numerator": 60,
        "denominator": 1,
    }
    assert receipt["E3_minus_E0"]["p_value"] == {
        "numerator": 1,
        "denominator": 1 << 30,
    }
    assert receipt["evaluator_promoted"] is True
    assert receipt["A_hold_real_domain_primary_passed"] is True
    assert receipt["Hippo_cross_relation_passed"] is True
    assert receipt["RAW_block_passed"] is True
    assert receipt["RAW_advantage_overcome"] is None
    assert receipt["complete_counts"] == {
        "E0": 0,
        "E3": 30,
        "HippoRAG": 3,
        "RAW": 0,
    }
    assert receipt["family_item_counts"] == {
        family: 10 for family in runner.FAMILIES
    }
    assert receipt["pairwise_total_U"] == {
        "E3_minus_E0": [60, 1],
        "E3_minus_HippoRAG": [54, 1],
        "E3_minus_RAW": [60, 1],
    }
    assert receipt["pairwise_family_sums"] == {
        "E3_minus_E0": {family: [20, 1] for family in runner.FAMILIES},
        "E3_minus_HippoRAG": {family: [18, 1] for family in runner.FAMILIES},
        "E3_minus_RAW": {family: [20, 1] for family in runner.FAMILIES},
    }
    assert receipt["complete_counts_by_family"] == {
        "E0": {family: 0 for family in runner.FAMILIES},
        "E3": {family: 10 for family in runner.FAMILIES},
        "HippoRAG": {family: 1 for family in runner.FAMILIES},
        "RAW": {family: 0 for family in runner.FAMILIES},
    }
    assert receipt["pairwise_complete_count_deltas"] == {
        "E3_minus_E0": 30,
        "E3_minus_HippoRAG": 27,
        "E3_minus_RAW": 30,
    }
    assert receipt["pairwise_complete_count_deltas_by_family"] == {
        "E3_minus_E0": {family: 10 for family in runner.FAMILIES},
        "E3_minus_HippoRAG": {family: 9 for family in runner.FAMILIES},
        "E3_minus_RAW": {family: 10 for family in runner.FAMILIES},
    }
    assert receipt["R7_action_aggregates"] == {
        "R7_candidate_item_count": 30,
        "R7_candidate_behavior_distinct_count": 30,
        "R7_candidate_outside_RAW5_sentence_count_sum": [30, 1],
        "R7_candidate_edge_deletion_action_change_count": 0,
        "E3_activated_R7_item_count": 30,
        "E3_activated_behavior_distinct_R7_count": 30,
        "E3_activated_outside_RAW5_sentence_count_sum": [30, 1],
        "E3_activated_edge_deletion_action_change_count": 0,
    }

    tampered = dict(receipt)
    tampered["evaluator_promoted"] = False
    tampered.pop("score_receipt_sha256")
    tampered["score_receipt_sha256"] = runner.stable_hash(tampered)
    with pytest.raises(
        runner.EraserEvidenceInferenceRunnerError,
        match="derived decision semantics drifted",
    ):
        runner.AnchorScoreSeal(
            block="A_hold",
            anchor_features=features,
            hippo_retrievals=hippo,
            policies=policy,
            a_hold_authorization=None,
            receipt_json=json.dumps(
                tampered,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    tampered_total = json.loads(json.dumps(receipt))
    tampered_total["pairwise_total_U"]["E3_minus_HippoRAG"] = [55, 1]
    tampered_total.pop("score_receipt_sha256")
    tampered_total["score_receipt_sha256"] = runner.stable_hash(tampered_total)
    with pytest.raises(
        runner.EraserEvidenceInferenceRunnerError,
        match="disagree with sign-flip nets",
    ):
        runner.AnchorScoreSeal(
            block="A_hold",
            anchor_features=features,
            hippo_retrievals=hippo,
            policies=policy,
            a_hold_authorization=None,
            receipt_json=json.dumps(
                tampered_total,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )


def test_null_pairwise_result_is_reported_without_promotion_or_primary() -> None:
    _fit, policy = _fit_and_policy()
    features, _labels, hippo = _anchor_inputs("A_hold", "AH_NULL")
    labels = tuple(
        runner.AnchorLabel(
            trace.item_commitment_sha256,
            (1,),
            runner.FAMILIES[index // 10],
        )
        for index, trace in enumerate(features.traces)
    )
    score = runner.score_anchor(
        block="A_hold",
        labels=labels,
        anchor_feature_seal=features,
        hippo_retrieval_seal=hippo,
        policy_seal=policy,
    )
    receipt = score.receipt

    assert receipt["behavior_distinct_R7_route_count"] == 30
    assert receipt["pairwise_total_U"] == {
        comparison: [0, 1] for comparison in runner.PAIRWISE_COMPARISONS
    }
    assert receipt["pairwise_family_sums"] == {
        comparison: {family: [0, 1] for family in runner.FAMILIES}
        for comparison in runner.PAIRWISE_COMPARISONS
    }
    assert receipt["complete_counts"] == {
        "E0": 30,
        "E3": 30,
        "HippoRAG": 30,
        "RAW": 30,
    }
    assert receipt["pairwise_complete_count_deltas"] == {
        comparison: 0 for comparison in runner.PAIRWISE_COMPARISONS
    }
    assert receipt["E3_minus_E0"]["p_value"] == {
        "numerator": 1,
        "denominator": 1,
    }
    assert receipt["evaluator_promoted"] is False
    assert receipt["A_hold_real_domain_primary_passed"] is False
    assert receipt["RAW_block_passed"] is False


def test_m_requires_promoted_a_hold_and_reports_l5_combined_claims() -> None:
    _fit, policy = _fit_and_policy()
    a_features, a_labels, a_hippo = _anchor_inputs("A_hold", "AH2")
    a_score = runner.score_anchor(
        block="A_hold",
        labels=a_labels,
        anchor_feature_seal=a_features,
        hippo_retrieval_seal=a_hippo,
        policy_seal=policy,
    )
    m_features, m_labels, m_hippo = _anchor_inputs("M_search", "M")

    with pytest.raises(
        runner.EraserEvidenceInferenceRunnerError, match="not authorized"
    ):
        runner.score_anchor(
            block="M_search",
            labels=m_labels,
            anchor_feature_seal=m_features,
            hippo_retrieval_seal=m_hippo,
            policy_seal=policy,
        )

    m_score = runner.score_anchor(
        block="M_search",
        labels=m_labels,
        anchor_feature_seal=m_features,
        hippo_retrieval_seal=m_hippo,
        policy_seal=policy,
        a_hold_authorization=a_score,
    )
    receipt = m_score.receipt
    assert receipt["evaluator_promoted"] is None
    assert receipt["M_L5_passed"] is True
    assert receipt["M_L5_passed"] is receipt["E3_minus_E0"]["promoted"]
    assert receipt["Hippo_cross_relation_passed"] is True
    assert receipt["cross_relation_stability_passed"] is True
    assert receipt["RAW_block_passed"] is True
    assert receipt["RAW_advantage_overcome"] is True
    assert (
        receipt["A_hold_authorization_score_receipt_sha256"]
        == a_score.score_receipt_sha256
    )

    tampered = dict(receipt)
    tampered["M_L5_passed"] = False
    tampered.pop("score_receipt_sha256")
    tampered["score_receipt_sha256"] = runner.stable_hash(tampered)
    with pytest.raises(
        runner.EraserEvidenceInferenceRunnerError,
        match="derived decision semantics drifted",
    ):
        runner.AnchorScoreSeal(
            block="M_search",
            anchor_features=m_features,
            hippo_retrievals=m_hippo,
            policies=policy,
            a_hold_authorization=a_score,
            receipt_json=json.dumps(
                tampered,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

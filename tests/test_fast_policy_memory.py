import unittest

from assumption_os.fast_policy_memory import (
    FAST_POLICY_MEMORY_VERSION,
    FastPolicyHypothesis,
    PromotionGateSpec,
    evaluate_fast_policy_promotion,
    select_fast_policies,
)


class FastPolicyMemoryTests(unittest.TestCase):
    def test_promoted_policy_selects_without_persisting_raw_content(self):
        policy = FastPolicyHypothesis(
            id="source_pair_binding_science_v1",
            kind="source_binding",
            action="source_pair_binding_lane",
            trigger_terms=["source", "relation", "option", "witness"],
            anti_trigger_terms=["self-contained"],
            expected_utility=0.7,
            expected_harm=0.1,
            promotion_status="promoted",
            evidence_rows=[{"run_id": "unseen24", "delta": 2}],
            fallback_behavior="preserve_slow_baseline",
        )

        decision = select_fast_policies(
            [policy],
            problem_text="Which option is supported by a source relation witness?",
        )

        self.assertEqual(decision["policy_version"], FAST_POLICY_MEMORY_VERSION)
        self.assertEqual(decision["selected_policy_ids"], ["source_pair_binding_science_v1"])
        self.assertEqual(decision["selected_policy_kinds"], ["source_binding"])
        self.assertTrue(decision["slow_baseline_required"])
        self.assertFalse(decision["raw_content_persisted"])
        self.assertTrue(decision["question_hash"])
        self.assertNotIn("Which option", str(decision["score_rows"]))

    def test_candidate_policy_is_shadow_only_until_promoted(self):
        policy = FastPolicyHypothesis(
            id="chem_solver_shadow_v1",
            kind="solver_lane",
            action="self_contained_solver_lane",
            trigger_terms=["chemistry", "probe", "alkyne"],
            expected_utility=0.8,
            expected_harm=0.1,
            promotion_status="candidate",
        )

        decision = select_fast_policies(
            [policy],
            problem_text="A chemistry probe contains an alkyne handle.",
        )

        self.assertEqual(decision["selected_policy_ids"], [])
        self.assertEqual(decision["score_rows"][0]["reason"], "status_not_allowed_for_decision")

    def test_anti_trigger_blocks_otherwise_matching_policy(self):
        policy = FastPolicyHypothesis(
            id="retrieval_source_lane_v1",
            kind="source_binding",
            action="source_pair_binding_lane",
            trigger_terms=["source", "evidence"],
            anti_trigger_terms=["self-contained", "calculation"],
            expected_utility=0.7,
            expected_harm=0.2,
            promotion_status="promoted",
        )

        decision = select_fast_policies(
            [policy],
            problem_text="This self-contained calculation mentions source evidence only in passing.",
        )

        self.assertEqual(decision["selected_policy_ids"], [])
        self.assertEqual(decision["score_rows"][0]["reason"], "anti_trigger_or_harm_blocks")

    def test_promotion_gate_requires_unseen_gain_or_stable_noninferiority(self):
        result = evaluate_fast_policy_promotion(
            candidate_metrics={
                "fixed_regression_accuracy": 0.5,
                "unseen_correct": 8,
                "unseen_total": 24,
                "selected_label_stability": 0.97,
                "no_fallback_count": 0,
                "unique_model_calls": 110,
                "failure_buckets": {"candidate_generation_missed_gold": 3},
            },
            baseline_metrics={
                "unseen_correct": 6,
                "unseen_total": 24,
                "unique_model_calls": 100,
                "failure_buckets": {"candidate_generation_missed_gold": 5},
            },
        )

        self.assertTrue(result["promotion_allowed"])
        self.assertEqual(result["decision"], "promote_fast_policy")
        self.assertEqual(result["metrics"]["unseen_correct_gain"], 2)
        self.assertLessEqual(
            result["metrics"]["failure_bucket_delta"]["candidate_generation_missed_gold"],
            0,
        )

    def test_promotion_gate_blocks_no_fallback_regression(self):
        result = evaluate_fast_policy_promotion(
            candidate_metrics={
                "fixed_regression_accuracy": 0.5,
                "unseen_correct": 8,
                "unseen_total": 24,
                "selected_label_stability": 0.98,
                "no_fallback_count": 1,
                "unique_model_calls": 110,
            },
            baseline_metrics={
                "unseen_correct": 6,
                "unseen_total": 24,
                "unique_model_calls": 100,
            },
            gate=PromotionGateSpec(min_unseen_correct_gain=2),
        )

        self.assertFalse(result["promotion_allowed"])
        self.assertIn("no_fallback_count_zero", result["failed_gates"])

    def test_noninferior_stable_latency_win_can_promote(self):
        result = evaluate_fast_policy_promotion(
            candidate_metrics={
                "fixed_regression_accuracy": 0.5,
                "unseen_correct": 6,
                "unseen_total": 24,
                "selected_label_stability": 0.99,
                "no_fallback_count": 0,
                "cost": 80,
            },
            baseline_metrics={
                "unseen_correct": 6,
                "unseen_total": 24,
                "cost": 100,
            },
        )

        self.assertTrue(result["promotion_allowed"])
        self.assertEqual(result["metrics"]["cost_ratio"], 0.8)


if __name__ == "__main__":
    unittest.main()

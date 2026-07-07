import unittest

from assumption_os.autonomy_journal import stable_hash
from assumption_os.fast_policy_memory import FastPolicyHypothesis, select_fast_policies
from assumption_os.hle_lane_router import route_option_lanes
from assumption_os.hle_option_matrix import build_option_matrix
from assumption_os.hle_option_span_bundle import build_option_span_bundles
from assumption_os.hle_pair_binding_comparator import (
    adjudicate_pair_binding,
    candidate_span_bundle_source_lane,
    source_verifier_audit_source_lane,
)
from assumption_os.hle_self_contained_operator_matrix import build_self_contained_operator_matrix


class HleOptionMatrixRouterTests(unittest.TestCase):
    def test_chem_probe_click_matrix_scores_unique_alkyne_option(self):
        matrix = build_self_contained_operator_matrix(
            category="Biology/Medicine",
            raw_subject="Biochemistry",
            stem=(
                "An alkyne probe was added to lysate, irradiated with 417 nm light, "
                "clicked with cy5-azide, and run on an SDS Page gel. The second probe "
                "changes the fluorescent signal. What is the molecule that causes the "
                "fluorescent difference?"
            ),
            options={
                "A": "2-fluoro-7-methoxy-9H-thioxanthen-9-one",
                "B": "phenoxyl radical",
                "C": "methyl (E)-4-oxo-4-(prop-2-yn-1-ylamino)but-2-enoate",
                "D": "carbene",
                "E": "cy5 azide",
            },
        )

        self.assertEqual(matrix["status"], "activated")
        self.assertEqual(matrix["selected_label"], "C")
        self.assertEqual(matrix["operator_family"], "chem_probe_click_matrix")
        rows = {row["label"]: row for row in matrix["option_rows"]}
        self.assertTrue(rows["C"]["is_alkyne_like"])
        self.assertFalse(rows["C"]["is_reagent_not_probe_handle"])
        self.assertTrue(rows["E"]["is_azide_dye"])
        self.assertTrue(matrix["matrix_hash"])
        self.assertFalse(matrix["raw_content_persisted"])

    def test_chem_probe_click_matrix_abstains_on_tied_alkyne_options(self):
        matrix = build_self_contained_operator_matrix(
            category="Chemistry",
            raw_subject="Chemical biology",
            stem=(
                "An alkyne probe was irradiated with light, clicked with cy5-azide, "
                "and measured by fluorescence on an SDS gel. A second probe changes "
                "the signal. Which molecule causes the fluorescent difference?"
            ),
            options={
                "A": "propargyl amide",
                "B": "propargyl ester",
                "C": "cy5 azide",
                "D": "carbene",
            },
        )

        self.assertEqual(matrix["status"], "abstained")
        self.assertIsNone(matrix["selected_label"])

    def test_antibiotic_susceptibility_matrix_selects_unique_all_s_option(self):
        matrix = build_self_contained_operator_matrix(
            category="Biology/Medicine",
            raw_subject="Clinical Medicine",
            stem=(
                "A patient has a severe skin infection. Culture results show: "
                "Amoxicillin - R; Ciprofloxacin - R; Clindamycin - S; "
                "Linezolid - S; Vancomycin - S; Tetracycline - I. "
                "Which antibiotics are reasonable treatment options?"
            ),
            options={
                "A": "Amoxicillin, Ciprofloxacin",
                "B": "Clindamycin, Amoxicillin",
                "C": "Vancomycin, Linezolid, Clindamycin",
                "D": "Vancomycin, Linezolid, Tetracycline",
            },
        )

        self.assertEqual(matrix["status"], "activated")
        self.assertEqual(matrix["selected_label"], "C")
        self.assertEqual(matrix["operator_family"], "antibiotic_susceptibility_profile")
        rows = {row["label"]: row for row in matrix["option_rows"]}
        self.assertTrue(rows["C"]["all_items_susceptible"])
        self.assertTrue(rows["D"]["has_resistant_or_intermediate_item"])
        self.assertFalse(matrix["raw_content_persisted"])

    def test_antibiotic_susceptibility_matrix_abstains_on_multiple_all_s_options(self):
        matrix = build_self_contained_operator_matrix(
            category="Medicine",
            raw_subject="Clinical Medicine",
            stem=(
                "A patient has an infection. Culture results show: "
                "Drug A - S; Drug B - S; Drug C - S; Drug D - R. "
                "Which antibiotics are reasonable treatment options?"
            ),
            options={
                "A": "Drug A, Drug B",
                "B": "Drug B, Drug C",
                "C": "Drug A, Drug D",
            },
        )

        self.assertEqual(matrix["status"], "abstained")
        self.assertEqual(matrix["operator_family"], "antibiotic_susceptibility_profile")
        self.assertIsNone(matrix["selected_label"])

    def test_antibiotic_susceptibility_matrix_uses_clinda_erythro_inducible_resistance_guard(self):
        matrix = build_self_contained_operator_matrix(
            category="Biology/Medicine",
            raw_subject="Clinical Medicine",
            stem=(
                "A patient has a severe skin infection. Culture results show: "
                "Clindamycin - S; Erythromycin - R; Linezolid - S; "
                "Vancomycin - S; Trimethoprim/Sulfamethoxazole - S. "
                "Which antibiotics are reasonable treatment options?"
            ),
            options={
                "A": "Clindamycin, Vancomycin, Linezolid",
                "B": "Erythromycin, Trimethoprim/Sulfamethoxazole, Linezolid",
                "C": "Vancomycin, Linezolid, Trimethoprim/Sulfamethoxazole",
            },
        )

        self.assertEqual(matrix["status"], "activated")
        self.assertEqual(matrix["selected_label"], "C")
        rows = {row["label"]: row for row in matrix["option_rows"]}
        self.assertTrue(rows["A"]["has_conditional_resistance_item"])
        self.assertFalse(rows["A"]["all_items_susceptible"])

    def test_antibiotic_susceptibility_matrix_abstains_when_guard_removes_only_candidate(self):
        matrix = build_self_contained_operator_matrix(
            category="Biology/Medicine",
            raw_subject="Clinical Medicine",
            stem=(
                "A patient has a severe skin infection. Culture results show: "
                "Amoxicillin - R; Ciprofloxacin - R; Clindamycin - S; "
                "Erythromycin - R; Linezolid - S; Vancomycin - S; "
                "Tetracycline - I. Which antibiotics are reasonable treatment options?"
            ),
            options={
                "A": "Amoxicillin, Ciprofloxacin",
                "B": "Clindamycin, Amoxicillin",
                "C": "Vancomycin, Linezolid, Clindamycin",
                "D": "Vancomycin, Linezolid, Tetracycline",
            },
        )

        self.assertEqual(matrix["status"], "abstained")
        self.assertEqual(matrix["operator_family"], "antibiotic_susceptibility_profile")
        self.assertIsNone(matrix["selected_label"])

    def test_span_bundle_and_pair_binding_accept_direct_positive_margin(self):
        bundles = build_option_span_bundles(
            question="Which molecule causes the fluorescent signal after azide click labeling?",
            options={"A": "phenoxyl radical", "B": "propargyl probe"},
            source_records_by_option={
                "A": [{"source_id": "a1", "snippet": "Phenoxyl radical is a transient species."}],
                "B": [{
                    "source_id": "b1",
                    "snippet": (
                        "The propargyl probe contains an alkyne handle and causes the "
                        "fluorescent signal after azide click labeling."
                    ),
                }],
            },
            relation_terms=["causes", "fluorescent", "signal", "azide", "click"],
            required_terms=["fluorescent", "azide"],
        )

        source_lane = adjudicate_pair_binding(span_bundles_by_option=bundles)

        self.assertEqual(source_lane["status"], "activated")
        self.assertEqual(source_lane["selected_label"], "B")
        self.assertTrue(source_lane["pair_binding_accept"])

    def test_lane_router_uses_solver_unique_override(self):
        route = route_option_lanes(
            source_lane={"status": "no_candidate", "pair_binding_verdict": "reject_generic"},
            solver_lane={
                "status": "activated",
                "selected_label": "C",
                "confidence_score": 0.86,
                "unique_margin": 3,
            },
            baseline_lane={"selected_label": "A"},
        )

        self.assertEqual(route["selected_label"], "C")
        self.assertEqual(route["selection_method"], "self_contained_solver_override")
        self.assertTrue(route["router_payload_hash"])

    def test_lane_router_falls_back_when_source_generic_and_solver_weak(self):
        route = route_option_lanes(
            source_lane={"status": "generic", "pair_binding_verdict": "reject_generic"},
            solver_lane={"status": "not_required"},
            baseline_lane={"hippo_label": "D", "raw_label": "B"},
        )

        self.assertEqual(route["selected_label"], "D")
        self.assertEqual(route["selection_method"], "hipporag_fallback")
        self.assertNotEqual(route["reason"], "no_fallback")
        self.assertTrue(route["slow_baseline_required"])

    def test_lane_router_records_fast_policy_memory_trace(self):
        fast_policy = select_fast_policies(
            [
                FastPolicyHypothesis(
                    id="solver_lane_promoted_v1",
                    kind="solver_lane",
                    action="self_contained_solver_lane",
                    trigger_terms=["chemistry", "probe"],
                    expected_utility=0.8,
                    expected_harm=0.1,
                    promotion_status="promoted",
                )
            ],
            problem_text="A chemistry probe question.",
        )

        route = route_option_lanes(
            source_lane={"status": "generic", "pair_binding_verdict": "reject_generic"},
            solver_lane={"status": "not_required"},
            baseline_lane={"raw_label": "B"},
            fast_policy_decision=fast_policy,
        )

        self.assertEqual(route["selected_label"], "B")
        self.assertEqual(
            route["fast_policy_memory"]["selected_policy_ids"],
            ["solver_lane_promoted_v1"],
        )
        self.assertEqual(
            route["fast_policy_memory"]["selected_actions"],
            ["self_contained_solver_lane"],
        )
        self.assertFalse(route["fast_policy_memory"]["raw_content_persisted"])

    def test_lane_router_prefers_pair_bound_source(self):
        route = route_option_lanes(
            source_lane={
                "status": "activated",
                "selected_label": "B",
                "pair_binding_accept": True,
                "pair_binding_verdict": "accept_direct",
                "best_direct_margin": 0.22,
                "confidence_score": 0.82,
            },
            solver_lane={
                "status": "activated",
                "selected_label": "C",
                "confidence_score": 0.86,
                "unique_margin": 2,
            },
            baseline_lane={"selected_label": "A"},
        )

        self.assertEqual(route["selected_label"], "B")
        self.assertEqual(route["selection_method"], "source_direct_override")

    def test_option_matrix_hash_is_stable(self):
        kwargs = {
            "question": (
                "An alkyne probe was irradiated with light, clicked with cy5 azide, "
                "and measured by fluorescence on an SDS gel. The second probe changes "
                "the signal. Which molecule causes the fluorescent signal after azide "
                "click labeling?"
            ),
            "category": "Chemistry",
            "raw_subject": "Chemical biology",
            "options": {
                "A": "phenoxyl radical",
                "B": "methyl (E)-4-oxo-4-(prop-2-yn-1-ylamino)but-2-enoate",
                "C": "cy5 azide",
                "D": "carbene",
            },
            "source_records_by_option": {
                "B": [{
                    "source_id": "b1",
                    "snippet": "The prop-2-yn probe causes fluorescent signal after azide click labeling.",
                }],
            },
            "baseline_predictions": {"raw": "A", "hippo": "B"},
        }

        first = build_option_matrix(**kwargs)
        second = build_option_matrix(**kwargs)

        self.assertEqual(first["option_matrix_hash"], second["option_matrix_hash"])
        self.assertEqual(first["self_contained_lane"]["selected_label"], "B")

    def test_source_verifier_audit_source_lane_accepts_strict_candidate_span(self):
        lane = source_verifier_audit_source_lane(
            label_by_hash={"hash-b": "B", "hash-c": "C"},
            candidate_summaries=[
                {
                    "label": "B",
                    "option_hash": "hash-b",
                    "candidate_direct_relation_span_count": 2,
                    "candidate_direct_relation_span_top_relation_signature_required_overlap": 2,
                    "candidate_direct_relation_span_top_relation_signature_missing_term_count": 0,
                    "candidate_direct_relation_span_top_relation_proximity": True,
                    "candidate_direct_relation_span_top_relation_signature_proximity": True,
                    "source_cache_answer_bearing_focused_retry_strict_direct_support_doc_count": 2,
                    "source_cache_answer_bearing_focused_retry_candidate_specific_span_doc_count": 4,
                    "source_cache_answer_bearing_focused_retry_directish_doc_count": 4,
                    "source_cache_answer_bearing_focused_retry_required_overlap_doc_count": 5,
                    "source_cache_answer_bearing_focused_retry_relation_proximity_doc_count": 5,
                    "source_quality_score": 10.0,
                    "source_quality_doc_count": 2,
                    "support_doc_count": 1,
                    "refute_doc_count": 0,
                    "ambiguous_doc_count": 0,
                },
                {
                    "label": "C",
                    "option_hash": "hash-c",
                    "candidate_direct_relation_span_count": 1,
                    "candidate_direct_relation_span_top_relation_signature_required_overlap": 1,
                    "candidate_direct_relation_span_top_relation_signature_missing_term_count": 2,
                    "candidate_direct_relation_span_top_relation_proximity": True,
                    "candidate_direct_relation_span_top_relation_signature_proximity": False,
                    "source_quality_score": 4.0,
                    "source_quality_doc_count": 1,
                    "refute_doc_count": 0,
                    "ambiguous_doc_count": 0,
                },
            ],
        )

        self.assertEqual(lane["status"], "activated")
        self.assertEqual(lane["selected_label"], "B")
        self.assertEqual(lane["pair_binding_verdict"], "accept_direct")
        self.assertTrue(lane["source_lane_payload_hash"])

    def test_source_verifier_audit_source_lane_rejects_shared_or_refuted_span(self):
        lane = source_verifier_audit_source_lane(
            label_by_hash={"hash-b": "B"},
            candidate_summaries=[
                {
                    "label": "B",
                    "option_hash": "hash-b",
                    "candidate_direct_relation_span_count": 2,
                    "candidate_direct_relation_span_top_relation_signature_required_overlap": 2,
                    "candidate_direct_relation_span_top_relation_signature_missing_term_count": 0,
                    "candidate_direct_relation_span_top_relation_proximity": True,
                    "candidate_direct_relation_span_top_relation_signature_proximity": True,
                    "candidate_direct_relation_span_top_shared_doc": True,
                    "source_cache_answer_bearing_focused_retry_strict_direct_support_doc_count": 3,
                    "source_cache_answer_bearing_focused_retry_candidate_specific_span_doc_count": 5,
                    "source_cache_answer_bearing_focused_retry_directish_doc_count": 5,
                    "source_cache_answer_bearing_focused_retry_required_overlap_doc_count": 5,
                    "source_cache_answer_bearing_focused_retry_relation_proximity_doc_count": 5,
                    "source_quality_score": 10.0,
                    "source_quality_doc_count": 2,
                    "refute_doc_count": 1,
                    "ambiguous_doc_count": 0,
                },
            ],
        )

        self.assertNotEqual(lane["status"], "activated")
        self.assertFalse(lane["pair_binding_accept"])

    def test_candidate_span_bundle_source_lane_accepts_strict_direct_bundle(self):
        hash_b = stable_hash({"option_label": "B"})
        hash_c = stable_hash({"option_label": "C"})
        lane = candidate_span_bundle_source_lane(
            label_by_hash={hash_b: "B", hash_c: "C"},
            candidate_span_bundle_detail={
                "status": "activated",
                "reason": "candidate_span_bundles_built",
                "bundle_hash": "bundle-hash",
                "rows_hash": "rows-hash",
                "selected_option_hash": hash_b,
                "direct_source_margin": 6.0,
                "recommendation_reason": "strong_direct_source_quality_margin",
                "option_with_direct_witness_count": 1,
                "direct_witness_count": 1,
                "option_bundles": [
                    {
                        "option_hash": hash_b,
                        "direct_witness_count": 1,
                        "top_witnesses": [
                            {
                                "witness_id": "wit-b",
                                "source_doc_hash": "doc-b",
                                "witness_type": "direct_relation",
                                "required_overlap": 2,
                                "required_missing_count": 0,
                                "option_overlap": 1,
                                "relation_overlap": 1,
                                "slot_coverage": 1,
                                "relation_proximity": True,
                                "candidate_specific": True,
                                "strict_answer_bearing": True,
                                "source_cache_answer_bearing": True,
                                "shared_or_other": False,
                                "refutation": False,
                                "source_quality_score": 12.0,
                            }
                        ],
                    },
                    {
                        "option_hash": hash_c,
                        "direct_witness_count": 0,
                        "top_witnesses": [
                            {
                                "witness_id": "wit-c",
                                "source_doc_hash": "doc-c",
                                "witness_type": "relation_only",
                                "required_overlap": 1,
                                "required_missing_count": 1,
                                "option_overlap": 0,
                                "relation_overlap": 1,
                                "slot_coverage": 0,
                                "relation_proximity": True,
                                "candidate_specific": False,
                                "strict_answer_bearing": False,
                                "shared_or_other": False,
                                "refutation": False,
                                "source_quality_score": 4.0,
                            }
                        ],
                    },
                ],
            },
        )

        self.assertEqual(lane["status"], "activated")
        self.assertEqual(lane["selected_label"], "B")
        self.assertEqual(lane["pair_binding_verdict"], "accept_direct")
        self.assertEqual(
            lane["policy"],
            "candidate_span_bundle_option_matrix_source_lane_v1",
        )
        self.assertTrue(lane["source_lane_payload_hash"])

    def test_candidate_span_bundle_source_lane_rejects_shared_refuted_or_incomplete(self):
        hash_b = stable_hash({"option_label": "B"})
        lane = candidate_span_bundle_source_lane(
            label_by_hash={hash_b: "B"},
            candidate_span_bundle_detail={
                "status": "activated",
                "bundle_hash": "bundle-hash",
                "rows_hash": "rows-hash",
                "selected_option_hash": hash_b,
                "option_bundles": [
                    {
                        "option_hash": hash_b,
                        "direct_witness_count": 1,
                        "top_witnesses": [
                            {
                                "witness_id": "shared",
                                "witness_type": "direct_relation",
                                "required_overlap": 2,
                                "required_missing_count": 0,
                                "option_overlap": 1,
                                "relation_overlap": 1,
                                "candidate_specific": True,
                                "strict_answer_bearing": True,
                                "shared_or_other": True,
                                "refutation": False,
                                "source_quality_score": 12.0,
                            },
                            {
                                "witness_id": "refuted",
                                "witness_type": "direct_relation",
                                "required_overlap": 2,
                                "required_missing_count": 0,
                                "option_overlap": 1,
                                "relation_overlap": 1,
                                "candidate_specific": True,
                                "strict_answer_bearing": True,
                                "shared_or_other": False,
                                "refutation": True,
                                "source_quality_score": 12.0,
                            },
                            {
                                "witness_id": "incomplete",
                                "witness_type": "direct_relation",
                                "required_overlap": 1,
                                "required_missing_count": 1,
                                "option_overlap": 1,
                                "relation_overlap": 1,
                                "candidate_specific": True,
                                "strict_answer_bearing": True,
                                "shared_or_other": False,
                                "refutation": False,
                                "source_quality_score": 12.0,
                            },
                        ],
                    }
                ],
            },
        )

        self.assertNotEqual(lane["status"], "activated")
        self.assertFalse(lane["pair_binding_accept"])


if __name__ == "__main__":
    unittest.main()

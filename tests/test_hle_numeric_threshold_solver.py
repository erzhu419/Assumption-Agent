import unittest

from assumption_os.hle_numeric_option_parser import parse_numeric_options, parse_numeric_values
from assumption_os.hle_numeric_relation_classifier import classify_numeric_relation
from assumption_os.hle_numeric_source_witness import numeric_same_row_directness_detail
from assumption_os.hle_numeric_threshold_solver import solve_numeric_threshold_lane


class HleNumericThresholdSolverTests(unittest.TestCase):
    def test_parse_temperature_option_normalizes_negative_celsius(self):
        values = parse_numeric_values("-78 C")

        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]["unit"], "degC")
        self.assertEqual(values[0]["normalized_unit"], "K")
        self.assertAlmostEqual(values[0]["normalized_value"], 195.15, places=2)

    def test_unitless_number_before_word_is_not_coerced_to_single_letter_unit(self):
        values = parse_numeric_values("1 coldest threshold option")

        self.assertTrue(values)
        self.assertIsNone(values[0]["unit"])
        self.assertIsNone(values[0]["normalized_unit"])
        self.assertEqual(values[0]["normalized_value"], 1.0)

    def test_parse_numeric_options_requires_all_options_numeric(self):
        parsed = parse_numeric_options({"A": "600 C", "B": "-78 C", "C": "xenon tetrafluoride"})

        self.assertEqual(parsed["status"], "abstained")
        self.assertEqual(parsed["numeric_option_count"], 2)

    def test_relation_classifier_detects_coldest_threshold(self):
        relation = classify_numeric_relation(
            "Which is the coldest temperature at which Xenon tetrafluoride can still be produced efficiently?",
            value_type="temperature",
        )

        self.assertEqual(relation["relation_family"], "threshold_minimum")
        self.assertEqual(relation["direction"], "lower_is_correct")
        self.assertEqual(relation["confidence"], "high")

    def test_solver_accepts_unique_source_grounded_numeric_direct_witness(self):
        result = solve_numeric_threshold_lane(
            stem=(
                "Using any method of synthesis, which of the following is the coldest "
                "temperature at which Xenon tetrafluoride can still be produced efficiently?"
            ),
            options={
                "A": "600 C",
                "B": "400 C",
                "C": "200 C",
                "D": "78 C",
            },
            docs_by_label={
                "B": [{
                    "title": "Xenon tetrafluoride preparation",
                    "snippet": (
                        "XeF4, xenon tetrafluoride, has a lowest efficient synthesis "
                        "temperature near 400 C for the reaction of xenon and fluorine."
                    ),
                    "source": "semantic_scholar",
                }],
                "A": [{
                    "title": "Unrelated furnace note",
                    "snippet": "A furnace can reach 600 C for many inorganic reactions.",
                    "source": "openalex",
                }],
            },
        )

        self.assertEqual(result["status"], "activated")
        self.assertEqual(result["selected_label"], "B")
        self.assertTrue(result["direct_high_confidence"])
        self.assertGreaterEqual(result["numeric_direct_witness_count"], 1)
        self.assertGreaterEqual(result["witness_count"], 1)
        self.assertGreaterEqual(result["witness_source_doc_count"], 1)
        self.assertGreaterEqual(result["witness_parsed_source_value_count"], 1)
        self.assertIsInstance(result["witness_direct_rejection_reason_counts"], dict)
        self.assertIsInstance(result["witness_value_match_failure_counts"], dict)
        self.assertTrue(result["parse_hash"])
        self.assertTrue(result["relation_hash"])
        self.assertTrue(result["numeric_witness_hash"])
        self.assertTrue(result["router_payload_hash"])

    def test_solver_rejects_generic_numeric_span_without_subject_anchor(self):
        result = solve_numeric_threshold_lane(
            stem=(
                "Using any method of synthesis, which of the following is the coldest "
                "temperature at which Xenon tetrafluoride can still be produced efficiently?"
            ),
            options={
                "A": "600 C",
                "B": "400 C",
                "C": "200 C",
                "D": "-78 C",
            },
            docs_by_label={
                "D": [{
                    "title": "Dry ice bath",
                    "snippet": "A dry ice acetone bath has a temperature near -78 C.",
                    "source": "wikipedia",
                }],
            },
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["reason"], "no_high_confidence_numeric_direct_witness")
        self.assertFalse(result["direct_high_confidence"])
        self.assertGreaterEqual(result["witness_parsed_source_value_count"], 1)
        self.assertIsInstance(result["witness_value_match_failure_counts"], dict)
        self.assertIsInstance(result["witness_direct_rejection_reason_counts"], dict)

    def test_threshold_non_extreme_numeric_mention_without_threshold_signal_abstains(self):
        result = solve_numeric_threshold_lane(
            stem=(
                "Using any method of synthesis, which of the following is the coldest "
                "temperature at which Xenon tetrafluoride can still be produced efficiently?"
            ),
            options={
                "A": "600 C",
                "B": "400 C",
                "C": "200 C",
                "D": "-78 C",
            },
            docs_by_label={
                "B": [{
                    "title": "Xenon tetrafluoride preparation",
                    "snippet": (
                        "XeF4, xenon tetrafluoride, is produced efficiently by reaction "
                        "of xenon and fluorine at about 400 C."
                    ),
                    "source": "semantic_scholar",
                }],
            },
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(
            result["reason"],
            "threshold_numeric_witness_missing_direction_extreme_evidence",
        )

    def test_threshold_extreme_numeric_witness_can_select_lowest_option(self):
        result = solve_numeric_threshold_lane(
            stem=(
                "Using any method of synthesis, which of the following is the coldest "
                "temperature at which Xenon tetrafluoride can still be produced efficiently?"
            ),
            options={
                "A": "600 C",
                "B": "400 C",
                "C": "200 C",
                "D": "-78 C",
            },
            docs_by_label={
                "D": [{
                    "title": "Low temperature XeF4 synthesis note",
                    "snippet": (
                        "XeF4, xenon tetrafluoride, is produced efficiently by reaction "
                        "of xenon and fluorine at about -78 C."
                    ),
                    "source": "semantic_scholar",
                }],
            },
        )

        self.assertEqual(result["status"], "activated")
        self.assertEqual(result["selected_label"], "D")
        self.assertTrue(result["direct_high_confidence"])

    def test_numeric_same_row_directness_requires_value_and_relation_terms(self):
        detail = numeric_same_row_directness_detail(
            stem=(
                "Using any method of synthesis, which of the following is the coldest "
                "temperature at which Xenon tetrafluoride can still be produced efficiently?"
            ),
            option_text="400 C",
            doc={
                "title": "Xenon tetrafluoride preparation",
                "snippet": (
                    "XeF4, xenon tetrafluoride, has a lowest efficient synthesis "
                    "temperature near 400 C for the reaction of xenon and fluorine."
                ),
                "source": "semantic_scholar",
            },
        )

        self.assertEqual(detail["status"], "evaluated")
        self.assertTrue(detail["numeric_same_row_value_match"])
        self.assertTrue(detail["numeric_same_row_direct"])
        self.assertGreaterEqual(detail["relation_overlap"], detail["required_relation_overlap"])
        self.assertGreater(detail["threshold_signal_overlap"], 0)
        self.assertFalse(detail["raw_content_persisted"])

    def test_numeric_same_row_directness_rejects_value_only_source_row(self):
        detail = numeric_same_row_directness_detail(
            stem=(
                "Using any method of synthesis, which of the following is the coldest "
                "temperature at which Xenon tetrafluoride can still be produced efficiently?"
            ),
            option_text="400 C",
            doc={
                "title": "Furnace calibration",
                "snippet": "The furnace controller was calibrated at 400 C during maintenance.",
                "source": "openalex",
            },
        )

        self.assertEqual(detail["status"], "evaluated")
        self.assertTrue(detail["numeric_same_row_value_match"])
        self.assertFalse(detail["numeric_same_row_direct"])
        self.assertIn("subject_anchor_not_satisfied", detail["rejection_reasons"])
        self.assertIn("relation_overlap_below_required", detail["rejection_reasons"])

    def test_numeric_same_row_directness_uses_primary_option_value_only(self):
        detail = numeric_same_row_directness_detail(
            stem=(
                "Using any method of synthesis, which of the following is the coldest "
                "temperature at which Xenon tetrafluoride can still be produced efficiently?"
            ),
            option_text="1 400 C",
            doc={
                "title": "Xenon tetrafluoride preparation",
                "snippet": (
                    "XeF4, xenon tetrafluoride, has a lowest efficient synthesis "
                    "temperature near 400 C for the reaction of xenon and fluorine."
                ),
                "source": "semantic_scholar",
            },
        )

        self.assertEqual(detail["status"], "evaluated")
        self.assertFalse(detail["numeric_same_row_value_match"])
        self.assertFalse(detail["numeric_same_row_direct"])
        self.assertIn("numeric_value_match_missing", detail["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()

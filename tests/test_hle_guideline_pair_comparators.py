import unittest

from assumption_os.hle_guideline_pair_comparators import (
    fe_hyperfine_pair_binding_detail,
    medical_guideline_permutation_ordering_detail,
)


class TestHleGuidelinePairComparators(unittest.TestCase):
    def test_medical_guideline_permutation_ordering_scores_patient_severity(self):
        detail = medical_guideline_permutation_ordering_detail(
            stem=(
                "Patient 1: Severe burst fracture of L2, no neurologic deficits. "
                "Patient 2: Compression fracture of L2 with mild traumatic "
                "spondylolisthesis of L1, no neurologic deficits. "
                "Patient 3: Split fracture of L2, with mildly disordered pelvic "
                "functions. Prioritize according to surgical indications."
            ),
            option_text="Patient 3, Patient 2, Patient 1",
            rows=[
                {
                    "title": "Thoracolumbar injury classification guideline",
                    "snippet": (
                        "TLICS thoracolumbar trauma classification assigns points "
                        "for morphology, neurologic status, and surgical treatment "
                        "indications."
                    ),
                    "source": "local_guideline",
                }
            ],
        )

        self.assertEqual(detail["status"], "activated")
        self.assertTrue(detail["candidate_exact_expected_order"])
        self.assertEqual(detail["candidate_rank_penalty"], 0)
        self.assertEqual(detail["evidence_row_count"], 1)
        serialized_rows = str(detail["score_rows"]).lower()
        self.assertNotIn("severe burst fracture", serialized_rows)
        self.assertNotIn("disordered pelvic", serialized_rows)

    def test_medical_guideline_permutation_ordering_blocks_without_guideline_source(self):
        detail = medical_guideline_permutation_ordering_detail(
            stem=(
                "Patient 1: Severe burst fracture of L2, no neurologic deficits. "
                "Patient 2: Compression fracture of L2 with mild traumatic "
                "spondylolisthesis of L1, no neurologic deficits. "
                "Prioritize according to surgical indications."
            ),
            option_text="Patient 2, Patient 1",
            rows=[{"title": "Unrelated", "snippet": "No guideline evidence."}],
        )

        self.assertEqual(detail["status"], "blocked")
        self.assertEqual(detail["reason"], "missing_guideline_source_evidence")

    def test_fe_hyperfine_pair_binding_is_partial_without_geometry_or_superlative(self):
        detail = fe_hyperfine_pair_binding_detail(
            stem="Which combination has the largest hyperfine field in 57Fe Mossbauer spectroscopy?",
            option_text="planar S = 5/2 Fe(III)",
            rows=[
                {
                    "title": "MossWinn paramagnetic hyperfine structure examples",
                    "snippet": (
                        "Iron is present in this complex in the high spin ferric "
                        "form ( Fe 3+, S = 5/2 ). The hyperfine magnetic "
                        "interaction tensor is modeled."
                    ),
                    "source": "local_fulltext",
                }
            ],
        )

        self.assertEqual(detail["status"], "evaluated")
        self.assertEqual(detail["partial_pair_binding_row_count"], 1)
        self.assertEqual(detail["direct_pair_binding_row_count"], 0)
        self.assertEqual(detail["missing_geometry_row_count"], 1)

    def test_fe_hyperfine_pair_binding_requires_same_row_geometry_relation(self):
        detail = fe_hyperfine_pair_binding_detail(
            stem="Which combination has the largest hyperfine field in 57Fe Mossbauer spectroscopy?",
            option_text="planar S = 5/2 Fe(III)",
            rows=[
                {
                    "title": "Planar ferric hyperfine field comparison",
                    "snippet": (
                        "The planar ferric Fe 3+ S = 5/2 complex has the largest "
                        "hyperfine field among the compared configurations."
                    ),
                    "source": "local_fulltext",
                }
            ],
        )

        self.assertEqual(detail["status"], "activated")
        self.assertEqual(detail["direct_pair_binding_row_count"], 1)
        self.assertGreater(detail["best_pair_binding_score"], 8.0)


if __name__ == "__main__":
    unittest.main()

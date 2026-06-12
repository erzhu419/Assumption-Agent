import unittest
from pathlib import Path

from assumption_os.finite_theorem_fragment import (
    blackwell_witness,
    build_finite_theorem_fragment_payload,
    extract_natural_language_diagram,
)


class FiniteTheoremFragmentTest(unittest.TestCase):
    def test_finite_theorem_fragment_passes_all_bounded_gates(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_theorem_fragment_payload(
            root=root,
            eval_id="unit_finite_theorem_fragment",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertTrue(metrics["finite_theorem_fragment_claim_allowed"])
        self.assertTrue(metrics["identity_law_pass"])
        self.assertTrue(metrics["associativity_pass"])
        self.assertTrue(metrics["functor_identity_pass"])
        self.assertTrue(metrics["functor_composition_pass"])
        self.assertTrue(metrics["naturality_pass"])
        self.assertTrue(metrics["finite_limit_colimit_pass"])
        self.assertTrue(metrics["adjunction_pass"])
        self.assertTrue(metrics["monoidal_pass"])
        self.assertTrue(metrics["blackwell_exact_witness_pass"])
        self.assertTrue(metrics["fisher_geometry_metric_laws_pass"])
        self.assertTrue(metrics["external_lean_check_passed"])
        self.assertGreaterEqual(metrics["external_lean_theorem_count"], 20)
        self.assertTrue(metrics["lean_verified_finite_theorem_fragment_claim_allowed"])
        self.assertTrue(metrics["external_proof_assistant_integrated"])
        self.assertFalse(metrics["full_theorem_prover_claim_allowed"])

    def test_blackwell_exact_witness_accepts_degradation_and_rejects_inverse_claim(self):
        identity = [[1.0, 0.0], [0.0, 1.0]]
        noisy = [[0.82, 0.18], [0.24, 0.76]]

        identity_to_noisy = blackwell_witness(identity, noisy)
        noisy_to_identity = blackwell_witness(noisy, identity)

        self.assertTrue(identity_to_noisy["dominates"])
        self.assertTrue(identity_to_noisy["row_stochastic_witness"])
        self.assertFalse(noisy_to_identity["dominates"])

    def test_natural_language_extractor_formalizes_known_patterns_and_abstains(self):
        formalized = extract_natural_language_diagram(
            "Lenz law and Le Chatelier principle both counteract perturbations and restore equilibrium."
        )
        unrelated = extract_natural_language_diagram("I prefer a blue button because it looks nice.")

        self.assertEqual(formalized["status"], "formalized")
        self.assertEqual(formalized["family"], "negative_feedback_regulation")
        self.assertTrue(formalized["certificate"]["validation"]["valid"])
        self.assertEqual(unrelated["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from assumption_os.finite_theorem_lean_verifier import (
    build_finite_theorem_lean_verifier_payload,
    render_finite_theorem_fragment_lean,
    validate_lean_text,
)


class FiniteTheoremLeanVerifierTest(unittest.TestCase):
    def test_lean_verifier_generates_and_checks_finite_fragment(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            lean_out = Path(temp_dir) / "finite_fragment.lean"
            payload = build_finite_theorem_lean_verifier_payload(
                root=root,
                eval_id="unit_finite_theorem_lean_verifier",
                lean_out=lean_out,
                run_lean_if_available=True,
            )

            self.assertTrue(payload["pass"], payload["failed_gates"])
            self.assertTrue(lean_out.exists())
            self.assertGreaterEqual(payload["metrics"]["lean_theorem_count"], 20)
            self.assertTrue(payload["metrics"]["external_lean_check_passed"])
            self.assertTrue(payload["metrics"]["finite_theorem_fragment_lean_verified"])
            self.assertFalse(payload["metrics"]["full_theorem_prover_claim_allowed"])

    def test_lean_text_is_self_contained_and_has_required_theorems(self):
        lean_text = render_finite_theorem_fragment_lean()
        validation = validate_lean_text(lean_text)

        self.assertTrue(validation["all_required_constructs_present"])
        self.assertTrue(validation["no_sorry_or_admit"])
        self.assertTrue(validation["mathlib_free"])
        self.assertGreaterEqual(validation["native_decide_count"], 20)


if __name__ == "__main__":
    unittest.main()

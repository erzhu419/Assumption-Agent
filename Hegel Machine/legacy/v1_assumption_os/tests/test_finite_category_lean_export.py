import tempfile
import unittest
from pathlib import Path

from assumption_os.finite_category_lean_export import (
    ALLOWED_GATE_OUTPUTS,
    FORBIDDEN_GENERATOR_OUTPUTS,
    build_finite_category_lean_export_payload,
    render_lean_export,
    validate_lean_export_text,
)
from assumption_os.finite_category_certificate import build_finite_category_certificate_payload


class FiniteCategoryLeanExportTest(unittest.TestCase):
    def test_lean_export_payload_passes_and_writes_file(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            lean_out = Path(temp_dir) / "finite_certificates.lean"
            payload = build_finite_category_lean_export_payload(
                root=root,
                eval_id="unit_finite_category_lean_export",
                lean_out=lean_out,
                run_lean_if_available=False,
            )

            self.assertTrue(payload["pass"], payload["failed_gates"])
            self.assertTrue(lean_out.exists())
            self.assertEqual(payload["metrics"]["certificate_count"], 16)
            self.assertEqual(payload["metrics"]["lean_definition_count"], 16)
            self.assertEqual(payload["metrics"]["forbidden_generator_output_count"], 0)
            self.assertFalse(payload["metrics"]["full_theorem_prover_claim_allowed"])

    def test_lean_text_contains_gate_only_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        cert_payload = build_finite_category_certificate_payload(
            root=root,
            eval_id="unit_finite_category_lean_export_source",
            write_engine_artifact=False,
        )
        lean_text = render_lean_export(cert_payload["certificates"])
        validation = validate_lean_export_text(lean_text, cert_payload["certificates"])

        self.assertTrue(validation["lean_readable_structures_present"])
        self.assertTrue(validation["expected_proof_obligations_listed"])
        self.assertTrue(validation["allowed_gate_outputs_only"])
        self.assertTrue(validation["no_forbidden_generator_outputs"])
        self.assertTrue(validation["not_claimed_boundaries_exported"])
        for output in ALLOWED_GATE_OUTPUTS:
            self.assertIn(output, lean_text)
        for output in FORBIDDEN_GENERATOR_OUTPUTS:
            self.assertNotIn(output, lean_text)

    def test_default_export_is_external_check_ready_when_lean_is_available(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_category_lean_export_payload(
            root=root,
            eval_id="unit_finite_category_lean_export_external",
            run_lean_if_available=True,
        )

        self.assertTrue(payload["pass"], payload["failed_gates"])
        if payload["external_check"]["available"]:
            self.assertTrue(payload["external_check"]["attempted"])
            self.assertTrue(payload["external_check"]["passed"], payload["external_check"])


if __name__ == "__main__":
    unittest.main()

import copy
import unittest
from pathlib import Path

from assumption_os.finite_category_certificate import (
    build_finite_category_certificate_payload,
    validate_certificate,
)


class FiniteCategoryCertificateTest(unittest.TestCase):
    def test_certificate_payload_passes_and_blocks_negative_controls(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_category_certificate_payload(
            root=root,
            eval_id="unit_finite_category_certificate",
            write_engine_artifact=False,
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["certificate_count"], 16)
        self.assertEqual(metrics["valid_certificate_count"], 16)
        self.assertEqual(metrics["accepted_certificate_count"], 9)
        self.assertEqual(metrics["blocked_certificate_count"], 7)
        self.assertEqual(metrics["proof_obligation_pass_rate"], 1.0)
        self.assertEqual(metrics["negative_control_blocked_count"], 7)
        self.assertFalse(metrics["unbounded_theorem_prover_claim_allowed"])

    def test_tampered_naturality_square_fails_validation(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_category_certificate_payload(
            root=root,
            eval_id="unit_finite_category_certificate_tamper",
            write_engine_artifact=False,
        )
        certificate = copy.deepcopy(payload["certificates"][0])
        square = certificate["naturality_squares"][0]
        certificate["category"]["composition_table"][f"{square['left']};{square['bottom']}"] = square["left"]

        report = validate_certificate(certificate)
        issue_names = {issue["issue"] for issue in report["issues"]}

        self.assertFalse(report["valid"])
        self.assertIn("naturality_square_not_commutative", issue_names)

    def test_certificates_record_not_claimed_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_category_certificate_payload(
            root=root,
            eval_id="unit_finite_category_certificate_boundaries",
            write_engine_artifact=False,
        )
        certificate = payload["certificates"][0]

        self.assertIn("arbitrary theorem proving", certificate["not_claimed"])
        self.assertIn("unbounded category-theory reasoning engine", certificate["not_claimed"])


if __name__ == "__main__":
    unittest.main()

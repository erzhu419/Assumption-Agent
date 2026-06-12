import unittest
from pathlib import Path

from assumption_os.conservative_generalization_gate import REQUIRED_PROMOTION_RELATIONS
from assumption_os.framework_object_model import (
    PROMOTED_STATUSES,
    build_framework_object_model_payload,
)
from assumption_os.schema import AssumptionType, EdgeType


class FrameworkObjectModelTest(unittest.TestCase):
    def test_framework_objects_roundtrip_as_first_class_graph_nodes(self):
        payload = build_framework_object_model_payload(
            root=Path("."),
            eval_id="unit_framework_object_model_roundtrip",
        )
        metrics = payload["metrics"]
        roundtrip = payload["roundtrip"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["framework_node_count"], 2)
        self.assertGreaterEqual(metrics["framework_branch_count"], 2)
        self.assertGreaterEqual(metrics["support_node_count"], 8)
        self.assertTrue(metrics["jsonl_roundtrip_exact"])
        self.assertEqual(roundtrip["framework_node_count_after"], metrics["framework_node_count"])
        self.assertEqual(roundtrip["framework_branch_count_after"], metrics["framework_branch_count"])
        self.assertEqual(roundtrip["certificate_count_after"], metrics["certificate_count"])

        framework_types = {row["type"] for row in payload["support_nodes"]}
        self.assertIn(AssumptionType.RESIDUAL.value, framework_types)
        self.assertIn(AssumptionType.CASE.value, framework_types)

    def test_promoted_frameworks_require_conservative_extension_certificates(self):
        payload = build_framework_object_model_payload(
            root=Path("."),
            eval_id="unit_framework_object_model_certificates",
        )
        metrics = payload["metrics"]
        promoted = [
            row
            for row in payload["framework_nodes"]
            if row["status"] in PROMOTED_STATUSES
        ]
        certificate_targets = {
            row["candidate_framework_id"]
            for row in payload["certificates"]
        }

        self.assertEqual(metrics["promoted_certificate_coverage"], 1.0)
        self.assertEqual(metrics["uncertified_active_framework_allowed_count"], 0)
        self.assertEqual(len(promoted), metrics["certificate_count"])
        self.assertTrue(all(row["id"] in certificate_targets for row in promoted))
        self.assertTrue(all(row["formal_certificate_refs"] for row in promoted))

    def test_required_lifecycle_and_conservative_relations_are_present(self):
        payload = build_framework_object_model_payload(
            root=Path("."),
            eval_id="unit_framework_object_model_edges",
        )
        metrics = payload["metrics"]
        edge_types = set(metrics["edge_type_counts"])
        required = set(REQUIRED_PROMOTION_RELATIONS) | {
            EdgeType.HAS_CERTIFICATE.value,
            EdgeType.DEMOTES_TO_BRANCH.value,
            EdgeType.REPLACES_BOUNDARY_OF.value,
        }

        self.assertEqual(metrics["required_relation_coverage"], 1.0)
        self.assertTrue(required.issubset(edge_types))
        self.assertGreaterEqual(metrics["demotes_to_branch_edge_count"], 1)
        self.assertGreaterEqual(metrics["replaces_boundary_of_edge_count"], 1)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()

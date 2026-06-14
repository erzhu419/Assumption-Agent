import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assumption_os.l4_wallclock_cumulative_evidence import build_l4_wallclock_cumulative_evidence_payload


class L4WallclockCumulativeEvidenceTest(unittest.TestCase):
    def test_cumulative_24h_passes_without_continuous_24h_claim(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            first = root / "first.json"
            second = root / "second.json"
            _write_component(
                first,
                eval_id="first",
                seconds=81109.5919,
                start="2026-06-13T00:53:42.582157+00:00",
                end="2026-06-13T23:25:32.174025+00:00",
            )
            _write_component(
                second,
                eval_id="second",
                seconds=6200.0175,
                start="2026-06-14T02:34:00.155806+00:00",
                end="2026-06-14T04:17:20.173307+00:00",
            )
            payload = build_l4_wallclock_cumulative_evidence_payload(
                root=root,
                components=[first, second],
            )

        metrics = payload["metrics"]
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["observed_wallclock_seconds"], 24 * 3600)
        self.assertTrue(metrics["cumulative_24h_claim_allowed"])
        self.assertFalse(metrics["continuous_24h_claim_allowed"])
        self.assertIn("continuous_24h_supervised_daemon", payload["blocked_claims"])
        self.assertEqual(metrics["component_interval_overlap_count"], 0)


def _write_component(path: Path, *, eval_id: str, seconds: float, start: str, end: str) -> None:
    payload = {
        "eval_id": eval_id,
        "pass": True,
        "wallclock_start": start,
        "wallclock_end": end,
        "metrics": {
            "observed_wallclock_seconds": seconds,
            "observed_uptime": 1.0,
            "cycle_count": 1,
            "auto_apply_count": 1,
            "manual_review_count": 0,
            "blocked_count": 0,
            "incident_count": 0,
            "graph_pollution_alert_count": 0,
            "forbidden_auto_apply_count": 0,
            "ungated_mutation_count": 0,
            "main_graph_mutation_count": 0,
            "rollback_success_rate": 1.0,
            "graph_journal_replayable": True,
            "queue_journal_replayable": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

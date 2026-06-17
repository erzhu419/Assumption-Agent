from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.hle_ablation_result_analyzer import analyze_hle_ablation_run


class HleAblationResultAnalyzerTest(unittest.TestCase):
    def test_analyzer_reports_performance_and_no_recursive_pollution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full"
            no_recursive = root / "no_recursive"
            full.mkdir()
            no_recursive.mkdir()
            _write_shard(
                full / "run_full_shard_000.json",
                rows=[
                    _row("p1", "raw", False),
                    _row("p1", "assumption_agent_recursive_verify", True),
                    _row("p1", "hipporag_baseline", False),
                    _row("p2", "raw", True),
                    _row("p2", "assumption_agent_recursive_verify", True),
                    _row("p2", "hipporag_baseline", True),
                ],
            )
            _write_shard(
                no_recursive / "run_no_recursive_shard_000.json",
                rows=[
                    _row("p1", "raw", False),
                    _row("p1", "assumption_agent_recursive_verify", True),
                ],
            )
            (no_recursive / "run_no_recursive_shard_000.jsonl").write_text(
                json.dumps({"event": "recursive_child_start"}) + "\n",
                encoding="utf-8",
            )

            payload = analyze_hle_ablation_run(
                run_dir=root,
                eval_id="unit",
                bootstrap_samples=20,
                seed=1,
            )

        self.assertTrue(payload["performance_validation"]["gates"]["reference_agent_not_below_raw"])
        self.assertEqual(payload["profiles"]["full"]["variant_summary"]["assumption_agent_recursive_verify"]["accuracy"], 1.0)
        self.assertEqual(payload["profiles"]["full"]["paired_control_comparison"]["agent_vs_raw"]["delta"], 0.5)
        self.assertFalse(payload["profiles"]["full"]["contamination"]["contaminated"])
        self.assertTrue(payload["profiles"]["no_recursive"]["contamination"]["contaminated"])
        self.assertIn("no_recursive", payload["pollution_summary"]["contaminated_profiles"])

    def test_analyzer_accepts_flat_parallel_run_and_ignores_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "flat_run"
            root.mkdir()
            manifest_dir = root / "fixed_problem_hash_manifests"
            manifest_dir.mkdir()
            _write_shard(
                root / "flat_shard_000.json",
                rows=[
                    _row("p1", "raw", False),
                    _row("p1", "assumption_agent_recursive_verify", True),
                ],
            )
            (manifest_dir / "flat_shard_000.problem_hashes.json").write_text(
                json.dumps(["p1"]),
                encoding="utf-8",
            )

            payload = analyze_hle_ablation_run(
                run_dir=root,
                eval_id="flat",
                bootstrap_samples=20,
                seed=2,
            )

        self.assertIn("flat_run", payload["profiles"])
        self.assertEqual(payload["profiles"]["flat_run"]["row_count"], 2)


def _write_shard(path: Path, *, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")


def _row(problem_id_hash: str, variant: str, correct: bool) -> dict:
    return {
        "answer_type": "multipleChoice",
        "category": "unit",
        "component_efficacy": {
            "flags": {
                "final_correct": correct,
                "raw_preserve_selector_activated": variant == "assumption_agent_recursive_verify",
            }
        },
        "correct": correct,
        "error": None,
        "gold_answer_persisted": False,
        "module_trace": [
            {"module": "answer_type_router", "status": "activated"},
            {"module": "recursive_child_validation", "status": "disabled"},
        ],
        "prediction_text_persisted": False,
        "problem_id_hash": problem_id_hash,
        "raw_question_persisted": False,
        "variant": variant,
    }


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.hle_paired_run_comparison import compare_hle_runs


class HlePairedRunComparisonTest(unittest.TestCase):
    def test_compares_flat_candidate_against_profiled_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            baseline = root / "baseline"
            (baseline / "full").mkdir(parents=True)
            (candidate / "fixed_problem_hash_manifests").mkdir(parents=True)
            _write_shard(
                candidate / "candidate_shard_000.json",
                rows=[
                    _row("p1", "assumption_agent_recursive_verify", True),
                    _row("p2", "assumption_agent_recursive_verify", True),
                    _row("p3", "assumption_agent_recursive_verify", False),
                ],
            )
            (candidate / "fixed_problem_hash_manifests" / "candidate_shard_000.problem_hashes.json").write_text(
                json.dumps(["p1", "p2", "p3"]),
                encoding="utf-8",
            )
            _write_shard(
                baseline / "full" / "baseline_full_shard_000.json",
                rows=[
                    _row("p1", "raw", False),
                    _row("p2", "raw", True),
                    _row("p3", "raw", False),
                    _row("p1", "hipporag_baseline", False),
                    _row("p2", "hipporag_baseline", False),
                    _row("p3", "hipporag_baseline", False),
                ],
            )

            payload = compare_hle_runs(
                candidate_run_dir=candidate,
                baseline_run_dir=baseline,
                baseline_profile="full",
                baseline_variants=["raw", "hipporag_baseline"],
                expected_sample_size=3,
                bootstrap_samples=20,
                seed=7,
            )

        self.assertTrue(payload["pass"])
        self.assertEqual(payload["candidate"]["problem_count"], 3)
        self.assertEqual(payload["comparisons"]["raw"]["shared_n"], 3)
        self.assertAlmostEqual(payload["comparisons"]["raw"]["delta"], 1 / 3)
        self.assertEqual(payload["comparisons"]["raw"]["wins"], 1)
        self.assertEqual(payload["oracle_summary"]["candidate_only_correct_count"], 1)
        self.assertFalse(payload["raw_content_persisted"])


def _write_shard(path: Path, *, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")


def _row(problem_id_hash: str, variant: str, correct: bool) -> dict:
    return {
        "answer_type": "multipleChoice",
        "category": "unit",
        "component_efficacy": {"flags": {"final_correct": correct}},
        "correct": correct,
        "error": None,
        "gold_answer_persisted": False,
        "module_trace": [{"module": "answer_type_router", "status": "activated"}],
        "prediction_text_persisted": False,
        "problem_id_hash": problem_id_hash,
        "raw_question_persisted": False,
        "variant": variant,
    }


if __name__ == "__main__":
    unittest.main()

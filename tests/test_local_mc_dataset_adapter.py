import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from assumption_os.autonomy_journal import stable_hash
from assumption_os.local_mc_dataset_adapter import build_local_mc_jsonl_from_zip


class LocalMcDatasetAdapterTest(unittest.TestCase):
    def test_builds_mmlu_jsonl_from_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "mmlu.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "data/test/demo_subject_test.csv",
                    "Question one?,A1,B1,C1,D1,A\n"
                    "Question two?,A2,B2,C2,D2,D\n",
                )
            out = root / "out.jsonl"
            payload = build_local_mc_jsonl_from_zip(
                root=root,
                dataset="mmlu",
                zip_path=zip_path,
                split="test",
                output_jsonl=out,
                sample_size=10,
                seed=7,
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(payload["sample_size_written"], 2)
            self.assertEqual(rows[0]["dataset"], "mmlu")
            self.assertEqual(rows[0]["subject"], "demo_subject")
            self.assertEqual(set(rows[0]["choices"]), {"A", "B", "C", "D"})
            self.assertIn(rows[0]["answer"], {"A", "D"})

    def test_skips_ceval_rows_without_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "ceval.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "test/demo_test.csv",
                    "id,question,A,B,C,D\n0,No answer?,A,B,C,D\n",
                )
                archive.writestr(
                    "val/demo_val.csv",
                    "id,question,A,B,C,D,answer\n0,Has answer?,A,B,C,D,B\n",
                )
            out = root / "out.jsonl"
            payload = build_local_mc_jsonl_from_zip(
                root=root,
                dataset="ceval",
                zip_path=zip_path,
                split="test",
                output_jsonl=out,
                sample_size=10,
                seed=0,
            )
            self.assertEqual(payload["sample_size_written"], 0)
            payload = build_local_mc_jsonl_from_zip(
                root=root,
                dataset="ceval",
                zip_path=zip_path,
                split="val",
                output_jsonl=out,
                sample_size=10,
                seed=0,
            )
            self.assertEqual(payload["sample_size_written"], 1)

    def test_excludes_question_hashes_from_prior_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "mmlu.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "data/test/demo_subject_test.csv",
                    "Question one?,A1,B1,C1,D1,A\n"
                    "Question two?,A2,B2,C2,D2,D\n",
                )
            excluded_question = "Question one?\nA. A1\nB. B1\nC. C1\nD. D1"
            prior = root / "prior.json"
            prior.write_text(
                json.dumps({
                    "rows": [{
                        "question_hash": stable_hash({"local_mc_question": excluded_question})
                    }]
                }),
                encoding="utf-8",
            )
            out = root / "out.jsonl"
            payload = build_local_mc_jsonl_from_zip(
                root=root,
                dataset="mmlu",
                zip_path=zip_path,
                split="test",
                output_jsonl=out,
                sample_size=10,
                seed=0,
                exclude_eval_paths=[prior],
            )
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(payload["sample_size_written"], 1)
            self.assertEqual(rows[0]["question"], "Question two?")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.hle_retry_manifest_runner import (
    RetryGroup,
    build_retry_command,
    load_retry_groups,
)


class TestHleRetryManifestRunner(unittest.TestCase):
    def test_load_retry_groups_groups_by_model_and_variant(self) -> None:
        payload = {
            "endpoint_retry_manifest": {
                "retry_items": [
                    {
                        "model": "gpt-5.4-mini",
                        "variant": "raw",
                        "seed_offset": 44,
                        "retry_key": "gpt-5.4-mini::raw::p44",
                    },
                    {
                        "model": "gpt-5.4-mini",
                        "variant": "hipporag_baseline",
                        "seed_offset": 44,
                        "retry_key": "gpt-5.4-mini::hipporag_baseline::p44",
                    },
                    {
                        "model": "gpt-5.4-mini",
                        "variant": "raw",
                        "seed_offset": 52,
                        "retry_key": "gpt-5.4-mini::raw::p52",
                    },
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            groups = load_retry_groups(path)

        self.assertEqual(
            [(group.model, group.variant, group.seed_offsets) for group in groups],
            [
                ("gpt-5.4-mini", "hipporag_baseline", (44,)),
                ("gpt-5.4-mini", "raw", (44, 52)),
            ],
        )

    def test_build_retry_command_preserves_explicit_seed_offsets(self) -> None:
        group = RetryGroup(
            model="gpt-5.4-mini",
            variant="raw",
            seed_offsets=(44, 52),
            retry_keys=("gpt-5.4-mini::raw::p44", "gpt-5.4-mini::raw::p52"),
        )
        command = build_retry_command(
            group=group,
            eval_id="retry_eval",
            run_dir=Path("runs"),
            md_dir=Path("md"),
            parallel_workers=1,
            model_router_attempts=2,
            model_router_transient_extra_attempts=0,
            model_router_per_attempt_timeout=0,
            model_router_no_byte_timeout_sec=600,
            model_router_global_concurrency=1,
            live_model_preflight_probe_count=1,
            live_model_preflight_max_error_rate=0,
            live_model_preflight_timeout_sec=60,
            live_model_preflight_prompt_chars=12000,
            live_model_preflight_max_tokens=512,
        )

        self.assertIn("--generalization-holdout-preserve-explicit-seed-offsets", command)
        self.assertEqual(command[command.index("--seed-offsets") + 1], "44,52")
        self.assertEqual(command[command.index("--variants") + 1], "raw")
        self.assertEqual(command[command.index("--models") + 1], "gpt-5.4-mini")


if __name__ == "__main__":
    unittest.main()

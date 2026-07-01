import json
import os
import tempfile
import unittest
from pathlib import Path

from assumption_os.private_env import load_private_env


class TestPrivateEnv(unittest.TestCase):
    def test_load_private_env_loads_names_without_values_in_metadata(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = Path(tmpdir) / "private.env"
            path.write_text(
                "\n".join([
                    "OPENAI_API_KEY=token_secret_value",
                    "export OPENAI_BASE_URL=https://example.invalid/v1",
                    "SEMANTIC_SCHOLAR_API_KEY='s2-secret-value'",
                    "OPENALEX_API_KEY=\"oa-secret-value\" # comment",
                ]),
                encoding="utf-8",
            )
            os.chmod(path, 0o600)
            env: dict[str, str] = {}

            status = load_private_env(environ=env, path=path)

        self.assertTrue(status["loaded"])
        self.assertEqual(env["OPENAI_API_KEY"], "token_secret_value")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://example.invalid/v1")
        serialized = json.dumps(status, sort_keys=True)
        self.assertIn("OPENAI_API_KEY", serialized)
        self.assertNotIn("token_secret_value", serialized)
        self.assertNotIn("s2-secret-value", serialized)
        self.assertNotIn("oa-secret-value", serialized)
        self.assertFalse(status["raw_content_persisted"])

    def test_load_private_env_refuses_group_or_world_readable_file(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = Path(tmpdir) / "private.env"
            path.write_text("OPENAI_API_KEY=token_secret_value\n", encoding="utf-8")
            os.chmod(path, 0o644)
            env: dict[str, str] = {}

            status = load_private_env(environ=env, path=path)

        self.assertFalse(status["loaded"])
        self.assertEqual(status["skipped_reason"], "private_env_file_not_private")
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_load_private_env_preserves_existing_values_by_default(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            path = Path(tmpdir) / "private.env"
            path.write_text("OPENAI_API_KEY=file_token_value\nOPENAI_BASE_URL=https://file.invalid/v1\n", encoding="utf-8")
            os.chmod(path, 0o600)
            env = {"OPENAI_API_KEY": "existing_token_value"}

            status = load_private_env(environ=env, path=path)

        self.assertEqual(env["OPENAI_API_KEY"], "existing_token_value")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://file.invalid/v1")
        self.assertIn("OPENAI_API_KEY", status["skipped_keys"])

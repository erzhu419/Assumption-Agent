from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_hybridqa_set_interaction_meta_development_v1.py"
)
SPEC = importlib.util.spec_from_file_location("set_interaction_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _write(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_acquisition_binding_rejects_self_consistent_pack_replacement(
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        acquisition = Path(temporary) / "acquisition"
        acquisition.mkdir(mode=0o700)
        private_hashes = {
            f"pack_{index}.private.json": _write(
                acquisition / f"pack_{index}.private.json",
                runner.canonical_bytes({"index": index}),
            )
            for index in range(7)
        }
        public = runner.self_hashed(
            {
                "schema": "synthetic_acquisition_public",
                "private_pack_file_sha256s": private_hashes,
            },
            "acquisition_receipt_sha256",
        )
        public_raw = runner.canonical_bytes(public)
        public_sha = _write(acquisition / "acquisition.public.json", public_raw)
        acquisition.chmod(0o500)
        freeze = {
            "acquisition_binding": {
                "public_filename": "acquisition.public.json",
                "public_file_sha256": public_sha,
                "acquisition_receipt_sha256": public[
                    "acquisition_receipt_sha256"
                ],
                "private_pack_file_sha256s": private_hashes,
            }
        }
        receipt = runner.validate_acquisition_binding(acquisition, freeze)
        assert receipt["private_pack_file_sha256s"] == private_hashes

        replaced = acquisition / "pack_3.private.json"
        replaced.write_bytes(runner.canonical_bytes({"index": 300}))
        with pytest.raises(
            runner.SetInteractionBootstrapError,
            match="private pack drifted",
        ):
            runner.validate_acquisition_binding(acquisition, freeze)


def test_output_parent_must_be_private_and_fresh() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        parent = Path(temporary) / "formal"
        parent.mkdir(mode=0o700)
        output = parent / "work"
        runner.validate_output_parent(output)
        output.mkdir()
        with pytest.raises(
            runner.SetInteractionBootstrapError,
            match="already exists",
        ):
            runner.validate_output_parent(output)

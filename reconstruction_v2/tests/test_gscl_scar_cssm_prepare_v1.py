from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import gscl_scar_cssm_source_v1 as source
from replication_runtime.gscl_scar_cssm_v1 import prepare


def _compilation() -> source.ScarCssmSourceCompilation:
    return source.ScarCssmSourceCompilation(
        action_pack={"action_commitment_sha256": "a" * 64, "items": []},
        label_pack={"items": []},
        safe_aggregate={"status": "fixture"},
    )


def test_prepare_separates_private_capabilities_and_publishes_only_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    calls = 0

    def compiler(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["secret"] == b"s" * 32
        return _compilation()

    terminal = prepare.prepare_once(
        source_path=tmp_path / "source.jsonl",
        output_root=root,
        study_id="study",
        secret_factory=lambda: b"s" * 32,
        compiler=compiler,
    )
    assert calls == 1
    assert terminal["formal_source_access_count"] == 1
    assert terminal["model_action_or_scorer_call_count"] == 0
    assert terminal["online_or_api_evaluator_call_count"] == 0
    assert (root / "compiler_secret.private.bin").read_bytes() == b"s" * 32
    assert json.loads((root / "action_pack.private.json").read_text())[
        "action_commitment_sha256"
    ] == "a" * 64
    assert terminal["secret"]["sha256"] == hashlib.sha256(b"s" * 32).hexdigest()
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in root.iterdir())
    with pytest.raises(prepare.ScarCssmPrepareError, match="PREPARE_ROOT_INVALID"):
        prepare.prepare_once(
            source_path=tmp_path / "source.jsonl",
            output_root=root,
            study_id="study",
            secret_factory=lambda: b"t" * 32,
            compiler=compiler,
        )
    assert calls == 1


def test_post_attempt_compiler_failure_is_not_retryable(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    def fail(**kwargs):
        raise RuntimeError("synthetic compiler failure")

    with pytest.raises(
        prepare.ScarCssmPrepareError,
        match="PREPARE_SOURCE_COMPILATION_FAILED",
    ):
        prepare.prepare_once(
            source_path=tmp_path / "source.jsonl",
            output_root=root,
            study_id="study",
            secret_factory=lambda: b"s" * 32,
            compiler=fail,
        )
    assert (root / "prepare.attempt.sentinel").exists()
    assert (root / "compiler_secret.private.bin").exists()
    assert not (root / "action_pack.private.json").exists()
    with pytest.raises(prepare.ScarCssmPrepareError, match="PREPARE_ROOT_INVALID"):
        prepare.prepare_once(
            source_path=tmp_path / "source.jsonl",
            output_root=root,
            study_id="study",
            secret_factory=lambda: b"s" * 32,
            compiler=lambda **kwargs: _compilation(),
        )


def test_invalid_secret_fails_before_compiler(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    called = False

    def compiler(**kwargs):
        nonlocal called
        called = True
        return _compilation()

    with pytest.raises(
        prepare.ScarCssmPrepareError,
        match="PREPARE_SECRET_GENERATION_FAILED",
    ):
        prepare.prepare_once(
            source_path=tmp_path / "source.jsonl",
            output_root=root,
            study_id="study",
            secret_factory=lambda: b"short",
            compiler=compiler,
        )
    assert called is False

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_source_epoch_rollover_v2 as rollover,
)


def _write_canonical(path: Path, value: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    os.chmod(path, mode)


def _predecessor_fixture(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    project.mkdir()
    marker = {
        "identity_full_compile_equivalence_qualification_sha256": (
            "9ab8aba9f2ba24394109ba0567233cee21e613ed5f5e39c22e5d347520117f2e"
        ),
        "implementation_freeze_sha256": (
            rollover.PREDECESSOR_IMPLEMENTATION_FREEZE_SHA256
        ),
        "marker_sha256": rollover.PREDECESSOR_MARKER_SHA256,
        "schema": "feverous_p6_e2_formal_acquisition_v1_one_shot_marker",
        "source_split": "TRAIN",
        "version": "feverous_p6_e2_formal_acquisition_v1",
    }
    failure = {
        "exception_message_sha256": (
            rollover.PREDECESSOR_FAILURE_EXCEPTION_MESSAGE_SHA256
        ),
        "exception_type": "FeverousFormalSourceError",
        "failure_sha256": rollover.PREDECESSOR_FAILURE_SHA256,
        "implementation_freeze_sha256": (
            rollover.PREDECESSOR_IMPLEMENTATION_FREEZE_SHA256
        ),
        "online_evaluator_calls": 0,
        "schema": "feverous_p6_e2_formal_acquisition_v1_terminal_failure",
        "status": "formal_acquisition_failed_no_retry_or_resample",
        "version": "feverous_p6_e2_formal_acquisition_v1",
    }
    marker_path = project / rollover.PREDECESSOR_MARKER_RELATIVE
    failure_path = project / rollover.PREDECESSOR_FAILURE_RELATIVE
    secret_path = project / rollover.PREDECESSOR_SECRET_RELATIVE
    _write_canonical(marker_path, marker)
    _write_canonical(failure_path, failure)
    secret_path.write_bytes(b"v1-secret-metadata-only-32-bytes")
    assert secret_path.stat().st_size == 32
    os.chmod(secret_path, 0o600)
    _write_canonical(
        project / rollover.MANIFEST_RELATIVE,
        rollover.form_rollover_manifest(),
        mode=0o644,
    )
    monkeypatch.setattr(
        rollover.train_loader_qualification,
        "verify_train_loader_qualification",
        lambda _project: {
            "qualification_sha256": rollover.TRAIN_LOADER_QUALIFICATION_SHA256
        },
    )
    return marker_path, failure_path, secret_path


def test_verifier_preserves_v1_and_never_reads_or_hashes_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker, failure, secret = _predecessor_fixture(tmp_path / "project", monkeypatch)
    before_public = {path: path.read_bytes() for path in (marker, failure)}
    before_secret = (secret.lstat(), secret.read_bytes())
    original_read_bytes = Path.read_bytes

    def spy_read_bytes(self: Path) -> bytes:
        if self == secret:
            pytest.fail("rollover verifier must not read predecessor secret")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
    verified = rollover.verify_rollover_manifest(tmp_path / "project")
    assert (
        verified["source_epoch_rollover_sha256"]
        == rollover.form_rollover_manifest()["source_epoch_rollover_sha256"]
    )
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
    assert {path: path.read_bytes() for path in (marker, failure)} == before_public
    after_stat = secret.lstat()
    assert secret.read_bytes() == before_secret[1]
    assert (
        after_stat.st_size,
        stat.S_IMODE(after_stat.st_mode),
        after_stat.st_mtime_ns,
    ) == (
        before_secret[0].st_size,
        stat.S_IMODE(before_secret[0].st_mode),
        before_secret[0].st_mtime_ns,
    )


def test_successor_root_or_unexpected_v1_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "present"
    _predecessor_fixture(project, monkeypatch)
    (project / rollover.SUCCESSOR_FORMAL_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(
        rollover.FeverousSourceEpochRolloverError,
        match="successor formal root already exists",
    ):
        rollover.verify_rollover_manifest(project)

    project = tmp_path / "v1-extra"
    _predecessor_fixture(project, monkeypatch)
    unexpected = project / rollover.PREDECESSOR_ABSENT_RELATIVES[0]
    unexpected.write_bytes(b"not a valid receipt")
    with pytest.raises(
        rollover.FeverousSourceEpochRolloverError,
        match="unexpectedly exists",
    ):
        rollover.verify_rollover_manifest(project)


def test_manifest_truthfully_separates_transient_decode_from_scientific_use() -> None:
    manifest = rollover.form_rollover_manifest()
    assert manifest["predecessor_raw_train_rows_transiently_json_decoded"] is True
    assert (
        manifest[
            "predecessor_raw_train_claim_label_and_evidence_fields_transiently_decoded"
        ]
        is True
    )
    assert manifest["predecessor_records_adapted_selected_or_persisted"] is False
    assert (
        manifest[
            "predecessor_cohort_pack_corpus_retrieval_utility_or_evaluator_use"
        ]
        is False
    )
    incident = manifest["predecessor_secret_pre_freeze_audit_incident"]
    assert incident["transient_diagnostic_output_observed"] is True
    assert incident["hash_value_may_exist_in_agent_tool_log"] is True
    assert incident["raw_secret_bytes_viewed_or_disclosed"] is False
    assert (
        incident[
            "hash_value_written_to_project_artifact_committed_or_used_by_successor"
        ]
        is False
    )
    assert incident["predecessor_secret_applied_to_records_selection_or_cohort"] is False
    assert incident["successor_uses_fresh_independent_os_random_secret"] is True

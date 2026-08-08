from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import hegel_machine.phase3_m3_formal_execution_cli_v1 as cli
import hegel_machine.phase3_m3_formal_execution_v1 as formal_execution
from hegel_machine.phase3_m3_local_admission_v1 import (
    LocalTwoCommitAdmissionResultV1,
    M3LocalAdmissionError,
)


RUNTIME_COMMIT_C = "c" * 40
APPROVAL_COMMIT_D = "d" * 40
PUBLICATION_COMMIT = cli.PUBLICATION_COMMIT_B


def _local_admission_result() -> LocalTwoCommitAdmissionResultV1:
    return LocalTwoCommitAdmissionResultV1(
        runtime_commit_c=RUNTIME_COMMIT_C,
        approval_commit_d=APPROVAL_COMMIT_D,
        artifact_fields=MappingProxyType(
            {
                "schema": "hegel-phase3-m3-local-two-commit-admission/1",
            }
        ),
        manifest_fields=MappingProxyType(
            {
                "schema": "hegel-phase3-m3-runtime-source-manifest/1",
                "runtime_commit": RUNTIME_COMMIT_C,
            }
        ),
        receipt_fields=MappingProxyType(
            {
                "schema": (
                    "hegel-phase3-m3-local-two-commit-admission-receipt/1"
                ),
                "claim_level": "LOCAL_TWO_COMMIT_ADMISSION",
                "receipt_sha256": "d" * 64,
            }
        ),
    )


def _patch_canonical_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "canonical-run"
    state_path = run_root / "m3-start-state.json"
    outcome_path = run_root / "m3-terminal-outcome.json"
    monkeypatch.setattr(cli, "canonical_run_root_v1", lambda run_id: run_root)
    monkeypatch.setattr(
        cli, "canonical_start_state_path_v1", lambda run_id: state_path
    )
    monkeypatch.setattr(
        cli, "canonical_terminal_outcome_path_v1", lambda run_id: outcome_path
    )
    return run_root, state_path, outcome_path


def _decode_stdout(capsys) -> tuple[dict[str, object], bytes, bytes]:
    captured = capsys.readouterr()
    raw_stdout = captured.out.encode("ascii")
    payload = json.loads(raw_stdout)
    return payload, raw_stdout, captured.err.encode("ascii")


def test_default_preflight_is_side_effect_free_and_does_not_invoke_docker(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    calls: list[object] = []
    run_root, state_path, outcome_path = _patch_canonical_paths(
        monkeypatch, tmp_path
    )

    def validate_admission(revision: str):
        calls.append(("admission", revision))
        return _local_admission_result()

    def load_blobs(*, publication_revision: str):
        calls.append(("load", publication_revision))
        return PUBLICATION_COMMIT, b"evidence", b"promotion"

    def prepare_candidate(
        evidence: bytes,
        promotion: bytes,
        *,
        publication_commit: str,
        recorded_at_unix_seconds: int,
    ):
        calls.append(
            (
                "candidate",
                evidence,
                promotion,
                publication_commit,
                recorded_at_unix_seconds,
            )
        )
        return {
            "basis_commit": "a" * 40,
            "run_id_hex": cli.FORMAL_RUN_ID_HEX,
            "state_record_root_hex": "1" * 64,
            "state_artifact_sha256": "2" * 64,
        }

    def forbidden(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("preflight crossed the formal execution boundary")

    monkeypatch.setattr(cli, "validate_live_local_admission_v1", validate_admission)
    monkeypatch.setattr(cli, "load_publication_blobs_v1", load_blobs)
    monkeypatch.setattr(cli, "prepare_m3_start_v1", prepare_candidate)
    monkeypatch.setattr(cli, "prepare_formal_execution_v1", forbidden)
    monkeypatch.setattr(cli, "execute_formal_m3_v1", forbidden)
    monkeypatch.setattr(formal_execution, "OfflineDockerEnumerationRunnerV1", forbidden)

    assert (
        cli.main(
            ["--admission-revision", APPROVAL_COMMIT_D],
            _launch_capability=cli._DIRECT_ENTRYPOINT_SEAL,
        )
        == 0
    )

    payload, raw_stdout, raw_stderr = _decode_stdout(capsys)
    assert raw_stderr == b""
    assert raw_stdout == cli.canonical_json_v1(payload)
    assert payload["action"] == "preflight"
    assert payload["runtime_source_identity"]["runtime_commit_c"] == RUNTIME_COMMIT_C
    assert payload["runtime_source_identity"]["approval_commit_d"] == (
        APPROVAL_COMMIT_D
    )
    assert payload["canonical_run_root"] == run_root.as_posix()
    assert payload["canonical_persisted_start_path"] == state_path.as_posix()
    assert payload["canonical_terminal_outcome_path"] == outcome_path.as_posix()
    assert payload["start_candidate_prepared"] is True
    assert payload["start_written"] is False
    assert payload["docker_invoked"] is False
    assert payload["formal_execution_invoked"] is False
    assert payload["state_changed"] is False
    assert not run_root.exists()
    assert [entry[0] for entry in calls] == ["admission", "load", "candidate"]
    assert calls[0] == ("admission", APPROVAL_COMMIT_D)


def test_execute_revalidates_admission_and_uses_only_canonical_paths(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    calls: list[object] = []
    run_root, state_path, outcome_path = _patch_canonical_paths(
        monkeypatch, tmp_path
    )
    prepared = object()

    def validate_admission(revision: str):
        calls.append(("admission", revision))
        return _local_admission_result()

    def load_blobs(*, publication_revision: str):
        calls.append(("load", publication_revision))
        return PUBLICATION_COMMIT, b"evidence", b"promotion"

    def prepare_execution(
        actual_state_path: Path,
        evidence: bytes,
        promotion: bytes,
        *,
        publication_commit: str,
        expected_admission_revision: str,
    ):
        calls.append(
            (
                "prepare",
                actual_state_path,
                evidence,
                promotion,
                publication_commit,
                expected_admission_revision,
            )
        )
        return prepared

    def execute_execution(
        actual_prepared: object,
        *,
        run_root: Path,
        attempt_id: str,
        outcome_path: Path,
    ):
        calls.append(
            ("execute", actual_prepared, run_root, attempt_id, outcome_path)
        )
        return SimpleNamespace(
            status="TERMINAL_PUBLISHED_NEW",
            document={
                "schema": "hegel-phase3-m3-formal-enumeration-outcome/1",
                "closure_status": "DSL_TOO_LARGE",
                "outcome_artifact_sha256": "e" * 64,
            },
        )

    def forbidden_candidate(*args, **kwargs):  # pragma: no cover
        raise AssertionError("execute reconstructed a start candidate")

    monkeypatch.setattr(cli, "validate_live_local_admission_v1", validate_admission)
    monkeypatch.setattr(cli, "load_publication_blobs_v1", load_blobs)
    monkeypatch.setattr(cli, "prepare_m3_start_v1", forbidden_candidate)
    monkeypatch.setattr(cli, "prepare_formal_execution_v1", prepare_execution)
    monkeypatch.setattr(cli, "execute_formal_m3_v1", execute_execution)

    assert (
        cli.main(
            ["execute", "--admission-revision", APPROVAL_COMMIT_D],
            _launch_capability=cli._DIRECT_ENTRYPOINT_SEAL,
        )
        == 0
    )

    payload, raw_stdout, raw_stderr = _decode_stdout(capsys)
    assert raw_stderr == b""
    assert raw_stdout == cli.canonical_json_v1(payload)
    assert payload["action"] == "execute"
    assert payload["persisted_start_read"] is True
    assert payload["formal_execution_invoked"] is True
    assert payload["publication_status"] == "TERMINAL_PUBLISHED_NEW"
    assert payload["terminal_status"] == "DSL_TOO_LARGE"
    assert payload["outcome_artifact_sha256"] == "e" * 64
    assert [entry[0] for entry in calls] == [
        "admission",
        "load",
        "prepare",
        "execute",
    ]
    assert calls[0] == ("admission", APPROVAL_COMMIT_D)
    assert calls[2][1:] == (
        state_path,
        b"evidence",
        b"promotion",
        PUBLICATION_COMMIT,
        APPROVAL_COMMIT_D,
    )
    assert calls[3][1:] == (
        prepared,
        run_root,
        cli.CANONICAL_ATTEMPT_ID,
        outcome_path,
    )


def test_cli_rejects_caller_selected_state_path_before_preflight(
    monkeypatch, capsys
) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover
        raise AssertionError("argument failure invoked runtime work")

    monkeypatch.setattr(cli, "validate_live_local_admission_v1", forbidden)

    assert (
        cli.main(
            [
                "execute",
                "--admission-revision",
                APPROVAL_COMMIT_D,
                "--state-path",
                "/tmp/attacker-selected.json",
            ],
            _launch_capability=cli._DIRECT_ENTRYPOINT_SEAL,
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert captured.err.encode("ascii") == cli.canonical_json_v1(payload)
    assert payload["ok"] is False
    assert payload["code"] == cli.FAIL_USAGE


def test_local_admission_failure_stops_before_public_blob_replay(
    monkeypatch, capsys
) -> None:
    def admission_failure(revision: str):
        raise M3LocalAdmissionError(
            "FAIL_M3_RUNTIME_ADMISSION_WORKTREE", "runtime source differs"
        )

    def forbidden(*args, **kwargs):  # pragma: no cover
        raise AssertionError("failed source preflight crossed the trust boundary")

    monkeypatch.setattr(cli, "validate_live_local_admission_v1", admission_failure)
    monkeypatch.setattr(cli, "load_publication_blobs_v1", forbidden)
    monkeypatch.setattr(cli, "prepare_formal_execution_v1", forbidden)
    monkeypatch.setattr(cli, "execute_formal_m3_v1", forbidden)

    assert (
        cli.main(
            ["execute", "--admission-revision", APPROVAL_COMMIT_D],
            _launch_capability=cli._DIRECT_ENTRYPOINT_SEAL,
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert captured.err.encode("ascii") == cli.canonical_json_v1(payload)
    assert payload == {
        "schema": cli.SCHEMA,
        "ok": False,
        "code": "FAIL_M3_RUNTIME_ADMISSION_WORKTREE",
        "detail": "runtime source differs",
    }

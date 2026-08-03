from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from hegel_machine.hashing import stable_hash
from hegel_machine import phase3_m25_secret_absence_v1 as absence
from hegel_machine.phase3_m25_secret_absence_v1 import (
    FAIL_STATUS,
    GenesisSecretAbsenceError,
    PASS_STATUS,
    repository_genesis_secret_absence_report,
    validate_repository_genesis_secret_absence_report,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "audit-repository"
    project = repository / "Hegel Machine"
    project.mkdir(parents=True)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "M25 Test")
    _git(repository, "config", "user.email", "m25@example.invalid")
    return repository, project


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "-m", message)
    commit_id = _git(repository, "rev-parse", "HEAD")
    assert len(commit_id) == 40
    return commit_id


def _synthetic_payload(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_kind": "SYNTHETIC_NON_AUTHORITATIVE",
        "machine_freeze_id": "hegel-freeze-p2b-p3-v1.1.2",
        "authority_boundary": {
            "contains_real_secret_material": False,
            "authoritative_root_generation": False,
            "seed_genesis_performed": False,
            "signature_claim": False,
        },
    }
    payload.update(extra)
    return payload


def test_clean_history_passes_twice_and_ignores_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    (project / "golden_vectors").mkdir()
    (project / "golden_vectors" / "synthetic_fixture.key").write_text(
        json.dumps(_synthetic_payload()), encoding="utf-8"
    )
    (project / "public.json").write_text(
        json.dumps({"private_key": None, "status": "public"}),
        encoding="utf-8",
    )
    commit_id = _commit(repository, "clean synthetic history")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)

    # Dirty bytes are deliberately outside the committed-object audit scope.
    (project / "untracked_private.key").write_text("not committed", encoding="utf-8")
    report = repository_genesis_secret_absence_report(commit_id)
    explicit_report = (
        absence.repository_genesis_secret_absence_report_for_repository_v1(
            repository, project, commit_id
        )
    )
    assert explicit_report == report
    validate_repository_genesis_secret_absence_report(
        report, expected_commit_id=commit_id
    )

    assert report["status"] == PASS_STATUS
    assert report["pass"] is True
    assert report["zero_findings"] is True
    assert report["findings"] == []
    assert report["counts"]["ancestor_commit_count"] == 1
    assert report["counts"]["unique_blob_count"] == 2
    assert report["counts"]["synthetic_vector_path_exemption_count"] == 1
    assert report["immediate_second_replay_equal"] is True
    assert report["scope"]["working_tree_consulted"] is False
    assert report["authority_boundary"]["universal_secret_detection_claim"] is False
    assert report["diagnostic_report_id"].startswith("phase3_m25_secret_absence_")


def test_deleted_ancestor_artifacts_remain_findings_and_values_are_not_disclosed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    archive = project / "archive"
    golden = project / "golden_vectors"
    archive.mkdir()
    golden.mkdir()
    (archive / "private.pem").write_text("extension-only hit", encoding="utf-8")
    private_header = b"-----BEGIN " + b"PRIVATE KEY-----"
    (archive / "key.txt").write_bytes(private_header + b"\nredacted-test-body\n")
    (archive / "config.json").write_text(
        json.dumps({"nested": {"privateKey": "DO_NOT_DISCLOSE_VALUE"}}),
        encoding="utf-8",
    )
    (golden / "synthetic_bad.key").write_text(
        json.dumps(_synthetic_payload(private_key="SYNTHETIC_VALUE_NOT_DISCLOSED")),
        encoding="utf-8",
    )
    (golden / "synthetic_header.key").write_bytes(private_header + b"\n")
    _commit(repository, "ancestor with forbidden artifacts")

    for path in tuple(archive.iterdir()) + tuple(golden.iterdir()):
        path.unlink()
    (project / "clean.txt").write_text("clean current tree\n", encoding="utf-8")
    commit_id = _commit(repository, "remove forbidden artifacts")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)

    report = repository_genesis_secret_absence_report(commit_id)
    validate_repository_genesis_secret_absence_report(
        report, expected_commit_id=commit_id
    )

    assert report["status"] == FAIL_STATUS
    assert report["pass"] is False
    assert report["zero_findings"] is False
    assert report["counts"]["ancestor_commit_count"] == 2
    codes = [finding["finding_code"] for finding in report["findings"]]
    assert "FORBIDDEN_SECRET_FILENAME_OR_EXTENSION" in codes
    assert "PRIVATE_KEY_MAGIC_HEADER" in codes
    assert "NON_NULL_FORBIDDEN_JSON_SECRET_KEY" in codes
    assert report["counts"]["synthetic_vector_path_exemption_count"] == 1

    serialized = json.dumps(report, sort_keys=True)
    assert "DO_NOT_DISCLOSE_VALUE" not in serialized
    assert "SYNTHETIC_VALUE_NOT_DISCLOSED" not in serialized
    assert private_header.decode("ascii") not in serialized
    assert all(
        set(finding) == {
            "finding_code",
            "blob_oid",
            "blob_byte_length",
            "repository_paths",
            "path_sha256_or_null",
            "policy_token",
            "json_location_sha256_or_null",
        }
        for finding in report["findings"]
    )


def test_side_parent_only_secret_remains_a_merge_history_finding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    (project / "public.txt").write_text("base\n", encoding="utf-8")
    _commit(repository, "base")

    _git(repository, "checkout", "-q", "-b", "side-secret")
    secret_path = project / "side-only-private.pem"
    secret_path.write_bytes(b"-----BEGIN " + b"PRIVATE KEY-----\n")
    _commit(repository, "side parent contains key material")
    secret_path.unlink()
    _commit(repository, "side parent removes key material")

    _git(repository, "checkout", "-q", "master")
    (project / "public.txt").write_text("main\n", encoding="utf-8")
    _commit(repository, "main diverges")
    _git(repository, "merge", "--no-ff", "-q", "side-secret", "-m", "merge side")
    merge_commit = _git(repository, "rev-parse", "HEAD")
    assert not secret_path.exists()
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)

    report = repository_genesis_secret_absence_report(merge_commit)

    assert report["pass"] is False
    findings = [
        finding
        for finding in report["findings"]
        if finding["finding_code"] == "PRIVATE_KEY_MAGIC_HEADER"
    ]
    assert len(findings) == 1
    assert findings[0]["repository_paths"] == [
        "Hegel Machine/side-only-private.pem"
    ]
    assert report["counts"]["ancestor_commit_count"] >= 5


def test_private_key_magic_header_is_record_anchored_not_a_midline_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    private_header = b"-----BEGIN " + b"PRIVATE KEY-----"
    (project / "source_example.py").write_bytes(
        b'assert encoded.startswith(b"' + private_header + b'")\n'
    )
    literal_commit = _commit(repository, "midline source literal")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)

    literal_report = repository_genesis_secret_absence_report(literal_commit)
    assert literal_report["pass"] is True
    assert literal_report["findings"] == []
    assert literal_report["policy"]["private_key_magic_header_match_rule"] == (
        "header begins at byte offset zero or after CR or LF plus optional "
        "horizontal ASCII whitespace; an isolated mid-record PEM/OpenPGP "
        "header is not a finding under this record-only rule"
    )
    assert literal_report["policy"]["private_key_complete_block_match_rule"] == (
        "a PEM/OpenPGP header followed by its matching footer is a finding "
        "at any byte offsets, including escaped structured-text strings; "
        "age, PuTTY, and binary OpenSSH magic are findings at any offset"
    )
    assert literal_report["policy"]["private_key_magic_header_residual_boundary"] == (
        "an isolated inline PEM/OpenPGP header example without a matching "
        "footer is outside the header/block rules unless another frozen "
        "filename or JSON-key rule independently matches"
    )

    # The first mention is still mid-record.  The second starts a CR-delimited
    # record and must be detected, proving that matching continues after a
    # harmless earlier occurrence in the same blob.
    (project / "key_material.txt").write_bytes(
        b"quoted=" + private_header + b"\r\t  " + private_header + b"\n"
    )
    key_commit = _commit(repository, "record-start private key header")
    key_report = repository_genesis_secret_absence_report(key_commit)

    assert key_report["pass"] is False
    header_findings = [
        finding
        for finding in key_report["findings"]
        if finding["finding_code"] == "PRIVATE_KEY_MAGIC_HEADER"
    ]
    assert len(header_findings) == 1
    assert header_findings[0]["policy_token"] == "PKCS8_PEM"
    assert header_findings[0]["repository_paths"] == [
        "Hegel Machine/key_material.txt"
    ]


def test_complete_private_key_blocks_are_detected_inside_structured_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    header = b"-----BEGIN " + b"PRIVATE KEY-----"
    footer = b"-----END " + b"PRIVATE KEY-----"
    embedded = header + b"\\nTEST-BODY\\n" + footer
    (project / "public.json").write_text(
        json.dumps({"data": embedded.decode("ascii")}), encoding="utf-8"
    )
    (project / "public.yaml").write_bytes(b'payload: "' + embedded + b'"\n')
    (project / "public.md").write_bytes(b"inline `" + embedded + b"`\n")
    commit_id = _commit(repository, "embedded complete key blocks")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)

    report = repository_genesis_secret_absence_report(commit_id)

    assert report["pass"] is False
    header_findings = [
        finding
        for finding in report["findings"]
        if finding["finding_code"] == "PRIVATE_KEY_MAGIC_HEADER"
    ]
    assert len(header_findings) == 3
    assert {
        tuple(finding["repository_paths"]) for finding in header_findings
    } == {
        ("Hegel Machine/public.json",),
        ("Hegel Machine/public.md",),
        ("Hegel Machine/public.yaml",),
    }


def test_history_audit_rejects_graft_metadata_instead_of_truncating_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    secret = project / "ancestor.pem"
    secret.write_bytes(b"-----BEGIN " + b"PRIVATE KEY-----\n")
    _commit(repository, "secret ancestor")
    secret.unlink()
    (project / "clean.txt").write_text("clean\n", encoding="utf-8")
    head = _commit(repository, "clean child")
    grafts = repository / ".git/info/grafts"
    grafts.parent.mkdir(parents=True, exist_ok=True)
    grafts.write_text(head + "\n", encoding="ascii")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)

    with pytest.raises(GenesisSecretAbsenceError, match="graft metadata"):
        repository_genesis_secret_absence_report(head)


def test_history_audit_rejects_shallow_and_promisor_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    (project / "base.txt").write_text("base\n", encoding="utf-8")
    _commit(repository, "base")
    (project / "child.txt").write_text("child\n", encoding="utf-8")
    head = _commit(repository, "child")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)

    shallow = repository / ".git/shallow"
    shallow.write_text(head + "\n", encoding="ascii")
    with pytest.raises(GenesisSecretAbsenceError, match="shallow repository"):
        repository_genesis_secret_absence_report(head)
    shallow.unlink()

    _git(repository, "config", "extensions.partialClone", "origin")
    with pytest.raises(GenesisSecretAbsenceError, match="promisor or partial-clone"):
        repository_genesis_secret_absence_report(head)


def test_invalid_json_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    (project / "broken.json").write_text('{"private_key":', encoding="utf-8")
    commit_id = _commit(repository, "invalid json")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)
    report = repository_genesis_secret_absence_report(commit_id)
    assert report["pass"] is False
    assert [item["finding_code"] for item in report["findings"]] == [
        "UNSCANNABLE_JSON_BLOB"
    ]


def test_validator_rejects_bool_int_confusion_self_hash_and_commit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project = _repository(tmp_path)
    (project / "clean.txt").write_text("clean\n", encoding="utf-8")
    commit_id = _commit(repository, "clean")
    monkeypatch.setattr(absence, "PROJECT_ROOT", project)
    report = repository_genesis_secret_absence_report(commit_id)

    mutated = deepcopy(report)
    mutated["pass"] = 1
    mutated.pop("diagnostic_report_id")
    mutated["diagnostic_report_id"] = stable_hash(
        mutated, prefix="phase3_m25_secret_absence_"
    )
    with pytest.raises(GenesisSecretAbsenceError, match="must be bool"):
        validate_repository_genesis_secret_absence_report(
            mutated, expected_commit_id=commit_id
        )

    mutated = deepcopy(report)
    mutated["status"] = FAIL_STATUS
    with pytest.raises(GenesisSecretAbsenceError, match="differs"):
        validate_repository_genesis_secret_absence_report(
            mutated, expected_commit_id=commit_id
        )

    with pytest.raises(GenesisSecretAbsenceError, match="commit mismatch"):
        validate_repository_genesis_secret_absence_report(
            report, expected_commit_id="0" * 40
        )


def test_policy_source_does_not_embed_the_magic_headers_it_scans() -> None:
    source = Path(absence.__file__).read_bytes()
    assert all(header not in source for header in absence.PRIVATE_KEY_MAGIC_HEADERS.values())

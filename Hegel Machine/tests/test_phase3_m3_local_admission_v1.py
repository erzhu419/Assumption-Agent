from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess

import pytest

import hegel_machine.phase3_m3_local_admission_v1 as admission


@dataclass(frozen=True, slots=True)
class CommitCFixture:
    repository: Path
    basis_a: str
    publication_b: str
    runtime_c: str
    runtime_paths: tuple[str, ...]
    built: admission.BuiltLocalAdmissionArtifactV1


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed.stdout


def _write(repository: Path, repository_path: str, payload: bytes) -> Path:
    path = repository / repository_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o644)
    return path


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "-A")
    _git(
        repository,
        "-c",
        "user.name=Hegel Local Admission Test",
        "-c",
        "user.email=hegel-local-admission@example.invalid",
        "commit",
        "--no-gpg-sign",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD").decode("ascii").strip()


@pytest.fixture
def commit_c_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> CommitCFixture:
    repository = (tmp_path / "local-admission-repository").resolve()
    repository.mkdir(mode=0o700)
    _git(repository, "init", "-q")

    _write(repository, "history.txt", b"basis-a\n")
    basis_a = _commit(repository, "basis A")
    _write(repository, "history.txt", b"publication-b\n")
    publication_b = _commit(repository, "publication B")

    runtime_paths = tuple(
        sorted(
            (
                admission.ADMISSION_MODULE_REPOSITORY_PATH,
                *admission.DIRECT_ENTRYPOINT_PATHS,
                "Hegel Machine/src/hegel_machine/runtime_helper_v1.py",
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    for index, repository_path in enumerate(runtime_paths, start=1):
        _write(
            repository,
            repository_path,
            f"# committed runtime source {index}\nVALUE = {index}\n".encode("ascii"),
        )
    runtime_c = _commit(repository, "runtime C")

    monkeypatch.setattr(admission, "BASIS_COMMIT_A", basis_a)
    monkeypatch.setattr(admission, "PUBLICATION_COMMIT_B", publication_b)
    monkeypatch.setattr(admission, "M3_RUNTIME_SOURCE_PATHS", runtime_paths)
    built = admission.build_local_admission_artifact_v1(
        runtime_c,
        repository_root=repository,
    )
    return CommitCFixture(
        repository=repository,
        basis_a=basis_a,
        publication_b=publication_b,
        runtime_c=runtime_c,
        runtime_paths=runtime_paths,
        built=built,
    )


def _publish_commit_d(
    fixture: CommitCFixture,
    payload: bytes | None = None,
) -> str:
    _write(
        fixture.repository,
        admission.APPROVAL_REPOSITORY_PATH,
        fixture.built.canonical_bytes if payload is None else payload,
    )
    return _commit(fixture.repository, "approval D")


def test_build_then_validate_exact_local_two_commit_admission(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    assert fixture.built.runtime_commit_c == fixture.runtime_c
    assert fixture.built.fields["claim_level"] == "LOCAL_TWO_COMMIT_ADMISSION"
    assert fixture.built.fields["external_actor_attestation"] is False
    assert fixture.built.fields["external_signatures"] == ()
    assert fixture.built.fields["network_fetch_allowed"] is False
    assert fixture.built.fields["docker_pull_allowed"] is False
    assert fixture.built.canonical_bytes == admission.canonical_json_v1(
        fixture.built.fields
    )

    approval_d = _publish_commit_d(fixture)
    result = admission.validate_live_local_admission_v1(
        approval_d,
        repository_root=fixture.repository,
    )
    assert result.runtime_commit_c == fixture.runtime_c
    assert result.approval_commit_d == approval_d
    assert result.artifact_fields == fixture.built.fields
    assert result.receipt_fields["head_commit"] == approval_d
    assert result.receipt_fields["commit_d_adds_only_approval_artifact"] is True
    assert result.receipt_fields["runtime_paths_equal_c_d_index_worktree"] is True
    assert result.receipt_fields["claim_level"] == admission.CLAIM_LEVEL
    assert isinstance(result.artifact_fields["runtime_source_paths"], tuple)
    assert isinstance(result.manifest_fields["runtime_source_files"], tuple)
    with pytest.raises(AttributeError):
        result.artifact_fields["external_signatures"].append("forged")
    with pytest.raises(TypeError):
        result.manifest_fields["runtime_source_files"][0]["sha256"] = "00" * 32
    admission.validate_local_admission_receipt_v1(
        result.receipt_fields,
        artifact_fields=result.artifact_fields,
        manifest_fields=result.manifest_fields,
    )


def test_live_validation_requires_explicit_full_d_and_head_equals_d(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    approval_d = _publish_commit_d(fixture)
    with pytest.raises(admission.M3LocalAdmissionError) as symbolic:
        admission.validate_live_local_admission_v1(
            "HEAD",
            repository_root=fixture.repository,
        )
    assert symbolic.value.code == admission.FAIL_HEAD

    _write(fixture.repository, "after-d.txt", b"later local work\n")
    _commit(fixture.repository, "later E")
    with pytest.raises(admission.M3LocalAdmissionError) as stale:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert stale.value.code == admission.FAIL_HEAD


@pytest.mark.parametrize("extra_kind", ("runtime-change", "second-path"))
def test_commit_d_must_add_only_the_fixed_approval_artifact(
    commit_c_fixture: CommitCFixture,
    extra_kind: str,
) -> None:
    fixture = commit_c_fixture
    _write(
        fixture.repository,
        admission.APPROVAL_REPOSITORY_PATH,
        fixture.built.canonical_bytes,
    )
    if extra_kind == "runtime-change":
        _write(
            fixture.repository,
            fixture.runtime_paths[-1],
            b"# changed in forbidden Commit D\n",
        )
    else:
        _write(fixture.repository, "unexpected-d-path.txt", b"forbidden\n")
    approval_d = _commit(fixture.repository, "invalid approval D")
    with pytest.raises(admission.M3LocalAdmissionError) as captured:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert captured.value.code == admission.FAIL_DIFF


def test_merge_commit_cannot_serve_as_commit_d(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    _git(fixture.repository, "checkout", "-q", "-b", "side", fixture.runtime_c)
    _write(fixture.repository, "side.txt", b"second parent\n")
    side = _commit(fixture.repository, "side parent")
    assert side != fixture.runtime_c
    _git(fixture.repository, "checkout", "-q", fixture.runtime_c)
    _publish_commit_d(fixture)
    _git(
        fixture.repository,
        "-c",
        "user.name=Hegel Local Admission Test",
        "-c",
        "user.email=hegel-local-admission@example.invalid",
        "merge",
        "--no-ff",
        "--no-gpg-sign",
        "-m",
        "merge approval",
        "side",
    )
    merge_d = _git(fixture.repository, "rev-parse", "HEAD").decode("ascii").strip()
    with pytest.raises(admission.M3LocalAdmissionError) as captured:
        admission.validate_live_local_admission_v1(
            merge_d,
            repository_root=fixture.repository,
        )
    assert captured.value.code == admission.FAIL_TOPOLOGY


@pytest.mark.parametrize("encoding", ("pretty", "duplicate"))
def test_approval_blob_must_be_unique_canonical_json(
    commit_c_fixture: CommitCFixture,
    encoding: str,
) -> None:
    fixture = commit_c_fixture
    if encoding == "pretty":
        payload = (
            json.dumps(dict(fixture.built.fields), indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
    else:
        payload = b'{"schema":"duplicate",' + fixture.built.canonical_bytes[1:]
    approval_d = _publish_commit_d(fixture, payload)
    with pytest.raises(admission.M3LocalAdmissionError) as captured:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert captured.value.code == admission.FAIL_APPROVAL_CANONICAL


def test_self_consistent_but_wrong_artifact_is_rejected(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    wrong = dict(fixture.built.fields)
    wrong["network_fetch_allowed"] = True
    body = dict(wrong)
    body.pop("approval_artifact_sha256")
    wrong["approval_artifact_sha256"] = admission._domain_hash(
        admission.ARTIFACT_HASH_DOMAIN,
        body,
    )
    approval_d = _publish_commit_d(fixture, admission.canonical_json_v1(wrong))
    with pytest.raises(admission.M3LocalAdmissionError) as captured:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert captured.value.code == admission.FAIL_ARTIFACT_BINDING


def test_staged_runtime_substitution_is_rejected(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    approval_d = _publish_commit_d(fixture)
    _write(fixture.repository, fixture.runtime_paths[-1], b"staged substitution\n")
    _git(fixture.repository, "add", "--", fixture.runtime_paths[-1])
    with pytest.raises(admission.M3LocalAdmissionError) as captured:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert captured.value.code == admission.FAIL_INDEX


def test_unstaged_runtime_substitution_is_rejected(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    approval_d = _publish_commit_d(fixture)
    _write(fixture.repository, fixture.runtime_paths[-1], b"unstaged substitution\n")
    with pytest.raises(admission.M3LocalAdmissionError) as captured:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert captured.value.code == admission.FAIL_WORKTREE


def test_symlink_hardlink_and_mode_substitutions_are_rejected(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    approval_d = _publish_commit_d(fixture)
    runtime_path = fixture.repository / fixture.runtime_paths[-1]
    original = runtime_path.read_bytes()

    runtime_path.unlink()
    runtime_path.symlink_to(fixture.repository / "history.txt")
    with pytest.raises(admission.M3LocalAdmissionError) as symlinked:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert symlinked.value.code == admission.FAIL_SYMLINK

    runtime_path.unlink()
    runtime_path.write_bytes(original)
    runtime_path.chmod(0o644)
    hardlink = fixture.repository / "runtime-hardlink"
    os.link(runtime_path, hardlink)
    with pytest.raises(admission.M3LocalAdmissionError) as linked:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert linked.value.code == admission.FAIL_WORKTREE

    hardlink.unlink()
    runtime_path.chmod(0o600)
    with pytest.raises(admission.M3LocalAdmissionError) as wrong_mode:
        admission.validate_live_local_admission_v1(
            approval_d,
            repository_root=fixture.repository,
        )
    assert wrong_mode.value.code == admission.FAIL_WORKTREE


def test_builder_rejects_preexisting_approval_path_and_non_child_c(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    _write(
        fixture.repository,
        admission.APPROVAL_REPOSITORY_PATH,
        fixture.built.canonical_bytes,
    )
    with pytest.raises(admission.M3LocalAdmissionError) as preexisting:
        admission.build_local_admission_artifact_v1(
            fixture.runtime_c,
            repository_root=fixture.repository,
        )
    assert preexisting.value.code == admission.FAIL_DIFF

    (fixture.repository / admission.APPROVAL_REPOSITORY_PATH).unlink()
    _write(fixture.repository, "post-c.txt", b"not direct child of B\n")
    later = _commit(fixture.repository, "post-C runtime candidate")
    with pytest.raises(admission.M3LocalAdmissionError) as topology:
        admission.build_local_admission_artifact_v1(
            later,
            repository_root=fixture.repository,
        )
    assert topology.value.code == admission.FAIL_TOPOLOGY


def test_receipt_replay_rejects_claim_inflation(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    approval_d = _publish_commit_d(fixture)
    result = admission.validate_live_local_admission_v1(
        approval_d,
        repository_root=fixture.repository,
    )
    inflated = dict(result.receipt_fields)
    inflated["external_actor_attestation"] = True
    body = dict(inflated)
    body.pop("receipt_sha256")
    inflated["receipt_sha256"] = admission._domain_hash(
        admission.RECEIPT_HASH_DOMAIN,
        body,
    )
    with pytest.raises(admission.M3LocalAdmissionError) as captured:
        admission.validate_local_admission_receipt_v1(
            inflated,
            artifact_fields=result.artifact_fields,
            manifest_fields=result.manifest_fields,
        )
    assert captured.value.code == admission.FAIL_RECEIPT

    inflated_artifact = dict(result.artifact_fields)
    inflated_artifact["external_actor_attestation"] = True
    artifact_body = dict(inflated_artifact)
    artifact_body.pop("approval_artifact_sha256")
    inflated_artifact["approval_artifact_sha256"] = admission._domain_hash(
        admission.ARTIFACT_HASH_DOMAIN,
        artifact_body,
    )
    with pytest.raises(admission.M3LocalAdmissionError) as artifact_claim:
        admission.validate_local_admission_receipt_v1(
            result.receipt_fields,
            artifact_fields=inflated_artifact,
            manifest_fields=result.manifest_fields,
        )
    assert artifact_claim.value.code == admission.FAIL_ARTIFACT_BINDING


def test_repository_local_fsmonitor_cannot_execute_during_validation(
    commit_c_fixture: CommitCFixture,
) -> None:
    fixture = commit_c_fixture
    approval_d = _publish_commit_d(fixture)
    marker = fixture.repository / "fsmonitor-was-invoked"
    hook = _write(
        fixture.repository,
        "malicious-fsmonitor.sh",
        (
            "#!/bin/sh\n"
            f"printf invoked > '{marker.as_posix()}'\n"
            "exit 0\n"
        ).encode("utf-8"),
    )
    hook.chmod(0o755)
    _git(fixture.repository, "config", "core.fsmonitor", hook.as_posix())
    result = admission.validate_live_local_admission_v1(
        approval_d,
        repository_root=fixture.repository,
    )
    assert result.approval_commit_d == approval_d
    assert not marker.exists()

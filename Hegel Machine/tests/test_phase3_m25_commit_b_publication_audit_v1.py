from __future__ import annotations

from dataclasses import fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

import hegel_machine.phase3_m25_commit_b_publication_audit_v1 as audit
from hegel_machine.phase3_container_actor_runtime_v1 import (
    TECHNICAL_ACTOR_DISCLOSURE_V1,
)


_TEST_ACTOR_IMAGE_REF = "python@sha256:" + "2" * 64
_PKCS8_HEADER = b"-----BEGIN PRIVATE " + b"KEY-----"
_PKCS8_FOOTER = b"-----END PRIVATE " + b"KEY-----"


def _complete_pkcs8_block() -> bytes:
    return _PKCS8_HEADER + b"\nAAAA\n" + _PKCS8_FOOTER + b"\n"


def _quoted_complete_pkcs8_note() -> str:
    return (
        "quoted "
        + _PKCS8_HEADER.decode("ascii")
        + "\nAAAA\n"
        + _PKCS8_FOOTER.decode("ascii")
        + " block"
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Publication Test")
    _git(repository, "config", "user.email", "publication@example.invalid")
    (repository / "basis.txt").write_text("basis\n", encoding="ascii")
    profile = repository / audit.PROFILE_REPOSITORY_PATH
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_bytes(
        audit.canonical_json_v1(
            {"images": {audit.PYTHON_IMAGE_KEY: _TEST_ACTOR_IMAGE_REF}}
        )
    )
    _git(repository, "add", "basis.txt", audit.PROFILE_REPOSITORY_PATH)
    _git(repository, "commit", "--quiet", "-m", "basis")
    return repository, _git(repository, "rev-parse", "HEAD")


def _stage(repository: Path, relative: str, payload: bytes, mode: int = 0o644) -> None:
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    _git(repository, "add", "--", relative)


def _public_payload(relative: str) -> bytes:
    if relative.endswith(".md"):
        return b"# External status\n\nNOT_RUN\n"
    value = {"role": audit.PUBLICATION_ROLE_REGISTRY[relative], "pass": True}
    if relative in audit.CANONICAL_JSON_REQUIRED_PATHS:
        return audit.canonical_json_v1(value)
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "ascii"
    )


def _stage_required(
    repository: Path,
    *,
    include_receipt: bool = False,
    overrides: dict[str, tuple[bytes, int]] | None = None,
) -> None:
    overrides = {} if overrides is None else overrides
    for relative in sorted(audit.PUBLICATION_ROLE_REGISTRY):
        if relative == audit.AUDIT_RECEIPT_REPOSITORY_PATH and not include_receipt:
            continue
        payload, mode = overrides.get(relative, (_public_payload(relative), 0o644))
        _stage(repository, relative, payload, mode)


def _write_status_inputs(repository: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for relative in sorted(
        set(audit.PUBLICATION_ROLE_REGISTRY)
        - {
            audit.AUDIT_RECEIPT_REPOSITORY_PATH,
            audit.EXTERNAL_STATUS_REPOSITORY_PATH,
        }
    ):
        payload = _public_payload(relative)
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o644)
        payloads[relative] = payload
    (repository / audit.EXTERNAL_STATUS_REPOSITORY_PATH).parent.mkdir(
        parents=True, exist_ok=True
    )
    return payloads


def test_status_renderer_reads_exact_ten_real_worktree_inputs(tmp_path: Path) -> None:
    repository, basis = _repository(tmp_path)
    payloads = _write_status_inputs(repository)

    rendered = audit.build_external_status_from_worktree_v1(
        repository=repository, basis_commit=basis
    )

    assert rendered == audit.render_external_status_v1(
        basis_commit=basis, files=payloads
    )
    assert b"formal_gates: `24/24`" in rendered
    assert b"child_state: `NOT_RUN`" in rendered


def test_commit_only_source_digest_reads_basis_blobs_not_dirty_worktree(
    tmp_path: Path,
) -> None:
    repository, basis = _repository(tmp_path)
    domain = b"HEGEL/TEST/COMMIT_ONLY_SOURCE_SET/V1\x00"
    path = b"basis.txt"
    expected = hashlib.sha256(domain)
    expected.update(len(path).to_bytes(4, "big"))
    expected.update(path)
    expected.update(hashlib.sha256(b"basis\n").digest())

    first = audit._commit_source_set_digest_v1(
        repository, basis, ("basis.txt",), domain=domain
    )
    (repository / "basis.txt").write_text("dirty worktree\n", encoding="ascii")
    second = audit._commit_source_set_digest_v1(
        repository, basis, ("basis.txt",), domain=domain
    )

    assert first == ("sha256:" + expected.hexdigest(), 1)
    assert second == first


def test_live_source_paths_and_digest_come_only_from_supplied_basis_tree(
    tmp_path: Path,
) -> None:
    repository, _initial = _repository(tmp_path)
    committed_paths = set(audit.LIVE_PROTOCOL_FIXED_NONPACKAGE_SOURCE_PATHS_V1)
    committed_paths.update(audit.LIVE_PROTOCOL_REQUIRED_PACKAGE_PATHS_V1)
    committed_paths.add(
        audit.LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1 + "/basis_only_module.py"
    )
    # A nested Python file was not part of the original direct ``glob('*.py')``
    # source-set rule and must remain excluded.
    committed_paths.add(
        audit.LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1 + "/nested/not_top_level.py"
    )
    for relative in sorted(committed_paths):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((relative + "\n").encode("utf-8"))
    _git(repository, "add", "--", *sorted(committed_paths))
    _git(repository, "commit", "--quiet", "-m", "basis live source set")
    basis = _git(repository, "rev-parse", "HEAD")

    first_paths = audit._live_protocol_source_paths_v1(repository, basis)
    first_digest = audit._commit_source_set_digest_v1(
        repository,
        basis,
        first_paths,
        domain=b"HEGEL/TEST/LIVE_SOURCE_SET/V1\x00",
    )
    untracked = (
        repository
        / audit.LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1
        / "verifier_checkout_only.py"
    )
    untracked.write_text("not in supplied basis\n", encoding="utf-8")

    second_paths = audit._live_protocol_source_paths_v1(repository, basis)
    second_digest = audit._commit_source_set_digest_v1(
        repository,
        basis,
        second_paths,
        domain=b"HEGEL/TEST/LIVE_SOURCE_SET/V1\x00",
    )

    assert second_paths == first_paths
    assert second_digest == first_digest
    assert untracked.relative_to(repository).as_posix() not in second_paths
    assert (
        audit.LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1 + "/nested/not_top_level.py"
        not in second_paths
    )


def test_actor_image_ref_reads_supplied_basis_profile_not_worktree(tmp_path: Path) -> None:
    repository, _basis = _repository(tmp_path)
    committed_ref = "python@sha256:" + "1" * 64
    profile = repository / audit.PROFILE_REPOSITORY_PATH
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_bytes(
        audit.canonical_json_v1(
            {"images": {audit.PYTHON_IMAGE_KEY: committed_ref}}
        )
    )
    _git(repository, "add", "--", audit.PROFILE_REPOSITORY_PATH)
    _git(repository, "commit", "--quiet", "-m", "basis profile")
    basis = _git(repository, "rev-parse", "HEAD")
    profile.write_text("not JSON\n", encoding="ascii")

    assert audit._load_image_ref(repository, basis) == committed_ref


def test_commit_only_actor_replay_binds_inputs_to_supplied_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hegel_machine.phase3_container_actor_runtime_v1 as actor_module

    repository, _basis = _repository(tmp_path)
    paths = {
        "profile": audit.PROFILE_REPOSITORY_PATH,
        "seccomp": "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
        "python_probe": "Hegel Machine/tools/phase3_container_actor_probe_v1.py",
        "rust_probe": "Hegel Machine/tools/phase3_container_actor_probe_v1.rs",
        "supervisor_runtime": (
            "Hegel Machine/src/hegel_machine/phase3_container_actor_runtime_v1.py"
        ),
    }
    payloads: dict[str, bytes] = {}
    for role, relative in paths.items():
        payload = (
            audit.canonical_json_v1({"profile_id": "committed-profile"})
            if role == "profile"
            else f"{role} committed\n".encode("ascii")
        )
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        payloads[role] = payload
    _git(repository, "add", "--", *paths.values())
    _git(repository, "commit", "--quiet", "-m", "actor basis inputs")
    basis = _git(repository, "rev-parse", "HEAD")
    bindings: dict[str, object] = {}
    for role, relative in paths.items():
        payload = payloads[role]
        blob_sha1 = hashlib.sha1(
            b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
        ).hexdigest()
        bindings[role] = {
            "repository_path": relative,
            "byte_length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob_sha1": blob_sha1,
            "basis_tree_blob_sha1_or_null": blob_sha1,
            "basis_commit_matches": True,
        }
    report: dict[str, object] = {
        "basis_commit": basis,
        "basis_commit_contains_all_inputs": True,
        "input_bindings": bindings,
    }
    observed: dict[str, object] = {}

    def fake_validate(value: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return dict(value)  # type: ignore[arg-type]

    monkeypatch.setattr(actor_module, "validate_qualification_report", fake_validate)
    for relative in paths.values():
        (repository / relative).write_text("dirty worktree\n", encoding="ascii")

    validated = audit._validate_actor_report_public_only_v1(
        report, repository=repository, basis_commit=basis
    )

    assert validated == report
    assert observed["profile_override"] == {"profile_id": "committed-profile"}


def test_archived_zero_secret_receipt_rejected_when_ancestor_contains_secret(
    tmp_path: Path,
) -> None:
    import hegel_machine.phase3_m25_secret_absence_v1 as secret_module

    repository, _basis = _repository(tmp_path)
    secret = repository / "Hegel Machine/ancestor-private.pem"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(_complete_pkcs8_block())
    _git(repository, "add", "--", "Hegel Machine/ancestor-private.pem")
    _git(repository, "commit", "--quiet", "-m", "ancestor secret")
    _git(repository, "rm", "--quiet", "--", "Hegel Machine/ancestor-private.pem")
    _git(repository, "commit", "--quiet", "-m", "delete visible secret")
    basis = _git(repository, "rev-parse", "HEAD")
    actual = secret_module.repository_genesis_secret_absence_report_for_repository_v1(
        repository, repository / "Hegel Machine", basis
    )
    assert actual["pass"] is False
    forged = json.loads(json.dumps(actual))
    forged.update(
        {
            "status": secret_module.PASS_STATUS,
            "pass": True,
            "zero_findings": True,
            "findings": [],
        }
    )
    forged["counts"]["finding_count"] = 0
    forged["counts"]["offending_unique_blob_count"] = 0
    forged.pop("diagnostic_report_id")
    forged["diagnostic_report_id"] = secret_module.stable_hash(
        forged, prefix="phase3_m25_secret_absence_"
    )

    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit._validate_archived_secret_absence_receipt_v1(
            forged, repository=repository, basis_commit=basis
        )
    assert captured.value.code == audit.FAIL_FORMAL_REPLAY


def test_commit_only_live_replay_disables_local_binding_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hegel_machine.phase3_m25_actor_protocol_qualification_v1 as live_module

    repository, basis = _repository(tmp_path)
    monkeypatch.setattr(
        audit,
        "_live_protocol_source_paths_v1",
        lambda _repository, _basis: ("basis.txt",),
    )
    source_digest, source_count = audit._commit_source_set_digest_v1(
        repository,
        basis,
        ("basis.txt",),
        domain=live_module.SOURCE_SET_HASH_DOMAIN,
    )
    rust_digest = "1" * 64
    bridge_binary_digest = "sha256:" + "2" * 64
    bridge_report = {"diagnostic_report_sha256": "sha256:" + "3" * 64}
    m3_receipt = {"rust": {"binary_digest": rust_digest}}
    report: dict[str, object] = {
        "commit_a_source_set_sha256": source_digest,
        "commit_a_source_file_count": source_count,
        "implementation_bindings": {
            "formal_rust_replay_binary_sha256": f"sha256:{rust_digest}",
            "bridge_rust_replay_binary_sha256": bridge_binary_digest,
            "bridge_rust_qualification_report_sha256": bridge_report[
                "diagnostic_report_sha256"
            ],
            "m3_implementation_qualification_receipt_sha256":
                audit._prefixed_sha256_v1(audit.canonical_json_v1(m3_receipt)),
            "m3_implementation_qualification_receipt": m3_receipt,
        },
    }
    observed: dict[str, object] = {}

    def fake_validate(value: object, **kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(report=value)

    monkeypatch.setattr(
        live_module, "validate_actor_protocol_qualification_report_v1", fake_validate
    )
    (repository / "basis.txt").write_text("dirty worktree\n", encoding="ascii")

    replayed = audit._validate_live_report_public_only_v1(
        report,
        repository=repository,
        basis_commit=basis,
        m3_receipt=m3_receipt,
        bridge_report=bridge_report,
        bridge_binary_digest=bridge_binary_digest,
    )

    assert replayed.report == report
    assert observed == {
        "expected_basis_commit": basis,
        "verify_commit_sources": False,
        "verify_local_implementation_bindings": False,
    }

    drifted = json.loads(json.dumps(report))
    drifted_receipt = drifted["implementation_bindings"][
        "m3_implementation_qualification_receipt"
    ]
    drifted_receipt["rust"]["build_stderr_sha256_or_null"] = "4" * 64
    drifted["implementation_bindings"][
        "m3_implementation_qualification_receipt_sha256"
    ] = audit._prefixed_sha256_v1(audit.canonical_json_v1(drifted_receipt))
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit._validate_live_report_public_only_v1(
            drifted,
            repository=repository,
            basis_commit=basis,
            m3_receipt=m3_receipt,
            bridge_report=bridge_report,
            bridge_binary_digest=bridge_binary_digest,
        )
    assert captured.value.code == audit.FAIL_FORMAL_REPLAY


def test_status_renderer_rejects_symlinked_input_ancestor(tmp_path: Path) -> None:
    repository, basis = _repository(tmp_path)
    _write_status_inputs(repository)
    artifact_parent = repository / "Hegel Machine/artifacts/phase3_m25_external"
    displaced = tmp_path / "displaced-public-inputs"
    artifact_parent.rename(displaced)
    artifact_parent.symlink_to(displaced, target_is_directory=True)

    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.build_external_status_from_worktree_v1(
            repository=repository, basis_commit=basis
        )
    assert captured.value.code == audit.FAIL_FILE_POLICY


def test_manifest_reads_every_dynamic_allowlisted_index_blob(tmp_path: Path) -> None:
    repository, basis = _repository(tmp_path)
    _stage_required(repository)

    manifest, files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis
    )

    expected = sorted(
        set(audit.PUBLICATION_ROLE_REGISTRY) - {audit.AUDIT_RECEIPT_REPOSITORY_PATH}
    )
    assert list(files) == expected
    assert manifest["audit_phase"] == "PREPARE_EXCLUDING_RECEIPT"
    assert manifest["candidate_file_count"] == len(expected)
    assert manifest["excluded_self_output_present_in_candidate"] is False
    assert all(row["git_mode"] == "100644" for row in manifest["candidate_files"])
    assert [row["role_id"] for row in manifest["candidate_files"]] == [
        audit.PUBLICATION_ROLE_REGISTRY[path] for path in expected
    ]
    assert manifest["candidate_inventory_sha256"] == audit._inventory_sha256(
        manifest["candidate_files"]
    )


def test_host_and_worker_freeze_the_same_fresh_formal_public_parent() -> None:
    worker_path = (
        Path(__file__).resolve().parents[1]
        / "tools/phase3_m25_commit_b_publication_audit_worker_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "commit_b_worker_path_freeze_test", worker_path
    )
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    expected_parent = (
        "Hegel Machine/artifacts/phase3_m25_external/formal_genesis_v2"
    )
    assert audit.FORMAL_PUBLIC_PARENT_REPOSITORY_PATH == expected_parent
    assert worker.FORMAL_PUBLIC_PARENT_PATH == expected_parent
    assert audit.FORMAL_EVIDENCE_REPOSITORY_PATH == worker.FORMAL_EVIDENCE_PATH
    assert audit.FORMAL_PROMOTION_REPOSITORY_PATH == worker.FORMAL_PROMOTION_PATH
    assert (
        audit.FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH
        == worker.FORMAL_TRANSACTION_RECEIPT_PATH
    )
    assert audit.PUBLICATION_ROLE_REGISTRY == worker.PUBLICATION_ROLE_REGISTRY


def test_manifest_rejects_nonallowlisted_executable_and_unstaged_drift(
    tmp_path: Path,
) -> None:
    repository, basis = _repository(tmp_path)
    _stage_required(repository)
    _stage(repository, "Hegel Machine/src/new.py", b"pass\n")
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.build_staged_candidate_manifest_v1(repository, basis_commit=basis)
    assert captured.value.code == audit.FAIL_PATH_POLICY

    repository, basis = _repository(tmp_path / "second")
    relative = next(
        path for path, role in audit.PUBLICATION_ROLE_REGISTRY.items()
        if role == "ERRATA_QUALIFICATION"
    )
    _stage_required(repository)
    (repository / relative).write_bytes(audit.canonical_json_v1({"pass": False}))
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.build_staged_candidate_manifest_v1(repository, basis_commit=basis)
    assert captured.value.code == audit.FAIL_FILE_POLICY

    repository, basis = _repository(tmp_path / "third")
    executable = next(
        path for path, role in audit.PUBLICATION_ROLE_REGISTRY.items()
        if role == "M3_IMPLEMENTATION_QUALIFICATION"
    )
    _stage_required(
        repository,
        overrides={executable: (_public_payload(executable), 0o755)},
    )
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.build_staged_candidate_manifest_v1(repository, basis_commit=basis)
    assert captured.value.code == audit.FAIL_FILE_POLICY


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b'{"a":1,"a":2}\n', audit.FAIL_JSON_POLICY),
        (audit.canonical_json_v1({"raw_split_seed": None}), audit.FAIL_SECRET_POLICY),
        (_PKCS8_HEADER + b"\nAAAA\n", audit.FAIL_SECRET_POLICY),
        (
            audit.canonical_json_v1({"note": _quoted_complete_pkcs8_note()}),
            audit.FAIL_SECRET_POLICY,
        ),
        (audit.canonical_json_v1({"path": "/home/alice/project/x"}), audit.FAIL_SECRET_POLICY),
    ],
)
def test_public_file_lint_is_fail_closed(payload: bytes, code: str) -> None:
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit._public_file_lint_v1(
            "Hegel Machine/artifacts/phase3_m25_external/test.json",
            payload,
            raw_path_tokens=(),
        )
    assert captured.value.code == code


def test_pretty_diagnostic_json_is_accepted_but_formal_json_is_bit_exact() -> None:
    diagnostic = next(
        path for path, role in audit.PUBLICATION_ROLE_REGISTRY.items()
        if role == "ACTOR_ELIGIBILITY"
    )
    pretty = b'{\n  "pass": true\n}\n'
    audit._public_file_lint_v1(diagnostic, pretty, raw_path_tokens=())
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit._public_file_lint_v1(
            audit.FORMAL_EVIDENCE_REPOSITORY_PATH, pretty, raw_path_tokens=()
        )
    assert captured.value.code == audit.FAIL_JSON_POLICY


def test_prepare_excludes_receipt_and_finalize_manifest_includes_it(tmp_path: Path) -> None:
    repository, basis = _repository(tmp_path)
    _stage_required(repository, include_receipt=True)
    with pytest.raises(audit.CommitBPublicationAuditError):
        audit.build_staged_candidate_manifest_v1(repository, basis_commit=basis)
    prepare, files = audit.build_staged_candidate_manifest_v1(
        repository,
        basis_commit=basis,
        exclude_receipt=True,
        permit_staged_receipt_for_replay=True,
    )
    assert prepare["audit_phase"] == "PREPARE_EXCLUDING_RECEIPT"
    assert set(files) == set(audit.PUBLICATION_ROLE_REGISTRY) - {
        audit.AUDIT_RECEIPT_REPOSITORY_PATH
    }
    final, final_files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis, exclude_receipt=False
    )
    assert final["audit_phase"] == "FINALIZE_INCLUDING_RECEIPT"
    assert audit.AUDIT_RECEIPT_REPOSITORY_PATH in final_files


def test_exact_role_registry_rejects_missing_and_unknown_prefix_file(tmp_path: Path) -> None:
    repository, basis = _repository(tmp_path / "missing")
    _stage_required(repository)
    missing = next(
        path for path, role in audit.PUBLICATION_ROLE_REGISTRY.items()
        if role == "BRIDGE_BINARY_QUALIFICATION"
    )
    _git(repository, "reset", "--quiet", "HEAD", "--", missing)
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.build_staged_candidate_manifest_v1(repository, basis_commit=basis)
    assert captured.value.code == audit.FAIL_PATH_POLICY

    repository, basis = _repository(tmp_path / "unknown")
    _stage_required(repository)
    _stage(
        repository,
        "Hegel Machine/artifacts/phase3_m25_external/unknown-but-canonical.json",
        audit.canonical_json_v1({"pass": True}),
    )
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.build_staged_candidate_manifest_v1(repository, basis_commit=basis)
    assert captured.value.code == audit.FAIL_PATH_POLICY
    assert len(audit.PUBLICATION_ROLE_REGISTRY) == len(
        set(audit.PUBLICATION_ROLE_REGISTRY.values())
    )


def test_container_command_is_pinned_offline_nonroot_readonly(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    runtime = tmp_path / "runtime"
    request = tmp_path / "request.json"
    candidate.mkdir()
    (runtime / "control").mkdir(parents=True)
    (runtime / "control/phase3_internal_actor_seccomp_v1.json").write_text("{}")
    request.write_text("{}")
    image = "python@sha256:" + "1" * 64
    command = audit.publication_actor_container_command_v1(
        candidate=candidate, runtime=runtime, request_path=request, image_ref=image
    )
    assert command[:2] == ("/usr/bin/docker", "--host=unix:///var/run/docker.sock")
    for flag in (
        "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true", "--user=65534:65534",
    ):
        assert flag in command
    mounts = [item for item in command if item.startswith("--mount=")]
    assert len(mounts) == 3
    assert all("readonly" in item for item in mounts)


def _actor_receipt(
    manifest: dict[str, object], *, image: str = _TEST_ACTOR_IMAGE_REF
) -> dict[str, object]:
    isolation_body: dict[str, object] = {
        "schema": "hegel-phase3-m25-commit-b-purpose4-live-isolation/1",
        "purpose_id": 4,
        "actor_image_ref": image,
        "uid": 65534,
        "gid": 65534,
        "required_checks": {
            "nonroot_exact": True,
            "capability_sets_zero": True,
            "no_new_privileges": True,
            "seccomp_filter": True,
            "network_loopback_only": True,
            "six_syscalls_blocked_eperm": True,
            "immutable_mounts_read_only": True,
            "tmp_private_writable": True,
            "environment_exact": True,
            "inherited_fds_exact": True,
            "cgroup_limits_exact": True,
        },
        "all_required_checks_passed": True,
    }
    isolation_body["receipt_sha256"] = hashlib.sha256(
        audit.canonical_json_v1(isolation_body)
    ).hexdigest()
    body: dict[str, object] = {
        "schema": audit.RECEIPT_SCHEMA,
        "artifact_kind": "DIAGNOSTIC_PUBLICATION_CONTROL",
        "policy_id": audit.POLICY_ID,
        "purpose_id": 4,
        "audit_phase": manifest["audit_phase"],
        "basis_commit_sha1": manifest["basis_commit_sha1"],
        "actor_image_ref": image,
        "request_sha256": "3" * 64,
        "candidate_manifest": manifest,
        "actor_recomputed_inventory_sha256": manifest["candidate_inventory_sha256"],
        "runtime_inventory_sha256": "4" * 64,
        "private_forbidden_raw_path_token_sha256s": ["5" * 64],
        "isolation_live_receipt": isolation_body,
        "required_checks": {
            "exact_manifest_and_file_set": True,
            "path_mode_size_sha256_bound": True,
            "nonallowlisted_and_executable_paths_absent": True,
            "json_strict_duplicate_free_and_required_bit_exact": True,
            "forbidden_secret_field_names_absent": True,
            "private_key_magic_and_complete_blocks_absent": True,
            "raw_author_or_host_paths_absent": True,
            "receipt_scope_exact_for_audit_phase": True,
            "no_key_seed_signature_marker_or_formal_action": True,
        },
        "all_required_checks_passed": True,
        "authority_disclosure": dict(TECHNICAL_ACTOR_DISCLOSURE_V1),
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
        },
    }
    body["receipt_sha256"] = hashlib.sha256(audit.canonical_json_v1(body)).hexdigest()
    return body


def test_actor_receipt_strictly_binds_manifest_and_no_authority() -> None:
    manifest = {
        "audit_phase": "PREPARE_EXCLUDING_RECEIPT",
        "basis_commit_sha1": "6" * 40,
        "candidate_inventory_sha256": "7" * 64,
    }
    receipt = _actor_receipt(manifest)
    validated = audit.validate_actor_receipt_v1(
        receipt,
        expected_manifest=manifest,
        expected_request_sha256="3" * 64,
        actor_image_ref=str(receipt["actor_image_ref"]),
    )
    assert validated["authority_boundary"]["formal_gate_delta"] == 0
    tampered = json.loads(json.dumps(receipt))
    tampered["required_checks"]["raw_author_or_host_paths_absent"] = False
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = hashlib.sha256(
        audit.canonical_json_v1(tampered)
    ).hexdigest()
    with pytest.raises(audit.CommitBPublicationAuditError):
        audit.validate_actor_receipt_v1(
            tampered,
            expected_manifest=manifest,
            expected_request_sha256="3" * 64,
            actor_image_ref=str(receipt["actor_image_ref"]),
        )


def test_actor_receipt_rejects_image_different_from_basis_profile(tmp_path: Path) -> None:
    repository, basis = _repository(tmp_path)
    _stage_required(repository)
    manifest, _files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis
    )
    forged = _actor_receipt(
        manifest, image="python@sha256:" + "9" * 64
    )
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.validate_actor_receipt_v1(
            forged,
            expected_manifest=manifest,
            expected_request_sha256=None,
            actor_image_ref=audit._load_image_ref(repository, basis),
        )
    assert captured.value.code == audit.FAIL_ACTOR_RESPONSE


def _formal_public_fixture() -> tuple[dict[str, bytes], str, dict[str, object]]:
    basis = "9" * 40
    evidence = audit.canonical_json_v1({"synthetic_test_transport": True})
    promotion: dict[str, object] = {
        "basis_commit": basis,
        "child_state": "NOT_RUN",
        "m3_run_started": False,
        "m3_entry_qualified": True,
        "phase3_m3_start_required_separately": True,
    }
    promotion_bytes = audit.canonical_json_v1(promotion)
    transaction: dict[str, object] = {
        "schema": "hegel-phase3-m25-publication-receipt/1",
        "basis_commit": basis,
        "run_id_hex": (b"r" * 16).hex(),
        "ledger_id_hex": (b"l" * 16).hex(),
        "public_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
        "public_promotion_sha256": hashlib.sha256(promotion_bytes).hexdigest(),
        "seed_custody_verification_receipt_sha256_or_null": "a" * 64,
        "prospective_public_replay_passed": True,
        "marker_was_complete_during_staging": False,
        "actor_cleanup_required_before_publication": True,
        "authority_disclosure": dict(TECHNICAL_ACTOR_DISCLOSURE_V1),
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "contains_split_assignment_rows": False,
    }
    return (
        {
            audit.FORMAL_EVIDENCE_REPOSITORY_PATH: evidence,
            audit.FORMAL_PROMOTION_REPOSITORY_PATH: promotion_bytes,
            audit.FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH: audit.canonical_json_v1(
                transaction
            ),
        },
        basis,
        transaction,
    )


def test_formal_transaction_receipt_binds_typed_run_ledger_and_seed_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hegel_machine.phase3_m25_formal_container_executor_v1 as formal

    files, basis, transaction = _formal_public_fixture()
    promotion = json.loads(files[audit.FORMAL_PROMOTION_REPOSITORY_PATH])
    typed = SimpleNamespace(
        execution_candidate_fields={"run_id": b"r" * 16},
        ledger_genesis_fields={"ledger_id": b"l" * 16},
    )
    monkeypatch.setattr(formal, "load_gate_evidence_inputs_v1", lambda _value: typed)
    monkeypatch.setattr(formal, "replay_public_gate_evidence_v1", lambda _value: promotion)
    monkeypatch.setattr(
        audit,
        "_validate_role_specific_public_payloads_v1",
        lambda _files, **_kwargs: {"role_specific_payload_count": 11},
    )
    result = audit._host_strict_replay_formal_public_payloads_v1(
        files, basis_commit=basis, require_formal_payloads=True
    )
    assert result["child_state"] == "NOT_RUN"

    tampered = dict(transaction)
    tampered["run_id_hex"] = (b"x" * 16).hex()
    files[audit.FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH] = audit.canonical_json_v1(
        tampered
    )
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit._host_strict_replay_formal_public_payloads_v1(
            files, basis_commit=basis, require_formal_payloads=True
        )
    assert captured.value.code == audit.FAIL_FORMAL_REPLAY

    tampered["run_id_hex"] = (b"r" * 16).hex()
    tampered["seed_custody_verification_receipt_sha256_or_null"] = None
    files[audit.FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH] = audit.canonical_json_v1(
        tampered
    )
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit._host_strict_replay_formal_public_payloads_v1(
            files, basis_commit=basis, require_formal_payloads=True
        )
    assert captured.value.code == audit.FAIL_FORMAL_REPLAY


def test_commit_only_host_gate_replay_uses_prevalidated_reports_not_live_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hegel_machine.phase3_container_actor_runtime_v1 as actor_runtime
    import hegel_machine.phase3_m25_container_ceremony_v1 as ceremony
    import hegel_machine.phase3_m25_errata_qualification_v1 as errata_module
    import hegel_machine.phase3_m25_formal_container_executor_v1 as formal

    repository, basis = _repository(tmp_path)
    dirty_profile = tmp_path / "dirty-current-workspace-profile.json"
    dirty_profile.write_text("not JSON\n", encoding="ascii")
    monkeypatch.setattr(actor_runtime, "PROFILE_PATH", dirty_profile)
    monkeypatch.setattr(errata_module, "APPROVED_TOOLCHAIN_POLICY_PATH", dirty_profile)
    actor_report = {
        "basis_commit": basis,
        "technical_actor_eligible": True,
    }
    errata_report = {"implementation_basis_commit": basis}
    run_id = b"r" * 16
    ledger_id = b"l" * 16
    values: dict[str, object] = {}
    tuple_fields = {
        "actor_key_manifests", "parent_top_level_path_rows", "parent_history_rows",
        "parent_touched_rows", "parent_legacy_rows", "external_envelopes",
        "canonical_binding_objects", "opaque_registration_intents",
        "opaque_registry_records", "opaque_registry_snapshots", "bridge_envelopes",
    }
    for field in fields(formal.GateEvidenceInputsV1):
        if field.name == "basis_commit":
            values[field.name] = basis
        elif field.name == "actor_qualification_report":
            values[field.name] = actor_report
        elif field.name == "errata_qualification_report":
            values[field.name] = errata_report
        elif field.name == "marker_snapshot":
            values[field.name] = formal.MarkerSnapshot(
                "COMPLETE", b"s" * 32, b"m" * 32, b"k" * 16, 1
            )
        elif field.name.endswith("_frame"):
            values[field.name] = b"frame"
        elif field.name in tuple_fields:
            values[field.name] = ()
        elif field.name == "execution_candidate_fields":
            values[field.name] = {"run_id": run_id}
        elif field.name == "ledger_genesis_fields":
            values[field.name] = {"ledger_id": ledger_id}
        else:
            values[field.name] = {}
    typed = formal.GateEvidenceInputsV1(**values)  # type: ignore[arg-type]
    evidence_value = formal.serialize_gate_evidence_inputs_v1(typed)
    evidence_bytes = audit.canonical_json_v1(evidence_value)
    gate_report = MappingProxyType(
        {
            "all_gates_15_24_passed": True,
            "gates_after": 24,
            "child_state": "NOT_RUN",
            "m3_run_started": False,
        }
    )
    qualified = ceremony.QualifiedGateEvidenceV1(
        basis_commit=basis,
        gate_report=gate_report,
        formal_roots=MappingProxyType({}),
        _seal=ceremony._PROMOTION_SEAL,
    )
    promotion = ceremony.promote_gate_evidence_v1(qualified)
    promotion_bytes = audit.canonical_json_v1(promotion)
    transaction: dict[str, object] = {
        "schema": formal.PUBLICATION_RECEIPT_SCHEMA,
        "basis_commit": basis,
        "run_id_hex": run_id.hex(),
        "ledger_id_hex": ledger_id.hex(),
        "public_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "public_promotion_sha256": hashlib.sha256(promotion_bytes).hexdigest(),
        "seed_custody_verification_receipt_sha256_or_null": "a" * 64,
        "prospective_public_replay_passed": True,
        "marker_was_complete_during_staging": False,
        "actor_cleanup_required_before_publication": True,
        "authority_disclosure": dict(TECHNICAL_ACTOR_DISCLOSURE_V1),
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "contains_split_assignment_rows": False,
    }
    files = {
        audit.FORMAL_EVIDENCE_REPOSITORY_PATH: evidence_bytes,
        audit.FORMAL_PROMOTION_REPOSITORY_PATH: promotion_bytes,
        audit.FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH:
            audit.canonical_json_v1(transaction),
    }
    observed: list[tuple[str, Path, str]] = []

    def pure_actor(report: object, **kwargs: object) -> dict[str, object]:
        observed.append(("actor", kwargs["repository"], kwargs["basis_commit"]))
        return dict(report)  # type: ignore[arg-type]

    def pure_errata(report: object, **kwargs: object) -> dict[str, object]:
        observed.append(("errata", kwargs["repository"], kwargs["basis_commit"]))
        return dict(report)  # type: ignore[arg-type]

    def prevalidated_replay(
        candidate: object, *, actor_report: object, errata_report: object
    ) -> object:
        ceremony._validate_report_basis(
            candidate,  # type: ignore[arg-type]
            prevalidated_actor_report=actor_report,  # type: ignore[arg-type]
            prevalidated_errata_report=errata_report,  # type: ignore[arg-type]
        )
        return qualified

    monkeypatch.setattr(audit, "_validate_actor_report_public_only_v1", pure_actor)
    monkeypatch.setattr(audit, "_validate_errata_report_public_only_v1", pure_errata)
    monkeypatch.setattr(
        ceremony,
        "_evaluate_gates_15_24_with_prevalidated_report_basis_v1",
        prevalidated_replay,
    )
    monkeypatch.setattr(
        ceremony,
        "validate_qualification_report",
        lambda _report: (_ for _ in ()).throw(AssertionError("live actor validator")),
    )
    monkeypatch.setattr(
        ceremony,
        "validate_dual_errata_qualification_report",
        lambda _report: (_ for _ in ()).throw(AssertionError("live errata validator")),
    )
    monkeypatch.setattr(
        formal,
        "replay_public_gate_evidence_v1",
        lambda _payload: (_ for _ in ()).throw(AssertionError("live gate replay")),
    )
    monkeypatch.setattr(
        audit,
        "_validate_role_specific_public_payloads_v1",
        lambda _files, **_kwargs: {"role_specific_payload_count": 11},
    )

    replayed = audit._host_strict_replay_formal_public_payloads_v1(
        files,
        basis_commit=basis,
        require_formal_payloads=True,
        repository=repository,
        commit_only=True,
    )

    assert replayed["child_state"] == "NOT_RUN"
    assert observed == [
        ("actor", repository, basis),
        ("errata", repository, basis),
    ]


def test_pre_genesis_roles_reject_contradictory_state_claims() -> None:
    basis = "b" * 40
    readiness: dict[str, object] = {
        "schema": "hegel-phase3-m25-formal-container-readiness/2",
        "basis_commit": basis,
        "ready_for_explicit_execute": True,
        "blockers": [],
        "formal_gates_before": 14,
        "formal_gates_after": 14,
        "child_state": "NOT_RUN",
        "m3_run_started": False,
        "qualification_side_effects_performed": True,
        "qualification_network_mode": "none",
        "qualification_persistent_rust_binary_verified_or_written": True,
        "qualification_non_authoritative_roots_computed": True,
        "ceremony_actor_key_seed_marker_side_effects_performed": False,
        "formal_authority_or_gate_effect": "NONE",
        "static_replay_roots_are_execution_bindings": False,
    }
    execution = {
        **readiness,
        "schema": "hegel-phase3-m25-execution-status/2",
        "ceremony_execution_enabled_for_basis": True,
        "external_genesis_executed": False,
        "blocking_prerequisites": [],
    }
    audit._validate_pre_genesis_readiness_v1(
        readiness, execution, basis_commit=basis
    )
    contradictory = dict(execution)
    contradictory["child_state"] = "COMPLETE"
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit._validate_pre_genesis_readiness_v1(
            readiness, contradictory, basis_commit=basis
        )
    assert captured.value.code == audit.FAIL_FORMAL_REPLAY

def test_finalize_index_invokes_fresh_purpose4_over_receipt_inclusive_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, basis = _repository(tmp_path)
    _stage_required(repository)
    prepare_manifest, _files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis
    )
    prepare_receipt = _actor_receipt(prepare_manifest)
    _stage(
        repository,
        audit.AUDIT_RECEIPT_REPOSITORY_PATH,
        audit.canonical_json_v1(prepare_receipt),
    )
    final_manifest, _final_files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis, exclude_receipt=False
    )
    final_receipt = _actor_receipt(final_manifest)
    observed: list[tuple[bool, bool]] = []

    def fresh_actor(**kwargs: object) -> audit.CommitBActorAuditResultV1:
        finalize = bool(kwargs.get("finalize_index"))
        observed.append(
            (finalize, bool(kwargs.get("permit_staged_receipt_for_prepare_replay")))
        )
        receipt = final_receipt if finalize else prepare_receipt
        manifest = final_manifest if finalize else prepare_manifest
        return audit.CommitBActorAuditResultV1(
            receipt=receipt,
            canonical_receipt_bytes=audit.canonical_json_v1(receipt),
            manifest=manifest,
            host_formal_replay={"formal_gate_replay_performed": True},
        )

    monkeypatch.setattr(audit, "run_commit_b_publication_actor_audit_v1", fresh_actor)
    result = audit.finalize_staged_commit_b_publication_v1(
        repository=repository, basis_commit=basis
    )
    assert observed == [(False, True), (True, False)]
    assert result["status"] == "PASS_EXACT_STAGED_COMMIT_B_PUBLICATION"
    assert result["finalize_actor_receipt"]["audit_phase"] == (
        "FINALIZE_INCLUDING_RECEIPT"
    )


def test_finalize_rejects_index_drift_between_fresh_actor_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, basis = _repository(tmp_path)
    _stage_required(repository)
    prepare_manifest, _files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis
    )
    prepare_receipt = _actor_receipt(prepare_manifest)
    _stage(
        repository,
        audit.AUDIT_RECEIPT_REPOSITORY_PATH,
        audit.canonical_json_v1(prepare_receipt),
    )

    def drifting_actor(**kwargs: object) -> audit.CommitBActorAuditResultV1:
        if not bool(kwargs.get("finalize_index")):
            return audit.CommitBActorAuditResultV1(
                receipt=prepare_receipt,
                canonical_receipt_bytes=audit.canonical_json_v1(prepare_receipt),
                manifest=prepare_manifest,
                host_formal_replay={"formal_gate_replay_performed": True},
            )
        _stage(
            repository,
            audit.ACTOR_QUALIFICATION_REPOSITORY_PATH,
            b'{"changed_between_actors":true}\n',
        )
        changed_manifest, _changed_files = audit.build_staged_candidate_manifest_v1(
            repository, basis_commit=basis, exclude_receipt=False
        )
        changed_receipt = _actor_receipt(changed_manifest)
        return audit.CommitBActorAuditResultV1(
            receipt=changed_receipt,
            canonical_receipt_bytes=audit.canonical_json_v1(changed_receipt),
            manifest=changed_manifest,
            host_formal_replay={"formal_gate_replay_performed": True},
        )

    monkeypatch.setattr(
        audit, "run_commit_b_publication_actor_audit_v1", drifting_actor
    )
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.finalize_staged_commit_b_publication_v1(
            repository=repository, basis_commit=basis
        )
    assert captured.value.code == audit.FAIL_FINAL_STAGED_SET


def test_verify_commit_requires_repo_external_finalize_receipt_and_tree_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, basis = _repository(tmp_path)
    _stage_required(repository)
    prepare_manifest, _files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis
    )
    prepare_receipt = _actor_receipt(prepare_manifest)
    prepare_bytes = audit.canonical_json_v1(prepare_receipt)
    _stage(repository, audit.AUDIT_RECEIPT_REPOSITORY_PATH, prepare_bytes)
    full_manifest, _full_files = audit.build_staged_candidate_manifest_v1(
        repository, basis_commit=basis, exclude_receipt=False
    )
    final_actor = _actor_receipt(full_manifest)
    formal_replay = {"formal_gate_replay_performed": True, "child_state": "NOT_RUN"}
    final_status: dict[str, object] = {
        "schema": audit.FINAL_STATUS_SCHEMA,
        "status": "PASS_EXACT_STAGED_COMMIT_B_PUBLICATION",
        "basis_commit_sha1": basis,
        "candidate_manifest_sha256": full_manifest["manifest_sha256"],
        "actor_receipt_sha256": prepare_receipt["receipt_sha256"],
        "finalize_actor_receipt_sha256": final_actor["receipt_sha256"],
        "finalize_actor_receipt": final_actor,
        "staged_audit_receipt_sha256": hashlib.sha256(prepare_bytes).hexdigest(),
        "final_staged_path_count": len(audit.PUBLICATION_ROLE_REGISTRY),
        "formal_host_replay": formal_replay,
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
            "commit_created_or_pushed": False,
        },
    }
    final_status["status_sha256"] = hashlib.sha256(
        audit.canonical_json_v1(final_status)
    ).hexdigest()
    external = tmp_path / "commit-b-finalize-receipt.json"
    external.write_bytes(audit.canonical_json_v1(final_status))
    external.chmod(0o600)
    _git(repository, "commit", "--quiet", "-m", "publication B")
    commit_b = _git(repository, "rev-parse", "HEAD")
    replay_call: dict[str, object] = {}

    def commit_only_replay(_files: object, **kwargs: object) -> dict[str, object]:
        replay_call.update(kwargs)
        return formal_replay

    monkeypatch.setattr(
        audit,
        "_host_strict_replay_formal_public_payloads_v1",
        commit_only_replay,
    )
    result = audit.verify_commit_b_publication_commit_v1(
        repository=repository,
        basis_commit=basis,
        publication_commit=commit_b,
        finalize_receipt_path=external,
    )
    assert result["finalize_tree_inventory_equal"] is True
    assert replay_call == {
        "basis_commit": basis,
        "require_formal_payloads": True,
        "repository": repository,
        "commit_only": True,
    }

    _git(repository, "commit", "--quiet", "--allow-empty", "-m", "grafted child")
    grafted_child = _git(repository, "rev-parse", "HEAD")
    grafts = repository / ".git/info/grafts"
    grafts.parent.mkdir(parents=True, exist_ok=True)
    grafts.write_text(f"{grafted_child} {basis}\n", encoding="ascii")
    with pytest.raises(audit.CommitBPublicationAuditError) as captured:
        audit.verify_commit_b_publication_commit_v1(
            repository=repository,
            basis_commit=basis,
            publication_commit=grafted_child,
            finalize_receipt_path=external,
        )
    assert captured.value.code == audit.FAIL_FINAL_STAGED_SET

    with pytest.raises(audit.CommitBPublicationAuditError):
        audit.verify_commit_b_publication_commit_v1(
            repository=repository,
            basis_commit=basis,
            publication_commit=commit_b,
            finalize_receipt_path=tmp_path / "missing.json",
        )


def test_standard_library_worker_independently_replays_candidate(tmp_path: Path) -> None:
    worker_path = Path(__file__).resolve().parents[1] / "tools/phase3_m25_commit_b_publication_audit_worker_v1.py"
    spec = importlib.util.spec_from_file_location("commit_b_worker_test", worker_path)
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    candidate = tmp_path / "candidate"
    file_root = candidate / "files"
    rows: list[dict[str, object]] = []
    worker_basis = "8" * 40
    required_paths = sorted(
        set(worker.PUBLICATION_ROLE_REGISTRY) - {worker.AUDIT_RECEIPT_PATH}
    )
    header_by_role = {
        "ACTOR_ELIGIBILITY": {"schema": "hegel-phase3-container-actor-qualification/1", "basis_commit": worker_basis},
        "ERRATA_QUALIFICATION": {"schema_version": "hegel-phase3-m25-exact-wire-errata-qualification/2", "implementation_basis_commit": worker_basis},
        "M3_IMPLEMENTATION_QUALIFICATION": {"schema_version": "hegel-m3-implementation-qualification/1", "basis_commit": worker_basis},
        "BRIDGE_BINARY_QUALIFICATION": {"schema_version": "hegel-phase3-m25-bridge-dag-rust-binary-qualification/1", "implementation_basis_commit": worker_basis},
        "LIVE_ACTOR_PROTOCOL_QUALIFICATION": {"schema_version": "hegel-phase3-m25-live-actor-protocol-qualification/2", "basis_commit": worker_basis},
        "PRE_GENESIS_EXECUTION_STATUS": {"schema": "hegel-phase3-m25-execution-status/2", "basis_commit": worker_basis},
        "PRE_GENESIS_READINESS": {"schema": "hegel-phase3-m25-formal-container-readiness/2", "basis_commit": worker_basis},
        "FORMAL_GATE_EVIDENCE": {"schema": "hegel-phase3-m25-public-gate-evidence-replay/1", "artifact_kind": "FORMAL_GATE_EVIDENCE_INPUTS_PUBLIC_REPLAY"},
        "FORMAL_GATE_PROMOTION": {"schema": "hegel-phase3-m25-container-ceremony/1", "basis_commit": worker_basis},
        "FORMAL_TRANSACTION_PUBLICATION_RECEIPT": {"schema": "hegel-phase3-m25-publication-receipt/1", "basis_commit": worker_basis},
    }
    payloads: dict[str, bytes] = {}
    for path in required_paths:
        role = worker.PUBLICATION_ROLE_REGISTRY[path]
        if role == "EXTERNAL_STATUS_DOCUMENT":
            continue
        value = header_by_role[role]
        payloads[path] = (
            audit.canonical_json_v1(value)
            if path in worker.CANONICAL_JSON_REQUIRED_PATHS
            else (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
        )
    status_path = "Hegel Machine/docs/phase3_m25_external_status.md"
    payloads[status_path] = worker.render_external_status(worker_basis, payloads)
    for path in required_paths:
        payload = payloads[path]
        destination = file_root.joinpath(*Path(path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        rows.append(
            {
                "path": path,
                "role_id": worker.PUBLICATION_ROLE_REGISTRY[path],
                "git_mode": "100644",
                "index_blob_sha1": hashlib.sha1(
                    b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
                ).hexdigest(),
                "byte_length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: dict[str, object] = {
        "schema": worker.MANIFEST_SCHEMA,
        "policy_id": worker.POLICY_ID,
        "audit_phase": "PREPARE_EXCLUDING_RECEIPT",
        "basis_commit_sha1": worker_basis,
        "changed_path_scope": "EXACT_GIT_INDEX_DIFF_FROM_BASIS_COMMIT",
        "allowed_public_prefixes": worker.ALLOWED_PUBLIC_PREFIXES,
        "executable_prefixes": worker.EXECUTABLE_PREFIXES,
        "excluded_self_output_repository_path": worker.AUDIT_RECEIPT_PATH,
        "excluded_self_output_present_in_candidate": False,
        "path_role_registry": [
            {
                "path": path,
                "role_id": worker.PUBLICATION_ROLE_REGISTRY[path],
                "required_cardinality": 1,
            }
            for path in required_paths
        ],
        "role_cardinalities": {
            worker.PUBLICATION_ROLE_REGISTRY[path]: 1 for path in required_paths
        },
        "candidate_files": rows,
        "candidate_file_count": len(rows),
        "candidate_total_byte_length": sum(int(row["byte_length"]) for row in rows),
        "candidate_inventory_sha256": worker.inventory_sha256(rows),
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(worker.canonical(manifest)).hexdigest()
    (candidate / "manifest.json").write_bytes(worker.canonical(manifest))
    assert worker.verify_candidate(
        candidate, manifest, [], "PREPARE_EXCLUDING_RECEIPT"
    ) == manifest["candidate_inventory_sha256"]
    embedded_complete_pem = worker.canonical(
        {"note": _quoted_complete_pkcs8_note()}
    )
    assert worker.has_private_key_magic(embedded_complete_pem) is True
    assert worker.has_private_key_magic(
        worker.canonical(
            {"note": "quoted " + _PKCS8_HEADER.decode("ascii") + " only"}
        )
    ) is False


def test_cli_render_status_is_exclusive_and_writer_cleans_failed_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = Path(__file__).resolve().parents[1] / "tools/phase3_m25_commit_b_publication_audit_v1.py"
    repository, basis = _repository(tmp_path / "cli")
    payloads = _write_status_inputs(repository)
    command = [
        sys.executable,
        str(tool),
        "--repository",
        str(repository),
        "render-status",
        "--basis-commit",
        basis,
    ]
    first = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    output = repository / audit.EXTERNAL_STATUS_REPOSITORY_PATH
    assert output.read_bytes() == audit.render_external_status_v1(
        basis_commit=basis, files=payloads
    )
    second = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert second.returncode == 70
    assert output.read_bytes() == audit.render_external_status_v1(
        basis_commit=basis, files=payloads
    )

    spec = importlib.util.spec_from_file_location("commit_b_cli_test", tool)
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fault_parent = tmp_path / "fault"
    fault_parent.mkdir()
    real_write = cli.os.write
    monkeypatch.setattr(cli.os, "write", lambda _fd, _view: (_ for _ in ()).throw(OSError("fault")))
    with pytest.raises(OSError):
        cli._write_exclusive_anchored(fault_parent, ("receipt.json",), b"ok\n", mode=0o600)
    assert not (fault_parent / "receipt.json").exists()
    monkeypatch.setattr(cli.os, "write", real_write)
    cli._write_exclusive_anchored(fault_parent, ("receipt.json",), b"ok\n", mode=0o600)
    with pytest.raises(FileExistsError):
        cli._write_exclusive_anchored(fault_parent, ("receipt.json",), b"new\n", mode=0o600)
    assert (fault_parent / "receipt.json").read_bytes() == b"ok\n"


def test_cli_has_real_four_subcommand_entrypoint() -> None:
    tool = Path(__file__).resolve().parents[1] / "tools/phase3_m25_commit_b_publication_audit_v1.py"
    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert "render-status" in completed.stdout
    assert "prepare" in completed.stdout
    assert "finalize-index" in completed.stdout
    assert "verify-commit" in completed.stdout

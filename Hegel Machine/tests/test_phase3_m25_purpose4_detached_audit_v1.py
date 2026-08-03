from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

import hegel_machine.phase3_m25_purpose4_detached_audit_v1 as detached
from hegel_machine.phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    OBJECT_TAGS,
    candidate_content_root,
    encode_formal_object,
    external_signature_preimage_v1,
    git_sha1_commit_id,
    id_digest_v1,
)


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
PARENT_RECEIPT = PROJECT / "artifacts/phase3_m25_parent_absence_audit_receipt_v1.json"


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        env={
            "LC_ALL": "C",
            "LANG": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _tiny_repository(tmp_path: Path) -> tuple[Path, bytes]:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Gate17 Test")
    _git(repository, "config", "user.email", "gate17@example.invalid")
    (repository / "one.txt").write_text("one\n", encoding="utf-8")
    _git(repository, "add", "one.txt")
    _git(repository, "commit", "--quiet", "-m", "one")
    (repository / "two.txt").write_text("two\n", encoding="utf-8")
    _git(repository, "add", "two.txt")
    _git(repository, "commit", "--quiet", "-m", "two")
    return repository, bytes.fromhex(_git(repository, "rev-parse", "HEAD"))


def _snapshot(tmp_path: Path) -> detached.DetachedParentSnapshotV1:
    repository, parent = _tiny_repository(tmp_path)
    return detached._prepare_snapshot(
        repository,
        parent=parent,
        basis_commit=parent.hex(),
        git_executable=detached._resolve_git_executable(),
        temporary_parent=None,
        require_frozen_parent=False,
    )


def _unseal(snapshot: detached.DetachedParentSnapshotV1) -> None:
    detached._set_snapshot_read_only(snapshot.root, False)


def test_detached_git_uses_only_exact_canonical_safe_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "detached snapshot with spaces"
    repository.mkdir()
    executable = tmp_path / "bound-git"
    executable.write_bytes(b"test-only")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(command, 0, b"ok\n", b"")

    monkeypatch.setattr(detached.subprocess, "run", fake_run)
    assert detached._run_git(
        executable,
        repository,
        ("status", "--porcelain"),
    ) == b"ok\n"
    safe_repository = repository.resolve(strict=True)
    assert captured["cwd"] == safe_repository
    assert captured["command"] == [
        str(executable),
        "-c",
        "core.quotePath=false",
        "-c",
        f"safe.directory={safe_repository}",
        "status",
        "--porcelain",
    ]
    assert "safe.directory=*" not in captured["command"]


@pytest.mark.skipif(
    not Path("/usr/bin/docker").is_file()
    or not Path("/var/run/docker.sock").exists(),
    reason="local offline Docker is unavailable",
)
def test_exact_safe_directory_allows_copied_git_across_actor_uid(
    tmp_path: Path,
) -> None:
    repository, head = _tiny_repository(tmp_path)
    if repository.stat().st_uid == 65534:
        pytest.skip("test repository owner unexpectedly equals actor uid")
    copied_git = tmp_path / "runtime-git"
    shutil.copyfile("/usr/bin/git", copied_git)
    copied_git.chmod(0o555)
    profile = json.loads(detached.PROFILE_PATH.read_text(encoding="utf-8"))
    image = profile["images"][detached.PYTHON_IMAGE_KEY]
    control = [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
    ]
    available = subprocess.run(
        [*control, "image", "inspect", image],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if available.returncode != 0:
        pytest.skip("digest-pinned actor image is not already local")
    common = [
        *control,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--env=HOME=/nonexistent",
        "--env=GIT_CONFIG_NOSYSTEM=1",
        "--env=GIT_CONFIG_GLOBAL=/dev/null",
        "--env=GIT_CONFIG_SYSTEM=/dev/null",
        f"--mount=type=bind,src={repository.resolve()},dst=/snapshot,readonly",
        f"--mount=type=bind,src={copied_git.resolve()},dst=/runtime/bin/git,readonly",
        "--entrypoint=/runtime/bin/git",
        image,
    ]
    rejected = subprocess.run(
        [*common, "-C", "/snapshot", "rev-parse", "HEAD"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert rejected.returncode != 0
    assert b"dubious ownership" in rejected.stderr

    accepted = subprocess.run(
        [
            *common,
            "-c",
            "core.quotePath=false",
            "-c",
            "safe.directory=/snapshot",
            "-C",
            "/snapshot",
            "rev-parse",
            "HEAD",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert accepted.returncode == 0, accepted.stderr.decode("utf-8", "replace")
    assert accepted.stdout.strip() == head.hex().encode("ascii")


def test_detached_snapshot_contains_exact_parent_closure_and_cleans_up(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    root = snapshot.root
    try:
        assert root.parent.parent == Path("/tmp")
        assert stat.S_IMODE(root.parent.stat().st_mode) == 0o700
        manifest = detached.validate_detached_parent_snapshot_v1(
            root,
            snapshot.manifest,
            git_executable=snapshot.git_executable,
            require_frozen_parent=False,
        )
        assert set(path.name for path in root.iterdir()) == {".git"}
        assert manifest["head_is_audited_parent"] is True
        assert manifest["alternate_object_directories_present"] is False
        assert manifest["promisor_or_partial_clone_present"] is False
        assert manifest["shallow_repository"] is False
        assert manifest["reachable_object_count"] == sum(
            manifest["reachable_object_counts_by_type"].values()
        )
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
        assert all(
            stat.S_IMODE(path.stat().st_mode) in {0o444, 0o555}
            for path in root.rglob("*")
        )
    finally:
        snapshot.close()
    assert not root.exists()


def test_snapshot_rejects_tampered_pack_object(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    try:
        _unseal(snapshot)
        pack = next((snapshot.root / ".git/objects/pack").glob("*.pack"))
        payload = bytearray(pack.read_bytes())
        payload[len(payload) // 2] ^= 1
        pack.write_bytes(payload)
        with pytest.raises(detached.Purpose4DetachedAuditError) as captured:
            detached.validate_detached_parent_snapshot_v1(
                snapshot.root,
                snapshot.manifest,
                git_executable=snapshot.git_executable,
                require_frozen_parent=False,
            )
        assert captured.value.code in {
            detached.FAIL_SNAPSHOT_BUILD,
            detached.FAIL_SNAPSHOT_INVENTORY,
        }
    finally:
        snapshot.close()


def test_snapshot_rejects_missing_audited_ref(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    try:
        _unseal(snapshot)
        (snapshot.root / ".git/refs/hegel/audited-parent").unlink()
        with pytest.raises(detached.Purpose4DetachedAuditError):
            detached.validate_detached_parent_snapshot_v1(
                snapshot.root,
                snapshot.manifest,
                git_executable=snapshot.git_executable,
                require_frozen_parent=False,
            )
    finally:
        snapshot.close()


@pytest.mark.parametrize(
    "relative,content",
    [
        (".git/objects/info/alternates", "/forbidden/object/store\n"),
        (".git/shallow", "00" * 20 + "\n"),
        (".git/objects/pack/injected.promisor", ""),
    ],
)
def test_snapshot_rejects_external_or_incomplete_object_dependencies(
    tmp_path: Path, relative: str, content: str
) -> None:
    snapshot = _snapshot(tmp_path)
    try:
        _unseal(snapshot)
        path = snapshot.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="ascii")
        with pytest.raises(detached.Purpose4DetachedAuditError) as captured:
            detached.validate_detached_parent_snapshot_v1(
                snapshot.root,
                snapshot.manifest,
                git_executable=snapshot.git_executable,
                require_frozen_parent=False,
            )
        assert captured.value.code == detached.FAIL_SNAPSHOT_POLICY
    finally:
        snapshot.close()


def test_purpose4_command_is_offline_pinned_nonroot_and_read_only(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    runtime = tmp_path / "runtime"
    request = tmp_path / "request.json"
    snapshot.mkdir()
    runtime.mkdir()
    request.write_text("{}", encoding="ascii")
    image = "python@sha256:" + "a" * 64
    command = detached.purpose4_container_command_v1(
        snapshot=snapshot,
        runtime=runtime,
        request_path=request,
        seccomp_path=detached.SECCOMP_PATH,
        image_ref=image,
    )
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--user=65534:65534" in command
    assert "--memory=512m" in command
    assert "--memory-swap=512m" in command
    assert any(item.startswith("--security-opt=seccomp=") for item in command)
    assert command[:2] == (
        "/usr/bin/docker", "--host=unix:///var/run/docker.sock"
    )
    assert not any("docker.sock" in item and "mount=" in item for item in command)
    mounts = [item for item in command if item.startswith("--mount=")]
    assert len(mounts) == 3
    assert all("readonly" in item for item in mounts)
    assert all(str(REPOSITORY.resolve()) not in item for item in mounts)
    assert image in command


def test_runtime_sources_must_equal_basis_commit_blobs(tmp_path: Path) -> None:
    repository = tmp_path / "basis"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Gate17 Test")
    _git(repository, "config", "user.email", "gate17@example.invalid")
    source = repository / "worker.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _git(repository, "add", "worker.py")
    _git(repository, "commit", "--quiet", "-m", "basis")
    basis_commit = _git(repository, "rev-parse", "HEAD")
    bindings = detached._runtime_source_bindings_v1(
        ((source, "worker.py"),),
        basis_commit=basis_commit,
        git_executable=detached._resolve_git_executable(),
        repository=repository,
    )
    assert bindings["basis_commit_sha1"] == basis_commit
    assert bindings["committed_source_files"][0]["basis_tree_blob_sha1"] == _git(
        repository, "rev-parse", "HEAD:worker.py"
    )

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(detached.Purpose4DetachedAuditError) as captured:
        detached._runtime_source_bindings_v1(
            ((source, "worker.py"),),
            basis_commit=basis_commit,
            git_executable=detached._resolve_git_executable(),
            repository=repository,
        )
    assert captured.value.code == detached.FAIL_RUNTIME_BASIS


def _formal_response_fixture() -> tuple[dict[str, object], dict[str, object]]:
    receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    key_id = bytes(range(16))
    timestamp = 1_800_000_000
    fields = {
        "parent_dsl_version_digest": id_digest_v1("hegel-old-dsl-v1.0.0"),
        "parent_freeze_version_digest": id_digest_v1("hegel-freeze-p2b-p3-v1.0.2"),
        "parent_repository_commit_id": git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1),
        "audit_bundle_root": bytes.fromhex(receipt["audit_bundle_root"]),
        "absence_reason_bitmask": 0b1111,
        "auditor_key_id": key_id,
        "audited_at_unix_seconds": timestamp,
    }
    cbor = encode_formal_object("ParentManifestAbsenceAttestationV2", fields)
    root = candidate_content_root("ParentManifestAbsenceAttestationV2", fields)
    preimage = external_signature_preimage_v1(
        OBJECT_TAGS["ParentManifestAbsenceAttestationV2"], root, 4, 0
    )
    basis_commit = "44" * 20
    image_ref = "python@sha256:" + "55" * 32
    git_binding = {
        "container_path": "/runtime/bin/git",
        "byte_length": 1,
        "sha256": "22" * 32,
        "version": "git version test",
    }
    runtime_inventory = {
        "files": [
            {"path": "bin/git", "byte_length": 1, "sha256": "22" * 32},
            {"path": "worker.py", "byte_length": 2, "sha256": "66" * 32},
        ],
        "file_count": 2,
        "inventory_sha256": "33" * 32,
    }
    source_binding_body: dict[str, object] = {
        "schema": "hegel-gate17-purpose4-runtime-source-bindings/1",
        "basis_commit_sha1": basis_commit,
        "committed_source_files": [
            {
                "runtime_path": "worker.py",
                "repository_path": "Hegel Machine/tools/worker.py",
                "basis_tree_mode": "100644",
                "basis_tree_blob_sha1": "77" * 20,
                "byte_length": 2,
                "sha256": "66" * 32,
            }
        ],
        "external_git_dependency": git_binding,
    }
    source_binding_body["binding_sha256"] = hashlib.sha256(
        detached.RUNTIME_SOURCE_BINDING_DOMAIN
        + detached._canonical_json(source_binding_body)
    ).hexdigest()
    request = detached.build_purpose4_actor_request_v1(
        snapshot_manifest={
            "basis_commit_sha1": basis_commit,
            "manifest_sha256": "11" * 32,
            "git_runtime_binding": git_binding,
        },
        runtime_inventory=runtime_inventory,
        runtime_source_bindings=source_binding_body,
        basis_commit=basis_commit,
        actor_image_ref=image_ref,
        auditor_key_id=key_id,
        audited_at_unix_seconds=timestamp,
    )
    environment = {
        "HEGEL_ACTOR_IMAGE_REF": image_ref,
        "HEGEL_ACTOR_PROFILE_ID": "hegel-owner-accepted-container-technical-actors-v1",
        "HEGEL_PURPOSE_ID": "4",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/runtime/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    live_probe_body: dict[str, object] = {
        "schema": "hegel-gate17-purpose4-live-probe/1",
        "profile_id": "hegel-owner-accepted-container-technical-actors-v1",
        "purpose_id": 4,
        "implementation": "python-ctypes-v1",
        "actor_image_ref": image_ref,
        "identity": {"uid": 65534, "gid": 65534, "pid": 1},
        "proc_status": {
            "CapInh": "0000000000000000",
            "CapPrm": "0000000000000000",
            "CapEff": "0000000000000000",
            "CapBnd": "0000000000000000",
            "CapAmb": "0000000000000000",
            "NoNewPrivs": 1,
            "Seccomp": 2,
        },
        "namespaces": {
            "pid": "pid:[1]",
            "mnt": "mnt:[2]",
            "net": "net:[3]",
            "ipc": "ipc:[4]",
            "uts": "uts:[5]",
        },
        "network_interfaces": ["lo"],
        "syscall_probes": [
            {"probe_id": name, "return_value": -1, "errno": 1}
            for name in detached.LIVE_PROBE_SYSCALL_IDS
        ],
        "filesystem_probes": {
            "root_write": {"denied": True, "errno": 30},
            "snapshot_write": {"denied": True, "errno": 30},
            "runtime_write": {"denied": True, "errno": 30},
            "request_write": {"denied": True, "errno": 30},
            "tmp_write": {"denied": False, "errno": 0},
            "forbidden_paths_present": [],
            "cross_purpose_paths_present": [],
        },
        "environment": environment,
        "open_fds": [0, 1, 2],
        "cgroup_limits": {
            "memory_max": str(512 * 1024 * 1024),
            "memory_swap_max": "0",
            "pids_max": "64",
        },
        "required_checks": {"synthetic_fixture": True},
        "all_required_checks_passed": True,
    }
    live_probe_body["receipt_sha256"] = hashlib.sha256(
        detached._canonical_json(live_probe_body)
    ).hexdigest()
    body: dict[str, object] = {
        "schema": detached.ACTOR_RESPONSE_SCHEMA,
        "purpose_id": 4,
        "basis_commit_sha1": basis_commit,
        "actor_image_ref": image_ref,
        "request_sha256": request["request_sha256"],
        "snapshot_manifest_sha256": "11" * 32,
        "runtime_inventory_sha256": "33" * 32,
        "runtime_source_binding_sha256": source_binding_body["binding_sha256"],
        "git_runtime_binding": request["snapshot_manifest"]["git_runtime_binding"],
        "isolation_live_probe_receipt": live_probe_body,
        "parent_absence_public_receipt": receipt,
        "attestation_cbor_hex": cbor.hex(),
        "attestation_root_hex": root.hex(),
        "signature_preimage_hex": preimage.hex(),
        "signer_purpose_id": 4,
        "signer_key_epoch": 0,
        "signature_present": False,
        "private_key_seed_marker_accessed": False,
        "network_access_performed": False,
    }
    body["response_sha256"] = hashlib.sha256(detached._canonical_json(body)).hexdigest()
    return request, body


def test_public_response_binds_complete_path_and_content_receipt() -> None:
    request, response = _formal_response_fixture()
    validated = detached.validate_purpose4_actor_response_v1(response, request=request)
    assert validated["signature_present"] is False
    assert validated["attestation_root_hex"] == response["attestation_root_hex"]


def test_tampered_path_receipt_fails_even_with_recomputed_transport_digest() -> None:
    request, response = _formal_response_fixture()
    tampered = copy.deepcopy(response)
    tampered["parent_absence_public_receipt"]["predicates"][0][
        "matched_unique_path_count"
    ] = 1
    tampered["response_sha256"] = hashlib.sha256(
        detached._canonical_json(
            {key: value for key, value in tampered.items() if key != "response_sha256"}
        )
    ).hexdigest()
    with pytest.raises(detached.Purpose4DetachedAuditError) as captured:
        detached.validate_purpose4_actor_response_v1(tampered, request=request)
    assert captured.value.code == detached.FAIL_RECEIPT_INCOMPLETE


def test_tampered_live_probe_fails_even_with_recomputed_receipt_and_transport() -> None:
    request, response = _formal_response_fixture()
    tampered = copy.deepcopy(response)
    probe = tampered["isolation_live_probe_receipt"]
    probe["cgroup_limits"]["memory_max"] = str(1024 * 1024 * 1024)
    probe["receipt_sha256"] = hashlib.sha256(
        detached._canonical_json(
            {key: value for key, value in probe.items() if key != "receipt_sha256"}
        )
    ).hexdigest()
    tampered["response_sha256"] = hashlib.sha256(
        detached._canonical_json(
            {key: value for key, value in tampered.items() if key != "response_sha256"}
        )
    ).hexdigest()
    with pytest.raises(detached.Purpose4DetachedAuditError) as captured:
        detached.validate_purpose4_actor_response_v1(tampered, request=request)
    assert captured.value.code == detached.FAIL_ACTOR_POLICY


def test_tampered_basis_commit_fails_even_with_recomputed_transport_digest() -> None:
    request, response = _formal_response_fixture()
    tampered = copy.deepcopy(response)
    tampered["basis_commit_sha1"] = "99" * 20
    tampered["response_sha256"] = hashlib.sha256(
        detached._canonical_json(
            {key: value for key, value in tampered.items() if key != "response_sha256"}
        )
    ).hexdigest()
    with pytest.raises(detached.Purpose4DetachedAuditError) as captured:
        detached.validate_purpose4_actor_response_v1(tampered, request=request)
    assert captured.value.code == detached.FAIL_ACTOR_RESPONSE


@pytest.mark.skipif(
    os.environ.get("HEGEL_RUN_GATE17_DETACHED_FULL") != "1",
    reason="full frozen-parent detached snapshot and 1.52 GB content replay is opt-in",
)
def test_full_frozen_parent_actor_replay_in_offline_container() -> None:
    with detached.prepare_detached_parent_snapshot_v1(
        REPOSITORY,
        basis_commit=_git(REPOSITORY, "rev-parse", "HEAD"),
    ) as snapshot:
        response = detached.run_purpose4_detached_audit_v1(
            snapshot,
            auditor_key_id=bytes(range(16)),
            audited_at_unix_seconds=1_800_000_000,
        )
    assert response["parent_absence_public_receipt"]["audit_bundle_root"] == (
        "136c9eee4c616d9f55dae699cb467e56921ce4706943ae87a5ad89bf9d82ff51"
    )
    assert response["signature_present"] is False
    assert response["private_key_seed_marker_accessed"] is False

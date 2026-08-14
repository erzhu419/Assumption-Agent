from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, TOOLS.as_posix())

import build_phase2b_recognizer_runtime_image_v3 as builder  # noqa: E402


IMAGE_DIGEST = "sha256:" + "a" * 64
ROOTFS_LAYER = "sha256:" + "b" * 64


def _image_inspect(
    *,
    wrong_user: bool = False,
    wrong_args_escaped: bool = False,
    extra_config_field: bool = False,
) -> bytes:
    config: dict[str, object] = {
        "ArgsEscaped": False if wrong_args_escaped else True,
        "User": "0:0" if wrong_user else builder.RECOGNIZER_RUNTIME_UID_GID_V3,
        "Env": list(builder.FROZEN_IMAGE_ENVIRONMENT_V3),
        "Entrypoint": list(builder.FROZEN_ENTRYPOINT_V3),
        "Cmd": list(builder.FROZEN_COMMAND_V3),
        "WorkingDir": "/opt/hegel",
        "Labels": dict(builder.FROZEN_IMAGE_LABELS_V3),
    }
    if extra_config_field:
        config["Healthcheck"] = None
    value = [
        {
            "Id": IMAGE_DIGEST,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": config,
            "RootFS": {"Type": "layers", "Layers": [ROOTFS_LAYER]},
        }
    ]
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


class FakeDocker:
    def __init__(
        self,
        *,
        wrong_image: bool = False,
        wrong_args_escaped: bool = False,
        bad_smoke: bool = False,
        wrong_smoke_stdout: bool = False,
        wrong_smoke_stderr: bool = False,
        extra_image_config: bool = False,
        wrong_embedded_seccomp: bool = False,
        path_seccomp_token: bool = False,
        forged_absence: bool = False,
        smoke_identity_tamper: str | None = None,
        isolation_tamper: str | None = None,
        buildx_unavailable: bool = False,
        wrong_buildx_version: bool = False,
        tamper_buildx_during_version: bool = False,
        tamper_buildx_during_build: bool = False,
        build_backend_unavailable: bool = False,
    ) -> None:
        self.wrong_image = wrong_image
        self.wrong_args_escaped = wrong_args_escaped
        self.bad_smoke = bad_smoke
        self.wrong_smoke_stdout = wrong_smoke_stdout
        self.wrong_smoke_stderr = wrong_smoke_stderr
        self.extra_image_config = extra_image_config
        self.wrong_embedded_seccomp = wrong_embedded_seccomp
        self.path_seccomp_token = path_seccomp_token
        self.forged_absence = forged_absence
        self.smoke_identity_tamper = smoke_identity_tamper
        self.isolation_tamper = isolation_tamper
        self.buildx_unavailable = buildx_unavailable
        self.wrong_buildx_version = wrong_buildx_version
        self.tamper_buildx_during_version = tamper_buildx_during_version
        self.tamper_buildx_during_build = tamper_buildx_during_build
        self.build_backend_unavailable = build_backend_unavailable
        self.calls: list[dict[str, Any]] = []
        self.container_name: str | None = None
        self.seccomp_path: str | None = None
        self.private_buildx_path: Path | None = None
        self.removed = False

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        stdin_payload: bytes | None,
        environment: dict[str, str],
        working_directory: Path,
        timeout_seconds: int,
        maximum_stdout_bytes: int,
        maximum_stderr_bytes: int,
    ) -> builder.BoundedDockerResultV3:
        row = {
            "command": command,
            "stdin_payload": stdin_payload,
            "environment": deepcopy(environment),
            "working_directory": working_directory,
            "timeout_seconds": timeout_seconds,
            "maximum_stdout_bytes": maximum_stdout_bytes,
            "maximum_stderr_bytes": maximum_stderr_bytes,
        }
        self.calls.append(row)
        assert command[:2] == (
            builder.DOCKER_EXECUTABLE_V3,
            f"--host={builder.DOCKER_HOST_V3}",
        )
        assert working_directory == builder.ROOT
        assert environment == {
            "BUILDKIT_PROGRESS": "plain",
            "DOCKER_BUILDKIT": "1",
            "DOCKER_CONFIG": environment["DOCKER_CONFIG"],
            "DOCKER_HOST": builder.DOCKER_HOST_V3,
            "HOME": environment["HOME"],
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "SOURCE_DATE_EPOCH": "0",
            "TMPDIR": environment["TMPDIR"],
        }
        if command[2:4] == ("buildx", "version"):
            self.private_buildx_path = (
                Path(environment["DOCKER_CONFIG"])
                / builder.BUILDX_PRIVATE_RELATIVE_PATH_V3
            )
            assert self.private_buildx_path.is_file()
            assert self.private_buildx_path.stat().st_mode & 0o777 == 0o755
            assert self.private_buildx_path.stat().st_size == builder.BUILDX_SIZE_BYTES_V3
            assert hashlib.sha256(self.private_buildx_path.read_bytes()).hexdigest() == (
                builder.BUILDX_SHA256_V3
            )
            if self.tamper_buildx_during_version:
                self.private_buildx_path.write_bytes(b"tampered-during-version")
            if self.buildx_unavailable:
                return builder.BoundedDockerResultV3(
                    command,
                    1,
                    b"",
                    b"docker: 'buildx' is not a docker command.\n",
                )
            if self.wrong_buildx_version:
                return builder.BoundedDockerResultV3(
                    command,
                    0,
                    b"github.com/docker/buildx 0.0.0 unknown\n",
                    b"",
                )
            return builder.BoundedDockerResultV3(
                command,
                0,
                builder.BUILDX_VERSION_STDOUT_V3,
                b"",
            )
        if command[2] == "build":
            if self.tamper_buildx_during_build:
                assert self.private_buildx_path is not None
                self.private_buildx_path.write_bytes(b"tampered-during-build")
            if self.build_backend_unavailable:
                return builder.BoundedDockerResultV3(
                    command,
                    1,
                    b"",
                    b"ERROR: BuildKit backend unavailable\n",
                )
            return builder.BoundedDockerResultV3(command, 0, b"build\n", b"")
        if command[2:4] == ("image", "inspect"):
            return builder.BoundedDockerResultV3(
                command,
                0,
                _image_inspect(
                    wrong_user=self.wrong_image,
                    wrong_args_escaped=self.wrong_args_escaped,
                    extra_config_field=self.extra_image_config,
                ),
                b"",
            )
        if command[2] == "run":
            self.container_name = next(
                value.removeprefix("--name=")
                for value in command
                if value.startswith("--name=")
            )
            self.seccomp_path = next(
                value.removeprefix("--security-opt=seccomp=")
                for value in command
                if value.startswith("--security-opt=seccomp=")
            )
            return builder.BoundedDockerResultV3(
                command,
                0 if self.bad_smoke else 2,
                (
                    b"unexpected"
                    if self.bad_smoke
                    else b"unexpected-stdout\n"
                    if self.wrong_smoke_stdout
                    else b""
                ),
                (
                    b""
                    if self.bad_smoke
                    else (
                        b"forged-failure\n"
                        if self.wrong_smoke_stderr
                        else builder.NEGATIVE_SMOKE_STDERR_V3
                    )
                ),
            )
        if command[2:4] == ("container", "inspect"):
            assert self.container_name is not None
            assert command[-1] == self.container_name
            if self.removed:
                if self.forged_absence:
                    return builder.BoundedDockerResultV3(
                        command,
                        1,
                        b"",
                        b"Error response from daemon: permission denied\n",
                    )
                return builder.BoundedDockerResultV3(
                    command,
                    1,
                    b"[]\n",
                    (
                        "Error response from daemon: No such container: "
                        + self.container_name
                        + "\n"
                    ).encode("ascii"),
                )
            assert self.seccomp_path is not None
            embedded_seccomp = json.loads(Path(self.seccomp_path).read_bytes())
            if self.wrong_embedded_seccomp:
                embedded_seccomp = dict(embedded_seccomp)
                embedded_seccomp["defaultAction"] = "SCMP_ACT_ERRNO"
            identity: dict[str, object] = {
                "Id": "c" * 64,
                "Name": "/" + self.container_name,
                "Path": builder.FROZEN_ENTRYPOINT_V3[0],
                "Args": list(builder.FROZEN_COMMAND_V3),
            }
            identity_tamper_values: dict[str, object] = {
                "Id": "not-a-container-id",
                "Name": "/wrong-container",
                "Path": "/bin/false",
                "Args": ["wrong"],
            }
            if self.smoke_identity_tamper is not None:
                identity[self.smoke_identity_tamper] = identity_tamper_values[
                    self.smoke_identity_tamper
                ]
            host_config: dict[str, object] = {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "AutoRemove": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": [
                    "no-new-privileges:true",
                    "seccomp="
                    + (
                        self.seccomp_path
                        if self.path_seccomp_token
                        else json.dumps(
                            embedded_seccomp,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                ],
                "IpcMode": "none",
                "PidsLimit": 64,
                "NanoCpus": 1_000_000_000,
                "Memory": 536_870_912,
                "MemorySwap": 536_870_912,
                "CgroupnsMode": "private",
                "PidMode": "",
                "UsernsMode": "",
                "Runtime": "runc",
                "Tmpfs": {
                    "/phase2b/input": (
                        "ro,noexec,nosuid,nodev,mode=0500,uid=65534,"
                        "gid=65534,size=1048576"
                    ),
                    "/phase2b/output": (
                        "rw,noexec,nosuid,nodev,mode=0700,uid=65534,"
                        "gid=65534,size=1048576"
                    ),
                    "/tmp": (
                        "rw,noexec,nosuid,nodev,mode=0700,uid=65534,"
                        "gid=65534,size=1048576"
                    ),
                },
                "Binds": None,
            }
            isolation_tamper_values: dict[str, object] = {
                "Privileged": True,
                "AutoRemove": True,
                "CgroupnsMode": "host",
                "PidMode": "host",
                "UsernsMode": "host",
                "Runtime": "not-runc",
                "NanoCpus": 2_000_000_000,
            }
            if self.isolation_tamper is not None:
                host_config[self.isolation_tamper] = isolation_tamper_values[
                    self.isolation_tamper
                ]
            payload = [
                {
                    **identity,
                    "Image": IMAGE_DIGEST,
                    "State": {
                        "Status": "exited",
                        "Running": False,
                        "Dead": False,
                        "OOMKilled": False,
                        "ExitCode": 2,
                        "Error": "",
                    },
                    "Config": {
                        "Image": IMAGE_DIGEST,
                        "User": builder.RECOGNIZER_RUNTIME_UID_GID_V3,
                        "Env": list(builder.FROZEN_IMAGE_ENVIRONMENT_V3),
                        "Entrypoint": list(builder.FROZEN_ENTRYPOINT_V3),
                        "Cmd": list(builder.FROZEN_COMMAND_V3),
                        "WorkingDir": "/opt/hegel",
                        "Volumes": None,
                    },
                    "HostConfig": host_config,
                    "Mounts": [],
                    "NetworkSettings": {"Ports": {}},
                }
            ]
            wire = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
            return builder.BoundedDockerResultV3(command, 0, wire, b"")
        if command[2:4] == ("container", "rm"):
            assert self.container_name is not None
            assert command[-1] == self.container_name
            self.removed = True
            return builder.BoundedDockerResultV3(
                command,
                0,
                (self.container_name + "\n").encode("ascii"),
                b"",
            )
        raise AssertionError(command)


def _independent_source_root(
    entries: tuple[tuple[str, str, int, str], ...],
) -> str:
    preimage = bytearray(builder.RECOGNIZER_SOURCE_ROOT_DOMAIN_V3)
    preimage.extend(struct.pack(">I", len(entries)))
    for installed, context_path, size, digest in entries:
        payload = (ROOT / context_path).read_bytes()
        assert len(payload) == size
        assert hashlib.sha256(payload).hexdigest() == digest
        encoded = installed.encode("ascii")
        preimage.extend(struct.pack(">I", len(encoded)))
        preimage.extend(encoded)
        preimage.extend(struct.pack(">Q", len(payload)))
        preimage.extend(hashlib.sha256(payload).digest())
    return hashlib.sha256(preimage).hexdigest()


def test_frozen_28_entry_source_manifest_and_30_file_ustar_are_exact() -> None:
    first = builder.load_frozen_build_inputs_v3()
    second = builder.load_frozen_build_inputs_v3()
    assert first == second
    assert len(first.source_entries) == 28
    assert len(first.context_entries) == 30
    assert first.source_root_sha256 == builder.RECOGNIZER_SOURCE_ROOT_SHA256_V3
    assert _independent_source_root(first.source_entries) == (
        builder.RECOGNIZER_SOURCE_ROOT_SHA256_V3
    )
    assert builder.RECOGNIZER_SOURCE_ROOT_DOMAIN_V3 == (
        b"HEGEL/PHASE2B/RECOGNIZER_RUNTIME_SOURCE_ROOT/V1\x00"
    )
    assert builder.RECOGNIZER_SOURCE_ROOT_DOMAIN_V3 != (
        b"HEGEL/PHASE2B/SCORER_RUNTIME_SOURCE_ROOT/V1\x00"
    )
    assert builder.NEGATIVE_SMOKE_STDERR_V3 == (
        b'{"disposition":"ABSTAIN","reason":'
        b'"phase2b_recognizer_runtime_v2_failed"}\n'
    )
    assert first.context_bytes == second.context_bytes
    assert hashlib.sha256(first.context_bytes).hexdigest() == first.context_sha256
    assert first.context_bytes[257:263] == b"ustar\x00"

    with tarfile.open(fileobj=io.BytesIO(first.context_bytes), mode="r:") as archive:
        members = archive.getmembers()
        assert tuple(member.name for member in members) == tuple(
            sorted(row[0] for row in first.context_entries)
        )
        assert len(members) == 30
        assert all(member.isfile() for member in members)
        assert all(member.uid == 0 and member.gid == 0 for member in members)
        assert all(member.uname == "" and member.gname == "" for member in members)
        assert all(member.mtime == 0 for member in members)
        for member, expected in zip(members, first.context_entries, strict=True):
            path, size, mode, digest = expected
            assert member.name == path
            assert member.size == size
            assert member.mode == mode
            stream = archive.extractfile(member)
            assert stream is not None
            assert hashlib.sha256(stream.read()).hexdigest() == digest

    context_names = {row[0] for row in first.context_entries}
    assert builder.DOCKERFILE_CONTEXT_PATH_V3 in context_names
    assert builder.DOCKERIGNORE_CONTEXT_PATH_V3 in context_names
    assert builder.PROFILE_PATH_V3 not in context_names
    assert builder.SECCOMP_PATH_V3 not in context_names
    assert first.dockerfile_sha256 == builder.DOCKERFILE_SHA256_V3
    assert first.dockerignore_sha256 == builder.DOCKERIGNORE_SHA256_V3
    assert first.profile_sha256 == builder.PROFILE_FILE_SHA256_V3
    assert first.seccomp_sha256 == builder.SECCOMP_SHA256_V3
    assert first.buildx_sha256 == builder.BUILDX_SHA256_V3
    assert first.buildx_size_bytes == builder.BUILDX_SIZE_BYTES_V3
    assert len(first.buildx_bytes) == builder.BUILDX_SIZE_BYTES_V3
    assert hashlib.sha256(first.buildx_bytes).hexdigest() == builder.BUILDX_SHA256_V3


def test_fake_docker_build_inspect_negative_smoke_cleanup_and_receipt() -> None:
    frozen = builder.load_frozen_build_inputs_v3()
    fake = FakeDocker()
    receipt, payload = builder.build_and_qualify_recognizer_image_v3(runner=fake)

    assert len(fake.calls) == 8
    version = fake.calls[0]
    assert version["command"] == (
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "buildx",
        "version",
    )
    assert version["stdin_payload"] is None
    build = fake.calls[1]
    assert build["command"][:6] == (
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "build",
        "--pull=false",
        "--network=none",
        "--no-cache",
    )
    assert f"--file={builder.DOCKERFILE_CONTEXT_PATH_V3}" in build["command"]
    assert f"--tag={builder.IMAGE_TAG_V3}" in build["command"]
    assert build["command"][-1] == "-"
    assert build["stdin_payload"] == frozen.context_bytes

    assert fake.calls[2]["command"][-1] == builder.IMAGE_TAG_V3
    smoke = fake.calls[3]["command"]
    assert smoke[2] == "run"
    assert "--network=none" in smoke
    assert "--read-only" in smoke
    assert "--cap-drop=ALL" in smoke
    assert "--security-opt=no-new-privileges:true" in smoke
    assert any(value.startswith("--security-opt=seccomp=/tmp/") for value in smoke)
    assert "--user=65534:65534" in smoke
    assert not any(value.startswith("--entrypoint") for value in smoke)
    assert smoke[-1] == IMAGE_DIGEST
    assert not any(value.startswith("--mount=") for value in smoke)
    assert sum(value.startswith("--tmpfs=") for value in smoke) == 3
    assert fake.calls[4]["command"][2:4] == ("container", "inspect")
    assert fake.calls[5]["command"][2:4] == ("container", "rm")
    assert fake.calls[6]["command"][2:4] == ("container", "inspect")
    assert fake.calls[7]["command"][-1] == IMAGE_DIGEST
    assert fake.private_buildx_path is not None
    assert not fake.private_buildx_path.exists()

    assert payload == builder._canonical_json_v3(receipt) + b"\n"
    receipt_without_id = dict(receipt)
    receipt_id = receipt_without_id.pop("receipt_id")
    assert receipt_id == (
        builder.RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_ID_PREFIX_V3
        + hashlib.sha256(
            builder.RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_DOMAIN_V3
            + builder._canonical_json_v3(receipt_without_id)
        ).hexdigest()
    )
    assert receipt["build"]["context_mode"] == (
        "EXACT_30_FILE_DETERMINISTIC_USTAR_STREAM"
    )
    assert receipt["build"]["context_sha256"] == frozen.context_sha256
    assert receipt["build_inputs"]["buildx_plugin_sha256"] == (
        builder.BUILDX_SHA256_V3
    )
    assert receipt["build_inputs"]["buildx_plugin_size_bytes"] == (
        builder.BUILDX_SIZE_BYTES_V3
    )
    assert receipt["build_inputs"]["buildx_version_stdout"] == (
        builder.BUILDX_VERSION_STDOUT_V3.decode("ascii").rstrip("\n")
    )
    assert receipt["image"]["image_digest"] == IMAGE_DIGEST
    assert receipt["image"]["args_escaped"] is True
    assert receipt["image"]["healthcheck"] is None
    assert receipt["image"]["declared_volumes"] is None
    assert receipt["build_inputs"]["profile_content_id"] == builder.PROFILE_CONTENT_ID_V3
    assert receipt["build_inputs"]["profile_canonical_sha256"] == (
        builder.PROFILE_CANONICAL_SHA256_V3
    )
    assert receipt["build_inputs"]["owner_accepted_authority_requirement_id"] == (
        builder.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
    )
    assert receipt["build_inputs"]["source_root_domain_hex"] == (
        builder.RECOGNIZER_SOURCE_ROOT_DOMAIN_V3.hex()
    )
    assert receipt["build_inputs"]["source_root_sha256"] == (
        builder.RECOGNIZER_SOURCE_ROOT_SHA256_V3
    )
    assert receipt["qualification"]["negative_smoke_exit_code"] == 2
    assert receipt["qualification"]["negative_smoke_stderr_sha256"] == (
        hashlib.sha256(builder.NEGATIVE_SMOKE_STDERR_V3).hexdigest()
    )
    assert receipt["qualification"]["container_output_host_mounts"] == 0
    assert {
        name for name, value in receipt["claims"].items() if value is True
    } == {
        "exact_source_closure_verified",
        "local_buildx_backend_verified",
        "local_build_command_verified",
        "local_image_built",
        "negative_smoke_executed",
        "runtime_image_config_verified",
    }
    assert all(
        receipt["claims"][name] is False
        for name in (
            "actual_960_case_run_verified",
            "actual_960_recognizer_executed",
            "actual_960_scorer_executed",
            "actor_profile_authority_verified",
            "authority_requirement_satisfied",
            "recognizer_operation_signature_verified",
            "runtime_attestation_authority_verified",
            "fresh_role_keys_generated",
            "key_generation_performed",
            "custodian_ceremony_executed",
            "signed_execution_registry_materialized",
            "signed_monotonic_time_materialized",
            "store_receipt_materialized",
            "scoring_performed",
            "prediction_scored",
            "formal_gate_evaluation_performed",
            "effect_evidence",
            "formal_c1_report_verified",
            "c1_exit_evidence",
        )
    )


def test_source_drift_fails_before_any_docker_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDocker()
    monkeypatch.setattr(builder, "DOCKERFILE_SHA256_V3", "0" * 64)
    with pytest.raises(builder.Phase2BRecognizerImageBuilderErrorV3, match="digest"):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.calls == []


def test_fixed_local_buildx_digest_tamper_fails_before_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeDocker()
    monkeypatch.setattr(builder, "BUILDX_SHA256_V3", "0" * 64)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="stable absolute file digest differs",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.calls == []


@pytest.mark.parametrize(
    "fake",
    (
        FakeDocker(buildx_unavailable=True),
        FakeDocker(wrong_buildx_version=True),
    ),
)
def test_buildx_backend_unavailable_or_wrong_version_stops_before_build(
    fake: FakeDocker,
) -> None:
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="backend qualification failed",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert len(fake.calls) == 1
    assert fake.calls[0]["command"][2:4] == ("buildx", "version")


def test_qualified_plugin_does_not_mask_unavailable_build_backend() -> None:
    fake = FakeDocker(build_backend_unavailable=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="offline recognizer image build failed",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert len(fake.calls) == 2
    assert fake.calls[0]["command"][2:4] == ("buildx", "version")
    assert fake.calls[1]["command"][2] == "build"


def test_private_buildx_tamper_during_version_is_caught_before_build() -> None:
    fake = FakeDocker(tamper_buildx_during_version=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="stable absolute file metadata differs",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert len(fake.calls) == 1


def test_private_buildx_tamper_during_build_is_caught_before_image_use() -> None:
    fake = FakeDocker(tamper_buildx_during_build=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="stable absolute file metadata differs",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert len(fake.calls) == 2
    assert fake.calls[0]["command"][2:4] == ("buildx", "version")
    assert fake.calls[1]["command"][2] == "build"


def test_wrong_image_config_fails_closed_before_smoke() -> None:
    fake = FakeDocker(wrong_image=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="platform/config/source binding",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert len(fake.calls) == 3


def test_even_null_extra_raw_image_config_field_is_rejected() -> None:
    fake = FakeDocker(extra_image_config=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="platform/config/source binding",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert len(fake.calls) == 3


def test_buildkit_args_escaped_must_be_exact_true() -> None:
    fake = FakeDocker(wrong_args_escaped=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="platform/config/source binding",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert len(fake.calls) == 3


@pytest.mark.parametrize("field", ("Id", "Name", "Path", "Args"))
def test_smoke_container_identity_surface_is_exact(field: str) -> None:
    fake = FakeDocker(smoke_identity_tamper=field)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="security/state differs",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True


@pytest.mark.parametrize(
    "field",
    (
        "Privileged",
        "AutoRemove",
        "CgroupnsMode",
        "PidMode",
        "UsernsMode",
        "Runtime",
        "NanoCpus",
    ),
)
def test_smoke_container_declared_isolation_surface_is_exact(field: str) -> None:
    fake = FakeDocker(isolation_tamper=field)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="security/state differs",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True


def test_bad_smoke_is_rejected_but_container_is_removed_and_absence_checked() -> None:
    fake = FakeDocker(bad_smoke=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="fail-closed exit 2",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True
    assert tuple(row["command"][2:4] for row in fake.calls[-2:]) == (
        ("container", "rm"),
        ("container", "inspect"),
    )


def test_wrong_generic_failure_wire_is_rejected_and_cleaned_up() -> None:
    fake = FakeDocker(wrong_smoke_stderr=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="fail-closed exit 2",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True


def test_nonempty_negative_smoke_stdout_is_rejected_and_cleaned_up() -> None:
    fake = FakeDocker(wrong_smoke_stdout=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="fail-closed exit 2",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True


def test_host_seccomp_path_token_cannot_masquerade_as_embedded_profile() -> None:
    fake = FakeDocker(path_seccomp_token=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="embedded seccomp is not one JSON value",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True


def test_wrong_embedded_seccomp_cannot_pass_smoke_inspect() -> None:
    fake = FakeDocker(wrong_embedded_seccomp=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="security/state differs",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True
    security = fake.calls[4]["command"]
    assert security[2:4] == ("container", "inspect")


def test_arbitrary_daemon_error_cannot_masquerade_as_container_absence() -> None:
    fake = FakeDocker(forged_absence=True)
    with pytest.raises(
        builder.Phase2BRecognizerImageBuilderErrorV3,
        match="removal/absence proof differs",
    ):
        builder.build_and_qualify_recognizer_image_v3(runner=fake)
    assert fake.removed is True
    assert fake.calls[-1]["command"][2:4] == ("container", "inspect")


def test_cli_publishes_exactly_once_without_writing_in_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = Path("/tmp/phase2b-recognizer-v3-builder-test-receipt.json")
    events: list[tuple[str, object]] = []
    payload = b'{"receipt_id":"synthetic"}\n'

    monkeypatch.setattr(
        builder,
        "assert_receipt_target_absent_v3",
        lambda path: events.append(("absent", path)),
    )
    monkeypatch.setattr(
        builder,
        "build_and_qualify_recognizer_image_v3",
        lambda: ({"receipt_id": "synthetic"}, payload),
    )
    monkeypatch.setattr(
        builder,
        "publish_receipt_no_replace_v3",
        lambda path, wire: events.append(("publish", (path, wire))),
    )
    assert builder.main(("--receipt", requested.as_posix())) == 0
    assert events == [
        ("absent", requested),
        ("publish", (requested, payload)),
    ]
    assert not requested.exists()


def test_publication_source_uses_hard_link_exclusive_create_and_fsync() -> None:
    source = (TOOLS / "build_phase2b_recognizer_runtime_image_v3.py").read_text(
        encoding="utf-8"
    )
    assert "os.replace" not in source
    assert "os.O_EXCL" in source
    assert "os.link(" in source
    assert source.count("os.fsync(") >= 5

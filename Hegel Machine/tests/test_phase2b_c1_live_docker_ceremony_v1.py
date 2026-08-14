from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, fields, replace
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from types import MappingProxyType

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import hegel_machine.phase2b_c1_live_docker_ceremony_v1 as ceremony
from hegel_machine.bootstrap import initial_theory
from hegel_machine.phase2b_c1_six_role_wire_v1 import (
    PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256,
    Phase2BC1SixRoleWireDispositionV1,
    phase2b_c1_attempt_id_v1,
)
from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from hegel_machine.phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    ExecutionFreezeManifest,
    frozen_phase2b_protocol,
)
from hegel_machine.phase2b_recognizer_runtime_cli_v2 import (
    encode_execution_freeze_manifest_json_v2,
)
from hegel_machine.phase3_local_runtime_v1 import LocalDockerControlPlaneV1


RECOGNIZER_IMAGE = (
    "sha256:d27247e9b4b6fcbeca38aea81359b29a855c05f7a7a6e9f2c2276d360160054f"
)
RECOGNIZER_ROOT = "d71f96926156b78bbd42d016b04c40aaa3e0abc5dad2510adc05c23e8dd54f54"
SCORER_IMAGE = (
    "sha256:1f89c913b94d2688080a65225fae9564f9c0925215d9ed13060eadc64918a3b9"
)
SCORER_ROOT = PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256
CAMPAIGN_ID = "phase2b_actual_unsealed_960_campaign_v2_" + "a" * 64
ATTEMPT_NONCE_SHA = "d" * 64


def _execution_freeze_bytes() -> bytes:
    protocol = frozen_phase2b_protocol()
    exact = frozen_phase2b_exact_freeze()
    spec_ids = {
        BaselineKind(spec.baseline_id): spec.content_id for spec in exact.baselines
    }
    baselines = tuple(
        BaselineRegistration(
            kind=kind,
            baseline_spec_id=spec_ids[kind],
            implementation_id=f"live_ceremony_fixture_{kind.value}",
            artifact_sha256=f"{index}" * 64,
            frozen_before_holdout_generation=True,
        )
        for index, kind in enumerate(BaselineKind, start=1)
    )
    manifest = ExecutionFreezeManifest(
        protocol_id=protocol.protocol_id,
        exact_freeze_id=exact.freeze_id,
        git_commit="1" * 40,
        recognizer_image_digest=RECOGNIZER_IMAGE,
        configuration_sha256="4" * 64,
        theory_version_id=initial_theory().version_id,
        adapter_implementation_sha256="5" * 64,
        selector_implementation_sha256="6" * 64,
        verifier_registry_sha256="7" * 64,
        baseline_registrations=baselines,
        isolation_profile_id=protocol.isolation_profile.profile_id,
    )
    return encode_execution_freeze_manifest_json_v2(manifest)


EXECUTION_FREEZE_BYTES = _execution_freeze_bytes()
EXECUTION_FREEZE_MANIFEST = ceremony.decode_execution_freeze_manifest_json_v2(
    EXECUTION_FREEZE_BYTES
)
EXECUTION_FREEZE_ID = EXECUTION_FREEZE_MANIFEST.manifest_id
EXECUTION_FREEZE_SHA = hashlib.sha256(EXECUTION_FREEZE_BYTES).hexdigest()


class InjectedBaseException(BaseException):
    pass


def _request() -> ceremony.Phase2BC1LiveDockerCeremonyRequestV1:
    attempt_id = phase2b_c1_attempt_id_v1(
        authority_requirement_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
        actor_profile_content_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
        actor_profile_canonical_sha256=PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
        campaign_id=CAMPAIGN_ID,
        protocol_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
        exact_freeze_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
        execution_freeze_manifest_id=EXECUTION_FREEZE_ID,
        execution_freeze_manifest_sha256=EXECUTION_FREEZE_SHA,
        attempt_nonce_sha256=ATTEMPT_NONCE_SHA,
    )
    return ceremony.Phase2BC1LiveDockerCeremonyRequestV1(
        campaign_id=CAMPAIGN_ID,
        protocol_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
        exact_freeze_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
        execution_freeze_manifest_id=EXECUTION_FREEZE_ID,
        execution_freeze_manifest_sha256=EXECUTION_FREEZE_SHA,
        execution_freeze_manifest_bytes=EXECUTION_FREEZE_BYTES,
        attempt_nonce_sha256=ATTEMPT_NONCE_SHA,
        attempt_id=attempt_id,
        recognizer_image_digest=RECOGNIZER_IMAGE,
        recognizer_source_root_sha256=RECOGNIZER_ROOT,
        scorer_image_digest=SCORER_IMAGE,
        scorer_source_root_sha256=SCORER_ROOT,
    )


def _json_line(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii") + b"\n"


class FakeDocker:
    def __init__(
        self,
        *,
        qualification_failure_purpose: int | None = None,
        namespace_collision: bool = False,
        image_label_drift: bool = False,
        store_output_drift: bool = False,
        absence_output_drift: bool = False,
        actor_create_fault_purpose: int | None = None,
        actor_create_base_exception: bool = False,
        volume_create_fault: bool = False,
        volume_post_create_inspect_fault: bool = False,
        initializer_create_fault: bool = False,
        initializer_start_fault: bool = False,
        cleanup_fault_kind: str | None = None,
        foreign_actor_purpose: int | None = None,
        foreign_volume: bool = False,
        security_drift: str | None = None,
        inspect_policy_drift: str | None = None,
    ) -> None:
        self.qualification_failure_purpose = qualification_failure_purpose
        self.namespace_collision = namespace_collision
        self.image_label_drift = image_label_drift
        self.store_output_drift = store_output_drift
        self.absence_output_drift = absence_output_drift
        self.actor_create_fault_purpose = actor_create_fault_purpose
        self.actor_create_base_exception = actor_create_base_exception
        self.volume_create_fault = volume_create_fault
        self.volume_post_create_inspect_fault = volume_post_create_inspect_fault
        self.initializer_create_fault = initializer_create_fault
        self.initializer_start_fault = initializer_start_fault
        self.cleanup_fault_kind = cleanup_fault_kind
        self.foreign_actor_purpose = foreign_actor_purpose
        self.foreign_volume = foreign_volume
        self.security_drift = security_drift
        self.inspect_policy_drift = inspect_policy_drift
        self.volume_create_fault_raised = False
        self.volume_post_inspect_fault_raised = False
        self.initializer_create_fault_raised = False
        self.initializer_start_fault_raised = False
        self.actor_create_fault_raised = False
        self.cleanup_fault_raised = False
        self.foreign_volume_name: str | None = None
        self.last_created_volume: str | None = None
        self.volumes: set[str] = set()
        self.volume_labels: dict[str, dict[str, str]] = {}
        self.containers: dict[str, dict[str, object]] = {}
        self.keys: dict[int, Ed25519PrivateKey] = {}
        self.qualifications: list[int] = []
        self.events: list[str] = []
        self.calls: list[tuple[tuple[str, ...], bytes | None]] = []
        self.counter = 0
        self.container_sequence = 100

    def _result(
        self, command: tuple[str, ...], *, returncode: int = 0,
        stdout: bytes = b"", stderr: bytes = b"",
    ) -> ceremony._BoundedProcessV1:
        self.counter += 1
        return ceremony._BoundedProcessV1(
            command=command, returncode=returncode, stdout=stdout, stderr=stderr,
            started_monotonic_ns=self.counter * 2,
            finished_monotonic_ns=self.counter * 2 + 1,
        )

    def _image(self, digest: str) -> dict[str, object]:
        recognizer = digest == RECOGNIZER_IMAGE
        labels: dict[str, str] = {
            "org.hegel-machine.phase2b.actor-profile.content-id": (
                PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID
            ),
            "org.hegel-machine.phase2b.actor-profile.canonical-sha256": (
                PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256
            ),
        }
        if recognizer:
            labels[
                "org.hegel-machine.phase2b.owner-accepted-authority-requirement.id"
            ] = PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID
            labels[
                "org.hegel-machine.phase2b.recognizer-source-root.sha256"
            ] = RECOGNIZER_ROOT
        else:
            labels[
                "org.hegel-machine.phase2b.scorer-source-root.sha256"
            ] = SCORER_ROOT
        if self.image_label_drift:
            labels["org.hegel-machine.phase2b.actor-profile.content-id"] = "drift"
        return {
            "Id": digest,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {
                "ArgsEscaped": True,
                "Cmd": (
                    ["--input", "/phase2b/input", "--output", "/phase2b/output"]
                    if recognizer
                    else [
                        "--input",
                        "/phase2b/input/phase2b_scorer_bundle_v1.bin",
                        "--output",
                        "/phase2b/output",
                    ]
                ),
                "Entrypoint": [
                    "/opt/hegel/bin/phase2b-recognizer"
                    if recognizer
                    else "/opt/hegel/bin/phase2b-scorer"
                ],
                "Env": list(ceremony._FINAL_IMAGE_ENVIRONMENT_V1),
                "Labels": labels,
                "User": "65534:65534",
                "WorkingDir": "/opt/hegel",
            },
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + "e" * 64]},
        }

    def _container_by_reference(self, reference: str) -> tuple[str, dict[str, object]] | None:
        if reference in self.containers:
            return reference, self.containers[reference]
        for name, row in self.containers.items():
            if row["id"] == reference:
                return name, row
        return None

    @staticmethod
    def _raise_fault(*, base_exception: bool) -> None:
        if base_exception:
            raise InjectedBaseException("injected base exception")
        raise ceremony.Phase2BC1LiveDockerCeremonyErrorV1(
            ceremony.FAIL_DOCKER_TIMEOUT, "injected post-side-effect timeout"
        )

    def __call__(
        self,
        command: tuple[str, ...] | list[str],
        *,
        environment: MappingProxyType[str, str] | dict[str, str],
        working_directory: Path,
        stdin_payload: bytes | None = None,
        **_: object,
    ) -> ceremony._BoundedProcessV1:
        cmd = tuple(command)
        self.calls.append((cmd, stdin_payload))
        assert cmd[:2] == ("/usr/bin/docker", "--host=unix:///var/run/docker.sock")
        args = cmd[2:]

        if args[:2] == ("image", "inspect"):
            return self._result(cmd, stdout=_json_line([self._image(args[2])]))

        if args[:2] == ("volume", "inspect"):
            name = args[2]
            if (
                self.volume_post_create_inspect_fault
                and not self.volume_post_inspect_fault_raised
                and name == self.last_created_volume
            ):
                self.volume_post_inspect_fault_raised = True
                raise InjectedBaseException("post-create volume inspect fault")
            if name not in self.volumes:
                if self.foreign_volume and self.foreign_volume_name is None:
                    self.foreign_volume_name = name
                if name == self.foreign_volume_name:
                    return self._result(
                        cmd,
                        stdout=_json_line(
                            [{
                                "Name": name,
                                "Driver": "local",
                                "Scope": "local",
                                "Options": None,
                                "Labels": {"foreign": "preserve"},
                            }]
                        ),
                    )
                if self.absence_output_drift:
                    self.absence_output_drift = False
                    return self._result(
                        cmd, returncode=1, stderr=b"permission denied\n"
                    )
                return self._result(
                    cmd,
                    returncode=1,
                    stdout=b"[]\n",
                    stderr=(
                        f"Error response from daemon: get {name}: no such volume\n"
                    ).encode("ascii"),
                )
            return self._result(
                cmd,
                stdout=_json_line(
                    [
                        {
                            "Name": name,
                            "Driver": "local",
                            "Scope": "local",
                            "Options": None,
                            "Labels": self.volume_labels[name],
                        }
                    ]
                ),
            )
        if args[:2] == ("volume", "create"):
            name = args[-1]
            assert re.fullmatch(r"hp2bc1v-[0-9a-f]{48}", name)
            assert name not in self.volumes
            self.volumes.add(name)
            self.last_created_volume = name
            self.volume_labels[name] = dict(
                item.removeprefix("--label=").split("=", 1)
                for item in args
                if item.startswith("--label=")
            )
            self.events.append("volume-create")
            if self.volume_create_fault and not self.volume_create_fault_raised:
                self.volume_create_fault_raised = True
                self._raise_fault(base_exception=False)
            return self._result(cmd, stdout=(name + "\n").encode("ascii"))
        if args[:2] == ("volume", "rm"):
            name = args[-1]
            if name not in self.volumes:
                return self._result(cmd, returncode=1, stderr=b"not found\n")
            self.volumes.remove(name)
            del self.volume_labels[name]
            self.events.append("volume-rm")
            if (
                self.cleanup_fault_kind in {"ordinary", "base"}
                and not self.cleanup_fault_raised
            ):
                # The injected fault occurs before reporting success but after
                # the daemon-side deletion, exercising BaseException capture
                # while allowing authoritative absence to succeed.
                self.cleanup_fault_raised = True
                if self.cleanup_fault_kind == "base":
                    raise InjectedBaseException("cleanup volume rm base fault")
                raise RuntimeError("cleanup volume rm ordinary fault")
            return self._result(cmd, stdout=(name + "\n").encode("ascii"))

        if args[:2] == ("container", "create"):
            name = next(item.split("=", 1)[1] for item in args if item.startswith("--name="))
            entrypoint_index = args.index("--entrypoint=/usr/local/bin/python3")
            image = args[entrypoint_index + 1]
            command_args = list(args[entrypoint_index + 2 :])
            explicit_labels = dict(
                item.removeprefix("--label=").split("=", 1)
                for item in args
                if item.startswith("--label=")
            )
            labels = dict(self._image(image)["Config"]["Labels"])
            labels.update(explicit_labels)
            kind = explicit_labels[ceremony._OBJECT_KIND_LABEL]
            purpose = int(explicit_labels[ceremony._PURPOSE_LABEL]) if kind == "actor" else 0
            role = explicit_labels.get(ceremony._ROLE_LABEL, "")
            mounts: list[tuple[str, str]] = []
            for item in args:
                if item.startswith("--mount=type=volume"):
                    values = dict(part.split("=", 1) for part in item.removeprefix("--mount=").split(",") if "=" in part)
                    mounts.append((values["dst"], values["src"]))
            if kind == "actor":
                paths = json.loads(
                    base64.b64decode(command_args[-1], validate=True).decode("ascii")
                )
                assert [path for path, _ in mounts] == paths
            self.container_sequence += 1
            container_id = f"{self.container_sequence:064x}"
            tmpfs = dict(
                item.removeprefix("--tmpfs=").split(":", 1)
                for item in args if item.startswith("--tmpfs=")
            )
            cap_add = [
                item.removeprefix("--cap-add=")
                for item in args if item.startswith("--cap-add=")
            ]
            self.containers[name] = {
                "id": container_id,
                "image": image,
                "labels": labels,
                "kind": kind,
                "purpose": purpose,
                "role": role,
                "mounts": mounts,
                "status": "created",
                "args": command_args,
                "user": next(item.removeprefix("--user=") for item in args if item.startswith("--user=")),
                "environment": [item.removeprefix("--env=") for item in args if item.startswith("--env=")],
                "memory": int(next(item.removeprefix("--memory=") for item in args if item.startswith("--memory="))),
                "pids": int(next(item.removeprefix("--pids-limit=") for item in args if item.startswith("--pids-limit="))),
                "nofile": int(next(item.removeprefix("--ulimit=nofile=").split(":", 1)[0] for item in args if item.startswith("--ulimit=nofile="))),
                "tmpfs": tmpfs or None,
                "cap_add": cap_add or None,
            }
            self.events.append(
                f"create-{purpose}" if kind == "actor" else "initializer-create"
            )
            if (
                kind == "actor"
                and purpose == self.actor_create_fault_purpose
                and not self.actor_create_fault_raised
            ):
                self.actor_create_fault_raised = True
                self._raise_fault(
                    base_exception=self.actor_create_base_exception
                )
            if (
                kind == "volume-initializer"
                and self.initializer_create_fault
                and not self.initializer_create_fault_raised
            ):
                self.initializer_create_fault_raised = True
                self._raise_fault(base_exception=False)
            return self._result(cmd, stdout=(container_id + "\n").encode("ascii"))
        if args[:2] == ("container", "start"):
            attached = args[2] == "--attach"
            reference = args[3] if attached else args[2]
            resolved = self._container_by_reference(reference)
            assert resolved is not None
            _, row = resolved
            if attached:
                assert row["kind"] == "volume-initializer"
                row["status"] = "exited"
                self.events.append("volume-init")
                if (
                    self.initializer_start_fault
                    and not self.initializer_start_fault_raised
                ):
                    self.initializer_start_fault_raised = True
                    self._raise_fault(base_exception=False)
                return self._result(cmd)
            row["status"] = "running"
            purpose = row["purpose"]
            self.events.append(f"start-{purpose}")
            return self._result(cmd, stdout=(reference + "\n").encode("ascii"))
        if args[:2] == ("container", "inspect"):
            reference = args[2]
            resolved = self._container_by_reference(reference)
            if resolved is None:
                if (
                    self.foreign_actor_purpose is not None
                    and reference.endswith(f"-p{self.foreign_actor_purpose}")
                ):
                    return self._result(
                        cmd,
                        stdout=_json_line(
                            [{
                                "Id": "f" * 64,
                                "Name": "/" + reference,
                                "Image": SCORER_IMAGE,
                                "Config": {"Labels": {"foreign": "preserve"}},
                            }]
                        ),
                    )
                return self._result(
                    cmd,
                    returncode=1,
                    stdout=b"[]\n",
                    stderr=(
                        f"Error response from daemon: No such container: {reference}\n"
                    ).encode("ascii"),
                )
            name, row = resolved
            seccomp_value = json.loads(
                ceremony.SECCOMP_PROFILE_PATH_V1.read_text(encoding="utf-8")
            )
            seccomp_wire = json.dumps(
                seccomp_value, ensure_ascii=True, separators=(",", ":"),
                sort_keys=True,
            )
            status = row["status"]
            value = {
                "Id": row["id"],
                "Name": "/" + name,
                "Image": row["image"],
                "Path": "/usr/local/bin/python3",
                "Args": row["args"],
                "Config": {
                    "Hostname": name,
                    "User": row["user"],
                    "Image": row["image"],
                    "Entrypoint": ["/usr/local/bin/python3"],
                    "Cmd": row["args"],
                    "WorkingDir": "/opt/hegel",
                    "Env": row["environment"],
                    "Labels": row["labels"],
                    "Healthcheck": {"Test": ["NONE"]},
                    "Volumes": None,
                    "StopTimeout": 1,
                },
                "HostConfig": {
                    "NetworkMode": "none",
                    "IpcMode": "private",
                    "ReadonlyRootfs": True,
                    "Privileged": False,
                    "AutoRemove": False,
                    "CapDrop": ["ALL"],
                    "CapAdd": row["cap_add"],
                    "SecurityOpt": [
                        "no-new-privileges:true",
                        "seccomp=" + seccomp_wire,
                    ],
                    "Memory": row["memory"],
                    "MemorySwap": row["memory"],
                    "PidsLimit": row["pids"],
                    "NanoCpus": 0,
                    "CpuShares": 0,
                    "CpuPeriod": 0,
                    "CpuQuota": 0,
                    "CpuRealtimePeriod": 0,
                    "CpuRealtimeRuntime": 0,
                    "CpusetCpus": "",
                    "CpusetMems": "",
                    "BlkioWeight": 0,
                    "BlkioWeightDevice": [],
                    "BlkioDeviceReadBps": [],
                    "BlkioDeviceWriteBps": [],
                    "BlkioDeviceReadIOps": [],
                    "BlkioDeviceWriteIOps": [],
                    "MemoryReservation": 0,
                    "MemorySwappiness": None,
                    "CpuCount": 0,
                    "CpuPercent": 0,
                    "IOMaximumIOps": 0,
                    "IOMaximumBandwidth": 0,
                    "Ulimits": [{"Name": "nofile", "Hard": row["nofile"], "Soft": row["nofile"]}],
                    "Tmpfs": row["tmpfs"],
                    "Runtime": "runc",
                    "Isolation": "",
                    "Cgroup": "",
                    "CgroupParent": "",
                    "CgroupnsMode": "private",
                    "PidMode": "",
                    "UsernsMode": "",
                    "UTSMode": "",
                    "Devices": [],
                    "DeviceCgroupRules": None,
                    "DeviceRequests": None,
                    "Binds": None,
                    "VolumesFrom": None,
                    "VolumeDriver": "",
                    "GroupAdd": None,
                    "Links": None,
                    "PortBindings": {},
                    "PublishAllPorts": False,
                    "Dns": None,
                    "DnsOptions": [],
                    "DnsSearch": [],
                    "ExtraHosts": None,
                    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                    "ContainerIDFile": "",
                    "ConsoleSize": [0, 0],
                    "ShmSize": 67_108_864,
                    "OomScoreAdj": 0,
                    "OomKillDisable": None,
                    "MaskedPaths": list(ceremony._TARGET_ENGINE_MASKED_PATHS_V1),
                    "ReadonlyPaths": list(ceremony._TARGET_ENGINE_READONLY_PATHS_V1),
                },
                "State": {
                    "Status": status,
                    "Running": status == "running",
                    "Paused": False,
                    "Restarting": False,
                    "OOMKilled": False,
                    "Dead": False,
                    "ExitCode": 0,
                    "Pid": int(row["id"][-8:], 16) if status == "running" else 0,
                },
                "Mounts": [
                    {
                        "Type": "volume",
                        "Name": volume,
                        "Source": f"/var/lib/docker/volumes/{volume}/_data",
                        "Destination": path,
                        "Driver": "local",
                        "Mode": "z",
                        "RW": True,
                        "Propagation": "",
                    }
                    for path, volume in row["mounts"]
                ],
            }
            if row["kind"] == "actor" and self.security_drift is not None:
                security = value["HostConfig"]["SecurityOpt"]
                if self.security_drift == "duplicate":
                    security.append(security[1])
                elif self.security_drift == "path":
                    security[1] = "seccomp=/tmp/untrusted.json"
                elif self.security_drift == "noncanonical":
                    security[1] = "seccomp=" + json.dumps(seccomp_value)
                elif self.security_drift == "wrong-object":
                    security[1] = 'seccomp={"defaultAction":"SCMP_ACT_ALLOW"}'
                else:
                    raise AssertionError(self.security_drift)
            if row["kind"] == "actor" and self.inspect_policy_drift is not None:
                if self.inspect_policy_drift == "extra-mount":
                    value["Mounts"].append(
                        {
                            "Type": "bind", "Source": "/host", "Destination": "/host",
                            "Driver": "", "Mode": "", "RW": False, "Propagation": "rprivate",
                        }
                    )
                elif self.inspect_policy_drift == "readonly-mount":
                    value["Mounts"][0]["RW"] = False
                elif self.inspect_policy_drift == "runtime":
                    value["HostConfig"]["Runtime"] = "kata"
                elif self.inspect_policy_drift == "args":
                    value["Args"] = ["drift"]
                else:
                    raise AssertionError(self.inspect_policy_drift)
            return self._result(cmd, stdout=_json_line([value]))
        if args[:3] == ("container", "rm", "--force"):
            reference = args[3]
            resolved = self._container_by_reference(reference)
            if resolved is None:
                return self._result(
                    cmd,
                    stderr=(
                        f"Error response from daemon: No such container: {reference}\n"
                    ).encode("ascii"),
                )
            name, removed_row = resolved
            del self.containers[name]
            self.events.append("container-rm")
            if (
                self.cleanup_fault_kind in {"container-ordinary", "container-base"}
                and removed_row["kind"] == "actor"
                and not self.cleanup_fault_raised
            ):
                self.cleanup_fault_raised = True
                if self.cleanup_fault_kind == "container-base":
                    raise InjectedBaseException("cleanup container rm base fault")
                raise RuntimeError("cleanup container rm ordinary fault")
            return self._result(cmd, stdout=(reference + "\n").encode("ascii"))

        if args[:3] == ("container", "exec", "--interactive"):
            reference = args[3]
            resolved = self._container_by_reference(reference)
            assert resolved is not None
            _, row = resolved
            assert row["status"] == "running"
            operation = args[-4]
            purpose = int(args[-3])
            role = args[-2]
            assert purpose == row["purpose"] and role == row["role"]
            assert stdin_payload is not None
            if operation == "qualify":
                assert not self.keys
                value = json.loads(stdin_payload.decode("ascii"))
                self.qualifications.append(purpose)
                self.events.append(f"qualify-{purpose}")
                namespaces = {
                    namespace: f"{namespace}:[{4000 + (1 if self.namespace_collision and purpose == 6 else purpose)}]"
                    for namespace in ceremony._NAMESPACE_ORDER
                }
                output = {
                    "operation": "QUALIFIED",
                    "purpose_id": purpose,
                    "role": role,
                    "uid": 65534,
                    "gid": 65534,
                    "nonce_sha256": value["nonce_sha256"],
                    "mounts": [
                        {
                            "path": path,
                            "device": 100 + purpose,
                            "mode_octal": "0700",
                            "owner_uid": 65534,
                            "owner_gid": 65534,
                            "fsync": True,
                            "rename": True,
                            "hardlink_no_replace": True,
                        }
                        for path in value["mount_paths"]
                    ],
                    "namespaces": namespaces,
                    "openssl_sha256": ceremony.OPENSSL_SHA256_V1,
                    "openssl_version": ceremony.OPENSSL_VERSION_V1,
                    "key_material_created": purpose == self.qualification_failure_purpose,
                }
                return self._result(cmd, stdout=_json_line(output))
            if operation == "keygen":
                assert self.qualifications == [1, 2, 3, 4, 5, 6]
                assert len(self.containers) == 6 and all(
                    item["status"] == "running"
                    for item in self.containers.values()
                )
                self.events.append(f"keygen-{purpose}")
                private = Ed25519PrivateKey.generate()
                self.keys[purpose] = private
                public = private.public_key().public_bytes(
                    serialization.Encoding.Raw, serialization.PublicFormat.Raw
                )
                spki = bytes.fromhex("302a300506032b6570032100") + public
                output = {
                    "operation": "KEY_GENERATED",
                    "purpose_id": purpose,
                    "role": role,
                    "algorithm": "Ed25519",
                    "public_key_hex": public.hex(),
                    "public_spki_der_sha256": hashlib.sha256(spki).hexdigest(),
                    "private_key_exported": False,
                    "private_key_file_mode_octal": "0600",
                    "file_fsync": True,
                    "directory_fsync": True,
                    "hardlink_no_replace": True,
                }
                return self._result(cmd, stdout=_json_line(output))
            if operation == "sign":
                self.events.append("governance-sign")
                signature = self.keys[purpose].sign(stdin_payload)
                return self._result(
                    cmd,
                    stdout=_json_line(
                        {
                            "operation": "SIGNED",
                            "purpose_id": purpose,
                            "role": role,
                            "preimage_sha256": hashlib.sha256(stdin_payload).hexdigest(),
                            "signature_hex": signature.hex(),
                        }
                    ),
                )
            if operation == "store-and-sign":
                self.events.append("store-and-sign")
                value = json.loads(stdin_payload.decode("ascii"))
                content = base64.b64decode(value["content_b64"], validate=True)
                preimage = base64.b64decode(value["preimage_b64"], validate=True)
                signature = self.keys[purpose].sign(preimage)
                output = {
                    "operation": "STORE_IF_ABSENT_SIGNED",
                    "purpose_id": purpose,
                    "role": role,
                    "condition": "IF_ABSENT",
                    "previous_generation": 0,
                    "new_generation": 1,
                    "stored_content_sha256": (
                        "0" * 64
                        if self.store_output_drift
                        else hashlib.sha256(content).hexdigest()
                    ),
                    "record_name_sha256": hashlib.sha256(
                        (
                            "registry-"
                            + hashlib.sha256(
                                value["object_key"].encode("ascii")
                            ).hexdigest()
                            + ".generation-0000000001"
                        ).encode("ascii")
                    ).hexdigest(),
                    "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
                    "signature_hex": signature.hex(),
                    "file_fsync": True,
                    "directory_fsync": True,
                    "hardlink_no_replace": True,
                }
                return self._result(cmd, stdout=_json_line(output))
            if operation == "time-genesis-and-sign":
                self.events.append("time-and-sign")
                value = json.loads(stdin_payload.decode("ascii"))
                preimage = base64.b64decode(value["preimage_b64"], validate=True)
                signature = self.keys[purpose].sign(preimage)
                record = json.dumps(
                    {
                        "logical_sequence": 0,
                        "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
                        "previous_logical_sequence_or_null": None,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii") + b"\n"
                output = {
                    "operation": "TIME_GENESIS_SIGNED",
                    "purpose_id": purpose,
                    "role": role,
                    "previous_logical_sequence_or_null": None,
                    "logical_sequence": 0,
                    "record_sha256": hashlib.sha256(record).hexdigest(),
                    "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
                    "signature_hex": signature.hex(),
                    "file_fsync": True,
                    "directory_fsync": True,
                    "hardlink_no_replace": True,
                }
                return self._result(cmd, stdout=_json_line(output))
            raise AssertionError(operation)
        raise AssertionError(args)


def _install_fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeDocker) -> None:
    runtime_root = tmp_path / "runtime"
    config = runtime_root / "docker-config"
    config.mkdir(parents=True)
    control = LocalDockerControlPlaneV1(
        executable=Path("/usr/bin/docker"),
        socket_path=Path("/run/docker.sock"),
        config_directory=config,
        environment=MappingProxyType(
            {
                "DOCKER_CONFIG": config.as_posix(),
                "DOCKER_HOST": "unix:///var/run/docker.sock",
                "HOME": (runtime_root / "docker-home").as_posix(),
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            }
        ),
        binding=MappingProxyType({"schema": "fake-local-control", "socket": "/run/docker.sock"}),
    )

    @contextmanager
    def prepared(_: Path):
        try:
            yield runtime_root, control
        finally:
            (runtime_root / "phase2b-c1-actor-seccomp.json").unlink(
                missing_ok=True
            )

    monkeypatch.setattr(ceremony, "_prepared_runtime_v1", prepared)
    monkeypatch.setattr(ceremony, "_run_docker_command_v1", fake)
    monkeypatch.setattr(ceremony.secrets, "token_hex", lambda length: "12" * length)
    nonce_counter = iter(bytes((index,)) * 32 for index in range(1, 7))
    monkeypatch.setattr(ceremony.secrets, "token_bytes", lambda length: next(nonce_counter))


def test_success_executes_exact_six_live_bootstrap_and_keeps_only_opaque_volumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, tmp_path, fake)
    result = ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())

    assert result.wire_verification.disposition is (
        Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
    )
    assert result.registry.execution_registry_id == result.store_receipt.body.execution_registry_id
    assert result.store_receipt.store_receipt_id == result.logical_genesis_token.body.store_receipt_id
    assert result.logical_genesis_token.body.logical_sequence == 0
    assert result.logical_genesis_token.body.previous_logical_sequence_or_null is None
    assert result.actual_960_case_run_verified is False
    assert result.cas_persistence_executed is False
    assert result.crash_recovery_verified is False
    assert result.recognizer_executed is False
    assert result.scoring_performed is False
    assert result.formal_gate_evaluation_performed is False
    assert result.effect_evidence is False
    assert result.c1_exit_evidence is False
    assert fake.containers == {}
    assert len(fake.volumes) == 11
    assert fake.volumes == {
        name
        for binding in result.role_runtime_bindings
        for _, name in binding.purpose_private_volumes
    }
    assert all(re.fullmatch(r"hp2bc1v-[0-9a-f]{48}", name) for name in fake.volumes)
    assert tuple(item.operation for item in result.operation_bindings) == (
        *("QUALIFY" for _ in range(6)),
        *("KEYGEN" for _ in range(6)),
        "GOVERNANCE_SIGN",
        "STORE_IF_ABSENT_AND_SIGN",
        "TIME_GENESIS_AND_SIGN",
    )
    first_keygen = fake.events.index("keygen-1")
    assert all(fake.events.index(f"qualify-{purpose}") < first_keygen for purpose in range(1, 7))
    assert fake.events.count("initializer-create") == 11
    assert fake.events.count("volume-init") == 11
    assert fake.events.count("container-rm") == 17
    assert not any(event == "volume-rm" for event in fake.events)


@pytest.mark.parametrize("failure_kind", ["qualification", "namespace"])
def test_qualification_failures_abort_before_keygen_and_remove_all_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_kind: str,
) -> None:
    fake = FakeDocker(
        qualification_failure_purpose=4 if failure_kind == "qualification" else None,
        namespace_collision=failure_kind == "namespace",
    )
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_QUALIFICATION
    assert not fake.keys
    assert fake.containers == {}
    assert fake.volumes == set()
    assert not any(event.startswith("keygen-") for event in fake.events)


def test_image_binding_drift_stops_before_any_volume_or_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker(image_label_drift=True)
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_IMAGE_BINDING
    assert fake.volumes == set() and fake.containers == {} and not fake.keys


def test_arbitrary_nonzero_is_not_accepted_as_authoritative_volume_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker(absence_output_drift=True)
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_VOLUME_PROVISIONING
    assert fake.volumes == set() and fake.containers == {} and not fake.keys


def test_post_key_store_rejection_destroys_failed_attempt_private_volumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker(store_output_drift=True)
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_STORE
    assert len(fake.keys) == 6
    assert fake.containers == {}
    assert fake.volumes == set()
    assert "store-and-sign" in fake.events
    assert "time-and-sign" not in fake.events


@pytest.mark.parametrize("base_exception", [False, True])
def test_actor_create_side_effect_exception_reconciles_and_removes_full_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, base_exception: bool,
) -> None:
    fake = FakeDocker(
        actor_create_fault_purpose=3,
        actor_create_base_exception=base_exception,
    )
    _install_fake(monkeypatch, tmp_path, fake)
    expected = InjectedBaseException if base_exception else ceremony.Phase2BC1LiveDockerCeremonyErrorV1
    with pytest.raises(expected):
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert fake.actor_create_fault_raised is True
    assert fake.containers == {} and fake.volumes == set()
    remove_targets = [
        call[0][-1]
        for call in fake.calls
        if call[0][2:5] == ("container", "rm", "--force")
    ]
    assert remove_targets
    assert all(re.fullmatch(r"[0-9a-f]{64}", item) for item in remove_targets)


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("volume_create_fault", ceremony.Phase2BC1LiveDockerCeremonyErrorV1),
        ("volume_post_create_inspect_fault", InjectedBaseException),
    ],
)
def test_volume_create_ambiguity_is_reconciled_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str,
    expected: type[BaseException],
) -> None:
    fake = FakeDocker(**{option: True})
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(expected):
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert fake.containers == {} and fake.volumes == set()
    assert fake.volume_create_fault_raised or fake.volume_post_inspect_fault_raised


@pytest.mark.parametrize("option", ["initializer_create_fault", "initializer_start_fault"])
def test_named_initializer_timeout_reconciles_and_cleans_every_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, option: str,
) -> None:
    fake = FakeDocker(**{option: True})
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_DOCKER_TIMEOUT
    assert fake.containers == {} and fake.volumes == set()
    assert any(event == "initializer-create" for event in fake.events)
    assert not any(call[0][2] == "run" for call in fake.calls)


@pytest.mark.parametrize("fault_kind", ["ordinary", "base"])
def test_cleanup_catches_all_exception_classes_and_continues_all_volume_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault_kind: str,
) -> None:
    fake = FakeDocker(
        store_output_drift=True,
        cleanup_fault_kind=fault_kind,
    )
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_CLEANUP
    assert fake.cleanup_fault_raised is True
    assert fake.events.count("volume-rm") == 11
    assert fake.volumes == set()
    inspected_after_fault = [
        call for call in fake.calls
        if call[0][2:4] == ("volume", "inspect")
    ]
    assert len(inspected_after_fault) >= 22


@pytest.mark.parametrize(
    "fault_kind", ["container-ordinary", "container-base"]
)
def test_cleanup_catches_all_exception_classes_and_continues_container_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault_kind: str,
) -> None:
    fake = FakeDocker(
        store_output_drift=True,
        cleanup_fault_kind=fault_kind,
    )
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_CLEANUP
    assert fake.cleanup_fault_raised is True
    # All six actors are deleted despite the first rm call raising after its
    # daemon-side effect; failed attempts also delete every private volume.
    assert fake.events.count("container-rm") == 17
    assert fake.containers == {} and fake.volumes == set()


def test_foreign_actor_collision_is_preserved_and_never_name_deleted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker(foreign_actor_purpose=3)
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_CONTAINER_LIFECYCLE
    foreign_name = "hegel-p2b-c1-" + "12" * 12 + "-p3"
    assert not any(
        call[0][2:5] == ("container", "rm", "--force")
        and call[0][-1] == foreign_name
        for call in fake.calls
    )
    assert fake.containers == {} and fake.volumes == set()


def test_foreign_volume_collision_is_preserved_and_never_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker(foreign_volume=True)
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_VOLUME_PROVISIONING
    assert fake.foreign_volume_name is not None
    assert not any(
        call[0][2:4] == ("volume", "rm")
        and call[0][-1] == fake.foreign_volume_name
        for call in fake.calls
    )


def test_receipt_issuance_base_exception_cleans_all_retained_volumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, tmp_path, fake)

    def fail_receipt(**_: object) -> object:
        raise InjectedBaseException("receipt issuance fault")

    monkeypatch.setattr(ceremony, "_issue_receipt_v1", fail_receipt)
    with pytest.raises(InjectedBaseException):
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert fake.containers == {} and fake.volumes == set()
    assert fake.events.count("volume-rm") == 11


@pytest.mark.parametrize(
    "security_drift", ["duplicate", "path", "noncanonical", "wrong-object"]
)
def test_securityopt_requires_exact_nnp_and_one_canonical_frozen_seccomp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, security_drift: str,
) -> None:
    fake = FakeDocker(security_drift=security_drift)
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_CLEANUP
    assert not any(event.startswith("keygen-") for event in fake.events)


@pytest.mark.parametrize(
    "inspect_policy_drift", ["extra-mount", "readonly-mount", "runtime", "args"]
)
def test_actor_inspect_rejects_full_policy_or_exact_mount_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inspect_policy_drift: str,
) -> None:
    fake = FakeDocker(inspect_policy_drift=inspect_policy_drift)
    _install_fake(monkeypatch, tmp_path, fake)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_CLEANUP
    assert not any(event.startswith("keygen-") for event in fake.events)


def test_retained_attempt_ledger_blocks_repeated_bootstrap_without_deleting_custody(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, tmp_path, fake)
    first = ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    retained = set(fake.volumes)
    ledger_name = dict(first.role_runtime_bindings[2].purpose_private_volumes)[
        "/ledger"
    ]
    assert ledger_name in retained and len(retained) == 11

    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(_request())
    assert captured.value.code == ceremony.FAIL_VOLUME_PROVISIONING
    assert fake.volumes == retained
    assert fake.containers == {}


def test_attempt_preflight_drift_rejects_without_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(_request(), attempt_id="phase2b_actual_unsealed_960_attempt_v2_" + "0" * 64)
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(ceremony, "_prepared_runtime_v1", forbidden)
    monkeypatch.setattr(ceremony, "_run_docker_command_v1", forbidden)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(request)
    assert captured.value.code == ceremony.FAIL_PREFLIGHT
    assert called is False


def test_execution_freeze_material_drift_rejects_before_docker_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(
        _request(),
        execution_freeze_manifest_bytes=(
            EXECUTION_FREEZE_BYTES[:-1] + b"x"
        ),
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(ceremony, "_prepared_runtime_v1", forbidden)
    monkeypatch.setattr(ceremony, "_run_docker_command_v1", forbidden)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(request)
    assert captured.value.code == ceremony.FAIL_PREFLIGHT
    assert called is False


@pytest.mark.parametrize(
    ("field_name", "drift"),
    [
        ("recognizer_image_digest", "sha256:" + "b" * 64),
        ("recognizer_source_root_sha256", "b" * 64),
        ("scorer_image_digest", "sha256:" + "b" * 64),
        ("scorer_source_root_sha256", "b" * 64),
    ],
)
def test_compiled_final_image_and_source_pins_reject_before_docker(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    drift: str,
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(ceremony, "_prepared_runtime_v1", forbidden)
    monkeypatch.setattr(ceremony, "_run_docker_command_v1", forbidden)
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1(
            replace(_request(), **{field_name: drift})
        )
    assert captured.value.code == ceremony.FAIL_PREFLIGHT
    assert called is False


def test_public_surface_is_frozen_private_and_has_no_external_qualification_input() -> None:
    assert ceremony.RECOGNIZER_RUNTIME_IMAGE_DIGEST_V1 == RECOGNIZER_IMAGE
    assert ceremony.RECOGNIZER_SOURCE_ROOT_SHA256_V1 == RECOGNIZER_ROOT
    assert ceremony.SCORER_RUNTIME_IMAGE_DIGEST_V1 == SCORER_IMAGE
    assert ceremony.SCORER_SOURCE_ROOT_SHA256_V1 == SCORER_ROOT
    assert tuple(item.name for item in fields(ceremony.Phase2BC1LiveDockerCeremonyRequestV1)) == (
        "campaign_id", "protocol_id", "exact_freeze_id",
        "execution_freeze_manifest_id", "execution_freeze_manifest_sha256",
        "execution_freeze_manifest_bytes",
        "attempt_nonce_sha256", "attempt_id", "recognizer_image_digest",
        "recognizer_source_root_sha256", "scorer_image_digest",
        "scorer_source_root_sha256",
    )
    parameters = inspect.signature(
        ceremony.execute_phase2b_c1_live_docker_ceremony_v1
    ).parameters
    assert tuple(parameters) == ("request", "runtime_parent")
    assert not any("qualification" in name for name in parameters)
    assert tuple(
        item.name
        for item in fields(ceremony.Phase2BC1LiveDockerCeremonyReceiptV1)
    ) == (
        "ceremony_receipt_id", "version", "claim_level", "reason",
        "authority_class", "request", "context", "registry", "store_receipt",
        "logical_genesis_token", "wire_verification", "role_runtime_bindings",
        "operation_bindings", "docker_control_plane_binding_sha256",
        "recognizer_image_inspect_sha256", "scorer_image_inspect_sha256",
        "openssl_sha256", "openssl_version", "stored_registry_sha256",
        "logical_genesis_record_sha256",
        *ceremony._TRUE_RESULT_CLAIMS,
        *ceremony._FALSE_RESULT_CLAIMS,
    )
    assert ceremony.__all__ == (
        "FAIL_CLEANUP", "FAIL_CONTAINER_LIFECYCLE", "FAIL_DOCKER_CAPTURE",
        "FAIL_DOCKER_CONTROL_PLANE", "FAIL_DOCKER_TIMEOUT",
        "FAIL_IMAGE_BINDING", "FAIL_KEYGEN", "FAIL_PREFLIGHT",
        "FAIL_QUALIFICATION", "FAIL_SECCOMP", "FAIL_SIGNATURE", "FAIL_STORE",
        "FAIL_TIME", "FAIL_VOLUME_PROVISIONING", "FAIL_WIRE",
        "OPENSSL_EXECUTABLE_V1", "OPENSSL_SHA256_V1", "OPENSSL_VERSION_V1",
        "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_CLAIM_LEVEL",
        "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_COMPLETE_REASON",
        "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_VERSION",
        "RECOGNIZER_RUNTIME_IMAGE_DIGEST_V1",
        "RECOGNIZER_SOURCE_ROOT_SHA256_V1",
        "SCORER_RUNTIME_IMAGE_DIGEST_V1", "SCORER_SOURCE_ROOT_SHA256_V1",
        "Phase2BC1LiveDockerCeremonyErrorV1",
        "Phase2BC1LiveDockerCeremonyReceiptV1",
        "Phase2BC1LiveDockerCeremonyRequestV1",
        "Phase2BC1LiveOperationBindingV1", "Phase2BC1LiveRoleRuntimeBindingV1",
        "execute_phase2b_c1_live_docker_ceremony_v1",
    )
    with pytest.raises(TypeError):
        ceremony.Phase2BC1LiveDockerCeremonyReceiptV1()
    request = _request()
    with pytest.raises(FrozenInstanceError):
        request.campaign_id = CAMPAIGN_ID  # type: ignore[misc]


def test_worker_freezes_openssl_and_durable_no_replace_without_python_crypto() -> None:
    source = Path(ceremony.__file__).read_text(encoding="utf-8")
    worker = ceremony._ACTOR_WORKER_CODE_V1
    compile(ceremony._VOLUME_INITIALIZER_CODE_V1, "<volume-initializer>", "exec")
    compile(ceremony._ACTOR_SUPERVISOR_CODE_V1, "<actor-supervisor>", "exec")
    compile(worker, "<actor-worker>", "exec")
    assert "cryptography" not in source
    assert ceremony.OPENSSL_EXECUTABLE_V1 == "/usr/bin/openssl"
    assert ceremony.OPENSSL_SHA256_V1 in worker
    assert ceremony.OPENSSL_VERSION_V1 in worker
    assert "302e020100300506032b657004220420" in worker
    assert "302a300506032b6570032100" in worker
    assert "os.O_EXCL" in worker
    assert "os.O_NOFOLLOW" in worker
    assert "os.link(" in worker
    assert "os.fsync(" in worker
    assert "fcntl.flock" in worker
    assert worker.index("record_sha = durable_publish") < worker.index(
        'signature = openssl(["pkeyutl","-sign"',
        worker.index("def store_and_sign"),
    )
    assert worker.index('os.link(pending, target', worker.index("def time_and_sign")) < worker.index(
        'signature = openssl(["pkeyutl","-sign"',
        worker.index("def time_and_sign"),
    )
    assert 'cas_persistence_executed",' in source
    assert 'crash_recovery_verified",' in source
    assert "/cas" in worker
    assert "durable_publish(ledger" in worker
    assert "durable_publish(\"/cas\"" not in worker


def test_command_surface_pins_local_control_plane_and_nonroot_actor_policy() -> None:
    request = _request()
    fake = FakeDocker()

    def binding(digest: str) -> ceremony._ImageBindingV1:
        labels = fake._image(digest)["Config"]["Labels"]
        return ceremony._ImageBindingV1(
            digest=digest,
            inspect_sha256="0" * 64,
            labels=tuple(sorted(labels.items())),
        )

    recognizer = binding(RECOGNIZER_IMAGE)
    scorer = binding(SCORER_IMAGE)
    roles = ceremony._make_role_states_v1(
        request, "ab" * 12,
        recognizer_image=recognizer, scorer_image=scorer,
    )
    other_roles = ceremony._make_role_states_v1(
        request, "cd" * 12,
        recognizer_image=recognizer, scorer_image=scorer,
    )
    assert dict(roles[2].volumes)["/ledger"] == dict(other_roles[2].volumes)[
        "/ledger"
    ]
    assert {
        name for row in roles for path, name in row.volumes if path != "/ledger"
    }.isdisjoint(
        {
            name
            for row in other_roles
            for path, name in row.volumes
            if path != "/ledger"
        }
    )
    control = LocalDockerControlPlaneV1(
        executable=Path("/usr/bin/docker"), socket_path=Path("/run/docker.sock"),
        config_directory=Path("/tmp/private/docker-config"),
        environment=MappingProxyType({}), binding=MappingProxyType({}),
    )
    command = ceremony._container_create_command_v1(
        control, ceremony._load_seccomp_v1(),
        ceremony._actor_target_v1(roles[2], "ab" * 12),
    )
    assert command[:2] == ["/usr/bin/docker", "--host=unix:///var/run/docker.sock"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--user=65534:65534" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--pull=never" in command
    mounts = [item for item in command if item.startswith("--mount=")]
    assert not any("/home/" in item or "/mnt/c" in item for item in mounts)
    assert len(mounts) == 3
    assert all(item.startswith("--mount=type=volume") for item in mounts)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_seccomp_source_stable_open_rejects_link_indirection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, link_kind: str,
) -> None:
    payload = ceremony.SECCOMP_PROFILE_PATH_V1.read_bytes()
    source = tmp_path / "seccomp-source.json"
    source.write_bytes(payload)
    candidate = tmp_path / "seccomp-candidate.json"
    if link_kind == "symlink":
        candidate.symlink_to(source)
    else:
        os.link(source, candidate)
    monkeypatch.setattr(ceremony, "SECCOMP_PROFILE_PATH_V1", candidate)
    monkeypatch.setattr(
        ceremony, "SECCOMP_PROFILE_SHA256_V1", hashlib.sha256(payload).hexdigest()
    )
    with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony._load_seccomp_v1()
    assert captured.value.code == ceremony.FAIL_SECCOMP


def test_seccomp_is_canonicalized_into_private_stable_fsynced_snapshot(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    source = ceremony._load_seccomp_v1()
    staged = ceremony._stage_seccomp_v1(source, runtime_root)
    metadata = staged.path.stat()
    assert staged.path.parent == runtime_root
    assert staged.path != ceremony.SECCOMP_PROFILE_PATH_V1
    assert staged.path.read_bytes() == source.canonical_bytes
    assert staged.path.read_bytes() == ceremony._canonical_json_bytes(
        json.loads(staged.path.read_bytes())
    )
    assert metadata.st_nlink == 1
    assert metadata.st_mode & 0o777 == 0o400


def test_one_reentrant_ceremony_deadline_reserves_cleanup_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(ceremony.time, "monotonic", lambda: clock[0])
    with ceremony._ceremony_deadline_scope_v1():
        outer = ceremony._CEREMONY_DEADLINES_V1.get()
        assert outer == (700.0, 1000.0)
        with ceremony._ceremony_deadline_scope_v1():
            assert ceremony._CEREMONY_DEADLINES_V1.get() is outer
            clock[0] = 699.5
            assert ceremony._bounded_by_ceremony_deadline_v1(60) == 0.5
            clock[0] = 700.0
            with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as normal:
                ceremony._bounded_by_ceremony_deadline_v1(1)
            assert normal.value.code == ceremony.FAIL_DOCKER_TIMEOUT
            with ceremony._ceremony_cleanup_phase_v1():
                clock[0] = 999.5
                assert ceremony._bounded_by_ceremony_deadline_v1(60) == 0.5
                clock[0] = 1000.0
                with pytest.raises(ceremony.Phase2BC1LiveDockerCeremonyErrorV1) as cleanup:
                    ceremony._bounded_by_ceremony_deadline_v1(1)
                assert cleanup.value.code == ceremony.FAIL_DOCKER_TIMEOUT
    assert ceremony._CEREMONY_DEADLINES_V1.get() is None

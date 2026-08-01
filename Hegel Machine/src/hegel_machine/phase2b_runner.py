"""Host-side isolation contract for an untrusted Phase-2B recognizer.

The functions here build and audit an OCI launch specification; they do not
pretend that constructing an argv tuple proves the container was actually run
with those controls.  A formal run additionally needs an external runtime
attestation and the one-shot custodian ledger in :mod:`phase2b_protocol`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Final

from .hashing import stable_hash
from .phase2b_protocol import ExecutionFreezeManifest, frozen_phase2b_protocol


RECOGNIZER_INPUT_MOUNT: Final = "/phase2b/input"
RECOGNIZER_OUTPUT_MOUNT: Final = "/phase2b/output"
RECOGNIZER_TMP_MOUNT: Final = "/tmp"
FROZEN_RECOGNIZER_ENTRYPOINT: Final = (
    "/opt/hegel/bin/phase2b-recognizer",
    "--input",
    RECOGNIZER_INPUT_MOUNT,
    "--output",
    RECOGNIZER_OUTPUT_MOUNT,
)

REQUIRED_RECOGNIZER_MODULES: Final = frozenset(
    {
        "hegel_machine.hashing",
        "hegel_machine.laws",
        "hegel_machine.milestones",
        "hegel_machine.phase2b_adapter",
        "hegel_machine.phase2b_selector",
        "hegel_machine.phase2b_wire",
        "hegel_machine.schema",
    }
)
FORBIDDEN_RECOGNIZER_MODULE_PREFIXES: Final = (
    "hegel_machine.benchmark",
    "hegel_machine.phase2_exit",
    "hegel_machine.phase2b_custodian",
    "hegel_machine.phase2b_evaluator",
    "hegel_machine.vertical_slice",
)


def _require_sha256(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_digest(value: str, name: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must use sha256:<digest>")


def _absolute_clean_path(value: str, name: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise ValueError(f"{name} must be a specific absolute path")
    return str(path)


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    cpus: str = "2.0"
    memory: str = "2g"
    pids: int = 128
    timeout_seconds: int = 3600
    tmpfs_bytes: int = 268_435_456
    maximum_output_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        if not self.cpus or not self.memory:
            raise ValueError("CPU and memory limits are required")
        for name in (
            "pids",
            "timeout_seconds",
            "tmpfs_bytes",
            "maximum_output_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class OciRecognizerRunSpec:
    freeze_manifest_id: str
    protocol_id: str
    image_digest: str
    input_host_directory: str
    output_host_directory: str
    repository_host_directory: str
    input_manifest_sha256: str
    expected_case_count: int
    entrypoint: tuple[str, ...]
    limits: RuntimeLimits = RuntimeLimits()

    def __post_init__(self) -> None:
        if not self.freeze_manifest_id or not self.protocol_id:
            raise ValueError("run spec must bind freeze and protocol manifests")
        if self.protocol_id != frozen_phase2b_protocol().protocol_id:
            raise ValueError("run spec protocol drift")
        _require_digest(self.image_digest, "recognizer image digest")
        _require_sha256(self.input_manifest_sha256, "input manifest SHA-256")
        input_path = _absolute_clean_path(
            self.input_host_directory,
            "input host directory",
        )
        output_path = _absolute_clean_path(
            self.output_host_directory,
            "output host directory",
        )
        repository_path = _absolute_clean_path(
            self.repository_host_directory,
            "repository host directory",
        )
        if len({input_path, output_path, repository_path}) != 3:
            raise ValueError("input, output, and repository paths must be distinct")
        if _is_within(input_path, repository_path) or _is_within(
            output_path,
            repository_path,
        ):
            raise ValueError("formal run directories must be outside the repository")
        if _is_within(input_path, output_path) or _is_within(output_path, input_path):
            raise ValueError("formal input and output directories cannot overlap")
        if self.expected_case_count != 720:
            raise ValueError("formal recognizer run requires exactly 720 cases")
        if not isinstance(self.entrypoint, tuple) or not self.entrypoint:
            raise TypeError("recognizer entrypoint must be a nonempty tuple")
        if any(not isinstance(item, str) or not item for item in self.entrypoint):
            raise TypeError("recognizer entrypoint items must be nonempty strings")
        if any(
            "answer" in item.casefold()
            or "generator" in item.casefold()
            or "custodian" in item.casefold()
            for item in self.entrypoint
        ):
            raise ValueError("recognizer entrypoint references a forbidden component")
        if self.entrypoint != FROZEN_RECOGNIZER_ENTRYPOINT:
            raise ValueError("recognizer entrypoint differs from the frozen contract")

    @property
    def run_spec_id(self) -> str:
        return stable_hash(self, prefix="phase2b_oci_run_spec_")

    def argv(self, runtime: str) -> tuple[str, ...]:
        """Return a deterministic Docker/Podman-compatible launch vector."""

        if runtime not in {"docker", "podman"}:
            raise ValueError("runtime must be docker or podman")
        return (
            runtime,
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            f"--cpus={self.limits.cpus}",
            f"--memory={self.limits.memory}",
            f"--pids-limit={self.limits.pids}",
            "--env=PYTHONHASHSEED=0",
            "--env=HOME=/nonexistent",
            "--env=NO_PROXY=*",
            (
                f"--tmpfs={RECOGNIZER_TMP_MOUNT}:rw,noexec,nosuid,nodev,"
                f"size={self.limits.tmpfs_bytes}"
            ),
            (
                "--mount=type=bind,src="
                f"{self.input_host_directory},dst={RECOGNIZER_INPUT_MOUNT},readonly"
            ),
            (
                "--mount=type=bind,src="
                f"{self.output_host_directory},dst={RECOGNIZER_OUTPUT_MOUNT}"
            ),
            self.image_digest,
            *self.entrypoint,
        )


def _is_within(candidate: str, parent: str) -> bool:
    candidate_parts = PurePosixPath(candidate).parts
    parent_parts = PurePosixPath(parent).parts
    return candidate_parts[: len(parent_parts)] == parent_parts


def build_oci_run_spec(
    *,
    freeze_manifest: ExecutionFreezeManifest,
    input_host_directory: str,
    output_host_directory: str,
    repository_host_directory: str,
    input_manifest_sha256: str,
    limits: RuntimeLimits = RuntimeLimits(),
) -> OciRecognizerRunSpec:
    return OciRecognizerRunSpec(
        freeze_manifest_id=freeze_manifest.manifest_id,
        protocol_id=freeze_manifest.protocol_id,
        image_digest=freeze_manifest.recognizer_image_digest,
        input_host_directory=input_host_directory,
        output_host_directory=output_host_directory,
        repository_host_directory=repository_host_directory,
        input_manifest_sha256=input_manifest_sha256,
        expected_case_count=720,
        entrypoint=FROZEN_RECOGNIZER_ENTRYPOINT,
        limits=limits,
    )


@dataclass(frozen=True, slots=True)
class ImageInventoryAudit:
    installed_modules: tuple[str, ...]
    missing_required_modules: tuple[str, ...]
    forbidden_modules_present: tuple[str, ...]
    unexpected_modules_present: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            not self.missing_required_modules
            and not self.forbidden_modules_present
            and not self.unexpected_modules_present
        )


def audit_recognizer_image_modules(
    installed_modules: tuple[str, ...],
) -> ImageInventoryAudit:
    if not isinstance(installed_modules, tuple):
        raise TypeError("installed module inventory must be an immutable tuple")
    if any(not isinstance(item, str) or not item for item in installed_modules):
        raise TypeError("installed module inventory contains an invalid name")
    if len(set(installed_modules)) != len(installed_modules):
        raise ValueError("installed module inventory repeats a module")
    installed = set(installed_modules)
    forbidden = tuple(
        sorted(
            item
            for item in installed
            if any(
                item == prefix or item.startswith(prefix + ".")
                for prefix in FORBIDDEN_RECOGNIZER_MODULE_PREFIXES
            )
        )
    )
    missing = tuple(sorted(REQUIRED_RECOGNIZER_MODULES - installed))
    unexpected = tuple(sorted(installed - REQUIRED_RECOGNIZER_MODULES))
    return ImageInventoryAudit(
        tuple(sorted(installed)),
        missing,
        forbidden,
        unexpected,
    )


@dataclass(frozen=True, slots=True)
class ExternalRuntimeAttestation:
    run_spec_id: str
    runtime_name: str
    runtime_version: str
    external_attestor_id: str
    detached_attestation_sha256: str
    prediction_archive_sha256: str
    audit_archive_sha256: str
    freeze_manifest_id: str
    protocol_id: str
    input_manifest_sha256: str
    prediction_case_count: int
    exit_code: int
    timed_out: bool
    output_size_bytes: int
    observed_network_disabled: bool
    observed_read_only_root: bool
    observed_repository_absent: bool
    observed_answer_manifest_absent: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.run_spec_id,
                self.runtime_name,
                self.runtime_version,
                self.external_attestor_id,
            )
        ):
            raise ValueError("runtime attestation identity is incomplete")
        for name in (
            "detached_attestation_sha256",
            "prediction_archive_sha256",
            "audit_archive_sha256",
            "input_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("runtime exit code must be an integer")
        if (
            isinstance(self.output_size_bytes, bool)
            or not isinstance(self.output_size_bytes, int)
            or self.output_size_bytes < 0
        ):
            raise ValueError("runtime output size must be nonnegative")
        if (
            type(self.prediction_case_count) is not int
            or self.prediction_case_count < 0
        ):
            raise ValueError("prediction case count must be a nonnegative integer")
        for name in (
            "timed_out",
            "observed_network_disabled",
            "observed_read_only_root",
            "observed_repository_absent",
            "observed_answer_manifest_absent",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("runtime attestation controls must be booleans")

    def validate(self, spec: OciRecognizerRunSpec) -> None:
        if self.run_spec_id != spec.run_spec_id:
            raise ValueError("runtime attestation binds a different run spec")
        if self.freeze_manifest_id != spec.freeze_manifest_id:
            raise ValueError("runtime attestation binds a different freeze manifest")
        if self.protocol_id != spec.protocol_id:
            raise ValueError("runtime attestation binds a different protocol")
        if self.input_manifest_sha256 != spec.input_manifest_sha256:
            raise ValueError("runtime attestation binds a different input manifest")
        if self.runtime_name not in {"docker", "podman"}:
            raise ValueError("runtime attestation names an unsupported runtime")
        if self.exit_code != 0 or self.timed_out:
            raise ValueError("untrusted recognizer did not complete successfully")
        if self.output_size_bytes > spec.limits.maximum_output_bytes:
            raise ValueError("untrusted recognizer exceeded its output limit")
        if self.output_size_bytes == 0:
            raise ValueError("untrusted recognizer produced an empty output archive")
        if self.prediction_case_count != spec.expected_case_count:
            raise ValueError("untrusted recognizer did not emit exactly 720 predictions")
        if not all(
            (
                self.observed_network_disabled,
                self.observed_read_only_root,
                self.observed_repository_absent,
                self.observed_answer_manifest_absent,
            )
        ):
            raise ValueError("runtime attestation reports a missing isolation control")


__all__ = (
    "ExternalRuntimeAttestation",
    "FROZEN_RECOGNIZER_ENTRYPOINT",
    "FORBIDDEN_RECOGNIZER_MODULE_PREFIXES",
    "ImageInventoryAudit",
    "OciRecognizerRunSpec",
    "RECOGNIZER_INPUT_MOUNT",
    "RECOGNIZER_OUTPUT_MOUNT",
    "REQUIRED_RECOGNIZER_MODULES",
    "RuntimeLimits",
    "audit_recognizer_image_modules",
    "build_oci_run_spec",
)

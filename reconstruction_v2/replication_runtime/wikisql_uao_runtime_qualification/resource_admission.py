"""Shared-node resource admission for the stable WikiSQL qualification.

This module observes capacity; it never kills, pauses, signals, renices, or
otherwise modifies an external process.  A foreign GPU process is telemetry,
not an automatic rejection.  Admission depends only on the resource thresholds
supplied by :class:`ResourceAdmissionConfig`.

The intended controller shape is::

    with resource_admission_guard(config) as decision:
        if decision.status == ADMITTED:
            run_qualification_while_the_cooperative_flock_is_held()

Expected contention is represented by ``DEFERRED_SHARED_RESOURCE`` and
``EX_TEMPFAIL``.  Broken or untrustworthy telemetry is represented by
``FAILED_INFRASTRUCTURE``; it is never silently reclassified as contention.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import re
import stat
from statistics import median
import subprocess
import time
from typing import Callable, Iterator, Mapping, Protocol, Sequence


ADMITTED = "ADMITTED"
DEFERRED_SHARED_RESOURCE = "DEFERRED_SHARED_RESOURCE"
FAILED_INFRASTRUCTURE = "FAILED_INFRASTRUCTURE"

EX_OK = 0
EX_SOFTWARE = 70
EX_TEMPFAIL = 75

REQUIRED_SAMPLE_COUNT = 3
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]{8,}\Z")
_MEMINFO_VALUE = re.compile(r"(?P<value>[0-9]+) kB\Z")
_NO_COMPUTE_PROCESS_MARKER = "No running processes found"
_RESOURCE_POLICY_SCHEMA = "wikisql_uao_shared_resource_policy_v1"


class ResourceAdmissionInfrastructureError(RuntimeError):
    """A resource observation could not be trusted."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class ResourceBusyError(RuntimeError):
    """The cooperative qualification flock is already held."""

    status = DEFERRED_SHARED_RESOURCE
    exit_code = EX_TEMPFAIL
    reason_code = "QUALIFICATION_FLOCK_OCCUPIED"


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


@dataclass(frozen=True, slots=True)
class GpuAdmissionThreshold:
    """Frozen admission requirements for one physical GPU."""

    expected_uuid: str
    min_free_memory_mib: int
    max_utilization_percent: float
    max_temperature_celsius: float

    def __post_init__(self) -> None:
        if _GPU_UUID.fullmatch(self.expected_uuid) is None:
            raise ValueError("expected_uuid must be an NVIDIA GPU UUID")
        if (
            isinstance(self.min_free_memory_mib, bool)
            or not isinstance(self.min_free_memory_mib, int)
            or self.min_free_memory_mib < 0
        ):
            raise ValueError("min_free_memory_mib must be a nonnegative integer")
        if (
            not _is_finite_number(self.max_utilization_percent)
            or not 0.0 <= float(self.max_utilization_percent) <= 100.0
        ):
            raise ValueError("max_utilization_percent must be in [0, 100]")
        if (
            not _is_finite_number(self.max_temperature_celsius)
            or not 0.0 <= float(self.max_temperature_celsius) <= 125.0
        ):
            raise ValueError("max_temperature_celsius must be in [0, 125]")


def _default_lock_path() -> Path:
    return Path(f"/run/user/{os.getuid()}/wikisql-uao-runtime-qualification.lock")


@dataclass(frozen=True, slots=True)
class ResourceAdmissionConfig:
    """All policy thresholds and sampling controls for one admission."""

    gpu_thresholds: Mapping[int, GpuAdmissionThreshold]
    min_host_mem_available_bytes: int
    max_host_cpu_busy_ratio: float
    min_swap_free_bytes: int = 0
    max_load1_per_cpu: float = 1.0
    nvidia_smi_path: Path = Path("/usr/bin/nvidia-smi")
    qualification_lock_path: Path = field(default_factory=_default_lock_path)
    sample_count: int = REQUIRED_SAMPLE_COUNT
    sample_interval_seconds: float = 2.0
    command_timeout_seconds: float = 30.0

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "ResourceAdmissionConfig":
        """Parse the canonical mapping used by a qualification controller."""

        if not isinstance(value, Mapping):
            raise ValueError("resource policy must be a mapping")
        raw_gpus = value.get("gpu_thresholds")
        if not isinstance(raw_gpus, Mapping) or not raw_gpus:
            raise ValueError("gpu_thresholds must be a nonempty mapping")
        thresholds: dict[int, GpuAdmissionThreshold] = {}
        for raw_index, raw_threshold in raw_gpus.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("GPU threshold index is malformed") from exc
            if isinstance(raw_index, bool) or str(index) != str(raw_index):
                raise ValueError("GPU threshold index is malformed")
            if not isinstance(raw_threshold, Mapping):
                raise ValueError("GPU threshold must be a mapping")
            required = {
                "expected_uuid",
                "min_free_memory_mib",
                "max_utilization_percent",
                "max_temperature_celsius",
            }
            if set(raw_threshold) != required:
                raise ValueError("GPU threshold fields drifted")
            thresholds[index] = GpuAdmissionThreshold(
                expected_uuid=str(raw_threshold["expected_uuid"]),
                min_free_memory_mib=raw_threshold["min_free_memory_mib"],  # type: ignore[arg-type]
                max_utilization_percent=raw_threshold["max_utilization_percent"],  # type: ignore[arg-type]
                max_temperature_celsius=raw_threshold["max_temperature_celsius"],  # type: ignore[arg-type]
            )
        allowed = {
            "gpu_thresholds",
            "min_host_mem_available_bytes",
            "max_host_cpu_busy_ratio",
            "min_swap_free_bytes",
            "max_load1_per_cpu",
            "nvidia_smi_path",
            "qualification_lock_path",
            "sample_count",
            "sample_interval_seconds",
            "command_timeout_seconds",
        }
        if set(value) - allowed:
            raise ValueError("resource policy fields drifted")
        try:
            return cls(
                gpu_thresholds=thresholds,
                min_host_mem_available_bytes=value[
                    "min_host_mem_available_bytes"
                ],  # type: ignore[arg-type]
                max_host_cpu_busy_ratio=value[
                    "max_host_cpu_busy_ratio"
                ],  # type: ignore[arg-type]
                min_swap_free_bytes=value.get(
                    "min_swap_free_bytes", 0
                ),  # type: ignore[arg-type]
                max_load1_per_cpu=value.get(
                    "max_load1_per_cpu", 1.0
                ),  # type: ignore[arg-type]
                nvidia_smi_path=Path(
                    str(value.get("nvidia_smi_path", "/usr/bin/nvidia-smi"))
                ),
                qualification_lock_path=Path(
                    str(value.get("qualification_lock_path", _default_lock_path()))
                ),
                sample_count=value.get(
                    "sample_count", REQUIRED_SAMPLE_COUNT
                ),  # type: ignore[arg-type]
                sample_interval_seconds=value.get(
                    "sample_interval_seconds", 2.0
                ),  # type: ignore[arg-type]
                command_timeout_seconds=value.get(
                    "command_timeout_seconds", 30.0
                ),  # type: ignore[arg-type]
            )
        except KeyError as exc:
            raise ValueError(f"resource policy missing field: {exc.args[0]}") from exc

    def __post_init__(self) -> None:
        keys = set(self.gpu_thresholds)
        if not keys or any(isinstance(key, bool) or not isinstance(key, int) or key < 0 for key in keys):
            raise ValueError("gpu_thresholds must use nonnegative integer indices")
        uuids = [threshold.expected_uuid for threshold in self.gpu_thresholds.values()]
        if len(set(uuids)) != len(uuids):
            raise ValueError("GPU UUIDs must be unique")
        if (
            isinstance(self.min_host_mem_available_bytes, bool)
            or not isinstance(self.min_host_mem_available_bytes, int)
            or self.min_host_mem_available_bytes < 0
        ):
            raise ValueError(
                "min_host_mem_available_bytes must be a nonnegative integer"
            )
        if (
            not _is_finite_number(self.max_host_cpu_busy_ratio)
            or not 0.0 <= float(self.max_host_cpu_busy_ratio) <= 1.0
        ):
            raise ValueError("max_host_cpu_busy_ratio must be in [0, 1]")
        if (
            isinstance(self.min_swap_free_bytes, bool)
            or not isinstance(self.min_swap_free_bytes, int)
            or self.min_swap_free_bytes < 0
        ):
            raise ValueError("min_swap_free_bytes must be a nonnegative integer")
        if (
            not _is_finite_number(self.max_load1_per_cpu)
            or float(self.max_load1_per_cpu) < 0.0
        ):
            raise ValueError("max_load1_per_cpu must be nonnegative")
        if self.sample_count != REQUIRED_SAMPLE_COUNT:
            raise ValueError(
                f"sample_count must remain exactly {REQUIRED_SAMPLE_COUNT}"
            )
        if (
            not _is_finite_number(self.sample_interval_seconds)
            or float(self.sample_interval_seconds) < 0.0
        ):
            raise ValueError("sample_interval_seconds must be nonnegative")
        if (
            not _is_finite_number(self.command_timeout_seconds)
            or float(self.command_timeout_seconds) <= 0.0
        ):
            raise ValueError("command_timeout_seconds must be positive")
        for field_name, path in (
            ("nvidia_smi_path", self.nvidia_smi_path),
            ("qualification_lock_path", self.qualification_lock_path),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field_name} must be an absolute Path")
            if "\x00" in str(path) or "\n" in str(path):
                raise ValueError(f"{field_name} is malformed")

    @property
    def expected_gpu_uuids(self) -> dict[int, str]:
        return {
            index: threshold.expected_uuid
            for index, threshold in sorted(self.gpu_thresholds.items())
        }


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    """Aggregate GPU telemetry with no external PID or command disclosure."""

    index: int
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_percent: float
    temperature_celsius: float
    external_compute_process_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
            "external_compute_process_count": self.external_compute_process_count,
        }


@dataclass(frozen=True, slots=True)
class HostTelemetry:
    mem_total_bytes: int
    mem_available_bytes: int
    swap_free_bytes: int
    cpu_busy_ratio: float | None
    load1: float
    logical_cpu_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mem_total_bytes": self.mem_total_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "cpu_busy_ratio": self.cpu_busy_ratio,
            "load1": self.load1,
            "logical_cpu_count": self.logical_cpu_count,
        }


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    observed_at_utc: str
    monotonic_ns: int
    gpus: tuple[GpuTelemetry, ...]
    host: HostTelemetry

    def gpu(self, index: int) -> GpuTelemetry:
        matches = [gpu for gpu in self.gpus if gpu.index == index]
        if len(matches) != 1:
            raise ResourceAdmissionInfrastructureError(
                "RESOURCE_SNAPSHOT_GPU_REGISTRY_MALFORMED"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_at_utc": self.observed_at_utc,
            "monotonic_ns": self.monotonic_ns,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "host": self.host.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    status: str
    reason_codes: tuple[str, ...]
    samples: tuple[ResourceSnapshot, ...] = ()
    effect_attempt_claimed: bool = False

    def __post_init__(self) -> None:
        if self.status not in {
            ADMITTED,
            DEFERRED_SHARED_RESOURCE,
            FAILED_INFRASTRUCTURE,
        }:
            raise ValueError("unknown admission status")
        if self.effect_attempt_claimed:
            raise ValueError("resource admission must not claim an effect attempt")

    @property
    def exit_code(self) -> int:
        return {
            ADMITTED: EX_OK,
            DEFERRED_SHARED_RESOURCE: EX_TEMPFAIL,
            FAILED_INFRASTRUCTURE: EX_SOFTWARE,
        }[self.status]

    @property
    def admitted(self) -> bool:
        return self.status == ADMITTED

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "exit_code": self.exit_code,
            "effect_attempt_claimed": self.effect_attempt_claimed,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @property
    def receipt(self) -> dict[str, object]:
        """Runner-facing, JSON-serializable admission receipt."""

        return self.to_dict()


@dataclass(frozen=True, slots=True)
class CpuCounters:
    total_ticks: int
    idle_ticks: int


class SnapshotSampler(Protocol):
    def __call__(self) -> ResourceSnapshot:
        """Return one observation or raise an infrastructure exception."""


def parse_nvidia_gpu_csv(
    stdout: str,
    *,
    expected_gpu_uuids: Mapping[int, str],
) -> tuple[GpuTelemetry, ...]:
    """Parse the fixed locale-free seven-column GPU query."""

    try:
        rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
    except (csv.Error, UnicodeError) as exc:
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_MALFORMED_OUTPUT"
        ) from exc
    parsed: dict[int, GpuTelemetry] = {}
    for row in rows:
        if not row or all(not field.strip() for field in row):
            continue
        fields = [field.strip() for field in row]
        if len(fields) != 7:
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_MALFORMED_OUTPUT"
            )
        try:
            index = int(fields[0])
            total = int(fields[2])
            used = int(fields[3])
            free = int(fields[4])
            utilization = float(fields[5])
            temperature = float(fields[6])
        except ValueError as exc:
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_MALFORMED_OUTPUT"
            ) from exc
        uuid = fields[1]
        if index in parsed or index not in expected_gpu_uuids:
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_MALFORMED_OUTPUT"
            )
        if uuid != expected_gpu_uuids[index]:
            raise ResourceAdmissionInfrastructureError("NVIDIA_SMI_UUID_DRIFT")
        if (
            total <= 0
            or used < 0
            or free < 0
            or used > total
            or free > total
            or not 0.0 <= utilization <= 100.0
            or not 0.0 <= temperature <= 125.0
        ):
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_MALFORMED_OUTPUT"
            )
        parsed[index] = GpuTelemetry(
            index=index,
            uuid=uuid,
            memory_total_mib=total,
            memory_used_mib=used,
            memory_free_mib=free,
            utilization_percent=utilization,
            temperature_celsius=temperature,
        )
    if set(parsed) != set(expected_gpu_uuids):
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_MALFORMED_OUTPUT"
        )
    return tuple(parsed[index] for index in sorted(parsed))


def parse_nvidia_compute_apps_csv(
    stdout: str,
    *,
    expected_gpu_uuids: Mapping[int, str],
) -> dict[int, int]:
    """Count foreign compute contexts without retaining PID values."""

    uuid_to_index = {
        uuid: index for index, uuid in expected_gpu_uuids.items()
    }
    counts = {index: 0 for index in expected_gpu_uuids}
    try:
        rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
    except (csv.Error, UnicodeError) as exc:
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_MALFORMED_OUTPUT"
        ) from exc
    for row in rows:
        if not row or all(not field.strip() for field in row):
            continue
        fields = [field.strip() for field in row]
        if any(_NO_COMPUTE_PROCESS_MARKER in field for field in fields):
            continue
        if len(fields) != 2:
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_MALFORMED_OUTPUT"
            )
        uuid, raw_pid = fields
        try:
            pid = int(raw_pid)
        except ValueError as exc:
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_MALFORMED_OUTPUT"
            ) from exc
        if uuid not in uuid_to_index or pid <= 1:
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_MALFORMED_OUTPUT"
            )
        counts[uuid_to_index[uuid]] += 1
    return counts


def _run_nvidia_smi(
    command: Sequence[str],
    *,
    runner: Callable[..., object],
    timeout_seconds: float,
) -> str:
    try:
        completed = runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
        )
    except subprocess.TimeoutExpired as exc:
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_TIMEOUT"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_EXECUTION_FAILED"
        ) from exc
    returncode = getattr(completed, "returncode", None)
    stdout = getattr(completed, "stdout", None)
    stderr = getattr(completed, "stderr", None)
    if (
        isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
    ):
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_RESULT_MALFORMED"
        )
    if returncode != 0:
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_NONZERO_EXIT"
        )
    return stdout


def sample_nvidia_smi(
    executable: Path,
    *,
    expected_gpu_uuids: Mapping[int, str],
    runner: Callable[..., object] = subprocess.run,
    timeout_seconds: float = 30.0,
) -> tuple[GpuTelemetry, ...]:
    """Collect GPU capacity and aggregate process-count telemetry."""

    metric_stdout = _run_nvidia_smi(
        (
            str(executable),
            (
                "--query-gpu=index,uuid,memory.total,memory.used,"
                "memory.free,utilization.gpu,temperature.gpu"
            ),
            "--format=csv,noheader,nounits",
        ),
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    process_stdout = _run_nvidia_smi(
        (
            str(executable),
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    metrics = parse_nvidia_gpu_csv(
        metric_stdout,
        expected_gpu_uuids=expected_gpu_uuids,
    )
    process_counts = parse_nvidia_compute_apps_csv(
        process_stdout,
        expected_gpu_uuids=expected_gpu_uuids,
    )
    return tuple(
        GpuTelemetry(
            index=gpu.index,
            uuid=gpu.uuid,
            memory_total_mib=gpu.memory_total_mib,
            memory_used_mib=gpu.memory_used_mib,
            memory_free_mib=gpu.memory_free_mib,
            utilization_percent=gpu.utilization_percent,
            temperature_celsius=gpu.temperature_celsius,
            external_compute_process_count=process_counts[gpu.index],
        )
        for gpu in metrics
    )


def parse_proc_meminfo(text: str) -> tuple[int, int, int]:
    """Return MemTotal, MemAvailable, and SwapFree in bytes."""

    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, raw = line.partition(":")
        if separator != ":" or key not in {"MemTotal", "MemAvailable", "SwapFree"}:
            continue
        match = _MEMINFO_VALUE.fullmatch(raw.strip())
        if match is None or key in values:
            raise ResourceAdmissionInfrastructureError("PROC_MEMINFO_MALFORMED")
        values[key] = int(match.group("value")) * 1024
    if set(values) != {"MemTotal", "MemAvailable", "SwapFree"}:
        raise ResourceAdmissionInfrastructureError("PROC_MEMINFO_MALFORMED")
    if (
        values["MemTotal"] <= 0
        or not 0 <= values["MemAvailable"] <= values["MemTotal"]
        or values["SwapFree"] < 0
    ):
        raise ResourceAdmissionInfrastructureError("PROC_MEMINFO_MALFORMED")
    return values["MemTotal"], values["MemAvailable"], values["SwapFree"]


def parse_proc_stat_cpu(text: str) -> CpuCounters:
    """Parse aggregate Linux CPU counters without inspecting processes."""

    first = text.splitlines()[0].split() if text.splitlines() else []
    if len(first) < 6 or first[0] != "cpu":
        raise ResourceAdmissionInfrastructureError("PROC_STAT_MALFORMED")
    try:
        counters = [int(value) for value in first[1:]]
    except ValueError as exc:
        raise ResourceAdmissionInfrastructureError("PROC_STAT_MALFORMED") from exc
    if any(value < 0 for value in counters):
        raise ResourceAdmissionInfrastructureError("PROC_STAT_MALFORMED")
    # guest and guest_nice are already included in user and nice.
    total = sum(counters[:8])
    idle = counters[3] + counters[4]
    if total <= 0 or idle > total:
        raise ResourceAdmissionInfrastructureError("PROC_STAT_MALFORMED")
    return CpuCounters(total_ticks=total, idle_ticks=idle)


def cpu_busy_ratio(
    previous: CpuCounters,
    current: CpuCounters,
) -> float | None:
    total_delta = current.total_ticks - previous.total_ticks
    idle_delta = current.idle_ticks - previous.idle_ticks
    if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
        return None
    return (total_delta - idle_delta) / total_delta


def parse_proc_loadavg(text: str) -> float:
    fields = text.split()
    if len(fields) < 3:
        raise ResourceAdmissionInfrastructureError("PROC_LOADAVG_MALFORMED")
    try:
        load1 = float(fields[0])
    except ValueError as exc:
        raise ResourceAdmissionInfrastructureError(
            "PROC_LOADAVG_MALFORMED"
        ) from exc
    if load1 < 0.0:
        raise ResourceAdmissionInfrastructureError("PROC_LOADAVG_MALFORMED")
    return load1


class SystemResourceSampler:
    """Stateful sampler that derives CPU utilization between observations."""

    def __init__(
        self,
        config: ResourceAdmissionConfig,
        *,
        command_runner: Callable[..., object] = subprocess.run,
        proc_root: Path = Path("/proc"),
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        logical_cpu_count: Callable[[], int | None] = os.cpu_count,
    ) -> None:
        self._config = config
        self._command_runner = command_runner
        self._proc_root = proc_root
        self._wall_clock = wall_clock
        self._monotonic_ns = monotonic_ns
        self._logical_cpu_count = logical_cpu_count
        self._previous_cpu: CpuCounters | None = None

    def __call__(self) -> ResourceSnapshot:
        gpus = sample_nvidia_smi(
            self._config.nvidia_smi_path,
            expected_gpu_uuids=self._config.expected_gpu_uuids,
            runner=self._command_runner,
            timeout_seconds=self._config.command_timeout_seconds,
        )
        try:
            meminfo = (self._proc_root / "meminfo").read_text(encoding="ascii")
            stat_text = (self._proc_root / "stat").read_text(encoding="ascii")
            loadavg = (self._proc_root / "loadavg").read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise ResourceAdmissionInfrastructureError(
                "PROC_TELEMETRY_UNAVAILABLE"
            ) from exc
        mem_total, mem_available, swap_free = parse_proc_meminfo(meminfo)
        current_cpu = parse_proc_stat_cpu(stat_text)
        busy = (
            cpu_busy_ratio(self._previous_cpu, current_cpu)
            if self._previous_cpu is not None
            else None
        )
        self._previous_cpu = current_cpu
        cpu_count = self._logical_cpu_count()
        if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count <= 0:
            raise ResourceAdmissionInfrastructureError(
                "LOGICAL_CPU_COUNT_UNAVAILABLE"
            )
        observed = self._wall_clock()
        if observed.tzinfo is None:
            raise ResourceAdmissionInfrastructureError("WALL_CLOCK_NOT_UTC")
        return ResourceSnapshot(
            observed_at_utc=observed.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            monotonic_ns=self._monotonic_ns(),
            gpus=gpus,
            host=HostTelemetry(
                mem_total_bytes=mem_total,
                mem_available_bytes=mem_available,
                swap_free_bytes=swap_free,
                cpu_busy_ratio=busy,
                load1=parse_proc_loadavg(loadavg),
                logical_cpu_count=cpu_count,
            ),
        )


def collect_resource_samples(
    config: ResourceAdmissionConfig,
    *,
    sampler: SnapshotSampler | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[ResourceSnapshot, ...]:
    """Collect exactly three observations, sleeping only between samples."""

    active_sampler = sampler or SystemResourceSampler(config)
    samples: list[ResourceSnapshot] = []
    for index in range(config.sample_count):
        samples.append(active_sampler())
        if index + 1 < config.sample_count:
            sleeper(float(config.sample_interval_seconds))
    return tuple(samples)


def _validate_resource_samples(
    config: ResourceAdmissionConfig,
    samples: Sequence[ResourceSnapshot],
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if len(samples) != config.sample_count:
        reasons.add("RESOURCE_SAMPLE_COUNT_MISMATCH")
        return tuple(sorted(reasons))
    expected_indices = set(config.gpu_thresholds)
    usable_cpu_deltas = 0
    previous_monotonic: int | None = None
    for sample in samples:
        indices = [gpu.index for gpu in sample.gpus]
        if len(indices) != len(set(indices)) or set(indices) != expected_indices:
            reasons.add("RESOURCE_SNAPSHOT_GPU_REGISTRY_MALFORMED")
            continue
        if (
            isinstance(sample.monotonic_ns, bool)
            or not isinstance(sample.monotonic_ns, int)
            or sample.monotonic_ns < 0
            or (
                previous_monotonic is not None
                and sample.monotonic_ns < previous_monotonic
            )
        ):
            reasons.add("RESOURCE_SNAPSHOT_CLOCK_MALFORMED")
        previous_monotonic = sample.monotonic_ns
        for gpu in sample.gpus:
            expected = config.gpu_thresholds[gpu.index]
            if gpu.uuid != expected.expected_uuid:
                reasons.add("NVIDIA_SMI_UUID_DRIFT")
            if (
                gpu.memory_total_mib <= 0
                or gpu.memory_used_mib < 0
                or gpu.memory_free_mib < 0
                or gpu.memory_used_mib > gpu.memory_total_mib
                or gpu.memory_free_mib > gpu.memory_total_mib
                or not 0.0 <= gpu.utilization_percent <= 100.0
                or not 0.0 <= gpu.temperature_celsius <= 125.0
                or isinstance(gpu.external_compute_process_count, bool)
                or not isinstance(gpu.external_compute_process_count, int)
                or gpu.external_compute_process_count < 0
            ):
                reasons.add("NVIDIA_SMI_MALFORMED_OUTPUT")
        host = sample.host
        if (
            host.mem_total_bytes <= 0
            or not 0 <= host.mem_available_bytes <= host.mem_total_bytes
            or host.swap_free_bytes < 0
            or host.load1 < 0.0
            or isinstance(host.logical_cpu_count, bool)
            or not isinstance(host.logical_cpu_count, int)
            or host.logical_cpu_count <= 0
        ):
            reasons.add("PROC_TELEMETRY_MALFORMED")
        if host.cpu_busy_ratio is not None:
            if not 0.0 <= host.cpu_busy_ratio <= 1.0:
                reasons.add("PROC_CPU_DELTA_MALFORMED")
            else:
                usable_cpu_deltas += 1
    if usable_cpu_deltas < config.sample_count - 1:
        reasons.add("PROC_CPU_DELTA_UNAVAILABLE")
    return tuple(sorted(reasons))


def decide_resource_admission(
    config: ResourceAdmissionConfig,
    samples: Sequence[ResourceSnapshot],
) -> AdmissionResult:
    """Classify all observations without fail-fast resource predicates."""

    frozen_samples = tuple(samples)
    infrastructure = _validate_resource_samples(config, frozen_samples)
    if infrastructure:
        return AdmissionResult(
            status=FAILED_INFRASTRUCTURE,
            reason_codes=infrastructure,
            samples=frozen_samples,
        )

    deferred: set[str] = set()
    for gpu_index, threshold in sorted(config.gpu_thresholds.items()):
        gpu_samples = [sample.gpu(gpu_index) for sample in frozen_samples]
        if any(
            gpu.memory_free_mib < threshold.min_free_memory_mib
            for gpu in gpu_samples
        ):
            deferred.add(f"GPU_{gpu_index}_FREE_MEMORY_BELOW_THRESHOLD")
        if (
            median(gpu.utilization_percent for gpu in gpu_samples)
            > threshold.max_utilization_percent
        ):
            deferred.add(f"GPU_{gpu_index}_UTILIZATION_ABOVE_THRESHOLD")
        if any(
            gpu.temperature_celsius > threshold.max_temperature_celsius
            for gpu in gpu_samples
        ):
            deferred.add(f"GPU_{gpu_index}_TEMPERATURE_ABOVE_THRESHOLD")
    for sample in frozen_samples:
        if sample.host.mem_available_bytes < config.min_host_mem_available_bytes:
            deferred.add("HOST_MEM_AVAILABLE_BELOW_THRESHOLD")
        if sample.host.swap_free_bytes < config.min_swap_free_bytes:
            deferred.add("HOST_SWAP_FREE_BELOW_THRESHOLD")
        if (
            sample.host.load1 / sample.host.logical_cpu_count
            > config.max_load1_per_cpu
        ):
            deferred.add("HOST_LOAD1_PER_CPU_ABOVE_THRESHOLD")
    cpu_samples = [
        sample.host.cpu_busy_ratio
        for sample in frozen_samples
        if sample.host.cpu_busy_ratio is not None
    ]
    if median(cpu_samples) > config.max_host_cpu_busy_ratio:
        deferred.add("HOST_CPU_BUSY_ABOVE_THRESHOLD")
    if deferred:
        return AdmissionResult(
            status=DEFERRED_SHARED_RESOURCE,
            reason_codes=tuple(sorted(deferred)),
            samples=frozen_samples,
        )
    return AdmissionResult(
        status=ADMITTED,
        reason_codes=(),
        samples=frozen_samples,
    )


def evaluate_resource_admission(
    config: ResourceAdmissionConfig,
    *,
    sampler: SnapshotSampler | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> AdmissionResult:
    """Collect and classify telemetry, mapping observation errors to infra."""

    samples: tuple[ResourceSnapshot, ...] = ()
    try:
        samples = collect_resource_samples(
            config,
            sampler=sampler,
            sleeper=sleeper,
        )
        return decide_resource_admission(config, samples)
    except ResourceAdmissionInfrastructureError as exc:
        return AdmissionResult(
            status=FAILED_INFRASTRUCTURE,
            reason_codes=(exc.reason_code,),
            samples=samples,
        )
    except subprocess.TimeoutExpired:
        return AdmissionResult(
            status=FAILED_INFRASTRUCTURE,
            reason_codes=("RESOURCE_SAMPLER_TIMEOUT",),
            samples=samples,
        )
    except Exception as exc:
        return AdmissionResult(
            status=FAILED_INFRASTRUCTURE,
            reason_codes=(f"RESOURCE_SAMPLER_EXCEPTION_{type(exc).__name__}",),
            samples=samples,
        )


class QualificationFlock:
    """A cooperative, owner-only flock held for the caller's full scope."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor = -1

    @property
    def acquired(self) -> bool:
        return self._descriptor >= 0

    def acquire_nonblocking(self) -> bool:
        if self.acquired:
            raise ResourceAdmissionInfrastructureError(
                "QUALIFICATION_LOCK_REENTRANT"
            )
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_metadata = self.path.parent.lstat()
        except OSError as exc:
            raise ResourceAdmissionInfrastructureError(
                "QUALIFICATION_LOCK_UNAVAILABLE"
            ) from exc
        if (
            self.path.parent.is_symlink()
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.getuid()
        ):
            raise ResourceAdmissionInfrastructureError(
                "QUALIFICATION_LOCK_PARENT_UNSAFE"
            )
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
            ):
                raise ResourceAdmissionInfrastructureError(
                    "QUALIFICATION_LOCK_FILE_UNSAFE"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                return False
            self._descriptor = descriptor
            return True
        except ResourceAdmissionInfrastructureError:
            if "descriptor" in locals():
                os.close(descriptor)
            raise
        except OSError as exc:
            if "descriptor" in locals():
                os.close(descriptor)
            raise ResourceAdmissionInfrastructureError(
                "QUALIFICATION_LOCK_UNAVAILABLE"
            ) from exc

    def release(self) -> None:
        if not self.acquired:
            return
        descriptor, self._descriptor = self._descriptor, -1
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "QualificationFlock":
        if not self.acquire_nonblocking():
            raise ResourceBusyError("qualification flock is occupied")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


@contextmanager
def resource_admission_guard(
    config: ResourceAdmissionConfig,
    *,
    sampler: SnapshotSampler | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> Iterator[AdmissionResult]:
    """Hold the qualification flock through the caller's admitted work."""

    lock = QualificationFlock(config.qualification_lock_path)
    try:
        try:
            acquired = lock.acquire_nonblocking()
        except ResourceAdmissionInfrastructureError as exc:
            yield AdmissionResult(
                status=FAILED_INFRASTRUCTURE,
                reason_codes=(exc.reason_code,),
            )
            return
        if not acquired:
            yield AdmissionResult(
                status=DEFERRED_SHARED_RESOURCE,
                reason_codes=("QUALIFICATION_FLOCK_OCCUPIED",),
            )
            return
        yield evaluate_resource_admission(
            config,
            sampler=sampler,
            sleeper=sleeper,
        )
    finally:
        lock.release()


@dataclass(frozen=True, slots=True)
class GpuRolePolicy:
    """Capacity requirement for one configured runtime role."""

    minimum_free_mib: int
    role: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_free_mib, bool)
            or not isinstance(self.minimum_free_mib, int)
            or self.minimum_free_mib < 0
        ):
            raise ValueError("minimum_free_mib must be a nonnegative integer")
        if (
            not isinstance(self.role, str)
            or not self.role
            or self.role != self.role.strip()
            or any(character in self.role for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("GPU role is malformed")


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Strict facade for the frozen shared-resource policy in runtime config."""

    schema: str
    gpu_roles: Mapping[int, GpuRolePolicy]
    maximum_gpu_temperature_celsius: float
    maximum_load1_per_cpu: float
    maximum_median_cpu_busy_ratio: float
    maximum_median_gpu_utilization_percent: float
    minimum_host_mem_available_mib: int
    minimum_swap_free_mib: int
    sample_count: int
    sample_interval_seconds: float
    telemetry_timeout_seconds: float

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "ResourcePolicy":
        """Parse exactly the policy emitted by ``prepare.py``."""

        required = {
            "gpu_roles",
            "maximum_gpu_temperature_celsius",
            "maximum_load1_per_cpu",
            "maximum_median_cpu_busy_ratio",
            "maximum_median_gpu_utilization_percent",
            "minimum_host_mem_available_mib",
            "minimum_swap_free_mib",
            "sample_count",
            "sample_interval_seconds",
            "schema",
            "telemetry_timeout_seconds",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("shared resource policy shape drifted")
        if value["schema"] != _RESOURCE_POLICY_SCHEMA:
            raise ValueError("shared resource policy schema drifted")
        raw_roles = value["gpu_roles"]
        if not isinstance(raw_roles, Mapping) or not raw_roles:
            raise ValueError("gpu_roles must be a nonempty mapping")
        roles: dict[int, GpuRolePolicy] = {}
        for raw_index, raw_role in raw_roles.items():
            if (
                not isinstance(raw_index, str)
                or re.fullmatch(r"0|[1-9][0-9]*", raw_index) is None
                or not isinstance(raw_role, Mapping)
                or set(raw_role) != {"minimum_free_mib", "role"}
            ):
                raise ValueError("GPU role policy shape drifted")
            index = int(raw_index)
            if index in roles:
                raise ValueError("GPU role index is duplicated")
            roles[index] = GpuRolePolicy(
                minimum_free_mib=raw_role["minimum_free_mib"],  # type: ignore[arg-type]
                role=raw_role["role"],  # type: ignore[arg-type]
            )
        return cls(
            schema=_RESOURCE_POLICY_SCHEMA,
            gpu_roles=roles,
            maximum_gpu_temperature_celsius=value[
                "maximum_gpu_temperature_celsius"
            ],  # type: ignore[arg-type]
            maximum_load1_per_cpu=value[
                "maximum_load1_per_cpu"
            ],  # type: ignore[arg-type]
            maximum_median_cpu_busy_ratio=value[
                "maximum_median_cpu_busy_ratio"
            ],  # type: ignore[arg-type]
            maximum_median_gpu_utilization_percent=value[
                "maximum_median_gpu_utilization_percent"
            ],  # type: ignore[arg-type]
            minimum_host_mem_available_mib=value[
                "minimum_host_mem_available_mib"
            ],  # type: ignore[arg-type]
            minimum_swap_free_mib=value[
                "minimum_swap_free_mib"
            ],  # type: ignore[arg-type]
            sample_count=value["sample_count"],  # type: ignore[arg-type]
            sample_interval_seconds=value[
                "sample_interval_seconds"
            ],  # type: ignore[arg-type]
            telemetry_timeout_seconds=value[
                "telemetry_timeout_seconds"
            ],  # type: ignore[arg-type]
        )

    def __post_init__(self) -> None:
        if self.schema != _RESOURCE_POLICY_SCHEMA:
            raise ValueError("shared resource policy schema drifted")
        if (
            not isinstance(self.gpu_roles, Mapping)
            or not self.gpu_roles
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or not isinstance(role, GpuRolePolicy)
                for index, role in self.gpu_roles.items()
            )
        ):
            raise ValueError("gpu_roles must use nonnegative integer indices")
        for field_name in (
            "minimum_host_mem_available_mib",
            "minimum_swap_free_mib",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be a nonnegative integer")
        bounded = (
            (
                "maximum_gpu_temperature_celsius",
                self.maximum_gpu_temperature_celsius,
                0.0,
                125.0,
            ),
            (
                "maximum_median_cpu_busy_ratio",
                self.maximum_median_cpu_busy_ratio,
                0.0,
                1.0,
            ),
            (
                "maximum_median_gpu_utilization_percent",
                self.maximum_median_gpu_utilization_percent,
                0.0,
                100.0,
            ),
        )
        for field_name, value, minimum, maximum in bounded:
            if (
                not _is_finite_number(value)
                or not minimum <= float(value) <= maximum
            ):
                raise ValueError(f"{field_name} is outside its valid range")
        if (
            not _is_finite_number(self.maximum_load1_per_cpu)
            or float(self.maximum_load1_per_cpu) < 0.0
        ):
            raise ValueError("maximum_load1_per_cpu must be nonnegative")
        if self.sample_count != REQUIRED_SAMPLE_COUNT:
            raise ValueError(
                f"sample_count must remain exactly {REQUIRED_SAMPLE_COUNT}"
            )
        for field_name in (
            "sample_interval_seconds",
            "telemetry_timeout_seconds",
        ):
            value = getattr(self, field_name)
            if not _is_finite_number(value) or float(value) <= 0.0:
                raise ValueError(f"{field_name} must be positive")

    def to_admission_config(
        self,
        *,
        expected_gpu_uuids: Mapping[object, object],
        nvidia_smi_path: Path,
        qualification_lock_path: Path | None = None,
    ) -> ResourceAdmissionConfig:
        """Bind physical GPU identity and executable path to this policy."""

        normalized_uuids = _normalize_expected_gpu_uuids(expected_gpu_uuids)
        if set(normalized_uuids) != set(self.gpu_roles):
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_UUID_POLICY_DRIFT"
            )
        if (
            not isinstance(nvidia_smi_path, Path)
            or not nvidia_smi_path.is_absolute()
            or "\x00" in str(nvidia_smi_path)
            or "\n" in str(nvidia_smi_path)
        ):
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_PATH_POLICY_DRIFT"
            )
        return ResourceAdmissionConfig(
            gpu_thresholds={
                index: GpuAdmissionThreshold(
                    expected_uuid=normalized_uuids[index],
                    min_free_memory_mib=role.minimum_free_mib,
                    max_utilization_percent=(
                        self.maximum_median_gpu_utilization_percent
                    ),
                    max_temperature_celsius=(
                        self.maximum_gpu_temperature_celsius
                    ),
                )
                for index, role in sorted(self.gpu_roles.items())
            },
            min_host_mem_available_bytes=(
                self.minimum_host_mem_available_mib * 1024**2
            ),
            min_swap_free_bytes=self.minimum_swap_free_mib * 1024**2,
            max_host_cpu_busy_ratio=self.maximum_median_cpu_busy_ratio,
            max_load1_per_cpu=self.maximum_load1_per_cpu,
            nvidia_smi_path=nvidia_smi_path,
            qualification_lock_path=(
                qualification_lock_path or _default_lock_path()
            ),
            sample_count=self.sample_count,
            sample_interval_seconds=self.sample_interval_seconds,
            command_timeout_seconds=self.telemetry_timeout_seconds,
        )


def _normalize_expected_gpu_uuids(
    value: Mapping[object, object],
) -> dict[int, str]:
    if not isinstance(value, Mapping) or not value:
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_UUID_POLICY_DRIFT"
        )
    normalized: dict[int, str] = {}
    for raw_index, raw_uuid in value.items():
        if isinstance(raw_index, bool):
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_UUID_POLICY_DRIFT"
            )
        if isinstance(raw_index, int):
            index = raw_index
        elif (
            isinstance(raw_index, str)
            and re.fullmatch(r"0|[1-9][0-9]*", raw_index) is not None
        ):
            index = int(raw_index)
        else:
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_UUID_POLICY_DRIFT"
            )
        if (
            index < 0
            or index in normalized
            or not isinstance(raw_uuid, str)
            or _GPU_UUID.fullmatch(raw_uuid) is None
        ):
            raise ResourceAdmissionInfrastructureError(
                "NVIDIA_SMI_UUID_POLICY_DRIFT"
            )
        normalized[index] = raw_uuid
    if len(set(normalized.values())) != len(normalized):
        raise ResourceAdmissionInfrastructureError(
            "NVIDIA_SMI_UUID_POLICY_DRIFT"
        )
    return normalized


AdmissionDecision = AdmissionResult


def sample_and_decide(
    policy: ResourcePolicy | Mapping[str, object],
    expected_gpu_uuids: Mapping[object, object],
    nvidia_smi_path: Path,
    *,
    sampler: SnapshotSampler | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    command_runner: Callable[..., object] = subprocess.run,
    proc_root: Path = Path("/proc"),
) -> AdmissionDecision:
    """Small facade for a controller that already owns its lifecycle lock."""

    try:
        parsed_policy = (
            policy if isinstance(policy, ResourcePolicy) else ResourcePolicy.parse(policy)
        )
        config = parsed_policy.to_admission_config(
            expected_gpu_uuids=expected_gpu_uuids,
            nvidia_smi_path=nvidia_smi_path,
        )
    except ResourceAdmissionInfrastructureError as exc:
        return AdmissionDecision(
            status=FAILED_INFRASTRUCTURE,
            reason_codes=(exc.reason_code,),
        )
    except (TypeError, ValueError):
        return AdmissionDecision(
            status=FAILED_INFRASTRUCTURE,
            reason_codes=("RESOURCE_POLICY_MALFORMED",),
        )
    active_sampler = sampler or SystemResourceSampler(
        config,
        command_runner=command_runner,
        proc_root=proc_root,
    )
    return evaluate_resource_admission(
        config,
        sampler=active_sampler,
        sleeper=sleeper,
    )


@contextmanager
def qualification_lock(path: Path) -> Iterator[bool]:
    """Acquire the cooperative run lock and hold it through the ``with`` body."""

    lock = QualificationFlock(path)
    acquired = lock.acquire_nonblocking()
    if not acquired:
        raise ResourceBusyError("qualification flock is occupied")
    try:
        yield True
    finally:
        lock.release()


__all__ = [
    "ADMITTED",
    "DEFERRED_SHARED_RESOURCE",
    "FAILED_INFRASTRUCTURE",
    "EX_OK",
    "EX_SOFTWARE",
    "EX_TEMPFAIL",
    "REQUIRED_SAMPLE_COUNT",
    "AdmissionDecision",
    "AdmissionResult",
    "CpuCounters",
    "GpuAdmissionThreshold",
    "GpuRolePolicy",
    "GpuTelemetry",
    "HostTelemetry",
    "QualificationFlock",
    "ResourceAdmissionConfig",
    "ResourceAdmissionInfrastructureError",
    "ResourceBusyError",
    "ResourceSnapshot",
    "ResourcePolicy",
    "SystemResourceSampler",
    "collect_resource_samples",
    "cpu_busy_ratio",
    "decide_resource_admission",
    "evaluate_resource_admission",
    "parse_nvidia_compute_apps_csv",
    "parse_nvidia_gpu_csv",
    "parse_proc_loadavg",
    "parse_proc_meminfo",
    "parse_proc_stat_cpu",
    "qualification_lock",
    "resource_admission_guard",
    "sample_and_decide",
    "sample_nvidia_smi",
]

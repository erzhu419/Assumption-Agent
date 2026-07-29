from __future__ import annotations

from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from replication_runtime.wikisql_uao_runtime_qualification import (
    resource_admission as subject,
)


GPU_UUIDS = {
    0: "GPU-00000000-0000-0000-0000-000000000000",
    1: "GPU-11111111-1111-1111-1111-111111111111",
}


def _config(tmp_path: Path, **changes: object) -> subject.ResourceAdmissionConfig:
    values: dict[str, object] = {
        "gpu_thresholds": {
            0: subject.GpuAdmissionThreshold(
                expected_uuid=GPU_UUIDS[0],
                min_free_memory_mib=6000,
                max_utilization_percent=50,
                max_temperature_celsius=82,
            ),
            1: subject.GpuAdmissionThreshold(
                expected_uuid=GPU_UUIDS[1],
                min_free_memory_mib=2000,
                max_utilization_percent=50,
                max_temperature_celsius=82,
            ),
        },
        "min_host_mem_available_bytes": 16 * 1024**3,
        "max_host_cpu_busy_ratio": 0.70,
        "nvidia_smi_path": Path("/usr/bin/nvidia-smi"),
        "qualification_lock_path": tmp_path / "qualification.lock",
        "sample_interval_seconds": 0,
    }
    values.update(changes)
    return subject.ResourceAdmissionConfig(**values)


def _snapshot(
    number: int,
    *,
    gpu0_free: int = 7000,
    gpu1_free: int = 7000,
    utilization: float = 10,
    temperature: float = 50,
    external_processes: tuple[int, int] = (2, 3),
    mem_available: int = 32 * 1024**3,
    cpu_busy: float | None = 0.25,
) -> subject.ResourceSnapshot:
    return subject.ResourceSnapshot(
        observed_at_utc=f"2026-07-29T00:00:0{number}Z",
        monotonic_ns=number,
        gpus=(
            subject.GpuTelemetry(
                index=0,
                uuid=GPU_UUIDS[0],
                memory_total_mib=8192,
                memory_used_mib=8192 - gpu0_free,
                memory_free_mib=gpu0_free,
                utilization_percent=utilization,
                temperature_celsius=temperature,
                external_compute_process_count=external_processes[0],
            ),
            subject.GpuTelemetry(
                index=1,
                uuid=GPU_UUIDS[1],
                memory_total_mib=8192,
                memory_used_mib=8192 - gpu1_free,
                memory_free_mib=gpu1_free,
                utilization_percent=utilization,
                temperature_celsius=temperature,
                external_compute_process_count=external_processes[1],
            ),
        ),
        host=subject.HostTelemetry(
            mem_total_bytes=64 * 1024**3,
            mem_available_bytes=mem_available,
            swap_free_bytes=0,
            cpu_busy_ratio=cpu_busy,
            load1=4.0,
            logical_cpu_count=16,
        ),
    )


def _sampler(samples: list[subject.ResourceSnapshot]):
    iterator = iter(samples)
    return lambda: next(iterator)


def test_external_gpu_processes_are_telemetry_not_an_exclusive_gate(
    tmp_path: Path,
) -> None:
    samples = [_snapshot(index) for index in range(3)]
    result = subject.evaluate_resource_admission(
        _config(tmp_path),
        sampler=_sampler(samples),
        sleeper=lambda _seconds: None,
    )

    assert result.status == subject.ADMITTED
    assert result.exit_code == 0
    assert result.effect_attempt_claimed is False
    assert [
        [gpu.external_compute_process_count for gpu in sample.gpus]
        for sample in result.samples
    ] == [[2, 3], [2, 3], [2, 3]]


def test_all_resource_shortages_are_collected_as_retryable_deferred(
    tmp_path: Path,
) -> None:
    samples = [
        _snapshot(
            index,
            gpu0_free=5000,
            gpu1_free=1500,
            utilization=80,
            temperature=90,
            mem_available=8 * 1024**3,
            cpu_busy=0.90,
        )
        for index in range(3)
    ]
    result = subject.evaluate_resource_admission(
        _config(tmp_path),
        sampler=_sampler(samples),
        sleeper=lambda _seconds: None,
    )

    assert result.status == subject.DEFERRED_SHARED_RESOURCE
    assert result.exit_code == subject.EX_TEMPFAIL == 75
    assert set(result.reason_codes) == {
        "GPU_0_FREE_MEMORY_BELOW_THRESHOLD",
        "GPU_0_TEMPERATURE_ABOVE_THRESHOLD",
        "GPU_0_UTILIZATION_ABOVE_THRESHOLD",
        "GPU_1_FREE_MEMORY_BELOW_THRESHOLD",
        "GPU_1_TEMPERATURE_ABOVE_THRESHOLD",
        "GPU_1_UTILIZATION_ABOVE_THRESHOLD",
        "HOST_CPU_BUSY_ABOVE_THRESHOLD",
        "HOST_MEM_AVAILABLE_BELOW_THRESHOLD",
    }
    assert result.effect_attempt_claimed is False


def test_flock_contention_defers_without_sampling_or_external_action(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.qualification_lock_path.touch(mode=0o600)
    with config.qualification_lock_path.open("r+") as occupied:
        fcntl.flock(occupied.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls = 0

        def forbidden_sampler() -> subject.ResourceSnapshot:
            nonlocal calls
            calls += 1
            raise AssertionError("sampler must not run while the flock is occupied")

        with subject.resource_admission_guard(
            config,
            sampler=forbidden_sampler,
            sleeper=lambda _seconds: None,
        ) as result:
            assert result.status == subject.DEFERRED_SHARED_RESOURCE
            assert result.reason_codes == ("QUALIFICATION_FLOCK_OCCUPIED",)
            assert result.exit_code == 75
        assert calls == 0
        fcntl.flock(occupied.fileno(), fcntl.LOCK_UN)


@pytest.mark.parametrize(
    ("runner", "reason"),
    [
        (
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="not,the,seven,required,fields\n",
                stderr="",
            ),
            "NVIDIA_SMI_MALFORMED_OUTPUT",
        ),
        (
            lambda *args, **_kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(args[0], 30)
            ),
            "NVIDIA_SMI_TIMEOUT",
        ),
        (
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=(
                    "0, GPU-drifted0-0000-0000-0000-000000000000, "
                    "8192, 100, 8092, 0, 40\n"
                    f"1, {GPU_UUIDS[1]}, 8192, 100, 8092, 0, 40\n"
                ),
                stderr="",
            ),
            "NVIDIA_SMI_UUID_DRIFT",
        ),
    ],
)
def test_untrustworthy_nvidia_smi_is_infrastructure_failure(
    tmp_path: Path,
    runner,
    reason: str,
) -> None:
    config = _config(tmp_path)
    sampler = subject.SystemResourceSampler(
        config,
        command_runner=runner,
        proc_root=tmp_path,
    )

    result = subject.evaluate_resource_admission(
        config,
        sampler=sampler,
        sleeper=lambda _seconds: None,
    )

    assert result.status == subject.FAILED_INFRASTRUCTURE
    assert result.exit_code == subject.EX_SOFTWARE
    assert result.reason_codes == (reason,)
    assert result.effect_attempt_claimed is False


def test_three_samples_and_two_between_sample_waits_are_fixed(
    tmp_path: Path,
) -> None:
    calls = 0
    waits: list[float] = []

    def sampler() -> subject.ResourceSnapshot:
        nonlocal calls
        value = _snapshot(calls)
        calls += 1
        return value

    config = _config(tmp_path, sample_interval_seconds=3.5)
    result = subject.evaluate_resource_admission(
        config,
        sampler=sampler,
        sleeper=waits.append,
    )

    assert result.status == subject.ADMITTED
    assert calls == subject.REQUIRED_SAMPLE_COUNT == 3
    assert waits == [3.5, 3.5]
    with pytest.raises(ValueError, match="exactly 3"):
        _config(tmp_path, sample_count=2)


def test_proc_parsers_and_cpu_delta_use_only_aggregate_host_telemetry() -> None:
    total, available, swap = subject.parse_proc_meminfo(
        "MemTotal:       65536 kB\n"
        "MemAvailable:   49152 kB\n"
        "SwapFree:           0 kB\n"
        "Unrelated:          1 kB\n"
    )
    previous = subject.parse_proc_stat_cpu("cpu  100 0 50 800 50 0 0 0 0 0\n")
    current = subject.parse_proc_stat_cpu("cpu  150 0 100 850 50 0 0 0 0 0\n")

    assert (total, available, swap) == (
        65536 * 1024,
        49152 * 1024,
        0,
    )
    assert subject.cpu_busy_ratio(previous, current) == pytest.approx(2 / 3)
    assert subject.parse_proc_loadavg("1.25 0.75 0.50 1/100 42\n") == 1.25


def test_real_sampler_merges_process_counts_but_never_exposes_pids(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    outputs = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=(
                    f"0, {GPU_UUIDS[0]}, 8192, 1000, 7192, 20, 55\n"
                    f"1, {GPU_UUIDS[1]}, 8192, 500, 7692, 10, 50\n"
                ),
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=(
                    f"{GPU_UUIDS[0]}, 1234\n"
                    f"{GPU_UUIDS[0]}, 2345\n"
                    f"{GPU_UUIDS[1]}, 3456\n"
                ),
                stderr="",
            ),
        ]
    )

    def runner(*_args, **_kwargs):
        return next(outputs)

    gpus = subject.sample_nvidia_smi(
        config.nvidia_smi_path,
        expected_gpu_uuids=config.expected_gpu_uuids,
        runner=runner,
    )

    assert [gpu.external_compute_process_count for gpu in gpus] == [2, 1]
    serialized = repr([gpu.to_dict() for gpu in gpus])
    assert all(pid not in serialized for pid in ("1234", "2345", "3456"))


def test_guard_holds_flock_for_the_entire_admitted_scope(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    samples = [_snapshot(index) for index in range(3)]

    with subject.resource_admission_guard(
        config,
        sampler=_sampler(samples),
        sleeper=lambda _seconds: None,
    ) as result:
        assert result.status == subject.ADMITTED
        contender = subject.QualificationFlock(config.qualification_lock_path)
        assert contender.acquire_nonblocking() is False
        contender.release()

    contender = subject.QualificationFlock(config.qualification_lock_path)
    assert contender.acquire_nonblocking() is True
    contender.release()


def test_system_sampler_reads_proc_and_derives_cpu_between_three_samples(
    tmp_path: Path,
) -> None:
    (tmp_path / "meminfo").write_text(
        "MemTotal: 67108864 kB\n"
        "MemAvailable: 50331648 kB\n"
        "SwapFree: 0 kB\n",
        encoding="ascii",
    )
    (tmp_path / "loadavg").write_text(
        "1.00 0.50 0.25 1/10 1\n",
        encoding="ascii",
    )
    stat_rows = iter(
        [
            "cpu 100 0 100 800 0 0 0 0 0 0\n",
            "cpu 150 0 150 900 0 0 0 0 0 0\n",
            "cpu 200 0 200 1000 0 0 0 0 0 0\n",
        ]
    )

    def nvidia_runner(command, **_kwargs):
        if command[1].startswith("--query-gpu="):
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"0, {GPU_UUIDS[0]}, 8192, 1000, 7192, 20, 55\n"
                    f"1, {GPU_UUIDS[1]}, 8192, 500, 7692, 10, 50\n"
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    config = _config(tmp_path)
    sampler = subject.SystemResourceSampler(
        config,
        command_runner=nvidia_runner,
        proc_root=tmp_path,
        wall_clock=lambda: datetime(2026, 7, 29, tzinfo=timezone.utc),
        monotonic_ns=iter((1, 2, 3)).__next__,
        logical_cpu_count=lambda: 16,
    )
    original_read_text = Path.read_text

    def changing_read_text(path: Path, *args, **kwargs):
        if path == tmp_path / "stat":
            return next(stat_rows)
        return original_read_text(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Path, "read_text", changing_read_text)
        result = subject.evaluate_resource_admission(
            config,
            sampler=sampler,
            sleeper=lambda _seconds: None,
        )

    assert result.status == subject.ADMITTED
    assert [sample.host.cpu_busy_ratio for sample in result.samples] == [
        None,
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]


def test_runner_facade_parses_policy_and_returns_receipt(
    tmp_path: Path,
) -> None:
    mapping = {
        "gpu_roles": {
            "0": {"minimum_free_mib": 6144, "role": "HippoRAG"},
            "1": {"minimum_free_mib": 2048, "role": "Agent"},
        },
        "maximum_gpu_temperature_celsius": 82,
        "maximum_load1_per_cpu": 0.8,
        "maximum_median_cpu_busy_ratio": 0.70,
        "maximum_median_gpu_utilization_percent": 50,
        "minimum_host_mem_available_mib": 16384,
        "minimum_swap_free_mib": 0,
        "sample_count": 3,
        "sample_interval_seconds": 1.0,
        "schema": "wikisql_uao_shared_resource_policy_v1",
        "telemetry_timeout_seconds": 30,
    }
    parsed = subject.ResourcePolicy.parse(mapping)
    decision = subject.sample_and_decide(
        parsed,
        expected_gpu_uuids={
            str(index): uuid for index, uuid in GPU_UUIDS.items()
        },
        nvidia_smi_path=Path("/usr/bin/nvidia-smi"),
        sampler=_sampler([_snapshot(index) for index in range(3)]),
        sleeper=lambda _seconds: None,
    )

    assert isinstance(decision, subject.AdmissionDecision)
    assert decision.status == subject.ADMITTED
    assert decision.receipt["status"] == subject.ADMITTED
    assert decision.receipt["effect_attempt_claimed"] is False


def test_gpu_and_cpu_utilization_use_three_sample_medians(
    tmp_path: Path,
) -> None:
    one_spike = [
        _snapshot(0, utilization=80, cpu_busy=0.90),
        _snapshot(1, utilization=10, cpu_busy=0.10),
        _snapshot(2, utilization=10, cpu_busy=0.10),
    ]
    sustained = [
        _snapshot(0, utilization=80, cpu_busy=0.90),
        _snapshot(1, utilization=80, cpu_busy=0.90),
        _snapshot(2, utilization=10, cpu_busy=0.10),
    ]

    admitted = subject.decide_resource_admission(
        _config(tmp_path),
        one_spike,
    )
    deferred = subject.decide_resource_admission(
        _config(tmp_path),
        sustained,
    )

    assert admitted.status == subject.ADMITTED
    assert set(deferred.reason_codes) == {
        "GPU_0_UTILIZATION_ABOVE_THRESHOLD",
        "GPU_1_UTILIZATION_ABOVE_THRESHOLD",
        "HOST_CPU_BUSY_ABOVE_THRESHOLD",
    }


def test_independent_qualification_lock_context_reports_contention(
    tmp_path: Path,
) -> None:
    path = tmp_path / "facade.lock"
    with subject.qualification_lock(path) as first:
        assert first is True
        with pytest.raises(subject.ResourceBusyError) as raised:
            with subject.qualification_lock(path):
                raise AssertionError("occupied lock body must not run")
        assert raised.value.status == subject.DEFERRED_SHARED_RESOURCE
        assert raised.value.exit_code == subject.EX_TEMPFAIL
    with subject.qualification_lock(path) as after_release:
        assert after_release is True

from __future__ import annotations

from copy import deepcopy
import json
import os

import pytest

from hegel_machine import phase3_container_actor_runtime_v1 as runtime
from hegel_machine.phase3_container_actor_runtime_v1 import (
    FAIL_INPUT_BINDING,
    FAIL_IMPLEMENTATION_MISMATCH,
    FAIL_LIVE_PROBE,
    FAIL_OUTPUT_FRAMING,
    FAIL_PURPOSE_SEPARATION,
    FAIL_REPORT_INVALID,
    PROFILE_ID,
    PROBE_SCHEMA,
    ContainerActorQualificationError,
    _actor_environment,
    _compare_python_rust,
    _decode_probe_output,
    _fault_injection_checks,
    _load_profile,
    _validate_cross_actor,
    _validate_probe,
    run_live_qualification,
    validate_qualification_report,
)


PROBE_IDS = (
    "socket(AF_INET, SOCK_STREAM)",
    "socket(AF_INET6, SOCK_STREAM)",
    "mount",
    "ptrace(PTRACE_TRACEME)",
    "bpf(BPF_MAP_CREATE)",
    "perf_event_open",
)


def _probe(purpose_id: int, *, implementation: str = "python-ctypes-v1") -> dict[str, object]:
    return {
        "schema": PROBE_SCHEMA,
        "implementation": implementation,
        "profile_id": PROFILE_ID,
        "purpose_id": purpose_id,
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
            kind: f"{kind}:[{purpose_id + index + 100}]"
            for index, kind in enumerate(("pid", "mnt", "net", "ipc", "uts"))
        },
        "network_interfaces": ["lo"],
        "syscall_probes": [
            {"probe_id": probe_id, "return_value": -1, "errno": 1}
            for probe_id in PROBE_IDS
        ],
        "filesystem_probes": {
            "root_write": {"denied": True, "errno": 30},
            "input_write": {"denied": True, "errno": 13},
            "forbidden_paths_present": [],
            "cross_purpose_paths_present": [],
        },
        "environment": _actor_environment(purpose_id),
        "open_fds": [0, 1, 2],
    }


def _actor(purpose_id: int) -> dict[str, object]:
    implementation = "rust-ffi-v1" if purpose_id == 3 else "python-ctypes-v1"
    return {
        "purpose_id": purpose_id,
        "container_id": f"sha256-container-{purpose_id}",
        "host_pid_while_running": 1000 + purpose_id,
        "live_probe": _probe(purpose_id, implementation=implementation),
    }


def test_profile_forbids_pull_registry_and_runtime_network() -> None:
    profile = _load_profile()
    assert profile["network_policy"] == {
        "allow_registry_access": False,
        "allow_runtime_network": False,
        "docker_network": "none",
        "pull_policy": "never",
    }
    flags = set(profile["required_runtime_flags"])
    assert {"--pull=never", "--network=none", "--read-only", "--cap-drop=ALL"} <= flags
    assert profile["authority_disclosure"] == dict(
        runtime.TECHNICAL_ACTOR_DISCLOSURE_V1
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["authority_disclosure"].pop(
            "hardware_key_nonexportability"
        ),
        lambda value: value["authority_disclosure"].update(
            {"unregistered_independence_claim": False}
        ),
        lambda value: value["authority_disclosure"].update(
            {"organizational_independence": True}
        ),
        lambda value: value["authority_disclosure"].update(
            {"same_admin_controller": 1}
        ),
    ],
    ids=("missing", "extra", "wrong-value", "wrong-json-type"),
)
def test_profile_rejects_nonexact_seven_field_authority_disclosure(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mutate
) -> None:
    profile = json.loads(runtime.PROFILE_PATH.read_text(encoding="utf-8"))
    mutate(profile)
    attacked = tmp_path / "actor-profile.json"
    attacked.write_text(
        json.dumps(profile, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )
    monkeypatch.setattr(runtime, "PROFILE_PATH", attacked)
    with pytest.raises(ContainerActorQualificationError) as caught:
        runtime._load_profile()
    assert caught.value.code == FAIL_INPUT_BINDING


def test_exact_custom_probe_passes() -> None:
    result = _validate_probe(
        _probe(1),
        purpose_id=1,
        implementation="python-ctypes-v1",
        require_custom_blocking=True,
    )
    assert result["blocked_syscalls"] == list(PROBE_IDS)
    assert result["exact_environment"] is True
    assert result["exact_inherited_fds"] is True


def test_live_probe_binds_actual_clone_path_without_disclosing_it() -> None:
    launch = runtime._actor_launch_environment(1)
    reported = _actor_environment(1)
    actual = runtime.REPOSITORY_ROOT.resolve().as_posix()

    assert launch["HEGEL_HOST_REPOSITORY_PATH"] == actual
    assert "HEGEL_HOST_REPOSITORY_PATH" not in reported
    assert reported["HEGEL_HOST_REPOSITORY_PATH_SHA256"] == (
        runtime.hashlib.sha256(actual.encode("utf-8")).hexdigest()
    )
    for probe_path in (runtime.PYTHON_PROBE_PATH, runtime.RUST_PROBE_PATH):
        source = probe_path.read_text(encoding="utf-8")
        assert "/home/" not in source
        assert "HEGEL_HOST_REPOSITORY_PATH" in source


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        (lambda value: value["proc_status"].update({"CapEff": "1"}), "capability"),
        (lambda value: value.update({"network_interfaces": ["eth0", "lo"]}), "network"),
        (lambda value: value.update({"open_fds": [0, 1, 2, 3]}), "FD"),
        (lambda value: value["environment"].update({"HOME": "/root"}), "environment"),
        (
            lambda value: value["syscall_probes"][0].update(
                {"return_value": 3, "errno": 0}
            ),
            "syscall",
        ),
    ],
)
def test_live_probe_mutations_fail_closed(mutation, detail: str) -> None:
    value = _probe(1)
    mutation(value)
    with pytest.raises(ContainerActorQualificationError) as caught:
        _validate_probe(
            value,
            purpose_id=1,
            implementation="python-ctypes-v1",
            require_custom_blocking=True,
        )
    assert caught.value.code == FAIL_LIVE_PROBE, detail


def test_default_docker_seccomp_evidence_is_rejected() -> None:
    value = _probe(1)
    for row in value["syscall_probes"][:2]:
        row.update({"return_value": 3, "errno": 0})
    value["syscall_probes"][3].update({"return_value": 0, "errno": 0})
    permissive = _validate_probe(
        value,
        purpose_id=1,
        implementation="python-ctypes-v1",
        require_custom_blocking=False,
    )
    assert set(permissive["allowed_syscalls"]) == {
        "socket(AF_INET, SOCK_STREAM)",
        "socket(AF_INET6, SOCK_STREAM)",
        "ptrace(PTRACE_TRACEME)",
    }
    with pytest.raises(ContainerActorQualificationError) as caught:
        _validate_probe(
            value,
            purpose_id=1,
            implementation="python-ctypes-v1",
            require_custom_blocking=True,
        )
    assert caught.value.code == FAIL_LIVE_PROBE


def test_python_rust_agreement_and_mismatch_injection() -> None:
    python_probe = _probe(2)
    rust_probe = _probe(3, implementation="rust-ffi-v1")
    assert len(_compare_python_rust(python_probe, rust_probe)) == 64
    rust_probe["network_interfaces"] = ["eth0", "lo"]
    with pytest.raises(ContainerActorQualificationError) as caught:
        _compare_python_rust(python_probe, rust_probe)
    assert caught.value.code == FAIL_IMPLEMENTATION_MISMATCH


def test_cross_actor_requires_distinct_containers_processes_and_namespaces() -> None:
    actors = [_actor(purpose_id) for purpose_id in range(1, 5)]
    result = _validate_cross_actor(actors)
    assert result["technical_role_independence"] is True
    assert all(result["namespace_identity_distinct_by_kind"].values())
    replay = deepcopy(actors)
    replay[3]["live_probe"]["namespaces"]["net"] = replay[0]["live_probe"]["namespaces"]["net"]
    with pytest.raises(ContainerActorQualificationError) as caught:
        _validate_cross_actor(replay)
    assert caught.value.code == FAIL_PURPOSE_SEPARATION


def test_all_declared_fault_injections_are_rejected() -> None:
    assert all(_fault_injection_checks([_actor(value) for value in range(1, 5)]).values())


def test_output_is_one_bounded_secret_free_json_line() -> None:
    encoded = json.dumps(_probe(1), separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
    assert _decode_probe_output(encoded)["purpose_id"] == 1
    for invalid in (b"{}\n{}\n", b'{"raw_seed":"00"}\n'):
        with pytest.raises(ContainerActorQualificationError) as caught:
            _decode_probe_output(invalid)
        assert caught.value.code == FAIL_OUTPUT_FRAMING


@pytest.mark.skipif(
    os.environ.get("HEGEL_RUN_CONTAINER_ACTOR_LIVE_TEST") != "1",
    reason="explicit opt-in required for the local Docker live qualification",
)
def test_real_offline_docker_qualification() -> None:
    report = run_live_qualification()
    assert validate_qualification_report(report) == report
    tampered = deepcopy(report)
    tampered["authority_disclosure"]["organizational_independence"] = True
    with pytest.raises(ContainerActorQualificationError) as caught:
        validate_qualification_report(tampered)
    assert caught.value.code == FAIL_REPORT_INVALID
    assert report["all_live_checks_passed"] is True
    assert report["network_and_registry_policy"] == {
        "registry_access_performed": False,
        "image_pull_performed": False,
        "image_build_performed": False,
        "runtime_network_enabled": False,
        "pull_policy": "never",
        "runtime_network": "none",
    }
    assert report["authority_disclosure"]["same_admin_controller"] is True
    assert report["authority_disclosure"]["organizational_independence"] is False
    assert report["ceremony_outputs"]["split_seed_generated"] is False
    assert report["ceremony_outputs"]["ephemeral_signing_keys_generated"] is False
    assert report["ceremony_outputs"]["formal_roots_generated"] is False
    assert all(actor["cleanup"]["container_and_descendants_absent"] for actor in report["actor_reports"])

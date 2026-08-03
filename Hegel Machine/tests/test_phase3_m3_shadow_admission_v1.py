from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from typing import Iterator

import pytest

from hegel_machine import phase3_m3_shadow_admission_v1 as admission
from hegel_machine import phase3_m3_shadow_cli_v1 as shadow_cli
from hegel_machine import phase3_m3_shadow_runtime_v1 as shadow_runtime
from hegel_machine.phase3_m3_shadow_wire_v1 import (
    FORMAL_TRACK_SNAPSHOT,
    SHADOW_ARTIFACT_KIND,
    SHADOW_OBJECT_TAGS,
    ShadowStateId,
    decode_shadow_object,
    shadow_object_digest,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


BASIS = "a" * 40
RUN_ID = bytes.fromhex("11" * 16)
TIMESTAMP = 1_800_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENDPOINT = (PROJECT_ROOT / "tools" / "phase3_split_calculator_fd3_v1.py").resolve()
RUST_ENDPOINT_SOURCE = (
    PROJECT_ROOT / "tools" / "phase3_split_calculator_fd3_v1.rs"
).resolve()


def _security(purpose_id: int) -> dict[str, object]:
    return {
        "isolation_level": "BWRAP_USER_PID_NET_IPC_UTS_SECCOMP_V1",
        "namespace_links": {
            name: f"{name}:[{purpose_id + 100}]"
            for name in ("user", "pid", "net", "ipc", "uts")
        },
        "namespace_unshared_from_orchestrator": {
            name: True for name in ("user", "pid", "net", "ipc", "uts")
        },
        "seccomp_mode": 2,
        "no_new_privs": 1,
        "effective_capabilities_hex": "0000000000000000",
        "seccomp_forbidden_syscalls": ["socket"],
        "seccomp_forbidden_syscall_count": 1,
        "attack_syscall_errno_rows": [
            {"attack_id": attack_id, "errno": "EPERM"}
            for attack_id in range(1, 7)
        ],
        "attack_syscall_probe_count": 6,
        "repository_mount_read_only_live_probe": True,
        "tmp_mount_type": "tmpfs",
        "network_interfaces": ["lo"],
        "landlock_status": "UNAVAILABLE_NONBLOCKING_GAP_DISCLOSED",
        "transient_capability_probe_incident_count": 0,
        "network_fetch_allowed": False,
    }


def _process(purpose_id: int) -> dict[str, object]:
    return {
        "role": f"ROLE_{purpose_id}",
        "purpose_id": purpose_id,
        "process_id": 1000 + purpose_id,
        "environment_keys": ["LANG", "LC_ALL", "PATH", "TZ"],
        "working_directory_mode_octal": "0700",
        "security_evidence": _security(purpose_id),
    }


def _probe_report(*, basis: str | None = BASIS) -> dict[str, object]:
    return {
        "schema_version": "hegel-phase3-m3-shadow-admission/1",
        "authority_class": SHADOW_ARTIFACT_KIND,
        "basis_commit_id_or_null": basis,
        "ceremony_id_hex": RUN_ID.hex(),
        "snapshot_manifest_sha256_hex": "22" * 32,
        "purpose_probe_receipts": [
            {
                "purpose_id": purpose_id,
                "role": f"ROLE_{purpose_id}",
                "process": _process(purpose_id),
                "receipt_sha256_hex": f"{purpose_id:064x}",
            }
            for purpose_id in range(1, 5)
        ],
        "isolation_plan_inputs": {
            "calculator_endpoint_sha256_hex": {
                "python": "91" * 32,
                "rust": "92" * 32,
            }
        },
        "side_effects": {
            "key_generated": False,
            "seed_generated": False,
            "marker_written": False,
            "formal_root_issued": False,
        },
        "authority_boundary": {
            "formal_gates_before": 14,
            "formal_gates_after": 14,
            "formal_gates_total": 24,
            "formal_gate_delta": 0,
            "formal_state_before": "NOT_RUN",
            "formal_state_after": "NOT_RUN",
        },
        "admission_status": "INTERNAL_SHADOW_ADMISSION_PASS",
    }


def _snapshot() -> dict[str, object]:
    return {
        "basis_commit_id": BASIS,
        "hegel_subtree_git_id": "b" * 40,
        "entry_count": 123,
        "snapshot_manifest_digest": "33" * 32,
        "bound_input_count": 40,
        "bound_input_manifest_digest": "44" * 32,
        "detached_git_objects_only": True,
        "live_worktree_input_count": 0,
        "read_only_materialization_required": True,
    }


@contextmanager
def _fake_detached(
    _basis: str, _paths: tuple[str, ...]
) -> Iterator[dict[str, Path]]:
    yield {"input_0000": Path("/detached/input_0000")}


def _patch_admission_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "_assert_basis_reachable", lambda _basis: None)
    monkeypatch.setattr(admission, "_assert_hegel_worktree_clean", lambda: None)
    monkeypatch.setattr(
        admission,
        "_artifact_blob_and_bindings",
        lambda _basis: (
            {"source_bindings": {}},
            {
                "path": admission.CHECKED_REPORT_RELATIVE_PATH,
                "committed_blob_sha256": "55" * 32,
                "implementation_basis_commit": "c" * 40,
                "diagnostic_report_id": "candidate",
                "status": "DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS",
                "source_binding_count": 27,
                "source_bindings_match_shadow_basis": True,
                "checked_artifact_matches_shadow_basis_blob": True,
            },
        ),
    )
    monkeypatch.setattr(
        admission,
        "_snapshot_receipt",
        lambda _basis, _checked: (_snapshot(), ("committed.txt",)),
    )
    monkeypatch.setattr(admission, "_git_blob", lambda _basis, _path: b"amendment")
    monkeypatch.setattr(admission, "_detached_readonly_inputs", _fake_detached)
    monkeypatch.setattr(admission, "_call_runtime_probe", lambda **_kwargs: _probe_report())
    monkeypatch.setattr(admission, "_validate_runtime_probe_report", lambda _report: None)


def _admitted(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    _patch_admission_inputs(monkeypatch)
    return admission.admit_internal_shadow(
        basis_commit_id=BASIS,
        shadow_run_id=RUN_ID,
        admitted_at_unix_seconds=TIMESTAMP,
        python_calculator_path=PYTHON_ENDPOINT,
        rust_calculator_path=RUST_ENDPOINT_SOURCE,
    )


def _runtime_report() -> dict[str, object]:
    envelopes: list[dict[str, object]] = []
    for purpose_id in range(1, 5):
        worker_id = bytes([purpose_id]) * 16
        payload = (
            1,
            1,
            b"internal",
            purpose_id,
            f"ROLE_{purpose_id}".encode("ascii"),
            RUN_ID,
            bytes.fromhex("66" * 32),
            bytes.fromhex("67" * 32),
            bytes.fromhex("68" * 32),
            bytes.fromhex("69" * 32),
            bytes.fromhex("6a" * 32),
            worker_id,
            bytes.fromhex("6b" * 32),
            2000 + purpose_id,
        )
        envelopes.append(
            {
                "purpose_id": purpose_id,
                "key_id_hex": (bytes([purpose_id + 10]) * 16).hex(),
                "public_key_hex": (bytes([purpose_id + 20]) * 32).hex(),
                "payload_cbor_hex": canonical_cbor_encode(payload).hex(),
            }
        )
    return {
        "schema_version": "hegel-phase3-m3-shadow-runtime/1",
        "ceremony_id_hex": RUN_ID.hex(),
        "envelopes": envelopes,
        "fresh_admission_probes": _probe_report(basis=None),
        "process_isolation": {
            "role_processes": [_process(purpose_id) for purpose_id in range(1, 5)]
        },
        "authority_boundary": {
            "formal_gate_delta": 0,
            "formal_state_before": "NOT_RUN",
            "formal_state_after": "NOT_RUN",
            "formal_roots_issued": False,
        },
    }


def test_builders_delegate_to_frozen_shadow_wire_registry() -> None:
    policy = admission.build_policy_binding(
        basis_commit_id=BASIS,
        amendment_blob_sha256=bytes.fromhex("aa" * 32),
    )
    decoded = decode_shadow_object(
        canonical_cbor_encode(policy), expected_name="ShadowPolicyBindingV1"
    )
    assert decoded.schema.tag == SHADOW_OBJECT_TAGS["ShadowPolicyBindingV1"]
    assert admission.shadow_digest("ShadowPolicyBindingV1", policy) == (
        shadow_object_digest("ShadowPolicyBindingV1", decoded.fields)
    )


def test_admission_is_exact_12_of_12_and_has_no_formal_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _admitted(monkeypatch)
    admission.validate_admission_artifact(report)
    assert report["formal_track"] == dict(FORMAL_TRACK_SNAPSHOT)
    assert report["formal_track_status"] == "14/24 / NOT_RUN"
    assert report["shadow_track"]["state"] == "ADMITTED_NOT_STARTED"
    assert len(report["gate_results"]) == 12
    assert len(report["security_probe_wire_receipts"]) == 4
    assert report["formal_follow_on_recommendation"].startswith("SEPARATE_OWNER_AMENDED")
    encoded = json.dumps(report, sort_keys=True)
    assert '"private_key"' not in encoded
    assert '"raw_seed"' not in encoded


def test_admission_rejects_formal_mutation_and_shadow_root_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _admitted(monkeypatch)
    mutated = deepcopy(report)
    mutated["formal_track"]["gates_satisfied"] = 24
    with pytest.raises(admission.ShadowAdmissionError):
        admission.validate_admission_artifact(mutated)

    mutated = deepcopy(report)
    mutated["shadow_track"]["candidate_root"] = "00" * 32
    with pytest.raises(admission.ShadowAdmissionError):
        admission.validate_admission_artifact(mutated)


def test_explicit_start_reuses_run_id_and_only_advances_shadow_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    admitted = _admitted(monkeypatch)
    monkeypatch.setattr(admission, "_runtime_ceremony", lambda **_kwargs: _runtime_report())
    monkeypatch.setattr(admission, "_validate_runtime_ceremony_report", lambda _report: None)
    started = admission.start_internal_shadow(
        admitted,
        state_directory=tmp_path / "private-state",
        python_calculator_path=PYTHON_ENDPOINT,
        rust_calculator_path=RUST_ENDPOINT_SOURCE,
        started_at_unix_seconds=TIMESTAMP + 1,
    )
    admission.validate_start_artifact(started)
    assert started["shadow_run_id"] == admitted["shadow_run_id"]
    assert started["shadow_track"] == {
        "admission_gates": "12/12",
        "state_id": ShadowStateId.RUNNING_CANONICAL_ENUMERATION.value,
        "state": "RUNNING_CANONICAL_ENUMERATION",
        "start_action": "phase3-m3-shadow-start",
        "purpose_ids": [1, 2, 3, 4],
    }
    assert started["formal_track"] == dict(FORMAL_TRACK_SNAPSHOT)
    assert len(started["purpose_worker_wire_manifests"]) == 4
    assert len(started["runtime_security_probe_wire_receipts"]) == 4
    state_value = decode_shadow_object(
        bytes.fromhex(started["wire_objects"]["state_record"]["cbor_hex"]),
        expected_name="ShadowStateRecordV1",
    ).value
    assert state_value[6].hex() == started["admission_state_record_digest"]


def test_missing_runtime_admission_api_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shadow_runtime, "probe_shadow_admission", None)
    with pytest.raises(admission.ShadowAdmissionError) as raised:
        admission._call_runtime_probe(
            basis_commit_id=BASIS,
            shadow_run_id=RUN_ID,
            input_files={"input": Path("/detached/input")},
            python_calculator_path=PYTHON_ENDPOINT,
            rust_calculator_path=RUST_ENDPOINT_SOURCE,
        )
    assert raised.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"


def test_exclusive_publication_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "shadow.json"
    admission.write_json_exclusive(target, {"first": True})
    with pytest.raises(admission.ShadowAdmissionError) as raised:
        admission.write_json_exclusive(target, {"second": True})
    assert raised.value.code == "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED"
    assert json.loads(target.read_text(encoding="utf-8")) == {"first": True}


def test_dedicated_cli_publishes_without_touching_formal_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "artifact_kind": SHADOW_ARTIFACT_KIND,
        "formal_track_status": "14/24 / NOT_RUN",
        "shadow_run_id": RUN_ID.hex(),
        "shadow_track": {"state": "ADMITTED_NOT_STARTED"},
    }
    forwarded: dict[str, object] = {}

    def fake_admit(**kwargs: object) -> dict[str, object]:
        forwarded.update(kwargs)
        return report

    monkeypatch.setattr(shadow_cli, "admit_internal_shadow", fake_admit)
    output = tmp_path / "cli-admission.json"
    assert shadow_cli.main(
        [
            "admit",
            "--python-calculator",
            str(PYTHON_ENDPOINT),
            "--rust-calculator",
            str(RUST_ENDPOINT_SOURCE),
            "--output",
            str(output),
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["formal_gate_delta"] == 0
    assert summary["shadow_state"] == "ADMITTED_NOT_STARTED"
    assert forwarded["python_calculator_path"] == PYTHON_ENDPOINT
    assert forwarded["rust_calculator_path"] == RUST_ENDPOINT_SOURCE
    assert output.is_file()

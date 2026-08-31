from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_supervisor_v5",
    ROOT / "tools" / "phase3_q1_start_supervisor_v5.py",
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
supervisor = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = supervisor
MODULE_SPEC.loader.exec_module(supervisor)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _digest(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


def _run_spec_payload(tmp_path: Path) -> bytes:
    run = tmp_path / "fresh-v5-run"
    run.mkdir()
    manifests: dict[str, list[dict[str, object]]] = {}
    for stage in supervisor.STAGES:
        root = run / f"{stage}-source"
        root.mkdir()
        payload = f"{stage}-v5\n".encode("ascii")
        (root / "actor.py").write_bytes(payload)
        manifests[stage] = [{
            "relative_path": "actor.py",
            "byte_count": len(payload),
            "raw_sha256": _digest(payload),
        }]
    rust_binary = tmp_path / "rust-binary"
    rust_binary.write_bytes(b"rust-v5")
    rust_binary.chmod(0o555)
    admission = tmp_path / "admission.json"
    admission.write_bytes(_canonical({"kind": "admission"}))
    seccomp = tmp_path / "seccomp.json"
    seccomp.write_bytes(_canonical({"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}))
    evidence_paths: dict[str, Path] = {}
    for name in (
        "start-capsule", "start-authorization", "successor-bundle",
        "base-prestart", "failed-v2-capsule", "failed-v2-authorization",
        "failed-v2-consumption", "failed-v2-terminal",
        "failed-v3-capsule", "failed-v3-authorization",
        "failed-v3-consumption", "failed-v3-terminal",
        "failed-v4-capsule", "failed-v4-authorization",
        "failed-v4-consumption", "failed-v4-terminal",
    ):
        path = (run if name == "start-capsule" else tmp_path) / f"{name}.json"
        path.write_bytes(_canonical({"name": name}))
        evidence_paths[name] = path
    digest = lambda token: "sha256:" + token * 64
    limits = {
        "python": supervisor.RuntimeLimitV5("24", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "rust": supervisor.RuntimeLimitV5("25", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "host": supervisor.RuntimeLimitV5("26", 67_108_864, 2_097_152, 1, 2_097_152, 3_600).object(),
    }
    return _canonical({
        "schema_version": supervisor.RUN_SPEC_SCHEMA,
        "start_capsule_digest": digest("1"),
        "start_capsule": str(evidence_paths["start-capsule"]),
        "start_capsule_raw_sha256": _digest(evidence_paths["start-capsule"].read_bytes()),
        "start_authorization": str(evidence_paths["start-authorization"]),
        "start_authorization_raw_sha256": _digest(evidence_paths["start-authorization"].read_bytes()),
        "start_authorization_digest": digest("2"),
        "successor_prestart_bundle": str(evidence_paths["successor-bundle"]),
        "successor_prestart_bundle_raw_sha256": _digest(evidence_paths["successor-bundle"].read_bytes()),
        "successor_prestart_bundle_root": digest("3"),
        "base_authoritative_artifact": str(evidence_paths["base-prestart"]),
        "base_authoritative_artifact_raw_sha256": _digest(evidence_paths["base-prestart"].read_bytes()),
        "failed_start_v2_capsule": str(evidence_paths["failed-v2-capsule"]),
        "failed_start_v2_capsule_raw_sha256": supervisor.FAILED_V2_CAPSULE_RAW_SHA256,
        "failed_start_v2_capsule_digest": supervisor.FAILED_V2_CAPSULE_DIGEST,
        "failed_start_v2_authorization": str(evidence_paths["failed-v2-authorization"]),
        "failed_start_v2_authorization_raw_sha256": supervisor.FAILED_V2_AUTHORIZATION_RAW_SHA256,
        "failed_start_v2_authorization_digest": supervisor.FAILED_V2_AUTHORIZATION_DIGEST,
        "failed_start_v2_consumption": str(evidence_paths["failed-v2-consumption"]),
        "failed_start_v2_consumption_raw_sha256": supervisor.FAILED_V2_CONSUMPTION_RAW_SHA256,
        "failed_start_v2_consumption_record_digest": supervisor.FAILED_V2_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v2_terminal": str(evidence_paths["failed-v2-terminal"]),
        "failed_start_v2_terminal_raw_sha256": supervisor.FAILED_V2_TERMINAL_RAW_SHA256,
        "failed_start_v2_terminal_record_digest": supervisor.FAILED_V2_TERMINAL_RECORD_DIGEST,
        "failed_start_v3_capsule": str(evidence_paths["failed-v3-capsule"]),
        "failed_start_v3_capsule_raw_sha256": supervisor.FAILED_V3_CAPSULE_RAW_SHA256,
        "failed_start_v3_capsule_digest": supervisor.FAILED_V3_CAPSULE_DIGEST,
        "failed_start_v3_authorization": str(evidence_paths["failed-v3-authorization"]),
        "failed_start_v3_authorization_raw_sha256": supervisor.FAILED_V3_AUTHORIZATION_RAW_SHA256,
        "failed_start_v3_authorization_digest": supervisor.FAILED_V3_AUTHORIZATION_DIGEST,
        "failed_start_v3_consumption": str(evidence_paths["failed-v3-consumption"]),
        "failed_start_v3_consumption_raw_sha256": supervisor.FAILED_V3_CONSUMPTION_RAW_SHA256,
        "failed_start_v3_consumption_record_digest": supervisor.FAILED_V3_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v3_terminal": str(evidence_paths["failed-v3-terminal"]),
        "failed_start_v3_terminal_raw_sha256": supervisor.FAILED_V3_TERMINAL_RAW_SHA256,
        "failed_start_v3_terminal_record_digest": supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST,
        "failed_start_v4_capsule": str(evidence_paths["failed-v4-capsule"]),
        "failed_start_v4_capsule_raw_sha256": supervisor.FAILED_V4_CAPSULE_RAW_SHA256,
        "failed_start_v4_capsule_digest": supervisor.FAILED_V4_CAPSULE_DIGEST,
        "failed_start_v4_authorization": str(evidence_paths["failed-v4-authorization"]),
        "failed_start_v4_authorization_raw_sha256": supervisor.FAILED_V4_AUTHORIZATION_RAW_SHA256,
        "failed_start_v4_authorization_digest": supervisor.FAILED_V4_AUTHORIZATION_DIGEST,
        "failed_start_v4_consumption": str(evidence_paths["failed-v4-consumption"]),
        "failed_start_v4_consumption_raw_sha256": supervisor.FAILED_V4_CONSUMPTION_RAW_SHA256,
        "failed_start_v4_consumption_record_digest": supervisor.FAILED_V4_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v4_terminal": str(evidence_paths["failed-v4-terminal"]),
        "failed_start_v4_terminal_raw_sha256": supervisor.FAILED_V4_TERMINAL_RAW_SHA256,
        "failed_start_v4_terminal_record_digest": supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST,
        "corrected_resource_envelope_root": digest("6"),
        "actor_source_manifest_roots": {"python": digest("7"), "rust": digest("8"), "host": digest("9")},
        "actor_source_closure_root": digest("a"),
        "source_commit": "b" * 40,
        "run_genesis_root": digest("c"),
        "prestart_admission_receipt_root": digest("d"),
        "python_image": "python@" + digest("e"),
        "rust_image": "rust@" + digest("f"),
        "rust_binary_raw_sha256": _digest(rust_binary.read_bytes()),
        "seccomp_profile_raw_sha256": _digest(seccomp.read_bytes()),
        "python_admission_raw_sha256": _digest(admission.read_bytes()),
        "actor_implementation_roots": {"python": digest("1"), "rust": digest("2"), "host": digest("3")},
        "python_source_root": str(run / "python-source"),
        "rust_source_root": str(run / "rust-source"),
        "host_source_root": str(run / "host-source"),
        "rust_binary": str(rust_binary),
        "python_admission_json": str(admission),
        "seccomp_profile": str(seccomp),
        "run_root": str(run),
        "checkpoint_root": str(run / "checkpoints"),
        "python_output": str(run / "python-output"),
        "python_scratch": str(run / "python-scratch"),
        "rust_output": str(run / "rust-output"),
        "rust_scratch": str(run / "rust-scratch"),
        "host_output": str(run / "host-output"),
        "host_scratch": str(run / "host-scratch"),
        "runtime_limits": limits,
        "actor_source_manifests": manifests,
    })


def _failed_lineage_fixture(version: int) -> tuple[dict[str, object], ...]:
    if version == 2:
        capsule_digest = supervisor.FAILED_V2_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V2_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V2_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V2_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V2_TERMINAL_STATUS
        failure: dict[str, object] = {}
        terminal_actor_starts = {"host": 0, "python": 0, "rust": 0}
    elif version == 3:
        capsule_digest = supervisor.FAILED_V3_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V3_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V3_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V3_TERMINAL_STATUS
        failure = {
            "failure_code": "REJECT_Q1_START_SUCCESSOR_BUNDLE",
            "failure_detail": (
                "validator import failed: AttributeError: module 'importlib' has no attribute 'util'"
            ),
        }
        terminal_actor_starts = {"host": 0, "python": 0, "rust": 0}
    elif version == 4:
        capsule_digest = supervisor.FAILED_V4_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V4_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V4_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V4_TERMINAL_STATUS
        failure = {
            "failure_code": "Q1_START_ENDPOINT_REPLAY_MISMATCH",
            "failure_detail": (
                "python:Q1FormalHostReplayError:REJECT_Q1_FORMAL_ENDPOINT_RESULT: "
                "same_run_record_set_root differs from replay"
            ),
            "supervisor_outcome_raw_sha256": (
                supervisor.FAILED_V4_SUPERVISOR_OUTCOME_RAW_SHA256
            ),
        }
        terminal_actor_starts = {"host": 0, "python": 1, "rust": 0}
    else:
        raise AssertionError(f"unsupported predecessor version {version}")
    capsule = {
        "schema_version": f"hegel-phase3a-q1-start-capsule/{version}",
        "status": f"Q1_START_V{version}_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": capsule_digest,
    }
    authorization = {
        "schema_version": f"hegel-phase3a-q1-start-authorization/{version}",
        "status": f"Q1_START_V{version}_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
    }
    consumption = {
        "schema_version": f"hegel-phase3a-q1-start-consumption/{version}",
        "status": f"Q1_START_V{version}_AUTHORIZATION_CONSUMED",
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
        "record_digest": consumption_digest,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    outcome = {
        "status": status,
        "capsule_digest": capsule_digest,
        "actor_start_count_by_actor": terminal_actor_starts,
        **failure,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    terminal = {
        "schema_version": f"hegel-phase3a-q1-start-terminal/{version}",
        "status": status,
        "record_digest": terminal_digest,
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
        "consumption_record_digest": consumption_digest,
        "logical_campaign_count": 1,
        "authorization_consumption_count": 1,
        "artifact_raw_sha256": None,
        "outcome": outcome,
    }
    return capsule, authorization, consumption, terminal


def _isolated_smoke_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    fixture = tmp_path / "isolated-fixture"
    tools = fixture / "tools"
    package = fixture / "src" / "hegel_machine"
    run = fixture / "run"
    tools.mkdir(parents=True)
    package.mkdir(parents=True)
    run.mkdir()

    predecessor_paths: dict[tuple[int, str], Path] = {}
    predecessor_raw: dict[tuple[int, str], str] = {}
    labels = ("capsule", "authorization", "consumption", "terminal")
    for version in (2, 3, 4):
        for label, value in zip(labels, _failed_lineage_fixture(version), strict=True):
            path = fixture / f"failed-v{version}-{label}.json"
            path.write_bytes(_canonical(value))
            predecessor_paths[version, label] = path
            predecessor_raw[version, label] = _digest(path.read_bytes())

    source_text = (ROOT / "tools" / "phase3_q1_start_supervisor_v5.py").read_text(
        encoding="utf-8"
    )
    raw_constant_names = {
        (2, "capsule"): "FAILED_V2_CAPSULE_RAW_SHA256",
        (2, "authorization"): "FAILED_V2_AUTHORIZATION_RAW_SHA256",
        (2, "consumption"): "FAILED_V2_CONSUMPTION_RAW_SHA256",
        (2, "terminal"): "FAILED_V2_TERMINAL_RAW_SHA256",
        (3, "capsule"): "FAILED_V3_CAPSULE_RAW_SHA256",
        (3, "authorization"): "FAILED_V3_AUTHORIZATION_RAW_SHA256",
        (3, "consumption"): "FAILED_V3_CONSUMPTION_RAW_SHA256",
        (3, "terminal"): "FAILED_V3_TERMINAL_RAW_SHA256",
        (4, "capsule"): "FAILED_V4_CAPSULE_RAW_SHA256",
        (4, "authorization"): "FAILED_V4_AUTHORIZATION_RAW_SHA256",
        (4, "consumption"): "FAILED_V4_CONSUMPTION_RAW_SHA256",
        (4, "terminal"): "FAILED_V4_TERMINAL_RAW_SHA256",
    }
    for key, constant_name in raw_constant_names.items():
        original = getattr(supervisor, constant_name)
        replacement = predecessor_raw[key]
        assert source_text.count(original) == 1
        source_text = source_text.replace(original, replacement)
    script = tools / "phase3_q1_start_supervisor_v5.py"
    script.write_text(source_text, encoding="utf-8")

    (package / "strict_cbor_v1.py").write_text("TOKEN = 'loaded'\n", encoding="utf-8")
    manifest_roots = {
        "python": "sha256:" + "7" * 64,
        "rust": "sha256:" + "8" * 64,
        "host": "sha256:" + "9" * 64,
    }
    (package / "phase3_q1_prestart_successor_candidate_v5.py").write_text(
        "from .strict_cbor_v1 import TOKEN\n"
        "import json\n"
        f"ROOTS = {manifest_roots!r}\n"
        "def replay_q1_prestart_successor_candidate_v5(payload):\n"
        "    assert TOKEN == 'loaded'\n"
        "    return json.loads(payload.decode('ascii'))\n"
        "def successor_bundle_root_v5(value): return value['bundle_root']\n"
        "def actor_source_manifest_root_v5(actor, rows): return ROOTS[actor]\n",
        encoding="utf-8",
    )

    manifests: dict[str, list[dict[str, object]]] = {}
    for stage in supervisor.STAGES:
        source_root = run / f"{stage}-source"
        source_root.mkdir()
        payload = f"{stage}-isolated-v5\n".encode("ascii")
        (source_root / "actor.py").write_bytes(payload)
        manifests[stage] = [{
            "relative_path": "actor.py",
            "byte_count": len(payload),
            "raw_sha256": _digest(payload),
        }]
    rust_binary = fixture / "rust-binary"
    rust_binary.write_bytes(b"rust-v5-isolated")
    rust_binary.chmod(0o555)
    admission = fixture / "admission.json"
    admission.write_bytes(_canonical({"kind": "admission"}))
    seccomp = fixture / "seccomp.json"
    seccomp.write_bytes(_canonical({"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}))

    digest = lambda token: "sha256:" + token * 64
    source_commit = "b" * 40
    bundle_root = digest("3")
    receipt_root = digest("d")
    run_genesis_root = digest("c")
    implementation_roots = {
        "python": digest("1"), "rust": digest("2"), "host": digest("3")
    }
    corrected_root = digest("6")
    bundle = {
        "schema_version": supervisor.SUCCESSOR_BUNDLE_SCHEMA,
        "status": supervisor.SUCCESSOR_BUNDLE_STATUS,
        "bundle_root": bundle_root,
        "receipt_root": receipt_root,
        "run_genesis_root": run_genesis_root,
        "actor_implementation_roots": implementation_roots,
        "actor_source_manifest_roots": manifest_roots,
        "actor_source_closure_root": digest("a"),
        "corrected_resource_envelope_root": corrected_root,
        "source_identity": {"source_commit": source_commit, "source_tree": digest("b")},
        "corrected_resource_envelope": {
            "endpoint": {
                "memory_bytes": 2_147_483_648,
                "output_bytes": 16_777_216,
                "scratch_bytes": 134_217_728,
                "wall_time_seconds": 3_600,
            },
            "trusted_host_replay": {
                "memory_bytes": 67_108_864,
                "output_bytes": 2_097_152,
                "scratch_bytes": 1,
                "wall_time_seconds": 3_600,
            },
        },
        "intended_q1_state": "NOT_RUN",
        "intended_q1_gate_count": 14,
        "intended_q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "candidate_only": True,
        "authorized": False,
        "authorization_consumption_count": 0,
        "campaign_execution_count": 0,
        "popen_count": 0,
        "published": False,
    }
    bundle_path = fixture / "successor-bundle.json"
    bundle_path.write_bytes(_canonical(bundle))
    base = {
        "status": supervisor.BASE_PRESTART_STATUS,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q1_execution_started": False,
        "future_q1_start_requires_distinct_authorization": True,
        "authorization_consumption_count": 1,
        "campaign_execution_count": 1,
        "popen_count": 0,
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    base_path = fixture / "base-prestart.json"
    base_path.write_bytes(_canonical(base))

    capsule_digest = digest("1")
    v2_descriptor = {
        "capsule_path": str(predecessor_paths[2, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[2, "capsule"],
        "capsule_digest": supervisor.FAILED_V2_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[2, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[2, "authorization"],
        "authorization_digest": supervisor.FAILED_V2_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[2, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[2, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V2_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[2, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[2, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V2_TERMINAL_RECORD_DIGEST,
    }
    v3_descriptor = {
        "capsule_path": str(predecessor_paths[3, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[3, "capsule"],
        "capsule_digest": supervisor.FAILED_V3_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[3, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[3, "authorization"],
        "authorization_digest": supervisor.FAILED_V3_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[3, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[3, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V3_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[3, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[3, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST,
        "terminal_no_resume": True,
        "work_product_reuse_allowed": False,
        "prior_actor_actions_conservatively_counted": False,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    v4_descriptor = {
        "capsule_path": str(predecessor_paths[4, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[4, "capsule"],
        "capsule_digest": supervisor.FAILED_V4_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[4, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[4, "authorization"],
        "authorization_digest": supervisor.FAILED_V4_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[4, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[4, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V4_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[4, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[4, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST,
        "status": supervisor.FAILED_V4_TERMINAL_STATUS,
        "failure_code": "Q1_START_ENDPOINT_REPLAY_MISMATCH",
        "failure_detail": (
            "python:Q1FormalHostReplayError:REJECT_Q1_FORMAL_ENDPOINT_RESULT: "
            "same_run_record_set_root differs from replay"
        ),
        "actor_start_count_by_actor": {"host": 0, "python": 1, "rust": 0},
        "completed_prefix": [],
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "terminal_no_resume": True,
        "prior_actor_actions_conservatively_counted": True,
        "work_product_reuse_allowed": False,
    }
    capsule = {
        "schema_version": supervisor.START_CAPSULE_SCHEMA,
        "status": "Q1_START_V5_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": capsule_digest,
        "formal_source_commit": source_commit,
        "formal_source_tree": bundle["source_identity"]["source_tree"],
        "successor_prestart_bundle": bundle,
        "successor_prestart_bundle_descriptor": {
            "path": str(bundle_path),
            "raw_sha256": _digest(bundle_path.read_bytes()),
            "bundle_root": bundle_root,
        },
        "base_authoritative_predecessor": {
            "prestart_artifact_path": str(base_path),
            "prestart_artifact_raw_sha256": _digest(base_path.read_bytes()),
        },
        "failed_start_v2_predecessor": v2_descriptor,
        "failed_start_v3_predecessor": v3_descriptor,
        "failed_start_v4_predecessor": v4_descriptor,
    }
    capsule_path = run / "capsule.json"
    capsule_path.write_bytes(_canonical(capsule))
    authorization_digest = digest("2")
    authorization = {
        "schema_version": supervisor.START_AUTHORIZATION_SCHEMA,
        "status": "Q1_START_V5_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
        "authorized_scope": supervisor.START_AUTHORIZATION_SCOPE,
        "successor_prestart_bundle_raw_sha256": _digest(bundle_path.read_bytes()),
        "successor_prestart_bundle_root": bundle_root,
        "base_authoritative_artifact_raw_sha256": _digest(base_path.read_bytes()),
        "authorization_consumption_limit": 1,
        "maximum_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "network_allowed": False,
        "q1_start_authorized": True,
        "q2_start_authorized": False,
        "adoption_promotion_current_active_authorized": False,
        "fresh_v5_actor_actions_authorized": True,
        "failed_start_v2_resume_authorized": False,
        "failed_start_v2_work_product_reuse_authorized": False,
        "failed_start_v3_resume_authorized": False,
        "failed_start_v3_work_product_reuse_authorized": False,
        "failed_start_v4_resume_authorized": False,
        "failed_start_v4_work_product_reuse_authorized": False,
        "preconsumption_startup_smoke_popen_limit_per_entry": 1,
        "startup_smoke_failure_preconsumption_pending": True,
    }
    for version, descriptor in (
        (2, v2_descriptor),
        (3, v3_descriptor),
        (4, v4_descriptor),
    ):
        for name in ("capsule", "authorization", "consumption", "terminal"):
            authorization[f"failed_start_v{version}_{name}_raw_sha256"] = descriptor[
                f"{name}_raw_sha256"
            ]
        authorization[f"failed_start_v{version}_capsule_digest"] = descriptor["capsule_digest"]
        authorization[f"failed_start_v{version}_authorization_digest"] = descriptor[
            "authorization_digest"
        ]
        authorization[f"failed_start_v{version}_consumption_record_digest"] = descriptor[
            "consumption_record_digest"
        ]
        authorization[f"failed_start_v{version}_terminal_record_digest"] = descriptor[
            "terminal_record_digest"
        ]
    authorization_path = fixture / "authorization.json"
    authorization_path.write_bytes(_canonical(authorization))

    limits = {
        "python": supervisor.RuntimeLimitV5("24", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "rust": supervisor.RuntimeLimitV5("25", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "host": supervisor.RuntimeLimitV5("26", 67_108_864, 2_097_152, 1, 2_097_152, 3_600).object(),
    }
    run_spec: dict[str, object] = {
        "schema_version": supervisor.RUN_SPEC_SCHEMA,
        "start_capsule_digest": capsule_digest,
        "start_capsule": str(capsule_path),
        "start_capsule_raw_sha256": _digest(capsule_path.read_bytes()),
        "start_authorization": str(authorization_path),
        "start_authorization_raw_sha256": _digest(authorization_path.read_bytes()),
        "start_authorization_digest": authorization_digest,
        "successor_prestart_bundle": str(bundle_path),
        "successor_prestart_bundle_raw_sha256": _digest(bundle_path.read_bytes()),
        "successor_prestart_bundle_root": bundle_root,
        "base_authoritative_artifact": str(base_path),
        "base_authoritative_artifact_raw_sha256": _digest(base_path.read_bytes()),
        "corrected_resource_envelope_root": corrected_root,
        "actor_source_manifest_roots": manifest_roots,
        "actor_source_closure_root": digest("a"),
        "source_commit": source_commit,
        "run_genesis_root": run_genesis_root,
        "prestart_admission_receipt_root": receipt_root,
        "python_image": "python@" + digest("e"),
        "rust_image": "rust@" + digest("f"),
        "rust_binary_raw_sha256": _digest(rust_binary.read_bytes()),
        "seccomp_profile_raw_sha256": _digest(seccomp.read_bytes()),
        "python_admission_raw_sha256": _digest(admission.read_bytes()),
        "actor_implementation_roots": implementation_roots,
        "python_source_root": str(run / "python-source"),
        "rust_source_root": str(run / "rust-source"),
        "host_source_root": str(run / "host-source"),
        "rust_binary": str(rust_binary),
        "python_admission_json": str(admission),
        "seccomp_profile": str(seccomp),
        "run_root": str(run),
        "checkpoint_root": str(run / "checkpoints"),
        "python_output": str(run / "python-output"),
        "python_scratch": str(run / "python-scratch"),
        "rust_output": str(run / "rust-output"),
        "rust_scratch": str(run / "rust-scratch"),
        "host_output": str(run / "host-output"),
        "host_scratch": str(run / "host-scratch"),
        "runtime_limits": limits,
        "actor_source_manifests": manifests,
    }
    for version in (2, 3, 4):
        run_spec[f"failed_start_v{version}_capsule"] = str(predecessor_paths[version, "capsule"])
        run_spec[f"failed_start_v{version}_capsule_raw_sha256"] = predecessor_raw[version, "capsule"]
        run_spec[f"failed_start_v{version}_capsule_digest"] = getattr(
            supervisor, f"FAILED_V{version}_CAPSULE_DIGEST"
        )
        run_spec[f"failed_start_v{version}_authorization"] = str(
            predecessor_paths[version, "authorization"]
        )
        run_spec[f"failed_start_v{version}_authorization_raw_sha256"] = predecessor_raw[
            version, "authorization"
        ]
        run_spec[f"failed_start_v{version}_authorization_digest"] = getattr(
            supervisor, f"FAILED_V{version}_AUTHORIZATION_DIGEST"
        )
        run_spec[f"failed_start_v{version}_consumption"] = str(
            predecessor_paths[version, "consumption"]
        )
        run_spec[f"failed_start_v{version}_consumption_raw_sha256"] = predecessor_raw[
            version, "consumption"
        ]
        run_spec[f"failed_start_v{version}_consumption_record_digest"] = getattr(
            supervisor, f"FAILED_V{version}_CONSUMPTION_RECORD_DIGEST"
        )
        run_spec[f"failed_start_v{version}_terminal"] = str(predecessor_paths[version, "terminal"])
        run_spec[f"failed_start_v{version}_terminal_raw_sha256"] = predecessor_raw[
            version, "terminal"
        ]
        run_spec[f"failed_start_v{version}_terminal_record_digest"] = getattr(
            supervisor, f"FAILED_V{version}_TERMINAL_RECORD_DIGEST"
        )
    run_spec_path = fixture / "run-spec.json"
    run_spec_path.write_bytes(_canonical(run_spec))
    return script, run_spec_path, run


def test_v5_run_spec_binds_full_v2_v3_v4_lineage_and_128_mib_endpoint_scratch(
    tmp_path: Path,
) -> None:
    payload = _run_spec_payload(tmp_path)
    spec = supervisor.load_run_spec_v5(payload)
    assert spec.start_capsule_digest != spec.failed_start_v2_capsule_digest
    assert spec.start_capsule_digest != spec.failed_start_v3_capsule_digest
    assert spec.start_capsule_digest != spec.failed_start_v4_capsule_digest
    assert spec.runtime_limits["python"].scratch_bytes == 134_217_728
    assert set(spec.authority_binding()) == supervisor._AUTHORITY_BINDING_KEYS
    assert spec.start_capsule.parent == spec.run_root

    value = json.loads(payload)
    value["runtime_limits"]["python"]["scratch_bytes"] = 4_194_304
    with pytest.raises(supervisor.Q1StartSupervisorError, match="frozen resource envelope differs"):
        supervisor.load_run_spec_v5(_canonical(value))


def test_v5_run_spec_rejects_capsule_outside_run_root_and_binds_full_authorization_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(_run_spec_payload(tmp_path))
    outside = tmp_path / "outside-capsule.json"
    outside.write_bytes(Path(value["start_capsule"]).read_bytes())
    value["start_capsule"] = str(outside)
    value["start_capsule_raw_sha256"] = _digest(outside.read_bytes())
    with pytest.raises(supervisor.Q1StartSupervisorError, match="direct run-root child"):
        supervisor.load_run_spec_v5(_canonical(value))

    second = tmp_path / "second"
    second.mkdir()
    value = json.loads(_run_spec_payload(second))
    spec = supervisor.load_run_spec_v5(_canonical(value))
    monkeypatch.setattr(supervisor, "_validate_successor_bundle", lambda current, raw: {})
    monkeypatch.setattr(supervisor, "_validate_start_capsule", lambda current, capsule, bundle: None)
    monkeypatch.setattr(supervisor, "_validate_base_authoritative_artifact", lambda value: None)
    monkeypatch.setattr(supervisor, "_validate_failed_v2_lineage", lambda *args: None)
    Path(value["start_authorization"]).write_bytes(_canonical({"tampered": True}))
    with pytest.raises(supervisor.PendingCondition, match="bytes differ"):
        supervisor._replay_successor_authority(spec)


def test_every_actor_has_two_authority_replays_before_one_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v5(_run_spec_payload(tmp_path))
    authority_calls: list[str] = []
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV5) -> dict[str, object]:
        authority_calls.append("replay")
        return {"authority": "exact"}

    def launcher(stage: str, _command: object, _spec: object) -> supervisor.ActorProcessResultV5:
        launches.append(stage)
        return supervisor.ActorProcessResultV5(0, b"", b"", True, {})

    def validate_endpoint(stage: str, _process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV5, _replayer: object):
        result = {"stage": stage}
        (current.output_for(stage) / "endpoint-result.json").write_bytes(_canonical(result))
        return result, {"runtime_evidence": {}}, len(_canonical(result))

    def validate_host(_process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV5):
        result = {"stage": "host"}
        (current.host_output / "host-result.json").write_bytes(_canonical(result))
        return result, {"runtime_evidence": {}}, len(_canonical(result))

    monkeypatch.setattr(supervisor, "_replay_successor_authority", authority)
    monkeypatch.setattr(supervisor, "_replay_stage_bindings", lambda stage, current: {"stage": stage})
    monkeypatch.setattr(supervisor, "_validate_endpoint", validate_endpoint)
    monkeypatch.setattr(supervisor, "_validate_host", validate_host)
    monkeypatch.setattr(
        supervisor,
        "_replay_checkpoint",
        lambda current, stage, command, replayer: supervisor._read_canonical_file(
            supervisor._stage_paths(current, stage)[1], f"{stage} checkpoint"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_success_outcome",
        lambda current, checkpoints: {
            "schema_version": supervisor.OUTCOME_SCHEMA,
            "status": supervisor.SUCCESS_STATUS,
            **current.binding(),
            **current.authority_binding(),
        },
    )
    outcome = supervisor.run_supervisor_v5(
        spec,
        launcher=launcher,
        preflight=lambda stage, current: None,
        endpoint_replayer=lambda stage, output, current: {},
    )
    assert outcome["status"] == supervisor.SUCCESS_STATUS
    assert launches == ["python", "rust", "host"]
    assert authority_calls == ["replay"] * 6
    for stage in supervisor.STAGES:
        intent = supervisor._read_canonical_file(supervisor._stage_paths(spec, stage)[0], f"{stage} intent")
        assert intent["successor_prestart_bundle_root"] == spec.successor_prestart_bundle_root
        assert intent["failed_start_v2_terminal_record_digest"] == spec.failed_start_v2_terminal_record_digest
        assert intent["failed_start_v3_terminal_record_digest"] == spec.failed_start_v3_terminal_record_digest
        assert intent["failed_start_v4_terminal_record_digest"] == spec.failed_start_v4_terminal_record_digest


def test_second_preintent_lineage_replay_pending_never_publishes_intent_or_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v5(_run_spec_payload(tmp_path))
    calls = 0
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV5) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise supervisor.PendingCondition("Q1_START_BOUND_FILE_PENDING", "capsule changed")
        return {"authority": "exact"}

    monkeypatch.setattr(supervisor, "_replay_successor_authority", authority)
    monkeypatch.setattr(supervisor, "_replay_stage_bindings", lambda stage, current: {})
    outcome = supervisor.run_supervisor_v5(
        spec,
        launcher=lambda stage, command, current: launches.append(stage),
        preflight=lambda stage, current: None,
        endpoint_replayer=lambda stage, output, current: {},
    )
    assert calls == 2
    assert launches == []
    assert outcome["status"] == supervisor.PENDING_STATUS
    assert outcome["next_actor_intent_absent"] is True
    assert not supervisor._stage_paths(spec, "python")[0].exists()
    assert not spec.python_output.exists()
    assert not spec.python_scratch.exists()


def test_failed_v2_terminal_must_be_exact_unknown_no_actor_lineage(tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v5(_run_spec_payload(tmp_path))
    capsule = {
        "schema_version": supervisor.FAILED_V2_CAPSULE_SCHEMA,
        "status": "Q1_START_V2_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": spec.failed_start_v2_capsule_digest,
    }
    authorization = {
        "schema_version": supervisor.FAILED_V2_AUTHORIZATION_SCHEMA,
        "status": "Q1_START_V2_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": spec.failed_start_v2_capsule_digest,
        "authorization_digest": spec.failed_start_v2_authorization_digest,
    }
    consumption = {
        "schema_version": supervisor.FAILED_V2_CONSUMPTION_SCHEMA,
        "status": "Q1_START_V2_AUTHORIZATION_CONSUMED",
        "capsule_digest": spec.failed_start_v2_capsule_digest,
        "authorization_digest": spec.failed_start_v2_authorization_digest,
        "record_digest": spec.failed_start_v2_consumption_record_digest,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    terminal = {
        "schema_version": supervisor.FAILED_V2_TERMINAL_SCHEMA,
        "status": supervisor.FAILED_V2_TERMINAL_STATUS,
        "record_digest": spec.failed_start_v2_terminal_record_digest,
        "capsule_digest": spec.failed_start_v2_capsule_digest,
        "authorization_digest": spec.failed_start_v2_authorization_digest,
        "consumption_record_digest": spec.failed_start_v2_consumption_record_digest,
        "logical_campaign_count": 1,
        "authorization_consumption_count": 1,
        "artifact_raw_sha256": None,
        "outcome": {
            "status": supervisor.FAILED_V2_TERMINAL_STATUS,
            "capsule_digest": spec.failed_start_v2_capsule_digest,
            "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 16_383,
            "q1_output_slots": [None] * 8,
            "network_accessed": False,
            "target_truth_accessed": False,
            "split_accessed": False,
            "role_evaluation_performed": False,
        },
    }
    supervisor._validate_failed_v2_lineage(
        spec, capsule, authorization, consumption, terminal
    )
    terminal["outcome"]["actor_start_count_by_actor"]["rust"] = 1
    with pytest.raises(supervisor.Q1StartSupervisorError, match="terminal no-resume state differs"):
        supervisor._validate_failed_v2_lineage(
            spec, capsule, authorization, consumption, terminal
        )


def test_failed_v3_terminal_must_be_exact_import_failure_no_actor_lineage(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v5(_run_spec_payload(tmp_path))
    capsule = {
        "schema_version": supervisor.FAILED_V3_CAPSULE_SCHEMA,
        "status": "Q1_START_V3_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": spec.failed_start_v3_capsule_digest,
    }
    authorization = {
        "schema_version": supervisor.FAILED_V3_AUTHORIZATION_SCHEMA,
        "status": "Q1_START_V3_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": spec.failed_start_v3_capsule_digest,
        "authorization_digest": spec.failed_start_v3_authorization_digest,
    }
    consumption = {
        "schema_version": supervisor.FAILED_V3_CONSUMPTION_SCHEMA,
        "status": "Q1_START_V3_AUTHORIZATION_CONSUMED",
        "capsule_digest": spec.failed_start_v3_capsule_digest,
        "authorization_digest": spec.failed_start_v3_authorization_digest,
        "record_digest": spec.failed_start_v3_consumption_record_digest,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    terminal = {
        "schema_version": supervisor.FAILED_V3_TERMINAL_SCHEMA,
        "status": supervisor.FAILED_V3_TERMINAL_STATUS,
        "record_digest": spec.failed_start_v3_terminal_record_digest,
        "capsule_digest": spec.failed_start_v3_capsule_digest,
        "authorization_digest": spec.failed_start_v3_authorization_digest,
        "consumption_record_digest": spec.failed_start_v3_consumption_record_digest,
        "logical_campaign_count": 1,
        "authorization_consumption_count": 1,
        "artifact_raw_sha256": None,
        "outcome": {
            "status": supervisor.FAILED_V3_TERMINAL_STATUS,
            "capsule_digest": spec.failed_start_v3_capsule_digest,
            "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
            "failure_code": "REJECT_Q1_START_SUCCESSOR_BUNDLE",
            "failure_detail": (
                "validator import failed: AttributeError: module 'importlib' has no attribute 'util'"
            ),
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 16_383,
            "q1_output_slots": [None] * 8,
            "network_accessed": False,
            "target_truth_accessed": False,
            "split_accessed": False,
            "role_evaluation_performed": False,
        },
    }
    supervisor._validate_failed_v3_lineage(
        spec, capsule, authorization, consumption, terminal
    )
    terminal["outcome"]["failure_detail"] = "different failure"
    with pytest.raises(supervisor.Q1StartSupervisorError, match="terminal no-resume state differs"):
        supervisor._validate_failed_v3_lineage(
            spec, capsule, authorization, consumption, terminal
        )


def test_failed_v4_terminal_must_be_exact_python_once_lineage(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v5(_run_spec_payload(tmp_path))
    capsule, authorization, consumption, terminal = _failed_lineage_fixture(4)
    supervisor._validate_failed_v4_lineage(
        spec, capsule, authorization, consumption, terminal
    )
    terminal["outcome"]["actor_start_count_by_actor"]["python"] = 0
    with pytest.raises(
        supervisor.Q1StartSupervisorError,
        match="terminal no-resume state differs",
    ):
        supervisor._validate_failed_v4_lineage(
            spec, capsule, authorization, consumption, terminal
        )


def test_startup_smoke_rejects_v4_descriptor_with_nonempty_completed_prefix(
    tmp_path: Path,
) -> None:
    script, run_spec_path, run_root = _isolated_smoke_fixture(tmp_path)
    run_spec = json.loads(run_spec_path.read_bytes())
    capsule_path = Path(run_spec["start_capsule"])
    capsule = json.loads(capsule_path.read_bytes())
    capsule["failed_start_v4_predecessor"]["completed_prefix"] = ["python"]
    capsule_path.write_bytes(_canonical(capsule))
    run_spec["start_capsule_raw_sha256"] = _digest(capsule_path.read_bytes())
    run_spec_path.write_bytes(_canonical(run_spec))
    before = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*"))
    process = subprocess.run(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            str(script),
            "--startup-smoke",
            "--run-spec",
            str(run_spec_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 1
    assert result["error_code"] == "REJECT_Q1_START_CAPSULE"
    assert result["campaign_supervisor_popen_count"] == 0
    assert result["actor_popen_count"] == 0
    assert result["checkpoint_or_mutable_roots_created"] is False
    assert sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*")) == before


def test_authority_startup_smoke_is_zero_popen_and_zero_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_spec_payload(tmp_path)
    spec = supervisor.load_run_spec_v5(payload)
    before = sorted(str(path.relative_to(spec.run_root)) for path in spec.run_root.rglob("*"))
    calls: list[str] = []
    monkeypatch.setattr(
        supervisor,
        "_replay_successor_authority",
        lambda current: {"capsule": current.start_capsule_digest},
    )
    monkeypatch.setattr(
        supervisor,
        "_replay_stage_bindings",
        lambda stage, current: {"stage": stage, "source": str(current.source_for(stage))},
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *args, **kwargs: calls.append("subprocess") or pytest.fail("subprocess called"),
    )
    result = supervisor.authority_startup_smoke_v5(payload)
    after = sorted(str(path.relative_to(spec.run_root)) for path in spec.run_root.rglob("*"))
    assert result["status"] == supervisor.AUTHORITY_SMOKE_STATUS
    assert result["actor_popen_count"] == 0
    assert result["campaign_supervisor_popen_count"] == 0
    assert result["checkpoint_or_mutable_roots_created"] is False
    assert result["docker_invoked"] is False
    assert result["failed_start_v3_terminal_record_digest"] == spec.failed_start_v3_terminal_record_digest
    assert result["failed_start_v4_terminal_record_digest"] == spec.failed_start_v4_terminal_record_digest
    assert before == after
    assert calls == []
    assert not spec.checkpoint_root.exists()


def test_startup_smoke_cli_runs_in_clean_isolated_python_without_actor_or_mutation(
    tmp_path: Path,
) -> None:
    script, run_spec, run_root = _isolated_smoke_fixture(tmp_path)
    before = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*"))
    process = subprocess.run(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            str(script),
            "--startup-smoke",
            "--run-spec",
            str(run_spec),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", errors="replace")
    result = json.loads(process.stdout)
    assert process.stdout == _canonical(result)
    assert process.stderr == b""
    assert result["schema_version"] == supervisor.AUTHORITY_SMOKE_SCHEMA
    assert result["status"] == supervisor.AUTHORITY_SMOKE_STATUS
    assert result["campaign_supervisor_popen_count"] == 0
    assert result["actor_popen_count"] == 0
    assert result["docker_invoked"] is False
    assert result["checkpoint_or_mutable_roots_created"] is False
    assert result["failed_start_v3_terminal_record_digest"] == supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST
    assert result["failed_start_v4_terminal_record_digest"] == supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST
    after = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*"))
    assert after == before
    assert not (run_root / "checkpoints").exists()


def test_startup_smoke_cli_failure_is_nonzero_complete_canonical_diagnostic(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-run-spec.json"
    process = subprocess.run(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            str(ROOT / "tools" / "phase3_q1_start_supervisor_v5.py"),
            "--startup-smoke",
            "--run-spec",
            str(missing),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 1
    assert process.stdout == _canonical(result)
    assert process.stderr == b""
    assert result == {
        "schema_version": supervisor.AUTHORITY_SMOKE_SCHEMA,
        "status": "Q1_START_V5_AUTHORITY_STARTUP_SMOKE_FAILED",
        "error_code": "REJECT_Q1_START_RUN_SPEC_IO",
        "error_detail": f"[Errno 2] No such file or directory: '{missing}'",
        "campaign_supervisor_popen_count": 0,
        "actor_popen_count": 0,
        "checkpoint_or_mutable_roots_created": False,
        "docker_invoked": False,
    }


def test_process_diagnostic_retains_bounded_head_and_tail() -> None:
    payload = b"head:" + b"a" * supervisor.PROCESS_DIAGNOSTIC_CAPTURE_BYTES + b":tail"
    captured = supervisor._captured_process_stream(payload)
    import base64

    head = base64.b64decode(captured["head_base64"])
    tail = base64.b64decode(captured["tail_base64"])
    assert head.startswith(b"head:")
    assert tail.endswith(b":tail")
    assert captured["captured_bytes"] == supervisor.PROCESS_DIAGNOSTIC_CAPTURE_BYTES
    assert captured["truncated"] is True


def test_private_bundle_loader_replays_fixture_without_executing_package_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "hegel_machine"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('package init must never execute')\n",
        encoding="utf-8",
    )
    (package / "strict_cbor_v1.py").write_text(
        "TOKEN = 'strict-cbor-loaded'\n",
        encoding="utf-8",
    )
    (package / "phase3_q1_prestart_successor_candidate_v5.py").write_text(
        "from .strict_cbor_v1 import TOKEN\n"
        "import json\n"
        "def replay_q1_prestart_successor_candidate_v5(payload):\n"
        "    value = json.loads(payload.decode('ascii'))\n"
        "    assert TOKEN == 'strict-cbor-loaded'\n"
        "    return value\n"
        "def successor_bundle_root_v5(value): return value['bundle_root']\n"
        "def actor_source_manifest_root_v5(actor, rows): return rows[0]['raw_sha256']\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "PACKAGE_ROOT", package)
    monkeypatch.delitem(sys.modules, "hegel_machine", raising=False)
    module = supervisor._load_successor_bundle_core_v5()
    bundle = _canonical({"bundle_root": "sha256:" + "a" * 64})
    replayed = module.replay_q1_prestart_successor_candidate_v5(bundle)
    assert module.successor_bundle_root_v5(replayed) == "sha256:" + "a" * 64
    assert "hegel_machine" not in sys.modules

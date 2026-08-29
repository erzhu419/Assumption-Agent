from __future__ import annotations

import ast
from copy import deepcopy
import dis
from hashlib import sha256
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Mapping, NoReturn, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CORE = importlib.import_module(
    "hegel_machine.phase3_q05b_shadow_retry_campaign_v2"
)
TRAJECTORY = importlib.import_module("hegel_machine.phase3_q05b_trajectory_v2")

TOOL_PATH = ROOT / "tools/phase3_q05b_shadow_retry_campaign_v2.py"
CORE_TEST_KAT_PATH = (
    ROOT / "tests/test_phase3_q05b_shadow_retry_campaign_v2_core.py"
)
SPEC = importlib.util.spec_from_file_location(
    "phase3_q05b_shadow_retry_campaign_v2_tool_test",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)

_REAL_REQUIRE_SINGLE_THREADED_SPAWN = (
    TOOL._require_single_threaded_spawn_controller_v2
)
_REAL_WAIT_JOINED_ADMISSION_TASKS = (
    TOOL._wait_for_joined_admission_tasks_to_quiesce_v2r17
)
_REAL_LOAD_PREAUTHORIZATION_LIVE_GAUNTLET = (
    TOOL._load_preauthorization_live_gauntlet_receipt_v2
)


@pytest.fixture(autouse=True)
def _permit_pytest_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production CLI is single-threaded; this pytest host is not."""

    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    monkeypatch.setattr(
        TOOL,
        "_wait_for_joined_admission_tasks_to_quiesce_v2r17",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        TOOL,
        "_load_preauthorization_live_gauntlet_receipt_v2",
        lambda contract, *, controller_git: (
            _gauntlet_receipt_binding_stub(contract, controller_git)
        ),
    )


_STAGES = (
    "FREEZE_SOURCE_IMAGES_AND_COMMAND_REGISTRY",
    "MATERIALIZE_AND_SEAL_THREE_SOURCE_SNAPSHOTS_AND_CARGO",
    "OFFLINE_RUST_TEST_AND_RELEASE_BUILD_AND_SEAL_BINARY",
    "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
    "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
    "RUN_HELD_TRUSTED_HOST_SEMANTIC_REPLAY",
    "CLOSE_HOST_RESOURCE_ISOLATION_NEGATIVE_VECTORS_AND_TOCTOU",
    "BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE",
    "ADD_PREDICATE20_AND_BUILD_CANONICAL_ARTIFACT_IN_MEMORY",
    "STRICTLY_REPLAY_COMPLETE_ARTIFACT_BEFORE_PUBLICATION",
)

_SAMPLE_FIELDS = (
    "role_id_or_null",
    "sample_ordinal_or_null",
    "previous_finish_ns_or_null",
    "scheduled_start_ns_or_null",
    "actual_start_ns_or_null",
    "finish_ns_or_null",
    "collection_duration_ns_or_null",
    "scheduler_lateness_ns_or_null",
    "target_wait_ns_or_null",
    "gap_ns_or_null",
    "maximum_gap_ns_or_null",
    "gap_exceeded_or_null",
)


def _raw(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _tagged(label: str) -> str:
    return "sha256:" + _raw(label)


def _dependency_preflight_receipt_stub(
    label: str = "dependency-preflight",
) -> dict[str, object]:
    contract = _contract()
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    policy = execution["python310_dependency_preflight_gate"]
    assert type(policy) is dict
    cargo_raw = (ROOT / str(policy["cargo_lock_relative_path"])).read_bytes()
    tomli = importlib.import_module("hegel_machine._vendor.tomli")
    parsed = tomli.loads(cargo_raw.decode("utf-8"))
    rows = [
        [row["name"], row["version"], row["source"], row["checksum"]]
        for row in parsed["package"]
        if row.get("source", "").startswith("registry+")
    ]
    receipt: dict[str, object] = {
        "schema_version": (
            TOOL._PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_SCHEMA_V1
        ),
        "status": "PASS",
        "target_commit": target["target_supervisor_commit"],
        "target_source_binding_digest": target[
            "target_source_binding_digest"
        ],
        "controller_commit": "de" * 20,
        "controller_changed_paths_from_target_in_exact_order": list(
            TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        ),
        "python_executable": {
            "path": policy["interpreter_path"],
            "raw_sha256": _raw(f"{label}-python"),
        },
        "argv_prefix_before_program": deepcopy(
            policy["argv_prefix_before_program"]
        ),
        "program_raw_sha256": policy["program_raw_sha256"],
        "runtime_environment_digest": TOOL._domain_digest(
            policy["runtime_environment_digest_domain"],
            TOOL._runtime_environment(contract),
        ),
        "supervisor_source": {
            "relative_path": policy["supervisor_relative_path"],
            "raw_sha256": target["required_raw_sha256_by_path"][
                policy["supervisor_relative_path"]
            ],
        },
        "artifact_source": {
            "relative_path": policy["artifact_relative_path"],
            "raw_sha256": target["required_raw_sha256_by_path"][
                policy["artifact_relative_path"]
            ],
        },
        "artifact_module_name": "hegel_machine.phase3_q05b_actual_artifact_v2",
        "artifact_schema_version": policy["artifact_schema_version"],
        "resource_transcript_ref_key": policy["resource_transcript_ref_key"],
        "held_final_sample_ref_key": policy["held_final_sample_ref_key"],
        "resource_reference_kat": {
            "resource_table_row_count": policy[
                "resource_reference_table_row_count"
            ],
            "resource_transcript_index_count": 3,
            "held_final_sample_index_count": 3,
            "projected": deepcopy(policy["resource_reference_kat_projected"]),
            "round_trip_exact": True,
            "stream_root_domain": policy["resource_stream_root_kat_domain"],
            "stream_root": policy["resource_stream_root_kat"],
            "canonical_root_matches_streaming": True,
        },
        "artifact_module_max_canonical_bytes": execution[
            "artifact_max_canonical_bytes"
        ],
        "supervisor_tool_max_canonical_bytes": execution[
            "artifact_max_canonical_bytes"
        ],
        "host_replay_module_name": "hegel_machine.phase3_q05b_host_replay_v1",
        "vendored_tomli_module_name": policy["vendored_tomli_module_name"],
        "tomli_top_level_spec_or_null": None,
        "top_level_tomli_in_sys_modules": False,
        "python_flags": {
            "isolated": 1,
            "no_site": 1,
            "ignore_environment": 1,
            "dont_write_bytecode": True,
            "utf8_mode": 1,
            "pycache_prefix": policy["pycache_prefix"],
            "pycache_prefix_absent_before_import": True,
            "pycache_prefix_absent_after_import": True,
        },
        "vendor_closure_rows": deepcopy(
            policy["vendor_closure_rows_in_exact_order"]
        ),
        "vendor_closure_digest": policy["vendor_closure_digest"],
        "loaded_vendored_modules": deepcopy(
            policy["loaded_vendored_modules_in_exact_order"]
        ),
        "host_source_allowlist_count": policy["host_source_allowlist_count"],
        "host_source_allowlist_canonical_lf_sha256": policy[
            "host_source_allowlist_canonical_lf_sha256"
        ],
        "cargo_lock_git_object": {
            "object_spec": (
                f"{target['target_supervisor_commit']}:"
                f"{execution['clone_source_project_relative_path']}/"
                f"{policy['cargo_lock_relative_path']}"
            ),
            "project_relative_path": policy["cargo_lock_relative_path"],
            "byte_count": policy["cargo_lock_byte_count"],
            "raw_sha256": policy["cargo_lock_raw_sha256"],
            "matches_current_controller_worktree": True,
        },
        "cargo_lock_parse": {
            "relative_path": policy["cargo_lock_relative_path"],
            "byte_count": len(cargo_raw),
            "raw_sha256": sha256(cargo_raw).hexdigest(),
            "format": parsed["version"],
            "package_count": len(parsed["package"]),
            "registry_package_count": len(rows),
            "registry_package_rows": rows,
            "registry_projection_digest": TOOL._domain_digest(
                policy["cargo_lock_registry_projection_domain"],
                rows,
            ),
        },
        "child_receipt": None,
        "process_facts": {
            "return_code": 0,
            "timed_out": False,
            "stdout_overflow": False,
            "stderr_overflow": False,
            "child_reaped": True,
            "process_group_absent": True,
            "wrapper_resources_closed": True,
        },
        "stdout": None,
        "stderr": {
            "byte_count": 0,
            "raw_sha256": sha256(b"").hexdigest(),
            "empty": True,
        },
        "receipt_digest": None,
    }
    child_receipt = {
        "schema_version": (
            TOOL._PYTHON310_DEPENDENCY_PREFLIGHT_CHILD_SCHEMA_V1
        ),
        "status": "PASS",
        "python_flags": deepcopy(receipt["python_flags"]),
        "tomli_top_level_spec_or_null": receipt[
            "tomli_top_level_spec_or_null"
        ],
        "artifact_module_name": receipt["artifact_module_name"],
        "artifact_schema_version": receipt["artifact_schema_version"],
        "resource_transcript_ref_key": receipt[
            "resource_transcript_ref_key"
        ],
        "held_final_sample_ref_key": receipt[
            "held_final_sample_ref_key"
        ],
        "resource_reference_kat": deepcopy(
            receipt["resource_reference_kat"]
        ),
        "artifact_module_max_canonical_bytes": receipt[
            "artifact_module_max_canonical_bytes"
        ],
        "supervisor_tool_max_canonical_bytes": receipt[
            "supervisor_tool_max_canonical_bytes"
        ],
        "vendored_tomli_module_name": receipt["vendored_tomli_module_name"],
        "top_level_tomli_in_sys_modules": receipt[
            "top_level_tomli_in_sys_modules"
        ],
        "host_replay_module_name": receipt["host_replay_module_name"],
        "host_source_allowlist_count": receipt[
            "host_source_allowlist_count"
        ],
        "host_source_allowlist_canonical_lf_sha256": receipt[
            "host_source_allowlist_canonical_lf_sha256"
        ],
        "vendor_closure_rows": deepcopy(receipt["vendor_closure_rows"]),
        "vendor_closure_digest": receipt["vendor_closure_digest"],
        "loaded_vendored_modules": deepcopy(
            receipt["loaded_vendored_modules"]
        ),
        "cargo_lock": deepcopy(receipt["cargo_lock_parse"]),
    }
    stdout = TOOL._canonical_json_bytes(child_receipt)
    receipt["child_receipt"] = child_receipt
    receipt["stdout"] = {
        "byte_count": len(stdout),
        "raw_sha256": sha256(stdout).hexdigest(),
        "canonical_single_lf": True,
    }
    unsigned = deepcopy(receipt)
    unsigned.pop("receipt_digest")
    receipt["receipt_digest"] = TOOL._domain_digest(
        TOOL._PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_DOMAIN_V1,
        unsigned,
    )
    return receipt


def _capsule_stub(
    label: str = "capsule",
    *,
    contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    dependency = _dependency_preflight_receipt_stub(
        f"{label}-dependency-preflight"
    )
    contract = _contract() if contract is None else contract
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    decision_audit = contract["decision_audit_policy"]
    assert type(decision_audit) is dict
    prior_v2r18 = TOOL._prior_v2r18_historical_evidence_v2(contract)
    prior_v2r19 = TOOL._prior_v2r19_historical_evidence_v2(contract)
    controller_git = {
        "commit": dependency["controller_commit"],
        "parent": target["target_supervisor_commit"],
        "parent_count": 1,
        "clean": True,
        "changed_paths_from_target_in_exact_order": deepcopy(
            dependency[
                "controller_changed_paths_from_target_in_exact_order"
            ]
        ),
    }
    gauntlet = _gauntlet_receipt_binding_stub(contract, controller_git)
    return {
        "capsule_digest": _tagged(label),
        "controller_git": controller_git,
        "target_git": {
            "commit": dependency["target_commit"],
            "target_source_binding_digest": dependency[
                "target_source_binding_digest"
            ],
            "required_raw_sha256_by_path": deepcopy(
                target["required_raw_sha256_by_path"]
            ),
            "cargo_lock_git_object": deepcopy(
                dependency["cargo_lock_git_object"]
            ),
            "local_object_closure_fsck_passed": True,
        },
        "executable_identities": {
            "python": deepcopy(dependency["python_executable"]),
        },
        "python310_dependency_preflight_receipt": dependency,
        "python310_dependency_preflight_receipt_digest": dependency[
            "receipt_digest"
        ],
        "preauthorization_live_gauntlet_receipt": gauntlet["receipt"],
        "preauthorization_live_gauntlet_receipt_digest": gauntlet[
            "receipt_digest"
        ],
        "preauthorization_live_gauntlet_receipt_raw_sha256": gauntlet[
            "raw_sha256"
        ],
        "preauthorization_live_gauntlet_receipt_byte_count": gauntlet[
            "byte_count"
        ],
        "preauthorization_live_gauntlet_root_identity": gauntlet[
            "root_identity"
        ],
        "preauthorization_live_gauntlet_record_identity": gauntlet[
            "record_identity"
        ],
        "decision_audit_policy": deepcopy(decision_audit),
        "decision_audit_policy_digest": decision_audit["policy_digest"],
        "decision_audit_core_schema_version": decision_audit[
            "core_audit_schema_version"
        ],
        "prior_v2r18_historical_evidence": prior_v2r18,
        "prior_v2r19_historical_evidence": prior_v2r19,
    }


def _contract() -> dict[str, object]:
    raw = TOOL.CONFIG_PATH.read_bytes()
    value = TOOL._pretty_json(raw, "campaign test contract")
    assert type(value) is dict
    contract = deepcopy(value)
    final_status = (
        "V2R20_FINAL_BOUND_AWAITING_EXACT_OWNER_AUTHORIZATION_NOT_EXECUTED"
    )
    contract["status"] = final_status
    contract["contract_digest"] = _tagged("contract")
    authority = contract["authority_boundary"]
    assert type(authority) is dict
    authority["preflight_authorized"] = True
    binding = contract["campaign_source_binding"]
    assert type(binding) is dict
    binding["status"] = "CAMPAIGN_SOURCE_BOUND"
    binding["source_binding_digest"] = _tagged("campaign-source")
    binding["core_raw_sha256"] = sha256(
        (SRC / "hegel_machine/phase3_q05b_shadow_retry_campaign_v2.py")
        .read_bytes()
    ).hexdigest()
    binding["core_test_raw_sha256"] = _raw("core-test-raw")
    binding["tool_raw_sha256"] = _raw("tool-raw")
    binding["tool_test_raw_sha256"] = _raw("tool-test-raw")
    binding["static_surface_sha256"] = _raw("static-surface")
    binding["document_raw_sha256"] = _raw("document-raw")
    binding["v1_safety_kernel_raw_sha256"] = _raw("v1-safety-kernel-raw")
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    target["status"] = "TARGET_COMMIT_A_BOUND"
    target["target_supervisor_commit"] = "12" * 20
    target["target_supervisor_tree"] = "34" * 20
    target["target_supervisor_parent_commit"] = "56" * 20
    target["target_supervisor_commit_object_raw_sha256"] = _raw(
        "target-commit-object"
    )
    required = target["required_raw_sha256_by_path"]
    assert type(required) is dict
    for index, relative in enumerate(required, start=1):
        required[relative] = _raw(f"target-{index}")
    target["target_config_static_policy_root"] = _raw("target-static-policy")
    target["target_source_binding_digest"] = _tagged("target-source-binding")
    target["target_contract_digest"] = _tagged("target-contract")
    retry = contract["retry_policy"]
    success = contract["success_policy"]
    forbidden = contract["forbidden_scope"]
    assert type(retry) is dict and type(success) is dict
    assert type(forbidden) is dict
    retry["retry_predicate_digest"] = _tagged("retry")
    success["success_predicate_digest"] = _tagged("success")
    forbidden["forbidden_scope_digest"] = _tagged("forbidden")
    decision_audit = contract["decision_audit_policy"]
    assert type(decision_audit) is dict
    unsigned_decision_audit = deepcopy(decision_audit)
    unsigned_decision_audit.pop("policy_digest")
    decision_audit["policy_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["decision_audit_policy"],
        unsigned_decision_audit,
    )
    ledger = contract["ledger_policy"]
    assert type(ledger) is dict
    ledger["ledger_state_registry_digest"] = _tagged("ledger-registry")
    execution = contract["execution_policy"]
    assert type(execution) is dict
    labels = execution["target_docker_label_policy"]
    assert type(labels) is dict
    labels["source_commit_label_value"] = target["target_supervisor_commit"]
    labels["pre_spawn_and_post_reap_readonly_absence_filter"] = (
        "label=org.hegel.q05b.source_commit=" + target["target_supervisor_commit"]
    )
    implementation = contract["implementation_state"]
    assert type(implementation) is dict
    implementation["target_commit_a_bound"] = True
    implementation["campaign_source_bound"] = True
    implementation["status"] = final_status
    return contract


def _gauntlet_receipt_binding_stub(
    contract: Mapping[str, object],
    controller_git: Mapping[str, object],
) -> dict[str, object]:
    policy = TOOL._preauthorization_live_gauntlet_policy_v2(contract)
    root_identity = {
        "dev": 101,
        "ino": 202,
        "mount_id": 303,
        "mode": stat.S_IFDIR | 0o700,
        "uid": os.geteuid(),
        "nlink": 2,
    }
    receipt: dict[str, object] = {
        "schema_version": TOOL._PREAUTHORIZATION_LIVE_GAUNTLET_RECEIPT_SCHEMA_V2,
        "status": "PASS",
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "controller_git": deepcopy(dict(controller_git)),
        "target_binding": TOOL._preauthorization_live_gauntlet_target_binding_v2(
            contract
        ),
        "external_root": policy["stable_external_root"],
        "external_root_identity": deepcopy(root_identity),
        "receipt_path": (
            f"{policy['stable_external_root']}/{policy['receipt_basename']}"
        ),
        "scheduling_policy": TOOL._scheduling_admission_policy_v2(contract),
        "scheduling_admission": _admitted_scheduling_evidence(contract),
        "formal_roots_in_exact_order": (
            TOOL._preauthorization_live_gauntlet_formal_roots_v2(contract)
        ),
        "formal_roots_absent_before_gate": True,
        "formal_roots_absent_after_gate": True,
        "scratch_path": (
            f"{policy['stable_external_root']}/{policy['scratch_basename']}"
        ),
        "scratch_cleanup_passed": True,
        "scratch_root_fsync_after_removal": True,
        "authorization_lookup_count": 0,
        "authorization_publication_count": 0,
        "authorization_consumption_count": 0,
        "ledger_creation_or_write_count": 0,
        "formal_slot_root_creation_or_write_count": 0,
        "campaign_execution_count": 0,
        "supervisor_invocation_count": 0,
        "docker_daemon_contact_count": 0,
        "network_contact_count": 0,
        "same_real_scheduling_gate_used": True,
        "final_clean_controller_commit_bound": True,
        "unique_controller_parent_equals_target_commit": True,
        "exact_six_path_diff_bound": True,
        "full_scheduling_policy_and_evidence_bound": True,
        "claim_ceiling": policy["claim_ceiling"],
        "receipt_digest": None,
    }
    unsigned = deepcopy(receipt)
    unsigned.pop("receipt_digest")
    receipt["receipt_digest"] = TOOL._domain_digest(
        TOOL._PREAUTHORIZATION_LIVE_GAUNTLET_RECEIPT_DOMAIN_V2,
        unsigned,
    )
    raw = TOOL._canonical_json_bytes(receipt)
    return {
        "receipt": receipt,
        "receipt_digest": receipt["receipt_digest"],
        "raw_sha256": sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "root_identity": root_identity,
        "record_identity": {
            "mode": stat.S_IFREG | 0o444,
            "uid": os.geteuid(),
            "nlink": 1,
            "size": len(raw),
        },
    }


def _rehash_gauntlet_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    value = deepcopy(dict(receipt))
    value.pop("receipt_digest", None)
    value["receipt_digest"] = TOOL._domain_digest(
        TOOL._PREAUTHORIZATION_LIVE_GAUNTLET_RECEIPT_DOMAIN_V2,
        value,
    )
    return value


def _dependency_preflight_inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    raw = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "dependency preflight KAT contract",
    )
    assert type(raw) is dict
    contract = deepcopy(raw)
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    policy = TOOL._python310_dependency_preflight_policy_v1(contract)
    target_git = {
        "commit": target["target_supervisor_commit"],
        "target_source_binding_digest": target[
            "target_source_binding_digest"
        ],
        "required_raw_sha256_by_path": deepcopy(
            target["required_raw_sha256_by_path"]
        ),
        "cargo_lock_git_object": {
            "object_spec": (
                f"{target['target_supervisor_commit']}:"
                f"{execution['clone_source_project_relative_path']}/"
                f"{policy['cargo_lock_relative_path']}"
            ),
            "project_relative_path": policy["cargo_lock_relative_path"],
            "byte_count": policy["cargo_lock_byte_count"],
            "raw_sha256": policy["cargo_lock_raw_sha256"],
            "matches_current_controller_worktree": True,
        },
    }
    controller_git = {
        "commit": "de" * 20,
        "parent": target["target_supervisor_commit"],
        "parent_count": 1,
        "clean": True,
        "changed_paths_from_target_in_exact_order": list(
            TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        ),
    }
    python_identity = TOOL._executable_identity(
        "/usr/bin/python3.10",
        "Python dependency KAT",
    )
    return contract, target_git, controller_git, python_identity


class _AdmissionClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1_000_000_000)


def _sustained_scheduling_evidence(
    policy: Mapping[str, object],
    slot_id: str,
    window: int,
    *,
    start_ns: int,
    durable_by_deadline: int | None = None,
    sampler_anchor_maximum_ns: int = 25_000_000,
) -> dict[str, object]:
    prefill = int(policy["sustained_prefill_retained_records"])
    timed = int(policy["sustained_timed_records"])
    duration = int(policy["sustained_measurement_duration_ns"])
    epoch_duration = int(policy["sustained_epoch_duration_ns"])
    records_per_cycle = int(policy["sustained_records_per_cycle"])
    cycle_count = int(policy["sustained_cycle_count"])
    cycle_interval = int(policy["sampling_target_wait_ns"])
    by_deadline = (
        timed if durable_by_deadline is None else durable_by_deadline
    )
    assert 0 <= by_deadline <= timed
    epoch_count = duration // epoch_duration
    counts = [by_deadline // epoch_count] * epoch_count
    for ordinal in range(by_deadline % epoch_count):
        counts[ordinal] += 1
    anchor_counts = [0] * epoch_count
    for cycle_index in range(cycle_count):
        anchor_counts[min(
            (cycle_index * cycle_interval) // epoch_duration,
            epoch_count - 1,
        )] += 2
    checkpoints = {prefill, prefill + timed}
    cumulative = 0
    cumulative_rows: list[int] = []
    for count in counts:
        cumulative += count
        cumulative_rows.append(prefill + cumulative)
        checkpoints.add(prefill + cumulative)
    chains = TOOL._scheduling_sustained_chain_checkpoints_v6(
        slot_id, window, prefill + timed, prefill, checkpoints
    )
    cumulative = 0
    rows: list[dict[str, object]] = []
    for epoch_ordinal, (count, anchor_count) in enumerate(
        zip(counts, anchor_counts), start=1
    ):
        previous = cumulative
        cumulative += count
        rows.append({
            "epoch_ordinal": epoch_ordinal,
            "start_offset_ns": (epoch_ordinal - 1) * epoch_duration,
            "finish_offset_ns": epoch_ordinal * epoch_duration,
            "records_durable": count,
            "first_global_ordinal_or_null": (
                prefill + previous + 1 if count else None
            ),
            "last_global_ordinal_or_null": (
                prefill + cumulative if count else None
            ),
            "total_publication_duration_ns": count,
            "minimum_publication_duration_ns_or_null": 1 if count else None,
            "maximum_publication_duration_ns_or_null": 1 if count else None,
            "cumulative_timed_records_durable": cumulative,
            "sampler_anchor_count": anchor_count,
            "maximum_sampler_release_to_post_enqueue_anchor_ns_or_null": (
                sampler_anchor_maximum_ns if anchor_count else None
            ),
            "chain_digest": chains[prefill + cumulative],
        })
    late = timed - by_deadline
    deadline = start_ns + duration
    finish = deadline + (1 if late else 0)
    passed = (
        late == 0
        and sampler_anchor_maximum_ns
        <= int(policy[
            "maximum_sampler_deadline_to_post_enqueue_anchor_ns"
        ])
    )
    result: dict[str, object] = {
        "schema_version": TOOL._SCHEDULING_SUSTAINED_EVIDENCE_SCHEMA_V7,
        "slot_id": slot_id,
        "window_ordinal": window,
        "clock_id": "CLOCK_MONOTONIC",
        "measurement_start_ns": start_ns,
        "measurement_deadline_ns": deadline,
        "measurement_finish_ns": finish,
        "measurement_duration_ns": duration,
        "offered_rate_records_per_second": int(
            policy["sustained_offered_rate_records_per_second"]
        ),
        "cycle_interval_ns": cycle_interval,
        "records_per_cycle": records_per_cycle,
        "cycle_count": cycle_count,
        "prefill_records_durable": prefill,
        "prefill_publication_duration_ns_total": prefill,
        "prefill_completed_before_measurement": True,
        "timed_records_enqueued": timed,
        "timed_records_durable_by_deadline": by_deadline,
        "timed_records_durable_after_deadline": late,
        "records_durable_total": prefill + timed,
        "retained_records_at_measurement_end": prefill + timed,
        "queue_start_depth_records": records_per_cycle,
        "queue_end_depth_records": 0,
        "queue_high_watermark_records": records_per_cycle,
        "timed_offer_terminal_disposition": (
            TOOL._SCHEDULING_SUSTAINED_ENQUEUED_V7
        ),
        "rejected_record_count": 0,
        "unoffered_after_failfast_record_count": 0,
        "scheduled_shortfall_record_count": 0,
        "full_measurement_duration_observed": True,
        "first_queue_capacity_rejection_or_null": None,
        "duplicate_record_count": 0,
        "reordered_record_count": 0,
        "last_timed_durable_finish_ns_or_null": finish,
        "terminal_drain_ns": 1 if late else 0,
        "sampler_anchor_count": sum(anchor_counts),
        "maximum_sampler_release_to_post_enqueue_anchor_ns": (
            sampler_anchor_maximum_ns
        ),
        "epoch_aggregates_in_strict_order": rows,
        "prefill_chain_digest": chains[prefill],
        "final_chain_digest": chains[prefill + timed],
        "cleanup_removed_records": prefill + timed,
        "cleanup_absence_passed": True,
        "passed": passed,
    }
    assert TOOL._scheduling_sustained_evidence_exact_v6(
        policy, slot_id, window, result
    )
    return result


def _admitted_scheduling_evidence(
    contract: Mapping[str, object],
    slot_id: str = "SLOT_01",
) -> dict[str, object]:
    policy = TOOL._scheduling_admission_policy_v2(contract)
    full_bursts = [
        _admitted_scheduling_burst(
            policy, slot_id, 1, ordinal,
            start_ns=30_000_000_000 + (ordinal - 1) * 1_000_000_000,
        )
        for ordinal in range(1, 17)
    ]
    bursts = [
        TOOL._compact_scheduling_admission_burst_v6(
            policy, slot_id, 1, row
        )
        for row in full_bursts
    ]
    evidence: dict[str, object] = {
        "schema_version": TOOL._SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V7,
        "slot_id": slot_id,
        "policy": deepcopy(policy),
        "windows_in_strict_order": [
            {
                "window_ordinal": 1,
                "quiet_start_ns": 0,
                "quiet_finish_ns": 30_000_000_000,
                "quiet_duration_ns": 30_000_000_000,
                "bursts_in_strict_order": bursts,
                "sustained_publication_or_null": _sustained_scheduling_evidence(
                    policy,
                    slot_id,
                    1,
                    start_ns=int(bursts[-1]["burst_finish_ns"]),
                ),
                "maximum_sampler_deadline_to_post_enqueue_anchor_ns": 25_000_000,
                "maximum_projected_gap_ns": int(
                    policy["maximum_projected_gap_ns"]
                ),
                "maximum_burst_drain_ns": 200_000_000,
                "maximum_queue_high_watermark_records": 7,
                "passed": True,
            }
        ],
        "selected_window_ordinal": 1,
        "passed": True,
    }
    assert TOOL._scheduling_admission_evidence_exact_v2(contract, evidence)
    return evidence


def test_preauthorization_live_gauntlet_receipt_rejects_rehashed_mutations(
) -> None:
    contract = _contract()
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    controller_git = {
        "commit": "ab" * 20,
        "tree": "cd" * 20,
        "parent": target["target_supervisor_commit"],
        "parent_count": 1,
        "clean": True,
        "changed_paths_from_target_in_exact_order": list(
            TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        ),
    }
    binding = _gauntlet_receipt_binding_stub(contract, controller_git)
    receipt = binding["receipt"]
    assert type(receipt) is dict
    assert TOOL._validate_preauthorization_live_gauntlet_receipt_v2(
        contract,
        receipt,
        controller_git=controller_git,
        observed_root_identity=binding["root_identity"],
    ) == receipt

    for mutation in (
        "receipt-digest",
        "controller-commit",
        "target-commit",
        "six-path-diff",
        "old-40rps-4800-record-policy",
        "old-sustained-evidence",
        "old-evidence-schema",
        "root-identity",
        "formal-side-effect",
    ):
        forged = deepcopy(receipt)
        rehash = True
        if mutation == "receipt-digest":
            forged["receipt_digest"] = _tagged("forged-gauntlet")
            rehash = False
        elif mutation == "controller-commit":
            forged["controller_git"]["commit"] = "ef" * 20
        elif mutation == "target-commit":
            forged["target_binding"]["target_supervisor_commit"] = (
                "01" * 20
            )
        elif mutation == "six-path-diff":
            forged["controller_git"][
                "changed_paths_from_target_in_exact_order"
            ].pop()
        elif mutation == "old-40rps-4800-record-policy":
            forged_policy = forged["scheduling_policy"]
            assert type(forged_policy) is dict
            forged_policy.update({
                "sustained_offered_rate_records_per_second": 40,
                "minimum_sustained_publication_rate_records_per_second": 40,
                "sustained_records_per_cycle": 4,
                "sustained_timed_records": 4_800,
                "minimum_final_retained_records": 10_211,
            })
        elif mutation == "old-sustained-evidence":
            admission = forged["scheduling_admission"]
            assert type(admission) is dict
            sustained = admission["windows_in_strict_order"][0][
                "sustained_publication_or_null"
            ]
            sustained["timed_records_enqueued"] = 4_800
        elif mutation == "old-evidence-schema":
            forged["scheduling_admission"]["schema_version"] = (
                "hegel-phase3a-q05b-scheduling-admission-evidence/6"
            )
        elif mutation == "root-identity":
            forged["external_root_identity"]["ino"] += 1
        elif mutation == "formal-side-effect":
            forged["formal_slot_root_creation_or_write_count"] = 1
        else:  # pragma: no cover - the mutation registry is closed above.
            raise AssertionError(mutation)
        if rehash:
            forged = _rehash_gauntlet_receipt(forged)
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._validate_preauthorization_live_gauntlet_receipt_v2(
                contract,
                forged,
                controller_git=controller_git,
                observed_root_identity=binding["root_identity"],
            )
        assert error.value.code == "FAIL_CAMPAIGN_SOURCE"


def _temporary_preauthorization_gauntlet_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path, list[Path]]:
    contract = _contract()
    policy = TOOL._preauthorization_live_gauntlet_policy_v2(contract)
    root = tmp_path / "live-gauntlet"
    policy["stable_external_root"] = os.fspath(root)
    formal_roots: list[Path] = []
    fixed = contract["fixed_external_roots"]
    assert type(fixed) is dict
    for name in ("authorization_root", "ledger_root"):
        path = tmp_path / f"formal-{name}"
        fixed[name] = os.fspath(path)
        formal_roots.append(path)
    slots = contract["slot_registry"]
    assert type(slots) is list
    for ordinal, slot in enumerate(slots, 1):
        assert type(slot) is dict
        for field in (
            "clone_root",
            "cargo_cache_root",
            "artifact_parent_root",
            "scheduling_probe_root",
            "evidence_root",
            "trace_root",
        ):
            path = tmp_path / f"formal-slot-{ordinal}-{field}"
            slot[field] = os.fspath(path)
            formal_roots.append(path)
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    controller_git = {
        "commit": "ab" * 20,
        "tree": "cd" * 20,
        "parent": target["target_supervisor_commit"],
        "parent_count": 1,
        "clean": True,
        "changed_paths_from_target_in_exact_order": list(
            TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        ),
    }
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(
        TOOL,
        "_preauthorization_live_gauntlet_policy_v2",
        lambda _contract: deepcopy(policy),
    )
    monkeypatch.setattr(
        TOOL,
        "_controller_git_identity",
        lambda _contract: deepcopy(controller_git),
    )
    return contract, controller_git, root, formal_roots


def test_preauthorization_live_gauntlet_publishes_immutable_bound_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, controller_git, root, formal_roots = (
        _temporary_preauthorization_gauntlet_context(monkeypatch, tmp_path)
    )
    monkeypatch.setattr(
        TOOL,
        "_run_scheduling_admission_gate_v2",
        lambda _contract, _fd, _slot: _admitted_scheduling_evidence(contract),
    )
    result = TOOL.create_preauthorization_live_gauntlet_receipt_v2()
    assert result["status"] == "PASS"
    assert result["formal_root_write_count"] == 0
    assert result["supervisor_invocation_count"] == 0
    assert result["docker_daemon_contact_count"] == 0
    assert result["network_contact_count"] == 0
    assert stat.S_IMODE(root.stat(follow_symlinks=False).st_mode) == 0o700
    receipt_path = Path(str(result["receipt_path"]))
    receipt_raw = receipt_path.read_bytes()
    receipt_stat = receipt_path.stat(follow_symlinks=False)
    assert stat.S_IMODE(receipt_stat.st_mode) == 0o444
    assert receipt_stat.st_nlink == 1
    assert result["receipt_raw_sha256"] == sha256(receipt_raw).hexdigest()
    assert result["receipt_byte_count"] == len(receipt_raw)
    assert not (root / "live-admission-scratch").exists()
    assert all(not path.exists() for path in formal_roots)
    loaded = _REAL_LOAD_PREAUTHORIZATION_LIVE_GAUNTLET(
        contract,
        controller_git=controller_git,
    )
    assert loaded["receipt_digest"] == result["receipt_digest"]
    assert loaded["raw_sha256"] == result["receipt_raw_sha256"]
    assert loaded["byte_count"] == result["receipt_byte_count"]
    assert loaded["record_identity"]["mode"] == receipt_stat.st_mode
    assert loaded["record_identity"]["uid"] == os.geteuid()
    assert loaded["record_identity"]["nlink"] == 1
    assert loaded["record_identity"]["size"] == len(receipt_raw)

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2):
        TOOL.create_preauthorization_live_gauntlet_receipt_v2()
    assert receipt_path.read_bytes() == receipt_raw


def test_preauthorization_live_gauntlet_failure_removes_scratch_and_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _contract_value, _controller, root, formal_roots = (
        _temporary_preauthorization_gauntlet_context(monkeypatch, tmp_path)
    )

    def fail_gate(*_args: object, **_kwargs: object) -> NoReturn:
        raise TOOL.ShadowRetryCampaignErrorV2(
            TOOL._SCHEDULING_ADMISSION_FAILURE_V2,
            "injected ordinary gauntlet failure",
            terminal=True,
        )

    monkeypatch.setattr(
        TOOL,
        "_run_scheduling_admission_gate_v2",
        fail_gate,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.create_preauthorization_live_gauntlet_receipt_v2()
    assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
    assert not root.exists()
    assert all(not path.exists() for path in formal_roots)


def _exact_scheduling_exhaustion_error(
    contract: Mapping[str, object],
    slot_id: str,
) -> TOOL.ShadowRetryCampaignErrorV2:
    clock = _AdmissionClock()
    policy = TOOL._scheduling_admission_policy_v2(contract)

    def failed_burst(
        _fd: int,
        observed_slot_id: str,
        window: int,
        burst: int,
        _policy: Mapping[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        result = _admitted_scheduling_burst(
            policy,
            observed_slot_id,
            window,
            burst,
            start_ns=clock.now_ns,
            critical_ns=25_000_001,
        )
        clock.now_ns = int(result["burst_finish_ns"])
        return result

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as generated:
        TOOL._run_scheduling_admission_gate_v2(
            contract,
            7,
            slot_id,
            clock_ns=clock,
            sleeper=clock.sleep,
            burst_runner=failed_burst,
        )
    error = generated.value
    assert TOOL._scheduling_gate_exhaustion_is_exact_v2r17(
        contract,
        slot_id,
        error,
    )
    return error


def _exact_joined_task_quiescence_timeout_error(
    contract: Mapping[str, object],
    slot_id: str,
    *,
    component: str = "CRITICAL_BURST",
) -> TOOL.ShadowRetryCampaignErrorV2:
    policy = TOOL._scheduling_admission_policy_v2(contract)
    window = 2
    burst = 15 if component == "CRITICAL_BURST" else None
    names = (
        [
            "q05b-admission-durability-w2-b15",
            "q05b-admission-controller-w2-b15",
            "q05b-admission-python_endpoint-w2-b15",
            "q05b-admission-rust_endpoint-w2-b15",
        ]
        if component == "CRITICAL_BURST"
        else ["q05b-admission-sustained-w2"]
    )
    native_ids = list(range(2001, 2001 + len(names)))
    lingering = [native_ids[-1]]
    receipt: dict[str, object] = {
        "schema_version": (
            TOOL._SCHEDULING_JOINED_TASK_QUIESCENCE_SCHEMA_V1
        ),
        "condition": (
            TOOL._SCHEDULING_JOINED_TASK_QUIESCENCE_CONDITION_V2R17
        ),
        "slot_id": slot_id,
        "window_ordinal": window,
        "component": component,
        "burst_ordinal_or_null": burst,
        "quiescence_timeout_ns": policy[
            "joined_task_quiescence_timeout_ns"
        ],
        "quiescence_poll_interval_ns": policy[
            "joined_task_quiescence_poll_interval_ns"
        ],
        "wait_start_ns": 100,
        "wait_finish_ns": 1_000_000_100,
        "wait_duration_ns": 1_000_000_000,
        "poll_count": 1001,
        "controller_task_id": 1999,
        "joined_thread_names_in_start_order": names,
        "joined_thread_native_ids_in_start_order": native_ids,
        "joined_thread_is_alive_in_start_order": [False] * len(names),
        "join_completed_in_start_order": [True] * len(names),
        "task_ids_at_timeout_in_sorted_order": [1999, *lingering],
        "lingering_known_joined_task_ids_in_sorted_order": lingering,
        "unknown_task_ids_at_timeout_in_sorted_order": [],
        "terminal_drain_policy_exceeded": False,
        "primary_error_present": False,
        "producer_error_count": 0,
        "publisher_error_count": 0,
        "cleanup_error_count": 0,
        "passed": False,
    }
    assert TOOL._joined_task_quiescence_timeout_receipt_exact_v2r17(
        policy,
        slot_id,
        receipt,
    )
    detail = TOOL._joined_task_quiescence_timeout_detail_v2r17(receipt)
    assert detail is not None
    error = TOOL.ShadowRetryCampaignErrorV2(
        TOOL._SCHEDULING_ADMISSION_FAILURE_V2,
        detail,
        terminal=True,
        receipt=receipt,
    )
    assert TOOL._scheduling_gate_joined_task_timeout_is_exact_v2r17(
        contract,
        slot_id,
        error,
    )
    return error


def _admitted_scheduling_burst(
    policy: Mapping[str, object],
    slot_id: str,
    window: int,
    burst: int,
    *,
    start_ns: int,
    critical_ns: int = 25_000_000,
    drain_ns: int = 200_000_000,
) -> dict[str, object]:
    lane_operations = (
        ("CONTROLLER", 1, "SUPERVISOR_STAGE_04_OK"),
        ("CONTROLLER", 2, "SUPERVISOR_STAGE_05_BEGIN"),
        ("CONTROLLER", 3, "HELD_ACTOR_CLOSE_BEGIN"),
        ("PYTHON_ENDPOINT", 1, "LIVE_RESOURCE_SAMPLE_BEGIN"),
        ("PYTHON_ENDPOINT", 2, "LIVE_RESOURCE_SAMPLE_OK"),
        ("RUST_ENDPOINT", 1, "LIVE_RESOURCE_SAMPLE_BEGIN"),
        ("RUST_ENDPOINT", 2, "LIVE_RESOURCE_SAMPLE_OK"),
    )
    enqueue_rows: list[dict[str, object]] = []
    for ordinal, (lane, lane_ordinal, operation) in enumerate(
        lane_operations, start=1
    ):
        payload = TOOL._scheduling_probe_payload_v6(
            slot_id, window, burst, ordinal, lane, lane_ordinal, operation
        )
        enqueue_rows.append({
            "enqueue_ordinal": ordinal,
            "lane_id": lane,
            "lane_event_ordinal": lane_ordinal,
            "operation": operation,
            "raw_sha256": sha256(payload).hexdigest(),
        })
    durable_rows = [
        {
            "durable_ordinal": ordinal,
            "enqueue_ordinal": ordinal,
            "duration_ns": 1_000_000,
            "raw_sha256": row["raw_sha256"],
        }
        for ordinal, row in enumerate(enqueue_rows, start=1)
    ]
    sampler_rows = [
        {
            "actor_id": actor,
            "enqueue_ordinal": enqueue_ordinal,
            "enqueue_start_ns": start_ns,
            "post_enqueue_anchor_ns": start_ns + critical_ns,
            "deadline_to_post_enqueue_anchor_ns": critical_ns,
            "projected_gap_ns": (
                int(policy["sampling_target_wait_ns"]) + critical_ns
            ),
            "passed": critical_ns <= 25_000_000,
        }
        for actor, enqueue_ordinal in (
            ("PYTHON_ENDPOINT", 4), ("RUST_ENDPOINT", 6)
        )
    ]
    passed = critical_ns <= 25_000_000 and drain_ns <= 5_000_000_000
    result: dict[str, object] = {
        "burst_ordinal": burst,
        "release_deadline_ns": start_ns,
        "enqueue_rows_in_global_order": enqueue_rows,
        "durable_rows_in_global_order": durable_rows,
        "sampler_rows_in_actor_order": sampler_rows,
        "queue_start_depth": 0,
        "queue_end_depth": 0,
        "queue_high_watermark_records": 7,
        "dropped_record_count": 0,
        "duplicate_record_count": 0,
        "reordered_record_count": 0,
        "records_enqueued": 7,
        "records_durable": 7,
        "burst_start_ns": start_ns,
        "burst_finish_ns": start_ns + drain_ns,
        "burst_drain_ns": drain_ns,
        "passed": passed,
    }
    assert TOOL._scheduling_admission_burst_exact_v6(
        policy, slot_id, window, result
    )
    return result


def test_v2r20_target_d_scheduling_policy_is_exact_and_rejects_old_profiles(
) -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    assert policy["schema_version"] == (
        TOOL._SCHEDULING_ADMISSION_POLICY_SCHEMA_V8
    )
    assert policy["producer_lanes_in_exact_order"] == [
        "CONTROLLER", "PYTHON_ENDPOINT", "RUST_ENDPOINT",
    ]
    assert policy["sampling_target_wait_ns"] == 100_000_000
    assert policy[
        "maximum_sampler_event_production_rate_records_per_second"
    ] == 40
    assert policy["sustained_offered_rate_records_per_second"] == 50
    assert policy[
        "minimum_sustained_publication_rate_records_per_second"
    ] == 50
    assert policy["maximum_utilization_numerator"] == 4
    assert policy["maximum_utilization_denominator"] == 5
    assert policy["sustained_cycle_count"] == 1_200
    assert policy["sustained_timed_records"] == 6_000
    assert policy["sustained_prefill_retained_records"] == 5_411
    assert policy["minimum_final_retained_records"] == 11_411
    assert policy["sustained_queue_target_records"] == 64
    assert policy["maximum_projected_gap_ns"] == 125_000_000
    assert policy["reserved_runtime_margin_ns"] == 125_000_000
    assert policy["retained_probe_success_total_fsync_count"] == 4

    old_mutations = (
        {"schema_version": "hegel-phase3a-q05b-scheduling-admission-policy/7"},
        {"sampling_target_wait_ns": 50_000_000},
        {"maximum_sampler_event_production_rate_records_per_second": 100},
        {"sustained_cycle_count": 2_400},
        {"sustained_timed_records": 12_000},
        {"minimum_final_retained_records": 17_411},
        {"sustained_queue_target_records": 512},
        {
            "sustained_offered_rate_records_per_second": 40,
            "minimum_sustained_publication_rate_records_per_second": 40,
            "sustained_records_per_cycle": 4,
            "sustained_timed_records": 4_800,
            "minimum_final_retained_records": 10_211,
        },
    )
    for updates in old_mutations:
        mutated = deepcopy(contract)
        gate = mutated["execution_policy"]["scheduling_admission_gate"]
        assert type(gate) is dict
        gate.update(updates)
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._scheduling_admission_policy_v2(mutated)
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


def test_scheduling_admission_selects_first_passing_window_at_boundary() -> None:
    contract = _contract()
    clock = _AdmissionClock()
    calls: list[tuple[int, int]] = []
    policy = TOOL._scheduling_admission_policy_v2(contract)

    def run_burst(
        _fd: int, slot_id: str, window: int, burst: int,
        _policy: Mapping[str, object], **_kwargs: object,
    ) -> dict[str, object]:
        calls.append((window, burst))
        start = clock.now_ns
        critical = 25_000_001 if window == 1 and burst == 16 else 25_000_000
        result = _admitted_scheduling_burst(
            policy, slot_id, window, burst,
            start_ns=start, critical_ns=critical,
        )
        clock.now_ns = int(result["burst_finish_ns"])
        return result

    def run_sustained(
        _fd: int, slot_id: str, window: int,
        _policy: Mapping[str, object], **_kwargs: object,
    ) -> dict[str, object]:
        result = _sustained_scheduling_evidence(
            policy, slot_id, window, start_ns=clock.now_ns
        )
        clock.now_ns = int(result["measurement_finish_ns"])
        return result

    result = TOOL._run_scheduling_admission_gate_v2(
        contract,
        7,
        "SLOT_01",
        clock_ns=clock,
        sleeper=clock.sleep,
        burst_runner=run_burst,
        sustained_runner=run_sustained,
    )
    assert result["selected_window_ordinal"] == 2
    assert len(result["windows_in_strict_order"]) == 2
    assert len(calls) == 32
    assert calls[0] == (1, 1)
    assert calls[-1] == (2, 16)
    assert TOOL._scheduling_admission_evidence_exact_v2(contract, result)


def test_scheduling_admission_exhaustion_is_typed_and_prevents_spawn() -> None:
    contract = _contract()
    clock = _AdmissionClock()
    spawned = False

    policy = TOOL._scheduling_admission_policy_v2(contract)

    def run_burst(
        _fd: int, slot_id: str, window: int, burst: int,
        _policy: Mapping[str, object], **_kwargs: object,
    ) -> dict[str, object]:
        result = _admitted_scheduling_burst(
            policy, slot_id, window, burst,
            start_ns=clock.now_ns, critical_ns=25_000_001,
        )
        clock.now_ns = int(result["burst_finish_ns"])
        return result

    try:
        TOOL._run_scheduling_admission_gate_v2(
            contract,
            7,
            "SLOT_01",
            clock_ns=clock,
            sleeper=clock.sleep,
            burst_runner=run_burst,
        )
        spawned = True
    except TOOL.ShadowRetryCampaignErrorV2 as error:
        assert error.code == "FAIL_CAMPAIGN_SCHEDULING_ADMISSION"
        assert "critical_ns': 25000001" in error.detail
        assert "before consume/Popen" in error.detail
        assert TOOL._scheduling_admission_evidence_semantics_v6(
            contract, error.receipt, require_passed=False
        )
        bounded = TOOL._bounded_receipt_or_null(error.receipt)
        assert bounded is not None
        assert bounded["schema_version"] == (
                TOOL._SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V7
        )
        assert not TOOL._terminal_error_receipt_semantics_v2(
            contract,
            bounded,
            terminal_error_code=error.code,
            terminal_error_detail=error.detail,
            top_level=True,
            expected_prespawn_slot_id_or_null="SLOT_01",
        )
        assert not TOOL._terminal_error_receipt_semantics_v2(
            contract,
            bounded,
            terminal_error_code=error.code,
            terminal_error_detail=error.detail,
            top_level=True,
            expected_prespawn_slot_id_or_null="SLOT_02",
        )
    assert spawned is False


def test_scheduling_admission_rejects_short_quiet_period() -> None:
    contract = _contract()
    clock = _AdmissionClock()
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._run_scheduling_admission_gate_v2(
            contract,
            7,
            "SLOT_01",
            clock_ns=clock,
            sleeper=lambda _seconds: None,
            burst_runner=lambda *_args, **_kwargs: {},
        )
    assert error.value.code == "FAIL_CAMPAIGN_SCHEDULING_ADMISSION"
    assert "quiet period" in error.value.detail


def test_scheduling_admission_wraps_injected_burst_failure() -> None:
    contract = _contract()
    clock = _AdmissionClock()

    def fail_burst(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError("injected publication failure")

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._run_scheduling_admission_gate_v2(
            contract,
            7,
            "SLOT_01",
            clock_ns=clock,
            sleeper=clock.sleep,
            burst_runner=fail_burst,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SCHEDULING_ADMISSION"
    assert "burst 1:1 failed" in error.value.detail
    assert "injected publication failure" in error.value.detail


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("window_ordinal", True),
        ("quiet_duration_ns", 30_000_000_000.0),
        ("maximum_burst_drain_ns", 200_000_000.0),
    ),
)
def test_scheduling_admission_evidence_rejects_bool_or_float_integer_forgery(
    field: str,
    forged_value: object,
) -> None:
    contract = _contract()
    evidence = _admitted_scheduling_evidence(contract)
    windows = evidence["windows_in_strict_order"]
    assert type(windows) is list and type(windows[0]) is dict
    windows[0][field] = forged_value
    assert not TOOL._scheduling_admission_evidence_exact_v2(contract, evidence)


@pytest.mark.parametrize("location", ("burst", "sampler", "durable"))
def test_scheduling_admission_nested_evidence_rejects_nonexact_integer(
    location: str,
) -> None:
    contract = _contract()
    evidence = _admitted_scheduling_evidence(contract)
    burst = evidence["windows_in_strict_order"][0]["bursts_in_strict_order"][0]
    if location == "burst":
        burst["queue_high_watermark_records"] = True
    elif location == "sampler":
        burst["sampler_rows_in_actor_order"][0][
            "deadline_to_post_enqueue_anchor_ns"
        ] = 25_000_000.0
    else:
        burst["durable_duration_ns_total"] = False
    assert not TOOL._scheduling_admission_evidence_exact_v2(contract, evidence)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("maximum_windows", 3.0),
        ("quiet_period_ns", 30_000_000_000.0),
        ("bursts_per_window", True),
        ("maximum_burst_drain_ns", 5_000_000_000.0),
        ("queue_capacity_records", 512.0),
        ("publication_payload_byte_count", 4096.0),
    ),
)
def test_scheduling_admission_policy_rejects_nonexact_integer_types(
    field: str,
    forged_value: object,
) -> None:
    contract = _contract()
    policy = contract["execution_policy"]["scheduling_admission_gate"]
    assert type(policy) is dict
    policy[field] = forged_value
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._scheduling_admission_policy_v2(contract)
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


def test_four_fsync_publication_probe_removes_all_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        root = Path(temporary)
        directory_fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
            getattr(os, "O_NOFOLLOW", 0),
        )
        real_fsync = TOOL.os.fsync
        fsync_targets: list[str] = []

        def observed_fsync(descriptor: int) -> None:
            fsync_targets.append(
                "directory" if descriptor == directory_fd else "file"
            )
            real_fsync(descriptor)

        monkeypatch.setattr(TOOL.os, "fsync", observed_fsync)
        try:
            duration = TOOL._four_fsync_publication_probe_v6(
                directory_fd,
                "w01-p001",
                b"x" * 4096,
            )
            assert type(duration) is int and duration >= 0
            assert fsync_targets == [
                "file", "directory", "directory", "directory", "directory",
            ]
            assert list(root.iterdir()) == []
        finally:
            os.close(directory_fd)


def test_four_fsync_publication_probe_fault_removes_all_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        root = Path(temporary)
        directory_fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
            getattr(os, "O_NOFOLLOW", 0),
        )
        monkeypatch.setattr(
            TOOL.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected admission link failure")
            ),
        )
        try:
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                TOOL._four_fsync_publication_probe_v6(
                    directory_fd,
                    "w01-p001",
                    b"x" * 4096,
                )
            assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
            assert "injected admission link failure" in error.value.detail
            assert list(root.iterdir()) == []
        finally:
            os.close(directory_fd)


def test_stage5_admission_burst_exact_replay_and_thread_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    quiescence_calls: list[dict[str, object]] = []

    def quiescence(
        _threads: Sequence[threading.Thread],
        **kwargs: object,
    ) -> None:
        quiescence_calls.append(dict(kwargs))

    monkeypatch.setattr(
        TOOL,
        "_wait_for_joined_admission_tasks_to_quiesce_v2r17",
        quiescence,
    )
    result = TOOL._run_stage5_scheduling_admission_burst_v6(
        7,
        "SLOT_01",
        1,
        1,
        policy,
        probe_publisher=lambda *_args: 1,
    )
    assert result["records_enqueued"] == result["records_durable"] == 7
    assert result["queue_start_depth"] == result["queue_end_depth"] == 0
    assert result["queue_high_watermark_records"] <= 7
    assert TOOL._scheduling_admission_burst_exact_v6(
        policy, "SLOT_01", 1, result
    )
    assert len(quiescence_calls) == 1
    assert quiescence_calls[0]["component"] == "CRITICAL_BURST"
    assert quiescence_calls[0]["burst_ordinal_or_null"] == 1


def test_stage5_admission_burst_propagates_exact_joined_task_quiescence_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    timeout = _exact_joined_task_quiescence_timeout_error(
        contract,
        "SLOT_01",
        component="CRITICAL_BURST",
    )
    assert type(timeout.receipt) is dict

    def quiescence(
        _threads: Sequence[threading.Thread],
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs["component"] == "CRITICAL_BURST"
        assert kwargs["window"] == 2
        assert kwargs["burst_ordinal_or_null"] == 15
        assert kwargs["allow_clean_timeout_receipt"] is True
        return deepcopy(timeout.receipt)

    monkeypatch.setattr(
        TOOL,
        "_wait_for_joined_admission_tasks_to_quiesce_v2r17",
        quiescence,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as generated:
        TOOL._run_stage5_scheduling_admission_burst_v6(
            7,
            "SLOT_01",
            2,
            15,
            policy,
            probe_publisher=lambda *_args: 1,
        )
    assert TOOL._scheduling_gate_joined_task_timeout_is_exact_v2r17(
        contract,
        "SLOT_01",
        generated.value,
    )


def _admission_threads() -> list[threading.Thread]:
    return [
        thread for thread in threading.enumerate()
        if thread.name.startswith("q05b-admission-")
    ]


def _joined_dead_admission_thread(name: str) -> threading.Thread:
    thread = threading.Thread(target=lambda: None, name=name, daemon=True)
    thread.start()
    thread.join()
    assert thread.native_id is not None
    assert thread.is_alive() is False
    return thread


def test_joined_known_dead_task_quiesces_transparently_before_strict_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _joined_dead_admission_thread(
        "q05b-admission-sustained-w1"
    )
    assert thread.native_id is not None
    controller = os.getpid()
    task_sets = [
        [str(controller), str(thread.native_id)],
        [str(controller)],
    ]
    clock_values = iter([10_000_000, 10_100_000])
    strict_calls = 0
    sleep_calls: list[float] = []

    def strict_proof() -> None:
        nonlocal strict_calls
        strict_calls += 1

    monkeypatch.setattr(
        TOOL,
        "_require_single_threaded_spawn_controller_v2",
        strict_proof,
    )
    receipt = _REAL_WAIT_JOINED_ADMISSION_TASKS(
        [thread],
        slot_id="SLOT_01",
        window=1,
        component="SUSTAINED_PUBLICATION",
        burst_ordinal_or_null=None,
        timeout_ns=1_000_000_000,
        poll_interval_ns=1_000_000,
        allow_clean_timeout_receipt=True,
        clock_ns=lambda: next(clock_values),
        sleeper=lambda seconds: sleep_calls.append(seconds),
        task_names_reader=lambda: task_sets.pop(0),
    )
    assert receipt is None
    assert strict_calls == 1
    assert sleep_calls == [0.001]


def test_joined_known_dead_task_timeout_receipt_is_exact() -> None:
    thread = _joined_dead_admission_thread(
        "q05b-admission-sustained-w2"
    )
    assert thread.native_id is not None
    controller = os.getpid()
    clock_values = iter([20_000_000, 1_020_000_000])
    receipt = _REAL_WAIT_JOINED_ADMISSION_TASKS(
        [thread],
        slot_id="SLOT_01",
        window=2,
        component="SUSTAINED_PUBLICATION",
        burst_ordinal_or_null=None,
        timeout_ns=1_000_000_000,
        poll_interval_ns=1_000_000,
        allow_clean_timeout_receipt=True,
        clock_ns=lambda: next(clock_values),
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("timeout path must not sleep after its final poll")
        ),
        task_names_reader=lambda: [
            str(controller),
            str(thread.native_id),
        ],
    )
    policy = TOOL._scheduling_admission_policy_v2(_contract())
    assert TOOL._joined_task_quiescence_timeout_receipt_exact_v2r17(
        policy,
        "SLOT_01",
        receipt,
    )
    assert receipt["wait_duration_ns"] == 1_000_000_000
    assert receipt[
        "lingering_known_joined_task_ids_in_sorted_order"
    ] == [thread.native_id]
    assert "terminal drain" not in (
        TOOL._joined_task_quiescence_timeout_detail_v2r17(receipt) or ""
    )


def test_joined_task_quiescence_rejects_unknown_and_alive_tasks() -> None:
    dead = _joined_dead_admission_thread(
        "q05b-admission-sustained-w1"
    )
    assert dead.native_id is not None
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as unknown:
        _REAL_WAIT_JOINED_ADMISSION_TASKS(
            [dead],
            slot_id="SLOT_01",
            window=1,
            component="SUSTAINED_PUBLICATION",
            burst_ordinal_or_null=None,
            timeout_ns=1_000_000_000,
            poll_interval_ns=1_000_000,
            allow_clean_timeout_receipt=True,
            task_names_reader=lambda: [str(os.getpid()), "2147483646"],
        )
    assert unknown.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "untracked task" in unknown.value.detail

    release = threading.Event()
    alive = threading.Thread(
        target=lambda: release.wait(5),
        name="q05b-admission-sustained-w1",
        daemon=True,
    )
    alive.start()
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as live:
            _REAL_WAIT_JOINED_ADMISSION_TASKS(
                [alive],
                slot_id="SLOT_01",
                window=1,
                component="SUSTAINED_PUBLICATION",
                burst_ordinal_or_null=None,
                timeout_ns=1_000_000_000,
                poll_interval_ns=1_000_000,
                allow_clean_timeout_receipt=True,
            )
        assert live.value.code == "FAIL_CAMPAIGN_PROCESS"
        assert "not all uniquely bound and joined" in live.value.detail
    finally:
        release.set()
        alive.join(timeout=2)


def test_stage5_admission_release_clock_fault_leaves_no_thread() -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._run_stage5_scheduling_admission_burst_v6(
            7, "SLOT_01", 1, 1, policy,
            clock_ns=lambda: True,
            probe_publisher=lambda *_args: 1,
        )
    assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
    assert "burst release clock differs" in error.value.detail
    assert _admission_threads() == []


def test_stage5_admission_thread_start_fault_leaves_no_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    real_start = threading.Thread.start
    admission_start_count = 0

    def start_with_fault(thread: threading.Thread) -> None:
        nonlocal admission_start_count
        if thread.name.startswith("q05b-admission-"):
            admission_start_count += 1
            if admission_start_count == 3:
                raise OSError("injected thread start failure")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_with_fault)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._run_stage5_scheduling_admission_burst_v6(
            7, "SLOT_01", 1, 1, policy,
            probe_publisher=lambda *_args: 1,
        )
    assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
    assert "injected thread start failure" in error.value.detail
    assert _admission_threads() == []


def test_stage5_admission_post_start_interrupt_leaves_no_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    real_start = threading.Thread.start
    admission_start_count = 0

    def start_then_interrupt(thread: threading.Thread) -> None:
        nonlocal admission_start_count
        if thread.name.startswith("q05b-admission-"):
            admission_start_count += 1
            if admission_start_count == 3:
                real_start(thread)
                raise KeyboardInterrupt("injected post-start interrupt")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_then_interrupt)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._run_stage5_scheduling_admission_burst_v6(
            7,
            "SLOT_01",
            1,
            1,
            policy,
            probe_publisher=lambda *_args: 1,
        )
    assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
    assert "injected post-start interrupt" in error.value.detail
    assert _admission_threads() == []


def test_stage5_admission_publication_fault_leaves_no_thread() -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)

    def fail_publication(*_args: object) -> int:
        raise OSError("injected durability failure")

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._run_stage5_scheduling_admission_burst_v6(
            7, "SLOT_01", 1, 1, policy,
            probe_publisher=fail_publication,
        )
    assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
    assert "injected durability failure" in error.value.detail
    assert _admission_threads() == []


def _fast_sustained_policy(
    production_policy: Mapping[str, object],
) -> dict[str, object]:
    policy = deepcopy(production_policy)
    policy.update({
        "sampling_target_wait_ns": 25_000_000,
        "maximum_sampler_deadline_to_post_enqueue_anchor_ns": 25_000_000,
        "sustained_offered_rate_records_per_second": 200,
        "minimum_sustained_publication_rate_records_per_second": 200,
        "sustained_records_per_cycle": 5,
        "sustained_cycle_count": 4,
        "sustained_measurement_duration_ns": 100_000_000,
        "sustained_epoch_duration_ns": 25_000_000,
        "sustained_prefill_retained_records": 6,
        "sustained_timed_records": 20,
        "prior_failed_trajectory_event_count": 5,
        "minimum_final_retained_records": 26,
        "sustained_queue_target_records": 10,
        "maximum_sustained_queue_high_watermark_records": 10,
        "maximum_sustained_terminal_drain_ns": 100_000_000,
    })
    return policy


def _retaining_test_probe(
    observed_identities: list[dict[str, object]],
    *,
    delay_seconds: float = 0.0,
) -> object:
    def retain_probe(
        directory_fd: int,
        token: str,
        payload: bytes,
        _clock_ns: object,
    ) -> int:
        identity = TOOL.json.loads(payload.decode("utf-8"))
        assert type(identity) is dict
        identity.pop("padding")
        identity.pop("record_digest")
        observed_identities.append(identity)
        final = TOOL._scheduling_sustained_final_name_v6(token)
        descriptor = os.open(
            final,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
            getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                assert written > 0
                offset += written
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        if delay_seconds:
            TOOL.time.sleep(delay_seconds)
        return max(1, int(delay_seconds * 1_000_000_000))

    return retain_probe


def test_stage5_sustained_admission_retains_across_nondivisible_prefill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    production_policy = TOOL._scheduling_admission_policy_v2(_contract())
    policy = _fast_sustained_policy(production_policy)
    observed_identities: list[dict[str, object]] = []
    quiescence_calls: list[dict[str, object]] = []

    def quiescence(
        _threads: Sequence[threading.Thread],
        **kwargs: object,
    ) -> None:
        quiescence_calls.append(dict(kwargs))

    monkeypatch.setattr(
        TOOL,
        "_wait_for_joined_admission_tasks_to_quiesce_v2r17",
        quiescence,
    )

    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        result = TOOL._run_stage5_sustained_publication_admission_v6(
            directory_fd,
            "SLOT_01",
            1,
            policy,
            probe_publisher=_retaining_test_probe(observed_identities),
        )
    finally:
        os.close(directory_fd)

    assert result["passed"] is True
    assert result["prefill_records_durable"] == 6
    assert result["timed_records_durable_by_deadline"] == 20
    assert result["timed_records_durable_after_deadline"] == 0
    assert result["records_durable_total"] == 26 > 5
    assert result["sampler_anchor_count"] == 8
    assert len(result["epoch_aggregates_in_strict_order"]) == 4
    assert TOOL._scheduling_sustained_evidence_exact_v6(
        policy, "SLOT_01", 1, result
    )
    assert observed_identities[5]["phase"] == "PREFILL_RETAINED"
    assert observed_identities[5]["phase_ordinal"] == 6
    assert observed_identities[6]["phase"] == "TIMED_SUSTAINED"
    assert observed_identities[6]["phase_ordinal"] == 1
    assert observed_identities[6]["cycle_ordinal"] == 1
    assert observed_identities[6]["cycle_event_ordinal"] == 1
    assert observed_identities[6]["operation"] == "LIVE_RESOURCE_SAMPLE_BEGIN"
    assert list(tmp_path.iterdir()) == []
    assert _admission_threads() == []
    assert len(quiescence_calls) == 1
    assert quiescence_calls[0]["component"] == "SUSTAINED_PUBLICATION"
    assert quiescence_calls[0]["burst_ordinal_or_null"] is None


def test_stage5_sustained_admission_propagates_exact_joined_task_quiescence_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    production_policy = TOOL._scheduling_admission_policy_v2(contract)
    policy = _fast_sustained_policy(production_policy)
    timeout = _exact_joined_task_quiescence_timeout_error(
        contract,
        "SLOT_01",
        component="SUSTAINED_PUBLICATION",
    )
    assert type(timeout.receipt) is dict
    observed_identities: list[dict[str, object]] = []

    def quiescence(
        _threads: Sequence[threading.Thread],
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs["component"] == "SUSTAINED_PUBLICATION"
        assert kwargs["window"] == 2
        assert kwargs["burst_ordinal_or_null"] is None
        assert kwargs["allow_clean_timeout_receipt"] is True
        return deepcopy(timeout.receipt)

    monkeypatch.setattr(
        TOOL,
        "_wait_for_joined_admission_tasks_to_quiesce_v2r17",
        quiescence,
    )
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as generated:
            TOOL._run_stage5_sustained_publication_admission_v6(
                directory_fd,
                "SLOT_01",
                2,
                policy,
                probe_publisher=_retaining_test_probe(observed_identities),
            )
    finally:
        os.close(directory_fd)
    assert list(tmp_path.iterdir()) == []
    assert TOOL._scheduling_gate_joined_task_timeout_is_exact_v2r17(
        contract,
        "SLOT_01",
        generated.value,
    )


def test_stage5_sustained_queue_pressure_drains_durable_prefix_and_cleans(
    tmp_path: Path,
) -> None:
    production_policy = TOOL._scheduling_admission_policy_v2(_contract())
    policy = _fast_sustained_policy(production_policy)
    policy["sustained_queue_target_records"] = 5
    policy["maximum_sustained_queue_high_watermark_records"] = 5
    observed_identities: list[dict[str, object]] = []
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        result = TOOL._run_stage5_sustained_publication_admission_v6(
            directory_fd,
            "SLOT_01",
            1,
            policy,
            probe_publisher=_retaining_test_probe(
                observed_identities,
                delay_seconds=0.015,
            ),
        )
    finally:
        os.close(directory_fd)

    assert result["passed"] is False
    assert result["timed_records_enqueued"] == 5
    assert result["timed_offer_terminal_disposition"] == (
        TOOL._SCHEDULING_SUSTAINED_QUEUE_REJECTED_V7
    )
    assert result["rejected_record_count"] == 5
    assert result["unoffered_after_failfast_record_count"] == 10
    assert result["scheduled_shortfall_record_count"] == 15
    assert result["full_measurement_duration_observed"] is True
    rejection = result["first_queue_capacity_rejection_or_null"]
    assert type(rejection) is dict
    assert rejection["disposition"] == (
        TOOL._SCHEDULING_SUSTAINED_QUEUE_REJECTED_V7
    )
    assert rejection["cycle_ordinal"] == 2
    assert rejection["due_ns"] == (
        result["measurement_start_ns"] + policy["sampling_target_wait_ns"]
    )
    assert rejection["attempt_ns"] >= rejection["due_ns"]
    assert rejection["lateness_ns"] == (
        rejection["attempt_ns"] - rejection["due_ns"]
    )
    assert 0 < rejection["queue_depth_records"] <= 5
    assert rejection["queue_depth_records"] + 5 > 5
    assert rejection["item_count_records"] == 5
    assert result["records_durable_total"] == 11
    assert result["retained_records_at_measurement_end"] == 11
    assert result["cleanup_removed_records"] == 11
    assert TOOL._scheduling_sustained_evidence_exact_v6(
        policy, "SLOT_01", 1, result
    )
    semantic_mutations = (
        ("timed_offer_terminal_disposition", TOOL._SCHEDULING_SUSTAINED_ENQUEUED_V7),
        ("rejected_record_count", 4),
        ("unoffered_after_failfast_record_count", 9),
        ("scheduled_shortfall_record_count", 14),
        ("full_measurement_duration_observed", False),
    )
    for field, value in semantic_mutations:
        mutated = deepcopy(result)
        mutated[field] = value
        assert not TOOL._scheduling_sustained_evidence_exact_v6(
            policy, "SLOT_01", 1, mutated
        )
    rejection_mutations = (
        ("disposition", TOOL._SCHEDULING_SUSTAINED_ENQUEUED_V7),
        ("cycle_ordinal", int(rejection["cycle_ordinal"]) + 1),
        ("due_ns", int(rejection["due_ns"]) + 1),
        ("attempt_ns", int(rejection["due_ns"]) - 1),
        ("lateness_ns", int(rejection["lateness_ns"]) + 1),
        ("queue_depth_records", 0),
        ("item_count_records", 4),
    )
    for field, value in rejection_mutations:
        mutated = deepcopy(result)
        first = mutated["first_queue_capacity_rejection_or_null"]
        assert type(first) is dict
        first[field] = value
        assert not TOOL._scheduling_sustained_evidence_exact_v6(
            policy, "SLOT_01", 1, mutated
        )
    assert list(tmp_path.iterdir()) == []
    assert _admission_threads() == []


def test_stage5_sustained_publisher_error_is_terminal_not_capacity_rejection(
    tmp_path: Path,
) -> None:
    policy = _fast_sustained_policy(
        TOOL._scheduling_admission_policy_v2(_contract())
    )
    observed_identities: list[dict[str, object]] = []
    retain = _retaining_test_probe(observed_identities)

    def fail_first_timed_publication(
        directory_fd: int,
        token: str,
        payload: bytes,
        clock_ns: object,
    ) -> int:
        identity = TOOL.json.loads(payload.decode("utf-8"))
        assert type(identity) is dict
        if identity["phase"] == "TIMED_SUSTAINED":
            raise OSError("injected sustained publisher failure")
        return retain(directory_fd, token, payload, clock_ns)

    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as generated:
            TOOL._run_stage5_sustained_publication_admission_v6(
                directory_fd,
                "SLOT_01",
                1,
                policy,
                probe_publisher=fail_first_timed_publication,
            )
    finally:
        os.close(directory_fd)
    assert generated.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
    assert "injected sustained publisher failure" in generated.value.detail
    assert "QUEUE_CAPACITY_REJECTED" not in generated.value.detail
    assert list(tmp_path.iterdir()) == []
    assert _admission_threads() == []


def test_retained_four_fsync_probe_keeps_only_immutable_final(
    tmp_path: Path,
) -> None:
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    token = "w01-r000001"
    final = TOOL._scheduling_sustained_final_name_v6(token)
    try:
        duration = TOOL._retained_four_fsync_publication_probe_v6(
            directory_fd,
            token,
            b"x" * 4096,
        )
        assert type(duration) is int and duration >= 0
        retained = os.stat(final, dir_fd=directory_fd, follow_symlinks=False)
        assert stat.S_IMODE(retained.st_mode) == 0o444
        assert retained.st_nlink == 1
        assert retained.st_size == 4096
        assert sorted(path.name for path in tmp_path.iterdir()) == [final]
        os.unlink(final, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    assert list(tmp_path.iterdir()) == []


def test_retained_probe_success_has_exact_four_fsync_and_full_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    token = "w01-r000002"
    staging = f".q05b-admission-{token}.staging-v7"
    final = TOOL._scheduling_sustained_final_name_v6(token)
    real_fsync = TOOL.os.fsync
    real_close = TOOL.os.close
    real_stat = TOOL.os.stat
    events: list[tuple[str, object]] = []
    clock_calls = 0

    def observed_fsync(descriptor: int) -> None:
        events.append((
            "fsync",
            "directory" if descriptor == directory_fd else "file",
        ))
        real_fsync(descriptor)

    def observed_close(descriptor: int) -> None:
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = "unresolved"
        events.append(("close", target))
        real_close(descriptor)

    def observed_stat(path: object, *args: object, **kwargs: object) -> object:
        if path == staging:
            events.append(("staging-absence-proof", path))
        return real_stat(path, *args, **kwargs)

    def observed_clock() -> int:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 1:
            return 100
        assert clock_calls == 2
        assert any(
            kind == "close" and str(value).endswith("/" + final)
            for kind, value in events
        )
        assert ("staging-absence-proof", staging) in events
        return 250

    monkeypatch.setattr(TOOL.os, "fsync", observed_fsync)
    monkeypatch.setattr(TOOL.os, "close", observed_close)
    monkeypatch.setattr(TOOL.os, "stat", observed_stat)
    try:
        duration = TOOL._retained_four_fsync_publication_probe_v6(
            directory_fd,
            token,
            b"x" * 4096,
            observed_clock,
        )
        assert duration == 150
        assert [value for kind, value in events if kind == "fsync"] == [
            "file", "directory", "directory", "directory",
        ]
        assert sorted(path.name for path in tmp_path.iterdir()) == [final]
        os.unlink(final, dir_fd=directory_fd)
        real_fsync(directory_fd)
    finally:
        real_close(directory_fd)
    assert list(tmp_path.iterdir()) == []


def test_retained_probe_failure_cleanup_mutation_is_durably_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    token = "w01-r000003"
    final = TOOL._scheduling_sustained_final_name_v6(token)
    real_open = TOOL.os.open
    real_unlink = TOOL.os.unlink
    real_fsync = TOOL.os.fsync
    events: list[tuple[str, object]] = []

    def fail_final_readback(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == final:
            raise OSError("injected final readback failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def observed_unlink(path: object, *, dir_fd: int | None = None) -> None:
        real_unlink(path, dir_fd=dir_fd)
        events.append(("unlink", path))

    def observed_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        events.append((
            "fsync",
            "directory" if descriptor == directory_fd else "file",
        ))

    monkeypatch.setattr(TOOL.os, "open", fail_final_readback)
    monkeypatch.setattr(TOOL.os, "unlink", observed_unlink)
    monkeypatch.setattr(TOOL.os, "fsync", observed_fsync)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._retained_four_fsync_publication_probe_v6(
                directory_fd,
                token,
                b"x" * 4096,
            )
        assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
        assert "injected final readback failure" in error.value.detail
        final_unlink_index = events.index(("unlink", final))
        cleanup_fsync_index = max(
            index for index, event in enumerate(events)
            if event == ("fsync", "directory")
        )
        assert cleanup_fsync_index > final_unlink_index
        assert [value for kind, value in events if kind == "fsync"] == [
            "file", "directory", "directory", "directory", "directory",
        ]
        assert list(tmp_path.iterdir()) == []
    finally:
        os.close(directory_fd)


def test_retained_probe_failure_without_namespace_mutation_does_not_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    real_open = TOOL.os.open
    fsync_calls: list[int] = []

    def fail_staging_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == ".q05b-admission-w01-r000004.staging-v7":
            raise OSError("injected pre-create failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(TOOL.os, "open", fail_staging_open)
    monkeypatch.setattr(
        TOOL.os,
        "fsync",
        lambda descriptor: fsync_calls.append(descriptor),
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._retained_four_fsync_publication_probe_v6(
                directory_fd,
                "w01-r000004",
                b"x" * 4096,
            )
        assert error.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
        assert "injected pre-create failure" in error.value.detail
        assert fsync_calls == []
        assert list(tmp_path.iterdir()) == []
    finally:
        os.close(directory_fd)


def test_stage5_sustained_6000_record_simulation_has_no_shortfall(
    tmp_path: Path,
) -> None:
    policy = TOOL._scheduling_admission_policy_v2(_contract())
    assert policy["sampling_target_wait_ns"] == 100_000_000
    assert policy["sustained_offered_rate_records_per_second"] == 50
    assert policy["sustained_cycle_count"] == 1_200
    assert policy["sustained_timed_records"] == 6_000
    assert policy["sustained_prefill_retained_records"] == 5_411
    assert policy["minimum_final_retained_records"] == 11_411
    clock = _AdmissionClock()
    observed_identities: list[dict[str, object]] = []

    def cooperative_sleeper(seconds: float) -> None:
        clock.sleep(seconds)
        TOOL.time.sleep(0.01)

    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
        getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        result = TOOL._run_stage5_sustained_publication_admission_v6(
            directory_fd,
            "SLOT_01",
            1,
            policy,
            clock_ns=clock,
            sleeper=cooperative_sleeper,
            probe_publisher=_retaining_test_probe(observed_identities),
        )
    finally:
        os.close(directory_fd)

    assert result["passed"] is True
    assert result["timed_records_enqueued"] == 6_000
    assert result["timed_records_durable_by_deadline"] == 6_000
    assert result["timed_records_durable_after_deadline"] == 0
    assert result["timed_offer_terminal_disposition"] == (
        TOOL._SCHEDULING_SUSTAINED_ENQUEUED_V7
    )
    assert result["rejected_record_count"] == 0
    assert result["unoffered_after_failfast_record_count"] == 0
    assert result["scheduled_shortfall_record_count"] == 0
    assert result["full_measurement_duration_observed"] is True
    assert result["first_queue_capacity_rejection_or_null"] is None
    assert result["queue_high_watermark_records"] <= 64
    assert result["cleanup_absence_passed"] is True
    assert len(observed_identities) == 11_411
    assert list(tmp_path.iterdir()) == []
    assert _admission_threads() == []

    semantic_mutations = (
        ("timed_offer_terminal_disposition", TOOL._SCHEDULING_SUSTAINED_QUEUE_REJECTED_V7),
        ("rejected_record_count", 1),
        ("unoffered_after_failfast_record_count", 1),
        ("scheduled_shortfall_record_count", 1),
        ("full_measurement_duration_observed", False),
        (
            "first_queue_capacity_rejection_or_null",
            {
                "disposition": TOOL._SCHEDULING_SUSTAINED_QUEUE_REJECTED_V7,
                "cycle_ordinal": 1,
                "due_ns": result["measurement_start_ns"],
                "attempt_ns": result["measurement_start_ns"],
                "lateness_ns": 0,
                "queue_depth_records": 64,
                "item_count_records": 5,
            },
        ),
    )
    for field, value in semantic_mutations:
        mutated = deepcopy(result)
        mutated[field] = value
        assert not TOOL._scheduling_sustained_evidence_exact_v6(
            policy, "SLOT_01", 1, mutated
        )


def test_sustained_admission_late_durability_and_anchor_overrun_fail_closed(
) -> None:
    policy = TOOL._scheduling_admission_policy_v2(_contract())
    late = _sustained_scheduling_evidence(
        policy,
        "SLOT_01",
        1,
        start_ns=1,
        durable_by_deadline=int(policy["sustained_timed_records"]) - 1,
    )
    assert late["passed"] is False
    assert late["timed_records_durable_by_deadline"] == 5_999
    assert late["timed_records_durable_after_deadline"] == 1
    assert late["terminal_drain_ns"] == 1
    assert TOOL._scheduling_sustained_evidence_exact_v6(
        policy, "SLOT_01", 1, late
    )

    overrun = _sustained_scheduling_evidence(
        policy,
        "SLOT_01",
        1,
        start_ns=1,
        sampler_anchor_maximum_ns=25_000_001,
    )
    assert overrun["passed"] is False
    assert TOOL._scheduling_sustained_evidence_exact_v6(
        policy, "SLOT_01", 1, overrun
    )


def test_three_failed_scheduling_windows_fit_direct_receipt_bound() -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    previous_finish = 0
    windows: list[dict[str, object]] = []
    for window in range(1, 4):
        quiet_start = previous_finish
        quiet_finish = quiet_start + int(policy["quiet_period_ns"])
        full_bursts: list[dict[str, object]] = []
        burst_start = quiet_finish
        for burst in range(1, 17):
            full = _admitted_scheduling_burst(
                policy,
                "SLOT_01",
                window,
                burst,
                start_ns=burst_start,
            )
            full_bursts.append(full)
            burst_start = int(full["burst_finish_ns"])
        bursts = [
            TOOL._compact_scheduling_admission_burst_v6(
                policy, "SLOT_01", window, row
            )
            for row in full_bursts
        ]
        sustained = _sustained_scheduling_evidence(
            policy,
            "SLOT_01",
            window,
            start_ns=burst_start,
            durable_by_deadline=(
                int(policy["sustained_timed_records"]) - 1
            ),
        )
        windows.append({
            "window_ordinal": window,
            "quiet_start_ns": quiet_start,
            "quiet_finish_ns": quiet_finish,
            "quiet_duration_ns": quiet_finish - quiet_start,
            "bursts_in_strict_order": bursts,
            "sustained_publication_or_null": sustained,
            "maximum_sampler_deadline_to_post_enqueue_anchor_ns": 25_000_000,
            "maximum_projected_gap_ns": int(
                policy["maximum_projected_gap_ns"]
            ),
            "maximum_burst_drain_ns": 200_000_000,
            "maximum_queue_high_watermark_records": 7,
            "passed": False,
        })
        previous_finish = int(sustained["measurement_finish_ns"])
    evidence: dict[str, object] = {
        "schema_version": TOOL._SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V7,
        "slot_id": "SLOT_01",
        "policy": deepcopy(policy),
        "windows_in_strict_order": windows,
        "selected_window_ordinal": None,
        "passed": False,
    }
    assert TOOL._scheduling_admission_evidence_semantics_v6(
        contract, evidence, require_passed=False
    )
    encoded = TOOL._canonical_json_bytes(evidence)
    assert len(encoded) <= TOOL._MAX_ERROR_RECEIPT_BYTES
    bounded = TOOL._bounded_receipt_or_null(evidence)
    assert bounded == evidence
    assert bounded["schema_version"] == (
        TOOL._SCHEDULING_ADMISSION_EVIDENCE_SCHEMA_V7
    )


@pytest.mark.parametrize("mutation", ("rehash", "reorder", "truncate"))
def test_scheduling_admission_evidence_rejects_record_forgery(
    mutation: str,
) -> None:
    contract = _contract()
    evidence = _admitted_scheduling_evidence(contract)
    burst = evidence["windows_in_strict_order"][0]["bursts_in_strict_order"][0]
    if mutation == "rehash":
        burst["payload_chain_digest"] = _raw("forged-payload")
    elif mutation == "reorder":
        burst["enqueue_order_codes_in_global_order"][0], burst[
            "enqueue_order_codes_in_global_order"
        ][1] = burst["enqueue_order_codes_in_global_order"][1], burst[
            "enqueue_order_codes_in_global_order"
        ][0]
    else:
        burst["enqueue_order_codes_in_global_order"].pop()
        burst["records_enqueued"] = 6
    assert not TOOL._scheduling_admission_evidence_exact_v2(contract, evidence)


def _slot(
    contract: Mapping[str, object],
    ordinal: int = 1,
) -> dict[str, object]:
    slots = contract["slot_registry"]
    assert type(slots) is list
    value = slots[ordinal - 1]
    assert type(value) is dict
    return deepcopy(value)


def test_prepare_clone_rebuilds_reachable_graph_without_source_only_objects(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    temporary = tempfile.TemporaryDirectory(
        prefix="hegel-q05b-clone-graph-kat-",
        dir="/tmp",
    )
    request.addfinalizer(temporary.cleanup)
    tmp_path = Path(temporary.name)
    git = "/usr/bin/git"
    source = tmp_path / "source"
    subprocess.run(
        [git, "init", "--initial-branch=main", os.fspath(source)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def source_git(*arguments: str) -> bytes:
        result = subprocess.run(
            [git, "-C", os.fspath(source), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    source_git("config", "user.name", "Q05b KAT")
    source_git("config", "user.email", "q05b-kat@example.invalid")
    project = source / ROOT.name
    contract = _contract()
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    required = target["required_raw_sha256_by_path"]
    assert type(required) is dict
    for index, relative in enumerate(required, start=1):
        path = project / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}-parent\n".encode("ascii"))
    source_git("add", ROOT.name)
    source_git("commit", "--quiet", "-m", "fixture parent")
    parent = source_git("rev-parse", "HEAD").decode("ascii").strip()
    first_required = project / str(next(iter(required)))
    first_required.write_bytes(b"fixture-target\n")
    source_git("add", ROOT.name)
    source_git("commit", "--quiet", "-m", "fixture target")
    target_commit = source_git("rev-parse", "HEAD").decode("ascii").strip()
    target_tree = source_git("rev-parse", "HEAD^{tree}").decode("ascii").strip()

    source_git("checkout", "--quiet", "-b", "source-only-ref")
    first_required.write_bytes(b"fixture-source-only-ref\n")
    source_git("add", ROOT.name)
    source_git("commit", "--quiet", "-m", "source-only ref commit")
    source_only_ref = source_git("rev-parse", "HEAD").decode("ascii").strip()
    source_git("checkout", "--quiet", "main")
    first_required.write_bytes(b"fixture-stash-only\n")
    source_git("add", ROOT.name)
    source_git("stash", "push", "--quiet", "-m", "source-only stash")
    source_only_stash = source_git("rev-parse", "refs/stash").decode(
        "ascii"
    ).strip()
    source_git("commit-graph", "write", "--reachable")
    source_git("commit-graph", "verify")
    source_graph = (source / ".git/objects/info/commit-graph").read_bytes()
    assert bytes.fromhex(source_only_ref) in source_graph
    assert bytes.fromhex(source_only_stash) in source_graph
    assert source_git("status", "--porcelain=v1", "-z") == b""

    execution["clone_source_repository"] = os.fspath(source)
    execution["clone_source_project_relative_path"] = ROOT.name
    target["target_supervisor_commit"] = target_commit
    target["target_supervisor_tree"] = target_tree
    target["target_supervisor_parent_commit"] = parent
    for relative in required:
        required[str(relative)] = sha256(
            (project / str(relative)).read_bytes()
        ).hexdigest()

    slot = contract["slot_registry"][0]
    assert type(slot) is dict
    clone = tmp_path / "slot-01-clone"
    clone_project = clone / ROOT.name
    artifact_parent = tmp_path / "slot-01-artifact"
    evidence = tmp_path / "slot-01-evidence"
    trace = tmp_path / "slot-01-trace"
    slot.update(
        {
            "clone_root": os.fspath(clone),
            "clone_project_root": os.fspath(clone_project),
            "cargo_cache_root": os.fspath(tmp_path / "slot-01-cargo"),
            "supervisor_artifact_path": os.fspath(
                clone_project
                / "artifacts/phase3_q05b_dual_qualification_v2.json"
            ),
            "artifact_parent_root": os.fspath(artifact_parent),
            "artifact_path": os.fspath(
                artifact_parent / "phase3_q05b_dual_qualification_v2.json"
            ),
            "scheduling_probe_root": os.fspath(
                tmp_path / "slot-01-scheduling-probe"
            ),
            "evidence_root": os.fspath(evidence),
            "trace_root": os.fspath(trace),
            "stdout_path": os.fspath(evidence / "supervisor.stdout"),
            "stderr_path": os.fspath(evidence / "supervisor.stderr"),
            "external_trajectory_binding_path": os.fspath(
                evidence / "trajectory-binding.json"
            ),
            "slot_result_path": os.fspath(evidence / "slot-result.json"),
        }
    )

    calls: list[tuple[str, list[str], dict[str, str]]] = []
    real_run_exact_command = TOOL._run_exact_command

    def recording_run_exact_command(
        argv: Sequence[str],
        **kwargs: object,
    ) -> bytes:
        environment = kwargs["env"]
        assert type(environment) is dict
        calls.append(
            (
                str(kwargs["label"]),
                list(argv),
                dict(environment),
            )
        )
        return real_run_exact_command(argv, **kwargs)

    monkeypatch.setattr(TOOL, "_run_exact_command", recording_run_exact_command)
    identity = TOOL._prepare_clone(contract, slot)
    assert identity["head"] == target_commit
    assert identity["tree"] == target_tree
    assert identity["parent"] == parent
    expected_target_ref = str(execution["clone_target_ref_name"])
    exact_ref = subprocess.run(
        [
            git,
            "-C",
            os.fspath(clone),
            "for-each-ref",
            "--format=%(refname)%00%(objectname)",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert exact_ref.stdout == (
        expected_target_ref + "\x00" + target_commit + "\n"
    ).encode("ascii")
    assert exact_ref.stderr == b""
    replay_identity = TOOL._verify_clone(contract, slot, artifact_state="ABSENT")
    assert replay_identity == identity

    clone_graph = (clone / ".git/objects/info/commit-graph").read_bytes()
    assert bytes.fromhex(source_only_ref) not in clone_graph
    assert bytes.fromhex(source_only_stash) not in clone_graph
    for stale_oid in (source_only_ref, source_only_stash):
        absent = subprocess.run(
            [git, "-C", os.fspath(clone), "cat-file", "-e", stale_oid],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert absent.returncode != 0
        assert absent.stdout == b""
    for command in (
        (git, "-C", os.fspath(clone), "commit-graph", "verify"),
        (
            git,
            "-C",
            os.fspath(clone),
            "fsck",
            "--full",
            "--strict",
            "--unreachable",
            "--no-reflogs",
        ),
        (
            git,
            "-c",
            "core.commitGraph=false",
            "-C",
            os.fspath(clone),
            "fsck",
            "--full",
            "--strict",
            "--unreachable",
            "--no-reflogs",
        ),
    ):
        replay = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert replay.returncode == 0
        assert replay.stdout == b""
        assert replay.stderr == b""

    assert calls
    for label, argv, environment in calls:
        assert TOOL._bounded_command_binding_is_exact_v2(contract, label, argv)
        assert environment["GIT_ALLOW_PROTOCOL"] == "file"
        assert environment["GIT_PROTOCOL_FROM_USER"] == "0"
        assert environment["GIT_SSH_COMMAND"] == "/bin/false"
    clone_call = next(argv for label, argv, _env in calls if label == "slot no-local clone")
    assert "--no-local" in clone_call
    assert "--local" not in clone_call


def _success_wire(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
) -> dict[str, object]:
    stage_roots = [_raw(f"stage-{index}") for index in range(1, 11)]
    result: dict[str, object] = {
        "artifact_path": slot["supervisor_artifact_path"],
        "artifact_sha256": _raw("artifact"),
        "artifact_set_root": _raw("artifact-set"),
        "candidate_receipt_root": _raw("candidate-receipt"),
        "final_receipt_root": _raw("final-receipt"),
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "actual_admission_prepublication_live_replay_root": _raw(
            "prepublication-live"
        ),
        "actual_admission_pre_artifact_live_replay_root": _raw(
            "pre-artifact-live"
        ),
        "actual_admission_stage10_live_replay_root": _raw("stage10-live"),
        "actual_admission_postpublication_live_replay_root": _raw(
            "postpublication-live"
        ),
        "actual_final_delivery_identity": None,
        "actual_final_delivery_root": "0" * 64,
        "publisher_handle_close_status": "CLOSED",
        "stage_roots": stage_roots,
    }
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    identity = {
        "schema_version": "hegel-phase3a-q05b-final-delivery-identity/2",
        "handoff_point": (
            "FINAL_ANCHORED_REPLAY_AFTER_POSTPUBLICATION_LIVE_"
            "BEFORE_HANDLE_CLOSE_AND_RETURN"
        ),
        "source_commit": target["target_supervisor_commit"],
        "artifact_path": result["artifact_path"],
        "ordered_stage_root_rows": [
            [index, name, stage_roots[index - 1]]
            for index, name in enumerate(_STAGES, start=1)
        ],
        "actual_admission_evidence_root": _raw("actual-admission"),
        "pre_artifact_live_marker_replay_root": result[
            "actual_admission_pre_artifact_live_replay_root"
        ],
        "stage_10_live_marker_replay_root": result[
            "actual_admission_stage10_live_replay_root"
        ],
        "prepublication_live_marker_replay_root": result[
            "actual_admission_prepublication_live_replay_root"
        ],
        "postpublication_live_marker_replay_root": result[
            "actual_admission_postpublication_live_replay_root"
        ],
        "artifact_set_root": result["artifact_set_root"],
        "candidate_receipt_root": result["candidate_receipt_root"],
        "final_receipt_root": result["final_receipt_root"],
        "publisher_parent_device": 2049,
        "publisher_parent_inode": 100,
        "publisher_file_device": 2049,
        "publisher_file_inode": 101,
        "publisher_file_mode": 0o444,
        "publisher_file_nlink": 1,
        "payload_length": 4096,
        "payload_sha256": result["artifact_sha256"],
        "first_anchored_replay_sha256": result["artifact_sha256"],
        "final_path_matches_publisher_owned_inode": True,
        "same_uid_mutation_after_handoff_outside_claim": True,
    }
    result["actual_final_delivery_identity"] = identity
    result["actual_final_delivery_root"] = sha256(
        b"HEGEL/Q05B/ACTUAL/FINAL_DELIVERY/V1\x00"
        + CORE.canonical_campaign_json_bytes_v2(identity)
    ).hexdigest()
    return result


def _error_wire(detail: str) -> dict[str, object]:
    return {
        "schema_version": CORE.ERROR_WIRE_SCHEMA_V2,
        "status": CORE.GAP_ERROR_CODE_V2,
        "detail": detail,
        "qualification_predicate_count": 0,
        "qualification_predicate_mask": 0,
        "qualification_predicate_total": 20,
        "qualification_candidate_receipt": None,
        "qualification_final_receipt": None,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "receipt_created": False,
        "artifact_written": False,
    }


def _propagation_row(
    *,
    sequence: int,
    timestamp: int,
    operation: str,
    span_id: str,
    stage: str | None,
    substep: str | None,
    actor: str | None,
    detail: str,
    error_detail: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sequence": sequence,
        "timestamp_ns": timestamp,
        "span_id": span_id,
        "operation": operation,
        "stage_or_null": stage,
        "substep_or_null": substep,
        "actor_or_null": actor,
        "detail_or_null": detail,
        "error_code": CORE.GAP_ERROR_CODE_V2,
        "error_detail": error_detail,
        "record_digest": _tagged(f"failure-{sequence}"),
    }
    row.update({field: None for field in _SAMPLE_FIELDS})
    return row


def _held_final_gap_rows() -> tuple[str, list[dict[str, object]]]:
    previous = 1_000_000_000
    scheduled = previous + CORE.TARGET_WAIT_NS_V2
    actual = previous + CORE.MAXIMUM_GAP_NS_V2 + 1
    ordinal = 2
    detail = (
        "held final sampling gap exceeded bound: "
        "actor_id=RUST_ENDPOINT; role_id=2; sample_ordinal=2; "
        f"previous_finish_ns={previous}; actual_start_ns={actual}; "
        f"gap_ns={actual - previous}; "
        f"maximum_gap_ns={CORE.MAXIMUM_GAP_NS_V2}; "
        f"target_wait_ns={CORE.TARGET_WAIT_NS_V2}; "
        "previous_duration_ns=1000000"
    )
    origin: dict[str, object] = {
        "sequence": 5,
        "timestamp_ns": actual + 5,
        "span_id": f"held-final-sample-RUST_ENDPOINT-{ordinal:06d}",
        "operation": "HELD_FINAL_RESOURCE_SAMPLE",
        "stage_or_null": "ACTOR_LIFECYCLE",
        "substep_or_null": "COLLECT_BOUND_HELD_FINAL_RESOURCES",
        "actor_or_null": "RUST_ENDPOINT",
        "detail_or_null": "MAXIMUM_UNSAMPLED_GAP_EXCEEDED",
        "error_code": CORE.GAP_ERROR_CODE_V2,
        "error_detail": detail,
        "record_digest": _tagged("failure-5"),
        "role_id_or_null": "2",
        "sample_ordinal_or_null": ordinal,
        "previous_finish_ns_or_null": previous,
        "scheduled_start_ns_or_null": scheduled,
        "actual_start_ns_or_null": actual,
        "finish_ns_or_null": actual,
        "collection_duration_ns_or_null": 0,
        "scheduler_lateness_ns_or_null": actual - scheduled,
        "target_wait_ns_or_null": CORE.TARGET_WAIT_NS_V2,
        "gap_ns_or_null": actual - previous,
        "maximum_gap_ns_or_null": CORE.MAXIMUM_GAP_NS_V2,
        "gap_exceeded_or_null": True,
    }
    rows = [origin]
    rows.append(
        _propagation_row(
            sequence=6,
            timestamp=actual + 6,
            operation="HELD_ACTOR_CLOSE",
            span_id="actor-close-RUST_ENDPOINT",
            stage="ACTOR_LIFECYCLE",
            substep="DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            actor="RUST_ENDPOINT",
            detail="ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
            error_detail=detail,
        )
    )
    rows.append(
        _propagation_row(
            sequence=9,
            timestamp=actual + 9,
            operation="SUPERVISOR_STAGE",
            span_id="supervisor-stage-05",
            stage="STAGE_05",
            substep="SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            actor=None,
            detail="FAILED",
            error_detail=detail,
        )
    )
    rows.append(
        _propagation_row(
            sequence=10,
            timestamp=actual + 10,
            operation="SUPERVISOR_RUN",
            span_id="supervisor-run",
            stage="RUN",
            substep=None,
            actor=None,
            detail="supervisor attempt failed",
            error_detail=detail,
        )
    )
    return detail, rows


def _continuous_gap_rows() -> tuple[str, list[dict[str, object]]]:
    previous = 1_000_000_000
    scheduled = previous + CORE.TARGET_WAIT_NS_V2
    actual = previous + CORE.MAXIMUM_GAP_NS_V2 + 1
    raw_detail = (
        "held actor sampling gap exceeded bound: "
        "actor_id=RUST_ENDPOINT; role_id=2; sample_ordinal=1; "
        f"previous_finish_ns={previous}; scheduled_start_ns={scheduled}; "
        f"actual_start_ns={actual}; gap_ns={actual - previous}; "
        f"maximum_gap_ns={CORE.MAXIMUM_GAP_NS_V2}; "
        f"target_wait_ns={CORE.TARGET_WAIT_NS_V2}; "
        "previous_duration_ns=1000000"
    )
    propagated_detail = "held actor sampler failed: " + raw_detail
    origin: dict[str, object] = {
        "sequence": 6,
        "timestamp_ns": actual + 6,
        "span_id": "sample-RUST_ENDPOINT-000001",
        "operation": "LIVE_RESOURCE_SAMPLE",
        "stage_or_null": "ACTOR_LIFECYCLE",
        "substep_or_null": "COLLECT_BOUND_LIVE_RESOURCES",
        "actor_or_null": "RUST_ENDPOINT",
        "detail_or_null": (
            "MAXIMUM_UNSAMPLED_GAP_EXCEEDED;"
            f"native_wake_ns={scheduled};"
            "native_wake_lateness_ns=0;"
            f"trajectory_admission_duration_ns={actual - scheduled}"
        ),
        "error_code": CORE.GAP_ERROR_CODE_V2,
        "error_detail": raw_detail,
        "record_digest": _tagged("failure-6"),
        "role_id_or_null": "2",
        "sample_ordinal_or_null": 1,
        "previous_finish_ns_or_null": previous,
        "scheduled_start_ns_or_null": scheduled,
        "actual_start_ns_or_null": actual,
        "finish_ns_or_null": actual,
        "collection_duration_ns_or_null": 0,
        "scheduler_lateness_ns_or_null": actual - scheduled,
        "target_wait_ns_or_null": CORE.TARGET_WAIT_NS_V2,
        "gap_ns_or_null": actual - previous,
        "maximum_gap_ns_or_null": CORE.MAXIMUM_GAP_NS_V2,
        "gap_exceeded_or_null": True,
    }
    contexts = (
        (
            "LIVE_RESOURCE_SAMPLER",
            "sampler-RUST_ENDPOINT",
            "ACTOR_LIFECYCLE",
            "CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            "RUST_ENDPOINT",
            "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; "
            "sample_span=sample-RUST_ENDPOINT-000001",
            raw_detail,
        ),
        (
            "HELD_ACTOR_CLOSE",
            "actor-close-RUST_ENDPOINT",
            "ACTOR_LIFECYCLE",
            "DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            "RUST_ENDPOINT",
            "ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
            propagated_detail,
        ),
        (
            "SUPERVISOR_STAGE",
            "supervisor-stage-05",
            "STAGE_05",
            "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            None,
            "FAILED",
            propagated_detail,
        ),
        (
            "SUPERVISOR_RUN",
            "supervisor-run",
            "RUN",
            None,
            None,
            "supervisor attempt failed",
            propagated_detail,
        ),
    )
    rows = [origin]
    for sequence, context in zip((7, 8, 11, 12), contexts, strict=True):
        operation, span_id, stage, substep, actor, event_detail, error_detail = (
            context
        )
        rows.append(
            _propagation_row(
                sequence=sequence,
                timestamp=actual + sequence,
                operation=operation,
                span_id=span_id,
                stage=stage,
                substep=substep,
                actor=actor,
                detail=event_detail,
                error_detail=error_detail,
            )
        )
    return propagated_detail, rows


def _event_record(
    *,
    trace_id: str,
    sequence: int,
    timestamp: int,
    event_type: str,
    span_id: str,
    operation: str,
    stage: str | None,
    substep: str | None,
    actor: str | None,
    detail: str | None,
    error_code: str | None = None,
    error_detail: str | None = None,
    record_digest: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": TRAJECTORY.EVENT_SCHEMA_V2,
        "trace_id": trace_id,
        "producer_id": "Q05B_SUPERVISOR_V6",
        "clock_id": "CLOCK_MONOTONIC",
        "sequence": sequence,
        "timestamp_ns": timestamp,
        "event_type": event_type,
        "span_id": span_id,
        "operation": operation,
        "stage_or_null": stage,
        "substep_or_null": substep,
        "actor_or_null": actor,
        "detail_or_null": detail,
        "error_code_or_null": error_code,
        "error_detail_or_null": error_detail,
        "record_digest": record_digest
        or _tagged(f"event-record-{sequence}-{event_type}"),
    }
    value.update({field: None for field in _SAMPLE_FIELDS})
    return value


def _event_records_for_failures(
    trace_id: str,
    failure_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [deepcopy(dict(row)) for row in failure_rows]
    assert rows and rows[0]["sequence"] == len(rows) + 1
    origin = rows[0]
    timestamp_base = int(origin["timestamp_ns"]) - int(origin["sequence"])
    events: list[dict[str, object]] = []
    for sequence, failure in enumerate(reversed(rows), start=1):
        events.append(
            _event_record(
                trace_id=trace_id,
                sequence=sequence,
                timestamp=timestamp_base + sequence,
                event_type="BEGIN",
                span_id=str(failure["span_id"]),
                operation=str(failure["operation"]),
                stage=failure["stage_or_null"],
                substep=failure["substep_or_null"],
                actor=failure["actor_or_null"],
                detail="ENTER",
            )
        )
    failures_by_sequence = {int(row["sequence"]): row for row in rows}
    cleanup_sequence = int(origin["sequence"]) + 1 + len(
        [
            row
            for row in rows[1:]
            if row["operation"]
            in {"LIVE_RESOURCE_SAMPLER", "HELD_ACTOR_CLOSE"}
        ]
    )
    cleanup_span = f"actor-cleanup-{origin['actor_or_null']}-000001"
    for sequence in range(int(origin["sequence"]), int(rows[-1]["sequence"]) + 1):
        failure = failures_by_sequence.get(sequence)
        if failure is not None:
            event = _event_record(
                trace_id=trace_id,
                    sequence=sequence,
                    timestamp=int(failure["timestamp_ns"]),
                    event_type="FAIL",
                    span_id=str(failure["span_id"]),
                    operation=str(failure["operation"]),
                    stage=failure["stage_or_null"],
                    substep=failure["substep_or_null"],
                    actor=failure["actor_or_null"],
                    detail=failure["detail_or_null"],
                    error_code=str(failure["error_code"]),
                    error_detail=str(failure["error_detail"]),
                    record_digest=str(failure["record_digest"]),
                )
            for field in _SAMPLE_FIELDS:
                event[field] = deepcopy(failure[field])
            events.append(event)
        elif sequence == cleanup_sequence:
            events.append(
                _event_record(
                    trace_id=trace_id,
                    sequence=sequence,
                    timestamp=timestamp_base + sequence,
                    event_type="BEGIN",
                    span_id=cleanup_span,
                    operation="HELD_ACTOR_ABORT_CLEANUP",
                    stage="ACTOR_LIFECYCLE",
                    substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                    actor=str(origin["actor_or_null"]),
                    detail="ENTER",
                )
            )
        elif sequence == cleanup_sequence + 1:
            events.append(
                _event_record(
                    trace_id=trace_id,
                    sequence=sequence,
                    timestamp=timestamp_base + sequence,
                    event_type="OK",
                    span_id=cleanup_span,
                    operation="HELD_ACTOR_ABORT_CLEANUP",
                    stage="ACTOR_LIFECYCLE",
                    substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                    actor=str(origin["actor_or_null"]),
                    detail="cleanup_error_count=0",
                )
            )
        else:  # pragma: no cover - fixture grammar is exact.
            raise AssertionError(f"unfilled event sequence {sequence}")
    return events


def _trusted_host_stage6_gap_trace(
    trace_id: str,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    previous = 1_000_000_000
    scheduled = previous + CORE.TARGET_WAIT_NS_V2
    gap = 323_536_475
    actual = previous + gap
    detail = (
        "held final sampling gap exceeded bound: "
        "actor_id=TRUSTED_HOST_REPLAY; role_id=3; "
        "sample_ordinal=7163; "
        f"previous_finish_ns={previous}; actual_start_ns={actual}; "
        f"gap_ns={gap}; maximum_gap_ns={CORE.MAXIMUM_GAP_NS_V2}; "
        f"target_wait_ns={CORE.TARGET_WAIT_NS_V2}; "
        "previous_duration_ns=172792053"
    )
    origin: dict[str, object] = {
        "sequence": 5,
        "timestamp_ns": actual + 5,
        "span_id": "held-final-sample-TRUSTED_HOST_REPLAY-007163",
        "operation": "HELD_FINAL_RESOURCE_SAMPLE",
        "stage_or_null": "ACTOR_LIFECYCLE",
        "substep_or_null": "COLLECT_BOUND_HELD_FINAL_RESOURCES",
        "actor_or_null": "TRUSTED_HOST_REPLAY",
        "detail_or_null": "MAXIMUM_UNSAMPLED_GAP_EXCEEDED",
        "error_code": CORE.GAP_ERROR_CODE_V2,
        "error_detail": detail,
        "record_digest": _tagged("v2r14-trusted-host-failure-5"),
        "role_id_or_null": "3",
        "sample_ordinal_or_null": 7163,
        "previous_finish_ns_or_null": previous,
        "scheduled_start_ns_or_null": scheduled,
        "actual_start_ns_or_null": actual,
        "finish_ns_or_null": actual,
        "collection_duration_ns_or_null": 0,
        "scheduler_lateness_ns_or_null": actual - scheduled,
        "target_wait_ns_or_null": CORE.TARGET_WAIT_NS_V2,
        "gap_ns_or_null": gap,
        "maximum_gap_ns_or_null": CORE.MAXIMUM_GAP_NS_V2,
        "gap_exceeded_or_null": True,
    }
    rows = [
        origin,
        _propagation_row(
            sequence=6,
            timestamp=actual + 6,
            operation="HELD_ACTOR_CLOSE",
            span_id="actor-close-TRUSTED_HOST_REPLAY",
            stage="ACTOR_LIFECYCLE",
            substep="DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            actor="TRUSTED_HOST_REPLAY",
            detail="ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
            error_detail=detail,
        ),
        _propagation_row(
            sequence=11,
            timestamp=actual + 11,
            operation="SUPERVISOR_STAGE",
            span_id="supervisor-stage-06",
            stage="STAGE_06",
            substep="RUN_HELD_TRUSTED_HOST_SEMANTIC_REPLAY",
            actor=None,
            detail="FAILED",
            error_detail=detail,
        ),
        _propagation_row(
            sequence=16,
            timestamp=actual + 16,
            operation="SUPERVISOR_RUN",
            span_id="supervisor-run",
            stage="RUN",
            substep=None,
            actor=None,
            detail="supervisor attempt failed",
            error_detail=detail,
        ),
    ]
    events = [
        _event_record(
            trace_id=trace_id,
            sequence=sequence,
            timestamp=actual + sequence,
            event_type="BEGIN",
            span_id=str(failure["span_id"]),
            operation=str(failure["operation"]),
            stage=failure["stage_or_null"],
            substep=failure["substep_or_null"],
            actor=failure["actor_or_null"],
            detail="ENTER",
        )
        for sequence, failure in enumerate(reversed(rows), start=1)
    ]
    cleanup_rows: dict[int, dict[str, object]] = {}
    for sequence, actor, suffix in (
        (7, "TRUSTED_HOST_REPLAY", 1),
        (9, "TRUSTED_HOST_REPLAY", 2),
        (12, "PYTHON_ENDPOINT", 3),
        (14, "RUST_ENDPOINT", 4),
    ):
        span_id = f"actor-cleanup-{actor}-{suffix:06d}"
        cleanup_rows[sequence] = _event_record(
            trace_id=trace_id,
            sequence=sequence,
            timestamp=actual + sequence,
            event_type="BEGIN",
            span_id=span_id,
            operation="HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor=actor,
            detail="ENTER",
        )
        cleanup_rows[sequence + 1] = _event_record(
            trace_id=trace_id,
            sequence=sequence + 1,
            timestamp=actual + sequence + 1,
            event_type="OK",
            span_id=span_id,
            operation="HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor=actor,
            detail="cleanup_error_count=0",
        )
    failures_by_sequence = {int(row["sequence"]): row for row in rows}
    for sequence in range(5, 17):
        failure = failures_by_sequence.get(sequence)
        if failure is not None:
            event = _event_record(
                trace_id=trace_id,
                sequence=sequence,
                timestamp=int(failure["timestamp_ns"]),
                event_type="FAIL",
                span_id=str(failure["span_id"]),
                operation=str(failure["operation"]),
                stage=failure["stage_or_null"],
                substep=failure["substep_or_null"],
                actor=failure["actor_or_null"],
                detail=failure["detail_or_null"],
                error_code=str(failure["error_code"]),
                error_detail=str(failure["error_detail"]),
                record_digest=str(failure["record_digest"]),
            )
            for field in _SAMPLE_FIELDS:
                event[field] = deepcopy(failure[field])
            events.append(event)
        else:
            cleanup = cleanup_rows.get(sequence)
            assert cleanup is not None
            events.append(cleanup)
    return detail, rows, events


def _minimal_event_records(
    trace_id: str,
    outcome: str,
    failure_rows: Sequence[Mapping[str, object]],
    terminal_error_detail: str | None,
) -> list[dict[str, object]]:
    if failure_rows:
        return _event_records_for_failures(trace_id, failure_rows)
    begin = _event_record(
        trace_id=trace_id,
        sequence=1,
        timestamp=1,
        event_type="BEGIN",
        span_id="supervisor-run",
        operation="SUPERVISOR_RUN",
        stage="RUN",
        substep=None,
        actor=None,
        detail="ENTER",
    )
    end = _event_record(
        trace_id=trace_id,
        sequence=2,
        timestamp=2,
        event_type=outcome,
        span_id="supervisor-run",
        operation="SUPERVISOR_RUN",
        stage="RUN",
        substep=None,
        actor=None,
        detail="COMPLETE" if outcome == "OK" else "supervisor attempt failed",
        error_code=None if outcome == "OK" else "FAIL_TEST_SUPERVISOR",
        error_detail=None
        if outcome == "OK"
        else (terminal_error_detail or "test supervisor failure"),
    )
    return [begin, end]


def _trajectory(
    slot: Mapping[str, object],
    *,
    outcome: str,
    failure_rows: Sequence[Mapping[str, object]] = (),
    terminal_error_detail: str | None = None,
) -> dict[str, object]:
    status = f"COMPLETE_{outcome}"
    effective_failures = [deepcopy(dict(row)) for row in failure_rows]
    if outcome == "FAIL" and not effective_failures:
        effective_failures = [
            {
                "sequence": 2,
                "timestamp_ns": 2,
                "span_id": "supervisor-run",
                "operation": "SUPERVISOR_RUN",
                "stage_or_null": "RUN",
                "substep_or_null": None,
                "actor_or_null": None,
                "detail_or_null": "supervisor attempt failed",
                "error_code": "FAIL_TEST_SUPERVISOR",
                "error_detail": terminal_error_detail or "test supervisor failure",
                **{field: None for field in _SAMPLE_FIELDS},
                "record_digest": _tagged("generic-supervisor-fail"),
            }
        ]
    event_records = _minimal_event_records(
        str(slot["trajectory_trace_id"]),
        outcome,
        effective_failures,
        terminal_error_detail,
    )
    contract = _contract()
    payloads = [TOOL._canonical_json_bytes(event) for event in event_records]
    partition = TOOL._trajectory_partition_evidence(
        contract,
        payloads,
        trajectory_status=status,
        failure_rows_in_sequence_order=effective_failures,
    )
    return {
        "verification_status": status,
        "complete": True,
        "complete_verification_passed": True,
        "verification_error_or_null": None,
        "trace_id_or_null": slot["trajectory_trace_id"],
        "producer_id_or_null": "Q05B_SUPERVISOR_V6",
        "event_count": len(event_records),
        "first_event_record_digest_or_null": event_records[0]["record_digest"],
        "final_event_record_digest_or_null": event_records[-1]["record_digest"],
        "terminal_outcome_or_null": outcome,
        "terminal_manifest_digest_or_null": _tagged("terminal-manifest"),
        "terminal_manifest_raw_sha256_or_null": _raw("terminal-manifest-raw"),
        "clock_id_or_null": "CLOCK_MONOTONIC",
        "closed_span_count_or_null": 4,
        "terminal_event_type_or_null": outcome,
        "terminal_error_code_or_null": (
            CORE.GAP_ERROR_CODE_V2 if outcome == "FAIL" else None
        ),
        "terminal_error_detail_or_null": terminal_error_detail,
        "torn_tail_staging_basename_or_null": None,
        "open_span_ids_in_begin_order": [],
        "open_span_contexts_in_begin_order": [],
        "failure_rows_retained_in_sequence_order": deepcopy(
            partition["failure_rows_retained_in_sequence_order"]
        ),
        "cleanup_rows_retained_in_sequence_order": deepcopy(
            partition["cleanup_rows_retained_in_sequence_order"]
        ),
        "partition_certificate_or_null": deepcopy(
            partition["partition_certificate_or_null"]
        ),
        "last_durable_event_row_or_null": deepcopy(
            partition["last_durable_event_row_or_null"]
        ),
        "canonical_event_payloads_digest": partition[
            "canonical_event_payloads_digest"
        ],
        "success_trace_bindings_or_null": (
            {
                "stage_roots": [
                    _raw(f"stage-{index}") for index in range(1, 11)
                ],
                "adapter_artifact_sha256": _raw("artifact"),
                "prepublication_live_marker_replay_root": _raw(
                    "prepublication-live"
                ),
                "publication_artifact_sha256": _raw("artifact"),
            }
            if outcome == "OK"
            else None
        ),
    }


def _process(stdout: bytes, return_code: int) -> dict[str, object]:
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": b"",
        "stdout_overflow": False,
        "stderr_overflow": False,
        "timed_out": False,
        "child_reaped": True,
        "process_group_absent": True,
        "wrapper_resources_closed": True,
    }


def _target_fsck_pending_error(
    contract: Mapping[str, object],
) -> object:
    result = _process(b"", -9)
    result["timed_out"] = True
    receipt = TOOL._bounded_process_diagnostic(
        TOOL._TARGET_FSCK_ADMISSION_LABEL_V2R17,
        TOOL._target_object_closure_fsck_argv_v2r17(contract),
        result,
        classification={
            "accepted": False,
            "phase": "TARGET_LOCAL_OBJECT_CLOSURE_FSCK",
            "condition": "READONLY_COMMAND_TIMEOUT",
            "elapsed_ns": 300_000_000_000,
            "timeout_seconds": 300,
            "disposition": "PENDING_REPAIRABLE_PRECONSUMPTION",
        },
    )
    return TOOL.ShadowRetryCampaignPendingV1(
        TOOL._TARGET_FSCK_PENDING_CODE_V2R17,
        "injected exact clean target fsck timeout",
        phase="TARGET_LOCAL_OBJECT_CLOSURE_FSCK",
        condition="READONLY_COMMAND_TIMEOUT",
        receipt=receipt,
        dependency_gate_invocation_count=1,
    )


def _valid_artifacts(
    wire: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    identity = wire["actual_final_delivery_identity"]
    assert type(identity) is dict
    supervisor = {
        "state": "VALID",
        "raw_sha256_or_null": wire["artifact_sha256"],
        "size_or_null": 4096,
        "replay_passed": True,
        "summary_or_null": {
            "canonical_sha256": wire["artifact_sha256"],
            "artifact_set_root": wire["artifact_set_root"],
            "candidate_receipt_root": wire["candidate_receipt_root"],
            "final_receipt_root": wire["final_receipt_root"],
            "qualification_count": wire["qualification_count"],
            "qualification_mask": wire["qualification_mask"],
            "closed_q1_authority": {
                "state": wire["q1_state"],
                "gate_count": wire["q1_gate_count"],
                "gate_mask": wire["q1_gate_mask"],
                "formal_output_roots": wire["q1_formal_output_roots"],
            },
            "actual_admission_source_commit": identity["source_commit"],
            "actual_admission_artifact_path": wire["artifact_path"],
            "pre_artifact_live_marker_replay_root": wire[
                "actual_admission_pre_artifact_live_replay_root"
            ],
            "actual_admission_evidence_root": identity[
                "actual_admission_evidence_root"
            ],
        },
        "error_or_null": None,
        "parent_identity_or_null": {
            "dev": 2049,
            "ino": 100,
            "mount_id": 7,
            "mode": stat.S_IFDIR | 0o755,
            "uid": os.geteuid(),
            "nlink": 2,
        },
        "file_identity_or_null": {
            "dev": 2049,
            "ino": 101,
            "mode": 0o444,
            "nlink": 1,
            "uid": os.geteuid(),
            "size": 4096,
        },
    }
    external = {
        "state": "VALID",
        "raw_sha256_or_null": wire["artifact_sha256"],
        "size_or_null": 4096,
        "replay_passed": True,
        "error_or_null": None,
        "parent_identity_or_null": {
            "dev": 2049,
            "ino": 200,
            "mount_id": 7,
            "mode": stat.S_IFDIR | 0o700,
            "uid": os.geteuid(),
            "nlink": 2,
        },
        "file_identity_or_null": {
            "dev": 2049,
            "ino": 201,
            "mode": 0o444,
            "nlink": 1,
            "uid": os.geteuid(),
            "size": 4096,
        },
    }
    return supervisor, external


def _absent_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    supervisor = {
        "state": "ABSENT",
        "raw_sha256_or_null": None,
        "size_or_null": None,
        "replay_passed": False,
        "summary_or_null": None,
        "error_or_null": None,
        "parent_identity_or_null": None,
        "file_identity_or_null": None,
    }
    external = {
        "state": "ABSENT",
        "raw_sha256_or_null": None,
        "size_or_null": None,
        "replay_passed": False,
        "error_or_null": None,
        "parent_identity_or_null": None,
        "file_identity_or_null": None,
    }
    return supervisor, external


def _source_replay(label: str) -> dict[str, object]:
    evidence = {"label": label}
    return {
        "certain": True,
        "error_or_null": None,
        "evidence": evidence,
        "root": TOOL._domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R20/SOURCE_REPLAY/V2",
            evidence,
        ),
    }


def _future_freshness() -> dict[str, object]:
    value: dict[str, object] = {
        "label": "future slots",
        "rows": [],
        "error_or_null": None,
        "certain": True,
        "root": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("root")
    value["root"] = TOOL._domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R20/SLOT_ROOT_ABSENCE/V2",
        unsigned,
    )
    return value


def _real_directory_stable_identity(path: Path) -> dict[str, int]:
    descriptor, _status = TOOL._open_directory_nofollow(
        path,
        f"test stable identity for {path}",
    )
    try:
        identity = TOOL._verify_held_directory_path(
            path,
            descriptor,
            f"test stable identity for {path}",
        )
    finally:
        os.close(descriptor)
    return TOOL._freshness_parent_stable_identity(identity)


def _semantic_capsule(
    contract: Mapping[str, object],
    *,
    real_parent_paths: Sequence[Path] = (),
) -> dict[str, object]:
    parents: dict[str, dict[str, int]] = {}
    slots = contract["slot_registry"]
    assert type(slots) is list
    for slot in slots:
        assert type(slot) is dict
        for field in (
            "clone_root",
            "cargo_cache_root",
            "artifact_parent_root",
            "scheduling_probe_root",
            "evidence_root",
            "trace_root",
        ):
            parent = os.fspath(Path(str(slot[field])).parent)
            parents.setdefault(
                parent,
                {
                    "dev": 2049,
                    "ino": 10_000 + len(parents),
                    "mount_id": 7,
                    "mode": stat.S_IFDIR | 0o700,
                    "uid": os.geteuid(),
                },
            )
    for path in real_parent_paths:
        parents[os.fspath(path)] = _real_directory_stable_identity(path)
    capsule = _capsule_stub("semantic-capsule", contract=contract)
    capsule.update({
        "freshness_parent_identities": parents,
    })
    return capsule


def _scheduling_probe_cleanup_receipt(
    capsule: Mapping[str, object],
    slot: Mapping[str, object],
) -> dict[str, object]:
    path = Path(str(slot["scheduling_probe_root"]))
    registry = capsule["freshness_parent_identities"]
    assert type(registry) is dict
    expected = deepcopy(registry[os.fspath(path.parent)])
    assert type(expected) is dict
    parent_identity = {**expected, "nlink": 2}
    root_identity = {
        "dev": expected["dev"],
        "ino": 90_000 + int(slot["slot_ordinal"]),
        "mount_id": expected["mount_id"],
        "mode": stat.S_IFDIR | 0o700,
        "uid": os.geteuid(),
        "nlink": 2,
    }
    receipt: dict[str, object] = {
        "schema_version": TOOL._SCHEDULING_PROBE_CLEANUP_SCHEMA_V1,
        "slot_id": slot["slot_id"],
        "probe_root_path": os.fspath(path),
        "expected_parent_stable_identity": expected,
        "parent_identity_before": deepcopy(parent_identity),
        "parent_identity_after_or_null": deepcopy(parent_identity),
        "probe_root_identity": root_identity,
        "same_filesystem_as_parent": True,
        "empty_before_removal": True,
        "rmdir_completed": True,
        "parent_fsync_completed": True,
        "pathname_absent": True,
        "root_descriptor_closed": True,
        "parent_descriptor_closed": True,
        "cleanup_errors": [],
        "passed": True,
    }
    assert TOOL._scheduling_probe_cleanup_receipt_exact_v2r17(
        capsule,
        slot,
        receipt,
    )
    return receipt


def _slot_absence_evidence(
    capsule: Mapping[str, object],
    slots: Sequence[Mapping[str, object]],
    label: str,
) -> dict[str, object]:
    registry = capsule["freshness_parent_identities"]
    assert type(registry) is dict
    rows: list[dict[str, object]] = []
    for slot in slots:
        for field in (
            "clone_root",
            "cargo_cache_root",
            "artifact_parent_root",
            "evidence_root",
            "trace_root",
        ):
            target = Path(str(slot[field]))
            parent = os.fspath(target.parent)
            stable = deepcopy(registry[parent])
            assert type(stable) is dict
            full = {**stable, "nlink": 2}
            rows.append(
                {
                    "slot_id": slot["slot_id"],
                    "field": field,
                    "target_path": os.fspath(target),
                    "state_parent_path": parent,
                    "expected_state_parent_stable_identity": deepcopy(stable),
                    "observed_state_parent_identity": deepcopy(full),
                    "observed_state_parent_stable_identity": deepcopy(stable),
                    "final_state_parent_identity": deepcopy(full),
                    "final_state_parent_stable_identity": deepcopy(stable),
                    "target_present_identity_or_null": None,
                    "absent_and_anchored": True,
                }
            )
    value: dict[str, object] = {
        "label": label,
        "rows": rows,
        "error_or_null": None,
        "certain": True,
        "root": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("root")
    value["root"] = TOOL._domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R20/SLOT_ROOT_ABSENCE/V2",
        unsigned,
    )
    return value


def _cleanup_replay() -> dict[str, object]:
    preimage = {"cleanup": True}
    return {
        "preimage": preimage,
        "root": TOOL._domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R20/CLEANUP_REPLAY/V2",
            preimage,
        ),
    }


def test_domain_digest_excludes_transport_lf_and_agrees_with_pure_core() -> None:
    domain = "HEGEL/TEST/Q05B/CAMPAIGN/V2"
    value = {"a": [True, None, 7], "z": "snow"}
    observed = TOOL._domain_digest(domain, value)
    expected = CORE.campaign_record_digest_v2(domain + "\x00", value)
    including_lf = "sha256:" + sha256(
        domain.encode("ascii")
        + b"\x00"
        + TOOL._canonical_json_bytes(value)
    ).hexdigest()
    assert observed == expected
    assert observed != including_lf


def test_fixed_system_executables_are_root_owned_read_only_capsule_inputs() -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    assert type(execution) is dict
    for field in (
        "supervisor_interpreter",
        "git_executable",
        "docker_executable",
    ):
        identity = TOOL._executable_identity(str(execution[field]), field)
        assert identity["uid"] == 0
        assert int(identity["mode"]) & 0o022 == 0
        assert int(identity["nlink"]) == 1


def test_python310_dependency_preflight_real_isolated_child_kat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, target_git, controller_git, python_identity = (
        _dependency_preflight_inputs()
    )
    real_bounded_process = TOOL._bounded_process
    observed: dict[str, object] = {}

    def capture(
        argv: Sequence[str],
        **kwargs: object,
    ) -> dict[str, object]:
        observed["argv"] = list(argv)
        observed.update(kwargs)
        return real_bounded_process(argv, **kwargs)

    monkeypatch.setattr(TOOL, "_bounded_process", capture)
    receipt = TOOL._python310_dependency_preflight_v1(
        contract,
        target_git=target_git,
        controller_git=controller_git,
        python_identity=python_identity,
    )
    policy = contract["execution_policy"][
        "python310_dependency_preflight_gate"
    ]
    assert observed["argv"][:9] == list(
        TOOL._DEPENDENCY_PREFLIGHT_ARGV_PREFIX_V1
    )
    assert observed["argv"][9] == TOOL._PYTHON310_DEPENDENCY_PREFLIGHT_PROGRAM_V1
    assert observed["cwd"] == "/"
    assert observed["timeout_seconds"] == policy["timeout_seconds"]
    assert receipt["status"] == "PASS"
    assert receipt["vendored_tomli_module_name"] == (
        "hegel_machine._vendor.tomli"
    )
    assert receipt["artifact_module_max_canonical_bytes"] == 2_147_483_648
    assert receipt["supervisor_tool_max_canonical_bytes"] == 2_147_483_648
    assert receipt["child_receipt"][
        "artifact_module_max_canonical_bytes"
    ] == 2_147_483_648
    assert receipt["child_receipt"][
        "supervisor_tool_max_canonical_bytes"
    ] == 2_147_483_648
    assert receipt["artifact_schema_version"] == (
        "hegel-phase3a-q05b-actual-artifact/3"
    )
    assert receipt["resource_transcript_ref_key"] == "resource_transcript_ref"
    assert receipt["held_final_sample_ref_key"] == "held_final_sample_ref"
    assert receipt["resource_reference_kat"] == {
        "resource_table_row_count": 3,
        "resource_transcript_index_count": 3,
        "held_final_sample_index_count": 3,
        "projected": policy["resource_reference_kat_projected"],
        "round_trip_exact": True,
        "stream_root_domain": policy["resource_stream_root_kat_domain"],
        "stream_root": policy["resource_stream_root_kat"],
        "canonical_root_matches_streaming": True,
    }
    assert receipt["tomli_top_level_spec_or_null"] is None
    assert receipt["top_level_tomli_in_sys_modules"] is False
    assert receipt["python_flags"][
        "pycache_prefix_absent_before_import"
    ] is True
    assert receipt["python_flags"][
        "pycache_prefix_absent_after_import"
    ] is True
    assert receipt["vendor_closure_digest"] == (
        TOOL._VENDORED_TOMLI_CLOSURE_DIGEST_V1
    )
    assert receipt["host_source_allowlist_count"] == 55
    assert receipt["loaded_vendored_modules"] == list(
        TOOL._DEPENDENCY_PREFLIGHT_LOADED_VENDORED_MODULES_V1
    )
    assert receipt["cargo_lock_parse"]["package_count"] == 29
    assert receipt["cargo_lock_parse"]["registry_package_count"] == 21
    unsigned = deepcopy(receipt)
    observed_digest = unsigned.pop("receipt_digest")
    assert observed_digest == TOOL._domain_digest(
        TOOL._PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_DOMAIN_V1,
        unsigned,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("timed_out", True),
        ("stdout_overflow", True),
        ("child_reaped", False),
        ("process_group_absent", False),
        ("wrapper_resources_closed", False),
        ("stderr", b"unexpected\n"),
    ),
)
def test_python310_dependency_preflight_process_failure_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    contract, target_git, controller_git, python_identity = (
        _dependency_preflight_inputs()
    )
    real_bounded_process = TOOL._bounded_process

    def corrupted(
        argv: Sequence[str],
        **kwargs: object,
    ) -> dict[str, object]:
        result = real_bounded_process(argv, **kwargs)
        result[field] = replacement
        return result

    monkeypatch.setattr(TOOL, "_bounded_process", corrupted)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._python310_dependency_preflight_v1(
            contract,
            target_git=target_git,
            controller_git=controller_git,
            python_identity=python_identity,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"


@pytest.mark.parametrize(
    "mutation",
    (
        "vendor",
        "allowlist",
        "loaded-module",
        "cargo-lock",
        "pycache",
        "artifact-module-cap",
        "supervisor-tool-cap",
        "artifact-schema",
        "resource-ref-key",
        "resource-stream-root",
        "whitespace",
        "multiple-lf",
    ),
)
def test_python310_dependency_preflight_child_receipt_drift_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    contract, target_git, controller_git, python_identity = (
        _dependency_preflight_inputs()
    )
    real_bounded_process = TOOL._bounded_process

    def corrupted(
        argv: Sequence[str],
        **kwargs: object,
    ) -> dict[str, object]:
        result = real_bounded_process(argv, **kwargs)
        raw = result["stdout"]
        assert type(raw) is bytes
        child = TOOL._strict_json(raw, "dependency corruption KAT")
        assert type(child) is dict
        if mutation == "vendor":
            child["vendor_closure_rows"][0][2] = "0" * 64
        elif mutation == "allowlist":
            child["host_source_allowlist_count"] = 54
        elif mutation == "loaded-module":
            child["loaded_vendored_modules"].append("tomli")
        elif mutation == "cargo-lock":
            child["cargo_lock"]["raw_sha256"] = "0" * 64
        elif mutation == "pycache":
            child["python_flags"][
                "pycache_prefix_absent_after_import"
            ] = False
        elif mutation == "artifact-module-cap":
            child["artifact_module_max_canonical_bytes"] = 67_108_864
        elif mutation == "supervisor-tool-cap":
            child["supervisor_tool_max_canonical_bytes"] = 2_147_483_649
        elif mutation == "artifact-schema":
            child["artifact_schema_version"] = (
                "hegel-phase3a-q05b-actual-artifact/2"
            )
        elif mutation == "resource-ref-key":
            child["resource_transcript_ref_key"] = "forged_ref"
        elif mutation == "resource-stream-root":
            child["resource_reference_kat"]["stream_root"] = "0" * 64
        result["stdout"] = TOOL._canonical_json_bytes(child)
        if mutation == "whitespace":
            result["stdout"] = b" " + result["stdout"]
        elif mutation == "multiple-lf":
            result["stdout"] += b"\n"
        return result

    monkeypatch.setattr(TOOL, "_bounded_process", corrupted)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._python310_dependency_preflight_v1(
            contract,
            target_git=target_git,
            controller_git=controller_git,
            python_identity=python_identity,
        )
    assert error.value.code in {
        "FAIL_CAMPAIGN_SOURCE",
        "FAIL_CAMPAIGN_CANONICAL",
    }


def test_python310_dependency_policy_and_target_provenance_are_exact() -> None:
    contract, target_git, controller_git, python_identity = (
        _dependency_preflight_inputs()
    )
    drifted_policy = deepcopy(contract)
    drifted_policy["execution_policy"][
        "python310_dependency_preflight_gate"
    ]["timeout_seconds"] = True
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._python310_dependency_preflight_policy_v1(drifted_policy)
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"

    for field, value in (
        ("commit", "0" * 40),
        ("target_source_binding_digest", _tagged("drifted-target")),
    ):
        drifted_target = deepcopy(target_git)
        drifted_target[field] = value
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._python310_dependency_preflight_v1(
                contract,
                target_git=drifted_target,
                controller_git=controller_git,
                python_identity=python_identity,
            )
        assert error.value.code == "FAIL_CAMPAIGN_SOURCE"

    drifted_object = deepcopy(target_git)
    drifted_object["cargo_lock_git_object"]["object_spec"] = "forged"
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._python310_dependency_preflight_v1(
            contract,
            target_git=drifted_object,
            controller_git=controller_git,
            python_identity=python_identity,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"


@pytest.mark.parametrize("value", (67_108_864, 2_147_483_649, True))
def test_artifact_capacity_contract_rejects_old_overflow_and_bool_values(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    assert type(execution) is dict
    assert type(execution["artifact_max_canonical_bytes"]) is int
    assert execution["artifact_max_canonical_bytes"] == 2_147_483_648
    execution["artifact_max_canonical_bytes"] = value
    monkeypatch.setattr(
        TOOL,
        "_pretty_json",
        lambda _raw, _label: deepcopy(contract),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"
    assert error.value.detail == "campaign authority boundary differs"


def test_artifact_capacity_is_used_only_at_five_outer_artifact_io_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap_name = "_ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES_V2R20"
    assert TOOL._ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES_V2R20 == 2_147_483_648
    assert TOOL._MAX_JSON_BYTES == 67_108_864
    assert CORE.CAPTURE_MAX_BYTES_V2 == 67_108_864
    assert _contract()["execution_policy"]["stdout_max_bytes"] == 67_108_864
    assert _contract()["execution_policy"]["stderr_max_bytes"] == 67_108_864
    assert inspect.getsource(TOOL._artifact_replay).count(cap_name) == 1
    assert inspect.getsource(TOOL._publish_external_artifact).count(cap_name) == 3
    assert inspect.getsource(TOOL._replay_external_artifact).count(cap_name) == 1
    assert inspect.getsource(TOOL._build_capsule).count(
        '"artifact_max_canonical_bytes"'
    ) == 1

    contract = _contract()
    dependency_receipt = {"receipt_digest": _tagged("dependency-cap-bound")}
    monkeypatch.setattr(
        TOOL,
        "_validate_dependency_receipt_v1",
        lambda *_args, **_kwargs: deepcopy(dependency_receipt),
    )
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    controller_git = {
        "commit": "de" * 20,
        "parent": target["target_supervisor_commit"],
        "parent_count": 1,
        "clean": True,
        "changed_paths_from_target_in_exact_order": list(
            TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        ),
    }
    capsule = TOOL._build_capsule(
        contract,
        controller_git=controller_git,
        target_git={"commit": "ad" * 20},
        cargo_identity={"manifest_digest": _tagged("cargo-cap-bound")},
        executable_identities={},
        python310_dependency_preflight_receipt=dependency_receipt,
        preauthorization_live_gauntlet=_gauntlet_receipt_binding_stub(
            contract,
            controller_git,
        ),
        slots=deepcopy(contract["slot_registry"]),
        freshness_parent_identities={},
    )
    resource_bounds = capsule["resource_bounds"]
    assert type(resource_bounds) is dict
    assert resource_bounds["artifact_max_canonical_bytes"] == 2_147_483_648
    unsigned = deepcopy(capsule)
    capsule_digest = unsigned.pop("capsule_digest")
    assert capsule_digest == TOOL._domain_digest(
        contract["hash_domains"]["capsule"],
        unsigned,
    )


def test_target_binding_replay_is_light_and_does_not_repeat_full_fsck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "light target replay fixture",
    )
    assert type(contract) is dict
    observed_labels: list[str] = []
    real_git = TOOL._git

    def recording_git(
        contract_value: Mapping[str, object],
        arguments: Sequence[str],
        label: str,
        **kwargs: object,
    ) -> bytes:
        observed_labels.append(label)
        assert "fsck" not in arguments
        return real_git(
            contract_value,
            arguments,
            label,
            **kwargs,
        )

    monkeypatch.setattr(TOOL, "_git", recording_git)
    replay = TOOL._verify_target_git_objects(contract)
    assert replay["local_object_closure_fsck_passed"] is False
    assert TOOL._TARGET_FSCK_ADMISSION_LABEL_V2R17 not in observed_labels


def test_preflight_orders_dependency_capsule_fsck_cpu_then_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule_digest = _tagged("ordered-preflight-capsule")
    events: list[str] = []

    monkeypatch.setattr(
        TOOL,
        "_slot_registry",
        lambda value: deepcopy(value["slot_registry"]),
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(TOOL, "_verify_target_git_objects", lambda _value: {})
    monkeypatch.setattr(TOOL, "_controller_git_identity", lambda _value: {})
    monkeypatch.setattr(TOOL, "_scan_tree", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(TOOL, "_executable_identity", lambda *_args: {})

    def dependency(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("dependency")
        return {"receipt_digest": _tagged("ordered-dependency")}

    def gauntlet(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("live-gauntlet-replay")
        return {"receipt_digest": _tagged("ordered-live-gauntlet")}

    def build(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("capsule")
        return {"capsule_digest": capsule_digest}

    def fsck(*_args: object, **kwargs: object) -> dict[str, object]:
        events.append("fsck")
        assert kwargs["dependency_gate_invocation_count"] == 1
        return {"status": "PASS"}

    def cpu(*_args: object, **kwargs: object) -> dict[str, object]:
        events.append("cpu")
        assert kwargs["dependency_gate_invocation_count"] == 1
        assert kwargs["target_fsck_invocation_count"] == 1
        return {"status": "PASS"}

    def authorization(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("authorization-template")
        return {}

    monkeypatch.setattr(
        TOOL,
        "_load_preauthorization_live_gauntlet_receipt_v2",
        gauntlet,
    )
    monkeypatch.setattr(TOOL, "_python310_dependency_preflight_v1", dependency)
    monkeypatch.setattr(TOOL, "_build_capsule", build)
    monkeypatch.setattr(
        TOOL,
        "_target_object_closure_fsck_admission_v2r17",
        fsck,
    )
    monkeypatch.setattr(
        TOOL,
        "_host_cpu_topology_affinity_admission_v2r17",
        cpu,
    )
    monkeypatch.setattr(TOOL, "_authorization_template", authorization)

    context = TOOL._preflight_context(
        contract,
        expected_capsule_digest_or_null=capsule_digest,
        allow_existing_run_state=True,
    )
    assert events == [
        "live-gauntlet-replay",
        "dependency",
        "capsule",
        "fsck",
        "cpu",
        "authorization-template",
    ]
    assert context["capsule"]["capsule_digest"] == capsule_digest


def test_wrong_capsule_is_rejected_before_fsck_cpu_or_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    events: list[str] = []
    monkeypatch.setattr(
        TOOL,
        "_slot_registry",
        lambda value: deepcopy(value["slot_registry"]),
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(TOOL, "_verify_target_git_objects", lambda _value: {})
    monkeypatch.setattr(TOOL, "_controller_git_identity", lambda _value: {})
    monkeypatch.setattr(TOOL, "_scan_tree", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(TOOL, "_executable_identity", lambda *_args: {})
    monkeypatch.setattr(
        TOOL,
        "_python310_dependency_preflight_v1",
        lambda *_args, **_kwargs: events.append("dependency") or {},
    )
    monkeypatch.setattr(
        TOOL,
        "_build_capsule",
        lambda *_args, **_kwargs: events.append("capsule")
        or {"capsule_digest": _tagged("actual-capsule")},
    )

    def forbidden(label: str):
        def invoke(*_args: object, **_kwargs: object) -> object:
            events.append(label)
            raise AssertionError(f"wrong capsule reached {label}")

        return invoke

    monkeypatch.setattr(
        TOOL,
        "_target_object_closure_fsck_admission_v2r17",
        forbidden("fsck"),
    )
    monkeypatch.setattr(
        TOOL,
        "_host_cpu_topology_affinity_admission_v2r17",
        forbidden("cpu"),
    )
    monkeypatch.setattr(
        TOOL,
        "_authorization_template",
        forbidden("authorization-template"),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._preflight_context(
            contract,
            expected_capsule_digest_or_null=_tagged("wrong-capsule"),
            allow_existing_run_state=True,
        )
    assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"
    assert events == ["dependency", "capsule"]


def test_dynamic_host_cpu_receipt_is_not_a_capsule_digest_input() -> None:
    contract = _contract()
    policy = contract["execution_policy"][
        "host_cpu_topology_affinity_admission"
    ]
    assert policy["dynamic_receipt_is_capsule_digest_input"] is False
    signature = inspect.signature(TOOL._build_capsule)
    assert "host_cpu_topology_affinity_admission_receipt" not in signature.parameters
    assert "host_cpu_topology_affinity_admission_receipt" not in inspect.getsource(
        TOOL._build_capsule
    )


def test_exact_target_fsck_clean_timeout_is_typed_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    result = _process(b"", -9)
    result["timed_out"] = True
    observed: dict[str, object] = {}

    def bounded(*args: object, **kwargs: object) -> dict[str, object]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return deepcopy(result)

    monotonic = iter((10_000, 300_000_010_000))
    monkeypatch.setattr(TOOL, "_bounded_process", bounded)
    monkeypatch.setattr(TOOL.time, "monotonic_ns", lambda: next(monotonic))
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._target_object_closure_fsck_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
        )
    assert pending.value.phase == "TARGET_LOCAL_OBJECT_CLOSURE_FSCK"
    assert pending.value.condition == "READONLY_COMMAND_TIMEOUT"
    assert pending.value.dependency_gate_invocation_count == 1
    receipt = pending.value.receipt
    assert receipt["label"] == TOOL._TARGET_FSCK_ADMISSION_LABEL_V2R17
    assert receipt["argv"] == (
        TOOL._target_object_closure_fsck_argv_v2r17(contract)
    )
    assert receipt["stdout"]["byte_count"] == 0
    assert receipt["stderr"]["byte_count"] == 0
    assert receipt["classification"]["elapsed_ns"] == 300_000_000_000
    assert observed["kwargs"]["timeout_seconds"] == 300.0
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is True


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    (
        ("oserror", "OSError:injected affinity read failure"),
        ("invalid", "INVALID_OR_DUPLICATE_EFFECTIVE_CPU_SET"),
    ),
)
def test_host_cpu_original_affinity_read_failure_is_exact_pending(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_error: str,
) -> None:
    contract = _contract()

    def get_affinity(_pid: int) -> object:
        if mode == "oserror":
            raise OSError("injected affinity read failure")
        return [0, 0]

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(
        TOOL.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unreachable")),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert pending.value.condition == "SCHED_AFFINITY_API_UNAVAILABLE"
    assert pending.value.receipt["failure_operation_or_null"] == (
        "READ_ORIGINAL_EFFECTIVE_AFFINITY"
    )
    assert pending.value.receipt["failure_error_or_null"] == expected_error
    assert pending.value.receipt["affinity_mutation_attempted"] is False
    assert pending.value.receipt["restore_attempted"] is False
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is True


@pytest.mark.parametrize(
    "mutation",
    (
        "code",
        "phase",
        "condition",
        "dependency-count",
        "schema",
        "label",
        "argv",
        "timed-out",
        "stdout-byte-count",
        "stderr-byte-count",
        "stdout-raw-sha",
        "stderr-raw-sha",
        "stdout-overflow",
        "stderr-overflow",
        "child-reaped",
        "process-group",
        "resources-closed",
        "return-code-type",
        "classification-type",
        "classification-accepted",
        "classification-phase",
        "classification-condition",
        "classification-timeout",
        "classification-disposition",
        "classification-elapsed-type",
        "classification-elapsed-too-short",
        "classification-elapsed-negative",
    ),
)
def test_pending_classifier_rejects_every_nonexact_receipt_mutation(
    mutation: str,
) -> None:
    contract = _contract()
    pending = _target_fsck_pending_error(contract)
    assert isinstance(pending, TOOL.ShadowRetryCampaignPendingV1)
    receipt = pending.receipt
    if mutation == "code":
        pending.code = "PENDING_OTHER"
    elif mutation == "phase":
        pending.phase = "OTHER_PHASE"
    elif mutation == "condition":
        pending.condition = "OTHER_CONDITION"
    elif mutation == "dependency-count":
        pending.dependency_gate_invocation_count = 0
    elif mutation == "schema":
        receipt["schema_version"] = "other-schema/1"
    elif mutation == "label":
        receipt["label"] = "other label"
    elif mutation == "argv":
        receipt["argv"] = ["/usr/bin/git", "fsck"]
    elif mutation == "timed-out":
        receipt["timed_out"] = False
    elif mutation == "stdout-byte-count":
        receipt["stdout"]["byte_count"] = 1
    elif mutation == "stderr-byte-count":
        receipt["stderr"]["byte_count"] = 1
    elif mutation == "stdout-raw-sha":
        receipt["stdout"]["raw_sha256"] = "0" * 64
    elif mutation == "stderr-raw-sha":
        receipt["stderr"]["raw_sha256"] = "0" * 64
    elif mutation == "stdout-overflow":
        receipt["stdout_overflow"] = True
    elif mutation == "stderr-overflow":
        receipt["stderr_overflow"] = True
    elif mutation == "child-reaped":
        receipt["child_reaped"] = False
    elif mutation == "process-group":
        receipt["process_group_absent"] = False
    elif mutation == "resources-closed":
        receipt["wrapper_resources_closed"] = False
    elif mutation == "return-code-type":
        receipt["return_code"] = True
    elif mutation == "classification-type":
        receipt["classification"] = []
    elif mutation == "classification-accepted":
        receipt["classification"]["accepted"] = True
    elif mutation == "classification-phase":
        receipt["classification"]["phase"] = "OTHER_PHASE"
    elif mutation == "classification-condition":
        receipt["classification"]["condition"] = "OTHER_CONDITION"
    elif mutation == "classification-timeout":
        receipt["classification"]["timeout_seconds"] = 301
    elif mutation == "classification-disposition":
        receipt["classification"]["disposition"] = "TERMINAL"
    elif mutation == "classification-elapsed-type":
        receipt["classification"]["elapsed_ns"] = True
    elif mutation == "classification-elapsed-too-short":
        receipt["classification"]["elapsed_ns"] = 299_999_999_999
    else:
        assert mutation == "classification-elapsed-negative"
        receipt["classification"]["elapsed_ns"] = -1
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending,
    ) is False


def test_host_cpu_affinity_admission_binds_exact_masks_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)
    set_calls: list[list[int]] = []

    def get_affinity(pid: int) -> set[int]:
        assert pid == 0
        return set(current)

    def set_affinity(pid: int, cpus: set[int]) -> None:
        assert pid == 0
        ordered = sorted(cpus)
        set_calls.append(ordered)
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    receipt = TOOL._host_cpu_topology_affinity_admission_v2r17(
        contract,
        dependency_gate_invocation_count=1,
        target_fsck_invocation_count=1,
    )

    assert receipt["status"] == "PASS"
    assert receipt["condition"] == "PASS"
    assert receipt["original_effective_cpus_or_null"] == list(range(40))
    assert [
        row["requested_cpus_in_exact_order"]
        for row in receipt["bind_probe_rows_in_exact_order"]
    ] == [[24], [25], [26], [27, 28, 29, 30, 31]]
    assert set_calls == [
        [24],
        [25],
        [26],
        [27, 28, 29, 30, 31],
        list(range(40)),
    ]
    assert current == original
    assert receipt["original_affinity_restored_exactly"] is True
    assert receipt["failure_operation_or_null"] is None
    assert receipt["failure_error_or_null"] is None
    assert TOOL._host_cpu_affinity_receipt_exact_v2r17(
        contract,
        receipt,
        required_status="PASS",
    ) is True


def test_host_cpu_and_gap_policy_cross_bind_pinned_target_d_constants() -> None:
    contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "Target D CPU and gap cross-binding KAT contract",
    )
    assert type(contract) is dict
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    assert target["target_supervisor_commit"] == (
        "5b38581f43c9fd88692b4f3c2cf8f13e8a0ca24b"
    )
    relative = "tools/phase3_q05b_dual_qualification_v2.py"
    target_tool = ROOT / relative
    raw = target_tool.read_bytes()
    assert sha256(raw).hexdigest() == target["required_raw_sha256_by_path"][
        relative
    ]
    module = ast.parse(raw, filename=os.fspath(target_tool))
    expressions = {
        node.target.id: node.value
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }

    def frozen_set(name: str) -> frozenset[int]:
        value = expressions[name]
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name) and value.func.id == "frozenset"
        assert len(value.args) == 1
        argument = value.args[0]
        if isinstance(argument, ast.Call):
            assert isinstance(argument.func, ast.Name)
            assert argument.func.id == "range"
            integers = [ast.literal_eval(item) for item in argument.args]
            return frozenset(range(*integers))
        return frozenset(ast.literal_eval(argument))

    target_wait = ast.literal_eval(expressions["RESOURCE_SAMPLER_TARGET_WAIT_NS"])
    maximum_gap = ast.literal_eval(
        expressions["RESOURCE_SAMPLER_MAXIMUM_UNSAMPLED_GAP_NS"]
    )
    sampler_cpus = ast.literal_eval(expressions["HOST_SAMPLER_CPU_BY_ROLE_V5"])
    required_cpus = frozen_set("HOST_REQUIRED_EFFECTIVE_CPUS_V5")
    publisher_cpus = frozen_set("HOST_TRAJECTORY_PUBLISHER_CPUS_V5")
    control_cpus = frozen_set("HOST_SUPERVISOR_CONTROL_CPUS_V5")

    cpu_policy = TOOL._host_cpu_topology_affinity_policy_v2r17(contract)
    scheduling = TOOL._scheduling_admission_policy_v2(contract)
    assert cpu_policy["required_effective_cpus_in_exact_order"] == sorted(
        required_cpus
    ) == list(range(32))
    assert sampler_cpus == {1: 24, 2: 25, 3: 24}
    assert publisher_cpus == frozenset({26})
    assert control_cpus == frozenset(range(27, 32))
    assert [
        row["cpus_in_exact_order"]
        for row in cpu_policy["bind_probe_rows_in_exact_order"]
    ] == [[24], [25], [26], [27, 28, 29, 30, 31]]
    assert scheduling["sampling_target_wait_ns"] == target_wait == 100_000_000
    assert scheduling["maximum_unsampled_gap_ns"] == maximum_gap == 250_000_000


def test_host_cpu_missing_required_set_is_exact_preconsumption_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    set_calls = 0

    def forbidden_set(_pid: int, _cpus: set[int]) -> None:
        nonlocal set_calls
        set_calls += 1
        raise AssertionError("missing CPU gate must not mutate affinity")

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", lambda _pid: set(range(31)))
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", forbidden_set)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert set_calls == 0
    assert pending.value.code == TOOL._HOST_CPU_PENDING_CODE_V2R17
    assert pending.value.phase == TOOL._HOST_CPU_ADMISSION_PHASE_V2R17
    assert pending.value.condition == "REQUIRED_EFFECTIVE_CPU_SET_UNAVAILABLE"
    assert pending.value.receipt["missing_required_cpus_in_exact_order"] == [31]
    assert pending.value.receipt["affinity_mutation_attempted"] is False
    assert pending.value.receipt["failure_operation_or_null"] == (
        "VALIDATE_REQUIRED_EFFECTIVE_CPU_SET"
    )
    assert pending.value.receipt["failure_error_or_null"] == (
        "MISSING_REQUIRED_CPUS:31"
    )
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is True
    missing_required = deepcopy(pending.value.receipt)
    missing_required["original_effective_cpus_or_null"] = list(range(40))
    missing_required["missing_required_cpus_in_exact_order"] = []
    missing_required["failure_error_or_null"] = "MISSING_REQUIRED_CPUS:"
    pending.value.receipt = missing_required
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is False


def test_host_cpu_affinity_api_unavailable_is_exact_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    monkeypatch.setattr(TOOL.os, "sched_getaffinity", None)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", None)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert pending.value.condition == "SCHED_AFFINITY_API_UNAVAILABLE"
    assert pending.value.receipt["affinity_mutation_attempted"] is False
    assert pending.value.receipt["restore_required"] is False
    assert pending.value.receipt["failure_operation_or_null"] == (
        "LOOKUP_SCHED_AFFINITY_API"
    )
    assert pending.value.receipt["failure_error_or_null"] == (
        "SCHED_AFFINITY_API_NOT_CALLABLE"
    )
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is True
    pending.value.receipt["failure_error_or_null"] = "fabricated"
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is False


def test_host_cpu_bind_failure_is_pending_only_after_exact_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)
    set_calls: list[list[int]] = []
    failed = False

    def get_affinity(_pid: int) -> set[int]:
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        nonlocal failed
        ordered = sorted(cpus)
        set_calls.append(ordered)
        if ordered == [25] and not failed:
            failed = True
            raise OSError("injected CPU 25 bind failure")
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert set_calls == [[24], [25], list(range(40))]
    assert current == original
    assert pending.value.condition == (
        "EXACT_BIND_OR_REPLAY_FAILED_WITH_EXACT_RESTORE"
    )
    assert pending.value.receipt["original_affinity_restored_exactly"] is True
    assert pending.value.receipt["failure_operation_or_null"] == (
        "BIND_OR_READBACK_PROBE_02"
    )
    assert pending.value.receipt["failure_error_or_null"] == (
        pending.value.receipt["bind_probe_rows_in_exact_order"][-1][
            "error_or_null"
        ]
    )
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is True
    missing_required = deepcopy(pending.value.receipt)
    missing_required["original_effective_cpus_or_null"].remove(0)
    missing_required["restored_effective_cpus_or_null"].remove(0)
    pending.value.receipt = missing_required
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is False


def test_host_cpu_readback_mismatch_is_pending_only_after_exact_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)

    def get_affinity(_pid: int) -> set[int]:
        if current == {24}:
            return {25}
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert current == original
    assert pending.value.condition == (
        "EXACT_BIND_OR_REPLAY_FAILED_WITH_EXACT_RESTORE"
    )
    failed_row = pending.value.receipt["bind_probe_rows_in_exact_order"][-1]
    assert failed_row["requested_cpus_in_exact_order"] == [24]
    assert failed_row["observed_cpus_or_null"] == [25]
    assert failed_row["passed"] is False
    assert pending.value.receipt["failure_operation_or_null"] == (
        "BIND_OR_READBACK_PROBE_01"
    )
    assert pending.value.receipt["failure_error_or_null"] == (
        failed_row["error_or_null"]
    )
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is True
    failed_row["error_or_null"] = "fabricated mismatch"
    pending.value.receipt["failure_error_or_null"] = "fabricated mismatch"
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is False


def test_host_cpu_restore_uncertainty_is_terminal_not_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)

    def get_affinity(_pid: int) -> set[int]:
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        if cpus == original:
            raise OSError("injected original-affinity restore failure")
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert error.value.code == "FAIL_CAMPAIGN_HOST_CPU_AFFINITY_RESTORE"
    assert error.value.terminal is True
    assert error.value.receipt["status"] == "TERMINAL_RESTORE_UNCERTAIN"


def test_host_cpu_restore_readback_mismatch_is_terminal_not_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)
    restore_seen = False

    def get_affinity(_pid: int) -> set[int]:
        if restore_seen:
            return set(range(39))
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        nonlocal restore_seen
        current.clear()
        current.update(cpus)
        if cpus == original:
            restore_seen = True

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert current == original
    assert error.value.code == "FAIL_CAMPAIGN_HOST_CPU_AFFINITY_RESTORE"
    assert error.value.terminal is True
    assert error.value.receipt["status"] == "TERMINAL_RESTORE_UNCERTAIN"
    assert error.value.receipt["condition"] == (
        "ORIGINAL_AFFINITY_RESTORE_FAILED"
    )
    assert error.value.receipt["original_affinity_restored_exactly"] is False
    assert str(error.value.receipt["failure_error_or_null"]).startswith(
        "AFFINITY_RESTORE_REPLAY_DIFFERS:"
    )


def test_host_cpu_keyboard_interrupt_is_never_pending_and_restores_after_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)
    interrupted = False

    def get_affinity(_pid: int) -> set[int]:
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        nonlocal interrupted
        current.clear()
        current.update(cpus)
        if cpus == {24} and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected after affinity mutation")

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert current == original
    assert error.value.code == "FAIL_CAMPAIGN_HOST_CPU_ADMISSION_INTERRUPTED"
    assert error.value.terminal is True
    assert error.value.receipt["status"] == (
        "TERMINAL_NONRESUMABLE_INTERRUPTION"
    )


def test_host_cpu_post_readback_interrupt_restores_before_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)
    set_calls: list[list[int]] = []
    conversion_count = 0
    real_conversion = TOOL._host_cpu_affinity_set_or_none_v2r17

    class InterruptAfterSuccessfulReadback(list[int]):
        def __ne__(self, other: object) -> bool:
            return False

        def __eq__(self, other: object) -> bool:
            raise KeyboardInterrupt("injected after successful readback")

    def get_affinity(_pid: int) -> set[int]:
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        set_calls.append(sorted(cpus))
        current.clear()
        current.update(cpus)

    def convert(value: object) -> list[int] | None:
        nonlocal conversion_count
        conversion_count += 1
        converted = real_conversion(value)
        if conversion_count == 2:
            assert converted == [24]
            return InterruptAfterSuccessfulReadback(converted)
        return converted

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(
        TOOL,
        "_host_cpu_affinity_set_or_none_v2r17",
        convert,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    assert set_calls == [[24], list(range(40))]
    assert current == original
    assert error.value.code == "FAIL_CAMPAIGN_HOST_CPU_ADMISSION_INTERRUPTED"
    assert error.value.terminal is True
    assert error.value.receipt["status"] == (
        "TERMINAL_NONRESUMABLE_INTERRUPTION"
    )
    assert error.value.receipt["failure_operation_or_null"] == (
        "BIND_OR_READBACK_PROBE_01"
    )
    assert error.value.receipt["failure_error_or_null"] == (
        "KeyboardInterrupt:injected after successful readback"
    )
    assert error.value.receipt["original_affinity_restored_exactly"] is True


def test_host_cpu_original_read_keyboard_interrupt_is_not_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    monkeypatch.setattr(
        TOOL.os,
        "sched_getaffinity",
        lambda _pid: (_ for _ in ()).throw(KeyboardInterrupt("injected read")),
    )
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", lambda *_args: None)
    with pytest.raises(KeyboardInterrupt, match="injected read"):
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )


@pytest.mark.parametrize(
    ("dependency_count", "fsck_count"),
    ((True, 1), (1, True), (0, 1), (1, 0)),
)
def test_host_cpu_gate_requires_exact_integer_predecessor_counts(
    monkeypatch: pytest.MonkeyPatch,
    dependency_count: object,
    fsck_count: object,
) -> None:
    contract = _contract()
    monkeypatch.setattr(
        TOOL.os,
        "sched_getaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unreachable")),
    )
    monkeypatch.setattr(
        TOOL.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unreachable")),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=dependency_count,
            target_fsck_invocation_count=fsck_count,
        )
    assert error.value.code == "FAIL_CAMPAIGN_HOST_CPU_ADMISSION"
    assert error.value.terminal is True


@pytest.mark.parametrize("dependency_count", (True, 0))
def test_target_fsck_gate_requires_exact_integer_dependency_count(
    monkeypatch: pytest.MonkeyPatch,
    dependency_count: object,
) -> None:
    contract = _contract()
    monkeypatch.setattr(
        TOOL,
        "_bounded_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid predecessor count reached fsck")
        ),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._target_object_closure_fsck_admission_v2r17(
            contract,
            dependency_gate_invocation_count=dependency_count,
        )
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "other/1"),
        ("status", "PASS"),
        ("phase", "OTHER_PHASE"),
        ("condition", "SCHED_AFFINITY_API_UNAVAILABLE"),
        ("required_effective_cpus_in_exact_order", list(range(31))),
        ("original_effective_cpus_or_null", [1, 0]),
        ("missing_required_cpus_in_exact_order", [31, 31]),
        ("affinity_mutation_attempted", 0),
        ("original_affinity_restored_exactly", 1),
        ("failure_operation_or_null", None),
        ("failure_error_or_null", None),
        ("dynamic_receipt_is_capsule_digest_input", True),
    ),
)
def test_host_cpu_pending_classifier_rejects_receipt_drift_and_bool_int_aliases(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    contract = _contract()
    monkeypatch.setattr(TOOL.os, "sched_getaffinity", lambda _pid: set(range(31)))
    monkeypatch.setattr(
        TOOL.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unreachable")),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    pending.value.receipt[field] = value
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is False


def test_host_cpu_pass_receipt_rejects_bool_probe_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)

    def get_affinity(_pid: int) -> set[int]:
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    receipt = TOOL._host_cpu_topology_affinity_admission_v2r17(
        contract,
        dependency_gate_invocation_count=1,
        target_fsck_invocation_count=1,
    )
    missing_required = deepcopy(receipt)
    missing_required["original_effective_cpus_or_null"].remove(0)
    missing_required["restored_effective_cpus_or_null"].remove(0)
    assert TOOL._host_cpu_affinity_receipt_exact_v2r17(
        contract,
        missing_required,
        required_status="PASS",
    ) is False
    receipt["bind_probe_rows_in_exact_order"][0]["probe_ordinal"] = True
    assert TOOL._host_cpu_affinity_receipt_exact_v2r17(
        contract,
        receipt,
        required_status="PASS",
    ) is False


@pytest.mark.parametrize("mutation", ("missing-error", "observed-with-set-failure"))
def test_host_cpu_failed_probe_receipt_requires_exact_error_semantics(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    contract = _contract()
    original = set(range(40))
    current = set(original)
    failed = False

    def get_affinity(_pid: int) -> set[int]:
        return set(current)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        nonlocal failed
        if cpus == {24} and not failed:
            failed = True
            raise OSError("injected bind failure")
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(TOOL.os, "sched_getaffinity", get_affinity)
    monkeypatch.setattr(TOOL.os, "sched_setaffinity", set_affinity)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as pending:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    row = pending.value.receipt["bind_probe_rows_in_exact_order"][0]
    if mutation == "missing-error":
        row["error_or_null"] = None
        pending.value.receipt["failure_error_or_null"] = None
    else:
        row["observed_cpus_or_null"] = [24]
    assert TOOL._preconsumption_pending_is_exact_v2r17(
        contract,
        pending.value,
    ) is False


def test_typed_pending_raise_sites_are_exact_fsck_cpu_and_slot_gate() -> None:
    tree = ast.parse(TOOL_PATH.read_bytes(), filename=os.fspath(TOOL_PATH))
    sites: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Raise(self, node: ast.Raise) -> None:
            call = node.exc
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "ShadowRetryCampaignPendingV1"
            ):
                sites.append(self.functions[-1] if self.functions else "<module>")
            self.generic_visit(node)

    Visitor().visit(tree)
    assert sites == [
        "_run_slot_scheduling_admission_v2r17",
        "_target_object_closure_fsck_admission_v2r17",
        "pause",
    ]


@pytest.mark.parametrize(
    "overrides",
    (
        {"return_code": 2},
        {"timed_out": True, "return_code": -9, "stdout": b"not empty"},
        {"timed_out": True, "return_code": -9, "stderr": b"not empty"},
        {"timed_out": True, "return_code": -9, "stdout_overflow": True},
        {"timed_out": True, "return_code": -9, "stderr_overflow": True},
        {"timed_out": True, "return_code": -9, "child_reaped": False},
        {
            "timed_out": True,
            "return_code": -9,
            "process_group_absent": False,
        },
        {
            "timed_out": True,
            "return_code": -9,
            "wrapper_resources_closed": False,
        },
    ),
)
def test_target_fsck_nonclean_outcomes_are_terminal_not_pending(
    monkeypatch: pytest.MonkeyPatch,
    overrides: Mapping[str, object],
) -> None:
    contract = _contract()
    result = _process(b"", 0)
    result.update(overrides)
    monotonic = iter((1_000, 2_000))
    monkeypatch.setattr(
        TOOL,
        "_bounded_process",
        lambda *_args, **_kwargs: deepcopy(result),
    )
    monkeypatch.setattr(TOOL.time, "monotonic_ns", lambda: next(monotonic))
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._target_object_closure_fsck_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"
    assert error.value.terminal is True


def test_only_exact_target_fsck_receives_300_second_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    observed: list[tuple[list[str], float]] = []

    def bounded(argv: Sequence[str], **kwargs: object) -> dict[str, object]:
        observed.append((list(argv), float(kwargs["timeout_seconds"])))
        return _process(b"", 0)

    monotonic = iter((1_000, 2_000))
    monkeypatch.setattr(TOOL, "_bounded_process", bounded)
    monkeypatch.setattr(TOOL.time, "monotonic_ns", lambda: next(monotonic))
    receipt = TOOL._target_object_closure_fsck_admission_v2r17(
        contract,
        dependency_gate_invocation_count=1,
    )
    repository = str(contract["execution_policy"]["clone_source_repository"])
    TOOL._git(
        contract,
        ["-C", repository, "rev-parse", "HEAD"],
        "controller HEAD replay",
    )
    assert receipt["classification"]["accepted"] is True
    assert observed == [
        (TOOL._target_object_closure_fsck_argv_v2r17(contract), 300.0),
        (["/usr/bin/git", "-C", repository, "rev-parse", "HEAD"], 120.0),
    ]


def test_real_tiny_repository_fixture_passes_exact_target_fsck(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "tiny-target"
    repository.mkdir()
    subprocess.run(
        ["/usr/bin/git", "init", "--quiet", os.fspath(repository)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (repository / "fixture.txt").write_text("bound tiny fixture\n", encoding="ascii")
    subprocess.run(
        ["/usr/bin/git", "-C", os.fspath(repository), "add", "fixture.txt"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-C",
            os.fspath(repository),
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    commit = subprocess.check_output(
        ["/usr/bin/git", "-C", os.fspath(repository), "rev-parse", "HEAD"],
        stderr=subprocess.PIPE,
        text=True,
    ).strip()
    contract = _contract()
    contract["execution_policy"]["clone_source_repository"] = os.fspath(
        repository
    )
    contract["target_commit_a_binding"]["target_supervisor_commit"] = commit
    receipt = TOOL._target_object_closure_fsck_admission_v2r17(
        contract,
        dependency_gate_invocation_count=1,
    )
    assert receipt["classification"]["accepted"] is True
    assert receipt["return_code"] == 0
    assert receipt["stdout"]["byte_count"] == 0
    assert receipt["stderr"]["byte_count"] == 0


def test_target_preflight_rejects_current_cargo_lock_drift_from_target_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _target_git, _controller_git, _python_identity = (
        _dependency_preflight_inputs()
    )
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    repository = str(execution["clone_source_repository"])
    commit = str(target["target_supervisor_commit"])
    project = str(execution["clone_source_project_relative_path"])
    commit_object = subprocess.check_output(
        ["/usr/bin/git", "-C", repository, "cat-file", "commit", commit]
    )

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        _label: str,
        **_kwargs: object,
    ) -> bytes:
        if arguments[-1] == f"{commit}^{{tree}}":
            return str(target["target_supervisor_tree"]).encode() + b"\n"
        if arguments[-1] == f"{commit}^":
            return str(target["target_supervisor_parent_commit"]).encode() + b"\n"
        if arguments[-3:-1] == ["cat-file", "commit"]:
            return commit_object
        if "fsck" in arguments:
            return b""
        if arguments[2:3] == ["show"]:
            object_spec = str(arguments[-1])
            relative = object_spec.split(":", 1)[1]
            assert relative.startswith(project + "/")
            return (ROOT / relative[len(project) + 1 :]).read_bytes()
        if "ls-tree" in arguments:
            return b""
        raise AssertionError(arguments)

    real_read = TOOL._read_regular_nofollow

    def drift_current(
        path: Path,
        label: str,
        maximum: int,
        **kwargs: object,
    ) -> tuple[bytes, os.stat_result]:
        raw, status = real_read(path, label, maximum, **kwargs)
        if path == ROOT / TOOL._CARGO_LOCK_RELATIVE_PATH_V1:
            return b"drifted Cargo.lock\n", status
        return raw, status

    monkeypatch.setattr(
        TOOL,
        "_verify_git_source_self_contained",
        lambda *_args, **_kwargs: {"self_contained": True},
    )
    monkeypatch.setattr(TOOL, "_git", fake_git)
    monkeypatch.setattr(TOOL, "_read_regular_nofollow", drift_current)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._verify_target_git_objects(contract)
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"
    assert "Cargo.lock" in error.value.detail


def test_controller_preflight_rejects_clean_later_descendant_of_commit_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    repository = str(execution["clone_source_repository"])
    clean_later_descendant_parent = "ab" * 20
    assert clean_later_descendant_parent != target["target_supervisor_commit"]

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        _label: str,
        **_kwargs: object,
    ) -> bytes:
        if arguments == ["-C", repository, "rev-parse", "HEAD"]:
            return b"cd" * 20 + b"\n"
        if arguments == ["-C", repository, "rev-parse", "HEAD^{tree}"]:
            return b"ef" * 20 + b"\n"
        if arguments == ["-C", repository, "rev-parse", "HEAD^"]:
            return clean_later_descendant_parent.encode("ascii") + b"\n"
        if arguments == [
            "-C", repository, "rev-list", "--parents", "-n", "1", "HEAD",
        ]:
            return (
                ("cd" * 20) + " " + clean_later_descendant_parent + "\n"
            ).encode("ascii")
        if arguments == [
            "-C", repository, "status", "--porcelain=v1", "-z",
            "--untracked-files=all",
        ]:
            return b""
        if arguments == ["-C", repository, "rev-parse", "--show-toplevel"]:
            return repository.encode("utf-8") + b"\n"
        if arguments == [
            "-C", repository, "diff-tree", "--no-commit-id", "--name-only",
            "--no-renames", "-r", "-z", "HEAD^", "HEAD",
        ]:
            project = str(execution["clone_source_project_relative_path"])
            return b"".join(
                os.fsencode(f"{project}/{relative}") + b"\x00"
                for relative in (
                    TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
                )
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(TOOL, "_git", fake_git)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._controller_git_identity(contract)
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"
    assert "exact Commit-B parent" in error.value.detail


@pytest.mark.parametrize(
    "mutation",
    (
        "valid",
        "missing",
        "extra",
        "reordered",
        "missing-prefix",
        "missing-nul",
        "double-nul",
        "merge-parent",
    ),
)
def test_controller_preflight_requires_exact_six_path_raw_diff_tree(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    repository = str(execution["clone_source_repository"])
    project = str(execution["clone_source_project_relative_path"])
    rows = [
        os.fsencode(f"{project}/{relative}") + b"\x00"
        for relative in TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(os.fsencode(f"{project}/unexpected") + b"\x00")
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "missing-prefix":
        rows[0] = os.fsencode(
            TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER[0]
        ) + b"\x00"
    raw_diff = b"".join(rows)
    if mutation == "missing-nul":
        raw_diff = raw_diff[:-1]
    elif mutation == "double-nul":
        raw_diff += b"\x00"

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        _label: str,
        **_kwargs: object,
    ) -> bytes:
        if arguments == ["-C", repository, "rev-parse", "HEAD"]:
            return b"cd" * 20 + b"\n"
        if arguments == ["-C", repository, "rev-parse", "HEAD^{tree}"]:
            return b"ef" * 20 + b"\n"
        if arguments == ["-C", repository, "rev-parse", "HEAD^"]:
            return str(target["target_supervisor_commit"]).encode() + b"\n"
        if arguments == [
            "-C", repository, "rev-list", "--parents", "-n", "1", "HEAD",
        ]:
            lineage = (
                ("cd" * 20)
                + " "
                + str(target["target_supervisor_commit"])
            )
            if mutation == "merge-parent":
                lineage += " " + ("12" * 20)
            return (lineage + "\n").encode()
        if arguments == [
            "-C", repository, "status", "--porcelain=v1", "-z",
            "--untracked-files=all",
        ]:
            return b""
        if arguments == ["-C", repository, "rev-parse", "--show-toplevel"]:
            return repository.encode() + b"\n"
        if arguments == [
            "-C", repository, "diff-tree", "--no-commit-id", "--name-only",
            "--no-renames", "-r", "-z", "HEAD^", "HEAD",
        ]:
            return raw_diff
        raise AssertionError(arguments)

    monkeypatch.setattr(TOOL, "_git", fake_git)
    if mutation == "valid":
        identity = TOOL._controller_git_identity(contract)
        assert identity["parent_count"] == 1
        assert identity["changed_paths_from_target_in_exact_order"] == list(
            TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
        )
        return
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._controller_git_identity(contract)
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"


@pytest.mark.parametrize(
    "forbidden_case",
    (
        "alternates",
        "http-alternates",
        "shallow",
        "grafts",
        "promisor-pack",
        "partial-clone-config",
        "promisor-config",
        "fsck-config",
        "include-config",
        "includeif-config",
    ),
)
def test_git_source_self_containment_rejects_external_or_partial_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forbidden_case: str,
) -> None:
    common = tmp_path / "common.git"
    objects = common / "objects"
    git_dir = common / "worktrees" / "bound-worktree"
    (objects / "info").mkdir(parents=True, mode=0o700)
    (objects / "pack").mkdir(mode=0o700)
    (common / "info").mkdir(mode=0o700)
    git_dir.mkdir(parents=True, mode=0o700)
    config_raw = b"core.repositoryformatversion\n0\x00"
    if forbidden_case == "alternates":
        (objects / "info" / "alternates").write_bytes(b"/external/objects\n")
    elif forbidden_case == "http-alternates":
        (objects / "info" / "http-alternates").write_bytes(
            b"https://invalid.example/objects\n"
        )
    elif forbidden_case == "shallow":
        (common / "shallow").write_bytes(b"1" * 40 + b"\n")
    elif forbidden_case == "grafts":
        (common / "info" / "grafts").write_bytes(b"1" * 40 + b"\n")
    elif forbidden_case == "promisor-pack":
        (objects / "pack" / ("pack-" + "1" * 40 + ".promisor")).write_bytes(
            b""
        )
    elif forbidden_case == "partial-clone-config":
        config_raw += b"extensions.partialclone\norigin\x00"
    elif forbidden_case == "promisor-config":
        config_raw += b"remote.origin.promisor\ntrue\x00"
    elif forbidden_case == "fsck-config":
        config_raw += b"fsck.skiplist\n/external/skip-list\x00"
    elif forbidden_case == "include-config":
        config_raw += b"include.path\n/external/config\x00"
    elif forbidden_case == "includeif-config":
        config_raw += (
            b"includeif.gitdir:/bound/repository.path\n/external/config\x00"
        )

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        _label: str,
        **_kwargs: object,
    ) -> bytes:
        if arguments[-1] == "--git-common-dir":
            return os.fsencode(common) + b"\n"
        if arguments[-1] == "--absolute-git-dir":
            return os.fsencode(git_dir) + b"\n"
        if arguments[-2:] == ["--git-path", "objects"]:
            return os.fsencode(objects) + b"\n"
        if arguments[-1] == "--is-shallow-repository":
            return b"false\n"
        if arguments[-1] == "--show-object-format":
            return b"sha1\n"
        if "config" in arguments:
            return config_raw
        raise AssertionError(arguments)

    monkeypatch.setattr(TOOL, "_git", fake_git)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._verify_git_source_self_contained({}, "/bound/repository")
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"
    assert any(
        token in error.value.detail
        for token in ("forbidden", "partial", "promisor")
    )


def test_git_source_self_containment_binds_common_object_and_config_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    common = tmp_path / "common.git"
    objects = common / "objects"
    git_dir = common / "worktrees" / "bound-worktree"
    (objects / "info").mkdir(parents=True, mode=0o700)
    (objects / "pack").mkdir(mode=0o700)
    (common / "info").mkdir(mode=0o700)
    git_dir.mkdir(parents=True, mode=0o700)
    config_raw = b"core.repositoryformatversion\n0\x00core.filemode\ntrue\x00"

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        _label: str,
        **_kwargs: object,
    ) -> bytes:
        if arguments[-1] == "--git-common-dir":
            return os.fsencode(common) + b"\n"
        if arguments[-1] == "--absolute-git-dir":
            return os.fsencode(git_dir) + b"\n"
        if arguments[-2:] == ["--git-path", "objects"]:
            return os.fsencode(objects) + b"\n"
        if arguments[-1] == "--is-shallow-repository":
            return b"false\n"
        if arguments[-1] == "--show-object-format":
            return b"sha1\n"
        if "config" in arguments:
            return config_raw
        raise AssertionError(arguments)

    monkeypatch.setattr(TOOL, "_git", fake_git)
    evidence = TOOL._verify_git_source_self_contained({}, "/bound/repository")
    assert evidence["git_common_dir"] == os.fspath(common)
    assert evidence["git_object_dir"] == os.fspath(objects)
    assert evidence["git_worktree_dir"] == os.fspath(git_dir)
    assert evidence["local_config_listing_sha256"] == sha256(config_raw).hexdigest()
    assert evidence["local_config_row_count"] == 2
    assert evidence["repository_format_version"] == 0
    assert evidence["object_format"] == "sha1"
    assert evidence["alternates_present"] is False
    assert evidence["promisor_metadata_present"] is False


def test_fixed_cargo_source_accepts_bound_root_owned_read_only_entries() -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    assert type(execution) is dict
    identity, rows = TOOL._scan_tree(
        Path(str(execution["cargo_copy_source"])),
        label="fixed Cargo source owner-policy KAT",
        allowed_uids=(os.geteuid(), 0),
        reject_group_or_world_writable=True,
    )
    assert rows
    assert {int(row["uid"]) for row in rows}.issubset(
        {os.geteuid(), 0}
    )
    assert any(int(row["uid"]) == 0 for row in rows)
    assert all(int(row["mode"]) & 0o022 == 0 for row in rows)
    assert identity["content_manifest_digest"].startswith("sha256:")


def test_freshness_parent_ignores_nlink_growth_but_rejects_inode_substitution(
    tmp_path: Path,
) -> None:
    state_parent = tmp_path / "state"
    state_parent.mkdir(mode=0o700)
    slot = {
        "slot_id": "SLOT_01",
        "clone_root": os.fspath(state_parent / "slot-clone"),
        "cargo_cache_root": os.fspath(state_parent / "slot-cargo"),
        "artifact_parent_root": os.fspath(state_parent / "slot-artifact"),
        "scheduling_probe_root": os.fspath(state_parent / "slot-probe"),
        "evidence_root": os.fspath(state_parent / "slot-evidence"),
        "trace_root": os.fspath(state_parent / "slot-trace"),
    }
    contract = {
        "fixed_external_roots": {
            "ledger_root": os.fspath(state_parent / "campaign-ledger")
        }
    }
    expected = TOOL._verify_fresh_targets(contract, [slot])
    parent_key = os.fspath(state_parent)
    assert "nlink" not in expected[parent_key]
    nlink_before = state_parent.stat().st_nlink

    (state_parent / "owner-authorization").mkdir(mode=0o700)
    (state_parent / "campaign-ledger").mkdir(mode=0o700)
    assert state_parent.stat().st_nlink > nlink_before
    context = {"freshness_parent_identities": expected}
    stable = TOOL._slot_roots_absence_evidence(
        context,
        [slot],
        label="nlink growth freshness KAT",
    )
    assert stable["certain"] is True
    assert all(row["absent_and_anchored"] is True for row in stable["rows"])
    assert any(
        row["observed_state_parent_identity"]["nlink"] != nlink_before
        for row in stable["rows"]
    )
    slot_with_artifact = {
        **slot,
        "artifact_path": os.fspath(
            Path(str(slot["artifact_parent_root"])) / "artifact.json"
        ),
    }
    absent_artifact = TOOL._absent_external_artifact(
        context,
        slot_with_artifact,
    )
    assert absent_artifact["state"] == "ABSENT"
    assert absent_artifact["error_or_null"] is None

    replaced_parent = tmp_path / "state-replaced"
    state_parent.rename(replaced_parent)
    state_parent.mkdir(mode=0o700)
    substituted = TOOL._slot_roots_absence_evidence(
        context,
        [slot],
        label="inode substitution freshness KAT",
    )
    assert substituted["certain"] is False
    assert "parent drifted" in str(substituted["error_or_null"])


def test_preflight_names_only_ledger_and_slot_freshness_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub(contract=contract)
    dependency = capsule["python310_dependency_preflight_receipt"]
    context = {
        "capsule": capsule,
        "python310_dependency_preflight_receipt": dependency,
        "preauthorization_live_gauntlet": (
            _gauntlet_receipt_binding_stub(
                contract,
                capsule["controller_git"],
            )
        ),
        "target_object_closure_fsck_admission_receipt": {
            "classification": {"accepted": True}
        },
        "host_cpu_topology_affinity_admission_receipt": {"status": "PASS"},
        "authorization_template": {"authorization_digest": _tagged("authorization")},
    }
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    monkeypatch.setattr(TOOL.os.path, "lexists", lambda _path: True)
    receipt = TOOL.preflight_phase3_q05b_shadow_retry_campaign_v2()
    assert receipt["owner_authorization_present"] is True
    assert receipt["ledger_and_slot_freshness_targets_absent"] is True
    assert receipt["python310_dependency_preflight_receipt"] == dependency
    assert receipt["python310_dependency_preflight_receipt_digest"] == (
        dependency["receipt_digest"]
    )
    assert receipt["target_object_closure_fsck_admission_receipt"] == {
        "classification": {"accepted": True}
    }
    assert receipt["host_cpu_topology_affinity_admission_receipt"] == {
        "status": "PASS"
    }
    assert "freshness_targets_absent" not in receipt


def test_preflight_has_no_docker_supervisor_clone_or_root_mutation_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    slots = TOOL._slot_registry(contract)
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    dependency = _dependency_preflight_receipt_stub()
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def invoke(*_args: object, **_kwargs: object) -> object:
            forbidden_calls.append(name)
            raise AssertionError(f"preflight reached forbidden call: {name}")

        return invoke

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda _contract, _slots, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_target_git_objects",
        lambda _contract: {
            "commit": target["target_supervisor_commit"],
            "tree": target["target_supervisor_tree"],
            "parent": target["target_supervisor_parent_commit"],
            "target_source_binding_digest": target[
                "target_source_binding_digest"
            ],
            "required_raw_sha256_by_path": deepcopy(
                target["required_raw_sha256_by_path"]
            ),
            "cargo_lock_git_object": deepcopy(
                dependency["cargo_lock_git_object"]
            ),
            "local_object_closure_fsck_passed": False,
        },
    )
    monkeypatch.setattr(
        TOOL,
        "_controller_git_identity",
        lambda _contract: {
            "commit": dependency["controller_commit"],
            "tree": "9a" * 20,
            "parent": target["target_supervisor_commit"],
            "parent_count": 1,
            "clean": True,
            "changed_paths_from_target_in_exact_order": list(
                TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER
            ),
        },
    )
    monkeypatch.setattr(
        TOOL,
        "_scan_tree",
        lambda *_args, **_kwargs: ({"manifest_digest": _tagged("cargo")}, []),
    )
    monkeypatch.setattr(
        TOOL,
        "_executable_identity",
        lambda path, _label: {
            "path": path,
            "raw_sha256": (
                dependency["python_executable"]["raw_sha256"]
                if path == "/usr/bin/python3.10"
                else _raw(path)
            ),
        },
    )
    monkeypatch.setattr(
        TOOL,
        "_python310_dependency_preflight_v1",
        lambda *_args, **_kwargs: deepcopy(dependency),
    )
    monkeypatch.setattr(
        TOOL,
        "_target_object_closure_fsck_admission_v2r17",
        lambda *_args, **_kwargs: {"classification": {"accepted": True}},
    )
    monkeypatch.setattr(TOOL.os.path, "lexists", lambda _path: False)
    for name in (
        "_docker_inventory",
        "_safe_docker_inventory",
        "_prepare_clone",
        "_copy_tree_exact",
        "_create_private_root",
        "_create_evidence_root",
        "_run_one_slot",
    ):
        monkeypatch.setattr(TOOL, name, forbidden(name))

    # Local Git replay is allowed to use Popen in the real preflight, so this
    # KAT guards the supervisor entry point rather than asserting Popen == 0.
    receipt = TOOL.preflight_phase3_q05b_shadow_retry_campaign_v2()
    assert receipt["supervisor_invocation_count"] == 0
    assert receipt["docker_daemon_contacted"] is False
    assert receipt["clone_cache_ledger_or_evidence_created"] is False
    assert receipt["capsule"]["ordered_slots"] == slots
    assert receipt["python310_dependency_preflight_receipt"] == dependency
    assert forbidden_calls == []


def test_all_three_entrypoints_request_a_fresh_preflight_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub("entrypoint-capsule")
    dependency = capsule["python310_dependency_preflight_receipt"]
    context = {
        "capsule": capsule,
        "slots": deepcopy(contract["slot_registry"]),
        "python310_dependency_preflight_receipt": dependency,
        "preauthorization_live_gauntlet": (
            _gauntlet_receipt_binding_stub(
                contract,
                capsule["controller_git"],
            )
        ),
        "target_object_closure_fsck_admission_receipt": {"status": "PASS"},
        "host_cpu_topology_affinity_admission_receipt": {"status": "PASS"},
        "authorization_template": {"authorization_digest": _tagged("auth")},
    }
    calls = 0

    def fresh_context(
        _contract: Mapping[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return deepcopy(context)

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", fresh_context)
    monkeypatch.setattr(TOOL.os.path, "lexists", lambda _path: False)
    TOOL.preflight_phase3_q05b_shadow_retry_campaign_v2()
    for entrypoint in (
        TOOL.create_phase3_q05b_shadow_retry_authorization_v2,
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2,
    ):
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            entrypoint(_tagged("different-capsule"))
        assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"
    assert calls == 3


def test_preflight_context_runs_dependency_then_one_fsck_then_authorization_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub("ordered-preflight")
    events: list[str] = []
    fsck_count = 0

    monkeypatch.setattr(
        TOOL,
        "_slot_registry",
        lambda _contract: deepcopy(contract["slot_registry"]),
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_target_git_objects",
        lambda _contract: {"local_object_closure_fsck_passed": False},
    )
    monkeypatch.setattr(TOOL, "_controller_git_identity", lambda _contract: {})
    monkeypatch.setattr(
        TOOL,
        "_scan_tree",
        lambda *_args, **_kwargs: ({"manifest_digest": _tagged("cargo")}, []),
    )
    monkeypatch.setattr(
        TOOL,
        "_executable_identity",
        lambda path, _label: {"path": path, "raw_sha256": _raw(path)},
    )

    def dependency(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("dependency")
        return {"receipt_digest": _tagged("dependency")}

    def gauntlet(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("live-gauntlet-replay")
        return {"receipt_digest": _tagged("live-gauntlet")}

    def build(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("capsule")
        return deepcopy(capsule)

    def fsck(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal fsck_count
        events.append("target-fsck")
        fsck_count += 1
        return {"classification": {"accepted": True}}

    def authorization(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("authorization-template")
        return {"authorization_digest": _tagged("authorization")}

    monkeypatch.setattr(
        TOOL,
        "_load_preauthorization_live_gauntlet_receipt_v2",
        gauntlet,
    )
    monkeypatch.setattr(TOOL, "_python310_dependency_preflight_v1", dependency)
    monkeypatch.setattr(TOOL, "_build_capsule", build)
    monkeypatch.setattr(
        TOOL,
        "_target_object_closure_fsck_admission_v2r17",
        fsck,
    )
    monkeypatch.setattr(TOOL, "_authorization_template", authorization)

    for _entry in range(2):
        context = TOOL._preflight_context(
            contract,
            expected_capsule_digest_or_null=capsule["capsule_digest"],
        )
        assert context["capsule"] == capsule
    assert fsck_count == 2
    assert events == [
        "live-gauntlet-replay",
        "dependency",
        "capsule",
        "target-fsck",
        "authorization-template",
    ] * 2


def test_wrong_valid_capsule_digest_stops_before_fsck_and_cannot_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub("candidate-before-fsck")
    fsck_count = 0
    authorization_template_count = 0
    monkeypatch.setattr(
        TOOL,
        "_slot_registry",
        lambda _contract: deepcopy(contract["slot_registry"]),
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_target_git_objects",
        lambda _contract: {"local_object_closure_fsck_passed": False},
    )
    monkeypatch.setattr(TOOL, "_controller_git_identity", lambda _contract: {})
    monkeypatch.setattr(
        TOOL,
        "_scan_tree",
        lambda *_args, **_kwargs: ({"manifest_digest": _tagged("cargo")}, []),
    )
    monkeypatch.setattr(
        TOOL,
        "_executable_identity",
        lambda path, _label: {"path": path, "raw_sha256": _raw(path)},
    )
    monkeypatch.setattr(
        TOOL,
        "_python310_dependency_preflight_v1",
        lambda *_args, **_kwargs: {"receipt_digest": _tagged("dependency")},
    )
    monkeypatch.setattr(
        TOOL,
        "_build_capsule",
        lambda *_args, **_kwargs: deepcopy(capsule),
    )

    def forbidden_fsck(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal fsck_count
        fsck_count += 1
        raise _target_fsck_pending_error(contract)

    def forbidden_authorization(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        nonlocal authorization_template_count
        authorization_template_count += 1
        return {}

    monkeypatch.setattr(
        TOOL,
        "_target_object_closure_fsck_admission_v2r17",
        forbidden_fsck,
    )
    monkeypatch.setattr(TOOL, "_authorization_template", forbidden_authorization)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._preflight_context(
            contract,
            expected_capsule_digest_or_null=_tagged("wrong-but-valid"),
        )
    assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"
    assert fsck_count == 0
    assert authorization_template_count == 0


def test_tree_manifest_and_closure_reject_ignored_file_and_empty_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "clone"
    root.mkdir(mode=0o700)
    (root / ".gitignore").write_text("ignored.tmp\n", encoding="ascii")
    (root / ".git").mkdir(mode=0o700)
    (root / ".git/config").write_text("baseline\n", encoding="ascii")
    before_identity, before_rows = TOOL._scan_tree(
        root,
        label="clone closure before",
        manifest_domain="HEGEL/TEST/CLONE-CLOSURE/V2",
    )
    (root / "ignored.tmp").write_bytes(b"residual\n")
    (root / "empty-residual-directory").mkdir(mode=0o700)
    (root / ".git/config").write_text("changed\n", encoding="ascii")
    after_identity, after_rows = TOOL._scan_tree(
        root,
        label="clone closure after",
        manifest_domain="HEGEL/TEST/CLONE-CLOSURE/V2",
    )
    assert before_identity["manifest_digest"] != after_identity["manifest_digest"]
    assert {row["relative_path"] for row in after_rows} - {
        row["relative_path"] for row in before_rows
    } == {"ignored.tmp", "empty-residual-directory"}

    source_before = {
        "certain": True,
        "error_or_null": None,
        "evidence": {
            "clone": {
                "clone_closure_without_artifact_digest": before_identity[
                    "manifest_digest"
                ],
                "clone_tree_entry_count": len(before_rows),
            }
        },
        "root": _tagged("before"),
        "clone_closure_rows_internal": before_rows,
    }
    source_after = {
        "certain": True,
        "error_or_null": None,
        "evidence": {
            "clone": {
                "clone_closure_without_artifact_digest": after_identity[
                    "manifest_digest"
                ],
                "clone_tree_entry_count": len(after_rows),
            }
        },
        "root": _tagged("after"),
        "clone_closure_rows_internal": after_rows,
    }
    closed = TOOL._enforce_clone_closure(
        {"slot_id": "SLOT_01"},
        source_before,
        source_after,
    )
    assert closed["certain"] is False
    assert "outside the single allowed artifact path" in str(
        closed["error_or_null"]
    )
    difference = closed["evidence"]["clone_closure_comparison"][
        "difference"
    ]
    assert {row["relative_path"] for row in difference["added_rows"]} == {
        "ignored.tmp",
        "empty-residual-directory",
    }
    assert [
        row["relative_path"] for row in difference["changed_rows"]
    ] == [".git/config"]


def test_slot_registry_accepts_exact_contract_and_rejects_path_drift() -> None:
    contract = _contract()
    slots = TOOL._slot_registry(contract)
    assert [slot["slot_id"] for slot in slots] == [
        "SLOT_01",
        "SLOT_02",
        "SLOT_03",
    ]
    assert [slot["slot_ordinal"] for slot in slots] == [1, 2, 3]

    drifted = deepcopy(contract)
    drifted_slots = drifted["slot_registry"]
    assert type(drifted_slots) is list and type(drifted_slots[0]) is dict
    drifted_slots[0]["artifact_parent_initial_state"] = "ABSENT"
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._slot_registry(drifted)
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"

    aliased = deepcopy(contract)
    aliased_slots = aliased["slot_registry"]
    assert type(aliased_slots) is list
    assert type(aliased_slots[0]) is dict and type(aliased_slots[1]) is dict
    aliased_slots[1]["cargo_cache_root"] = aliased_slots[0]["cargo_cache_root"]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._slot_registry(aliased)
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


def test_v2r20_instance_isolation_rejects_all_historical_authority_and_roots() -> None:
    raw_contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "v2r20 target Git object closure",
    )
    assert type(raw_contract) is dict
    target_git_object_closure = raw_contract["target_commit_a_binding"]
    assert type(target_git_object_closure) is dict
    assert target_git_object_closure["status"] == "TARGET_COMMIT_A_BOUND"
    assert target_git_object_closure["target_supervisor_commit"] == (
        "5b38581f43c9fd88692b4f3c2cf8f13e8a0ca24b"
    )

    contract = _contract()
    TOOL._require_v2r20_instance_isolation_v2(contract)
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    protected_roots = protected["protected_roots_in_exact_order"]
    assert type(protected_roots) is list
    assert len(protected_roots) == 394
    assert protected_roots == list(TOOL._EXPECTED_PROTECTED_ROOTS_V2R20)
    assert set(TOOL._PRIOR_V2_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R1_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R2_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R3_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R4_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R5_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R6_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R7_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R8_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R9_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R10_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R11_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R12_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R13_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R14_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R15_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R16_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R17_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R18_PROTECTED_ROOTS).issubset(protected_roots)
    assert set(TOOL._PRIOR_V2R19_PROTECTED_ROOTS).issubset(protected_roots)
    assert protected["prior_v2r1_campaign_contract_id"] == (
        TOOL._PRIOR_V2R1_CONTRACT_ID
    )
    assert protected["prior_v2r1_contract_digest"] == (
        TOOL._PRIOR_V2R1_CONTRACT_DIGEST
    )
    assert protected["prior_v2r1_capsule_digest"] == (
        TOOL._PRIOR_V2R1_CAPSULE_DIGEST
    )
    assert protected["prior_v2r1_authorization_digest"] == (
        TOOL._PRIOR_V2R1_AUTHORIZATION_DIGEST
    )
    assert protected["prior_v2r1_terminal_ledger_record_digest"] == (
        TOOL._PRIOR_V2R1_TERMINAL_LEDGER_RECORD_DIGEST
    )
    assert protected["prior_v2r1_completion_ledger_record_digest"] == (
        TOOL._PRIOR_V2R1_COMPLETION_LEDGER_RECORD_DIGEST
    )
    exact_v2r2_facts = {
        "prior_v2r2_campaign_contract_id": TOOL._PRIOR_V2R2_CONTRACT_ID,
        "prior_v2r2_controller_commit": TOOL._PRIOR_V2R2_CONTROLLER_COMMIT,
        "prior_v2r2_controller_tree": TOOL._PRIOR_V2R2_CONTROLLER_TREE,
        "prior_v2r2_target_commit": TOOL._PRIOR_V2R2_TARGET_COMMIT,
        "prior_v2r2_campaign_source_binding_digest": (
            TOOL._PRIOR_V2R2_CAMPAIGN_SOURCE_BINDING_DIGEST
        ),
        "prior_v2r2_contract_digest": TOOL._PRIOR_V2R2_CONTRACT_DIGEST,
        "prior_v2r2_capsule_digest": TOOL._PRIOR_V2R2_CAPSULE_DIGEST,
        "prior_v2r2_owner_confirmation_digest": (
            TOOL._PRIOR_V2R2_OWNER_CONFIRMATION_DIGEST
        ),
        "prior_v2r2_authorization_digest": (
            TOOL._PRIOR_V2R2_AUTHORIZATION_DIGEST
        ),
        "prior_v2r2_terminal_status": TOOL._PRIOR_V2R2_TERMINAL_STATUS,
        "prior_v2r2_terminal_body_digest": (
            TOOL._PRIOR_V2R2_TERMINAL_BODY_DIGEST
        ),
        "prior_v2r2_terminal_ledger_record_digest": (
            TOOL._PRIOR_V2R2_TERMINAL_LEDGER_RECORD_DIGEST
        ),
        "prior_v2r2_completion_body_digest": (
            TOOL._PRIOR_V2R2_COMPLETION_BODY_DIGEST
        ),
        "prior_v2r2_completion_ledger_record_digest": (
            TOOL._PRIOR_V2R2_COMPLETION_LEDGER_RECORD_DIGEST
        ),
    }
    for field, expected in exact_v2r2_facts.items():
        assert protected[field] == expected
    assert protected["prior_v2r2_supervisor_invocation_count"] == 3
    assert protected["prior_v2r2_each_slot_trajectory_event_count"] == 42
    assert protected["prior_v2r2_each_slot_retained_failure_rows"] == 7
    assert protected["prior_v2r2_each_slot_retained_cleanup_rows"] == 8
    assert protected["prior_v2r2_each_slot_unpartitioned_event_count"] == 0
    exact_v2r3_facts = {
        "prior_v2r3_campaign_contract_id": TOOL._PRIOR_V2R3_CONTRACT_ID,
        "prior_v2r3_controller_commit": TOOL._PRIOR_V2R3_CONTROLLER_COMMIT,
        "prior_v2r3_controller_tree": TOOL._PRIOR_V2R3_CONTROLLER_TREE,
        "prior_v2r3_target_commit": TOOL._PRIOR_V2R3_TARGET_COMMIT,
        "prior_v2r3_target_tree": TOOL._PRIOR_V2R3_TARGET_TREE,
        "prior_v2r3_campaign_source_binding_digest": (
            TOOL._PRIOR_V2R3_CAMPAIGN_SOURCE_BINDING_DIGEST
        ),
        "prior_v2r3_contract_digest": TOOL._PRIOR_V2R3_CONTRACT_DIGEST,
        "prior_v2r3_capsule_digest": TOOL._PRIOR_V2R3_CAPSULE_DIGEST,
        "prior_v2r3_owner_confirmation_digest": (
            TOOL._PRIOR_V2R3_OWNER_CONFIRMATION_DIGEST
        ),
        "prior_v2r3_authorization_digest": (
            TOOL._PRIOR_V2R3_AUTHORIZATION_DIGEST
        ),
        "prior_v2r3_terminal_status": TOOL._PRIOR_V2R3_TERMINAL_STATUS,
        "prior_v2r3_terminal_body_digest": (
            TOOL._PRIOR_V2R3_TERMINAL_BODY_DIGEST
        ),
        "prior_v2r3_terminal_ledger_record_digest": (
            TOOL._PRIOR_V2R3_TERMINAL_LEDGER_RECORD_DIGEST
        ),
        "prior_v2r3_completion_body_digest": (
            TOOL._PRIOR_V2R3_COMPLETION_BODY_DIGEST
        ),
        "prior_v2r3_completion_ledger_record_digest": (
            TOOL._PRIOR_V2R3_COMPLETION_LEDGER_RECORD_DIGEST
        ),
    }
    for field, expected in exact_v2r3_facts.items():
        assert protected[field] == expected
    assert protected["prior_v2r3_supervisor_invocation_count"] == 1
    assert protected["prior_v2r3_ledger_record_count"] == 8
    assert protected["prior_v2r3_last_consumed_slot_id"] == "SLOT_01"
    assert protected["prior_v2r3_slot_01_decision"] == "TERMINAL_NONRETRYABLE"
    assert protected["prior_v2r3_slot_01_trajectory_status"] == "INCOMPLETE_PREFIX"
    assert protected["prior_v2r3_slot_01_trajectory_event_count"] == 5_410
    assert protected["prior_v2r3_slot_01_retained_failure_rows"] == 0
    assert protected["prior_v2r3_slot_01_retained_cleanup_rows"] == 0
    assert protected["prior_v2r3_slot_01_open_span_count"] == 6
    assert protected["prior_v2r3_slot_01_terminal_manifest_present"] is False
    assert protected["prior_v2r3_slot_01_partition_certificate_present"] is False
    assert protected["prior_v2r3_slot_01_unpartitioned_event_count_or_null"] is None
    assert protected["prior_v2r3_slot_02_invoked"] is False
    assert protected["prior_v2r3_slot_03_invoked"] is False
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R4_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R4_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R5_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R5_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R6_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R6_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R7_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R7_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R8_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R8_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R9_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R9_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R10_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R10_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R11_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R11_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R12_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R12_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R13_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R13_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R14_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R14_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R15_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R15_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R16_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R16_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R17_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R17_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS
    assert {
        field: protected[field]
        for field in TOOL._PRIOR_V2R19_PROTECTED_POLICY_FACTS
    } == TOOL._PRIOR_V2R19_PROTECTED_POLICY_FACTS
    assert len(TOOL._PRIOR_V2R6_RUNTIME_ROOTS) == 17
    assert TOOL._PRIOR_V2R6_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R6_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R6_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R8_RUNTIME_ROOTS) == 17
    assert len(TOOL._PRIOR_V2R8_RUNTIME_ROOT_STATES_IN_EXACT_ORDER) == 17
    assert TOOL._PRIOR_V2R8_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 6
    assert TOOL._PRIOR_V2R8_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 11
    assert TOOL._PRIOR_V2R8_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R8_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R8_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R9_RUNTIME_ROOTS) == 17
    assert TOOL._PRIOR_V2R9_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R9_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R9_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R10_RUNTIME_ROOTS) == 17
    assert TOOL._PRIOR_V2R10_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 1
    assert TOOL._PRIOR_V2R10_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 16
    assert TOOL._PRIOR_V2R10_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R10_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R10_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R12_RUNTIME_ROOTS) == 20
    assert len(TOOL._PRIOR_V2R12_RUNTIME_ROOT_STATES_IN_EXACT_ORDER) == 20
    assert TOOL._PRIOR_V2R12_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 10
    assert TOOL._PRIOR_V2R12_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 10
    assert TOOL._PRIOR_V2R12_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R12_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R12_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R13_RUNTIME_ROOTS) == 20
    assert len(TOOL._PRIOR_V2R13_RUNTIME_ROOT_STATES_IN_EXACT_ORDER) == 20
    assert TOOL._PRIOR_V2R13_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 6
    assert TOOL._PRIOR_V2R13_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 14
    assert TOOL._PRIOR_V2R13_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R13_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R13_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R14_RUNTIME_ROOTS) == 20
    assert len(TOOL._PRIOR_V2R14_RUNTIME_ROOT_STATES_IN_EXACT_ORDER) == 20
    assert TOOL._PRIOR_V2R14_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 6
    assert TOOL._PRIOR_V2R14_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 14
    assert TOOL._PRIOR_V2R14_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R14_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R14_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R15_RUNTIME_ROOTS) == 20
    assert len(TOOL._PRIOR_V2R15_RUNTIME_ROOT_STATES_IN_EXACT_ORDER) == 20
    assert TOOL._PRIOR_V2R15_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 2
    assert TOOL._PRIOR_V2R15_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 18
    assert TOOL._PRIOR_V2R15_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R15_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R15_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R16_RUNTIME_ROOTS) == 20
    assert len(TOOL._PRIOR_V2R16_RUNTIME_ROOT_STATES_IN_EXACT_ORDER) == 20
    assert TOOL._PRIOR_V2R16_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 14
    assert TOOL._PRIOR_V2R16_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 6
    assert TOOL._PRIOR_V2R16_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R16_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R16_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R17_RUNTIME_ROOTS) == 20
    assert TOOL._PRIOR_V2R17_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 2
    assert TOOL._PRIOR_V2R17_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 18
    assert TOOL._PRIOR_V2R17_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R17_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R17_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R18_RUNTIME_ROOTS) == 20
    assert TOOL._PRIOR_V2R18_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 6
    assert TOOL._PRIOR_V2R18_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 14
    assert TOOL._PRIOR_V2R18_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R18_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R18_RUNTIME_ROOTS,
    )
    assert len(TOOL._PRIOR_V2R19_RUNTIME_ROOTS) == 20
    assert TOOL._PRIOR_V2R19_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "PRESENT"
    ) == 6
    assert TOOL._PRIOR_V2R19_RUNTIME_ROOT_STATES_IN_EXACT_ORDER.count(
        "ABSENT"
    ) == 14
    assert TOOL._PRIOR_V2R19_PROTECTED_ROOTS == (
        TOOL._PRIOR_V2R19_WORKTREE_ROOT,
        *TOOL._PRIOR_V2R19_RUNTIME_ROOTS,
    )
    assert protected["preauthorization_historical_read_only_review_count"] == 11
    assert protected[
        "preauthorization_historical_read_only_review_scope_in_exact_order"
    ] == list(TOOL._PREAUTHORIZATION_HISTORICAL_READ_ONLY_REVIEW_SCOPE_IN_EXACT_ORDER)
    assert protected[
        "preauthorization_historical_read_only_review_write_or_mutation_count"
    ] == 0
    assert protected[
        "preauthorization_historical_read_only_review_was_execution_authority"
    ] is False
    assert protected[
        "preauthorization_historical_read_only_review_was_adoption_or_reuse_authority"
    ] is False
    assert protected["historical_campaign_root_open_access_prohibition_scope"] == (
        TOOL._HISTORICAL_CAMPAIGN_ROOT_PROHIBITION_SCOPE
    )
    assert protected[
        "historical_campaign_root_open_access_prohibition_does_not_deny_disclosed_preauthorization_read_only_review"
    ] is True
    assert protected[
        "prior_v2r17_preauthorization_bound_source_worktree_validation"
    ] == (
        TOOL._PRIOR_V2R17_PREAUTHORIZATION_BOUND_SOURCE_WORKTREE_VALIDATION_V1
    )
    bound_source_validation = protected[
        "prior_v2r17_preauthorization_bound_source_worktree_validation"
    ]
    assert type(bound_source_validation) is dict
    assert bound_source_validation[
        "bound_source_worktree_read_access_disclosed"
    ] is True
    assert bound_source_validation[
        "authorization_ledger_slot_evidence_or_runtime_state_roots_accessed"
    ] is False
    assert bound_source_validation[
        "bound_source_campaign_full_suite_invocation_count"
    ] == 1
    assert bound_source_validation[
        "bound_source_campaign_full_suite_status"
    ] == "PASS"
    assert bound_source_validation[
        "bound_source_campaign_full_suite_git_ignored_pytest_cache_write_disclosed"
    ] is True
    assert bound_source_validation[
        "bound_source_campaign_full_suite_git_ignored_pytest_cache_write_paths_in_exact_order"
    ] == [".pytest_cache/v/cache/nodeids"]
    assert bound_source_validation[
        "bound_source_campaign_full_suite_ephemeral_pytest_scratch_write_disclosed"
    ] is True
    assert bound_source_validation[
        "bound_source_status_porcelain_excludes_git_ignored_paths"
    ] is True
    assert bound_source_validation[
        "formal_authorization_ledger_slot_evidence_or_runtime_state_write_or_mutation_count"
    ] == 0
    assert bound_source_validation[
        "live_strict_loader_source_worktree_write_or_mutation_count_or_null"
    ] == 0
    assert bound_source_validation[
        "live_strict_loader_validation_invocation_count"
    ] == 1
    assert bound_source_validation[
        "live_strict_loader_validation_status_or_null"
    ] == "PASS"
    live_gauntlet = bound_source_validation[
        "preauthorization_live_admission_gauntlet"
    ]
    assert type(live_gauntlet) is dict
    assert live_gauntlet["controller_commit"] == (
        "fb9219ac2d0853624cb8576fbd61ac3e75817b66"
    )
    assert live_gauntlet["controller_tree"] == (
        "4e6e387618c3055c766615da15aadc8c435bcba3"
    )
    assert live_gauntlet["checkpoint_raw_sha256"] == (
        "43be54a56e1621f5ba871dfa1b32a010b8e35c74e74cad2bc3bf043ec4b7545a"
    )
    assert live_gauntlet["checkpoint_byte_count"] == 68_543
    assert live_gauntlet["all_cases_passed"] is True
    assert live_gauntlet["supervisor_invocation_count"] == 0
    assert live_gauntlet["authorization_present"] is False
    assert live_gauntlet["ledger_present"] is False
    assert live_gauntlet["formal_root_write_count"] == 0
    assert live_gauntlet["docker_contacted"] is False
    assert live_gauntlet["network_contacted"] is False
    assert live_gauntlet["live_admission_scratch_cleared"] is True
    live_scheduling = live_gauntlet["scheduling_admission"]
    assert type(live_scheduling) is dict
    assert live_scheduling["selected_window_ordinal"] == 2
    assert live_scheduling["passed"] is True
    assert live_scheduling[
        "window_progression_was_one_gate_not_capsule_run_or_retry"
    ] is True
    live_windows = live_scheduling["windows_in_strict_order"]
    assert type(live_windows) is list
    assert [row["window_ordinal"] for row in live_windows] == [1, 2]
    assert [row["passed"] for row in live_windows] == [False, True]
    assert live_windows[0]["sustained_publication"][
        "timed_records_durable_by_deadline"
    ] == 565
    assert live_windows[0]["sustained_publication"][
        "dropped_record_count"
    ] == 4_235
    assert live_windows[1]["sustained_publication"][
        "timed_records_durable_by_deadline"
    ] == 4_800
    assert live_windows[1]["sustained_publication"][
        "records_durable_total"
    ] == 10_211
    assert live_windows[1]["sustained_publication"][
        "dropped_record_count"
    ] == 0
    assert bound_source_validation[
        "git_metadata_replayed_before_live_in_exact_order"
    ] == ["HEAD", "TREE", "STATUS_PORCELAIN"]
    assert bound_source_validation[
        "git_metadata_replayed_after_strict_loader_in_exact_order"
    ] == ["STATUS_PORCELAIN"]

    slots = contract["slot_registry"]
    assert type(slots) is list
    fixed = contract["fixed_external_roots"]
    assert type(fixed) is dict
    active_roots = {
        fixed["authorization_root"],
        fixed["ledger_root"],
    }
    for ordinal, slot in enumerate(slots, start=1):
        assert type(slot) is dict
        assert slot["trajectory_trace_id"] == (
            f"q05b-shadow-retry-campaign-v2r20-slot-{ordinal:02d}"
        )
        active_roots.update(
            {
                slot["clone_root"],
                slot["cargo_cache_root"],
                slot["artifact_parent_root"],
                slot["scheduling_probe_root"],
                slot["evidence_root"],
                slot["trace_root"],
            }
        )
    assert len(active_roots) == 20
    assert active_roots.isdisjoint(protected_roots)

    mutations = []
    prior_id = deepcopy(contract)
    prior_id["contract_id"] = TOOL._PRIOR_V2_CONTRACT_ID
    mutations.append(prior_id)

    reusable_prior = deepcopy(contract)
    reusable_prior["protected_state_policy"][
        "prior_v2_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior)

    reusable_prior_v2r1 = deepcopy(contract)
    reusable_prior_v2r1["protected_state_policy"][
        "prior_v2r1_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r1)

    reusable_prior_v2r2 = deepcopy(contract)
    reusable_prior_v2r2["protected_state_policy"][
        "prior_v2r2_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r2)

    reusable_prior_v2r3 = deepcopy(contract)
    reusable_prior_v2r3["protected_state_policy"][
        "prior_v2r3_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r3)

    reusable_prior_v2r4 = deepcopy(contract)
    reusable_prior_v2r4["protected_state_policy"][
        "prior_v2r4_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r4)

    reusable_prior_v2r5 = deepcopy(contract)
    reusable_prior_v2r5["protected_state_policy"][
        "prior_v2r5_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r5)

    reusable_prior_v2r6 = deepcopy(contract)
    reusable_prior_v2r6["protected_state_policy"][
        "prior_v2r6_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r6)

    reusable_prior_v2r7 = deepcopy(contract)
    reusable_prior_v2r7["protected_state_policy"][
        "prior_v2r7_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r7)

    reusable_prior_v2r8 = deepcopy(contract)
    reusable_prior_v2r8["protected_state_policy"][
        "prior_v2r8_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior_v2r8)

    for field in (
        "prior_v2r9_capsule_authorization_or_ledger_reuse_allowed",
        "prior_v2r9_campaign_root_open_allowed",
        "prior_v2r9_campaign_root_access_allowed",
    ):
        drifted_prior_v2r9_boundary = deepcopy(contract)
        drifted_prior_v2r9_boundary["protected_state_policy"][field] = True
        mutations.append(drifted_prior_v2r9_boundary)

    for field in (
        "prior_v2r10_capsule_authorization_or_ledger_reuse_allowed",
        "prior_v2r10_campaign_root_open_allowed",
        "prior_v2r10_campaign_root_access_allowed",
    ):
        drifted_prior_v2r10_boundary = deepcopy(contract)
        drifted_prior_v2r10_boundary["protected_state_policy"][field] = True
        mutations.append(drifted_prior_v2r10_boundary)

    for version in (
        "v2r11", "v2r12", "v2r13", "v2r14", "v2r15", "v2r16", "v2r17",
    ):
        for suffix in (
            "capsule_authorization_or_ledger_reuse_allowed",
            "campaign_root_open_allowed",
            "campaign_root_access_allowed",
        ):
            drifted_boundary = deepcopy(contract)
            drifted_boundary["protected_state_policy"][
                f"prior_{version}_{suffix}"
            ] = True
            mutations.append(drifted_boundary)

    missing_prior_v2r1_root = deepcopy(contract)
    missing_prior_v2r1_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R1_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r1_root)

    missing_prior_v2r2_root = deepcopy(contract)
    missing_prior_v2r2_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R2_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r2_root)

    missing_prior_v2r3_root = deepcopy(contract)
    missing_prior_v2r3_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R3_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r3_root)

    missing_prior_v2r4_root = deepcopy(contract)
    missing_prior_v2r4_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R4_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r4_root)

    missing_prior_v2r5_root = deepcopy(contract)
    missing_prior_v2r5_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R5_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r5_root)

    missing_prior_v2r6_root = deepcopy(contract)
    missing_prior_v2r6_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R6_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r6_root)

    missing_prior_v2r7_root = deepcopy(contract)
    missing_prior_v2r7_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R7_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r7_root)

    missing_prior_v2r8_root = deepcopy(contract)
    missing_prior_v2r8_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R8_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r8_root)

    missing_prior_v2r9_root = deepcopy(contract)
    missing_prior_v2r9_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R9_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r9_root)

    missing_prior_v2r10_root = deepcopy(contract)
    missing_prior_v2r10_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R10_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r10_root)

    missing_prior_v2r11_root = deepcopy(contract)
    missing_prior_v2r11_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(TOOL._PRIOR_V2R11_WORKTREE_ROOT)
    mutations.append(missing_prior_v2r11_root)

    missing_prior_v2r12_probe_root = deepcopy(contract)
    missing_prior_v2r12_probe_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(
        f"{TOOL._PRIOR_V2R12_STATE_PREFIX}-slot-01-scheduling-probe"
    )
    mutations.append(missing_prior_v2r12_probe_root)

    missing_prior_v2r13_probe_root = deepcopy(contract)
    missing_prior_v2r13_probe_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(
        f"{TOOL._PRIOR_V2R13_STATE_PREFIX}-slot-01-scheduling-probe"
    )
    mutations.append(missing_prior_v2r13_probe_root)

    missing_prior_v2r14_probe_root = deepcopy(contract)
    missing_prior_v2r14_probe_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(
        f"{TOOL._PRIOR_V2R14_STATE_PREFIX}-slot-01-scheduling-probe"
    )
    mutations.append(missing_prior_v2r14_probe_root)

    missing_prior_v2r15_probe_root = deepcopy(contract)
    missing_prior_v2r15_probe_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(
        f"{TOOL._PRIOR_V2R15_STATE_PREFIX}-slot-01-scheduling-probe"
    )
    mutations.append(missing_prior_v2r15_probe_root)

    missing_prior_v2r16_probe_root = deepcopy(contract)
    missing_prior_v2r16_probe_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(
        f"{TOOL._PRIOR_V2R16_STATE_PREFIX}-slot-01-scheduling-probe"
    )
    mutations.append(missing_prior_v2r16_probe_root)

    missing_prior_v2r17_probe_root = deepcopy(contract)
    missing_prior_v2r17_probe_root["protected_state_policy"][
        "protected_roots_in_exact_order"
    ].remove(
        f"{TOOL._PRIOR_V2R17_STATE_PREFIX}-slot-01-scheduling-probe"
    )
    mutations.append(missing_prior_v2r17_probe_root)

    prior_authorization_root = deepcopy(contract)
    prior_authorization_root["fixed_external_roots"]["authorization_root"] = (
        f"{TOOL._PRIOR_V2_STATE_PREFIX}-authorization"
    )
    mutations.append(prior_authorization_root)

    prior_v2r2_ledger_root = deepcopy(contract)
    prior_v2r2_ledger_root["fixed_external_roots"]["ledger_root"] = (
        f"{TOOL._PRIOR_V2R2_STATE_PREFIX}-ledger"
    )
    mutations.append(prior_v2r2_ledger_root)

    prior_v2r4_authorization_root = deepcopy(contract)
    prior_v2r4_authorization_root["fixed_external_roots"][
        "authorization_root"
    ] = f"{TOOL._PRIOR_V2R4_STATE_PREFIX}-authorization"
    mutations.append(prior_v2r4_authorization_root)

    prior_clone_root = deepcopy(contract)
    prior_clone_root["slot_registry"][0]["clone_root"] = (
        f"{TOOL._PRIOR_V2_STATE_PREFIX}-slot-01-clone"
    )
    mutations.append(prior_clone_root)

    prior_trace_id = deepcopy(contract)
    prior_trace_id["slot_registry"][0]["trajectory_trace_id"] = (
        "q05b-shadow-retry-campaign-v2-slot-01"
    )
    mutations.append(prior_trace_id)

    prior_v2r2_trace_id = deepcopy(contract)
    prior_v2r2_trace_id["slot_registry"][0]["trajectory_trace_id"] = (
        "q05b-shadow-retry-campaign-v2r2-slot-01"
    )
    mutations.append(prior_v2r2_trace_id)

    prior_v2r3_trace_id = deepcopy(contract)
    prior_v2r3_trace_id["slot_registry"][0]["trajectory_trace_id"] = (
        "q05b-shadow-retry-campaign-v2r3-slot-01"
    )
    mutations.append(prior_v2r3_trace_id)

    prior_v2r4_trace_id = deepcopy(contract)
    prior_v2r4_trace_id["slot_registry"][0]["trajectory_trace_id"] = (
        "q05b-shadow-retry-campaign-v2r4-slot-01"
    )
    mutations.append(prior_v2r4_trace_id)

    prior_v2r5_trace_id = deepcopy(contract)
    prior_v2r5_trace_id["slot_registry"][0]["trajectory_trace_id"] = (
        "q05b-shadow-retry-campaign-v2r5-slot-01"
    )
    mutations.append(prior_v2r5_trace_id)

    prior_v2r6_trace_id = deepcopy(contract)
    prior_v2r6_trace_id["slot_registry"][0]["trajectory_trace_id"] = (
        "q05b-shadow-retry-campaign-v2r6-slot-01"
    )
    mutations.append(prior_v2r6_trace_id)

    drifted_v2r2_fact = deepcopy(contract)
    drifted_v2r2_fact["protected_state_policy"][
        "prior_v2r2_terminal_ledger_record_digest"
    ] = _tagged("forged-v2r2-terminal")
    mutations.append(drifted_v2r2_fact)

    drifted_v2r3_fact = deepcopy(contract)
    drifted_v2r3_fact["protected_state_policy"][
        "prior_v2r3_completion_ledger_record_digest"
    ] = _tagged("forged-v2r3-completion")
    mutations.append(drifted_v2r3_fact)

    drifted_v2r4_fact = deepcopy(contract)
    drifted_v2r4_fact["protected_state_policy"][
        "prior_v2r4_completion_ledger_record_digest"
    ] = _tagged("forged-v2r4-completion")
    mutations.append(drifted_v2r4_fact)

    drifted_v2r5_fact = deepcopy(contract)
    drifted_v2r5_fact["protected_state_policy"][
        "prior_v2r5_completion_ledger_record_digest"
    ] = _tagged("forged-v2r5-completion")
    mutations.append(drifted_v2r5_fact)

    drifted_v2r5_type = deepcopy(contract)
    drifted_v2r5_type["protected_state_policy"][
        "prior_v2r5_slot_03_invoked"
    ] = 0
    mutations.append(drifted_v2r5_type)

    drifted_v2r6_fact = deepcopy(contract)
    drifted_v2r6_fact["protected_state_policy"][
        "prior_v2r6_observed_target_source_binding_digest"
    ] = _tagged("forged-v2r6-observed-target-source")
    mutations.append(drifted_v2r6_fact)

    drifted_v2r6_type = deepcopy(contract)
    drifted_v2r6_type["protected_state_policy"][
        "prior_v2r6_dependency_preflight_invocation_count"
    ] = False
    mutations.append(drifted_v2r6_type)

    drifted_v2r7_fact = deepcopy(contract)
    drifted_v2r7_fact["protected_state_policy"][
        "prior_v2r7_completion_ledger_record_digest"
    ] = _tagged("forged-v2r7-completion")
    mutations.append(drifted_v2r7_fact)

    drifted_v2r7_type = deepcopy(contract)
    drifted_v2r7_type["protected_state_policy"][
        "prior_v2r7_supervisor_invocation_count"
    ] = False
    mutations.append(drifted_v2r7_type)

    drifted_v2r8_fact = deepcopy(contract)
    drifted_v2r8_fact["protected_state_policy"][
        "prior_v2r8_completion_ledger_record_digest"
    ] = _tagged("forged-v2r8-completion")
    mutations.append(drifted_v2r8_fact)

    drifted_v2r8_type = deepcopy(contract)
    drifted_v2r8_type["protected_state_policy"][
        "prior_v2r8_supervisor_invocation_count"
    ] = False
    mutations.append(drifted_v2r8_type)

    drifted_v2r9_fact = deepcopy(contract)
    drifted_v2r9_fact["protected_state_policy"][
        "prior_v2r9_retired_capsule_digest"
    ] = _tagged("forged-v2r9-retired-capsule")
    mutations.append(drifted_v2r9_fact)

    drifted_v2r9_type = deepcopy(contract)
    drifted_v2r9_type["protected_state_policy"][
        "prior_v2r9_owner_authorization_created"
    ] = 0
    mutations.append(drifted_v2r9_type)

    drifted_v2r10_fact = deepcopy(contract)
    drifted_v2r10_fact["protected_state_policy"][
        "prior_v2r10_retired_authorization_digest"
    ] = _tagged("forged-v2r10-retired-authorization")
    mutations.append(drifted_v2r10_fact)

    drifted_v2r10_type = deepcopy(contract)
    drifted_v2r10_type["protected_state_policy"][
        "prior_v2r10_run_authorization_consumed"
    ] = 0
    mutations.append(drifted_v2r10_type)

    drifted_v2r11_fact = deepcopy(contract)
    drifted_v2r11_fact["protected_state_policy"][
        "prior_v2r11_completion_ledger_record_digest"
    ] = _tagged("forged-v2r11-completion")
    mutations.append(drifted_v2r11_fact)

    drifted_v2r12_fact = deepcopy(contract)
    drifted_v2r12_fact["protected_state_policy"][
        "prior_v2r12_slot_02_outcome_digest"
    ] = _tagged("forged-v2r12-slot-02-outcome")
    mutations.append(drifted_v2r12_fact)

    drifted_v2r12_type = deepcopy(contract)
    drifted_v2r12_type["protected_state_policy"][
        "prior_v2r12_slot_03_invoked"
    ] = 0
    mutations.append(drifted_v2r12_type)

    drifted_v2r13_fact = deepcopy(contract)
    drifted_v2r13_fact["protected_state_policy"][
        "prior_v2r13_slot_01_outcome_digest"
    ] = _tagged("forged-v2r13-slot-01-outcome")
    mutations.append(drifted_v2r13_fact)

    drifted_v2r13_type = deepcopy(contract)
    drifted_v2r13_type["protected_state_policy"][
        "prior_v2r13_slot_03_invoked"
    ] = 0
    mutations.append(drifted_v2r13_type)

    drifted_v2r14_fact = deepcopy(contract)
    drifted_v2r14_fact["protected_state_policy"][
        "prior_v2r14_slot_01_outcome_digest"
    ] = _tagged("forged-v2r14-slot-01-outcome")
    mutations.append(drifted_v2r14_fact)

    drifted_v2r14_type = deepcopy(contract)
    drifted_v2r14_type["protected_state_policy"][
        "prior_v2r14_offline_reclassification_changes_history"
    ] = 0
    mutations.append(drifted_v2r14_type)

    drifted_v2r15_fact = deepcopy(contract)
    drifted_v2r15_fact["protected_state_policy"][
        "prior_v2r15_terminal_ledger_record_digest"
    ] = _tagged("forged-v2r15-terminal")
    mutations.append(drifted_v2r15_fact)

    drifted_v2r15_type = deepcopy(contract)
    drifted_v2r15_type["protected_state_policy"][
        "prior_v2r15_slot_01_invoked"
    ] = 0
    mutations.append(drifted_v2r15_type)

    drifted_v2r16_fact = deepcopy(contract)
    drifted_v2r16_fact["protected_state_policy"][
        "prior_v2r16_terminal_ledger_record_digest"
    ] = _tagged("forged-v2r16-terminal")
    mutations.append(drifted_v2r16_fact)

    drifted_v2r16_type = deepcopy(contract)
    drifted_v2r16_type["protected_state_policy"][
        "prior_v2r16_supervisor_invocation_count"
    ] = False
    mutations.append(drifted_v2r16_type)

    drifted_v2r17_fact = deepcopy(contract)
    drifted_v2r17_fact["protected_state_policy"][
        "prior_v2r17_ledger_record_digests_in_exact_order"
    ][1] = _tagged("forged-v2r17-consumed")
    mutations.append(drifted_v2r17_fact)

    drifted_v2r17_type = deepcopy(contract)
    drifted_v2r17_type["protected_state_policy"][
        "prior_v2r17_supervisor_invocation_count"
    ] = False
    mutations.append(drifted_v2r17_type)

    for field, value in (
        ("preauthorization_historical_read_only_review_count", 6),
        (
            "preauthorization_historical_read_only_review_scope_in_exact_order",
            ["SELECTED_V2_THROUGH_V2R8_LEDGER_OUTCOME_AND_TRAJECTORY_BINDING_JSON"],
        ),
        (
            "preauthorization_historical_read_only_review_write_or_mutation_count",
            1,
        ),
        (
            "preauthorization_historical_read_only_review_was_execution_authority",
            True,
        ),
        (
            "historical_campaign_root_open_access_prohibition_scope",
            "ALL_TIME",
        ),
    ):
        drifted_disclosure = deepcopy(contract)
        drifted_disclosure["protected_state_policy"][field] = value
        mutations.append(drifted_disclosure)

    for field, value in (
        ("status", "FAIL"),
        (
            "authorization_ledger_slot_evidence_or_runtime_state_roots_accessed",
            True,
        ),
        (
            "bound_source_campaign_full_suite_git_ignored_pytest_cache_write_disclosed",
            False,
        ),
        (
            "bound_source_campaign_full_suite_git_ignored_pytest_cache_write_paths_in_exact_order",
            [],
        ),
        (
            "bound_source_campaign_full_suite_ephemeral_pytest_scratch_write_disclosed",
            False,
        ),
        ("bound_source_status_porcelain_excludes_git_ignored_paths", False),
        (
            "formal_authorization_ledger_slot_evidence_or_runtime_state_write_or_mutation_count",
            1,
        ),
        (
            "live_strict_loader_source_worktree_write_or_mutation_count_or_null",
            1,
        ),
        ("bound_source_campaign_full_suite_invocation_count", 0),
        ("bound_source_campaign_full_suite_status", "FAIL"),
        ("git_metadata_replayed_after_strict_loader_in_exact_order", []),
    ):
        drifted_bound_source_validation = deepcopy(contract)
        drifted_bound_source_validation["protected_state_policy"][
            "prior_v2r17_preauthorization_bound_source_worktree_validation"
        ][field] = value
        mutations.append(drifted_bound_source_validation)

    drifted_live_gauntlet = deepcopy(contract)
    drifted_live_gauntlet["protected_state_policy"][
        "prior_v2r17_preauthorization_bound_source_worktree_validation"
    ]["preauthorization_live_admission_gauntlet"]["scheduling_admission"][
        "selected_window_ordinal"
    ] = 1
    mutations.append(drifted_live_gauntlet)

    missing_bound_source_validation_path = deepcopy(contract)
    missing_bound_source_validation_path["protected_state_policy"][
        "prior_v2r17_preauthorization_bound_source_worktree_validation"
    ]["strict_loader_target_binding_paths_in_exact_registry_order"].pop()
    mutations.append(missing_bound_source_validation_path)

    for mutation in mutations:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._require_v2r20_instance_isolation_v2(mutation)
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


def test_v2r18_terminal_history_is_frozen_and_never_resumable() -> None:
    contract = _contract()
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    assert TOOL._prior_v2r18_historical_evidence_v2(contract) == (
        TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS
    )
    assert protected["prior_v2r18_terminal_status"] == (
        "CAMPAIGN_FAILED_CLOSED_TERMINAL"
    )
    assert protected["prior_v2r18_ledger_record_count"] == 8
    assert protected["prior_v2r18_supervisor_invocation_count"] == 1
    assert protected["prior_v2r18_last_consumed_slot_id"] == "SLOT_01"
    assert protected["prior_v2r18_completion_commit_verified"] is True
    assert protected[
        "prior_v2r18_terminal_is_historical_nonresumable_evidence_only"
    ] is True
    assert protected["prior_v2r18_campaign_root_open_allowed"] is False
    assert protected["prior_v2r18_campaign_root_access_allowed"] is False
    assert protected[
        "prior_v2r18_capsule_authorization_or_ledger_reuse_allowed"
    ] is False
    assert set(TOOL._PRIOR_V2R18_PROTECTED_ROOTS).issubset(
        protected["protected_roots_in_exact_order"]
    )
    for field, forged_value in (
        ("prior_v2r18_campaign_root_open_allowed", True),
        ("prior_v2r18_campaign_root_access_allowed", True),
        (
            "prior_v2r18_capsule_authorization_or_ledger_reuse_allowed",
            True,
        ),
        (
            "prior_v2r18_terminal_is_historical_nonresumable_evidence_only",
            False,
        ),
        ("prior_v2r18_offline_corrected_decision", "TERMINAL_NONRETRYABLE"),
    ):
        mutation = deepcopy(contract)
        mutation["protected_state_policy"][field] = forged_value
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._require_v2r20_instance_isolation_v2(mutation)
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


def test_v2r19_terminal_size_failure_history_and_observer_cleanup_are_frozen() -> None:
    contract = _contract()
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    assert TOOL._prior_v2r19_historical_evidence_v2(contract) == (
        TOOL._PRIOR_V2R19_PROTECTED_POLICY_FACTS
    )
    assert protected["prior_v2r19_slot_01_decision_audit"] == (
        TOOL._PRIOR_V2R19_DECISION_AUDIT
    )
    assert protected["prior_v2r19_slot_01_resource_sample_count"] == 8_047
    assert protected["prior_v2r19_slot_01_gap_exceeded_count"] == 0
    assert protected["prior_v2r19_slot_01_failed_stage"] == "STAGE_09"
    assert protected["prior_v2r19_slot_01_stage_10_entered"] is False
    assert protected[
        "prior_v2r19_slot_01_observed_artifact_canonical_bytes"
    ] == 4_730_892_384
    assert protected[
        "prior_v2r19_slot_01_frozen_artifact_max_canonical_bytes"
    ] == 2_147_483_648
    assert protected["prior_v2r19_slot_02_invoked"] is False
    assert protected["prior_v2r19_slot_03_invoked"] is False
    assert protected["prior_v2r19_post_terminal_observer_pyc_currently_absent"] is True
    assert protected[
        "prior_v2r19_post_terminal_observer_contamination_execution_authority"
    ] is False
    assert protected[
        "prior_v2r19_terminal_is_historical_nonresumable_evidence_only"
    ] is True
    for field, forged_value in (
        ("prior_v2r19_slot_01_resource_sample_count", 8_046),
        ("prior_v2r19_slot_01_failed_stage", "STAGE_08"),
        ("prior_v2r19_slot_02_invoked", True),
        ("prior_v2r19_post_terminal_observer_pyc_currently_absent", False),
        ("prior_v2r19_campaign_root_open_allowed", True),
    ):
        mutation = deepcopy(contract)
        mutation["protected_state_policy"][field] = forged_value
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._require_v2r20_instance_isolation_v2(mutation)
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


def test_prior_v2r8_exact_eight_record_stage8_terminal_history_is_frozen() -> None:
    protected = _contract()["protected_state_policy"]
    assert type(protected) is dict
    assert protected["prior_v2r8_ledger_record_count"] == 8
    assert protected["prior_v2r8_ledger_record_names_in_sequence_order"] == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
        "slot-01.intent.json",
        "slot-01.consumed.json",
        "slot-01.trajectory-binding.json",
        "slot-01.outcome.json",
        "campaign.terminal.json",
        "campaign.completion.json",
    ]
    assert protected["prior_v2r8_supervisor_invocation_count"] == 1
    assert protected["prior_v2r8_campaign_tool_raw_sha256"] == (
        "f756d70fadc6553e58fc8d80fc3c001b39de27ab09798a8db3a85966ed4a603a"
    )
    assert protected["prior_v2r8_campaign_config_raw_sha256"] == (
        "d9a5f496c7b3dffb66b403ef3c88ae5cde4b5f83e12338200984d02c98de0fa6"
    )
    assert "prior_v2r8_target_tool_raw_sha256" not in protected
    assert "prior_v2r8_target_config_raw_sha256" not in protected
    assert protected["prior_v2r8_last_slot_id"] == "SLOT_01"
    assert protected["prior_v2r8_slot_01_decision"] == "TERMINAL_NONRETRYABLE"
    assert protected["prior_v2r8_slot_01_trajectory_status"] == "COMPLETE_FAIL"
    assert protected["prior_v2r8_slot_01_trajectory_event_count"] == 14_536
    assert protected["prior_v2r8_slot_01_failure_rows_retained_count"] == 2
    assert protected["prior_v2r8_slot_01_failure_operations_in_sequence_order"] == [
        "SUPERVISOR_STAGE",
        "SUPERVISOR_RUN",
    ]
    assert protected["prior_v2r8_slot_01_terminal_error_code"] == (
        "FAIL_Q05B_DUAL_ARTIFACT"
    )
    assert protected["prior_v2r8_slot_01_terminal_error_detail"] == (
        "candidate evidence strict replay failed: REJECT_Q05B_ARTIFACT_HOST: "
        "loaded module closure is incomplete"
    )
    assert protected["prior_v2r8_slot_01_terminal_stage"] == (
        "STAGE_08_BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE"
    )
    assert protected["prior_v2r8_scheduling_admission_window_count"] == 1
    assert protected["prior_v2r8_scheduling_admission_passed"] is True
    assert protected[
        "prior_v2r8_sustained_timed_records_durable_by_deadline"
    ] == 4_800
    assert protected["prior_v2r8_sustained_dropped_records"] == 0
    assert protected["prior_v2r8_sustained_cleanup_absence"] is True
    assert protected["prior_v2r8_slot_01_artifact_state"] == "ABSENT"
    assert protected[
        "prior_v2r8_slot_01_artifact_absence_verified_and_replayed"
    ] is True
    assert protected["prior_v2r8_slot_02_invoked"] is False
    assert protected["prior_v2r8_slot_03_invoked"] is False
    assert protected["prior_v2r8_terminal_error_code_or_null"] is None


def test_retired_v2r9_capsule_is_not_v2r17_authority(tmp_path: Path) -> None:
    contract, capsule, authorization, ledger, _result = _completed_real_ledger(
        tmp_path
    )
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    retired = protected["prior_v2r9_retired_capsule_digest"]
    assert retired == TOOL._PRIOR_V2R9_PROTECTED_POLICY_FACTS[
        "prior_v2r9_retired_capsule_digest"
    ]
    assert protected["prior_v2r9_capsule_retired"] is True
    assert protected["prior_v2r9_owner_authorization_created"] is False
    assert protected["prior_v2r9_record_authorization_invocation_count"] == 0
    assert protected["prior_v2r9_run_invocation_count"] == 0
    assert protected["prior_v2r9_supervisor_invocation_count"] == 0
    assert capsule["capsule_digest"] != retired
    rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                {"capsule_digest": retired},
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_retired_v2r10_capsule_and_authorization_are_not_v2r17_authority(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger, _result = _completed_real_ledger(
        tmp_path
    )
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    retired_capsule = protected["prior_v2r10_retired_capsule_digest"]
    assert retired_capsule == (
        "sha256:4ba7e59fe951e3c5b7c5aee0d8b4a1c4d3f74abec6099b49ee2d903af0c362ab"
    )
    assert protected["prior_v2r10_retired_authorization_digest"] == (
        "sha256:fd9042ea3c72671962e80eff78264a9e8905144ef40cf5921fa5c7746c972da2"
    )
    assert protected["prior_v2r10_capsule_retired"] is True
    assert protected["prior_v2r10_authorization_retired"] is True
    assert protected["prior_v2r10_run_authorization_lookup_count"] == 0
    assert protected["prior_v2r10_run_authorization_consumed"] is False
    assert protected["prior_v2r10_run_ledger_created"] is False
    assert protected["prior_v2r10_run_slot_root_created_count"] == 0
    assert protected["prior_v2r10_supervisor_invocation_count"] == 0
    assert capsule["capsule_digest"] != retired_capsule
    rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                {"capsule_digest": retired_capsule},
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_prior_v2r6_preflight_failure_receipt_and_zero_authority_are_exact() -> None:
    contract = _contract()
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    cli_error = {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-campaign-cli-error/2"
        ),
        "status": protected["prior_v2r6_preflight_cli_status"],
        "error_code": protected["prior_v2r6_preflight_error_code"],
        "error_detail": protected["prior_v2r6_preflight_error_detail"],
        "error_receipt_or_null": protected[
            "prior_v2r6_preflight_error_receipt_or_null"
        ],
        "supervisor_invocation_count_or_null": protected[
            "prior_v2r6_preflight_cli_supervisor_invocation_count_or_null"
        ],
        "resumable": protected["prior_v2r6_preflight_resumable"],
    }
    raw = TOOL._canonical_json_bytes(cli_error)
    assert len(raw) == protected["prior_v2r6_preflight_error_raw_byte_count"] == 290
    assert sha256(raw).hexdigest() == protected[
        "prior_v2r6_preflight_error_raw_sha256"
    ] == "8119c23e0df62403f01b3eddb5508664f0d49410e41f72b3196300c4c21fa1bb"
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert protected["prior_v2r6_preflight_exit_code"] == 1
    assert protected["prior_v2r6_preflight_stderr_empty"] is True
    assert protected["prior_v2r6_preflight_stderr_raw_sha256"] == sha256(
        b""
    ).hexdigest()
    assert protected["prior_v2r6_preflight_freshness_target_count"] == 16
    assert protected["prior_v2r6_runtime_root_count"] == 17
    assert protected["prior_v2r6_all_runtime_roots_absent"] is True
    assert protected["prior_v2r6_dependency_preflight_invocation_count"] == 0
    assert protected["prior_v2r6_capsule_digest_or_null"] is None
    assert protected["prior_v2r6_owner_confirmation_digest_or_null"] is None
    assert protected["prior_v2r6_authorization_digest_or_null"] is None
    assert protected["prior_v2r6_state_write_or_consumption_count"] == 0
    assert protected["prior_v2r6_supervisor_invocation_count"] == 0


def test_prior_v2r6_source_binding_failure_precedes_every_later_preflight_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    slots = deepcopy(contract["slot_registry"])
    calls: list[str] = []

    def slot_registry(_contract: Mapping[str, object]) -> list[object]:
        calls.append("slot-registry")
        return deepcopy(slots)

    def freshness(
        _contract: Mapping[str, object],
        _slots: Sequence[Mapping[str, object]],
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append("freshness")
        return {}

    def target_replay(_contract: Mapping[str, object]) -> NoReturn:
        calls.append("target-source-binding-compare")
        TOOL._fail("FAIL_CAMPAIGN_SOURCE", "target derived binding differs")

    def unreached(name: str):
        def invoke(*_args: object, **_kwargs: object) -> NoReturn:
            calls.append(name)
            raise AssertionError(f"v2r6 failure reached {name}")

        return invoke

    monkeypatch.setattr(TOOL, "_slot_registry", slot_registry)
    monkeypatch.setattr(TOOL, "_verify_fresh_targets", freshness)
    monkeypatch.setattr(TOOL, "_verify_target_git_objects", target_replay)
    for name in (
        "_controller_git_identity",
        "_scan_tree",
        "_executable_identity",
        "_python310_dependency_preflight_v1",
        "_build_capsule",
        "_authorization_template",
    ):
        monkeypatch.setattr(TOOL, name, unreached(name))

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._preflight_context(contract)
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"
    assert error.value.detail == "target derived binding differs"
    assert calls == [
        "slot-registry",
        "freshness",
        "target-source-binding-compare",
    ]


def test_v2r20_contract_is_source_only_or_exact_final_bound() -> None:
    raw_contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "v2r20 campaign contract binding phase",
    )
    assert type(raw_contract) is dict
    assert raw_contract["contract_id"] == (
        "hegel-phase3a-q05b-shadow-retry-campaign-v2r20"
    )
    assert raw_contract["target_commit_a_binding"]["target_supervisor_commit"] == (
        "5b38581f43c9fd88692b4f3c2cf8f13e8a0ca24b"
    )
    assert raw_contract["implementation_state"]["target_commit_a_bound"] is True
    if raw_contract["status"] == (
        "V2R20_SOURCE_IMPLEMENTED_AWAITING_FINAL_BINDING_NOT_EXECUTED"
    ):
        assert raw_contract["campaign_source_binding"]["status"] == (
            "CAMPAIGN_SOURCE_AWAITING_FINAL_BINDING"
        )
        assert raw_contract["authority_boundary"]["preflight_authorized"] is True
        assert raw_contract["implementation_state"]["campaign_source_bound"] is False
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"
    else:
        assert raw_contract["status"] == (
            "V2R20_FINAL_BOUND_AWAITING_EXACT_OWNER_AUTHORIZATION_NOT_EXECUTED"
        )
        assert raw_contract["campaign_source_binding"]["status"] == (
            "CAMPAIGN_SOURCE_BOUND"
        )
        assert raw_contract["authority_boundary"]["preflight_authorized"] is True
        assert raw_contract["implementation_state"]["campaign_source_bound"] is True
        assert (
            TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
            == raw_contract
        )


def test_contract_loader_rejects_rehashed_v2r2_trajectory_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    contract["trajectory_binding_policy"]["producer_id"] = (
        "Q05B_SUPERVISOR_V2"
    )
    unsigned_contract = deepcopy(contract)
    unsigned_contract.pop("contract_digest")
    contract["contract_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["contract"],
        unsigned_contract,
    )
    monkeypatch.setattr(
        TOOL,
        "_pretty_json",
        lambda _raw, _label: deepcopy(contract),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"
    assert error.value.detail == (
        "trajectory binding domain or v2r20 producer differs"
    )


def test_v2r20_target_d_binding_is_exact() -> None:
    contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "v2r20 exact Target D binding",
    )
    assert type(contract) is dict
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    assert target["status"] == "TARGET_COMMIT_A_BOUND"
    assert target["target_supervisor_commit"] == (
        "5b38581f43c9fd88692b4f3c2cf8f13e8a0ca24b"
    )
    assert target["target_supervisor_tree"] == (
        "d979f4e102ed16d99dd6eaf9d2c9a77ce3a9c57c"
    )
    assert target["target_supervisor_parent_commit"] == (
        "6bc3f93cae23ff45e84b1d3d389594a52b756585"
    )
    assert target["target_supervisor_commit_object_raw_sha256"] == (
        "b02b3932f3a769be39065d62919bd5fbec0c6dd888ce0cd6dd06e2aa186b61bb"
    )
    literal_required_paths = [
        "config/phase3_q05b_dual_isolation_v2.json",
        "src/hegel_machine/_vendor/__init__.py",
        "src/hegel_machine/_vendor/tomli/LICENSE",
        "src/hegel_machine/_vendor/tomli/__init__.py",
        "src/hegel_machine/_vendor/tomli/_parser.py",
        "src/hegel_machine/_vendor/tomli/_re.py",
        "src/hegel_machine/_vendor/tomli/_types.py",
        "src/hegel_machine/phase3_q05b_actual_admission_v2.py",
        "src/hegel_machine/phase3_q05b_actual_artifact_v2.py",
        "src/hegel_machine/phase3_q05b_trajectory_v2.py",
        "tests/test_phase3_q05b_dual_qualification_v2.py",
        "tests/test_phase3_q05b_trajectory_v2.py",
        "tools/phase3_q05b_dual_qualification_v2.py",
    ]
    assert target["required_paths_in_exact_order"] == literal_required_paths
    assert list(target["required_raw_sha256_by_path"]) == literal_required_paths
    assert target["required_raw_sha256_by_path"] == {
        "config/phase3_q05b_dual_isolation_v2.json": (
            "159b4e80b60d59005bd96b2b38136cb6504f0700319be0cd99886c3d487569f6"
        ),
        "src/hegel_machine/_vendor/__init__.py": (
            "ee9748e550c7a30a129da0ca1fb2a93116f599541a57039ebd687a679dd86673"
        ),
        "src/hegel_machine/_vendor/tomli/LICENSE": (
            "b80816b0d530b8accb4c2211783790984a6e3b61922c2b5ee92f3372ab2742fe"
        ),
        "src/hegel_machine/_vendor/tomli/__init__.py": (
            "9eb042d7c0db5d14c2168ec4946e410de5a91c9cce86892f5e4db5e4633c6762"
        ),
        "src/hegel_machine/_vendor/tomli/_parser.py": (
            "a412234c86bf710b361e0943276961f0e25fa6d7c36ba7a0e7eec87a3e018c7b"
        ),
        "src/hegel_machine/_vendor/tomli/_re.py": (
            "a12359fe294523a72112e434d58452a14c9d050affa2417f9927474e4166bfdd"
        ),
        "src/hegel_machine/_vendor/tomli/_types.py": (
            "f864c6d9552a929c7032ace654ee05ef26ca75d21b027b801d77e65907138b74"
        ),
        "src/hegel_machine/phase3_q05b_actual_admission_v2.py": (
            "9983f41a5d4888dc280cb96018ed79534fc9c1cd9bfedbee14ce689dad7c8847"
        ),
        "src/hegel_machine/phase3_q05b_actual_artifact_v2.py": (
            "e9e9eee9907bcb46728c87bf56830773c4de1a405b7bc5eb2545be10c1963e74"
        ),
        "src/hegel_machine/phase3_q05b_trajectory_v2.py": (
            "634826c13ee51c72c6e87c7b40f657ee315c3a91446ee12f2d807cdccc3030ed"
        ),
        "tests/test_phase3_q05b_dual_qualification_v2.py": (
            "95ac497e4ab210ca8d87466394ca1cb23430ae5e0ee1b743312250f7597ea18e"
        ),
        "tests/test_phase3_q05b_trajectory_v2.py": (
            "e38453beaa2ade4fc6a24669bfc2cccf7ded7b31bde63fac0d4e9fe3730f84b6"
        ),
        "tools/phase3_q05b_dual_qualification_v2.py": (
            "17fc3e126df9c76e841716415b3bcc9cdad2183e55ef4550fdba3e9212e84c07"
        ),
    }
    assert target["target_config_static_policy_root"] == (
        "67b0cb763fa2c81b256c1831fd5d75f4cc70ab315c90a1cbf1ac2111d29cba96"
    )
    assert target["target_contract_digest"] == (
        "sha256:159b4e80b60d59005bd96b2b38136cb6504f0700319be0cd99886c3d487569f6"
    )
    assert target["target_source_binding_digest"] == (
        "sha256:f62edc3b4efdd169c0ac71b1570c5dd83aa19af19a23d258f63d89397264bb61"
    )
    docker_labels = contract["execution_policy"]["target_docker_label_policy"]
    assert docker_labels["source_commit_label_value"] == (
        target["target_supervisor_commit"]
    )
    assert docker_labels["pre_spawn_and_post_reap_readonly_absence_filter"] == (
        "label=org.hegel.q05b.source_commit="
        "5b38581f43c9fd88692b4f3c2cf8f13e8a0ca24b"
    )
    dependency = contract["execution_policy"][
        "python310_dependency_preflight_gate"
    ]
    assert dependency["cargo_lock_relative_path"] not in target[
        "required_paths_in_exact_order"
    ]
    assert dependency["cargo_lock_raw_sha256"] == TOOL._CARGO_LOCK_RAW_SHA256_V1


def test_target_git_object_source_digest_strips_lf_and_rejects_lf_value() -> None:
    contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "v2r17 target Git-object source-digest KAT",
    )
    assert type(contract) is dict
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    repository = str(execution["clone_source_repository"])
    project_prefix = str(execution["clone_source_project_relative_path"])
    commit = str(target["target_supervisor_commit"])

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["/usr/bin/git", "-C", repository, *arguments],
            stderr=subprocess.PIPE,
        )

    tree = git("rev-parse", f"{commit}^{{tree}}").decode("ascii").strip()
    parent = git("rev-parse", f"{commit}^").decode("ascii").strip()
    commit_object = git("cat-file", "commit", commit)
    observed_hashes: dict[str, str] = {}
    target_config_raw = b""
    for relative in target["required_paths_in_exact_order"]:
        raw = git("show", f"{commit}:{project_prefix}/{relative}")
        observed_hashes[relative] = sha256(raw).hexdigest()
        assert observed_hashes[relative] == target[
            "required_raw_sha256_by_path"
        ][relative]
        if relative == "config/phase3_q05b_dual_isolation_v2.json":
            target_config_raw = raw
    target_config = TOOL._decoded_json_bytes(
        target_config_raw,
        "v2r17 target Git-object config KAT",
    )
    assert type(target_config) is dict
    static_policy = {
        key: value
        for key, value in target_config.items()
        if key not in {"engineering_status", "actual_preconditions"}
    }
    static_root = sha256(
        TOOL._TARGET_STATIC_POLICY_DOMAIN
        + TOOL._canonical_json_bytes(static_policy)
    ).hexdigest()
    target_contract_digest = "sha256:" + sha256(target_config_raw).hexdigest()
    preimage = {
        "target_supervisor_commit": commit,
        "target_supervisor_tree": tree,
        "target_supervisor_parent_commit": parent,
        "target_supervisor_commit_object_raw_sha256": sha256(
            commit_object
        ).hexdigest(),
        "required_raw_sha256_by_path": observed_hashes,
        "target_config_static_policy_root": static_root,
        "target_contract_digest": target_contract_digest,
    }
    canonical_with_lf = TOOL._canonical_json_bytes(preimage)
    assert canonical_with_lf.endswith(b"\n")
    assert len(canonical_with_lf) == 2_030
    assert len(canonical_with_lf[:-1]) == 2_029
    correct = TOOL._domain_digest(TOOL._TARGET_SOURCE_BINDING_DOMAIN, preimage)
    wrong_with_lf = "sha256:" + sha256(
        TOOL._TARGET_SOURCE_BINDING_DOMAIN.encode("ascii")
        + b"\x00"
        + canonical_with_lf
    ).hexdigest()
    assert correct == (
        "sha256:f62edc3b4efdd169c0ac71b1570c5dd83aa19af19a23d258f63d89397264bb61"
    )
    assert wrong_with_lf == (
        "sha256:6d8bbecc4176d967d55b80a752a5b852b82a396332740a9bd93023dc7b77c97c"
    )
    assert target["target_source_binding_digest"] == correct != wrong_with_lf
    replay = TOOL._verify_target_git_objects(contract)
    assert replay["target_source_binding_digest"] == correct

    wrong_contract = deepcopy(contract)
    wrong_contract["target_commit_a_binding"][
        "target_source_binding_digest"
    ] = wrong_with_lf
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._verify_target_git_objects(wrong_contract)
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"
    assert error.value.detail == "target derived binding differs"


def test_controller_commit_b_exact_six_path_registry_is_literal() -> None:
    literal = [
        "config/phase3_q05b_shadow_retry_campaign_v2.json",
        "docs/Hegel_Machine_Q05b_Shadow_Retry_Campaign_v2.md",
        "src/hegel_machine/phase3_q05b_shadow_retry_campaign_v2.py",
        "tests/test_phase3_q05b_shadow_retry_campaign_v2_core.py",
        "tests/test_phase3_q05b_shadow_retry_campaign_v2_tool.py",
        "tools/phase3_q05b_shadow_retry_campaign_v2.py",
    ]
    contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "v2r17 exact six-path controller registry",
    )
    assert list(TOOL._CONTROLLER_CHANGED_PATHS_FROM_TARGET_IN_EXACT_ORDER) == literal
    assert contract["campaign_source_binding"][
        "controller_commit_b_changed_paths_from_target_in_exact_order"
    ] == literal


def test_v2r20_instance_domains_preserve_version_neutral_domains() -> None:
    contract = _contract()
    assert contract["hash_domains"] == TOOL._EXPECTED_HASH_DOMAINS_V2R20
    assert all(
        "SHADOW_RETRY_CAMPAIGN_V2R20" in domain
        for domain in contract["hash_domains"].values()
    )
    assert TOOL._TARGET_SOURCE_BINDING_DOMAIN == (
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN/TARGET_SOURCE_BINDING/V2"
    )
    slot_value: dict[str, object] = {"record_digest": None}
    assert CORE.slot_observation_record_digest_v2(
        slot_value
    ) == CORE.campaign_record_digest_v2(
        "HEGEL/Q05B/SHADOW-RETRY-CAMPAIGN/SLOT-OBSERVATION/V2\x00",
        {},
    )


def test_v2r20_decision_audit_policy_is_exact_and_rehashed_drift_rejects() -> None:
    contract = _contract()
    policy = TOOL._decision_audit_policy_v1(contract)
    assert policy["schema_version"] == TOOL._DECISION_AUDIT_POLICY_SCHEMA_V1
    assert policy["core_api"] == "evaluate_slot_observation_with_audit_v2"
    assert policy["core_audit_schema_version"] == (
        CORE.DECISION_AUDIT_SCHEMA_V2
    )
    assert policy["predicate_clauses_in_exact_order"] == list(
        CORE.DECISION_AUDIT_CLAUSES_V2
    )
    assert policy["persistence_surfaces_in_exact_order"] == [
        "EXTERNAL_SLOT_RESULT",
        "LEDGER_SLOT_OUTCOME",
    ]
    unsigned = deepcopy(policy)
    observed_digest = unsigned.pop("policy_digest")
    assert observed_digest == TOOL._domain_digest(
        contract["hash_domains"]["decision_audit_policy"],
        unsigned,
    )
    for field, forged in (
        ("core_api", "evaluate_slot_observation_v2"),
        ("core_audit_schema_version", "forged/1"),
        (
            "predicate_clauses_in_exact_order",
            list(reversed(CORE.DECISION_AUDIT_CLAUSES_V2)),
        ),
        (
            "persistence_surfaces_in_exact_order",
            ["LEDGER_SLOT_OUTCOME", "EXTERNAL_SLOT_RESULT"],
        ),
        ("tamper_disposition", "FORGED"),
    ):
        mutation = deepcopy(contract)
        mutated_policy = mutation["decision_audit_policy"]
        mutated_policy[field] = forged
        unsigned_mutation = deepcopy(mutated_policy)
        unsigned_mutation.pop("policy_digest")
        mutated_policy["policy_digest"] = TOOL._domain_digest(
            mutation["hash_domains"]["decision_audit_policy"],
            unsigned_mutation,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._decision_audit_policy_v1(mutation)
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


@pytest.mark.parametrize(
    "domain_name",
    tuple(TOOL._EXPECTED_HASH_DOMAINS_V2R20),
)
def test_contract_loader_rejects_rehashed_v2r20_hash_domain_drift(
    monkeypatch: pytest.MonkeyPatch,
    domain_name: str,
) -> None:
    base = _contract()
    expected_domain = TOOL._EXPECTED_HASH_DOMAINS_V2R20[domain_name]
    prior_v2_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN",
    )
    prior_v2r2_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R2",
    )
    prior_v2r3_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R3",
    )
    prior_v2r4_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R4",
    )
    prior_v2r5_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R5",
    )
    prior_v2r6_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R6",
    )
    prior_v2r7_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R7",
    )
    prior_v2r8_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R8",
    )
    prior_v2r9_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R9",
    )
    prior_v2r10_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R10",
    )
    prior_v2r13_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R13",
    )
    prior_v2r14_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R14",
    )
    prior_v2r15_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R15",
    )
    prior_v2r17_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R20",
        "SHADOW_RETRY_CAMPAIGN_V2R17",
    )
    for forged_domain in (
        prior_v2_domain,
        prior_v2r2_domain,
        prior_v2r3_domain,
        prior_v2r4_domain,
        prior_v2r5_domain,
        prior_v2r6_domain,
        prior_v2r7_domain,
        prior_v2r8_domain,
        prior_v2r9_domain,
        prior_v2r10_domain,
        prior_v2r13_domain,
        prior_v2r14_domain,
        prior_v2r15_domain,
        prior_v2r17_domain,
        f"HEGEL/FORGED/Q05B/{domain_name.upper()}/V2",
    ):
        contract = deepcopy(base)
        domains = contract["hash_domains"]
        assert type(domains) is dict
        domains[domain_name] = forged_domain
        unsigned_contract = deepcopy(contract)
        unsigned_contract.pop("contract_digest")
        contract["contract_digest"] = TOOL._domain_digest(
            domains["contract"],
            unsigned_contract,
        )
        monkeypatch.setattr(
            TOOL,
            "_pretty_json",
            lambda _raw, _label, value=contract: deepcopy(value),
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"
        assert error.value.detail == "campaign hash-domain registry differs"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("clone_target_ref_name", "refs/hegel/drifted-target"),
        ("clone_ref_policy", "TEMPORARY_OR_MULTIPLE_REFS_ALLOWED"),
    ),
)
def test_contract_loader_rejects_clone_target_ref_policy_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    assert type(execution) is dict
    execution[field] = value
    monkeypatch.setattr(
        TOOL,
        "_pretty_json",
        lambda _raw, _label: deepcopy(contract),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"
    assert error.value.detail == "campaign authority boundary differs"


@pytest.mark.parametrize(
    ("historical_campaign_id", "historical_state_prefix", "historical_capsule", "historical_authorization"),
    (
        (
            TOOL._PRIOR_V2_CONTRACT_ID,
            TOOL._PRIOR_V2_STATE_PREFIX,
            TOOL._PRIOR_V2_CAPSULE_DIGEST,
            TOOL._PRIOR_V2_AUTHORIZATION_DIGEST,
        ),
        (
            TOOL._PRIOR_V2R1_CONTRACT_ID,
            TOOL._PRIOR_V2R1_STATE_PREFIX,
            TOOL._PRIOR_V2R1_CAPSULE_DIGEST,
            TOOL._PRIOR_V2R1_AUTHORIZATION_DIGEST,
        ),
        (
            TOOL._PRIOR_V2R2_CONTRACT_ID,
            TOOL._PRIOR_V2R2_STATE_PREFIX,
            TOOL._PRIOR_V2R2_CAPSULE_DIGEST,
            TOOL._PRIOR_V2R2_AUTHORIZATION_DIGEST,
        ),
        (
            TOOL._PRIOR_V2R3_CONTRACT_ID,
            TOOL._PRIOR_V2R3_STATE_PREFIX,
            TOOL._PRIOR_V2R3_CAPSULE_DIGEST,
            TOOL._PRIOR_V2R3_AUTHORIZATION_DIGEST,
        ),
        (
            TOOL._PRIOR_V2R4_CONTRACT_ID,
            TOOL._PRIOR_V2R4_STATE_PREFIX,
            TOOL._PRIOR_V2R4_CAPSULE_DIGEST,
            TOOL._PRIOR_V2R4_AUTHORIZATION_DIGEST,
        ),
        (
            TOOL._PRIOR_V2R5_PROTECTED_POLICY_FACTS[
                "prior_v2r5_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R5_STATE_PREFIX,
            TOOL._PRIOR_V2R5_PROTECTED_POLICY_FACTS[
                "prior_v2r5_capsule_digest"
            ],
            TOOL._PRIOR_V2R5_PROTECTED_POLICY_FACTS[
                "prior_v2r5_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R7_PROTECTED_POLICY_FACTS[
                "prior_v2r7_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R7_STATE_PREFIX,
            TOOL._PRIOR_V2R7_PROTECTED_POLICY_FACTS[
                "prior_v2r7_capsule_digest"
            ],
            TOOL._PRIOR_V2R7_PROTECTED_POLICY_FACTS[
                "prior_v2r7_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R8_PROTECTED_POLICY_FACTS[
                "prior_v2r8_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R8_STATE_PREFIX,
            TOOL._PRIOR_V2R8_PROTECTED_POLICY_FACTS[
                "prior_v2r8_capsule_digest"
            ],
            TOOL._PRIOR_V2R8_PROTECTED_POLICY_FACTS[
                "prior_v2r8_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R11_PROTECTED_POLICY_FACTS[
                "prior_v2r11_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R11_STATE_PREFIX,
            TOOL._PRIOR_V2R11_PROTECTED_POLICY_FACTS[
                "prior_v2r11_capsule_digest"
            ],
            TOOL._PRIOR_V2R11_PROTECTED_POLICY_FACTS[
                "prior_v2r11_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R12_PROTECTED_POLICY_FACTS[
                "prior_v2r12_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R12_STATE_PREFIX,
            TOOL._PRIOR_V2R12_PROTECTED_POLICY_FACTS[
                "prior_v2r12_capsule_digest"
            ],
            TOOL._PRIOR_V2R12_PROTECTED_POLICY_FACTS[
                "prior_v2r12_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R13_PROTECTED_POLICY_FACTS[
                "prior_v2r13_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R13_STATE_PREFIX,
            TOOL._PRIOR_V2R13_PROTECTED_POLICY_FACTS[
                "prior_v2r13_capsule_digest"
            ],
            TOOL._PRIOR_V2R13_PROTECTED_POLICY_FACTS[
                "prior_v2r13_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R14_PROTECTED_POLICY_FACTS[
                "prior_v2r14_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R14_STATE_PREFIX,
            TOOL._PRIOR_V2R14_PROTECTED_POLICY_FACTS[
                "prior_v2r14_capsule_digest"
            ],
            TOOL._PRIOR_V2R14_PROTECTED_POLICY_FACTS[
                "prior_v2r14_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R15_PROTECTED_POLICY_FACTS[
                "prior_v2r15_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R15_STATE_PREFIX,
            TOOL._PRIOR_V2R15_PROTECTED_POLICY_FACTS[
                "prior_v2r15_capsule_digest"
            ],
            TOOL._PRIOR_V2R15_PROTECTED_POLICY_FACTS[
                "prior_v2r15_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R16_PROTECTED_POLICY_FACTS[
                "prior_v2r16_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R16_STATE_PREFIX,
            TOOL._PRIOR_V2R16_PROTECTED_POLICY_FACTS[
                "prior_v2r16_capsule_digest"
            ],
            TOOL._PRIOR_V2R16_PROTECTED_POLICY_FACTS[
                "prior_v2r16_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R17_PROTECTED_POLICY_FACTS[
                "prior_v2r17_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R17_STATE_PREFIX,
            TOOL._PRIOR_V2R17_PROTECTED_POLICY_FACTS[
                "prior_v2r17_capsule_digest"
            ],
            TOOL._PRIOR_V2R17_PROTECTED_POLICY_FACTS[
                "prior_v2r17_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS[
                "prior_v2r18_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R18_STATE_PREFIX,
            TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS[
                "prior_v2r18_capsule_digest"
            ],
            TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS[
                "prior_v2r18_authorization_digest"
            ],
        ),
        (
            TOOL._PRIOR_V2R19_PROTECTED_POLICY_FACTS[
                "prior_v2r19_campaign_contract_id"
            ],
            TOOL._PRIOR_V2R19_STATE_PREFIX,
            TOOL._PRIOR_V2R19_PROTECTED_POLICY_FACTS[
                "prior_v2r19_capsule_digest"
            ],
            TOOL._PRIOR_V2R19_PROTECTED_POLICY_FACTS[
                "prior_v2r19_authorization_digest"
            ],
        ),
    ),
    ids=(
        "prior-v2",
        "prior-v2r1",
        "prior-v2r2",
        "prior-v2r3",
        "prior-v2r4",
        "prior-v2r5",
        "prior-v2r7",
        "prior-v2r8",
        "prior-v2r11",
        "prior-v2r12",
        "prior-v2r13",
        "prior-v2r14",
        "prior-v2r15",
        "prior-v2r16",
        "prior-v2r17",
        "prior-v2r18",
        "prior-v2r19",
    ),
)
def test_historical_capsule_authorization_and_rehashed_ledger_are_not_v2r20_authority(
    tmp_path: Path,
    historical_campaign_id: str,
    historical_state_prefix: str,
    historical_capsule: str,
    historical_authorization: str,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        copied_prior_capsule = {
            "capsule_digest": historical_capsule,
        }
        copied_prior_authorization = {
            "authorization_digest": historical_authorization,
        }
        rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                copied_prior_capsule,
                copied_prior_authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"

        prior_capsule_preimage = {
            "campaign_id": historical_campaign_id,
            "authorization_root": (
                f"{historical_state_prefix}-authorization"
            ),
            "ledger_root": f"{historical_state_prefix}-ledger",
        }
        rehashed_prior_capsule = {
            **prior_capsule_preimage,
            "capsule_digest": TOOL._domain_digest(
                contract["hash_domains"]["capsule"],
                prior_capsule_preimage,
            ),
        }
        rehashed_prior_authorization = deepcopy(authorization)
        rehashed_prior_authorization.update(
            {
                "campaign_id": historical_campaign_id,
                "contract_digest": _tagged("historical-contract"),
                "capsule_digest": rehashed_prior_capsule["capsule_digest"],
                "authorization_root": (
                    f"{historical_state_prefix}-authorization"
                ),
                "ledger_root": f"{historical_state_prefix}-ledger",
            }
        )
        confirmation_fields = (
            "campaign_id",
            "contract_digest",
            "target_supervisor_commit",
            "target_source_binding_digest",
            "campaign_source_binding_digest",
            "capsule_digest",
            "python310_dependency_preflight_receipt_digest",
            "preauthorization_live_gauntlet_receipt_digest",
            "decision_audit_policy_digest",
            "decision_audit_core_schema_version",
            "prior_v2r18_historical_evidence",
            "prior_v2r19_historical_evidence",
            "authorization_root",
            "ledger_root",
            "three_exact_slot_path_records",
            "authorized_campaign_execution_count",
            "maximum_supervisor_invocations",
            "strictly_sequential_one_shot_slots",
            "preconsumption_pending_validation_reentry_is_campaign_execution",
            "preconsumption_pending_validation_reentry_is_authorization_consumption",
            "preconsumption_pending_validation_reentry_is_retry_transition",
            "preconsumption_pending_validation_reentry_is_supervisor_invocation",
            "preconsumption_pending_reentry_requires_same_entrypoint",
            "preconsumption_pending_reentry_requires_same_capsule_digest",
            "preconsumption_pending_reentry_requires_fresh_complete_entry_replay",
            "preconsumption_pending_eligible_phase_condition_pairs_in_exact_order",
            "scheduling_checkpoint_reopen_validation_is_campaign_execution",
            "scheduling_checkpoint_reopen_validation_is_authorization_consumption",
            "scheduling_checkpoint_reopen_validation_is_retry_transition",
            "scheduling_checkpoint_reopen_validation_is_supervisor_invocation",
            "scheduling_pending_reentry_requires_same_run_entrypoint_and_capsule_digest",
            "scheduling_pending_reentry_requires_exact_locked_authoritative_prefix",
            "exact_clean_scheduling_checkpoint_continuation_authorized",
            "scheduling_pending_eligible_conditions_in_exact_order",
            "exact_clean_scheduling_checkpoint_requires_prior_pending_cli_receipt",
            "other_crash_state_resume_authorized",
            "retry_predicate_digest",
            "success_predicate_digest",
            "terminal_no_resume_policy",
            "forbidden_scope_digest",
        )
        authorization_fields = tuple(authorization)
        assert confirmation_fields == authorization_fields[
            2:authorization_fields.index("owner_confirmation_digest")
        ]
        rehashed_prior_authorization["owner_confirmation_digest"] = (
            TOOL._domain_digest(
                "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R20/"
                "OWNER_CONFIRMATION/V2",
                {
                    field: deepcopy(rehashed_prior_authorization[field])
                    for field in confirmation_fields
                },
            )
        )
        authorization_unsigned = deepcopy(rehashed_prior_authorization)
        authorization_unsigned.pop("authorization_digest")
        rehashed_prior_authorization["authorization_digest"] = (
            TOOL._domain_digest(
                contract["hash_domains"]["authorization"],
                authorization_unsigned,
            )
        )
        auth_contract = deepcopy(contract)
        auth_root = tmp_path / "rehashed-historical-authorization"
        auth_root.mkdir(mode=0o700)
        auth_root.chmod(0o700)
        auth_contract["fixed_external_roots"]["authorization_root"] = (
            os.fspath(auth_root)
        )
        auth_contract["authorization_policy"]["authorization_root"] = (
            os.fspath(auth_root)
        )
        auth_basename = str(
            auth_contract["authorization_policy"][
                "authorization_record_name"
            ]
        )
        auth_root_fd = os.open(
            auth_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            TOOL._publish_immutable_at(
                auth_root_fd,
                auth_basename,
                rehashed_prior_authorization,
            )
        finally:
            os.close(auth_root_fd)
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._load_authorization(auth_contract, capsule)
        assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"

        rehashed_rows = deepcopy(rows)
        previous_digest: str | None = None
        for _basename, record in rehashed_rows:
            record["campaign_id"] = historical_campaign_id
            record["contract_digest"] = _tagged("historical-contract")
            record["campaign_source_binding_digest"] = _tagged(
                "prior-v2-source-binding"
            )
            record["capsule_digest"] = historical_capsule
            record["authorization_digest"] = historical_authorization
            record["previous_record_digest_or_null"] = previous_digest
            unsigned = deepcopy(record)
            unsigned.pop("record_digest")
            record["record_digest"] = TOOL._domain_digest(
                contract["hash_domains"]["ledger_record"],
                unsigned,
            )
            previous_digest = str(record["record_digest"])
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                copied_prior_capsule,
                copied_prior_authorization,
                rehashed_rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_external_binding_exact_fields_digest_and_disk_replay(
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = str(trace_root)
    process = _process(b'{"result":"test"}\n', 1)
    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        terminal_error_detail="test terminal failure",
    )
    supervisor, external = _absent_artifacts()
    capsule = _capsule_stub(contract=contract)
    binding = TOOL._binding_record(
        contract,
        capsule,
        slot,
        process,
        {},
        trajectory,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    policy = contract["trajectory_binding_policy"]
    assert type(policy) is dict
    assert set(binding) == set(policy["external_binding_exact_fields"])
    unsigned = deepcopy(binding)
    unsigned.pop("binding_digest")
    assert binding["binding_digest"] == TOOL._domain_digest(
        contract["hash_domains"]["trajectory_binding"],
        unsigned,
    )
    payload = TOOL._canonical_json_bytes(binding)
    assert TOOL._strict_json(payload, "binding replay") == binding

    drifted = deepcopy(contract)
    drifted_policy = drifted["trajectory_binding_policy"]
    assert type(drifted_policy) is dict
    drifted_policy["external_binding_exact_fields"] = list(
        drifted_policy["external_binding_exact_fields"]
    )[:-1]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._binding_record(
            drifted,
            capsule,
            slot,
            process,
            {},
            trajectory,
            supervisor,
            external,
            True,
            [],
            None,
            _source_replay("source-before"),
            _source_replay("source-after"),
            _future_freshness(),
            _cleanup_replay(),
        )
    assert error.value.code == "FAIL_CAMPAIGN_BINDING"


def test_external_binding_trace_identity_failure_is_never_downgraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = os.fspath(trace_root)
    process = _process(b'{"result":"test"}\n', 1)
    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        terminal_error_detail="test terminal failure",
    )
    supervisor, external = _absent_artifacts()

    def fail_mount_identity(*_args: object, **_kwargs: object) -> int:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_FILESYSTEM",
            "injected trace mount-id failure",
            terminal=True,
        )

    monkeypatch.setattr(TOOL, "_fd_mount_id", fail_mount_identity)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._binding_record(
            contract,
            _capsule_stub(),
            slot,
            process,
            {},
            trajectory,
            supervisor,
            external,
            True,
            [],
            None,
            _source_replay("source-before"),
            _source_replay("source-after"),
            _future_freshness(),
            _cleanup_replay(),
        )
    assert error.value.code == "FAIL_CAMPAIGN_FILESYSTEM"
    assert "injected trace mount-id failure" in error.value.detail


def test_external_binding_persists_all_postprocess_failure_preimages(
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "trace"
    trace_root.mkdir(mode=0o700)
    slot["trace_root"] = os.fspath(trace_root)
    process = _process(b"not-success\n", 1)
    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        terminal_error_detail="trajectory exact reason",
    )
    supervisor, external = _absent_artifacts()
    supervisor["state"] = "RESIDUAL_INVALID"
    supervisor["error_or_null"] = "artifact replay exact reason"
    external["state"] = "RESIDUAL_INVALID"
    external["error_or_null"] = "artifact export exact reason"
    source_before = _source_replay("source-before")
    source_after = _source_replay("source-after")
    source_after["certain"] = False
    source_after["error_or_null"] = "source replay exact reason"
    source_after["evidence"] = {
        "error": "source replay exact reason",
        "changed_path": "target/ignored.tmp",
    }
    future = _future_freshness()
    future["certain"] = False
    future["error_or_null"] = "future slot exact reason"
    cleanup = {
        "preimage": {
            "docker_rows": [{"container_id": "ab" * 32}],
            "docker_error_or_null": "docker exact reason",
        },
        "root": _tagged("cleanup-errors"),
    }
    binding = TOOL._binding_record(
        contract,
        _capsule_stub(),
        slot,
        process,
        {},
        trajectory,
        supervisor,
        external,
        False,
        [{"container_id": "ab" * 32}],
        "docker exact reason",
        source_before,
        source_after,
        future,
        cleanup,
    )
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir(mode=0o700)
    descriptor = os.open(
        evidence_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        TOOL._publish_immutable_at(descriptor, "binding.json", binding)
    finally:
        os.close(descriptor)
    raw, _status = TOOL._read_regular_nofollow(
        evidence_root / "binding.json",
        "durable binding test",
    )
    replay = TOOL._strict_json(raw, "durable binding test")
    assert replay == binding
    for reason in (
        "trajectory exact reason",
        "artifact replay exact reason",
        "artifact export exact reason",
        "source replay exact reason",
        "future slot exact reason",
        "docker exact reason",
        "target/ignored.tmp",
    ):
        assert reason.encode("ascii") in raw


def test_slot_observation_success_gap_and_uncertainty_use_real_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)

    success_wire = _success_wire(contract, slot)
    success_stdout = CORE.canonical_campaign_json_bytes_v2(success_wire)
    success_process = _process(success_stdout, 0)
    success_supervisor, success_external = _valid_artifacts(success_wire)
    success = TOOL._build_slot_observation(
        contract,
        slot,
        success_process,
        CORE.classify_supervisor_stdout_v2(success_stdout),
        _trajectory(slot, outcome="OK"),
        success_supervisor,
        success_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert success["record_digest"] == CORE.slot_observation_record_digest_v2(
        success
    )
    assert CORE.evaluate_slot_observation_v2(success) == CORE.SUCCESS_V2

    gap_detail, gap_rows = _held_final_gap_rows()
    gap_stdout = CORE.canonical_campaign_json_bytes_v2(
        _error_wire(gap_detail)
    )
    gap_process = _process(gap_stdout, 1)
    gap_supervisor, gap_external = _absent_artifacts()
    gap = TOOL._build_slot_observation(
        contract,
        slot,
        gap_process,
        CORE.classify_supervisor_stdout_v2(gap_stdout),
        _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=gap_rows,
            terminal_error_detail=gap_detail,
        ),
        gap_supervisor,
        gap_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert CORE.evaluate_slot_observation_v2(gap) == CORE.RETRY_EXACT_GAP_V2

    uncertain_process = deepcopy(gap_process)
    uncertain_process["timed_out"] = True
    uncertain = TOOL._build_slot_observation(
        contract,
        slot,
        uncertain_process,
        CORE.classify_supervisor_stdout_v2(gap_stdout),
        _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=gap_rows,
            terminal_error_detail=gap_detail,
        ),
        gap_supervisor,
        gap_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert (
        CORE.evaluate_slot_observation_v2(uncertain)
        == CORE.TERMINAL_NONRETRYABLE_V2
    )

    slot_three = _slot(contract, 3)
    exhausted = TOOL._build_slot_observation(
        contract,
        slot_three,
        gap_process,
        CORE.classify_supervisor_stdout_v2(gap_stdout),
        _trajectory(
            slot_three,
            outcome="FAIL",
            failure_rows=gap_rows,
            terminal_error_detail=gap_detail,
        ),
        gap_supervisor,
        gap_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert (
        CORE.evaluate_slot_observation_v2(exhausted)
        == CORE.EXHAUSTED_RETRYABLE_GAP_V2
    )


def test_v2r18_full_formal_observation_replays_corrected_audit_and_outcome(
) -> None:
    spec = importlib.util.spec_from_file_location(
        "phase3_q05b_shadow_retry_campaign_v2_core_formal_kat",
        CORE_TEST_KAT_PATH,
    )
    assert spec is not None and spec.loader is not None
    fixture_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture_module
    spec.loader.exec_module(fixture_module)
    observation = fixture_module._v2r18_formal_observation_rebound_v2r20()
    contract = _contract()
    audit = TOOL._evaluate_slot_observation_with_exact_audit_v2(
        contract,
        observation,
        core_module=CORE,
    )
    assert audit == TOOL._PRIOR_V2R18_OFFLINE_CORRECTED_DECISION_AUDIT
    assert audit["decision"] == CORE.RETRY_EXACT_GAP_V2
    assert audit["matched_path_or_null"] == "EXACT_GAP"
    assert audit["first_reject_clause_or_null"] is None
    origin = observation["trajectory_failure_rows"][0]
    frozen_gap = TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS[
        "prior_v2r18_slot_01_physical_exact_gap"
    ]
    assert origin["actor_or_null"] == frozen_gap["actor_id"]
    assert origin["sample_ordinal_or_null"] == frozen_gap["sample_ordinal"]
    assert origin["gap_ns_or_null"] == frozen_gap["gap_ns"]
    assert origin["detail_or_null"] == frozen_gap["detail"]
    outcome = TOOL._slot_outcome_record(
        contract,
        _capsule_stub("formal-audit", contract=contract),
        _slot(contract),
        {"binding_digest": _tagged("formal-binding")},
        observation,
        audit,
    )
    assert outcome["decision"] == CORE.RETRY_EXACT_GAP_V2
    assert outcome["decision_audit"] == audit
    assert outcome["record_digest"] == TOOL._domain_digest(
        contract["hash_domains"]["slot_outcome"],
        {key: value for key, value in outcome.items() if key != "record_digest"},
    )


@pytest.mark.parametrize(
    ("slot_ordinal", "expected"),
    [
        (1, CORE.RETRY_EXACT_GAP_V2),
        (2, CORE.RETRY_EXACT_GAP_V2),
        (3, CORE.EXHAUSTED_RETRYABLE_GAP_V2),
    ],
)
def test_v2r17_trusted_host_stage6_raw_partition_drives_real_core(
    monkeypatch: pytest.MonkeyPatch,
    slot_ordinal: int,
    expected: str,
) -> None:
    contract = _contract()
    slot = _slot(contract, slot_ordinal)
    trace_id = str(slot["trajectory_trace_id"])
    detail, failure_rows, events = _trusted_host_stage6_gap_trace(trace_id)
    payloads = [TOOL._canonical_json_bytes(event) for event in events]
    partition = TOOL._trajectory_partition_evidence(
        contract,
        payloads,
        trajectory_status="COMPLETE_FAIL",
        failure_rows_in_sequence_order=failure_rows,
    )
    certificate = partition["partition_certificate_or_null"]
    assert type(certificate) is dict
    assert certificate["event_count"] == 16
    assert certificate["origin_sequence"] == 5
    assert certificate["suffix_event_count"] == 12
    assert certificate["failure_event_count"] == 4
    assert certificate["failure_rows_retained_count"] == 4
    assert certificate["failure_rows_truncated"] is False
    assert certificate["cleanup_nonfailure_event_count"] == 8
    assert certificate["cleanup_rows_retained_count"] == 8
    assert certificate["cleanup_rows_truncated"] is False
    assert certificate["other_nonfailure_event_count"] == 0
    assert certificate["partitioned_event_count"] == 12
    assert certificate["unpartitioned_event_count"] == 0
    cleanups = partition["cleanup_rows_retained_in_sequence_order"]
    assert [
        row["actor_or_null"]
        for row in cleanups
        if row["event_type"] == "BEGIN"
    ] == [
        "TRUSTED_HOST_REPLAY",
        "TRUSTED_HOST_REPLAY",
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
    ]

    trajectory = {
        "verification_status": "COMPLETE_FAIL",
        "complete": True,
        "complete_verification_passed": True,
        "verification_error_or_null": None,
        "trace_id_or_null": trace_id,
            "producer_id_or_null": "Q05B_SUPERVISOR_V6",
        "event_count": len(events),
        "first_event_record_digest_or_null": events[0]["record_digest"],
        "final_event_record_digest_or_null": events[-1]["record_digest"],
        "terminal_outcome_or_null": "FAIL",
        "terminal_manifest_digest_or_null": _tagged(
            "v2r14-trusted-host-terminal-manifest"
        ),
        "terminal_manifest_raw_sha256_or_null": _raw(
            "v2r14-trusted-host-terminal-manifest-raw"
        ),
        "clock_id_or_null": "CLOCK_MONOTONIC",
        "closed_span_count_or_null": 8,
        "terminal_event_type_or_null": "FAIL",
        "terminal_error_code_or_null": CORE.GAP_ERROR_CODE_V2,
        "terminal_error_detail_or_null": detail,
        "torn_tail_staging_basename_or_null": None,
        "open_span_ids_in_begin_order": [],
        "open_span_contexts_in_begin_order": [],
        "failure_rows_retained_in_sequence_order": deepcopy(
            partition["failure_rows_retained_in_sequence_order"]
        ),
        "cleanup_rows_retained_in_sequence_order": deepcopy(cleanups),
        "partition_certificate_or_null": deepcopy(certificate),
        "last_durable_event_row_or_null": deepcopy(
            partition["last_durable_event_row_or_null"]
        ),
        "canonical_event_payloads_digest": partition[
            "canonical_event_payloads_digest"
        ],
        "success_trace_bindings_or_null": None,
    }
    stdout = CORE.canonical_campaign_json_bytes_v2(_error_wire(detail))
    process = _process(stdout, 1)
    supervisor, external = _absent_artifacts()
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        trajectory,
        supervisor,
        external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert observation["trajectory_partition_certificate_or_null"] == (
        certificate
    )
    assert len(observation["trajectory_failure_rows"]) == 4
    assert len(observation["trajectory_cleanup_rows"]) == 8
    assert CORE.evaluate_slot_observation_v2(observation) == expected


def test_continuous_sampler_gap_preserves_real_two_tier_detail_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    terminal_detail, rows = _continuous_gap_rows()
    stdout = CORE.canonical_campaign_json_bytes_v2(
        _error_wire(terminal_detail)
    )
    supervisor, external = _absent_artifacts()
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        _process(stdout, 1),
        CORE.classify_supervisor_stdout_v2(stdout),
        _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=rows,
            terminal_error_detail=terminal_detail,
        ),
        supervisor,
        external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert (
        CORE.evaluate_slot_observation_v2(observation)
        == CORE.RETRY_EXACT_GAP_V2
    )


def test_real_stage4_to_stage5_benign_partition_and_cleanup_drive_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_parent = tmp_path / "trajectory-parent"
    trace_parent.mkdir(mode=0o700)
    trace_parent.chmod(0o700)
    trace_root = trace_parent / "trajectory"
    slot["trace_root"] = os.fspath(trace_root)
    previous = 100
    scheduled = previous + CORE.TARGET_WAIT_NS_V2
    actual = previous + CORE.MAXIMUM_GAP_NS_V2 + 1
    raw_detail = (
        "held actor sampling gap exceeded bound: "
        "actor_id=PYTHON_ENDPOINT; role_id=1; sample_ordinal=1; "
        f"previous_finish_ns={previous}; scheduled_start_ns={scheduled}; "
        f"actual_start_ns={actual}; gap_ns={actual - previous}; "
        f"maximum_gap_ns={CORE.MAXIMUM_GAP_NS_V2}; "
        f"target_wait_ns={CORE.TARGET_WAIT_NS_V2}; previous_duration_ns=0"
    )
    terminal_detail = "held actor sampler failed: " + raw_detail
    peer_previous = 1_000
    peer_scheduled = peer_previous + CORE.TARGET_WAIT_NS_V2
    peer_actual = peer_scheduled
    peer_finish = peer_actual + 10
    ticks = iter(range(1_000_000_000, 1_000_000_100))
    writer = TRAJECTORY.create_durable_trajectory_writer_v2(
        trace_root,
        expected_parent=trace_parent,
        trace_id=str(slot["trajectory_trace_id"]),
        producer_id="Q05B_SUPERVISOR_V6",
        clock_ns=lambda: next(ticks),
    )
    try:
        writer.begin(
            "supervisor-run",
            "SUPERVISOR_RUN",
            stage="RUN",
            detail="ENTER",
        )
        writer.begin(
            "supervisor-stage-04",
            "SUPERVISOR_STAGE",
            stage="STAGE_04",
            substep="RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
            detail="ENTER",
        )
        writer.begin(
            "sampler-PYTHON_ENDPOINT",
            "LIVE_RESOURCE_SAMPLER",
            stage="ACTOR_LIFECYCLE",
            substep="CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.begin(
            "sample-PYTHON_ENDPOINT-000001",
            "LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="PYTHON_ENDPOINT",
            detail="sample_ordinal=1",
        )
        writer.begin(
            "sampler-RUST_ENDPOINT",
            "LIVE_RESOURCE_SAMPLER",
            stage="ACTOR_LIFECYCLE",
            substep="CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            actor="RUST_ENDPOINT",
            detail="ENTER",
        )
        writer.begin(
            "sample-RUST_ENDPOINT-000001",
            "LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="RUST_ENDPOINT",
            detail="sample_ordinal=1",
        )
        writer.fail(
            "sample-PYTHON_ENDPOINT-000001",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=raw_detail,
            detail=(
                "MAXIMUM_UNSAMPLED_GAP_EXCEEDED;"
                f"native_wake_ns={scheduled};"
                "native_wake_lateness_ns=0;"
                f"trajectory_admission_duration_ns={actual - scheduled}"
            ),
            role_id="1",
            sample_ordinal=1,
            previous_finish_ns=previous,
            scheduled_start_ns=scheduled,
            actual_start_ns=actual,
            finish_ns=actual,
            collection_duration_ns=0,
            scheduler_lateness_ns=actual - scheduled,
            target_wait_ns=CORE.TARGET_WAIT_NS_V2,
            gap_ns=actual - previous,
            maximum_gap_ns=CORE.MAXIMUM_GAP_NS_V2,
            gap_exceeded=True,
        )
        writer.fail(
            "sampler-PYTHON_ENDPOINT",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=raw_detail,
            detail=(
                "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; "
                "sample_span=sample-PYTHON_ENDPOINT-000001"
            ),
        )
        writer.begin(
            "actor-cleanup-PYTHON_ENDPOINT-000001",
            "HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.ok(
            "actor-cleanup-PYTHON_ENDPOINT-000001",
            detail="cleanup_error_count=0",
        )
        writer.begin(
            "actor-cleanup-RUST_ENDPOINT-000002",
            "HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="RUST_ENDPOINT",
            detail="ENTER",
        )
        writer.ok(
            "sample-RUST_ENDPOINT-000001",
            detail="LIVE_RESOURCE_COLLECTION_COMPLETE",
            role_id="2",
            sample_ordinal=1,
            previous_finish_ns=peer_previous,
            scheduled_start_ns=peer_scheduled,
            actual_start_ns=peer_actual,
            finish_ns=peer_finish,
            collection_duration_ns=peer_finish - peer_actual,
            scheduler_lateness_ns=peer_actual - peer_scheduled,
            target_wait_ns=CORE.TARGET_WAIT_NS_V2,
            gap_ns=peer_actual - peer_previous,
            maximum_gap_ns=CORE.MAXIMUM_GAP_NS_V2,
            gap_exceeded=False,
        )
        writer.ok(
            "sampler-RUST_ENDPOINT",
            detail="sample_count=1",
        )
        writer.ok(
            "actor-cleanup-RUST_ENDPOINT-000002",
            detail="cleanup_error_count=0",
        )
        writer.begin(
            "actor-cleanup-PYTHON_ENDPOINT-000003",
            "HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.ok(
            "actor-cleanup-PYTHON_ENDPOINT-000003",
            detail="cleanup_error_count=0",
        )
        writer.ok(
            "supervisor-stage-04",
            detail="stage_evidence_root=" + _raw("real-stage-04"),
        )
        writer.begin(
            "supervisor-stage-05",
            "SUPERVISOR_STAGE",
            stage="STAGE_05",
            substep="SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            detail="ENTER",
        )
        writer.begin(
            "actor-close-PYTHON_ENDPOINT",
            "HELD_ACTOR_CLOSE",
            stage="ACTOR_LIFECYCLE",
            substep="DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.fail(
            "actor-close-PYTHON_ENDPOINT",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=terminal_detail,
            detail="ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
        )
        writer.fail(
            "supervisor-stage-05",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=terminal_detail,
            detail="FAILED",
        )
        writer.fail(
            "supervisor-run",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=terminal_detail,
            detail="supervisor attempt failed",
        )
        writer.seal_fail()
    finally:
        writer.close()

    monkeypatch.setattr(
        TOOL,
        "_trajectory_module",
        lambda _contract, _slot: TRAJECTORY,
    )
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    trajectory = TOOL._trajectory_evidence(contract, slot)
    certificate = trajectory["partition_certificate_or_null"]
    assert type(certificate) is dict
    assert certificate["origin_sequence"] == 7
    assert certificate["suffix_event_count"] == 16
    assert certificate["failure_event_count"] == 5
    assert certificate["failure_rows_retained_count"] == 5
    assert certificate["failure_rows_truncated"] is False
    assert certificate["cleanup_nonfailure_event_count"] == 6
    assert certificate["cleanup_rows_retained_count"] == 6
    assert certificate["cleanup_rows_truncated"] is False
    assert certificate["other_nonfailure_event_count"] == 5
    assert certificate["partitioned_event_count"] == 16
    assert certificate["unpartitioned_event_count"] == 0
    assert [
        row["operation"]
        for row in trajectory["failure_rows_retained_in_sequence_order"]
    ] == [
        "LIVE_RESOURCE_SAMPLE",
        "LIVE_RESOURCE_SAMPLER",
        "HELD_ACTOR_CLOSE",
        "SUPERVISOR_STAGE",
        "SUPERVISOR_RUN",
    ]
    cleanups = trajectory["cleanup_rows_retained_in_sequence_order"]
    assert [
        row["actor_or_null"]
        for row in cleanups
        if row["event_type"] == "OK"
    ] == ["PYTHON_ENDPOINT", "RUST_ENDPOINT", "PYTHON_ENDPOINT"]
    assert trajectory["last_durable_event_row_or_null"]["operation"] == (
        "SUPERVISOR_RUN"
    )
    stdout = CORE.canonical_campaign_json_bytes_v2(_error_wire(terminal_detail))
    process = _process(stdout, 1)
    supervisor, external = _absent_artifacts()
    binding = TOOL._binding_record(
        contract,
        _capsule_stub(),
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        trajectory,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    assert binding["partition_certificate_or_null"] == certificate
    assert binding["cleanup_rows_retained_in_sequence_order"] == cleanups
    assert binding["failure_rows_retained_in_sequence_order"] == trajectory[
        "failure_rows_retained_in_sequence_order"
    ]
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        trajectory,
        supervisor,
        external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert observation["trajectory_partition_certificate_or_null"] == certificate
    assert observation["trajectory_cleanup_rows"] == cleanups
    assert observation["trajectory_failure_rows"] == trajectory[
        "failure_rows_retained_in_sequence_order"
    ]
    assert CORE.evaluate_slot_observation_v2(observation) == (
        CORE.RETRY_EXACT_GAP_V2
    )


def test_real_incomplete_prefix_preserves_last_durable_event_certificate_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_parent = tmp_path / "incomplete-trajectory-parent"
    trace_parent.mkdir(mode=0o700)
    trace_parent.chmod(0o700)
    trace_root = trace_parent / "trajectory"
    slot["trace_root"] = os.fspath(trace_root)
    writer = TRAJECTORY.create_durable_trajectory_writer_v2(
        trace_root,
        expected_parent=trace_parent,
        trace_id=str(slot["trajectory_trace_id"]),
        producer_id="Q05B_SUPERVISOR_V6",
    )
    try:
        writer.begin(
            "supervisor-run",
            "SUPERVISOR_RUN",
            stage="RUN",
            detail="source_commit=" + "12" * 20,
        )
    finally:
        writer.close()
    monkeypatch.setattr(
        TOOL,
        "_trajectory_module",
        lambda _contract, _slot: TRAJECTORY,
    )
    evidence = TOOL._trajectory_evidence(contract, slot)
    assert evidence["verification_status"] == "INCOMPLETE_PREFIX"
    assert evidence["complete"] is False
    assert evidence["complete_verification_passed"] is False
    assert type(evidence["verification_error_or_null"]) is str
    assert evidence["terminal_error_code_or_null"] is None
    assert evidence["terminal_error_detail_or_null"] is None
    assert evidence["open_span_ids_in_begin_order"] == ["supervisor-run"]
    assert evidence["open_span_contexts_in_begin_order"][0]["operation"] == (
        "SUPERVISOR_RUN"
    )
    assert evidence["partition_certificate_or_null"] is None
    assert evidence["failure_rows_retained_in_sequence_order"] == []
    assert evidence["cleanup_rows_retained_in_sequence_order"] == []
    assert evidence["last_durable_event_row_or_null"] == {
        "sequence": 1,
        "timestamp_ns": evidence["last_durable_event_row_or_null"][
            "timestamp_ns"
        ],
        "event_type": "BEGIN",
        "span_id": "supervisor-run",
        "operation": "SUPERVISOR_RUN",
        "stage_or_null": "RUN",
        "substep_or_null": None,
        "actor_or_null": None,
        "detail_or_null": "source_commit=" + "12" * 20,
        "error_code_or_null": None,
        "error_detail_or_null": None,
        "record_digest": evidence["final_event_record_digest_or_null"],
    }
    assert type(evidence["canonical_event_payloads_digest"]) is str
    assert TOOL._trajectory_complete_and_bound(contract, slot, evidence) is False
    stdout = CORE.canonical_campaign_json_bytes_v2(
        _error_wire("controller observed incomplete trajectory")
    )
    process = _process(stdout, 1)
    supervisor, external = _absent_artifacts()
    binding = TOOL._binding_record(
        contract,
        _capsule_stub(),
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        evidence,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    assert binding["partition_certificate_or_null"] is None
    assert binding["failure_rows_retained_in_sequence_order"] == []
    assert binding["cleanup_rows_retained_in_sequence_order"] == []
    assert binding["last_durable_event_row_or_null"] == evidence[
        "last_durable_event_row_or_null"
    ]
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        evidence,
        supervisor,
        external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert observation["trajectory_partition_certificate_or_null"] is None
    assert observation["trajectory_failure_rows"] == []
    assert observation["trajectory_cleanup_rows"] == []
    assert CORE.evaluate_slot_observation_v2(observation) == (
        CORE.TERMINAL_NONRETRYABLE_V2
    )


def test_large_verified_prefix_uses_bounded_certificate_without_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_fsync(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("synthetic bounded-certificate KAT must not fsync")

    monkeypatch.setattr(TOOL.os, "fsync", unexpected_fsync)
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "large-prefix-trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = os.fspath(trace_root)
    trace_id = str(slot["trajectory_trace_id"])

    prefix_count = 8_192
    events: list[dict[str, object]] = []
    for sequence in range(1, prefix_count + 1):
        span_ordinal = (sequence + 1) // 2
        events.append(
            _event_record(
                trace_id=trace_id,
                sequence=sequence,
                timestamp=sequence,
                event_type="BEGIN" if sequence % 2 else "OK",
                span_id=f"bulk-nonfailure-{span_ordinal:06d}",
                operation="BULK_NONFAILURE_EVENT",
                stage="RUN",
                substep=None,
                actor=None,
                detail="x" * 256,
                record_digest="sha256:" + f"{sequence:064x}",
            )
        )

    terminal_detail, failure_rows = _held_final_gap_rows()
    suffix = _event_records_for_failures(trace_id, failure_rows)
    failures_by_digest = {
        row["record_digest"]: row for row in failure_rows
    }
    for event in suffix:
        old_sequence = int(event["sequence"])
        event["sequence"] = prefix_count + old_sequence
        if event["event_type"] != "FAIL":
            event["record_digest"] = _tagged(
                f"large-suffix-{old_sequence}-{event['event_type']}"
            )
        failure = failures_by_digest.get(event["record_digest"])
        if failure is not None:
            failure["sequence"] = event["sequence"]
            failure["timestamp_ns"] = event["timestamp_ns"]
    events.extend(suffix)

    old_projection = [
        {
            field: deepcopy(event[field])
            for field in TOOL._TRAJECTORY_MINIMAL_EVENT_FIELDS
        }
        for event in events
    ]
    old_bytes = TOOL._canonical_json_bytes(
        {"event_skeleton_rows_in_sequence_order": old_projection}
    )
    assert len(old_bytes) > TOOL._MAX_LEDGER_BYTES

    payloads = [TOOL._canonical_json_bytes(event) for event in events]
    partition = TOOL._trajectory_partition_evidence(
        contract,
        payloads,
        trajectory_status="COMPLETE_FAIL",
        failure_rows_in_sequence_order=failure_rows,
    )
    certificate = partition["partition_certificate_or_null"]
    assert type(certificate) is dict
    assert len(TOOL._canonical_json_bytes(certificate)) <= (
        TOOL._MAX_PARTITION_CERTIFICATE_BYTES
    )
    assert certificate["origin_sequence"] == prefix_count + 5
    assert certificate["suffix_event_count"] == 6
    assert certificate["failure_event_count"] == 4
    assert certificate["failure_rows_retained_count"] == 4
    assert certificate["failure_rows_truncated"] is False
    assert certificate["cleanup_nonfailure_event_count"] == 2
    assert certificate["cleanup_rows_retained_count"] == 2
    assert certificate["cleanup_rows_truncated"] is False
    assert certificate["other_nonfailure_event_count"] == 0
    assert certificate["partitioned_event_count"] == 6
    assert certificate["unpartitioned_event_count"] == 0

    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        failure_rows=_held_final_gap_rows()[1],
        terminal_error_detail=terminal_detail,
    )
    trajectory.update(
        {
            "event_count": len(events),
            "first_event_record_digest_or_null": events[0]["record_digest"],
            "final_event_record_digest_or_null": events[-1]["record_digest"],
            "failure_rows_retained_in_sequence_order": deepcopy(
                partition["failure_rows_retained_in_sequence_order"]
            ),
            "cleanup_rows_retained_in_sequence_order": deepcopy(
                partition["cleanup_rows_retained_in_sequence_order"]
            ),
            "partition_certificate_or_null": deepcopy(certificate),
            "last_durable_event_row_or_null": deepcopy(
                partition["last_durable_event_row_or_null"]
            ),
            "canonical_event_payloads_digest": partition[
                "canonical_event_payloads_digest"
            ],
        }
    )
    assert TOOL._trajectory_complete_and_bound(contract, slot, trajectory) is True
    process = _process(
        CORE.canonical_campaign_json_bytes_v2(_error_wire(terminal_detail)),
        1,
    )
    supervisor, external = _absent_artifacts()
    binding = TOOL._binding_record(
        contract,
        _capsule_stub(),
        slot,
        process,
        {},
        trajectory,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    binding_bytes = TOOL._canonical_json_bytes(binding)
    assert len(binding_bytes) <= TOOL._MAX_LEDGER_BYTES
    assert TOOL._strict_json(
        binding_bytes,
        "large-prefix bounded binding",
        TOOL._MAX_LEDGER_BYTES,
    ) == binding


def test_dependency_receipt_nested_semantic_mutation_rejects_rehashed_capsule() -> None:
    contract = _contract()
    capsule = _capsule_stub("nested-receipt")
    TOOL._authorization_template(contract, capsule)
    forged = deepcopy(capsule)
    receipt = forged["python310_dependency_preflight_receipt"]
    assert type(receipt) is dict
    receipt["vendor_closure_rows"][0][2] = "0" * 64
    unsigned_receipt = deepcopy(receipt)
    unsigned_receipt.pop("receipt_digest")
    receipt["receipt_digest"] = TOOL._domain_digest(
        TOOL._PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_DOMAIN_V1,
        unsigned_receipt,
    )
    forged["python310_dependency_preflight_receipt_digest"] = receipt[
        "receipt_digest"
    ]
    unsigned_capsule = deepcopy(forged)
    unsigned_capsule.pop("capsule_digest")
    forged["capsule_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["capsule"],
        unsigned_capsule,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._authorization_template(contract, forged)
    assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"


@pytest.mark.parametrize(
    "mutation",
    (
        "controller",
        "python",
        "runtime-environment",
        "stdout",
        "target-source-context",
        "cargo-object-context",
    ),
)
def test_dependency_receipt_rejects_fully_rehashed_context_or_stdout_forgery(
    mutation: str,
) -> None:
    contract = _contract()
    forged = deepcopy(_capsule_stub(f"dependency-context-{mutation}"))
    receipt = forged["python310_dependency_preflight_receipt"]
    assert type(receipt) is dict
    receipt_changed = True
    if mutation == "controller":
        receipt["controller_commit"] = "ab" * 20
    elif mutation == "python":
        receipt["python_executable"]["raw_sha256"] = _raw(
            "forged-python"
        )
    elif mutation == "runtime-environment":
        receipt["runtime_environment_digest"] = _tagged(
            "forged-runtime-environment"
        )
    elif mutation == "stdout":
        receipt["stdout"]["raw_sha256"] = _raw("forged-stdout")
    elif mutation == "target-source-context":
        receipt_changed = False
        target_git = forged["target_git"]
        assert type(target_git) is dict
        target_hashes = target_git["required_raw_sha256_by_path"]
        assert type(target_hashes) is dict
        target_hashes[
            "tools/phase3_q05b_dual_qualification_v2.py"
        ] = _raw("forged-supervisor-source-context")
    elif mutation == "cargo-object-context":
        receipt_changed = False
        target_git = forged["target_git"]
        assert type(target_git) is dict
        cargo_git = target_git["cargo_lock_git_object"]
        assert type(cargo_git) is dict
        cargo_git["raw_sha256"] = _raw("forged-cargo-object-context")
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(mutation)
    if receipt_changed:
        unsigned_receipt = deepcopy(receipt)
        unsigned_receipt.pop("receipt_digest")
        receipt["receipt_digest"] = TOOL._domain_digest(
            TOOL._PYTHON310_DEPENDENCY_PREFLIGHT_RECEIPT_DOMAIN_V1,
            unsigned_receipt,
        )
        forged["python310_dependency_preflight_receipt_digest"] = receipt[
            "receipt_digest"
        ]
    unsigned_capsule = deepcopy(forged)
    unsigned_capsule.pop("capsule_digest")
    forged["capsule_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["capsule"],
        unsigned_capsule,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._authorization_template(contract, forged)
    assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"


def test_authorization_rejects_each_live_gauntlet_capsule_binding_mutation(
) -> None:
    contract = _contract()
    baseline = _capsule_stub("gauntlet-binding", contract=contract)
    TOOL._authorization_template(contract, baseline)
    for mutation in (
        "receipt-digest",
        "raw-sha256",
        "byte-count",
        "root-identity",
        "record-mode",
        "record-uid",
        "record-nlink",
        "record-size",
    ):
        forged = deepcopy(baseline)
        if mutation == "receipt-digest":
            forged["preauthorization_live_gauntlet_receipt_digest"] = (
                _tagged("forged-gauntlet-receipt")
            )
        elif mutation == "raw-sha256":
            forged[
                "preauthorization_live_gauntlet_receipt_raw_sha256"
            ] = _raw("forged-gauntlet-raw")
        elif mutation == "byte-count":
            forged["preauthorization_live_gauntlet_receipt_byte_count"] += 1
        elif mutation == "root-identity":
            forged["preauthorization_live_gauntlet_root_identity"]["ino"] += 1
        elif mutation == "record-mode":
            forged["preauthorization_live_gauntlet_record_identity"][
                "mode"
            ] = stat.S_IFREG | 0o644
        elif mutation == "record-uid":
            forged["preauthorization_live_gauntlet_record_identity"][
                "uid"
            ] = os.geteuid() + 1
        elif mutation == "record-nlink":
            forged["preauthorization_live_gauntlet_record_identity"][
                "nlink"
            ] = 2
        elif mutation == "record-size":
            forged["preauthorization_live_gauntlet_record_identity"][
                "size"
            ] += 1
        else:  # pragma: no cover - the mutation registry is closed above.
            raise AssertionError(mutation)
        unsigned = deepcopy(forged)
        unsigned.pop("capsule_digest")
        forged["capsule_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["capsule"],
            unsigned,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._authorization_template(contract, forged)
        assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"


@pytest.mark.parametrize(
    "mutation",
    (
        "audit-policy",
        "audit-policy-digest",
        "audit-core-schema",
        "prior-v2r18-history",
        "prior-v2r19-history",
    ),
)
def test_authorization_rejects_decision_audit_or_prior_history_tamper(
    mutation: str,
) -> None:
    contract = _contract()
    baseline = _capsule_stub("audit-history-binding", contract=contract)
    authorization = TOOL._authorization_template(contract, baseline)
    assert authorization["decision_audit_policy_digest"] == (
        contract["decision_audit_policy"]["policy_digest"]
    )
    assert authorization["decision_audit_core_schema_version"] == (
        CORE.DECISION_AUDIT_SCHEMA_V2
    )
    assert authorization["prior_v2r18_historical_evidence"] == (
        TOOL._PRIOR_V2R18_PROTECTED_POLICY_FACTS
    )
    assert authorization["prior_v2r19_historical_evidence"] == (
        TOOL._PRIOR_V2R19_PROTECTED_POLICY_FACTS
    )
    forged = deepcopy(baseline)
    if mutation == "audit-policy":
        forged["decision_audit_policy"][
            "tamper_disposition"
        ] = "FORGED"
    elif mutation == "audit-policy-digest":
        forged["decision_audit_policy_digest"] = _tagged("forged-audit")
    elif mutation == "audit-core-schema":
        forged["decision_audit_core_schema_version"] = "forged/1"
    elif mutation == "prior-v2r18-history":
        forged["prior_v2r18_historical_evidence"][
            "prior_v2r18_offline_corrected_decision"
        ] = CORE.TERMINAL_NONRETRYABLE_V2
    elif mutation == "prior-v2r19-history":
        forged["prior_v2r19_historical_evidence"][
            "prior_v2r19_slot_01_resource_sample_count"
        ] = 8_046
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)
    unsigned = deepcopy(forged)
    unsigned.pop("capsule_digest")
    forged["capsule_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["capsule"],
        unsigned,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._authorization_template(contract, forged)
    assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"


def test_authorization_digest_exact_load_and_no_replace(
    tmp_path: Path,
) -> None:
    contract = _contract()
    capsule = _capsule_stub(contract=contract)
    prefix = f"q05b-{tmp_path.name}-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as raw_root:
        root = Path(raw_root) / "authorization"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        contract["fixed_external_roots"]["authorization_root"] = str(root)
        contract["authorization_policy"]["authorization_root"] = str(root)
        capsule = _capsule_stub(contract=contract)
        authorization = TOOL._authorization_template(contract, capsule)
        unsigned = deepcopy(authorization)
        unsigned.pop("authorization_digest")
        assert authorization["authorization_digest"] == TOOL._domain_digest(
            contract["hash_domains"]["authorization"],
            unsigned,
        )
        basename = contract["authorization_policy"][
            "authorization_record_name"
        ]
        assert type(basename) is str
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            payload = TOOL._publish_immutable_at(
                root_fd, basename, authorization
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                TOOL._publish_immutable_at(root_fd, basename, authorization)
        finally:
            os.close(root_fd)
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        path = root / basename
        assert path.read_bytes() == payload
        status = path.stat(follow_symlinks=False)
        assert stat.S_IMODE(status.st_mode) == 0o444
        assert status.st_nlink == 1
        assert TOOL._load_authorization(contract, capsule) == authorization


def test_authorization_binds_exact_once_campaign_and_pending_reentry_semantics() -> None:
    contract = _contract()
    authorization = TOOL._authorization_template(contract, _capsule_stub())
    assert authorization["authorized_campaign_execution_count"] == 1
    assert authorization["maximum_supervisor_invocations"] == 3
    assert (
        authorization[
            "preconsumption_pending_validation_reentry_is_campaign_execution"
        ]
        is False
    )
    assert (
        authorization[
            "preconsumption_pending_validation_reentry_is_authorization_consumption"
        ]
        is False
    )
    assert (
        authorization[
            "preconsumption_pending_validation_reentry_is_retry_transition"
        ]
        is False
    )
    assert (
        authorization[
            "preconsumption_pending_validation_reentry_is_supervisor_invocation"
        ]
        is False
    )
    assert authorization[
        "preconsumption_pending_reentry_requires_same_entrypoint"
    ] is True
    assert authorization[
        "preconsumption_pending_reentry_requires_same_capsule_digest"
    ] is True
    assert authorization[
        "preconsumption_pending_reentry_requires_fresh_complete_entry_replay"
    ] is True
    assert authorization[
        "preconsumption_pending_eligible_phase_condition_pairs_in_exact_order"
    ] == [
        ["TARGET_LOCAL_OBJECT_CLOSURE_FSCK", "READONLY_COMMAND_TIMEOUT"],
        [
            "HOST_CPU_TOPOLOGY_AFFINITY_ADMISSION",
            "SCHED_AFFINITY_API_UNAVAILABLE",
        ],
        [
            "HOST_CPU_TOPOLOGY_AFFINITY_ADMISSION",
            "REQUIRED_EFFECTIVE_CPU_SET_UNAVAILABLE",
        ],
        [
            "HOST_CPU_TOPOLOGY_AFFINITY_ADMISSION",
            "EXACT_BIND_OR_REPLAY_FAILED_WITH_EXACT_RESTORE",
        ],
    ]


def test_authorization_postpublication_result_failure_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    root = tmp_path / "owner-authorization"
    contract["fixed_external_roots"]["authorization_root"] = os.fspath(root)
    contract["authorization_policy"]["authorization_root"] = os.fspath(root)
    capsule = _capsule_stub(contract=contract)
    authorization = TOOL._authorization_template(contract, capsule)
    context = {
        "capsule": capsule,
        "authorization_template": authorization,
    }
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    real_publish = TOOL._publish_immutable_at
    real_fspath = TOOL.os.fspath
    helper_succeeded = False

    def publish_then_mark(*args: object, **kwargs: object) -> bytes:
        nonlocal helper_succeeded
        result = real_publish(*args, **kwargs)
        helper_succeeded = True
        return result

    def fail_receipt_path(value: object) -> str:
        if helper_succeeded and isinstance(value, Path) and value.parent == root:
            raise MemoryError("injected authorization receipt construction failure")
        return real_fspath(value)

    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_mark)
    monkeypatch.setattr(TOOL.os, "fspath", fail_receipt_path)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.create_phase3_q05b_shadow_retry_authorization_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "receipt construction failure" in error.value.detail
    basename = str(contract["authorization_policy"]["authorization_record_name"])
    assert (root / basename).is_file()


def test_authorization_postpublication_close_failure_is_uncertain_and_closes_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    root = tmp_path / "owner-authorization"
    contract["fixed_external_roots"]["authorization_root"] = os.fspath(root)
    contract["authorization_policy"]["authorization_root"] = os.fspath(root)
    capsule = _capsule_stub(contract=contract)
    authorization = TOOL._authorization_template(contract, capsule)
    context = {
        "capsule": capsule,
        "authorization_template": authorization,
    }
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    real_create = TOOL._create_private_root
    real_publish = TOOL._publish_immutable_at
    real_close = TOOL.os.close
    captured_fds: list[int] = []
    helper_succeeded = False
    close_failed = False
    parent_closed = False

    def create_and_capture(path: Path, label: str) -> tuple[int, int]:
        result = real_create(path, label)
        captured_fds.extend(result)
        return result

    def publish_then_mark(*args: object, **kwargs: object) -> bytes:
        nonlocal helper_succeeded
        result = real_publish(*args, **kwargs)
        helper_succeeded = True
        return result

    def fail_root_close_once(descriptor: int) -> None:
        nonlocal close_failed, parent_closed
        if (
            helper_succeeded
            and captured_fds
            and descriptor == captured_fds[1]
            and not close_failed
        ):
            close_failed = True
            raise OSError("injected authorization root close failure")
        if captured_fds and descriptor == captured_fds[0]:
            parent_closed = True
        real_close(descriptor)

    monkeypatch.setattr(TOOL, "_create_private_root", create_and_capture)
    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_mark)
    monkeypatch.setattr(TOOL.os, "close", fail_root_close_once)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL.create_phase3_q05b_shadow_retry_authorization_v2(
                capsule["capsule_digest"]
            )
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "root close failure" in error.value.detail
        assert close_failed is True
        assert parent_closed is True
    finally:
        if captured_fds and close_failed:
            real_close(captured_fds[1])


@pytest.mark.parametrize(
    "fault_stage",
    (
        "write",
        "file_fsync",
        "directory_fsync",
        "replay_open",
        "replay_read",
        "pathname_stat",
    ),
)
def test_publication_fault_after_create_is_always_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault_stage: str,
) -> None:
    root = tmp_path / fault_stage
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    real_open = TOOL.os.open
    real_write = TOOL.os.write
    real_fsync = TOOL.os.fsync
    real_read = TOOL.os.read
    real_stat = TOOL.os.stat
    fsync_count = 0
    open_count = 0

    def injected_open(*args: object, **kwargs: object) -> int:
        nonlocal open_count
        open_count += 1
        if fault_stage == "replay_open" and open_count == 2:
            raise OSError("injected replay open failure")
        return real_open(*args, **kwargs)

    def injected_write(*args: object, **kwargs: object) -> int:
        if fault_stage == "write":
            raise OSError("injected write failure")
        return real_write(*args, **kwargs)

    def injected_fsync(*args: object, **kwargs: object) -> None:
        nonlocal fsync_count
        fsync_count += 1
        if fault_stage == "file_fsync" and fsync_count == 1:
            raise OSError("injected file fsync failure")
        if fault_stage == "directory_fsync" and fsync_count == 2:
            raise OSError("injected directory fsync failure")
        real_fsync(*args, **kwargs)

    def injected_read(*args: object, **kwargs: object) -> bytes:
        if fault_stage == "replay_read":
            raise OSError("injected replay read failure")
        return real_read(*args, **kwargs)

    def injected_stat(*args: object, **kwargs: object) -> os.stat_result:
        if fault_stage == "pathname_stat":
            raise OSError("injected pathname stat failure")
        return real_stat(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(TOOL.os, "open", injected_open)
            patch.setattr(TOOL.os, "write", injected_write)
            patch.setattr(TOOL.os, "fsync", injected_fsync)
            patch.setattr(TOOL.os, "read", injected_read)
            patch.setattr(TOOL.os, "stat", injected_stat)
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                TOOL._publish_immutable_at(root_fd, "record.json", b"x\n")
    finally:
        os.close(root_fd)
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert (root / "record.json").exists()


@pytest.mark.parametrize(
    ("basename", "publication_label"),
    (
        ("supervisor.stdout", "slot-supervisor-stdout"),
        ("supervisor.stderr", "slot-supervisor-stderr"),
    ),
)
def test_immutable_publication_diagnostic_binds_exact_stream_step(
    tmp_path: Path,
    basename: str,
    publication_label: str,
) -> None:
    root = tmp_path / publication_label
    root.mkdir(mode=0o700)
    (root / basename).write_bytes(b"existing")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_immutable_at(
                root_fd,
                basename,
                b"new",
                publication_label=publication_label,
                known_failure_code="FAIL_CAMPAIGN_EVIDENCE",
            )
    finally:
        os.close(root_fd)
    assert error.value.code == "FAIL_CAMPAIGN_EVIDENCE"
    assert error.value.receipt == {
        "schema_version": TOOL._PUBLICATION_RECEIPT_SCHEMA_V2,
        "publication_outcome": "KNOWN_NO_PUBLICATION",
        "publication_label": publication_label,
        "publication_basename": basename,
    }
    assert f"publication_label={publication_label}" in error.value.detail
    assert f"basename={basename}" in error.value.detail


def test_external_artifact_posthelper_replay_failure_is_publication_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "supervisor-artifact.json"
    raw = b'{"status":"valid"}\n'
    source.write_bytes(raw)
    source.chmod(0o444)
    source_status = source.stat()
    target_parent = tmp_path / "external-artifact"
    target = target_parent / "artifact.json"
    slot = {
        "supervisor_artifact_path": os.fspath(source),
        "artifact_parent_root": os.fspath(target_parent),
        "artifact_path": os.fspath(target),
    }
    evidence = {
        "state": "VALID",
        "replay_passed": True,
        "raw_sha256_or_null": sha256(raw).hexdigest(),
        "file_identity_or_null": {
            "dev": source_status.st_dev,
            "ino": source_status.st_ino,
            "mode": stat.S_IMODE(source_status.st_mode),
            "nlink": source_status.st_nlink,
            "uid": source_status.st_uid,
            "size": source_status.st_size,
        },
    }
    receipt = {"argv": ["post-helper-read"], "return_code": 7}

    def fail_posthelper_read(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[bytes, os.stat_result]:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_FILESYSTEM",
            "injected external post-helper replay failure",
            terminal=True,
            receipt=receipt,
        )

    monkeypatch.setattr(TOOL, "_read_regular_at_nofollow", fail_posthelper_read)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._publish_external_artifact(slot, evidence)
    assert target.is_file()
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "injected external post-helper replay failure" in error.value.detail
    assert type(error.value.receipt) is dict
    assert error.value.receipt["origin_error_or_null"][
        "error_receipt_or_null"
    ] == receipt


def test_external_artifact_double_fault_preserves_primary_publication_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "supervisor-artifact.json"
    raw = b'{"status":"valid"}\n'
    source.write_bytes(raw)
    source.chmod(0o444)
    source_status = source.stat()
    target_parent = tmp_path / "external-artifact"
    target = target_parent / "artifact.json"
    slot = {
        "supervisor_artifact_path": os.fspath(source),
        "artifact_parent_root": os.fspath(target_parent),
        "artifact_path": os.fspath(target),
    }
    evidence = {
        "state": "VALID",
        "replay_passed": True,
        "raw_sha256_or_null": sha256(raw).hexdigest(),
        "file_identity_or_null": {
            "dev": source_status.st_dev,
            "ino": source_status.st_ino,
            "mode": stat.S_IMODE(source_status.st_mode),
            "nlink": source_status.st_nlink,
            "uid": source_status.st_uid,
            "size": source_status.st_size,
        },
    }
    primary_receipt = {"primary": "post-helper replay"}
    real_create = TOOL._create_private_root
    real_close = TOOL.os.close
    root_fd = -1
    injected_close = False

    def capture_root(path: Path, label: str) -> tuple[int, int]:
        nonlocal root_fd
        pair = real_create(path, label)
        root_fd = pair[1]
        return pair

    def fail_primary(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[bytes, os.stat_result]:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_FILESYSTEM",
            "injected primary external replay failure",
            terminal=True,
            receipt=primary_receipt,
        )

    def fail_root_close(descriptor: int) -> None:
        nonlocal injected_close
        if descriptor == root_fd and target.exists() and not injected_close:
            injected_close = True
            raise OSError("injected secondary external root close failure")
        real_close(descriptor)

    monkeypatch.setattr(TOOL, "_create_private_root", capture_root)
    monkeypatch.setattr(TOOL, "_read_regular_at_nofollow", fail_primary)
    monkeypatch.setattr(TOOL.os, "close", fail_root_close)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_external_artifact(slot, evidence)
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "primary external replay failure" in error.value.detail
        assert "secondary external root close" not in error.value.detail
        assert type(error.value.receipt) is dict
        assert error.value.receipt["origin_error_or_null"][
            "error_receipt_or_null"
        ] == primary_receipt
        assert injected_close is True
    finally:
        if injected_close and root_fd >= 0:
            real_close(root_fd)


def test_ledger_posthelper_root_failure_is_publication_uncertain_without_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    contract = _contract()
    ledger = object.__new__(TOOL._LedgerV2)
    ledger.contract = contract
    ledger.capsule = _capsule_stub()
    ledger.authorization = {"authorization_digest": _tagged("authorization")}
    ledger.root_path = root
    ledger.root_fd = root_fd
    ledger.root_identity = TOOL._verify_held_directory_path(
        root,
        root_fd,
        "ledger post-helper KAT baseline",
    )
    ledger.previous_record_digest = None
    ledger.sequence = 0
    ledger.published_names = []
    ledger.published_records = []
    ledger.completion_commit_verified = False
    ledger.allowed_names = ["campaign.intent.json"]
    ledger.last_name_index = -1
    original_verify = TOOL._verify_held_directory_path
    verify_count = 0

    def fail_second_verify(
        path: Path,
        descriptor: int,
        label: str,
    ) -> dict[str, object]:
        nonlocal verify_count
        verify_count += 1
        if verify_count == 2:
            raise TOOL.ShadowRetryCampaignErrorV2(
                "FAIL_CAMPAIGN_FILESYSTEM",
                "injected ledger post-helper root failure",
                terminal=True,
                receipt={"root_replay": "failed"},
            )
        return original_verify(path, descriptor, label)

    def publish_then_commit(
        *_args: object,
        **kwargs: object,
    ) -> bytes:
        callback = kwargs["commit_callback"]
        assert callable(callback)
        callback()
        return b"publication helper reported success"

    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_commit)
    monkeypatch.setattr(TOOL, "_verify_held_directory_path", fail_second_verify)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            ledger.publish(
                "campaign.intent.json",
                "CAMPAIGN_INTENT",
                "INTENT",
                {"purpose": "post-helper fault KAT"},
            )
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "injected ledger post-helper root failure" in error.value.detail
        assert type(error.value.receipt) is dict
        assert error.value.receipt["origin_error_or_null"][
            "error_receipt_or_null"
        ] == {"root_replay": "failed"}
        assert ledger.sequence == 0
        assert ledger.published_names == []
    finally:
        os.close(root_fd)


def test_ledger_next_state_is_prebuilt_before_immutable_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ExplodingRecordList(list[dict[str, object]]):
        def __iter__(self) -> object:
            raise MemoryError("injected ledger next-state prebuild failure")

    root = tmp_path / "ledger"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    contract = _contract()
    ledger = object.__new__(TOOL._LedgerV2)
    ledger.contract = contract
    ledger.capsule = _capsule_stub()
    ledger.authorization = {"authorization_digest": _tagged("authorization")}
    ledger.root_path = root
    ledger.root_fd = root_fd
    ledger.root_identity = TOOL._verify_held_directory_path(
        root,
        root_fd,
        "ledger state-commit KAT baseline",
    )
    ledger.previous_record_digest = None
    ledger.sequence = 0
    ledger.published_names = []
    ledger.published_records = ExplodingRecordList()
    ledger.completion_commit_verified = False
    ledger.allowed_names = ["campaign.intent.json"]
    ledger.last_name_index = -1
    helper_calls = 0

    def mark_publication(*_args: object, **_kwargs: object) -> bytes:
        nonlocal helper_calls
        helper_calls += 1
        return b"publication helper reported success"

    monkeypatch.setattr(
        TOOL,
        "_publish_immutable_at",
        mark_publication,
    )
    try:
        with pytest.raises(MemoryError) as error:
            ledger.publish(
                "campaign.intent.json",
                "CAMPAIGN_INTENT",
                "INTENT",
                {"purpose": "state-commit fault KAT"},
            )
        assert "next-state prebuild failure" in str(error.value)
        assert helper_calls == 0
        assert ledger.last_name_index == -1
        assert ledger.published_names == []
    finally:
        os.close(root_fd)


def test_ledger_close_failure_after_publication_is_uncertain_and_closes_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger-root"
    )
    ledger = TOOL._LedgerV2(
        contract,
        _capsule_stub(),
        {"authorization_digest": _tagged("authorization")},
    )
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {"test": True},
    )
    root_fd = ledger.root_fd
    parent_fd = ledger.parent_fd
    real_close = TOOL.os.close
    root_failed = False
    parent_closed = False

    def fail_root_close_once(descriptor: int) -> None:
        nonlocal root_failed, parent_closed
        if descriptor == root_fd and not root_failed:
            root_failed = True
            raise OSError("injected ledger root close failure")
        if descriptor == parent_fd:
            parent_closed = True
        real_close(descriptor)

    monkeypatch.setattr(TOOL.os, "close", fail_root_close_once)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            ledger.close()
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "ledger root close failure" in error.value.detail
        assert root_failed is True
        assert parent_closed is True
        assert ledger.root_fd is None
        assert ledger.parent_fd is None
    finally:
        if root_failed:
            real_close(root_fd)


def test_ledger_close_double_fault_preserves_spawn_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger-root"
    )
    ledger = TOOL._LedgerV2(
        contract,
        _capsule_stub(),
        {"authorization_digest": _tagged("authorization")},
    )
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {"test": True},
    )
    root_fd = ledger.root_fd
    real_close = TOOL.os.close
    root_failed = False

    def fail_root_close_once(descriptor: int) -> None:
        nonlocal root_failed
        if descriptor == root_fd and not root_failed:
            root_failed = True
            raise OSError("injected ledger close double fault")
        real_close(descriptor)

    monkeypatch.setattr(TOOL.os, "close", fail_root_close_once)
    original = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
        "injected spawn identity uncertainty",
        terminal=True,
    )
    try:
        try:
            raise original
        except TOOL.ShadowRetryCampaignErrorV2:
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                ledger.close()
        assert error.value is original
        assert root_failed is True
    finally:
        if root_failed:
            real_close(root_fd)


def test_known_evidence_precreate_failure_is_not_ledger_uncertainty(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    (root / "stdout").write_bytes(b"preexisting")
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_immutable_at(
                descriptor,
                "stdout",
                b"new",
                known_failure_code="FAIL_CAMPAIGN_EVIDENCE",
            )
    finally:
        os.close(descriptor)
    assert error.value.code == "FAIL_CAMPAIGN_EVIDENCE"
    assert (root / "stdout").read_bytes() == b"preexisting"


def test_o_creat_return_gap_is_publication_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "publication-root"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    real_open = TOOL.os.open
    real_close = TOOL.os.close

    def create_then_interrupt(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "record.json" and flags & os.O_CREAT:
            real_close(descriptor)
            raise KeyboardInterrupt("injected after kernel O_CREAT")
        return descriptor

    monkeypatch.setattr(TOOL.os, "open", create_then_interrupt)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_immutable_at(
                root_fd,
                "record.json",
                {"value": 1},
            )
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert (root / "record.json").exists()
    finally:
        real_close(root_fd)


def test_prepared_ledger_mapping_is_frozen_before_check_and_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "frozen-ledger"
    )
    ledger = TOOL._LedgerV2(
        contract,
        _capsule_stub(),
        {"authorization_digest": _tagged("authorization")},
    )
    prepared = ledger.prepare(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {"nested": {"value": "checked"}},
    )
    frozen = deepcopy(prepared)
    real_publish = TOOL._publish_immutable_at

    def mutate_original_then_publish(
        directory_fd: int,
        basename: str,
        value_or_bytes: Mapping[str, object] | bytes,
        **kwargs: object,
    ) -> bytes:
        assert type(value_or_bytes) is bytes
        prepared["body"]["nested"]["value"] = "mutated-after-freeze"
        return real_publish(
            directory_fd,
            basename,
            value_or_bytes,
            **kwargs,
        )

    monkeypatch.setattr(
        TOOL,
        "_publish_immutable_at",
        mutate_original_then_publish,
    )
    try:
        published = ledger.publish_prepared("campaign.intent.json", prepared)
        assert published == frozen
        raw = Path(
            str(contract["fixed_external_roots"]["ledger_root"])
        ).joinpath("campaign.intent.json").read_bytes()
        assert TOOL._strict_json(raw, "frozen ledger KAT") == frozen
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "fault_stage",
    (
        "external_artifact_return",
        "postbinding_freshness",
        "postoutcome_evidence_close",
        "spawn_uncertain_evidence_close",
    ),
)
def test_slot_postbinding_failures_never_fall_through_to_a_second_terminal_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault_stage: str,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    slot.update(
        {
            "clone_root": os.fspath(tmp_path / "clone"),
            "clone_project_root": os.fspath(tmp_path / "clone" / "Hegel Machine"),
            "cargo_cache_root": os.fspath(tmp_path / "cargo"),
            "trace_root": os.fspath(tmp_path / "trace"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "stdout_path": os.fspath(tmp_path / "evidence" / "stdout.bin"),
            "stderr_path": os.fspath(tmp_path / "evidence" / "stderr.bin"),
            "supervisor_artifact_path": os.fspath(
                tmp_path / "clone" / "Hegel Machine" / "artifact.json"
            ),
            "artifact_parent_root": os.fspath(tmp_path / "artifact-parent"),
            "artifact_path": os.fspath(tmp_path / "artifact-parent" / "artifact.json"),
            "scheduling_probe_root": os.fspath(
                tmp_path / "scheduling-probe"
            ),
            "external_trajectory_binding_path": os.fspath(
                tmp_path / "evidence" / "trajectory-binding.json"
            ),
            "slot_result_path": os.fspath(tmp_path / "evidence" / "outcome.json"),
        }
    )
    capsule = _capsule_stub()
    capsule["freshness_parent_identities"] = {
        os.fspath(tmp_path): _real_directory_stable_identity(tmp_path)
    }
    context = {
        "slots": [slot],
        "cargo_rows": [],
        "cargo_identity": {"content_manifest_digest": _tagged("cargo")},
        "freshness_parent_identities": deepcopy(
            capsule["freshness_parent_identities"]
        ),
    }
    ledger = _FakeLedger()
    invocation_counter = [0]
    freshness = _future_freshness()
    source_before = _source_replay("source-before")
    source_before["clone_closure_rows_internal"] = []
    prepared_clone = {
        "artifact_state": "ABSENT",
        "clone_closure_rows_internal": [],
    }
    prepared_clone_reuse_count = 0
    postrun_clone_replay_count = 0
    source_after = _source_replay("source-after")
    supervisor, external = _absent_artifacts()
    trajectory = _trajectory(
        slot,
        outcome=("OK" if fault_stage == "external_artifact_return" else "FAIL"),
        terminal_error_detail=(
            None
            if fault_stage == "external_artifact_return"
            else "injected supervisor failure"
        ),
    )
    binding_published = False
    outcome_published = False
    external_outcome_value: dict[str, object] | None = None
    evidence_fd: int | None = None
    close_failed = False
    real_create_evidence = TOOL._create_evidence_root
    real_close = TOOL.os.close

    class FakeCore:
        DECISION_AUDIT_SCHEMA_V2 = CORE.DECISION_AUDIT_SCHEMA_V2
        DECISION_AUDIT_CLAUSES_V2 = CORE.DECISION_AUDIT_CLAUSES_V2
        SUCCESS_V2 = CORE.SUCCESS_V2
        RETRY_EXACT_GAP_V2 = CORE.RETRY_EXACT_GAP_V2
        EXHAUSTED_RETRYABLE_GAP_V2 = CORE.EXHAUSTED_RETRYABLE_GAP_V2
        TERMINAL_NONRETRYABLE_V2 = CORE.TERMINAL_NONRETRYABLE_V2

        @staticmethod
        def classify_supervisor_stdout_v2(_raw: bytes) -> dict[str, object]:
            return {}

        @staticmethod
        def evaluate_slot_observation_v2(
            _observation: Mapping[str, object],
        ) -> str:
            return CORE.TERMINAL_NONRETRYABLE_V2

        @staticmethod
        def evaluate_slot_observation_with_audit_v2(
            _observation: Mapping[str, object],
        ) -> dict[str, object]:
            return {
                "schema_version": CORE.DECISION_AUDIT_SCHEMA_V2,
                "decision": CORE.TERMINAL_NONRETRYABLE_V2,
                "matched_path_or_null": None,
                "first_reject_clause_or_null": "OBSERVATION_SHAPE",
                "predicate_results": [
                    {
                        "clause": clause,
                        "result": (
                            "REJECT"
                            if clause == "OBSERVATION_SHAPE"
                            else "NOT_EVALUATED"
                        ),
                    }
                    for clause in CORE.DECISION_AUDIT_CLAUSES_V2
                ],
            }

    def freshness_replay(*_args: object, **_kwargs: object) -> dict[str, object]:
        if (
            fault_stage == "postbinding_freshness"
            and binding_published
            and not outcome_published
        ):
            raise TOOL.ShadowRetryCampaignErrorV2(
                "FAIL_CAMPAIGN_FRESHNESS",
                "injected postbinding freshness failure",
                terminal=True,
            )
        return deepcopy(freshness)

    def bounded_process(*_args: object, **kwargs: object) -> dict[str, object]:
        callback = kwargs.get("before_popen")
        assert callable(callback)
        callback()
        if fault_stage == "spawn_uncertain_evidence_close":
            raise TOOL.ShadowRetryCampaignErrorV2(
                "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
                "injected spawn uncertainty",
                terminal=True,
                receipt={"cleanup_certain": False},
            )
        return _process(
            b"{}\n",
            0 if fault_stage == "external_artifact_return" else 1,
        )

    def create_evidence(
        slot_value: Mapping[str, object],
    ) -> tuple[int, int]:
        nonlocal evidence_fd
        result = real_create_evidence(slot_value)
        evidence_fd = result[1]
        return result

    def publish_evidence(
        _directory_fd: int,
        basename: str,
        value: object,
        **_kwargs: object,
    ) -> bytes:
        nonlocal binding_published, outcome_published, external_outcome_value
        if basename == "trajectory-binding.json":
            binding_published = True
        elif basename == "outcome.json":
            outcome_published = True
            assert type(value) is dict
            external_outcome_value = deepcopy(value)
        return b"published"

    def close_with_injected_outcome_fault(descriptor: int) -> None:
        nonlocal close_failed
        if (
            fault_stage
            in {
                "postoutcome_evidence_close",
                "spawn_uncertain_evidence_close",
            }
            and (
                outcome_published
                or fault_stage == "spawn_uncertain_evidence_close"
            )
            and evidence_fd is not None
            and descriptor == evidence_fd
            and not close_failed
        ):
            close_failed = True
            raise OSError("injected postoutcome evidence close failure")
        real_close(descriptor)

    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: FakeCore)
    monkeypatch.setattr(TOOL, "_slot_argv", lambda *_args: ["/usr/bin/true"])
    monkeypatch.setattr(TOOL, "_runtime_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(TOOL, "_slot_roots_absence_evidence", freshness_replay)
    monkeypatch.setattr(
        TOOL,
        "_prepare_clone",
        lambda *_args: prepared_clone,
    )
    monkeypatch.setattr(
        TOOL,
        "_copy_tree_exact",
        lambda *_args, **_kwargs: {
            "content_manifest_digest": context["cargo_identity"][
                "content_manifest_digest"
            ]
        },
    )
    def source_state_before(
        *_args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal prepared_clone_reuse_count
        assert kwargs["verified_clone"] is prepared_clone
        prepared_clone_reuse_count += 1
        return deepcopy(source_before)

    monkeypatch.setattr(TOOL, "_source_state_replay", source_state_before)
    monkeypatch.setattr(TOOL, "_safe_docker_inventory", lambda _contract: (True, [], None))
    monkeypatch.setattr(TOOL, "_create_evidence_root", create_evidence)
    monkeypatch.setattr(
        TOOL,
        "_assert_existing_directory_roots_nonaliased",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        TOOL,
        "_augment_source_replay_with_alias_rows",
        lambda value, _rows: deepcopy(dict(value)),
    )
    probe_cleanup = _scheduling_probe_cleanup_receipt(capsule, slot)
    monkeypatch.setattr(
        TOOL,
        "_run_slot_scheduling_admission_v2r17",
        lambda *_args, **_kwargs: (
            _admitted_scheduling_evidence(contract),
            deepcopy(probe_cleanup),
            {},
        ),
    )
    monkeypatch.setattr(TOOL, "_bounded_process", bounded_process)
    monkeypatch.setattr(TOOL, "_publish_raw_streams", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "_postpublication_directory_identity",
        lambda _path, _descriptor, _label, expected: deepcopy(dict(expected)),
    )
    def source_state_after(
        *_args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal postrun_clone_replay_count
        assert "verified_clone" not in kwargs
        postrun_clone_replay_count += 1
        return deepcopy(source_after)

    monkeypatch.setattr(TOOL, "_safe_source_state_replay", source_state_after)
    monkeypatch.setattr(
        TOOL,
        "_enforce_clone_closure",
        lambda _slot, _before, after: deepcopy(dict(after)),
    )
    monkeypatch.setattr(
        TOOL,
        "_safe_trajectory_evidence",
        lambda *_args, **_kwargs: deepcopy(trajectory),
    )
    monkeypatch.setattr(TOOL, "_artifact_replay", lambda *_args: deepcopy(supervisor))
    monkeypatch.setattr(
        TOOL,
        "_trajectory_complete_and_bound",
        lambda *_args: fault_stage == "external_artifact_return",
    )
    monkeypatch.setattr(
        TOOL,
        "_artifact_matches_success_wire",
        lambda *_args: fault_stage == "external_artifact_return",
    )

    def external_artifact_then_return_fault(
        *_args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        callback = kwargs["commit_callback"]
        assert callable(callback)
        callback({"state": "VALID", "replay_passed": True})
        raise KeyboardInterrupt("injected external artifact return gap")

    if fault_stage == "external_artifact_return":
        monkeypatch.setattr(
            TOOL,
            "_publish_external_artifact",
            external_artifact_then_return_fault,
        )
    monkeypatch.setattr(TOOL, "_absent_external_artifact", lambda *_args: deepcopy(external))
    monkeypatch.setattr(TOOL, "_cleanup_replay_evidence", lambda *_args: _cleanup_replay())
    monkeypatch.setattr(
        TOOL,
        "_binding_record",
        lambda *_args: {"binding_digest": _tagged("binding")},
    )
    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_evidence)
    monkeypatch.setattr(TOOL, "_build_slot_observation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(TOOL.os, "close", close_with_injected_outcome_fault)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._run_one_slot(
                contract,
                context,
                capsule,
                ledger,  # type: ignore[arg-type]
                slot,
                invocation_counter,
                {
                    "supervisor_invocation_count": 0,
                    "ledger_write_count": 0,
                },
            )
        assert prepared_clone_reuse_count == 1
        if fault_stage == "spawn_uncertain_evidence_close":
            assert postrun_clone_replay_count == 0
        else:
            assert postrun_clone_replay_count == 1
        if fault_stage == "spawn_uncertain_evidence_close":
            assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
        else:
            assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        if fault_stage == "external_artifact_return":
            assert binding_published is False
            assert outcome_published is False
            assert "external artifact return gap" in error.value.detail
        elif fault_stage == "postbinding_freshness":
            assert binding_published is True
            assert outcome_published is False
            assert "postbinding freshness failure" in error.value.detail
        elif fault_stage == "postoutcome_evidence_close":
            assert binding_published is True
            assert outcome_published is True
            assert type(external_outcome_value) is dict
            ledger_outcome = next(
                body
                for basename, record_type, _status, body in ledger.rows
                if basename == "slot-01.outcome.json"
                and record_type == "SLOT_OUTCOME"
            )
            assert ledger_outcome == external_outcome_value
            assert external_outcome_value["decision"] == (
                CORE.TERMINAL_NONRETRYABLE_V2
            )
            assert external_outcome_value["decision_audit"] == (
                FakeCore.evaluate_slot_observation_with_audit_v2({})
            )
            assert close_failed is True
            assert "postoutcome evidence close failure" in error.value.detail
        elif fault_stage == "spawn_uncertain_evidence_close":
            assert binding_published is False
            assert outcome_published is False
            assert close_failed is True
            assert "injected spawn uncertainty" in error.value.detail
    finally:
        if close_failed and evidence_fd is not None:
            real_close(evidence_fd)


def test_consumption_callback_failure_never_spawns_and_preserves_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    popen_calls = 0

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(TOOL.subprocess, "Popen", forbidden_popen)

    def typed_callback() -> None:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_LEDGER",
            "injected known no-publication failure",
            terminal=True,
        )

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as typed_error:
        TOOL._bounded_process(
            ["/usr/bin/true"],
            cwd=os.fspath(tmp_path),
            env={},
            timeout_seconds=1,
            stdout_max_bytes=1,
            stderr_max_bytes=1,
            before_popen=typed_callback,
        )
    assert typed_error.value.code == "FAIL_CAMPAIGN_LEDGER"

    def untyped_callback() -> None:
        raise OSError("injected ambiguous callback failure")

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as untyped_error:
        TOOL._bounded_process(
            ["/usr/bin/true"],
            cwd=os.fspath(tmp_path),
            env={},
            timeout_seconds=1,
            stdout_max_bytes=1,
            stderr_max_bytes=1,
            before_popen=untyped_callback,
        )
    assert (
        untyped_error.value.code
        == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    )
    assert popen_calls == 0


def test_registered_spawn_restores_child_signal_mask_and_leaks_no_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    program = (
        "import json,os,signal;"
        "m=sorted(int(x) for x in signal.pthread_sigmask(signal.SIG_BLOCK,set()));"
        "f={};"
        "[(f.__setitem__(n,os.readlink('/proc/self/fd/'+n)) "
        "if os.path.exists('/proc/self/fd/'+n) else None) "
        "for n in os.listdir('/proc/self/fd')];"
        "print(json.dumps({'mask':m,'fds':f,'pid':os.getpid(),"
        "'pgrp':os.getpgrp(),'sid':os.getsid(0)},"
        "sort_keys=True,separators=(',',':')))"
    )
    result = TOOL._bounded_process(
        ["/usr/bin/python3.10", "-I", "-S", "-B", "-c", program],
        cwd=os.fspath(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    decoded = TOOL._strict_json(
        result["stdout"], "registered spawn child state", 4096
    )
    assert decoded["mask"] == []
    assert set(decoded["fds"]) <= {"0", "1", "2"}
    assert decoded["pid"] == decoded["pgrp"] == decoded["sid"]
    assert result["return_code"] == 0
    assert result["process_group_absent"] is True


def test_spawn_uses_source_bound_launcher_without_preexec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[list[str], dict[str, object]]] = []
    masks_at_popen: list[set[object]] = []
    real_init = TOOL.subprocess.Popen.__init__
    parent_mask_before = TOOL.signal.pthread_sigmask(TOOL.signal.SIG_BLOCK, set())

    def inspect_init(process: object, args: object, **kwargs: object) -> None:
        assert type(args) is list
        observed.append((list(args), dict(kwargs)))
        masks_at_popen.append(
            TOOL.signal.pthread_sigmask(TOOL.signal.SIG_BLOCK, set())
        )
        real_init(process, args, **kwargs)

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", inspect_init)
    result = TOOL._bounded_process(
        ["/usr/bin/true"],
        cwd=os.fspath(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert result["return_code"] == 0
    assert len(observed) == 1
    launcher_argv, kwargs = observed[0]
    assert launcher_argv[:7] == [
        TOOL._SPAWN_LAUNCHER_INTERPRETER_V2,
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        "-c",
    ]
    assert launcher_argv[7] == TOOL._SPAWN_LAUNCHER_SOURCE_V2
    assert launcher_argv[-1] == "/usr/bin/true"
    assert "preexec_fn" not in kwargs
    assert TOOL.signal.SIGINT in masks_at_popen[0]
    assert (
        TOOL.signal.pthread_sigmask(TOOL.signal.SIG_BLOCK, set())
        == parent_mask_before
    )


def test_sigchld_ignore_is_rejected_before_consumption_or_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback_calls = 0
    popen_calls = 0
    original = TOOL.signal.getsignal(TOOL.signal.SIGCHLD)

    def callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    def forbidden_init(*_args: object, **_kwargs: object) -> None:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", forbidden_init)
    TOOL.signal.signal(TOOL.signal.SIGCHLD, TOOL.signal.SIG_IGN)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                ["/usr/bin/true"],
                cwd=os.fspath(tmp_path),
                env={},
                timeout_seconds=1,
                stdout_max_bytes=1,
                stderr_max_bytes=1,
                before_popen=callback,
            )
    finally:
        TOOL.signal.signal(TOOL.signal.SIGCHLD, original)
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "SIGCHLD=SIG_DFL" in error.value.detail
    assert callback_calls == 0
    assert popen_calls == 0


def test_thread_created_by_consumption_callback_is_rejected_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = threading.Event()
    started = threading.Event()
    worker: list[threading.Thread] = []
    gate_calls = 0
    popen_calls = 0

    def gated_proof() -> None:
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            _REAL_REQUIRE_SINGLE_THREADED_SPAWN()

    def callback() -> None:
        thread = threading.Thread(
            target=lambda: (started.set(), release.wait(5)),
        )
        worker.append(thread)
        thread.start()
        assert started.wait(1)

    def forbidden_init(*_args: object, **_kwargs: object) -> None:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", gated_proof
    )
    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", forbidden_init)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                ["/usr/bin/true"],
                cwd=os.fspath(tmp_path),
                env={},
                timeout_seconds=1,
                stdout_max_bytes=1,
                stderr_max_bytes=1,
                before_popen=callback,
            )
    finally:
        release.set()
        for thread in worker:
            thread.join(timeout=2)
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "not exactly single-threaded" in error.value.detail
    assert gate_calls == 2
    assert popen_calls == 0


def test_launcher_preserves_actual_exit_and_maps_actual_exec_failure(
    tmp_path: Path,
) -> None:
    exited = TOOL._bounded_process(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            "-c",
            "import sys;sys.exit(37)",
        ],
        cwd=os.fspath(tmp_path),
        env={},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert exited["return_code"] == 37
    assert exited["child_reaped"] is True
    missing = TOOL._bounded_process(
        ["/definitely/missing/q05b-actual-executable"],
        cwd=os.fspath(tmp_path),
        env={},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert missing["return_code"] == 127
    assert missing["stderr"] == b"HEGEL_Q05B_LAUNCHER_ACTUAL_EXEC_FAILED\n"
    assert missing["child_reaped"] is True


def test_missing_initial_launcher_never_signals_or_probes_reaped_pid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kill_calls: list[tuple[object, object]] = []
    probe_calls: list[object] = []

    def forbidden_killpg(pid: object, value: object) -> None:
        kill_calls.append((pid, value))
        raise AssertionError("already-reaped PID must not be signalled")

    def forbidden_probe(pid: object) -> bool:
        probe_calls.append(pid)
        raise AssertionError("already-reaped PID must not be probed")

    monkeypatch.setattr(
        TOOL,
        "_SPAWN_LAUNCHER_INTERPRETER_V2",
        "/definitely/missing/q05b-launcher",
    )
    monkeypatch.setattr(TOOL.os, "killpg", forbidden_killpg)
    monkeypatch.setattr(TOOL, "_process_group_present", forbidden_probe)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            ["/usr/bin/true"],
            cwd=os.fspath(tmp_path),
            env={},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
    assert error.value.receipt["registration_state"] == "MISSING"
    assert error.value.receipt["cleanup_kill_attempted"] is False
    assert error.value.receipt["cleanup_direct_child_reaped"] is True
    assert error.value.receipt["cleanup_process_group_absent"] is False
    assert error.value.receipt["later_ledger_write_permitted"] is False
    assert kill_calls == []
    assert probe_calls == []


def test_spawn_controller_rejects_unproven_or_multiple_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.os, "listdir", lambda _path: ["11", "12"])
    monkeypatch.setattr(TOOL.os, "getpid", lambda: 11)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        _REAL_REQUIRE_SINGLE_THREADED_SPAWN()
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "not exactly single-threaded" in error.value.detail


@pytest.mark.parametrize(
    "store_name",
    ("pid", "_child_created"),
)
def test_registered_spawn_recovers_popen_internal_store_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store_name: str,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    code = TOOL.subprocess.Popen._execute_child.__code__
    target_offset = next(
        instruction.offset
        for instruction in dis.get_instructions(code)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == store_name
    )
    injected = False

    def trace(frame: object, event: str, _arg: object) -> object:
        nonlocal injected
        if getattr(frame, "f_code") is code:
            frame.f_trace_opcodes = True
            if event == "opcode" and frame.f_lasti == target_offset:
                injected = True
                sys.settrace(None)
                raise KeyboardInterrupt(f"injected before {store_name} store")
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                [
                    "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                    "import time;time.sleep(60)",
                ],
                cwd=os.fspath(tmp_path),
                env={"PATH": "/usr/bin:/bin"},
                timeout_seconds=5,
                stdout_max_bytes=4096,
                stderr_max_bytes=4096,
                reap_seconds=2,
            )
    finally:
        sys.settrace(None)
    assert injected is True
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    receipt = error.value.receipt
    assert type(receipt) is dict
    assert receipt["registration_state"] == "VALID"
    assert receipt["cleanup_direct_child_reaped"] is True
    assert receipt["cleanup_process_group_absent"] is True
    assert receipt["cleanup_certain"] is True


def test_registered_spawn_recovers_popen_return_to_caller_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    real_init = TOOL.subprocess.Popen.__init__

    def init_then_interrupt(process: object, *args: object, **kwargs: object) -> None:
        real_init(process, *args, **kwargs)
        raise KeyboardInterrupt("injected after Popen init return")

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", init_then_interrupt)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                "import time;time.sleep(60)",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
            reap_seconds=2,
        )
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert error.value.receipt["pid_match_or_null"] is True
    assert error.value.receipt["cleanup_certain"] is True


@pytest.mark.parametrize(
    ("raw", "expected_state"),
    (
        (b"", "MISSING"),
        (b"123", "TORN"),
        (b"123\nextra", "MALFORMED"),
        (b"2147483647\n", "VALID"),
    ),
)
def test_invalid_or_mismatched_spawn_receipt_fails_uncertain_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: bytes,
    expected_state: str,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    monkeypatch.setattr(
        TOOL,
        "_read_spawn_registration_receipt_v2",
        lambda _fd: (raw, True),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                "import time;time.sleep(60)",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
            reap_seconds=2,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
    assert error.value.receipt["registration_state"] == expected_state
    assert error.value.receipt["later_ledger_write_permitted"] is False


def test_spawn_cleanup_double_fault_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    spawned: list[object] = []
    real_init = TOOL.subprocess.Popen.__init__

    def remember_then_interrupt(process: object, *args: object, **kwargs: object) -> None:
        real_init(process, *args, **kwargs)
        spawned.append(process)
        raise KeyboardInterrupt("injected post-spawn fault")

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", remember_then_interrupt)
    monkeypatch.setattr(
        TOOL,
        "_cleanup_registered_spawn_v2",
        lambda *_args, **_kwargs: {
            "cleanup_target_pid_or_null": getattr(spawned[0], "pid", None),
            "cleanup_kill_attempted": True,
            "cleanup_direct_child_reaped": False,
            "cleanup_process_group_absent": False,
            "cleanup_identity_consistent": True,
            "cleanup_errors": ["injected cleanup double fault"],
            "cleanup_certain": False,
        },
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                [
                    "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                    "import time;time.sleep(60)",
                ],
                cwd=os.fspath(tmp_path),
                env={"PATH": "/usr/bin:/bin"},
                timeout_seconds=5,
                stdout_max_bytes=4096,
                stderr_max_bytes=4096,
                reap_seconds=2,
            )
        assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
        assert error.value.receipt["cleanup_certain"] is False
        assert error.value.receipt["later_ledger_write_permitted"] is False
    finally:
        if spawned:
            process = spawned[0]
            try:
                TOOL.os.killpg(process.pid, TOOL.signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)


class _FakeLedger:
    def __init__(self) -> None:
        self.authorization = {"authorization_digest": _tagged("authorization")}
        self.published_names: list[str] = []
        self.published_records: list[dict[str, object]] = []
        self.closed = False
        self.rows: list[tuple[str, str, str, dict[str, object]]] = []
        self.completion_commit_verified = False
        self.completion_result_candidate_or_null: dict[str, object] | None = None
        self.authoritative_completion_result_or_null: dict[str, object] | None = None

    def prepare(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "sequence": len(self.rows) + 1,
            "record_type": record_type,
            "status": status_text,
            "body": deepcopy(dict(body)),
            "record_digest": _tagged(f"ledger-{len(self.rows) + 1}"),
        }

    def publish_prepared(
        self,
        basename: str,
        prepared_record: Mapping[str, object] | bytes,
    ) -> dict[str, object]:
        if type(prepared_record) is bytes:
            decoded = TOOL._strict_json(
                prepared_record,
                "fake prepared ledger record",
            )
            assert type(decoded) is dict
            prepared_record = decoded
        record_type = prepared_record["record_type"]
        status_text = prepared_record["status"]
        body = prepared_record["body"]
        assert type(record_type) is str
        assert type(status_text) is str
        assert type(body) is dict
        self.published_names.append(basename)
        self.published_records.append(deepcopy(dict(prepared_record)))
        self.rows.append(
            (basename, record_type, status_text, deepcopy(body))
        )
        if basename == "campaign.completion.json":
            assert type(self.completion_result_candidate_or_null) is dict
            self.authoritative_completion_result_or_null = deepcopy(
                self.completion_result_candidate_or_null
            )
            self.completion_commit_verified = True
        return deepcopy(dict(prepared_record))

    def prepare_bytes(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> bytes:
        return TOOL._canonical_json_bytes(
            self.prepare(basename, record_type, status_text, body)
        )

    def cache_completion_result_candidate(
        self,
        result: Mapping[str, object],
        _completion_ledger_record: Mapping[str, object],
    ) -> dict[str, object]:
        self.completion_result_candidate_or_null = deepcopy(dict(result))
        return deepcopy(dict(result))

    def publish(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        return self.publish_prepared(
            basename,
            self.prepare(basename, record_type, status_text, body),
        )

    def close(self) -> None:
        self.closed = True


def _completed_real_ledger(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    object,
    dict[str, object],
]:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "completed-ledger"
    )
    capsule = _capsule_stub(contract=contract)
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    result = TOOL._publish_terminal(
        ledger,
        contract,
        capsule,
        status_text=contract["terminal_failure_policy"]["crash_disposition"],
        invocation_count=0,
        last_slot_id_or_null=None,
        last_slot_outcome_digest_or_null=None,
        error_code_or_null="KeyboardInterrupt",
        error_detail_or_null="injected legal pre-slot terminal",
    )
    return contract, capsule, authorization, ledger, result


def _completed_real_success_ledger(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    object,
    dict[str, object],
]:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "success-ledger"
    )
    slots = contract["slot_registry"]
    assert type(slots) is list
    slot = slots[0]
    assert type(slot) is dict
    trace_root = tmp_path / "slot-01-trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = os.fspath(trace_root)
    capsule = _semantic_capsule(contract)
    authorization = TOOL._authorization_template(contract, capsule)
    wire = _success_wire(contract, slot)
    stdout = CORE.canonical_campaign_json_bytes_v2(wire)
    stdout_facts = CORE.classify_supervisor_stdout_v2(stdout)
    process = _process(stdout, 0)
    trajectory = _trajectory(slot, outcome="OK")
    supervisor_artifact, external_artifact = _valid_artifacts(wire)
    source_before = _source_replay("real-success-source-before")
    source_after = _source_replay("real-success-source-after")
    future = _slot_absence_evidence(
        capsule,
        slots[1:],
        "SLOT_01 future slots before outcome",
    )
    cleanup = TOOL._cleanup_replay_evidence(
        process,
        trajectory,
        True,
        [],
        None,
    )
    binding = TOOL._binding_record(
        contract,
        capsule,
        slot,
        process,
        stdout_facts,
        trajectory,
        supervisor_artifact,
        external_artifact,
        True,
        [],
        None,
        source_before,
        source_after,
        future,
        cleanup,
    )
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        process,
        stdout_facts,
        trajectory,
        supervisor_artifact,
        external_artifact,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    decision = CORE.evaluate_slot_observation_v2(observation)
    assert decision == CORE.SUCCESS_V2
    decision_audit = CORE.evaluate_slot_observation_with_audit_v2(
        observation
    )
    assert decision_audit["decision"] == decision
    outcome = TOOL._slot_outcome_record(
        contract,
        capsule,
        slot,
        binding,
        observation,
        decision_audit,
    )
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    ledger.publish(
        "slot-01.intent.json",
        "SLOT_INTENT",
        "SLOT_01_INTENT_STAGED",
        {
            "slot": deepcopy(slot),
            "argv": TOOL._slot_argv(contract, slot),
            "environment": TOOL._runtime_environment(
                contract,
                git_worktree=str(slot["clone_root"]),
            ),
            "previous_slot_id_or_null": None,
            "slot_freshness_before_intent": _slot_absence_evidence(
                capsule,
                [slot],
                "SLOT_01 before intent",
            ),
        },
    )
    ledger.publish(
        "slot-01.consumed.json",
        "SLOT_CONSUMED",
        "SLOT_01_CONSUMED_OUTCOME_UNKNOWN",
        {
            "slot_id": "SLOT_01",
            "slot_ordinal": 1,
            "argv": TOOL._slot_argv(contract, slot),
            "source_replay_before_root": source_before["root"],
            "docker_absence_before_spawn": True,
            "scheduling_admission": _admitted_scheduling_evidence(contract),
            "scheduling_probe_cleanup": _scheduling_probe_cleanup_receipt(
                capsule,
                slot,
            ),
        },
    )
    ledger.publish(
        "slot-01.trajectory-binding.json",
        "EXTERNAL_TRAJECTORY_BINDING",
        "COMPLETE_OK",
        binding,
    )
    ledger.publish(
        "slot-01.outcome.json",
        "SLOT_OUTCOME",
        decision,
        outcome,
    )
    result = TOOL._publish_terminal(
        ledger,
        contract,
        capsule,
        status_text=contract["success_policy"]["terminal_status"],
        invocation_count=999,
        last_slot_id_or_null=None,
        last_slot_outcome_digest_or_null=None,
    )
    return contract, capsule, authorization, ledger, result


def _full_real_gap_ledger_prefix(
    tmp_path: Path,
    *,
    completed_gap_slots: int = 3,
    scheduling_probe_parent: Path | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    object,
]:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "full-gap-prefix-ledger"
    )
    slots = contract["slot_registry"]
    assert type(slots) is list
    for ordinal, slot in enumerate(slots, start=1):
        assert type(slot) is dict
        if scheduling_probe_parent is not None:
            slot["scheduling_probe_root"] = os.fspath(
                scheduling_probe_parent
                / f"slot-{ordinal:02d}-scheduling-probe"
            )
        trace_root = tmp_path / f"slot-{ordinal:02d}-trace"
        trace_root.mkdir(mode=0o700)
        trace_root.chmod(0o700)
        slot["trace_root"] = os.fspath(trace_root)
    assert 0 <= completed_gap_slots <= 3
    capsule = _semantic_capsule(
        contract,
        real_parent_paths=(
            Path(str(contract["fixed_external_roots"]["ledger_root"])).parent,
        ),
    )
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    for ordinal, slot in enumerate(slots[:completed_gap_slots], start=1):
        assert type(slot) is dict
        detail, failure_rows = _held_final_gap_rows()
        stdout = CORE.canonical_campaign_json_bytes_v2(_error_wire(detail))
        stdout_facts = CORE.classify_supervisor_stdout_v2(stdout)
        process = _process(stdout, 1)
        trajectory = _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=failure_rows,
            terminal_error_detail=detail,
        )
        supervisor_artifact, external_artifact = _absent_artifacts()
        source_before = _source_replay(f"gap-{ordinal}-source-before")
        source_after = _source_replay(f"gap-{ordinal}-source-after")
        future = _slot_absence_evidence(
            capsule,
            slots[ordinal:],
            f"{slot['slot_id']} future slots before outcome",
        )
        cleanup = TOOL._cleanup_replay_evidence(
            process,
            trajectory,
            True,
            [],
            None,
        )
        binding = TOOL._binding_record(
            contract,
            capsule,
            slot,
            process,
            stdout_facts,
            trajectory,
            supervisor_artifact,
            external_artifact,
            True,
            [],
            None,
            source_before,
            source_after,
            future,
            cleanup,
        )
        observation = TOOL._build_slot_observation(
            contract,
            slot,
            process,
            stdout_facts,
            trajectory,
            supervisor_artifact,
            external_artifact,
            source_before_certain=True,
            source_after_certain=True,
            docker_absent=True,
            external_binding_certain=True,
        )
        decision = CORE.evaluate_slot_observation_v2(observation)
        assert decision == (
            CORE.RETRY_EXACT_GAP_V2
            if ordinal < 3
            else CORE.EXHAUSTED_RETRYABLE_GAP_V2
        )
        decision_audit = CORE.evaluate_slot_observation_with_audit_v2(
            observation
        )
        assert decision_audit["decision"] == decision
        outcome = TOOL._slot_outcome_record(
            contract,
            capsule,
            slot,
            binding,
            observation,
            decision_audit,
        )
        stem = f"slot-{ordinal:02d}"
        ledger.publish(
            f"{stem}.intent.json",
            "SLOT_INTENT",
            f"{slot['slot_id']}_INTENT_STAGED",
            {
                "slot": deepcopy(slot),
                "argv": TOOL._slot_argv(contract, slot),
                "environment": TOOL._runtime_environment(
                    contract,
                    git_worktree=str(slot["clone_root"]),
                ),
                "previous_slot_id_or_null": slot["previous_slot_id"],
                "slot_freshness_before_intent": _slot_absence_evidence(
                    capsule,
                    [slot],
                    f"{slot['slot_id']} before intent",
                ),
            },
        )
        ledger.publish(
            f"{stem}.consumed.json",
            "SLOT_CONSUMED",
            f"{slot['slot_id']}_CONSUMED_OUTCOME_UNKNOWN",
            {
                "slot_id": slot["slot_id"],
                "slot_ordinal": ordinal,
                "argv": TOOL._slot_argv(contract, slot),
                "source_replay_before_root": source_before["root"],
                "docker_absence_before_spawn": True,
                "scheduling_admission": _admitted_scheduling_evidence(
                    contract,
                    str(slot["slot_id"]),
                ),
                "scheduling_probe_cleanup": (
                    _scheduling_probe_cleanup_receipt(capsule, slot)
                ),
            },
        )
        ledger.publish(
            f"{stem}.trajectory-binding.json",
            "EXTERNAL_TRAJECTORY_BINDING",
            "COMPLETE_FAIL",
            binding,
        )
        ledger.publish(
            f"{stem}.outcome.json",
            "SLOT_OUTCOME",
            decision,
            outcome,
        )
    return contract, capsule, authorization, ledger


def _produce_exact_scheduling_pending(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    completed_gap_slots: int,
    run_reentry: bool,
    reentry_start_completed_gap_slots: int | None = None,
    pending_condition: str = "COMPLETE_GATE_EXHAUSTION",
) -> tuple[
    dict[str, object],
    object,
    TOOL.ShadowRetryCampaignPendingV1,
    object,
]:
    root.mkdir(mode=0o700)
    contract, capsule, _authorization, ledger = _full_real_gap_ledger_prefix(
        root,
        completed_gap_slots=completed_gap_slots,
        scheduling_probe_parent=root,
    )
    slots = contract["slot_registry"]
    assert type(slots) is list
    slot = slots[completed_gap_slots]
    assert type(slot) is dict
    context = {
        "freshness_parent_identities": capsule[
            "freshness_parent_identities"
        ],
        "slots": slots,
    }
    assert pending_condition in {
        "COMPLETE_GATE_EXHAUSTION",
        TOOL._SCHEDULING_JOINED_TASK_QUIESCENCE_CONDITION_V2R17,
    }

    def absence(
        _context: Mapping[str, object],
        observed_slots: Sequence[Mapping[str, object]],
        *,
        label: str,
    ) -> dict[str, object]:
        return _slot_absence_evidence(capsule, observed_slots, label)

    gate_error = (
        _exact_scheduling_exhaustion_error(
            contract,
            str(slot["slot_id"]),
        )
        if pending_condition == "COMPLETE_GATE_EXHAUSTION"
        else _exact_joined_task_quiescence_timeout_error(
            contract,
            str(slot["slot_id"]),
        )
    )

    def exhaust_gate(*_args: object, **_kwargs: object) -> NoReturn:
        raise gate_error

    monkeypatch.setattr(TOOL, "_slot_roots_absence_evidence", absence)
    monkeypatch.setattr(
        TOOL,
        "_slot_registry",
        lambda observed_contract: deepcopy(
            observed_contract["slot_registry"]
        ),
    )
    monkeypatch.setattr(TOOL, "_run_scheduling_admission_gate_v2", exhaust_gate)
    reentry_start = (
        completed_gap_slots
        if reentry_start_completed_gap_slots is None
        else reentry_start_completed_gap_slots
    )
    assert not run_reentry or 0 <= reentry_start <= completed_gap_slots
    supervisor_this_entry = (
        completed_gap_slots - reentry_start
        if run_reentry
        else completed_gap_slots
    )
    retry_this_entry = supervisor_this_entry
    run_entry_state: dict[str, object] = {
        "run_reentry": run_reentry,
        "authorization_lookup_count": 1,
        "authorization_consumption_count": 0 if run_reentry else 1,
        "ledger_creation_count": 0 if run_reentry else 1,
        "ledger_write_count": (
            0 if run_reentry else 2 + 4 * completed_gap_slots
        ),
        "campaign_execution_count": 0 if run_reentry else 1,
        "retry_transition_count": retry_this_entry,
        "supervisor_invocation_count": supervisor_this_entry,
        "starting_prefix_record_count": (
            2 + 4 * reentry_start if run_reentry else 0
        ),
        "starting_supervisor_invocation_count": (
            reentry_start if run_reentry else 0
        ),
        "starting_slot_ordinal": (
            reentry_start + 1 if run_reentry else 1
        ),
    }
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as generated:
        TOOL._run_slot_scheduling_admission_v2r17(
            contract,
            context,
            capsule,
            ledger,
            slot,
            run_entry_state,
        )
    error = generated.value
    options = TOOL.argparse.Namespace(
        run=True,
        capsule_digest=capsule["capsule_digest"],
    )
    return contract, options, error, ledger


@pytest.mark.parametrize("completed_gap_slots", (0, 1, 2))
def test_scheduling_pending_prefix_verifier_accepts_exact_clean_boundaries(
    tmp_path: Path,
    completed_gap_slots: int,
) -> None:
    root = tmp_path / f"prefix-{completed_gap_slots}"
    root.mkdir(mode=0o700)
    contract, capsule, authorization, ledger = _full_real_gap_ledger_prefix(
        root,
        completed_gap_slots=completed_gap_slots,
    )
    try:
        prefix = TOOL._verify_scheduling_pending_ledger_prefix_v2r17(
            contract,
            capsule,
            authorization,
            ledger.published_names,
            ledger.published_records,
        )
        assert prefix["record_count"] == 2 + 4 * completed_gap_slots
        assert prefix["pending_slot_ordinal"] == completed_gap_slots + 1
        assert prefix["pending_slot_id"] == (
            f"SLOT_{completed_gap_slots + 1:02d}"
        )
        assert prefix["supervisor_invocation_count"] == completed_gap_slots
        assert prefix["authorization_consumption_count_durable"] == 1
    finally:
        ledger.close()


def test_scheduling_pending_prefix_verifier_rejects_shape_chain_and_nongap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prefix-rejections"
    root.mkdir(mode=0o700)
    contract, capsule, authorization, ledger = _full_real_gap_ledger_prefix(
        root,
        completed_gap_slots=1,
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as illegal_count:
            TOOL._verify_scheduling_pending_ledger_prefix_v2r17(
                contract,
                capsule,
                authorization,
                ledger.published_names[:-1],
                ledger.published_records[:-1],
            )
        assert illegal_count.value.code == "FAIL_CAMPAIGN_LEDGER"

        corrupt_records = deepcopy(ledger.published_records)
        corrupt_records[-1]["record_digest"] = _tagged("corrupt-prefix")
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as corrupt:
            TOOL._verify_scheduling_pending_ledger_prefix_v2r17(
                contract,
                capsule,
                authorization,
                ledger.published_names,
                corrupt_records,
            )
        assert corrupt.value.code == "FAIL_CAMPAIGN_LEDGER"

        audit_tamper_records = deepcopy(ledger.published_records)
        audit_body = audit_tamper_records[-1]["body"]
        assert type(audit_body) is dict
        audit = audit_body["decision_audit"]
        assert type(audit) is dict
        predicate_results = audit["predicate_results"]
        assert type(predicate_results) is list
        predicate_results[3]["result"] = "REJECT"
        audit["first_reject_clause_or_null"] = "GAP_GRAMMAR"
        audit["matched_path_or_null"] = None
        unsigned_outcome = deepcopy(audit_body)
        unsigned_outcome.pop("record_digest")
        audit_body["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["slot_outcome"],
            unsigned_outcome,
        )
        unsigned_ledger = deepcopy(audit_tamper_records[-1])
        unsigned_ledger.pop("record_digest")
        audit_tamper_records[-1]["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["ledger_record"],
            unsigned_ledger,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as audit_tamper:
            TOOL._verify_scheduling_pending_ledger_prefix_v2r17(
                contract,
                capsule,
                authorization,
                ledger.published_names,
                audit_tamper_records,
            )
        assert audit_tamper.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert audit_tamper.value.terminal is True

        nongap_records = deepcopy(ledger.published_records)
        nongap_records[-1]["body"]["decision"] = CORE.TERMINAL_NONRETRYABLE_V2
        unsigned = deepcopy(nongap_records[-1])
        unsigned.pop("record_digest")
        nongap_records[-1]["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["ledger_record"],
            unsigned,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as nongap:
            TOOL._verify_scheduling_pending_ledger_prefix_v2r17(
                contract,
                capsule,
                authorization,
                ledger.published_names,
                nongap_records,
            )
        assert nongap.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_open_scheduling_pending_prefix_restores_cursor_and_rejects_contention(
    tmp_path: Path,
) -> None:
    root = tmp_path / "open-prefix"
    root.mkdir(mode=0o700)
    contract, capsule, authorization, ledger = _full_real_gap_ledger_prefix(
        root,
        completed_gap_slots=1,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as contention:
        TOOL._LedgerV2.open_scheduling_pending_prefix(
            contract,
            capsule,
            authorization,
        )
    assert contention.value.code == "FAIL_CAMPAIGN_LEDGER"
    ledger.close()

    reopened, prefix = TOOL._LedgerV2.open_scheduling_pending_prefix(
        contract,
        capsule,
        authorization,
    )
    try:
        assert reopened.locked is True
        assert reopened.sequence == 6
        assert reopened.last_name_index == 5
        assert reopened.published_names == list(TOOL._LEDGER_RECORD_NAMES_V2[:6])
        assert reopened.previous_record_digest == (
            reopened.published_records[-1]["record_digest"]
        )
        assert prefix["pending_slot_id"] == "SLOT_02"
        assert prefix["supervisor_invocation_count"] == 1
    finally:
        reopened.close()


def test_transient_scheduling_probe_cleanup_is_durable_and_exact(
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = deepcopy(_slot(contract))
    probe = tmp_path / "scheduling-probe"
    slot["scheduling_probe_root"] = os.fspath(probe)
    capsule = _semantic_capsule(
        contract,
        real_parent_paths=(tmp_path,),
    )
    context = {
        "freshness_parent_identities": capsule[
            "freshness_parent_identities"
        ]
    }
    parent_fd, root_fd, root_identity = (
        TOOL._create_transient_scheduling_probe_root_v2r17(context, slot)
    )
    receipt = TOOL._cleanup_scheduling_probe_root_v2r17(
        contract,
        context,
        slot,
        parent_fd,
        root_fd,
        root_identity,
    )
    assert receipt["passed"] is True
    assert receipt["pathname_absent"] is True
    assert not probe.exists()
    assert TOOL._scheduling_probe_cleanup_receipt_exact_v2r17(
        context,
        slot,
        receipt,
    )


@pytest.mark.parametrize("fault", ("identity", "rmdir", "fsync"))
def test_transient_scheduling_probe_cleanup_fault_is_terminal_not_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    contract = _contract()
    slot = deepcopy(_slot(contract))
    probe = tmp_path / f"scheduling-probe-{fault}"
    slot["scheduling_probe_root"] = os.fspath(probe)
    capsule = _semantic_capsule(
        contract,
        real_parent_paths=(tmp_path,),
    )
    context = {
        "freshness_parent_identities": capsule[
            "freshness_parent_identities"
        ]
    }
    parent_fd, root_fd, root_identity = (
        TOOL._create_transient_scheduling_probe_root_v2r17(context, slot)
    )
    real_verify = TOOL._verify_held_directory_path
    real_rmdir = TOOL.os.rmdir
    real_fsync = TOOL.os.fsync
    if fault == "identity":
        monkeypatch.setattr(
            TOOL,
            "_verify_held_directory_path",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected cleanup identity failure")
            ),
        )
    elif fault == "rmdir":
        monkeypatch.setattr(
            TOOL.os,
            "rmdir",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected cleanup rmdir failure")
            ),
        )
    else:
        monkeypatch.setattr(
            TOOL.os,
            "fsync",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected cleanup fsync failure")
            ),
        )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as terminal:
        TOOL._cleanup_scheduling_probe_root_v2r17(
            contract,
            context,
            slot,
            parent_fd,
            root_fd,
            root_identity,
        )
    assert not isinstance(terminal.value, TOOL.ShadowRetryCampaignPendingV1)
    assert terminal.value.code == TOOL._SCHEDULING_ADMISSION_FAILURE_V2
    assert terminal.value.terminal is True
    assert terminal.value.receipt["passed"] is False
    monkeypatch.setattr(TOOL, "_verify_held_directory_path", real_verify)
    monkeypatch.setattr(TOOL.os, "rmdir", real_rmdir)
    monkeypatch.setattr(TOOL.os, "fsync", real_fsync)
    if probe.exists():
        probe.rmdir()


@pytest.mark.parametrize(
    (
        "completed_gap_slots",
        "run_reentry",
        "reentry_start_completed_gap_slots",
    ),
    (
        (0, False, None),
        (1, False, None),
        (2, False, None),
        (1, True, None),
        (2, True, 1),
    ),
)
@pytest.mark.parametrize(
    "pending_condition",
    (
        "COMPLETE_GATE_EXHAUSTION",
        TOOL._SCHEDULING_JOINED_TASK_QUIESCENCE_CONDITION_V2R17,
    ),
)
def test_exact_scheduling_condition_produces_classified_pending_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    completed_gap_slots: int,
    run_reentry: bool,
    reentry_start_completed_gap_slots: int | None,
    pending_condition: str,
) -> None:
    contract, options, error, ledger = _produce_exact_scheduling_pending(
        monkeypatch,
        tmp_path / (
            f"pending-{completed_gap_slots}-{int(run_reentry)}-"
            f"{reentry_start_completed_gap_slots}"
        ),
        completed_gap_slots=completed_gap_slots,
        run_reentry=run_reentry,
        reentry_start_completed_gap_slots=(
            reentry_start_completed_gap_slots
        ),
        pending_condition=pending_condition,
    )
    try:
        assert TOOL._scheduling_pending_is_exact_v2r17(
            contract,
            options,
            error,
        )
        receipt = error.receipt
        assert receipt["condition"] == pending_condition
        expected_this_entry = (
            completed_gap_slots
            - (
                completed_gap_slots
                if reentry_start_completed_gap_slots is None
                else reentry_start_completed_gap_slots
            )
            if run_reentry
            else completed_gap_slots
        )
        assert receipt["supervisor_invocation_count_durable"] == (
            completed_gap_slots
        )
        assert receipt["supervisor_invocation_count_this_entry"] == (
            expected_this_entry
        )
        assert receipt["retry_transition_count_this_entry"] == (
            expected_this_entry
        )
        assert receipt["scheduling_admission_invocation_count_this_entry"] == (
            expected_this_entry + 1
        )
        assert receipt["transient_probe_root_creation_count_this_entry"] == (
            expected_this_entry + 1
        )
        assert receipt["transient_probe_root_removal_count_this_entry"] == (
            expected_this_entry + 1
        )
        assert receipt["docker_daemon_contact_count_this_entry"] == (
            2 * expected_this_entry
        )
        assert not Path(
            str(
                contract["slot_registry"][completed_gap_slots][
                    "scheduling_probe_root"
                ]
            )
        ).exists()
    finally:
        ledger.close()


def test_joined_task_quiescence_receipt_mutations_are_rejected() -> None:
    contract = _contract()
    policy = TOOL._scheduling_admission_policy_v2(contract)
    exact = _exact_joined_task_quiescence_timeout_error(
        contract,
        "SLOT_01",
    ).receipt
    assert type(exact) is dict

    def mutate(field: str, value: object) -> dict[str, object]:
        candidate = deepcopy(exact)
        candidate[field] = value
        return candidate

    candidates = [
        mutate("condition", "COMPLETE_GATE_EXHAUSTION"),
        mutate("quiescence_timeout_ns", 1_000_000_001),
        mutate("quiescence_poll_interval_ns", True),
        mutate("component", "SUSTAINED_PUBLICATION"),
        mutate(
            "joined_thread_names_in_start_order",
            list(reversed(exact["joined_thread_names_in_start_order"])),
        ),
        mutate(
            "joined_thread_native_ids_in_start_order",
            [2001, 2001, 2003, 2004],
        ),
        mutate(
            "joined_thread_is_alive_in_start_order",
            [False, False, False, True],
        ),
        mutate(
            "join_completed_in_start_order",
            [True, True, True, False],
        ),
        mutate(
            "lingering_known_joined_task_ids_in_sorted_order",
            [2999],
        ),
        mutate("unknown_task_ids_at_timeout_in_sorted_order", [2999]),
        mutate("terminal_drain_policy_exceeded", True),
        mutate("primary_error_present", True),
        mutate("publisher_error_count", 1),
        mutate("wait_duration_ns", 999_999_999),
    ]
    for candidate in candidates:
        assert not TOOL._joined_task_quiescence_timeout_receipt_exact_v2r17(
            policy,
            "SLOT_01",
            candidate,
        )


def test_joined_task_quiescence_pending_nested_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, options, error, ledger = _produce_exact_scheduling_pending(
        monkeypatch,
        tmp_path / "joined-timeout-pending-mutation",
        completed_gap_slots=0,
        run_reentry=False,
        pending_condition=(
            TOOL._SCHEDULING_JOINED_TASK_QUIESCENCE_CONDITION_V2R17
        ),
    )
    try:
        assert TOOL._scheduling_pending_is_exact_v2r17(
            contract,
            options,
            error,
        )
        forged_receipt = deepcopy(error.receipt)
        forged_receipt["scheduling_admission_receipt"][
            "unknown_task_ids_at_timeout_in_sorted_order"
        ] = [2999]
        forged = TOOL.ShadowRetryCampaignPendingV1(
            error.code,
            error.detail,
            phase=error.phase,
            condition=error.condition,
            receipt=forged_receipt,
            dependency_gate_invocation_count=1,
        )
        assert not TOOL._scheduling_pending_is_exact_v2r17(
            contract,
            options,
            forged,
        )
    finally:
        ledger.close()


def test_scheduling_pending_classifier_rejects_bool_and_count_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, options, error, ledger = _produce_exact_scheduling_pending(
        monkeypatch,
        tmp_path / "pending-classifier",
        completed_gap_slots=2,
        run_reentry=False,
    )
    try:
        assert TOOL._scheduling_pending_is_exact_v2r17(
            contract,
            options,
            error,
        )
        integer_fields = (
            "pending_slot_ordinal",
            "authorization_lookup_count_this_entry",
            "authorization_consumption_count_this_entry",
            "ledger_creation_count_this_entry",
            "ledger_write_count_this_entry",
            "campaign_execution_count_this_entry",
            "retry_transition_count_this_entry",
            "supervisor_invocation_count_this_entry",
            "authorization_consumption_count_durable",
            "supervisor_invocation_count_durable",
            "scheduling_admission_invocation_count_this_entry",
            "transient_probe_root_creation_count_this_entry",
            "transient_probe_root_removal_count_this_entry",
            "docker_daemon_contact_count_this_entry",
            "pending_occurrence_count_this_process",
        )
        for field in integer_fields:
            forged_receipt = deepcopy(error.receipt)
            forged_receipt[field] = True
            forged = TOOL.ShadowRetryCampaignPendingV1(
                error.code,
                error.detail,
                phase=error.phase,
                condition=error.condition,
                receipt=forged_receipt,
                dependency_gate_invocation_count=(
                    error.dependency_gate_invocation_count
                ),
            )
            assert not TOOL._scheduling_pending_is_exact_v2r17(
                contract,
                options,
                forged,
            ), field

        forged_receipt = deepcopy(error.receipt)
        forged_receipt[
            "scheduling_admission_invocation_count_this_entry"
        ] += 1
        forged = TOOL.ShadowRetryCampaignPendingV1(
            error.code,
            error.detail,
            phase=error.phase,
            condition=error.condition,
            receipt=forged_receipt,
            dependency_gate_invocation_count=1,
        )
        assert not TOOL._scheduling_pending_is_exact_v2r17(
            contract,
            options,
            forged,
        )
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "pending_condition",
    (
        "COMPLETE_GATE_EXHAUSTION",
        TOOL._SCHEDULING_JOINED_TASK_QUIESCENCE_CONDITION_V2R17,
    ),
)
def test_scheduling_pending_cli_returns_75_and_invalid_receipt_returns_1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
    pending_condition: str,
) -> None:
    contract, options, pending, ledger = _produce_exact_scheduling_pending(
        monkeypatch,
        tmp_path / "pending-cli",
        completed_gap_slots=1,
        run_reentry=True,
        pending_condition=pending_condition,
    )
    digest = str(options.capsule_digest)
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _value: None)
    monkeypatch.setattr(
        TOOL,
        "run_phase3_q05b_shadow_retry_campaign_v2",
        lambda _digest: (_ for _ in ()).throw(pending),
    )
    try:
        assert TOOL.main(["--run", "--capsule-digest", digest]) == 75
        accepted = TOOL._strict_json(
            capsysbinary.readouterr().out,
            "scheduling pending CLI",
        )
        assert accepted["schema_version"] == TOOL._SCHEDULING_PENDING_SCHEMA_V2
        assert accepted["status"] == "CAMPAIGN_SCHEDULING_ADMISSION_PENDING"
        assert accepted["pending_slot_id"] == "SLOT_02"
        assert accepted["run_reentry"] is True
        assert accepted["resumable"] is True
        assert accepted["condition"] == pending_condition

        forged_receipt = deepcopy(pending.receipt)
        forged_receipt["pending_slot_ordinal"] = True
        forged = TOOL.ShadowRetryCampaignPendingV1(
            pending.code,
            pending.detail,
            phase=pending.phase,
            condition=pending.condition,
            receipt=forged_receipt,
            dependency_gate_invocation_count=1,
        )
        monkeypatch.setattr(
            TOOL,
            "run_phase3_q05b_shadow_retry_campaign_v2",
            lambda _digest: (_ for _ in ()).throw(forged),
        )
        assert TOOL.main(["--run", "--capsule-digest", digest]) == 1
        rejected = TOOL._strict_json(
            capsysbinary.readouterr().out,
            "invalid scheduling pending CLI",
        )
        assert rejected["status"] == "CAMPAIGN_COMMAND_FAILED_CLOSED"
        assert rejected["resumable"] is False
        assert rejected["error_receipt_or_null"] == forged_receipt
    finally:
        ledger.close()


def test_run_one_slot_pending_gate_precedes_all_slot_mutation_and_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub("caller-gate-order")
    slot = _slot(contract)
    ledger = _FakeLedger()
    calls: list[str] = []
    pending = TOOL.ShadowRetryCampaignPendingV1(
        TOOL._SCHEDULING_PENDING_CODE_V2R17,
        "injected exact gate pending",
        phase="SLOT_SCHEDULING_ADMISSION",
        condition="COMPLETE_GATE_EXHAUSTION",
        receipt={},
        dependency_gate_invocation_count=1,
    )

    def gate(*_args: object, **_kwargs: object) -> NoReturn:
        calls.append("gate")
        raise pending

    def forbidden(label: str):
        def invoke(*_args: object, **_kwargs: object) -> NoReturn:
            calls.append(label)
            raise AssertionError(f"pending gate reached {label}")

        return invoke

    monkeypatch.setattr(TOOL, "_run_slot_scheduling_admission_v2r17", gate)
    monkeypatch.setattr(TOOL, "_prepare_clone", forbidden("clone"))
    monkeypatch.setattr(TOOL, "_copy_tree_exact", forbidden("cargo"))
    monkeypatch.setattr(TOOL, "_safe_docker_inventory", forbidden("docker"))
    monkeypatch.setattr(TOOL, "_bounded_process", forbidden("popen"))
    run_entry_state = {
        "run_reentry": False,
        "authorization_lookup_count": 1,
        "authorization_consumption_count": 1,
        "ledger_creation_count": 1,
        "ledger_write_count": 2,
        "campaign_execution_count": 1,
        "retry_transition_count": 0,
        "supervisor_invocation_count": 0,
        "starting_prefix_record_count": 0,
        "starting_supervisor_invocation_count": 0,
        "starting_slot_ordinal": 1,
    }
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as observed:
        TOOL._run_one_slot(
            contract,
            {"slots": contract["slot_registry"]},
            capsule,
            ledger,  # type: ignore[arg-type]
            slot,
            [0],
            run_entry_state,
        )
    assert observed.value is pending
    assert calls == ["gate"]
    assert ledger.published_names == []


def test_run_initial_entry_wires_single_consumption_before_slot_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "initial-ledger"
    )
    capsule = _capsule_stub("initial-run-wiring")
    authorization = {"authorization_digest": _tagged("initial-auth")}
    slots = contract["slot_registry"]
    assert type(slots) is list
    context = {
        "capsule": capsule,
        "slots": slots,
        "freshness_parent_identities": {},
    }
    ledger = _FakeLedger()
    observed_state: dict[str, object] = {}
    pending = TOOL.ShadowRetryCampaignPendingV1(
        "PENDING_TEST",
        "stop after initial wiring",
        phase="SLOT_SCHEDULING_ADMISSION",
        condition="COMPLETE_GATE_EXHAUSTION",
        receipt={},
        dependency_gate_invocation_count=1,
    )

    def stop_at_slot(
        _contract: Mapping[str, object],
        _context: Mapping[str, object],
        _capsule: Mapping[str, object],
        _ledger: object,
        slot: Mapping[str, object],
        invocation_counter: list[int],
        run_entry_state: dict[str, object],
    ) -> NoReturn:
        assert slot["slot_id"] == "SLOT_01"
        assert invocation_counter == [0]
        observed_state.update(deepcopy(run_entry_state))
        raise pending

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(
        TOOL,
        "_preflight_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(TOOL, "_load_authorization", lambda *_args: authorization)
    monkeypatch.setattr(TOOL, "_verify_fresh_targets", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", stop_at_slot)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as stopped:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            str(capsule["capsule_digest"])
        )
    assert stopped.value is pending
    assert observed_state["run_reentry"] is False
    assert observed_state["authorization_consumption_count"] == 1
    assert observed_state["ledger_creation_count"] == 1
    assert observed_state["campaign_execution_count"] == 1
    assert observed_state["ledger_write_count"] == 2
    assert ledger.published_names == list(TOOL._LEDGER_RECORD_NAMES_V2[:2])


def test_run_reopens_prefix6_at_slot2_without_new_consumption_or_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    ledger_path = tmp_path / "resume-ledger"
    ledger_path.mkdir(mode=0o700)
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(ledger_path)
    capsule = _capsule_stub("resume-run-wiring")
    authorization = {"authorization_digest": _tagged("resume-auth")}
    slots = contract["slot_registry"]
    assert type(slots) is list
    context = {
        "capsule": capsule,
        "slots": slots,
        "freshness_parent_identities": {},
    }
    ledger = _FakeLedger()
    ledger.published_names = list(TOOL._LEDGER_RECORD_NAMES_V2[:6])
    prefix = {
        "pending_slot_ordinal": 2,
        "pending_slot_id": "SLOT_02",
        "record_count": 6,
        "record_names_in_sequence_order": list(TOOL._LEDGER_RECORD_NAMES_V2[:6]),
        "record_digests_in_sequence_order": [_tagged(str(i)) for i in range(6)],
        "authorization_consumption_count_durable": 1,
        "supervisor_invocation_count": 1,
        "last_slot_id_or_null": "SLOT_01",
        "last_slot_outcome_digest_or_null": _tagged("slot-01-outcome"),
    }
    observed_state: dict[str, object] = {}
    pending = TOOL.ShadowRetryCampaignPendingV1(
        "PENDING_TEST",
        "stop after resume wiring",
        phase="SLOT_SCHEDULING_ADMISSION",
        condition="COMPLETE_GATE_EXHAUSTION",
        receipt={},
        dependency_gate_invocation_count=1,
    )

    class ResumeLedger:
        def __init__(self, *_args: object) -> None:
            raise AssertionError("resume must not create a ledger")

        @classmethod
        def open_scheduling_pending_prefix(
            cls,
            *_args: object,
        ) -> tuple[object, dict[str, object]]:
            return ledger, deepcopy(prefix)

    def stop_at_slot(
        _contract: Mapping[str, object],
        _context: Mapping[str, object],
        _capsule: Mapping[str, object],
        _ledger: object,
        slot: Mapping[str, object],
        invocation_counter: list[int],
        run_entry_state: dict[str, object],
    ) -> NoReturn:
        assert slot["slot_id"] == "SLOT_02"
        assert invocation_counter == [1]
        observed_state.update(deepcopy(run_entry_state))
        raise pending

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(
        TOOL,
        "_preflight_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(TOOL, "_load_authorization", lambda *_args: authorization)
    monkeypatch.setattr(TOOL, "_LedgerV2", ResumeLedger)
    monkeypatch.setattr(
        TOOL,
        "_slot_roots_absence_evidence",
        lambda *_args, **_kwargs: {"certain": True, "error_or_null": None},
    )
    monkeypatch.setattr(
        TOOL,
        "_verify_scheduling_probe_roots_absent_v2r17",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(TOOL, "_run_one_slot", stop_at_slot)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as stopped:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            str(capsule["capsule_digest"])
        )
    assert stopped.value is pending
    assert observed_state["run_reentry"] is True
    assert observed_state["authorization_consumption_count"] == 0
    assert observed_state["ledger_creation_count"] == 0
    assert observed_state["campaign_execution_count"] == 0
    assert observed_state["starting_prefix_record_count"] == 6
    assert observed_state["starting_supervisor_invocation_count"] == 1
    assert observed_state["starting_slot_ordinal"] == 2
    assert ledger.published_names == list(TOOL._LEDGER_RECORD_NAMES_V2[:6])


@pytest.mark.parametrize("fault", ("parent_fsync", "child_open", "root_identity"))
def test_transient_probe_creation_fault_cleans_created_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    contract = _contract()
    slot = deepcopy(_slot(contract))
    probe = tmp_path / f"creation-fault-{fault}"
    slot["scheduling_probe_root"] = os.fspath(probe)
    capsule = _semantic_capsule(contract, real_parent_paths=(tmp_path,))
    context = {
        "freshness_parent_identities": capsule[
            "freshness_parent_identities"
        ]
    }
    real_fsync = TOOL.os.fsync
    real_open = TOOL.os.open
    real_verify = TOOL._verify_held_directory_path
    fsync_count = 0

    def fsync(descriptor: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        if fault == "parent_fsync" and fsync_count == 1:
            raise OSError("injected post-mkdir parent fsync failure")
        real_fsync(descriptor)

    def open_child(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if fault == "child_open" and path == probe.name and dir_fd is not None:
            raise OSError("injected child open failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def verify(
        path: Path,
        descriptor: int,
        label: str,
    ) -> dict[str, int]:
        if fault == "root_identity" and path == probe:
            raise OSError("injected root identity failure")
        return real_verify(path, descriptor, label)

    monkeypatch.setattr(TOOL.os, "fsync", fsync)
    monkeypatch.setattr(TOOL.os, "open", open_child)
    monkeypatch.setattr(TOOL, "_verify_held_directory_path", verify)
    with pytest.raises(BaseException) as failed:
        TOOL._create_transient_scheduling_probe_root_v2r17(context, slot)
    assert not isinstance(failed.value, TOOL.ShadowRetryCampaignPendingV1)
    assert not probe.exists()


@pytest.mark.parametrize("completed_gap_slots", (0, 1, 2))
def test_cleanup_uncertainty_terminal_is_authoritative_only_at_clean_boundary(
    tmp_path: Path,
    completed_gap_slots: int,
) -> None:
    root = tmp_path / f"cleanup-terminal-{completed_gap_slots}"
    root.mkdir(mode=0o700)
    contract, capsule, authorization, ledger = _full_real_gap_ledger_prefix(
        root,
        completed_gap_slots=completed_gap_slots,
    )
    try:
        slot = contract["slot_registry"][completed_gap_slots]
        assert type(slot) is dict
        receipt = _scheduling_probe_cleanup_receipt(capsule, slot)
        receipt.update({
            "rmdir_completed": False,
            "parent_fsync_completed": False,
            "pathname_absent": False,
            "cleanup_errors": ["OSError:injected cleanup uncertainty"],
            "passed": False,
        })
        detail = f"{slot['slot_id']} scheduling probe cleanup or absence uncertain"
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        _terminalize_test_prefix_with_error(
            contract,
            capsule,
            authorization,
            rows,
            error_code=TOOL._SCHEDULING_ADMISSION_FAILURE_V2,
            error_detail=detail,
            error_receipt=receipt,
        )
        assert TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            rows,
        )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"

        forged_receipt = deepcopy(receipt)
        forged_receipt["expected_parent_stable_identity"]["ino"] += 1
        forged_rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))
        )
        _terminalize_test_prefix_with_error(
            contract,
            capsule,
            authorization,
            forged_rows,
            error_code=TOOL._SCHEDULING_ADMISSION_FAILURE_V2,
            error_detail=detail,
            error_receipt=forged_receipt,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as rejected:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                forged_rows,
            )
        assert rejected.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_authorization_explicitly_binds_clean_checkpoint_continuation() -> None:
    contract = _contract()
    authorization = TOOL._authorization_template(contract, _capsule_stub())
    assert authorization[
        "scheduling_checkpoint_reopen_validation_is_campaign_execution"
    ] is False
    assert authorization[
        "scheduling_checkpoint_reopen_validation_is_authorization_consumption"
    ] is False
    assert authorization[
        "scheduling_checkpoint_reopen_validation_is_retry_transition"
    ] is False
    assert authorization[
        "scheduling_checkpoint_reopen_validation_is_supervisor_invocation"
    ] is False
    assert authorization[
        "scheduling_pending_reentry_requires_same_run_entrypoint_and_capsule_digest"
    ] is True
    assert authorization[
        "scheduling_pending_reentry_requires_exact_locked_authoritative_prefix"
    ] is True
    assert authorization[
        "exact_clean_scheduling_checkpoint_continuation_authorized"
    ] is True
    assert authorization[
        "scheduling_pending_eligible_conditions_in_exact_order"
    ] == [
        "COMPLETE_GATE_EXHAUSTION",
        "KNOWN_DEAD_ADMISSION_TASK_QUIESCENCE_TIMEOUT",
    ]
    assert authorization[
        "exact_clean_scheduling_checkpoint_requires_prior_pending_cli_receipt"
    ] is False
    assert authorization["other_crash_state_resume_authorized"] is False


def _append_test_ledger_record(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    records: list[tuple[str, dict[str, object]]],
    basename: str,
    record_type: str,
    status_text: str,
    body: Mapping[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-campaign-ledger-record/2"
        ),
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "campaign_source_binding_digest": contract[
            "campaign_source_binding"
        ]["source_binding_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "sequence": len(records) + 1,
        "record_type": record_type,
        "status": status_text,
        "previous_record_digest_or_null": (
            records[-1][1]["record_digest"] if records else None
        ),
        "body": deepcopy(dict(body)),
        "record_digest": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("record_digest")
    value["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["ledger_record"],
        unsigned,
    )
    records.append((basename, value))
    return value


def _terminalize_test_prefix_with_error(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    rows: list[tuple[str, dict[str, object]]],
    *,
    error_code: str,
    error_detail: str,
    error_receipt: Mapping[str, object],
) -> None:
    progress = TOOL._ledger_progress_snapshot_v2(
        [record for _name, record in rows]
    )
    status = TOOL._error_code_terminal_disposition(contract, error_code)
    terminal = TOOL._campaign_terminal_body(
        contract,
        capsule,
        status_text=status,
        invocation_count=int(progress["supervisor_invocation_count"]),
        last_slot_id_or_null=progress["last_slot_id_or_null"],
        last_slot_outcome_digest_or_null=progress[
            "last_slot_outcome_digest_or_null"
        ],
        error_code_or_null=error_code,
        error_detail_or_null=error_detail,
        error_receipt_or_null=error_receipt,
    )
    terminal_record = _append_test_ledger_record(
        contract,
        capsule,
        authorization,
        rows,
        "campaign.terminal.json",
        "CAMPAIGN_TERMINAL",
        status,
        terminal,
    )
    completion = TOOL._campaign_completion_body(
        contract,
        capsule,
        authorization_digest=str(authorization["authorization_digest"]),
        terminal=terminal,
        terminal_ledger_record=terminal_record,
        preceding_ledger_names=[name for name, _record in rows],
    )
    _append_test_ledger_record(
        contract,
        capsule,
        authorization,
        rows,
        "campaign.completion.json",
        "CAMPAIGN_COMPLETION",
        "CAMPAIGN_COMPLETION_COMMITTED",
        completion,
    )


def _rewrite_terminal_completion_chain(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    rows: list[tuple[str, dict[str, object]]],
    *,
    error_code: str,
    error_detail: str,
    error_receipt: Mapping[str, object],
) -> None:
    terminal_record = rows[-2][1]
    terminal = terminal_record["body"]
    assert type(terminal) is dict
    terminal["status"] = TOOL._error_code_terminal_disposition(
        contract, error_code
    )
    terminal["error_code_or_null"] = error_code
    terminal["error_detail_or_null"] = error_detail
    terminal["error_receipt_or_null"] = deepcopy(dict(error_receipt))
    terminal_unsigned = deepcopy(terminal)
    terminal_unsigned.pop("record_digest")
    terminal["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["campaign_terminal"],
        terminal_unsigned,
    )
    terminal_record["status"] = terminal["status"]
    terminal_record_unsigned = deepcopy(terminal_record)
    terminal_record_unsigned.pop("record_digest")
    terminal_record["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["ledger_record"],
        terminal_record_unsigned,
    )
    completion_record = rows[-1][1]
    completion_record["previous_record_digest_or_null"] = terminal_record[
        "record_digest"
    ]
    completion = TOOL._campaign_completion_body(
        contract,
        capsule,
        authorization_digest=str(authorization["authorization_digest"]),
        terminal=terminal,
        terminal_ledger_record=terminal_record,
        preceding_ledger_names=[name for name, _record in rows[:-1]],
    )
    completion_record["body"] = completion
    completion_record_unsigned = deepcopy(completion_record)
    completion_record_unsigned.pop("record_digest")
    completion_record["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["ledger_record"],
        completion_record_unsigned,
    )


def test_completion_record_is_authoritative_and_offline_verifiable(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        assert ledger.published_names[-2:] == [
            "campaign.terminal.json",
            "campaign.completion.json",
        ]
        assert ledger.completion_commit_verified is True
        assert result["completion_commit_verified"] is True
        assert result["authoritative_completion_record_name"] == (
            "campaign.completion.json"
        )
        rows = list(zip(ledger.published_names, ledger.published_records))
        verification = TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            rows,
        )
        assert verification["status"] == (
            "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
        )
        assert verification["ledger_completion_record_digest"] == result[
            "ledger_completion_record_digest"
        ]
        assert verification["campaign_completion_digest"] == result[
            "campaign_completion_digest"
        ]
        root = Path(str(contract["fixed_external_roots"]["ledger_root"]))
        assert (root / "campaign.terminal.json").is_file()
        assert (root / "campaign.completion.json").is_file()
        disk_verification = TOOL._verify_campaign_completion_ledger_at_v2(
            contract,
            capsule,
            authorization,
            root,
        )
        assert disk_verification == verification
    finally:
        ledger.close()


def test_offline_verifier_rejects_rehashed_unknown_receipt_schema(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        _rewrite_terminal_completion_chain(
            contract,
            capsule,
            authorization,
            rows,
            error_code="KeyboardInterrupt",
            error_detail="injected legal pre-slot terminal",
            error_receipt={
                "schema_version": "forged-receipt/1",
                "argv": ["/not/the/command"],
                "label": "forged label",
                "phase": "FAKE",
            },
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "terminal error disposition" in error.value.detail
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown-field",
        "argv",
        "label",
        "phase",
        "terminal-detail-crosslink",
        "recursive-prior",
    ),
)
def test_offline_verifier_rejects_rehashed_receipt_semantic_forgery(
    tmp_path: Path,
    mutation: str,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        receipt = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-exception-"
                "diagnostic/2"
            ),
            "label": "target tree replay",
            "argv": [
                contract["execution_policy"]["git_executable"],
                "-C",
                contract["execution_policy"]["clone_source_repository"],
                "rev-parse",
                f"{contract['target_commit_a_binding']['target_supervisor_commit']}^{{tree}}",
            ],
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "OSError",
            "error_detail": "injected selector failure",
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": None,
        }
        detail = (
            "target tree replay bounded process raised before a "
            "result: OSError:injected selector failure"
        )
        if mutation == "unknown-field":
            receipt["forged"] = True
        elif mutation == "argv":
            receipt["argv"] = ["/usr/bin/docker", "ps", "--forged"]
        elif mutation == "label":
            receipt["label"] = "forged label"
        elif mutation == "phase":
            receipt["phase"] = "FORGED_PHASE"
        elif mutation == "terminal-detail-crosslink":
            detail += " forged"
        else:
            nested = deepcopy(receipt)
            nested["prior_receipt_or_null"] = None
            receipt["prior_receipt_or_null"] = nested
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        _rewrite_terminal_completion_chain(
            contract,
            capsule,
            authorization,
            rows,
            error_code="FAIL_CAMPAIGN_PROCESS",
            error_detail=detail,
            error_receipt=receipt,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "terminal error disposition" in error.value.detail
    finally:
        ledger.close()


def test_offline_verifier_accepts_each_registered_authoritative_receipt(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        base_rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))
        )
        docker_argv = TOOL._docker_inventory_argv_v2(contract)
        empty_stream = {
            "byte_count": 0,
            "raw_sha256": sha256(b"").hexdigest(),
            "excerpt_hex": "",
            "excerpt_byte_count": 0,
            "excerpt_truncated": False,
        }
        diagnostic = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-diagnostic/2"
            ),
            "label": "Docker read-only inventory",
            "argv": docker_argv,
            "return_code": 7,
            "timed_out": False,
            "stdout_overflow": False,
            "stderr_overflow": False,
            "child_reaped": True,
            "process_group_absent": True,
            "wrapper_resources_closed": True,
            "stdout": deepcopy(empty_stream),
            "stderr": deepcopy(empty_stream),
            "classification": {
                "accepted": False,
                "reason": "injected Docker failure",
            },
        }
        diagnostic_detail = (
            "Docker read-only inventory failed closed: return_code=7; "
            "timed_out=False; stdout_overflow=False; stderr_overflow=False; "
            "stderr_excerpt_hex="
        )
        git_argv = [
            contract["execution_policy"]["git_executable"],
            "-C",
            contract["execution_policy"]["clone_source_repository"],
            "rev-parse",
            f"{contract['target_commit_a_binding']['target_supervisor_commit']}^{{tree}}",
        ]
        git_diagnostic = deepcopy(diagnostic)
        git_diagnostic["label"] = "target tree replay"
        git_diagnostic["argv"] = git_argv
        git_diagnostic_detail = diagnostic_detail.replace(
            "Docker read-only inventory", "target tree replay", 1
        )
        exception = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-exception-"
                "diagnostic/2"
            ),
            "label": "target tree replay",
            "argv": git_argv,
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "OSError",
            "error_detail": "injected selector failure",
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": None,
        }
        exception_detail = (
            "target tree replay bounded process raised before a "
            "result: OSError:injected selector failure"
        )
        docker_error = TOOL._canonical_json_bytes(diagnostic).decode(
            "ascii"
        ).rstrip("\n")
        prespawn = {
            "schema_version": TOOL._PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
            "docker_rows": [],
            "docker_error_or_null": docker_error,
            "bounded_command_diagnostic_or_null": deepcopy(diagnostic),
        }
        prespawn_detail = (
            "SLOT_01 pre-spawn Docker absence is uncertain: " + docker_error
        )
        publication = {
            "schema_version": TOOL._PUBLICATION_RECEIPT_SCHEMA_V2,
            "publication_outcome": "KNOWN_NO_PUBLICATION",
            "publication_label": "slot-supervisor-stdout",
            "publication_basename": "supervisor.stdout",
        }
        publication_detail = (
            "publication_label=slot-supervisor-stdout; "
            "basename=supervisor.stdout; exclusive publication failed"
        )
        omission = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-receipt-omission/2"
            ),
            "status": "RECEIPT_OMITTED_OVER_BOUND",
        }
        cases = (
            (
                "FAIL_CAMPAIGN_PROCESS",
                git_diagnostic_detail,
                git_diagnostic,
            ),
            ("FAIL_CAMPAIGN_PROCESS", exception_detail, exception),
            ("FAIL_CAMPAIGN_EVIDENCE", publication_detail, publication),
            ("KeyboardInterrupt", "bounded omission", omission),
        )
        for code, detail, receipt in cases:
            rows = deepcopy(base_rows)
            _rewrite_terminal_completion_chain(
                contract,
                capsule,
                authorization,
                rows,
                error_code=code,
                error_detail=detail,
                error_receipt=receipt,
            )
            verification = TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
            assert verification["status"] == (
                "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
            )

        prespawn_root = tmp_path / "prespawn"
        prespawn_root.mkdir()
        full_contract, full_capsule, full_authorization, full_ledger = (
            _full_real_gap_ledger_prefix(prespawn_root)
        )
        try:
            prespawn_rows = deepcopy(
                list(zip(full_ledger.published_names, full_ledger.published_records))[
                    :3
                ]
            )
            progress = TOOL._ledger_progress_snapshot_v2(
                [record for _name, record in prespawn_rows]
            )
            status = TOOL._error_code_terminal_disposition(
                full_contract, "FAIL_CAMPAIGN_DOCKER"
            )
            terminal = TOOL._campaign_terminal_body(
                full_contract,
                full_capsule,
                status_text=status,
                invocation_count=int(progress["supervisor_invocation_count"]),
                last_slot_id_or_null=progress["last_slot_id_or_null"],
                last_slot_outcome_digest_or_null=progress[
                    "last_slot_outcome_digest_or_null"
                ],
                error_code_or_null="FAIL_CAMPAIGN_DOCKER",
                error_detail_or_null=prespawn_detail,
                error_receipt_or_null=prespawn,
            )
            terminal_record = _append_test_ledger_record(
                full_contract,
                full_capsule,
                full_authorization,
                prespawn_rows,
                "campaign.terminal.json",
                "CAMPAIGN_TERMINAL",
                status,
                terminal,
            )
            completion = TOOL._campaign_completion_body(
                full_contract,
                full_capsule,
                authorization_digest=str(
                    full_authorization["authorization_digest"]
                ),
                terminal=terminal,
                terminal_ledger_record=terminal_record,
                preceding_ledger_names=[
                    name for name, _record in prespawn_rows
                ],
            )
            _append_test_ledger_record(
                full_contract,
                full_capsule,
                full_authorization,
                prespawn_rows,
                "campaign.completion.json",
                "CAMPAIGN_COMPLETION",
                "CAMPAIGN_COMPLETION_COMMITTED",
                completion,
            )
            assert TOOL._verify_campaign_completion_chain_v2(
                full_contract,
                full_capsule,
                full_authorization,
                prespawn_rows,
            )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
            wrong_slot_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                wrong_slot_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=prespawn_detail.replace(
                    "SLOT_01", "SLOT_02", 1
                ),
                error_receipt=prespawn,
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2):
                TOOL._verify_campaign_completion_chain_v2(
                    full_contract,
                    full_capsule,
                    full_authorization,
                    wrong_slot_rows,
                )

            outer_exception = TOOL._bounded_process_exception_diagnostic(
                "Docker read-only inventory",
                TOOL._docker_inventory_argv_v2(full_contract),
                RuntimeError("injected outer inventory failure"),
            )
            outer_error = TOOL._canonical_json_bytes(
                outer_exception
            ).decode("ascii").rstrip("\n")
            outer_receipt = {
                "schema_version": TOOL._PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
                "docker_rows": [],
                "docker_error_or_null": outer_error,
                "bounded_command_diagnostic_or_null": outer_exception,
            }
            outer_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                outer_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=(
                    "SLOT_01 pre-spawn Docker absence is uncertain: "
                    + outer_error
                ),
                error_receipt=outer_receipt,
            )
            assert TOOL._verify_campaign_completion_chain_v2(
                full_contract,
                full_capsule,
                full_authorization,
                outer_rows,
            )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"

            oversized_exception = TOOL._bounded_process_exception_diagnostic(
                "Docker read-only inventory",
                TOOL._docker_inventory_argv_v2(full_contract),
                TOOL.ShadowRetryCampaignErrorV2(
                    "FAIL_CAMPAIGN_PROCESS",
                    "injected oversized diagnostic",
                    receipt={"opaque_diagnostic": "x" * (20 * 1024)},
                ),
            )
            oversized_error = TOOL._canonical_json_bytes(
                oversized_exception
            ).decode("ascii").rstrip("\n")
            assert len(oversized_error.encode("ascii") + b"\n") > 16 * 1024
            omission_receipt = {
                "schema_version": TOOL._PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
                "docker_rows": [],
                "docker_error_or_null": oversized_error,
                "bounded_command_diagnostic_or_null": deepcopy(omission),
            }
            oversized_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                oversized_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=(
                    "SLOT_01 pre-spawn Docker absence is uncertain: "
                    + oversized_error
                )[:4096],
                error_receipt=omission_receipt,
            )
            assert TOOL._verify_campaign_completion_chain_v2(
                full_contract,
                full_capsule,
                full_authorization,
                oversized_rows,
            )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"

            forged_small_omission = deepcopy(prespawn)
            forged_small_omission[
                "bounded_command_diagnostic_or_null"
            ] = deepcopy(omission)
            forged_small_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                forged_small_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=prespawn_detail,
                error_receipt=forged_small_omission,
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2):
                TOOL._verify_campaign_completion_chain_v2(
                    full_contract,
                    full_capsule,
                    full_authorization,
                    forged_small_rows,
                )
        finally:
            full_ledger.close()
    finally:
        ledger.close()


def test_offline_verifier_binds_supervisor_and_clone_receipts_to_active_slot(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object = (
        _full_real_gap_ledger_prefix(tmp_path)
    )
    ledger = ledger_object
    try:
        published = list(zip(ledger.published_names, ledger.published_records))
        slots = contract["slot_registry"]
        assert type(slots) is list
        slot_01_argv = TOOL._slot_argv(contract, slots[0])
        spawn = {
            "schema_version": TOOL._SPAWN_REGISTRATION_SCHEMA_V2,
            "phase": "SPAWN_REGISTRATION",
            "argv": deepcopy(slot_01_argv),
            "registration_state": "VALID",
            "registration_raw_hex": b"123\n".hex(),
            "registered_pid_or_null": 123,
            "popen_pid_or_null": 123,
            "popen_child_created": True,
            "pid_match_or_null": True,
            "constructor_error_or_null": {
                "error_code": "OSError",
                "error_detail": "injected post-registration constructor error",
                "error_terminal": False,
                "error_receipt_or_null": None,
            },
            "cleanup_target_pid_or_null": 123,
            "cleanup_kill_attempted": True,
            "cleanup_direct_child_reaped": True,
            "cleanup_process_group_absent": True,
            "cleanup_identity_consistent": True,
            "cleanup_errors": [],
            "cleanup_certain": True,
            "later_ledger_write_permitted": True,
        }
        spawn_detail = (
            "registered Popen did not cross its commit boundary: "
            "registration=VALID; pid_match=True; restore_errors=[]; "
            "error=OSError:injected post-registration constructor error"
        )
        supervisor_receipt = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-exception-"
                "diagnostic/2"
            ),
            "label": "SLOT_01 bound supervisor",
            "argv": deepcopy(slot_01_argv),
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "FAIL_CAMPAIGN_PROCESS",
            "error_detail": spawn_detail,
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": spawn,
        }
        supervisor_detail = (
            "SLOT_01 bound supervisor bounded process raised before a result: "
            "FAIL_CAMPAIGN_PROCESS:" + spawn_detail
        )

        slot_01_consumed_rows = deepcopy(published[:4])
        _terminalize_test_prefix_with_error(
            contract,
            capsule,
            authorization,
            slot_01_consumed_rows,
            error_code="FAIL_CAMPAIGN_PROCESS",
            error_detail=supervisor_detail,
            error_receipt=supervisor_receipt,
        )
        assert TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            slot_01_consumed_rows,
        )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"

        slot_02_consumed_rows = deepcopy(published[:8])
        _terminalize_test_prefix_with_error(
            contract,
            capsule,
            authorization,
            slot_02_consumed_rows,
            error_code="FAIL_CAMPAIGN_PROCESS",
            error_detail=supervisor_detail,
            error_receipt=supervisor_receipt,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as wrong_supervisor:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                slot_02_consumed_rows,
            )
        assert wrong_supervisor.value.code == "FAIL_CAMPAIGN_LEDGER"

        nested_wrong_slot = deepcopy(supervisor_receipt)
        nested_wrong_slot["label"] = "SLOT_02 bound supervisor"
        nested_wrong_slot["argv"] = TOOL._slot_argv(contract, slots[1])
        nested_wrong_slot_detail = (
            "SLOT_02 bound supervisor bounded process raised before a result: "
            "FAIL_CAMPAIGN_PROCESS:" + spawn_detail
        )
        nested_wrong_rows = deepcopy(published[:8])
        _terminalize_test_prefix_with_error(
            contract,
            capsule,
            authorization,
            nested_wrong_rows,
            error_code="FAIL_CAMPAIGN_PROCESS",
            error_detail=nested_wrong_slot_detail,
            error_receipt=nested_wrong_slot,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as wrong_nested:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                nested_wrong_rows,
            )
        assert wrong_nested.value.code == "FAIL_CAMPAIGN_LEDGER"

        git = str(contract["execution_policy"]["git_executable"])
        clone_argv = [
            git,
            "-C",
            str(slots[0]["clone_root"]),
            "rev-parse",
            "HEAD",
        ]
        clone_receipt = TOOL._bounded_process_diagnostic(
            "slot clone HEAD replay",
            clone_argv,
            {
                "return_code": 7,
                "timed_out": False,
                "stdout_overflow": False,
                "stderr_overflow": False,
                "child_reaped": True,
                "process_group_absent": True,
                "wrapper_resources_closed": True,
                "stdout": b"",
                "stderr": b"injected clone replay failure\n",
            },
            classification={
                "require_empty_stdout": False,
                "require_empty_stderr": True,
                "accepted": False,
            },
        )
        clone_detail = (
            "slot clone HEAD replay failed closed: return_code=7; "
            "timed_out=False; stdout_overflow=False; stderr_overflow=False; "
            f"stderr_excerpt_hex={clone_receipt['stderr']['excerpt_hex']}"
        )
        slot_01_intent_rows = deepcopy(published[:3])
        _terminalize_test_prefix_with_error(
            contract,
            capsule,
            authorization,
            slot_01_intent_rows,
            error_code="FAIL_CAMPAIGN_SOURCE",
            error_detail=clone_detail,
            error_receipt=clone_receipt,
        )
        assert TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            slot_01_intent_rows,
        )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"

        slot_02_intent_rows = deepcopy(published[:7])
        _terminalize_test_prefix_with_error(
            contract,
            capsule,
            authorization,
            slot_02_intent_rows,
            error_code="FAIL_CAMPAIGN_SOURCE",
            error_detail=clone_detail,
            error_receipt=clone_receipt,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as wrong_clone:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                slot_02_intent_rows,
            )
        assert wrong_clone.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_safe_docker_inventory_normalizes_outer_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()

    def raise_outer(_contract: Mapping[str, object]) -> object:
        raise RuntimeError("injected outer Docker inventory failure")

    monkeypatch.setattr(TOOL, "_docker_inventory", raise_outer)
    absent, rows, error = TOOL._safe_docker_inventory(contract)
    assert absent is False
    assert rows == []
    assert type(error) is str
    receipt = TOOL._strict_json(
        error.encode("ascii") + b"\n",
        "outer Docker inventory exception KAT",
        16 * 1024,
    )
    assert receipt["schema_version"] == (
        TOOL._BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2
    )
    assert receipt["label"] == "Docker read-only inventory"
    assert receipt["argv"] == TOOL._docker_inventory_argv_v2(contract)
    assert receipt["phase"] == "BOUNDED_PROCESS_RAISED_BEFORE_RESULT"
    assert receipt["error_code"] == "RuntimeError"
    assert receipt["error_detail"] == (
        "injected outer Docker inventory failure"
    )


def test_offline_verifier_rejects_rehashed_spawn_uncertain_completion(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        inner_detail = "injected missing spawn registration"
        receipt = {
            "schema_version": TOOL._BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2,
            "label": "SLOT_01 bound supervisor",
            "argv": TOOL._slot_argv(contract, contract["slot_registry"][0]),
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            "error_detail": inner_detail,
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": {
                "schema_version": TOOL._SPAWN_REGISTRATION_SCHEMA_V2,
                "phase": "SPAWN_REGISTRATION",
                "registration_state": "PROCESS_OBJECT_MISSING",
                "later_ledger_write_permitted": False,
            },
        }
        detail = (
            "SLOT_01 bound supervisor bounded process raised before a result: "
            "FAIL_CAMPAIGN_SPAWN_UNCERTAIN:" + inner_detail
        )
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        _rewrite_terminal_completion_chain(
            contract,
            capsule,
            authorization,
            rows,
            error_code="FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            error_detail=detail,
            error_receipt=receipt,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_real_legal_success_chain_is_authoritative_and_progress_is_frozen(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, result = (
        _completed_real_success_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        assert result["status"] == contract["success_policy"]["terminal_status"]
        assert result["supervisor_invocation_count"] == 1
        assert result["last_slot_id_or_null"] == "SLOT_01"
        verification = TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            list(zip(ledger.published_names, ledger.published_records)),
        )
        assert verification["status"] == (
            "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
        )
    finally:
        ledger.close()


@pytest.mark.parametrize("prefix_length", (2, 3, 4, 6, 7, 8, 10, 11, 12, 14))
def test_offline_verifier_accepts_each_legal_error_prefix(
    tmp_path: Path,
    prefix_length: int,
) -> None:
    contract, capsule, authorization, ledger_object = (
        _full_real_gap_ledger_prefix(tmp_path)
    )
    ledger = ledger_object
    try:
        records = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))[
                :prefix_length
            ]
        )
        progress = TOOL._ledger_progress_snapshot_v2(
            [record for _name, record in records]
        )
        status_text = contract["terminal_failure_policy"]["crash_disposition"]
        terminal = TOOL._campaign_terminal_body(
            contract,
            capsule,
            status_text=status_text,
            invocation_count=int(progress["supervisor_invocation_count"]),
            last_slot_id_or_null=progress["last_slot_id_or_null"],
            last_slot_outcome_digest_or_null=progress[
                "last_slot_outcome_digest_or_null"
            ],
            error_code_or_null="KeyboardInterrupt",
            error_detail_or_null=(
                f"injected legal terminal at prefix {prefix_length}"
            ),
        )
        terminal_record = _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            records,
            "campaign.terminal.json",
            "CAMPAIGN_TERMINAL",
            status_text,
            terminal,
        )
        completion = TOOL._campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(authorization["authorization_digest"]),
            terminal=terminal,
            terminal_ledger_record=terminal_record,
            preceding_ledger_names=[name for name, _record in records],
        )
        _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            records,
            "campaign.completion.json",
            "CAMPAIGN_COMPLETION",
            "CAMPAIGN_COMPLETION_COMMITTED",
            completion,
        )
        verification = TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            records,
        )
        assert verification["status"] == (
            "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
        )
    finally:
        ledger.close()


def test_exact_scheduling_exhaustion_after_intent_is_not_authoritative(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object = (
        _full_real_gap_ledger_prefix(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))[:3]
        )
        assert [name for name, _record in rows] == list(
            TOOL._LEDGER_RECORD_NAMES_V2[:3]
        )
        progress = TOOL._ledger_progress_snapshot_v2(
            [record for _name, record in rows]
        )
        assert progress["supervisor_invocation_count"] == 0
        error_code = TOOL._SCHEDULING_ADMISSION_FAILURE_V2
        status_text = TOOL._error_code_terminal_disposition(
            contract,
            error_code,
        )
        clock = _AdmissionClock()
        policy = TOOL._scheduling_admission_policy_v2(contract)

        def failed_burst(
            _fd: int,
            slot_id: str,
            window: int,
            burst: int,
            _policy: Mapping[str, object],
            **_kwargs: object,
        ) -> dict[str, object]:
            result = _admitted_scheduling_burst(
                policy,
                slot_id,
                window,
                burst,
                start_ns=clock.now_ns,
                critical_ns=25_000_001,
            )
            clock.now_ns = int(result["burst_finish_ns"])
            return result

        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as generated:
            TOOL._run_scheduling_admission_gate_v2(
                contract,
                7,
                "SLOT_01",
                clock_ns=clock,
                sleeper=clock.sleep,
                burst_runner=failed_burst,
            )
        receipt = generated.value.receipt
        assert type(receipt) is dict and receipt["slot_id"] == "SLOT_01"
        terminal = TOOL._campaign_terminal_body(
            contract,
            capsule,
            status_text=status_text,
            invocation_count=0,
            last_slot_id_or_null=None,
            last_slot_outcome_digest_or_null=None,
            error_code_or_null=error_code,
            error_detail_or_null=generated.value.detail,
            error_receipt_or_null=receipt,
        )
        terminal_record = _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            rows,
            "campaign.terminal.json",
            "CAMPAIGN_TERMINAL",
            status_text,
            terminal,
        )
        completion = TOOL._campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(authorization["authorization_digest"]),
            terminal=terminal,
            terminal_ledger_record=terminal_record,
            preceding_ledger_names=[name for name, _record in rows],
        )
        _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            rows,
            "campaign.completion.json",
            "CAMPAIGN_COMPLETION",
            "CAMPAIGN_COMPLETION_COMMITTED",
            completion,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as rejected:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert rejected.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_offline_verifier_rejects_rehashed_wrong_slot_admission_receipt(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object = (
        _full_real_gap_ledger_prefix(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))[:7]
        )
        assert [name for name, _record in rows] == list(
            TOOL._LEDGER_RECORD_NAMES_V2[:7]
        )
        assert rows[-1][0] == "slot-02.intent.json"

        clock = _AdmissionClock()
        policy = TOOL._scheduling_admission_policy_v2(contract)

        def failed_burst(
            _fd: int,
            slot_id: str,
            window: int,
            burst: int,
            _policy: Mapping[str, object],
            **_kwargs: object,
        ) -> dict[str, object]:
            result = _admitted_scheduling_burst(
                policy,
                slot_id,
                window,
                burst,
                start_ns=clock.now_ns,
                critical_ns=25_000_001,
            )
            clock.now_ns = int(result["burst_finish_ns"])
            return result

        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as generated:
            TOOL._run_scheduling_admission_gate_v2(
                contract,
                7,
                "SLOT_01",
                clock_ns=clock,
                sleeper=clock.sleep,
                burst_runner=failed_burst,
            )
        wrong_slot_receipt = generated.value.receipt
        assert type(wrong_slot_receipt) is dict
        assert wrong_slot_receipt["slot_id"] == "SLOT_01"
        assert TOOL._scheduling_admission_evidence_semantics_v6(
            contract,
            wrong_slot_receipt,
            require_passed=False,
        )

        progress = TOOL._ledger_progress_snapshot_v2(
            [record for _name, record in rows]
        )
        error_code = TOOL._SCHEDULING_ADMISSION_FAILURE_V2
        status_text = TOOL._error_code_terminal_disposition(
            contract,
            error_code,
        )
        terminal = TOOL._campaign_terminal_body(
            contract,
            capsule,
            status_text=status_text,
            invocation_count=int(progress["supervisor_invocation_count"]),
            last_slot_id_or_null=progress["last_slot_id_or_null"],
            last_slot_outcome_digest_or_null=progress[
                "last_slot_outcome_digest_or_null"
            ],
            error_code_or_null=error_code,
            error_detail_or_null=generated.value.detail,
            error_receipt_or_null=wrong_slot_receipt,
        )
        assert terminal["error_receipt_or_null"]["slot_id"] == "SLOT_01"
        terminal_record = _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            rows,
            "campaign.terminal.json",
            "CAMPAIGN_TERMINAL",
            status_text,
            terminal,
        )
        completion = TOOL._campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(authorization["authorization_digest"]),
            terminal=terminal,
            terminal_ledger_record=terminal_record,
            preceding_ledger_names=[name for name, _record in rows],
        )
        _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            rows,
            "campaign.completion.json",
            "CAMPAIGN_COMPLETION",
            "CAMPAIGN_COMPLETION_COMMITTED",
            completion,
        )

        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as rejected:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert rejected.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert rejected.value.detail == (
            "ledger semantic state differs: terminal error disposition differs"
        )
    finally:
        ledger.close()


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("window_ordinal", True),
        ("quiet_duration_ns", 30_000_000_000.0),
        ("maximum_burst_drain_ns", 200_000_000.0),
    ),
)
def test_offline_verifier_rejects_rehashed_scheduling_integer_type_forgery(
    tmp_path: Path,
    field: str,
    forged_value: object,
) -> None:
    contract, capsule, authorization, ledger_object = (
        _full_real_gap_ledger_prefix(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))[:4]
        )
        consumed_record = rows[-1][1]
        consumed_body = consumed_record["body"]
        assert type(consumed_body) is dict
        admission = consumed_body["scheduling_admission"]
        assert type(admission) is dict
        windows = admission["windows_in_strict_order"]
        assert type(windows) is list and type(windows[0]) is dict
        windows[0][field] = forged_value
        consumed_unsigned = deepcopy(consumed_record)
        consumed_unsigned.pop("record_digest")
        consumed_record["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["ledger_record"],
            consumed_unsigned,
        )
        progress = TOOL._ledger_progress_snapshot_v2(
            [record for _name, record in rows]
        )
        status_text = contract["terminal_failure_policy"]["crash_disposition"]
        terminal = TOOL._campaign_terminal_body(
            contract,
            capsule,
            status_text=status_text,
            invocation_count=int(progress["supervisor_invocation_count"]),
            last_slot_id_or_null=progress["last_slot_id_or_null"],
            last_slot_outcome_digest_or_null=progress[
                "last_slot_outcome_digest_or_null"
            ],
            error_code_or_null="KeyboardInterrupt",
            error_detail_or_null="injected rehashed admission type forgery",
        )
        terminal_record = _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            rows,
            "campaign.terminal.json",
            "CAMPAIGN_TERMINAL",
            status_text,
            terminal,
        )
        completion = TOOL._campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(authorization["authorization_digest"]),
            terminal=terminal,
            terminal_ledger_record=terminal_record,
            preceding_ledger_names=[name for name, _record in rows],
        )
        _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            rows,
            "campaign.completion.json",
            "CAMPAIGN_COMPLETION",
            "CAMPAIGN_COMPLETION_COMMITTED",
            completion,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "consumed mapping" in error.value.detail
    finally:
        ledger.close()


def test_offline_verifier_rejects_rehashed_semantic_progress_forgery(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_success_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        terminal_record = rows[-2][1]
        terminal = terminal_record["body"]
        assert type(terminal) is dict
        terminal["supervisor_invocation_count"] = 0
        terminal_unsigned = deepcopy(terminal)
        terminal_unsigned.pop("record_digest")
        terminal["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["campaign_terminal"],
            terminal_unsigned,
        )
        terminal_record_unsigned = deepcopy(terminal_record)
        terminal_record_unsigned.pop("record_digest")
        terminal_record["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["ledger_record"],
            terminal_record_unsigned,
        )
        completion_record = rows[-1][1]
        completion_record["previous_record_digest_or_null"] = terminal_record[
            "record_digest"
        ]
        completion = TOOL._campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(authorization["authorization_digest"]),
            terminal=terminal,
            terminal_ledger_record=terminal_record,
            preceding_ledger_names=[name for name, _record in rows[:-1]],
        )
        completion_record["body"] = completion
        completion_record_unsigned = deepcopy(completion_record)
        completion_record_unsigned.pop("record_digest")
        completion_record["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["ledger_record"],
            completion_record_unsigned,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "terminal progress" in error.value.detail
    finally:
        ledger.close()


@pytest.mark.parametrize("corruption", ("missing", "semantic-link", "order"))
def test_offline_completion_verifier_rejects_non_authoritative_chain(
    tmp_path: Path,
    corruption: str,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        if corruption == "missing":
            rows = rows[:-1]
        elif corruption == "order":
            rows[-2], rows[-1] = rows[-1], rows[-2]
        else:
            completion_record = rows[-1][1]
            completion_body = completion_record["body"]
            assert type(completion_body) is dict
            completion_body["terminal_body_digest"] = _tagged("wrong-terminal")
            completion_unsigned = deepcopy(completion_body)
            completion_unsigned.pop("record_digest")
            completion_body["record_digest"] = TOOL._domain_digest(
                contract["hash_domains"]["campaign_completion"],
                completion_unsigned,
            )
            completion_record_unsigned = deepcopy(completion_record)
            completion_record_unsigned.pop("record_digest")
            completion_record["record_digest"] = TOOL._domain_digest(
                contract["hash_domains"]["ledger_record"],
                completion_record_unsigned,
            )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_completion_record_cannot_be_prepared_without_terminal_predecessor(
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    ledger = TOOL._LedgerV2(
        contract,
        _capsule_stub(),
        {"authorization_digest": _tagged("authorization")},
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            ledger.prepare(
                "campaign.completion.json",
                "CAMPAIGN_COMPLETION",
                "CAMPAIGN_COMPLETION_COMMITTED",
                {"record_digest": _tagged("completion")},
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "immediately follow terminal" in error.value.detail
    finally:
        ledger.close()


def test_verified_completion_makes_session_close_faults_nonrevoking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _contract_value, _capsule, _authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    root_fd = ledger.root_fd
    parent_fd = ledger.parent_fd
    real_flock = TOOL.fcntl.flock
    real_close = TOOL.os.close
    unlock_attempted = False
    close_attempts: list[int] = []

    def fail_unlock(descriptor: int, operation: int) -> None:
        nonlocal unlock_attempted
        if descriptor == root_fd and operation == TOOL.fcntl.LOCK_UN:
            unlock_attempted = True
            raise OSError("injected post-completion unlock failure")
        real_flock(descriptor, operation)

    def fail_session_close(descriptor: int) -> None:
        if descriptor in {root_fd, parent_fd}:
            close_attempts.append(descriptor)
            raise OSError("injected post-completion descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(TOOL.fcntl, "flock", fail_unlock)
    monkeypatch.setattr(TOOL.os, "close", fail_session_close)
    try:
        ledger.close()
        assert unlock_attempted is True
        assert close_attempts == [root_fd, parent_fd]
        assert ledger.completion_commit_verified is True
        assert ledger.root_fd is None
        assert ledger.parent_fd is None
    finally:
        real_close(root_fd)
        real_close(parent_fd)


def test_completion_publication_fault_forbids_any_later_ledger_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    capsule = _capsule_stub(contract=contract)
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    real_publish = TOOL._publish_immutable_at

    def publish_then_fault(
        directory_fd: int,
        basename: str,
        value_or_bytes: Mapping[str, object] | bytes,
        **kwargs: object,
    ) -> bytes:
        payload = real_publish(
            directory_fd,
            basename,
            value_or_bytes,
            **kwargs,
        )
        if basename == "campaign.completion.json":
            raise OSError("injected post-helper completion fault")
        return payload

    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_fault)
    try:
        result = TOOL._publish_terminal(
            ledger,
            contract,
            capsule,
            status_text=contract["terminal_failure_policy"][
                "crash_disposition"
            ],
            invocation_count=0,
            last_slot_id_or_null=None,
            last_slot_outcome_digest_or_null=None,
            error_code_or_null="KeyboardInterrupt",
            error_detail_or_null="injected pre-slot terminal",
        )
        assert result["completion_commit_verified"] is True
        assert ledger.published_names.count("campaign.terminal.json") == 1
        assert ledger.published_names.count("campaign.completion.json") == 1
        assert ledger.completion_commit_verified is True
        root = Path(str(contract["fixed_external_roots"]["ledger_root"]))
        assert (root / "campaign.completion.json").is_file()
    finally:
        ledger.close()


def test_ledger_precreate_failure_does_not_advance_chain_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    ledger_root = tmp_path / "ledger-root"
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        ledger_root
    )
    capsule = _capsule_stub()
    authorization = {"authorization_digest": _tagged("authorization")}
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                TOOL,
                "_publish_immutable_at",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    MemoryError("injected before O_CREAT")
                ),
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                ledger.publish(
                    "campaign.intent.json",
                    "CAMPAIGN_INTENT",
                    "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
                    {"test": True},
                )
            assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert ledger.sequence == 0
        assert ledger.previous_record_digest is None
        assert ledger.published_names == []
        assert ledger.last_name_index == -1
        assert not ledger_root.joinpath("campaign.intent.json").exists()
    finally:
        ledger.close()


@pytest.mark.parametrize(
    ("decisions", "expected_calls", "status_key"),
    [
        ((CORE.SUCCESS_V2,), 1, "success"),
        ((CORE.RETRY_EXACT_GAP_V2, CORE.SUCCESS_V2), 2, "success"),
        (
            (
                CORE.RETRY_EXACT_GAP_V2,
                CORE.RETRY_EXACT_GAP_V2,
                CORE.EXHAUSTED_RETRYABLE_GAP_V2,
            ),
            3,
            "exhausted",
        ),
        ((CORE.TERMINAL_NONRETRYABLE_V2,), 1, "terminal"),
        (
            (
                CORE.RETRY_EXACT_GAP_V2,
                CORE.RETRY_EXACT_GAP_V2,
                CORE.RETRY_EXACT_GAP_V2,
            ),
            3,
            "crash",
        ),
    ],
)
def test_run_state_machine_is_finite_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    decisions: tuple[str, ...],
    expected_calls: int,
    status_key: str,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = str(tmp_path / "ledger")
    capsule = _capsule_stub()
    authorization = {"authorization_digest": _tagged("authorization")}
    slots = [
        {"slot_id": f"SLOT_{index:02d}", "slot_ordinal": index}
        for index in range(1, 4)
    ]
    context = {
        "capsule": capsule,
        "slots": slots,
        "freshness_parent_identities": {},
    }
    ledger = _FakeLedger()
    calls: list[str] = []

    def run_one_slot(
        _contract_value: Mapping[str, object],
        _context: Mapping[str, object],
        _capsule: Mapping[str, object],
        _ledger: object,
        slot: Mapping[str, object],
        invocation_counter: list[int],
        _run_entry_state: dict[str, object],
    ) -> tuple[dict[str, object], str]:
        index = len(calls)
        assert index < len(decisions)
        calls.append(str(slot["slot_id"]))
        _ledger.publish(
            f"slot-{index + 1:02d}.consumed.json",
            "SLOT_CONSUMED",
            f"{slot['slot_id']}_CONSUMED_OUTCOME_UNKNOWN",
            {"slot_id": slot["slot_id"]},
        )
        invocation_counter[0] += 1
        outcome = {
            "slot_id": slot["slot_id"],
            "record_digest": _tagged(f"outcome-{index + 1}"),
        }
        _ledger.publish(
            f"slot-{index + 1:02d}.outcome.json",
            "SLOT_OUTCOME",
            decisions[index],
            outcome,
        )
        return (
            outcome,
            decisions[index],
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", run_one_slot)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)

    result = TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
        capsule["capsule_digest"]
    )
    expected_status = {
        "success": contract["success_policy"]["terminal_status"],
        "exhausted": contract["terminal_failure_policy"][
            "slot_03_exact_gap_disposition"
        ],
        "terminal": contract["terminal_failure_policy"][
            "generic_nonretry_disposition"
        ],
        "crash": contract["terminal_failure_policy"]["crash_disposition"],
    }[status_key]
    assert result["status"] == expected_status
    assert result["supervisor_invocation_count"] == expected_calls
    assert calls == [f"SLOT_{index:02d}" for index in range(1, expected_calls + 1)]
    assert len(calls) <= 3
    assert ledger.published_names[:2] == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert ledger.published_names[-2:] == [
        "campaign.terminal.json",
        "campaign.completion.json",
    ]
    assert ledger.published_names.count("campaign.terminal.json") == 1
    assert ledger.published_names.count("campaign.completion.json") == 1
    assert result["completion_commit_verified"] is True
    assert ledger.closed is True


def test_publication_uncertainty_forbids_terminal_second_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    capsule = _capsule_stub()
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
        "freshness_parent_identities": {},
    }
    ledger = _FakeLedger()

    def uncertain_slot(*_args: object, **_kwargs: object) -> object:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
            "injected consumed publication uncertainty",
            terminal=True,
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", uncertain_slot)

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert ledger.published_names == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert "campaign.terminal.json" not in ledger.published_names
    assert ledger.closed is True


def test_spawn_uncertainty_forbids_terminal_completion_or_later_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    capsule = _capsule_stub()
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
        "freshness_parent_identities": {},
    }
    ledger = _FakeLedger()

    def uncertain_spawn(*_args: object, **_kwargs: object) -> object:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            "injected missing registration receipt",
            terminal=True,
            receipt={"cleanup_certain": False},
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", uncertain_spawn)

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
    assert ledger.published_names == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert "campaign.terminal.json" not in ledger.published_names
    assert "campaign.completion.json" not in ledger.published_names
    assert ledger.closed is True


def test_terminal_publication_uncertainty_supersedes_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub()
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
        "freshness_parent_identities": {},
    }

    class TerminalUncertainLedger(_FakeLedger):
        def publish(
            self,
            basename: str,
            record_type: str,
            status_text: str,
            body: Mapping[str, object],
        ) -> dict[str, object]:
            if basename == "campaign.terminal.json":
                raise TOOL.ShadowRetryCampaignErrorV2(
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                    "injected terminal fsync uncertainty",
                    terminal=True,
                )
            return super().publish(basename, record_type, status_text, body)

    ledger = TerminalUncertainLedger()

    def source_failure(*_args: object, **_kwargs: object) -> object:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_SOURCE",
            "original source error",
            terminal=True,
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", source_failure)

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "original source error" not in error.value.detail
    assert "injected terminal fsync uncertainty" in error.value.detail
    assert type(error.value.receipt) is dict
    assert error.value.receipt["schema_version"] == (
        "hegel-phase3a-q05b-shadow-retry-terminalization-failure/2"
    )
    assert error.value.receipt["original_error"]["error_code"] == (
        "FAIL_CAMPAIGN_SOURCE"
    )
    assert error.value.receipt["terminalization_error"]["error_code"] == (
        "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    )
    assert len(TOOL._canonical_json_bytes(error.value.receipt)) <= (
        TOOL._MAX_ERROR_RECEIPT_BYTES
    )
    assert ledger.published_names == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert ledger.closed is True


def test_result_construction_failure_after_terminal_staging_leaves_no_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub()
    ledger = _FakeLedger()

    def fail_result(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise MemoryError("injected completion result construction failure")

    monkeypatch.setattr(
        TOOL,
        "_campaign_result_from_prepared_completion",
        fail_result,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._publish_terminal(
            ledger,  # type: ignore[arg-type]
            contract,
            capsule,
            status_text="SUCCESS",
            invocation_count=1,
            last_slot_id_or_null="SLOT_01",
            last_slot_outcome_digest_or_null=_tagged("outcome"),
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "completion result construction failure" in error.value.detail
    assert ledger.published_names == ["campaign.terminal.json"]
    assert ledger.completion_commit_verified is False


def test_cli_result_is_fully_built_before_completion_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_built = False
    real_builder = TOOL._campaign_result_from_prepared_completion

    class GuardedLedger(_FakeLedger):
        def publish_prepared(
            self,
            basename: str,
            prepared_record: Mapping[str, object],
        ) -> dict[str, object]:
            if basename == "campaign.completion.json":
                assert result_built is True
            return super().publish_prepared(basename, prepared_record)

    def build_then_mark(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal result_built
        result = real_builder(*args, **kwargs)
        result_built = True
        return result

    monkeypatch.setattr(
        TOOL,
        "_campaign_result_from_prepared_completion",
        build_then_mark,
    )
    ledger = GuardedLedger()
    result = TOOL._publish_terminal(
        ledger,  # type: ignore[arg-type]
        _contract(),
        _capsule_stub(),
        status_text="SUCCESS",
        invocation_count=1,
        last_slot_id_or_null="SLOT_01",
        last_slot_outcome_digest_or_null=_tagged("outcome"),
    )
    assert result_built is True
    assert result["completion_commit_verified"] is True
    assert ledger.published_names == [
        "campaign.terminal.json",
        "campaign.completion.json",
    ]


def test_verified_completion_supersedes_postcommit_async_fault() -> None:
    class PostcommitAsyncLedger(_FakeLedger):
        def publish_prepared(
            self,
            basename: str,
            prepared_record: Mapping[str, object],
        ) -> dict[str, object]:
            published = super().publish_prepared(basename, prepared_record)
            if basename == "campaign.completion.json":
                assert self.completion_commit_verified is True
                raise KeyboardInterrupt("injected after verified completion")
            return published

    ledger = PostcommitAsyncLedger()
    result = TOOL._publish_terminal(
        ledger,  # type: ignore[arg-type]
        _contract(),
        _capsule_stub(),
        status_text="SUCCESS",
        invocation_count=1,
        last_slot_id_or_null="SLOT_01",
        last_slot_outcome_digest_or_null=_tagged("outcome"),
    )
    assert result["status"] == "SUCCESS"
    assert result["completion_commit_verified"] is True
    assert ledger.completion_commit_verified is True
    assert ledger.published_names[-1] == "campaign.completion.json"


def test_completion_store_attr_gap_recovers_from_offline_verified_disk(
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "opcode-gap-ledger"
    )
    capsule = _capsule_stub(contract=contract)
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    callback_code = next(
        constant
        for constant in TOOL._LedgerV2.publish_prepared.__code__.co_consts
        if type(constant) is type(TOOL._LedgerV2.publish_prepared.__code__)
        and constant.co_name == "commit_verified_publication"
    )
    store_offset = next(
        instruction.offset
        for instruction in dis.get_instructions(callback_code)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "authoritative_completion_result_or_null"
    )
    injected = False

    def tracer(frame: object, event: str, _arg: object) -> object:
        nonlocal injected
        if getattr(frame, "f_code", None) is callback_code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif (
                event == "opcode"
                and frame.f_lasti == store_offset
                and not injected
            ):
                injected = True
                raise KeyboardInterrupt(
                    "injected before authoritative STORE_ATTR"
                )
        return tracer

    sys.settrace(tracer)
    try:
        result = TOOL._publish_terminal(
            ledger,
            contract,
            capsule,
            status_text=contract["terminal_failure_policy"][
                "crash_disposition"
            ],
            invocation_count=0,
            last_slot_id_or_null=None,
            last_slot_outcome_digest_or_null=None,
            error_code_or_null="KeyboardInterrupt",
            error_detail_or_null="legal pre-slot opcode-gap terminal",
        )
    finally:
        sys.settrace(None)
    try:
        assert injected is True
        assert result["completion_commit_verified"] is True
        assert ledger.completion_commit_verified is True
        assert ledger.published_names.count("campaign.terminal.json") == 1
        assert Path(
            str(contract["fixed_external_roots"]["ledger_root"])
        ).joinpath("campaign.completion.json").is_file()
    finally:
        ledger.close()


def test_run_recovers_verified_completion_across_caller_return_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "run-return-gap-ledger"
    )
    capsule = _capsule_stub()
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
        "freshness_parent_identities": {},
    }
    ledger = _FakeLedger()

    def successful_slot(
        _contract_value: Mapping[str, object],
        _context: Mapping[str, object],
        _capsule: Mapping[str, object],
        ledger_value: _FakeLedger,
        slot: Mapping[str, object],
        invocation_counter: list[int],
        _run_entry_state: dict[str, object],
    ) -> tuple[dict[str, object], str]:
        ledger_value.publish(
            "slot-01.consumed.json",
            "SLOT_CONSUMED",
            "SLOT_01_CONSUMED_OUTCOME_UNKNOWN",
            {"slot_id": "SLOT_01"},
        )
        invocation_counter[0] += 1
        outcome = {
            "slot_id": slot["slot_id"],
            "record_digest": _tagged("run-return-outcome"),
        }
        ledger_value.publish(
            "slot-01.outcome.json",
            "SLOT_OUTCOME",
            CORE.SUCCESS_V2,
            outcome,
        )
        return outcome, CORE.SUCCESS_V2

    real_terminal = TOOL._publish_terminal

    def terminal_then_return_fault(*args: object, **kwargs: object) -> object:
        real_terminal(*args, **kwargs)
        raise KeyboardInterrupt("injected caller RETURN-to-STORE gap")

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", successful_slot)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    monkeypatch.setattr(TOOL, "_publish_terminal", terminal_then_return_fault)
    result = TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
        capsule["capsule_digest"]
    )
    assert result["completion_commit_verified"] is True
    assert ledger.published_names.count("campaign.terminal.json") == 1
    assert ledger.published_names.count("campaign.completion.json") == 1


def test_bounded_command_failure_receipt_is_durable_in_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    argv = [
        "/usr/bin/python3.10",
        "-I",
        "-S",
        "-B",
        "-c",
        "import sys;sys.stderr.write('exact clone reason\\n');sys.exit(7)",
    ]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as command_error:
        TOOL._run_exact_command(
            argv,
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            maximum_bytes=4096,
            label="injected clone command",
        )
    captured = command_error.value
    assert captured.code == "FAIL_CAMPAIGN_PROCESS"
    assert type(captured.receipt) is dict
    assert captured.receipt["argv"] == argv
    assert captured.receipt["return_code"] == 7
    assert bytes.fromhex(
        str(captured.receipt["stderr"]["excerpt_hex"])
    ) == b"exact clone reason\n"

    contract = _contract()
    capsule = _capsule_stub()
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
        "freshness_parent_identities": {},
    }
    ledger = _FakeLedger()
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract, **_kwargs: context)
    monkeypatch.setattr(
        TOOL,
        "_verify_fresh_targets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(
        TOOL,
        "_run_one_slot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(captured),
    )
    result = TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
        capsule["capsule_digest"]
    )
    assert result["error_receipt_or_null"] == captured.receipt
    terminal_body = ledger.rows[-2][3]
    assert terminal_body["error_receipt_or_null"] == captured.receipt
    assert terminal_body["error_receipt_or_null"]["stderr"][
        "raw_sha256"
    ] == sha256(b"exact clone reason\n").hexdigest()


def _registered_certain_git_failure(
    contract: Mapping[str, object],
) -> tuple[list[str], str, dict[str, object], str]:
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    arguments = [
        "-C",
        str(execution["clone_source_repository"]),
        "rev-parse",
        f"{target['target_supervisor_commit']}^{{tree}}",
    ]
    label = "target tree replay"
    argv = [str(execution["git_executable"]), *arguments]
    receipt = TOOL._bounded_process_diagnostic(
        label,
        argv,
        {
            "return_code": 7,
            "timed_out": False,
            "stdout_overflow": False,
            "stderr_overflow": False,
            "child_reaped": True,
            "process_group_absent": True,
            "wrapper_resources_closed": True,
            "stdout": b"",
            "stderr": b"fatal: exact Git source failure\n",
        },
        classification={
            "require_empty_stdout": False,
            "require_empty_stderr": True,
            "accepted": False,
        },
    )
    detail = (
        f"{label} failed closed: return_code=7; timed_out=False; "
        "stdout_overflow=False; stderr_overflow=False; "
        f"stderr_excerpt_hex={receipt['stderr']['excerpt_hex']}"
    )
    return arguments, label, receipt, detail


def test_certain_registered_git_failure_is_source_not_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    arguments, label, receipt, detail = _registered_certain_git_failure(
        contract
    )
    failure = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_PROCESS",
        detail,
        terminal=True,
        receipt=receipt,
    )

    def fail_exact_command(*_args: object, **_kwargs: object) -> bytes:
        raise failure

    monkeypatch.setattr(TOOL, "_run_exact_command", fail_exact_command)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._git(contract, arguments, label)
    captured = error.value
    assert captured.code == "FAIL_CAMPAIGN_SOURCE"
    assert captured.detail == detail
    assert captured.receipt == receipt
    assert captured.terminal is True
    assert TOOL._error_code_terminal_disposition(
        contract,
        captured.code,
    ) == contract["terminal_failure_policy"][
        "source_or_binding_uncertainty_disposition"
    ]
    assert TOOL._terminal_error_receipt_semantics_v2(
        contract,
        receipt,
        terminal_error_code="FAIL_CAMPAIGN_SOURCE",
        terminal_error_detail=detail,
        top_level=True,
    )
    assert not TOOL._terminal_error_receipt_semantics_v2(
        contract,
        receipt,
        terminal_error_code="FAIL_CAMPAIGN_PROCESS",
        terminal_error_detail=detail,
        top_level=True,
    )


@pytest.mark.parametrize(
    "uncertainty",
    (
        "spawn",
        "exception",
        "close",
        "cleanup",
        "timeout",
        "overflow",
        "unregistered",
    ),
)
def test_git_never_reclassifies_process_or_receipt_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    uncertainty: str,
) -> None:
    contract = _contract()
    arguments, label, base_receipt, detail = _registered_certain_git_failure(
        contract
    )
    receipt = deepcopy(base_receipt)
    if uncertainty == "spawn":
        receipt = {
            "schema_version": TOOL._SPAWN_REGISTRATION_SCHEMA_V2,
            "registration_state": "INVALID",
        }
    elif uncertainty == "exception":
        receipt["schema_version"] = (
            TOOL._BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2
        )
    elif uncertainty == "close":
        receipt["wrapper_resources_closed"] = False
        receipt["classification"] = {
            "accepted": False,
            "close_errors": ["injected close uncertainty"],
            "prior_error_or_null": None,
        }
    elif uncertainty == "cleanup":
        receipt["child_reaped"] = False
    elif uncertainty == "timeout":
        receipt["timed_out"] = True
    elif uncertainty == "overflow":
        receipt["stderr_overflow"] = True
    else:
        receipt["label"] = "unregistered Git command"
    failure = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_PROCESS",
        detail,
        terminal=True,
        receipt=receipt,
    )

    def fail_exact_command(*_args: object, **_kwargs: object) -> bytes:
        raise failure

    monkeypatch.setattr(TOOL, "_run_exact_command", fail_exact_command)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._git(contract, arguments, label)
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert error.value.receipt == receipt


def test_bounded_process_exception_receipts_preserve_command_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def selector_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError("injected selector setup failure")

    monkeypatch.setattr(TOOL, "_bounded_process", selector_failure)
    argv = ["/usr/bin/git", "status", "--porcelain=v1"]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as command_error:
        TOOL._run_exact_command(
            argv,
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            maximum_bytes=4096,
            label="post-run Git replay",
        )
    assert command_error.value.code == "FAIL_CAMPAIGN_PROCESS"
    receipt = command_error.value.receipt
    assert type(receipt) is dict
    assert receipt["label"] == "post-run Git replay"
    assert receipt["argv"] == argv
    assert receipt["phase"] == "BOUNDED_PROCESS_RAISED_BEFORE_RESULT"
    assert receipt["error_code"] == "OSError"
    assert receipt["stdout"] == {"available": False}
    assert "injected selector setup failure" in str(receipt["error_detail"])

    absent, rows, docker_error = TOOL._docker_inventory(_contract())
    assert absent is False
    assert rows == []
    assert type(docker_error) is str
    docker_receipt = TOOL._strict_json(
        docker_error.encode("ascii") + b"\n",
        "Docker exception receipt KAT",
        64 * 1024,
    )
    assert docker_receipt["label"] == "Docker read-only inventory"
    assert docker_receipt["argv"][2] == "ps"
    assert docker_receipt["phase"] == "BOUNDED_PROCESS_RAISED_BEFORE_RESULT"
    assert "injected selector setup failure" in docker_receipt["error_detail"]


def test_bounded_process_never_signals_reused_group_after_child_reap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kill_calls: list[int] = []

    def forbidden_postreap_kill(
        process: object,
        *,
        first_signal: int,
        grace_seconds: float,
        reap_seconds: float,
    ) -> bool:
        del first_signal, grace_seconds, reap_seconds
        kill_calls.append(int(getattr(process, "pid")))
        raise AssertionError("numeric PGID must not be signalled after reap")

    monkeypatch.setattr(TOOL, "_process_group_present", lambda _pid: True)
    monkeypatch.setattr(TOOL, "_kill_and_reap", forbidden_postreap_kill)
    result = TOOL._bounded_process(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            "-c",
            "pass",
        ],
        cwd=os.fspath(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert result["return_code"] == 0
    assert result["child_reaped"] is True
    assert result["process_group_absent"] is False
    assert result["wrapper_resources_closed"] is True
    assert kill_calls == []


def test_selector_setup_failure_reaps_live_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cleanup_rows: list[tuple[int, bool, int | None]] = []
    real_kill_and_reap = TOOL._kill_and_reap

    def tracked_kill_and_reap(
        process: object,
        *,
        first_signal: int,
        grace_seconds: float,
        reap_seconds: float,
    ) -> bool:
        reaped = real_kill_and_reap(
            process,
            first_signal=first_signal,
            grace_seconds=grace_seconds,
            reap_seconds=reap_seconds,
        )
        cleanup_rows.append(
            (int(getattr(process, "pid")), reaped, process.poll())
        )
        return reaped

    def fail_selector_setup() -> object:
        raise OSError("injected selector construction failure")

    monkeypatch.setattr(TOOL, "_kill_and_reap", tracked_kill_and_reap)
    monkeypatch.setattr(TOOL.selectors, "DefaultSelector", fail_selector_setup)
    with pytest.raises(OSError, match="selector construction failure"):
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10",
                "-I",
                "-S",
                "-B",
                "-c",
                "import time;time.sleep(60)",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
        )
    assert len(cleanup_rows) == 1
    assert cleanup_rows[0][1] is True
    assert cleanup_rows[0][2] is not None


def test_selector_close_failure_is_typed_and_retains_process_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_factory = TOOL.selectors.DefaultSelector

    class CloseFailSelector:
        def __init__(self) -> None:
            self.delegate = real_factory()

        def __getattr__(self, name: str) -> object:
            return getattr(self.delegate, name)

        def close(self) -> None:
            self.delegate.close()
            raise OSError("injected selector close failure")

    monkeypatch.setattr(TOOL.selectors, "DefaultSelector", CloseFailSelector)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10",
                "-I",
                "-S",
                "-B",
                "-c",
                "pass",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
        )
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "injected selector close failure" in error.value.detail
    assert type(error.value.receipt) is dict
    assert error.value.receipt["return_code"] == 0
    assert error.value.receipt["child_reaped"] is True
    assert error.value.receipt["wrapper_resources_closed"] is False
    assert error.value.receipt["classification"]["accepted"] is False


def test_safe_postrun_source_replay_preserves_bounded_command_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-bounded-command-diagnostic/2"
        ),
        "argv": ["/usr/bin/git", "status", "--porcelain=v1"],
        "return_code": 7,
        "stderr": {
            "byte_count": 19,
            "raw_sha256": sha256(b"post-run git failed").hexdigest(),
            "excerpt_hex": b"post-run git failed".hex(),
            "excerpt_byte_count": 19,
            "excerpt_truncated": False,
        },
    }
    failure = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_PROCESS",
        "post-run source Git replay failed closed",
        terminal=True,
        receipt=receipt,
    )

    def fail_replay(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(TOOL, "_source_state_replay", fail_replay)
    result = TOOL._safe_source_state_replay(
        {},
        {},
        {},
        artifact_presence="ABSENT",
    )
    assert result["certain"] is False
    assert result["error_or_null"] == (
        "FAIL_CAMPAIGN_PROCESS:post-run source Git replay failed closed"
    )
    assert result["evidence"]["error_receipt_or_null"] == receipt
    assert result["evidence"]["error_receipt_or_null"]["argv"] == receipt["argv"]
    assert result["root"] == TOOL._domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R20/SOURCE_REPLAY/V2",
        result["evidence"],
    )


def test_source_replay_reuses_prepared_clone_then_verifies_once_postrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    light_target = {
        "commit": "12" * 20,
        "local_object_closure_fsck_passed": False,
    }
    admitted_target = deepcopy(light_target)
    admitted_target["local_object_closure_fsck_passed"] = True
    controller = {"commit": "34" * 20, "clean": True}
    context = {
        "target_git": admitted_target,
        "controller_git": controller,
        "cargo_rows": [],
        "cargo_identity": {
            "manifest_digest": _tagged("cargo-source"),
            "content_manifest_digest": _tagged("cargo-content"),
        },
    }
    prepared_clone = {
        "artifact_state": "ABSENT",
        "clone_closure_rows_internal": [],
    }
    verify_clone_calls: list[str] = []

    monkeypatch.setattr(
        TOOL,
        "_verify_target_git_objects",
        lambda _contract: deepcopy(light_target),
    )
    monkeypatch.setattr(
        TOOL,
        "_controller_git_identity",
        lambda _contract: deepcopy(controller),
    )

    def scan(_path: Path, *, label: str, **_kwargs: object):
        if "Cargo source" in label:
            return ({"manifest_digest": _tagged("cargo-source")}, [])
        assert "Cargo cache" in label
        return ({"content_manifest_digest": _tagged("cargo-content")}, [])

    def verify_clone(
        _contract: Mapping[str, object],
        _slot: Mapping[str, object],
        *,
        artifact_state: str,
    ) -> dict[str, object]:
        verify_clone_calls.append(artifact_state)
        return {
            "artifact_state": artifact_state,
            "clone_closure_rows_internal": [],
        }

    monkeypatch.setattr(TOOL, "_scan_tree", scan)
    monkeypatch.setattr(TOOL, "_verify_clone", verify_clone)
    monkeypatch.setattr(
        TOOL,
        "_target_object_closure_fsck_admission_v2r17",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("slot replay repeated target full fsck")
        ),
    )
    before = TOOL._source_state_replay(
        contract,
        context,
        slot,
        artifact_presence="ABSENT",
        verified_clone=prepared_clone,
    )
    assert before["certain"] is True
    assert verify_clone_calls == []
    after = TOOL._safe_source_state_replay(
        contract,
        context,
        slot,
        artifact_presence="PRESENT",
    )
    assert after["certain"] is True
    assert verify_clone_calls == ["PRESENT"]


def test_run_cli_exact_fsck_timeout_returns_exit75_with_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    contract = _contract()
    digest = _tagged("pending-run-capsule")
    pending = _target_fsck_pending_error(contract)
    calls = {
        "preflight": 0,
        "authorization_lookup": 0,
        "ledger": 0,
        "slot": 0,
        "popen": 0,
    }

    def preflight(
        _contract: Mapping[str, object],
        *,
        expected_capsule_digest_or_null: str | None = None,
        allow_existing_run_state: bool = False,
    ) -> dict[str, object]:
        calls["preflight"] += 1
        assert expected_capsule_digest_or_null == digest
        assert allow_existing_run_state is True
        raise pending

    def counted(name: str):
        def invoke(*_args: object, **_kwargs: object) -> object:
            calls[name] += 1
            raise AssertionError(f"pending reached {name}")

        return invoke

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _value: None)
    monkeypatch.setattr(TOOL, "_preflight_context", preflight)
    monkeypatch.setattr(TOOL, "_load_authorization", counted("authorization_lookup"))
    monkeypatch.setattr(TOOL, "_LedgerV2", counted("ledger"))
    monkeypatch.setattr(TOOL, "_run_one_slot", counted("slot"))
    monkeypatch.setattr(TOOL, "_bounded_process", counted("popen"))

    assert TOOL.main(["--run", "--capsule-digest", digest]) == 75
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    result = TOOL._strict_json(captured.out, "pending run CLI")
    assert result["schema_version"] == TOOL._RUN_VALIDATION_PENDING_SCHEMA_V2
    assert result["status"] == "CAMPAIGN_RUN_VALIDATION_PENDING"
    assert result["entrypoint"] == "RUN"
    assert result["phase"] == "TARGET_LOCAL_OBJECT_CLOSURE_FSCK"
    assert result["condition"] == "READONLY_COMMAND_TIMEOUT"
    assert result["expected_capsule_digest"] == digest
    assert result["capsule_derivation_completed"] is True
    assert result["capsule_digest_validation_completed"] is True
    assert result["python310_dependency_gate_invocation_count_this_entry"] == 1
    assert result["target_closure_fsck_invocation_count_this_entry"] == 1
    assert result[
        "host_cpu_topology_affinity_admission_invocation_count_this_entry"
    ] == 0
    for field in (
        "authorization_lookup_count_this_entry",
        "authorization_consumption_count_this_entry",
        "ledger_lookup_or_lock_count_this_entry",
        "ledger_creation_or_write_count_this_entry",
        "slot_state_creation_or_write_count_this_entry",
        "supervisor_popen_count_this_entry",
        "docker_daemon_contact_count_this_entry",
        "network_contact_count_this_entry",
    ):
        assert result[field] == 0
    assert result["prior_durable_campaign_state"] == "UNINSPECTED_UNKNOWN"
    assert result["prior_authorization_consumption_count_or_null"] is None
    assert result["prior_ledger_record_count_or_null"] is None
    assert result["prior_supervisor_invocation_count_or_null"] is None
    assert result["does_not_claim_zero_prior_durable_campaign_state"] is True
    assert result["pending_occurrence_count_in_this_process"] == 1
    assert result["resumable"] is True
    assert result["resume_same_entrypoint"] is True
    assert result["same_capsule_required"] is True
    assert result["fresh_complete_entry_replay_required"] is True
    assert calls == {
        "preflight": 1,
        "authorization_lookup": 0,
        "ledger": 0,
        "slot": 0,
        "popen": 0,
    }


def test_run_cli_exact_host_cpu_pending_returns_exit75_before_authority_or_state(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    contract = _contract()
    digest = _tagged("host-cpu-pending-run-capsule")
    monkeypatch.setattr(TOOL.os, "sched_getaffinity", lambda _pid: set(range(31)))
    monkeypatch.setattr(
        TOOL.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unreachable")),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as raised:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    pending = raised.value
    calls = {"authorization": 0, "ledger": 0, "slot": 0, "popen": 0}

    def forbidden(name: str):
        def invoke(*_args: object, **_kwargs: object) -> object:
            calls[name] += 1
            raise AssertionError(f"host CPU pending reached {name}")

        return invoke

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _value: None)
    monkeypatch.setattr(
        TOOL,
        "_preflight_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(pending),
    )
    monkeypatch.setattr(TOOL, "_load_authorization", forbidden("authorization"))
    monkeypatch.setattr(TOOL, "_LedgerV2", forbidden("ledger"))
    monkeypatch.setattr(TOOL, "_run_one_slot", forbidden("slot"))
    monkeypatch.setattr(TOOL, "_bounded_process", forbidden("popen"))

    assert TOOL.main(["--run", "--capsule-digest", digest]) == 75
    result = TOOL._strict_json(
        capsysbinary.readouterr().out,
        "host CPU pending run CLI",
    )
    assert result["schema_version"] == TOOL._RUN_VALIDATION_PENDING_SCHEMA_V2
    assert result["phase"] == TOOL._HOST_CPU_ADMISSION_PHASE_V2R17
    assert result["condition"] == "REQUIRED_EFFECTIVE_CPU_SET_UNAVAILABLE"
    assert result["condition_receipt"] == pending.receipt
    assert result["python310_dependency_gate_invocation_count_this_entry"] == 1
    assert result["target_closure_fsck_invocation_count_this_entry"] == 1
    assert result[
        "host_cpu_topology_affinity_admission_invocation_count_this_entry"
    ] == 1
    for field in (
        "authorization_lookup_count_this_entry",
        "authorization_consumption_count_this_entry",
        "ledger_lookup_or_lock_count_this_entry",
        "ledger_creation_or_write_count_this_entry",
        "slot_state_creation_or_write_count_this_entry",
        "supervisor_popen_count_this_entry",
        "docker_daemon_contact_count_this_entry",
        "network_contact_count_this_entry",
    ):
        assert result[field] == 0
    assert calls == {"authorization": 0, "ledger": 0, "slot": 0, "popen": 0}


def test_record_authorization_host_cpu_pending_has_no_publication_or_state(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    contract = _contract()
    digest = _tagged("host-cpu-pending-record-capsule")
    monkeypatch.setattr(TOOL.os, "sched_getaffinity", lambda _pid: set(range(31)))
    monkeypatch.setattr(
        TOOL.os,
        "sched_setaffinity",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unreachable")),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1) as raised:
        TOOL._host_cpu_topology_affinity_admission_v2r17(
            contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    pending = raised.value
    publication_count = 0

    def forbidden_publication(*_args: object, **_kwargs: object) -> bytes:
        nonlocal publication_count
        publication_count += 1
        raise AssertionError("host CPU pending reached authorization publication")

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _value: None)
    monkeypatch.setattr(
        TOOL,
        "_preflight_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(pending),
    )
    monkeypatch.setattr(TOOL, "_publish_immutable_at", forbidden_publication)

    assert TOOL.main(
        ["--record-authorization", "--capsule-digest", digest]
    ) == 75
    result = TOOL._strict_json(
        capsysbinary.readouterr().out,
        "host CPU pending record-authorization CLI",
    )
    assert result["schema_version"] == TOOL._PRECONSUMPTION_PENDING_SCHEMA_V2
    assert result["entrypoint"] == "RECORD_AUTHORIZATION"
    assert result["condition_receipt"] == pending.receipt
    assert result["target_closure_fsck_invocation_count"] == 1
    assert result["host_cpu_topology_affinity_admission_invocation_count"] == 1
    assert result["authorization_lookup_count"] == 0
    assert result["authorization_publication_count"] == 0
    assert result["ledger_state_creation_count"] == 0
    assert result["slot_state_creation_count"] == 0
    assert result["supervisor_popen_count"] == 0
    assert publication_count == 0


def test_preflight_pending_owner_authorization_fields_are_unambiguous(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    contract = _contract()
    pending = _target_fsck_pending_error(contract)
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _value: None)
    monkeypatch.setattr(
        TOOL,
        "_preflight_context",
        lambda _contract, **_kwargs: (_ for _ in ()).throw(pending),
    )
    assert TOOL.main(["--preflight"]) == 75
    result = TOOL._strict_json(
        capsysbinary.readouterr().out,
        "pending preflight CLI",
    )
    assert result["entrypoint"] == "PREFLIGHT"
    assert result["expected_capsule_digest_or_null"] is None
    assert result["capsule_derivation_completed"] is True
    assert result["capsule_digest_validation_completed"] is False
    assert result["same_capsule_required"] is False
    assert result["new_owner_authorization_required_for_pending_reentry"] is False
    assert result[
        "execution_owner_authorization_required_after_successful_preflight"
    ] is True


def test_same_run_entrypoint_and_capsule_replay_after_pending_before_auth_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = _capsule_stub("same-run-reentry")
    pending = _target_fsck_pending_error(contract)
    preflight_expected: list[str | None] = []
    authorization_lookup_count = 0

    def preflight(
        _contract: Mapping[str, object],
        *,
        expected_capsule_digest_or_null: str | None = None,
        allow_existing_run_state: bool = False,
    ) -> dict[str, object]:
        preflight_expected.append(expected_capsule_digest_or_null)
        assert allow_existing_run_state is True
        if len(preflight_expected) == 1:
            raise pending
        return {
            "capsule": deepcopy(capsule),
            "slots": deepcopy(contract["slot_registry"]),
        }

    def authorization_lookup(
        _contract: Mapping[str, object],
        _capsule: Mapping[str, object],
    ) -> dict[str, object]:
        nonlocal authorization_lookup_count
        authorization_lookup_count += 1
        raise TOOL.ShadowRetryCampaignErrorV2(
            "REACHED_AUTHORIZATION_LOOKUP_AFTER_REENTRY",
            "test stop",
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", preflight)
    monkeypatch.setattr(TOOL, "_load_authorization", authorization_lookup)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1):
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(capsule["capsule_digest"])
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as reached:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(capsule["capsule_digest"])
    assert reached.value.code == "REACHED_AUTHORIZATION_LOOKUP_AFTER_REENTRY"
    assert preflight_expected == [capsule["capsule_digest"]] * 2
    assert authorization_lookup_count == 1


def test_record_authorization_pending_then_same_digest_publishes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    root = tmp_path / "authorization"
    contract["fixed_external_roots"]["authorization_root"] = os.fspath(root)
    contract["authorization_policy"]["authorization_root"] = os.fspath(root)
    capsule = _capsule_stub("same-record-reentry", contract=contract)
    authorization = TOOL._authorization_template(contract, capsule)
    pending = _target_fsck_pending_error(contract)
    preflight_count = 0
    publication_count = 0
    real_publish = TOOL._publish_immutable_at

    def preflight(
        _contract: Mapping[str, object],
        *,
        expected_capsule_digest_or_null: str | None = None,
    ) -> dict[str, object]:
        nonlocal preflight_count
        preflight_count += 1
        assert expected_capsule_digest_or_null == capsule["capsule_digest"]
        if preflight_count == 1:
            raise pending
        return {
            "capsule": deepcopy(capsule),
            "authorization_template": deepcopy(authorization),
        }

    def publish(*args: object, **kwargs: object) -> bytes:
        nonlocal publication_count
        publication_count += 1
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", preflight)
    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish)
    with pytest.raises(TOOL.ShadowRetryCampaignPendingV1):
        TOOL.create_phase3_q05b_shadow_retry_authorization_v2(
            capsule["capsule_digest"]
        )
    assert not root.exists()
    receipt = TOOL.create_phase3_q05b_shadow_retry_authorization_v2(
        capsule["capsule_digest"]
    )
    assert receipt["status"] == "AUTHORIZED_EXACT_CAPSULE_ONCE"
    path = root / str(contract["authorization_policy"]["authorization_record_name"])
    assert path.is_file()
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2):
        TOOL.create_phase3_q05b_shadow_retry_authorization_v2(
            capsule["capsule_digest"]
        )
    assert publication_count == 1
    assert path.read_bytes() == TOOL._canonical_json_bytes(authorization)


@pytest.mark.parametrize(
    ("updates", "expected_policy_field"),
    (
        (
            {"trajectory_status": "INVALID"},
            "trajectory_incomplete_or_unverifiable_disposition",
        ),
        (
            {"cleanup_certain": False},
            "cleanup_or_survivor_uncertainty_disposition",
        ),
        (
            {"source_binding_certain": False},
            "source_or_binding_uncertainty_disposition",
        ),
    ),
)
def test_terminal_disposition_uses_explicit_observation_class(
    updates: Mapping[str, object],
    expected_policy_field: str,
) -> None:
    contract = _contract()
    observation: dict[str, object] = {
        "trajectory_status": "COMPLETE_FAIL",
        "trajectory_torn_tail_or_null": None,
        "trajectory_open_span_ids": [],
        "trajectory_trace_id": "trace",
        "expected_trace_id": "trace",
        "trajectory_producer_id": "producer",
        "expected_producer_id": "producer",
        "cleanup_certain": True,
        "docker_absence_certain": True,
        "supervisor_reaped": True,
        "process_group_absent": True,
        "clone_clean_certain": True,
        "cargo_isolation_certain": True,
        "source_binding_certain": True,
        "external_binding_certain": True,
        "replay_certain": True,
    }
    observation.update(updates)
    assert TOOL._observation_terminal_disposition(
        contract,
        observation,
    ) == contract["terminal_failure_policy"][expected_policy_field]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--pre"],
        ["--preflight", "--run"],
        ["--preauthorization-live-gauntlet", "--run"],
        ["--run", "--unknown"],
    ],
)
def test_cli_parser_rejects_missing_ambiguous_or_abbreviated_modes(
    arguments: list[str],
) -> None:
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._parser().parse_args(arguments)
    assert error.value.code == "FAIL_CAMPAIGN_ARGUMENT"


def test_cli_runtime_rejects_nonisolated_test_interpreter() -> None:
    contract = _contract()
    contract["execution_policy"]["supervisor_interpreter"] = sys.executable
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._require_exact_cli_runtime(contract)
    assert error.value.code == "FAIL_CAMPAIGN_RUNTIME"


def test_cli_preflight_rejects_capsule_argument_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    contract = _contract()
    called = False

    def forbidden_preflight() -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _contract: None)
    monkeypatch.setattr(
        TOOL,
        "preflight_phase3_q05b_shadow_retry_campaign_v2",
        forbidden_preflight,
    )
    assert TOOL.main(
        ["--preflight", "--capsule-digest", _tagged("capsule")]
    ) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    value = TOOL._strict_json(captured.out, "CLI error")
    assert type(value) is dict
    assert value["status"] == "CAMPAIGN_COMMAND_FAILED_CLOSED"
    assert value["error_code"] == "FAIL_CAMPAIGN_ARGUMENT"
    assert value["resumable"] is False
    assert called is False


def test_cli_preauthorization_live_gauntlet_dispatch_and_capsule_rejection(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    contract = _contract()
    calls = 0
    expected = {
        "schema_version": (
            "hegel-phase3a-q05b-preauthorization-live-admission-"
            "gauntlet-command-result/2"
        ),
        "status": "PASS",
    }

    def create_gauntlet() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return deepcopy(expected)

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _value: None)
    monkeypatch.setattr(
        TOOL,
        "create_preauthorization_live_gauntlet_receipt_v2",
        create_gauntlet,
    )
    assert TOOL.main(["--preauthorization-live-gauntlet"]) == 0
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert TOOL._strict_json(captured.out, "gauntlet CLI result") == expected
    assert calls == 1

    assert TOOL.main([
        "--preauthorization-live-gauntlet",
        "--capsule-digest",
        _tagged("forbidden-gauntlet-capsule"),
    ]) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    failed = TOOL._strict_json(captured.out, "gauntlet CLI failure")
    assert type(failed) is dict
    assert failed["error_code"] == "FAIL_CAMPAIGN_ARGUMENT"
    assert failed["resumable"] is False
    assert calls == 1

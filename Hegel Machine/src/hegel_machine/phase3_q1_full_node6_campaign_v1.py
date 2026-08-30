"""Pure wire and state transitions for the full-node6 diagnostic campaign."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Final, Mapping, NoReturn


CONTRACT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-campaign-contract/6"
)
CONTRACT_ID: Final = "hegel-phase3a-q1-full-node6-campaign-successor-v6"
CAPSULE_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-capsule/1"
PENDING_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-pending/1"
LEDGER_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-ledger-record/3"
OUTCOME_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-outcome/2"

PREFLIGHT_ADMITTED: Final = "FULL_NODE6_PREFLIGHT_ADMITTED_NOT_CONSUMED"
PENDING_NOT_CONSUMED: Final = "PENDING_FULL_NODE6_NOT_CONSUMED"
CONSUMED: Final = "FULL_NODE6_AUTHORIZATION_CONSUMED"
SATURATED: Final = "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY"
GUARD_HIT: Final = "PREFLIGHT_CAPACITY_GUARD_HIT"
RESOURCE_INFEASIBLE: Final = "PREFLIGHT_RESOURCE_ENVELOPE_INFEASIBLE"
FAILED_TERMINAL: Final = "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME"
OUTCOME_UNKNOWN_TERMINAL: Final = (
    "FULL_NODE6_CONSUMED_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"
)

OWNER_AUTHORIZATION_TEXT: Final = (
    "授权 Q1 full-node6 capacity-preflight diagnostic "
    "successor-v6 distinct-outer actual；确认 successor-v5 known terminal "
    "failure、terminal no-resume，且 Python Odd/Sink 与 Rust Odd/Sink "
    "measurement 各一次 conservatively count，trusted host replay "
    "一次 conservatively count；另行授权 exact v6 capsule-only additional "
    "fresh remeasurement；禁止恢复 successor-v5 或复用 successor-v5 work product"
)
OWNER_AUTHORIZATION_BODY: Final = {
    "schema_version": "hegel-phase3a-q1-full-node6-owner-authorization/6",
    "authority_role": "PROJECT_OWNER",
    "authorization_text": OWNER_AUTHORIZATION_TEXT,
    "superseded_v5_resume_allowed": False,
    "distinct_fresh_successor_v6_logical_campaign_only": True,
    "future_exact_user_authorization_required": True,
    "prior_measurement_outcome_known_terminal_failure_acknowledged": True,
    "prior_python_odd_measurement_count": 1,
    "prior_python_sink_measurement_count": 1,
    "prior_rust_odd_measurement_count": 1,
    "prior_rust_sink_measurement_count": 1,
    "prior_trusted_host_replay_measurement_count": 1,
    "new_measurement_budget_scope": "V6_CAPSULE_ONLY",
    "new_measurement_is_explicitly_additional": True,
    "v5_work_product_reuse_allowed": False,
    "logical_campaign_count": 1,
    "maximum_supervisor_popen_count": 1,
    "maximum_quotient_measurement_count_per_endpoint_signature": 1,
    "same_capsule_preconsumption_pending_replay_allowed": True,
    "pending_replay_is_campaign_execution": False,
    "pending_replay_consumes_authorization": False,
    "measurement_outcome_unknown_terminal_no_resume": True,
    "q1_start_allowed": False,
    "network_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "current_pointer_write_allowed": False,
}

DOMAINS: Final = {
    "authorization": b"HEGEL/Q1/FULL_NODE6/AUTHORIZATION/SUCCESSOR_V6/V1\x00",
    "capsule": b"HEGEL/Q1/FULL_NODE6/CAPSULE/V1\x00",
    "contract": b"HEGEL/Q1/FULL_NODE6/CONTRACT/SUCCESSOR_V6/V1\x00",
    "ledger": b"HEGEL/Q1/FULL_NODE6/LEDGER/V1\x00",
    "outcome": b"HEGEL/Q1/FULL_NODE6/OUTCOME/V1\x00",
    "pending": b"HEGEL/Q1/FULL_NODE6/PENDING/V1\x00",
}

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class FullNode6CampaignValidationError(ValueError):
    pass


def _fail(detail: str) -> NoReturn:
    raise FullNode6CampaignValidationError(detail)


def canonical_json_bytes_v1(value: object) -> bytes:
    def finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            _fail("non-finite JSON is forbidden")
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                _fail("JSON object keys must be strings")
            for child in item.values():
                finite(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                finite(child)

    finite(value)
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        _fail(f"canonical JSON rejected: {error}")


def domain_digest_v1(domain: bytes, value: object) -> str:
    if type(domain) is not bytes or not domain:
        _fail("digest domain must be nonempty bytes")
    return "sha256:" + sha256(domain + canonical_json_bytes_v1(value)).hexdigest()


OWNER_AUTHORIZATION_DIGEST: Final = domain_digest_v1(
    DOMAINS["authorization"], OWNER_AUTHORIZATION_BODY
)
OWNER_AUTHORIZATION: Final = {
    **OWNER_AUTHORIZATION_BODY,
    "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
}


def _object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    return deepcopy(value)


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _fail(f"{label} must be tagged sha256")
    return value


def _commit(value: object, label: str) -> str:
    if type(value) is not str or _COMMIT.fullmatch(value) is None:
        _fail(f"{label} must be a full lowercase commit")
    return value


def validate_contract_v1(value: object) -> dict[str, Any]:
    contract = _object(value, "contract")
    required = {
        "schema_version",
        "contract_id",
        "status",
        "contract_digest",
        "authorization",
        "source_lineage",
        "predecessor",
        "superseded_campaign",
        "target",
        "fixed_paths",
        "preconsumption_gate_order",
        "execution_policy",
        "state_machine",
        "authority",
    }
    if set(contract) != required:
        _fail("contract field registry differs")
    observed = _digest(contract["contract_digest"], "contract_digest")
    unsigned = deepcopy(contract)
    unsigned.pop("contract_digest")
    if (
        contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["contract_id"] != CONTRACT_ID
        or contract["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or contract["authorization"] != OWNER_AUTHORIZATION
        or domain_digest_v1(DOMAINS["contract"], unsigned) != observed
    ):
        _fail("contract header, authorization or digest differs")
    lineage = _object(contract["source_lineage"], "source_lineage")
    _commit(lineage.get("required_ancestor_commit"), "required_ancestor_commit")
    _commit(lineage.get("required_parent_commit"), "required_parent_commit")
    if (
        lineage.get("project_root")
        != "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine"
        or lineage.get("required_ancestor_commit")
        != "33dc9cbb7d753f83650169d6d6b9caa508927764"
        or lineage.get("required_parent_commit")
        != "3086adf389b8d9fa1a93c0c36fd300ed8bca19db"
        or lineage.get("clean_head_required") is not True
        or lineage.get("exact_additions_and_modifications_only") is not True
        or type(lineage.get("additive_paths_in_exact_order")) is not list
        or not lineage["additive_paths_in_exact_order"]
        or "docs/Hegel_Machine_Current_State_Reconciliation_v6.md"
        not in lineage["additive_paths_in_exact_order"]
        or "docs/Hegel_Machine_Current_State_Reconciliation_v7.md"
        not in lineage["additive_paths_in_exact_order"]
        or lineage.get("modified_paths_in_exact_order")
        != [
            "src/hegel_machine/phase3_q1_capacity_preflight_v1.py",
            "src/hegel_machine/phase3_q1_partition_snapshot_v1.py",
            "tests/test_phase3_q1_capacity_preflight_v1.py",
            "tests/test_phase3_q1_partition_snapshot_v1.py",
        ]
    ):
        _fail("source lineage differs")
    predecessor = _object(contract["predecessor"], "predecessor")
    for name in (
        "capsule_digest",
        "terminal_ledger_record_digest",
        "artifact_sha256",
    ):
        _digest(predecessor.get(name), f"predecessor.{name}")
    if (
        predecessor.get("terminal_status")
        != "ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN"
        or predecessor.get("qualification_count") != 20
        or predecessor.get("qualification_mask") != 0xFFFFF
        or predecessor.get("q1_state") != "NOT_RUN"
    ):
        _fail("predecessor terminal boundary differs")
    superseded = _object(
        contract["superseded_campaign"],
        "superseded_campaign",
    )
    expected_superseded = {
        "capsule_digest": (
            "sha256:6819416eed0a41573f21b4280a6a39e97481b87f8dd1c9e0054529e478bc7d9d"
        ),
        "source_commit": "3086adf389b8d9fa1a93c0c36fd300ed8bca19db",
        "source_tree": "9cbdbee241e709296d40f9cf7b66fe26d038c67e",
        "source_closure_digest": (
            "sha256:bd87e0af65e864fabd07af58e4ba552def3325d0a4493a1cade16f7a3a7c252d"
        ),
        "contract_digest": (
            "sha256:23e8631c3633c24c230a32c92bb02b067553f35c0afef1ab19b4448f9a435008"
        ),
        "authorization_schema_version": (
            "hegel-phase3a-q1-full-node6-owner-authorization/5"
        ),
        "authorization_text": (
            "授权 Q1 full-node6 capacity-preflight diagnostic successor-v5 "
            "distinct-outer actual；确认 successor-v4 known terminal failure、"
            "terminal no-resume，且 Python Odd measurement 与 Rust Odd "
            "measurement 各一次 conservatively count，Sink/trusted host replay 0；"
            "另行授权 exact v5 capsule-only additional fresh remeasurement；"
            "禁止恢复 successor-v4 或复用 successor-v4 work product"
        ),
        "authorization_digest": (
            "sha256:d55d5cbde77f58385ed6decbf5756f2042a4cf3819281819ded853554356da8d"
        ),
        "consumed_ledger_path": (
            "/home/erzhu419/.local/state/hegel-machine/"
            "q1-full-node6-capacity-v5-ledger/campaign.consumed.json"
        ),
        "consumed_ledger_record_digest": (
            "sha256:98945a148bec0a3a356f997891d625990be40f16539990e4abac77daab395f4d"
        ),
        "consumed_ledger_file_sha256": (
            "sha256:43ee028f57b93b9b166005bfe640b4aa46090ddb5e0a1240a7162b2b6e1e7d4d"
        ),
        "terminal_ledger_path": (
            "/home/erzhu419/.local/state/hegel-machine/"
            "q1-full-node6-capacity-v5-ledger/campaign.terminal.json"
        ),
        "terminal_ledger_record_digest": (
            "sha256:d2cb047773d716f310991d9ead0580058ffd778c7ce3183bf2610867e8de2aa6"
        ),
        "terminal_ledger_file_sha256": (
            "sha256:2ee9dbe129bfa2e1f0ddeff1bfb95e97d81b3c093c64a3d7280aa563f7d9f5b9"
        ),
        "outcome_path": (
            "/home/erzhu419/.local/state/hegel-machine/"
            "q1-full-node6-capacity-v5-evidence/campaign.outcome.json"
        ),
        "outcome_file_sha256": (
            "sha256:3f08fa61d3901ee970f8b901300c7bc870d3863004bb2747d24ea92e947168d0"
        ),
        "outcome_digest": (
            "sha256:108f4dc6971499b9b736f899ffeb1d8b3003fd9b9d56c85c185da60f5206b43d"
        ),
        "supervisor_stdout_path": (
            "/home/erzhu419/.local/state/hegel-machine/"
            "q1-full-node6-capacity-v5-evidence/supervisor.stdout"
        ),
        "supervisor_stdout_file_sha256": (
            "sha256:474d16bf26e2576d76db979bbb926916467b3e2c8e64364b74af7274d7a85ebe"
        ),
        "supervisor_stderr_path": (
            "/home/erzhu419/.local/state/hegel-machine/"
            "q1-full-node6-capacity-v5-evidence/supervisor.stderr"
        ),
        "supervisor_stderr_file_sha256": (
            "sha256:8d898ef4222871214aa4141702e612aae7656ad7272c5eaa4c66d863a662f07c"
        ),
        "trajectory_event_log_path": (
            "/home/erzhu419/.local/state/hegel-machine/"
            "q1-full-node6-capacity-v5-trajectory/supervisor-trajectory-v1.jsonl"
        ),
        "trajectory_event_log_file_sha256": (
            "sha256:fcf6940289491851e6675d8854a7aa99938e4e0a501f2feef83dfddca9251e3f"
        ),
        "trajectory_terminal_path": (
            "/home/erzhu419/.local/state/hegel-machine/"
            "q1-full-node6-capacity-v5-trajectory/"
            "supervisor-trajectory-terminal-v1.json"
        ),
        "trajectory_terminal_file_sha256": (
            "sha256:30daf6dd4efd56bff9cdff560d04a85629acb1bb95c34d882bf031f9200bdeb9"
        ),
        "trajectory_event_count": 34,
        "error_code": "FULL_NODE6_HOST_REPLAY_FAILED",
        "error_detail": "trusted replay did not exit cleanly",
        "terminal_status": FAILED_TERMINAL,
        "measurement_outcome": "KNOWN_TERMINAL_FAILURE",
        "measurement_endpoint_start_count": 2,
        "python_endpoint_start_count": 1,
        "rust_endpoint_start_count": 1,
        "trusted_host_replay_start_count": 1,
        "python_odd_measurement_count": 1,
        "python_sink_measurement_count": 1,
        "rust_odd_measurement_count": 1,
        "rust_sink_measurement_count": 1,
        "trusted_host_replay_measurement_count": 1,
        "campaign_execution_count": 1,
        "authorization_consumption_count": 1,
        "supervisor_popen_count": 1,
        "supervisor_return_code": 1,
        "supervisor_replay_passed": False,
        "artifact_sha256_or_null": None,
        "postconsumption_error_or_null": None,
        "endpoint_results": {
            "python": {
                "return_code": 0,
                "stdout_path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/python.stdout"
                ),
                "stdout_file_sha256": (
                    "sha256:f200c5a466954c5f12d8182c28a4187b9031cddca45be42adcb839ed63451d9b"
                ),
                "stderr_path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/python.stderr"
                ),
                "stderr_file_sha256": (
                    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
                "stdout_summary": {
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-capacity-diagnostic/1"
                    ),
                    "actor_id": "PYTHON_ENDPOINT",
                    "capsule_digest": (
                        "sha256:6819416eed0a41573f21b4280a6a39e97481b87f8dd1c9e0054529e478bc7d9d"
                    ),
                    "source_commit": "3086adf389b8d9fa1a93c0c36fd300ed8bca19db",
                    "terminal_status": SATURATED,
                    "actual_preflight_peak_scratch_bytes": 40452166,
                    "actual_preflight_terminal_live_scratch_bytes": 0,
                    "diagnostic_only": True,
                    "q1_execution_started": False,
                    "q1_formal_roots": None,
                    "q1_gate_count": 0,
                    "q1_gate_mask": 0,
                    "q1_output_slots": [None] * 8,
                    "q1_receipt": None,
                    "q1_state": "NOT_RUN",
                    "q2_state": "NOT_RUN",
                    "role_evaluation_performed": False,
                    "split_accessed": False,
                    "target_truth_accessed": False,
                    "active_transition_allowed": False,
                },
            },
            "rust": {
                "return_code": 0,
                "stdout_path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/rust.stdout"
                ),
                "stdout_file_sha256": (
                    "sha256:b667b13b281b832dc78c2e13411a7f5c615b634ffb0fc4f88792bd0ca6a68b31"
                ),
                "stderr_path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/rust.stderr"
                ),
                "stderr_file_sha256": (
                    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
                "stdout_summary": {
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-capacity-diagnostic/1"
                    ),
                    "actor_id": "RUST_ENDPOINT",
                    "capsule_digest": (
                        "sha256:6819416eed0a41573f21b4280a6a39e97481b87f8dd1c9e0054529e478bc7d9d"
                    ),
                    "source_commit": "3086adf389b8d9fa1a93c0c36fd300ed8bca19db",
                    "terminal_status": SATURATED,
                    "actual_preflight_peak_scratch_bytes": 0,
                    "actual_preflight_terminal_live_scratch_bytes": 0,
                    "diagnostic_only": True,
                    "q1_execution_started": False,
                    "q1_formal_roots": None,
                    "q1_gate_count": 0,
                    "q1_gate_mask": 0,
                    "q1_output_slots": [None] * 8,
                    "q1_receipt": None,
                    "q1_state": "NOT_RUN",
                    "q2_state": "NOT_RUN",
                    "role_evaluation_performed": False,
                    "split_accessed": False,
                    "target_truth_accessed": False,
                    "active_transition_allowed": False,
                },
            },
        },
        "trusted_host_replay_failure": {
            "return_code": 1,
            "stdout_path": (
                "/home/erzhu419/.local/state/hegel-machine/"
                "q1-full-node6-capacity-v5-run/host-replay.stdout"
            ),
            "stdout_file_sha256": (
                "sha256:11c1ebed96bd610584294cd175978c4a44b0c710d0ec08d0801b64898ef93573"
            ),
            "stderr_path": (
                "/home/erzhu419/.local/state/hegel-machine/"
                "q1-full-node6-capacity-v5-run/host-replay.stderr"
            ),
            "stderr_file_sha256": (
                "sha256:f7111d670e7d0d1d0c1421a10fabceb9a9308bb0d5c7d567190455c4829dc29a"
            ),
            "stderr_text": (
                "FAIL_FULL_NODE6_HOST_RESOURCE: RUST_ENDPOINT actual scratch "
                "accounting differs\n"
            ),
            "stdout_object": {
                "error_code": "FAIL_FULL_NODE6_HOST_RESOURCE",
                "error_detail": "RUST_ENDPOINT actual scratch accounting differs",
                "q1_gate_count": 0,
                "q1_gate_mask": 0,
                "q1_output_slots": [None] * 8,
                "q1_state": "NOT_RUN",
                "schema_version": (
                    "hegel-phase3a-q1-full-node6-host-replay-error/1"
                ),
                "terminal_status": (
                    "FULL_NODE6_HOST_REPLAY_FAILED_TERMINAL_NO_RESUME"
                ),
            },
        },
        "resource_evidence": {
            "python": {
                "path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/python-resource-evidence-v1.json"
                ),
                "file_sha256": (
                    "sha256:9d4facca1c85f2cc5a27bc2225527bd4b29cddf26821bc648252b1171686c6e2"
                ),
                "row_count": 1,
                "actor_id": "PYTHON_ENDPOINT",
                "actor_command_return_code": 0,
                "endpoint_sequence_ordinal": 1,
            },
            "rust": {
                "path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/rust-resource-evidence-v1.json"
                ),
                "file_sha256": (
                    "sha256:b82085715b23ce5c64bfc3d344a319018fb773ec9f62584a069dedfd7fbbabc2"
                ),
                "row_count": 1,
                "actor_id": "RUST_ENDPOINT",
                "actor_command_return_code": 0,
                "endpoint_sequence_ordinal": 2,
            },
            "aggregate": {
                "path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/endpoint-resource-evidence-v1.json"
                ),
                "file_sha256": (
                    "sha256:259730b31c525b6d58d0ea8390a87a9736704e4045932fe496e8f5912d3cf263"
                ),
                "row_count": 2,
                "actor_order": ["PYTHON_ENDPOINT", "RUST_ENDPOINT"],
                "maximum_concurrently_live_endpoint_containers": 1,
                "measurement_intervals_overlap": False,
            },
            "host": {
                "path": (
                    "/home/erzhu419/.local/state/hegel-machine/"
                    "q1-full-node6-capacity-v5-run/host-replay-resource-evidence-v1.json"
                ),
                "file_sha256": (
                    "sha256:705350f854c526fa7f2e2e73906e674b9bf79ea51d7d29b0ebf8e17913ccc3c5"
                ),
                "row_count": 1,
                "actor_id": "TRUSTED_HOST_REPLAY",
                "actor_command_return_code": 1,
            },
        },
        "scratch_cleanup": {
            "path": (
                "/home/erzhu419/.local/state/hegel-machine/"
                "q1-full-node6-capacity-v5-run/scratch-cleanup-v1.json"
            ),
            "file_sha256": (
                "sha256:505e39652b74c45353ee48c665d16dba35b552a8ae72cfa2bba703e34001882f"
            ),
            "terminal_absence_proved": True,
            "failure_rows": [],
            "root_absence": {"python-scratch": True, "rust-scratch": True},
        },
        "host_replay_scratch_cleanup": {
            "path": (
                "/home/erzhu419/.local/state/hegel-machine/"
                "q1-full-node6-capacity-v5-run/host-replay-scratch-cleanup-v1.json"
            ),
            "file_sha256": (
                "sha256:5830397ea43e8d65897e30d0c0075b759cfd9f33995f5c7f9a46f04532b98792"
            ),
            "terminal_absence_proved": True,
            "failure_rows": [],
            "root_absence": {"host-replay-scratch": True},
        },
        "postconsumption_cleanup": {
            "schema_version": (
                "hegel-phase3a-q1-full-node6-postconsumption-cleanup/1"
            ),
            "status": "PASS",
            "supervisor_process_group": (
                "PASS:group_kill=ABSENT,group_absence=PASS,"
                "fallback_kill=NOT_REQUIRED,wait=PASS"
            ),
            "production_containers": "PASS",
            "control_containers": "PASS",
            "probe_root": "PASS",
            "production_scratch": "PASS",
        },
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "retry_or_resume_allowed": False,
    }
    if superseded != expected_superseded:
        _fail("superseded campaign terminal boundary differs")
    target = _object(contract["target"], "target")
    if (
        target.get("action_id") != "full-node6-capacity-preflight-v1"
        or target.get("technical_contract_relative_path")
        != "config/phase3_q1_full_node6_capacity_preflight_v1.json"
        or target.get("artifact_relative_path")
        != "artifacts/phase3_q1_full_node6_capacity_preflight_v6.json"
        or target.get("allowed_success_status") != SATURATED
        or target.get("maximum_supervisor_popen_count") != 1
        or target.get("maximum_measurements_per_endpoint_signature") != 1
        or target.get("fresh_quotient_rerun_for_projection_forbidden") is not True
    ):
        _fail("target execution boundary differs")
    fixed = _object(contract["fixed_paths"], "fixed_paths")
    if set(fixed) != {
        "project_root",
        "cargo_cache_source",
        "campaign_lock_path",
        "ledger_root",
        "evidence_root",
        "trajectory_root",
        "run_root",
        "probe_root",
        "artifact_path",
        "stdout_path",
        "stderr_path",
    } or any(type(item) is not str or not item for item in fixed.values()):
        _fail("fresh successor-v6 paths differ")
    if contract["preconsumption_gate_order"] != [
        "AUTHORIZATION_TEXT",
        "PATH_AND_FORMAT",
        "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
        "PREDECESSOR_TERMINAL_AND_ARTIFACT",
        "SUPERSEDED_CAMPAIGN_TERMINAL",
        "TECHNICAL_CONTRACT_AND_FROZEN_SEMANTICS",
        "EXACT_COMMIT_SNAPSHOT_MATERIALIZATION",
        "PYTHON_RUST_OFFLINE_DEPENDENCIES",
        "PINNED_LOCAL_IMAGES_AND_DAEMON",
        "CPU_TOPOLOGY_AND_EXACT_AFFINITY",
        "RUNTIME_ISOLATION_LIVE_GAUNTLET",
        "MEMORY_AND_STORAGE_RESERVATION",
        "CURRENT_FUTURE_FORMAL_AND_PROBE_ROOTS_ABSENT",
    ]:
        _fail("preconsumption gate order differs")
    execution = _object(contract["execution_policy"], "execution_policy")
    if (
        execution.get("logical_campaign_count") != 1
        or execution.get("maximum_supervisor_popen_count") != 1
        or execution.get("supervisor_timeout_seconds") != 1212000
        or execution.get("pending_exit_code") != 75
        or execution.get("all_gates_before_consumption_and_popen") is not True
        or execution.get("pending_requires_zero_ledger_write_and_zero_popen")
        is not True
        or execution.get("same_capsule_pending_reentry_allowed") is not True
        or execution.get("runtime_isolation_live_gauntlet_required_receipt")
        != {
            "probe_command": "/bin/true",
            "quotient_engine_invocation_count": 0,
            "signature_selection_count": 0,
            "production_endpoint_start_count": 0,
            "supervisor_popen_count": 0,
        }
        or execution.get(
            "postconsumption_cleanup_receipt_required_on_all_child_return_paths"
        )
        is not True
        or execution.get("postconsumption_cleanup_required_components")
        != [
            "SUPERVISOR_PROCESS_GROUP_ABSENCE",
            "EXACT_PRODUCTION_CONTAINER_ABSENCE",
            "EXACT_CONTROL_CONTAINER_ABSENCE",
            "PROBE_ROOT_ABSENCE",
            "EXACT_PRODUCTION_SCRATCH_ABSENCE",
        ]
        or execution.get("postconsumption_cleanup_preserves_run_evidence")
        is not True
        or execution.get(
            "postconsumption_cleanup_uncertain_is_unknown_terminal"
        )
        is not True
        or execution.get("postconsumption_terminal_no_resume") is not True
        or execution.get("network_allowed") is not False
        or execution.get("q1_state") != "NOT_RUN"
    ):
        _fail("execution policy differs")
    authority = _object(contract["authority"], "authority")
    if authority != {
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_roots": None,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "active_transition_allowed": False,
        "adoption_allowed": False,
        "promotion_allowed": False,
        "current_pointer_write_allowed": False,
        "external_scientific_claim_allowed": False,
    }:
        _fail("authority boundary differs")
    return contract


def build_capsule_v1(
    contract: Mapping[str, Any],
    *,
    source_commit: str,
    source_tree: str,
    source_closure_digest: str,
    static_gate_receipts: Mapping[str, object],
) -> dict[str, Any]:
    validated = validate_contract_v1(contract)
    _commit(source_commit, "source_commit")
    _commit(source_tree, "source_tree")
    _digest(source_closure_digest, "source_closure_digest")
    if type(static_gate_receipts) is not dict or not static_gate_receipts:
        _fail("static_gate_receipts must be one nonempty object")
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "status": PREFLIGHT_ADMITTED,
        "contract_digest": validated["contract_digest"],
        "authorization": deepcopy(OWNER_AUTHORIZATION),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_closure_digest": source_closure_digest,
        "static_gate_receipts": deepcopy(dict(static_gate_receipts)),
        "maximum_supervisor_popen_count": 1,
        "maximum_measurements_per_endpoint_signature": 1,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = domain_digest_v1(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule_v1(
    value: object,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    capsule = _object(value, "capsule")
    validated = validate_contract_v1(contract)
    observed = _digest(capsule.get("capsule_digest"), "capsule_digest")
    unsigned = deepcopy(capsule)
    unsigned.pop("capsule_digest", None)
    if (
        capsule.get("schema_version") != CAPSULE_SCHEMA_VERSION
        or capsule.get("status") != PREFLIGHT_ADMITTED
        or capsule.get("contract_digest") != validated["contract_digest"]
        or capsule.get("authorization") != OWNER_AUTHORIZATION
        or capsule.get("maximum_supervisor_popen_count") != 1
        or capsule.get("maximum_measurements_per_endpoint_signature") != 1
        or capsule.get("q1_state") != "NOT_RUN"
        or capsule.get("q1_gate_count") != 0
        or capsule.get("q1_gate_mask") != 0
        or capsule.get("q1_output_slots") != [None] * 8
        or domain_digest_v1(DOMAINS["capsule"], unsigned) != observed
    ):
        _fail("capsule differs")
    _commit(capsule.get("source_commit"), "capsule.source_commit")
    _commit(capsule.get("source_tree"), "capsule.source_tree")
    _digest(capsule.get("source_closure_digest"), "capsule.source_closure_digest")
    return capsule


def build_pending_v1(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
) -> dict[str, Any]:
    validated = validate_contract_v1(contract)
    cap = validate_capsule_v1(capsule, validated)
    if any(type(value) is not str or not value for value in (phase, condition, detail)):
        _fail("pending phase, condition and detail must be nonempty")
    body = {
        "schema_version": PENDING_SCHEMA_VERSION,
        "status": PENDING_NOT_CONSUMED,
        "contract_digest": validated["contract_digest"],
        "capsule_digest": cap["capsule_digest"],
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "supervisor_popen_count": 0,
        "retry_transition_count": 0,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = domain_digest_v1(DOMAINS["pending"], unsigned)
    return body


def build_ledger_record_v1(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    sequence: int,
    record_type: str,
    status: str,
    previous_record_digest_or_null: str | None,
    body: Mapping[str, object],
) -> dict[str, Any]:
    validated = validate_contract_v1(contract)
    cap = validate_capsule_v1(capsule, validated)
    if type(sequence) is not int or sequence not in (1, 2):
        _fail("ledger sequence differs")
    if (sequence, record_type) not in ((1, "CONSUMED"), (2, "TERMINAL")):
        _fail("ledger record type differs")
    if sequence == 1 and previous_record_digest_or_null is not None:
        _fail("consumed ledger predecessor must be null")
    if sequence == 2:
        _digest(previous_record_digest_or_null, "previous_record_digest_or_null")
    value = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "contract_digest": validated["contract_digest"],
        "capsule_digest": cap["capsule_digest"],
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "sequence": sequence,
        "record_type": record_type,
        "status": status,
        "previous_record_digest_or_null": previous_record_digest_or_null,
        "body": deepcopy(dict(body)),
        "record_digest": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("record_digest")
    value["record_digest"] = domain_digest_v1(DOMAINS["ledger"], unsigned)
    return value


@dataclass(frozen=True, slots=True)
class CampaignDecisionV1:
    status: str
    success: bool
    retry_or_resume_allowed: bool


def decide_supervisor_return_v1(
    *,
    return_code: int,
    terminal_status_or_null: str | None,
    replay_passed: bool,
) -> CampaignDecisionV1:
    if type(return_code) is not int or type(replay_passed) is not bool:
        _fail("supervisor decision inputs differ")
    if (
        return_code == 0
        and terminal_status_or_null == SATURATED
        and replay_passed
    ):
        return CampaignDecisionV1(SATURATED, True, False)
    if (
        return_code == 0
        and terminal_status_or_null == GUARD_HIT
        and replay_passed
    ):
        return CampaignDecisionV1(GUARD_HIT, False, False)
    if (
        return_code == 3
        and terminal_status_or_null == RESOURCE_INFEASIBLE
        and replay_passed
    ):
        return CampaignDecisionV1(RESOURCE_INFEASIBLE, False, False)
    return CampaignDecisionV1(FAILED_TERMINAL, False, False)


__all__ = [
    "CAPSULE_SCHEMA_VERSION",
    "CONSUMED",
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "CampaignDecisionV1",
    "FAILED_TERMINAL",
    "FullNode6CampaignValidationError",
    "GUARD_HIT",
    "LEDGER_SCHEMA_VERSION",
    "OUTCOME_SCHEMA_VERSION",
    "OUTCOME_UNKNOWN_TERMINAL",
    "OWNER_AUTHORIZATION",
    "OWNER_AUTHORIZATION_DIGEST",
    "OWNER_AUTHORIZATION_TEXT",
    "PENDING_NOT_CONSUMED",
    "PREFLIGHT_ADMITTED",
    "RESOURCE_INFEASIBLE",
    "SATURATED",
    "build_capsule_v1",
    "build_ledger_record_v1",
    "build_pending_v1",
    "canonical_json_bytes_v1",
    "decide_supervisor_return_v1",
    "domain_digest_v1",
    "validate_capsule_v1",
    "validate_contract_v1",
]

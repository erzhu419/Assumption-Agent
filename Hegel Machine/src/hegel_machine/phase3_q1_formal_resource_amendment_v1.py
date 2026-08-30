"""Pure derivation of the post-v6 Q1 formal resource candidate.

This module accepts exact bytes from the successful successor-v6 artifact and
the two small endpoint stdout records.  It verifies their existing byte
bindings and applies only the preregistered resource formulas.  It has no file,
process, network, ledger, publication, run-genesis, gate-transition or Q1
execution authority.
"""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Final, Mapping, NoReturn


SCHEMA_VERSION: Final = "hegel-phase3a-q1-formal-resource-amendment-candidate/1"
ACTION_ID: Final = "phase3-q1-formal-resource-amendment-v1"
PREDECESSOR_ACTION_ID: Final = "full-node6-capacity-preflight-v1"
STATUS: Final = "FORMAL_RESOURCE_ENVELOPE_DERIVED_Q1_NOT_RUN_0_OF_20"
V6_CAPSULE_DIGEST: Final = (
    "sha256:a22cc6660e01c56b5670cca770d29d06c40bd06851d18868cbff07b61e918a29"
)
V6_ARTIFACT_SHA256: Final = (
    "sha256:73c319b54c53af75c3dbe01d64cd74aeb903c7d2edf1491cea7a8b7eb38194f4"
)
V6_SOURCE_COMMIT: Final = "86cedd4a43f6aa769a52ec9b246cfc1fa678a2a0"
V6_TERMINAL_STATUS: Final = "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY"
Q1_STATE: Final = "NOT_RUN"
Q1_GATE_COUNT: Final = 0
Q1_GATE_MASK: Final = 0
Q1_OUTPUT_SLOT_COUNT: Final = 8
MAXIMUM_INPUT_BYTES: Final = 16 * 1024 * 1024
ENDPOINT_METADATA_RESERVATION_BYTES: Final = 1_048_576
HOUR_SECONDS: Final = 3_600
NANOSECONDS_PER_SECOND: Final = 1_000_000_000

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

_SEMANTIC_HIGH_WATER_FIELDS: Final = (
    ("peak_raw_operator_application_count", "maximum_raw_operator_applications"),
    ("peak_behavior_class_count", "maximum_behavior_classes"),
    ("peak_visible_frontier_point_count", "maximum_visible_frontier_points"),
    (
        "peak_visible_frontier_points_per_class",
        "maximum_visible_frontier_points_per_class",
    ),
    (
        "peak_continuation_bank_point_count",
        "maximum_continuation_bank_points",
    ),
    (
        "peak_continuation_bank_points_per_class",
        "maximum_continuation_bank_points_per_class",
    ),
)


class FormalResourceAmendmentError(RuntimeError):
    """Stable rejection from the side-effect-free amendment derivation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FormalResourceAmendmentError(code, detail)


def canonical_json_bytes_v1(value: object) -> bytes:
    """Return the one-line canonical JSON representation used by this action."""

    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_RESOURCE_CANDIDATE_JSON", str(error))


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail("REJECT_Q1_RESOURCE_INPUT_JSON", f"duplicate key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> NoReturn:
    _fail("REJECT_Q1_RESOURCE_INPUT_JSON", f"non-finite token {value!r}")


def decode_canonical_json_object_v1(payload: bytes, label: str) -> dict[str, object]:
    """Decode one exact canonical JSON object without accepting aliases."""

    if type(payload) is not bytes:
        _fail("REJECT_Q1_RESOURCE_INPUT_TYPE", f"{label} must be exact bytes")
    if not payload or len(payload) > MAXIMUM_INPUT_BYTES:
        _fail("REJECT_Q1_RESOURCE_INPUT_SIZE", f"{label} byte length differs")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except FormalResourceAmendmentError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("REJECT_Q1_RESOURCE_INPUT_JSON", f"{label}: {error}")
    if type(value) is not dict:
        _fail("REJECT_Q1_RESOURCE_INPUT_JSON", f"{label} must be an object")
    if canonical_json_bytes_v1(value) != payload:
        _fail("REJECT_Q1_RESOURCE_INPUT_CANONICAL", f"{label} bytes differ")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail("REJECT_Q1_RESOURCE_INPUT_SHAPE", f"{label} must be an object")
    return value


def _rows(value: object, label: str, length: int) -> list[object]:
    if type(value) is not list or len(value) != length:
        _fail(
            "REJECT_Q1_RESOURCE_INPUT_SHAPE",
            f"{label} must contain exactly {length} rows",
        )
    return value


def _uint(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail("REJECT_Q1_RESOURCE_INPUT_UINT", f"{label} must be a uint")
    return value


def _sha256(value: object, label: str, *, prefixed: bool = True) -> str:
    expression = _SHA256 if prefixed else _SHA256_HEX
    if type(value) is not str or expression.fullmatch(value) is None:
        suffix = "sha256:<64hex>" if prefixed else "64 lowercase hex"
        _fail("REJECT_Q1_RESOURCE_INPUT_DIGEST", f"{label} must be {suffix}")
    return value


def _require_q1_closed(value: Mapping[str, object], label: str) -> None:
    if (
        value.get("q1_state") != Q1_STATE
        or type(value.get("q1_gate_count")) is not int
        or value.get("q1_gate_count") != Q1_GATE_COUNT
        or type(value.get("q1_gate_mask")) is not int
        or value.get("q1_gate_mask") != Q1_GATE_MASK
        or value.get("q1_output_slots") != [None] * Q1_OUTPUT_SLOT_COUNT
    ):
        _fail("REJECT_Q1_RESOURCE_AUTHORITY", f"{label} Q1 authority differs")


def pow2ceil_v1(value: int) -> int:
    """Least positive power of two greater than or equal to ``max(1, value)``."""

    value = _uint(value, "pow2ceil input")
    return 1 << (max(1, value) - 1).bit_length()


def hour_rounded_wall_guard_seconds_v1(elapsed_ns: int) -> int:
    """Apply the preregistered three-times wall observation, rounded by hour."""

    elapsed_ns = _uint(elapsed_ns, "wall observation")
    hour_ns = HOUR_SECONDS * NANOSECONDS_PER_SECOND
    return HOUR_SECONDS * ((3 * elapsed_ns + hour_ns - 1) // hour_ns)


def _partition_rows_v1(
    endpoint: Mapping[str, object], actor_id: str
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        endpoint.get("action_id") != PREDECESSOR_ACTION_ID
        or endpoint.get("actor_id") != actor_id
        or endpoint.get("capsule_digest") != V6_CAPSULE_DIGEST
        or endpoint.get("source_commit") != V6_SOURCE_COMMIT
        or endpoint.get("terminal_status") != V6_TERMINAL_STATUS
        or endpoint.get("diagnostic_only") is not True
        or endpoint.get("q1_execution_started") is not False
        or endpoint.get("q1_formal_roots") is not None
        or endpoint.get("q1_receipt") is not None
        or endpoint.get("target_truth_accessed") is not False
        or endpoint.get("split_accessed") is not False
        or endpoint.get("role_evaluation_performed") is not False
    ):
        _fail("REJECT_Q1_RESOURCE_ENDPOINT", f"{actor_id} boundary differs")
    _require_q1_closed(endpoint, actor_id)
    partitions = _rows(endpoint.get("partitions"), f"{actor_id} partitions", 2)
    output: list[dict[str, object]] = []
    for ordinal, raw in enumerate(partitions, 1):
        row = _object(raw, f"{actor_id} partition {ordinal}")
        expected_name = "OddInputV1" if ordinal == 1 else "SinkInputV1"
        fixed_point = _object(
            row.get("fixed_point"), f"{actor_id} partition {ordinal} fixed point"
        )
        if (
            row.get("input_signature_id") != ordinal
            or type(row.get("input_signature_id")) is not int
            or row.get("input_signature_name") != expected_name
            or row.get("terminal_status") != V6_TERMINAL_STATUS
            or row.get("resource_guard_id") is not None
            or row.get("resource_guard_name") is not None
            or row.get("full_v16_structural_limits_applied") is not True
            or fixed_point.get("no_resource_guard_hit") is not True
            or fixed_point.get("saturated_diagnostic_only") is not True
        ):
            _fail(
                "REJECT_Q1_RESOURCE_PARTITION",
                f"{actor_id} partition {ordinal} is not saturated",
            )
        for observation, _guard in _SEMANTIC_HIGH_WATER_FIELDS:
            _uint(row.get(observation), f"{actor_id} {expected_name} {observation}")
        _uint(
            row.get("peak_work_queue_points"),
            f"{actor_id} {expected_name} peak_work_queue_points",
        )
        _uint(
            row.get("peak_saturation_round_count"),
            f"{actor_id} {expected_name} peak_saturation_round_count",
        )
        output.append(row)
    return output[0], output[1]


def _actor_rows_v1(host_replay: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows = _rows(host_replay.get("endpoint_rows"), "host endpoint rows", 2)
    output: dict[str, dict[str, object]] = {}
    for raw in rows:
        row = _object(raw, "host endpoint row")
        actor_id = row.get("actor_id")
        if actor_id not in ("PYTHON_ENDPOINT", "RUST_ENDPOINT") or actor_id in output:
            _fail("REJECT_Q1_RESOURCE_HOST_REPLAY", "endpoint actor rows differ")
        _sha256(row.get("stdout_sha256"), f"{actor_id} stdout", prefixed=False)
        output[actor_id] = row
    if set(output) != {"PYTHON_ENDPOINT", "RUST_ENDPOINT"}:
        _fail("REJECT_Q1_RESOURCE_HOST_REPLAY", "endpoint actor set differs")
    return output


def _resource_envelope_v1(
    artifact: Mapping[str, object], host_replay: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    adjudication = _object(
        artifact.get("resource_adjudication"), "resource adjudication"
    )
    if (
        adjudication.get("terminal_status") != V6_TERMINAL_STATUS
        or adjudication.get("input_terminal_status") != V6_TERMINAL_STATUS
        or adjudication.get("host_replay_passed") is not True
        or adjudication.get("first_resource_guard_id") is not None
        or adjudication.get("first_resource_guard_name") is not None
        or adjudication.get("infeasible_derived_fields") != []
    ):
        _fail("REJECT_Q1_RESOURCE_ADJUDICATION", "terminal boundary differs")
    direct_rows = _rows(
        adjudication.get("direct_endpoint_resources"), "direct endpoint resources", 2
    )
    direct: dict[str, dict[str, object]] = {}
    for raw in direct_rows:
        row = _object(raw, "direct endpoint resource row")
        actor_id = row.get("actor_id")
        if actor_id not in ("PYTHON_ENDPOINT", "RUST_ENDPOINT") or actor_id in direct:
            _fail("REJECT_Q1_RESOURCE_ADJUDICATION", "resource actor rows differ")
        direct[actor_id] = row
    if set(direct) != {"PYTHON_ENDPOINT", "RUST_ENDPOINT"}:
        _fail("REJECT_Q1_RESOURCE_ADJUDICATION", "resource actor set differs")

    formal = _object(
        host_replay.get("formal_resource_projection"), "formal resource projection"
    )
    projected_output = _uint(
        formal.get("projected_formal_total_output_bytes"),
        "projected formal total output",
    )
    projected_scratch = _uint(
        formal.get("projected_peak_formal_scratch_bytes"),
        "projected peak formal scratch",
    )
    maximum_endpoint_rss = max(
        _uint(row.get("actual_peak_rss_bytes"), f"{actor} peak RSS")
        for actor, row in direct.items()
    )
    slower_endpoint_wall = max(
        _uint(row.get("actual_wall_time_ns"), f"{actor} wall time")
        for actor, row in direct.items()
    )
    host_actual = _object(
        adjudication.get("host_replay_actual_resources"), "host actual resources"
    )
    host_rss = _uint(host_actual.get("actual_peak_rss_bytes"), "host peak RSS")
    host_wall = _uint(host_actual.get("actual_wall_time_ns"), "host wall time")
    host_output = _uint(
        host_replay.get("projected_host_replay_output_bytes"),
        "projected host output",
    )
    host_scratch = _uint(
        host_replay.get("projected_peak_host_replay_scratch_bytes"),
        "projected host scratch",
    )
    derived = {
        "endpoint_memory_guard_bytes": pow2ceil_v1(2 * maximum_endpoint_rss),
        "endpoint_output_guard_bytes": pow2ceil_v1(2 * projected_output),
        "endpoint_scratch_guard_bytes": pow2ceil_v1(2 * projected_scratch),
        "endpoint_wall_guard_seconds": hour_rounded_wall_guard_seconds_v1(
            slower_endpoint_wall
        ),
        "evaluated": True,
        "host_memory_guard_bytes": pow2ceil_v1(2 * host_rss),
        "host_output_guard_bytes": pow2ceil_v1(2 * host_output),
        "host_scratch_guard_bytes": pow2ceil_v1(2 * host_scratch),
        "host_wall_guard_seconds": hour_rounded_wall_guard_seconds_v1(host_wall),
    }
    if adjudication.get("derived_envelope") != derived:
        _fail("REJECT_Q1_RESOURCE_DERIVATION", "persisted resource envelope differs")
    observations = {
        "endpoint": {
            "maximum_peak_rss_bytes": maximum_endpoint_rss,
            "projected_formal_total_output_bytes": projected_output,
            "projected_peak_formal_scratch_bytes": projected_scratch,
            "slower_wall_time_ns": slower_endpoint_wall,
        },
        "trusted_host_replay": {
            "peak_rss_bytes": host_rss,
            "projected_output_bytes": host_output,
            "projected_peak_scratch_bytes": host_scratch,
            "wall_time_ns": host_wall,
        },
    }
    envelope = {
        "endpoint": {
            "memory_bytes": derived["endpoint_memory_guard_bytes"],
            "output_bytes": derived["endpoint_output_guard_bytes"],
            "scratch_bytes": derived["endpoint_scratch_guard_bytes"],
            "wall_time_seconds": derived["endpoint_wall_guard_seconds"],
        },
        "trusted_host_replay": {
            "memory_bytes": derived["host_memory_guard_bytes"],
            "output_bytes": derived["host_output_guard_bytes"],
            "scratch_bytes": derived["host_scratch_guard_bytes"],
            "wall_time_seconds": derived["host_wall_guard_seconds"],
        },
    }
    return observations, envelope


def derive_formal_resource_amendment_v1(
    artifact_payload: bytes,
    python_stdout_payload: bytes,
    rust_stdout_payload: bytes,
    *,
    expected_artifact_sha256: str = V6_ARTIFACT_SHA256,
    expected_capsule_digest: str = V6_CAPSULE_DIGEST,
    expected_source_commit: str = V6_SOURCE_COMMIT,
) -> dict[str, object]:
    """Validate the v6 byte bindings and return one deterministic candidate.

    The explicit expected arguments make synthetic known-answer tests possible.
    The actual CLI does not expose overrides and therefore always binds the
    constants above.
    """

    _sha256(expected_artifact_sha256, "expected artifact SHA-256")
    _sha256(expected_capsule_digest, "expected capsule digest")
    if type(expected_source_commit) is not str or _COMMIT.fullmatch(
        expected_source_commit
    ) is None:
        _fail("REJECT_Q1_RESOURCE_INPUT_COMMIT", "expected source commit differs")
    actual_artifact_sha256 = f"sha256:{sha256(artifact_payload).hexdigest()}"
    if actual_artifact_sha256 != expected_artifact_sha256:
        _fail("REJECT_Q1_RESOURCE_ARTIFACT_BINDING", "artifact SHA-256 differs")

    artifact = decode_canonical_json_object_v1(artifact_payload, "v6 artifact")
    python = decode_canonical_json_object_v1(
        python_stdout_payload, "Python endpoint stdout"
    )
    rust = decode_canonical_json_object_v1(rust_stdout_payload, "Rust endpoint stdout")
    if (
        artifact.get("action_id") != PREDECESSOR_ACTION_ID
        or artifact.get("capsule_digest") != expected_capsule_digest
        or expected_capsule_digest != V6_CAPSULE_DIGEST
        or expected_source_commit != V6_SOURCE_COMMIT
        or artifact.get("terminal_status") != V6_TERMINAL_STATUS
        or artifact.get("diagnostic_only") is not True
        or artifact.get("host_replay_passed") is not True
    ):
        _fail("REJECT_Q1_RESOURCE_ARTIFACT", "v6 terminal boundary differs")
    _require_q1_closed(artifact, "v6 artifact")
    host_replay = _object(artifact.get("host_replay"), "host replay")
    if (
        host_replay.get("capsule_digest") != expected_capsule_digest
        or host_replay.get("source_commit") != expected_source_commit
        or host_replay.get("terminal_status") != V6_TERMINAL_STATUS
        or host_replay.get("diagnostic_only") is not True
        or host_replay.get("registered_resource_guard_id") is not None
    ):
        _fail("REJECT_Q1_RESOURCE_HOST_REPLAY", "host terminal boundary differs")
    _require_q1_closed(host_replay, "host replay")

    actor_rows = _actor_rows_v1(host_replay)
    endpoint_payloads = {
        "PYTHON_ENDPOINT": python_stdout_payload,
        "RUST_ENDPOINT": rust_stdout_payload,
    }
    endpoint_objects = {"PYTHON_ENDPOINT": python, "RUST_ENDPOINT": rust}
    endpoint_stdout_sha256: dict[str, str] = {}
    for actor_id, payload in endpoint_payloads.items():
        digest = sha256(payload).hexdigest()
        if actor_rows[actor_id].get("stdout_sha256") != digest:
            _fail(
                "REJECT_Q1_RESOURCE_STDOUT_BINDING",
                f"{actor_id} stdout SHA-256 differs",
            )
        endpoint_stdout_sha256[actor_id] = f"sha256:{digest}"

    partitions = {
        actor_id: _partition_rows_v1(value, actor_id)
        for actor_id, value in endpoint_objects.items()
    }
    per_signature: list[dict[str, object]] = []
    work_queue_observations: list[int] = []
    all_round_observations: list[int] = []
    for index, expected_name in enumerate(("OddInputV1", "SinkInputV1")):
        python_row = partitions["PYTHON_ENDPOINT"][index]
        rust_row = partitions["RUST_ENDPOINT"][index]
        observations: dict[str, int] = {}
        guards: dict[str, int] = {}
        for observation_field, guard_field in _SEMANTIC_HIGH_WATER_FIELDS:
            python_value = _uint(
                python_row.get(observation_field), f"Python {observation_field}"
            )
            rust_value = _uint(
                rust_row.get(observation_field), f"Rust {observation_field}"
            )
            if python_value != rust_value:
                _fail(
                    "REJECT_Q1_RESOURCE_DUAL_AGREEMENT",
                    f"{expected_name} {observation_field} differs",
                )
            observations[observation_field] = python_value
            guards[guard_field] = pow2ceil_v1(2 * python_value)
        queue = max(
            _uint(python_row.get("peak_work_queue_points"), "Python work queue"),
            _uint(rust_row.get("peak_work_queue_points"), "Rust work queue"),
        )
        work_queue_observations.append(queue)
        observations["peak_work_queue_points"] = queue
        guards["maximum_work_queue_points"] = pow2ceil_v1(2 * queue)
        all_round_observations.extend(
            (
                _uint(
                    python_row.get("peak_saturation_round_count"),
                    "Python saturation rounds",
                ),
                _uint(
                    rust_row.get("peak_saturation_round_count"),
                    "Rust saturation rounds",
                ),
            )
        )
        per_signature.append(
            {
                "formal_guard": guards,
                "input_signature_id": index + 1,
                "input_signature_name": expected_name,
                "observed_high_water": observations,
            }
        )
    round_observation = max(all_round_observations)
    round_guard = round_observation + 2
    for row in per_signature:
        observed = _object(row["observed_high_water"], "observed high water")
        guard = _object(row["formal_guard"], "formal guard")
        observed["peak_saturation_round_count"] = round_observation
        guard["maximum_saturation_rounds"] = round_guard

    schedule = _object(
        host_replay.get("schedule_high_water_aggregation"),
        "schedule high-water aggregation",
    )
    if (
        schedule.get("peak_work_queue_points_by_input_signature")
        != work_queue_observations
        or schedule.get("peak_saturation_round_count_across_endpoints_and_signatures")
        != round_observation
    ):
        _fail("REJECT_Q1_RESOURCE_SCHEDULE", "host schedule aggregation differs")
    for actor_id, actor_row in actor_rows.items():
        endpoint_partitions = partitions[actor_id]
        if actor_row.get("peak_work_queue_points_by_input_signature") != [
            row["peak_work_queue_points"] for row in endpoint_partitions
        ] or actor_row.get("peak_saturation_round_count_by_input_signature") != [
            row["peak_saturation_round_count"] for row in endpoint_partitions
        ]:
            _fail(
                "REJECT_Q1_RESOURCE_SCHEDULE",
                f"{actor_id} schedule vector differs",
            )

    formal_projection = _object(
        host_replay.get("formal_resource_projection"), "formal projection"
    )
    projected_output = _uint(
        formal_projection.get("projected_formal_total_output_bytes"),
        "projected formal output",
    )
    projected_scratch = _uint(
        formal_projection.get("projected_peak_formal_scratch_bytes"),
        "projected formal scratch",
    )
    for actor_id, endpoint in endpoint_objects.items():
        if (
            endpoint.get("projected_formal_total_output_bytes") != projected_output
            or endpoint.get("projected_peak_formal_scratch_bytes")
            != projected_scratch
            or _uint(
                endpoint.get("projected_formal_archive_payload_bytes"),
                f"{actor_id} projected archive payload",
            )
            + ENDPOINT_METADATA_RESERVATION_BYTES
            != projected_output
        ):
            _fail(
                "REJECT_Q1_RESOURCE_FORMAL_PROJECTION",
                f"{actor_id} formal projection differs",
            )

    resource_observations, resource_envelope = _resource_envelope_v1(
        artifact, host_replay
    )
    agreement = _sha256(
        host_replay.get("dual_semantic_agreement_sha256"),
        "dual semantic agreement",
        prefixed=False,
    )
    return {
        "action_id": ACTION_ID,
        "all_derived_values_within_preregistered_hard_ceilings": True,
        "authoritative_apply_capsule_digest": None,
        "authoritative_apply_requires_new_exact_capsule": True,
        "candidate_only": True,
        "diagnostic_only": True,
        "dual_semantic_agreement_sha256": f"sha256:{agreement}",
        "formal_resource_envelope": resource_envelope,
        "infeasible_derived_fields": [],
        "network_accessed": False,
        "per_signature_formal_count_guards": per_signature,
        "popen_count": 0,
        "predecessor": {
            "artifact_sha256": actual_artifact_sha256,
            "capsule_digest": expected_capsule_digest,
            "endpoint_stdout_sha256": endpoint_stdout_sha256,
            "source_commit": expected_source_commit,
            "terminal_status": V6_TERMINAL_STATUS,
        },
        "q1_execution_started": False,
        "q1_formal_roots": None,
        "q1_gate_count": Q1_GATE_COUNT,
        "q1_gate_mask": Q1_GATE_MASK,
        "q1_output_slots": [None] * Q1_OUTPUT_SLOT_COUNT,
        "q1_receipt": None,
        "q1_state": Q1_STATE,
        "resource_observations": resource_observations,
        "role_evaluation_performed": False,
        "run_genesis": None,
        "schema_version": SCHEMA_VERSION,
        "split_accessed": False,
        "status": STATUS,
        "target_truth_accessed": False,
    }


__all__ = (
    "ACTION_ID",
    "FormalResourceAmendmentError",
    "SCHEMA_VERSION",
    "STATUS",
    "V6_ARTIFACT_SHA256",
    "V6_CAPSULE_DIGEST",
    "V6_SOURCE_COMMIT",
    "canonical_json_bytes_v1",
    "decode_canonical_json_object_v1",
    "derive_formal_resource_amendment_v1",
    "hour_rounded_wall_guard_seconds_v1",
    "pow2ceil_v1",
)

"""Target-blind Q1 completion admission for Phase-3A-Q2.

This module has no Q2 execution surface.  It accepts only the sealed Q1 v8c
terminal and completion receipt, proves that they publish the exact Q1
``COMPLETE`` projection required by Q2, and returns an immutable local view of
the eight ordered Q1 roots.  It neither opens target or split inputs nor reads
the Q1 class archives.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, NoReturn

from . import phase3_q1_completion_replay_successor_v1 as _q1_state
from . import phase3_q1_start_completion_wire_v1 as _q1_completion
from .phase3_q1_formal_archive_contract_v1 import Q1_OUTPUT_SLOT_NAMES


PREREGISTRATION_SCHEMA: Final = "hegel-phase3a-q2-role-membership-preregistration/1"
PREREGISTRATION_ID: Final = "hegel-phase3a-q2-role-membership-v1"
PREREGISTRATION_STATUS: Final = "Q2_SPEC_FROZEN_NOT_AUTHORIZED"
ROLE_MATCH_PROFILE_ID: Final = "BOOL_BIT_EXACT_PREDICATE_MATCH_V1"

Q1_TERMINAL_SCHEMA: Final = "hegel-phase3a-q1-completion-replay-terminal/v8c"
Q1_COMPLETE_STATUS: Final = "Q1_COMPLETE_20_OF_20"
Q1_COMPLETE_GATE_COUNT: Final = 20
Q1_COMPLETE_GATE_MASK: Final = 1_048_575

_TAGGED_SHA256_PREFIX: Final = "sha256:"


class Q2AdmissionError(ValueError):
    """A sealed Q1 input is not admissible as the Q2 predecessor."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q2AdmissionError(code, detail)


def _tagged_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(_TAGGED_SHA256_PREFIX)
        or len(value) != len(_TAGGED_SHA256_PREFIX) + 64
    ):
        _fail("REJECT_Q2_Q1_BINDING", f"{label} is not one tagged SHA-256")
    suffix = value[len(_TAGGED_SHA256_PREFIX) :]
    if any(character not in "0123456789abcdef" for character in suffix):
        _fail("REJECT_Q2_Q1_BINDING", f"{label} is not one tagged SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class Q1CompletionBindingV1:
    """Exact predecessor identities selected by the Q2 preregistration."""

    capsule_digest: str
    terminal_record_digest: str
    completion_artifact_raw_sha256: str
    completion_receipt_root: str
    ordered_q1_output_slots: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "capsule_digest",
            "terminal_record_digest",
            "completion_artifact_raw_sha256",
            "completion_receipt_root",
        ):
            _tagged_sha256(getattr(self, field_name), field_name)
        if type(self.ordered_q1_output_slots) is not tuple or len(
            self.ordered_q1_output_slots
        ) != len(Q1_OUTPUT_SLOT_NAMES):
            _fail("REJECT_Q2_Q1_BINDING", "Q1 output slots must be one ordered 8-tuple")
        for index, root in enumerate(self.ordered_q1_output_slots, 1):
            _tagged_sha256(root, f"Q1 output slot {index}")
        if self.ordered_q1_output_slots[-1] != self.completion_receipt_root:
            _fail("REJECT_Q2_Q1_BINDING", "Q1 slot 8 must be the completion receipt root")


Q1_V8C_BINDING: Final = Q1CompletionBindingV1(
    capsule_digest=(
        "sha256:e6eafec73cb97b8048fc6322c21466b9d2f17e8f6f8f6f92d586d00bc361e0c0"
    ),
    terminal_record_digest=(
        "sha256:2163d03e5b295b3bb93418e3b05e7ae216254b264272ab682a26f9d70745ba9b"
    ),
    completion_artifact_raw_sha256=(
        "sha256:623bdfa6ce60b244d3dff9d0ed274dda2ef52048b3992f7dae07ce8983c3ce18"
    ),
    completion_receipt_root=(
        "sha256:c75d08e038dc17c01aa4c9c6b69a2d4ef95034901a55e7982fb6a93d78f7798a"
    ),
    ordered_q1_output_slots=(
        "sha256:496582bc6685de4323e372247bc23dc1a89999c80c1e1ba54ef6f4febb2a9200",
        "sha256:e5f901ca83f90b80c8c1a8f1a8836947a18ed50c4c5bee47c0db69f2a5030a6a",
        "sha256:a774cfedd950e648adee8a7f42de4bab1d1a90da700586cf9bf53fafa4f672bb",
        "sha256:a82cd0197c9bed90af04974ff3348c35058af7d08c4855415ef8f1bd9ded9026",
        "sha256:4c011a8b75cca0c52994340475be5cf3b29c5079a57f1b372d5fa59d22b1fe51",
        "sha256:2a94390022a2df730b5791cc139a576811da64365b456a1541ab67c439938cd0",
        "sha256:d0c96cfb6eb877a9b75367bc061a8721da1c167d6fe34cc33879023ed4cdec58",
        "sha256:c75d08e038dc17c01aa4c9c6b69a2d4ef95034901a55e7982fb6a93d78f7798a",
    ),
)


@dataclass(frozen=True, slots=True)
class Q1CompletionAdmissionV1:
    """The exact Q1 facts that a later, separately authorized Q2 may consume."""

    binding: Q1CompletionBindingV1
    ordered_output_slots: tuple[tuple[int, bytes, bytes], ...]

    @property
    def odd_signature_manifest_root(self) -> bytes:
        return self.ordered_output_slots[0][2]

    @property
    def odd_saturation_state_root(self) -> bytes:
        return self.ordered_output_slots[1][2]

    @property
    def sink_signature_manifest_root(self) -> bytes:
        return self.ordered_output_slots[2][2]

    @property
    def sink_saturation_state_root(self) -> bytes:
        return self.ordered_output_slots[3][2]


@dataclass(frozen=True, slots=True)
class Q2RoleCommitmentV1:
    role_id: int
    role_name: str
    input_signature_id: int
    row_count: int
    universe_root: str
    truth_root: str
    pass_membership_verdict: str
    pass_match_count_relation: str

    def __post_init__(self) -> None:
        expected = {
            1: (
                "OUTSIDE_TARGET",
                1,
                480,
                "OUTSIDE_FROZEN_QUOTIENT_CLOSURE",
                "EQUALS_ZERO",
            ),
            2: (
                "IN_LANGUAGE_NULL",
                2,
                85,
                "IN_FROZEN_QUOTIENT_CLOSURE",
                "GREATER_THAN_ZERO",
            ),
        }.get(self.role_id)
        if expected is None or (
            self.role_name,
            self.input_signature_id,
            self.row_count,
            self.pass_membership_verdict,
            self.pass_match_count_relation,
        ) != expected:
            _fail("REJECT_Q2_ROLE_COMMITMENT", "role registry row differs")
        _tagged_sha256(self.universe_root, "role universe root")
        _tagged_sha256(self.truth_root, "role truth root")


Q2_ROLE_COMMITMENTS: Final = (
    Q2RoleCommitmentV1(
        1,
        "OUTSIDE_TARGET",
        1,
        480,
        "sha256:b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05",
        "sha256:f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506",
        "OUTSIDE_FROZEN_QUOTIENT_CLOSURE",
        "EQUALS_ZERO",
    ),
    Q2RoleCommitmentV1(
        2,
        "IN_LANGUAGE_NULL",
        2,
        85,
        "sha256:1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5",
        "sha256:9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808",
        "IN_FROZEN_QUOTIENT_CLOSURE",
        "GREATER_THAN_ZERO",
    ),
)


def _record_digest(value: dict[str, object]) -> str:
    supplied = value.get("record_digest")
    unsigned = deepcopy(value)
    unsigned.pop("record_digest", None)
    computed = _TAGGED_SHA256_PREFIX + sha256(
        _q1_state.DOMAINS["terminal"] + _q1_state.canonical_json_bytes(unsigned)
    ).hexdigest()
    if supplied != computed:
        _fail("REJECT_Q2_Q1_TERMINAL", "Q1 terminal record digest does not replay")
    return computed


def _artifact_digest(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail("REJECT_Q2_Q1_ARTIFACT", "Q1 completion artifact must be exact bytes")
    return _TAGGED_SHA256_PREFIX + sha256(payload).hexdigest()


def admit_q1_completion_v1(
    terminal_payload: bytes,
    completion_artifact_payload: bytes,
    *,
    binding: Q1CompletionBindingV1 = Q1_V8C_BINDING,
) -> Q1CompletionAdmissionV1:
    """Validate the sealed Q1 predecessor without opening any Q2 input."""

    try:
        terminal = _q1_state.strict_json_bytes(terminal_payload, "Q1 terminal")
    except ValueError as error:
        _fail("REJECT_Q2_Q1_TERMINAL", str(error))
    if terminal.get("schema_version") != Q1_TERMINAL_SCHEMA:
        _fail("REJECT_Q2_Q1_TERMINAL", "Q1 terminal schema differs")
    if terminal.get("status") != Q1_COMPLETE_STATUS:
        _fail("REJECT_Q2_Q1_TERMINAL", "Q1 terminal is not complete")
    if terminal.get("capsule_digest") != binding.capsule_digest:
        _fail("REJECT_Q2_Q1_TERMINAL", "Q1 capsule digest differs")
    if _record_digest(terminal) != binding.terminal_record_digest:
        _fail("REJECT_Q2_Q1_TERMINAL", "Q1 terminal lineage differs")
    if terminal.get("authorization_consumption_count") != 1 or terminal.get(
        "logical_campaign_count"
    ) != 1:
        _fail("REJECT_Q2_Q1_TERMINAL", "Q1 completion counts differ")

    artifact_digest = _artifact_digest(completion_artifact_payload)
    if artifact_digest != binding.completion_artifact_raw_sha256 or terminal.get(
        "artifact_raw_sha256"
    ) != artifact_digest:
        _fail("REJECT_Q2_Q1_ARTIFACT", "Q1 completion artifact digest differs")
    try:
        receipt = _q1_completion.decode_completion_receipt_v1(
            completion_artifact_payload
        )
    except ValueError as error:
        _fail("REJECT_Q2_Q1_ARTIFACT", str(error))

    ordered_output_slots = receipt.output_slots
    expected_ids = tuple(range(1, 9))
    if tuple(row[0] for row in ordered_output_slots) != expected_ids or tuple(
        row[1] for row in ordered_output_slots
    ) != Q1_OUTPUT_SLOT_NAMES:
        _fail("REJECT_Q2_Q1_ARTIFACT", "Q1 output-slot registry differs")
    tagged_slots = tuple(_TAGGED_SHA256_PREFIX + row[2].hex() for row in ordered_output_slots)
    if tagged_slots != binding.ordered_q1_output_slots:
        _fail("REJECT_Q2_Q1_ARTIFACT", "Q1 output-slot roots differ")
    receipt_root = _TAGGED_SHA256_PREFIX + receipt.receipt_root.hex()
    if receipt_root != binding.completion_receipt_root:
        _fail("REJECT_Q2_Q1_ARTIFACT", "Q1 completion receipt root differs")

    outcome = terminal.get("outcome")
    if type(outcome) is not dict:
        _fail("REJECT_Q2_Q1_TERMINAL", "Q1 terminal outcome is absent")
    required_projection = {
        "status": Q1_COMPLETE_STATUS,
        "q1_state": "COMPLETE",
        "q1_gate_count": Q1_COMPLETE_GATE_COUNT,
        "q1_gate_mask": Q1_COMPLETE_GATE_MASK,
        "q1_output_slots": list(tagged_slots),
        "q2_state": "NOT_RUN",
        "completion_receipt_root": receipt_root,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "network_accessed": False,
        "v7r1_resumed": False,
        "v7r1_outputs_mutated": False,
        "v7r1_outputs_read_only_replayed": True,
        "terminal_no_resume": False,
    }
    for field_name, expected in required_projection.items():
        if outcome.get(field_name) != expected:
            _fail("REJECT_Q2_Q1_TERMINAL", f"Q1 outcome {field_name} differs")
    return Q1CompletionAdmissionV1(binding, ordered_output_slots)


def q2_preregistration_object_v1() -> dict[str, object]:
    """Return the target-blind, not-authorized Q2 freeze object."""

    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "preregistration_id": PREREGISTRATION_ID,
        "status": PREREGISTRATION_STATUS,
        "q1_completion_binding": {
            "capsule_digest": Q1_V8C_BINDING.capsule_digest,
            "terminal_record_digest": Q1_V8C_BINDING.terminal_record_digest,
            "completion_artifact_raw_sha256": (
                Q1_V8C_BINDING.completion_artifact_raw_sha256
            ),
            "completion_receipt_root": Q1_V8C_BINDING.completion_receipt_root,
            "ordered_output_slots": list(Q1_V8C_BINDING.ordered_q1_output_slots),
        },
        "role_match_profile_id": ROLE_MATCH_PROFILE_ID,
        "role_commitments": [
            {
                "role_id": row.role_id,
                "role_name": row.role_name,
                "input_signature_id": row.input_signature_id,
                "row_count": row.row_count,
                "universe_root": row.universe_root,
                "truth_root": row.truth_root,
                "pass_membership_verdict": row.pass_membership_verdict,
                "pass_match_count_relation": row.pass_match_count_relation,
            }
            for row in Q2_ROLE_COMMITMENTS
        ],
        "evaluation_order": ["OUTSIDE_TARGET", "IN_LANGUAGE_NULL"],
        "result_contract": {
            "role_result_fields": [
                "role_id",
                "input_signature_id",
                "universe_root",
                "truth_root",
                "q1_signature_manifest_root",
                "q1_saturation_state_root",
                "q1_class_archive_root",
                "evaluated_class_count",
                "ordered_matching_class_ids",
                "match_count",
                "membership_verdict",
            ],
            "pass_status": "Q2_ROLE_MEMBERSHIP_PASS",
            "complete_fail_status": "Q2_ROLE_MEMBERSHIP_COMPLETE_FAIL",
            "invalidated_status": (
                "Q2_ROLE_MEMBERSHIP_INVALIDATED_TERMINAL_NO_RESUME"
            ),
            "pass_requires": [
                "OUTSIDE_TARGET_MATCH_COUNT_EQUALS_ZERO",
                "IN_LANGUAGE_NULL_MATCH_COUNT_GREATER_THAN_ZERO",
            ],
            "q2_issues_certificate": False,
        },
        "archive_policy": {
            "q1_archives_are_read_only": True,
            "q1_actor_work_product_copy_allowed": False,
            "selected_endpoint_kind": "python",
            "class_stream_only_is_sufficient_for_membership": True,
            "split_required_for_role_membership": False,
        },
        "resource_envelope": {
            "measurement_basis": "TARGET_BLIND_ACTUAL_V8C_Q1_CLASS_REPLAY",
            "measured_odd_class_count": 418,
            "measured_sink_class_count": 89,
            "measured_class_stream_framed_bytes": 907_665,
            "measured_selected_reader_total_bytes": 919_757,
            "measured_peak_rss_bytes": 71_274_496,
            "maximum_class_stream_framed_bytes": 1_048_576,
            "maximum_selected_reader_total_bytes": 1_048_576,
            "maximum_role_rows": 565,
            "maximum_result_bytes": 1_048_576,
            "minimum_fresh_persistent_storage_bytes": 4_194_304,
            "maximum_peak_rss_bytes": 268_435_456,
            "maximum_wall_time_seconds": 180,
            "maximum_internal_popen_count": 0,
            "maximum_actor_start_count": 0,
            "maximum_docker_invocation_count": 0,
        },
        "authority": {
            "q1_state": "COMPLETE",
            "q1_gate_count": Q1_COMPLETE_GATE_COUNT,
            "q1_gate_mask": Q1_COMPLETE_GATE_MASK,
            "q2_state": "NOT_RUN",
            "q2_start_authorized": False,
            "target_truth_access_allowed_now": False,
            "split_access_allowed_now": False,
            "role_evaluation_allowed_now": False,
            "popen_allowed_now": 0,
            "actor_start_allowed_now": 0,
            "docker_invocation_allowed_now": 0,
            "network_allowed": False,
            "adoption_allowed": False,
            "promotion_allowed": False,
            "current_pointer_update_allowed": False,
        },
        "claim_boundary": {
            "q1_predecessor_admission_only": True,
            "q2_membership_result_published": False,
            "outside_certificate_issued": False,
            "current_generic_odd_is_disclosed_development_control": True,
            "untouched_or_independently_hidden_claim_allowed": False,
            "future_result_is_bounded_to_frozen_universes": True,
        },
    }


__all__ = [
    "PREREGISTRATION_ID",
    "PREREGISTRATION_SCHEMA",
    "PREREGISTRATION_STATUS",
    "Q1CompletionAdmissionV1",
    "Q1CompletionBindingV1",
    "Q1_V8C_BINDING",
    "Q2AdmissionError",
    "Q2RoleCommitmentV1",
    "Q2_ROLE_COMMITMENTS",
    "ROLE_MATCH_PROFILE_ID",
    "admit_q1_completion_v1",
    "q2_preregistration_object_v1",
]

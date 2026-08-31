from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hegel_machine import phase3_m25_container_ceremony_v1 as m25
from hegel_machine import phase3_q0_quotient_contract_v1 as q0
from hegel_machine import phase3_q1_completion_replay_successor_v1 as q1_state
from hegel_machine import phase3_q2_admission_v1 as q2


Q2_CONFIG_PATH = ROOT / "config/phase3_q2_role_membership_v1.json"
Q1_V8C_CONFIG_PATH = ROOT / "config/phase3_q1_completion_successor_v8c.json"


def _q2_config() -> dict[str, object]:
    value = json.loads(Q2_CONFIG_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _actual_v8c_payloads() -> tuple[bytes, bytes]:
    config = json.loads(Q1_V8C_CONFIG_PATH.read_text(encoding="utf-8"))
    fixed = config["fixed_paths"]
    terminal_path = Path(fixed["terminal_path"])
    artifact_path = Path(fixed["artifact_path"])
    if not terminal_path.is_file() or not artifact_path.is_file():
        pytest.skip("the external sealed Q1 v8c artifacts are not present")
    return terminal_path.read_bytes(), artifact_path.read_bytes()


def _retag_terminal(
    terminal_payload: bytes,
    field_name: str,
    value: object,
) -> tuple[bytes, str]:
    terminal = q1_state.strict_json_bytes(terminal_payload, "Q1 terminal")
    changed = deepcopy(terminal)
    outcome = changed["outcome"]
    assert type(outcome) is dict
    outcome[field_name] = value
    changed.pop("record_digest")
    digest = "sha256:" + sha256(
        q1_state.DOMAINS["terminal"] + q1_state.canonical_json_bytes(changed)
    ).hexdigest()
    changed["record_digest"] = digest
    return q1_state.canonical_json_bytes(changed), digest


def test_preregistration_config_is_the_exact_target_blind_code_projection() -> None:
    assert _q2_config() == q2.q2_preregistration_object_v1()
    source = Path(q2.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name.endswith(("phase3_m25_rows_v1", "phase3_m25_split_v1"))
        for name in imported
    )
    authority = _q2_config()["authority"]
    assert authority == {
        "q1_state": "COMPLETE",
        "q1_gate_count": 20,
        "q1_gate_mask": 1_048_575,
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
    }


def test_role_commitments_match_frozen_roots_and_matcher_identity() -> None:
    assert q2.ROLE_MATCH_PROFILE_ID == q0.ROLE_MATCH_PROFILE_ID
    odd, sink = q2.Q2_ROLE_COMMITMENTS
    assert odd.universe_root == "sha256:" + m25.ODD_UNIVERSE_ROOT.hex()
    assert odd.truth_root == "sha256:" + m25.ODD_TRUTH_ROOT.hex()
    assert sink.universe_root == "sha256:" + m25.SINK_UNIVERSE_ROOT.hex()
    assert sink.truth_root == "sha256:" + m25.SINK_TRUTH_ROOT.hex()
    assert (odd.role_id, odd.role_name, odd.row_count) == (
        1,
        "OUTSIDE_TARGET",
        480,
    )
    assert (sink.role_id, sink.role_name, sink.row_count) == (
        2,
        "IN_LANGUAGE_NULL",
        85,
    )


def test_actual_v8c_terminal_and_receipt_admit_exactly_eight_slots() -> None:
    terminal, artifact = _actual_v8c_payloads()
    admitted = q2.admit_q1_completion_v1(terminal, artifact)
    assert admitted.binding == q2.Q1_V8C_BINDING
    assert tuple(row[0] for row in admitted.ordered_output_slots) == tuple(range(1, 9))
    assert tuple("sha256:" + row[2].hex() for row in admitted.ordered_output_slots) == (
        q2.Q1_V8C_BINDING.ordered_q1_output_slots
    )


def test_actual_v8c_admission_rejects_changed_terminal_or_artifact_bytes() -> None:
    terminal, artifact = _actual_v8c_payloads()
    with pytest.raises(q2.Q2AdmissionError, match="terminal"):
        q2.admit_q1_completion_v1(terminal.replace(b"20", b"21", 1), artifact)
    changed_artifact = bytearray(artifact)
    changed_artifact[-1] ^= 1
    with pytest.raises(q2.Q2AdmissionError, match="artifact digest"):
        q2.admit_q1_completion_v1(terminal, bytes(changed_artifact))


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("q1_gate_count", 19),
        ("q1_gate_mask", 524_287),
        ("q2_state", "COMPLETE"),
        ("target_truth_accessed", True),
        ("split_accessed", True),
        ("role_evaluation_performed", True),
    ),
)
def test_admission_rejects_changed_q1_success_projection(
    field_name: str,
    value: object,
) -> None:
    terminal, artifact = _actual_v8c_payloads()
    changed_terminal, changed_digest = _retag_terminal(terminal, field_name, value)
    binding = replace(q2.Q1_V8C_BINDING, terminal_record_digest=changed_digest)
    with pytest.raises(q2.Q2AdmissionError, match=field_name):
        q2.admit_q1_completion_v1(changed_terminal, artifact, binding=binding)


def test_admission_rejects_receipt_slots_outside_the_frozen_binding() -> None:
    terminal, artifact = _actual_v8c_payloads()
    slots = list(q2.Q1_V8C_BINDING.ordered_q1_output_slots)
    slots[0] = "sha256:" + "00" * 32
    binding = replace(q2.Q1_V8C_BINDING, ordered_q1_output_slots=tuple(slots))
    with pytest.raises(q2.Q2AdmissionError, match="output-slot roots"):
        q2.admit_q1_completion_v1(terminal, artifact, binding=binding)

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine import phase3_q1_formal_resource_amendment_v1 as amendment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_q1_formal_resource_amendment_v1.py"
CONFIG_PATH = PROJECT_ROOT / "config/phase3_q1_formal_resource_amendment_v1.json"

COUNT_HARD_CEILINGS = {
    "maximum_raw_operator_applications": 4_294_967_295,
    "maximum_behavior_classes": 16_777_216,
    "maximum_visible_frontier_points": 33_554_432,
    "maximum_visible_frontier_points_per_class": 65_536,
    "maximum_continuation_bank_points": 67_108_864,
    "maximum_continuation_bank_points_per_class": 65_536,
    "maximum_work_queue_points": 67_108_864,
    "maximum_saturation_rounds": 16,
}
RESOURCE_HARD_CEILINGS = {
    "memory_bytes": 15_032_385_536,
    "output_bytes": 34_359_738_368,
    "scratch_bytes": 51_539_607_552,
    "wall_time_seconds": 172_800,
}


def _partition(
    signature_id: int,
    signature_name: str,
    *,
    raw: int,
    classes: int,
    frontier: int,
    final_frontier_per_class: int,
    peak_frontier_per_class: int,
    bank: int,
    bank_per_class: int,
    work_queue: int,
    rounds: int,
) -> dict[str, object]:
    return {
        "fixed_point": {
            "no_resource_guard_hit": True,
            "saturated_diagnostic_only": True,
        },
        "full_v16_structural_limits_applied": True,
        "input_signature_id": signature_id,
        "input_signature_name": signature_name,
        # This final value is deliberately lower than the accepted-state
        # high-water for Sink.  It must never drive the formal guard.
        "maximum_frontier_points_per_class": final_frontier_per_class,
        "peak_behavior_class_count": classes,
        "peak_continuation_bank_point_count": bank,
        "peak_continuation_bank_points_per_class": bank_per_class,
        "peak_raw_operator_application_count": raw,
        "peak_saturation_round_count": rounds,
        "peak_visible_frontier_point_count": frontier,
        "peak_visible_frontier_points_per_class": peak_frontier_per_class,
        "peak_work_queue_points": work_queue,
        "resource_guard_id": None,
        "resource_guard_name": None,
        "terminal_status": amendment.V6_TERMINAL_STATUS,
    }


def _endpoint(actor_id: str) -> dict[str, object]:
    return {
        "action_id": amendment.PREDECESSOR_ACTION_ID,
        "actor_id": actor_id,
        "actual_preflight_peak_scratch_bytes": (
            40_452_166 if actor_id == "PYTHON_ENDPOINT" else 3_194_598
        ),
        "capsule_digest": amendment.V6_CAPSULE_DIGEST,
        "diagnostic_only": True,
        "partitions": [
            _partition(
                1,
                "OddInputV1",
                raw=11_846,
                classes=418,
                frontier=1_098,
                final_frontier_per_class=14,
                peak_frontier_per_class=14,
                bank=3_611,
                bank_per_class=1_175,
                work_queue=8_150,
                rounds=5,
            ),
            _partition(
                2,
                "SinkInputV1",
                raw=12_908,
                classes=89,
                frontier=372,
                final_frontier_per_class=84,
                peak_frontier_per_class=124,
                bank=3_277,
                bank_per_class=1_667,
                work_queue=7_774,
                rounds=5,
            ),
        ],
        "projected_formal_archive_payload_bytes": 4_742_060,
        "projected_formal_total_output_bytes": 5_790_636,
        "projected_peak_formal_scratch_bytes": 1_646_592,
        "q1_execution_started": False,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "role_evaluation_performed": False,
        "source_commit": amendment.V6_SOURCE_COMMIT,
        "split_accessed": False,
        "target_truth_accessed": False,
        "terminal_status": amendment.V6_TERMINAL_STATUS,
    }


def _artifact() -> dict[str, object]:
    return {
        "action_id": amendment.PREDECESSOR_ACTION_ID,
        "capsule_digest": amendment.V6_CAPSULE_DIGEST,
        "diagnostic_only": True,
        "host_replay": {
            "capsule_digest": amendment.V6_CAPSULE_DIGEST,
            "diagnostic_only": True,
            "dual_semantic_agreement_sha256": "a" * 64,
            "endpoint_rows": [],
            "formal_resource_projection": {
                "projected_formal_total_output_bytes": 5_790_636,
                "projected_peak_formal_scratch_bytes": 1_646_592,
            },
            "projected_host_replay_output_bytes": 1_048_576,
            "projected_peak_host_replay_scratch_bytes": 0,
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_output_slots": [None] * 8,
            "q1_state": "NOT_RUN",
            "registered_resource_guard_id": None,
            "schedule_high_water_aggregation": {
                "peak_saturation_round_count_across_endpoints_and_signatures": 5,
                "peak_work_queue_points_by_input_signature": [8_150, 7_774],
            },
            "source_commit": amendment.V6_SOURCE_COMMIT,
            "terminal_status": amendment.V6_TERMINAL_STATUS,
        },
        "host_replay_passed": True,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "resource_adjudication": {
            "derived_envelope": {
                "endpoint_memory_guard_bytes": 2_147_483_648,
                "endpoint_output_guard_bytes": 16_777_216,
                "endpoint_scratch_guard_bytes": 4_194_304,
                "endpoint_wall_guard_seconds": 3_600,
                "evaluated": True,
                "host_memory_guard_bytes": 67_108_864,
                "host_output_guard_bytes": 2_097_152,
                "host_scratch_guard_bytes": 1,
                "host_wall_guard_seconds": 3_600,
            },
            "direct_endpoint_resources": [
                {
                    "actor_id": "PYTHON_ENDPOINT",
                    "actual_diagnostic_output_bytes": 49_900,
                    "actual_peak_rss_bytes": 112_295_936,
                    "actual_preflight_peak_scratch_bytes": 40_452_166,
                    "actual_wall_time_ns": 729_612_002_970,
                },
                {
                    "actor_id": "RUST_ENDPOINT",
                    "actual_diagnostic_output_bytes": 49_906,
                    "actual_peak_rss_bytes": 540_782_592,
                    "actual_preflight_peak_scratch_bytes": 3_194_598,
                    "actual_wall_time_ns": 7_702_460_263,
                },
            ],
            "first_resource_guard_id": None,
            "first_resource_guard_name": None,
            "host_replay_actual_resources": {
                "actual_peak_rss_bytes": 17_690_624,
                "actual_wall_time_ns": 1_469_117_302,
            },
            "host_replay_passed": True,
            "infeasible_derived_fields": [],
            "input_terminal_status": amendment.V6_TERMINAL_STATUS,
            "terminal_status": amendment.V6_TERMINAL_STATUS,
        },
        "terminal_status": amendment.V6_TERMINAL_STATUS,
    }


def _bound_payloads(
    *,
    artifact: dict[str, object] | None = None,
    python: dict[str, object] | None = None,
    rust: dict[str, object] | None = None,
) -> tuple[bytes, bytes, bytes, str]:
    artifact = deepcopy(_artifact() if artifact is None else artifact)
    endpoints = {
        "PYTHON_ENDPOINT": deepcopy(
            _endpoint("PYTHON_ENDPOINT") if python is None else python
        ),
        "RUST_ENDPOINT": deepcopy(
            _endpoint("RUST_ENDPOINT") if rust is None else rust
        ),
    }
    endpoint_payloads = {
        actor: amendment.canonical_json_bytes_v1(value)
        for actor, value in endpoints.items()
    }
    host_replay = artifact["host_replay"]
    assert isinstance(host_replay, dict)
    resource_adjudication = artifact["resource_adjudication"]
    assert isinstance(resource_adjudication, dict)
    direct_rows = resource_adjudication["direct_endpoint_resources"]
    assert isinstance(direct_rows, list)
    direct_by_actor = {row["actor_id"]: row for row in direct_rows}
    host_replay["endpoint_rows"] = [
        {
            "actor_id": actor,
            "actual_diagnostic_output_bytes": direct_by_actor[actor][
                "actual_diagnostic_output_bytes"
            ],
            "actual_preflight_peak_scratch_bytes": direct_by_actor[actor][
                "actual_preflight_peak_scratch_bytes"
            ],
            "peak_saturation_round_count_by_input_signature": [
                row["peak_saturation_round_count"]
                for row in endpoints[actor]["partitions"]
            ],
            "peak_work_queue_points_by_input_signature": [
                row["peak_work_queue_points"]
                for row in endpoints[actor]["partitions"]
            ],
            "stdout_sha256": sha256(endpoint_payloads[actor]).hexdigest(),
        }
        for actor in ("PYTHON_ENDPOINT", "RUST_ENDPOINT")
    ]
    schedule = host_replay["schedule_high_water_aggregation"]
    assert isinstance(schedule, dict)
    schedule["peak_saturation_round_count_across_endpoints_and_signatures"] = max(
        row["peak_saturation_round_count"]
        for endpoint in endpoints.values()
        for row in endpoint["partitions"]
    )
    schedule["peak_work_queue_points_by_input_signature"] = [
        max(
            endpoint["partitions"][index]["peak_work_queue_points"]
            for endpoint in endpoints.values()
        )
        for index in range(2)
    ]
    artifact_payload = amendment.canonical_json_bytes_v1(artifact)
    return (
        artifact_payload,
        endpoint_payloads["PYTHON_ENDPOINT"],
        endpoint_payloads["RUST_ENDPOINT"],
        f"sha256:{sha256(artifact_payload).hexdigest()}",
    )


def _derive(
    artifact_payload: bytes,
    python_payload: bytes,
    rust_payload: bytes,
    artifact_sha256: str,
) -> dict[str, object]:
    return amendment.derive_formal_resource_amendment_v1(
        artifact_payload,
        python_payload,
        rust_payload,
        expected_artifact_sha256=artifact_sha256,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    ((0, 1), (1, 1), (2, 2), (3, 4), (11_846 * 2, 32_768)),
)
def test_preregistered_pow2ceil_formula(value: int, expected: int) -> None:
    assert amendment.pow2ceil_v1(value) == expected


@pytest.mark.parametrize(
    ("elapsed_ns", "expected"),
    (
        (0, 0),
        (1, 3_600),
        (1_200_000_000_000, 3_600),
        (1_200_000_000_001, 7_200),
        (729_612_002_970, 3_600),
    ),
)
def test_preregistered_wall_formula(elapsed_ns: int, expected: int) -> None:
    assert amendment.hour_rounded_wall_guard_seconds_v1(elapsed_ns) == expected


def test_exact_v6_formula_known_answer_and_sink_high_water() -> None:
    artifact, python, rust, artifact_sha256 = _bound_payloads()
    result = _derive(artifact, python, rust, artifact_sha256)

    assert result["status"] == (
        "FORMAL_RESOURCE_ENVELOPE_DERIVED_Q1_NOT_RUN_0_OF_20"
    )
    rows = result["per_signature_formal_count_guards"]
    assert isinstance(rows, list)
    assert rows == [
        {
            "formal_guard": {
                "maximum_behavior_classes": 1_024,
                "maximum_continuation_bank_points": 8_192,
                "maximum_continuation_bank_points_per_class": 4_096,
                "maximum_raw_operator_applications": 32_768,
                "maximum_saturation_rounds": 7,
                "maximum_visible_frontier_points": 4_096,
                "maximum_visible_frontier_points_per_class": 32,
                "maximum_work_queue_points": 16_384,
            },
            "input_signature_id": 1,
            "input_signature_name": "OddInputV1",
            "observed_high_water": {
                "peak_behavior_class_count": 418,
                "peak_continuation_bank_point_count": 3_611,
                "peak_continuation_bank_points_per_class": 1_175,
                "peak_raw_operator_application_count": 11_846,
                "peak_saturation_round_count": 5,
                "peak_visible_frontier_point_count": 1_098,
                "peak_visible_frontier_points_per_class": 14,
                "peak_work_queue_points": 8_150,
            },
        },
        {
            "formal_guard": {
                "maximum_behavior_classes": 256,
                "maximum_continuation_bank_points": 8_192,
                "maximum_continuation_bank_points_per_class": 4_096,
                "maximum_raw_operator_applications": 32_768,
                "maximum_saturation_rounds": 7,
                "maximum_visible_frontier_points": 1_024,
                "maximum_visible_frontier_points_per_class": 256,
                "maximum_work_queue_points": 16_384,
            },
            "input_signature_id": 2,
            "input_signature_name": "SinkInputV1",
            "observed_high_water": {
                "peak_behavior_class_count": 89,
                "peak_continuation_bank_point_count": 3_277,
                "peak_continuation_bank_points_per_class": 1_667,
                "peak_raw_operator_application_count": 12_908,
                "peak_saturation_round_count": 5,
                "peak_visible_frontier_point_count": 372,
                "peak_visible_frontier_points_per_class": 124,
                "peak_work_queue_points": 7_774,
            },
        },
    ]
    # The Sink terminal value is 84, but the accepted-state high-water is 124.
    sink_input = json.loads(rust)["partitions"][1]
    assert sink_input["maximum_frontier_points_per_class"] == 84
    assert rows[1]["observed_high_water"][
        "peak_visible_frontier_points_per_class"
    ] == 124
    assert rows[1]["formal_guard"][
        "maximum_visible_frontier_points_per_class"
    ] == 256

    assert result["formal_resource_envelope"] == {
        "endpoint": {
            "memory_bytes": 2_147_483_648,
            "output_bytes": 16_777_216,
            "scratch_bytes": 4_194_304,
            "wall_time_seconds": 3_600,
        },
        "trusted_host_replay": {
            "memory_bytes": 67_108_864,
            "output_bytes": 2_097_152,
            "scratch_bytes": 1,
            "wall_time_seconds": 3_600,
        },
    }


def test_diagnostic_output_and_scratch_do_not_replace_formal_projection() -> None:
    artifact, python, rust, artifact_sha256 = _bound_payloads()
    baseline = _derive(artifact, python, rust, artifact_sha256)

    artifact_value = json.loads(artifact)
    python_value = json.loads(python)
    rust_value = json.loads(rust)
    python_value["actual_preflight_peak_scratch_bytes"] = 1
    rust_value["actual_preflight_peak_scratch_bytes"] = 2
    direct_rows = artifact_value["resource_adjudication"][
        "direct_endpoint_resources"
    ]
    direct_rows[0]["actual_diagnostic_output_bytes"] = 3
    direct_rows[0]["actual_preflight_peak_scratch_bytes"] = 1
    direct_rows[1]["actual_diagnostic_output_bytes"] = 4
    direct_rows[1]["actual_preflight_peak_scratch_bytes"] = 2
    changed = _bound_payloads(
        artifact=artifact_value,
        python=python_value,
        rust=rust_value,
    )
    candidate = _derive(*changed)

    assert candidate["formal_resource_envelope"] == baseline[
        "formal_resource_envelope"
    ]
    assert candidate["resource_observations"] == baseline[
        "resource_observations"
    ]
    assert candidate["per_signature_formal_count_guards"] == baseline[
        "per_signature_formal_count_guards"
    ]


def test_dual_schedule_uses_per_signature_queue_max_and_global_round_max() -> None:
    python_value = _endpoint("PYTHON_ENDPOINT")
    rust_value = _endpoint("RUST_ENDPOINT")
    python_value["partitions"][0]["peak_work_queue_points"] = 8_100
    python_value["partitions"][1]["peak_work_queue_points"] = 7_000
    rust_value["partitions"][0]["peak_work_queue_points"] = 8_000
    rust_value["partitions"][1]["peak_work_queue_points"] = 7_774
    python_value["partitions"][0]["peak_saturation_round_count"] = 3
    python_value["partitions"][1]["peak_saturation_round_count"] = 4
    rust_value["partitions"][0]["peak_saturation_round_count"] = 5
    rust_value["partitions"][1]["peak_saturation_round_count"] = 2

    candidate = _derive(
        *_bound_payloads(python=python_value, rust=rust_value)
    )
    rows = candidate["per_signature_formal_count_guards"]
    assert [
        row["observed_high_water"]["peak_work_queue_points"] for row in rows
    ] == [8_100, 7_774]
    assert [
        row["formal_guard"]["maximum_work_queue_points"] for row in rows
    ] == [16_384, 16_384]
    assert [
        row["observed_high_water"]["peak_saturation_round_count"]
        for row in rows
    ] == [5, 5]
    assert [
        row["formal_guard"]["maximum_saturation_rounds"] for row in rows
    ] == [7, 7]


def test_artifact_and_small_stdout_hash_bindings_fail_closed() -> None:
    artifact, python, rust, artifact_sha256 = _bound_payloads()

    mutated_artifact = json.loads(artifact)
    mutated_artifact["diagnostic_only"] = False
    with pytest.raises(amendment.FormalResourceAmendmentError) as artifact_error:
        _derive(
            amendment.canonical_json_bytes_v1(mutated_artifact),
            python,
            rust,
            artifact_sha256,
        )
    assert artifact_error.value.code == "REJECT_Q1_RESOURCE_ARTIFACT_BINDING"

    mutated_python = json.loads(python)
    mutated_python["actual_preflight_peak_scratch_bytes"] += 1
    with pytest.raises(amendment.FormalResourceAmendmentError) as stdout_error:
        _derive(
            artifact,
            amendment.canonical_json_bytes_v1(mutated_python),
            rust,
            artifact_sha256,
        )
    assert stdout_error.value.code == "REJECT_Q1_RESOURCE_STDOUT_BINDING"


def test_candidate_is_within_hard_ceilings_and_keeps_q1_closed() -> None:
    candidate = _derive(*_bound_payloads())

    for row in candidate["per_signature_formal_count_guards"]:
        for name, value in row["formal_guard"].items():
            assert value <= COUNT_HARD_CEILINGS[name]
    for envelope in candidate["formal_resource_envelope"].values():
        for name, value in envelope.items():
            assert value <= RESOURCE_HARD_CEILINGS[name]

    assert candidate["q1_state"] == "NOT_RUN"
    assert candidate["q1_execution_started"] is False
    assert candidate["q1_gate_count"] == 0
    assert candidate["q1_gate_mask"] == 0
    assert candidate["q1_output_slots"] == [None] * 8
    assert candidate["q1_formal_roots"] is None
    assert candidate["q1_receipt"] is None
    assert candidate["target_truth_accessed"] is False
    assert candidate["split_accessed"] is False
    assert candidate["role_evaluation_performed"] is False
    assert candidate["candidate_only"] is True
    assert candidate[
        "all_derived_values_within_preregistered_hard_ceilings"
    ] is True
    assert candidate["infeasible_derived_fields"] == []
    assert candidate["authoritative_apply_requires_new_exact_capsule"] is True
    assert candidate["authoritative_apply_capsule_digest"] is None
    assert candidate["network_accessed"] is False
    assert candidate["popen_count"] == 0
    assert candidate["run_genesis"] is None


def test_machine_candidate_config_matches_the_mechanical_result() -> None:
    candidate = _derive(*_bound_payloads())
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["schema_version"] == amendment.SCHEMA_VERSION
    assert config["authority"] == {
        "active_transition_allowed": False,
        "adoption_allowed": False,
        "current_pointer_write_allowed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "promotion_allowed": False,
        "q1_gate_count": candidate["q1_gate_count"],
        "q1_gate_mask": candidate["q1_gate_mask"],
        "q1_gate_total": 20,
        "q1_output_slots": candidate["q1_output_slots"],
        "q1_prestart_admission_receipt": None,
        "q1_receipt": candidate["q1_receipt"],
        "q1_run_genesis": candidate["run_genesis"],
        "q1_state": candidate["q1_state"],
        "q2_state": "NOT_RUN",
        "role_evaluation_performed": candidate["role_evaluation_performed"],
        "split_accessed": candidate["split_accessed"],
        "target_truth_accessed": candidate["target_truth_accessed"],
    }
    configured = config["formal_resource_envelope_candidate"]
    assert configured["all_derived_values_within_preregistered_hard_ceilings"] is (
        candidate["all_derived_values_within_preregistered_hard_ceilings"]
    )
    assert configured["infeasible_derived_fields"] == candidate[
        "infeasible_derived_fields"
    ]
    assert configured["endpoint_formal_guards"] == {
        "memory_bytes": candidate["formal_resource_envelope"]["endpoint"][
            "memory_bytes"
        ],
        "output_bytes": candidate["formal_resource_envelope"]["endpoint"][
            "output_bytes"
        ],
        "scratch_bytes": candidate["formal_resource_envelope"]["endpoint"][
            "scratch_bytes"
        ],
        "wall_time_seconds": candidate["formal_resource_envelope"]["endpoint"][
            "wall_time_seconds"
        ],
    }
    assert configured["trusted_host_replay"]["formal_guards"] == candidate[
        "formal_resource_envelope"
    ]["trusted_host_replay"]
    for configured_row, candidate_row in zip(
        configured["per_input_signature"],
        candidate["per_signature_formal_count_guards"],
        strict=True,
    ):
        assert configured_row["input_signature_id"] == candidate_row[
            "input_signature_id"
        ]
        guard = candidate_row["formal_guard"]
        assert configured_row["formal_count_guards"] == {
            "raw_operator_applications": guard[
                "maximum_raw_operator_applications"
            ],
            "behavior_classes": guard["maximum_behavior_classes"],
            "visible_frontier_total": guard["maximum_visible_frontier_points"],
            "visible_frontier_per_class": guard[
                "maximum_visible_frontier_points_per_class"
            ],
            "continuation_bank_total": guard[
                "maximum_continuation_bank_points"
            ],
            "continuation_bank_per_class": guard[
                "maximum_continuation_bank_points_per_class"
            ],
            "work_queue_points": guard["maximum_work_queue_points"],
        }
    assert configured["saturation_rounds"]["formal_guard"] == candidate[
        "per_signature_formal_count_guards"
    ][0]["formal_guard"]["maximum_saturation_rounds"]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "phase3_q1_formal_resource_amendment_tool_test", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_preflight_is_canonical_and_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    artifact, python, rust, artifact_sha256 = _bound_payloads()
    paths = {
        "artifact": tmp_path / "artifact.json",
        "python": tmp_path / "python.stdout",
        "rust": tmp_path / "rust.stdout",
    }
    for path, payload in zip(paths.values(), (artifact, python, rust), strict=True):
        path.write_bytes(payload)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in paths.values()
    }

    def derive_with_synthetic_binding(
        artifact_payload: bytes,
        python_payload: bytes,
        rust_payload: bytes,
    ) -> dict[str, object]:
        return amendment.derive_formal_resource_amendment_v1(
            artifact_payload,
            python_payload,
            rust_payload,
            expected_artifact_sha256=artifact_sha256,
        )

    monkeypatch.setattr(
        tool._CORE,
        "derive_formal_resource_amendment_v1",
        derive_with_synthetic_binding,
    )
    assert tool.main(
        [
            "--preflight",
            "--artifact",
            str(paths["artifact"]),
            "--python-stdout",
            str(paths["python"]),
            "--rust-stdout",
            str(paths["rust"]),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    value = json.loads(captured.out)
    assert captured.out.encode("ascii") == amendment.canonical_json_bytes_v1(value)
    assert value["status"] == amendment.STATUS
    assert value["popen_count"] == 0
    assert value["network_accessed"] is False
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in paths.values()
    }
    assert after == before
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(before)

#!/usr/bin/env python3
"""Isolated Python endpoint for the Phase-3A-Q1 capacity preflight.

The endpoint must be executed with ``python -I -S -B`` from a read-only
target-blind source snapshot.  This preregistration version deliberately
rejects the no-argument full run because the archive projection wire, Rust
endpoint, supervisor, and execution manifest are not frozen yet.
``--local-subset-node-count`` exists only for engineering qualification and
can never emit the full-preflight success status.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Final, NoReturn


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = PROJECT_ROOT / "src"
PACKAGE_ROOT: Final = SOURCE_ROOT / "hegel_machine"
CONFIG_PATH: Final = PROJECT_ROOT / "config/phase3_q1_capacity_preflight_v1.json"

ENDPOINT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-python-capacity-preflight-endpoint/1"
)
ERROR_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-python-capacity-preflight-endpoint-error/1"
)
IMPLEMENTATION_ID: Final = "HEGEL_Q1_CAPACITY_PREFLIGHT_PYTHON_V1"

MODULE_NAMES: Final = (
    "phase3_m3_bounded_enumerator_shrink2_v1",
    "phase3_m3_bounded_enumerator_shrink3_v1",
    "phase3_m3_bounded_enumerator_shrink4_v1",
    "phase3_m3_bounded_enumerator_shrink5_v1",
    "phase3_m3_bounded_enumerator_shrink6_v1",
    "phase3_m3_bounded_enumerator_v1",
    "phase3_m3_dsl_core_v1",
    "phase3_m3_record_wire_v1",
    "phase3_m3_shrink1_core_v1",
    "phase3_m3_shrink2_core_v1",
    "phase3_m3_shrink3_core_v1",
    "phase3_m3_shrink4_core_v1",
    "phase3_m3_shrink5_core_v1",
    "phase3_m3_shrink6_core_v1",
    "phase3_q0_evaluator_v1",
    "phase3_q0_input_adapter_v1",
    "phase3_q1_capacity_preflight_v1",
    "phase3_q1_quotient_contract_v1",
    "phase3_q1_universe_v1",
    "strict_ast_shrink1_v1",
    "strict_ast_shrink2_v1",
    "strict_ast_shrink3_v1",
    "strict_ast_shrink4_v1",
    "strict_ast_shrink5_v1",
    "strict_ast_shrink6_v1",
    "strict_ast_v1",
    "strict_cbor_v1",
)
MODULE_SOURCE_PATHS: Final = tuple(
    f"src/hegel_machine/{name}.py" for name in MODULE_NAMES
)
IMPLEMENTATION_SOURCE_PATHS: Final = tuple(
    sorted(
        MODULE_SOURCE_PATHS
        + (
            "config/phase3_q1_capacity_preflight_v1.json",
            "tools/phase3_q1_python_capacity_preflight_entrypoint_v1.py",
        )
    )
)
FORBIDDEN_MODULES: Final = (
    "hegel_machine.__init__",
    "hegel_machine.phase3_dsl_v1",
    "hegel_machine.phase3_m25_rows_v1",
    "hegel_machine.phase3_m25_split_v1",
    "hegel_machine.phase3_m25_formal_static_basis_v1",
    "hegel_machine.phase3_q0_quotient_contract_v1",
)


class EndpointError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise EndpointError(code, detail)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail("FAIL_Q1_CONFIG_WIRE", f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> NoReturn:
    _fail("FAIL_Q1_CONFIG_WIRE", f"non-finite JSON token {value!r}")


def _load_config() -> tuple[dict[str, object], bytes]:
    payload = CONFIG_PATH.read_bytes()
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_Q1_CONFIG_WIRE", str(error))
    if type(value) is not dict:
        _fail("FAIL_Q1_CONFIG_WIRE", "configuration must be one JSON object")
    return value, payload


# Never execute the historical package initializer: it exports target and
# control objects.  A private empty package namespace admits only the exact
# allowlisted dependency closure below.
sys.path.insert(0, str(SOURCE_ROOT))
if "hegel_machine" in sys.modules:
    _fail("FAIL_Q1_PACKAGE_ISOLATION", "hegel_machine existed before bootstrap")
package = ModuleType("hegel_machine")
package.__path__ = [str(PACKAGE_ROOT)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package

from hegel_machine import phase3_q1_capacity_preflight_v1 as preflight  # noqa: E402
from hegel_machine import phase3_q1_universe_v1 as universe  # noqa: E402


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _source_file_set_digest() -> str:
    digest = sha256()
    for relative in IMPLEMENTATION_SOURCE_PATHS:
        payload = (PROJECT_ROOT / relative).read_bytes()
        path_bytes = relative.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _validate_config(config: dict[str, object]) -> None:
    def exact(path: tuple[str, ...], expected: object) -> None:
        value: object = config
        for key in path:
            if type(value) is not dict or key not in value:
                _fail("FAIL_Q1_CONFIG_BINDING", ".".join(path) + " is absent")
            value = value[key]
        if type(value) is not type(expected) or value != expected:
            _fail(
                "FAIL_Q1_CONFIG_BINDING",
                f"{'.'.join(path)} differs: {value!r}",
            )

    exact(("schema_version",), "hegel-phase3a-q1-capacity-preflight-preregistration/1")
    exact(("preflight_id",), preflight.PREFLIGHT_ID)
    exact(("dsl_version",), preflight.DSL_VERSION)
    exact(("closure_semantics_version",), preflight.CLOSURE_SEMANTICS_VERSION)
    exact(("preflight_status",), "PREREGISTERED_NOT_RUN")
    exact(("authority", "q1_state"), "NOT_RUN")
    exact(("authority", "q1_gate_count"), 0)
    exact(("authority", "q1_gate_mask"), 0)
    exact(("authority", "q1_formal_roots"), None)
    exact(("authority", "q1_receipt"), None)
    exact(("authority", "q2_state"), "NOT_RUN")
    exact(("authority", "m3_formal_roots"), None)
    exact(("authority", "outside_certificate_issued"), False)
    exact(("authority", "active_transition_allowed"), False)
    exact(("input_isolation", "target_truth_accessed"), False)
    exact(("input_isolation", "split_accessed"), False)
    exact(("input_isolation", "role_evaluation_performed"), False)
    exact(("endpoint_schedule", "signature_order_within_each_endpoint"), [1, 2])
    exact(("endpoint_schedule", "signatures_execute_concurrently_within_one_endpoint"), False)
    exact(
        ("authoritative_preflight_admission", "allowed_current_execution"),
        "IMPORT_ISOLATED_LOCAL_SUBSET_QUALIFICATION_ONLY",
    )
    exact(
        ("authoritative_preflight_admission", "full_dual_node6_preflight_allowed_now"),
        False,
    )
    exact(("structural_limits", "maximum_ast_depth"), 3)
    exact(("structural_limits", "maximum_ast_node_count"), 6)

    envelope = config.get("provisional_hard_envelope")
    if type(envelope) is not dict:
        _fail("FAIL_Q1_CONFIG_BINDING", "provisional_hard_envelope is absent")
    expected_limits = preflight.PreflightLimitsV1()
    bindings = {
        "max_raw_operator_applications_per_signature": (
            expected_limits.maximum_raw_operator_applications
        ),
        "max_behavior_classes_per_signature": (
            expected_limits.maximum_behavior_classes
        ),
        "max_continuation_bank_points_per_signature": (
            expected_limits.maximum_continuation_bank_points
        ),
        "max_continuation_bank_points_per_class": (
            expected_limits.maximum_continuation_bank_points_per_class
        ),
        "max_visible_frontier_points_per_signature": (
            expected_limits.maximum_visible_frontier_points
        ),
        "max_visible_frontier_points_per_class": (
            expected_limits.maximum_visible_frontier_points_per_class
        ),
        "max_work_queue_points_per_signature": (
            expected_limits.maximum_work_queue_points
        ),
        "max_saturation_rounds_per_signature": (
            expected_limits.maximum_saturation_rounds
        ),
        "max_wall_time_seconds_per_endpoint": (
            expected_limits.maximum_wall_time_seconds
        ),
    }
    for key, expected in bindings.items():
        if type(envelope.get(key)) is not int or envelope[key] != expected:
            _fail("FAIL_Q1_CONFIG_BINDING", f"envelope {key} differs")

    registry = config.get("resource_guard_registry")
    expected_registry = [[guard_id, name] for guard_id, name in preflight.RESOURCE_GUARD_REGISTRY]
    if registry != expected_registry:
        _fail("FAIL_Q1_CONFIG_BINDING", "resource guard registry differs")

    universe_rows = config.get("universes")
    if type(universe_rows) is not list or len(universe_rows) != 2:
        _fail("FAIL_Q1_CONFIG_BINDING", "exactly two universes are required")
    generated = universe.all_production_universes_v1()
    for row, item in zip(universe_rows, generated, strict=True):
        if type(row) is not dict:
            _fail("FAIL_Q1_CONFIG_BINDING", "universe binding must be an object")
        expected = {
            "historical_payload_universe_root": (
                "sha256:" + item.universe_root.hex()
            ),
            "input_signature_id": item.input_signature_id,
            "row_count": len(item.rows),
            "truth_root_in_preflight": None,
        }
        for key, expected_value in expected.items():
            if type(row.get(key)) is not type(expected_value) or row.get(key) != expected_value:
                _fail("FAIL_Q1_CONFIG_BINDING", f"universe {key} differs")


def _loaded_project_modules() -> tuple[str, ...]:
    loaded: list[str] = []
    expected = {f"hegel_machine.{name}" for name in MODULE_NAMES}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("hegel_machine."):
            continue
        path_value = getattr(module, "__file__", None)
        if type(path_value) is not str:
            _fail("FAIL_Q1_PACKAGE_ISOLATION", f"{name} has no regular source path")
        path = Path(path_value).resolve()
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            _fail("FAIL_Q1_PACKAGE_ISOLATION", f"{name} escaped source root")
        if name not in expected or relative not in MODULE_SOURCE_PATHS:
            _fail(
                "FAIL_Q1_PACKAGE_ISOLATION",
                f"unallowlisted project module loaded: {name} ({relative})",
            )
        loaded.append(name)
    if set(loaded) != expected:
        missing = sorted(expected - set(loaded))
        _fail("FAIL_Q1_PACKAGE_ISOLATION", f"allowlisted modules not loaded: {missing}")
    forbidden = [name for name in FORBIDDEN_MODULES if name in sys.modules]
    if forbidden:
        _fail("FAIL_Q1_PACKAGE_ISOLATION", f"forbidden modules loaded: {forbidden}")
    return tuple(sorted(loaded))


def endpoint_object(local_subset_node_count: int | None) -> dict[str, object]:
    config, config_bytes = _load_config()
    _validate_config(config)
    if local_subset_node_count is None:
        admission = config.get("authoritative_preflight_admission")
        if type(admission) is not dict or admission.get(
            "full_dual_node6_preflight_allowed_now"
        ) is not False:
            _fail(
                "FAIL_Q1_PREFLIGHT_ADMISSION",
                "unexpected authoritative-preflight admission state",
            )
        _fail(
            "Q1_FULL_PREFLIGHT_NOT_ADMITTED",
            "archive wire, Rust endpoint, dual supervisor, source commit, and execution manifest are not yet qualified",
        )
    limits = (
        preflight.PreflightLimitsV1()
        if local_subset_node_count is None
        else preflight.PreflightLimitsV1(
            maximum_ast_node_count=local_subset_node_count
        )
    )
    result = preflight.run_q1_capacity_preflight_v1(limits=limits)
    engine_object = preflight.capacity_preflight_diagnostic_object_v1(result)
    engine_bytes = preflight.canonical_capacity_preflight_json_bytes_v1(result)
    loaded = _loaded_project_modules()
    return {
        "active_transition_allowed": False,
        "config_sha256": "sha256:" + sha256(config_bytes).hexdigest(),
        "dual_agreement_claimed": False,
        "endpoint_schema_version": ENDPOINT_SCHEMA_VERSION,
        "engine_diagnostic": engine_object,
        "engine_diagnostic_json_sha256": "sha256:" + sha256(engine_bytes).hexdigest(),
        "formal_roots_generated": False,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_source_file_set_digest": _source_file_set_digest(),
        "implementation_source_paths": list(IMPLEMENTATION_SOURCE_PATHS),
        "import_allowlist_isolation_passed": True,
        "loaded_project_modules": list(loaded),
        "m3_formal_roots": None,
        "normal_package_initializer_executed": False,
        "outside_certificate_issued": False,
        "preregistered_full_limits_used": local_subset_node_count is None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "q2_state": "NOT_RUN",
        "role_evaluation_performed": False,
        "source_commit_bound": False,
        "source_snapshot_filesystem_isolated": False,
        "split_accessed": False,
        "target_truth_accessed": False,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--local-subset-node-count",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
        help="engineering-only subset; never emits full-preflight success",
    )
    return parser.parse_args()


def main() -> int:
    try:
        arguments = _arguments()
        payload = endpoint_object(arguments.local_subset_node_count)
        output = _canonical_json_bytes(payload)
        config, _ = _load_config()
        envelope = config["provisional_hard_envelope"]
        assert isinstance(envelope, dict)
        maximum = envelope["max_output_bytes_per_endpoint"]
        assert type(maximum) is int
        if len(output) > maximum:
            _fail("PREFLIGHT_CAPACITY_GUARD_HIT", "9/OUTPUT_BYTES")
    except EndpointError as error:
        output = _canonical_json_bytes(
            {
                "active_transition_allowed": False,
                "authority_claimed": False,
                "detail": error.detail,
                "error_code": error.code,
                "q1_gate_count": 0,
                "q1_gate_mask": 0,
                "q1_state": "NOT_RUN",
                "resource_guard_id": 9 if error.detail == "9/OUTPUT_BYTES" else None,
                "resource_guard_name": "OUTPUT_BYTES" if error.detail == "9/OUTPUT_BYTES" else None,
                "schema_version": ERROR_SCHEMA_VERSION,
                "source_snapshot_filesystem_isolated": False,
            }
        )
        sys.stdout.buffer.write(output)
        return 1
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

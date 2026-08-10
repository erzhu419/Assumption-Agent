"""Host-replayed pre-dual readiness evidence for Phase-3A-Q0.

This module qualifies only the eleven Q0 gates that can be established by a
single, source-bound host replay.  Gates 11, 13, and 14 deliberately remain
pending for the dual-isolation supervisor.  In particular, this module cannot
construct ``Q0SaturationReceiptV1`` and cannot publish a 14/14 result.
"""

from __future__ import annotations

import ast as python_ast
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import Final, Mapping, NoReturn

from . import phase3_q0_input_adapter_v1 as _adapter
from . import phase3_q0_quotient_contract_v1 as _contract
from . import phase3_q0_quotient_oracle_v1 as _oracle
from .strict_ast_shrink6_v1 import (
    canonicalize_shrink6_source_ast,
    decode_shrink6_canonical_ast,
)
from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
)


SCHEMA_VERSION: Final = "hegel-phase3a-q0-pre-dual-gate-evidence/1"
SOURCE_BINDING_DOMAIN: Final = b"HEGEL/Q0/PRE_DUAL_SOURCE_BINDING/V1\x00"
PRE_DUAL_PASS_GATE_IDS: Final = tuple(range(1, 11)) + (12,)
PENDING_DUAL_GATE_IDS: Final = (11, 13, 14)
PRE_DUAL_GATE_MASK: Final = sum(1 << (gate_id - 1) for gate_id in PRE_DUAL_PASS_GATE_IDS)
PRE_DUAL_GATE_COUNT: Final = len(PRE_DUAL_PASS_GATE_IDS)

_SOURCE_ENTRY_MODULE: Final = "hegel_machine.phase3_q0_gate_qualification_v1"
_SOURCE_CLOSURE_ALGORITHM: Final = "RECURSIVE_LOCAL_IMPORT_AST_CLOSURE_V1"
_SOURCE_CONFIG_PATH: Final = "config/phase3_q0_quotient_freeze_v1.json"
_FORBIDDEN_SOURCE_PATH_TOKENS: Final = (
    "target",
    "truth",
    "split",
    "phase3_dsl_v1",
)
_FROZEN_LEAF_OUTPUT_SORT_IDS: Final = (
    5,
    5,
    5,
    2,
    2,
    4,
    5,
    4,
    5,
    5,
    5,
    5,
    4,
    1,
    1,
)


class Q0GateQualificationError(RuntimeError):
    """Stable fail-closed error for host pre-dual gate qualification."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q0GateQualificationError(code, detail)


def canonical_gate_json_bytes_v1(value: object) -> bytes:
    """Return the one diagnostic JSON encoding admitted for gate evidence."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        _fail("FAIL_Q0_GATE_JSON", f"evidence is not canonical-JSON-safe: {error}")


def _root_path(project_root: Path | None) -> Path:
    root = (
        Path(__file__).resolve().parents[2]
        if project_root is None
        else Path(project_root).resolve()
    )
    if not root.is_dir():
        _fail("FAIL_Q0_GATE_SOURCE", f"project root is absent: {root}")
    return root


def _normative_path(root: Path) -> Path:
    relative = Path(_contract.NORMATIVE_DOCUMENT_PATH)
    if relative.parts and relative.parts[0] == root.name:
        relative = Path(*relative.parts[1:])
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        _fail("FAIL_Q0_GATE_SOURCE", "normative path escapes project root")
    return path


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("FAIL_Q0_GATE_SOURCE", f"cannot decode {path}: {error}")
    if type(value) is not dict:
        _fail("FAIL_Q0_GATE_SOURCE", f"{path} is not a JSON object")
    return value


def _python_module_path(root: Path, module: str) -> Path | None:
    prefix = "hegel_machine."
    if not module.startswith(prefix):
        return None
    suffix = module[len(prefix) :]
    if not suffix or "." in suffix:
        return None
    candidate = root / "src/hegel_machine" / f"{suffix}.py"
    return candidate if candidate.is_file() else None


def _local_imports(path: Path, module: str) -> set[str]:
    """Resolve local imports without importing the historical package root."""

    try:
        tree = python_ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError) as error:
        _fail("FAIL_Q0_GATE_SOURCE", f"cannot parse Python dependency {path}: {error}")
    found: set[str] = set()
    for node in python_ast.walk(tree):
        if isinstance(node, python_ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("hegel_machine.")
            )
            continue
        if not isinstance(node, python_ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module == "hegel_machine":
                found.update(
                    f"hegel_machine.{alias.name}" for alias in node.names
                )
            elif node.module and node.module.startswith("hegel_machine."):
                found.add(node.module)
            continue
        package_parts = module.split(".")[:-1]
        remove = node.level - 1
        if remove > len(package_parts):
            _fail(
                "FAIL_Q0_GATE_SOURCE",
                f"relative import escapes package in {path}",
            )
        anchor = package_parts[: len(package_parts) - remove]
        if node.module:
            found.add(".".join((*anchor, *node.module.split("."))))
        else:
            found.update(
                ".".join((*anchor, alias.name)) for alias in node.names
            )
    return found


def _source_paths_v1(root: Path) -> tuple[str, ...]:
    """Discover the complete recursive local import closure for this replay."""

    pending = {_SOURCE_ENTRY_MODULE}
    modules: set[str] = set()
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in modules:
            continue
        path = _python_module_path(root, module)
        if path is None:
            _fail(
                "FAIL_Q0_GATE_SOURCE",
                f"local source module is absent: {module}",
            )
        relative = path.relative_to(root).as_posix()
        if any(token in relative.lower() for token in _FORBIDDEN_SOURCE_PATH_TOKENS):
            _fail(
                "FAIL_Q0_GATE_SOURCE",
                f"source closure reaches forbidden path: {relative}",
            )
        modules.add(module)
        pending.update(
            dependency
            for dependency in _local_imports(path, module)
            if _python_module_path(root, dependency) is not None
            and dependency not in modules
        )
    paths = {
        _SOURCE_CONFIG_PATH,
        *(
            _python_module_path(root, module).relative_to(root).as_posix()
            for module in modules
            if _python_module_path(root, module) is not None
        ),
    }
    if "src/hegel_machine/__init__.py" in paths:
        _fail("FAIL_Q0_GATE_SOURCE", "historical package initializer entered closure")
    return tuple(sorted(paths))


def _source_manifest_v1(root: Path) -> tuple[list[dict[str, object]], str]:
    source_paths = _source_paths_v1(root)
    rows: list[dict[str, object]] = []
    for relative in source_paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            _fail("FAIL_Q0_GATE_SOURCE", f"source path escapes root: {relative}")
        if not path.is_file() or path.is_symlink():
            _fail("FAIL_Q0_GATE_SOURCE", f"source is not a regular file: {relative}")
        payload = path.read_bytes()
        rows.append(
            {
                "path": relative,
                "byte_length": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        )
    root_digest = sha256(
        SOURCE_BINDING_DOMAIN + canonical_gate_json_bytes_v1(rows)
    ).hexdigest()
    return rows, root_digest


def _gate(
    gate_id: int,
    predicates: Mapping[str, bool],
    evidence: Mapping[str, object],
    source_manifest_root: str,
    *,
    pending_dual: bool = False,
) -> dict[str, object]:
    if type(gate_id) is not int or not 1 <= gate_id <= 14:
        _fail("FAIL_Q0_GATE_REGISTRY", "gate id is outside 1..14")
    material = dict(predicates)
    if not material or any(type(name) is not str for name in material):
        _fail("FAIL_Q0_GATE_PREDICATE", f"Gate {gate_id} predicates are malformed")
    if any(type(value) is not bool for value in material.values()):
        _fail("FAIL_Q0_GATE_PREDICATE", f"Gate {gate_id} predicate is not bool")
    if pending_dual:
        if gate_id not in PENDING_DUAL_GATE_IDS or any(material.values()):
            _fail("FAIL_Q0_GATE_PENDING", f"Gate {gate_id} is not fail-closed pending")
        passed = False
    else:
        if gate_id not in PRE_DUAL_PASS_GATE_IDS or not all(material.values()):
            failed = sorted(name for name, value in material.items() if not value)
            _fail("FAIL_Q0_PRE_DUAL_GATE", f"Gate {gate_id} failed: {failed}")
        passed = True
    details = dict(evidence)
    details["source_manifest_root"] = source_manifest_root
    return {
        "gate_id": gate_id,
        "name": _contract.Q0_READINESS_GATES[gate_id - 1],
        "passed": passed,
        "predicates": material,
        "evidence": details,
        "pending_dual": pending_dual,
    }


def _rejection_code(callable_object: object) -> str | None:
    try:
        callable_object()  # type: ignore[operator]
    except (ValueError, RuntimeError) as error:
        code = getattr(error, "code", None)
        return code if type(code) is str else type(error).__name__
    return None


def _leaf_ast(source: tuple[object, ...]):
    return canonicalize_shrink6_source_ast(source)


def _cohort_and_dominance_vectors() -> tuple[dict[str, bool], dict[str, object]]:
    context = _leaf_ast(("context_flag", 0))
    task = _leaf_ast(("task_flag", 0))
    context_behavior = _oracle.behavior_blob_for_ast_v1(context)
    task_behavior = _oracle.behavior_blob_for_ast_v1(task)
    context_signature = _contract.future_signature_from_ast_v1(context)
    task_signature = _contract.future_signature_from_ast_v1(task)

    tie = _oracle.QuotientAccumulatorV1()
    first_delta = tie.add_ast(context)
    second_delta = tie.add_ast(task)
    tie_record = tie.records()[0]
    tie_bank = tie.continuation_bank_object()
    and2 = _leaf_ast(
        ("top_level_AND", ("context_flag", 0), ("task_flag", 0))
    )

    aggregate = _leaf_ast(("aggregate", 0, 3, 0, ()))
    absolute_aggregate = _leaf_ast(("absolute", ("aggregate", 0, 3, 0, ())))
    latent_dominated = _leaf_ast(
        (
            "absolute",
            (
                "difference",
                ("scalar_const", 3),
                ("aggregate", 0, 3, 0, ()),
            ),
        )
    )
    reservoir = _oracle.QuotientAccumulatorV1()
    for ast in (aggregate, absolute_aggregate, latent_dominated):
        reservoir.add_ast(ast)

    shorter = context_signature
    longer = replace(
        context_signature,
        mdl_length_q32=context_signature.mdl_length_q32 + (1 << 32),
    )
    structurally_better_but_longer = replace(
        context_signature,
        mdl_length_q32=context_signature.mdl_length_q32 + (1 << 32),
    )
    structurally_worse_but_shorter = replace(
        context_signature,
        ast_node_count=context_signature.ast_node_count + 1,
    )
    no_slots = replace(context_signature, distinct_bit_slot_bitmap=0)
    slot_zero = replace(context_signature, distinct_bit_slot_bitmap=1)

    predicates = {
        "equal_behavior_distinct_ast_witnesses": (
            context_behavior.canonical_bytes == task_behavior.canonical_bytes
            and context.cbor_bytes != task.cbor_bytes
        ),
        "equal_complete_construction_signature": context_signature == task_signature,
        "two_identity_sensitive_witnesses_retained": (
            first_delta == (1, 1, 1)
            and second_delta == (0, 1, 1)
            and len(tie_record.frontier) == 2
            and tuple(entry.normalization_witness_rank for entry in tie_record.frontier)
            == (0, 1)
            and len(tie_bank) == 1
            and len(tie_bank[0][3]) == 2
        ),
        "single_representative_counterexample_constructible": (
            and2.cbor_bytes.hex() == "82018204828300040083000500"
        ),
        "mdl_participates_in_dominance": (
            shorter.dominates(longer) and not longer.dominates(shorter)
        ),
        "structural_mdl_tradeoff_is_pareto_incomparable": (
            not structurally_better_but_longer.dominates(
                structurally_worse_but_shorter
            )
            and not structurally_worse_but_shorter.dominates(
                structurally_better_but_longer
            )
        ),
        "bit_slot_subset_is_exact_not_popcount": (
            no_slots.dominates(slot_zero) and not slot_zero.dominates(no_slots)
        ),
        "dominated_cohort_remains_in_continuation_bank": (
            reservoir.frontier_point_count == 2
            and reservoir.continuation_bank_point_count == 3
        ),
    }
    evidence = {
        "tie_behavior_id": context_behavior.behavior_id.hex(),
        "tie_ast_cbor_hex": [context.cbor_bytes.hex(), task.cbor_bytes.hex()],
        "tie_frontier_ranks": [
            entry.normalization_witness_rank for entry in tie_record.frontier
        ],
        "and2_counterexample_cbor_hex": and2.cbor_bytes.hex(),
        "reservoir_visible_frontier_count": reservoir.frontier_point_count,
        "reservoir_continuation_bank_count": reservoir.continuation_bank_point_count,
        "sort_witness_capacities": {
            sort_id.name: _contract.normalization_witness_capacity_v1(sort_id)
            for sort_id in _contract.OutputSortId
        },
    }
    return predicates, evidence


def _adversarial_vectors(
    result: _oracle.Q0OracleEndpointResultV1,
) -> tuple[dict[str, bool], dict[str, object]]:
    probe = _contract.Q0ProbeInputV1()
    all_bottom = (_contract.BehaviorCellV1.bottom(),) * 4
    bool_bottom = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        _contract.OutputSortId.BOOL,
        all_bottom,
    )
    bit_bottom = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        _contract.OutputSortId.BIT,
        all_bottom,
    )
    positional_a = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        _contract.OutputSortId.BOOL,
        (
            _contract.BehaviorCellV1.exact(True),
            _contract.BehaviorCellV1.bottom(),
            _contract.BehaviorCellV1.exact(False),
            _contract.BehaviorCellV1.exact(True),
        ),
    )
    positional_b = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        _contract.OutputSortId.BOOL,
        (
            _contract.BehaviorCellV1.exact(True),
            _contract.BehaviorCellV1.exact(False),
            _contract.BehaviorCellV1.bottom(),
            _contract.BehaviorCellV1.exact(True),
        ),
    )

    aliased_rows = ((True, probe.rows[0][1], probe.rows[0][2]),) + probe.rows[1:]
    bool_alias_code = _rejection_code(lambda: _contract.Q0ProbeInputV1(aliased_rows))
    cell_alias_code = _rejection_code(
        lambda: _contract.BehaviorCellV1.exact(1).canonical_object(
            _contract.OutputSortId.BOOL
        )
    )

    collision_ast = _leaf_ast(("scalar_const", 1))
    collision_behavior = _oracle.behavior_blob_for_ast_v1(collision_ast)
    collision_accumulator = _oracle.QuotientAccumulatorV1()
    collision_accumulator._digest_preimages[collision_behavior.behavior_id] = (
        b"adversarial-distinct-preimage"
    )
    collision_code = _rejection_code(
        lambda: collision_accumulator.add_ast(collision_ast)
    )

    reversed_code = _rejection_code(
        lambda: _contract.quotient_class_archive_root_v1(
            tuple(reversed(result.syntax_class_records[:2]))
        )
    )
    predicates = {
        "bool_uint_alias_rejected_in_probe": bool_alias_code == "REJECT_Q0_PROBE_INPUT",
        "bool_uint_alias_rejected_in_behavior_cell": (
            cell_alias_code == "REJECT_Q0_BEHAVIOR_CELL"
        ),
        "all_bottom_behavior_is_sort_bound": (
            bool_bottom.canonical_bytes != bit_bottom.canonical_bytes
            and bool_bottom.behavior_id != bit_bottom.behavior_id
        ),
        "bottom_position_is_behavior_identity": (
            positional_a.canonical_bytes != positional_b.canonical_bytes
            and positional_a.behavior_id != positional_b.behavior_id
        ),
        "digest_collision_guard_fails_closed": (
            collision_code == "FAIL_SHA256_PREIMAGE_COLLISION"
        ),
        "noncanonical_class_sort_is_rejected": (
            reversed_code == "REJECT_Q0_QUOTIENT_ARCHIVE"
        ),
    }
    evidence = {
        "bool_alias_failure_code": bool_alias_code,
        "behavior_cell_alias_failure_code": cell_alias_code,
        "collision_failure_code": collision_code,
        "class_sort_failure_code": reversed_code,
        "bool_bottom_behavior_id": bool_bottom.behavior_id.hex(),
        "bit_bottom_behavior_id": bit_bottom.behavior_id.hex(),
        "bottom_position_behavior_ids": [
            positional_a.behavior_id.hex(),
            positional_b.behavior_id.hex(),
        ],
    }
    return predicates, evidence


def qualify_q0_pre_dual_gates_v1(
    project_root: Path | None = None,
) -> dict[str, object]:
    """Replay Gates 1--10 and 12; leave 11, 13, and 14 pending.

    The returned value is JSON-safe diagnostic evidence.  Any internal gate
    mismatch raises before a payload can be returned.  No receipt is created.
    """

    root = _root_path(project_root)
    source_rows, source_root = _source_manifest_v1(root)
    config_path = root / "config/phase3_q0_quotient_freeze_v1.json"
    config = _read_json_object(config_path)
    normative_path = _normative_path(root)
    if not normative_path.is_file() or normative_path.is_symlink():
        _fail("FAIL_Q0_GATE_SOURCE", "normative direction is not a regular file")
    normative_bytes = normative_path.read_bytes()
    normative_sha = sha256(normative_bytes).hexdigest()

    gates: list[dict[str, object]] = []
    direction_config = config.get("normative_direction")
    if type(direction_config) is not dict:
        _fail("FAIL_Q0_GATE_SOURCE", "normative_direction config is malformed")
    gates.append(
        _gate(
            1,
            {
                "document_bytes_sha256_exact": (
                    normative_sha == _contract.NORMATIVE_DOCUMENT_SHA256
                ),
                "document_id_exact": (
                    direction_config.get("document_id")
                    == _contract.NORMATIVE_DOCUMENT_ID
                ),
                "document_path_exact": (
                    direction_config.get("document_path")
                    == _contract.NORMATIVE_DOCUMENT_PATH
                ),
                "config_digest_matches_bytes": (
                    direction_config.get("document_sha256") == normative_sha
                ),
                "c3_route_frozen_and_route_a_rejected": (
                    direction_config.get("selected_route")
                    == "C3_PRIMARY_THEN_TARGET_BLIND_B_IF_NEEDED_ELSE_D"
                    and direction_config.get("syntactic_budget_increase_route_a")
                    == "REJECTED"
                ),
            },
            {
                "document_id": _contract.NORMATIVE_DOCUMENT_ID,
                "document_path": _contract.NORMATIVE_DOCUMENT_PATH,
                "byte_length": len(normative_bytes),
                "sha256": normative_sha,
                "selected_route": direction_config.get("selected_route"),
            },
            source_root,
        )
    )

    semantic = config.get("semantic_bindings")
    if type(semantic) is not dict:
        _fail("FAIL_Q0_GATE_SOURCE", "semantic_bindings config is malformed")
    five_roots = {
        "child_dsl_spec_root": _contract.Q0_CHILD_DSL_SPEC_ROOT,
        "operator_semantics_root": _contract.Q0_OPERATOR_SEMANTICS_ROOT,
        "identifier_registry_root": _contract.Q0_IDENTIFIER_REGISTRY_ROOT,
        "canonical_ast_schema_root": _contract.Q0_CANONICAL_AST_SCHEMA_ROOT,
        "canonical_cbor_profile_root": _contract.Q0_CANONICAL_CBOR_PROFILE_ROOT,
    }
    leaf_asts = tuple(
        canonicalize_shrink6_source_ast(seed.source_ast)
        for seed in _oracle.Q0_FROZEN_LEAF_SEEDS
    )
    leaf_sorts = tuple(
        int(_oracle.behavior_blob_for_ast_v1(ast).output_sort_id)
        for ast in leaf_asts
    )
    semantic_object = _contract.q0_semantic_binding_object_v1()
    gates.append(
        _gate(
            2,
            {
                "four_version_identities_exact": (
                    config.get("dsl_version") == _contract.DSL_VERSION
                    and config.get("dsl_freeze_version")
                    == _contract.DSL_FREEZE_VERSION
                    and config.get("closure_semantics_version")
                    == _contract.CLOSURE_SEMANTICS_VERSION
                    and config.get("freeze_version") == _contract.Q0_FREEZE_VERSION
                ),
                "five_v16_roots_exact": all(
                    semantic.get(name) == value.hex()
                    for name, value in five_roots.items()
                ),
                "projection_manifest_root_replayed": (
                    _contract.q0_projection_manifest_root_v1().hex()
                    == semantic.get("projection_manifest_root_hex")
                ),
                "semantic_binding_root_replayed": (
                    _contract.q0_semantic_binding_root_v1().hex()
                    == semantic.get("semantic_binding_root_hex")
                    and len(semantic_object) == 17
                ),
                "output_sort_registry_exact": (
                    [(int(item), item.name) for item in _contract.OutputSortId]
                    == [
                        (1, "BOOL"),
                        (2, "BIT"),
                        (3, "SIGN"),
                        (4, "BOUNDED_INT"),
                        (5, "RATIONAL_VALUE"),
                    ]
                ),
                "fifteen_leaf_typing_replayed": (
                    len(leaf_asts) == 15
                    and tuple(ast.value[1] for ast in leaf_asts)
                    == _contract.Q0_FROZEN_LEAF_CANONICAL_NODES
                    and leaf_sorts == _FROZEN_LEAF_OUTPUT_SORT_IDS
                ),
                "adapter_registry_exact": (
                    _adapter.DSL_VERSION == _contract.DSL_VERSION
                    and _adapter.ACTIVE_AGGREGATE_MAP_IDS == (0, 1, 5)
                    and _adapter.TOMBSTONED_AGGREGATE_MAP_IDS == (2, 3, 4)
                    and _adapter.ACTIVE_RATIONAL_PARAMETER_IDS == (1, 3, 5)
                ),
            },
            {
                "versions": [
                    _contract.DSL_VERSION,
                    _contract.DSL_FREEZE_VERSION,
                    _contract.CLOSURE_SEMANTICS_VERSION,
                    _contract.Q0_FREEZE_VERSION,
                ],
                "five_v16_roots": {
                    name: value.hex() for name, value in five_roots.items()
                },
                "projection_manifest_root": (
                    _contract.q0_projection_manifest_root_v1().hex()
                ),
                "semantic_binding_root": (
                    _contract.q0_semantic_binding_root_v1().hex()
                ),
                "leaf_output_sort_ids": list(leaf_sorts),
            },
            source_root,
        )
    )

    probe = _contract.Q0ProbeInputV1()
    decoded_probe = canonical_cbor_decode(probe.canonical_bytes)
    environments = probe.observation_environments()
    decoded_environments = tuple(
        _adapter.decode_observation_environment_v1(
            canonical_cbor_encode(source_object)
        )
        for _, _, source_object in probe.rows
    )
    adapter_predicates = {
        "probe_roundtrip_is_byte_exact": (
            decoded_probe == probe.canonical_object()
            and canonical_cbor_encode(decoded_probe) == probe.canonical_bytes
        ),
        "probe_golden_length_and_root_exact": (
            len(probe.canonical_bytes) == 172
            and probe.universe_root.hex()
            == config["projection"]["frozen_universe_root_hex"]  # type: ignore[index]
        ),
        "four_ordered_typed_rows_exact": (
            len(environments) == 4
            and tuple(item.input_signature_id for item in environments)
            == (1, 1, 2, 2)
            and tuple(item.set_size for item in environments) == (5, 8, 4, 4)
        ),
        "adapter_decode_replays_same_observations": (
            environments == decoded_environments
        ),
        "missing_observations_remain_typed_bottom": (
            all(
                value is _adapter.BOTTOM
                for environment in environments[:2]
                for value in environment.context_flags + environment.task_flags
            )
            and all(
                environment.bit_at(0) is _adapter.BOTTOM
                for environment in environments[2:]
            )
        ),
    }
    gates.append(
        _gate(
            3,
            adapter_predicates,
            {
                "probe_cbor_hex": probe.canonical_bytes.hex(),
                "probe_universe_root": probe.universe_root.hex(),
                "input_signature_ids": [
                    item.input_signature_id for item in environments
                ],
                "set_sizes": [item.set_size for item in environments],
            },
            source_root,
        )
    )

    codec_vectors = {
        "bottom_bool": canonical_cbor_encode(
            _contract.BehaviorCellV1.bottom().canonical_object(
                _contract.OutputSortId.BOOL
            )
        ).hex(),
        "bool_true": canonical_cbor_encode(
            _contract.BehaviorCellV1.exact(True).canonical_object(
                _contract.OutputSortId.BOOL
            )
        ).hex(),
        "bit_one": canonical_cbor_encode(
            _contract.BehaviorCellV1.exact(1).canonical_object(
                _contract.OutputSortId.BIT
            )
        ).hex(),
        "sign_negative": canonical_cbor_encode(
            _contract.BehaviorCellV1.exact(-1).canonical_object(
                _contract.OutputSortId.SIGN
            )
        ).hex(),
        "bounded_int_negative_eight": canonical_cbor_encode(
            _contract.BehaviorCellV1.exact(-8).canonical_object(
                _contract.OutputSortId.BOUNDED_INT
            )
        ).hex(),
        "rational_negative_two_thirds": canonical_cbor_encode(
            _contract.BehaviorCellV1.exact(Fraction(-2, 3)).canonical_object(
                _contract.OutputSortId.RATIONAL_VALUE
            )
        ).hex(),
    }
    bit_behavior = _oracle.behavior_blob_for_ast_v1(_leaf_ast(("bit_at", 0)))
    gates.append(
        _gate(
            4,
            {
                "explicit_bottom_codec_exact": codec_vectors["bottom_bool"] == "8100",
                "bool_bit_rational_codecs_exact": (
                    codec_vectors["bool_true"] == "8201f5"
                    and codec_vectors["bit_one"] == "820101"
                    and codec_vectors["rational_negative_two_thirds"]
                    == "8201822103"
                ),
                "sign_and_bounded_int_codecs_exact": (
                    codec_vectors["sign_negative"] == "820120"
                    and codec_vectors["bounded_int_negative_eight"] == "820127"
                ),
                "bottom_propagates_positionally": (
                    tuple(cell.defined for cell in bit_behavior.cells)
                    == (True, True, False, False)
                ),
                "behavior_self_id_replayed": (
                    bit_behavior.behavior_id
                    == content_hash(
                        _contract.BEHAVIOR_ID_DOMAIN,
                        bit_behavior.canonical_object(),
                    )
                ),
            },
            {
                "codec_cbor_hex": codec_vectors,
                "bit_at_0_behavior_id": bit_behavior.behavior_id.hex(),
                "bit_at_0_defined_positions": [
                    cell.defined for cell in bit_behavior.cells
                ],
            },
            source_root,
        )
    )

    identity_cells = (
        _contract.BehaviorCellV1.exact(False),
        _contract.BehaviorCellV1.bottom(),
        _contract.BehaviorCellV1.exact(True),
        _contract.BehaviorCellV1.exact(False),
    )
    base_blob = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        _contract.OutputSortId.BOOL,
        identity_cells,
    )
    other_universe = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        bytes(reversed(probe.universe_root)),
        _contract.OutputSortId.BOOL,
        identity_cells,
    )
    other_signature = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID + 1,
        probe.universe_root,
        _contract.OutputSortId.BOOL,
        identity_cells,
    )
    other_sort = _contract.BehaviorBlobV1(
        _contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        _contract.OutputSortId.BIT,
        tuple(
            _contract.BehaviorCellV1.bottom()
            if not cell.defined
            else _contract.BehaviorCellV1.exact(int(bool(cell.value)))
            for cell in identity_cells
        ),
    )
    identity_blobs = (base_blob, other_universe, other_signature, other_sort)
    gates.append(
        _gate(
            5,
            {
                "behavior_wire_binds_input_signature": (
                    base_blob.behavior_id != other_signature.behavior_id
                ),
                "behavior_wire_binds_frozen_universe": (
                    base_blob.behavior_id != other_universe.behavior_id
                ),
                "behavior_wire_binds_output_sort": (
                    base_blob.behavior_id != other_sort.behavior_id
                ),
                "four_identity_components_are_digest_distinct": (
                    len({blob.behavior_id for blob in identity_blobs}) == 4
                ),
                "probe_root_is_only_q0_universe_binding": (
                    base_blob.frozen_universe_root == probe.universe_root
                    and base_blob.input_signature_id
                    == _contract.Q0_PROBE_INPUT_SIGNATURE_ID
                ),
            },
            {
                "identity_behavior_ids": [
                    blob.behavior_id.hex() for blob in identity_blobs
                ],
                "bound_input_signature_id": base_blob.input_signature_id,
                "bound_universe_root": base_blob.frozen_universe_root.hex(),
                "bound_output_sort_id": int(base_blob.output_sort_id),
            },
            source_root,
        )
    )

    # One host run supplies the live source-bound mechanics for Gates 6--10
    # and the adversarial vectors in Gate 12.  Its single-endpoint PASS is not
    # promoted to dual Gate 11 by this module.
    result = _oracle.run_q0_python_oracle_v1()
    syntax_record_bytes = tuple(
        record.canonical_bytes for record in result.syntax_class_records
    )
    direct_record_bytes = tuple(
        record.canonical_bytes for record in result.direct_class_records
    )
    behavior_pairs = tuple(
        (record.behavior.behavior_id, record.behavior.canonical_bytes)
        for record in result.syntax_class_records
    )
    adversarial_predicates, adversarial_evidence = _adversarial_vectors(result)
    gates.append(
        _gate(
            6,
            {
                "complete_behavior_preimages_match_across_paths": (
                    syntax_record_bytes == direct_record_bytes
                ),
                "behavior_ids_replay_from_complete_preimages": all(
                    behavior_id
                    == content_hash(
                        _contract.BEHAVIOR_ID_DOMAIN,
                        record.behavior.canonical_object(),
                    )
                    for behavior_id, record in zip(
                        (item[0] for item in behavior_pairs),
                        result.syntax_class_records,
                        strict=True,
                    )
                ),
                "class_archive_root_replayed": (
                    _contract.quotient_class_archive_root_v1(
                        result.syntax_class_records
                    )
                    == result.syntax_class_archive_root
                    == result.direct_class_archive_root
                ),
                "behavior_digest_preimages_are_unique": (
                    len({item[0] for item in behavior_pairs})
                    == len({item[1] for item in behavior_pairs})
                    == len(behavior_pairs)
                ),
                "collision_preimage_guard_executed": adversarial_predicates[
                    "digest_collision_guard_fails_closed"
                ],
                "cross_sort_identity_executed": adversarial_predicates[
                    "all_bottom_behavior_is_sort_bound"
                ],
            },
            {
                "behavior_class_count": len(behavior_pairs),
                "class_archive_root": result.syntax_class_archive_root.hex(),
                "output_sort_ids": sorted(
                    {int(record.behavior.output_sort_id) for record in result.syntax_class_records}
                ),
                "collision_failure_code": adversarial_evidence[
                    "collision_failure_code"
                ],
            },
            source_root,
        )
    )

    signatures = tuple(
        (entry, decode_shrink6_canonical_ast(entry.representative_ast_cbor))
        for record in result.syntax_class_records
        for entry in record.frontier
    )
    cohort_predicates, cohort_evidence = _cohort_and_dominance_vectors()
    gates.append(
        _gate(
            7,
            {
                "all_frontier_signatures_replay_from_ast": all(
                    entry.signature == _contract.future_signature_from_ast_v1(ast)
                    for entry, ast in signatures
                ),
                "all_eleven_signature_fields_bound": all(
                    len(entry.signature.canonical_object()) == 14
                    for entry, _ in signatures
                ),
                "all_old_law_composition_depths_zero": all(
                    entry.signature.old_law_composition_depth == 0
                    for entry, _ in signatures
                ),
                "all_cohort_ranks_within_sort_capacity": all(
                    entry.normalization_witness_rank
                    < _contract.normalization_witness_capacity_v1(
                        entry.signature.output_sort_id
                    )
                    for entry, _ in signatures
                ),
                "identity_sensitive_two_witness_counterexample_replayed": (
                    cohort_predicates["two_identity_sensitive_witnesses_retained"]
                    and cohort_predicates[
                        "single_representative_counterexample_constructible"
                    ]
                ),
            },
            {
                "frontier_signature_count": len(signatures),
                "signature_schema_field_count": 14,
                **cohort_evidence,
            },
            source_root,
        )
    )

    gates.append(
        _gate(
            8,
            {
                "every_visible_frontier_is_pareto_idempotent": all(
                    record.frontier == _contract.pareto_frontier_v1(record.frontier)
                    for record in result.syntax_class_records
                ),
                "every_class_mdl_is_minimum_retained_representative": all(
                    record.minimum_mdl_length_q32
                    == min(entry.signature.mdl_length_q32 for entry in record.frontier)
                    for record in result.syntax_class_records
                ),
                "mdl_and_bitmask_dominance_vectors_pass": (
                    cohort_predicates["mdl_participates_in_dominance"]
                    and cohort_predicates["bit_slot_subset_is_exact_not_popcount"]
                ),
                "structural_mdl_tradeoff_preserved": cohort_predicates[
                    "structural_mdl_tradeoff_is_pareto_incomparable"
                ],
                "latent_dominated_cohort_bank_reservoir_preserved": (
                    cohort_predicates[
                        "dominated_cohort_remains_in_continuation_bank"
                    ]
                ),
                "normalization_multiplicity_counterexample_preserved": (
                    cohort_predicates[
                        "equal_behavior_distinct_ast_witnesses"
                    ]
                    and cohort_predicates["equal_complete_construction_signature"]
                ),
            },
            cohort_evidence,
            source_root,
        )
    )

    syntax_coverage = result.syntax_coverage_records
    direct_coverage = result.quotient_coverage_records
    zero_syntax = [row[0] for row in syntax_coverage if row[1] == 0]
    zero_direct = [row[0] for row in direct_coverage if row[1] == 0]
    gates.append(
        _gate(
            9,
            {
                "all_27_operator_codes_present_in_both_paths": (
                    tuple(row[0] for row in syntax_coverage)
                    == tuple(row[0] for row in direct_coverage)
                    == _contract.Q0_COVERAGE_CODES
                    and len(syntax_coverage) == len(direct_coverage) == 27
                ),
                "coverage_rows_have_exact_six_field_wire": all(
                    len(row) == _contract.Q0_COVERAGE_RECORD_LENGTH
                    for row in syntax_coverage + direct_coverage
                ),
                "every_eligible_application_is_strict_admitted": all(
                    row[1] == row[2] for row in syntax_coverage + direct_coverage
                ),
                "structurally_unreachable_sign_pairs_explicitly_zero": (
                    zero_syntax == zero_direct == [0x2005, 0x2006]
                ),
                "independent_operator_coverage_roots_replayed": (
                    _oracle.operator_coverage_root_v1(syntax_coverage)
                    == result.syntax_operator_coverage_root
                    and _oracle.operator_coverage_root_v1(direct_coverage)
                    == result.quotient_operator_coverage_root
                ),
                "per_operator_congruence_reaches_identical_class_archive": (
                    syntax_record_bytes == direct_record_bytes
                    and result.syntax_class_archive_root
                    == result.direct_class_archive_root
                ),
            },
            {
                "syntax_coverage_rows": [list(row) for row in syntax_coverage],
                "direct_coverage_rows": [list(row) for row in direct_coverage],
                "syntax_coverage_root": result.syntax_operator_coverage_root.hex(),
                "direct_coverage_root": result.quotient_operator_coverage_root.hex(),
                "explicit_zero_coverage_codes": zero_syntax,
            },
            source_root,
        )
    )

    golden = config.get("golden_expectations")
    if type(golden) is not dict:
        _fail("FAIL_Q0_GATE_SOURCE", "golden_expectations config is malformed")
    actual_counts = {
        "syntax_raw_and_strict_admitted_count": result.syntax_raw_application_count,
        "direct_raw_and_strict_admitted_count": result.quotient_raw_application_count,
        "syntax_rewrite_count": result.rewrite_collapse_syntax_count,
        "direct_rewrite_count": result.rewrite_collapse_quotient_count,
        "canonical_syntax_program_count": result.canonical_syntax_program_count,
        "behavior_class_count": result.behavior_class_count,
        "visible_frontier_point_count": result.frontier_point_count,
        "maximum_visible_frontier_points_per_class": (
            result.maximum_frontier_points_per_class
        ),
        "continuation_bank_point_count_each_path": (
            result.syntax_continuation_bank_point_count
        ),
        "maximum_bank_points_per_class_each_path": (
            result.maximum_syntax_bank_points_per_class
        ),
        "saturation_round_count": result.saturation_round_count,
    }
    actual_roots = {
        "syntax_program_archive_root": result.syntax_program_archive_root.hex(),
        "visible_class_archive_root": result.syntax_class_archive_root.hex(),
        "syntax_coverage_root": result.syntax_operator_coverage_root.hex(),
        "direct_coverage_root": result.quotient_operator_coverage_root.hex(),
        "syntax_state_root": result.syntax_state_root.hex(),
        "direct_state_root": result.direct_state_root.hex(),
        "single_endpoint_output_root": result.endpoint_state_root.hex(),
    }
    final_round = result.round_deltas[-1]
    gates.append(
        _gate(
            10,
            {
                "fifteen_leaf_induction_base_complete": all(
                    row[1] == row[2] == row[5] == 1
                    for row in syntax_coverage[:15] + direct_coverage[:15]
                ),
                "visible_frontier_recursive_projection_admission_replayed": all(
                    ast.metrics.depth <= _contract.Q0_PROJECTION_MAX_AST_DEPTH
                    and ast.metrics.node_count <= _contract.Q0_PROJECTION_MAX_NODE_COUNT
                    and ast.metrics.aggregate_leaf_count
                    <= _contract.Q0_PROJECTION_MAX_AGGREGATE_LEAVES
                    and ast.metrics.top_level_clause_count
                    <= _contract.Q0_PROJECTION_MAX_TOP_LEVEL_CLAUSES
                    for _, ast in signatures
                ),
                "all_seen_programs_bound_by_guarded_archive_and_state": (
                    result.canonical_syntax_program_count == 537
                    and result.syntax_program_archive_root.hex()
                    == golden.get("syntax_program_archive_root")
                    and result.syntax_state_root.hex()
                    == golden.get("syntax_state_root")
                    and result.all_guards_respected is True
                ),
                "complete_continuation_banks_equal_and_guarded": (
                    result.syntax_continuation_bank_point_count
                    == result.quotient_continuation_bank_point_count
                    == golden.get("continuation_bank_point_count_each_path")
                    and result.maximum_syntax_bank_points_per_class
                    == result.maximum_quotient_bank_points_per_class
                    == golden.get("maximum_bank_points_per_class_each_path")
                    and result.syntax_state_root.hex()
                    == golden.get("syntax_state_root")
                    and result.direct_state_root.hex()
                    == golden.get("direct_state_root")
                ),
                "queue_empty_zero_delta_fixed_point": (
                    result.work_queue_empty is True
                    and result.zero_delta_full_round is True
                    and final_round.queued_application_count == 0
                    and final_round.new_canonical_program_count == 0
                    and final_round.new_behavior_class_count == 0
                    and final_round.frontier_mutation_count == 0
                    and final_round.bank_mutation_count == 0
                    and final_round.complete_state_changed is False
                ),
                "all_counts_match_source_bound_freeze": all(
                    golden.get(name) == value for name, value in actual_counts.items()
                ),
                "all_state_and_archive_roots_match_source_bound_freeze": all(
                    golden.get(name) == value for name, value in actual_roots.items()
                ),
                "single_endpoint_non_authoritative_and_guards_respected": (
                    result.endpoint_status == _contract.Q0_ENDPOINT_PASS_STATUS
                    and result.all_guards_respected is True
                    and result.authoritative_claim_allowed is False
                    and "DUAL" not in result.endpoint_status
                ),
            },
            {
                "counts": actual_counts,
                "roots": actual_roots,
                "round_deltas": [
                    {
                        "round_index": row.round_index,
                        "queued_application_count": row.queued_application_count,
                        "new_canonical_program_count": row.new_canonical_program_count,
                        "new_behavior_class_count": row.new_behavior_class_count,
                        "frontier_mutation_count": row.frontier_mutation_count,
                        "bank_mutation_count": row.bank_mutation_count,
                        "complete_state_changed": row.complete_state_changed,
                    }
                    for row in result.round_deltas
                ],
                "endpoint_status": result.endpoint_status,
                "admission_evidence_scope": {
                    "visible_frontier_ast_replay_count": len(signatures),
                    "all_seen_program_count_bound_by_program_archive": (
                        result.canonical_syntax_program_count
                    ),
                    "complete_continuation_bank_count_each_path_bound_by_state_roots": (
                        result.syntax_continuation_bank_point_count
                    ),
                    "syntax_program_archive_root": (
                        result.syntax_program_archive_root.hex()
                    ),
                    "syntax_state_root": result.syntax_state_root.hex(),
                    "direct_state_root": result.direct_state_root.hex(),
                },
            },
            source_root,
        )
    )

    gates.append(
        _gate(
            11,
            {
                "python_rust_endpoint_equality_verified": False,
                "host_class_archive_replay_verified": False,
                "dual_endpoint_source_binding_verified": False,
            },
            {
                "required_producer": "phase3_q0_dual_qualification_v1",
                "single_host_endpoint_root": result.endpoint_state_root.hex(),
                "reason": "dual isolated endpoints and host replay not supplied here",
            },
            source_root,
            pending_dual=True,
        )
    )
    gates.append(
        _gate(
            12,
            adversarial_predicates,
            {
                **adversarial_evidence,
                "producer_scope": "HOST_PYTHON_CONTRACT_REPLAY",
                "dual_adversarial_execution_claimed": False,
            },
            source_root,
        )
    )
    gates.append(
        _gate(
            13,
            {
                "commit_bound_python_manifest_verified": False,
                "commit_bound_rust_and_cargo_manifest_verified": False,
                "isolated_no_network_read_only_execution_verified": False,
            },
            {
                "required_producer": "phase3_q0_dual_qualification_v1",
                "reason": "container isolation and committed manifests are external inputs",
            },
            source_root,
            pending_dual=True,
        )
    )
    gates.append(
        _gate(
            14,
            {
                "dual_host_agreement_verified": False,
                "host_only_receipt_replay_verified": False,
                "q1_q2_outputs_null_and_not_run_verified_by_issuer": False,
            },
            {
                "required_producer": "phase3_q0_dual_qualification_v1",
                "reason": "host issuer has not supplied agreement or receipt",
            },
            source_root,
            pending_dual=True,
        )
    )

    gates.sort(key=lambda row: int(row["gate_id"]))
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "qualification_id": _contract.Q0_QUALIFICATION_ID,
        "q0_state": "PRE_DUAL_11_OF_14",
        "readiness_gate_total": _contract.Q0_READINESS_GATE_TOTAL,
        "readiness_gates_passed": PRE_DUAL_GATE_COUNT,
        "readiness_gate_mask": PRE_DUAL_GATE_MASK,
        "receipt_created": False,
        "authoritative_claim_allowed": False,
        "source_binding": {
            "domain_hex": SOURCE_BINDING_DOMAIN.hex(),
            "entry_module": _SOURCE_ENTRY_MODULE,
            "closure_algorithm": _SOURCE_CLOSURE_ALGORITHM,
            "file_count": len(source_rows),
            "manifest_root": source_root,
            "files": source_rows,
        },
        "target_truth_accessed": result.target_truth_accessed,
        "split_accessed": result.split_accessed,
        "q1_status_id": 0,
        "q1_output_root": None,
        "q2_status_id": 0,
        "role_evaluation_performed": result.role_evaluation_performed,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "gates": gates,
    }
    validate_pre_dual_gate_evidence_v1(payload)
    canonical_gate_json_bytes_v1(payload)
    return payload


def validate_pre_dual_gate_evidence_v1(payload: object) -> None:
    """Validate the exact fail-closed 11/14 pre-dual evidence shape."""

    if type(payload) is not dict:
        _fail("FAIL_Q0_GATE_EVIDENCE", "payload must be a dictionary")
    exact_keys = {
        "schema_version",
        "qualification_id",
        "q0_state",
        "readiness_gate_total",
        "readiness_gates_passed",
        "readiness_gate_mask",
        "receipt_created",
        "authoritative_claim_allowed",
        "source_binding",
        "target_truth_accessed",
        "split_accessed",
        "q1_status_id",
        "q1_output_root",
        "q2_status_id",
        "role_evaluation_performed",
        "m3_formal_roots",
        "outside_certificate_issued",
        "gates",
    }
    if set(payload) != exact_keys:
        _fail("FAIL_Q0_GATE_EVIDENCE", "top-level evidence fields differ")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("qualification_id") != _contract.Q0_QUALIFICATION_ID
        or payload.get("q0_state") != "PRE_DUAL_11_OF_14"
        or payload.get("readiness_gate_total") != 14
        or payload.get("readiness_gates_passed") != PRE_DUAL_GATE_COUNT
        or payload.get("readiness_gate_mask") != PRE_DUAL_GATE_MASK
    ):
        _fail("FAIL_Q0_GATE_EVIDENCE", "pre-dual summary differs")
    if any(
        payload.get(name) is not expected
        for name, expected in (
            ("receipt_created", False),
            ("authoritative_claim_allowed", False),
            ("target_truth_accessed", False),
            ("split_accessed", False),
            ("q1_output_root", None),
            ("role_evaluation_performed", False),
            ("m3_formal_roots", None),
            ("outside_certificate_issued", False),
        )
    ) or payload.get("q1_status_id") != 0 or payload.get("q2_status_id") != 0:
        _fail("FAIL_Q0_GATE_EVIDENCE", "non-authority/downstream guard differs")

    source = payload.get("source_binding")
    if type(source) is not dict or set(source) != {
        "domain_hex",
        "entry_module",
        "closure_algorithm",
        "file_count",
        "manifest_root",
        "files",
    }:
        _fail("FAIL_Q0_GATE_EVIDENCE", "source binding is malformed")
    if (
        source.get("domain_hex") != SOURCE_BINDING_DOMAIN.hex()
        or source.get("entry_module") != _SOURCE_ENTRY_MODULE
        or source.get("closure_algorithm") != _SOURCE_CLOSURE_ALGORITHM
    ):
        _fail("FAIL_Q0_GATE_EVIDENCE", "source binding identity differs")
    files = source.get("files")
    current_rows, current_root = _source_manifest_v1(_root_path(None))
    expected_paths = _source_paths_v1(_root_path(None))
    if (
        type(files) is not list
        or type(source.get("file_count")) is not int
        or source.get("file_count") != len(files)
        or files != current_rows
    ):
        _fail("FAIL_Q0_GATE_EVIDENCE", "source file manifest differs")
    if [row.get("path") for row in files if type(row) is dict] != list(expected_paths):
        _fail("FAIL_Q0_GATE_EVIDENCE", "source manifest path order differs")
    if any(
        type(row) is not dict
        or set(row) != {"path", "byte_length", "sha256"}
        or type(row.get("path")) is not str
        or type(row.get("byte_length")) is not int
        or row.get("byte_length", -1) < 0
        or type(row.get("sha256")) is not str
        or len(row.get("sha256", "")) != 64
        or any(character not in "0123456789abcdef" for character in row["sha256"])
        for row in files
    ):
        _fail("FAIL_Q0_GATE_EVIDENCE", "source manifest row is malformed")
    expected_source_root = sha256(
        SOURCE_BINDING_DOMAIN + canonical_gate_json_bytes_v1(files)
    ).hexdigest()
    if (
        source.get("manifest_root") != expected_source_root
        or expected_source_root != current_root
    ):
        _fail("FAIL_Q0_GATE_EVIDENCE", "source manifest root differs")

    gates = payload.get("gates")
    if type(gates) is not list or len(gates) != 14:
        _fail("FAIL_Q0_GATE_EVIDENCE", "gate registry must contain 14 rows")
    for index, row in enumerate(gates, start=1):
        if type(row) is not dict or set(row) != {
            "gate_id",
            "name",
            "passed",
            "predicates",
            "evidence",
            "pending_dual",
        }:
            _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} row shape differs")
        if (
            type(row.get("gate_id")) is not int
            or row.get("gate_id") != index
            or row.get("name") != _contract.Q0_READINESS_GATES[index - 1]
            or type(row.get("passed")) is not bool
            or type(row.get("pending_dual")) is not bool
            or type(row.get("predicates")) is not dict
            or not row.get("predicates")
            or type(row.get("evidence")) is not dict
        ):
            _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} identity differs")
        predicates = row["predicates"]
        if any(type(key) is not str for key in predicates) or any(
            type(value) is not bool for value in predicates.values()
        ):
            _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} predicates differ")
        evidence = row["evidence"]
        if evidence.get("source_manifest_root") != expected_source_root:
            _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} source binding differs")
        if index in PRE_DUAL_PASS_GATE_IDS:
            if row["passed"] is not True or row["pending_dual"] is not False:
                _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} must pass pre-dual")
            if not all(predicates.values()):
                _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} predicate is false")
        else:
            if row["passed"] is not False or row["pending_dual"] is not True:
                _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} must remain pending")
            if any(predicates.values()):
                _fail("FAIL_Q0_GATE_EVIDENCE", f"Gate {index} was pre-qualified")
    canonical_gate_json_bytes_v1(payload)


__all__ = [
    "PRE_DUAL_GATE_COUNT",
    "PRE_DUAL_GATE_MASK",
    "PRE_DUAL_PASS_GATE_IDS",
    "PENDING_DUAL_GATE_IDS",
    "Q0GateQualificationError",
    "SCHEMA_VERSION",
    "canonical_gate_json_bytes_v1",
    "qualify_q0_pre_dual_gates_v1",
    "validate_pre_dual_gate_evidence_v1",
]

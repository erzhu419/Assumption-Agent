"""Direct target-free replay entrypoint for the shrink-2 capacity subset.

Normal package import executes the historical public ``hegel_machine`` export
surface, which includes target and split modules.  Qualification therefore
invokes this file directly.  A minimal package shell admits only the strict
canonicalizer's target-free dependency closure and the process fails closed if
any other project module is loaded.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


if __package__ not in {None, ""}:
    raise RuntimeError("shrink-2 replay requires its direct file entrypoint")
if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
    raise RuntimeError("shrink-2 replay requires python -I -S -B")
if len(sys.argv) != 2 or sys.argv[1] not in {
    "--capacity-replay",
    "--golden-replay",
}:
    raise RuntimeError(
        "shrink-2 replay requires exactly --capacity-replay or --golden-replay"
    )
_REPLAY_MODE = sys.argv[1]

package = ModuleType("hegel_machine")
package.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package
__package__ = "hegel_machine"

from .phase3_m3_shrink2_core_v1 import (  # noqa: E402
    ACTIVE_RATIONAL_PARAMETER_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    RATIONAL_ACTIVE_AGGREGATE_IDS,
    RATIONAL_PARAMETER_REGISTRY_NAMESPACE,
    REMOVED_AGGREGATE_ERROR,
    REMOVED_RATIONAL_PARAMETER_ERROR,
    RESERVED_RATIONAL_PARAMETER_IDS,
    SHRINK_STEP_ID,
    TOMBSTONED_AGGREGATE_IDS,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
    UNKNOWN_RATIONAL_PARAMETER_ERROR,
)
from .phase3_shrink2_capacity_v1 import (  # noqa: E402
    EXPECTED_SHRINK2_SOURCE_COUNT,
    SHRINK2_CAPACITY_GENERATOR_RULE,
    iter_shrink2_capacity_candidate_asts,
    shrink2_constant_atoms_v1,
    shrink2_mixed_atoms_v1,
    shrink2_rational_aggregate_leaves_v1,
)
from .strict_ast_shrink2_v1 import (  # noqa: E402
    canonicalize_shrink2_source_ast,
    decode_shrink2_canonical_ast,
)
from .strict_ast_shrink1_v1 import canonicalize_shrink1_source_ast  # noqa: E402
from .strict_ast_v1 import StrictAstError  # noqa: E402
from .strict_cbor_v1 import canonical_cbor_encode  # noqa: E402


_ALLOWED_PROJECT_MODULES = {
    "hegel_machine.hashing",
    "hegel_machine.phase3_m3_dsl_core_v1",
    "hegel_machine.phase3_m3_shrink1_core_v1",
    "hegel_machine.phase3_m3_shrink2_core_v1",
    "hegel_machine.phase3_shrink2_capacity_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
}
_CAPACITY_SET_DOMAIN = b"HEGEL/STRICT_CAPACITY_SET/V1"


def _accepted_set_commitment(blobs: tuple[bytes, ...]) -> str:
    digest = sha256()
    digest.update(_CAPACITY_SET_DOMAIN)
    digest.update(b"\x00")
    for blob in blobs:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return "sha256:" + digest.hexdigest()


def replay() -> dict[str, object]:
    loaded_before = {
        name for name in sys.modules if name.startswith("hegel_machine.")
    }
    unexpected = loaded_before - _ALLOWED_PROJECT_MODULES
    if unexpected:
        raise RuntimeError(
            "target-free module closure violation: " + ",".join(sorted(unexpected))
        )

    sources = tuple(iter_shrink2_capacity_candidate_asts())
    programs = tuple(canonicalize_shrink2_source_ast(source) for source in sources)
    canonical_blobs = tuple(sorted({program.cbor_bytes for program in programs}))
    if len(sources) != EXPECTED_SHRINK2_SOURCE_COUNT:
        raise RuntimeError("shrink-2 capacity source count drift")
    if len(programs) != EXPECTED_SHRINK2_SOURCE_COUNT:
        raise RuntimeError("shrink-2 capacity acceptance count drift")
    if len(canonical_blobs) != EXPECTED_SHRINK2_SOURCE_COUNT:
        raise RuntimeError("shrink-2 capacity accepted-set cardinality drift")
    if any(
        decode_shrink2_canonical_ast(program.cbor_bytes) != program
        for program in programs
    ):
        raise RuntimeError("shrink-2 capacity formal round-trip mismatch")

    loaded_after = {
        name for name in sys.modules if name.startswith("hegel_machine.")
    }
    unexpected = loaded_after - _ALLOWED_PROJECT_MODULES
    if unexpected:
        raise RuntimeError(
            "target-free module closure changed during replay: "
            + ",".join(sorted(unexpected))
        )
    first = decode_shrink2_canonical_ast(canonical_blobs[0])
    last = decode_shrink2_canonical_ast(canonical_blobs[-1])
    return {
        "schema_version": "hegel-strict-capacity-replay-shrink2/1",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "generator_rule": SHRINK2_CAPACITY_GENERATOR_RULE,
        "active_rational_parameter_ids": list(ACTIVE_RATIONAL_PARAMETER_IDS),
        "rational_aggregate_map_ids": list(RATIONAL_ACTIVE_AGGREGATE_IDS),
        "constant_atom_count": len(shrink2_constant_atoms_v1()),
        "rational_aggregate_count": len(shrink2_rational_aggregate_leaves_v1()),
        "mixed_atom_count": len(shrink2_mixed_atoms_v1()),
        "source_candidate_count": len(sources),
        "accepted_source_count": len(programs),
        "accepted_unique_count": len(canonical_blobs),
        "rejected_count": 0,
        "rejection_counts": {},
        "rewrite_collapsed_count": 0,
        "accepted_set_commitment": _accepted_set_commitment(canonical_blobs),
        "first_canonical_cbor_hex": canonical_blobs[0].hex(),
        "first_canonical_ast_hash": first.hash_id,
        "last_canonical_cbor_hex": canonical_blobs[-1].hex(),
        "last_canonical_ast_hash": last.hash_id,
        "canonical_program_budget": 50_000,
        "first_out_of_budget_ordinal": None,
        "subset_status": "SUBSET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "loaded_hegel_modules": sorted(loaded_after),
        "target_or_split_modules_loaded": False,
    }


def _error_code(callable_: object) -> str:
    try:
        callable_()  # type: ignore[operator]
    except StrictAstError as error:
        return error.code
    raise RuntimeError("golden rejection vector was unexpectedly accepted")


def replay_golden() -> dict[str, object]:
    loaded_before = {
        name for name in sys.modules if name.startswith("hegel_machine.")
    }
    unexpected = loaded_before - _ALLOWED_PROJECT_MODULES
    if unexpected:
        raise RuntimeError(
            "target-free module closure violation: " + ",".join(sorted(unexpected))
        )

    vector_count = 0
    surviving_identity_checks = 0
    operator_preservation_checks = 0
    source_rejection_checks = 0
    source_boundary_checks = 0
    source_wide_integer_checks = 0
    source_malformed_checks = 0
    tombstone_priority_checks = 0
    formal_rejection_checks = 0
    formal_failure_code_checks = 0

    for numeric_id in ACTIVE_RATIONAL_PARAMETER_IDS:
        vector_count += 1
        source = ["scalar_const", numeric_id]
        parent = canonicalize_shrink1_source_ast(source)
        child = canonicalize_shrink2_source_ast(source)
        if child.cbor_bytes != parent.cbor_bytes or child.hash_id != parent.hash_id:
            raise RuntimeError("active rational parameter identity changed")
        surviving_identity_checks += 1

    surviving_sources = (
        ["add", ["scalar_const", 1], ["scalar_const", 5]],
        ["absolute", ["scalar_const", 1]],
        [
            "aggregate",
            "signed_balance_v1",
            "scope_all_observed_v1",
            "q0",
            [],
        ],
        [
            "less_equal",
            ["scalar_const", 1],
            ["aggregate", "sum_v1", "scope_primary_only_v1", "q1", []],
        ],
    )
    for source in surviving_sources:
        vector_count += 1
        parent = canonicalize_shrink1_source_ast(source)
        child = canonicalize_shrink2_source_ast(source)
        if child.cbor_bytes != parent.cbor_bytes or child.hash_id != parent.hash_id:
            raise RuntimeError("surviving AST bytes or hash changed")
        surviving_identity_checks += 1

    for numeric_id in TOMBSTONED_RATIONAL_PARAMETER_IDS:
        vector_count += 1
        code = _error_code(
            lambda numeric_id=numeric_id: canonicalize_shrink2_source_ast(
                ["scalar_const", numeric_id]
            )
        )
        if code != REMOVED_RATIONAL_PARAMETER_ERROR:
            raise RuntimeError("source rational tombstone code drift")
        source_rejection_checks += 1

    for map_id in TOMBSTONED_AGGREGATE_IDS:
        vector_count += 1
        code = _error_code(
            lambda map_id=map_id: canonicalize_shrink2_source_ast(
                ["aggregate", map_id, 0, 0, []]
            )
        )
        if code != REMOVED_AGGREGATE_ERROR:
            raise RuntimeError("source aggregate tombstone code drift")
        source_rejection_checks += 1

    vector_count += 1
    if _error_code(
        lambda: canonicalize_shrink2_source_ast(["scalar_const", 7])
    ) != UNKNOWN_RATIONAL_PARAMETER_ERROR:
        raise RuntimeError("reserved rational parameter code drift")
    source_rejection_checks += 1

    source_boundary_cases = (
        ["scalar_const", -1],
        ["scalar_const", -1, -1],
        ["scalar_const", -2, -1],
        ["bit_at", -1],
        ["context_flag", -1],
        ["task_flag", -1],
        ["scalar_const", "bad-index"],
        ["bit_at", True],
        ["aggregate", False, "scope_all_observed_v1", "q0", []],
        ["context_flag", []],
        ["aggregate", -1, "scope_all_observed_v1", "q0", []],
        ["aggregate", "sum_v1", -1, "q0", []],
        ["aggregate", "sum_v1", "scope_all_observed_v1", -1, []],
        [
            "approx_equal",
            ["scalar_const", 1],
            ["scalar_const", 5],
            -1,
        ],
        [
            "approx_equal",
            ["scalar_const", 1],
            ["scalar_const", 5],
            -1,
            -4,
        ],
    )
    for source in source_boundary_cases:
        vector_count += 1
        if _error_code(
            lambda source=source: canonicalize_shrink2_source_ast(source)
        ) != "REJECT_REGISTRY_INDEX_OUT_OF_RANGE":
            raise RuntimeError("source numeric boundary code drift")
        source_rejection_checks += 1
        source_boundary_checks += 1

    vector_count += 1
    malformed_tolerance = [
        "approx_equal",
        ["scalar_const", 1],
        ["scalar_const", 5],
        "not-an-index",
    ]
    if _error_code(
        lambda: canonicalize_shrink2_source_ast(malformed_tolerance)
    ) != "REJECT_MALFORMED_SOURCE_AST":
        raise RuntimeError("malformed tolerance shorthand code drift")
    source_rejection_checks += 1
    source_malformed_checks += 1

    wide_integer = 10**100
    vector_count += 1
    wide_active = canonicalize_shrink2_source_ast(
        ["scalar_const", wide_integer, wide_integer]
    )
    if wide_active.value != (1, (0, 0, 5)):
        raise RuntimeError("arbitrary-width rational alias resolution drift")
    source_wide_integer_checks += 1
    for source in (
        ["scalar_const", wide_integer, 1],
        ["scalar_const", wide_integer],
    ):
        vector_count += 1
        if _error_code(
            lambda source=source: canonicalize_shrink2_source_ast(source)
        ) != "REJECT_REGISTRY_INDEX_OUT_OF_RANGE":
            raise RuntimeError("arbitrary-width source boundary code drift")
        source_rejection_checks += 1
        source_wide_integer_checks += 1

    vector_count += 1
    mixed_tombstones = [
        "less_equal",
        ["scalar_const", 0],
        ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
    ]
    if _error_code(
        lambda: canonicalize_shrink2_source_ast(mixed_tombstones)
    ) != REMOVED_AGGREGATE_ERROR:
        raise RuntimeError("mixed tombstone priority drift")
    source_rejection_checks += 1
    tombstone_priority_checks += 1

    preserved_operators = (
        (
            ["add", ["scalar_const", 5], ["scalar_const", 5]],
            0,
            "82018402008300000583000005",
        ),
        (
            ["difference", ["scalar_const", 1], ["scalar_const", 5]],
            1,
            "82018402018300000183000005",
        ),
        (
            ["add", ["scalar_const", 1], ["scalar_const", 1]],
            0,
            "82018402008300000183000001",
        ),
        (
            ["difference", ["scalar_const", 5], ["scalar_const", 1]],
            1,
            "82018402018300000583000001",
        ),
    )
    for source, operator_id, expected_hex in preserved_operators:
        vector_count += 1
        program = canonicalize_shrink2_source_ast(source)
        if (
            program.value[1][0] != 2  # type: ignore[index]
            or program.value[1][1] != operator_id  # type: ignore[index]
            or program.cbor_bytes.hex() != expected_hex
            or decode_shrink2_canonical_ast(program.cbor_bytes) != program
        ):
            raise RuntimeError("inactive-result operator preservation drift")
        operator_preservation_checks += 1

    active_folds = (
        (["add", ["scalar_const", 1], ["scalar_const", 5]], 3),
        (["difference", ["scalar_const", 5], ["scalar_const", 5]], 3),
        (["absolute", ["scalar_const", 1]], 5),
    )
    for source, expected_id in active_folds:
        vector_count += 1
        program = canonicalize_shrink2_source_ast(source)
        if program.value != (1, (0, 0, expected_id)):
            raise RuntimeError("active-result fold drift")
        operator_preservation_checks += 1

    for numeric_id in TOMBSTONED_RATIONAL_PARAMETER_IDS:
        vector_count += 1
        payload = canonical_cbor_encode((1, (0, 0, numeric_id)))
        if _error_code(
            lambda payload=payload: decode_shrink2_canonical_ast(payload)
        ) != REMOVED_RATIONAL_PARAMETER_ERROR:
            raise RuntimeError("formal rational tombstone code drift")
        formal_rejection_checks += 1

    vector_count += 1
    formal_reserved = canonical_cbor_encode((1, (0, 0, 7)))
    if _error_code(
        lambda: decode_shrink2_canonical_ast(formal_reserved)
    ) != UNKNOWN_RATIONAL_PARAMETER_ERROR:
        raise RuntimeError("formal reserved parameter code drift")
    formal_rejection_checks += 1

    formal_failure_cases = (
        ((2, (0, 0, 0)), "REJECT_UNKNOWN_AST_SCHEMA"),
        ((1, (-1,)), "REJECT_UNKNOWN_EXPRESSION"),
        ((1, (0, -1)), "REJECT_UNKNOWN_EXPRESSION"),
        ((1, (0, 0, -1)), "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"),
        ((1, (1, 4, (0, 0, 3))), "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"),
        (
            (1, (2, 7, (0, 0, 3), (0, 0, 3))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (3, 1, (0, 0, 3), (0, 0, 3), 0)),
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
        ),
        (
            (1, (4, ((0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3)))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (0, 3, 0, 0, 0, ((0, False), (0, True)))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (0, 3, 0, 0, 0, ((0, False), (1, False), (2, False)))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (0, 3, 99, 0, 0, ((0,),))),
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
        ),
        (
            (1, (2, 2, (0, 0, 0), (0, 0, 3, 99))),
            "REJECT_NONCANONICAL_AST",
        ),
    )
    for formal, expected_code in formal_failure_cases:
        vector_count += 1
        payload = canonical_cbor_encode(formal)
        if _error_code(
            lambda payload=payload: decode_shrink2_canonical_ast(payload)
        ) != expected_code:
            raise RuntimeError("formal failure-code vector drift")
        formal_rejection_checks += 1
        formal_failure_code_checks += 1

    loaded_after = {
        name for name in sys.modules if name.startswith("hegel_machine.")
    }
    unexpected = loaded_after - _ALLOWED_PROJECT_MODULES
    if unexpected:
        raise RuntimeError(
            "target-free module closure changed during golden replay: "
            + ",".join(sorted(unexpected))
        )
    if vector_count != 59:
        raise RuntimeError("golden vector count drift")
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink2-golden/1",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "rational_parameter_registry_namespace": (
            RATIONAL_PARAMETER_REGISTRY_NAMESPACE
        ),
        "active_rational_parameter_ids": list(ACTIVE_RATIONAL_PARAMETER_IDS),
        "tombstoned_rational_parameter_ids": list(
            TOMBSTONED_RATIONAL_PARAMETER_IDS
        ),
        "reserved_rational_parameter_ids": list(RESERVED_RATIONAL_PARAMETER_IDS),
        "vector_count": vector_count,
        "passed_count": vector_count,
        "surviving_identity_checks": surviving_identity_checks,
        "operator_preservation_checks": operator_preservation_checks,
        "source_rejection_checks": source_rejection_checks,
        "source_boundary_checks": source_boundary_checks,
        "source_wide_integer_checks": source_wide_integer_checks,
        "source_malformed_checks": source_malformed_checks,
        "tombstone_priority_checks": tombstone_priority_checks,
        "formal_rejection_checks": formal_rejection_checks,
        "formal_failure_code_checks": formal_failure_code_checks,
        "execution_state": "NOT_RUN",
        "closure_executed": False,
        "formal_roots_generated": False,
        "formal_roots": None,
        "loaded_hegel_modules": sorted(loaded_after),
        "target_or_split_modules_loaded": False,
    }


if __name__ == "__main__":
    report = replay() if _REPLAY_MODE == "--capacity-replay" else replay_golden()
    sys.stdout.write(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )

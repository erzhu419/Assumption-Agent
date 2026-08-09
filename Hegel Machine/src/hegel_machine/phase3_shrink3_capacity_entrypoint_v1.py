"""Direct target-free strict/capacity replay for shrink step 3.

Qualification invokes this file with ``python -I -S -B`` so the historical
public package initializer cannot load target or split APIs.  The module
allowlist is exact and any dependency-closure expansion fails closed.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


if __package__ not in {None, ""}:
    raise RuntimeError("shrink-3 replay requires its direct file entrypoint")
if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
    raise RuntimeError("shrink-3 replay requires python -I -S -B")
if len(sys.argv) != 2 or sys.argv[1] not in {
    "--capacity-replay",
    "--golden-replay",
}:
    raise RuntimeError(
        "shrink-3 replay requires exactly --capacity-replay or --golden-replay"
    )
_REPLAY_MODE = sys.argv[1]

package = ModuleType("hegel_machine")
package.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package
__package__ = "hegel_machine"

from .phase3_m3_shrink3_core_v1 import (  # noqa: E402
    ACTIVE_FORMAL_BINARY_OPERATOR_IDS,
    ACTIVE_SOURCE_BINARY_OPERATOR_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    REMOVED_BINARY_OPERATOR_ERROR,
    RESERVED_BINARY_OPERATOR_IDS,
    SHRINK_STEP_ID,
    SOURCE_ALIAS_BINARY_OPERATOR_IDS,
    TOMBSTONED_BINARY_OPERATOR_IDS,
)
from .phase3_shrink3_capacity_v1 import (  # noqa: E402
    EXPECTED_SHRINK3_SOURCE_COUNT,
    SHRINK3_CAPACITY_GENERATOR_RULE,
    iter_shrink3_capacity_candidate_asts,
    shrink3_constant_atoms_v1,
    shrink3_mixed_atoms_v1,
    shrink3_rational_aggregate_leaves_v1,
)
from .phase3_shrink3_golden_vectors_v1 import (  # noqa: E402
    ACCEPT_PARENT_IDENTITY,
    STRICT_GOLDEN_VECTORS_V1,
    accepted_outcome_bytes,
    rejected_outcome_bytes,
    strict_golden_manifest_root_v1,
    strict_golden_outcome_root_v1,
)
from .strict_ast_shrink2_v1 import (  # noqa: E402
    canonicalize_shrink2_source_ast,
)
from .strict_ast_shrink3_v1 import (  # noqa: E402
    canonicalize_shrink3_source_ast,
    decode_shrink3_canonical_ast,
)
from .strict_ast_v1 import StrictAstError  # noqa: E402
from .strict_cbor_v1 import canonical_cbor_encode  # noqa: E402


_ALLOWED_PROJECT_MODULES = {
    "hegel_machine.hashing",
    "hegel_machine.phase3_m3_dsl_core_v1",
    "hegel_machine.phase3_m3_shrink1_core_v1",
    "hegel_machine.phase3_m3_shrink2_core_v1",
    "hegel_machine.phase3_m3_shrink3_core_v1",
    "hegel_machine.phase3_shrink2_capacity_v1",
    "hegel_machine.phase3_shrink3_capacity_v1",
    "hegel_machine.phase3_shrink3_golden_vectors_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_shrink3_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
}
_CAPACITY_SET_DOMAIN = b"HEGEL/STRICT_CAPACITY_SET/V1"


def _loaded_project_modules() -> tuple[str, ...]:
    return tuple(sorted(name for name in sys.modules if name.startswith("hegel_machine.")))


def _assert_target_free() -> tuple[str, ...]:
    loaded = _loaded_project_modules()
    unexpected = set(loaded) - _ALLOWED_PROJECT_MODULES
    if unexpected:
        raise RuntimeError(
            "target-free module closure violation: "
            + ",".join(sorted(unexpected))
        )
    return loaded


def _accepted_set_commitment(blobs: tuple[bytes, ...]) -> str:
    digest = sha256()
    digest.update(_CAPACITY_SET_DOMAIN)
    digest.update(b"\x00")
    for blob in blobs:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return "sha256:" + digest.hexdigest()


def _error_code(callable_: object) -> str:
    try:
        callable_()  # type: ignore[operator]
    except StrictAstError as error:
        return error.code
    raise RuntimeError("golden rejection vector was unexpectedly accepted")


def _expect_error(callable_: object, expected: str, label: str) -> None:
    observed = _error_code(callable_)
    if observed != expected:
        raise RuntimeError(
            f"{label} error drift: expected {expected}, observed {observed}"
        )


def _nonconstant_binary(name: str) -> list[object]:
    return [
        name,
        ["bit_to_scalar", ["bit_at", 0]],
        ["scalar_const", 1, 1],
    ]


def replay_capacity() -> dict[str, object]:
    """Replay the exact inherited survivor subset under both DSL versions."""

    _assert_target_free()
    sources = tuple(iter_shrink3_capacity_candidate_asts())
    parents = tuple(canonicalize_shrink2_source_ast(source) for source in sources)
    children = tuple(canonicalize_shrink3_source_ast(source) for source in sources)
    if len(sources) != EXPECTED_SHRINK3_SOURCE_COUNT:
        raise RuntimeError("shrink-3 survivor source count drift")
    if len(parents) != len(children) or any(
        parent.cbor_bytes != child.cbor_bytes or parent.hash_id != child.hash_id
        for parent, child in zip(parents, children)
    ):
        raise RuntimeError("shrink-3 survivor identity differs from parent")
    canonical_blobs = tuple(sorted({program.cbor_bytes for program in children}))
    if len(canonical_blobs) != EXPECTED_SHRINK3_SOURCE_COUNT:
        raise RuntimeError("shrink-3 survivor accepted-set cardinality drift")
    if any(
        decode_shrink3_canonical_ast(program.cbor_bytes) != program
        for program in children
    ):
        raise RuntimeError("shrink-3 survivor formal round-trip mismatch")

    loaded = _assert_target_free()
    first = decode_shrink3_canonical_ast(canonical_blobs[0])
    last = decode_shrink3_canonical_ast(canonical_blobs[-1])
    return {
        "schema_version": "hegel-strict-capacity-replay-shrink3/1",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "generator_rule": SHRINK3_CAPACITY_GENERATOR_RULE,
        "removed_binary_operator_ids": list(TOMBSTONED_BINARY_OPERATOR_IDS),
        "retained_difference_id": 1,
        "constant_atom_count": len(shrink3_constant_atoms_v1()),
        "rational_aggregate_count": len(
            shrink3_rational_aggregate_leaves_v1()
        ),
        "mixed_atom_count": len(shrink3_mixed_atoms_v1()),
        "source_candidate_count": len(sources),
        "accepted_source_count": len(children),
        "accepted_unique_count": len(canonical_blobs),
        "parent_identity_match_count": len(children),
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
        "subset_status": "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "loaded_hegel_modules": list(loaded),
        "target_or_split_modules_loaded": False,
    }


def replay_golden() -> dict[str, object]:
    """Replay the sealed 36-vector manifest and bind every outcome."""

    _assert_target_free()
    counts = {
        "surviving_identity_checks": 0,
        "source_add_rejection_checks": 0,
        "source_priority_checks": 0,
        "formal_add_rejection_checks": 0,
        "formal_priority_checks": 0,
        "formal_shape_priority_checks": 0,
        "formal_alias_or_reserved_checks": 0,
    }
    outcomes: dict[str, bytes] = {}
    for vector in STRICT_GOLDEN_VECTORS_V1:
        if vector.expected_disposition == ACCEPT_PARENT_IDENTITY:
            if vector.boundary != "SOURCE_JSON":
                raise RuntimeError("accepted golden vector must use source boundary")
            source = json.loads(vector.input_wire.decode("utf-8"))
            parent = canonicalize_shrink2_source_ast(source)
            child = canonicalize_shrink3_source_ast(source)
            if (
                parent.cbor_bytes != child.cbor_bytes
                or parent.hash_id != child.hash_id
                or decode_shrink3_canonical_ast(child.cbor_bytes) != child
            ):
                raise RuntimeError(
                    f"{vector.vector_id} surviving parent identity changed"
                )
            outcomes[vector.vector_id] = accepted_outcome_bytes(
                child.cbor_bytes, child.digest
            )
        else:
            if vector.boundary == "SOURCE_JSON":
                source = json.loads(vector.input_wire.decode("utf-8"))
                observed = _error_code(
                    lambda source=source: canonicalize_shrink3_source_ast(source)
                )
            else:
                payload = vector.input_wire
                observed = _error_code(
                    lambda payload=payload: decode_shrink3_canonical_ast(payload)
                )
            if observed != vector.expected_disposition:
                raise RuntimeError(
                    f"{vector.vector_id} expected {vector.expected_disposition}, "
                    f"observed {observed}"
                )
            outcomes[vector.vector_id] = rejected_outcome_bytes(observed)
        counts[vector.category] += 1

    if len(outcomes) != 36 or counts != {
        "surviving_identity_checks": 8,
        "source_add_rejection_checks": 4,
        "source_priority_checks": 6,
        "formal_add_rejection_checks": 3,
        "formal_priority_checks": 6,
        "formal_shape_priority_checks": 6,
        "formal_alias_or_reserved_checks": 3,
    }:
        raise RuntimeError("sealed golden vector category count drift")

    loaded = _assert_target_free()
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink3-golden/2",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "active_source_binary_operator_ids": list(
            ACTIVE_SOURCE_BINARY_OPERATOR_IDS
        ),
        "active_formal_binary_operator_ids": list(
            ACTIVE_FORMAL_BINARY_OPERATOR_IDS
        ),
        "source_alias_binary_operator_ids": list(
            SOURCE_ALIAS_BINARY_OPERATOR_IDS
        ),
        "tombstoned_binary_operator_ids": list(
            TOMBSTONED_BINARY_OPERATOR_IDS
        ),
        "reserved_binary_operator_ids": list(RESERVED_BINARY_OPERATOR_IDS),
        "removed_binary_operator_error": REMOVED_BINARY_OPERATOR_ERROR,
        "golden_vector_manifest_root": strict_golden_manifest_root_v1(),
        "golden_outcome_root": strict_golden_outcome_root_v1(outcomes),
        "ordered_vector_ids": list(outcomes),
        "vector_count": len(outcomes),
        "passed_count": len(outcomes),
        **counts,
        "execution_state": "NOT_RUN",
        "closure_executed": False,
        "formal_roots_generated": False,
        "formal_roots": None,
        "loaded_hegel_modules": list(loaded),
        "target_or_split_modules_loaded": False,
    }


if __name__ == "__main__":
    report = replay_capacity() if _REPLAY_MODE == "--capacity-replay" else replay_golden()
    sys.stdout.write(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )

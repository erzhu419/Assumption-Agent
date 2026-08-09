"""Direct target-free strict/capacity replay for shrink step 4."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


if __package__ not in {None, ""}:
    raise RuntimeError("shrink-4 replay requires its direct file entrypoint")
if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
    raise RuntimeError("shrink-4 replay requires python -I -S -B")
if len(sys.argv) != 2 or sys.argv[1] not in {
    "--capacity-replay",
    "--golden-replay",
}:
    raise RuntimeError(
        "shrink-4 replay requires exactly --capacity-replay or --golden-replay"
    )
_REPLAY_MODE = sys.argv[1]

package = ModuleType("hegel_machine")
package.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package
__package__ = "hegel_machine"

from .phase3_m3_bounded_enumerator_v1 import program_mdl_length_q32  # noqa: E402
from .phase3_m3_shrink4_core_v1 import (  # noqa: E402
    ACTIVE_FORMAL_BINARY_OPERATOR_IDS,
    ACTIVE_SOURCE_BINARY_OPERATOR_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    MAX_TOP_LEVEL_CLAUSES,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    REMOVED_BINARY_OPERATOR_ERROR,
    RESERVED_BINARY_OPERATOR_IDS,
    SHRINK_STEP_ID,
    SOURCE_ALIAS_BINARY_OPERATOR_IDS,
    TOMBSTONED_BINARY_OPERATOR_IDS,
)
from .phase3_shrink4_capacity_v1 import (  # noqa: E402
    EXPECTED_SHRINK4_SOURCE_COUNT,
    SHRINK4_CAPACITY_GENERATOR_RULE,
    iter_shrink4_capacity_candidate_asts,
    shrink4_constant_atoms_v1,
    shrink4_mixed_atoms_v1,
    shrink4_rational_aggregate_leaves_v1,
)
from .phase3_shrink4_golden_vectors_v1 import (  # noqa: E402
    ACCEPT_PARENT_IDENTITY,
    STRICT_GOLDEN_VECTORS_V1,
    accepted_outcome_bytes,
    rejected_outcome_bytes,
    strict_golden_manifest_root_v1,
    strict_golden_outcome_root_v1,
)
from .strict_ast_shrink3_v1 import (  # noqa: E402
    canonicalize_shrink3_source_ast,
    decode_shrink3_canonical_ast,
)
from .strict_ast_shrink4_v1 import (  # noqa: E402
    canonicalize_shrink4_source_ast,
    decode_shrink4_canonical_ast,
)
from .strict_ast_v1 import StrictAstError  # noqa: E402


_ALLOWED_PROJECT_MODULES = {
    "hegel_machine.hashing",
    "hegel_machine.phase3_m3_bounded_enumerator_v1",
    "hegel_machine.phase3_m3_dsl_core_v1",
    "hegel_machine.phase3_m3_record_wire_v1",
    "hegel_machine.phase3_m3_shrink1_core_v1",
    "hegel_machine.phase3_m3_shrink2_core_v1",
    "hegel_machine.phase3_m3_shrink3_core_v1",
    "hegel_machine.phase3_m3_shrink4_core_v1",
    "hegel_machine.phase3_shrink2_capacity_v1",
    "hegel_machine.phase3_shrink3_capacity_v1",
    "hegel_machine.phase3_shrink4_capacity_v1",
    "hegel_machine.phase3_shrink4_golden_vectors_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_shrink3_v1",
    "hegel_machine.strict_ast_shrink4_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
}
_CAPACITY_SET_DOMAIN = b"HEGEL/STRICT_CAPACITY_SET/V1"


def _loaded_project_modules() -> tuple[str, ...]:
    return tuple(sorted(name for name in sys.modules if name.startswith("hegel_machine.")))


def _assert_target_free() -> tuple[str, ...]:
    loaded = _loaded_project_modules()
    unexpected = set(loaded) - _ALLOWED_PROJECT_MODULES
    missing = _ALLOWED_PROJECT_MODULES - set(loaded)
    if unexpected or missing:
        raise RuntimeError(
            "target-free module closure violation: "
            f"missing={sorted(missing)!r}; unexpected={sorted(unexpected)!r}"
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


def replay_capacity() -> dict[str, object]:
    """Replay every one of the 2,160 inherited AND2 survivors."""

    _assert_target_free()
    sources = tuple(iter_shrink4_capacity_candidate_asts())
    parents = tuple(canonicalize_shrink3_source_ast(source) for source in sources)
    children = tuple(canonicalize_shrink4_source_ast(source) for source in sources)
    if len(sources) != EXPECTED_SHRINK4_SOURCE_COUNT:
        raise RuntimeError("shrink-4 survivor source count drift")
    if any(
        child.metrics.top_level_clause_count != MAX_TOP_LEVEL_CLAUSES
        for child in children
    ):
        raise RuntimeError("shrink-4 capacity source is not normalized AND2")
    if any(
        parent.cbor_bytes != child.cbor_bytes
        or parent.hash_id != child.hash_id
        or program_mdl_length_q32(parent) != program_mdl_length_q32(child)
        for parent, child in zip(parents, children)
    ):
        raise RuntimeError("shrink-4 survivor identity or MDL differs from parent")
    canonical_blobs = tuple(sorted({program.cbor_bytes for program in children}))
    if len(canonical_blobs) != EXPECTED_SHRINK4_SOURCE_COUNT:
        raise RuntimeError("shrink-4 survivor accepted-set cardinality drift")
    if any(
        decode_shrink4_canonical_ast(child.cbor_bytes) != child
        for child in children
    ):
        raise RuntimeError("shrink-4 survivor formal round-trip mismatch")

    loaded = _assert_target_free()
    first = decode_shrink4_canonical_ast(canonical_blobs[0])
    last = decode_shrink4_canonical_ast(canonical_blobs[-1])
    return {
        "schema_version": "hegel-strict-capacity-replay-shrink4/1",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "generator_rule": SHRINK4_CAPACITY_GENERATOR_RULE,
        "maximum_top_level_clauses": MAX_TOP_LEVEL_CLAUSES,
        "constant_atom_count": len(shrink4_constant_atoms_v1()),
        "rational_aggregate_count": len(shrink4_rational_aggregate_leaves_v1()),
        "mixed_atom_count": len(shrink4_mixed_atoms_v1()),
        "source_candidate_count": len(sources),
        "normalized_and2_count": len(children),
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
        "removed_binary_operator_ids": list(TOMBSTONED_BINARY_OPERATOR_IDS),
        "retained_difference_id": 1,
        "canonical_program_budget": 50_000,
        "first_out_of_budget_ordinal": None,
        "subset_status": "FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "loaded_hegel_modules": list(loaded),
        "target_or_split_modules_loaded": False,
    }


def replay_golden() -> dict[str, object]:
    """Replay and commit every outcome in the sealed 22-vector manifest."""

    _assert_target_free()
    counts: dict[str, int] = {}
    outcomes: dict[str, bytes] = {}
    for vector in STRICT_GOLDEN_VECTORS_V1:
        if vector.expected_disposition == ACCEPT_PARENT_IDENTITY:
            if vector.boundary == "SOURCE_JSON":
                source = json.loads(vector.input_wire.decode("utf-8"))
                parent = canonicalize_shrink3_source_ast(source)
                child = canonicalize_shrink4_source_ast(source)
            else:
                parent = decode_shrink3_canonical_ast(vector.input_wire)
                child = decode_shrink4_canonical_ast(vector.input_wire)
            if (
                parent.cbor_bytes != child.cbor_bytes
                or parent.hash_id != child.hash_id
                or program_mdl_length_q32(parent) != program_mdl_length_q32(child)
                or decode_shrink4_canonical_ast(child.cbor_bytes) != child
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
                    lambda source=source: canonicalize_shrink4_source_ast(source)
                )
            else:
                payload = vector.input_wire
                observed = _error_code(
                    lambda payload=payload: decode_shrink4_canonical_ast(payload)
                )
            if observed != vector.expected_disposition:
                raise RuntimeError(
                    f"{vector.vector_id} expected {vector.expected_disposition}, "
                    f"observed {observed}"
                )
            outcomes[vector.vector_id] = rejected_outcome_bytes(observed)
        counts[vector.category] = counts.get(vector.category, 0) + 1

    if len(outcomes) != len(STRICT_GOLDEN_VECTORS_V1):
        raise RuntimeError("sealed golden vector outcome count drift")
    loaded = _assert_target_free()
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink4-golden/1",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "maximum_top_level_clauses": MAX_TOP_LEVEL_CLAUSES,
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
        "closure_executed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "loaded_hegel_modules": list(loaded),
        "target_or_split_modules_loaded": False,
    }


if __name__ == "__main__":
    report = replay_capacity() if _REPLAY_MODE == "--capacity-replay" else replay_golden()
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")

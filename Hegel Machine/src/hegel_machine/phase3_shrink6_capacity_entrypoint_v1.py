"""Direct target-free strict/capacity replay for shrink step 6."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


if __package__ not in {None, ""}:
    raise RuntimeError("shrink-6 replay requires its direct file entrypoint")
if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.dont_write_bytecode:
    raise RuntimeError("shrink-6 replay requires python -I -S -B")
if len(sys.argv) != 2 or sys.argv[1] not in {
    "--capacity-replay",
    "--golden-replay",
}:
    raise RuntimeError(
        "shrink-6 replay requires exactly --capacity-replay or --golden-replay"
    )
_REPLAY_MODE = sys.argv[1]

package = ModuleType("hegel_machine")
package.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package
__package__ = "hegel_machine"

from .phase3_m3_bounded_enumerator_v1 import program_mdl_length_q32  # noqa: E402
from .phase3_m3_shrink6_core_v1 import (  # noqa: E402
    ACTIVE_FORMAL_BINARY_OPERATOR_IDS,
    ACTIVE_SOURCE_BINARY_OPERATOR_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    MAXIMUM_AST_DEPTH,
    MAXIMUM_AST_NODE_COUNT,
    MAX_TOP_LEVEL_CLAUSES,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    REMOVED_BINARY_OPERATOR_ERROR,
    RESERVED_BINARY_OPERATOR_IDS,
    SHRINK_STEP_ID,
    SOURCE_ALIAS_BINARY_OPERATOR_IDS,
    TOMBSTONED_BINARY_OPERATOR_IDS,
)
from .phase3_shrink6_capacity_v1 import (  # noqa: E402
    EXPECTED_SHRINK6_CHALLENGE_PARENT_ACCEPTED_SOURCE_COUNT,
    EXPECTED_SHRINK6_CHALLENGE_SOURCE_COUNT,
    EXPECTED_SHRINK6_FULL_SURVIVOR_SOURCE_COUNT,
    EXPECTED_SHRINK6_FULL_SURVIVOR_UNIQUE_COUNT,
    EXPECTED_SHRINK6_INHERITED_SURVIVOR_SOURCE_COUNT,
    EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_SOURCE_COUNT,
    EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_UNIQUE_COUNT,
    EXPECTED_SHRINK6_PARENT_CANONICAL_UNIQUE_COUNT,
    EXPECTED_SHRINK6_PARENT_ONLY_FAMILY_COUNTS,
    EXPECTED_SHRINK6_PARENT_ONLY_SOURCE_COUNT,
    EXPECTED_SHRINK6_PARENT_ONLY_UNIQUE_COUNT,
    SHRINK6_CAPACITY_GENERATOR_RULE,
    SUBSET_STATUS,
    iter_shrink6_depth4_challenge_sources_v1,
    iter_shrink6_inherited_survivor_candidate_asts,
)
from .phase3_shrink5_capacity_v1 import (  # noqa: E402
    shrink5_constant_atoms_v1,
    shrink5_mixed_atoms_v1,
    shrink5_rational_aggregate_leaves_v1,
)
from .phase3_shrink6_golden_vectors_v1 import (  # noqa: E402
    ACCEPT_PARENT_IDENTITY,
    STRICT_GOLDEN_VECTORS_V1,
    accepted_outcome_bytes,
    rejected_outcome_bytes,
    strict_golden_manifest_root_v1,
    strict_golden_outcome_root_v1,
)
from .strict_ast_shrink5_v1 import (  # noqa: E402
    canonicalize_shrink5_source_ast,
    decode_shrink5_canonical_ast,
)
from .strict_ast_shrink6_v1 import (  # noqa: E402
    canonicalize_shrink6_source_ast,
    decode_shrink6_canonical_ast,
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
    "hegel_machine.phase3_m3_shrink5_core_v1",
    "hegel_machine.phase3_m3_shrink6_core_v1",
    "hegel_machine.phase3_shrink2_capacity_v1",
    "hegel_machine.phase3_shrink3_capacity_v1",
    "hegel_machine.phase3_shrink4_capacity_v1",
    "hegel_machine.phase3_shrink5_capacity_v1",
    "hegel_machine.phase3_shrink6_capacity_v1",
    "hegel_machine.phase3_shrink6_golden_vectors_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_shrink3_v1",
    "hegel_machine.strict_ast_shrink4_v1",
    "hegel_machine.strict_ast_shrink5_v1",
    "hegel_machine.strict_ast_shrink6_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
}
_CHALLENGE_SOURCE_LATTICE_DOMAIN = (
    b"HEGEL/SHRINK6/STRICT_DEPTH4_CHALLENGE_SOURCE_LATTICE/V1"
)
_INHERITED_SURVIVOR_SET_DOMAIN = (
    b"HEGEL/SHRINK6/STRICT_INHERITED_SURVIVOR_SET/V1"
)
_NORMALIZED_SURVIVOR_SET_DOMAIN = (
    b"HEGEL/SHRINK6/STRICT_NORMALIZED_SURVIVOR_SET/V1"
)
_FULL_SURVIVOR_SET_DOMAIN = b"HEGEL/SHRINK6/STRICT_FULL_SURVIVOR_SET/V1"
_PARENT_CANONICAL_SET_DOMAIN = (
    b"HEGEL/SHRINK6/STRICT_CHALLENGE_PARENT_CANONICAL_SET/V1"
)
_PARENT_ONLY_DEPTH4_SET_DOMAIN = (
    b"HEGEL/SHRINK6/STRICT_PARENT_ONLY_DEPTH4_NODE6_SET/V1"
)
_PARENT_ONLY_SOURCE_REJECTION_DOMAIN = (
    b"HEGEL/SHRINK6/STRICT_PARENT_ONLY_DEPTH4_SOURCE_REJECTION/V1"
)
_PARENT_ONLY_FORMAL_REJECTION_DOMAIN = (
    b"HEGEL/SHRINK6/STRICT_PARENT_ONLY_DEPTH4_FORMAL_REJECTION/V1"
)


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


def _set_commitment(domain: bytes, blobs: tuple[bytes, ...]) -> str:
    digest = sha256()
    digest.update(domain)
    digest.update(b"\x00")
    for blob in blobs:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return "sha256:" + digest.hexdigest()


def _source_lattice_commitment(rows: tuple[object, ...]) -> str:
    digest = sha256()
    digest.update(_CHALLENGE_SOURCE_LATTICE_DOMAIN)
    digest.update(b"\x00")
    for row in rows:
        family = row.family.encode("ascii")  # type: ignore[attr-defined]
        source_wire = json.dumps(
            row.source_ast,  # type: ignore[attr-defined]
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        for field in (family, source_wire):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _rejection_commitment(domain: bytes, blobs: tuple[bytes, ...]) -> str:
    digest = sha256()
    digest.update(domain)
    digest.update(b"\x00")
    error = b"REJECT_STRUCTURAL_LIMIT"
    for blob in blobs:
        for field in (blob, error):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _error_code(callable_: object) -> str:
    try:
        callable_()  # type: ignore[operator]
    except StrictAstError as error:
        return error.code
    raise RuntimeError("golden rejection vector was unexpectedly accepted")


def replay_capacity() -> dict[str, object]:
    """Replay the frozen depth-four challenge lattice; never infer closure."""

    _assert_target_free()
    inherited_sources = tuple(iter_shrink6_inherited_survivor_candidate_asts())
    challenge_rows = tuple(iter_shrink6_depth4_challenge_sources_v1())
    if len(inherited_sources) != EXPECTED_SHRINK6_INHERITED_SURVIVOR_SOURCE_COUNT:
        raise RuntimeError("shrink-6 inherited survivor source count drift")
    if len(challenge_rows) != EXPECTED_SHRINK6_CHALLENGE_SOURCE_COUNT:
        raise RuntimeError("shrink-6 challenge source count drift")

    challenge_parents = tuple(
        canonicalize_shrink5_source_ast(row.source_ast) for row in challenge_rows
    )
    if len(challenge_parents) != EXPECTED_SHRINK6_CHALLENGE_PARENT_ACCEPTED_SOURCE_COUNT:
        raise RuntimeError("shrink-6 challenge parent acceptance count drift")
    challenge_parent_blobs = tuple(
        sorted({program.cbor_bytes for program in challenge_parents})
    )
    if len(challenge_parent_blobs) != EXPECTED_SHRINK6_PARENT_CANONICAL_UNIQUE_COUNT:
        raise RuntimeError("shrink-6 challenge parent canonical count drift")

    normalized_rows = tuple(
        row
        for row, parent in zip(challenge_rows, challenge_parents)
        if parent.metrics.depth <= MAXIMUM_AST_DEPTH
    )
    normalized_parents = tuple(
        parent
        for parent in challenge_parents
        if parent.metrics.depth <= MAXIMUM_AST_DEPTH
    )
    parent_only_rows = tuple(
        row
        for row, parent in zip(challenge_rows, challenge_parents)
        if parent.metrics.depth > MAXIMUM_AST_DEPTH
    )
    parent_only = tuple(
        parent
        for parent in challenge_parents
        if parent.metrics.depth > MAXIMUM_AST_DEPTH
    )
    if len(normalized_rows) != EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_SOURCE_COUNT:
        raise RuntimeError("shrink-6 normalized survivor source count drift")
    if len(parent_only_rows) != EXPECTED_SHRINK6_PARENT_ONLY_SOURCE_COUNT:
        raise RuntimeError("shrink-6 parent-only source count drift")
    if any(
        program.metrics.depth != 4 or program.metrics.node_count != 6
        for program in parent_only
    ):
        raise RuntimeError("shrink-6 parent-only set is not exact depth-four node-six")

    inherited_parents = tuple(
        canonicalize_shrink5_source_ast(source) for source in inherited_sources
    )
    inherited_children = tuple(
        canonicalize_shrink6_source_ast(source) for source in inherited_sources
    )
    normalized_children = tuple(
        canonicalize_shrink6_source_ast(row.source_ast) for row in normalized_rows
    )
    full_sources = inherited_sources + tuple(row.source_ast for row in normalized_rows)
    full_parents = inherited_parents + normalized_parents
    survivors = inherited_children + normalized_children
    if len(full_sources) != EXPECTED_SHRINK6_FULL_SURVIVOR_SOURCE_COUNT:
        raise RuntimeError("shrink-6 full survivor source count drift")
    if any(
        parent.cbor_bytes != child.cbor_bytes
        or parent.hash_id != child.hash_id
        or program_mdl_length_q32(parent) != program_mdl_length_q32(child)
        or decode_shrink6_canonical_ast(child.cbor_bytes) != child
        for parent, child in zip(full_parents, survivors)
    ):
        raise RuntimeError("shrink-6 survivor identity or MDL differs from parent")

    inherited_blobs = tuple(
        sorted({program.cbor_bytes for program in inherited_children})
    )
    normalized_blobs = tuple(
        sorted({program.cbor_bytes for program in normalized_children})
    )
    survivor_blobs = tuple(sorted({program.cbor_bytes for program in survivors}))
    if len(inherited_blobs) != EXPECTED_SHRINK6_INHERITED_SURVIVOR_SOURCE_COUNT:
        raise RuntimeError("shrink-6 inherited survivor unique count drift")
    if len(normalized_blobs) != EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_UNIQUE_COUNT:
        raise RuntimeError("shrink-6 normalized survivor unique count drift")
    if len(survivor_blobs) != EXPECTED_SHRINK6_FULL_SURVIVOR_UNIQUE_COUNT:
        raise RuntimeError("shrink-6 full survivor unique count drift")

    parent_only_blobs = tuple(sorted({program.cbor_bytes for program in parent_only}))
    if len(parent_only_blobs) != EXPECTED_SHRINK6_PARENT_ONLY_UNIQUE_COUNT:
        raise RuntimeError("shrink-6 parent-only unique count drift")
    source_errors = tuple(
        _error_code(
            lambda source=row.source_ast: canonicalize_shrink6_source_ast(source)
        )
        for row in parent_only_rows
    )
    formal_errors = tuple(
        _error_code(
            lambda payload=program.cbor_bytes: decode_shrink6_canonical_ast(payload)
        )
        for program in parent_only
    )
    if Counter(source_errors) != {"REJECT_STRUCTURAL_LIMIT": len(parent_only)}:
        raise RuntimeError("shrink-6 source boundary disposition drift")
    if Counter(formal_errors) != {"REJECT_STRUCTURAL_LIMIT": len(parent_only)}:
        raise RuntimeError("shrink-6 formal boundary disposition drift")

    challenge_family_counts = Counter(row.family for row in challenge_rows)
    normalized_family_counts = Counter(row.family for row in normalized_rows)
    parent_only_family_counts = Counter(row.family for row in parent_only_rows)
    if dict(parent_only_family_counts) != EXPECTED_SHRINK6_PARENT_ONLY_FAMILY_COUNTS:
        raise RuntimeError("shrink-6 parent-only family partition drift")

    loaded = _assert_target_free()
    first = decode_shrink6_canonical_ast(survivor_blobs[0])
    last = decode_shrink6_canonical_ast(survivor_blobs[-1])
    return {
        "schema_version": "hegel-strict-capacity-replay-shrink6/1",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "generator_rule": SHRINK6_CAPACITY_GENERATOR_RULE,
        "maximum_ast_depth": MAXIMUM_AST_DEPTH,
        "maximum_ast_node_count": MAXIMUM_AST_NODE_COUNT,
        "maximum_top_level_clauses": MAX_TOP_LEVEL_CLAUSES,
        "constant_atom_count": len(shrink5_constant_atoms_v1()),
        "rational_aggregate_count": len(shrink5_rational_aggregate_leaves_v1()),
        "mixed_atom_count": len(shrink5_mixed_atoms_v1()),
        "challenge_source_candidate_count": len(challenge_rows),
        "challenge_source_family_counts": dict(challenge_family_counts),
        "challenge_source_lattice_commitment": _source_lattice_commitment(
            challenge_rows
        ),
        "challenge_parent_accepted_count": len(challenge_parents),
        "challenge_parent_canonical_unique_count": len(challenge_parent_blobs),
        "challenge_parent_canonical_set_commitment": _set_commitment(
            _PARENT_CANONICAL_SET_DOMAIN, challenge_parent_blobs
        ),
        "normalized_survivor_source_count": len(normalized_rows),
        "normalized_survivor_unique_count": len(normalized_blobs),
        "normalized_survivor_source_family_counts": dict(
            normalized_family_counts
        ),
        "normalized_survivor_set_commitment": _set_commitment(
            _NORMALIZED_SURVIVOR_SET_DOMAIN, normalized_blobs
        ),
        "inherited_survivor_source_count": len(inherited_sources),
        "inherited_survivor_unique_count": len(inherited_blobs),
        "inherited_survivor_set_commitment": _set_commitment(
            _INHERITED_SURVIVOR_SET_DOMAIN, inherited_blobs
        ),
        "survivor_source_candidate_count": len(full_sources),
        "survivor_accepted_count": len(survivors),
        "survivor_unique_count": len(survivor_blobs),
        "survivor_parent_identity_match_count": len(survivors),
        "survivor_rejected_count": 0,
        "survivor_rejection_counts": {},
        "survivor_accepted_set_commitment": _set_commitment(
            _FULL_SURVIVOR_SET_DOMAIN, survivor_blobs
        ),
        "first_survivor_canonical_cbor_hex": survivor_blobs[0].hex(),
        "first_survivor_canonical_ast_hash": first.hash_id,
        "last_survivor_canonical_cbor_hex": survivor_blobs[-1].hex(),
        "last_survivor_canonical_ast_hash": last.hash_id,
        "parent_only_source_candidate_count": len(parent_only_rows),
        "parent_only_parent_accepted_count": len(parent_only),
        "parent_only_unique_count": len(parent_only_blobs),
        "parent_only_source_family_counts": dict(parent_only_family_counts),
        "parent_only_source_child_rejected_count": len(source_errors),
        "parent_only_source_child_rejection_counts": {
            "REJECT_STRUCTURAL_LIMIT": len(source_errors)
        },
        "parent_only_source_rejection_outcome_commitment": _rejection_commitment(
            _PARENT_ONLY_SOURCE_REJECTION_DOMAIN, parent_only_blobs
        ),
        "parent_only_formal_child_rejected_count": len(formal_errors),
        "parent_only_formal_child_rejection_counts": {
            "REJECT_STRUCTURAL_LIMIT": len(formal_errors)
        },
        "parent_only_formal_rejection_outcome_commitment": _rejection_commitment(
            _PARENT_ONLY_FORMAL_REJECTION_DOMAIN, parent_only_blobs
        ),
        "parent_only_depth": 4,
        "parent_only_node_count": 6,
        "parent_only_set_commitment": _set_commitment(
            _PARENT_ONLY_DEPTH4_SET_DOMAIN, parent_only_blobs
        ),
        "removed_binary_operator_ids": list(TOMBSTONED_BINARY_OPERATOR_IDS),
        "retained_difference_id": 1,
        "canonical_program_budget": 50_000,
        "first_out_of_budget_ordinal": None,
        "subset_status": SUBSET_STATUS,
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "loaded_hegel_modules": list(loaded),
        "target_or_split_modules_loaded": False,
    }


def replay_golden() -> dict[str, object]:
    """Replay and commit every outcome in the sealed 25-vector manifest."""

    _assert_target_free()
    counts: dict[str, int] = {}
    outcomes: dict[str, bytes] = {}
    for vector in STRICT_GOLDEN_VECTORS_V1:
        if vector.expected_disposition == ACCEPT_PARENT_IDENTITY:
            if vector.boundary == "SOURCE_JSON":
                source = json.loads(vector.input_wire.decode("utf-8"))
                parent = canonicalize_shrink5_source_ast(source)
                child = canonicalize_shrink6_source_ast(source)
            else:
                parent = decode_shrink5_canonical_ast(vector.input_wire)
                child = decode_shrink6_canonical_ast(vector.input_wire)
            if (
                parent.cbor_bytes != child.cbor_bytes
                or parent.hash_id != child.hash_id
                or program_mdl_length_q32(parent) != program_mdl_length_q32(child)
                or decode_shrink6_canonical_ast(child.cbor_bytes) != child
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
                    lambda source=source: canonicalize_shrink6_source_ast(source)
                )
            else:
                payload = vector.input_wire
                observed = _error_code(
                    lambda payload=payload: decode_shrink6_canonical_ast(payload)
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
        "schema_version": "hegel-strict-canonicalizer-shrink6-golden/1",
        "implementation": "python",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "maximum_ast_depth": MAXIMUM_AST_DEPTH,
        "maximum_ast_node_count": MAXIMUM_AST_NODE_COUNT,
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

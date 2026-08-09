"""Frozen non-formal bindings for the shrink-step-3 dual diagnostic.

These values are deliberately domain-separated from the formal M3 root DAG.
They make Python and Rust diagnostic archives byte-comparable, but they are
not child formal roots and have no state-transition authority.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Final


PROFILE_ID: Final = "hegel-m3-shrink3-dual-diagnostic-profile-v1"
CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
BINDING_PROFILE_ID: Final = "NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1"

CHILD_DSL_SPEC_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00"
    b"hegel-old-dsl-v1.3.0\x00hegel-freeze-p2b-p3-v1.3.0\x00shrink-step3"
)
OPERATOR_SEMANTICS_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00"
    b"hegel-old-dsl-v1.3.0\x00hegel-canonical-ast-v1\x00"
    b"hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00"
    b"binary-tombstones:0\x00binary-source-alias:4"
)
IDENTIFIER_REGISTRY_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00"
    b"hegel-old-dsl-v1.3.0\x00aggregate-active:0,1,5\x00"
    b"aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00"
    b"rational-tombstones:0,2,4,6\x00rational-reserved:7\x00"
    b"binary-source-active:1,2,3,4,5,6\x00"
    b"binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00"
    b"binary-tombstones:0\x00binary-reserved:7"
)
CANONICAL_AST_SCHEMA_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK3/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00"
    b"hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array"
)
CANONICAL_CBOR_PROFILE_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK3/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00"
    b"hegel-cbor-det-v1\x00"
    b"RFC8949-deterministic-no-map-text-float-tag-indefinite"
)

NON_FORMAL_SYNTHETIC_CHILD_BINDINGS: Final = tuple(
    sha256(value).digest()
    for value in (
        CHILD_DSL_SPEC_PREIMAGE,
        OPERATOR_SEMANTICS_PREIMAGE,
        IDENTIFIER_REGISTRY_PREIMAGE,
    )
)
CANONICAL_AST_SCHEMA_ROOT: Final = sha256(
    CANONICAL_AST_SCHEMA_PREIMAGE
).digest()
CANONICAL_CBOR_PROFILE_ROOT: Final = sha256(
    CANONICAL_CBOR_PROFILE_PREIMAGE
).digest()

PARENT_DIAGNOSTIC_RESULT_COMMIT: Final = (
    "d9334589343554841d9f9fd30456a7402bcc7d33"
)
PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS: Final = (
    "f94cf1fb27c6734f24d4510efba0ca3726132706"
)
PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID: Final = (
    "phase3_shrink2_dual_complete_enumeration_diagnostic_"
    "e118f3809b2f5eef0ebd1c97936da746472a4188e0cc3feecc3e01688922b966"
)
STRICT_QUALIFICATION_EVIDENCE_COMMIT: Final = (
    "d2c5427b2e3344ab46e28d97cade8db1592ba67c"
)
STRICT_QUALIFICATION_STATUS: Final = "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
RUST_IMAGE_DIGEST: Final = (
    "sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
PYTHON_IMAGE_DIGEST: Final = (
    "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)


def diagnostic_root_hex_v1() -> dict[str, str]:
    """Return the five frozen non-formal diagnostic roots."""

    child, operator, registry = NON_FORMAL_SYNTHETIC_CHILD_BINDINGS
    return {
        "child_dsl_spec_root": child.hex(),
        "operator_semantics_root": operator.hex(),
        "identifier_registry_root": registry.hex(),
        "canonical_ast_schema_root": CANONICAL_AST_SCHEMA_ROOT.hex(),
        "canonical_cbor_profile_root": CANONICAL_CBOR_PROFILE_ROOT.hex(),
    }


__all__ = [
    "BINDING_PROFILE_ID",
    "CANONICAL_AST_SCHEMA_PREIMAGE",
    "CANONICAL_AST_SCHEMA_ROOT",
    "CANONICAL_CBOR_PROFILE_PREIMAGE",
    "CANONICAL_CBOR_PROFILE_ROOT",
    "CHILD_DSL_SPEC_PREIMAGE",
    "CLAIM_LEVEL",
    "IDENTIFIER_REGISTRY_PREIMAGE",
    "NON_FORMAL_SYNTHETIC_CHILD_BINDINGS",
    "OPERATOR_SEMANTICS_PREIMAGE",
    "PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID",
    "PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS",
    "PARENT_DIAGNOSTIC_RESULT_COMMIT",
    "PROFILE_ID",
    "PYTHON_IMAGE_DIGEST",
    "RUST_IMAGE_DIGEST",
    "STRICT_QUALIFICATION_EVIDENCE_COMMIT",
    "STRICT_QUALIFICATION_STATUS",
    "diagnostic_root_hex_v1",
]

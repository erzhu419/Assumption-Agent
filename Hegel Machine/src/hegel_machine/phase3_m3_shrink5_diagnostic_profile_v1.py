"""Frozen non-formal bindings for shrink-step-5 engineering."""

from __future__ import annotations

from hashlib import sha256
from typing import Final


PROFILE_ID: Final = "hegel-m3-shrink5-dual-diagnostic-profile-v1"
CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
BINDING_PROFILE_ID: Final = "NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1"

CHILD_DSL_SPEC_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK5/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00"
    b"hegel-old-dsl-v1.5.0\x00hegel-freeze-p2b-p3-v1.5.0\x00shrink-step5\x00"
    b"maximum-total-node-count:6\x00maximum-top-level-clauses:2"
)
OPERATOR_SEMANTICS_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK5/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00"
    b"hegel-old-dsl-v1.5.0\x00hegel-canonical-ast-v1\x00"
    b"hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00"
    b"binary-tombstones:0\x00binary-source-alias:4\x00"
    b"maximum-total-node-count:6\x00maximum-top-level-clauses:2"
)
IDENTIFIER_REGISTRY_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK5/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00"
    b"hegel-old-dsl-v1.5.0\x00aggregate-active:0,1,5\x00"
    b"aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00"
    b"rational-tombstones:0,2,4,6\x00rational-reserved:7\x00"
    b"binary-source-active:1,2,3,4,5,6\x00"
    b"binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00"
    b"binary-tombstones:0\x00binary-reserved:7"
)
CANONICAL_AST_SCHEMA_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK5/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00"
    b"hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array\x00"
    b"maximum-total-node-count:6\x00maximum-top-level-clauses:2"
)
CANONICAL_CBOR_PROFILE_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK5/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00"
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
CANONICAL_AST_SCHEMA_ROOT: Final = sha256(CANONICAL_AST_SCHEMA_PREIMAGE).digest()
CANONICAL_CBOR_PROFILE_ROOT: Final = sha256(
    CANONICAL_CBOR_PROFILE_PREIMAGE
).digest()

PARENT_DIAGNOSTIC_RESULT_COMMIT: Final = (
    "1bbdae8f3131625621c0bc1cfdfe5d7da6035e13"
)
PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS: Final = (
    "103eb6ad2d8500024580193b895809784d894609"
)
PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID: Final = (
    "phase3_shrink4_dual_complete_enumeration_diagnostic_"
    "5693b38315689969a1a525b75bec2917f95af1aa54951267797a0319afc60521"
)
PARENT_DIAGNOSTIC_ARTIFACT_PATH: Final = (
    "Hegel Machine/artifacts/phase3_shrink4_dual_complete_enumeration_"
    "diagnostic_v1.json"
)
PARENT_DIAGNOSTIC_ARTIFACT_SHA256: Final = (
    "2d653f667d8d43e0e8e68c54d6f0a939aab57bf6ba3add9b334809ca17745058"
)
PARENT_DIAGNOSTIC_STATUS: Final = "DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS"
PARENT_DIAGNOSTIC_CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
STRICT_QUALIFICATION_SOURCE_COMMIT: Final = (
    "320b0a3458901090cb738023a4398220fb1d9277"
)
STRICT_QUALIFICATION_EVIDENCE_COMMIT: Final = (
    "01b66cd8effeab258797998f594b250188d823da"
)
STRICT_QUALIFICATION_STATUS: Final = (
    "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
)
STRICT_QUALIFICATION_ARTIFACT_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m3_runtime/"
    "phase3_shrink5_sealed_dual_strict_qualification_v1.json"
)
STRICT_QUALIFICATION_ARTIFACT_SHA256: Final = (
    "75761fc536d96d5d0bc91c5c0ba30dbc7c9ee21aac8d3f1dc5c96f6aca919b76"
)
STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH: Final = (
    "sha256:5ee04b21477fd9f09271272fd6ecbf876b885b7831b37a868343a93996a187db"
)
DUAL_COMPLETE_ENUMERATION_STATUS: Final = "NOT_RUN"


def diagnostic_root_hex_v1() -> dict[str, str]:
    child, operator, registry = NON_FORMAL_SYNTHETIC_CHILD_BINDINGS
    return {
        "child_dsl_spec_root": child.hex(),
        "operator_semantics_root": operator.hex(),
        "identifier_registry_root": registry.hex(),
        "canonical_ast_schema_root": CANONICAL_AST_SCHEMA_ROOT.hex(),
        "canonical_cbor_profile_root": CANONICAL_CBOR_PROFILE_ROOT.hex(),
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "diagnostic_root_hex_v1"
]

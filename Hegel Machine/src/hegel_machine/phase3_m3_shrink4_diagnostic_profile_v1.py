"""Frozen non-formal bindings for the shrink-step-4 dual diagnostic."""

from __future__ import annotations

from hashlib import sha256
from typing import Final


PROFILE_ID: Final = "hegel-m3-shrink4-dual-diagnostic-profile-v1"
CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
BINDING_PROFILE_ID: Final = "NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1"

CHILD_DSL_SPEC_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00"
    b"hegel-old-dsl-v1.4.0\x00hegel-freeze-p2b-p3-v1.4.0\x00shrink-step4\x00"
    b"maximum-top-level-clauses:2"
)
OPERATOR_SEMANTICS_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00"
    b"hegel-old-dsl-v1.4.0\x00hegel-canonical-ast-v1\x00"
    b"hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00"
    b"binary-tombstones:0\x00binary-source-alias:4\x00"
    b"maximum-top-level-clauses:2"
)
IDENTIFIER_REGISTRY_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00"
    b"hegel-old-dsl-v1.4.0\x00aggregate-active:0,1,5\x00"
    b"aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00"
    b"rational-tombstones:0,2,4,6\x00rational-reserved:7\x00"
    b"binary-source-active:1,2,3,4,5,6\x00"
    b"binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00"
    b"binary-tombstones:0\x00binary-reserved:7"
)
CANONICAL_AST_SCHEMA_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK4/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00"
    b"hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array\x00"
    b"maximum-top-level-clauses:2"
)
CANONICAL_CBOR_PROFILE_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK4/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00"
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
    "c286732c140bd9adcfd3eef2b1788b3eac0eb3e9"
)
PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS: Final = (
    "d17b03e14f3f3e8a63c924706086f17367fbc0d6"
)
PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID: Final = (
    "phase3_shrink3_dual_complete_enumeration_diagnostic_"
    "3030ad10f2cd4f767a8397597be1ab3ed6cac7cd71975d69f59cc5abec6a4f5a"
)
STRICT_QUALIFICATION_SOURCE_COMMIT: Final = (
    "cd2c32bd3a27004b40f4550229f33afd73647433"
)
STRICT_QUALIFICATION_EVIDENCE_COMMIT: Final = (
    "c78e19b44ca85645d20790d7aefe1d8137b4e2bb"
)
STRICT_QUALIFICATION_STATUS: Final = (
    "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
)
STRICT_QUALIFICATION_ARTIFACT_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m3_runtime/"
    "phase3_shrink4_sealed_dual_strict_qualification_v1.json"
)
STRICT_QUALIFICATION_ARTIFACT_SHA256: Final = (
    "41fdea5fd9b16ab436386ef7794412ffa46e17e68efc6b8448deed17c7f99aae"
)
STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH: Final = (
    "sha256:44b4e0c0a2b79f6afb67ace348c1b3726e0ba64058c97c4c61be0c111ef6acec"
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

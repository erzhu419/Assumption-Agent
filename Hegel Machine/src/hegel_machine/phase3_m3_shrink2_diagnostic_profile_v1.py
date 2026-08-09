"""Frozen non-formal bindings for the shrink-step-2 dual diagnostic.

These values are deliberately domain-separated from the formal M3 root DAG.
They make the Python and Rust diagnostic archives byte-comparable, but they
must never be promoted or copied into a formal child execution manifest.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Final


PROFILE_ID: Final = "hegel-m3-shrink2-dual-diagnostic-profile-v1"
CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
BINDING_PROFILE_ID: Final = "NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1"

CHILD_DSL_SPEC_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK2/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00"
    b"hegel-old-dsl-v1.2.0\x00hegel-freeze-p2b-p3-v1.2.0\x00shrink-step2"
)
OPERATOR_SEMANTICS_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK2/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00"
    b"hegel-old-dsl-v1.2.0\x00hegel-canonical-ast-v1\x00"
    b"hegel-mdl-prefix-v1.0.0"
)
IDENTIFIER_REGISTRY_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK2/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00"
    b"hegel-old-dsl-v1.2.0\x00aggregate-active:0,1,5\x00"
    b"aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00"
    b"rational-tombstones:0,2,4,6\x00rational-reserved:7"
)
CANONICAL_AST_SCHEMA_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK2/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00"
    b"hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array"
)
CANONICAL_CBOR_PROFILE_PREIMAGE: Final = (
    b"HEGEL/M3/SHRINK2/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00"
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

PARENT_TERMINAL_COMMIT: Final = (
    "db612e403bb46e6a295fed01e85649f8af0924b4"
)
PARENT_FORMAL_RUN_ID: Final = "e4af9f57c38fb298462ec628c4ed8a03"
PARENT_TERMINAL_STATUS: Final = "DSL_TOO_LARGE"
RUST_IMAGE_DIGEST: Final = (
    "sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
PYTHON_IMAGE_DIGEST: Final = (
    "sha256:e5300dc020a26a34a19337a57602955a2510e22abeb176edd6de6cd2cc927dd4"
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
    "PARENT_FORMAL_RUN_ID",
    "PARENT_TERMINAL_COMMIT",
    "PARENT_TERMINAL_STATUS",
    "PROFILE_ID",
    "PYTHON_IMAGE_DIGEST",
    "RUST_IMAGE_DIGEST",
    "diagnostic_root_hex_v1",
]

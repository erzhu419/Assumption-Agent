"""Offline static image/runtime-closure mechanics for Phase-2B Step 5.

This module closes one deliberately narrow construction gap.  It loads the
raw, out-of-band-pinned authority/profile/registry V2 graph, selects exactly
one of its five static service bindings, and replays a byte-level OCI graph
plus a frozen *static* runtime-closure policy and evidence object.

The replay hashes the supplied raw OCI index, manifest, config, ordered layer
chunks, and five support artifacts.  It also validates exact CPython 3.10.12 /
``cp310`` identity and ordered static standard-library, import, native-loader,
and shared-library manifests.  These inputs are caller-supplied offline bytes.
They are not proof that an OCI image exists in a registry, was pulled or
executed, or that a live process exhibited the declared closure.  In
particular, this module does not start Docker/systemd, inspect ``/proc``, issue
signatures, attest a UID/control domain/key, set R01, or change readiness.

The policy is intentionally a pre-profile object: its payload hash is what a
service binding pins, so its body must not contain the later profile, registry,
or service-binding content IDs.  The separately content-addressed evidence
object supplies that later lineage under the raw Stage-B graph.  This ordering
avoids a policy-SHA -> binding-ID -> profile-ID -> policy-payload cycle.

Production entry points accept canonical bytes and exact tuples of exact byte
chunks only.  They expose no decoded-object path and no injected verifier,
digest, reader, or signature callback.  Successful results are constructor-
disabled and retain a closure-private, mutation-detecting issuance lineage.
That is an ordinary Python caller/KAT boundary, not resistance to arbitrary
interpreter-memory, closure-cell, or loaded-code mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import posixpath
import re
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    AUTHORITY_PROFILE_PREFIX_V2,
    ROLE_KEY_PREFIX_V1,
    SERVICE_ORDER_V2,
    SIGNER_REGISTRY_PREFIX_V2,
    STAGE_B_TRUST_LOCK_PREFIX_V2,
    Phase2BStep5AuthorityErrorV2,
    canonical_step5_json_bytes_v2 as _authority_canonical_json_bytes_v2,
    decode_canonical_step5_json_v2 as _authority_decode_canonical_json_v2,
    load_step5_authority_profile_v2 as _authority_load_profile_v2,
    load_step5_signer_registry_v2 as _authority_load_registry_v2,
    load_step5_stage_b_trust_lock_v2 as _authority_load_lock_v2,
    require_locked_step5_authority_graph_v2 as _authority_require_graph_v2,
)


RUNTIME_CLOSURE_POLICY_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-runtime-closure-policy/1"
)
IMAGE_CLOSURE_EVIDENCE_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-image-closure-evidence/1"
)
STATIC_RUNTIME_MANIFEST_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-static-runtime-manifest/1"
)

RUNTIME_CLOSURE_POLICY_PREFIX_V1: Final = (
    "phase2b_step5_runtime_closure_policy_v1_"
)
IMAGE_CLOSURE_EVIDENCE_PREFIX_V1: Final = (
    "phase2b_step5_image_closure_evidence_v1_"
)

CPYTHON_IMPLEMENTATION_V1: Final = "CPython"
CPYTHON_VERSION_V1: Final = "3.10.12"
CPYTHON_LANGUAGE_VERSION_V1: Final = "3.10"
CPYTHON_ABI_TAG_V1: Final = "cp310"
CPYTHON_CACHE_TAG_V1: Final = "cpython-310"

SUPPORT_ARTIFACT_ROLES_V1: Final = (
    "source_archive",
    "dependency_lock",
    "sbom",
    "provenance",
    "python_executable",
)
SUPPORT_ARTIFACT_MEDIA_TYPES_V1: Final = MappingProxyType(
    {
        "source_archive": "application/vnd.hegel.source-archive",
        "dependency_lock": "text/plain",
        "sbom": "application/spdx+json",
        "provenance": "application/vnd.in-toto+json",
        "python_executable": "application/vnd.hegel.python-executable",
    }
)

STATIC_MANIFEST_KINDS_V1: Final = (
    "PYTHON_STDLIB",
    "IMPORT_CLOSURE",
    "NATIVE_LOADER",
    "SHARED_LIBRARIES",
)
PYTHON_ORIGIN_KINDS_V1: Final = ("FILE", "BUILT_IN", "FROZEN")
PYTHON_MODULE_CLASSIFICATIONS_V1: Final = (
    "STDLIB",
    "PROJECT",
    "THIRD_PARTY",
)

OCI_INDEX_MEDIA_TYPE_V1: Final = "application/vnd.oci.image.index.v1+json"
OCI_MANIFEST_MEDIA_TYPE_V1: Final = (
    "application/vnd.oci.image.manifest.v1+json"
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_RUNTIME_CLOSURE_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_RUNTIME_CLOSURE_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_RUNTIME_CLOSURE_V1_SCHEMA"
FAIL_TRUST_ANCHOR_V1: Final = "FAIL_STEP5_RUNTIME_CLOSURE_V1_TRUST_ANCHOR"
FAIL_BINDING_V1: Final = "FAIL_STEP5_RUNTIME_CLOSURE_V1_BINDING"
FAIL_DIGEST_V1: Final = "FAIL_STEP5_RUNTIME_CLOSURE_V1_DIGEST"
FAIL_OCI_GRAPH_V1: Final = "FAIL_STEP5_RUNTIME_CLOSURE_V1_OCI_GRAPH"
FAIL_CLAIM_BOUNDARY_V1: Final = (
    "FAIL_STEP5_RUNTIME_CLOSURE_V1_CLAIM_BOUNDARY"
)

_MAX_POLICY_OR_EVIDENCE_BYTES: Final = 8 * 1024 * 1024
_HARD_MAX_OCI_METADATA_BYTES: Final = 32 * 1024 * 1024
_HARD_MAX_BLOB_BYTES: Final = 8 * 1024 * 1024 * 1024
_HARD_MAX_TOTAL_LAYER_BYTES: Final = 64 * 1024 * 1024 * 1024
_HARD_MAX_CHUNK_BYTES: Final = 64 * 1024 * 1024
_HARD_MAX_LAYERS: Final = 128
_HARD_MAX_MANIFEST_ENTRIES: Final = 250_000
_HARD_MAX_PATH_UTF8_BYTES: Final = 16 * 1024
_HARD_MAX_MODULE_NAME_UTF8_BYTES: Final = 4096
_SAFE_INTEGER: Final = (1 << 53) - 1
_HARD_JSON_DEPTH: Final = 64
_HARD_JSON_NODES: Final = 250_000

_HARD_RESOURCE_BOUNDS_V1: Final = MappingProxyType(
    {
        "max_oci_metadata_bytes": _HARD_MAX_OCI_METADATA_BYTES,
        "max_blob_bytes": _HARD_MAX_BLOB_BYTES,
        "max_total_layer_bytes": _HARD_MAX_TOTAL_LAYER_BYTES,
        "max_layers": _HARD_MAX_LAYERS,
        "max_manifest_entries": _HARD_MAX_MANIFEST_ENTRIES,
        "max_path_utf8_bytes": _HARD_MAX_PATH_UTF8_BYTES,
        "max_module_name_utf8_bytes": _HARD_MAX_MODULE_NAME_UTF8_BYTES,
        "max_chunk_bytes": _HARD_MAX_CHUNK_BYTES,
    }
)

_HEX64_RE: Final = re.compile(r"[0-9a-f]{64}")
_OCI_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_MODULE_NAME_RE: Final = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)
_SONAME_RE: Final = re.compile(r"[^/\\\x00]{1,512}")
_ELF_BUILD_ID_RE: Final = re.compile(r"[0-9a-f]{8,128}")
_OCI_PLATFORM_OS_ARCH_RE: Final = re.compile(
    r"[a-z0-9][a-z0-9_.-]{0,63}"
)
_OCI_PLATFORM_VARIANT_RE: Final = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"
)

_POLICY_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/STEP5/RUNTIME_CLOSURE_POLICY/V1\x00"
)
_EVIDENCE_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/STEP5/IMAGE_CLOSURE_EVIDENCE/V1\x00"
)

_POLICY_ROOT_FIELDS_V1: Final = frozenset({"body", "id"})
_EVIDENCE_ROOT_FIELDS_V1: Final = frozenset({"body", "id"})
_RUNTIME_IDENTITY_FIELDS_V1: Final = frozenset(
    {
        "implementation",
        "implementation_version",
        "language_version",
        "abi_tag",
        "cache_tag",
        "executable_artifact_role",
    }
)
_SUPPORT_ARTIFACT_FIELDS_V1: Final = frozenset(
    {"ordinal", "role", "media_type", "sha256", "byte_size"}
)
_MANIFEST_COMMITMENT_FIELDS_V1: Final = frozenset(
    {
        "stdlib_manifest_sha256",
        "import_manifest_sha256",
        "native_loader_manifest_sha256",
        "shared_library_manifest_sha256",
    }
)
_BOUNDS_FIELDS_V1: Final = frozenset(
    {
        "max_oci_metadata_bytes",
        "max_blob_bytes",
        "max_total_layer_bytes",
        "max_layers",
        "max_manifest_entries",
        "max_path_utf8_bytes",
        "max_module_name_utf8_bytes",
        "max_chunk_bytes",
    }
)
_OCI_PLATFORM_FIELDS_V1: Final = frozenset(
    {"os", "architecture", "variant"}
)
_OCI_MEDIA_POLICY_FIELDS_V1: Final = frozenset(
    {
        "index_media_type",
        "manifest_media_type",
        "config_media_type",
        "allowed_layer_media_types_in_exact_order",
    }
)
_STATIC_REQUIREMENTS_FIELDS_V1: Final = frozenset(
    {
        "raw_index_descriptor_required",
        "raw_manifest_descriptors_required",
        "raw_config_bytes_required",
        "all_ordered_layer_bytes_required",
        "all_support_artifact_bytes_required",
        "python_identity_exact_required",
        "ordered_static_manifests_required",
        "duplicate_free_inventory_required",
        "live_observation_required",
    }
)
_PERMANENT_NONCLAIMS_FIELDS_V1: Final = frozenset(
    {
        "service_image_exists",
        "service_image_pulled",
        "service_image_executed",
        "live_runtime_closure_observed",
        "entry_exit_closure_equal",
        "live_uid_gid_verified",
        "live_control_domain_verified",
        "key_custody_verified",
        "runtime_closure_signature_verified",
        "remote_attestation_verified",
        "external_cas_verified",
        "authority_evidence",
        "readiness_evidence",
        "r01_satisfied",
        "execution_authorized",
    }
)
_POLICY_BODY_FIELDS_V1: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "canonicalization",
        "service_name",
        "service_type",
        "role",
        "control_domain",
        "uid",
        "gid",
        "role_key_id",
        "key_epoch",
        "oci_platform",
        "oci_media_policy",
        "runtime_identity",
        "bound_artifacts_in_exact_order",
        "static_manifest_commitments",
        "resource_bounds",
        "static_verification_requirements",
        "permanent_nonclaims",
    }
)
_STATIC_MANIFEST_FIELDS_V1: Final = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "inventory_scope",
        "entries_in_exact_order",
        "entry_count",
        "static_inventory_complete_declared",
        "live_observation",
    }
)
_PYTHON_ENTRY_FIELDS_V1: Final = frozenset(
    {
        "ordinal",
        "module_name",
        "classification",
        "origin_kind",
        "path",
        "sha256",
        "byte_size",
    }
)
_NATIVE_ENTRY_FIELDS_V1: Final = frozenset(
    {
        "ordinal",
        "path",
        "soname",
        "sha256",
        "byte_size",
        "elf_build_id",
    }
)
_OCI_LAYER_DESCRIPTOR_FIELDS_V1: Final = frozenset(
    {"ordinal", "media_type", "digest", "byte_size"}
)
_OCI_IMAGE_FIELDS_V1: Final = frozenset(
    {
        "index_media_type",
        "index_digest",
        "index_byte_size",
        "manifest_media_type",
        "manifest_digest",
        "manifest_byte_size",
        "config_media_type",
        "config_digest",
        "config_byte_size",
        "platform",
        "rootfs_diff_ids_in_exact_order",
        "ordered_layer_descriptors",
    }
)
_VERIFICATION_STATE_FIELDS_V1: Final = frozenset(
    {
        "raw_oci_index_digest_matched",
        "raw_oci_manifest_descriptor_matched",
        "raw_oci_config_descriptor_matched",
        "ordered_raw_layer_bytes_matched",
        "support_artifact_bytes_matched",
        "python_static_identity_matched",
        "ordered_static_manifests_matched",
        "offline_static_byte_graph_mechanics_only",
        "service_image_exists",
        "service_image_pulled",
        "service_image_executed",
        "live_runtime_closure_observed",
        "entry_exit_closure_equal",
        "live_uid_gid_verified",
        "live_control_domain_verified",
        "key_custody_verified",
        "runtime_closure_signature_verified",
        "remote_attestation_verified",
        "external_cas_verified",
        "authority_evidence",
        "readiness_evidence",
        "r01_satisfied",
        "execution_authorized",
    }
)
_EVIDENCE_BODY_FIELDS_V1: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "canonicalization",
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
        "service_name",
        "service_binding_id",
        "closure_policy_id",
        "closure_policy_payload_sha256",
        "oci_image",
        "runtime_identity",
        "bound_artifacts_in_exact_order",
        "stdlib_manifest",
        "import_manifest",
        "native_loader_manifest",
        "shared_library_manifest",
        "verification_state",
    }
)

# Public field manifests are immutable aliases used to audit exact schemas.
RUNTIME_CLOSURE_POLICY_BODY_FIELDS_V1: Final = _POLICY_BODY_FIELDS_V1
IMAGE_CLOSURE_EVIDENCE_BODY_FIELDS_V1: Final = _EVIDENCE_BODY_FIELDS_V1
STATIC_RUNTIME_MANIFEST_FIELDS_V1: Final = _STATIC_MANIFEST_FIELDS_V1

_EXACT_RUNTIME_IDENTITY_V1: Final = MappingProxyType(
    {
        "implementation": CPYTHON_IMPLEMENTATION_V1,
        "implementation_version": CPYTHON_VERSION_V1,
        "language_version": CPYTHON_LANGUAGE_VERSION_V1,
        "abi_tag": CPYTHON_ABI_TAG_V1,
        "cache_tag": CPYTHON_CACHE_TAG_V1,
        "executable_artifact_role": "python_executable",
    }
)
_EXACT_STATIC_REQUIREMENTS_V1: Final = MappingProxyType(
    {
        "raw_index_descriptor_required": True,
        "raw_manifest_descriptors_required": True,
        "raw_config_bytes_required": True,
        "all_ordered_layer_bytes_required": True,
        "all_support_artifact_bytes_required": True,
        "python_identity_exact_required": True,
        "ordered_static_manifests_required": True,
        "duplicate_free_inventory_required": True,
        "live_observation_required": False,
    }
)
_EXACT_PERMANENT_NONCLAIMS_V1: Final = MappingProxyType(
    {field: False for field in _PERMANENT_NONCLAIMS_FIELDS_V1}
)
_EXACT_VERIFICATION_STATE_V1: Final = MappingProxyType(
    {
        "raw_oci_index_digest_matched": True,
        "raw_oci_manifest_descriptor_matched": True,
        "raw_oci_config_descriptor_matched": True,
        "ordered_raw_layer_bytes_matched": True,
        "support_artifact_bytes_matched": True,
        "python_static_identity_matched": True,
        "ordered_static_manifests_matched": True,
        "offline_static_byte_graph_mechanics_only": True,
        **{field: False for field in _PERMANENT_NONCLAIMS_FIELDS_V1},
    }
)


class Phase2BStep5RuntimeClosureErrorV1(ValueError):
    """Stable fail-closed error for offline closure mechanics V1."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5RuntimeClosureErrorV1(code, detail)


def _object_exact(
    value: object,
    fields: frozenset[str],
    name: str,
    _fail_exact=_fail,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(fields):
        _fail_exact(FAIL_SCHEMA_V1, f"{name} exact field set differs")
    return value


def _text(
    value: object,
    name: str,
    *,
    maximum_utf8_bytes: int = 4096,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or not value:
        _fail_exact(FAIL_SCHEMA_V1, f"{name} must be exact nonempty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise Phase2BStep5RuntimeClosureErrorV1(
            FAIL_SCHEMA_V1, f"{name} Unicode differs"
        ) from exc
    if len(encoded) > maximum_utf8_bytes:
        _fail_exact(FAIL_SCHEMA_V1, f"{name} exceeds UTF-8 bound")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
    _fail_exact=_fail,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail_exact(FAIL_SCHEMA_V1, f"{name} exact integer bound differs")
    return value


def _boolean(
    value: object,
    expected: bool,
    name: str,
    _fail_exact=_fail,
) -> bool:
    if type(value) is not bool or value is not expected:
        _fail_exact(FAIL_CLAIM_BOUNDARY_V1, f"{name} boolean differs")
    return value


def _sha256_hex(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
    _text_exact=_text,
    _fullmatch=_HEX64_RE.fullmatch,
    _fail_exact=_fail,
) -> str:
    text = _text_exact(value, name)
    if _fullmatch(text) is None or (
        not allow_zero and text == "0" * 64
    ):
        _fail_exact(FAIL_SCHEMA_V1, f"{name} SHA-256 syntax differs")
    return text


def _oci_digest(
    value: object,
    name: str,
    _text_exact=_text,
    _fullmatch=_OCI_DIGEST_RE.fullmatch,
    _fail_exact=_fail,
) -> str:
    text = _text_exact(value, name)
    if (
        _fullmatch(text) is None
        or text == "sha256:" + "0" * 64
    ):
        _fail_exact(FAIL_SCHEMA_V1, f"{name} OCI digest syntax differs")
    return text


def _content_identifier(
    value: object,
    prefix: str,
    name: str,
    _text_exact=_text,
    _hex_fullmatch=_HEX64_RE.fullmatch,
    _fail_exact=_fail,
) -> str:
    text = _text_exact(value, name)
    if not text.startswith(prefix) or _hex_fullmatch(text[len(prefix) :]) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{name} content-ID syntax differs")
    return text


def _equal_exact(left: object, right: object) -> bool:
    def equal(item: object, other: object) -> bool:
        if isinstance(item, Mapping) and isinstance(other, Mapping):
            return set(item) == set(other) and all(
                equal(item[key], other[key]) for key in item
            )
        if isinstance(item, (list, tuple)) and isinstance(other, (list, tuple)):
            return len(item) == len(other) and all(
                equal(child, counterpart)
                for child, counterpart in zip(item, other, strict=True)
            )
        return type(item) is type(other) and item == other

    return equal(left, right)


def _freeze(value: object, _proxy=MappingProxyType) -> object:
    def freeze(item: object) -> object:
        if type(item) is dict:
            return _proxy({key: freeze(child) for key, child in item.items()})
        if type(item) is list:
            return tuple(freeze(child) for child in item)
        return item

    return freeze(value)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedOfflineImageClosureMechanicsV1:
    """Privately issued result for exact offline static-byte replay only."""

    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    service_name: str
    service_binding_id: str
    closure_policy_id: str
    closure_policy_payload_sha256: str
    evidence_id: str
    oci_index_digest: str
    oci_manifest_digest: str
    oci_config_digest: str
    ordered_layer_digests: tuple[str, ...]
    support_artifact_digests: tuple[str, ...]
    static_manifest_digests: tuple[str, ...]
    offline_static_byte_graph_mechanics_verified: bool
    mechanics_only: bool
    service_image_exists: bool
    service_image_pulled: bool
    service_image_executed: bool
    live_runtime_closure_observed: bool
    entry_exit_closure_equal: bool
    live_uid_gid_verified: bool
    live_control_domain_verified: bool
    key_custody_verified: bool
    runtime_closure_signature_verified: bool
    remote_attestation_verified: bool
    external_cas_verified: bool
    authority_evidence: bool
    readiness_evidence: bool
    r01_satisfied: bool
    execution_authorized: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "VerifiedOfflineImageClosureMechanicsV1":
        raise TypeError("offline image-closure results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("offline image-closure results are privately issued")


def _build_private_result_lineage_v1():
    result_type = VerifiedOfflineImageClosureMechanicsV1
    fail_exact = _fail
    issued: dict[
        int, tuple[VerifiedOfflineImageClosureMechanicsV1, tuple[object, ...]]
    ] = {}
    fields = tuple(result_type.__dataclass_fields__)

    def require(value: object) -> VerifiedOfflineImageClosureMechanicsV1:
        if type(value) is not result_type:
            fail_exact(FAIL_INPUT_V1, "privately issued offline closure result required")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            fail_exact(FAIL_INPUT_V1, "offline closure result lacks private lineage")
        expected_values = record[1]
        try:
            intact = all(
                getattr(value, field) is expected
                for field, expected in zip(fields, expected_values, strict=True)
            )
        except (AttributeError, TypeError):
            intact = False
        if not intact:
            fail_exact(FAIL_INPUT_V1, "offline closure result lineage was mutated")
        for field in (
            "offline_static_byte_graph_mechanics_verified",
            "mechanics_only",
        ):
            if type(getattr(value, field)) is not bool or getattr(value, field) is not True:
                fail_exact(FAIL_CLAIM_BOUNDARY_V1, f"result {field} differs")
        for field in (
            "service_image_exists",
            "service_image_pulled",
            "service_image_executed",
            "live_runtime_closure_observed",
            "entry_exit_closure_equal",
            "live_uid_gid_verified",
            "live_control_domain_verified",
            "key_custody_verified",
            "runtime_closure_signature_verified",
            "remote_attestation_verified",
            "external_cas_verified",
            "authority_evidence",
            "readiness_evidence",
            "r01_satisfied",
            "execution_authorized",
        ):
            if type(getattr(value, field)) is not bool or getattr(value, field) is not False:
                fail_exact(FAIL_CLAIM_BOUNDARY_V1, f"result {field} differs")
        return value

    def issue(**values: object) -> VerifiedOfflineImageClosureMechanicsV1:
        if set(values) != set(fields):
            fail_exact(FAIL_INPUT_V1, "private result issuance field set differs")
        value = object.__new__(result_type)
        for field in fields:
            object.__setattr__(value, field, values[field])
        expected_values = tuple(getattr(value, field) for field in fields)
        issued[id(value)] = (value, expected_values)
        return require(value)

    return issue, require


(
    _hidden_issue_offline_closure_result_v1,
    _hidden_require_offline_closure_result_v1,
) = _build_private_result_lineage_v1()
del _build_private_result_lineage_v1


def _runtime_identity(
    value: object,
    _object=_object_exact,
    _equal=_equal_exact,
    _expected=_EXACT_RUNTIME_IDENTITY_V1,
    _fields=_RUNTIME_IDENTITY_FIELDS_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    identity = _object(value, _fields, "runtime identity")
    if not _equal(identity, _expected):
        _fail_exact(
            FAIL_BINDING_V1,
            "runtime identity must be exact CPython 3.10.12/cp310",
        )
    return identity


def _absolute_normalized_path(
    value: object,
    name: str,
    *,
    maximum_utf8_bytes: int,
    _text_exact=_text,
    _fail_exact=_fail,
) -> str:
    path = _text_exact(
        value, name, maximum_utf8_bytes=maximum_utf8_bytes
    )
    if (
        not path.startswith("/")
        or path == "/"
        or "\\" in path
        or "\x00" in path
        or "//" in path
        or posixpath.normpath(path) != path
        or any(part in ("", ".", "..") for part in path.split("/")[1:])
    ):
        _fail_exact(FAIL_SCHEMA_V1, f"{name} must be normalized absolute POSIX path")
    return path


def _validate_support_artifacts(
    value: object,
    *,
    max_blob_bytes: int,
    source_binding: Mapping[str, object],
    _object=_object_exact,
    _integer_exact=_integer,
    _sha_exact=_sha256_hex,
    _roles=SUPPORT_ARTIFACT_ROLES_V1,
    _media_types=SUPPORT_ARTIFACT_MEDIA_TYPES_V1,
    _fields=_SUPPORT_ARTIFACT_FIELDS_V1,
    _fail_exact=_fail,
) -> list[dict[str, object]]:
    if type(value) is not list or len(value) != len(_roles):
        _fail_exact(
            FAIL_SCHEMA_V1,
            "bound support artifacts require the exact five-role order",
        )
    rows: list[dict[str, object]] = []
    for ordinal, (expected_role, raw_row) in enumerate(
        zip(_roles, value, strict=True)
    ):
        row = _object(
            raw_row,
            _fields,
            f"{expected_role} support artifact",
        )
        _integer_exact(
            row["ordinal"],
            f"{expected_role} ordinal",
            minimum=ordinal,
            maximum=ordinal,
        )
        if (
            row["role"] != expected_role
            or row["media_type"] != _media_types[expected_role]
        ):
            _fail_exact(
                FAIL_SCHEMA_V1, f"{expected_role} support artifact role/media differs"
            )
        digest = _sha_exact(row["sha256"], f"{expected_role} SHA-256")
        _integer_exact(
            row["byte_size"],
            f"{expected_role} byte size",
            minimum=1,
            maximum=max_blob_bytes,
        )
        if expected_role == "source_archive" and digest != source_binding[
            "source_archive_sha256"
        ]:
            _fail_exact(
                FAIL_BINDING_V1,
                "source archive digest differs from authority profile source binding",
            )
        if expected_role == "dependency_lock" and digest != source_binding[
            "dependency_lock_sha256"
        ]:
            _fail_exact(
                FAIL_BINDING_V1,
                "dependency lock digest differs from authority profile source binding",
            )
        rows.append(row)
    digests = [row["sha256"] for row in rows]
    if len(set(digests)) != len(digests):
        _fail_exact(FAIL_SCHEMA_V1, "support artifact digest roles alias")
    return rows


def _validate_resource_bounds(
    value: object,
    _object=_object_exact,
    _integer_exact=_integer,
    _fields=_BOUNDS_FIELDS_V1,
    _hard_maxima=_HARD_RESOURCE_BOUNDS_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    bounds = _object(value, _fields, "resource bounds")
    for field, hard_maximum in _hard_maxima.items():
        _integer_exact(
            bounds[field], field, minimum=1, maximum=hard_maximum
        )
    if bounds["max_chunk_bytes"] > bounds["max_blob_bytes"]:
        _fail_exact(FAIL_SCHEMA_V1, "chunk bound exceeds blob bound")
    if bounds["max_oci_metadata_bytes"] > bounds["max_blob_bytes"]:
        _fail_exact(FAIL_SCHEMA_V1, "OCI metadata bound exceeds blob bound")
    if bounds["max_blob_bytes"] > bounds["max_total_layer_bytes"]:
        _fail_exact(FAIL_SCHEMA_V1, "blob bound exceeds total-layer bound")
    return bounds


def _validate_oci_platform(
    value: object,
    _object=_object_exact,
    _text_exact=_text,
    _fields=_OCI_PLATFORM_FIELDS_V1,
    _os_arch_fullmatch=_OCI_PLATFORM_OS_ARCH_RE.fullmatch,
    _variant_fullmatch=_OCI_PLATFORM_VARIANT_RE.fullmatch,
    _fail_exact=_fail,
) -> dict[str, object]:
    platform = _object(value, _fields, "OCI platform")
    for field in ("os", "architecture"):
        text = _text_exact(platform[field], f"OCI platform {field}", maximum_utf8_bytes=128)
        if not text.isascii() or _os_arch_fullmatch(text) is None:
            _fail_exact(FAIL_SCHEMA_V1, f"OCI platform {field} syntax differs")
    variant = platform["variant"]
    if variant is not None:
        text = _text_exact(variant, "OCI platform variant", maximum_utf8_bytes=128)
        if not text.isascii() or _variant_fullmatch(text) is None:
            _fail_exact(FAIL_SCHEMA_V1, "OCI platform variant syntax differs")
    return platform


def _validate_oci_media_policy(
    value: object,
    _object=_object_exact,
    _text_exact=_text,
    _fields=_OCI_MEDIA_POLICY_FIELDS_V1,
    _index_media_type=OCI_INDEX_MEDIA_TYPE_V1,
    _manifest_media_type=OCI_MANIFEST_MEDIA_TYPE_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    policy = _object(value, _fields, "OCI media policy")
    if (
        policy["index_media_type"] != _index_media_type
        or policy["manifest_media_type"] != _manifest_media_type
    ):
        _fail_exact(FAIL_SCHEMA_V1, "OCI index/manifest media type differs")
    config_type = _text_exact(
        policy["config_media_type"], "OCI config media type", maximum_utf8_bytes=256
    )
    if not config_type.startswith("application/vnd.oci.image.config.v1+"):
        _fail_exact(FAIL_SCHEMA_V1, "OCI config media type differs")
    layer_types = policy["allowed_layer_media_types_in_exact_order"]
    if type(layer_types) is not list or not 1 <= len(layer_types) <= 16:
        _fail_exact(FAIL_SCHEMA_V1, "allowed OCI layer media types differ")
    checked: list[str] = []
    for index, item in enumerate(layer_types):
        media_type = _text_exact(
            item, f"OCI layer media type {index}", maximum_utf8_bytes=256
        )
        if not media_type.startswith("application/vnd.oci.image.layer.v1"):
            _fail_exact(FAIL_SCHEMA_V1, "OCI layer media type namespace differs")
        checked.append(media_type)
    if len(set(checked)) != len(checked):
        _fail_exact(FAIL_SCHEMA_V1, "allowed OCI layer media types repeat")
    return policy


def _validate_manifest_commitments(
    value: object,
    _object=_object_exact,
    _sha_exact=_sha256_hex,
    _fields=_MANIFEST_COMMITMENT_FIELDS_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    commitments = _object(
        value, _fields, "static manifest commitments"
    )
    for field in _fields:
        _sha_exact(commitments[field], field)
    if len(set(commitments.values())) != len(commitments):
        _fail_exact(FAIL_SCHEMA_V1, "static manifest commitment digests alias")
    return commitments


def _runtime_closure_policy_id(
    body: object,
    _canonical=_authority_canonical_json_bytes_v2,
    _sha256=hashlib.sha256,
    _domain=_POLICY_DOMAIN_V1,
    _prefix=RUNTIME_CLOSURE_POLICY_PREFIX_V1,
) -> str:
    return _prefix + _sha256(_domain + _canonical(body)).hexdigest()


def _image_closure_evidence_id(
    body: object,
    _canonical=_authority_canonical_json_bytes_v2,
    _sha256=hashlib.sha256,
    _domain=_EVIDENCE_DOMAIN_V1,
    _prefix=IMAGE_CLOSURE_EVIDENCE_PREFIX_V1,
) -> str:
    return _prefix + _sha256(_domain + _canonical(body)).hexdigest()


def _validate_runtime_closure_policy(
    value: object,
    *,
    profile: object,
    service_binding: Mapping[str, object],
    _object=_object_exact,
    _content_identifier_exact=_content_identifier,
    _runtime_identity_exact=_runtime_identity,
    _validate_bounds=_validate_resource_bounds,
    _validate_platform=_validate_oci_platform,
    _validate_media=_validate_oci_media_policy,
    _validate_support=_validate_support_artifacts,
    _validate_commitments=_validate_manifest_commitments,
    _equal=_equal_exact,
    _derive_id=_runtime_closure_policy_id,
    _root_fields=_POLICY_ROOT_FIELDS_V1,
    _body_fields=_POLICY_BODY_FIELDS_V1,
    _schema=RUNTIME_CLOSURE_POLICY_SCHEMA_V1,
    _role_key_prefix=ROLE_KEY_PREFIX_V1,
    _requirements_fields=_STATIC_REQUIREMENTS_FIELDS_V1,
    _exact_requirements=_EXACT_STATIC_REQUIREMENTS_V1,
    _nonclaims_fields=_PERMANENT_NONCLAIMS_FIELDS_V1,
    _exact_nonclaims=_EXACT_PERMANENT_NONCLAIMS_V1,
    _fail_exact=_fail,
) -> tuple[dict[str, object], dict[str, object]]:
    root = _object(value, _root_fields, "runtime-closure policy")
    body = _object(root["body"], _body_fields, "runtime-closure policy body")
    if (
        body["schema_version"] != _schema
        or body["status"] != "STATIC_POLICY_CANDIDATE_NOT_OBSERVED_LIVE"
        or body["claim_level"]
        != (
            "OFFLINE_STATIC_BYTE_POLICY_ONLY_NOT_IMAGE_EXISTENCE_NOT_LIVE_"
            "RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS"
        )
        or body["canonicalization"] != "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
    ):
        _fail_exact(FAIL_SCHEMA_V1, "runtime-closure policy header differs")
    _content_identifier_exact(
        body["role_key_id"], _role_key_prefix, "policy role-key ID"
    )
    for field in (
        "service_name",
        "service_type",
        "role",
        "control_domain",
        "uid",
        "gid",
        "role_key_id",
        "key_epoch",
    ):
        if not _equal(body[field], service_binding[field]):
            _fail_exact(FAIL_BINDING_V1, f"policy {field} differs from service binding")
    platform = _validate_platform(body["oci_platform"])
    _validate_media(body["oci_media_policy"])
    _runtime_identity_exact(body["runtime_identity"])
    bounds = _validate_bounds(body["resource_bounds"])
    _validate_support(
        body["bound_artifacts_in_exact_order"],
        max_blob_bytes=bounds["max_blob_bytes"],
        source_binding=profile.body["source_binding"],
    )
    _validate_commitments(body["static_manifest_commitments"])
    requirements = _object(
        body["static_verification_requirements"],
        _requirements_fields,
        "static verification requirements",
    )
    if not _equal(requirements, _exact_requirements):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V1, "static verification requirements differ")
    nonclaims = _object(
        body["permanent_nonclaims"],
        _nonclaims_fields,
        "runtime-closure permanent nonclaims",
    )
    if not _equal(nonclaims, _exact_nonclaims):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V1, "runtime-closure nonclaims differ")
    expected_id = _derive_id(body)
    if root["id"] != expected_id:
        _fail_exact(FAIL_BINDING_V1, "runtime-closure policy ID differs")
    return body, platform


def _validate_python_manifest_entry(
    value: object,
    *,
    ordinal: int,
    manifest_kind: str,
    max_path_utf8_bytes: int,
    max_module_name_utf8_bytes: int,
    max_blob_bytes: int,
    _object=_object_exact,
    _integer_exact=_integer,
    _text_exact=_text,
    _path_exact=_absolute_normalized_path,
    _sha_exact=_sha256_hex,
    _fields=_PYTHON_ENTRY_FIELDS_V1,
    _module_name_fullmatch=_MODULE_NAME_RE.fullmatch,
    _classifications=PYTHON_MODULE_CLASSIFICATIONS_V1,
    _origin_kinds=PYTHON_ORIGIN_KINDS_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    row = _object(value, _fields, f"{manifest_kind} entry {ordinal}")
    _integer_exact(
        row["ordinal"], f"{manifest_kind} ordinal", minimum=ordinal, maximum=ordinal
    )
    module_name = _text_exact(
        row["module_name"],
        f"{manifest_kind} module name",
        maximum_utf8_bytes=max_module_name_utf8_bytes,
    )
    if _module_name_fullmatch(module_name) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{manifest_kind} module-name syntax differs")
    classification = row["classification"]
    if classification not in _classifications or type(classification) is not str:
        _fail_exact(FAIL_SCHEMA_V1, f"{manifest_kind} classification differs")
    if manifest_kind == "PYTHON_STDLIB" and classification != "STDLIB":
        _fail_exact(FAIL_SCHEMA_V1, "stdlib manifest contains non-stdlib row")
    origin = row["origin_kind"]
    if origin not in _origin_kinds or type(origin) is not str:
        _fail_exact(FAIL_SCHEMA_V1, f"{manifest_kind} origin kind differs")
    if origin == "FILE":
        _path_exact(
            row["path"],
            f"{manifest_kind} module path",
            maximum_utf8_bytes=max_path_utf8_bytes,
        )
        _sha_exact(row["sha256"], f"{manifest_kind} module SHA-256")
        _integer_exact(
            row["byte_size"],
            f"{manifest_kind} module byte size",
            minimum=1,
            maximum=max_blob_bytes,
        )
    elif (
        row["path"] is not None
        or row["sha256"] is not None
        or type(row["byte_size"]) is not int
        or row["byte_size"] != 0
    ):
        _fail_exact(
            FAIL_SCHEMA_V1,
            f"{manifest_kind} built-in/frozen row fabricates file evidence",
        )
    return row


def _validate_native_manifest_entry(
    value: object,
    *,
    ordinal: int,
    manifest_kind: str,
    max_path_utf8_bytes: int,
    max_blob_bytes: int,
    _object=_object_exact,
    _integer_exact=_integer,
    _path_exact=_absolute_normalized_path,
    _text_exact=_text,
    _sha_exact=_sha256_hex,
    _fields=_NATIVE_ENTRY_FIELDS_V1,
    _soname_fullmatch=_SONAME_RE.fullmatch,
    _build_id_fullmatch=_ELF_BUILD_ID_RE.fullmatch,
    _fail_exact=_fail,
) -> dict[str, object]:
    row = _object(value, _fields, f"{manifest_kind} entry {ordinal}")
    _integer_exact(
        row["ordinal"], f"{manifest_kind} ordinal", minimum=ordinal, maximum=ordinal
    )
    _path_exact(
        row["path"],
        f"{manifest_kind} path",
        maximum_utf8_bytes=max_path_utf8_bytes,
    )
    soname = _text_exact(
        row["soname"], f"{manifest_kind} soname", maximum_utf8_bytes=512
    )
    if _soname_fullmatch(soname) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{manifest_kind} soname syntax differs")
    _sha_exact(row["sha256"], f"{manifest_kind} SHA-256")
    _integer_exact(
        row["byte_size"],
        f"{manifest_kind} byte size",
        minimum=1,
        maximum=max_blob_bytes,
    )
    build_id = _text_exact(
        row["elf_build_id"],
        f"{manifest_kind} ELF build ID",
        maximum_utf8_bytes=128,
    )
    if _build_id_fullmatch(build_id) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{manifest_kind} ELF build ID differs")
    return row


def _validate_static_manifest(
    value: object,
    *,
    expected_kind: str,
    bounds: Mapping[str, object],
    _object=_object_exact,
    _integer_exact=_integer,
    _python_entry=_validate_python_manifest_entry,
    _native_entry=_validate_native_manifest_entry,
    _manifest_fields=_STATIC_MANIFEST_FIELDS_V1,
    _manifest_schema=STATIC_RUNTIME_MANIFEST_SCHEMA_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    manifest = _object(value, _manifest_fields, f"{expected_kind} manifest")
    if (
        manifest["schema_version"] != _manifest_schema
        or manifest["manifest_kind"] != expected_kind
        or manifest["inventory_scope"]
        != "STATIC_IMAGE_DECLARATION_NOT_LIVE_OBSERVATION"
        or type(manifest["static_inventory_complete_declared"]) is not bool
        or manifest["static_inventory_complete_declared"] is not True
        or type(manifest["live_observation"]) is not bool
        or manifest["live_observation"] is not False
    ):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V1, f"{expected_kind} manifest header differs")
    entries = manifest["entries_in_exact_order"]
    if type(entries) is not list:
        _fail_exact(FAIL_SCHEMA_V1, f"{expected_kind} entries must be exact list")
    if expected_kind == "NATIVE_LOADER":
        minimum_entries = maximum_entries = 1
    elif expected_kind in ("PYTHON_STDLIB", "SHARED_LIBRARIES"):
        minimum_entries = 1
        maximum_entries = bounds["max_manifest_entries"]
    else:
        minimum_entries = 0
        maximum_entries = bounds["max_manifest_entries"]
    if not minimum_entries <= len(entries) <= maximum_entries:
        _fail_exact(FAIL_SCHEMA_V1, f"{expected_kind} cardinality differs")
    _integer_exact(
        manifest["entry_count"],
        f"{expected_kind} entry count",
        minimum=len(entries),
        maximum=len(entries),
    )
    checked: list[dict[str, object]] = []
    for ordinal, entry in enumerate(entries):
        if expected_kind in ("PYTHON_STDLIB", "IMPORT_CLOSURE"):
            checked.append(
                _python_entry(
                    entry,
                    ordinal=ordinal,
                    manifest_kind=expected_kind,
                    max_path_utf8_bytes=bounds["max_path_utf8_bytes"],
                    max_module_name_utf8_bytes=bounds[
                        "max_module_name_utf8_bytes"
                    ],
                    max_blob_bytes=bounds["max_blob_bytes"],
                )
            )
        else:
            checked.append(
                _native_entry(
                    entry,
                    ordinal=ordinal,
                    manifest_kind=expected_kind,
                    max_path_utf8_bytes=bounds["max_path_utf8_bytes"],
                    max_blob_bytes=bounds["max_blob_bytes"],
                )
            )
    if expected_kind in ("PYTHON_STDLIB", "IMPORT_CLOSURE"):
        module_names = [row["module_name"] for row in checked]
        if len(set(module_names)) != len(module_names):
            _fail_exact(FAIL_SCHEMA_V1, f"{expected_kind} repeats module identity")
        file_paths = [row["path"] for row in checked if row["origin_kind"] == "FILE"]
        if len(set(file_paths)) != len(file_paths):
            _fail_exact(FAIL_SCHEMA_V1, f"{expected_kind} repeats file path")
    else:
        for field in ("path", "soname"):
            values = [row[field] for row in checked]
            if len(set(values)) != len(values):
                _fail_exact(FAIL_SCHEMA_V1, f"{expected_kind} repeats {field}")
    return manifest


def _static_manifest_sha256(
    manifest: object,
    _canonical=_authority_canonical_json_bytes_v2,
    _sha256=hashlib.sha256,
) -> str:
    return _sha256(_canonical(manifest)).hexdigest()


def _json_pairs_no_duplicates(
    pairs: list[tuple[str, object]],
    _fail_exact=_fail,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail_exact(FAIL_OCI_GRAPH_V1, "OCI JSON has duplicate/non-string key")
        result[key] = value
    return result


def _reject_oci_float(value: str, _fail_exact=_fail) -> NoReturn:
    _fail_exact(FAIL_OCI_GRAPH_V1, f"OCI JSON floating number is forbidden: {value}")


def _reject_oci_constant(value: str, _fail_exact=_fail) -> NoReturn:
    _fail_exact(FAIL_OCI_GRAPH_V1, f"OCI JSON nonfinite constant is forbidden: {value}")


def _validate_plain_json_tree(
    value: object,
    *,
    depth: int = 0,
    counter: list[int] | None = None,
    _safe_integer=_SAFE_INTEGER,
    _max_depth=_HARD_JSON_DEPTH,
    _max_nodes=_HARD_JSON_NODES,
    _fail_exact=_fail,
) -> object:
    active = [0] if counter is None else counter

    def visit(item: object, item_depth: int) -> object:
        active[0] += 1
        if item_depth > _max_depth or active[0] > _max_nodes:
            _fail_exact(FAIL_OCI_GRAPH_V1, "OCI JSON structure exceeds bounds")
        if item is None or type(item) is bool or type(item) is str:
            return item
        if type(item) is int:
            if not -_safe_integer <= item <= _safe_integer:
                _fail_exact(FAIL_OCI_GRAPH_V1, "OCI JSON integer exceeds safe range")
            return item
        if type(item) is list:
            for child in item:
                visit(child, item_depth + 1)
            return item
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    _fail_exact(FAIL_OCI_GRAPH_V1, "OCI JSON object key differs")
                visit(child, item_depth + 1)
            return item
        _fail_exact(FAIL_OCI_GRAPH_V1, "OCI JSON value outside exact data model")

    return visit(value, depth)


def _decode_oci_json(
    payload: bytes,
    *,
    maximum_bytes: int,
    _loads=json.loads,
    _pairs=_json_pairs_no_duplicates,
    _reject_float=_reject_oci_float,
    _reject_constant=_reject_oci_constant,
    _validate_tree=_validate_plain_json_tree,
    _hard_maximum_bytes=_HARD_MAX_OCI_METADATA_BYTES,
    _json_decode_error=json.JSONDecodeError,
    _fail_exact=_fail,
) -> object:
    if (
        type(payload) is not bytes
        or type(maximum_bytes) is not int
        or not 1 <= maximum_bytes <= _hard_maximum_bytes
        or not 1 <= len(payload) <= maximum_bytes
    ):
        _fail_exact(FAIL_INPUT_V1, "OCI metadata payload/bound differs")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = _loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, _json_decode_error, RecursionError) as exc:
        raise Phase2BStep5RuntimeClosureErrorV1(
            FAIL_OCI_GRAPH_V1, "OCI JSON decoding failed"
        ) from exc
    return _validate_tree(value)


def _hash_exact_bytes(
    payload: bytes,
    *,
    maximum_bytes: int,
    name: str,
    _sha256=hashlib.sha256,
    _fail_exact=_fail,
) -> tuple[str, int]:
    if (
        type(payload) is not bytes
        or type(maximum_bytes) is not int
        or not 1 <= len(payload) <= maximum_bytes
    ):
        _fail_exact(FAIL_INPUT_V1, f"{name} raw bytes/bound differs")
    return _sha256(payload).hexdigest(), len(payload)


def _hash_chunk_tuple(
    chunks: object,
    *,
    maximum_blob_bytes: int,
    maximum_chunk_bytes: int,
    name: str,
    _sha256=hashlib.sha256,
    _fail_exact=_fail,
) -> tuple[str, int]:
    if type(chunks) is not tuple or not chunks:
        _fail_exact(FAIL_INPUT_V1, f"{name} must be exact nonempty tuple of chunks")
    digest = _sha256()
    total = 0
    for index, chunk in enumerate(chunks):
        if type(chunk) is not bytes or not 1 <= len(chunk) <= maximum_chunk_bytes:
            _fail_exact(FAIL_INPUT_V1, f"{name} chunk {index} differs")
        total += len(chunk)
        if total > maximum_blob_bytes:
            _fail_exact(FAIL_INPUT_V1, f"{name} exceeds frozen blob bound")
        digest.update(chunk)
    return digest.hexdigest(), total


def _preflight_chunk_matrix_shape(
    value: object,
    *,
    name: str,
    minimum_outer_items: int,
    maximum_outer_items: int,
    _hard_maximum_chunk_bytes=_HARD_MAX_CHUNK_BYTES,
    _fail_exact=_fail,
) -> tuple[tuple[bytes, ...], ...]:
    """Validate the complete nested transport shape before any blob is hashed."""

    if (
        type(value) is not tuple
        or type(minimum_outer_items) is not int
        or type(maximum_outer_items) is not int
        or not 0 <= minimum_outer_items <= len(value) <= maximum_outer_items
    ):
        _fail_exact(FAIL_INPUT_V1, f"{name} outer tuple/cardinality differs")
    for outer_index, chunks in enumerate(value):
        if type(chunks) is not tuple or not chunks:
            _fail_exact(FAIL_INPUT_V1, f"{name} item {outer_index} is not exact chunk tuple")
        for chunk_index, chunk in enumerate(chunks):
            if (
                type(chunk) is not bytes
                or not 1 <= len(chunk) <= _hard_maximum_chunk_bytes
            ):
                _fail_exact(
                    FAIL_INPUT_V1,
                    f"{name} item {outer_index} chunk {chunk_index} differs",
                )
    return value


_OCI_DESCRIPTOR_FIELDS_V1: Final = frozenset(
    {"mediaType", "digest", "size"}
)
_OCI_INDEX_DESCRIPTOR_FIELDS_V1: Final = frozenset(
    {"mediaType", "digest", "size", "platform"}
)


def _validate_oci_descriptor(
    value: object,
    *,
    name: str,
    expected_media_type: str,
    expected_digest: str,
    expected_size: int,
    include_platform: bool = False,
    expected_platform: Mapping[str, object] | None = None,
    _object=_object_exact,
    _oci_digest_exact=_oci_digest,
    _integer_exact=_integer,
    _equal=_equal_exact,
    _descriptor_fields=_OCI_DESCRIPTOR_FIELDS_V1,
    _index_descriptor_fields=_OCI_INDEX_DESCRIPTOR_FIELDS_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    fields = (
        _index_descriptor_fields
        if include_platform
        else _descriptor_fields
    )
    descriptor = _object(value, fields, name)
    if descriptor["mediaType"] != expected_media_type:
        _fail_exact(FAIL_OCI_GRAPH_V1, f"{name} media type differs")
    _oci_digest_exact(descriptor["digest"], f"{name} digest")
    _integer_exact(
        descriptor["size"],
        f"{name} size",
        minimum=expected_size,
        maximum=expected_size,
    )
    if descriptor["digest"] != expected_digest:
        _fail_exact(FAIL_OCI_GRAPH_V1, f"{name} raw digest differs")
    if include_platform:
        if expected_platform is None:
            _fail_exact(FAIL_INPUT_V1, f"{name} missing expected platform")
        expected_platform_fields = {"os", "architecture"}
        if expected_platform["variant"] is not None:
            expected_platform_fields.add("variant")
        parsed_platform = _object(
            descriptor["platform"],
            frozenset(expected_platform_fields),
            f"{name} platform",
        )
        expected_raw_platform = {
            field: expected_platform[field] for field in expected_platform_fields
        }
        if not _equal(parsed_platform, expected_raw_platform):
            _fail_exact(FAIL_OCI_GRAPH_V1, f"{name} platform differs")
    return descriptor


def _validate_evidence_oci_image(
    value: object,
    *,
    media_policy: Mapping[str, object],
    expected_platform: Mapping[str, object],
    maximum_layers: int,
    maximum_blob_bytes: int,
    _object=_object_exact,
    _validate_platform=_validate_oci_platform,
    _oci_digest_exact=_oci_digest,
    _integer_exact=_integer,
    _equal=_equal_exact,
    _image_fields=_OCI_IMAGE_FIELDS_V1,
    _layer_fields=_OCI_LAYER_DESCRIPTOR_FIELDS_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    image = _object(value, _image_fields, "evidence OCI image")
    if (
        image["index_media_type"] != media_policy["index_media_type"]
        or image["manifest_media_type"] != media_policy["manifest_media_type"]
        or image["config_media_type"] != media_policy["config_media_type"]
    ):
        _fail_exact(FAIL_SCHEMA_V1, "evidence OCI media policy differs")
    for field in ("index_digest", "manifest_digest", "config_digest"):
        _oci_digest_exact(image[field], f"evidence OCI {field}")
    for field in ("index_byte_size", "manifest_byte_size", "config_byte_size"):
        _integer_exact(
            image[field],
            f"evidence OCI {field}",
            minimum=1,
            maximum=maximum_blob_bytes,
        )
    if not _equal(_validate_platform(image["platform"]), expected_platform):
        _fail_exact(FAIL_BINDING_V1, "evidence OCI platform differs from policy")
    diff_ids = image["rootfs_diff_ids_in_exact_order"]
    layers = image["ordered_layer_descriptors"]
    if (
        type(diff_ids) is not list
        or type(layers) is not list
        or not 1 <= len(layers) <= maximum_layers
        or len(diff_ids) != len(layers)
    ):
        _fail_exact(FAIL_SCHEMA_V1, "evidence OCI layer/diff-ID cardinality differs")
    checked_diff_ids = [
        _oci_digest_exact(item, f"rootfs diff ID {index}")
        for index, item in enumerate(diff_ids)
    ]
    if len(set(checked_diff_ids)) != len(checked_diff_ids):
        _fail_exact(FAIL_SCHEMA_V1, "rootfs diff IDs repeat")
    allowed_media = media_policy["allowed_layer_media_types_in_exact_order"]
    checked_layers: list[dict[str, object]] = []
    for ordinal, value_row in enumerate(layers):
        row = _object(
            value_row,
            _layer_fields,
            f"evidence OCI layer {ordinal}",
        )
        _integer_exact(
            row["ordinal"], "evidence OCI layer ordinal", minimum=ordinal, maximum=ordinal
        )
        if type(row["media_type"]) is not str or row["media_type"] not in allowed_media:
            _fail_exact(FAIL_SCHEMA_V1, f"evidence OCI layer {ordinal} media differs")
        _oci_digest_exact(row["digest"], f"evidence OCI layer {ordinal} digest")
        _integer_exact(
            row["byte_size"],
            f"evidence OCI layer {ordinal} size",
            minimum=1,
            maximum=maximum_blob_bytes,
        )
        checked_layers.append(row)
    if len({row["digest"] for row in checked_layers}) != len(checked_layers):
        _fail_exact(FAIL_SCHEMA_V1, "evidence OCI layer digests repeat")
    return image


def _validate_image_closure_evidence(
    value: object,
    *,
    lock: object,
    profile: object,
    registry: object,
    service_binding: Mapping[str, object],
    policy_id: str,
    policy_payload_sha256: str,
    policy_body: Mapping[str, object],
    expected_platform: Mapping[str, object],
    _object=_object_exact,
    _content_identifier_exact=_content_identifier,
    _runtime_identity_exact=_runtime_identity,
    _validate_support=_validate_support_artifacts,
    _validate_manifest=_validate_static_manifest,
    _manifest_sha=_static_manifest_sha256,
    _validate_image=_validate_evidence_oci_image,
    _equal=_equal_exact,
    _derive_id=_image_closure_evidence_id,
    _body_fields=_EVIDENCE_BODY_FIELDS_V1,
    _state_fields=_VERIFICATION_STATE_FIELDS_V1,
    _exact_state=_EXACT_VERIFICATION_STATE_V1,
    _manifest_kinds=STATIC_MANIFEST_KINDS_V1,
    _root_fields=_EVIDENCE_ROOT_FIELDS_V1,
    _schema=IMAGE_CLOSURE_EVIDENCE_SCHEMA_V1,
    _lock_prefix=STAGE_B_TRUST_LOCK_PREFIX_V2,
    _profile_prefix=AUTHORITY_PROFILE_PREFIX_V2,
    _registry_prefix=SIGNER_REGISTRY_PREFIX_V2,
    _policy_prefix=RUNTIME_CLOSURE_POLICY_PREFIX_V1,
    _fail_exact=_fail,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    tuple[str, ...],
]:
    root = _object(value, _root_fields, "image-closure evidence")
    body = _object(root["body"], _body_fields, "image-closure evidence body")
    if (
        body["schema_version"] != _schema
        or body["status"] != "OFFLINE_STATIC_IMAGE_CLOSURE_MECHANICS_CANDIDATE"
        or body["claim_level"]
        != (
            "RAW_STATIC_BYTES_MATCHED_ONLY_NOT_IMAGE_EXISTENCE_NOT_EXECUTION_"
            "NOT_LIVE_RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS"
        )
        or body["canonicalization"] != "RFC8785_EXACT_BYTES_NO_TRAILING_LF"
    ):
        _fail_exact(FAIL_SCHEMA_V1, "image-closure evidence header differs")
    _content_identifier_exact(
        body["stage_b_trust_lock_id"],
        _lock_prefix,
        "evidence Stage-B trust-lock ID",
    )
    _content_identifier_exact(
        body["authority_profile_id"],
        _profile_prefix,
        "evidence authority-profile ID",
    )
    _content_identifier_exact(
        body["signer_registry_id"],
        _registry_prefix,
        "evidence signer-registry ID",
    )
    _content_identifier_exact(
        body["closure_policy_id"],
        _policy_prefix,
        "evidence closure-policy ID",
    )
    if (
        body["stage_b_trust_lock_id"] != lock.artifact_id
        or body["authority_profile_id"] != profile.artifact_id
        or body["signer_registry_id"] != registry.artifact_id
        or body["service_name"] != service_binding["service_name"]
        or body["service_binding_id"] != service_binding["binding_id"]
        or body["closure_policy_id"] != policy_id
        or body["closure_policy_payload_sha256"] != policy_payload_sha256
    ):
        _fail_exact(FAIL_TRUST_ANCHOR_V1, "image-closure evidence lineage differs")
    _runtime_identity_exact(body["runtime_identity"])
    bounds = policy_body["resource_bounds"]
    support_rows = _validate_support(
        body["bound_artifacts_in_exact_order"],
        max_blob_bytes=bounds["max_blob_bytes"],
        source_binding=profile.body["source_binding"],
    )
    if not _equal(
        support_rows, policy_body["bound_artifacts_in_exact_order"]
    ):
        _fail_exact(FAIL_BINDING_V1, "evidence support commitments differ from policy")
    manifest_values = (
        body["stdlib_manifest"],
        body["import_manifest"],
        body["native_loader_manifest"],
        body["shared_library_manifest"],
    )
    manifest_keys = (
        "stdlib_manifest_sha256",
        "import_manifest_sha256",
        "native_loader_manifest_sha256",
        "shared_library_manifest_sha256",
    )
    manifest_digests: list[str] = []
    for expected_kind, commitment_key, manifest_value in zip(
        _manifest_kinds, manifest_keys, manifest_values, strict=True
    ):
        manifest = _validate_manifest(
            manifest_value, expected_kind=expected_kind, bounds=bounds
        )
        digest = _manifest_sha(manifest)
        if digest != policy_body["static_manifest_commitments"][commitment_key]:
            _fail_exact(FAIL_DIGEST_V1, f"{expected_kind} commitment differs")
        manifest_digests.append(digest)
    if len(set(manifest_digests)) != len(manifest_digests):
        _fail_exact(FAIL_SCHEMA_V1, "evidence static manifest digests alias")
    state = _object(
        body["verification_state"],
        _state_fields,
        "evidence verification state",
    )
    if not _equal(state, _exact_state):
        _fail_exact(FAIL_CLAIM_BOUNDARY_V1, "evidence verification state differs")
    image = _validate_image(
        body["oci_image"],
        media_policy=policy_body["oci_media_policy"],
        expected_platform=expected_platform,
        maximum_layers=bounds["max_layers"],
        maximum_blob_bytes=bounds["max_blob_bytes"],
    )
    expected_id = _derive_id(body)
    if root["id"] != expected_id:
        _fail_exact(FAIL_BINDING_V1, "image-closure evidence ID differs")
    return body, image, support_rows, tuple(manifest_digests)


def _verify_raw_oci_graph(
    *,
    raw_oci_index_payload: bytes,
    raw_oci_manifest_payload: bytes,
    raw_oci_config_payload: bytes,
    ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...],
    service_binding: Mapping[str, object],
    policy_body: Mapping[str, object],
    evidence_image: Mapping[str, object],
    expected_platform: Mapping[str, object],
    _hash_bytes=_hash_exact_bytes,
    _hash_chunks=_hash_chunk_tuple,
    _decode_oci=_decode_oci_json,
    _object=_object_exact,
    _validate_descriptor=_validate_oci_descriptor,
    _validate_platform=_validate_oci_platform,
    _oci_digest_exact=_oci_digest,
    _integer_exact=_integer,
    _equal=_equal_exact,
    _descriptor_fields=_OCI_DESCRIPTOR_FIELDS_V1,
    _fail_exact=_fail,
) -> tuple[str, str, str, tuple[str, ...]]:
    bounds = policy_body["resource_bounds"]
    media = policy_body["oci_media_policy"]
    index_sha, index_size = _hash_bytes(
        raw_oci_index_payload,
        maximum_bytes=bounds["max_oci_metadata_bytes"],
        name="raw OCI index",
    )
    manifest_sha, manifest_size = _hash_bytes(
        raw_oci_manifest_payload,
        maximum_bytes=bounds["max_oci_metadata_bytes"],
        name="raw OCI manifest",
    )
    config_sha, config_size = _hash_bytes(
        raw_oci_config_payload,
        maximum_bytes=bounds["max_oci_metadata_bytes"],
        name="raw OCI config",
    )
    index_digest = "sha256:" + index_sha
    manifest_digest = "sha256:" + manifest_sha
    config_digest = "sha256:" + config_sha
    if (
        index_digest != service_binding["oci_index_digest"]
        or manifest_digest != service_binding["oci_manifest_digest"]
        or config_digest != service_binding["oci_config_digest"]
    ):
        _fail_exact(FAIL_DIGEST_V1, "raw OCI graph differs from service binding")

    profile_layer_digests = service_binding["ordered_layer_digests"]
    if (
        type(ordered_raw_layer_chunk_tuples) is not tuple
        or type(profile_layer_digests) is not tuple
        or len(ordered_raw_layer_chunk_tuples) != len(profile_layer_digests)
        or not 1 <= len(ordered_raw_layer_chunk_tuples) <= bounds["max_layers"]
    ):
        _fail_exact(FAIL_INPUT_V1, "ordered raw OCI layer tuple cardinality differs")
    layer_digests: list[str] = []
    layer_sizes: list[int] = []
    total_layer_bytes = 0
    for ordinal, chunks in enumerate(ordered_raw_layer_chunk_tuples):
        if type(chunks) is not tuple:
            _fail_exact(FAIL_INPUT_V1, f"OCI layer {ordinal} chunks type differs")
        digest, byte_size = _hash_chunks(
            chunks,
            maximum_blob_bytes=bounds["max_blob_bytes"],
            maximum_chunk_bytes=bounds["max_chunk_bytes"],
            name=f"raw OCI layer {ordinal}",
        )
        total_layer_bytes += byte_size
        if total_layer_bytes > bounds["max_total_layer_bytes"]:
            _fail_exact(FAIL_INPUT_V1, "raw OCI layers exceed total-layer bound")
        oci_digest = "sha256:" + digest
        if oci_digest != profile_layer_digests[ordinal]:
            _fail_exact(FAIL_DIGEST_V1, f"raw OCI layer {ordinal} differs from profile")
        layer_digests.append(oci_digest)
        layer_sizes.append(byte_size)

    index = _object(
        _decode_oci(
            raw_oci_index_payload,
            maximum_bytes=bounds["max_oci_metadata_bytes"],
        ),
        frozenset({"schemaVersion", "mediaType", "manifests"}),
        "raw OCI index",
    )
    if (
        type(index["schemaVersion"]) is not int
        or index["schemaVersion"] != 2
        or index["mediaType"] != media["index_media_type"]
        or type(index["manifests"]) is not list
        or len(index["manifests"]) != 1
    ):
        _fail_exact(FAIL_OCI_GRAPH_V1, "raw OCI index header/cardinality differs")
    _validate_descriptor(
        index["manifests"][0],
        name="raw OCI index manifest descriptor",
        expected_media_type=media["manifest_media_type"],
        expected_digest=manifest_digest,
        expected_size=manifest_size,
        include_platform=True,
        expected_platform=expected_platform,
    )

    manifest = _object(
        _decode_oci(
            raw_oci_manifest_payload,
            maximum_bytes=bounds["max_oci_metadata_bytes"],
        ),
        frozenset({"schemaVersion", "mediaType", "config", "layers"}),
        "raw OCI manifest",
    )
    if (
        type(manifest["schemaVersion"]) is not int
        or manifest["schemaVersion"] != 2
        or manifest["mediaType"] != media["manifest_media_type"]
        or type(manifest["layers"]) is not list
        or len(manifest["layers"]) != len(layer_digests)
    ):
        _fail_exact(FAIL_OCI_GRAPH_V1, "raw OCI manifest header/cardinality differs")
    _validate_descriptor(
        manifest["config"],
        name="raw OCI config descriptor",
        expected_media_type=media["config_media_type"],
        expected_digest=config_digest,
        expected_size=config_size,
    )
    allowed_layer_media = media["allowed_layer_media_types_in_exact_order"]
    raw_layer_rows: list[dict[str, object]] = []
    for ordinal, (raw_descriptor, digest, byte_size) in enumerate(
        zip(manifest["layers"], layer_digests, layer_sizes, strict=True)
    ):
        descriptor = _object(
            raw_descriptor,
            _descriptor_fields,
            f"raw OCI layer descriptor {ordinal}",
        )
        if type(descriptor["mediaType"]) is not str or descriptor[
            "mediaType"
        ] not in allowed_layer_media:
            _fail_exact(FAIL_OCI_GRAPH_V1, f"raw OCI layer {ordinal} media differs")
        _oci_digest_exact(descriptor["digest"], f"raw OCI layer {ordinal} digest")
        _integer_exact(
            descriptor["size"],
            f"raw OCI layer {ordinal} size",
            minimum=byte_size,
            maximum=byte_size,
        )
        if descriptor["digest"] != digest:
            _fail_exact(FAIL_OCI_GRAPH_V1, f"raw OCI layer {ordinal} descriptor differs")
        raw_layer_rows.append(
            {
                "ordinal": ordinal,
                "media_type": descriptor["mediaType"],
                "digest": digest,
                "byte_size": byte_size,
            }
        )

    config = _decode_oci(
        raw_oci_config_payload,
        maximum_bytes=bounds["max_oci_metadata_bytes"],
    )
    if type(config) is not dict or not {"architecture", "os", "rootfs"}.issubset(config):
        _fail_exact(FAIL_OCI_GRAPH_V1, "raw OCI config required fields differ")
    if (
        config["architecture"] != expected_platform["architecture"]
        or config["os"] != expected_platform["os"]
    ):
        _fail_exact(FAIL_OCI_GRAPH_V1, "raw OCI config platform differs")
    if (
        (expected_platform["variant"] is None and "variant" in config)
        or (
            expected_platform["variant"] is not None
            and config.get("variant") != expected_platform["variant"]
        )
    ):
        _fail_exact(FAIL_OCI_GRAPH_V1, "raw OCI config platform variant differs")
    rootfs = _object(
        config["rootfs"],
        frozenset({"type", "diff_ids"}),
        "raw OCI config rootfs",
    )
    if rootfs["type"] != "layers" or type(rootfs["diff_ids"]) is not list:
        _fail_exact(FAIL_OCI_GRAPH_V1, "raw OCI rootfs header differs")
    diff_ids = [
        _oci_digest_exact(item, f"raw OCI rootfs diff ID {index}")
        for index, item in enumerate(rootfs["diff_ids"])
    ]
    if len(diff_ids) != len(layer_digests) or len(set(diff_ids)) != len(diff_ids):
        _fail_exact(FAIL_OCI_GRAPH_V1, "raw OCI rootfs diff-ID cardinality differs")

    expected_image = {
        "index_media_type": media["index_media_type"],
        "index_digest": index_digest,
        "index_byte_size": index_size,
        "manifest_media_type": media["manifest_media_type"],
        "manifest_digest": manifest_digest,
        "manifest_byte_size": manifest_size,
        "config_media_type": media["config_media_type"],
        "config_digest": config_digest,
        "config_byte_size": config_size,
        "platform": dict(expected_platform),
        "rootfs_diff_ids_in_exact_order": diff_ids,
        "ordered_layer_descriptors": raw_layer_rows,
    }
    if not _equal(evidence_image, expected_image):
        _fail_exact(FAIL_OCI_GRAPH_V1, "evidence OCI graph differs from raw bytes")
    return index_digest, manifest_digest, config_digest, tuple(layer_digests)


def _verify_support_artifact_bytes(
    *,
    ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...],
    support_rows: list[dict[str, object]],
    bounds: Mapping[str, object],
    _hash_chunks=_hash_chunk_tuple,
    _roles=SUPPORT_ARTIFACT_ROLES_V1,
    _fail_exact=_fail,
) -> tuple[str, ...]:
    if (
        type(ordered_support_artifact_chunk_tuples) is not tuple
        or len(ordered_support_artifact_chunk_tuples) != len(_roles)
    ):
        _fail_exact(FAIL_INPUT_V1, "ordered support-artifact chunk tuple differs")
    digests: list[str] = []
    for ordinal, (chunks, row, role) in enumerate(
        zip(
            ordered_support_artifact_chunk_tuples,
            support_rows,
            _roles,
            strict=True,
        )
    ):
        if type(chunks) is not tuple:
            _fail_exact(FAIL_INPUT_V1, f"{role} chunk sequence type differs")
        digest, byte_size = _hash_chunks(
            chunks,
            maximum_blob_bytes=bounds["max_blob_bytes"],
            maximum_chunk_bytes=bounds["max_chunk_bytes"],
            name=f"support artifact {ordinal} {role}",
        )
        if digest != row["sha256"] or byte_size != row["byte_size"]:
            _fail_exact(FAIL_DIGEST_V1, f"{role} raw bytes differ from commitment")
        digests.append(digest)
    return tuple(digests)


def canonical_runtime_closure_json_bytes_v1(
    value: object,
    _canonical=_authority_canonical_json_bytes_v2,
) -> bytes:
    """Return the shared exact RFC-8785 transport bytes for V1 objects."""

    return _canonical(value)


def phase2b_step5_runtime_closure_policy_id_v1(
    body: object,
    _derive=_runtime_closure_policy_id,
) -> str:
    """Derive the domain-separated closure-policy/1 content ID."""

    return _derive(body)


def phase2b_step5_image_closure_evidence_id_v1(
    body: object,
    _derive=_image_closure_evidence_id,
) -> str:
    """Derive the domain-separated image-closure-evidence/1 content ID."""

    return _derive(body)


def _verify_offline_image_closure_mechanics_implementation_v1(
    *,
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    service_name: str,
    runtime_closure_policy_payload: bytes,
    image_closure_evidence_payload: bytes,
    raw_oci_index_payload: bytes,
    raw_oci_manifest_payload: bytes,
    raw_oci_config_payload: bytes,
    ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...],
    ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...],
    _load_lock=_authority_load_lock_v2,
    _load_profile=_authority_load_profile_v2,
    _load_registry=_authority_load_registry_v2,
    _require_graph=_authority_require_graph_v2,
    _decode_canonical=_authority_decode_canonical_json_v2,
    _validate_policy=_validate_runtime_closure_policy,
    _validate_evidence=_validate_image_closure_evidence,
    _verify_oci=_verify_raw_oci_graph,
    _verify_support=_verify_support_artifact_bytes,
    _preflight_chunks=_preflight_chunk_matrix_shape,
    _issue_result=_hidden_issue_offline_closure_result_v1,
    _sha256=hashlib.sha256,
    _authority_error=Phase2BStep5AuthorityErrorV2,
    _services=SERVICE_ORDER_V2,
    _maximum_policy_or_evidence_bytes=_MAX_POLICY_OR_EVIDENCE_BYTES,
    _hard_maximum_layers=_HARD_MAX_LAYERS,
    _support_roles=SUPPORT_ARTIFACT_ROLES_V1,
    _fail_exact=_fail,
) -> VerifiedOfflineImageClosureMechanicsV1:
    """Replay one service's raw static image graph under the raw V2 lock.

    Every transport and blob input is immutable ``bytes`` or an exact tuple of
    immutable byte chunks.  The result proves only local offline hashing and
    schema/binding mechanics.  It never upgrades a static binding to observed
    image existence or live runtime closure.
    """

    raw_authority_payloads = (
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
    )
    if any(type(payload) is not bytes for payload in raw_authority_payloads):
        _fail_exact(FAIL_INPUT_V1, "authority graph transports must be exact bytes")
    if (
        type(out_of_band_expected_lock_id_bytes) is not bytes
        or type(out_of_band_expected_lock_payload_sha256_digest) is not bytes
        or len(out_of_band_expected_lock_payload_sha256_digest) != 32
    ):
        _fail_exact(FAIL_INPUT_V1, "out-of-band Stage-B pins must be exact raw bytes")
    if type(service_name) is not str or service_name not in _services:
        _fail_exact(FAIL_INPUT_V1, "service name must be exactly one frozen service")
    _preflight_chunks(
        ordered_raw_layer_chunk_tuples,
        name="ordered raw OCI layers",
        minimum_outer_items=1,
        maximum_outer_items=_hard_maximum_layers,
    )
    _preflight_chunks(
        ordered_support_artifact_chunk_tuples,
        name="ordered support artifacts",
        minimum_outer_items=len(_support_roles),
        maximum_outer_items=len(_support_roles),
    )
    if (
        type(runtime_closure_policy_payload) is not bytes
        or not 1
        <= len(runtime_closure_policy_payload)
        <= _maximum_policy_or_evidence_bytes
        or runtime_closure_policy_payload.endswith(b"\n")
        or type(image_closure_evidence_payload) is not bytes
        or not 1
        <= len(image_closure_evidence_payload)
        <= _maximum_policy_or_evidence_bytes
        or image_closure_evidence_payload.endswith(b"\n")
    ):
        _fail_exact(FAIL_INPUT_V1, "policy/evidence canonical transport differs")
    try:
        lock = _load_lock(
            stage_b_trust_lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        profile = _load_profile(authority_profile_payload, lock)
        registry = _load_registry(
            signer_registry_payload, lock, profile
        )
        lock, profile, registry = _require_graph(lock, profile, registry)
    except _authority_error as exc:
        raise Phase2BStep5RuntimeClosureErrorV1(
            FAIL_TRUST_ANCHOR_V1,
            f"raw authority/profile/registry V2 graph rejected: {exc.code}",
        ) from exc
    services = profile.body["service_static_bindings_in_exact_order"]
    if type(services) is not tuple or len(services) != len(_services):
        _fail_exact(FAIL_TRUST_ANCHOR_V1, "verified profile service shape differs")
    service_ordinal = _services.index(service_name)
    service_binding = services[service_ordinal]
    if (
        not isinstance(service_binding, Mapping)
        or service_binding["service_name"] != service_name
    ):
        _fail_exact(FAIL_TRUST_ANCHOR_V1, "verified profile service order differs")

    policy_payload_sha256 = _sha256(runtime_closure_policy_payload).hexdigest()
    if policy_payload_sha256 != service_binding[
        "static_runtime_closure_policy_sha256"
    ]:
        _fail_exact(
            FAIL_DIGEST_V1,
            "runtime-closure policy bytes differ from static service binding",
        )
    try:
        policy_value = _decode_canonical(
            runtime_closure_policy_payload,
            maximum_bytes=_maximum_policy_or_evidence_bytes,
        )
        evidence_value = _decode_canonical(
            image_closure_evidence_payload,
            maximum_bytes=_maximum_policy_or_evidence_bytes,
        )
    except _authority_error as exc:
        raise Phase2BStep5RuntimeClosureErrorV1(
            FAIL_CANONICAL_V1,
            f"policy/evidence canonical bytes rejected: {exc.code}",
        ) from exc
    policy_body, platform = _validate_policy(
        policy_value,
        profile=profile,
        service_binding=service_binding,
    )
    policy_id = policy_value["id"]
    evidence_body, evidence_image, support_rows, manifest_digests = (
        _validate_evidence(
            evidence_value,
            lock=lock,
            profile=profile,
            registry=registry,
            service_binding=service_binding,
            policy_id=policy_id,
            policy_payload_sha256=policy_payload_sha256,
            policy_body=policy_body,
            expected_platform=platform,
        )
    )
    index_digest, manifest_digest, config_digest, layer_digests = _verify_oci(
        raw_oci_index_payload=raw_oci_index_payload,
        raw_oci_manifest_payload=raw_oci_manifest_payload,
        raw_oci_config_payload=raw_oci_config_payload,
        ordered_raw_layer_chunk_tuples=ordered_raw_layer_chunk_tuples,
        service_binding=service_binding,
        policy_body=policy_body,
        evidence_image=evidence_image,
        expected_platform=platform,
    )
    support_digests = _verify_support(
        ordered_support_artifact_chunk_tuples=(
            ordered_support_artifact_chunk_tuples
        ),
        support_rows=support_rows,
        bounds=policy_body["resource_bounds"],
    )
    return _issue_result(
        stage_b_trust_lock_id=lock.artifact_id,
        authority_profile_id=profile.artifact_id,
        signer_registry_id=registry.artifact_id,
        service_name=service_name,
        service_binding_id=service_binding["binding_id"],
        closure_policy_id=policy_id,
        closure_policy_payload_sha256=policy_payload_sha256,
        evidence_id=evidence_value["id"],
        oci_index_digest=index_digest,
        oci_manifest_digest=manifest_digest,
        oci_config_digest=config_digest,
        ordered_layer_digests=layer_digests,
        support_artifact_digests=support_digests,
        static_manifest_digests=manifest_digests,
        offline_static_byte_graph_mechanics_verified=True,
        mechanics_only=True,
        service_image_exists=False,
        service_image_pulled=False,
        service_image_executed=False,
        live_runtime_closure_observed=False,
        entry_exit_closure_equal=False,
        live_uid_gid_verified=False,
        live_control_domain_verified=False,
        key_custody_verified=False,
        runtime_closure_signature_verified=False,
        remote_attestation_verified=False,
        external_cas_verified=False,
        authority_evidence=False,
        readiness_evidence=False,
        r01_satisfied=False,
        execution_authorized=False,
    )


def _build_public_runtime_closure_entrypoints_v1(
    *,
    _verify_impl=_verify_offline_image_closure_mechanics_implementation_v1,
    _require_impl=_hidden_require_offline_closure_result_v1,
):
    def verify(
        *,
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
        service_name: str,
        runtime_closure_policy_payload: bytes,
        image_closure_evidence_payload: bytes,
        raw_oci_index_payload: bytes,
        raw_oci_manifest_payload: bytes,
        raw_oci_config_payload: bytes,
        ordered_raw_layer_chunk_tuples: tuple[tuple[bytes, ...], ...],
        ordered_support_artifact_chunk_tuples: tuple[tuple[bytes, ...], ...],
    ) -> VerifiedOfflineImageClosureMechanicsV1:
        return _verify_impl(
            stage_b_trust_lock_payload=stage_b_trust_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
            service_name=service_name,
            runtime_closure_policy_payload=runtime_closure_policy_payload,
            image_closure_evidence_payload=image_closure_evidence_payload,
            raw_oci_index_payload=raw_oci_index_payload,
            raw_oci_manifest_payload=raw_oci_manifest_payload,
            raw_oci_config_payload=raw_oci_config_payload,
            ordered_raw_layer_chunk_tuples=ordered_raw_layer_chunk_tuples,
            ordered_support_artifact_chunk_tuples=(
                ordered_support_artifact_chunk_tuples
            ),
        )

    def require(value: object) -> VerifiedOfflineImageClosureMechanicsV1:
        return _require_impl(value)

    return verify, require


(
    verify_offline_step5_image_closure_mechanics_v1,
    require_verified_offline_step5_image_closure_mechanics_v1,
) = _build_public_runtime_closure_entrypoints_v1()
del _build_public_runtime_closure_entrypoints_v1
del _verify_offline_image_closure_mechanics_implementation_v1
del _hidden_issue_offline_closure_result_v1
del _hidden_require_offline_closure_result_v1


__all__ = [
    "CPYTHON_ABI_TAG_V1",
    "CPYTHON_CACHE_TAG_V1",
    "CPYTHON_IMPLEMENTATION_V1",
    "CPYTHON_LANGUAGE_VERSION_V1",
    "CPYTHON_VERSION_V1",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_DIGEST_V1",
    "FAIL_INPUT_V1",
    "FAIL_OCI_GRAPH_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_TRUST_ANCHOR_V1",
    "IMAGE_CLOSURE_EVIDENCE_PREFIX_V1",
    "IMAGE_CLOSURE_EVIDENCE_BODY_FIELDS_V1",
    "IMAGE_CLOSURE_EVIDENCE_SCHEMA_V1",
    "OCI_INDEX_MEDIA_TYPE_V1",
    "OCI_MANIFEST_MEDIA_TYPE_V1",
    "PYTHON_MODULE_CLASSIFICATIONS_V1",
    "PYTHON_ORIGIN_KINDS_V1",
    "Phase2BStep5RuntimeClosureErrorV1",
    "RUNTIME_CLOSURE_POLICY_PREFIX_V1",
    "RUNTIME_CLOSURE_POLICY_BODY_FIELDS_V1",
    "RUNTIME_CLOSURE_POLICY_SCHEMA_V1",
    "STATIC_MANIFEST_KINDS_V1",
    "STATIC_RUNTIME_MANIFEST_SCHEMA_V1",
    "STATIC_RUNTIME_MANIFEST_FIELDS_V1",
    "SUPPORT_ARTIFACT_MEDIA_TYPES_V1",
    "SUPPORT_ARTIFACT_ROLES_V1",
    "VerifiedOfflineImageClosureMechanicsV1",
    "canonical_runtime_closure_json_bytes_v1",
    "phase2b_step5_image_closure_evidence_id_v1",
    "phase2b_step5_runtime_closure_policy_id_v1",
    "require_verified_offline_step5_image_closure_mechanics_v1",
    "verify_offline_step5_image_closure_mechanics_v1",
]

#!/usr/bin/env python3
"""Commit-bound, offline dual strict qualification shell for shrink step 6.

The supervisor is an evidence generator, not a recognizer.  It extracts an
exact Git snapshot and is prepared to feed the sealed 25-vector wires to two
separately executed recognizers. Its roots, challenge-lattice commitments and
endpoint schemas were frozen only after independent Python/Rust reconciliation.
It never runs closure, creates formal roots, or advances M3 out of ``NOT_RUN``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha1, sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import Final, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
MODULE_ROOT: Final = PROJECT_ROOT / "src/hegel_machine"

# Load only the sealed evidence-generator dependency closure.  Do not execute
# the package initializer, which exposes unrelated target and split APIs.
if "hegel_machine" not in sys.modules:
    package = ModuleType("hegel_machine")
    package.__path__ = [str(MODULE_ROOT)]  # type: ignore[attr-defined]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package

from hegel_machine.phase3_m3_shrink6_diagnostic_profile_v1 import (  # noqa: E402
    diagnostic_root_hex_v1,
)
from hegel_machine.phase3_shrink6_golden_vectors_v1 import (  # noqa: E402
    ACCEPT_PARENT_IDENTITY,
    STRICT_GOLDEN_VECTORS_V1,
    accepted_outcome_bytes,
    rejected_outcome_bytes,
    strict_golden_manifest_root_v1,
    strict_golden_outcome_root_v1,
)


SCHEMA: Final = "hegel-shrink6-sealed-dual-strict-qualification/1"
CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_STRICT_QUALIFICATION"
STATUS_PASS: Final = "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
SOURCE_SET_DOMAIN: Final = b"HEGEL/SHRINK6/DUAL_STRICT_SOURCE_SET/V1"
REPORT_DOMAIN: Final = b"HEGEL/SHRINK6/DUAL_STRICT_REPORT/V1"
DAEMON_RECEIPT_DOMAIN: Final = b"HEGEL/SHRINK6/DOCKER_DAEMON_RECEIPT/V1"
CARGO_SEED_MANIFEST_DOMAIN: Final = b"HEGEL/SHRINK6/CARGO_SEED_MANIFEST/V1"
AST_HASH_DOMAIN: Final = b"HEGEL/AST/V1"
PARENT_EVIDENCE_COMMIT: Final = "5bfe8474ca63abbadb1d3484a51ce3012081dfb3"
PARENT_EVIDENCE_PATH: Final = (
    "Hegel Machine/artifacts/"
    "phase3_shrink5_dual_complete_enumeration_diagnostic_v1.json"
)
PARENT_EVIDENCE_RECORD_ID: Final = (
    "phase3_shrink5_dual_complete_enumeration_diagnostic_"
    "f33b86f3fbab70acb7d8e61fa47f59568a0d56c884c4cf75dfef961cc73dd34b"
)
PARENT_EVIDENCE_SHA256: Final = (
    "99a799e34876754a8f938f8e25f756992d0784b03bae398b1434e57320b80c82"
)
# Frozen only after independent Python and Rust replays agreed bit-for-bit.
# These are shrink-6 values; no shrink-5 commitment is carried forward.
EXPECTED_DIAGNOSTIC_ROOTS: Final[dict[str, str] | None] = {
    "child_dsl_spec_root": (
        "da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae"
    ),
    "operator_semantics_root": (
        "922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03"
    ),
    "identifier_registry_root": (
        "64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1"
    ),
    "canonical_ast_schema_root": (
        "5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd"
    ),
    "canonical_cbor_profile_root": (
        "ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab"
    ),
}
EXPECTED_MANIFEST_ROOT: Final[str | None] = (
    "sha256:2690413926d15db52dbd5a502ebe3fdfb1dc74d5ee3c82b2ed868cd16ab34a42"
)
EXPECTED_OUTCOME_ROOT: Final[str | None] = (
    "sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960"
)
EXPECTED_CHALLENGE_SOURCE_LATTICE_COMMITMENT: Final[str | None] = (
    "sha256:a8cfb37278000933c2c51a2797e5bc0f4e7aad6970b37e178fc681f9358574d0"
)
EXPECTED_CHALLENGE_PARENT_CANONICAL_SET_COMMITMENT: Final[str | None] = (
    "sha256:8f125763d3098d087dd7e9eb484b93097295ebd765b6f079795e8009623fb13e"
)
EXPECTED_INHERITED_SURVIVOR_COMMITMENT: Final[str | None] = (
    "sha256:477a5abe659a7a7e7d2d50b2a5bda61b0dae1019c44fe84950c4a05036258619"
)
EXPECTED_NORMALIZED_SURVIVOR_COMMITMENT: Final[str | None] = (
    "sha256:dcbb5562fc754fdef932188b189dbcdc0f7c500d3fc49651ee4dbb0f271afd29"
)
EXPECTED_SURVIVOR_COMMITMENT: Final[str | None] = (
    "sha256:6787cd6c0782fda149e1ee93b37ca8d425f5ac78850c610e21cebf9da13a16d1"
)
EXPECTED_PARENT_ONLY_SET_COMMITMENT: Final[str | None] = (
    "sha256:d3eb2b2d9caf1eece5a709d8113540e4709d579cdfbe3194f1cf176c9100b20d"
)
EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT: Final[str | None] = (
    "sha256:9b0b766a4139db6297aea8b6032ad49147c1a26bf9b56291444a83681428cb0e"
)
EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT: Final[str | None] = (
    "sha256:97d50c34f51683a2502157961acc79d3b4e108b28bdaa266cf3721ffda8b3a96"
)
EXPECTED_FIRST_SURVIVOR_CBOR_HEX: Final[str | None] = (
    "820183010283010083000100"
)
EXPECTED_FIRST_SURVIVOR_HASH: Final[str | None] = (
    "sha256:0f319bb95ea24abc9b4c62d03274a20cefe5dbb92fcfffbce0f0e9449aab04a6"
)
EXPECTED_LAST_SURVIVOR_CBOR_HEX: Final[str | None] = "820186000305030180"
EXPECTED_LAST_SURVIVOR_HASH: Final[str | None] = (
    "sha256:e35153f2bdd1a6e25d629ed3ab9afb178bb45ecd163efba4960a2a69db40ce2c"
)
EXPECTED_VECTOR_IDS: Final[tuple[str, ...] | None] = (
    "S01", "S02", "S03", "N01", "N02", "L01", "L02", "L03", "P01",
    "P02", "P03", "P04", "P05", "F01", "F02", "F03", "F04", "F05",
    "F06", "F07", "F08", "F09", "F10", "F11", "F12",
)
EXPECTED_VECTOR_COUNT: Final = 25
PREREGISTERED_FORMAL_BUCKET_COUNT: Final = 120
CAPACITY_GENERATOR_RULE: Final = (
    "family order A,B_abs,B_sign; operand outer, R inner, direction 0 then 1; "
    "R is active constants -1,0,1 followed by the exact inherited 16 rational "
    "aggregate leaves in map/scope/quantity order; A-U1 is bit_to_scalar(bit_at "
    "0..7), int_to_scalar(set_size), int_to_scalar(count_nonzero) in "
    "scope/quantity order, then absolute of the inherited rational aggregates; "
    "B-Q is absolute of the first 17 A-U1 non-rational-aggregate forms; rows "
    "with aggregate-bearing operand and aggregate R are excluded; no source "
    "deduplication; this is FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE"
)
PYTHON_IMAGE: Final = (
    "python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE: Final = (
    "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
RUST_TOOLCHAIN_BIN: Final = (
    "/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin"
)
DEFAULT_CARGO_REGISTRY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache/registry"
)
DOCKER_EXECUTABLE: Final = "/usr/bin/docker"
DOCKER_HOST_ARGUMENT: Final = "--host=unix:///var/run/docker.sock"
DOCKER_SOCKET: Final = Path("/var/run/docker.sock")
RUNTIME_SECCOMP_PATH: Final = (
    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json"
)
BUILD_SECCOMP_PATH: Final = (
    "Hegel Machine/config/phase3_m3_offline_build_seccomp_v1.json"
)
BUILD_PROFILE_PATH: Final = (
    "Hegel Machine/config/phase3_shrink6_offline_build_profile_v1.json"
)
RUNTIME_TMPFS: Final = (
    "/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700"
)
BUILD_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m,uid=0,gid=0,mode=0700"
_DOCKER_ENV: dict[str, str] | None = None

FAIL_ARGUMENT = "FAIL_SHRINK6_DUAL_STRICT_ARGUMENT"
FAIL_GIT_BINDING = "FAIL_SHRINK6_DUAL_STRICT_GIT_BINDING"
FAIL_ARCHIVE = "FAIL_SHRINK6_DUAL_STRICT_ARCHIVE"
FAIL_DOCKER_POLICY = "FAIL_SHRINK6_DUAL_STRICT_DOCKER_POLICY"
FAIL_BUILD = "FAIL_SHRINK6_DUAL_STRICT_RUST_BUILD"
FAIL_ENDPOINT = "FAIL_SHRINK6_DUAL_STRICT_ENDPOINT"
FAIL_VECTOR = "FAIL_SHRINK6_DUAL_STRICT_VECTOR"
FAIL_CAPACITY = "FAIL_SHRINK6_DUAL_STRICT_CAPACITY"
FAIL_GUARD = "FAIL_SHRINK6_DUAL_STRICT_AUTHORITY_GUARD"
FAIL_CLEANUP = "FAIL_SHRINK6_DUAL_STRICT_CLEANUP"
FAIL_INTERNAL = "FAIL_SHRINK6_DUAL_STRICT_INTERNAL"
FAIL_UNSEALED = "FAIL_SHRINK6_DUAL_STRICT_UNSEALED_COMMITMENTS"

PYTHON_SOURCE_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/hashing.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_record_wire_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink4_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink5_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink6_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink6_diagnostic_profile_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink2_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink3_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink4_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink5_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_capacity_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_golden_vectors_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_strict_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink2_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink3_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink4_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink5_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink6_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
    "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
)
RUST_SOURCE_DIRS: Final = (
    "Hegel Machine/rust/strict_canonicalizer",
    "Hegel Machine/rust/strict_canonicalizer_shrink1",
    "Hegel Machine/rust/strict_canonicalizer_shrink2",
    "Hegel Machine/rust/strict_canonicalizer_shrink3",
    "Hegel Machine/rust/strict_canonicalizer_shrink4",
    "Hegel Machine/rust/strict_canonicalizer_shrink5",
    "Hegel Machine/rust/strict_canonicalizer_shrink6",
)
SUPERVISOR_PATH: Final = (
    "Hegel Machine/tools/phase3_shrink6_dual_strict_qualification_v1.py"
)
FREEZE_DOC_PATH: Final = (
    "Hegel Machine/docs/Hegel_Machine_Phase3_Shrink_Step6_Engineering_Freeze_v1.md"
)
PROTOCOL_DOC_PATH: Final = (
    "Hegel Machine/docs/"
    "Hegel_Machine_Phase3_Shrink6_Sealed_Dual_Strict_Qualification_Protocol_v1.md"
)
SUPERVISOR_TEST_PATH: Final = (
    "Hegel Machine/tests/test_phase3_shrink6_dual_strict_qualification_v1.py"
)
PROFILE_PATH: Final = "Hegel Machine/config/phase3_container_actor_profile_v1.json"
ARCHIVE_PATHS: Final = (
    SUPERVISOR_PATH,
    FREEZE_DOC_PATH,
    PROTOCOL_DOC_PATH,
    SUPERVISOR_TEST_PATH,
    PARENT_EVIDENCE_PATH,
    PROFILE_PATH,
    BUILD_PROFILE_PATH,
    RUNTIME_SECCOMP_PATH,
    BUILD_SECCOMP_PATH,
    *PYTHON_SOURCE_PATHS,
    *RUST_SOURCE_DIRS,
)

CARGO_SEED_SUBTREES: Final = ("cache", "index")
CARGO_SEED_FILE_FIELDS: Final = frozenset({"path", "mode", "size", "sha256"})
CARGO_SEED_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "hash_domain",
        "subtrees",
        "file_count",
        "total_byte_count",
        "files",
        "manifest_root",
    }
)
SOURCE_FILE_ROW_FIELDS: Final = frozenset(
    {"path", "mode", "git_blob_oid", "sha256", "size"}
)
REPOSITORY_BINDING_FIELDS: Final = frozenset(
    {
        "qualification_basis_commit",
        "qualification_basis_parent_commits",
        "qualification_basis_subject",
        "project_tree_oid",
        "archive_sha256",
        "source_file_count",
        "source_file_set_root",
        "supervisor_source_sha256",
        "parent_evidence_binding",
        "source_files",
    }
)
DUAL_VECTOR_REPLAY_FIELDS: Final = frozenset(
    {
        "status",
        "vector_count",
        "python_outcome_root",
        "rust_outcome_root",
        "all_normalized_outcomes_equal",
        "vectors",
    }
)
VECTOR_RECEIPT_FIELDS: Final = frozenset(
    {
        "vector_id",
        "category",
        "boundary",
        "input_wire_sha256",
        "input_wire_size",
        "expected_disposition",
        "python_exit",
        "rust_exit",
        "normalized",
        "normalized_outcome_sha256",
        "dual_equal",
    }
)

CAPACITY_EXCLUDED_FIELDS: Final = frozenset(
    {"implementation", "loaded_hegel_modules"}
)
ACCEPT_COMPARISON_FIELDS: Final = (
    "status",
    "canonical_cbor_hex",
    "canonical_ast_hash",
    "root_operator_id",
    "output_sort",
    "depth",
    "node_count",
    "maximum_ast_depth",
    "maximum_ast_node_count",
    "maximum_top_level_clauses",
)
EXPECTED_PYTHON_STRICT_MODULES: Final = [
    "hegel_machine.phase3_m3_dsl_core_v1",
    "hegel_machine.phase3_m3_shrink1_core_v1",
    "hegel_machine.phase3_m3_shrink2_core_v1",
    "hegel_machine.phase3_m3_shrink3_core_v1",
    "hegel_machine.phase3_m3_shrink4_core_v1",
    "hegel_machine.phase3_m3_shrink5_core_v1",
    "hegel_machine.phase3_m3_shrink6_core_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_shrink3_v1",
    "hegel_machine.strict_ast_shrink4_v1",
    "hegel_machine.strict_ast_shrink5_v1",
    "hegel_machine.strict_ast_shrink6_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
]
EXPECTED_PYTHON_CAPACITY_MODULES: Final = [
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
]
_STRICT_COMMON_FIELDS: Final = frozenset(
    {
        "schema_version", "implementation", "dsl_version", "freeze_version",
        "boundary", "status", "maximum_ast_depth", "maximum_ast_node_count",
        "maximum_top_level_clauses",
        "target_or_split_modules_loaded",
    }
)
_STRICT_ACCEPT_FIELDS: Final = frozenset(
    {
        "canonical_cbor_hex", "canonical_ast_hash", "root_operator_id",
        "output_sort", "depth", "node_count",
    }
)
PYTHON_STRICT_ACCEPT_FIELDS: Final = (
    _STRICT_COMMON_FIELDS | _STRICT_ACCEPT_FIELDS | {"loaded_hegel_modules"}
)
PYTHON_STRICT_REJECT_FIELDS: Final = (
    _STRICT_COMMON_FIELDS | {"error_code", "error_detail", "loaded_hegel_modules"}
)
_RUST_STRICT_METADATA_FIELDS: Final = frozenset(
    {
        "parent_dsl_version", "parent_freeze_version", "cbor_profile_id",
        "ast_schema_id", "ast_hash_domain",
    }
)
RUST_STRICT_ACCEPT_FIELDS: Final = (
    _STRICT_COMMON_FIELDS
    | _STRICT_ACCEPT_FIELDS
    | _RUST_STRICT_METADATA_FIELDS
    | {"scalar_parameter_occurrence_count"}
)
RUST_STRICT_REJECT_FIELDS: Final = (
    _STRICT_COMMON_FIELDS
    | _RUST_STRICT_METADATA_FIELDS
    | {"error_code", "error_message"}
)

RUST_CAPACITY_FIELDS: Final = frozenset(
    {
        "schema_version", "implementation", "parent_dsl_version",
        "parent_freeze_version", "dsl_version", "freeze_version",
        "human_amendment_id", "shrink_step_id", "generator_rule",
        "removed_binary_operator_ids", "retained_difference_id",
        "constant_atom_count", "rational_aggregate_count", "mixed_atom_count",
        "challenge_source_candidate_count", "challenge_source_family_counts",
        "challenge_source_lattice_commitment",
        "challenge_parent_accepted_count",
        "challenge_parent_canonical_unique_count",
        "challenge_parent_canonical_set_commitment",
        "inherited_survivor_source_count", "inherited_survivor_unique_count",
        "inherited_survivor_set_commitment",
        "normalized_survivor_source_count",
        "normalized_survivor_source_family_counts",
        "normalized_survivor_unique_count",
        "normalized_survivor_set_commitment",
        "survivor_source_candidate_count", "survivor_accepted_count",
        "survivor_unique_count", "survivor_parent_identity_match_count",
        "survivor_rejected_count", "survivor_rejection_counts",
        "survivor_accepted_set_commitment",
        "first_survivor_canonical_cbor_hex",
        "first_survivor_canonical_ast_hash",
        "last_survivor_canonical_cbor_hex",
        "last_survivor_canonical_ast_hash",
        "parent_only_source_candidate_count",
        "parent_only_parent_accepted_count", "parent_only_depth",
        "parent_only_node_count", "parent_only_unique_count",
        "parent_only_source_family_counts",
        "parent_only_set_commitment",
        "parent_only_source_child_rejected_count",
        "parent_only_source_child_rejection_counts",
        "parent_only_source_rejection_outcome_commitment",
        "parent_only_formal_child_rejected_count",
        "parent_only_formal_child_rejection_counts",
        "parent_only_formal_rejection_outcome_commitment",
        "canonical_program_budget", "first_out_of_budget_ordinal",
        "subset_status", "executed_closure_status",
        "complete_closure_enumerated", "interpreted_as_complete_closure",
        "formal_roots", "target_or_split_modules_loaded",
        "maximum_ast_depth", "maximum_ast_node_count",
        "maximum_top_level_clauses",
    }
)
PYTHON_CAPACITY_FIELDS: Final = RUST_CAPACITY_FIELDS | {"loaded_hegel_modules"}

RUST_GOLDEN_FIELDS: Final = frozenset(
    {
        "schema_version", "implementation", "parent_dsl_version",
        "parent_freeze_version", "dsl_version", "freeze_version",
        "human_amendment_id", "shrink_step_id",
        "active_source_binary_operator_ids", "active_formal_binary_operator_ids",
        "source_alias_binary_operator_ids", "tombstoned_binary_operator_ids",
        "reserved_binary_operator_ids", "removed_binary_operator_error",
        "maximum_ast_depth", "maximum_ast_node_count",
        "maximum_top_level_clauses",
        "vector_count", "passed_count",
        "surviving_identity_checks", "source_normalization_before_limit_checks",
        "source_depth_limit_checks", "source_priority_checks",
        "formal_surviving_identity_checks", "formal_depth_limit_checks",
        "formal_priority_checks", "execution_state", "closure_executed",
        "formal_roots_generated", "formal_roots",
        "golden_vector_manifest_root", "golden_outcome_root",
        "ordered_vector_ids",
        "target_or_split_modules_loaded",
    }
)
PYTHON_GOLDEN_FIELDS: Final = RUST_GOLDEN_FIELDS | {"loaded_hegel_modules"}

DUAL_CAPACITY_REPLAY_FIELDS: Final = frozenset(
    (set(RUST_CAPACITY_FIELDS) - {"implementation"})
    | {"status", "all_comparable_fields_equal", "comparable_report_sha256"}
)
RUNTIME_ISOLATION_FIELDS: Final = frozenset(
    {
        "role_topology", "same_admin_controller", "organizational_independence",
        "independent_human_actors", "technical_role_independence",
        "owner_accepted_threat_model", "docker_daemon_identity_receipt",
        "committed_profile_receipt", "python_image_ref", "python_image_id",
        "rust_image_ref", "rust_image_id", "pull_policy", "network_mode",
        "capabilities_dropped", "no_new_privileges",
        "container_root_filesystem_read_only",
        "source_snapshot_mount_read_only",
        "fresh_ephemeral_rust_target_volume",
        "rust_target_volume_removed_after_run", "rust_target_volume_receipt",
        "cargo_locked", "cargo_offline", "cargo_seed_subtrees",
        "cargo_seed_manifest_receipt", "preunpacked_registry_src_mounted",
        "cargo_home", "rust_build_user", "rust_build_tmpfs",
        "recognizer_runtime_user", "recognizer_runtime_tmpfs", "memory_limit",
        "memory_swap_limit", "pids_limit", "python_flags", "worker_count",
        "python_runtime", "rust_runtime", "rust_binary_sha256",
    }
)
DOCKER_DAEMON_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version", "docker_executable", "explicit_host_argument",
        "socket", "socket_device", "socket_inode", "socket_uid", "socket_gid",
        "private_empty_client_config_sha256", "host_environment_keys", "server",
        "daemon", "diagnostic_receipt_hash",
    }
)
DOCKER_SERVER_RECEIPT_FIELDS: Final = frozenset(
    {
        "version", "api_version", "minimum_api_version", "os", "architecture",
        "git_commit", "go_version", "kernel_version", "build_time",
        "raw_canonical_sha256",
    }
)
DOCKER_INFO_RECEIPT_FIELDS: Final = frozenset(
    {
        "id", "name", "driver", "operating_system", "os_type",
        "architecture", "docker_root_dir", "security_options",
    }
)
TARGET_VOLUME_RECEIPT_FIELDS: Final = frozenset(
    {"name", "driver", "scope", "options", "labels", "fresh_before_run"}
)
PYTHON_RUNTIME_RECEIPT_FIELDS: Final = frozenset(
    {"executable", "executable_sha256", "version"}
)
RUST_RUNTIME_RECEIPT_FIELDS: Final = frozenset({"rustc_version_verbose"})


class QualificationError(RuntimeError):
    """Stable fail-closed supervisor error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise QualificationError(code, detail)


def _require_sealed_commitments() -> None:
    """Reject a partially edited source before Git or Docker execution."""

    pending = {
        "diagnostic_roots": EXPECTED_DIAGNOSTIC_ROOTS,
        "golden_manifest_root": EXPECTED_MANIFEST_ROOT,
        "golden_outcome_root": EXPECTED_OUTCOME_ROOT,
        "challenge_source_lattice_commitment": (
            EXPECTED_CHALLENGE_SOURCE_LATTICE_COMMITMENT
        ),
        "challenge_parent_canonical_set_commitment": (
            EXPECTED_CHALLENGE_PARENT_CANONICAL_SET_COMMITMENT
        ),
        "inherited_survivor_commitment": EXPECTED_INHERITED_SURVIVOR_COMMITMENT,
        "normalized_survivor_commitment": (
            EXPECTED_NORMALIZED_SURVIVOR_COMMITMENT
        ),
        "survivor_commitment": EXPECTED_SURVIVOR_COMMITMENT,
        "parent_only_set_commitment": EXPECTED_PARENT_ONLY_SET_COMMITMENT,
        "parent_only_source_rejection_commitment": (
            EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT
        ),
        "parent_only_formal_rejection_commitment": (
            EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT
        ),
        "first_survivor_cbor_hex": EXPECTED_FIRST_SURVIVOR_CBOR_HEX,
        "first_survivor_hash": EXPECTED_FIRST_SURVIVOR_HASH,
        "last_survivor_cbor_hex": EXPECTED_LAST_SURVIVOR_CBOR_HEX,
        "last_survivor_hash": EXPECTED_LAST_SURVIVOR_HASH,
        "ordered_vector_ids": EXPECTED_VECTOR_IDS,
    }
    unsealed = sorted(name for name, value in pending.items() if value is None)
    if unsealed:
        _fail(
            FAIL_UNSEALED,
            "implementation-generated commitments are not sealed: "
            + ",".join(unsealed),
        )
    assert EXPECTED_VECTOR_IDS is not None
    if len(EXPECTED_VECTOR_IDS) != EXPECTED_VECTOR_COUNT:
        _fail(FAIL_UNSEALED, "sealed vector ID count is not exactly 25")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_exact_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""

    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _reject_json_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _docker_prefix() -> list[str]:
    return [DOCKER_EXECUTABLE, DOCKER_HOST_ARGUMENT]


def _docker_environment() -> dict[str, str]:
    if _DOCKER_ENV is None:
        _fail(FAIL_DOCKER_POLICY, "private Docker control environment is not initialized")
    return _DOCKER_ENV


def _cleanup_timed_out_container(command: Sequence[str]) -> str | None:
    if not command or command[0] != DOCKER_EXECUTABLE or "run" not in command:
        return None
    try:
        marker = command.index("--name")
        name = command[marker + 1]
    except (ValueError, IndexError):
        return "Docker run timeout had no exact container name"
    if re.fullmatch(r"hegel-s6-[0-9a-f]{16}", name) is None:
        return "Docker run timeout carried an invalid cleanup name"
    try:
        cleanup = subprocess.run(
            [*_docker_prefix(), "rm", "-f", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_docker_environment(),
            timeout=30,
            check=False,
        )
        inspect = subprocess.run(
            [*_docker_prefix(), "container", "inspect", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_docker_environment(),
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"container cleanup transport failed: {type(error).__name__}: {error}"
    if inspect.returncode == 1:
        return None
    return (
        f"container cleanup failed: rm={cleanup.returncode}, "
        f"inspect={inspect.returncode}"
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    allowed_codes: frozenset[int] = frozenset({0}),
    code: str = FAIL_ENDPOINT,
) -> subprocess.CompletedProcess[bytes]:
    environment = None
    if command and command[0] == DOCKER_EXECUTABLE:
        environment = _docker_environment()
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        cleanup_detail = _cleanup_timed_out_container(command)
        suffix = "" if cleanup_detail is None else f"; {cleanup_detail}"
        _fail(code, f"command timed out after {timeout}s{suffix}")
    except OSError as error:
        _fail(code, f"command transport failed: {error}")
    if result.returncode not in allowed_codes:
        detail = result.stderr.decode("utf-8", "replace").strip()
        _fail(code, f"exit {result.returncode}: {detail[:1000]}")
    return result


def _one_json(result: subprocess.CompletedProcess[bytes], label: str) -> dict[str, object]:
    try:
        value = json.loads(
            result.stdout,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        _fail(FAIL_ENDPOINT, f"{label} did not emit one JSON object: {error}")
    if type(value) is not dict:
        _fail(FAIL_ENDPOINT, f"{label} emitted non-object JSON")
    return value


def _git(*arguments: str, binary: bool = False) -> bytes | str:
    result = _run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        code=FAIL_GIT_BINDING,
    )
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def _git_blob_oid(payload: bytes) -> str:
    return sha1(b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload).hexdigest()


def source_file_set_root_v1(rows: Sequence[Mapping[str, object]]) -> str:
    digest = sha256()
    digest.update(SOURCE_SET_DOMAIN + b"\x00")
    previous = ""
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != SOURCE_FILE_ROW_FIELDS:
            _fail(FAIL_GUARD, f"source-file row {index} fields differ")
        path = row["path"]
        mode = row["mode"]
        oid = row["git_blob_oid"]
        file_hash = row["sha256"]
        size = row["size"]
        pure = PurePosixPath(path) if type(path) is str else None
        if (
            type(path) is not str
            or pure is None
            or pure.is_absolute()
            or pure.as_posix() != path
            or not pure.parts
            or pure.parts[0] != "Hegel Machine"
            or any(part in {"", ".", ".."} for part in pure.parts)
            or path <= previous
        ):
            _fail(FAIL_GIT_BINDING, "source-file path/order is not canonical")
        previous = path
        if type(mode) is not str or mode not in {"100644", "100755"}:
            _fail(FAIL_GIT_BINDING, f"invalid source-file mode: {path}")
        if type(oid) is not str or re.fullmatch(r"[0-9a-f]{40}", oid) is None:
            _fail(FAIL_GIT_BINDING, f"invalid source-file Git blob ID: {path}")
        if type(file_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", file_hash) is None:
            _fail(FAIL_GIT_BINDING, f"invalid source-file hash: {path}")
        if type(size) is not int or size < 0 or size >= 1 << 64:
            _fail(FAIL_GIT_BINDING, f"invalid source-file size: {path}")
        try:
            fields = (
                path.encode("utf-8"),
                mode.encode("ascii"),
                bytes.fromhex(oid),
                bytes.fromhex(file_hash),
                size.to_bytes(8, "big"),
            )
        except (ValueError, OverflowError) as error:
            _fail(FAIL_GIT_BINDING, f"invalid source-file row {path}: {error}")
        if len(fields[2]) != 20 or len(fields[3]) != 32:
            _fail(FAIL_GIT_BINDING, f"invalid source-file digest width: {path}")
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def cargo_seed_manifest_root_v1(
    rows: Sequence[Mapping[str, object]], *, code: str = FAIL_GUARD
) -> str:
    """Hash the exact ordered regular-file transport mounted for offline Cargo."""

    digest = sha256()
    digest.update(CARGO_SEED_MANIFEST_DOMAIN + b"\x00")
    previous = ""
    for index, row_value in enumerate(rows):
        if type(row_value) is not dict or set(row_value) != CARGO_SEED_FILE_FIELDS:
            _fail(code, f"Cargo seed file row {index} fields differ")
        row = row_value
        path = row.get("path")
        mode = row.get("mode")
        size = row.get("size")
        file_hash = row.get("sha256")
        if type(path) is not str:
            _fail(code, f"Cargo seed file row {index} path is not text")
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or pure.as_posix() != path
            or not pure.parts
            or pure.parts[0] not in CARGO_SEED_SUBTREES
            or any(part in {"", ".", ".."} for part in pure.parts)
            or path <= previous
        ):
            _fail(code, f"Cargo seed file rows are not canonical: {path!r}")
        previous = path
        if type(mode) is not str or re.fullmatch(r"[0-7]{4}", mode) is None:
            _fail(code, f"Cargo seed file mode differs: {path}")
        if type(size) is not int or size < 0 or size >= 1 << 64:
            _fail(code, f"Cargo seed file size differs: {path}")
        if type(file_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", file_hash) is None:
            _fail(code, f"Cargo seed file hash differs: {path}")
        fields = (
            path.encode("utf-8"),
            mode.encode("ascii"),
            size.to_bytes(8, "big"),
            bytes.fromhex(file_hash),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _validate_cargo_seed_receipt(
    receipt_value: object, *, code: str = FAIL_GUARD
) -> dict[str, object]:
    if type(receipt_value) is not dict or set(receipt_value) != CARGO_SEED_RECEIPT_FIELDS:
        _fail(code, "Cargo seed manifest receipt fields differ")
    receipt = receipt_value
    files = receipt.get("files")
    if type(files) is not list or not files:
        _fail(code, "Cargo seed manifest file rows differ")
    manifest_root = cargo_seed_manifest_root_v1(files, code=code)
    total = sum(int(row["size"]) for row in files)
    if (
        receipt.get("schema_version") != "hegel-shrink6-cargo-seed-manifest/1"
        or receipt.get("hash_domain") != CARGO_SEED_MANIFEST_DOMAIN.decode("ascii")
        or not _json_exact_equal(
            receipt.get("subtrees"), list(CARGO_SEED_SUBTREES)
        )
        or type(receipt.get("file_count")) is not int
        or receipt.get("file_count") != len(files)
        or type(receipt.get("total_byte_count")) is not int
        or receipt.get("total_byte_count") != total
        or receipt.get("manifest_root") != manifest_root
    ):
        _fail(code, "Cargo seed manifest receipt differs")
    return receipt


def _cargo_seed_receipt(cargo_registry: Path) -> dict[str, object]:
    """Read and bind every regular file copied into the fresh Cargo home."""

    rows: list[dict[str, object]] = []
    try:
        for subtree in CARGO_SEED_SUBTREES:
            root = cargo_registry / subtree
            root_stat = root.lstat()
            if not stat.S_ISDIR(root_stat.st_mode):
                _fail(FAIL_BUILD, f"Cargo seed subtree is not a real directory: {root}")
            for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
                before = path.lstat()
                if stat.S_ISDIR(before.st_mode):
                    continue
                if not stat.S_ISREG(before.st_mode):
                    _fail(FAIL_BUILD, f"Cargo seed contains a non-regular entry: {path}")
                payload = path.read_bytes()
                after = path.lstat()
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_size,
                    before.st_mtime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if before_identity != after_identity or len(payload) != before.st_size:
                    _fail(FAIL_BUILD, f"Cargo seed changed while hashing: {path}")
                rows.append(
                    {
                        "path": path.relative_to(cargo_registry).as_posix(),
                        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
                        "size": len(payload),
                        "sha256": sha256(payload).hexdigest(),
                    }
                )
    except OSError as error:
        _fail(FAIL_BUILD, f"Cargo seed manifest is unreadable: {error}")
    rows.sort(key=lambda row: str(row["path"]))
    receipt: dict[str, object] = {
        "schema_version": "hegel-shrink6-cargo-seed-manifest/1",
        "hash_domain": CARGO_SEED_MANIFEST_DOMAIN.decode("ascii"),
        "subtrees": list(CARGO_SEED_SUBTREES),
        "file_count": len(rows),
        "total_byte_count": sum(int(row["size"]) for row in rows),
        "files": rows,
        "manifest_root": cargo_seed_manifest_root_v1(rows, code=FAIL_BUILD),
    }
    return _validate_cargo_seed_receipt(receipt, code=FAIL_BUILD)


def _assert_cargo_seed_unchanged(
    cargo_registry: Path, expected_receipt: Mapping[str, object]
) -> None:
    observed = _cargo_seed_receipt(cargo_registry)
    if not _json_exact_equal(observed, expected_receipt):
        _fail(FAIL_BUILD, "Cargo seed bytes changed after manifest commitment")


def _source_rows(basis_commit: str) -> tuple[list[dict[str, object]], str]:
    listing = _git("ls-tree", "-r", basis_commit, "--", *ARCHIVE_PATHS)
    if type(listing) is not str:
        _fail(FAIL_GIT_BINDING, "Git tree listing is not text")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in listing.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        if kind != "blob" or path in seen:
            _fail(FAIL_GIT_BINDING, f"invalid source tree row for {path}")
        payload = _git("show", f"{basis_commit}:{path}", binary=True)
        if type(payload) is not bytes:
            _fail(FAIL_GIT_BINDING, f"Git blob is not bytes: {path}")
        worktree_path = REPOSITORY_ROOT / path
        if not worktree_path.is_file() or worktree_path.read_bytes() != payload:
            _fail(FAIL_GIT_BINDING, f"worktree source differs from {basis_commit}: {path}")
        if _git_blob_oid(payload) != oid:
            _fail(FAIL_GIT_BINDING, f"Git blob identity mismatch: {path}")
        seen.add(path)
        rows.append(
            {
                "path": path,
                "mode": mode,
                "git_blob_oid": oid,
                "sha256": sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    required_files = set(PYTHON_SOURCE_PATHS) | {
        SUPERVISOR_PATH,
        FREEZE_DOC_PATH,
        PROTOCOL_DOC_PATH,
        SUPERVISOR_TEST_PATH,
        PARENT_EVIDENCE_PATH,
        PROFILE_PATH,
        BUILD_PROFILE_PATH,
        RUNTIME_SECCOMP_PATH,
        BUILD_SECCOMP_PATH,
    }
    if not required_files.issubset(seen):
        _fail(FAIL_GIT_BINDING, f"missing required source blobs: {sorted(required_files - seen)}")
    rows.sort(key=lambda row: str(row["path"]))
    return rows, source_file_set_root_v1(rows)


def _parent_evidence_binding(basis_commit: str) -> dict[str, object]:
    """Bind the sole engineering admission authority without copying outputs."""

    _git("merge-base", "--is-ancestor", PARENT_EVIDENCE_COMMIT, basis_commit)
    frozen = _git(
        "show", f"{PARENT_EVIDENCE_COMMIT}:{PARENT_EVIDENCE_PATH}", binary=True
    )
    current = _git("show", f"{basis_commit}:{PARENT_EVIDENCE_PATH}", binary=True)
    if (
        type(frozen) is not bytes
        or current != frozen
        or sha256(frozen).hexdigest() != PARENT_EVIDENCE_SHA256
    ):
        _fail(FAIL_GIT_BINDING, "parent Evidence V bytes differ from the frozen commit")
    try:
        evidence = json.loads(
            frozen,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        _fail(FAIL_GIT_BINDING, f"parent Evidence V is not valid JSON: {error}")
    if type(evidence) is not dict:
        _fail(FAIL_GIT_BINDING, "parent Evidence V is not an object")
    if (
        evidence.get("evidence_record_id") != PARENT_EVIDENCE_RECORD_ID
        or evidence.get("status") != "DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS"
        or evidence.get("claim_level") != "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
        or evidence.get("dsl_version") != "hegel-old-dsl-v1.5.0"
        or evidence.get("freeze_version") != "hegel-freeze-p2b-p3-v1.5.0"
        or evidence.get("execution_state") != "NOT_RUN"
        or evidence.get("formal_roots_generated") is not False
        or evidence.get("formal_roots") is not None
        or evidence.get("formal_state_transition_allowed") is not False
        or not _json_exact_equal(evidence.get("routing"), {
            "authority": "ENGINEERING_ONLY",
            "formal_status_promotion_allowed": False,
            "from_max_total_ast_depth": 4,
            "maximum_ast_node_count_remains": 6,
            "maximum_top_level_clauses_remains": 2,
            "only_open_route": True,
            "operation": "reduce max_total_ast_depth from 4 to 3",
            "preregistered_shrink_order_step": 6,
            "to_max_total_ast_depth": 3,
        })
    ):
        _fail(FAIL_GIT_BINDING, "parent Evidence V authority fields differ")
    return {
        "evidence_commit": PARENT_EVIDENCE_COMMIT,
        "evidence_path": PARENT_EVIDENCE_PATH,
        "evidence_record_id": PARENT_EVIDENCE_RECORD_ID,
        "evidence_sha256": "sha256:" + sha256(frozen).hexdigest(),
        "admitted_operation": "reduce max_total_ast_depth from 4 to 3",
        "formal_status_promotion_allowed": False,
    }


def _safe_extract_git_archive(payload: bytes, destination: Path) -> None:
    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    _fail(FAIL_ARCHIVE, f"unsafe archive path: {member.name}")
                if not (member.isdir() or member.isfile()):
                    _fail(FAIL_ARCHIVE, f"non-regular archive member: {member.name}")
            archive.extractall(destination)
    except (tarfile.TarError, OSError) as error:
        _fail(FAIL_ARCHIVE, f"archive extraction failed: {error}")


def _validate_snapshot(
    snapshot_root: Path, source_rows: Sequence[Mapping[str, object]]
) -> None:
    for row in source_rows:
        path = snapshot_root / str(row["path"])
        try:
            payload = path.read_bytes()
        except OSError as error:
            _fail(FAIL_ARCHIVE, f"snapshot source unreadable: {path}: {error}")
        if (
            len(payload) != row["size"]
            or sha256(payload).hexdigest() != row["sha256"]
            or _git_blob_oid(payload) != row["git_blob_oid"]
        ):
            _fail(FAIL_ARCHIVE, f"snapshot bytes differ from Git blob: {row['path']}")


def _docker_server_receipt(server: Mapping[str, object]) -> dict[str, object]:
    selected = {
        "version": server.get("Version"),
        "api_version": server.get("ApiVersion"),
        "minimum_api_version": server.get("MinAPIVersion"),
        "os": server.get("Os"),
        "architecture": server.get("Arch"),
        "git_commit": server.get("GitCommit"),
        "go_version": server.get("GoVersion"),
        "kernel_version": server.get("KernelVersion"),
        "build_time": server.get("BuildTime"),
        "raw_canonical_sha256": sha256(_canonical_json_bytes(server)).hexdigest(),
    }
    if set(selected) != DOCKER_SERVER_RECEIPT_FIELDS or any(
        type(value) is not str or not value
        for key, value in selected.items()
        if key != "raw_canonical_sha256"
    ):
        _fail(FAIL_DOCKER_POLICY, "Docker server identity fields differ")
    return selected


def _initialize_docker_environment(control_root: Path) -> dict[str, object]:
    global _DOCKER_ENV
    if Path(DOCKER_EXECUTABLE).resolve() != Path("/usr/bin/docker") or not os.access(
        DOCKER_EXECUTABLE, os.X_OK
    ):
        _fail(FAIL_DOCKER_POLICY, "the exact /usr/bin/docker executable is unavailable")
    try:
        socket_stat = DOCKER_SOCKET.stat()
    except OSError as error:
        _fail(FAIL_DOCKER_POLICY, f"Docker socket is unavailable: {error}")
    if not stat.S_ISSOCK(socket_stat.st_mode):
        _fail(FAIL_DOCKER_POLICY, "Docker control endpoint is not a Unix socket")
    config = control_root / "docker-config"
    home = control_root / "home"
    config.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    config_file = config / "config.json"
    config_file.write_bytes(b"{}\n")
    config_file.chmod(0o600)
    _DOCKER_ENV = {
        "DOCKER_CONFIG": str(config),
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    server_raw = _one_json(
        _run(
            (*_docker_prefix(), "version", "--format", "{{json .Server}}"),
            code=FAIL_DOCKER_POLICY,
        ),
        "Docker server version",
    )
    server = _docker_server_receipt(server_raw)
    info_template = (
        "{\"id\":{{json .ID}},\"name\":{{json .Name}},"
        "\"driver\":{{json .Driver}},\"operating_system\":{{json .OperatingSystem}},"
        "\"os_type\":{{json .OSType}},\"architecture\":{{json .Architecture}},"
        "\"docker_root_dir\":{{json .DockerRootDir}},"
        "\"security_options\":{{json .SecurityOptions}}}"
    )
    daemon = _one_json(
        _run(
            (*_docker_prefix(), "info", "--format", info_template),
            code=FAIL_DOCKER_POLICY,
        ),
        "Docker daemon identity",
    )
    if (
        daemon.get("os_type") != "linux"
        or type(daemon.get("id")) is not str
        or not daemon["id"]
        or type(daemon.get("driver")) is not str
        or not daemon["driver"]
    ):
        _fail(FAIL_DOCKER_POLICY, "live Docker daemon is not the required local Linux runtime")
    receipt: dict[str, object] = {
        "schema_version": "hegel-shrink6-local-docker-daemon-receipt/1",
        "docker_executable": DOCKER_EXECUTABLE,
        "explicit_host_argument": DOCKER_HOST_ARGUMENT,
        "socket": str(DOCKER_SOCKET),
        "socket_device": socket_stat.st_dev,
        "socket_inode": socket_stat.st_ino,
        "socket_uid": socket_stat.st_uid,
        "socket_gid": socket_stat.st_gid,
        "private_empty_client_config_sha256": sha256(b"{}\n").hexdigest(),
        "host_environment_keys": sorted(_DOCKER_ENV),
        "server": server,
        "daemon": daemon,
    }
    receipt["diagnostic_receipt_hash"] = "sha256:" + sha256(
        DAEMON_RECEIPT_DOMAIN + b"\x00" + _canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def _profile_images(snapshot_project: Path) -> dict[str, object]:
    try:
        actor_payload = (
            snapshot_project / "config/phase3_container_actor_profile_v1.json"
        ).read_bytes()
        build_payload = (
            snapshot_project / "config/phase3_shrink6_offline_build_profile_v1.json"
        ).read_bytes()
        profile = json.loads(
            actor_payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
        build_profile = json.loads(
            build_payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
        images = profile["images"]
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError) as error:
        _fail(FAIL_DOCKER_POLICY, f"container profile unreadable: {error}")
    if type(profile) is not dict or type(build_profile) is not dict:
        _fail(FAIL_DOCKER_POLICY, "container/build profile is not an object")
    if type(images) is not dict:
        _fail(FAIL_DOCKER_POLICY, "container profile images field is not an object")
    if images.get("python_attester") != PYTHON_IMAGE or images.get("rust_attester") != RUST_IMAGE:
        _fail(FAIL_DOCKER_POLICY, "pinned image references differ from the committed profile")
    if not _json_exact_equal(profile.get("network_policy"), {
        "allow_registry_access": False,
        "allow_runtime_network": False,
        "docker_network": "none",
        "pull_policy": "never",
    }):
        _fail(FAIL_DOCKER_POLICY, "committed network policy differs")
    control = profile.get("docker_control_plane_policy")
    if type(control) is not dict or (
        control.get("executable") != DOCKER_EXECUTABLE
        or control.get("explicit_host_argument") != DOCKER_HOST_ARGUMENT
        or control.get("socket") != str(DOCKER_SOCKET)
        or control.get("client_config") != "empty-private-config-json"
        or control.get("host_environment_keys_exact") != sorted(
            ["DOCKER_CONFIG", "DOCKER_HOST", "HOME", "LANG", "LC_ALL", "PATH"]
        )
        or control.get("ambient_proxy_or_docker_variables_allowed") is not False
        or control.get("live_local_linux_daemon_identity_receipt_required") is not True
    ):
        _fail(FAIL_DOCKER_POLICY, "committed Docker control-plane policy differs")
    if not _json_exact_equal(profile.get("resource_limits"), {
        "memory": "512m",
        "pids": 64,
        "tmpfs": RUNTIME_TMPFS,
    }):
        _fail(FAIL_DOCKER_POLICY, "committed actor resource limits differ")
    if not _json_exact_equal(profile.get("required_runtime_flags"), [
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
        "--pids-limit=64",
    ]):
        _fail(FAIL_DOCKER_POLICY, "committed required runtime flags differ")
    if (
        profile.get("seccomp_profile")
        != "config/phase3_internal_actor_seccomp_v1.json"
        or profile.get("offline_build_seccomp_profile")
        != "config/phase3_m3_offline_build_seccomp_v1.json"
    ):
        _fail(FAIL_DOCKER_POLICY, "committed seccomp profile paths differ")
    disclosure = profile.get("authority_disclosure")
    if type(disclosure) is not dict or (
        disclosure.get("same_admin_controller") is not True
        or disclosure.get("organizational_independence") is not False
        or disclosure.get("independent_human_actors") is not False
        or disclosure.get("technical_role_independence") is not True
        or disclosure.get("owner_accepted_threat_model") is not True
    ):
        _fail(FAIL_DOCKER_POLICY, "committed authority disclosure differs")
    if not _json_exact_equal(build_profile, {
        "profile_id": "hegel-shrink6-rust-offline-build-v1",
        "purpose": "build-only target-free shrink6 strict recognizer",
        "authority": (
            "engineering qualification only; no closure result, formal root, "
            "or state transition"
        ),
        "image": RUST_IMAGE,
        "network": "none",
        "pull_policy": "never",
        "root_filesystem_read_only": True,
        "user": "0:0",
        "capabilities_dropped": "ALL",
        "no_new_privileges": True,
        "pids_limit": 64,
        "memory": "512m",
        "memory_swap": "512m",
        "nofile_ulimit": "128:128",
        "tmpfs": BUILD_TMPFS,
        "seccomp_profile": "config/phase3_m3_offline_build_seccomp_v1.json",
        "source_mount": "read-only committed git archive snapshot",
        "cargo_registry_mount": "read-only cache and index subtrees only; no pre-unpacked src",
        "cargo_seed_manifest": (
            "exact regular-file rows with mode, size and sha256 under a "
            "domain-separated root; checked before and after build"
        ),
        "cargo_home": "fresh build-container tmpfs",
        "target_mount": "fresh local-driver docker volume, build rw then runtime ro",
        "cargo_flags": ["--release", "--locked", "--offline"],
        "target_or_split_inputs_allowed": False,
        "seed_key_signature_or_formal_root_access_allowed": False,
    }):
        _fail(FAIL_DOCKER_POLICY, "committed shrink-6 build profile differs")
    return {
        "actor_profile_id": profile.get("profile_id"),
        "actor_profile_sha256": sha256(actor_payload).hexdigest(),
        "build_profile_id": build_profile["profile_id"],
        "build_profile_sha256": sha256(build_payload).hexdigest(),
    }


def _inspect_image(image: str) -> str:
    result = _run(
        (*_docker_prefix(), "image", "inspect", image, "--format", "{{.Id}}"),
        code=FAIL_DOCKER_POLICY,
    )
    observed = result.stdout.decode("ascii", "strict").strip()
    expected = image.split("@", 1)[1]
    if observed != expected:
        _fail(FAIL_DOCKER_POLICY, f"local image identity differs for {image}")
    return observed


def _docker_common(
    image: str,
    *,
    seccomp_path: Path,
    user: str | None = None,
    tmpfs: str = RUNTIME_TMPFS,
) -> list[str]:
    command = [
        *_docker_prefix(),
        "run", "--rm", "--name", f"hegel-s6-{secrets.token_hex(8)}",
        "--pull=never", "--network=none", "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--security-opt", f"seccomp={seccomp_path}",
        "--read-only", "--pids-limit=64", "--memory=512m",
        "--memory-swap=512m", "--ulimit=nofile=128:128",
        "--tmpfs", tmpfs,
        "-e", "HOME=/tmp", "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8",
        "-e", "TZ=UTC",
    ]
    if user is not None:
        command.extend(("--user", user))
    command.append(image)
    return command


def python_runtime_command(snapshot_project: Path, arguments: Sequence[str]) -> list[str]:
    command = _docker_common(
        PYTHON_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = [
        "-e", "PYTHONHASHSEED=0",
        "-v", f"{snapshot_project}:/workspace:ro",
        "-w", "/workspace",
    ]
    command.extend(
        (
            "/usr/local/bin/python3", "-I", "-S", "-B",
            "/workspace/src/hegel_machine/phase3_shrink6_strict_entrypoint_v1.py",
            *arguments,
        )
    )
    return command


def python_capacity_command(snapshot_project: Path, mode: str) -> list[str]:
    command = _docker_common(
        PYTHON_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = [
        "-e", "PYTHONHASHSEED=0",
        "-v", f"{snapshot_project}:/workspace:ro",
        "-w", "/workspace",
    ]
    command.extend(
        (
            "/usr/local/bin/python3", "-I", "-S", "-B",
            "/workspace/src/hegel_machine/phase3_shrink6_capacity_entrypoint_v1.py",
            mode,
        )
    )
    return command


def rust_runtime_command(
    snapshot_project: Path, volume: str, arguments: Sequence[str]
) -> list[str]:
    command = _docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command[-1:-1] = ["-v", f"{volume}:/cargo-target:ro"]
    command.extend(("/cargo-target/release/hegel-strict-canonicalizer-shrink6", *arguments))
    return command


def _create_fresh_volume(name: str, basis_commit: str) -> dict[str, object]:
    inspected = _run(
        (*_docker_prefix(), "volume", "inspect", name),
        allowed_codes=frozenset({0, 1}),
        code=FAIL_DOCKER_POLICY,
    )
    if inspected.returncode == 0:
        _fail(FAIL_DOCKER_POLICY, f"dedicated target volume already exists: {name}")
    if inspected.returncode != 1:
        _fail(FAIL_DOCKER_POLICY, "unable to establish target-volume absence")
    created = _run(
        (
            *_docker_prefix(), "volume", "create", "--driver", "local",
            "--label", "hegel.machine.role=shrink6-dual-strict",
            "--label", f"hegel.machine.basis={basis_commit}",
            "--label", "hegel.machine.network=none",
            name,
        ),
        code=FAIL_DOCKER_POLICY,
    ).stdout.decode("utf-8").strip()
    if created != name:
        cleanup = _run(
            (*_docker_prefix(), "volume", "rm", name),
            allowed_codes=frozenset({0, 1}),
            code=FAIL_CLEANUP,
        )
        _fail(
            FAIL_DOCKER_POLICY,
            "Docker returned a different target volume name; "
            f"cleanup_exit={cleanup.returncode}",
        )
    expected_labels = {
        "hegel.machine.role": "shrink6-dual-strict",
        "hegel.machine.basis": basis_commit,
        "hegel.machine.network": "none",
    }
    try:
        detail = _one_json(
            _run(
                (*_docker_prefix(), "volume", "inspect", name, "--format", "{{json .}}"),
                code=FAIL_DOCKER_POLICY,
            ),
            "Docker target volume",
        )
        if (
            detail.get("Name") != name
            or detail.get("Driver") != "local"
            or detail.get("Scope") != "local"
            or detail.get("Options") not in (None, {})
            or detail.get("Labels") != expected_labels
            or type(detail.get("Mountpoint")) is not str
            or not str(detail["Mountpoint"]).startswith("/")
        ):
            _fail(FAIL_DOCKER_POLICY, "fresh target volume is not exact local storage")
    except BaseException as primary:
        try:
            _remove_volume(name)
        except QualificationError as cleanup_error:
            if isinstance(primary, QualificationError):
                raise QualificationError(
                    primary.code,
                    f"{primary.detail}; secondary cleanup failure: "
                    f"{cleanup_error.code}: {cleanup_error.detail}",
                ) from primary
            raise
        raise
    return {
        "name": name,
        "driver": "local",
        "scope": "local",
        "options": None,
        "labels": expected_labels,
        "fresh_before_run": True,
    }


def _remove_volume(name: str) -> None:
    result = _run(
        (*_docker_prefix(), "volume", "rm", name),
        timeout=120,
        code=FAIL_CLEANUP,
    )
    if result.stdout.decode("utf-8").strip() != name:
        _fail(FAIL_CLEANUP, "Docker did not confirm target-volume removal")
    absent = _run(
        (*_docker_prefix(), "volume", "inspect", name),
        allowed_codes=frozenset({1}),
        code=FAIL_CLEANUP,
    )
    if absent.returncode != 1:
        _fail(FAIL_CLEANUP, "target volume still exists after removal")


def _build_rust(
    snapshot_project: Path,
    cargo_registry: Path,
    volume: str,
    workers: int,
    cargo_seed_receipt: Mapping[str, object],
) -> str:
    if not cargo_registry.is_dir():
        _fail(FAIL_BUILD, f"offline Cargo registry is absent: {cargo_registry}")
    cargo_cache = cargo_registry / "cache"
    cargo_index = cargo_registry / "index"
    if not cargo_cache.is_dir() or not cargo_index.is_dir():
        _fail(FAIL_BUILD, "offline Cargo cache/index seed is incomplete")
    _assert_cargo_seed_unchanged(cargo_registry, cargo_seed_receipt)
    command = _docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_m3_offline_build_seccomp_v1.json",
        tmpfs=BUILD_TMPFS,
    )
    command[-1:-1] = [
        "-e", "CARGO_HOME=/tmp/cargo-home",
        "-e", "CARGO_NET_OFFLINE=true",
        "-e", f"CARGO_BUILD_JOBS={workers}",
        "-e", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1",
        "-e", "CARGO_TARGET_DIR=/cargo-target",
        "-e", f"RUSTC={RUST_TOOLCHAIN_BIN}/rustc",
        "-e", f"RUSTDOC={RUST_TOOLCHAIN_BIN}/rustdoc",
        "-v", f"{cargo_cache}:/cargo-seed/cache:ro",
        "-v", f"{cargo_index}:/cargo-seed/index:ro",
        "-v", f"{snapshot_project / 'rust'}:/workspace/rust:ro",
        "-v", f"{volume}:/cargo-target:rw",
        "-w", "/workspace/rust/strict_canonicalizer_shrink6",
    ]
    command.extend(
        (
            "/bin/sh",
            "-euc",
            "mkdir -p /tmp/cargo-home/registry; "
            "cp -a --no-preserve=ownership /cargo-seed/cache "
            "/tmp/cargo-home/registry/cache; "
            "cp -a --no-preserve=ownership /cargo-seed/index "
            "/tmp/cargo-home/registry/index; "
            f"exec {RUST_TOOLCHAIN_BIN}/cargo build --release --locked --offline",
        )
    )
    _run(command, timeout=900, code=FAIL_BUILD)
    _assert_cargo_seed_unchanged(cargo_registry, cargo_seed_receipt)
    hash_result = _run(
        rust_runtime_command(snapshot_project, volume, ()),
        timeout=60,
        allowed_codes=frozenset({2}),
        code=FAIL_BUILD,
    )
    # The no-argument run proves the binary is executable but intentionally
    # returns its CLI error.  Hash it in a separate immutable-container call.
    hash_command = _docker_common(
        RUST_IMAGE,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    hash_command[-1:-1] = ["-v", f"{volume}:/cargo-target:ro"]
    hash_command.extend(("/usr/bin/sha256sum", "/cargo-target/release/hegel-strict-canonicalizer-shrink6"))
    digest = _run(hash_command, code=FAIL_BUILD).stdout.decode("ascii").split()[0]
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None or not hash_result.stderr:
        _fail(FAIL_BUILD, "Rust binary executable/hash qualification failed")
    return digest


def _accepted_metadata_guard(
    report: Mapping[str, object], *, code: str, label: str
) -> None:
    cbor_hex = report.get("canonical_cbor_hex")
    if (
        type(cbor_hex) is not str
        or not cbor_hex
        or len(cbor_hex) % 2 != 0
        or re.fullmatch(r"[0-9a-f]+", cbor_hex) is None
        or type(report.get("canonical_ast_hash")) is not str
        or type(report.get("root_operator_id")) is not int
        or type(report.get("output_sort")) is not str
        or type(report.get("depth")) is not int
        or not 0 <= int(report["depth"]) <= 3
        or type(report.get("node_count")) is not int
        or not 1 <= int(report["node_count"]) <= 6
        or not _json_exact_equal(report.get("maximum_ast_depth"), 3)
        or not _json_exact_equal(report.get("maximum_ast_node_count"), 6)
        or not _json_exact_equal(report.get("maximum_top_level_clauses"), 2)
    ):
        _fail(code, f"{label} accepted metadata is outside the shrink-6 limits")


def normalize_endpoint_report(
    report: Mapping[str, object], *, implementation: str
) -> tuple[bytes, dict[str, object]]:
    if report.get("schema_version") != "hegel-strict-canonicalizer-shrink6-replay/1":
        _fail(FAIL_VECTOR, f"{implementation} replay schema differs")
    if report.get("implementation") != implementation:
        _fail(FAIL_VECTOR, f"{implementation} identity differs")
    if report.get("dsl_version") != "hegel-old-dsl-v1.6.0" or report.get("freeze_version") != "hegel-freeze-p2b-p3-v1.6.0":
        _fail(FAIL_VECTOR, f"{implementation} DSL/freeze binding differs")
    status = report.get("status")
    if status == "REJECTED":
        error_code = report.get("error_code")
        if type(error_code) is not str or re.fullmatch(r"[A-Z0-9_]+", error_code) is None:
            _fail(FAIL_VECTOR, f"{implementation} rejection code is invalid")
        return rejected_outcome_bytes(error_code), {
            "status": "REJECTED", "error_code": error_code
        }
    if status != "ACCEPTED":
        _fail(FAIL_VECTOR, f"{implementation} disposition is invalid")
    normalized = {field: report.get(field) for field in ACCEPT_COMPARISON_FIELDS}
    if any(normalized[field] is None for field in ACCEPT_COMPARISON_FIELDS):
        _fail(FAIL_VECTOR, f"{implementation} accepted metadata is incomplete")
    _accepted_metadata_guard(
        normalized,
        code=FAIL_VECTOR,
        label=implementation,
    )
    try:
        cbor_bytes = bytes.fromhex(str(normalized["canonical_cbor_hex"]))
    except ValueError as error:
        _fail(FAIL_VECTOR, f"{implementation} accepted payload is invalid: {error}")
    hash_id = str(normalized["canonical_ast_hash"])
    computed = sha256(AST_HASH_DOMAIN + b"\x00" + cbor_bytes).digest()
    if hash_id != "sha256:" + computed.hex():
        _fail(FAIL_VECTOR, f"{implementation} AST hash does not bind its CBOR")
    return accepted_outcome_bytes(cbor_bytes, computed), normalized


def _strict_report_guard(
    report: Mapping[str, object], *, implementation: str, boundary: str
) -> None:
    status = report.get("status")
    if implementation == "python":
        expected = (
            PYTHON_STRICT_ACCEPT_FIELDS
            if status == "ACCEPTED"
            else PYTHON_STRICT_REJECT_FIELDS
        )
    elif implementation == "rust":
        expected = (
            RUST_STRICT_ACCEPT_FIELDS
            if status == "ACCEPTED"
            else RUST_STRICT_REJECT_FIELDS
        )
    else:
        _fail(FAIL_VECTOR, "unknown strict implementation")
    if implementation == "rust" and boundary == "FORMAL_CBOR":
        expected = expected | {"generic_cbor_parse"}
    if status not in {"ACCEPTED", "REJECTED"} or set(report) != expected:
        _fail(
            FAIL_VECTOR,
            f"{implementation} {boundary} strict report fields differ: "
            f"expected={sorted(expected)}, observed={sorted(report)}",
        )
    if (
        not _json_exact_equal(report.get("maximum_ast_depth"), 3)
        or not _json_exact_equal(report.get("maximum_ast_node_count"), 6)
        or not _json_exact_equal(report.get("maximum_top_level_clauses"), 2)
    ):
        _fail(
            FAIL_VECTOR,
            f"{implementation} did not bind maximum_ast_depth=3, "
            "maximum_ast_node_count=6, and maximum_top_level_clauses=2",
        )
    common = {
        "schema_version": "hegel-strict-canonicalizer-shrink6-replay/1",
        "implementation": implementation,
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.6.0",
        "boundary": boundary,
        "target_or_split_modules_loaded": False,
    }
    if any(
        not _json_exact_equal(report.get(field), value)
        for field, value in common.items()
    ):
        _fail(FAIL_VECTOR, f"{implementation} strict identity fields differ")
    if status == "ACCEPTED":
        _accepted_metadata_guard(
            report,
            code=FAIL_VECTOR,
            label=f"{implementation} {boundary}",
        )
        if implementation == "rust" and type(
            report.get("scalar_parameter_occurrence_count")
        ) is not int:
            _fail(FAIL_VECTOR, "Rust scalar occurrence count type differs")
    elif (
        type(report.get("error_code")) is not str
        or (
            implementation == "python"
            and type(report.get("error_detail")) is not str
        )
        or (
            implementation == "rust"
            and type(report.get("error_message")) is not str
        )
    ):
        _fail(FAIL_VECTOR, f"{implementation} rejected field types differ")
    if implementation == "python":
        if report.get("loaded_hegel_modules") != EXPECTED_PYTHON_STRICT_MODULES:
            _fail(FAIL_GUARD, "Python strict module closure differs")
    elif any(
        not _json_exact_equal(report.get(field), value)
        for field, value in {
            "parent_dsl_version": "hegel-old-dsl-v1.5.0",
            "parent_freeze_version": "hegel-freeze-p2b-p3-v1.5.0",
            "cbor_profile_id": "hegel-cbor-det-v1",
            "ast_schema_id": "hegel-canonical-ast-v1",
            "ast_hash_domain": "HEGEL/AST/V1",
        }.items()
    ):
        _fail(FAIL_VECTOR, "Rust strict parent/profile identity fields differ")


def _capacity_guard_values() -> dict[str, object]:
    """Return the complete frozen shrink-6 challenge-lattice contract."""

    return {
        "schema_version": "hegel-strict-capacity-replay-shrink6/1",
        "parent_dsl_version": "hegel-old-dsl-v1.5.0",
        "parent_freeze_version": "hegel-freeze-p2b-p3-v1.5.0",
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.6.0",
        "human_amendment_id": "hegel-freeze-p2b-p3-v1.6.0-shrink-step6",
        "shrink_step_id": "SHRINK_STEP_6_REDUCE_MAX_TOTAL_AST_DEPTH_4_TO_3",
        "generator_rule": CAPACITY_GENERATOR_RULE,
        "removed_binary_operator_ids": [0],
        "retained_difference_id": 1,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_top_level_clauses": 2,
        "constant_atom_count": 15,
        "rational_aggregate_count": 16,
        "mixed_atom_count": 144,
        "challenge_source_candidate_count": 1_266,
        "challenge_source_family_counts": {
            "A": 486,
            "B_abs": 390,
            "B_sign": 390,
        },
        "challenge_source_lattice_commitment": (
            EXPECTED_CHALLENGE_SOURCE_LATTICE_COMMITMENT
        ),
        "challenge_parent_accepted_count": 1_266,
        "challenge_parent_canonical_unique_count": 1_249,
        "challenge_parent_canonical_set_commitment": (
            EXPECTED_CHALLENGE_PARENT_CANONICAL_SET_COMMITMENT
        ),
        "inherited_survivor_source_count": 175,
        "inherited_survivor_unique_count": 175,
        "inherited_survivor_set_commitment": (
            EXPECTED_INHERITED_SURVIVOR_COMMITMENT
        ),
        "normalized_survivor_source_count": 67,
        "normalized_survivor_source_family_counts": {
            "A": 33,
            "B_abs": 17,
            "B_sign": 17,
        },
        "normalized_survivor_unique_count": 50,
        "normalized_survivor_set_commitment": (
            EXPECTED_NORMALIZED_SURVIVOR_COMMITMENT
        ),
        "survivor_source_candidate_count": 242,
        "survivor_accepted_count": 242,
        "survivor_unique_count": 225,
        "survivor_parent_identity_match_count": 242,
        "survivor_rejected_count": 0,
        "survivor_rejection_counts": {},
        "survivor_accepted_set_commitment": EXPECTED_SURVIVOR_COMMITMENT,
        "first_survivor_canonical_cbor_hex": EXPECTED_FIRST_SURVIVOR_CBOR_HEX,
        "first_survivor_canonical_ast_hash": EXPECTED_FIRST_SURVIVOR_HASH,
        "last_survivor_canonical_cbor_hex": EXPECTED_LAST_SURVIVOR_CBOR_HEX,
        "last_survivor_canonical_ast_hash": EXPECTED_LAST_SURVIVOR_HASH,
        "parent_only_source_candidate_count": 1_199,
        "parent_only_parent_accepted_count": 1_199,
        "parent_only_depth": 4,
        "parent_only_node_count": 6,
        "parent_only_unique_count": 1_199,
        "parent_only_source_family_counts": {
            "A": 453,
            "B_abs": 373,
            "B_sign": 373,
        },
        "parent_only_set_commitment": EXPECTED_PARENT_ONLY_SET_COMMITMENT,
        "parent_only_source_child_rejected_count": 1_199,
        "parent_only_source_child_rejection_counts": {
            "REJECT_STRUCTURAL_LIMIT": 1_199,
        },
        "parent_only_source_rejection_outcome_commitment": (
            EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT
        ),
        "parent_only_formal_child_rejected_count": 1_199,
        "parent_only_formal_child_rejection_counts": {
            "REJECT_STRUCTURAL_LIMIT": 1_199,
        },
        "parent_only_formal_rejection_outcome_commitment": (
            EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT
        ),
        "canonical_program_budget": 50_000,
        "first_out_of_budget_ordinal": None,
        "subset_status": "FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "target_or_split_modules_loaded": False,
    }


def compare_capacity_reports(
    python_report: Mapping[str, object], rust_report: Mapping[str, object]
) -> dict[str, object]:
    if python_report.get("implementation") != "python" or rust_report.get("implementation") != "rust":
        _fail(FAIL_CAPACITY, "capacity implementation identities differ")
    if set(python_report) != PYTHON_CAPACITY_FIELDS:
        _fail(FAIL_CAPACITY, "Python capacity report fields differ")
    if set(rust_report) != RUST_CAPACITY_FIELDS:
        _fail(FAIL_CAPACITY, "Rust capacity report fields differ")
    if python_report.get("loaded_hegel_modules") != EXPECTED_PYTHON_CAPACITY_MODULES:
        _fail(FAIL_CAPACITY, "Python capacity module closure differs")
    python_common = {
        key: value for key, value in python_report.items() if key not in CAPACITY_EXCLUDED_FIELDS
    }
    rust_common = {
        key: value for key, value in rust_report.items() if key not in CAPACITY_EXCLUDED_FIELDS
    }
    if not _json_exact_equal(python_common, rust_common):
        differing = sorted(
            key for key in set(python_common) | set(rust_common)
            if not _json_exact_equal(python_common.get(key), rust_common.get(key))
        )
        _fail(FAIL_CAPACITY, f"capacity fields differ: {differing}")
    for field, expected in _capacity_guard_values().items():
        if not _json_exact_equal(python_common.get(field), expected):
            _fail(FAIL_CAPACITY, f"capacity guard differs: {field}")
    return {
        "status": (
            "DUAL_SHRINK6_FROZEN_DEPTH4_CHALLENGE_LATTICE_REPLAY_PASS_"
            "NOT_COMPLETE"
        ),
        "all_comparable_fields_equal": True,
        **python_common,
        "comparable_report_sha256": sha256(_canonical_json_bytes(python_common)).hexdigest(),
    }


@dataclass(frozen=True)
class EndpointPair:
    vector_id: str
    python_exit: int
    python_report: dict[str, object]
    rust_exit: int
    rust_report: dict[str, object]


def _run_vector(snapshot_project: Path, volume: str, vector: object) -> EndpointPair:
    vector_id = str(vector.vector_id)
    if vector.boundary == "SOURCE_JSON":
        payload = vector.input_wire.decode("utf-8")
        python_args = ("--source-json", payload)
        rust_args = ("--ast-json", payload)
    else:
        payload = vector.input_wire.hex()
        python_args = ("--formal-cbor-hex", payload)
        rust_args = ("--decode-cbor-hex", payload)
    python_result = _run(
        python_runtime_command(snapshot_project, python_args),
        timeout=90,
        allowed_codes=frozenset({0}),
    )
    rust_result = _run(
        rust_runtime_command(snapshot_project, volume, rust_args),
        timeout=90,
        allowed_codes=frozenset({0, 1}),
    )
    python_report = _one_json(python_result, f"Python {vector_id}")
    rust_report = _one_json(rust_result, f"Rust {vector_id}")
    _strict_report_guard(
        python_report, implementation="python", boundary=vector.boundary
    )
    _strict_report_guard(
        rust_report, implementation="rust", boundary=vector.boundary
    )
    if python_report.get("boundary") != vector.boundary:
        _fail(FAIL_VECTOR, f"Python boundary differs at {vector_id}")
    loaded_modules = python_report.get("loaded_hegel_modules")
    if (
        loaded_modules != EXPECTED_PYTHON_STRICT_MODULES
        or python_report.get("target_or_split_modules_loaded") is not False
    ):
        _fail(FAIL_GUARD, f"Python target/split isolation differs at {vector_id}")
    if (
        rust_report.get("boundary") != vector.boundary
        or rust_report.get("target_or_split_modules_loaded") is not False
    ):
        _fail(FAIL_GUARD, f"Rust boundary/target isolation differs at {vector_id}")
    if vector.boundary == "FORMAL_CBOR" and rust_report.get("generic_cbor_parse") is not True:
        _fail(FAIL_VECTOR, f"Rust generic CBOR parse differs at {vector_id}")
    return EndpointPair(
        vector_id,
        python_result.returncode,
        python_report,
        rust_result.returncode,
        rust_report,
    )


def _golden_guard(report: Mapping[str, object], implementation: str) -> None:
    if implementation == "python":
        expected_fields = PYTHON_GOLDEN_FIELDS
    elif implementation == "rust":
        expected_fields = RUST_GOLDEN_FIELDS
    else:
        _fail(FAIL_GUARD, "unknown golden implementation")
    if set(report) != expected_fields:
        _fail(
            FAIL_GUARD,
            f"{implementation} built-in golden report fields differ: "
            f"expected={sorted(expected_fields)}, observed={sorted(report)}",
        )
    expected = {
        "schema_version": "hegel-strict-canonicalizer-shrink6-golden/1",
        "implementation": implementation,
        "parent_dsl_version": "hegel-old-dsl-v1.5.0",
        "parent_freeze_version": "hegel-freeze-p2b-p3-v1.5.0",
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.6.0",
        "human_amendment_id": "hegel-freeze-p2b-p3-v1.6.0-shrink-step6",
        "shrink_step_id": "SHRINK_STEP_6_REDUCE_MAX_TOTAL_AST_DEPTH_4_TO_3",
        "active_source_binary_operator_ids": [1, 2, 3, 4, 5, 6],
        "active_formal_binary_operator_ids": [1, 2, 3, 5, 6],
        "source_alias_binary_operator_ids": [4],
        "tombstoned_binary_operator_ids": [0],
        "reserved_binary_operator_ids": [7],
        "removed_binary_operator_error": "REJECT_REMOVED_BINARY_OPERATOR",
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_top_level_clauses": 2,
        "vector_count": EXPECTED_VECTOR_COUNT,
        "passed_count": EXPECTED_VECTOR_COUNT,
        "surviving_identity_checks": 3,
        "source_normalization_before_limit_checks": 2,
        "source_depth_limit_checks": 3,
        "source_priority_checks": 5,
        "formal_surviving_identity_checks": 1,
        "formal_depth_limit_checks": 3,
        "formal_priority_checks": 8,
        "execution_state": "NOT_RUN",
        "closure_executed": False,
        "formal_roots_generated": False,
        "formal_roots": None,
        "golden_vector_manifest_root": EXPECTED_MANIFEST_ROOT,
        "golden_outcome_root": EXPECTED_OUTCOME_ROOT,
        "ordered_vector_ids": list(EXPECTED_VECTOR_IDS),
        "target_or_split_modules_loaded": False,
    }
    for field, value in expected.items():
        if not _json_exact_equal(report.get(field), value):
            _fail(FAIL_GUARD, f"{implementation} built-in golden guard differs: {field}")
    if implementation == "python" and (
        report.get("loaded_hegel_modules") != EXPECTED_PYTHON_CAPACITY_MODULES
    ):
        _fail(FAIL_GUARD, "Python sealed built-in order/module receipt differs")


def _runtime_identity(
    snapshot_project: Path, image: str, program: Sequence[str]
) -> dict[str, object]:
    command = _docker_common(
        image,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command.extend(program)
    return _one_json(_run(command, code=FAIL_DOCKER_POLICY), f"{image} runtime identity")


def _runtime_text(snapshot_project: Path, image: str, program: Sequence[str]) -> str:
    command = _docker_common(
        image,
        seccomp_path=snapshot_project / "config/phase3_internal_actor_seccomp_v1.json",
        user="65534:65534",
    )
    command.extend(program)
    result = _run(command, code=FAIL_DOCKER_POLICY)
    try:
        text = result.stdout.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as error:
        _fail(FAIL_DOCKER_POLICY, f"{image} runtime identity is not UTF-8: {error}")
    if not text:
        _fail(FAIL_DOCKER_POLICY, f"{image} runtime identity is empty")
    return text


def _qualify(
    snapshot_project: Path,
    volume: str,
    *,
    repository_binding: dict[str, object],
    workers: int,
    cargo_registry: Path,
    python_image_id: str,
    rust_image_id: str,
    daemon_receipt: dict[str, object],
    volume_receipt: dict[str, object],
    profile_receipt: dict[str, object],
) -> dict[str, object]:
    cargo_seed_receipt = _cargo_seed_receipt(cargo_registry)
    binary_sha256 = _build_rust(
        snapshot_project,
        cargo_registry,
        volume,
        workers,
        cargo_seed_receipt,
    )

    python_identity = _runtime_identity(
        snapshot_project,
        PYTHON_IMAGE,
        (
            "/usr/local/bin/python3", "-I", "-S", "-B", "-c",
            "import hashlib,json,pathlib,sys;p=pathlib.Path(sys.executable).resolve();print(json.dumps({'executable':str(p),'executable_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'version':sys.version},sort_keys=True,separators=(',',':')))",
        ),
    )
    rust_identity = {
        "rustc_version_verbose": _runtime_text(
            snapshot_project,
            RUST_IMAGE,
            (f"{RUST_TOOLCHAIN_BIN}/rustc", "--version", "--verbose"),
        )
    }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(
            pool.map(
                lambda vector: _run_vector(snapshot_project, volume, vector),
                STRICT_GOLDEN_VECTORS_V1,
            )
        )
    pair_by_id = {pair.vector_id: pair for pair in pairs}
    if len(pair_by_id) != EXPECTED_VECTOR_COUNT:
        _fail(FAIL_VECTOR, "dual endpoint result IDs are not unique")

    python_outcomes: dict[str, bytes] = {}
    rust_outcomes: dict[str, bytes] = {}
    vector_rows: list[dict[str, object]] = []
    for vector in STRICT_GOLDEN_VECTORS_V1:
        pair = pair_by_id[vector.vector_id]
        python_bytes, python_normalized = normalize_endpoint_report(
            pair.python_report, implementation="python"
        )
        rust_bytes, rust_normalized = normalize_endpoint_report(
            pair.rust_report, implementation="rust"
        )
        expected = vector.expected_disposition
        observed = (
            ACCEPT_PARENT_IDENTITY
            if python_normalized["status"] == "ACCEPTED"
            else python_normalized["error_code"]
        )
        if (
            observed != expected
            or not _json_exact_equal(python_normalized, rust_normalized)
            or python_bytes != rust_bytes
        ):
            _fail(FAIL_VECTOR, f"dual normalized outcome differs at {vector.vector_id}")
        if pair.rust_exit != (0 if observed == ACCEPT_PARENT_IDENTITY else 1):
            _fail(FAIL_VECTOR, f"Rust exit/disposition differs at {vector.vector_id}")
        python_outcomes[vector.vector_id] = python_bytes
        rust_outcomes[vector.vector_id] = rust_bytes
        vector_rows.append(
            {
                "vector_id": vector.vector_id,
                "category": vector.category,
                "boundary": vector.boundary,
                "input_wire_sha256": sha256(vector.input_wire).hexdigest(),
                "input_wire_size": len(vector.input_wire),
                "expected_disposition": expected,
                "python_exit": pair.python_exit,
                "rust_exit": pair.rust_exit,
                "normalized": python_normalized,
                "normalized_outcome_sha256": sha256(python_bytes).hexdigest(),
                "dual_equal": True,
            }
        )

    python_root = strict_golden_outcome_root_v1(python_outcomes)
    rust_root = strict_golden_outcome_root_v1(rust_outcomes)
    manifest_root = strict_golden_manifest_root_v1()
    if manifest_root != EXPECTED_MANIFEST_ROOT or python_root != EXPECTED_OUTCOME_ROOT or rust_root != EXPECTED_OUTCOME_ROOT:
        _fail(FAIL_VECTOR, "sealed manifest/outcome root differs")

    python_golden_result = _run(
        python_capacity_command(snapshot_project, "--golden-replay"), timeout=180
    )
    rust_golden_result = _run(
        rust_runtime_command(snapshot_project, volume, ("--golden-replay",)),
        timeout=180,
    )
    python_golden = _one_json(python_golden_result, "Python built-in golden")
    rust_golden = _one_json(rust_golden_result, "Rust built-in golden")
    _golden_guard(python_golden, "python")
    _golden_guard(rust_golden, "rust")

    with ThreadPoolExecutor(max_workers=2) as pool:
        python_future = pool.submit(
            _run, python_capacity_command(snapshot_project, "--capacity-replay"), timeout=300
        )
        rust_future = pool.submit(
            _run,
            rust_runtime_command(snapshot_project, volume, ("--capacity-replay",)),
            timeout=300,
        )
        python_capacity = _one_json(python_future.result(), "Python capacity")
        rust_capacity = _one_json(rust_future.result(), "Rust capacity")
    capacity = compare_capacity_reports(python_capacity, rust_capacity)

    report: dict[str, object] = {
        "schema_version": SCHEMA,
        "artifact_kind": "COMMIT_BOUND_ENGINEERING_QUALIFICATION_EVIDENCE",
        "status": STATUS_PASS,
        "claim_level": CLAIM_LEVEL,
        "repository_binding": repository_binding,
        "sealed_basis": {
            "parent_evidence_binding": repository_binding[
                "parent_evidence_binding"
            ],
            "nonformal_diagnostic_roots": diagnostic_root_hex_v1(),
            "golden_vector_manifest_root": manifest_root,
            "expected_outcome_root": EXPECTED_OUTCOME_ROOT,
            "ordered_vector_ids": [vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1],
            "vector_count": EXPECTED_VECTOR_COUNT,
            "preregistered_formal_bucket_count": (
                PREREGISTERED_FORMAL_BUCKET_COUNT
            ),
        },
        "runtime_isolation": {
            "role_topology": "HOST_SUPERVISOR_PLUS_TWO_DISJOINT_PINNED_CONTAINERS",
            "same_admin_controller": True,
            "organizational_independence": False,
            "independent_human_actors": False,
            "technical_role_independence": True,
            "owner_accepted_threat_model": True,
            "docker_daemon_identity_receipt": daemon_receipt,
            "committed_profile_receipt": profile_receipt,
            "python_image_ref": PYTHON_IMAGE,
            "python_image_id": python_image_id,
            "rust_image_ref": RUST_IMAGE,
            "rust_image_id": rust_image_id,
            "pull_policy": "never",
            "network_mode": "none",
            "capabilities_dropped": "ALL",
            "no_new_privileges": True,
            "container_root_filesystem_read_only": True,
            "source_snapshot_mount_read_only": True,
            "fresh_ephemeral_rust_target_volume": True,
            "rust_target_volume_removed_after_run": True,
            "rust_target_volume_receipt": volume_receipt,
            "cargo_locked": True,
            "cargo_offline": True,
            "cargo_seed_subtrees": ["cache", "index"],
            "cargo_seed_manifest_receipt": cargo_seed_receipt,
            "preunpacked_registry_src_mounted": False,
            "cargo_home": "fresh-build-tmpfs",
            "rust_build_user": "0:0",
            "rust_build_tmpfs": BUILD_TMPFS,
            "recognizer_runtime_user": "65534:65534",
            "recognizer_runtime_tmpfs": RUNTIME_TMPFS,
            "memory_limit": "512m",
            "memory_swap_limit": "512m",
            "pids_limit": 64,
            "python_flags": ["-I", "-S", "-B"],
            "worker_count": workers,
            "python_runtime": python_identity,
            "rust_runtime": rust_identity,
            "rust_binary_sha256": binary_sha256,
        },
        "dual_vector_replay": {
            "status": STATUS_PASS,
            "vector_count": EXPECTED_VECTOR_COUNT,
            "python_outcome_root": python_root,
            "rust_outcome_root": rust_root,
            "all_normalized_outcomes_equal": True,
            "vectors": vector_rows,
        },
        "built_in_replay_controls": {
            "python_report_sha256": sha256(_canonical_json_bytes(python_golden)).hexdigest(),
            "rust_report_sha256": sha256(_canonical_json_bytes(rust_golden)).hexdigest(),
            "python_passed_count": EXPECTED_VECTOR_COUNT,
            "rust_passed_count": EXPECTED_VECTOR_COUNT,
            "python_golden_field_count": len(python_golden),
            "python_capacity_field_count": len(python_capacity),
            "python_combined_field_count": len(python_golden) + len(python_capacity),
            "rust_golden_field_count": len(rust_golden),
            "rust_capacity_field_count": len(rust_capacity),
            "rust_combined_field_count": len(rust_golden) + len(rust_capacity),
        },
        "dual_capacity_replay": capacity,
        "authority_guards": {
            "execution_state": "NOT_RUN",
            "closure_executed": False,
            "formal_roots_generated": False,
            "formal_roots": None,
            "certificate_issued": False,
            "signature_generated": False,
            "seed_generated": False,
            "target_roles_evaluated": False,
            "active_governance_changed": False,
            "formal_state_transition_allowed": False,
        },
    }
    report["diagnostic_report_hash"] = "sha256:" + sha256(
        REPORT_DOMAIN + b"\x00" + _canonical_json_bytes(report)
    ).hexdigest()
    return report


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(FAIL_GUARD, f"{label} is not an object")
    return value


def _exact_mapping(
    value: object, label: str, expected_fields: frozenset[str] | set[str]
) -> dict[str, object]:
    mapping = _mapping(value, label)
    if set(mapping) != expected_fields:
        _fail(
            FAIL_GUARD,
            f"{label} fields differ: expected={sorted(expected_fields)}, "
            f"observed={sorted(mapping)}",
        )
    return mapping


def _validate_dual_capacity_receipt(capacity_value: object) -> dict[str, object]:
    capacity = _exact_mapping(
        capacity_value,
        "dual capacity replay",
        DUAL_CAPACITY_REPLAY_FIELDS,
    )
    capacity_guards = {
        "status": (
            "DUAL_SHRINK6_FROZEN_DEPTH4_CHALLENGE_LATTICE_REPLAY_PASS_"
            "NOT_COMPLETE"
        ),
        "all_comparable_fields_equal": True,
        **_capacity_guard_values(),
    }
    if any(
        not _json_exact_equal(capacity.get(field), expected)
        for field, expected in capacity_guards.items()
    ):
        _fail(FAIL_GUARD, "dual capacity guard differs")
    common = {
        key: value
        for key, value in capacity.items()
        if key
        not in {"status", "all_comparable_fields_equal", "comparable_report_sha256"}
    }
    if (
        set(common) != (set(RUST_CAPACITY_FIELDS) - {"implementation"})
        or capacity.get("comparable_report_sha256")
        != sha256(_canonical_json_bytes(common)).hexdigest()
    ):
        _fail(FAIL_GUARD, "dual capacity comparison hash differs")
    return capacity


def validate_qualification_report(report_value: object) -> None:
    """Recompute every portable receipt invariant before publication."""

    report = _mapping(report_value, "qualification report")
    if set(report) != {
        "schema_version",
        "artifact_kind",
        "status",
        "claim_level",
        "repository_binding",
        "sealed_basis",
        "runtime_isolation",
        "dual_vector_replay",
        "built_in_replay_controls",
        "dual_capacity_replay",
        "authority_guards",
        "diagnostic_report_hash",
    }:
        _fail(FAIL_GUARD, "qualification report fields differ")
    if (
        report["schema_version"] != SCHEMA
        or report["status"] != STATUS_PASS
        or report["claim_level"] != CLAIM_LEVEL
        or report["artifact_kind"]
        != "COMMIT_BOUND_ENGINEERING_QUALIFICATION_EVIDENCE"
    ):
        _fail(FAIL_GUARD, "qualification report identity differs")
    body = dict(report)
    claimed_hash = body.pop("diagnostic_report_hash")
    expected_hash = "sha256:" + sha256(
        REPORT_DOMAIN + b"\x00" + _canonical_json_bytes(body)
    ).hexdigest()
    if claimed_hash != expected_hash:
        _fail(FAIL_GUARD, "diagnostic report hash differs")

    repository = _exact_mapping(
        report["repository_binding"],
        "repository binding",
        REPOSITORY_BINDING_FIELDS,
    )
    commit = repository.get("qualification_basis_commit")
    rows = repository.get("source_files")
    parents = repository.get("qualification_basis_parent_commits")
    if (
        type(commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or type(parents) is not list
        or any(
            type(parent) is not str or re.fullmatch(r"[0-9a-f]{40}", parent) is None
            for parent in parents
        )
        or parents != [PARENT_EVIDENCE_COMMIT]
        or type(repository.get("qualification_basis_subject")) is not str
        or type(repository.get("project_tree_oid")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("project_tree_oid"))) is None
        or type(repository.get("archive_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", str(repository.get("archive_sha256"))) is None
        or type(rows) is not list
        or not rows
        or type(repository.get("source_file_count")) is not int
        or repository.get("source_file_count") != len(rows)
        or repository.get("source_file_set_root") != source_file_set_root_v1(rows)
    ):
        _fail(FAIL_GUARD, "repository/source-file binding differs")
    source_hashes = {str(row["path"]): row["sha256"] for row in rows}
    if (
        len(source_hashes) != len(rows)
        or SUPERVISOR_PATH not in source_hashes
        or PARENT_EVIDENCE_PATH not in source_hashes
        or source_hashes.get(PARENT_EVIDENCE_PATH) != PARENT_EVIDENCE_SHA256
        or PROFILE_PATH not in source_hashes
        or BUILD_PROFILE_PATH not in source_hashes
        or repository.get("supervisor_source_sha256") != source_hashes[SUPERVISOR_PATH]
    ):
        _fail(FAIL_GUARD, "repository source identities differ")

    parent_binding = _mapping(
        repository.get("parent_evidence_binding"), "parent evidence binding"
    )
    if not _json_exact_equal(parent_binding, {
        "evidence_commit": PARENT_EVIDENCE_COMMIT,
        "evidence_path": PARENT_EVIDENCE_PATH,
        "evidence_record_id": PARENT_EVIDENCE_RECORD_ID,
        "evidence_sha256": "sha256:" + source_hashes[PARENT_EVIDENCE_PATH],
        "admitted_operation": "reduce max_total_ast_depth from 4 to 3",
        "formal_status_promotion_allowed": False,
    }):
        _fail(FAIL_GUARD, "parent Evidence V binding differs")

    sealed = _mapping(report["sealed_basis"], "sealed basis")
    expected_ids = [vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1]
    if not _json_exact_equal(sealed, {
        "parent_evidence_binding": parent_binding,
        "nonformal_diagnostic_roots": EXPECTED_DIAGNOSTIC_ROOTS,
        "golden_vector_manifest_root": EXPECTED_MANIFEST_ROOT,
        "expected_outcome_root": EXPECTED_OUTCOME_ROOT,
        "ordered_vector_ids": expected_ids,
        "vector_count": EXPECTED_VECTOR_COUNT,
        "preregistered_formal_bucket_count": (
            PREREGISTERED_FORMAL_BUCKET_COUNT
        ),
    }):
        _fail(FAIL_GUARD, "sealed basis differs")

    replay = _exact_mapping(
        report["dual_vector_replay"],
        "dual vector replay",
        DUAL_VECTOR_REPLAY_FIELDS,
    )
    vector_rows = replay.get("vectors")
    if type(vector_rows) is not list or len(vector_rows) != EXPECTED_VECTOR_COUNT:
        _fail(FAIL_GUARD, "dual vector rows differ")
    outcomes: dict[str, bytes] = {}
    for vector, row_value in zip(STRICT_GOLDEN_VECTORS_V1, vector_rows, strict=True):
        row = _exact_mapping(
            row_value,
            f"vector {vector.vector_id}",
            VECTOR_RECEIPT_FIELDS,
        )
        expected_row_fields = {
            "vector_id": vector.vector_id,
            "category": vector.category,
            "boundary": vector.boundary,
            "input_wire_sha256": sha256(vector.input_wire).hexdigest(),
            "input_wire_size": len(vector.input_wire),
            "expected_disposition": vector.expected_disposition,
            "python_exit": 0,
            "dual_equal": True,
        }
        if any(
            not _json_exact_equal(row.get(field), expected)
            for field, expected in expected_row_fields.items()
        ):
            _fail(FAIL_GUARD, f"vector binding differs: {vector.vector_id}")
        normalized = _mapping(row.get("normalized"), f"normalized {vector.vector_id}")
        if vector.expected_disposition == ACCEPT_PARENT_IDENTITY:
            if (
                set(normalized) != set(ACCEPT_COMPARISON_FIELDS)
                or normalized.get("status") != "ACCEPTED"
                or not _json_exact_equal(row.get("rust_exit"), 0)
            ):
                _fail(FAIL_GUARD, f"accepted vector disposition differs: {vector.vector_id}")
            _accepted_metadata_guard(
                normalized,
                code=FAIL_GUARD,
                label=f"accepted vector {vector.vector_id}",
            )
            try:
                cbor = bytes.fromhex(str(normalized["canonical_cbor_hex"]))
            except (KeyError, ValueError) as error:
                _fail(FAIL_GUARD, f"accepted vector CBOR differs: {vector.vector_id}: {error}")
            digest = sha256(AST_HASH_DOMAIN + b"\x00" + cbor).digest()
            if normalized.get("canonical_ast_hash") != "sha256:" + digest.hex():
                _fail(FAIL_GUARD, f"accepted vector hash differs: {vector.vector_id}")
            outcome = accepted_outcome_bytes(cbor, digest)
        else:
            if (
                not _json_exact_equal(normalized, {
                    "status": "REJECTED",
                    "error_code": vector.expected_disposition,
                })
                or not _json_exact_equal(row.get("rust_exit"), 1)
            ):
                _fail(FAIL_GUARD, f"rejected vector disposition differs: {vector.vector_id}")
            outcome = rejected_outcome_bytes(vector.expected_disposition)
        if row.get("normalized_outcome_sha256") != sha256(outcome).hexdigest():
            _fail(FAIL_GUARD, f"normalized outcome hash differs: {vector.vector_id}")
        outcomes[vector.vector_id] = outcome
    outcome_root = strict_golden_outcome_root_v1(outcomes)
    if (
        replay.get("status") != STATUS_PASS
        or not _json_exact_equal(
            replay.get("vector_count"), EXPECTED_VECTOR_COUNT
        )
        or replay.get("python_outcome_root") != outcome_root
        or replay.get("rust_outcome_root") != outcome_root
        or outcome_root != EXPECTED_OUTCOME_ROOT
        or replay.get("all_normalized_outcomes_equal") is not True
    ):
        _fail(FAIL_GUARD, "dual outcome root differs")

    _validate_dual_capacity_receipt(report["dual_capacity_replay"])

    controls = _mapping(
        report["built_in_replay_controls"], "built-in replay controls"
    )
    if set(controls) != {
        "python_report_sha256", "rust_report_sha256",
        "python_passed_count", "rust_passed_count",
        "python_golden_field_count", "python_capacity_field_count",
        "python_combined_field_count", "rust_golden_field_count",
        "rust_capacity_field_count", "rust_combined_field_count",
    } or any(
        re.fullmatch(r"[0-9a-f]{64}", str(controls.get(field))) is None
        for field in ("python_report_sha256", "rust_report_sha256")
    ) or not _json_exact_equal({
        "python_passed_count": controls.get("python_passed_count"),
        "rust_passed_count": controls.get("rust_passed_count"),
    }, {
        "python_passed_count": 25,
        "rust_passed_count": 25,
    }) or any(
        type(controls.get(field)) is not int or int(controls[field]) <= 0
        for field in (
            "python_golden_field_count", "python_capacity_field_count",
            "python_combined_field_count", "rust_golden_field_count",
            "rust_capacity_field_count", "rust_combined_field_count",
        )
    ):
        _fail(FAIL_GUARD, "built-in replay control receipt differs")

    runtime = _exact_mapping(
        report["runtime_isolation"],
        "runtime isolation",
        RUNTIME_ISOLATION_FIELDS,
    )
    if (
        runtime.get("role_topology")
        != "HOST_SUPERVISOR_PLUS_TWO_DISJOINT_PINNED_CONTAINERS"
        or runtime.get("python_image_ref") != PYTHON_IMAGE
        or runtime.get("python_image_id") != PYTHON_IMAGE.split("@", 1)[1]
        or runtime.get("rust_image_ref") != RUST_IMAGE
        or runtime.get("rust_image_id") != RUST_IMAGE.split("@", 1)[1]
        or runtime.get("pull_policy") != "never"
        or runtime.get("network_mode") != "none"
        or runtime.get("capabilities_dropped") != "ALL"
        or runtime.get("no_new_privileges") is not True
        or runtime.get("container_root_filesystem_read_only") is not True
        or runtime.get("source_snapshot_mount_read_only") is not True
        or runtime.get("fresh_ephemeral_rust_target_volume") is not True
        or runtime.get("rust_target_volume_removed_after_run") is not True
        or runtime.get("cargo_locked") is not True
        or runtime.get("cargo_offline") is not True
        or not _json_exact_equal(
            runtime.get("cargo_seed_subtrees"), ["cache", "index"]
        )
        or runtime.get("preunpacked_registry_src_mounted") is not False
        or runtime.get("cargo_home") != "fresh-build-tmpfs"
        or runtime.get("rust_build_user") != "0:0"
        or runtime.get("rust_build_tmpfs") != BUILD_TMPFS
        or runtime.get("recognizer_runtime_user") != "65534:65534"
        or runtime.get("recognizer_runtime_tmpfs") != RUNTIME_TMPFS
        or runtime.get("memory_limit") != "512m"
        or runtime.get("memory_swap_limit") != "512m"
        or not _json_exact_equal(runtime.get("pids_limit"), 64)
        or not _json_exact_equal(
            runtime.get("python_flags"), ["-I", "-S", "-B"]
        )
        or runtime.get("same_admin_controller") is not True
        or runtime.get("organizational_independence") is not False
        or runtime.get("independent_human_actors") is not False
        or runtime.get("technical_role_independence") is not True
        or runtime.get("owner_accepted_threat_model") is not True
        or type(runtime.get("worker_count")) is not int
        or not 1 <= int(runtime["worker_count"]) <= 16
        or re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("rust_binary_sha256")))
        is None
    ):
        _fail(FAIL_GUARD, "runtime isolation guard differs")
    _validate_cargo_seed_receipt(runtime.get("cargo_seed_manifest_receipt"))
    daemon = _exact_mapping(
        runtime.get("docker_daemon_identity_receipt"),
        "Docker daemon receipt",
        DOCKER_DAEMON_RECEIPT_FIELDS,
    )
    daemon_body = dict(daemon)
    daemon_hash = daemon_body.pop("diagnostic_receipt_hash", None)
    if daemon_hash != "sha256:" + sha256(
        DAEMON_RECEIPT_DOMAIN + b"\x00" + _canonical_json_bytes(daemon_body)
    ).hexdigest():
        _fail(FAIL_GUARD, "Docker daemon receipt hash differs")
    if (
        daemon.get("schema_version")
        != "hegel-shrink6-local-docker-daemon-receipt/1"
        or daemon.get("docker_executable") != DOCKER_EXECUTABLE
        or daemon.get("explicit_host_argument") != DOCKER_HOST_ARGUMENT
        or daemon.get("socket") != str(DOCKER_SOCKET)
        or daemon.get("private_empty_client_config_sha256")
        != sha256(b"{}\n").hexdigest()
        or daemon.get("host_environment_keys")
        != ["DOCKER_CONFIG", "DOCKER_HOST", "HOME", "LANG", "LC_ALL", "PATH"]
        or any(
            type(daemon.get(field)) is not int or int(daemon[field]) < 0
            for field in ("socket_device", "socket_inode", "socket_uid", "socket_gid")
        )
    ):
        _fail(FAIL_GUARD, "Docker daemon fixed receipt fields differ")
    server = _exact_mapping(
        daemon.get("server"), "Docker server receipt", DOCKER_SERVER_RECEIPT_FIELDS
    )
    if any(
        type(server.get(field)) is not str or not str(server[field])
        for field in DOCKER_SERVER_RECEIPT_FIELDS
    ) or re.fullmatch(r"[0-9a-f]{64}", str(server.get("raw_canonical_sha256"))) is None:
        _fail(FAIL_GUARD, "Docker server receipt values differ")
    daemon_info = _exact_mapping(
        daemon.get("daemon"), "Docker info receipt", DOCKER_INFO_RECEIPT_FIELDS
    )
    if (
        daemon_info.get("os_type") != "linux"
        or any(
            type(daemon_info.get(field)) is not str or not str(daemon_info[field])
            for field in (
                "id", "name", "driver", "operating_system", "os_type",
                "architecture", "docker_root_dir",
            )
        )
        or type(daemon_info.get("security_options")) is not list
        or any(type(item) is not str for item in daemon_info["security_options"])
    ):
        _fail(FAIL_GUARD, "Docker info receipt values differ")
    profiles = _mapping(runtime.get("committed_profile_receipt"), "profile receipt")
    if not _json_exact_equal(profiles, {
        "actor_profile_id": "hegel-owner-accepted-container-technical-actors-v1",
        "actor_profile_sha256": source_hashes[PROFILE_PATH],
        "build_profile_id": "hegel-shrink6-rust-offline-build-v1",
        "build_profile_sha256": source_hashes[BUILD_PROFILE_PATH],
    }):
        _fail(FAIL_GUARD, "committed profile receipt differs")
    volume = _exact_mapping(
        runtime.get("rust_target_volume_receipt"),
        "target volume receipt",
        TARGET_VOLUME_RECEIPT_FIELDS,
    )
    if (
        volume.get("name") != f"hegel-shrink6-sealed-{commit[:12]}"
        or volume.get("driver") != "local"
        or volume.get("scope") != "local"
        or volume.get("options") is not None
        or volume.get("fresh_before_run") is not True
        or not _json_exact_equal(volume.get("labels"), {
            "hegel.machine.role": "shrink6-dual-strict",
            "hegel.machine.basis": commit,
            "hegel.machine.network": "none",
        })
    ):
        _fail(FAIL_GUARD, "target-volume receipt differs")

    python_runtime = _exact_mapping(
        runtime.get("python_runtime"),
        "Python runtime receipt",
        PYTHON_RUNTIME_RECEIPT_FIELDS,
    )
    if (
        type(python_runtime.get("executable")) is not str
        or not str(python_runtime["executable"]).startswith("/")
        or re.fullmatch(
            r"[0-9a-f]{64}", str(python_runtime.get("executable_sha256"))
        )
        is None
        or type(python_runtime.get("version")) is not str
        or not python_runtime["version"]
    ):
        _fail(FAIL_GUARD, "Python runtime receipt values differ")
    rust_runtime = _exact_mapping(
        runtime.get("rust_runtime"),
        "Rust runtime receipt",
        RUST_RUNTIME_RECEIPT_FIELDS,
    )
    if (
        type(rust_runtime.get("rustc_version_verbose")) is not str
        or not rust_runtime["rustc_version_verbose"]
    ):
        _fail(FAIL_GUARD, "Rust runtime receipt values differ")

    if not _json_exact_equal(report["authority_guards"], {
        "execution_state": "NOT_RUN",
        "closure_executed": False,
        "formal_roots_generated": False,
        "formal_roots": None,
        "certificate_issued": False,
        "signature_generated": False,
        "seed_generated": False,
        "target_roles_evaluated": False,
        "active_governance_changed": False,
        "formal_state_transition_allowed": False,
    }):
        _fail(FAIL_GUARD, "authority guards differ")


def qualify(
    basis_commit: str,
    *,
    workers: int,
    cargo_registry: Path,
) -> dict[str, object]:
    global _DOCKER_ENV
    if re.fullmatch(r"[0-9a-f]{40}", basis_commit) is None:
        _fail(FAIL_ARGUMENT, "basis commit must be a full lowercase SHA-1")
    _require_sealed_commitments()
    resolved = _git("rev-parse", "--verify", f"{basis_commit}^{{commit}}")
    if resolved != basis_commit:
        _fail(FAIL_GIT_BINDING, "basis commit does not resolve exactly")
    if not 1 <= workers <= 16:
        _fail(FAIL_ARGUMENT, "workers must be in [1,16]")
    source_rows, source_root = _source_rows(basis_commit)
    parent_evidence_binding = _parent_evidence_binding(basis_commit)
    project_tree_oid = _git("rev-parse", f"{basis_commit}:Hegel Machine")
    subject = _git("show", "-s", "--format=%s", basis_commit)
    parents = str(_git("show", "-s", "--format=%P", basis_commit)).split()
    if parents != [PARENT_EVIDENCE_COMMIT]:
        _fail(
            FAIL_GIT_BINDING,
            "shrink-6 source basis must be a single-parent direct child of Evidence V",
        )
    archive = _git("archive", "--format=tar", basis_commit, "--", *ARCHIVE_PATHS, binary=True)
    if type(archive) is not bytes:
        _fail(FAIL_ARCHIVE, "Git archive is not bytes")
    archive_sha256 = sha256(archive).hexdigest()
    repository_binding = {
        "qualification_basis_commit": basis_commit,
        "qualification_basis_parent_commits": parents,
        "qualification_basis_subject": subject,
        "project_tree_oid": project_tree_oid,
        "archive_sha256": archive_sha256,
        "source_file_count": len(source_rows),
        "source_file_set_root": source_root,
        "supervisor_source_sha256": next(
            row["sha256"] for row in source_rows if row["path"] == SUPERVISOR_PATH
        ),
        "parent_evidence_binding": parent_evidence_binding,
        "source_files": source_rows,
    }

    volume = f"hegel-shrink6-sealed-{basis_commit[:12]}"
    with (
        tempfile.TemporaryDirectory(prefix="hegel-shrink6-docker-control-") as control,
        tempfile.TemporaryDirectory(prefix="hegel-shrink6-sealed-snapshot-") as temporary,
    ):
        try:
            daemon_receipt = _initialize_docker_environment(Path(control))
            python_image_id = _inspect_image(PYTHON_IMAGE)
            rust_image_id = _inspect_image(RUST_IMAGE)
            snapshot_root = Path(temporary)
            _safe_extract_git_archive(archive, snapshot_root)
            _validate_snapshot(snapshot_root, source_rows)
            snapshot_project = snapshot_root / "Hegel Machine"
            profile_receipt = _profile_images(snapshot_project)
            volume_receipt = _create_fresh_volume(volume, basis_commit)
            primary: BaseException | None = None
            report: dict[str, object] | None = None
            try:
                report = _qualify(
                    snapshot_project,
                    volume,
                    repository_binding=repository_binding,
                    workers=workers,
                    cargo_registry=cargo_registry.resolve(),
                    python_image_id=python_image_id,
                    rust_image_id=rust_image_id,
                    daemon_receipt=daemon_receipt,
                    volume_receipt=volume_receipt,
                    profile_receipt=profile_receipt,
                )
            except BaseException as error:
                primary = error
            try:
                _remove_volume(volume)
            except QualificationError as cleanup_error:
                if primary is not None:
                    if isinstance(primary, QualificationError):
                        raise QualificationError(
                            primary.code,
                            f"{primary.detail}; secondary cleanup failure: "
                            f"{cleanup_error.code}: {cleanup_error.detail}",
                        ) from primary
                    raise QualificationError(
                        FAIL_CLEANUP,
                        f"primary {type(primary).__name__}; secondary cleanup "
                        f"failure: {cleanup_error.detail}",
                    ) from primary
                raise
            if primary is not None:
                raise primary
            if report is None:
                _fail(FAIL_ENDPOINT, "qualification returned no report")
            validate_qualification_report(report)
        finally:
            _DOCKER_ENV = None
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-commit", required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--cargo-registry", type=Path, default=DEFAULT_CARGO_REGISTRY)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        report = qualify(
            arguments.basis_commit,
            workers=arguments.workers,
            cargo_registry=arguments.cargo_registry,
        )
    except QualificationError as error:
        sys.stderr.write(
            json.dumps(
                {"status": "FAIL_CLOSED", "failure_code": error.code, "detail": error.detail},
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
        )
        return 2
    except Exception as error:
        sys.stderr.write(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "failure_code": FAIL_INTERNAL,
                    "detail": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
        )
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""History-complete, content-nondisclosing genesis-secret absence audit.

The audit is intentionally narrow.  It detects only the frozen classes of
Phase-3A M2.5 genesis secret artifacts below; it is not a general secret
scanner and must not be described as one.  A specified Git commit and every
ancestor state of the ``Hegel Machine/`` subtree are replayed from Git object
storage, so dirty working-tree bytes are neither trusted nor inspected.

The public receipt contains Git/path metadata and policy identifiers only.  It
never includes a blob body, a matched private-key header, or a JSON value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Final

from .hashing import stable_hash


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_SCOPE_PREFIX: Final = "Hegel Machine/"
MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
ARTIFACT_NAME: Final = "phase3_m25_repository_genesis_secret_absence_v1"
REPORT_SCHEMA: Final = "hegel-phase3-m25-repository-genesis-secret-absence/1"
POLICY_ID: Final = "hegel-m25-frozen-genesis-secret-artifact-policy-v1"
ARTIFACT_KIND: Final = "DIAGNOSTIC_NON_AUTHORITATIVE"
PASS_STATUS: Final = "FROZEN_GENESIS_SECRET_ARTIFACTS_ABSENT"
FAIL_STATUS: Final = "FROZEN_GENESIS_SECRET_ARTIFACTS_DETECTED"
CLAIM_BOUNDARY: Final = (
    "No artifact matching the frozen Phase-3A M2.5 genesis-secret filename, "
    "private-key-header, or non-null forbidden-JSON-key policy was found in "
    "unique Git blobs reachable under Hegel Machine/ at the specified commit "
    "or any ancestor state. This is not a universal secret-detection claim."
)


FORBIDDEN_EXACT_BASENAMES: Final = frozenset(
    {
        ".env",
        ".envrc",
        "auditor_private_key",
        "custodian_private_key",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "k_split_master",
        "k_split_master.bin",
        "k_split_master.hex",
        "private_key",
        "private_key.bin",
        "private_key.hex",
        "python_attester_private_key",
        "raw_split_seed",
        "raw_split_seed.bin",
        "raw_split_seed.hex",
        "rust_attester_private_key",
        "split_master_seed",
        "split_master_seed.bin",
        "split_master_seed.hex",
        "split_seed",
        "split_seed.bin",
        "split_seed.hex",
    }
)

FORBIDDEN_EXTENSIONS: Final = frozenset(
    {
        ".jks",
        ".key",
        ".keystore",
        ".p12",
        ".p8",
        ".pem",
        ".pfx",
        ".pk8",
        ".pkcs8",
        ".secret",
        ".seed",
    }
)

# Split the byte literals so the policy implementation's own Git blob cannot
# match the signatures it is responsible for detecting.
_PEM_BEGIN: Final = b"-----BEGIN "
_PEM_PRIVATE_END: Final = b" PRIVATE KEY-----"
PRIVATE_KEY_MAGIC_HEADERS: Final = {
    "AGE_SECRET_KEY": b"AGE-SECRET-" + b"KEY-1",
    "DSA_PEM": _PEM_BEGIN + b"DSA" + _PEM_PRIVATE_END,
    "EC_PEM": _PEM_BEGIN + b"EC" + _PEM_PRIVATE_END,
    "ENCRYPTED_PKCS8_PEM": _PEM_BEGIN + b"ENCRYPTED" + _PEM_PRIVATE_END,
    "OPENPGP_PRIVATE_PEM": _PEM_BEGIN + b"PGP" + b" PRIVATE KEY BLOCK-----",
    "OPENSSH_BINARY": b"openssh-key-v1" + bytes((0,)),
    "OPENSSH_PEM": _PEM_BEGIN + b"OPENSSH" + _PEM_PRIVATE_END,
    "PKCS8_PEM": _PEM_BEGIN + b"PRIVATE KEY-----",
    "PUTTY_PRIVATE_KEY": b"PuTTY-User-" + b"Key-File-",
    "RSA_PEM": _PEM_BEGIN + b"RSA" + _PEM_PRIVATE_END,
}

FORBIDDEN_JSON_SECRET_KEYS: Final = frozenset(
    {
        "assignment_rows",
        "auditor_private_key",
        "custodian_private_key",
        "derived_role_key",
        "ed25519_private_key",
        "ed25519_private_key_seed",
        "k_role",
        "k_split_master",
        "master_seed_hex",
        "pre_final_match_set",
        "pre_final_output_archive",
        "private_key",
        "private_key_base64",
        "private_key_bytes",
        "private_key_der",
        "private_key_pem",
        "private_key_seed",
        "python_attester_private_key",
        "raw_k_split_master",
        "raw_private_key",
        "raw_seed",
        "raw_split_seed",
        "rust_attester_private_key",
        "sealed_membership",
        "sealed_prediction_membership",
        "split_master_seed",
        "split_seed",
        "split_seed_base64",
        "split_seed_bytes",
        "split_seed_hex",
        "validation_membership",
    }
)

SYNTHETIC_VECTOR_PREFIX: Final = "Hegel Machine/golden_vectors/"
SYNTHETIC_VECTOR_ARTIFACT_KINDS: Final = frozenset(
    {
        "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE",
        "SYNTHETIC_NON_AUTHORITATIVE",
    }
)


class GenesisSecretAbsenceError(RuntimeError):
    """Fail-closed Git replay or receipt validation error."""


@dataclass(frozen=True)
class _JsonObject:
    pairs: tuple[tuple[str, object], ...]


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _require_commit_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GenesisSecretAbsenceError(
            "audited commit must be a lowercase 40-hex Git SHA-1"
        )
    return value


def _repository_root() -> Path:
    output = _git_bytes(
        PROJECT_ROOT.parent, ["rev-parse", "--show-toplevel"], timeout=30
    )
    root = Path(output.decode("utf-8", "strict").strip()).resolve()
    if (root / REPOSITORY_SCOPE_PREFIX.rstrip("/")).resolve() != PROJECT_ROOT.resolve():
        raise GenesisSecretAbsenceError("Hegel Machine project root/scope mismatch")
    return root


def _git_bytes(repository_root: Path, arguments: list[str], *, timeout: int = 120) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_SSH_COMMAND": "false",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    if completed.returncode != 0:
        command = " ".join(arguments[:2])
        raise GenesisSecretAbsenceError(f"Git audit command failed: {command}")
    return completed.stdout


def _normalize_json_key(key: str) -> str:
    with_camel_breaks = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", with_camel_breaks.casefold()).strip("_")


def _path_display(path_bytes: bytes) -> str:
    try:
        return path_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        digest = hashlib.sha256(path_bytes).hexdigest()
        return f"<non-utf8-path:sha256:{digest}>"


def _path_sha256(path_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(path_bytes).hexdigest()


def _json_location_sha256(components: tuple[str, ...]) -> str:
    encoded = json.dumps(
        list(components), ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _decode_json_pairs(blob: bytes) -> object:
    text = blob.decode("utf-8", errors="strict")
    return json.loads(
        text,
        object_pairs_hook=lambda pairs: _JsonObject(tuple(pairs)),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON constant: {value}")
        ),
    )


def _plain_json_object(value: object) -> object:
    if isinstance(value, _JsonObject):
        result: dict[str, object] = {}
        for key, child in value.pairs:
            result[key] = _plain_json_object(child)
        return result
    if isinstance(value, list):
        return [_plain_json_object(child) for child in value]
    return value


def _json_secret_key_hits(value: object) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    stack: list[tuple[object, tuple[str, ...]]] = [(value, ())]
    while stack:
        current, location = stack.pop()
        if isinstance(current, _JsonObject):
            for pair_index in range(len(current.pairs) - 1, -1, -1):
                key, child = current.pairs[pair_index]
                normalized = _normalize_json_key(key)
                child_location = location + (
                    f"object-pair:{pair_index}",
                    f"key:{key}",
                )
                if normalized in FORBIDDEN_JSON_SECRET_KEYS and child is not None:
                    hits.append(
                        (normalized, _json_location_sha256(child_location))
                    )
                stack.append((child, child_location))
        elif isinstance(current, list):
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], location + (f"array:{index}",)))
    return sorted(hits)


def _synthetic_vector_descriptor(value: object) -> bool:
    plain = _plain_json_object(value)
    if not isinstance(plain, dict):
        return False
    if plain.get("artifact_kind") not in SYNTHETIC_VECTOR_ARTIFACT_KINDS:
        return False
    if plain.get("machine_freeze_id") != MACHINE_FREEZE_ID:
        return False
    boundary = plain.get("authority_boundary")
    return (
        isinstance(boundary, dict)
        and boundary.get("contains_real_secret_material") is False
        and boundary.get("authoritative_root_generation") is False
        and boundary.get("seed_genesis_performed") is False
        and boundary.get("signature_claim") is False
    )


def _filename_policy_token(path_bytes: bytes) -> str | None:
    basename = path_bytes.rsplit(b"/", 1)[-1].decode("ascii", errors="ignore").casefold()
    if basename == ".env" or basename.startswith(".env."):
        return "ENV_FILENAME_RULE"
    if basename in FORBIDDEN_EXACT_BASENAMES:
        return "EXACT_BASENAME:" + basename
    suffix = Path(basename).suffix.casefold()
    if suffix in FORBIDDEN_EXTENSIONS:
        return "FORBIDDEN_EXTENSION:" + suffix
    return None


def _tree_blob_paths(
    repository_root: Path,
    commit_ids: tuple[str, ...],
) -> tuple[dict[str, set[bytes]], int, int]:
    paths_by_blob: dict[str, set[bytes]] = {}
    observation_count = 0
    non_blob_entry_count = 0
    for commit_id in commit_ids:
        output = _git_bytes(
            repository_root,
            [
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                commit_id,
                "--",
                REPOSITORY_SCOPE_PREFIX.rstrip("/"),
            ],
        )
        for record in output.split(b"\x00"):
            if not record:
                continue
            try:
                metadata, path_bytes = record.split(b"\t", 1)
                _mode, object_type, object_id = metadata.split(b" ", 2)
            except ValueError as exc:
                raise GenesisSecretAbsenceError("malformed git ls-tree record") from exc
            observation_count += 1
            if object_type != b"blob":
                non_blob_entry_count += 1
                continue
            oid = object_id.decode("ascii", errors="strict")
            paths_by_blob.setdefault(oid, set()).add(path_bytes)
    return paths_by_blob, observation_count, non_blob_entry_count


def _policy_payload() -> dict[str, object]:
    return {
        "policy_id": POLICY_ID,
        "claim_scope": "FROZEN_PHASE3A_M25_GENESIS_SECRET_ARTIFACT_CLASSES_ONLY",
        "forbidden_exact_basenames": sorted(FORBIDDEN_EXACT_BASENAMES),
        "forbidden_env_filename_rule": "basename == .env OR basename starts .env.",
        "forbidden_extensions": sorted(FORBIDDEN_EXTENSIONS),
        "private_key_magic_header_ids": sorted(PRIVATE_KEY_MAGIC_HEADERS),
        "forbidden_json_secret_keys": sorted(FORBIDDEN_JSON_SECRET_KEYS),
        "json_key_normalization": (
            "ASCII camel-boundary insertion; Unicode casefold; non-[a-z0-9] "
            "runs become underscore; trim underscores"
        ),
        "json_non_null_rule": "every value other than JSON null is a finding",
        "synthetic_vector_exemption": {
            "path_prefix": SYNTHETIC_VECTOR_PREFIX,
            "allowed_artifact_kinds": sorted(SYNTHETIC_VECTOR_ARTIFACT_KINDS),
            "requires_machine_freeze_id": MACHINE_FREEZE_ID,
            "requires_contains_real_secret_material_false": True,
            "requires_authoritative_root_generation_false": True,
            "requires_seed_genesis_performed_false": True,
            "requires_signature_claim_false": True,
            "applies_only_to": ["SECRET_FILENAME", "SECRET_FILE_EXTENSION"],
            "never_applies_to": [
                "PRIVATE_KEY_MAGIC_HEADER",
                "NON_NULL_FORBIDDEN_JSON_SECRET_KEY",
            ],
        },
    }


def _scan_once(commit_id: str) -> dict[str, object]:
    repository_root = _repository_root()
    _git_bytes(repository_root, ["cat-file", "-e", f"{commit_id}^{{commit}}"])
    ancestor_ids = tuple(
        line.decode("ascii")
        for line in _git_bytes(repository_root, ["rev-list", commit_id]).splitlines()
        if line
    )
    relevant_ids = {
        line.decode("ascii")
        for line in _git_bytes(
            repository_root,
            [
                "rev-list",
                "--full-history",
                commit_id,
                "--",
                REPOSITORY_SCOPE_PREFIX.rstrip("/"),
            ],
        ).splitlines()
        if line
    }
    relevant_ids.add(commit_id)
    path_state_commits = tuple(sorted(relevant_ids))
    paths_by_blob, observations, non_blob_entries = _tree_blob_paths(
        repository_root, path_state_commits
    )

    findings: list[dict[str, object]] = []
    json_blob_count = 0
    synthetic_exemptions: set[tuple[str, bytes]] = set()
    total_bytes = 0
    for blob_oid in sorted(paths_by_blob):
        paths = tuple(sorted(paths_by_blob[blob_oid]))
        blob = _git_bytes(repository_root, ["cat-file", "blob", blob_oid])
        total_bytes += len(blob)
        has_json_path = any(path.lower().endswith(b".json") for path in paths)
        has_synthetic_path = any(
            path.startswith(SYNTHETIC_VECTOR_PREFIX.encode("utf-8")) for path in paths
        )
        decoded_json: object | None = None
        json_parse_failed = False
        if has_json_path or has_synthetic_path:
            json_blob_count += 1
            try:
                decoded_json = _decode_json_pairs(blob)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
                json_parse_failed = True
                if has_json_path:
                    findings.append(
                        {
                            "finding_code": "UNSCANNABLE_JSON_BLOB",
                            "blob_oid": blob_oid,
                            "blob_byte_length": len(blob),
                            "repository_paths": [_path_display(path) for path in paths],
                            "path_sha256_or_null": None,
                            "policy_token": "STRICT_UTF8_RFC8259_JSON_REQUIRED",
                            "json_location_sha256_or_null": None,
                        }
                    )

        synthetic_descriptor = (
            decoded_json is not None
            and not json_parse_failed
            and _synthetic_vector_descriptor(decoded_json)
        )
        for path in paths:
            filename_token = _filename_policy_token(path)
            if filename_token is None:
                continue
            exempt = (
                path.startswith(SYNTHETIC_VECTOR_PREFIX.encode("utf-8"))
                and synthetic_descriptor
            )
            if exempt:
                synthetic_exemptions.add((blob_oid, path))
                continue
            findings.append(
                {
                    "finding_code": "FORBIDDEN_SECRET_FILENAME_OR_EXTENSION",
                    "blob_oid": blob_oid,
                    "blob_byte_length": len(blob),
                    "repository_paths": [_path_display(path)],
                    "path_sha256_or_null": _path_sha256(path),
                    "policy_token": filename_token,
                    "json_location_sha256_or_null": None,
                }
            )

        for header_id, header in sorted(PRIVATE_KEY_MAGIC_HEADERS.items()):
            if header in blob:
                findings.append(
                    {
                        "finding_code": "PRIVATE_KEY_MAGIC_HEADER",
                        "blob_oid": blob_oid,
                        "blob_byte_length": len(blob),
                        "repository_paths": [_path_display(path) for path in paths],
                        "path_sha256_or_null": None,
                        "policy_token": header_id,
                        "json_location_sha256_or_null": None,
                    }
                )

        if decoded_json is not None and not json_parse_failed:
            for forbidden_key, location_digest in _json_secret_key_hits(decoded_json):
                findings.append(
                    {
                        "finding_code": "NON_NULL_FORBIDDEN_JSON_SECRET_KEY",
                        "blob_oid": blob_oid,
                        "blob_byte_length": len(blob),
                        "repository_paths": [_path_display(path) for path in paths],
                        "path_sha256_or_null": None,
                        "policy_token": forbidden_key,
                        "json_location_sha256_or_null": location_digest,
                    }
                )

    if non_blob_entries:
        findings.append(
            {
                "finding_code": "UNSUPPORTED_NON_BLOB_TREE_ENTRY",
                "blob_oid": None,
                "blob_byte_length": 0,
                "repository_paths": [],
                "path_sha256_or_null": None,
                "policy_token": "GITLINK_OR_NON_BLOB_UNDER_SCOPE",
                "json_location_sha256_or_null": None,
            }
        )
    findings.sort(
        key=lambda item: (
            str(item["finding_code"]),
            "" if item["blob_oid"] is None else str(item["blob_oid"]),
            tuple(item["repository_paths"]),
            str(item["policy_token"]),
            ""
            if item["json_location_sha256_or_null"] is None
            else str(item["json_location_sha256_or_null"]),
        )
    )
    offending_blobs = {
        str(item["blob_oid"])
        for item in findings
        if item["blob_oid"] is not None
    }
    return {
        "ancestor_commit_count": len(ancestor_ids),
        "path_state_commit_count": len(path_state_commits),
        "tree_entry_observation_count": observations,
        "unique_blob_count": len(paths_by_blob),
        "unique_blob_path_association_count": sum(
            len(paths) for paths in paths_by_blob.values()
        ),
        "unique_blob_bytes_scanned": total_bytes,
        "json_blob_count": json_blob_count,
        "synthetic_vector_path_exemption_count": len(synthetic_exemptions),
        "unsupported_non_blob_entry_count": non_blob_entries,
        "finding_count": len(findings),
        "offending_unique_blob_count": len(offending_blobs),
        "findings": findings,
    }


def repository_genesis_secret_absence_report(
    commit_id: str,
) -> dict[str, object]:
    """Audit one commit plus all ancestors and return a replay-stable receipt."""

    audited_commit = _require_commit_id(commit_id)
    first = _scan_once(audited_commit)
    second = _scan_once(audited_commit)
    if not _strict_json_equal(first, second):
        raise GenesisSecretAbsenceError(
            "repository genesis-secret audit changed across immediate replay"
        )
    findings = first["findings"]
    assert isinstance(findings, list)
    passed = len(findings) == 0
    payload: dict[str, object] = {
        "artifact": ARTIFACT_NAME,
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "pass": passed,
        "audited_commit_id": audited_commit,
        "scope": {
            "repository_relative_prefix": REPOSITORY_SCOPE_PREFIX,
            "history_scope": "SPECIFIED_COMMIT_AND_ALL_ANCESTORS",
            "object_scope": "UNIQUE_BLOBS_FROM_ALL_SUBTREE_PATH_STATES",
            "working_tree_consulted": False,
            "blob_content_disclosed_in_receipt": False,
        },
        "policy": _policy_payload(),
        "counts": {
            key: value for key, value in first.items() if key != "findings"
        },
        "zero_findings": passed,
        "findings": findings,
        "immediate_second_replay_equal": True,
        "authority_boundary": {
            "diagnostic_only": True,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_gate_delta": 0,
            "child_state_effect": "NONE",
            "universal_secret_detection_claim": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["diagnostic_report_id"] = stable_hash(
        payload, prefix="phase3_m25_secret_absence_"
    )
    return payload


def validate_repository_genesis_secret_absence_report(
    report: Mapping[str, object],
    *,
    expected_commit_id: str,
) -> None:
    """Strictly replay and validate a diagnostic absence receipt."""

    expected_commit = _require_commit_id(expected_commit_id)
    if not isinstance(report, Mapping):
        raise GenesisSecretAbsenceError("secret-absence receipt must be a mapping")
    if report.get("audited_commit_id") != expected_commit:
        raise GenesisSecretAbsenceError("secret-absence receipt commit mismatch")
    if type(report.get("pass")) is not bool:
        raise GenesisSecretAbsenceError("secret-absence pass field must be bool")
    if type(report.get("zero_findings")) is not bool:
        raise GenesisSecretAbsenceError("secret-absence zero_findings must be bool")
    expected = repository_genesis_secret_absence_report(expected_commit)
    if not _strict_json_equal(dict(report), expected):
        raise GenesisSecretAbsenceError(
            "secret-absence receipt differs from current history replay"
        )


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_NAME",
    "CLAIM_BOUNDARY",
    "FAIL_STATUS",
    "FORBIDDEN_EXACT_BASENAMES",
    "FORBIDDEN_EXTENSIONS",
    "FORBIDDEN_JSON_SECRET_KEYS",
    "GenesisSecretAbsenceError",
    "MACHINE_FREEZE_ID",
    "PASS_STATUS",
    "POLICY_ID",
    "PRIVATE_KEY_MAGIC_HEADERS",
    "REPORT_SCHEMA",
    "REPOSITORY_SCOPE_PREFIX",
    "repository_genesis_secret_absence_report",
    "validate_repository_genesis_secret_absence_report",
]

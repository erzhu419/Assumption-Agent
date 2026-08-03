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
import os
from pathlib import Path
import re
import subprocess
from typing import Final

from .hashing import stable_hash


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_SCOPE_PREFIX: Final = "Hegel Machine/"
MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
ARTIFACT_NAME: Final = "phase3_m25_repository_genesis_secret_absence_v1"
REPORT_SCHEMA: Final = "hegel-phase3-m25-repository-genesis-secret-absence/2"
POLICY_ID: Final = "hegel-m25-frozen-genesis-secret-artifact-policy-v2"
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

# A record-start header is sufficient for every class above.  In addition,
# complete PEM/OpenPGP blocks are findings even when serialized inside a JSON,
# YAML, Markdown, or source-code string.  The non-PEM formats have no matching
# footer, so their identifying magic is forbidden at every byte offset.
PRIVATE_KEY_MAGIC_FOOTERS: Final = {
    "DSA_PEM": b"-----END " + b"DSA PRIVATE KEY-----",
    "EC_PEM": b"-----END " + b"EC PRIVATE KEY-----",
    "ENCRYPTED_PKCS8_PEM": b"-----END " + b"ENCRYPTED PRIVATE KEY-----",
    "OPENPGP_PRIVATE_PEM": b"-----END " + b"PGP PRIVATE KEY BLOCK-----",
    "OPENSSH_PEM": b"-----END " + b"OPENSSH PRIVATE KEY-----",
    "PKCS8_PEM": b"-----END " + b"PRIVATE KEY-----",
    "RSA_PEM": b"-----END " + b"RSA PRIVATE KEY-----",
}
PRIVATE_KEY_ANY_OFFSET_IDS: Final = frozenset(
    {"AGE_SECRET_KEY", "OPENSSH_BINARY", "PUTTY_PRIVATE_KEY"}
)

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


def _validated_repository_scope_v1(
    repository_root: Path, project_root: Path
) -> tuple[Path, Path]:
    """Validate an explicit Git toplevel and its lexical Hegel scope."""

    if not isinstance(repository_root, Path) or not isinstance(project_root, Path):
        raise TypeError("repository_root and project_root must be pathlib.Path")
    try:
        requested_repository = Path(os.path.abspath(os.fspath(repository_root)))
        requested_project = Path(os.path.abspath(os.fspath(project_root)))
        repository = requested_repository.resolve(strict=True)
        project = requested_project.resolve(strict=True)
    except OSError as exc:
        raise GenesisSecretAbsenceError(
            "explicit repository/project scope is unavailable"
        ) from exc
    if (
        repository != requested_repository
        or project != requested_project
        or not repository.is_dir()
        or not project.is_dir()
        or project != repository / REPOSITORY_SCOPE_PREFIX.rstrip("/")
    ):
        raise GenesisSecretAbsenceError(
            "explicit repository/project scope is aliased or mismatched"
        )
    top = _git_bytes(repository, ["rev-parse", "--show-toplevel"], timeout=30)
    try:
        top_path = Path(top.decode("utf-8", "strict").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise GenesisSecretAbsenceError("explicit Git toplevel is invalid") from exc
    if top_path != repository:
        raise GenesisSecretAbsenceError("explicit repository is not the Git toplevel")
    return repository, project


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


def _git_path(repository_root: Path, name: str) -> Path:
    """Resolve one Git administrative path without accepting an empty alias."""

    raw = _git_bytes(repository_root, ["rev-parse", "--git-path", name], timeout=30)
    try:
        text = raw.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as exc:
        raise GenesisSecretAbsenceError("Git administrative path is not UTF-8") from exc
    if not text or "\x00" in text:
        raise GenesisSecretAbsenceError("Git administrative path is malformed")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return candidate


def _reject_incomplete_or_rewritten_history(repository_root: Path) -> None:
    """Reject metadata that can truncate or externally rewrite history.

    Parent traversal below reads raw commit-object headers and is therefore
    immune to graft semantics.  These checks additionally make the receipt's
    object-completeness boundary explicit and fail closed for shallow,
    promisor, alternate-object, graft, or replace-ref repositories.
    """

    shallow = _git_bytes(
        repository_root, ["rev-parse", "--is-shallow-repository"], timeout=30
    )
    if shallow != b"false\n":
        raise GenesisSecretAbsenceError(
            "history-complete audit rejects a shallow repository"
        )

    local_config = _git_bytes(
        repository_root, ["config", "--local", "--null", "--list"], timeout=30
    ).lower()
    if b"promisor" in local_config or b"partialclone" in local_config:
        raise GenesisSecretAbsenceError(
            "history-complete audit rejects promisor or partial-clone config"
        )

    forbidden_admin_paths = (
        _git_path(repository_root, "objects/info/alternates"),
        _git_path(repository_root, "objects/info/http-alternates"),
        _git_path(repository_root, "info/grafts"),
    )
    if any(path.exists() or path.is_symlink() for path in forbidden_admin_paths):
        raise GenesisSecretAbsenceError(
            "history-complete audit rejects alternate-object or graft metadata"
        )

    object_directory = _git_path(repository_root, "objects")
    pack_directory = object_directory / "pack"
    try:
        if pack_directory.is_dir() and any(pack_directory.glob("*.promisor")):
            raise GenesisSecretAbsenceError(
                "history-complete audit rejects promisor object packs"
            )
    except OSError as exc:
        raise GenesisSecretAbsenceError(
            "history-complete audit cannot inspect the object pack directory"
        ) from exc

    replace_refs = _git_bytes(
        repository_root,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
        timeout=30,
    )
    if replace_refs:
        raise GenesisSecretAbsenceError(
            "history-complete audit rejects replacement refs"
        )


def _raw_commit_identity(
    repository_root: Path, commit_id: str
) -> tuple[str, tuple[str, ...]]:
    """Return the raw tree and parents encoded by one commit object."""

    commit = _require_commit_id(commit_id)
    raw = _git_bytes(repository_root, ["cat-file", "commit", commit])
    if b"\x00" in raw or b"\r" in raw or b"\n\n" not in raw:
        raise GenesisSecretAbsenceError("raw commit object framing is malformed")
    header, _message = raw.split(b"\n\n", 1)
    trees: list[str] = []
    parents: list[str] = []
    saw_header = False
    for line in header.split(b"\n"):
        if line.startswith(b" "):
            if not saw_header:
                raise GenesisSecretAbsenceError(
                    "raw commit object has an orphan continuation"
                )
            continue
        try:
            key, value = line.split(b" ", 1)
        except ValueError as exc:
            raise GenesisSecretAbsenceError("raw commit header is malformed") from exc
        if re.fullmatch(rb"[a-z][a-z0-9-]*", key) is None or not value:
            raise GenesisSecretAbsenceError("raw commit header key/value is malformed")
        saw_header = True
        if key not in {b"tree", b"parent"}:
            continue
        try:
            object_id = value.decode("ascii", "strict")
        except UnicodeDecodeError as exc:
            raise GenesisSecretAbsenceError(
                "raw commit tree/parent is not ASCII"
            ) from exc
        _require_commit_id(object_id)
        if key == b"tree":
            trees.append(object_id)
        else:
            parents.append(object_id)
    if len(trees) != 1 or len(set(parents)) != len(parents):
        raise GenesisSecretAbsenceError(
            "raw commit must encode exactly one tree and unique parent rows"
        )
    _git_bytes(repository_root, ["cat-file", "-e", f"{trees[0]}^{{tree}}"])
    return trees[0], tuple(parents)


def _raw_ancestor_trees(
    repository_root: Path, commit_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Traverse every raw parent edge and require every object to be local."""

    pending = [commit_id]
    seen: set[str] = set()
    trees: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        tree, parents = _raw_commit_identity(repository_root, current)
        seen.add(current)
        trees.add(tree)
        pending.extend(reversed(parents))
    if commit_id not in seen or not seen or not trees:
        raise GenesisSecretAbsenceError("raw ancestor traversal is empty")
    return tuple(sorted(seen)), tuple(sorted(trees))


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


def _magic_header_starts_record(blob: bytes, header: bytes) -> bool:
    """Return whether ``header`` starts a CR/LF-delimited logical record.

    Private-key headers are serialized record headers, not arbitrary byte
    substrings.  Anchoring the match prevents source-code assertions and other
    quoted examples from becoming findings merely because they mention a
    header in the middle of a line.  The loop deliberately continues past a
    mid-record occurrence because the same blob may contain a later, genuine
    record-start occurrence.
    """

    offset = blob.find(header)
    while offset >= 0:
        line_start = max(blob.rfind(b"\n", 0, offset), blob.rfind(b"\r", 0, offset)) + 1
        if all(byte in (0x09, 0x20) for byte in blob[line_start:offset]):
            return True
        offset = blob.find(header, offset + 1)
    return False


def _private_key_magic_hit(blob: bytes, header_id: str, header: bytes) -> bool:
    """Match a record header, a complete embedded block, or non-PEM magic."""

    if _magic_header_starts_record(blob, header):
        return True
    if header_id in PRIVATE_KEY_ANY_OFFSET_IDS:
        return header in blob
    footer = PRIVATE_KEY_MAGIC_FOOTERS.get(header_id)
    if footer is None:
        raise AssertionError(f"private-key footer registry is incomplete: {header_id}")
    offset = blob.find(header)
    while offset >= 0:
        if blob.find(footer, offset + len(header)) >= 0:
            return True
        offset = blob.find(header, offset + 1)
    return False


def _tree_blob_paths(
    repository_root: Path,
    tree_ids: tuple[str, ...],
) -> tuple[dict[str, set[bytes]], int, int]:
    paths_by_blob: dict[str, set[bytes]] = {}
    observation_count = 0
    non_blob_entry_count = 0
    for tree_id in tree_ids:
        output = _git_bytes(
            repository_root,
            [
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                tree_id,
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
        "private_key_magic_header_match_rule": (
            "header begins at byte offset zero or after CR or LF plus optional "
            "horizontal ASCII whitespace; an isolated mid-record PEM/OpenPGP "
            "header is not a finding under this record-only rule"
        ),
        "private_key_complete_block_match_rule": (
            "a PEM/OpenPGP header followed by its matching footer is a finding "
            "at any byte offsets, including escaped structured-text strings; "
            "age, PuTTY, and binary OpenSSH magic are findings at any offset"
        ),
        "private_key_magic_header_residual_boundary": (
            "an isolated inline PEM/OpenPGP header example without a matching "
            "footer is outside the header/block rules unless another frozen "
            "filename or JSON-key rule independently matches"
        ),
        "history_enumeration_rule": (
            "raw cat-file commit parent traversal plus every unique raw ancestor "
            "tree; revision-walk graft semantics are not consulted"
        ),
        "history_completeness_rule": (
            "reject shallow, promisor, partial-clone, alternate-object, graft, "
            "and replace-ref metadata; every raw parent/tree/blob must exist locally"
        ),
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


def _scan_once(commit_id: str, *, repository_root: Path) -> dict[str, object]:
    _reject_incomplete_or_rewritten_history(repository_root)
    ancestor_ids, ancestor_tree_ids = _raw_ancestor_trees(repository_root, commit_id)
    paths_by_blob, observations, non_blob_entries = _tree_blob_paths(
        repository_root, ancestor_tree_ids
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
            if _private_key_magic_hit(blob, header_id, header):
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
        "path_state_commit_count": len(ancestor_tree_ids),
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


def _repository_genesis_secret_absence_report_v1(
    commit_id: str, *, repository_root: Path
) -> dict[str, object]:
    audited_commit = _require_commit_id(commit_id)
    first = _scan_once(audited_commit, repository_root=repository_root)
    second = _scan_once(audited_commit, repository_root=repository_root)
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


def repository_genesis_secret_absence_report(
    commit_id: str,
) -> dict[str, object]:
    """Audit the configured repository commit plus every raw ancestor."""

    return _repository_genesis_secret_absence_report_v1(
        commit_id, repository_root=_repository_root()
    )


def repository_genesis_secret_absence_report_for_repository_v1(
    repository_root: Path,
    project_root: Path,
    commit_id: str,
) -> dict[str, object]:
    """Replay the receipt against an explicitly supplied repository.

    This entry point exists for Commit-B post-commit verification.  It never
    consults this module's configured worktree and returns the same path-free
    receipt bytes as the default wrapper for the same Git object graph.
    """

    repository, _project = _validated_repository_scope_v1(
        repository_root, project_root
    )
    return _repository_genesis_secret_absence_report_v1(
        commit_id, repository_root=repository
    )


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
    "repository_genesis_secret_absence_report_for_repository_v1",
    "validate_repository_genesis_secret_absence_report",
]

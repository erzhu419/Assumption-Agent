"""One-way, gold-safe custody export for fresh MuSiQue development.

The formal entry point first authenticates every published public trust anchor
and only then opens the exact private ``development.jsonl`` path.  It never
discovers a sibling.  Six anonymous, runner-native gold-free inputs are
materialized separately from a 0600 private offline-evaluator index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

# A checked-in script must work from an arbitrary cwd as well as through -m.
if __package__ in {None, ""}:  # pragma: no cover - exercised by subprocess tests
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "assumption_agent.benchmarks"

from ..models import stable_hash
from .musique_official_core_comparison_v1 import (
    ACQUISITION_SCHEMA,
    PRIVATE_PACK_SCHEMA,
    normalize_answer_primary,
    normalize_answer_secondary,
)
from .musique_typed_retriever_formation_v1 import (
    CLAIM_SCOPE,
    OPERATOR_VERSION,
    TypedRetrievalProgram,
    verify_live_implementation,
)


CUSTODY_VERSION = "musique_development_gold_safe_custody_v2"
PUBLIC_RECEIPT_SCHEMA = "musique_development_gold_safe_custody_receipt_v2"
GENERATION_ITEM_SCHEMA = "musique_homologous_three_arm_gold_free_input_v1"
EVALUATOR_INDEX_SCHEMA = "musique_development_private_evaluator_index_v2"
RUNNER_PRIVATE_INDEX_VERSION = "musique_homologous_three_arm_private_index_v1"
RUNNER_PRIVATE_INDEX_NAME = "private_index.runner.json"
PRIVATE_INDEX_BINDING_SCHEMA = "musique_development_private_index_public_binding_v1"
FORMAL_PUBLIC_CUSTODY_RECEIPT_RELATIVE = (
    "manifests/musique_development_gold_safe_custody_receipt_v2.json"
)
FORMAL_PUBLIC_PRIVATE_INDEX_BINDING_RELATIVE = (
    "manifests/musique_development_private_index_public_binding_v1.json"
)
DEVELOPMENT_IMPLEMENTATION_SCHEMA = "musique_development_implementation_set_v2"
DEVELOPMENT_ITEM_COUNT = 6
GENERATION_DIRECTORY = "generation"
EVALUATOR_INDEX_NAME = "evaluator.private.json"
MAX_SECURE_READ_BYTES = 64 * 1024 * 1024

MUSIQUE_OFFICIAL_COMMIT = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
HIPPORAG_OFFICIAL_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"

PUBLISHED_ANCHORS: dict[str, dict[str, str]] = {
    "preregistration": {
        "relative_path": "manifests/musique_official_core_comparison_v1_preregistration.json",
        "file_sha256": "65ac27e7100c6e371bc172f8e92129389f1880619a6fc6d6e04ffbcbcc42e078",
        "self_field": "preregistration_sha256",
        "self_sha256": "1ab838457ea8f3ef0c3daad56b6b056d3fbaf12450f3538aaf85a417cdacbce0",
    },
    "acquisition": {
        "relative_path": "manifests/musique_official_core_comparison_v1_acquisition.json",
        "file_sha256": "1f0ec3fa1b7c96c911659b2b41de534f135adccd6e8cb18b4c3bb7eb3572761c",
        "self_field": "acquisition_sha256",
        "self_sha256": "86cd288145d9b155659e2ae6854f6a7422d9ca2a5325842a90d3e8d5375cd742",
    },
    "formation": {
        "relative_path": "manifests/musique_typed_retriever_formation_result_v1.json",
        "file_sha256": "6d2dd85add45a3419627809304890266249a9027495751331c89c67a938b3ea1",
        "self_field": "receipt_hash",
        "self_sha256": "03731387335b4ad804b49c0a49399c4d2601653a01d97221f71a5e4029541f7e",
    },
    "program": {
        "relative_path": "manifests/musique_typed_retriever_program_v1.json",
        "file_sha256": "9028565df238a670b6a83bd6a7b7f369f559954b6c5df5bb5afc27c956cb555f",
        "self_field": "program_hash",
        "self_sha256": "a240449027544508748c09c4e6fd7124bf3ffab9fe231646a2e64a9b5f095119",
    },
    "qualification": {
        "relative_path": "manifests/official_hipporag_runtime_adapter_qualification_v1.json",
        "file_sha256": "89bbe828a5941197031baf01180f12d9ae82cb10ddbe7b6b371b59894aed1d90",
        "self_field": "qualification_sha256",
        "self_sha256": "c2a6b540e4b91347a23bbe918b495caebcc35a23fbacee9754cd1b7661fda4e4",
    },
    "official_adapter": {
        "relative_path": "manifests/musique_official_hipporag_retrieve_only_binding_v1.json",
        "file_sha256": "effdd1de442c88760d15950739d30479613cdcccba90166bcd7e622f6a7dd03b",
        "self_field": "receipt_sha256",
        "self_sha256": "522d31926df70f983ae2f644f05c9f3ee45fcd08e0d847642e144652df5a45d0",
    },
}

# Explicit closure of the real formal runner, official adapter, and imported
# package initializers.  This is intentionally fixed rather than discovered.
DEVELOPMENT_IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/benchmarks/musique_development_custody_v1.py",
    "assumption_agent/benchmarks/musique_development_freeze_v1.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_three_arm_formal_runner_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/benchmarks/official_hipporag_runtime_adapter_qualification_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/__init__.py",
    "replication_runtime/financial_semantic_v2/__init__.py",
    "replication_runtime/financial_semantic_v2/durable_state.py",
    "replication_runtime/musique_official_hipporag_v1/__init__.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
    "replication_runtime/noaa_gsod_v1/__init__.py",
    "replication_runtime/noaa_gsod_v1/acquire.py",
    "replication_runtime/noaa_gsod_v1/contract.py",
    "replication_runtime/noaa_gsod_v1/development_freeze.py",
    "replication_runtime/noaa_gsod_v1/development_implementation.py",
    "replication_runtime/noaa_gsod_v1/development_runner.py",
    "replication_runtime/noaa_gsod_v1/development_schemas.py",
    "replication_runtime/noaa_gsod_v1/development_source.py",
    "replication_runtime/noaa_gsod_v1/oracle_sqlite.py",
    "replication_runtime/noaa_gsod_v1/oracle_stdlib.py",
    "replication_runtime/noaa_gsod_v1/pack.py",
    "replication_runtime/noaa_gsod_v1/schemas.py",
    "replication_runtime/noaa_gsod_v1/train_export.py",
    "replication_runtime/noaa_gsod_v1/train_schemas.py",
    "replication_runtime/noaa_gsod_v1/typed_relational.py",
)
_SHA256_CHARS = frozenset("0123456789abcdef")


class MuSiQueDevelopmentCustodyError(RuntimeError):
    """The one-way development custody contract failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise MuSiQueDevelopmentCustodyError(f"{field} must be lowercase SHA-256")
    return value


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: str | Path, field: str) -> Path:
    candidate = _absolute_lexical(path)
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueDevelopmentCustodyError(
                f"{field} contains a symbolic-link component"
            )
    return candidate


def _secure_read_bytes(
    path: str | Path,
    *,
    field: str,
    maximum_bytes: int = MAX_SECURE_READ_BYTES,
) -> bytes:
    """Read through pinned O_NOFOLLOW directory descriptors and recheck inode."""

    absolute = _absolute_lexical(path)
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        raise MuSiQueDevelopmentCustodyError(f"{field} path is invalid")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(parts[0], directory_flags | nofollow)
        descriptors.append(current)
        for component in parts[1:-1]:
            current = os.open(
                component,
                directory_flags | nofollow,
                dir_fd=current,
            )
            descriptors.append(current)
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
            dir_fd=current,
        )
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise MuSiQueDevelopmentCustodyError(f"{field} is not a bounded regular file")
        chunks: list[bytes] = []
        observed = 0
        while True:
            block = os.read(file_descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
            if not block:
                break
            chunks.append(block)
            observed += len(block)
            if observed > maximum_bytes:
                raise MuSiQueDevelopmentCustodyError(f"{field} exceeds its byte bound")
        after = os.fstat(file_descriptor)
        lexical = os.stat(parts[-1], dir_fd=descriptors[-2], follow_symlinks=False)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or (after.st_dev, after.st_ino) != (
            lexical.st_dev,
            lexical.st_ino,
        ):
            raise MuSiQueDevelopmentCustodyError(f"{field} changed during secure read")
        return b"".join(chunks)
    except (OSError, ValueError) as exc:
        raise MuSiQueDevelopmentCustodyError(f"{field} could not be securely read") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _secure_json(path: str | Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    raw = _secure_read_bytes(path, field=field)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MuSiQueDevelopmentCustodyError(f"{field} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise MuSiQueDevelopmentCustodyError(f"{field} must be a JSON object")
    return value, raw


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(_secure_read_bytes(path, field="bound file"))


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _containing_git_repository(path: Path) -> Path | None:
    anchor = path if path.is_dir() else path.parent
    while not anchor.exists():
        if anchor.parent == anchor:
            return None
        anchor = anchor.parent
    completed = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve(strict=True)


def _require_ignored_untracked_if_in_repository(
    path: Path, field: str, *, directory: bool = False
) -> None:
    repository = _containing_git_repository(path)
    if repository is None:
        return
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(repository).as_posix()
    except ValueError as exc:
        raise MuSiQueDevelopmentCustodyError(
            f"{field} repository containment is ambiguous"
        ) from exc
    ignore_candidate = f"{relative.rstrip('/')}/" if directory else relative
    ignored = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--no-index", "-q", "--", ignore_candidate],
        check=False,
        capture_output=True,
        timeout=30,
    )
    tracked = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--", relative],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if ignored.returncode != 0 or tracked.returncode != 0 or tracked.stdout.strip():
        raise MuSiQueDevelopmentCustodyError(
            f"{field} must be ignored and untracked inside a repository"
        )


def current_development_implementation_binding(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(__file__).resolve(strict=True).parents[2]
        if project_root is None
        else Path(project_root).resolve(strict=True)
    )
    rows = []
    for relative in DEVELOPMENT_IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        raw = _secure_read_bytes(path, field=f"implementation file {relative}")
        rows.append({"path": relative, "sha256": _sha256_bytes(raw)})
    return {
        "schema": DEVELOPMENT_IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, expected: str | None = None
) -> str:
    declared = _require_sha256(payload.get(field), field)
    body = dict(payload)
    body.pop(field, None)
    if stable_hash(body) != declared or (expected is not None and declared != expected):
        raise MuSiQueDevelopmentCustodyError(f"{field} self-hash mismatch")
    return declared


def verify_formal_anchor_bundle(
    *,
    preregistration_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    qualification_path: str | Path,
    official_adapter_binding_path: str | Path,
) -> dict[str, Any]:
    """Authenticate every published trust root without touching development."""

    project = Path(__file__).resolve(strict=True).parents[2]
    supplied = {
        "preregistration": preregistration_path,
        "acquisition": acquisition_receipt_path,
        "formation": formation_receipt_path,
        "program": frozen_program_path,
        "qualification": qualification_path,
        "official_adapter": official_adapter_binding_path,
    }
    payloads: dict[str, dict[str, Any]] = {}
    file_hashes: dict[str, str] = {}
    for name, requested in supplied.items():
        specification = PUBLISHED_ANCHORS[name]
        expected_path = (project / specification["relative_path"]).resolve(strict=True)
        lexical = _reject_symlink_components(requested, f"published {name}")
        if lexical.resolve(strict=True) != expected_path:
            raise MuSiQueDevelopmentCustodyError(
                f"published {name} path is not the registered trust root"
            )
        payload, raw = _secure_json(expected_path, field=f"published {name}")
        observed_hash = _sha256_bytes(raw)
        if observed_hash != specification["file_sha256"]:
            raise MuSiQueDevelopmentCustodyError(f"published {name} file hash drifted")
        self_field = specification["self_field"]
        if name == "program":
            if payload.get(self_field) != specification["self_sha256"]:
                raise MuSiQueDevelopmentCustodyError("published program hash drifted")
        else:
            _verify_self_hash(
                payload,
                field=self_field,
                expected=specification["self_sha256"],
            )
        payloads[name] = payload
        file_hashes[name] = observed_hash

    preregistration = payloads["preregistration"]
    acquisition = payloads["acquisition"]
    formation = payloads["formation"]
    program = payloads["program"]
    qualification = payloads["qualification"]
    adapter = payloads["official_adapter"]
    if (
        preregistration.get("source", {}).get("commit") != MUSIQUE_OFFICIAL_COMMIT
        or acquisition.get("source", {}).get("commit") != MUSIQUE_OFFICIAL_COMMIT
        or acquisition.get("ordering", {}).get("preregistration_sha256")
        != PUBLISHED_ANCHORS["preregistration"]["self_sha256"]
        or formation.get("source_binding", {}).get("acquisition_sha256")
        != PUBLISHED_ANCHORS["acquisition"]["self_sha256"]
        or program.get("formation_receipt_hash")
        != PUBLISHED_ANCHORS["formation"]["self_sha256"]
        or formation.get("selection_receipt", {}).get("selected_program_hash")
        != PUBLISHED_ANCHORS["program"]["self_sha256"]
        or qualification.get("source_binding", {}).get("commit")
        != HIPPORAG_OFFICIAL_COMMIT
        or adapter.get("official_source_binding", {}).get("commit")
        != HIPPORAG_OFFICIAL_COMMIT
        or adapter.get("qualification_binding", {}).get("file_sha256")
        != PUBLISHED_ANCHORS["qualification"]["file_sha256"]
        or adapter.get("qualification_binding", {}).get("qualification_sha256")
        != PUBLISHED_ANCHORS["qualification"]["self_sha256"]
    ):
        raise MuSiQueDevelopmentCustodyError("published anchor chain is inconsistent")
    return {"payloads": payloads, "file_hashes": file_hashes}


def _verify_acquisition_payload(
    payload: Mapping[str, Any], raw: bytes
) -> tuple[dict[str, Any], dict[str, Any]]:
    acquisition = dict(payload)
    if acquisition.get("schema") != ACQUISITION_SCHEMA:
        raise MuSiQueDevelopmentCustodyError("acquisition schema mismatch")
    _verify_self_hash(acquisition, field="acquisition_sha256")
    if acquisition.get("decision") != "private_pack_formed_no_model_execution_authorized":
        raise MuSiQueDevelopmentCustodyError("acquisition decision mismatch")
    if acquisition.get("source", {}).get("claim_scope") != CLAIM_SCOPE:
        raise MuSiQueDevelopmentCustodyError("acquisition claim scope mismatch")
    counts = acquisition.get("counts")
    commitments = acquisition.get("commitments")
    if not isinstance(counts, Mapping) or not isinstance(commitments, Mapping):
        raise MuSiQueDevelopmentCustodyError("acquisition envelope malformed")
    split_counts = counts.get("splits")
    split_files = commitments.get("split_files")
    if (
        not isinstance(split_counts, Mapping)
        or set(split_counts) != {"train", "development", "residual_sealed"}
        or split_counts.get("development") != DEVELOPMENT_ITEM_COUNT
        or not isinstance(split_files, list)
        or len(split_files) != 3
    ):
        raise MuSiQueDevelopmentCustodyError("acquisition split contract mismatch")
    seen: set[str] = set()
    development_rows: list[dict[str, Any]] = []
    for row in split_files:
        if not isinstance(row, dict) or set(row) != {
            "split", "count", "file_sha256", "item_commitment_set_sha256"
        }:
            raise MuSiQueDevelopmentCustodyError("acquisition split binding malformed")
        split = row.get("split")
        if not isinstance(split, str) or split in seen or split_counts.get(split) != row.get("count"):
            raise MuSiQueDevelopmentCustodyError("acquisition split binding mismatch")
        seen.add(split)
        _require_sha256(row.get("file_sha256"), "split file")
        _require_sha256(row.get("item_commitment_set_sha256"), "split item set")
        if split == "development":
            development_rows.append(row)
    if seen != set(split_counts) or len(development_rows) != 1:
        raise MuSiQueDevelopmentCustodyError("development split binding is ambiguous")
    if stable_hash(split_files) != commitments.get("private_pack_sha256"):
        raise MuSiQueDevelopmentCustodyError("private pack commitment mismatch")
    if counts.get("selected_rows") != sum(split_counts.values()):
        raise MuSiQueDevelopmentCustodyError("selected row count mismatch")
    if _sha256_bytes(raw) == "0" * 64:  # unreachable defensive shape check
        raise MuSiQueDevelopmentCustodyError("acquisition file hash is invalid")
    return acquisition, development_rows[0]


def _verify_acquisition_receipt(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Public exact-path verifier consumed by the formal runner."""

    payload, raw = _secure_json(path, field="acquisition receipt")
    return _verify_acquisition_payload(payload, raw)


def _verify_formation_payloads(
    *,
    acquisition: Mapping[str, Any],
    formation_payload: Mapping[str, Any],
    formation_raw: bytes,
    program_payload: Mapping[str, Any],
    program_raw: bytes,
) -> dict[str, str]:
    receipt = dict(formation_payload)
    if receipt.get("schema") != "musique_typed_retriever_formation_v1_receipt":
        raise MuSiQueDevelopmentCustodyError("formation receipt schema mismatch")
    _verify_self_hash(receipt, field="receipt_hash")
    if (
        receipt.get("raw_content_persisted") is not False
        or receipt.get("source_binding", {}).get("acquisition_sha256")
        != acquisition.get("acquisition_sha256")
        or receipt.get("source_binding", {}).get("private_pack_sha256")
        != acquisition.get("commitments", {}).get("private_pack_sha256")
        or receipt.get("offline_contract", {}).get("partition") != "train"
        or receipt.get("offline_contract", {}).get("development_execution_authorized") is not False
        or receipt.get("offline_contract", {}).get("sealed_execution_authorized") is not False
        or any(
            receipt.get("offline_contract", {}).get(field) != 0
            for field in ("model_calls", "network_calls", "online_evaluator_calls")
        )
    ):
        raise MuSiQueDevelopmentCustodyError("formation is not exact TRAIN-only output")
    expected_program_fields = {
        "operator_version", "implementation", "program", "program_hash",
        "formation_receipt_hash", "raw_content_persisted"
    }
    if (
        set(program_payload) != expected_program_fields
        or program_payload.get("operator_version") != OPERATOR_VERSION
        or program_payload.get("raw_content_persisted") is not False
        or program_payload.get("formation_receipt_hash") != receipt.get("receipt_hash")
        or program_payload.get("implementation") != receipt.get("implementation")
    ):
        raise MuSiQueDevelopmentCustodyError("frozen program envelope mismatch")
    try:
        program = TypedRetrievalProgram.from_dict(program_payload["program"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MuSiQueDevelopmentCustodyError("frozen typed program malformed") from exc
    if (
        program.to_dict() != program_payload.get("program")
        or program.type_issues()
        or program.program_hash != program_payload.get("program_hash")
        or receipt.get("selection_receipt", {}).get("selected_program_hash")
        != program.program_hash
    ):
        raise MuSiQueDevelopmentCustodyError("frozen typed program drifted")
    implementation = receipt.get("implementation")
    if not isinstance(implementation, Mapping):
        raise MuSiQueDevelopmentCustodyError("formation implementation missing")
    verify_live_implementation(implementation)
    return {
        "formation_receipt_file_sha256": _sha256_bytes(formation_raw),
        "formation_receipt_hash": _require_sha256(receipt.get("receipt_hash"), "formation receipt"),
        "formation_implementation_set_sha256": _require_sha256(
            implementation.get("set_sha256"), "formation implementation"
        ),
        "frozen_program_file_sha256": _sha256_bytes(program_raw),
        "frozen_program_hash": program.program_hash,
    }


def _parse_exact_development(
    raw: bytes, binding: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _sha256_bytes(raw) != binding.get("file_sha256"):
        raise MuSiQueDevelopmentCustodyError("exact development file hash mismatch")
    if not raw or not raw.endswith(b"\n"):
        raise MuSiQueDevelopmentCustodyError("development JSONL framing mismatch")
    rows = []
    for line in raw.splitlines():
        if not line:
            raise MuSiQueDevelopmentCustodyError("blank development JSONL row")
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise MuSiQueDevelopmentCustodyError("development row must be an object")
        rows.append(value)
    if len(rows) != DEVELOPMENT_ITEM_COUNT or len(rows) != binding.get("count"):
        raise MuSiQueDevelopmentCustodyError("development row count mismatch")
    if raw != b"".join(_canonical_bytes(row) + b"\n" for row in rows):
        raise MuSiQueDevelopmentCustodyError("development JSONL is not canonical")

    expected_fields = {
        "schema", "split", "item_id", "question", "corpus", "answers",
        "normalized_answers", "support_indices", "source_row_sha256"
    }
    generation_rows = []
    evaluator_rows = []
    commitments = []
    source_ids: set[str] = set()
    for ordinal, row in enumerate(rows):
        if set(row) != expected_fields or row.get("schema") != PRIVATE_PACK_SCHEMA or row.get("split") != "development":
            raise MuSiQueDevelopmentCustodyError("development row schema mismatch")
        source_id = row.get("item_id")
        question = row.get("question")
        answers = row.get("answers")
        normalized = row.get("normalized_answers")
        support = row.get("support_indices")
        if (
            not isinstance(source_id, str) or not source_id or source_id in source_ids
            or not isinstance(question, str) or not question.strip()
            or not isinstance(answers, list) or not answers
            or any(not isinstance(value, str) for value in answers)
            or not isinstance(normalized, list) or len(normalized) < 2
            or any(not isinstance(value, str) or not value for value in normalized)
            or not isinstance(support, list) or any(type(value) is not int for value in support)
        ):
            raise MuSiQueDevelopmentCustodyError("development private fields malformed")
        source_ids.add(source_id)
        _require_sha256(row.get("source_row_sha256"), "source row")
        primary = list(dict.fromkeys(value for answer in answers if (value := normalize_answer_primary(answer))))
        secondary = list(dict.fromkeys(value for answer in answers if (value := normalize_answer_secondary(answer))))
        if primary != secondary or normalized != primary:
            raise MuSiQueDevelopmentCustodyError("answer normalization drifted")
        corpus = row.get("corpus")
        if not isinstance(corpus, list) or len(corpus) < 5:
            raise MuSiQueDevelopmentCustodyError("candidate corpus malformed")
        gold_free = []
        observed_support = []
        for index, paragraph in enumerate(corpus):
            if not isinstance(paragraph, dict) or set(paragraph) != {"idx", "title", "text", "is_supporting"}:
                raise MuSiQueDevelopmentCustodyError("paragraph schema mismatch")
            if (
                paragraph.get("idx") != index
                or not isinstance(paragraph.get("title"), str) or not paragraph["title"].strip()
                or not isinstance(paragraph.get("text"), str) or not paragraph["text"].strip()
                or type(paragraph.get("is_supporting")) is not bool
            ):
                raise MuSiQueDevelopmentCustodyError("paragraph value malformed")
            gold_free.append({
                "idx": index,
                "title": paragraph["title"],
                "paragraph_text": paragraph["text"],
            })
            if paragraph["is_supporting"]:
                observed_support.append(index)
        if support != observed_support or len(support) < 2:
            raise MuSiQueDevelopmentCustodyError("support label drifted")
        anonymous = f"development_item_{ordinal:02d}"
        generation_rows.append({
            "schema": GENERATION_ITEM_SCHEMA,
            "anonymous_item_id": anonymous,
            "question": question,
            "corpus": gold_free,
        })
        evaluator_rows.append({
            "anonymous_item_id": anonymous,
            "answers": answers,
            "normalized_answers": normalized,
            "support_indices": support,
        })
        commitments.append(stable_hash(row))
    if stable_hash(commitments) != binding.get("item_commitment_set_sha256"):
        raise MuSiQueDevelopmentCustodyError("development item set mismatch")
    return generation_rows, evaluator_rows


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if mode == 0o600 else 0o755)
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def verify_public_custody_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    if set(receipt) != {"schema", "hashes", "counts", "receipt_sha256"} or receipt.get("schema") != PUBLIC_RECEIPT_SCHEMA:
        raise MuSiQueDevelopmentCustodyError("public custody receipt schema mismatch")
    _verify_self_hash(receipt, field="receipt_sha256")
    hashes = receipt.get("hashes")
    expected_hash_fields = {
        "acquisition_receipt_file_sha256", "acquisition_sha256",
        "development_file_sha256", "development_item_commitment_set_sha256",
        "development_implementation_set_sha256",
        "formation_implementation_set_sha256", "formation_receipt_file_sha256",
        "formation_receipt_hash", "frozen_program_file_sha256", "frozen_program_hash",
        "generation_view_set_sha256", "musique_official_commit_sha256",
        "official_adapter_binding_file_sha256", "official_adapter_binding_receipt_sha256",
        "official_hipporag_commit_sha256", "preregistration_file_sha256",
        "preregistration_sha256", "private_pack_sha256",
        "qualification_file_sha256", "qualification_sha256",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_hash_fields:
        raise MuSiQueDevelopmentCustodyError("public custody hash set mismatch")
    for field, value in hashes.items():
        _require_sha256(value, field)
    if receipt.get("counts") != {
        "development_items": 6,
        "generation_files": 6,
        "private_evaluator_items": 6,
        "runner_private_index_items": 6,
    }:
        raise MuSiQueDevelopmentCustodyError("public custody counts mismatch")
    serialized = json.dumps(receipt, ensure_ascii=True, sort_keys=True)
    for forbidden in ('"question"', '"corpus"', '"answers"', '"support_indices"', '"item_id"', "/artifacts/"):
        if forbidden in serialized:
            raise MuSiQueDevelopmentCustodyError("public custody receipt leaks content")
    if any(f"development_item_{ordinal:02d}" in serialized for ordinal in range(6)):
        raise MuSiQueDevelopmentCustodyError("public custody receipt leaks item identity")
    return receipt


def load_public_custody_receipt(path: str | Path) -> dict[str, Any]:
    payload, _ = _secure_json(path, field="public custody receipt")
    return verify_public_custody_receipt(payload)


def verify_public_private_index_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(payload)
    expected = {
        "schema", "custody_receipt_sha256", "private_index_file_sha256",
        "private_index_hash", "item_count", "binding_sha256"
    }
    if set(binding) != expected or binding.get("schema") != PRIVATE_INDEX_BINDING_SCHEMA:
        raise MuSiQueDevelopmentCustodyError("private-index public binding schema mismatch")
    _verify_self_hash(binding, field="binding_sha256")
    for field in (
        "custody_receipt_sha256", "private_index_file_sha256", "private_index_hash"
    ):
        _require_sha256(binding.get(field), field)
    if binding.get("item_count") != 6:
        raise MuSiQueDevelopmentCustodyError("private-index public binding count mismatch")
    return binding


def load_public_private_index_binding(path: str | Path) -> dict[str, Any]:
    payload, _ = _secure_json(path, field="public private-index binding")
    return verify_public_private_index_binding(payload)


def load_generation_item(source_view_root: str | Path, ordinal: int) -> dict[str, Any]:
    if type(ordinal) is not int or not 0 <= ordinal < 6:
        raise MuSiQueDevelopmentCustodyError("generation ordinal out of range")
    path = _absolute_lexical(source_view_root) / GENERATION_DIRECTORY / f"development_item_{ordinal:02d}.json"
    payload, _ = _secure_json(path, field="gold-free generation item")
    expected_id = f"development_item_{ordinal:02d}"
    if (
        set(payload) != {"schema", "anonymous_item_id", "question", "corpus"}
        or payload.get("schema") != GENERATION_ITEM_SCHEMA
        or payload.get("anonymous_item_id") != expected_id
        or not isinstance(payload.get("question"), str) or not payload["question"].strip()
        or not isinstance(payload.get("corpus"), list) or len(payload["corpus"]) < 5
    ):
        raise MuSiQueDevelopmentCustodyError("generation item schema mismatch")
    for index, paragraph in enumerate(payload["corpus"]):
        if (
            not isinstance(paragraph, dict)
            or set(paragraph) != {"idx", "title", "paragraph_text"}
            or paragraph.get("idx") != index
            or not isinstance(paragraph.get("title"), str) or not paragraph["title"].strip()
            or not isinstance(paragraph.get("paragraph_text"), str) or not paragraph["paragraph_text"].strip()
        ):
            raise MuSiQueDevelopmentCustodyError("generation corpus schema mismatch")
    return payload


def load_private_evaluator_index(source_view_root: str | Path) -> dict[str, Any]:
    path = _absolute_lexical(source_view_root) / EVALUATOR_INDEX_NAME
    payload, _ = _secure_json(path, field="private evaluator index")
    declared = payload.get("index_sha256")
    body = dict(payload)
    body.pop("index_sha256", None)
    if (
        set(payload) != {
            "schema", "partition", "role", "custody_receipt_sha256",
            "items", "item_count", "gold_released", "index_sha256"
        }
        or payload.get("schema") != EVALUATOR_INDEX_SCHEMA
        or payload.get("partition") != "development"
        or payload.get("item_count") != 6
        or payload.get("gold_released") is not False
        or stable_hash(body) != declared
        or not isinstance(payload.get("custody_receipt_sha256"), str)
        or len(payload["custody_receipt_sha256"]) != 64
        or not isinstance(payload.get("items"), list)
        or len(payload["items"]) != 6
    ):
        raise MuSiQueDevelopmentCustodyError("private evaluator index mismatch")
    for ordinal, item in enumerate(payload["items"]):
        if (
            not isinstance(item, dict)
            or set(item) != {"anonymous_item_id", "answers", "normalized_answers", "support_indices"}
            or item.get("anonymous_item_id") != f"development_item_{ordinal:02d}"
            or not isinstance(item.get("answers"), list) or not item["answers"]
            or any(not isinstance(value, str) for value in item["answers"])
            or not isinstance(item.get("normalized_answers"), list) or not item["normalized_answers"]
            or not isinstance(item.get("support_indices"), list) or not item["support_indices"]
            or any(type(value) is not int for value in item["support_indices"])
        ):
            raise MuSiQueDevelopmentCustodyError("private evaluator item mismatch")
    return payload


def generation_view_set_sha256(source_view_root: str | Path) -> str:
    root = _absolute_lexical(source_view_root)
    rows = []
    for ordinal in range(6):
        item = load_generation_item(root, ordinal)
        raw = _secure_read_bytes(
            root / GENERATION_DIRECTORY / f"development_item_{ordinal:02d}.json",
            field="gold-free generation item",
        )
        rows.append({"anonymous_item_id": item["anonymous_item_id"], "file_sha256": _sha256_bytes(raw)})
    return stable_hash(rows)


def _export(
    *,
    development_jsonl_path: str | Path,
    preregistration_path: str | Path | None,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    qualification_path: str | Path | None,
    official_adapter_binding_path: str | Path | None,
    source_view_root: str | Path,
    public_receipt_path: str | Path,
    public_private_index_binding_path: str | Path,
    formal: bool,
) -> dict[str, Any]:
    development = _absolute_lexical(development_jsonl_path)
    if development.name != "development.jsonl":
        raise MuSiQueDevelopmentCustodyError("only exact development.jsonl is accepted")
    root = _reject_symlink_components(source_view_root, "source-view root")
    public = _reject_symlink_components(public_receipt_path, "public receipt")
    private_binding_public = _reject_symlink_components(
        public_private_index_binding_path, "public private-index binding"
    )
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if public.exists() or public.is_symlink():
        raise FileExistsError(public)
    if private_binding_public.exists() or private_binding_public.is_symlink():
        raise FileExistsError(private_binding_public)
    _require_ignored_untracked_if_in_repository(development, "exact development file")
    _require_ignored_untracked_if_in_repository(root, "source-view root", directory=True)

    # Formal trust anchors are authenticated before the first development open.
    if formal:
        project = Path(__file__).resolve(strict=True).parents[2]
        if (
            public != project / FORMAL_PUBLIC_CUSTODY_RECEIPT_RELATIVE
            or private_binding_public
            != project / FORMAL_PUBLIC_PRIVATE_INDEX_BINDING_RELATIVE
        ):
            raise MuSiQueDevelopmentCustodyError(
                "formal public custody outputs must use registered manifest paths"
            )
        if preregistration_path is None or qualification_path is None or official_adapter_binding_path is None:
            raise MuSiQueDevelopmentCustodyError("formal published anchor paths are required")
        anchors = verify_formal_anchor_bundle(
            preregistration_path=preregistration_path,
            acquisition_receipt_path=acquisition_receipt_path,
            formation_receipt_path=formation_receipt_path,
            frozen_program_path=frozen_program_path,
            qualification_path=qualification_path,
            official_adapter_binding_path=official_adapter_binding_path,
        )
        acquisition = anchors["payloads"]["acquisition"]
        acquisition_raw = _secure_read_bytes(acquisition_receipt_path, field="published acquisition")
        formation_payload = anchors["payloads"]["formation"]
        formation_raw = _secure_read_bytes(formation_receipt_path, field="published formation")
        program_payload = anchors["payloads"]["program"]
        program_raw = _secure_read_bytes(frozen_program_path, field="published program")
        anchor_hashes = anchors["file_hashes"]
    else:
        acquisition, acquisition_raw = _secure_json(acquisition_receipt_path, field="synthetic acquisition")
        formation_payload, formation_raw = _secure_json(formation_receipt_path, field="synthetic formation")
        program_payload, program_raw = _secure_json(frozen_program_path, field="synthetic program")
        anchor_hashes = {
            "preregistration": "0" * 64,
            "qualification": "0" * 64,
            "official_adapter": "0" * 64,
        }
    acquisition, development_binding = _verify_acquisition_payload(acquisition, acquisition_raw)
    formation = _verify_formation_payloads(
        acquisition=acquisition,
        formation_payload=formation_payload,
        formation_raw=formation_raw,
        program_payload=program_payload,
        program_raw=program_raw,
    )
    development_raw = _secure_read_bytes(development, field="exact development JSONL")
    generation_rows, evaluator_rows = _parse_exact_development(development_raw, development_binding)
    implementation = current_development_implementation_binding()
    root_resolved = root.resolve(strict=False)
    public_resolved = public.resolve(strict=False)
    private_binding_resolved = private_binding_public.resolve(strict=False)
    if (
        _paths_overlap(root_resolved, public_resolved)
        or _paths_overlap(root_resolved, private_binding_resolved)
        or _paths_overlap(public_resolved, private_binding_resolved)
        or any(
        _paths_overlap(root_resolved, _absolute_lexical(path).resolve(strict=True))
        for path in (development, acquisition_receipt_path, formation_receipt_path, frozen_program_path)
        )
    ):
        raise MuSiQueDevelopmentCustodyError("custody paths overlap")

    try:
        root.mkdir(parents=True, mode=0o700)
        os.chmod(root, 0o700)
        generation_root = root / GENERATION_DIRECTORY
        generation_root.mkdir(mode=0o700)
        for ordinal, row in enumerate(generation_rows):
            _write_json_exclusive(generation_root / f"development_item_{ordinal:02d}.json", row, mode=0o600)
        adapter_payload = (
            anchors["payloads"]["official_adapter"] if formal else {"receipt_sha256": "0" * 64}
        )
        receipt_body = {
            "schema": PUBLIC_RECEIPT_SCHEMA,
            "hashes": {
                "acquisition_receipt_file_sha256": _sha256_bytes(acquisition_raw),
                "acquisition_sha256": acquisition["acquisition_sha256"],
                "development_file_sha256": _sha256_bytes(development_raw),
                "development_item_commitment_set_sha256": development_binding["item_commitment_set_sha256"],
                "development_implementation_set_sha256": implementation["set_sha256"],
                **formation,
                "generation_view_set_sha256": generation_view_set_sha256(root),
                "musique_official_commit_sha256": _sha256_bytes(MUSIQUE_OFFICIAL_COMMIT.encode()),
                "official_adapter_binding_file_sha256": anchor_hashes["official_adapter"],
                "official_adapter_binding_receipt_sha256": adapter_payload["receipt_sha256"],
                "official_hipporag_commit_sha256": _sha256_bytes(HIPPORAG_OFFICIAL_COMMIT.encode()),
                "preregistration_file_sha256": anchor_hashes["preregistration"],
                "preregistration_sha256": (
                    PUBLISHED_ANCHORS["preregistration"]["self_sha256"] if formal else "0" * 64
                ),
                "private_pack_sha256": acquisition["commitments"]["private_pack_sha256"],
                "qualification_file_sha256": anchor_hashes["qualification"],
                "qualification_sha256": (
                    PUBLISHED_ANCHORS["qualification"]["self_sha256"] if formal else "0" * 64
                ),
            },
            "counts": {
                "development_items": 6,
                "generation_files": 6,
                "private_evaluator_items": 6,
                "runner_private_index_items": 6,
            },
        }
        receipt = {**receipt_body, "receipt_sha256": stable_hash(receipt_body)}
        verify_public_custody_receipt(receipt)
        _write_json_exclusive(public, receipt, mode=0o644)
        evaluator_body = {
            "schema": EVALUATOR_INDEX_SCHEMA,
            "partition": "development",
            "role": "offline_gold_release_after_18_terminal_joins_only",
            "custody_receipt_sha256": receipt["receipt_sha256"],
            "items": evaluator_rows,
            "item_count": 6,
            "gold_released": False,
        }
        evaluator = {**evaluator_body, "index_sha256": stable_hash(evaluator_body)}
        evaluator_path = root / EVALUATOR_INDEX_NAME
        _write_json_exclusive(evaluator_path, evaluator, mode=0o600)
        if stat.S_IMODE(evaluator_path.stat().st_mode) & 0o077:
            raise MuSiQueDevelopmentCustodyError("evaluator permissions are too broad")
        runner_private_body = {
            "private_index_version": RUNNER_PRIVATE_INDEX_VERSION,
            "custody_receipt_sha256": receipt["receipt_sha256"],
            "items": [
                {
                    "anonymous_item_id": row["anonymous_item_id"],
                    "accepted_aliases": row["answers"],
                    "support_indices": row["support_indices"],
                }
                for row in evaluator_rows
            ],
        }
        runner_private = {
            **runner_private_body,
            "private_index_hash": stable_hash(runner_private_body),
        }
        runner_private_path = root / RUNNER_PRIVATE_INDEX_NAME
        _write_json_exclusive(runner_private_path, runner_private, mode=0o600)
        if stat.S_IMODE(runner_private_path.stat().st_mode) & 0o077:
            raise MuSiQueDevelopmentCustodyError("runner private index permissions are too broad")
        private_binding_body = {
            "schema": PRIVATE_INDEX_BINDING_SCHEMA,
            "custody_receipt_sha256": receipt["receipt_sha256"],
            "private_index_file_sha256": _sha256_file(runner_private_path),
            "private_index_hash": runner_private["private_index_hash"],
            "item_count": 6,
        }
        private_binding = {
            **private_binding_body,
            "binding_sha256": stable_hash(private_binding_body),
        }
        verify_public_private_index_binding(private_binding)
        _write_json_exclusive(private_binding_public, private_binding, mode=0o644)
    except BaseException:
        if root.exists() and root.is_dir() and not root.is_symlink():
            shutil.rmtree(root)
        if public.exists() and public.is_file() and not public.is_symlink():
            public.unlink()
        if (
            private_binding_public.exists()
            and private_binding_public.is_file()
            and not private_binding_public.is_symlink()
        ):
            private_binding_public.unlink()
        raise
    return receipt


def export_development_source_view(
    *,
    development_jsonl_path: str | Path,
    preregistration_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    qualification_path: str | Path,
    official_adapter_binding_path: str | Path,
    source_view_root: str | Path,
    public_receipt_path: str | Path,
    public_private_index_binding_path: str | Path,
) -> dict[str, Any]:
    return _export(
        development_jsonl_path=development_jsonl_path,
        preregistration_path=preregistration_path,
        acquisition_receipt_path=acquisition_receipt_path,
        formation_receipt_path=formation_receipt_path,
        frozen_program_path=frozen_program_path,
        qualification_path=qualification_path,
        official_adapter_binding_path=official_adapter_binding_path,
        source_view_root=source_view_root,
        public_receipt_path=public_receipt_path,
        public_private_index_binding_path=public_private_index_binding_path,
        formal=True,
    )


def export_synthetic_development_source_view_for_tests(
    *,
    development_jsonl_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    source_view_root: str | Path,
    public_receipt_path: str | Path,
    public_private_index_binding_path: str | Path,
) -> dict[str, Any]:
    """Explicitly non-formal entry for generated fixtures only."""

    return _export(
        development_jsonl_path=development_jsonl_path,
        preregistration_path=None,
        acquisition_receipt_path=acquisition_receipt_path,
        formation_receipt_path=formation_receipt_path,
        frozen_program_path=frozen_program_path,
        qualification_path=None,
        official_adapter_binding_path=None,
        source_view_root=source_view_root,
        public_receipt_path=public_receipt_path,
        public_private_index_binding_path=public_private_index_binding_path,
        formal=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-jsonl", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--formation-receipt", type=Path, required=True)
    parser.add_argument("--frozen-program", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--official-adapter-binding", type=Path, required=True)
    parser.add_argument("--source-view-root", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    parser.add_argument("--public-private-index-binding", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = export_development_source_view(
        development_jsonl_path=arguments.development_jsonl,
        preregistration_path=arguments.preregistration,
        acquisition_receipt_path=arguments.acquisition_receipt,
        formation_receipt_path=arguments.formation_receipt,
        frozen_program_path=arguments.frozen_program,
        qualification_path=arguments.qualification,
        official_adapter_binding_path=arguments.official_adapter_binding,
        source_view_root=arguments.source_view_root,
        public_receipt_path=arguments.public_receipt,
        public_private_index_binding_path=arguments.public_private_index_binding,
    )
    print(json.dumps({"counts": receipt["counts"], "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

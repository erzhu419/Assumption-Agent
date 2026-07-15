"""Container-side runtime for typed, bijective file assignment.

This module deliberately uses only the Python standard library.  It is copied
into a SkillLearn task container and run in three phases:

``prepare``
    Parse the public organize-task destinations, snapshot the task tree, and
    write a bounded, read-only evidence profile for the agent.
``apply``
    Verify that only ``plan.json`` was added, validate the typed plan, move the
    files transactionally, and reconcile the resulting tree.
``reconcile``
    Re-open and independently reconcile an already applied plan.

The receipts emitted on stdout contain hashes and counts only.  File names,
instruction text, and extracted document text remain inside the container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET
import zipfile


TYPED_ASSIGNMENT_RUNTIME_POLICY_V3 = (
    "typed_assignment_prepare_plan_apply_reconcile_v3"
)

DEFAULT_EVIDENCE_FILENAME = "evidence.json"
DEFAULT_PRE_MANIFEST_FILENAME = "pre_manifest.json"
DEFAULT_PLAN_SCHEMA_FILENAME = "plan_schema.json"
DEFAULT_PREPARE_STATE_FILENAME = "prepare_state.json"
DEFAULT_PREPARE_RECEIPT_FILENAME = "prepare_receipt.json"
DEFAULT_PLAN_FILENAME = "plan.json"
DEFAULT_RECONCILIATION_RECEIPT_FILENAME = "reconciliation_receipt.json"

PUBLIC_ORGANIZE_DESTINATIONS = (
    "LLM",
    "trapped_ion_and_qc",
    "black_hole",
    "DNA",
    "music_history",
)
SUPPORTED_SUFFIXES = (".docx", ".pdf", ".pptx")
PLAN_BASES = ("positive_content_evidence", "public_default")

DEFAULT_MAX_FILES = 256
DEFAULT_MAX_TREE_ENTRIES = 4096
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_INSTRUCTION_BYTES = 512 * 1024
DEFAULT_MAX_EXTRACTED_CHARACTERS = 4096
DEFAULT_MAX_XML_MEMBER_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_EVIDENCE_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_PDF_PAGES = 2
DEFAULT_PDF_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_PLAN_BYTES = 512 * 1024

_DESTINATION_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class TypedAssignmentRuntimeError(RuntimeError):
    """Raised when the runtime contract cannot be prepared or enforced."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the one canonical JSON encoding used for contract hashes."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _runtime_tool_sha256() -> str:
    return sha256_file(Path(__file__).resolve(strict=True))


def _receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_hash"] = _payload_hash(result)
    return result


def _verify_receipt(payload: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypedAssignmentRuntimeError(f"{label} is not a JSON object")
    receipt_hash = payload.get("receipt_hash")
    if not isinstance(receipt_hash, str) or not _SHA256_RE.fullmatch(
        receipt_hash
    ):
        raise TypedAssignmentRuntimeError(f"{label} receipt hash is invalid")
    body = dict(payload)
    del body["receipt_hash"]
    if _payload_hash(body) != receipt_hash:
        raise TypedAssignmentRuntimeError(f"{label} receipt hash mismatch")
    return payload


def _atomic_write_json(path: Path, payload: Any, *, readonly: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444 if readonly else 0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_bounded_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    if maximum <= 0:
        raise TypedAssignmentRuntimeError(f"{label} bound is invalid")
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum + 1)
    except OSError as error:
        raise TypedAssignmentRuntimeError(f"cannot read {label}") from error
    if len(raw) > maximum:
        raise TypedAssignmentRuntimeError(f"{label} exceeds its byte bound")
    return raw


def _read_json_file(path: Path, *, maximum: int, label: str) -> Any:
    raw = _read_bounded_bytes(path, maximum=maximum, label=label)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TypedAssignmentRuntimeError(f"{label} is not valid UTF-8 JSON") from error


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TypedAssignmentRuntimeError(f"{label} is not a sha256 digest")
    return value


def parse_public_destination_spec(
    public_instruction: str,
) -> tuple[tuple[str, ...], str | None]:
    """Parse the closed organize vocabulary and an explicitly public default.

    A default is available only when the public instruction says either that
    ``music_history`` is the default or that non-matching items go to the last
    of the five publicly named categories.  Absence of such wording disables
    the ``public_default`` plan basis.
    """

    if not isinstance(public_instruction, str) or not public_instruction.strip():
        raise TypedAssignmentRuntimeError("public instruction is empty")
    missing = [
        name
        for name in PUBLIC_ORGANIZE_DESTINATIONS
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
            public_instruction,
        )
        is None
    ]
    if missing:
        raise TypedAssignmentRuntimeError(
            "public instruction does not expose the closed destination set"
        )

    lower = " ".join(public_instruction.casefold().split())
    direct_default = bool(
        re.search(
            r"(?:default(?:ed)?|fallback|otherwise).{0,100}music_history",
            lower,
        )
        or re.search(
            r"music_history.{0,100}(?:default|fallback)",
            lower,
        )
    )
    first_four_default = bool(
        (
            "first 4" in lower
            or "first four" in lower
            or "other 4" in lower
            or "other four" in lower
        )
        and ("last" in lower or "fifth" in lower or "otherwise" in lower)
    )
    public_default = (
        PUBLIC_ORGANIZE_DESTINATIONS[-1]
        if direct_default or first_four_default
        else None
    )
    return PUBLIC_ORGANIZE_DESTINATIONS, public_default


def _normalize_text(raw: str, *, maximum_characters: int) -> tuple[str, bool]:
    cleaned = "".join(
        character
        if character in "\n\t" or ord(character) >= 32
        else " "
        for character in raw
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    truncated = len(cleaned) > maximum_characters
    return cleaned[:maximum_characters], truncated


def _extract_pdf_text(
    path: Path,
    *,
    pdftotext_binary: str,
    pdf_pages: int,
    timeout_seconds: float,
    maximum_characters: int,
) -> tuple[str, bool, str]:
    with tempfile.TemporaryDirectory(prefix="typed-assignment-pdf-") as folder:
        output = Path(folder) / "first-pages.txt"
        try:
            completed = subprocess.run(
                [
                    pdftotext_binary,
                    "-f",
                    "1",
                    "-l",
                    str(pdf_pages),
                    "-nopgbrk",
                    str(path),
                    str(output),
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "", False, "unavailable"
        if completed.returncode != 0 or not output.is_file():
            return "", False, "unavailable"
        raw = _read_bounded_bytes(
            output,
            maximum=max(64 * 1024, maximum_characters * 8),
            label="pdftotext output",
        )
    text, truncated = _normalize_text(
        raw.decode("utf-8", errors="replace"),
        maximum_characters=maximum_characters,
    )
    return text, truncated, "ok" if text else "unavailable"


def _natural_slide_key(name: str) -> tuple[int, str]:
    match = re.search(r"slide([0-9]+)\.xml\Z", name)
    return (int(match.group(1)) if match else 2**31 - 1, name)


def _extract_openxml_text(
    path: Path,
    *,
    suffix: str,
    maximum_characters: int,
    maximum_member_bytes: int,
    maximum_slides: int,
) -> tuple[str, bool, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            if suffix == ".docx":
                members = ["word/document.xml"]
            else:
                members = sorted(
                    (
                        name
                        for name in archive.namelist()
                        if re.fullmatch(r"ppt/slides/slide[0-9]+\.xml", name)
                    ),
                    key=_natural_slide_key,
                )[:maximum_slides]
            if not members:
                return "", False, "unavailable"
            fragments: list[str] = []
            for member in members:
                try:
                    information = archive.getinfo(member)
                except KeyError:
                    return "", False, "unavailable"
                if information.file_size > maximum_member_bytes:
                    return "", False, "unavailable"
                with archive.open(information) as handle:
                    raw = handle.read(maximum_member_bytes + 1)
                if len(raw) > maximum_member_bytes:
                    return "", False, "unavailable"
                try:
                    root = ET.fromstring(raw)
                except ET.ParseError:
                    return "", False, "unavailable"
                fragments.extend(text for text in root.itertext() if text)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return "", False, "unavailable"
    text, truncated = _normalize_text(
        " ".join(fragments),
        maximum_characters=maximum_characters,
    )
    return text, truncated, "ok" if text else "unavailable"


def _extract_content_evidence(
    path: Path,
    *,
    pdftotext_binary: str,
    pdf_pages: int,
    pdf_timeout_seconds: float,
    maximum_characters: int,
    maximum_xml_member_bytes: int,
) -> tuple[str, bool, str, str]:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        text, truncated, status = _extract_pdf_text(
            path,
            pdftotext_binary=pdftotext_binary,
            pdf_pages=pdf_pages,
            timeout_seconds=pdf_timeout_seconds,
            maximum_characters=maximum_characters,
        )
        return text, truncated, status, "pdf_first_pages_text"
    if suffix in {".docx", ".pptx"}:
        text, truncated, status = _extract_openxml_text(
            path,
            suffix=suffix,
            maximum_characters=maximum_characters,
            maximum_member_bytes=maximum_xml_member_bytes,
            maximum_slides=pdf_pages,
        )
        kind = "docx_document_xml_text" if suffix == ".docx" else "pptx_first_slides_xml_text"
        return text, truncated, status, kind
    raise TypedAssignmentRuntimeError("unsupported source file type")


def _scan_task_tree(
    task_root: Path,
    *,
    maximum_entries: int,
    maximum_file_bytes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(task_root.rglob("*"), key=lambda item: item.as_posix()):
        if len(rows) >= maximum_entries:
            raise TypedAssignmentRuntimeError("task tree exceeds its entry bound")
        relative = path.relative_to(task_root).as_posix()
        try:
            stat = path.lstat()
        except OSError as error:
            raise TypedAssignmentRuntimeError("cannot stat task-tree entry") from error
        if path.is_symlink():
            raise TypedAssignmentRuntimeError("task tree contains a symlink")
        if path.is_dir():
            rows.append({"path": relative, "type": "directory"})
            continue
        if not path.is_file():
            raise TypedAssignmentRuntimeError("task tree contains a special file")
        if stat.st_size > maximum_file_bytes:
            raise TypedAssignmentRuntimeError("task file exceeds its byte bound")
        rows.append(
            {
                "path": relative,
                "type": "file",
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
        "entries": rows,
    }
    payload["manifest_hash"] = _payload_hash(payload)
    return payload


def _verify_manifest(payload: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "runtime_policy",
        "entries",
        "manifest_hash",
    }:
        raise TypedAssignmentRuntimeError(f"{label} schema is invalid")
    manifest_hash = _require_sha256(
        payload["manifest_hash"], label=f"{label} hash"
    )
    body = dict(payload)
    del body["manifest_hash"]
    if _payload_hash(body) != manifest_hash:
        raise TypedAssignmentRuntimeError(f"{label} hash mismatch")
    if payload["runtime_policy"] != TYPED_ASSIGNMENT_RUNTIME_POLICY_V3:
        raise TypedAssignmentRuntimeError(f"{label} policy mismatch")
    if not isinstance(payload["entries"], list):
        raise TypedAssignmentRuntimeError(f"{label} entries are invalid")
    return payload


def _source_files(
    source_dir: Path,
    *,
    maximum_files: int,
    maximum_file_bytes: int,
) -> list[Path]:
    paths: list[Path] = []
    try:
        children = sorted(source_dir.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise TypedAssignmentRuntimeError("cannot list source directory") from error
    for child in children:
        if child.is_symlink() or not child.is_file():
            raise TypedAssignmentRuntimeError(
                "source directory must contain direct regular files only"
            )
        if child.suffix.casefold() not in SUPPORTED_SUFFIXES:
            raise TypedAssignmentRuntimeError("source contains an unsupported file")
        if child.stat().st_size > maximum_file_bytes:
            raise TypedAssignmentRuntimeError("source file exceeds its byte bound")
        paths.append(child)
    if not paths:
        raise TypedAssignmentRuntimeError("source directory is empty")
    if len(paths) > maximum_files:
        raise TypedAssignmentRuntimeError("source directory exceeds its file bound")
    return paths


def _plan_schema_payload(destinations: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract_hash", "evidence_set_hash", "assignments"],
        "properties": {
            "contract_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "evidence_set_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "file_id",
                        "destination",
                        "basis",
                        "evidence_ids",
                    ],
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "destination": {
                            "type": "string",
                            "enum": list(destinations),
                        },
                        "basis": {"type": "string", "enum": list(PLAN_BASES)},
                        "evidence_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                            "uniqueItems": True,
                        },
                    },
                },
            },
        },
    }


def prepare_assignment_runtime(
    *,
    task_root: Path,
    source_dir: Path,
    public_instruction_file: Path,
    sidecar_dir: Path,
    maximum_files: int = DEFAULT_MAX_FILES,
    maximum_tree_entries: int = DEFAULT_MAX_TREE_ENTRIES,
    maximum_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    maximum_extracted_characters: int = DEFAULT_MAX_EXTRACTED_CHARACTERS,
    maximum_xml_member_bytes: int = DEFAULT_MAX_XML_MEMBER_BYTES,
    maximum_evidence_file_bytes: int = DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    pdf_pages: int = DEFAULT_PDF_PAGES,
    pdf_timeout_seconds: float = DEFAULT_PDF_TIMEOUT_SECONDS,
    pdftotext_binary: str = "pdftotext",
) -> dict[str, Any]:
    """Prepare the immutable evidence/state sidecar and return a safe receipt."""

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (
            maximum_files,
            maximum_tree_entries,
            maximum_file_bytes,
            maximum_extracted_characters,
            maximum_xml_member_bytes,
            maximum_evidence_file_bytes,
            pdf_pages,
        )
    ):
        raise TypedAssignmentRuntimeError("prepare bounds are invalid")
    if not isinstance(pdf_timeout_seconds, (int, float)) or pdf_timeout_seconds <= 0:
        raise TypedAssignmentRuntimeError("PDF timeout is invalid")

    task_root = task_root.resolve(strict=True)
    source_dir = source_dir.resolve(strict=True)
    instruction_path = public_instruction_file.resolve(strict=True)
    sidecar_dir = sidecar_dir.resolve(strict=False)
    if not task_root.is_dir() or not source_dir.is_dir():
        raise TypedAssignmentRuntimeError("task root or source directory is invalid")
    if source_dir.parent != task_root:
        raise TypedAssignmentRuntimeError(
            "source directory must be a direct child of task root"
        )
    if _is_relative_to(sidecar_dir, task_root) or _is_relative_to(
        task_root, sidecar_dir
    ):
        raise TypedAssignmentRuntimeError("sidecar must be outside task root")
    if sidecar_dir.exists():
        if not sidecar_dir.is_dir() or any(sidecar_dir.iterdir()):
            raise TypedAssignmentRuntimeError("sidecar directory is not empty")
    else:
        sidecar_dir.mkdir(parents=True, mode=0o755)

    instruction_raw = _read_bounded_bytes(
        instruction_path,
        maximum=DEFAULT_MAX_INSTRUCTION_BYTES,
        label="public instruction",
    )
    try:
        public_instruction = instruction_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TypedAssignmentRuntimeError(
            "public instruction is not UTF-8"
        ) from error
    destinations, public_default = parse_public_destination_spec(
        public_instruction
    )
    if any(_DESTINATION_RE.fullmatch(value) is None for value in destinations):
        raise TypedAssignmentRuntimeError("public destination is unsafe")

    source_files = _source_files(
        source_dir,
        maximum_files=maximum_files,
        maximum_file_bytes=maximum_file_bytes,
    )
    if any(path.suffix.casefold() == ".pdf" for path in source_files):
        resolved_pdftotext = shutil.which(pdftotext_binary)
        if resolved_pdftotext is None:
            raise TypedAssignmentRuntimeError("pdftotext is unavailable")
    else:
        resolved_pdftotext = pdftotext_binary

    pre_manifest = _scan_task_tree(
        task_root,
        maximum_entries=maximum_tree_entries,
        maximum_file_bytes=maximum_file_bytes,
    )
    schema_body = _plan_schema_payload(destinations)
    tool_sha256 = _runtime_tool_sha256()
    public_instruction_sha256 = sha256_bytes(instruction_raw)
    extraction_policy = {
        "supported_suffixes": list(SUPPORTED_SUFFIXES),
        "pdf_first_pages": pdf_pages,
        "maximum_extracted_characters_per_file": maximum_extracted_characters,
        "maximum_xml_member_bytes": maximum_xml_member_bytes,
        "pdf_timeout_seconds": float(pdf_timeout_seconds),
        "network_used": False,
    }
    contract_payload = {
        "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
        "runtime_tool_sha256": tool_sha256,
        "public_instruction_sha256": public_instruction_sha256,
        "source_relative_path": source_dir.relative_to(task_root).as_posix(),
        "destinations": list(destinations),
        "public_default": public_default,
        "extraction_policy": extraction_policy,
        "plan_schema_hash": _payload_hash(schema_body),
    }
    contract_hash = _payload_hash(contract_payload)

    evidence_rows: list[dict[str, Any]] = []
    state_files: list[dict[str, Any]] = []
    evidence_count = 0
    unavailable_count = 0
    pre_file_index = {
        str(row["path"]): row
        for row in pre_manifest["entries"]
        if row.get("type") == "file"
    }
    for path in source_files:
        source_name = path.name
        manifest_row = pre_file_index.get(
            f"{source_dir.relative_to(task_root).as_posix()}/{source_name}"
        )
        if not isinstance(manifest_row, dict):
            raise TypedAssignmentRuntimeError(
                "source file is absent from the pre manifest"
            )
        content_sha256 = _require_sha256(
            manifest_row.get("sha256"), label="source content hash"
        )
        size_bytes = manifest_row.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise TypedAssignmentRuntimeError("source file size is invalid")
        file_id = _payload_hash(
            {
                "contract_hash": contract_hash,
                "source_name": source_name,
                "content_sha256": content_sha256,
            }
        )
        text, truncated, status, evidence_kind = _extract_content_evidence(
            path,
            pdftotext_binary=resolved_pdftotext,
            pdf_pages=pdf_pages,
            pdf_timeout_seconds=float(pdf_timeout_seconds),
            maximum_characters=maximum_extracted_characters,
            maximum_xml_member_bytes=maximum_xml_member_bytes,
        )
        evidence: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        if status == "ok" and text:
            text_sha256 = sha256_bytes(text.encode("utf-8"))
            evidence_id = _payload_hash(
                {
                    "contract_hash": contract_hash,
                    "file_id": file_id,
                    "kind": evidence_kind,
                    "text_sha256": text_sha256,
                }
            )
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "kind": evidence_kind,
                    "text": text,
                    "text_sha256": text_sha256,
                    "truncated": truncated,
                }
            )
            evidence_ids.append(evidence_id)
            evidence_count += 1
        else:
            unavailable_count += 1
        evidence_rows.append(
            {
                "file_id": file_id,
                "filename": source_name,
                "content_sha256": content_sha256,
                "size_bytes": size_bytes,
                "media_type": path.suffix.casefold().lstrip("."),
                "extraction_status": status,
                "evidence": evidence,
            }
        )
        state_files.append(
            {
                "file_id": file_id,
                "source_name": source_name,
                "content_sha256": content_sha256,
                "size_bytes": size_bytes,
                "evidence_ids": evidence_ids,
            }
        )

    evidence_body = {
        "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
        "contract_hash": contract_hash,
        "destinations": list(destinations),
        "public_default": public_default,
        "extraction_policy": extraction_policy,
        "files": evidence_rows,
    }
    evidence_set_hash = _payload_hash(evidence_body)
    evidence_profile = dict(evidence_body)
    evidence_profile["evidence_set_hash"] = evidence_set_hash

    schema_profile = {
        "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
        "contract_hash": contract_hash,
        "evidence_set_hash": evidence_set_hash,
        "plan_filename": DEFAULT_PLAN_FILENAME,
        "schema": schema_body,
        "basis_rules": {
            "positive_content_evidence": (
                "evidence_ids must be non-empty and belong to the same file_id"
            ),
            "public_default": (
                "destination must equal public_default and evidence_ids must be empty"
            ),
        },
    }

    evidence_path = sidecar_dir / DEFAULT_EVIDENCE_FILENAME
    manifest_path = sidecar_dir / DEFAULT_PRE_MANIFEST_FILENAME
    schema_path = sidecar_dir / DEFAULT_PLAN_SCHEMA_FILENAME
    state_path = sidecar_dir / DEFAULT_PREPARE_STATE_FILENAME
    receipt_path = sidecar_dir / DEFAULT_PREPARE_RECEIPT_FILENAME
    _atomic_write_json(evidence_path, evidence_profile, readonly=True)
    if evidence_path.stat().st_size > maximum_evidence_file_bytes:
        raise TypedAssignmentRuntimeError("evidence profile exceeds its byte bound")
    _atomic_write_json(manifest_path, pre_manifest, readonly=True)
    _atomic_write_json(schema_path, schema_profile, readonly=True)

    state_payload = {
        "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
        "runtime_tool_sha256": tool_sha256,
        "contract_hash": contract_hash,
        "evidence_set_hash": evidence_set_hash,
        "task_root": str(task_root),
        "source_dir": str(source_dir),
        "source_relative_path": source_dir.relative_to(task_root).as_posix(),
        "destinations": list(destinations),
        "public_default": public_default,
        "files": state_files,
        "maximum_tree_entries": maximum_tree_entries,
        "maximum_file_bytes": maximum_file_bytes,
        "pre_manifest_hash": pre_manifest["manifest_hash"],
        "evidence_file_sha256": sha256_file(evidence_path),
        "pre_manifest_file_sha256": sha256_file(manifest_path),
        "plan_schema_file_sha256": sha256_file(schema_path),
    }
    _atomic_write_json(state_path, state_payload, readonly=True)

    prepare_receipt = _receipt(
        {
            "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
            "runtime_tool_sha256": tool_sha256,
            "contract_hash": contract_hash,
            "evidence_set_hash": evidence_set_hash,
            "evidence_file_sha256": sha256_file(evidence_path),
            "pre_manifest_hash": pre_manifest["manifest_hash"],
            "pre_manifest_file_sha256": sha256_file(manifest_path),
            "plan_schema_file_sha256": sha256_file(schema_path),
            "prepare_state_file_sha256": sha256_file(state_path),
            "public_instruction_sha256": public_instruction_sha256,
            "destination_set_hash": _payload_hash(
                {"destinations": list(destinations)}
            ),
            "file_count": len(state_files),
            "evidence_count": evidence_count,
            "extraction_unavailable_count": unavailable_count,
            "container_evidence_profile_persisted": True,
            "raw_public_instruction_in_receipt": False,
            "raw_content_evidence_in_receipt": False,
            "source_filenames_in_receipt": False,
            "host_safe_receipt": True,
        }
    )
    _atomic_write_json(receipt_path, prepare_receipt, readonly=True)
    return prepare_receipt


def _load_verified_prepare_state(
    *, sidecar_dir: Path, expected_prepare_receipt_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = _require_sha256(
        expected_prepare_receipt_sha256,
        label="expected prepare receipt sha256",
    )
    sidecar_dir = sidecar_dir.resolve(strict=True)
    if not sidecar_dir.is_dir():
        raise TypedAssignmentRuntimeError("sidecar directory is invalid")
    receipt_path = sidecar_dir / DEFAULT_PREPARE_RECEIPT_FILENAME
    if sha256_file(receipt_path) != expected:
        raise TypedAssignmentRuntimeError("prepare receipt file hash mismatch")
    prepare_receipt = _verify_receipt(
        _read_json_file(
            receipt_path,
            maximum=DEFAULT_MAX_PLAN_BYTES,
            label="prepare receipt",
        ),
        label="prepare receipt",
    )
    if prepare_receipt.get("runtime_policy") != TYPED_ASSIGNMENT_RUNTIME_POLICY_V3:
        raise TypedAssignmentRuntimeError("prepare receipt policy mismatch")
    if prepare_receipt.get("runtime_tool_sha256") != _runtime_tool_sha256():
        raise TypedAssignmentRuntimeError("runtime tool hash mismatch")

    state_path = sidecar_dir / DEFAULT_PREPARE_STATE_FILENAME
    if sha256_file(state_path) != prepare_receipt.get(
        "prepare_state_file_sha256"
    ):
        raise TypedAssignmentRuntimeError("prepare state file hash mismatch")
    state = _read_json_file(
        state_path,
        maximum=DEFAULT_MAX_EVIDENCE_FILE_BYTES,
        label="prepare state",
    )
    if not isinstance(state, dict):
        raise TypedAssignmentRuntimeError("prepare state schema is invalid")
    for key in (
        "contract_hash",
        "evidence_set_hash",
        "runtime_tool_sha256",
    ):
        if state.get(key) != prepare_receipt.get(key):
            raise TypedAssignmentRuntimeError(f"prepare state {key} mismatch")
    if state.get("runtime_policy") != TYPED_ASSIGNMENT_RUNTIME_POLICY_V3:
        raise TypedAssignmentRuntimeError("prepare state policy mismatch")

    evidence_path = sidecar_dir / DEFAULT_EVIDENCE_FILENAME
    manifest_path = sidecar_dir / DEFAULT_PRE_MANIFEST_FILENAME
    schema_path = sidecar_dir / DEFAULT_PLAN_SCHEMA_FILENAME
    bound_files = (
        (evidence_path, "evidence_file_sha256"),
        (manifest_path, "pre_manifest_file_sha256"),
        (schema_path, "plan_schema_file_sha256"),
    )
    for path, key in bound_files:
        if sha256_file(path) != prepare_receipt.get(key) or sha256_file(
            path
        ) != state.get(key):
            raise TypedAssignmentRuntimeError(f"{key} mismatch")
    pre_manifest = _verify_manifest(
        _read_json_file(
            manifest_path,
            maximum=DEFAULT_MAX_EVIDENCE_FILE_BYTES,
            label="pre manifest",
        ),
        label="pre manifest",
    )
    if pre_manifest["manifest_hash"] != state.get(
        "pre_manifest_hash"
    ) or pre_manifest["manifest_hash"] != prepare_receipt.get(
        "pre_manifest_hash"
    ):
        raise TypedAssignmentRuntimeError("pre manifest binding mismatch")
    return prepare_receipt, state, pre_manifest


def _validate_sidecar_contents(
    sidecar_dir: Path, *, allow_reconciliation_receipt: bool
) -> None:
    allowed = {
        DEFAULT_EVIDENCE_FILENAME,
        DEFAULT_PRE_MANIFEST_FILENAME,
        DEFAULT_PLAN_SCHEMA_FILENAME,
        DEFAULT_PREPARE_STATE_FILENAME,
        DEFAULT_PREPARE_RECEIPT_FILENAME,
        DEFAULT_PLAN_FILENAME,
    }
    if allow_reconciliation_receipt:
        allowed.add(DEFAULT_RECONCILIATION_RECEIPT_FILENAME)
    actual: set[str] = set()
    for path in sidecar_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise TypedAssignmentRuntimeError("sidecar contains a non-file entry")
        actual.add(path.name)
    if actual - allowed:
        raise TypedAssignmentRuntimeError("sidecar contains an unregistered file")
    required = allowed - {
        DEFAULT_RECONCILIATION_RECEIPT_FILENAME,
    }
    if actual != required and not (
        allow_reconciliation_receipt
        and actual == required | {DEFAULT_RECONCILIATION_RECEIPT_FILENAME}
    ):
        raise TypedAssignmentRuntimeError("sidecar artifact set is incomplete")


def _validate_plan(
    payload: Any, *, state: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    if not isinstance(payload, dict) or set(payload) != {
        "contract_hash",
        "evidence_set_hash",
        "assignments",
    }:
        raise TypedAssignmentRuntimeError("plan top-level schema is invalid")
    if payload["contract_hash"] != state.get("contract_hash"):
        raise TypedAssignmentRuntimeError("plan contract hash mismatch")
    if payload["evidence_set_hash"] != state.get("evidence_set_hash"):
        raise TypedAssignmentRuntimeError("plan evidence-set hash mismatch")
    assignments = payload["assignments"]
    files = state.get("files")
    destinations = state.get("destinations")
    if not isinstance(assignments, list) or not isinstance(files, list):
        raise TypedAssignmentRuntimeError("plan assignments are invalid")
    if not isinstance(destinations, list) or not destinations:
        raise TypedAssignmentRuntimeError("prepare destinations are invalid")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("file_id"), str)
        for row in files
    ):
        raise TypedAssignmentRuntimeError("prepare file index is invalid")
    file_index = {str(row["file_id"]): row for row in files}
    if len(file_index) != len(files):
        raise TypedAssignmentRuntimeError("prepare file IDs are duplicated")
    normalized: list[dict[str, Any]] = []
    assignment_index: dict[str, Mapping[str, Any]] = {}
    for row in assignments:
        if not isinstance(row, dict) or set(row) != {
            "file_id",
            "destination",
            "basis",
            "evidence_ids",
        }:
            raise TypedAssignmentRuntimeError("assignment schema is invalid")
        file_id = row["file_id"]
        destination = row["destination"]
        basis = row["basis"]
        evidence_ids = row["evidence_ids"]
        if not isinstance(file_id, str) or not _SHA256_RE.fullmatch(file_id):
            raise TypedAssignmentRuntimeError("assignment file ID is invalid")
        if file_id not in file_index or file_id in assignment_index:
            raise TypedAssignmentRuntimeError(
                "assignment coverage is duplicated or unknown"
            )
        if not isinstance(destination, str) or destination not in destinations:
            raise TypedAssignmentRuntimeError("assignment destination is invalid")
        if basis not in PLAN_BASES:
            raise TypedAssignmentRuntimeError("assignment basis is invalid")
        if (
            not isinstance(evidence_ids, list)
            or any(
                not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
                for value in evidence_ids
            )
            or len(set(evidence_ids)) != len(evidence_ids)
        ):
            raise TypedAssignmentRuntimeError("assignment evidence IDs are invalid")
        available_evidence = file_index[file_id].get("evidence_ids")
        if not isinstance(available_evidence, list):
            raise TypedAssignmentRuntimeError("prepare evidence index is invalid")
        if basis == "positive_content_evidence":
            if not evidence_ids or not set(evidence_ids).issubset(
                set(available_evidence)
            ):
                raise TypedAssignmentRuntimeError(
                    "positive assignment lacks same-file evidence"
                )
        else:
            if (
                state.get("public_default") is None
                or destination != state.get("public_default")
                or evidence_ids
            ):
                raise TypedAssignmentRuntimeError(
                    "public-default assignment is not publicly licensed"
                )
        normalized_row = {
            "file_id": file_id,
            "destination": destination,
            "basis": basis,
            "evidence_ids": sorted(evidence_ids),
        }
        normalized.append(normalized_row)
        assignment_index[file_id] = normalized_row
    if set(assignment_index) != set(file_index):
        raise TypedAssignmentRuntimeError("plan is not a total file bijection")
    normalized.sort(key=lambda row: row["file_id"])
    normalized_plan = {
        "contract_hash": payload["contract_hash"],
        "evidence_set_hash": payload["evidence_set_hash"],
        "assignments": normalized,
    }
    return normalized_plan, assignment_index


def _paths_from_state(
    state: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Mapping[str, Any]]]:
    try:
        task_root = Path(str(state["task_root"])).resolve(strict=True)
        source_dir = Path(str(state["source_dir"])).resolve(strict=True)
        files = state["files"]
    except (KeyError, OSError) as error:
        raise TypedAssignmentRuntimeError("prepare paths are invalid") from error
    if (
        not task_root.is_dir()
        or not source_dir.is_dir()
        or source_dir.parent != task_root
        or not isinstance(files, list)
    ):
        raise TypedAssignmentRuntimeError("prepare paths are invalid")
    index = {
        str(row["file_id"]): row
        for row in files
        if isinstance(row, dict) and "file_id" in row
    }
    if len(index) != len(files):
        raise TypedAssignmentRuntimeError("prepare file index is invalid")
    return task_root, source_dir, index


def _assert_pre_agent_tree_unchanged(
    *, state: Mapping[str, Any], expected_manifest: Mapping[str, Any]
) -> None:
    task_root, _, _ = _paths_from_state(state)
    actual = _scan_task_tree(
        task_root,
        maximum_entries=int(state["maximum_tree_entries"]),
        maximum_file_bytes=int(state["maximum_file_bytes"]),
    )
    if actual != expected_manifest:
        raise TypedAssignmentRuntimeError("task tree changed before plan application")


def _expected_final_manifest(
    *,
    state: Mapping[str, Any],
    pre_manifest: Mapping[str, Any],
    assignment_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _, _, file_index = _paths_from_state(state)
    rows = [dict(row) for row in pre_manifest["entries"]]
    row_index = {str(row["path"]): row for row in rows}
    source_relative = str(state["source_relative_path"])
    # Application materializes the complete public destination vocabulary,
    # including an empty category.  Bind that behavior in the expected tree
    # instead of silently requiring every category to receive a file.
    for destination in state["destinations"]:
        destination = str(destination)
        if destination not in row_index:
            row_index[destination] = {
                "path": destination,
                "type": "directory",
            }
        elif row_index[destination].get("type") != "directory":
            raise TypedAssignmentRuntimeError("destination collides with a file")
    for file_id, assignment in assignment_index.items():
        file_row = file_index[file_id]
        source_path = f"{source_relative}/{file_row['source_name']}"
        original = row_index.pop(source_path, None)
        if (
            not isinstance(original, dict)
            or original.get("type") != "file"
            or original.get("sha256") != file_row.get("content_sha256")
        ):
            raise TypedAssignmentRuntimeError("pre manifest file binding is invalid")
        destination = str(assignment["destination"])
        destination_path = f"{destination}/{file_row['source_name']}"
        if destination_path in row_index:
            raise TypedAssignmentRuntimeError("destination file already exists")
        moved = dict(original)
        moved["path"] = destination_path
        row_index[destination_path] = moved
    body = {
        "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
        "entries": sorted(row_index.values(), key=lambda row: str(row["path"])),
    }
    body["manifest_hash"] = _payload_hash(body)
    return body


def _reconcile_final_tree(
    *,
    state: Mapping[str, Any],
    pre_manifest: Mapping[str, Any],
    assignment_index: Mapping[str, Mapping[str, Any]],
) -> tuple[str, int, bool]:
    task_root, source_dir, file_index = _paths_from_state(state)
    expected = _expected_final_manifest(
        state=state,
        pre_manifest=pre_manifest,
        assignment_index=assignment_index,
    )
    actual = _scan_task_tree(
        task_root,
        maximum_entries=int(state["maximum_tree_entries"]),
        maximum_file_bytes=int(state["maximum_file_bytes"]),
    )
    if actual != expected:
        raise TypedAssignmentRuntimeError("final task tree does not reconcile")
    if any(source_dir.iterdir()):
        raise TypedAssignmentRuntimeError("source directory is not empty")
    # ``_scan_task_tree`` above re-opened and hashed every destination file.
    # The exact-manifest comparison therefore supplies the content readback;
    # the loop below verifies the path/type projection without hashing all
    # large documents a second time.
    reopened = 0
    for file_id, assignment in assignment_index.items():
        file_row = file_index[file_id]
        destination_path = (
            task_root / str(assignment["destination"]) / str(file_row["source_name"])
        )
        if (
            destination_path.is_symlink()
            or not destination_path.is_file()
        ):
            raise TypedAssignmentRuntimeError(
                "destination file content does not reconcile"
            )
        reopened += 1
    return actual["manifest_hash"], reopened, True


def _build_reconciliation_receipt(
    *,
    mode: str,
    state: Mapping[str, Any],
    normalized_plan: Mapping[str, Any],
    prepare_receipt_file_sha256: str,
    plan_file_sha256: str,
    final_task_manifest_hash: str,
    reopened_file_count: int,
    transactional_apply: bool,
) -> dict[str, Any]:
    assignments = normalized_plan["assignments"]
    distribution = {
        destination: sum(
            1 for row in assignments if row["destination"] == destination
        )
        for destination in state["destinations"]
    }
    return _receipt(
        {
            "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
            "runtime_tool_sha256": _runtime_tool_sha256(),
            "mode": mode,
            "contract_hash": state["contract_hash"],
            "evidence_set_hash": state["evidence_set_hash"],
            "prepare_receipt_file_sha256": prepare_receipt_file_sha256,
            "plan_file_sha256": plan_file_sha256,
            "normalized_plan_hash": _payload_hash(normalized_plan),
            "assignment_count": len(assignments),
            "positive_evidence_assignment_count": sum(
                row["basis"] == "positive_content_evidence"
                for row in assignments
            ),
            "public_default_assignment_count": sum(
                row["basis"] == "public_default" for row in assignments
            ),
            "reopened_file_count": reopened_file_count,
            "source_empty": True,
            "destination_set_hash": _payload_hash(
                {"destinations": list(state["destinations"])}
            ),
            "destination_distribution_hash": _payload_hash(distribution),
            "final_task_manifest_hash": final_task_manifest_hash,
            "all_destination_content_hashes_match": True,
            "transactional_apply": transactional_apply,
            "rollback_required": False,
            "raw_public_instruction_in_receipt": False,
            "raw_content_evidence_in_receipt": False,
            "source_filenames_in_receipt": False,
            "host_safe_receipt": True,
        }
    )


def apply_assignment_plan(
    *, sidecar_dir: Path, expected_prepare_receipt_sha256: str
) -> dict[str, Any]:
    """Validate and apply ``plan.json`` transactionally, then reconcile it."""

    sidecar_dir = sidecar_dir.resolve(strict=True)
    prepare_receipt, state, pre_manifest = _load_verified_prepare_state(
        sidecar_dir=sidecar_dir,
        expected_prepare_receipt_sha256=expected_prepare_receipt_sha256,
    )
    _validate_sidecar_contents(
        sidecar_dir, allow_reconciliation_receipt=False
    )
    plan_path = sidecar_dir / DEFAULT_PLAN_FILENAME
    plan = _read_json_file(
        plan_path,
        maximum=DEFAULT_MAX_PLAN_BYTES,
        label="assignment plan",
    )
    normalized_plan, assignment_index = _validate_plan(plan, state=state)
    _assert_pre_agent_tree_unchanged(
        state=state, expected_manifest=pre_manifest
    )

    task_root, source_dir, file_index = _paths_from_state(state)
    moved: list[tuple[Path, Path]] = []
    created_directories: list[Path] = []
    try:
        for destination in state["destinations"]:
            path = task_root / str(destination)
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise TypedAssignmentRuntimeError(
                        "destination directory is unsafe"
                    )
            else:
                path.mkdir(mode=0o755)
                created_directories.append(path)
        for file_id in sorted(assignment_index):
            assignment = assignment_index[file_id]
            file_row = file_index[file_id]
            source = source_dir / str(file_row["source_name"])
            destination = (
                task_root
                / str(assignment["destination"])
                / str(file_row["source_name"])
            )
            if (
                source.is_symlink()
                or not source.is_file()
                or sha256_file(source) != file_row.get("content_sha256")
                or destination.exists()
            ):
                raise TypedAssignmentRuntimeError(
                    "source or destination changed during application"
                )
            if source.stat().st_dev != destination.parent.stat().st_dev:
                raise TypedAssignmentRuntimeError(
                    "transactional move crosses filesystems"
                )
            os.replace(source, destination)
            moved.append((source, destination))
        final_manifest_hash, reopened_count, hashes_match = (
            _reconcile_final_tree(
                state=state,
                pre_manifest=pre_manifest,
                assignment_index=assignment_index,
            )
        )
        if not hashes_match:
            raise TypedAssignmentRuntimeError(
                "destination content hashes do not reconcile"
            )
    except Exception as error:
        rollback_error: Exception | None = None
        for source, destination in reversed(moved):
            try:
                if destination.exists() and not source.exists():
                    os.replace(destination, source)
            except Exception as nested:
                rollback_error = nested
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        if rollback_error is not None:
            raise TypedAssignmentRuntimeError(
                "transaction failed and rollback was incomplete"
            ) from rollback_error
        try:
            _assert_pre_agent_tree_unchanged(
                state=state, expected_manifest=pre_manifest
            )
        except Exception as nested:
            raise TypedAssignmentRuntimeError(
                "transaction failed and the pre-tree was not restored"
            ) from nested
        if isinstance(error, TypedAssignmentRuntimeError):
            raise
        raise TypedAssignmentRuntimeError("transactional application failed") from error

    receipt = _build_reconciliation_receipt(
        mode="apply_and_reconcile",
        state=state,
        normalized_plan=normalized_plan,
        prepare_receipt_file_sha256=expected_prepare_receipt_sha256,
        plan_file_sha256=sha256_file(plan_path),
        final_task_manifest_hash=final_manifest_hash,
        reopened_file_count=reopened_count,
        transactional_apply=True,
    )
    _atomic_write_json(
        sidecar_dir / DEFAULT_RECONCILIATION_RECEIPT_FILENAME,
        receipt,
        readonly=True,
    )
    return receipt


def reconcile_assignment_runtime(
    *, sidecar_dir: Path, expected_prepare_receipt_sha256: str
) -> dict[str, Any]:
    """Independently re-open and reconcile an already applied assignment."""

    sidecar_dir = sidecar_dir.resolve(strict=True)
    _, state, pre_manifest = _load_verified_prepare_state(
        sidecar_dir=sidecar_dir,
        expected_prepare_receipt_sha256=expected_prepare_receipt_sha256,
    )
    _validate_sidecar_contents(
        sidecar_dir, allow_reconciliation_receipt=True
    )
    plan_path = sidecar_dir / DEFAULT_PLAN_FILENAME
    plan = _read_json_file(
        plan_path,
        maximum=DEFAULT_MAX_PLAN_BYTES,
        label="assignment plan",
    )
    normalized_plan, assignment_index = _validate_plan(plan, state=state)
    final_manifest_hash, reopened_count, hashes_match = _reconcile_final_tree(
        state=state,
        pre_manifest=pre_manifest,
        assignment_index=assignment_index,
    )
    if not hashes_match:
        raise TypedAssignmentRuntimeError(
            "destination content hashes do not reconcile"
        )
    receipt = _build_reconciliation_receipt(
        mode="reconcile_existing",
        state=state,
        normalized_plan=normalized_plan,
        prepare_receipt_file_sha256=expected_prepare_receipt_sha256,
        plan_file_sha256=sha256_file(plan_path),
        final_task_manifest_hash=final_manifest_hash,
        reopened_file_count=reopened_count,
        transactional_apply=False,
    )
    _atomic_write_json(
        sidecar_dir / DEFAULT_RECONCILIATION_RECEIPT_FILENAME,
        receipt,
        readonly=True,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and enforce a typed assignment plan inside a task container."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--task-root", type=Path, required=True)
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--public-instruction-file", type=Path, required=True)
    prepare.add_argument("--sidecar-dir", type=Path, required=True)
    prepare.add_argument("--maximum-files", type=int, default=DEFAULT_MAX_FILES)
    prepare.add_argument(
        "--maximum-tree-entries", type=int, default=DEFAULT_MAX_TREE_ENTRIES
    )
    prepare.add_argument(
        "--maximum-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES
    )
    prepare.add_argument(
        "--maximum-extracted-characters",
        type=int,
        default=DEFAULT_MAX_EXTRACTED_CHARACTERS,
    )
    prepare.add_argument(
        "--maximum-xml-member-bytes",
        type=int,
        default=DEFAULT_MAX_XML_MEMBER_BYTES,
    )
    prepare.add_argument(
        "--maximum-evidence-file-bytes",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    )
    prepare.add_argument("--pdf-pages", type=int, default=DEFAULT_PDF_PAGES)
    prepare.add_argument(
        "--pdf-timeout-seconds",
        type=float,
        default=DEFAULT_PDF_TIMEOUT_SECONDS,
    )
    prepare.add_argument("--pdftotext-binary", default="pdftotext")

    for command in ("apply", "reconcile"):
        child = subparsers.add_parser(command)
        child.add_argument("--sidecar-dir", type=Path, required=True)
        child.add_argument(
            "--expected-prepare-receipt-sha256", required=True
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            receipt = prepare_assignment_runtime(
                task_root=args.task_root,
                source_dir=args.source_dir,
                public_instruction_file=args.public_instruction_file,
                sidecar_dir=args.sidecar_dir,
                maximum_files=args.maximum_files,
                maximum_tree_entries=args.maximum_tree_entries,
                maximum_file_bytes=args.maximum_file_bytes,
                maximum_extracted_characters=args.maximum_extracted_characters,
                maximum_xml_member_bytes=args.maximum_xml_member_bytes,
                maximum_evidence_file_bytes=args.maximum_evidence_file_bytes,
                pdf_pages=args.pdf_pages,
                pdf_timeout_seconds=args.pdf_timeout_seconds,
                pdftotext_binary=args.pdftotext_binary,
            )
        elif args.command == "apply":
            receipt = apply_assignment_plan(
                sidecar_dir=args.sidecar_dir,
                expected_prepare_receipt_sha256=(
                    args.expected_prepare_receipt_sha256
                ),
            )
        else:
            receipt = reconcile_assignment_runtime(
                sidecar_dir=args.sidecar_dir,
                expected_prepare_receipt_sha256=(
                    args.expected_prepare_receipt_sha256
                ),
            )
    except TypedAssignmentRuntimeError as error:
        print(
            json.dumps(
                {
                    "runtime_policy": TYPED_ASSIGNMENT_RUNTIME_POLICY_V3,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Trusted one-shot acquisition boundary for MAUD extraction P2.

This process is the only component allowed to parse source JSON.  It writes
label-free action views and separate mode-0600 gold packs, then exits.  The
action controller never imports this module or receives a ``PreparedSplit``.
F_search answers are never semantically decoded or persisted.  TEST parsing is
physically separate and requires a real A_hold promotion receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping, Sequence

from . import maud_extraction_p2_download_v1 as download
from . import maud_extraction_p2_source_v1 as source


VERSION = "maud_extraction_p2_acquisition_v1"
STUDY_ID = source.STUDY_ID
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
INITIAL_RECEIPT_KEYS = frozenset(
    {
        "F_search_gold_pack_created",
        "TEST_source_opened_or_parsed",
        "download_binding",
        "private_archives",
        "raw_title_context_question_answer_or_offset_in_receipt",
        "retry_replay_resample_or_secret_rotation_count",
        "schema",
        "selection_secret_commitment_sha256",
        "self_sha256",
        "split_aggregates",
        "status",
        "study_id",
        "version",
    }
)
INITIAL_MEMBERSHIP_KEYS = frozenset(
    {
        "DEV_included_context_sha256",
        "DEV_included_normalized_title_sha256",
        "TRAIN_included_context_sha256",
        "TRAIN_included_normalized_title_sha256",
        "cross_split_context_overlap_count",
        "cross_split_title_overlap_count",
        "schema",
        "selection_secret_commitment_sha256",
        "self_sha256",
        "study_id",
    }
)
DOWNLOAD_RECEIPT_KEYS = frozenset(
    {
        "GET_count",
        "JSON_parse_or_row_open_count",
        "file_count",
        "files",
        "online_evaluator_or_model_call_count",
        "retry_resume_or_mirror_switch_count",
        "schema",
        "self_sha256",
        "source_commit",
        "source_custody_self_sha256",
        "status",
        "study_id",
        "total_size_bytes",
        "version",
    }
)
DOWNLOAD_FILE_KEYS = frozenset(
    {
        "git_blob_sha1",
        "local_name",
        "official_relative_path",
        "sha256",
        "size_bytes",
        "split",
    }
)
INITIAL_MEMBERSHIP_NAME = "initial.contract_membership.private.json"
PROMOTION_SCHEMA = (
    "maud_extraction_p2_formal_controller_v1_"
    "A_hold_promotion_receipt_v1"
)
PROMOTION_RULE = (
    "net_strictly_positive_and_complete_contract_sign_flip_"
    "reference_tail_at_most_1_over_10"
)
PROMOTION_KEYS = frozenset(
    {
        "A_hold_action_archive_file_sha256",
        "A_hold_action_archive_semantic_sha256",
        "A_hold_gold_file_sha256",
        "A_hold_gold_semantic_sha256",
        "E1_minus_E0_comparison",
        "M_search_authorized",
        "challenger_evaluator_id",
        "challenger_model_self_sha256",
        "challenger_model_sha256",
        "incumbent_evaluator_id",
        "initial_acquisition_receipt_self_sha256",
        "online_evaluator_API_or_fine_tune_count",
        "promoted",
        "promotion_rule",
        "retry_replay_resample_refit_or_gate_change_count",
        "schema",
        "self_sha256",
        "source_custody_self_sha256",
        "study_design_self_sha256",
        "study_id",
    }
)
PROMOTION_COMPARISON_KEYS = frozenset(
    {
        "contract_count",
        "exact_sign_flip_reference_tail",
        "net",
        "nonzero_contract_count",
    }
)
FRACTION_KEYS = frozenset({"denominator", "numerator"})
STUDY_DESIGN_SELF_SHA256 = (
    "01a1d2ef33eb9721f1644ca748ed13b26b9a6d3b96fba62c603363a104a87cbd"
)


class MaudAcquisitionError(RuntimeError):
    """The trusted source-to-private-archive boundary failed closed."""


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MaudAcquisitionError("acquisition value is not canonical JSON") from exc


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise MaudAcquisitionError("body already contains self_sha256")
    result = dict(body)
    result["self_sha256"] = semantic_sha256(result)
    return result


def write_exclusive(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    raw = canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise MaudAcquisitionError("private archive mode drifted")
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_sha256": semantic_sha256(value),
        "size_bytes": len(raw),
        "mode_octal": "0600",
    }


def _secret(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise MaudAcquisitionError("selection secret is unavailable")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise MaudAcquisitionError("selection secret mode drifted")
    value = path.read_bytes()
    if len(value) != 32:
        raise MaudAcquisitionError("selection secret must be exactly 32 bytes")
    return value


def _source_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise MaudAcquisitionError("frozen source file is unavailable")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise MaudAcquisitionError("frozen source file mode drifted")
    return path


def _read_canonical_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MaudAcquisitionError(f"{label} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaudAcquisitionError(f"{label} is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or canonical_bytes(value) != raw
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise MaudAcquisitionError(f"{label} is not canonical mode-0600 JSON")
    return value


def _validate_self_hash(value: Mapping[str, object], *, label: str) -> str:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or semantic_sha256(body) != declared
    ):
        raise MaudAcquisitionError(f"{label} self hash drifted")
    return declared


def _hash_source_bytes(path: Path, frozen: download.FrozenSource) -> str:
    checked = _source_file(path)
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()
    blob.update(f"blob {frozen.size_bytes}\0".encode("ascii"))
    size = 0
    with checked.open("rb") as handle:
        while True:
            chunk = handle.read(download.READ_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > frozen.size_bytes:
                raise MaudAcquisitionError("frozen source exceeded bound size")
            sha256.update(chunk)
            blob.update(chunk)
    if size != frozen.size_bytes or blob.hexdigest() != frozen.git_blob_sha1:
        raise MaudAcquisitionError("frozen source byte identity drifted")
    return sha256.hexdigest()


def _validated_download_binding(
    *,
    receipt_path: Path,
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Bind only the source splits this acquisition phase may open."""

    receipt = _read_canonical_object(
        receipt_path, label="download receipt"
    )
    if (
        set(receipt) != DOWNLOAD_RECEIPT_KEYS
        or receipt.get("schema") != f"{download.VERSION}_receipt_v1"
        or receipt.get("version") != download.VERSION
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("status")
        != "three_frozen_byte_streams_downloaded_not_JSON_parsed"
        or receipt.get("source_commit") != download.SOURCE_COMMIT
        or receipt.get("source_custody_self_sha256")
        != download.SOURCE_CUSTODY_SELF_SHA256
        or receipt.get("file_count") != len(download.SOURCES)
        or receipt.get("GET_count") != len(download.SOURCES)
        or receipt.get("JSON_parse_or_row_open_count") != 0
        or receipt.get("retry_resume_or_mirror_switch_count") != 0
        or receipt.get("online_evaluator_or_model_call_count") != 0
    ):
        raise MaudAcquisitionError("download receipt contract drifted")
    receipt_self_sha256 = _validate_self_hash(
        receipt, label="download receipt"
    )
    raw_rows = receipt.get("files")
    if (
        isinstance(raw_rows, (str, bytes))
        or not isinstance(raw_rows, Sequence)
        or len(raw_rows) != len(download.SOURCES)
    ):
        raise MaudAcquisitionError("download receipt file rows drifted")
    rows: dict[str, Mapping[str, object]] = {}
    expected_by_split = {row.split: row for row in download.SOURCES}
    for raw in raw_rows:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != DOWNLOAD_FILE_KEYS
            or raw.get("split") not in expected_by_split
        ):
            raise MaudAcquisitionError("download receipt file row drifted")
        split = str(raw["split"])
        if split in rows:
            raise MaudAcquisitionError("duplicate download receipt split")
        frozen = expected_by_split[split]
        if (
            raw.get("official_relative_path") != frozen.relative_path
            or raw.get("local_name") != frozen.local_name
            or raw.get("size_bytes") != frozen.size_bytes
            or raw.get("git_blob_sha1") != frozen.git_blob_sha1
            or not isinstance(raw.get("sha256"), str)
            or _HEX64.fullmatch(str(raw["sha256"])) is None
        ):
            raise MaudAcquisitionError("download receipt source identity drifted")
        rows[split] = raw
    if set(rows) != set(expected_by_split):
        raise MaudAcquisitionError("download receipt split set drifted")
    if receipt.get("total_size_bytes") != sum(
        row.size_bytes for row in download.SOURCES
    ):
        raise MaudAcquisitionError("download receipt total size drifted")

    expected_root = receipt_path.parent / "source_bytes"
    opened: dict[str, object] = {}
    for split, path in source_paths.items():
        if split not in expected_by_split:
            raise MaudAcquisitionError("unrecognized source split binding")
        frozen = expected_by_split[split]
        expected_path = expected_root / frozen.local_name
        if (
            path.is_symlink()
            or expected_path.is_symlink()
            or path.resolve(strict=True) != expected_path.resolve(strict=True)
        ):
            raise MaudAcquisitionError("source path is outside download custody")
        observed_sha256 = _hash_source_bytes(path, frozen)
        if observed_sha256 != rows[split].get("sha256"):
            raise MaudAcquisitionError("source SHA-256 differs from receipt")
        opened[split] = {
            "sha256": observed_sha256,
            "size_bytes": frozen.size_bytes,
            "git_blob_sha1": frozen.git_blob_sha1,
        }
    return {
        "receipt_self_sha256": receipt_self_sha256,
        "receipt_file_sha256": hashlib.sha256(
            canonical_bytes(receipt)
        ).hexdigest(),
        "opened_splits": opened,
    }


def _safe_split_counts(prepared: source.PreparedSplit) -> dict[str, object]:
    result: dict[str, object] = {
        "source_contract_count": prepared.source_contract_count,
        "excluded_contract_count": prepared.excluded_contract_count,
        "remaining_contract_count": len(prepared.contracts),
    }
    for block in source.BLOCKS:
        try:
            rows = prepared.contracts_for(block)
        except source.MaudSourceError:
            continue
        result[block] = {
            "contract_count": len(rows),
            "item_count": sum(len(row.items) for row in rows),
        }
    return result


def _hex_set(value: object, *, label: str) -> set[str]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(
            not isinstance(row, str) or _HEX64.fullmatch(row) is None
            for row in value
        )
    ):
        raise MaudAcquisitionError(f"{label} drifted")
    rows = list(value)
    if rows != sorted(rows) or len(rows) != len(set(rows)):
        raise MaudAcquisitionError(f"{label} is not a canonical set")
    return set(rows)


def _validated_initial_state(
    *, output_root: Path, secret: bytes
) -> dict[str, object]:
    receipt_path = output_root / "initial.receipt.json"
    receipt = _read_canonical_object(
        receipt_path, label="initial acquisition receipt"
    )
    commitment = hashlib.sha256(secret).hexdigest()
    if (
        set(receipt) != INITIAL_RECEIPT_KEYS
        or receipt.get("schema") != f"{VERSION}_initial_receipt_v1"
        or receipt.get("version") != VERSION
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("status")
        != "trusted_initial_parse_complete_and_process_must_exit"
        or receipt.get("selection_secret_commitment_sha256") != commitment
        or receipt.get("F_search_gold_pack_created") is not False
        or receipt.get("TEST_source_opened_or_parsed") is not False
        or receipt.get(
            "raw_title_context_question_answer_or_offset_in_receipt"
        )
        is not False
        or receipt.get(
            "retry_replay_resample_or_secret_rotation_count"
        )
        != 0
    ):
        raise MaudAcquisitionError("initial acquisition receipt drifted")
    receipt_self_sha256 = _validate_self_hash(
        receipt, label="initial acquisition receipt"
    )
    membership_path = output_root / INITIAL_MEMBERSHIP_NAME
    membership = _read_canonical_object(
        membership_path, label="initial private contract membership"
    )
    if (
        set(membership) != INITIAL_MEMBERSHIP_KEYS
        or membership.get("schema")
        != f"{VERSION}_initial_contract_membership_private_v1"
        or membership.get("study_id") != STUDY_ID
        or membership.get("selection_secret_commitment_sha256")
        != commitment
        or membership.get("cross_split_title_overlap_count") != 0
        or membership.get("cross_split_context_overlap_count") != 0
    ):
        raise MaudAcquisitionError("initial private membership drifted")
    membership_self_sha256 = _validate_self_hash(
        membership, label="initial private contract membership"
    )
    private_archives = receipt.get("private_archives")
    if not isinstance(private_archives, Mapping):
        raise MaudAcquisitionError("initial private archive bindings drifted")
    membership_binding = private_archives.get(
        "initial_contract_membership"
    )
    membership_raw = canonical_bytes(membership)
    if (
        not isinstance(membership_binding, Mapping)
        or membership_binding.get("file_sha256")
        != hashlib.sha256(membership_raw).hexdigest()
        or membership_binding.get("semantic_sha256")
        != semantic_sha256(membership)
        or membership_binding.get("size_bytes") != len(membership_raw)
        or membership_binding.get("mode_octal") != "0600"
    ):
        raise MaudAcquisitionError(
            "initial private membership binding drifted"
        )
    train_titles = _hex_set(
        membership.get("TRAIN_included_normalized_title_sha256"),
        label="TRAIN title membership",
    )
    dev_titles = _hex_set(
        membership.get("DEV_included_normalized_title_sha256"),
        label="DEV title membership",
    )
    train_contexts = _hex_set(
        membership.get("TRAIN_included_context_sha256"),
        label="TRAIN context membership",
    )
    dev_contexts = _hex_set(
        membership.get("DEV_included_context_sha256"),
        label="DEV context membership",
    )
    if train_titles.intersection(dev_titles) or train_contexts.intersection(
        dev_contexts
    ):
        raise MaudAcquisitionError("initial split membership overlaps")
    return {
        "receipt_self_sha256": receipt_self_sha256,
        "membership_self_sha256": membership_self_sha256,
        "train_titles": train_titles,
        "dev_titles": dev_titles,
        "train_contexts": train_contexts,
        "dev_contexts": dev_contexts,
        "download_binding": receipt.get("download_binding"),
        "private_archives": dict(private_archives),
    }


def run_initial_acquisition(
    *,
    train_path: Path,
    dev_path: Path,
    download_receipt_path: Path,
    secret_path: Path,
    output_root: Path,
    parser: Callable[..., source.PreparedSplit] = source.parse_split,
) -> dict[str, object]:
    """Create A_form/F_search/A_hold private packs and exit."""

    if output_root.exists() or output_root.is_symlink():
        raise MaudAcquisitionError("initial acquisition root is consumed")
    secret = _secret(secret_path)
    output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_root.mkdir(mode=0o700)
    write_exclusive(
        output_root / "initial.attempt.json",
        self_hashed(
            {
                "schema": f"{VERSION}_initial_attempt_v1",
                "study_id": STUDY_ID,
                "selection_secret_commitment_sha256": hashlib.sha256(secret).hexdigest(),
            }
        ),
    )
    rows: dict[str, object] = {}
    try:
        download_binding = _validated_download_binding(
            receipt_path=download_receipt_path,
            source_paths={"train": train_path, "dev": dev_path},
        )
        train = parser(
            _source_file(train_path),
            split="TRAIN",
            selection_secret=secret,
        )
        dev = parser(
            _source_file(dev_path),
            split="DEV",
            selection_secret=secret,
        )
        train_title_hashes = {
            row.normalized_title_sha256 for row in train.contracts
        }
        dev_title_hashes = {
            row.normalized_title_sha256 for row in dev.contracts
        }
        train_context_hashes = {
            row.context_sha256 for row in train.contracts
        }
        dev_context_hashes = {
            row.context_sha256 for row in dev.contracts
        }
        if train_title_hashes.intersection(
            dev_title_hashes
        ) or train_context_hashes.intersection(dev_context_hashes):
            raise MaudAcquisitionError(
                "TRAIN and DEV overlap by title or exact context"
            )
        membership = self_hashed(
            {
                "schema": f"{VERSION}_initial_contract_membership_private_v1",
                "study_id": STUDY_ID,
                "selection_secret_commitment_sha256": hashlib.sha256(
                    secret
                ).hexdigest(),
                "TRAIN_included_normalized_title_sha256": sorted(
                    train_title_hashes
                ),
                "DEV_included_normalized_title_sha256": sorted(
                    dev_title_hashes
                ),
                "TRAIN_included_context_sha256": sorted(
                    train_context_hashes
                ),
                "DEV_included_context_sha256": sorted(
                    dev_context_hashes
                ),
                "cross_split_title_overlap_count": 0,
                "cross_split_context_overlap_count": 0,
            }
        )
        rows["initial_contract_membership"] = write_exclusive(
            output_root / INITIAL_MEMBERSHIP_NAME, membership
        )
        for prepared, block in (
            (train, "A_form"),
            (train, "F_search"),
            (dev, "A_hold"),
        ):
            rows[f"{block}_action"] = write_exclusive(
                output_root / f"{block}.action.private.json",
                prepared.action_view(block),
            )
        for prepared, block in ((train, "A_form"), (dev, "A_hold")):
            pack = prepared.gold_pack(block)
            file_sha256 = source.write_gold_pack_exclusive(
                output_root / f"{block}.gold.sealed.private.json",
                pack,
            )
            rows[f"{block}_gold"] = {
                "file_sha256": file_sha256,
                "semantic_sha256": pack["gold_pack_sha256"],
                "mode_octal": "0600",
            }
        receipt = self_hashed(
            {
                "schema": f"{VERSION}_initial_receipt_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "trusted_initial_parse_complete_and_process_must_exit",
                "selection_secret_commitment_sha256": hashlib.sha256(secret).hexdigest(),
                "download_binding": download_binding,
                "split_aggregates": {
                    "TRAIN": _safe_split_counts(train),
                    "DEV": _safe_split_counts(dev),
                },
                "private_archives": rows,
                "F_search_gold_pack_created": False,
                "TEST_source_opened_or_parsed": False,
                "raw_title_context_question_answer_or_offset_in_receipt": False,
                "retry_replay_resample_or_secret_rotation_count": 0,
            }
        )
        write_exclusive(output_root / "initial.receipt.json", receipt)
        return receipt
    except BaseException as exc:
        failure = self_hashed(
            {
                "schema": f"{VERSION}_initial_terminal_failure_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "terminal_no_retry_replay_resample_or_secret_rotation",
                "error_type": type(exc).__name__,
                "raw_source_or_gold_content_included": False,
            }
        )
        try:
            write_exclusive(output_root / "initial.terminal.json", failure)
        except OSError:
            pass
        raise


def _exact_fraction(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, Mapping) or set(value) != FRACTION_KEYS:
        raise MaudAcquisitionError(f"{label} fraction drifted")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise MaudAcquisitionError(f"{label} fraction drifted")
    return numerator, denominator


def _promotion_capability(
    path: Path, *, initial_state: Mapping[str, object]
) -> source.TestParseCapability:
    value = _read_canonical_object(
        path, label="A_hold promotion receipt"
    )
    declared = _validate_self_hash(
        value, label="A_hold promotion receipt"
    )
    comparison = value.get("E1_minus_E0_comparison")
    if (
        set(value) != PROMOTION_KEYS
        or value.get("schema") != PROMOTION_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("source_custody_self_sha256")
        != download.SOURCE_CUSTODY_SELF_SHA256
        or value.get("initial_acquisition_receipt_self_sha256")
        != initial_state.get("receipt_self_sha256")
        or value.get("incumbent_evaluator_id")
        != "E0_FIXED_GENERAL_COVERAGE"
        or value.get("challenger_evaluator_id")
        != "E1_AFORM_CENTERED_RIDGE_L2_1"
        or value.get("promotion_rule") != PROMOTION_RULE
        or value.get(
            "retry_replay_resample_refit_or_gate_change_count"
        )
        != 0
        or value.get("online_evaluator_API_or_fine_tune_count") != 0
        or not isinstance(comparison, Mapping)
        or set(comparison) != PROMOTION_COMPARISON_KEYS
    ):
        raise MaudAcquisitionError("A_hold promotion capability drifted")
    for field in (
        "A_hold_action_archive_file_sha256",
        "A_hold_action_archive_semantic_sha256",
        "A_hold_gold_file_sha256",
        "A_hold_gold_semantic_sha256",
        "challenger_model_sha256",
        "challenger_model_self_sha256",
    ):
        if (
            not isinstance(value.get(field), str)
            or _HEX64.fullmatch(str(value[field])) is None
        ):
            raise MaudAcquisitionError(
                "A_hold promotion binding drifted"
            )
    private_archives = initial_state.get("private_archives")
    gold_binding = (
        private_archives.get("A_hold_gold")
        if isinstance(private_archives, Mapping)
        else None
    )
    if (
        not isinstance(gold_binding, Mapping)
        or value.get("A_hold_gold_file_sha256")
        != gold_binding.get("file_sha256")
        or value.get("A_hold_gold_semantic_sha256")
        != gold_binding.get("semantic_sha256")
    ):
        raise MaudAcquisitionError(
            "A_hold promotion gold binding drifted"
        )
    contract_count = comparison.get("contract_count")
    nonzero_count = comparison.get("nonzero_contract_count")
    if (
        isinstance(contract_count, bool)
        or not isinstance(contract_count, int)
        or contract_count <= 0
        or isinstance(nonzero_count, bool)
        or not isinstance(nonzero_count, int)
        or not 0 <= nonzero_count <= contract_count
    ):
        raise MaudAcquisitionError(
            "A_hold promotion contract counts drifted"
        )
    net_numerator, net_denominator = _exact_fraction(
        comparison.get("net"), label="promotion net"
    )
    tail_numerator, tail_denominator = _exact_fraction(
        comparison.get("exact_sign_flip_reference_tail"),
        label="promotion tail",
    )
    if (
        not 0 <= tail_numerator <= tail_denominator
        or (1 << nonzero_count) % tail_denominator != 0
    ):
        raise MaudAcquisitionError(
            "A_hold promotion sign-flip tail drifted"
        )
    derived = (
        net_numerator > 0
        and tail_numerator * 10 <= tail_denominator
    )
    if (
        net_denominator <= 0
        or value.get("promoted") is not derived
        or value.get("M_search_authorized") is not derived
        or derived is not True
    ):
        raise MaudAcquisitionError(
            "A_hold promotion did not authorize TEST"
        )
    return source.TestParseCapability(
        a_hold_promotion_receipt_sha256=declared
    )


def run_test_acquisition(
    *,
    test_path: Path,
    download_receipt_path: Path,
    secret_path: Path,
    promotion_receipt_path: Path,
    output_root: Path,
    parser: Callable[..., source.PreparedSplit] = source.parse_split,
) -> dict[str, object]:
    """Parse TEST exactly once after a real A_hold promotion."""

    if not output_root.is_dir() or output_root.is_symlink():
        raise MaudAcquisitionError("initial acquisition root is unavailable")
    marker = output_root / "test_parse.attempt.json"
    secret = _secret(secret_path)
    initial_state = _validated_initial_state(
        output_root=output_root, secret=secret
    )
    capability = _promotion_capability(
        promotion_receipt_path, initial_state=initial_state
    )
    write_exclusive(
        marker,
        self_hashed(
            {
                "schema": f"{VERSION}_test_attempt_v1",
                "study_id": STUDY_ID,
                "a_hold_promotion_receipt_sha256": (
                    capability.a_hold_promotion_receipt_sha256
                ),
            }
        ),
    )
    try:
        download_binding = _validated_download_binding(
            receipt_path=download_receipt_path,
            source_paths={"test": test_path},
        )
        initial_download = initial_state.get("download_binding")
        if (
            not isinstance(initial_download, Mapping)
            or download_binding.get("receipt_self_sha256")
            != initial_download.get("receipt_self_sha256")
            or download_binding.get("receipt_file_sha256")
            != initial_download.get("receipt_file_sha256")
        ):
            raise MaudAcquisitionError(
                "TEST download receipt differs from initial custody"
            )
        prepared = parser(
            _source_file(test_path),
            split="TEST",
            selection_secret=secret,
            test_parse_capability=capability,
        )
        test_titles = {
            row.normalized_title_sha256 for row in prepared.contracts
        }
        test_contexts = {
            row.context_sha256 for row in prepared.contracts
        }
        if test_titles.intersection(
            initial_state["train_titles"]  # type: ignore[arg-type]
        ) or test_titles.intersection(
            initial_state["dev_titles"]  # type: ignore[arg-type]
        ) or test_contexts.intersection(
            initial_state["train_contexts"]  # type: ignore[arg-type]
        ) or test_contexts.intersection(
            initial_state["dev_contexts"]  # type: ignore[arg-type]
        ):
            raise MaudAcquisitionError(
                "TEST overlaps an initial contract by title or context"
            )
        action_binding = write_exclusive(
            output_root / "M_search.action.private.json",
            prepared.action_view("M_search"),
        )
        gold = prepared.gold_pack("M_search")
        gold_file_sha256 = source.write_gold_pack_exclusive(
            output_root / "M_search.gold.sealed.private.json",
            gold,
        )
        receipt = self_hashed(
            {
                "schema": f"{VERSION}_test_receipt_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "promotion_authorized_TEST_parse_complete_and_process_must_exit",
                "a_hold_promotion_receipt_sha256": (
                    capability.a_hold_promotion_receipt_sha256
                ),
                "initial_acquisition_receipt_self_sha256": (
                    initial_state["receipt_self_sha256"]
                ),
                "download_binding": download_binding,
                "cross_split_title_overlap_count": 0,
                "cross_split_context_overlap_count": 0,
                "split_aggregates": _safe_split_counts(prepared),
                "private_archives": {
                    "M_search_action": action_binding,
                    "M_search_gold": {
                        "file_sha256": gold_file_sha256,
                        "semantic_sha256": gold["gold_pack_sha256"],
                        "mode_octal": "0600",
                    },
                },
                "raw_title_context_question_answer_or_offset_in_receipt": False,
                "retry_replay_resample_or_secret_rotation_count": 0,
            }
        )
        write_exclusive(output_root / "test_parse.receipt.json", receipt)
        return receipt
    except BaseException as exc:
        failure = self_hashed(
            {
                "schema": f"{VERSION}_test_terminal_failure_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "terminal_no_retry_replay_resample_or_secret_rotation",
                "error_type": type(exc).__name__,
                "raw_source_or_gold_content_included": False,
            }
        )
        try:
            write_exclusive(output_root / "test_parse.terminal.json", failure)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser("initial")
    initial.add_argument("--train", required=True, type=Path)
    initial.add_argument("--dev", required=True, type=Path)
    initial.add_argument("--download-receipt", required=True, type=Path)
    initial.add_argument("--secret", required=True, type=Path)
    initial.add_argument("--output-root", required=True, type=Path)
    test = subparsers.add_parser("test")
    test.add_argument("--test", required=True, type=Path)
    test.add_argument("--download-receipt", required=True, type=Path)
    test.add_argument("--secret", required=True, type=Path)
    test.add_argument("--promotion-receipt", required=True, type=Path)
    test.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "initial":
        receipt = run_initial_acquisition(
            train_path=args.train,
            dev_path=args.dev,
            download_receipt_path=args.download_receipt,
            secret_path=args.secret,
            output_root=args.output_root,
        )
    else:
        receipt = run_test_acquisition(
            test_path=args.test,
            download_receipt_path=args.download_receipt,
            secret_path=args.secret,
            promotion_receipt_path=args.promotion_receipt,
            output_root=args.output_root,
        )
    print(json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

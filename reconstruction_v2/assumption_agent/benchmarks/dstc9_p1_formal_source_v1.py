"""One-shot formal DSTC9 source compiler and deterministic block selector.

This process has no action, model, evaluator, score, or online capability.  It
first authenticates the exact successful P0 receipt and its mode-0600 private
eligibility manifest.  Only then does it open the exact deterministic USTAR
bundle.  Knowledge, TRAIN logs/labels, and VALIDATION logs/labels are each
opened exactly once; identity-only members are not opened and TEST is absent.

Selection is whole-study and deterministic.  One representative is selected
per P0 dialogue group with a frozen seed, then fixed TRAIN and VALIDATION
quotas are assigned in fixed block/family order.  The compiler emits a shared
label-free corpus, four model-visible item blocks, three mode-0400 late-qrel
packs, and one safe aggregate receipt.  M_search's public block is also
mode-0400 and remains sealed for a later promotion-authorized controller.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
from typing import Any, BinaryIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assumption_agent.benchmarks import (  # noqa: E402
    dstc9_p0_public_source_qualification_v1 as p0,
)
from assumption_agent.benchmarks import (  # noqa: E402
    dstc9_p1_typed_core_v1 as core,
)


VERSION = "dstc9_p1_formal_source_v1"
STUDY_ID = "DSTC9_P1_HIERARCHICAL_KNOWLEDGE_EVALUATOR_L5_V1"

P0_RECEIPT_FILE_SHA256 = (
    "e8e7f4c211102b5db693e1c076163a1ff12f714466ea5cec6749c7eeccdd3c0e"
)
P0_RECEIPT_SELF_SHA256 = (
    "fe2bde40a89b6fc1beaff0cdf1b925b2ff00f48041323ee7e7e0e645e8652a67"
)
P0_PRIVATE_MANIFEST_FILE_SHA256 = (
    "a41073b17de6c546007031f41fbba30746e8a51621936e435f4583947c9379f6"
)
P0_PRIVATE_MANIFEST_SELF_SHA256 = (
    "0d2467e2e4a4aa92a12ba1471425ecbcfb2532f3e717e97b7b2e9ed7ef6ef8f5"
)
TYPED_CORE_SHA256 = (
    "a8290586595922e074e0a1aff52fd0d3eee396d0f1d366ccfc8407a5db65aa32"
)
P0_IMPLEMENTATION_SHA256 = (
    "70d01888afe888a926c955ef29b73bd01ba0282a3adad71f9d7962406126e1dd"
)

SELECTION_SEED = (
    "2c94a2373d09052771e134425fb6c569d82fbda8bd82e1f675c2d279c15dface"
)
SELECTION_RULE = (
    "seeded_unique_dialogue_representative_then_fixed_block_family_quota_v1"
)
WORK_ID_RULE = "dstc9-work-v1-prefixed-seeded-p0-item-sha256-v1"

FAMILIES = ("hotel", "restaurant", "taxi", "train")
SPLITS = ("TRAIN", "VALIDATION")
BLOCKS = ("A_form", "F_search", "A_hold", "M_search")
QREL_BLOCKS = ("A_form", "A_hold", "M_search")
BLOCK_SPLIT = {
    "A_form": "TRAIN",
    "F_search": "TRAIN",
    "A_hold": "VALIDATION",
    "M_search": "VALIDATION",
}
SPLIT_BLOCKS = {
    "TRAIN": ("A_form", "F_search"),
    "VALIDATION": ("A_hold", "M_search"),
}
DEFAULT_BLOCK_FAMILY_QUOTAS = {
    "A_form": {family: 24 for family in FAMILIES},
    "F_search": {family: 8 for family in FAMILIES},
    "A_hold": {family: 12 for family in FAMILIES},
    "M_search": {family: 12 for family in FAMILIES},
}

PUBLIC_CORPUS_SCHEMA = "dstc9_p1_public_corpus_v1"
PUBLIC_BLOCK_SCHEMA = "dstc9_p1_public_item_block_v1"
PRIVATE_QREL_SCHEMA = "dstc9_p1_private_late_qrel_pack_v1"
SELECTION_RECEIPT_SCHEMA = "dstc9_p1_formal_source_selection_receipt_v1"

PUBLIC_ITEM_KEYS = frozenset(
    {"history", "normalized_query_sha256", "work_id"}
)
PUBLIC_TURN_KEYS = frozenset({"speaker", "text"})
PRIVATE_QREL_ROW_KEYS = frozenset(
    {"family", "gold_ordinal", "work_id"}
)
ELIGIBILITY_ROW_KEYS = frozenset(
    {
        "dialogue_group_sha256",
        "domain",
        "family",
        "normalized_query_sha256",
        "opaque_item_id",
    }
)
P0_RECEIPT_KEYS = frozenset(
    {
        "access_boundary",
        "archive_topology",
        "cross_split_query_aggregate",
        "eligibility_exclusion_aggregate",
        "final_eligible_aggregate",
        "knowledge_aggregate",
        "member_receipts",
        "prefix_trie_aggregate",
        "private_manifest_binding",
        "public_example_exclusion_binding",
        "self_sha256",
        "source",
        "split_source_aggregate",
        "status",
        "study_id",
        "typed_core_binding",
        "version",
    }
)
PRIVATE_MANIFEST_KEYS = frozenset(
    {
        "eligibility_rule_version",
        "eligible_rows_by_split",
        "query_group_contract",
        "self_sha256",
        "source_binding",
        "study_id",
        "typed_core_binding",
        "version",
    }
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CANONICAL_NUMERIC_ID = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_WORK_ID_RE = re.compile(r"dstc9-work-v1-[0-9a-f]{64}\Z")


class Dstc9P1FormalSourceError(RuntimeError):
    """Stable failure from the source-only formal compiler."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.stage: str | None = None
        self.payload_open_counts: Mapping[str, int] | None = None


@dataclass(frozen=True)
class FormalSourceContract:
    """All pre-source identity and deterministic-selection bindings."""

    source_contract: p0.QualificationContract
    p0_receipt_file_sha256: str
    p0_receipt_self_sha256: str
    private_manifest_file_sha256: str
    private_manifest_self_sha256: str
    typed_core_sha256: str
    selection_seed: str
    block_family_quotas: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_contract, p0.QualificationContract)
            or any(
                _HEX64.fullmatch(value) is None or value == "0" * 64
                for value in (
                    self.p0_receipt_file_sha256,
                    self.p0_receipt_self_sha256,
                    self.private_manifest_file_sha256,
                    self.private_manifest_self_sha256,
                    self.typed_core_sha256,
                    self.selection_seed,
                )
            )
            or set(self.block_family_quotas) != set(BLOCKS)
            or any(
                set(self.block_family_quotas[block]) != set(FAMILIES)
                or any(
                    type(self.block_family_quotas[block][family]) is not int
                    or self.block_family_quotas[block][family] < 1
                    for family in FAMILIES
                )
                for block in BLOCKS
            )
        ):
            raise Dstc9P1FormalSourceError(
                "formal_contract_invalid",
                "formal source contract is invalid",
            )
        for split in SPLITS:
            for family in FAMILIES:
                required = sum(
                    self.block_family_quotas[block][family]
                    for block in SPLIT_BLOCKS[split]
                )
                if (
                    self.source_contract
                    .minimum_unique_dialogue_groups[split][family]
                    < required
                ):
                    raise Dstc9P1FormalSourceError(
                        "formal_contract_capacity_invalid",
                        "formal quota exceeds the qualified source capacity",
                    )


DEFAULT_CONTRACT = FormalSourceContract(
    source_contract=p0.OFFICIAL_CONTRACT,
    p0_receipt_file_sha256=P0_RECEIPT_FILE_SHA256,
    p0_receipt_self_sha256=P0_RECEIPT_SELF_SHA256,
    private_manifest_file_sha256=P0_PRIVATE_MANIFEST_FILE_SHA256,
    private_manifest_self_sha256=P0_PRIVATE_MANIFEST_SELF_SHA256,
    typed_core_sha256=TYPED_CORE_SHA256,
    selection_seed=SELECTION_SEED,
    block_family_quotas=DEFAULT_BLOCK_FAMILY_QUOTAS,
)


@dataclass(frozen=True)
class FormalOutputPaths:
    public_corpus: Path
    public_a_form: Path
    public_f_search: Path
    public_a_hold: Path
    public_m_search: Path
    private_a_form_qrels: Path
    private_a_hold_qrels: Path
    private_m_search_qrels: Path
    safe_selection_receipt: Path

    def public_blocks(self) -> Mapping[str, Path]:
        return {
            "A_form": self.public_a_form,
            "F_search": self.public_f_search,
            "A_hold": self.public_a_hold,
            "M_search": self.public_m_search,
        }

    def private_qrels(self) -> Mapping[str, Path]:
        return {
            "A_form": self.private_a_form_qrels,
            "A_hold": self.private_a_hold_qrels,
            "M_search": self.private_m_search_qrels,
        }

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.public_corpus,
            *tuple(self.public_blocks()[block] for block in BLOCKS),
            *tuple(self.private_qrels()[block] for block in QREL_BLOCKS),
            self.safe_selection_receipt,
        )


@dataclass
class _Audit:
    stage: str = "preflight"
    formal_source_access_count: int = 0
    payload_open_counts: dict[str, int] = field(
        default_factory=lambda: {
            "FAQ_identity": 0,
            "LICENSE_identity": 0,
            "NOTICE_identity": 0,
            "knowledge_JSON": 0,
            "TRAIN_labels_JSON": 0,
            "TRAIN_logs_JSON": 0,
            "VALIDATION_labels_JSON": 0,
            "VALIDATION_logs_JSON": 0,
        }
    )

    def begin_source(self) -> None:
        self.formal_source_access_count += 1
        if self.formal_source_access_count != 1:
            raise Dstc9P1FormalSourceError(
                "formal_source_reopen",
                "formal source capability was consumed more than once",
            )

    def opened(self, member: str) -> None:
        key = {
            p0.KNOWLEDGE_MEMBER: "knowledge_JSON",
            p0.TRAIN_LABELS_MEMBER: "TRAIN_labels_JSON",
            p0.TRAIN_LOGS_MEMBER: "TRAIN_logs_JSON",
            p0.VALIDATION_LABELS_MEMBER: "VALIDATION_labels_JSON",
            p0.VALIDATION_LOGS_MEMBER: "VALIDATION_logs_JSON",
        }.get(member)
        if key is None:
            raise Dstc9P1FormalSourceError(
                "forbidden_member_open",
                "formal compiler attempted to open a non-payload member",
            )
        self.payload_open_counts[key] += 1
        if self.payload_open_counts[key] != 1:
            raise Dstc9P1FormalSourceError(
                "payload_reopen",
                "formal source payload was opened more than once",
            )


@dataclass(frozen=True)
class _EligibleRow:
    split: str
    opaque_item_id: str
    family: str
    normalized_query_sha256: str
    dialogue_group_sha256: str


@dataclass(frozen=True)
class _SelectedRow:
    block: str
    split: str
    family: str
    opaque_item_id: str
    normalized_query_sha256: str
    dialogue_group_sha256: str
    selection_key: str
    work_id: str


@dataclass(frozen=True)
class _FileIdentity:
    file_sha256: str
    size_bytes: int
    mode: int


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Dstc9P1FormalSourceError(
            "canonical_json_failure",
            "value is not canonical JSON",
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise Dstc9P1FormalSourceError(
            "self_hash_duplicate",
            "self hash was supplied twice",
        )
    value = dict(body)
    value["self_sha256"] = stable_hash(value)
    return value


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Dstc9P1FormalSourceError(
            "sha256_invalid",
            f"{field_name} is not a lowercase SHA-256",
        )
    return value


def _verify_self_hash(value: Mapping[str, Any], field_name: str) -> str:
    body = dict(value)
    claimed = _required_sha256(
        body.pop("self_sha256", None),
        f"{field_name} self hash",
    )
    if stable_hash(body) != claimed:
        raise Dstc9P1FormalSourceError(
            "self_hash_mismatch",
            f"{field_name} self hash drifted",
        )
    return claimed


def _assert_no_symlink_components(path: Path, field_name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise Dstc9P1FormalSourceError(
                "path_symlink_component",
                f"{field_name} contains a symbolic-link component",
            )


def _read_bound_canonical_json(
    path: Path,
    *,
    field_name: str,
    expected_file_sha256: str,
    expected_mode: int,
) -> tuple[dict[str, Any], _FileIdentity]:
    absolute = path.absolute()
    _assert_no_symlink_components(absolute, field_name)
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise Dstc9P1FormalSourceError(
            "bound_input_unavailable",
            f"{field_name} is unavailable",
        ) from exc
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != expected_mode
    ):
        raise Dstc9P1FormalSourceError(
            "bound_input_metadata",
            f"{field_name} metadata drifted",
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ) != (metadata.st_dev, metadata.st_ino, metadata.st_size):
                raise Dstc9P1FormalSourceError(
                    "bound_input_changed",
                    f"{field_name} changed during open",
                )
            parts: list[bytes] = []
            while True:
                raw = os.read(descriptor, p0.READ_CHUNK_BYTES)
                if not raw:
                    break
                parts.append(raw)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise Dstc9P1FormalSourceError(
            "bound_input_read_failed",
            f"{field_name} could not be read",
        ) from exc
    raw = b"".join(parts)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        file_sha256 != expected_file_sha256
        or len(raw) != metadata.st_size
    ):
        raise Dstc9P1FormalSourceError(
            "bound_input_identity",
            f"{field_name} byte identity drifted",
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9P1FormalSourceError(
            "bound_input_json",
            f"{field_name} is invalid JSON",
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != canonical_bytes(value, newline=True)
    ):
        raise Dstc9P1FormalSourceError(
            "bound_input_noncanonical",
            f"{field_name} is not canonical JSON",
        )
    return value, _FileIdentity(
        file_sha256=file_sha256,
        size_bytes=len(raw),
        mode=expected_mode,
    )


def _typed_core_binding(expected_sha256: str) -> dict[str, str]:
    try:
        binding = dict(p0._typed_core_binding(expected_sha256))
    except p0.Dstc9P0QualificationError as exc:
        raise Dstc9P1FormalSourceError(
            "typed_core_binding",
            "typed core binding drifted",
        ) from exc
    if (
        binding
        != {
            "sha256": expected_sha256,
            "study_id": STUDY_ID,
            "version": core.VERSION,
        }
        or core.VERSION != "dstc9_p1_typed_core_v1"
        or core.STUDY_ID != STUDY_ID
    ):
        raise Dstc9P1FormalSourceError(
            "typed_core_binding",
            "typed core binding drifted",
        )
    return binding


def _p0_implementation_binding() -> dict[str, str]:
    path = Path(p0.__file__).absolute()
    _assert_no_symlink_components(path, "P0 implementation")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Dstc9P1FormalSourceError(
            "p0_implementation_unavailable",
            "P0 implementation is unavailable",
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise Dstc9P1FormalSourceError(
            "p0_implementation_metadata",
            "P0 implementation metadata drifted",
        )
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ) != (metadata.st_dev, metadata.st_ino, metadata.st_size):
                raise Dstc9P1FormalSourceError(
                    "p0_implementation_changed",
                    "P0 implementation changed during open",
                )
            while True:
                raw = os.read(descriptor, p0.READ_CHUNK_BYTES)
                if not raw:
                    break
                digest.update(raw)
                size += len(raw)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise Dstc9P1FormalSourceError(
            "p0_implementation_read",
            "P0 implementation could not be authenticated",
        ) from exc
    if (
        size != metadata.st_size
        or digest.hexdigest() != P0_IMPLEMENTATION_SHA256
        or p0.VERSION != "dstc9_p0_public_source_qualification_v1"
        or p0.STUDY_ID != STUDY_ID
    ):
        raise Dstc9P1FormalSourceError(
            "p0_implementation_binding",
            "P0 implementation binding drifted",
        )
    return {
        "sha256": P0_IMPLEMENTATION_SHA256,
        "study_id": STUDY_ID,
        "version": p0.VERSION,
    }


def _expected_member_identity(
    contract: p0.QualificationContract,
) -> dict[str, dict[str, Any]]:
    return {
        member.path: {
            "git_blob_sha1": member.git_blob_sha1,
            "payload_open_count": 1,
            "sha256": member.sha256,
            "size_bytes": member.size_bytes,
        }
        for member in contract.members
    }


def _expected_source_binding(
    contract: p0.QualificationContract,
) -> dict[str, Any]:
    return {
        "bundle_filename": contract.bundle_filename,
        "bundle_sha256": contract.bundle_sha256,
        "bundle_size_bytes": contract.bundle_size_bytes,
        "commit": p0.OFFICIAL_COMMIT,
        "repository": p0.OFFICIAL_REPOSITORY,
    }


def _validate_p0_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: FormalSourceContract,
    typed_binding: Mapping[str, str],
    private_identity: _FileIdentity,
) -> None:
    if set(receipt) != P0_RECEIPT_KEYS:
        raise Dstc9P1FormalSourceError(
            "p0_receipt_schema",
            "P0 receipt schema drifted",
        )
    self_hash = _verify_self_hash(receipt, "P0 receipt")
    if (
        self_hash != contract.p0_receipt_self_sha256
        or receipt.get("version") != p0.VERSION
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("status")
        != "qualified_public_non_scoring_schema_prefix_group_and_capacity"
        or receipt.get("typed_core_binding") != typed_binding
        or receipt.get("source")
        != _expected_source_binding(contract.source_contract)
        or receipt.get("member_receipts")
        != _expected_member_identity(contract.source_contract)
        or receipt.get("archive_topology")
        != {
            "directory_link_or_special_member_count": 0,
            "mode_0600_member_count": 8,
            "mtime_zero_member_count": 8,
            "regular_member_count": 8,
            "test_member_count": 0,
            "uid_gid_zero_member_count": 8,
            "ustar_header_count": 8,
        }
    ):
        raise Dstc9P1FormalSourceError(
            "p0_receipt_binding",
            "P0 qualification binding drifted",
        )
    access = receipt.get("access_boundary")
    if access != {
        "action_model_evaluator_score_or_secret_count": 0,
        "bundle_full_extraction_count": 0,
        "individual_identifier_text_entity_doc_qrel_or_row_hash_output_count": 0,
        "online_or_API_evaluation_count": 0,
        "payload_member_reopen_count": 0,
        "payload_open_counts": {
            "FAQ_identity": 1,
            "LICENSE_identity": 1,
            "NOTICE_identity": 1,
            "TRAIN_labels_JSON": 1,
            "TRAIN_logs_JSON": 1,
            "VALIDATION_labels_JSON": 1,
            "VALIDATION_logs_JSON": 1,
            "knowledge_JSON": 1,
        },
        "test_member_count": 0,
    }:
        raise Dstc9P1FormalSourceError(
            "p0_access_boundary",
            "P0 access boundary drifted",
        )
    binding = receipt.get("private_manifest_binding")
    if (
        not isinstance(binding, Mapping)
        or set(binding)
        != {"file_sha256", "row_count", "self_sha256", "size_bytes"}
        or binding.get("file_sha256")
        != contract.private_manifest_file_sha256
        or binding.get("self_sha256")
        != contract.private_manifest_self_sha256
        or binding.get("size_bytes") != private_identity.size_bytes
        or not isinstance(binding.get("row_count"), Mapping)
        or set(binding["row_count"]) != set(SPLITS)
    ):
        raise Dstc9P1FormalSourceError(
            "p0_private_binding",
            "P0 private-manifest binding drifted",
        )
    final = receipt.get("final_eligible_aggregate")
    if not isinstance(final, Mapping) or set(final) != set(SPLITS):
        raise Dstc9P1FormalSourceError(
            "p0_capacity_schema",
            "P0 final capacity schema drifted",
        )
    for split in SPLITS:
        split_value = final.get(split)
        if (
            not isinstance(split_value, Mapping)
            or set(split_value)
            != {
                "family_unique_dialogue_group_count",
                "normalized_query_grouping",
                "row_count",
            }
            or not isinstance(
                split_value.get("family_unique_dialogue_group_count"),
                Mapping,
            )
            or set(split_value["family_unique_dialogue_group_count"])
            != set(FAMILIES)
            or not isinstance(split_value.get("normalized_query_grouping"), Mapping)
            or split_value["normalized_query_grouping"].get(
                "maximum_selected_items_per_group"
            )
            != 1
            or split_value["normalized_query_grouping"].get(
                "duplicate_group_count"
            )
            != 0
            or split_value["normalized_query_grouping"].get(
                "duplicate_row_count"
            )
            != 0
            or split_value["normalized_query_grouping"].get(
                "excess_duplicate_row_count"
            )
            != 0
            or split_value["normalized_query_grouping"].get("group_count")
            != split_value.get("row_count")
        ):
            raise Dstc9P1FormalSourceError(
                "p0_capacity_schema",
                "P0 normalized-query capacity drifted",
            )
        for family in FAMILIES:
            required = sum(
                contract.block_family_quotas[block][family]
                for block in SPLIT_BLOCKS[split]
            )
            observed = split_value["family_unique_dialogue_group_count"].get(
                family
            )
            if type(observed) is not int or observed < required:
                raise Dstc9P1FormalSourceError(
                    "p0_capacity_insufficient",
                    "P0 receipt does not qualify the frozen formal quota",
                )


def _validate_private_manifest(
    manifest: Mapping[str, Any],
    *,
    contract: FormalSourceContract,
    typed_binding: Mapping[str, str],
    receipt: Mapping[str, Any],
    private_identity: _FileIdentity,
) -> tuple[_EligibleRow, ...]:
    if set(manifest) != PRIVATE_MANIFEST_KEYS:
        raise Dstc9P1FormalSourceError(
            "private_manifest_schema",
            "private eligibility manifest schema drifted",
        )
    self_hash = _verify_self_hash(manifest, "private eligibility manifest")
    expected_member_identity = _expected_member_identity(
        contract.source_contract
    )
    if (
        self_hash != contract.private_manifest_self_sha256
        or manifest.get("version") != p0.VERSION
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("eligibility_rule_version")
        != p0.ELIGIBILITY_RULE_VERSION
        or manifest.get("typed_core_binding") != typed_binding
        or manifest.get("query_group_contract")
        != {
            "cross_split_policy": "exclude_all_rows",
            "group_field": "normalized_query_sha256",
            "maximum_selected_items_per_group": 1,
        }
        or manifest.get("source_binding")
        != {
            "bundle_sha256": contract.source_contract.bundle_sha256,
            "bundle_size_bytes": contract.source_contract.bundle_size_bytes,
            "commit": p0.OFFICIAL_COMMIT,
            "member_identity": expected_member_identity,
            "repository": p0.OFFICIAL_REPOSITORY,
        }
    ):
        raise Dstc9P1FormalSourceError(
            "private_manifest_binding",
            "private eligibility manifest binding drifted",
        )
    rows_by_split = manifest.get("eligible_rows_by_split")
    if not isinstance(rows_by_split, Mapping) or set(rows_by_split) != set(
        SPLITS
    ):
        raise Dstc9P1FormalSourceError(
            "private_manifest_rows",
            "private eligibility rows are malformed",
        )
    receipt_binding = receipt["private_manifest_binding"]
    assert isinstance(receipt_binding, Mapping)
    receipt_counts = receipt_binding["row_count"]
    assert isinstance(receipt_counts, Mapping)
    final = receipt["final_eligible_aggregate"]
    assert isinstance(final, Mapping)

    result: list[_EligibleRow] = []
    seen_items: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    query_splits: dict[str, set[str]] = defaultdict(set)
    seen_queries: set[str] = set()
    for split in SPLITS:
        raw_rows = rows_by_split.get(split)
        if not isinstance(raw_rows, list):
            raise Dstc9P1FormalSourceError(
                "private_manifest_rows",
                "private eligibility split is not a list",
            )
        expected_sorted = sorted(
            raw_rows,
            key=lambda row: (
                row.get("family") if isinstance(row, Mapping) else "",
                (
                    row.get("dialogue_group_sha256")
                    if isinstance(row, Mapping)
                    else ""
                ),
                row.get("opaque_item_id") if isinstance(row, Mapping) else "",
            ),
        )
        if raw_rows != expected_sorted:
            raise Dstc9P1FormalSourceError(
                "private_manifest_order",
                "private eligibility row order drifted",
            )
        if (
            len(raw_rows) != receipt_counts.get(split)
            or not isinstance(final.get(split), Mapping)
            or len(raw_rows) != final[split].get("row_count")
        ):
            raise Dstc9P1FormalSourceError(
                "private_manifest_row_count",
                "private eligibility row count drifted",
            )
        for raw in raw_rows:
            if not isinstance(raw, Mapping) or set(raw) != ELIGIBILITY_ROW_KEYS:
                raise Dstc9P1FormalSourceError(
                    "private_manifest_row_schema",
                    "private eligibility row schema drifted",
                )
            opaque_item_id = _required_sha256(
                raw.get("opaque_item_id"), "opaque item id"
            )
            family = raw.get("family")
            domain = raw.get("domain")
            query = _required_sha256(
                raw.get("normalized_query_sha256"),
                "normalized query hash",
            )
            group = _required_sha256(
                raw.get("dialogue_group_sha256"),
                "dialogue group hash",
            )
            if (
                family not in FAMILIES
                or domain != family
                or opaque_item_id in seen_items
                or query in seen_queries
            ):
                raise Dstc9P1FormalSourceError(
                    "private_manifest_row_binding",
                    "private eligibility row binding drifted",
                )
            seen_items.add(opaque_item_id)
            seen_queries.add(query)
            group_splits[group].add(split)
            query_splits[query].add(split)
            result.append(
                _EligibleRow(
                    split=split,
                    opaque_item_id=opaque_item_id,
                    family=family,
                    normalized_query_sha256=query,
                    dialogue_group_sha256=group,
                )
            )
    if (
        any(len(splits) != 1 for splits in group_splits.values())
        or any(len(splits) != 1 for splits in query_splits.values())
        or private_identity.file_sha256
        != contract.private_manifest_file_sha256
    ):
        raise Dstc9P1FormalSourceError(
            "private_manifest_cross_block_overlap",
            "private eligibility groups are not block-disjoint",
        )
    return tuple(result)


def _select_rows(
    rows: Sequence[_EligibleRow],
    *,
    contract: FormalSourceContract,
) -> Mapping[str, tuple[_SelectedRow, ...]]:
    """Select once, without skipping, retry, or replacement."""

    grouped: dict[tuple[str, str], list[_EligibleRow]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, _EligibleRow):
            raise Dstc9P1FormalSourceError(
                "eligible_row_type",
                "eligible row type drifted",
            )
        grouped[(row.split, row.dialogue_group_sha256)].append(row)

    representatives: list[tuple[_EligibleRow, str]] = []
    for (split, group), candidates in grouped.items():
        ranked = sorted(
            (
                stable_hash(
                    {
                        "dialogue_group_sha256": group,
                        "normalized_query_sha256": row.normalized_query_sha256,
                        "opaque_item_id": row.opaque_item_id,
                        "seed": contract.selection_seed,
                        "split": split,
                        "stage": "representative",
                    }
                ),
                row.opaque_item_id,
                row,
            )
            for row in candidates
        )
        representatives.append((ranked[0][2], ranked[0][0]))

    by_split_family: dict[tuple[str, str], list[tuple[str, _EligibleRow]]] = (
        defaultdict(list)
    )
    for row, representative_key in representatives:
        allocation_key = stable_hash(
            {
                "dialogue_group_sha256": row.dialogue_group_sha256,
                "family": row.family,
                "normalized_query_sha256": row.normalized_query_sha256,
                "opaque_item_id": row.opaque_item_id,
                "representative_key": representative_key,
                "seed": contract.selection_seed,
                "split": row.split,
                "stage": "allocation",
            }
        )
        by_split_family[(row.split, row.family)].append(
            (allocation_key, row)
        )
    for value in by_split_family.values():
        value.sort(key=lambda pair: (pair[0], pair[1].opaque_item_id))

    selected: dict[str, list[_SelectedRow]] = {
        block: [] for block in BLOCKS
    }
    selected_groups: set[str] = set()
    selected_queries: set[str] = set()
    selected_items: set[str] = set()
    selected_work_ids: set[str] = set()
    for split in SPLITS:
        for family in FAMILIES:
            candidates = by_split_family.get((split, family), [])
            required = sum(
                contract.block_family_quotas[block][family]
                for block in SPLIT_BLOCKS[split]
            )
            if len(candidates) < required:
                raise Dstc9P1FormalSourceError(
                    "representative_capacity",
                    "unique representative capacity is insufficient",
                )
            chosen = candidates[:required]
            offset = 0
            for block in SPLIT_BLOCKS[split]:
                count = contract.block_family_quotas[block][family]
                for selection_key, row in chosen[offset : offset + count]:
                    work_id = "dstc9-work-v1-" + stable_hash(
                        {
                            "opaque_item_id": row.opaque_item_id,
                            "rule": WORK_ID_RULE,
                            "seed": contract.selection_seed,
                            "study_id": STUDY_ID,
                        }
                    )
                    if (
                        row.dialogue_group_sha256 in selected_groups
                        or row.normalized_query_sha256 in selected_queries
                        or row.opaque_item_id in selected_items
                        or work_id in selected_work_ids
                        or _WORK_ID_RE.fullmatch(work_id) is None
                    ):
                        raise Dstc9P1FormalSourceError(
                            "selection_overlap",
                            "whole-study selection is not block-disjoint",
                        )
                    selected_groups.add(row.dialogue_group_sha256)
                    selected_queries.add(row.normalized_query_sha256)
                    selected_items.add(row.opaque_item_id)
                    selected_work_ids.add(work_id)
                    selected[block].append(
                        _SelectedRow(
                            block=block,
                            split=split,
                            family=family,
                            opaque_item_id=row.opaque_item_id,
                            normalized_query_sha256=(
                                row.normalized_query_sha256
                            ),
                            dialogue_group_sha256=(
                                row.dialogue_group_sha256
                            ),
                            selection_key=selection_key,
                            work_id=work_id,
                        )
                    )
                offset += count
    return {
        block: tuple(selected[block])
        for block in BLOCKS
    }


def _canonical_numeric_id(value: object, field_name: str) -> str:
    checked = p0._identifier(value, field_name=field_name)
    if _CANONICAL_NUMERIC_ID.fullmatch(checked) is None:
        raise Dstc9P1FormalSourceError(
            "knowledge_numeric_id",
            f"{field_name} is not a canonical numeric string",
        )
    return checked


def _canonical_entity_id(
    value: object,
    *,
    entity_name: object,
) -> tuple[str, tuple[int, int]]:
    """Order numeric IDs canonically while preserving DSTC9's ``*`` sentinel."""

    checked = p0._identifier(value, field_name="knowledge entity id")
    if checked == "*":
        if entity_name is not None:
            raise Dstc9P1FormalSourceError(
                "knowledge_wildcard_entity",
                "wildcard entity must have a null public name",
            )
        return checked, (0, 0)
    numeric = _canonical_numeric_id(checked, "knowledge entity id")
    return numeric, (1, int(numeric))


def _next_event_or_none(
    events: Iterator[tuple[str, Any]],
) -> tuple[str, Any] | None:
    try:
        return next(events)
    except StopIteration:
        return None


def _read_knowledge(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    member_contract: p0.MemberContract,
    source_contract: p0.QualificationContract,
    audit: _Audit,
) -> tuple[
    tuple[dict[str, object], ...],
    Mapping[tuple[str, str, str], int],
    Mapping[str, Any],
]:
    extracted, reader = p0._open_member(
        archive,
        member,
        member_contract,
        audit,
    )
    try:
        events = iter(p0._ijson_basic_parse(reader))
        first = _next_event_or_none(events)
        if first != ("start_map", None):
            raise Dstc9P1FormalSourceError(
                "knowledge_root",
                "knowledge root is not an object",
            )
        root = p0._read_stream_value(events, first, depth=0)
        if _next_event_or_none(events) is not None:
            raise Dstc9P1FormalSourceError(
                "knowledge_trailing",
                "knowledge contains trailing JSON",
            )
    finally:
        extracted.close()
    member_receipt = p0._verify_member_reader(reader, member_contract)
    if not isinstance(root, dict) or set(root) != set(FAMILIES):
        raise Dstc9P1FormalSourceError(
            "knowledge_family_registry",
            "knowledge family registry drifted",
        )

    snippets: list[dict[str, object]] = []
    qrel_map: dict[tuple[str, str, str], int] = {}
    for family in FAMILIES:
        entities = root[family]
        if not isinstance(entities, dict) or not entities:
            raise Dstc9P1FormalSourceError(
                "knowledge_entity_registry",
                "knowledge entity registry is invalid",
            )
        entity_rows: list[
            tuple[tuple[int, int], str, Mapping[str, Any]]
        ] = []
        for raw_entity_id, raw_entity in entities.items():
            if (
                not isinstance(raw_entity, Mapping)
                or set(raw_entity) != p0.KNOWLEDGE_ENTITY_KEYS
            ):
                raise Dstc9P1FormalSourceError(
                    "knowledge_entity_schema",
                    "knowledge entity schema drifted",
                )
            entity_id, entity_order = _canonical_entity_id(
                raw_entity_id,
                entity_name=raw_entity["name"],
            )
            entity_rows.append((entity_order, entity_id, raw_entity))
        entity_rows.sort(key=lambda row: row[0])
        for _entity_order, entity_id, entity in entity_rows:
            name = entity["name"]
            if name is not None:
                p0._text(name, field_name="knowledge entity name")
            documents = entity["docs"]
            if not isinstance(documents, dict) or not documents:
                raise Dstc9P1FormalSourceError(
                    "knowledge_document_registry",
                    "knowledge document registry is invalid",
                )
            document_rows: list[tuple[str, Mapping[str, Any]]] = []
            for raw_doc_id, raw_doc in documents.items():
                doc_id = _canonical_numeric_id(
                    raw_doc_id,
                    "knowledge document id",
                )
                if (
                    not isinstance(raw_doc, Mapping)
                    or set(raw_doc) != p0.KNOWLEDGE_DOC_KEYS
                ):
                    raise Dstc9P1FormalSourceError(
                        "knowledge_document_schema",
                        "knowledge document schema drifted",
                    )
                document_rows.append((doc_id, raw_doc))
            document_rows.sort(key=lambda row: int(row[0]))
            for doc_id, document in document_rows:
                ordinal = len(snippets)
                try:
                    snippet = core.KnowledgeSnippet(
                        ordinal=ordinal,
                        entity_name=name,
                        title=document["title"],
                        body=document["body"],
                    )
                except core.Dstc9P1TypedCoreError as exc:
                    raise Dstc9P1FormalSourceError(
                        "typed_snippet_contract",
                        "knowledge snippet violates the typed core",
                    ) from exc
                key = (family, entity_id, doc_id)
                if key in qrel_map:
                    raise Dstc9P1FormalSourceError(
                        "knowledge_reference_duplicate",
                        "knowledge reference is duplicated",
                    )
                qrel_map[key] = ordinal
                snippets.append(core.snippet_public_payload(snippet))
    if len(snippets) != source_contract.expected_knowledge_snippets:
        raise Dstc9P1FormalSourceError(
            "knowledge_count",
            "knowledge snippet count drifted",
        )
    return tuple(snippets), qrel_map, member_receipt


def _source_item_id(split: str, source_ordinal: int) -> str:
    return p0.stable_hash(
        {"source_ordinal": source_ordinal, "split": split}
    )


def _read_logs(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    member_contract: p0.MemberContract,
    source_contract: p0.QualificationContract,
    audit: _Audit,
    *,
    split: str,
    selected_by_item: Mapping[str, _SelectedRow],
) -> tuple[dict[str, dict[str, object]], Mapping[str, Any]]:
    extracted, reader = p0._open_member(
        archive,
        member,
        member_contract,
        audit,
    )
    public_items: dict[str, dict[str, object]] = {}
    observed = 0
    try:
        for ordinal, raw_log in enumerate(p0._iter_top_array(reader)):
            if (
                not isinstance(raw_log, list)
                or not 1 <= len(raw_log) <= p0.MAX_TURNS_PER_LOG
            ):
                raise Dstc9P1FormalSourceError(
                    "log_schema",
                    "log is not a bounded nonempty history",
                )
            history: list[core.DialogueTurn] = []
            for raw_turn in raw_log:
                if (
                    not isinstance(raw_turn, Mapping)
                    or set(raw_turn) != p0.TURN_KEYS
                    or raw_turn.get("speaker") not in {"U", "S"}
                ):
                    raise Dstc9P1FormalSourceError(
                        "turn_schema",
                        "turn schema drifted",
                    )
                try:
                    history.append(
                        core.DialogueTurn(
                            speaker=raw_turn["speaker"],  # type: ignore[arg-type]
                            text=raw_turn["text"],  # type: ignore[arg-type]
                        )
                    )
                except core.Dstc9P1TypedCoreError as exc:
                    raise Dstc9P1FormalSourceError(
                        "typed_turn_contract",
                        "turn violates the typed core",
                    ) from exc
            try:
                query_sha256 = core.normalized_query_sha256(history)
                public_history = [
                    core.turn_public_payload(turn) for turn in history
                ]
            except core.Dstc9P1TypedCoreError as exc:
                raise Dstc9P1FormalSourceError(
                    "typed_history_contract",
                    "history violates the typed core",
                ) from exc
            item_id = _source_item_id(split, ordinal)
            selected = selected_by_item.get(item_id)
            if selected is not None:
                if (
                    selected.split != split
                    or query_sha256 != selected.normalized_query_sha256
                    or item_id in public_items
                ):
                    raise Dstc9P1FormalSourceError(
                        "selected_query_binding",
                        "selected query does not match the P0 commitment",
                    )
                public_items[item_id] = {
                    "history": public_history,
                    "normalized_query_sha256": query_sha256,
                    "work_id": selected.work_id,
                }
            observed += 1
    finally:
        extracted.close()
    member_receipt = p0._verify_member_reader(reader, member_contract)
    if observed != source_contract.expected_split_rows[split]:
        raise Dstc9P1FormalSourceError(
            "log_row_count",
            "log row count drifted",
        )
    return public_items, member_receipt


def _read_labels(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    member_contract: p0.MemberContract,
    source_contract: p0.QualificationContract,
    audit: _Audit,
    *,
    split: str,
    selected_by_item: Mapping[str, _SelectedRow],
    qrel_map: Mapping[tuple[str, str, str], int],
) -> tuple[dict[str, dict[str, object]], Mapping[str, Any]]:
    extracted, reader = p0._open_member(
        archive,
        member,
        member_contract,
        audit,
    )
    qrels: dict[str, dict[str, object]] = {}
    observed = 0
    try:
        for ordinal, label in enumerate(p0._iter_top_array(reader)):
            if (
                not isinstance(label, Mapping)
                or type(label.get("target")) is not bool
            ):
                raise Dstc9P1FormalSourceError(
                    "label_schema",
                    "label target schema drifted",
                )
            item_id = _source_item_id(split, ordinal)
            selected = selected_by_item.get(item_id)
            if label["target"] is False:
                if set(label) != p0.TARGET_FALSE_LABEL_KEYS:
                    raise Dstc9P1FormalSourceError(
                        "target_false_schema",
                        "target=false label schema drifted",
                    )
                if selected is not None:
                    raise Dstc9P1FormalSourceError(
                        "selected_target_false",
                        "selected P0 item is no longer target=true",
                    )
            else:
                if set(label) != p0.TARGET_TRUE_LABEL_KEYS:
                    raise Dstc9P1FormalSourceError(
                        "target_true_schema",
                        "target=true label schema drifted",
                    )
                references = label["knowledge"]
                if not isinstance(references, list) or len(references) != 1:
                    raise Dstc9P1FormalSourceError(
                        "target_knowledge_cardinality",
                        "target=true knowledge is not a singleton",
                    )
                reference = references[0]
                if (
                    not isinstance(reference, Mapping)
                    or set(reference) != p0.KNOWLEDGE_REFERENCE_KEYS
                ):
                    raise Dstc9P1FormalSourceError(
                        "knowledge_reference_schema",
                        "knowledge reference schema drifted",
                    )
                family = p0._identifier(
                    reference["domain"],
                    field_name="knowledge reference domain",
                )
                entity_id = p0._reference_identifier(
                    reference["entity_id"],
                    field_name="knowledge reference entity",
                )
                doc_id = p0._reference_identifier(
                    reference["doc_id"],
                    field_name="knowledge reference document",
                )
                if family not in FAMILIES:
                    raise Dstc9P1FormalSourceError(
                        "knowledge_family",
                        "knowledge family is outside the frozen registry",
                    )
                gold_ordinal = qrel_map.get((family, entity_id, doc_id))
                if gold_ordinal is None:
                    raise Dstc9P1FormalSourceError(
                        "knowledge_reference_unresolved",
                        "knowledge reference does not resolve exactly",
                    )
                p0._text(
                    label["response"],
                    field_name="target response",
                )
                if selected is not None:
                    if (
                        selected.family != family
                        or (
                            selected.block in QREL_BLOCKS
                            and item_id in qrels
                        )
                    ):
                        raise Dstc9P1FormalSourceError(
                            "selected_qrel_binding",
                            "selected qrel does not match the P0 family",
                        )
                    if selected.block in QREL_BLOCKS:
                        qrels[item_id] = {
                            "family": family,
                            "gold_ordinal": gold_ordinal,
                            "work_id": selected.work_id,
                        }
            observed += 1
    finally:
        extracted.close()
    member_receipt = p0._verify_member_reader(reader, member_contract)
    if observed != source_contract.expected_split_rows[split]:
        raise Dstc9P1FormalSourceError(
            "label_row_count",
            "label row count drifted",
        )
    return qrels, member_receipt


def _validate_outputs_fresh(paths: FormalOutputPaths) -> FormalOutputPaths:
    all_paths = tuple(path.absolute() for path in paths.all_paths())
    if len(set(all_paths)) != len(all_paths):
        raise Dstc9P1FormalSourceError(
            "output_path_overlap",
            "formal output paths are not distinct",
        )
    for path in all_paths:
        if path.exists() or path.is_symlink():
            raise Dstc9P1FormalSourceError(
                "output_not_fresh",
                "formal output path is not fresh",
            )
        parent = path.parent
        _assert_no_symlink_components(parent, "formal output parent")
        if (
            not parent.is_dir()
            or parent.is_symlink()
            or not os.access(parent, os.W_OK | os.X_OK)
        ):
            raise Dstc9P1FormalSourceError(
                "output_parent",
                "formal output parent is invalid",
            )
    return FormalOutputPaths(*(all_paths))  # type: ignore[arg-type]


def _write_exclusive_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    mode: int,
) -> _FileIdentity:
    raw = canonical_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
        try:
            os.fchmod(descriptor, mode)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise Dstc9P1FormalSourceError(
                        "output_write_stalled",
                        "formal output write stalled",
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise Dstc9P1FormalSourceError(
            "output_write_failed",
            "formal output could not be written",
        ) from exc
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != mode
        or metadata.st_size != len(raw)
    ):
        raise Dstc9P1FormalSourceError(
            "output_metadata",
            "formal output metadata drifted",
        )
    return _FileIdentity(
        file_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        mode=mode,
    )


def _artifact_binding(
    payload: Mapping[str, Any],
    identity: _FileIdentity,
    *,
    row_count: int,
) -> dict[str, Any]:
    return {
        "file_sha256": identity.file_sha256,
        "mode": f"{identity.mode:04o}",
        "row_count": row_count,
        "self_sha256": payload["self_sha256"],
        "size_bytes": identity.size_bytes,
    }


def compile_formal_source(
    *,
    p0_receipt_path: Path,
    private_eligibility_manifest_path: Path,
    bundle_path: Path,
    outputs: FormalOutputPaths,
    contract: FormalSourceContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Consume the frozen formal source once and emit sealed study inputs."""

    audit = _Audit()
    try:
        if not isinstance(contract, FormalSourceContract):
            raise Dstc9P1FormalSourceError(
                "formal_contract_type",
                "formal source contract type drifted",
            )
        contract.__post_init__()
        outputs = _validate_outputs_fresh(outputs)
        audit.stage = "p0_implementation_binding"
        p0_implementation_binding = _p0_implementation_binding()
        audit.stage = "typed_core_binding"
        typed_binding = _typed_core_binding(contract.typed_core_sha256)

        audit.stage = "private_manifest_identity"
        private_manifest, private_identity = _read_bound_canonical_json(
            private_eligibility_manifest_path,
            field_name="private eligibility manifest",
            expected_file_sha256=contract.private_manifest_file_sha256,
            expected_mode=0o600,
        )
        audit.stage = "p0_receipt_identity"
        p0_receipt, p0_identity = _read_bound_canonical_json(
            p0_receipt_path,
            field_name="P0 qualification receipt",
            expected_file_sha256=contract.p0_receipt_file_sha256,
            expected_mode=0o600,
        )
        _validate_p0_receipt(
            p0_receipt,
            contract=contract,
            typed_binding=typed_binding,
            private_identity=private_identity,
        )
        audit.stage = "private_manifest_validation"
        eligible_rows = _validate_private_manifest(
            private_manifest,
            contract=contract,
            typed_binding=typed_binding,
            receipt=p0_receipt,
            private_identity=private_identity,
        )

        audit.stage = "whole_study_selection"
        selected = _select_rows(eligible_rows, contract=contract)
        selected_by_item = {
            row.opaque_item_id: row
            for block in BLOCKS
            for row in selected[block]
        }
        if len(selected_by_item) != sum(
            contract.block_family_quotas[block][family]
            for block in BLOCKS
            for family in FAMILIES
        ):
            raise Dstc9P1FormalSourceError(
                "selection_count",
                "whole-study selected item count drifted",
            )

        audit.stage = "formal_source_identity"
        audit.begin_source()
        bundle_path = bundle_path.absolute()
        _assert_no_symlink_components(bundle_path, "formal source bundle")
        if bundle_path.name != contract.source_contract.bundle_filename:
            raise Dstc9P1FormalSourceError(
                "bundle_filename",
                "formal source bundle filename drifted",
            )
        original_snapshot = p0._verify_bundle_identity(
            bundle_path,
            contract.source_contract,
        )
        try:
            archive = tarfile.open(bundle_path, mode="r:")
        except (OSError, tarfile.TarError) as exc:
            raise Dstc9P1FormalSourceError(
                "archive_open",
                "formal source archive could not be opened",
            ) from exc
        member_receipts: dict[str, Any] = {}
        try:
            audit.stage = "archive_topology"
            members, topology = p0._validate_ustar_topology(
                bundle_path,
                archive,
                contract.source_contract,
            )
            if topology.get("test_member_count") != 0:
                raise Dstc9P1FormalSourceError(
                    "test_member_present",
                    "TEST is present in the formal source archive",
                )
            member_contracts = contract.source_contract.member_map

            audit.stage = "knowledge_JSON"
            corpus_rows, qrel_map, member_receipts[p0.KNOWLEDGE_MEMBER] = (
                _read_knowledge(
                    archive,
                    members[p0.KNOWLEDGE_MEMBER],
                    member_contracts[p0.KNOWLEDGE_MEMBER],
                    contract.source_contract,
                    audit,
                )
            )
            audit.stage = "TRAIN_logs_JSON"
            train_public, member_receipts[p0.TRAIN_LOGS_MEMBER] = _read_logs(
                archive,
                members[p0.TRAIN_LOGS_MEMBER],
                member_contracts[p0.TRAIN_LOGS_MEMBER],
                contract.source_contract,
                audit,
                split="TRAIN",
                selected_by_item=selected_by_item,
            )
            audit.stage = "TRAIN_labels_JSON"
            train_qrels, member_receipts[p0.TRAIN_LABELS_MEMBER] = _read_labels(
                archive,
                members[p0.TRAIN_LABELS_MEMBER],
                member_contracts[p0.TRAIN_LABELS_MEMBER],
                contract.source_contract,
                audit,
                split="TRAIN",
                selected_by_item=selected_by_item,
                qrel_map=qrel_map,
            )
            audit.stage = "VALIDATION_logs_JSON"
            validation_public, member_receipts[
                p0.VALIDATION_LOGS_MEMBER
            ] = _read_logs(
                archive,
                members[p0.VALIDATION_LOGS_MEMBER],
                member_contracts[p0.VALIDATION_LOGS_MEMBER],
                contract.source_contract,
                audit,
                split="VALIDATION",
                selected_by_item=selected_by_item,
            )
            audit.stage = "VALIDATION_labels_JSON"
            validation_qrels, member_receipts[
                p0.VALIDATION_LABELS_MEMBER
            ] = _read_labels(
                archive,
                members[p0.VALIDATION_LABELS_MEMBER],
                member_contracts[p0.VALIDATION_LABELS_MEMBER],
                contract.source_contract,
                audit,
                split="VALIDATION",
                selected_by_item=selected_by_item,
                qrel_map=qrel_map,
            )
        finally:
            archive.close()
        if p0._snapshot(bundle_path) != original_snapshot:
            raise Dstc9P1FormalSourceError(
                "bundle_changed",
                "formal source archive changed during compilation",
            )
        if member_receipts != {
            member: _expected_member_identity(contract.source_contract)[member]
            for member in p0.JSON_MEMBERS
        }:
            raise Dstc9P1FormalSourceError(
                "formal_member_identity",
                "formal payload member identity drifted",
            )
        expected_opens = {
            "FAQ_identity": 0,
            "LICENSE_identity": 0,
            "NOTICE_identity": 0,
            "TRAIN_labels_JSON": 1,
            "TRAIN_logs_JSON": 1,
            "VALIDATION_labels_JSON": 1,
            "VALIDATION_logs_JSON": 1,
            "knowledge_JSON": 1,
        }
        if audit.payload_open_counts != expected_opens:
            raise Dstc9P1FormalSourceError(
                "formal_open_count",
                "formal payload open count drifted",
            )

        public_by_item = {**train_public, **validation_public}
        qrel_by_item = {**train_qrels, **validation_qrels}
        expected_qrel_items = {
            row.opaque_item_id
            for block in QREL_BLOCKS
            for row in selected[block]
        }
        if (
            set(public_by_item) != set(selected_by_item)
            or set(qrel_by_item) != expected_qrel_items
        ):
            raise Dstc9P1FormalSourceError(
                "selected_source_resolution",
                "not every selected item resolved exactly once",
            )

        audit.stage = "output_projection"
        public_corpus_payload = self_hashed(
            {
                "schema": PUBLIC_CORPUS_SCHEMA,
                "snippets": list(corpus_rows),
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        )
        public_block_payloads: dict[str, dict[str, Any]] = {}
        private_qrel_payloads: dict[str, dict[str, Any]] = {}
        for block in BLOCKS:
            ordered = selected[block]
            public_items = [
                public_by_item[row.opaque_item_id] for row in ordered
            ]
            qrel_rows = (
                [
                    qrel_by_item[row.opaque_item_id]
                    for row in ordered
                ]
                if block in QREL_BLOCKS
                else []
            )
            if (
                any(set(item) != PUBLIC_ITEM_KEYS for item in public_items)
                or any(
                    not isinstance(item["history"], list)
                    or any(
                        not isinstance(turn, Mapping)
                        or set(turn) != PUBLIC_TURN_KEYS
                        for turn in item["history"]
                    )
                    for item in public_items
                )
                or (
                    block in QREL_BLOCKS
                    and any(
                        set(row) != PRIVATE_QREL_ROW_KEYS
                        for row in qrel_rows
                    )
                )
            ):
                raise Dstc9P1FormalSourceError(
                    "output_projection_schema",
                    "public/private output projection drifted",
                )
            public_block_payloads[block] = self_hashed(
                {
                    "block_id": block,
                    "items": public_items,
                    "schema": PUBLIC_BLOCK_SCHEMA,
                    "study_id": STUDY_ID,
                    "version": VERSION,
                }
            )
            if block in QREL_BLOCKS:
                private_qrel_payloads[block] = self_hashed(
                    {
                        "block_id": block,
                        "qrels": qrel_rows,
                        "schema": PRIVATE_QREL_SCHEMA,
                        "study_id": STUDY_ID,
                        "version": VERSION,
                    }
                )

        audit.stage = "sealed_output_write"
        corpus_identity = _write_exclusive_json(
            outputs.public_corpus,
            public_corpus_payload,
            mode=0o600,
        )
        public_identities: dict[str, _FileIdentity] = {}
        qrel_identities: dict[str, _FileIdentity] = {}
        for block in BLOCKS:
            public_identities[block] = _write_exclusive_json(
                outputs.public_blocks()[block],
                public_block_payloads[block],
                mode=0o400 if block == "M_search" else 0o600,
            )
        for block in QREL_BLOCKS:
            qrel_identities[block] = _write_exclusive_json(
                outputs.private_qrels()[block],
                private_qrel_payloads[block],
                mode=0o400,
            )

        selected_group_count = len(
            {
                row.dialogue_group_sha256
                for block in BLOCKS
                for row in selected[block]
            }
        )
        selected_query_count = len(
            {
                row.normalized_query_sha256
                for block in BLOCKS
                for row in selected[block]
            }
        )
        selected_count = len(selected_by_item)
        safe_receipt = self_hashed(
            {
                "artifact_binding": {
                    "private_qrels": {
                        block: _artifact_binding(
                            private_qrel_payloads[block],
                            qrel_identities[block],
                            row_count=len(selected[block]),
                        )
                        for block in QREL_BLOCKS
                    },
                    "public_blocks": {
                        block: _artifact_binding(
                            public_block_payloads[block],
                            public_identities[block],
                            row_count=len(selected[block]),
                        )
                        for block in BLOCKS
                    },
                    "public_corpus": _artifact_binding(
                        public_corpus_payload,
                        corpus_identity,
                        row_count=len(corpus_rows),
                    ),
                },
                "compiler_boundary": {
                    "action_count": 0,
                    "model_call_count": 0,
                    "online_or_API_evaluation_count": 0,
                    "score_count": 0,
                },
                "disjointness_aggregate": {
                    "selected_dialogue_group_count": selected_group_count,
                    "selected_item_count": selected_count,
                    "selected_normalized_query_count": selected_query_count,
                    "cross_block_dialogue_group_overlap_count": 0,
                    "cross_block_item_overlap_count": 0,
                    "cross_block_normalized_query_overlap_count": 0,
                },
                "p0_binding": {
                    "implementation": p0_implementation_binding,
                    "private_manifest_file_sha256": (
                        private_identity.file_sha256
                    ),
                    "private_manifest_self_sha256": (
                        contract.private_manifest_self_sha256
                    ),
                    "qualification_receipt_file_sha256": (
                        p0_identity.file_sha256
                    ),
                    "qualification_receipt_self_sha256": (
                        contract.p0_receipt_self_sha256
                    ),
                },
                "quota": {
                    block: {
                        family: contract.block_family_quotas[block][family]
                        for family in FAMILIES
                    }
                    for block in BLOCKS
                },
                "schema": SELECTION_RECEIPT_SCHEMA,
                "seal_contract": {
                    "M_search_public_block_mode": "0400",
                    "M_search_qrel_pack_mode": "0400",
                    "M_search_open_authorization": (
                        "promotion_authorization_required"
                    ),
                    "other_late_qrel_pack_mode": "0400",
                },
                "selection": {
                    "block_order": list(BLOCKS),
                    "family_order": list(FAMILIES),
                    "rule": SELECTION_RULE,
                    "seed": contract.selection_seed,
                    "work_id_rule": WORK_ID_RULE,
                },
                "source_access": {
                    "bundle_sha256": contract.source_contract.bundle_sha256,
                    "formal_source_access_count": (
                        audit.formal_source_access_count
                    ),
                    "identity_member_payload_open_count": 0,
                    "payload_member_open_counts": dict(
                        audit.payload_open_counts
                    ),
                    "payload_member_reopen_count": 0,
                    "test_member_count": 0,
                },
                "status": "selected_and_sealed",
                "study_id": STUDY_ID,
                "typed_core_binding": typed_binding,
                "version": VERSION,
            }
        )
        audit.stage = "safe_receipt_write"
        _write_exclusive_json(
            outputs.safe_selection_receipt,
            safe_receipt,
            mode=0o600,
        )
        audit.stage = "success"
        return safe_receipt
    except Dstc9P1FormalSourceError as exc:
        exc.stage = audit.stage
        exc.payload_open_counts = dict(audit.payload_open_counts)
        raise
    except p0.Dstc9P0QualificationError as exc:
        converted = Dstc9P1FormalSourceError(
            f"p0_{exc.error_code}",
            "inherited P0 source validation failed",
        )
        converted.stage = audit.stage
        converted.payload_open_counts = dict(audit.payload_open_counts)
        raise converted from exc
    except Exception as exc:
        converted = Dstc9P1FormalSourceError(
            "internal_failure",
            "formal source compiler failed closed",
        )
        converted.stage = audit.stage
        converted.payload_open_counts = dict(audit.payload_open_counts)
        raise converted from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-receipt", required=True, type=Path)
    parser.add_argument(
        "--private-eligibility-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--public-corpus", required=True, type=Path)
    parser.add_argument("--public-a-form", required=True, type=Path)
    parser.add_argument("--public-f-search", required=True, type=Path)
    parser.add_argument("--public-a-hold", required=True, type=Path)
    parser.add_argument("--public-m-search", required=True, type=Path)
    parser.add_argument("--private-a-form-qrels", required=True, type=Path)
    parser.add_argument("--private-a-hold-qrels", required=True, type=Path)
    parser.add_argument("--private-m-search-qrels", required=True, type=Path)
    parser.add_argument(
        "--safe-selection-receipt",
        required=True,
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = compile_formal_source(
        p0_receipt_path=args.p0_receipt,
        private_eligibility_manifest_path=(
            args.private_eligibility_manifest
        ),
        bundle_path=args.bundle,
        outputs=FormalOutputPaths(
            public_corpus=args.public_corpus,
            public_a_form=args.public_a_form,
            public_f_search=args.public_f_search,
            public_a_hold=args.public_a_hold,
            public_m_search=args.public_m_search,
            private_a_form_qrels=args.private_a_form_qrels,
            private_a_hold_qrels=args.private_a_hold_qrels,
            private_m_search_qrels=args.private_m_search_qrels,
            safe_selection_receipt=args.safe_selection_receipt,
        ),
    )
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
                "study_id": receipt["study_id"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKS",
    "DEFAULT_BLOCK_FAMILY_QUOTAS",
    "DEFAULT_CONTRACT",
    "Dstc9P1FormalSourceError",
    "FAMILIES",
    "FormalOutputPaths",
    "FormalSourceContract",
    "QREL_BLOCKS",
    "SELECTION_SEED",
    "STUDY_ID",
    "VERSION",
    "canonical_bytes",
    "compile_formal_source",
    "self_hashed",
    "stable_hash",
]

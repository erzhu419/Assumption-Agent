"""One-shot formal-source compiler for the BioASQ P1 typed study.

The compiler is deliberately source-only and standard-library-only (apart
from the two local, frozen BioASQ contract modules).  It authenticates the
exact successful P0 safe receipt and private non-cohort commitment manifest,
creates exactly one private whole-study HMAC secret with ``O_EXCL``, and then
opens, hashes, and strictly decodes the bound ``training11b.json`` once.

The source is reconstructed independently from the P0 implementation
contract.  Query, document, and document-NUL-case-preserved-snippet
components must reproduce every P0 commitment before selection can proceed.
Selection is a deterministic, secret-keyed joint component allocation with
at most one question per connected component.  The resulting 2,900-passage
pool contains every selected qrel plus secret-keyed fillers, and every action
arm receives the same ordinal/text corpus.

Only ``work_id`` and ``query_text`` are projected into public item blocks.
Family labels and set-valued qrels remain in separate mode-0400 late packs.
The M_search public block is also mode-0400 and its receipt explicitly
requires promotion authorization before a controller may open it.  Failures
are terminal: the compiler makes a best effort to write one aggregate-only
safe no-retry receipt and never removes or replaces already-created outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, BinaryIO
import unicodedata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assumption_agent.benchmarks import (  # noqa: E402
    bioasq_p0_public_source_qualification_v1 as p0,
)
from assumption_agent.benchmarks import (  # noqa: E402
    bioasq_p1_typed_core_v1 as core,
)


VERSION = "bioasq_p1_formal_source_v2"
STUDY_ID = "BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1"

# These are the exact remote P0 output identities recorded in the tracked
# aggregate receipt.  The tracked receipt itself is also bound so the remote
# safe/private pair cannot be detached from its public audit lineage.
P0_PUBLIC_AUDIT_RECEIPT_FILE_SHA256 = (
    "271e6f9bc95f87d2978165ce52facdaaca8cf1c0ee0903715eda103d303b02f7"
)
P0_PUBLIC_AUDIT_RECEIPT_SELF_SHA256 = (
    "9a9abc0e0f35c80d076fdaf5e5ada9a0f5462840fcb3322df5ca0de358a9c14d"
)
P0_SAFE_RECEIPT_FILE_SHA256 = (
    "344682626cbe138d73bdabf512aedb57fe8d44e041850fa353c340a07fdc73c1"
)
P0_SAFE_RECEIPT_SELF_SHA256 = (
    "6ea803504d3ec7c65063b696fbad80cb68d80657dd8990ef817e7d1f4b75364f"
)
# DSTC9-compatible public constant names.
P0_RECEIPT_FILE_SHA256 = P0_SAFE_RECEIPT_FILE_SHA256
P0_RECEIPT_SELF_SHA256 = P0_SAFE_RECEIPT_SELF_SHA256
P0_PRIVATE_MANIFEST_FILE_SHA256 = (
    "67a8ee8364fd344d0f49eb85cf775597bece4c8937e1d334c248f174be09b71e"
)
P0_PRIVATE_MANIFEST_SELF_SHA256 = (
    "3d714f8cbb1c9ffc8bd93a00b0cc27979d8eb8cbc91d6b1ad71e3cd596822183"
)
P0_IMPLEMENTATION_SHA256 = (
    "a6230718e485149b674df1e23a1b2254d03e14e25033f31222dc27e9557975bc"
)
TYPED_CORE_SHA256 = (
    "6bfd386431b977043f43eac0984a67b688fad9def276d37902b2fb3c4cff9342"
)

OFFICIAL_SOURCE_SIZE_BYTES = 37_639_648
OFFICIAL_SOURCE_SHA256 = (
    "6df656862ca860efc355c7805d07ddca700d64ecc3785c519a49afccaaeeac98"
)

FAMILIES = ("yesno", "factoid", "list", "summary")
BLOCKS = ("A_form", "F_search", "A_hold", "M_search")
QREL_BLOCKS = ("A_form", "A_hold", "M_search")
DEFAULT_BLOCK_FAMILY_QUOTAS = {
    "A_form": {family: 24 for family in FAMILIES},
    "F_search": {family: 8 for family in FAMILIES},
    "A_hold": {family: 12 for family in FAMILIES},
    "M_search": {family: 12 for family in FAMILIES},
}

CORPUS_SIZE = 2_900
HMAC_SECRET_BYTES = 32
SELECTION_SECRET_BYTES = HMAC_SECRET_BYTES
READ_CHUNK_BYTES = 1 << 20
SELECTION_RULE = (
    "one_secret_joint_component_capacity_then_family_block_HMAC_order_v2"
)
CORPUS_RULE = (
    "all_selected_unique_qrels_plus_nonselected_eligible_HMAC_fillers_then_"
    "whole_pool_HMAC_ordinals_v1"
)
WORK_ID_RULE = "bioasq-work-v2-secret-HMAC-of-p0-item-commitment-v1"

PUBLIC_CORPUS_SCHEMA = "bioasq_p1_public_ordinal_text_corpus_v2"
PUBLIC_BLOCK_SCHEMA = "bioasq_p1_public_item_block_v2"
PRIVATE_QREL_SCHEMA = "bioasq_p1_private_late_qrel_pack_v2"
SELECTION_RECEIPT_SCHEMA = "bioasq_p1_formal_source_selection_receipt_v2"
FAILURE_RECEIPT_SCHEMA = "bioasq_p1_formal_source_failure_receipt_v2"

PUBLIC_PASSAGE_KEYS = frozenset({"ordinal", "text"})
PUBLIC_ITEM_KEYS = frozenset({"query_text", "work_id"})
PRIVATE_QREL_ROW_KEYS = frozenset(
    {"family", "gold_ordinals", "work_id"}
)
P0_PRIVATE_ROW_KEYS = frozenset(
    {
        "component_commitment",
        "family",
        "opaque_item_commitment",
        "query_commitment",
        "snippet_commitments",
    }
)
P0_SAFE_RECEIPT_KEYS = frozenset(
    {
        "access_boundary",
        "capacity",
        "component_aggregate",
        "eligible_question_count_by_family",
        "formal_ineligible_reason_counts",
        "private_manifest_binding",
        "schema",
        "self_sha256",
        "source_binding",
        "source_question_count",
        "source_question_count_by_family",
        "status",
        "study_id",
    }
)
P0_PRIVATE_MANIFEST_KEYS = frozenset(
    {
        "claim_boundary",
        "component_rule",
        "family_order",
        "rows",
        "schema",
        "self_sha256",
        "source_binding",
        "status",
        "study_id",
    }
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID = re.compile(r"bioasq-work-v2-[0-9a-f]{64}\Z")
_MAX_TEXT_CHARACTERS = 10_000_000
_MAX_IDENTIFIER_CHARACTERS = 100_000


class BioasqP1FormalSourceError(RuntimeError):
    """Stable, fail-closed error from the one-shot formal compiler."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.stage: str | None = None
        self.source_access: Mapping[str, int] | None = None
        self.safe_failure_receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class FormalSourceContract:
    """All byte identities and frozen selection/corpus requirements."""

    source_contract: p0.SourceFileContract
    qualification_contract: p0.QualificationContract
    p0_receipt_file_sha256: str
    p0_receipt_self_sha256: str
    private_manifest_file_sha256: str
    private_manifest_self_sha256: str
    p0_implementation_sha256: str
    typed_core_sha256: str
    block_family_quotas: Mapping[str, Mapping[str, int]]
    corpus_size: int = CORPUS_SIZE

    def __post_init__(self) -> None:
        digests = (
            self.p0_receipt_file_sha256,
            self.p0_receipt_self_sha256,
            self.private_manifest_file_sha256,
            self.private_manifest_self_sha256,
            self.p0_implementation_sha256,
            self.typed_core_sha256,
        )
        if (
            not isinstance(self.source_contract, p0.SourceFileContract)
            or not isinstance(
                self.qualification_contract,
                p0.QualificationContract,
            )
            or any(
                not isinstance(value, str)
                or _HEX64.fullmatch(value) is None
                or value == "0" * 64
                for value in digests
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
            or self.corpus_size != CORPUS_SIZE
            or self.corpus_size != core.CORPUS_SIZE
        ):
            raise BioasqP1FormalSourceError(
                "formal_contract_invalid",
                "formal source contract is invalid",
            )
        for family in FAMILIES:
            required = sum(
                self.block_family_quotas[block][family]
                for block in BLOCKS
            )
            if (
                self.qualification_contract
                .minimum_components_per_family[family]
                < required
            ):
                raise BioasqP1FormalSourceError(
                    "formal_contract_capacity_invalid",
                    "formal quota exceeds the P0-qualified component demand",
                )


DEFAULT_CONTRACT = FormalSourceContract(
    source_contract=p0.SourceFileContract(
        size_bytes=OFFICIAL_SOURCE_SIZE_BYTES,
        sha256=OFFICIAL_SOURCE_SHA256,
    ),
    qualification_contract=p0.DEFAULT_CONTRACT,
    p0_receipt_file_sha256=P0_SAFE_RECEIPT_FILE_SHA256,
    p0_receipt_self_sha256=P0_SAFE_RECEIPT_SELF_SHA256,
    private_manifest_file_sha256=P0_PRIVATE_MANIFEST_FILE_SHA256,
    private_manifest_self_sha256=P0_PRIVATE_MANIFEST_SELF_SHA256,
    p0_implementation_sha256=P0_IMPLEMENTATION_SHA256,
    typed_core_sha256=TYPED_CORE_SHA256,
    block_family_quotas=DEFAULT_BLOCK_FAMILY_QUOTAS,
)


@dataclass(frozen=True, slots=True)
class FormalOutputPaths:
    """Exclusive output paths, mirroring the DSTC9 formal-source surface."""

    private_selection_secret: Path
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
            self.private_selection_secret,
            self.public_corpus,
            *tuple(self.public_blocks()[block] for block in BLOCKS),
            *tuple(self.private_qrels()[block] for block in QREL_BLOCKS),
            self.safe_selection_receipt,
        )


@dataclass(slots=True)
class _Audit:
    stage: str = "preflight"
    formal_source_access_count: int = 0
    source_open_count: int = 0
    source_hash_count: int = 0
    source_json_decode_count: int = 0
    selection_secret_generation_count: int = 0
    selection_secret_file_create_count: int = 0

    def source_payload(self) -> dict[str, int]:
        return {
            "formal_source_access_count": self.formal_source_access_count,
            "source_hash_count": self.source_hash_count,
            "source_json_decode_count": self.source_json_decode_count,
            "source_open_count": self.source_open_count,
        }


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    file_sha256: str
    size_bytes: int
    mode: int


@dataclass(frozen=True, slots=True)
class _P0Row:
    opaque_item_commitment: str
    family: str
    component_commitment: str
    query_commitment: str
    snippet_commitments: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "component_commitment": self.component_commitment,
            "family": self.family,
            "opaque_item_commitment": self.opaque_item_commitment,
            "query_commitment": self.query_commitment,
            "snippet_commitments": list(self.snippet_commitments),
        }


@dataclass(frozen=True, slots=True)
class _Snippet:
    commitment: str
    document: str
    text: str


@dataclass(frozen=True, slots=True)
class _SourceRow:
    opaque_item_commitment: str
    family: str
    component_commitment: str
    query_commitment: str
    query_text: str
    document_commitments: tuple[str, ...]
    snippets: tuple[_Snippet, ...]

    @property
    def snippet_commitments(self) -> tuple[str, ...]:
        return tuple(snippet.commitment for snippet in self.snippets)

    def p0_payload(self) -> dict[str, Any]:
        return {
            "component_commitment": self.component_commitment,
            "family": self.family,
            "opaque_item_commitment": self.opaque_item_commitment,
            "query_commitment": self.query_commitment,
            "snippet_commitments": list(self.snippet_commitments),
        }


@dataclass(frozen=True, slots=True)
class _SelectedRow:
    block: str
    family: str
    component_commitment: str
    selection_hmac_sha256: str
    work_id: str
    source: _SourceRow


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse: int
    capacity: int


class _DuplicateJsonKey(ValueError):
    pass


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    """Return the frozen ASCII canonical-JSON representation."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BioasqP1FormalSourceError(
            "canonical_json_failure",
            "formal value is not canonical JSON",
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise BioasqP1FormalSourceError(
            "self_hash_duplicate",
            "self hash was supplied twice",
        )
    value = dict(body)
    value["self_sha256"] = stable_hash(value)
    return value


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise BioasqP1FormalSourceError(
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
    if not hmac.compare_digest(stable_hash(body), claimed):
        raise BioasqP1FormalSourceError(
            "self_hash_mismatch",
            f"{field_name} self hash drifted",
        )
    return claimed


def _reject_constant(_value: str) -> None:
    raise BioasqP1FormalSourceError(
        "strict_json_nonfinite",
        "JSON contains a non-finite number",
    )


def _no_duplicate_object(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _decode_strict_json(raw: bytes, *, field_name: str) -> object:
    if not isinstance(raw, bytes) or not raw:
        raise BioasqP1FormalSourceError(
            "strict_json_empty",
            f"{field_name} bytes are empty",
        )
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateJsonKey as exc:
        raise BioasqP1FormalSourceError(
            "strict_json_duplicate_key",
            f"{field_name} contains a duplicate object key",
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BioasqP1FormalSourceError(
            "strict_json_invalid",
            f"{field_name} is not strict UTF-8 JSON",
        ) from exc


def _assert_no_symlink_components(path: Path, field_name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise BioasqP1FormalSourceError(
                "path_symlink_component",
                f"{field_name} contains a symbolic-link component",
            )


def _read_regular_file_once(
    path: Path,
    *,
    field_name: str,
    expected_mode: int,
) -> tuple[bytes, _FileIdentity]:
    absolute = path.absolute()
    _assert_no_symlink_components(absolute, field_name)
    try:
        before = absolute.lstat()
    except OSError as exc:
        raise BioasqP1FormalSourceError(
            "bound_input_unavailable",
            f"{field_name} is unavailable",
        ) from exc
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise BioasqP1FormalSourceError(
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
            ) != (before.st_dev, before.st_ino, before.st_size):
                raise BioasqP1FormalSourceError(
                    "bound_input_changed",
                    f"{field_name} changed during open",
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, READ_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ):
                raise BioasqP1FormalSourceError(
                    "bound_input_changed",
                    f"{field_name} changed while being read",
                )
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise BioasqP1FormalSourceError(
            "bound_input_read_failed",
            f"{field_name} could not be read",
        ) from exc
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise BioasqP1FormalSourceError(
            "bound_input_changed",
            f"{field_name} size drifted while being read",
        )
    return raw, _FileIdentity(
        file_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        mode=expected_mode,
    )


def _read_bound_canonical_json(
    path: Path,
    *,
    field_name: str,
    expected_file_sha256: str,
    expected_mode: int = 0o600,
) -> tuple[dict[str, Any], _FileIdentity]:
    raw, identity = _read_regular_file_once(
        path,
        field_name=field_name,
        expected_mode=expected_mode,
    )
    if not hmac.compare_digest(
        identity.file_sha256,
        expected_file_sha256,
    ):
        raise BioasqP1FormalSourceError(
            "bound_input_identity",
            f"{field_name} byte identity drifted",
        )
    value = _decode_strict_json(raw, field_name=field_name)
    if (
        not isinstance(value, dict)
        or raw != canonical_bytes(value, newline=True)
    ):
        raise BioasqP1FormalSourceError(
            "bound_input_noncanonical",
            f"{field_name} is not canonical JSON",
        )
    return value, identity


def _module_binding(
    module: object,
    *,
    expected_sha256: str,
    expected_version: str,
    field_name: str,
) -> dict[str, str]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise BioasqP1FormalSourceError(
            "implementation_binding_unavailable",
            f"{field_name} path is unavailable",
        )
    path = Path(raw_path).absolute()
    raw, identity = _read_regular_file_once(
        path,
        field_name=field_name,
        expected_mode=stat.S_IMODE(path.lstat().st_mode),
    )
    del raw
    if (
        not hmac.compare_digest(identity.file_sha256, expected_sha256)
        or getattr(module, "VERSION", None) != expected_version
        or getattr(module, "STUDY_ID", None) != STUDY_ID
    ):
        raise BioasqP1FormalSourceError(
            "implementation_binding_mismatch",
            f"{field_name} binding drifted",
        )
    return {
        "sha256": identity.file_sha256,
        "study_id": STUDY_ID,
        "version": expected_version,
    }


def _expected_source_binding(
    contract: FormalSourceContract,
) -> dict[str, Any]:
    return {
        "file_sha256": contract.source_contract.sha256,
        "size_bytes": contract.source_contract.size_bytes,
        "synthetic_source_free_canary_input": False,
    }


def _validate_p0_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: FormalSourceContract,
    private_identity: _FileIdentity,
) -> None:
    if set(receipt) != P0_SAFE_RECEIPT_KEYS:
        raise BioasqP1FormalSourceError(
            "p0_receipt_schema",
            "P0 safe receipt schema drifted",
        )
    self_hash = _verify_self_hash(receipt, "P0 safe receipt")
    if (
        not hmac.compare_digest(
            self_hash,
            contract.p0_receipt_self_sha256,
        )
        or receipt.get("schema") != p0.SAFE_RECEIPT_SCHEMA
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("status")
        != "qualified_public_non_scoring_schema_component_capacity"
        or receipt.get("source_binding")
        != _expected_source_binding(contract)
        or receipt.get("source_question_count")
        != contract.qualification_contract.expected_question_count
    ):
        raise BioasqP1FormalSourceError(
            "p0_receipt_binding",
            "P0 safe receipt binding drifted",
        )
    if receipt.get("access_boundary") != {
        "action_model_retrieval_evaluator_or_score_count": 0,
        "cohort_assignment_or_selection_secret_count": 0,
        "individual_item_query_document_snippet_or_commitment_published": False,
        "online_or_API_evaluation_count": 0,
        "real_source_access_count": 1,
        "source_hash_count": 1,
        "source_json_decode_count": 1,
        "source_open_count": 1,
    }:
        raise BioasqP1FormalSourceError(
            "p0_access_boundary",
            "P0 access boundary drifted",
        )
    private_binding = receipt.get("private_manifest_binding")
    if (
        not isinstance(private_binding, Mapping)
        or set(private_binding)
        != {"file_sha256", "row_count", "self_sha256"}
        or private_binding.get("file_sha256")
        != contract.private_manifest_file_sha256
        or private_binding.get("self_sha256")
        != contract.private_manifest_self_sha256
        or private_binding.get("file_sha256")
        != private_identity.file_sha256
        or type(private_binding.get("row_count")) is not int
        or private_binding["row_count"] < 1
    ):
        raise BioasqP1FormalSourceError(
            "p0_private_binding",
            "P0 private-manifest binding drifted",
        )
    capacity = receipt.get("capacity")
    if (
        not isinstance(capacity, Mapping)
        or set(capacity)
        != {
            "assignable_component_count_by_family",
            "demand_by_family",
            "maximum_flow_assigned_count_by_family",
            "maximum_flow_assigned_total",
            "required_total",
            "simultaneous_component_capacity_saturated",
        }
        or capacity.get("simultaneous_component_capacity_saturated")
        is not True
    ):
        raise BioasqP1FormalSourceError(
            "p0_capacity_schema",
            "P0 component capacity schema drifted",
        )
    for key in (
        "assignable_component_count_by_family",
        "demand_by_family",
        "maximum_flow_assigned_count_by_family",
    ):
        values = capacity.get(key)
        if not isinstance(values, Mapping) or set(values) != set(FAMILIES):
            raise BioasqP1FormalSourceError(
                "p0_capacity_schema",
                "P0 family capacity registry drifted",
            )
        for family in FAMILIES:
            observed = values.get(family)
            required = sum(
                contract.block_family_quotas[block][family]
                for block in BLOCKS
            )
            if type(observed) is not int or observed < required:
                raise BioasqP1FormalSourceError(
                    "p0_capacity_insufficient",
                    "P0 capacity does not cover the formal family quota",
                )
    source_counts = receipt.get("source_question_count_by_family")
    eligible_counts = receipt.get("eligible_question_count_by_family")
    if (
        not isinstance(source_counts, Mapping)
        or set(source_counts) != set(FAMILIES)
        or not isinstance(eligible_counts, Mapping)
        or set(eligible_counts) != set(FAMILIES)
        or any(
            type(source_counts[family]) is not int
            or source_counts[family] < 0
            or type(eligible_counts[family]) is not int
            or eligible_counts[family] < 0
            or eligible_counts[family] > source_counts[family]
            for family in FAMILIES
        )
        or sum(source_counts.values())
        != contract.qualification_contract.expected_question_count
    ):
        raise BioasqP1FormalSourceError(
            "p0_family_aggregate",
            "P0 family aggregate drifted",
        )
    component = receipt.get("component_aggregate")
    if (
        not isinstance(component, Mapping)
        or set(component)
        != {
            "component_count",
            "component_family_profile_counts",
            "multi_question_component_count",
            "row_count_in_multi_question_components",
        }
        or any(
            type(component.get(key)) is not int
            or component[key] < 0
            for key in (
                "component_count",
                "multi_question_component_count",
                "row_count_in_multi_question_components",
            )
        )
        or not isinstance(
            component.get("component_family_profile_counts"),
            Mapping,
        )
        or not isinstance(
            receipt.get("formal_ineligible_reason_counts"),
            Mapping,
        )
    ):
        raise BioasqP1FormalSourceError(
            "p0_component_aggregate",
            "P0 component aggregate drifted",
        )


def _validate_private_manifest(
    manifest: Mapping[str, Any],
    *,
    contract: FormalSourceContract,
    receipt: Mapping[str, Any],
) -> tuple[_P0Row, ...]:
    if set(manifest) != P0_PRIVATE_MANIFEST_KEYS:
        raise BioasqP1FormalSourceError(
            "private_manifest_schema",
            "P0 private manifest schema drifted",
        )
    self_hash = _verify_self_hash(manifest, "P0 private manifest")
    if (
        not hmac.compare_digest(
            self_hash,
            contract.private_manifest_self_sha256,
        )
        or manifest.get("schema") != p0.PRIVATE_MANIFEST_SCHEMA
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("status")
        != "private_noncohort_component_commitments"
        or manifest.get("component_rule") != p0.COMPONENT_RULE
        or manifest.get("family_order") != list(FAMILIES)
        or manifest.get("source_binding")
        != _expected_source_binding(contract)
        or manifest.get("claim_boundary")
        != {
            "action_model_retrieval_evaluator_or_score_count": 0,
            "cohort_assignment_or_selection_secret_count": 0,
            "contains_source_text_document_identifier_or_qrel_value": False,
            "noncohort_eligibility_commitments_only": True,
        }
    ):
        raise BioasqP1FormalSourceError(
            "private_manifest_binding",
            "P0 private manifest binding drifted",
        )
    raw_rows = manifest.get("rows")
    receipt_binding = receipt.get("private_manifest_binding")
    if (
        not isinstance(raw_rows, list)
        or not isinstance(receipt_binding, Mapping)
        or len(raw_rows) != receipt_binding.get("row_count")
    ):
        raise BioasqP1FormalSourceError(
            "private_manifest_row_count",
            "P0 private manifest row count drifted",
        )
    rows: list[_P0Row] = []
    seen_items: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != P0_PRIVATE_ROW_KEYS:
            raise BioasqP1FormalSourceError(
                "private_manifest_row_schema",
                "P0 private manifest row schema drifted",
            )
        item = _required_sha256(
            raw.get("opaque_item_commitment"),
            "P0 opaque item commitment",
        )
        component = _required_sha256(
            raw.get("component_commitment"),
            "P0 component commitment",
        )
        query = _required_sha256(
            raw.get("query_commitment"),
            "P0 query commitment",
        )
        family = raw.get("family")
        snippets = raw.get("snippet_commitments")
        if (
            family not in FAMILIES
            or item in seen_items
            or not isinstance(snippets, list)
            or not snippets
            or any(
                not isinstance(value, str)
                or _HEX64.fullmatch(value) is None
                for value in snippets
            )
            or snippets != sorted(set(snippets))
        ):
            raise BioasqP1FormalSourceError(
                "private_manifest_row_binding",
                "P0 private manifest row binding drifted",
            )
        seen_items.add(item)
        rows.append(
            _P0Row(
                opaque_item_commitment=item,
                family=family,
                component_commitment=component,
                query_commitment=query,
                snippet_commitments=tuple(snippets),
            )
        )
    expected = sorted(
        rows,
        key=lambda row: (
            row.component_commitment,
            FAMILIES.index(row.family),
            row.opaque_item_commitment,
        ),
    )
    if rows != expected:
        raise BioasqP1FormalSourceError(
            "private_manifest_order",
            "P0 private manifest row order drifted",
        )
    return tuple(rows)


def _normalize_text(
    value: object,
    *,
    field_name: str,
    casefold: bool,
    maximum_length: int = _MAX_TEXT_CHARACTERS,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum_length
        or "\x00" in value
    ):
        raise BioasqP1FormalSourceError(
            "source_text_schema",
            f"{field_name} schema drifted",
        )
    normalized = unicodedata.normalize("NFKC", value)
    if casefold:
        normalized = normalized.casefold()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise BioasqP1FormalSourceError(
            "source_text_schema",
            f"{field_name} schema drifted",
        )
    return normalized


def _family(value: object) -> str:
    if not isinstance(value, str):
        raise BioasqP1FormalSourceError(
            "source_family_schema",
            "source question family schema drifted",
        )
    normalized = " ".join(value.strip().casefold().split())
    if normalized not in FAMILIES:
        raise BioasqP1FormalSourceError(
            "source_family_registry",
            "source question family is outside the frozen registry",
        )
    return normalized


def _string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BioasqP1FormalSourceError(
            "source_list_schema",
            f"{field_name} schema drifted",
        )
    rows = tuple(
        _normalize_text(
            row,
            field_name=field_name,
            casefold=False,
            maximum_length=_MAX_IDENTIFIER_CHARACTERS,
        )
        for row in value
    )
    if len(set(rows)) != len(rows):
        raise BioasqP1FormalSourceError(
            "source_list_duplicate",
            f"{field_name} contains duplicate normalized values",
        )
    return rows


def _p0_commit(kind: str, normalized_value: str) -> str:
    """Reproduce the exact public P0 commitment namespace."""

    return stable_hash(
        {
            "kind": kind,
            "normalized_value": normalized_value,
            "version": p0.VERSION,
        }
    )


def _componentize(
    rows: Sequence[_SourceRow],
) -> tuple[
    tuple[_SourceRow, ...],
    Mapping[str, frozenset[str]],
    Mapping[str, Any],
]:
    if not rows:
        raise BioasqP1FormalSourceError(
            "source_no_eligible_rows",
            "source contains no eligible rows",
        )
    union = _DisjointSet(len(rows))
    owners: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = (
            (("query", row.query_commitment),)
            + tuple(
                ("document", commitment)
                for commitment in row.document_commitments
            )
            + tuple(
                ("snippet", commitment)
                for commitment in row.snippet_commitments
            )
        )
        for key in keys:
            previous = owners.setdefault(key, index)
            union.union(index, previous)
    grouped: dict[int, list[int]] = {}
    for index in range(len(rows)):
        grouped.setdefault(union.find(index), []).append(index)

    completed: list[_SourceRow] = []
    component_families: dict[str, frozenset[str]] = {}
    profiles: Counter[str] = Counter()
    multi_count = 0
    multi_rows = 0
    for indices in grouped.values():
        ordered = tuple(
            sorted(
                (rows[index] for index in indices),
                key=lambda row: row.opaque_item_commitment,
            )
        )
        component = stable_hash(
            {
                "component_rule": p0.COMPONENT_RULE,
                "opaque_item_commitments": [
                    row.opaque_item_commitment for row in ordered
                ],
            }
        )
        families = frozenset(row.family for row in ordered)
        component_families[component] = families
        profiles[
            "+".join(
                family for family in FAMILIES if family in families
            )
        ] += 1
        if len(ordered) > 1:
            multi_count += 1
            multi_rows += len(ordered)
        completed.extend(
            _SourceRow(
                opaque_item_commitment=row.opaque_item_commitment,
                family=row.family,
                component_commitment=component,
                query_commitment=row.query_commitment,
                query_text=row.query_text,
                document_commitments=row.document_commitments,
                snippets=row.snippets,
            )
            for row in ordered
        )
    completed.sort(
        key=lambda row: (
            row.component_commitment,
            FAMILIES.index(row.family),
            row.opaque_item_commitment,
        )
    )
    return (
        tuple(completed),
        component_families,
        {
            "component_count": len(grouped),
            "component_family_profile_counts": dict(sorted(profiles.items())),
            "multi_question_component_count": multi_count,
            "row_count_in_multi_question_components": multi_rows,
        },
    )


def _parse_and_reconstruct_source(
    decoded: object,
    *,
    contract: FormalSourceContract,
) -> tuple[
    tuple[_SourceRow, ...],
    Mapping[str, _Snippet],
    Mapping[str, frozenset[str]],
    Mapping[str, Any],
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
]:
    if not isinstance(decoded, Mapping) or set(decoded) != {"questions"}:
        raise BioasqP1FormalSourceError(
            "source_root_schema",
            "source root must contain only questions",
        )
    raw_questions = decoded.get("questions")
    if (
        not isinstance(raw_questions, list)
        or len(raw_questions)
        != contract.qualification_contract.expected_question_count
    ):
        raise BioasqP1FormalSourceError(
            "source_question_count",
            "source question count drifted",
        )
    preliminary: list[_SourceRow] = []
    source_family_counts: Counter[str] = Counter()
    eligible_family_counts: Counter[str] = Counter()
    ineligible: Counter[str] = Counter()
    seen_items: set[str] = set()
    snippet_registry: dict[str, _Snippet] = {}
    for raw_question in raw_questions:
        if not isinstance(raw_question, Mapping):
            raise BioasqP1FormalSourceError(
                "source_question_schema",
                "source question row schema drifted",
            )
        if not {
            "body",
            "documents",
            "id",
            "snippets",
            "type",
        } <= set(raw_question):
            raise BioasqP1FormalSourceError(
                "source_question_schema",
                "source question is missing a required P0 field",
            )
        family = _family(raw_question.get("type"))
        source_family_counts[family] += 1
        source_id = _normalize_text(
            raw_question.get("id"),
            field_name="question id",
            casefold=False,
            maximum_length=_MAX_IDENTIFIER_CHARACTERS,
        )
        item = _p0_commit("question_id", source_id)
        if item in seen_items:
            raise BioasqP1FormalSourceError(
                "source_question_id_duplicate",
                "source question id registry contains a duplicate",
            )
        seen_items.add(item)
        try:
            query_text = _normalize_text(
                raw_question.get("body"),
                field_name="question body",
                casefold=False,
            )
            query_identity = _normalize_text(
                raw_question.get("body"),
                field_name="question body",
                casefold=True,
            )
        except BioasqP1FormalSourceError:
            ineligible["empty_or_invalid_query"] += 1
            continue
        try:
            checked_query = core.validate_query_text(query_text)
        except core.BioasqP1TypedCoreError as exc:
            raise BioasqP1FormalSourceError(
                "typed_query_contract",
                "source query is not consumable by the frozen typed core",
            ) from exc
        if checked_query != query_text:
            raise BioasqP1FormalSourceError(
                "typed_query_normalization",
                "typed-core query normalization differs from P0 projection",
            )
        documents = _string_list(
            raw_question.get("documents"),
            field_name="question documents",
        )
        document_set = set(documents)
        raw_snippets = raw_question.get("snippets")
        if not isinstance(raw_snippets, list):
            raise BioasqP1FormalSourceError(
                "source_snippet_schema",
                "source snippets schema drifted",
            )
        snippets_by_pair: dict[tuple[str, str], _Snippet] = {}
        observed_snippet_documents: set[str] = set()
        for raw_snippet in raw_snippets:
            if (
                not isinstance(raw_snippet, Mapping)
                or not {"document", "text"} <= set(raw_snippet)
            ):
                raise BioasqP1FormalSourceError(
                    "source_snippet_schema",
                    "source snippet row schema drifted",
                )
            document = _normalize_text(
                raw_snippet.get("document"),
                field_name="snippet document",
                casefold=False,
                maximum_length=_MAX_IDENTIFIER_CHARACTERS,
            )
            if document not in document_set:
                raise BioasqP1FormalSourceError(
                    "source_snippet_document",
                    "snippet document is absent from question documents",
                )
            observed_snippet_documents.add(document)
            try:
                text = _normalize_text(
                    raw_snippet.get("text"),
                    field_name="snippet text",
                    casefold=False,
                )
            except BioasqP1FormalSourceError:
                continue
            commitment = _p0_commit(
                "normalized_gold_snippet",
                document + "\0" + text,
            )
            snippet = _Snippet(
                commitment=commitment,
                document=document,
                text=text,
            )
            pair = (document, text)
            snippets_by_pair[pair] = snippet
            previous = snippet_registry.setdefault(commitment, snippet)
            if (
                previous.document != document
                or previous.text != text
            ):
                raise BioasqP1FormalSourceError(
                    "source_snippet_hash_collision",
                    "distinct snippets share one P0 commitment",
                )
        if not documents:
            ineligible["no_gold_document"] += 1
            continue
        if not snippets_by_pair:
            ineligible["no_nonempty_gold_snippet"] += 1
            continue
        if not observed_snippet_documents:
            ineligible["no_snippet_document"] += 1
            continue
        eligible_family_counts[family] += 1
        preliminary.append(
            _SourceRow(
                opaque_item_commitment=item,
                family=family,
                component_commitment="",
                query_commitment=_p0_commit(
                    "normalized_query",
                    query_identity,
                ),
                query_text=query_text,
                document_commitments=tuple(
                    sorted(
                        _p0_commit("gold_document", document)
                        for document in documents
                    )
                ),
                snippets=tuple(
                    sorted(
                        snippets_by_pair.values(),
                        key=lambda snippet: snippet.commitment,
                    )
                ),
            )
        )
    rows, component_families, component_aggregate = _componentize(
        preliminary
    )
    return (
        rows,
        snippet_registry,
        component_families,
        component_aggregate,
        {family: source_family_counts[family] for family in FAMILIES},
        {family: eligible_family_counts[family] for family in FAMILIES},
        dict(sorted(ineligible.items())),
    )


def _match_p0_commitments(
    rows: Sequence[_SourceRow],
    p0_rows: Sequence[_P0Row],
    *,
    receipt: Mapping[str, Any],
    component_aggregate: Mapping[str, Any],
    source_family_counts: Mapping[str, int],
    eligible_family_counts: Mapping[str, int],
    ineligible: Mapping[str, int],
) -> None:
    if [row.p0_payload() for row in rows] != [
        row.payload() for row in p0_rows
    ]:
        raise BioasqP1FormalSourceError(
            "p0_commitment_reconstruction",
            "formal source does not reproduce the complete P0 commitment set",
        )
    if (
        dict(component_aggregate) != receipt.get("component_aggregate")
        or dict(source_family_counts)
        != receipt.get("source_question_count_by_family")
        or dict(eligible_family_counts)
        != receipt.get("eligible_question_count_by_family")
        or dict(ineligible)
        != receipt.get("formal_ineligible_reason_counts")
    ):
        raise BioasqP1FormalSourceError(
            "p0_aggregate_reconstruction",
            "formal source does not reproduce the P0 safe aggregates",
        )


def _hmac_digest(
    secret: bytes,
    purpose: str,
    values: Mapping[str, Any],
) -> bytes:
    return hmac.new(
        secret,
        canonical_bytes(
            {
                **dict(values),
                "purpose": purpose,
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        ),
        hashlib.sha256,
    ).digest()


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    left: int,
    right: int,
    capacity: int,
) -> _FlowEdge:
    forward = _FlowEdge(right, len(graph[right]), capacity)
    reverse = _FlowEdge(left, len(graph[left]), 0)
    graph[left].append(forward)
    graph[right].append(reverse)
    return forward


def _joint_component_family_assignment(
    component_families: Mapping[str, frozenset[str]],
    *,
    secret: bytes,
    demands: Mapping[str, int],
) -> Mapping[str, str]:
    components = sorted(
        component_families,
        key=lambda component: (
            _hmac_digest(
                secret,
                "component_joint_order",
                {"component_commitment": component},
            ),
            component,
        ),
    )
    source_node = 0
    component_offset = 1
    family_offset = component_offset + len(components)
    sink = family_offset + len(FAMILIES)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    family_nodes = {
        family: family_offset + index
        for index, family in enumerate(FAMILIES)
    }
    source_edges: dict[str, _FlowEdge] = {}
    assignment_edges: dict[tuple[str, str], _FlowEdge] = {}
    for index, component in enumerate(components):
        node = component_offset + index
        source_edges[component] = _add_flow_edge(
            graph,
            source_node,
            node,
            1,
        )
        ordered_families = sorted(
            component_families[component],
            key=lambda family: (
                _hmac_digest(
                    secret,
                    "component_family_preference",
                    {
                        "component_commitment": component,
                        "family": family,
                    },
                ),
                FAMILIES.index(family),
            ),
        )
        for family in ordered_families:
            assignment_edges[(component, family)] = _add_flow_edge(
                graph,
                node,
                family_nodes[family],
                1,
            )
    for family in FAMILIES:
        _add_flow_edge(
            graph,
            family_nodes[family],
            sink,
            demands[family],
        )

    flow = 0
    required = sum(demands.values())
    while flow < required:
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        previous[source_node] = (-1, -1)
        queue: deque[int] = deque((source_node,))
        while queue and previous[sink] is None:
            node = queue.popleft()
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0 or previous[edge.to] is not None:
                    continue
                previous[edge.to] = (node, edge_index)
                queue.append(edge.to)
        if previous[sink] is None:
            break
        node = sink
        while node != source_node:
            parent_edge = previous[node]
            if parent_edge is None:
                raise BioasqP1FormalSourceError(
                    "selection_flow_predecessor",
                    "joint component assignment predecessor drifted",
                )
            parent, edge_index = parent_edge
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1
    if flow != required:
        raise BioasqP1FormalSourceError(
            "selection_component_capacity",
            "joint component assignment cannot satisfy all family quotas",
        )
    assignment: dict[str, str] = {}
    for component in components:
        if source_edges[component].capacity != 0:
            continue
        matched = [
            family
            for family in FAMILIES
            if (component, family) in assignment_edges
            and assignment_edges[(component, family)].capacity == 0
        ]
        if len(matched) != 1:
            raise BioasqP1FormalSourceError(
                "selection_component_assignment",
                "selected component does not have exactly one family",
            )
        assignment[component] = matched[0]
    if len(assignment) != required:
        raise BioasqP1FormalSourceError(
            "selection_component_count",
            "joint component assignment count drifted",
        )
    return assignment


def _select_rows(
    rows: Sequence[_SourceRow],
    component_families: Mapping[str, frozenset[str]],
    *,
    secret: bytes,
    contract: FormalSourceContract,
) -> Mapping[str, tuple[_SelectedRow, ...]]:
    if not isinstance(secret, bytes) or len(secret) != HMAC_SECRET_BYTES:
        raise BioasqP1FormalSourceError(
            "selection_secret_invalid",
            "whole-study HMAC secret must be exactly 32 bytes",
        )
    demands = {
        family: sum(
            contract.block_family_quotas[block][family]
            for block in BLOCKS
        )
        for family in FAMILIES
    }
    assignment = _joint_component_family_assignment(
        component_families,
        secret=secret,
        demands=demands,
    )
    rows_by_component_family: dict[
        tuple[str, str], list[_SourceRow]
    ] = defaultdict(list)
    for row in rows:
        rows_by_component_family[
            (row.component_commitment, row.family)
        ].append(row)
    representatives: dict[str, list[tuple[bytes, _SourceRow]]] = {
        family: [] for family in FAMILIES
    }
    for component, family in assignment.items():
        candidates = rows_by_component_family.get((component, family), [])
        if not candidates:
            raise BioasqP1FormalSourceError(
                "selection_representative_missing",
                "assigned component has no question in its assigned family",
            )
        ranked = sorted(
            candidates,
            key=lambda row: (
                _hmac_digest(
                    secret,
                    "component_question_representative",
                    {
                        "component_commitment": component,
                        "family": family,
                        "opaque_item_commitment": (
                            row.opaque_item_commitment
                        ),
                    },
                ),
                row.opaque_item_commitment,
            ),
        )
        selected = ranked[0]
        order_digest = _hmac_digest(
            secret,
            "family_block_allocation_order",
            {
                "component_commitment": component,
                "family": family,
                "opaque_item_commitment": (
                    selected.opaque_item_commitment
                ),
            },
        )
        representatives[family].append((order_digest, selected))
    for family in FAMILIES:
        representatives[family].sort(
            key=lambda pair: (
                pair[0],
                pair[1].opaque_item_commitment,
            )
        )
        if len(representatives[family]) != demands[family]:
            raise BioasqP1FormalSourceError(
                "selection_family_count",
                "selected component family count drifted",
            )

    selected_by_block: dict[str, list[_SelectedRow]] = {
        block: [] for block in BLOCKS
    }
    for family in FAMILIES:
        offset = 0
        for block in BLOCKS:
            count = contract.block_family_quotas[block][family]
            for allocation_digest, source in representatives[family][
                offset : offset + count
            ]:
                work_digest = _hmac_digest(
                    secret,
                    "opaque_work_id",
                    {
                        "opaque_item_commitment": (
                            source.opaque_item_commitment
                        ),
                        "rule": WORK_ID_RULE,
                    },
                ).hex()
                work_id = "bioasq-work-v2-" + work_digest
                selected_by_block[block].append(
                    _SelectedRow(
                        block=block,
                        family=family,
                        component_commitment=(
                            source.component_commitment
                        ),
                        selection_hmac_sha256=(
                            allocation_digest.hex()
                        ),
                        work_id=work_id,
                        source=source,
                    )
                )
            offset += count
    for block in BLOCKS:
        selected_by_block[block].sort(
            key=lambda row: (
                _hmac_digest(
                    secret,
                    "public_block_item_order",
                    {"block": block, "work_id": row.work_id},
                ),
                row.work_id,
            )
        )
    flattened = [
        row for block in BLOCKS for row in selected_by_block[block]
    ]
    components = [row.component_commitment for row in flattened]
    items = [row.source.opaque_item_commitment for row in flattened]
    work_ids = [row.work_id for row in flattened]
    if (
        len(set(components)) != len(components)
        or len(set(items)) != len(items)
        or len(set(work_ids)) != len(work_ids)
        or any(_WORK_ID.fullmatch(value) is None for value in work_ids)
    ):
        raise BioasqP1FormalSourceError(
            "selection_overlap",
            "whole-study selection is not component-disjoint",
        )
    for block in BLOCKS:
        counts = Counter(row.family for row in selected_by_block[block])
        if counts != Counter(contract.block_family_quotas[block]):
            raise BioasqP1FormalSourceError(
                "selection_quota",
                "selected block family quota drifted",
            )
    return {
        block: tuple(selected_by_block[block])
        for block in BLOCKS
    }


def _build_corpus(
    rows: Sequence[_SourceRow],
    snippet_registry: Mapping[str, _Snippet],
    selected: Mapping[str, Sequence[_SelectedRow]],
    *,
    secret: bytes,
    corpus_size: int,
) -> tuple[
    tuple[dict[str, object], ...],
    Mapping[str, int],
    int,
    int,
]:
    eligible_commitments = {
        commitment
        for row in rows
        for commitment in row.snippet_commitments
    }
    if eligible_commitments != set(snippet_registry):
        raise BioasqP1FormalSourceError(
            "corpus_snippet_registry",
            "eligible snippet registry is not total",
        )
    selected_qrels = {
        commitment
        for block in BLOCKS
        for row in selected[block]
        for commitment in row.source.snippet_commitments
    }
    if len(selected_qrels) > corpus_size:
        raise BioasqP1FormalSourceError(
            "corpus_selected_qrel_overflow",
            "selected unique qrels exceed the frozen corpus size",
        )
    remaining = sorted(
        eligible_commitments - selected_qrels,
        key=lambda commitment: (
            _hmac_digest(
                secret,
                "corpus_filler_membership_order",
                {"snippet_commitment": commitment},
            ),
            commitment,
        ),
    )
    filler_count = corpus_size - len(selected_qrels)
    if len(remaining) < filler_count:
        raise BioasqP1FormalSourceError(
            "corpus_filler_capacity",
            "eligible unique snippet capacity is below 2900",
        )
    pool = set(selected_qrels)
    pool.update(remaining[:filler_count])
    if len(pool) != corpus_size:
        raise BioasqP1FormalSourceError(
            "corpus_membership_count",
            "frozen corpus membership count drifted",
        )
    ordered = sorted(
        pool,
        key=lambda commitment: (
            _hmac_digest(
                secret,
                "corpus_public_ordinal_order",
                {"snippet_commitment": commitment},
            ),
            commitment,
        ),
    )
    ordinal_by_commitment = {
        commitment: ordinal
        for ordinal, commitment in enumerate(ordered)
    }
    passages: list[dict[str, object]] = []
    for ordinal, commitment in enumerate(ordered):
        snippet = snippet_registry[commitment]
        try:
            passage = core.Passage(
                ordinal=ordinal,
                text=snippet.text,
            )
            payload = core.passage_public_payload(passage)
        except core.BioasqP1TypedCoreError as exc:
            raise BioasqP1FormalSourceError(
                "typed_corpus_contract",
                "formal passage violates the frozen typed core",
            ) from exc
        if set(payload) != PUBLIC_PASSAGE_KEYS:
            raise BioasqP1FormalSourceError(
                "public_corpus_projection",
                "public corpus passage projection drifted",
            )
        passages.append(payload)
    return (
        tuple(passages),
        ordinal_by_commitment,
        len(selected_qrels),
        filler_count,
    )


def _open_binary(path: Path) -> BinaryIO:
    """Single indirection used by tests to audit the sole source open."""

    return path.open("rb")


def _read_source_once(
    path: Path,
    contract: p0.SourceFileContract,
    *,
    audit: _Audit,
) -> bytes:
    absolute = path.absolute()
    _assert_no_symlink_components(absolute, "formal source")
    audit.stage = "formal_source_metadata"
    try:
        before = absolute.lstat()
    except OSError as exc:
        raise BioasqP1FormalSourceError(
            "source_unavailable",
            "formal source is unavailable",
        ) from exc
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size != contract.size_bytes
    ):
        raise BioasqP1FormalSourceError(
            "source_metadata",
            "formal source metadata drifted",
        )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    audit.stage = "formal_source_single_open_hash"
    audit.formal_source_access_count += 1
    if audit.formal_source_access_count != 1:
        raise BioasqP1FormalSourceError(
            "source_reopen",
            "formal source capability was consumed more than once",
        )
    try:
        handle = _open_binary(absolute)
        audit.source_open_count += 1
        with handle:
            opened = os.fstat(handle.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ) != (before.st_dev, before.st_ino, before.st_size):
                raise BioasqP1FormalSourceError(
                    "source_changed",
                    "formal source changed during open",
                )
            while True:
                chunk = handle.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
            after = os.fstat(handle.fileno())
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ):
                raise BioasqP1FormalSourceError(
                    "source_changed",
                    "formal source changed while being read",
                )
    except OSError as exc:
        raise BioasqP1FormalSourceError(
            "source_read",
            "formal source could not be read",
        ) from exc
    raw = b"".join(chunks)
    audit.source_hash_count += 1
    if (
        len(raw) != contract.size_bytes
        or not hmac.compare_digest(digest.hexdigest(), contract.sha256)
    ):
        raise BioasqP1FormalSourceError(
            "source_identity",
            "formal source byte identity drifted",
        )
    return raw


def _validate_outputs_fresh(paths: FormalOutputPaths) -> FormalOutputPaths:
    if not isinstance(paths, FormalOutputPaths):
        raise BioasqP1FormalSourceError(
            "output_contract_type",
            "formal output path contract drifted",
        )
    absolute = tuple(path.absolute() for path in paths.all_paths())
    if len(set(absolute)) != len(absolute):
        raise BioasqP1FormalSourceError(
            "output_path_overlap",
            "formal output paths are not distinct",
        )
    for path in absolute:
        if path.exists() or path.is_symlink():
            raise BioasqP1FormalSourceError(
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
            raise BioasqP1FormalSourceError(
                "output_parent",
                "formal output parent is invalid",
            )
    return FormalOutputPaths(*(absolute))  # type: ignore[arg-type]


def _write_exclusive_bytes(
    path: Path,
    raw: bytes,
    *,
    mode: int,
) -> _FileIdentity:
    if not isinstance(raw, bytes) or not raw:
        raise BioasqP1FormalSourceError(
            "output_bytes_invalid",
            "formal output bytes are invalid",
        )
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
                    raise BioasqP1FormalSourceError(
                        "output_write_stalled",
                        "formal output write stalled",
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise BioasqP1FormalSourceError(
            "output_write_failed",
            "formal output could not be written exclusively",
        ) from exc
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != mode
        or metadata.st_size != len(raw)
    ):
        raise BioasqP1FormalSourceError(
            "output_metadata",
            "formal output metadata drifted",
        )
    return _FileIdentity(
        file_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        mode=mode,
    )


def _write_exclusive_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    mode: int,
) -> _FileIdentity:
    return _write_exclusive_bytes(
        path,
        canonical_bytes(value, newline=True),
        mode=mode,
    )


def _create_selection_secret_once(
    path: Path,
    *,
    secret_factory: Callable[[int], bytes],
    audit: _Audit,
) -> tuple[bytes, _FileIdentity]:
    audit.selection_secret_generation_count += 1
    if audit.selection_secret_generation_count != 1:
        raise BioasqP1FormalSourceError(
            "selection_secret_regenerated",
            "whole-study selection secret was generated more than once",
        )
    try:
        secret = secret_factory(HMAC_SECRET_BYTES)
    except BaseException as exc:
        raise BioasqP1FormalSourceError(
            "selection_secret_generation",
            "whole-study selection secret generation failed",
        ) from exc
    if type(secret) is not bytes or len(secret) != HMAC_SECRET_BYTES:
        raise BioasqP1FormalSourceError(
            "selection_secret_invalid",
            "secret factory did not return exactly 32 bytes",
        )
    identity = _write_exclusive_bytes(path, secret, mode=0o600)
    audit.selection_secret_file_create_count += 1
    if audit.selection_secret_file_create_count != 1:
        raise BioasqP1FormalSourceError(
            "selection_secret_recreated",
            "whole-study selection secret file was created more than once",
        )
    return secret, identity


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


def _safe_failure_receipt(
    *,
    audit: _Audit,
    contract: FormalSourceContract,
    error: BioasqP1FormalSourceError,
) -> dict[str, Any]:
    return self_hashed(
        {
            "access_boundary": {
                **audit.source_payload(),
                "action_count": 0,
                "model_call_count": 0,
                "online_or_API_evaluation_count": 0,
                "score_count": 0,
                "selection_secret_file_create_count": (
                    audit.selection_secret_file_create_count
                ),
                "selection_secret_generation_count": (
                    audit.selection_secret_generation_count
                ),
            },
            "aggregate_only_public_receipt": True,
            "expected_binding": {
                "p0_private_manifest_file_sha256": (
                    contract.private_manifest_file_sha256
                ),
                "p0_safe_receipt_file_sha256": (
                    contract.p0_receipt_file_sha256
                ),
                "source_file_sha256": contract.source_contract.sha256,
                "source_size_bytes": contract.source_contract.size_bytes,
            },
            "failure_code": error.error_code,
            "failure_exception_message_sha256": hashlib.sha256(
                str(error).encode("utf-8", errors="replace")
            ).hexdigest(),
            "failure_exception_type_sha256": hashlib.sha256(
                type(error).__name__.encode("ascii", errors="replace")
            ).hexdigest(),
            "failure_stage": audit.stage,
            "individual_query_document_snippet_qrel_or_source_id_published": (
                False
            ),
            "retry_replay_resample_secret_rotation_source_or_parser_change": (
                False
            ),
            "schema": FAILURE_RECEIPT_SCHEMA,
            "status": "terminal_formal_source_failure_no_retry",
            "study_id": STUDY_ID,
            "version": VERSION,
        }
    )


def _try_write_failure_receipt(
    outputs: FormalOutputPaths,
    receipt: Mapping[str, Any],
) -> bool:
    path = outputs.safe_selection_receipt.absolute()
    try:
        if path.exists() or path.is_symlink() or not path.parent.is_dir():
            return False
        _assert_no_symlink_components(path.parent, "failure output parent")
        _write_exclusive_json(path, receipt, mode=0o600)
        return True
    except Exception:
        return False


def compile_formal_source(
    *,
    p0_receipt_path: Path,
    private_eligibility_manifest_path: Path,
    source_path: Path,
    outputs: FormalOutputPaths,
    contract: FormalSourceContract = DEFAULT_CONTRACT,
    _secret_factory: Callable[[int], bytes] | None = None,
) -> dict[str, Any]:
    """Consume one exact BioASQ source and emit all sealed study inputs.

    ``_secret_factory`` is a synthetic-test seam.  The default formal path
    invokes ``os.urandom(32)`` exactly once and never accepts an existing
    secret.  In either case the resulting bytes are created at
    ``outputs.private_selection_secret`` with one ``O_EXCL`` write.
    """

    audit = _Audit()
    normalized_outputs = outputs
    try:
        if not isinstance(contract, FormalSourceContract):
            raise BioasqP1FormalSourceError(
                "formal_contract_type",
                "formal source contract type drifted",
            )
        contract.__post_init__()
        normalized_outputs = _validate_outputs_fresh(outputs)

        audit.stage = "p0_implementation_binding"
        p0_binding = _module_binding(
            p0,
            expected_sha256=contract.p0_implementation_sha256,
            expected_version="bioasq_p0_public_source_qualification_v1",
            field_name="P0 implementation",
        )
        audit.stage = "typed_core_binding"
        typed_binding = _module_binding(
            core,
            expected_sha256=contract.typed_core_sha256,
            expected_version="bioasq_p1_typed_core_v1",
            field_name="typed core",
        )

        audit.stage = "private_manifest_identity"
        private_manifest, private_identity = _read_bound_canonical_json(
            private_eligibility_manifest_path,
            field_name="P0 private eligibility manifest",
            expected_file_sha256=(
                contract.private_manifest_file_sha256
            ),
        )
        audit.stage = "p0_receipt_identity"
        p0_receipt, p0_identity = _read_bound_canonical_json(
            p0_receipt_path,
            field_name="P0 safe qualification receipt",
            expected_file_sha256=contract.p0_receipt_file_sha256,
        )
        _validate_p0_receipt(
            p0_receipt,
            contract=contract,
            private_identity=private_identity,
        )
        audit.stage = "private_manifest_validation"
        p0_rows = _validate_private_manifest(
            private_manifest,
            contract=contract,
            receipt=p0_receipt,
        )

        audit.stage = "one_shot_selection_secret"
        factory = os.urandom if _secret_factory is None else _secret_factory
        secret, secret_identity = _create_selection_secret_once(
            normalized_outputs.private_selection_secret,
            secret_factory=factory,
            audit=audit,
        )
        secret_commitment = hashlib.sha256(secret).hexdigest()

        raw = _read_source_once(
            source_path,
            contract.source_contract,
            audit=audit,
        )
        audit.stage = "formal_source_strict_json_decode"
        audit.source_json_decode_count += 1
        if audit.source_json_decode_count != 1:
            raise BioasqP1FormalSourceError(
                "source_redecode",
                "formal source was decoded more than once",
            )
        decoded = _decode_strict_json(
            raw,
            field_name="formal source",
        )
        del raw

        audit.stage = "p0_commitment_reconstruction"
        (
            source_rows,
            snippet_registry,
            component_families,
            component_aggregate,
            source_family_counts,
            eligible_family_counts,
            ineligible,
        ) = _parse_and_reconstruct_source(
            decoded,
            contract=contract,
        )
        del decoded
        _match_p0_commitments(
            source_rows,
            p0_rows,
            receipt=p0_receipt,
            component_aggregate=component_aggregate,
            source_family_counts=source_family_counts,
            eligible_family_counts=eligible_family_counts,
            ineligible=ineligible,
        )

        audit.stage = "joint_component_hmac_selection"
        selected = _select_rows(
            source_rows,
            component_families,
            secret=secret,
            contract=contract,
        )
        selected_rows = [
            row for block in BLOCKS for row in selected[block]
        ]

        audit.stage = "fixed_2900_corpus"
        (
            passages,
            ordinal_by_commitment,
            selected_qrel_count,
            filler_count,
        ) = _build_corpus(
            source_rows,
            snippet_registry,
            selected,
            secret=secret,
            corpus_size=contract.corpus_size,
        )

        audit.stage = "public_private_projection"
        public_corpus = self_hashed(
            {
                "passages": list(passages),
                "schema": PUBLIC_CORPUS_SCHEMA,
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        )
        public_blocks: dict[str, dict[str, Any]] = {}
        private_qrels: dict[str, dict[str, Any]] = {}
        for block in BLOCKS:
            items = [
                {
                    "query_text": row.source.query_text,
                    "work_id": row.work_id,
                }
                for row in selected[block]
            ]
            if any(set(item) != PUBLIC_ITEM_KEYS for item in items):
                raise BioasqP1FormalSourceError(
                    "public_item_projection",
                    "public item projection contains a forbidden field",
                )
            public_blocks[block] = self_hashed(
                {
                    "block_id": block,
                    "items": items,
                    "schema": PUBLIC_BLOCK_SCHEMA,
                    "study_id": STUDY_ID,
                    "version": VERSION,
                }
            )
            if block in QREL_BLOCKS:
                qrel_rows = [
                    {
                        "family": row.family,
                        "gold_ordinals": sorted(
                            {
                                ordinal_by_commitment[commitment]
                                for commitment in (
                                    row.source.snippet_commitments
                                )
                            }
                        ),
                        "work_id": row.work_id,
                    }
                    for row in selected[block]
                ]
                if any(
                    set(row) != PRIVATE_QREL_ROW_KEYS
                    or not row["gold_ordinals"]
                    or row["gold_ordinals"]
                    != sorted(set(row["gold_ordinals"]))
                    for row in qrel_rows
                ):
                    raise BioasqP1FormalSourceError(
                        "private_qrel_projection",
                        "private set-valued qrel projection drifted",
                    )
                private_qrels[block] = self_hashed(
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
            normalized_outputs.public_corpus,
            public_corpus,
            mode=0o600,
        )
        public_identities: dict[str, _FileIdentity] = {}
        qrel_identities: dict[str, _FileIdentity] = {}
        for block in BLOCKS:
            public_identities[block] = _write_exclusive_json(
                normalized_outputs.public_blocks()[block],
                public_blocks[block],
                mode=0o400 if block == "M_search" else 0o600,
            )
        for block in QREL_BLOCKS:
            qrel_identities[block] = _write_exclusive_json(
                normalized_outputs.private_qrels()[block],
                private_qrels[block],
                mode=0o400,
            )

        selected_count = len(selected_rows)
        selected_components = {
            row.component_commitment for row in selected_rows
        }
        selected_items = {
            row.source.opaque_item_commitment
            for row in selected_rows
        }
        selected_queries = {
            row.source.query_commitment for row in selected_rows
        }
        corpus_sha256 = corpus_identity.file_sha256
        safe_receipt = self_hashed(
            {
                "artifact_binding": {
                    "private_qrels": {
                        block: _artifact_binding(
                            private_qrels[block],
                            qrel_identities[block],
                            row_count=len(selected[block]),
                        )
                        for block in QREL_BLOCKS
                    },
                    "private_selection_secret": {
                        "mode": f"{secret_identity.mode:04o}",
                        "selection_secret_commitment_sha256": (
                            secret_commitment
                        ),
                        "selection_secret_persisted_publicly": False,
                        "size_bytes": secret_identity.size_bytes,
                    },
                    "public_blocks": {
                        block: _artifact_binding(
                            public_blocks[block],
                            public_identities[block],
                            row_count=len(selected[block]),
                        )
                        for block in BLOCKS
                    },
                    "public_corpus": _artifact_binding(
                        public_corpus,
                        corpus_identity,
                        row_count=len(passages),
                    ),
                },
                "compiler_boundary": {
                    "action_count": 0,
                    "model_call_count": 0,
                    "online_or_API_evaluation_count": 0,
                    "score_count": 0,
                },
                "corpus_aggregate": {
                    "arm_corpus_file_sha256": {
                        "Agent": corpus_sha256,
                        "RAW": corpus_sha256,
                        "official_HippoRAG": corpus_sha256,
                    },
                    "filler_unique_snippet_count": filler_count,
                    "ordinal_text_row_count": len(passages),
                    "rule": CORPUS_RULE,
                    "selected_unique_qrel_count": selected_qrel_count,
                },
                "disjointness_aggregate": {
                    "cross_block_component_overlap_count": 0,
                    "cross_block_item_overlap_count": 0,
                    "cross_block_normalized_query_overlap_count": 0,
                    "maximum_selected_items_per_component": 1,
                    "selected_component_count": len(
                        selected_components
                    ),
                    "selected_item_count": len(selected_items),
                    "selected_normalized_query_count": len(
                        selected_queries
                    ),
                },
                "p0_binding": {
                    "implementation": p0_binding,
                    "private_manifest_file_sha256": (
                        private_identity.file_sha256
                    ),
                    "private_manifest_self_sha256": (
                        contract.private_manifest_self_sha256
                    ),
                    "public_audit_receipt_file_sha256": (
                        P0_PUBLIC_AUDIT_RECEIPT_FILE_SHA256
                    ),
                    "public_audit_receipt_self_sha256": (
                        P0_PUBLIC_AUDIT_RECEIPT_SELF_SHA256
                    ),
                    "safe_receipt_file_sha256": p0_identity.file_sha256,
                    "safe_receipt_self_sha256": (
                        contract.p0_receipt_self_sha256
                    ),
                },
                "quota": {
                    block: {
                        family: (
                            contract.block_family_quotas[block][family]
                        )
                        for family in FAMILIES
                    }
                    for block in BLOCKS
                },
                "schema": SELECTION_RECEIPT_SCHEMA,
                "seal_contract": {
                    "M_search_open_authorization": (
                        "controller_promotion_authorization_required"
                    ),
                    "M_search_presealed": True,
                    "M_search_public_block_mode": "0400",
                    "M_search_qrel_pack_mode": "0400",
                    "other_late_qrel_pack_mode": "0400",
                    "qrel_release_only_after_scored_block_actions_sealed": (
                        True
                    ),
                },
                "selection": {
                    "block_order": list(BLOCKS),
                    "family_order": list(FAMILIES),
                    "rule": SELECTION_RULE,
                    "selection_secret_commitment_sha256": (
                        secret_commitment
                    ),
                    "selection_secret_file_create_count": (
                        audit.selection_secret_file_create_count
                    ),
                    "selection_secret_generation_count": (
                        audit.selection_secret_generation_count
                    ),
                    "selection_secret_persisted_publicly": False,
                    "work_id_rule": WORK_ID_RULE,
                },
                "source_access": {
                    **audit.source_payload(),
                    "file_sha256": contract.source_contract.sha256,
                    "size_bytes": contract.source_contract.size_bytes,
                },
                "status": "selected_and_sealed",
                "study_id": STUDY_ID,
                "typed_core_binding": typed_binding,
                "version": VERSION,
            }
        )
        if (
            selected_count != len(selected_components)
            or selected_count != len(selected_items)
            or selected_count != len(selected_queries)
        ):
            raise BioasqP1FormalSourceError(
                "selection_disjointness_aggregate",
                "selected disjointness aggregate drifted",
            )
        audit.stage = "safe_selection_receipt_write"
        _write_exclusive_json(
            normalized_outputs.safe_selection_receipt,
            safe_receipt,
            mode=0o600,
        )
        audit.stage = "success"
        del secret
        return safe_receipt
    except BioasqP1FormalSourceError as error:
        error.stage = audit.stage
        error.source_access = audit.source_payload()
        failure = _safe_failure_receipt(
            audit=audit,
            contract=(
                contract
                if isinstance(contract, FormalSourceContract)
                else DEFAULT_CONTRACT
            ),
            error=error,
        )
        if isinstance(normalized_outputs, FormalOutputPaths):
            _try_write_failure_receipt(normalized_outputs, failure)
        error.safe_failure_receipt = failure
        raise
    except Exception as exc:
        error = BioasqP1FormalSourceError(
            "internal_failure",
            "formal source compiler failed closed",
        )
        error.stage = audit.stage
        error.source_access = audit.source_payload()
        failure = _safe_failure_receipt(
            audit=audit,
            contract=(
                contract
                if isinstance(contract, FormalSourceContract)
                else DEFAULT_CONTRACT
            ),
            error=error,
        )
        if isinstance(normalized_outputs, FormalOutputPaths):
            _try_write_failure_receipt(normalized_outputs, failure)
        error.safe_failure_receipt = failure
        raise error from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-receipt", required=True, type=Path)
    parser.add_argument(
        "--private-eligibility-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--private-selection-secret",
        required=True,
        type=Path,
    )
    parser.add_argument("--public-corpus", required=True, type=Path)
    parser.add_argument("--public-a-form", required=True, type=Path)
    parser.add_argument("--public-f-search", required=True, type=Path)
    parser.add_argument("--public-a-hold", required=True, type=Path)
    parser.add_argument("--public-m-search", required=True, type=Path)
    parser.add_argument(
        "--private-a-form-qrels",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--private-a-hold-qrels",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--private-m-search-qrels",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--safe-selection-receipt",
        required=True,
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    outputs = FormalOutputPaths(
        private_selection_secret=arguments.private_selection_secret,
        public_corpus=arguments.public_corpus,
        public_a_form=arguments.public_a_form,
        public_f_search=arguments.public_f_search,
        public_a_hold=arguments.public_a_hold,
        public_m_search=arguments.public_m_search,
        private_a_form_qrels=arguments.private_a_form_qrels,
        private_a_hold_qrels=arguments.private_a_hold_qrels,
        private_m_search_qrels=arguments.private_m_search_qrels,
        safe_selection_receipt=arguments.safe_selection_receipt,
    )
    try:
        receipt = compile_formal_source(
            p0_receipt_path=arguments.p0_receipt,
            private_eligibility_manifest_path=(
                arguments.private_eligibility_manifest
            ),
            source_path=arguments.source,
            outputs=outputs,
        )
        status = 0
    except BioasqP1FormalSourceError as exc:
        receipt = (
            dict(exc.safe_failure_receipt)
            if exc.safe_failure_receipt is not None
            else {
                "schema": FAILURE_RECEIPT_SCHEMA,
                "self_sha256": "0" * 64,
                "status": "terminal_formal_source_failure_no_retry",
                "study_id": STUDY_ID,
            }
        )
        status = 1
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
                "study_id": receipt["study_id"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKS",
    "CORPUS_SIZE",
    "DEFAULT_BLOCK_FAMILY_QUOTAS",
    "DEFAULT_CONTRACT",
    "FAILURE_RECEIPT_SCHEMA",
    "FAMILIES",
    "FormalOutputPaths",
    "FormalSourceContract",
    "HMAC_SECRET_BYTES",
    "P0_PRIVATE_MANIFEST_FILE_SHA256",
    "P0_PRIVATE_MANIFEST_SELF_SHA256",
    "P0_RECEIPT_FILE_SHA256",
    "P0_RECEIPT_SELF_SHA256",
    "P0_SAFE_RECEIPT_FILE_SHA256",
    "P0_SAFE_RECEIPT_SELF_SHA256",
    "PRIVATE_QREL_ROW_KEYS",
    "PUBLIC_ITEM_KEYS",
    "PUBLIC_PASSAGE_KEYS",
    "QREL_BLOCKS",
    "SELECTION_RECEIPT_SCHEMA",
    "SELECTION_SECRET_BYTES",
    "STUDY_ID",
    "VERSION",
    "BioasqP1FormalSourceError",
    "canonical_bytes",
    "compile_formal_source",
    "self_hashed",
    "stable_hash",
]

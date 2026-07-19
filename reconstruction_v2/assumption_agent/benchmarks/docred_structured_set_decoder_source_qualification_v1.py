"""Aggregate-only qualification for the frozen DocRED G8 source.

The formal entry point opens exactly four already-public manifests and three
byte-bound official files: ``train_annotated.json``, ``dev.json``, and
``rel_info.json``.  It has no path to official TEST or distant TRAIN, creates
no selection secret, and selects no cohort.

Private document values are retained only long enough to validate the official
minimum schema and calculate collision-safe capacity.  The public receipt
contains fixed-schema counts and whole-file/public-schema hashes only.  It
never contains a title, entity, relation triple, alias, sentence text,
evidence ordinal, item identity, or per-document hash.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any
import unicodedata


VERSION = "v1"
SCHEMA = "docred_structured_set_decoder_source_qualification_v1"

FORMAL_CUSTODY_RELATIVE_PATH = Path(
    "manifests/docred_structured_set_decoder_source_custody_v1.json"
)
FORMAL_SOURCE_ACCESS_RELATIVE_PATH = Path(
    "manifests/docred_structured_set_decoder_source_access_v1.json"
)
FORMAL_K3_AMENDMENT_RELATIVE_PATH = Path(
    "manifests/docred_structured_set_decoder_k3_pre_row_amendment_v1.json"
)
FORMAL_FAMILY_FREEZE_RELATIVE_PATH = Path(
    "manifests/docred_structured_set_decoder_relation_family_freeze_v1.json"
)

FORMAL_TRAIN_RELATIVE_PATH = Path(
    "artifacts/docred_official_source_v1/train_annotated.json"
)
FORMAL_DEV_RELATIVE_PATH = Path("artifacts/docred_official_source_v1/dev.json")
FORMAL_RELATION_METADATA_RELATIVE_PATH = Path(
    "artifacts/docred_official_source_v1/rel_info.json"
)

FORMAL_CUSTODY_FILE_SHA256 = (
    "8f9eae7c5c19de40d8cbd19ac2e0354fc58abbb03402c3cbf51a375cfd90fdfa"
)
FORMAL_CUSTODY_SELF_SHA256 = (
    "6cdad6ead4b278fe204ebdcb95b771ec4eee5ef5310101e39310bc5fd9fb50b2"
)
FORMAL_SOURCE_ACCESS_FILE_SHA256 = (
    "cf30877de97d63dc2db04aa3e9884e7032dd0ec3fb6a5d3933d48d4124b7b5ff"
)
FORMAL_SOURCE_ACCESS_SELF_SHA256 = (
    "47ad9f3e7150f81cb1ad02896745528a6a45c1d2b0fb86c94d2f3a5fb9b9d65f"
)
FORMAL_K3_AMENDMENT_FILE_SHA256 = (
    "2a4e808cae3f0b24f437d3eb3cccaea434c9bb80565821dc3a9e7fe9ca8fd31d"
)
FORMAL_K3_AMENDMENT_SELF_SHA256 = (
    "4db719cb0f4fa4d4175e1afb8c441afcef7b5d0c214a6db0d0b9eab463cccd3c"
)
FORMAL_FAMILY_FREEZE_FILE_SHA256 = (
    "c642bd8fff84fbe31598dc25e3c47a3ed1ea6122eeee8e6fe9cba71e8e8112be"
)
FORMAL_FAMILY_FREEZE_SELF_SHA256 = (
    "ab33255be53005cb03ee54b582b65ee3d9195e165de8a6bf5c5774482f7d1387"
)
FORMAL_CUSTODY_COMMIT = "1fafd20977b6fee3006fdb40060a6a86a84e536d"
FORMAL_SOURCE_FREEZE_COMMIT = "3eb0a31bf8985669cecc4516bccff508b81ae82b"
FORMAL_OFFICIAL_GIT_COMMIT = "64622e608d06e83deda6b6e0e871effb9d1fd74c"

FORMAL_TRAIN_SHA256 = (
    "7e706348a02cf91f38bd8c379f934ab61aedadc901fca10d962c1d82ab78e95b"
)
FORMAL_TRAIN_SIZE = 13_029_595
FORMAL_DEV_SHA256 = (
    "4554f7487a6fda3bab4d4e59432e065b7485dfb885bd7f05fd60fc7e93ee7e3e"
)
FORMAL_DEV_SIZE = 4_287_303
FORMAL_RELATION_METADATA_SHA256 = (
    "5ecf4e5e55c179fc83a3a3d19baa01efffecb26ba5edc0b4ac5a54ddf61fe3de"
)
FORMAL_RELATION_METADATA_SIZE = 2_452

FAMILY_PROPERTIES: dict[str, tuple[str, ...]] = {
    "GEO_SOVEREIGNTY": ("P17", "P19", "P27", "P36", "P131", "P150"),
    "MEMBERSHIP_STRUCTURE": (
        "P39",
        "P69",
        "P264",
        "P361",
        "P463",
        "P527",
        "P749",
    ),
    "PERSON_CREATIVE_LIFE": (
        "P26",
        "P57",
        "P161",
        "P170",
        "P175",
        "P569",
        "P800",
    ),
}
FAMILIES = tuple(FAMILY_PROPERTIES)
PROPERTY_TO_FAMILY = {
    property_id: family
    for family, property_ids in FAMILY_PROPERTIES.items()
    for property_id in property_ids
}
FAMILY_PROPERTY_UNION = frozenset(PROPERTY_TO_FAMILY)

SPLIT_FAMILY_DEMANDS: dict[str, dict[str, int]] = {
    "train": {family: 60 for family in FAMILIES},
    "dev": {family: 20 for family in FAMILIES},
}
MIN_NONEMPTY_SENTENCE_COUNT = 10
MIN_GOLD_SENTENCE_COUNT = 1
MAX_GOLD_SENTENCE_COUNT = 3

DOCUMENT_REQUIRED_FIELDS = frozenset({"title", "sents", "vertexSet", "labels"})
MENTION_REQUIRED_FIELDS = frozenset({"name", "sent_id", "pos", "type"})
LABEL_REQUIRED_FIELDS = frozenset({"h", "t", "r", "evidence"})

SCHEMA_ANOMALY_KINDS = (
    "document_not_object",
    "document_required_fields",
    "title",
    "sentences",
    "sentence",
    "sentence_token",
    "vertex_set",
    "entity_cluster",
    "mention_not_object",
    "mention_required_fields",
    "mention_name",
    "mention_sentence",
    "mention_position",
    "mention_type",
    "labels",
    "label_not_object",
    "label_required_fields",
    "label_endpoint",
    "label_relation",
    "label_evidence",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROPERTY_ID_RE = re.compile(r"P[1-9][0-9]*\Z")

SOURCE_BINDING_KEYS = frozenset(
    {
        "custody_manifest_file_sha256",
        "custody_manifest_self_sha256",
        "source_access_manifest_file_sha256",
        "source_access_manifest_self_sha256",
        "k3_amendment_file_sha256",
        "k3_amendment_self_sha256",
        "family_freeze_file_sha256",
        "family_freeze_self_sha256",
        "official_git_commit",
        "source_freeze_commit",
        "train_file_sha256",
        "train_file_size",
        "dev_file_sha256",
        "dev_file_size",
        "relation_metadata_file_sha256",
        "relation_metadata_file_size",
    }
)
SOURCE_BINDING_HASH_KEYS = frozenset(
    key for key in SOURCE_BINDING_KEYS if key.endswith("_sha256")
)
SOURCE_BINDING_SIZE_KEYS = frozenset(
    {
        "train_file_size",
        "dev_file_size",
        "relation_metadata_file_size",
    }
)


class DocredSourceQualificationError(RuntimeError):
    """A frozen manifest, source binding, or strict JSON contract drifted."""


class _DocumentSchemaError(RuntimeError):
    """One private document failed a fixed minimum-schema branch."""

    def __init__(self, kind: str):
        if kind not in SCHEMA_ANOMALY_KINDS:
            raise AssertionError(kind)
        self.kind = kind
        super().__init__(kind)


@dataclass(frozen=True)
class _SourceSpec:
    relative_path: Path
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True)
class _DocumentRecord:
    split: str
    normalized_title_sha256: str
    serialized_document_sha256: str
    eligible_families: frozenset[str]
    eligible_query_hashes: tuple[str, ...]
    sentence_count: int
    nonempty_sentence_count: int
    entity_count: int
    mention_count: int
    label_count: int
    evidence_raw_cardinalities: tuple[int, ...]
    evidence_union_cardinalities: tuple[int, ...]
    duplicate_evidence_ordinal_count: int
    duplicate_exact_triple_count: int
    eligible_label_counts: Mapping[str, int]
    mention_keyset_hash_counts: Mapping[str, int]
    label_keyset_hash_counts: Mapping[str, int]


@dataclass(frozen=True)
class _SplitAudit:
    split: str
    root_schema_valid: bool
    document_count: int
    records: tuple[_DocumentRecord, ...]
    anomaly_counts: Mapping[str, int]
    document_keyset_hash_counts: Mapping[str, int]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DocredSourceQualificationError("value is not canonical JSON") from exc


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_constant(_value: str) -> None:
        raise DocredSourceQualificationError(
            f"{label} contains a non-finite JSON constant"
        )

    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DocredSourceQualificationError(
                    f"{label} contains a duplicate JSON object key"
                )
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except DocredSourceQualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DocredSourceQualificationError(f"{label} is not strict JSON") from exc


def _regular_file_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DocredSourceQualificationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DocredSourceQualificationError(
            f"{label} must be a non-symlink regular file"
        )
    return metadata


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    before = _regular_file_metadata(path, label=label)
    try:
        with path.open("rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise DocredSourceQualificationError(f"{label} read failed") from exc
    after = _regular_file_metadata(path, label=label)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(raw) != before.st_size:
        raise DocredSourceQualificationError(f"{label} changed while being read")
    return raw


def _load_manifest(
    path: Path,
    *,
    schema: str,
    self_hash_field: str,
    formal_file_sha256: str,
    formal_self_sha256: str,
    enforce_formal_identity: bool,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    raw = _read_stable_bytes(path, label="public manifest")
    payload = _strict_json(raw, label="public manifest")
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise DocredSourceQualificationError("public manifest schema drifted")
    declared = payload.get(self_hash_field)
    if not isinstance(declared, str) or _SHA256_RE.fullmatch(declared) is None:
        raise DocredSourceQualificationError("public manifest self hash is invalid")
    body = dict(payload)
    body.pop(self_hash_field)
    observed_self = _stable_hash(body)
    observed_file = hashlib.sha256(raw).hexdigest()
    if observed_self != declared:
        raise DocredSourceQualificationError("public manifest self hash drifted")
    if enforce_formal_identity and (
        observed_file != formal_file_sha256
        or observed_self != formal_self_sha256
    ):
        raise DocredSourceQualificationError("formal public manifest identity drifted")
    return payload, {
        "file_sha256": observed_file,
        "self_sha256": observed_self,
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DocredSourceQualificationError(f"{label} drifted")
    return value


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise DocredSourceQualificationError("frozen contract is incomplete")
        current = current[key]
    return current


def _expect(value: Mapping[str, Any], keys: tuple[str, ...], expected: Any) -> None:
    if _nested(value, *keys) != expected:
        raise DocredSourceQualificationError("frozen contract drifted")


def _source_spec(
    access: Mapping[str, Any],
    *,
    key: str,
    expected_path: Path,
) -> _SourceSpec:
    record = _mapping(
        _nested(access, "allowed_local_files", key),
        label="source file binding",
    )
    path = record.get("relative_path")
    sha256 = record.get("sha256")
    size = record.get("size")
    mode = record.get("mode")
    if (
        path != expected_path.as_posix()
        or not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
        or type(size) is not int
        or size <= 0
        or mode != 0o600
    ):
        raise DocredSourceQualificationError("source file binding drifted")
    return _SourceSpec(expected_path, sha256, size, mode)


def _validate_frozen_contracts(
    project_root: Path,
    *,
    enforce_formal_identity: bool,
) -> tuple[dict[str, _SourceSpec], dict[str, Any]]:
    custody, custody_binding = _load_manifest(
        project_root / FORMAL_CUSTODY_RELATIVE_PATH,
        schema="docred_structured_set_decoder_source_custody_v1",
        self_hash_field="source_custody_sha256",
        formal_file_sha256=FORMAL_CUSTODY_FILE_SHA256,
        formal_self_sha256=FORMAL_CUSTODY_SELF_SHA256,
        enforce_formal_identity=enforce_formal_identity,
    )
    access, access_binding = _load_manifest(
        project_root / FORMAL_SOURCE_ACCESS_RELATIVE_PATH,
        schema="docred_structured_set_decoder_source_access_v1",
        self_hash_field="source_access_sha256",
        formal_file_sha256=FORMAL_SOURCE_ACCESS_FILE_SHA256,
        formal_self_sha256=FORMAL_SOURCE_ACCESS_SELF_SHA256,
        enforce_formal_identity=enforce_formal_identity,
    )
    k3, k3_binding = _load_manifest(
        project_root / FORMAL_K3_AMENDMENT_RELATIVE_PATH,
        schema="docred_structured_set_decoder_k3_pre_row_amendment_v1",
        self_hash_field="k3_amendment_sha256",
        formal_file_sha256=FORMAL_K3_AMENDMENT_FILE_SHA256,
        formal_self_sha256=FORMAL_K3_AMENDMENT_SELF_SHA256,
        enforce_formal_identity=enforce_formal_identity,
    )
    family, family_binding = _load_manifest(
        project_root / FORMAL_FAMILY_FREEZE_RELATIVE_PATH,
        schema="docred_structured_set_decoder_relation_family_freeze_v1",
        self_hash_field="relation_family_freeze_sha256",
        formal_file_sha256=FORMAL_FAMILY_FREEZE_FILE_SHA256,
        formal_self_sha256=FORMAL_FAMILY_FREEZE_SELF_SHA256,
        enforce_formal_identity=enforce_formal_identity,
    )

    if (
        _nested(custody, "official_source", "official_git_commit")
        != FORMAL_OFFICIAL_GIT_COMMIT
        or _nested(access, "prerequisite", "official_git_commit")
        != FORMAL_OFFICIAL_GIT_COMMIT
        or _nested(family, "prerequisite", "official_git_commit")
        != FORMAL_OFFICIAL_GIT_COMMIT
    ):
        raise DocredSourceQualificationError("official Git binding drifted")
    _expect(
        access,
        ("prerequisite", "custody_self_sha256"),
        custody_binding["self_sha256"],
    )
    _expect(
        access,
        ("prerequisite", "custody_commit"),
        FORMAL_CUSTODY_COMMIT,
    )
    _expect(
        k3,
        ("prerequisite", "custody_self_sha256"),
        custody_binding["self_sha256"],
    )
    _expect(
        k3,
        ("prerequisite", "source_access_self_sha256"),
        access_binding["self_sha256"],
    )
    _expect(
        family,
        ("prerequisite", "custody_self_sha256"),
        custody_binding["self_sha256"],
    )
    _expect(
        family,
        ("prerequisite", "source_access_self_sha256"),
        access_binding["self_sha256"],
    )
    _expect(
        family,
        ("prerequisite", "k_amendment_self_sha256"),
        k3_binding["self_sha256"],
    )

    specs = {
        "train": _source_spec(
            access,
            key="train_annotated",
            expected_path=FORMAL_TRAIN_RELATIVE_PATH,
        ),
        "dev": _source_spec(
            access,
            key="dev",
            expected_path=FORMAL_DEV_RELATIVE_PATH,
        ),
        "relation_metadata": _source_spec(
            access,
            key="relation_metadata",
            expected_path=FORMAL_RELATION_METADATA_RELATIVE_PATH,
        ),
    }
    for manifest, source_key, prerequisite_key in (
        (k3, "train", "train_sha256"),
        (k3, "dev", "dev_sha256"),
        (k3, "relation_metadata", "relation_metadata_sha256"),
        (family, "train", "train_sha256"),
        (family, "dev", "dev_sha256"),
        (family, "relation_metadata", "relation_metadata_sha256"),
    ):
        _expect(
            manifest,
            ("prerequisite", prerequisite_key),
            specs[source_key].sha256,
        )

    if enforce_formal_identity and (
        specs["train"].sha256 != FORMAL_TRAIN_SHA256
        or specs["train"].size != FORMAL_TRAIN_SIZE
        or specs["dev"].sha256 != FORMAL_DEV_SHA256
        or specs["dev"].size != FORMAL_DEV_SIZE
        or specs["relation_metadata"].sha256
        != FORMAL_RELATION_METADATA_SHA256
        or specs["relation_metadata"].size != FORMAL_RELATION_METADATA_SIZE
    ):
        raise DocredSourceQualificationError("formal source file identity drifted")

    access_claim = _mapping(access.get("claim_boundary"), label="access boundary")
    incident = _mapping(
        access.get("acquisition_incident"), label="download incident"
    )
    exclusions = _mapping(
        access.get("terminal_exclusions"), label="terminal exclusions"
    )
    if (
        access_claim.get("train_row_or_label_parsed") is not False
        or access_claim.get("dev_row_or_label_parsed") is not False
        or access_claim.get("selection_secret_or_cohort_exists") is not False
        or incident.get("test_content_downloaded_or_opened") is not False
        or incident.get("train_distant_content_downloaded_or_opened") is not False
        or exclusions.get("official_test_present_downloaded_opened_or_used")
        is not False
        or exclusions.get("train_distant_present_downloaded_opened_or_used")
        is not False
    ):
        raise DocredSourceQualificationError("source access boundary drifted")

    if (
        _nested(k3, "amendment_scope", "claim_change")
        != "this_study_estimates_fixed_top3_evidence_retrieval_and_cannot_be_used_as_top5_evidence"
        or _nested(k3, "amendment_scope", "gold_eligibility")
        != "one_through_three_official_evidence_sentences"
        or _nested(k3, "row_access_boundary", "TRAIN_rows_or_labels_opened")
        is not False
        or _nested(k3, "row_access_boundary", "DEV_rows_or_labels_opened")
        is not False
        or _nested(
            k3, "row_access_boundary", "test_or_train_distant_downloaded_or_opened"
        )
        is not False
    ):
        raise DocredSourceQualificationError("K3 amendment boundary drifted")

    frozen_family = _mapping(family.get("family_contract"), label="family freeze")
    observed_properties: dict[str, tuple[str, ...]] = {}
    for family_name in FAMILIES:
        values = frozen_family.get(family_name)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise DocredSourceQualificationError("family property list drifted")
        observed_properties[family_name] = tuple(values)
    if observed_properties != FAMILY_PROPERTIES:
        raise DocredSourceQualificationError("family property list drifted")
    flattened = [
        property_id
        for property_ids in observed_properties.values()
        for property_id in property_ids
    ]
    if len(flattened) != 20 or len(set(flattened)) != 20:
        raise DocredSourceQualificationError("family property lists overlap")

    _expect(
        family,
        ("derived_query_contract", "retrieval_output"),
        "exactly_three_unique_in_document_sentence_ordinals",
    )
    _expect(family, ("eligibility_contract", "gold_size"), "one_through_three")
    _expect(
        family,
        ("eligibility_contract", "document_sentence_count"),
        "at_least_10_nonempty_official_sentences",
    )
    _expect(
        family,
        ("eligibility_contract", "relation_ID_outside_the_frozen_20_ID_union"),
        "ineligible_and_never_selected",
    )
    _expect(
        family,
        ("formation_and_block_contract", "all_blocks_document_disjoint"),
        True,
    )
    for block_name, expected in (
        ("G_form_TRAIN", (32, 96)),
        ("A_form_TRAIN", (16, 48)),
        ("F_search_TRAIN", (12, 36)),
        ("A_hold_DEV", (10, 30)),
        ("M_search_DEV", (10, 30)),
    ):
        block = _mapping(
            _nested(family, "formation_and_block_contract", block_name),
            label="block contract",
        )
        if (block.get("per_family"), block.get("total")) != expected:
            raise DocredSourceQualificationError("block demand drifted")

    ownership = _mapping(
        family.get("fit_and_label_ownership"), label="label ownership"
    )
    if (
        ownership.get("A_form_labels_available_to_generator") is not False
        or ownership.get("G_form_labels_available_to_E1_fit") is not False
        or ownership.get("cross_block_label_pooling") is not False
        or not isinstance(ownership.get("G_form"), str)
        or not isinstance(ownership.get("A_form"), str)
        or not isinstance(ownership.get("F_search"), str)
    ):
        raise DocredSourceQualificationError("label ownership drifted")
    row_boundary = _mapping(
        family.get("row_access_boundary"), label="family row boundary"
    )
    if any(value is not False for value in row_boundary.values()):
        raise DocredSourceQualificationError("family row boundary drifted")
    if _nested(family, "private_assignment_contract", "secret") != (
        "one_new_32_byte_os_random_secret_created_once_only_after_qualification_pass_and_never_rotated"
    ):
        raise DocredSourceQualificationError("post-qualification secret boundary drifted")

    return specs, {
        "custody_manifest_file_sha256": custody_binding["file_sha256"],
        "custody_manifest_self_sha256": custody_binding["self_sha256"],
        "source_access_manifest_file_sha256": access_binding["file_sha256"],
        "source_access_manifest_self_sha256": access_binding["self_sha256"],
        "k3_amendment_file_sha256": k3_binding["file_sha256"],
        "k3_amendment_self_sha256": k3_binding["self_sha256"],
        "family_freeze_file_sha256": family_binding["file_sha256"],
        "family_freeze_self_sha256": family_binding["self_sha256"],
    }


def _read_bound_source(path: Path, spec: _SourceSpec, *, label: str) -> bytes:
    before = _regular_file_metadata(path, label=label)
    if stat.S_IMODE(before.st_mode) != spec.mode:
        raise DocredSourceQualificationError(f"{label} mode drifted")
    raw = _read_stable_bytes(path, label=label)
    metadata = _regular_file_metadata(path, label=label)
    if stat.S_IMODE(metadata.st_mode) != spec.mode:
        raise DocredSourceQualificationError(f"{label} mode drifted")
    if len(raw) != spec.size or hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise DocredSourceQualificationError(f"{label} byte binding drifted")
    return raw


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _text(value: Any, *, kind: str) -> str:
    if not isinstance(value, str) or "\x00" in value or not value.strip():
        raise _DocumentSchemaError(kind)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _DocumentSchemaError(kind) from exc
    return value


def _integer(value: Any) -> bool:
    return type(value) is int


def _keyset_hash(value: Mapping[str, Any]) -> str:
    return _stable_hash(tuple(sorted(value)))


def _parse_document(
    value: Any,
    *,
    split: str,
    relation_metadata: Mapping[str, str],
) -> tuple[_DocumentRecord, str]:
    if not isinstance(value, Mapping):
        raise _DocumentSchemaError("document_not_object")
    document_keyset_hash = _keyset_hash(value)
    if not DOCUMENT_REQUIRED_FIELDS.issubset(value):
        raise _DocumentSchemaError("document_required_fields")
    title = _text(value.get("title"), kind="title")

    raw_sentences = value.get("sents")
    if not isinstance(raw_sentences, list) or not raw_sentences:
        raise _DocumentSchemaError("sentences")
    sentences: list[tuple[str, ...]] = []
    for raw_sentence in raw_sentences:
        if not isinstance(raw_sentence, list) or not raw_sentence:
            raise _DocumentSchemaError("sentence")
        tokens: list[str] = []
        for raw_token in raw_sentence:
            tokens.append(_text(raw_token, kind="sentence_token"))
        sentences.append(tuple(tokens))

    raw_vertex_set = value.get("vertexSet")
    if not isinstance(raw_vertex_set, list):
        raise _DocumentSchemaError("vertex_set")
    aliases: list[tuple[str, ...]] = []
    mention_keysets: Counter[str] = Counter()
    mention_count = 0
    for raw_cluster in raw_vertex_set:
        if not isinstance(raw_cluster, list) or not raw_cluster:
            raise _DocumentSchemaError("entity_cluster")
        cluster_aliases: set[str] = set()
        for raw_mention in raw_cluster:
            if not isinstance(raw_mention, Mapping):
                raise _DocumentSchemaError("mention_not_object")
            mention_keysets[_keyset_hash(raw_mention)] += 1
            if not MENTION_REQUIRED_FIELDS.issubset(raw_mention):
                raise _DocumentSchemaError("mention_required_fields")
            name = _text(raw_mention.get("name"), kind="mention_name")
            sent_id = raw_mention.get("sent_id")
            if not _integer(sent_id) or not (0 <= sent_id < len(sentences)):
                raise _DocumentSchemaError("mention_sentence")
            raw_position = raw_mention.get("pos")
            if (
                not isinstance(raw_position, list)
                or len(raw_position) != 2
                or not all(_integer(position) for position in raw_position)
            ):
                raise _DocumentSchemaError("mention_position")
            start, end = raw_position
            if not (0 <= start < end <= len(sentences[sent_id])):
                raise _DocumentSchemaError("mention_position")
            _text(raw_mention.get("type"), kind="mention_type")
            cluster_aliases.add(_normalize_text(name))
            mention_count += 1
        aliases.append(tuple(sorted(cluster_aliases)))

    raw_labels = value.get("labels")
    if not isinstance(raw_labels, list):
        raise _DocumentSchemaError("labels")
    label_keysets: Counter[str] = Counter()
    evidence_raw_cardinalities: list[int] = []
    evidence_union_cardinalities: list[int] = []
    duplicate_evidence_count = 0
    duplicate_triple_count = 0
    seen_triples: set[tuple[int, str, int]] = set()
    eligible_families: set[str] = set()
    eligible_queries: list[str] = []
    eligible_label_counts = Counter({family: 0 for family in FAMILIES})
    nonempty_sentence_count = sum(bool(sentence) for sentence in sentences)

    for raw_label in raw_labels:
        if not isinstance(raw_label, Mapping):
            raise _DocumentSchemaError("label_not_object")
        label_keysets[_keyset_hash(raw_label)] += 1
        if not LABEL_REQUIRED_FIELDS.issubset(raw_label):
            raise _DocumentSchemaError("label_required_fields")
        head = raw_label.get("h")
        tail = raw_label.get("t")
        if (
            not _integer(head)
            or not _integer(tail)
            or head == tail
            or not (0 <= head < len(aliases))
            or not (0 <= tail < len(aliases))
        ):
            raise _DocumentSchemaError("label_endpoint")
        relation = raw_label.get("r")
        if (
            not isinstance(relation, str)
            or relation not in relation_metadata
            or _PROPERTY_ID_RE.fullmatch(relation) is None
        ):
            raise _DocumentSchemaError("label_relation")
        raw_evidence = raw_label.get("evidence")
        if not isinstance(raw_evidence, list) or any(
            not _integer(ordinal)
            or ordinal < 0
            or ordinal >= len(sentences)
            for ordinal in raw_evidence
        ):
            raise _DocumentSchemaError("label_evidence")
        evidence_union = frozenset(raw_evidence)
        evidence_raw_cardinalities.append(len(raw_evidence))
        evidence_union_cardinalities.append(len(evidence_union))
        duplicate_evidence_count += len(raw_evidence) - len(evidence_union)
        triple = (head, relation, tail)
        duplicate_triple_count += int(triple in seen_triples)
        seen_triples.add(triple)

        family = PROPERTY_TO_FAMILY.get(relation)
        eligible = (
            family is not None
            and nonempty_sentence_count >= MIN_NONEMPTY_SENTENCE_COUNT
            and MIN_GOLD_SENTENCE_COUNT
            <= len(evidence_union)
            <= MAX_GOLD_SENTENCE_COUNT
        )
        if eligible:
            assert family is not None
            eligible_families.add(family)
            eligible_label_counts[family] += 1
            eligible_queries.append(
                _stable_hash(
                    {
                        "head_aliases": aliases[head],
                        "relation_description": _normalize_text(
                            relation_metadata[relation]
                        ),
                        "tail_aliases": aliases[tail],
                    }
                )
            )

    return (
        _DocumentRecord(
            split=split,
            normalized_title_sha256=hashlib.sha256(
                _normalize_text(title).encode("utf-8")
            ).hexdigest(),
            serialized_document_sha256=_stable_hash(
                {
                    "normalized_title": _normalize_text(title),
                    "rendered_sentences": tuple(
                        " ".join(sentence) for sentence in sentences
                    ),
                }
            ),
            eligible_families=frozenset(eligible_families),
            eligible_query_hashes=tuple(eligible_queries),
            sentence_count=len(sentences),
            nonempty_sentence_count=nonempty_sentence_count,
            entity_count=len(aliases),
            mention_count=mention_count,
            label_count=len(raw_labels),
            evidence_raw_cardinalities=tuple(evidence_raw_cardinalities),
            evidence_union_cardinalities=tuple(evidence_union_cardinalities),
            duplicate_evidence_ordinal_count=duplicate_evidence_count,
            duplicate_exact_triple_count=duplicate_triple_count,
            eligible_label_counts={
                family: eligible_label_counts[family] for family in FAMILIES
            },
            mention_keyset_hash_counts=dict(mention_keysets),
            label_keyset_hash_counts=dict(label_keysets),
        ),
        document_keyset_hash,
    )


def _audit_split(
    payload: Any,
    *,
    split: str,
    relation_metadata: Mapping[str, str],
) -> _SplitAudit:
    anomaly_counts = Counter({kind: 0 for kind in SCHEMA_ANOMALY_KINDS})
    document_keysets: Counter[str] = Counter()
    if not isinstance(payload, list):
        return _SplitAudit(
            split=split,
            root_schema_valid=False,
            document_count=0,
            records=(),
            anomaly_counts=dict(anomaly_counts),
            document_keyset_hash_counts={},
        )
    records: list[_DocumentRecord] = []
    for value in payload:
        try:
            record, document_keyset = _parse_document(
                value,
                split=split,
                relation_metadata=relation_metadata,
            )
            records.append(record)
            document_keysets[document_keyset] += 1
        except _DocumentSchemaError as exc:
            anomaly_counts[exc.kind] += 1
            if isinstance(value, Mapping):
                document_keysets[_keyset_hash(value)] += 1
    return _SplitAudit(
        split=split,
        root_schema_valid=True,
        document_count=len(payload),
        records=tuple(records),
        anomaly_counts=dict(anomaly_counts),
        document_keyset_hash_counts=dict(document_keysets),
    )


def _parse_relation_metadata(payload: Any) -> dict[str, str]:
    if not isinstance(payload, Mapping) or not payload:
        raise DocredSourceQualificationError("relation metadata root drifted")
    output: dict[str, str] = {}
    for property_id, description in payload.items():
        if (
            not isinstance(property_id, str)
            or _PROPERTY_ID_RE.fullmatch(property_id) is None
            or not isinstance(description, str)
            or "\x00" in description
            or not description.strip()
        ):
            raise DocredSourceQualificationError("relation metadata schema drifted")
        output[property_id] = description
    if not FAMILY_PROPERTY_UNION.issubset(output):
        raise DocredSourceQualificationError(
            "relation metadata omits a frozen family property"
        )
    return output


def _counter_map(values: Sequence[int]) -> dict[str, int]:
    counter = Counter(values)
    return {str(key): counter[key] for key in sorted(counter)}


def _hash_counter_map(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _duplicate_aggregates(values: Sequence[str]) -> dict[str, int]:
    counter = Counter(values)
    duplicate_counts = [count for count in counter.values() if count > 1]
    return {
        "unique_value_count": len(counter),
        "duplicate_group_count": len(duplicate_counts),
        "occurrence_count_in_duplicate_groups": sum(duplicate_counts),
        "duplicate_excess_occurrence_count": sum(
            count - 1 for count in duplicate_counts
        ),
    }


def _split_public_aggregate(audit: _SplitAudit) -> dict[str, Any]:
    records = audit.records
    mention_keysets: Counter[str] = Counter()
    label_keysets: Counter[str] = Counter()
    eligible_labels = Counter({family: 0 for family in FAMILIES})
    eligible_docs = Counter({family: 0 for family in FAMILIES})
    raw_evidence: list[int] = []
    union_evidence: list[int] = []
    query_hashes: list[str] = []
    for record in records:
        mention_keysets.update(record.mention_keyset_hash_counts)
        label_keysets.update(record.label_keyset_hash_counts)
        raw_evidence.extend(record.evidence_raw_cardinalities)
        union_evidence.extend(record.evidence_union_cardinalities)
        query_hashes.extend(record.eligible_query_hashes)
        for family in FAMILIES:
            eligible_labels[family] += record.eligible_label_counts[family]
            eligible_docs[family] += int(family in record.eligible_families)
    sentence_counts = [record.sentence_count for record in records]
    nonempty_sentence_counts = [
        record.nonempty_sentence_count for record in records
    ]
    schema_invalid_count = audit.document_count - len(records)
    return {
        "schema": {
            "root_is_list": audit.root_schema_valid,
            "document_count": audit.document_count,
            "valid_document_count": len(records),
            "invalid_document_count": schema_invalid_count,
            "minimum_schema_equivalent": (
                audit.root_schema_valid and schema_invalid_count == 0
            ),
            "document_keyset_hash_counts": _hash_counter_map(
                audit.document_keyset_hash_counts
            ),
            "mention_keyset_hash_counts": _hash_counter_map(mention_keysets),
            "label_keyset_hash_counts": _hash_counter_map(label_keysets),
            "anomaly_counts": {
                kind: int(audit.anomaly_counts.get(kind, 0))
                for kind in SCHEMA_ANOMALY_KINDS
            },
        },
        "document_and_sentence_counts": {
            "document_count": len(records),
            "sentence_count_total": sum(sentence_counts),
            "sentence_count_min": min(sentence_counts) if sentence_counts else 0,
            "sentence_count_max": max(sentence_counts) if sentence_counts else 0,
            "sentence_count_histogram": _counter_map(sentence_counts),
            "nonempty_sentence_count_total": sum(nonempty_sentence_counts),
            "document_with_at_least_10_nonempty_sentences_count": sum(
                value >= MIN_NONEMPTY_SENTENCE_COUNT
                for value in nonempty_sentence_counts
            ),
            "entity_cluster_count_total": sum(
                record.entity_count for record in records
            ),
            "mention_count_total": sum(record.mention_count for record in records),
            "positive_relation_label_count_total": sum(
                record.label_count for record in records
            ),
        },
        "evidence_cardinality_counts": {
            "raw_evidence_list_cardinality_histogram": _counter_map(raw_evidence),
            "deduplicated_evidence_union_cardinality_histogram": _counter_map(
                union_evidence
            ),
            "duplicate_evidence_ordinal_occurrence_count": sum(
                record.duplicate_evidence_ordinal_count for record in records
            ),
        },
        "family_eligibility_counts": {
            family: {
                "eligible_label_count_before_one_document_cap": eligible_labels[
                    family
                ],
                "eligible_unique_source_document_count": eligible_docs[family],
            }
            for family in FAMILIES
        },
        "duplicate_counts": {
            "normalized_title": _duplicate_aggregates(
                [record.normalized_title_sha256 for record in records]
            ),
            "serialized_document": _duplicate_aggregates(
                [record.serialized_document_sha256 for record in records]
            ),
            "eligible_derived_query": _duplicate_aggregates(query_hashes),
            "duplicate_exact_h_r_t_label_occurrence_count": sum(
                record.duplicate_exact_triple_count for record in records
            ),
        },
    }


class _UnionFind:
    def __init__(self, size: int):
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


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int


def _add_flow_edge(graph: list[list[_FlowEdge]], left: int, right: int, cap: int) -> None:
    forward = _FlowEdge(right, len(graph[right]), cap)
    reverse = _FlowEdge(left, len(graph[left]), 0)
    graph[left].append(forward)
    graph[right].append(reverse)


def _maximum_flow(
    component_targets: Sequence[frozenset[tuple[str, str]]],
) -> tuple[int, dict[tuple[str, str], int]]:
    targets = tuple(
        (split, family)
        for split in ("train", "dev")
        for family in FAMILIES
    )
    source = 0
    component_offset = 1
    target_offset = component_offset + len(component_targets)
    sink = target_offset + len(targets)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    for component_index, eligible_targets in enumerate(component_targets):
        node = component_offset + component_index
        _add_flow_edge(graph, source, node, 1)
        for target_index, target in enumerate(targets):
            if target in eligible_targets:
                _add_flow_edge(graph, node, target_offset + target_index, 1)
    target_sink_edges: dict[tuple[str, str], _FlowEdge] = {}
    for target_index, target in enumerate(targets):
        demand = SPLIT_FAMILY_DEMANDS[target[0]][target[1]]
        node = target_offset + target_index
        _add_flow_edge(graph, node, sink, demand)
        target_sink_edges[target] = graph[node][-1]

    total = 0
    while True:
        levels = [-1] * len(graph)
        levels[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for edge in graph[node]:
                if edge.capacity > 0 and levels[edge.to] < 0:
                    levels[edge.to] = levels[node] + 1
                    queue.append(edge.to)
        if levels[sink] < 0:
            break
        next_edge = [0] * len(graph)

        def send(node: int, available: int) -> int:
            if node == sink:
                return available
            while next_edge[node] < len(graph[node]):
                edge = graph[node][next_edge[node]]
                if edge.capacity > 0 and levels[edge.to] == levels[node] + 1:
                    sent = send(edge.to, min(available, edge.capacity))
                    if sent:
                        edge.capacity -= sent
                        graph[edge.to][edge.reverse].capacity += sent
                        return sent
                next_edge[node] += 1
            return 0

        while True:
            sent = send(source, 1 << 30)
            if not sent:
                break
            total += sent
    assigned = {
        target: SPLIT_FAMILY_DEMANDS[target[0]][target[1]] - edge.capacity
        for target, edge in target_sink_edges.items()
    }
    return total, assigned


def _capacity_and_cross_split_receipt(
    train: _SplitAudit,
    dev: _SplitAudit,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [*train.records, *dev.records]
    union_find = _UnionFind(len(records))
    for attribute in (
        "normalized_title_sha256",
        "serialized_document_sha256",
    ):
        first: dict[str, int] = {}
        for index, record in enumerate(records):
            value = getattr(record, attribute)
            if value in first:
                union_find.union(first[value], index)
            else:
                first[value] = index
    targets_by_root: dict[int, set[tuple[str, str]]] = {}
    member_count_by_root: Counter[int] = Counter()
    for index, record in enumerate(records):
        root = union_find.find(index)
        member_count_by_root[root] += 1
        targets = targets_by_root.setdefault(root, set())
        targets.update((record.split, family) for family in record.eligible_families)
    ordered_roots = sorted(targets_by_root)
    component_targets = [
        frozenset(targets_by_root[root]) for root in ordered_roots
    ]
    maximum_flow, assigned = _maximum_flow(component_targets)
    total_required = sum(
        demand
        for split_demands in SPLIT_FAMILY_DEMANDS.values()
        for demand in split_demands.values()
    )
    target_receipt: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        target_receipt[split] = {}
        for family in FAMILIES:
            target = (split, family)
            candidate_components = sum(
                target in targets for targets in component_targets
            )
            required = SPLIT_FAMILY_DEMANDS[split][family]
            target_receipt[split][family] = {
                "candidate_assignable_document_component_count": (
                    candidate_components
                ),
                "required_assigned_document_count": required,
                "deterministic_max_flow_assigned_document_count": assigned[target],
                "assigned_requirement_met": assigned[target] == required,
            }

    train_title = Counter(
        record.normalized_title_sha256 for record in train.records
    )
    dev_title = Counter(record.normalized_title_sha256 for record in dev.records)
    train_serialized = Counter(
        record.serialized_document_sha256 for record in train.records
    )
    dev_serialized = Counter(
        record.serialized_document_sha256 for record in dev.records
    )

    def cross(counter_left: Counter[str], counter_right: Counter[str]) -> dict[str, int]:
        shared = set(counter_left).intersection(counter_right)
        return {
            "collision_group_count": len(shared),
            "train_document_occurrence_count": sum(
                counter_left[value] for value in shared
            ),
            "dev_document_occurrence_count": sum(
                counter_right[value] for value in shared
            ),
        }

    capacity = {
        "per_split_family": target_receipt,
        "collision_component_count": len(component_targets),
        "multi_document_collision_component_count": sum(
            count > 1 for count in member_count_by_root.values()
        ),
        "document_occurrence_count_in_multi_document_components": sum(
            count for count in member_count_by_root.values() if count > 1
        ),
        "required_global_document_count": total_required,
        "deterministic_max_flow_assigned_document_count": maximum_flow,
        "simultaneous_all_block_document_disjoint_feasible": (
            maximum_flow == total_required
        ),
    }
    collisions = {
        "normalization": "Unicode_NFKC_then_whitespace_collapse_then_casefold",
        "cross_split_normalized_title": cross(train_title, dev_title),
        "cross_split_canonical_serialized_document": cross(
            train_serialized, dev_serialized
        ),
        "title_or_serialized_document_collision_components_are_capacity_one": True,
        "private_collision_value_or_hash_emitted_count": 0,
    }
    return capacity, collisions


def _validated_source_binding(
    source_binding: Mapping[str, Any],
    *,
    formal_identity_enforced: bool,
) -> dict[str, Any]:
    if type(formal_identity_enforced) is not bool:
        raise DocredSourceQualificationError(
            "formal identity flag must be an exact boolean"
        )
    if set(source_binding) != SOURCE_BINDING_KEYS:
        raise DocredSourceQualificationError("source binding keyset drifted")
    for key in SOURCE_BINDING_HASH_KEYS:
        value = source_binding[key]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise DocredSourceQualificationError("source binding hash drifted")
    for key in SOURCE_BINDING_SIZE_KEYS:
        value = source_binding[key]
        if type(value) is not int or value <= 0:
            raise DocredSourceQualificationError("source binding size drifted")
    if (
        source_binding["official_git_commit"] != FORMAL_OFFICIAL_GIT_COMMIT
        or source_binding["source_freeze_commit"] != FORMAL_SOURCE_FREEZE_COMMIT
    ):
        raise DocredSourceQualificationError("source commit binding drifted")
    if formal_identity_enforced:
        formal_values = {
            "custody_manifest_file_sha256": FORMAL_CUSTODY_FILE_SHA256,
            "custody_manifest_self_sha256": FORMAL_CUSTODY_SELF_SHA256,
            "source_access_manifest_file_sha256": FORMAL_SOURCE_ACCESS_FILE_SHA256,
            "source_access_manifest_self_sha256": FORMAL_SOURCE_ACCESS_SELF_SHA256,
            "k3_amendment_file_sha256": FORMAL_K3_AMENDMENT_FILE_SHA256,
            "k3_amendment_self_sha256": FORMAL_K3_AMENDMENT_SELF_SHA256,
            "family_freeze_file_sha256": FORMAL_FAMILY_FREEZE_FILE_SHA256,
            "family_freeze_self_sha256": FORMAL_FAMILY_FREEZE_SELF_SHA256,
            "train_file_sha256": FORMAL_TRAIN_SHA256,
            "train_file_size": FORMAL_TRAIN_SIZE,
            "dev_file_sha256": FORMAL_DEV_SHA256,
            "dev_file_size": FORMAL_DEV_SIZE,
            "relation_metadata_file_sha256": FORMAL_RELATION_METADATA_SHA256,
            "relation_metadata_file_size": FORMAL_RELATION_METADATA_SIZE,
        }
        if any(source_binding[key] != value for key, value in formal_values.items()):
            raise DocredSourceQualificationError(
                "formal receipt source binding drifted"
            )
    return {key: source_binding[key] for key in sorted(SOURCE_BINDING_KEYS)}


def _qualify_decoded_sources(
    train_payload: Any,
    dev_payload: Any,
    relation_metadata_payload: Any,
    *,
    source_binding: Mapping[str, Any],
    formal_identity_enforced: bool,
) -> dict[str, Any]:
    """Validate decoded TRAIN/DEV and return an aggregate-only receipt."""

    safe_source_binding = _validated_source_binding(
        source_binding,
        formal_identity_enforced=formal_identity_enforced,
    )
    relation_metadata = _parse_relation_metadata(relation_metadata_payload)
    train = _audit_split(
        train_payload,
        split="train",
        relation_metadata=relation_metadata,
    )
    dev = _audit_split(
        dev_payload,
        split="dev",
        relation_metadata=relation_metadata,
    )
    train_public = _split_public_aggregate(train)
    dev_public = _split_public_aggregate(dev)
    capacity, collisions = _capacity_and_cross_split_receipt(train, dev)
    invalid_document_count = (
        train_public["schema"]["invalid_document_count"]
        + dev_public["schema"]["invalid_document_count"]
    )
    invalid_root_count = int(not train.root_schema_valid) + int(
        not dev.root_schema_valid
    )
    global_shortfall = (
        capacity["required_global_document_count"]
        - capacity["deterministic_max_flow_assigned_document_count"]
    )
    schema_passed = invalid_root_count == 0 and invalid_document_count == 0
    passed = (
        schema_passed
        and capacity["simultaneous_all_block_document_disjoint_feasible"]
    )
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": (
            "passed_source_qualification_no_selection"
            if passed
            else "terminal_source_incompatible_no_selection"
        ),
        "formal_identity_enforced": formal_identity_enforced,
        "source_binding": safe_source_binding,
        "relation_metadata_aggregate": {
            "property_count": len(relation_metadata),
            "property_ID_set_sha256": _stable_hash(
                tuple(sorted(relation_metadata))
            ),
            "property_description_mapping_sha256": _stable_hash(
                relation_metadata
            ),
            "frozen_family_property_count": len(FAMILY_PROPERTY_UNION),
            "missing_frozen_family_property_count": 0,
        },
        "split_aggregates": {
            "train": train_public,
            "dev": dev_public,
        },
        "cross_split_collision_counts": collisions,
        "simultaneous_document_assignment_capacity": capacity,
        "terminal_reason_counts": {
            "invalid_root_schema_count": invalid_root_count,
            "invalid_document_schema_count": invalid_document_count,
            "simultaneous_assignment_shortfall_count": global_shortfall,
        },
        "opened_content_boundary": {
            "authorized_official_file_count": 3,
            "train_annotated_open_count": 1,
            "dev_open_count": 1,
            "relation_metadata_open_count": 1,
            "official_test_open_count": 0,
            "train_distant_open_count": 0,
        },
        "claim_boundary": {
            "qualification_only_no_efficacy_claim": True,
            "fixed_top3_not_top5": True,
            "selection_secret_generated_or_opened": False,
            "cohort_selected_or_materialized": False,
            "retrieval_action_evaluator_or_score_run": False,
            "online_or_external_evaluation_used": False,
            "title_entity_relation_triple_alias_text_or_ordinal_emitted": False,
            "per_document_or_per_item_hash_emitted": False,
        },
    }
    body["qualification_sha256"] = _stable_hash(body)
    return body


def qualify_source_files(
    project_root: str | Path,
    *,
    enforce_formal_identity: bool = True,
) -> dict[str, Any]:
    """Open only the three authorized source files and qualify them."""

    root = Path(project_root).resolve(strict=True)
    specs, manifest_binding = _validate_frozen_contracts(
        root,
        enforce_formal_identity=enforce_formal_identity,
    )
    raw_sources: dict[str, bytes] = {}
    for key in ("relation_metadata", "train", "dev"):
        spec = specs[key]
        raw_sources[key] = _read_bound_source(
            root / spec.relative_path,
            spec,
            label=f"authorized {key} source",
        )
    source_binding = {
        **manifest_binding,
        "official_git_commit": FORMAL_OFFICIAL_GIT_COMMIT,
        "source_freeze_commit": FORMAL_SOURCE_FREEZE_COMMIT,
        "train_file_sha256": specs["train"].sha256,
        "train_file_size": specs["train"].size,
        "dev_file_sha256": specs["dev"].sha256,
        "dev_file_size": specs["dev"].size,
        "relation_metadata_file_sha256": specs["relation_metadata"].sha256,
        "relation_metadata_file_size": specs["relation_metadata"].size,
    }
    return _qualify_decoded_sources(
        _strict_json(raw_sources["train"], label="authorized TRAIN source"),
        _strict_json(raw_sources["dev"], label="authorized DEV source"),
        _strict_json(
            raw_sources["relation_metadata"],
            label="authorized relation metadata",
        ),
        source_binding=source_binding,
        formal_identity_enforced=enforce_formal_identity,
    )


def build_formal_qualification(project_root: str | Path) -> dict[str, Any]:
    """Run the committed formal binding without selecting a cohort."""

    return qualify_source_files(project_root, enforce_formal_identity=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate-only DocRED structured-set source qualifier"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="reconstruction_v2 project root",
    )
    args = parser.parse_args(argv)
    receipt = build_formal_qualification(args.project_root)
    sys.stdout.buffer.write(_canonical_json(receipt) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

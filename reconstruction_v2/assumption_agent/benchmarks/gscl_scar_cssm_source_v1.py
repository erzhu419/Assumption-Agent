"""Frozen SCAR-English source compiler for the CSSM intrinsic study.

The compiler is intentionally model- and scorer-free.  It opens one private
copy of the pinned English SCAR JSONL exactly once, validates the complete
source topology, and separates the closed-task action view from the gold
label view.  The action view contains independently permuted, side-local
opaque slots; neither the original mapping order nor any label stratum is
present there.

SCAR's repository does not declare a dataset license.  This module therefore
returns in-memory packs only.  It does not copy, publish, or persist the raw
source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


VERSION = "gscl_scar_cssm_source_v1"
ACTION_SCHEMA = f"{VERSION}.action_pack.v1"
LABEL_SCHEMA = f"{VERSION}.label_pack.v1"
SAFE_AGGREGATE_SCHEMA = f"{VERSION}.safe_aggregate.v1"

EXPECTED_SOURCE_SIZE_BYTES = 1_393_355
EXPECTED_SOURCE_SHA256 = (
    "12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369"
)
EXPECTED_SOURCE_ROW_COUNT = 400
EXPECTED_SOURCE_MAPPING_COUNT = 1_618
EXPECTED_SOURCE_IDS = frozenset(range(1, 401))
EXPECTED_SOURCE_KEYS = (
    "id",
    "lang",
    "system_a",
    "system_b",
    "mappings",
    "system_a_domain",
    "system_b_domain",
    "system_a_background",
    "system_b_background",
    "Explanation",
)
EXPECTED_DOMAIN_COUNT = 13

PUBLIC_DENY_RAW_LINE_SHA256_BY_ID = {
    12: "ed7c37a94561001aa70b9371bf51da48564b3496efa1d51e437397381dfae647",
    29: "3b957a8e4491a9db2daf655b3d73cdd17a3068053042234527351a7372f5b56e",
    55: "f19488387ba66d98d5906bc3fcc58eb16d537e288ad2d32810b4ebcb6b5ef491",
    85: "27a2b5e104e1bbf8faa604aa1c99ad725e2accc49ccab0cef0946c1390f53a1b",
    107: "ab6794270678880b4f60551866f7b92f4938873bbe2fc78f339a73aadb0c85eb",
    129: "c22b833fc7cc1f6633ee624b0389171936c4e2b8e2df1f94bb49706fb0456153",
    180: "b7ac06fb592a02244ae786104e0021c5fbc1a097ee640617287e534d3200a055",
    347: "52e995762f929a3bfdb8e91a7390bc8e92cb026a830f8a10e1061c83c0eb67da",
    351: "340f7b4ed00e8f5a1baab1df69cf706473805ec984fab28b9ce85eb85ce33874",
}
EXPECTED_PUBLIC_DENY_ROW_COUNT = 9
EXPECTED_PUBLIC_DENY_MAPPING_COUNT = 46
EXPECTED_PUBLIC_DENY_LINE_HASH_LIST_SHA256 = (
    "08ed304f8ba4033d0e84e7b0f13a14557d9757b90e5b1687098940c32369eff9"
)

EXPECTED_NORMALIZED_DUPLICATE_SLOT_ROW_IDS = frozenset(
    {
        24,
        42,
        61,
        69,
        273,
        275,
        280,
        281,
        282,
        283,
        284,
        285,
        287,
        288,
        289,
        291,
        292,
        293,
        294,
        295,
        296,
        304,
        308,
        312,
        315,
        316,
        318,
        319,
        341,
    }
)
EXPECTED_AMBIGUOUS_ROW_COUNT = 29
EXPECTED_AMBIGUOUS_MAPPING_COUNT = 233
EXPECTED_AMBIGUOUS_LINE_HASH_LIST_SHA256 = (
    "3067b68aedb13f41006717149a3f6b77418ea9c79ad4d5ba2ccc493225917fe5"
)

EXPECTED_PRIMARY_ROW_COUNT = 362
EXPECTED_PRIMARY_MAPPING_COUNT = 1_339
EXPECTED_PRIMARY_ARITY_COUNTS = {
    2: 31,
    3: 187,
    4: 77,
    5: 32,
    6: 12,
    7: 12,
    8: 8,
    9: 2,
    10: 1,
}
EXPECTED_PRIMARY_INTRA_DOMAIN_COUNT = 108
EXPECTED_PRIMARY_CROSS_DOMAIN_COUNT = 254
EXPECTED_PRIMARY_LINE_HASH_LIST_SHA256 = (
    "c9ed6bc9967b9fe0bb5373868b2363c3a3b2f293c84e1ad53ec58b0a28822916"
)

EXPECTED_ACTION_ITEM_COUNT = 391
EXPECTED_ACTION_VARIANT_COUNT = 782
EXPECTED_NONPUBLIC_MAPPING_COUNT = 1_572
VARIANT_NAMES = ("base", "system_swap")

HMAC_SECRET_BYTES = 32
_ITEM_TOKEN_DOMAIN = b"SCAR_CSSM_V1_ITEM_TOKEN_HMAC_SHA256\x00"
_ITEM_ORDER_DOMAIN = b"SCAR_CSSM_V1_ITEM_ORDER_HMAC_SHA256\x00"
_SLOT_TOKEN_DOMAIN = b"SCAR_CSSM_V1_SLOT_TOKEN_HMAC_SHA256\x00"
_SLOT_ORDER_DOMAIN = b"SCAR_CSSM_V1_SLOT_ORDER_HMAC_SHA256\x00"
_PACK_BINDING_DOMAIN = b"SCAR_CSSM_V1_PACK_BINDING_HMAC_SHA256\x00"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_STUDY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ITEM_TOKEN = re.compile(r"scar-item-v1-[0-9a-f]{64}\Z")
_SLOT_TOKEN = re.compile(r"scar-slot-v1-[0-9a-f]{64}\Z")

_ACTION_FORBIDDEN_KEYS = frozenset(
    {
        "Explanation",
        "cohort",
        "domain",
        "explanation",
        "gold_pairs",
        "id",
        "index",
        "mapping",
        "mapping_index",
        "mappings",
        "original_index",
        "raw_id",
        "source_id",
        "strata",
        "system_a_domain",
        "system_b_domain",
    }
)


class ScarCssmSourceError(RuntimeError):
    """One stable fail-closed source-compiler error."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


@dataclass(frozen=True)
class ScarCssmSourceCompilation:
    """In-memory result; callers decide how private packs are persisted."""

    action_pack: dict[str, Any]
    label_pack: dict[str, Any]
    safe_aggregate: dict[str, Any]


@dataclass(frozen=True)
class _SourceRow:
    raw_line_sha256: str
    source_id: int
    system_a: str
    system_b: str
    system_a_domain: str
    system_b_domain: str
    system_a_background: str
    system_b_background: str
    mappings: tuple[tuple[str, str], ...]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ScarCssmSourceError("SCAR_CANONICAL_JSON_INVALID") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _framed(parts: Iterable[bytes]) -> bytes:
    framed = bytearray()
    for part in parts:
        framed.extend(len(part).to_bytes(8, "big"))
        framed.extend(part)
    return bytes(framed)


def _hmac_hex(secret: bytes, domain: bytes, *parts: bytes) -> str:
    return hmac.new(secret, domain + _framed(parts), hashlib.sha256).hexdigest()


def _require_secret(secret: bytes) -> None:
    if type(secret) is not bytes or len(secret) != HMAC_SECRET_BYTES:
        raise ScarCssmSourceError("SCAR_HMAC_SECRET_INVALID")


def _require_study_id(study_id: str) -> None:
    if not isinstance(study_id, str) or _STUDY_ID.fullmatch(study_id) is None:
        raise ScarCssmSourceError("SCAR_STUDY_ID_INVALID")


def _read_private_regular_source_once(path: Path) -> bytes:
    if not isinstance(path, Path):
        raise ScarCssmSourceError("SCAR_SOURCE_PATH_INVALID")
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ScarCssmSourceError("SCAR_SOURCE_OPEN_FAILED") from exc
    if (
        not path.is_absolute()
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or observed.st_size != EXPECTED_SOURCE_SIZE_BYTES
    ):
        raise ScarCssmSourceError("SCAR_SOURCE_FILE_CONTRACT_INVALID")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ScarCssmSourceError("SCAR_SOURCE_OPEN_FAILED") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != observed.st_uid
            or before.st_nlink != 1
            or before.st_size != EXPECTED_SOURCE_SIZE_BYTES
            or (before.st_dev, before.st_ino)
            != (observed.st_dev, observed.st_ino)
        ):
            raise ScarCssmSourceError("SCAR_SOURCE_FILE_CONTRACT_INVALID")
        chunks: list[bytes] = []
        remaining = EXPECTED_SOURCE_SIZE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ScarCssmSourceError("SCAR_SOURCE_READ_FAILED") from exc
    finally:
        os.close(descriptor)

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
        before.st_uid,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
        after.st_uid,
    )
    if identity_before != identity_after or len(raw) != EXPECTED_SOURCE_SIZE_BYTES:
        raise ScarCssmSourceError("SCAR_SOURCE_CHANGED_DURING_READ")
    if not hmac.compare_digest(_sha256(raw), EXPECTED_SOURCE_SHA256):
        raise ScarCssmSourceError("SCAR_SOURCE_IDENTITY_INVALID")
    return raw


def _reject_json_constant(_: str) -> None:
    raise ValueError("non_finite_json_constant")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _parse_source_rows(
    raw: bytes,
    *,
    expected_row_count: int,
    expected_mapping_count: int,
    expected_ids: frozenset[int],
) -> tuple[_SourceRow, ...]:
    """Pure strict parser; explicit expectations permit synthetic tests."""

    if not isinstance(raw, bytes):
        raise ScarCssmSourceError("SCAR_SOURCE_BYTES_INVALID")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ScarCssmSourceError("SCAR_SOURCE_UTF8_INVALID") from exc

    nonempty_lines = tuple(line for line in raw.splitlines() if line.strip())
    if len(nonempty_lines) != expected_row_count:
        raise ScarCssmSourceError("SCAR_SOURCE_ROW_COUNT_INVALID")
    rows: list[_SourceRow] = []
    seen_ids: set[int] = set()
    mapping_count = 0
    for raw_line in nonempty_lines:
        try:
            value = json.loads(
                raw_line.decode("utf-8", errors="strict"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_json_object,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ScarCssmSourceError("SCAR_SOURCE_JSON_INVALID") from exc
        if type(value) is not dict or tuple(value) != EXPECTED_SOURCE_KEYS:
            raise ScarCssmSourceError("SCAR_SOURCE_SCHEMA_INVALID")
        source_id = value["id"]
        if (
            type(source_id) is not int
            or source_id not in expected_ids
            or source_id in seen_ids
            or value["lang"] != "en"
            or type(value["lang"]) is not str
        ):
            raise ScarCssmSourceError("SCAR_SOURCE_ID_OR_LANG_INVALID")
        seen_ids.add(source_id)
        for key in (
            "system_a",
            "system_b",
            "system_a_domain",
            "system_b_domain",
            "system_a_background",
            "system_b_background",
        ):
            if type(value[key]) is not str or not value[key].strip():
                raise ScarCssmSourceError("SCAR_SOURCE_TEXT_FIELD_INVALID")
        mappings = value["mappings"]
        # Explanation is deliberately not iterated or cardinality-matched.
        if type(mappings) is not list or type(value["Explanation"]) is not list:
            raise ScarCssmSourceError("SCAR_SOURCE_SEQUENCE_FIELD_INVALID")
        parsed_mappings: list[tuple[str, str]] = []
        for pair in mappings:
            if (
                type(pair) is not list
                or len(pair) != 2
                or any(type(term) is not str or not term.strip() for term in pair)
            ):
                raise ScarCssmSourceError("SCAR_SOURCE_MAPPING_INVALID")
            parsed_mappings.append((pair[0], pair[1]))
        if not 2 <= len(parsed_mappings) <= 14:
            raise ScarCssmSourceError("SCAR_SOURCE_ARITY_INVALID")
        if len(set(parsed_mappings)) != len(parsed_mappings):
            raise ScarCssmSourceError("SCAR_SOURCE_MAPPING_DUPLICATE")
        mapping_count += len(parsed_mappings)
        rows.append(
            _SourceRow(
                raw_line_sha256=_sha256(raw_line),
                source_id=source_id,
                system_a=value["system_a"],
                system_b=value["system_b"],
                system_a_domain=value["system_a_domain"],
                system_b_domain=value["system_b_domain"],
                system_a_background=value["system_a_background"],
                system_b_background=value["system_b_background"],
                mappings=tuple(parsed_mappings),
            )
        )
    if seen_ids != expected_ids:
        raise ScarCssmSourceError("SCAR_SOURCE_ID_SET_INVALID")
    if mapping_count != expected_mapping_count:
        raise ScarCssmSourceError("SCAR_SOURCE_MAPPING_COUNT_INVALID")
    return tuple(rows)


def _normalized_slot_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _has_normalized_duplicate_slot(row: _SourceRow) -> bool:
    for side in (0, 1):
        values = tuple(_normalized_slot_key(pair[side]) for pair in row.mappings)
        if len(values) != len(set(values)):
            return True
    return False


def _line_hash_list_commitment(rows: Sequence[_SourceRow]) -> str:
    payload = ("\n".join(row.raw_line_sha256 for row in rows) + "\n").encode(
        "ascii"
    )
    return _sha256(payload)


def _validate_official_topology(rows: tuple[_SourceRow, ...]) -> None:
    by_id = {row.source_id: row for row in rows}
    deny_ids = frozenset(PUBLIC_DENY_RAW_LINE_SHA256_BY_ID)
    if (
        len(deny_ids) != EXPECTED_PUBLIC_DENY_ROW_COUNT
        or 347 not in deny_ids
        or 90 in deny_ids
    ):
        raise ScarCssmSourceError("SCAR_PUBLIC_DENY_CONTRACT_INVALID")
    for source_id, expected_hash in PUBLIC_DENY_RAW_LINE_SHA256_BY_ID.items():
        row = by_id.get(source_id)
        if row is None or not hmac.compare_digest(
            row.raw_line_sha256, expected_hash
        ):
            raise ScarCssmSourceError("SCAR_PUBLIC_DENY_SOURCE_DRIFT")

    deny_rows = tuple(row for row in rows if row.source_id in deny_ids)
    ambiguous_rows = tuple(row for row in rows if _has_normalized_duplicate_slot(row))
    ambiguous_ids = frozenset(row.source_id for row in ambiguous_rows)
    primary_rows = tuple(
        row
        for row in rows
        if row.source_id not in deny_ids and row.source_id not in ambiguous_ids
    )
    arity_counts: dict[int, int] = {}
    for row in primary_rows:
        arity_counts[len(row.mappings)] = arity_counts.get(len(row.mappings), 0) + 1
    domains = {
        domain
        for row in rows
        for domain in (row.system_a_domain, row.system_b_domain)
    }
    if (
        ambiguous_ids != EXPECTED_NORMALIZED_DUPLICATE_SLOT_ROW_IDS
        or deny_ids & ambiguous_ids
        or len(deny_rows) != EXPECTED_PUBLIC_DENY_ROW_COUNT
        or sum(len(row.mappings) for row in deny_rows)
        != EXPECTED_PUBLIC_DENY_MAPPING_COUNT
        or _line_hash_list_commitment(deny_rows)
        != EXPECTED_PUBLIC_DENY_LINE_HASH_LIST_SHA256
        or len(ambiguous_rows) != EXPECTED_AMBIGUOUS_ROW_COUNT
        or sum(len(row.mappings) for row in ambiguous_rows)
        != EXPECTED_AMBIGUOUS_MAPPING_COUNT
        or _line_hash_list_commitment(ambiguous_rows)
        != EXPECTED_AMBIGUOUS_LINE_HASH_LIST_SHA256
        or len(primary_rows) != EXPECTED_PRIMARY_ROW_COUNT
        or sum(len(row.mappings) for row in primary_rows)
        != EXPECTED_PRIMARY_MAPPING_COUNT
        or arity_counts != EXPECTED_PRIMARY_ARITY_COUNTS
        or _line_hash_list_commitment(primary_rows)
        != EXPECTED_PRIMARY_LINE_HASH_LIST_SHA256
        or sum(
            row.system_a_domain == row.system_b_domain for row in primary_rows
        )
        != EXPECTED_PRIMARY_INTRA_DOMAIN_COUNT
        or sum(
            row.system_a_domain != row.system_b_domain for row in primary_rows
        )
        != EXPECTED_PRIMARY_CROSS_DOMAIN_COUNT
        or len(domains) != EXPECTED_DOMAIN_COUNT
    ):
        raise ScarCssmSourceError("SCAR_COHORT_TOPOLOGY_INVALID")


def _item_token(secret: bytes, study_id: str, row: _SourceRow) -> str:
    digest = _hmac_hex(
        secret,
        _ITEM_TOKEN_DOMAIN,
        study_id.encode("ascii"),
        row.raw_line_sha256.encode("ascii"),
    )
    return f"scar-item-v1-{digest}"


def _slot_token(
    secret: bytes,
    study_id: str,
    item_token: str,
    side: str,
    occurrence: int,
    surface: str,
) -> str:
    digest = _hmac_hex(
        secret,
        _SLOT_TOKEN_DOMAIN,
        study_id.encode("ascii"),
        item_token.encode("ascii"),
        side.encode("ascii"),
        occurrence.to_bytes(4, "big"),
        _normalized_slot_key(surface).encode("utf-8"),
    )
    return f"scar-slot-v1-{digest}"


def _slot_order_key(
    secret: bytes, study_id: str, item_token: str, side: str, slot_id: str
) -> str:
    return _hmac_hex(
        secret,
        _SLOT_ORDER_DOMAIN,
        study_id.encode("ascii"),
        item_token.encode("ascii"),
        side.encode("ascii"),
        slot_id.encode("ascii"),
    )


def _item_order_key(secret: bytes, study_id: str, item_token: str) -> str:
    return _hmac_hex(
        secret,
        _ITEM_ORDER_DOMAIN,
        study_id.encode("ascii"),
        item_token.encode("ascii"),
    )


def _side_action(
    *, system: str, background: str, slots: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    return {
        "background": background,
        "slots": [
            {"opaque_slot_id": slot_id, "surface": surface.strip()}
            for slot_id, surface in slots
        ],
        "system": system,
    }


def _build_core_packs(
    rows: tuple[_SourceRow, ...], *, secret: bytes, study_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    deny_ids = frozenset(PUBLIC_DENY_RAW_LINE_SHA256_BY_ID)
    ambiguous_ids = EXPECTED_NORMALIZED_DUPLICATE_SLOT_ROW_IDS
    built: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        if row.source_id in deny_ids:
            continue
        item_token = _item_token(secret, study_id, row)
        side_a_slots: list[tuple[str, str]] = []
        side_b_slots: list[tuple[str, str]] = []
        gold_base: list[list[str]] = []
        for occurrence, (surface_a, surface_b) in enumerate(row.mappings):
            slot_a = _slot_token(
                secret,
                study_id,
                item_token,
                "a",
                occurrence,
                surface_a,
            )
            slot_b = _slot_token(
                secret,
                study_id,
                item_token,
                "b",
                occurrence,
                surface_b,
            )
            side_a_slots.append((slot_a, surface_a))
            side_b_slots.append((slot_b, surface_b))
            gold_base.append([slot_a, slot_b])
        keyed_side_a = [
            (
                _slot_order_key(secret, study_id, item_token, "a", pair[0]),
                pair,
            )
            for pair in side_a_slots
        ]
        keyed_side_b = [
            (
                _slot_order_key(secret, study_id, item_token, "b", pair[0]),
                pair,
            )
            for pair in side_b_slots
        ]
        if (
            len({key for key, _ in keyed_side_a}) != len(keyed_side_a)
            or len({key for key, _ in keyed_side_b}) != len(keyed_side_b)
        ):
            raise ScarCssmSourceError("SCAR_HMAC_ORDER_COLLISION")
        side_a_slots = [pair for _, pair in sorted(keyed_side_a)]
        side_b_slots = [pair for _, pair in sorted(keyed_side_b)]
        action_a = _side_action(
            system=row.system_a,
            background=row.system_a_background,
            slots=side_a_slots,
        )
        action_b = _side_action(
            system=row.system_b,
            background=row.system_b_background,
            slots=side_b_slots,
        )
        action_item = {
            "item_token": item_token,
            "variants": {
                "base": {"left": action_a, "right": action_b},
                "system_swap": {"left": action_b, "right": action_a},
            },
        }
        cohort = (
            "ambiguous_secondary"
            if row.source_id in ambiguous_ids
            else "primary_unique_slot"
        )
        label_item = {
            "gold_pairs": {
                "base": gold_base,
                "system_swap": [[right, left] for left, right in gold_base],
            },
            "item_token": item_token,
            "strata": {
                "arity": len(row.mappings),
                "cohort": cohort,
                "domain_relation": (
                    "intra"
                    if row.system_a_domain == row.system_b_domain
                    else "cross"
                ),
                "system_a_domain": row.system_a_domain,
                "system_b_domain": row.system_b_domain,
            },
        }
        built.append(
            (
                _item_order_key(secret, study_id, item_token),
                action_item,
                label_item,
            )
        )
    if len({value[0] for value in built}) != len(built):
        raise ScarCssmSourceError("SCAR_HMAC_ORDER_COLLISION")
    built.sort(key=lambda value: value[0])
    action_core = {
        "items": [value[1] for value in built],
        "schema": ACTION_SCHEMA,
        "slot_collection_semantics": "unordered",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
        "study_id": study_id,
        "variant_names": list(VARIANT_NAMES),
    }
    label_core = {
        "items": [value[2] for value in built],
        "schema": LABEL_SCHEMA,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
        "study_id": study_id,
        "variant_names": list(VARIANT_NAMES),
    }
    return action_core, label_core


def _pack_binding_hmac(
    secret: bytes,
    *,
    study_id: str,
    action_commitment: str,
    label_commitment: str,
) -> str:
    return _hmac_hex(
        secret,
        _PACK_BINDING_DOMAIN,
        study_id.encode("ascii"),
        EXPECTED_SOURCE_SHA256.encode("ascii"),
        action_commitment.encode("ascii"),
        label_commitment.encode("ascii"),
    )


def _finish_packs(
    action_core: dict[str, Any],
    label_core: dict[str, Any],
    *,
    secret: bytes,
    study_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    action_commitment = _content_hash(action_core)
    label_commitment = _content_hash(label_core)
    cross_binding = _pack_binding_hmac(
        secret,
        study_id=study_id,
        action_commitment=action_commitment,
        label_commitment=label_commitment,
    )
    action_body = {
        **action_core,
        "action_commitment_sha256": action_commitment,
        "cross_binding_hmac_sha256": cross_binding,
        "label_commitment_sha256": label_commitment,
    }
    label_body = {
        **label_core,
        "action_commitment_sha256": action_commitment,
        "cross_binding_hmac_sha256": cross_binding,
        "label_commitment_sha256": label_commitment,
    }
    return (
        {**action_body, "self_sha256": _content_hash(action_body)},
        {**label_body, "self_sha256": _content_hash(label_body)},
    )


def _walk_forbidden_action_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key in _ACTION_FORBIDDEN_KEYS:
                raise ScarCssmSourceError("SCAR_ACTION_LABEL_LEAKAGE")
            _walk_forbidden_action_keys(child)
    elif isinstance(value, list):
        for child in value:
            _walk_forbidden_action_keys(child)


def _without_keys(value: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in keys}


def _validate_action_shape(action_core: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        set(action_core)
        != {
            "items",
            "schema",
            "slot_collection_semantics",
            "source_sha256",
            "source_size_bytes",
            "study_id",
            "variant_names",
        }
        or action_core.get("schema") != ACTION_SCHEMA
        or action_core.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or action_core.get("source_size_bytes") != EXPECTED_SOURCE_SIZE_BYTES
        or action_core.get("slot_collection_semantics") != "unordered"
        or action_core.get("variant_names") != list(VARIANT_NAMES)
        or type(action_core.get("items")) is not list
        or len(action_core["items"]) != EXPECTED_ACTION_ITEM_COUNT
    ):
        raise ScarCssmSourceError("SCAR_ACTION_PACK_INVALID")
    _walk_forbidden_action_keys(action_core["items"])
    by_token: dict[str, dict[str, Any]] = {}
    all_slot_ids: set[str] = set()
    for item in action_core["items"]:
        if type(item) is not dict or set(item) != {"item_token", "variants"}:
            raise ScarCssmSourceError("SCAR_ACTION_ITEM_INVALID")
        token = item["item_token"]
        variants = item["variants"]
        if (
            not isinstance(token, str)
            or _ITEM_TOKEN.fullmatch(token) is None
            or token in by_token
            or type(variants) is not dict
            or tuple(variants) != VARIANT_NAMES
        ):
            raise ScarCssmSourceError("SCAR_ACTION_ITEM_INVALID")
        base = variants["base"]
        swapped = variants["system_swap"]
        if (
            type(base) is not dict
            or set(base) != {"left", "right"}
            or type(swapped) is not dict
            or set(swapped) != {"left", "right"}
            or swapped["left"] != base["right"]
            or swapped["right"] != base["left"]
        ):
            raise ScarCssmSourceError("SCAR_ACTION_VARIANT_INVALID")
        side_ids: list[set[str]] = []
        for side in (base["left"], base["right"]):
            if (
                type(side) is not dict
                or set(side) != {"background", "slots", "system"}
                or type(side["system"]) is not str
                or not side["system"].strip()
                or type(side["background"]) is not str
                or not side["background"].strip()
                or type(side["slots"]) is not list
                or not 2 <= len(side["slots"]) <= 14
            ):
                raise ScarCssmSourceError("SCAR_ACTION_SIDE_INVALID")
            ids: set[str] = set()
            for slot in side["slots"]:
                if (
                    type(slot) is not dict
                    or set(slot) != {"opaque_slot_id", "surface"}
                    or not isinstance(slot["opaque_slot_id"], str)
                    or _SLOT_TOKEN.fullmatch(slot["opaque_slot_id"]) is None
                    or slot["opaque_slot_id"] in ids
                    or slot["opaque_slot_id"] in all_slot_ids
                    or type(slot["surface"]) is not str
                    or not slot["surface"].strip()
                ):
                    raise ScarCssmSourceError("SCAR_ACTION_SLOT_INVALID")
                ids.add(slot["opaque_slot_id"])
                all_slot_ids.add(slot["opaque_slot_id"])
            side_ids.append(ids)
        if side_ids[0] & side_ids[1] or len(side_ids[0]) != len(side_ids[1]):
            raise ScarCssmSourceError("SCAR_ACTION_SIDE_SLOT_SET_INVALID")
        by_token[token] = item
    return by_token


def _validate_label_shape(
    label_core: Mapping[str, Any], action_by_token: Mapping[str, dict[str, Any]]
) -> None:
    if (
        set(label_core)
        != {
            "items",
            "schema",
            "source_sha256",
            "source_size_bytes",
            "study_id",
            "variant_names",
        }
        or label_core.get("schema") != LABEL_SCHEMA
        or label_core.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or label_core.get("source_size_bytes") != EXPECTED_SOURCE_SIZE_BYTES
        or label_core.get("variant_names") != list(VARIANT_NAMES)
        or type(label_core.get("items")) is not list
        or len(label_core["items"]) != EXPECTED_ACTION_ITEM_COUNT
    ):
        raise ScarCssmSourceError("SCAR_LABEL_PACK_INVALID")
    seen: set[str] = set()
    cohort_counts = {"primary_unique_slot": 0, "ambiguous_secondary": 0}
    mapping_counts = {"primary_unique_slot": 0, "ambiguous_secondary": 0}
    intra_primary = 0
    cross_primary = 0
    domains: set[str] = set()
    for item in label_core["items"]:
        if (
            type(item) is not dict
            or set(item) != {"gold_pairs", "item_token", "strata"}
        ):
            raise ScarCssmSourceError("SCAR_LABEL_ITEM_INVALID")
        token = item["item_token"]
        action_item = action_by_token.get(token)
        if action_item is None or token in seen:
            raise ScarCssmSourceError("SCAR_ACTION_LABEL_ITEM_CROSS_BINDING_INVALID")
        seen.add(token)
        strata = item["strata"]
        cohort_value = strata.get("cohort") if type(strata) is dict else None
        domain_relation = (
            strata.get("domain_relation") if type(strata) is dict else None
        )
        if (
            type(strata) is not dict
            or set(strata)
            != {
                "arity",
                "cohort",
                "domain_relation",
                "system_a_domain",
                "system_b_domain",
            }
            or type(strata["arity"]) is not int
            or isinstance(strata["arity"], bool)
            or not 2 <= strata["arity"] <= 14
            or type(cohort_value) is not str
            or cohort_value not in cohort_counts
            or type(domain_relation) is not str
            or domain_relation not in {"intra", "cross"}
            or type(strata["system_a_domain"]) is not str
            or not strata["system_a_domain"]
            or type(strata["system_b_domain"]) is not str
            or not strata["system_b_domain"]
            or (
                strata["domain_relation"] == "intra"
            )
            != (strata["system_a_domain"] == strata["system_b_domain"])
        ):
            raise ScarCssmSourceError("SCAR_LABEL_STRATA_INVALID")
        cohort = cohort_value
        cohort_counts[cohort] += 1
        domains.update((strata["system_a_domain"], strata["system_b_domain"]))
        if cohort == "primary_unique_slot":
            intra_primary += strata["domain_relation"] == "intra"
            cross_primary += strata["domain_relation"] == "cross"

        pairs = item["gold_pairs"]
        if type(pairs) is not dict or tuple(pairs) != VARIANT_NAMES:
            raise ScarCssmSourceError("SCAR_LABEL_GOLD_INVALID")
        base_action = action_item["variants"]["base"]
        left_ids = {
            slot["opaque_slot_id"] for slot in base_action["left"]["slots"]
        }
        right_ids = {
            slot["opaque_slot_id"] for slot in base_action["right"]["slots"]
        }
        base_pairs = pairs["base"]
        swapped_pairs = pairs["system_swap"]
        if (
            type(base_pairs) is not list
            or type(swapped_pairs) is not list
            or len(base_pairs) != strata["arity"]
            or len(swapped_pairs) != strata["arity"]
            or any(
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
                or pair[0] not in left_ids
                or pair[1] not in right_ids
                for pair in base_pairs
            )
            or len({pair[0] for pair in base_pairs}) != len(left_ids)
            or len({pair[1] for pair in base_pairs}) != len(right_ids)
            or swapped_pairs != [[right, left] for left, right in base_pairs]
        ):
            raise ScarCssmSourceError("SCAR_LABEL_GOLD_INVALID")
        mapping_counts[cohort] += len(base_pairs)
    if seen != set(action_by_token):
        raise ScarCssmSourceError("SCAR_ACTION_LABEL_ITEM_CROSS_BINDING_INVALID")
    if (
        cohort_counts
        != {
            "primary_unique_slot": EXPECTED_PRIMARY_ROW_COUNT,
            "ambiguous_secondary": EXPECTED_AMBIGUOUS_ROW_COUNT,
        }
        or mapping_counts
        != {
            "primary_unique_slot": EXPECTED_PRIMARY_MAPPING_COUNT,
            "ambiguous_secondary": EXPECTED_AMBIGUOUS_MAPPING_COUNT,
        }
        or intra_primary != EXPECTED_PRIMARY_INTRA_DOMAIN_COUNT
        or cross_primary != EXPECTED_PRIMARY_CROSS_DOMAIN_COUNT
        or len(domains) != EXPECTED_DOMAIN_COUNT
    ):
        raise ScarCssmSourceError("SCAR_LABEL_AGGREGATE_INVALID")


def validate_scar_cssm_action_pack_v1(
    action_pack: Mapping[str, Any], *, study_id: str
) -> None:
    """Validate the complete action-only worker surface.

    This boundary deliberately has no source path, label pack, or HMAC
    secret.  It proves the action core/self commitments and the frozen action
    topology.  The label commitment and cross-binding fields are checked only
    as present SHA-256-shaped commitments; validating their relationship to a
    label pack remains the responsibility of
    :func:`validate_scar_cssm_pack_binding_v1`.
    """

    _require_study_id(study_id)
    if not isinstance(action_pack, Mapping):
        raise ScarCssmSourceError("SCAR_PACK_TYPE_INVALID")
    final_keys = frozenset(
        {
            "action_commitment_sha256",
            "cross_binding_hmac_sha256",
            "label_commitment_sha256",
            "self_sha256",
        }
    )
    action_core = _without_keys(action_pack, final_keys)
    if set(action_pack) != set(action_core) | final_keys:
        raise ScarCssmSourceError("SCAR_ACTION_PACK_INVALID")
    if action_core.get("study_id") != study_id:
        raise ScarCssmSourceError("SCAR_STUDY_CROSS_BINDING_INVALID")
    for key in final_keys:
        value = action_pack.get(key)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise ScarCssmSourceError("SCAR_PACK_COMMITMENT_INVALID")
    expected_action_commitment = _content_hash(action_core)
    if not hmac.compare_digest(
        action_pack["action_commitment_sha256"],
        expected_action_commitment,
    ):
        raise ScarCssmSourceError("SCAR_PACK_COMMITMENT_INVALID")
    action_body = _without_keys(action_pack, frozenset({"self_sha256"}))
    if not hmac.compare_digest(
        action_pack["self_sha256"], _content_hash(action_body)
    ):
        raise ScarCssmSourceError("SCAR_PACK_SELF_HASH_INVALID")
    _validate_action_shape(action_core)


def validate_scar_cssm_pack_binding_v1(
    action_pack: Mapping[str, Any],
    label_pack: Mapping[str, Any],
    *,
    secret: bytes,
    study_id: str,
) -> None:
    """Recompute both pack commitments and their secret cross-binding."""

    _require_secret(secret)
    _require_study_id(study_id)
    if not isinstance(action_pack, Mapping) or not isinstance(label_pack, Mapping):
        raise ScarCssmSourceError("SCAR_PACK_TYPE_INVALID")
    final_keys = frozenset(
        {
            "action_commitment_sha256",
            "cross_binding_hmac_sha256",
            "label_commitment_sha256",
            "self_sha256",
        }
    )
    action_core = _without_keys(action_pack, final_keys)
    label_core = _without_keys(label_pack, final_keys)
    if set(action_pack) != set(action_core) | final_keys:
        raise ScarCssmSourceError("SCAR_ACTION_PACK_INVALID")
    if set(label_pack) != set(label_core) | final_keys:
        raise ScarCssmSourceError("SCAR_LABEL_PACK_INVALID")
    if (
        action_core.get("study_id") != study_id
        or label_core.get("study_id") != study_id
    ):
        raise ScarCssmSourceError("SCAR_STUDY_CROSS_BINDING_INVALID")
    action_commitment = _content_hash(action_core)
    label_commitment = _content_hash(label_core)
    expected_binding = _pack_binding_hmac(
        secret,
        study_id=study_id,
        action_commitment=action_commitment,
        label_commitment=label_commitment,
    )
    action_body = _without_keys(action_pack, frozenset({"self_sha256"}))
    label_body = _without_keys(label_pack, frozenset({"self_sha256"}))
    for pack in (action_pack, label_pack):
        if (
            not isinstance(pack.get("action_commitment_sha256"), str)
            or not isinstance(pack.get("label_commitment_sha256"), str)
            or not isinstance(pack.get("cross_binding_hmac_sha256"), str)
            or not isinstance(pack.get("self_sha256"), str)
            or any(
                _HEX64.fullmatch(pack[key]) is None
                for key in final_keys
            )
            or not hmac.compare_digest(
                pack["action_commitment_sha256"], action_commitment
            )
            or not hmac.compare_digest(
                pack["label_commitment_sha256"], label_commitment
            )
            or not hmac.compare_digest(
                pack["cross_binding_hmac_sha256"], expected_binding
            )
        ):
            raise ScarCssmSourceError("SCAR_PACK_COMMITMENT_INVALID")
    if (
        not hmac.compare_digest(
            action_pack["self_sha256"], _content_hash(action_body)
        )
        or not hmac.compare_digest(
            label_pack["self_sha256"], _content_hash(label_body)
        )
    ):
        raise ScarCssmSourceError("SCAR_PACK_SELF_HASH_INVALID")
    action_by_token = _validate_action_shape(action_core)
    _validate_label_shape(label_core, action_by_token)


def _safe_aggregate(
    *, action_pack: Mapping[str, Any], label_pack: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "access_counts": {
            "model_call_count": 0,
            "network_call_count": 0,
            "scorer_call_count": 0,
            "source_access_count": 1,
        },
        "action_item_count": EXPECTED_ACTION_ITEM_COUNT,
        "action_pack_commitment_sha256": action_pack[
            "action_commitment_sha256"
        ],
        "action_variant_count": EXPECTED_ACTION_VARIANT_COUNT,
        "ambiguous_mapping_count": EXPECTED_AMBIGUOUS_MAPPING_COUNT,
        "ambiguous_row_count": EXPECTED_AMBIGUOUS_ROW_COUNT,
        "ambiguous_row_set_commitment_sha256": (
            EXPECTED_AMBIGUOUS_LINE_HASH_LIST_SHA256
        ),
        "cross_binding_hmac_sha256": action_pack[
            "cross_binding_hmac_sha256"
        ],
        "domain_count": EXPECTED_DOMAIN_COUNT,
        "label_pack_commitment_sha256": label_pack[
            "label_commitment_sha256"
        ],
        "nonpublic_mapping_count": EXPECTED_NONPUBLIC_MAPPING_COUNT,
        "primary_arity_counts": {
            str(key): value for key, value in EXPECTED_PRIMARY_ARITY_COUNTS.items()
        },
        "primary_cross_domain_count": EXPECTED_PRIMARY_CROSS_DOMAIN_COUNT,
        "primary_intra_domain_count": EXPECTED_PRIMARY_INTRA_DOMAIN_COUNT,
        "primary_mapping_count": EXPECTED_PRIMARY_MAPPING_COUNT,
        "primary_row_count": EXPECTED_PRIMARY_ROW_COUNT,
        "primary_row_set_commitment_sha256": (
            EXPECTED_PRIMARY_LINE_HASH_LIST_SHA256
        ),
        "public_deny_mapping_count": EXPECTED_PUBLIC_DENY_MAPPING_COUNT,
        "public_deny_row_count": EXPECTED_PUBLIC_DENY_ROW_COUNT,
        "public_deny_row_set_commitment_sha256": (
            EXPECTED_PUBLIC_DENY_LINE_HASH_LIST_SHA256
        ),
        "safe_claim_scope": (
            "source_identity_topology_action_label_isolation_and_opaque_"
            "two_variant_construction_only_no_model_no_scorer_no_effect"
        ),
        "schema": SAFE_AGGREGATE_SCHEMA,
        "source_binding": {
            "row_count": EXPECTED_SOURCE_ROW_COUNT,
            "sha256": EXPECTED_SOURCE_SHA256,
            "size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
            "total_mapping_count": EXPECTED_SOURCE_MAPPING_COUNT,
        },
        "status": "qualified",
        "variant_names": list(VARIANT_NAMES),
        "version": VERSION,
    }
    return {**body, "self_sha256": _content_hash(body)}


def compile_scar_cssm_source_v1(
    source_path: Path,
    *,
    secret: bytes,
    study_id: str,
) -> ScarCssmSourceCompilation:
    """Compile the exact private SCAR-English copy into isolated packs."""

    _require_secret(secret)
    _require_study_id(study_id)
    raw = _read_private_regular_source_once(source_path)
    rows = _parse_source_rows(
        raw,
        expected_row_count=EXPECTED_SOURCE_ROW_COUNT,
        expected_mapping_count=EXPECTED_SOURCE_MAPPING_COUNT,
        expected_ids=EXPECTED_SOURCE_IDS,
    )
    _validate_official_topology(rows)
    action_core, label_core = _build_core_packs(
        rows, secret=secret, study_id=study_id
    )
    action_pack, label_pack = _finish_packs(
        action_core,
        label_core,
        secret=secret,
        study_id=study_id,
    )
    validate_scar_cssm_pack_binding_v1(
        action_pack,
        label_pack,
        secret=secret,
        study_id=study_id,
    )
    aggregate = _safe_aggregate(action_pack=action_pack, label_pack=label_pack)
    return ScarCssmSourceCompilation(
        action_pack=action_pack,
        label_pack=label_pack,
        safe_aggregate=aggregate,
    )


__all__ = [
    "ACTION_SCHEMA",
    "EXPECTED_ACTION_ITEM_COUNT",
    "EXPECTED_ACTION_VARIANT_COUNT",
    "EXPECTED_AMBIGUOUS_LINE_HASH_LIST_SHA256",
    "EXPECTED_AMBIGUOUS_MAPPING_COUNT",
    "EXPECTED_AMBIGUOUS_ROW_COUNT",
    "EXPECTED_NORMALIZED_DUPLICATE_SLOT_ROW_IDS",
    "EXPECTED_PRIMARY_LINE_HASH_LIST_SHA256",
    "EXPECTED_PRIMARY_MAPPING_COUNT",
    "EXPECTED_PRIMARY_ROW_COUNT",
    "EXPECTED_PUBLIC_DENY_LINE_HASH_LIST_SHA256",
    "EXPECTED_PUBLIC_DENY_MAPPING_COUNT",
    "EXPECTED_PUBLIC_DENY_ROW_COUNT",
    "EXPECTED_SOURCE_MAPPING_COUNT",
    "EXPECTED_SOURCE_ROW_COUNT",
    "EXPECTED_SOURCE_SHA256",
    "EXPECTED_SOURCE_SIZE_BYTES",
    "HMAC_SECRET_BYTES",
    "LABEL_SCHEMA",
    "PUBLIC_DENY_RAW_LINE_SHA256_BY_ID",
    "SAFE_AGGREGATE_SCHEMA",
    "ScarCssmSourceCompilation",
    "ScarCssmSourceError",
    "VARIANT_NAMES",
    "VERSION",
    "compile_scar_cssm_source_v1",
    "validate_scar_cssm_action_pack_v1",
    "validate_scar_cssm_pack_binding_v1",
]

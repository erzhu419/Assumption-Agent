"""One-pass, in-memory, offline scorer for the frozen SCAR CSSM study.

The prediction pack is sealed before labels are opened.  One later call scores
all five arms together; there is no model, file, network, API, or online
evaluator access in this module.  The public entry point accepts only the
official 391-case source packs and delegates their cross-binding validation to
``gscl_scar_cssm_source_v1.validate_scar_cssm_pack_binding_v1``.

The underscore-prefixed fixture entry points exist only so the schema and
statistics can be tested with tiny in-memory packs.  They do not authorize an
effect claim and cannot impersonate the official entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from assumption_agent.benchmarks import gscl_scar_cssm_source_v1 as source


VERSION = "gscl_scar_cssm_score_v1"
PREDICTION_PACK_SCHEMA = f"{VERSION}.prediction_pack.v1"
PRIVATE_RESULT_SCHEMA = f"{VERSION}.private_result.v1"
SAFE_AGGREGATE_SCHEMA = f"{VERSION}.safe_aggregate.v1"

ARM_IDS = (
    "semantic_only",
    "flat_structural",
    "full_no_composition",
    "full_with_length2_composition",
    "full_with_length2_composition_target_color_shuffle",
)
VARIANT_NAMES = source.VARIANT_NAMES
DISPOSITIONS = frozenset({"ANSWER", "ABSTAIN", "ERROR"})
STRUCTURAL_STATUSES = frozenset(
    {"EXECUTED_WITHOUT_TYPED_FAILURE", "TYPED_FAILURE"}
)
ERROR_CODES = frozenset(
    {
        "DOCUMENT_EXTRACTOR_TYPED_FAILURE",
        "BOUNDED_CONSUMER_TYPED_FAILURE",
        "SLOT_BINDER_TYPED_FAILURE",
        "PROPOSAL_CONSTRUCTION_TYPED_FAILURE",
        "NO_FEASIBLE_INJECTIVE_PAIR_SET",
        "INTERNAL_TYPED_FAILURE",
    }
)
MAX_PROPOSALS_PER_POOL = 64
MAX_DIAGNOSTIC_COUNT = 4_096
MAX_SCORE_ABS = 1_000_000_000

BINDER_COVERAGE_DISPOSITIONS = frozenset(
    {"COMPLETE_SELECTED_SET", "PARTIAL_SELECTED_SET", "EMPTY_ABSTENTION"}
)
OPERATOR_IDS = frozenset(
    f"ori_{orientation}.pol_{polarity}.slots_{slot_order}"
    for orientation in ("keep", "inv")
    for polarity in ("keep", "inv")
    for slot_order in ("identity", "reverse")
)
MAPPING_ARM_IDS = ARM_IDS[:4]

DOCUMENT_ENVELOPE_RECEIPT_SCHEMA = (
    "gscl_narrative_document_envelope_v1.safe_receipt.v1"
)
LEAF_RECEIPT_SCHEMA = (
    "gscl_narrative_hierarchical_closed_choice_v2.private_selection_receipt.v1"
)
BOUNDED_SET_RECEIPT_SCHEMA = (
    "gscl_bounded_narrative_relation_set_consumer_v1.safe_receipt.v1"
)
BINDER_RECEIPT_SCHEMA = (
    "scar.within_side_minilm_slot_graph_binder.v1.safe_receipt.v1"
)
SEMANTIC_MATRIX_RECEIPT_SCHEMA = (
    "scar.within_side_minilm_slot_graph_binder.v1.semantic_matrix.safe_receipt.v1"
)
SLOT_GRAPH_RECEIPT_SCHEMA = (
    "scar.categorical_slot_set_mapping.v1.slot_graph.safe_receipt.v1"
)
MAPPING_RECEIPT_SCHEMA = (
    "scar.categorical_slot_set_mapping.v1.mapping.safe_receipt.v1"
)

BOOTSTRAP_SEED = 20_260_801
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95

# This is deliberately one confirmatory comparison, not a growing collection
# of effect gates.  It directly tests the study's central claim that the full
# generalized-counterpoint mechanism adds value beyond semantic similarity.
# Every other arm, strict-exact endpoint, and stratum remains a predeclared
# secondary diagnostic and cannot change the primary disposition.
PRIMARY_ARM_ID = "full_with_length2_composition"
PRIMARY_COMPARATOR_ID = "semantic_only"
PRIMARY_EFFECT_NAME = "full_minus_semantic"
PRIMARY_ENDPOINT = "paired_item_mean_of_base_and_swap_pair_f1"
PRIMARY_SUCCESS_RULE = (
    "paired_item_bootstrap_95pct_ci_lower_bound_strictly_greater_than_zero"
)

_PAIRED_COMPARISONS = (
    (PRIMARY_EFFECT_NAME, PRIMARY_COMPARATOR_ID),
    ("full_minus_flat", "flat_structural"),
    ("full_minus_no_composition", "full_no_composition"),
    (
        "full_minus_target_color_shuffle",
        "full_with_length2_composition_target_color_shuffle",
    ),
)
_STRICT_BOOTSTRAP_SEED_OFFSET = 100
_STRATIFIED_BOOTSTRAP_SEED_OFFSET = 200

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_STUDY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ITEM_TOKEN = re.compile(r"scar-item-v1-[0-9a-f]{64}\Z")
_SLOT_TOKEN = re.compile(r"scar-slot-v1-[0-9a-f]{64}\Z")

_PACK_FINAL_KEYS = frozenset(
    {
        "action_commitment_sha256",
        "cross_binding_hmac_sha256",
        "label_commitment_sha256",
        "self_sha256",
    }
)


class ScarCssmScoreError(RuntimeError):
    """Stable fail-closed scoring error."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


@dataclass(frozen=True)
class ScarCssmScoreResult:
    """Private per-case result plus disclosure-safe aggregate."""

    private_result: dict[str, Any]
    safe_aggregate: dict[str, Any]


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
        raise ScarCssmScoreError("SCAR_SCORE_CANONICAL_JSON_INVALID") from exc


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _require_secret_and_study(secret: bytes, study_id: str) -> None:
    if type(secret) is not bytes or len(secret) != source.HMAC_SECRET_BYTES:
        raise ScarCssmScoreError("SCAR_SCORE_SECRET_INVALID")
    if not isinstance(study_id, str) or _STUDY_ID.fullmatch(study_id) is None:
        raise ScarCssmScoreError("SCAR_SCORE_STUDY_ID_INVALID")


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], issue_id: str
) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ScarCssmScoreError(issue_id)


def _without_keys(
    value: Mapping[str, Any], keys: frozenset[str]
) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in keys}


def _validate_pack_hashes(pack: Mapping[str, Any], *, kind: str) -> None:
    if type(pack) is not dict:
        raise ScarCssmScoreError(f"SCAR_SCORE_{kind}_PACK_INVALID")
    core = _without_keys(pack, _PACK_FINAL_KEYS)
    if set(pack) != set(core) | _PACK_FINAL_KEYS:
        raise ScarCssmScoreError(f"SCAR_SCORE_{kind}_PACK_INVALID")
    for key in _PACK_FINAL_KEYS:
        if not _is_hex64(pack.get(key)):
            raise ScarCssmScoreError(f"SCAR_SCORE_{kind}_PACK_HASH_INVALID")
    if not hmac.compare_digest(
        pack["action_commitment_sha256"],
        _content_hash(_without_keys(pack, _PACK_FINAL_KEYS)),
    ) and kind == "ACTION":
        raise ScarCssmScoreError("SCAR_SCORE_ACTION_COMMITMENT_INVALID")
    if not hmac.compare_digest(
        pack["label_commitment_sha256"],
        _content_hash(_without_keys(pack, _PACK_FINAL_KEYS)),
    ) and kind == "LABEL":
        raise ScarCssmScoreError("SCAR_SCORE_LABEL_COMMITMENT_INVALID")
    body = _without_keys(pack, frozenset({"self_sha256"}))
    if not hmac.compare_digest(pack["self_sha256"], _content_hash(body)):
        raise ScarCssmScoreError(f"SCAR_SCORE_{kind}_PACK_SELF_HASH_INVALID")


def _index_action_pack(
    action_pack: Mapping[str, Any], *, expected_case_count: int
) -> dict[str, dict[str, Any]]:
    _validate_pack_hashes(action_pack, kind="ACTION")
    core = _without_keys(action_pack, _PACK_FINAL_KEYS)
    if (
        set(core)
        != {
            "items",
            "schema",
            "slot_collection_semantics",
            "source_sha256",
            "source_size_bytes",
            "study_id",
            "variant_names",
        }
        or core.get("schema") != source.ACTION_SCHEMA
        or core.get("source_sha256") != source.EXPECTED_SOURCE_SHA256
        or core.get("source_size_bytes") != source.EXPECTED_SOURCE_SIZE_BYTES
        or core.get("slot_collection_semantics") != "unordered"
        or core.get("variant_names") != list(VARIANT_NAMES)
        or type(core.get("items")) is not list
        or len(core["items"]) != expected_case_count
    ):
        raise ScarCssmScoreError("SCAR_SCORE_ACTION_PACK_INVALID")

    indexed: dict[str, dict[str, Any]] = {}
    all_slots: set[str] = set()
    for item in core["items"]:
        _require_exact_keys(
            item, {"item_token", "variants"}, "SCAR_SCORE_ACTION_ITEM_INVALID"
        )
        token = item["item_token"]
        variants = item["variants"]
        if (
            not isinstance(token, str)
            or _ITEM_TOKEN.fullmatch(token) is None
            or token in indexed
            or type(variants) is not dict
            or tuple(variants) != VARIANT_NAMES
        ):
            raise ScarCssmScoreError("SCAR_SCORE_ACTION_ITEM_INVALID")
        base = variants["base"]
        swap = variants["system_swap"]
        if (
            type(base) is not dict
            or set(base) != {"left", "right"}
            or type(swap) is not dict
            or set(swap) != {"left", "right"}
            or swap["left"] != base["right"]
            or swap["right"] != base["left"]
        ):
            raise ScarCssmScoreError("SCAR_SCORE_ACTION_VARIANT_INVALID")
        side_sets: list[set[str]] = []
        for side in (base["left"], base["right"]):
            if (
                type(side) is not dict
                or set(side) != {"background", "slots", "system"}
                or type(side["background"]) is not str
                or not side["background"].strip()
                or type(side["system"]) is not str
                or not side["system"].strip()
                or type(side["slots"]) is not list
                or not 2 <= len(side["slots"]) <= 14
            ):
                raise ScarCssmScoreError("SCAR_SCORE_ACTION_SIDE_INVALID")
            side_ids: set[str] = set()
            for slot in side["slots"]:
                if (
                    type(slot) is not dict
                    or set(slot) != {"opaque_slot_id", "surface"}
                    or not isinstance(slot["opaque_slot_id"], str)
                    or _SLOT_TOKEN.fullmatch(slot["opaque_slot_id"]) is None
                    or slot["opaque_slot_id"] in side_ids
                    or slot["opaque_slot_id"] in all_slots
                    or type(slot["surface"]) is not str
                    or not slot["surface"].strip()
                ):
                    raise ScarCssmScoreError("SCAR_SCORE_ACTION_SLOT_INVALID")
                side_ids.add(slot["opaque_slot_id"])
                all_slots.add(slot["opaque_slot_id"])
            side_sets.append(side_ids)
        if side_sets[0] & side_sets[1] or len(side_sets[0]) != len(side_sets[1]):
            raise ScarCssmScoreError("SCAR_SCORE_ACTION_SIDE_SET_INVALID")
        indexed[token] = item
    return indexed


def _side_slot_sets(
    action_item: Mapping[str, Any], variant_name: str
) -> tuple[set[str], set[str]]:
    variant = action_item["variants"][variant_name]
    return (
        {slot["opaque_slot_id"] for slot in variant["left"]["slots"]},
        {slot["opaque_slot_id"] for slot in variant["right"]["slots"]},
    )


def _index_label_pack(
    label_pack: Mapping[str, Any],
    *,
    action_index: Mapping[str, Mapping[str, Any]],
    expected_primary_count: int,
    expected_ambiguous_count: int,
) -> dict[str, dict[str, Any]]:
    _validate_pack_hashes(label_pack, kind="LABEL")
    core = _without_keys(label_pack, _PACK_FINAL_KEYS)
    if (
        set(core)
        != {
            "items",
            "schema",
            "source_sha256",
            "source_size_bytes",
            "study_id",
            "variant_names",
        }
        or core.get("schema") != source.LABEL_SCHEMA
        or core.get("source_sha256") != source.EXPECTED_SOURCE_SHA256
        or core.get("source_size_bytes") != source.EXPECTED_SOURCE_SIZE_BYTES
        or core.get("variant_names") != list(VARIANT_NAMES)
        or type(core.get("items")) is not list
        or len(core["items"]) != len(action_index)
    ):
        raise ScarCssmScoreError("SCAR_SCORE_LABEL_PACK_INVALID")

    indexed: dict[str, dict[str, Any]] = {}
    cohort_counts = {"primary_unique_slot": 0, "ambiguous_secondary": 0}
    for item in core["items"]:
        _require_exact_keys(
            item,
            {"gold_pairs", "item_token", "strata"},
            "SCAR_SCORE_LABEL_ITEM_INVALID",
        )
        token = item["item_token"]
        if token not in action_index or token in indexed:
            raise ScarCssmScoreError("SCAR_SCORE_LABEL_ITEM_INVALID")
        strata = item["strata"]
        _require_exact_keys(
            strata,
            {
                "arity",
                "cohort",
                "domain_relation",
                "system_a_domain",
                "system_b_domain",
            },
            "SCAR_SCORE_LABEL_STRATA_INVALID",
        )
        if (
            type(strata["arity"]) is not int
            or isinstance(strata["arity"], bool)
            or not 2 <= strata["arity"] <= 14
            or strata["cohort"] not in cohort_counts
            or strata["domain_relation"] not in {"intra", "cross"}
            or type(strata["system_a_domain"]) is not str
            or not strata["system_a_domain"]
            or type(strata["system_b_domain"]) is not str
            or not strata["system_b_domain"]
            or (strata["domain_relation"] == "intra")
            != (strata["system_a_domain"] == strata["system_b_domain"])
        ):
            raise ScarCssmScoreError("SCAR_SCORE_LABEL_STRATA_INVALID")
        pairs_by_variant = item["gold_pairs"]
        if (
            type(pairs_by_variant) is not dict
            or tuple(pairs_by_variant) != VARIANT_NAMES
        ):
            raise ScarCssmScoreError("SCAR_SCORE_LABEL_PAIR_SET_INVALID")
        normalized: dict[str, list[list[str]]] = {}
        for variant_name in VARIANT_NAMES:
            left_ids, right_ids = _side_slot_sets(
                action_index[token], variant_name
            )
            pairs = _normalize_complete_pair_set(
                pairs_by_variant[variant_name],
                left_ids=left_ids,
                right_ids=right_ids,
                issue_id="SCAR_SCORE_LABEL_PAIR_SET_INVALID",
            )
            if len(pairs) != strata["arity"]:
                raise ScarCssmScoreError("SCAR_SCORE_LABEL_PAIR_SET_INVALID")
            normalized[variant_name] = pairs
        if normalized["system_swap"] != sorted(
            [[right, left] for left, right in normalized["base"]]
        ):
            raise ScarCssmScoreError("SCAR_SCORE_LABEL_SWAP_INVALID")
        cohort_counts[strata["cohort"]] += 1
        indexed[token] = item
    if set(indexed) != set(action_index) or cohort_counts != {
        "primary_unique_slot": expected_primary_count,
        "ambiguous_secondary": expected_ambiguous_count,
    }:
        raise ScarCssmScoreError("SCAR_SCORE_LABEL_COHORT_INVALID")
    return indexed


def _normalize_complete_pair_set(
    value: Any,
    *,
    left_ids: set[str],
    right_ids: set[str],
    issue_id: str,
) -> list[list[str]]:
    if type(value) is not list or len(value) != len(left_ids):
        raise ScarCssmScoreError(issue_id)
    pairs: list[list[str]] = []
    for pair in value:
        if (
            type(pair) is not list
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or pair[0] not in left_ids
            or pair[1] not in right_ids
        ):
            raise ScarCssmScoreError(issue_id)
        pairs.append([pair[0], pair[1]])
    if (
        len({pair[0] for pair in pairs}) != len(left_ids)
        or len({pair[1] for pair in pairs}) != len(right_ids)
        or len({(pair[0], pair[1]) for pair in pairs}) != len(pairs)
    ):
        raise ScarCssmScoreError(issue_id)
    return sorted(pairs)


def _normalize_disposition(
    value: Any, *, left_ids: set[str], right_ids: set[str]
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "disposition",
        "error_code",
        "pairs",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_DISPOSITION_INVALID")
    disposition = value["disposition"]
    if disposition not in DISPOSITIONS:
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_DISPOSITION_INVALID")
    if disposition == "ANSWER":
        if value["error_code"] is not None:
            raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ANSWER_INVALID")
        pairs = _normalize_complete_pair_set(
            value["pairs"],
            left_ids=left_ids,
            right_ids=right_ids,
            issue_id="SCAR_SCORE_PREDICTION_ANSWER_INVALID",
        )
        return {"disposition": "ANSWER", "pairs": pairs, "error_code": None}
    if disposition == "ABSTAIN":
        if value["pairs"] is not None or value["error_code"] is not None:
            raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ABSTAIN_INVALID")
        return {"disposition": "ABSTAIN", "pairs": None, "error_code": None}
    if value["pairs"] is not None or value["error_code"] not in ERROR_CODES:
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ERROR_INVALID")
    return {
        "disposition": "ERROR",
        "pairs": None,
        "error_code": value["error_code"],
    }


def _normalize_pool(
    value: Any, *, left_ids: set[str], right_ids: set[str]
) -> list[list[list[str]]]:
    if type(value) is not list or len(value) > MAX_PROPOSALS_PER_POOL:
        raise ScarCssmScoreError("SCAR_SCORE_PROPOSAL_POOL_INVALID")
    normalized: list[list[list[str]]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for candidate in value:
        pairs = _normalize_complete_pair_set(
            candidate,
            left_ids=left_ids,
            right_ids=right_ids,
            issue_id="SCAR_SCORE_PROPOSAL_POOL_INVALID",
        )
        key = tuple((left, right) for left, right in pairs)
        if key in seen:
            raise ScarCssmScoreError("SCAR_SCORE_PROPOSAL_POOL_INVALID")
        seen.add(key)
        normalized.append(pairs)
    return normalized


def _normalize_receipt_entry(value: Any, *, expected_schema: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "receipt",
        "receipt_sha256",
        "trailing_lf",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    receipt = value["receipt"]
    if (
        type(receipt) is not dict
        or receipt.get("schema") != expected_schema
        or type(value["trailing_lf"]) is not bool
        or not _is_hex64(value["receipt_sha256"])
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    body = dict(receipt)
    claimed = body.pop("self_sha256", None)
    if not _is_hex64(claimed) or not hmac.compare_digest(
        claimed, _content_hash(body)
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    raw = _canonical_bytes(receipt) + (b"\n" if value["trailing_lf"] else b"")
    if not hmac.compare_digest(
        value["receipt_sha256"], hashlib.sha256(raw).hexdigest()
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return {
        "receipt": json.loads(_canonical_bytes(receipt)),
        "receipt_sha256": value["receipt_sha256"],
        "trailing_lf": value["trailing_lf"],
    }


def _normalize_leaf_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "segment_id",
        "wire_completion",
        "wire_completion_sha256",
        "canonical_completion",
        "canonical_completion_sha256",
        "leaf_receipt",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if (
        not isinstance(value["segment_id"], str)
        or not value["segment_id"]
        or not isinstance(value["wire_completion"], str)
        or not isinstance(value["canonical_completion"], str)
        or not _is_hex64(value["wire_completion_sha256"])
        or not _is_hex64(value["canonical_completion_sha256"])
        or value["wire_completion_sha256"]
        != hashlib.sha256(value["wire_completion"].encode("utf-8")).hexdigest()
        or value["canonical_completion_sha256"]
        != hashlib.sha256(
            value["canonical_completion"].encode("utf-8")
        ).hexdigest()
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return {
        "segment_id": value["segment_id"],
        "wire_completion": value["wire_completion"],
        "wire_completion_sha256": value["wire_completion_sha256"],
        "canonical_completion": value["canonical_completion"],
        "canonical_completion_sha256": value["canonical_completion_sha256"],
        "leaf_receipt": _normalize_receipt_entry(
            value["leaf_receipt"], expected_schema=LEAF_RECEIPT_SCHEMA
        ),
    }


def _plain_int(value: Any, *, maximum: int = MAX_DIAGNOSTIC_COUNT) -> int:
    if type(value) is not int or isinstance(value, bool) or not 0 <= value <= maximum:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return value


def _normalize_coverage_row(value: Any) -> dict[str, Any]:
    keys = {
        "segment_id",
        "parent_sentence_id",
        "core_start_byte",
        "core_end_byte",
        "lexical_token_count",
        "chunk_index",
        "chunk_count",
        "leaf_eligible",
        "leaf_called",
        "disposition",
        "error_code",
        "relation_count",
    }
    if type(value) is not dict or set(value) != keys:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if (
        not isinstance(value["segment_id"], str)
        or not value["segment_id"]
        or not isinstance(value["parent_sentence_id"], str)
        or not value["parent_sentence_id"]
        or type(value["leaf_eligible"]) is not bool
        or type(value["leaf_called"]) is not bool
        or value["disposition"]
        not in {
            "EXTRACTED",
            "NO_RELATION",
            "CONTEXT_ONLY_SHORT_SENTENCE",
            "TYPED_FAILURE",
        }
        or (
            value["error_code"] is not None
            and not isinstance(value["error_code"], str)
        )
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    for key in (
        "core_start_byte",
        "core_end_byte",
        "lexical_token_count",
        "chunk_index",
        "chunk_count",
        "relation_count",
    ):
        _plain_int(value[key], maximum=1_000_000)
    if (
        value["core_end_byte"] <= value["core_start_byte"]
        or value["chunk_count"] < 1
        or value["chunk_index"] >= value["chunk_count"]
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return dict(value)


def _normalize_unit_row(value: Any) -> dict[str, Any]:
    keys = {
        "unit_id",
        "segment_id",
        "parent_sentence_id",
        "slot0_object_id",
        "slot1_object_id",
        "structural_relation_id",
        "anchor_span_id",
        "slot0_span_id",
        "slot1_span_id",
        "generator_kind",
        "polarity",
        "temporal_orientation",
        "causal_orientation",
        "relation_type",
        "semantic_signature_sha256",
        "evidence_binding_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    string_keys = keys - {"semantic_signature_sha256", "evidence_binding_sha256"}
    if any(not isinstance(value[key], str) or not value[key] for key in string_keys):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if not _is_hex64(value["semantic_signature_sha256"]) or not _is_hex64(
        value["evidence_binding_sha256"]
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return dict(value)


def _normalize_endpoint_binding(value: Any, *, side_ids: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "span_id",
        "selected_slot_id",
        "maximum_quantized_cosine",
        "tied_maximum_count",
        "score_vector_commitment",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if (
        not isinstance(value["span_id"], str)
        or not value["span_id"]
        or value["selected_slot_id"] not in side_ids | {None}
        or type(value["maximum_quantized_cosine"]) is not int
        or isinstance(value["maximum_quantized_cosine"], bool)
        or abs(value["maximum_quantized_cosine"]) > 1_000_000
        or type(value["tied_maximum_count"]) is not int
        or isinstance(value["tied_maximum_count"], bool)
        or not 1 <= value["tied_maximum_count"] <= len(side_ids)
        or not _is_hex64(value["score_vector_commitment"])
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return dict(value)


def _normalize_graph_slot(value: Any, *, side_ids: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "slot_id",
        "normalized_label_sha256",
        "evidence_binding_sha256",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if (
        value["slot_id"] not in side_ids
        or not _is_hex64(value["normalized_label_sha256"])
        or not _is_hex64(value["evidence_binding_sha256"])
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return dict(value)


def _normalize_graph_relation(value: Any, *, side_ids: set[str]) -> dict[str, Any]:
    keys = {
        "relation_id",
        "slot0_id",
        "slot1_id",
        "generator_kind",
        "polarity",
        "temporal_orientation",
        "causal_orientation",
        "evidence_binding_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if (
        not isinstance(value["relation_id"], str)
        or not value["relation_id"]
        or value["slot0_id"] not in side_ids
        or value["slot1_id"] not in side_ids
        or any(
            not isinstance(value[key], str) or not value[key]
            for key in (
                "generator_kind",
                "polarity",
                "temporal_orientation",
                "causal_orientation",
            )
        )
        or not _is_hex64(value["evidence_binding_sha256"])
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return dict(value)


def _normalize_side_receipts(value: Any, *, side_ids: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "document_envelope",
        "bounded_set",
        "binder",
        "slot_graph",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    document = value["document_envelope"]
    bounded = value["bounded_set"]
    binder = value["binder"]
    graph = value["slot_graph"]
    if type(document) is not dict or set(document) != {"receipt", "leaf_records"}:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if type(document["leaf_records"]) is not list:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    leaf_records = [_normalize_leaf_record(row) for row in document["leaf_records"]]
    if len({row["segment_id"] for row in leaf_records}) != len(leaf_records):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    normalized_document = {
        "receipt": _normalize_receipt_entry(
            document["receipt"], expected_schema=DOCUMENT_ENVELOPE_RECEIPT_SCHEMA
        ),
        "leaf_records": leaf_records,
    }
    if type(bounded) is not dict or set(bounded) != {
        "coverage",
        "units",
        "relation_set_signature_ascii",
        "relation_set_signature_sha256",
        "receipt",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if type(bounded["coverage"]) is not list or type(bounded["units"]) is not list:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    coverage = [_normalize_coverage_row(row) for row in bounded["coverage"]]
    units = [_normalize_unit_row(row) for row in bounded["units"]]
    signature = bounded["relation_set_signature_ascii"]
    signature_hash = bounded["relation_set_signature_sha256"]
    if (signature is None) != (signature_hash is None) or (
        signature is not None
        and (
            not isinstance(signature, str)
            or not _is_hex64(signature_hash)
            or hashlib.sha256(signature.encode("ascii")).hexdigest() != signature_hash
        )
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    normalized_bounded = {
        "coverage": coverage,
        "units": units,
        "relation_set_signature_ascii": signature,
        "relation_set_signature_sha256": signature_hash,
        "receipt": _normalize_receipt_entry(
            bounded["receipt"], expected_schema=BOUNDED_SET_RECEIPT_SCHEMA
        ),
    }
    if type(binder) is not dict or set(binder) != {"endpoint_bindings", "receipt"}:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if type(binder["endpoint_bindings"]) is not list:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    endpoint_bindings = [
        _normalize_endpoint_binding(row, side_ids=side_ids)
        for row in binder["endpoint_bindings"]
    ]
    if len({row["span_id"] for row in endpoint_bindings}) != len(endpoint_bindings):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    normalized_binder = {
        "endpoint_bindings": endpoint_bindings,
        "receipt": _normalize_receipt_entry(
            binder["receipt"], expected_schema=BINDER_RECEIPT_SCHEMA
        ),
    }
    if type(graph) is not dict or set(graph) != {
        "slots",
        "relations",
        "coverage_complete",
        "extractor_binding_sha256",
        "graph_evidence_binding_sha256",
        "receipt",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if (
        type(graph["slots"]) is not list
        or type(graph["relations"]) is not list
        or type(graph["coverage_complete"]) is not bool
        or graph["coverage_complete"] is not False
        or not _is_hex64(graph["extractor_binding_sha256"])
        or not _is_hex64(graph["graph_evidence_binding_sha256"])
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    slots = [_normalize_graph_slot(row, side_ids=side_ids) for row in graph["slots"]]
    relations = [
        _normalize_graph_relation(row, side_ids=side_ids) for row in graph["relations"]
    ]
    if {row["slot_id"] for row in slots} != side_ids:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    normalized_graph = {
        "slots": slots,
        "relations": relations,
        "coverage_complete": False,
        "extractor_binding_sha256": graph["extractor_binding_sha256"],
        "graph_evidence_binding_sha256": graph["graph_evidence_binding_sha256"],
        "receipt": _normalize_receipt_entry(
            graph["receipt"], expected_schema=SLOT_GRAPH_RECEIPT_SCHEMA
        ),
    }
    return {
        "document_envelope": normalized_document,
        "bounded_set": normalized_bounded,
        "binder": normalized_binder,
        "slot_graph": normalized_graph,
    }


def _normalize_mapping_receipts(value: Any, *, arity: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "assignment_subproblems_solved",
        "choices",
        "proposals",
        "receipt",
        "target_color_shuffle_effective",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    solved = _plain_int(value["assignment_subproblems_solved"], maximum=100_000)
    if (
        type(value["choices"]) is not list
        or type(value["proposals"]) is not list
        or type(value["target_color_shuffle_effective"]) is not bool
        or len(value["choices"]) != len(MAPPING_ARM_IDS)
        or len(value["proposals"]) > MAX_PROPOSALS_PER_POOL
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    proposals: list[dict[str, Any]] = []
    proposal_index: dict[str, dict[str, Any]] = {}
    proposal_keys = {
        "flat_structural_score",
        "injective_verified",
        "length2_composition_verified",
        "length2_path_matched",
        "length2_path_total",
        "operator_id",
        "origins",
        "semantic_score",
        "target_indices",
        "typed_incidence_matched",
        "typed_incidence_total",
        "typed_incidence_verified",
        "proposal_hash",
    }
    for proposal in value["proposals"]:
        if type(proposal) is not dict or set(proposal) != proposal_keys:
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        body = dict(proposal)
        proposal_hash = body.pop("proposal_hash")
        if (
            not _is_hex64(proposal_hash)
            or proposal_hash != _content_hash(body)
            or proposal_hash in proposal_index
            or proposal["operator_id"] not in OPERATOR_IDS
            or type(proposal["origins"]) is not list
            or proposal["origins"] != sorted(set(proposal["origins"]))
            or not set(proposal["origins"]).issubset(
                {"semantic_kbest", "structure_kbest"}
            )
            or not proposal["origins"]
            or type(proposal["target_indices"]) is not list
            or sorted(proposal["target_indices"]) != list(range(arity))
            or any(
                type(proposal[key]) is not bool
                for key in (
                    "injective_verified",
                    "typed_incidence_verified",
                    "length2_composition_verified",
                )
            )
            or proposal["injective_verified"] is not True
        ):
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        for key in (
            "flat_structural_score",
            "semantic_score",
            "length2_path_matched",
            "length2_path_total",
            "typed_incidence_matched",
            "typed_incidence_total",
        ):
            if type(proposal[key]) is not int or isinstance(proposal[key], bool):
                raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        if (
            abs(proposal["flat_structural_score"]) > MAX_SCORE_ABS
            or abs(proposal["semantic_score"]) > MAX_SCORE_ABS
            or not 0
            <= proposal["length2_path_matched"]
            <= proposal["length2_path_total"]
            <= MAX_DIAGNOSTIC_COUNT
            or not 0
            <= proposal["typed_incidence_matched"]
            <= proposal["typed_incidence_total"]
            <= MAX_DIAGNOSTIC_COUNT
        ):
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        canonical = dict(proposal)
        proposals.append(canonical)
        proposal_index[proposal_hash] = canonical
    choices: list[dict[str, Any]] = []
    for expected_arm, choice in zip(MAPPING_ARM_IDS, value["choices"], strict=True):
        if type(choice) is not dict or set(choice) != {
            "arm",
            "disposition",
            "proposal_hash",
            "reason_ids",
        }:
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        if (
            choice["arm"] != expected_arm
            or choice["disposition"] not in {"SELECTED", "ABSTAIN"}
            or type(choice["reason_ids"]) is not list
            or any(
                not isinstance(reason, str) or not reason
                for reason in choice["reason_ids"]
            )
            or (
                choice["disposition"] == "SELECTED"
                and choice["proposal_hash"] not in proposal_index
            )
            or (
                choice["disposition"] == "ABSTAIN"
                and choice["proposal_hash"] is not None
            )
        ):
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        choices.append(dict(choice))
    return {
        "assignment_subproblems_solved": solved,
        "choices": choices,
        "proposals": proposals,
        "receipt": _normalize_receipt_entry(
            value["receipt"], expected_schema=MAPPING_RECEIPT_SCHEMA
        ),
        "target_color_shuffle_effective": value[
            "target_color_shuffle_effective"
        ],
    }


def _normalize_private_mechanism_receipts(
    value: Any,
    *,
    action_item: Mapping[str, Any],
    normalized_variants: Mapping[str, Any],
    normalized_diagnostics: Mapping[str, Any],
    normalized_execution: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "availability",
        "error_code",
        "semantic_matrix",
        "sides",
        "variants",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if type(value["sides"]) is not dict or tuple(value["sides"]) != ("left", "right"):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if type(value["variants"]) is not dict or tuple(value["variants"]) != VARIANT_NAMES:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    if value["availability"] == "PREMODEL_TYPED_FAILURE":
        expected = {
            "availability": "PREMODEL_TYPED_FAILURE",
            "error_code": "SLOT_BINDER_TYPED_FAILURE",
            "semantic_matrix": None,
            "sides": {"left": None, "right": None},
            "variants": {"base": None, "system_swap": None},
        }
        if value != expected or normalized_execution != {
            "structural_status": "TYPED_FAILURE",
            "error_code": "SLOT_BINDER_TYPED_FAILURE",
            "document_call_count": 0,
        }:
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        return expected
    if (
        value["availability"] != "COMPLETE"
        or value["error_code"] is not None
        or normalized_execution
        != {
            "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
            "error_code": None,
            "document_call_count": 2,
        }
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    matrix = value["semantic_matrix"]
    if type(matrix) is not dict or set(matrix) != {"receipt", "rows"}:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    left_ids, right_ids = _side_slot_sets(action_item, "base")
    if type(matrix["rows"]) is not list or len(matrix["rows"]) != len(
        left_ids
    ) * len(right_ids):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    matrix_rows: list[list[Any]] = []
    seen_matrix: set[tuple[str, str]] = set()
    for row in matrix["rows"]:
        if (
            type(row) is not list
            or len(row) != 3
            or row[0] not in left_ids
            or row[1] not in right_ids
            or type(row[2]) is not int
            or isinstance(row[2], bool)
            or abs(row[2]) > 1_000_000
            or (row[0], row[1]) in seen_matrix
        ):
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        seen_matrix.add((row[0], row[1]))
        matrix_rows.append(list(row))
    if seen_matrix != {(left, right) for left in left_ids for right in right_ids}:
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    matrix_entry = _normalize_receipt_entry(
        matrix["receipt"], expected_schema=SEMANTIC_MATRIX_RECEIPT_SCHEMA
    )
    if matrix_entry["receipt"].get("matrix_commitment") != _content_hash(matrix_rows):
        raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    sides = {
        "left": _normalize_side_receipts(value["sides"]["left"], side_ids=left_ids),
        "right": _normalize_side_receipts(
            value["sides"]["right"], side_ids=right_ids
        ),
    }
    normalized_mapping_variants: dict[str, Any] = {}
    for variant_name in VARIANT_NAMES:
        variant = value["variants"][variant_name]
        if type(variant) is not dict or set(variant) != {
            "semantic_mapping",
            "structural_mapping",
            "target_color_shuffle_mapping",
        }:
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        normalized_mapping_variants[variant_name] = {
            name: _normalize_mapping_receipts(variant[name], arity=len(left_ids))
            for name in (
                "semantic_mapping",
                "structural_mapping",
                "target_color_shuffle_mapping",
            )
        }
        diagnostic = normalized_diagnostics[variant_name]
        side_left = sides["left" if variant_name == "base" else "right"]
        side_right = sides["right" if variant_name == "base" else "left"]
        if (
            diagnostic["left_graph_receipt_sha256"]
            != side_left["slot_graph"]["receipt"]["receipt_sha256"]
            or diagnostic["right_graph_receipt_sha256"]
            != side_right["slot_graph"]["receipt"]["receipt_sha256"]
        ):
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
        mapping_hashes = diagnostic["mapping_receipt_sha256_by_arm"]
        normalized_maps = normalized_mapping_variants[variant_name]
        if (
            mapping_hashes["semantic_only"]
            != normalized_maps["semantic_mapping"]["receipt"]["receipt_sha256"]
            or any(
                mapping_hashes[arm_id]
                != normalized_maps["structural_mapping"]["receipt"][
                    "receipt_sha256"
                ]
                for arm_id in (
                    "flat_structural",
                    "full_no_composition",
                    "full_with_length2_composition",
                )
            )
            or mapping_hashes[
                "full_with_length2_composition_target_color_shuffle"
            ]
            != normalized_maps["target_color_shuffle_mapping"]["receipt"][
                "receipt_sha256"
            ]
            or diagnostic["target_color_shuffle_effective"]
            is not normalized_maps["target_color_shuffle_mapping"][
                "target_color_shuffle_effective"
            ]
        ):
            raise ScarCssmScoreError("SCAR_SCORE_PRIVATE_RECEIPT_INVALID")
    return {
        "availability": "COMPLETE",
        "error_code": None,
        "semantic_matrix": {"receipt": matrix_entry, "rows": matrix_rows},
        "sides": sides,
        "variants": normalized_mapping_variants,
    }


def _normalize_execution(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "structural_status",
        "error_code",
        "document_call_count",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_EXECUTION_INVALID")
    status = value["structural_status"]
    count = value["document_call_count"]
    if (
        status not in STRUCTURAL_STATUSES
        or type(count) is not int
        or isinstance(count, bool)
        or count not in {0, 1, 2}
    ):
        raise ScarCssmScoreError("SCAR_SCORE_EXECUTION_INVALID")
    if status == "EXECUTED_WITHOUT_TYPED_FAILURE":
        if value["error_code"] is not None or count != 2:
            raise ScarCssmScoreError("SCAR_SCORE_EXECUTION_INVALID")
    elif value["error_code"] not in ERROR_CODES:
        raise ScarCssmScoreError("SCAR_SCORE_EXECUTION_INVALID")
    return {
        "structural_status": status,
        "error_code": value["error_code"],
        "document_call_count": count,
    }


def _diagnostic_count(value: Any) -> int:
    if (
        type(value) is not int
        or isinstance(value, bool)
        or not 0 <= value <= MAX_DIAGNOSTIC_COUNT
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    return value


def _normalize_binder_diagnostic(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "coverage_disposition",
        "unbound_count",
        "dropped_edge_count",
        "retained_edge_count",
        "zero_degree_count",
        "endpoint_count",
        "self_loop_count",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    if value["coverage_disposition"] not in BINDER_COVERAGE_DISPOSITIONS:
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    normalized = {
        "coverage_disposition": value["coverage_disposition"],
        "unbound_count": _diagnostic_count(value["unbound_count"]),
        "dropped_edge_count": _diagnostic_count(value["dropped_edge_count"]),
        "retained_edge_count": _diagnostic_count(value["retained_edge_count"]),
        "zero_degree_count": _diagnostic_count(value["zero_degree_count"]),
        "endpoint_count": _diagnostic_count(value["endpoint_count"]),
        "self_loop_count": _diagnostic_count(value["self_loop_count"]),
    }
    if (
        normalized["unbound_count"] > normalized["endpoint_count"]
        or normalized["self_loop_count"] > normalized["retained_edge_count"]
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    return normalized


def _empty_arm_diagnostic() -> dict[str, Any]:
    return {
        "selected_operator": None,
        "semantic_origin_count": 0,
        "structural_origin_count": 0,
        "incidence_match_count": 0,
        "incidence_total_count": 0,
        "length2_path_count": 0,
        "length2_path_total_count": 0,
        "typed_incidence_verified": False,
        "length2_composition_verified": False,
        "proposal_hash": None,
        "semantic_score": None,
        "flat_structural_score": None,
    }


def _normalize_arm_diagnostic(
    value: Any, *, prediction: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "selected_operator",
        "semantic_origin_count",
        "structural_origin_count",
        "incidence_match_count",
        "incidence_total_count",
        "length2_path_count",
        "length2_path_total_count",
        "typed_incidence_verified",
        "length2_composition_verified",
        "proposal_hash",
        "semantic_score",
        "flat_structural_score",
    }
    if type(value) is not dict or set(value) != expected:
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    counts = {
        key: _diagnostic_count(value[key])
        for key in (
            "semantic_origin_count",
            "structural_origin_count",
            "incidence_match_count",
            "incidence_total_count",
            "length2_path_count",
            "length2_path_total_count",
        )
    }
    disposition = prediction["disposition"]
    if disposition != "ANSWER":
        if (
            value["selected_operator"] is not None
            or value["proposal_hash"] is not None
            or value["semantic_score"] is not None
            or value["flat_structural_score"] is not None
            or type(value["typed_incidence_verified"]) is not bool
            or value["typed_incidence_verified"]
            or type(value["length2_composition_verified"]) is not bool
            or value["length2_composition_verified"]
            or any(counts.values())
        ):
            raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
        return _empty_arm_diagnostic()
    if (
        value["selected_operator"] not in OPERATOR_IDS
        or not _is_hex64(value["proposal_hash"])
        or type(value["semantic_score"]) is not int
        or isinstance(value["semantic_score"], bool)
        or abs(value["semantic_score"]) > MAX_SCORE_ABS
        or type(value["flat_structural_score"]) is not int
        or isinstance(value["flat_structural_score"], bool)
        or abs(value["flat_structural_score"]) > MAX_SCORE_ABS
        or type(value["typed_incidence_verified"]) is not bool
        or type(value["length2_composition_verified"]) is not bool
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    if counts["semantic_origin_count"] not in {0, 1} or counts[
        "structural_origin_count"
    ] not in {0, 1}:
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    if not (
        counts["semantic_origin_count"] or counts["structural_origin_count"]
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    if (
        counts["incidence_match_count"] > counts["incidence_total_count"]
        or counts["length2_path_count"] > counts["length2_path_total_count"]
        or (
            value["typed_incidence_verified"]
            and (
                counts["incidence_total_count"] == 0
                or counts["incidence_match_count"]
                != counts["incidence_total_count"]
            )
        )
        or (
            value["length2_composition_verified"]
            and (
                not value["typed_incidence_verified"]
                or counts["length2_path_total_count"] == 0
                or counts["length2_path_count"]
                != counts["length2_path_total_count"]
            )
        )
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    return {
        "selected_operator": value["selected_operator"],
        **counts,
        "typed_incidence_verified": value["typed_incidence_verified"],
        "length2_composition_verified": value[
            "length2_composition_verified"
        ],
        "proposal_hash": value["proposal_hash"],
        "semantic_score": value["semantic_score"],
        "flat_structural_score": value["flat_structural_score"],
    }


def _normalize_variant_diagnostic(
    value: Any, *, variant_predictions: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "structural_diagnostics_available",
        "target_color_shuffle_effective",
        "left_binder",
        "right_binder",
        "left_graph_receipt_sha256",
        "right_graph_receipt_sha256",
        "mapping_receipt_sha256_by_arm",
        "arms",
    }
    if type(value) is not dict or set(value) != expected:
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    available = value["structural_diagnostics_available"]
    if type(available) is not bool:
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    mapping_hashes = value["mapping_receipt_sha256_by_arm"]
    arms = value["arms"]
    if (
        type(mapping_hashes) is not dict
        or tuple(mapping_hashes) != ARM_IDS
        or type(arms) is not dict
        or tuple(arms) != ARM_IDS
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    if not available:
        if (
            value["target_color_shuffle_effective"] is not None
            or value["left_binder"] is not None
            or value["right_binder"] is not None
            or value["left_graph_receipt_sha256"] is not None
            or value["right_graph_receipt_sha256"] is not None
            or any(mapping_hashes[arm_id] is not None for arm_id in ARM_IDS)
        ):
            raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
        if any(arms[arm_id] != _empty_arm_diagnostic() for arm_id in ARM_IDS):
            raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
        normalized_arms = {arm_id: _empty_arm_diagnostic() for arm_id in ARM_IDS}
        return {
            "structural_diagnostics_available": False,
            "target_color_shuffle_effective": None,
            "left_binder": None,
            "right_binder": None,
            "left_graph_receipt_sha256": None,
            "right_graph_receipt_sha256": None,
            "mapping_receipt_sha256_by_arm": {
                arm_id: None for arm_id in ARM_IDS
            },
            "arms": normalized_arms,
        }
    if (
        type(value["target_color_shuffle_effective"]) is not bool
        or not _is_hex64(value["left_graph_receipt_sha256"])
        or not _is_hex64(value["right_graph_receipt_sha256"])
        or any(not _is_hex64(mapping_hashes[arm_id]) for arm_id in ARM_IDS)
        or mapping_hashes["flat_structural"]
        != mapping_hashes["full_no_composition"]
        or mapping_hashes["flat_structural"]
        != mapping_hashes["full_with_length2_composition"]
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    normalized_arms = {
        arm_id: _normalize_arm_diagnostic(
            arms[arm_id], prediction=variant_predictions[arm_id]
        )
        for arm_id in ARM_IDS
    }
    semantic = normalized_arms["semantic_only"]
    if variant_predictions["semantic_only"]["disposition"] == "ANSWER" and (
        semantic["selected_operator"]
        != "ori_keep.pol_keep.slots_identity"
        or semantic["semantic_origin_count"] != 1
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    flat = normalized_arms["flat_structural"]
    no_composition = normalized_arms["full_no_composition"]
    full = normalized_arms["full_with_length2_composition"]
    shuffled = normalized_arms[
        "full_with_length2_composition_target_color_shuffle"
    ]
    if (
        (
            variant_predictions["flat_structural"]["disposition"] == "ANSWER"
            and flat["incidence_total_count"] == 0
        )
        or (
            variant_predictions["full_no_composition"]["disposition"]
            == "ANSWER"
            and not no_composition["typed_incidence_verified"]
        )
        or (
            variant_predictions["full_with_length2_composition"][
                "disposition"
            ]
            == "ANSWER"
            and not full["length2_composition_verified"]
        )
        or (
            variant_predictions[
                "full_with_length2_composition_target_color_shuffle"
            ]["disposition"]
            == "ANSWER"
            and not shuffled["length2_composition_verified"]
        )
    ):
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    return {
        "structural_diagnostics_available": True,
        "target_color_shuffle_effective": value[
            "target_color_shuffle_effective"
        ],
        "left_binder": _normalize_binder_diagnostic(value["left_binder"]),
        "right_binder": _normalize_binder_diagnostic(value["right_binder"]),
        "left_graph_receipt_sha256": value["left_graph_receipt_sha256"],
        "right_graph_receipt_sha256": value["right_graph_receipt_sha256"],
        "mapping_receipt_sha256_by_arm": {
            arm_id: mapping_hashes[arm_id] for arm_id in ARM_IDS
        },
        "arms": normalized_arms,
    }


def _normalize_diagnostics(
    value: Any, *, normalized_variants: Mapping[str, Any]
) -> dict[str, Any]:
    if type(value) is not dict or tuple(value) != VARIANT_NAMES:
        raise ScarCssmScoreError("SCAR_SCORE_DIAGNOSTICS_INVALID")
    return {
        variant_name: _normalize_variant_diagnostic(
            value[variant_name],
            variant_predictions=normalized_variants[variant_name]["arms"],
        )
        for variant_name in VARIANT_NAMES
    }


def _normalize_prediction_item(
    value: Any, *, action_item: Mapping[str, Any]
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "item_token",
        "variants",
        "proposal_pools",
        "execution",
        "diagnostics",
        "private_mechanism_receipts",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ITEM_INVALID")
    if value["item_token"] != action_item["item_token"]:
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ITEM_INVALID")
    variants = value["variants"]
    pools = value["proposal_pools"]
    if (
        type(variants) is not dict
        or tuple(variants) != VARIANT_NAMES
        or type(pools) is not dict
        or tuple(pools) != VARIANT_NAMES
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_VARIANTS_INVALID")
    normalized_variants: dict[str, Any] = {}
    normalized_pools: dict[str, Any] = {}
    for variant_name in VARIANT_NAMES:
        left_ids, right_ids = _side_slot_sets(action_item, variant_name)
        variant = variants[variant_name]
        if type(variant) is not dict or set(variant) != {"arms"}:
            raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_VARIANTS_INVALID")
        arms = variant["arms"]
        if type(arms) is not dict or tuple(arms) != ARM_IDS:
            raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ARMS_INVALID")
        normalized_variants[variant_name] = {
            "arms": {
                arm_id: _normalize_disposition(
                    arms[arm_id], left_ids=left_ids, right_ids=right_ids
                )
                for arm_id in ARM_IDS
            }
        }
        pool = pools[variant_name]
        if type(pool) is not dict or set(pool) != {
            "semantic_kbest",
            "structure_kbest",
        }:
            raise ScarCssmScoreError("SCAR_SCORE_PROPOSAL_POOL_INVALID")
        normalized_pools[variant_name] = {
            "semantic_kbest": _normalize_pool(
                pool["semantic_kbest"], left_ids=left_ids, right_ids=right_ids
            ),
            "structure_kbest": _normalize_pool(
                pool["structure_kbest"], left_ids=left_ids, right_ids=right_ids
            ),
        }
    normalized_execution = _normalize_execution(value["execution"])
    normalized_diagnostics = _normalize_diagnostics(
        value["diagnostics"], normalized_variants=normalized_variants
    )
    normalized_private_receipts = _normalize_private_mechanism_receipts(
        value["private_mechanism_receipts"],
        action_item=action_item,
        normalized_variants=normalized_variants,
        normalized_diagnostics=normalized_diagnostics,
        normalized_execution=normalized_execution,
    )
    return {
        "item_token": value["item_token"],
        "variants": normalized_variants,
        "proposal_pools": normalized_pools,
        "execution": normalized_execution,
        "diagnostics": normalized_diagnostics,
        "private_mechanism_receipts": normalized_private_receipts,
    }


def _seal_prediction_pack(
    action_pack: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
    secret: bytes,
    study_id: str,
    expected_case_count: int,
) -> dict[str, Any]:
    _require_secret_and_study(secret, study_id)
    action_index = _index_action_pack(
        action_pack, expected_case_count=expected_case_count
    )
    if action_pack.get("study_id") != study_id:
        raise ScarCssmScoreError("SCAR_SCORE_ACTION_STUDY_INVALID")
    # The action pack exposes only the label commitment, not label contents.
    # Recomputing the compiler's secret cross-binding authenticates the action
    # capability while preserving the pre-label-open boundary.
    expected_cross_binding = source._pack_binding_hmac(  # noqa: SLF001
        secret,
        study_id=study_id,
        action_commitment=action_pack["action_commitment_sha256"],
        label_commitment=action_pack["label_commitment_sha256"],
    )
    if not hmac.compare_digest(
        action_pack["cross_binding_hmac_sha256"], expected_cross_binding
    ):
        raise ScarCssmScoreError("SCAR_SCORE_ACTION_SECRET_BINDING_INVALID")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ITEMS_INVALID")
    supplied: dict[str, Mapping[str, Any]] = {}
    for value in items:
        if type(value) is not dict:
            raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ITEM_INVALID")
        token = value.get("item_token")
        if token not in action_index or token in supplied:
            raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_ITEM_INVALID")
        supplied[token] = value
    if set(supplied) != set(action_index):
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_COVERAGE_INVALID")
    normalized = [
        _normalize_prediction_item(supplied[token], action_item=action_index[token])
        for token in sorted(action_index)
    ]
    core = {
        "schema": PREDICTION_PACK_SCHEMA,
        "study_id": study_id,
        "source_action_commitment_sha256": action_pack[
            "action_commitment_sha256"
        ],
        "arm_ids": list(ARM_IDS),
        "variant_names": list(VARIANT_NAMES),
        "items": normalized,
    }
    return {**core, "self_sha256": _content_hash(core)}


def seal_scar_cssm_prediction_pack_v1(
    action_pack: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
    secret: bytes,
    study_id: str,
) -> dict[str, Any]:
    """Seal the exact official 391-case prediction pack before label access."""

    try:
        source.validate_scar_cssm_action_pack_v1(
            action_pack, study_id=study_id
        )
    except source.ScarCssmSourceError as exc:
        raise ScarCssmScoreError(
            f"SCAR_SCORE_ACTION_PACK_INVALID__{exc.issue_id}"
        ) from exc
    return _seal_prediction_pack(
        action_pack,
        items=items,
        secret=secret,
        study_id=study_id,
        expected_case_count=source.EXPECTED_ACTION_ITEM_COUNT,
    )


def _seal_scar_cssm_prediction_pack_for_test_v1(
    action_pack: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
    secret: bytes,
    study_id: str,
    expected_case_count: int,
) -> dict[str, Any]:
    """Internal tiny-fixture constructor; never confers official authority."""

    return _seal_prediction_pack(
        action_pack,
        items=items,
        secret=secret,
        study_id=study_id,
        expected_case_count=expected_case_count,
    )


def _validate_prediction_pack(
    prediction_pack: Mapping[str, Any],
    *,
    action_pack: Mapping[str, Any],
    action_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if type(prediction_pack) is not dict or set(prediction_pack) != {
        "schema",
        "study_id",
        "source_action_commitment_sha256",
        "arm_ids",
        "variant_names",
        "items",
        "self_sha256",
    }:
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_PACK_INVALID")
    body = dict(prediction_pack)
    claimed = body.pop("self_sha256")
    if (
        prediction_pack["schema"] != PREDICTION_PACK_SCHEMA
        or prediction_pack["study_id"] != action_pack["study_id"]
        or prediction_pack["source_action_commitment_sha256"]
        != action_pack["action_commitment_sha256"]
        or prediction_pack["arm_ids"] != list(ARM_IDS)
        or prediction_pack["variant_names"] != list(VARIANT_NAMES)
        or not _is_hex64(claimed)
        or not hmac.compare_digest(claimed, _content_hash(body))
        or type(prediction_pack["items"]) is not list
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_PACK_INVALID")
    expected_tokens = sorted(action_index)
    if [row.get("item_token") for row in prediction_pack["items"]] != expected_tokens:
        raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_COVERAGE_INVALID")
    indexed: dict[str, dict[str, Any]] = {}
    for row, token in zip(prediction_pack["items"], expected_tokens, strict=True):
        normalized = _normalize_prediction_item(
            row, action_item=action_index[token]
        )
        if normalized != row:
            raise ScarCssmScoreError("SCAR_SCORE_PREDICTION_NOT_CANONICAL")
        indexed[token] = normalized
    return indexed


def _pair_key(value: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((pair[0], pair[1]) for pair in value))


def _variant_arm_metric(
    prediction: Mapping[str, Any], gold_pairs: Sequence[Sequence[str]]
) -> dict[str, Any]:
    gold = set(_pair_key(gold_pairs))
    answered = prediction["disposition"] == "ANSWER"
    proposed = set(_pair_key(prediction["pairs"])) if answered else set()
    correct = len(gold & proposed)
    false_positive = len(proposed - gold)
    false_negative = len(gold - proposed)
    denominator = 2 * correct + false_positive + false_negative
    f1 = (2 * correct / denominator) if denominator else 0.0
    return {
        "disposition": prediction["disposition"],
        "correct_pair_count": correct,
        "reference_pair_count": len(gold),
        "pair_f1": f1,
        "strict_exact": answered and proposed == gold,
        "answered": answered,
    }


def _proposal_metric(
    pools: Mapping[str, Any], gold_pairs: Sequence[Sequence[str]]
) -> dict[str, Any]:
    gold = _pair_key(gold_pairs)
    semantic = {_pair_key(candidate) for candidate in pools["semantic_kbest"]}
    structure = {_pair_key(candidate) for candidate in pools["structure_kbest"]}
    semantic_hit = gold in semantic
    structure_hit = gold in structure
    return {
        "semantic_pool_size": len(semantic),
        "structure_pool_size": len(structure),
        "semantic_exact_mapping_hit": semantic_hit,
        "structure_exact_mapping_hit": structure_hit,
        "structure_only_added_exact_mapping_hit": structure_hit
        and not semantic_hit,
    }


def _private_receipt_counts(bundle: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "semantic_matrix": 0,
        "document_envelope": 0,
        "leaf": 0,
        "bounded_set": 0,
        "binder": 0,
        "slot_graph": 0,
        "mapping": 0,
    }
    if bundle["availability"] == "COMPLETE":
        counts["semantic_matrix"] = 1
        for side_name in ("left", "right"):
            side = bundle["sides"][side_name]
            counts["document_envelope"] += 1
            counts["leaf"] += len(side["document_envelope"]["leaf_records"])
            counts["bounded_set"] += 1
            counts["binder"] += 1
            counts["slot_graph"] += 1
        counts["mapping"] = len(VARIANT_NAMES) * 3
    counts["total"] = sum(counts.values())
    return counts


def _score_per_case(
    *,
    action_index: Mapping[str, Mapping[str, Any]],
    label_index: Mapping[str, Mapping[str, Any]],
    prediction_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token in sorted(action_index):
        label = label_index[token]
        prediction = prediction_index[token]
        variants: dict[str, Any] = {}
        proposal_metrics: dict[str, Any] = {}
        for variant_name in VARIANT_NAMES:
            gold_pairs = label["gold_pairs"][variant_name]
            variants[variant_name] = {
                arm_id: _variant_arm_metric(
                    prediction["variants"][variant_name]["arms"][arm_id],
                    gold_pairs,
                )
                for arm_id in ARM_IDS
            }
            proposal_metrics[variant_name] = _proposal_metric(
                prediction["proposal_pools"][variant_name], gold_pairs
            )
        arm_case_metrics: dict[str, Any] = {}
        for arm_id in ARM_IDS:
            base_metric = variants["base"][arm_id]
            swap_metric = variants["system_swap"][arm_id]
            both_answered = base_metric["answered"] and swap_metric["answered"]
            if both_answered:
                base_pairs = prediction["variants"]["base"]["arms"][arm_id][
                    "pairs"
                ]
                swap_pairs = prediction["variants"]["system_swap"]["arms"][
                    arm_id
                ]["pairs"]
                consistent: bool | None = _pair_key(swap_pairs) == tuple(
                    sorted((right, left) for left, right in base_pairs)
                )
            else:
                consistent = None
            arm_case_metrics[arm_id] = {
                "mean_variant_pair_f1": (
                    base_metric["pair_f1"] + swap_metric["pair_f1"]
                )
                / 2,
                "both_variants_strict_exact": base_metric["strict_exact"]
                and swap_metric["strict_exact"],
                "both_variants_answered": both_answered,
                "base_swap_consistent": consistent,
            }
        rows.append(
            {
                "item_token": token,
                "strata": dict(label["strata"]),
                "execution": dict(prediction["execution"]),
                "diagnostics": prediction["diagnostics"],
                "variants": variants,
                "arm_case_metrics": arm_case_metrics,
                "proposal_metrics": proposal_metrics,
                "private_mechanism_receipt_availability": prediction[
                    "private_mechanism_receipts"
                ]["availability"],
                "private_mechanism_receipt_counts": _private_receipt_counts(
                    prediction["private_mechanism_receipts"]
                ),
                "private_mechanism_bundle_sha256": _content_hash(
                    prediction["private_mechanism_receipts"]
                ),
            }
        )
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _arm_summary(rows: Sequence[Mapping[str, Any]], arm_id: str) -> dict[str, Any]:
    correct_pairs = 0
    reference_pairs = 0
    variant_f1_sum = 0.0
    strict_variants = 0
    answered_variants = 0
    strict_cases = 0
    answered_cases = 0
    swap_consistent = 0
    for row in rows:
        for variant_name in VARIANT_NAMES:
            metric = row["variants"][variant_name][arm_id]
            correct_pairs += metric["correct_pair_count"]
            reference_pairs += metric["reference_pair_count"]
            variant_f1_sum += metric["pair_f1"]
            strict_variants += metric["strict_exact"]
            answered_variants += metric["answered"]
        case_metric = row["arm_case_metrics"][arm_id]
        strict_cases += case_metric["both_variants_strict_exact"]
        answered_cases += case_metric["both_variants_answered"]
        swap_consistent += case_metric["base_swap_consistent"] is True
    case_count = len(rows)
    variant_count = case_count * len(VARIANT_NAMES)
    return {
        "case_count": case_count,
        "variant_count": variant_count,
        "pair_micro_accuracy": _ratio(correct_pairs, reference_pairs),
        "item_macro_pair_f1": _ratio(
            sum(
                row["arm_case_metrics"][arm_id]["mean_variant_pair_f1"]
                for row in rows
            ),
            case_count,
        ),
        "strict_exact_rate": _ratio(strict_cases, case_count),
        "variant_strict_exact_rate": _ratio(strict_variants, variant_count),
        "answer_coverage": _ratio(answered_variants, variant_count),
        "both_variant_answer_coverage": _ratio(answered_cases, case_count),
        "base_swap_consistency": (
            _ratio(swap_consistent, answered_cases) if answered_cases else None
        ),
        "base_swap_consistency_coverage": _ratio(answered_cases, case_count),
    }


def _mechanism_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variant_count = len(rows) * len(VARIANT_NAMES)
    semantic_hits = structure_hits = added_hits = 0
    for row in rows:
        for variant_name in VARIANT_NAMES:
            metric = row["proposal_metrics"][variant_name]
            semantic_hits += metric["semantic_exact_mapping_hit"]
            structure_hits += metric["structure_exact_mapping_hit"]
            added_hits += metric["structure_only_added_exact_mapping_hit"]
    return {
        "variant_count": variant_count,
        "semantic_pool_complete_reference_mapping_recall": _ratio(
            semantic_hits, variant_count
        ),
        "structure_pool_complete_reference_mapping_recall": _ratio(
            structure_hits, variant_count
        ),
        "structure_only_added_pool_complete_reference_mapping_recall": _ratio(
            added_hits, variant_count
        ),
    }


def _diagnostic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variant_count = len(rows) * len(VARIANT_NAMES)
    available = 0
    shuffle_effective = 0
    binder_count = 0
    binder_totals = {
        "unbound_count": 0,
        "dropped_edge_count": 0,
        "retained_edge_count": 0,
        "zero_degree_count": 0,
        "endpoint_count": 0,
        "self_loop_count": 0,
    }
    coverage_counts = {
        value: 0 for value in sorted(BINDER_COVERAGE_DISPOSITIONS)
    }
    arm_totals = {
        arm_id: {
            "selected_count": 0,
            "semantic_origin_count": 0,
            "structural_origin_count": 0,
            "incidence_match_count": 0,
            "incidence_total_count": 0,
            "length2_path_count": 0,
            "length2_path_total_count": 0,
            "typed_incidence_verified_count": 0,
            "length2_composition_verified_count": 0,
            "selected_operator_counts": {
                operator_id: 0 for operator_id in sorted(OPERATOR_IDS)
            },
        }
        for arm_id in ARM_IDS
    }
    for row in rows:
        for variant_name in VARIANT_NAMES:
            diagnostic = row["diagnostics"][variant_name]
            if not diagnostic["structural_diagnostics_available"]:
                continue
            available += 1
            shuffle_effective += diagnostic["target_color_shuffle_effective"]
            for side_name in ("left_binder", "right_binder"):
                binder = diagnostic[side_name]
                binder_count += 1
                coverage_counts[binder["coverage_disposition"]] += 1
                for key in binder_totals:
                    binder_totals[key] += binder[key]
            for arm_id in ARM_IDS:
                arm = diagnostic["arms"][arm_id]
                if arm["selected_operator"] is None:
                    continue
                totals = arm_totals[arm_id]
                totals["selected_count"] += 1
                totals["selected_operator_counts"][arm["selected_operator"]] += 1
                for key in (
                    "semantic_origin_count",
                    "structural_origin_count",
                    "incidence_match_count",
                    "incidence_total_count",
                    "length2_path_count",
                    "length2_path_total_count",
                ):
                    totals[key] += arm[key]
                totals["typed_incidence_verified_count"] += arm[
                    "typed_incidence_verified"
                ]
                totals["length2_composition_verified_count"] += arm[
                    "length2_composition_verified"
                ]
    arm_summary: dict[str, Any] = {}
    for arm_id, totals in arm_totals.items():
        selected = totals["selected_count"]
        arm_summary[arm_id] = {
            "selected_diagnostic_count": selected,
            "selected_diagnostic_coverage": _ratio(selected, variant_count),
            "semantic_origin_rate_among_selected": _ratio(
                totals["semantic_origin_count"], selected
            ),
            "structural_origin_rate_among_selected": _ratio(
                totals["structural_origin_count"], selected
            ),
            "mean_incidence_match_count_among_selected": _ratio(
                totals["incidence_match_count"], selected
            ),
            "incidence_match_rate": _ratio(
                totals["incidence_match_count"], totals["incidence_total_count"]
            ),
            "mean_length2_path_count_among_selected": _ratio(
                totals["length2_path_count"], selected
            ),
            "length2_path_match_rate": _ratio(
                totals["length2_path_count"],
                totals["length2_path_total_count"],
            ),
            "typed_incidence_verified_rate_among_selected": _ratio(
                totals["typed_incidence_verified_count"], selected
            ),
            "length2_composition_verified_rate_among_selected": _ratio(
                totals["length2_composition_verified_count"], selected
            ),
            "selected_operator_counts": totals["selected_operator_counts"],
        }
    return {
        "variant_count": variant_count,
        "structural_diagnostics_available_count": available,
        "structural_diagnostics_available_rate": _ratio(
            available, variant_count
        ),
        "target_color_shuffle_effective_count": shuffle_effective,
        "target_color_shuffle_effective_rate_among_available": _ratio(
            shuffle_effective, available
        ),
        "binder_side_diagnostic_count": binder_count,
        "binder_coverage_disposition_counts": coverage_counts,
        "binder_totals": binder_totals,
        "binder_retained_edge_rate": _ratio(
            binder_totals["retained_edge_count"],
            binder_totals["retained_edge_count"]
            + binder_totals["dropped_edge_count"],
        ),
        "binder_unbound_endpoint_rate": _ratio(
            binder_totals["unbound_count"], binder_totals["endpoint_count"]
        ),
        "binder_self_loop_rate": _ratio(
            binder_totals["self_loop_count"],
            binder_totals["retained_edge_count"],
        ),
        "arms": arm_summary,
    }


def _private_receipt_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = sum(
        row["private_mechanism_receipt_availability"] == "COMPLETE"
        for row in rows
    )
    premodel = sum(
        row["private_mechanism_receipt_availability"]
        == "PREMODEL_TYPED_FAILURE"
        for row in rows
    )
    count_keys = (
        "semantic_matrix",
        "document_envelope",
        "leaf",
        "bounded_set",
        "binder",
        "slot_graph",
        "mapping",
        "total",
    )
    return {
        "case_count": len(rows),
        "complete_count": complete,
        "complete_rate": _ratio(complete, len(rows)),
        "premodel_typed_failure_count": premodel,
        "receipt_counts": {
            key: sum(row["private_mechanism_receipt_counts"][key] for row in rows)
            for key in count_keys
        },
        "bundle_set_commitment_sha256": _content_hash(
            sorted(row["private_mechanism_bundle_sha256"] for row in rows)
        ),
    }


def _percentile_interval(values: list[float]) -> list[float]:
    values.sort()
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    lower = math.floor(alpha * (len(values) - 1))
    upper = math.ceil((1.0 - alpha) * (len(values) - 1))
    return [values[lower], values[upper]]


def _paired_bootstrap_effect(
    differences: Sequence[float],
    *,
    endpoint: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not differences:
        raise ScarCssmScoreError("SCAR_SCORE_PRIMARY_COHORT_EMPTY")
    count = len(differences)
    rng = random.Random(bootstrap_seed)
    bootstrap = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return {
        "endpoint": endpoint,
        "mean_difference": sum(differences) / count,
        "bootstrap_confidence_interval": _percentile_interval(bootstrap),
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": bootstrap_seed,
        "paired_case_count": count,
    }


def _exact_two_sided_mcnemar_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for comparison_index, (name, comparator) in enumerate(_PAIRED_COMPARISONS):
        f1_differences = [
            row["arm_case_metrics"][PRIMARY_ARM_ID]["mean_variant_pair_f1"]
            - row["arm_case_metrics"][comparator]["mean_variant_pair_f1"]
            for row in rows
        ]
        effect = _paired_bootstrap_effect(
            f1_differences,
            endpoint=PRIMARY_ENDPOINT,
            bootstrap_seed=BOOTSTRAP_SEED + comparison_index,
        )
        strict_differences = [
            float(
                row["arm_case_metrics"][PRIMARY_ARM_ID][
                    "both_variants_strict_exact"
                ]
            )
            - float(
                row["arm_case_metrics"][comparator][
                    "both_variants_strict_exact"
                ]
            )
            for row in rows
        ]
        strict_effect = _paired_bootstrap_effect(
            strict_differences,
            endpoint="paired_both_variants_strict_exact_indicator",
            bootstrap_seed=(
                BOOTSTRAP_SEED
                + _STRICT_BOOTSTRAP_SEED_OFFSET
                + comparison_index
            ),
        )
        strict_wins = sum(difference > 0 for difference in strict_differences)
        strict_losses = sum(difference < 0 for difference in strict_differences)
        strict_effect.update(
            {
                "primary_arm_win_count": strict_wins,
                "comparator_win_count": strict_losses,
                "tie_count": len(strict_differences)
                - strict_wins
                - strict_losses,
                "mcnemar_exact_two_sided_p_value": (
                    _exact_two_sided_mcnemar_p_value(
                        strict_wins, strict_losses
                    )
                ),
                "effect_authority": "SECONDARY_DESCRIPTIVE_ONLY",
            }
        )
        is_primary = name == PRIMARY_EFFECT_NAME
        interval = effect["bootstrap_confidence_interval"]
        passes_primary = bool(interval[0] > 0.0) if is_primary else None
        result[name] = {
            **effect,
            "primary_arm_id": PRIMARY_ARM_ID,
            "comparator_arm_id": comparator,
            "effect_authority": (
                "SOLE_PRIMARY_CONFIRMATORY"
                if is_primary
                else "SECONDARY_MECHANISM_DIAGNOSTIC_ONLY"
            ),
            "primary_success_rule_applies": is_primary,
            "passes_primary_success_rule": passes_primary,
            "strict_exact_paired_effect": strict_effect,
        }
    return result


def _stratified_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    relation: dict[str, Any] = {}
    for name in ("intra", "cross"):
        selected = [row for row in rows if row["strata"]["domain_relation"] == name]
        relation[name] = {
            "case_count": len(selected),
            "arms": {arm_id: _arm_summary(selected, arm_id) for arm_id in ARM_IDS},
            "full_minus_semantic_pair_f1_secondary": (
                _paired_bootstrap_effect(
                    [
                        row["arm_case_metrics"][PRIMARY_ARM_ID][
                            "mean_variant_pair_f1"
                        ]
                        - row["arm_case_metrics"][PRIMARY_COMPARATOR_ID][
                            "mean_variant_pair_f1"
                        ]
                        for row in selected
                    ],
                    endpoint=PRIMARY_ENDPOINT,
                    bootstrap_seed=(
                        BOOTSTRAP_SEED
                        + _STRATIFIED_BOOTSTRAP_SEED_OFFSET
                        + (0 if name == "intra" else 1)
                    ),
                )
                if selected
                else None
            ),
            "effect_authority": "SECONDARY_DESCRIPTIVE_ONLY",
        }
    arities = sorted({row["strata"]["arity"] for row in rows})
    by_arity: dict[str, Any] = {}
    for arity in arities:
        selected = [row for row in rows if row["strata"]["arity"] == arity]
        by_arity[str(arity)] = {
            "case_count": len(selected),
            "arms": {arm_id: _arm_summary(selected, arm_id) for arm_id in ARM_IDS},
            "effect_authority": "SECONDARY_DESCRIPTIVE_ONLY",
        }
    return {"domain_relation": relation, "arity": by_arity}


def _cohort_summary(
    rows: Sequence[Mapping[str, Any]], *, primary_authority: bool
) -> dict[str, Any]:
    summary = {
        "case_count": len(rows),
        "variant_count": len(rows) * len(VARIANT_NAMES),
        "effect_authority": primary_authority,
        "arms": {arm_id: _arm_summary(rows, arm_id) for arm_id in ARM_IDS},
        "mechanism": _mechanism_summary(rows),
        "execution_diagnostics": _diagnostic_summary(rows),
        "private_receipt_archive": _private_receipt_summary(rows),
        "stratified": _stratified_summary(rows),
    }
    if primary_authority:
        summary["paired_effects"] = _paired_effects(rows)
        primary_effect = summary["paired_effects"][PRIMARY_EFFECT_NAME]
        summary["confirmatory_contract"] = {
            "primary_arm_id": PRIMARY_ARM_ID,
            "primary_comparator_arm_id": PRIMARY_COMPARATOR_ID,
            "primary_endpoint": PRIMARY_ENDPOINT,
            "primary_success_rule": PRIMARY_SUCCESS_RULE,
            "multiplicity": "single_predeclared_primary_comparison_no_adjustment",
            "sampling_unit": (
                "scar_primary_item_with_base_and_system_swap_averaged_within_item"
            ),
            "population_scope": (
                "frozen_scar_primary_cohort_only_no_population_generalization"
            ),
            "secondary_endpoints_do_not_change_primary_disposition": True,
        }
        summary["primary_effect_disposition"] = (
            "PASS"
            if primary_effect["passes_primary_success_rule"]
            else "FAIL"
        )
        summary["disposition"] = "PRIMARY_EFFECT_COHORT"
    else:
        summary["paired_effects"] = None
        summary["confirmatory_contract"] = None
        summary["primary_effect_disposition"] = None
        summary["disposition"] = "SECONDARY_EXECUTION_DIAGNOSTIC_ONLY"
    return summary


def _validate_safe_aggregate(value: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "item_token",
        "opaque_slot_id",
        "background",
        "surface",
        "gold_pairs",
        "per_item",
        "proposal_hash",
        "left_graph_receipt_sha256",
        "right_graph_receipt_sha256",
        "mapping_receipt_sha256_by_arm",
    }

    def walk(child: Any) -> Iterable[Any]:
        if isinstance(child, dict):
            for key, nested in child.items():
                if key in forbidden_keys:
                    raise ScarCssmScoreError("SCAR_SCORE_SAFE_AGGREGATE_LEAKAGE")
                yield key
                yield from walk(nested)
        elif isinstance(child, list):
            for nested in child:
                yield from walk(nested)
        else:
            yield child

    for atom in walk(value):
        if isinstance(atom, str) and (
            "scar-item-v1-" in atom or "scar-slot-v1-" in atom
        ):
            raise ScarCssmScoreError("SCAR_SCORE_SAFE_AGGREGATE_LEAKAGE")


def _score_validated(
    action_pack: Mapping[str, Any],
    label_pack: Mapping[str, Any],
    prediction_pack: Mapping[str, Any],
    *,
    secret: bytes,
    study_id: str,
    expected_primary_count: int,
    expected_ambiguous_count: int,
) -> ScarCssmScoreResult:
    _require_secret_and_study(secret, study_id)
    expected_count = expected_primary_count + expected_ambiguous_count
    action_index = _index_action_pack(
        action_pack, expected_case_count=expected_count
    )
    if (
        action_pack.get("study_id") != study_id
        or label_pack.get("study_id") != study_id
    ):
        raise ScarCssmScoreError("SCAR_SCORE_STUDY_BINDING_INVALID")
    for key in (
        "action_commitment_sha256",
        "label_commitment_sha256",
        "cross_binding_hmac_sha256",
    ):
        if action_pack.get(key) != label_pack.get(key):
            raise ScarCssmScoreError("SCAR_SCORE_PACK_CROSS_BINDING_INVALID")
    expected_cross_binding = source._pack_binding_hmac(  # noqa: SLF001
        secret,
        study_id=study_id,
        action_commitment=action_pack["action_commitment_sha256"],
        label_commitment=label_pack["label_commitment_sha256"],
    )
    if not hmac.compare_digest(
        action_pack["cross_binding_hmac_sha256"], expected_cross_binding
    ):
        raise ScarCssmScoreError("SCAR_SCORE_PACK_CROSS_BINDING_INVALID")
    label_index = _index_label_pack(
        label_pack,
        action_index=action_index,
        expected_primary_count=expected_primary_count,
        expected_ambiguous_count=expected_ambiguous_count,
    )
    prediction_index = _validate_prediction_pack(
        prediction_pack, action_pack=action_pack, action_index=action_index
    )
    rows = _score_per_case(
        action_index=action_index,
        label_index=label_index,
        prediction_index=prediction_index,
    )
    primary = [
        row for row in rows if row["strata"]["cohort"] == "primary_unique_slot"
    ]
    ambiguous = [
        row for row in rows if row["strata"]["cohort"] == "ambiguous_secondary"
    ]
    if (
        len(primary) != expected_primary_count
        or len(ambiguous) != expected_ambiguous_count
    ):
        raise ScarCssmScoreError("SCAR_SCORE_LABEL_COHORT_INVALID")
    cohort_aggregates = {
        "primary_unique_slot": _cohort_summary(primary, primary_authority=True),
        "ambiguous_secondary": _cohort_summary(
            ambiguous, primary_authority=False
        ),
    }
    bindings = {
        "action_commitment_sha256": action_pack["action_commitment_sha256"],
        "label_commitment_sha256": label_pack["label_commitment_sha256"],
        "prediction_pack_self_sha256": prediction_pack["self_sha256"],
    }
    private_body = {
        "schema": PRIVATE_RESULT_SCHEMA,
        "study_id": study_id,
        "status": "SCORED_OFFLINE_ONCE",
        "bindings": dict(bindings),
        "arm_ids": list(ARM_IDS),
        "variant_names": list(VARIANT_NAMES),
        "cohorts": json.loads(_canonical_bytes(cohort_aggregates)),
        "per_item": rows,
        "access_counts": {
            "source_file_access_count": 0,
            "model_call_count": 0,
            "network_call_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "offline_scorer_call_count": 1,
        },
    }
    private_result = {
        **private_body,
        "self_sha256": _content_hash(private_body),
    }
    safe_body = {
        "schema": SAFE_AGGREGATE_SCHEMA,
        "study_id": study_id,
        "status": "SCORED_OFFLINE_ONCE",
        "bindings": bindings,
        "arm_ids": list(ARM_IDS),
        "variant_names": list(VARIANT_NAMES),
        "cohorts": cohort_aggregates,
        "private_receipt_archive": _private_receipt_summary(rows),
        "access_counts": {
            "source_file_access_count": 0,
            "model_call_count": 0,
            "network_call_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "offline_scorer_call_count": 1,
        },
        "primary_effect_cohort": "primary_unique_slot_only",
        "primary_effect_disposition": cohort_aggregates[
            "primary_unique_slot"
        ]["primary_effect_disposition"],
        "confirmatory_contract": cohort_aggregates["primary_unique_slot"][
            "confirmatory_contract"
        ],
        "secondary_cohort_scope": "execution_diagnostic_only_no_effect_authority",
        "secondary_endpoint_scope": (
            "ablation_strict_exact_domain_relation_and_arity_descriptive_only_"
            "cannot_change_primary_disposition"
        ),
        "safe_claim_scope": (
            "frozen_scar_primary_cohort_intrinsic_full_minus_semantic_pair_f1_"
            "effect_only_no_population_or_reality_transfer_authority"
        ),
    }
    _validate_safe_aggregate(safe_body)
    safe_aggregate = {**safe_body, "self_sha256": _content_hash(safe_body)}
    _validate_safe_aggregate(safe_aggregate)
    return ScarCssmScoreResult(
        private_result=private_result, safe_aggregate=safe_aggregate
    )


def score_scar_cssm_predictions_v1(
    action_pack: Mapping[str, Any],
    label_pack: Mapping[str, Any],
    prediction_pack: Mapping[str, Any],
    *,
    secret: bytes,
    study_id: str,
) -> ScarCssmScoreResult:
    """Validate the official closure and score all arms in one offline pass."""

    _require_secret_and_study(secret, study_id)
    try:
        source.validate_scar_cssm_pack_binding_v1(
            action_pack,
            label_pack,
            secret=secret,
            study_id=study_id,
        )
    except source.ScarCssmSourceError as exc:
        raise ScarCssmScoreError(
            f"SCAR_SCORE_SOURCE_BINDING_INVALID__{exc.issue_id}"
        ) from exc
    return _score_validated(
        action_pack,
        label_pack,
        prediction_pack,
        secret=secret,
        study_id=study_id,
        expected_primary_count=source.EXPECTED_PRIMARY_ROW_COUNT,
        expected_ambiguous_count=source.EXPECTED_AMBIGUOUS_ROW_COUNT,
    )


def _score_scar_cssm_fixture_v1(
    action_pack: Mapping[str, Any],
    label_pack: Mapping[str, Any],
    prediction_pack: Mapping[str, Any],
    *,
    secret: bytes,
    study_id: str,
    expected_primary_count: int,
    expected_ambiguous_count: int,
) -> ScarCssmScoreResult:
    """Internal tiny-fixture path with no official/effect authority."""

    return _score_validated(
        action_pack,
        label_pack,
        prediction_pack,
        secret=secret,
        study_id=study_id,
        expected_primary_count=expected_primary_count,
        expected_ambiguous_count=expected_ambiguous_count,
    )


__all__ = [
    "ARM_IDS",
    "BINDER_COVERAGE_DISPOSITIONS",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "DISPOSITIONS",
    "ERROR_CODES",
    "MAX_DIAGNOSTIC_COUNT",
    "MAX_PROPOSALS_PER_POOL",
    "MAX_SCORE_ABS",
    "OPERATOR_IDS",
    "PRIMARY_ARM_ID",
    "PRIMARY_COMPARATOR_ID",
    "PRIMARY_EFFECT_NAME",
    "PRIMARY_ENDPOINT",
    "PRIMARY_SUCCESS_RULE",
    "PREDICTION_PACK_SCHEMA",
    "PRIVATE_RESULT_SCHEMA",
    "SAFE_AGGREGATE_SCHEMA",
    "STRUCTURAL_STATUSES",
    "ScarCssmScoreError",
    "ScarCssmScoreResult",
    "seal_scar_cssm_prediction_pack_v1",
    "score_scar_cssm_predictions_v1",
]

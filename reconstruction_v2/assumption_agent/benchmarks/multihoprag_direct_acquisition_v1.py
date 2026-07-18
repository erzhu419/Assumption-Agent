"""Fail-closed one-shot acquisition for the MultiHopRAG joint v2 study.

The module has two formal phases.  ``create_source_custody`` byte-binds the
two pinned public source files, verifies the already committed implementation
freeze, consumes an exclusive marker, makes exactly one 32-byte selection
secret, and publishes only its commitment.  After that custody receipt is
committed, ``formal_acquire`` consumes a second exclusive marker before it
parses either source and forms the four globally disjoint HMAC blocks.

Pure helpers accept caller-supplied in-memory payloads for synthetic tests.
They never download, discover, cache, or open an official source.  The action
view contains the query and URL-free corpus only.  It never contains source
``question_type``, answer, evidence, URL, gold IDs, source-row ordinals, or a
candidate identity commitment.  F_search has no label pack.  M_search view
and labels are sealed behind an explicit, self-hashed A_hold-promotion receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any
import unicodedata


VERSION = "multihoprag_direct_acquisition_v1"
PUBLIC_RECEIPT_SCHEMA = VERSION
SOURCE_CUSTODY_SCHEMA = "multihoprag_source_selection_custody_v1"
IMPLEMENTATION_FREEZE_SCHEMA = "multihoprag_joint_graph_implementation_freeze_v1"
ATTEMPT_MARKER_SCHEMA = f"{VERSION}_one_shot_marker"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure"
CORPUS_VIEW_SCHEMA = f"{VERSION}_corpus_view"
BLOCK_VIEW_SCHEMA = f"{VERSION}_block_view"
BLOCK_LABEL_SCHEMA = f"{VERSION}_block_labels"
VIEW_ITEM_SCHEMA = f"{VERSION}_view_item"
LABEL_ITEM_SCHEMA = f"{VERSION}_label_item"
PROMOTION_SCHEMA = "multihoprag_a_hold_promotion_v1"
PROMOTION_RELATIVE = "manifests/multihoprag_a_hold_promotion_v1.json"
F_POLICY_FREEZE_SCHEMA = "multihoprag_f_search_policy_freeze_v1"
F_POLICY_FREEZE_RELATIVE = "manifests/multihoprag_f_search_policy_freeze_v1.json"
A_FORM_POLICY_FREEZE_SCHEMA = "multihoprag_a_form_policy_freeze_v1"
A_FORM_POLICY_FREEZE_RELATIVE = (
    "manifests/multihoprag_a_form_policy_freeze_v1.json"
)
ACTION_SEAL_RELATIVES = {
    "A_form": "manifests/multihoprag_a_form_action_seal_v1.json",
    "A_hold": "manifests/multihoprag_a_hold_action_seal_v1.json",
    "M_search": "manifests/multihoprag_m_search_action_seal_v1.json",
}
ACTION_SEAL_SCHEMAS = {
    block: f"multihoprag_{block.casefold()}_action_seal_v1"
    for block in ACTION_SEAL_RELATIVES
}

DATASET_REPOSITORY = "yixuantt/MultiHopRAG"
DATASET_COMMIT = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
CODE_COMMIT = "cde8e844af14b3012f20158abc2854fe8458212a"
QUERY_SOURCE_NAME = "MultiHopRAG.json"
QUERY_SOURCE_SIZE = 5_171_312
QUERY_SOURCE_GIT_BLOB_SHA1 = "fcb9efe65c7730dd4126a42afb6c2c7e45721ebb"
CORPUS_SOURCE_NAME = "corpus.json"
CORPUS_SOURCE_SIZE = 6_785_567
CORPUS_SOURCE_GIT_BLOB_SHA1 = "bb98345ef3921312aad05fd117b5a3e39888de2c"
QUERY_RECORD_COUNT = 2_556
CORPUS_RECORD_COUNT = 609
EXPOSED_QUERY_STOP = 128

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
FAMILIES = ("comparison_query", "inference_query", "temporal_query")
NULL_FAMILY = "null_query"
ALL_SOURCE_QUESTION_TYPES = (*FAMILIES, NULL_FAMILY)
FAMILY_QUOTAS = {
    "A_form": 32,
    "F_search": 16,
    "A_hold": 24,
    "M_search": 24,
}
BLOCK_COUNTS = {
    block: FAMILY_QUOTAS[block] * len(FAMILIES) for block in BLOCK_ORDER
}
TOTAL_SELECTED = sum(BLOCK_COUNTS.values())
AGENT_ACTION_IDS = (
    "P0_IND_SUM",
    "P1_IND_MAXIMIN",
    "P2_ENTITY_BRIDGE",
    "P3_TOPIC_BRIDGE",
    "P4_META_ASSIGN",
    "P5_FAMILY_UNION",
)
STAGE_OUTPUT_ARCHIVE_RELATIVES = {
    block: (
        "artifacts/multihoprag_joint_graph_formal_v1/"
        f"{block}.stage_output.private.json"
    )
    for block in BLOCK_ORDER
}
STAGE_OUTPUT_ARCHIVE_SCHEMAS = {
    block: f"multihoprag_{block.casefold()}_stage_output_archive_v1"
    for block in BLOCK_ORDER
}
STAGE_RUNTIME_BINDING_KEYS = (
    "preparation_sha256",
    "graph_sha256",
    "embedding_index_sha256",
    "ner_runtime_receipt_sha256",
    "ner_entity_matrix_sha256",
    "hippo_build_receipt_sha256",
    "hippo_retrieval_receipt_sha256",
    "execution_matrix_sha256",
)

DESIGN_RELATIVE = "manifests/multihoprag_joint_graph_evaluator_design_v2.json"
TYPED_CORE_RELATIVE = (
    "assumption_agent/benchmarks/multihoprag_typed_operator_v2.py"
)
ACQUISITION_RELATIVE = (
    "assumption_agent/benchmarks/multihoprag_direct_acquisition_v1.py"
)
HIPPO_ADAPTER_RELATIVE = (
    "replication_runtime/multihoprag_official_hipporag_v1/adapter.py"
)
IMPLEMENTATION_FREEZE_RELATIVE = (
    "manifests/multihoprag_joint_graph_implementation_freeze_v1.json"
)
SOURCE_CUSTODY_RELATIVE = (
    "manifests/multihoprag_source_selection_custody_v1.json"
)
PUBLIC_RECEIPT_RELATIVE = "manifests/multihoprag_direct_acquisition_v1.json"

QUERY_SOURCE_RELATIVE = (
    "artifacts/multihoprag_official_source_v1/MultiHopRAG-71ac0d0b.json"
)
CORPUS_SOURCE_RELATIVE = (
    "artifacts/multihoprag_official_source_v1/corpus-71ac0d0b.json"
)
PRIVATE_ROOT_RELATIVE = "artifacts/multihoprag_direct_acquisition_v1"
SELECTION_SECRET_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/selection.key"
SECRET_MARKER_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/secret_creation.marker"
ACQUISITION_MARKER_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/acquisition.marker"
FAILURE_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/terminal_failure.json"
CORPUS_VIEW_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/corpus_view.private.json"
BLOCK_VIEW_RELATIVES = {
    block: f"{PRIVATE_ROOT_RELATIVE}/{block}.view.private.json"
    for block in BLOCK_ORDER
}
BLOCK_LABEL_RELATIVES = {
    block: f"{PRIVATE_ROOT_RELATIVE}/{block}.labels.private.json"
    for block in ("A_form", "A_hold", "M_search")
}

REQUIRED_FREEZE_ROLES = frozenset(
    {
        "design",
        "typed_core",
        "acquisition",
        "minilm_asset_manifest",
        "minilm_base_runtime_binding",
        "minilm_runtime_binding",
        "ner_asset_manifest",
        "ner_contract",
        "ner_runtime_binding",
        "ner_worker",
        "global_hipporag_contract",
        "global_hipporag_adapter",
        "global_hipporag_worker",
        "hipporag_v3_attestation",
        "formal_runner",
    }
)
FIXED_FREEZE_ROLE_PATHS = {
    "design": DESIGN_RELATIVE,
    "typed_core": TYPED_CORE_RELATIVE,
    "acquisition": ACQUISITION_RELATIVE,
    "minilm_asset_manifest": "manifests/qasper_minilm_runtime_asset_v1.json",
    "minilm_base_runtime_binding": (
        "replication_runtime/qasper_minilm_v1/binding.py"
    ),
    "minilm_runtime_binding": (
        "replication_runtime/multihoprag_minilm_v1/adapter.py"
    ),
    "ner_asset_manifest": "manifests/multihoprag_ner_runtime_asset_v1.json",
    "ner_contract": "replication_runtime/multihoprag_ner_v1/contract.py",
    "ner_runtime_binding": "replication_runtime/multihoprag_ner_v1/binding.py",
    "ner_worker": "replication_runtime/multihoprag_ner_v1/worker.py",
    "global_hipporag_contract": (
        "replication_runtime/multihoprag_official_hipporag_v1/contract.py"
    ),
    "global_hipporag_adapter": HIPPO_ADAPTER_RELATIVE,
    "global_hipporag_worker": (
        "replication_runtime/multihoprag_official_hipporag_v1/worker.py"
    ),
    "hipporag_v3_attestation": (
        "manifests/musique_official_hipporag_runtime_attestation_v3.json"
    ),
    "formal_runner": (
        "assumption_agent/benchmarks/multihoprag_joint_graph_formal_runner_v1.py"
    ),
}

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HMAC_DOMAIN = b"multihoprag_direct_acquisition_v1/hmac-sha256/v1"
_FORMAL_ENTRY_ACTIVE = False
_FORMAL_EXECUTION_CAPABILITY = object()


class MultiHopRAGAcquisitionError(RuntimeError):
    """Raised when a frozen acquisition or isolation invariant drifts."""


@dataclass(frozen=True)
class SourceFileBinding:
    logical_name: str
    sha256: str
    git_blob_sha1: str
    byte_size: int


@dataclass(frozen=True)
class CorpusArticle:
    article_id: int
    exact_url: str
    title: str
    author: str
    source: str
    published_at: str
    category: str
    body: str

    def view_row(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "author": self.author,
            "source": self.source,
            "published_at": self.published_at,
            "category": self.category,
            "body": self.body,
        }


@dataclass(frozen=True)
class PrivateCandidate:
    source_row_ordinal: int
    query: str
    normalized_query: str
    normalized_query_sha256: str
    question_type: str
    answer: str
    gold_article_ids: tuple[int, ...]
    gold_url_identity_sha256: str
    evidence_object_hashes: tuple[str, ...]
    evidence_object_sequence_sha256: str
    identity_commitment_sha256: str
    source_record_commitment_sha256: str


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    count: int
    view_file_sha256: str
    view_semantic_sha256: str
    label_file_sha256: str | None
    label_semantic_sha256: str | None


@dataclass(frozen=True)
class AcquisitionPaths:
    marker: Path
    failure: Path
    corpus_view: Path
    block_views: Mapping[str, Path]
    block_labels: Mapping[str, Path]
    public_receipt: Path


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MultiHopRAGAcquisitionError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MultiHopRAGAcquisitionError(f"{field} is not a SHA256")
    return value


def _strict_text(value: object, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise MultiHopRAGAcquisitionError(f"{field} is not valid text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MultiHopRAGAcquisitionError(f"{field} is not valid Unicode") from exc
    if nonempty and not value.strip():
        raise MultiHopRAGAcquisitionError(f"{field} is empty")
    return value


def normalize_query(value: str) -> str:
    """Exact frozen NFKC + casefold + Unicode-whitespace normalization."""

    exact = _strict_text(value, "query", nonempty=True)
    normalized = " ".join(unicodedata.normalize("NFKC", exact).casefold().split())
    if not normalized:
        raise MultiHopRAGAcquisitionError("normalized query is empty")
    return normalized


def _frame_part(raw: bytes) -> bytes:
    return len(raw).to_bytes(8, byteorder="big", signed=False) + raw


def hmac_digest(secret: bytes, purpose: str, *parts: str) -> bytes:
    """Length-prefixed, purpose-separated private ordering digest."""

    if not isinstance(secret, bytes) or len(secret) != 32:
        raise MultiHopRAGAcquisitionError("selection secret must be exactly 32 bytes")
    encoded = [_strict_text(purpose, "HMAC purpose", nonempty=True).encode("utf-8")]
    for part in parts:
        encoded.append(_strict_text(part, "HMAC part").encode("utf-8"))
    message = _frame_part(_HMAC_DOMAIN) + b"".join(_frame_part(row) for row in encoded)
    return hmac.new(secret, message, hashlib.sha256).digest()


def _strict_json(raw: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise MultiHopRAGAcquisitionError(f"{label} contains {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MultiHopRAGAcquisitionError(
                    f"{label} contains a duplicate object key"
                )
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiHopRAGAcquisitionError(f"{label} is invalid strict JSON") from exc


def _with_self_hash(
    body: Mapping[str, Any], hash_field: str
) -> dict[str, Any]:
    if hash_field in body:
        raise MultiHopRAGAcquisitionError("self-hash field was already populated")
    payload = dict(body)
    payload[hash_field] = stable_hash(body)
    return payload


def verify_self_hash(
    payload: Mapping[str, Any], *, hash_field: str, schema: str | None = None
) -> str:
    if schema is not None and payload.get("schema") != schema:
        raise MultiHopRAGAcquisitionError(f"{schema} schema mismatch")
    declared = _require_sha256(payload.get(hash_field), hash_field)
    body = dict(payload)
    del body[hash_field]
    observed = stable_hash(body)
    if not hmac.compare_digest(declared, observed):
        raise MultiHopRAGAcquisitionError(f"{hash_field} self-hash mismatch")
    return observed


def _json_scalar(value: object) -> bool:
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _parse_corpus(
    payload: Any, *, enforce_formal_count: bool
) -> tuple[tuple[CorpusArticle, ...], dict[str, int]]:
    if not isinstance(payload, list):
        raise MultiHopRAGAcquisitionError("corpus root must be a list")
    if enforce_formal_count and len(payload) != CORPUS_RECORD_COUNT:
        raise MultiHopRAGAcquisitionError("formal corpus record count drifted")
    articles: list[CorpusArticle] = []
    url_owner: dict[str, int] = {}
    expected_keys = {
        "title",
        "author",
        "source",
        "published_at",
        "category",
        "url",
        "body",
    }
    for article_id, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise MultiHopRAGAcquisitionError("corpus article schema drifted")
        values = {
            key: _strict_text(
                raw.get(key),
                f"corpus article {key}",
                nonempty=key in {"source", "url"},
            )
            for key in expected_keys
        }
        url = values["url"]
        if url in url_owner:
            raise MultiHopRAGAcquisitionError("corpus exact URLs are not unique")
        url_owner[url] = article_id
        articles.append(
            CorpusArticle(
                article_id=article_id,
                exact_url=url,
                title=values["title"],
                author=values["author"],
                source=values["source"],
                published_at=values["published_at"],
                category=values["category"],
                body=values["body"],
            )
        )
    if not articles:
        raise MultiHopRAGAcquisitionError("corpus is empty")
    return tuple(articles), url_owner


def _evidence_identity(
    evidence_list: object,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], Counter[str]]:
    if not isinstance(evidence_list, list) or not evidence_list:
        raise MultiHopRAGAcquisitionError("evidence_list must be a nonempty list")
    urls: list[str] = []
    facts: list[str] = []
    object_hashes: list[str] = []
    key_sets: Counter[str] = Counter()
    for evidence in evidence_list:
        if (
            not isinstance(evidence, Mapping)
            or not 2 <= len(evidence) <= 16
            or "url" not in evidence
            or "fact" not in evidence
            or any(not isinstance(key, str) for key in evidence)
            or any(not _json_scalar(value) for value in evidence.values())
        ):
            raise MultiHopRAGAcquisitionError("evidence object schema drifted")
        url = _strict_text(evidence.get("url"), "evidence URL", nonempty=True)
        fact = _strict_text(evidence.get("fact"), "evidence fact", nonempty=True)
        for key, value in evidence.items():
            _strict_text(key, "evidence metadata key", nonempty=True)
            if isinstance(value, str):
                _strict_text(value, "evidence scalar metadata")
        exact = dict(evidence)
        urls.append(url)
        facts.append(fact)
        object_hashes.append(stable_hash(exact))
        key_sets[stable_hash(sorted(exact))] += 1
    return tuple(urls), tuple(facts), tuple(object_hashes), key_sets


def _normalized_evidence_text(value: str, field: str) -> str:
    exact = _strict_text(value, field, nonempty=True)
    normalized = " ".join(unicodedata.normalize("NFKC", exact).casefold().split())
    if not normalized:
        raise MultiHopRAGAcquisitionError(f"normalized {field} is empty")
    return normalized


def _candidate_identity(
    *,
    source_row_ordinal: int,
    exact_query: str,
    exact_answer: str,
    normalized_query_sha256: str,
    question_type: str,
    exact_urls: Sequence[str],
    evidence_object_hashes: Sequence[str],
) -> tuple[str, str, str, str]:
    url_hashes = tuple(sorted({_sha256_bytes(url.encode("utf-8")) for url in exact_urls}))
    gold_url_identity = stable_hash(
        [VERSION, "gold_exact_url_identity", list(url_hashes)]
    )
    evidence_sequence = stable_hash(
        [VERSION, "complete_evidence_object_sequence", list(evidence_object_hashes)]
    )
    identity = stable_hash(
        {
            "domain": f"{VERSION}/selection-identity/v1",
            "source_row_ordinal": source_row_ordinal,
            "normalized_query_sha256": normalized_query_sha256,
            "question_type": question_type,
            "gold_url_identity_sha256": gold_url_identity,
        }
    )
    source_record = stable_hash(
        {
            "domain": f"{VERSION}/complete-private-source-record/v1",
            "selection_identity_commitment_sha256": identity,
            "exact_query_sha256": _sha256_bytes(exact_query.encode("utf-8")),
            "exact_answer_sha256": _sha256_bytes(exact_answer.encode("utf-8")),
            "evidence_object_sequence_sha256": evidence_sequence,
        }
    )
    return gold_url_identity, evidence_sequence, identity, source_record


def parse_source_payloads(
    *,
    query_payload: Any,
    corpus_payload: Any,
    enforce_formal_counts: bool = False,
) -> tuple[tuple[CorpusArticle, ...], tuple[PrivateCandidate, ...], dict[str, Any]]:
    """Qualify source schema and form eligible private candidate identities."""

    articles, url_owner = _parse_corpus(
        corpus_payload, enforce_formal_count=enforce_formal_counts
    )
    if not isinstance(query_payload, list):
        raise MultiHopRAGAcquisitionError("query root must be a list")
    if enforce_formal_counts and len(query_payload) != QUERY_RECORD_COUNT:
        raise MultiHopRAGAcquisitionError("formal query record count drifted")
    expected_keys = {"query", "evidence_list", "question_type", "answer"}
    exclusion_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    key_set_counts: Counter[str] = Counter()
    evidence_count_histogram: Counter[int] = Counter()
    gold_count_histogram: Counter[int] = Counter()
    candidates: list[PrivateCandidate] = []
    exposed_url = articles[0].exact_url
    for ordinal, raw in enumerate(query_payload):
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise MultiHopRAGAcquisitionError("query row schema drifted")
        query = _strict_text(raw.get("query"), "query", nonempty=True)
        answer = _strict_text(raw.get("answer"), "answer", nonempty=True)
        question_type = _strict_text(
            raw.get("question_type"), "question_type", nonempty=True
        )
        if question_type not in ALL_SOURCE_QUESTION_TYPES:
            raise MultiHopRAGAcquisitionError("source question_type vocabulary drifted")
        type_counts[question_type] += 1
        normalized = normalize_query(query)
        normalized_sha = _sha256_bytes(normalized.encode("utf-8"))
        exact_urls, exact_facts, evidence_hashes, row_key_sets = _evidence_identity(
            raw.get("evidence_list")
        )
        key_set_counts.update(row_key_sets)
        evidence_count_histogram[len(exact_urls)] += 1
        joined_ids = tuple(sorted({url_owner[url] for url in exact_urls if url in url_owner}))
        gold_count_histogram[len(joined_ids)] += 1
        gold_url_identity, evidence_sequence, identity, source_record = _candidate_identity(
            source_row_ordinal=ordinal,
            exact_query=query,
            exact_answer=answer,
            normalized_query_sha256=normalized_sha,
            question_type=question_type,
            exact_urls=exact_urls,
            evidence_object_hashes=evidence_hashes,
        )

        # Eligibility exclusions are deliberately ordered and exhaustive.  The
        # entire row was schema-qualified first, so exposed rows cannot hide
        # structural drift in the pinned source.
        if ordinal < EXPOSED_QUERY_STOP:
            exclusion_counts["known_public_query_window"] += 1
            continue
        if exposed_url in exact_urls:
            exclusion_counts["gold_references_exposed_article0"] += 1
            continue
        if question_type == NULL_FAMILY:
            exclusion_counts["null_query"] += 1
            continue
        if any(url not in url_owner for url in exact_urls):
            exclusion_counts["exact_url_join_failure"] += 1
            continue
        if not 2 <= len(joined_ids) <= 4:
            exclusion_counts["deduplicated_gold_size_not_2_to_4"] += 1
            continue
        if any(
            _normalized_evidence_text(fact, "evidence fact")
            not in _normalized_evidence_text(
                articles[url_owner[url]].body, "joined article body"
            )
            for url, fact in zip(exact_urls, exact_facts, strict=True)
        ):
            exclusion_counts["evidence_fact_not_in_joined_article_body"] += 1
            continue
        candidates.append(
            PrivateCandidate(
                source_row_ordinal=ordinal,
                query=query,
                normalized_query=normalized,
                normalized_query_sha256=normalized_sha,
                question_type=question_type,
                answer=answer,
                gold_article_ids=joined_ids,
                gold_url_identity_sha256=gold_url_identity,
                evidence_object_hashes=evidence_hashes,
                evidence_object_sequence_sha256=evidence_sequence,
                identity_commitment_sha256=identity,
                source_record_commitment_sha256=source_record,
            )
        )
    stats = {
        "root_counts": {
            "query_records": len(query_payload),
            "corpus_articles": len(articles),
        },
        "source_question_type_counts": {
            key: type_counts[key] for key in ALL_SOURCE_QUESTION_TYPES
        },
        "eligibility_exclusion_counts": {
            key: exclusion_counts[key] for key in sorted(exclusion_counts)
        },
        "eligible_before_collision_grouping": len(candidates),
        "evidence_object_count_histogram": {
            str(key): evidence_count_histogram[key]
            for key in sorted(evidence_count_histogram)
        },
        "deduplicated_gold_article_count_histogram": {
            str(key): gold_count_histogram[key]
            for key in sorted(gold_count_histogram)
        },
        "observed_evidence_key_set_sha256_histogram": {
            key: key_set_counts[key] for key in sorted(key_set_counts)
        },
        "only_url_and_fact_consumed_for_join_and_gold": True,
        "all_eligible_evidence_facts_exact_normalized_body_contained": True,
        "all_evidence_scalar_metadata_identity_bound_but_not_semantically_interpreted": True,
        "complete_evidence_object_hash_excluded_from_HMAC_selection_identity": True,
    }
    return articles, tuple(candidates), stats


def select_private_blocks(
    candidates: Sequence[PrivateCandidate], *, secret: bytes
) -> tuple[dict[str, tuple[PrivateCandidate, ...]], dict[str, Any]]:
    """Choose one collision representative, then one total order per family."""

    if not isinstance(secret, bytes) or len(secret) != 32:
        raise MultiHopRAGAcquisitionError("selection secret must be exactly 32 bytes")
    groups: dict[str, list[PrivateCandidate]] = defaultdict(list)
    for candidate in candidates:
        if (
            not isinstance(candidate, PrivateCandidate)
            or candidate.question_type not in FAMILIES
            or not 2 <= len(candidate.gold_article_ids) <= 4
        ):
            raise MultiHopRAGAcquisitionError("eligible candidate drifted")
        groups[candidate.normalized_query].append(candidate)

    representatives: list[PrivateCandidate] = []
    collision_size_histogram: Counter[int] = Counter()
    for normalized_query in sorted(groups):
        rows = groups[normalized_query]
        collision_size_histogram[len(rows)] += 1
        normalized_sha = _sha256_bytes(normalized_query.encode("utf-8"))
        representative = min(
            rows,
            key=lambda row: (
                hmac_digest(
                    secret,
                    "collision_representative",
                    normalized_sha,
                    row.identity_commitment_sha256,
                ),
                row.identity_commitment_sha256,
                row.source_row_ordinal,
            ),
        )
        representatives.append(representative)

    family_orders: dict[str, tuple[PrivateCandidate, ...]] = {}
    required_per_family = sum(FAMILY_QUOTAS.values())
    for family in FAMILIES:
        rows = [row for row in representatives if row.question_type == family]
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (
                    hmac_digest(
                        secret,
                        "family_continuous_total_order",
                        family,
                        row.identity_commitment_sha256,
                    ),
                    row.identity_commitment_sha256,
                ),
            )
        )
        if len(ordered) < required_per_family:
            raise MultiHopRAGAcquisitionError(
                f"source capacity insufficient for exact {family} quotas"
            )
        family_orders[family] = ordered

    selected: dict[str, list[PrivateCandidate]] = {
        block: [] for block in BLOCK_ORDER
    }
    offsets = {family: 0 for family in FAMILIES}
    for block in BLOCK_ORDER:
        quota = FAMILY_QUOTAS[block]
        for family in FAMILIES:
            start = offsets[family]
            stop = start + quota
            selected[block].extend(family_orders[family][start:stop])
            offsets[family] = stop

    # Membership follows the contiguous family orders above.  A separate
    # private presentation permutation prevents view ordinal from revealing
    # which source question_type stratum supplied an item.
    for block in BLOCK_ORDER:
        selected[block].sort(
            key=lambda row: (
                hmac_digest(
                    secret,
                    "block_private_presentation_order",
                    block,
                    row.identity_commitment_sha256,
                ),
                row.identity_commitment_sha256,
            )
        )

    flattened = [row for block in BLOCK_ORDER for row in selected[block]]
    if (
        len(flattened) != TOTAL_SELECTED
        or len({row.identity_commitment_sha256 for row in flattened})
        != TOTAL_SELECTED
        or len({row.normalized_query for row in flattened}) != TOTAL_SELECTED
    ):
        raise MultiHopRAGAcquisitionError("selected blocks are not globally disjoint")
    exact_counts = {
        block: dict(Counter(row.question_type for row in selected[block]))
        for block in BLOCK_ORDER
    }
    for block in BLOCK_ORDER:
        if exact_counts[block] != {
            family: FAMILY_QUOTAS[block] for family in FAMILIES
        }:
            raise MultiHopRAGAcquisitionError("selected family quotas drifted")
    stats = {
        "normalized_query_collision_group_count": len(groups),
        "collision_group_size_histogram": {
            str(key): collision_size_histogram[key]
            for key in sorted(collision_size_histogram)
        },
        "selected_block_counts": {
            block: len(selected[block]) for block in BLOCK_ORDER
        },
        "selected_exact_family_counts": exact_counts,
        "selection_contract": {
            "collision_representative_before_family_ranking": True,
            "one_continuous_total_order_per_family": True,
            "block_membership_uses_contiguous_family_slices": True,
            "block_presentation_order_is_private_HMAC_blinded": True,
            "view_ordinal_reveals_source_question_type": False,
            "block_order": list(BLOCK_ORDER),
            "global_normalized_query_disjointness": True,
            "secret_rotation_replay_resample_or_replacement": False,
        },
    }
    return {block: tuple(selected[block]) for block in BLOCK_ORDER}, stats


_CORPUS_ARTICLE_VIEW_KEYS = {
    "article_id",
    "title",
    "author",
    "source",
    "published_at",
    "category",
    "body",
}
_VIEW_ITEM_KEYS = {"schema", "block", "ordinal", "query"}
_LABEL_ITEM_KEYS = {
    "schema",
    "block",
    "ordinal",
    "view_sha256",
    "identity_commitment_sha256",
    "source_record_commitment_sha256",
    "question_type",
    "answer",
    "gold_article_ids",
}


def _validate_corpus_view(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema",
        "version",
        "article_count",
        "corpus_locator_fields_included",
        "articles",
        "corpus_view_sha256",
    }:
        raise MultiHopRAGAcquisitionError("corpus view envelope drifted")
    verify_self_hash(
        payload, hash_field="corpus_view_sha256", schema=CORPUS_VIEW_SCHEMA
    )
    articles = payload.get("articles")
    if (
        payload.get("version") != VERSION
        or payload.get("corpus_locator_fields_included") is not False
        or not isinstance(articles, list)
        or payload.get("article_count") != len(articles)
        or any(
            not isinstance(row, Mapping)
            or set(row) != _CORPUS_ARTICLE_VIEW_KEYS
            for row in articles
        )
        or [row.get("article_id") for row in articles] != list(range(len(articles)))
    ):
        raise MultiHopRAGAcquisitionError("URL-free corpus view drifted")
    for row in articles:
        for key in _CORPUS_ARTICLE_VIEW_KEYS - {"article_id"}:
            _strict_text(row.get(key), f"corpus view {key}")


def _validate_view_item(item: Mapping[str, Any], *, block: str, ordinal: int) -> None:
    if (
        set(item) != _VIEW_ITEM_KEYS
        or item.get("schema") != VIEW_ITEM_SCHEMA
        or item.get("block") != block
        or item.get("ordinal") != ordinal
        or not isinstance(item.get("query"), str)
        or not item["query"].strip()
    ):
        raise MultiHopRAGAcquisitionError("gold-free view item drifted")


def _validate_block_view(payload: Mapping[str, Any], *, expected_block: str) -> None:
    if set(payload) != {
        "schema",
        "version",
        "block",
        "item_count",
        "late_label_fields_included",
        "items",
        "block_view_sha256",
    }:
        raise MultiHopRAGAcquisitionError("block view envelope drifted")
    verify_self_hash(
        payload, hash_field="block_view_sha256", schema=BLOCK_VIEW_SCHEMA
    )
    items = payload.get("items")
    if (
        expected_block not in BLOCK_ORDER
        or payload.get("version") != VERSION
        or payload.get("block") != expected_block
        or payload.get("late_label_fields_included") is not False
        or not isinstance(items, list)
        or payload.get("item_count") != len(items)
        or len(items) != BLOCK_COUNTS[expected_block]
    ):
        raise MultiHopRAGAcquisitionError("block view identity drifted")
    for ordinal, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise MultiHopRAGAcquisitionError("block view item is not an object")
        _validate_view_item(item, block=expected_block, ordinal=ordinal)
    if len({stable_hash(row) for row in items}) != len(items):
        raise MultiHopRAGAcquisitionError("block view item hashes overlap")


def _validate_block_labels(
    payload: Mapping[str, Any], *, expected_block: str
) -> None:
    if expected_block == "F_search":
        raise MultiHopRAGAcquisitionError("F_search label pack does not exist")
    if set(payload) != {
        "schema",
        "version",
        "block",
        "item_count",
        "source_locator_payload_included",
        "items",
        "block_labels_sha256",
    }:
        raise MultiHopRAGAcquisitionError("block label envelope drifted")
    verify_self_hash(
        payload, hash_field="block_labels_sha256", schema=BLOCK_LABEL_SCHEMA
    )
    items = payload.get("items")
    if (
        expected_block not in BLOCK_ORDER
        or payload.get("version") != VERSION
        or payload.get("block") != expected_block
        or payload.get("source_locator_payload_included") is not False
        or not isinstance(items, list)
        or payload.get("item_count") != len(items)
        or len(items) != BLOCK_COUNTS[expected_block]
    ):
        raise MultiHopRAGAcquisitionError("block label identity drifted")
    identities: set[str] = set()
    source_records: set[str] = set()
    view_hashes: set[str] = set()
    family_counts: Counter[str] = Counter()
    for ordinal, item in enumerate(items):
        if (
            not isinstance(item, Mapping)
            or set(item) != _LABEL_ITEM_KEYS
            or item.get("schema") != LABEL_ITEM_SCHEMA
            or item.get("block") != expected_block
            or item.get("ordinal") != ordinal
            or item.get("question_type") not in FAMILIES
            or not isinstance(item.get("answer"), str)
            or not item["answer"].strip()
            or not isinstance(item.get("gold_article_ids"), list)
            or not 2 <= len(item["gold_article_ids"]) <= 4
            or any(type(index) is not int or index < 0 for index in item["gold_article_ids"])
            or item["gold_article_ids"] != sorted(set(item["gold_article_ids"]))
        ):
            raise MultiHopRAGAcquisitionError("late label item drifted")
        identities.add(
            _require_sha256(
                item.get("identity_commitment_sha256"), "candidate identity"
            )
        )
        source_records.add(
            _require_sha256(
                item.get("source_record_commitment_sha256"),
                "complete private source record",
            )
        )
        view_hashes.add(_require_sha256(item.get("view_sha256"), "view hash"))
        family_counts[str(item["question_type"])] += 1
    if (
        len(identities) != len(items)
        or len(source_records) != len(items)
        or len(view_hashes) != len(items)
        or family_counts
        != Counter(
            {family: FAMILY_QUOTAS[expected_block] for family in FAMILIES}
        )
    ):
        raise MultiHopRAGAcquisitionError("late label commitments or quotas drifted")


def materialize_private_payloads(
    *,
    articles: Sequence[CorpusArticle],
    blocks: Mapping[str, Sequence[PrivateCandidate]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Build URL-free views and separate late labels entirely in memory."""

    if set(blocks) != set(BLOCK_ORDER):
        raise MultiHopRAGAcquisitionError("private block set drifted")
    corpus = _with_self_hash(
        {
            "schema": CORPUS_VIEW_SCHEMA,
            "version": VERSION,
            "article_count": len(articles),
            "corpus_locator_fields_included": False,
            "articles": [article.view_row() for article in articles],
        },
        "corpus_view_sha256",
    )
    _validate_corpus_view(corpus)

    views: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}
    for block in BLOCK_ORDER:
        rows = tuple(blocks[block])
        if len(rows) != BLOCK_COUNTS[block]:
            raise MultiHopRAGAcquisitionError("private block count drifted")
        view_items = [
            {
                "schema": VIEW_ITEM_SCHEMA,
                "block": block,
                "ordinal": ordinal,
                "query": row.query,
            }
            for ordinal, row in enumerate(rows)
        ]
        view = _with_self_hash(
            {
                "schema": BLOCK_VIEW_SCHEMA,
                "version": VERSION,
                "block": block,
                "item_count": len(view_items),
                "late_label_fields_included": False,
                "items": view_items,
            },
            "block_view_sha256",
        )
        _validate_block_view(view, expected_block=block)
        views[block] = view
        if block == "F_search":
            continue
        label_items = [
            {
                "schema": LABEL_ITEM_SCHEMA,
                "block": block,
                "ordinal": ordinal,
                "view_sha256": stable_hash(view_items[ordinal]),
                "identity_commitment_sha256": row.identity_commitment_sha256,
                "source_record_commitment_sha256": (
                    row.source_record_commitment_sha256
                ),
                "question_type": row.question_type,
                "answer": row.answer,
                "gold_article_ids": list(row.gold_article_ids),
            }
            for ordinal, row in enumerate(rows)
        ]
        label = _with_self_hash(
            {
                "schema": BLOCK_LABEL_SCHEMA,
                "version": VERSION,
                "block": block,
                "item_count": len(label_items),
                "source_locator_payload_included": False,
                "items": label_items,
            },
            "block_labels_sha256",
        )
        _validate_block_labels(label, expected_block=block)
        labels[block] = label
    if set(labels) != {"A_form", "A_hold", "M_search"}:
        raise MultiHopRAGAcquisitionError("late label pack set drifted")
    return corpus, views, labels


def _require_regular_file(path: Path, *, field: str, mode: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise MultiHopRAGAcquisitionError(f"{field} is not a regular file")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise MultiHopRAGAcquisitionError(f"{field} is not a regular file")
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise MultiHopRAGAcquisitionError(f"{field} mode drifted")


def _absolute_without_resolving(path: Path) -> Path:
    """Make a path absolute without hiding any symlink component."""

    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_ancestors(path: Path) -> Path:
    """Reject every existing ancestor symlink, including the direct parent."""

    absolute = _absolute_without_resolving(path)
    chain = (*reversed(absolute.parent.parents), absolute.parent)
    for ancestor in chain:
        try:
            info = os.lstat(ancestor)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise MultiHopRAGAcquisitionError("output ancestor cannot be attested") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MultiHopRAGAcquisitionError("output ancestor is unsafe")
    return absolute


def _safe_parent(path: Path) -> Path:
    absolute = _reject_symlink_ancestors(path)
    absolute.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Recheck the complete chain after mkdir so an existing indirect symlink is
    # never accepted merely because the final parent itself is a directory.
    return _reject_symlink_ancestors(absolute).parent


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    absolute = _absolute_without_resolving(path)
    parent = _safe_parent(absolute)
    parent_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(parent, parent_flags)
    except OSError as exc:
        raise MultiHopRAGAcquisitionError("exclusive output parent is unsafe") from exc
    parent_identity = os.fstat(parent_descriptor)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(
                absolute.name,
                flags,
                mode,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise MultiHopRAGAcquisitionError(
                "exclusive output already exists or cannot be created: "
                f"{absolute.name}"
            ) from exc
        # ``os.open(..., mode)`` is masked by the invoking process umask.  Set
        # the frozen mode on the still-exclusive fd before publishing bytes.
        try:
            os.fchmod(descriptor, mode)
        except BaseException:
            os.close(descriptor)
            raise
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            target_info = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(target_info.st_mode)
                or stat.S_IMODE(target_info.st_mode) != mode
            ):
                raise MultiHopRAGAcquisitionError("exclusive output mode drifted")
        current_parent = os.stat(parent, follow_symlinks=False)
        if (
            current_parent.st_dev != parent_identity.st_dev
            or current_parent.st_ino != parent_identity.st_ino
        ):
            raise MultiHopRAGAcquisitionError("exclusive output parent identity drifted")
        _reject_symlink_ancestors(absolute)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    mode: int,
) -> tuple[str, int]:
    raw = _canonical_bytes(payload) + b"\n"
    _write_exclusive(path, raw, mode=mode)
    return _sha256_bytes(raw), len(raw)


def _read_json_object(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    _require_regular_file(path, field=field)
    raw = path.read_bytes()
    payload = _strict_json(raw, field)
    if not isinstance(payload, dict):
        raise MultiHopRAGAcquisitionError(f"{field} root is not an object")
    return payload, raw


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    """Reject public row payloads while permitting aggregate audit wording."""

    forbidden_exact_keys = {
        "query",
        "answer",
        "fact",
        "url",
        "evidence_list",
        "question_type",
        "gold_article_ids",
        "identity_commitment_sha256",
        "source_record_commitment_sha256",
        "normalized_query_sha256",
        "source_row_ordinal",
        "items",
        "articles",
    }

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in forbidden_exact_keys:
                    raise MultiHopRAGAcquisitionError(
                        f"public payload contains private field {key}"
                    )
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)


def _private_file_binding(
    *, path: Path, semantic_sha256: str, file_sha256: str, byte_size: int
) -> dict[str, Any]:
    return {
        "relative_name": path.name,
        "file_sha256": file_sha256,
        "semantic_sha256": semantic_sha256,
        "byte_size": byte_size,
        "mode": "0600",
    }


def persist_private_payloads(
    *,
    corpus: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    paths: AcquisitionPaths,
) -> dict[str, Any]:
    """Persist exactly one corpus, four views, and three label packs."""

    if set(paths.block_views) != set(BLOCK_ORDER) or set(paths.block_labels) != {
        "A_form",
        "A_hold",
        "M_search",
    }:
        raise MultiHopRAGAcquisitionError("canonical private output path set drifted")
    if set(views) != set(BLOCK_ORDER) or set(labels) != {
        "A_form",
        "A_hold",
        "M_search",
    }:
        raise MultiHopRAGAcquisitionError("private payload set drifted")
    _validate_corpus_view(corpus)
    corpus_file_sha, corpus_size = _write_json_exclusive(
        paths.corpus_view, corpus, mode=0o600
    )
    block_rows: dict[str, Any] = {}
    for block in BLOCK_ORDER:
        _validate_block_view(views[block], expected_block=block)
        view_sha, view_size = _write_json_exclusive(
            paths.block_views[block], views[block], mode=0o600
        )
        row = {
            "count": BLOCK_COUNTS[block],
            "view": _private_file_binding(
                path=paths.block_views[block],
                semantic_sha256=str(views[block]["block_view_sha256"]),
                file_sha256=view_sha,
                byte_size=view_size,
            ),
            "labels": {"created": False},
        }
        if block != "F_search":
            _validate_block_labels(labels[block], expected_block=block)
            label_sha, label_size = _write_json_exclusive(
                paths.block_labels[block], labels[block], mode=0o600
            )
            row["labels"] = {
                "created": True,
                **_private_file_binding(
                    path=paths.block_labels[block],
                    semantic_sha256=str(labels[block]["block_labels_sha256"]),
                    file_sha256=label_sha,
                    byte_size=label_size,
                ),
            }
        block_rows[block] = row
    return {
        "corpus_view": _private_file_binding(
            path=paths.corpus_view,
            semantic_sha256=str(corpus["corpus_view_sha256"]),
            file_sha256=corpus_file_sha,
            byte_size=corpus_size,
        ),
        "blocks": block_rows,
        "private_file_count": 8,
        "F_search_label_pack_created": False,
        "M_search_files_sealed_until_promotion": True,
    }


def _read_bound_private_json(
    *, path: Path, file_sha256: str, field: str
) -> dict[str, Any]:
    _require_sha256(file_sha256, f"{field} file hash")
    _require_regular_file(path, field=field, mode=0o600)
    raw = path.read_bytes()
    if _sha256_bytes(raw) != file_sha256:
        raise MultiHopRAGAcquisitionError(f"{field} file hash drifted")
    payload = _strict_json(raw, field)
    if not isinstance(payload, dict):
        raise MultiHopRAGAcquisitionError(f"{field} root is not an object")
    return payload


def _validated_private_file_binding(
    value: object, *, expected_name: str, field: str
) -> dict[str, Any]:
    expected_keys = {
        "relative_name",
        "file_sha256",
        "semantic_sha256",
        "byte_size",
        "mode",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("relative_name") != expected_name
        or type(value.get("byte_size")) is not int
        or value["byte_size"] <= 0
        or value.get("mode") != "0600"
    ):
        raise MultiHopRAGAcquisitionError(f"{field} private binding drifted")
    _require_sha256(value.get("file_sha256"), f"{field} file hash")
    _require_sha256(value.get("semantic_sha256"), f"{field} semantic hash")
    return dict(value)


def _validated_private_pack_commitments(
    receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = receipt.get("private_pack_commitments")
    if not isinstance(value, Mapping) or set(value) != {
        "corpus_view",
        "blocks",
        "private_file_count",
        "F_search_label_pack_created",
        "M_search_files_sealed_until_promotion",
    }:
        raise MultiHopRAGAcquisitionError("private pack commitment schema drifted")
    if (
        value.get("private_file_count") != 8
        or value.get("F_search_label_pack_created") is not False
        or value.get("M_search_files_sealed_until_promotion") is not True
    ):
        raise MultiHopRAGAcquisitionError("private pack count or seal drifted")
    corpus = _validated_private_file_binding(
        value.get("corpus_view"),
        expected_name=Path(CORPUS_VIEW_RELATIVE).name,
        field="corpus view",
    )
    raw_blocks = value.get("blocks")
    if not isinstance(raw_blocks, Mapping) or set(raw_blocks) != set(BLOCK_ORDER):
        raise MultiHopRAGAcquisitionError("private block commitment set drifted")
    blocks: dict[str, dict[str, Any]] = {}
    for block in BLOCK_ORDER:
        row = raw_blocks[block]
        if not isinstance(row, Mapping) or set(row) != {"count", "view", "labels"}:
            raise MultiHopRAGAcquisitionError("private block commitment drifted")
        if row.get("count") != BLOCK_COUNTS[block]:
            raise MultiHopRAGAcquisitionError("private block commitment count drifted")
        view = _validated_private_file_binding(
            row.get("view"),
            expected_name=Path(BLOCK_VIEW_RELATIVES[block]).name,
            field=f"{block} view",
        )
        raw_labels = row.get("labels")
        if block == "F_search":
            if raw_labels != {"created": False}:
                raise MultiHopRAGAcquisitionError("F_search label commitment exists")
            labels: dict[str, Any] = {"created": False}
        else:
            if not isinstance(raw_labels, Mapping) or raw_labels.get("created") is not True:
                raise MultiHopRAGAcquisitionError(f"{block} label commitment is missing")
            label_body = dict(raw_labels)
            del label_body["created"]
            labels = {
                "created": True,
                **_validated_private_file_binding(
                    label_body,
                    expected_name=Path(BLOCK_LABEL_RELATIVES[block]).name,
                    field=f"{block} labels",
                ),
            }
        blocks[block] = {"count": row["count"], "view": view, "labels": labels}
    return corpus, blocks


def _validate_promotion_payload(
    promotion: Mapping[str, Any],
    *,
    acquisition_sha256: str,
    implementation_freeze_sha256: str,
) -> str:
    expected_keys = {
        "schema",
        "version",
        "status",
        "acquisition_sha256",
        "implementation_freeze_sha256",
        "f_search_policy_freeze_sha256",
        "a_hold_action_seal_sha256",
        "a_hold_output_archive_file_sha256",
        "a_hold_output_archive_semantic_sha256",
        "e0_action_id",
        "e0_policy_sha256",
        "e1_action_id",
        "e1_policy_sha256",
        "a_hold_item_count",
        "a_hold_exact_family_counts",
        "family_balanced_delta_total",
        "one_sided_magnitude_signflip_p",
        "promotion_rule_id",
        "challenger_promoted",
        "outcome_used_to_change_action_evaluator_or_threshold",
        "same_source_replay_authorized",
        "promotion_sha256",
    }
    if set(promotion) != expected_keys:
        raise MultiHopRAGAcquisitionError("promotion authorization schema drifted")
    verify_self_hash(
        promotion, hash_field="promotion_sha256", schema=PROMOTION_SCHEMA
    )
    delta = _fraction_or_int(
        promotion.get("family_balanced_delta_total"),
        field="promotion delta",
    )
    p_value = _fraction_or_int(
        promotion.get("one_sided_magnitude_signflip_p"),
        field="promotion p-value",
    )
    if (
        not isinstance(delta, Fraction)
        or delta <= 0
        or not isinstance(p_value, Fraction)
        or not 0 <= p_value <= Fraction(1, 10)
    ):
        raise MultiHopRAGAcquisitionError("promotion statistics do not pass the rule")
    e0_policy = _require_sha256(promotion.get("e0_policy_sha256"), "E0 policy")
    e1_policy = _require_sha256(promotion.get("e1_policy_sha256"), "E1 policy")
    for field in (
        "f_search_policy_freeze_sha256",
        "a_hold_action_seal_sha256",
        "a_hold_output_archive_file_sha256",
        "a_hold_output_archive_semantic_sha256",
    ):
        _require_sha256(promotion.get(field), f"promotion {field}")
    if (
        promotion.get("version") != "v1"
        or promotion.get("status") != "A_hold_challenger_promoted"
        or promotion.get("acquisition_sha256") != acquisition_sha256
        or promotion.get("implementation_freeze_sha256")
        != implementation_freeze_sha256
        or promotion.get("e0_action_id") not in AGENT_ACTION_IDS
        or promotion.get("e1_action_id") not in AGENT_ACTION_IDS
        or promotion.get("e0_action_id") == promotion.get("e1_action_id")
        or e0_policy == e1_policy
        or promotion.get("a_hold_item_count") != BLOCK_COUNTS["A_hold"]
        or promotion.get("a_hold_exact_family_counts")
        != {family: FAMILY_QUOTAS["A_hold"] for family in FAMILIES}
        or promotion.get("promotion_rule_id")
        != "positive_total_and_one_sided_magnitude_signflip_p_le_0.10"
        or promotion.get("challenger_promoted") is not True
        or promotion.get("outcome_used_to_change_action_evaluator_or_threshold")
        is not False
        or promotion.get("same_source_replay_authorized") is not False
    ):
        raise MultiHopRAGAcquisitionError("M_search promotion authorization is invalid")
    return str(promotion["promotion_sha256"])


def load_committed_acquisition_receipt(
    project: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _canonical_project(project)
    path = root / PUBLIC_RECEIPT_RELATIVE
    payload, raw = _read_json_object(path, field="acquisition receipt")
    expected_keys = {
        "schema",
        "version",
        "status",
        "dataset_identity",
        "custody_binding",
        "attempt_marker_sha256",
        "source_bindings",
        "source_qualification",
        "selection_qualification",
        "private_pack_commitments",
        "label_isolation",
        "public_candidate_identity_query_answer_fact_URL_evidence_or_gold",
        "same_source_replay_secret_rotation_resample_replacement_or_retry_authorized",
        "acquisition_sha256",
    }
    if set(payload) != expected_keys:
        raise MultiHopRAGAcquisitionError("acquisition receipt schema drifted")
    verify_self_hash(
        payload, hash_field="acquisition_sha256", schema=PUBLIC_RECEIPT_SCHEMA
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "private_four_block_pack_formed"
        or payload.get(
            "public_candidate_identity_query_answer_fact_URL_evidence_or_gold"
        )
        is not False
        or payload.get(
            "same_source_replay_secret_rotation_resample_replacement_or_retry_authorized"
        )
        is not False
    ):
        raise MultiHopRAGAcquisitionError("acquisition receipt contract drifted")
    _assert_public_safe(payload)
    _validated_private_pack_commitments(payload)
    head, oids = _verify_head_blobs(
        project=root, relative_paths=(PUBLIC_RECEIPT_RELATIVE,)
    )
    if not hmac.compare_digest(
        _git_blob_sha1(raw), oids[PUBLIC_RECEIPT_RELATIVE]
    ):
        raise MultiHopRAGAcquisitionError(
            "parsed acquisition receipt does not match committed HEAD blob"
        )
    custody, verified_custody_binding = verify_committed_source_custody(root)
    custody_binding = payload.get("custody_binding")
    source_bindings = payload.get("source_bindings")
    if (
        not isinstance(custody_binding, Mapping)
        or set(custody_binding)
        != {
            "source_custody_sha256",
            "source_custody_file_sha256",
            "source_custody_git_blob_sha1",
            "custody_verified_at_git_HEAD",
        }
        or _HEX40.fullmatch(
            str(custody_binding.get("custody_verified_at_git_HEAD", ""))
        )
        is None
        or any(
            custody_binding.get(field) != verified_custody_binding.get(field)
            for field in (
                "source_custody_sha256",
                "source_custody_file_sha256",
                "source_custody_git_blob_sha1",
            )
        )
        or source_bindings != custody.get("source_bindings")
        or payload.get("dataset_identity") != custody.get("dataset_identity")
    ):
        raise MultiHopRAGAcquisitionError(
            "acquisition receipt breaks the committed source custody chain"
        )
    failure_path = root / FAILURE_RELATIVE
    if failure_path.exists() or failure_path.is_symlink():
        raise MultiHopRAGAcquisitionError(
            "terminal failure coexists with acquisition receipt"
        )
    marker_path = root / ACQUISITION_MARKER_RELATIVE
    _require_regular_file(marker_path, field="acquisition marker", mode=0o600)
    marker_raw = marker_path.read_bytes()
    marker = _strict_json(marker_raw, "acquisition marker")
    if not isinstance(marker, Mapping):
        raise MultiHopRAGAcquisitionError("acquisition marker root drifted")
    marker_sha = _validate_marker(
        marker, phase="formal_source_parse_and_block_formation"
    )
    if (
        not isinstance(custody_binding, Mapping)
        or not isinstance(source_bindings, Mapping)
        or not isinstance(source_bindings.get("query_source"), Mapping)
        or not isinstance(source_bindings.get("corpus_source"), Mapping)
    ):
        raise MultiHopRAGAcquisitionError("receipt marker inputs drifted")
    expected_marker_bindings = {
        "source_custody_sha256": _require_sha256(
            custody_binding.get("source_custody_sha256"), "receipt custody"
        ),
        "query_source_sha256": _require_sha256(
            source_bindings["query_source"].get("sha256"), "receipt query source"
        ),
        "corpus_source_sha256": _require_sha256(
            source_bindings["corpus_source"].get("sha256"), "receipt corpus source"
        ),
    }
    if (
        marker_sha != payload.get("attempt_marker_sha256")
        or marker.get("bindings") != expected_marker_bindings
    ):
        raise MultiHopRAGAcquisitionError("acquisition marker receipt binding drifted")
    return payload, {
        "file_sha256": _sha256_bytes(raw),
        "git_blob_sha1": oids[PUBLIC_RECEIPT_RELATIVE],
        "verified_at_git_HEAD": head,
    }


def _validate_output_top5(value: object, *, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(type(index) is not int for index in value)
        or len(set(value)) != 5
        or any(not 0 <= index < CORPUS_RECORD_COUNT for index in value)
    ):
        raise MultiHopRAGAcquisitionError(f"{field} top5 drifted")
    return tuple(value)


def _validate_method_output(value: object, *, method: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "method",
        "terminal",
        "output_top5",
        "output_sha256",
    }:
        raise MultiHopRAGAcquisitionError(f"{method} terminal output schema drifted")
    verify_self_hash(value, hash_field="output_sha256")
    _validate_output_top5(value.get("output_top5"), field=method)
    if value.get("method") != method or value.get("terminal") is not True:
        raise MultiHopRAGAcquisitionError(f"{method} output is not terminal")
    return dict(value)


def _fraction_or_int(value: object, *, field: str) -> Fraction | int:
    if type(value) is int:
        return value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(type(part) is int for part in value)
        and value[1] > 0
    ):
        fraction = Fraction(value[0], value[1])
        if [fraction.numerator, fraction.denominator] != value:
            raise MultiHopRAGAcquisitionError(f"{field} fraction is noncanonical")
        return fraction
    raise MultiHopRAGAcquisitionError(f"{field} number drifted")


def encode_typed_action_trace(trace: object) -> dict[str, Any]:
    """Canonical full ActionTrace envelope used by the formal stage writer."""

    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        ActionTrace,
        VERSION as TYPED_CORE_VERSION,
        recompute_action_trace_sha256,
    )

    if not isinstance(trace, ActionTrace):
        raise MultiHopRAGAcquisitionError("typed action trace has the wrong type")
    observed = recompute_action_trace_sha256(trace)
    if not hmac.compare_digest(observed, trace.trace_sha256):
        raise MultiHopRAGAcquisitionError("typed action trace is not self-consistent")

    def number(value: Fraction | int) -> int | list[int]:
        return (
            [value.numerator, value.denominator]
            if isinstance(value, Fraction)
            else value
        )

    receipt = {
        "action_id": trace.action_id,
        "causal": [
            trace.causal.necessary_count,
            number(trace.causal.necessary_fraction),
            number(trace.causal.minimum_leave_one_out_loss),
            number(trace.causal.minimum_replacement_loss),
            number(trace.causal.path_connectivity),
        ],
        "core": list(trace.core),
        "core_quality": [number(value) for value in trace.core_quality],
        "coverage": [trace.coverage.covered, trace.coverage.total],
        "coverage_slot_keys": list(trace.coverage.slot_keys),
        "coverage_covered_slot_keys": list(
            trace.coverage.covered_slot_keys
        ),
        "e0": [number(value) for value in trace.e0_key],
        "e1": [number(value) for value in trace.e1_key],
        "extension_scan_count": trace.extension_scan_count,
        "graph_sha256": trace.graph_sha256,
        "output_top5": list(trace.output_top5),
        "ordered_pair_scan_count": trace.ordered_pair_scan_count,
        "plan_sha256": trace.plan_sha256,
        "query_sha256": trace.query_sha256,
        "relevance_sha256": trace.relevance_sha256,
        "version": TYPED_CORE_VERSION,
    }
    envelope = {
        "action_id": trace.action_id,
        "terminal": True,
        "trace": receipt,
        "trace_sha256": observed,
    }
    _decode_and_verify_action_trace(
        envelope, expected_action_id=trace.action_id
    )
    return envelope


def _validate_dense_relevance_ints(
    value: object,
) -> tuple[tuple[int, ...], str]:
    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        INTEGER_SCALE,
    )

    if (
        not isinstance(value, (list, tuple))
        or len(value) != CORPUS_RECORD_COUNT
        or any(
            type(score) is not int
            or not -INTEGER_SCALE <= score <= INTEGER_SCALE
            for score in value
        )
    ):
        raise MultiHopRAGAcquisitionError("dense relevance vector drifted")
    rows = tuple(value)
    return rows, stable_hash(
        {"integer_scale": INTEGER_SCALE, "values": list(rows)}
    )


def _validate_stage_runtime_binding(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(
        STAGE_RUNTIME_BINDING_KEYS
    ):
        raise MultiHopRAGAcquisitionError("stage runtime binding schema drifted")
    return {
        field: _require_sha256(value.get(field), f"stage runtime {field}")
        for field in STAGE_RUNTIME_BINDING_KEYS
    }


def build_stage_output_record(
    *,
    block: str,
    ordinal: int,
    view_sha256: str,
    dense_relevance_ints: Sequence[int],
    raw_top5: Sequence[int],
    hipporag_top5: Sequence[int],
    action_traces: Sequence[object],
) -> dict[str, Any]:
    """Build one canonical gold-free RAW/Hippo/six-Agent terminal record."""

    if block not in BLOCK_ORDER or type(ordinal) is not int or ordinal < 0:
        raise MultiHopRAGAcquisitionError("stage output record identity is invalid")
    _require_sha256(view_sha256, "stage record view")

    def method_output(method: str, output: Sequence[int]) -> dict[str, Any]:
        top5 = list(_validate_output_top5(list(output), field=method))
        return _with_self_hash(
            {"method": method, "terminal": True, "output_top5": top5},
            "output_sha256",
        )

    relevance, relevance_sha256 = _validate_dense_relevance_ints(
        dense_relevance_ints
    )
    traces = [encode_typed_action_trace(trace) for trace in action_traces]
    if [row["action_id"] for row in traces] != list(AGENT_ACTION_IDS):
        raise MultiHopRAGAcquisitionError("stage record action order drifted")
    body = {
        "schema": f"multihoprag_{block.casefold()}_stage_output_record_v1",
        "block": block,
        "ordinal": ordinal,
        "view_sha256": view_sha256,
        "dense_relevance_ints": list(relevance),
        "relevance_sha256": relevance_sha256,
        "raw_output": method_output("RAW", raw_top5),
        "hipporag_output": method_output("HippoRAG", hipporag_top5),
        "agent_action_traces": traces,
    }
    payload = _with_self_hash(body, "record_sha256")
    _validate_stage_record(
        payload,
        block=block,
        ordinal=ordinal,
        expected_view_sha256=view_sha256,
    )
    return payload


def _decode_and_rebuild_action_trace(
    value: object, *, expected_action_id: str
) -> tuple[dict[str, Any], str, object]:
    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        ActionTrace,
        CausalSignature,
        CoverageSignature,
        VERSION as TYPED_CORE_VERSION,
        recompute_action_trace_sha256,
    )

    if not isinstance(value, Mapping) or set(value) != {
        "action_id",
        "terminal",
        "trace",
        "trace_sha256",
    }:
        raise MultiHopRAGAcquisitionError("Agent action trace envelope drifted")
    receipt = value.get("trace")
    expected_receipt_keys = {
        "action_id",
        "causal",
        "core",
        "core_quality",
        "coverage",
        "coverage_covered_slot_keys",
        "coverage_slot_keys",
        "e0",
        "e1",
        "extension_scan_count",
        "graph_sha256",
        "output_top5",
        "ordered_pair_scan_count",
        "plan_sha256",
        "query_sha256",
        "relevance_sha256",
        "version",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_receipt_keys:
        raise MultiHopRAGAcquisitionError("typed ActionTrace receipt schema drifted")
    if (
        value.get("action_id") != expected_action_id
        or value.get("terminal") is not True
        or receipt.get("action_id") != expected_action_id
        or receipt.get("version") != TYPED_CORE_VERSION
    ):
        raise MultiHopRAGAcquisitionError("typed ActionTrace identity drifted")
    core = receipt.get("core")
    output = receipt.get("output_top5")
    coverage = receipt.get("coverage")
    slot_keys = receipt.get("coverage_slot_keys")
    covered_slot_keys = receipt.get("coverage_covered_slot_keys")
    causal = receipt.get("causal")
    quality = receipt.get("core_quality")
    e0 = receipt.get("e0")
    e1 = receipt.get("e1")
    if (
        not isinstance(core, list)
        or len(core) != 4
        or any(type(index) is not int for index in core)
        or len(set(core)) != 4
        or any(not 0 <= index < CORPUS_RECORD_COUNT for index in core)
        or not isinstance(output, list)
        or tuple(output[:4]) != tuple(core)
        or not isinstance(coverage, list)
        or len(coverage) != 2
        or not isinstance(causal, list)
        or len(causal) != 5
        or type(causal[0]) is not int
        or not 0 <= causal[0] <= 4
        or not isinstance(quality, list)
        or not quality
        or not isinstance(e0, list)
        or len(e0) != 3
        or not isinstance(e1, list)
        or len(e1) != 6
    ):
        raise MultiHopRAGAcquisitionError("typed ActionTrace shape drifted")
    covered, total = coverage
    coverage_value = (
        Fraction(covered, total)
        if type(covered) is int and type(total) is int and total > 0
        else None
    )
    if (
        type(covered) is not int
        or type(total) is not int
        or total <= 0
        or not 0 <= covered <= total
        or not isinstance(coverage_value, Fraction)
        or not isinstance(slot_keys, list)
        or len(slot_keys) != total
        or any(not isinstance(key, str) or not key for key in slot_keys)
        or len(set(slot_keys)) != len(slot_keys)
        or not isinstance(covered_slot_keys, list)
        or len(covered_slot_keys) != covered
        or any(
            not isinstance(key, str) or key not in slot_keys
            for key in covered_slot_keys
        )
        or len(set(covered_slot_keys)) != len(covered_slot_keys)
    ):
        raise MultiHopRAGAcquisitionError("typed ActionTrace coverage drifted")
    output_top5 = _validate_output_top5(output, field=f"Agent {expected_action_id}")
    quality_values = tuple(
        _fraction_or_int(row, field="core quality") for row in quality
    )
    causal_values = (
        causal[0],
        *(
            _fraction_or_int(row, field="causal signature")
            for row in causal[1:]
        ),
    )
    if any(not isinstance(row, Fraction) for row in causal_values[1:]):
        raise MultiHopRAGAcquisitionError("causal signature fractions drifted")
    e0_values = tuple(_fraction_or_int(row, field="E0 key") for row in e0)
    e1_values = tuple(_fraction_or_int(row, field="E1 key") for row in e1)
    necessary_fraction = causal_values[1]
    if (
        necessary_fraction != Fraction(causal_values[0], 4)
        or e1_values
        != (
            causal_values[1],
            causal_values[2],
            causal_values[4],
            *e0_values,
        )
        or receipt.get("ordered_pair_scan_count")
        != CORPUS_RECORD_COUNT * (CORPUS_RECORD_COUNT - 1)
        or receipt.get("extension_scan_count")
        != (CORPUS_RECORD_COUNT - 2) + (CORPUS_RECORD_COUNT - 3)
    ):
        raise MultiHopRAGAcquisitionError("typed ActionTrace causal contract drifted")
    hashes = {
        field: _require_sha256(receipt.get(field), f"ActionTrace {field}")
        for field in (
            "graph_sha256",
            "plan_sha256",
            "query_sha256",
            "relevance_sha256",
        )
    }
    typed = ActionTrace(
        action_id=expected_action_id,
        output_top5=output_top5,
        core=tuple(core),
        core_quality=quality_values,
        coverage=CoverageSignature(
            covered=covered,
            total=total,
            value=coverage_value,
            slot_keys=tuple(slot_keys),
            covered_slot_keys=tuple(covered_slot_keys),
        ),
        causal=CausalSignature(
            necessary_count=causal_values[0],
            necessary_fraction=causal_values[1],
            minimum_leave_one_out_loss=causal_values[2],
            minimum_replacement_loss=causal_values[3],
            path_connectivity=causal_values[4],
        ),
        e0_key=e0_values,
        e1_key=e1_values,
        ordered_pair_scan_count=receipt["ordered_pair_scan_count"],
        extension_scan_count=receipt["extension_scan_count"],
        graph_sha256=hashes["graph_sha256"],
        plan_sha256=hashes["plan_sha256"],
        query_sha256=hashes["query_sha256"],
        relevance_sha256=hashes["relevance_sha256"],
        trace_sha256=_require_sha256(value.get("trace_sha256"), "typed trace"),
    )
    observed = recompute_action_trace_sha256(typed)
    if not hmac.compare_digest(observed, typed.trace_sha256):
        raise MultiHopRAGAcquisitionError("typed ActionTrace SHA256 drifted")
    return dict(value), observed, typed


def _decode_and_verify_action_trace(
    value: object, *, expected_action_id: str
) -> tuple[dict[str, Any], str]:
    checked, observed, _typed = _decode_and_rebuild_action_trace(
        value, expected_action_id=expected_action_id
    )
    return checked, observed


def _typed_query_sha256(query: object) -> str:
    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        normalize_text,
    )

    if not isinstance(query, str):
        raise MultiHopRAGAcquisitionError("stage-bound query is invalid")
    return hashlib.sha256(normalize_text(query).encode("utf-8")).hexdigest()


def _validate_stage_record(
    value: object,
    *,
    block: str,
    ordinal: int,
    expected_view_sha256: str,
    expected_query_sha256: str | None = None,
    expected_graph_sha256: str | None = None,
) -> tuple[dict[str, Any], str, str, tuple[str, ...]]:
    expected_keys = {
        "schema",
        "block",
        "ordinal",
        "view_sha256",
        "dense_relevance_ints",
        "relevance_sha256",
        "raw_output",
        "hipporag_output",
        "agent_action_traces",
        "record_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise MultiHopRAGAcquisitionError(f"{block} stage record schema drifted")
    verify_self_hash(value, hash_field="record_sha256")
    if (
        value.get("schema")
        != f"multihoprag_{block.casefold()}_stage_output_record_v1"
        or value.get("block") != block
        or value.get("ordinal") != ordinal
        or value.get("view_sha256") != expected_view_sha256
    ):
        raise MultiHopRAGAcquisitionError(f"{block} stage record identity drifted")
    raw = _validate_method_output(value.get("raw_output"), method="RAW")
    relevance, relevance_sha256 = _validate_dense_relevance_ints(
        value.get("dense_relevance_ints")
    )
    if value.get("relevance_sha256") != relevance_sha256:
        raise MultiHopRAGAcquisitionError("stage relevance receipt drifted")
    expected_raw_top5 = tuple(
        sorted(
            range(CORPUS_RECORD_COUNT),
            key=lambda index: (-relevance[index], index),
        )[:5]
    )
    if tuple(raw["output_top5"]) != expected_raw_top5:
        raise MultiHopRAGAcquisitionError(
            "RAW output differs from archived dense relevance"
        )
    hippo = _validate_method_output(
        value.get("hipporag_output"), method="HippoRAG"
    )
    traces = value.get("agent_action_traces")
    if not isinstance(traces, list) or len(traces) != len(AGENT_ACTION_IDS):
        raise MultiHopRAGAcquisitionError("Agent six-action trace row drifted")
    trace_hashes: list[str] = []
    input_receipts: set[tuple[str, str, str, str]] = set()
    for action_id, trace in zip(AGENT_ACTION_IDS, traces, strict=True):
        checked_trace, trace_hash = _decode_and_verify_action_trace(
            trace, expected_action_id=action_id
        )
        trace_hashes.append(trace_hash)
        receipt = checked_trace["trace"]
        input_receipts.add(
            (
                str(receipt["graph_sha256"]),
                str(receipt["plan_sha256"]),
                str(receipt["query_sha256"]),
                str(receipt["relevance_sha256"]),
            )
        )
    if len(input_receipts) != 1:
        raise MultiHopRAGAcquisitionError(
            "Agent actions do not share one stage observation input"
        )
    observed_query_sha256 = next(iter(input_receipts))[2]
    observed_relevance_sha256 = next(iter(input_receipts))[3]
    observed_graph_sha256 = next(iter(input_receipts))[0]
    if (
        expected_query_sha256 is not None
        and observed_query_sha256
        != _require_sha256(expected_query_sha256, "stage-bound query")
    ):
        raise MultiHopRAGAcquisitionError(
            "Agent action query does not match the stage view"
        )
    if observed_relevance_sha256 != relevance_sha256:
        raise MultiHopRAGAcquisitionError(
            "Agent actions do not bind archived dense relevance"
        )
    if (
        expected_graph_sha256 is not None
        and observed_graph_sha256
        != _require_sha256(expected_graph_sha256, "stage-bound graph")
    ):
        raise MultiHopRAGAcquisitionError(
            "Agent actions do not bind the stage runtime graph"
        )
    return (
        dict(value),
        str(raw["output_sha256"]),
        str(hippo["output_sha256"]),
        tuple(trace_hashes),
    )


def _authorize_block_private_access(
    *, root: Path, block: str, acquisition: Mapping[str, Any]
) -> None:
    """Run a block capability gate before its private path is statted."""

    acquisition_sha256 = _require_sha256(
        acquisition.get("acquisition_sha256"), "private block acquisition"
    )
    if block == "A_hold":
        load_f_search_policy_freeze(project=root, acquisition=acquisition)
    elif block == "M_search":
        load_committed_promotion_authorization(
            project=root, acquisition_sha256=acquisition_sha256
        )


def load_stage_output_archive(
    *, project: Path, block: str, acquisition: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the actual canonical terminal output archive for one stage."""

    if block not in BLOCK_ORDER:
        raise MultiHopRAGAcquisitionError("stage output block is invalid")
    root = _canonical_project(project)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "stage archive acquisition"
    )
    _authorize_block_private_access(
        root=root, block=block, acquisition=acquisition
    )
    corpus_binding, blocks = _validated_private_pack_commitments(acquisition)
    view_binding = blocks[block]["view"]
    view = _read_bound_private_json(
        path=root / BLOCK_VIEW_RELATIVES[block],
        file_sha256=str(view_binding["file_sha256"]),
        field=f"{block} stage-bound view",
    )
    _validate_block_view(view, expected_block=block)
    if view.get("block_view_sha256") != view_binding["semantic_sha256"]:
        raise MultiHopRAGAcquisitionError("stage-bound view semantic hash drifted")
    path = root / STAGE_OUTPUT_ARCHIVE_RELATIVES[block]
    _require_regular_file(path, field=f"{block} output archive", mode=0o600)
    raw = path.read_bytes()
    payload = _strict_json(raw, f"{block} output archive")
    expected_keys = {
        "schema",
        "version",
        "block",
        "acquisition_sha256",
        "corpus_view_semantic_sha256",
        "block_view_semantic_sha256",
        "stage_runtime_binding",
        "item_count",
        "raw_output_set_sha256",
        "hipporag_output_set_sha256",
        "agent_complete_six_action_trace_matrix_sha256",
        "all_three_methods_terminal",
        "gold_fields_included",
        "records",
        "stage_output_archive_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise MultiHopRAGAcquisitionError(f"{block} output archive schema drifted")
    semantic = verify_self_hash(
        payload,
        hash_field="stage_output_archive_sha256",
        schema=STAGE_OUTPUT_ARCHIVE_SCHEMAS[block],
    )
    records = payload.get("records")
    runtime_binding = _validate_stage_runtime_binding(
        payload.get("stage_runtime_binding")
    )
    for field in (
        "raw_output_set_sha256",
        "hipporag_output_set_sha256",
        "agent_complete_six_action_trace_matrix_sha256",
    ):
        _require_sha256(payload.get(field), f"{block} archive {field}")
    if (
        payload.get("version") != "v1"
        or payload.get("block") != block
        or payload.get("acquisition_sha256") != acquisition_sha
        or payload.get("corpus_view_semantic_sha256")
        != corpus_binding["semantic_sha256"]
        or payload.get("block_view_semantic_sha256")
        != view_binding["semantic_sha256"]
        or payload.get("item_count") != BLOCK_COUNTS[block]
        or payload.get("all_three_methods_terminal") is not True
        or payload.get("gold_fields_included") is not False
        or not isinstance(records, list)
        or len(records) != BLOCK_COUNTS[block]
    ):
        raise MultiHopRAGAcquisitionError(f"{block} output archive binding drifted")
    validated_records: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    hippo_hashes: list[str] = []
    trace_matrix: list[list[str]] = []
    for ordinal, record in enumerate(records):
        checked, raw_hash, hippo_hash, trace_hashes = _validate_stage_record(
            record,
            block=block,
            ordinal=ordinal,
            expected_view_sha256=stable_hash(view["items"][ordinal]),
            expected_query_sha256=_typed_query_sha256(
                view["items"][ordinal]["query"]
            ),
            expected_graph_sha256=runtime_binding["graph_sha256"],
        )
        validated_records.append(checked)
        raw_hashes.append(raw_hash)
        hippo_hashes.append(hippo_hash)
        trace_matrix.append(list(trace_hashes))
    if (
        stable_hash(raw_hashes) != payload.get("raw_output_set_sha256")
        or stable_hash(hippo_hashes) != payload.get("hipporag_output_set_sha256")
        or stable_hash(trace_matrix)
        != payload.get("agent_complete_six_action_trace_matrix_sha256")
        or len({row["view_sha256"] for row in validated_records})
        != BLOCK_COUNTS[block]
        or len({row["record_sha256"] for row in validated_records})
        != BLOCK_COUNTS[block]
    ):
        raise MultiHopRAGAcquisitionError(
            f"{block} output archive derived commitments drifted"
        )
    return dict(payload), {
        "file_sha256": _sha256_bytes(raw),
        "semantic_sha256": semantic,
        "byte_size": len(raw),
        "mode": "0600",
    }


def create_stage_output_archive_once(
    *,
    project: Path,
    block: str,
    records: Sequence[Mapping[str, Any]],
    stage_runtime_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonical O_EXCL writer for one complete gold-free stage archive."""

    if block not in BLOCK_ORDER:
        raise MultiHopRAGAcquisitionError("stage output block is invalid")
    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "stage archive acquisition"
    )
    _authorize_block_private_access(
        root=root, block=block, acquisition=acquisition
    )
    exact_runtime_binding = _validate_stage_runtime_binding(
        stage_runtime_binding
    )
    corpus_binding, blocks = _validated_private_pack_commitments(acquisition)
    view_binding = blocks[block]["view"]
    view = _read_bound_private_json(
        path=root / BLOCK_VIEW_RELATIVES[block],
        file_sha256=str(view_binding["file_sha256"]),
        field=f"{block} archive writer view",
    )
    _validate_block_view(view, expected_block=block)
    if len(records) != BLOCK_COUNTS[block]:
        raise MultiHopRAGAcquisitionError("stage archive writer count drifted")
    exact_records: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    hippo_hashes: list[str] = []
    trace_matrix: list[list[str]] = []
    for ordinal, record in enumerate(records):
        checked, raw_hash, hippo_hash, trace_hashes = _validate_stage_record(
            record,
            block=block,
            ordinal=ordinal,
            expected_view_sha256=stable_hash(view["items"][ordinal]),
            expected_query_sha256=_typed_query_sha256(
                view["items"][ordinal]["query"]
            ),
            expected_graph_sha256=exact_runtime_binding["graph_sha256"],
        )
        exact_records.append(checked)
        raw_hashes.append(raw_hash)
        hippo_hashes.append(hippo_hash)
        trace_matrix.append(list(trace_hashes))
    payload = _with_self_hash(
        {
            "schema": STAGE_OUTPUT_ARCHIVE_SCHEMAS[block],
            "version": "v1",
            "block": block,
            "acquisition_sha256": acquisition_sha,
            "corpus_view_semantic_sha256": corpus_binding["semantic_sha256"],
            "block_view_semantic_sha256": view_binding["semantic_sha256"],
            "stage_runtime_binding": exact_runtime_binding,
            "item_count": BLOCK_COUNTS[block],
            "raw_output_set_sha256": stable_hash(raw_hashes),
            "hipporag_output_set_sha256": stable_hash(hippo_hashes),
            "agent_complete_six_action_trace_matrix_sha256": stable_hash(
                trace_matrix
            ),
            "all_three_methods_terminal": True,
            "gold_fields_included": False,
            "records": exact_records,
        },
        "stage_output_archive_sha256",
    )
    _write_json_exclusive(
        root / STAGE_OUTPUT_ARCHIVE_RELATIVES[block], payload, mode=0o600
    )
    return load_stage_output_archive(
        project=root, block=block, acquisition=acquisition
    )


def _recompute_policy_selections(
    archive: Mapping[str, Any], *, block: str
) -> tuple[object, object, bool]:
    """Rebuild exact typed observations and select both policies for one stage."""

    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        ACTION_IDS,
        EvaluationObservation,
        MultiHopRAGTypedOperatorV2Error,
        policies_identifiable,
        select_global_policy,
    )

    records = archive.get("records")
    if (
        tuple(ACTION_IDS) != AGENT_ACTION_IDS
        or not isinstance(records, list)
        or block not in {"A_form", "F_search"}
        or len(records) != BLOCK_COUNTS[block]
    ):
        raise MultiHopRAGAcquisitionError(f"{block} policy trace matrix drifted")
    observations: list[object] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise MultiHopRAGAcquisitionError(
                f"{block} policy stage record drifted"
            )
        envelopes = record.get("agent_action_traces")
        if not isinstance(envelopes, list) or len(envelopes) != len(ACTION_IDS):
            raise MultiHopRAGAcquisitionError(
                f"{block} policy action matrix drifted"
            )
        traces: dict[str, object] = {}
        for action_id, envelope in zip(ACTION_IDS, envelopes, strict=True):
            _checked, _trace_sha256, typed = _decode_and_rebuild_action_trace(
                envelope, expected_action_id=action_id
            )
            traces[action_id] = typed
        observations.append(EvaluationObservation(traces_by_action=traces))
    try:
        e0 = select_global_policy(
            evaluator_id="E0_INDEPENDENT_V2", observations=observations
        )
        e1 = select_global_policy(
            evaluator_id="E1_CAUSAL_NECESSITY_V2", observations=observations
        )
        identifiable = policies_identifiable(e0, e1, observations)
    except MultiHopRAGTypedOperatorV2Error as exc:
        raise MultiHopRAGAcquisitionError(
            f"{block} policy selection could not be authoritatively recomputed"
        ) from exc
    return e0, e1, identifiable


def _recompute_f_search_policy_selections(
    archive: Mapping[str, Any],
) -> tuple[object, object, bool]:
    return _recompute_policy_selections(archive, block="F_search")


def _recompute_a_form_policy_selections(
    archive: Mapping[str, Any],
) -> tuple[object, object, bool]:
    return _recompute_policy_selections(archive, block="A_form")


def load_a_form_policy_freeze(
    *, project: Path, acquisition: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the pre-label A_form descriptive selection freeze."""

    root = _canonical_project(project)
    if acquisition is None:
        acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "A_form policy acquisition"
    )
    _corpus_binding, block_bindings = _validated_private_pack_commitments(
        acquisition
    )
    archive, archive_binding = load_stage_output_archive(
        project=root, block="A_form", acquisition=acquisition
    )
    action_seal = load_action_seal(
        project=root, block="A_form", acquisition=acquisition
    )
    e0_selection, e1_selection, identifiable = (
        _recompute_a_form_policy_selections(archive)
    )
    path = root / A_FORM_POLICY_FREEZE_RELATIVE
    _require_regular_file(path, field="A_form policy freeze", mode=0o644)
    raw = path.read_bytes()
    payload = _strict_json(raw, "A_form policy freeze")
    expected_keys = {
        "schema",
        "version",
        "status",
        "acquisition_sha256",
        "a_form_view_file_sha256",
        "a_form_view_semantic_sha256",
        "a_form_item_count",
        "a_form_output_archive_file_sha256",
        "a_form_output_archive_semantic_sha256",
        "a_form_action_seal_sha256",
        "complete_a_form_trace_matrix_receipt_sha256",
        "e0_action_id",
        "e0_policy_sha256",
        "e1_action_id",
        "e1_policy_sha256",
        "policies_identifiable",
        "selection_purpose",
        "A_form_gold_opened_before_policy_freeze",
        "created_with_O_EXCL",
        "same_stage_replay_or_policy_reselection_authorized",
        "a_form_policy_freeze_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise MultiHopRAGAcquisitionError(
            "A_form policy freeze schema drifted"
        )
    verify_self_hash(
        payload,
        hash_field="a_form_policy_freeze_sha256",
        schema=A_FORM_POLICY_FREEZE_SCHEMA,
    )
    a_form_view = block_bindings["A_form"]["view"]
    if (
        payload.get("version") != "v1"
        or payload.get("status")
        != "A_form_prelabel_descriptive_policies_frozen"
        or payload.get("acquisition_sha256") != acquisition_sha
        or payload.get("a_form_view_file_sha256")
        != a_form_view["file_sha256"]
        or payload.get("a_form_view_semantic_sha256")
        != a_form_view["semantic_sha256"]
        or payload.get("a_form_item_count") != BLOCK_COUNTS["A_form"]
        or payload.get("a_form_output_archive_file_sha256")
        != archive_binding["file_sha256"]
        or payload.get("a_form_output_archive_semantic_sha256")
        != archive_binding["semantic_sha256"]
        or payload.get("a_form_action_seal_sha256")
        != action_seal["action_seal_sha256"]
        or payload.get("complete_a_form_trace_matrix_receipt_sha256")
        != archive["agent_complete_six_action_trace_matrix_sha256"]
        or payload.get("e0_action_id") != e0_selection.action_id
        or payload.get("e0_policy_sha256") != e0_selection.selection_sha256
        or payload.get("e1_action_id") != e1_selection.action_id
        or payload.get("e1_policy_sha256") != e1_selection.selection_sha256
        or payload.get("policies_identifiable") is not identifiable
        or payload.get("selection_purpose")
        != "prelabel_descriptive_only_not_F_policy"
        or payload.get("A_form_gold_opened_before_policy_freeze") is not False
        or payload.get("created_with_O_EXCL") is not True
        or payload.get("same_stage_replay_or_policy_reselection_authorized")
        is not False
    ):
        raise MultiHopRAGAcquisitionError(
            "A_form policy freeze binding drifted"
        )
    return dict(payload)


def create_a_form_policy_freeze_once(*, project: Path) -> dict[str, Any]:
    """Freeze A_form evaluator selections before its gold is opened."""

    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "A_form policy acquisition"
    )
    _corpus_binding, blocks = _validated_private_pack_commitments(acquisition)
    a_form_view = blocks["A_form"]["view"]
    archive, archive_binding = load_stage_output_archive(
        project=root, block="A_form", acquisition=acquisition
    )
    action_seal = load_action_seal(
        project=root, block="A_form", acquisition=acquisition
    )
    e0, e1, identifiable = _recompute_a_form_policy_selections(archive)
    payload = _with_self_hash(
        {
            "schema": A_FORM_POLICY_FREEZE_SCHEMA,
            "version": "v1",
            "status": "A_form_prelabel_descriptive_policies_frozen",
            "acquisition_sha256": acquisition_sha,
            "a_form_view_file_sha256": a_form_view["file_sha256"],
            "a_form_view_semantic_sha256": a_form_view["semantic_sha256"],
            "a_form_item_count": BLOCK_COUNTS["A_form"],
            "a_form_output_archive_file_sha256": archive_binding[
                "file_sha256"
            ],
            "a_form_output_archive_semantic_sha256": archive_binding[
                "semantic_sha256"
            ],
            "a_form_action_seal_sha256": action_seal["action_seal_sha256"],
            "complete_a_form_trace_matrix_receipt_sha256": archive[
                "agent_complete_six_action_trace_matrix_sha256"
            ],
            "e0_action_id": e0.action_id,
            "e0_policy_sha256": e0.selection_sha256,
            "e1_action_id": e1.action_id,
            "e1_policy_sha256": e1.selection_sha256,
            "policies_identifiable": identifiable,
            "selection_purpose": "prelabel_descriptive_only_not_F_policy",
            "A_form_gold_opened_before_policy_freeze": False,
            "created_with_O_EXCL": True,
            "same_stage_replay_or_policy_reselection_authorized": False,
        },
        "a_form_policy_freeze_sha256",
    )
    _write_json_exclusive(
        root / A_FORM_POLICY_FREEZE_RELATIVE, payload, mode=0o644
    )
    return load_a_form_policy_freeze(project=root, acquisition=acquisition)


def load_f_search_policy_freeze(
    *, project: Path, acquisition: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the fixed O_EXCL F terminal policy freeze before A_hold gold."""

    root = _canonical_project(project)
    if acquisition is None:
        acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "F policy acquisition"
    )
    corpus_binding, block_bindings = _validated_private_pack_commitments(acquisition)
    archive, archive_binding = load_stage_output_archive(
        project=root, block="F_search", acquisition=acquisition
    )
    e0_selection, e1_selection, identifiable = (
        _recompute_f_search_policy_selections(archive)
    )
    path = root / F_POLICY_FREEZE_RELATIVE
    _require_regular_file(path, field="F_search policy freeze", mode=0o644)
    raw = path.read_bytes()
    payload = _strict_json(raw, "F_search policy freeze")
    expected_keys = {
        "schema",
        "version",
        "status",
        "acquisition_sha256",
        "corpus_view_file_sha256",
        "corpus_view_semantic_sha256",
        "f_search_view_file_sha256",
        "f_search_view_semantic_sha256",
        "f_search_item_count",
        "f_search_output_archive_file_sha256",
        "f_search_output_archive_semantic_sha256",
        "complete_f_trace_matrix_receipt_sha256",
        "e0_action_id",
        "e0_policy_sha256",
        "e1_action_id",
        "e1_policy_sha256",
        "policies_identifiable",
        "F_search_label_pack_exists",
        "created_with_O_EXCL",
        "same_stage_replay_or_policy_reselection_authorized",
        "policy_freeze_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise MultiHopRAGAcquisitionError("F_search policy freeze schema drifted")
    verify_self_hash(
        payload,
        hash_field="policy_freeze_sha256",
        schema=F_POLICY_FREEZE_SCHEMA,
    )
    f_view = block_bindings["F_search"]["view"]
    e0_policy = _require_sha256(payload.get("e0_policy_sha256"), "F E0 policy")
    e1_policy = _require_sha256(payload.get("e1_policy_sha256"), "F E1 policy")
    _require_sha256(
        payload.get("complete_f_trace_matrix_receipt_sha256"),
        "complete F trace matrix receipt",
    )
    if (
        payload.get("version") != "v1"
        or payload.get("status") != "F_search_terminal_policies_frozen"
        or payload.get("acquisition_sha256") != acquisition_sha
        or payload.get("corpus_view_file_sha256") != corpus_binding["file_sha256"]
        or payload.get("corpus_view_semantic_sha256")
        != corpus_binding["semantic_sha256"]
        or payload.get("f_search_view_file_sha256") != f_view["file_sha256"]
        or payload.get("f_search_view_semantic_sha256")
        != f_view["semantic_sha256"]
        or payload.get("f_search_item_count") != BLOCK_COUNTS["F_search"]
        or payload.get("f_search_output_archive_file_sha256")
        != archive_binding["file_sha256"]
        or payload.get("f_search_output_archive_semantic_sha256")
        != archive_binding["semantic_sha256"]
        or payload.get("complete_f_trace_matrix_receipt_sha256")
        != archive["agent_complete_six_action_trace_matrix_sha256"]
        or payload.get("e0_action_id") != e0_selection.action_id
        or e0_policy != e0_selection.selection_sha256
        or payload.get("e1_action_id") != e1_selection.action_id
        or e1_policy != e1_selection.selection_sha256
        or identifiable is not True
        or payload.get("policies_identifiable") is not identifiable
        or payload.get("F_search_label_pack_exists") is not False
        or payload.get("created_with_O_EXCL") is not True
        or payload.get("same_stage_replay_or_policy_reselection_authorized")
        is not False
    ):
        raise MultiHopRAGAcquisitionError("F_search policy freeze binding drifted")
    return dict(payload)


def create_f_search_policy_freeze_once(
    *, project: Path
) -> dict[str, Any]:
    """Authoritatively select and freeze both policies from canonical F traces."""

    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "F policy acquisition"
    )
    corpus_binding, blocks = _validated_private_pack_commitments(acquisition)
    f_view = blocks["F_search"]["view"]
    archive, archive_binding = load_stage_output_archive(
        project=root, block="F_search", acquisition=acquisition
    )
    e0, e1, identifiable = _recompute_f_search_policy_selections(archive)
    if not identifiable:
        raise MultiHopRAGAcquisitionError("F policies are not identifiable")
    payload = _with_self_hash(
        {
            "schema": F_POLICY_FREEZE_SCHEMA,
            "version": "v1",
            "status": "F_search_terminal_policies_frozen",
            "acquisition_sha256": acquisition_sha,
            "corpus_view_file_sha256": corpus_binding["file_sha256"],
            "corpus_view_semantic_sha256": corpus_binding["semantic_sha256"],
            "f_search_view_file_sha256": f_view["file_sha256"],
            "f_search_view_semantic_sha256": f_view["semantic_sha256"],
            "f_search_item_count": BLOCK_COUNTS["F_search"],
            "f_search_output_archive_file_sha256": archive_binding[
                "file_sha256"
            ],
            "f_search_output_archive_semantic_sha256": archive_binding[
                "semantic_sha256"
            ],
            "complete_f_trace_matrix_receipt_sha256": archive[
                "agent_complete_six_action_trace_matrix_sha256"
            ],
            "e0_action_id": e0.action_id,
            "e0_policy_sha256": e0.selection_sha256,
            "e1_action_id": e1.action_id,
            "e1_policy_sha256": e1.selection_sha256,
            "policies_identifiable": True,
            "F_search_label_pack_exists": False,
            "created_with_O_EXCL": True,
            "same_stage_replay_or_policy_reselection_authorized": False,
        },
        "policy_freeze_sha256",
    )
    _write_json_exclusive(root / F_POLICY_FREEZE_RELATIVE, payload, mode=0o644)
    return load_f_search_policy_freeze(project=root, acquisition=acquisition)


def load_action_seal(
    *, project: Path, block: str, acquisition: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate terminal RAW/Hippo/Agent outputs before any block gold opens."""

    if block not in ACTION_SEAL_RELATIVES:
        raise MultiHopRAGAcquisitionError("block has no late-label action seal")
    root = _canonical_project(project)
    if acquisition is None:
        acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "action seal acquisition"
    )
    corpus_binding, block_bindings = _validated_private_pack_commitments(acquisition)
    block_view = block_bindings[block]["view"]
    archive, archive_binding = load_stage_output_archive(
        project=root, block=block, acquisition=acquisition
    )
    path = root / ACTION_SEAL_RELATIVES[block]
    _require_regular_file(path, field=f"{block} action seal", mode=0o644)
    raw = path.read_bytes()
    payload = _strict_json(raw, f"{block} action seal")
    expected_keys = {
        "schema",
        "version",
        "status",
        "block",
        "acquisition_sha256",
        "corpus_view_file_sha256",
        "corpus_view_semantic_sha256",
        "block_view_file_sha256",
        "block_view_semantic_sha256",
        "item_count",
        "raw_output_set_sha256",
        "hipporag_output_set_sha256",
        "agent_complete_six_action_trace_matrix_sha256",
        "stage_output_archive_file_sha256",
        "stage_output_archive_semantic_sha256",
        "all_three_methods_terminal",
        "policy_stage",
        "policy_authorization_sha256",
        "e0_policy_sha256",
        "e1_policy_sha256",
        "label_pack_opened_before_action_seal",
        "created_with_O_EXCL",
        "same_block_replay_authorized",
        "action_seal_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise MultiHopRAGAcquisitionError(f"{block} action seal schema drifted")
    verify_self_hash(
        payload,
        hash_field="action_seal_sha256",
        schema=ACTION_SEAL_SCHEMAS[block],
    )
    for field in (
        "raw_output_set_sha256",
        "hipporag_output_set_sha256",
        "agent_complete_six_action_trace_matrix_sha256",
    ):
        _require_sha256(payload.get(field), f"{block} {field}")
    expected_policy_stage: str
    expected_authorization: str | None
    expected_e0: str | None
    expected_e1: str | None
    if block == "A_form":
        expected_policy_stage = "formation_all_six_actions_no_F_policy"
        expected_authorization = None
        expected_e0 = None
        expected_e1 = None
    elif block == "A_hold":
        policy = load_f_search_policy_freeze(
            project=root, acquisition=acquisition
        )
        expected_policy_stage = "frozen_F_E0_and_E1_policies"
        expected_authorization = str(policy["policy_freeze_sha256"])
        expected_e0 = str(policy["e0_policy_sha256"])
        expected_e1 = str(policy["e1_policy_sha256"])
    else:
        promotion = load_committed_promotion_authorization(
            project=root, acquisition_sha256=acquisition_sha
        )
        expected_policy_stage = "promoted_E1_and_counterfactual_E0_policies"
        expected_authorization = str(promotion["promotion_sha256"])
        expected_e0 = str(promotion["e0_policy_sha256"])
        expected_e1 = str(promotion["e1_policy_sha256"])
    if (
        payload.get("version") != "v1"
        or payload.get("status") != f"{block}_all_methods_terminal"
        or payload.get("block") != block
        or payload.get("acquisition_sha256") != acquisition_sha
        or payload.get("corpus_view_file_sha256") != corpus_binding["file_sha256"]
        or payload.get("corpus_view_semantic_sha256")
        != corpus_binding["semantic_sha256"]
        or payload.get("block_view_file_sha256") != block_view["file_sha256"]
        or payload.get("block_view_semantic_sha256")
        != block_view["semantic_sha256"]
        or payload.get("item_count") != BLOCK_COUNTS[block]
        or payload.get("raw_output_set_sha256")
        != archive["raw_output_set_sha256"]
        or payload.get("hipporag_output_set_sha256")
        != archive["hipporag_output_set_sha256"]
        or payload.get("agent_complete_six_action_trace_matrix_sha256")
        != archive["agent_complete_six_action_trace_matrix_sha256"]
        or payload.get("stage_output_archive_file_sha256")
        != archive_binding["file_sha256"]
        or payload.get("stage_output_archive_semantic_sha256")
        != archive_binding["semantic_sha256"]
        or payload.get("all_three_methods_terminal") is not True
        or payload.get("policy_stage") != expected_policy_stage
        or payload.get("policy_authorization_sha256") != expected_authorization
        or payload.get("e0_policy_sha256") != expected_e0
        or payload.get("e1_policy_sha256") != expected_e1
        or payload.get("label_pack_opened_before_action_seal") is not False
        or payload.get("created_with_O_EXCL") is not True
        or payload.get("same_block_replay_authorized") is not False
    ):
        raise MultiHopRAGAcquisitionError(f"{block} action seal binding drifted")
    return dict(payload)


def create_action_seal_once(*, project: Path, block: str) -> dict[str, Any]:
    """Bind one verified terminal archive before that block's labels open."""

    if block not in ACTION_SEAL_RELATIVES:
        raise MultiHopRAGAcquisitionError("block has no action seal")
    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "action seal acquisition"
    )
    corpus_binding, blocks = _validated_private_pack_commitments(acquisition)
    view_binding = blocks[block]["view"]
    archive, archive_binding = load_stage_output_archive(
        project=root, block=block, acquisition=acquisition
    )
    if block == "A_form":
        policy_stage = "formation_all_six_actions_no_F_policy"
        authorization = None
        e0 = None
        e1 = None
    elif block == "A_hold":
        policy = load_f_search_policy_freeze(
            project=root, acquisition=acquisition
        )
        policy_stage = "frozen_F_E0_and_E1_policies"
        authorization = policy["policy_freeze_sha256"]
        e0 = policy["e0_policy_sha256"]
        e1 = policy["e1_policy_sha256"]
    else:
        promotion = load_committed_promotion_authorization(
            project=root, acquisition_sha256=acquisition_sha
        )
        policy_stage = "promoted_E1_and_counterfactual_E0_policies"
        authorization = promotion["promotion_sha256"]
        e0 = promotion["e0_policy_sha256"]
        e1 = promotion["e1_policy_sha256"]
    payload = _with_self_hash(
        {
            "schema": ACTION_SEAL_SCHEMAS[block],
            "version": "v1",
            "status": f"{block}_all_methods_terminal",
            "block": block,
            "acquisition_sha256": acquisition_sha,
            "corpus_view_file_sha256": corpus_binding["file_sha256"],
            "corpus_view_semantic_sha256": corpus_binding["semantic_sha256"],
            "block_view_file_sha256": view_binding["file_sha256"],
            "block_view_semantic_sha256": view_binding["semantic_sha256"],
            "item_count": BLOCK_COUNTS[block],
            "raw_output_set_sha256": archive["raw_output_set_sha256"],
            "hipporag_output_set_sha256": archive["hipporag_output_set_sha256"],
            "agent_complete_six_action_trace_matrix_sha256": archive[
                "agent_complete_six_action_trace_matrix_sha256"
            ],
            "stage_output_archive_file_sha256": archive_binding["file_sha256"],
            "stage_output_archive_semantic_sha256": archive_binding[
                "semantic_sha256"
            ],
            "all_three_methods_terminal": True,
            "policy_stage": policy_stage,
            "policy_authorization_sha256": authorization,
            "e0_policy_sha256": e0,
            "e1_policy_sha256": e1,
            "label_pack_opened_before_action_seal": False,
            "created_with_O_EXCL": True,
            "same_block_replay_authorized": False,
        },
        "action_seal_sha256",
    )
    _write_json_exclusive(
        root / ACTION_SEAL_RELATIVES[block], payload, mode=0o644
    )
    return load_action_seal(project=root, block=block, acquisition=acquisition)


def _assess_a_hold_promotion(
    *, root: Path, acquisition: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the sole A_hold decision from canonical sealed evidence."""

    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        item_utility,
        paired_utility_summary,
    )

    freeze = load_f_search_policy_freeze(
        project=root, acquisition=acquisition
    )
    # The action seal loader verifies the complete A_hold archive before the
    # late-label loader below is allowed to stat the label path.
    seal = load_action_seal(
        project=root, block="A_hold", acquisition=acquisition
    )
    archive, archive_binding = load_stage_output_archive(
        project=root, block="A_hold", acquisition=acquisition
    )
    view = load_block_view(project=root, expected_block="A_hold")
    labels = load_block_labels(project=root, expected_block="A_hold")
    joined = join_late_labels(
        view=view, labels=labels, expected_block="A_hold"
    )
    records = archive.get("records")
    if not isinstance(records, list) or len(records) != len(joined):
        raise MultiHopRAGAcquisitionError("A_hold evidence cardinality drifted")
    e0_action_id = str(freeze.get("e0_action_id"))
    e1_action_id = str(freeze.get("e1_action_id"))
    if (
        e0_action_id not in AGENT_ACTION_IDS
        or e1_action_id not in AGENT_ACTION_IDS
        or e0_action_id == e1_action_id
    ):
        raise MultiHopRAGAcquisitionError("A_hold frozen policy identity drifted")
    e0_index = AGENT_ACTION_IDS.index(e0_action_id)
    e1_index = AGENT_ACTION_IDS.index(e1_action_id)
    e0_utilities: list[Fraction] = []
    e1_utilities: list[Fraction] = []
    family_counts: Counter[str] = Counter()
    family_deltas = {family: Fraction(0) for family in FAMILIES}
    for ordinal, (record, (_view_item, label)) in enumerate(
        zip(records, joined, strict=True)
    ):
        if not isinstance(record, Mapping) or label.get("ordinal") != ordinal:
            raise MultiHopRAGAcquisitionError("A_hold evidence order drifted")
        family = label.get("question_type")
        gold = label.get("gold_article_ids")
        if family not in FAMILIES or not isinstance(gold, list):
            raise MultiHopRAGAcquisitionError("A_hold late label drifted")
        envelopes = record.get("agent_action_traces")
        if not isinstance(envelopes, list) or len(envelopes) != len(AGENT_ACTION_IDS):
            raise MultiHopRAGAcquisitionError("A_hold action matrix drifted")
        _e0_checked, _e0_sha, e0_trace = _decode_and_rebuild_action_trace(
            envelopes[e0_index], expected_action_id=e0_action_id
        )
        _e1_checked, _e1_sha, e1_trace = _decode_and_rebuild_action_trace(
            envelopes[e1_index], expected_action_id=e1_action_id
        )
        e0_value = item_utility(e0_trace.output_top5, gold)
        e1_value = item_utility(e1_trace.output_top5, gold)
        e0_utilities.append(e0_value)
        e1_utilities.append(e1_value)
        family_counts[family] += 1
        family_deltas[family] += e1_value - e0_value
    expected_family_counts = {
        family: FAMILY_QUOTAS["A_hold"] for family in FAMILIES
    }
    if dict(family_counts) != expected_family_counts:
        raise MultiHopRAGAcquisitionError("A_hold exact family balance drifted")
    summary = paired_utility_summary(e1_utilities, e0_utilities)
    promoted = (
        summary.delta_total > 0
        and summary.exact_one_sided_p <= Fraction(1, 10)
    )
    return {
        "status": "promote" if promoted else "valid_nonpromotion",
        "challenger_promoted": promoted,
        "item_count": summary.count,
        "exact_family_counts": expected_family_counts,
        "family_delta_totals": dict(family_deltas),
        "family_balanced_delta_total": summary.delta_total,
        "one_sided_magnitude_signflip_p": summary.exact_one_sided_p,
        "gains": summary.gains,
        "harms": summary.harms,
        "ties": summary.ties,
        "e0_action_id": e0_action_id,
        "e1_action_id": e1_action_id,
        "e0_policy_sha256": freeze["e0_policy_sha256"],
        "e1_policy_sha256": freeze["e1_policy_sha256"],
        "f_search_policy_freeze_sha256": freeze["policy_freeze_sha256"],
        "a_hold_action_seal_sha256": seal["action_seal_sha256"],
        "a_hold_output_archive_file_sha256": archive_binding["file_sha256"],
        "a_hold_output_archive_semantic_sha256": archive_binding[
            "semantic_sha256"
        ],
    }


def assess_a_hold_promotion(*, project: Path) -> dict[str, Any]:
    """Return the authoritative exact A_hold decision without writing capability."""

    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    return _assess_a_hold_promotion(root=root, acquisition=acquisition)


def load_committed_promotion_authorization(
    *, project: Path, acquisition_sha256: str
) -> dict[str, Any]:
    """Read the only canonical committed promotion capability for M_search."""

    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    expected_acquisition = _require_sha256(
        acquisition_sha256, "expected acquisition receipt"
    )
    if acquisition.get("acquisition_sha256") != expected_acquisition:
        raise MultiHopRAGAcquisitionError("M_search acquisition binding drifted")
    implementation = verify_committed_implementation_freeze(root)
    path = root / PROMOTION_RELATIVE
    _require_regular_file(path, field="A_hold promotion receipt", mode=0o644)
    raw = path.read_bytes()
    promotion = _strict_json(raw, "A_hold promotion receipt")
    if not isinstance(promotion, Mapping):
        raise MultiHopRAGAcquisitionError("A_hold promotion receipt root drifted")
    _validate_promotion_payload(
        promotion,
        acquisition_sha256=expected_acquisition,
        implementation_freeze_sha256=str(
            implementation["implementation_freeze_sha256"]
        ),
    )
    assessment = _assess_a_hold_promotion(
        root=root, acquisition=acquisition
    )
    delta = assessment["family_balanced_delta_total"]
    p_value = assessment["one_sided_magnitude_signflip_p"]
    if (
        assessment.get("challenger_promoted") is not True
        or promotion.get("a_hold_action_seal_sha256")
        != assessment.get("a_hold_action_seal_sha256")
        or promotion.get("f_search_policy_freeze_sha256")
        != assessment.get("f_search_policy_freeze_sha256")
        or promotion.get("a_hold_output_archive_file_sha256")
        != assessment.get("a_hold_output_archive_file_sha256")
        or promotion.get("a_hold_output_archive_semantic_sha256")
        != assessment.get("a_hold_output_archive_semantic_sha256")
        or promotion.get("e0_action_id") != assessment.get("e0_action_id")
        or promotion.get("e1_action_id") != assessment.get("e1_action_id")
        or promotion.get("e0_policy_sha256")
        != assessment.get("e0_policy_sha256")
        or promotion.get("e1_policy_sha256")
        != assessment.get("e1_policy_sha256")
        or promotion.get("family_balanced_delta_total")
        != [delta.numerator, delta.denominator]
        or promotion.get("one_sided_magnitude_signflip_p")
        != [p_value.numerator, p_value.denominator]
    ):
        raise MultiHopRAGAcquisitionError(
            "promotion differs from authoritative A_hold recomputation"
        )
    return {
        "promotion_sha256": promotion["promotion_sha256"],
        "promotion_file_sha256": _sha256_bytes(raw),
        "promotion_runtime_commit": "canonical_O_EXCL_fsync_self_hash",
        "a_hold_action_seal_sha256": promotion[
            "a_hold_action_seal_sha256"
        ],
        "f_search_policy_freeze_sha256": promotion[
            "f_search_policy_freeze_sha256"
        ],
        "a_hold_output_archive_file_sha256": promotion[
            "a_hold_output_archive_file_sha256"
        ],
        "a_hold_output_archive_semantic_sha256": promotion[
            "a_hold_output_archive_semantic_sha256"
        ],
        "e0_action_id": promotion["e0_action_id"],
        "e1_action_id": promotion["e1_action_id"],
        "e0_policy_sha256": promotion["e0_policy_sha256"],
        "e1_policy_sha256": promotion["e1_policy_sha256"],
        "family_balanced_delta_total": delta,
        "one_sided_magnitude_signflip_p": p_value,
    }


def create_a_hold_promotion_once(
    *, project: Path
) -> dict[str, Any]:
    """Write promotion only after authoritative canonical recomputation."""
    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha = _require_sha256(
        acquisition.get("acquisition_sha256"), "promotion acquisition"
    )
    implementation = verify_committed_implementation_freeze(root)
    assessment = _assess_a_hold_promotion(
        root=root, acquisition=acquisition
    )
    if assessment.get("challenger_promoted") is not True:
        raise MultiHopRAGAcquisitionError("A_hold result does not promote")
    delta = assessment["family_balanced_delta_total"]
    p_value = assessment["one_sided_magnitude_signflip_p"]
    payload = _with_self_hash(
        {
            "schema": PROMOTION_SCHEMA,
            "version": "v1",
            "status": "A_hold_challenger_promoted",
            "acquisition_sha256": acquisition_sha,
            "implementation_freeze_sha256": implementation[
                "implementation_freeze_sha256"
            ],
            "f_search_policy_freeze_sha256": assessment[
                "f_search_policy_freeze_sha256"
            ],
            "a_hold_action_seal_sha256": assessment[
                "a_hold_action_seal_sha256"
            ],
            "a_hold_output_archive_file_sha256": assessment[
                "a_hold_output_archive_file_sha256"
            ],
            "a_hold_output_archive_semantic_sha256": assessment[
                "a_hold_output_archive_semantic_sha256"
            ],
            "e0_action_id": assessment["e0_action_id"],
            "e0_policy_sha256": assessment["e0_policy_sha256"],
            "e1_action_id": assessment["e1_action_id"],
            "e1_policy_sha256": assessment["e1_policy_sha256"],
            "a_hold_item_count": BLOCK_COUNTS["A_hold"],
            "a_hold_exact_family_counts": {
                family: FAMILY_QUOTAS["A_hold"] for family in FAMILIES
            },
            "family_balanced_delta_total": [
                delta.numerator,
                delta.denominator,
            ],
            "one_sided_magnitude_signflip_p": [
                p_value.numerator,
                p_value.denominator,
            ],
            "promotion_rule_id": (
                "positive_total_and_one_sided_magnitude_signflip_p_le_0.10"
            ),
            "challenger_promoted": True,
            "outcome_used_to_change_action_evaluator_or_threshold": False,
            "same_source_replay_authorized": False,
        },
        "promotion_sha256",
    )
    _write_json_exclusive(root / PROMOTION_RELATIVE, payload, mode=0o644)
    return load_committed_promotion_authorization(
        project=root, acquisition_sha256=acquisition_sha
    )


def assess_m_search(*, project: Path) -> dict[str, Any]:
    """Authoritatively recompute all frozen M_search exact boundaries."""

    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        item_utility,
        paired_utility_summary,
    )

    root = _canonical_project(project)
    acquisition, _binding = load_committed_acquisition_receipt(root)
    acquisition_sha256 = _require_sha256(
        acquisition.get("acquisition_sha256"), "M assessment acquisition"
    )
    promotion = load_committed_promotion_authorization(
        project=root, acquisition_sha256=acquisition_sha256
    )
    seal = load_action_seal(
        project=root, block="M_search", acquisition=acquisition
    )
    archive, archive_binding = load_stage_output_archive(
        project=root, block="M_search", acquisition=acquisition
    )
    view = load_block_view(project=root, expected_block="M_search")
    labels = load_block_labels(project=root, expected_block="M_search")
    joined = join_late_labels(
        view=view, labels=labels, expected_block="M_search"
    )
    records = archive.get("records")
    if not isinstance(records, list) or len(records) != len(joined):
        raise MultiHopRAGAcquisitionError("M_search evidence cardinality drifted")
    e0_action_id = str(promotion.get("e0_action_id"))
    e1_action_id = str(promotion.get("e1_action_id"))
    if (
        e0_action_id not in AGENT_ACTION_IDS
        or e1_action_id not in AGENT_ACTION_IDS
        or e0_action_id == e1_action_id
    ):
        raise MultiHopRAGAcquisitionError("M_search policy identity drifted")
    e0_index = AGENT_ACTION_IDS.index(e0_action_id)
    e1_index = AGENT_ACTION_IDS.index(e1_action_id)
    e0_values: list[Fraction] = []
    e1_values: list[Fraction] = []
    hippo_values: list[Fraction] = []
    raw_values: list[Fraction] = []
    family_counts: Counter[str] = Counter()
    family_agent_hippo = {family: Fraction(0) for family in FAMILIES}
    agent_complete = 0
    raw_complete = 0
    for ordinal, (record, (_view_item, label)) in enumerate(
        zip(records, joined, strict=True)
    ):
        if not isinstance(record, Mapping) or label.get("ordinal") != ordinal:
            raise MultiHopRAGAcquisitionError("M_search evidence order drifted")
        family = label.get("question_type")
        gold = label.get("gold_article_ids")
        if family not in FAMILIES or not isinstance(gold, list):
            raise MultiHopRAGAcquisitionError("M_search late label drifted")
        envelopes = record.get("agent_action_traces")
        if not isinstance(envelopes, list) or len(envelopes) != len(AGENT_ACTION_IDS):
            raise MultiHopRAGAcquisitionError("M_search action matrix drifted")
        _e0_checked, _e0_sha, e0_trace = _decode_and_rebuild_action_trace(
            envelopes[e0_index], expected_action_id=e0_action_id
        )
        _e1_checked, _e1_sha, e1_trace = _decode_and_rebuild_action_trace(
            envelopes[e1_index], expected_action_id=e1_action_id
        )
        raw = _validate_method_output(record.get("raw_output"), method="RAW")
        hippo = _validate_method_output(
            record.get("hipporag_output"), method="HippoRAG"
        )
        e0_value = item_utility(e0_trace.output_top5, gold)
        e1_value = item_utility(e1_trace.output_top5, gold)
        hippo_value = item_utility(hippo["output_top5"], gold)
        raw_value = item_utility(raw["output_top5"], gold)
        e0_values.append(e0_value)
        e1_values.append(e1_value)
        hippo_values.append(hippo_value)
        raw_values.append(raw_value)
        family_counts[family] += 1
        family_agent_hippo[family] += e1_value - hippo_value
        agent_complete += int(set(gold) <= set(e1_trace.output_top5))
        raw_complete += int(set(gold) <= set(raw["output_top5"]))
    expected_family_counts = {
        family: FAMILY_QUOTAS["M_search"] for family in FAMILIES
    }
    if dict(family_counts) != expected_family_counts:
        raise MultiHopRAGAcquisitionError("M_search exact family balance drifted")
    l5 = paired_utility_summary(e1_values, e0_values)
    agent_hippo = paired_utility_summary(e1_values, hippo_values)
    agent_raw = paired_utility_summary(e1_values, raw_values)
    l5_passed = (
        l5.delta_total > 0
        and l5.exact_one_sided_p <= Fraction(1, 10)
    )
    cross_family = (
        agent_hippo.delta_total > 0
        and agent_hippo.exact_one_sided_p <= Fraction(1, 10)
        and all(delta > 0 for delta in family_agent_hippo.values())
    )
    complete_delta = agent_complete - raw_complete
    return {
        "status": "M_search_authoritatively_assessed",
        "item_count": len(records),
        "exact_family_counts": expected_family_counts,
        "e0_action_id": e0_action_id,
        "e1_action_id": e1_action_id,
        "l5_delta_total": l5.delta_total,
        "l5_signflip_p": l5.exact_one_sided_p,
        "l5_passed": l5_passed,
        "agent_minus_hippo_delta_total": agent_hippo.delta_total,
        "agent_minus_hippo_signflip_p": agent_hippo.exact_one_sided_p,
        "agent_minus_hippo_family_deltas": dict(family_agent_hippo),
        "cross_family_agent_over_hippo_passed": cross_family,
        "agent_minus_raw_delta_total": agent_raw.delta_total,
        "agent_minus_raw_signflip_p": agent_raw.exact_one_sided_p,
        "agent_complete_count": agent_complete,
        "raw_complete_count": raw_complete,
        "agent_minus_raw_complete_delta": complete_delta,
        "raw_complete_advantage_overcome": complete_delta >= 0,
        "promotion_sha256": promotion["promotion_sha256"],
        "m_search_action_seal_sha256": seal["action_seal_sha256"],
        "m_search_output_archive_file_sha256": archive_binding["file_sha256"],
        "m_search_output_archive_semantic_sha256": archive_binding[
            "semantic_sha256"
        ],
    }


def load_corpus_view(*, project: Path) -> dict[str, Any]:
    root = _canonical_project(project)
    receipt, _receipt_binding = load_committed_acquisition_receipt(root)
    binding, _blocks = _validated_private_pack_commitments(receipt)
    payload = _read_bound_private_json(
        path=root / CORPUS_VIEW_RELATIVE,
        file_sha256=str(binding["file_sha256"]),
        field="corpus view",
    )
    _validate_corpus_view(payload)
    if payload.get("corpus_view_sha256") != binding["semantic_sha256"]:
        raise MultiHopRAGAcquisitionError("corpus semantic commitment drifted")
    return payload


def load_block_view(
    *,
    project: Path,
    expected_block: str,
) -> dict[str, Any]:
    if expected_block not in BLOCK_ORDER:
        raise MultiHopRAGAcquisitionError("private view block is invalid")
    root = _canonical_project(project)
    receipt, _receipt_binding = load_committed_acquisition_receipt(root)
    _require_sha256(receipt.get("acquisition_sha256"), "view acquisition")
    _corpus, blocks = _validated_private_pack_commitments(receipt)
    _authorize_block_private_access(
        root=root, block=expected_block, acquisition=receipt
    )
    binding = blocks[expected_block]["view"]
    payload = _read_bound_private_json(
        path=root / BLOCK_VIEW_RELATIVES[expected_block],
        file_sha256=str(binding["file_sha256"]),
        field=f"{expected_block} view",
    )
    _validate_block_view(payload, expected_block=expected_block)
    if payload.get("block_view_sha256") != binding["semantic_sha256"]:
        raise MultiHopRAGAcquisitionError(
            f"{expected_block} view semantic commitment drifted"
        )
    return payload


def load_block_labels(
    *,
    project: Path,
    expected_block: str,
) -> dict[str, Any]:
    # These gates intentionally run before any filesystem operation.
    if expected_block == "F_search":
        raise MultiHopRAGAcquisitionError("F_search label pack does not exist")
    if expected_block not in ACTION_SEAL_RELATIVES:
        raise MultiHopRAGAcquisitionError("private label block is invalid")
    root = _canonical_project(project)
    receipt, _receipt_binding = load_committed_acquisition_receipt(root)
    _corpus, blocks = _validated_private_pack_commitments(receipt)
    # A_form: archive terminal -> action seal -> authoritative descriptive
    # policy freeze.  A_hold: F policies frozen + action terminal.  M_search:
    # promotion + M action terminal.  Every capability runs before this label
    # path is statted.
    if expected_block == "A_form":
        load_a_form_policy_freeze(project=root, acquisition=receipt)
    else:
        load_action_seal(
            project=root, block=expected_block, acquisition=receipt
        )
    binding = blocks[expected_block]["labels"]
    if binding.get("created") is not True:
        raise MultiHopRAGAcquisitionError("authorized label commitment is missing")
    payload = _read_bound_private_json(
        path=root / BLOCK_LABEL_RELATIVES[expected_block],
        file_sha256=str(binding["file_sha256"]),
        field=f"{expected_block} labels",
    )
    _validate_block_labels(payload, expected_block=expected_block)
    if payload.get("block_labels_sha256") != binding["semantic_sha256"]:
        raise MultiHopRAGAcquisitionError(
            f"{expected_block} label semantic commitment drifted"
        )
    return payload


def join_late_labels(
    *,
    view: Mapping[str, Any],
    labels: Mapping[str, Any],
    expected_block: str,
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Join already authorized labels to sealed action views by view hash."""

    _validate_block_view(view, expected_block=expected_block)
    _validate_block_labels(labels, expected_block=expected_block)
    view_items = view["items"]
    label_items = labels["items"]
    label_map = {row["view_sha256"]: row for row in label_items}
    if len(label_map) != len(label_items):
        raise MultiHopRAGAcquisitionError("late label view hashes overlap")
    joined: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    identities: set[str] = set()
    for ordinal, item in enumerate(view_items):
        label = label_map.get(stable_hash(item))
        if label is None or label.get("ordinal") != ordinal:
            raise MultiHopRAGAcquisitionError("late label join is incomplete")
        identity = str(label["identity_commitment_sha256"])
        if identity in identities:
            raise MultiHopRAGAcquisitionError("late label identity overlaps")
        identities.add(identity)
        joined.append((item, label))
    if len(joined) != len(view_items) or len(label_items) != len(view_items):
        raise MultiHopRAGAcquisitionError("late label join cardinality drifted")
    return tuple(joined)


def hash_source_file(path: Path, *, logical_name: str) -> SourceFileBinding:
    _require_regular_file(path, field=f"{logical_name} source")
    raw = path.read_bytes()
    return SourceFileBinding(
        logical_name=logical_name,
        sha256=_sha256_bytes(raw),
        git_blob_sha1=_git_blob_sha1(raw),
        byte_size=len(raw),
    )


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise MultiHopRAGAcquisitionError("freeze path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MultiHopRAGAcquisitionError("freeze path is unsafe")
    return path.as_posix()


def _repository_root(project: Path) -> Path:
    root = project.resolve(strict=True)
    for candidate in (root, *root.parents):
        marker = candidate / ".git"
        if (marker.is_dir() or marker.is_file()) and not marker.is_symlink():
            return candidate
    raise MultiHopRAGAcquisitionError("project is not inside a Git repository")


def _verify_head_blobs(
    *, project: Path, relative_paths: Sequence[str]
) -> tuple[str, dict[str, str]]:
    root = project.resolve(strict=True)
    repository = _repository_root(root)
    project_prefix = root.relative_to(repository)
    safe = tuple(_safe_relative(path) for path in relative_paths)
    if len(safe) != len(set(safe)):
        raise MultiHopRAGAcquisitionError("committed binding paths overlap")
    repository_paths = tuple(
        (PurePosixPath(project_prefix.as_posix()) / path).as_posix()
        for path in safe
    )
    def parse_head(raw: bytes) -> str:
        try:
            value = raw.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise MultiHopRAGAcquisitionError("Git HEAD is malformed") from exc
        if _HEX40.fullmatch(value) is None:
            raise MultiHopRAGAcquisitionError("Git HEAD is malformed")
        return value

    try:
        head_result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        head = parse_head(head_result.stdout)
        tree_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "-z",
                head,
                "--",
                *repository_paths,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MultiHopRAGAcquisitionError("read-only Git verification failed") from exc
    observed: dict[str, str] = {}
    for record in tree_result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise MultiHopRAGAcquisitionError("Git tree output is malformed")
        mode, kind, raw_oid = fields
        try:
            path = raw_path.decode("utf-8", errors="strict")
            oid = raw_oid.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise MultiHopRAGAcquisitionError("Git tree output is malformed") from exc
        if (
            kind != b"blob"
            or mode in {b"120000", b"160000"}
            or _HEX40.fullmatch(oid) is None
            or path in observed
        ):
            raise MultiHopRAGAcquisitionError("Git tree entry is not a regular blob")
        observed[path] = oid
    if set(observed) != set(repository_paths):
        raise MultiHopRAGAcquisitionError("bound files are not exactly present at HEAD")
    result: dict[str, str] = {}
    for relative, repository_relative in zip(safe, repository_paths, strict=True):
        path = root / relative
        _require_regular_file(path, field="freeze-bound protocol file")
        raw = path.read_bytes()
        oid = _git_blob_sha1(raw)
        if oid != observed[repository_relative]:
            raise MultiHopRAGAcquisitionError(
                "freeze-bound protocol file does not byte-match HEAD"
            )
        result[relative] = oid
    try:
        postcheck = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MultiHopRAGAcquisitionError("Git HEAD postcheck failed") from exc
    if parse_head(postcheck.stdout) != head:
        raise MultiHopRAGAcquisitionError("Git HEAD drifted during verification")
    return head, result


def verify_committed_implementation_freeze(project: Path) -> dict[str, Any]:
    """Verify self-hash, exact roles, working bytes, and committed Git blobs."""

    root = project.resolve(strict=True)
    freeze_path = root / IMPLEMENTATION_FREEZE_RELATIVE
    payload, raw = _read_json_object(freeze_path, field="implementation freeze")
    verify_self_hash(
        payload,
        hash_field="implementation_freeze_sha256",
        schema=IMPLEMENTATION_FREEZE_SCHEMA,
    )
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != REQUIRED_FREEZE_ROLES:
        raise MultiHopRAGAcquisitionError("implementation freeze role set drifted")
    paths: dict[str, str] = {}
    for role, entry in bindings.items():
        if not isinstance(role, str) or not isinstance(entry, Mapping):
            raise MultiHopRAGAcquisitionError("implementation freeze binding drifted")
        relative = _safe_relative(entry.get("relative_path"))
        file_sha = _require_sha256(entry.get("file_sha256"), f"{role} file hash")
        git_oid = entry.get("git_blob_sha1")
        if not isinstance(git_oid, str) or _HEX40.fullmatch(git_oid) is None:
            raise MultiHopRAGAcquisitionError(f"{role} Git blob is invalid")
        if role in FIXED_FREEZE_ROLE_PATHS and relative != FIXED_FREEZE_ROLE_PATHS[role]:
            raise MultiHopRAGAcquisitionError(f"{role} frozen path drifted")
        if relative in paths.values():
            raise MultiHopRAGAcquisitionError("implementation freeze paths overlap")
        paths[role] = relative
        path = root / relative
        _require_regular_file(path, field=f"{role} implementation")
        current_raw = path.read_bytes()
        if _sha256_bytes(current_raw) != file_sha or _git_blob_sha1(current_raw) != git_oid:
            raise MultiHopRAGAcquisitionError(f"{role} working bytes drifted")
    all_paths = (IMPLEMENTATION_FREEZE_RELATIVE, *[paths[role] for role in sorted(paths)])
    _head, head_oids = _verify_head_blobs(project=root, relative_paths=all_paths)
    if not hmac.compare_digest(
        _git_blob_sha1(raw), head_oids[IMPLEMENTATION_FREEZE_RELATIVE]
    ):
        raise MultiHopRAGAcquisitionError(
            "parsed implementation freeze does not match committed HEAD blob"
        )
    for role, relative in paths.items():
        if head_oids[relative] != bindings[role]["git_blob_sha1"]:
            raise MultiHopRAGAcquisitionError(f"{role} committed Git blob drifted")
    return {
        "schema": IMPLEMENTATION_FREEZE_SCHEMA,
        "implementation_freeze_sha256": payload[
            "implementation_freeze_sha256"
        ],
        "implementation_freeze_file_sha256": _sha256_bytes(raw),
        "implementation_freeze_git_blob_sha1": head_oids[
            IMPLEMENTATION_FREEZE_RELATIVE
        ],
        "required_role_count": len(REQUIRED_FREEZE_ROLES),
        "required_role_set_sha256": stable_hash(sorted(REQUIRED_FREEZE_ROLES)),
        "all_bindings_byte_match_committed_HEAD": True,
    }


def _verify_loaded_acquisition_origin(
    *, project: Path, implementation_receipt: Mapping[str, Any]
) -> str:
    """Bind this running acquisition module to the verified project checkout."""

    root = project.resolve(strict=True)
    if (
        implementation_receipt.get("all_bindings_byte_match_committed_HEAD")
        is not True
        or implementation_receipt.get("required_role_count")
        != len(REQUIRED_FREEZE_ROLES)
    ):
        raise MultiHopRAGAcquisitionError(
            "implementation freeze verification is not closed"
        )
    expected = (root / ACQUISITION_RELATIVE).resolve(strict=True)
    module = sys.modules.get(__name__)
    module_file = None if module is None else getattr(module, "__file__", None)
    spec = None if module is None else getattr(module, "__spec__", None)
    origin = None if spec is None else getattr(spec, "origin", None)
    if not isinstance(module_file, str) or not isinstance(origin, str):
        raise MultiHopRAGAcquisitionError(
            "loaded acquisition module origin is unavailable"
        )
    if (
        Path(module_file).resolve(strict=True) != expected
        or Path(origin).resolve(strict=True) != expected
    ):
        raise MultiHopRAGAcquisitionError(
            "loaded acquisition module is outside the frozen project role path"
        )
    return str(expected)


def _canonical_project(project: Path) -> Path:
    root = project.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise MultiHopRAGAcquisitionError("project root is unsafe")
    return root


def _require_private_artifacts_ignored(project: Path) -> None:
    repository = _repository_root(project)
    prefix = project.relative_to(repository)
    relative_paths = (
        QUERY_SOURCE_RELATIVE,
        CORPUS_SOURCE_RELATIVE,
        SELECTION_SECRET_RELATIVE,
        SECRET_MARKER_RELATIVE,
        ACQUISITION_MARKER_RELATIVE,
        FAILURE_RELATIVE,
        CORPUS_VIEW_RELATIVE,
        *BLOCK_VIEW_RELATIVES.values(),
        *BLOCK_LABEL_RELATIVES.values(),
        *STAGE_OUTPUT_ARCHIVE_RELATIVES.values(),
    )
    repository_paths = tuple(
        (PurePosixPath(prefix.as_posix()) / relative).as_posix()
        for relative in relative_paths
    )
    stdin = b"\0".join(path.encode("utf-8") for path in repository_paths) + b"\0"
    try:
        tracked_index = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z", "--", *repository_paths],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        tracked_head = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "-z",
                "HEAD",
                "--",
                *repository_paths,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "check-ignore",
                "--no-index",
                "-z",
                "--stdin",
            ],
            input=stdin,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MultiHopRAGAcquisitionError("Git ignore verification failed") from exc
    if tracked_index.stdout or tracked_head.stdout:
        raise MultiHopRAGAcquisitionError(
            "a private source/artifact path is tracked in index or HEAD"
        )
    try:
        ignored = tuple(
            row.decode("utf-8", errors="strict")
            for row in result.stdout.split(b"\0")
            if row
        )
    except UnicodeDecodeError as exc:
        raise MultiHopRAGAcquisitionError("Git ignore output is malformed") from exc
    if result.returncode != 0 or set(ignored) != set(repository_paths):
        raise MultiHopRAGAcquisitionError("not all private source/artifact paths are ignored")


def _default_paths(project: Path) -> AcquisitionPaths:
    return AcquisitionPaths(
        marker=project / ACQUISITION_MARKER_RELATIVE,
        failure=project / FAILURE_RELATIVE,
        corpus_view=project / CORPUS_VIEW_RELATIVE,
        block_views={
            block: project / BLOCK_VIEW_RELATIVES[block] for block in BLOCK_ORDER
        },
        block_labels={
            block: project / BLOCK_LABEL_RELATIVES[block]
            for block in ("A_form", "A_hold", "M_search")
        },
        public_receipt=project / PUBLIC_RECEIPT_RELATIVE,
    )


def _source_binding_payload(binding: SourceFileBinding) -> dict[str, Any]:
    return {
        "logical_name": binding.logical_name,
        "sha256": binding.sha256,
        "git_blob_sha1": binding.git_blob_sha1,
        "byte_size": binding.byte_size,
    }


def _validate_official_source_binding(binding: SourceFileBinding) -> None:
    expected = {
        QUERY_SOURCE_NAME: (QUERY_SOURCE_SIZE, QUERY_SOURCE_GIT_BLOB_SHA1),
        CORPUS_SOURCE_NAME: (CORPUS_SOURCE_SIZE, CORPUS_SOURCE_GIT_BLOB_SHA1),
    }
    if binding.logical_name not in expected:
        raise MultiHopRAGAcquisitionError("source logical name drifted")
    size, oid = expected[binding.logical_name]
    if binding.byte_size != size or binding.git_blob_sha1 != oid:
        raise MultiHopRAGAcquisitionError(
            f"pinned {binding.logical_name} byte identity drifted"
        )


def consume_one_shot_marker(
    *, path: Path, phase: str, bindings: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "schema": ATTEMPT_MARKER_SCHEMA,
        "version": VERSION,
        "phase": phase,
        "bindings": dict(bindings),
        "replay_secret_rotation_resample_replacement_or_retry_authorized": False,
    }
    payload = _with_self_hash(body, "marker_sha256")
    _write_json_exclusive(path, payload, mode=0o600)
    return payload


def _validate_marker(payload: Mapping[str, Any], *, phase: str) -> str:
    if set(payload) != {
        "schema",
        "version",
        "phase",
        "bindings",
        "replay_secret_rotation_resample_replacement_or_retry_authorized",
        "marker_sha256",
    }:
        raise MultiHopRAGAcquisitionError("one-shot marker schema drifted")
    marker_sha = verify_self_hash(
        payload, hash_field="marker_sha256", schema=ATTEMPT_MARKER_SCHEMA
    )
    if (
        payload.get("version") != VERSION
        or payload.get("phase") != phase
        or not isinstance(payload.get("bindings"), Mapping)
        or payload.get(
            "replay_secret_rotation_resample_replacement_or_retry_authorized"
        )
        is not False
    ):
        raise MultiHopRAGAcquisitionError("one-shot marker contract drifted")
    return marker_sha


def create_secret_custody_once(
    *,
    marker_path: Path,
    secret_path: Path,
    public_custody_path: Path,
    implementation_binding: Mapping[str, Any],
    query_binding: SourceFileBinding,
    corpus_binding: SourceFileBinding,
    urandom: Callable[[int], bytes] = os.urandom,
) -> dict[str, Any]:
    """Consume marker, call ``urandom(32)`` once, and publish commitment."""

    for path in (marker_path, secret_path, public_custody_path):
        if path.exists() or path.is_symlink():
            raise MultiHopRAGAcquisitionError("custody output exists; replay is forbidden")
    marker = consume_one_shot_marker(
        path=marker_path,
        phase="selection_secret_creation",
        bindings={
            "implementation_freeze_sha256": _require_sha256(
                implementation_binding.get("implementation_freeze_sha256"),
                "implementation freeze hash",
            ),
            "query_source_sha256": query_binding.sha256,
            "corpus_source_sha256": corpus_binding.sha256,
        },
    )
    # This is the sole randomness call in either formal phase.  Do not use
    # random temporary names, nonces, canaries, shuffles, or fallback secrets.
    secret = urandom(32)
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise MultiHopRAGAcquisitionError("urandom(32) did not return 32 raw bytes")
    _write_exclusive(secret_path, secret, mode=0o600)
    body = {
        "schema": SOURCE_CUSTODY_SCHEMA,
        "version": "v1",
        "status": "pinned_sources_and_one_selection_secret_committed",
        "dataset_identity": {
            "repository": DATASET_REPOSITORY,
            "dataset_commit": DATASET_COMMIT,
            "code_commit": CODE_COMMIT,
            "license": "ODC-BY",
        },
        "implementation_binding": dict(implementation_binding),
        "source_bindings": {
            "query_source": _source_binding_payload(query_binding),
            "corpus_source": _source_binding_payload(corpus_binding),
        },
        "selection_secret_commitment_sha256": _sha256_bytes(secret),
        "secret_marker_sha256": marker["marker_sha256"],
        "secret_raw_byte_count": 32,
        "secret_file_mode": "0600",
        "os_urandom_call_count": 1,
        "os_urandom_requested_bytes": 32,
        "preselection_exposure_record": {
            "absolute_base_model_row_zero_claimed": False,
            "human_item_content_used_for_mechanism_design": False,
            "private_selection_secret_preexisted": False,
            "formal_item_selection_or_retrieval_score_preexisted": False,
            "known_HF_viewer_first_page_content_exposure_recorded": True,
            "complete_unpinned_source_machine_aggregate_parse_recorded": True,
            "corpus_article0_metadata_without_body_exposure_recorded": True,
            "mechanism_change_basis_limited_to_schema_and_feasibility": True,
            "fixed_query_source_ordinals_excluded": "0_through_127_inclusive",
            "gold_reference_to_exposed_corpus_article0_excluded": True,
        },
        "same_source_secret_rotation_replay_resample_replacement_or_retry_authorized": False,
    }
    payload = _with_self_hash(body, "source_custody_sha256")
    _assert_public_safe(payload)
    _write_json_exclusive(public_custody_path, payload, mode=0o644)
    return payload


def create_source_custody(project: Path) -> dict[str, Any]:
    """Formal byte-only source custody; this function never decodes JSON."""

    root = _canonical_project(project)
    _require_private_artifacts_ignored(root)
    implementation = verify_committed_implementation_freeze(root)
    _verify_loaded_acquisition_origin(
        project=root, implementation_receipt=implementation
    )
    query_path = root / QUERY_SOURCE_RELATIVE
    corpus_path = root / CORPUS_SOURCE_RELATIVE
    query_binding = hash_source_file(query_path, logical_name=QUERY_SOURCE_NAME)
    corpus_binding = hash_source_file(corpus_path, logical_name=CORPUS_SOURCE_NAME)
    _validate_official_source_binding(query_binding)
    _validate_official_source_binding(corpus_binding)
    return create_secret_custody_once(
        marker_path=root / SECRET_MARKER_RELATIVE,
        secret_path=root / SELECTION_SECRET_RELATIVE,
        public_custody_path=root / SOURCE_CUSTODY_RELATIVE,
        implementation_binding=implementation,
        query_binding=query_binding,
        corpus_binding=corpus_binding,
    )


def _validate_custody_payload(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "version",
        "status",
        "dataset_identity",
        "implementation_binding",
        "source_bindings",
        "selection_secret_commitment_sha256",
        "secret_marker_sha256",
        "secret_raw_byte_count",
        "secret_file_mode",
        "os_urandom_call_count",
        "os_urandom_requested_bytes",
        "preselection_exposure_record",
        "same_source_secret_rotation_replay_resample_replacement_or_retry_authorized",
        "source_custody_sha256",
    }
    expected_exposure = {
        "absolute_base_model_row_zero_claimed": False,
        "human_item_content_used_for_mechanism_design": False,
        "private_selection_secret_preexisted": False,
        "formal_item_selection_or_retrieval_score_preexisted": False,
        "known_HF_viewer_first_page_content_exposure_recorded": True,
        "complete_unpinned_source_machine_aggregate_parse_recorded": True,
        "corpus_article0_metadata_without_body_exposure_recorded": True,
        "mechanism_change_basis_limited_to_schema_and_feasibility": True,
        "fixed_query_source_ordinals_excluded": "0_through_127_inclusive",
        "gold_reference_to_exposed_corpus_article0_excluded": True,
    }
    if set(payload) != expected_keys:
        raise MultiHopRAGAcquisitionError("source custody top-level schema drifted")
    verify_self_hash(
        payload, hash_field="source_custody_sha256", schema=SOURCE_CUSTODY_SCHEMA
    )
    if (
        payload.get("version") != "v1"
        or payload.get("status")
        != "pinned_sources_and_one_selection_secret_committed"
        or payload.get("secret_raw_byte_count") != 32
        or payload.get("secret_file_mode") != "0600"
        or payload.get("os_urandom_call_count") != 1
        or payload.get("os_urandom_requested_bytes") != 32
        or payload.get("dataset_identity")
        != {
            "repository": DATASET_REPOSITORY,
            "dataset_commit": DATASET_COMMIT,
            "code_commit": CODE_COMMIT,
            "license": "ODC-BY",
        }
        or payload.get("preselection_exposure_record") != expected_exposure
        or not isinstance(payload.get("implementation_binding"), Mapping)
        or payload.get(
            "same_source_secret_rotation_replay_resample_replacement_or_retry_authorized"
        )
        is not False
    ):
        raise MultiHopRAGAcquisitionError("source custody contract drifted")
    source_bindings = payload.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != {
        "query_source",
        "corpus_source",
    }:
        raise MultiHopRAGAcquisitionError("source custody bindings drifted")
    for role, logical_name, size, oid in (
        (
            "query_source",
            QUERY_SOURCE_NAME,
            QUERY_SOURCE_SIZE,
            QUERY_SOURCE_GIT_BLOB_SHA1,
        ),
        (
            "corpus_source",
            CORPUS_SOURCE_NAME,
            CORPUS_SOURCE_SIZE,
            CORPUS_SOURCE_GIT_BLOB_SHA1,
        ),
    ):
        row = source_bindings.get(role)
        if (
            not isinstance(row, Mapping)
            or row.get("logical_name") != logical_name
            or row.get("byte_size") != size
            or row.get("git_blob_sha1") != oid
        ):
            raise MultiHopRAGAcquisitionError("pinned custody source identity drifted")
        _require_sha256(row.get("sha256"), f"{role} SHA256")
    _require_sha256(
        payload.get("selection_secret_commitment_sha256"), "secret commitment"
    )


def verify_committed_source_custody(
    project: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _canonical_project(project)
    implementation = verify_committed_implementation_freeze(root)
    custody_path = root / SOURCE_CUSTODY_RELATIVE
    custody, raw = _read_json_object(custody_path, field="source custody")
    _validate_custody_payload(custody)
    if custody.get("implementation_binding") != implementation:
        raise MultiHopRAGAcquisitionError("custody implementation binding drifted")
    head, oids = _verify_head_blobs(
        project=root, relative_paths=(SOURCE_CUSTODY_RELATIVE,)
    )
    if not hmac.compare_digest(
        _git_blob_sha1(raw), oids[SOURCE_CUSTODY_RELATIVE]
    ):
        raise MultiHopRAGAcquisitionError(
            "parsed source custody does not match committed HEAD blob"
        )
    marker_path = root / SECRET_MARKER_RELATIVE
    _require_regular_file(marker_path, field="secret marker", mode=0o600)
    marker_raw = marker_path.read_bytes()
    marker = _strict_json(marker_raw, "secret marker")
    if not isinstance(marker, Mapping):
        raise MultiHopRAGAcquisitionError("secret marker root drifted")
    marker_sha = _validate_marker(marker, phase="selection_secret_creation")
    expected_marker_bindings = {
        "implementation_freeze_sha256": implementation[
            "implementation_freeze_sha256"
        ],
        "query_source_sha256": custody["source_bindings"]["query_source"][
            "sha256"
        ],
        "corpus_source_sha256": custody["source_bindings"]["corpus_source"][
            "sha256"
        ],
    }
    if (
        marker_sha != custody.get("secret_marker_sha256")
        or marker.get("bindings") != expected_marker_bindings
    ):
        raise MultiHopRAGAcquisitionError("secret marker custody binding drifted")
    binding = {
        "source_custody_sha256": custody["source_custody_sha256"],
        "source_custody_file_sha256": _sha256_bytes(raw),
        "source_custody_git_blob_sha1": oids[SOURCE_CUSTODY_RELATIVE],
        "custody_verified_at_git_HEAD": head,
    }
    return custody, binding


def load_committed_source_custody(
    project: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    root = _canonical_project(project)
    custody, binding = verify_committed_source_custody(root)
    secret_path = root / SELECTION_SECRET_RELATIVE
    _require_regular_file(secret_path, field="selection secret", mode=0o600)
    secret = secret_path.read_bytes()
    if (
        len(secret) != 32
        or _sha256_bytes(secret)
        != custody["selection_secret_commitment_sha256"]
    ):
        raise MultiHopRAGAcquisitionError("selection secret commitment drifted")
    return custody, binding, secret


def _binding_from_custody(
    custody: Mapping[str, Any], role: str
) -> SourceFileBinding:
    row = custody["source_bindings"][role]
    return SourceFileBinding(
        logical_name=str(row["logical_name"]),
        sha256=str(row["sha256"]),
        git_blob_sha1=str(row["git_blob_sha1"]),
        byte_size=int(row["byte_size"]),
    )


def _read_bound_source(path: Path, expected: SourceFileBinding) -> bytes:
    _require_regular_file(path, field=f"{expected.logical_name} source")
    raw = path.read_bytes()
    observed = SourceFileBinding(
        logical_name=expected.logical_name,
        sha256=_sha256_bytes(raw),
        git_blob_sha1=_git_blob_sha1(raw),
        byte_size=len(raw),
    )
    if observed != expected:
        raise MultiHopRAGAcquisitionError(
            f"bound {expected.logical_name} source bytes drifted"
        )
    return raw


def _preflight_acquisition_outputs(paths: AcquisitionPaths) -> None:
    all_paths = [
        paths.marker,
        paths.failure,
        paths.corpus_view,
        paths.public_receipt,
        *paths.block_views.values(),
        *paths.block_labels.values(),
    ]
    if len(all_paths) != len(set(all_paths)):
        raise MultiHopRAGAcquisitionError("acquisition output paths overlap")
    for path in all_paths:
        if path.exists() or path.is_symlink():
            raise MultiHopRAGAcquisitionError(
                "acquisition output exists; replay is forbidden"
            )


def _terminal_failure(
    *,
    path: Path,
    marker_sha256: str,
    stage: str,
    exc: BaseException,
) -> None:
    failure_class = "unexpected_infrastructure_invalid"
    message = str(exc).casefold()
    if "capacity insufficient" in message:
        failure_class = "source_capacity_insufficient"
    elif "schema" in message or "json" in message or "source" in message:
        failure_class = "source_or_schema_invalid"
    elif "persist" in message or "output" in message:
        failure_class = "persistence_invalid"
    body = {
        "schema": FAILURE_SCHEMA,
        "version": VERSION,
        "status": "terminal_no_replay",
        "failure_class": failure_class,
        "failure_stage": stage,
        "marker_sha256": _require_sha256(marker_sha256, "failure marker hash"),
        "exception_type_sha256": _sha256_bytes(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ),
        "private_row_content_persisted_publicly": False,
        "same_source_replay_secret_rotation_resample_or_replacement_authorized": False,
    }
    payload = _with_self_hash(body, "failure_sha256")
    try:
        _assert_public_safe(payload)
        _write_json_exclusive(path, payload, mode=0o644)
    except BaseException:
        pass


def _execute_acquisition_once(
    *,
    project: Path,
    capability: object,
    query_path: Path,
    corpus_path: Path,
    query_binding: SourceFileBinding,
    corpus_binding: SourceFileBinding,
    secret: bytes,
    custody_binding: Mapping[str, Any],
    paths: AcquisitionPaths,
) -> dict[str, Any]:
    """Unique source parser and block former; no diagnostic/replay entrypoint."""

    if (
        _FORMAL_ENTRY_ACTIVE is not True
        or capability is not _FORMAL_EXECUTION_CAPABILITY
    ):
        raise MultiHopRAGAcquisitionError(
            "formal acquisition capability is absent"
        )
    root = _canonical_project(project)
    if (
        query_path != root / QUERY_SOURCE_RELATIVE
        or corpus_path != root / CORPUS_SOURCE_RELATIVE
        or paths != _default_paths(root)
    ):
        raise MultiHopRAGAcquisitionError(
            "formal acquisition paths are not canonical"
        )
    _preflight_acquisition_outputs(paths)
    marker = consume_one_shot_marker(
        path=paths.marker,
        phase="formal_source_parse_and_block_formation",
        bindings={
            "source_custody_sha256": _require_sha256(
                custody_binding.get("source_custody_sha256"), "custody hash"
            ),
            "query_source_sha256": query_binding.sha256,
            "corpus_source_sha256": corpus_binding.sha256,
        },
    )
    stage = "read_bound_sources_after_marker"
    try:
        query_raw = _read_bound_source(query_path, query_binding)
        corpus_raw = _read_bound_source(corpus_path, corpus_binding)
        stage = "strict_source_JSON_decode"
        query_payload = _strict_json(query_raw, "pinned MultiHopRAG query source")
        corpus_payload = _strict_json(corpus_raw, "pinned MultiHopRAG corpus source")
        stage = "source_qualification_and_fixed_exclusions"
        articles, candidates, source_stats = parse_source_payloads(
            query_payload=query_payload,
            corpus_payload=corpus_payload,
            enforce_formal_counts=True,
        )
        stage = "collision_representative_and_continuous_HMAC_selection"
        blocks, selection_stats = select_private_blocks(candidates, secret=secret)
        stage = "materialize_URL_free_views_and_late_labels"
        corpus_view, views, labels = materialize_private_payloads(
            articles=articles, blocks=blocks
        )
        stage = "persist_private_packs"
        commitments = persist_private_payloads(
            corpus=corpus_view,
            views=views,
            labels=labels,
            paths=paths,
        )
        body = {
            "schema": PUBLIC_RECEIPT_SCHEMA,
            "version": VERSION,
            "status": "private_four_block_pack_formed",
            "dataset_identity": {
                "repository": DATASET_REPOSITORY,
                "dataset_commit": DATASET_COMMIT,
                "code_commit": CODE_COMMIT,
                "license": "ODC-BY",
            },
            "custody_binding": dict(custody_binding),
            "attempt_marker_sha256": marker["marker_sha256"],
            "source_bindings": {
                "query_source": _source_binding_payload(query_binding),
                "corpus_source": _source_binding_payload(corpus_binding),
            },
            "source_qualification": source_stats,
            "selection_qualification": selection_stats,
            "private_pack_commitments": commitments,
            "label_isolation": {
                "A_form": "separate_late_label_pack",
                "F_search": "label_pack_not_created",
                "A_hold": "separate_late_label_pack",
                "M_search": "view_and_label_pack_sealed_until_valid_A_hold_promotion",
            },
            "public_candidate_identity_query_answer_fact_URL_evidence_or_gold": False,
            "same_source_replay_secret_rotation_resample_replacement_or_retry_authorized": False,
        }
        receipt = _with_self_hash(body, "acquisition_sha256")
        _assert_public_safe(receipt)
        stage = "persist_public_receipt"
        _write_json_exclusive(paths.public_receipt, receipt, mode=0o644)
        return receipt
    except BaseException as exc:
        _terminal_failure(
            path=paths.failure,
            marker_sha256=str(marker["marker_sha256"]),
            stage=stage,
            exc=exc,
        )
        raise


def formal_acquire(project: Path) -> dict[str, Any]:
    if _FORMAL_ENTRY_ACTIVE is not True:
        raise MultiHopRAGAcquisitionError(
            "official source access is available only through --formal-acquire"
        )
    root = _canonical_project(project)
    _require_private_artifacts_ignored(root)
    implementation = verify_committed_implementation_freeze(root)
    _verify_loaded_acquisition_origin(
        project=root, implementation_receipt=implementation
    )
    custody, custody_binding, secret = load_committed_source_custody(root)
    query_binding = _binding_from_custody(custody, "query_source")
    corpus_binding = _binding_from_custody(custody, "corpus_source")
    query_path = root / QUERY_SOURCE_RELATIVE
    corpus_path = root / CORPUS_SOURCE_RELATIVE
    # Byte-only verification is permitted before the attempt marker; decoding
    # and row access remain exclusively inside execute_acquisition_once.
    if hash_source_file(query_path, logical_name=QUERY_SOURCE_NAME) != query_binding:
        raise MultiHopRAGAcquisitionError("query source binding drifted pre-marker")
    if hash_source_file(corpus_path, logical_name=CORPUS_SOURCE_NAME) != corpus_binding:
        raise MultiHopRAGAcquisitionError("corpus source binding drifted pre-marker")
    return _execute_acquisition_once(
        project=root,
        capability=_FORMAL_EXECUTION_CAPABILITY,
        query_path=query_path,
        corpus_path=corpus_path,
        query_binding=query_binding,
        corpus_binding=corpus_binding,
        secret=secret,
        custody_binding=custody_binding,
        paths=_default_paths(root),
    )


def _synthetic_corpus() -> list[dict[str, str]]:
    return [
        {
            "title": f"Synthetic title {index}",
            "author": f"Synthetic author {index % 7}",
            "source": f"synthetic-source-{index % 49}",
            "published_at": f"2025-{(index % 12) + 1:02d}-01",
            "category": f"synthetic-category-{index % 5}",
            "url": f"synthetic://article/{index}",
            "body": (
                f"Synthetic body for article {index}. "
                f"Synthetic fact from article {index}."
            ),
        }
        for index in range(CORPUS_RECORD_COUNT)
    ]


def _synthetic_evidence(left: int, right: int) -> list[dict[str, str]]:
    return [
        {
            "url": f"synthetic://article/{left}",
            "fact": f"Synthetic fact from article {left}.",
        },
        {
            "url": f"synthetic://article/{right}",
            "fact": f"Synthetic fact from article {right}.",
        },
    ]


def run_synthetic_qualification() -> dict[str, Any]:
    """Row-free/offline qualification of selection, isolation, and hashes."""

    queries: list[dict[str, Any]] = []
    for index in range(EXPOSED_QUERY_STOP):
        queries.append(
            {
                "query": f"Excluded synthetic query {index}?",
                "evidence_list": _synthetic_evidence(1, 2),
                "question_type": NULL_FAMILY,
                "answer": "synthetic excluded answer",
            }
        )
    for family_i, family in enumerate(FAMILIES):
        for item_i in range(101):
            # The first two comparison rows deliberately collide after NFKC,
            # casefold, and whitespace collapse.  101 rows leave 100 groups.
            if family == "comparison_query" and item_i in {0, 1}:
                query = (
                    "ＳＹＮＴＨＥＴＩＣ   COLLISION?"
                    if item_i == 0
                    else "synthetic collision?"
                )
            else:
                query = f"Synthetic {family} item {item_i}?"
            left = 1 + ((family_i * 202 + item_i * 2) % 600)
            right = 1 + ((left + 1) % 608)
            queries.append(
                {
                    "query": query,
                    "evidence_list": _synthetic_evidence(left, right),
                    "question_type": family,
                    "answer": f"synthetic answer {family_i}-{item_i}",
                }
            )
    articles, candidates, source_stats = parse_source_payloads(
        query_payload=queries,
        corpus_payload=_synthetic_corpus(),
        enforce_formal_counts=False,
    )
    secret = bytes(range(32))
    blocks, selection_stats = select_private_blocks(candidates, secret=secret)
    corpus, views, labels = materialize_private_payloads(
        articles=articles, blocks=blocks
    )
    for block in BLOCK_ORDER:
        _validate_block_view(views[block], expected_block=block)
        if block != "F_search":
            _validate_block_labels(labels[block], expected_block=block)
            if len(
                join_late_labels(
                    view=views[block], labels=labels[block], expected_block=block
                )
            ) != BLOCK_COUNTS[block]:
                raise MultiHopRAGAcquisitionError("synthetic late join drifted")
    public = {
        "schema": f"{VERSION}_synthetic_qualification",
        "status": "pass",
        "source_root_counts": source_stats["root_counts"],
        "eligible_before_collision_grouping": source_stats[
            "eligible_before_collision_grouping"
        ],
        "all_eligible_evidence_facts_exact_normalized_body_contained": (
            source_stats[
                "all_eligible_evidence_facts_exact_normalized_body_contained"
            ]
        ),
        "selected_block_counts": selection_stats["selected_block_counts"],
        "selected_exact_family_counts": selection_stats[
            "selected_exact_family_counts"
        ],
        "corpus_view_sha256": corpus["corpus_view_sha256"],
        "block_view_semantic_sha256": {
            block: views[block]["block_view_sha256"] for block in BLOCK_ORDER
        },
        "late_label_semantic_sha256": {
            block: labels[block]["block_labels_sha256"] for block in labels
        },
        "F_search_label_pack_created": False,
        "M_search_pre_promotion_loader_guard_implemented": True,
        "network_or_official_source_access": False,
    }
    _assert_public_safe(public)
    return _with_self_hash(public, "qualification_sha256")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-source-custody", action="store_true")
    mode.add_argument("--formal-acquire", action="store_true")
    mode.add_argument("--synthetic-qualification", action="store_true")
    parser.add_argument("--project", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.synthetic_qualification:
        print(_canonical_bytes(run_synthetic_qualification()).decode("utf-8"))
        return 0
    if arguments.project is None:
        raise MultiHopRAGAcquisitionError("--project is required for a formal phase")
    if arguments.create_source_custody:
        create_source_custody(arguments.project)
        return 0
    global _FORMAL_ENTRY_ACTIVE
    _FORMAL_ENTRY_ACTIVE = True
    try:
        formal_acquire(arguments.project)
    finally:
        _FORMAL_ENTRY_ACTIVE = False
    return 0


__all__ = [
    "A_FORM_POLICY_FREEZE_RELATIVE",
    "A_FORM_POLICY_FREEZE_SCHEMA",
    "ACTION_SEAL_RELATIVES",
    "ACTION_SEAL_SCHEMAS",
    "AGENT_ACTION_IDS",
    "AcquisitionPaths",
    "BLOCK_COUNTS",
    "BLOCK_LABEL_RELATIVES",
    "BLOCK_ORDER",
    "BLOCK_VIEW_RELATIVES",
    "BlockCommitment",
    "CORPUS_VIEW_RELATIVE",
    "CORPUS_RECORD_COUNT",
    "CorpusArticle",
    "FAMILIES",
    "FAMILY_QUOTAS",
    "F_POLICY_FREEZE_RELATIVE",
    "F_POLICY_FREEZE_SCHEMA",
    "IMPLEMENTATION_FREEZE_RELATIVE",
    "IMPLEMENTATION_FREEZE_SCHEMA",
    "MultiHopRAGAcquisitionError",
    "PROMOTION_RELATIVE",
    "PROMOTION_SCHEMA",
    "PrivateCandidate",
    "QUERY_RECORD_COUNT",
    "REQUIRED_FREEZE_ROLES",
    "SOURCE_CUSTODY_RELATIVE",
    "STAGE_OUTPUT_ARCHIVE_RELATIVES",
    "STAGE_OUTPUT_ARCHIVE_SCHEMAS",
    "STAGE_RUNTIME_BINDING_KEYS",
    "SourceFileBinding",
    "TOTAL_SELECTED",
    "assess_a_hold_promotion",
    "assess_m_search",
    "build_stage_output_record",
    "consume_one_shot_marker",
    "create_a_hold_promotion_once",
    "create_action_seal_once",
    "create_a_form_policy_freeze_once",
    "create_f_search_policy_freeze_once",
    "create_secret_custody_once",
    "create_source_custody",
    "create_stage_output_archive_once",
    "encode_typed_action_trace",
    "formal_acquire",
    "hash_source_file",
    "hmac_digest",
    "join_late_labels",
    "load_action_seal",
    "load_a_form_policy_freeze",
    "load_block_labels",
    "load_block_view",
    "load_committed_acquisition_receipt",
    "load_committed_promotion_authorization",
    "load_corpus_view",
    "load_f_search_policy_freeze",
    "load_committed_source_custody",
    "load_stage_output_archive",
    "materialize_private_payloads",
    "normalize_query",
    "parse_source_payloads",
    "persist_private_payloads",
    "run_synthetic_qualification",
    "select_private_blocks",
    "stable_hash",
    "verify_committed_implementation_freeze",
    "verify_committed_source_custody",
    "verify_self_hash",
]


if __name__ == "__main__":
    raise SystemExit(main())

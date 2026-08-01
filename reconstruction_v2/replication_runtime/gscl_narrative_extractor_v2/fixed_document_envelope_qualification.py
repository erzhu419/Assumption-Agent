"""Fixed public qualification of the GSCL v2 document envelope.

This module is an iteration of the source-free runtime qualification harness,
not a benchmark or an effect study.  It owns four immutable public documents
that exercise the 176, 351, mixed Unicode/punctuation, and 1024 lexical-token
paths.  A verified offline Qwen runtime executes each document twice; only
canonical, content-free commitments and aggregate resource counts are
published.

The qualification establishes that the exact runtime can execute the bounded
document orchestration.  It deliberately does not promote the generic
``NarrativeDocumentEnvelopeV1`` receipt into formal leaf evidence and does not
make the envelope eligible for a downstream effect measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
from types import MappingProxyType
from typing import Mapping, Sequence

from replication_runtime.gscl_narrative_extractor_v1 import (
    contract as v1_contract,
)
from replication_runtime.gscl_narrative_extractor_v1 import worker

from . import document_envelope
from . import fixed_public_qualification as leaf_qualification
from . import memory_safe_qwen


VERSION = "gscl_v2_fixed_public_document_envelope_qualification_v1"
SHARD_RECEIPT_SCHEMA = f"{VERSION}.shard.safe.v1"
AGGREGATE_RECEIPT_SCHEMA = f"{VERSION}.aggregate.safe.v1"
SHARD_OUTPUT_NAME = "document_envelope.shard.safe.json"
AGGREGATE_OUTPUT_NAME = "document_envelope.aggregate.safe.json"
SHARD_COUNT = 2
REPEAT_COUNT = 2
SHARD_FIXTURE_ORDINALS = ((3,), (0, 1, 2))
MAXIMUM_SAFE_RECEIPT_BYTES = 2 * 1024 * 1024
MAXIMUM_IMPLEMENTATION_FILE_BYTES = 4 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SUCCESS_STATUS = "EXECUTED_WITHOUT_TYPED_FAILURE"
_NO_RELATION_STATUS = "FUNCTIONAL_EXTRACTED_BRANCH_NOT_EXERCISED"
_TYPED_FAILURE_STATUS = "TYPED_FAILURE"
_REPEAT_MISMATCH_STATUS = "REPEAT_MISMATCH"
_CANARY_NOT_EXECUTED_STATUS = "NOT_EXECUTED_AFTER_CANARY_FAILURE"
_OUTCOME_FAILURE_CODES = frozenset(
    {
        "DOCUMENT_FUNCTIONAL_EXTRACTED_BRANCH_NOT_EXERCISED",
        "DOCUMENT_REPEAT_BYTE_MISMATCH",
        "DOCUMENT_TEACHER_FORCED_CANARY_FAILED",
        "DOCUMENT_TYPED_FAILURE_REPORTED",
    }
)
_RESOURCE_PEAK_FAILURE_CODE = "DOCUMENT_RESOURCE_PEAK_COLLECTION_FAILED"
_QUALIFICATION_FAILURE_CODES = (
    _OUTCOME_FAILURE_CODES | {_RESOURCE_PEAK_FAILURE_CODE}
)


class FixedDocumentEnvelopeQualificationError(RuntimeError):
    """Stable qualification failure that never embeds fixture content."""


@dataclass(frozen=True, slots=True)
class PublicDocumentFixture:
    fixture_id: str
    ordinal: int
    story_text: str = field(repr=False)
    lexical_token_count: int
    expected_segment_token_counts: tuple[int, ...]
    expected_leaf_call_count: int
    minimum_extracted_leaf_count: int
    feature_flags: tuple[str, ...]

    @property
    def input_sha256(self) -> str:
        return hashlib.sha256(self.story_text.encode("utf-8")).hexdigest()

    @property
    def fixture_commitment(self) -> str:
        return _safe_hash(_fixture_payload(self, include_commitment=False))


def _tokens(
    *,
    tag: str,
    count: int,
    replacements: Mapping[int, str] | None = None,
) -> str:
    fixed = {} if replacements is None else dict(replacements)
    return " ".join(
        fixed.get(index, f"{tag}Token{index:04d}")
        for index in range(count)
    )


def _mixed_unicode_story() -> str:
    sizes = (16, 17, 176, 3)
    terminators = (".", "。", "！", "?")
    replacements = (
        {0: "Élodie", 1: "observes", 2: "São", 3: "Paulo"},
        {0: "甲方", 1: "guides", 2: "乙方", 3: "e\u0301quipe"},
        {
            0: "München",
            1: "supports",
            2: "東京",
            88: "東京",
            89: "reverses",
            90: "Theta",
        },
        {0: "尾部", 1: "stays", 2: "visible"},
    )
    return " ".join(
        _tokens(
            tag=f"Mixed{index}",
            count=size,
            replacements=replacements[index],
        )
        + terminators[index]
        for index, size in enumerate(sizes)
    )


PUBLIC_DOCUMENT_FIXTURES = (
    PublicDocumentFixture(
        fixture_id="document_0176_balanced_pair",
        ordinal=0,
        story_text=_tokens(
            tag="D176",
            count=176,
            replacements={
                0: "Aster",
                1: "supports",
                2: "Birch",
                88: "Birch",
                89: "precedes",
                90: "Cedar",
            },
        ),
        lexical_token_count=176,
        expected_segment_token_counts=(88, 88),
        expected_leaf_call_count=2,
        minimum_extracted_leaf_count=2,
        feature_flags=("balanced_split", "single_sentence"),
    ),
    PublicDocumentFixture(
        fixture_id="document_0351_balanced_triple",
        ordinal=1,
        story_text=_tokens(
            tag="D351",
            count=351,
            replacements={
                0: "North",
                1: "influences",
                2: "South",
                117: "South",
                118: "causes",
                119: "East",
                234: "East",
                235: "follows",
                236: "West",
            },
        ),
        lexical_token_count=351,
        expected_segment_token_counts=(117, 117, 117),
        expected_leaf_call_count=3,
        minimum_extracted_leaf_count=3,
        feature_flags=("balanced_split", "single_sentence"),
    ),
    PublicDocumentFixture(
        fixture_id="document_mixed_unicode_punctuation",
        ordinal=2,
        story_text=_mixed_unicode_story(),
        # The frozen lexical policy treats the combining mark in e\u0301quipe
        # as a boundary, so this public NFD fixture has 213, not 212, tokens.
        lexical_token_count=213,
        expected_segment_token_counts=(16, 18, 88, 88, 3),
        expected_leaf_call_count=3,
        minimum_extracted_leaf_count=3,
        feature_flags=(
            "ascii_punctuation",
            "cjk_punctuation",
            "context_only_segments",
            "nfd_unicode",
            "unicode",
        ),
    ),
    PublicDocumentFixture(
        fixture_id="document_1024_root_cap",
        ordinal=3,
        story_text=_tokens(
            tag="D1024",
            count=1024,
            replacements={
                0: "Alpha",
                1: "supports",
                2: "Beta",
                171: "Beta",
                172: "precedes",
                173: "Gamma",
                342: "Gamma",
                343: "causes",
                344: "Delta",
                513: "Delta",
                514: "follows",
                515: "Epsilon",
                684: "Epsilon",
                685: "supports",
                686: "Zeta",
                854: "Zeta",
                855: "precedes",
                856: "Eta",
            },
        ),
        lexical_token_count=1024,
        expected_segment_token_counts=(171, 171, 171, 171, 170, 170),
        expected_leaf_call_count=6,
        minimum_extracted_leaf_count=6,
        feature_flags=("balanced_split", "root_token_cap", "single_sentence"),
    ),
)


def _canonical_bytes(value: object) -> bytes:
    return v1_contract.canonical_json_bytes(value)


def _safe_hash(value: object) -> str:
    return v1_contract.semantic_sha256(value)


def _require_hex64(value: object, issue: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FixedDocumentEnvelopeQualificationError(issue)
    return value


def _fixture_payload(
    fixture: PublicDocumentFixture,
    *,
    include_commitment: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "expected_leaf_call_count": fixture.expected_leaf_call_count,
        "expected_segment_token_counts": list(
            fixture.expected_segment_token_counts
        ),
        "feature_flags": list(fixture.feature_flags),
        "fixture_id": fixture.fixture_id,
        "input_sha256": fixture.input_sha256,
        "lexical_token_count": fixture.lexical_token_count,
        "minimum_extracted_leaf_count": (
            fixture.minimum_extracted_leaf_count
        ),
        "ordinal": fixture.ordinal,
    }
    if include_commitment:
        payload["fixture_commitment"] = fixture.fixture_commitment
    return payload


FIXTURE_SUITE_SHA256 = _safe_hash(
    [_fixture_payload(row) for row in PUBLIC_DOCUMENT_FIXTURES]
)
EXPECTED_FIXTURE_SUITE_SHA256 = (
    "ac75bc1e0c3c07cb565e111f51cfd7a2c2750b0e5f14d8dc73c8bc8ca1f6bce0"
)
EXPECTED_FIXTURE_INPUT_SHA256S = (
    "1f0fed9a8f4ce1df9876b1043fc0754a4f0418b40c5dd27e0097c9250df50716",
    "2fc33ad30164830b3b361e36739f3ec5bbbd6e024d1cf17286562fbfb07609c9",
    "ced3867a3d919d1414d8c8da31694fca731a84731b15c3e70d17dfb1f8e100eb",
    "d276d14fb4033d17ad4e9d2d63abeb91748d31588e12a10139685136e8120cb7",
)
FIXTURE_COMMITMENTS = MappingProxyType(
    {
        row.fixture_id: row.fixture_commitment
        for row in PUBLIC_DOCUMENT_FIXTURES
    }
)


def _validate_public_document_fixtures() -> None:
    if (
        tuple(row.ordinal for row in PUBLIC_DOCUMENT_FIXTURES)
        != tuple(range(len(PUBLIC_DOCUMENT_FIXTURES)))
        or len({row.fixture_id for row in PUBLIC_DOCUMENT_FIXTURES})
        != len(PUBLIC_DOCUMENT_FIXTURES)
        or tuple(row.lexical_token_count for row in PUBLIC_DOCUMENT_FIXTURES)
        != (176, 351, 213, 1024)
        or FIXTURE_SUITE_SHA256 != EXPECTED_FIXTURE_SUITE_SHA256
        or tuple(row.input_sha256 for row in PUBLIC_DOCUMENT_FIXTURES)
        != EXPECTED_FIXTURE_INPUT_SHA256S
        or SHARD_FIXTURE_ORDINALS != ((3,), (0, 1, 2))
        or sorted(
            ordinal
            for shard in SHARD_FIXTURE_ORDINALS
            for ordinal in shard
        )
        != list(range(len(PUBLIC_DOCUMENT_FIXTURES)))
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_fixture_topology_invalid"
        )
    for fixture in PUBLIC_DOCUMENT_FIXTURES:
        plans = document_envelope.plan_document_segments(fixture.story_text)
        observed = tuple(row.lexical_token_count for row in plans)
        if (
            observed != fixture.expected_segment_token_counts
            or sum(observed) != fixture.lexical_token_count
            or sum(row.leaf_eligible for row in plans)
            != fixture.expected_leaf_call_count
            or not 1
            <= fixture.minimum_extracted_leaf_count
            <= fixture.expected_leaf_call_count
            or _HEX64.fullmatch(fixture.input_sha256) is None
            or _HEX64.fullmatch(fixture.fixture_commitment) is None
        ):
            raise FixedDocumentEnvelopeQualificationError(
                "document_fixture_contract_invalid"
            )


_validate_public_document_fixtures()


def _implementation_closure() -> dict[str, str]:
    project_root = Path(__file__).parents[2]
    package_root = Path(__file__).parent
    v1_root = project_root / "replication_runtime" / "gscl_narrative_extractor_v1"
    manifest_root = project_root / "manifests"
    paths = {
        "document_envelope_qualification.py": Path(__file__),
        "document_envelope.py": package_root / "document_envelope.py",
        "v2_closed_choice.py": package_root / "closed_choice.py",
        "v2_contract.py": package_root / "contract.py",
        "v2_memory_safe_qwen.py": package_root / "memory_safe_qwen.py",
        "v2_leaf_public_qualification.py": (
            package_root / "fixed_public_qualification.py"
        ),
        "v1_closed_choice_worker.py": v1_root / "closed_choice_worker.py",
        "v1_contract.py": v1_root / "contract.py",
        "v1_worker.py": v1_root / "worker.py",
        "narrative_correspondence_parser.py": (
            project_root
            / "assumption_agent"
            / "gscl_narrative_correspondence_v1.py"
        ),
        "qualification_shard0.service": (
            manifest_root
            / "gscl_document_envelope_fixed_qualification_shard0.service"
        ),
        "qualification_shard1.service": (
            manifest_root
            / "gscl_document_envelope_fixed_qualification_shard1.service"
        ),
        "qualification_aggregate.service": (
            manifest_root
            / "gscl_document_envelope_fixed_qualification_aggregate.service"
        ),
    }
    result: dict[str, str] = {}
    for logical_name, path in paths.items():
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise FixedDocumentEnvelopeQualificationError(
                "document_implementation_unreadable"
            ) from exc
        if not raw or len(raw) > MAXIMUM_IMPLEMENTATION_FILE_BYTES:
            raise FixedDocumentEnvelopeQualificationError(
                "document_implementation_size_invalid"
            )
        result[logical_name] = hashlib.sha256(raw).hexdigest()
    if len(result) != len(paths):
        raise FixedDocumentEnvelopeQualificationError(
            "document_implementation_name_collision"
        )
    return dict(sorted(result.items()))


def _zero_counters(*, aggregate: bool = False) -> dict[str, int]:
    return {
        "api_access_count": 0,
        "external_evaluator_scorer_access_count": 0,
        "external_fixture_source_access_count": 0,
        "free_form_generation_count": 0,
        "label_access_count": 0,
        "network_access_count": 0,
        "online_evaluator_access_count": 0,
        "external_source_access_count": 0,
        "shard_receipt_access_count": 2 if aggregate else 0,
    }


def _publish_once(path: Path, raw: bytes) -> None:
    if (
        not isinstance(path, Path)
        or not raw
        or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_publish_arguments_invalid"
        )
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.parent.lstat()
    except OSError as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_output_root_invalid"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.parent.is_symlink():
        raise FixedDocumentEnvelopeQualificationError(
            "document_output_root_invalid"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_receipt_publish_failed"
        ) from exc


def _prepare_output_path(path: Path) -> None:
    """Reject stale output before the multi-gigabyte model is loaded."""

    if not isinstance(path, Path):
        raise FixedDocumentEnvelopeQualificationError(
            "document_output_root_invalid"
        )
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent = path.parent.lstat()
        output_exists = path.exists() or path.is_symlink()
    except OSError as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_output_root_invalid"
        ) from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or path.parent.is_symlink()
        or output_exists
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_output_not_fresh"
        )


def _load_upstream_leaf_aggregate(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_upstream_receipt_unreadable"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or not raw
        or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_upstream_receipt_size_invalid"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_upstream_receipt_json_invalid"
        ) from exc
    if (
        type(value) is not dict
        or _canonical_bytes(value) != raw
        or value.get("schema")
        != leaf_qualification.AGGREGATE_RECEIPT_SCHEMA
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_upstream_receipt_canonical_invalid"
        )
    supplied = _require_hex64(
        value.get("self_sha256"),
        "document_upstream_receipt_self_invalid",
    )
    body = {
        key: child for key, child in value.items() if key != "self_sha256"
    }
    expected_implementation = leaf_qualification._implementation_closure()
    if (
        supplied != _safe_hash(body)
        or value.get("qualification_passed") is not True
        or value.get("repeat_byte_exact") is not True
        or value.get("repeat_count") != leaf_qualification.REPEAT_COUNT
        or value.get("fixture_suite_sha256")
        != leaf_qualification.FIXTURE_SUITE_SHA256
        or value.get("fixture_ordinals")
        != list(range(len(leaf_qualification.PUBLIC_FIXTURES)))
        or value.get("fixture_count")
        != len(leaf_qualification.PUBLIC_FIXTURES)
        or value.get("fixture_commitments")
        != dict(leaf_qualification.FIXTURE_COMMITMENTS)
        or value.get("outcome_counts")
        != {
            "success": len(leaf_qualification.PUBLIC_FIXTURES),
            "typed_abstention": 0,
            "typed_error": 0,
        }
        or value.get("counters") != leaf_qualification._zero_counters()
        or value.get("implementation_closure") != expected_implementation
        or value.get("implementation_closure_sha256")
        != _safe_hash(expected_implementation)
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_upstream_receipt_contract_invalid"
        )
    value["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _manifest_commitments(
    manifest: worker.ModelAssetManifest,
) -> dict[str, str]:
    return leaf_qualification._manifest_commitments(manifest)


def _verify_model_binding(
    *, model_root: Path, manifest: worker.ModelAssetManifest
) -> None:
    leaf_qualification._verify_model_binding(
        model_root=model_root, manifest=manifest
    )


def _load_exact_runtime(
    *, model_root: Path, manifest: worker.ModelAssetManifest
) -> object:
    return leaf_qualification._load_exact_runtime(
        model_root=model_root, manifest=manifest
    )


def _validate_exact_runtime(
    *, runtime: object, manifest: worker.ModelAssetManifest
) -> str:
    """Reject qualification fakes and re-trigger the runtime's own seal."""

    if (
        type(runtime) is not memory_safe_qwen.MemorySafeQwenRuntime
        or getattr(runtime, "_exact", None) is not True
        or getattr(runtime, "_marker", None)
        is not memory_safe_qwen._EXACT_RUNTIME_MARKER
        or getattr(runtime, "_manifest_commitment", None)
        != manifest.manifest_file_sha256
        or getattr(runtime, "_device", None) != memory_safe_qwen.DEVICE
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_exact_runtime_authority_invalid"
        )
    try:
        return _require_hex64(
            runtime.runtime_commitment,
            "document_runtime_commitment_invalid",
        )
    except FixedDocumentEnvelopeQualificationError:
        raise
    except Exception as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_runtime_commitment_unavailable"
        ) from exc


def _select_document(
    *, runtime: object, story_text: str
) -> document_envelope.NarrativeDocumentEnvelopeV1:
    return document_envelope.select_document_runtime_only(
        story_text, runtime=runtime
    )


def _safe_segment_topology(
    result: document_envelope.NarrativeDocumentEnvelopeV1,
) -> list[dict[str, object]]:
    return [
        {
            "chunk_count": row.plan.chunk_count,
            "chunk_index": row.plan.chunk_index,
            "disposition": row.disposition.value,
            "leaf_called": row.leaf_called,
            "leaf_eligible": row.plan.leaf_eligible,
            "lexical_token_count": row.plan.lexical_token_count,
            "parent_sentence_id": row.plan.parent_sentence_id,
            "segment_id": row.plan.segment_id,
        }
        for row in result.segments
    ]


def _minimal_negative_outcome(
    *,
    fixture: PublicDocumentFixture,
    status: str,
    failure_code: str,
) -> dict[str, object]:
    """Return a content-free outcome when no stable envelope may be retained."""

    if (
        status not in {_REPEAT_MISMATCH_STATUS, _CANARY_NOT_EXECUTED_STATUS}
        or failure_code not in _OUTCOME_FAILURE_CODES
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_negative_outcome_arguments_invalid"
        )
    return {
        "disposition_counts": None,
        "document_envelope_receipt_sha256": None,
        "document_envelope_self_sha256": None,
        "exact_runtime_binding_observed_on_extracted_leaves": False,
        "extracted_leaf_count": 0,
        "failure_code": failure_code,
        "fixture_commitment": fixture.fixture_commitment,
        "fixture_id": fixture.fixture_id,
        "input_sha256": fixture.input_sha256,
        "ordinal": fixture.ordinal,
        "partial_projection_available": False,
        "projection_commitment": None,
        "repeat_byte_exact": False,
        "repeat_count": REPEAT_COUNT,
        "repeat_outcome_sha256": None,
        "resource_summary": None,
        "segment_topology": None,
        "segment_topology_sha256": None,
        "segments_commitment": None,
        "status": status,
    }


def _run_document_once(
    *,
    runtime: object,
    runtime_commitment: str,
    fixture: PublicDocumentFixture,
) -> dict[str, object]:
    result = _select_document(runtime=runtime, story_text=fixture.story_text)
    if type(result) is not document_envelope.NarrativeDocumentEnvelopeV1:
        raise FixedDocumentEnvelopeQualificationError(
            "document_result_type_invalid"
        )
    receipt = dict(result.receipt)
    resource_summary = receipt.get("resource_summary")
    if type(resource_summary) is not dict:
        raise FixedDocumentEnvelopeQualificationError(
            "document_resource_summary_invalid"
        )
    plans = tuple(row.plan for row in result.segments)
    observed_counts = tuple(row.lexical_token_count for row in plans)
    allowed = {
        document_envelope.SegmentDisposition.EXTRACTED,
        document_envelope.SegmentDisposition.NO_RELATION,
        document_envelope.SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE,
        document_envelope.SegmentDisposition.TYPED_FAILURE,
    }
    observed_leaf_call_count = sum(row.leaf_called for row in result.segments)
    typed_failure_count = sum(
        row.disposition is document_envelope.SegmentDisposition.TYPED_FAILURE
        for row in result.segments
    )
    if (
        observed_counts != fixture.expected_segment_token_counts
        or resource_summary.get("root_lexical_token_count")
        != fixture.lexical_token_count
        or receipt.get("root_source_sha256") != fixture.input_sha256
        or receipt.get("segmentation_policy_sha256")
        != document_envelope.SEGMENTATION_POLICY_SHA256
        or receipt.get("byte_outcome_coverage_complete") is not True
        or receipt.get("typed_failure_count") != typed_failure_count
        or receipt.get("formal_leaf_authority_established") is not False
        or receipt.get("downstream_eligible") is not False
        or any(row.disposition not in allowed for row in result.segments)
        or observed_leaf_call_count > fixture.expected_leaf_call_count
        or any(
            row.disposition
            is document_envelope.SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE
            and row.leaf_called
            for row in result.segments
        )
        or resource_summary.get("leaf_call_count")
        != observed_leaf_call_count
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_execution_contract_invalid"
        )
    leaf_runtime_commitments = {
        row.leaf_decision.receipt.get("model_runtime_commitment")
        for row in result.segments
        if row.leaf_decision is not None
    }
    if leaf_runtime_commitments and leaf_runtime_commitments != {
        runtime_commitment
    }:
        raise FixedDocumentEnvelopeQualificationError(
            "document_leaf_runtime_binding_mismatch"
        )
    extracted_leaf_count = sum(
        row.disposition is document_envelope.SegmentDisposition.EXTRACTED
        for row in result.segments
    )
    if typed_failure_count:
        status = _TYPED_FAILURE_STATUS
        failure_code: str | None = "DOCUMENT_TYPED_FAILURE_REPORTED"
    elif extracted_leaf_count < fixture.minimum_extracted_leaf_count:
        status = _NO_RELATION_STATUS
        failure_code = (
            "DOCUMENT_FUNCTIONAL_EXTRACTED_BRANCH_NOT_EXERCISED"
        )
    else:
        status = _SUCCESS_STATUS
        failure_code = None
    topology = _safe_segment_topology(result)
    body: dict[str, object] = {
        "disposition_counts": dict(receipt["disposition_counts"]),
        "document_envelope_receipt_sha256": hashlib.sha256(
            result.receipt_bytes
        ).hexdigest(),
        "document_envelope_self_sha256": _require_hex64(
            receipt.get("self_sha256"),
            "document_envelope_self_hash_invalid",
        ),
        "exact_runtime_binding_observed_on_extracted_leaves": bool(
            leaf_runtime_commitments
        ),
        "extracted_leaf_count": extracted_leaf_count,
        "failure_code": failure_code,
        "fixture_commitment": fixture.fixture_commitment,
        "fixture_id": fixture.fixture_id,
        "input_sha256": fixture.input_sha256,
        "ordinal": fixture.ordinal,
        "partial_projection_available": bool(
            receipt["partial_projection_available"]
        ),
        "projection_commitment": _require_hex64(
            receipt.get("projection_commitment"),
            "document_projection_hash_invalid",
        ),
        "resource_summary": dict(resource_summary),
        "segment_topology": topology,
        "segment_topology_sha256": _safe_hash(topology),
        "segments_commitment": _require_hex64(
            receipt.get("segments_commitment"),
            "document_segments_hash_invalid",
        ),
        "status": status,
    }
    return body


def _run_document_twice(
    *,
    runtime: object,
    runtime_commitment: str,
    fixture: PublicDocumentFixture,
) -> dict[str, object]:
    first = _run_document_once(
        runtime=runtime,
        runtime_commitment=runtime_commitment,
        fixture=fixture,
    )
    second = _run_document_once(
        runtime=runtime,
        runtime_commitment=runtime_commitment,
        fixture=fixture,
    )
    first_raw = _canonical_bytes(first)
    second_raw = _canonical_bytes(second)
    if first_raw != second_raw:
        return _minimal_negative_outcome(
            fixture=fixture,
            status=_REPEAT_MISMATCH_STATUS,
            failure_code="DOCUMENT_REPEAT_BYTE_MISMATCH",
        )
    return {
        **first,
        "repeat_byte_exact": True,
        "repeat_count": REPEAT_COUNT,
        "repeat_outcome_sha256": hashlib.sha256(first_raw).hexdigest(),
    }


def _reset_cuda_peaks() -> None:
    try:
        leaf_qualification._reset_cuda_peaks()
    except Exception as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_cuda_peak_reset_failed"
        ) from exc


def _resource_peaks(outcomes: Sequence[Mapping[str, object]]) -> dict[str, int]:
    summaries = [row["resource_summary"] for row in outcomes]
    if any(type(row) is not dict for row in summaries):
        raise FixedDocumentEnvelopeQualificationError(
            "document_resource_summary_invalid"
        )
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        torch.cuda.synchronize(0)
        cuda_allocated = int(torch.cuda.max_memory_allocated(0))
        cuda_reserved = int(torch.cuda.max_memory_reserved(0))
    except Exception as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_cuda_peak_read_failed"
        ) from exc
    return {
        "cuda_max_memory_allocated_bytes": cuda_allocated,
        "cuda_max_memory_reserved_bytes": cuda_reserved,
        "max_leaf_call_count": max(
            (int(row["leaf_call_count"]) for row in summaries), default=0
        ),
        "max_projected_relation_count": max(
            (
                int(row["projected_relation_count"])
                for row in summaries
            ),
            default=0,
        ),
        "max_reported_success_candidate_count": max(
            (
                int(row["reported_success_candidate_count"])
                for row in summaries
            ),
            default=0,
        ),
        "max_reported_success_forward_batch_count": max(
            (
                int(row["reported_success_forward_batch_count"])
                for row in summaries
            ),
            default=0,
        ),
        "max_root_lexical_token_count": max(
            (
                int(row["root_lexical_token_count"])
                for row in summaries
            ),
            default=0,
        ),
        "max_segment_count": max(
            (int(row["segment_count"]) for row in summaries), default=0
        ),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _negative_resource_peaks() -> dict[str, int]:
    """Bounded, content-free resource terminal when detailed rows are absent."""

    try:
        process_max_rss_kib = max(
            1, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        )
    except Exception:
        process_max_rss_kib = 1
    return {
        "cuda_max_memory_allocated_bytes": 0,
        "cuda_max_memory_reserved_bytes": 0,
        "max_leaf_call_count": 0,
        "max_projected_relation_count": 0,
        "max_reported_success_candidate_count": 0,
        "max_reported_success_forward_batch_count": 0,
        "max_root_lexical_token_count": 0,
        "max_segment_count": 0,
        "process_max_rss_kib": process_max_rss_kib,
    }


def _validate_success_canary(value: object) -> None:
    if type(value) is not dict:
        raise FixedDocumentEnvelopeQualificationError(
            "document_teacher_forced_canary_contract_invalid"
        )
    supplied = value.get("self_sha256")
    if (
        _HEX64.fullmatch(supplied) is None
        if isinstance(supplied, str)
        else True
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_teacher_forced_canary_contract_invalid"
        )
    if (
        supplied
        != _safe_hash(
            {
                key: child
                for key, child in value.items()
                if key != "self_sha256"
            }
        )
        or value.get("short_strategy_vs_full_reference_exact") is not True
        or value.get("long_repeat_byte_exact") is not True
        or value.get("fallback_independent_full_reference_passed") is not True
        or value.get("free_form_generation_count") != 0
        or value.get("schema") != memory_safe_qwen.FIXED_CANARY_SCHEMA
        or value.get("short_pair_sha256")
        != memory_safe_qwen.FIXED_SHORT_CANARY_PAIR_SHA256
        or value.get("long_pair_sha256")
        != memory_safe_qwen.FIXED_LONG_CANARY_PAIR_SHA256
        or value.get("strategy")
        not in {
            memory_safe_qwen.SPARSE_STRATEGY,
            memory_safe_qwen.FALLBACK_STRATEGY,
        }
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_teacher_forced_canary_contract_invalid"
        )


def run_fixed_document_envelope_qualification_shard(
    *,
    model_root: Path,
    manifest: worker.ModelAssetManifest,
    upstream_aggregate_receipt: Path,
    output_root: Path,
    shard_index: int,
    shard_count: int,
) -> Mapping[str, object]:
    """Execute one immutable two-GPU shard with no caller content surface."""

    if (
        not isinstance(output_root, Path)
        or not isinstance(upstream_aggregate_receipt, Path)
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index not in {0, 1}
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count != SHARD_COUNT
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_coordinate_invalid"
        )
    output_path = output_root / SHARD_OUTPUT_NAME
    _prepare_output_path(output_path)
    upstream = _load_upstream_leaf_aggregate(upstream_aggregate_receipt)
    _verify_model_binding(model_root=model_root, manifest=manifest)
    manifest_binding = _manifest_commitments(manifest)
    if upstream.get("manifest_commitments") != manifest_binding:
        raise FixedDocumentEnvelopeQualificationError(
            "document_upstream_manifest_mismatch"
        )
    implementation = _implementation_closure()
    runtime = _load_exact_runtime(model_root=model_root, manifest=manifest)
    runtime_commitment = _validate_exact_runtime(
        runtime=runtime, manifest=manifest
    )
    if upstream.get("runtime_commitment") != runtime_commitment:
        raise FixedDocumentEnvelopeQualificationError(
            "document_upstream_runtime_mismatch"
        )
    fixtures = tuple(
        PUBLIC_DOCUMENT_FIXTURES[ordinal]
        for ordinal in SHARD_FIXTURE_ORDINALS[shard_index]
    )
    canary: dict[str, object] | None
    canary_failure_code: str | None = None
    try:
        _reset_cuda_peaks()
        canary = dict(
            leaf_qualification._run_fixed_teacher_forced_canary(runtime)
        )
        _validate_success_canary(canary)
    except Exception:
        canary = None
        canary_failure_code = "DOCUMENT_TEACHER_FORCED_CANARY_FAILED"
    if canary_failure_code is None:
        outcomes = [
            _run_document_twice(
                runtime=runtime,
                runtime_commitment=runtime_commitment,
                fixture=fixture,
            )
            for fixture in fixtures
        ]
    else:
        outcomes = [
            _minimal_negative_outcome(
                fixture=fixture,
                status=_CANARY_NOT_EXECUTED_STATUS,
                failure_code=canary_failure_code,
            )
            for fixture in fixtures
        ]
    failure_codes = sorted(
        {
            row["failure_code"]
            for row in outcomes
            if row["failure_code"] is not None
        }
    )
    if canary is not None and all(
        type(row["resource_summary"]) is dict for row in outcomes
    ):
        try:
            resource_peaks = _resource_peaks(outcomes)
        except Exception:
            resource_peaks = _negative_resource_peaks()
            failure_codes = sorted(
                {*failure_codes, _RESOURCE_PEAK_FAILURE_CODE}
            )
    else:
        resource_peaks = _negative_resource_peaks()
    passed = (
        canary is not None
        and not failure_codes
        and all(
            row["status"] == _SUCCESS_STATUS
            and row["repeat_byte_exact"] is True
            for row in outcomes
        )
    )
    body: dict[str, object] = {
        "claim_scope": "fixed_public_non_scoring_document_envelope_runtime_compatibility_only",
        "canary_passed": canary is not None,
        "counters": _zero_counters(),
        "downstream_eligible": False,
        "effect_quality_gate_added": False,
        "effect_or_quality_measurement": False,
        "fixture_commitments": {
            row.fixture_id: row.fixture_commitment for row in fixtures
        },
        "fixture_count": len(fixtures),
        "fixture_ordinals": [row.ordinal for row in fixtures],
        "fixture_suite_sha256": FIXTURE_SUITE_SHA256,
        "formal_effect_evidence": False,
        "formal_leaf_authority_established_by_generic_envelope": False,
        "functional_extracted_branch_coverage_required": True,
        "in_process_private_leaf_consistency_validation": True,
        "implementation_closure": implementation,
        "implementation_closure_sha256": _safe_hash(implementation),
        "manifest_commitments": manifest_binding,
        "outcomes": outcomes,
        "outcomes_commitment": _safe_hash(outcomes),
        "qualification_passed": passed,
        "qualification_failure_codes": failure_codes,
        "private_leaf_evidence_retained": False,
        "repeat_byte_exact": all(
            row["repeat_byte_exact"] is True for row in outcomes
        ),
        "repeat_count": REPEAT_COUNT,
        "resource_peaks": resource_peaks,
        "runtime_commitment": runtime_commitment,
        "schema": SHARD_RECEIPT_SCHEMA,
        "segmentation_policy_sha256": (
            document_envelope.SEGMENTATION_POLICY_SHA256
        ),
        "shard_count": SHARD_COUNT,
        "shard_index": shard_index,
        "teacher_forced_canary": canary,
        "teacher_forced_canary_self_sha256": (
            None if canary is None else canary["self_sha256"]
        ),
        "upstream_leaf_aggregate_file_sha256": upstream["_file_sha256"],
        "upstream_leaf_aggregate_self_sha256": upstream["self_sha256"],
        "version": VERSION,
    }
    receipt = {**body, "self_sha256": _safe_hash(body)}
    _publish_once(output_path, _canonical_bytes(receipt))
    return MappingProxyType(receipt)


def _load_shard_receipt(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_receipt_unreadable"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or not raw
        or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_receipt_size_invalid"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_receipt_json_invalid"
        ) from exc
    if (
        type(value) is not dict
        or _canonical_bytes(value) != raw
        or value.get("schema") != SHARD_RECEIPT_SCHEMA
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_receipt_canonical_invalid"
        )
    supplied = _require_hex64(
        value.get("self_sha256"), "document_shard_receipt_self_invalid"
    )
    body = {
        key: child for key, child in value.items() if key != "self_sha256"
    }
    if supplied != _safe_hash(body):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_receipt_self_mismatch"
        )
    return value


_OUTCOME_FIELDS = {
    "disposition_counts",
    "document_envelope_receipt_sha256",
    "document_envelope_self_sha256",
    "exact_runtime_binding_observed_on_extracted_leaves",
    "extracted_leaf_count",
    "failure_code",
    "fixture_commitment",
    "fixture_id",
    "input_sha256",
    "ordinal",
    "partial_projection_available",
    "projection_commitment",
    "repeat_byte_exact",
    "repeat_count",
    "repeat_outcome_sha256",
    "resource_summary",
    "segment_topology",
    "segment_topology_sha256",
    "segments_commitment",
    "status",
}
_DOCUMENT_RESOURCE_FIELDS = {
    "declared_candidate_bound",
    "declared_forward_batch_call_bound",
    "declared_leaf_batch_capacity",
    "leaf_call_count",
    "projected_mention_count",
    "projected_relation_count",
    "reported_success_candidate_count",
    "reported_success_forward_batch_count",
    "root_byte_count",
    "root_lexical_token_count",
    "segment_count",
}
_SEGMENT_TOPOLOGY_FIELDS = {
    "chunk_count",
    "chunk_index",
    "disposition",
    "leaf_called",
    "leaf_eligible",
    "lexical_token_count",
    "parent_sentence_id",
    "segment_id",
}
_RESOURCE_PEAK_FIELDS = {
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "max_leaf_call_count",
    "max_projected_relation_count",
    "max_reported_success_candidate_count",
    "max_reported_success_forward_batch_count",
    "max_root_lexical_token_count",
    "max_segment_count",
    "process_max_rss_kib",
}


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_safe_outcome(
    value: Mapping[str, object], fixture: PublicDocumentFixture
) -> None:
    if set(value) != _OUTCOME_FIELDS:
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_outcome_fields_invalid"
        )
    if (
        value.get("fixture_id") != fixture.fixture_id
        or value.get("fixture_commitment") != fixture.fixture_commitment
        or value.get("input_sha256") != fixture.input_sha256
        or value.get("ordinal") != fixture.ordinal
        or isinstance(value.get("ordinal"), bool)
        or value.get("repeat_count") != REPEAT_COUNT
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_outcome_contract_invalid"
        )
    status = value.get("status")
    if status in {_REPEAT_MISMATCH_STATUS, _CANARY_NOT_EXECUTED_STATUS}:
        expected_code = (
            "DOCUMENT_REPEAT_BYTE_MISMATCH"
            if status == _REPEAT_MISMATCH_STATUS
            else "DOCUMENT_TEACHER_FORCED_CANARY_FAILED"
        )
        if (
            value.get("failure_code") != expected_code
            or value.get("repeat_byte_exact") is not False
            or value.get("exact_runtime_binding_observed_on_extracted_leaves")
            is not False
            or value.get("extracted_leaf_count") != 0
            or value.get("partial_projection_available") is not False
            or value.get("disposition_counts") is not None
            or value.get("resource_summary") is not None
            or value.get("segment_topology") is not None
            or any(
                value.get(field) is not None
                for field in (
                    "document_envelope_receipt_sha256",
                    "document_envelope_self_sha256",
                    "projection_commitment",
                    "repeat_outcome_sha256",
                    "segment_topology_sha256",
                    "segments_commitment",
                )
            )
        ):
            raise FixedDocumentEnvelopeQualificationError(
                "document_safe_negative_outcome_invalid"
            )
        return
    if status not in {
        _SUCCESS_STATUS,
        _NO_RELATION_STATUS,
        _TYPED_FAILURE_STATUS,
    }:
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_outcome_status_invalid"
        )
    for field in (
        "document_envelope_receipt_sha256",
        "document_envelope_self_sha256",
        "fixture_commitment",
        "input_sha256",
        "projection_commitment",
        "repeat_outcome_sha256",
        "segment_topology_sha256",
        "segments_commitment",
    ):
        _require_hex64(value.get(field), "document_safe_outcome_hash_invalid")
    if value.get("repeat_byte_exact") is not True:
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_outcome_repeat_invalid"
        )
    topology = value.get("segment_topology")
    plans = document_envelope.plan_document_segments(fixture.story_text)
    if type(topology) is not list or len(topology) != len(plans):
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_topology_invalid"
        )
    for observed, plan in zip(topology, plans, strict=True):
        disposition = observed.get("disposition") if type(observed) is dict else None
        if (
            type(observed) is not dict
            or set(observed) != _SEGMENT_TOPOLOGY_FIELDS
            or observed.get("segment_id") != plan.segment_id
            or observed.get("parent_sentence_id") != plan.parent_sentence_id
            or observed.get("chunk_index") != plan.chunk_index
            or isinstance(observed.get("chunk_index"), bool)
            or observed.get("chunk_count") != plan.chunk_count
            or isinstance(observed.get("chunk_count"), bool)
            or observed.get("lexical_token_count")
            != plan.lexical_token_count
            or isinstance(observed.get("lexical_token_count"), bool)
            or observed.get("leaf_eligible") is not plan.leaf_eligible
            or (
                not plan.leaf_eligible
                and (
                    observed.get("leaf_called") is not False
                    or disposition
                    != document_envelope.SegmentDisposition.CONTEXT_ONLY_SHORT_SENTENCE.value
                )
            )
            or (
                plan.leaf_eligible
                and (
                    disposition
                    not in {
                        document_envelope.SegmentDisposition.EXTRACTED.value,
                        document_envelope.SegmentDisposition.NO_RELATION.value,
                        document_envelope.SegmentDisposition.TYPED_FAILURE.value,
                    }
                    or (
                        observed.get("leaf_called") is False
                        and disposition
                        != document_envelope.SegmentDisposition.TYPED_FAILURE.value
                    )
                    or not isinstance(observed.get("leaf_called"), bool)
                )
            )
        ):
            raise FixedDocumentEnvelopeQualificationError(
                "document_safe_topology_invalid"
            )
    if value.get("segment_topology_sha256") != _safe_hash(topology):
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_topology_hash_mismatch"
        )
    expected_dispositions = {
        disposition.value: sum(
            row["disposition"] == disposition.value for row in topology
        )
        for disposition in document_envelope.SegmentDisposition
    }
    if value.get("disposition_counts") != expected_dispositions:
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_disposition_counts_invalid"
        )
    summary = value.get("resource_summary")
    extracted_count = expected_dispositions[
        document_envelope.SegmentDisposition.EXTRACTED.value
    ]
    typed_count = expected_dispositions[
        document_envelope.SegmentDisposition.TYPED_FAILURE.value
    ]
    no_relation_count = expected_dispositions[
        document_envelope.SegmentDisposition.NO_RELATION.value
    ]
    observed_leaf_calls = sum(bool(row["leaf_called"]) for row in topology)
    if (
        type(summary) is not dict
        or set(summary) != _DOCUMENT_RESOURCE_FIELDS
        or any(not _is_nonnegative_int(row) for row in summary.values())
        or summary["declared_candidate_bound"]
        != observed_leaf_calls
        * document_envelope.MAXIMUM_CANDIDATES_PER_SINGLE_SENTENCE_LEAF
        or summary["declared_forward_batch_call_bound"]
        != observed_leaf_calls
        * document_envelope.MAXIMUM_FORWARD_BATCH_CALLS_PER_SINGLE_SENTENCE_LEAF
        or summary["declared_leaf_batch_capacity"]
        != document_envelope.SCORING_BATCH_SIZE
        or summary["leaf_call_count"] != observed_leaf_calls
        or summary["root_byte_count"]
        != len(fixture.story_text.encode("utf-8"))
        or summary["root_lexical_token_count"] != fixture.lexical_token_count
        or summary["segment_count"] != len(plans)
        or summary["projected_relation_count"] < extracted_count
        or summary["projected_relation_count"]
        > document_envelope.MAXIMUM_PROJECTED_RELATIONS
        or summary["projected_mention_count"]
        != 3 * summary["projected_relation_count"]
        or not 0 <= summary["reported_success_candidate_count"]
        <= summary["declared_candidate_bound"]
        or not 0 <= summary["reported_success_forward_batch_count"]
        <= summary["declared_forward_batch_call_bound"]
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_resource_summary_invalid"
        )
    if (
        value.get("extracted_leaf_count") != extracted_count
        or value.get("exact_runtime_binding_observed_on_extracted_leaves")
        is not (extracted_count > 0)
        or value.get("partial_projection_available")
        is not (typed_count == 0 and summary["projected_relation_count"] > 0)
        or (
            status == _SUCCESS_STATUS
            and (
                value.get("failure_code") is not None
                or typed_count != 0
                or no_relation_count != 0
                or extracted_count != fixture.expected_leaf_call_count
            )
        )
        or (
            status == _NO_RELATION_STATUS
            and (
                value.get("failure_code")
                != "DOCUMENT_FUNCTIONAL_EXTRACTED_BRANCH_NOT_EXERCISED"
                or typed_count != 0
                or no_relation_count == 0
                or extracted_count >= fixture.minimum_extracted_leaf_count
            )
        )
        or (
            status == _TYPED_FAILURE_STATUS
            and (
                value.get("failure_code")
                != "DOCUMENT_TYPED_FAILURE_REPORTED"
                or typed_count == 0
            )
        )
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_safe_outcome_semantics_invalid"
        )


def _validate_resource_peaks(
    value: object,
    outcomes: Sequence[Mapping[str, object]],
    *,
    collection_failed: bool,
) -> None:
    if (
        type(value) is not dict
        or set(value) != _RESOURCE_PEAK_FIELDS
        or any(not _is_nonnegative_int(row) for row in value.values())
        or value["process_max_rss_kib"] <= 0
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_resource_peaks_invalid"
        )
    if collection_failed or any(
        row.get("resource_summary") is None for row in outcomes
    ):
        if any(value[key] != 0 for key in _RESOURCE_PEAK_FIELDS - {"process_max_rss_kib"}):
            raise FixedDocumentEnvelopeQualificationError(
                "document_negative_resource_peaks_invalid"
            )
        return
    if (
        value["cuda_max_memory_allocated_bytes"] <= 0
        or value["cuda_max_memory_reserved_bytes"] <= 0
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_resource_peaks_invalid"
        )
    summaries = [row["resource_summary"] for row in outcomes]
    expected = {
        "max_leaf_call_count": max(int(row["leaf_call_count"]) for row in summaries),
        "max_projected_relation_count": max(
            int(row["projected_relation_count"]) for row in summaries
        ),
        "max_reported_success_candidate_count": max(
            int(row["reported_success_candidate_count"]) for row in summaries
        ),
        "max_reported_success_forward_batch_count": max(
            int(row["reported_success_forward_batch_count"])
            for row in summaries
        ),
        "max_root_lexical_token_count": max(
            int(row["root_lexical_token_count"]) for row in summaries
        ),
        "max_segment_count": max(int(row["segment_count"]) for row in summaries),
    }
    if any(value[key] != expected[key] for key in expected):
        raise FixedDocumentEnvelopeQualificationError(
            "document_resource_peaks_semantics_invalid"
        )


def _validate_shard_receipt(value: Mapping[str, object]) -> None:
    index = value.get("shard_index")
    expected_root_fields = {
        "claim_scope",
        "canary_passed",
        "counters",
        "downstream_eligible",
        "effect_quality_gate_added",
        "effect_or_quality_measurement",
        "fixture_commitments",
        "fixture_count",
        "fixture_ordinals",
        "fixture_suite_sha256",
        "formal_effect_evidence",
        "formal_leaf_authority_established_by_generic_envelope",
        "functional_extracted_branch_coverage_required",
        "implementation_closure",
        "implementation_closure_sha256",
        "in_process_private_leaf_consistency_validation",
        "manifest_commitments",
        "outcomes",
        "outcomes_commitment",
        "private_leaf_evidence_retained",
        "qualification_passed",
        "qualification_failure_codes",
        "repeat_byte_exact",
        "repeat_count",
        "resource_peaks",
        "runtime_commitment",
        "schema",
        "segmentation_policy_sha256",
        "self_sha256",
        "shard_count",
        "shard_index",
        "teacher_forced_canary",
        "teacher_forced_canary_self_sha256",
        "upstream_leaf_aggregate_file_sha256",
        "upstream_leaf_aggregate_self_sha256",
        "version",
    }
    expected = tuple(
        PUBLIC_DOCUMENT_FIXTURES[ordinal]
        for ordinal in (
            SHARD_FIXTURE_ORDINALS[index]
            if index in {0, 1} and not isinstance(index, bool)
            else ()
        )
    )
    outcomes = value.get("outcomes")
    implementation = value.get("implementation_closure")
    canary = value.get("teacher_forced_canary")
    if (
        set(value) != expected_root_fields
        or index not in {0, 1}
        or isinstance(index, bool)
        or value.get("shard_count") != SHARD_COUNT
        or value.get("fixture_suite_sha256") != FIXTURE_SUITE_SHA256
        or value.get("fixture_ordinals") != [row.ordinal for row in expected]
        or value.get("fixture_count") != len(expected)
        or value.get("fixture_commitments")
        != {row.fixture_id: row.fixture_commitment for row in expected}
        or value.get("claim_scope")
        != "fixed_public_non_scoring_document_envelope_runtime_compatibility_only"
        or value.get("downstream_eligible") is not False
        or value.get("formal_effect_evidence") is not False
        or value.get("effect_quality_gate_added") is not False
        or value.get("effect_or_quality_measurement") is not False
        or value.get("formal_leaf_authority_established_by_generic_envelope")
        is not False
        or value.get("functional_extracted_branch_coverage_required")
        is not True
        or value.get("in_process_private_leaf_consistency_validation")
        is not True
        or value.get("private_leaf_evidence_retained") is not False
        or value.get("counters") != _zero_counters()
        or value.get("repeat_count") != REPEAT_COUNT
        or not isinstance(value.get("repeat_byte_exact"), bool)
        or not isinstance(value.get("qualification_passed"), bool)
        or not isinstance(value.get("canary_passed"), bool)
        or value.get("segmentation_policy_sha256")
        != document_envelope.SEGMENTATION_POLICY_SHA256
        or type(implementation) is not dict
        or value.get("implementation_closure_sha256")
        != _safe_hash(implementation)
        or type(outcomes) is not list
        or [row.get("ordinal") for row in outcomes]
        != [row.ordinal for row in expected]
        or value.get("outcomes_commitment") != _safe_hash(outcomes)
        or value.get("version") != VERSION
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_receipt_contract_invalid"
        )
    for row, fixture in zip(outcomes, expected, strict=True):
        if type(row) is not dict:
            raise FixedDocumentEnvelopeQualificationError(
                "document_safe_outcome_fields_invalid"
            )
        _validate_safe_outcome(row, fixture)
    outcome_failure_codes = sorted(
        {
            row["failure_code"]
            for row in outcomes
            if row["failure_code"] is not None
        }
    )
    supplied_failure_codes = value.get("qualification_failure_codes")
    if (
        type(supplied_failure_codes) is not list
        or supplied_failure_codes != sorted(set(supplied_failure_codes))
        or any(
            not isinstance(code, str)
            or code not in _QUALIFICATION_FAILURE_CODES
            for code in supplied_failure_codes
        )
        or any(
            code not in supplied_failure_codes
            for code in outcome_failure_codes
        )
        or any(
            code not in outcome_failure_codes
            and code != _RESOURCE_PEAK_FAILURE_CODE
            for code in supplied_failure_codes
        )
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_failure_codes_invalid"
        )
    resource_peak_collection_failed = (
        _RESOURCE_PEAK_FAILURE_CODE in supplied_failure_codes
    )
    canary_passed = value["canary_passed"]
    if canary_passed:
        _validate_success_canary(canary)
        if value.get("teacher_forced_canary_self_sha256") != canary.get(
            "self_sha256"
        ):
            raise FixedDocumentEnvelopeQualificationError(
                "document_teacher_forced_canary_binding_invalid"
            )
    elif (
        canary is not None
        or value.get("teacher_forced_canary_self_sha256") is not None
        or outcome_failure_codes
        != ["DOCUMENT_TEACHER_FORCED_CANARY_FAILED"]
        or any(
            row["status"] != _CANARY_NOT_EXECUTED_STATUS for row in outcomes
        )
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_teacher_forced_canary_negative_invalid"
        )
    expected_passed = (
        canary_passed
        and not supplied_failure_codes
        and all(
            row["status"] == _SUCCESS_STATUS
            and row["repeat_byte_exact"] is True
            for row in outcomes
        )
    )
    if (
        value.get("qualification_passed") is not expected_passed
        or value.get("repeat_byte_exact")
        is not all(row["repeat_byte_exact"] is True for row in outcomes)
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_qualification_semantics_invalid"
        )
    _validate_resource_peaks(
        value.get("resource_peaks"),
        outcomes,
        collection_failed=resource_peak_collection_failed,
    )
    for field in (
        "runtime_commitment",
        "upstream_leaf_aggregate_file_sha256",
        "upstream_leaf_aggregate_self_sha256",
    ):
        _require_hex64(value.get(field), "document_shard_binding_invalid")
    manifest = value.get("manifest_commitments")
    if (
        type(manifest) is not dict
        or set(manifest)
        != {
            "manifest_file_sha256",
            "manifest_self_sha256",
            "model_tree_sha256",
        }
        or any(
            not isinstance(row, str) or _HEX64.fullmatch(row) is None
            for row in manifest.values()
        )
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_shard_binding_invalid"
        )


def aggregate_fixed_document_envelope_qualification(
    *, shard_receipts: tuple[Path, Path], output_root: Path
) -> Mapping[str, object]:
    """Pure-offline, manual-only aggregate of the two immutable shards."""

    if (
        type(shard_receipts) is not tuple
        or len(shard_receipts) != SHARD_COUNT
        or any(not isinstance(path, Path) for path in shard_receipts)
        or not isinstance(output_root, Path)
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_aggregate_arguments_invalid"
        )
    output_path = output_root / AGGREGATE_OUTPUT_NAME
    _prepare_output_path(output_path)
    rows = [_load_shard_receipt(path) for path in shard_receipts]
    for row in rows:
        _validate_shard_receipt(row)
    rows.sort(key=lambda row: int(row["shard_index"]))
    if [row["shard_index"] for row in rows] != [0, 1]:
        raise FixedDocumentEnvelopeQualificationError(
            "document_aggregate_shards_invalid"
        )
    consistency = (
        "fixture_suite_sha256",
        "implementation_closure",
        "implementation_closure_sha256",
        "manifest_commitments",
        "runtime_commitment",
        "segmentation_policy_sha256",
        "upstream_leaf_aggregate_file_sha256",
        "upstream_leaf_aggregate_self_sha256",
        "version",
    )
    if any(rows[0][field] != rows[1][field] for field in consistency):
        raise FixedDocumentEnvelopeQualificationError(
            "document_aggregate_binding_mismatch"
        )
    if (
        rows[0]["canary_passed"] is True
        and rows[1]["canary_passed"] is True
        and (
            rows[0]["teacher_forced_canary"]
            != rows[1]["teacher_forced_canary"]
            or rows[0]["teacher_forced_canary_self_sha256"]
            != rows[1]["teacher_forced_canary_self_sha256"]
        )
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_aggregate_canary_mismatch"
        )
    current_implementation = _implementation_closure()
    if (
        rows[0]["implementation_closure"] != current_implementation
        or rows[0]["implementation_closure_sha256"]
        != _safe_hash(current_implementation)
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_aggregate_implementation_drifted"
        )
    outcomes = sorted(
        [outcome for row in rows for outcome in row["outcomes"]],
        key=lambda row: int(row["ordinal"]),
    )
    if [row["ordinal"] for row in outcomes] != list(
        range(len(PUBLIC_DOCUMENT_FIXTURES))
    ):
        raise FixedDocumentEnvelopeQualificationError(
            "document_aggregate_fixture_union_invalid"
        )
    resource_keys = tuple(rows[0]["resource_peaks"])
    if set(resource_keys) != set(rows[1]["resource_peaks"]):
        raise FixedDocumentEnvelopeQualificationError(
            "document_aggregate_resource_fields_invalid"
        )
    resource_peaks = {
        key: max(int(row["resource_peaks"][key]) for row in rows)
        for key in resource_keys
    }
    aggregate_passed = all(
        row["qualification_passed"] is True for row in rows
    )
    aggregate_canary_passed = all(row["canary_passed"] is True for row in rows)
    failure_codes = sorted(
        {
            code
            for row in rows
            for code in row["qualification_failure_codes"]
        }
    )
    aggregate_canary = rows[0]["teacher_forced_canary"] if aggregate_canary_passed else None
    body: dict[str, object] = {
        "claim_scope": "fixed_public_non_scoring_document_envelope_runtime_compatibility_only",
        "canary_passed": aggregate_canary_passed,
        "counters": _zero_counters(aggregate=True),
        "downstream_eligible": False,
        "effect_quality_gate_added": False,
        "effect_or_quality_measurement": False,
        "fixture_commitments": dict(FIXTURE_COMMITMENTS),
        "fixture_count": len(PUBLIC_DOCUMENT_FIXTURES),
        "fixture_ordinals": list(range(len(PUBLIC_DOCUMENT_FIXTURES))),
        "fixture_suite_sha256": FIXTURE_SUITE_SHA256,
        "formal_effect_evidence": False,
        "formal_leaf_authority_established_by_generic_envelope": False,
        "functional_extracted_branch_coverage_required": True,
        "in_process_private_leaf_consistency_validation": True,
        "implementation_closure": current_implementation,
        "implementation_closure_sha256": _safe_hash(current_implementation),
        "manifest_commitments": rows[0]["manifest_commitments"],
        "outcome_status_counts": {
            "executed_without_typed_failure": sum(
                row["status"] == _SUCCESS_STATUS
                for row in outcomes
            ),
            "functional_extracted_branch_not_exercised": sum(
                row["status"] == _NO_RELATION_STATUS for row in outcomes
            ),
            "not_executed_after_canary_failure": sum(
                row["status"] == _CANARY_NOT_EXECUTED_STATUS
                for row in outcomes
            ),
            "repeat_mismatch": sum(
                row["status"] == _REPEAT_MISMATCH_STATUS for row in outcomes
            ),
            "typed_failure": sum(
                row["status"] == _TYPED_FAILURE_STATUS for row in outcomes
            ),
        },
        "outcomes_commitment": _safe_hash(outcomes),
        "qualification_passed": aggregate_passed,
        "qualification_failure_codes": failure_codes,
        "private_leaf_evidence_retained": False,
        "repeat_byte_exact": all(
            row["repeat_byte_exact"] is True for row in rows
        ),
        "repeat_count": REPEAT_COUNT,
        "resource_peaks": resource_peaks,
        "runtime_commitment": rows[0]["runtime_commitment"],
        "schema": AGGREGATE_RECEIPT_SCHEMA,
        "segmentation_policy_sha256": rows[0][
            "segmentation_policy_sha256"
        ],
        "shard_count": SHARD_COUNT,
        "shard_receipt_self_sha256": {
            str(row["shard_index"]): row["self_sha256"] for row in rows
        },
        "teacher_forced_canary": aggregate_canary,
        "teacher_forced_canary_self_sha256": (
            rows[0]["teacher_forced_canary_self_sha256"]
            if aggregate_canary_passed
            else None
        ),
        "upstream_leaf_aggregate_file_sha256": rows[0][
            "upstream_leaf_aggregate_file_sha256"
        ],
        "upstream_leaf_aggregate_self_sha256": rows[0][
            "upstream_leaf_aggregate_self_sha256"
        ],
        "version": VERSION,
    }
    receipt = {**body, "self_sha256": _safe_hash(body)}
    _publish_once(output_path, _canonical_bytes(receipt))
    return MappingProxyType(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed source-free document-envelope qualification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--model-root", required=True, type=Path)
    shard.add_argument("--model-manifest", required=True, type=Path)
    shard.add_argument(
        "--upstream-aggregate-receipt", required=True, type=Path
    )
    shard.add_argument("--output-root", required=True, type=Path)
    shard.add_argument("--shard-index", required=True, type=int, choices=(0, 1))
    shard.add_argument("--shard-count", required=True, type=int, choices=(2,))
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--shard-0-receipt", required=True, type=Path)
    aggregate.add_argument("--shard-1-receipt", required=True, type=Path)
    aggregate.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "shard":
        manifest = worker.load_model_asset_manifest(
            manifest_path=arguments.model_manifest,
            model_root=arguments.model_root,
        )
        receipt = run_fixed_document_envelope_qualification_shard(
            model_root=arguments.model_root,
            manifest=manifest,
            upstream_aggregate_receipt=(
                arguments.upstream_aggregate_receipt
            ),
            output_root=arguments.output_root,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    else:
        receipt = aggregate_fixed_document_envelope_qualification(
            shard_receipts=(
                arguments.shard_0_receipt,
                arguments.shard_1_receipt,
            ),
            output_root=arguments.output_root,
        )
    print(receipt["self_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_OUTPUT_NAME",
    "AGGREGATE_RECEIPT_SCHEMA",
    "FIXTURE_COMMITMENTS",
    "FIXTURE_SUITE_SHA256",
    "FixedDocumentEnvelopeQualificationError",
    "PUBLIC_DOCUMENT_FIXTURES",
    "PublicDocumentFixture",
    "REPEAT_COUNT",
    "SHARD_COUNT",
    "SHARD_OUTPUT_NAME",
    "SHARD_RECEIPT_SCHEMA",
    "VERSION",
    "aggregate_fixed_document_envelope_qualification",
    "main",
    "run_fixed_document_envelope_qualification_shard",
]

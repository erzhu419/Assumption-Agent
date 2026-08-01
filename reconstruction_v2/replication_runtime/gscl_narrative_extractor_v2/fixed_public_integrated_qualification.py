"""Fixed public qualification of extractor v2 and all four intrinsic arms.

This is an iteration of the source-free, non-scoring qualification harness.
It is not a benchmark study and has no label, reference-answer, online
evaluator, API, network, or caller-supplied story/scorer surface.

Three program-owned public fixtures are extracted twice with the exact Qwen
runtime.  The resulting query/candidate triple is then passed twice through
the fixed semantic-only, legacy-keyword, flat-label/no-verifier, and full
GSCL paths.  The two structural arms must consume the identical polynomial
proposal set; only the full arm may call the score-free structural checker.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
from types import MappingProxyType
from typing import Mapping, Sequence

from assumption_agent import gscl_arn_intrinsic_arms_v1 as arms_v1
from assumption_agent import gscl_arn_intrinsic_arms_v2 as arms_v2
from assumption_agent.gscl_narrative_correspondence_v1 import (
    GeneratorKind,
    Mention,
    NarrativeExtraction,
    SemanticScoreTable,
)
from assumption_agent.gscl_unit_mapping_v2 import (
    UnitMappingSearchConfigV2,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    contract as v1_contract,
)
from replication_runtime.gscl_narrative_extractor_v1 import worker

from . import closed_choice
from . import fixed_public_qualification as extractor_qualification


VERSION = "gscl_v2_fixed_public_integrated_qualification_v1"
RECEIPT_SCHEMA = f"{VERSION}.safe_receipt.v1"
OUTPUT_NAME = "integrated_qualification.safe.json"
REPEAT_COUNT = 2
PUBLIC_ITEM_FIXTURE_ORDINALS = (0, 1, 2)
MAXIMUM_SAFE_RECEIPT_BYTES = 2 * 1024 * 1024
MAXIMUM_IMPLEMENTATION_FILE_BYTES = 4 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_LEXICAL = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)

LEGACY_FEATURE_IDS = (
    "generator_count",
    "object_count",
    "relation_count",
    "state_change_count",
    "temporal_count",
    "causal_count",
    "positive_count",
    "neutral_count",
    "negative_count",
    "temporal_forward_count",
    "causal_forward_count",
)


class FixedPublicIntegratedQualificationError(RuntimeError):
    """Stable failure without fixture text or model output."""


def _canonical_bytes(value: object) -> bytes:
    return v1_contract.canonical_json_bytes(value)


def _safe_hash(value: object) -> str:
    return v1_contract.semantic_sha256(value)


def _require_hex64(value: object, issue: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FixedPublicIntegratedQualificationError(issue)
    return value


def _zero_counters() -> dict[str, int]:
    return {
        "api_access_count": 0,
        "free_form_generation_count": 0,
        "label_access_count": 0,
        "network_access_count": 0,
        "source_access_count": 0,
    }


def _implementation_closure() -> dict[str, str]:
    project_root = Path(__file__).parents[2]
    package_root = Path(__file__).parent
    v1_root = project_root / "replication_runtime" / (
        "gscl_narrative_extractor_v1"
    )
    paths = {
        "integrated_qualification.py": Path(__file__),
        "integrated_qualification.service": (
            project_root
            / "manifests"
            / "gscl_narrative_extractor_v2_fixed_integrated_qualification.service"
        ),
        "v2_package_init.py": package_root / "__init__.py",
        "v2_closed_choice.py": package_root / "closed_choice.py",
        "v2_contract.py": package_root / "contract.py",
        "v2_memory_safe_qwen.py": package_root / "memory_safe_qwen.py",
        "v2_fixed_extractor_qualification.py": (
            package_root / "fixed_public_qualification.py"
        ),
        "v1_closed_choice_worker.py": (
            v1_root / "closed_choice_worker.py"
        ),
        "v1_contract.py": v1_root / "contract.py",
        "v1_worker.py": v1_root / "worker.py",
        "assumption_agent_init.py": (
            project_root / "assumption_agent" / "__init__.py"
        ),
        "assumption_agent_models.py": (
            project_root / "assumption_agent" / "models.py"
        ),
        "narrative_correspondence.py": (
            project_root
            / "assumption_agent"
            / "gscl_narrative_correspondence_v1.py"
        ),
        "intrinsic_arms_v1.py": (
            project_root
            / "assumption_agent"
            / "gscl_arn_intrinsic_arms_v1.py"
        ),
        "intrinsic_arms_v2.py": (
            project_root
            / "assumption_agent"
            / "gscl_arn_intrinsic_arms_v2.py"
        ),
        "unit_mapping_v2.py": (
            project_root
            / "assumption_agent"
            / "gscl_unit_mapping_v2.py"
        ),
    }
    rows: dict[str, str] = {}
    for logical_name, path in paths.items():
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise FixedPublicIntegratedQualificationError(
                "integrated_implementation_unreadable"
            ) from exc
        if not raw or len(raw) > MAXIMUM_IMPLEMENTATION_FILE_BYTES:
            raise FixedPublicIntegratedQualificationError(
                "integrated_implementation_size_invalid"
            )
        rows[logical_name] = hashlib.sha256(raw).hexdigest()
    if len(rows) != len(paths):
        raise FixedPublicIntegratedQualificationError(
            "integrated_implementation_name_collision"
        )
    return dict(sorted(rows.items()))


def _publish_once(path: Path, raw: bytes) -> None:
    if (
        not isinstance(path, Path)
        or not raw
        or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES
    ):
        raise FixedPublicIntegratedQualificationError(
            "integrated_publish_arguments_invalid"
        )
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.parent.lstat()
    except OSError as exc:
        raise FixedPublicIntegratedQualificationError(
            "integrated_output_root_invalid"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.parent.is_symlink():
        raise FixedPublicIntegratedQualificationError(
            "integrated_output_root_invalid"
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
        raise FixedPublicIntegratedQualificationError(
            "integrated_receipt_publish_failed"
        ) from exc


def _load_upstream_aggregate(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FixedPublicIntegratedQualificationError(
            "integrated_upstream_receipt_unreadable"
        ) from exc
    if not raw or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES:
        raise FixedPublicIntegratedQualificationError(
            "integrated_upstream_receipt_size_invalid"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise FixedPublicIntegratedQualificationError(
            "integrated_upstream_receipt_json_invalid"
        ) from exc
    if (
        type(value) is not dict
        or _canonical_bytes(value) != raw
        or value.get("schema")
        != extractor_qualification.AGGREGATE_RECEIPT_SCHEMA
    ):
        raise FixedPublicIntegratedQualificationError(
            "integrated_upstream_receipt_canonical_invalid"
        )
    supplied = _require_hex64(
        value.get("self_sha256"),
        "integrated_upstream_receipt_self_invalid",
    )
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    expected_implementation = (
        extractor_qualification._implementation_closure()
    )
    if (
        supplied != _safe_hash(body)
        or value.get("qualification_passed") is not True
        or value.get("repeat_byte_exact") is not True
        or value.get("repeat_count")
        != extractor_qualification.REPEAT_COUNT
        or value.get("fixture_suite_sha256")
        != extractor_qualification.FIXTURE_SUITE_SHA256
        or value.get("fixture_ordinals")
        != list(range(len(extractor_qualification.PUBLIC_FIXTURES)))
        or value.get("fixture_count")
        != len(extractor_qualification.PUBLIC_FIXTURES)
        or value.get("fixture_commitments")
        != dict(extractor_qualification.FIXTURE_COMMITMENTS)
        or value.get("outcome_counts")
        != {
            "success": len(extractor_qualification.PUBLIC_FIXTURES),
            "typed_abstention": 0,
            "typed_error": 0,
        }
        or value.get("counters") != _zero_counters()
        or value.get("implementation_closure")
        != expected_implementation
        or value.get("implementation_closure_sha256")
        != _safe_hash(expected_implementation)
    ):
        raise FixedPublicIntegratedQualificationError(
            "integrated_upstream_receipt_contract_invalid"
        )
    value["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _fixture_rows() -> tuple[
    extractor_qualification.PublicFixture, ...
]:
    fixtures = tuple(
        extractor_qualification.PUBLIC_FIXTURES[ordinal]
        for ordinal in PUBLIC_ITEM_FIXTURE_ORDINALS
    )
    if (
        tuple(row.ordinal for row in fixtures)
        != PUBLIC_ITEM_FIXTURE_ORDINALS
        or len({row.input_sha256 for row in fixtures}) != 3
        or any(
            _HEX64.fullmatch(row.fixture_commitment) is None
            for row in fixtures
        )
    ):
        raise FixedPublicIntegratedQualificationError(
            "integrated_public_item_fixture_invalid"
        )
    return fixtures


def _extract_once(
    *,
    runtime: object,
    runtime_commitment: str,
    fixture: extractor_qualification.PublicFixture,
) -> tuple[closed_choice.ClosedChoiceV2Decision, dict[str, object]]:
    selector = getattr(runtime, "select_story", None)
    if not callable(selector):
        raise FixedPublicIntegratedQualificationError(
            "integrated_runtime_select_surface_invalid"
        )
    try:
        decision = selector(fixture.story_text)
    except Exception as exc:
        raise FixedPublicIntegratedQualificationError(
            "integrated_fixed_fixture_extraction_failed"
        ) from exc
    if type(decision) is not closed_choice.ClosedChoiceV2Decision:
        raise FixedPublicIntegratedQualificationError(
            "integrated_decision_type_invalid"
        )
    safe = extractor_qualification._safe_success_outcome(
        fixture=fixture,
        decision=decision,
        runtime_commitment=runtime_commitment,
    )
    if (
        type(decision.extraction) is not NarrativeExtraction
        or decision.extraction.source.source_sha256
        != fixture.input_sha256
    ):
        raise FixedPublicIntegratedQualificationError(
            "integrated_extraction_type_invalid"
        )
    return decision, safe


def _extract_twice(
    *,
    runtime: object,
    runtime_commitment: str,
    fixture: extractor_qualification.PublicFixture,
) -> tuple[
    NarrativeExtraction,
    NarrativeExtraction,
    dict[str, object],
]:
    first, first_safe = _extract_once(
        runtime=runtime,
        runtime_commitment=runtime_commitment,
        fixture=fixture,
    )
    second, second_safe = _extract_once(
        runtime=runtime,
        runtime_commitment=runtime_commitment,
        fixture=fixture,
    )
    first_binding = {
        "canonical_completion": first.canonical_completion,
        "decision_receipt": json.loads(
            first.receipt_bytes.decode("ascii")
        ),
        "extraction": first.extraction.safe_payload(),
        "wire_completion": first.wire_completion,
    }
    second_binding = {
        "canonical_completion": second.canonical_completion,
        "decision_receipt": json.loads(
            second.receipt_bytes.decode("ascii")
        ),
        "extraction": second.extraction.safe_payload(),
        "wire_completion": second.wire_completion,
    }
    if (
        _canonical_bytes(first_binding)
        != _canonical_bytes(second_binding)
        or _canonical_bytes(first_safe)
        != _canonical_bytes(second_safe)
    ):
        raise FixedPublicIntegratedQualificationError(
            "integrated_extraction_repeat_mismatch"
        )
    return first.extraction, second.extraction, {
        "decision_receipt_sha256": first_safe[
            "decision_receipt_sha256"
        ],
        "extraction_provenance_hash": (
            first.extraction.provenance_hash
        ),
        "extraction_semantic_hash": (
            first.extraction.semantic_hash
        ),
        "fixture_commitment": fixture.fixture_commitment,
        "fixture_id": fixture.fixture_id,
        "generator_count": len(first.extraction.generators),
        "input_sha256": fixture.input_sha256,
        "mention_count": len(first.extraction.mentions),
        "ordinal": fixture.ordinal,
        "repeat_byte_exact": True,
        "repeat_count": REPEAT_COUNT,
        "repeat_outcome_sha256": hashlib.sha256(
            _canonical_bytes(first_binding)
        ).hexdigest(),
    }


def _lexical_counter(payload: bytes | str) -> Counter[str]:
    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="strict")
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("integrated_lexical_payload_invalid")
    return Counter(
        match.group(0).casefold()
        for match in _LEXICAL.finditer(text)
    )


def _fixed_raw_text_scorer(query: bytes, candidate: bytes) -> int:
    left = _lexical_counter(query)
    right = _lexical_counter(candidate)
    shared = sum((left & right).values())
    return 10_000 * shared - abs(sum(left.values()) - sum(right.values()))


def _fixed_legacy_vectorizer(
    extraction: NarrativeExtraction,
    feature_ids: tuple[str, ...],
) -> tuple[int, ...]:
    if feature_ids != LEGACY_FEATURE_IDS:
        raise FixedPublicIntegratedQualificationError(
            "integrated_legacy_registry_invalid"
        )
    generators = extraction.generators
    values = {
        "generator_count": len(generators),
        "object_count": len(
            extraction.hypergraph.object_mention_ids
        ),
        "relation_count": sum(
            generator.generator_kind
            is GeneratorKind.RELATION
            for generator in generators
        ),
        "state_change_count": sum(
            generator.generator_kind
            is GeneratorKind.STATE_CHANGE
            for generator in generators
        ),
        "temporal_count": sum(
            generator.generator_kind
            is GeneratorKind.TEMPORAL
            for generator in generators
        ),
        "causal_count": sum(
            generator.generator_kind
            is GeneratorKind.CAUSAL
            for generator in generators
        ),
        "positive_count": sum(
            generator.polarity.value == "positive"
            for generator in generators
        ),
        "neutral_count": sum(
            generator.polarity.value == "neutral"
            for generator in generators
        ),
        "negative_count": sum(
            generator.polarity.value == "negative"
            for generator in generators
        ),
        "temporal_forward_count": sum(
            generator.temporal_orientation.value == "forward"
            for generator in generators
        ),
        "causal_forward_count": sum(
            generator.causal_orientation.value == "forward"
            for generator in generators
        ),
    }
    return tuple(int(values[feature_id]) for feature_id in feature_ids)


def _mention_score(left: Mention, right: Mention) -> int:
    left_tokens = _lexical_counter(left.quote)
    right_tokens = _lexical_counter(right.quote)
    shared = sum((left_tokens & right_tokens).values())
    exact = int(left.quote.casefold() == right.quote.casefold())
    length_closeness = max(
        0, 100 - abs(len(left.quote) - len(right.quote))
    )
    return 10_000 * shared + 1_000 * exact + length_closeness


def _fixed_structural_scorer(
    query: NarrativeExtraction,
    candidate: NarrativeExtraction,
) -> SemanticScoreTable:
    query_mentions = {
        mention.mention_id: mention for mention in query.mentions
    }
    candidate_mentions = {
        mention.mention_id: mention for mention in candidate.mentions
    }
    object_scores = {
        (source_id, target_id): _mention_score(
            query_mentions[source_id],
            candidate_mentions[target_id],
        )
        for source_id in query.hypergraph.object_mention_ids
        for target_id in candidate.hypergraph.object_mention_ids
    }
    generator_scores: dict[tuple[str, str], int] = {}
    for source in query.generators:
        source_anchor = query_mentions[source.anchor_mention_id]
        for target in candidate.generators:
            target_anchor = candidate_mentions[
                target.anchor_mention_id
            ]
            generator_scores[
                (source.generator_id, target.generator_id)
            ] = (
                _mention_score(source_anchor, target_anchor)
                + 4_000
                * int(source.generator_kind is target.generator_kind)
                + 1_000 * int(source.polarity is target.polarity)
                + 500
                * int(
                    source.temporal_orientation
                    is target.temporal_orientation
                )
                + 500
                * int(
                    source.causal_orientation
                    is target.causal_orientation
                )
            )
    return SemanticScoreTable.from_mappings(
        object_scores=object_scores,
        generator_scores=generator_scores,
    )


RAW_TEXT_SCORER_COMMITMENT = _safe_hash(
    {
        "name": "fixed_public_multiset_lexical_overlap",
        "version": VERSION,
    }
)
LEGACY_VECTORIZER_COMMITMENT = _safe_hash(
    {
        "feature_ids": list(LEGACY_FEATURE_IDS),
        "name": "fixed_public_typed_count_vector",
        "version": VERSION,
    }
)
STRUCTURAL_SCORER_COMMITMENT = _safe_hash(
    {
        "name": "fixed_public_complete_mention_and_generator_table",
        "version": VERSION,
    }
)
OPAQUE_ITEM_ID = _safe_hash(
    {
        "fixture_ordinals": list(PUBLIC_ITEM_FIXTURE_ORDINALS),
        "fixture_suite_sha256": (
            extractor_qualification.FIXTURE_SUITE_SHA256
        ),
        "version": VERSION,
    }
)


def _evaluate_once(
    extractions: tuple[
        NarrativeExtraction,
        NarrativeExtraction,
        NarrativeExtraction,
    ],
) -> arms_v1.IntrinsicItemResult:
    return arms_v2.evaluate_intrinsic_item_v2(
        opaque_item_id=OPAQUE_ITEM_ID,
        query=extractions[0],
        candidates=(extractions[1], extractions[2]),
        raw_text_scorer=_fixed_raw_text_scorer,
        legacy_vectorizer=_fixed_legacy_vectorizer,
        legacy_feature_ids=LEGACY_FEATURE_IDS,
        structural_scorer=_fixed_structural_scorer,
        mapping_config=UnitMappingSearchConfigV2(),
        raw_text_scorer_commitment=RAW_TEXT_SCORER_COMMITMENT,
        legacy_vectorizer_commitment=(
            LEGACY_VECTORIZER_COMMITMENT
        ),
        structural_scorer_commitment=(
            STRUCTURAL_SCORER_COMMITMENT
        ),
    )


def _public_arm_summary(
    result: arms_v1.IntrinsicItemResult,
) -> dict[str, object]:
    if type(result) is not arms_v1.IntrinsicItemResult:
        raise FixedPublicIntegratedQualificationError(
            "integrated_arm_result_type_invalid"
        )
    return {
        "candidate_receipts": [
            receipt.safe_payload()
            for receipt in result.candidate_receipts
        ],
        "implementation_commitments": dict(
            result.implementation_commitments
        ),
        "input_commitment": (
            result.predictions[0].input_commitment
        ),
        "opaque_item_id": result.opaque_item_id,
        "predictions": [
            prediction.safe_payload()
            for prediction in result.predictions
        ],
        "result_hash": result.result_hash,
    }


def _validate_arm_summary(value: Mapping[str, object]) -> bool:
    predictions = value.get("predictions")
    receipts = value.get("candidate_receipts")
    if (
        type(predictions) is not list
        or [row.get("arm") for row in predictions]
        != [arm.value for arm in arms_v1.IntrinsicArm]
        or type(receipts) is not list
        or len(receipts) != 2
    ):
        return False
    for ordinal, receipt in enumerate(receipts):
        if (
            type(receipt) is not dict
            or receipt.get("candidate_ordinal") != ordinal
            or receipt.get("status") != "complete"
            or receipt.get("flat_proposal_set_hash") is None
            or receipt.get("flat_proposal_set_hash")
            != receipt.get("full_proposal_set_hash")
            or receipt.get("flat_choice_commitment") is None
            or receipt.get("full_choice_commitment") is None
        ):
            return False
    return True


def _resource_peaks() -> dict[str, int]:
    allocated = 0
    reserved = 0
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            allocated = int(torch.cuda.max_memory_allocated(0))
            reserved = int(torch.cuda.max_memory_reserved(0))
    except Exception as exc:
        raise FixedPublicIntegratedQualificationError(
            "integrated_cuda_peak_read_failed"
        ) from exc
    return {
        "cuda_max_memory_allocated_bytes": allocated,
        "cuda_max_memory_reserved_bytes": reserved,
        "process_max_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }


def run_fixed_public_integrated_qualification(
    *,
    model_root: Path,
    manifest: worker.ModelAssetManifest,
    upstream_aggregate_receipt: Path,
    output_root: Path,
) -> Mapping[str, object]:
    """Run the immutable public triple through extractor and four arms."""

    if (
        not isinstance(model_root, Path)
        or type(manifest) is not worker.ModelAssetManifest
        or not isinstance(upstream_aggregate_receipt, Path)
        or not isinstance(output_root, Path)
    ):
        raise FixedPublicIntegratedQualificationError(
            "integrated_arguments_invalid"
        )
    upstream = _load_upstream_aggregate(
        upstream_aggregate_receipt
    )
    extractor_qualification._verify_model_binding(
        model_root=model_root, manifest=manifest
    )
    expected_manifest = extractor_qualification._manifest_commitments(
        manifest
    )
    if upstream.get("manifest_commitments") != expected_manifest:
        raise FixedPublicIntegratedQualificationError(
            "integrated_upstream_model_binding_mismatch"
        )
    implementation = _implementation_closure()
    runtime = extractor_qualification._load_exact_runtime(
        model_root=model_root, manifest=manifest
    )
    try:
        runtime_commitment = _require_hex64(
            runtime.runtime_commitment,
            "integrated_runtime_commitment_invalid",
        )
    except FixedPublicIntegratedQualificationError:
        raise
    except Exception as exc:
        raise FixedPublicIntegratedQualificationError(
            "integrated_runtime_commitment_unavailable"
        ) from exc
    if upstream.get("runtime_commitment") != runtime_commitment:
        raise FixedPublicIntegratedQualificationError(
            "integrated_upstream_runtime_binding_mismatch"
        )
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
    except Exception as exc:
        raise FixedPublicIntegratedQualificationError(
            "integrated_cuda_peak_reset_failed"
        ) from exc

    first_extraction_rows: list[NarrativeExtraction] = []
    second_extraction_rows: list[NarrativeExtraction] = []
    extraction_summaries: list[dict[str, object]] = []
    for fixture in _fixture_rows():
        first_extraction, second_extraction, summary = _extract_twice(
            runtime=runtime,
            runtime_commitment=runtime_commitment,
            fixture=fixture,
        )
        first_extraction_rows.append(first_extraction)
        second_extraction_rows.append(second_extraction)
        extraction_summaries.append(summary)
    first_extractions = tuple(first_extraction_rows)
    second_extractions = tuple(second_extraction_rows)
    if len(first_extractions) != 3 or len(second_extractions) != 3:
        raise FixedPublicIntegratedQualificationError(
            "integrated_extraction_count_invalid"
        )
    first = _public_arm_summary(_evaluate_once(first_extractions))
    second = _public_arm_summary(_evaluate_once(second_extractions))
    arm_repeat_byte_exact = (
        _canonical_bytes(first) == _canonical_bytes(second)
    )
    arm_contract_passed = _validate_arm_summary(first)
    qualification_passed = bool(
        arm_repeat_byte_exact
        and arm_contract_passed
        and all(
            row["repeat_byte_exact"] is True
            for row in extraction_summaries
        )
    )
    body: dict[str, object] = {
        "arm_core_version": arms_v2.ARMS_CORE_VERSION,
        "arm_contract_passed": arm_contract_passed,
        "arm_name_mapping": {
            "flat": "flat_label_no_verifier",
            "full": "full_gscl",
            "legacy": "legacy_keyword",
            "semantic_only": "semantic_only",
        },
        "full_checker_candidate_count": (
            2 if arm_contract_passed else 0
        ),
        "arm_repeat_byte_exact": arm_repeat_byte_exact,
        "arm_repeat_count": REPEAT_COUNT,
        "arm_summary": first,
        "arm_summary_commitment": _safe_hash(first),
        "counters": _zero_counters(),
        "extractor_repeat_byte_exact": all(
            row["repeat_byte_exact"] is True
            for row in extraction_summaries
        ),
        "extractor_repeat_count": REPEAT_COUNT,
        "extractor_runtime_commitment": runtime_commitment,
        "fixture_ordinals": list(
            PUBLIC_ITEM_FIXTURE_ORDINALS
        ),
        "fixture_suite_sha256": (
            extractor_qualification.FIXTURE_SUITE_SHA256
        ),
        "implementation_closure": implementation,
        "implementation_closure_sha256": _safe_hash(
            implementation
        ),
        "input_extractions": extraction_summaries,
        "input_extractions_commitment": _safe_hash(
            extraction_summaries
        ),
        "manifest_commitments": expected_manifest,
        "mapping_config": UnitMappingSearchConfigV2().safe_payload(),
        "mapping_config_sha256": (
            UnitMappingSearchConfigV2().config_hash
        ),
        "qualification_passed": qualification_passed,
        "resource_peaks": _resource_peaks(),
        "schema": RECEIPT_SCHEMA,
        "upstream_aggregate": {
            "file_sha256": upstream["_file_sha256"],
            "implementation_closure_sha256": upstream[
                "implementation_closure_sha256"
            ],
            "self_sha256": upstream["self_sha256"],
        },
        "version": VERSION,
    }
    receipt = {
        **body,
        "self_sha256": _safe_hash(body),
    }
    _publish_once(
        output_root / OUTPUT_NAME,
        _canonical_bytes(receipt),
    )
    return MappingProxyType(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed public source-free GSCL v2 integrated "
            "qualification"
        )
    )
    parser.add_argument(
        "--model-root", required=True, type=Path
    )
    parser.add_argument(
        "--model-manifest", required=True, type=Path
    )
    parser.add_argument(
        "--upstream-aggregate-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-root", required=True, type=Path
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = worker.load_model_asset_manifest(
        manifest_path=arguments.model_manifest,
        model_root=arguments.model_root,
    )
    receipt = run_fixed_public_integrated_qualification(
        model_root=arguments.model_root,
        manifest=manifest,
        upstream_aggregate_receipt=(
            arguments.upstream_aggregate_receipt
        ),
        output_root=arguments.output_root,
    )
    print(receipt["self_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LEGACY_FEATURE_IDS",
    "OPAQUE_ITEM_ID",
    "OUTPUT_NAME",
    "PUBLIC_ITEM_FIXTURE_ORDINALS",
    "RECEIPT_SCHEMA",
    "VERSION",
    "FixedPublicIntegratedQualificationError",
    "main",
    "run_fixed_public_integrated_qualification",
]

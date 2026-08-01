"""Label-blind ARN input compatibility for the qualified GSCL v2 extractor.

This is the next stage of the same non-scoring qualification harness.  The
input is the exact predictor-only pack sealed by the consumed ARN v1 attempt;
its separately stored linkage and label packs are neither accepted nor read.
The diagnostic therefore measures representability/coverage only.  It is not
an untouched cohort and cannot establish efficacy.
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
from typing import Any, Mapping, Sequence

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    contract as v1_contract,
)
from replication_runtime.gscl_narrative_extractor_v1 import worker

from . import closed_choice
from . import contract
from . import fixed_public_integrated_qualification as integrated
from . import fixed_public_qualification as extractor_qualification


VERSION = "gscl_v2_fixed_arn_input_compatibility_v1"
SHARD_RECEIPT_SCHEMA = f"{VERSION}.shard.safe.v1"
AGGREGATE_RECEIPT_SCHEMA = f"{VERSION}.aggregate.safe.v1"
SHARD_OUTPUT_NAME = "compatibility.shard.safe.json"
AGGREGATE_OUTPUT_NAME = "compatibility.aggregate.safe.json"
SHARD_COUNT = 2
EXPECTED_ITEM_COUNT = 871
EXPECTED_STORY_COUNT = 3 * EXPECTED_ITEM_COUNT
EXPECTED_PREDICTOR_FILE_SHA256 = (
    "6c1d9f7397da246298ac12d86803b7f1b41ad0829c984ec7c3124716ba51154b"
)
EXPECTED_PREDICTOR_SCHEMA = (
    "gscl_arn_intrinsic_protocol_v1.official_predictor_pack.v2"
)
EXPECTED_LINEAGE = "official_arn_measurement"
EXPECTED_SOURCE_SHA256 = (
    "a866fe5341ce4a29f00f24987a12278303b2b8ad788352f549b0fe051ad4a7a8"
)
EXPECTED_SOURCE_VERIFICATION_SELF_SHA256 = (
    "099885d3981fbbf388601a551eca5c315c99863fb7547548f075f0c44ed877e2"
)
EXPECTED_ADAPTER_QUALIFICATION_SELF_SHA256 = (
    "b92841f06d91f84753221a0d94985396438fa0092a890d6c06c53b616c20ac95"
)
PREDICTOR_COLUMNS = (
    "query_narrative",
    "first_choice",
    "second_choice",
)
MAXIMUM_PREDICTOR_BYTES = 4 * 1024 * 1024
MAXIMUM_SAFE_RECEIPT_BYTES = 2 * 1024 * 1024
MAXIMUM_IMPLEMENTATION_FILE_BYTES = 4 * 1024 * 1024
PROGRESS_INTERVAL_ROWS = 25
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class FixedArnInputCompatibilityError(RuntimeError):
    """Stable non-content compatibility error."""


def _canonical_bytes(value: object) -> bytes:
    return v1_contract.canonical_json_bytes(value)


def _safe_hash(value: object) -> str:
    return v1_contract.semantic_sha256(value)


def _require_hex64(value: object, issue: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FixedArnInputCompatibilityError(issue)
    return value


def _access_counters(*, aggregate: bool) -> dict[str, int]:
    predictor_access = SHARD_COUNT if aggregate else 1
    return {
        "api_access_count": 0,
        "free_form_generation_count": 0,
        "label_access_count": 0,
        "network_access_count": 0,
        "online_evaluator_access_count": 0,
        "predictor_pack_access_count": predictor_access,
        "raw_source_access_count": 0,
        "scorer_access_count": 0,
        "source_access_count": predictor_access,
    }


def _publish_once(path: Path, raw: bytes) -> None:
    if (
        not isinstance(path, Path)
        or not raw
        or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_publish_arguments_invalid"
        )
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.parent.lstat()
    except OSError as exc:
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_output_root_invalid"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.parent.is_symlink():
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_output_root_invalid"
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
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_receipt_publish_failed"
        ) from exc


def _claim_attempt(
    *,
    output_root: Path,
    shard_index: int,
    upstream: Mapping[str, object],
) -> Mapping[str, object]:
    """Create the pre-source O_EXCL barrier for this exact shard attempt."""

    body: dict[str, object] = {
        "integrated_qualification_file_sha256": upstream[
            "_file_sha256"
        ],
        "predictor_file_sha256": EXPECTED_PREDICTOR_FILE_SHA256,
        "schema": f"{VERSION}.attempt_start.safe.v1",
        "shard_count": SHARD_COUNT,
        "shard_index": shard_index,
        "version": VERSION,
    }
    receipt = {
        **body,
        "self_sha256": _safe_hash(body),
    }
    _publish_once(
        output_root / "attempt.started.safe.json",
        _canonical_bytes(receipt),
    )
    return MappingProxyType(receipt)


def _implementation_closure() -> dict[str, str]:
    project_root = Path(__file__).parents[2]
    package_root = Path(__file__).parent
    v1_root = project_root / "replication_runtime" / (
        "gscl_narrative_extractor_v1"
    )
    manifest_root = project_root / "manifests"
    paths = {
        "arn_input_compatibility.py": Path(__file__),
        "arn_input_compatibility_shard0.service": (
            manifest_root
            / "gscl_narrative_extractor_v2_fixed_arn_compatibility_shard0.service"
        ),
        "arn_input_compatibility_shard1.service": (
            manifest_root
            / "gscl_narrative_extractor_v2_fixed_arn_compatibility_shard1.service"
        ),
        "arn_input_compatibility_aggregate.service": (
            manifest_root
            / "gscl_narrative_extractor_v2_fixed_arn_compatibility_aggregate.service"
        ),
        "v2_closed_choice.py": package_root / "closed_choice.py",
        "v2_contract.py": package_root / "contract.py",
        "v2_memory_safe_qwen.py": package_root / "memory_safe_qwen.py",
        "v2_fixed_public_qualification.py": (
            package_root / "fixed_public_qualification.py"
        ),
        "v2_integrated_qualification.py": (
            package_root
            / "fixed_public_integrated_qualification.py"
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
            raise FixedArnInputCompatibilityError(
                "arn_compatibility_implementation_unreadable"
            ) from exc
        if not raw or len(raw) > MAXIMUM_IMPLEMENTATION_FILE_BYTES:
            raise FixedArnInputCompatibilityError(
                "arn_compatibility_implementation_size_invalid"
            )
        rows[logical_name] = hashlib.sha256(raw).hexdigest()
    if len(rows) != len(paths):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_implementation_name_collision"
        )
    return dict(sorted(rows.items()))


def _load_json_bytes(
    path: Path, *, maximum: int, issue_prefix: str
) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FixedArnInputCompatibilityError(
            f"{issue_prefix}_unreadable"
        ) from exc
    if not raw or len(raw) > maximum:
        raise FixedArnInputCompatibilityError(
            f"{issue_prefix}_size_invalid"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise FixedArnInputCompatibilityError(
            f"{issue_prefix}_json_invalid"
        ) from exc
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise FixedArnInputCompatibilityError(
            f"{issue_prefix}_canonical_invalid"
        )
    return raw, value


def _load_integrated_receipt(path: Path) -> dict[str, Any]:
    raw, value = _load_json_bytes(
        path,
        maximum=MAXIMUM_SAFE_RECEIPT_BYTES,
        issue_prefix="arn_compatibility_upstream",
    )
    supplied = _require_hex64(
        value.get("self_sha256"),
        "arn_compatibility_upstream_self_invalid",
    )
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    if (
        value.get("schema") != integrated.RECEIPT_SCHEMA
        or supplied != _safe_hash(body)
        or value.get("qualification_passed") is not True
        or value.get("arm_contract_passed") is not True
        or value.get("arm_repeat_byte_exact") is not True
        or value.get("extractor_repeat_byte_exact") is not True
        or value.get("counters") != integrated._zero_counters()
        or value.get("implementation_closure")
        != integrated._implementation_closure()
        or value.get("implementation_closure_sha256")
        != _safe_hash(integrated._implementation_closure())
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_upstream_contract_invalid"
        )
    value["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _load_predictor_pack(
    path: Path,
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    raw, value = _load_json_bytes(
        path,
        maximum=MAXIMUM_PREDICTOR_BYTES,
        issue_prefix="arn_compatibility_predictor",
    )
    if (
        hashlib.sha256(raw).hexdigest()
        != EXPECTED_PREDICTOR_FILE_SHA256
        or set(value)
        != {
            "adapter_qualification_self_hash",
            "column_contract",
            "lineage",
            "rows",
            "schema",
            "source_sha256",
            "source_verification_self_hash",
        }
        or value.get("schema") != EXPECTED_PREDICTOR_SCHEMA
        or value.get("lineage") != EXPECTED_LINEAGE
        or value.get("source_sha256") != EXPECTED_SOURCE_SHA256
        or value.get("source_verification_self_hash")
        != EXPECTED_SOURCE_VERIFICATION_SELF_SHA256
        or value.get("adapter_qualification_self_hash")
        != EXPECTED_ADAPTER_QUALIFICATION_SELF_SHA256
        or value.get("column_contract")
        != list(PREDICTOR_COLUMNS)
        or type(value.get("rows")) is not list
        or len(value["rows"]) != EXPECTED_ITEM_COUNT
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_predictor_contract_invalid"
        )
    normalized: list[dict[str, str]] = []
    observed_ids: set[str] = set()
    for row in value["rows"]:
        if (
            type(row) is not dict
            or set(row)
            != {
                "first_choice",
                "opaque_item_id",
                "query_narrative",
                "second_choice",
            }
            or _HEX64.fullmatch(
                str(row.get("opaque_item_id"))
            )
            is None
            or row["opaque_item_id"] in observed_ids
            or any(
                not isinstance(row.get(field), str)
                or not row[field]
                or "\x00" in row[field]
                for field in PREDICTOR_COLUMNS
            )
        ):
            raise FixedArnInputCompatibilityError(
                "arn_compatibility_predictor_row_invalid"
            )
        observed_ids.add(row["opaque_item_id"])
        normalized.append(
            {
                "opaque_item_id": row["opaque_item_id"],
                **{
                    field: row[field]
                    for field in PREDICTOR_COLUMNS
                },
            }
        )
    return (
        {
            "adapter_qualification_self_hash": value[
                "adapter_qualification_self_hash"
            ],
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "lineage": value["lineage"],
            "schema": value["schema"],
            "source_sha256": value["source_sha256"],
            "source_verification_self_hash": value[
                "source_verification_self_hash"
            ],
        },
        tuple(normalized),
    )


def _validate_success(
    *,
    story: str,
    decision: object,
    runtime_commitment: str,
) -> Mapping[str, int]:
    if type(decision) is not closed_choice.ClosedChoiceV2Decision:
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_decision_type_invalid"
        )
    if (
        type(decision.extraction) is not NarrativeExtraction
        or decision.extraction.source.source_sha256
        != hashlib.sha256(story.encode("utf-8")).hexdigest()
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_extraction_binding_invalid"
        )
    receipt = dict(decision.receipt)
    supplied = _require_hex64(
        receipt.get("self_sha256"),
        "arn_compatibility_decision_self_invalid",
    )
    body = {
        key: child
        for key, child in receipt.items()
        if key != "self_sha256"
    }
    summary = receipt.get("resource_summary")
    generators = decision.extraction.generators
    object_ids = decision.extraction.hypergraph.object_mention_ids
    slot_ids = tuple(
        slot
        for generator in generators
        for slot in generator.slot_mention_ids
    )
    if (
        supplied != _safe_hash(body)
        or receipt.get("schema") != closed_choice.RECEIPT_SCHEMA
        or receipt.get("model_runtime_commitment")
        != runtime_commitment
        or receipt.get("story_commitment")
        != decision.extraction.source.source_sha256
        or receipt.get("free_form_generation_count") != 0
        or receipt.get("exclusive_endpoint_ownership") is not True
        or type(summary) is not dict
        or summary.get("relation_count") != len(generators)
        or len(object_ids) != 2 * len(generators)
        or len(slot_ids) != len(object_ids)
        or len(set(slot_ids)) != len(slot_ids)
        or set(slot_ids) != set(object_ids)
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_decision_contract_invalid"
        )
    expected_fields = {
        "candidate_count",
        "episode_count",
        "forward_batch_count",
        "maximum_candidates_in_one_batch",
        "maximum_span_lexical_width",
        "relation_count",
        "sentence_count",
    }
    if (
        set(summary) != expected_fields
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in summary.values()
        )
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_resource_summary_invalid"
        )
    return MappingProxyType(
        {key: int(summary[key]) for key in expected_fields}
    )


def _empty_resource_peaks() -> dict[str, int]:
    return {
        "max_candidate_count": 0,
        "max_episode_count": 0,
        "max_forward_batch_count": 0,
        "max_relation_count": 0,
        "max_sentence_count": 0,
        "max_span_lexical_width": 0,
    }


def _merge_resource_summary(
    peaks: dict[str, int], summary: Mapping[str, int]
) -> None:
    mapping = {
        "max_candidate_count": "candidate_count",
        "max_episode_count": "episode_count",
        "max_forward_batch_count": "forward_batch_count",
        "max_relation_count": "relation_count",
        "max_sentence_count": "sentence_count",
        "max_span_lexical_width": (
            "maximum_span_lexical_width"
        ),
    }
    for target, source in mapping.items():
        peaks[target] = max(peaks[target], int(summary[source]))


def _cuda_and_rss_peaks() -> dict[str, int]:
    allocated = 0
    reserved = 0
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            allocated = int(torch.cuda.max_memory_allocated(0))
            reserved = int(torch.cuda.max_memory_reserved(0))
    except Exception as exc:
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_cuda_peak_read_failed"
        ) from exc
    return {
        "cuda_max_memory_allocated_bytes": allocated,
        "cuda_max_memory_reserved_bytes": reserved,
        "process_max_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }


def _failure_record(
    exc: Exception,
) -> tuple[str, str, str]:
    if isinstance(exc, contract.ClosedChoiceV2Abstention):
        return (
            "typed_abstention",
            exc.category.value,
            exc.issue_id,
        )
    if isinstance(exc, contract.ClosedChoiceV2Error):
        return ("typed_error", exc.category.value, exc.issue_id)
    return (
        "untyped_error",
        "untyped_runtime",
        "UNTYPED_RUNTIME_ERROR",
    )


def run_fixed_arn_input_compatibility_shard(
    *,
    model_root: Path,
    manifest: worker.ModelAssetManifest,
    integrated_qualification_receipt: Path,
    predictor_pack: Path,
    output_root: Path,
    shard_index: int,
    shard_count: int,
) -> Mapping[str, object]:
    """Run one fixed ordinal-mod-two label-blind compatibility shard."""

    if (
        type(manifest) is not worker.ModelAssetManifest
        or not all(
            isinstance(path, Path)
            for path in (
                model_root,
                integrated_qualification_receipt,
                predictor_pack,
                output_root,
            )
        )
        or isinstance(shard_index, bool)
        or shard_index not in {0, 1}
        or isinstance(shard_count, bool)
        or shard_count != SHARD_COUNT
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_shard_arguments_invalid"
        )
    upstream = _load_integrated_receipt(
        integrated_qualification_receipt
    )
    attempt_start = _claim_attempt(
        output_root=output_root,
        shard_index=shard_index,
        upstream=upstream,
    )
    predictor_binding, rows = _load_predictor_pack(
        predictor_pack
    )
    extractor_qualification._verify_model_binding(
        model_root=model_root, manifest=manifest
    )
    manifest_binding = (
        extractor_qualification._manifest_commitments(manifest)
    )
    if upstream.get("manifest_commitments") != manifest_binding:
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_upstream_model_mismatch"
        )
    implementation = _implementation_closure()
    runtime = extractor_qualification._load_exact_runtime(
        model_root=model_root, manifest=manifest
    )
    try:
        runtime_commitment = _require_hex64(
            runtime.runtime_commitment,
            "arn_compatibility_runtime_commitment_invalid",
        )
    except FixedArnInputCompatibilityError:
        raise
    except Exception as exc:
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_runtime_commitment_unavailable"
        ) from exc
    if upstream.get("extractor_runtime_commitment") != (
        runtime_commitment
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_upstream_runtime_mismatch"
        )
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
    except Exception as exc:
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_cuda_peak_reset_failed"
        ) from exc

    selected_rows = tuple(
        row
        for ordinal, row in enumerate(rows)
        if ordinal % SHARD_COUNT == shard_index
    )
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    error_codes: Counter[str] = Counter()
    resources = _empty_resource_peaks()
    selector = getattr(runtime, "select_story", None)
    if not callable(selector):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_runtime_select_surface_invalid"
        )
    processed_stories = 0
    for local_row_index, row in enumerate(selected_rows, start=1):
        for field in PREDICTOR_COLUMNS:
            try:
                decision = selector(row[field])
                summary = _validate_success(
                    story=row[field],
                    decision=decision,
                    runtime_commitment=runtime_commitment,
                )
            except Exception as exc:
                disposition, category, code = _failure_record(exc)
                counts[disposition] += 1
                categories[category] += 1
                error_codes[code] += 1
            else:
                counts["success"] += 1
                _merge_resource_summary(resources, summary)
            processed_stories += 1
        if (
            local_row_index % PROGRESS_INTERVAL_ROWS == 0
            or local_row_index == len(selected_rows)
        ):
            print(
                json.dumps(
                    {
                        "processed_rows": local_row_index,
                        "processed_stories": processed_stories,
                        "shard_index": shard_index,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
    expected_row_count = sum(
        ordinal % SHARD_COUNT == shard_index
        for ordinal in range(EXPECTED_ITEM_COUNT)
    )
    expected_story_count = 3 * expected_row_count
    outcome_counts = {
        key: int(counts[key])
        for key in (
            "success",
            "typed_abstention",
            "typed_error",
            "untyped_error",
        )
    }
    body: dict[str, object] = {
        "access_counters": _access_counters(aggregate=False),
        "attempt_start_self_sha256": attempt_start[
            "self_sha256"
        ],
        "error_category_counts": dict(sorted(categories.items())),
        "error_code_counts": dict(sorted(error_codes.items())),
        "implementation_closure": implementation,
        "implementation_closure_sha256": _safe_hash(
            implementation
        ),
        "integrated_qualification": {
            "file_sha256": upstream["_file_sha256"],
            "self_sha256": upstream["self_sha256"],
        },
        "manifest_commitments": manifest_binding,
        "outcome_counts": outcome_counts,
        "predictor_binding": predictor_binding,
        "qualification_passed": (
            len(selected_rows) == expected_row_count
            and processed_stories == expected_story_count
            and outcome_counts["success"]
            == expected_story_count
            and sum(
                outcome_counts[key]
                for key in (
                    "typed_abstention",
                    "typed_error",
                    "untyped_error",
                )
            )
            == 0
        ),
        "resource_peaks": {
            **resources,
            **_cuda_and_rss_peaks(),
        },
        "runtime_commitment": runtime_commitment,
        "schema": SHARD_RECEIPT_SCHEMA,
        "shard_count": SHARD_COUNT,
        "shard_index": shard_index,
        "shard_row_count": len(selected_rows),
        "shard_story_count": processed_stories,
        "version": VERSION,
    }
    receipt = {
        **body,
        "self_sha256": _safe_hash(body),
    }
    _publish_once(
        output_root / SHARD_OUTPUT_NAME,
        _canonical_bytes(receipt),
    )
    return MappingProxyType(receipt)


def _load_shard_receipt(path: Path) -> dict[str, Any]:
    raw, value = _load_json_bytes(
        path,
        maximum=MAXIMUM_SAFE_RECEIPT_BYTES,
        issue_prefix="arn_compatibility_shard_receipt",
    )
    supplied = _require_hex64(
        value.get("self_sha256"),
        "arn_compatibility_shard_receipt_self_invalid",
    )
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    expected_fields = {
        "access_counters",
        "attempt_start_self_sha256",
        "error_category_counts",
        "error_code_counts",
        "implementation_closure",
        "implementation_closure_sha256",
        "integrated_qualification",
        "manifest_commitments",
        "outcome_counts",
        "predictor_binding",
        "qualification_passed",
        "resource_peaks",
        "runtime_commitment",
        "schema",
        "self_sha256",
        "shard_count",
        "shard_index",
        "shard_row_count",
        "shard_story_count",
        "version",
    }
    index = value.get("shard_index")
    expected_row_count = (
        sum(
            ordinal % SHARD_COUNT == index
            for ordinal in range(EXPECTED_ITEM_COUNT)
        )
        if index in {0, 1}
        else -1
    )
    expected_story_count = 3 * expected_row_count
    count_keys = {
        "success",
        "typed_abstention",
        "typed_error",
        "untyped_error",
    }
    outcome_counts = value.get("outcome_counts")
    category_counts = value.get("error_category_counts")
    code_counts = value.get("error_code_counts")
    failure_count = (
        sum(
            int(outcome_counts[key])
            for key in count_keys - {"success"}
        )
        if type(outcome_counts) is dict
        and set(outcome_counts) == count_keys
        and all(
            isinstance(child, int)
            and not isinstance(child, bool)
            and child >= 0
            for child in outcome_counts.values()
        )
        else -1
    )
    expected_predictor_binding = {
        "adapter_qualification_self_hash": (
            EXPECTED_ADAPTER_QUALIFICATION_SELF_SHA256
        ),
        "file_sha256": EXPECTED_PREDICTOR_FILE_SHA256,
        "lineage": EXPECTED_LINEAGE,
        "schema": EXPECTED_PREDICTOR_SCHEMA,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_verification_self_hash": (
            EXPECTED_SOURCE_VERIFICATION_SELF_SHA256
        ),
    }
    expected_resource_fields = {
        "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
        "max_candidate_count",
        "max_episode_count",
        "max_forward_batch_count",
        "max_relation_count",
        "max_sentence_count",
        "max_span_lexical_width",
        "process_max_rss_kib",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != SHARD_RECEIPT_SCHEMA
        or supplied != _safe_hash(body)
        or value.get("shard_count") != SHARD_COUNT
        or index not in {0, 1}
        or value.get("access_counters")
        != _access_counters(aggregate=False)
        or value.get("shard_row_count") != expected_row_count
        or value.get("shard_story_count")
        != expected_story_count
        or type(outcome_counts) is not dict
        or set(outcome_counts) != count_keys
        or any(
            not isinstance(child, int)
            or isinstance(child, bool)
            or child < 0
            for child in outcome_counts.values()
        )
        or sum(outcome_counts.values()) != expected_story_count
        or failure_count < 0
        or type(category_counts) is not dict
        or type(code_counts) is not dict
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(child, int)
            or isinstance(child, bool)
            or child <= 0
            for counts in (category_counts, code_counts)
            for key, child in counts.items()
        )
        or sum(category_counts.values()) != failure_count
        or sum(code_counts.values()) != failure_count
        or value.get("qualification_passed")
        is not (
            outcome_counts["success"] == expected_story_count
            and failure_count == 0
        )
        or value.get("predictor_binding")
        != expected_predictor_binding
        or type(value.get("resource_peaks")) is not dict
        or set(value["resource_peaks"])
        != expected_resource_fields
        or any(
            not isinstance(child, int)
            or isinstance(child, bool)
            or child < 0
            for child in value["resource_peaks"].values()
        )
        or _HEX64.fullmatch(
            str(value.get("attempt_start_self_sha256"))
        )
        is None
        or _HEX64.fullmatch(
            str(value.get("runtime_commitment"))
        )
        is None
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_shard_receipt_contract_invalid"
        )
    value["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def aggregate_fixed_arn_input_compatibility(
    *,
    shard_receipts: tuple[Path, Path],
    output_root: Path,
) -> Mapping[str, object]:
    """Verify and aggregate both completed compatibility shards offline."""

    if (
        not isinstance(shard_receipts, tuple)
        or len(shard_receipts) != SHARD_COUNT
        or any(not isinstance(path, Path) for path in shard_receipts)
        or not isinstance(output_root, Path)
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_aggregate_arguments_invalid"
        )
    rows = sorted(
        (_load_shard_receipt(path) for path in shard_receipts),
        key=lambda value: int(value["shard_index"]),
    )
    if [row["shard_index"] for row in rows] != [0, 1]:
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_shard_union_invalid"
        )
    consistency_fields = (
        "implementation_closure",
        "implementation_closure_sha256",
        "integrated_qualification",
        "manifest_commitments",
        "predictor_binding",
        "runtime_commitment",
        "version",
    )
    if any(
        rows[0][field] != rows[1][field]
        for field in consistency_fields
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_shard_binding_mismatch"
        )
    current_implementation = _implementation_closure()
    if (
        rows[0]["implementation_closure"]
        != current_implementation
        or rows[0]["implementation_closure_sha256"]
        != _safe_hash(current_implementation)
    ):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_implementation_drifted"
        )
    count_keys = (
        "success",
        "typed_abstention",
        "typed_error",
        "untyped_error",
    )
    counts = {
        key: sum(
            int(row["outcome_counts"][key]) for row in rows
        )
        for key in count_keys
    }
    categories: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    for row in rows:
        categories.update(row["error_category_counts"])
        codes.update(row["error_code_counts"])
    resource_keys = set(rows[0]["resource_peaks"])
    if resource_keys != set(rows[1]["resource_peaks"]):
        raise FixedArnInputCompatibilityError(
            "arn_compatibility_resource_fields_mismatch"
        )
    peaks = {
        key: max(
            int(row["resource_peaks"][key]) for row in rows
        )
        for key in sorted(resource_keys)
    }
    total_rows = sum(int(row["shard_row_count"]) for row in rows)
    total_stories = sum(
        int(row["shard_story_count"]) for row in rows
    )
    body: dict[str, object] = {
        "access_counters": _access_counters(aggregate=True),
        "error_category_counts": dict(sorted(categories.items())),
        "error_code_counts": dict(sorted(codes.items())),
        "implementation_closure": current_implementation,
        "implementation_closure_sha256": _safe_hash(
            current_implementation
        ),
        "integrated_qualification": rows[0][
            "integrated_qualification"
        ],
        "manifest_commitments": rows[0]["manifest_commitments"],
        "outcome_counts": counts,
        "predictor_binding": rows[0]["predictor_binding"],
        "qualification_passed": (
            total_rows == EXPECTED_ITEM_COUNT
            and total_stories == EXPECTED_STORY_COUNT
            and counts["success"] == EXPECTED_STORY_COUNT
            and sum(counts[key] for key in count_keys[1:]) == 0
            and all(
                row["qualification_passed"] is True for row in rows
            )
        ),
        "resource_peaks": peaks,
        "runtime_commitment": rows[0]["runtime_commitment"],
        "schema": AGGREGATE_RECEIPT_SCHEMA,
        "shard_count": SHARD_COUNT,
        "shard_receipts": {
            str(row["shard_index"]): {
                "file_sha256": row["_file_sha256"],
                "self_sha256": row["self_sha256"],
            }
            for row in rows
        },
        "total_row_count": total_rows,
        "total_story_count": total_stories,
        "version": VERSION,
    }
    receipt = {
        **body,
        "self_sha256": _safe_hash(body),
    }
    _publish_once(
        output_root / AGGREGATE_OUTPUT_NAME,
        _canonical_bytes(receipt),
    )
    return MappingProxyType(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or aggregate fixed label-blind ARN input "
            "compatibility"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--model-root", required=True, type=Path)
    shard.add_argument(
        "--model-manifest", required=True, type=Path
    )
    shard.add_argument(
        "--integrated-qualification-receipt",
        required=True,
        type=Path,
    )
    shard.add_argument(
        "--predictor-pack", required=True, type=Path
    )
    shard.add_argument("--output-root", required=True, type=Path)
    shard.add_argument(
        "--shard-index", required=True, type=int, choices=(0, 1)
    )
    shard.add_argument(
        "--shard-count", required=True, type=int, choices=(2,)
    )
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument(
        "--shard-0-receipt", required=True, type=Path
    )
    aggregate.add_argument(
        "--shard-1-receipt", required=True, type=Path
    )
    aggregate.add_argument(
        "--output-root", required=True, type=Path
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "shard":
        manifest = worker.load_model_asset_manifest(
            manifest_path=arguments.model_manifest,
            model_root=arguments.model_root,
        )
        receipt = run_fixed_arn_input_compatibility_shard(
            model_root=arguments.model_root,
            manifest=manifest,
            integrated_qualification_receipt=(
                arguments.integrated_qualification_receipt
            ),
            predictor_pack=arguments.predictor_pack,
            output_root=arguments.output_root,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    else:
        receipt = aggregate_fixed_arn_input_compatibility(
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
    "EXPECTED_ITEM_COUNT",
    "EXPECTED_PREDICTOR_FILE_SHA256",
    "EXPECTED_STORY_COUNT",
    "FixedArnInputCompatibilityError",
    "SHARD_COUNT",
    "SHARD_OUTPUT_NAME",
    "SHARD_RECEIPT_SCHEMA",
    "VERSION",
    "aggregate_fixed_arn_input_compatibility",
    "main",
    "run_fixed_arn_input_compatibility_shard",
]

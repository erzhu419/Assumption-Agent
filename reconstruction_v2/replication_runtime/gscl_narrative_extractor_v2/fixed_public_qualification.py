"""Fixed, public, source-free Qwen/CUDA qualification for extractor v2.

This runner is deliberately outside every formal study.  It accepts only a
verified local model asset, a fixed two-way shard coordinate, and an output
directory.  Stories and teacher-forced pairs are program-owned constants;
there is no prompt, story, source, label, scorer, backend, network, or API
injection surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from . import closed_choice
from . import contract
from . import memory_safe_qwen


VERSION = "gscl_narrative_extractor_v2_fixed_public_qualification_v1"
SHARD_RECEIPT_SCHEMA = f"{VERSION}.shard_receipt.v1"
AGGREGATE_RECEIPT_SCHEMA = f"{VERSION}.aggregate_receipt.v1"
SHARD_COUNT = 2
REPEAT_COUNT = 2
SHARD_OUTPUT_NAME = "qualification.safe.json"
AGGREGATE_OUTPUT_NAME = "qualification.aggregate.safe.json"
MAXIMUM_SAFE_RECEIPT_BYTES = 2 * 1024 * 1024
MAXIMUM_IMPLEMENTATION_FILE_BYTES = 4 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class FixedPublicQualificationError(RuntimeError):
    """A stable, content-free qualification error."""


@dataclass(frozen=True, slots=True)
class PublicFixture:
    fixture_id: str
    ordinal: int
    story_text: str
    lexical_token_count: int
    sentence_count: int
    feature_flags: tuple[str, ...]

    @property
    def input_sha256(self) -> str:
        return hashlib.sha256(
            self.story_text.encode("utf-8")
        ).hexdigest()

    @property
    def fixture_commitment(self) -> str:
        return v1_contract.semantic_sha256(
            {
                "feature_flags": list(self.feature_flags),
                "fixture_id": self.fixture_id,
                "input_sha256": self.input_sha256,
                "lexical_token_count": self.lexical_token_count,
                "ordinal": self.ordinal,
                "sentence_count": self.sentence_count,
            }
        )


def _build_public_story(
    *,
    fixture_tag: str,
    sentence_sizes: tuple[int, ...],
    replacements: Mapping[int, str],
    terminal_punctuation: bool,
) -> str:
    total = sum(sentence_sizes)
    tokens = [
        replacements.get(
            index, f"{fixture_tag}Token{index:03d}"
        )
        for index in range(total)
    ]
    rows: list[str] = []
    offset = 0
    punctuation = (".", "?", "!", ".")
    for sentence_index, size in enumerate(sentence_sizes):
        row = " ".join(tokens[offset : offset + size])
        offset += size
        if (
            terminal_punctuation
            or sentence_index < len(sentence_sizes) - 1
        ):
            row += punctuation[
                sentence_index % len(punctuation)
            ]
        rows.append(row)
    return "\n".join(rows)


PUBLIC_FIXTURES = (
    PublicFixture(
        fixture_id="public_017_repeat_multiword",
        ordinal=0,
        story_text=_build_public_story(
            fixture_tag="F017",
            sentence_sizes=(17,),
            replacements={
                0: "Aster",
                1: "guides",
                2: "New",
                3: "York",
                4: "City",
                5: "while",
                6: "Aster",
                7: "supports",
                8: "Birch",
                9: "and",
                10: "Birch",
                11: "follows",
                12: "Cedar",
                13: "before",
                14: "Cedar",
                15: "helps",
                16: "Dune",
            },
            terminal_punctuation=True,
        ),
        lexical_token_count=17,
        sentence_count=1,
        feature_flags=("multiword", "repeated_phrase"),
    ),
    PublicFixture(
        fixture_id="public_033_unicode_two_sentence",
        ordinal=1,
        story_text=_build_public_story(
            fixture_tag="F033",
            sentence_sizes=(16, 17),
            replacements={
                0: "Élodie",
                1: "observes",
                2: "São",
                3: "Paulo",
                4: "while",
                5: "甲方",
                6: "guides",
                7: "乙方",
                16: "η",
                17: "links",
                18: "Theta",
                19: "and",
                20: "Theta",
                21: "supports",
                22: "Iota",
            },
            terminal_punctuation=True,
        ),
        lexical_token_count=33,
        sentence_count=2,
        feature_flags=("multiword", "repeated_phrase", "unicode"),
    ),
    PublicFixture(
        fixture_id="public_064_five_sentence",
        ordinal=2,
        story_text=_build_public_story(
            fixture_tag="F064",
            sentence_sizes=(12, 13, 13, 13, 13),
            replacements={
                0: "River",
                1: "crosses",
                2: "Old",
                3: "Stone",
                4: "Bridge",
                12: "River",
                13: "supports",
                14: "Harbor",
                25: "Harbor",
                26: "precedes",
                27: "Market",
            },
            terminal_punctuation=False,
        ),
        lexical_token_count=64,
        sentence_count=5,
        feature_flags=(
            "multiword",
            "no_terminal_punctuation",
            "repeated_phrase",
        ),
    ),
    PublicFixture(
        fixture_id="public_128_long_two_sentence",
        ordinal=3,
        story_text=_build_public_story(
            fixture_tag="F128",
            sentence_sizes=(64, 64),
            replacements={
                0: "North",
                1: "Atlantic",
                2: "Current",
                3: "influences",
                4: "Coast",
                64: "Coast",
                65: "responds",
                66: "to",
                67: "North",
                68: "Atlantic",
                69: "Current",
            },
            terminal_punctuation=True,
        ),
        lexical_token_count=128,
        sentence_count=2,
        feature_flags=("long_sentence", "multiword", "repeated_phrase"),
    ),
    PublicFixture(
        fixture_id="public_175_maximum_five_sentence_unicode",
        ordinal=4,
        story_text=_build_public_story(
            fixture_tag="F175",
            sentence_sizes=(35, 35, 35, 35, 35),
            replacements={
                0: "München",
                1: "connects",
                2: "Central",
                3: "Rail",
                4: "Station",
                35: "Central",
                36: "Rail",
                37: "Station",
                38: "serves",
                39: "Zürich",
                70: "Zürich",
                71: "coordinates",
                72: "with",
                73: "東京",
            },
            terminal_punctuation=True,
        ),
        lexical_token_count=175,
        sentence_count=5,
        feature_flags=(
            "maximum_length",
            "multiword",
            "repeated_phrase",
            "unicode",
        ),
    ),
)


def _fixture_payload(fixture: PublicFixture) -> dict[str, object]:
    return {
        "feature_flags": list(fixture.feature_flags),
        "fixture_commitment": fixture.fixture_commitment,
        "fixture_id": fixture.fixture_id,
        "input_sha256": fixture.input_sha256,
        "lexical_token_count": fixture.lexical_token_count,
        "ordinal": fixture.ordinal,
        "sentence_count": fixture.sentence_count,
    }


FIXTURE_SUITE_SHA256 = v1_contract.semantic_sha256(
    [_fixture_payload(row) for row in PUBLIC_FIXTURES]
)
FIXTURE_COMMITMENTS = MappingProxyType(
    {
        row.fixture_id: row.fixture_commitment
        for row in PUBLIC_FIXTURES
    }
)


def _validate_public_fixtures() -> None:
    if (
        tuple(row.ordinal for row in PUBLIC_FIXTURES)
        != tuple(range(len(PUBLIC_FIXTURES)))
        or len({row.fixture_id for row in PUBLIC_FIXTURES})
        != len(PUBLIC_FIXTURES)
        or tuple(
            row.lexical_token_count for row in PUBLIC_FIXTURES
        )
        != (17, 33, 64, 128, 175)
        or {row.sentence_count for row in PUBLIC_FIXTURES}
        != {1, 2, 5}
    ):
        raise FixedPublicQualificationError(
            "fixed_public_fixture_topology_invalid"
        )
    for fixture in PUBLIC_FIXTURES:
        episodes = closed_choice.build_hierarchical_episodes(
            fixture.story_text
        )
        if (
            sum(len(row.atoms) for row in episodes)
            != fixture.lexical_token_count
            or len({row.sentence_id for row in episodes})
            != fixture.sentence_count
            or _HEX64.fullmatch(fixture.input_sha256)
            is None
            or _HEX64.fullmatch(
                fixture.fixture_commitment
            )
            is None
        ):
            raise FixedPublicQualificationError(
                "fixed_public_fixture_contract_invalid"
            )


_validate_public_fixtures()


def _canonical_bytes(value: object) -> bytes:
    return v1_contract.canonical_json_bytes(value)


def _safe_hash(value: object) -> str:
    return v1_contract.semantic_sha256(value)


def _require_hex64(value: object, issue: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FixedPublicQualificationError(issue)
    return value


def _implementation_closure() -> dict[str, str]:
    project_root = Path(__file__).parents[2]
    manifest_root = project_root / "manifests"
    v1_root = project_root / "replication_runtime" / (
        "gscl_narrative_extractor_v1"
    )
    paths = {
        "fixed_public_qualification.py": Path(__file__),
        "fixed_public_qualification_aggregate.py": (
            Path(__file__).with_name(
                "fixed_public_qualification_aggregate.py"
            )
        ),
        "v2_closed_choice.py": Path(closed_choice.__file__),
        "v2_contract.py": Path(contract.__file__),
        "v2_memory_safe_qwen.py": Path(memory_safe_qwen.__file__),
        "v1_closed_choice_worker.py": (
            v1_root / "closed_choice_worker.py"
        ),
        "v1_contract.py": v1_root / "contract.py",
        "v1_worker.py": v1_root / "worker.py",
        "narrative_correspondence_parser.py": (
            project_root
            / "assumption_agent"
            / "gscl_narrative_correspondence_v1.py"
        ),
        "assumption_agent_init.py": (
            project_root / "assumption_agent" / "__init__.py"
        ),
        "assumption_agent_models.py": (
            project_root / "assumption_agent" / "models.py"
        ),
        "qualification_shard0.service": (
            manifest_root
            / "gscl_narrative_extractor_v2_fixed_qualification_shard0.service"
        ),
        "qualification_shard1.service": (
            manifest_root
            / "gscl_narrative_extractor_v2_fixed_qualification_shard1.service"
        ),
        "qualification_aggregate.service": (
            manifest_root
            / "gscl_narrative_extractor_v2_fixed_qualification_aggregate.service"
        ),
    }
    rows: dict[str, str] = {}
    for logical_name, path in paths.items():
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise FixedPublicQualificationError(
                "fixed_public_implementation_unreadable"
            ) from exc
        if not raw or len(raw) > MAXIMUM_IMPLEMENTATION_FILE_BYTES:
            raise FixedPublicQualificationError(
                "fixed_public_implementation_size_invalid"
            )
        rows[logical_name] = hashlib.sha256(raw).hexdigest()
    if len(rows) != len(paths):
        raise FixedPublicQualificationError(
            "fixed_public_implementation_name_collision"
        )
    return dict(sorted(rows.items()))


def _manifest_commitments(
    manifest: worker.ModelAssetManifest,
) -> dict[str, str]:
    return {
        "manifest_file_sha256": _require_hex64(
            manifest.manifest_file_sha256,
            "fixed_public_manifest_file_hash_invalid",
        ),
        "manifest_self_sha256": _require_hex64(
            manifest.self_sha256,
            "fixed_public_manifest_self_hash_invalid",
        ),
        "model_tree_sha256": _require_hex64(
            manifest.tree_sha256,
            "fixed_public_manifest_tree_hash_invalid",
        ),
    }


def _verify_model_binding(
    *,
    model_root: Path,
    manifest: worker.ModelAssetManifest,
) -> None:
    if (
        not isinstance(model_root, Path)
        or type(manifest) is not worker.ModelAssetManifest
        or manifest._marker
        is not worker._VERIFIED_MANIFEST_MARKER
    ):
        raise FixedPublicQualificationError(
            "fixed_public_model_authority_invalid"
        )
    try:
        observed = worker._scan_model_tree(model_root)
    except Exception as exc:
        raise FixedPublicQualificationError(
            "fixed_public_model_tree_unreadable"
        ) from exc
    if (
        observed != manifest.files
        or _safe_hash(list(observed)) != manifest.tree_sha256
    ):
        raise FixedPublicQualificationError(
            "fixed_public_model_tree_drifted"
        )


def _load_exact_runtime(
    *,
    model_root: Path,
    manifest: worker.ModelAssetManifest,
) -> memory_safe_qwen.MemorySafeQwenRuntime:
    return memory_safe_qwen.load_exact_cuda_fp16_runtime(
        model_root=model_root,
        manifest=manifest,
    )


def _run_fixed_teacher_forced_canary(
    runtime: object,
) -> Mapping[str, object]:
    operation = getattr(
        runtime, "run_fixed_teacher_forced_canary", None
    )
    if not callable(operation):
        raise FixedPublicQualificationError(
            "fixed_public_canary_surface_invalid"
        )
    value = operation()
    if not isinstance(value, Mapping):
        raise FixedPublicQualificationError(
            "fixed_public_canary_result_invalid"
        )
    result = dict(value)
    supplied = _require_hex64(
        result.get("self_sha256"),
        "fixed_public_canary_self_hash_invalid",
    )
    body = {
        key: child
        for key, child in result.items()
        if key != "self_sha256"
    }
    if (
        supplied != _safe_hash(body)
        or result.get("schema")
        != memory_safe_qwen.FIXED_CANARY_SCHEMA
        or result.get("short_pair_sha256")
        != memory_safe_qwen.FIXED_SHORT_CANARY_PAIR_SHA256
        or result.get("long_pair_sha256")
        != memory_safe_qwen.FIXED_LONG_CANARY_PAIR_SHA256
        or result.get(
            "short_strategy_vs_full_reference_exact"
        )
        is not True
        or isinstance(
            result.get("short_strategy_microunits"), bool
        )
        or not isinstance(
            result.get("short_strategy_microunits"), int
        )
        or result.get("short_strategy_microunits")
        != result.get("short_full_reference_microunits")
        or result.get("long_repeat_byte_exact") is not True
        or result.get(
            "fallback_independent_full_reference_passed"
        )
        is not True
        or not isinstance(
            result.get("long_answer_position_count"), int
        )
        or result["long_answer_position_count"]
        <= memory_safe_qwen.MAXIMUM_SPARSE_POSITIONS
        or result.get("free_form_generation_count") != 0
        or result.get("strategy")
        not in {
            memory_safe_qwen.SPARSE_STRATEGY,
            memory_safe_qwen.FALLBACK_STRATEGY,
        }
    ):
        raise FixedPublicQualificationError(
            "fixed_public_canary_verification_failed"
        )
    if (
        result["strategy"] == memory_safe_qwen.SPARSE_STRATEGY
        and (
            not isinstance(result.get("sparse_chunk_count"), int)
            or result["sparse_chunk_count"] < 2
        )
    ):
        raise FixedPublicQualificationError(
            "fixed_public_canary_chunking_failed"
        )
    return MappingProxyType(result)


def _safe_success_outcome(
    *,
    fixture: PublicFixture,
    decision: closed_choice.ClosedChoiceV2Decision,
    runtime_commitment: str,
) -> dict[str, object]:
    if type(decision) is not closed_choice.ClosedChoiceV2Decision:
        raise FixedPublicQualificationError(
            "fixed_public_decision_type_invalid"
        )
    receipt = dict(decision.receipt)
    if receipt.get("model_runtime_commitment") != runtime_commitment:
        raise FixedPublicQualificationError(
            "fixed_public_runtime_receipt_mismatch"
        )
    resource_summary = receipt.get("resource_summary")
    expected_resource_fields = {
        "candidate_count",
        "episode_count",
        "forward_batch_count",
        "maximum_candidates_in_one_batch",
        "maximum_span_lexical_width",
        "relation_count",
        "sentence_count",
    }
    if (
        type(resource_summary) is not dict
        or set(resource_summary) != expected_resource_fields
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in resource_summary.values()
        )
    ):
        raise FixedPublicQualificationError(
            "fixed_public_resource_summary_invalid"
        )
    try:
        extraction_semantic_hash = (
            decision.extraction.semantic_hash
        )
        mention_count = len(decision.extraction.mentions)
        generators = tuple(decision.extraction.generators)
        generator_count = len(generators)
        object_ids = tuple(
            decision.extraction.hypergraph.object_mention_ids
        )
        slot_ids = tuple(
            slot
            for generator in generators
            for slot in generator.slot_mention_ids
        )
    except Exception as exc:
        raise FixedPublicQualificationError(
            "fixed_public_extraction_surface_invalid"
        ) from exc
    endpoint_receipts = receipt.get(
        "endpoint_selection_receipt_commitments"
    )
    if (
        receipt.get("exclusive_endpoint_ownership") is not True
        or resource_summary["relation_count"] != generator_count
        or mention_count != 3 * generator_count
        or len(object_ids) != 2 * generator_count
        or len(slot_ids) != len(object_ids)
        or len(set(slot_ids)) != len(slot_ids)
        or set(slot_ids) != set(object_ids)
        or any(
            len(generator.slot_mention_ids) != 2
            for generator in generators
        )
        or type(endpoint_receipts) is not dict
        or len(endpoint_receipts) != generator_count
        or any(
            type(value) is not dict
            or set(value) != {"anchor", "object0", "object1"}
            or any(
                not isinstance(digest, str)
                or _HEX64.fullmatch(digest) is None
                for digest in value.values()
            )
            for value in endpoint_receipts.values()
        )
    ):
        raise FixedPublicQualificationError(
            "fixed_public_ownership_contract_invalid"
        )
    return {
        "canonical_completion_sha256": hashlib.sha256(
            decision.canonical_completion.encode("utf-8")
        ).hexdigest(),
        "decision_receipt_sha256": hashlib.sha256(
            decision.receipt_bytes
        ).hexdigest(),
        "disposition": "success",
        "extraction_semantic_hash": _require_hex64(
            extraction_semantic_hash,
            "fixed_public_extraction_hash_invalid",
        ),
        "fixture_commitment": fixture.fixture_commitment,
        "fixture_id": fixture.fixture_id,
        "generator_count": generator_count,
        "input_sha256": fixture.input_sha256,
        "mention_count": mention_count,
        "ordinal": fixture.ordinal,
        "resource_summary": dict(resource_summary),
        "selected_answer_token_count": (
            decision.selected_answer_token_count
        ),
        "wire_completion_sha256": hashlib.sha256(
            decision.wire_completion.encode("ascii")
        ).hexdigest(),
        "wire_completion_token_count": (
            decision.wire_completion_token_count
        ),
    }


def _safe_typed_failure_outcome(
    *,
    fixture: PublicFixture,
    error: contract.ClosedChoiceV2Error,
) -> dict[str, object]:
    failure = dict(contract.non_content_failure_record(error))
    return {
        "disposition": (
            "typed_abstention"
            if isinstance(error, contract.ClosedChoiceV2Abstention)
            else "typed_error"
        ),
        "error_category": failure["error_category"],
        "error_code": failure["error_code"],
        "fixture_commitment": fixture.fixture_commitment,
        "fixture_id": fixture.fixture_id,
        "input_sha256": fixture.input_sha256,
        "ordinal": fixture.ordinal,
        "pre_model_abstention": failure[
            "pre_model_abstention"
        ],
    }


def _run_fixture_once(
    *,
    runtime: object,
    runtime_commitment: str,
    fixture: PublicFixture,
) -> dict[str, object]:
    selector = getattr(runtime, "select_story", None)
    if not callable(selector):
        raise FixedPublicQualificationError(
            "fixed_public_runtime_select_surface_invalid"
        )
    try:
        decision = selector(fixture.story_text)
    except contract.ClosedChoiceV2Error as exc:
        return _safe_typed_failure_outcome(
            fixture=fixture, error=exc
        )
    return _safe_success_outcome(
        fixture=fixture,
        decision=decision,
        runtime_commitment=runtime_commitment,
    )


def _run_fixture_twice(
    *,
    runtime: object,
    runtime_commitment: str,
    fixture: PublicFixture,
) -> dict[str, object]:
    first = _run_fixture_once(
        runtime=runtime,
        runtime_commitment=runtime_commitment,
        fixture=fixture,
    )
    second = _run_fixture_once(
        runtime=runtime,
        runtime_commitment=runtime_commitment,
        fixture=fixture,
    )
    first_bytes = _canonical_bytes(first)
    second_bytes = _canonical_bytes(second)
    if first_bytes != second_bytes:
        raise FixedPublicQualificationError(
            "fixed_public_fixture_repeat_mismatch"
        )
    return {
        **first,
        "repeat_byte_exact": True,
        "repeat_count": REPEAT_COUNT,
        "repeat_outcome_sha256": hashlib.sha256(
            first_bytes
        ).hexdigest(),
    }


def _reset_cuda_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
    except Exception as exc:
        raise FixedPublicQualificationError(
            "fixed_public_cuda_peak_reset_failed"
        ) from exc


def _resource_peaks(
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    peak_fields = {
        "max_candidate_count": 0,
        "max_episode_count": 0,
        "max_forward_batch_count": 0,
        "max_relation_count": 0,
        "max_sentence_count": 0,
        "max_span_lexical_width": 0,
    }
    source_fields = {
        "max_candidate_count": "candidate_count",
        "max_episode_count": "episode_count",
        "max_forward_batch_count": "forward_batch_count",
        "max_relation_count": "relation_count",
        "max_sentence_count": "sentence_count",
        "max_span_lexical_width": (
            "maximum_span_lexical_width"
        ),
    }
    for outcome in outcomes:
        summary = outcome.get("resource_summary")
        if type(summary) is not dict:
            continue
        for target, source in source_fields.items():
            peak_fields[target] = max(
                peak_fields[target], int(summary[source])
            )
    allocated = 0
    reserved = 0
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)
            allocated = int(torch.cuda.max_memory_allocated(0))
            reserved = int(torch.cuda.max_memory_reserved(0))
    except Exception as exc:
        raise FixedPublicQualificationError(
            "fixed_public_cuda_peak_read_failed"
        ) from exc
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        **peak_fields,
        "cuda_max_memory_allocated_bytes": allocated,
        "cuda_max_memory_reserved_bytes": reserved,
        "process_max_rss_kib": int(usage.ru_maxrss),
    }


def _zero_counters() -> dict[str, int]:
    return {
        "api_access_count": 0,
        "free_form_generation_count": 0,
        "label_access_count": 0,
        "network_access_count": 0,
        "source_access_count": 0,
    }


def _publish_once(path: Path, raw: bytes) -> None:
    if (
        not isinstance(path, Path)
        or not raw
        or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES
    ):
        raise FixedPublicQualificationError(
            "fixed_public_publish_arguments_invalid"
        )
    parent = path.parent
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = parent.lstat()
    except OSError as exc:
        raise FixedPublicQualificationError(
            "fixed_public_output_root_invalid"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink():
        raise FixedPublicQualificationError(
            "fixed_public_output_root_invalid"
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
        raise FixedPublicQualificationError(
            "fixed_public_receipt_publish_failed"
        ) from exc


def run_fixed_public_qualification(
    *,
    model_root: Path,
    manifest: worker.ModelAssetManifest,
    output_root: Path,
    shard_index: int,
    shard_count: int,
) -> Mapping[str, object]:
    """Run one immutable fixture shard with one exact offline runtime."""

    if (
        not isinstance(output_root, Path)
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index not in {0, 1}
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count != SHARD_COUNT
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_coordinate_invalid"
        )
    _verify_model_binding(
        model_root=model_root, manifest=manifest
    )
    manifest_binding = _manifest_commitments(manifest)
    implementation = _implementation_closure()
    runtime = _load_exact_runtime(
        model_root=model_root, manifest=manifest
    )
    try:
        runtime_commitment = _require_hex64(
            runtime.runtime_commitment,
            "fixed_public_runtime_commitment_invalid",
        )
    except FixedPublicQualificationError:
        raise
    except Exception as exc:
        raise FixedPublicQualificationError(
            "fixed_public_runtime_commitment_unavailable"
        ) from exc
    _reset_cuda_peaks()
    canary = dict(_run_fixed_teacher_forced_canary(runtime))
    fixtures = tuple(
        row
        for row in PUBLIC_FIXTURES
        if row.ordinal % SHARD_COUNT == shard_index
    )
    outcomes = [
        _run_fixture_twice(
            runtime=runtime,
            runtime_commitment=runtime_commitment,
            fixture=fixture,
        )
        for fixture in fixtures
    ]
    counts = {
        "success": sum(
            row["disposition"] == "success"
            for row in outcomes
        ),
        "typed_abstention": sum(
            row["disposition"] == "typed_abstention"
            for row in outcomes
        ),
        "typed_error": sum(
            row["disposition"] == "typed_error"
            for row in outcomes
        ),
    }
    all_relation_bearing_fixtures_succeeded = (
        counts["success"] == len(fixtures)
        and counts["typed_abstention"] == 0
        and counts["typed_error"] == 0
    )
    body: dict[str, object] = {
        "counters": _zero_counters(),
        "fixture_commitments": {
            row.fixture_id: row.fixture_commitment
            for row in fixtures
        },
        "fixture_count": len(fixtures),
        "fixture_ordinals": [
            row.ordinal for row in fixtures
        ],
        "fixture_suite_sha256": FIXTURE_SUITE_SHA256,
        "implementation_closure": implementation,
        "implementation_closure_sha256": _safe_hash(
            implementation
        ),
        "manifest_commitments": manifest_binding,
        "outcome_counts": counts,
        "outcomes": outcomes,
        "outcomes_commitment": _safe_hash(outcomes),
        "qualification_passed": (
            all_relation_bearing_fixtures_succeeded
        ),
        "repeat_byte_exact": all(
            row["repeat_byte_exact"] is True
            for row in outcomes
        ),
        "repeat_count": REPEAT_COUNT,
        "resource_peaks": _resource_peaks(outcomes),
        "runtime_commitment": runtime_commitment,
        "schema": SHARD_RECEIPT_SCHEMA,
        "shard_count": SHARD_COUNT,
        "shard_index": shard_index,
        "teacher_forced_canary": canary,
        "teacher_forced_canary_self_sha256": canary[
            "self_sha256"
        ],
        "version": VERSION,
    }
    receipt = {
        **body,
        "self_sha256": _safe_hash(body),
    }
    raw = _canonical_bytes(receipt)
    _publish_once(output_root / SHARD_OUTPUT_NAME, raw)
    return MappingProxyType(receipt)


def _load_shard_receipt(path: Path) -> dict[str, object]:
    if not isinstance(path, Path):
        raise FixedPublicQualificationError(
            "fixed_public_shard_receipt_path_invalid"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FixedPublicQualificationError(
            "fixed_public_shard_receipt_unreadable"
        ) from exc
    if not raw or len(raw) > MAXIMUM_SAFE_RECEIPT_BYTES:
        raise FixedPublicQualificationError(
            "fixed_public_shard_receipt_size_invalid"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise FixedPublicQualificationError(
            "fixed_public_shard_receipt_json_invalid"
        ) from exc
    if (
        type(value) is not dict
        or _canonical_bytes(value) != raw
        or value.get("schema") != SHARD_RECEIPT_SCHEMA
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_receipt_canonical_invalid"
        )
    supplied = _require_hex64(
        value.get("self_sha256"),
        "fixed_public_shard_receipt_self_invalid",
    )
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    if supplied != _safe_hash(body):
        raise FixedPublicQualificationError(
            "fixed_public_shard_receipt_self_mismatch"
        )
    return value


def _safe_outcome_is_valid(value: Mapping[str, object]) -> bool:
    common = {
        "disposition",
        "fixture_commitment",
        "fixture_id",
        "input_sha256",
        "ordinal",
        "repeat_byte_exact",
        "repeat_count",
        "repeat_outcome_sha256",
    }
    disposition = value.get("disposition")
    if disposition == "success":
        expected = common | {
            "canonical_completion_sha256",
            "decision_receipt_sha256",
            "extraction_semantic_hash",
            "generator_count",
            "mention_count",
            "resource_summary",
            "selected_answer_token_count",
            "wire_completion_sha256",
            "wire_completion_token_count",
        }
        summary = value.get("resource_summary")
        hashes = (
            value.get("canonical_completion_sha256"),
            value.get("decision_receipt_sha256"),
            value.get("extraction_semantic_hash"),
            value.get("input_sha256"),
            value.get("repeat_outcome_sha256"),
            value.get("wire_completion_sha256"),
        )
        generator_count = value.get("generator_count")
        return bool(
            set(value) == expected
            and type(summary) is dict
            and set(summary)
            == {
                "candidate_count",
                "episode_count",
                "forward_batch_count",
                "maximum_candidates_in_one_batch",
                "maximum_span_lexical_width",
                "relation_count",
                "sentence_count",
            }
            and all(
                isinstance(digest, str)
                and _HEX64.fullmatch(digest) is not None
                for digest in hashes
            )
            and isinstance(generator_count, int)
            and not isinstance(generator_count, bool)
            and generator_count >= 1
            and value.get("mention_count")
            == 3 * generator_count
            and summary["relation_count"] == generator_count
            and 0 < summary["candidate_count"]
            <= closed_choice.MAXIMUM_TOTAL_CANDIDATES
            and 0 < summary["forward_batch_count"]
            <= closed_choice.MAXIMUM_FORWARD_BATCH_CALLS
            and summary["maximum_candidates_in_one_batch"]
            <= closed_choice.SCORING_BATCH_SIZE
            and 1 <= summary["maximum_span_lexical_width"]
            <= closed_choice.MAXIMUM_SPAN_LEXICAL_WIDTH
            and isinstance(
                value.get("selected_answer_token_count"), int
            )
            and value["selected_answer_token_count"] > 0
            and isinstance(
                value.get("wire_completion_token_count"), int
            )
            and 0 < value["wire_completion_token_count"]
            < closed_choice.MAXIMUM_WIRE_COMPLETION_TOKENS
        )
    if disposition in {"typed_abstention", "typed_error"}:
        expected = common | {
            "error_category",
            "error_code",
            "pre_model_abstention",
        }
        code = value.get("error_code")
        category = value.get("error_category")
        expected_category = contract.ERROR_TAXONOMY.get(code)
        abstention_category = expected_category in {
            contract.ErrorCategory.CATALOG,
            contract.ErrorCategory.CONTEXT,
            contract.ErrorCategory.SELECTION,
        }
        return bool(
            set(value) == expected
            and expected_category is not None
            and category == expected_category.value
            and (
                (disposition == "typed_abstention")
                == abstention_category
            )
            and type(value.get("pre_model_abstention")) is bool
            and isinstance(value.get("input_sha256"), str)
            and _HEX64.fullmatch(value["input_sha256"])
            is not None
            and isinstance(
                value.get("repeat_outcome_sha256"), str
            )
            and _HEX64.fullmatch(
                value["repeat_outcome_sha256"]
            )
            is not None
        )
    return False


def _validate_shard_receipt(
    value: Mapping[str, object],
) -> None:
    expected_root_fields = {
        "counters",
        "fixture_commitments",
        "fixture_count",
        "fixture_ordinals",
        "fixture_suite_sha256",
        "implementation_closure",
        "implementation_closure_sha256",
        "manifest_commitments",
        "outcome_counts",
        "outcomes",
        "outcomes_commitment",
        "qualification_passed",
        "repeat_byte_exact",
        "repeat_count",
        "resource_peaks",
        "runtime_commitment",
        "schema",
        "self_sha256",
        "shard_count",
        "shard_index",
        "teacher_forced_canary",
        "teacher_forced_canary_self_sha256",
        "version",
    }
    index = value.get("shard_index")
    if (
        set(value) != expected_root_fields
        or index not in {0, 1}
        or isinstance(index, bool)
        or value.get("shard_count") != SHARD_COUNT
        or value.get("fixture_suite_sha256")
        != FIXTURE_SUITE_SHA256
        or value.get("repeat_count") != REPEAT_COUNT
        or value.get("repeat_byte_exact") is not True
        or value.get("counters") != _zero_counters()
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_receipt_contract_invalid"
        )
    expected = [
        row for row in PUBLIC_FIXTURES if row.ordinal % 2 == index
    ]
    if (
        value.get("fixture_ordinals")
        != [row.ordinal for row in expected]
        or value.get("fixture_count") != len(expected)
        or value.get("fixture_commitments")
        != {
            row.fixture_id: row.fixture_commitment
            for row in expected
        }
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_fixture_set_invalid"
        )
    outcomes = value.get("outcomes")
    if (
        type(outcomes) is not list
        or [row.get("ordinal") for row in outcomes]
        != [row.ordinal for row in expected]
        or any(
            type(row) is not dict
            or row.get("fixture_id") != fixture.fixture_id
            or row.get("fixture_commitment")
            != fixture.fixture_commitment
            or row.get("input_sha256") != fixture.input_sha256
            or row.get("repeat_count") != REPEAT_COUNT
            or row.get("repeat_byte_exact") is not True
            or not _safe_outcome_is_valid(row)
            for row, fixture in zip(
                outcomes, expected, strict=True
            )
        )
        or value.get("outcomes_commitment")
        != _safe_hash(outcomes)
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_outcomes_invalid"
        )
    observed_counts = {
        "success": sum(
            row["disposition"] == "success" for row in outcomes
        ),
        "typed_abstention": sum(
            row["disposition"] == "typed_abstention"
            for row in outcomes
        ),
        "typed_error": sum(
            row["disposition"] == "typed_error"
            for row in outcomes
        ),
    }
    if (
        value.get("outcome_counts") != observed_counts
        or value.get("qualification_passed")
        is not (
            observed_counts["success"] == len(expected)
            and observed_counts["typed_abstention"] == 0
            and observed_counts["typed_error"] == 0
        )
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_outcome_counts_invalid"
        )
    canary = value.get("teacher_forced_canary")
    if (
        type(canary) is not dict
        or canary.get("self_sha256")
        != value.get("teacher_forced_canary_self_sha256")
        or canary.get("self_sha256")
        != _safe_hash(
            {
                key: child
                for key, child in canary.items()
                if key != "self_sha256"
            }
        )
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_canary_invalid"
        )
    if (
        canary.get(
            "short_strategy_vs_full_reference_exact"
        )
        is not True
        or canary.get("long_repeat_byte_exact") is not True
        or canary.get(
            "fallback_independent_full_reference_passed"
        )
        is not True
        or canary.get("free_form_generation_count") != 0
        or not isinstance(
            canary.get("long_answer_position_count"), int
        )
        or canary["long_answer_position_count"]
        <= memory_safe_qwen.MAXIMUM_SPARSE_POSITIONS
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_canary_semantics_invalid"
        )
    implementation = value.get("implementation_closure")
    manifest = value.get("manifest_commitments")
    resources = value.get("resource_peaks")
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
        type(implementation) is not dict
        or value.get("implementation_closure_sha256")
        != _safe_hash(implementation)
        or type(manifest) is not dict
        or set(manifest)
        != {
            "manifest_file_sha256",
            "manifest_self_sha256",
            "model_tree_sha256",
        }
        or any(
            not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
            for digest in manifest.values()
        )
        or _HEX64.fullmatch(
            str(value.get("runtime_commitment"))
        )
        is None
        or type(resources) is not dict
        or set(resources) != expected_resource_fields
        or any(
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < 0
            for observed in resources.values()
        )
    ):
        raise FixedPublicQualificationError(
            "fixed_public_shard_binding_invalid"
        )


def aggregate_fixed_public_qualification(
    *,
    shard_receipts: tuple[Path, Path],
    output_root: Path,
) -> Mapping[str, object]:
    """Pure-offline verifier for the two immutable shard receipts."""

    if (
        type(shard_receipts) is not tuple
        or len(shard_receipts) != SHARD_COUNT
        or any(
            not isinstance(path, Path) for path in shard_receipts
        )
        or not isinstance(output_root, Path)
    ):
        raise FixedPublicQualificationError(
            "fixed_public_aggregate_arguments_invalid"
        )
    rows = [_load_shard_receipt(path) for path in shard_receipts]
    for row in rows:
        _validate_shard_receipt(row)
    rows.sort(key=lambda row: int(row["shard_index"]))
    if [row["shard_index"] for row in rows] != [0, 1]:
        raise FixedPublicQualificationError(
            "fixed_public_aggregate_shards_invalid"
        )
    consistency_fields = (
        "fixture_suite_sha256",
        "implementation_closure",
        "implementation_closure_sha256",
        "manifest_commitments",
        "runtime_commitment",
        "teacher_forced_canary",
        "teacher_forced_canary_self_sha256",
        "version",
    )
    if any(
        rows[0][field] != rows[1][field]
        for field in consistency_fields
    ):
        raise FixedPublicQualificationError(
            "fixed_public_aggregate_binding_mismatch"
        )
    current_implementation = _implementation_closure()
    if (
        rows[0]["implementation_closure"]
        != current_implementation
        or rows[0]["implementation_closure_sha256"]
        != _safe_hash(current_implementation)
    ):
        raise FixedPublicQualificationError(
            "fixed_public_aggregate_implementation_drifted"
        )
    all_outcomes = sorted(
        [
            outcome
            for row in rows
            for outcome in row["outcomes"]
        ],
        key=lambda outcome: int(outcome["ordinal"]),
    )
    if (
        [row["ordinal"] for row in all_outcomes]
        != list(range(len(PUBLIC_FIXTURES)))
        or len(
            {row["fixture_id"] for row in all_outcomes}
        )
        != len(PUBLIC_FIXTURES)
    ):
        raise FixedPublicQualificationError(
            "fixed_public_aggregate_fixture_union_invalid"
        )
    count_keys = ("success", "typed_abstention", "typed_error")
    counts = {
        key: sum(
            int(row["outcome_counts"][key]) for row in rows
        )
        for key in count_keys
    }
    resource_keys = tuple(rows[0]["resource_peaks"])
    if set(resource_keys) != set(rows[1]["resource_peaks"]):
        raise FixedPublicQualificationError(
            "fixed_public_aggregate_resource_fields_invalid"
        )
    peaks = {
        key: max(
            int(row["resource_peaks"][key]) for row in rows
        )
        for key in resource_keys
    }
    body: dict[str, object] = {
        "counters": _zero_counters(),
        "fixture_commitments": dict(FIXTURE_COMMITMENTS),
        "fixture_count": len(PUBLIC_FIXTURES),
        "fixture_ordinals": list(
            range(len(PUBLIC_FIXTURES))
        ),
        "fixture_suite_sha256": FIXTURE_SUITE_SHA256,
        "implementation_closure": current_implementation,
        "implementation_closure_sha256": _safe_hash(
            current_implementation
        ),
        "manifest_commitments": rows[0][
            "manifest_commitments"
        ],
        "outcome_counts": counts,
        "outcomes_commitment": _safe_hash(all_outcomes),
        "qualification_passed": (
            counts["success"] == len(PUBLIC_FIXTURES)
            and counts["typed_abstention"] == 0
            and counts["typed_error"] == 0
            and all(
                row["qualification_passed"] is True
                for row in rows
            )
        ),
        "repeat_byte_exact": all(
            row["repeat_byte_exact"] is True for row in rows
        ),
        "repeat_count": REPEAT_COUNT,
        "resource_peaks": peaks,
        "runtime_commitment": rows[0]["runtime_commitment"],
        "schema": AGGREGATE_RECEIPT_SCHEMA,
        "shard_count": SHARD_COUNT,
        "shard_receipt_self_sha256": {
            str(row["shard_index"]): row["self_sha256"]
            for row in rows
        },
        "teacher_forced_canary": rows[0][
            "teacher_forced_canary"
        ],
        "teacher_forced_canary_self_sha256": rows[0][
            "teacher_forced_canary_self_sha256"
        ],
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
            "Run one fixed, public, source-free extractor-v2 "
            "qualification shard"
        )
    )
    parser.add_argument(
        "--model-root", required=True, type=Path
    )
    parser.add_argument(
        "--model-manifest", required=True, type=Path
    )
    parser.add_argument(
        "--output-root", required=True, type=Path
    )
    parser.add_argument(
        "--shard-index", required=True, type=int, choices=(0, 1)
    )
    parser.add_argument(
        "--shard-count", required=True, type=int, choices=(2,)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = worker.load_model_asset_manifest(
        manifest_path=arguments.model_manifest,
        model_root=arguments.model_root,
    )
    receipt = run_fixed_public_qualification(
        model_root=arguments.model_root,
        manifest=manifest,
        output_root=arguments.output_root,
        shard_index=arguments.shard_index,
        shard_count=arguments.shard_count,
    )
    print(receipt["self_sha256"])
    return 0


def _aggregate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pure-offline aggregate verifier for both fixed "
            "extractor-v2 qualification shards"
        )
    )
    parser.add_argument(
        "--shard-0-receipt", required=True, type=Path
    )
    parser.add_argument(
        "--shard-1-receipt", required=True, type=Path
    )
    parser.add_argument(
        "--output-root", required=True, type=Path
    )
    return parser


def main_aggregate(argv: Sequence[str] | None = None) -> int:
    arguments = _aggregate_parser().parse_args(argv)
    receipt = aggregate_fixed_public_qualification(
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
    "FixedPublicQualificationError",
    "PUBLIC_FIXTURES",
    "PublicFixture",
    "REPEAT_COUNT",
    "SHARD_COUNT",
    "SHARD_OUTPUT_NAME",
    "SHARD_RECEIPT_SCHEMA",
    "VERSION",
    "aggregate_fixed_public_qualification",
    "main",
    "main_aggregate",
    "run_fixed_public_qualification",
]

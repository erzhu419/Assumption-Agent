from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pytest

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_worker as closed_v1,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    contract as contract_v1,
)
from replication_runtime.gscl_narrative_extractor_v1 import worker
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as closed_v2,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    contract as v2_contract,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    fixed_arn_input_compatibility as compatibility,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    fixed_public_integrated_qualification as integrated,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    fixed_public_qualification as extractor,
)


_RUNTIME_COMMITMENT = hashlib.sha256(
    b"fixed-arn-compatibility-runtime"
).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    if isinstance(value, Mapping):
        value = dict(value)
    return contract_v1.canonical_json_bytes(value)


def _manifest(root: Path) -> worker.ModelAssetManifest:
    model_root = root / "model"
    model_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = b"synthetic fixed compatibility model\n"
    model_file = model_root / "model.safetensors"
    model_file.write_bytes(payload)
    files = (
        {
            "path": model_file.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        },
    )
    return worker.ModelAssetManifest(
        declarations={},
        files=files,
        runtime_requirements={},
        tree_sha256=contract_v1.semantic_sha256(list(files)),
        self_sha256=hashlib.sha256(b"manifest-self").hexdigest(),
        manifest_file_sha256=hashlib.sha256(
            b"manifest-file"
        ).hexdigest(),
        _marker=worker._VERIFIED_MANIFEST_MARKER,
    )


def _manifest_binding(
    manifest: worker.ModelAssetManifest,
) -> dict[str, str]:
    return {
        "manifest_file_sha256": (
            manifest.manifest_file_sha256
        ),
        "manifest_self_sha256": manifest.self_sha256,
        "model_tree_sha256": manifest.tree_sha256,
    }


def _integrated_receipt(
    path: Path,
    manifest: worker.ModelAssetManifest,
    *,
    passed: bool = True,
) -> Path:
    implementation = integrated._implementation_closure()
    body: dict[str, object] = {
        "arm_contract_passed": passed,
        "arm_repeat_byte_exact": passed,
        "counters": integrated._zero_counters(),
        "extractor_repeat_byte_exact": passed,
        "extractor_runtime_commitment": _RUNTIME_COMMITMENT,
        "implementation_closure": implementation,
        "implementation_closure_sha256": (
            contract_v1.semantic_sha256(implementation)
        ),
        "manifest_commitments": _manifest_binding(manifest),
        "qualification_passed": passed,
        "schema": integrated.RECEIPT_SCHEMA,
    }
    path.write_bytes(
        _canonical_bytes(
            {
                **body,
                "self_sha256": contract_v1.semantic_sha256(
                    body
                ),
            }
        )
    )
    return path


def _story(tag: str) -> str:
    return (
        f"{tag} guides Alpha toward Beta while Gamma supports Delta "
        "and Epsilon follows Zeta before Eta helps Theta in this "
        "fixed public compatibility narrative."
    )


def _predictor_pack(
    path: Path,
    *,
    row_count: int,
    add_label: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    stories: list[str] = []
    rows: list[dict[str, object]] = []
    for ordinal in range(row_count):
        row_stories = (
            _story(f"Q{ordinal}"),
            _story(f"A{ordinal}"),
            _story(f"B{ordinal}"),
        )
        stories.extend(row_stories)
        row: dict[str, object] = {
            "opaque_item_id": hashlib.sha256(
                f"item-{ordinal}".encode()
            ).hexdigest(),
            "query_narrative": row_stories[0],
            "first_choice": row_stories[1],
            "second_choice": row_stories[2],
        }
        if add_label:
            row["correct_answer"] = 0
        rows.append(row)
    pack = {
        "adapter_qualification_self_hash": (
            compatibility.EXPECTED_ADAPTER_QUALIFICATION_SELF_SHA256
        ),
        "column_contract": list(
            compatibility.PREDICTOR_COLUMNS
        ),
        "lineage": compatibility.EXPECTED_LINEAGE,
        "rows": rows,
        "schema": compatibility.EXPECTED_PREDICTOR_SCHEMA,
        "source_sha256": compatibility.EXPECTED_SOURCE_SHA256,
        "source_verification_self_hash": (
            compatibility.EXPECTED_SOURCE_VERIFICATION_SELF_SHA256
        ),
    }
    raw = _canonical_bytes(pack)
    path.write_bytes(raw)
    return path, tuple(stories)


class _Backend:
    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def score_batch(
        self, pairs: tuple[closed_v1.PromptAnswer, ...]
    ) -> tuple[closed_v2.TeacherForcedScore, ...]:
        rows: list[closed_v2.TeacherForcedScore] = []
        for pair in pairs:
            preferred = int(
                pair.candidate_key.endswith(
                    ".plan.one_relation"
                )
            )
            tokens = max(1, len(pair.answer.split()))
            rows.append(
                closed_v2.TeacherForcedScore(
                    total_logprob_microunits=(
                        preferred * 1_000_000 * tokens
                    ),
                    answer_token_count=tokens,
                    context_and_answer_token_count=tokens + 64,
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        return max(1, len(completion.encode("utf-8")) // 4)


def _parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource(
            "fixed.compatibility."
            + hashlib.sha256(
                story.encode("utf-8")
            ).hexdigest()[:24],
            story,
        ),
        completion,
    )


class _Runtime:
    def __init__(
        self, *, abstain_sha256: str | None = None
    ) -> None:
        self.calls: Counter[str] = Counter()
        self.abstain_sha256 = abstain_sha256

    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def select_story(
        self, story_text: str
    ) -> closed_v2.ClosedChoiceV2Decision:
        digest = hashlib.sha256(
            story_text.encode("utf-8")
        ).hexdigest()
        self.calls[digest] += 1
        if digest == self.abstain_sha256:
            raise v2_contract.ClosedChoiceV2Abstention(
                "V2_PLAN_NO_RELATION_SELECTED",
                before_model_forward=False,
            )
        return closed_v2.select_hierarchical_qualification_only(
            story_text,
            backend=_Backend(),
            narrative_parser=_parser,
        )


def _prepare(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    row_count: int = 4,
    add_label: bool = False,
    passed_upstream: bool = True,
    abstain_story: str | None = None,
) -> tuple[
    worker.ModelAssetManifest,
    Path,
    Path,
    list[_Runtime],
    tuple[str, ...],
]:
    root.mkdir(mode=0o700, parents=True)
    manifest = _manifest(root)
    upstream = _integrated_receipt(
        root / "integrated.safe.json",
        manifest,
        passed=passed_upstream,
    )
    predictor, stories = _predictor_pack(
        root / "predictor.json",
        row_count=row_count,
        add_label=add_label,
    )
    raw = predictor.read_bytes()
    monkeypatch.setattr(
        compatibility, "EXPECTED_ITEM_COUNT", row_count
    )
    monkeypatch.setattr(
        compatibility,
        "EXPECTED_STORY_COUNT",
        3 * row_count,
    )
    monkeypatch.setattr(
        compatibility,
        "EXPECTED_PREDICTOR_FILE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        extractor,
        "_verify_model_binding",
        lambda *, model_root, manifest: None,
    )
    runtimes: list[_Runtime] = []
    abstain_sha = (
        None
        if abstain_story is None
        else hashlib.sha256(
            abstain_story.encode("utf-8")
        ).hexdigest()
    )

    def load(**_kwargs: object) -> _Runtime:
        runtime = _Runtime(abstain_sha256=abstain_sha)
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(
        extractor, "_load_exact_runtime", load
    )
    return manifest, upstream, predictor, runtimes, stories


def _run_shard(
    *,
    manifest: worker.ModelAssetManifest,
    upstream: Path,
    predictor: Path,
    output: Path,
    index: int,
) -> Mapping[str, object]:
    return compatibility.run_fixed_arn_input_compatibility_shard(
        model_root=manifest_path_root(manifest, predictor),
        manifest=manifest,
        integrated_qualification_receipt=upstream,
        predictor_pack=predictor,
        output_root=output,
        shard_index=index,
        shard_count=2,
    )


def manifest_path_root(
    _manifest: worker.ModelAssetManifest, predictor: Path
) -> Path:
    return predictor.parent / "model"


def _nested_strings(value: object) -> tuple[str, ...]:
    rows: list[str] = []
    if isinstance(value, str):
        rows.append(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            rows.extend(_nested_strings(key))
            rows.extend(_nested_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            rows.extend(_nested_strings(child))
    return tuple(rows)


def test_shard_surface_has_no_label_or_scorer_input() -> None:
    signature = inspect.signature(
        compatibility.run_fixed_arn_input_compatibility_shard
    )
    assert tuple(signature.parameters) == (
        "model_root",
        "manifest",
        "integrated_qualification_receipt",
        "predictor_pack",
        "output_root",
        "shard_index",
        "shard_count",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not {
        "answer",
        "label",
        "linkage",
        "scorer",
        "online_evaluator",
    } & set(signature.parameters)


def test_two_shards_and_aggregate_cover_every_story_without_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, upstream, predictor, runtimes, stories = _prepare(
        monkeypatch, tmp_path / "happy"
    )
    shard_paths: list[Path] = []
    for index in (0, 1):
        output = tmp_path / "happy" / f"shard{index}"
        receipt = _run_shard(
            manifest=manifest,
            upstream=upstream,
            predictor=predictor,
            output=output,
            index=index,
        )
        assert receipt["qualification_passed"] is True
        assert receipt["outcome_counts"] == {
            "success": 6,
            "typed_abstention": 0,
            "typed_error": 0,
            "untyped_error": 0,
        }
        assert receipt["shard_row_count"] == 2
        assert receipt["shard_story_count"] == 6
        assert receipt["access_counters"] == {
            "api_access_count": 0,
            "free_form_generation_count": 0,
            "label_access_count": 0,
            "network_access_count": 0,
            "online_evaluator_access_count": 0,
            "predictor_pack_access_count": 1,
            "raw_source_access_count": 0,
            "scorer_access_count": 0,
            "source_access_count": 1,
        }
        nested = _nested_strings(receipt)
        assert all(story not in nested for story in stories)
        shard_paths.append(
            output / compatibility.SHARD_OUTPUT_NAME
        )
    assert len(runtimes) == 2
    aggregate = compatibility.aggregate_fixed_arn_input_compatibility(
        shard_receipts=(shard_paths[0], shard_paths[1]),
        output_root=tmp_path / "happy" / "aggregate",
    )
    assert aggregate["qualification_passed"] is True
    assert aggregate["total_row_count"] == 4
    assert aggregate["total_story_count"] == 12
    assert aggregate["outcome_counts"]["success"] == 12
    assert aggregate["access_counters"][
        "predictor_pack_access_count"
    ] == 2
    assert aggregate["access_counters"]["source_access_count"] == 2
    assert all(
        sum(runtime.calls.values()) == 6
        for runtime in runtimes
    )
    body = {
        key: child
        for key, child in aggregate.items()
        if key != "self_sha256"
    }
    assert aggregate["self_sha256"] == (
        contract_v1.semantic_sha256(body)
    )


def test_label_like_predictor_field_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, upstream, predictor, runtimes, _stories = _prepare(
        monkeypatch,
        tmp_path / "label-field",
        add_label=True,
    )
    with pytest.raises(
        compatibility.FixedArnInputCompatibilityError,
        match="predictor",
    ):
        _run_shard(
            manifest=manifest,
            upstream=upstream,
            predictor=predictor,
            output=tmp_path / "label-field" / "out",
            index=0,
        )
    assert runtimes == []


def test_failed_upstream_is_rejected_before_predictor_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, upstream, predictor, runtimes, _stories = _prepare(
        monkeypatch,
        tmp_path / "upstream-failed",
        passed_upstream=False,
    )
    with pytest.raises(
        compatibility.FixedArnInputCompatibilityError,
        match="upstream",
    ):
        _run_shard(
            manifest=manifest,
            upstream=upstream,
            predictor=predictor,
            output=tmp_path / "upstream-failed" / "out",
            index=0,
        )
    assert runtimes == []


def test_typed_abstention_is_aggregated_without_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_story = _story("Q0")
    manifest, upstream, predictor, _runtimes, stories = _prepare(
        monkeypatch,
        tmp_path / "abstention",
        abstain_story=selected_story,
    )
    receipt = _run_shard(
        manifest=manifest,
        upstream=upstream,
        predictor=predictor,
        output=tmp_path / "abstention" / "shard0",
        index=0,
    )
    assert receipt["qualification_passed"] is False
    assert receipt["outcome_counts"] == {
        "success": 5,
        "typed_abstention": 1,
        "typed_error": 0,
        "untyped_error": 0,
    }
    assert receipt["error_category_counts"] == {
        "selection": 1
    }
    assert receipt["error_code_counts"] == {
        "V2_PLAN_NO_RELATION_SELECTED": 1
    }
    nested = _nested_strings(receipt)
    assert all(story not in nested for story in stories)


def test_attempt_barrier_prevents_manual_second_source_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, upstream, predictor, runtimes, _stories = _prepare(
        monkeypatch, tmp_path / "single-attempt"
    )
    output = tmp_path / "single-attempt" / "shard0"
    first = _run_shard(
        manifest=manifest,
        upstream=upstream,
        predictor=predictor,
        output=output,
        index=0,
    )
    assert first["qualification_passed"] is True
    assert len(runtimes) == 1
    with pytest.raises(
        compatibility.FixedArnInputCompatibilityError,
        match="publish",
    ):
        _run_shard(
            manifest=manifest,
            upstream=upstream,
            predictor=predictor,
            output=output,
            index=0,
        )
    assert len(runtimes) == 1


def test_systemd_templates_are_manual_only_and_aggregate_is_pure(
) -> None:
    manifest_root = Path(__file__).parents[1] / "manifests"
    shard_names = (
        "gscl_narrative_extractor_v2_fixed_arn_compatibility_shard0.service",
        "gscl_narrative_extractor_v2_fixed_arn_compatibility_shard1.service",
    )
    shard_text = [
        (manifest_root / name).read_text("utf-8")
        for name in shard_names
    ]
    aggregate = (
        manifest_root
        / "gscl_narrative_extractor_v2_fixed_arn_compatibility_aggregate.service"
    ).read_text("utf-8")
    assert all("[Install]" not in text for text in shard_text)
    assert all(
        "After=default.target" not in text for text in shard_text
    )
    assert "[Install]" not in aggregate
    assert "Requires=" not in aggregate
    assert all(name in aggregate for name in shard_names)

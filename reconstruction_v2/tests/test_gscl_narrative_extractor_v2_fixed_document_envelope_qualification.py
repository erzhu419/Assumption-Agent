from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import pytest

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as leaf_v2,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    document_envelope,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    fixed_document_envelope_qualification as qualification,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    fixed_public_qualification as leaf_qualification,
)
from replication_runtime.gscl_narrative_extractor_v2 import memory_safe_qwen
from replication_runtime.gscl_narrative_extractor_v2.contract import (
    ClosedChoiceV2Abstention,
)


_RUNTIME_COMMITMENT = hashlib.sha256(
    b"fixed-document-envelope-test-runtime"
).hexdigest()
_LEXICAL = re.compile(r"[^\W_]+", re.UNICODE)


def _canonical_bytes(value: object) -> bytes:
    return qualification._canonical_bytes(value)


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    return {**body, "self_sha256": qualification._safe_hash(dict(body))}


class _Backend:
    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def score_batch(
        self, pairs: tuple[PromptAnswer, ...]
    ) -> tuple[leaf_v2.TeacherForcedScore, ...]:
        rows: list[leaf_v2.TeacherForcedScore] = []
        for pair in pairs:
            preferred = int(
                pair.candidate_key.endswith(".plan.one_relation")
            )
            answer_tokens = max(1, len(pair.answer.split()))
            rows.append(
                leaf_v2.TeacherForcedScore(
                    total_logprob_microunits=(
                        preferred * 1_000_000 * answer_tokens
                    ),
                    answer_token_count=answer_tokens,
                    context_and_answer_token_count=answer_tokens + 64,
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(self, completion: str) -> int:
        return max(1, len(completion.encode("utf-8")) // 4)


def _parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource(
            "fixed.document."
            + hashlib.sha256(story.encode("utf-8")).hexdigest()[:20],
            story,
        ),
        completion,
    )


class _LeafSelector:
    def __init__(
        self, *, no_relation: bool = False, typed_failure: bool = False
    ) -> None:
        self.no_relation = no_relation
        self.typed_failure = typed_failure

    def select_story(self, story_text: str) -> leaf_v2.ClosedChoiceV2Decision:
        if self.no_relation:
            raise ClosedChoiceV2Abstention(
                "V2_PLAN_NO_RELATION_SELECTED",
                before_model_forward=False,
            )
        if self.typed_failure:
            raise RuntimeError("private runtime failure must not enter receipt")
        return leaf_v2.select_hierarchical_qualification_only(
            story_text,
            backend=_Backend(),
            narrative_parser=_parser,
        )


def _canary() -> Mapping[str, object]:
    body: dict[str, object] = {
        "fallback_independent_full_reference_passed": True,
        "free_form_generation_count": 0,
        "long_answer_position_count": 200,
        "long_pair_sha256": memory_safe_qwen.FIXED_LONG_CANARY_PAIR_SHA256,
        "long_repeat_byte_exact": True,
        "long_score_sha256": hashlib.sha256(b"long").hexdigest(),
        "schema": memory_safe_qwen.FIXED_CANARY_SCHEMA,
        "short_full_reference_microunits": -100,
        "short_pair_sha256": memory_safe_qwen.FIXED_SHORT_CANARY_PAIR_SHA256,
        "short_strategy_microunits": -100,
        "short_strategy_vs_full_reference_exact": True,
        "sparse_chunk_count": 2,
        "strategy": memory_safe_qwen.SPARSE_STRATEGY,
    }
    return MappingProxyType(_self_hashed(body))


class _Runtime:
    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def run_fixed_teacher_forced_canary(self) -> Mapping[str, object]:
        return _canary()


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        manifest_file_sha256=hashlib.sha256(b"manifest-file").hexdigest(),
        self_sha256=hashlib.sha256(b"manifest-self").hexdigest(),
        tree_sha256=hashlib.sha256(b"model-tree").hexdigest(),
    )


def _manifest_commitments(manifest: SimpleNamespace) -> dict[str, str]:
    return {
        "manifest_file_sha256": manifest.manifest_file_sha256,
        "manifest_self_sha256": manifest.self_sha256,
        "model_tree_sha256": manifest.tree_sha256,
    }


def _write_upstream(path: Path, manifest: SimpleNamespace) -> Path:
    implementation = leaf_qualification._implementation_closure()
    body: dict[str, object] = {
        "counters": leaf_qualification._zero_counters(),
        "fixture_commitments": dict(leaf_qualification.FIXTURE_COMMITMENTS),
        "fixture_count": len(leaf_qualification.PUBLIC_FIXTURES),
        "fixture_ordinals": list(range(len(leaf_qualification.PUBLIC_FIXTURES))),
        "fixture_suite_sha256": leaf_qualification.FIXTURE_SUITE_SHA256,
        "implementation_closure": implementation,
        "implementation_closure_sha256": qualification._safe_hash(
            implementation
        ),
        "manifest_commitments": _manifest_commitments(manifest),
        "outcome_counts": {
            "success": len(leaf_qualification.PUBLIC_FIXTURES),
            "typed_abstention": 0,
            "typed_error": 0,
        },
        "qualification_passed": True,
        "repeat_byte_exact": True,
        "repeat_count": leaf_qualification.REPEAT_COUNT,
        "runtime_commitment": _RUNTIME_COMMITMENT,
        "schema": leaf_qualification.AGGREGATE_RECEIPT_SCHEMA,
    }
    path.write_bytes(_canonical_bytes(_self_hashed(body)))
    path.chmod(0o600)
    return path


def _fake_resource_peaks(
    outcomes: list[Mapping[str, object]],
) -> dict[str, int]:
    summaries = [row["resource_summary"] for row in outcomes]
    return {
        "cuda_max_memory_allocated_bytes": 1,
        "cuda_max_memory_reserved_bytes": 1,
        "max_leaf_call_count": max(row["leaf_call_count"] for row in summaries),
        "max_projected_relation_count": max(
            row["projected_relation_count"] for row in summaries
        ),
        "max_reported_success_candidate_count": max(
            row["reported_success_candidate_count"] for row in summaries
        ),
        "max_reported_success_forward_batch_count": max(
            row["reported_success_forward_batch_count"] for row in summaries
        ),
        "max_root_lexical_token_count": max(
            row["root_lexical_token_count"] for row in summaries
        ),
        "max_segment_count": max(row["segment_count"] for row in summaries),
        "process_max_rss_kib": 1,
    }


def _install_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch, manifest: SimpleNamespace
) -> _Runtime:
    runtime = _Runtime()
    monkeypatch.setattr(
        qualification, "_verify_model_binding", lambda **_: None
    )
    monkeypatch.setattr(
        qualification, "_load_exact_runtime", lambda **_: runtime
    )
    monkeypatch.setattr(
        qualification,
        "_validate_exact_runtime",
        lambda *, runtime, manifest: _RUNTIME_COMMITMENT,
    )
    monkeypatch.setattr(qualification, "_reset_cuda_peaks", lambda: None)
    monkeypatch.setattr(
        qualification, "_resource_peaks", _fake_resource_peaks
    )
    monkeypatch.setattr(
        qualification,
        "_select_document",
        lambda *, runtime, story_text: (
            document_envelope.select_document_qualification_only(
                story_text, leaf_selector=_LeafSelector()
            )
        ),
    )
    return runtime


def _run_both_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    manifest = _manifest()
    upstream = _write_upstream(tmp_path / "upstream.safe.json", manifest)
    _install_runtime_fakes(monkeypatch, manifest)
    receipts: list[dict[str, object]] = []
    paths: list[Path] = []
    for index in (0, 1):
        root = tmp_path / f"shard{index}"
        receipt = qualification.run_fixed_document_envelope_qualification_shard(
            model_root=tmp_path / "model",
            manifest=manifest,  # type: ignore[arg-type]
            upstream_aggregate_receipt=upstream,
            output_root=root,
            shard_index=index,
            shard_count=2,
        )
        receipts.append(dict(receipt))
        paths.append(root / qualification.SHARD_OUTPUT_NAME)
    return receipts[0], receipts[1], paths[0], paths[1]


def test_public_boundary_has_no_caller_content_or_scorer_surface() -> None:
    signature = inspect.signature(
        qualification.run_fixed_document_envelope_qualification_shard
    )
    assert tuple(signature.parameters) == (
        "model_root",
        "manifest",
        "upstream_aggregate_receipt",
        "output_root",
        "shard_index",
        "shard_count",
    )
    assert not {
        "backend",
        "label",
        "parser",
        "prompt",
        "query",
        "scorer",
        "source",
        "story",
        "text",
    } & set(signature.parameters)


def test_frozen_fixture_hashes_plans_and_core_motifs_are_exact() -> None:
    assert qualification.FIXTURE_SUITE_SHA256 == (
        qualification.EXPECTED_FIXTURE_SUITE_SHA256
    )
    assert tuple(
        row.input_sha256 for row in qualification.PUBLIC_DOCUMENT_FIXTURES
    ) == qualification.EXPECTED_FIXTURE_INPUT_SHA256S
    expected_motifs = {
        0: (
            ("Aster", "supports", "Birch"),
            ("Birch", "precedes", "Cedar"),
        ),
        1: (
            ("North", "influences", "South"),
            ("South", "causes", "East"),
            ("East", "follows", "West"),
        ),
        2: (
            ("甲方", "guides", "乙方"),
            ("München", "supports", "東京"),
            ("東京", "reverses", "Theta"),
        ),
        3: (
            ("Alpha", "supports", "Beta"),
            ("Beta", "precedes", "Gamma"),
            ("Gamma", "causes", "Delta"),
            ("Delta", "follows", "Epsilon"),
            ("Epsilon", "supports", "Zeta"),
            ("Zeta", "precedes", "Eta"),
        ),
    }
    for fixture in qualification.PUBLIC_DOCUMENT_FIXTURES:
        raw = fixture.story_text.encode("utf-8")
        plans = document_envelope.plan_document_segments(fixture.story_text)
        assert tuple(row.lexical_token_count for row in plans) == (
            fixture.expected_segment_token_counts
        )
        eligible_motifs = []
        for plan in plans:
            if not plan.leaf_eligible:
                continue
            core = raw[plan.core_start_byte : plan.core_end_byte].decode(
                "utf-8"
            )
            eligible_motifs.append(tuple(_LEXICAL.findall(core)[:3]))
        assert tuple(eligible_motifs) == expected_motifs[fixture.ordinal]


def test_two_shards_execute_repeat_exact_and_aggregate_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shard0, shard1, path0, path1 = _run_both_shards(
        monkeypatch, tmp_path
    )
    assert shard0["fixture_ordinals"] == [3]
    assert shard1["fixture_ordinals"] == [0, 1, 2]
    for receipt in (shard0, shard1):
        qualification._validate_shard_receipt(receipt)
        assert receipt["qualification_passed"] is True
        assert receipt["repeat_byte_exact"] is True
        assert receipt["formal_effect_evidence"] is False
        assert receipt["downstream_eligible"] is False
        assert receipt["effect_quality_gate_added"] is False
        assert receipt["functional_extracted_branch_coverage_required"] is True
        assert receipt["counters"]["external_fixture_source_access_count"] == 0
        assert receipt["counters"]["external_evaluator_scorer_access_count"] == 0
        assert receipt["teacher_forced_canary"][
            "long_repeat_byte_exact"
        ] is True
        assert all(
            row["partial_projection_available"] is True
            and row["extracted_leaf_count"]
            == qualification.PUBLIC_DOCUMENT_FIXTURES[row["ordinal"]].expected_leaf_call_count
            for row in receipt["outcomes"]
        )
    aggregate_root = tmp_path / "aggregate"
    aggregate = qualification.aggregate_fixed_document_envelope_qualification(
        shard_receipts=(path0, path1), output_root=aggregate_root
    )
    assert aggregate["qualification_passed"] is True
    assert aggregate["outcome_status_counts"] == {
        "executed_without_typed_failure": 4,
        "functional_extracted_branch_not_exercised": 0,
        "not_executed_after_canary_failure": 0,
        "repeat_mismatch": 0,
        "typed_failure": 0,
    }
    assert aggregate["formal_effect_evidence"] is False
    assert aggregate["downstream_eligible"] is False
    assert (aggregate_root / qualification.AGGREGATE_OUTPUT_NAME).stat().st_mode & 0o777 == 0o600

    nested = json.dumps(
        [shard0, shard1, dict(aggregate)], ensure_ascii=False
    )
    assert "D1024Token0500" not in nested
    assert "Mixed2Token0100" not in nested


def test_relation_bearing_suite_returns_safe_negative_for_zero_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification,
        "_select_document",
        lambda *, runtime, story_text: (
            document_envelope.select_document_qualification_only(
                story_text, leaf_selector=_LeafSelector(no_relation=True)
            )
        ),
    )
    outcome = qualification._run_document_once(
        runtime=_Runtime(),
        runtime_commitment=_RUNTIME_COMMITMENT,
        fixture=qualification.PUBLIC_DOCUMENT_FIXTURES[0],
    )
    assert outcome["status"] == qualification._NO_RELATION_STATUS
    assert outcome["failure_code"] == (
        "DOCUMENT_FUNCTIONAL_EXTRACTED_BRANCH_NOT_EXERCISED"
    )


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_code"),
    (
        (
            "no_relation",
            qualification._NO_RELATION_STATUS,
            "DOCUMENT_FUNCTIONAL_EXTRACTED_BRANCH_NOT_EXERCISED",
        ),
        (
            "typed_failure",
            qualification._TYPED_FAILURE_STATUS,
            "DOCUMENT_TYPED_FAILURE_REPORTED",
        ),
        (
            "repeat_mismatch",
            qualification._REPEAT_MISMATCH_STATUS,
            "DOCUMENT_REPEAT_BYTE_MISMATCH",
        ),
        (
            "canary_failure",
            qualification._CANARY_NOT_EXECUTED_STATUS,
            "DOCUMENT_TEACHER_FORCED_CANARY_FAILED",
        ),
    ),
)
def test_legal_negative_qualification_publishes_content_free_shard_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    expected_status: str,
    expected_code: str,
) -> None:
    manifest = _manifest()
    upstream = _write_upstream(tmp_path / "upstream.safe.json", manifest)
    _install_runtime_fakes(monkeypatch, manifest)
    if mode in {"no_relation", "typed_failure"}:
        selector = _LeafSelector(
            no_relation=mode == "no_relation",
            typed_failure=mode == "typed_failure",
        )
        monkeypatch.setattr(
            qualification,
            "_select_document",
            lambda *, runtime, story_text: (
                document_envelope.select_document_qualification_only(
                    story_text, leaf_selector=selector
                )
            ),
        )
    elif mode == "repeat_mismatch":
        original = qualification._run_document_once
        calls = 0

        def alternating(**kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            row = original(**kwargs)  # type: ignore[arg-type]
            if calls % 2 == 0:
                row["projection_commitment"] = "f" * 64
            return row

        monkeypatch.setattr(qualification, "_run_document_once", alternating)
    else:
        def fail_canary(_runtime: object) -> Mapping[str, object]:
            raise RuntimeError("private canary failure must not enter receipt")

        monkeypatch.setattr(
            leaf_qualification,
            "_run_fixed_teacher_forced_canary",
            fail_canary,
        )
    root = tmp_path / "negative-shard"
    receipt = dict(
        qualification.run_fixed_document_envelope_qualification_shard(
            model_root=tmp_path / "model",
            manifest=manifest,  # type: ignore[arg-type]
            upstream_aggregate_receipt=upstream,
            output_root=root,
            shard_index=0,
            shard_count=2,
        )
    )
    terminal = root / qualification.SHARD_OUTPUT_NAME
    assert terminal.is_file()
    assert receipt["schema"] == qualification.SHARD_RECEIPT_SCHEMA
    assert receipt["qualification_passed"] is False
    assert receipt["qualification_failure_codes"] == [expected_code]
    assert receipt["outcomes"][0]["status"] == expected_status
    assert receipt["outcomes"][0]["failure_code"] == expected_code
    qualification._validate_shard_receipt(receipt)
    serialized = terminal.read_text(encoding="ascii")
    assert "private canary failure" not in serialized
    assert "private runtime failure" not in serialized
    assert "D1024Token0500" not in serialized


def test_resource_peak_failure_publishes_content_free_negative_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    upstream = _write_upstream(tmp_path / "upstream.safe.json", manifest)
    _install_runtime_fakes(monkeypatch, manifest)

    def fail_resource_peaks(
        _outcomes: object,
    ) -> dict[str, int]:
        raise RuntimeError("private resource failure must not enter receipt")

    monkeypatch.setattr(qualification, "_resource_peaks", fail_resource_peaks)
    root = tmp_path / "resource-negative-shard"
    receipt = dict(
        qualification.run_fixed_document_envelope_qualification_shard(
            model_root=tmp_path / "model",
            manifest=manifest,  # type: ignore[arg-type]
            upstream_aggregate_receipt=upstream,
            output_root=root,
            shard_index=0,
            shard_count=2,
        )
    )
    terminal = root / qualification.SHARD_OUTPUT_NAME
    assert terminal.is_file()
    assert receipt["qualification_passed"] is False
    assert receipt["qualification_failure_codes"] == [
        qualification._RESOURCE_PEAK_FAILURE_CODE
    ]
    assert receipt["resource_peaks"] == {
        **{
            key: 0
            for key in qualification._RESOURCE_PEAK_FIELDS
            if key != "process_max_rss_kib"
        },
        "process_max_rss_kib": receipt["resource_peaks"][
            "process_max_rss_kib"
        ],
    }
    assert receipt["resource_peaks"]["process_max_rss_kib"] > 0
    qualification._validate_shard_receipt(receipt)
    assert "private resource failure" not in terminal.read_text(
        encoding="ascii"
    )


def test_aggregate_accepts_positive_and_negative_shards_and_publishes_negative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    upstream = _write_upstream(tmp_path / "upstream.safe.json", manifest)
    _install_runtime_fakes(monkeypatch, manifest)
    monkeypatch.setattr(
        qualification,
        "_select_document",
        lambda *, runtime, story_text: (
            document_envelope.select_document_qualification_only(
                story_text, leaf_selector=_LeafSelector(no_relation=True)
            )
        ),
    )
    root0 = tmp_path / "shard0"
    qualification.run_fixed_document_envelope_qualification_shard(
        model_root=tmp_path / "model",
        manifest=manifest,  # type: ignore[arg-type]
        upstream_aggregate_receipt=upstream,
        output_root=root0,
        shard_index=0,
        shard_count=2,
    )
    monkeypatch.setattr(
        qualification,
        "_select_document",
        lambda *, runtime, story_text: (
            document_envelope.select_document_qualification_only(
                story_text, leaf_selector=_LeafSelector()
            )
        ),
    )
    root1 = tmp_path / "shard1"
    qualification.run_fixed_document_envelope_qualification_shard(
        model_root=tmp_path / "model",
        manifest=manifest,  # type: ignore[arg-type]
        upstream_aggregate_receipt=upstream,
        output_root=root1,
        shard_index=1,
        shard_count=2,
    )
    aggregate_root = tmp_path / "aggregate"
    aggregate = dict(
        qualification.aggregate_fixed_document_envelope_qualification(
            shard_receipts=(
                root0 / qualification.SHARD_OUTPUT_NAME,
                root1 / qualification.SHARD_OUTPUT_NAME,
            ),
            output_root=aggregate_root,
        )
    )
    assert aggregate["qualification_passed"] is False
    assert aggregate["qualification_failure_codes"] == [
        "DOCUMENT_FUNCTIONAL_EXTRACTED_BRANCH_NOT_EXERCISED"
    ]
    assert aggregate["outcome_status_counts"] == {
        "executed_without_typed_failure": 3,
        "functional_extracted_branch_not_exercised": 1,
        "not_executed_after_canary_failure": 0,
        "repeat_mismatch": 0,
        "typed_failure": 0,
    }
    assert (aggregate_root / qualification.AGGREGATE_OUTPUT_NAME).is_file()


def test_qualification_fake_runtime_is_not_exact_authority() -> None:
    manifest = _manifest()
    with pytest.raises(
        qualification.FixedDocumentEnvelopeQualificationError,
        match="exact_runtime_authority_invalid",
    ):
        qualification._validate_exact_runtime(
            runtime=_Runtime(), manifest=manifest  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_root",
        "version",
        "partial_projection",
        "topology",
        "resource_bool",
        "resource_peak_string",
    ),
)
def test_rehashed_shard_forgery_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    shard0, _shard1, _path0, _path1 = _run_both_shards(
        monkeypatch, tmp_path
    )
    forged = deepcopy(shard0)
    if mutation == "unknown_root":
        forged["forged"] = True
    elif mutation == "version":
        forged["version"] = "forged"
    elif mutation == "partial_projection":
        forged["outcomes"][0]["partial_projection_available"] = False
    elif mutation == "topology":
        forged["outcomes"][0]["segment_topology"][0][
            "chunk_index"
        ] = 1
        forged["outcomes"][0]["segment_topology_sha256"] = (
            qualification._safe_hash(
                forged["outcomes"][0]["segment_topology"]
            )
        )
    elif mutation == "resource_bool":
        forged["outcomes"][0]["resource_summary"][
            "root_lexical_token_count"
        ] = True
    else:
        forged["resource_peaks"]["max_segment_count"] = "6"
    forged["outcomes_commitment"] = qualification._safe_hash(
        forged["outcomes"]
    )
    body = {key: value for key, value in forged.items() if key != "self_sha256"}
    forged["self_sha256"] = qualification._safe_hash(body)
    with pytest.raises(
        qualification.FixedDocumentEnvelopeQualificationError
    ):
        qualification._validate_shard_receipt(forged)


def test_aggregate_rejects_cross_shard_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _shard0, shard1, path0, _path1 = _run_both_shards(
        monkeypatch, tmp_path
    )
    forged = deepcopy(shard1)
    forged["runtime_commitment"] = "f" * 64
    body = {key: value for key, value in forged.items() if key != "self_sha256"}
    forged["self_sha256"] = qualification._safe_hash(body)
    forged_path = tmp_path / "forged-shard1.safe.json"
    forged_path.write_bytes(_canonical_bytes(forged))
    forged_path.chmod(0o600)
    with pytest.raises(
        qualification.FixedDocumentEnvelopeQualificationError,
        match="binding_mismatch",
    ):
        qualification.aggregate_fixed_document_envelope_qualification(
            shard_receipts=(path0, forged_path),
            output_root=tmp_path / "forged-aggregate",
        )


def test_stale_output_rejected_before_runtime_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / qualification.SHARD_OUTPUT_NAME).write_text("stale")
    called = False

    def load(**_: object) -> object:
        nonlocal called
        called = True
        return _Runtime()

    monkeypatch.setattr(qualification, "_load_exact_runtime", load)
    with pytest.raises(
        qualification.FixedDocumentEnvelopeQualificationError,
        match="output_not_fresh",
    ):
        qualification.run_fixed_document_envelope_qualification_shard(
            model_root=tmp_path / "model",
            manifest=_manifest(),  # type: ignore[arg-type]
            upstream_aggregate_receipt=tmp_path / "missing",
            output_root=output,
            shard_index=0,
            shard_count=2,
        )
    assert called is False


def test_units_are_manual_only_bounded_and_use_physical_gpu_placeholders() -> None:
    manifest_root = Path(qualification.__file__).parents[2] / "manifests"
    for name in (
        "gscl_document_envelope_fixed_qualification_shard0.service",
        "gscl_document_envelope_fixed_qualification_shard1.service",
        "gscl_document_envelope_fixed_qualification_aggregate.service",
    ):
        text = (manifest_root / name).read_text(encoding="utf-8")
        assert "Restart=no" in text
        assert "ExecStartPre=" not in text
        assert "[Install]" not in text
        assert "Requires=" not in text
        assert "Wants=" not in text
        assert "IPAddressDeny=any" in text
    shard0 = (
        manifest_root
        / "gscl_document_envelope_fixed_qualification_shard0.service"
    ).read_text(encoding="utf-8")
    shard1 = (
        manifest_root
        / "gscl_document_envelope_fixed_qualification_shard1.service"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=@GPU0_UUID@" in shard0
    assert "CUDA_VISIBLE_DEVICES=@GPU1_UUID@" in shard1
    for text in (shard0, shard1):
        assert "OMP_NUM_THREADS=1" in text
        assert "OPENBLAS_NUM_THREADS=1" in text
        assert "UnsetEnvironment=OPENAI_API_KEY" in text

from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping

import pytest
import torch

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
    fixed_public_qualification as qualification,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    memory_safe_qwen,
)
from replication_runtime.gscl_narrative_extractor_v2.contract import (
    ClosedChoiceV2Abstention,
)


_FAKE_RUNTIME_COMMITMENT = hashlib.sha256(
    b"gscl-v2-fixed-public-qualification-sealed-fake"
).hexdigest()
_CANARY_SHORT_MICROUNITS = -10_001
_CANARY_SHORT_SHA256 = contract_v1.semantic_sha256(
    _CANARY_SHORT_MICROUNITS
)


def _canonical_bytes(value: object) -> bytes:
    if isinstance(value, Mapping):
        value = dict(value)
    return contract_v1.canonical_json_bytes(value)


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    return {
        **body,
        "self_sha256": contract_v1.semantic_sha256(dict(body)),
    }


def _assert_self_hash(value: Mapping[str, object]) -> None:
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    assert value["self_sha256"] == contract_v1.semantic_sha256(body)


def _synthetic_declarations() -> dict[str, object]:
    digest = hashlib.sha256
    return {
        "attention_implementation": "sdpa",
        "chat_template_sha256": digest(b"fixed chat").hexdigest(),
        "context_limit": 32_768,
        "critical_config": dict(worker.QWEN_ARCHITECTURE),
        "loaded_config_sha256": digest(b"fixed config").hexdigest(),
        "model_class": "Qwen2ForCausalLM",
        "special_token_ids": {
            "bos_token_id": None,
            "eos_token_id": 151_645,
            "pad_token_id": 151_643,
        },
        "tokenizer_class": "Qwen2TokenizerFast",
    }


def _synthetic_runtime_requirements() -> dict[str, object]:
    digest = hashlib.sha256
    return {
        "attention_implementation": "sdpa",
        "cuda_version": "12.synthetic",
        "cudnn_version": 90_100,
        "gpu_compute_capability": [8, 6],
        "gpu_name": "Synthetic qualification GPU",
        "python_executable_sha256": digest(
            b"fixed interpreter"
        ).hexdigest(),
        "python_implementation": "CPython",
        "python_version": "3.synthetic",
        "torch_version": "2.synthetic",
        "torch_distribution_sha256": digest(
            b"fixed torch"
        ).hexdigest(),
        "transformers_version": "4.synthetic",
        "transformers_distribution_sha256": digest(
            b"fixed transformers"
        ).hexdigest(),
    }


def _verified_manifest(
    root: Path,
) -> tuple[Path, Path, worker.ModelAssetManifest]:
    model_root = root / "model"
    model_root.mkdir(mode=0o700, parents=True)
    model_file = model_root / "synthetic.safetensors"
    model_raw = b"fixed source-free synthetic model asset\n"
    model_file.write_bytes(model_raw)
    model_file.chmod(0o600)
    files = (
        {
            "path": model_file.name,
            "sha256": hashlib.sha256(model_raw).hexdigest(),
            "size": len(model_raw),
        },
    )
    tree_sha256 = contract_v1.semantic_sha256(list(files))
    manifest = worker.ModelAssetManifest(
        declarations=_synthetic_declarations(),
        files=files,
        runtime_requirements=_synthetic_runtime_requirements(),
        tree_sha256=tree_sha256,
        self_sha256=hashlib.sha256(
            b"fixed synthetic verified manifest self"
        ).hexdigest(),
        manifest_file_sha256=hashlib.sha256(
            b"fixed synthetic verified manifest file"
        ).hexdigest(),
        _marker=worker._VERIFIED_MANIFEST_MARKER,
    )
    manifest_path = root / "model.manifest.json"
    manifest_path.write_bytes(
        _canonical_bytes(
            {
                "manifest_file_sha256": (
                    manifest.manifest_file_sha256
                ),
                "self_sha256": manifest.self_sha256,
                "tree_sha256": manifest.tree_sha256,
            }
        )
    )
    manifest_path.chmod(0o600)
    return model_root, manifest_path, manifest


class _DeterministicBackend:
    """Finite fake scorer that prefers ONE and first-listed alternatives."""

    @property
    def runtime_commitment(self) -> str:
        return _FAKE_RUNTIME_COMMITMENT

    def score_batch(
        self, pairs: tuple[closed_v1.PromptAnswer, ...]
    ) -> tuple[closed_v2.TeacherForcedScore, ...]:
        rows: list[closed_v2.TeacherForcedScore] = []
        for pair in pairs:
            preferred = int(
                pair.candidate_key.endswith(".plan.one_relation")
            )
            answer_tokens = max(1, len(pair.answer.split()))
            rows.append(
                closed_v2.TeacherForcedScore(
                    total_logprob_microunits=(
                        preferred * 1_000_000 * answer_tokens
                    ),
                    answer_token_count=answer_tokens,
                    context_and_answer_token_count=(
                        answer_tokens + 64
                    ),
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
            (
                "fixed-public."
                + hashlib.sha256(story.encode("utf-8")).hexdigest()[:24]
            ),
            story,
        ),
        completion,
    )


def _fixed_canary_receipt(
    *,
    strategy: str = memory_safe_qwen.SPARSE_STRATEGY,
) -> Mapping[str, object]:
    body: dict[str, object] = {
        "fallback_independent_full_reference_passed": True,
        "free_form_generation_count": 0,
        "long_answer_position_count": 200,
        "long_pair_sha256": (
            memory_safe_qwen.FIXED_LONG_CANARY_PAIR_SHA256
        ),
        "long_repeat_byte_exact": True,
        "long_score_sha256": hashlib.sha256(
            b"fixed-long-answer-score"
        ).hexdigest(),
        "schema": memory_safe_qwen.FIXED_CANARY_SCHEMA,
        "short_full_reference_microunits": (
            _CANARY_SHORT_MICROUNITS
        ),
        "short_pair_sha256": (
            memory_safe_qwen.FIXED_SHORT_CANARY_PAIR_SHA256
        ),
        "short_strategy_microunits": (
            _CANARY_SHORT_MICROUNITS
        ),
        "short_strategy_vs_full_reference_exact": True,
        "sparse_chunk_count": 2,
        "strategy": strategy,
    }
    return MappingProxyType(_self_hashed(body))


class _SealedFakeRuntime:
    """Loader-only fake; it accepts no caller-controlled scorer or fixture."""

    __slots__ = (
        "_abstain_commitments",
        "_canary",
        "_calls",
        "_drift_on_repeat",
    )

    def __init__(
        self,
        *,
        abstain_commitments: frozenset[str] = frozenset(),
        drift_on_repeat: bool = False,
    ) -> None:
        self._abstain_commitments = abstain_commitments
        self._drift_on_repeat = drift_on_repeat
        self._calls: Counter[str] = Counter()
        self._canary = _fixed_canary_receipt()

    @property
    def runtime_commitment(self) -> str:
        return _FAKE_RUNTIME_COMMITMENT

    @property
    def calls(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._calls))

    def run_fixed_teacher_forced_canary(
        self,
    ) -> Mapping[str, object]:
        self._calls["canary"] += 1
        return self._canary

    def select_story(
        self, story_text: str
    ) -> closed_v2.ClosedChoiceV2Decision:
        commitment = hashlib.sha256(
            story_text.encode("utf-8")
        ).hexdigest()
        self._calls[commitment] += 1
        if commitment in self._abstain_commitments:
            raise ClosedChoiceV2Abstention(
                "V2_PLAN_NO_RELATION_SELECTED",
                before_model_forward=False,
            )
        decision = closed_v2.select_hierarchical_qualification_only(
            story_text,
            backend=_DeterministicBackend(),
            narrative_parser=_parser,
        )
        if (
            self._drift_on_repeat
            and self._calls[commitment] % 2 == 0
        ):
            # The runner must compare bytes, not merely disposition names.
            body = {
                key: value
                for key, value in decision.receipt.items()
                if key != "self_sha256"
            }
            body["qualification_fake_repeat_drift"] = True
            drifted = _self_hashed(body)
            return closed_v2.ClosedChoiceV2Decision(
                wire_completion=decision.wire_completion,
                canonical_completion=decision.canonical_completion,
                extraction=decision.extraction,
                selected_answer_token_count=(
                    decision.selected_answer_token_count
                ),
                wire_completion_token_count=(
                    decision.wire_completion_token_count
                ),
                receipt_bytes=_canonical_bytes(drifted),
            )
        return decision


def _fixture_story(fixture: object) -> str:
    story = getattr(fixture, "story_text")
    assert type(story) is str
    return story


def _fixture_ordinal(fixture: object) -> int:
    ordinal = getattr(fixture, "ordinal")
    assert type(ordinal) is int
    return ordinal


def _fixture_id(fixture: object) -> str:
    fixture_id = getattr(fixture, "fixture_id")
    assert type(fixture_id) is str
    return fixture_id


def _fixture_tags(fixture: object) -> tuple[str, ...]:
    tags = getattr(fixture, "feature_flags")
    assert type(tags) is tuple
    assert all(type(tag) is str for tag in tags)
    return tags


def _run_shard(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    shard_index: int,
    runtime: _SealedFakeRuntime | None = None,
) -> tuple[
    Mapping[str, object],
    bytes,
    Path,
    _SealedFakeRuntime,
    worker.ModelAssetManifest,
]:
    root.mkdir(mode=0o700)
    model_root, _manifest_path, manifest = _verified_manifest(root)
    output_root = root / "output"
    output_root.mkdir(mode=0o700)
    selected = runtime or _SealedFakeRuntime()

    def load_exact_runtime(
        *,
        model_root: Path,
        manifest: worker.ModelAssetManifest,
    ) -> _SealedFakeRuntime:
        assert model_root == model_root_value
        assert manifest is manifest_value
        return selected

    model_root_value = model_root
    manifest_value = manifest
    monkeypatch.setattr(
        qualification,
        "_load_exact_runtime",
        load_exact_runtime,
    )
    monkeypatch.setattr(
        worker,
        "_scan_model_tree",
        lambda observed_root: (
            manifest.files
            if observed_root == model_root
            else ()
        ),
    )
    receipt = qualification.run_fixed_public_qualification(
        model_root=model_root,
        manifest=manifest,
        output_root=output_root,
        shard_index=shard_index,
        shard_count=2,
    )
    receipt_path = output_root / "qualification.safe.json"
    raw = receipt_path.read_bytes()
    assert json.loads(raw.decode("ascii")) == receipt
    return receipt, raw, receipt_path, selected, manifest


def _all_nested_strings(value: object) -> tuple[str, ...]:
    rows: list[str] = []
    if isinstance(value, str):
        rows.append(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            rows.extend(_all_nested_strings(key))
            rows.extend(_all_nested_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            rows.extend(_all_nested_strings(child))
    return tuple(rows)


def test_programmatic_boundary_is_fixed_two_shard_and_has_no_content_input(
    tmp_path: Path,
) -> None:
    signature = inspect.signature(
        qualification.run_fixed_public_qualification
    )
    assert tuple(signature.parameters) == (
        "model_root",
        "manifest",
        "output_root",
        "shard_index",
        "shard_count",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    _model_root, _manifest_path, manifest = _verified_manifest(
        tmp_path
    )
    with pytest.raises(TypeError):
        qualification.run_fixed_public_qualification(  # type: ignore[misc]
            tmp_path,
            manifest,
            tmp_path / "out",
            0,
            2,
        )
    with pytest.raises(TypeError):
        qualification.run_fixed_public_qualification(
            model_root=tmp_path,
            manifest=manifest,
            output_root=tmp_path / "out",
            shard_index=0,
            shard_count=2,
            story_text="caller supplied content",  # type: ignore[call-arg]
        )
    forbidden = {
        "answer",
        "backend",
        "fixture",
        "prompt",
        "query",
        "source",
        "story",
        "text",
    }
    assert not forbidden & set(signature.parameters)


def test_public_fixture_suite_has_exact_fixed_coverage() -> None:
    fixtures = qualification.PUBLIC_FIXTURES
    assert type(fixtures) is tuple
    assert len(fixtures) == 5
    assert tuple(map(_fixture_ordinal, fixtures)) == tuple(range(5))
    assert len({_fixture_id(row) for row in fixtures}) == len(fixtures)

    observed_lengths: list[int] = []
    observed_sentences: list[int] = []
    tags: set[str] = set()
    stories: list[str] = []
    for fixture in fixtures:
        story = _fixture_story(fixture)
        stories.append(story)
        episodes = closed_v2.build_hierarchical_episodes(story)
        observed_lengths.append(
            sum(len(episode.atoms) for episode in episodes)
        )
        observed_sentences.append(
            len({episode.sentence_id for episode in episodes})
        )
        tags.update(_fixture_tags(fixture))
    assert observed_lengths == [17, 33, 64, 128, 175]
    assert {1, 2, 5} <= set(observed_sentences)
    assert {"multiword", "repeated_phrase", "unicode"} <= tags
    assert any(
        len(tokens) != len(set(tokens))
        for tokens in (
            [
                atom.quote.casefold()
                for episode in closed_v2.build_hierarchical_episodes(
                    story
                )
                for atom in episode.atoms
            ]
            for story in stories
        )
    )
    assert any(not story.isascii() for story in stories)
    assert qualification.FIXTURE_SUITE_SHA256 == (
        contract_v1.semantic_sha256(
            [
                {
                    "feature_flags": list(_fixture_tags(fixture)),
                    "fixture_commitment": (
                        fixture.fixture_commitment
                    ),
                    "fixture_id": _fixture_id(fixture),
                    "input_sha256": fixture.input_sha256,
                    "lexical_token_count": (
                        fixture.lexical_token_count
                    ),
                    "ordinal": _fixture_ordinal(fixture),
                    "sentence_count": fixture.sentence_count,
                }
                for fixture in fixtures
            ]
        )
    )


@pytest.mark.parametrize("shard_index", [0, 1])
def test_shard_runs_only_ordinal_mod_two_and_emits_safe_aggregate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shard_index: int,
) -> None:
    receipt, raw, receipt_path, runtime, manifest = _run_shard(
        monkeypatch,
        tmp_path / f"shard-{shard_index}",
        shard_index=shard_index,
    )
    expected_ordinals = [
        ordinal
        for ordinal in range(len(qualification.PUBLIC_FIXTURES))
        if ordinal % 2 == shard_index
    ]
    expected_commitments = {
        _fixture_id(qualification.PUBLIC_FIXTURES[ordinal]): (
            qualification.PUBLIC_FIXTURES[
                ordinal
            ].fixture_commitment
        )
        for ordinal in expected_ordinals
    }
    expected_input_sha256 = {
        qualification.PUBLIC_FIXTURES[ordinal].input_sha256
        for ordinal in expected_ordinals
    }
    assert receipt["shard_index"] == shard_index
    assert receipt["shard_count"] == 2
    assert receipt["fixture_count"] == len(expected_ordinals)
    assert receipt["fixture_ordinals"] == expected_ordinals
    assert receipt["fixture_commitments"] == expected_commitments
    assert receipt["fixture_suite_sha256"] == (
        qualification.FIXTURE_SUITE_SHA256
    )
    assert runtime.calls["canary"] == 1
    assert all(
        runtime.calls[commitment] == 2
        for commitment in expected_input_sha256
    )
    assert all(
        commitment not in runtime.calls
        for commitment in {
            hashlib.sha256(
                _fixture_story(fixture).encode("utf-8")
            ).hexdigest()
            for fixture in qualification.PUBLIC_FIXTURES
            if _fixture_ordinal(fixture) not in expected_ordinals
        }
    )
    assert receipt["repeat_count"] == 2
    assert receipt["repeat_byte_exact"] is True
    assert receipt["outcome_counts"] == {
        "success": len(expected_ordinals),
        "typed_abstention": 0,
        "typed_error": 0,
    }
    assert len(receipt["outcomes"]) == len(expected_ordinals)
    assert all(
        row["disposition"] == "success"
        and row["repeat_count"] == 2
        and row["repeat_byte_exact"] is True
        and set(row)
        == {
            "canonical_completion_sha256",
            "decision_receipt_sha256",
            "disposition",
            "extraction_semantic_hash",
            "fixture_commitment",
            "fixture_id",
            "generator_count",
            "input_sha256",
            "mention_count",
            "ordinal",
            "repeat_byte_exact",
            "repeat_count",
            "repeat_outcome_sha256",
            "resource_summary",
            "selected_answer_token_count",
            "wire_completion_sha256",
            "wire_completion_token_count",
        }
        for row in receipt["outcomes"]
    )
    assert receipt["outcomes_commitment"] == (
        contract_v1.semantic_sha256(receipt["outcomes"])
    )
    resource = receipt["resource_peaks"]
    assert set(resource) == {
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
    assert 0 < resource["max_candidate_count"] <= (
        closed_v2.MAXIMUM_TOTAL_CANDIDATES
    )
    assert 0 < resource["max_forward_batch_count"] <= (
        closed_v2.MAXIMUM_FORWARD_BATCH_CALLS
    )
    assert 1 <= resource["max_span_lexical_width"] <= 4
    canary = receipt["teacher_forced_canary"]
    _assert_self_hash(canary)
    assert receipt["teacher_forced_canary_self_sha256"] == (
        canary["self_sha256"]
    )
    assert canary["short_strategy_vs_full_reference_exact"] is True
    assert (
        canary["short_strategy_microunits"]
        == canary["short_full_reference_microunits"]
    )
    assert canary["long_answer_position_count"] > 128
    assert 2 <= canary["sparse_chunk_count"]
    assert canary["long_repeat_byte_exact"] is True
    assert (
        canary["fallback_independent_full_reference_passed"] is True
    )
    assert receipt["runtime_commitment"] == (
        _FAKE_RUNTIME_COMMITMENT
    )
    assert receipt["manifest_commitments"][
        "manifest_file_sha256"
    ] == manifest.manifest_file_sha256
    assert receipt["manifest_commitments"]["model_tree_sha256"] == (
        manifest.tree_sha256
    )
    assert receipt["implementation_closure_sha256"] == (
        contract_v1.semantic_sha256(
            receipt["implementation_closure"]
        )
    )
    assert {
        "assumption_agent_init.py",
        "assumption_agent_models.py",
        "narrative_correspondence_parser.py",
        "v1_closed_choice_worker.py",
        "v1_contract.py",
        "v1_worker.py",
        "v2_closed_choice.py",
        "v2_contract.py",
        "v2_memory_safe_qwen.py",
    }.issubset(receipt["implementation_closure"])
    assert receipt["counters"] == {
        "api_access_count": 0,
        "free_form_generation_count": 0,
        "label_access_count": 0,
        "network_access_count": 0,
        "source_access_count": 0,
    }
    assert receipt["qualification_passed"] is True
    _assert_self_hash(receipt)
    assert raw == _canonical_bytes(receipt)
    assert (receipt_path.stat().st_mode & 0o777) == 0o600

    nested = _all_nested_strings(receipt)
    for fixture in qualification.PUBLIC_FIXTURES:
        story = _fixture_story(fixture)
        assert story not in nested
        for episode in closed_v2.build_hierarchical_episodes(story):
            for atom in episode.atoms:
                assert atom.quote not in nested


def test_typed_abstention_is_preserved_without_content_or_runtime_collapse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_fixture = qualification.PUBLIC_FIXTURES[0]
    commitment = hashlib.sha256(
        _fixture_story(selected_fixture).encode("utf-8")
    ).hexdigest()
    fixture_commitment = selected_fixture.fixture_commitment
    runtime = _SealedFakeRuntime(
        abstain_commitments=frozenset({commitment})
    )
    receipt, _raw, _path, _runtime, _manifest = _run_shard(
        monkeypatch,
        tmp_path / "typed-abstention",
        shard_index=0,
        runtime=runtime,
    )
    assert receipt["outcome_counts"] == {
        "success": receipt["fixture_count"] - 1,
        "typed_abstention": 1,
        "typed_error": 0,
    }
    row = next(
        outcome
        for outcome in receipt["outcomes"]
        if outcome["fixture_commitment"] == fixture_commitment
    )
    assert row["disposition"] == "typed_abstention"
    assert row["error_category"] == "selection"
    assert row["error_code"] == "V2_PLAN_NO_RELATION_SELECTED"
    assert row["pre_model_abstention"] is False
    assert receipt["qualification_passed"] is False
    assert "MODEL" not in json.dumps(row)


def test_repeat_comparison_is_byte_exact_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _SealedFakeRuntime(drift_on_repeat=True)
    root = tmp_path / "repeat-drift"
    root.mkdir(mode=0o700)
    model_root, _manifest_path, manifest = _verified_manifest(root)
    output_root = root / "output"
    output_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        qualification,
        "_load_exact_runtime",
        lambda *, model_root, manifest: runtime,
    )
    monkeypatch.setattr(
        worker,
        "_scan_model_tree",
        lambda observed_root: (
            manifest.files
            if observed_root == model_root
            else ()
        ),
    )
    with pytest.raises(RuntimeError, match="repeat"):
        qualification.run_fixed_public_qualification(
            model_root=model_root,
            manifest=manifest,
            output_root=output_root,
            shard_index=0,
            shard_count=2,
        )
    assert not (output_root / "qualification.safe.json").exists()


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    [(-1, 2), (2, 2), (0, 1), (0, 3), (True, 2), (0, True)],
)
def test_only_exact_two_shard_partition_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shard_index: int,
    shard_count: int,
) -> None:
    model_root, _manifest_path, manifest = _verified_manifest(
        tmp_path
    )
    output_root = tmp_path / "output"
    output_root.mkdir(mode=0o700)
    called = False

    def forbidden_loader(**_: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("invalid shard reached loader")

    monkeypatch.setattr(
        qualification, "_load_exact_runtime", forbidden_loader
    )
    with pytest.raises(RuntimeError, match="shard"):
        qualification.run_fixed_public_qualification(
            model_root=model_root,
            manifest=manifest,
            output_root=output_root,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    assert called is False


def test_boundary_requires_exact_verified_manifest_before_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(mode=0o700)
    called = False

    def forbidden_loader(**_: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("unverified manifest reached loader")

    monkeypatch.setattr(
        qualification, "_load_exact_runtime", forbidden_loader
    )
    with pytest.raises(RuntimeError, match="authority"):
        qualification.run_fixed_public_qualification(
            model_root=tmp_path,
            manifest=object(),  # type: ignore[arg-type]
            output_root=output_root,
            shard_index=0,
            shard_count=2,
        )
    assert called is False


def test_shard_semantic_projection_is_exact_across_output_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, first_raw, _path, _runtime, _manifest = _run_shard(
        monkeypatch,
        tmp_path / "first",
        shard_index=1,
    )
    second, second_raw, _path, _runtime, _manifest = _run_shard(
        monkeypatch,
        tmp_path / "second",
        shard_index=1,
    )
    # Host/GPU peak telemetry is intentionally observational.  Every
    # semantic decision and fixed binding remains byte-identical.
    volatile = {"resource_peaks", "self_sha256"}
    assert {
        key: value
        for key, value in first.items()
        if key not in volatile
    } == {
        key: value
        for key, value in second.items()
        if key not in volatile
    }
    assert first_raw
    assert second_raw


def test_cli_has_only_fixed_inputs_and_rejects_dynamic_content_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = inspect.getsource(qualification._parser)
    for flag in (
        "--model-root",
        "--model-manifest",
        "--output-root",
        "--shard-index",
        "--shard-count",
    ):
        assert flag in source
    for forbidden in (
        "--answer",
        "--backend",
        "--fixture",
        "--prompt",
        "--query",
        "--source",
        "--story",
        "--text",
    ):
        assert forbidden not in source
        with pytest.raises(SystemExit) as error:
            qualification.main([forbidden, "dynamic"])
        assert error.value.code == 2
        capsys.readouterr()


def test_fixed_canary_seam_calls_only_no_argument_runtime_method() -> None:
    class CanaryRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def run_fixed_teacher_forced_canary(
            self,
        ) -> Mapping[str, object]:
            self.calls += 1
            return _fixed_canary_receipt()

    runtime = CanaryRuntime()
    receipt = qualification._run_fixed_teacher_forced_canary(runtime)
    assert receipt == _fixed_canary_receipt()
    assert runtime.calls == 1
    assert tuple(
        inspect.signature(
            CanaryRuntime.run_fixed_teacher_forced_canary
        ).parameters
    ) == ("self",)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("short_strategy_vs_full_reference_exact", False),
        ("long_answer_position_count", 128),
        ("sparse_chunk_count", 1),
        ("long_repeat_byte_exact", False),
        ("fallback_independent_full_reference_passed", False),
        ("free_form_generation_count", 1),
    ],
)
def test_fixed_canary_seam_rejects_semantically_incomplete_receipt(
    field: str,
    replacement: object,
) -> None:
    body = {
        key: value
        for key, value in _fixed_canary_receipt().items()
        if key != "self_sha256"
    }
    body[field] = replacement
    invalid = _self_hashed(body)

    class InvalidRuntime:
        def run_fixed_teacher_forced_canary(
            self,
        ) -> Mapping[str, object]:
            return invalid

    with pytest.raises(RuntimeError, match="canary"):
        qualification._run_fixed_teacher_forced_canary(
            InvalidRuntime()
        )


def test_aggregate_boundary_is_exactly_two_receipts_and_output_root() -> None:
    signature = inspect.signature(
        qualification.aggregate_fixed_public_qualification
    )
    assert tuple(signature.parameters) == (
        "shard_receipts",
        "output_root",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    with pytest.raises(TypeError):
        qualification.aggregate_fixed_public_qualification(  # type: ignore[misc]
            (Path("zero"), Path("one")),
            Path("output"),
        )
    with pytest.raises(TypeError):
        qualification.aggregate_fixed_public_qualification(
            shard_receipts=(Path("zero"), Path("one")),
            output_root=Path("output"),
            story_text="forbidden",  # type: ignore[call-arg]
        )


def test_pure_offline_aggregate_requires_exact_union_and_merges_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shard_zero, _raw, path_zero, _runtime, manifest_zero = _run_shard(
        monkeypatch,
        tmp_path / "zero",
        shard_index=0,
    )
    shard_one, _raw, path_one, _runtime, manifest_one = _run_shard(
        monkeypatch,
        tmp_path / "one",
        shard_index=1,
    )
    assert manifest_zero.manifest_file_sha256 == (
        manifest_one.manifest_file_sha256
    )
    output_root = tmp_path / "aggregate"
    output_root.mkdir(mode=0o700)
    aggregate = qualification.aggregate_fixed_public_qualification(
        shard_receipts=(path_one, path_zero),
        output_root=output_root,
    )
    output_path = (
        output_root / "qualification.aggregate.safe.json"
    )
    raw = output_path.read_bytes()
    assert json.loads(raw.decode("ascii")) == aggregate
    assert aggregate["shard_count"] == 2
    assert aggregate["fixture_count"] == len(
        qualification.PUBLIC_FIXTURES
    )
    assert aggregate["fixture_ordinals"] == list(
        range(len(qualification.PUBLIC_FIXTURES))
    )
    assert len(set(aggregate["fixture_commitments"].values())) == len(
        qualification.PUBLIC_FIXTURES
    )
    assert aggregate["outcome_counts"] == {
        "success": len(qualification.PUBLIC_FIXTURES),
        "typed_abstention": 0,
        "typed_error": 0,
    }
    assert aggregate["outcomes_commitment"] == (
        contract_v1.semantic_sha256(
            sorted(
                [
                    *shard_zero["outcomes"],
                    *shard_one["outcomes"],
                ],
                    key=lambda row: row["ordinal"],
            )
        )
    )
    for key in shard_zero["resource_peaks"]:
        assert aggregate["resource_peaks"][key] == max(
            shard_zero["resource_peaks"][key],
            shard_one["resource_peaks"][key],
        )
    assert aggregate["fixture_suite_sha256"] == (
        qualification.FIXTURE_SUITE_SHA256
    )
    assert shard_zero[
        "teacher_forced_canary_self_sha256"
    ] == shard_one["teacher_forced_canary_self_sha256"]
    assert aggregate["teacher_forced_canary_self_sha256"] == (
        shard_zero["teacher_forced_canary_self_sha256"]
    )
    assert shard_zero["runtime_commitment"] == shard_one[
        "runtime_commitment"
    ]
    assert aggregate["runtime_commitment"] == shard_zero[
        "runtime_commitment"
    ]
    assert shard_zero["manifest_commitments"] == shard_one[
        "manifest_commitments"
    ]
    assert aggregate["manifest_commitments"] == shard_zero[
        "manifest_commitments"
    ]
    assert shard_zero[
        "implementation_closure_sha256"
    ] == shard_one["implementation_closure_sha256"]
    assert aggregate["implementation_closure_sha256"] == (
        shard_zero["implementation_closure_sha256"]
    )
    assert aggregate["shard_receipt_self_sha256"] == {
        "0": shard_zero["self_sha256"],
        "1": shard_one["self_sha256"],
    }
    assert aggregate["counters"] == {
        "api_access_count": 0,
        "free_form_generation_count": 0,
        "label_access_count": 0,
        "network_access_count": 0,
        "source_access_count": 0,
    }
    assert aggregate["qualification_passed"] is True
    _assert_self_hash(aggregate)
    assert raw == _canonical_bytes(aggregate)
    assert (output_path.stat().st_mode & 0o777) == 0o600
    nested = _all_nested_strings(aggregate)
    assert all(
        _fixture_story(fixture) not in nested
        for fixture in qualification.PUBLIC_FIXTURES
    )


def _rewrite_self_hashed(
    source: Path,
    target: Path,
    *,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    value = json.loads(source.read_text(encoding="ascii"))
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    mutate(body)
    target.write_bytes(_canonical_bytes(_self_hashed(body)))
    target.chmod(0o600)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_shard",
        "manifest",
        "runtime",
        "implementation",
        "canary",
        "suite",
    ],
)
def test_aggregate_rejects_duplicate_or_inconsistent_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    _zero, _raw, path_zero, _runtime, _manifest = _run_shard(
        monkeypatch,
        tmp_path / "zero",
        shard_index=0,
    )
    _one, _raw, path_one, _runtime, _manifest = _run_shard(
        monkeypatch,
        tmp_path / "one",
        shard_index=1,
    )
    if mutation == "duplicate_shard":
        paths = (path_zero, path_zero)
    else:
        changed = tmp_path / f"{mutation}.safe.json"

        def mutate(body: dict[str, object]) -> None:
            if mutation == "manifest":
                body["manifest_commitments"] = {
                    **body["manifest_commitments"],
                    "model_tree_sha256": hashlib.sha256(
                        b"drift"
                    ).hexdigest(),
                }
            elif mutation == "runtime":
                body["runtime_commitment"] = hashlib.sha256(
                    b"runtime drift"
                ).hexdigest()
            elif mutation == "implementation":
                body["implementation_closure_sha256"] = hashlib.sha256(
                    b"implementation drift"
                ).hexdigest()
            elif mutation == "canary":
                body["teacher_forced_canary_self_sha256"] = (
                    hashlib.sha256(b"canary drift").hexdigest()
                )
            elif mutation == "suite":
                body["fixture_suite_sha256"] = hashlib.sha256(
                    b"suite drift"
                ).hexdigest()

        _rewrite_self_hashed(path_one, changed, mutate=mutate)
        paths = (path_zero, changed)
    output_root = tmp_path / "aggregate"
    output_root.mkdir(mode=0o700)
    with pytest.raises(RuntimeError):
        qualification.aggregate_fixed_public_qualification(
            shard_receipts=paths,
            output_root=output_root,
        )
    assert not (
        output_root / "qualification.aggregate.safe.json"
    ).exists()


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = closed_v1.MAXIMUM_CONTEXT_TOKENS

    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size

    def __call__(
        self, text: str, **_: object
    ) -> dict[str, list[int]]:
        return {
            "input_ids": [
                ord(character) % (self.vocabulary_size - 1) + 1
                for character in text
            ]
        }


class _ToyDecoder:
    def __init__(self, owner: "_ToyCausalModel") -> None:
        self.owner = owner

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        assert use_cache is True
        assert return_dict is True
        hidden = self.owner.full_logits(input_ids)
        self.owner.decoder_calls.append(
            (
                int(input_ids.shape[0]),
                int(input_ids.shape[1]),
                int(attention_mask.shape[1]),
                past_key_values is not None,
            )
        )
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=("cache", len(self.owner.decoder_calls)),
        )


class _IdentityHead:
    def __init__(self, owner: "_ToyCausalModel") -> None:
        self.owner = owner

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        self.owner.head_shapes.append(tuple(hidden.shape))
        return hidden


class _ToyCausalModel:
    def __init__(self, *, sparse: bool) -> None:
        self.config = SimpleNamespace(
            vocab_size=19,
            max_position_embeddings=closed_v1.MAXIMUM_CONTEXT_TOKENS,
        )
        self.training = False
        self.sparse = sparse
        self.sparse_calls: list[tuple[int, ...]] = []
        self.decoder_calls: list[tuple[int, int, int, bool]] = []
        self.head_shapes: list[tuple[int, ...]] = []
        self.model = _ToyDecoder(self)
        self.lm_head = _IdentityHead(self)

    def named_parameters(self):
        return iter(())

    def parameters(self):
        return iter(())

    def full_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        vocabulary = torch.arange(
            self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, 1, -1)
        return -(
            vocabulary - input_ids.unsqueeze(-1).float()
        ).square() / 5.0


class _SparseToyCausalModel(_ToyCausalModel):
    def __init__(self) -> None:
        super().__init__(sparse=True)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: torch.Tensor | int | None = None,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        logits = self.full_logits(input_ids)
        if logits_to_keep is None or (
            isinstance(logits_to_keep, int)
            and not isinstance(logits_to_keep, bool)
            and logits_to_keep == 0
        ):
            return SimpleNamespace(logits=logits)
        self.sparse_calls.append(
            tuple(int(row) for row in logits_to_keep.tolist())
        )
        return SimpleNamespace(
            logits=logits.index_select(1, logits_to_keep)
        )

    __call__ = forward


class _FallbackToyCausalModel(_ToyCausalModel):
    def __init__(self) -> None:
        super().__init__(sparse=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(logits=self.full_logits(input_ids))

    __call__ = forward


@pytest.mark.parametrize(
    ("model_factory", "strategy"),
    [
        (_SparseToyCausalModel, memory_safe_qwen.SPARSE_STRATEGY),
        (
            _FallbackToyCausalModel,
            memory_safe_qwen.FALLBACK_STRATEGY,
        ),
    ],
)
def test_real_fixed_canary_is_cpu_source_free_and_matches_full_reference(
    monkeypatch: pytest.MonkeyPatch,
    model_factory: type[_ToyCausalModel],
    strategy: str,
) -> None:
    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("network/API path used by fixed canary")

    monkeypatch.setattr("socket.create_connection", forbidden)
    model = model_factory()
    runtime = memory_safe_qwen.build_fake_runtime_qualification_only(
        model=model,
        tokenizer=_CharacterTokenizer(model.config.vocab_size),
        torch_module=torch,
        device="cpu",
        strategy=strategy,
    )
    receipt = qualification._run_fixed_teacher_forced_canary(
        runtime
    )
    assert receipt["strategy"] == strategy
    assert receipt["short_strategy_vs_full_reference_exact"] is True
    assert (
        receipt["short_strategy_microunits"]
        == receipt["short_full_reference_microunits"]
    )
    assert receipt["long_answer_position_count"] > 128
    assert receipt["long_repeat_byte_exact"] is True
    assert (
        receipt["fallback_independent_full_reference_passed"] is True
    )
    assert receipt["free_form_generation_count"] == 0
    _assert_self_hash(receipt)
    if strategy == memory_safe_qwen.SPARSE_STRATEGY:
        assert receipt["sparse_chunk_count"] >= 2
        assert model.sparse_calls
        assert max(map(len, model.sparse_calls)) <= 128
    else:
        assert all(shape[1] == 1 for shape in model.head_shapes)


def test_runner_source_has_no_dynamic_content_or_network_api_surface() -> None:
    source = inspect.getsource(qualification)
    assert ".generate(" not in source
    assert "requests." not in source
    assert "urllib." not in source
    assert "http://" not in source
    assert "https://" not in source
    assert '"run_fixed_teacher_forced_canary"' in source
    assert "value = operation()" in source
    assert "row.ordinal % SHARD_COUNT == shard_index" in source
    assert tuple(
        inspect.signature(
            qualification._load_exact_runtime
        ).parameters
    ) == ("model_root", "manifest")


def test_fixed_services_use_shared_visible_precreated_tmp_roots() -> None:
    manifest_root = (
        Path(__file__).resolve().parents[1] / "manifests"
    )
    expected = {
        "gscl_narrative_extractor_v2_fixed_qualification_shard0.service": (
            "@QUALIFICATION_ROOT@/shard0/tmp"
        ),
        "gscl_narrative_extractor_v2_fixed_qualification_shard1.service": (
            "@QUALIFICATION_ROOT@/shard1/tmp"
        ),
        "gscl_narrative_extractor_v2_fixed_qualification_aggregate.service": (
            "@QUALIFICATION_ROOT@/aggregate/tmp"
        ),
    }
    for name, tmp_root in expected.items():
        source = (manifest_root / name).read_text(encoding="utf-8")
        assert source.count("PrivateTmp=no") == 1
        assert "PrivateTmp=yes" not in source
        assert f"Environment=TMPDIR={tmp_root}" in source
        assert "/var/tmp" not in source
        assert source.count("ReadWritePaths=") == 1
        assert "ReadWritePaths=@QUALIFICATION_ROOT@" not in source
        assert "ProtectSystem=no" in source
        assert "ProtectSystem=strict" not in source
        assert "ProtectHome=no" in source
        assert "ProtectHome=read-only" not in source
        assert "IPAddressDeny=any" in source
        assert "RestrictAddressFamilies=AF_UNIX" in source
        # Deployment must create these exact directories before start; the
        # service itself cannot silently substitute a namespace-private tmp.
        assert "ExecStartPre=" not in source

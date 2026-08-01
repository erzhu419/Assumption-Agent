from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from replication_runtime.gscl_narrative_extractor_v1 import contract
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_worker as closed,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_qwen_runtime as qwen_closed,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_actual_qualification as actual_qualification,
)
from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_manifest_builder as manifest_builder,
)


STORY = "Aster guides Birch while Birch supports Cedar."
_RUNTIME_COMMITMENT = hashlib.sha256(
    b"qualification-fake-teacher-forced-backend-v1"
).hexdigest()


def _parser(story: str, completion: str) -> object:
    return parse_untrusted_generator_completion(
        NarrativeSource("qualification.story", story),
        completion,
    )


class FakeTeacherForcedBackend:
    def __init__(
        self,
        *,
        preferred: tuple[str, ...] = (),
        tie: bool = False,
        malformed: bool = False,
    ) -> None:
        self.preferred = preferred
        self.tie = tie
        self.malformed = malformed
        self.calls: list[tuple[closed.PromptAnswer, ...]] = []

    @property
    def runtime_commitment(self) -> str:
        return _RUNTIME_COMMITMENT

    def score_batch(
        self, pairs: tuple[closed.PromptAnswer, ...]
    ) -> tuple[closed.TeacherForcedScore, ...]:
        self.calls.append(pairs)
        if self.malformed:
            return ()
        rows: list[closed.TeacherForcedScore] = []
        for pair in pairs:
            token_count = max(1, len(pair.answer.split()))
            if self.tie:
                total = -10 * token_count
            else:
                preference = (
                    len(self.preferred)
                    - self.preferred.index(pair.candidate_key)
                    if pair.candidate_key in self.preferred
                    else 0
                )
                stable = int(
                    hashlib.sha256(
                        pair.candidate_key.encode("ascii")
                    ).hexdigest()[:8],
                    16,
                )
                total = (preference * 10**12 + stable) * token_count
            rows.append(
                closed.TeacherForcedScore(
                    total_logprob_microunits=total,
                    answer_token_count=token_count,
                    context_and_answer_token_count=token_count + 40,
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(
        self, completion: str
    ) -> int:
        # A deterministic fake tokenizer; it observes the assembled wire only
        # after selection and never authors any of its bytes.
        return max(1, len(completion.encode("utf-8")) // 4)


def _decision(
    backend: FakeTeacherForcedBackend,
    *,
    batch_size: int = closed.SCORING_BATCH_SIZE,
) -> closed.ClosedChoiceDecision:
    return closed.select_closed_choice_qualification_only(
        STORY,
        backend=backend,
        narrative_parser=_parser,
        scoring_batch_size=batch_size,
    )


def _selected_spans(completion: str) -> tuple[str, str, str]:
    value = json.loads(completion)
    return (
        value["generators"][0]["anchor_span_id"],
        value["objects"][0]["span_id"],
        value["objects"][1]["span_id"],
    )


def _intervals(story: str) -> dict[str, tuple[int, int]]:
    rows: dict[str, tuple[int, int]] = {}
    for row in contract.build_story_span_catalog(story):
        quote = row["quote"]
        occurrence = row["occurrence"]
        starts: list[int] = []
        offset = 0
        while True:
            position = story.find(quote, offset)
            if position < 0:
                break
            starts.append(position)
            offset = position + 1
        start = starts[occurrence]
        rows[row["span_id"]] = (start, start + len(quote))
    return rows


def test_program_owns_every_required_wire_field_and_parser_abi() -> None:
    decision = _decision(FakeTeacherForcedBackend(tie=True))
    wire = json.loads(decision.completion)
    assert set(wire) == {"generators", "objects", "schema_version"}
    assert wire["schema_version"] == contract.WIRE_COMPLETION_SCHEMA
    assert len(wire["objects"]) == 2
    assert [row["object_id"] for row in wire["objects"]] == ["o0", "o1"]
    assert all(set(row) == {"object_id", "span_id"} for row in wire["objects"])
    assert len(wire["generators"]) == 1
    generator = wire["generators"][0]
    assert set(generator) == {
        "anchor_span_id",
        "causal_orientation",
        "generator_id",
        "generator_kind",
        "polarity",
        "slot_object_ids",
        "temporal_orientation",
    }
    assert generator["generator_id"] == "g0"
    assert generator["slot_object_ids"] == ["o0", "o1"]
    assert contract.validate_completion(
        STORY,
        decision.completion,
        narrative_parser=_parser,
    ) == decision.canonical_completion


def test_unknown_primitives_are_not_in_the_candidate_language() -> None:
    decision = _decision(FakeTeacherForcedBackend(tie=True))
    generator = json.loads(decision.completion)["generators"][0]
    assert generator["generator_kind"] in closed.GENERATOR_KINDS
    assert generator["polarity"] in closed.POLARITIES
    assert generator["temporal_orientation"] in closed.ORIENTATIONS
    assert generator["causal_orientation"] in closed.ORIENTATIONS
    all_answers = [
        pair.answer
        for batch in FakeTeacherForcedBackend().calls
        for pair in batch
    ]
    assert "unknown" not in " ".join(all_answers)


def test_overlapping_spans_are_filtered_before_scoring() -> None:
    # s000 is "Aster"; s001/s002/s003 contain it and must never be offered
    # after anchor selection. s004 is "guides"; its containing spans must then
    # be absent from the object1 alternatives.
    backend = FakeTeacherForcedBackend(
        preferred=("anchor:s000", "object0:s004", "object1:s008")
    )
    decision = _decision(backend)
    anchor, object0, object1 = _selected_spans(decision.completion)
    intervals = _intervals(STORY)
    selected = [intervals[anchor], intervals[object0], intervals[object1]]
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            assert left[1] <= right[0] or right[1] <= left[0]

    object0_candidates = {
        pair.candidate_key
        for batch in backend.calls
        for pair in batch
        if pair.candidate_key.startswith("object0:")
    }
    assert {"object0:s001", "object0:s002", "object0:s003"}.isdisjoint(
        object0_candidates
    )


def test_exact_ties_use_catalog_and_enum_enumeration_order() -> None:
    first = _decision(FakeTeacherForcedBackend(tie=True))
    second = _decision(FakeTeacherForcedBackend(tie=True))
    assert first.completion == second.completion
    assert first.canonical_completion == second.canonical_completion
    assert first.receipt_bytes == second.receipt_bytes
    wire = json.loads(first.completion)
    assert wire["generators"][0]["generator_kind"] == closed.GENERATOR_KINDS[0]
    assert wire["generators"][0]["polarity"] == closed.POLARITIES[0]
    assert (
        wire["generators"][0]["temporal_orientation"]
        == closed.ORIENTATIONS[0]
    )
    assert (
        wire["generators"][0]["causal_orientation"]
        == closed.ORIENTATIONS[0]
    )


def test_selection_is_invariant_to_candidate_batch_partition() -> None:
    one = _decision(FakeTeacherForcedBackend(), batch_size=1)
    seven = _decision(FakeTeacherForcedBackend(), batch_size=7)
    sixteen = _decision(
        FakeTeacherForcedBackend(),
        batch_size=closed.SCORING_BATCH_SIZE,
    )
    assert one.completion == seven.completion == sixteen.completion
    assert (
        one.canonical_completion
        == seven.canonical_completion
        == sixteen.canonical_completion
    )
    assert one.receipt_bytes == seven.receipt_bytes == sixteen.receipt_bytes


def test_unrepresentable_story_abstains_before_any_model_call() -> None:
    backend = FakeTeacherForcedBackend()
    with pytest.raises(closed.ClosedChoiceAbstention) as error:
        closed.select_closed_choice_qualification_only(
            "Aster guides",
            backend=backend,
            narrative_parser=_parser,
        )
    assert error.value.issue_id == (
        "closed_choice_nonoverlapping_triple_unavailable"
    )
    assert error.value.pre_model is True
    assert backend.calls == []


def test_backend_shape_tamper_fails_closed() -> None:
    with pytest.raises(closed.ClosedChoiceError) as error:
        _decision(FakeTeacherForcedBackend(malformed=True))
    assert error.value.issue_id == "closed_choice_score_batch_invalid"


def test_engine_construction_token_is_not_caller_forgeable() -> None:
    with pytest.raises(closed.ClosedChoiceError) as error:
        closed._ClosedChoiceEngine(object())
    assert error.value.issue_id == "closed_choice_engine_authority_invalid"


def test_no_free_form_generate_call_exists() -> None:
    source = inspect.getsource(closed)
    assert ".generate(" not in source
    assert "teacher_forced_forward_log_softmax" in source
    assert closed.SCORING_POLICY["free_form_generation_count"] == 0


def test_safe_receipt_commits_but_does_not_reveal_story_or_quotes() -> None:
    decision = _decision(FakeTeacherForcedBackend(tie=True))
    receipt_text = decision.receipt_bytes.decode("ascii")
    receipt = decision.receipt
    assert STORY not in receipt_text
    assert "Aster" not in receipt_text
    assert "Birch" not in receipt_text
    assert receipt["free_form_generation_count"] == 0
    assert receipt["prompt_closure_sha256"] == (
        closed.PROMPT_CLOSURE_SHA256
    )
    assert receipt["model_runtime_commitment"] == _RUNTIME_COMMITMENT
    assert receipt["selected_answer_token_count"] == (
        decision.selected_answer_token_count
    )
    assert receipt["wire_shape"] == {
        "generator_count": 1,
        "object_count": 2,
        "slot_count": 2,
    }
    assert len(receipt["steps"]) == 4
    assert all(
        set(step)
        == {
            "candidate_count",
            "candidate_set_commitment",
            "score_summary_commitment",
            "selected_candidate_commitment",
            "step",
        }
        for step in receipt["steps"]
    )


def test_receipt_binding_changes_when_scores_change() -> None:
    tied = _decision(FakeTeacherForcedBackend(tie=True))
    ranked = _decision(
        FakeTeacherForcedBackend(
            preferred=(
                "anchor:s004",
                "object0:s000",
                "object1:s008",
                "enum:107",
            )
        )
    )
    assert tied.receipt_bytes != ranked.receipt_bytes
    assert tied.receipt["self_sha256"] != ranked.receipt["self_sha256"]


def test_teacher_forced_score_rejects_nonfinite_or_oversized_shapes() -> None:
    with pytest.raises(closed.ClosedChoiceError):
        closed.TeacherForcedScore(
            total_logprob_microunits=10**16,
            answer_token_count=1,
            context_and_answer_token_count=1,
        )
    with pytest.raises(closed.ClosedChoiceError):
        closed.TeacherForcedScore(
            total_logprob_microunits=0,
            answer_token_count=2,
            context_and_answer_token_count=1,
        )


def test_candidate_batches_are_nonempty_and_bounded() -> None:
    backend = FakeTeacherForcedBackend(tie=True)
    _decision(backend)
    assert backend.calls
    assert all(
        1 <= len(batch) <= closed.SCORING_BATCH_SIZE
        for batch in backend.calls
    )
    assert all(
        pair.prompt and pair.answer and pair.candidate_key
        for batch in backend.calls
        for pair in batch
    )


def test_wire_token_count_is_not_selected_answer_token_count() -> None:
    decision = _decision(FakeTeacherForcedBackend(tie=True))
    assert decision.wire_completion_token_count == len(
        decision.completion.encode("utf-8")
    ) // 4
    assert decision.selected_answer_token_count != (
        decision.wire_completion_token_count
    )


def test_formal_runtime_has_no_injected_score_or_generation_surface() -> None:
    signature = inspect.signature(qwen_closed.process_formal_pack)
    assert tuple(signature.parameters) == ("pack", "runtime")
    assert "backend" not in signature.parameters
    assert "parser" not in signature.parameters
    assert "logits" not in signature.parameters
    assert "completion" not in signature.parameters
    assert not hasattr(
        qwen_closed.LocalQwenClosedChoiceRuntime, "score_batch"
    )
    assert not hasattr(
        qwen_closed.LocalQwenClosedChoiceRuntime, "generate"
    )
    source = inspect.getsource(qwen_closed)
    assert ".generate(" not in source
    assert "torch.logsumexp" in source
    assert "free_form_generation_count" in source
    assert (
        1
        <= qwen_closed.FORMAL_SCORING_BATCH_SIZE
        <= closed.SCORING_BATCH_SIZE
    )


def test_private_exact_backend_rejects_forged_authority() -> None:
    with pytest.raises(closed.ClosedChoiceError) as error:
        qwen_closed._ExactTeacherForcedBackend(
            object(), object(), object()
        )
    assert error.value.issue_id == (
        "closed_choice_backend_authority_invalid"
    )


def test_actual_qualification_cli_has_no_source_or_scorer_input() -> None:
    signature = inspect.signature(
        actual_qualification.run_source_free_actual_qualification
    )
    assert tuple(signature.parameters) == (
        "model_root",
        "model_manifest_path",
        "output_path",
    )
    source = inspect.getsource(actual_qualification)
    assert ".generate(" not in source
    assert 'parser.add_argument("--story"' not in source
    assert 'parser.add_argument("--source"' not in source
    assert "formal_measurement" in source


def test_qwen_runtime_real_attribute_shapes_and_module_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (7, 5)

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Fake GPU"

    class FakeCudnn:
        @staticmethod
        def version() -> int:
            return 90100

    fake_torch = SimpleNamespace(
        __file__="/venv/site-packages/torch/__init__.py",
        __version__="2.8.0+cu128",
        backends=SimpleNamespace(cudnn=FakeCudnn()),
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="12.8"),
    )
    fake_transformers = SimpleNamespace(
        __file__="/venv/site-packages/transformers/__init__.py",
        __version__="5.10.1",
    )
    fake_config = SimpleNamespace(
        _attn_implementation="sdpa",
        max_position_embeddings=32768,
    )
    runtime = object.__new__(
        qwen_closed.LocalQwenClosedChoiceRuntime
    )
    object.__setattr__(
        runtime, "_model", SimpleNamespace(config=fake_config)
    )
    object.__setattr__(
        runtime,
        "_tokenizer",
        SimpleNamespace(model_max_length=32768),
    )
    object.__setattr__(runtime, "_torch", fake_torch)
    object.__setattr__(
        runtime, "_transformers", fake_transformers
    )

    observed: list[tuple[str, tuple[Path, ...]]] = []

    def distribution(
        name: str, *, required_module_origins: tuple[Path, ...]
    ) -> str:
        observed.append((name, required_module_origins))
        return hashlib.sha256(name.encode("ascii")).hexdigest()

    monkeypatch.setattr(
        qwen_closed.worker,
        "_distribution_closure_sha256",
        distribution,
    )
    monkeypatch.setattr(
        qwen_closed.worker,
        "_hash_runtime_executable",
        lambda: hashlib.sha256(b"python").hexdigest(),
    )
    assert runtime._context_limit() == 32768
    environment = runtime._runtime_environment()
    assert environment["gpu_compute_capability"] == [7, 5]
    assert observed == [
        (
            "torch",
            (Path("/venv/site-packages/torch/__init__.py"),),
        ),
        (
            "transformers",
            (
                Path(
                    "/venv/site-packages/transformers/__init__.py"
                ),
            ),
        ),
    ]


def test_qwen_runtime_proves_exact_prompt_answer_token_boundary() -> None:
    class CharacterTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": [ord(character) for character in text]}

    runtime = object.__new__(
        qwen_closed.LocalQwenClosedChoiceRuntime
    )
    object.__setattr__(runtime, "_tokenizer", CharacterTokenizer())
    pair = closed.PromptAnswer(
        candidate_key="fixture",
        prompt="prompt\n",
        answer="answer",
    )
    prompt, answer, combined = runtime._prompt_answer_token_ids(pair)
    assert combined == prompt + answer


def test_qwen_runtime_rejects_tokenizer_boundary_merge() -> None:
    class BoundaryMergingTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            mapping = {
                "prompt\n": [10],
                "answer": [20],
                "prompt\nanswer": [30],
            }
            return {"input_ids": mapping[text]}

    runtime = object.__new__(
        qwen_closed.LocalQwenClosedChoiceRuntime
    )
    object.__setattr__(
        runtime, "_tokenizer", BoundaryMergingTokenizer()
    )
    pair = closed.PromptAnswer(
        candidate_key="fixture",
        prompt="prompt\n",
        answer="answer",
    )
    with pytest.raises(closed.ClosedChoiceError) as error:
        runtime._prompt_answer_token_ids(pair)
    assert error.value.issue_id == (
        "closed_choice_token_boundary_invalid"
    )


def test_qwen_runtime_binds_all_critical_dependency_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        module_name: SimpleNamespace(
            __file__=f"/venv/site-packages/{module_name}/__init__.py"
        )
        for module_name, _ in (
            qwen_closed.CRITICAL_DEPENDENCY_DISTRIBUTIONS
        )
    }
    observed: list[tuple[str, tuple[Path, ...]]] = []

    monkeypatch.setattr(
        qwen_closed,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        qwen_closed.importlib_metadata,
        "version",
        lambda name: f"{name}-version",
    )

    def distribution(
        name: str, *, required_module_origins: tuple[Path, ...]
    ) -> str:
        observed.append((name, required_module_origins))
        return hashlib.sha256(name.encode("ascii")).hexdigest()

    monkeypatch.setattr(
        qwen_closed.worker,
        "_distribution_closure_sha256",
        distribution,
    )
    closure = qwen_closed._critical_dependency_closure()
    rows = closure["dependencies"]
    assert [row["module"] for row in rows] == [
        module_name
        for module_name, _ in (
            qwen_closed.CRITICAL_DEPENDENCY_DISTRIBUTIONS
        )
    ]
    assert observed == [
        (
            distribution_name,
            (
                Path(
                    f"/venv/site-packages/{module_name}/__init__.py"
                ),
            ),
        )
        for module_name, distribution_name in (
            qwen_closed.CRITICAL_DEPENDENCY_DISTRIBUTIONS
        )
    ]
    body = {
        key: value
        for key, value in closure.items()
        if key != "self_sha256"
    }
    assert closure["self_sha256"] == contract.semantic_sha256(body)


def test_manifest_builder_has_no_weight_or_source_input() -> None:
    signature = inspect.signature(
        manifest_builder.build_manifest_without_weight_load
    )
    assert tuple(signature.parameters) == (
        "model_root",
        "output_path",
    )
    source = inspect.getsource(manifest_builder)
    assert "AutoModel" not in source
    assert ".generate(" not in source
    assert 'parser.add_argument("--story"' not in source
    assert 'parser.add_argument("--source"' not in source
    assert "required_module_origins" in source
    assert "model_weight_load_count" in source


def test_manifest_builder_rejects_output_inside_model_tree_preload(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(closed.ClosedChoiceError) as error:
        manifest_builder.build_manifest_without_weight_load(
            model_root=model,
            output_path=model / "manifest.json",
        )
    assert error.value.issue_id == (
        "closed_choice_manifest_inside_model_tree"
    )


def test_manifest_publish_failure_leaves_no_final_or_pending(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"

    def reject(_: Path) -> None:
        raise RuntimeError("synthetic validation failure")

    with pytest.raises(RuntimeError, match="synthetic validation"):
        manifest_builder._publish_validated_once(
            output,
            b'{"candidate":true}\n',
            validate_pending=reject,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".manifest.json.pending-*")) == []


def test_manifest_publish_is_validated_and_never_overwrites(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"
    raw = b'{"candidate":true}\n'
    observed: list[bytes] = []

    def validate(path: Path) -> None:
        observed.append(path.read_bytes())

    digest = manifest_builder._publish_validated_once(
        output,
        raw,
        validate_pending=validate,
    )
    assert observed == [raw]
    assert output.read_bytes() == raw
    assert digest == hashlib.sha256(raw).hexdigest()
    with pytest.raises(closed.ClosedChoiceError) as error:
        manifest_builder._publish_validated_once(
            output,
            b"replacement",
            validate_pending=validate,
        )
    assert error.value.issue_id == (
        "closed_choice_manifest_output_invalid"
    )
    assert output.read_bytes() == raw


def test_actual_receipt_is_finally_validated_before_publication(
    tmp_path: Path,
) -> None:
    source = inspect.getsource(
        actual_qualification.run_source_free_actual_qualification
    )
    final_validation = source.rfind(
        "runtime._validate_formal_binding()"
    )
    publication = source.rfind("_publish_once(output_path, raw)")
    assert 0 <= final_validation < publication

    output = tmp_path / "actual.safe.json"
    raw = b'{"status":"PASS"}\n'
    assert actual_qualification._publish_once(output, raw) == (
        hashlib.sha256(raw).hexdigest()
    )
    assert output.read_bytes() == raw
    with pytest.raises(closed.ClosedChoiceError):
        actual_qualification._publish_once(
            output, b'{"status":"REPLACED"}\n'
        )
    assert output.read_bytes() == raw

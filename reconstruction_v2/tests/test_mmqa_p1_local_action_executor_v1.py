from __future__ import annotations

import ast
import hashlib
import inspect
import math

import pytest

from assumption_agent.benchmarks import mmqa_p1_action_integration_v1 as integration
from assumption_agent.benchmarks import mmqa_p1_local_action_executor_v1 as executor
from assumption_agent.benchmarks import mmqa_p1_typed_proof_e5_core_v1 as core
from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    contract as eraser_hippo,
)


def _work_item() -> integration.AnonymousWorkItem:
    return integration.validate_anonymous_work_item(
        {
            "schema": integration.ANONYMOUS_WORK_ITEM_SCHEMA,
            "question": "Which Aurora launch year is established?",
            "rows": [
                {
                    "ordinal": 0,
                    "serialized_content": "Aurora launch year | 2012",
                },
                {
                    "ordinal": 1,
                    "serialized_content": "Borealis launch year | 2014",
                },
                {
                    "ordinal": 2,
                    "serialized_content": "Aurora operator | North Lab",
                },
                {
                    "ordinal": 3,
                    "serialized_content": "Southern programme | status active",
                },
            ],
            "texts": [
                {
                    "ordinal": 4,
                    "serialized_content": "Aurora launch year was 2012.",
                },
                {
                    "ordinal": 5,
                    "serialized_content": "Borealis began in 2014.",
                },
                {
                    "ordinal": 6,
                    "serialized_content": "North Lab operates Aurora.",
                },
                {
                    "ordinal": 7,
                    "serialized_content": "The programme remains active.",
                },
            ],
            "exact_row_text_links": [
                {"row_ordinal": 0, "text_ordinal": 4},
                {"row_ordinal": 1, "text_ordinal": 4},
                {"row_ordinal": 1, "text_ordinal": 5},
                {"row_ordinal": 2, "text_ordinal": 6},
                {"row_ordinal": 3, "text_ordinal": 7},
            ],
        }
    )


def _binding(**overrides: object) -> executor.FrozenLocalModelBinding:
    values: dict[str, object] = {
        "minilm_model_path": "/frozen/models/all-MiniLM-L6-v2",
        "minilm_required_tree_sha256": executor.MINILM_REQUIRED_TREE_SHA256,
        "cross_encoder_model_path": "/frozen/models/ms-marco-MiniLM-L-6-v2",
        "cross_encoder_required_tree_sha256": (
            executor.CROSS_ENCODER_REQUIRED_TREE_SHA256
        ),
        "local_runtime_identity_sha256": "a" * 64,
    }
    values.update(overrides)
    return executor.FrozenLocalModelBinding(**values)  # type: ignore[arg-type]


def _vector(first: float, second: float) -> tuple[float, ...]:
    return (first, second) + (0.0,) * (executor.MINILM_EMBEDDING_DIMENSION - 2)


class _Recorder:
    def __init__(self) -> None:
        self.minilm_calls: list[dict[str, object]] = []
        self.ce_calls: list[dict[str, object]] = []

    def encode(self, **kwargs: object) -> tuple[tuple[float, ...], ...]:
        self.minilm_calls.append(kwargs)
        # Question, then eight source-local units.  The first three unit
        # cosines are 1, -1 and 0; all remaining rows are finite/nonzero.
        return (
            _vector(1.0, 0.0),
            _vector(1.0, 0.0),
            _vector(-1.0, 0.0),
            _vector(0.0, 1.0),
            _vector(1.0, 1.0),
            _vector(1.0, -1.0),
            _vector(2.0, 1.0),
            _vector(-1.0, 1.0),
            _vector(1.0, 2.0),
        )

    def score(self, **kwargs: object) -> tuple[float, ...]:
        self.ce_calls.append(kwargs)
        return (
            0.0,
            math.log(3.0),
            -math.log(3.0),
            2.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
        )


def _execute(
    recorder: _Recorder | None = None,
) -> tuple[executor.LocalActionExecution, _Recorder]:
    recorder = recorder or _Recorder()
    result = executor.execute_local_actions(
        _work_item(),
        model_binding=_binding(),
        batch_functions=executor.LocalBatchFunctions(
            encode_minilm=recorder.encode,
            score_cross_encoder=recorder.score,
        ),
    )
    return result, recorder


def test_frozen_constants_and_source_free_surface() -> None:
    assert executor.STUDY_ID == "MMQA_P1_LOCAL_PROOF_E5_V1"
    assert executor.STUDY_DESIGN_SELF_SHA256 == (
        "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
    )
    assert executor.MINILM_EMBEDDING_DIMENSION == 384
    assert executor.MINILM_BATCH_SIZE == 32
    assert executor.CROSS_ENCODER_BATCH_SIZE == 64
    assert executor.MINILM_MAX_LENGTH == 256
    assert executor.CROSS_ENCODER_MAX_LENGTH == 512

    parameters = inspect.signature(executor.execute_local_actions).parameters
    assert set(parameters) == {
        "work_item",
        "model_binding",
        "batch_functions",
        "e5_model",
    }
    assert not {
        "source",
        "gold",
        "answer",
        "support",
        "family",
        "qid",
    }.intersection(parameters)

    tree = ast.parse(inspect.getsource(executor))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported_roots.intersection(
        {
            "aiohttp",
            "datasets",
            "httpx",
            "os",
            "pathlib",
            "requests",
            "socket",
            "subprocess",
            "torch",
            "transformers",
            "urllib",
        }
    )


def test_binding_is_exact_offline_caller_verified_identity() -> None:
    binding = _binding()
    public = binding.public_binding()
    assert public["minilm_required_tree_sha256"] == (
        executor.MINILM_REQUIRED_TREE_SHA256
    )
    assert public["cross_encoder_required_tree_sha256"] == (
        executor.CROSS_ENCODER_REQUIRED_TREE_SHA256
    )
    assert public["asset_identity_verified"] is True
    assert public["local_files_only"] is True
    assert public["trust_remote_code"] is False
    assert public["network_disabled"] is True
    assert public["retry_count"] == 0
    assert binding.minilm_model_path not in str(public)
    assert binding.cross_encoder_model_path not in str(public)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minilm_model_path", "relative/model", "absolute lexical"),
        ("minilm_model_path", "/models/../escape", "absolute lexical"),
        ("minilm_required_tree_sha256", "0" * 64, "identity drifted"),
        ("cross_encoder_required_tree_sha256", "0" * 64, "identity drifted"),
        ("local_runtime_identity_sha256", "ABC", "lowercase SHA-256"),
        ("asset_identity_verified", False, "policy drifted"),
        ("local_files_only", False, "policy drifted"),
        ("trust_remote_code", True, "policy drifted"),
        ("network_disabled", False, "policy drifted"),
        ("retry_count", 1, "policy drifted"),
    ],
)
def test_binding_rejects_identity_or_offline_policy_drift(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(executor.MmqaP1LocalActionExecutorError, match=message):
        _binding(**{field: value})


def test_executor_accepts_only_validated_anonymous_work_item() -> None:
    raw = _work_item().anonymous_payload()
    recorder = _Recorder()
    with pytest.raises(
        executor.MmqaP1LocalActionExecutorError, match="AnonymousWorkItem only"
    ):
        executor.execute_local_actions(  # type: ignore[arg-type]
            raw,
            model_binding=_binding(),
            batch_functions=executor.LocalBatchFunctions(
                recorder.encode, recorder.score
            ),
        )
    assert recorder.minilm_calls == []
    assert recorder.ce_calls == []


def test_one_local_batch_each_produces_exact_scores_flags_and_actions() -> None:
    result, recorder = _execute()
    assert len(recorder.minilm_calls) == 1
    assert len(recorder.ce_calls) == 1
    minilm = recorder.minilm_calls[0]
    ce = recorder.ce_calls[0]
    assert minilm == {
        "model_path": "/frozen/models/all-MiniLM-L6-v2",
        "texts": (
            _work_item().question,
            *tuple(unit.serialized_content for unit in _work_item().units),
        ),
        "batch_size": 32,
        "max_length": 256,
        "normalize_embeddings": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "network_disabled": True,
        "deterministic": True,
    }
    assert ce["model_path"] == "/frozen/models/ms-marco-MiniLM-L-6-v2"
    assert ce["question"] == _work_item().question
    assert ce["documents"] == tuple(
        unit.serialized_content for unit in _work_item().units
    )
    assert ce["batch_size"] == 64
    assert ce["max_length"] == 512
    assert ce["local_files_only"] is True
    assert ce["trust_remote_code"] is False
    assert ce["network_disabled"] is True
    assert ce["deterministic"] is True

    coordinates = result.actions.unit_coordinates
    assert tuple(row.ordinal for row in coordinates) == tuple(range(8))
    assert coordinates[0].minilm_similarity == pytest.approx(1.0)
    assert coordinates[1].minilm_similarity == pytest.approx(0.0)
    assert coordinates[2].minilm_similarity == pytest.approx(0.5)
    assert coordinates[0].cross_encoder_relevance == pytest.approx(0.5)
    assert coordinates[1].cross_encoder_relevance == pytest.approx(0.75)
    assert coordinates[2].cross_encoder_relevance == pytest.approx(0.25)
    assert (
        coordinates[0].entity_anchor,
        coordinates[0].relation_anchor,
        coordinates[0].numeric_or_temporal_anchor,
    ) == (1, 1, 1)
    assert result.actions.shared_closure.ordinals == tuple(range(8))
    assert result.actions.e5_ranking is None


def test_real_local_numpy_batch_outputs_are_accepted() -> None:
    numpy = pytest.importorskip("numpy")
    recorder = _Recorder()

    def encode(**kwargs: object) -> object:
        recorder.minilm_calls.append(kwargs)
        return numpy.asarray(_Recorder().encode(), dtype=numpy.float32)

    def score(**kwargs: object) -> object:
        recorder.ce_calls.append(kwargs)
        return numpy.asarray(_Recorder().score(), dtype=numpy.float32)

    result = executor.execute_local_actions(
        _work_item(),
        model_binding=_binding(),
        batch_functions=executor.LocalBatchFunctions(encode, score),
    )
    assert result.actions.unit_coordinates[0].minilm_similarity == pytest.approx(1.0)
    assert result.actions.unit_coordinates[1].cross_encoder_relevance == (
        pytest.approx(0.75)
    )
    assert len(recorder.minilm_calls) == len(recorder.ce_calls) == 1


def test_receipt_is_hash_only_and_audits_exact_call_budget() -> None:
    result, _recorder = _execute()
    payload = result.receipt.payload()
    assert payload["anonymous_projection_sha256"] == (
        _work_item().anonymous_projection_sha256
    )
    for key in (
        "anonymous_projection_sha256",
        "local_model_binding_sha256",
        "minilm_score_vector_sha256",
        "cross_encoder_score_vector_sha256",
        "anchor_flag_vector_sha256",
    ):
        assert isinstance(payload[key], str)
        assert len(payload[key]) == 64
    assert payload["minilm_batch_call_count"] == 1
    assert payload["cross_encoder_batch_call_count"] == 1
    assert payload["model_call_count"] == 2
    assert payload["source_reader_call_count"] == 0
    assert payload["gold_answer_support_family_qid_read_count"] == 0
    assert payload["network_or_api_call_count"] == 0
    assert payload["retry_replay_resample_count"] == 0
    serialized = str(payload)
    assert _work_item().question not in serialized
    assert "Aurora launch year" not in serialized
    assert "/frozen/models" not in serialized


@pytest.mark.parametrize(
    ("question", "content", "expected"),
    [
        (
            "Which Aurora launch year?",
            "Aurora launch year was 2012.",
            (1, 1, 1),
        ),
        ("Which Aurora launch year?", "Unrelated prose.", (0, 0, 0)),
        ("Which Aurora launch year?", "The value was 1999-04-03.", (0, 0, 1)),
        ("Did aurora launch?", "aurora launch happened.", (0, 1, 0)),
    ],
)
def test_anchor_parser_is_deterministic_exact_surface_only(
    question: str, content: str, expected: tuple[int, int, int]
) -> None:
    first = executor.deterministic_anchor_flags(question, content)
    assert first == expected
    assert executor.deterministic_anchor_flags(question, content) == first


@pytest.mark.parametrize(
    "embeddings",
    [
        (_vector(1.0, 0.0),) * 8,
        ((_vector(1.0, 0.0)[:-1]),) * 9,
        ((_vector(0.0, 0.0)),) * 9,
        ((_vector(float("nan"), 0.0)),) * 9,
    ],
)
def test_malformed_embedding_batch_fails_before_cross_encoder(
    embeddings: tuple[tuple[float, ...], ...],
) -> None:
    calls = {"minilm": 0, "ce": 0}

    def encode(**_kwargs: object) -> object:
        calls["minilm"] += 1
        return embeddings

    def score(**_kwargs: object) -> object:
        calls["ce"] += 1
        return (0.0,) * 8

    with pytest.raises(executor.MmqaP1LocalActionExecutorError, match="MiniLM"):
        executor.execute_local_actions(
            _work_item(),
            model_binding=_binding(),
            batch_functions=executor.LocalBatchFunctions(encode, score),
        )
    assert calls == {"minilm": 1, "ce": 0}


@pytest.mark.parametrize(
    "logits",
    [
        (0.0,) * 7,
        (0.0,) * 7 + (float("inf"),),
        (0.0,) * 7 + (True,),
    ],
)
def test_malformed_cross_encoder_batch_fails_once(logits: tuple[object, ...]) -> None:
    recorder = _Recorder()

    def score(**kwargs: object) -> object:
        recorder.ce_calls.append(kwargs)
        return logits

    with pytest.raises(
        executor.MmqaP1LocalActionExecutorError, match="cross-encoder"
    ):
        executor.execute_local_actions(
            _work_item(),
            model_binding=_binding(),
            batch_functions=executor.LocalBatchFunctions(recorder.encode, score),
        )
    assert len(recorder.minilm_calls) == 1
    assert len(recorder.ce_calls) == 1


@pytest.mark.parametrize("failure_stage", ["minilm", "cross_encoder"])
def test_backend_exception_is_not_retried(failure_stage: str) -> None:
    recorder = _Recorder()

    def encode(**kwargs: object) -> object:
        recorder.minilm_calls.append(kwargs)
        if failure_stage == "minilm":
            raise OSError("synthetic failure")
        return _Recorder().encode()

    def score(**kwargs: object) -> object:
        recorder.ce_calls.append(kwargs)
        raise OSError("synthetic failure")

    with pytest.raises(executor.MmqaP1LocalActionExecutorError, match="failed"):
        executor.execute_local_actions(
            _work_item(),
            model_binding=_binding(),
            batch_functions=executor.LocalBatchFunctions(encode, score),
        )
    assert len(recorder.minilm_calls) == 1
    assert len(recorder.ce_calls) == int(failure_stage == "cross_encoder")


def test_optional_frozen_e5_is_forwarded_without_extra_model_calls() -> None:
    zero_e5 = core.E5Model(
        population_mean=(0.5,) * 10 + (-0.5,),
        population_std=(1.0,) * len(core.FEATURE_ORDER),
        coefficients=(0.0,) * len(core.FEATURE_ORDER),
        training_item_count=1,
        training_bundle_count=2,
        solver="numpy_deterministic_lbfgs_m10_v1",
        iterations=0,
        converged=True,
        objective=0.0,
    )
    recorder = _Recorder()
    result = executor.execute_local_actions(
        _work_item(),
        model_binding=_binding(),
        batch_functions=executor.LocalBatchFunctions(
            recorder.encode, recorder.score
        ),
        e5_model=zero_e5,
    )
    assert result.actions.e5_ranking is not None
    assert result.actions.e5_ranking.policy_id == "E5"
    assert len(recorder.minilm_calls) == len(recorder.ce_calls) == 1


def test_hipporag_payload_is_exact_anonymous_common_closure_contract() -> None:
    result, _recorder = _execute()
    payload = executor.build_candidate_restricted_hipporag_payload(result.actions)
    closure = result.actions.shared_closure
    expected_texts = tuple(unit.serialized_content for unit in closure.units)
    assert payload.logical_source_ordinals == closure.ordinals
    assert payload.exact_sentence_texts == expected_texts
    assert payload.worker_payload() == {
        "query": _work_item().question,
        "schema": eraser_hippo.INPUT_SCHEMA,
        "sentence_texts": list(expected_texts),
    }
    assert payload.canonical_worker_bytes() == (
        eraser_hippo.canonical_json_bytes(payload.worker_payload())
    )
    assert all(
        not text.startswith(f"{ordinal}:")
        for ordinal, text in zip(
            payload.logical_source_ordinals,
            payload.exact_sentence_texts,
            strict=True,
        )
    )
    binding = payload.anonymous_binding()
    assert binding["closure_ordinal_bytes_sha256"] == (
        closure.ordinal_bytes_sha256
    )
    assert binding["logical_document_count"] == 8
    assert binding["model_run_count_in_this_adapter"] == 0
    assert binding["network_disabled"] is True
    assert binding["source_reader_call_count"] == 0
    assert binding["retry_replay_resample_count"] == 0


def test_hipporag_exact_text_quotient_keeps_logical_ordinals_separate() -> None:
    result, _recorder = _execute()
    closure = result.actions.shared_closure
    duplicate_texts = (
        "duplicate",
        "duplicate",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
    )
    payload = executor.CandidateRestrictedHippoRAGPayload(
        query=_work_item().question,
        logical_source_ordinals=closure.ordinals,
        exact_sentence_texts=duplicate_texts,
        closure_ordinal_bytes_sha256=closure.ordinal_bytes_sha256,
        exact_text_quotient_count=7,
    )
    assert payload.anonymous_binding()["logical_document_count"] == 8
    assert payload.anonymous_binding()["exact_text_quotient_count"] == 7
    assert payload.worker_payload()["sentence_texts"] == list(duplicate_texts)


def test_hipporag_terminal_maps_logical_positions_to_source_ordinals() -> None:
    result, _recorder = _execute()
    payload = executor.build_candidate_restricted_hipporag_payload(result.actions)
    raw = b"[4,0,7,2,5]\n"
    terminal = executor.parse_candidate_restricted_hipporag_terminal(payload, raw)
    assert terminal.top5_source_ordinals == (4, 0, 7, 2, 5)
    output = terminal.payload()
    assert output["worker_output_sha256"] == hashlib.sha256(raw).hexdigest()
    assert output["closure_ordinal_bytes_sha256"] == (
        result.actions.shared_closure.ordinal_bytes_sha256
    )
    assert output["model_run_count_in_this_adapter"] == 0
    assert output["source_reader_call_count"] == 0
    assert output["network_or_api_call_count"] == 0
    assert output["retry_replay_resample_count"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        b"[0, 1, 2, 3, 4]\n",
        b'{"ordinals":[0,1,2,3,4]}\n',
        b"[0,1,2,3,4]",
        b"[0,1,2,3,3]\n",
        b"[0,1,2,3,8]\n",
        b"[0,1,2,3,true]\n",
        b"[0,1,2,3,4]\ntrailing",
    ],
)
def test_hipporag_terminal_rejects_noncanonical_or_invalid_output(
    raw: bytes,
) -> None:
    result, _recorder = _execute()
    payload = executor.build_candidate_restricted_hipporag_payload(result.actions)
    with pytest.raises(
        executor.MmqaP1LocalActionExecutorError, match="terminal drifted"
    ):
        executor.parse_candidate_restricted_hipporag_terminal(payload, raw)


def test_actions_and_adapter_do_not_mutate_the_validated_item() -> None:
    item = _work_item()
    before = item.anonymous_payload()
    recorder = _Recorder()
    result = executor.execute_local_actions(
        item,
        model_binding=_binding(),
        batch_functions=executor.LocalBatchFunctions(
            recorder.encode, recorder.score
        ),
    )
    payload = executor.build_candidate_restricted_hipporag_payload(result.actions)
    executor.parse_candidate_restricted_hipporag_terminal(
        payload, b"[0,1,2,3,4]\n"
    )
    assert item.anonymous_payload() == before

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from assumption_agent.benchmarks import hitab_p1_dmc1_core_v1 as core
from assumption_agent.benchmarks import hitab_p1_runtime_v1 as runtime
from replication_runtime.birco_official_hipporag_v1 import contract as hippo_contract
from replication_runtime.bright_query_generator_v1 import contract as planner_contract


def _item() -> runtime.RuntimeItem:
    units = tuple(
        f"VALUE type=integer surface={index} | LEFT_PATH Region {index % 2} | "
        f"TOP_PATH Metric > 2024"
        for index in range(10)
    )
    return runtime.RuntimeItem(
        question="Which synthetic regional value is largest in 2024?",
        ordered_unit_strings=units,
        corpus_commitment=runtime.ordered_corpus_commitment(units),
        unit_types=tuple(
            "INTEGER" if index < 8 else "TEXT" for index in range(10)
        ),
        typed_edges=tuple(
            sorted(
                (
                    core.TypedEdge(
                        0, 1, "FORWARD_SHARED_AXIS_OR_HEADER"
                    ),
                    core.TypedEdge(
                        1, 2, "FORWARD_SHARED_AXIS_OR_HEADER"
                    ),
                    core.TypedEdge(
                        8, 9, "FORWARD_SHARED_AXIS_OR_HEADER"
                    ),
                )
            )
        ),
    )


class _Planner:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.calls: list[bytes] = []

    def __call__(self, canonical_input: bytes) -> bytes:
        items = planner_contract.parse_input(canonical_input)
        self.calls.append(canonical_input)
        if self.valid:
            completion = json.dumps(
                {
                    "entity_query": "synthetic regional values",
                    "relation_query": "largest displayed regional value",
                    "mechanism_query": "compare the displayed values",
                    "constraint_query": "restrict comparison to 2024",
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        else:
            completion = "invalid completion"
        row = planner_contract.build_output_item(
            ordinal=0,
            completion=completion,
            completion_token_count=24,
            query=items[0].query,
        )
        return planner_contract.canonical_json_bytes(
            planner_contract.output_payload((row,))
        )


class _CrossEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def __call__(self, pairs):
        rows = tuple(pairs)
        self.calls.append(rows)
        # Each query receives the same strict ordinal ordering.
        return tuple(float(index % 10) - 4.5 for index in range(len(rows)))


def _vector(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest * 12, dtype=np.uint8)[:384].astype(np.float32)
    raw = raw - np.float32(127.5)
    raw /= np.float32(np.linalg.norm(raw.astype(np.float64)))
    return raw


class _Encoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        rows = tuple(texts)
        self.calls.append(rows)
        return np.stack([_vector(text) for text in rows]).astype(np.float32)


class _Hippo:
    def __init__(self, *, synchronize_first_two: bool = False) -> None:
        self.calls: list[tuple[int, int, bytes]] = []
        self._lock = threading.Lock()
        self._active_total = 0
        self._active_by_gpu = {0: 0, 1: 0}
        self.maximum_active_total = 0
        self.maximum_active_by_gpu = {0: 0, 1: 0}
        self._barrier = (
            threading.Barrier(2, timeout=5)
            if synchronize_first_two
            else None
        )

    def __call__(
        self,
        canonical_input: bytes,
        *,
        physical_gpu: int,
        cpu_thread_limit: int,
        launch_ack,
    ) -> bytes:
        value = json.loads(canonical_input.decode("ascii"))
        validated = hippo_contract.validate_input(
            value["work_id"],
            value["objective"],
            value["query"],
            value["documents"],
            value["common_projection_sha256"],
        )
        with self._lock:
            call_index = len(self.calls)
            self.calls.append(
                (physical_gpu, cpu_thread_limit, canonical_input)
            )
            self._active_total += 1
            self._active_by_gpu[physical_gpu] += 1
            self.maximum_active_total = max(
                self.maximum_active_total, self._active_total
            )
            self.maximum_active_by_gpu[physical_gpu] = max(
                self.maximum_active_by_gpu[physical_gpu],
                self._active_by_gpu[physical_gpu],
            )
        try:
            launch_ack()
            if self._barrier is not None and call_index < 2:
                self._barrier.wait()
            candidate_count = len(validated[3])
            payload = hippo_contract.output_payload(
                work_id=validated[0],
                common_projection_sha256=validated[4],
                candidate_count=candidate_count,
                rank_ordinals=tuple(reversed(range(candidate_count))),
                graph_nodes=11,
                graph_edges=10,
            )
            return hippo_contract.canonical_json_bytes(payload)
        finally:
            with self._lock:
                self._active_total -= 1
                self._active_by_gpu[physical_gpu] -= 1


def test_runtime_compiles_all_q6_tensors_and_holds_raw_outside_view() -> None:
    item = _item()
    planner = _Planner()
    scorer = _CrossEncoder()
    encoder = _Encoder()
    compiled = runtime.compile_runtime(
        item,
        planner_runner=planner,
        cross_encoder_scorer=scorer,
        minilm_encoder=encoder,
        physical_gpu=0,
    )
    assert len(planner.calls) == len(scorer.calls) == len(encoder.calls) == 1
    assert compiled.planner.question_facets == (
        item.question,
        "synthetic regional values",
        "largest displayed regional value",
        "compare the displayed values",
        "restrict comparison to 2024",
    )
    assert compiled.planner.inferred_operation == "UNSPECIFIED"
    assert len(scorer.calls[0]) == 5 * len(item.ordered_unit_strings)
    assert scorer.calls[0][0] == (
        item.question,
        item.ordered_unit_strings[0],
    )
    assert encoder.calls[0] == (
        *compiled.planner.question_facets,
        *item.ordered_unit_strings,
    )
    assert len(compiled.view.ce_facet_unit) == 5
    assert len(compiled.view.minilm_facet_unit) == 5
    assert len(compiled.view.minilm_unit_unit) == 10
    assert all(
        -core.QUANT_SCALE <= value <= core.QUANT_SCALE
        for row in compiled.view.minilm_unit_unit
        for value in row
    )
    assert compiled.raw_top5 == (9, 8, 7, 6, 5)
    assert compiled.view.corpus_commitment == item.corpus_commitment
    view_text = str(compiled.view.payload()).casefold()
    assert "raw" not in view_text
    assert "hippo" not in view_text
    assert compiled == runtime.compile_runtime(
        item,
        planner_runner=_Planner(),
        cross_encoder_scorer=_CrossEncoder(),
        minilm_encoder=_Encoder(),
        physical_gpu=0,
    )


def test_invalid_planner_has_one_fallback_and_zero_retry() -> None:
    item = _item()
    planner = _Planner(valid=False)
    scorer = _CrossEncoder()
    encoder = _Encoder()
    compiled = runtime.compile_runtime(
        item,
        planner_runner=planner,
        cross_encoder_scorer=scorer,
        minilm_encoder=encoder,
        physical_gpu=0,
    )
    assert compiled.planner.generation_valid is False
    assert compiled.planner.question_facets == (item.question,)
    assert compiled.planner.inferred_operation == "UNSPECIFIED"
    assert len(planner.calls) == 1
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == len(item.ordered_unit_strings)
    assert len(encoder.calls) == 1


def test_independent_hippo_adapter_uses_same_query_and_units_only() -> None:
    item = _item()
    runner = _Hippo()
    launch_acks: list[int] = []
    action = runtime.run_official_hippo(
        item.question,
        item.ordered_unit_strings,
        runner,
        physical_gpu=1,
        launch_ack=lambda: launch_acks.append(1),
    )
    assert action.top5_ordinals == (9, 8, 7, 6, 5)
    assert action.corpus_commitment == item.corpus_commitment
    assert len(runner.calls) == 1
    assert launch_acks == [1]
    physical_gpu, cpu_threads, raw = runner.calls[0]
    assert physical_gpu == 1
    assert cpu_threads == runtime.CPU_THREAD_LIMIT_PER_GPU_LANE == 4
    value = json.loads(raw.decode("ascii"))
    assert value["query"] == item.question
    assert tuple(row["text"] for row in value["documents"]) == (
        item.ordered_unit_strings
    )
    assert all(
        key not in value
        for key in ("family", "qrel", "raw_rank", "agent", "recipe_id")
    )


def test_hippo_adapter_requires_exactly_one_launch_acknowledgement() -> None:
    item = _item()

    class MissingAck(_Hippo):
        def __call__(self, canonical_input: bytes, **kwargs) -> bytes:
            return super().__call__(
                canonical_input,
                physical_gpu=kwargs["physical_gpu"],
                cpu_thread_limit=kwargs["cpu_thread_limit"],
                launch_ack=lambda: None,
            )

    class RepeatedAck(_Hippo):
        def __call__(self, canonical_input: bytes, **kwargs) -> bytes:
            outer_ack = kwargs["launch_ack"]

            def repeated_but_swallowed() -> None:
                outer_ack()
                try:
                    outer_ack()
                except runtime.HitabP1RuntimeError:
                    pass

            return super().__call__(
                canonical_input,
                physical_gpu=kwargs["physical_gpu"],
                cpu_thread_limit=kwargs["cpu_thread_limit"],
                launch_ack=repeated_but_swallowed,
            )

    for bad_runner in (MissingAck(), RepeatedAck()):
        with pytest.raises(
            runtime.HitabP1RuntimeError, match="acknowledg"
        ):
            runtime.run_official_hippo(
                item.question,
                item.ordered_unit_strings,
                bad_runner,
                physical_gpu=1,
            )


def test_hippo_queue_is_two_fresh_process_lanes_and_forbids_aform() -> None:
    items = tuple(_item() for _ in range(6))
    runner = _Hippo(synchronize_first_two=True)
    actions = runtime.run_official_hippo_queue(
        items, runner, block="A_hold"
    )
    assert len(actions) == len(items)
    assert all(
        action.corpus_commitment == item.corpus_commitment
        for action, item in zip(actions, items)
    )
    assert [row[0] for row in runner.calls].count(0) == 3
    assert [row[0] for row in runner.calls].count(1) == 3
    assert runner.maximum_active_total == 2
    assert runner.maximum_active_by_gpu == {0: 1, 1: 1}
    assert all(
        row[1] == runtime.CPU_THREAD_LIMIT_PER_GPU_LANE
        for row in runner.calls
    )
    with pytest.raises(runtime.HitabP1RuntimeError, match="A_hold or M_search"):
        runtime.run_official_hippo_queue(items, runner, block="A_form")


def test_model_failures_close_without_fallback_or_retry() -> None:
    item = _item()

    class BadEncoder:
        def encode(self, texts):
            return np.zeros((len(texts), 3), dtype=np.float32)

    planner = _Planner()
    scorer = _CrossEncoder()
    with pytest.raises(runtime.HitabP1RuntimeError, match="MiniLM output"):
        runtime.compile_runtime(
            item,
            planner_runner=planner,
            cross_encoder_scorer=scorer,
            minilm_encoder=BadEncoder(),
            physical_gpu=0,
        )
    assert len(planner.calls) == len(scorer.calls) == 1

    with pytest.raises(runtime.HitabP1RuntimeError, match="physical GPU"):
        runtime.run_official_hippo(
            item.question,
            item.ordered_unit_strings,
            _Hippo(),
            physical_gpu=2,
        )

    duplicate_strings = runtime.RuntimeItem(
        question=item.question,
        ordered_unit_strings=tuple(item.ordered_unit_strings[0] for _ in range(10)),
        corpus_commitment=runtime.ordered_corpus_commitment(
            tuple(item.ordered_unit_strings[0] for _ in range(10))
        ),
        unit_types=tuple("INTEGER" for _ in range(10)),
        typed_edges=(),
    )
    assert len(duplicate_strings.ordered_unit_strings) == 10
    # Official content addressing remains unique because the generic pinned
    # adapter serializes each local ordinal beside the possibly repeated text.
    runtime.hippo_input_bytes(
        duplicate_strings.question,
        duplicate_strings.ordered_unit_strings,
    )


def test_public_surface_has_no_source_family_gold_or_baseline_feature_input() -> None:
    assert not set(runtime.RuntimeItem.__dataclass_fields__).intersection(
        {
            "source",
            "split",
            "family",
            "item_id",
            "gold",
            "qrel",
            "raw",
            "hipporag",
            "recipe_id",
        }
    )
    compile_parameters = set(inspect.signature(runtime.compile_runtime).parameters)
    assert not compile_parameters.intersection(
        {"source", "split", "family", "gold", "qrel", "raw", "hipporag"}
    )
    assert "physical_gpu" in inspect.signature(
        runtime.BrightPlannerProductionRunner
    ).parameters
    assert "physical_gpu" in inspect.signature(
        runtime.BrightCrossEncoderProductionScorer
    ).parameters
    assert "physical_gpu" in inspect.signature(
        runtime.bind_bright_minilm_production_encoder
    ).parameters


def test_direct_minilm_v2_topology_and_addendum_are_frozen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    (root / "1_Pooling").mkdir(parents=True)
    rows = {
        "modules.json": runtime._DIRECT_MINILM_MODULES,
        "1_Pooling/config.json": runtime._DIRECT_MINILM_POOLING,
        "sentence_bert_config.json": {
            "do_lower_case": False,
            "max_seq_length": 256,
        },
        "config.json": {
            "architectures": ["BertModel"],
            "hidden_size": runtime.EMBEDDING_DIMENSION,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "pad_token_id": 0,
        },
    }
    for relative, value in rows.items():
        (root / relative).write_text(
            json.dumps(value), encoding="utf-8"
        )
    runtime._validate_direct_minilm_topology(root)
    (root / "1_Pooling/config.json").write_text(
        json.dumps(
            {
                **runtime._DIRECT_MINILM_POOLING,
                "pooling_mode_mean_tokens": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        runtime.HitabP1RuntimeError,
        match="topology drifted",
    ):
        runtime._validate_direct_minilm_topology(root)

    direct_source = inspect.getsource(
        runtime.DirectTransformersMiniLMEncoder.__init__
    )
    assert "from sentence_transformers" not in direct_source
    manifest_path = (
        Path(runtime.__file__).resolve().parents[2]
        / "manifests/hitab_p1_direct_transformers_minilm_addendum_v2.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.pop("self_sha256")
    assert claimed == hashlib.sha256(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()

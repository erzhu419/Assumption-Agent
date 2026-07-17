from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import contractnli_typed_clause_graph_v1 as core
from assumption_agent.benchmarks import synthetic_typed_graph_causal_grammar_v1 as grammar
from assumption_agent.benchmarks import synthetic_typed_graph_multiseed_runner_v1 as runner


# These fixtures are deliberately public and synthetic.  This test module never
# invokes the formal entry point or opens any canonical private artifact.
ACTION_PACK_SCHEMA = "synthetic_typed_graph_multiseed_action_pack_v1"
ACTION_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_action_item_v1"
LABEL_PACK_SCHEMA = "synthetic_typed_graph_multiseed_label_pack_v1"
LABEL_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_label_item_v1"
SEED_COUNT = 8
ITEMS_PER_SEED = 64
TOTAL_ITEMS = SEED_COUNT * ITEMS_PER_SEED
PRIVATE_MODE = 0o600
FIXED_OFFICIAL = (0, 1, 2, 3, 4)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")
    path.chmod(PRIVATE_MODE)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HashEncoder:
    """Small deterministic offline-only encoder used by unit tests."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.frombuffer(digest * 12, dtype=np.uint8).astype(np.float32) + 1.0
            row /= np.linalg.norm(row)
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


class FakeOfficialRuntime:
    """Thread-safe fake with enough latency to observe the cap."""

    def __init__(self, *, drift_postflight: bool = False) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.live = 0
        self.max_live = 0
        self.postflights = 0
        self.drift_postflight = drift_postflight

    @property
    def safe_binding(self) -> Mapping[str, object]:
        return {"runtime": "offline_fake_test_only", "revision": 1}

    def retrieve(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]:
        assert question
        assert len(paragraphs) == grammar.NODE_COUNT
        assert all(
            paragraph.get("title") == "synthetic_typed_graph_causal_v1"
            for paragraph in paragraphs
        )
        assert work_root.name
        with self._lock:
            self.calls += 1
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        try:
            time.sleep(0.001)
            return FIXED_OFFICIAL
        finally:
            with self._lock:
                self.live -= 1

    def fresh_reverify(self) -> Mapping[str, object]:
        with self._lock:
            self.postflights += 1
        if self.drift_postflight:
            return {"runtime": "offline_fake_test_only", "revision": 2}
        return dict(self.safe_binding)


@pytest.fixture(scope="module")
def private_packs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path_factory.mktemp("synthetic-multiseed-private-packs")
    action_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for seed_index in range(SEED_COUNT):
        seed = hashlib.sha256(f"public-unit-seed-{seed_index}".encode("ascii")).digest()
        items = grammar.generate_block(seed, "A_hold")
        assert len(items) == ITEMS_PER_SEED
        for seed_ordinal, item in enumerate(items):
            global_ordinal = seed_index * ITEMS_PER_SEED + seed_ordinal
            action_body = {
                "schema": ACTION_ITEM_SCHEMA,
                "global_ordinal": global_ordinal,
                "seed_index": seed_index,
                "seed_ordinal": seed_ordinal,
                "question": item.question,
                "context": item.context,
                "nodes": [
                    {
                        "span_i": node.span_i,
                        "start": node.start,
                        "end": node.end,
                        "identity_text": node.identity_text,
                    }
                    for node in item.nodes
                ],
                "designated_edges": [
                    {
                        "edge_family": edge.edge_family,
                        "left_span_i": edge.left_span_i,
                        "right_span_i": edge.right_span_i,
                    }
                    for edge in item.designated_edges
                ],
            }
            action = {
                **action_body,
                "action_item_sha256": _semantic_hash(action_body),
            }
            action_rows.append(action)
            label_body = {
                "schema": LABEL_ITEM_SCHEMA,
                "global_ordinal": global_ordinal,
                "seed_index": seed_index,
                "seed_ordinal": seed_ordinal,
                "action_item_sha256": action["action_item_sha256"],
                "gold_node_indices": list(item.gold_node_indices),
                "family_id": item.family_id,
                "family_role": item.family_role,
                "polarity": item.polarity,
                "edge_family": item.edge_family,
            }
            label_rows.append(
                {**label_body, "label_item_sha256": _semantic_hash(label_body)}
            )
    action_body = {
        "schema": ACTION_PACK_SCHEMA,
        "version": runner.DESIGN_VERSION,
        "block": "A_hold",
        "seed_count": SEED_COUNT,
        "item_count_per_seed": ITEMS_PER_SEED,
        "total_item_count": TOTAL_ITEMS,
        "labels_included": False,
        "items": action_rows,
    }
    action_pack = {**action_body, "pack_sha256": _semantic_hash(action_body)}
    label_body = {
        "schema": LABEL_PACK_SCHEMA,
        "version": runner.DESIGN_VERSION,
        "block": "A_hold",
        "seed_count": SEED_COUNT,
        "item_count_per_seed": ITEMS_PER_SEED,
        "total_item_count": TOTAL_ITEMS,
        "items": label_rows,
    }
    label_pack = {**label_body, "pack_sha256": _semantic_hash(label_body)}
    action_path = root / "action_pack.json"
    label_path = root / "label_pack.json"
    _write_private(action_path, action_pack)
    _write_private(label_path, label_pack)
    return action_path, label_path, action_pack, label_pack


@pytest.fixture(scope="module")
def prepared(
    private_packs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> tuple[object, object, tuple[object, ...]]:
    action_path, label_path, _action_json, _label_json = private_packs
    action = runner.load_action_pack(action_path)
    labels = runner.load_label_pack(label_path)
    tensors = runner.precompute_local_tensors(action, HashEncoder())
    return action, labels, tensors


def _summary(
    coordinates: Sequence[int],
    outputs: Sequence[Sequence[int]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    hits = complete = support = total_u = 0
    for coordinate in coordinates:
        gold = tuple(labels[coordinate]["gold_node_indices"])
        item_hits, item_complete, utility = core.item_utility(
            outputs[coordinate], gold, source_count=grammar.NODE_COUNT
        )
        hits += item_hits
        complete += item_complete
        support += len(gold)
        total_u += utility
    return {
        "item_count": len(coordinates),
        "support_hit_count": hits,
        "support_total": support,
        "complete_count": complete,
        "total_U": total_u,
    }


def _expected_aggregates(
    action_json: Mapping[str, Any],
    label_json: Mapping[str, Any],
    tensors: Sequence[object],
) -> dict[str, dict[str, object]]:
    actions = action_json["items"]
    labels = label_json["items"]
    outputs: dict[str, list[tuple[int, ...]]] = {
        "RAW": [],
        "official_HippoRAG": [],
        "Agent_R1": [],
    }
    for action, tensor in zip(actions, tensors):
        spans = tuple(
            core.SourceSpan(
                node["span_i"], node["start"], node["end"], node["identity_text"]
            )
            for node in action["nodes"]
        )
        edges = core.build_typed_clause_graph(spans)
        table = core.build_common_candidate_table(
            spans, edges, FIXED_OFFICIAL, tensor.query_similarities
        )
        trace = core.execute_recipe(
            FIXED_OFFICIAL,
            table,
            tensor.query_similarities,
            "R1_DEFINITION_1SWAP",
        )
        outputs["RAW"].append(tuple(tensor.raw_top5))
        outputs["official_HippoRAG"].append(FIXED_OFFICIAL)
        outputs["Agent_R1"].append(tuple(trace.output_top5))

    all_coordinates = tuple(range(TOTAL_ITEMS))
    result: dict[str, dict[str, object]] = {}
    for arm, arm_outputs in outputs.items():
        by_seed = {
            f"seed_{seed_index:02d}": _summary(
                tuple(
                    range(
                        seed_index * ITEMS_PER_SEED,
                        (seed_index + 1) * ITEMS_PER_SEED,
                    )
                ),
                arm_outputs,
                labels,
            )
            for seed_index in range(SEED_COUNT)
        }
        family_coordinates: dict[str, list[int]] = defaultdict(list)
        polarity_coordinates: dict[str, list[int]] = defaultdict(list)
        for coordinate, label in enumerate(labels):
            family_coordinates[label["family_id"]].append(coordinate)
            polarity_coordinates[label["polarity"]].append(coordinate)
        result[arm] = {
            "overall": _summary(all_coordinates, arm_outputs, labels),
            "by_seed": by_seed,
            "by_family": {
                family_id: _summary(coordinates, arm_outputs, labels)
                for family_id, coordinates in sorted(family_coordinates.items())
            },
            "by_polarity": {
                polarity: _summary(coordinates, arm_outputs, labels)
                for polarity, coordinates in sorted(polarity_coordinates.items())
            },
        }
    return result


def _cluster_summary(
    aggregates: Mapping[str, Mapping[str, object]], left: str, right: str
) -> dict[str, object]:
    left_seeds = aggregates[left]["by_seed"]
    right_seeds = aggregates[right]["by_seed"]
    assert isinstance(left_seeds, Mapping) and isinstance(right_seeds, Mapping)
    values = tuple(
        int(left_seeds[seed_id]["total_U"])
        - int(right_seeds[seed_id]["total_U"])
        for seed_id in sorted(left_seeds)
    )
    comparison = f"{left}_minus_{right}"
    return {
        "comparison": comparison,
        "ordered_seed_deltas": list(values),
        "mean_delta": sum(values) / len(values),
        "median_delta": float(statistics.median(values)),
        "minimum_delta": min(values),
        "maximum_delta": max(values),
        "range_delta": max(values) - min(values),
        "K_positive": sum(value > 0 for value in values),
    }


def test_private_pack_loaders_bind_exact_8x64_shape_and_commitments(
    private_packs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
) -> None:
    action_path, label_path, action_json, label_json = private_packs
    action = runner.load_action_pack(action_path)
    labels = runner.load_label_pack(label_path)
    assert len(action.rows) == len(labels.rows) == TOTAL_ITEMS
    assert action.file_sha256 == _file_sha256(action_path)
    assert labels.file_sha256 == _file_sha256(label_path)
    assert action.pack_sha256 == action_json["pack_sha256"]
    assert labels.pack_sha256 == label_json["pack_sha256"]
    assert action.item_commitment_set_sha256 == _semantic_hash(
        [row["action_item_sha256"] for row in action_json["items"]]
    )
    assert labels.item_commitment_set_sha256 == _semantic_hash(
        [row["label_item_sha256"] for row in label_json["items"]]
    )
    assert Counter(row.seed_index for row in action.rows) == {index: 64 for index in range(8)}
    assert tuple(row.global_ordinal for row in action.rows) == tuple(range(TOTAL_ITEMS))
    assert all(len(row.nodes) == grammar.NODE_COUNT for row in action.rows)
    assert tuple(row.action_item_sha256 for row in action.rows) == tuple(
        row.action_item_sha256 for row in labels.rows
    )
    serialized_action = json.dumps(action_json, sort_keys=True)
    for forbidden in (
        "gold_node_indices",
        "family_id",
        "family_role",
        "polarity",
        "label_item_sha256",
    ):
        assert forbidden not in serialized_action


def test_official_paragraph_title_preserves_original_causal_semantics(
    prepared: tuple[object, object, tuple[object, ...]],
) -> None:
    action, _labels, _tensors = prepared
    assert isinstance(action, runner.ActionPack)
    assert {
        paragraph["title"]
        for item in action.rows
        for paragraph in item.paragraphs
    } == {"synthetic_typed_graph_causal_v1"}


def test_pack_loaders_reject_mode_noncanonical_shape_unknown_keys_and_hash_tampering(
    private_packs: tuple[Path, Path, dict[str, Any], dict[str, Any]], tmp_path: Path
) -> None:
    action_path, _label_path, action_json, _label_json = private_packs

    public = tmp_path / "public.json"
    public.write_bytes(action_path.read_bytes())
    public.chmod(0o644)
    with pytest.raises(runner.SyntheticTypedGraphMultiseedRunnerError, match="mode|private"):
        runner.load_action_pack(public)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(action_json, indent=2), encoding="ascii")
    noncanonical.chmod(PRIVATE_MODE)
    with pytest.raises(runner.SyntheticTypedGraphMultiseedRunnerError, match="canonical"):
        runner.load_action_pack(noncanonical)

    short = json.loads(json.dumps(action_json))
    short["items"].pop()
    body = dict(short)
    body.pop("pack_sha256")
    short["pack_sha256"] = _semantic_hash(body)
    short_path = tmp_path / "short.json"
    _write_private(short_path, short)
    with pytest.raises(
        runner.SyntheticTypedGraphMultiseedRunnerError, match="512|shape|count|schema"
    ):
        runner.load_action_pack(short_path)

    unknown = json.loads(json.dumps(action_json))
    unknown["items"][0]["not_frozen"] = True
    item_body = dict(unknown["items"][0])
    item_body.pop("action_item_sha256")
    unknown["items"][0]["action_item_sha256"] = _semantic_hash(item_body)
    pack_body = dict(unknown)
    pack_body.pop("pack_sha256")
    unknown["pack_sha256"] = _semantic_hash(pack_body)
    unknown_path = tmp_path / "unknown.json"
    _write_private(unknown_path, unknown)
    with pytest.raises(
        runner.SyntheticTypedGraphMultiseedRunnerError, match="keys|schema|field set"
    ):
        runner.load_action_pack(unknown_path)

    tampered = json.loads(json.dumps(action_json))
    tampered["items"][0]["question"] += " tampered"
    pack_body = dict(tampered)
    pack_body.pop("pack_sha256")
    tampered["pack_sha256"] = _semantic_hash(pack_body)
    tampered_path = tmp_path / "tampered.json"
    _write_private(tampered_path, tampered)
    with pytest.raises(
        runner.SyntheticTypedGraphMultiseedRunnerError,
        match="item hash|action_item_sha256",
    ):
        runner.load_action_pack(tampered_path)


def test_three_arm_wave_is_eager_capped_label_late_and_aggregates_exactly(
    private_packs: tuple[Path, Path, dict[str, Any], dict[str, Any]],
    prepared: tuple[object, object, tuple[object, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _action_path, _label_path, action_json, label_json = private_packs
    action, labels, tensors = prepared
    real_executor = ThreadPoolExecutor
    submit_events: list[int] = []
    executor_instances: list[object] = []

    class RecordingExecutor(real_executor):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.recorded_max_workers = int(self._max_workers)
            self.recorded_submits = 0
            executor_instances.append(self)

        def submit(self, fn: object, /, *args: object, **kwargs: object) -> Future[object]:
            self.recorded_submits += 1
            submit_events.append(self.recorded_max_workers)
            return super().submit(fn, *args, **kwargs)

    # Tensor construction is a separate offline preparation step.  Auditing the
    # executors below therefore observes exactly the three action arms.
    monkeypatch.setattr(runner, "precompute_local_tensors", lambda _pack, _encoder: tensors)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", RecordingExecutor)
    real_future_result = Future.result
    main_thread = threading.get_ident()
    submit_counts_at_main_join: list[int] = []
    submit_counts_at_worker_join: list[int] = []

    def audited_result(self: Future[object], *args: object, **kwargs: object) -> object:
        if threading.get_ident() == main_thread:
            submit_counts_at_main_join.append(len(submit_events))
        else:
            submit_counts_at_worker_join.append(len(submit_events))
        return real_future_result(self, *args, **kwargs)

    monkeypatch.setattr(Future, "result", audited_result)
    runtime = FakeOfficialRuntime()
    seal_path = tmp_path / "actions.sealed.json"
    label_opens: list[tuple[int, int]] = []

    def load_labels() -> object:
        label_opens.append((runtime.calls, runtime.postflights))
        assert seal_path.is_file()
        assert seal_path.stat().st_mode & 0o777 == PRIVATE_MODE
        seal = json.loads(seal_path.read_text(encoding="ascii"))
        assert seal["action_work_unit_count"] == 1536
        assert seal["official_retrieve_action_count"] == 512
        assert seal["RAW_action_count"] == 512
        assert seal["Agent_R1_action_count"] == 512
        assert seal["labels_opened_before_action_seal"] is False
        seal_body = dict(seal)
        declared_seal_hash = seal_body.pop("action_seal_sha256")
        assert declared_seal_hash == _semantic_hash(seal_body)
        return labels

    outcome = runner.run_multiseed_replication(
        action,
        label_loader=load_labels,
        encoder=HashEncoder(),
        runtime=runtime,
        work_root=tmp_path / "official-work",
        action_seal_path=seal_path,
    )
    assert runtime.calls == 512
    assert runtime.postflights == 1
    assert runtime.max_live == runner.OFFICIAL_CONCURRENCY_CAP == 8
    assert runner.LOCAL_CONCURRENCY_CAP == 64
    assert label_opens == [(512, 1)]
    assert Counter(submit_events) == {8: 512, 64: 1024}
    assert submit_counts_at_main_join
    assert min(submit_counts_at_main_join) == 1536
    assert len(submit_counts_at_worker_join) == TOTAL_ITEMS
    assert min(submit_counts_at_worker_join) == 1536
    assert sum(instance.recorded_submits for instance in executor_instances) == 1536

    expected = _expected_aggregates(action_json, label_json, tensors)
    assert outcome.aggregates == expected
    assert outcome.cluster_differences == {
        "Agent_R1_minus_official_HippoRAG": _cluster_summary(
            expected, "Agent_R1", "official_HippoRAG"
        ),
        "Agent_R1_minus_RAW": _cluster_summary(expected, "Agent_R1", "RAW"),
    }
    public = runner.multiseed_public_result(outcome)
    assert public["aggregates"] == expected
    assert public["cluster_differences"] == outcome.cluster_differences
    public_body = dict(public)
    declared_result_hash = public_body.pop("receipt_sha256")
    assert declared_result_hash == _semantic_hash(public_body)
    serialized = json.dumps(public, sort_keys=True).casefold()
    assert "gold_node_indices" not in serialized
    assert "action_rows" not in serialized
    for forbidden in (
        "threshold",
        "p_value",
        "pvalue",
        "pass_fail",
        "passfail",
        "promotion",
        "gate",
        "m_search",
    ):
        assert re.search(rf"(?<![a-z0-9_]){re.escape(forbidden)}(?![a-z0-9_])", serialized) is None


def test_incomplete_submission_releases_agent_workers_without_joining_official(
    prepared: tuple[object, object, tuple[object, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action, _labels, tensors = prepared
    assert isinstance(action, runner.ActionPack)
    real_executor = ThreadPoolExecutor
    real_agent_action = runner._agent_action
    real_future_result = Future.result
    agent_started = threading.Event()
    main_thread = threading.get_ident()
    worker_future_result_calls = 0

    def observed_agent_action(*args: object) -> object:
        agent_started.set()
        return real_agent_action(*args)  # type: ignore[arg-type]

    class FailingSubmitExecutor(real_executor):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.local_submit_count = 0

        def submit(self, fn: object, /, *args: object, **kwargs: object) -> Future[object]:
            self.local_submit_count += 1
            if (
                self._max_workers == runner.LOCAL_CONCURRENCY_CAP
                and self.local_submit_count == TOTAL_ITEMS + 2
            ):
                assert agent_started.wait(timeout=5)
                raise RuntimeError("deliberate action submit failure")
            return super().submit(fn, *args, **kwargs)

    def audited_result(self: Future[object], *args: object, **kwargs: object) -> object:
        nonlocal worker_future_result_calls
        if threading.get_ident() != main_thread:
            worker_future_result_calls += 1
        return real_future_result(self, *args, **kwargs)

    monkeypatch.setattr(runner, "precompute_local_tensors", lambda _pack, _encoder: tensors)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", FailingSubmitExecutor)
    monkeypatch.setattr(runner, "_agent_action", observed_agent_action)
    monkeypatch.setattr(Future, "result", audited_result)
    label_opens = 0

    def forbidden_labels() -> object:
        nonlocal label_opens
        label_opens += 1
        raise AssertionError("labels opened after incomplete action submission")

    seal_path = tmp_path / "incomplete-submit.seal.json"
    with pytest.raises(RuntimeError, match="deliberate action submit failure"):
        runner.run_multiseed_replication(
            action,
            label_loader=forbidden_labels,
            encoder=HashEncoder(),
            runtime=FakeOfficialRuntime(),
            work_root=tmp_path / "incomplete-submit-work",
            action_seal_path=seal_path,
        )
    assert agent_started.is_set()
    assert worker_future_result_calls == 0
    assert label_opens == 0
    assert not seal_path.exists()


def test_postflight_failure_never_opens_labels_and_same_root_cannot_replay(
    prepared: tuple[object, object, tuple[object, ...]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action, _labels, tensors = prepared
    monkeypatch.setattr(runner, "precompute_local_tensors", lambda _pack, _encoder: tensors)
    runtime = FakeOfficialRuntime(drift_postflight=True)
    work_root = tmp_path / "failed-official-work"
    seal_path = tmp_path / "must-not-exist.seal.json"
    label_opens = 0

    def forbidden_label_loader() -> object:
        nonlocal label_opens
        label_opens += 1
        raise AssertionError("labels opened after an invalid postflight")

    with pytest.raises(
        runner.SyntheticTypedGraphMultiseedRunnerError, match="postflight|binding"
    ):
        runner.run_multiseed_replication(
            action,
            label_loader=forbidden_label_loader,
            encoder=HashEncoder(),
            runtime=runtime,
            work_root=work_root,
            action_seal_path=seal_path,
        )
    calls_after_failure = runtime.calls
    assert calls_after_failure == 512
    assert runtime.postflights == 1
    assert label_opens == 0
    assert not seal_path.exists()

    with pytest.raises(runner.SyntheticTypedGraphMultiseedRunnerError, match="exists|replay"):
        runner.run_multiseed_replication(
            action,
            label_loader=forbidden_label_loader,
            encoder=HashEncoder(),
            runtime=runtime,
            work_root=work_root,
            action_seal_path=seal_path,
        )
    assert runtime.calls == calls_after_failure
    assert runtime.postflights == 1
    assert label_opens == 0

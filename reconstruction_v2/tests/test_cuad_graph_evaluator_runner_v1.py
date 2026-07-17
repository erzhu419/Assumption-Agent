from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from assumption_agent.models import stable_hash
from assumption_agent.benchmarks import cuad_graph_evaluator_runner_v1 as r


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _node_texts(with_edge: bool = True) -> tuple[str, ...]:
    if with_edge:
        return (
            '"Alpha" means the premium service.',
            "Beta ordinary provision.",
            "Gamma ordinary provision.",
            "Delta ordinary provision.",
            "Epsilon ordinary provision.",
            "Alpha target evidence.",
        )
    return tuple(f"Unrelated sentence {index}." for index in range(6))


def _view_item(block: str, ordinal: int, *, with_edge: bool = True) -> r.LabelFreeItem:
    nodes: list[r.PrivateNode] = []
    cursor = 0
    for span_i, text in enumerate(_node_texts(with_edge)):
        nodes.append(r.PrivateNode(span_i, cursor, cursor + len(text), text))
        cursor += len(text) + 1
    return r.LabelFreeItem(
        block=block,
        ordinal=ordinal,
        item_commitment_sha256=_digest(f"item:{block}:{ordinal}"),
        component_commitment_sha256=_digest(f"component:{block}:{ordinal}"),
        question=f"Find Alpha target evidence for synthetic item {ordinal}.",
        nodes=tuple(nodes),
    )


def _view_block(block: str, *, with_edge: bool = True) -> r.LabelFreeBlock:
    return r.LabelFreeBlock(
        block=block,
        block_sha256=_digest(f"view-block:{block}"),
        file_sha256=_digest(f"view-file:{block}"),
        rows=tuple(_view_item(block, ordinal, with_edge=with_edge) for ordinal in range(64)),
    )


def _label_block(block: str, *, gold: tuple[int, ...] = (5,)) -> r.LabelBlock:
    return r.LabelBlock(
        block=block,
        block_sha256=_digest(f"label-block:{block}"),
        file_sha256=_digest(f"label-file:{block}"),
        rows=tuple(
            r.LabelItem(
                block=block,
                ordinal=ordinal,
                item_commitment_sha256=_digest(f"item:{block}:{ordinal}"),
                gold_node_indices=gold,
            )
            for ordinal in range(64)
        ),
    )


class SyntheticEncoder:
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def encode(self, texts):
        self.call_sizes.append(len(texts))
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for row_i, text in enumerate(texts):
            if text.startswith("Find Alpha") or text == "Alpha target evidence.":
                dimension = 0
            elif text.startswith('"Alpha" means'):
                dimension = 1
            elif text.startswith("Beta"):
                dimension = 2
            elif text.startswith("Gamma"):
                dimension = 3
            elif text.startswith("Delta"):
                dimension = 4
            elif text.startswith("Epsilon"):
                dimension = 5
            else:
                dimension = 10 + (row_i % 100)
            matrix[row_i, dimension] = 1.0
        return matrix


class SyntheticRuntime:
    def __init__(self, *, delay: float = 0.001, fail_once: bool = False) -> None:
        body = {"schema": "synthetic_official_runtime", "status": "verified"}
        self._safe_binding = {**body, "binding_sha256": stable_hash(body)}
        self.delay = delay
        self.fail_once = fail_once
        self._failed = False
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.retrieve_count = 0
        self.payload_key_sets: list[frozenset[str]] = []
        self.postflight_count = 0

    @property
    def safe_binding(self):
        return dict(self._safe_binding)

    def retrieve(self, *, question, paragraphs, work_root):
        assert isinstance(question, str)
        assert all(set(row) == {"idx", "title", "paragraph_text"} for row in paragraphs)
        assert all(row["title"] == "CUAD_contract" for row in paragraphs)
        assert "gold" not in json.dumps(paragraphs).casefold()
        assert isinstance(work_root, Path)
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.retrieve_count += 1
            should_fail = self.fail_once and not self._failed
            if should_fail:
                self._failed = True
        try:
            time.sleep(self.delay)
            if should_fail:
                raise RuntimeError("synthetic official failure with private sentinel")
            return (0, 1, 2, 3, 4)
        finally:
            with self._lock:
                self.active -= 1

    def fresh_reverify(self):
        self.postflight_count += 1
        return dict(self._safe_binding)


def _row_payload(item: r.LabelFreeItem) -> dict[str, object]:
    return {
        "schema": r.LABEL_FREE_ITEM_SCHEMA,
        "block": item.block,
        "ordinal": item.ordinal,
        "item_commitment_sha256": item.item_commitment_sha256,
        "component_commitment_sha256": item.component_commitment_sha256,
        "question": item.question,
        "title": "CUAD_contract",
        "nodes": [
            {
                "span_i": node.span_i,
                "start": node.start,
                "end": node.end,
                "identity_text": node.identity_text,
            }
            for node in item.nodes
        ],
    }


def _write_private(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)


def _write_view_pack(path: Path, block: str) -> None:
    body: dict[str, object] = {
        "schema": r.LABEL_FREE_SCHEMA,
        "block": block,
        "count": 64,
        "rows": [_row_payload(_view_item(block, ordinal)) for ordinal in range(64)],
    }
    _write_private(path, {**body, "block_sha256": stable_hash(body)})


def _write_label_pack(path: Path, block: str, *, gold=(5,)) -> None:
    body: dict[str, object] = {
        "schema": r.LABEL_SCHEMA,
        "block": block,
        "count": 64,
        "rows": [
            {
                "schema": r.LABEL_ITEM_SCHEMA,
                "block": block,
                "ordinal": ordinal,
                "item_commitment_sha256": _digest(f"item:{block}:{ordinal}"),
                "gold_node_indices": list(gold),
            }
            for ordinal in range(64)
        ],
    }
    _write_private(path, {**body, "block_sha256": stable_hash(body)})


@pytest.fixture(scope="module")
def formation_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("cuad-formation")
    encoder = SyntheticEncoder()
    runtime = SyntheticRuntime(delay=0.002)
    events: list[tuple[str, str | None, int | None]] = []
    lock = threading.Lock()

    def progress(event, block, ordinal):
        with lock:
            events.append((event, block, ordinal))

    label_calls = []

    def load_labels():
        assert runtime.retrieve_count == 128
        assert runtime.postflight_count == 1
        assert sum(event[0] == "action_terminal" for event in events) == 128
        label_calls.append(True)
        return _label_block("A_form")

    outcome = r.run_formation_wave(
        _view_block("A_form"),
        _view_block("F_search"),
        a_label_loader=load_labels,
        encoder=encoder,
        runtime=runtime,
        work_root=root / "work",
        progress=progress,
    )
    return outcome, encoder, runtime, events, label_calls


def test_final_design_and_frozen_core_bindings() -> None:
    design = r.verify_design_binding(PROJECT_ROOT)
    assert design["design_sha256"] == r.DESIGN_SHA256
    assert r.DESIGN_FILE_SHA256 == (
        "3c85a6949d18408013e2e8e9da0f140b16da434e63a7a053924532525163052c"
    )
    assert len(r.RECIPE_IDS) == 9
    assert len(r.EVALUATOR_IDS) == 16


def test_stage_specific_private_loaders_exact_schema_and_no_f_labels(tmp_path: Path) -> None:
    a_view = tmp_path / "a.view.json"
    a_labels = tmp_path / "a.labels.json"
    f_view = tmp_path / "f.view.json"
    _write_view_pack(a_view, "A_form")
    _write_label_pack(a_labels, "A_form")
    _write_view_pack(f_view, "F_search")
    loaded_a = r.load_a_form_view(a_view)
    loaded_labels = r.load_a_form_labels(a_labels)
    loaded_f = r.load_f_search_view(f_view)
    assert len(loaded_a.rows) == len(loaded_labels.rows) == len(loaded_f.rows) == 64
    assert not hasattr(r, "load_f_search_labels")
    assert all(
        left.item_commitment_sha256 == right.item_commitment_sha256
        for left, right in zip(loaded_a.rows, loaded_labels.rows)
    )


def test_private_loader_rejects_tamper_and_non_private_mode(tmp_path: Path) -> None:
    path = tmp_path / "view.json"
    _write_view_pack(path, "A_form")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["question"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(r.CuadGraphEvaluatorRunnerError, match="self-hash"):
        r.load_a_form_view(path)
    _write_view_pack(path, "A_form")
    path.chmod(0o644)
    with pytest.raises(r.CuadGraphEvaluatorRunnerError, match="mode or size"):
        r.load_a_form_view(path)


def test_formation_completion_pipeline_cap8_gold_order_and_selection(formation_result) -> None:
    outcome, encoder, runtime, events, label_calls = formation_result
    assert encoder.call_sizes == [64 * 7, 64 * 7]
    assert runtime.retrieve_count == 128
    assert 2 <= runtime.maximum_active <= r.OFFICIAL_CONCURRENCY_CAP == 8
    assert runtime.postflight_count == 1
    assert label_calls == [True]
    assert events.index(("labels_open", "A_form", None)) > max(
        index for index, event in enumerate(events) if event[0] == "action_terminal"
    )
    assert outcome.f_selection.recipe_id == "R1_DEFINITION_1SWAP"
    assert outcome.identifiable_transition is True
    assert outcome.a_arm_aggregates["official_HippoRAG"]["total_U"] == 0
    assert outcome.a_arm_aggregates["Agent"]["total_U"] == 128


def test_all_nine_recipes_and_sixteen_evaluators_are_in_action_hash(formation_result) -> None:
    outcome, _encoder, _runtime, _events, _labels = formation_result
    assert len(outcome.a_selection.evaluator_results) == 16
    assert outcome.a_selection.evaluator_results[0].coverage_comparisons == 64 * 9
    assert outcome.f_selection.coverage_comparisons == 64 * 9
    assert len(outcome.action_table_sha256) == 64


def test_same_behavior_is_terminal_without_runner_up(formation_result) -> None:
    outcome = formation_result[0]
    terminal = replace(outcome, identifiable_transition=False)
    receipt = r.formation_public_receipt(terminal)
    assert receipt["status"] == "terminal_unidentifiable_transition"
    assert receipt["A_hold_authorized"] is False
    assert "runner_up" not in json.dumps(receipt).casefold()


def test_a_hold_runs_only_r0_and_agent_then_exact_promotion(tmp_path: Path, monkeypatch) -> None:
    runtime = SyntheticRuntime(delay=0.001)
    encoder = SyntheticEncoder()
    seen_recipes: list[str] = []
    lock = threading.Lock()
    original = r.execute_recipe

    def tracked(*args, **kwargs):
        recipe_id = args[3] if len(args) > 3 else kwargs["recipe_id"]
        with lock:
            seen_recipes.append(recipe_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(r, "execute_recipe", tracked)
    label_calls: list[bool] = []

    def labels():
        assert runtime.retrieve_count == 64
        assert runtime.postflight_count == 1
        label_calls.append(True)
        return _label_block("A_hold")

    outcome = r.run_measurement_wave(
        _view_block("A_hold"),
        selected_recipe_id="R1_DEFINITION_1SWAP",
        label_loader=labels,
        encoder=encoder,
        runtime=runtime,
        work_root=tmp_path / "work",
    )
    assert label_calls == [True]
    assert seen_recipes.count(r.R0) == 64
    assert seen_recipes.count("R1_DEFINITION_1SWAP") == 64
    assert set(seen_recipes) == {r.R0, "R1_DEFINITION_1SWAP"}
    assert outcome.arm_aggregates["official_HippoRAG"]["total_U"] == 0
    assert outcome.arm_aggregates["Agent"]["total_U"] == 128
    assert outcome.exact_test["observed_net_U"] == 128
    assert outcome.exact_test["promoted"] is True
    assert outcome.exact_test["p_value_numerator"] == 1
    assert outcome.exact_test["p_value_denominator"] == 2**64


def test_public_receipt_is_aggregate_and_redacted(tmp_path: Path) -> None:
    runtime = SyntheticRuntime()
    outcome = r.run_measurement_wave(
        _view_block("A_hold"),
        selected_recipe_id="R1_DEFINITION_1SWAP",
        label_loader=lambda: _label_block("A_hold"),
        encoder=SyntheticEncoder(),
        runtime=runtime,
        work_root=tmp_path / "work",
    )
    receipt = r.measurement_public_receipt(outcome)
    rendered = json.dumps(receipt, sort_keys=True)
    for forbidden in (
        "Find Alpha",
        "Alpha target evidence",
        "identity_text",
        "gold_node_indices",
        "item_commitment_sha256",
        "component_commitment_sha256",
    ):
        assert forbidden not in rendered
    body = dict(receipt)
    declared = body.pop("receipt_sha256")
    assert stable_hash(body) == declared


def test_m_stays_sealed_without_a_hold_promotion(tmp_path: Path) -> None:
    calls: list[str] = []

    def view():
        calls.append("view")
        return _view_block("M_search")

    def labels():
        calls.append("labels")
        return _label_block("M_search")

    with pytest.raises(r.CuadGraphEvaluatorRunnerError, match="sealed"):
        r.run_m_if_authorized(
            authorized=False,
            view_loader=view,
            label_loader=labels,
            selected_recipe_id="R1_DEFINITION_1SWAP",
            encoder=SyntheticEncoder(),
            runtime=SyntheticRuntime(),
            work_root=tmp_path / "never-created",
        )
    assert calls == []
    assert not (tmp_path / "never-created").exists()


def test_official_failure_never_opens_gold(tmp_path: Path) -> None:
    label_calls: list[bool] = []
    with pytest.raises(RuntimeError, match="synthetic official failure"):
        r.run_measurement_wave(
            _view_block("A_hold"),
            selected_recipe_id="R1_DEFINITION_1SWAP",
            label_loader=lambda: label_calls.append(True) or _label_block("A_hold"),
            encoder=SyntheticEncoder(),
            runtime=SyntheticRuntime(fail_once=True),
            work_root=tmp_path / "work",
        )
    assert label_calls == []


def test_one_shot_marker_and_redacted_failure_receipt(tmp_path: Path) -> None:
    bad_a = tmp_path / "bad-a.json"
    bad_a.write_text("not-json private sentinel question", encoding="utf-8")
    bad_a.chmod(0o600)
    f_view = tmp_path / "f.json"
    a_labels = tmp_path / "a-labels.json"
    _write_view_pack(f_view, "F_search")
    _write_label_pack(a_labels, "A_form")
    stage_root = tmp_path / "stage"
    failure = tmp_path / "failure.json"
    with pytest.raises(r.CuadGraphEvaluatorRunnerError):
        r.execute_formation_stage(
            project_root=PROJECT_ROOT,
            a_view_path=bad_a,
            a_label_path=a_labels,
            f_view_path=f_view,
            stage_root=stage_root,
            receipt_path=tmp_path / "receipt.json",
            failure_receipt_path=failure,
            encoder=SyntheticEncoder(),
            runtime=SyntheticRuntime(),
        )
    marker = stage_root / "formation.attempt.marker"
    assert marker.exists()
    payload = json.loads(failure.read_text(encoding="ascii"))
    assert payload["status"] == "terminal_infrastructure_invalid_no_replay"
    rendered = json.dumps(payload)
    assert "private sentinel" not in rendered
    assert str(bad_a) not in rendered
    with pytest.raises(FileExistsError):
        r.consume_stage_marker(marker, "formation")

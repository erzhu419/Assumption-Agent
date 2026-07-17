from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from assumption_agent.benchmarks import evidencebench_graph_evaluator_runner_v1 as r


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_protocol_binding() -> dict[str, object]:
    return {
        "implementation_freeze_sha256": _digest("freeze-semantic"),
        "implementation_freeze_file_sha256": _digest("freeze-file"),
        "git_HEAD": hashlib.sha1(b"synthetic-head").hexdigest(),
        "git_verification": {
            "HEAD": hashlib.sha1(b"synthetic-head").hexdigest(),
            "commands": [],
            "source_secret_private_or_current_stage_output_path_passed": False,
        },
        "design_sha256": _digest("design-semantic"),
        "design_file_sha256": _digest("design-file"),
        "custody_sha256": _digest("custody-semantic"),
        "custody_file_sha256": _digest("custody-file"),
        "source_access_sha256": _digest("access-semantic"),
        "source_access_file_sha256": _digest("access-file"),
        "graph_core_sha256": _digest("core-file"),
        "acquisition_runner_sha256": _digest("acquisition-file"),
        "evaluator_runner_sha256": _digest("runner-file"),
    }


def _node_texts() -> tuple[str, ...]:
    rows = [f"Ordinary unique bucket {index:02d}." for index in range(32)]
    rows[5] = "TargetBridge evidence establishes the synthetic result."
    rows[6] = "Alternative bucket for the synthetic result."
    rows[7] = "Second aspect primary evidence."
    rows[8] = "Second aspect alternative evidence."
    return tuple(rows)


def _view_item(block: str, ordinal: int) -> r.LabelFreeItem:
    nodes: list[r.PrivateNode] = []
    cursor = 0
    for span_i, identity_text in enumerate(_node_texts()):
        nodes.append(
            r.PrivateNode(
                span_i,
                cursor,
                cursor + len(identity_text),
                identity_text,
            )
        )
        cursor += len(identity_text) + 1
    return r.LabelFreeItem(
        block=block,
        ordinal=ordinal,
        item_commitment_sha256=_digest(f"item:{block}:{ordinal}"),
        paper_commitment_sha256=_digest(f"paper:{block}:{ordinal}"),
        component_commitment_sha256=_digest(f"component:{block}:{ordinal}"),
        hypothesis=f"Find TargetBridge evidence for synthetic item {ordinal}.",
        nodes=tuple(nodes),
    )


def _view_block(block: str) -> r.LabelFreeBlock:
    return r.LabelFreeBlock(
        block=block,
        block_sha256=_digest(f"view-block:{block}"),
        file_sha256=_digest(f"view-file:{block}"),
        rows=tuple(_view_item(block, ordinal) for ordinal in range(r.BLOCK_COUNT)),
    )


def _label_block(
    block: str,
    *,
    aspects: tuple[tuple[int, ...], ...] = ((5, 6),),
) -> r.LabelBlock:
    return r.LabelBlock(
        block=block,
        block_sha256=_digest(f"label-block:{block}"),
        file_sha256=_digest(f"label-file:{block}"),
        rows=tuple(
            r.LabelItem(
                block=block,
                ordinal=ordinal,
                item_commitment_sha256=_digest(f"item:{block}:{ordinal}"),
                gold_aspect_node_indices=aspects,
            )
            for ordinal in range(r.BLOCK_COUNT)
        ),
    )


class SyntheticEncoder:
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def encode(self, texts):
        self.call_sizes.append(len(texts))
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for row_i, text in enumerate(texts):
            if text.startswith("Find TargetBridge") or text.startswith(
                "TargetBridge evidence"
            ):
                dimension = 0
            else:
                dimension = 1 + (
                    int.from_bytes(
                        hashlib.sha256(text.encode("utf-8")).digest()[:2],
                        "big",
                    )
                    % 350
                )
            matrix[row_i, dimension] = 1.0
        return matrix


class SyntheticRuntime:
    def __init__(self, *, delay: float = 0.001, fail_once: bool = False) -> None:
        body = {"schema": "synthetic_official_runtime", "status": "verified"}
        self._safe_binding = {**body, "binding_sha256": _canonical_hash(body)}
        self.delay = delay
        self.fail_once = fail_once
        self._failed = False
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.retrieve_count = 0
        self.postflight_count = 0

    @property
    def safe_binding(self):
        return dict(self._safe_binding)

    def retrieve(self, *, question, paragraphs, work_root):
        assert isinstance(question, str)
        assert isinstance(work_root, Path)
        assert len(paragraphs) == 32
        assert all(
            set(row) == {"idx", "title", "paragraph_text"}
            for row in paragraphs
        )
        assert all(row["title"] == "EvidenceBench_paper" for row in paragraphs)
        rendered = json.dumps(paragraphs, sort_keys=True).casefold()
        assert "gold_aspect" not in rendered
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
                raise RuntimeError(
                    "synthetic official failure with PRIVATE_SENTINEL_934"
                )
            return (0, 1, 2, 3, 4)
        finally:
            with self._lock:
                self.active -= 1

    def fresh_reverify(self):
        self.postflight_count += 1
        return dict(self._safe_binding)


def _view_row_payload(item: r.LabelFreeItem) -> dict[str, object]:
    return {
        "schema": r.LABEL_FREE_ITEM_SCHEMA,
        "block": item.block,
        "ordinal": item.ordinal,
        "item_commitment_sha256": item.item_commitment_sha256,
        "paper_commitment_sha256": item.paper_commitment_sha256,
        "component_commitment_sha256": item.component_commitment_sha256,
        "hypothesis": item.hypothesis,
        "title": "EvidenceBench_paper",
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
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_view_pack(path: Path, block: str) -> None:
    body: dict[str, object] = {
        "schema": r.LABEL_FREE_SCHEMA,
        "block": block,
        "count": r.BLOCK_COUNT,
        "rows": [
            _view_row_payload(_view_item(block, ordinal))
            for ordinal in range(r.BLOCK_COUNT)
        ],
    }
    _write_private(path, {**body, "block_sha256": _canonical_hash(body)})


def _write_label_pack(
    path: Path,
    block: str,
    *,
    aspects: tuple[tuple[int, ...], ...] = ((5, 6),),
) -> None:
    body: dict[str, object] = {
        "schema": r.LABEL_SCHEMA,
        "block": block,
        "count": r.BLOCK_COUNT,
        "rows": [
            {
                "schema": r.LABEL_ITEM_SCHEMA,
                "block": block,
                "ordinal": ordinal,
                "item_commitment_sha256": _digest(f"item:{block}:{ordinal}"),
                "gold_aspect_node_indices": [list(row) for row in aspects],
            }
            for ordinal in range(r.BLOCK_COUNT)
        ],
    }
    _write_private(path, {**body, "block_sha256": _canonical_hash(body)})


@pytest.fixture(scope="module")
def formation_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidencebench-formation")
    encoder = SyntheticEncoder()
    runtime = SyntheticRuntime(delay=0.002)
    events: list[tuple[str, str | None, int | None]] = []
    lock = threading.Lock()

    def progress(event, block, ordinal):
        with lock:
            events.append((event, block, ordinal))

    label_calls: list[bool] = []

    def load_labels():
        assert runtime.retrieve_count == 2 * r.BLOCK_COUNT
        assert runtime.postflight_count == 1
        assert sum(event[0] == "action_terminal" for event in events) == (
            2 * r.BLOCK_COUNT
        )
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


def test_external_freeze_interface_is_centralized_and_fail_closed() -> None:
    assert len(r.RECIPE_IDS) == 9
    assert len(r.EVALUATOR_IDS) == 16
    assert r.EXPECTED_BINDING_INTERFACES["acquisition_runner"][
        "relative_path"
    ].endswith(
        "evidencebench_direct_acquisition_v1.py"
    )
    assert r.IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix() == (
        "manifests/evidencebench_implementation_freeze_v1.json"
    )
    assert set(r.EXPECTED_BINDING_INTERFACES) == {
        "design",
        "custody",
        "source_access",
        "graph_core",
        "graph_core_test",
        "acquisition_runner",
        "acquisition_test",
        "evaluator_runner",
        "evaluator_test",
    }
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError,
        match="implementation freeze is unavailable",
    ):
        r.verify_design_binding(PROJECT_ROOT)


def test_external_freeze_verifies_all_nine_roles_and_freeze_head_blob(
    tmp_path: Path, monkeypatch
) -> None:
    def write(relative: str, raw: bytes) -> tuple[str, str]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest(), r._git_blob_sha1(raw)

    bindings: dict[str, dict[str, str]] = {}
    required_design_strings = [
        r.IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix(),
        r.IMPLEMENTATION_FREEZE_SCHEMA,
        r.EXPECTED_BINDING_INTERFACES["acquisition_runner"]["relative_path"],
        r.EXPECTED_BINDING_INTERFACES["acquisition_runner"]["version"],
        r.EXPECTED_BINDING_INTERFACES["evaluator_runner"]["relative_path"],
        r.VERSION,
        r.LABEL_FREE_SCHEMA,
        r.LABEL_SCHEMA,
        r.LABEL_FREE_ITEM_SCHEMA,
        r.LABEL_ITEM_SCHEMA,
    ]
    for role, interface in r.EXPECTED_BINDING_INTERFACES.items():
        if "schema" in interface:
            hash_field = interface["semantic_hash_field"]
            body: dict[str, object] = {
                "schema": interface["schema"],
                "interface_declarations": (
                    required_design_strings if role == "design" else [role]
                ),
            }
            semantic = _canonical_hash(body)
            raw = (
                json.dumps(
                    {**body, hash_field: semantic},
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            ).encode("ascii")
            file_sha, blob_sha = write(interface["relative_path"], raw)
            bindings[role] = {
                "relative_path": interface["relative_path"],
                "schema": interface["schema"],
                "semantic_sha256": semantic,
                "file_sha256": file_sha,
                "git_blob_sha1": blob_sha,
            }
        else:
            raw = f"synthetic {role}\n".encode("ascii")
            file_sha, blob_sha = write(interface["relative_path"], raw)
            bindings[role] = {
                "relative_path": interface["relative_path"],
                "file_sha256": file_sha,
                "git_blob_sha1": blob_sha,
            }
            if "version" in interface:
                bindings[role]["version"] = interface["version"]

    freeze_body: dict[str, object] = {
        "schema": r.IMPLEMENTATION_FREEZE_SCHEMA,
        "bindings": bindings,
        "source_binding": {"status": "synthetic_not_opened"},
        "selection_secret_commitment": _digest("synthetic-secret"),
        "freeze_hash_contract": {
            "algorithm": "sha256",
            "canonicalization": "sorted_compact_ascii_json",
            "excluded_top_level_fields": [
                r.IMPLEMENTATION_FREEZE_HASH_FIELD
            ],
        },
    }
    freeze = {
        **freeze_body,
        r.IMPLEMENTATION_FREEZE_HASH_FIELD: _canonical_hash(freeze_body),
    }
    freeze_raw = (
        json.dumps(freeze, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("ascii")
    write(r.IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix(), freeze_raw)
    expected_paths = {
        r.IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix(),
        *(row["relative_path"] for row in bindings.values()),
    }

    def fake_head(*, project_root, relative_paths):
        assert project_root == tmp_path.resolve()
        assert set(relative_paths) == expected_paths
        blobs = {
            relative: r._git_blob_sha1((tmp_path / relative).read_bytes())
            for relative in relative_paths
        }
        head = hashlib.sha1(b"synthetic-head").hexdigest()
        return head, blobs, {
            "HEAD": head,
            "commands": [
                {"command": "rev-parse_HEAD", "returncode": 0},
                {
                    "command": "ls-tree_r_HEAD_restricted_paths",
                    "returncode": 0,
                    "path_count": len(relative_paths),
                },
            ],
            "source_secret_private_or_current_stage_output_path_passed": False,
        }

    monkeypatch.setattr(r, "_head_blob_table", fake_head)
    protocol = r.verify_design_binding(tmp_path)
    assert protocol["implementation_freeze_sha256"] == freeze[
        r.IMPLEMENTATION_FREEZE_HASH_FIELD
    ]
    assert protocol["git_verification"][
        "source_secret_private_or_current_stage_output_path_passed"
    ] is False
    assert protocol["evaluator_runner_sha256"] == bindings[
        "evaluator_runner"
    ]["file_sha256"]


def test_stage_loaders_match_acquisition_schema_and_f_has_no_label_loader(
    tmp_path: Path,
) -> None:
    a_view = tmp_path / "a.view.json"
    a_labels = tmp_path / "a.labels.json"
    f_view = tmp_path / "f.view.json"
    _write_view_pack(a_view, "A_form")
    _write_label_pack(a_labels, "A_form", aspects=((5, 6), (7, 8)))
    _write_view_pack(f_view, "F_search")
    loaded_a = r.load_a_form_view(a_view)
    loaded_labels = r.load_a_form_labels(a_labels)
    loaded_f = r.load_f_search_view(f_view)
    assert len(loaded_a.rows) == len(loaded_labels.rows) == len(loaded_f.rows) == 64
    assert loaded_labels.rows[0].gold_aspect_node_indices == ((5, 6), (7, 8))
    assert len({row.component_commitment_sha256 for row in loaded_a.rows}) == 64
    assert not hasattr(r, "load_f_search_labels")


def test_loader_treats_offsets_as_sentence_indices_without_hidden_char_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sentence-offsets.json"
    _write_view_pack(path, "A_form")
    payload = json.loads(path.read_text(encoding="utf-8"))
    first_nodes = payload["rows"][0]["nodes"]
    first_nodes[0]["identity_text"] = "L" * 15_001
    for span_i, node in enumerate(first_nodes):
        node["start"] = span_i
        node["end"] = span_i + 1
    body = dict(payload)
    body.pop("block_sha256")
    _write_private(path, {**body, "block_sha256": _canonical_hash(body)})
    loaded = r.load_a_form_view(path)
    assert loaded.rows[0].nodes[0].start == 0
    assert loaded.rows[0].nodes[0].end == 1
    assert len(loaded.rows[0].nodes[0].identity_text) == 15_001


def test_runner_has_no_dataset_acquisition_or_network_loader_import() -> None:
    source = Path(r.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    rendered = "\n".join(sorted(imports))
    assert "evidencebench_direct_acquisition" not in rendered
    assert not any(
        forbidden in rendered
        for forbidden in ("requests", "urllib", "httpx", "datasets", "zipfile")
    )
    assert "load_f_search_labels" not in source


def test_loader_rejects_tamper_mode_component_overlap_and_empty_aspect(
    tmp_path: Path,
) -> None:
    path = tmp_path / "view.json"
    _write_view_pack(path, "A_form")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["hypothesis"] = "tampered"
    _write_private(path, payload)
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="self-hash"
    ):
        r.load_a_form_view(path)

    _write_view_pack(path, "A_form")
    path.chmod(0o644)
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="mode or size"
    ):
        r.load_a_form_view(path)

    _write_view_pack(path, "A_form")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][1]["component_commitment_sha256"] = payload["rows"][0][
        "component_commitment_sha256"
    ]
    body = dict(payload)
    body.pop("block_sha256")
    _write_private(path, {**body, "block_sha256": _canonical_hash(body)})
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="not disjoint"
    ):
        r.load_a_form_view(path)

    labels = tmp_path / "labels.json"
    _write_label_pack(labels, "A_form")
    payload = json.loads(labels.read_text(encoding="utf-8"))
    payload["rows"][0]["gold_aspect_node_indices"] = [[]]
    body = dict(payload)
    body.pop("block_sha256")
    _write_private(labels, {**body, "block_sha256": _canonical_hash(body)})
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="bucket set"
    ):
        r.load_a_form_labels(labels)


def test_formation_uses_shared_cap8_wave_and_opens_only_a_after_barrier(
    formation_result,
) -> None:
    outcome, encoder, runtime, events, label_calls = formation_result
    assert encoder.call_sizes == [r.BLOCK_COUNT * 33, r.BLOCK_COUNT * 33]
    assert runtime.retrieve_count == 2 * r.BLOCK_COUNT
    assert 2 <= runtime.maximum_active <= r.OFFICIAL_CONCURRENCY_CAP == 8
    assert runtime.postflight_count == 1
    assert label_calls == [True]
    assert events.index(("labels_open", "A_form", None)) > max(
        index
        for index, event in enumerate(events)
        if event[0] == "action_terminal"
    )
    assert outcome.f_selection.recipe_id != r.R0
    assert outcome.identifiable_transition is True
    assert outcome.a_arm_aggregates["official_HippoRAG"]["total_U"] == 0
    assert outcome.a_arm_aggregates["Agent"]["total_U"] == 128_000


def test_all_recipes_and_evaluators_are_scanned(formation_result) -> None:
    outcome = formation_result[0]
    assert len(outcome.a_selection.evaluator_results) == 16
    assert outcome.a_selection.evaluator_results[0].coverage_comparisons == 64 * 9
    assert outcome.f_selection.coverage_comparisons == 64 * 9
    assert len(outcome.action_table_sha256) == 64


def test_same_behavior_is_terminal_without_fallback(formation_result) -> None:
    terminal = replace(formation_result[0], identifiable_transition=False)
    receipt = r.formation_public_receipt(terminal)
    assert receipt["status"] == "terminal_same_behavior_no_runner_up"
    assert receipt["A_hold_authorized"] is False
    assert receipt["runner_up_or_fallback_attempted"] is False


def test_alternative_bucket_evidence_counts_once_per_aspect() -> None:
    item = _view_item("A_hold", 0)
    label = r.LabelItem(
        block="A_hold",
        ordinal=0,
        item_commitment_sha256=item.item_commitment_sha256,
        gold_aspect_node_indices=((5, 6), (7, 8)),
    )
    aggregates, utilities = r._arm_aggregates(
        ((item, label),),
        {"Agent": ((6, 8, 0, 1, 2),)},
    )
    assert aggregates["Agent"] == {
        "item_count": 1,
        "aspect_covered_count": 2,
        "aspect_total": 2,
        "complete_count": 1,
        "total_U": 2000,
    }
    assert utilities["Agent"] == (2000,)
    histograms = r._label_histograms(((item, label),))
    assert histograms["aspects_with_alternative_evidence_histogram"] == {"2": 1}


def test_a_hold_runs_only_r0_and_agent_and_uses_exact_promotion(
    tmp_path: Path, monkeypatch, formation_result
) -> None:
    selected = formation_result[0].f_selection.recipe_id
    runtime = SyntheticRuntime()
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
        assert runtime.retrieve_count == r.BLOCK_COUNT
        assert runtime.postflight_count == 1
        label_calls.append(True)
        return _label_block("A_hold")

    outcome = r.run_measurement_wave(
        _view_block("A_hold"),
        selected_recipe_id=selected,
        selected_evaluator_id=formation_result[0].a_selection.evaluator_id,
        label_loader=labels,
        encoder=SyntheticEncoder(),
        runtime=runtime,
        work_root=tmp_path / "work",
    )
    assert label_calls == [True]
    assert seen_recipes.count(r.R0) == 64
    assert seen_recipes.count(selected) == 64
    assert set(seen_recipes) == {r.R0, selected}
    assert outcome.arm_aggregates["official_HippoRAG"]["total_U"] == 0
    assert outcome.arm_aggregates["Agent"]["total_U"] == 128_000
    assert outcome.exact_test["observed_net_U"] == 128_000
    assert outcome.exact_test["promoted"] is True
    assert outcome.exact_test["p_value_numerator"] == 1
    assert outcome.exact_test["p_value_denominator"] == 2**64


def test_public_report_contains_only_aggregate_label_statistics(
    tmp_path: Path, formation_result
) -> None:
    outcome = r.run_measurement_wave(
        _view_block("A_hold"),
        selected_recipe_id=formation_result[0].f_selection.recipe_id,
        selected_evaluator_id=formation_result[0].a_selection.evaluator_id,
        label_loader=lambda: _label_block(
            "A_hold", aspects=((5, 6), (7, 8))
        ),
        encoder=SyntheticEncoder(),
        runtime=SyntheticRuntime(),
        work_root=tmp_path / "work",
    )
    receipt = r.measurement_public_receipt(outcome)
    rendered = json.dumps(receipt, sort_keys=True)
    for forbidden in (
        "Find TargetBridge",
        "TargetBridge evidence",
        "identity_text",
        "gold_aspect_node_indices",
        "item_commitment_sha256",
        "paper_commitment_sha256",
        "component_commitment_sha256",
        "PRIVATE_SENTINEL",
    ):
        assert forbidden not in rendered
    assert receipt["arm_aggregates"]["Agent"]["aspect_total"] == 128
    assert receipt["label_histograms"]["aspect_count_histogram"] == {"2": 64}
    body = dict(receipt)
    declared = body.pop("receipt_sha256")
    assert _canonical_hash(body) == declared


def test_m_is_sealed_without_promotion_before_any_loader_or_work(tmp_path: Path) -> None:
    calls: list[str] = []

    def view():
        calls.append("view")
        return _view_block("M_search")

    def labels():
        calls.append("labels")
        return _label_block("M_search")

    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="sealed"
    ):
        r.run_m_if_authorized(
            authorized=False,
            view_loader=view,
            label_loader=labels,
            selected_recipe_id="R1_ADJACENT_1SWAP",
            encoder=SyntheticEncoder(),
            runtime=SyntheticRuntime(),
            work_root=tmp_path / "never-created",
        )
    assert calls == []
    assert not (tmp_path / "never-created").exists()


def test_official_failure_is_redacted_and_never_opens_labels(tmp_path: Path) -> None:
    label_calls: list[bool] = []
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError,
        match="official runtime item failed",
    ) as caught:
        r.run_measurement_wave(
            _view_block("A_hold"),
            selected_recipe_id="R1_ADJACENT_1SWAP",
            label_loader=lambda: label_calls.append(True)
            or _label_block("A_hold"),
            encoder=SyntheticEncoder(),
            runtime=SyntheticRuntime(fail_once=True),
            work_root=tmp_path / "work",
        )
    assert "PRIVATE_SENTINEL" not in str(caught.value)
    assert label_calls == []


def test_parent_stage_marker_is_one_shot_and_failure_receipt_is_redacted(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = _synthetic_protocol_binding()
    monkeypatch.setattr(
        r, "verify_design_binding", lambda _root: protocol
    )
    acquisition = r.AcquisitionReceiptBinding(
        acquisition_sha256=_digest("acquisition-semantic"),
        file_sha256=_digest("acquisition-file"),
        git_blob_sha1=hashlib.sha1(b"acquisition-blob").hexdigest(),
        verified_at_git_HEAD=str(protocol["git_HEAD"]),
        commitments_by_block={},
        payload={},
    )
    monkeypatch.setattr(
        r,
        "_load_canonical_acquisition_receipt",
        lambda **_kwargs: acquisition,
    )
    bad_a = tmp_path / r.CANONICAL_PRIVATE_PACKS["A_form"][0]
    bad_a.parent.mkdir(parents=True)
    bad_a.write_text(
        "not-json PRIVATE_SENTINEL_771 hypothesis", encoding="utf-8"
    )
    bad_a.chmod(0o600)
    paths = r._canonical_stage_absolutes(tmp_path, "formation")
    marker = paths["root"] / "formation.attempt.marker"
    original_loader = r.load_a_form_view

    def checked_loader(path: Path):
        assert path == bad_a
        assert marker.is_file()
        return original_loader(path)

    monkeypatch.setattr(r, "load_a_form_view", checked_loader)
    with pytest.raises(r.EvidenceBenchGraphEvaluatorRunnerError):
        r.execute_formation_stage(
            project_root=tmp_path,
            encoder=SyntheticEncoder(),
            runtime=SyntheticRuntime(),
        )
    assert marker.exists()
    payload = json.loads(paths["failure"].read_text(encoding="ascii"))
    assert payload["status"] == "terminal_infrastructure_invalid_no_replay"
    rendered = json.dumps(payload)
    assert "PRIVATE_SENTINEL" not in rendered
    assert str(bad_a) not in rendered
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="replay is forbidden"
    ):
        r.consume_stage_marker(marker, "formation")


def test_formal_stage_entrypoints_do_not_accept_path_overrides() -> None:
    expected = {"project_root", "encoder", "runtime", "protocol_binding", "progress"}
    for entrypoint in (
        r.execute_formation_stage,
        r.execute_a_hold_stage,
        r.execute_m_search_stage,
    ):
        assert set(inspect.signature(entrypoint).parameters) == expected


def test_project_root_alias_and_existing_stage_destination_fail_closed(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="symbolic link"
    ):
        r._canonical_project_root(alias)

    paths = r._canonical_stage_absolutes(real_root, "formation")
    paths["root"].mkdir(parents=True)
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError, match="replay is forbidden"
    ):
        r._preflight_canonical_stage_outputs(paths)
    assert not (paths["root"] / "formation.attempt.marker").exists()


def test_resource_preparation_cannot_change_loaded_code_binding() -> None:
    initial = _synthetic_protocol_binding()
    r._require_protocol_binding_unchanged(initial, dict(initial))
    drifted = {**initial, "git_HEAD": hashlib.sha1(b"new-head").hexdigest()}
    with pytest.raises(
        r.EvidenceBenchGraphEvaluatorRunnerError,
        match="changed during resource preparation",
    ):
        r._require_protocol_binding_unchanged(initial, drifted)

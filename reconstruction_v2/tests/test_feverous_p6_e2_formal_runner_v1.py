from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_e2_evaluator_v1 as evaluator
from assumption_agent.benchmarks import feverous_p6_e2_formal_runner_v1 as subject
from assumption_agent.benchmarks import feverous_p6_query_anchored_operator_v1 as operator


def _sentence_text(target: str, title: str = "Page") -> str:
    return (
        f"TARGET: {target}\n"
        f"TITLE: {title}\n"
        "SECTION_PATH: <ROOT>\n"
        "TYPE: sentence"
    )


def _cell_text(target: str, kind: str) -> str:
    return (
        f"TARGET: {target}\n"
        "TITLE: Structured\n"
        "SECTION_PATH: <ROOT>\n"
        f"TYPE: {kind}\n"
        "TABLE_CAPTION: Demo\n"
        "APPLICABLE_HEADERS: ROW[Header] COLUMN[Header]\n"
        "ROW_WITH_TARGET_MARKED: [TARGET]"
    )


def _item_text(target: str) -> str:
    return (
        f"TARGET: {target}\n"
        "TITLE: List\n"
        "SECTION_PATH: <ROOT>\n"
        "TYPE: item\n"
        "LIST_ANCESTOR_PATH: <ROOT>"
    )


def _sidecar(
    *,
    page: str,
    local_id: str,
    unit_type: str,
    coordinates: tuple[int, ...],
    official_ordinal: int,
    table_id: str | None = None,
    applicable_row_header_ids: tuple[str, ...] = (),
    applicable_column_header_ids: tuple[str, ...] = (),
    list_id: str | None = None,
    list_ancestor_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    is_cell = unit_type in {"cell", "header_cell"}
    is_table = is_cell or unit_type == "table_caption"
    return {
        "linearizer_version": acquisition.ATOMIC_LINEARIZER_VERSION,
        "page": page,
        "local_id": local_id,
        "unit_type": unit_type,
        "coordinates": coordinates,
        "section_ids": (),
        "section_path": (),
        "official_ordinal": official_ordinal,
        "previous_atomic_local_id": None,
        "next_atomic_local_id": None,
        "table_id": table_id,
        "table_kind": "normal" if is_table else None,
        "table_caption": "Demo" if is_table else None,
        "row_span": 1 if is_cell else None,
        "column_span": 1 if is_cell else None,
        "applicable_row_header_ids": applicable_row_header_ids,
        "applicable_column_header_ids": applicable_column_header_ids,
        "list_id": list_id,
        "list_ancestor_ids": list_ancestor_ids,
    }


def _view() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    special = (
        {
            "text": _cell_text("Header", "header_cell"),
            "unit_type": "header_cell",
            "sidecar": _sidecar(
                page="Structured",
                local_id="header_cell_0_0_0",
                unit_type="header_cell",
                coordinates=(0, 0, 0),
                official_ordinal=0,
                table_id="table_0",
            ),
        },
        {
            "text": _cell_text("Value", "cell"),
            "unit_type": "cell",
            "sidecar": _sidecar(
                page="Structured",
                local_id="cell_0_0_1",
                unit_type="cell",
                coordinates=(0, 0, 1),
                official_ordinal=1,
                table_id="table_0",
                applicable_row_header_ids=("header_cell_0_0_0",),
            ),
        },
        {
            "text": _item_text("First"),
            "unit_type": "item",
            "sidecar": _sidecar(
                page="ListPage",
                local_id="item_0_0",
                unit_type="item",
                coordinates=(0, 0),
                official_ordinal=0,
                list_id="list_0",
            ),
        },
        {
            "text": _item_text("Second"),
            "unit_type": "item",
            "sidecar": _sidecar(
                page="ListPage",
                local_id="item_0_1",
                unit_type="item",
                coordinates=(0, 1),
                official_ordinal=1,
                list_id="list_0",
            ),
        },
    )
    for ordinal in range(acquisition.CORPUS_UNIT_COUNT):
        if ordinal < len(special):
            row = dict(special[ordinal])
        else:
            row = {
                "text": _sentence_text(f"payload {ordinal}", f"Page {ordinal}"),
                "unit_type": "sentence",
                "sidecar": _sidecar(
                    page=f"Page_{ordinal}",
                    local_id="sentence_0",
                    unit_type="sentence",
                    coordinates=(0,),
                    official_ordinal=0,
                ),
            }
        rows.append({"unit_i": ordinal, **row})
    return acquisition.self_hashed(
        {
            "schema": acquisition.CORPUS_VIEW_SCHEMA,
            "version": acquisition.VERSION,
            "unit_count": acquisition.CORPUS_UNIT_COUNT,
            "gold_origin_or_membership_included": False,
            "units": rows,
        },
        "corpus_view_sha256",
    )


def _action(recipe: str, output: tuple[int, int, int, int, int]) -> operator.ActionTrace:
    trace = operator.ActionTrace(
        recipe_id=recipe,
        output_top5=output,
        retained_raw_top3=(output[0], output[1], output[2]),
        selection_steps=(),
        raw_dense_order_sha256="1" * 64,
        graph_sha256="2" * 64,
        query_sha256="3" * 64,
        semantic_tensor_sha256="4" * 64,
        reachability_sha256="5" * 64,
        candidate_scan_sha256="6" * 64,
        candidate_universe_size=acquisition.CORPUS_UNIT_COUNT,
        candidate_score_evaluations=acquisition.CORPUS_UNIT_COUNT,
        semantic_cell_scan_count=acquisition.CORPUS_UNIT_COUNT,
        hipporag_candidate_or_feature_count=0,
        trace_sha256="0" * 64,
    )
    return replace(
        trace,
        trace_sha256=operator.recompute_action_trace_sha256(trace),
    )


def _execution(block: str) -> subject.BlockExecution:
    items = []
    for ordinal in range(subject.BLOCK_COUNTS[block]):
        actions = (
            _action("R0_DENSE5", (0, 2, 3, 4, 5)),
            _action("R1_P6_DIRECT_B2", (0, 6, 7, 8, 9)),
            _action("R2_P6_PATH1_B2", (0, 1, 10, 11, 12)),
            _action("R3_P6_PATH2_B2", (0, 13, 14, 15, 16)),
        )
        items.append(
            subject.ItemExecution(
                block=block,
                ordinal=ordinal,
                item_commitment_sha256=subject.stable_hash([block, ordinal]),
                semantic_build=object(),  # late scorer cannot inspect semantics
                action_traces=actions,
                feature_traces=(),
                operator_receipt_sha256="8" * 64,
            )
        )
    return subject.BlockExecution(
        block=block,
        items=tuple(items),
        feature_receipt=MappingProxyType({}),
        receipt=MappingProxyType({}),
    )


def _labels(block: str) -> dict[str, object]:
    rows = []
    for ordinal in range(subject.BLOCK_COUNTS[block]):
        family = acquisition.FAMILIES[
            ordinal // (subject.BLOCK_COUNTS[block] // len(acquisition.FAMILIES))
        ]
        rows.append(
            {
                "ordinal": ordinal,
                "gold_unit_indices": [0, 1],
                "family": family,
                "verdict": acquisition.VERDICTS[ordinal % 2],
            }
        )
    return acquisition.self_hashed(
        {
            "schema": acquisition.BLOCK_LABEL_SCHEMA,
            "version": acquisition.VERSION,
            "block": block,
            "item_count": len(rows),
            "items": rows,
        },
        "block_labels_sha256",
    )


def test_corpus_conversion_preserves_table_headers_and_real_list_root() -> None:
    units = subject.corpus_view_to_semantic_units(_view())
    assert len(units) == acquisition.CORPUS_UNIT_COUNT
    assert units[0].table_key == units[1].table_key
    assert units[1].table_row == 0
    assert units[1].applicable_header_ordinals == (0,)
    assert units[2].list_parent_path
    assert units[2].list_parent_path == units[3].list_parent_path


def test_anchor_scoring_proves_primary_promotion_and_raw_complete_rule() -> None:
    block = _execution("A_hold")
    # R2 is complete; E0=R1, Hippo and RAW each recover only one of two.
    hippo = [(0, 20, 21, 22, 23)] * subject.BLOCK_COUNTS["A_hold"]
    receipt = subject.score_anchor_block(
        block=block,
        labels=_labels("A_hold"),
        hippo_top5=hippo,
        e0_recipe_id="R1_P6_DIRECT_B2",
        e2_recipe_id="R2_P6_PATH1_B2",
        evaluator_comparison_identifiable=True,
    )
    assert receipt["A_hold_real_domain_primary_passed"] is True
    assert receipt["evaluator_promoted"] is True
    assert receipt["RAW_complete_advantage_overcome"] is True
    assert all(
        value == [27, 1]
        for value in receipt["E2_minus_HippoRAG_family_sums"].values()
    )
    assert receipt["complete_counts"] == {
        "E0": 0,
        "E2": 72,
        "HippoRAG": 0,
        "RAW": 0,
    }
    assert receipt["all_four_recipe_aggregates"] == {
        "R0_DENSE5": {"total_U": [36, 1], "complete_count": 0},
        "R1_P6_DIRECT_B2": {"total_U": [36, 1], "complete_count": 0},
        "R2_P6_PATH1_B2": {"total_U": [144, 1], "complete_count": 72},
        "R3_P6_PATH2_B2": {"total_U": [36, 1], "complete_count": 0},
    }


def test_unidentifiable_policy_cannot_promote_even_with_positive_delta() -> None:
    receipt = subject.score_anchor_block(
        block=_execution("A_hold"),
        labels=_labels("A_hold"),
        hippo_top5=[(0, 20, 21, 22, 23)] * subject.BLOCK_COUNTS["A_hold"],
        e0_recipe_id="R1_P6_DIRECT_B2",
        e2_recipe_id="R2_P6_PATH1_B2",
        evaluator_comparison_identifiable=False,
    )
    assert receipt["E2_minus_E0"]["promoted"] is True
    assert receipt["evaluator_promoted"] is False


def test_a_form_utility_matrix_has_every_item_recipe_pair() -> None:
    block = _execution("A_form")
    utilities = subject.a_form_utility_matrix(
        block=block,
        labels=_labels("A_form"),
    )
    assert len(utilities) == subject.BLOCK_COUNTS["A_form"] * len(subject.RECIPE_IDS)
    first = block.items[0].item_commitment_sha256
    assert utilities[(first, "R0_DENSE5")] == Fraction(1, 2)
    assert utilities[(first, "R2_P6_PATH1_B2")] == Fraction(2, 1)


def test_claim_view_remains_claim_only_without_block_or_family() -> None:
    view = acquisition.self_hashed(
        {
            "schema": acquisition.BLOCK_VIEW_SCHEMA,
            "version": acquisition.VERSION,
            "item_count": subject.BLOCK_COUNTS["F_search"],
            "late_label_fields_included": False,
            "items": [
                {"claim": f"claim {ordinal}"}
                for ordinal in range(subject.BLOCK_COUNTS["F_search"])
            ],
        },
        "block_view_sha256",
    )
    assert len(subject.claims_from_block_view(view, block="F_search")) == 48
    body = dict(view)
    body.pop("block_view_sha256")
    body["block"] = "F_search"
    tampered = acquisition.self_hashed(body, "block_view_sha256")
    with pytest.raises(subject.FeverousFormalRunnerError):
        subject.claims_from_block_view(tampered, block="F_search")


def test_cross_table_header_and_duplicate_official_order_fail_closed() -> None:
    view = _view()
    body = dict(view)
    body.pop("corpus_view_sha256")
    rows = [dict(row) for row in body["units"]]
    rows[1] = dict(rows[1])
    sidecar = dict(rows[1]["sidecar"])
    sidecar["applicable_row_header_ids"] = ["header_cell_1_0_0"]
    rows[1]["sidecar"] = sidecar
    body["units"] = rows
    with pytest.raises(subject.FeverousFormalRunnerError):
        subject.corpus_view_to_semantic_units(
            acquisition.self_hashed(body, "corpus_view_sha256")
        )

    body = dict(_view())
    body.pop("corpus_view_sha256")
    rows = [dict(row) for row in body["units"]]
    rows[1] = dict(rows[1])
    sidecar = dict(rows[1]["sidecar"])
    sidecar["official_ordinal"] = 0
    rows[1]["sidecar"] = sidecar
    body["units"] = rows
    with pytest.raises(subject.FeverousFormalRunnerError):
        subject.corpus_view_to_semantic_units(
            acquisition.self_hashed(body, "corpus_view_sha256")
        )


def test_action_receipt_contains_complete_recomputable_content_free_trace() -> None:
    trace = _action("R2_P6_PATH1_B2", (0, 1, 10, 11, 12))
    receipt = subject.action_trace_content_free_receipt(trace)
    declared = receipt.pop("trace_sha256")
    assert operator.stable_hash(receipt) == declared == trace.trace_sha256
    assert receipt["selection_steps"] == []
    assert "linearized_text" not in receipt


def test_shared_formation_pool_interleaves_all_144_before_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    submissions: list[tuple[str, int]] = []
    pool_counts: list[int] = []

    class FakeFuture:
        def __init__(self, block: str, ordinal: int) -> None:
            self.block = block
            self.ordinal = ordinal

        def result(self) -> SimpleNamespace:
            assert len(submissions) == 144
            events.append(("result", self.block, self.ordinal))
            return SimpleNamespace(block=self.block, ordinal=self.ordinal)

    class FakePool:
        def __init__(self, *, max_workers: int) -> None:
            pool_counts.append(max_workers)

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def submit(self, _fn: object, **kwargs: object) -> FakeFuture:
            block = str(kwargs["block"])
            ordinal = int(kwargs["ordinal"])
            submissions.append((block, ordinal))
            events.append(("submit", block, ordinal))
            return FakeFuture(block, ordinal)

    def fake_assemble(
        *, block: str, items: object, **_kwargs: object
    ) -> subject.BlockExecution:
        rows = tuple(items)
        return subject.BlockExecution(
            block=block,
            items=rows,
            feature_receipt=MappingProxyType({}),
            receipt=MappingProxyType(
                {"block_receipt_sha256": subject.stable_hash(block)}
            ),
        )

    monkeypatch.setattr(subject, "ThreadPoolExecutor", FakePool)
    monkeypatch.setattr(subject, "_assemble_block_execution", fake_assemble)
    formed = subject.execute_formation_blocks(
        A_form_claims=["A"] * 96,
        F_search_claims=["F"] * 48,
        prepared_corpus=None,
        minilm_backend=None,
        ner_backend=None,
        nli_backend=None,
    )
    assert pool_counts == [64]
    assert submissions[:6] == [
        ("A_form", 0),
        ("F_search", 0),
        ("A_form", 1),
        ("F_search", 1),
        ("A_form", 2),
        ("F_search", 2),
    ]
    assert len(submissions) == 144
    assert next(index for index, event in enumerate(events) if event[0] == "result") == 144
    assert len(formed.A_form.items) == 96
    assert len(formed.F_search.items) == 48
    assert formed.receipt["single_shared_thread_pool"] is True
    assert formed.receipt["all_144_items_submitted_before_first_join"] is True


def test_separate_formation_execution_is_forbidden() -> None:
    with pytest.raises(subject.FeverousFormalRunnerError):
        subject.execute_local_block(
            block="F_search",
            claims=["F"] * 48,
            prepared_corpus=None,
            minilm_backend=None,
            ner_backend=None,
            nli_backend=None,
        )


def test_anchor_feature_receipt_cannot_masquerade_as_formation() -> None:
    fake_traces = tuple(
        SimpleNamespace(
            payload=lambda ordinal=ordinal, recipe=recipe: {
                "item": ordinal,
                "recipe": recipe,
            }
        )
        for ordinal in range(subject.BLOCK_COUNTS["A_hold"])
        for recipe in subject.RECIPE_IDS
    )
    items = []
    for ordinal in range(subject.BLOCK_COUNTS["A_hold"]):
        start = ordinal * len(subject.RECIPE_IDS)
        rows = fake_traces[start : start + len(subject.RECIPE_IDS)]
        items.append(
            SimpleNamespace(
                block="A_hold",
                ordinal=ordinal,
                recipe_traces=rows,
                item_commitment_sha256=subject.stable_hash(["A_hold", ordinal]),
                semantic_build=SimpleNamespace(
                    receipt={"semantic_receipt_sha256": "1" * 64}
                ),
                operator_receipt_sha256="2" * 64,
                feature_traces=tuple(
                    SimpleNamespace(production_trace_sha256="3" * 64)
                    for _recipe in subject.RECIPE_IDS
                ),
            )
        )
    anchor = subject._assemble_block_execution(
        block="A_hold",
        items=items,
        worker_count=64,
        execution_scope="single_anchor_block_pool",
        formation_total_items_eager_submitted=None,
    )
    assert anchor.feature_receipt["schema"].endswith("_anchor_feature_receipt")
    assert anchor.feature_receipt["evaluator_fit_or_policy_selection_authorized"] is False
    with pytest.raises(evaluator.FeverousEvaluatorError):
        evaluator.verify_feature_receipt(
            anchor.feature_receipt,
            block="A_form",
            traces=(),
        )


def test_non_boolean_identifiability_fails_closed() -> None:
    with pytest.raises(subject.FeverousFormalRunnerError):
        subject.score_anchor_block(
            block=_execution("A_hold"),
            labels=_labels("A_hold"),
            hippo_top5=[(0, 20, 21, 22, 23)] * subject.BLOCK_COUNTS["A_hold"],
            e0_recipe_id="R1_P6_DIRECT_B2",
            e2_recipe_id="R2_P6_PATH1_B2",
            evaluator_comparison_identifiable=1,
        )

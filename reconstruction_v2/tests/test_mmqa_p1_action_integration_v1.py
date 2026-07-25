from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json

import pytest

from assumption_agent.benchmarks import mmqa_p1_action_integration_v1 as action
from assumption_agent.benchmarks import mmqa_p1_typed_proof_e5_core_v1 as core


def _work_item() -> dict[str, object]:
    return {
        "schema": action.ANONYMOUS_WORK_ITEM_SCHEMA,
        "question": "Which table row and linked text establish Aurora's launch year?",
        "rows": [
            {
                "ordinal": ordinal,
                "serialized_content": (
                    f"Table Aurora | column entity | column year | row {ordinal}"
                ),
            }
            for ordinal in range(4)
        ],
        "texts": [
            {
                "ordinal": 4 + ordinal,
                "serialized_content": (
                    f"Text {ordinal} | Aurora evidence paragraph {ordinal}."
                ),
            }
            for ordinal in range(4)
        ],
        "exact_row_text_links": [
            {"row_ordinal": row, "text_ordinal": text}
            for row, text in ((0, 4), (1, 4), (1, 5), (2, 6), (3, 7))
        ],
    }


def _coordinates(
    *,
    cross_encoder: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2),
) -> list[dict[str, object]]:
    return [
        {
            "schema": action.UNIT_COORDINATES_SCHEMA,
            "ordinal": ordinal,
            "minilm_similarity": max(0.0, score - 0.05),
            "cross_encoder_relevance": score,
            "entity_anchor": int(ordinal in {0, 1, 4, 5}),
            "relation_anchor": int(ordinal in {1, 2, 5, 6}),
            "numeric_or_temporal_anchor": int(ordinal in {0, 4}),
        }
        for ordinal, score in enumerate(cross_encoder)
    ]


def _zero_e5_model() -> core.E5Model:
    return core.E5Model(
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


def _large_item(*, only_pruned_rows_linked: bool = False) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = [
        {
            "ordinal": ordinal,
            "serialized_content": f"Row {ordinal} | value {ordinal}",
        }
        for ordinal in range(52)
    ]
    texts = [
        {
            "ordinal": 100 + ordinal,
            "serialized_content": f"Text {ordinal} | evidence {ordinal}",
        }
        for ordinal in range(5)
    ]
    linked_rows = range(4) if only_pruned_rows_linked else range(52)
    item = {
        "schema": action.ANONYMOUS_WORK_ITEM_SCHEMA,
        "question": "Which ranked row has an exact link to the text?",
        "rows": rows,
        "texts": texts,
        "exact_row_text_links": [
            {"row_ordinal": ordinal, "text_ordinal": 100}
            for ordinal in linked_rows
        ],
    }
    coordinates = []
    for ordinal in (*range(52), *range(100, 105)):
        score = ordinal / 200.0 if ordinal < 52 else 0.25
        coordinates.append(
            {
                "schema": action.UNIT_COORDINATES_SCHEMA,
                "ordinal": ordinal,
                "minilm_similarity": score,
                "cross_encoder_relevance": score,
                "entity_anchor": int(ordinal % 2 == 0),
                "relation_anchor": int(ordinal % 3 == 0),
                "numeric_or_temporal_anchor": 0,
            }
        )
    return item, coordinates


def _all_keys(value: object) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        output.update(str(key) for key in value)
        for nested in value.values():
            output.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            output.update(_all_keys(nested))
    return output


def test_module_is_source_free_and_form_actions_has_no_gold_surface() -> None:
    assert action.STUDY_ID == "MMQA_P1_LOCAL_PROOF_E5_V1"
    assert action.STUDY_DESIGN_SELF_SHA256 == (
        "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
    )
    assert action.MAXIMUM_ROW_NODES == 48
    assert action.MAXIMUM_TEXT_NODES == 48
    assert "gold" not in inspect.signature(action.form_actions).parameters
    assert "label" not in inspect.signature(action.form_actions).parameters

    tree = ast.parse(inspect.getsource(action))
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


@pytest.mark.parametrize(
    "forbidden",
    [
        "qid",
        "family",
        "type",
        "answers",
        "supporting_context",
        "metadata_id",
    ],
)
def test_anonymous_work_item_rejects_forbidden_source_fields(forbidden: str) -> None:
    value = _work_item()
    value[forbidden] = "forbidden"
    with pytest.raises(action.MmqaP1ActionIntegrationError, match=forbidden):
        action.validate_anonymous_work_item(value)


def test_nested_units_links_and_coordinates_reject_metadata_or_label_fields() -> None:
    item = _work_item()
    item["rows"][0]["source_id"] = "row-1"  # type: ignore[index]
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="source_id"):
        action.validate_anonymous_work_item(item)

    item = _work_item()
    item["exact_row_text_links"][0]["support"] = True  # type: ignore[index]
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="support"):
        action.validate_anonymous_work_item(item)

    coordinates = _coordinates()
    coordinates[0]["family"] = "hidden"
    validated = action.validate_anonymous_work_item(_work_item())
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="family"):
        action.validate_unit_coordinates(validated, coordinates)


def test_anonymous_text_ordinals_and_exact_links_are_strict() -> None:
    item = _work_item()
    item["texts"][0]["ordinal"] = 0  # type: ignore[index]
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="globally distinct"):
        action.validate_anonymous_work_item(item)

    item = _work_item()
    item["rows"][0]["serialized_content"] = " noncanonical "  # type: ignore[index]
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="noncanonical"):
        action.validate_anonymous_work_item(item)

    item = _work_item()
    item["exact_row_text_links"][0]["text_ordinal"] = 999  # type: ignore[index]
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="outside"):
        action.validate_anonymous_work_item(item)


def test_unit_coordinates_must_cover_every_unit_once_in_source_ordinal_order() -> None:
    item = action.validate_anonymous_work_item(_work_item())
    coordinates = _coordinates()
    assert tuple(
        row.ordinal for row in action.validate_unit_coordinates(item, coordinates)
    ) == tuple(range(8))
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="exactly follow"):
        action.validate_unit_coordinates(item, list(reversed(coordinates)))
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="exactly follow"):
        action.validate_unit_coordinates(item, coordinates[:-1])


def test_action_formation_has_one_byte_identical_three_arm_closure() -> None:
    result = action.form_actions(_work_item(), _coordinates())
    shared = result.shared_closure
    assert shared.ordinals == tuple(range(8))
    assert shared.agent_ordinal_bytes == shared.raw_ordinal_bytes
    assert shared.agent_ordinal_bytes == shared.hipporag_ordinal_bytes
    assert shared.agent_ordinal_bytes == b"[0,1,2,3,4,5,6,7]"
    assert shared.ordinal_bytes_sha256 == hashlib.sha256(
        shared.agent_ordinal_bytes
    ).hexdigest()
    assert tuple(node.ordinal for node in result.core_closure.graph.nodes) == (
        shared.ordinals
    )
    assert all(
        set(ranking.top5_ordinals).issubset(shared.ordinals)
        for ranking in (result.e0_ranking, result.raw_ranking)
    )


def test_exact_structural_links_become_reciprocal_directed_core_edges() -> None:
    result = action.form_actions(_work_item(), _coordinates())
    pairs = {
        (edge.source_ordinal, edge.target_ordinal, edge.edge_type)
        for edge in result.core_closure.graph.edges
    }
    for row, text in ((0, 4), (1, 4), (1, 5), (2, 6), (3, 7)):
        assert (row, text, core.ROW_TO_TEXT) in pairs
        assert (text, row, core.TEXT_TO_ROW) in pairs
    assert len(pairs) == 10
    assert result.bundles
    assert len(result.bundles) <= 256
    assert all(
        core.validate_connected_bundle(result.core_closure.graph, bundle)
        == bundle
        for bundle in result.bundles
    )


def test_raw_is_direct_cross_encoder_top5_with_ordinal_tie_break() -> None:
    coordinates = _coordinates(
        cross_encoder=(0.8, 0.8, 0.6, 0.5, 0.9, 0.7, 0.4, 0.3)
    )
    result = action.form_actions(_work_item(), coordinates)
    assert result.raw_ranking.top5_ordinals == (4, 0, 1, 5, 2)
    assert result.raw_ranking.selected_bundle_ordinals is None
    assert result.raw_ranking.selected_bundle_energy is None


def test_optional_e5_reuses_the_same_bundles_and_closure() -> None:
    model = _zero_e5_model()
    result = action.form_actions(_work_item(), _coordinates(), e5_model=model)
    assert result.e5_ranking is not None
    assert result.e5_ranking.policy_id == "E5"
    assert len(result.e5_ranking.top5_ordinals) == 5
    assert set(result.e5_ranking.top5_ordinals).issubset(
        result.shared_closure.ordinals
    )
    expected_bundle = min(result.bundles, key=lambda row: row.node_ordinals)
    assert result.e5_ranking.selected_bundle_ordinals == (
        expected_bundle.node_ordinals
    )
    assert result.e5_ranking.selected_bundle_energy == 0.0


def test_row_selection_is_fixed_top48_and_all_texts_are_retained() -> None:
    item, coordinates = _large_item()
    result = action.form_actions(item, coordinates)
    expected = tuple((*range(4, 52), *range(100, 105)))
    assert result.shared_closure.ordinals == expected
    assert len(result.shared_closure.ordinals) == 53
    assert all(text in result.shared_closure.ordinals for text in range(100, 105))
    assert len(
        [
            ordinal
            for ordinal in result.shared_closure.ordinals
            if ordinal < 100
        ]
    ) == 48


def test_no_post_cap_structural_link_is_a_terminal_action_failure() -> None:
    item, coordinates = _large_item(only_pruned_rows_linked=True)
    with pytest.raises(
        action.MmqaP1ActionIntegrationError, match="removed every exact"
    ):
        action.form_actions(item, coordinates)


def test_action_feature_archive_is_label_free_canonical_and_complete() -> None:
    result = action.form_actions(
        _work_item(), _coordinates(), e5_model=_zero_e5_model()
    )
    archive = result.action_feature_archive
    payload = archive.payload()
    assert payload["schema"] == action.ACTION_FEATURE_ARCHIVE_SCHEMA
    assert payload["study_design_self_sha256"] == (
        action.STUDY_DESIGN_SELF_SHA256
    )
    assert payload["three_arm_closure_ordinals_byte_identical"] is True
    hashes = payload["three_arm_closure_ordinal_bytes_sha256"]
    assert len(set(hashes.values())) == 1
    assert payload["gold_or_support_read_count"] == 0
    assert payload["network_call_count"] == 0
    assert payload["model_call_count"] == 0
    assert payload["source_reader_call_count"] == 0
    assert payload["retry_replay_resample_count"] == 0
    assert len(payload["bundle_feature_rows"]) == len(result.bundles)
    forbidden_archive_keys = {
        "answer",
        "answers",
        "family",
        "gold",
        "metadata",
        "qid",
        "question",
        "serialized_content",
        "supporting_context",
    }
    assert not (_all_keys(payload) & forbidden_archive_keys)
    expected = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert archive.canonical_bytes() == expected


def test_late_gold_scores_sealed_actions_without_mutating_archive() -> None:
    result = action.form_actions(
        _work_item(), _coordinates(), e5_model=_zero_e5_model()
    )
    before = result.action_feature_archive.canonical_bytes()
    scores = action.score_late_gold(
        result,
        (0, 4),
        exact_gold_pairs=({"row_ordinal": 0, "text_ordinal": 4},),
        hipporag_top5_ordinals=(4, 0, 1, 2, 3),
    )
    after = result.action_feature_archive.canonical_bytes()
    assert before == after
    assert scores.e0.integer_utility == core.integer_utility_from_ndcg(
        scores.e0.ndcg_at_5
    )
    assert scores.e5 is not None
    assert scores.raw.integer_utility == core.integer_utility_from_ndcg(
        scores.raw.ndcg_at_5
    )
    assert scores.hipporag is not None
    assert scores.hipporag.ndcg_at_5 == 1.0
    assert scores.hipporag.recall_at_5 == 1.0
    assert scores.hipporag.connected_gold_row_text_pair_recovered


def test_late_gold_and_external_hippo_must_stay_inside_sealed_closure() -> None:
    result = action.form_actions(_work_item(), _coordinates())
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="universe"):
        action.score_late_gold(result, (999,))
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="five unique"):
        action.score_late_gold(
            result,
            (0, 4),
            hipporag_top5_ordinals=(0, 1, 2, 3, 999),
        )


def test_late_gold_outside_capped_closure_is_a_valid_miss_not_failure() -> None:
    item, coordinates = _large_item()
    result = action.form_actions(item, coordinates)
    assert 0 in {row.ordinal for row in result.work_item.rows}
    assert 0 not in result.shared_closure.ordinals
    assert 100 in result.shared_closure.ordinals

    scores = action.score_late_gold(
        result,
        (0, 100),
        exact_gold_pairs=({"row_ordinal": 0, "text_ordinal": 100},),
    )
    for score in (scores.e0, scores.raw):
        assert score.recall_at_5 <= 0.5
        assert score.integer_utility < core.INTEGER_UTILITY_SCALE
        assert not score.connected_gold_row_text_pair_recovered


@pytest.mark.parametrize(
    "pairs",
    [
        (),
        ({"row_ordinal": 0, "text_ordinal": 999},),
        ({"row_ordinal": 4, "text_ordinal": 0},),
        ({"row_ordinal": 0, "text_ordinal": 5},),
        (
            {"row_ordinal": 0, "text_ordinal": 4},
            {"row_ordinal": 0, "text_ordinal": 4},
        ),
        ({"row_ordinal": 0, "text_ordinal": 4, "family": "forbidden"},),
    ],
)
def test_late_exact_gold_pairs_fail_closed_against_original_item(
    pairs: tuple[dict[str, object], ...],
) -> None:
    result = action.form_actions(_work_item(), _coordinates())
    with pytest.raises(action.MmqaP1ActionIntegrationError, match="gold pair"):
        action.score_late_gold(
            result,
            (0, 4),
            exact_gold_pairs=pairs,
        )


def test_inputs_are_not_mutated_by_validation_or_action_formation() -> None:
    item = _work_item()
    coordinates = _coordinates()
    expected_item = copy.deepcopy(item)
    expected_coordinates = copy.deepcopy(coordinates)
    action.form_actions(item, coordinates)
    assert item == expected_item
    assert coordinates == expected_coordinates

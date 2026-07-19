from __future__ import annotations

from fractions import Fraction
import hashlib

import numpy as np
import pytest

from assumption_agent.benchmarks import feverous_e2_evaluator_v1 as evaluator
from assumption_agent.benchmarks import hybridqa_query_anchored_formal_runner_v1 as runner
from assumption_agent.benchmarks import hybridqa_query_anchored_operator_v1 as operator
from replication_runtime.multihoprag_minilm_v1 import adapter as minilm


HASH = "a" * 64


class _FakeEncoder:
    def __init__(self, *, scale: float = 1.0):
        identity = minilm.frozen_minilm_runtime_identity()
        self.runtime_receipt = {
            "asset_file_sha256": identity["asset_file_sha256"],
            "asset_manifest_path": "/synthetic/fixed/asset.json",
            "asset_sha256": identity["asset_sha256"],
            "embedding_dimension": identity["embedding_dimension"],
            "maximum_sequence_length": identity["maximum_sequence_length"],
            "model_root": "/synthetic/fixed/model",
            "model_tree_sha256": identity["model_tree_sha256"],
            "runtime_versions": identity["runtime_versions"],
            "status": identity["status"],
            "weights_sha256": identity["weights_sha256"],
        }
        self.canary_receipt = {
            "float32_bytes_sha256": identity["canary_float32_bytes_sha256"],
            "quantized_embedding_matrix_sha256": identity[
                "canary_quantized_embedding_sha256"
            ],
            "qasper_rows_or_archives_accessed_by_canary": False,
            "repeat_count": 2,
            "repeat_exact": True,
            "sentence_count": identity["canary_sentence_count"],
            "status": "passed_exact_row_free_synthetic_canary",
            "text_vector_sha256": identity["canary_text_vector_sha256"],
        }
        self.scale = np.float32(scale)
        self.call_count = 0

    def encode(self, texts):
        self.call_count += 1
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            first = int.from_bytes(digest[:2], "big") % 384
            second = int.from_bytes(digest[2:4], "big") % 384
            matrix[row, first] += np.float32(1.0)
            matrix[row, second] += np.float32(0.5)
            matrix[row] /= np.linalg.norm(matrix[row])
        return matrix * self.scale


def _embedding_index():
    encoder = _FakeEncoder()
    articles = tuple(
        minilm.ArticleText(index, f"Title {index}", f"body token {index}")
        for index in range(operator.CORPUS_UNIT_COUNT)
    )
    return minilm.build_corpus_embedding_index(articles=articles, encoder=encoder)


def _units() -> tuple[operator.AtomicUnit, ...]:
    rows: list[operator.AtomicUnit] = [
        operator.AtomicUnit(0, "table_row", "t0", 0, ("l0",)),
        operator.AtomicUnit(1, "linked_passage", "t0", None, ("l0",)),
        operator.AtomicUnit(2, "table_row", "t0", 1, ()),
    ]
    rows.extend(
        operator.AtomicUnit(i, "table_row", f"t{i}", 0, ())
        for i in range(3, operator.CORPUS_UNIT_COUNT)
    )
    return tuple(rows)


def _tensor() -> operator.QuerySemanticTensor:
    facets = (
        operator.make_query_facet(0, "entity", "alpha"),
        operator.make_query_facet(1, "relation_clause", "relates to"),
    )
    dense = tuple(1000 - i for i in range(operator.CORPUS_UNIT_COUNT))
    coverage = [list(dense), list(dense)]
    coverage[0][1] = 5000
    coverage[1][2] = 6000
    anchors = [[0] * operator.CORPUS_UNIT_COUNT for _ in facets]
    anchors[0][0] = 1000
    return operator.make_query_semantic_tensor(
        query_sha256=HASH,
        facets=facets,
        semantic_coverage_ints=coverage,
        direct_anchor_strength_ints=anchors,
        dense_relevance_ints=dense,
    )


def _trace(item: str, recipe: str, value: int) -> evaluator.RecipeTrace:
    return evaluator.RecipeTrace.from_mapping(
        item_commitment_sha256=item,
        recipe_id=recipe,
        behavior_sha256=runner.stable_hash([item, recipe, value]),
        features={name: Fraction(value + index, 10) for index, name in enumerate(runner.FEATURE_ORDER)},
    )


def _matrix(count: int) -> tuple[evaluator.RecipeTrace, ...]:
    rows: list[evaluator.RecipeTrace] = []
    for item_i in range(count):
        item = runner.stable_hash(["item", item_i])
        for recipe_i, recipe in enumerate(runner.RECIPE_IDS):
            rows.append(_trace(item, recipe, recipe_i + (item_i % 3)))
    return tuple(rows)


def _policy_matrix(count: int) -> tuple[evaluator.RecipeTrace, ...]:
    patterns = {
        "R0_DENSE5": (0, 0),
        "R1_P6_DIRECT_B2": (10, 0),
        "R2_P6_PATH1_B2": (0, 1),
        "R3_P6_PATH2_B2": (0, 2),
    }
    rows: list[evaluator.RecipeTrace] = []
    for item_i in range(count):
        item = runner.stable_hash(["policy", count, item_i])
        for recipe in runner.RECIPE_IDS:
            first, remainder = patterns[recipe]
            rows.append(
                evaluator.RecipeTrace.from_mapping(
                    item_commitment_sha256=item,
                    recipe_id=recipe,
                    behavior_sha256=runner.stable_hash([item, recipe]),
                    features={
                        name: Fraction(first if index == 0 else remainder)
                        for index, name in enumerate(runner.FEATURE_ORDER)
                    },
                )
            )
    return tuple(rows)


def _distinct_policy_seal() -> runner.PolicySeal:
    a_features = runner.seal_feature_matrix(
        block="A_form", traces=_policy_matrix(runner.BLOCK_COUNTS["A_form"])
    )
    utilities = {
        (trace.item_commitment_sha256, trace.recipe_id): (
            2 if trace.recipe_id == "R1_P6_DIRECT_B2" else 0
        )
        for trace in a_features.traces
    }
    fit = runner.fit_e2(
        feature_seal=a_features, utilities=utilities, fold_secret=b"F" * 32
    )
    f_features = runner.seal_feature_matrix(
        block="F_search", traces=_policy_matrix(runner.BLOCK_COUNTS["F_search"])
    )
    return runner.freeze_f_policies(feature_seal=f_features, fit_seal=fit)


def _executions(count: int) -> tuple[runner.ItemExecution, ...]:
    graph = operator.build_typed_graph(_units())
    tensor = _tensor()
    actions = operator.run_all_recipes(graph=graph, semantic_tensor=tensor)
    rows: list[runner.ItemExecution] = []
    for item_i in range(count):
        commitment = runner.stable_hash(["anchor", count, item_i])
        traces: list[evaluator.RecipeTrace] = []
        for recipe_i, action in enumerate(actions):
            behavior = runner.stable_hash(
                {
                    "graph_sha256": action.graph_sha256,
                    "ordered_top5": list(action.output_top5),
                    "query_sha256": action.query_sha256,
                    "semantic_tensor_sha256": action.semantic_tensor_sha256,
                    "version": runner.VERSION,
                }
            )
            traces.append(
                evaluator.RecipeTrace.from_mapping(
                    item_commitment_sha256=commitment,
                    recipe_id=action.recipe_id,
                    behavior_sha256=behavior,
                    features={
                        name: Fraction(recipe_i + feature_i, 10)
                        for feature_i, name in enumerate(runner.FEATURE_ORDER)
                    },
                )
            )
        rows.append(runner.ItemExecution(commitment, actions, tuple(traces)))
    return tuple(rows)


def test_pos_facets_follow_fixed_type_order_and_deduplicate() -> None:
    facets = runner.extract_query_facets(
        "Which Alpha team had 12 wins",
        "WDT NNP NN VBD CD NNS",
    )
    assert tuple(f.facet_i for f in facets) == tuple(range(len(facets)))
    assert tuple(f.facet_type for f in facets) == tuple(
        sorted((f.facet_type for f in facets), key=operator.FACET_TYPES.index)
    )
    assert len({f.normalized_text for f in facets}) == len(facets)
    assert facets[-1].facet_type == "relation_clause"

    quota_facets = runner.extract_query_facets(
        "12 was 12 was 34 was 56 was 78 was 90 was 91",
        "CD VBD CD VBD CD VBD CD VBD CD VBD CD VBD CD",
    )
    assert sum(f.facet_type == "entity" for f in quota_facets) == 4
    assert tuple(
        f.normalized_text
        for f in quota_facets
        if f.facet_type == "numeric_or_date"
    ) == ("90", "91")
    assert len(quota_facets) == runner.FACET_LIMIT


def test_bulk_tensor_schedule_matches_per_item_and_checks_l2() -> None:
    index = _embedding_index()
    rows = (
        runner.BulkQueryInput(
            runner.stable_hash(["bulk", 1]),
            "Which Alpha team had 12 wins",
            "WDT NNP NN VBD CD NNS",
        ),
        runner.BulkQueryInput(
            runner.stable_hash(["bulk", 0]),
            "When did Beta reach 20",
            "WRB VBD NNP VB CD",
        ),
    )
    bulk_encoder = _FakeEncoder()
    bulk = runner.build_query_semantic_tensors_bulk(
        rows=rows, index=index, encoder=bulk_encoder
    )
    assert bulk_encoder.call_count == 1
    per_encoder = _FakeEncoder()
    expected = {
        row.item_commitment_sha256: runner.build_query_semantic_tensor(
            question=row.question,
            question_postag=row.question_postag,
            index=index,
            encoder=per_encoder,
        )
        for row in rows
    }
    assert bulk == dict(sorted(expected.items()))
    assert per_encoder.call_count == 2 * len(rows)

    with pytest.raises(runner.HybridQaFormalRunnerError, match="L2 normalized"):
        runner.build_query_semantic_tensors_bulk(
            rows=rows, index=index, encoder=_FakeEncoder(scale=0.5)
        )
    with pytest.raises(runner.HybridQaFormalRunnerError, match="L2 normalized"):
        runner.build_query_semantic_tensor(
            question=rows[0].question,
            question_postag=rows[0].question_postag,
            index=index,
            encoder=_FakeEncoder(scale=0.5),
        )


def test_exact_features_scan_replacements_and_form_recipe_trace() -> None:
    graph = operator.build_typed_graph(_units())
    tensor = _tensor()
    action = operator.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=tensor
    )

    values = runner.exact_action_features(graph=graph, tensor=tensor, trace=action)
    trace = runner.recipe_trace_from_action(
        item_commitment_sha256=HASH,
        graph=graph,
        tensor=tensor,
        trace=action,
    )

    assert tuple(values) == runner.FEATURE_ORDER
    assert all(isinstance(value, Fraction) for value in values.values())
    assert trace.recipe_id == "R2_P6_PATH1_B2"
    assert len(trace.features) == 8


def test_utility_is_exact_recall_plus_complete_bonus() -> None:
    assert runner.item_utility((0, 1, 2, 3, 4), (0, 4)) == (Fraction(2), True)
    assert runner.item_utility((0, 1, 2, 3, 4), (0, 8)) == (Fraction(1, 2), False)


def test_hybrid_counts_fit_and_freeze_without_feverous_block_constants() -> None:
    a_traces = _matrix(runner.BLOCK_COUNTS["A_form"])
    a_features = runner.seal_feature_matrix(block="A_form", traces=a_traces)
    utilities = {
        (trace.item_commitment_sha256, trace.recipe_id): Fraction(
            runner.RECIPE_IDS.index(trace.recipe_id), 3
        )
        for trace in a_traces
    }
    fit = runner.fit_e2(
        feature_seal=a_features, utilities=utilities, fold_secret=b"F" * 32
    )
    f_traces = _matrix(runner.BLOCK_COUNTS["F_search"])
    f_features = runner.seal_feature_matrix(block="F_search", traces=f_traces)
    policy = runner.freeze_f_policies(feature_seal=f_features, fit_seal=fit)
    receipt = fit.receipt

    assert receipt["item_count"] == 48
    assert receipt["pair_count"] == 48 * 6
    assert sum(row["held_item_count"] for row in receipt["crossfit"]) == 48
    assert "fold_secret_sha256" not in receipt
    assert receipt["feature_receipt_sha256"] == a_features.feature_receipt_sha256
    assert policy.receipt["item_count"] == 36
    assert policy.e0_recipe_id in runner.RECIPE_IDS
    assert policy.e2_recipe_id in runner.RECIPE_IDS


def test_anchor_scores_three_times_n_and_all_prespecified_families() -> None:
    items = _executions(runner.BLOCK_COUNTS["A_hold"])
    labels = tuple(
        runner.AnchorLabel(
            item.item_commitment_sha256,
            (0,),
            runner.FAMILIES[index // 10],
        )
        for index, item in enumerate(items)
    )
    hippo_rows = tuple(
        runner.HippoRetrieval(item.item_commitment_sha256, item.outputs["R0_DENSE5"])
        for item in reversed(items)
    )
    traces = tuple(trace for item in items for trace in item.recipe_traces)
    anchor_features = runner.seal_feature_matrix(block="A_hold", traces=traces)
    hippo = runner.seal_hippo_retrievals(block="A_hold", rows=hippo_rows)
    policy = _distinct_policy_seal()

    score = runner.score_anchor(
        block="A_hold",
        items=tuple(reversed(items)),
        labels=tuple(reversed(labels)),
        anchor_feature_seal=anchor_features,
        hippo_retrieval_seal=hippo,
        policy_seal=policy,
    )
    receipt = score.receipt

    assert receipt["logical_RAW_HippoRAG_Agent_work_units"] == 90
    assert receipt["anchor_feature_receipt_sha256"] == anchor_features.feature_receipt_sha256
    assert receipt["policy_receipt_sha256"] == policy.policy_receipt_sha256
    assert receipt["family_item_counts"] == {
        family: 10 for family in runner.FAMILIES
    }
    assert set(receipt["E2_minus_HippoRAG_family_sums"]) == set(runner.FAMILIES)

    wrong_labels = labels[:-1] + (
        runner.AnchorLabel("f" * 64, (0,), runner.FAMILIES[-1]),
    )
    with pytest.raises(
        runner.HybridQaFormalRunnerError, match="commitment-keyed alignment"
    ):
        runner.score_anchor(
            block="A_hold",
            items=items,
            labels=wrong_labels,
            anchor_feature_seal=anchor_features,
            hippo_retrieval_seal=hippo,
            policy_seal=policy,
        )

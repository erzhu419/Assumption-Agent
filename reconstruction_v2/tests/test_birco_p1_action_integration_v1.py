from __future__ import annotations

import copy
import inspect
from typing import Mapping

import pytest

from assumption_agent.benchmarks import birco_p1_action_integration_v1 as glue
from assumption_agent.benchmarks import birco_p1_typed_constraint_e4_core_v1 as core
from replication_runtime.birco_gpt54_semantic_v1 import contract as semantic


def _action_item(candidate_count: int = 50) -> dict[str, object]:
    work_id = "birco-work-v1-" + "1" * 64
    objective = "Rank candidates against every typed condition."
    query = "Find alpha, require beta, and exclude gamma."
    candidates = tuple(
        semantic.project_candidate_text(
            f"Candidate {ordinal} states alpha. Beta evidence {ordinal}; gamma status unknown.",
            candidate_ordinal=ordinal,
        )
        for ordinal in range(candidate_count)
    )
    documents = [
        {"ordinal": row.ordinal, "text": row.projection_text} for row in candidates
    ]
    pool_hash = semantic.semantic_hash(
        {"documents": documents, "objective": objective, "query": query}
    )
    return {
        "schema": glue.SELECTOR_ACTION_ITEM_SCHEMA,
        "block_ordinal": 3,
        "work_id": work_id,
        "candidate_count": candidate_count,
        "common_projection_sha256": pool_hash,
        "hipporag_input": {
            "schema": glue.HIPPORAG_INPUT_SCHEMA,
            "work_id": work_id,
            "objective": objective,
            "query": query,
            "documents": documents,
            "common_projection_sha256": pool_hash,
        },
    }


def _provider() -> dict[str, object]:
    return {
        "api_key_hmac_sha256": "2" * 64,
        "api_origin": glue.PROVIDER_ORIGIN,
        "key_commitment_version": glue.KEY_COMMITMENT_VERSION,
        "model": semantic.MODEL_ID,
        "provider_label": "synthetic",
        "secret_persisted": False,
    }


def _terminal(
    mode: str,
    expected_input: Mapping[str, object],
    action: Mapping[str, object],
    *,
    generation_valid: bool = True,
) -> dict[str, object]:
    body: dict[str, object] = {
        "action": dict(action),
        "attempt_count": 1,
        "generation_valid": generation_valid,
        "input_sha256": semantic.semantic_hash(expected_input),
        "mode": mode,
        "model_request_sha256": "3" * 64,
        "provider": _provider(),
        "raw_completion_persisted": False,
        "response_sha256": "4" * 64,
        "retry_replay_resample_or_provider_switch_count": 0,
        "schema": semantic.TERMINAL_OUTPUT_SCHEMA,
        "terminal_category": "success" if generation_valid else "output_totalized",
        "transport": glue.SEMANTIC_TRANSPORT_ID,
        "transport_succeeded": True,
        "work_id": expected_input["work_id"],
    }
    if mode in {"matrix", "raw"}:
        for name in (
            "batch_count",
            "batch_ordinal",
            "batch_common_projection_sha256",
            "pool_candidate_count",
            "pool_common_projection_sha256",
        ):
            body[name] = expected_input[name]
    return {**body, "self_sha256": semantic.semantic_hash(body)}


def _semantic_plan() -> semantic.Plan:
    return semantic.Plan(
        facets=(
            semantic.Facet(0, "REQUIRED", "alpha is present", 4),
            semantic.Facet(1, "REQUIRED", "beta is present", 4),
            semantic.Facet(2, "EXCLUDED", "gamma is present", 3),
        ),
        edges=(semantic.PlanEdge(0, 1, "REQUIRES"),),
        generation_valid=True,
    )


def _matrix_terminals(
    stage: glue.CanonicalMatrixInputs,
) -> tuple[dict[str, object], ...]:
    terminals = []
    for expected_input in stage.matrix_inputs:
        rows = []
        candidates = expected_input["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            ordinal = candidate["ordinal"]
            evidence_units = candidate["evidence_units"]
            evidence_count = len(evidence_units)
            rows.append(
                {
                    "ordinal": ordinal,
                    "rows": [
                        [4 if ordinal % 3 == 0 else 1, 0, 0],
                        [3 if ordinal % 2 == 0 else 1, 0, min(1, evidence_count - 1)],
                        [0, 4 if ordinal % 5 == 0 else 0, None],
                    ],
                }
            )
        terminals.append(
            _terminal("matrix", expected_input, {"matrix": rows})
        )
    return tuple(terminals)


def _raw_terminals(
    prepared: glue.CanonicalActionInputs,
) -> tuple[dict[str, object], ...]:
    terminals = []
    for expected_input in prepared.raw_inputs:
        candidates = expected_input["candidates"]
        assert isinstance(candidates, list)
        rows = [
            {"ordinal": row["ordinal"], "score": 100 - (row["ordinal"] % 7)}
            for row in candidates
        ]
        terminals.append(_terminal("raw", expected_input, {"scores": rows}))
    return tuple(terminals)


def _zero_e4_model() -> core.E4Model:
    return core.E4Model(
        population_mean=(0.0,) * len(core.FEATURE_ORDER),
        population_std=(1.0,) * len(core.FEATURE_ORDER),
        coefficients=(0.0,) * len(core.FEATURE_ORDER),
        laplace_covariance=tuple(
            (0.0,) * len(core.FEATURE_ORDER)
            for _ in range(len(core.FEATURE_ORDER))
        ),
        solver="synthetic_zero",
        iterations=0,
        converged=True,
        objective=0.0,
    )


def _prepared_stage(candidate_count: int = 50):
    prepared = glue.prepare_canonical_action_inputs(_action_item(candidate_count))
    plan = _semantic_plan()
    plan_terminal = _terminal(
        "plan", prepared.planner_input, {"plan": plan.payload()}
    )
    stage = glue.build_canonical_matrix_inputs(prepared, plan_terminal)
    return prepared, stage


def test_selector_projection_validation_reopens_semantic_candidates_only() -> None:
    item = _action_item(25)
    checked = glue.validate_selector_action_item(item)
    assert checked.candidate_count == 25
    assert checked.batch_count == 2
    assert tuple(row.ordinal for row in checked.candidates) == tuple(range(25))
    assert checked.common_projection_sha256 == item["common_projection_sha256"]

    leaked = copy.deepcopy(item)
    leaked["qrel_values"] = [1]
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="schema drifted"):
        glue.validate_selector_action_item(leaked)
    tampered = copy.deepcopy(item)
    tampered["hipporag_input"]["documents"][0]["text"] += " "  # type: ignore[index]
    with pytest.raises(
        glue.BircoP1ActionIntegrationError, match="not canonical"
    ):
        glue.validate_selector_action_item(tampered)


def test_full_pool_hash_and_fixed_24_slices_bind_plan_and_raw_inputs() -> None:
    prepared = glue.prepare_canonical_action_inputs(_action_item(50))
    assert prepared.action_item.batch_count == 3
    assert tuple(map(len, prepared.batch_candidate_ordinals)) == (24, 24, 2)
    assert prepared.batch_candidate_ordinals[0] == tuple(range(24))
    assert prepared.batch_candidate_ordinals[1] == tuple(range(24, 48))
    assert prepared.batch_candidate_ordinals[2] == (48, 49)
    assert prepared.planner_input["schema"] == semantic.PLAN_INPUT_SCHEMA
    assert len(prepared.raw_inputs) == 3
    for batch_ordinal, payload in enumerate(prepared.raw_inputs):
        assert payload["schema"] == semantic.RAW_INPUT_SCHEMA
        assert payload["batch_ordinal"] == batch_ordinal
        assert payload["batch_count"] == 3
        assert payload["pool_candidate_count"] == 50
        assert (
            payload["pool_common_projection_sha256"]
            == prepared.pool_common_projection_sha256
        )


def test_plan_terminal_builds_canonical_matrix_inputs_and_binds_hashes() -> None:
    prepared, stage = _prepared_stage(50)
    assert stage.semantic_plan == _semantic_plan()
    assert len(stage.core_plan.facets) == 3
    assert stage.core_plan.edges[0].edge_type == "REQUIRES"
    assert len(stage.matrix_inputs) == 3
    for ordinal, payload in enumerate(stage.matrix_inputs):
        assert payload["schema"] == semantic.MATRIX_INPUT_SCHEMA
        assert payload["batch_ordinal"] == ordinal
        assert payload["plan"] == stage.semantic_plan.payload()
        assert payload["pool_common_projection_sha256"] == (
            prepared.pool_common_projection_sha256
        )

    plan = _semantic_plan()
    terminal = _terminal("plan", prepared.planner_input, {"plan": plan.payload()})
    terminal["input_sha256"] = "f" * 64
    body = dict(terminal)
    body.pop("self_sha256")
    terminal["self_sha256"] = semantic.semantic_hash(body)
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="input binding"):
        glue.build_canonical_matrix_inputs(prepared, terminal)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda row: row.__setitem__("attempt_count", 2), "control fields"),
        (
            lambda row: row.__setitem__(
                "retry_replay_resample_or_provider_switch_count", 1
            ),
            "control fields",
        ),
        (
            lambda row: row["provider"].__setitem__("model", "other-model"),
            "provider receipt",
        ),
        (
            lambda row: row["provider"].__setitem__("api_key", "leak"),
            "provider receipt schema",
        ),
    ),
)
def test_terminal_provider_model_one_attempt_and_no_retry_are_strict(
    mutate, message: str
) -> None:
    prepared = glue.prepare_canonical_action_inputs(_action_item(25))
    plan = _semantic_plan()
    terminal = _terminal("plan", prepared.planner_input, {"plan": plan.payload()})
    mutate(terminal)
    body = dict(terminal)
    body.pop("self_sha256")
    terminal["self_sha256"] = semantic.semantic_hash(body)
    with pytest.raises(glue.BircoP1ActionIntegrationError, match=message):
        glue.build_canonical_matrix_inputs(prepared, terminal)


def test_compact_rows_convert_by_canonical_facet_index_and_merge_every_batch() -> None:
    _prepared, stage = _prepared_stage(50)
    terminals = _matrix_terminals(stage)
    matrix = glue.merge_matrix_terminals(stage, tuple(reversed(terminals)))
    assert matrix.candidate_count == 50
    assert tuple(row.candidate_ordinal for row in matrix.candidates) == tuple(range(50))
    first = matrix.candidates[0]
    assert tuple(row.facet_ordinal for row in first.facet_evidence) == (0, 1, 2)
    assert first.facet_evidence[0] == core.FacetEvidence(0, 4, 0, 0)
    assert first.facet_evidence[2] == core.FacetEvidence(2, 0, 4, None)
    assert first.evidence_unit_count == len(
        stage.prepared.action_item.candidates[0].evidence_units
    )

    missing = terminals[:-1]
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="batch count"):
        glue.merge_matrix_terminals(stage, missing)
    malformed = copy.deepcopy(terminals)
    malformed[0]["action"]["matrix"][0]["rows"][0] = [4, 0]  # type: ignore[index]
    body = dict(malformed[0])
    body.pop("self_sha256")
    malformed[0]["self_sha256"] = semantic.semantic_hash(body)
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="width three"):
        glue.merge_matrix_terminals(stage, malformed)

    repeated_candidate = copy.deepcopy(terminals)
    repeated_candidate[0]["action"]["matrix"][1]["ordinal"] = 0  # type: ignore[index]
    body = dict(repeated_candidate[0])
    body.pop("self_sha256")
    repeated_candidate[0]["self_sha256"] = semantic.semantic_hash(body)
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="ordinal"):
        glue.merge_matrix_terminals(stage, repeated_candidate)


def test_terminal_totalized_matrix_is_valid_but_tampering_fails_closed() -> None:
    _prepared, stage = _prepared_stage(25)
    terminals = list(_matrix_terminals(stage))
    totalized_action = {
        "matrix": [
            {
                "ordinal": candidate["ordinal"],
                "rows": [[0, 0, None]] * len(stage.core_plan.facets),
            }
            for candidate in stage.matrix_inputs[0]["candidates"]
        ]
    }
    terminals[0] = _terminal(
        "matrix",
        stage.matrix_inputs[0],
        totalized_action,
        generation_valid=False,
    )
    matrix = glue.merge_matrix_terminals(stage, terminals)
    assert all(
        row.support == row.contradiction == 0
        for row in matrix.candidates[0].facet_evidence
    )

    tampered = copy.deepcopy(terminals)
    tampered[1]["batch_common_projection_sha256"] = "e" * 64
    body = dict(tampered[1])
    body.pop("self_sha256")
    tampered[1]["self_sha256"] = semantic.semantic_hash(body)
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="binding drifted"):
        glue.merge_matrix_terminals(stage, tampered)


def test_raw_terminal_merge_covers_full_pool_and_uses_ordinal_ties() -> None:
    prepared = glue.prepare_canonical_action_inputs(_action_item(50))
    terminals = _raw_terminals(prepared)
    result = glue.merge_raw_terminals(prepared, tuple(reversed(terminals)))
    assert len(result.scores_by_candidate_ordinal) == 50
    assert set(result.candidate_ordinals) == set(range(50))
    assert result.candidate_ordinals[:3] == (0, 7, 14)

    duplicate = (terminals[0], terminals[0], terminals[2])
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="duplicated"):
        glue.merge_raw_terminals(prepared, duplicate)
    malformed = copy.deepcopy(terminals)
    malformed[1]["action"]["scores"][0]["ordinal"] = 0  # type: ignore[index]
    body = dict(malformed[1])
    body.pop("self_sha256")
    malformed[1]["self_sha256"] = semantic.semantic_hash(body)
    with pytest.raises(glue.BircoP1ActionIntegrationError, match="ordinal"):
        glue.merge_raw_terminals(prepared, malformed)


def test_four_rankings_features_and_e0_e4_are_complete_and_content_free() -> None:
    _prepared, stage = _prepared_stage(50)
    evaluation = glue.produce_e0_e4_evaluation(
        stage, _matrix_terminals(stage), e4_model=_zero_e4_model()
    )
    assert tuple(ranking.recipe_id for ranking in evaluation.rankings) == core.RECIPE_IDS
    assert tuple(recipe for recipe, _ in evaluation.action_features) == core.RECIPE_IDS
    for ranking in evaluation.rankings:
        assert len(ranking.candidate_ordinals) == 50
        assert set(ranking.candidate_ordinals) == set(range(50))
    for _recipe, values in evaluation.action_features:
        assert len(values) == len(core.FEATURE_ORDER) == 12
    assert evaluation.e0_recipe_id == core.R3_DEPENDENCY_FLOW
    assert evaluation.e4_selection.selected_recipe_id == evaluation.e0_recipe_id
    assert evaluation.e4_ranking == evaluation.e0_ranking


def test_public_glue_surface_has_no_qrel_source_file_network_or_secret_input() -> None:
    forbidden = {
        "qrel",
        "qrels",
        "family",
        "source",
        "source_path",
        "path",
        "url",
        "api_key",
        "secret",
    }
    for function in (
        glue.validate_selector_action_item,
        glue.prepare_canonical_action_inputs,
        glue.build_canonical_matrix_inputs,
        glue.merge_matrix_terminals,
        glue.merge_raw_terminals,
        glue.produce_e0_e4_evaluation,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)

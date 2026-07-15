from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks.execution_contract_prompt_v2 import (
    _instruction_payload,
)
from assumption_agent.models import ResidualExample, SplitName, stable_hash
from assumption_agent.typed_execution_contract import (
    CompletionCheckKind,
    CompletionPhaseKind,
    InvariantKind,
    RuntimeRole,
    TRACE_REFINED_ORGANIZATION_CONTRACT_VERSION,
    derive_train_execution_contract,
    derive_train_trace_refined_organization_contract_v2,
    verify_execution_contract_payload,
)
from assumption_agent.typed_operator_grammar import (
    PrimitiveRef,
    TrialTraceEvidence,
    WorkflowKind,
    build_family_capability_graph,
)


def _residual(
    *,
    family: str,
    index: int,
    profile_hash: str,
    split: SplitName = SplitName.TRAIN,
    feedback: tuple[str, ...] = (),
    context_extra: Mapping[str, Any] | None = None,
) -> ResidualExample:
    return ResidualExample(
        transition_id=f"transition-{index}",
        task_id=f"item-{index}",
        family=family,
        split=split,
        features={"family": family},
        failure_type="synthetic_policy_off_failure",
        evaluator_feedback=feedback,
        baseline_success=False,
        context={
            "action_context_profile_hash": profile_hash,
            **dict(context_extra or {}),
        },
    )


def _graph_fixture(*, path: str = "/root/data.csv"):
    family = "synthetic-family"
    profile = {
        "runtime_environment": {
            "declared_task_local_paths": [path],
            "copied_task_files": [path],
            "environment_source_files": [path.rsplit("/", 1)[-1]],
        },
        "baseline_action_trace": {},
    }
    profile_hash = stable_hash(profile)
    residuals = tuple(
        _residual(
            family=family,
            index=index,
            profile_hash=profile_hash,
        )
        for index in range(1, 4)
    )
    trials = {
        stable_hash({"item_id": residual.task_id}): TrialTraceEvidence(
            trial_id_hash=stable_hash({"item_id": residual.task_id}),
            family_hash=stable_hash({"family": family}),
            trace_hash=stable_hash({"trace": residual.task_id}),
            action_budget_receipt_hash=stable_hash(
                {"receipt": residual.task_id}
            ),
            action_event_hash=stable_hash({"events": residual.task_id}),
            baseline_success=False,
            action_budget_limit=100,
            trace_complete=True,
            action_start_count=1,
            command_span_count=0,
            discarded_command_count=0,
            changed_artifacts=(PrimitiveRef("artifact", path),),
            spans=(),
        )
        for residual in residuals
    }
    graph = build_family_capability_graph(
        target_family=family,
        failures=residuals,
        action_profiles={profile_hash: profile},
        trial_evidence=trials,
    )
    return graph, residuals, profile_hash


def _recipe(graph, workflow: WorkflowKind):
    return next(row for row in graph.recipes if row.workflow is workflow)


def test_contract_is_deterministic_closed_and_does_not_mutate_v1_graph() -> None:
    graph, residuals, _ = _graph_fixture()
    recipe = _recipe(graph, WorkflowKind.DERIVE_TASK_OUTPUT)
    graph_payload_before = deepcopy(graph.safe_payload())
    graph_hash_before = graph.graph_hash

    first = derive_train_execution_contract(
        graph=graph,
        recipe_id=recipe.recipe_id,
        residuals=residuals,
    )
    replay = derive_train_execution_contract(
        graph=graph,
        recipe_id=recipe.recipe_id,
        residuals=tuple(reversed(residuals)),
    )

    assert first == replay
    assert first.contract_hash == replay.contract_hash
    assert first.validate(graph) == ()
    assert graph.safe_payload() == graph_payload_before
    assert graph.graph_hash == graph_hash_before
    assert all(len(row.supports) == 3 for row in first.invariants)
    assert {
        row.kind for row in first.invariants
    } == {
        InvariantKind.PRIMARY_ARTIFACT_READ_BEFORE_MUTATION,
        InvariantKind.TASK_DELTA_ONLY,
        InvariantKind.FINAL_OUTPUT_REOPENED,
    }
    assert first.completion.phase_order == (
        CompletionPhaseKind.APPLY_REGISTERED_MUTATION,
        CompletionPhaseKind.REOPEN_MATERIALIZED_OUTPUT,
        CompletionPhaseKind.CHECK_CLOSED_INVARIANTS,
        CompletionPhaseKind.BOUNDED_REPAIR,
        CompletionPhaseKind.FINALIZE_EFFECT_RECEIPT,
    )
    assert first.completion.self_evaluation_source_role is (
        RuntimeRole.FINAL_MATERIALIZED_OUTPUT
    )
    assert first.resources.max_repair_attempts < first.resources.max_mutations
    text = json.dumps(first.safe_payload(), sort_keys=True)
    for forbidden in (
        "/root",
        "synthetic-family",
        "transition-1",
        "item-1",
        "verifier",
        "validation",
        "arbitrary_shell_command",
    ):
        assert forbidden not in text


def test_feedback_context_and_non_target_family_cannot_change_contract() -> None:
    graph, residuals, profile_hash = _graph_fixture()
    recipe = _recipe(graph, WorkflowKind.BUILD_VISUALIZATION)
    clean = derive_train_execution_contract(
        graph=graph,
        recipe_id=recipe.recipe_id,
        residuals=residuals,
    )
    poisoned_train_rows = tuple(
        replace(
            residual,
            evaluator_feedback=(
                "verifier literal must not enter the candidate",
            ),
            context={
                "action_context_profile_hash": profile_hash,
                "untrusted_validation_or_test_text": "hidden answer",
            },
            failure_type="different_untrusted_label",
        )
        for residual in residuals
    )
    other_family = tuple(
        replace(
            residual,
            transition_id=f"other-{index}",
            task_id=f"other-item-{index}",
            family="other-family",
        )
        for index, residual in enumerate(residuals, start=1)
    )
    replay = derive_train_execution_contract(
        graph=graph,
        recipe_id=recipe.recipe_id,
        residuals=(*poisoned_train_rows, *other_family),
    )

    assert replay == clean
    assert CompletionCheckKind.REPLAY_OBSERVABLE_INTERACTION in (
        replay.completion.checks
    )
    assert "verifier literal" not in json.dumps(replay.safe_payload())

    validation_row = replace(residuals[0], split=SplitName.VALIDATION)
    with pytest.raises(PermissionError, match="non-TRAIN"):
        derive_train_execution_contract(
            graph=graph,
            recipe_id=recipe.recipe_id,
            residuals=(*residuals, validation_row),
        )


def test_contract_requires_two_independent_same_family_train_failures() -> None:
    graph, residuals, _ = _graph_fixture()
    recipe = _recipe(graph, WorkflowKind.TRANSFORM_IN_PLACE)

    with pytest.raises(PermissionError, match="two independent"):
        derive_train_execution_contract(
            graph=graph,
            recipe_id=recipe.recipe_id,
            residuals=(residuals[0],),
        )
    duplicate_transition = replace(
        residuals[1],
        transition_id=residuals[0].transition_id,
        task_id=residuals[0].task_id,
    )
    with pytest.raises(PermissionError, match="two independent"):
        derive_train_execution_contract(
            graph=graph,
            recipe_id=recipe.recipe_id,
            residuals=(residuals[0], duplicate_transition),
        )
    successful = replace(residuals[1], baseline_success=True)
    with pytest.raises(PermissionError, match="two independent"):
        derive_train_execution_contract(
            graph=graph,
            recipe_id=recipe.recipe_id,
            residuals=(residuals[0], successful),
        )


def test_trace_refined_organization_contract_is_closed_and_literal_free() -> None:
    graph, residuals, _ = _graph_fixture(path="/root/files")
    recipe = _recipe(graph, WorkflowKind.ORGANIZE_COLLECTION)

    contract = derive_train_trace_refined_organization_contract_v2(
        graph=graph,
        recipe_id=recipe.recipe_id,
        residuals=residuals,
    )

    assert contract.contract_version == (
        TRACE_REFINED_ORGANIZATION_CONTRACT_VERSION
    )
    assert contract.validate(graph) == ()
    assert {
        InvariantKind.ORGANIZATION_DESTINATIONS_FROM_PUBLIC_TASK,
        InvariantKind.ORGANIZATION_ASSIGNMENTS_REQUIRE_POSITIVE_EVIDENCE,
        InvariantKind.ORGANIZATION_DESTINATION_LAYOUT_REOPENED,
    }.issubset({row.kind for row in contract.invariants})
    assert len(contract.invariants) == 6
    payload_text = json.dumps(contract.safe_payload(), sort_keys=True)
    instruction_text = json.dumps(
        _instruction_payload(contract),
        sort_keys=True,
    )
    assert "siblings of" in instruction_text
    assert "fallback or catch-all" in instruction_text
    assert "frozen pre-move manifest" in instruction_text
    for forbidden in (
        "/root",
        "data.csv",
        "synthetic-family",
        "item-1",
        "verifier",
        "hidden answer",
    ):
        assert forbidden not in payload_text

    wrong_version = replace(
        contract,
        contract_version="unknown_trace_refinement",
    )
    assert "execution_contract_version_mismatch" in (
        wrong_version.validate(graph)
    )

    other_recipe = _recipe(graph, WorkflowKind.TRANSFORM_IN_PLACE)
    with pytest.raises(PermissionError, match="does not support"):
        derive_train_trace_refined_organization_contract_v2(
            graph=graph,
            recipe_id=other_recipe.recipe_id,
            residuals=residuals,
        )


def test_contract_and_payload_tampering_fail_closed() -> None:
    graph, residuals, _ = _graph_fixture()
    recipe = _recipe(graph, WorkflowKind.DERIVE_TASK_OUTPUT)
    contract = derive_train_execution_contract(
        graph=graph,
        recipe_id=recipe.recipe_id,
        residuals=residuals,
    )

    payload = contract.safe_payload()
    verify_execution_contract_payload(contract, payload, graph=graph)
    tampered_payload = deepcopy(payload)
    tampered_payload["resources"]["max_action_starts"] = 101
    with pytest.raises(PermissionError, match="payload or hash drifted"):
        verify_execution_contract_payload(
            contract,
            tampered_payload,
            graph=graph,
        )

    wrong_graph_hash = replace(contract, graph_hash="f" * 64)
    assert "execution_contract_graph_hash_mismatch" in (
        wrong_graph_hash.validate(graph)
    )
    duplicated_support = replace(
        contract.invariants[0],
        supports=(
            contract.invariants[0].supports[0],
            contract.invariants[0].supports[0],
        ),
    )
    bad_invariants = tuple(
        sorted(
            (duplicated_support, *contract.invariants[1:]),
            key=lambda row: (row.kind.value, row.operation.value),
        )
    )
    duplicate_contract = replace(contract, invariants=bad_invariants)
    assert "invariant_independent_support_insufficient" in (
        duplicate_contract.validate(graph)
    )
    wrong_budget = replace(
        contract,
        resources=replace(contract.resources, max_repair_attempts=8),
    )
    assert "resource_repair_attempt_limit_invalid" in (
        wrong_budget.validate(graph)
    )


def test_configure_and_run_binds_finite_search_space_and_count_receipt() -> None:
    graph, residuals, _ = _graph_fixture(path="/root/temperature.nc")
    recipe = _recipe(graph, WorkflowKind.CONFIGURE_AND_RUN)
    with pytest.raises(PermissionError, match="finite search space"):
        derive_train_execution_contract(
            graph=graph,
            recipe_id=recipe.recipe_id,
            residuals=residuals,
        )

    candidate_hashes = tuple(
        stable_hash({"candidate": index}) for index in range(5)
    )
    contract = derive_train_execution_contract(
        graph=graph,
        recipe_id=recipe.recipe_id,
        residuals=residuals,
        search_candidate_hashes=tuple(reversed(candidate_hashes)),
    )

    assert contract.validate(graph) == ()
    assert contract.search_space.candidate_hashes == tuple(
        sorted(candidate_hashes)
    )
    assert contract.search_space.evaluation_limit == 5
    assert contract.resources.max_search_evaluations == 5
    assert CompletionCheckKind.VERIFY_SEARCH_EVALUATION_COUNT in (
        contract.completion.checks
    )
    assert InvariantKind.FINITE_SEARCH_SPACE_DECLARED in {
        row.kind for row in contract.invariants
    }
    payload = contract.safe_payload()
    assert payload["search_space"]["candidate_count"] == 5
    assert payload["search_space"]["search_space_hash"] == (
        contract.search_space.search_space_hash
    )
    assert payload["resources"]["runtime_receipt_required"] is True
    assert payload["runtime_enforcement_claimed"] is False

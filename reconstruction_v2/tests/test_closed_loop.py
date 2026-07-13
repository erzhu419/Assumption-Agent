from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import pytest

from assumption_agent.archive import (
    AnchorScore,
    EvaluatorEpoch,
    EvaluatorEpochController,
    EvaluatorSpec,
    PolicyArchive,
)
from assumption_agent.evaluation import (
    CANDIDATE_MAY_ONLY_TIGHTEN,
    CounterfactualRunner,
    PromotionGate,
    PromotionGateSpec,
    RUNTIME_BASELINE_PRESERVATION,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.evolution import (
    CANDIDATE_BUNDLE_POLICY_VERSION,
    COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION,
    COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION,
    CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    CounterfactualEvidenceReplayCache,
    EvolutionKernel,
    EvolutionRunResult,
)
from assumption_agent.models import (
    ExternalOutcome,
    HypothesisProgram,
    HypothesisStatus,
    LaneResult,
    ResidualExample,
    SplitName,
    TaskInput,
    stable_hash,
)
from assumption_agent.proposer import (
    HypothesisProposalCallError,
    PROPOSAL_DIVERSITY_POLICY_VERSION,
    REPAIR_BRANCH_ID_POLICY_VERSION,
    REPAIR_REQUEST_SCOPE_POLICY_VERSION,
    StructuredHypothesisProposer,
)
from assumption_agent.runtime import LaneRegistry, PolicyRuntime
from assumption_agent.splits import SplitAccessGuard, SplitManifest
from assumption_agent.validation import (
    EvaluatorEpochCheck,
    RecursiveValidationEngine,
    RuntimeCandidateKindCheck,
    RuntimeActionCheck,
    SchemaCheck,
    TrainingSupportCheck,
    TriggerVocabularyCheck,
    ValidationContext,
    build_trigger_feature_catalog,
)


@dataclass
class StaticLane:
    name: str
    answer: str
    confidence: float
    cost: float

    def run(self, task: TaskInput, parameters: Mapping[str, Any]) -> LaneResult:
        return LaneResult(
            lane=self.name,
            answer=self.answer,
            confidence=self.confidence,
            cost=self.cost,
        )


class TruthEvaluator:
    id = "external_truth"
    epoch = "epoch-0"

    def evaluate(self, task: TaskInput, execution) -> ExternalOutcome:
        success = execution.selected_result.answer == task.payload["expected"]
        return ExternalOutcome(
            task_id=task.id,
            success=success,
            score=float(success),
            evaluator_id=self.id,
            evaluator_epoch=self.epoch,
            metrics={
                "task_success": float(success),
                "application_fidelity": float(execution.action_activated),
            },
        )


def _promotion_spec(
    *,
    minimum_pairs: int = 10,
    minimum_net_gain_count: int = 2,
) -> PromotionGateSpec:
    return PromotionGateSpec(
        metric="task_success",
        minimum_pairs=minimum_pairs,
        confidence=0.9,
        minimum_net_gain_count=minimum_net_gain_count,
        minimum_activation_rate=0.1,
        minimum_effect_lower_bound=0.0,
        maximum_harm_rate=0.05,
        maximum_cost_ratio=3.0,
        baseline_safety_policy=RUNTIME_BASELINE_PRESERVATION,
        candidate_threshold_policy=CANDIDATE_MAY_ONLY_TIGHTEN,
    )


class QueueProposalModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[Mapping[str, Any]] = []

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(payload)
        return self.responses.pop(0)


def test_policy_program_changes_runtime_and_preserves_baseline() -> None:
    sink = MemoryEventSink()
    runtime = _runtime(sink)
    program = HypothesisProgram.from_dict(_program_dict(status="promoted"))
    task = _task("task-1")

    execution = runtime.execute(task, (program,))

    assert execution.selected_result.lane == "relation_solver"
    assert execution.selected_result.answer == "correct"
    assert execution.action_activated is True
    assert execution.baseline_preserved is True
    assert {result.lane for result in execution.lane_results} == {"raw", "relation_solver"}
    assert any(row["event"] == "hypothesis_runtime_decision" for row in sink.events)

    miss = runtime.execute(
        TaskInput(id="task-miss", family="other", features={"relation_type": "none"}, payload={"expected": "correct"}),
        (program,),
    )
    assert miss.selected_result.lane == "raw"
    assert miss.action_activated is False


def test_failed_hypothesis_is_repaired_and_recursively_revalidated() -> None:
    sink = MemoryEventSink()
    bad = _program_dict(lane="missing_lane")
    bad["expected_effect"]["metric"] = "root_self_reported_metric"
    repaired = _program_dict(hypothesis_id="hyp-repaired")
    repaired["expected_effect"]["metric"] = "repair_self_reported_metric"
    model = QueueProposalModel([
        {"hypotheses": [bad]},
        {"hypothesis": repaired},
    ])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    residuals = _residuals()
    root = proposer.propose(residuals, evaluator_epoch="epoch-0", max_hypotheses=1)[0]
    validator = _validator(proposer, sink)

    tree = validator.validate(root, _validation_context(residuals))

    assert len(tree.nodes) == 2
    assert tree.nodes[0].passed is False
    assert tree.accepted_program is not None
    assert tree.accepted_program.id.startswith("repair_")
    assert tree.accepted_program.id != "hyp-repaired"
    assert tree.accepted_program.parent_id == root.id
    assert tree.accepted_program.lineage == (root.id,)
    assert root.expected_effect.metric == "task_success"
    assert tree.accepted_program.expected_effect.metric == "task_success"
    assert all(not _contains_forbidden_answer_key(request) for request in model.requests)
    repair_event = next(
        row for row in sink.events if row["event"] == "hypothesis_repair_proposed"
    )
    assert (
        repair_event["payload"]["branch_id_policy"]
        == REPAIR_BRANCH_ID_POLICY_VERSION
    )
    assert repair_event["payload"]["child_id"] == (
        f"repair_{repair_event['payload']['branch_identity_hash']}"
    )
    assert repair_event["payload"]["model_supplied_child_id_used"] is False


def test_repair_request_excludes_root_batch_contract_but_retains_coverage() -> None:
    model = QueueProposalModel(
        [{"hypothesis": _program_dict(hypothesis_id="scoped-repair")}]
    )
    sink = MemoryEventSink()
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    parent = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="repair-scope-parent")
    )
    coverage_objective = {
        "policy": "train_family_coverage_v1",
        "coverage_unit": "distinct_failure_family",
        "failure_activation_family_target": 2,
        "validation_outcomes_used": False,
    }
    capabilities = {
        "repair_request_scope_policy": REPAIR_REQUEST_SCOPE_POLICY_VERSION,
        "proposal_batch_contract": {
            "policy": PROPOSAL_DIVERSITY_POLICY_VERSION,
            "required_count": 3,
            "max_action_nodes_per_hypothesis": 4,
        },
        "train_coverage_objective": coverage_objective,
    }

    proposer.revise(
        parent,
        failed_checks=({"check": "runtime_action", "passed": False},),
        residuals=_residuals(),
        depth=1,
        capabilities=capabilities,
        trace_id="repair-request-scope",
    )

    request = model.requests[0]
    assert request["repair_request_scope_policy"] == (
        REPAIR_REQUEST_SCOPE_POLICY_VERSION
    )
    assert request["repair_response_contract"] == {
        "response_field": "hypothesis",
        "response_type": "object",
        "required_count": 1,
        "root_batch_contract_applies": False,
        "compact_output": True,
    }
    assert "proposal_batch_contract" not in request
    assert "proposal_batch_contract" not in request["capabilities"]
    assert request["capabilities"]["train_coverage_objective"] == coverage_objective
    assert "proposal_count_must_equal_requested_count" not in request["constraints"]
    assert request["constraints"]["candidate_search_uses_train_only"] is True
    assert request["constraints"]["candidate_search_family_target"] == 2
    assert set(request["output_schema"]) == {"hypothesis"}
    assert "hypotheses" not in request["output_schema"]
    assert "proposal_batch_contract" in capabilities
    requested = next(
        row for row in sink.events if row["event"] == "hypothesis_repair_requested"
    )
    assert requested["payload"]["repair_request_scope_policy"] == (
        REPAIR_REQUEST_SCOPE_POLICY_VERSION
    )


def test_repair_model_failure_rejects_only_that_candidate_branch() -> None:
    sink = MemoryEventSink()
    bad = _program_dict(lane="missing_lane")
    model = QueueProposalModel([{"hypotheses": [bad]}])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    residuals = _residuals()
    root = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        max_hypotheses=1,
    )[0]

    tree = _validator(proposer, sink).validate(
        root,
        _validation_context(residuals),
        trace_id="repair-model-failure",
    )

    assert tree.accepted_program is None
    assert tree.repair_model_failure_count == 1
    assert tree.nodes[0].terminal_reason == "repair_model_failed"
    assert any(
        row["event"] == "hypothesis_proposal_model_call_failed"
        for row in sink.events
    )
    assert any(
        row["event"] == "hypothesis_repair_abandoned_after_model_failure"
        and row["payload"]["candidate_local_failure"] is True
        for row in sink.events
    )


@pytest.mark.parametrize(
    ("case", "expected_phase"),
    [
        ("top_level_list", "response_envelope"),
        ("missing_field", "response_envelope"),
        ("nonlist_field", "response_envelope"),
        ("empty_list", "response_envelope"),
        ("nonmapping_row", "response_envelope"),
        ("parse_error", "response_program_parse"),
    ],
)
def test_malformed_root_response_contract_is_typed_and_sanitized(
    case: str,
    expected_phase: str,
) -> None:
    invalid_program = _program_dict(hypothesis_id="invalid-root-response")
    invalid_program["kind"] = "not-a-hypothesis-kind"
    responses: dict[str, Any] = {
        "top_level_list": [],
        "missing_field": {},
        "nonlist_field": {"hypotheses": {}},
        "empty_list": {"hypotheses": []},
        "nonmapping_row": {"hypotheses": ["not-an-object"]},
        "parse_error": {"hypotheses": [invalid_program]},
    }
    response = responses[case]
    sink = MemoryEventSink()
    model = QueueProposalModel([response])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)

    with pytest.raises(HypothesisProposalCallError) as caught:
        proposer.propose(
            _residuals(),
            evaluator_epoch="epoch-0",
            max_hypotheses=2,
            trace_id=f"malformed-root-{case}",
        )

    error = caught.value
    assert error.request_kind == "propose_hypothesis_programs"
    assert error.failure_phase == expected_phase
    assert error.response_hash == stable_hash(response)
    rejected = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_proposal_response_rejected"
    )
    assert rejected["payload"]["request_hash"] == error.request_hash
    assert rejected["payload"]["response_hash"] == error.response_hash
    assert rejected["payload"]["candidate_local_failure"] is False
    assert rejected["payload"]["raw_content_persisted"] is False
    assert "top_level_key_set_hash" in rejected["payload"]
    assert not any(row["event"] == "hypothesis_proposed" for row in sink.events)
    assert not any(
        row["event"] == "root_proposal_evidence_recorded" for row in sink.events
    )


def test_mixed_root_response_is_atomic_and_not_replayed_after_rejection() -> None:
    sink = MemoryEventSink()
    valid = _program_dict(hypothesis_id="atomic-valid-root")
    malformed = _program_dict(hypothesis_id="atomic-malformed-root")
    malformed["kind"] = "not-a-hypothesis-kind"
    model = QueueProposalModel(
        [
            {"hypotheses": [valid, malformed]},
            {"hypotheses": [valid]},
        ]
    )
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    residuals = _residuals()

    with pytest.raises(HypothesisProposalCallError) as caught:
        proposer.propose(
            residuals,
            evaluator_epoch="epoch-0",
            max_hypotheses=2,
            trace_id="atomic-root-rejected",
        )

    assert caught.value.failure_phase == "response_program_parse"
    assert len(model.requests) == 1
    assert not any(row["event"] == "hypothesis_proposed" for row in sink.events)
    assert not any(
        row["event"] == "root_proposal_evidence_recorded" for row in sink.events
    )

    accepted = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        max_hypotheses=2,
        trace_id="atomic-root-retry",
    )
    replayed = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        max_hypotheses=2,
        trace_id="atomic-root-replay",
    )

    assert accepted == replayed
    assert len(model.requests) == 2
    assert sum(row["event"] == "hypothesis_proposed" for row in sink.events) == 1
    assert sum(
        row["event"] == "root_proposal_evidence_recorded" for row in sink.events
    ) == 1
    assert sum(
        row["event"] == "root_proposal_evidence_replayed" for row in sink.events
    ) == 1


@pytest.mark.parametrize("returned_count", [0, 1, 3])
def test_exact_count_proposal_batch_rejects_short_or_overlong_atomically(
    returned_count: int,
) -> None:
    sink = MemoryEventSink()
    rows = [
        _program_dict(hypothesis_id=f"batch-count-{index}", priority=10 + index)
        for index in range(returned_count)
    ]
    model = QueueProposalModel([{"hypotheses": rows}])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    capabilities = {
        "proposal_batch_contract": {
            "policy": PROPOSAL_DIVERSITY_POLICY_VERSION,
            "profile_roles": ["precision", "coverage"],
        }
    }

    with pytest.raises(HypothesisProposalCallError) as caught:
        proposer.propose(
            _residuals(),
            evaluator_epoch="epoch-0",
            max_hypotheses=2,
            capabilities=capabilities,
            trace_id=f"batch-count-{returned_count}",
        )

    assert caught.value.failure_phase == "response_exact_count"
    assert len(model.requests) == 1
    request_contract = model.requests[0]["proposal_batch_contract"]
    assert request_contract["policy"] == PROPOSAL_DIVERSITY_POLICY_VERSION
    assert request_contract["response_type"] == "array"
    assert request_contract["required_count"] == 2
    assert request_contract["diversity_unit"] == (
        "train_failure_activation_or_action_treatment"
    )
    assert request_contract["activation_signature_distinctness"] == (
        "search_preference_audit_only"
    )
    assert request_contract["action_treatment_diversity"] == (
        "allowed_when_activation_signatures_coincide"
    )
    assert request_contract["max_action_nodes_per_hypothesis"] == 4
    assert request_contract["profile_roles"] == ["precision", "coverage"]
    rejected = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_proposal_response_rejected"
    )
    assert rejected["payload"]["expected_item_count"] == 2
    assert rejected["payload"]["expected_field_item_count"] == returned_count
    assert rejected["payload"]["raw_content_persisted"] is False
    assert not any(row["event"] == "hypothesis_proposed" for row in sink.events)
    assert not any(
        row["event"] == "root_proposal_evidence_recorded" for row in sink.events
    )


def test_proposal_batch_audits_duplicate_signatures_without_rejection_or_retry() -> None:
    sink = MemoryEventSink()
    rows = [
        _program_dict(hypothesis_id="duplicate-signature-a", priority=10),
        _program_dict(hypothesis_id="duplicate-signature-b", priority=20),
    ]
    model = QueueProposalModel([{"hypotheses": rows}])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)

    programs = proposer.propose(
        _residuals(),
        evaluator_epoch="epoch-0",
        max_hypotheses=2,
        capabilities={
            "proposal_batch_contract": {
                "policy": PROPOSAL_DIVERSITY_POLICY_VERSION,
            }
        },
        trace_id="duplicate-activation-signatures",
    )

    assert len(programs) == 2
    assert len(model.requests) == 1
    audited = next(
        row
        for row in sink.events
        if row["event"] == "proposal_activation_signature_audited"
    )
    assert audited["payload"]["failure_train_row_count"] == 2
    assert audited["payload"]["distinct_activation_signature_count"] == 1
    assert audited["payload"]["activation_signature_group_sizes"] == [2]
    assert len(audited["payload"]["activation_signature_group_hashes"]) == 1
    assert audited["payload"]["pairwise_distinct_required"] is False
    assert audited["payload"]["search_preference_only"] is True
    assert audited["payload"]["response_rejected"] is False
    assert audited["payload"]["proposal_retry_requested"] is False
    assert audited["payload"]["raw_content_persisted"] is False
    assert sum(row["event"] == "hypothesis_proposed" for row in sink.events) == 2
    assert not any(
        row["event"] == "hypothesis_proposal_response_rejected"
        for row in sink.events
    )
    assert any(
        row["event"] == "root_proposal_evidence_recorded" for row in sink.events
    )


def test_exact_distinct_proposal_batch_is_accepted_and_replayed() -> None:
    sink = MemoryEventSink()
    matching = _program_dict(hypothesis_id="distinct-signature-match")
    nonmatching = _program_dict(hypothesis_id="distinct-signature-miss")
    nonmatching["trigger"]["all_of"][0]["value"] = "other_relation"
    model = QueueProposalModel([{"hypotheses": [matching, nonmatching]}])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    residuals = _residuals()
    capabilities = {
        "proposal_batch_contract": {
            "policy": PROPOSAL_DIVERSITY_POLICY_VERSION,
        }
    }

    accepted = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        max_hypotheses=2,
        capabilities=capabilities,
        trace_id="distinct-batch-accepted",
    )
    replayed = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        max_hypotheses=2,
        capabilities=capabilities,
        trace_id="distinct-batch-replayed",
    )

    assert accepted == replayed
    assert len(accepted) == 2
    assert len(model.requests) == 1
    assert sum(row["event"] == "hypothesis_proposed" for row in sink.events) == 2
    assert sum(
        row["event"] == "root_proposal_evidence_recorded" for row in sink.events
    ) == 1
    assert sum(
        row["event"] == "root_proposal_evidence_replayed" for row in sink.events
    ) == 1
    audited = next(
        row
        for row in sink.events
        if row["event"] == "proposal_activation_signature_audited"
    )
    assert audited["payload"]["distinct_activation_signature_count"] == 2
    assert audited["payload"]["activation_signature_group_sizes"] == [1, 1]
    assert len(audited["payload"]["activation_signature_group_hashes"]) == 2


@pytest.mark.parametrize(
    ("case", "expected_phase"),
    [
        ("missing_field", "response_envelope"),
        ("null_field", "response_envelope"),
        ("list_field", "response_envelope"),
        ("root_style", "response_envelope"),
        ("parse_error", "response_program_parse"),
    ],
)
def test_malformed_repair_response_contract_is_typed_and_local(
    case: str,
    expected_phase: str,
) -> None:
    invalid_program = _program_dict(hypothesis_id="invalid-repair-response")
    invalid_program["kind"] = "not-a-hypothesis-kind"
    responses = {
        "missing_field": {},
        "null_field": {"hypothesis": None},
        "list_field": {"hypothesis": []},
        "root_style": {"hypotheses": [_program_dict()]},
        "parse_error": {"hypothesis": invalid_program},
    }
    response = responses[case]
    sink = MemoryEventSink()
    proposer = StructuredHypothesisProposer(
        QueueProposalModel([response]),
        event_sink=sink,
    )
    parent = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="malformed-repair-parent")
    )

    with pytest.raises(HypothesisProposalCallError) as caught:
        proposer.revise(
            parent,
            failed_checks=({"check": "runtime_action", "passed": False},),
            residuals=_residuals(),
            depth=1,
            trace_id=f"malformed-repair-{case}",
        )

    error = caught.value
    assert error.request_kind == "repair_hypothesis_program"
    assert error.failure_phase == expected_phase
    assert error.response_hash == stable_hash(response)
    rejected = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_proposal_response_rejected"
    )
    assert rejected["payload"]["candidate_local_failure"] is True
    assert rejected["payload"]["response_hash"] == error.response_hash
    assert not any(
        row["event"] == "hypothesis_repair_proposed" for row in sink.events
    )


def test_malformed_repair_response_is_isolated_by_recursive_validator() -> None:
    sink = MemoryEventSink()
    root = HypothesisProgram.from_dict(
        _program_dict(
            hypothesis_id="malformed-repair-validator-root",
            lane="missing_lane",
        )
    )
    malformed_response = {"hypotheses": [_program_dict()]}
    proposer = StructuredHypothesisProposer(
        QueueProposalModel([malformed_response]),
        event_sink=sink,
    )

    tree = _validator(proposer, sink).validate(
        root,
        _validation_context(_residuals()),
        trace_id="malformed-repair-validator",
    )

    assert tree.accepted_program is None
    assert tree.repair_model_failure_count == 1
    assert tree.nodes[0].terminal_reason == "repair_model_failed"
    rejected = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_proposal_response_rejected"
    )
    abandoned = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_repair_abandoned_after_model_failure"
    )
    assert abandoned["payload"]["failure_phase"] == "response_envelope"
    assert abandoned["payload"]["response_hash"] == rejected["payload"]["response_hash"]


def test_recursive_validator_does_not_swallow_untyped_harness_errors() -> None:
    class BrokenHarnessProposer:
        def revise(self, *args, **kwargs):
            raise ValueError("harness invariant failed")

    sink = MemoryEventSink()
    root = HypothesisProgram.from_dict(
        _program_dict(
            hypothesis_id="untyped-error-root",
            lane="missing_lane",
        )
    )

    with pytest.raises(ValueError, match="harness invariant failed"):
        _validator(BrokenHarnessProposer(), sink).validate(
            root,
            _validation_context(_residuals()),
            trace_id="untyped-harness-error",
        )


def test_same_model_repair_id_is_parent_scoped_deterministic_and_archive_safe() -> None:
    sink = MemoryEventSink()
    residuals = _residuals()
    shared_repair = _program_dict(
        hypothesis_id="challenger-epoch-completion-gate-v1-repair"
    )
    model = QueueProposalModel(
        [
            {"hypothesis": shared_repair},
            {"hypothesis": shared_repair},
        ]
    )
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    runtime = _runtime(sink)
    runner = CounterfactualRunner(
        runtime=runtime,
        evaluator=TruthEvaluator(),
        event_sink=sink,
    )
    archive = PolicyArchive(event_sink=sink)
    validation_tasks = tuple(_task(f"validation-{index}") for index in range(2))
    manifest = SplitManifest(
        benchmark="synthetic_repair_id_test",
        protocol="instance_holdout",
        seed="unit",
        train_ids=("train-0", "train-1"),
        validation_ids=tuple(task.id for task in validation_tasks),
        test_ids=("sealed-test-0",),
        family_by_id={
            "train-0": "relation",
            "train-1": "relation",
            **{task.id: "relation" for task in validation_tasks},
            "sealed-test-0": "relation",
        },
    )
    root_a = HypothesisProgram.from_dict(
        _program_dict(
            hypothesis_id="repair-root-a",
            lane="missing_lane",
            priority=10,
        )
    )
    root_b = HypothesisProgram.from_dict(
        _program_dict(
            hypothesis_id="repair-root-b",
            lane="missing_lane",
            priority=11,
        )
    )
    kernel = EvolutionKernel(
        proposer=proposer,
        validator=_validator(proposer, sink),
        counterfactual_runner=runner,
        promotion_gate=PromotionGate(_promotion_spec(), event_sink=sink),
        archive=archive,
        split_guard=SplitAccessGuard(manifest, event_sink=sink),
        event_sink=sink,
    )

    result = kernel.evolve_once(
        residuals=residuals,
        validation_tasks=validation_tasks,
        validation_context=_validation_context(residuals),
        proposal_candidates=(root_a, root_b),
        trace_id="same-model-repair-id",
    )

    repair_events = [
        row for row in sink.events if row["event"] == "hypothesis_repair_proposed"
    ]
    child_ids = {row["payload"]["child_id"] for row in repair_events}
    assert len(repair_events) == 2
    assert len(child_ids) == 2
    assert child_ids <= set(archive.hypotheses)
    assert {
        archive.hypotheses[child_id].parent_id for child_id in child_ids
    } == {root_a.id, root_b.id}
    assert all(
        row["payload"]["branch_id_policy"] == REPAIR_BRANCH_ID_POLICY_VERSION
        and row["payload"]["child_id"]
        == f"repair_{row['payload']['branch_identity_hash']}"
        and row["payload"]["model_supplied_child_id_used"] is False
        for row in repair_events
    )
    assert result.static_accepted_candidate_count == 2

    replay_proposer = StructuredHypothesisProposer(
        QueueProposalModel([{"hypothesis": shared_repair}])
    )
    replay_tree = _validator(replay_proposer, MemoryEventSink()).validate(
        root_a,
        _validation_context(residuals),
        trace_id="same-parent-repair-replay",
    )
    assert replay_tree.accepted_program is not None
    assert replay_tree.accepted_program.id == next(
        child_id
        for child_id in child_ids
        if archive.hypotheses[child_id].parent_id == root_a.id
    )


def test_same_model_id_across_recursive_depths_preserves_branch_lineage() -> None:
    sink = MemoryEventSink()
    residuals = _residuals()
    shared_model_id = "challenger-epoch-completion-gate-v1-repair"
    depth_one = _program_dict(
        hypothesis_id=shared_model_id,
        lane="missing_lane",
        priority=11,
    )
    depth_two = _program_dict(
        hypothesis_id=shared_model_id,
        lane="relation_solver",
        priority=12,
    )
    proposer = StructuredHypothesisProposer(
        QueueProposalModel(
            [
                {"hypothesis": depth_one},
                {"hypothesis": depth_two},
            ]
        ),
        event_sink=sink,
    )
    root = HypothesisProgram.from_dict(
        _program_dict(
            hypothesis_id="recursive-repair-root",
            lane="missing_lane",
            priority=10,
        )
    )

    tree = _validator(proposer, sink).validate(
        root,
        _validation_context(residuals),
        trace_id="same-model-id-recursive-depths",
    )

    assert len(tree.nodes) == 3
    first_child = tree.nodes[1].program
    second_child = tree.nodes[2].program
    assert first_child.id != second_child.id
    assert tree.nodes[0].child_id == first_child.id
    assert tree.nodes[1].child_id == second_child.id
    assert tree.nodes[2].child_id is None
    assert first_child.parent_id == root.id
    assert first_child.lineage == (root.id,)
    assert second_child.parent_id == first_child.id
    assert second_child.lineage == (root.id, first_child.id)
    assert tree.accepted_program == second_child

    archive = PolicyArchive()
    for node in tree.nodes:
        archive.register_hypothesis(node.program)
    assert set(archive.hypotheses) == {root.id, first_child.id, second_child.id}
    assert not any(
        row["event"] == "hypothesis_proposal_response_rejected"
        for row in sink.events
    )


def test_repair_branch_id_uses_canonical_defaults_and_ignores_unknown_keys() -> None:
    residuals = _residuals()
    parent = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="canonical-repair-parent")
    )
    explicit = _program_dict(hypothesis_id="model-repair-id")
    omitted = dict(explicit)
    omitted_verifier = dict(explicit["verifier"])
    omitted_verifier.pop("repair_on_failure")
    omitted_verifier.pop("max_repair_depth")
    omitted_verifier["unknown_verifier_key"] = "ignored"
    omitted["verifier"] = omitted_verifier
    omitted.pop("fallback")
    omitted.pop("status")
    omitted["unknown_top_level_key"] = {"ignored": True}

    explicit_child = _revise_once(parent, explicit, residuals=residuals)
    omitted_child = _revise_once(parent, omitted, residuals=residuals)

    assert explicit_child.id == omitted_child.id
    assert explicit_child.to_dict() == omitted_child.to_dict()


def test_repair_branch_id_ignores_model_supplied_identifier() -> None:
    residuals = _residuals()
    parent = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="model-id-independent-parent")
    )
    first = _program_dict(hypothesis_id="model-label-a")
    second = _program_dict(hypothesis_id="model-label-b")

    first_child = _revise_once(parent, first, residuals=residuals)
    second_child = _revise_once(parent, second, residuals=residuals)

    assert first_child.id == second_child.id
    assert first_child.to_dict() == second_child.to_dict()


def test_repair_child_status_is_harness_owned_candidate() -> None:
    residuals = _residuals()
    parent = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="status-owned-parent")
    )
    rejected = _program_dict(
        hypothesis_id="status-owned-model-label",
        status="rejected",
    )
    promoted = _program_dict(
        hypothesis_id="status-owned-model-label",
        status="promoted",
    )
    omitted = _program_dict(hypothesis_id="status-owned-model-label")
    omitted.pop("status")

    children = (
        _revise_once(parent, rejected, residuals=residuals),
        _revise_once(parent, promoted, residuals=residuals),
        _revise_once(parent, omitted, residuals=residuals),
    )

    assert {child.id for child in children} == {children[0].id}
    assert {child.status for child in children} == {HypothesisStatus.CANDIDATE}
    assert all(child.to_dict() == children[0].to_dict() for child in children[1:])


def test_repair_branch_id_is_stable_across_parent_status_changes() -> None:
    residuals = _residuals()
    candidate_parent = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="status-stable-parent")
    )
    rejected_parent = replace(candidate_parent, status=HypothesisStatus.REJECTED)
    repair = _program_dict(hypothesis_id="status-stable-model-label")

    candidate_child = _revise_once(
        candidate_parent,
        repair,
        residuals=residuals,
    )
    rejected_child = _revise_once(
        rejected_parent,
        repair,
        residuals=residuals,
    )

    assert candidate_child.id == rejected_child.id
    assert candidate_child.to_dict() == rejected_child.to_dict()


def test_archive_still_rejects_same_hypothesis_id_with_different_content() -> None:
    archive = PolicyArchive()
    original = HypothesisProgram.from_dict(_program_dict(hypothesis_id="fixed-id"))
    collision = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="fixed-id", priority=11)
    )
    archive.register_hypothesis(original)

    with pytest.raises(ValueError, match="hypothesis ID collision: fixed-id"):
        archive.register_hypothesis(collision)


def test_root_proposal_replay_requires_the_exact_request_state() -> None:
    sink = MemoryEventSink()
    first_payload = _program_dict(hypothesis_id="root-first")
    changed_payload = _program_dict(hypothesis_id="root-state-changed")
    model = QueueProposalModel(
        [
            {"hypotheses": [first_payload]},
            {"hypotheses": [changed_payload]},
        ]
    )
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    residuals = _residuals()

    first = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        capabilities={"prior_state": "same"},
        trace_id="root-replay-source",
    )
    replayed = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        capabilities={"prior_state": "same"},
        trace_id="root-replay-target",
    )
    changed = proposer.propose(
        residuals,
        evaluator_epoch="epoch-0",
        capabilities={"prior_state": "changed"},
        trace_id="root-replay-state-miss",
    )

    assert replayed == first
    assert changed[0].id == "root-state-changed"
    assert len(model.requests) == 2
    replay = next(
        row for row in sink.events if row["event"] == "root_proposal_evidence_replayed"
    )
    assert replay["payload"]["source_trace_id"] == "root-replay-source"
    assert replay["payload"]["target_trace_id"] == "root-replay-target"
    assert replay["payload"]["request_identical"] is True
    assert replay["payload"]["new_proposal_model_executions"] == 0
    assert sum(
        row["event"] == "root_proposal_evidence_recorded" for row in sink.events
    ) == 2


def test_paired_counterfactual_gain_promotes_program() -> None:
    sink = MemoryEventSink()
    runtime = _runtime(sink)
    program = HypothesisProgram.from_dict(_program_dict())
    runner = CounterfactualRunner(runtime=runtime, evaluator=TruthEvaluator(), event_sink=sink)
    tasks = tuple(_task(f"validation-{index}") for index in range(20))
    pairs = runner.run(tasks, program=program, split=SplitName.VALIDATION)
    gate = PromotionGate(
        _promotion_spec(),
        event_sink=sink,
    )

    decision = gate.evaluate(program, pairs, sealed_test_accessed=False)

    assert decision.allowed is True
    assert decision.summary.baseline_success_count == 0
    assert decision.summary.candidate_success_count == 20
    assert decision.summary.activation_count == 20
    assert decision.effect_lower_bound == 1.0


def test_candidate_cannot_relax_protocol_effect_lower_bound() -> None:
    sink = MemoryEventSink()
    payload = _program_dict()
    payload["expected_effect"] = {
        "metric": "task_success",
        "minimum_delta": -1.0,
        "maximum_harm_rate": 1.0,
        "maximum_cost_ratio": 99.0,
    }
    program = HypothesisProgram.from_dict(payload)
    runner = CounterfactualRunner(
        runtime=_runtime(sink), evaluator=TruthEvaluator(), event_sink=sink
    )
    pairs = list(
        runner.run(
            tuple(_task(f"validation-{index}") for index in range(20)),
            program=program,
            split=SplitName.VALIDATION,
        )
    )
    for index in range(1, len(pairs)):
        pair = pairs[index]
        pairs[index] = replace(
            pair,
            candidate_outcome=replace(
                pair.candidate_outcome,
                success=False,
                score=0.0,
            ),
        )

    decision = PromotionGate(
        _promotion_spec(minimum_net_gain_count=1)
    ).evaluate(program, pairs, sealed_test_accessed=False)

    assert "paired_effect_lower_bound_below_target" in decision.blockers
    assert decision.candidate_thresholds["minimum_effect_lower_bound"] == -1.0
    assert decision.effective_thresholds["minimum_effect_lower_bound"] == 0.0


def test_candidate_cannot_relax_protocol_harm_or_cost_limits() -> None:
    sink = MemoryEventSink()
    payload = _program_dict()
    payload["expected_effect"] = {
        "metric": "task_success",
        "minimum_delta": -1.0,
        "maximum_harm_rate": 1.0,
        "maximum_cost_ratio": 99.0,
    }
    program = HypothesisProgram.from_dict(payload)
    runner = CounterfactualRunner(
        runtime=_runtime(sink), evaluator=TruthEvaluator(), event_sink=sink
    )
    pairs = list(
        runner.run(
            tuple(_task(f"validation-{index}") for index in range(20)),
            program=program,
            split=SplitName.VALIDATION,
        )
    )
    for index in range(2):
        pair = pairs[index]
        pairs[index] = replace(
            pair,
            baseline_outcome=replace(
                pair.baseline_outcome,
                success=True,
                score=1.0,
            ),
            candidate_outcome=replace(
                pair.candidate_outcome,
                success=False,
                score=0.0,
            ),
        )

    harm_decision = PromotionGate(_promotion_spec()).evaluate(
        program, pairs, sealed_test_accessed=False
    )
    cost_decision = PromotionGate(
        replace(_promotion_spec(), maximum_harm_rate=1.0, maximum_cost_ratio=1.5)
    ).evaluate(program, pairs[2:], sealed_test_accessed=False)

    assert "harm_rate_exceeded" in harm_decision.blockers
    assert harm_decision.effective_thresholds["maximum_harm_rate"] == 0.05
    assert "cost_ratio_exceeded" in cost_decision.blockers
    assert cost_decision.effective_thresholds["maximum_cost_ratio"] == 1.5


def test_candidate_stricter_thresholds_remain_effective() -> None:
    program = HypothesisProgram.from_dict(_program_dict())
    protocol_spec = replace(
        _promotion_spec(),
        minimum_effect_lower_bound=0.0,
        maximum_harm_rate=0.2,
        maximum_cost_ratio=5.0,
    )

    assert protocol_spec.effective_thresholds(program) == {
        "minimum_effect_lower_bound": 0.1,
        "maximum_harm_rate": 0.05,
        "maximum_cost_ratio": 3.0,
    }


def test_counterfactual_replay_requires_identical_executable_behavior() -> None:
    sink = MemoryEventSink()
    runner = CounterfactualRunner(
        runtime=_runtime(sink),
        evaluator=TruthEvaluator(),
        event_sink=sink,
    )
    cache = CounterfactualEvidenceReplayCache(event_sink=sink)
    tasks = (_task("validation-1"), _task("validation-2"))
    program = HypothesisProgram.from_dict(_program_dict())
    same_behavior = HypothesisProgram.from_dict(
        _program_dict(hypothesis_id="same-behavior-new-id")
    )
    different_payload = _program_dict(hypothesis_id="different-behavior")
    different_payload["trigger"]["all_of"][0]["value"] = "other"
    different_behavior = HypothesisProgram.from_dict(different_payload)

    recorded = cache.run_or_replay(
        runner=runner,
        tasks=tasks,
        program=program,
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="replay-source",
    )
    replayed = cache.run_or_replay(
        runner=runner,
        tasks=tasks,
        program=same_behavior,
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="replay-target",
    )
    cache.run_or_replay(
        runner=runner,
        tasks=tasks,
        program=different_behavior,
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="replay-miss",
    )

    assert replayed == recorded
    assert sum(
        row["event"] == "counterfactual_pair_completed" for row in sink.events
    ) == 4
    assert sum(
        row["event"] == "counterfactual_evidence_recorded" for row in sink.events
    ) == 2
    replay_event = next(
        row
        for row in sink.events
        if row["event"] == "counterfactual_evidence_replayed"
    )
    assert replay_event["payload"]["behavior_identical"] is True
    assert replay_event["payload"]["new_counterfactual_executions"] == 0
    assert replay_event["payload"]["sealed_test_accessed"] is False
    with pytest.raises(PermissionError, match="unsealed validation"):
        cache.run_or_replay(
            runner=runner,
            tasks=tasks,
            program=program,
            baseline_programs=(),
            split=SplitName.TEST,
            trace_id="sealed-replay-forbidden",
        )


def test_complementary_precise_candidates_are_selected_as_one_train_bundle() -> None:
    result, archive, sink = _run_bundle_evolution(low_precision_complement=False)

    assert result.selected_candidate_hypothesis_ids == (
        "bundle-a",
        "bundle-b",
    )
    assert result.accepted_hypothesis_id == "bundle-a"
    assert result.archive_node is not None
    assert result.archive_node.active_hypothesis_ids == ("bundle-a", "bundle-b")
    assert archive.hypotheses["bundle-c"].status is HypothesisStatus.SHADOW
    selection = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    )
    selected_subset = next(
        row for row in selection["payload"]["candidate_subsets"] if row["selected"]
    )
    assert selected_subset["accepted_hypothesis_ids"] == ["bundle-a", "bundle-b"]
    assert selected_subset["union_training_metrics"][
        "activation_precision_numerator"
    ] == 4
    assert selected_subset["union_training_metrics"][
        "activation_precision_denominator"
    ] == 4
    assert selection["payload"]["selection_uses_validation"] is False


def test_low_precision_complement_does_not_override_precision_first_singleton() -> None:
    result, archive, sink = _run_bundle_evolution(low_precision_complement=True)

    assert result.selected_candidate_hypothesis_ids == ("bundle-a",)
    assert result.accepted_hypothesis_id == "bundle-a"
    assert archive.hypotheses["bundle-b"].status is HypothesisStatus.SHADOW
    selection = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    )
    selected_subset = next(
        row for row in selection["payload"]["candidate_subsets"] if row["selected"]
    )
    assert selected_subset["accepted_hypothesis_ids"] == ["bundle-a"]
    assert selected_subset["union_training_metrics"][
        "failure_activation_family_deficit"
    ] == 1


def test_v314_no_recursive_g2_fixture_prefers_three_family_support() -> None:
    programs, residuals, validation_tasks = _no_recursive_g2_bundle_fixture()
    legacy, _, legacy_sink = _run_bundle_evolution(
        low_precision_complement=False,
        proposal_candidates=programs,
        residuals=residuals,
        validation_tasks=validation_tasks,
        minimum_activation_rate=0.5,
    )
    revised, _, revised_sink = _run_bundle_evolution(
        low_precision_complement=False,
        selection_policy=(
            COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
        ),
        proposal_candidates=programs,
        residuals=residuals,
        validation_tasks=validation_tasks,
        minimum_activation_rate=0.5,
    )

    assert legacy.selected_candidate_hypothesis_ids == ("g2-court", "g2-poster")
    assert revised.selected_candidate_hypothesis_ids == (
        "g2-court",
        "g2-poster",
        "g2-schedule",
    )
    assert revised.repaired_candidate_count == 0

    legacy_selection = next(
        row
        for row in legacy_sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    )
    legacy_subset = next(
        row
        for row in legacy_selection["payload"]["candidate_subsets"]
        if row["selected"]
    )
    assert legacy_subset["union_training_metrics"]["failure_support"] == 5
    assert legacy_subset["union_training_metrics"][
        "failure_activation_family_count"
    ] == 2

    revised_selection = next(
        row
        for row in revised_sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    )
    revised_subset = next(
        row
        for row in revised_selection["payload"]["candidate_subsets"]
        if row["selected"]
    )
    assert revised_subset["union_training_metrics"][
        "activation_precision_numerator"
    ] == 7
    assert revised_subset["union_training_metrics"][
        "activation_precision_denominator"
    ] == 7
    assert revised_subset["union_training_metrics"]["failure_support"] == 7
    assert revised_subset["union_training_metrics"][
        "failure_activation_family_count"
    ] == 3
    assert revised_subset["ranking_priority"] == [
        "precision_desc",
        "family_target_deficit_asc",
        "success_false_positives_asc",
        "overlap_asc",
        "actual_family_count_desc",
        "failure_support_desc",
        "bundle_size_asc",
        "complexity_asc",
        "canonical_set_hash_asc",
    ]


def test_v314_no_recursive_g2_bundle_selection_is_order_invariant() -> None:
    programs, residuals, validation_tasks = _no_recursive_g2_bundle_fixture()
    results: list[tuple[tuple[str, ...], str]] = []
    for proposal_candidates in (programs, tuple(reversed(programs))):
        result, _, sink = _run_bundle_evolution(
            low_precision_complement=False,
            selection_policy=(
                COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
            ),
            proposal_candidates=proposal_candidates,
            residuals=residuals,
            validation_tasks=validation_tasks,
            minimum_activation_rate=0.5,
        )
        selection = next(
            row
            for row in sink.events
            if row["event"]
            == "hypothesis_training_candidate_selection_completed"
        )
        results.append(
            (
                result.selected_candidate_hypothesis_ids,
                selection["payload"]["selected_candidate_set_hash"],
            )
        )

    assert results[0] == results[1]


def test_v314_no_recursive_g2_lower_precision_false_positive_cannot_win() -> None:
    programs, residuals, validation_tasks = _no_recursive_g2_bundle_fixture(
        low_precision_schedule=True
    )
    result, _, sink = _run_bundle_evolution(
        low_precision_complement=False,
        selection_policy=(
            COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
        ),
        proposal_candidates=programs,
        residuals=residuals,
        validation_tasks=validation_tasks,
        minimum_activation_rate=0.5,
    )

    assert result.selected_candidate_hypothesis_ids == ("g2-court", "g2-poster")
    selection = next(
        row
        for row in sink.events
        if row["event"] == "hypothesis_training_candidate_selection_completed"
    )
    selected_subset = next(
        row for row in selection["payload"]["candidate_subsets"] if row["selected"]
    )
    broad_subset = next(
        row
        for row in selection["payload"]["candidate_subsets"]
        if len(row["accepted_hypothesis_ids"]) == 3
    )
    assert selected_subset["union_training_metrics"][
        "success_false_positive_activation_count"
    ] == 0
    assert broad_subset["union_training_metrics"][
        "activation_precision_numerator"
    ] == 7
    assert broad_subset["union_training_metrics"][
        "activation_precision_denominator"
    ] == 8
    assert broad_subset["union_training_metrics"][
        "success_false_positive_activation_count"
    ] == 1
    assert broad_subset["union_training_metrics"][
        "failure_activation_family_count"
    ] == 3


def test_bundle_runtime_and_replay_identity_are_order_invariant_and_set_specific() -> None:
    sink = MemoryEventSink()
    runner = CounterfactualRunner(
        runtime=_runtime(sink),
        evaluator=TruthEvaluator(),
        event_sink=sink,
    )
    programs = _bundle_programs(low_precision_complement=False)
    program_a, program_b, program_c = programs
    tasks = (_bundle_task("validation-a", "a"), _bundle_task("validation-b", "b"))

    forward = runner.run_bundle(
        tasks,
        programs=(program_a, program_b),
        split=SplitName.VALIDATION,
        trace_id="bundle-order",
    )
    reverse = runner.run_bundle(
        tasks,
        programs=(program_b, program_a),
        split=SplitName.VALIDATION,
        trace_id="bundle-order",
    )
    assert runner.behavior_set_hash((program_a, program_b)) == (
        runner.behavior_set_hash((program_b, program_a))
    )
    assert [row.candidate.plan_hash for row in forward] == [
        row.candidate.plan_hash for row in reverse
    ]

    replay_sink = MemoryEventSink()
    replay_runner = CounterfactualRunner(
        runtime=_runtime(replay_sink),
        evaluator=TruthEvaluator(),
        event_sink=replay_sink,
    )
    cache = CounterfactualEvidenceReplayCache(event_sink=replay_sink)
    cache.run_or_replay_bundle(
        runner=replay_runner,
        tasks=tasks,
        programs=(program_a, program_b),
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="bundle-replay-source",
    )
    cache.run_or_replay_bundle(
        runner=replay_runner,
        tasks=tasks,
        programs=(program_b, program_a),
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="bundle-replay-hit",
    )
    cache.run_or_replay_bundle(
        runner=replay_runner,
        tasks=tasks,
        programs=(program_a, program_c),
        baseline_programs=(),
        split=SplitName.VALIDATION,
        trace_id="bundle-replay-miss",
    )
    replay_events = [
        row
        for row in replay_sink.events
        if row["event"] == "counterfactual_evidence_replayed"
    ]
    recorded_events = [
        row
        for row in replay_sink.events
        if row["event"] == "counterfactual_evidence_recorded"
    ]
    assert len(replay_events) == 1
    assert len(recorded_events) == 2
    assert replay_events[0]["payload"]["candidate_hypothesis_ids"] == [
        "bundle-a",
        "bundle-b",
    ]
    assert recorded_events[0]["payload"]["replay_key"] != (
        recorded_events[1]["payload"]["replay_key"]
    )


def test_bundle_gate_aggregates_thresholds_and_archive_handles_both_outcomes() -> None:
    sink = MemoryEventSink()
    runner = CounterfactualRunner(
        runtime=_runtime(sink),
        evaluator=TruthEvaluator(),
        event_sink=sink,
    )
    program_a, program_b, _ = _bundle_programs(
        low_precision_complement=False
    )
    program_a = replace(
        program_a,
        expected_effect=replace(
            program_a.expected_effect,
            minimum_delta=0.2,
            maximum_harm_rate=0.04,
            maximum_cost_ratio=2.5,
        ),
    )
    program_b = replace(
        program_b,
        expected_effect=replace(
            program_b.expected_effect,
            minimum_delta=0.3,
            maximum_harm_rate=0.02,
            maximum_cost_ratio=2.0,
        ),
    )
    tasks = (_bundle_task("validation-a", "a"), _bundle_task("validation-b", "b"))
    pairs = runner.run_bundle(
        tasks,
        programs=(program_b, program_a),
        split=SplitName.VALIDATION,
    )
    permissive_spec = replace(
        _promotion_spec(minimum_pairs=2, minimum_net_gain_count=1),
        minimum_activation_rate=1.0,
        maximum_harm_rate=0.2,
        maximum_cost_ratio=5.0,
    )
    allowed = PromotionGate(permissive_spec).evaluate_bundle(
        (program_a, program_b),
        pairs,
        sealed_test_accessed=False,
    )
    assert allowed.allowed is True
    assert allowed.candidate_thresholds == {
        "minimum_effect_lower_bound": 0.3,
        "maximum_harm_rate": 0.02,
        "maximum_cost_ratio": 2.0,
    }
    assert allowed.policy == "evaluator_owned_paired_validation_v2"

    promoted_archive = PolicyArchive()
    for program in (program_a, program_b):
        promoted_archive.register_hypothesis(program)
    promoted_node = promoted_archive.create_node(
        active_hypothesis_ids=(program_b.id, program_a.id, program_b.id),
        evaluator_epoch_id="epoch-0",
        runtime_version="runtime-v1",
    )
    assert promoted_node.active_hypothesis_ids == ("bundle-a", "bundle-b")
    promoted_archive.apply_promotion(
        candidate_node_id=promoted_node.id,
        decision=allowed,
    )
    assert {
        promoted_archive.hypotheses[row.id].status
        for row in (program_a, program_b)
    } == {HypothesisStatus.PROMOTED}

    rejected = PromotionGate(
        replace(permissive_spec, minimum_pairs=3)
    ).evaluate_bundle(
        (program_b, program_a),
        pairs,
        sealed_test_accessed=False,
    )
    assert rejected.allowed is False
    rejected_archive = PolicyArchive()
    for program in (program_a, program_b):
        rejected_archive.register_hypothesis(program)
    rejected_node = rejected_archive.create_node(
        active_hypothesis_ids=(program_a.id, program_b.id),
        evaluator_epoch_id="epoch-0",
        runtime_version="runtime-v1",
    )
    rejected_archive.apply_promotion(
        candidate_node_id=rejected_node.id,
        decision=rejected,
        retain_rejected_hypotheses_as_shadow=True,
    )
    assert {
        rejected_archive.hypotheses[row.id].status
        for row in (program_a, program_b)
    } == {HypothesisStatus.SHADOW}


def test_evaluator_replacement_invalidates_only_dependent_epoch_scores() -> None:
    sink = MemoryEventSink()
    archive = PolicyArchive(event_sink=sink)
    program = HypothesisProgram.from_dict(_program_dict())
    archive.register_hypothesis(program)
    node = archive.create_node(
        active_hypothesis_ids=(program.id,),
        evaluator_epoch_id="epoch-0",
        runtime_version="runtime-v1",
    )
    old_score = archive.record_score(
        archive_node_id=node.id,
        split="validation",
        evaluator_epoch_id="epoch-0",
        metric="task_success",
        successes=6,
        total=10,
        item_ids=tuple(f"v-{index}" for index in range(10)),
    )
    independent_score = archive.record_score(
        archive_node_id=node.id,
        split="validation",
        evaluator_epoch_id="fixed-objective",
        metric="external_success",
        successes=7,
        total=10,
        item_ids=tuple(f"v-{index}" for index in range(10)),
    )
    incumbent = EvaluatorSpec("judge-a", "1", "impl-a", "criteria-a", "anchor-1")
    challenger = EvaluatorSpec("judge-b", "1", "impl-b", "criteria-b", "anchor-1")
    controller = EvaluatorEpochController(
        EvaluatorEpoch("epoch-0", 0, incumbent),
        confidence=0.9,
        event_sink=sink,
    )

    transition = controller.consider_challenger(
        challenger,
        incumbent_score=AnchorScore("judge-a", "anchor-1", 60, 100),
        challenger_score=AnchorScore("judge-b", "anchor-1", 90, 100),
        archive=archive,
    )

    assert transition.promoted is True
    assert old_score.id in transition.invalidated_score_record_ids
    assert archive.score_records[old_score.id].valid is False
    assert archive.score_records[independent_score.id].valid is True
    assert controller.current.id != "epoch-0"


def test_evolution_kernel_closes_proposal_validation_promotion_runtime_loop() -> None:
    sink = MemoryEventSink()
    proposal = _program_dict()
    proposal["expected_effect"]["metric"] = "candidate_self_reported_metric"
    model = QueueProposalModel([{"hypotheses": [proposal]}])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    validator = _validator(proposer, sink)
    runtime = _runtime(sink)
    runner = CounterfactualRunner(runtime=runtime, evaluator=TruthEvaluator(), event_sink=sink)
    gate = PromotionGate(
        _promotion_spec(),
        event_sink=sink,
    )
    archive = PolicyArchive(event_sink=sink)
    validation_tasks = tuple(_task(f"validation-{index}") for index in range(20))
    manifest = SplitManifest(
        benchmark="synthetic_contract_test",
        protocol="instance_holdout",
        seed="unit",
        train_ids=("train-0", "train-1"),
        validation_ids=tuple(task.id for task in validation_tasks),
        test_ids=("sealed-test-0",),
        family_by_id={
            "train-0": "relation",
            "train-1": "relation",
            **{task.id: "relation" for task in validation_tasks},
            "sealed-test-0": "relation",
        },
    )
    guard = SplitAccessGuard(manifest, event_sink=sink)
    kernel = EvolutionKernel(
        proposer=proposer,
        validator=validator,
        counterfactual_runner=runner,
        promotion_gate=gate,
        archive=archive,
        split_guard=guard,
        event_sink=sink,
    )
    residuals = _residuals()

    result = kernel.evolve_once(
        residuals=residuals,
        validation_tasks=validation_tasks,
        validation_context=_validation_context(residuals),
    )

    assert result.promoted is True
    assert result.accepted_hypothesis_id is not None
    promoted = archive.hypotheses[result.accepted_hypothesis_id]
    assert promoted.status is HypothesisStatus.PROMOTED
    assert promoted.expected_effect.metric == "task_success"
    assert result.promotion_decision.candidate_metric == "task_success"
    assert model.requests[0]["capabilities"]["primary_metric"] == "task_success"
    assert archive.incumbent_id == result.archive_node.id
    assert {row.metric for row in archive.score_records.values()} == {
        "task_success"
    }
    future = runtime.execute(_task("future-unseen"), (promoted,))
    assert future.selected_result.lane == "relation_solver"
    assert any(row["event"] == "evolution_generation_completed" for row in sink.events)

    second_model = QueueProposalModel(
        [{
            "hypotheses": [
                _program_dict(
                    hypothesis_id="hyp-no-increment-challenger",
                    priority=9,
                )
            ]
        }]
    )
    second_proposer = StructuredHypothesisProposer(second_model, event_sink=sink)
    second_kernel = EvolutionKernel(
        proposer=second_proposer,
        validator=_validator(second_proposer, sink),
        counterfactual_runner=runner,
        promotion_gate=gate,
        archive=archive,
        split_guard=guard,
        event_sink=sink,
    )

    second = second_kernel.evolve_once(
        residuals=residuals,
        validation_tasks=validation_tasks,
        validation_context=_validation_context(residuals),
        trace_id="second-generation",
    )

    assert second.promoted is False
    assert second.promotion_decision.summary.baseline_success_count == 20
    assert second.promotion_decision.summary.candidate_success_count == 20
    assert archive.hypotheses[second.accepted_hypothesis_id].status is HypothesisStatus.REJECTED
    assert archive.incumbent_id == result.archive_node.id


def _bundle_programs(
    *,
    low_precision_complement: bool,
) -> tuple[HypothesisProgram, HypothesisProgram, HypothesisProgram]:
    def program(
        hypothesis_id: str,
        family: str,
        *,
        require_failure_marker: bool,
        require_common_marker: bool = False,
    ) -> HypothesisProgram:
        payload = _program_dict(hypothesis_id=hypothesis_id)
        predicates: list[dict[str, Any]] = [
            {"key": "bundle_family", "op": "eq", "value": family}
        ]
        if require_failure_marker:
            predicates.append(
                {"key": "baseline_failed", "op": "eq", "value": True}
            )
        if require_common_marker:
            predicates.append(
                {"key": "common_marker", "op": "eq", "value": True}
            )
        payload["trigger"] = {"all_of": predicates}
        payload["statement"] = f"Bundle candidate for family {family}."
        return HypothesisProgram.from_dict(payload)

    return (
        program(
            "bundle-a",
            "a",
            require_failure_marker=True,
        ),
        program(
            "bundle-b",
            "b",
            require_failure_marker=not low_precision_complement,
        ),
        program(
            "bundle-c",
            "a",
            require_failure_marker=True,
            require_common_marker=True,
        ),
    )


def _bundle_residuals() -> tuple[ResidualExample, ...]:
    rows: list[ResidualExample] = []
    for family in ("a", "b"):
        for index in range(2):
            rows.append(
                ResidualExample(
                    transition_id=f"bundle-transition-{family}-{index}",
                    task_id=f"bundle-train-{family}-{index}",
                    family=f"family-{family}",
                    split=SplitName.TRAIN,
                    features={
                        "bundle_family": family,
                        "baseline_failed": True,
                        "common_marker": True,
                        "requires_live_source": False,
                    },
                    failure_type="bundle_baseline_failure",
                    evaluator_feedback=("use the family-specific solver",),
                    baseline_success=False,
                )
            )
        rows.append(
            ResidualExample(
                transition_id=f"bundle-transition-{family}-success",
                task_id=f"bundle-train-{family}-success",
                family=f"family-{family}",
                split=SplitName.TRAIN,
                features={
                    "bundle_family": family,
                    "baseline_failed": False,
                    "common_marker": True,
                    "requires_live_source": False,
                },
                failure_type="success_control",
                evaluator_feedback=(),
                baseline_success=True,
            )
        )
    return tuple(rows)


def _bundle_task(task_id: str, family: str) -> TaskInput:
    return TaskInput(
        id=task_id,
        family=f"family-{family}",
        features={
            "bundle_family": family,
            "baseline_failed": True,
            "common_marker": True,
            "requires_live_source": False,
        },
        payload={"expected": "correct"},
    )


def _no_recursive_g2_bundle_fixture(
    *,
    low_precision_schedule: bool = False,
) -> tuple[
    tuple[HypothesisProgram, ...],
    tuple[ResidualExample, ...],
    tuple[TaskInput, ...],
]:
    fixture_families = (
        ("anthropic-poster-design", "poster", 2),
        ("court-form-filling", "court", 3),
        ("schedule-coordination", "schedule", 2),
    )

    def program(family: str, selector: str) -> HypothesisProgram:
        payload = _program_dict(hypothesis_id=f"g2-{selector}")
        predicates: list[dict[str, Any]] = [
            {"key": "g2_selector", "op": "eq", "value": selector}
        ]
        if not (low_precision_schedule and selector == "schedule"):
            predicates.append(
                {"key": "baseline_failed", "op": "eq", "value": True}
            )
        if selector == "schedule":
            predicates.append(
                {"key": "common_marker", "op": "eq", "value": True}
            )
        payload["trigger"] = {"all_of": predicates}
        payload["statement"] = f"No-recursive G2 {family} specialist."
        return HypothesisProgram.from_dict(payload)

    programs = tuple(
        program(family, selector)
        for family, selector, _ in fixture_families
    )
    residuals: list[ResidualExample] = []
    tasks: list[TaskInput] = []
    for family, selector, failure_support in fixture_families:
        for index in range(failure_support):
            residuals.append(
                ResidualExample(
                    transition_id=f"g2-transition-{selector}-{index}",
                    task_id=f"g2-train-{selector}-{index}",
                    family=family,
                    split=SplitName.TRAIN,
                    features={
                        "g2_selector": selector,
                        "baseline_failed": True,
                        "common_marker": True,
                        "requires_live_source": False,
                    },
                    failure_type="no_recursive_g2_failure",
                    evaluator_feedback=("use the family specialist",),
                    baseline_success=False,
                )
            )
        residuals.append(
            ResidualExample(
                transition_id=f"g2-transition-{selector}-success",
                task_id=f"g2-train-{selector}-success",
                family=family,
                split=SplitName.TRAIN,
                features={
                    "g2_selector": selector,
                    "baseline_failed": False,
                    "common_marker": True,
                    "requires_live_source": False,
                },
                failure_type="success_control",
                evaluator_feedback=(),
                baseline_success=True,
            )
        )
        tasks.append(
            TaskInput(
                id=f"g2-validation-{selector}",
                family=family,
                features={
                    "g2_selector": selector,
                    "baseline_failed": True,
                    "common_marker": True,
                    "requires_live_source": False,
                },
                payload={"expected": "correct"},
            )
        )
    return programs, tuple(residuals), tuple(tasks)


def _run_bundle_evolution(
    *,
    low_precision_complement: bool,
    selection_policy: str = (
        COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION
    ),
    proposal_candidates: tuple[HypothesisProgram, ...] | None = None,
    residuals: tuple[ResidualExample, ...] | None = None,
    validation_tasks: tuple[TaskInput, ...] | None = None,
    minimum_activation_rate: float = 1.0,
) -> tuple[EvolutionRunResult, PolicyArchive, MemoryEventSink]:
    sink = MemoryEventSink()
    residuals = residuals or _bundle_residuals()
    validation_tasks = validation_tasks or (
        _bundle_task("bundle-validation-a", "a"),
        _bundle_task("bundle-validation-b", "b"),
    )
    manifest = SplitManifest(
        benchmark="synthetic_bundle_selection",
        protocol="instance_holdout",
        seed="unit",
        train_ids=tuple(row.task_id for row in residuals),
        validation_ids=tuple(row.id for row in validation_tasks),
        test_ids=("bundle-sealed-test",),
        family_by_id={
            **{row.task_id: row.family for row in residuals},
            **{row.id: row.family for row in validation_tasks},
            "bundle-sealed-test": "family-a",
        },
    )
    proposer = StructuredHypothesisProposer(QueueProposalModel([]))
    archive = PolicyArchive(event_sink=sink)
    kernel = EvolutionKernel(
        proposer=proposer,
        validator=_validator(proposer, sink),
        counterfactual_runner=CounterfactualRunner(
            runtime=_runtime(sink),
            evaluator=TruthEvaluator(),
            event_sink=sink,
        ),
        promotion_gate=PromotionGate(
            replace(
                _promotion_spec(minimum_pairs=2, minimum_net_gain_count=1),
                minimum_activation_rate=minimum_activation_rate,
            ),
            event_sink=sink,
        ),
        archive=archive,
        split_guard=SplitAccessGuard(manifest, event_sink=sink),
        proposal_candidates_per_generation=3,
        candidate_selection_policy=selection_policy,
        candidate_bundle_policy=CANDIDATE_BUNDLE_POLICY_VERSION,
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
        event_sink=sink,
    )
    context = ValidationContext(
        evaluator_epoch="epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"raw", "relation_solver"}),
        baseline_lane="raw",
        trigger_feature_catalog=build_trigger_feature_catalog(residuals),
        contrastive_training_evidence_policy=(
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
        ),
    )
    result = kernel.evolve_once(
        residuals=residuals,
        validation_tasks=validation_tasks,
        validation_context=context,
        proposal_candidates=(
            proposal_candidates
            or _bundle_programs(
                low_precision_complement=low_precision_complement
            )
        ),
        trace_id="bundle-selection",
    )
    return result, archive, sink


def _runtime(sink: MemoryEventSink) -> PolicyRuntime:
    return PolicyRuntime(
        registry=LaneRegistry(
            [
                StaticLane("raw", "wrong", 0.8, 1.0),
                StaticLane("relation_solver", "correct", 0.95, 1.0),
            ]
        ),
        baseline_lane="raw",
        event_sink=sink,
        runtime_version="runtime-v1",
    )


def _task(task_id: str) -> TaskInput:
    return TaskInput(
        id=task_id,
        family="relation",
        features={"relation_type": "controlled_comparison", "self_contained": True},
        payload={"expected": "correct"},
    )


def _residuals() -> tuple[ResidualExample, ...]:
    return tuple(
        ResidualExample(
            transition_id=f"transition-{index}",
            task_id=f"train-{index}",
            family="relation",
            split=SplitName.TRAIN,
            features={
                "relation_type": "controlled_comparison",
                "self_contained": True,
                "requires_live_source": False,
            },
            failure_type="baseline_missed_explicit_relation",
            evaluator_feedback=("compare the controlled relation before selecting",),
            baseline_success=False,
        )
        for index in range(2)
    )


def _validation_context(residuals: tuple[ResidualExample, ...]) -> ValidationContext:
    return ValidationContext(
        evaluator_epoch="epoch-0",
        residuals=residuals,
        available_lanes=frozenset({"raw", "relation_solver"}),
        baseline_lane="raw",
        trigger_feature_catalog=build_trigger_feature_catalog(residuals),
    )


def _validator(proposer: StructuredHypothesisProposer, sink: MemoryEventSink) -> RecursiveValidationEngine:
    return RecursiveValidationEngine(
        [
            SchemaCheck(),
            RuntimeCandidateKindCheck(),
            TriggerVocabularyCheck(),
            TrainingSupportCheck(min_support=2),
            RuntimeActionCheck(),
            EvaluatorEpochCheck(),
        ],
        proposer=proposer,
        event_sink=sink,
    )


def _revise_once(
    parent: HypothesisProgram,
    response: Mapping[str, Any],
    *,
    residuals: tuple[ResidualExample, ...],
) -> HypothesisProgram:
    proposer = StructuredHypothesisProposer(
        QueueProposalModel([{"hypothesis": dict(response)}])
    )
    return proposer.revise(
        parent,
        failed_checks=({"check": "runtime_action", "passed": False},),
        residuals=residuals,
        depth=1,
        trace_id="canonical-repair-id-test",
    )


def _program_dict(
    *,
    lane: str = "relation_solver",
    hypothesis_id: str = "hyp-controlled-relation",
    status: str = "candidate",
    priority: int = 10,
) -> dict[str, Any]:
    return {
        "id": hypothesis_id,
        "kind": "policy",
        "statement": "Controlled-comparison tasks benefit from the relation solver lane.",
        "trigger": {
            "all_of": [
                {"key": "relation_type", "op": "eq", "value": "controlled_comparison"},
                {"key": "self_contained", "op": "eq", "value": True},
            ]
        },
        "anti_trigger": {
            "any_of": [{"key": "requires_live_source", "op": "eq", "value": True}]
        },
        "action_graph": [
            {"id": "enable", "operation": "enable_lane", "target": lane},
            {
                "id": "prioritize",
                "operation": "prioritize_lane",
                "target": lane,
                "value": priority,
                "depends_on": ["enable"],
            },
            {
                "id": "verify",
                "operation": "require_verifier",
                "target": "external_truth",
                "depends_on": ["prioritize"],
            },
        ],
        "expected_effect": {
            "metric": "task_success",
            "minimum_delta": 0.1,
            "maximum_harm_rate": 0.05,
            "maximum_cost_ratio": 3.0,
        },
        "verifier": {
            "checks": ["schema", "training_support", "runtime_action", "paired_validation"],
            "required_evidence": ["policy_off_outcome", "policy_on_outcome"],
            "anchor_id": "external_truth_anchor",
            "repair_on_failure": True,
            "max_repair_depth": 2,
        },
        "evaluator_epoch": "epoch-0",
        "fallback": "preserve_baseline",
        "status": status,
    }


def _contains_forbidden_answer_key(value: Any) -> bool:
    forbidden = {"gold", "gold_label", "correct_answer", "_answer"}
    if isinstance(value, Mapping):
        return bool(forbidden & set(value)) or any(
            _contains_forbidden_answer_key(child) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_answer_key(child) for child in value)
    return False

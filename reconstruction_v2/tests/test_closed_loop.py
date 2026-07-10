from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from assumption_agent.archive import (
    AnchorScore,
    EvaluatorEpoch,
    EvaluatorEpochController,
    EvaluatorSpec,
    PolicyArchive,
)
from assumption_agent.evaluation import (
    CounterfactualRunner,
    PromotionGate,
    PromotionGateSpec,
)
from assumption_agent.events import MemoryEventSink
from assumption_agent.evolution import EvolutionKernel
from assumption_agent.models import (
    ExternalOutcome,
    HypothesisProgram,
    HypothesisStatus,
    LaneResult,
    ResidualExample,
    SplitName,
    TaskInput,
)
from assumption_agent.proposer import StructuredHypothesisProposer
from assumption_agent.runtime import LaneRegistry, PolicyRuntime
from assumption_agent.splits import SplitAccessGuard, SplitManifest
from assumption_agent.validation import (
    EvaluatorEpochCheck,
    RecursiveValidationEngine,
    RuntimeActionCheck,
    SchemaCheck,
    TrainingSupportCheck,
    ValidationContext,
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
    repaired = _program_dict(hypothesis_id="hyp-repaired")
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
    assert tree.accepted_program.id == "hyp-repaired"
    assert tree.accepted_program.parent_id == root.id
    assert tree.accepted_program.lineage == (root.id,)
    assert all(not _contains_forbidden_answer_key(request) for request in model.requests)
    assert any(row["event"] == "hypothesis_repair_proposed" for row in sink.events)


def test_paired_counterfactual_gain_promotes_program() -> None:
    sink = MemoryEventSink()
    runtime = _runtime(sink)
    program = HypothesisProgram.from_dict(_program_dict())
    runner = CounterfactualRunner(runtime=runtime, evaluator=TruthEvaluator(), event_sink=sink)
    tasks = tuple(_task(f"validation-{index}") for index in range(20))
    pairs = runner.run(tasks, program=program, split=SplitName.VALIDATION)
    gate = PromotionGate(
        PromotionGateSpec(minimum_pairs=10, confidence=0.9, minimum_net_gain_count=2),
        event_sink=sink,
    )

    decision = gate.evaluate(program, pairs, sealed_test_accessed=False)

    assert decision.allowed is True
    assert decision.summary.baseline_success_count == 0
    assert decision.summary.candidate_success_count == 20
    assert decision.summary.activation_count == 20
    assert decision.effect_lower_bound == 1.0


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
    model = QueueProposalModel([{"hypotheses": [_program_dict()]}])
    proposer = StructuredHypothesisProposer(model, event_sink=sink)
    validator = _validator(proposer, sink)
    runtime = _runtime(sink)
    runner = CounterfactualRunner(runtime=runtime, evaluator=TruthEvaluator(), event_sink=sink)
    gate = PromotionGate(
        PromotionGateSpec(minimum_pairs=10, confidence=0.9, minimum_net_gain_count=2),
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
    assert archive.incumbent_id == result.archive_node.id
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
            features={"relation_type": "controlled_comparison", "self_contained": True},
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
    )


def _validator(proposer: StructuredHypothesisProposer, sink: MemoryEventSink) -> RecursiveValidationEngine:
    return RecursiveValidationEngine(
        [SchemaCheck(), TrainingSupportCheck(min_support=2), RuntimeActionCheck(), EvaluatorEpochCheck()],
        proposer=proposer,
        event_sink=sink,
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

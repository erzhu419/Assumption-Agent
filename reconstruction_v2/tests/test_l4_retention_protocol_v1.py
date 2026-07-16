from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from assumption_agent.archive import ArchiveNodeStatus, PolicyArchive
from assumption_agent.benchmarks.l4_retention_protocol_v1 import (
    ARM_IDS,
    CONSUMPTION_FILENAME,
    FAILURE_FILENAME,
    FREEZE_FILENAME,
    PRIMARY_METRIC,
    REPORT_FILENAME,
    ArchiveEpochBinding,
    FixedItemBlockManifest,
    FrozenOperatorSlot,
    L4RetentionPlan,
    L4RetentionProtocolError,
    OperatorQuery,
    RetentionItem,
    deterministic_rrf,
    execute_l4_retention_plan,
)
from assumption_agent.models import (
    ActionNode,
    ExpectedEffect,
    HypothesisKind,
    HypothesisProgram,
    HypothesisStatus,
    TriggerSpec,
    VerifierContract,
    stable_hash,
)


EPOCH = "l4-synthetic-evaluator-epoch-v1"


def _program(
    program_id: str, *, status: HypothesisStatus
) -> HypothesisProgram:
    return HypothesisProgram(
        id=program_id,
        kind=HypothesisKind.POLICY,
        statement="Frozen synthetic ranked operator.",
        trigger=TriggerSpec(),
        anti_trigger=TriggerSpec(),
        action_graph=(
            ActionNode(
                id="retrieve",
                operation="enable_lane",
                target=f"lane-{program_id}",
            ),
        ),
        expected_effect=ExpectedEffect(
            metric="offline_micro_support_recall_at_5",
            minimum_delta=0.0,
            maximum_harm_rate=1.0,
            maximum_cost_ratio=1.0,
        ),
        verifier=VerifierContract(
            checks=("offline_support_recall_at_5",),
            anchor_id="synthetic-l4-anchor",
            repair_on_failure=False,
            max_repair_depth=0,
        ),
        evaluator_epoch=EPOCH,
        fallback="preserve_baseline",
        status=status,
    )


def _archive_binding() -> ArchiveEpochBinding:
    archive = PolicyArchive()
    retained = _program("retained-P", status=HypothesisStatus.PROMOTED)
    challenger = _program("novel-Q", status=HypothesisStatus.CANDIDATE)
    archive.register_hypothesis(retained)
    archive.register_hypothesis(challenger)
    node = archive.create_node(
        active_hypothesis_ids=(retained.id,),
        evaluator_epoch_id=EPOCH,
        runtime_version="synthetic-l4-runtime-v1",
    )
    archive.nodes[node.id] = replace(
        node, status=ArchiveNodeStatus.INCUMBENT
    )
    archive.incumbent_id = node.id
    return ArchiveEpochBinding.from_archive(
        archive,
        p_hypothesis_id=retained.id,
        q_hypothesis_id=challenger.id,
    )


def _items() -> tuple[RetentionItem, ...]:
    return (
        RetentionItem(
            item_id="private-item-one",
            block_id="fresh-block-a",
            document_ids=(
                "s1a",
                "s1b",
                "p12",
                "p13",
                "p14",
                "p15",
                "q12",
                "q13",
                "q14",
                "q15",
            ),
            support_document_ids=("s1a", "s1b"),
            operator_input={"query": "private-query-one"},
        ),
        RetentionItem(
            item_id="private-item-two",
            block_id="fresh-block-b",
            document_ids=(
                "s2a",
                "s2b",
                "p21",
                "p22",
                "p23",
                "p24",
                "q22",
                "q23",
                "q24",
                "q25",
            ),
            support_document_ids=("s2a", "s2b"),
            operator_input={"query": "private-query-two"},
        ),
    )


def _plan(
    items: tuple[RetentionItem, ...], root: Path
) -> tuple[L4RetentionPlan, dict[str, str]]:
    archive = _archive_binding()
    implementation_hashes = {
        "P": stable_hash({"operator": "synthetic-P-v1"}),
        "Q": stable_hash({"operator": "synthetic-Q-v1"}),
    }
    slots = (
        FrozenOperatorSlot(
            slot_id="P",
            operator_id="synthetic-operator-P",
            implementation_hash=implementation_hashes["P"],
            hypothesis_id=archive.p_hypothesis_id,
            hypothesis_hash=archive.p_hypothesis_hash,
            candidate_budget=5,
        ),
        FrozenOperatorSlot(
            slot_id="Q",
            operator_id="synthetic-operator-Q",
            implementation_hash=implementation_hashes["Q"],
            hypothesis_id=archive.q_hypothesis_id,
            hypothesis_hash=archive.q_hypothesis_hash,
            candidate_budget=5,
        ),
    )
    manifest = FixedItemBlockManifest.freeze(
        items,
        dataset_binding_hash=stable_hash({"dataset": "synthetic-fresh-v1"}),
        fresh_partition_hash=stable_hash(
            {"partition": "never-previously-measured-v1"}
        ),
    )
    return (
        L4RetentionPlan.freeze(
            manifest=manifest,
            archive_epoch=archive,
            operator_slots=slots,
            authorization_hash=stable_hash(
                {"authorization": "one-shot-synthetic-v1"}
            ),
            execution_root=root,
            baseline_candidate_budget=5,
        ),
        implementation_hashes,
    )


class _MapOperator:
    def __init__(
        self,
        *,
        slot_id: str,
        operator_id: str,
        implementation_hash: str,
        calls: list[tuple[str, str, int]],
        lock: threading.Lock,
        fail: bool = False,
    ) -> None:
        self.slot_id = slot_id
        self.operator_id = operator_id
        self.implementation_hash = implementation_hash
        self.calls = calls
        self.lock = lock
        self.fail = fail

    def retrieve(
        self, query: OperatorQuery, *, candidate_budget: int
    ) -> tuple[str, ...]:
        with self.lock:
            self.calls.append((self.slot_id, query.item_id, candidate_budget))
        assert "support_document_ids" not in query.payload
        if self.fail:
            raise RuntimeError("synthetic private failure text")
        rankings = {
            ("P", "private-item-one"): (
                "s1a",
                "p12",
                "p13",
                "p14",
                "p15",
            ),
            ("Q", "private-item-one"): (
                "s1b",
                "q12",
                "q13",
                "q14",
                "q15",
            ),
            ("P", "private-item-two"): (
                "p21",
                "p22",
                "p23",
                "p24",
                "s2a",
            ),
            ("Q", "private-item-two"): (
                "s2b",
                "q22",
                "q23",
                "q24",
                "q25",
            ),
        }
        return rankings[(self.slot_id, query.item_id)]


def test_fixed_four_arm_retention_measurement_is_hash_safe_and_one_shot(
    tmp_path: Path,
) -> None:
    items = _items()
    root = tmp_path / "fresh-l4-root"
    plan, hashes = _plan(items, root)
    calls: list[tuple[str, str, int]] = []
    lock = threading.Lock()

    def factory(slot: FrozenOperatorSlot) -> _MapOperator:
        return _MapOperator(
            slot_id=slot.slot_id,
            operator_id=slot.operator_id,
            implementation_hash=hashes[slot.slot_id],
            calls=calls,
            lock=lock,
        )

    report = execute_l4_retention_plan(
        plan=plan,
        items=items,
        operator_factory=factory,
        execution_root=root,
    )

    assert report["valid"] is True
    measurement = report["measurement"]
    assert measurement["primary_metric"] == PRIMARY_METRIC
    assert tuple(measurement["arm_metrics"]) == ARM_IDS
    arms = measurement["arm_metrics"]
    assert arms["empty"]["support_hit_count"] == 0
    assert arms["P"]["support_hit_count"] == 2
    assert arms["Q"]["support_hit_count"] == 2
    assert arms["P_plus_Q"]["support_hit_count"] == 3
    assert arms["P_plus_Q"]["micro_support_recall_at_5"] == 0.75
    assert measurement["retention"]["estimand"] == "Y(P_plus_Q)-Y(Q)"
    assert measurement["retention"]["delta"] == 0.25
    assert measurement["novelty"]["net_delta"] == 0.25
    assert measurement["novelty"]["new_support_coverage_rate"] == 0.5
    assert measurement["forgetting"]["support_forgetting_rate"] == 0.25
    assert measurement["forgetting"]["forgotten_support_count"] == 1
    assert len(measurement["block_aggregates"]) == 2

    assert len(calls) == 4
    assert sorted(slot for slot, _, _ in calls) == ["P", "P", "Q", "Q"]
    assert all(budget == 5 for _, _, budget in calls)
    assert report["execution"]["operator_call_count"] == 4
    assert report["execution"][
        "all_operator_terminals_joined_before_gold_scoring"
    ] is True
    assert report["execution"]["retries"] == 0
    assert report["execution"]["replays"] == 0
    assert report["execution"]["resamples"] == 0
    assert report["execution"]["outcome_dependent_arm_changes"] == 0
    assert report["online_evaluator_calls"] == 0
    assert report["raw_content_persisted"] is False

    assert (root / FREEZE_FILENAME).is_file()
    assert (root / CONSUMPTION_FILENAME).is_file()
    assert (root / REPORT_FILENAME).is_file()
    persisted = json.loads((root / REPORT_FILENAME).read_text("utf-8"))
    declared = persisted.pop("report_hash")
    assert stable_hash(persisted) == declared == report["report_hash"]

    persisted_text = "\n".join(
        path.read_text("utf-8") for path in sorted(root.iterdir())
    )
    for private_text in (
        "private-item-one",
        "private-item-two",
        "private-query-one",
        "private-query-two",
        "fresh-block-a",
        "fresh-block-b",
        "s1a",
        "s2a",
    ):
        assert private_text not in persisted_text

    calls_before_replay = list(calls)
    with pytest.raises(L4RetentionProtocolError, match="replay is forbidden"):
        execute_l4_retention_plan(
            plan=plan,
            items=items,
            operator_factory=factory,
            execution_root=root,
        )
    assert calls == calls_before_replay


def test_manifest_drift_is_rejected_before_root_or_operator_access(
    tmp_path: Path,
) -> None:
    items = _items()
    root = tmp_path / "fresh-l4-root"
    plan, _ = _plan(items, root)
    changed = (
        replace(items[0], operator_input={"query": "changed-private-query"}),
        items[1],
    )
    factory_called = False

    def factory(slot):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("must not materialize")

    with pytest.raises(L4RetentionProtocolError, match="fixed item/block"):
        execute_l4_retention_plan(
            plan=plan,
            items=changed,
            operator_factory=factory,
            execution_root=root,
        )
    assert root.exists() is False
    assert factory_called is False


def test_operator_failure_closes_root_without_retry_or_error_text(
    tmp_path: Path,
) -> None:
    items = _items()
    root = tmp_path / "failed-l4-root"
    plan, hashes = _plan(items, root)
    calls: list[tuple[str, str, int]] = []
    lock = threading.Lock()

    def factory(slot: FrozenOperatorSlot) -> _MapOperator:
        return _MapOperator(
            slot_id=slot.slot_id,
            operator_id=slot.operator_id,
            implementation_hash=hashes[slot.slot_id],
            calls=calls,
            lock=lock,
            fail=slot.slot_id == "Q",
        )

    with pytest.raises(L4RetentionProtocolError, match="cannot be replayed"):
        execute_l4_retention_plan(
            plan=plan,
            items=items,
            operator_factory=factory,
            execution_root=root,
        )

    failure = json.loads((root / FAILURE_FILENAME).read_text("utf-8"))
    declared = failure.pop("failure_hash")
    assert stable_hash(failure) == declared
    assert failure["authorization_consumed"] is True
    assert failure["retries"] == failure["replays"] == failure["resamples"] == 0
    assert failure["replay_authorized"] is False
    assert failure["operator_attempt_count"] <= 4
    assert failure["operator_terminal_count"] < 4
    failure_text = (root / FAILURE_FILENAME).read_text("utf-8")
    assert "synthetic private failure text" not in failure_text
    assert "private-item" not in failure_text

    call_count = len(calls)
    with pytest.raises(L4RetentionProtocolError, match="replay is forbidden"):
        execute_l4_retention_plan(
            plan=plan,
            items=items,
            operator_factory=factory,
            execution_root=root,
        )
    assert len(calls) == call_count


def test_rrf_is_exact_deterministic_and_rejects_protocol_drift() -> None:
    assert deterministic_rrf((('b', 'd'), ('a', 'c'))) == (
        "a",
        "b",
        "c",
        "d",
    )
    assert deterministic_rrf((('shared', 'x'), ('shared', 'y'))) == (
        "shared",
        "x",
        "y",
    )
    with pytest.raises(L4RetentionProtocolError, match="top_k"):
        deterministic_rrf((('a',),), top_k=4)
    with pytest.raises(L4RetentionProtocolError, match="constant"):
        deterministic_rrf((('a',),), rrf_constant=1)


def test_archive_binding_rejects_Q_that_is_already_retained() -> None:
    archive = PolicyArchive()
    retained = _program("retained-P", status=HypothesisStatus.PROMOTED)
    challenger = _program("novel-Q", status=HypothesisStatus.CANDIDATE)
    archive.register_hypothesis(retained)
    archive.register_hypothesis(challenger)
    node = archive.create_node(
        active_hypothesis_ids=(retained.id, challenger.id),
        evaluator_epoch_id=EPOCH,
        runtime_version="synthetic-l4-runtime-v1",
    )
    archive.nodes[node.id] = replace(
        node, status=ArchiveNodeStatus.INCUMBENT
    )
    archive.incumbent_id = node.id

    with pytest.raises(L4RetentionProtocolError, match="already belongs"):
        ArchiveEpochBinding.from_archive(
            archive,
            p_hypothesis_id=retained.id,
            q_hypothesis_id=challenger.id,
        )

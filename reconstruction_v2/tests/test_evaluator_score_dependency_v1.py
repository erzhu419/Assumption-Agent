from __future__ import annotations

import copy

import pytest

from assumption_agent.benchmarks.evaluator_score_dependency_v1 import (
    CachedOutputBinding,
    EVALUATOR_DEPENDENT,
    INDEPENDENT_OBJECTIVE,
    ScoreDependencyError,
    ScoreDependencyLedger,
    cached_rescore_receipt,
)
from assumption_agent.models import stable_hash


def _sha(label: str) -> str:
    return stable_hash({"label": label})


def _cache() -> tuple[CachedOutputBinding, list[str]]:
    item_hashes = [_sha("output-a"), _sha("output-b")]
    return (
        CachedOutputBinding(
            producer_execution_sha256=_sha("execution"),
            output_set_sha256=stable_hash(item_hashes),
            item_set_sha256=_sha("items"),
            item_count=2,
        ),
        item_hashes,
    )


def test_epoch_transition_invalidates_only_dependent_scores() -> None:
    cache, _ = _cache()
    ledger = ScoreDependencyLedger()
    dependent = ledger.record(
        archive_node_id="node-P",
        metric="search_surrogate",
        successes=7,
        total=8,
        item_set_sha256=_sha("items"),
        evidence_sha256=_sha("dependent-evidence"),
        dependency_kind=EVALUATOR_DEPENDENT,
        evaluator_epoch_id="epoch-0",
        cached_outputs=cache,
    )
    independent = ledger.record(
        archive_node_id="node-P",
        metric="official_support_recall",
        successes=6,
        total=8,
        item_set_sha256=_sha("items"),
        evidence_sha256=_sha("independent-evidence"),
        dependency_kind=INDEPENDENT_OBJECTIVE,
    )
    invalidated = ledger.transition_epoch(
        old_epoch_id="epoch-0",
        new_epoch_id="epoch-1",
        transition_evidence_sha256=_sha("transition"),
    )
    assert invalidated == (dependent.id,)
    assert ledger.records[dependent.id].valid is False
    assert ledger.records[independent.id].valid is True
    assert ScoreDependencyLedger.from_dict(ledger.to_dict()).to_dict() == ledger.to_dict()


def test_cached_rescore_binds_same_outputs_and_zero_producer_calls() -> None:
    cache, item_hashes = _cache()
    receipt = cached_rescore_receipt(
        cached_outputs=cache,
        prior_epoch_id="epoch-0",
        new_epoch_id="epoch-1",
        evaluator_implementation_sha256=_sha("evaluator"),
        score_evidence_sha256=_sha("score"),
        item_result_hashes=item_hashes,
    )
    assert receipt["producer_rerun_calls"] == 0
    assert receipt["cached_output_binding_sha256"] == cache.binding_sha256
    with pytest.raises(ScoreDependencyError, match="output set"):
        cached_rescore_receipt(
            cached_outputs=cache,
            prior_epoch_id="epoch-0",
            new_epoch_id="epoch-1",
            evaluator_implementation_sha256=_sha("evaluator"),
            score_evidence_sha256=_sha("score"),
            item_result_hashes=list(reversed(item_hashes)),
        )


def test_ledger_fails_closed_on_tamper_and_duplicate_transition() -> None:
    ledger = ScoreDependencyLedger()
    ledger.record(
        archive_node_id="node-P",
        metric="official_support_recall",
        successes=1,
        total=2,
        item_set_sha256=_sha("items"),
        evidence_sha256=_sha("evidence"),
        dependency_kind=INDEPENDENT_OBJECTIVE,
    )
    payload = ledger.to_dict()
    tampered = copy.deepcopy(payload)
    next(iter(tampered["records"].values()))["successes"] = 2
    with pytest.raises(ScoreDependencyError, match="hash"):
        ScoreDependencyLedger.from_dict(tampered)

    ledger.transition_epoch(
        old_epoch_id="epoch-0",
        new_epoch_id="epoch-1",
        transition_evidence_sha256=_sha("transition"),
    )
    with pytest.raises(ScoreDependencyError, match="already replaced"):
        ledger.transition_epoch(
            old_epoch_id="epoch-0",
            new_epoch_id="epoch-2",
            transition_evidence_sha256=_sha("transition-2"),
        )


def test_rehashed_transition_cannot_detach_invalidated_record_or_revive_epoch() -> None:
    ledger = ScoreDependencyLedger()
    dependent = ledger.record(
        archive_node_id="node-P",
        metric="search_surrogate",
        successes=1,
        total=2,
        item_set_sha256=_sha("items"),
        evidence_sha256=_sha("dependent"),
        dependency_kind=EVALUATOR_DEPENDENT,
        evaluator_epoch_id="epoch-0",
    )
    ledger.transition_epoch(
        old_epoch_id="epoch-0",
        new_epoch_id="epoch-1",
        transition_evidence_sha256=_sha("transition"),
    )
    payload = copy.deepcopy(ledger.to_dict())
    transition = payload["transitions"][0]
    transition["invalidated_score_record_ids"] = []
    transition_body = dict(transition)
    transition_body.pop("transition_sha256")
    transition["transition_sha256"] = stable_hash(transition_body)
    ledger_body = dict(payload)
    ledger_body.pop("ledger_sha256")
    payload["ledger_sha256"] = stable_hash(ledger_body)
    with pytest.raises(ScoreDependencyError, match="not closed"):
        ScoreDependencyLedger.from_dict(payload)

    with pytest.raises(ScoreDependencyError, match="already replaced"):
        ledger.record(
            archive_node_id="node-Q",
            metric="search_surrogate",
            successes=1,
            total=2,
            item_set_sha256=_sha("items-q"),
            evidence_sha256=_sha("dependent-q"),
            dependency_kind=EVALUATOR_DEPENDENT,
            evaluator_epoch_id="epoch-0",
        )
    assert ledger.records[dependent.id].valid is False

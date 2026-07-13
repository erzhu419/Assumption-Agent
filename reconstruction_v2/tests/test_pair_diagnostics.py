from __future__ import annotations

import pytest

from assumption_agent.benchmarks import paper_freeze
from assumption_agent.evaluation import summarize_pairs
from assumption_agent.models import (
    CounterfactualPair,
    ExternalOutcome,
    LaneResult,
    RuntimeExecution,
    SplitName,
)


def test_pair_diagnostics_report_activation_precision_and_abstention() -> None:
    pairs = [
        _pair(
            index=0,
            activated=True,
            baseline_success=False,
            candidate_success=True,
        )
    ]
    pairs.extend(
        _pair(
            index=index,
            activated=index < 6,
            baseline_success=False,
            candidate_success=False,
        )
        for index in range(1, 16)
    )

    summary = summarize_pairs(pairs)
    payload = summary.to_dict()

    assert summary.activation_count == 6
    assert summary.valid_activation_count == 6
    assert summary.activated_gain_count == 1
    assert summary.activated_harm_count == 0
    assert payload["activation_precision"] == pytest.approx(1 / 6)
    assert payload["activation_precision_defined"] is True
    assert payload["activated_harm_rate"] == 0.0
    assert payload["activated_harm_rate_defined"] is True
    assert summary.abstention_count == 10
    assert payload["abstention_rate"] == pytest.approx(10 / 16)


def test_pair_diagnostics_use_null_for_zero_activation_denominator() -> None:
    summary = summarize_pairs(
        [
            _pair(
                index=index,
                activated=False,
                baseline_success=False,
                candidate_success=False,
            )
            for index in range(2)
        ]
    )
    payload = summary.to_dict()

    assert summary.valid_activation_count == 0
    assert payload["activation_precision"] is None
    assert payload["activation_precision_defined"] is False
    assert payload["activated_harm_rate"] is None
    assert payload["activated_harm_rate_defined"] is False
    assert summary.abstention_count == 2
    assert payload["abstention_rate"] == 1.0


def test_pair_diagnostics_exclude_invalid_activated_pair() -> None:
    summary = summarize_pairs(
        [
            _pair(
                index=0,
                activated=True,
                baseline_success=False,
                candidate_success=True,
            ),
            _pair(
                index=1,
                activated=True,
                baseline_success=True,
                candidate_success=False,
            ),
            _pair(
                index=2,
                activated=True,
                baseline_success=False,
                candidate_success=True,
                valid=False,
            ),
        ]
    )

    assert summary.activation_count == 3
    assert summary.invalid_pair_count == 1
    assert summary.valid_activation_count == 2
    assert summary.activated_gain_count == 1
    assert summary.activated_harm_count == 1
    assert summary.activation_precision == 0.5
    assert summary.activated_harm_rate == 0.5


@pytest.mark.parametrize(
    ("mismatch", "count_field"),
    (
        ("provider", "provider_mismatch_count"),
        ("budget", "budget_mismatch_count"),
    ),
)
def test_pair_diagnostics_exclude_mismatched_activated_pair(
    mismatch: str,
    count_field: str,
) -> None:
    summary = summarize_pairs(
        [
            _pair(
                index=0,
                activated=True,
                baseline_success=False,
                candidate_success=True,
                mismatch=mismatch,
            )
        ]
    )

    assert getattr(summary, count_field) == 1
    assert summary.valid_activation_count == 0
    assert summary.activated_gain_count == 0
    assert summary.activation_precision is None


@pytest.mark.parametrize(
    "protocol_version",
    (
        "3.6.0",
        "3.7.0",
        "3.8.0",
        "3.9.0",
        "3.10.0",
        "3.11.0",
        "3.12.0",
        "3.13.0",
        "3.14.0",
    ),
)
def test_freeze_contrastive_protocol_rejects_tampered_pair_diagnostic(
    protocol_version: str,
) -> None:
    summary = summarize_pairs(
        [
            _pair(
                index=0,
                activated=True,
                baseline_success=False,
                candidate_success=True,
            ),
            _pair(
                index=1,
                activated=True,
                baseline_success=False,
                candidate_success=False,
            ),
        ]
    ).to_dict()
    summary["activation_precision"] = 1.0

    with pytest.raises(
        ValueError,
        match="promotion summary diagnostic value is inconsistent",
    ):
        paper_freeze._pair_summary_from_mapping(
            summary,
            confidence=0.9,
            protocol_version=protocol_version,
        )


@pytest.mark.parametrize(
    "protocol_version",
    (
        "3.6.0",
        "3.7.0",
        "3.8.0",
        "3.9.0",
        "3.10.0",
        "3.11.0",
        "3.12.0",
        "3.13.0",
        "3.14.0",
    ),
)
def test_freeze_contrastive_protocol_rejects_tampered_activated_gain_count(
    protocol_version: str,
) -> None:
    summary = summarize_pairs(
        [
            _pair(
                index=0,
                activated=True,
                baseline_success=False,
                candidate_success=True,
            )
        ]
    ).to_dict()
    summary["activated_gain_count"] = 0
    summary["activation_precision"] = 0.0

    with pytest.raises(
        ValueError,
        match="promotion summary activated outcomes are inconsistent",
    ):
        paper_freeze._pair_summary_from_mapping(
            summary,
            confidence=0.9,
            protocol_version=protocol_version,
        )


def test_freeze_preserves_v3_1_through_v3_5_pair_summary_schema() -> None:
    summary = summarize_pairs(
        [
            _pair(
                index=0,
                activated=True,
                baseline_success=False,
                candidate_success=True,
            )
        ]
    ).to_dict()
    for key in paper_freeze._PROMOTION_SUMMARY_DIAGNOSTIC_COUNT_KEYS:
        summary.pop(key)
    for key in paper_freeze._PROMOTION_SUMMARY_DIAGNOSTIC_DERIVED_KEYS:
        summary.pop(key)

    for protocol_version in ("3.1.0", "3.2.0", "3.3.0", "3.4.0", "3.5.0"):
        parsed = paper_freeze._pair_summary_from_mapping(
            summary,
            confidence=0.9,
            protocol_version=protocol_version,
        )
        assert parsed.pair_count == 1
        assert parsed.gain_count == 1


def _pair(
    *,
    index: int,
    activated: bool,
    baseline_success: bool,
    candidate_success: bool,
    valid: bool = True,
    mismatch: str | None = None,
) -> CounterfactualPair:
    task_id = f"task-{index}"
    baseline_metadata = {
        "provider_fingerprint": "provider-fixed",
        "fairness_fingerprint": "budget-fixed",
    }
    candidate_metadata = dict(baseline_metadata)
    if mismatch == "provider":
        candidate_metadata["provider_fingerprint"] = "provider-mismatch"
    elif mismatch == "budget":
        candidate_metadata["fairness_fingerprint"] = "budget-mismatch"
    elif mismatch is not None:
        raise ValueError(f"unsupported mismatch: {mismatch}")
    baseline = RuntimeExecution(
        task_id=task_id,
        selected_result=LaneResult(
            lane="baseline",
            answer=f"baseline-{index}",
            confidence=1.0,
            cost=1.0,
            metadata=baseline_metadata,
        ),
        lane_results=(),
        activated_hypothesis_ids=(),
        plan_hash=f"baseline-{index}",
        action_activated=False,
        baseline_preserved=True,
    )
    candidate = RuntimeExecution(
        task_id=task_id,
        selected_result=LaneResult(
            lane="candidate",
            answer=f"candidate-{index}",
            confidence=1.0,
            cost=1.0,
            metadata=candidate_metadata,
        ),
        lane_results=(),
        activated_hypothesis_ids=("candidate",) if activated else (),
        plan_hash=f"candidate-{index}",
        action_activated=activated,
        baseline_preserved=True,
    )
    evaluation_valid = 1.0 if valid else 0.0
    return CounterfactualPair(
        task_id=task_id,
        split=SplitName.VALIDATION,
        evaluator_epoch="epoch-fixed",
        baseline=baseline,
        candidate=candidate,
        baseline_outcome=ExternalOutcome(
            task_id=task_id,
            success=baseline_success,
            score=float(baseline_success),
            evaluator_id="offline-fixed",
            evaluator_epoch="epoch-fixed",
            metrics={"evaluation_valid": evaluation_valid},
        ),
        candidate_outcome=ExternalOutcome(
            task_id=task_id,
            success=candidate_success,
            score=float(candidate_success),
            evaluator_id="offline-fixed",
            evaluator_epoch="epoch-fixed",
            metrics={"evaluation_valid": evaluation_valid},
        ),
    )

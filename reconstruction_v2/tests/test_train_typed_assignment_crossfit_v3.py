from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

import pytest

from assumption_agent.benchmarks.train_typed_assignment_crossfit_v3 import (
    EXPECTED_ACTIVE_EXECUTION_COUNT,
    EXPECTED_INACTIVE_RAW_REPLAY_COUNT,
    EXECUTION_IMPLEMENTATION_RELATIVE_PATHS,
    IMPLEMENTATION_RELATIVE_PATHS,
    MAXIMUM_CONCURRENT_RUNNER_CALLS,
    MINIMUM_DISTINCT_FOLD_RECOVERIES,
    PREREGISTRATION_RELATIVE_PATH,
    REGISTERED_HELDOUT_ITEM_IDS,
    TYPED_ASSIGNMENT_PROVIDER_POLICY,
    TrainTypedAssignmentCrossfitError,
    TypedAssignmentCrossfitCompileV3,
    _candidate_id,
    _runtime_receipt_bodies_are_bound,
    _verify_provider_selection_receipt_v3,
    compile_v320_typed_assignment_crossfit_v3,
    verify_preregistration_commit_v3,
    write_plus_transport_failure_receipt_v3,
    write_provider_selection_receipt_v3,
)
from assumption_agent.benchmarks.train_execution_contract_crossfit_v2 import (
    SOURCE_RANKING_REPORT_RELATIVE_PATH,
)
from assumption_agent.benchmarks.v320_train_candidate_material_v2 import (
    V320_SOURCE_RELATIVE_ROOT,
)
from assumption_agent.events import Event
from assumption_agent.models import stable_hash


SHA_A = "a" * 64
SHA_B = "b" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / V320_SOURCE_RELATIVE_ROOT
SOURCE_RANKING_REPORT = PROJECT_ROOT / SOURCE_RANKING_REPORT_RELATIVE_PATH


@dataclass(frozen=True)
class _CellStub:
    heldout_item_id: str

    def preregistration_payload(self) -> dict[str, object]:
        return {"heldout_item_id": self.heldout_item_id}


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_canary(path: Path, *, accepted: bool) -> None:
    root_hash = stable_hash({"canary": "root"})
    path.write_text(
        json.dumps(
            {
                "canary_version": "proposal_canary_v1",
                "model": "gpt-5.4-mini",
                "provider_chain": ["openai_compatible"],
                "provider_chain_hash": stable_hash(
                    {
                        "providers": ["openai_compatible"],
                        "model": "gpt-5.4-mini",
                    }
                ),
                "root_hypothesis_id": "canary-root",
                "root_hypothesis_hash": root_hash,
                "recursive_node_count": 1,
                "recursive_depth": 0,
                "accepted": accepted,
                "accepted_program": ({"id": "accepted"} if accepted else None),
                "nodes": [
                    {
                        "hypothesis_id": "canary-root",
                        "hypothesis_hash": root_hash,
                        "depth": 0,
                        "passed": accepted,
                        "checks": [],
                        "child_id": None,
                    }
                ],
                "api_key_present": True,
                "secret_value_persisted": False,
                "raw_content_persisted": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_transport_failure_events(path: Path) -> None:
    rows = (
        Event(
            event="model_attempt_started",
            stage="model.transport",
            trace_id="live-proposal-canary",
            payload={
                "request_hash": SHA_A,
                "attempt": 1,
                "attempt_limit": 1,
                "model": "gpt-5.4-mini",
                "timeout_seconds": 30.0,
                "endpoint_hash": SHA_B,
            },
        ).to_dict(),
        Event(
            event="model_attempt_failed",
            stage="model.transport",
            trace_id="live-proposal-canary",
            payload={
                "request_hash": SHA_A,
                "attempt": 1,
                "elapsed_seconds": 30.0,
                "error_type": "TimeoutError",
                "http_status": None,
                "retryable": False,
                "model": "gpt-5.4-mini",
            },
        ).to_dict(),
        Event(
            event="model_provider_failed",
            stage="model.provider_chain",
            trace_id="live-proposal-canary",
            payload={
                "provider": "openai_compatible",
                "provider_chain_hash": SHA_B,
                "request_hash": SHA_A,
                "model": "gpt-5.4-mini",
                "error_type": "RuntimeError",
                "fallback_available": False,
                "circuit_opened": False,
                "raw_error_persisted": False,
            },
        ).to_dict(),
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_candidate_identity_binds_runtime_class_and_all_file_hashes() -> None:
    candidate_id = _candidate_id(
        heldout_item_id="organize-messy-files-5",
        runtime_class_hash=SHA_A,
        implementation_file_set_hash=SHA_B,
    )

    assert "organize-5-typed-assignment-v3" in candidate_id
    assert SHA_A in candidate_id
    assert SHA_B in candidate_id
    with pytest.raises(TrainTypedAssignmentCrossfitError):
        _candidate_id(
            heldout_item_id="organize-messy-files-3",
            runtime_class_hash=SHA_A,
            implementation_file_set_hash=SHA_B,
        )


def test_actual_report_binds_complete_runtime_receipt_bodies() -> None:
    tool_sha = "c" * 64
    prepare_body = {
        "runtime_tool_sha256": tool_sha,
        "contract_hash": "d" * 64,
        "evidence_set_hash": "e" * 64,
        "host_safe_receipt": True,
        "raw_public_instruction_in_receipt": False,
        "raw_content_evidence_in_receipt": False,
        "source_filenames_in_receipt": False,
    }
    reconciliation_body = {
        "runtime_tool_sha256": tool_sha,
        "contract_hash": "d" * 64,
        "evidence_set_hash": "e" * 64,
        "host_safe_receipt": True,
        "raw_public_instruction_in_receipt": False,
        "raw_content_evidence_in_receipt": False,
        "source_filenames_in_receipt": False,
    }
    prepare = {
        **prepare_body,
        "receipt_hash": stable_hash(prepare_body),
    }
    reconciliation = {
        **reconciliation_body,
        "receipt_hash": stable_hash(reconciliation_body),
    }
    row = {
        "prepare_receipt_hash": prepare["receipt_hash"],
        "reconciliation_receipt_hash": reconciliation["receipt_hash"],
        "contract_hash": "d" * 64,
        "evidence_set_hash": "e" * 64,
        "prepare_receipt_body": prepare,
        "reconciliation_receipt_body": reconciliation,
        "post_agent_runtime_delivery": {
            "runtime_tool_sha256": tool_sha,
            "container_readback_sha256": tool_sha,
            "fresh_unpredictable_path_selected_after_agent_exit": True,
            "pre_agent_prepare_tool_removed_before_agent_start": True,
        },
    }

    assert _runtime_receipt_bodies_are_bound(row) is True
    row["post_agent_runtime_delivery"] = {
        **row["post_agent_runtime_delivery"],
        "container_readback_sha256": "f" * 64,
    }
    assert _runtime_receipt_bodies_are_bound(row) is False


def test_preregistration_labels_history_and_fixes_one_parallel_batch() -> None:
    implementation_without_hash = {
        "implementation_binding_policy": "test",
        "runtime_class_hash": SHA_A,
        "implementation_files": [],
        "implementation_file_count": 0,
        "implementation_file_set_hash": SHA_B,
        "raw_implementation_content_persisted": False,
    }
    implementation_receipt = {
        **implementation_without_hash,
        "implementation_receipt_hash": stable_hash(
            implementation_without_hash
        ),
    }
    compilation = TypedAssignmentCrossfitCompileV3(
        output_root=Path("unused"),
        cells=tuple(_CellStub(value) for value in REGISTERED_HELDOUT_ITEM_IDS),
        implementation_receipt=implementation_receipt,
        report={},
    )

    payload = compilation.preregistration_without_hash()

    assert payload["historically_informed_candidate_execution"] is True
    assert payload["prior_outcome_design_used"] is True
    assert payload["score_cohort_previously_observed"] is True
    assert payload["globally_unbiased_crossfit"] is False
    assert payload["new_cell_outcomes_observed_at_registration_time"] is False
    assert payload["expected_active_execution_count"] == (
        EXPECTED_ACTIVE_EXECUTION_COUNT
    )
    assert payload["expected_inactive_raw_replay_count"] == (
        EXPECTED_INACTIVE_RAW_REPLAY_COUNT
    )
    assert payload["maximum_concurrent_runner_calls"] == (
        MAXIMUM_CONCURRENT_RUNNER_CALLS
    )
    assert payload["maximum_concurrent_model_calls"] == 3
    assert payload["candidate_search_success_definition"][
        "minimum_distinct_fold_recoveries"
    ] == MINIMUM_DISTINCT_FOLD_RECOVERIES
    assert payload["candidate_search_success_definition"][
        "this_is_not_a_promotion_gate"
    ] is True
    provider = payload["provider_policy"]
    assert provider["policy"] == TYPED_ASSIGNMENT_PROVIDER_POLICY
    assert provider["initial_probe_provider_label"] == "plus"
    assert provider["complete_plus_model_response_always_selected"] is True
    assert provider["plus_semantic_acceptance_used_for_selection"] is False
    assert provider[
        "pro_requires_verified_plus_transport_or_no_response_failure"
    ] is True
    assert provider["mid_batch_provider_switch_authorized"] is False
    assert provider["mid_batch_retry_authorized"] is False
    assert provider["valid_failure_retry_authorized"] is False
    assert provider["resampling_authorized"] is False


@pytest.mark.parametrize("accepted", (False, True))
def test_any_complete_plus_model_response_selects_plus_without_semantic_gate(
    tmp_path: Path,
    accepted: bool,
) -> None:
    plus = tmp_path / "plus.json"
    selection = tmp_path / "selection.json"
    _write_canary(plus, accepted=accepted)

    write_provider_selection_receipt_v3(
        plus_canary_report_path=plus,
        output_path=selection,
    )
    receipt = _verify_provider_selection_receipt_v3(
        selection_receipt_path=selection,
        plus_canary_report_path=plus,
        selected_canary_report_path=plus,
        provider_label="plus",
    )

    assert receipt["plus_transport_failure_before_pro_selection"] is False
    assert receipt["semantic_acceptance_used_for_provider_selection"] is False
    assert receipt["selected_model_response_receipt"][
        "canary_semantic_accepted"
    ] is accepted
    assert receipt["mid_batch_retry_authorized"] is False
    if accepted is False:
        pro = tmp_path / "pro.json"
        _write_canary(pro, accepted=True)
        with pytest.raises(TrainTypedAssignmentCrossfitError):
            write_provider_selection_receipt_v3(
                plus_canary_report_path=plus,
                pro_canary_report_path=pro,
                output_path=tmp_path / "forbidden-pro-selection.json",
            )


def test_no_plus_model_response_receipt_then_complete_pro_selects_pro(
    tmp_path: Path,
) -> None:
    plus_report = tmp_path / "plus.json"
    plus_events = tmp_path / "plus.events.jsonl"
    plus_failure = tmp_path / "plus.failure.json"
    pro = tmp_path / "pro.json"
    selection = tmp_path / "selection.json"
    _write_transport_failure_events(plus_events)
    write_plus_transport_failure_receipt_v3(
        event_ledger_path=plus_events,
        expected_canary_report_path=plus_report,
        process_exit_code=1,
        output_path=plus_failure,
    )
    _write_canary(pro, accepted=False)
    write_provider_selection_receipt_v3(
        plus_transport_failure_receipt_path=plus_failure,
        plus_failure_event_ledger_path=plus_events,
        plus_expected_canary_report_path=plus_report,
        pro_canary_report_path=pro,
        output_path=selection,
    )

    receipt = _verify_provider_selection_receipt_v3(
        selection_receipt_path=selection,
        selected_canary_report_path=pro,
        provider_label="pro",
        plus_transport_failure_receipt_path=plus_failure,
        plus_failure_event_ledger_path=plus_events,
        plus_expected_canary_report_path=plus_report,
    )
    assert receipt["plus_transport_failure_before_pro_selection"] is True
    assert receipt["selected_model_response_receipt"][
        "canary_semantic_accepted"
    ] is False


def test_provider_selection_rejects_forged_or_drifted_failure_evidence(
    tmp_path: Path,
) -> None:
    plus_report = tmp_path / "plus.json"
    plus_events = tmp_path / "plus.events.jsonl"
    plus_failure = tmp_path / "plus.failure.json"
    pro = tmp_path / "pro.json"
    selection = tmp_path / "selection.json"
    _write_transport_failure_events(plus_events)
    write_plus_transport_failure_receipt_v3(
        event_ledger_path=plus_events,
        expected_canary_report_path=plus_report,
        process_exit_code=1,
        output_path=plus_failure,
    )
    _write_canary(pro, accepted=True)
    write_provider_selection_receipt_v3(
        plus_transport_failure_receipt_path=plus_failure,
        plus_failure_event_ledger_path=plus_events,
        plus_expected_canary_report_path=plus_report,
        pro_canary_report_path=pro,
        output_path=selection,
    )

    original_events = plus_events.read_text(encoding="utf-8")
    succeeded = Event(
        event="model_attempt_succeeded",
        stage="model.transport",
        trace_id="live-proposal-canary",
        payload={
            "request_hash": SHA_A,
            "response_hash": SHA_B,
            "attempt": 1,
            "elapsed_seconds": 1.0,
            "model": "gpt-5.4-mini",
        },
    ).to_dict()
    plus_events.write_text(
        original_events + json.dumps(succeeded, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainTypedAssignmentCrossfitError):
        _verify_provider_selection_receipt_v3(
            selection_receipt_path=selection,
            selected_canary_report_path=pro,
            provider_label="pro",
            plus_transport_failure_receipt_path=plus_failure,
            plus_failure_event_ledger_path=plus_events,
            plus_expected_canary_report_path=plus_report,
        )
    plus_events.write_text(original_events, encoding="utf-8")

    failure_payload = json.loads(plus_failure.read_text(encoding="utf-8"))
    failure_payload["process_exit_code"] = 2
    plus_failure.write_text(
        json.dumps(failure_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainTypedAssignmentCrossfitError):
        _verify_provider_selection_receipt_v3(
            selection_receipt_path=selection,
            selected_canary_report_path=pro,
            provider_label="pro",
            plus_transport_failure_receipt_path=plus_failure,
            plus_failure_event_ledger_path=plus_events,
            plus_expected_canary_report_path=plus_report,
        )


def test_actual_requires_manifest_and_implementation_at_clean_git_head(
    tmp_path: Path,
) -> None:
    for relative_path in (
        PREREGISTRATION_RELATIVE_PATH,
        *IMPLEMENTATION_RELATIVE_PATHS,
        *EXECUTION_IMPLEMENTATION_RELATIVE_PATHS,
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path + "\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "preregister")

    receipt = verify_preregistration_commit_v3(tmp_path)

    assert receipt[
        "manifest_and_implementation_tracked_at_clean_head"
    ] is True
    assert receipt["raw_commit_ids_persisted"] is False

    dirty_path = tmp_path / IMPLEMENTATION_RELATIVE_PATHS[0]
    dirty_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(TrainTypedAssignmentCrossfitError):
        verify_preregistration_commit_v3(tmp_path)


@pytest.mark.skipif(
    not SOURCE_ROOT.is_dir() or not SOURCE_RANKING_REPORT.is_file(),
    reason="historical v3.20 source ranking is not installed",
)
def test_compiles_all_three_runtime_bound_cells_without_scoring(
    tmp_path: Path,
) -> None:
    result = compile_v320_typed_assignment_crossfit_v3(
        project_root=PROJECT_ROOT,
        output_root=tmp_path / "typed-crossfit",
    )

    result.verify()
    assert tuple(cell.heldout_item_id for cell in result.cells) == (
        REGISTERED_HELDOUT_ITEM_IDS
    )
    assert len(result.candidates) == EXPECTED_ACTIVE_EXECUTION_COUNT
    assert len(result.candidate_bundles_by_hash) == (
        EXPECTED_ACTIVE_EXECUTION_COUNT
    )
    assert result.report["expected_active_execution_count"] == 3
    assert result.report["expected_inactive_raw_replay_count"] == 111
    assert result.report["maximum_concurrent_compile_calls"] == 3
    assert result.report["model_calls"] == 0
    assert result.report["evaluator_calls"] == 0
    assert result.report["online_judge_calls"] == 0
    assert result.report["globally_unbiased_crossfit"] is False
    runtime_class_hash = result.implementation_receipt[
        "runtime_class_hash"
    ]
    implementation_set_hash = result.implementation_receipt[
        "implementation_file_set_hash"
    ]
    assert all(
        runtime_class_hash in candidate.candidate_id
        and implementation_set_hash in candidate.candidate_id
        for candidate in result.candidates
    )

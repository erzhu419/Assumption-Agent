from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import skilllearn_lifecycle
from assumption_agent.benchmarks import train_proposal_diagnostic as diagnostic
from assumption_agent import provider_chain
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    _extract_train_action_trace_profile,
)
from assumption_agent.benchmarks.skilllearnbench import SkillLearnBenchAdapter
from assumption_agent.benchmarks.train_proposal_diagnostic import (
    build_v315_train_source_receipt,
    run_train_proposal_diagnostic,
    verify_existing_train_proposal_diagnostic,
)
from assumption_agent.events import Event, EventSink, JsonlEventSink, MemoryEventSink
from assumption_agent.models import stable_hash
from assumption_agent.proposer import (
    TRAIN_ACTION_DESIGN_POLICY_VERSION,
)
from assumption_agent.splits import (
    AccessPhase,
    BenchmarkItem,
    SplitAccessGuard,
    SplitManifest,
)


RAW_MARKER = "RAW-TRAIN-INSTRUCTION-DO-NOT-PERSIST"
FORBIDDEN_MARKER = "FORBIDDEN-VALIDATION-TEST-OR-VERIFIER-CONTENT"


@dataclass(frozen=True)
class DiagnosticFixture:
    root: Path
    manifest_path: Path
    source_root: Path
    source_train_receipt: Path
    protocol_lock_path: Path
    protocol_path: Path
    report_path: Path
    source_events_path: Path


class FakeProposalModel:
    def __init__(self, *, id_marker: str = "diagnostic-root") -> None:
        self.calls: list[Mapping[str, Any]] = []
        self.id_marker = id_marker

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(payload)
        assert payload["max_hypotheses"] == 1
        assert payload["output_schema"]["required"] == ["hypothesis"]
        assert "proposal_batch_contract" not in payload
        capabilities = payload["capabilities"]
        contract = capabilities["family_slot_contract"]
        portable = contract["portable_recipe_policy"]
        assert contract["validation_outcomes_used"] is False
        assert contract["verifier_content_used"] is False
        assert contract["test_content_used"] is False
        assert len(
            [
                row
                for row in payload["residuals"]
                if row["baseline_success"] is True
                and row["context"] == {}
            ]
        ) == 6
        preferred = portable["preferred_allowlisted_profile_primitives"]
        assert "train_action_design_profiles" not in capabilities
        assert "failed_profile_primitives_to_avoid" not in portable
        assert portable["failed_primitive_values_disclosed"] is False
        artifact_row = portable["recommended_artifact"]
        assert artifact_row in preferred
        assert artifact_row["kind"].startswith("artifact_")
        assert artifact_row["reusable_across_same_family_failures"] is True
        artifact = artifact_row["value"]
        blueprint = portable["required_artifact_workflow_blueprint"]
        assert artifact in blueprint
        family = contract["target_failure_family"]
        hypothesis_schema = payload["output_schema"]["properties"]["hypothesis"]
        assert hypothesis_schema["trigger"] == {
            "all_of": [{"key": "family", "op": "eq", "value": family}],
            "any_of": [],
            "none_of": [],
        }
        assert hypothesis_schema["anti_trigger"] == {
            "all_of": [],
            "any_of": [],
            "none_of": [],
        }
        return {
            "hypothesis": {
                "id": f"{self.id_marker}-{len(self.calls)}",
                "kind": "policy",
                "statement": (
                    "A profile-grounded local artifact operation improves this "
                    "TRAIN failure family."
                ),
                "trigger": {
                    "all_of": [
                        {"key": "family", "op": "eq", "value": family}
                    ],
                    "any_of": [],
                    "none_of": [],
                },
                "anti_trigger": {
                    "all_of": [],
                    "any_of": [],
                    "none_of": [],
                },
                "action_graph": [
                    {
                        "id": "profile-grounded-local-operation",
                        "operation": "execute_step",
                        "target": "task-local artifact procedure",
                        "value": blueprint,
                        "depends_on": [],
                    }
                ],
                "expected_effect": {
                    "metric": "task_success",
                    "minimum_delta": 0.0,
                    "maximum_harm_rate": 0.05,
                    "maximum_cost_ratio": 1.5,
                },
                "verifier": {
                    "checks": ["external offline task success"],
                    "required_evidence": ["task-local artifact"],
                    "anchor_id": "offline-post-agent-verifier",
                    "repair_on_failure": True,
                    "max_repair_depth": 2,
                },
                "evaluator_epoch": payload["evaluator_epoch"],
                "fallback": "preserve_baseline",
                "parent_id": None,
                "lineage": [],
                "created_from_transition_ids": [],
                "status": "candidate",
            }
        }


class TracedFakeProposalModel:
    def __init__(
        self,
        delegate: FakeProposalModel,
        *,
        event_sink: EventSink,
        retry_slot: int | None = None,
    ) -> None:
        self.delegate = delegate
        self.event_sink = event_sink
        self.retry_slot = retry_slot
        self.model = "gpt-5.4-mini"
        self.provider = "openai_compatible"
        self.chain_hash = stable_hash(
            {"providers": [self.provider], "model": self.model}
        )
        self.endpoint_hash = stable_hash(
            {"url": "https://ruoli.dev/v1/chat/completions"}
        )
        self.event_sink.emit(
            Event(
                event="model_provider_chain_built",
                stage="model.provider_chain",
                trace_id="provider-chain-config",
                payload={
                    "requested_providers": [self.provider],
                    "active_providers": [self.provider],
                    "unavailable_providers": [],
                    "model": self.model,
                    "provider_chain_hash": self.chain_hash,
                    "secret_value_persisted": False,
                },
            )
        )

    @property
    def calls(self) -> list[Mapping[str, Any]]:
        return self.delegate.calls

    def complete_with_trace(
        self,
        payload: Mapping[str, Any],
        *,
        trace_id: str,
    ) -> Mapping[str, Any]:
        slot_number = len(self.delegate.calls) + 1
        proposal_request_hash = stable_hash(payload)
        transport_request_hash = stable_hash(
            {"test_transport_request": payload}
        )
        self.event_sink.emit(
            Event(
                event="model_provider_attempted",
                stage="model.provider_chain",
                trace_id=trace_id,
                payload={
                    "provider": self.provider,
                    "provider_position": 0,
                    "provider_count": 1,
                    "provider_chain_hash": self.chain_hash,
                    "request_hash": proposal_request_hash,
                    "model": self.model,
                },
            )
        )
        self._attempt_started(
            trace_id,
            request_hash=transport_request_hash,
            attempt=1,
        )
        success_attempt = 1
        if self.retry_slot == slot_number:
            self.event_sink.emit(
                Event(
                    event="model_attempt_failed",
                    stage="model.transport",
                    trace_id=trace_id,
                    payload={
                        "request_hash": transport_request_hash,
                        "attempt": 1,
                        "elapsed_seconds": 0.01,
                        "error_type": "TimeoutError",
                        "http_status": None,
                        "retryable": True,
                        "model": self.model,
                    },
                )
            )
            success_attempt = 2
            self._attempt_started(
                trace_id,
                request_hash=transport_request_hash,
                attempt=2,
            )
        response = self.delegate.complete(payload)
        response_hash = stable_hash(response)
        self.event_sink.emit(
            Event(
                event="model_attempt_succeeded",
                stage="model.transport",
                trace_id=trace_id,
                payload={
                    "request_hash": transport_request_hash,
                    "response_hash": response_hash,
                    "attempt": success_attempt,
                    "elapsed_seconds": 0.02,
                    "model": self.model,
                },
            )
        )
        self.event_sink.emit(
            Event(
                event="model_provider_selected",
                stage="model.provider_chain",
                trace_id=trace_id,
                payload={
                    "provider": self.provider,
                    "provider_chain_hash": self.chain_hash,
                    "request_hash": proposal_request_hash,
                    "response_hash": response_hash,
                    "model": self.model,
                    "failover_used": False,
                    "prior_failure_count": 0,
                },
            )
        )
        return response

    def _attempt_started(
        self,
        trace_id: str,
        *,
        request_hash: str,
        attempt: int,
    ) -> None:
        self.event_sink.emit(
            Event(
                event="model_attempt_started",
                stage="model.transport",
                trace_id=trace_id,
                payload={
                    "request_hash": request_hash,
                    "attempt": attempt,
                    "attempt_limit": 2,
                    "model": self.model,
                    "timeout_seconds": 300.0,
                    "endpoint_hash": self.endpoint_hash,
                },
            )
        )


class MalformedKindProposalModel(FakeProposalModel):
    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = dict(super().complete(payload))
        hypothesis = dict(response["hypothesis"])
        hypothesis["kind"] = "RAW-MODEL-PARSE-sk-malicious-secret-value"
        response["hypothesis"] = hypothesis
        return response


def test_v317_diagnostic_uses_production_slots_without_forbidden_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    model = FakeProposalModel()
    sink = MemoryEventSink()

    def forbidden_discover(self: SkillLearnBenchAdapter) -> Any:
        raise AssertionError("all-task discovery is forbidden in TRAIN diagnostic")

    monkeypatch.setattr(SkillLearnBenchAdapter, "discover", forbidden_discover)

    def forbidden_runtime_constructor(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("backend/evaluator construction is forbidden")

    for name in (
        "SkillLearnBackendPool",
        "SkillLearnSubprocessBackend",
        "SkillLearnExternalEvaluator",
    ):
        monkeypatch.setattr(
            skilllearn_lifecycle,
            name,
            forbidden_runtime_constructor,
        )

    forbidden_parts = {
        "tests",
        "verifier",
        "solution",
        "validation-only",
        "sealed-only",
    }
    original_contained_file = diagnostic._contained_file
    original_read_json_object = diagnostic._read_json_object
    original_sha256_file = diagnostic._sha256_file
    original_path_read_text = Path.read_text
    original_load_instruction = SkillLearnBenchAdapter.load_instruction
    original_load_context = SkillLearnBenchAdapter.load_action_design_context
    original_extract_profile = (
        skilllearn_lifecycle._extract_train_action_trace_profile
    )

    def reject_forbidden(path: Path) -> None:
        if path.name in {
            "development_recursive.report.json",
            "development_recursive.events.jsonl",
        }:
            raise AssertionError(f"legacy development ledger read: {path}")
        if forbidden_parts & set(path.parts):
            raise AssertionError(f"forbidden content read: {path}")

    def guarded_path_read_text(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        reject_forbidden(path)
        return original_path_read_text(path, *args, **kwargs)

    def guarded_contained_file(
        path: Path,
        *,
        anchor: Path,
        label: str,
    ) -> Path:
        reject_forbidden(path)
        return original_contained_file(path, anchor=anchor, label=label)

    def guarded_read_json_object(
        path: Path,
        *,
        label: str,
    ) -> Mapping[str, Any]:
        reject_forbidden(path)
        return original_read_json_object(path, label=label)

    def guarded_sha256_file(path: Path) -> str:
        reject_forbidden(path)
        return original_sha256_file(path)

    def guarded_load_instruction(
        adapter: SkillLearnBenchAdapter,
        item_id: str,
        *,
        phase: AccessPhase,
        guard: SplitAccessGuard,
    ) -> str:
        assert item_id in guard.manifest.train_ids
        return original_load_instruction(
            adapter,
            item_id,
            phase=phase,
            guard=guard,
        )

    def guarded_load_context(
        adapter: SkillLearnBenchAdapter,
        item_id: str,
        *,
        phase: AccessPhase,
        guard: SplitAccessGuard,
    ) -> dict[str, Any]:
        assert item_id in guard.manifest.train_ids
        return original_load_context(
            adapter,
            item_id,
            phase=phase,
            guard=guard,
        )

    def guarded_extract_profile(
        path: Path,
        *,
        containment_root: Path | None = None,
    ) -> dict[str, Any]:
        reject_forbidden(path)
        return original_extract_profile(
            path,
            containment_root=containment_root,
        )

    monkeypatch.setattr(diagnostic, "_contained_file", guarded_contained_file)
    monkeypatch.setattr(
        diagnostic,
        "_read_json_object",
        guarded_read_json_object,
    )
    monkeypatch.setattr(diagnostic, "_sha256_file", guarded_sha256_file)
    monkeypatch.setattr(Path, "read_text", guarded_path_read_text)
    monkeypatch.setattr(
        SkillLearnBenchAdapter,
        "load_instruction",
        guarded_load_instruction,
    )
    monkeypatch.setattr(
        SkillLearnBenchAdapter,
        "load_action_design_context",
        guarded_load_context,
    )
    monkeypatch.setattr(
        skilllearn_lifecycle,
        "_extract_train_action_trace_profile",
        guarded_extract_profile,
    )

    report = run_train_proposal_diagnostic(
        root=fixture.root,
        manifest_path=fixture.manifest_path,
        source_run_root=fixture.source_root,
        source_train_receipt=fixture.source_train_receipt,
        protocol_path=fixture.protocol_path,
        proposal_model=model,
        event_sink=sink,
    )

    assert report["diagnostic_passed"] is True
    assert report["production_evolution_kernel_used"] is True
    assert report["validation_task_count"] == 0
    assert report["backend_call_count"] == 0
    assert report["evaluator_call_count"] == 0
    assert len(model.calls) == 3
    assert len(report["proposals"]) == 3
    assert len(
        {
            row["target_family_hash"] for row in report["proposals"]
        }
    ) == 3
    assert len(
        {
            row["activation_signature_hash"] for row in report["proposals"]
        }
    ) == 3
    assert all(
        row["matched_failure_family_count"] == 1
        and row["matched_target_support"] >= 2
        and row["target_anti_trigger_self_block_count"] == 0
        and row["profile_environment_binding_count"] > 0
        and {
            "concrete_local_tool_command",
            "artifact_internal_manipulation",
        }
        & set(row["profile_grounded_delta_kinds"])
        and row["restatement_only_action_count"] == 0
        for row in report["proposals"]
    )
    assert report["acceptance"] == {
        key: True for key in report["acceptance"]
    }

    serialized = json.dumps(report, sort_keys=True)
    event_serialized = json.dumps(sink.events, sort_keys=True)
    for forbidden in (
        RAW_MARKER,
        FORBIDDEN_MARKER,
        "/root/family-a.json",
        "/root/family-b.json",
        "/root/family-c.json",
        "broken-family-a",
        "python3",
        "diagnostic-root",
    ):
        assert forbidden not in serialized
        assert forbidden not in event_serialized
    required_false_flags = {
        "backend_accessed",
        "evaluator_accessed",
        "validation_accessed",
        "validation_features_used",
        "validation_outcomes_used",
        "verifier_content_used",
        "test_content_accessed",
        "test_content_used",
        "sealed_test_accessed",
        "secret_value_persisted",
    }
    assert sink.events
    assert all(
        all(row["payload"].get(key) is False for key in required_false_flags)
        for row in sink.events
    )


def test_v317_diagnostic_fails_closed_on_source_profile_provenance_drift(
    tmp_path: Path,
) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    receipt = json.loads(
        fixture.source_train_receipt.read_text(encoding="utf-8")
    )
    receipt["action_profile_set_hash"] = "f" * 64
    _rehash_source_train_receipt(receipt)
    fixture.source_train_receipt.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    model = FakeProposalModel()

    with pytest.raises(
        PermissionError,
        match="source TRAIN receipt drift: action_profile_set_hash",
    ):
        run_train_proposal_diagnostic(
            root=fixture.root,
            manifest_path=fixture.manifest_path,
            source_run_root=fixture.source_root,
            source_train_receipt=fixture.source_train_receipt,
            protocol_path=fixture.protocol_path,
            proposal_model=model,
        )

    assert model.calls == []


def test_v317_reuse_never_opens_legacy_development_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, report_path, events_path = _persisted_diagnostic_fixture(
        tmp_path
    )
    original_read_text = Path.read_text

    def guarded_read_text(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if path.name in {
            "development_recursive.report.json",
            "development_recursive.events.jsonl",
        }:
            raise AssertionError(f"legacy development ledger read: {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    verified = verify_existing_train_proposal_diagnostic(
        root=fixture.root,
        manifest_path=fixture.manifest_path,
        source_run_root=fixture.source_root,
        source_train_receipt=fixture.source_train_receipt,
        protocol_path=fixture.protocol_path,
        report_path=report_path,
        events_path=events_path,
    )
    assert verified["diagnostic_reuse_verified"] is True


def test_v317_diagnostic_rejects_protocol_lock_drift(tmp_path: Path) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    lock = json.loads(fixture.protocol_lock_path.read_text(encoding="utf-8"))
    lock.pop("lock_hash")
    lock["fixture_metadata"] = "changed-after-receipt"
    lock["lock_hash"] = stable_hash(lock)
    fixture.protocol_lock_path.write_text(
        json.dumps(lock, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        PermissionError,
        match="source TRAIN receipt drift: source_identity",
    ):
        run_train_proposal_diagnostic(
            root=fixture.root,
            manifest_path=fixture.manifest_path,
            source_run_root=fixture.source_root,
            source_train_receipt=fixture.source_train_receipt,
            protocol_path=fixture.protocol_path,
            proposal_model=FakeProposalModel(),
        )


def test_v317_diagnostic_persistence_redacts_malicious_id_and_family(
    tmp_path: Path,
) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    events_path = tmp_path / "diagnostic.events.jsonl"
    report_path = tmp_path / "diagnostic.report.json"
    malicious_id = f"{RAW_MARKER}-sk-malicious-secret-value"
    sink = diagnostic._DiagnosticEventSink(JsonlEventSink(events_path))
    model = TracedFakeProposalModel(
        FakeProposalModel(id_marker=malicious_id),
        event_sink=sink,
    )
    report = run_train_proposal_diagnostic(
        root=fixture.root,
        manifest_path=fixture.manifest_path,
        source_run_root=fixture.source_root,
        source_train_receipt=fixture.source_train_receipt,
        protocol_path=fixture.protocol_path,
        proposal_model=model,
        event_sink=sink,
    )
    diagnostic._write_json_atomic(report_path, report)

    raw_audit = json.dumps(sink.events, sort_keys=True)
    persisted_events = events_path.read_text(encoding="utf-8")
    persisted_report = report_path.read_text(encoding="utf-8")
    assert malicious_id in raw_audit
    for forbidden in (
        malicious_id,
        RAW_MARKER,
        "sk-malicious-secret-value",
        "family-a",
        "family-b",
        "family-c",
        "/root/family-a.json",
    ):
        assert forbidden not in persisted_events
        assert forbidden not in persisted_report
    assert '"hypothesis_id"' not in persisted_events
    assert '"target_family"' not in persisted_events
    assert '"target_failure_family"' not in persisted_report
    assert all(
        "hypothesis_id_hash" in row
        and "target_family_hash" in row
        for row in report["proposals"]
    )

    verified = verify_existing_train_proposal_diagnostic(
        root=fixture.root,
        manifest_path=fixture.manifest_path,
        source_run_root=fixture.source_root,
        source_train_receipt=fixture.source_train_receipt,
        protocol_path=fixture.protocol_path,
        report_path=report_path,
        events_path=events_path,
    )
    assert verified["diagnostic_reuse_verified"] is True
    assert verified["secret_value_persisted"] is False
    assert verified["raw_content_persisted"] is False


def test_v317_existing_diagnostic_reuse_rejects_event_tamper(
    tmp_path: Path,
) -> None:
    fixture, report_path, events_path = _persisted_diagnostic_fixture(
        tmp_path
    )
    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed = next(
        row
        for row in rows
        if row["event"] == "train_proposal_diagnostic_completed"
    )
    completed["payload"]["proposal_count"] = 99
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="payload hash mismatch"):
        verify_existing_train_proposal_diagnostic(
            root=fixture.root,
            manifest_path=fixture.manifest_path,
            source_run_root=fixture.source_root,
            source_train_receipt=fixture.source_train_receipt,
            protocol_path=fixture.protocol_path,
            report_path=report_path,
            events_path=events_path,
        )


def test_v317_existing_diagnostic_reuse_rejects_stale_source_artifact(
    tmp_path: Path,
) -> None:
    fixture, report_path, events_path = _persisted_diagnostic_fixture(
        tmp_path
    )
    result_path = next(
        (
            fixture.source_root
            / "development_recursive"
            / "upstream_trials"
            / "no_skill"
        ).rglob("result.json")
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["agent_stdout"] = "changed-after-diagnostic"
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        PermissionError,
        match="source TRAIN receipt drift: source_observation_set_hash",
    ):
        verify_existing_train_proposal_diagnostic(
            root=fixture.root,
            manifest_path=fixture.manifest_path,
            source_run_root=fixture.source_root,
            source_train_receipt=fixture.source_train_receipt,
            protocol_path=fixture.protocol_path,
            report_path=report_path,
            events_path=events_path,
        )


def test_v317_existing_diagnostic_requires_each_live_model_selection(
    tmp_path: Path,
) -> None:
    fixture, report_path, events_path = _persisted_diagnostic_fixture(
        tmp_path
    )
    rows = _event_file_rows(events_path)
    removed = False
    retained: list[dict[str, Any]] = []
    for row in rows:
        if row["event"] == "model_provider_selected" and not removed:
            removed = True
            continue
        retained.append(row)
    assert removed is True
    _write_event_file(events_path, retained)

    with pytest.raises(
        PermissionError,
        match="live ledger count mismatch: model_provider_selected",
    ):
        verify_existing_train_proposal_diagnostic(
            root=fixture.root,
            manifest_path=fixture.manifest_path,
            source_run_root=fixture.source_root,
            source_train_receipt=fixture.source_train_receipt,
            protocol_path=fixture.protocol_path,
            report_path=report_path,
            events_path=events_path,
        )


def test_v317_existing_diagnostic_rejects_semantic_response_hash_tamper(
    tmp_path: Path,
) -> None:
    fixture, report_path, events_path = _persisted_diagnostic_fixture(
        tmp_path
    )
    rows = _event_file_rows(events_path)
    selected = next(
        row for row in rows if row["event"] == "model_provider_selected"
    )
    selected["payload"]["response_hash"] = "f" * 64
    _rehash_event(selected)
    _write_event_file(events_path, rows)

    with pytest.raises(
        PermissionError,
        match="live ledger response hash mismatch",
    ):
        verify_existing_train_proposal_diagnostic(
            root=fixture.root,
            manifest_path=fixture.manifest_path,
            source_run_root=fixture.source_root,
            source_train_receipt=fixture.source_train_receipt,
            protocol_path=fixture.protocol_path,
            report_path=report_path,
            events_path=events_path,
        )


@pytest.mark.parametrize("mutation", ("missing", "changed"))
def test_v317_existing_diagnostic_binds_failed_profile_audit(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture, report_path, events_path = _persisted_diagnostic_fixture(
        tmp_path
    )
    rows = _event_file_rows(events_path)
    completed = next(
        row
        for row in rows
        if row["event"] == "proposal_family_slot_completed"
    )
    if mutation == "missing":
        completed["payload"].pop("failed_profile_binding_count")
    else:
        completed["payload"]["failed_profile_binding_count"] = 1
    _rehash_event(completed)
    _write_event_file(events_path, rows)

    with pytest.raises(PermissionError, match="failed-binding"):
        verify_existing_train_proposal_diagnostic(
            root=fixture.root,
            manifest_path=fixture.manifest_path,
            source_run_root=fixture.source_root,
            source_train_receipt=fixture.source_train_receipt,
            protocol_path=fixture.protocol_path,
            report_path=report_path,
            events_path=events_path,
        )


def test_v317_existing_diagnostic_accepts_one_transport_retry(
    tmp_path: Path,
) -> None:
    fixture, report_path, events_path = _persisted_diagnostic_fixture(
        tmp_path,
        retry_slot=2,
    )
    verified = verify_existing_train_proposal_diagnostic(
        root=fixture.root,
        manifest_path=fixture.manifest_path,
        source_run_root=fixture.source_root,
        source_train_receipt=fixture.source_train_receipt,
        protocol_path=fixture.protocol_path,
        report_path=report_path,
        events_path=events_path,
    )
    rows = _event_file_rows(events_path)
    assert verified["live_provider_model_ledger_verified"] is True
    assert sum(row["event"] == "model_attempt_started" for row in rows) == 4
    assert sum(row["event"] == "model_attempt_failed" for row in rows) == 1


def test_v317_cli_hides_model_controlled_parse_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _diagnostic_fixture(tmp_path)
    env_file = tmp_path / "diagnostic.env"
    env_file.write_text("", encoding="utf-8")
    events_path = tmp_path / "cli.events.jsonl"
    report_path = tmp_path / "cli.report.json"
    monkeypatch.setenv("ASSUMPTION_V2_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("ASSUMPTION_V2_PROVIDER_CHAIN", "openai_compatible")
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev")
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", "test-only-key")

    def fake_build_proposal_model(
        *,
        event_sink: EventSink,
        max_tokens: int,
    ) -> TracedFakeProposalModel:
        assert max_tokens > 0
        return TracedFakeProposalModel(
            MalformedKindProposalModel(),
            event_sink=event_sink,
        )

    monkeypatch.setattr(
        provider_chain,
        "build_proposal_model",
        fake_build_proposal_model,
    )
    with pytest.raises(SystemExit) as caught:
        diagnostic.main(
            [
                "--root",
                str(fixture.root),
                "--manifest",
                str(fixture.manifest_path),
                "--source-run-root",
                str(fixture.source_root),
                "--source-train-receipt",
                str(fixture.source_train_receipt),
                "--protocol",
                str(fixture.protocol_path),
                "--env-file",
                str(env_file),
                "--events",
                str(events_path),
                "--out",
                str(report_path),
            ]
        )

    captured = capsys.readouterr()
    marker = "RAW-MODEL-PARSE-sk-malicious-secret-value"
    assert caught.value.code == 2
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert marker not in captured.err
    assert "test-only-key" not in captured.err
    assert json.loads(captured.err) == {
        "error_type": "proposal_model_or_response_failure",
        "raw_error_persisted": False,
        "secret_value_persisted": False,
        "status": "proposal_diagnostic_failed",
    }
    assert report_path.exists() is False
    assert marker not in events_path.read_text(encoding="utf-8")


def _event_file_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rehash_event(row: dict[str, Any]) -> None:
    row["payload_hash"] = stable_hash(row["payload"])
    row["event_id"] = stable_hash(
        {
            "event": row["event"],
            "stage": row["stage"],
            "trace_id": row["trace_id"],
            "payload": row["payload"],
        }
    )[:24]


def _write_event_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _persisted_diagnostic_fixture(
    tmp_path: Path,
    *,
    retry_slot: int | None = None,
) -> tuple[DiagnosticFixture, Path, Path]:
    fixture = _diagnostic_fixture(tmp_path)
    events_path = tmp_path / "diagnostic.events.jsonl"
    report_path = tmp_path / "diagnostic.report.json"
    sink = diagnostic._DiagnosticEventSink(JsonlEventSink(events_path))
    model = TracedFakeProposalModel(
        FakeProposalModel(),
        event_sink=sink,
        retry_slot=retry_slot,
    )
    report = run_train_proposal_diagnostic(
        root=fixture.root,
        manifest_path=fixture.manifest_path,
        source_run_root=fixture.source_root,
        source_train_receipt=fixture.source_train_receipt,
        protocol_path=fixture.protocol_path,
        proposal_model=model,
        event_sink=sink,
    )
    diagnostic._write_json_atomic(report_path, report)
    verified = verify_existing_train_proposal_diagnostic(
        root=fixture.root,
        manifest_path=fixture.manifest_path,
        source_run_root=fixture.source_root,
        source_train_receipt=fixture.source_train_receipt,
        protocol_path=fixture.protocol_path,
        report_path=report_path,
        events_path=events_path,
    )
    assert verified["diagnostic_reuse_verified"] is True
    return fixture, report_path, events_path


def _diagnostic_fixture(tmp_path: Path) -> DiagnosticFixture:
    root = tmp_path / "SkillLearnBench"
    source_root = tmp_path / "source-v315"
    manifest_path = tmp_path / "manifest.json"
    protocol_path = tmp_path / "protocol-v317.json"
    protocol_lock_path = source_root / "protocol_lock.json"
    source_train_receipt = source_root / "train_source_receipt.json"

    failures = {
        "family-a": 11,
        "family-b": 11,
        "family-c": 10,
    }
    rows: list[tuple[str, str, bool]] = []
    for family, count in failures.items():
        rows.extend(
            (family, f"{family}-failure-{index:02d}", False)
            for index in range(count)
        )
    rows.extend(
        ("success-family", f"success-control-{index:02d}", True)
        for index in range(6)
    )
    train_ids = tuple(item_id for _, item_id, _ in rows)
    family_by_id = {
        item_id: family for family, item_id, _ in rows
    }
    family_by_id.update(
        {
            "validation-only": "validation-family",
            "sealed-only": "sealed-family",
        }
    )
    manifest = SplitManifest(
        benchmark="skilllearnbench",
        protocol="instance_holdout",
        seed="diagnostic-fixture",
        train_ids=train_ids,
        validation_ids=("validation-only",),
        test_ids=("sealed-only",),
        family_by_id=family_by_id,
        sealed_test=True,
    )
    manifest.write(manifest_path)

    upstream = (
        source_root
        / "development_recursive"
        / "upstream_trials"
        / "no_skill"
    )
    failure_profile_index = 0
    for family, item_id, passed in rows:
        instance = root / "tasks" / family / item_id
        environment = instance / "environment"
        environment.mkdir(parents=True, exist_ok=True)
        (instance / "task.toml").write_text(
            "[metadata]\ncategory = \"diagnostic\"\n"
            "difficulty = \"fixture\"\ntags = [\"offline\"]\n",
            encoding="utf-8",
        )
        (instance / "instruction.md").write_text(
            f"{RAW_MARKER}: transform the current {family} artifact for {item_id}.\n",
            encoding="utf-8",
        )
        (environment / "Dockerfile").write_text(
            "FROM scratch\n"
            "RUN apt-get update && apt-get install -y python3\n"
            "WORKDIR /root\n"
            f"COPY {family}.json /root/{family}.json\n",
            encoding="utf-8",
        )
        (environment / f"{family}.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        trial = upstream / family / item_id / "v2_policy_off_fixture"
        agent = trial / "agent"
        agent.mkdir(parents=True, exist_ok=True)
        profile_marker = item_id
        if not passed:
            profile_marker = (
                "shared-profile"
                if failure_profile_index < 2
                else item_id
            )
            failure_profile_index += 1
        trace_text = _trace_text(family, profile_marker)
        trace_path = agent / "codex.txt"
        trace_path.write_text(trace_text, encoding="utf-8")
        trace_hash = hashlib.sha256(trace_text.encode("utf-8")).hexdigest()
        receipt = {
            "trace_sha256": trace_hash,
            "turn_completed_observed": True,
            "turn_completed_count": 1,
            "turn_failed_count": 0,
            "invalid_terminal_usage_count": 0,
            "invalid_action_event_count": 0,
            "token_usage_complete": True,
            "agent_processes_exit_confirmed": True,
            "process_task_scan_complete": True,
            "residual_process_count": 0,
            "observed_steps": 2,
            "token_usage": {"total_tokens": 101},
        }
        receipt["receipt_hash"] = stable_hash(receipt)
        (agent / "codex_action_budget_receipt.json").write_text(
            json.dumps(receipt, sort_keys=True),
            encoding="utf-8",
        )
        result = {
            "task_id": f"{family}/{item_id}",
            "trial_name": "v2_policy_off_fixture",
            "trial_id": "v2_policy_off_fixture",
            "agent": "codex",
            "model": "gpt-5.4-mini",
            "skill_config": "no_skill",
            "skill_source_dir": None,
            "passed": passed,
            "reward": 1 if passed else 0,
            "agent_exit": 0,
            "agent_timed_out": False,
            "verifier_exit": 0,
            "agent_stdout": f"{RAW_MARKER}-AGENT-STDOUT",
            "agent_stderr": "",
            "token_usage": {},
            "token_usage_source": None,
        }
        (trial / "result.json").write_text(
            json.dumps(result, sort_keys=True),
            encoding="utf-8",
        )

    # These files prove that the diagnostic neither discovers nor reads
    # verifier/solution trees or validation/sealed task content.
    train_forbidden = root / "tasks" / "family-a" / rows[0][1]
    for directory in ("tests", "verifier", "solution"):
        path = train_forbidden / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "secret.txt").write_text(
            FORBIDDEN_MARKER,
            encoding="utf-8",
        )
    for family, item_id in (
        ("validation-family", "validation-only"),
        ("sealed-family", "sealed-only"),
    ):
        path = root / "tasks" / family / item_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "instruction.md").write_text(
            FORBIDDEN_MARKER,
            encoding="utf-8",
        )

    profile_hashes = _fixture_profile_hashes(root, upstream, manifest, rows)
    assert len(profile_hashes) == 31
    report = {
        "forbidden_marker": FORBIDDEN_MARKER,
        "validation_only_marker": "must-never-be-opened",
    }
    report_path = source_root / "development_recursive.report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True),
        encoding="utf-8",
    )
    source_events_path = source_root / "development_recursive.events.jsonl"
    source_events_path.write_text(
        json.dumps({"forbidden_marker": FORBIDDEN_MARKER}) + "\n",
        encoding="utf-8",
    )

    lock = {
        "protocol_id": "skilllearn-paper-v3.15-fixture",
        "protocol_hash": "a" * 64,
        "primary_manifest_hash": manifest.manifest_hash,
        "model": "gpt-5.4-mini",
        "fixture_metadata": "locked-but-not-source-evidence",
    }
    lock["lock_hash"] = stable_hash(lock)
    protocol_lock_path.write_text(
        json.dumps(lock, sort_keys=True),
        encoding="utf-8",
    )
    source_train_receipt.write_text(
        json.dumps(
            build_v315_train_source_receipt(
                root=root,
                manifest_path=manifest_path,
                source_run_root=source_root,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    repository_root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            repository_root
            / "manifests"
            / "skilllearn_paper_protocol_v3_17_ruoli_gpt54mini.json"
        ).read_text(encoding="utf-8")
    )
    protocol_path.write_text(
        json.dumps(protocol, sort_keys=True),
        encoding="utf-8",
    )
    return DiagnosticFixture(
        root=root,
        manifest_path=manifest_path,
        source_root=source_root,
        source_train_receipt=source_train_receipt,
        protocol_lock_path=protocol_lock_path,
        protocol_path=protocol_path,
        report_path=report_path,
        source_events_path=source_events_path,
    )


def _trace_text(family: str, profile_marker: str) -> str:
    rows = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "id": f"failed-{family}",
                "command": f"broken-{family} --check",
                "status": "failed",
                "exit_code": 1,
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "id": f"success-{family}",
                "command": f"python3 --version /root/{family}.json",
                "status": "completed",
                "exit_code": 0,
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "file_change",
                "id": f"change-{family}",
                "changes": [
                    {"path": f"/root/{family}-{profile_marker}-output.json"}
                ],
            },
        },
    ]
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _fixture_profile_hashes(
    root: Path,
    upstream: Path,
    manifest: SplitManifest,
    rows: list[tuple[str, str, bool]],
) -> set[str]:
    items = {
        item_id: BenchmarkItem(
            id=item_id,
            family=family,
            features={"benchmark": "skilllearnbench", "family": family},
            content_ref=f"tasks/{family}/{item_id}/instruction.md",
            verifier_ref_hash=stable_hash(
                {"item_id": item_id, "verifier_content_accessed": False}
            ),
        )
        for family, item_id, _ in rows
    }
    adapter = SkillLearnBenchAdapter(root)
    adapter._items = items  # type: ignore[attr-defined]
    adapter._required_env_by_item = {  # type: ignore[attr-defined]
        item_id: () for item_id in items
    }
    guard = SplitAccessGuard(manifest)
    profile_hashes: set[str] = set()
    for family, item_id, passed in rows:
        if passed:
            continue
        trace_path = (
            upstream
            / family
            / item_id
            / "v2_policy_off_fixture"
            / "agent"
            / "codex.txt"
        )
        profile = {
            "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
            "runtime_environment": adapter.load_action_design_context(
                item_id,
                phase=AccessPhase.PROPOSAL,
                guard=guard,
            ),
            "baseline_action_trace": _extract_train_action_trace_profile(
                trace_path,
                containment_root=upstream,
            ),
            "evidence_scope": "train_policy_off_nonoracle_only",
            "validation_outcomes_used": False,
            "verifier_content_used": False,
            "test_content_used": False,
        }
        profile_hashes.add(stable_hash(profile))
    return profile_hashes


def _rehash_source_train_receipt(receipt: dict[str, Any]) -> None:
    receipt.pop("receipt_hash", None)
    receipt.pop("source_checkpoint_hash", None)
    receipt["source_checkpoint_hash"] = stable_hash(receipt)
    receipt["receipt_hash"] = stable_hash(receipt)

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import skilllearn_lifecycle
from assumption_agent.benchmarks.skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import SplitName, stable_hash
from replication_runtime.financial_semantic_v2 import backends
from replication_runtime.financial_semantic_v2.backends import (
    DurableFinancialSemanticSubprocessBackendV2,
    DurableRawSubprocessBackendV2,
    FinancialSemanticReplicationBackendError,
)
from replication_runtime.financial_semantic_v2.terminal_audit import (
    audit_codex_terminal_trace_v2,
)
from replication_runtime.financial_semantic_v2.treatment import (
    FinancialSemanticReplicationTreatmentError,
    FixedFinancialCandidateIdentityV1,
    build_replication_evaluator_binding_v1,
    validate_replication_evaluator_binding_v1,
)


def _hash(label: str) -> str:
    return stable_hash({"test_fixture": label})


def _trace(*events: tuple[str, str]) -> str:
    return "\n".join(
        json.dumps({"type": event_type, "message": message})
        for event_type, message in events
    )


def _candidate(tmp_path: Path) -> FixedFinancialCandidateIdentityV1:
    recipe_id = _hash("recipe")
    return FixedFinancialCandidateIdentityV1(
        parent_manifest_hash=_hash("parent-manifest"),
        candidate_id=_hash("candidate"),
        candidate_manifest_hash=_hash("candidate-manifest"),
        recipe_id=recipe_id,
        program_set_hash=stable_hash({"recipe_ids": [recipe_id]}),
        parent_treatment_id=_hash("parent-treatment"),
        external_skill_source_receipt_hash=_hash("external-source"),
        candidate_skill_source=tmp_path / "candidate-skill",
        operator_asset_path=tmp_path / "operator",
        minilm_runtime_asset_path=tmp_path / "minilm",
        qa_runtime_asset_path=tmp_path / "qa",
    )


def _binding(
    candidate: FixedFinancialCandidateIdentityV1,
    *,
    pack_commitment_hash: str | None,
) -> dict[str, object]:
    return build_replication_evaluator_binding_v1(
        candidate=candidate,
        preregistration_hash=_hash("preregistration"),
        runtime_source_closure_hash=_hash("runtime-closure"),
        pack_commitment_hash=pack_commitment_hash,
    )


def _request(
    *,
    variant: TrialVariant,
    program_id: str | None = None,
    program_set_hash: str = "",
    treatment_hash: str = "",
    external_source_hash: str = "",
) -> SkillLearnTrialRequest:
    return SkillLearnTrialRequest(
        item_id="financial-analysis-fixture",
        family="financial-analysis",
        split=SplitName.TRAIN,
        variant=variant,
        evaluator_epoch="financial-semantic-v2-test",
        pair_id="pair-fixture",
        repeat=1,
        agent_id="codex",
        model="offline-test-model",
        max_steps=100,
        manifest_hash=_hash("trial-manifest"),
        program_id=program_id,
        program_set_hash=program_set_hash,
        treatment_hash=treatment_hash,
        external_skill_source_receipt_hash=external_source_hash,
    )


def test_future_terminal_semantics_recovers_full_trace_but_defers_snippet() -> None:
    transient = ("error", "temporary transport interruption")
    completed_trace = _trace(
        transient,
        ("turn.started", "retrying"),
        ("turn.completed", "done"),
    )
    truncated_trace = _trace(transient)

    completed_audit = audit_codex_terminal_trace_v2(completed_trace)
    truncated_audit = audit_codex_terminal_trace_v2(truncated_trace)

    assert completed_audit.valid
    assert completed_audit.recovered_transient_error
    assert not truncated_audit.valid
    assert "codex_turn_completed_missing" in truncated_audit.issue_types
    assert backends._future_terminal_error_label_v2(completed_trace) is None
    # Sanitization passes deliberately truncated snippets through this hook;
    # their generic failure must be decided later from the durable full trace.
    assert backends._future_terminal_error_label_v2(truncated_trace) is None


def test_future_terminal_semantics_retains_fatal_provider_snippet() -> None:
    incomplete = _trace(("error", "429: too many requests; rate limit"))

    assert not audit_codex_terminal_trace_v2(incomplete).valid
    assert (
        backends._future_terminal_error_label_v2(incomplete)
        == "provider_rate_limit"
    )


def test_nested_future_terminal_patch_remains_active_until_outer_exit() -> None:
    original = skilllearn_lifecycle._codex_terminal_error_label

    with backends.future_terminal_semantics_v2():
        assert (
            skilllearn_lifecycle._codex_terminal_error_label
            is backends._future_terminal_error_label_v2
        )
        with backends.future_terminal_semantics_v2():
            assert (
                skilllearn_lifecycle._codex_terminal_error_label
                is backends._future_terminal_error_label_v2
            )
        assert (
            skilllearn_lifecycle._codex_terminal_error_label
            is backends._future_terminal_error_label_v2
        )

    assert skilllearn_lifecycle._codex_terminal_error_label is original


def test_binding_rejects_extra_fields_even_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    payload = _binding(candidate, pack_commitment_hash=_hash("pack"))
    payload["candidate_prompt_override"] = "forbidden"
    body = dict(payload)
    body.pop("binding_hash")
    payload["binding_hash"] = stable_hash(body)

    with pytest.raises(
        FinancialSemanticReplicationTreatmentError,
        match="fields drifted",
    ):
        validate_replication_evaluator_binding_v1(
            payload,
            candidate=candidate,
            require_pack_commitment=True,
        )


def test_invalid_program_set_is_rejected_at_build_and_validation(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    invalid = replace(candidate, program_set_hash=_hash("wrong-program-set"))
    payload = _binding(candidate, pack_commitment_hash=_hash("pack"))

    with pytest.raises(
        FinancialSemanticReplicationTreatmentError,
        match="identity is malformed",
    ):
        _binding(invalid, pack_commitment_hash=_hash("pack"))
    with pytest.raises(
        FinancialSemanticReplicationTreatmentError,
        match="identity is malformed",
    ):
        validate_replication_evaluator_binding_v1(
            payload,
            candidate=invalid,
            require_pack_commitment=True,
        )


def test_binding_without_pack_fails_closed_when_execution_requires_it(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    payload = _binding(candidate, pack_commitment_hash=None)

    assert validate_replication_evaluator_binding_v1(
        payload,
        candidate=candidate,
        require_pack_commitment=False,
    ) == payload["binding_hash"]
    with pytest.raises(
        FinancialSemanticReplicationTreatmentError,
        match="failed closed",
    ):
        validate_replication_evaluator_binding_v1(
            payload,
            candidate=candidate,
            require_pack_commitment=True,
        )


def test_raw_backend_rejects_nonempty_skill_source_before_execution(
    tmp_path: Path,
) -> None:
    request = _request(
        variant=TrialVariant.POLICY_OFF,
        treatment_hash=NO_SKILL_TREATMENT_HASH,
    )
    backend = object.__new__(DurableRawSubprocessBackendV2)
    backend.durable_request_hash = request.request_hash

    with pytest.raises(
        FinancialSemanticReplicationBackendError,
        match="RAW arm identity or source drifted",
    ):
        backend.run(
            request,
            skill_source_dir=tmp_path,
            trace_id="must-not-execute",
        )


def test_candidate_backend_rejects_program_set_drift_before_execution(
    tmp_path: Path,
) -> None:
    program_id = _hash("candidate-program")
    program_set_hash = _hash("candidate-program-set")
    treatment_hash = _hash("candidate-treatment")
    external_source_hash = _hash("candidate-source")
    request = _request(
        variant=TrialVariant.POLICY_ON,
        program_id=program_id,
        program_set_hash=program_set_hash,
        treatment_hash=treatment_hash,
        external_source_hash=external_source_hash,
    )
    backend = object.__new__(DurableFinancialSemanticSubprocessBackendV2)
    backend.durable_request_hash = request.request_hash
    backend.expected_program_id = program_id
    backend.expected_program_set_hash = _hash("different-program-set")
    backend.expected_treatment_hash = treatment_hash
    backend.expected_external_skill_source_receipt_hash = external_source_hash

    with pytest.raises(
        FinancialSemanticReplicationBackendError,
        match="candidate arm identity or source drifted",
    ):
        backend.run(
            request,
            skill_source_dir=tmp_path,
            trace_id="must-not-execute",
        )

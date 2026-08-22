from __future__ import annotations

from dataclasses import dataclass, fields
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 as time_core
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_v1 as logical_time
from hegel_machine import phase2b_formal_wire_covert_step5_owner_status_replay_core_v1 as core
from hegel_machine import phase2b_formal_wire_covert_step5_owner_status_v1 as owner_status
from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1

# Reuse only the authority-v2 raw-graph fixture.  Owner and logical-time bytes
# below are independently produced as one coupled chain for this core.
from test_phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 import (
    CandidateGraph,
    _build_candidate_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OWNER_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "phase2b_c1_owner_actual_unsealed_960_authorization_v1.json"
)


@dataclass(frozen=True)
class CoupledChain:
    graph: CandidateGraph
    attempt_context_id: str
    owner_payload: bytes
    decisions: tuple[bytes, ...]
    events: tuple[bytes, ...]
    owner_checkpoints: tuple[bytes, ...]
    owner_heads: tuple[bytes, ...]
    owner_projections: tuple[bytes, ...]
    time_requests: tuple[bytes, ...]
    time_tokens: tuple[bytes, ...]
    time_checkpoints: tuple[bytes, ...]
    time_heads: tuple[bytes, ...]
    event_plan: tuple[time_core.ExpectedLogicalEventCoreV1, ...]


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8"))
    assert type(value) is dict
    return value


def _address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _build_coupled_chain(tmp_path: Path, event_count: int) -> CoupledChain:
    assert event_count in (1, 2)
    graph = _build_candidate_graph()
    attempt = "phase2b_step5_attempt_" + "a" * 64
    governance_private = Ed25519PrivateKey.from_private_bytes(bytes([1]) * 32)
    registry = _json(graph.registry_payload)
    registry_body = registry["body"]
    assert type(registry_body) is dict
    role_rows = registry_body["role_keys_in_exact_order"]
    governance_row = role_rows[authority.ROLE_ORDER_V2.index("GOVERNANCE_ANCHOR")]
    cas_row = role_rows[authority.ROLE_ORDER_V2.index("CAS_STORE")]
    assert type(governance_row) is dict and type(cas_row) is dict
    owner_payload = OWNER_CONFIG.read_bytes()

    decisions: list[bytes] = []
    events: list[bytes] = []
    owner_checkpoints: list[bytes] = []
    owner_heads: list[bytes] = []
    owner_projections: list[bytes] = []
    time_requests: list[bytes] = []
    time_tokens: list[bytes] = []
    time_checkpoints: list[bytes] = []
    time_heads: list[bytes] = []
    event_plan: list[time_core.ExpectedLogicalEventCoreV1] = []

    owner_root = tmp_path / "owner"
    time_root = tmp_path / "time"
    owner_root.mkdir()
    owner_root.chmod(0o700)
    with owner_status.OwnerStatusLedgerV1(
        str(owner_root),
        authority_profile_id=graph.profile_id,
        signer_registry_id=graph.registry_id,
        governance_key_id=governance_row["key_id"],
        governance_key_epoch=1,
        attempt_context_id=attempt,
        governance_public_key=bytes.fromhex(governance_row["public_key_hex"]),
        logical_time_public_key=graph.logical_time_public_key,
        owner_authorization_payload=owner_payload,
        initialize_empty=True,
    ) as owner_ledger, logical_time.LogicalTimeLedgerV1(
        root_directory=str(time_root),
        authority_profile_id=graph.profile_id,
        signer_registry_id=graph.registry_id,
        logical_time_key_id=graph.logical_time_key_id,
        logical_time_key_epoch=1,
        attempt_context_id=attempt,
        logical_time_public_key=graph.logical_time_public_key,
        create=True,
    ) as time_ledger:
        owner_heads.append(owner_ledger.current_head_bytes())
        previous_time_head = time_ledger.current_head_bytes()
        time_heads.append(previous_time_head)
        prior_token_id = logical_time.GENESIS_TOKEN_ID_V1
        for index in range(event_count):
            previous_owner_head = owner_ledger.current_head_bytes()
            decision = owner_status.build_owner_status_decision_v1(
                previous_owner_head,
                owner_authorization_payload=owner_payload,
                reason_code=(
                    "INITIAL_OWNER_STATUS"
                    if index == 0
                    else "OWNER_AUTHORIZATION_REVOKED"
                ),
                decision_nonce=("1" if index == 0 else "3") * 64,
            )
            decision_body = _json(decision)
            nonce = ("2" if index == 0 else "4") * 64
            request = logical_time.build_logical_time_request_v1(
                authority_profile_id=graph.profile_id,
                signer_registry_id=graph.registry_id,
                logical_time_key_id=graph.logical_time_key_id,
                logical_time_key_epoch=1,
                attempt_context_id=attempt,
                request_nonce=nonce,
                purpose=core.LOGICAL_TIME_EVENT_PURPOSE_V1,
                bound_artifact_id=decision_body["decision_id"],
                bound_artifact_sha256=_address(decision),
                prior_token_id=prior_token_id,
                next_sequence=index,
            )
            issuance = time_ledger.issue_service_side_v1(
                request_payload=request,
                expected_previous_head_payload=previous_time_head,
                private_key=graph.logical_time_private_key,
            )
            token_body = _json(issuance.signed_token_payload)
            logical_reference = (
                owner_status.build_logical_time_dependency_reference_v1(
                    decision_payload=decision,
                    logical_time_token_payload=issuance.signed_token_payload,
                    logical_time_public_key=graph.logical_time_public_key,
                )
            )
            receipt = _address(f"receipt-{index}".encode("ascii"))
            cas_checkpoint = _address(f"cas-checkpoint-{index}".encode("ascii"))
            cas_head = _address(f"cas-head-{index}".encode("ascii"))
            cas_reference = owner_status.build_cas_store_dependency_reference_v1(
                decision_payload=decision,
                retention_session_id="owner-status-retention-session",
                chain_id="owner-status-cas-chain",
                signer_key_id=cas_row["key_id"],
                signed_receipt_schema_version=owner_status.CAS_SIGNED_RECEIPT_SCHEMA_V1,
                signed_receipt_id=receipt,
                signed_receipt_canonical_sha256=receipt,
                checkpoint_schema_version=owner_status.CAS_CHECKPOINT_SCHEMA_V1,
                checkpoint_id=cas_checkpoint,
                checkpoint_canonical_sha256=cas_checkpoint,
                checkpoint_index=index,
                head_schema_version=owner_status.CAS_HEAD_SCHEMA_V1,
                head_id=cas_head,
                head_canonical_sha256=cas_head,
                head_last_checkpoint_id=cas_checkpoint,
                head_checkpoint_count=index + 1,
            )
            appended = owner_ledger.append_service_side_v1(
                decision_payload=decision,
                cas_store_dependency_reference_payload=cas_reference,
                logical_time_dependency_reference_payload=logical_reference,
                logical_time_token_payload=issuance.signed_token_payload,
                expected_previous_head_payload=previous_owner_head,
                governance_private_key=governance_private,
            )
            decisions.append(decision)
            events.append(appended.signed_event_payload)
            owner_checkpoints.append(appended.checkpoint_payload)
            owner_heads.append(appended.head_payload)
            owner_projections.append(appended.projection_payload)
            time_requests.append(request)
            time_tokens.append(issuance.signed_token_payload)
            time_checkpoints.append(issuance.checkpoint_payload)
            time_heads.append(issuance.head_payload)
            event_plan.append(
                time_core.ExpectedLogicalEventCoreV1(
                    core.LOGICAL_TIME_EVENT_PURPOSE_V1,
                    decision_body["decision_id"],
                    _address(decision),
                    nonce,
                )
            )
            previous_time_head = issuance.head_payload
            prior_token_id = token_body["token_id"]

    return CoupledChain(
        graph=graph,
        attempt_context_id=attempt,
        owner_payload=owner_payload,
        decisions=tuple(decisions),
        events=tuple(events),
        owner_checkpoints=tuple(owner_checkpoints),
        owner_heads=tuple(owner_heads),
        owner_projections=tuple(owner_projections),
        time_requests=tuple(time_requests),
        time_tokens=tuple(time_tokens),
        time_checkpoints=tuple(time_checkpoints),
        time_heads=tuple(time_heads),
        event_plan=tuple(event_plan),
    )


def _replay(chain: CoupledChain, **changes: object) -> core.VerifiedOwnerStatusReplayCoreV1:
    values: dict[str, object] = {
        "stage_b_trust_lock_payload": chain.graph.lock_payload,
        "authority_profile_payload": chain.graph.profile_payload,
        "signer_registry_payload": chain.graph.registry_payload,
        "owner_authorization_payload": chain.owner_payload,
        "decision_payloads_in_exact_order": chain.decisions,
        "signed_event_payloads_in_exact_order": chain.events,
        "owner_checkpoint_payloads_in_exact_order": chain.owner_checkpoints,
        "owner_head_payloads_in_exact_order": chain.owner_heads,
        "caller_supplied_current_head_payload": chain.owner_heads[-1],
        "owner_projection_payload": chain.owner_projections[-1],
        "logical_time_request_payloads_in_exact_order": chain.time_requests,
        "logical_time_signed_token_payloads_in_exact_order": chain.time_tokens,
        "logical_time_checkpoint_payloads_in_exact_order": chain.time_checkpoints,
        "logical_time_head_payloads_in_exact_order": chain.time_heads,
        "out_of_band_expected_lock_id_bytes": chain.graph.lock_id.encode("ascii"),
        "out_of_band_expected_payload_sha256_digest": hashlib.sha256(
            chain.graph.lock_payload
        ).digest(),
        "expected_attempt_context_id": chain.attempt_context_id,
        "expected_logical_time_event_plan": chain.event_plan,
    }
    values.update(changes)
    return core.verify_owner_status_replay_core_v1(**values)  # type: ignore[arg-type]


def _expect_failure(operation: Callable[[], object]) -> core.Phase2BStep5OwnerStatusReplayCoreErrorV1:
    with pytest.raises(core.Phase2BStep5OwnerStatusReplayCoreErrorV1) as caught:
        operation()
    return caught.value


@pytest.mark.parametrize("event_count", [1, 2])
def test_real_active_only_and_active_then_revoked_chains(event_count: int, tmp_path: Path) -> None:
    chain = _build_coupled_chain(tmp_path, event_count)
    result = _replay(chain)
    assert core.require_verified_owner_status_replay_core_v1(result) is result
    assert result.verified_event_count == event_count
    assert result.latest_effective_status == (
        core.STATUS_ACTIVE_V1 if event_count == 1 else core.STATUS_REVOKED_V1
    )
    assert result.revocation_count == event_count - 1
    assert result.head_ids[-1] == _json(chain.owner_heads[-1])["head_id"]
    assert result.logical_time_token_ids == tuple(
        _json(payload)["token_id"] for payload in chain.time_tokens
    )


def test_every_positive_claim_is_exact_and_every_authority_claim_stays_false(tmp_path: Path) -> None:
    result = _replay(_build_coupled_chain(tmp_path, 2))
    true_fields = (
        "raw_authority_v2_graph_verified",
        "governance_two_purpose_policy_verified",
        "governance_static_service_binding_verified",
        "governance_signatures_verified",
        "logical_time_supplied_chain_verified",
        "owner_status_supplied_chain_verified",
        "owner_status_projection_verified",
        "caller_supplied_current_head_byte_equal",
        "legacy_cas_candidate_references_exact",
        "verifier_only",
    )
    false_fields = (
        "legacy_cas_receipts_authenticated",
        "external_cas_checkpoint_verified",
        "external_currentness_verified",
        "external_anti_fork_anchor_verified",
        "runtime_closure_verified",
        "owner_signature_authenticated",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
        "private_key_present",
        "issuer_path_present",
        "filesystem_observed",
        "service_observed",
    )
    assert all(getattr(result, name) is True for name in true_fields)
    assert all(getattr(result, name) is False for name in false_fields)
    assert _expect_failure(core.verify_cas_dependencies_v1).code == core.FAIL_CLAIM_BOUNDARY_V1
    assert _expect_failure(core.verify_external_currentness_v1).code == core.FAIL_CLAIM_BOUNDARY_V1


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("decision_payloads_in_exact_order", lambda c: c.decisions[:-1]),
        ("signed_event_payloads_in_exact_order", lambda c: (*c.events, c.events[-1])),
        ("owner_checkpoint_payloads_in_exact_order", lambda c: tuple(reversed(c.owner_checkpoints))),
        ("owner_head_payloads_in_exact_order", lambda c: (c.owner_heads[0], c.owner_heads[2], c.owner_heads[1])),
        ("logical_time_request_payloads_in_exact_order", lambda c: c.time_requests[:-1]),
        ("logical_time_signed_token_payloads_in_exact_order", lambda c: (*c.time_tokens, c.time_tokens[-1])),
        ("logical_time_checkpoint_payloads_in_exact_order", lambda c: tuple(reversed(c.time_checkpoints))),
        ("logical_time_head_payloads_in_exact_order", lambda c: (c.time_heads[0], c.time_heads[2], c.time_heads[1])),
    ],
)
def test_seven_raw_families_and_heads_reject_missing_extra_or_reordering(
    tmp_path: Path, field: str, mutation: Callable[[CoupledChain], object]
) -> None:
    chain = _build_coupled_chain(tmp_path, 2)
    _expect_failure(lambda: _replay(chain, **{field: mutation(chain)}))


@pytest.mark.parametrize(
    "field",
    [
        "decision_payloads_in_exact_order",
        "signed_event_payloads_in_exact_order",
        "owner_checkpoint_payloads_in_exact_order",
        "owner_head_payloads_in_exact_order",
        "logical_time_request_payloads_in_exact_order",
        "logical_time_signed_token_payloads_in_exact_order",
        "logical_time_checkpoint_payloads_in_exact_order",
        "logical_time_head_payloads_in_exact_order",
    ],
)
def test_each_raw_family_rejects_canonical_field_drift(tmp_path: Path, field: str) -> None:
    chain = _build_coupled_chain(tmp_path, 2)
    payloads = list(
        {
            "decision_payloads_in_exact_order": chain.decisions,
            "signed_event_payloads_in_exact_order": chain.events,
            "owner_checkpoint_payloads_in_exact_order": chain.owner_checkpoints,
            "owner_head_payloads_in_exact_order": chain.owner_heads,
            "logical_time_request_payloads_in_exact_order": chain.time_requests,
            "logical_time_signed_token_payloads_in_exact_order": chain.time_tokens,
            "logical_time_checkpoint_payloads_in_exact_order": chain.time_checkpoints,
            "logical_time_head_payloads_in_exact_order": chain.time_heads,
        }[field]
    )
    index = 1 if "head" in field else 0
    body = _json(payloads[index])
    body["schema_version"] = str(body["schema_version"]) + "-drift"
    payloads[index] = canonicalize_rfc8785_v1(body)
    _expect_failure(lambda: _replay(chain, **{field: tuple(payloads)}))


def test_every_top_level_field_of_all_raw_owner_and_time_artifacts_is_bound(
    tmp_path: Path,
) -> None:
    chain = _build_coupled_chain(tmp_path, 2)
    families: tuple[tuple[str, tuple[bytes, ...], int], ...] = (
        ("decision_payloads_in_exact_order", chain.decisions, 0),
        ("signed_event_payloads_in_exact_order", chain.events, 0),
        ("owner_checkpoint_payloads_in_exact_order", chain.owner_checkpoints, 0),
        ("owner_head_payloads_in_exact_order", chain.owner_heads, 1),
        ("logical_time_request_payloads_in_exact_order", chain.time_requests, 0),
        ("logical_time_signed_token_payloads_in_exact_order", chain.time_tokens, 0),
        ("logical_time_checkpoint_payloads_in_exact_order", chain.time_checkpoints, 0),
        ("logical_time_head_payloads_in_exact_order", chain.time_heads, 1),
    )

    def drift(value: object) -> object:
        if type(value) is str:
            return value + "x"
        if type(value) is int:
            return value + 1
        if type(value) is bool:
            return not value
        if type(value) is dict:
            return {**value, "unexpected_field": "x"}
        if type(value) is list:
            return [*value, "unexpected_item"]
        return "unexpected"

    for argument, original_payloads, index in families:
        original = _json(original_payloads[index])
        for field_name in original:
            changed_body = copy.deepcopy(original)
            changed_body[field_name] = drift(changed_body[field_name])
            changed_payloads = list(original_payloads)
            changed_payloads[index] = canonicalize_rfc8785_v1(changed_body)
            _expect_failure(
                lambda argument=argument, changed_payloads=changed_payloads: _replay(
                    chain, **{argument: tuple(changed_payloads)}
                )
            )

    owner_wrapper = _json(chain.owner_payload)
    for field_name in owner_wrapper:
        changed = copy.deepcopy(owner_wrapper)
        changed[field_name] = drift(changed[field_name])
        _expect_failure(
            lambda changed=changed: _replay(
                chain,
                owner_authorization_payload=canonicalize_rfc8785_v1(changed),
            )
        )

    projection = _json(chain.owner_projections[-1])
    for field_name in projection:
        changed = copy.deepcopy(projection)
        changed[field_name] = drift(changed[field_name])
        _expect_failure(
            lambda changed=changed: _replay(
                chain,
                owner_projection_payload=canonicalize_rfc8785_v1(changed),
            )
        )


@pytest.mark.parametrize("case", ["embedded", "separate", "current"])
def test_embedded_separate_and_current_heads_must_agree(tmp_path: Path, case: str) -> None:
    chain = _build_coupled_chain(tmp_path, 2)
    if case == "embedded":
        checkpoint = _json(chain.owner_checkpoints[-1])
        checkpoint["previous_head_id"] = _json(chain.owner_heads[0])["head_id"]
        changed = (*chain.owner_checkpoints[:-1], canonicalize_rfc8785_v1(checkpoint))
        operation = lambda: _replay(chain, owner_checkpoint_payloads_in_exact_order=changed)
    elif case == "separate":
        operation = lambda: _replay(
            chain,
            owner_head_payloads_in_exact_order=(chain.owner_heads[0], chain.owner_heads[1], chain.owner_heads[1]),
        )
    else:
        operation = lambda: _replay(chain, caller_supplied_current_head_payload=chain.owner_heads[1])
    _expect_failure(operation)


def test_signature_authority_and_time_chain_fail_closed(tmp_path: Path) -> None:
    chain = _build_coupled_chain(tmp_path, 1)
    event = _json(chain.events[0])
    signature = event["signature_hex"]
    assert type(signature) is str
    event["signature_hex"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    _expect_failure(
        lambda: _replay(
            chain,
            signed_event_payloads_in_exact_order=(canonicalize_rfc8785_v1(event),),
        )
    )
    _expect_failure(
        lambda: _replay(
            chain,
            out_of_band_expected_lock_id_bytes=(authority.STAGE_B_TRUST_LOCK_PREFIX_V2 + "f" * 64).encode("ascii"),
        )
    )
    _expect_failure(
        lambda: _replay(
            chain,
            logical_time_head_payloads_in_exact_order=(chain.time_heads[0], chain.time_heads[0]),
        )
    )


def test_fully_readdressed_resigned_alternate_authority_schema_graph_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _build_coupled_chain(tmp_path, 1)
    monkeypatch.setattr(
        authority,
        "AUTHORITY_PROFILE_SCHEMA_V2",
        "hegel-machine-phase2b-formal-wire-covert-step5-authority-profile/EVIL",
    )
    evil = _build_candidate_graph(salt=8)
    failure = _expect_failure(
        lambda: _replay(
            chain,
            stage_b_trust_lock_payload=evil.lock_payload,
            authority_profile_payload=evil.profile_payload,
            signer_registry_payload=evil.registry_payload,
            out_of_band_expected_lock_id_bytes=evil.lock_id.encode("ascii"),
            out_of_band_expected_payload_sha256_digest=hashlib.sha256(
                evil.lock_payload
            ).digest(),
        )
    )
    assert failure.code == core.FAIL_AUTHORITY_V1


def test_result_private_lineage_rejects_new_copy_and_mutation(tmp_path: Path) -> None:
    result = _replay(_build_coupled_chain(tmp_path, 1))
    with pytest.raises(TypeError):
        core.VerifiedOwnerStatusReplayCoreV1()
    forged = object.__new__(core.VerifiedOwnerStatusReplayCoreV1)
    for field in fields(core.VerifiedOwnerStatusReplayCoreV1):
        object.__setattr__(forged, field.name, getattr(result, field.name))
    assert _expect_failure(lambda: core.require_verified_owner_status_replay_core_v1(forged)).code == core.FAIL_RESULT_V1
    try:
        copied = copy.copy(result)
    except TypeError:
        copied = None
    if copied is not None and copied is not result:
        assert _expect_failure(lambda: core.require_verified_owner_status_replay_core_v1(copied)).code == core.FAIL_RESULT_V1
    object.__setattr__(result, "authority_evidence", True)
    assert _expect_failure(lambda: core.require_verified_owner_status_replay_core_v1(result)).code == core.FAIL_RESULT_V1


def test_ordinary_module_attribute_replacement_does_not_redefine_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _build_coupled_chain(tmp_path, 1)
    replacements = {
        "SCHEMA_VERSION": "evil-result/1",
        "AUTHORITY_CLAIM_BOUNDARY_V1": "EVIL_CLAIM",
        "OWNER_STATUS_DECISION_SCHEMA_V1": "evil-decision/1",
        "SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1": "evil-event/1",
        "OWNER_STATUS_CHECKPOINT_SCHEMA_V1": "evil-checkpoint/1",
        "OWNER_STATUS_HEAD_SCHEMA_V1": "evil-head/1",
        "OWNER_STATUS_PROJECTION_SCHEMA_V1": "evil-projection/1",
        "CAS_DEPENDENCY_REFERENCE_SCHEMA_V1": "evil-cas/1",
        "LOGICAL_TIME_DEPENDENCY_REFERENCE_SCHEMA_V1": "evil-time/1",
        "CHECKPOINT_PURPOSE_V1": "EVIL_CHECKPOINT_PURPOSE",
        "EVENT_INITIAL_ACTIVE_V1": "EVIL_INITIAL",
        "EVENT_REVOKED_V1": "EVIL_REVOKED",
        "GENESIS_EVENT_ID_V1": core.EVENT_INITIAL_ACTIVE_V1 + "_evil",
        "GENESIS_CHECKPOINT_ID_V1": core.EVENT_REVOKED_V1 + "_evil",
        "GENESIS_COMPLETE_EVENT_SET_ROOT_V1": "sha256:" + "f" * 64,
        "GOVERNANCE_SIGNATURE_DOMAIN_V1": "EVIL_SIGNATURE_DOMAIN",
        "HEAD_PURPOSE_V1": "EVIL_HEAD_PURPOSE",
        "LOGICAL_TIME_EVENT_PURPOSE_V1": "EVIL_TIME_PURPOSE",
        "PROJECTION_CLAIM_LEVEL_V1": "EVIL_PROJECTION_CLAIM",
        "STATUS_ACTIVE_V1": "EVIL_ACTIVE",
        "STATUS_REVOKED_V1": "EVIL_REVOKED_STATUS",
        "SIGN_OWNER_STATUS_EVENT_PURPOSE_V1": "EVIL_PURPOSE",
        "_Ed25519PublicKey": None,
        "_InvalidSignature": None,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(core, name, value)
    result = _replay(chain)
    assert result.latest_effective_status == "ACTIVE"
    assert core.require_verified_owner_status_replay_core_v1(result) is result


def test_public_entry_ignores_rebound_transitive_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = _build_coupled_chain(tmp_path, 1)
    for name in (
        "_extract_governance_authority",
        "_parse_decision",
        "_parse_event",
        "_parse_checkpoint",
        "_parse_head",
        "_parse_projection",
        "_verify_time_core_v1",
        "_require_time_result_v1",
    ):
        monkeypatch.setattr(
            core,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(_name)
            ),
        )
    assert _replay(chain).verified_event_count == 1


def test_clean_import_has_no_producer_store_runtime_auditor_or_qv2_modules() -> None:
    script = r'''
import json, sys
import hegel_machine.phase2b_formal_wire_covert_step5_owner_status_replay_core_v1
forbidden = [name for name in sys.modules if any(token in name for token in (
    "phase2b_formal_wire_covert_step5_owner_status_v1",
    "phase2b_formal_wire_covert_step5_store_v1",
    "phase2b_formal_wire_covert_step5_store_v2",
    "phase2b_formal_wire_covert_step5_runtime_v1",
    "phase2b_formal_wire_covert_step5_runtime_closure_v1",
    "phase2b_formal_covert_auditor_v1",
    "phase2b_formal_wire_covert_qualification_v2",
))]
print(json.dumps(forbidden))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    # -I deliberately ignores PYTHONPATH, so use an explicit sys.path bootstrap.
    if completed.returncode != 0:
        bootstrap = f"import sys; sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r});" + script
        completed = subprocess.run(
            [sys.executable, "-I", "-c", bootstrap],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    assert json.loads(completed.stdout) == []


def test_public_surface_contains_no_service_or_authority_upgrade_entry() -> None:
    folded = " ".join(core.__all__).casefold()
    assert "service" not in folded
    assert "authority_evidence" not in folded
    assert "r01" not in folded
    assert set(core.__all__) == {
        "AUTHORITY_CLAIM_BOUNDARY_V1",
        "CAS_DEPENDENCY_REFERENCE_SCHEMA_V1",
        "CHECKPOINT_PURPOSE_V1",
        "EVENT_INITIAL_ACTIVE_V1",
        "EVENT_REVOKED_V1",
        "ExpectedLogicalEventCoreV1",
        "FAIL_AUTHORITY_V1",
        "FAIL_BINDING_V1",
        "FAIL_CANONICAL_V1",
        "FAIL_CHAIN_V1",
        "FAIL_CLAIM_BOUNDARY_V1",
        "FAIL_CONTENT_ID_V1",
        "FAIL_INPUT_V1",
        "FAIL_LOGICAL_TIME_V1",
        "FAIL_RESULT_V1",
        "FAIL_SCHEMA_V1",
        "FAIL_SIGNATURE_BACKEND_V1",
        "FAIL_SIGNATURE_V1",
        "GENESIS_CHECKPOINT_ID_V1",
        "GENESIS_COMPLETE_EVENT_SET_ROOT_V1",
        "GENESIS_EVENT_ID_V1",
        "GOVERNANCE_SIGNATURE_DOMAIN_V1",
        "HEAD_PURPOSE_V1",
        "LOGICAL_TIME_EVENT_PURPOSE_V1",
        "OWNER_STATUS_CHECKPOINT_SCHEMA_V1",
        "OWNER_STATUS_DECISION_SCHEMA_V1",
        "OWNER_STATUS_HEAD_SCHEMA_V1",
        "OWNER_STATUS_PROJECTION_SCHEMA_V1",
        "PROJECTION_CLAIM_LEVEL_V1",
        "Phase2BStep5OwnerStatusReplayCoreErrorV1",
        "SCHEMA_VERSION",
        "SIGNED_OWNER_STATUS_EVENT_SCHEMA_V1",
        "STATUS_ACTIVE_V1",
        "STATUS_REVOKED_V1",
        "VerifiedOwnerStatusReplayCoreV1",
        "require_verified_owner_status_replay_core_v1",
        "verify_cas_dependencies_v1",
        "verify_external_currentness_v1",
        "verify_owner_status_replay_core_v1",
    }

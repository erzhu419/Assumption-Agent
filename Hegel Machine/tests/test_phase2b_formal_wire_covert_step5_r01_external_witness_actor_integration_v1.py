from __future__ import annotations

import copy
from dataclasses import dataclass, fields
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable, Mapping

import pytest

from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_durable_anchor_v1 as durable_anchor,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1
    as admission,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_online_witness_v1 as online_witness,
)
from test_phase2b_formal_wire_covert_step5_r01_durable_anchor_v1 import (
    _construct as _construct_a,
)
from test_phase2b_formal_wire_covert_step5_r01_durable_anchor_v1 import (
    _private_spool as _private_a_spool,
)
from test_phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1 import (
    _append_post_q2_chain,
    _build_pre_cut_chain,
    _sha,
    _signed_admission_object,
)
from test_phase2b_formal_wire_covert_step5_r01_online_witness_v1 import (
    _ActorFixture,
    _a_outputs,
    _assert_all_elevated_false,
    _common,
    _expected_prepare_payloads,
    _expected_q2_response,
    _q2_request,
    actor_factory,
)


_B480_SOURCE_SHA256 = (
    "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285"
)
_A_SOURCE_SHA256 = (
    "98478a96c832d48a5ffc574af8a20ed25408845c3d9918e3c391606cc34fdfff"
)
_W_SOURCE_SHA256 = (
    "1560e3ed93e2caf16f40d2f31455ed6c10a2044d7a7883288d619d91d6389fcb"
)


@dataclass(frozen=True)
class _IntegratedActorCorpus:
    fixture: _ActorFixture
    prepared: object
    anchored: object
    q2_request_payload: bytes
    q2_request_root: Mapping[str, object]
    completed: object
    transcript_payload: bytes
    transcript_root: Mapping[str, object]
    logical_requests: tuple[bytes, ...]
    logical_tokens: tuple[bytes, ...]
    logical_checkpoints: tuple[bytes, ...]
    logical_heads: tuple[bytes, ...]
    result_payload: bytes
    result_root: Mapping[str, object]
    verifier_call: Mapping[str, object]
    verified: object


def _expected_a_payloads(
    fixture: _ActorFixture, prepared: object
) -> tuple[bytes, bytes, bytes]:
    corpus = fixture.corpus
    common = _common(corpus)
    durable_policy = corpus.roots["durable_policy"]["body"]
    durable_key_id = durable_policy["anchor_key_id"]
    q1_request = corpus.roots["q1_request"]
    q1_response = json.loads(prepared.q1_response_payload)
    a_request = json.loads(prepared.a_compare_and_append_request_payload)
    prepare_nonce = q1_request["body"]["prepare_nonce"]
    confirmation_nonce = q1_response["body"]["confirmation_nonce"]

    checkpoint_body = {
        **common,
        "schema_version": admission.A_DURABLE_CHECKPOINT_SCHEMA_V1,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(
            prepared.a_compare_and_append_request_payload
        ),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "previous_mapping": copy.deepcopy(corpus.genesis_mapping),
        "committed_mapping": copy.deepcopy(corpus.candidate_mapping),
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "checkpoint_sequence": 1,
        "previous_a_checkpoint_id": durable_policy[
            "oob_high_water_checkpoint_id"
        ],
    }
    checkpoint_payload, checkpoint = _signed_admission_object(
        artifact_kind="a_durable_checkpoint",
        schema=admission.A_DURABLE_CHECKPOINT_SCHEMA_V1,
        id_field="a_checkpoint_id",
        body=checkpoint_body,
        private_key=corpus.durable_private_key,
        signer_key_id=durable_key_id,
        purpose=admission.SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1,
    )

    head_body = {
        **common,
        "schema_version": admission.A_DURABLE_HEAD_SCHEMA_V1,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(
            prepared.a_compare_and_append_request_payload
        ),
        "a_checkpoint_id": checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(checkpoint_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "committed_mapping": copy.deepcopy(corpus.candidate_mapping),
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "head_sequence": 1,
        "previous_a_head_id": durable_policy["oob_high_water_head_id"],
    }
    head_payload, head = _signed_admission_object(
        artifact_kind="a_durable_head",
        schema=admission.A_DURABLE_HEAD_SCHEMA_V1,
        id_field="a_head_id",
        body=head_body,
        private_key=corpus.durable_private_key,
        signer_key_id=durable_key_id,
        purpose=admission.SIGN_A_DURABLE_HEAD_PURPOSE_V1,
    )

    receipt_body = {
        **common,
        "schema_version": admission.A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(
            prepared.a_compare_and_append_request_payload
        ),
        "a_checkpoint_id": checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(checkpoint_payload),
        "a_head_id": head["a_head_id"],
        "a_head_payload_sha256": _sha(head_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "previous_mapping": copy.deepcopy(corpus.genesis_mapping),
        "committed_mapping": copy.deepcopy(corpus.candidate_mapping),
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "compare_matched": True,
        "append_applied": True,
        "oob_high_water_checkpoint_id": durable_policy[
            "oob_high_water_checkpoint_id"
        ],
        "oob_high_water_head_id": durable_policy["oob_high_water_head_id"],
        "durable_before_return": True,
    }
    receipt_payload, _ = _signed_admission_object(
        artifact_kind="a_compare_and_append_receipt",
        schema=admission.A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1,
        id_field="a_receipt_id",
        body=receipt_body,
        private_key=corpus.durable_private_key,
        signer_key_id=durable_key_id,
        purpose=admission.SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1,
    )
    return checkpoint_payload, head_payload, receipt_payload


def _actor_transcript(
    fixture: _ActorFixture,
    prepared: object,
    anchored: object,
    q2_request_payload: bytes,
    completed: object,
    *,
    post_q2_nonce: str,
) -> tuple[bytes, Mapping[str, object]]:
    corpus = fixture.corpus
    q1_request = corpus.roots["q1_request"]
    q1_response = json.loads(prepared.q1_response_payload)
    a_request = json.loads(prepared.a_compare_and_append_request_payload)
    checkpoint = json.loads(anchored.a_durable_checkpoint_payload)
    head = json.loads(anchored.a_durable_head_payload)
    receipt = json.loads(anchored.a_compare_and_append_receipt_payload)
    q2_request = json.loads(q2_request_payload)
    q2_response = json.loads(completed.q2_response_payload)
    body = {
        **_common(corpus),
        "schema_version": admission.ADMISSION_TRANSCRIPT_SCHEMA_V1,
        "prepare_nonce": q1_request["body"]["prepare_nonce"],
        "confirmation_nonce": q1_response["body"]["confirmation_nonce"],
        "candidate_mapping": copy.deepcopy(corpus.candidate_mapping),
        "q1_request_id": q1_request["q1_request_id"],
        "q1_request_payload_sha256": _sha(corpus.payloads["q1_request"]),
        "q1_response_id": q1_response["q1_response_id"],
        "q1_response_payload_sha256": _sha(prepared.q1_response_payload),
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(
            prepared.a_compare_and_append_request_payload
        ),
        "a_checkpoint_id": checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(
            anchored.a_durable_checkpoint_payload
        ),
        "a_head_id": head["a_head_id"],
        "a_head_payload_sha256": _sha(anchored.a_durable_head_payload),
        "a_receipt_id": receipt["a_receipt_id"],
        "a_receipt_payload_sha256": _sha(
            anchored.a_compare_and_append_receipt_payload
        ),
        "q2_request_id": q2_request["q2_request_id"],
        "q2_request_payload_sha256": _sha(q2_request_payload),
        "q2_response_id": q2_response["q2_response_id"],
        "q2_response_payload_sha256": _sha(completed.q2_response_payload),
        "prepare_outcome": admission.PREPARE_OUTCOME_V1,
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "confirm_outcome": admission.CONFIRM_OUTCOME_V1,
        "runtime_custody_binding": copy.deepcopy(
            q2_response["body"]["runtime_custody_binding"]
        ),
        "post_q2_logical_time_request_nonce": post_q2_nonce,
        "presented_q1_a_q2_chain_complete": True,
        "runtime_closure_verified": False,
        "key_custody_verified": False,
        "distinct_live_uid_gid_verified": False,
        "live_channel_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
    }
    bootstrap = corpus.roots["bootstrap_policy"]
    return _signed_admission_object(
        artifact_kind="admission_transcript",
        schema=admission.ADMISSION_TRANSCRIPT_SCHEMA_V1,
        id_field="admission_transcript_id",
        body=body,
        private_key=corpus.graph.witness_private_key,
        signer_key_id=bootstrap["witness_key_id"],
        purpose=admission.SIGN_ADMISSION_TRANSCRIPT_PURPOSE_V1,
    )


def _actor_candidate_result(
    fixture: _ActorFixture,
    transcript_payload: bytes,
    transcript_root: Mapping[str, object],
    requests: tuple[bytes, ...],
    tokens: tuple[bytes, ...],
    checkpoints: tuple[bytes, ...],
    heads: tuple[bytes, ...],
) -> tuple[bytes, Mapping[str, object]]:
    corpus = fixture.corpus
    request = json.loads(requests[-1])
    token = json.loads(tokens[-1])
    checkpoint = json.loads(checkpoints[-1])
    head = json.loads(heads[-1])
    body = {
        **_common(corpus),
        "schema_version": admission.ADMISSION_CANDIDATE_RESULT_SCHEMA_V1,
        "admission_transcript_id": transcript_root["admission_transcript_id"],
        "admission_transcript_payload_sha256": _sha(transcript_payload),
        "logical_time_ledger_id": head["ledger_id"],
        "post_commit_logical_time_request_id": request["request_id"],
        "post_commit_logical_time_token_id": token["token_id"],
        "post_commit_logical_time_checkpoint_id": checkpoint["checkpoint_id"],
        "post_commit_logical_time_head_id": head["head_id"],
        "post_commit_logical_time_head_payload_sha256": (
            "sha256:" + _sha(heads[-1])
        ),
        "verified_signature_count": (
            admission.NON_LOGICAL_VERIFIED_SIGNATURE_COUNT_V1 + len(tokens)
        ),
        "verified_signature_scope": admission.VERIFIED_SIGNATURE_SCOPE_V1,
        "presented_transcript_mechanics_verified": True,
        "post_q2_supplied_logical_time_chain_verified": True,
        "runtime_closure_verified": False,
        "key_custody_verified": False,
        "distinct_live_uid_gid_verified": False,
        "live_channel_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
    }
    bootstrap = corpus.roots["bootstrap_policy"]
    return _signed_admission_object(
        artifact_kind="admission_candidate_result",
        schema=admission.ADMISSION_CANDIDATE_RESULT_SCHEMA_V1,
        id_field="candidate_result_id",
        body=body,
        private_key=corpus.graph.witness_private_key,
        signer_key_id=bootstrap["witness_key_id"],
        purpose=admission.SIGN_ADMISSION_CANDIDATE_RESULT_PURPOSE_V1,
    )


@pytest.fixture
def integrated_actor_corpus(
    actor_factory: Callable[[], _ActorFixture],
) -> _IntegratedActorCorpus:
    fixture = actor_factory()
    corpus = fixture.corpus
    prepared = fixture.actor.begin_prepare(corpus.payloads["q1_request"])
    expected_q1, expected_a_request, _, runtime_binding = (
        _expected_prepare_payloads(fixture, prepared)
    )
    assert prepared.q1_response_payload == expected_q1
    assert prepared.a_compare_and_append_request_payload == expected_a_request

    anchored = _a_outputs(fixture, prepared)
    assert anchored.payloads_in_contract_order == _expected_a_payloads(
        fixture, prepared
    )
    q2_request_payload, q2_request_root = _q2_request(
        fixture, prepared, anchored
    )
    completed = fixture.actor.confirm(
        *anchored.payloads_in_contract_order, q2_request_payload
    )
    assert completed.q2_response_payload == _expected_q2_response(
        fixture,
        q2_request_payload,
        q2_request_root,
        anchored,
        runtime_binding,
    )

    post_q2_nonce = "c" * 64
    transcript_payload, transcript_root = _actor_transcript(
        fixture,
        prepared,
        anchored,
        q2_request_payload,
        completed,
        post_q2_nonce=post_q2_nonce,
    )
    prefix = _build_pre_cut_chain(
        fixture.root / "cross-actor-complete-logical-time-chain", corpus.graph
    )
    anchor_body = corpus.roots["candidate_anchor"]["body"]
    pre_cut_token = json.loads(prefix.tokens[-1])
    pre_cut_checkpoint = json.loads(prefix.checkpoints[-1])
    pre_cut_head = json.loads(prefix.heads[-1])
    assert pre_cut_token["token_id"] == anchor_body["current_cut_token_id"]
    assert "sha256:" + _sha(prefix.tokens[-1]) == anchor_body[
        "current_cut_token_payload_sha256"
    ]
    assert pre_cut_checkpoint["checkpoint_id"] == anchor_body[
        "current_cut_checkpoint_id"
    ]
    assert "sha256:" + _sha(prefix.checkpoints[-1]) == anchor_body[
        "current_cut_checkpoint_payload_sha256"
    ]
    assert pre_cut_head["head_id"] == anchor_body["current_cut_head_id"]
    assert "sha256:" + _sha(prefix.heads[-1]) == anchor_body[
        "current_cut_head_payload_sha256"
    ]
    requests, tokens, checkpoints, heads = _append_post_q2_chain(
        prefix,
        corpus.graph,
        transcript_id=transcript_root["admission_transcript_id"],
        transcript_payload=transcript_payload,
        request_nonce=post_q2_nonce,
    )
    assert json.loads(tokens[-2])["token_id"] == anchor_body[
        "current_cut_token_id"
    ]
    assert json.loads(tokens[-1])["prior_token_id"] == json.loads(tokens[-2])[
        "token_id"
    ]
    assert json.loads(requests[-1])["bound_artifact_id"] == (
        transcript_root["admission_transcript_id"]
    )

    result_payload, result_root = _actor_candidate_result(
        fixture,
        transcript_payload,
        transcript_root,
        requests,
        tokens,
        checkpoints,
        heads,
    )
    call = dict(corpus.call)
    call.update(
        {
            "q1_prepare_response_payload": prepared.q1_response_payload,
            "a_compare_and_append_request_payload": (
                prepared.a_compare_and_append_request_payload
            ),
            "a_durable_checkpoint_payload": anchored.a_durable_checkpoint_payload,
            "a_durable_head_payload": anchored.a_durable_head_payload,
            "a_compare_and_append_receipt_payload": (
                anchored.a_compare_and_append_receipt_payload
            ),
            "q2_confirm_request_payload": q2_request_payload,
            "q2_confirm_response_payload": completed.q2_response_payload,
            "admission_transcript_payload": transcript_payload,
            "post_commit_logical_time_request_payloads_in_exact_order": requests,
            "post_commit_signed_logical_time_token_payloads_in_exact_order": (
                tokens
            ),
            "post_commit_logical_time_checkpoint_payloads_in_exact_order": (
                checkpoints
            ),
            "post_commit_logical_time_head_payloads_in_exact_order": heads,
            "admission_candidate_result_payload": result_payload,
        }
    )
    verified = admission.verify_external_witness_admission_transcript_candidate_v1(
        **call
    )
    assert admission.require_verified_external_witness_admission_candidate_v1(
        verified
    ) is verified
    return _IntegratedActorCorpus(
        fixture=fixture,
        prepared=prepared,
        anchored=anchored,
        q2_request_payload=q2_request_payload,
        q2_request_root=q2_request_root,
        completed=completed,
        transcript_payload=transcript_payload,
        transcript_root=transcript_root,
        logical_requests=requests,
        logical_tokens=tokens,
        logical_checkpoints=checkpoints,
        logical_heads=heads,
        result_payload=result_payload,
        result_root=result_root,
        verifier_call=call,
        verified=verified,
    )


def _resign_a_request(
    fixture: _ActorFixture,
    prepared: object,
    mutate: Callable[[dict[str, object]], None],
) -> bytes:
    root = json.loads(prepared.a_compare_and_append_request_payload)
    body = copy.deepcopy(root["body"])
    mutate(body)
    online_policy = fixture.corpus.roots["online_policy"]["body"]
    payload, _ = _signed_admission_object(
        artifact_kind="a_compare_and_append_request",
        schema=admission.A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        id_field="a_request_id",
        body=body,
        private_key=fixture.corpus.online_private_key,
        signer_key_id=online_policy["online_key_id"],
        purpose=admission.SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1,
    )
    return payload


def _sha256_file(module: object) -> str:
    path = Path(module.__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_sources_and_public_raw_actor_sequence() -> None:
    assert _sha256_file(admission) == _B480_SOURCE_SHA256
    assert _sha256_file(durable_anchor) == _A_SOURCE_SHA256
    assert _sha256_file(online_witness) == _W_SOURCE_SHA256

    witness_parameters = list(
        inspect.signature(online_witness.R01OnlineWitnessActorV1).parameters
    )
    assert witness_parameters == [
        "stage_b_trust_lock_payload",
        "authority_profile_payload",
        "signer_registry_payload",
        "bootstrap_external_witness_policy_payload",
        "online_witness_policy_payload",
        "durable_anchor_policy_payload",
        "candidate_anchor_payload",
        "candidate_receipt_payload",
        "candidate_ledger_payload",
        "candidate_checkpoint_payload",
        "candidate_head_payload",
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
        "out_of_band_expected_bootstrap_external_witness_policy_id_bytes",
        "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest",
        "out_of_band_expected_online_witness_policy_id_bytes",
        "out_of_band_expected_online_witness_policy_payload_sha256_digest",
        "out_of_band_expected_durable_anchor_policy_id_bytes",
        "out_of_band_expected_durable_anchor_policy_payload_sha256_digest",
        "expected_attempt_context_id_bytes",
        "online_witness_private_key",
    ]
    assert list(
        inspect.signature(online_witness.R01OnlineWitnessActorV1.begin_prepare).parameters
    ) == ["self", "q1_prepare_request_payload"]
    assert list(
        inspect.signature(online_witness.R01OnlineWitnessActorV1.confirm).parameters
    ) == [
        "self",
        "a_durable_checkpoint_payload",
        "a_durable_head_payload",
        "a_compare_and_append_receipt_payload",
        "q2_confirm_request_payload",
    ]
    assert list(
        inspect.signature(
            durable_anchor.compare_and_append_local_durable_anchor_v1
        ).parameters
    ) == [
        "actor",
        "q1_prepare_request_payload",
        "q1_prepare_response_payload",
        "a_compare_and_append_request_payload",
    ]
    for signature in (
        inspect.signature(online_witness.R01OnlineWitnessActorV1),
        inspect.signature(durable_anchor.construct_local_durable_anchor_actor_v1),
    ):
        names = set(signature.parameters)
        assert not names & {
            "decoded_mapping",
            "runtime_observation",
            "uid",
            "gid",
            "pid",
            "confirmation_nonce",
            "callback",
            "outcome",
        }


def test_public_w_a_w_full_same_ledger_chain_is_byte_exact_and_b480_accepts(
    integrated_actor_corpus: _IntegratedActorCorpus,
) -> None:
    integrated = integrated_actor_corpus
    assert integrated.anchored.payloads_in_contract_order == _expected_a_payloads(
        integrated.fixture, integrated.prepared
    )
    assert integrated.completed.q2_response_payload == _expected_q2_response(
        integrated.fixture,
        integrated.q2_request_payload,
        integrated.q2_request_root,
        integrated.anchored,
        json.loads(integrated.prepared.q1_response_payload)["body"][
            "runtime_custody_binding"
        ],
    )
    assert len(integrated.logical_requests) == 2
    assert len(integrated.logical_tokens) == 2
    assert len(integrated.logical_checkpoints) == 2
    assert len(integrated.logical_heads) == 3
    assert integrated.verified.presented_q1_a_q2_chain_verified is True
    assert (
        integrated.verified.supplied_post_q2_logical_time_chain_verified is True
    )


def test_cross_actor_results_never_upgrade_real_admission_claims(
    integrated_actor_corpus: _IntegratedActorCorpus,
) -> None:
    integrated = integrated_actor_corpus
    _assert_all_elevated_false(integrated.prepared)
    _assert_all_elevated_false(integrated.completed)
    for name in (
        "key_custody_verified",
        "external_durability_verified",
        "host_rollback_resistance_verified",
        "non_equivocation_verified",
        "live_service_verified",
        "external_currentness_verified",
        "global_anti_fork_verified",
        "logical_time_ancestry_verified",
        "authority_established",
        "r01_satisfied",
        "readiness_satisfied",
        "execution_authorized",
    ):
        assert getattr(integrated.anchored, name) is False
    false_admission_fields = {
        "runtime_closure_verified",
        "key_custody_verified",
        "distinct_live_uid_gid_verified",
        "private_key_present",
        "spool_path_present",
        "decoded_mapping_input_present",
        "callback_present",
        "caller_nonce_present",
        "live_channel_verified",
        "external_currentness_verified",
        "durability_verified",
        "non_equivocation_verified",
        "global_anti_fork_verified",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
    }
    names = {field.name for field in fields(integrated.verified)}
    assert false_admission_fields <= names
    for name in false_admission_fields:
        assert getattr(integrated.verified, name) is False


@pytest.mark.parametrize(
    "field",
    ["stage_b_trust_lock_id", "authority_profile_id", "signer_registry_id"],
)
def test_independently_resigned_divergent_common_authority_triple_is_rejected_before_intent(
    actor_factory: Callable[[], _ActorFixture],
    field: str,
) -> None:
    fixture = actor_factory()
    prepared = fixture.actor.begin_prepare(fixture.corpus.payloads["q1_request"])
    divergent = _resign_a_request(
        fixture,
        prepared,
        lambda body: body.__setitem__(field, str(body[field]) + "_fork"),
    )
    spool = _private_a_spool(fixture.root, f"divergent-{field}")
    actor = _construct_a(fixture.corpus, spool)
    with pytest.raises(
        durable_anchor.Phase2BStep5R01DurableAnchorErrorV1
    ) as rejected:
        durable_anchor.compare_and_append_local_durable_anchor_v1(
            actor,
            fixture.corpus.payloads["q1_request"],
            prepared.q1_response_payload,
            divergent,
        )
    assert rejected.value.code == durable_anchor.FAIL_BINDING_V1
    assert list(spool.iterdir()) == []


def test_candidate_mapping_swap_is_rejected_before_a_intent(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    prepared = fixture.actor.begin_prepare(fixture.corpus.payloads["q1_request"])

    def swap(body: dict[str, object]) -> None:
        body["expected_current_mapping"] = copy.deepcopy(
            fixture.corpus.candidate_mapping
        )
        body["proposed_current_mapping"] = copy.deepcopy(
            fixture.corpus.genesis_mapping
        )

    divergent = _resign_a_request(fixture, prepared, swap)
    spool = _private_a_spool(fixture.root, "mapping-swap")
    actor = _construct_a(fixture.corpus, spool)
    with pytest.raises(
        durable_anchor.Phase2BStep5R01DurableAnchorErrorV1
    ) as rejected:
        durable_anchor.compare_and_append_local_durable_anchor_v1(
            actor,
            fixture.corpus.payloads["q1_request"],
            prepared.q1_response_payload,
            divergent,
        )
    assert rejected.value.code == durable_anchor.FAIL_BINDING_V1
    assert list(spool.iterdir()) == []


def test_valid_but_cross_session_q2_swap_poisons_prepared_w_slot(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    prepared = fixture.actor.begin_prepare(fixture.corpus.payloads["q1_request"])
    anchored = _a_outputs(fixture, prepared)
    correct_q2, _ = _q2_request(fixture, prepared, anchored)
    with pytest.raises(
        online_witness.Phase2BStep5R01OnlineWitnessErrorV1
    ) as cross_swap:
        fixture.actor.confirm(
            *anchored.payloads_in_contract_order,
            fixture.corpus.payloads["q2_request"],
        )
    assert cross_swap.value.code == online_witness.FAIL_BINDING_V1
    with pytest.raises(
        online_witness.Phase2BStep5R01OnlineWitnessErrorV1
    ) as poisoned:
        fixture.actor.confirm(*anchored.payloads_in_contract_order, correct_q2)
    assert poisoned.value.code == online_witness.FAIL_POISONED_V1


def test_same_attempt_detached_genesis_fork_cannot_claim_post_q2_ancestry(
    integrated_actor_corpus: _IntegratedActorCorpus,
) -> None:
    integrated = integrated_actor_corpus
    fixture = integrated.fixture
    detached_prefix = _build_pre_cut_chain(
        fixture.root / "detached-logical-time-fork",
        fixture.corpus.graph,
        request_nonce="e" * 64,
    )
    requests, tokens, checkpoints, heads = _append_post_q2_chain(
        detached_prefix,
        fixture.corpus.graph,
        transcript_id=integrated.transcript_root["admission_transcript_id"],
        transcript_payload=integrated.transcript_payload,
        request_nonce="c" * 64,
    )
    assert tokens[-2] != integrated.logical_tokens[-2]
    call = dict(integrated.verifier_call)
    call.update(
        {
            "post_commit_logical_time_request_payloads_in_exact_order": requests,
            "post_commit_signed_logical_time_token_payloads_in_exact_order": tokens,
            "post_commit_logical_time_checkpoint_payloads_in_exact_order": (
                checkpoints
            ),
            "post_commit_logical_time_head_payloads_in_exact_order": heads,
        }
    )
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    ) as rejected:
        admission.verify_external_witness_admission_transcript_candidate_v1(
            **call
        )
    assert rejected.value.code in {
        admission.FAIL_BINDING_V1,
        admission.FAIL_LOGICAL_TIME_V1,
    }
